#![allow(clippy::unwrap_used)]

mod support;

use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    os::unix::fs::PermissionsExt,
    path::Path,
    pin::Pin,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{Router, body::Body, http::StatusCode};
use futures_util::{SinkExt, Stream, StreamExt};
use nexa_proto::{
    sockets,
    v1::{
        audio_service_server::{AudioService, AudioServiceServer},
        brain_control_service_server::{BrainControlService, BrainControlServiceServer},
        memory_service_server::{MemoryService, MemoryServiceServer},
        *,
    },
};
use nexa_web::{
    EVALUATOR_TOKEN, HOME_ID, HOME_TOKEN, OPERATOR_TOKEN, TestMode, WebConfig, auth::Origin,
    build_app, clients::Clients, sys::PlatformMonitor,
};
use prost::Message as _;
use serde_json::{Value, json};
use tokio::{
    net::{TcpListener, UnixListener},
    task::JoinHandle,
};
use tokio_stream::wrappers::UnixListenerStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, client::IntoClientRequest, protocol::frame::coding::CloseCode},
};
use tonic::{Request, Response, Status, transport::Server};
use uuid::Uuid;

use support::{ORIGIN, call, document};

type TestResult<T> = Result<Response<T>, Status>;
type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[derive(Debug, Clone, Default)]
struct Daemons {
    observed: Arc<Mutex<Vec<ActorContext>>>,
    metadata_actors: Arc<Mutex<Vec<ActorContext>>>,
    purged: Arc<Mutex<Vec<String>>>,
}

impl Daemons {
    #[allow(clippy::result_large_err)]
    fn record_metadata_actor(&self, request: &Request<Empty>, key: &str) -> Result<(), Status> {
        let bytes = request
            .metadata()
            .get_bin(key)
            .ok_or_else(|| Status::unauthenticated("actor metadata required"))?
            .to_bytes()
            .map_err(|_| Status::unauthenticated("invalid actor metadata"))?;
        let actor = ActorContext::decode(bytes)
            .map_err(|_| Status::unauthenticated("invalid actor context"))?;
        self.metadata_actors.lock().unwrap().push(actor);
        Ok(())
    }
}

macro_rules! service {
    (impl $service:ident {
        [$($stream:ident = $element:ty),*]
        $($method:ident($param:ident: $request:ty) -> $response:ty => |$server:ident| $body:block)*
        ; $($unused:ident($unhandled:ty) -> $out:ty),* $(,)?
    }) => {
        #[tonic::async_trait]
        impl $service for Daemons {
            $(type $stream = RpcStream<$element>;)*
            $(async fn $method(&self,$param:Request<$request>) -> TestResult<$response> {
                let $server = self;
                $body
            })*
            $(async fn $unused(&self,_:Request<$unhandled>) -> TestResult<$out> {
                Err(Status::unimplemented(stringify!($unused)))
            })*
        }
    };
}

// The closure argument in a macro is explicit so the same fixture can serve
// three generated Tonic services while recording real wire requests.
service! {
    impl AudioService {
        [SubscribeUtterancesStream = AudioUtterance]
        get_microphones(_request: Empty) -> MicrophonesSnapshot => |_server| {
            Ok(Response::new(MicrophonesSnapshot {
                observed_at_unix_ms:1_700_000_000_000,
                microphones:vec![MicrophoneRoomStatus { room_id:"wire-room".into(),..Default::default() }],
                ..Default::default()
            }))
        }
        set_capture_mute(request: SetCaptureMuteRequest) -> CaptureControlStatus => |server| {
            let input = request.into_inner();
            server.observed.lock().unwrap().push(input.actor.unwrap());
            Ok(Response::new(CaptureControlStatus {
                revision:input.expected_revision+1, phase:CapturePhase::Muted as i32,
                persisted_muted:input.muted,ingress_sealed:true,supported:true,..Default::default()
            }))
        }
        ; get_health(Empty) -> HealthResponse,
          get_zones(Empty) -> ZonesSnapshot,
          set_zone_volume(SetZoneVolumeRequest) -> ZoneState,
          get_capture_control(Empty) -> CaptureControlStatus,
          forget_carried_owners(ForgetCarriedOwnersRequest) -> ForgetCarriedOwnersResponse,
          subscribe_utterances(SubscribeUtterancesRequest) -> Self::SubscribeUtterancesStream,
          play_audio_stream(tonic::Streaming<PlayAudioChunk>) -> PlayAudioReceipt,
          cancel_playback(CancelPlaybackRequest) -> Empty,
    }
}

service! {
    impl BrainControlService {
        [SubscribeLiveEventsStream = LiveRuntimeEvent]
        get_home_catalog(request: GetHomeCatalogRequest) -> HomeCatalogSnapshot => |server| {
            server.observed.lock().unwrap().push(request.into_inner().actor.unwrap());
            Ok(Response::new(HomeCatalogSnapshot {
                observed_at_unix_ms:1_700_000_000_000,
                rooms:vec![HomeRoom { room_id:"wire-room".into(),name:"Wire Room".into(),..Default::default() }],
                ..Default::default()
            }))
        }
        list_routines(request: ListRoutinesRequest) -> RoutinesSnapshot => |server| {
            server.observed.lock().unwrap().push(request.into_inner().actor.unwrap());
            Ok(Response::new(RoutinesSnapshot { routines:vec![RoutineRecord {
                routine_id:"wire-routine".into(),name:"Wire Routine".into(),spec_json:"{}".into(),revision:7,
                schedule_json:r#"{"at":"09:00"}"#.into(), updated_at_unix_ms:1_700_000_000_000,
                ..Default::default()
            }]}))
        }
        get_llm_endpoints(request: Empty) -> LlmEndpointsSnapshot => |server| {
            server.record_metadata_actor(&request, "x-nexa-actor-bin")?;
            Ok(Response::new(LlmEndpointsSnapshot {
                endpoints:vec![LlmEndpointStatus { endpoint_id:"wire-llm".into(),..Default::default() }],
            }))
        }
        get_cloud_consent(request: Empty) -> CloudConsentStatus => |server| {
            server.record_metadata_actor(&request, "x-nexa-actor-bin")?;
            Ok(Response::new(CloudConsentStatus { consent_granted:true,..Default::default() }))
        }
        sync_household_lists(request: ActorRequest) -> ListSyncResponse => |_server| {
            let actor = request.into_inner().actor.unwrap();
            assert_eq!(actor.user_id, HOME_ID);
            assert!(actor.permissions.contains(&(Permission::TasksWrite as i32)));
            Ok(Response::new(ListSyncResponse {payload_json:r#"{"synced":true}"#.into()}))
        }
        list_reminder_notices(request: ActorRequest) -> ReminderNoticesResponse => |_server| {
            let actor = request.into_inner().actor.unwrap();
            assert_eq!(actor.user_id, HOME_ID);
            assert!(actor.permissions.contains(&(Permission::TasksRead as i32)));
            Ok(Response::new(ReminderNoticesResponse {
                payload_json:r#"{"source":"reminder-store","notices":[]}"#.into(),
            }))
        }
        get_reminder_status(request: ActorRequest) -> ReminderStatusResponse => |_server| {
            let actor = request.into_inner().actor.unwrap();
            assert_eq!(actor.role, Role::Operator as i32);
            assert!(actor.permissions.contains(&(Permission::DiagnosticsRead as i32)));
            Ok(Response::new(ReminderStatusResponse {payload_json:r#"{"source":"reminder-store"}"#.into()}))
        }
        ; get_health(Empty) -> HealthResponse,
          execute_home_action(HomeActionRequest) -> HomeActionReceipt,
          get_home_action_receipt(GetHomeActionReceiptRequest) -> HomeActionReceipt,
          clear_llm_cooldown(ClearLlmCooldownRequest) -> Empty,
          withdraw_cloud_consent(WithdrawCloudConsentRequest) -> CloudConsentStatus,
          subscribe_live_events(SubscribeLiveEventsRequest) -> Self::SubscribeLiveEventsStream,
          get_routine(GetRoutineRequest) -> RoutineRecord,
          preview_routine(PreviewRoutineRequest) -> RoutinePreviewResponse,
          upsert_routine(UpsertRoutineRequest) -> RoutineRecord,
          delete_routine(DeleteRoutineRequest) -> Empty,
          execute_routine(ExecuteRoutineRequest) -> RoutineExecutionReceipt,
          list_household_lists(ActorRequest) -> HouseholdListsSnapshot,
          mutate_household_list(HouseholdListMutationRequest) -> HouseholdListRecord,
          list_reminders(ListRemindersRequest) -> RemindersSnapshot,
          mutate_reminder(ReminderMutationRequest) -> ReminderRecord,
          list_alerts(ActorRequest) -> AlertsSnapshot,
          ack_alert(AlertMutationRequest) -> AlertRecord,
          delete_alert(AlertMutationRequest) -> Empty,
          get_alert_zones(ActorRequest) -> AlertZonesConfig,
          set_alert_zones(SetAlertZonesRequest) -> AlertZonesConfig,
          list_conversations(ListConversationsRequest) -> ConversationsSnapshot,
          list_ha_actions(ListHaActionsRequest) -> HaActionsSnapshot,
          list_memory_overview(MemoryOverviewRequest) -> MemoryOverviewResponse,
          mutate_memory_item(MemoryMutationRequest) -> MemoryMutationResponse,
          get_voice_global_switch(ActorRequest) -> VoiceGlobalSwitchStatus,
          set_voice_global_switch(SetVoiceGlobalSwitchRequest) -> VoiceGlobalSwitchStatus,
          get_voice_capture_review(VoiceCaptureReviewRequest) -> VoiceCaptureReviewResponse,
          query_voice_captures(VoiceCaptureQueryRequest) -> VoiceCaptureQueryResponse,
          get_evaluation_sample(EvaluationSampleRequest) -> EvaluationSampleResponse,
    }
}

service! {
    impl MemoryService {
        []
        get_ambient_control(request: Empty) -> AmbientControlStatus => |server| {
            server.record_metadata_actor(&request, "actor-context-bin")?;
            Ok(Response::new(AmbientControlStatus { paused:true,revision:7,updated_at_unix_ms:1_700_000_000_000 }))
        }
        purge_retired_user(request: PurgeRetiredUserRequest) -> ErasureReceipt => |server| {
            let request = request.into_inner();
            server.observed.lock().unwrap().push(request.actor.unwrap());
            let mut purged = server.purged.lock().unwrap();
            purged.push(request.user_id);
            // The first attempt exercises the pending state and a later HTTP retry.
            if purged.len() == 1 {
                return Err(Status::unavailable("injected transport failure"));
            }
            Ok(Response::new(ErasureReceipt { tombstone_recorded:true,..Default::default() }))
        }
        mutate_memory_item(request: MemoryMutationRequest) -> MemoryMutationResponse => |_server| {
            let request = request.into_inner();
            let actor = request.actor.unwrap();
            assert_eq!(actor.role, Role::Operator as i32);
            assert!(actor.permissions.contains(&(Permission::VoicesReview as i32)));
            let body: Value = serde_json::from_str(&request.payload_json).unwrap();
            let payload = match request.operation.as_str() {
                "confirm_voice_profile" => {
                    assert_eq!(body["fingerprint_ids"], json!(["cap-a","cap-b","cap-c"]));
                    assert!(matches!(request.target_id.as_str(), "" | "profile-target"));
                    let revision = if request.target_id.is_empty() {
                        assert!(body.get("expected_revision").is_none());
                        1
                    } else {
                        assert_eq!(body["expected_revision"], 1);
                        2
                    };
                    json!({"profile_id":"profile-target","revision":revision})
                }
                "correct_voice_group" => {
                    assert_eq!(request.target_id, "group-target");
                    assert_eq!(body["action"], "merge");
                    assert_eq!(body["profile_ids"], json!(["profile-target"]));
                    assert_eq!(body["expected_revisions"], json!({"group-target":2}));
                    json!({"group_id":"group-target","revision":3})
                }
                _ => return Err(Status::unimplemented("memory operation")),
            };
            Ok(Response::new(MemoryMutationResponse {payload_json:payload.to_string()}))
        }
        ; get_health(Empty) -> HealthResponse,
          store_fact(StoreFactRequest) -> MemoryFact,
          query_memory(QueryMemoryRequest) -> QueryMemoryResponse,
          delete_fact(DeleteFactRequest) -> ErasureReceipt,
          record_voice_observation(RecordVoiceObservationRequest) -> VoiceObservationReceipt,
          list_voice_profiles(ListVoiceProfilesRequest) -> ListVoiceProfilesResponse,
          link_voice_profile_to_person(LinkVoiceProfileRequest) -> VoiceProfile,
          set_voice_consent(SetVoiceConsentRequest) -> VoiceConsentStatus,
          erase_conversation(EraseConversationRequest) -> ErasureReceipt,
          set_ambient_pause(SetAmbientPauseRequest) -> AmbientControlStatus,
          list_memory_overview(MemoryOverviewRequest) -> MemoryOverviewResponse,
          get_voice_global_switch(ActorRequest) -> VoiceGlobalSwitchStatus,
          set_voice_global_switch(SetVoiceGlobalSwitchRequest) -> VoiceGlobalSwitchStatus,
          get_voice_capture_review(VoiceCaptureReviewRequest) -> VoiceCaptureReviewResponse,
          query_voice_captures(VoiceCaptureQueryRequest) -> VoiceCaptureQueryResponse,
          get_evaluation_sample(EvaluationSampleRequest) -> EvaluationSampleResponse,
    }
}

fn uds(path: &Path) -> UnixListenerStream {
    let listener = UnixListener::bind(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
    UnixListenerStream::new(listener)
}

#[tokio::test]
async fn tonic_uds_contracts_and_failed_purge_retry() {
    // Short socket paths fit Darwin's sockaddr_un.
    let directory = tempfile::tempdir_in("/tmp").unwrap();
    let audio = uds(&directory.path().join(sockets::AUDIO));
    let brain = uds(&directory.path().join(sockets::BRAIN));
    let memory = uds(&directory.path().join(sockets::MEMORY));
    let daemons = Daemons::default();
    let audio_server = tokio::spawn(
        Server::builder()
            .add_service(AudioServiceServer::new(daemons.clone()))
            .serve_with_incoming(audio),
    );
    let brain_server = tokio::spawn(
        Server::builder()
            .add_service(BrainControlServiceServer::new(daemons.clone()))
            .serve_with_incoming(brain),
    );
    let memory_server = tokio::spawn(
        Server::builder()
            .add_service(MemoryServiceServer::new(daemons.clone()))
            .serve_with_incoming(memory),
    );
    let fixture = TestMode::new().await.unwrap();
    let app = build_app(
        WebConfig {
            origin: Origin::parse(ORIGIN).unwrap(),
            allow_insecure_cookies: false,
            private_dir: None,
            static_dir: None,
        },
        fixture.auth.clone(),
        Clients::uds(directory.path()),
        Arc::new(PlatformMonitor),
    )
    .unwrap();

    let catalog = call(
        &app,
        "GET",
        "/api/home/catalog",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(catalog.status(), StatusCode::OK);
    assert_eq!(document(catalog).await["rooms"][0]["room_id"], "wire-room");
    let denied = call(
        &app,
        "GET",
        "/api/home/microphones",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    let mics = call(
        &app,
        "GET",
        "/api/home/microphones",
        Some(OPERATOR_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(
        document(mics).await["microphones"][0]["room_id"],
        "wire-room"
    );
    let ambient = call(
        &app,
        "GET",
        "/api/ambient/control",
        Some(OPERATOR_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(ambient.status(), StatusCode::OK);
    assert_eq!(document(ambient).await["revision"], 7);
    let endpoints = call(
        &app,
        "GET",
        "/api/llm/endpoints",
        Some(OPERATOR_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(endpoints.status(), StatusCode::OK);
    assert_eq!(document(endpoints).await[0]["id"], "wire-llm");
    let cloud = call(
        &app,
        "GET",
        "/api/admin/cloud-consent",
        Some(OPERATOR_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(cloud.status(), StatusCode::OK);
    assert_eq!(document(cloud).await["consent_granted"], true);
    let routines = call(
        &app,
        "GET",
        "/api/routines",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    let routines = document(routines).await;
    assert_eq!(routines[0]["id"], "wire-routine");
    assert_eq!(routines[0]["routine_id"], "wire-routine");
    assert_eq!(routines[0]["schedule"]["at"], "09:00");
    assert_eq!(routines[0]["updated_at_unix_ms"], 1_700_000_000_000_u64);

    for (method, path, token, expected) in [
        (
            "POST",
            "/api/lists/sync",
            HOME_TOKEN,
            json!({"synced":true}),
        ),
        (
            "GET",
            "/api/reminders/notices",
            HOME_TOKEN,
            json!({"source":"reminder-store","notices":[]}),
        ),
        (
            "GET",
            "/api/reminders/status",
            OPERATOR_TOKEN,
            json!({"source":"reminder-store"}),
        ),
    ] {
        let response = call(&app, method, path, Some(token), Some(ORIGIN), Body::empty()).await;
        assert_eq!(response.status(), StatusCode::OK, "{method} {path}");
        assert_eq!(document(response).await, expected);
    }
    for (path, payload, status, revision) in [
        (
            "/api/voices/profiles",
            json!({"fingerprint_ids":["cap-a","cap-b","cap-c"],"name":"Wire Name","request_id":"request"}),
            StatusCode::OK,
            1,
        ),
        (
            "/api/voices/profiles",
            json!({"profile_id":"profile-target","fingerprint_ids":["cap-a","cap-b","cap-c"],"expected_revision":1}),
            StatusCode::OK,
            2,
        ),
        (
            "/api/voices/profiles",
            json!({"fingerprint_ids":["cap-a","cap-a"]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            0,
        ),
        (
            "/api/voices/groups/corrections",
            json!({"group_id":"group-target","action":"merge","profile_ids":["profile-target"],"expected_revisions":{"group-target":2}}),
            StatusCode::OK,
            3,
        ),
    ] {
        let response = call(
            &app,
            "POST",
            path,
            Some(OPERATOR_TOKEN),
            Some(ORIGIN),
            Body::from(payload.to_string()),
        )
        .await;
        assert_eq!(response.status(), status, "{path}");
        if status.is_success() {
            assert_eq!(document(response).await["revision"], revision);
        }
    }

    let forged = call(
        &app,
        "POST",
        "/api/capture/control",
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::from(r#"{"muted":true,"expected_revision":0,"actor":{"role":1,"user_id":"forged"}}"#),
    )
    .await;
    assert_eq!(forged.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let capture = call(
        &app,
        "POST",
        "/api/capture/control",
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::from(r#"{"muted":true,"expected_revision":7}"#),
    )
    .await;
    assert_eq!(capture.status(), StatusCode::ACCEPTED);
    let capture = document(capture).await;
    assert_eq!(capture["revision"], 8);
    assert_eq!(capture["total_mute_confirmed"], true);
    let route = format!("/api/admin/users/{HOME_ID}");
    let retired = call(
        &app,
        "DELETE",
        &route,
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::empty(),
    )
    .await;
    assert_eq!(retired.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        document(retired).await["detail"]["credentials_retired"],
        true
    );
    assert_eq!(
        call(
            &app,
            "GET",
            "/api/home/catalog",
            Some(HOME_TOKEN),
            None,
            Body::empty()
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    for _ in 0..2 {
        let retry = call(
            &app,
            "DELETE",
            &route,
            Some(OPERATOR_TOKEN),
            Some(ORIGIN),
            Body::empty(),
        )
        .await;
        assert_eq!(retry.status(), StatusCode::OK);
    }
    assert_eq!(*daemons.purged.lock().unwrap(), vec![HOME_ID, HOME_ID]);
    let actors = daemons.observed.lock().unwrap();
    assert_eq!(actors.len(), 5);
    // Capture mute carries the same authenticated caller in the request body.
    assert_eq!(
        *daemons.metadata_actors.lock().unwrap(),
        vec![actors[2].clone(); 3]
    );
    for (index, actor) in actors.iter().enumerate() {
        assert_eq!(actor.lease_id.len(), 64);
        assert_eq!(actor.lease_epoch, 1);
        if index < 2 {
            assert_eq!(actor.role, Role::Home as i32);
            assert_eq!(actor.user_id, HOME_ID);
            assert!(
                !actor
                    .permissions
                    .contains(&(Permission::AdminManage as i32))
            );
        } else {
            assert_eq!(actor.role, Role::Operator as i32);
            assert!(actor.user_id.is_empty());
            assert!(
                actor
                    .permissions
                    .contains(&(Permission::AdminManage as i32))
            );
        }
    }
    for server in [audio_server, brain_server, memory_server] {
        server.abort();
    }
}

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

async fn serve(app: Router) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    (
        addr,
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        }),
    )
}

async fn connect(
    addr: SocketAddr,
    path: &str,
    token: &str,
) -> Result<Socket, tokio_tungstenite::tungstenite::Error> {
    let mut req = format!("ws://{addr}{path}").into_client_request().unwrap();
    req.headers_mut().insert("origin", ORIGIN.parse().unwrap());
    req.headers_mut()
        .insert("authorization", format!("Bearer {token}").parse().unwrap());
    connect_async(req).await.map(|(socket, _)| socket)
}

async fn next(socket: &mut Socket) -> Message {
    tokio::time::timeout(Duration::from_secs(3), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
}

async fn send(socket: &mut Socket, frame: Value) {
    socket
        .send(Message::Text(frame.to_string().into()))
        .await
        .unwrap();
}

async fn next_json(socket: &mut Socket) -> Value {
    serde_json::from_str(next(socket).await.into_text().unwrap().as_str()).unwrap()
}

async fn v3(socket: &mut Socket) -> String {
    send(
        socket,
        json!({"type":"live_hello","version":3,"client_id":Uuid::new_v4().to_string()}),
    )
    .await;
    let reset = next_json(socket).await;
    assert_eq!(reset["type"], "live_reset");
    send(socket,json!({"type":"live_reset_ack","id":reset["payload"]["id"],"nonce":reset["payload"]["nonce"]})).await;
    let ready = next_json(socket).await;
    assert_eq!(ready["type"], "rest_ready");
    ready["connection_id"].as_str().unwrap().to_owned()
}

async fn close_policy(socket: &mut Socket, reason: &str) {
    match next(socket).await {
        Message::Close(Some(frame)) => {
            assert_eq!(frame.code, CloseCode::Policy);
            assert_eq!(frame.reason.as_str(), reason);
        }
        frame => panic!("expected policy close, received {frame:?}"),
    }
}

#[tokio::test]
async fn websocket_v3_receipts_erasure_and_logout() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let (addr, server) = serve(app.clone()).await;
    assert!(
        matches!(connect(addr,"/ws",OPERATOR_TOKEN).await,Err(tokio_tungstenite::tungstenite::Error::Http(response)) if response.status() == 404)
    );
    assert!(
        matches!(connect(addr,"/ws/live",HOME_TOKEN).await,Err(tokio_tungstenite::tungstenite::Error::Http(response)) if response.status() == 403)
    );
    assert!(
        matches!(connect(addr,"/ws/evaluation",OPERATOR_TOKEN).await,Err(tokio_tungstenite::tungstenite::Error::Http(response)) if response.status() == 403)
    );
    let session = call(
        &app,
        "POST",
        "/api/admin/auth/login",
        None,
        Some(ORIGIN),
        Body::from(json!({"token":OPERATOR_TOKEN}).to_string()),
    )
    .await;
    let token = session.headers()["set-cookie"]
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .split_once('=')
        .unwrap()
        .1;
    let mut live = connect(addr, "/ws/live", token).await.unwrap();
    let mut voices = connect(addr, "/ws/voices", OPERATOR_TOKEN).await.unwrap();
    let mut evaluation = connect(addr, "/ws/evaluation", EVALUATOR_TOKEN)
        .await
        .unwrap();
    let connection = v3(&mut live).await;
    v3(&mut voices).await;
    v3(&mut evaluation).await;
    let rest = app
        .clone()
        .oneshot(
            axum::http::Request::get("/api/conversations")
                .header("authorization", format!("Bearer {token}"))
                .header("x-kza-live-connection", &connection)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rest.status(), StatusCode::OK);
    assert_eq!(document(rest).await["connection_id"], connection);
    fixture.emit(LiveRuntimeEvent {
        kind: LiveEventKind::Turn as i32,
        payload_json: r#"{"text":"fixture"}"#.into(),
        ..Default::default()
    });
    let event = next_json(&mut live).await;
    assert_eq!(event["type"], "turn");
    send(
        &mut live,
        json!({"type":"live_ack","receipts":[event["receipt"].clone()]}),
    )
    .await;

    let revoked = call(
        &app,
        "POST",
        "/api/admin/auth/logout",
        Some(token),
        Some(ORIGIN),
        Body::empty(),
    )
    .await;
    assert_eq!(revoked.status(), StatusCode::OK);
    assert_eq!(next_json(&mut live).await["type"], "retire");
    assert_eq!(next_json(&mut live).await["type"], "live_reset");
    close_policy(&mut live, "lease_retired").await;

    fixture.emit(LiveRuntimeEvent {
        kind: LiveEventKind::ErasureFence as i32,
        payload_json: r#"{"private":"discard"}"#.into(),
        ..Default::default()
    });
    for socket in [&mut voices, &mut evaluation] {
        assert_eq!(next_json(socket).await["type"], "live_reset");
        close_policy(socket, "erasure_fence").await;
    }
    server.abort();
}

#[tokio::test]
async fn voice_profile_pagination_preserves_receipts_and_confirmed_delete_fences() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let (addr, server) = serve(app.clone()).await;
    let mut voice = connect(addr, "/ws/voices", OPERATOR_TOKEN).await.unwrap();
    let connection = v3(&mut voice).await;
    let read = |path: String| {
        let app = app.clone();
        let connection = connection.clone();
        async move {
            app.oneshot(
                axum::http::Request::get(path)
                    .header("authorization", format!("Bearer {OPERATOR_TOKEN}"))
                    .header("x-kza-live-connection", connection)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    };
    let path = "/api/voices/profiles";
    let first = read(path.into()).await;
    assert_eq!(first.status(), StatusCode::OK);
    let first = document(first).await;
    let first_row = &first["snapshot"]["rows"][0];
    let cursor = first["next_cursor"].as_str().unwrap();
    assert_eq!(first["connection_id"], connection);
    assert_eq!(first["snapshot"]["next_cursor"], cursor);
    assert_eq!(first_row["id"], cursor);
    assert!(first_row["name"].as_str().is_some());
    let garbage = read(format!("{path}?cursor=%00")).await;
    assert_eq!(garbage.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let second = read(format!("{path}?cursor={cursor}")).await;
    let second = document(second).await;
    assert_ne!(second["snapshot"]["rows"][0]["id"], cursor);
    assert!(second["next_cursor"].is_null());

    let target = format!("{path}/{cursor}");
    let stale = call(
        &app,
        "DELETE",
        &target,
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::from(r#"{"expected_revision":0}"#),
    )
    .await;
    assert_eq!(stale.status(), StatusCode::CONFLICT);
    assert_eq!(read(path.into()).await.status(), StatusCode::OK);

    let target = format!("{target}?expected_revision=1");
    let deleted = call(
        &app,
        "DELETE",
        &target,
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::empty(),
    )
    .await;
    assert_eq!(deleted.status(), StatusCode::OK);
    assert_eq!(document(deleted).await["deleted"], true);
    let retired = next_json(&mut voice).await;
    assert_eq!(retired["type"], "rest_retire");
    assert_eq!(retired["channel"], "voices:profiles");
    assert_eq!(next_json(&mut voice).await["type"], "live_reset");
    close_policy(&mut voice, "erasure_fence").await;
    assert_eq!(
        read(path.into()).await.status(),
        StatusCode::PRECONDITION_REQUIRED
    );
    server.abort();
}

#[tokio::test]
async fn websocket_expiration_is_checked_while_idle() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let (addr, server) = serve(app).await;
    let mut socket = connect(addr, "/ws/evaluation", EVALUATOR_TOKEN)
        .await
        .unwrap();
    v3(&mut socket).await;
    rusqlite::Connection::open(fixture.database_path())
        .unwrap()
        .execute(
            "UPDATE credentials SET expires_at=0 WHERE role='evaluator'",
            [],
        )
        .unwrap();
    assert_eq!(next_json(&mut socket).await["type"], "live_reset");
    close_policy(&mut socket, "lease_retired").await;
    server.abort();
}

struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn cli(insecure: bool, static_dir: &Path) -> (Process, SocketAddr) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nexa-web"));
    cmd.args(["--test-mode", "--bind", "127.0.0.1:0", "--static-dir"])
        .arg(static_dir)
        .stdout(Stdio::piped())
        .env_remove("NOTIFY_SOCKET")
        .env_remove("CLICOLOR_FORCE")
        .env("NO_COLOR", "1");
    if insecure {
        cmd.arg("--allow-insecure-cookies");
    }
    #[cfg(target_os = "linux")]
    let (_notify_dir, notify) = {
        let directory = tempfile::tempdir_in("/tmp").unwrap();
        let path = directory.path().join("notify.sock");
        let socket = std::os::unix::net::UnixDatagram::bind(&path).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        cmd.env("NOTIFY_SOCKET", path).env_remove("WATCHDOG_USEC");
        (directory, socket)
    };
    let mut child = Process(cmd.spawn().unwrap());
    let mut output = BufReader::new(child.0.stdout.take().unwrap());
    let (send, recv) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        if output.read_line(&mut line).unwrap() > 0 {
            let address = line
                .split_whitespace()
                .find_map(|word| word.strip_prefix("address="));
            if let Some(address) = address {
                let _ = send.send(address.parse::<SocketAddr>().unwrap());
            }
        }
    });
    let addr = recv.recv_timeout(Duration::from_secs(5)).unwrap();
    #[cfg(target_os = "linux")]
    {
        let mut packet = [0; 64];
        let size = notify.recv(&mut packet).unwrap();
        assert!(
            std::str::from_utf8(&packet[..size])
                .unwrap()
                .lines()
                .any(|line| line == "READY=1")
        );
    }
    (child, addr)
}

fn http(addr: SocketAddr, method: &str, path: &str, headers: &str, body: &str) -> String {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    write!(stream,"{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{headers}\r\n{body}",body.len()).unwrap();
    let mut buffer = Vec::new();
    stream.read_to_end(&mut buffer).unwrap();
    String::from_utf8_lossy(&buffer).into_owned()
}

#[test]
fn cli_test_mode_serves_spa_and_seed_cookies() {
    let static_dir = tempfile::tempdir().unwrap();
    fs::write(
        static_dir.path().join("index.html"),
        "<html>Playwright fixture</html>",
    )
    .unwrap();
    for insecure in [false, true] {
        let (_child, addr) = cli(insecure, static_dir.path());
        assert!(http(addr, "GET", "/obs", "", "").ends_with("<html>Playwright fixture</html>"));
        let origin = format!("Origin: http://{addr}\r\n");
        let login = http(
            addr,
            "POST",
            "/api/admin/auth/login",
            &origin,
            &json!({"token":HOME_TOKEN}).to_string(),
        );
        assert!(login.starts_with("HTTP/1.1 200"));
        let cookie = login
            .lines()
            .find_map(|line| line.strip_prefix("set-cookie: "))
            .unwrap();
        assert_eq!(cookie.ends_with("; Secure"), !insecure);
        let cookie = format!("Cookie: {}\r\n", cookie.split(';').next().unwrap());
        assert!(
            http(addr, "GET", "/api/admin/auth/whoami", &cookie, "").contains("\"role\":\"home\"")
        );
        assert!(http(addr, "GET", "/api/home/model", &cookie, "").contains("glTF"));
    }
}

use tower::ServiceExt;
