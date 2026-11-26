#![allow(clippy::expect_used)]

mod support;

use std::{
    collections::HashMap,
    num::{NonZeroU32, NonZeroU64},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime},
};

use futures_util::poll;
use nexa_config::HaControlPolicyConfig;
use nexa_ha::{
    ActionRequest, ActionStatus, Brightness, CircuitBreakerConfig, ControlError, ControlPolicy,
    ControlService, EffectIntent, HaClient, NotSentReason, Operation, OperationError, PlanError,
    StateReadError, authorize_action,
    readers::{
        self, CameraHealth, DiagnosticHealth, DiagnosticKind, EnergyKind, MediaState, ReadError,
    },
};
use serde_json::{Value, json};
use support::{
    HaOrigin, StateSync, TOKEN, authenticated, config, delta, entity, paused, recv, response,
    wait_for,
};
use tokio::time::{self, Instant};
use wiremock::{
    Mock, MockServer,
    matchers::{body_json, method, path},
};

fn client(url: String) -> HaClient {
    HaClient::new(&config(url), TOKEN).expect("client")
}

fn grants(entries: &[(&str, &[&str])]) -> HaControlPolicyConfig {
    HaControlPolicyConfig {
        entities: entries
            .iter()
            .map(|(id, ops)| ((*id).into(), ops.iter().map(|op| (*op).into()).collect()))
            .collect(),
        ..HaControlPolicyConfig::default()
    }
}

fn intent(entity_id: &str, operation: Operation) -> EffectIntent {
    EffectIntent {
        entity_id: entity_id.into(),
        operation,
    }
}

fn req(request_id: &str, entity_id: &str, operation: Operation) -> ActionRequest {
    ActionRequest {
        request_id: request_id.into(),
        action: intent(entity_id, operation),
    }
}

async fn status(
    service: &ControlService,
    id: &str,
    entity_id: &str,
    operation: Operation,
) -> ActionStatus {
    service
        .execute(req(id, entity_id, operation))
        .await
        .expect("receipt")
        .status
}

async fn mount_state(server: &MockServer, id: &str, state: &str, attributes: Value) {
    authenticated("GET", &format!("/api/states/{id}"))
        .respond_with(response(200).set_body_json(entity(id, state, attributes)))
        .mount(server)
        .await;
}

fn frame_has(ha: &HaClient, entity_id: &str, expected: &str) -> bool {
    ha.frame()
        .entity(entity_id, Instant::now())
        .is_ok_and(|state| state.state() == expected)
}

#[tokio::test]
async fn websocket_reconnect_reconciles_and_projects_updates() {
    paused(async {
        let mut origin = HaOrigin::start().await;
        let recovered = Arc::new(AtomicBool::new(false));
        let snapshot_recovered = Arc::clone(&recovered);
        authenticated("GET", "/api/states")
            .respond_with(move |_: &wiremock::Request| {
                let state = if snapshot_recovered.load(Ordering::SeqCst) {
                    "on"
                } else {
                    "off"
                };
                response(200).set_body_json([entity("light.main", state, json!({}))])
            })
            .mount(&origin.http)
            .await;
        let ha = client(origin.url.clone());
        let _sync = StateSync::start(&ha);
        let mut first = origin.sessions.recv().await.expect("first subscription");
        wait_for(|| frame_has(&ha, "light.main", "off")).await;
        assert!(ha.run_state_sync().await.is_err());
        let before_event = ha.frame();
        delta(
            &mut first,
            "light.event",
            entity("light.event", "on", json!({})),
        )
        .await;
        wait_for(|| frame_has(&ha, "light.event", "on")).await;
        assert!(matches!(
            before_event.entity("light.event", Instant::now()),
            Err(StateReadError::Missing)
        ));
        recovered.store(true, Ordering::SeqCst);
        first.close(None).await.expect("disconnect");
        use futures_util::StreamExt;
        assert!(first.next().await.is_none_or(|message| message.is_err()));
        wait_for(|| frame_has(&ha, "light.main", "on")).await;
        // The reconciliation starts before the reconnect timer is armed.
        time::advance(Duration::from_millis(1001)).await;
        let mut second = origin
            .sessions
            .recv()
            .await
            .expect("reconnected subscription");
        delta(
            &mut second,
            "light.main",
            entity("light.main", "unknown", json!({})),
        )
        .await;
        wait_for(|| frame_has(&ha, "light.main", "unknown")).await;
    })
    .await;
}

#[tokio::test]
async fn silent_websocket_requires_pong_and_reconnects() {
    paused(async {
        let mut origin = HaOrigin::start().await;
        authenticated("GET", "/api/states")
            .respond_with(response(200).set_body_json(Vec::<Value>::new()))
            .mount(&origin.http)
            .await;
        let ha = client(origin.url.clone());
        let _sync = StateSync::start(&ha);
        let mut peer = origin.sessions.recv().await.expect("first subscription");
        delta(
            &mut peer,
            "light.marker",
            entity("light.marker", "on", json!({})),
        )
        .await;
        wait_for(|| frame_has(&ha, "light.marker", "on")).await;
        time::advance(Duration::from_millis(30001)).await;
        let ping = recv(&mut peer).await;
        assert_eq!(ping["type"], "ping");
        time::advance(Duration::from_millis(2001)).await;
        // The closed transport establishes the timeout before advancing backoff.
        use futures_util::StreamExt;
        assert!(peer.next().await.is_none_or(|message| message.is_err()));
        time::advance(Duration::from_millis(1001)).await;
        let _second = origin.sessions.recv().await.expect("heartbeat reconnect");
    })
    .await;
}

#[tokio::test]
async fn in_flight_snapshot_preserves_newer_delta_and_removal() {
    paused(async {
        let mut origin = HaOrigin::start().await;
        let ha = client(origin.url.clone());
        authenticated("GET", "/api/states")
            .respond_with(response(200).set_body_json([
                entity("light.a", "off", json!({})),
                entity("light.b", "off", json!({})),
                entity("light.snapshot", "off", json!({})),
            ]))
            .mount(&origin.http)
            .await;
        let (arrived, release) = origin.hold_next("/api/states");
        let _sync = StateSync::start(&ha);
        arrived.await.expect("snapshot request in flight");
        // Keep any immediate follow-up snapshot from masking the first publication.
        let (_follow_up, _continue) = origin.hold_next("/api/states");
        let mut peer = origin.sessions.recv().await.expect("subscription");
        let mut changed = entity("light.a", "on", json!({}));
        changed["last_updated"] = json!("2026-11-25T00:00:01Z");
        delta(&mut peer, "light.a", changed).await;
        delta(&mut peer, "light.b", Value::Null).await;
        delta(
            &mut peer,
            "light.marker",
            entity("light.marker", "on", json!({})),
        )
        .await;
        wait_for(|| frame_has(&ha, "light.marker", "on")).await;
        release.send(()).expect("release REST");
        wait_for(|| frame_has(&ha, "light.snapshot", "off")).await;
        assert!(frame_has(&ha, "light.a", "on"));
        assert!(matches!(
            ha.frame().entity("light.b", Instant::now()),
            Err(StateReadError::Missing)
        ));
    })
    .await;
}

#[tokio::test]
async fn periodic_rest_refreshes_even_when_websocket_is_unavailable() {
    paused(async {
        let server = MockServer::start().await;
        let mut settings = config(server.uri());
        settings.full_refresh_interval_s = 3;
        let ha = HaClient::new(&settings, TOKEN).expect("client");
        let updated = Arc::new(AtomicBool::new(false));
        let snapshot_updated = Arc::clone(&updated);
        authenticated("GET", "/api/states")
            .respond_with(move |_: &wiremock::Request| {
                let state = if snapshot_updated.load(Ordering::SeqCst) {
                    "on"
                } else {
                    "off"
                };
                response(200).set_body_json([entity("light.main", state, json!({}))])
            })
            .expect(2)
            .mount(&server)
            .await;
        let _sync = StateSync::start(&ha);
        wait_for(|| frame_has(&ha, "light.main", "off")).await;
        updated.store(true, Ordering::SeqCst);
        time::advance(Duration::from_millis(3001)).await;
        wait_for(|| frame_has(&ha, "light.main", "on")).await;
    })
    .await;
}

#[tokio::test]
async fn fresh_unavailable_blocks_unknown_scene_runs_and_expired_unavailable_rechecks() {
    paused(async {
        let server = MockServer::start().await;
        authenticated("GET", "/api/states")
            .respond_with(response(200).set_body_json([
                entity("light.recovered", "unavailable", json!({})),
                entity("light.outage", "unavailable", json!({})),
                entity(
                    "scene.first",
                    "unknown",
                    json!({"entity_id": ["light.scene_leaf"]}),
                ),
                entity("light.scene_leaf", "off", json!({})),
            ]))
            .mount(&server)
            .await;
        authenticated("GET", "/api/states/light.recovered")
            .respond_with(response(200).set_body_json(entity(
                "light.recovered",
                "unknown",
                json!({}),
            )))
            .expect(1)
            .mount(&server)
            .await;
        authenticated("GET", "/api/states/light.outage")
            .respond_with(response(503))
            .expect(1)
            .mount(&server)
            .await;
        for id in ["light.recovered", "light.scene_leaf"] {
            authenticated("POST", "/api/services/light/turn_on")
                .and(body_json(json!({"entity_id": id})))
                .respond_with(response(200))
                .expect(1)
                .mount(&server)
                .await;
        }
        Mock::given(method("POST"))
            .and(path("/api/services/scene/turn_on"))
            .respond_with(response(500))
            .expect(0)
            .mount(&server)
            .await;
        let ha = client(server.uri());
        let _sync = StateSync::start(&ha);
        wait_for(|| frame_has(&ha, "light.recovered", "unavailable")).await;
        let policy = ControlPolicy::new(grants(&[
            ("light.recovered", &["turn_on"]),
            ("light.outage", &["turn_on"]),
            ("scene.first", &["activate_scene"]),
            ("light.scene_leaf", &["turn_on"]),
        ]))
        .with_effects(
            HashMap::from([(
                "scene.first".into(),
                vec![intent("light.scene_leaf", Operation::TurnOn)],
            )]),
            HashMap::new(),
        );
        let service = ControlService::new(ha.clone(), policy);
        assert_eq!(
            status(&service, "blocked", "light.recovered", Operation::TurnOn).await,
            ActionStatus::NotSent(NotSentReason::Unavailable)
        );
        assert_eq!(
            status(&service, "scene", "scene.first", Operation::ActivateScene).await,
            ActionStatus::Accepted
        );
        time::advance(Duration::from_secs(5)).await;
        assert!(matches!(
            ha.frame().entity("light.recovered", Instant::now()),
            Err(StateReadError::Stale)
        ));
        assert_eq!(
            status(&service, "recovered", "light.recovered", Operation::TurnOn).await,
            ActionStatus::Accepted
        );
        assert_eq!(
            status(
                &service,
                "still-uncertain",
                "light.outage",
                Operation::TurnOn
            )
            .await,
            ActionStatus::NotSent(NotSentReason::StateUnverified)
        );
    })
    .await;
}

#[tokio::test]
async fn fail_closed_denies_ungranted_restricted_and_unreviewed_effects() {
    paused(async {
        let server = MockServer::start().await;
        mock_no_traffic(&server).await;
        let ha = client(server.uri());
        let policy = grants(&[
            ("lock.front", &["unlock", "turn_on"]),
            ("alarm_control_panel.home", &["alarm_disarm", "turn_off"]),
            ("scene.missing", &["activate_scene"]),
            ("scene.hidden", &["activate_scene"]),
            ("light.dim", &["brightness"]),
            ("light.patio", &["color_temp_kelvin"]),
        ]);
        assert!(authorize_action(&policy, "lock.front", "unlock").is_err());
        assert!(authorize_action(&policy, "alarm_control_panel.home", "alarm_disarm").is_err());
        let service = ControlService::new(
            ha,
            ControlPolicy::new(policy).with_effects(
                HashMap::from([(
                    "scene.hidden".into(),
                    vec![intent("lock.front", Operation::TurnOn)],
                )]),
                HashMap::new(),
            ),
        );
        for (id, entity_id, op, reason) in [
            ("one", "lock.front", Operation::TurnOn, PlanError::Denied),
            (
                "two",
                "alarm_control_panel.home",
                Operation::TurnOff,
                PlanError::Denied,
            ),
            (
                "three",
                "light.unselected",
                Operation::TurnOn,
                PlanError::Denied,
            ),
            (
                "four",
                "scene.missing",
                Operation::ActivateScene,
                PlanError::Unreviewed,
            ),
            (
                "five",
                "scene.hidden",
                Operation::ActivateScene,
                PlanError::Denied,
            ),
            (
                "six",
                "light.dim",
                Operation::Brightness {
                    brightness: Brightness::Percent(101),
                },
                PlanError::InvalidParameters,
            ),
            (
                "seven",
                "light.patio",
                Operation::ColorTempKelvin {
                    color_temp_kelvin: 2700,
                    brightness: Some(Brightness::Percent(50)),
                },
                PlanError::Denied,
            ),
        ] {
            assert_eq!(
                status(&service, id, entity_id, op).await,
                ActionStatus::NotSent(NotSentReason::Policy(reason))
            );
        }
        assert!(
            serde_json::from_value::<Operation>(
                json!({"type": "turn_on", "entity_id": "lock.front"})
            )
            .is_err()
        );
    })
    .await;
}

async fn mock_no_traffic(server: &MockServer) {
    for verb in ["GET", "POST"] {
        Mock::given(method(verb))
            .respond_with(response(500))
            .expect(0)
            .mount(server)
            .await;
    }
}

#[tokio::test]
async fn receipts_survive_cancellation_and_fifo_is_per_entity() {
    paused(async {
        let origin = HaOrigin::start().await;
        let ha = client(origin.url.clone());
        let service = ControlService::new(
            ha,
            ControlPolicy::new(grants(&[
                ("light.same", &["turn_on", "turn_off"]),
                ("light.other", &["turn_on"]),
            ])),
        );
        mount_state(&origin.http, "light.same", "off", json!({})).await;
        mount_state(&origin.http, "light.other", "off", json!({})).await;
        for (verb, entity_id) in [
            ("turn_on", "light.same"),
            ("turn_off", "light.same"),
            ("turn_on", "light.other"),
        ] {
            authenticated("POST", &format!("/api/services/light/{verb}"))
                .and(body_json(json!({"entity_id": entity_id})))
                .respond_with(response(200))
                .expect(1)
                .mount(&origin.http)
                .await;
        }
        let (arrived, release) = origin.hold_next("/api/services/light/turn_on");
        let request = req("idempotent", "light.same", Operation::TurnOn);
        let mut first = Box::pin(service.execute(request.clone()));
        assert!(poll!(&mut first).is_pending());
        arrived.await.expect("first POST");
        drop(first);
        let conflict = service
            .execute(req("idempotent", "light.same", Operation::TurnOff))
            .await
            .expect_err("body conflict");
        assert_eq!(conflict, ControlError::IdempotencyConflict);
        assert_eq!(conflict.http_status_code(), 409);
        let mut second = Box::pin(service.execute(req("next", "light.same", Operation::TurnOff)));
        let mut replay = Box::pin(service.execute(request.clone()));
        assert!(poll!(&mut second).is_pending());
        assert!(poll!(&mut replay).is_pending());
        assert_eq!(
            status(&service, "other", "light.other", Operation::TurnOn).await,
            ActionStatus::Accepted
        );
        let requests = origin.http.received_requests().await.expect("request log");
        assert!(
            !requests
                .iter()
                .any(|request| request.url.path() == "/api/services/light/turn_off")
        );
        assert!(service.get_receipt("idempotent").is_none());
        release.send(()).expect("release original POST");
        let original = replay.await.expect("original receipt");
        assert_eq!(
            second.await.expect("queued receipt").status,
            ActionStatus::Accepted
        );
        assert_eq!(original.status, ActionStatus::Accepted);
        assert!(Arc::ptr_eq(
            &original,
            &service.get_receipt("idempotent").expect("stored receipt")
        ));
        assert!(Arc::ptr_eq(
            &original,
            &service.execute(request).await.expect("later replay")
        ));
    })
    .await;
}

#[tokio::test]
async fn breaker_opens_after_failures_and_allows_only_one_half_open_probe() {
    paused(async {
        let origin = HaOrigin::start().await;
        let service = ControlService::with_breaker(
            client(origin.url.clone()),
            ControlPolicy::new(grants(&[
                ("light.first", &["turn_on"]),
                ("light.second", &["turn_on"]),
            ])),
            CircuitBreakerConfig {
                failure_threshold: NonZeroU32::new(2).expect("positive threshold"),
                cooldown_s: NonZeroU64::new(10).expect("positive duration"),
            },
        );
        for id in ["light.first", "light.second"] {
            mount_state(&origin.http, id, "off", json!({})).await;
        }
        let healthy = Arc::new(AtomicBool::new(false));
        let response_healthy = Arc::clone(&healthy);
        authenticated("POST", "/api/services/light/turn_on")
            .respond_with(move |_: &wiremock::Request| {
                response(if response_healthy.load(Ordering::SeqCst) {
                    200
                } else {
                    503
                })
            })
            .expect(5)
            .mount(&origin.http)
            .await;
        for id in ["first-failure", "second-failure"] {
            assert_eq!(
                status(&service, id, "light.first", Operation::TurnOn).await,
                ActionStatus::Uncertain
            );
        }
        assert_eq!(
            status(&service, "blocked", "light.second", Operation::TurnOn).await,
            ActionStatus::NotSent(NotSentReason::CircuitOpen)
        );
        time::advance(Duration::from_secs(10)).await;
        let (arrived, release) = origin.hold_next("/api/services/light/turn_on");
        let mut probe =
            Box::pin(service.execute(req("probe-fails", "light.first", Operation::TurnOn)));
        assert!(poll!(&mut probe).is_pending());
        arrived.await.expect("half-open probe started");
        assert_eq!(
            status(
                &service,
                "other-during-probe",
                "light.second",
                Operation::TurnOn
            )
            .await,
            ActionStatus::NotSent(NotSentReason::CircuitOpen)
        );
        release.send(()).expect("let probe fail");
        assert_eq!(
            probe.await.expect("probe receipt").status,
            ActionStatus::Uncertain
        );
        healthy.store(true, Ordering::SeqCst);
        assert_eq!(
            status(&service, "still-open", "light.second", Operation::TurnOn).await,
            ActionStatus::NotSent(NotSentReason::CircuitOpen)
        );
        time::advance(Duration::from_secs(10)).await;
        assert_eq!(
            status(&service, "probe-recovers", "light.first", Operation::TurnOn).await,
            ActionStatus::Accepted
        );
        assert_eq!(
            status(&service, "closed", "light.second", Operation::TurnOn).await,
            ActionStatus::Accepted
        );
    })
    .await;
}

#[tokio::test]
async fn timeout_after_dispatch_is_recorded_without_replay() {
    paused(async {
        let origin = HaOrigin::start().await;
        let service = ControlService::new(
            client(origin.url.clone()),
            ControlPolicy::new(grants(&[("light.target", &["turn_on"])])),
        );
        mount_state(&origin.http, "light.target", "off", json!({})).await;
        let post = authenticated("POST", "/api/services/light/turn_on")
            .respond_with(response(200))
            .expect(1)
            .mount_as_scoped(&origin.http)
            .await;
        let (_arrived, _release) = origin.hold_next("/api/services/light/turn_on");
        let action = req("timeout", "light.target", Operation::TurnOn);
        let mut pending = Box::pin(service.execute(action.clone()));
        assert!(poll!(&mut pending).is_pending());
        post.wait_until_satisfied().await;
        time::advance(Duration::from_millis(2001)).await;
        assert_eq!(
            pending.await.expect("timeout receipt").status,
            ActionStatus::Uncertain
        );
        assert_eq!(
            service
                .execute(action)
                .await
                .expect("original timeout")
                .status,
            ActionStatus::Uncertain
        );
    })
    .await;
}

#[tokio::test]
async fn typed_operations_apply_observed_bounds_and_emit_exact_service_data() {
    paused(async {
        let server = MockServer::start().await;
        let service = ControlService::new(client(server.uri()), ControlPolicy::new(grants(&[
            ("light.test", &["brightness", "color_temp_kelvin"]),
            ("light.unreviewed_group", &["turn_on"]),
            ("cover.window", &["set_cover_position"]),
            ("climate.test", &["set_temperature"]),
        ])));
        mount_state(&server, "light.test", "off", json!({
            "supported_color_modes": ["color_temp"], "min_color_temp_kelvin": 2500, "max_color_temp_kelvin": 5000,
        })).await;
        mount_state(&server, "light.unreviewed_group", "on", json!({"entity_id": ["light.test"]})).await;
        mount_state(&server, "cover.window", "open", json!({"device_class": "blind", "supported_features": 4})).await;
        mount_state(&server, "climate.test", "cool", json!({
            "supported_features": 1, "hvac_modes": ["off", "cool"], "min_temp": 17.0, "max_temp": 29.0, "target_temp_step": 0.5,
        })).await;
        authenticated("GET", "/api/config").respond_with(response(200).set_body_json(json!({"unit_system":{"temperature":"°C"}}))).expect(2).mount(&server).await;
        for (route, data) in [
            ("/api/services/light/turn_on", json!({"entity_id": "light.test", "color_temp_kelvin": 2700, "brightness": 127})),
            ("/api/services/light/turn_off", json!({"entity_id": "light.test"})),
            ("/api/services/cover/set_cover_position", json!({"entity_id": "cover.window", "position": 55})),
            ("/api/services/climate/set_temperature", json!({"entity_id": "climate.test", "temperature": 22.5})),
        ] {
            authenticated("POST", route).and(body_json(data)).respond_with(response(200)).expect(1).mount(&server).await;
        }
        for (id, entity_id, operation, expected) in [
            ("good-color", "light.test", Operation::ColorTempKelvin { color_temp_kelvin: 2700, brightness: Some(Brightness::Raw(127)) }, ActionStatus::Accepted),
            ("off-by-dim", "light.test", Operation::Brightness { brightness: Brightness::Percent(0) }, ActionStatus::Accepted),
            ("bad-color", "light.test", Operation::ColorTempKelvin { color_temp_kelvin: 2000, brightness: None }, ActionStatus::NotSent(NotSentReason::Capability(OperationError::Unsupported))),
            ("cover-pos", "cover.window", Operation::SetCoverPosition { position: 55 }, ActionStatus::Accepted),
            ("good-temp", "climate.test", Operation::SetTemperature { temperature: 22.5 }, ActionStatus::Accepted),
            ("bad-step", "climate.test", Operation::SetTemperature { temperature: 22.2 }, ActionStatus::NotSent(NotSentReason::Capability(OperationError::Unsupported))),
            ("native-group", "light.unreviewed_group", Operation::TurnOn, ActionStatus::NotSent(NotSentReason::Capability(OperationError::UnreviewedGroup))),
        ] {
            assert_eq!(status(&service, id, entity_id, operation).await, expected);
        }
        assert_eq!(server.received_requests().await.expect("request log").iter().filter(|request| request.method == wiremock::http::Method::POST).count(), 4);
    }).await;
}

#[tokio::test]
async fn readers_project_only_current_state_and_expire_at_the_same_bound() {
    paused(async {
        let server = MockServer::start().await;
        authenticated("GET", "/api/states").respond_with(response(200).set_body_json([
            entity("sensor.power", "1.5", json!({"device_class": "power", "unit_of_measurement": "kW"})),
            entity("calendar.work", "off", json!({
                "message": "Synthetic event", "start_time": "2026-11-25 09:00:00", "end_time": "2026-11-25 10:00:00",
            })),
            entity("media_player.music", "playing", json!({"volume_level": 0.25, "is_volume_muted": false, "media_title": "Test song"})),
            entity("camera.safe", "recording", json!({"access_token": "synthetic-private-data"})),
            entity("binary_sensor.camera_net", "off", json!({"device_class": "connectivity"})),
        ])).mount(&server).await;
        let ha = client(server.uri());
        let _sync = StateSync::start(&ha);
        wait_for(|| frame_has(&ha, "sensor.power", "1.5")).await;
        let frame = ha.frame();
        let reported = chrono::DateTime::parse_from_rfc3339(support::UPDATED).expect("fixture timestamp");
        let wall = SystemTime::from(reported) + Duration::from_secs(4);
        let max_age = Duration::from_secs(5);
        let now = Instant::now();
        let energy = readers::energy(&frame, "sensor.power", EnergyKind::Power, now, wall, max_age).expect("energy");
        assert_eq!((energy.value, energy.unit), (1500.0, "W"));
        assert_eq!(
            readers::energy(&frame, "sensor.power", EnergyKind::Power, now, wall + Duration::from_secs(1), max_age),
            Err(ReadError::StaleMeasurement)
        );
        let calendar = readers::calendar(&frame, "calendar.work", now).expect("calendar");
        assert!(!calendar.active);
        assert_eq!(calendar.current_or_next.expect("upcoming event").summary, "Synthetic event");
        let media = readers::media(&frame, "media_player.music", now).expect("media");
        assert_eq!((media.state, media.volume), (MediaState::Playing, Some(0.25)));
        assert_eq!(readers::camera_health(&frame, "camera.safe", now), Ok(CameraHealth::Online));
        assert_eq!(readers::camera_diagnostic(&frame, "binary_sensor.camera_net", DiagnosticKind::Connectivity, now), Ok(DiagnosticHealth::Issue));

        time::advance(max_age).await;
        let now = Instant::now();
        let stale = ReadError::Observation(StateReadError::Stale);
        assert_eq!(readers::energy(&frame, "sensor.power", EnergyKind::Power, now, wall, max_age), Err(stale));
        assert_eq!(readers::calendar(&frame, "calendar.work", now), Err(stale));
        assert_eq!(readers::media(&frame, "media_player.music", now), Err(stale));
        assert_eq!(readers::camera_health(&frame, "camera.safe", now), Err(stale));
        assert_eq!(readers::camera_diagnostic(&frame, "binary_sensor.camera_net", DiagnosticKind::Connectivity, now), Err(stale));
    }).await;
}

#[tokio::test]
async fn point_lookup_cannot_resurrect_an_entity_removed_while_it_was_in_flight() {
    paused(async {
        let mut origin = HaOrigin::start().await;
        authenticated("GET", "/api/states")
            .respond_with(response(200).set_body_json([entity("light.marker", "off", json!({}))]))
            .mount(&origin.http)
            .await;
        let stale = authenticated("GET", "/api/states/light.removed")
            .respond_with(response(200).set_body_json(entity("light.removed", "off", json!({}))))
            .expect(1)
            .mount_as_scoped(&origin.http)
            .await;
        Mock::given(method("POST"))
            .respond_with(response(200))
            .expect(0)
            .mount(&origin.http)
            .await;
        let ha = client(origin.url.clone());
        let _sync = StateSync::start(&ha);
        let mut peer = origin.sessions.recv().await.expect("subscription");
        wait_for(|| frame_has(&ha, "light.marker", "off")).await;
        // Ensure a follow-up snapshot cannot erase the tombstone during this test.
        let (_follow_up, _continue) = origin.hold_next("/api/states");
        let (_arrived, release) = origin.hold_next("/api/states/light.removed");
        let service = ControlService::new(
            ha.clone(),
            ControlPolicy::new(grants(&[("light.removed", &["turn_on"])])),
        );
        let mut pending =
            Box::pin(service.execute(req("remove-race", "light.removed", Operation::TurnOn)));
        assert!(poll!(&mut pending).is_pending());
        stale.wait_until_satisfied().await;
        delta(&mut peer, "light.removed", Value::Null).await;
        delta(
            &mut peer,
            "light.marker",
            entity("light.marker", "on", json!({})),
        )
        .await;
        wait_for(|| frame_has(&ha, "light.marker", "on")).await;
        release.send(()).expect("release point lookup");
        assert_eq!(
            pending.await.expect("removal receipt").status,
            ActionStatus::NotSent(NotSentReason::StateUnverified)
        );
    })
    .await;
}

#[tokio::test]
async fn scene_preflight_verifies_membership_and_every_leaf_before_dispatch() {
    paused(async {
        let server = MockServer::start().await;
        for (entity_id, state, attrs) in [
            (
                "scene.drift",
                "unknown",
                json!({"entity_id": ["light.first", "lock.hidden"]}),
            ),
            (
                "scene.correct",
                "unknown",
                json!({"entity_id": ["light.second", "light.first"]}),
            ),
            ("light.first", "off", json!({})),
            ("light.second", "unavailable", json!({})),
        ] {
            mount_state(&server, entity_id, state, attrs).await;
        }
        Mock::given(method("POST"))
            .respond_with(response(200))
            .expect(0)
            .mount(&server)
            .await;
        let leaves = vec![
            intent("light.first", Operation::TurnOn),
            intent("light.second", Operation::TurnOn),
        ];
        let policy = ControlPolicy::new(grants(&[
            ("scene.drift", &["activate_scene"]),
            ("scene.correct", &["activate_scene"]),
            ("light.first", &["turn_on"]),
            ("light.second", &["turn_on"]),
        ]))
        .with_effects(
            HashMap::from([
                ("scene.drift".into(), leaves.clone()),
                ("scene.correct".into(), leaves),
            ]),
            HashMap::new(),
        );
        let service = ControlService::new(client(server.uri()), policy);
        assert_eq!(
            status(
                &service,
                "membership-changed",
                "scene.drift",
                Operation::ActivateScene
            )
            .await,
            ActionStatus::NotSent(NotSentReason::Policy(PlanError::MembershipChanged))
        );
        assert_eq!(
            status(
                &service,
                "leaf-unavailable",
                "scene.correct",
                Operation::ActivateScene
            )
            .await,
            ActionStatus::NotSent(NotSentReason::Unavailable)
        );
    })
    .await;
}
