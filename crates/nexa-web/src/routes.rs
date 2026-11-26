//! Explicit HTTP inventory. Each method is attached to its own authority policy.

mod contracts;

use std::{
    collections::HashMap,
    fs::File,
    path::{Path as FsPath, PathBuf},
};

use axum::{
    Json, Router,
    body::{Body, to_bytes},
    extract::{FromRequestParts, Path, Request, State},
    handler::Handler,
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{MethodFilter, get, on},
};
use nexa_proto::v1::{
    Permission::{self, *},
    *,
};
use rustix::fs::{Mode, OFlags, open, openat};
use serde_json::{Value, json};
use tokio_util::io::ReaderStream;

use crate::{
    AppState,
    auth::{self, AuthError, Lease, Presented},
    sys::MonitorError,
    ws,
};
use contracts::*;

#[derive(Debug)]
pub(crate) struct ApiError(pub StatusCode, pub &'static str);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.0, Json(json!({"detail":self.1}))).into_response();
        no_store(&mut response);
        if self.0 == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

impl From<AuthError> for ApiError {
    fn from(error: AuthError) -> Self {
        let status = match error {
            AuthError::Unauthorized => StatusCode::UNAUTHORIZED,
            AuthError::Forbidden => StatusCode::FORBIDDEN,
            AuthError::Invalid => StatusCode::UNPROCESSABLE_ENTITY,
            AuthError::NotFound => StatusCode::NOT_FOUND,
            AuthError::Conflict => StatusCode::CONFLICT,
            AuthError::Unavailable | AuthError::EvaluationUnavailable => {
                StatusCode::SERVICE_UNAVAILABLE
            }
        };
        let detail = match error {
            AuthError::Unauthorized => "dashboard_session_required",
            AuthError::Forbidden => "dashboard_permission_denied",
            AuthError::Invalid => "invalid_authority_request",
            AuthError::NotFound => "authority_record_not_found",
            AuthError::Conflict => "authority_revision_conflict",
            AuthError::Unavailable => "authority_unavailable",
            AuthError::EvaluationUnavailable => "evaluation_source_binding_unavailable",
        };
        Self(status, detail)
    }
}

impl From<tonic::Status> for ApiError {
    fn from(error: tonic::Status) -> Self {
        use tonic::Code::*;
        match error.code() {
            InvalidArgument | OutOfRange => {
                Self(StatusCode::UNPROCESSABLE_ENTITY, "invalid_upstream_request")
            }
            Unauthenticated => AuthError::Unauthorized.into(),
            PermissionDenied => AuthError::Forbidden.into(),
            NotFound => Self(StatusCode::NOT_FOUND, "upstream_record_not_found"),
            AlreadyExists | Aborted | FailedPrecondition => {
                Self(StatusCode::CONFLICT, "upstream_conflict")
            }
            DeadlineExceeded => Self(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout"),
            ResourceExhausted => Self(StatusCode::TOO_MANY_REQUESTS, "upstream_capacity"),
            _ => Self(StatusCode::SERVICE_UNAVAILABLE, "upstream_unavailable"),
        }
    }
}

impl From<MonitorError> for ApiError {
    fn from(error: MonitorError) -> Self {
        match error {
            MonitorError::Denied => Self(StatusCode::FORBIDDEN, "service_restart_denied"),
            MonitorError::Unavailable => Self(
                StatusCode::SERVICE_UNAVAILABLE,
                "system_monitor_unavailable",
            ),
            MonitorError::Timeout => Self(StatusCode::GATEWAY_TIMEOUT, "restart_timeout"),
            MonitorError::Failed => {
                Self(StatusCode::INTERNAL_SERVER_ERROR, "restart_command_failed")
            }
            MonitorError::Demo => Self(StatusCode::CONFLICT, "observability_demo_mode"),
        }
    }
}

const INVALID: ApiError = ApiError(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request");
const BAD_UPSTREAM: ApiError = ApiError(
    StatusCode::SERVICE_UNAVAILABLE,
    "upstream_projection_unavailable",
);
const NOT_FOUND: ApiError = ApiError(StatusCode::NOT_FOUND, "Not Found");
type Result<T> = std::result::Result<T, ApiError>;

#[derive(Clone, Copy)]
enum Op {
    Health,
    Login,
    Logout,
    Whoami,
    Model,
    Plan,
    Catalog,
    Microphones,
    HomeAction,
    HomeReceipt,
    Routines,
    Routine,
    RoutinePreview,
    RoutineUpsert,
    RoutineDelete,
    RoutineExecute,
    RoutineEnabled(bool),
    Lists,
    ListSync,
    ListItems,
    ListMutation(&'static str),
    Reminders,
    ReminderNotices,
    ReminderStatus,
    ReminderMutation(&'static str),
    Memory(&'static str),
    MemoryMutation(&'static str),
    DeleteFact,
    VoiceSwitch,
    VoiceSwitchWrite,
    VoiceProfiles,
    VoiceLink,
    VoiceReview(bool),
    VoiceCaptures,
    Capture,
    CaptureWrite,
    ForgetCarried,
    Ambient,
    AmbientWrite,
    EvaluationGrants,
    EvaluationSample,
    Zones,
    Conversations,
    HaEntities,
    HaActions,
    Llm,
    Cooldown,
    Users,
    UserWrite,
    UserRetire,
    Alerts,
    AlertZones,
    AlertZonesWrite,
    AlertAck,
    AlertDelete,
    Gpus,
    Services,
    Restart,
    Cloud,
    CloudWithdraw,
    Credentials,
    CredentialMutation(&'static str),
    CredentialAttempt,
    Unavailable(&'static str),
}

#[derive(Clone, Copy)]
struct Policy {
    op: Op,
    permissions: &'static [Permission],
    public: bool,
}

pub(crate) fn router(state: AppState) -> Router {
    let mut router = Router::new();
    macro_rules! routes {
        ($($verb:ident $path:literal => $op:expr, $perms:expr;)*) => {
            $(router = add(router, &state, MethodFilter::$verb, $path, $op, $perms);)*
        };
    }
    use Op::*;
    routes! {
        GET "/api/health" => Health, &[];
        POST "/api/admin/auth/login" => Login, &[];
        POST "/api/admin/auth/logout" => Logout, &[];
        GET "/api/admin/auth/whoami" => Whoami, &[];

        GET "/api/home/catalog" => Catalog, &[HomeRead];
        GET "/api/home/microphones" => Microphones, &[HomeRead, DiagnosticsRead];
        POST "/api/home/actions" => HomeAction, &[HomeControl];
        GET "/api/home/actions" => HomeReceipt, &[HomeRead];
        GET "/api/home/model" => Model, &[HomeRead];
        GET "/api/home/plan" => Plan, &[HomeRead];

        GET "/api/routines" => Routines, &[HomeRead];
        GET "/api/routines/{id}" => Routine, &[HomeRead];
        POST "/api/routines/preview" => RoutinePreview, &[RoutinesManage];
        POST "/api/routines" => RoutineUpsert, &[RoutinesManage];
        PUT "/api/routines/{id}" => RoutineUpsert, &[RoutinesManage];
        DELETE "/api/routines/{id}" => RoutineDelete, &[RoutinesManage];
        POST "/api/routines/{id}/execute" => RoutineExecute, &[RoutinesExecute];
        POST "/api/routines/{id}/enable" => RoutineEnabled(true), &[RoutinesManage];
        POST "/api/routines/{id}/disable" => RoutineEnabled(false), &[RoutinesManage];
        GET "/api/routines/{id}/runs/{run_id}" => Unavailable("routine_receipt_unavailable"), &[RoutinesExecute];
        GET "/api/routines/{id}/attempts" => Unavailable("routine_receipt_unavailable"), &[RoutinesExecute];
        GET "/api/routines/proposals" => Unavailable("routine_proposals_unavailable"), &[RoutinesManage];
        GET "/api/routines/proposals/{id}" => Unavailable("routine_proposals_unavailable"), &[RoutinesManage];
        POST "/api/routines/proposals/{id}/confirm" => Unavailable("routine_proposals_unavailable"), &[RoutinesManage];
        POST "/api/routines/proposals/{id}/reject" => Unavailable("routine_proposals_unavailable"), &[RoutinesManage];

        GET "/api/lists" => Lists, &[TasksRead];
        POST "/api/lists" => ListMutation("create_list"), &[TasksWrite];
        POST "/api/lists/sync" => ListSync, &[TasksWrite];
        GET "/api/lists/{id}/items" => ListItems, &[TasksRead];
        POST "/api/lists/{id}/items" => ListMutation("add_item"), &[TasksWrite];
        DELETE "/api/lists/{id}/items/{item_id}" => ListMutation("delete_item"), &[TasksWrite];

        GET "/api/reminders" => Reminders, &[TasksRead];
        POST "/api/reminders" => ReminderMutation("create"), &[TasksWrite];
        PUT "/api/reminders/{id}" => ReminderMutation("update"), &[TasksWrite];
        DELETE "/api/reminders/{id}" => ReminderMutation("cancel"), &[TasksWrite];
        GET "/api/reminders/notices" => ReminderNotices, &[TasksRead];
        POST "/api/reminders/notices/dismiss" => ReminderMutation("dismiss_notice"), &[TasksWrite];
        GET "/api/reminders/status" => ReminderStatus, &[DiagnosticsRead];

        GET "/api/memory" => Memory("summary"), &[MemoryRead];
        GET "/api/memory/facts" => Memory("facts"), &[MemoryRead];
        GET "/api/memory/proposals" => Memory("proposals"), &[MemoryRead];
        GET "/api/memory/returned" => Memory("returned"), &[MemoryRead, MemoryAssign];
        GET "/api/memory/owners/counts" => Memory("owners_counts"), &[MemoryRead, MemoryAssign];
        POST "/api/memory/facts/{id}/owner" => MemoryMutation("assign_owner"), &[MemoryAssign];
        POST "/api/memory/proposals/{id}/confirm" => MemoryMutation("confirm_proposal"), &[MemoryReview];
        POST "/api/memory/proposals/{id}/reject" => MemoryMutation("reject_proposal"), &[MemoryReview];
        DELETE "/api/memory/facts/{id}" => DeleteFact, &[MemoryReview];
        GET "/api/memory/mine" => Memory("mine"), &[MemoryOwnRead];
        DELETE "/api/memory/mine/facts/{id}" => DeleteFact, &[MemoryOwnWrite];
        POST "/api/memory/mine/facts/{id}/return" => MemoryMutation("return_fact"), &[MemoryOwnWrite];
        GET "/api/memory/mine/voice" => Memory("voice_mine"), &[MemoryOwnVoice];
        POST "/api/memory/mine/voice/consent" => MemoryMutation("voice_consent_grant"), &[MemoryOwnVoice];
        DELETE "/api/memory/mine/voice/consent" => MemoryMutation("voice_consent_revoke"), &[MemoryOwnVoice];
        POST "/api/memory/mine/voice/verification" => MemoryMutation("voice_verification"), &[MemoryOwnVoice];
        GET "/api/memory/voice/consents" => Memory("voice_consents"), &[MemoryRead, MemoryAssign];
        POST "/api/memory/voice/consents/{id}/reopen" => MemoryMutation("reopen_voice_consent"), &[MemoryAssign];
        GET "/api/memory/voice/switch" => VoiceSwitch, &[MemoryRead, MemoryAssign];
        POST "/api/memory/voice/switch" => VoiceSwitchWrite, &[MemoryAssign, AdminManage];

        GET "/api/voices" => VoiceCaptures, &[VoicesRead, VoicesReview];
        GET "/api/voices/profiles" => VoiceProfiles, &[VoicesRead, VoicesReview];
        POST "/api/voices/profiles" => MemoryMutation("confirm_voice_profile"), &[VoicesReview];
        DELETE "/api/voices/profiles/{id}" => MemoryMutation("delete_voice_profile"), &[VoicesReview];
        PUT "/api/voices/profiles/{id}/person" => VoiceLink, &[VoicesReview, AdminManage];
        POST "/api/voices/profiles/{id}/attribution/resume" => MemoryMutation("resume_attribution"), &[VoicesReview, AdminManage, MemoryAssign];
        POST "/api/voices/groups/corrections" => MemoryMutation("correct_voice_group"), &[VoicesReview];
        GET "/api/voices/captures/{id}/review" => VoiceReview(false), &[VoicesRead, VoicesReview];
        GET "/api/voices/captures/{id}/transcript" => VoiceReview(true), &[VoicesRead, VoicesReview];
        POST "/api/voices/captures/query" => VoiceCaptures, &[VoicesRead, VoicesReview];

        GET "/api/capture/control" => Capture, &[AdminManage];
        POST "/api/capture/control" => CaptureWrite, &[AdminManage];
        POST "/api/capture/control/forget-carried" => ForgetCarried, &[AdminManage];
        GET "/api/ambient/control" => Ambient, &[AmbientControl];
        POST "/api/ambient/control" => AmbientWrite, &[AmbientControl];

        GET "/api/evaluation/grants" => EvaluationGrants, &[EvaluationRead];
        GET "/api/evaluation/samples/{id}" => EvaluationSample, &[EvaluationRead];
        GET "/api/zones" => Zones, &[DiagnosticsRead];
        GET "/api/conversations" => Conversations, &[ConversationsRead];
        GET "/api/ha/entities" => HaEntities, &[DiagnosticsRead];
        GET "/api/ha/actions" => HaActions, &[DiagnosticsRead];
        GET "/api/llm/endpoints" => Llm, &[DiagnosticsRead];
        POST "/api/llm/endpoints/{id}/clear-cooldown" => Cooldown, &[AdminManage];
        GET "/api/users" => Users, &[DiagnosticsRead];

        GET "/api/alerts" => Alerts, &[HomeRead];
        GET "/api/alerts/zones" => AlertZones, &[HomeRead];
        PUT "/api/alerts/zones" => AlertZonesWrite, &[AdminManage];
        GET "/api/system/gpus" => Gpus, &[DiagnosticsRead];
        GET "/api/system/services" => Services, &[DiagnosticsRead];
        GET "/api/admin/users" => Users, &[AdminManage];
        POST "/api/admin/users" => UserWrite, &[AdminManage];
        PUT "/api/admin/users/{id}/permissions" => UserWrite, &[AdminManage];
        DELETE "/api/admin/users/{id}" => UserRetire, &[AdminManage, AuthManage];
        POST "/api/admin/users/{id}/enroll" => Unavailable("voice_enrollment_unavailable"), &[AdminManage];
        POST "/api/admin/alerts/{id}/ack" => AlertAck, &[AdminManage];
        DELETE "/api/admin/alerts/{id}" => AlertDelete, &[AdminManage];
        POST "/api/admin/services/{id}/restart" => Restart, &[AdminManage];
        GET "/api/admin/cloud-consent" => Cloud, &[AdminManage];
        POST "/api/admin/cloud-consent/withdraw" => CloudWithdraw, &[AdminManage];

        GET "/api/admin/auth/credentials" => Credentials, &[AuthManage];
        POST "/api/admin/auth/credentials" => CredentialMutation("issue"), &[AuthManage];
        PUT "/api/admin/auth/credentials/{id}" => CredentialMutation("revise"), &[AuthManage];
        DELETE "/api/admin/auth/credentials/{id}" => CredentialMutation("revoke"), &[AuthManage];
        GET "/api/admin/auth/credential-attempts/{id}" => CredentialAttempt, &[AuthManage];
    }
    router
        .route("/", get(index))
        .route("/obs", get(index))
        .route("/assets/{*file}", get(public_asset))
        .route("/favicon.ico", get(favicon))
        .with_state(state)
}

fn add(
    router: Router<AppState>,
    state: &AppState,
    method: MethodFilter,
    path: &str,
    op: Op,
    permissions: &'static [Permission],
) -> Router<AppState> {
    let policy = Policy {
        op,
        permissions,
        public: matches!(op, Op::Health | Op::Login),
    };
    let handler =
        move |State(state): State<AppState>, request: Request| dispatch(state, policy, request);
    let handler = handler.layer(middleware::from_fn_with_state(
        (state.clone(), policy),
        authorize,
    ));
    router.route(path, on(method, handler))
}

fn read(method: &Method) -> bool {
    matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS)
}

async fn authorize(
    State((state, policy)): State<(AppState, Policy)>,
    mut request: Request,
    next: Next,
) -> Response {
    if (!read(request.method()) || request.headers().contains_key(header::ORIGIN))
        && state.config.origin.check(request.headers()).is_err()
    {
        return ApiError(StatusCode::FORBIDDEN, "origin denied").into_response();
    }
    if !policy.public {
        let Some(proof) = auth::presented(request.headers()) else {
            return ApiError::from(AuthError::Unauthorized).into_response();
        };
        let lease = match state.auth.validate(&proof, policy.permissions).await {
            Ok(lease) => lease,
            Err(error) => return ApiError::from(error).into_response(),
        };
        request.extensions_mut().insert(proof.bind(&lease));
    }
    let mut response = next.run(request).await;
    no_store(&mut response);
    response
}

async fn dispatch(state: AppState, policy: Policy, request: Request) -> Result<Response> {
    let (mut parts, body) = request.into_parts();
    let path = Path::<HashMap<String, String>>::from_request_parts(&mut parts, &state)
        .await
        .map(|Path(path)| path)
        .unwrap_or_default();
    let query =
        serde_urlencoded::from_str::<Vec<(String, String)>>(parts.uri.query().unwrap_or_default())
            .map_err(|_| INVALID)?;
    let mut unique = HashMap::new();
    for (key, value) in query {
        if unique.insert(key, value).is_some() {
            return Err(INVALID);
        }
    }
    let body = if read(&parts.method) {
        json!({})
    } else {
        json_body(&parts.headers, body).await?
    };
    if matches!(policy.op, Op::Health) {
        return Ok(Json(json!({"status":"ok","service":"nexa-web"})).into_response());
    }
    if matches!(policy.op, Op::Login) {
        let token = string(&body, "token")?;
        let (token, age) = state
            .auth
            .login(token, auth::cookie(&parts.headers))
            .await?;
        let mut response = Json(json!({"ok":true})).into_response();
        response.headers_mut().insert(
            header::SET_COOKIE,
            auth::session_cookie(&token, age, state.config.allow_insecure_cookies)?,
        );
        return Ok(response);
    }
    let proof = parts
        .extensions
        .remove::<Presented>()
        .ok_or(AuthError::Unauthorized)?;
    // Reading the body can wait on the client. Obtain a fresh lease after it finishes.
    let lease = state.auth.validate(&proof, policy.permissions).await?;
    let fields = Fields {
        path,
        query: unique,
        body,
        headers: parts.headers,
    };
    let response = execute(&state, policy, &fields, &proof, &lease).await?;
    if !matches!(policy.op, Op::Logout) && response.status().is_success() {
        state.auth.validate(&proof, policy.permissions).await?;
    }
    Ok(response)
}

async fn json_body(headers: &HeaderMap, body: Body) -> Result<Value> {
    let bytes = to_bytes(body, 256 * 1024)
        .await
        .map_err(|_| ApiError(StatusCode::PAYLOAD_TOO_LARGE, "request_body_too_large"))?;
    if bytes.is_empty() {
        return Ok(json!({}));
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    if !content_type.is_some_and(|value| {
        value
            .split(';')
            .next()
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))
    }) {
        return Err(ApiError(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "json_required",
        ));
    }
    parse_object(&bytes)
}

struct Fields {
    path: HashMap<String, String>,
    query: HashMap<String, String>,
    body: Value,
    headers: HeaderMap,
}

impl Fields {
    fn id(&self) -> &str {
        self.path.get("id").map_or("", String::as_str)
    }

    fn query(&self, key: &str) -> &str {
        self.query.get(key).map_or("", String::as_str)
    }

    fn limit(&self, default: u32, max: u32) -> Result<u32> {
        match self.query.get("limit") {
            Some(value) => value
                .parse()
                .ok()
                .filter(|value| (1..=max).contains(value))
                .ok_or(INVALID),
            None => Ok(default),
        }
    }

    fn revision(&self) -> Result<u64> {
        match self.body.get("expected_revision") {
            Some(value) => value.as_u64().ok_or(INVALID),
            None if self.query.contains_key("expected_revision") => {
                self.query("expected_revision").parse().map_err(|_| INVALID)
            }
            None => Err(INVALID),
        }
    }

    fn optional_revision(&self) -> Result<u64> {
        if self.body.get("expected_revision").is_some()
            || self.query.contains_key("expected_revision")
        {
            self.revision()
        } else {
            Ok(0)
        }
    }
}

async fn memory_overview(
    state: &AppState,
    fields: &Fields,
    view: &str,
    actor: ActorContext,
) -> Result<Value> {
    let limit = fields.limit(50, 100)?;
    let request = |view: &str, cursor: &str| MemoryOverviewRequest {
        view: view.into(),
        limit,
        actor: Some(actor.clone()),
        cursor: cursor.into(),
    };
    let brain = &state.clients.brain;
    let query = &fields.query;
    let cursor = page_cursor(query, &["cursor", "after"])?;
    if view == "summary" {
        // MemoryService returns a next cursor per page only on the individual views.
        let cursor_for = |key| {
            if (query.contains_key("cursor") || query.contains_key("after"))
                && query.contains_key(key)
            {
                return Err(INVALID);
            }
            if query.contains_key(key) {
                page_cursor(query, &[key])
            } else {
                Ok(cursor)
            }
        };
        let (facts, proposals) = tokio::try_join!(
            brain.list_memory_overview(request("facts", cursor_for("facts_after")?)),
            brain.list_memory_overview(request("proposals", cursor_for("proposals_after")?)),
        )?;
        let mut facts = projection(facts.payload_json)?;
        let mut proposals = projection(proposals.payload_json)?;
        if !facts["facts"].is_array() || !proposals["proposals"].is_array() {
            return Err(BAD_UPSTREAM);
        }
        return Ok(json!({
            "facts": facts["facts"].take(),
            "facts_next": facts["next"].take(),
            "proposals": proposals["proposals"].take(),
            "proposals_next": proposals["next"].take(),
        }));
    }
    let alias = format!("{view}_after");
    let req = request(view, page_cursor(query, &["cursor", "after", &alias])?);
    let page = match view {
        "voice_mine" | "voice_consents" => state.clients.memory.list_memory_overview(req).await?,
        _ => brain.list_memory_overview(req).await?,
    };
    projection(page.payload_json)
}

fn memory_mutation<'a>(fields: &'a Fields, operation: &str) -> Result<(&'a str, Value)> {
    let body = &fields.body;
    let mut target = fields.id();
    let mut payload = body.clone();
    match operation {
        "voice_consent_grant" | "voice_consent_revoke" => {
            allowed_fields(
                body,
                &[
                    "expected_revision",
                    "expected_profile_revision",
                    "expected_link_revision",
                ],
            )?;
            for key in [
                "expected_revision",
                "expected_profile_revision",
                "expected_link_revision",
            ] {
                if body.get(key).is_some() {
                    unsigned(body, key)?;
                }
            }
        }
        "voice_verification" => {
            allowed_fields(body, &["expected_revision", "outcome", "room_id"])?;
            if body.get("room_id").is_some() {
                memory_id(string(body, "room_id")?)?;
                if body.get("outcome").is_some() {
                    return Err(INVALID);
                }
            } else if !matches!(string(body, "outcome")?, "passed" | "failed") {
                return Err(INVALID);
            }
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
        }
        "confirm_voice_profile" => {
            allowed_fields(
                body,
                &[
                    "profile_id",
                    "fingerprint_ids",
                    "expected_revision",
                    "name",
                    "request_id",
                ],
            )?;
            if let Some(profile) = optional_string(body, "profile_id")?.filter(|id| !id.is_empty())
            {
                target = memory_id(profile)?;
            }
            voice_ids(body, "fingerprint_ids")?;
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
            if body.get("name").is_some() {
                bounded_text(body, "name", 64)?;
            }
            if body.get("request_id").is_some() {
                request_id(string(body, "request_id")?)?;
            }
        }
        "delete_voice_profile" => {
            memory_id(target)?;
            allowed_fields(body, &["expected_revision"])?;
            if body.get("expected_revision").is_some()
                || fields.query.contains_key("expected_revision")
            {
                payload["expected_revision"] = json!(fields.revision()?);
            }
        }
        "correct_voice_group" => {
            allowed_fields(
                body,
                &[
                    "group_id",
                    "action",
                    "profile_ids",
                    "fingerprint_ids",
                    "expected_revision",
                    "expected_revisions",
                ],
            )?;
            if let Some(group) = optional_string(body, "group_id")?.filter(|id| !id.is_empty()) {
                target = memory_id(group)?;
            }
            let action = optional_string(body, "action")?.unwrap_or_default();
            if !matches!(action, "" | "merge" | "split" | "unknown")
                || (!target.is_empty() && matches!(action, "split" | "unknown"))
            {
                return Err(INVALID);
            }
            match (body.get("profile_ids"), body.get("fingerprint_ids")) {
                (Some(_), None) => voice_ids(body, "profile_ids")?,
                (None, Some(_)) => voice_ids(body, "fingerprint_ids")?,
                _ => return Err(INVALID),
            }
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
            if let Some(revisions) = body.get("expected_revisions") {
                for (id, value) in revisions.as_object().ok_or(INVALID)? {
                    memory_id(id)?;
                    value.as_u64().ok_or(INVALID)?;
                }
            }
        }
        _ => {}
    }
    Ok((target, payload))
}

async fn execute(
    state: &AppState,
    policy: Policy,
    fields: &Fields,
    proof: &Presented,
    lease: &Lease,
) -> Result<Response> {
    use Op::*;
    let brain = &state.clients.brain;
    let audio = &state.clients.audio;
    let memory = &state.clients.memory;
    let id = fields.id();
    let body = &fields.body;
    let actor = || Some(lease.actor());
    let actor_request = || ActorRequest { actor: actor() };
    let task_query = match policy.op {
        Lists => Some(&["scope"][..]),
        Reminders => Some(&["scope", "include_completed"][..]),
        ListItems | ListSync | ReminderNotices | ReminderStatus => Some(&[][..]),
        ListMutation(_) | ReminderMutation(_) => Some(&["expected_revision"][..]),
        _ => None,
    };
    if task_query.is_some_and(|allowed| {
        fields
            .query
            .keys()
            .any(|key| !allowed.contains(&key.as_str()))
    }) {
        return Err(INVALID);
    }
    let mut status = StatusCode::OK;
    let value = match policy.op {
        Health | Login => return Err(BAD_UPSTREAM),
        Logout => {
            let revoked = state.auth.logout(proof.clone()).await?;
            let mut response = Json(json!({"ok":true,"session_revoked":revoked})).into_response();
            response.headers_mut().insert(
                header::SET_COOKIE,
                auth::session_cookie("", 0, state.config.allow_insecure_cookies)?,
            );
            return Ok(response);
        }
        Whoami => lease.whoami(),
        Model => {
            if !fields.query.is_empty() {
                return Err(NOT_FOUND);
            }
            return private_asset(state, &["casa.glb"], "model/gltf-binary").await;
        }
        Plan => {
            if !fields.query.is_empty() {
                return Err(NOT_FOUND);
            }
            match private_asset(state, &["house-plan.js"], "text/javascript; charset=utf-8").await {
                Err(ApiError(StatusCode::NOT_FOUND, _)) => {
                    return private_asset(state, &["house-plan.json"], "application/json").await;
                }
                result => return result,
            }
        }
        Catalog => home_catalog(
            brain
                .get_home_catalog(GetHomeCatalogRequest { actor: actor() })
                .await?,
        )?,
        Microphones => microphones(audio.get_microphones(Empty {}).await?)?,
        HomeAction => home_receipt(
            brain
                .execute_home_action(home_intent(body, actor())?)
                .await?,
        ),
        HomeReceipt => home_receipt(
            brain
                .get_home_action_receipt(GetHomeActionReceiptRequest {
                    request_id: request_id(fields.query("request_id"))?.into(),
                    actor: actor(),
                })
                .await?,
        ),
        Routines => {
            let enabled_only = optional_bool(&fields.query, "enabled_only")?;
            let snapshot = brain
                .list_routines(ListRoutinesRequest {
                    enabled_only,
                    actor: actor(),
                })
                .await?;
            let rows = snapshot
                .routines
                .into_iter()
                .map(routine)
                .collect::<Result<Vec<_>>>()?;
            json!(rows)
        }
        Routine => routine(
            brain
                .get_routine(GetRoutineRequest {
                    routine_id: id.into(),
                    actor: actor(),
                })
                .await?,
        )?,
        RoutinePreview => {
            let response = brain
                .preview_routine(PreviewRoutineRequest {
                    spec_json: body.to_string(),
                    actor: actor(),
                })
                .await?;
            if response.preview_json.is_empty() {
                json!({"valid":response.valid,"errors":response.errors})
            } else {
                projection(response.preview_json)?
            }
        }
        RoutineUpsert | RoutineEnabled(_) => {
            let revision = if id.is_empty() { 0 } else { fields.revision()? };
            let mut spec = body.clone();
            if let RoutineEnabled(enabled) = policy.op {
                let old = brain
                    .get_routine(GetRoutineRequest {
                        routine_id: id.into(),
                        actor: actor(),
                    })
                    .await?;
                if old.revision != revision {
                    return Err(ApiError(StatusCode::CONFLICT, "routine_revision_conflict"));
                }
                spec = projection(old.spec_json)?;
                spec.as_object_mut()
                    .ok_or(BAD_UPSTREAM)?
                    .insert("enabled".into(), json!(enabled));
                state.auth.validate(proof, policy.permissions).await?;
            } else if let Some(object) = spec.as_object_mut() {
                object.remove("expected_revision");
            }
            routine(
                brain
                    .upsert_routine(UpsertRoutineRequest {
                        routine_id: id.into(),
                        expected_revision: revision,
                        spec_json: spec.to_string(),
                        actor: actor(),
                    })
                    .await?,
            )?
        }
        RoutineDelete => {
            brain
                .delete_routine(DeleteRoutineRequest {
                    routine_id: id.into(),
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            json!({"success":true})
        }
        RoutineExecute => {
            let receipt = brain
                .execute_routine(ExecuteRoutineRequest {
                    routine_id: id.into(),
                    request_id: request_id(string(body, "request_id")?)?.into(),
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            json!({"run_id":receipt.run_id,"routine_id":receipt.routine_id,"status":receipt.status,
                "accepted_at_unix_ms":receipt.accepted_at_unix_ms})
        }
        Lists | ListItems => {
            let scope = if matches!(policy.op, Lists) {
                task_scope(fields.query.get("scope").map(String::as_str), lease)?
            } else {
                None
            };
            let rows = brain.list_household_lists(actor_request()).await?.lists;
            let user = lease.actor().user_id;
            if matches!(policy.op, ListItems) {
                let row = rows
                    .into_iter()
                    .find(|row| {
                        row.list_id == id
                            && (row.owner_user_id.is_empty() || row.owner_user_id == user)
                    })
                    .ok_or(NOT_FOUND)?;
                list_items(&row)?
            } else {
                json!(
                    rows.into_iter()
                        .filter(|row| scope_matches(&row.owner_user_id, &user, scope))
                        .map(list_record)
                        .collect::<Vec<_>>()
                )
            }
        }
        ListMutation(operation) => {
            let allowed = match operation {
                "create_list" => &["name", "shared", "expected_revision"][..],
                "add_item" => &["text", "expected_revision"],
                _ => &["expected_revision"],
            };
            allowed_fields(body, allowed)?;
            if operation == "create_list" {
                bounded_text(body, "name", 200)?;
                let shared = body
                    .get("shared")
                    .map(|_| boolean(body, "shared"))
                    .transpose()?
                    .unwrap_or(false);
                if !shared {
                    lease.require(&[TasksPersonal])?;
                }
            } else if operation == "add_item" {
                bounded_text(body, "text", 500)?;
            }
            let mut payload = body.clone();
            if let Some(item) = fields.path.get("item_id") {
                payload["item_id"] = json!(item);
            }
            let record = brain
                .mutate_household_list(HouseholdListMutationRequest {
                    operation: operation.into(),
                    list_id: id.into(),
                    expected_revision: fields.optional_revision()?,
                    payload_json: payload.to_string(),
                    actor: actor(),
                })
                .await?;
            match operation {
                "delete_item" => json!({"status":"deleted","outcome":committed()}),
                "add_item" => {
                    let text = string(body, "text")?;
                    let items = list_items(&record)?;
                    let mut item = items
                        .as_array()
                        .and_then(|rows| rows.iter().rev().find(|row| row["text"] == text))
                        .cloned()
                        .ok_or(BAD_UPSTREAM)?;
                    item["outcome"] = committed();
                    item
                }
                _ => {
                    let mut result = list_record(record);
                    result["outcome"] = committed();
                    result
                }
            }
        }
        ListSync => {
            exact_fields(body, &[])?;
            projection(
                brain
                    .sync_household_lists(actor_request())
                    .await?
                    .payload_json,
            )?
        }
        ReminderNotices => projection(
            brain
                .list_reminder_notices(actor_request())
                .await?
                .payload_json,
        )?,
        ReminderStatus => projection(
            brain
                .get_reminder_status(actor_request())
                .await?
                .payload_json,
        )?,
        Reminders => {
            let scope = task_scope(fields.query.get("scope").map(String::as_str), lease)?;
            let user = lease.actor().user_id;
            let rows = brain
                .list_reminders(ListRemindersRequest {
                    include_personal: !user.is_empty() && scope != Some(false),
                    include_completed: optional_bool(&fields.query, "include_completed")?
                        .unwrap_or(false),
                    actor: actor(),
                })
                .await?
                .reminders;
            json!(
                rows.into_iter()
                    .filter(|row| scope_matches(&row.owner_user_id, &user, scope))
                    .map(reminder)
                    .collect::<Result<Vec<_>>>()?
            )
        }
        ReminderMutation(operation) => {
            let fields_allowed = match operation {
                "create" => &[
                    "text",
                    "trigger_at",
                    "scope",
                    "recurrence",
                    "ha_actions",
                    "delivery_room",
                    "expected_revision",
                ][..],
                "update" => &[
                    "text",
                    "trigger_at",
                    "scope",
                    "ha_actions",
                    "delivery_room",
                    "expected_revision",
                ],
                "dismiss_notice" => &["reminder_id", "trigger_at", "expected_revision"],
                _ => &["expected_revision"],
            };
            allowed_fields(body, fields_allowed)?;
            let mut payload = body.clone();
            if matches!(operation, "create" | "update") {
                if operation == "create" || body.get("text").is_some() {
                    bounded_text(body, "text", 500)?;
                }
                let scope = optional_string(body, "scope")?.unwrap_or("private");
                match scope {
                    "private" => lease.require(&[TasksPersonal])?,
                    "shared" => {}
                    _ => return Err(INVALID),
                }
                if operation == "update" {
                    payload["scope"] = json!(scope);
                    optional_string(body, "delivery_room")?;
                    if body
                        .get("ha_actions")
                        .is_some_and(|value| !value.is_array() && !value.is_null())
                    {
                        return Err(INVALID);
                    }
                }
            }
            if (matches!(operation, "create" | "dismiss_notice")
                || (operation == "update" && body.get("trigger_at").is_some()))
                && !body
                    .get("trigger_at")
                    .and_then(Value::as_f64)
                    .is_some_and(|time| time.is_finite() && time > 0.0)
            {
                return Err(INVALID);
            }
            let record = brain
                .mutate_reminder(ReminderMutationRequest {
                    operation: operation.into(),
                    reminder_id: id.into(),
                    expected_revision: if operation == "update" {
                        fields.revision()?
                    } else {
                        fields.optional_revision()?
                    },
                    payload_json: payload.to_string(),
                    actor: actor(),
                })
                .await?;
            if matches!(operation, "create" | "update") {
                let mut row = reminder(record)?;
                row["outcome"] = committed();
                row
            } else {
                let result = if operation == "cancel" {
                    "cancelled"
                } else {
                    &record.status
                };
                json!({"status":result,"outcome":committed()})
            }
        }
        Memory(view) => memory_overview(state, fields, view, lease.actor()).await?,
        MemoryMutation(operation) => {
            let (target, payload) = memory_mutation(fields, operation)?;
            let request = MemoryMutationRequest {
                operation: operation.into(),
                target_id: target.into(),
                payload_json: payload.to_string(),
                actor: actor(),
            };
            match operation {
                "delete_voice_profile" => {
                    let memory = memory.clone();
                    let live = state.live.clone();
                    // Complete the erasure fence even if the request is disconnected.
                    tokio::spawn(async move {
                        let doc =
                            projection(memory.mutate_memory_item(request).await?.payload_json)?;
                        if doc["deleted"] != true {
                            return Err(BAD_UPSTREAM);
                        }
                        live.fence();
                        Ok(doc)
                    })
                    .await
                    .map_err(|_| BAD_UPSTREAM)??
                }
                "voice_consent_grant"
                | "voice_consent_revoke"
                | "voice_verification"
                | "confirm_voice_profile"
                | "correct_voice_group" => {
                    projection(memory.mutate_memory_item(request).await?.payload_json)?
                }
                _ => projection(brain.mutate_memory_item(request).await?.payload_json)?,
            }
        }
        DeleteFact => {
            let receipt = memory
                .delete_fact(DeleteFactRequest {
                    fact_id: id.into(),
                    actor: actor(),
                })
                .await?;
            if !receipt.tombstone_recorded {
                return Err(BAD_UPSTREAM);
            }
            state.live.fence();
            json!({"deleted":receipt.facts_deleted > 0})
        }
        VoiceSwitch => voice_switch(brain.get_voice_global_switch(actor_request()).await?),
        VoiceSwitchWrite => {
            exact_fields(body, &["enabled", "expected_revision"])?;
            voice_switch(
                brain
                    .set_voice_global_switch(SetVoiceGlobalSwitchRequest {
                        enabled: boolean(body, "enabled")?,
                        expected_revision: fields.revision()?,
                        actor: actor(),
                    })
                    .await?,
            )
        }
        VoiceProfiles => {
            let include_person = optional_bool(&fields.query, "include_person")?.unwrap_or(false);
            if include_person {
                lease.require(&[AdminManage])?;
            } else if fields.query.contains_key("person_id") {
                return Err(INVALID);
            }
            let cursor = page_cursor(&fields.query, &["cursor"])?;
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceProfiles)?;
            let page = memory
                .list_voice_profiles(ListVoiceProfilesRequest {
                    include_person,
                    actor: actor(),
                    cursor: cursor.into(),
                })
                .await?;
            let rows = page
                .profiles
                .into_iter()
                .filter(|row| {
                    !include_person
                        || fields
                            .query
                            .get("person_id")
                            .is_none_or(|id| row.linked_person_id == *id)
                })
                .map(|row| voice_profile(row, include_person))
                .collect::<Vec<_>>();
            state.live.projection(
                connection,
                ws::Channel::VoiceProfiles,
                json!({
                    "rows": rows,
                    "next_cursor": (!page.next_cursor.is_empty()).then_some(page.next_cursor),
                }),
            )?
        }
        VoiceLink => {
            let row = memory
                .link_voice_profile_to_person(LinkVoiceProfileRequest {
                    profile_id: id.into(),
                    person_id: optional_string(body, "user_id")?.unwrap_or("").into(),
                    expected_profile_revision: unsigned(body, "expected_profile_revision")?,
                    expected_link_revision: unsigned(body, "expected_link_revision")?,
                    actor: actor(),
                })
                .await?;
            json!({"ok":true,"profile_id":row.profile_id,"link_revision":row.link_revision})
        }
        VoiceReview(transcript) => {
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceReview)?;
            let doc = projection(
                brain
                    .get_voice_capture_review(VoiceCaptureReviewRequest {
                        fingerprint_id: id.into(),
                        include_transcript: transcript,
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::VoiceReview, doc)?
        }
        VoiceCaptures => {
            if body.get("text").is_some_and(|value| !value.is_null()) {
                lease.require(&[AdminManage])?;
            }
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::VoiceCaptures)?;
            let query = if fields.query.is_empty() {
                body.clone()
            } else if body == &json!({}) {
                json!(fields.query)
            } else {
                return Err(INVALID);
            };
            let doc = projection(
                brain
                    .query_voice_captures(VoiceCaptureQueryRequest {
                        query_json: query.to_string(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::VoiceCaptures, doc)?
        }
        Capture => capture(audio.get_capture_control(Empty {}).await?),
        CaptureWrite => {
            exact_fields(body, &["muted", "expected_revision"])?;
            let doc = audio
                .set_capture_mute(SetCaptureMuteRequest {
                    muted: boolean(body, "muted")?,
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            capture(doc)
        }
        ForgetCarried => {
            exact_fields(body, &["expected_carried", "no_capture_running"])?;
            if !boolean(body, "no_capture_running")? {
                return Err(INVALID);
            }
            let receipt = audio
                .forget_carried_owners(ForgetCarriedOwnersRequest {
                    expected_carried: unsigned(body, "expected_carried")?
                        .try_into()
                        .map_err(|_| INVALID)?,
                    no_capture_running: true,
                    actor: actor(),
                })
                .await?;
            json!({"status":capture(receipt.status.ok_or(BAD_UPSTREAM)?),"forgotten_capture_owners":receipt.forgotten_capture_owners})
        }
        Ambient => ambient(memory.get_ambient_control(lease.actor()).await?),
        AmbientWrite => {
            exact_fields(body, &["paused", "expected_revision"])?;
            let value = memory
                .set_ambient_pause(SetAmbientPauseRequest {
                    paused: boolean(body, "paused")?,
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            ambient(value)
        }
        EvaluationGrants => {
            require_evaluator(lease)?;
            lease.evaluation_grants()
        }
        EvaluationSample => {
            require_evaluator(lease)?;
            let modality = fields.query("modality");
            if fields.query.len() != 1 || !["audio_preview", "transcript_full"].contains(&modality)
            {
                return Err(INVALID);
            }
            let sources = lease.evaluation_sources(id, modality)?;
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::Evaluation)?;
            let doc = projection(
                brain
                    .get_evaluation_sample(EvaluationSampleRequest {
                        fingerprint_id: id.into(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            let current = state.auth.validate(proof, policy.permissions).await?;
            current.evaluation_sources(id, modality)?;
            state.live.projection(
                connection,
                ws::Channel::Evaluation,
                verified_sample(doc, id, modality, sources)?,
            )?
        }
        Zones => json!(
            audio
                .get_zones(Empty {})
                .await?
                .zones
                .into_iter()
                .map(zone)
                .collect::<Vec<_>>()
        ),
        Conversations => {
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::Conversations)?;
            if fields
                .query
                .keys()
                .any(|key| !matches!(key.as_str(), "limit" | "zone"))
            {
                return Err(ApiError(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "conversation_filter_unavailable",
                ));
            }
            let rows = projection(
                brain
                    .list_conversations(ListConversationsRequest {
                        limit: fields.limit(100, 200)?,
                        room_id: fields.query("zone").into(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::Conversations, rows)?
        }
        HaEntities => {
            let output = brain
                .get_home_catalog(GetHomeCatalogRequest { actor: actor() })
                .await?;
            let domain = fields.query("domain");
            let entities: Vec<_> = if output.stale {
                vec![]
            } else {
                output
                    .entities
                    .into_iter()
                    .filter(|row| domain.is_empty() || row.domain == domain)
                    .map(entity)
                    .collect()
            };
            json!(entities)
        }
        HaActions => json!(
            brain
                .list_ha_actions(ListHaActionsRequest {
                    limit: fields.limit(50, 200)?,
                    actor: actor()
                })
                .await?
                .receipts
                .into_iter()
                .map(home_receipt)
                .collect::<Vec<_>>()
        ),
        Llm => json!(
            brain
                .get_llm_endpoints(lease.actor())
                .await?
                .endpoints
                .into_iter()
                .map(llm_endpoint)
                .collect::<Vec<_>>()
        ),
        Cooldown => {
            brain
                .clear_llm_cooldown(ClearLlmCooldownRequest {
                    endpoint_id: id.into(),
                    actor: actor(),
                })
                .await?;
            json!({"ok":true,"endpoint_id":id})
        }
        Users => state.auth.users().await?,
        UserWrite => {
            state
                .auth
                .write_user(
                    proof.clone(),
                    (!id.is_empty()).then(|| id.to_owned()),
                    body.clone(),
                )
                .await?
        }
        UserRetire => {
            if state.auth.retire(proof.clone(), id.into()).await? {
                let current = state.auth.validate(proof, policy.permissions).await?;
                let receipt = memory
                    .purge_retired_user(PurgeRetiredUserRequest {
                        user_id: id.into(),
                        actor: Some(current.actor()),
                    })
                    .await;
                if !receipt.is_ok_and(|value| value.tombstone_recorded) {
                    return Ok((StatusCode::SERVICE_UNAVAILABLE,Json(json!({
                        "detail":{"code":"retired_user_purge_pending","credentials_retired":true}
                    }))).into_response());
                }
                state.live.fence();
                state.auth.finish_purge(id.into()).await?;
            }
            json!({"ok":true,"user_id":id,"credentials_retired":true})
        }
        Alerts => {
            let filter = fields.query("status");
            let alerts: Vec<_> = brain
                .list_alerts(actor_request())
                .await?
                .alerts
                .into_iter()
                .filter(|row| match filter {
                    "active" => !row.acknowledged,
                    "acked" => row.acknowledged,
                    _ => true,
                })
                .map(alert)
                .collect();
            let output = brain
                .get_alert_zones(actor_request())
                .await
                .map(alert_zones)
                .unwrap_or_else(|_| json!({"available":false}));
            json!({"alerts":alerts,"available":true,"output":output,"producers":[]})
        }
        AlertZones => alert_zones(brain.get_alert_zones(actor_request()).await?),
        AlertZonesWrite => {
            let zones = body
                .get("zones")
                .and_then(Value::as_array)
                .ok_or(INVALID)?
                .iter()
                .map(|row| row.as_str().map(str::to_owned).ok_or(INVALID))
                .collect::<Result<Vec<_>>>()?;
            alert_zones(
                brain
                    .set_alert_zones(SetAlertZonesRequest {
                        expected_revision: fields.revision()?,
                        zone_ids: zones,
                        actor: actor(),
                    })
                    .await?,
            )
        }
        AlertAck | AlertDelete => {
            let request = AlertMutationRequest {
                alert_id: id.into(),
                expected_revision: fields.optional_revision()?,
                actor: actor(),
            };
            if matches!(policy.op, AlertAck) {
                brain.ack_alert(request).await?;
                json!({"ok":true,"alert_id":id,"acked":true})
            } else {
                brain.delete_alert(request).await?;
                json!({"ok":true,"alert_id":id,"dismissed":true})
            }
        }
        Gpus => json!(state.system.gpus().await),
        Services => json!(state.system.services().await),
        Restart => {
            if !crate::sys::restartable(id) {
                return Err(MonitorError::Denied.into());
            }
            state.system.restart(id).await?;
            json!({"ok":true,"service":id,"command_completed":true})
        }
        Cloud => cloud(brain.get_cloud_consent(lease.actor()).await?, false),
        CloudWithdraw => cloud(
            brain
                .withdraw_cloud_consent(WithdrawCloudConsentRequest { actor: actor() })
                .await?,
            true,
        ),
        Credentials => {
            if fields
                .query
                .keys()
                .any(|key| !["limit", "after"].contains(&key.as_str()))
            {
                return Err(INVALID);
            }
            state
                .auth
                .credential_catalog(fields.query("after").into(), fields.limit(50, 50)?)
                .await?
        }
        CredentialMutation(action) => {
            if !fields.query.is_empty() {
                return Err(INVALID);
            }
            let allowed = match action {
                "issue" => &[
                    "request_id",
                    "subject",
                    "role",
                    "expires_at",
                    "user_id",
                    "selections",
                ][..],
                "revise" => &[
                    "request_id",
                    "expected_revision",
                    "role",
                    "expires_at",
                    "user_id",
                    "selections",
                ],
                _ => &["request_id", "expected_revision"],
            };
            allowed_fields(body, allowed)?;
            state
                .auth
                .change_credential(
                    proof.clone(),
                    action,
                    (!id.is_empty()).then(|| id.to_owned()),
                    body.clone(),
                )
                .await?
        }
        CredentialAttempt => {
            state
                .auth
                .credential_attempt(proof.clone(), id.into())
                .await?
        }
        Unavailable(reason) => {
            return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, reason));
        }
    };
    Ok((status, Json(value)).into_response())
}

pub(crate) fn no_store(response: &mut Response) {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
}

async fn private_asset(
    state: &AppState,
    parts: &[&str],
    media_type: &'static str,
) -> Result<Response> {
    let root = state.config.private_dir.as_ref().ok_or(NOT_FOUND)?;
    asset(
        root.clone(),
        parts.iter().map(|part| (*part).into()).collect(),
        media_type,
    )
    .await
}

async fn index(State(state): State<AppState>) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    asset(root, vec!["index.html".into()], "text/html; charset=utf-8").await
}

async fn favicon(State(state): State<AppState>) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    asset(root, vec!["favicon.ico".into()], "image/x-icon").await
}

async fn public_asset(State(state): State<AppState>, Path(file): Path<String>) -> Result<Response> {
    let root = state.config.static_dir.ok_or(NOT_FOUND)?;
    let path = FsPath::new(&file);
    let parts: Vec<_> = file.split('/').collect();
    if parts
        .iter()
        .any(|part| part.is_empty() || part.starts_with('.') || part.contains(['\\', '%']))
        || parts
            .last()
            .is_some_and(|name| name.starts_with("house-plan") || name.starts_with("casa."))
    {
        return Err(NOT_FOUND);
    }
    let media_type = match path.extension().and_then(|extension| extension.to_str()) {
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("webp") => "image/webp",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("woff2") => "font/woff2",
        _ => return Err(NOT_FOUND),
    };
    let parts = std::iter::once("assets")
        .chain(parts)
        .map(str::to_owned)
        .collect();
    asset(root, parts, media_type).await
}

async fn asset(root: PathBuf, parts: Vec<String>, media_type: &'static str) -> Result<Response> {
    let file = tokio::task::spawn_blocking(move || {
        let (leaf, parents) = parts.split_last()?;
        let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let mut directory = open(&root, flags | OFlags::DIRECTORY, Mode::empty()).ok()?;
        for name in parents {
            directory = openat(&directory, name, flags | OFlags::DIRECTORY, Mode::empty()).ok()?;
        }
        let file =
            File::from(openat(directory, leaf, flags | OFlags::NONBLOCK, Mode::empty()).ok()?);
        let metadata = file.metadata().ok()?;
        metadata.is_file().then_some((file, metadata.len()))
    })
    .await
    .map_err(|_| NOT_FOUND)?
    .ok_or(NOT_FOUND)?;
    let headers = [
        (header::CONTENT_TYPE, HeaderValue::from_static(media_type)),
        (header::CONTENT_LENGTH, HeaderValue::from(file.1)),
        (
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ),
    ];
    let stream = ReaderStream::new(tokio::fs::File::from_std(file.0));
    Ok((headers, Body::from_stream(stream)).into_response())
}
