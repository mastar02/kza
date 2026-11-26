//! Explicit HTTP inventory. Each method is attached to its own authority policy.

mod contracts;
mod projections;
use projections::*;

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

mod admin;
mod assets;
mod home;
mod household;
mod inventory;
mod memory;
mod obs;
mod routines;
mod voices;
use assets::*;
pub(crate) use inventory::router;
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
    Home(home::Action),
    Routines(routines::Action),
    Household(household::Action),
    Memory(memory::Action),
    Voices(voices::Action),
    Obs(obs::Action),
    Admin(admin::Action),
}
#[derive(Clone, Copy)]
struct Policy {
    op: Op,
    permissions: &'static [Permission],
    public: bool,
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

async fn execute(
    state: &AppState,
    policy: Policy,
    fields: &Fields,
    proof: &Presented,
    lease: &Lease,
) -> Result<Response> {
    match policy.op {
        Op::Health | Op::Login => Err(BAD_UPSTREAM),
        Op::Logout => {
            let revoked = state.auth.logout(proof.clone()).await?;
            let mut response = Json(json!({"ok":true,"session_revoked":revoked})).into_response();
            response.headers_mut().insert(
                header::SET_COOKIE,
                auth::session_cookie("", 0, state.config.allow_insecure_cookies)?,
            );
            Ok(response)
        }
        Op::Whoami => Ok(Json(lease.whoami()).into_response()),
        Op::Home(op) => home::execute(op, state, fields, lease).await,
        Op::Routines(op) => routines::execute(op, state, fields, lease, proof, policy).await,
        Op::Household(op) => household::execute(op, state, fields, lease).await,
        Op::Memory(op) => memory::execute(op, state, fields, lease).await,
        Op::Voices(op) => voices::execute(op, state, fields, lease, proof, policy).await,
        Op::Obs(op) => obs::execute(op, state, fields, lease).await,
        Op::Admin(op) => admin::execute(op, state, fields, proof, policy).await,
    }
}

pub(crate) fn no_store(response: &mut Response) {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
}
