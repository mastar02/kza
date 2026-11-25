use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, header},
    response::Response,
};
use serde_json::Value;
use tower::ServiceExt;

pub const ORIGIN: &str = "http://localhost:9500";

pub async fn call(
    app: &Router,
    method: &str,
    path: &str,
    token: Option<&str>,
    origin: Option<&str>,
    body: Body,
) -> Response {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(token) = token {
        request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    if let Some(origin) = origin {
        request = request.header(header::ORIGIN, origin);
    }
    app.clone()
        .oneshot(
            request
                .header(header::CONTENT_TYPE, "application/json")
                .body(body)
                .unwrap(),
        )
        .await
        .unwrap()
}

pub async fn document(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
