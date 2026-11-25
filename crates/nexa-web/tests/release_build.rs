#![allow(clippy::unwrap_used)]

mod support;

#[cfg(not(feature = "test-mode"))]
use std::process::Command;
use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use nexa_proto::v1::{Permission, Role};
use nexa_web::{
    WebConfig,
    auth::{AuthStore, Origin, permission_name, permissions, sha256},
    build_app,
    clients::Clients,
    sys::PlatformMonitor,
};
use serde_json::json;
use tower::ServiceExt;

use support::{ORIGIN, call, document};

#[test]
fn sha256_and_role_matrix_match_authority() {
    use Permission::*;
    assert_eq!(
        sha256("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let operator = [
        HomeRead,
        HomeControl,
        TasksRead,
        TasksWrite,
        RoutinesExecute,
        RoutinesManage,
        DiagnosticsRead,
        ConversationsRead,
        MemoryRead,
        MemoryReview,
        MemoryAssign,
        VoicesRead,
        VoicesReview,
        EvaluationRead,
        AmbientControl,
        AdminManage,
        AuthManage,
    ];
    let home = [
        HomeRead,
        HomeControl,
        TasksRead,
        TasksWrite,
        RoutinesExecute,
    ];
    let own = [TasksPersonal, MemoryOwnRead, MemoryOwnWrite, MemoryOwnVoice];
    for personal in [true, false] {
        assert_eq!(permissions(Role::Operator, personal), operator);
        assert_eq!(permissions(Role::Evaluator, personal), [EvaluationRead]);
        assert!(permissions(Role::Unspecified, personal).is_empty());
    }
    assert_eq!(permissions(Role::Home, false), home);
    assert_eq!(
        permissions(Role::Home, true),
        [home.as_slice(), &own].concat()
    );
    assert_eq!(permission_name(MemoryOwnVoice), "memory:own:voice");
}

#[cfg(not(feature = "test-mode"))]
#[test]
fn default_binary_rejects_test_mode_and_excludes_fixture_tokens() {
    let bin = env!("CARGO_BIN_EXE_nexa-web");
    let output = Command::new(bin).arg("--test-mode").output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unexpected argument '--test-mode'"),
        "{stderr}"
    );
    let strings = Command::new("strings").arg(bin).output().unwrap();
    assert!(strings.status.success());
    let dump = String::from_utf8_lossy(&strings.stdout);
    for token in ["test-operator", "test-home", "test-evaluator"] {
        assert!(
            !dump.contains(token),
            "fixture token in default binary: {token}"
        );
    }
}

#[tokio::test]
async fn origin_enforces_scheme_host_and_port_without_trusting_headers() {
    let dir = tempfile::tempdir().unwrap();
    let app = build_app(
        WebConfig {
            origin: Origin::parse(ORIGIN).unwrap(),
            allow_insecure_cookies: false,
            private_dir: None,
            static_dir: None,
        },
        AuthStore::open(&dir.path().join("auth.db")).unwrap(),
        Clients::uds(dir.path()),
        Arc::new(PlatformMonitor),
    )
    .unwrap();
    for origin in [
        None,
        Some("null"),
        Some("https://localhost:9500"),
        Some("http://localhost"),
        Some("http://localhost:9501"),
        Some("http://LOCALHOST.evil:9500"),
        Some("http://localhost:9500/"),
        Some("http://user@localhost:9500"),
        Some("http://localhost:99999"),
        Some("http://localhost:9500, http://localhost:9500"),
    ] {
        let response = call(
            &app,
            "POST",
            "/api/admin/auth/login",
            None,
            origin,
            Body::from(json!({"token":"invalid"}).to_string()),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN, "{origin:?}");
    }
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/admin/auth/login")
                .header(header::ORIGIN, "http://other:9500")
                .header(header::HOST, "other:9500")
                .header("x-forwarded-host", "other:9500")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let response = call(
        &app,
        "POST",
        "/api/admin/auth/login",
        None,
        Some(ORIGIN),
        Body::from(json!({"token":"invalid"}).to_string()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        document(response).await["detail"],
        "dashboard_session_required"
    );
}
