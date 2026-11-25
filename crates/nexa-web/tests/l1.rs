#![allow(clippy::unwrap_used)]

mod support;

use std::{
    collections::HashSet,
    convert::Infallible,
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    sync::Arc,
};

use axum::{
    body::{Body, Bytes, to_bytes},
    http::{Request, StatusCode, header},
};
use nexa_proto::v1::{Permission, Role};
use nexa_web::{
    EVALUATOR_TOKEN, HOME_ID, HOME_TOKEN, OPERATOR_TOKEN, TestMode,
    auth::{AuthStore, Origin, permission_name, permissions, sha256},
};
use rusqlite::Connection;
use serde_json::json;
use tokio::sync::{Notify, oneshot};
use tower::ServiceExt;

use support::{ORIGIN, call, document};

#[test]
fn sha256_and_role_matrix_match_authority() {
    assert_eq!(
        sha256("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let operator = permissions(Role::Operator, false);
    let home = permissions(Role::Home, false);
    let personal = permissions(Role::Home, true);
    let evaluator = permissions(Role::Evaluator, false);
    assert_eq!(operator.len(), 17);
    assert_eq!(home.len(), 5);
    assert_eq!(personal.len(), 9);
    assert_eq!(evaluator, vec![Permission::EvaluationRead]);
    assert!(!operator.contains(&Permission::TasksPersonal));
    assert!(personal.contains(&Permission::MemoryOwnVoice));
    let names: HashSet<_> = operator
        .into_iter()
        .chain(personal)
        .chain(evaluator)
        .map(permission_name)
        .collect();
    assert_eq!(names.len(), 21);
}

#[tokio::test]
async fn login_hashes_secrets_and_cookies_default_to_secure() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let path = fixture.database_path();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert!(AuthStore::open(&path).is_err());
    let db = Connection::open(&path).unwrap();
    let mode: String = db
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
    let stored: String = db
        .query_row(
            "SELECT digest FROM credentials WHERE subject='home'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stored, sha256(HOME_TOKEN));
    let response = call(
        &app,
        "POST",
        "/api/admin/auth/login",
        None,
        Some(ORIGIN),
        Body::from(json!({"token":HOME_TOKEN}).to_string()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let cookie = response.headers()[header::SET_COOKIE].to_str().unwrap();
    for flag in ["HttpOnly", "SameSite=Strict", "Path=/", "Secure"] {
        assert!(cookie.split("; ").any(|actual| actual == flag), "{cookie}");
    }
    let pair = cookie.split(';').next().unwrap();
    let (_, secret) = pair.split_once('=').unwrap();
    let stored: String = db
        .query_row("SELECT digest FROM sessions", [], |row| row.get(0))
        .unwrap();
    assert_eq!(stored, sha256(secret));
    let who = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/admin/auth/whoami")
                .header(header::COOKIE, pair)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = document(who).await;
    assert_eq!(body["role"], "home");
    assert_eq!(body["personal_scope"], true);
    assert!(!body.to_string().contains(HOME_ID));
    let denied = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/admin/auth/whoami")
                .header(header::COOKIE, pair)
                .header(header::AUTHORIZATION, "Bearer bad")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
    let logout = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/admin/auth/logout")
                .header(header::COOKIE, pair)
                .header(header::ORIGIN, ORIGIN)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(logout.status(), StatusCode::OK);
    assert!(
        logout.headers()[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .contains("Max-Age=0")
    );
    let response = call(
        &app,
        "GET",
        "/api/admin/auth/whoami",
        Some(secret),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let insecure = fixture
        .app(Origin::parse(ORIGIN).unwrap(), true, None)
        .unwrap();
    let response = call(
        &insecure,
        "POST",
        "/api/admin/auth/login",
        None,
        Some(ORIGIN),
        Body::from(json!({"token":OPERATOR_TOKEN}).to_string()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        !response.headers()[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .contains("Secure")
    );
}

#[tokio::test]
async fn origin_enforces_scheme_host_and_port_without_trusting_headers() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
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
            Body::from(json!({"token":HOME_TOKEN}).to_string()),
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
}

#[tokio::test]
async fn private_asset_permissions_symlinks_and_dead_route_inventory() {
    let fixture = TestMode::new().await.unwrap();
    let public = tempfile::tempdir().unwrap();
    fs::create_dir(public.path().join("assets")).unwrap();
    fs::write(public.path().join("index.html"), "<html>fixture SPA</html>").unwrap();
    fs::write(public.path().join("assets/fixture.js"), "export {};").unwrap();
    symlink(
        fixture.private_dir().join("house-plan.js"),
        public.path().join("assets/leaked.js"),
    )
    .unwrap();
    let app = fixture
        .app(
            Origin::parse(ORIGIN).unwrap(),
            false,
            Some(public.path().to_path_buf()),
        )
        .unwrap();
    for (path, media_type) in [
        ("/api/home/model", "model/gltf-binary"),
        ("/api/home/plan", "text/javascript; charset=utf-8"),
    ] {
        for (token, status) in [
            (None, StatusCode::UNAUTHORIZED),
            (Some(EVALUATOR_TOKEN), StatusCode::FORBIDDEN),
            (Some(HOME_TOKEN), StatusCode::OK),
        ] {
            let response = call(&app, "GET", path, token, None, Body::empty()).await;
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
            if status == StatusCode::OK {
                assert_eq!(response.headers()[header::CONTENT_TYPE], media_type);
            }
        }
    }
    for path in ["/", "/obs", "/assets/fixture.js"] {
        assert_eq!(
            call(&app, "GET", path, None, None, Body::empty())
                .await
                .status(),
            StatusCode::OK
        );
    }
    for path in [
        "/ws",
        "/api/metrics",
        "/api/subsystems",
        "/api/failures",
        "/api/presence",
        "/api/entities",
        "/api/services",
        "/api/templates",
        "/api/integrations",
        "/api/integrations/energy",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/whoami",
        "/api/credentials",
        "/api/home/model/../plan",
        "/api/home/plan/%2e%2e%2f",
        "/api/home/model%2f../plan",
        "/api/home/model?path=../../auth.db",
        "/assets/%2e%2e/auth.db",
        "/assets/leaked.js",
    ] {
        let response = call(&app, "GET", path, Some(OPERATOR_TOKEN), None, Body::empty()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
    for path in ["/api/auth/login", "/api/auth/logout"] {
        assert_eq!(
            call(
                &app,
                "POST",
                path,
                Some(OPERATOR_TOKEN),
                Some(ORIGIN),
                Body::empty()
            )
            .await
            .status(),
            StatusCode::NOT_FOUND,
        );
    }
    let plan = fixture.private_dir().join("house-plan.js");
    fs::remove_file(&plan).unwrap();
    fs::write(
        fixture.private_dir().join("house-plan.json"),
        "{\"rooms\":[]}",
    )
    .unwrap();
    let response = call(
        &app,
        "GET",
        "/api/home/plan",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    assert_eq!(document(response).await, json!({"rooms":[]}));
    fs::remove_file(fixture.private_dir().join("house-plan.json")).unwrap();
    symlink(public.path().join("assets/fixture.js"), &plan).unwrap();
    let response = call(
        &app,
        "GET",
        "/api/home/plan",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");

    let model = call(
        &app,
        "GET",
        "/api/home/model",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(
        &to_bytes(model.into_body(), 1024).await.unwrap()[..4],
        b"glTF"
    );
}

#[tokio::test]
async fn revocation_during_http_body_read_prevents_dispatch() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let reading = Arc::new(Notify::new());
    let (send, recv) = oneshot::channel();
    let body = Body::from_stream(futures_util::stream::once({
        let reading = reading.clone();
        async move {
            reading.notify_one();
            let bytes = recv.await.unwrap();
            Ok::<Bytes, Infallible>(bytes)
        }
    }));
    let pending = tokio::spawn({
        let app = app.clone();
        async move {
            call(
                &app,
                "POST",
                "/api/home/actions",
                Some(HOME_TOKEN),
                Some(ORIGIN),
                body,
            )
            .await
        }
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), reading.notified())
        .await
        .unwrap();
    let retired = call(
        &app,
        "DELETE",
        &format!("/api/admin/users/{HOME_ID}"),
        Some(OPERATOR_TOKEN),
        Some(ORIGIN),
        Body::empty(),
    )
    .await;
    assert_eq!(retired.status(), StatusCode::OK);
    send.send(Bytes::from_static(
        br#"{"entity_id":"light.fixture","operation":"turn_on","request_id":"action"}"#,
    ))
    .unwrap();
    assert_eq!(pending.await.unwrap().status(), StatusCode::UNAUTHORIZED);
    let response = call(
        &app,
        "GET",
        "/api/home/catalog",
        Some(HOME_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn credential_issue_replay_revision_and_revocation() {
    let fixture = TestMode::new().await.unwrap();
    let app = fixture
        .app(Origin::parse(ORIGIN).unwrap(), false, None)
        .unwrap();
    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 3600;
    let request_id = uuid::Uuid::new_v4().to_string();
    let command = json!({"request_id":request_id,"subject":"household","role":"home","expires_at":expires_at});
    let mutate = |method: &'static str, route: String, body: serde_json::Value| {
        let app = app.clone();
        async move {
            call(
                &app,
                method,
                &route,
                Some(OPERATOR_TOKEN),
                Some(ORIGIN),
                Body::from(body.to_string()),
            )
            .await
        }
    };
    let issue = mutate(
        "POST",
        "/api/admin/auth/credentials".into(),
        command.clone(),
    )
    .await;
    assert_eq!(issue.status(), StatusCode::OK);
    let issued = document(issue).await;
    assert_eq!(issued["receipt"]["credential_id"], request_id);
    let token = issued["token"].as_str().unwrap();
    let replay =
        document(mutate("POST", "/api/admin/auth/credentials".into(), command).await).await;
    assert_eq!(replay["receipt"], issued["receipt"]);
    assert_eq!(replay["applied"], false);
    assert!(replay.get("token").is_none());
    let attempt = call(
        &app,
        "GET",
        &format!("/api/admin/auth/credential-attempts/{request_id}"),
        Some(OPERATOR_TOKEN),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(document(attempt).await, issued["receipt"]);

    let session = call(
        &app,
        "POST",
        "/api/admin/auth/login",
        None,
        Some(ORIGIN),
        Body::from(json!({"token":token}).to_string()),
    )
    .await;
    let cookie = session.headers()[header::SET_COOKIE]
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap();
    let path = format!("/api/admin/auth/credentials/{request_id}");
    let revised = mutate(
        "PUT",
        path.clone(),
        json!({
            "request_id":uuid::Uuid::new_v4().to_string(),"expected_revision":1,
            "role":"home","expires_at":expires_at,"user_id":HOME_ID
        }),
    )
    .await;
    assert_eq!(revised.status(), StatusCode::OK);
    assert_eq!(document(revised).await["receipt"]["result_revision"], 2);
    let old_session = app
        .clone()
        .oneshot(
            Request::get("/api/admin/auth/whoami")
                .header(header::COOKIE, cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(old_session.status(), StatusCode::UNAUTHORIZED);
    let current = call(
        &app,
        "GET",
        "/api/admin/auth/whoami",
        Some(token),
        None,
        Body::empty(),
    )
    .await;
    assert_eq!(document(current).await["personal_scope"], true);

    let stale = mutate(
        "DELETE",
        path.clone(),
        json!({
            "request_id":uuid::Uuid::new_v4().to_string(),"expected_revision":1
        }),
    )
    .await;
    assert_eq!(stale.status(), StatusCode::CONFLICT);
    let revoked = mutate(
        "DELETE",
        path,
        json!({
            "request_id":uuid::Uuid::new_v4().to_string(),"expected_revision":2
        }),
    )
    .await;
    assert_eq!(document(revoked).await["receipt"]["result_revision"], 2);
    assert_eq!(
        call(
            &app,
            "GET",
            "/api/admin/auth/whoami",
            Some(token),
            None,
            Body::empty()
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
}
