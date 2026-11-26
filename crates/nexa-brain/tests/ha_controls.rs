#![allow(clippy::expect_used)]

use nexa_brain::HomeDependencies;
use nexa_config::{ConfigError, LoadedBundle};
use nexa_ha::{
    ActionRequest, ActionStatus, EffectIntent, NotSentReason, Operation, OperationError, PlanError,
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const TOKEN: &str = "ha-test-token";
const CONTROLS: &str = r#"
    [entities]
    "light.group" = ["turn_on", "brightness", "color_temp_kelvin"]
    "light.leaf" = ["turn_on", "brightness", "color_temp_kelvin"]
    "light.unreviewed" = ["turn_on"]
    "light.drifted" = ["turn_on"]
    "scene.reading" = ["activate_scene"]
    "scene.warm" = ["activate_scene"]
    "scene.unreviewed" = ["activate_scene"]

    [groups]
    "light.group" = ["light.leaf"]
    "light.drifted" = ["light.leaf"]

    [scenes]
    "scene.reading" = [
        { entity_id = "light.group", operation = "brightness", brightness_pct = 40 },
    ]
    "scene.warm" = [
        { entity_id = "light.group", operation = "color_temp_kelvin", color_temp_kelvin = 3000, brightness = 120 },
    ]
"#;

fn bundle(controls: &str) -> LoadedBundle {
    let dir = tempfile::tempdir().expect("config directory");
    for (name, contents) in [
        ("nexa.toml", include_str!("../../../config/nexa/nexa.toml")),
        (
            "rooms.toml",
            include_str!("../../../config/nexa/rooms.example.toml"),
        ),
        ("ha-controls.toml", controls),
    ] {
        std::fs::write(dir.path().join(name), contents).expect("write test config");
    }
    LoadedBundle::load_from_dir(dir.path()).expect("load test config")
}

async fn mount_state(server: &MockServer, id: &str, state: &str, attrs: Value) {
    Mock::given(method("GET"))
        .and(path(format!("/api/states/{id}")))
        .and(header("authorization", format!("Bearer {TOKEN}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "entity_id": id, "state": state, "attributes": attrs,
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn reviewed_controls_dispatch_only_to_leaves_after_live_membership_checks() {
    let server = MockServer::start().await;
    let mut config = bundle(CONTROLS);
    config.core.home_assistant.url = server.uri();
    let home = HomeDependencies::new(&config, TOKEN).expect("home dependencies");

    for id in ["light.group", "light.unreviewed"] {
        mount_state(&server, id, "on", json!({"entity_id": ["light.leaf"]})).await;
    }
    mount_state(
        &server,
        "light.drifted",
        "on",
        json!({"entity_id": ["light.other"]}),
    )
    .await;
    for id in ["scene.reading", "scene.warm"] {
        mount_state(
            &server,
            id,
            "unknown",
            json!({"entity_id": ["light.group"]}),
        )
        .await;
    }
    mount_state(
        &server,
        "light.leaf",
        "off",
        json!({
            "supported_color_modes": ["color_temp"],
            "min_color_temp_kelvin": 2000,
            "max_color_temp_kelvin": 6500,
        }),
    )
    .await;

    for body in [
        json!({"entity_id": "light.leaf"}),
        json!({"entity_id": "light.leaf", "brightness_pct": 40}),
        json!({"entity_id": "light.leaf", "color_temp_kelvin": 3000, "brightness": 120}),
    ] {
        Mock::given(method("POST"))
            .and(path("/api/services/light/turn_on"))
            .and(header("authorization", format!("Bearer {TOKEN}")))
            .and(body_json(body))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
    }

    let accepted = ActionStatus::Accepted;
    let drifted = ActionStatus::NotSent(NotSentReason::Policy(PlanError::MembershipChanged));
    let unreviewed_group =
        ActionStatus::NotSent(NotSentReason::Capability(OperationError::UnreviewedGroup));
    let unreviewed_scene = ActionStatus::NotSent(NotSentReason::Policy(PlanError::Unreviewed));
    for (id, operation, expected) in [
        ("light.group", Operation::TurnOn, accepted),
        ("light.unreviewed", Operation::TurnOn, unreviewed_group),
        ("light.drifted", Operation::TurnOn, drifted),
        ("scene.reading", Operation::ActivateScene, accepted),
        ("scene.warm", Operation::ActivateScene, accepted),
        (
            "scene.unreviewed",
            Operation::ActivateScene,
            unreviewed_scene,
        ),
    ] {
        let receipt = home
            .control
            .execute(ActionRequest {
                request_id: id.into(),
                action: EffectIntent {
                    entity_id: id.into(),
                    operation,
                },
            })
            .await
            .expect("receipt");
        assert_eq!(receipt.status, expected, "{id}");
        if expected == accepted {
            assert_eq!(receipt.effects.len(), 1);
            assert_eq!(receipt.effects[0].entity_id, "light.leaf");
        }
    }
    let requests = server
        .received_requests()
        .await
        .expect("recorded HA requests");
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.method.as_str() == "POST")
            .count(),
        3
    );
    server.verify().await;
}

#[test]
fn invalid_scene_operations_fail_home_dependencies_at_boot() {
    for effect in [
        r#"operation = "activate_scene""#,
        r#"operation = "turn_off", brightness_pct = 40"#,
        r#"operation = "brightness""#,
        r#"operation = "color_temp_kelvin", color_temp_kelvin = 0"#,
        r#"operation = "color_temp_kelvin", color_temp_kelvin = 3000, brightness = 0"#,
    ] {
        let config = bundle(&format!(
            r#"[scenes]
            "scene.test" = [{{ entity_id = "light.leaf", {effect} }}]"#,
        ));
        let error = HomeDependencies::new(&config, TOKEN).expect_err(effect);
        assert!(matches!(
            error.downcast_ref::<ConfigError>(),
            Some(ConfigError::Validation(message)) if message.contains("scene.test")
        ));
    }
}
