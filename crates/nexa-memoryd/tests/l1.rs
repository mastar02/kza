#![allow(clippy::expect_used)]

mod support;

use nexa_memoryd::db::Store;
use nexa_memoryd::service::MemoryRpc;
use nexa_proto::v1::memory_service_server::MemoryService;
use nexa_proto::v1::*;
use tonic::{Code, Request};

use support::*;

fn service(enable_attribution: bool) -> MemoryRpc {
    let mut store = Store::in_memory().expect("in-memory schema");
    store
        .configure_voice_attribution(enable_attribution)
        .expect("voice switch");
    MemoryRpc::new(store, true)
}

#[tokio::test]
async fn semantic_knn_filters_scope_before_truncating() {
    let rpc = service(false);
    for i in 0..8 {
        rpc.store_fact(Request::new(fact(&format!("private-{i}"), "other-user", 0)))
            .await
            .expect("store unrelated private fact");
    }
    rpc.store_fact(Request::new(fact("personal", "person", 2)))
        .await
        .expect("store owner fact");
    rpc.store_fact(Request::new(fact("shared", "", 0)))
        .await
        .expect("store shared fact");
    let hits = rpc
        .query_memory(Request::new(query("person", false, 1)))
        .await
        .expect("personal KNN")
        .into_inner()
        .hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0].fact.as_ref().expect("fact").conversation_id,
        "personal"
    );
    let hits = rpc
        .query_memory(Request::new(query("person", true, 2)))
        .await
        .expect("combined KNN")
        .into_inner()
        .hits;
    let scopes: Vec<_> = hits
        .iter()
        .map(|hit| hit.fact.as_ref().expect("fact").conversation_id.as_str())
        .collect();
    assert_eq!(scopes, ["shared", "personal"]);
    assert!((hits[0].score - 1.0).abs() < f32::EPSILON);
}

#[tokio::test]
async fn rejects_mismatched_dimensions_models_and_nonfinite_embeddings() {
    let rpc = service(true);
    let mut invalid = fact("conv", "", 0);
    invalid.embedding.pop();
    assert_eq!(
        rpc.store_fact(Request::new(invalid))
            .await
            .expect_err("fact dimensions")
            .code(),
        Code::InvalidArgument
    );
    let mut invalid = query("", true, 1);
    invalid.query_embedding[0] = f32::NAN;
    assert_eq!(
        rpc.query_memory(Request::new(invalid))
            .await
            .expect_err("NaN")
            .code(),
        Code::InvalidArgument
    );
    let mut invalid = observation("voice", 0);
    invalid.get_mut().embedding.pop();
    assert_eq!(
        rpc.record_voice_observation(invalid)
            .await
            .expect_err("voice dimensions")
            .code(),
        Code::InvalidArgument
    );
    let mut invalid = observation("voice", 0);
    invalid.get_mut().encoder_model_key = "unknown".into();
    assert_eq!(
        rpc.record_voice_observation(invalid)
            .await
            .expect_err("voice model")
            .code(),
        Code::InvalidArgument
    );
    let profiles = rpc
        .list_voice_profiles(Request::new(list_profiles()))
        .await
        .expect("list profiles")
        .into_inner();
    assert!(profiles.profiles.is_empty());
}

#[tokio::test]
async fn voice_knn_requires_both_opt_ins_and_revokes_on_unlink() {
    let rpc = service(true);
    let first = rpc
        .record_voice_observation(observation("first", 0))
        .await
        .expect("first observation")
        .into_inner();
    assert!(first.matched_profile_id.is_empty());
    let profile = rpc
        .list_voice_profiles(Request::new(list_profiles()))
        .await
        .expect("list")
        .into_inner()
        .profiles
        .remove(0);
    let initial = link(&profile, "alice");
    let linked = rpc
        .link_voice_profile_to_person(Request::new(initial.clone()))
        .await
        .expect("link")
        .into_inner();
    assert_eq!(linked.link_revision, 1);
    assert_eq!(
        rpc.link_voice_profile_to_person(Request::new(initial))
            .await
            .expect_err("stale link revision")
            .code(),
        Code::Aborted
    );
    let mut stale_profile = link(&linked, "");
    stale_profile.expected_profile_revision += 1;
    assert_eq!(
        rpc.link_voice_profile_to_person(Request::new(stale_profile))
            .await
            .expect_err("stale profile revision")
            .code(),
        Code::Aborted
    );

    let unmatched = rpc
        .record_voice_observation(observation("pre-consent", 0))
        .await
        .expect("unconsented sample")
        .into_inner();
    assert!(unmatched.matched_profile_id.is_empty());

    let mut spoofed = consent("alice", true);
    spoofed.actor = Some(home("bob", &[Permission::MemoryOwnVoice]));
    assert_eq!(
        rpc.set_voice_consent(Request::new(spoofed))
            .await
            .expect_err("consent is personal")
            .code(),
        Code::PermissionDenied
    );
    rpc.set_voice_consent(Request::new(consent("alice", true)))
        .await
        .expect("opt-in");
    let matched = rpc
        .record_voice_observation(observation("matched", 0))
        .await
        .expect("attributed sample")
        .into_inner();
    assert_eq!(matched.matched_profile_id, profile.profile_id);
    assert!((matched.similarity_score - 1.0).abs() < f32::EPSILON);
    let far = rpc
        .record_voice_observation(observation("far", 1))
        .await
        .expect("dissimilar sample")
        .into_inner();
    assert!(far.matched_profile_id.is_empty());

    let unlinked = rpc
        .link_voice_profile_to_person(Request::new(link(&linked, "")))
        .await
        .expect("unlink")
        .into_inner();
    assert_eq!(unlinked.link_revision, 2);
    let relinked = rpc
        .link_voice_profile_to_person(Request::new(link(&unlinked, "alice")))
        .await
        .expect("relink")
        .into_inner();
    assert_eq!(relinked.link_revision, 3);
    let after = rpc
        .record_voice_observation(observation("after-unlink", 0))
        .await
        .expect("post-unlink sample")
        .into_inner();
    assert!(after.matched_profile_id.is_empty());
    let replay = rpc
        .record_voice_observation(observation("matched", 0))
        .await
        .expect("known sample replay")
        .into_inner();
    assert_eq!(replay.capture_id, matched.capture_id);
    assert!(replay.matched_profile_id.is_empty());

    let disabled = service(false);
    assert_eq!(
        disabled
            .set_voice_consent(Request::new(consent("alice", true)))
            .await
            .expect_err("global opt-out")
            .code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn fact_deletion_is_owned_and_retirement_is_irreversible() {
    let rpc = service(true);
    let alice = rpc
        .store_fact(Request::new(fact("alice-conversation", "alice", 0)))
        .await
        .expect("store alice")
        .into_inner();
    let bob = rpc
        .store_fact(Request::new(fact("bob-conversation", "bob", 0)))
        .await
        .expect("store bob")
        .into_inner();
    let mut deletion = DeleteFactRequest {
        fact_id: bob.fact_id.clone(),
        actor: Some(home("alice", &[Permission::MemoryOwnWrite])),
    };
    assert_eq!(
        rpc.delete_fact(Request::new(deletion.clone()))
            .await
            .expect_err("cross owner")
            .code(),
        Code::NotFound
    );
    deletion.fact_id = alice.fact_id;
    let receipt = rpc
        .delete_fact(Request::new(deletion))
        .await
        .expect("owner delete")
        .into_inner();
    assert_eq!((receipt.facts_deleted, receipt.vectors_deleted), (1, 1));

    rpc.record_voice_observation(observation("bob-conversation", 0))
        .await
        .expect("capture");
    let profile = rpc
        .list_voice_profiles(Request::new(list_profiles()))
        .await
        .expect("profiles")
        .into_inner()
        .profiles
        .remove(0);
    rpc.link_voice_profile_to_person(Request::new(link(&profile, "bob")))
        .await
        .expect("link");
    rpc.set_voice_consent(Request::new(consent("bob", true)))
        .await
        .expect("opt-in");
    let retired = rpc
        .purge_retired_user(Request::new(purge("bob")))
        .await
        .expect("purge")
        .into_inner();
    assert_eq!(
        (
            retired.facts_deleted,
            retired.vectors_deleted,
            retired.captures_deleted
        ),
        (1, 2, 1)
    );
    assert_eq!(
        rpc.purge_retired_user(Request::new(purge("bob")))
            .await
            .expect("retry purge")
            .into_inner(),
        retired
    );
    assert_eq!(
        rpc.store_fact(Request::new(fact("new-bob-conversation", "bob", 0)))
            .await
            .expect_err("retired owner")
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        rpc.store_fact(Request::new(fact("bob-conversation", "", 0)))
            .await
            .expect_err("retired conversation")
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        rpc.record_voice_observation(observation("bob-conversation", 0))
            .await
            .expect_err("retired capture")
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        rpc.set_voice_consent(Request::new(consent("bob", false)))
            .await
            .expect_err("retired consent")
            .code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn permissions_are_checked_even_for_valid_roles_and_voice_ids() {
    let rpc = service(true);
    let no_permissions = operator(&[]);
    let code = rpc
        .get_health(actor_metadata(Empty {}, &no_permissions))
        .await
        .expect_err("health permissions")
        .code();
    assert_eq!(code, Code::PermissionDenied);
    assert_eq!(
        rpc.get_health(Request::new(Empty {}))
            .await
            .expect_err("missing actor metadata")
            .code(),
        Code::Unauthenticated
    );
    assert_eq!(
        rpc.get_ambient_control(actor_metadata(Empty {}, &no_permissions))
            .await
            .expect_err("ambient read permissions")
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        rpc.record_voice_observation(actor_metadata(
            observation("x", 0).into_inner(),
            &no_permissions
        ))
        .await
        .expect_err("capture permissions")
        .code(),
        Code::PermissionDenied
    );
    let mut bad = fact("private", "alice", 0);
    bad.actor = Some(home("bob", &[Permission::MemoryOwnWrite]));
    assert_eq!(
        rpc.store_fact(Request::new(bad))
            .await
            .expect_err("wrong owner")
            .code(),
        Code::PermissionDenied
    );
    let mut bad = query("alice", false, 1);
    bad.actor = Some(home("bob", &[Permission::MemoryOwnRead]));
    assert_eq!(
        rpc.query_memory(Request::new(bad))
            .await
            .expect_err("wrong read scope")
            .code(),
        Code::PermissionDenied
    );

    let mut denied = no_permissions.clone();
    denied.permissions = vec![Permission::MemoryRead as i32, i32::MAX];
    let mut bad = query("", true, 1);
    bad.actor = Some(denied);
    assert_eq!(
        rpc.query_memory(Request::new(bad))
            .await
            .expect_err("unknown permission")
            .code(),
        Code::PermissionDenied
    );
    let mut ambiguous = actor_metadata(Empty {}, &operator(&[Permission::DiagnosticsRead]));
    ambiguous.metadata_mut().append_bin(
        nexa_memoryd::service::ACTOR_METADATA,
        tonic::metadata::MetadataValue::from_bytes(b"duplicate"),
    );
    assert_eq!(
        rpc.get_health(ambiguous)
            .await
            .expect_err("duplicate actor")
            .code(),
        Code::Unauthenticated
    );

    let mut unauthorised = list_profiles();
    unauthorised.actor = Some(operator(&[
        Permission::VoicesRead,
        Permission::VoicesReview,
    ]));
    assert_eq!(
        rpc.list_voice_profiles(Request::new(unauthorised.clone()))
            .await
            .expect_err("person metadata requires admin")
            .code(),
        Code::PermissionDenied
    );
    unauthorised.include_person = false;
    rpc.list_voice_profiles(Request::new(unauthorised))
        .await
        .expect("anonymous profiles");
    let mut unauthorised = link(&VoiceProfile::default(), "alice");
    unauthorised.actor = Some(operator(&[Permission::VoicesReview]));
    assert_eq!(
        rpc.link_voice_profile_to_person(Request::new(unauthorised))
            .await
            .expect_err("link requires admin")
            .code(),
        Code::PermissionDenied
    );

    let mut denied = erase("conversation");
    denied.actor = Some(no_permissions.clone());
    assert_eq!(
        rpc.erase_conversation(Request::new(denied))
            .await
            .expect_err("no review")
            .code(),
        Code::PermissionDenied
    );
    let mut denied = purge("alice");
    denied.actor = Some(operator(&[Permission::MemoryReview]));
    assert_eq!(
        rpc.purge_retired_user(Request::new(denied))
            .await
            .expect_err("no admin")
            .code(),
        Code::PermissionDenied
    );
    let denied = SetAmbientPauseRequest {
        paused: true,
        expected_revision: 0,
        actor: Some(no_permissions),
    };
    assert_eq!(
        rpc.set_ambient_pause(Request::new(denied))
            .await
            .expect_err("no ambient permission")
            .code(),
        Code::PermissionDenied
    );
}

#[tokio::test]
async fn ambient_pause_uses_compare_and_swap_and_seals_observations() {
    let rpc = service(true);
    let actor = operator(&[Permission::AmbientControl]);
    let state = rpc
        .get_ambient_control(actor_metadata(Empty {}, &actor))
        .await
        .expect("read ambient")
        .into_inner();
    assert!(!state.paused);
    let pause = SetAmbientPauseRequest {
        paused: true,
        expected_revision: state.revision,
        actor: Some(actor.clone()),
    };
    let state = rpc
        .set_ambient_pause(Request::new(pause.clone()))
        .await
        .expect("pause")
        .into_inner();
    assert!(state.paused);
    assert_eq!(
        rpc.set_ambient_pause(Request::new(pause))
            .await
            .expect_err("stale")
            .code(),
        Code::Aborted
    );
    assert_eq!(
        rpc.record_voice_observation(observation("paused", 0))
            .await
            .expect_err("paused capture")
            .code(),
        Code::FailedPrecondition
    );
    let unpause = SetAmbientPauseRequest {
        paused: false,
        expected_revision: state.revision,
        actor: Some(actor),
    };
    rpc.set_ambient_pause(Request::new(unpause))
        .await
        .expect("unpause");
    rpc.record_voice_observation(observation("paused", 0))
        .await
        .expect("capture admitted again");
}

#[tokio::test]
async fn second_profile_link_invalidates_consent_and_requires_fresh_opt_in() {
    let rpc = service(true);
    rpc.record_voice_observation(observation("first", 0))
        .await
        .expect("first template");
    let first = rpc
        .list_voice_profiles(Request::new(list_profiles()))
        .await
        .expect("first profile")
        .into_inner()
        .profiles
        .remove(0);
    rpc.link_voice_profile_to_person(Request::new(link(&first, "alice")))
        .await
        .expect("link first");
    rpc.set_voice_consent(Request::new(consent("alice", true)))
        .await
        .expect("consent");
    rpc.record_voice_observation(observation("second", 1))
        .await
        .expect("second template");
    let second = rpc
        .list_voice_profiles(Request::new(list_profiles()))
        .await
        .expect("both profiles")
        .into_inner()
        .profiles
        .into_iter()
        .find(|profile| profile.profile_id != first.profile_id)
        .expect("second profile");
    let second = rpc
        .link_voice_profile_to_person(Request::new(link(&second, "alice")))
        .await
        .expect("ambiguous link")
        .into_inner();
    assert_eq!(
        rpc.set_voice_consent(Request::new(consent("alice", true)))
            .await
            .expect_err("ambiguous")
            .code(),
        Code::FailedPrecondition
    );
    rpc.link_voice_profile_to_person(Request::new(link(&second, "")))
        .await
        .expect("resolve ambiguity");
    let capture = rpc
        .record_voice_observation(observation("probe", 0))
        .await
        .expect("capture")
        .into_inner();
    assert!(capture.matched_profile_id.is_empty());
}

#[tokio::test]
async fn retirement_preserves_other_owners_in_a_mixed_conversation() {
    let rpc = service(false);
    rpc.store_fact(Request::new(fact("mixed", "alice", 0)))
        .await
        .expect("alice fact");
    let bob = rpc
        .store_fact(Request::new(fact("mixed", "bob", 0)))
        .await
        .expect("bob fact")
        .into_inner();
    let receipt = rpc
        .purge_retired_user(Request::new(purge("alice")))
        .await
        .expect("purge")
        .into_inner();
    assert_eq!((receipt.facts_deleted, receipt.vectors_deleted), (1, 1));
    let hits = rpc
        .query_memory(Request::new(query("bob", false, 10)))
        .await
        .expect("bob query")
        .into_inner()
        .hits;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].fact.as_ref(), Some(&bob));
    assert_eq!(
        rpc.store_fact(Request::new(fact("mixed", "alice", 0)))
            .await
            .expect_err("retired owner")
            .code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn malformed_rpc_inputs_fail_before_reading_private_state() {
    use Permission::*;

    let rpc = service(false);
    let review = operator(&[VoicesRead, VoicesReview]);
    for query_json in [
        "[]",
        r#"{"review_status":"mystery"}"#,
        r#"{"limit":-1}"#,
        r#"{"text":"private"}"#,
    ] {
        let failure = rpc
            .query_voice_captures(Request::new(VoiceCaptureQueryRequest {
                query_json: query_json.into(),
                actor: Some(review.clone()),
            }))
            .await
            .expect_err("unsupported filter");
        assert_eq!(failure.code(), Code::InvalidArgument);
    }
    let failure = rpc
        .get_voice_capture_review(Request::new(VoiceCaptureReviewRequest {
            fingerprint_id: "invalid\n".into(),
            actor: None,
            include_transcript: true,
        }))
        .await
        .expect_err("actor required before ID parsing");
    assert_eq!(failure.code(), Code::Unauthenticated);

    let response = rpc
        .mutate_memory_item(Request::new(MemoryMutationRequest {
            operation: "voice_consent_revoke".into(),
            payload_json: r#"{"expected_revision":0}"#.into(),
            actor: Some(home("alice", &[MemoryOwnVoice])),
            ..Default::default()
        }))
        .await
        .expect("a person may revoke even with the global switch off")
        .into_inner();
    let payload: serde_json::Value = serde_json::from_str(&response.payload_json).expect("JSON");
    assert_eq!(payload["granted"], false);
    assert_eq!(payload["global_switch_enabled"], false);
    assert!(
        payload["revision"]
            .as_u64()
            .is_some_and(|revision| revision > 0)
    );

    let failure = rpc
        .mutate_memory_item(Request::new(MemoryMutationRequest {
            operation: "voice_consent_revoke".into(),
            payload_json: r#"{"expected_revision":"0"}"#.into(),
            actor: Some(home("alice", &[MemoryOwnVoice])),
            ..Default::default()
        }))
        .await
        .expect_err("no consent revision coercion");
    assert_eq!(failure.code(), Code::InvalidArgument);
}
