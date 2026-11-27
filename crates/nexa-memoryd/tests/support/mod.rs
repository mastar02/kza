#![allow(dead_code)]

use nexa_memoryd::db::{FACT_DIMENSIONS, VOICE_DIMENSIONS, VOICE_MODEL};
use nexa_memoryd::service::ACTOR_METADATA;
use nexa_proto::v1::*;
use prost::Message;
use tonic::{Request, metadata::MetadataValue};

pub fn operator(permissions: &[Permission]) -> ActorContext {
    ActorContext {
        subject: "operator".into(),
        role: Role::Operator as i32,
        permissions: permissions.iter().map(|value| *value as i32).collect(),
        ..Default::default()
    }
}

pub fn home(user: &str, permissions: &[Permission]) -> ActorContext {
    ActorContext {
        subject: format!("user:{user}"),
        role: Role::Home as i32,
        user_id: user.into(),
        permissions: permissions.iter().map(|value| *value as i32).collect(),
        ..Default::default()
    }
}

pub fn actor_metadata<T>(msg: T, actor: &ActorContext) -> Request<T> {
    let mut req = Request::new(msg);
    req.metadata_mut().insert_bin(
        ACTOR_METADATA,
        MetadataValue::from_bytes(&actor.encode_to_vec()),
    );
    req
}

pub fn unit(dimensions: usize, axis: usize) -> Vec<f32> {
    let mut coords = vec![0.0; dimensions];
    coords[axis] = 1.0;
    coords
}

pub fn fact(conversation: &str, owner: &str, axis: usize) -> StoreFactRequest {
    StoreFactRequest {
        conversation_id: conversation.into(),
        owner_user_id: owner.into(),
        content: format!("synthetic content {conversation}"),
        category: "test".into(),
        embedding: unit(FACT_DIMENSIONS, axis),
        actor: Some(operator(&[
            Permission::MemoryReview,
            Permission::MemoryAssign,
        ])),
        proposed: false,
    }
}

pub fn query(owner: &str, shared: bool, top_k: u32) -> QueryMemoryRequest {
    QueryMemoryRequest {
        query_embedding: unit(FACT_DIMENSIONS, 0),
        top_k,
        owner_user_id: owner.into(),
        include_shared: shared,
        actor: Some(operator(&[
            Permission::MemoryRead,
            Permission::MemoryAssign,
        ])),
    }
}

pub fn observation(utterance: &str, axis: usize) -> Request<RecordVoiceObservationRequest> {
    actor_metadata(
        RecordVoiceObservationRequest {
            utterance_id: utterance.into(),
            room_id: "room".into(),
            encoder_model_key: VOICE_MODEL.into(),
            transcript_preview: "synthetic speech".into(),
            captured_at_unix_ms: 123,
            embedding: unit(VOICE_DIMENSIONS, axis),
            conversation_id: String::new(),
        },
        &operator(&[Permission::VoicesReview]),
    )
}

pub fn list_profiles() -> ListVoiceProfilesRequest {
    ListVoiceProfilesRequest {
        include_person: true,
        actor: Some(operator(&[
            Permission::VoicesRead,
            Permission::VoicesReview,
            Permission::AdminManage,
        ])),
        cursor: String::new(),
    }
}

pub fn link(profile: &VoiceProfile, person: &str) -> LinkVoiceProfileRequest {
    LinkVoiceProfileRequest {
        profile_id: profile.profile_id.clone(),
        person_id: person.into(),
        expected_profile_revision: profile.profile_revision,
        expected_link_revision: profile.link_revision,
        actor: Some(operator(&[
            Permission::VoicesReview,
            Permission::AdminManage,
        ])),
    }
}

pub fn consent(user: &str, granted: bool) -> SetVoiceConsentRequest {
    SetVoiceConsentRequest {
        user_id: user.into(),
        granted,
        actor: Some(home(user, &[Permission::MemoryOwnVoice])),
    }
}

pub fn erase(conversation: &str) -> EraseConversationRequest {
    EraseConversationRequest {
        conversation_id: conversation.into(),
        actor: Some(operator(&[Permission::MemoryReview])),
    }
}

pub fn purge(user: &str) -> PurgeRetiredUserRequest {
    PurgeRetiredUserRequest {
        user_id: user.into(),
        actor: Some(operator(&[Permission::AdminManage])),
    }
}
