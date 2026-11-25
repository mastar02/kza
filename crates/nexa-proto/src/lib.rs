//! Shared Protobuf/gRPC wire contracts and core domain invariants for Nexa v2.
//!
//! This crate is the dependency-free leaf of the workspace DAG. All inter-daemon
//! communication over local Unix Domain Sockets (`$XDG_RUNTIME_DIR/nexa/*.sock`)
//! compiles against these generated schemas.

#![allow(clippy::all)]

pub mod v1 {
    tonic::include_proto!("nexa.v1");
}

/// Canonical socket file names under `$XDG_RUNTIME_DIR/nexa/`.
pub mod sockets {
    pub const AUDIO: &str = "audio.sock";
    pub const ML: &str = "ml.sock";
    pub const MEMORY: &str = "memory.sock";
    pub const BRAIN: &str = "brain.sock";
}

/// Pure helper enforcing KZA's honest mute confirmation invariant:
/// silence never counts as a stop ACK.
#[must_use]
pub fn is_total_mute_confirmed(status: &v1::CaptureControlStatus) -> bool {
    status.phase == v1::CapturePhase::Muted as i32
        && status.persisted_muted
        && status.ingress_sealed
        && status.supported
        && status.pending_operations == 0
        && status.live_capture_owners == 0
        && status.carried_capture_owners == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn total_mute_confirmed_requires_all_barriers_and_zero_owners() {
        let mut status = v1::CaptureControlStatus {
            phase: v1::CapturePhase::Muted as i32,
            revision: 3,
            persisted_muted: true,
            ingress_sealed: true,
            supported: true,
            pending_operations: 0,
            live_capture_owners: 0,
            carried_capture_owners: 0,
            total_mute_confirmed: true,
        };
        assert!(is_total_mute_confirmed(&status));

        status.carried_capture_owners = 1;
        assert!(!is_total_mute_confirmed(&status));

        status.carried_capture_owners = 0;
        status.live_capture_owners = 1;
        assert!(!is_total_mute_confirmed(&status));

        status.live_capture_owners = 0;
        status.ingress_sealed = false;
        assert!(!is_total_mute_confirmed(&status));

        status.ingress_sealed = true;
        status.phase = v1::CapturePhase::Muting as i32;
        assert!(!is_total_mute_confirmed(&status));
    }

    #[test]
    fn actor_context_round_trips_over_protobuf() {
        let actor = v1::ActorContext {
            subject: "home-user-1".to_string(),
            role: v1::Role::Home as i32,
            permissions: vec![
                v1::Permission::HomeRead as i32,
                v1::Permission::HomeControl as i32,
            ],
            user_id: "person-1".to_string(),
            lease_id: "lease-abc".to_string(),
            lease_epoch: 42,
        };
        let encoded = actor.encode_to_vec();
        let decoded = v1::ActorContext::decode(encoded.as_slice()).expect("decode ActorContext");
        assert_eq!(decoded, actor);
    }
}
