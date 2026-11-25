#![allow(clippy::unwrap_used)]

use nexa_audiod::capture_seal::{CaptureControl, CaptureError, CapturePhase, CaptureState};
use nexa_proto::is_total_mute_confirmed;
use nexa_proto::v1::CaptureControlStatus;
use proptest::prelude::*;

fn running() -> CaptureControl {
    let mut capture = CaptureControl::restore(CaptureState::default(), true).unwrap();
    capture.acknowledge_resumed(0).unwrap();
    capture
}

fn status(capture: &CaptureControl, phase: CapturePhase) -> CaptureControlStatus {
    let snapshot = capture.status();
    assert_eq!(snapshot.phase(), phase);
    assert_eq!(
        snapshot.total_mute_confirmed,
        is_total_mute_confirmed(&snapshot)
    );
    snapshot
}

#[test]
fn mute_requires_durable_policy_sealed_ingress_explicit_drain_and_stop() {
    let mut capture = running();
    let snapshot = status(&capture, CapturePhase::Running);
    assert!(
        !snapshot.ingress_sealed && !snapshot.persisted_muted && !snapshot.total_mute_confirmed
    );
    let active = capture.begin_capture().unwrap();
    capture.acknowledge_started(active).unwrap();
    let starting = capture.begin_capture().unwrap();

    capture.set_muted(true, 0).unwrap();
    let snapshot = status(&capture, CapturePhase::Muting);
    assert!(snapshot.ingress_sealed && snapshot.persisted_muted && !snapshot.total_mute_confirmed);
    assert_eq!(snapshot.pending_operations, 2);
    assert_eq!(snapshot.live_capture_owners, 2);
    assert_eq!(capture.begin_capture(), Err(CaptureError::Sealed));
    assert_eq!(capture.running_epoch(), None);
    assert_eq!(capture.set_muted(false, 1), Err(CaptureError::Busy));
    assert_eq!(
        capture.acknowledge_seal(0),
        Err(CaptureError::RevisionConflict { current: 1 })
    );

    capture.acknowledge_seal(1).unwrap();
    capture.acknowledge_stopped(active).unwrap();
    let snapshot = status(&capture, CapturePhase::Muting);
    assert!(!snapshot.total_mute_confirmed);
    assert_eq!(snapshot.pending_operations, 1);
    assert_eq!(snapshot.live_capture_owners, 1);
    capture.acknowledge_started(starting).unwrap();
    assert_eq!(
        capture.acknowledge_started(starting),
        Err(CaptureError::StaleAcknowledgment)
    );
    let snapshot = status(&capture, CapturePhase::Muting);
    assert_eq!(snapshot.pending_operations, 0);
    assert_eq!(snapshot.live_capture_owners, 1);
    assert!(!snapshot.total_mute_confirmed);

    capture.acknowledge_stopped(starting).unwrap();
    assert!(status(&capture, CapturePhase::Muted).total_mute_confirmed);
    assert_eq!(
        capture.acknowledge_stopped(starting),
        Err(CaptureError::StaleAcknowledgment)
    );
}

#[test]
fn silence_and_no_owners_cannot_supply_the_drain_ack() {
    let mut capture = running();
    let owner = capture.begin_capture().unwrap();
    capture.set_muted(true, 0).unwrap();
    capture.acknowledge_stopped(owner).unwrap();
    let snapshot = status(&capture, CapturePhase::Muting);
    assert_eq!(snapshot.live_capture_owners, 0);
    assert_eq!(snapshot.pending_operations, 1);
    assert!(!snapshot.total_mute_confirmed);
    capture.acknowledge_seal(1).unwrap();
    assert!(status(&capture, CapturePhase::Muted).total_mute_confirmed);
}

#[test]
fn restore_keeps_mute_and_rolls_unacknowledged_owners_into_carried() {
    let mut original = running();
    let first = original.begin_capture().unwrap();
    original.acknowledge_started(first).unwrap();
    let second = original.begin_capture().unwrap();
    original.set_muted(true, 0).unwrap();

    let mut recovered = CaptureControl::restore(original.durable_state().clone(), true).unwrap();
    let snapshot = status(&recovered, CapturePhase::Muting);
    assert!(snapshot.persisted_muted && snapshot.ingress_sealed);
    assert_eq!(snapshot.revision, 1);
    assert_eq!(snapshot.carried_capture_owners, 2);
    assert_eq!(snapshot.live_capture_owners, 0);
    assert_eq!(
        recovered.acknowledge_stopped(first),
        Err(CaptureError::StaleAcknowledgment)
    );
    recovered.acknowledge_seal(1).unwrap();
    assert!(!status(&recovered, CapturePhase::Muted).total_mute_confirmed);
    assert_eq!(
        recovered.forget_carried_owners(2, false),
        Err(CaptureError::ConfirmationRequired)
    );
    assert_eq!(
        recovered.forget_carried_owners(1, true),
        Err(CaptureError::CarriedCountConflict { current: 2 })
    );
    assert_eq!(recovered.status().carried_capture_owners, 2);

    recovered.reclaim_carried(first).unwrap();
    assert_eq!(
        recovered.forget_carried_owners(2, true),
        Err(CaptureError::CarriedCountConflict { current: 1 })
    );
    assert_eq!(recovered.forget_carried_owners(1, true), Ok(1));
    assert_eq!(
        recovered.reclaim_carried(second),
        Err(CaptureError::StaleAcknowledgment)
    );
    let snapshot = status(&recovered, CapturePhase::Muting);
    assert!(!snapshot.total_mute_confirmed);
    assert_eq!(snapshot.carried_capture_owners, 0);
    assert_eq!(snapshot.live_capture_owners, 1);
    assert_eq!(
        recovered.forget_carried_owners(1, true),
        Err(CaptureError::CarriedCountConflict { current: 0 })
    );
    assert_eq!(recovered.forget_carried_owners(0, true), Ok(0));
    recovered.acknowledge_stopped(first).unwrap();
    assert!(status(&recovered, CapturePhase::Muted).total_mute_confirmed);

    let roundtrip = CaptureControl::restore(recovered.durable_state().clone(), true).unwrap();
    assert_eq!(roundtrip.status().carried_capture_owners, 0);
}

#[test]
fn resume_keeps_ingress_sealed_and_durable_policy_muted_until_ack() {
    let mut capture = running();
    let old_owner = capture.begin_capture().unwrap();
    capture.acknowledge_stopped(old_owner).unwrap();
    capture.set_muted(true, 0).unwrap();
    capture.acknowledge_seal(1).unwrap();

    capture.set_muted(false, 1).unwrap();
    let snapshot = status(&capture, CapturePhase::Resuming);
    assert_eq!(snapshot.revision, 2);
    assert!(snapshot.ingress_sealed && snapshot.persisted_muted && !snapshot.total_mute_confirmed);
    assert_eq!(snapshot.pending_operations, 1);
    assert_eq!(capture.begin_capture(), Err(CaptureError::Sealed));
    assert_eq!(
        capture.acknowledge_resumed(1),
        Err(CaptureError::RevisionConflict { current: 2 })
    );
    let interrupted = CaptureControl::restore(capture.durable_state().clone(), true).unwrap();
    assert_eq!(interrupted.status().phase(), CapturePhase::Muting);
    assert!(interrupted.status().ingress_sealed);
    assert!(interrupted.status().persisted_muted);

    capture.acknowledge_resumed(2).unwrap();
    assert_eq!(capture.running_epoch(), Some(2));
    assert!(!status(&capture, CapturePhase::Running).persisted_muted);
    let owner = capture.begin_capture().unwrap();
    assert_ne!(owner, old_owner);
    assert_eq!(
        capture.acknowledge_stopped(old_owner),
        Err(CaptureError::StaleAcknowledgment)
    );
    assert_eq!(capture.status().live_capture_owners, 1);
}

#[test]
fn reversing_resume_invalidates_its_pending_ack() {
    let mut capture = running();
    capture.set_muted(true, 0).unwrap();
    capture.acknowledge_seal(1).unwrap();
    capture.set_muted(false, 1).unwrap();
    capture.set_muted(true, 2).unwrap();
    assert_eq!(
        capture.acknowledge_resumed(2),
        Err(CaptureError::RevisionConflict { current: 3 })
    );
    capture.acknowledge_seal(3).unwrap();
    assert!(status(&capture, CapturePhase::Muted).total_mute_confirmed);
}

#[test]
fn unsupported_capture_stays_sealed_without_claiming_confirmation() {
    let mut capture = running();
    capture.set_muted(true, 0).unwrap();
    let mut unsupported = CaptureControl::restore(capture.durable_state().clone(), false).unwrap();
    let snapshot = status(&unsupported, CapturePhase::Unspecified);
    assert!(snapshot.ingress_sealed && snapshot.persisted_muted);
    assert!(!snapshot.supported);
    assert!(!snapshot.total_mute_confirmed);
    assert_eq!(
        unsupported.acknowledge_seal(1),
        Err(CaptureError::Unsupported)
    );
    assert_eq!(
        unsupported.set_muted(false, 1),
        Err(CaptureError::Unsupported)
    );
    assert_eq!(
        unsupported.forget_carried_owners(0, true),
        Err(CaptureError::Unsupported)
    );
}

#[test]
fn each_proto_barrier_is_required_independently() {
    for broken in 0..=127u32 {
        let status = CaptureControlStatus {
            phase: if broken & 1 == 0 {
                CapturePhase::Muted
            } else {
                CapturePhase::Muting
            } as i32,
            persisted_muted: broken & 2 == 0,
            ingress_sealed: broken & 4 == 0,
            supported: broken & 8 == 0,
            pending_operations: u32::from(broken & 16 != 0),
            live_capture_owners: u32::from(broken & 32 != 0),
            carried_capture_owners: u32::from(broken & 64 != 0),
            total_mute_confirmed: true,
            revision: 42,
        };
        assert_eq!(
            is_total_mute_confirmed(&status),
            broken == 0,
            "broken barriers: {broken:#09b}"
        );
    }
}

proptest! {
    #[test]
    fn stale_cas_leaves_mute_policy_and_owners_untouched(wrong_revision in 1..u64::MAX) {
        let mut capture = running();
        capture.begin_capture().unwrap();
        let before = capture.status();
        let durable = capture.durable_state().clone();
        prop_assert_eq!(
            capture.set_muted(true, wrong_revision),
            Err(CaptureError::RevisionConflict { current: 0 })
        );
        prop_assert_eq!(capture.status(), before);
        prop_assert_eq!(capture.durable_state(), &durable);
    }
}
