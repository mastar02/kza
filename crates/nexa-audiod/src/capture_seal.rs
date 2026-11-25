//! Capture policy, explicit stop acknowledgments and recoverable ownership.

use std::collections::{BTreeMap, BTreeSet};

use nexa_proto::is_total_mute_confirmed;
use nexa_proto::v1::CaptureControlStatus;
pub use nexa_proto::v1::CapturePhase;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub struct CaptureOwner(u64);

/// Durable portion of capture control. Owners still present on restart become carried.
/// T2.1 keeps this state in memory. The runtime can persist a snapshot before starting I/O.
#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CaptureState {
    revision: u64,
    muted: bool,
    last_owner: u64,
    owners: BTreeSet<CaptureOwner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerPhase {
    Starting,
    Active,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureTransition {
    Seal { revision: u64 },
    Resume { revision: u64 },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CaptureError {
    #[error("capture control is unsupported")]
    Unsupported,
    #[error("capture revision conflict (current {current})")]
    RevisionConflict { current: u64 },
    #[error("capture transition is in progress")]
    Busy,
    #[error("capture ingress is sealed")]
    Sealed,
    #[error("capture owner or operation is no longer current")]
    StaleAcknowledgment,
    #[error("the operator must confirm that no carried owner is still capturing")]
    ConfirmationRequired,
    #[error("carried owner count conflict (current {current})")]
    CarriedCountConflict { current: u32 },
    #[error("invalid persisted capture state")]
    InvalidState,
    #[error("capture state capacity exhausted")]
    Exhausted,
}

/// A single writer serializes policy, owner changes and acknowledgments.
#[derive(Debug)]
pub struct CaptureControl {
    state: CaptureState,
    phase: CapturePhase,
    supported: bool,
    transition: watch::Sender<Option<CaptureTransition>>,
    live: BTreeMap<CaptureOwner, OwnerPhase>,
    carried: BTreeSet<CaptureOwner>,
    starting: u32,
}

impl CaptureControl {
    /// Restore with ingress sealed until the runtime acknowledges its current transition.
    pub fn restore(state: CaptureState, supported: bool) -> Result<Self, CaptureError> {
        if state.owners.len() >= u32::MAX as usize
            || state.owners.contains(&CaptureOwner(0))
            || state
                .owners
                .last()
                .is_some_and(|id| id.0 > state.last_owner)
        {
            return Err(CaptureError::InvalidState);
        }
        let revision = state.revision;
        let (phase, transition) = match (supported, state.muted) {
            (false, _) => (CapturePhase::Unspecified, None),
            (true, true) => (
                CapturePhase::Muting,
                Some(CaptureTransition::Seal { revision }),
            ),
            (true, false) => (
                CapturePhase::Resuming,
                Some(CaptureTransition::Resume { revision }),
            ),
        };
        Ok(Self {
            phase,
            supported,
            transition: watch::channel(transition).0,
            live: BTreeMap::new(),
            carried: state.owners.clone(),
            starting: 0,
            state,
        })
    }

    #[must_use]
    pub fn durable_state(&self) -> &CaptureState {
        &self.state
    }

    /// Observe the latest pending transition without polling. Copy the current
    /// value before calling capture control or awaiting device I/O.
    #[must_use]
    pub fn watch_transitions(&self) -> watch::Receiver<Option<CaptureTransition>> {
        self.transition.subscribe()
    }

    #[must_use]
    pub fn running_epoch(&self) -> Option<u64> {
        (self.phase == CapturePhase::Running).then_some(self.state.revision)
    }

    #[must_use]
    pub fn status(&self) -> CaptureControlStatus {
        let mut status = CaptureControlStatus {
            phase: self.phase as i32,
            revision: self.state.revision,
            persisted_muted: self.state.muted,
            ingress_sealed: self.phase != CapturePhase::Running,
            supported: self.supported,
            pending_operations: self.starting + u32::from(self.transition.borrow().is_some()),
            live_capture_owners: self.live.len() as u32,
            carried_capture_owners: self.carried.len() as u32,
            total_mute_confirmed: false,
        };
        status.total_mute_confirmed = is_total_mute_confirmed(&status);
        status
    }

    /// Change policy by compare-and-swap. Mute seals immediately. Resume stays
    /// durably muted until explicitly acknowledged.
    pub fn set_muted(&mut self, muted: bool, expected_revision: u64) -> Result<(), CaptureError> {
        self.require_supported()?;
        self.check_revision(expected_revision)?;
        let target_muted = matches!(self.phase, CapturePhase::Muting | CapturePhase::Muted);
        if muted == target_muted {
            return Ok(());
        }
        if !muted && self.phase == CapturePhase::Muting {
            return Err(CaptureError::Busy);
        }
        let revision = self
            .state
            .revision
            .checked_add(1)
            .ok_or(CaptureError::Exhausted)?;
        self.state.revision = revision;
        let (phase, transition) = if muted {
            self.state.muted = true;
            (CapturePhase::Muting, CaptureTransition::Seal { revision })
        } else {
            (
                CapturePhase::Resuming,
                CaptureTransition::Resume { revision },
            )
        };
        self.phase = phase;
        self.transition.send_replace(Some(transition));
        Ok(())
    }

    /// Confirm that ingress and capture-dependent operations are drained for this revision.
    pub fn acknowledge_seal(&mut self, revision: u64) -> Result<(), CaptureError> {
        self.check_transition(CapturePhase::Muting, revision)?;
        self.transition.send_replace(None);
        self.advance_mute();
        Ok(())
    }

    /// Confirm readiness to accept new owners. It is still forbidden to start I/O
    /// until an owner has been registered with `begin_capture`.
    pub fn acknowledge_resumed(&mut self, revision: u64) -> Result<(), CaptureError> {
        self.check_transition(CapturePhase::Resuming, revision)?;
        if !self.live.is_empty() {
            return Err(CaptureError::Busy);
        }
        self.state.muted = false;
        self.phase = CapturePhase::Running;
        self.transition.send_replace(None);
        Ok(())
    }

    /// Reserve a durable owner before starting a capture source or capture-dependent task.
    pub fn begin_capture(&mut self) -> Result<CaptureOwner, CaptureError> {
        if self.running_epoch().is_none() {
            return Err(CaptureError::Sealed);
        }
        if self.state.owners.len() >= u32::MAX as usize - 1 {
            return Err(CaptureError::Exhausted);
        }
        let id = self
            .state
            .last_owner
            .checked_add(1)
            .ok_or(CaptureError::Exhausted)?;
        let owner = CaptureOwner(id);
        self.state.last_owner = id;
        self.state.owners.insert(owner);
        self.live.insert(owner, OwnerPhase::Starting);
        self.starting += 1;
        Ok(owner)
    }

    /// Complete a start operation. The owner remains live until it explicitly stops.
    pub fn acknowledge_started(&mut self, owner: CaptureOwner) -> Result<(), CaptureError> {
        let Some(phase @ OwnerPhase::Starting) = self.live.get_mut(&owner) else {
            return Err(CaptureError::StaleAcknowledgment);
        };
        *phase = OwnerPhase::Active;
        self.starting -= 1;
        Ok(())
    }

    /// Retire an owner only when its I/O has explicitly stopped (or never started).
    pub fn acknowledge_stopped(&mut self, owner: CaptureOwner) -> Result<(), CaptureError> {
        match self.live.remove(&owner) {
            Some(OwnerPhase::Starting) => self.starting -= 1,
            Some(OwnerPhase::Active) => (),
            None => return Err(CaptureError::StaleAcknowledgment),
        }
        self.state.owners.remove(&owner);
        self.advance_mute();
        Ok(())
    }

    /// A returning owner is live again and must explicitly retire.
    pub fn reclaim_carried(&mut self, owner: CaptureOwner) -> Result<(), CaptureError> {
        self.require_supported()?;
        if !self.carried.remove(&owner) {
            return Err(CaptureError::StaleAcknowledgment);
        }
        self.live.insert(owner, OwnerPhase::Active);
        if self.phase == CapturePhase::Muted {
            self.phase = CapturePhase::Muting;
        }
        Ok(())
    }

    /// Forget only carried owners. An exact count, including zero, is required.
    pub fn forget_carried_owners(
        &mut self,
        expected_carried: u32,
        no_capture_running: bool,
    ) -> Result<u32, CaptureError> {
        self.require_supported()?;
        if !no_capture_running {
            return Err(CaptureError::ConfirmationRequired);
        }
        let current = self.carried.len() as u32;
        if current != expected_carried {
            return Err(CaptureError::CarriedCountConflict { current });
        }
        for owner in &self.carried {
            self.state.owners.remove(owner);
        }
        self.carried.clear();
        Ok(current)
    }

    fn check_revision(&self, revision: u64) -> Result<(), CaptureError> {
        if revision == self.state.revision {
            Ok(())
        } else {
            Err(CaptureError::RevisionConflict {
                current: self.state.revision,
            })
        }
    }

    fn require_supported(&self) -> Result<(), CaptureError> {
        if self.supported {
            Ok(())
        } else {
            Err(CaptureError::Unsupported)
        }
    }

    fn check_transition(&self, phase: CapturePhase, revision: u64) -> Result<(), CaptureError> {
        self.require_supported()?;
        self.check_revision(revision)?;
        if self.phase == phase && self.transition.borrow().is_some() {
            Ok(())
        } else {
            Err(CaptureError::StaleAcknowledgment)
        }
    }

    fn advance_mute(&mut self) {
        if self.phase == CapturePhase::Muting
            && self.transition.borrow().is_none()
            && self.live.is_empty()
        {
            self.phase = CapturePhase::Muted;
        }
    }
}
