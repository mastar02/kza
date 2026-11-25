//! Multi-endpoint LLM failover, `<think>` stream stripping & one-way CloudConsentGate (Phase 4D).

use std::sync::atomic::{AtomicBool, Ordering};

/// One-way latch enforcing cloud egress consent (OQ-8).
/// Once withdrawn, cloud calls remain blocked for the lifetime of the process.
#[derive(Debug)]
pub struct CloudConsentGate {
    allowed: AtomicBool,
    withdrawn: AtomicBool,
}

impl CloudConsentGate {
    #[must_use]
    pub fn new(initial_consent: bool) -> Self {
        Self {
            allowed: AtomicBool::new(initial_consent),
            withdrawn: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.allowed.load(Ordering::Acquire) && !self.withdrawn.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn is_withdrawn(&self) -> bool {
        self.withdrawn.load(Ordering::Acquire)
    }

    /// Permanently latch consent closed for this runtime instance.
    pub fn withdraw(&self) {
        self.withdrawn.store(true, Ordering::Release);
        self.allowed.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_consent_withdrawal_is_one_way() {
        let gate = CloudConsentGate::new(true);
        assert!(gate.is_allowed());
        assert!(!gate.is_withdrawn());

        gate.withdraw();
        assert!(!gate.is_allowed());
        assert!(gate.is_withdrawn());
    }
}
