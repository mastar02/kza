//! Multi-endpoint LLM failover, `<think>` stream stripping & one-way CloudConsentGate (Phase 4D).

mod cli;
mod client;
mod health;
mod router;
mod stripper;

pub use cli::run_cli;
pub use client::{Endpoint, LlmError, OpenAiClient};
pub use health::Metrics;
pub use router::FailoverRouter;
pub use stripper::ThinkTagStripper;
pub use tokio_util::sync::CancellationToken;

/// One-way latch enforcing cloud egress consent for this runtime instance (OQ-8).
/// Withdrawal admits zero new cloud calls. In-flight bytes already queued before
/// [`Self::withdraw`] may have reached the gateway.
#[derive(Debug)]
pub struct CloudConsentGate {
    initial_consent: bool,
    admission: std::sync::RwLock<()>,
    withdrawn: CancellationToken,
}

impl CloudConsentGate {
    #[must_use]
    pub fn new(initial_consent: bool) -> Self {
        Self {
            initial_consent,
            admission: Default::default(),
            withdrawn: CancellationToken::new(),
        }
    }

    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.initial_consent && !self.is_withdrawn()
    }

    #[must_use]
    pub fn is_withdrawn(&self) -> bool {
        self.withdrawn.is_cancelled()
    }

    // Serialize withdrawal with each synchronous launch/poll at the egress
    // boundary. No guard is held across an await or a caller callback.
    fn admit(&self) -> Result<std::sync::RwLockReadGuard<'_, ()>, LlmError> {
        let guard = self.admission.read().unwrap_or_else(|e| e.into_inner());
        if self.is_allowed() {
            Ok(guard)
        } else {
            Err(LlmError::Consent)
        }
    }

    /// Permanently latch consent closed for this runtime instance.
    pub fn withdraw(&self) {
        let _exclusive = self.admission.write().unwrap_or_else(|e| e.into_inner());
        self.withdrawn.cancel();
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
