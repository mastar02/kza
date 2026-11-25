//! Home Assistant WebSocket state cache, freshness bounds & fail-closed control policy (Phase 4B).

use nexa_config::HaControlPolicyConfig;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ControlPolicyError {
    #[error("control denied by fail-closed policy for {entity_id}.{operation}")]
    Denied {
        entity_id: String,
        operation: String,
    },
}

/// Fail-closed gate verifying every Home Assistant mutation against explicit grants.
pub fn authorize_action(
    policy: &HaControlPolicyConfig,
    entity_id: &str,
    operation: &str,
) -> Result<(), ControlPolicyError> {
    if policy.allows(entity_id, operation) {
        Ok(())
    } else {
        Err(ControlPolicyError::Denied {
            entity_id: entity_id.to_string(),
            operation: operation.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_policy_denies_all_mutations_fail_closed() {
        let policy = HaControlPolicyConfig::default();
        assert!(authorize_action(&policy, "light.living", "turn_on").is_err());
    }
}
