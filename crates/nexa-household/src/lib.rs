//! Household domain core: timers, reminders, routines, lists, alerts & intercom (Phase 4C).
//!
//! Consolidates the legacy fragmented SQLite stores into a single WAL-mode `household.db`
//! owned exclusively by `nexa-brain`.

/// Check whether a reminder's home actions are still within the grace window.
#[must_use]
pub fn home_actions_within_grace(delay_seconds: u64, late_action_grace_s: u32) -> bool {
    delay_seconds <= u64::from(late_action_grace_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_reminder_actions_respect_grace_boundary() {
        assert!(home_actions_within_grace(0, 120));
        assert!(home_actions_within_grace(120, 120));
        assert!(!home_actions_within_grace(121, 120));
    }
}
