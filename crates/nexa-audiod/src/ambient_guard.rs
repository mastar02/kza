//! Per-room wake hysteresis and cooldown for repeated rejected captures.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

const REJECT_WINDOW: Duration = Duration::from_secs(180);
const STRICT_REJECTS: usize = 3;
const COOLDOWN_REJECTS: usize = 6;
const COOLDOWN: Duration = Duration::from_secs(30);
const QUIET_TO_NORMAL: Duration = Duration::from_secs(120);
const FOLLOW_UP_GRACE: Duration = Duration::from_secs(12);
const STRICT_SCORE: f32 = 0.45;
const COOLDOWN_OVERRIDE: f32 = 0.85;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GuardState {
    #[default]
    Normal,
    Strict,
    Cooldown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureOutcome {
    Accepted,
    Rejected,
    OtherFailure,
}

/// Keep a separate instance per room. The caller supplies the monotonic clock.
#[derive(Debug, Default)]
pub struct AmbientGuard {
    state: GuardState,
    last_reject: Option<Instant>,
    last_accept: Option<Instant>,
    cooldown_start: Option<Instant>,
    rejects: VecDeque<Instant>,
}

impl AmbientGuard {
    pub fn state(&mut self, now: Instant) -> GuardState {
        if self
            .cooldown_start
            .is_some_and(|start| now.saturating_duration_since(start) >= COOLDOWN)
        {
            self.state = GuardState::Strict;
            self.cooldown_start = None;
            self.rejects.clear();
        }
        if self.state == GuardState::Strict
            && self
                .last_reject
                .is_some_and(|last| now.saturating_duration_since(last) >= QUIET_TO_NORMAL)
        {
            self.state = GuardState::Normal;
            self.rejects.clear();
        }
        self.state
    }

    /// Apply guard policy to a candidate already detected by the wake engine.
    pub fn allow_wake(&mut self, now: Instant, score: f32) -> bool {
        let state = self.state(now);
        if !(0.0..=1.0).contains(&score) {
            return false;
        }
        match state {
            GuardState::Normal => true,
            GuardState::Strict if score < STRICT_SCORE => {
                self.last_reject = Some(now);
                false
            }
            GuardState::Strict => true,
            GuardState::Cooldown => score >= COOLDOWN_OVERRIDE,
        }
    }

    pub fn record(&mut self, now: Instant, outcome: CaptureOutcome) {
        let state = self.state(now);
        match outcome {
            CaptureOutcome::Accepted => self.last_accept = Some(now),
            CaptureOutcome::OtherFailure => (),
            CaptureOutcome::Rejected => {
                self.last_reject = Some(now);
                let limit = match state {
                    GuardState::Normal => STRICT_REJECTS,
                    GuardState::Strict => COOLDOWN_REJECTS,
                    GuardState::Cooldown => return,
                };
                self.rejects
                    .retain(|time| now.saturating_duration_since(*time) <= REJECT_WINDOW);
                self.rejects.push_back(now);
                if self.rejects.len() >= limit {
                    self.rejects.clear();
                    self.state = if state == GuardState::Normal {
                        GuardState::Strict
                    } else {
                        self.cooldown_start = Some(now);
                        GuardState::Cooldown
                    };
                }
            }
        }
    }

    /// In strict mode, allow follow-up only within the grace period of an accepted capture.
    pub fn allow_follow_up(&mut self, now: Instant) -> bool {
        match self.state(now) {
            GuardState::Normal => true,
            GuardState::Strict => self
                .last_accept
                .is_some_and(|last| now.saturating_duration_since(last) < FOLLOW_UP_GRACE),
            GuardState::Cooldown => false,
        }
    }
}
