//! Cross-room acoustic wake arbitration so one spoken command executes in one room.
//!
//! Every microphone remains bound 1:1 to its configured room. When multiple rooms
//! detect the same spoken wake word, each room starts capturing immediately with
//! zero added latency and tracks its running peak voice RMS. When an endpoint
//! completes, only the loudest overlapping room dispatches to `nexa-brain`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const OVERLAP_WINDOW: Duration = Duration::from_millis(1_500);
const RETENTION_WINDOW: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
struct Claim {
    id: u64,
    room_id: String,
    started_at: Instant,
    completed_at: Option<Instant>,
    peak_rms: f32,
    wake_score: f32,
    won: bool,
}

#[derive(Debug, Default)]
struct State {
    next_id: u64,
    claims: Vec<Claim>,
}

/// Shared multi-room wake arbiter owned by `nexa-audiod` across all rooms.
#[derive(Debug, Clone, Default)]
pub struct WakeArbiter {
    state: Arc<Mutex<State>>,
}

/// Outcome of arbitration when a room's post-wake endpoint finishes as `Complete`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArbitrationOutcome {
    Won,
    Yielded { winner_room: String },
}

/// RAII handle for an active room wake claim; automatically cancels if dropped early.
#[derive(Debug)]
pub struct WakeClaim {
    arbiter: WakeArbiter,
    id: u64,
    settled: bool,
}

impl WakeArbiter {
    /// Register a newly detected acoustic wake for `room_id`.
    #[must_use]
    pub fn begin(
        &self,
        room_id: &str,
        now: Instant,
        wake_score: f32,
        initial_rms: f32,
    ) -> WakeClaim {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune(&mut state.claims, now);
        state.next_id = state.next_id.wrapping_add(1).max(1);
        let id = state.next_id;
        state.claims.push(Claim {
            id,
            room_id: room_id.to_owned(),
            started_at: now,
            completed_at: None,
            peak_rms: sanitize(initial_rms),
            wake_score: sanitize(wake_score),
            won: false,
        });
        WakeClaim {
            arbiter: self.clone(),
            id,
            settled: false,
        }
    }

    fn update_peak(&self, id: u64, rms: f32) {
        let rms = sanitize(rms);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(claim) = state.claims.iter_mut().find(|claim| claim.id == id) {
            claim.peak_rms = claim.peak_rms.max(rms);
        }
    }

    fn cancel(&self, id: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.claims.retain(|claim| claim.id != id);
    }

    fn finish(&self, id: u64, final_rms: f32, now: Instant) -> ArbitrationOutcome {
        let final_rms = sanitize(final_rms);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        prune(&mut state.claims, now);
        let Some(index) = state.claims.iter().position(|claim| claim.id == id) else {
            return ArbitrationOutcome::Won;
        };
        state.claims[index].peak_rms = state.claims[index].peak_rms.max(final_rms);
        state.claims[index].completed_at = Some(now);
        let current = state.claims[index].clone();

        let better = state
            .claims
            .iter()
            .filter(|other| other.id != current.id && other.room_id != current.room_id)
            .filter(|other| overlaps(&current, other, now))
            .find(|other| {
                if other.won {
                    return true;
                }
                match other.peak_rms.total_cmp(&current.peak_rms) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => {
                        match other.wake_score.total_cmp(&current.wake_score) {
                            std::cmp::Ordering::Greater => true,
                            std::cmp::Ordering::Less => false,
                            std::cmp::Ordering::Equal => other.id < current.id,
                        }
                    }
                }
            })
            .map(|other| other.room_id.clone());

        if let Some(winner_room) = better {
            // Remove yielded claim so it never blocks a subsequent room.
            state.claims.retain(|claim| claim.id != id);
            ArbitrationOutcome::Yielded { winner_room }
        } else {
            state.claims[index].won = true;
            ArbitrationOutcome::Won
        }
    }
}

impl WakeClaim {
    /// Track running voice energy while this room's endpoint is still listening.
    pub fn observe_rms(&self, rms: f32) {
        self.arbiter.update_peak(self.id, rms);
    }

    /// Resolve whether this room won the multi-room wake arbitration.
    #[must_use]
    pub fn finish(mut self, final_rms: f32, now: Instant) -> ArbitrationOutcome {
        self.settled = true;
        self.arbiter.finish(self.id, final_rms, now)
    }
}

impl Drop for WakeClaim {
    fn drop(&mut self) {
        if !self.settled {
            self.arbiter.cancel(self.id);
        }
    }
}

fn sanitize(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn overlaps(current: &Claim, other: &Claim, now: Instant) -> bool {
    let start_diff = current
        .started_at
        .checked_duration_since(other.started_at)
        .or_else(|| other.started_at.checked_duration_since(current.started_at))
        .unwrap_or(Duration::ZERO);
    if start_diff <= OVERLAP_WINDOW {
        return true;
    }
    // Also consider another room that is still actively recording right now, or finished very recently.
    match other.completed_at {
        None => true,
        Some(done) => now.saturating_duration_since(done) <= OVERLAP_WINDOW,
    }
}

fn prune(claims: &mut Vec<Claim>, now: Instant) {
    claims.retain(|claim| match claim.completed_at {
        None => now.saturating_duration_since(claim.started_at) <= RETENTION_WINDOW,
        Some(done) => now.saturating_duration_since(done) <= RETENTION_WINDOW,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn louder_overlapping_room_wins_even_when_quieter_room_finishes_first() {
        let arbiter = WakeArbiter::default();
        let t0 = Instant::now();
        let escritorio = arbiter.begin("escritorio", t0, 0.83, 0.01);
        escritorio.observe_rms(0.164);

        let hall = arbiter.begin("hall", t0 + Duration::from_millis(400), 0.57, 0.004);
        hall.observe_rms(0.0067);

        // Hall finishes first, but escritorio is actively recording 24x louder speech.
        assert_eq!(
            hall.finish(0.0049, t0 + Duration::from_millis(1_800)),
            ArbitrationOutcome::Yielded {
                winner_room: "escritorio".into()
            }
        );
        assert_eq!(
            escritorio.finish(0.0484, t0 + Duration::from_millis(2_400)),
            ArbitrationOutcome::Won
        );
    }

    #[test]
    fn cancelled_louder_room_does_not_block_valid_room() {
        let arbiter = WakeArbiter::default();
        let t0 = Instant::now();
        let noisy_false_wake = arbiter.begin("living", t0, 0.90, 0.25);
        let escritorio = arbiter.begin("escritorio", t0 + Duration::from_millis(100), 0.70, 0.05);
        drop(noisy_false_wake); // e.g. NoSpeech or IncompleteAtMax.
        assert_eq!(
            escritorio.finish(0.05, t0 + Duration::from_millis(1_500)),
            ArbitrationOutcome::Won
        );
    }

    #[test]
    fn non_overlapping_commands_in_different_rooms_both_win() {
        let arbiter = WakeArbiter::default();
        let t0 = Instant::now();
        let first = arbiter.begin("escritorio", t0, 0.80, 0.10);
        assert_eq!(
            first.finish(0.10, t0 + Duration::from_secs(1)),
            ArbitrationOutcome::Won
        );

        let second = arbiter.begin("cocina", t0 + Duration::from_secs(4), 0.75, 0.04);
        assert_eq!(
            second.finish(0.04, t0 + Duration::from_secs(5)),
            ArbitrationOutcome::Won
        );
    }
}
