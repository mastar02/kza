//! Sustained near-field energy detection while a playback is active.

use std::time::Duration;

#[derive(Debug, Default)]
pub struct BargeIn {
    above_threshold: Duration,
    fired: bool,
}

impl BargeIn {
    pub const RMS_THRESHOLD: f32 = 0.03;
    pub const SUSTAINED: Duration = Duration::from_millis(200);

    /// Emit once per playback after consecutive above-threshold audio frames.
    pub fn observe(&mut self, playback_active: bool, rms: f32, duration: Duration) -> bool {
        if !playback_active {
            *self = Self::default();
            return false;
        }
        if self.fired {
            return false;
        }
        if !rms.is_finite() || rms <= Self::RMS_THRESHOLD {
            self.above_threshold = Duration::ZERO;
        } else {
            self.above_threshold = self.above_threshold.saturating_add(duration);
            self.fired = self.above_threshold >= Self::SUSTAINED;
        }
        self.fired
    }
}
