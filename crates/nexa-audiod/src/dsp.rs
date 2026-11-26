//! Frame energy and per-room ambient history for the 16 kHz capture path.

use std::collections::VecDeque;
use std::time::Duration;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_SAMPLES: usize = 1_280;
pub const NOISE_FLOOR_MIN: f32 = 0.0006;
pub const NOISE_FLOOR_DEFAULT: f32 = 0.002;
pub const NOISE_FLOOR_MAX: f32 = 0.05;

const NOISE_WINDOW_FRAMES: usize = 125; // 10 seconds of 80 ms frames.
const MIN_HISTORY_FRAMES: usize = 13; // At least one second.

/// RMS of the frame. Invalid PCM yields a non-finite result; empty frames have zero energy.
#[must_use]
pub fn frame_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let energy: f64 = samples
        .iter()
        .map(|&sample| f64::from(sample).powi(2))
        .sum();
    (energy / samples.len() as f64).sqrt() as f32
}

/// Audio time of a block of mono samples at the capture rate.
#[must_use]
pub fn frame_duration(samples: usize) -> Duration {
    Duration::from_secs_f64(samples as f64 / f64::from(SAMPLE_RATE))
}

/// Rolling p20 RMS of the last ten seconds of ambient frames for one room.
#[derive(Debug)]
pub struct NoiseFloorTracker {
    history: VecDeque<f32>,
}

impl Default for NoiseFloorTracker {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(NOISE_WINDOW_FRAMES),
        }
    }
}

impl NoiseFloorTracker {
    /// Record one complete 80 ms ambient frame while the room is idle.
    pub fn observe(&mut self, rms: f32) {
        if !rms.is_finite() || rms < 0.0 {
            return;
        }
        if self.history.len() == NOISE_WINDOW_FRAMES {
            self.history.pop_front();
        }
        self.history.push_back(rms);
    }

    /// Estimate the current floor using the linearly interpolated 20th percentile.
    #[must_use]
    pub fn floor(&self) -> f32 {
        let len = self.history.len();
        if len < MIN_HISTORY_FRAMES {
            return NOISE_FLOOR_DEFAULT;
        }

        let mut rms = [0.0; NOISE_WINDOW_FRAMES];
        for (slot, sample) in rms.iter_mut().zip(&self.history) {
            *slot = *sample;
        }
        let rank = (len - 1) as f32 / 5.0;
        let (_, lower, greater) =
            rms[..len].select_nth_unstable_by(rank.floor() as usize, f32::total_cmp);
        let upper = greater
            .iter()
            .copied()
            .min_by(f32::total_cmp)
            .unwrap_or(*lower);
        let value = *lower + (upper - *lower) * rank.fract();
        value.clamp(NOISE_FLOOR_MIN, NOISE_FLOOR_MAX)
    }
}
