//! Post-wake endpointing on elapsed capture audio.

use std::time::Duration;

use crate::dsp::{NOISE_FLOOR_DEFAULT, NOISE_FLOOR_MAX, NOISE_FLOOR_MIN};

const MIN_LISTEN: Duration = Duration::from_secs(1);
const MAX_CAPTURE: Duration = Duration::from_secs(5);
const NO_SPEECH_TIMEOUT: Duration = Duration::from_millis(2_500);
const MIN_VOICE: Duration = Duration::from_millis(240);
const LEVEL_END_SILENCE: Duration = Duration::from_millis(300);
const VAD_END_SILENCE: Duration = Duration::from_millis(600);
const STALL_MARGIN: Duration = Duration::from_secs(1);
const NOISE_RATIO: f32 = 1.8;
const VAD_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointDecision {
    Listening,
    Complete,
    NoSpeech,
    IncompleteAtMax,
}

impl EndpointDecision {
    #[must_use]
    pub fn dispatches(self) -> bool {
        // VC-Q2: capped captures stay incomplete.
        self == Self::Complete
    }
}

/// One post-wake capture. The room's noise floor is fixed at wake time.
#[derive(Debug)]
pub struct PostWakeEndpoint {
    level_threshold: f32,
    elapsed: Duration,
    voice: Duration,
    level_voice: Duration,
    last_voice: Duration,
    last_level_voice: Duration,
    decision: EndpointDecision,
}

impl PostWakeEndpoint {
    #[must_use]
    pub fn new(noise_floor: f32) -> Self {
        let floor = if noise_floor.is_finite() {
            noise_floor.clamp(NOISE_FLOOR_MIN, NOISE_FLOOR_MAX)
        } else {
            NOISE_FLOOR_DEFAULT
        };
        Self {
            level_threshold: floor * NOISE_RATIO,
            elapsed: Duration::ZERO,
            voice: Duration::ZERO,
            level_voice: Duration::ZERO,
            last_voice: Duration::ZERO,
            last_level_voice: Duration::ZERO,
            decision: EndpointDecision::Listening,
        }
    }

    #[must_use]
    pub fn level_threshold(&self) -> f32 {
        self.level_threshold
    }

    /// Remaining audio before the cap. The caller can trim the final frame to this duration.
    #[must_use]
    pub fn remaining(&self) -> Duration {
        MAX_CAPTURE - self.elapsed
    }

    /// Consume a post-wake frame. Pre-roll belongs to the caller and never counts as speech.
    /// Once closed, the decision stays fixed for the lifetime of this capture.
    pub fn observe(
        &mut self,
        rms: f32,
        vad_probability: Option<f32>,
        duration: Duration,
    ) -> EndpointDecision {
        if self.decision != EndpointDecision::Listening || duration.is_zero() {
            return self.decision;
        }
        let duration = duration.min(self.remaining());
        self.elapsed += duration;

        let by_level = rms.is_finite() && rms >= self.level_threshold;
        let by_vad = vad_probability.is_some_and(|p| (VAD_THRESHOLD..=1.0).contains(&p));
        if by_level || by_vad {
            self.voice += duration;
            self.last_voice = self.elapsed;
        }
        if by_level {
            self.level_voice += duration;
            self.last_level_voice = self.elapsed;
        }

        let quiet = if self.level_voice >= MIN_VOICE {
            // Once RMS confirmed the speech, disregard VAD hangover during trailing silence.
            self.elapsed - self.last_level_voice >= LEVEL_END_SILENCE
        } else {
            self.elapsed - self.last_voice >= VAD_END_SILENCE
        };

        if self.voice >= MIN_VOICE && self.elapsed >= MIN_LISTEN && quiet {
            self.decision = EndpointDecision::Complete;
        } else if self.elapsed >= MAX_CAPTURE {
            self.close_at_max();
        } else if self.voice < MIN_VOICE && self.elapsed >= NO_SPEECH_TIMEOUT {
            self.decision = EndpointDecision::NoSpeech;
        }
        self.decision
    }

    /// Wall-clock guard for a capture whose source stopped delivering frames.
    pub fn check_stall(&mut self, since_wake: Duration) -> EndpointDecision {
        if self.decision == EndpointDecision::Listening && since_wake >= MAX_CAPTURE + STALL_MARGIN
        {
            self.close_at_max();
        }
        self.decision
    }

    fn close_at_max(&mut self) {
        self.decision = if self.voice >= MIN_VOICE {
            EndpointDecision::IncompleteAtMax
        } else {
            EndpointDecision::NoSpeech
        };
    }
}
