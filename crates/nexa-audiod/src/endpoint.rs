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
const AGC_RELEASE_RATIO: f32 = 0.30;
const AGC_RELEASE_CAP: f32 = NOISE_FLOOR_DEFAULT * NOISE_RATIO;
const VAD_AGC_MIN_PEAK_RMS: f32 = 0.008;
const VAD_DROP_RATIO: f32 = 0.45;
const AGC_DOWNWARD_MIN_FRAMES: usize = 3;
const AGC_DOWNWARD_DROP_RATIO: f32 = 0.85;

#[derive(Debug, Clone, Copy)]
struct ObservedFrame {
    rms: f32,
    vad_probability: Option<f32>,
    duration: Duration,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EndpointTelemetry {
    pub level_threshold: f32,
    pub effective_level_threshold: f32,
    pub peak_voice_rms: f32,
    pub last_rms: f32,
    pub voice_ms: u64,
    pub vad_voice_ms: u64,
}

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

/// One post-wake capture. Starts from the room's pre-wake noise floor and can adapt
/// downward if hardware AGC ducks the post-wake room floor after the wake word.
#[derive(Debug)]
pub struct PostWakeEndpoint {
    initial_floor: f32,
    current_floor: f32,
    level_threshold: f32,
    peak_voice_rms: f32,
    last_rms: f32,
    elapsed: Duration,
    voice: Duration,
    level_voice: Duration,
    vad_voice: Duration,
    last_voice: Duration,
    last_level_voice: Duration,
    last_vad_voice: Duration,
    last_strong_level_voice: Duration,
    frames: Vec<ObservedFrame>,
    quiet_below_floor: Vec<f32>,
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
            initial_floor: floor,
            current_floor: floor,
            level_threshold: floor * NOISE_RATIO,
            peak_voice_rms: 0.0,
            last_rms: 0.0,
            elapsed: Duration::ZERO,
            voice: Duration::ZERO,
            level_voice: Duration::ZERO,
            vad_voice: Duration::ZERO,
            last_voice: Duration::ZERO,
            last_level_voice: Duration::ZERO,
            last_vad_voice: Duration::ZERO,
            last_strong_level_voice: Duration::ZERO,
            frames: Vec::with_capacity(64),
            quiet_below_floor: Vec::with_capacity(64),
            decision: EndpointDecision::Listening,
        }
    }

    #[must_use]
    pub fn level_threshold(&self) -> f32 {
        self.level_threshold
    }

    #[must_use]
    pub fn effective_level_threshold(&self) -> f32 {
        let agc_floor = (self.peak_voice_rms * AGC_RELEASE_RATIO).min(AGC_RELEASE_CAP);
        self.level_threshold.max(agc_floor)
    }

    #[must_use]
    pub fn telemetry(&self) -> EndpointTelemetry {
        EndpointTelemetry {
            level_threshold: self.level_threshold,
            effective_level_threshold: self.effective_level_threshold(),
            peak_voice_rms: self.peak_voice_rms,
            last_rms: self.last_rms,
            voice_ms: self.voice.as_millis() as u64,
            vad_voice_ms: self.vad_voice.as_millis() as u64,
        }
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
        let frame = ObservedFrame {
            rms,
            vad_probability,
            duration,
        };
        self.frames.push(frame);
        self.apply_frame(frame);

        let finite_rms = rms.is_finite() && rms >= 0.0;
        let by_vad = vad_probability.is_some_and(|p| (VAD_THRESHOLD..=1.0).contains(&p));
        if finite_rms && !by_vad && rms < self.current_floor {
            self.quiet_below_floor.push(rms);
            if self.quiet_below_floor.len() >= AGC_DOWNWARD_MIN_FRAMES {
                let mut sorted = self.quiet_below_floor.clone();
                sorted.sort_by(f32::total_cmp);
                // Require at least 3 quiet frames (240 ms) below the stale pre-wake floor so a
                // single-frame glitch never lowers the threshold.
                let candidate =
                    sorted[AGC_DOWNWARD_MIN_FRAMES - 1].clamp(NOISE_FLOOR_MIN, self.initial_floor);
                if candidate < self.current_floor * AGC_DOWNWARD_DROP_RATIO {
                    self.current_floor = candidate;
                    self.level_threshold = candidate * NOISE_RATIO;
                    self.replay_frames();
                }
            }
        }

        let level_quiet = if self.level_voice >= MIN_VOICE {
            // Once RMS confirmed the speech, disregard VAD hangover during trailing silence.
            self.elapsed - self.last_level_voice >= LEVEL_END_SILENCE
        } else {
            self.elapsed - self.last_voice >= VAD_END_SILENCE
        };
        // On ATTNS-gated channels (e.g. XVF3800 ch0 Conference), AGC ramp-up during speech can
        // keep post-speech room tone above a squelched pre-wake floor even after Silero confirms
        // silence and RMS has dropped significantly below the speaker's peak.
        let vad_agc_quiet = self.vad_voice >= MIN_VOICE
            && self.peak_voice_rms >= VAD_AGC_MIN_PEAK_RMS
            && self.elapsed - self.last_vad_voice >= VAD_END_SILENCE
            && self.elapsed - self.last_strong_level_voice >= LEVEL_END_SILENCE;
        let quiet = level_quiet || vad_agc_quiet;

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

    fn apply_frame(&mut self, frame: ObservedFrame) {
        self.elapsed += frame.duration;
        let finite_rms = frame.rms.is_finite() && frame.rms >= 0.0;
        if finite_rms {
            self.last_rms = frame.rms;
        }
        let by_level = finite_rms && frame.rms >= self.effective_level_threshold();
        let by_vad = frame
            .vad_probability
            .is_some_and(|p| (VAD_THRESHOLD..=1.0).contains(&p));
        if by_level || by_vad {
            self.voice += frame.duration;
            self.last_voice = self.elapsed;
            if finite_rms {
                self.peak_voice_rms = self.peak_voice_rms.max(frame.rms);
            }
        }
        if by_level {
            self.level_voice += frame.duration;
            self.last_level_voice = self.elapsed;
        }
        if by_vad {
            self.vad_voice += frame.duration;
            self.last_vad_voice = self.elapsed;
        }
        if finite_rms
            && self.peak_voice_rms > 0.0
            && frame.rms >= self.peak_voice_rms * VAD_DROP_RATIO
        {
            self.last_strong_level_voice = self.elapsed;
        }
    }

    fn replay_frames(&mut self) {
        self.peak_voice_rms = 0.0;
        self.last_rms = 0.0;
        self.elapsed = Duration::ZERO;
        self.voice = Duration::ZERO;
        self.level_voice = Duration::ZERO;
        self.vad_voice = Duration::ZERO;
        self.last_voice = Duration::ZERO;
        self.last_level_voice = Duration::ZERO;
        self.last_vad_voice = Duration::ZERO;
        self.last_strong_level_voice = Duration::ZERO;
        for index in 0..self.frames.len() {
            let frame = self.frames[index];
            self.apply_frame(frame);
        }
    }

    fn close_at_max(&mut self) {
        self.decision = if self.voice >= MIN_VOICE {
            EndpointDecision::IncompleteAtMax
        } else {
            EndpointDecision::NoSpeech
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapts_downward_when_hardware_agc_ducks_post_wake_room_floor() {
        // Reproduces live XVF3800 ch1 behavior after idle silence: pre-wake AGC floor is 0.01165
        // (threshold 0.02097), then AGC ducks on wake so speech is 0.012-0.018 and silence is 0.004.
        let mut endpoint = PostWakeEndpoint::new(0.01165);
        let frame = Duration::from_millis(80);
        let mut decision = EndpointDecision::Listening;
        // 12 frames (960 ms) of conversational speech below stale 0.02097 threshold, VAD < 0.5.
        for _ in 0..12 {
            decision = endpoint.observe(0.014, Some(0.25), frame);
            assert_eq!(decision, EndpointDecision::Listening);
        }
        // Trailing post-AGC silence at 0.004 lowers the floor and completes cleanly.
        for _ in 0..6 {
            decision = endpoint.observe(0.004, Some(0.02), frame);
            if decision != EndpointDecision::Listening {
                break;
            }
        }
        assert_eq!(decision, EndpointDecision::Complete);
        assert!(endpoint.telemetry().voice_ms >= 960);
    }

    #[test]
    fn post_wake_silence_after_agc_duck_still_rejects_as_no_speech() {
        let mut endpoint = PostWakeEndpoint::new(0.012);
        let frame = Duration::from_millis(80);
        let mut decision = EndpointDecision::Listening;
        // Pure post-AGC silence at 0.004 with no spoken command must remain NoSpeech.
        for _ in 0..40 {
            decision = endpoint.observe(0.004, Some(0.01), frame);
            if decision != EndpointDecision::Listening {
                break;
            }
        }
        assert_eq!(decision, EndpointDecision::NoSpeech);
    }
}
