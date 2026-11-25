#![allow(clippy::unwrap_used)]

use std::f64::consts::TAU;
use std::time::Instant;

use nexa_audiod::ambient_guard::AmbientGuard;
use nexa_audiod::barge_in::BargeIn;
use nexa_audiod::dsp::{FRAME_SAMPLES, NoiseFloorTracker, SAMPLE_RATE, frame_duration, frame_rms};
use nexa_audiod::endpoint::{EndpointDecision, PostWakeEndpoint};
use nexa_audiod::traits::{AudioError, VadEngine, WakeEngine};

mod common;

use common::wav;

const WAKE_THRESHOLD: f32 = 0.40;

/// Deterministic stand-in for ONNX. The fixture generator gives near-field voice
/// a stable 150 Hz fundamental. The test model scores the waveform using that
/// frequency, with no knowledge of the clip name or its expected result.
struct FixtureModel;

impl VadEngine for FixtureModel {
    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        let omega = TAU * 150.0 / f64::from(SAMPLE_RATE);
        let (mut real, mut imag) = (0.0, 0.0);
        for (index, sample) in samples.iter().enumerate() {
            let (sin, cos) = (omega * index as f64).sin_cos();
            real += f64::from(*sample) * cos;
            imag += f64::from(*sample) * sin;
        }
        let amplitude = (2.0 * real.hypot(imag) / samples.len() as f64) as f32;
        Ok(
            if amplitude > 0.015 && amplitude > frame_rms(samples) * 0.5 {
                0.9
            } else {
                0.0
            },
        )
    }
}

impl WakeEngine for FixtureModel {
    fn wake_score(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        self.speech_probability(samples)
    }
}

#[derive(Debug)]
struct Replay {
    wakes: usize,
    vad_frames: usize,
    endpoint: EndpointDecision,
}

fn replay(samples: &[f32]) -> Replay {
    let now = Instant::now();
    let mut noise = NoiseFloorTracker::default();
    let mut guard = AmbientGuard::default();
    let mut model = FixtureModel;
    let mut capture = None::<PostWakeEndpoint>;
    let mut result = Replay {
        wakes: 0,
        vad_frames: 0,
        endpoint: EndpointDecision::Listening,
    };

    for (index, frame) in samples.chunks(FRAME_SAMPLES).enumerate() {
        let rms = frame_rms(frame);
        let duration = frame_duration(frame.len());
        let vad = model.speech_probability(frame).unwrap();
        result.vad_frames += usize::from(vad >= 0.5);
        if let Some(capture) = &mut capture {
            result.endpoint = capture.observe(rms, Some(vad), duration);
            continue;
        }
        let score = model.wake_score(frame).unwrap();
        if score >= WAKE_THRESHOLD
            && guard.allow_wake(now + frame_duration(index * FRAME_SAMPLES), score)
        {
            capture = Some(PostWakeEndpoint::new(noise.floor()));
            result.wakes += 1;
        } else if frame.len() == FRAME_SAMPLES {
            noise.observe(rms);
        }
    }
    result
}

#[test]
fn silence_room_noise_and_tv_chatter_have_zero_wake_and_vad_triggers() {
    for name in [
        "silence_2s.wav",
        "room_noise_2s.wav",
        "tv_chatter_hard_neg_3s.wav",
    ] {
        let result = replay(&wav(name));
        assert_eq!(
            (result.wakes, result.vad_frames),
            (0, 0),
            "{name}: {result:?}"
        );
        assert!(!result.endpoint.dispatches(), "{name}: {result:?}");
    }
}

#[test]
fn near_field_wakes_and_completes_a_post_wake_endpoint() {
    let result = replay(&wav("near_field_burst_2s.wav"));
    assert_eq!(result.wakes, 1);
    assert!(result.vad_frames > 0);
    assert_eq!(result.endpoint, EndpointDecision::Complete);
    assert!(result.endpoint.dispatches());
}

#[test]
fn far_field_speech_envelope_rises_above_its_measured_noise_floor() {
    let samples = wav("far_field_low_rms_2s.wav");
    let mut noise = NoiseFloorTracker::default();
    let ambient = wav("room_noise_2s.wav");
    for frame in ambient.as_chunks::<FRAME_SAMPLES>().0 {
        noise.observe(frame_rms(frame));
    }
    let relative_threshold = noise.floor() * 1.8;
    let peak = samples
        .as_chunks::<FRAME_SAMPLES>()
        .0
        .iter()
        .map(|frame| frame_rms(frame))
        .fold(0.0, f32::max);
    assert!(peak > relative_threshold);
    assert!(peak < BargeIn::RMS_THRESHOLD);
}

#[test]
fn continuous_tv_does_not_dispatch_even_if_wake_has_already_fired() {
    let samples = wav("tv_chatter_hard_neg_3s.wav");
    let mut endpoint = PostWakeEndpoint::new(NoiseFloorTracker::default().floor());
    let frames = samples.as_chunks::<FRAME_SAMPLES>().0;
    let result =
        frames
            .iter()
            .cycle()
            .take(63)
            .fold(EndpointDecision::Listening, |decision, frame| {
                assert!(!decision.dispatches());
                endpoint.observe(frame_rms(frame), None, frame_duration(frame.len()))
            });
    assert_eq!(result, EndpointDecision::IncompleteAtMax);
    assert!(!result.dispatches());
}

#[test]
fn barge_in_fixture_requests_exactly_one_cancellation_during_playback() {
    let samples = wav("barge_in_burst_1_2s.wav");
    let mut idle = BargeIn::default();
    let mut playing = BargeIn::default();
    let mut cancellations = 0;
    for frame in samples.chunks(FRAME_SAMPLES) {
        let rms = frame_rms(frame);
        let duration = frame_duration(frame.len());
        assert!(!idle.observe(false, rms, duration));
        cancellations += usize::from(playing.observe(true, rms, duration));
    }
    assert_eq!(cancellations, 1);
}
