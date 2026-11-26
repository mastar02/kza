#![allow(clippy::unwrap_used)]

use std::time::{Duration, Instant};

use nexa_audiod::ambient_guard::{AmbientGuard, CaptureOutcome, GuardState};
use nexa_audiod::barge_in::BargeIn;
use nexa_audiod::dsp::{
    FRAME_SAMPLES, NOISE_FLOOR_DEFAULT, NOISE_FLOOR_MAX, NOISE_FLOOR_MIN, NoiseFloorTracker,
    frame_rms,
};
use nexa_audiod::endpoint::{EndpointDecision, PostWakeEndpoint};
use nexa_audiod::ma1260::{
    Command, CommandFrame, FrameError, is_acknowledgment, percent_to_step, step_to_percent,
};
use nexa_audiod::ring::{PushError, capture_ring};
use proptest::prelude::*;

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}

#[test]
fn rms_handles_empty_invalid_and_extreme_pcm() {
    assert_eq!(frame_rms(&[]), 0.0);
    assert_eq!(frame_rms(&[0.0; FRAME_SAMPLES]), 0.0);
    assert_eq!(frame_rms(&[f32::MAX, -f32::MAX]), f32::MAX);
    assert!(frame_rms(&[f32::NAN, 0.0]).is_nan());
    assert!(frame_rms(&[f32::INFINITY]).is_infinite());
    assert!((frame_rms(&[0.0, -1.0]) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-7);
}

#[test]
fn noise_floor_interpolates_p20_and_evicts_after_ten_seconds() {
    let mut tracker = NoiseFloorTracker::default();
    for i in (0..12).rev() {
        tracker.observe(0.001 + i as f32 * 0.00005);
    }
    assert_eq!(tracker.floor(), NOISE_FLOOR_DEFAULT);
    tracker.observe(f32::NAN);
    tracker.observe(-1.0);
    assert_eq!(tracker.floor(), NOISE_FLOOR_DEFAULT);
    tracker.observe(0.0016);
    assert!((tracker.floor() - 0.00112).abs() < 1e-8);

    // Real rooms with higher idle noise (e.g. hall=0.0052, cocina=0.0038) must not clamp at 0.002.
    let mut loud_room = NoiseFloorTracker::default();
    for _ in 0..20 {
        loud_room.observe(0.0052);
    }
    assert!((loud_room.floor() - 0.0052).abs() < 1e-6);

    for _ in 0..125 {
        tracker.observe(0.0);
    }
    assert_eq!(tracker.floor(), NOISE_FLOOR_MIN);
    for _ in 0..99 {
        tracker.observe(0.1);
    }
    assert_eq!(tracker.floor(), NOISE_FLOOR_MIN);
    for _ in 0..2 {
        tracker.observe(0.1);
    }
    assert_eq!(tracker.floor(), NOISE_FLOOR_MAX);
}

#[test]
fn ring_capacity_whole_writes_and_wraparound() {
    let (mut producer, mut consumer) = capture_ring();
    let input: Vec<_> = (0..FRAME_SAMPLES).map(|index| index as f32).collect();
    let mut output = [0.0; FRAME_SAMPLES];
    assert_eq!(consumer.try_read(&mut output), 0);
    producer.try_push(&input).unwrap();
    assert_eq!(producer.try_push(&[70.0]), Err(PushError::Full));
    assert_eq!(consumer.try_read(&mut output[..12]), 12);
    assert_eq!(&output[..12], &input[..12]);
    assert_eq!(producer.try_push(&[90.0; 13]), Err(PushError::Full));
    producer.try_push(&[100.0; 12]).unwrap();
    assert_eq!(consumer.try_read(&mut output), FRAME_SAMPLES);
    assert_eq!(&output[..FRAME_SAMPLES - 12], &input[12..]);
    assert_eq!(&output[FRAME_SAMPLES - 12..], &[100.0; 12]);
    drop(consumer);
    assert_eq!(producer.try_push(&[0.0]), Err(PushError::Closed));
}

#[tokio::test]
async fn ring_waits_for_data_and_drains_after_producer_disconnects() {
    let (mut producer, mut consumer) = capture_ring();
    let (waiting, ready) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        producer.try_push(&[0.2, 0.4]).unwrap();
        // Drop follows the last write without waiting for the reader.
    });
    let mut output = [0.0];
    {
        let mut read = std::pin::pin!(consumer.read(&mut output));
        std::future::poll_fn(|cx| {
            use std::future::Future;
            assert!(read.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        waiting.send(()).unwrap();
        assert_eq!(tokio::time::timeout(ms(1_000), read).await.unwrap(), 1);
    }
    assert_eq!(output, [0.2]);
    assert_eq!(consumer.read(&mut output).await, 1);
    assert_eq!(output, [0.4]);
    assert_eq!(consumer.read(&mut output).await, 0);
    writer.join().unwrap();
}

#[test]
fn endpoint_waits_for_minimum_listen_and_ignores_vad_tail_after_level_speech() {
    let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
    let level_boundary = NOISE_FLOOR_DEFAULT * 1.8;
    assert_eq!(
        endpoint.observe(level_boundary, Some(0.0), ms(240)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.0, Some(0.99), ms(759)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.0, Some(0.99), ms(1)),
        EndpointDecision::Complete
    );
    assert!(endpoint.observe(1.0, None, ms(10_000)).dispatches());

    let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
    endpoint.observe(0.0, None, ms(700));
    endpoint.observe(level_boundary, None, ms(240));
    assert_eq!(
        endpoint.observe(0.0, Some(0.95), ms(299)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.0, Some(0.95), ms(1)),
        EndpointDecision::Complete
    );
}

#[test]
fn vad_only_endpoint_requires_six_hundred_ms_of_joint_silence() {
    let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
    endpoint.observe(0.0, None, ms(500));
    endpoint.observe(0.001, Some(0.5), ms(240));
    assert_eq!(
        endpoint.observe(0.0, None, ms(599)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.001, Some(0.499), ms(1)),
        EndpointDecision::Complete
    );
}

#[test]
fn endpoint_discards_short_click_and_invalid_signals_at_no_speech_deadline() {
    let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
    endpoint.observe(0.8, None, ms(239));
    assert_eq!(
        endpoint.observe(f32::INFINITY, Some(f32::NAN), ms(2_260)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.0, Some(1.1), ms(1)),
        EndpointDecision::NoSpeech
    );
}

#[test]
fn endpoint_stall_is_bounded_and_incomplete() {
    for (onset, expected) in [
        (0, EndpointDecision::NoSpeech),
        (240, EndpointDecision::IncompleteAtMax),
    ] {
        let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
        endpoint.observe(0.05, None, ms(onset));
        assert_eq!(endpoint.check_stall(ms(5_999)), EndpointDecision::Listening);
        assert_eq!(endpoint.check_stall(ms(6_000)), expected);
        assert!(!expected.dispatches());
    }
}

#[test]
fn endpoint_completes_cleanly_in_louder_rooms_above_default_floor() {
    // Regression: hall (~0.0052) and cocina (~0.0038) exceed NOISE_FLOOR_DEFAULT (0.002)
    // and previously clamped to 0.002 -> level_threshold=0.0036 -> IncompleteAtMax on room silence.
    let mut endpoint = PostWakeEndpoint::new(0.0052);
    assert!((endpoint.level_threshold() - 0.0052 * 1.8).abs() < 1e-6);
    assert_eq!(
        endpoint.observe(0.015, Some(0.9), ms(400)),
        EndpointDecision::Listening
    );
    assert_eq!(
        endpoint.observe(0.0052, Some(0.1), ms(700)),
        EndpointDecision::Complete
    );
}

#[test]
fn ambient_guard_has_independent_room_hysteresis_and_cooldown() {
    let start = Instant::now();
    let mut living = AmbientGuard::default();
    let mut hall = AmbientGuard::default();
    for second in 0..3 {
        living.record(
            start + Duration::from_secs(second),
            CaptureOutcome::Rejected,
        );
    }
    let now = start + Duration::from_secs(3);
    assert_eq!(living.state(now), GuardState::Strict);
    assert_eq!(hall.state(now), GuardState::Normal);
    assert!(hall.allow_wake(now, 0.41));
    assert!(!living.allow_wake(now, 0.449));
    assert!(living.allow_wake(now, 0.45));
    assert!(!living.allow_wake(now, f32::NAN));
    assert!(!living.allow_follow_up(now));
    living.record(now, CaptureOutcome::OtherFailure);
    assert!(!living.allow_follow_up(now));
    living.record(now, CaptureOutcome::Accepted);
    assert!(living.allow_follow_up(now + ms(11_999)));
    assert!(!living.allow_follow_up(now + ms(12_000)));

    let cooldown_start = now + ms(15_000);
    for _ in 0..5 {
        living.record(cooldown_start, CaptureOutcome::Rejected);
    }
    assert_eq!(living.state(cooldown_start), GuardState::Strict);
    living.record(cooldown_start, CaptureOutcome::Rejected);
    assert_eq!(living.state(cooldown_start), GuardState::Cooldown);
    assert!(!living.allow_wake(cooldown_start, 0.849));
    assert!(living.allow_wake(cooldown_start, 0.85));
    assert!(!living.allow_follow_up(cooldown_start));
    assert_eq!(
        living.state(cooldown_start + ms(29_999)),
        GuardState::Cooldown
    );
    assert_eq!(
        living.state(cooldown_start + ms(30_000)),
        GuardState::Strict
    );
    assert!(!living.allow_wake(cooldown_start + ms(119_999), 0.4));
    assert_eq!(
        living.state(cooldown_start + ms(239_998)),
        GuardState::Strict
    );
    assert_eq!(
        living.state(cooldown_start + ms(239_999)),
        GuardState::Normal
    );
}

#[test]
fn ambient_guard_expires_old_rejections() {
    let mut guard = AmbientGuard::default();
    let start = Instant::now();
    guard.record(start, CaptureOutcome::Rejected);
    let later = start + ms(180_001);
    guard.record(later, CaptureOutcome::Rejected);
    guard.record(later, CaptureOutcome::Rejected);
    assert_eq!(guard.state(later), GuardState::Normal);
}

#[test]
fn barge_in_requires_an_active_playback_and_a_contiguous_run() {
    let mut barge = BargeIn::default();
    assert!(!barge.observe(false, 0.2, ms(1_000)));
    assert!(!barge.observe(true, 0.2, ms(199)));
    assert!(!barge.observe(true, BargeIn::RMS_THRESHOLD, ms(1)));
    assert!(!barge.observe(true, 0.2, ms(199)));
    assert!(!barge.observe(true, f32::NAN, ms(1)));
    assert!(!barge.observe(true, 0.2, ms(199)));
    assert!(barge.observe(true, 0.2, ms(1)));
    assert!(!barge.observe(true, 0.2, ms(500)));
    assert!(!barge.observe(false, 0.2, ms(500)));
    assert!(barge.observe(true, 0.2, ms(200)));
}

#[test]
fn ma1260_framing_validates_header_zone_command_and_payload() {
    let frame = CommandFrame::new(3, Command::Volume, percent_to_step(75)).unwrap();
    assert_eq!(frame.encode(), [0x02, 3, 0x02, 29, 0x03]);
    assert_eq!(CommandFrame::decode(&frame.encode()), Ok(frame));
    assert_eq!(
        CommandFrame::decode(&[0x02, 2, 0x02, 20]),
        Err(FrameError::Framing)
    );
    assert_eq!(
        CommandFrame::decode(&[0x02, 2, 0x02, 20, 0]),
        Err(FrameError::Framing)
    );
    assert_eq!(
        CommandFrame::decode(&[0x02, 7, 0x02, 20, 3]),
        Err(FrameError::Zone(7))
    );
    assert_eq!(
        CommandFrame::decode(&[0x02, 2, 0xff, 20, 3]),
        Err(FrameError::Command(0xff))
    );
    assert_eq!(
        CommandFrame::decode(&[0x02, 2, 0x02, 39, 3]),
        Err(FrameError::Data)
    );
    assert_eq!(CommandFrame::new(0, Command::Mute, 1).unwrap().zone(), 0);
    assert_eq!(CommandFrame::new(6, Command::Source, 8).unwrap().data(), 8);
    assert!(is_acknowledgment(&[0x06]));
    assert!(!is_acknowledgment(&[]));
    assert!(!is_acknowledgment(&[0x15]));
    assert!(!is_acknowledgment(&[0x06, 0x00]));
    assert_eq!(percent_to_step(0), 0);
    assert_eq!(step_to_percent(0), 0);
    assert_eq!(percent_to_step(100), 38);
    assert_eq!(step_to_percent(38), 100);
    assert_eq!(percent_to_step(u32::MAX), 38);
    assert_eq!(step_to_percent(u8::MAX), 100);
}

proptest! {
    #[test]
    fn rms_is_invariant_under_sign_and_scales_linearly(
        samples in prop::collection::vec(-1.0f32..1.0, 1..=FRAME_SAMPLES),
        gain in 0.0f32..2.0,
    ) {
        let scaled: Vec<_> = samples.iter().map(|sample| -sample * gain).collect();
        prop_assert!((frame_rms(&scaled) - gain * frame_rms(&samples)).abs() < 3e-7);
    }

    #[test]
    fn sustained_speech_caps_at_five_seconds_without_dispatch(frame_ms in 1..=320u64) {
        let mut endpoint = PostWakeEndpoint::new(NOISE_FLOOR_DEFAULT);
        let mut elapsed = 0;
        while elapsed < 5_000 {
            elapsed += frame_ms;
            let decision = endpoint.observe(0.1, None, ms(frame_ms));
            if elapsed < 5_000 {
                prop_assert_eq!(decision, EndpointDecision::Listening);
            } else {
                prop_assert_eq!(decision, EndpointDecision::IncompleteAtMax);
                prop_assert!(!decision.dispatches());
                prop_assert_eq!(endpoint.remaining(), Duration::ZERO);
            }
        }
    }

    #[test]
    fn barge_in_never_fires_without_a_qualifying_run(
        sequence in prop::collection::vec((any::<bool>(), 1..=120u64), 1..100)
    ) {
        let mut detector = BargeIn::default();
        let mut contiguous = 0;
        let mut fired = false;
        for (loud, duration) in sequence {
            contiguous = if loud { contiguous + duration } else { 0 };
            let should_fire = !fired && contiguous >= 200;
            let rms = if loud { 0.04 } else { 0.005 };
            prop_assert_eq!(detector.observe(true, rms, ms(duration)), should_fire);
            fired |= should_fire;
        }
    }

    #[test]
    fn normal_guard_enters_strict_only_after_three_rejections_in_window(
        gaps in prop::collection::vec(0..=250u64, 0..25)
    ) {
        let start = Instant::now();
        let mut guard = AmbientGuard::default();
        let mut elapsed = 0;
        let mut times = Vec::new();
        for gap in gaps {
            elapsed += gap;
            times.retain(|&last| elapsed - last <= 180);
            times.push(elapsed);
            let now = start + Duration::from_secs(elapsed);
            guard.record(now, CaptureOutcome::Rejected);
            if times.len() >= 3 {
                prop_assert_eq!(guard.state(now), GuardState::Strict);
                break;
            }
            prop_assert_eq!(guard.state(now), GuardState::Normal);
        }
    }

    #[test]
    fn volume_roundtrip_has_only_quantization_error(
        percent in 0..=100u32, step in 0..=38u8, zone in 0..=6u8,
    ) {
        let roundtrip = step_to_percent(percent_to_step(percent));
        prop_assert!(percent.abs_diff(roundtrip) <= 1);
        prop_assert_eq!(percent_to_step(step_to_percent(step)), step);
        let command = CommandFrame::new(zone, Command::Volume, step).unwrap();
        prop_assert_eq!(CommandFrame::decode(&command.encode()), Ok(command));
    }
}
