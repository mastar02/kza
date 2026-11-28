#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nexa_audiod::capture_seal::{CaptureControl, CapturePhase, CaptureState};
use nexa_audiod::dsp::FRAME_SAMPLES;
use nexa_audiod::grpc::AudioService;
use nexa_audiod::hw::{amp::Ma1260, simulation::Simulation};
use nexa_audiod::ring::{CaptureProducer, PushError};
use nexa_audiod::runtime::{RoomPipeline, SharedShadow, run_capture_control};
use nexa_audiod::traits::{
    AudioError, CaptureCounters, CaptureSource, DspProbe, ShadowStt, VadEngine, WakeEngine,
};
use nexa_config::{Ma1260Connection, RoomDefinition, XvfTuning};
use nexa_proto::v1::audio_service_server::AudioService as AudioRpc;
use nexa_proto::v1::{Empty, SubscribeUtterancesRequest, UtteranceKind};
use tokio::sync::{Notify, mpsc, watch};
use tokio_stream::StreamExt;
use tonic::Request;

#[derive(Debug)]
struct Script {
    attempts: Arc<Mutex<Vec<Instant>>>,
    failures: usize,
    utterances: usize,
    sample: f32,
    frame: usize,
    broken: bool,
}

impl CaptureSource for Script {
    async fn start(&mut self) -> Result<(), AudioError> {
        self.attempts.lock().unwrap().push(Instant::now());
        self.broken = self.failures != 0;
        self.failures = self.failures.saturating_sub(1);
        self.frame = 0;
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            if self.broken {
                return Err(AudioError::Unavailable);
            }
            if self.frame == 15 * self.utterances {
                return std::future::pending().await;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
            buf.fill(match self.frame % 15 {
                0 => 0.8,
                1..=4 => self.sample,
                _ => 0.0,
            });
            self.frame += 1;
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}

#[derive(Debug)]
struct Detector;

impl VadEngine for Detector {
    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        Ok(if samples[0] > 0.01 { 1.0 } else { 0.0 })
    }
}

impl WakeEngine for Detector {
    fn wake_score(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        Ok(if samples[0] > 0.5 { 0.9 } else { 0.0 })
    }
}

#[derive(Debug)]
struct Shadow;

impl ShadowStt for Shadow {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AudioError> {
        let is_living = samples.iter().any(|&sample| (sample - 0.2).abs() < 1e-4);
        Ok(if is_living { "living" } else { "hall" }.into())
    }
}

fn room(id: &str) -> RoomDefinition {
    RoomDefinition {
        room_id: id.into(),
        name: id.into(),
        mic_usb_port: None,
        mic_serial: None,
        mic_channel: 0,
        ma1260_zone: 1,
        default_light: None,
        default_climate: None,
        xvf_tuning: XvfTuning {
            pp_agc_max_gain: 16.0,
            pp_attns_mode: 1,
            pp_attns_slope: 1.0,
            pp_min_nn: 0.4,
        },
    }
}

fn services() -> (
    Arc<Mutex<CaptureControl>>,
    Arc<AudioService<Ma1260, Simulation>>,
) {
    let mut capture = CaptureControl::restore(CaptureState::default(), true).unwrap();
    capture.acknowledge_resumed(0).unwrap();
    let capture = Arc::new(Mutex::new(capture));
    let service = AudioService::new(
        capture.clone(),
        Ma1260::new(Ma1260Connection::Simulation),
        Simulation,
        [],
    )
    .unwrap();
    (capture, Arc::new(service))
}

struct FrameSource(mpsc::Receiver<Result<f32, AudioError>>);

struct CountShadow(Arc<AtomicUsize>);

impl ShadowStt for CountShadow {
    fn transcribe(&mut self, _: &[f32]) -> Result<String, AudioError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok("shadow transcript".into())
    }
}

impl CaptureSource for FrameSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            let level = self.0.recv().await.ok_or(AudioError::Unavailable)??;
            buf.fill(level);
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}

async fn frames(sender: &mpsc::Sender<Result<f32, AudioError>>, count: usize, level: f32) {
    for _ in 0..count {
        sender.send(Ok(level)).await.unwrap();
    }
}

async fn wait_for_frames(service: &AudioService<Ma1260, Simulation>, count: u64) {
    loop {
        let snapshot = service
            .get_microphones(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        if snapshot
            .microphones
            .iter()
            .any(|room| room.frames_captured >= count)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

async fn assert_quiet(
    stream: &mut <AudioService<Ma1260, Simulation> as AudioRpc>::SubscribeUtterancesStream,
) {
    assert!(
        tokio::time::timeout(Duration::from_millis(40), stream.next())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn ambient_requires_opt_in_and_never_runs_shadow_stt() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let (sender, receiver) = mpsc::channel(32);
        let shadow_calls = Arc::new(AtomicUsize::new(0));
        let mut commands = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest::default()))
            .await
            .unwrap()
            .into_inner();
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                FrameSource(receiver),
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(CountShadow(shadow_calls.clone()))),
                shutdown,
            ),
        );

        // A full segment without ambient demand must not even attempt publication.
        frames(&sender, 12, 0.2).await;
        frames(&sender, 9, 0.0).await;
        wait_for_frames(&service, 21).await;
        assert_quiet(&mut commands).await;
        let health = service
            .get_health(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            health
                .detail
                .contains("utterances_published=0; utterances_dropped=0;")
        );

        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        // Switching opt-in streams must discard the departing audience's partial segment.
        frames(&sender, 9, 0.2).await;
        wait_for_frames(&service, 30).await;
        drop(ambient);
        ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        frames(&sender, 1, 0.2).await;
        frames(&sender, 8, 0.0).await;
        wait_for_frames(&service, 39).await;
        assert_quiet(&mut ambient).await;

        frames(&sender, 10, 0.3).await;
        frames(&sender, 8, 0.0).await;
        let utterance = ambient.next().await.unwrap().unwrap();
        assert_eq!(utterance.kind(), UtteranceKind::Ambient);
        assert_eq!(utterance.room_id, "living");
        assert_eq!(utterance.capture_epoch, 0);
        let mut expected = 0.0_f32.to_le_bytes().repeat(6 * FRAME_SAMPLES);
        expected.extend(0.3_f32.to_le_bytes().repeat(10 * FRAME_SAMPLES));
        assert_eq!(utterance.pcm_f32le, expected);
        assert!(utterance.parakeet_shadow_text.is_empty());
        assert_eq!(utterance.parakeet_latency_ms, 0);

        frames(&sender, 12, 0.2).await;
        frames(&sender, 8, 0.0).await;
        wait_for_frames(&service, 77).await;
        assert_quiet(&mut ambient).await; // Cooldown suppresses a second segment.
        assert_quiet(&mut commands).await;
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        assert_eq!(shadow_calls.load(Ordering::Relaxed), 0);
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn wake_discards_pending_ambient_and_arms_one_same_epoch_follow_up() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let (sender, receiver) = mpsc::channel(32);
        let mut stream = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                FrameSource(receiver),
                Detector,
                Detector,
                0.4,
            )
            .run(service, Arc::new(SharedShadow::new(Shadow)), shutdown),
        );
        frames(&sender, 11, 0.03).await;
        frames(&sender, 1, 0.8).await;
        frames(&sender, 4, 0.2).await;
        frames(&sender, 9, 0.0).await;
        let command = stream.next().await.unwrap().unwrap();
        assert_eq!(command.kind(), UtteranceKind::Command);
        assert_eq!(command.parakeet_shadow_text, "living");
        // Command includes bounded wake pre-roll + wake frame (0.8) + post-wake speech (0.2) + trailing silence.
        assert!(command.pcm_f32le.len() > (4 + 9) * 1280 * 4);
        assert!(
            command
                .pcm_f32le
                .as_chunks::<4>()
                .0
                .iter()
                .any(|&b| b == 0.8_f32.to_le_bytes())
        );
        assert!(
            command
                .pcm_f32le
                .as_chunks::<4>()
                .0
                .iter()
                .any(|&b| b == 0.2_f32.to_le_bytes())
        );

        frames(&sender, 3, 0.05).await;
        frames(&sender, 8, 0.0).await;
        let follow_up = stream.next().await.unwrap().unwrap();
        assert_eq!(follow_up.kind(), UtteranceKind::FollowUp);
        assert_eq!(follow_up.capture_epoch, command.capture_epoch);
        assert_eq!(follow_up.parakeet_shadow_text, "hall");
        assert_eq!(follow_up.pcm_f32le, 0.05_f32.to_le_bytes().repeat(3 * 1280));

        frames(&sender, 10, 0.2).await;
        frames(&sender, 8, 0.0).await;
        assert_eq!(
            stream.next().await.unwrap().unwrap().kind(),
            UtteranceKind::Ambient
        );
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn capture_seal_drops_partial_ambient_before_muting_and_resuming() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let (sender, receiver) = mpsc::channel(32);
        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                FrameSource(receiver),
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(FailedShadow)),
                shutdown.clone(),
            ),
        );
        let coordinator = tokio::spawn(run_capture_control(capture.clone(), shutdown));
        frames(&sender, 11, 0.2).await;
        wait_for_frames(&service, 11).await;
        capture.lock().unwrap().set_muted(true, 0).unwrap();
        while !capture.lock().unwrap().status().total_mute_confirmed {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(capture.lock().unwrap().status().live_capture_owners, 0);
        assert_quiet(&mut ambient).await;
        capture.lock().unwrap().set_muted(false, 1).unwrap();

        frames(&sender, 9, 0.0).await;
        wait_for_frames(&service, 20).await;
        assert_quiet(&mut ambient).await;
        frames(&sender, 10, 0.4).await;
        frames(&sender, 8, 0.0).await;
        let resumed = ambient.next().await.unwrap().unwrap();
        assert_eq!(resumed.kind(), UtteranceKind::Ambient);
        assert_eq!(resumed.capture_epoch, 2);
        let mut expected = 0.0_f32.to_le_bytes().repeat(6 * FRAME_SAMPLES);
        expected.extend(0.4_f32.to_le_bytes().repeat(10 * FRAME_SAMPLES));
        assert_eq!(resumed.pcm_f32le, expected);
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        coordinator.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn ambient_recovery_discards_partial_pcm_and_caps_continuous_speech() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let (sender, receiver) = mpsc::channel(32);
        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                FrameSource(receiver),
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(FailedShadow)),
                shutdown,
            ),
        );
        frames(&sender, 11, 0.2).await;
        wait_for_frames(&service, 11).await;
        sender.send(Err(AudioError::Unavailable)).await.unwrap();
        frames(&sender, 9, 0.0).await;
        wait_for_frames(&service, 20).await;
        assert_quiet(&mut ambient).await;

        frames(&sender, 150, 0.3).await;
        let capped = ambient.next().await.unwrap().unwrap();
        assert_eq!(capped.kind(), UtteranceKind::Ambient);
        assert_eq!(capped.capture_epoch, 0);
        let mut expected = 0.0_f32.to_le_bytes().repeat(6 * FRAME_SAMPLES);
        expected.extend(0.3_f32.to_le_bytes().repeat(144 * FRAME_SAMPLES));
        assert_eq!(capped.pcm_f32le, expected);
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn room_retries_do_not_interrupt_another_rooms_utterances() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let shadow = Arc::new(SharedShadow::new(Shadow));
        let mut stream = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest::default()))
            .await
            .unwrap()
            .into_inner();
        let healthy_attempts = Arc::new(Mutex::new(Vec::new()));
        let broken_attempts = Arc::new(Mutex::new(Vec::new()));
        let healthy = RoomPipeline::new(
            &room("living"),
            Script {
                attempts: healthy_attempts.clone(),
                failures: 0,
                utterances: 5,
                sample: 0.2,
                frame: 0,
                broken: false,
            },
            Detector,
            Detector,
            0.4,
        );
        let broken = RoomPipeline::new(
            &room("hall"),
            Script {
                attempts: broken_attempts.clone(),
                failures: 2,
                utterances: 2,
                sample: 0.05,
                frame: 0,
                broken: false,
            },
            Detector,
            Detector,
            0.4,
        );
        let first = tokio::spawn(healthy.run(service.clone(), shadow.clone(), shutdown.clone()));
        let second = tokio::spawn(broken.run(service.clone(), shadow, shutdown));

        let mut ids = std::collections::HashSet::new();
        let mut counts = std::collections::BTreeMap::new();
        for index in 0..7 {
            let utterance = stream.next().await.unwrap().unwrap();
            if index == 0 {
                assert_eq!(utterance.room_id, "living");
            }
            assert_eq!(utterance.room_id, utterance.parakeet_shadow_text);
            assert_eq!(utterance.sample_rate, 16_000);
            assert_eq!(utterance.pcm_f32le.len() % (1280 * 4), 0);
            let frame_count = utterance.pcm_f32le.len() / (1280 * 4);
            assert!(
                (14..=24).contains(&frame_count),
                "unexpected frame_count={frame_count}"
            );
            assert!(ids.insert(utterance.utterance_id));
            *counts.entry(utterance.room_id).or_insert(0) += 1;
        }
        assert_eq!(counts.get("living"), Some(&5));
        assert_eq!(counts.get("hall"), Some(&2));
        let attempts = broken_attempts.lock().unwrap().clone();
        assert_eq!(attempts.len(), 3);
        assert!(attempts[1] - attempts[0] >= Duration::from_millis(100));
        assert!(attempts[2] - attempts[1] >= Duration::from_millis(200));
        assert_eq!(healthy_attempts.lock().unwrap().len(), 1);

        let microphones = service
            .get_microphones(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        let hall = microphones
            .microphones
            .iter()
            .find(|room| room.room_id == "hall")
            .unwrap();
        let living = microphones
            .microphones
            .iter()
            .find(|room| room.room_id == "living")
            .unwrap();
        assert_eq!(hall.recoveries, 2);
        assert_eq!(living.recoveries, 0);

        stop.send_replace(true);
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        assert_eq!(capture.lock().unwrap().status().live_capture_owners, 0);
    })
    .await
    .unwrap();
}

struct FailedShadow;

impl ShadowStt for FailedShadow {
    fn transcribe(&mut self, _: &[f32]) -> Result<String, AudioError> {
        std::thread::sleep(Duration::from_millis(5));
        Err(AudioError::Adapter(
            "injected Parakeet inference failure".into(),
        ))
    }
}

#[tokio::test]
async fn shadow_inference_error_preserves_the_command_and_reports_model_latency() {
    let (_, service) = services();
    let (stop, shutdown) = watch::channel(false);
    let mut stream = service
        .subscribe_utterances(Request::new(SubscribeUtterancesRequest::default()))
        .await
        .unwrap()
        .into_inner();
    let room = RoomPipeline::new(
        &room("living"),
        Script {
            attempts: Arc::default(),
            failures: 0,
            utterances: 1,
            sample: 0.2,
            frame: 0,
            broken: false,
        },
        Detector,
        Detector,
        0.4,
    );
    let worker =
        tokio::spawn(room.run(service, Arc::new(SharedShadow::new(FailedShadow)), shutdown));

    let utterance = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(utterance.room_id, "living");
    assert!(utterance.parakeet_shadow_text.is_empty());
    assert!(utterance.parakeet_latency_ms >= 5);
    assert!(!utterance.pcm_f32le.is_empty());

    stop.send_replace(true);
    worker.await.unwrap().unwrap();
}

struct BlockingShadow {
    entered: Arc<Notify>,
    release: std::sync::mpsc::Receiver<()>,
}

impl ShadowStt for BlockingShadow {
    fn transcribe(&mut self, _: &[f32]) -> Result<String, AudioError> {
        self.entered.notify_one();
        self.release.recv_timeout(Duration::from_secs(4)).unwrap();
        Ok("sealed transcript".into())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn total_mute_waits_for_inflight_shadow_work_and_discards_its_result() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let entered = Arc::new(Notify::new());
        let (release, wait) = std::sync::mpsc::channel();
        let shadow = Arc::new(SharedShadow::new(BlockingShadow {
            entered: entered.clone(),
            release: wait,
        }));
        let mut stream = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest::default()))
            .await
            .unwrap()
            .into_inner();
        let pipeline = RoomPipeline::new(
            &room("living"),
            Script {
                attempts: Arc::default(),
                failures: 0,
                utterances: 1,
                sample: 0.2,
                frame: 0,
                broken: false,
            },
            Detector,
            Detector,
            0.4,
        );
        let worker = tokio::spawn(pipeline.run(service, shadow, shutdown.clone()));
        let coordinator = tokio::spawn(run_capture_control(capture.clone(), shutdown));

        entered.notified().await;
        capture.lock().unwrap().set_muted(true, 0).unwrap();
        let status = capture.lock().unwrap().status();
        assert!(status.ingress_sealed && !status.total_mute_confirmed);
        assert_eq!(status.live_capture_owners, 1);
        release.send(()).unwrap();

        loop {
            if capture.lock().unwrap().status().total_mute_confirmed {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            capture.lock().unwrap().status().phase(),
            CapturePhase::Muted
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(30), stream.next())
                .await
                .is_err()
        );
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        coordinator.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct DeadIsochSource {
    reboots: Arc<AtomicUsize>,
    starts: Arc<AtomicUsize>,
    frames: usize,
    fail_forever: bool,
}

impl CaptureSource for DeadIsochSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        self.starts.fetch_add(1, Ordering::Relaxed);
        self.frames = 0;
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            let value = if !self.fail_forever && self.reboots.load(Ordering::Relaxed) >= 1 {
                0.002
            } else {
                0.0
            };
            buf.fill(value);
            self.frames += 1;
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn supports_dead_isoch_watchdog(&self) -> bool {
        true
    }

    async fn reboot(&mut self) -> Result<(), AudioError> {
        self.reboots.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::test]
async fn dead_isochronous_watchdog_reboots_and_enforces_hourly_rate_limit() {
    tokio::time::timeout(Duration::from_secs(8), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let reboots = Arc::new(AtomicUsize::new(0));
        let starts = Arc::new(AtomicUsize::new(0));
        let shadow = Arc::new(SharedShadow::new(CountShadow(Arc::default())));
        let pipeline = RoomPipeline::new(
            &room("living"),
            DeadIsochSource {
                reboots: reboots.clone(),
                starts: starts.clone(),
                frames: 0,
                fail_forever: true,
            },
            Detector,
            Detector,
            0.4,
        );
        let worker = tokio::spawn(pipeline.run(service, shadow, shutdown.clone()));
        let coordinator = tokio::spawn(run_capture_control(capture, shutdown));

        while starts.load(Ordering::Relaxed) < 6 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(reboots.load(Ordering::Relaxed), 4);
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        coordinator.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct FixedAzimuthProbe(f32);

#[tonic::async_trait]
impl DspProbe for FixedAzimuthProbe {
    async fn read_azimuth(&self) -> Option<f32> {
        Some(self.0)
    }
}

struct ContinuousDoaSource {
    frames: usize,
}

impl CaptureSource for ContinuousDoaSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            let val = if self.frames < 5 {
                0.001
            } else if self.frames < 310 {
                0.3
            } else {
                0.0
            };
            buf.fill(val);
            self.frames += 1;
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn dsp_probe(&self) -> Option<Arc<dyn DspProbe>> {
        Some(Arc::new(FixedAzimuthProbe(1.0)))
    }
}

#[tokio::test]
async fn continuous_ambient_speech_chains_without_blind_spot_and_tracks_doa() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                ContinuousDoaSource { frames: 0 },
                Detector,
                Detector,
                0.4,
            )
            .run(
                service,
                Arc::new(SharedShadow::new(SilentShadowStub)),
                shutdown,
            ),
        );

        let first = ambient.next().await.unwrap().unwrap();
        let second = ambient.next().await.unwrap().unwrap();
        assert!(!first.continues_segment);
        assert!(second.continues_segment);
        for seg in [&first, &second] {
            assert!(seg.has_doa);
            assert!((seg.doa_azimuth_rad - 1.0).abs() < 0.05);
            assert!(seg.doa_stability > 0.9);
            assert!(seg.speech_ratio > 0.8);
        }
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct SilentShadowStub;

impl ShadowStt for SilentShadowStub {
    fn transcribe(&mut self, _: &[f32]) -> Result<String, AudioError> {
        Ok(String::new())
    }
}

struct LongContinuousSource {
    frames: usize,
    total_frames: usize,
}

impl CaptureSource for LongContinuousSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            if self.frames >= self.total_frames + 16 {
                return std::future::pending().await;
            }
            let val = if self.frames < 5 || self.frames >= self.total_frames {
                0.001
            } else {
                0.3
            };
            buf.fill(val);
            self.frames += 1;
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}

#[tokio::test]
async fn one_hour_continuous_ambient_speech_streams_without_drops_or_unbounded_buffers() {
    tokio::time::timeout(Duration::from_secs(20), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        // 45,000 frames * 80 ms = 3,600 seconds (1 hour of continuous speech).
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                LongContinuousSource {
                    frames: 0,
                    total_frames: 45_000,
                },
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(SilentShadowStub)),
                shutdown,
            ),
        );

        let mut segments = 0_usize;
        while segments < 300 {
            let seg = ambient.next().await.unwrap().unwrap();
            assert_eq!(seg.kind(), UtteranceKind::Ambient);
            assert!(
                seg.pcm_f32le.len() <= 150 * FRAME_SAMPLES * 4,
                "segment {} exceeded 12s PCM cap: {} bytes",
                segments,
                seg.pcm_f32le.len()
            );
            if segments == 0 {
                assert!(!seg.continues_segment);
            } else {
                assert!(
                    seg.continues_segment,
                    "segment {segments} should chain cleanly"
                );
            }
            segments += 1;
        }
        let health = service
            .get_health(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            health.detail.contains("utterances_dropped=0"),
            "unexpected dropped utterances: {}",
            health.detail
        );
        assert!(
            health.detail.contains("utterances_lagged=0"),
            "unexpected lagged utterances: {}",
            health.detail
        );

        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct CountingDspProbe(Arc<AtomicUsize>);

#[tonic::async_trait]
impl DspProbe for CountingDspProbe {
    async fn read_azimuth(&self) -> Option<f32> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Some(0.75)
    }
}

struct ProbedFrameSource {
    rx: mpsc::Receiver<Result<f32, AudioError>>,
    probe_calls: Arc<AtomicUsize>,
}

impl CaptureSource for ProbedFrameSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let mut buf = [0.0_f32; FRAME_SAMPLES];
        loop {
            let level = self.rx.recv().await.ok_or(AudioError::Unavailable)??;
            buf.fill(level);
            match producer.push(&buf).await {
                Ok(()) => (),
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    fn dsp_probe(&self) -> Option<Arc<dyn DspProbe>> {
        Some(Arc::new(CountingDspProbe(Arc::clone(&self.probe_calls))))
    }
}

#[tokio::test]
async fn dsp_probe_runs_only_with_ambient_demand_and_pauses_during_active_wake_command() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let (sender, receiver) = mpsc::channel(32);
        let probe_calls = Arc::new(AtomicUsize::new(0));
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                ProbedFrameSource {
                    rx: receiver,
                    probe_calls: Arc::clone(&probe_calls),
                },
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(Shadow)),
                shutdown,
            ),
        );

        // Without ambient demand, DSP polling must stay completely idle.
        frames(&sender, 5, 0.0).await;
        wait_for_frames(&service, 5).await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(probe_calls.load(Ordering::Relaxed), 0);

        // Subscribing with include_ambient activates the independent DSP poller.
        let mut ambient = service
            .subscribe_utterances(Request::new(SubscribeUtterancesRequest {
                include_ambient: true,
            }))
            .await
            .unwrap()
            .into_inner();
        let deadline = Instant::now() + Duration::from_secs(1);
        while probe_calls.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(probe_calls.load(Ordering::Relaxed) >= 1);

        // Trigger a wake command: capture_busy must become true and pause DSP polling
        // across a >500ms window while the post-wake endpoint is still listening.
        frames(&sender, 1, 0.8).await;
        wait_for_frames(&service, 6).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        let during_wake = probe_calls.load(Ordering::Relaxed);
        for step in 0..4 {
            tokio::time::sleep(Duration::from_millis(160)).await;
            frames(&sender, 1, 0.2).await;
            wait_for_frames(&service, 7 + step).await;
        }
        assert_eq!(probe_calls.load(Ordering::Relaxed), during_wake);

        frames(&sender, 9, 0.0).await;
        let cmd = ambient.next().await.unwrap().unwrap();
        assert_eq!(cmd.kind(), UtteranceKind::Command);
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct NoAudioFailSource {
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl CaptureSource for NoAudioFailSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("start");
        Ok(())
    }

    async fn run(
        &mut self,
        _producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        Err(AudioError::Unavailable)
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("stop");
        Ok(())
    }

    fn supports_dead_isoch_watchdog(&self) -> bool {
        true
    }

    async fn reboot(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("reboot");
        Ok(())
    }
}

#[tokio::test]
async fn watchdog_stops_before_reboot_and_reboots_after_consecutive_no_audio_failures() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let events = Arc::new(Mutex::new(Vec::new()));
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                NoAudioFailSource {
                    events: Arc::clone(&events),
                },
                Detector,
                Detector,
                0.4,
            )
            .run(
                service.clone(),
                Arc::new(SharedShadow::new(SilentShadowStub)),
                shutdown.clone(),
            ),
        );
        let coordinator = tokio::spawn(run_capture_control(capture, shutdown));

        while !events.lock().unwrap().contains(&"reboot") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let prefix = {
            let log = events.lock().unwrap();
            let pos = log.iter().position(|&ev| ev == "reboot").unwrap();
            log[..=pos].to_vec()
        };
        assert_eq!(
            prefix,
            vec!["start", "stop", "start", "stop", "start", "stop", "reboot"]
        );

        let health = service
            .get_health(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            health.detail.contains("watchdog_reboots=1"),
            "unexpected health detail: {}",
            health.detail
        );

        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        coordinator.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct PartialZeroSource {
    events: Arc<Mutex<Vec<&'static str>>>,
    chunks_pushed: Arc<AtomicUsize>,
    reboots: Arc<AtomicUsize>,
}

impl CaptureSource for PartialZeroSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("start");
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        _counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let chunk = if self.reboots.load(Ordering::Relaxed) >= 1 {
            [0.002_f32; 320]
        } else {
            [0.0_f32; 320]
        };
        loop {
            match producer.push(&chunk).await {
                Ok(()) => {
                    self.chunks_pushed.fetch_add(1, Ordering::Relaxed);
                }
                Err(PushError::Closed) => return Ok(()),
                Err(PushError::Full) => return Err(AudioError::Unavailable),
            }
            tokio::task::yield_now().await;
        }
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("stop");
        Ok(())
    }

    fn supports_dead_isoch_watchdog(&self) -> bool {
        true
    }

    async fn reboot(&mut self) -> Result<(), AudioError> {
        self.events.lock().unwrap().push("reboot");
        self.reboots.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::test]
async fn dead_isoch_watchdog_counts_exact_80ms_frames_even_on_partial_chunks() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (capture, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let events = Arc::new(Mutex::new(Vec::new()));
        let chunks_pushed = Arc::new(AtomicUsize::new(0));
        let reboots = Arc::new(AtomicUsize::new(0));
        let worker = tokio::spawn(
            RoomPipeline::new(
                &room("living"),
                PartialZeroSource {
                    events: Arc::clone(&events),
                    chunks_pushed: Arc::clone(&chunks_pushed),
                    reboots: Arc::clone(&reboots),
                },
                Detector,
                Detector,
                0.4,
            )
            .run(
                service,
                Arc::new(SharedShadow::new(SilentShadowStub)),
                shutdown.clone(),
            ),
        );
        let coordinator = tokio::spawn(run_capture_control(capture, shutdown));

        while reboots.load(Ordering::Relaxed) < 1 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // 75 frames * 1280 samples = 300 chunks of 320 samples (must NOT fire at 75 chunks).
        assert!(chunks_pushed.load(Ordering::Relaxed) >= 300);
        let log = events.lock().unwrap().clone();
        let reboot_idx = log.iter().position(|&ev| ev == "reboot").unwrap();
        assert_eq!(&log[..=reboot_idx], &["start", "stop", "reboot"]);

        stop.send_replace(true);
        worker.await.unwrap().unwrap();
        coordinator.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}

struct CounterSource;

impl CaptureSource for CounterSource {
    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn run(
        &mut self,
        mut producer: CaptureProducer,
        counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        counters.xruns.fetch_add(2, Ordering::Relaxed);
        counters.ring_overruns.fetch_add(3, Ordering::Relaxed);
        let _ = producer.push(&[0.002_f32; FRAME_SAMPLES]).await;
        std::future::pending().await
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}

#[tokio::test]
async fn capture_counters_are_reported_in_health_detail() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let (_, service) = services();
        let (stop, shutdown) = watch::channel(false);
        let worker = tokio::spawn(
            RoomPipeline::new(&room("living"), CounterSource, Detector, Detector, 0.4).run(
                service.clone(),
                Arc::new(SharedShadow::new(SilentShadowStub)),
                shutdown,
            ),
        );
        wait_for_frames(&service, 1).await;
        let health = service
            .get_health(Request::new(Empty {}))
            .await
            .unwrap()
            .into_inner();
        assert!(
            health.detail.contains("capture_xruns=2"),
            "detail={}",
            health.detail
        );
        assert!(
            health.detail.contains("ring_overruns=3"),
            "detail={}",
            health.detail
        );
        assert!(
            health.detail.contains("watchdog_reboots=0"),
            "detail={}",
            health.detail
        );
        stop.send_replace(true);
        worker.await.unwrap().unwrap();
    })
    .await
    .unwrap();
}
