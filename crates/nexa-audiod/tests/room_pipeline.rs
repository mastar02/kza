#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nexa_audiod::capture_seal::{CaptureControl, CapturePhase, CaptureState};
use nexa_audiod::dsp::FRAME_SAMPLES;
use nexa_audiod::grpc::AudioService;
use nexa_audiod::hw::{amp::Ma1260, simulation::Simulation};
use nexa_audiod::runtime::{RoomPipeline, SharedShadow, run_capture_control};
use nexa_audiod::traits::{AudioError, CaptureSource, ShadowStt, VadEngine, WakeEngine};
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

    async fn read(&mut self, samples: &mut [f32]) -> Result<usize, AudioError> {
        if self.broken {
            return Err(AudioError::Unavailable);
        }
        if self.frame == 15 * self.utterances {
            return std::future::pending().await;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
        samples.fill(match self.frame % 15 {
            0 => 0.8,
            1..=4 => self.sample,
            _ => 0.0,
        });
        self.frame += 1;
        Ok(samples.len())
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

    async fn read(&mut self, samples: &mut [f32]) -> Result<usize, AudioError> {
        samples.fill(self.0.recv().await.ok_or(AudioError::Unavailable)??);
        Ok(samples.len())
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
        assert_eq!(
            utterance.pcm_f32le,
            0.3_f32.to_le_bytes().repeat(10 * FRAME_SAMPLES)
        );
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
        assert_eq!(resumed.pcm_f32le, 0.4_f32.to_le_bytes().repeat(10 * 1280));
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
        assert_eq!(
            capped.pcm_f32le,
            0.3_f32.to_le_bytes().repeat(150 * FRAME_SAMPLES)
        );
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
