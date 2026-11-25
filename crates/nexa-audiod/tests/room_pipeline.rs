#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nexa_audiod::capture_seal::{CaptureControl, CapturePhase, CaptureState};
use nexa_audiod::grpc::AudioService;
use nexa_audiod::hw::{amp::Ma1260, simulation::Simulation};
use nexa_audiod::runtime::{RoomPipeline, SharedShadow, run_capture_control};
use nexa_audiod::traits::{AudioError, CaptureSource, ShadowStt, VadEngine, WakeEngine};
use nexa_config::{Ma1260Connection, RoomDefinition, XvfTuning};
use nexa_proto::v1::audio_service_server::AudioService as AudioRpc;
use nexa_proto::v1::{Empty, SubscribeUtterancesRequest};
use tokio::sync::{Notify, watch};
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
        Ok(if samples[0] > 0.1 { "living" } else { "hall" }.into())
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
            assert_eq!(utterance.pcm_f32le.len(), 13 * 1280 * 4);
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
