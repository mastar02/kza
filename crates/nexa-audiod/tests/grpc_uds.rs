#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use hyper_util::rt::TokioIo;
use nexa_audiod::capture_seal::{CaptureControl, CapturePhase, CaptureState, CaptureTransition};
use nexa_audiod::dsp::FRAME_SAMPLES;
use nexa_audiod::grpc::AudioService;
use nexa_audiod::ma1260::CommandFrame;
use nexa_audiod::traits::{
    AmpTransport, AudioError, PlaybackOutput, PlaybackSession, PlaybackSpec,
};
use nexa_proto::v1::audio_service_client::AudioServiceClient;
use nexa_proto::v1::audio_service_server::AudioServiceServer;
use nexa_proto::v1::{
    ActorContext, AudioUtterance, CancelPlaybackRequest, Empty, ForgetCarriedOwnersRequest,
    HealthState, MicCaptureState, MicrophoneRoomStatus, Permission, PlayAudioChunk, Role,
    SetCaptureMuteRequest, SetZoneVolumeRequest, SubscribeUtterancesRequest, UtteranceKind,
    ZoneState,
};
use tempfile::TempDir;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::Code;
use tonic::transport::{Channel, Endpoint, Server};
use tower::service_fn;

mod common;

use common::wav;

const TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Default, Clone)]
struct FakeAmp(Arc<Mutex<Vec<[u8; 5]>>>);

impl AmpTransport for FakeAmp {
    async fn send(&self, frame: CommandFrame) -> Result<(), AudioError> {
        self.0.lock().unwrap().push(frame.encode());
        Ok(())
    }
}

#[derive(Debug, Default)]
struct OutputEvents {
    samples: Vec<f32>,
    finished: usize,
    cancelled: usize,
}

#[derive(Debug, Default, Clone)]
struct FakeOutput {
    events: Arc<Mutex<OutputEvents>>,
    wrote: Arc<Notify>,
    fail: Arc<Mutex<Option<Failure>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Failure {
    Open,
    Write,
    Finish,
}

impl FakeOutput {
    fn fail_at(&self, stage: Failure) -> Result<(), AudioError> {
        let mut failure = self.fail.lock().unwrap();
        if *failure == Some(stage) {
            failure.take();
            return Err(AudioError::Adapter("injected output failure".into()));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct FakeSession {
    output: FakeOutput,
    finished: bool,
}

impl PlaybackOutput for FakeOutput {
    type Session = FakeSession;

    async fn open(&self, spec: &PlaybackSpec) -> Result<Self::Session, AudioError> {
        assert_eq!(spec.zone_id, "living-speakers");
        assert_eq!(spec.sample_rate, 24_000);
        self.fail_at(Failure::Open)?;
        Ok(FakeSession {
            output: self.clone(),
            finished: false,
        })
    }
}

impl PlaybackSession for FakeSession {
    async fn write(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        self.output.fail_at(Failure::Write)?;
        self.output.events.lock().unwrap().samples.extend(samples);
        self.output.wrote.notify_one();
        Ok(())
    }

    async fn finish(&mut self) -> Result<(), AudioError> {
        self.output.fail_at(Failure::Finish)?;
        self.finished = true;
        self.output.events.lock().unwrap().finished += 1;
        Ok(())
    }
}

impl Drop for FakeSession {
    fn drop(&mut self) {
        if !self.finished {
            self.output.events.lock().unwrap().cancelled += 1;
        }
    }
}

struct Harness {
    client: AudioServiceClient<Channel>,
    service: Arc<AudioService<FakeAmp, FakeOutput>>,
    control: Arc<Mutex<CaptureControl>>,
    amp: FakeAmp,
    output: FakeOutput,
    server: JoinHandle<Result<(), tonic::transport::Error>>,
    _temp: TempDir,
}

impl Harness {
    async fn start(control: CaptureControl) -> Self {
        // Keep the path short on macOS, where sun_path is limited to 104 bytes.
        let temp = tempfile::Builder::new()
            .prefix("nexa-")
            .tempdir_in("/tmp")
            .unwrap();
        let path = temp.path().join("audio.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let control = Arc::new(Mutex::new(control));
        let amp = FakeAmp::default();
        let output = FakeOutput::default();
        let service = Arc::new(
            AudioService::new(
                Arc::clone(&control),
                amp.clone(),
                output.clone(),
                [ZoneState {
                    zone_id: "living-speakers".into(),
                    room_id: "living".into(),
                    ma1260_zone: 2,
                    ..ZoneState::default()
                }],
            )
            .unwrap(),
        );
        let rpc = AudioServiceServer::from_arc(Arc::clone(&service));
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(rpc)
                .serve_with_incoming(UnixListenerStream::new(listener))
                .await
        });
        let channel = Endpoint::from_static("http://[::]:50051")
            .connect_timeout(TIMEOUT)
            .timeout(TIMEOUT)
            .connect_with_connector(service_fn(move |_| {
                let path = path.clone();
                async { UnixStream::connect(path).await.map(TokioIo::new) }
            }))
            .await
            .unwrap();
        Self {
            client: AudioServiceClient::new(channel),
            service,
            control,
            amp,
            output,
            server,
            _temp: temp,
        }
    }

    async fn shutdown(self) {
        self.server.abort();
        assert!(self.server.await.unwrap_err().is_cancelled());
    }
}

fn running_capture() -> CaptureControl {
    let mut capture = CaptureControl::restore(CaptureState::default(), true).unwrap();
    capture.acknowledge_resumed(0).unwrap();
    capture
}

fn actor(role: Role, permission: Permission) -> ActorContext {
    ActorContext {
        subject: "test-session".into(),
        role: role as i32,
        permissions: vec![permission as i32],
        ..ActorContext::default()
    }
}

fn admin() -> Option<ActorContext> {
    Some(actor(Role::Operator, Permission::AdminManage))
}

fn mute(expected_revision: u64) -> SetCaptureMuteRequest {
    SetCaptureMuteRequest {
        muted: true,
        expected_revision,
        actor: admin(),
    }
}

fn forget(expected_carried: u32, no_capture_running: bool) -> ForgetCarriedOwnersRequest {
    ForgetCarriedOwnersRequest {
        expected_carried,
        no_capture_running,
        actor: admin(),
    }
}

fn chunk(token: &str, is_final: bool) -> PlayAudioChunk {
    PlayAudioChunk {
        playback_id: "test-playback".into(),
        zone_id: "living-speakers".into(),
        sample_rate: 24_000,
        pcm_f32le: [-0.25f32, 0.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        priority: 1,
        barge_in_token: token.into(),
        is_final,
    }
}

#[tokio::test]
async fn health_microphones_and_concurrent_revision_cas_over_uds() {
    let mut harness = Harness::start(running_capture()).await;
    let client = &mut harness.client;
    let health = client.get_health(Empty {}).await.unwrap().into_inner();
    assert_eq!(health.service, "nexa-audiod");
    assert_eq!(health.state(), HealthState::Ok);
    assert!(
        client
            .get_microphones(Empty {})
            .await
            .unwrap()
            .into_inner()
            .stale
    );
    harness
        .service
        .update_microphones(vec![MicrophoneRoomStatus {
            room_id: "living".into(),
            state: MicCaptureState::Receiving as i32,
            ..MicrophoneRoomStatus::default()
        }])
        .unwrap();

    let mut transitions = harness.control.lock().unwrap().watch_transitions();
    let mut concurrent = client.clone();
    let (first, second) = tokio::join!(
        client.set_capture_mute(mute(0)),
        concurrent.set_capture_mute(mute(0)),
    );
    let (accepted, rejected) = if first.is_ok() {
        (first, second)
    } else {
        (second, first)
    };
    let status = accepted.unwrap().into_inner();
    assert_eq!(status.phase(), CapturePhase::Muting);
    assert_eq!(status.revision, 1);
    assert!(status.ingress_sealed && status.persisted_muted && !status.total_mute_confirmed);
    assert_eq!(rejected.unwrap_err().code(), Code::Aborted);
    assert_eq!(
        client
            .get_health(Empty {})
            .await
            .unwrap()
            .into_inner()
            .state(),
        HealthState::Degraded
    );

    timeout(TIMEOUT, transitions.changed())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *transitions.borrow_and_update(),
        Some(CaptureTransition::Seal { revision: 1 })
    );

    // Quiet or missing frames are unrelated to explicit stop acknowledgments.
    assert_eq!(
        harness
            .service
            .observe_capture_frame("living", 0, &[0.0; FRAME_SAMPLES])
            .unwrap(),
        0
    );
    assert!(
        !client
            .get_capture_control(Empty {})
            .await
            .unwrap()
            .into_inner()
            .total_mute_confirmed
    );
    harness.control.lock().unwrap().acknowledge_seal(1).unwrap();
    assert!(
        client
            .get_capture_control(Empty {})
            .await
            .unwrap()
            .into_inner()
            .total_mute_confirmed
    );
    let mics = client.get_microphones(Empty {}).await.unwrap().into_inner();
    assert!(!mics.stale);
    assert_eq!(mics.microphones[0].state(), MicCaptureState::Muted);
    harness.shutdown().await;
}

#[tokio::test]
async fn forget_requires_authority_affirmation_and_the_exact_current_count_over_uds() {
    let mut original = running_capture();
    original.begin_capture().unwrap();
    original.begin_capture().unwrap();
    original.set_muted(true, 0).unwrap();
    let mut restored = CaptureControl::restore(original.durable_state().clone(), true).unwrap();
    restored.acknowledge_seal(1).unwrap();
    let mut harness = Harness::start(restored).await;
    let client = &mut harness.client;

    let untrusted = SetCaptureMuteRequest {
        actor: None,
        ..mute(1)
    };
    assert_eq!(
        client.set_capture_mute(untrusted).await.unwrap_err().code(),
        Code::Unauthenticated
    );
    let home = ForgetCarriedOwnersRequest {
        actor: Some(actor(Role::Home, Permission::AdminManage)),
        ..forget(2, true)
    };
    assert_eq!(
        client.forget_carried_owners(home).await.unwrap_err().code(),
        Code::PermissionDenied
    );
    assert_eq!(
        client
            .forget_carried_owners(forget(2, false))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    assert_eq!(
        client
            .forget_carried_owners(forget(1, true))
            .await
            .unwrap_err()
            .code(),
        Code::Aborted
    );
    let snapshot = client
        .get_capture_control(Empty {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(snapshot.carried_capture_owners, 2);
    assert!(!snapshot.total_mute_confirmed);

    let changed = client
        .forget_carried_owners(forget(2, true))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(changed.forgotten_capture_owners, 2);
    let status = changed.status.unwrap();
    assert_eq!(status.carried_capture_owners, 0);
    assert!(status.total_mute_confirmed);
    assert_eq!(
        client
            .forget_carried_owners(forget(2, true))
            .await
            .unwrap_err()
            .code(),
        Code::Aborted
    );
    assert_eq!(
        client
            .forget_carried_owners(forget(0, true))
            .await
            .unwrap()
            .into_inner()
            .forgotten_capture_owners,
        0
    );
    harness.shutdown().await;
}

#[tokio::test]
async fn sustained_audio_cancels_only_the_active_matching_playback_over_uds() {
    let mut harness = Harness::start(running_capture()).await;
    let (tx, rx) = mpsc::channel(2);
    let mut streaming = harness.client.clone();
    let call =
        tokio::spawn(async move { streaming.play_audio_stream(ReceiverStream::new(rx)).await });
    tx.send(chunk("current-token", false)).await.unwrap();
    timeout(TIMEOUT, harness.output.wrote.notified())
        .await
        .unwrap();
    assert!(
        harness
            .client
            .get_zones(Empty {})
            .await
            .unwrap()
            .into_inner()
            .zones[0]
            .speaking
    );

    harness
        .client
        .cancel_playback(CancelPlaybackRequest {
            zone_id: "living-speakers".into(),
            barge_in_token: "prior-token".into(),
        })
        .await
        .unwrap();

    let samples = wav("barge_in_burst_1_2s.wav");
    let loud = &samples[4 * FRAME_SAMPLES..5 * FRAME_SAMPLES];
    assert_eq!(
        harness
            .service
            .observe_capture_frame("living", 0, loud)
            .unwrap(),
        0
    );
    for _ in 0..3 {
        assert_eq!(
            harness
                .service
                .observe_capture_frame("hall", 0, loud)
                .unwrap(),
            0
        );
        assert_eq!(
            harness
                .service
                .observe_capture_frame("living", 99, loud)
                .unwrap(),
            0
        );
    }

    harness.client.set_capture_mute(mute(0)).await.unwrap();
    assert_eq!(
        harness
            .service
            .observe_capture_frame("living", 0, loud)
            .unwrap(),
        0
    );
    harness.control.lock().unwrap().acknowledge_seal(1).unwrap();
    harness
        .client
        .set_capture_mute(SetCaptureMuteRequest {
            muted: false,
            ..mute(1)
        })
        .await
        .unwrap();
    harness
        .control
        .lock()
        .unwrap()
        .acknowledge_resumed(2)
        .unwrap();
    let observed: Vec<_> = samples[3 * FRAME_SAMPLES..]
        .chunks(FRAME_SAMPLES)
        .map(|frame| {
            harness
                .service
                .observe_capture_frame("living", 2, frame)
                .unwrap()
        })
        .collect();
    assert_eq!(observed.iter().position(|&count| count != 0), Some(2));
    assert_eq!(observed.iter().sum::<usize>(), 1);
    let receipt = timeout(TIMEOUT, call)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_inner();
    assert_eq!(receipt.playback_id, "test-playback");
    assert!(!receipt.completed);
    assert!(receipt.interrupted_by_barge_in);
    assert!(
        !harness
            .client
            .get_zones(Empty {})
            .await
            .unwrap()
            .into_inner()
            .zones[0]
            .speaking
    );
    {
        let events = harness.output.events.lock().unwrap();
        assert_eq!(events.samples, [-0.25, 0.5]);
        assert_eq!(events.finished, 0);
        assert_eq!(events.cancelled, 1);
    }
    drop(tx);
    harness.shutdown().await;
}

#[tokio::test]
async fn volume_roundtrip_and_final_pcm_reach_injected_outputs_over_uds() {
    let mut harness = Harness::start(running_capture()).await;
    let client = &mut harness.client;
    let zone = client
        .set_zone_volume(SetZoneVolumeRequest {
            zone_id: "living-speakers".into(),
            volume_pct: 75,
            actor: Some(actor(Role::Home, Permission::HomeControl)),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(zone.volume_pct, 76);
    assert_eq!(*harness.amp.0.lock().unwrap(), [[0x02, 2, 0x02, 29, 0x03]]);

    let receipt = client
        .play_audio_stream(tokio_stream::iter([chunk("once", true)]))
        .await
        .unwrap()
        .into_inner();
    assert!(receipt.completed);
    assert!(!receipt.interrupted_by_barge_in);
    assert_eq!(harness.output.events.lock().unwrap().finished, 1);
    harness.shutdown().await;
}

#[tokio::test]
async fn playback_adapter_errors_retire_the_zone_and_increment_health_over_uds() {
    let mut harness = Harness::start(running_capture()).await;
    for (index, stage) in [Failure::Open, Failure::Write, Failure::Finish]
        .into_iter()
        .enumerate()
    {
        *harness.output.fail.lock().unwrap() = Some(stage);
        let error = harness
            .client
            .play_audio_stream(tokio_stream::iter([chunk("test-token", true)]))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::Unavailable);
        assert_eq!(error.message(), "audio adapter unavailable");
        assert!(
            !harness
                .client
                .get_zones(Empty {})
                .await
                .unwrap()
                .into_inner()
                .zones[0]
                .speaking
        );
        let health = harness
            .client
            .get_health(Empty {})
            .await
            .unwrap()
            .into_inner();
        assert!(
            health
                .detail
                .contains(&format!("playbacks_failed={}", index + 1))
        );

        let retry = harness
            .client
            .play_audio_stream(tokio_stream::iter([chunk("retry-token", true)]))
            .await
            .unwrap()
            .into_inner();
        assert!(retry.completed);
    }
    {
        let events = harness.output.events.lock().unwrap();
        assert_eq!(events.cancelled, 2);
        assert_eq!(events.finished, 3);
    }
    harness.shutdown().await;
}

#[tokio::test]
async fn utterance_subscription_applies_kind_and_capture_epoch_over_uds() {
    let mut harness = Harness::start(running_capture()).await;
    let client = &mut harness.client;
    let mut utterances = client
        .subscribe_utterances(SubscribeUtterancesRequest {
            include_ambient: false,
        })
        .await
        .unwrap()
        .into_inner();
    let event = |capture_epoch, kind| AudioUtterance {
        utterance_id: format!("epoch-{capture_epoch}"),
        capture_epoch,
        kind: kind as i32,
        ..AudioUtterance::default()
    };
    assert!(
        harness
            .service
            .publish_utterance(event(0, UtteranceKind::Ambient))
            .unwrap()
    );
    assert!(
        harness
            .service
            .publish_utterance(event(0, UtteranceKind::Command))
            .unwrap()
    );
    let first = timeout(TIMEOUT, utterances.message())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.kind(), UtteranceKind::Command);
    client.set_capture_mute(mute(0)).await.unwrap();
    assert!(
        !harness
            .service
            .publish_utterance(event(0, UtteranceKind::Command))
            .unwrap()
    );
    harness.control.lock().unwrap().acknowledge_seal(1).unwrap();
    client
        .set_capture_mute(SetCaptureMuteRequest {
            muted: false,
            ..mute(1)
        })
        .await
        .unwrap();
    assert!(
        !harness
            .service
            .publish_utterance(event(2, UtteranceKind::Command))
            .unwrap()
    );
    harness
        .control
        .lock()
        .unwrap()
        .acknowledge_resumed(2)
        .unwrap();
    assert!(
        !harness
            .service
            .publish_utterance(event(0, UtteranceKind::Command))
            .unwrap()
    );
    assert!(
        harness
            .service
            .publish_utterance(event(2, UtteranceKind::Command))
            .unwrap()
    );
    let resumed = timeout(TIMEOUT, utterances.message())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(resumed.capture_epoch, 2);
    drop(utterances);
    harness.shutdown().await;
}
