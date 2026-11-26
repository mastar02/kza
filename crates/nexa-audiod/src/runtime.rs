//! Room supervision and capture ownership across device and inference I/O.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nexa_config::RoomDefinition;
use nexa_proto::v1::{
    AudioUtterance, CaptureControlStatus, MicCaptureState, MicrophoneRoomStatus, UtteranceKind,
};
use tokio::sync::{Semaphore, mpsc, watch};

use crate::ambient_guard::{AmbientGuard, CaptureOutcome};
use crate::capture_seal::{
    CaptureError, CaptureTransition, SharedCapture, change_capture, running_epoch,
};
use crate::dsp::{FRAME_SAMPLES, NoiseFloorTracker, SAMPLE_RATE, frame_duration, frame_rms};
use crate::endpoint::{EndpointDecision, PostWakeEndpoint};
use crate::grpc::AudioService;
use crate::ring::{CaptureConsumer, CaptureProducer, capture_ring};
use crate::traits::{
    AmpTransport, AudioError, CaptureSource, PlaybackOutput, ShadowStt, VadEngine, WakeEngine,
};

const FIRST_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);
const STABLE_CAPTURE: Duration = Duration::from_secs(5);
const FRAME_TIMEOUT: Duration = Duration::from_secs(2);

pub fn zone_id(room_id: &str) -> String {
    format!("zone_{room_id}")
}

pub(crate) fn unix_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis().try_into().unwrap_or(u64::MAX),
        Err(error) => {
            tracing::warn!(%error, "system clock precedes the Unix epoch");
            0
        }
    }
}

fn unavailable(operation: &'static str, error: impl std::fmt::Display) -> AudioError {
    tracing::warn!(operation, %error, "audio operation failed");
    AudioError::Unavailable
}

fn update_microphone<A: AmpTransport, P: PlaybackOutput>(
    service: &AudioService<A, P>,
    room: &MicrophoneRoomStatus,
) {
    if let Err(error) = service.update_microphone(room.clone()) {
        tracing::warn!(room_id = %room.room_id, %error, "microphone diagnostics update failed");
    }
}

/// Acknowledge sealed admission and resume policy. Live room owners are retired
/// separately, after device stop and all capture-dependent work have completed.
pub async fn run_capture_control(
    capture: SharedCapture,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), CaptureError> {
    if *shutdown.borrow() {
        return Ok(());
    }
    let mut transitions =
        change_capture(&capture, |control| Ok(control.watch_transitions())).await?;
    loop {
        let transition = *transitions.borrow_and_update();
        if let Some(transition) = transition {
            let result = change_capture(&capture, move |control| match transition {
                CaptureTransition::Seal { revision } => control.acknowledge_seal(revision),
                CaptureTransition::Resume { revision } => control.acknowledge_resumed(revision),
            })
            .await;
            match result {
                Ok(()) => {}
                Err(
                    error @ (CaptureError::RevisionConflict { .. }
                    | CaptureError::StaleAcknowledgment),
                ) => {
                    tracing::warn!(%error, "capture transition was superseded");
                }
                Err(error) => return Err(error),
            }
        }
        tokio::select! {
            result = shutdown.wait_for(|stopping| *stopping) => {
                if let Err(error) = result {
                    tracing::warn!(%error, "capture shutdown channel closed");
                }
                return Ok(());
            }
            result = transitions.changed() => if let Err(error) = result {
                tracing::warn!(%error, "capture transition channel closed");
                return Ok(());
            },
        }
    }
}

#[derive(Clone)]
struct CaptureGate {
    control: SharedCapture,
    status: watch::Receiver<CaptureControlStatus>,
    shutdown: watch::Receiver<bool>,
}

impl CaptureGate {
    fn epoch(&self) -> Option<u64> {
        if *self.shutdown.borrow() {
            return None;
        }
        running_epoch(&self.status.borrow())
    }

    async fn changed(&mut self) -> bool {
        tokio::select! {
            result = self.shutdown.wait_for(|stopping| *stopping) => {
                if let Err(error) = result {
                    tracing::warn!(%error, "room shutdown channel closed");
                }
                false
            }
            change = self.status.changed() => {
                if let Err(error) = change {
                    tracing::warn!(%error, "capture status channel closed");
                    return false;
                }
                true
            }
        }
    }

    async fn wait_for_epoch(&mut self) -> Option<u64> {
        loop {
            if let Some(epoch) = self.epoch() {
                return Some(epoch);
            }
            if !self.changed().await {
                return None;
            }
        }
    }

    async fn closed(&mut self, epoch: u64) {
        while self.epoch() == Some(epoch) && self.changed().await {}
    }
}

/// A single shared recognizer with at most one blocking worker, including while
/// an epoch is being sealed. Waiting rooms can cancel before entering the worker.
#[derive(Debug)]
pub struct SharedShadow<S> {
    model: Arc<Mutex<S>>,
    permit: Arc<Semaphore>,
}

struct ShadowResult {
    recording: Recording,
    text: Result<String, AudioError>,
    latency_ms: u32,
}

impl<S: ShadowStt + 'static> SharedShadow<S> {
    pub fn new(model: S) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            permit: Arc::new(Semaphore::new(1)),
        }
    }

    async fn transcribe(
        &self,
        mut gate: CaptureGate,
        epoch: u64,
        recording: Recording,
    ) -> Result<Option<ShadowResult>, AudioError> {
        let permit = tokio::select! {
            biased;
            _ = gate.closed(epoch) => return Ok(None),
            permit = Arc::clone(&self.permit).acquire_owned() =>
                permit.map_err(|error| unavailable("acquire Parakeet permit", error))?,
        };
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut model = model
                .lock()
                .map_err(|error| unavailable("lock Parakeet model", error))?;
            let started = Instant::now();
            let text = model.transcribe(&recording.samples);
            let latency_ms = started.elapsed().as_millis().try_into().unwrap_or(u32::MAX);
            Ok(Some(ShadowResult {
                recording,
                text,
                latency_ms,
            }))
        })
        .await
        .map_err(|error| unavailable("join Parakeet worker", error))?
    }
}

#[derive(Debug)]
pub struct RoomPipeline<C, V, W> {
    source: C,
    processor: RoomProcessor<V, W>,
}

#[derive(Debug)]
struct RoomProcessor<V, W> {
    room: MicrophoneRoomStatus,
    inference: Arc<Mutex<(V, W)>>,
    wake_threshold: f32,
}

impl<C, V, W> RoomPipeline<C, V, W>
where
    C: CaptureSource,
    V: VadEngine + 'static,
    W: WakeEngine + 'static,
{
    pub fn new(room: &RoomDefinition, source: C, vad: V, wake: W, wake_threshold: f32) -> Self {
        Self {
            source,
            processor: RoomProcessor {
                room: MicrophoneRoomStatus {
                    room_id: room.room_id.clone(),
                    room_name: room.name.clone(),
                    usb_port: room.mic_usb_port.clone().unwrap_or_default(),
                    state: MicCaptureState::Waiting as i32,
                    ..Default::default()
                },
                inference: Arc::new(Mutex::new((vad, wake))),
                wake_threshold,
            },
        }
    }

    pub async fn run<A: AmpTransport, P: PlaybackOutput, S: ShadowStt + 'static>(
        mut self,
        service: Arc<AudioService<A, P>>,
        shadow: Arc<SharedShadow<S>>,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(), CaptureError> {
        let mut gate = CaptureGate {
            control: service.capture_controller(),
            status: service.capture_status_receiver(),
            shutdown,
        };
        let mut retry = FIRST_RETRY;
        let mut attempt = 0_u32;
        loop {
            if gate.epoch().is_none() {
                self.processor.room.state = MicCaptureState::Muted as i32;
                self.processor.room.last_rms = 0.0;
                update_microphone(&service, &self.processor.room);
                retry = FIRST_RETRY;
                attempt = 0;
            }
            let Some(epoch) = gate.wait_for_epoch().await else {
                return Ok(());
            };
            let shutdown = gate.shutdown.clone();
            let owner = match change_capture(&gate.control, move |control| {
                if *shutdown.borrow() || control.running_epoch() != Some(epoch) {
                    return Err(CaptureError::Sealed);
                }
                control.begin_capture()
            })
            .await
            {
                Err(CaptureError::Sealed) => continue,
                result => result?,
            };

            let started = Instant::now();
            let (producer, consumer) = capture_ring();
            let (utterances, pending) = mpsc::channel(1);
            self.processor.room.state = MicCaptureState::Waiting as i32;
            update_microphone(&service, &self.processor.room);

            let room_id = self.processor.room.room_id.clone();
            let capture = async {
                let mut device = None;
                let result = async {
                    if gate.epoch() != Some(epoch) {
                        return Ok(());
                    }
                    let result = self.source.start().await;
                    if attempt > 0 {
                        match &result {
                            Ok(()) => tracing::info!(%room_id, attempt, result = "ok", "XVF3800 capture reopen"),
                            Err(error) => tracing::warn!(%room_id, attempt, result = "error", error = %error.cause(), "XVF3800 capture reopen"),
                        }
                    }
                    result?;
                    let (card, channels) = self.source.device();
                    tracing::info!(%room_id, card, channels, "room capture opened");
                    device = Some((card.to_owned(), channels));
                    change_capture(&gate.control, move |control| {
                        control.acknowledge_started(owner)
                    })
                    .await
                    .map_err(|error| unavailable("acknowledge capture start", error))?;
                    pump_frames(&mut self.source, producer, gate.clone(), epoch).await
                }
                .await;
                // Even a failed start can have performed partial hardware I/O.
                let mut delay = FIRST_RETRY;
                while let Err(error) = self.source.stop().await {
                    tracing::warn!(%room_id, error = %error.cause(), "capture stop failed; ownership retained");
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(MAX_RETRY);
                }
                if let Some((card, channels)) = device {
                    tracing::info!(%room_id, %card, channels, "room capture closed");
                }
                result
            };
            let processing = process_frames(
                &mut self.processor,
                &service,
                consumer,
                utterances,
                gate.clone(),
                epoch,
            );
            let publish =
                publish_utterances(&service, &shadow, pending, gate.clone(), epoch, &room_id);
            let (capture, processing, ()) = tokio::join!(capture, processing, publish);
            let retired = change_capture(&gate.control, move |control| {
                control.acknowledge_stopped(owner)
            })
            .await;
            for (operation, result) in [("capture", capture), ("processing", processing)] {
                if let Err(error) = result {
                    tracing::warn!(%room_id, operation, error = %error.cause(), "room audio failed");
                }
            }
            retired?;
            if gate.epoch() == Some(epoch) {
                self.processor.room.state = MicCaptureState::Recovering as i32;
                self.processor.room.last_rms = 0.0;
                self.processor.room.recoveries = self.processor.room.recoveries.saturating_add(1);
                update_microphone(&service, &self.processor.room);
                if started.elapsed() >= STABLE_CAPTURE {
                    retry = FIRST_RETRY;
                    attempt = 0;
                }
                attempt = attempt.saturating_add(1);
                tracing::info!(%room_id, attempt, retry_ms = retry.as_millis() as u64, "room capture recovery scheduled");
                tokio::select! {
                    _ = gate.closed(epoch) => (),
                    _ = tokio::time::sleep(retry) => retry = (retry * 2).min(MAX_RETRY),
                }
            }
        }
    }
}

async fn pump_frames(
    source: &mut impl CaptureSource,
    mut producer: CaptureProducer,
    mut gate: CaptureGate,
    epoch: u64,
) -> Result<(), AudioError> {
    let mut samples = [0.0; FRAME_SAMPLES];
    loop {
        tokio::select! {
            biased;
            _ = gate.closed(epoch) => return Ok(()),
            result = async {
                let count = tokio::time::timeout(FRAME_TIMEOUT, source.read(&mut samples))
                    .await.map_err(|error| unavailable("read capture frame", error))??;
                let samples = samples.get(..count).filter(|frame| !frame.is_empty())
                    .ok_or_else(|| unavailable("read capture frame", "empty or oversized frame"))?;
                producer.push(samples).await.map_err(|error| unavailable("enqueue capture frame", error))
            } => result?,
        }
    }
}

#[derive(Debug)]
struct Recording {
    id: uuid::Uuid,
    endpoint: PostWakeEndpoint,
    samples: Vec<f32>,
    score: f32,
    time: u64,
}

impl Recording {
    fn log_endpoint(
        &self,
        room_id: &str,
        decision: EndpointDecision,
        parakeet_ms: u32,
        parakeet_ok: bool,
    ) {
        tracing::info!(
            room_id, utterance_id = %self.id, kind = UtteranceKind::Command.as_str_name(),
            ?decision, duration_ms = frame_duration(self.samples.len()).as_millis() as u64,
            rms = frame_rms(&self.samples), parakeet_ms, parakeet_ok,
            "utterance endpoint"
        );
    }
}

async fn process_frames<A: AmpTransport, P: PlaybackOutput>(
    processor: &mut RoomProcessor<impl VadEngine + 'static, impl WakeEngine + 'static>,
    service: &AudioService<A, P>,
    mut consumer: CaptureConsumer,
    utterances: mpsc::Sender<Recording>,
    mut gate: CaptureGate,
    epoch: u64,
) -> Result<(), AudioError> {
    let RoomProcessor {
        room,
        inference,
        wake_threshold,
    } = processor;
    let reset = Arc::clone(inference);
    tokio::task::spawn_blocking(move || {
        let mut engines = reset
            .lock()
            .map_err(|error| unavailable("lock room models for reset", error))?;
        engines.0.reset();
        engines.1.reset();
        Ok::<_, AudioError>(())
    })
    .await
    .map_err(|error| unavailable("join room model reset", error))??;

    let mut noise = NoiseFloorTracker::default();
    let mut guard = AmbientGuard::default();
    let mut current: Option<Recording> = None;
    let mut frame = [0.0; FRAME_SAMPLES];
    let mut filled = 0;
    loop {
        let count = tokio::select! {
            biased;
            _ = gate.closed(epoch) => return Ok(()),
            count = consumer.read(&mut frame[filled..]) => count,
        };
        if count == 0 {
            return Ok(());
        }
        filled += count;
        if filled < frame.len() {
            continue;
        }
        filled = 0;

        let rms = frame_rms(&frame);
        if !rms.is_finite() {
            return Err(AudioError::Rejected);
        }
        if let Err(error) = service.observe_capture_frame(&room.room_id, epoch, &frame) {
            tracing::warn!(room_id = %room.room_id, %error, "barge-in observation failed");
        }
        room.state = MicCaptureState::Receiving as i32;
        room.last_rms = rms;
        room.frames_captured = room.frames_captured.saturating_add(1);
        room.last_frame_unix_ms = unix_ms();
        update_microphone(service, room);

        // Await an in-flight model call on seal before retiring the capture owner.
        let models = Arc::clone(inference);
        let (speech, score) = tokio::task::spawn_blocking(move || {
            let mut engines = models
                .lock()
                .map_err(|error| unavailable("lock room models for inference", error))?;
            Ok::<_, AudioError>((
                engines.0.speech_probability(&frame)?,
                engines.1.wake_score(&frame)?,
            ))
        })
        .await
        .map_err(|error| unavailable("join room inference worker", error))??;
        let now = Instant::now();
        if let Some(mut recording) = current.take() {
            let remaining =
                recording.endpoint.remaining().as_millis() as usize * SAMPLE_RATE as usize / 1000;
            let used = remaining.min(frame.len());
            recording.samples.extend_from_slice(&frame[..used]);
            let decision = recording
                .endpoint
                .observe(rms, Some(speech), frame_duration(used));
            if decision == EndpointDecision::Listening {
                current = Some(recording);
            } else if decision.dispatches() {
                guard.record(now, CaptureOutcome::Accepted);
                tokio::select! {
                    biased;
                    _ = gate.closed(epoch) => return Ok(()),
                    result = utterances.send(recording) => result.map_err(|error| unavailable("enqueue utterance", error))?,
                }
            } else {
                recording.log_endpoint(&room.room_id, decision, 0, false);
                guard.record(now, CaptureOutcome::Rejected);
            }
        } else {
            if speech < 0.5 {
                noise.observe(rms);
            }
            if score >= *wake_threshold && guard.allow_wake(now, score) {
                tracing::info!(room_id = %room.room_id, score, threshold = *wake_threshold, "wake word detected");
                current = Some(Recording {
                    id: uuid::Uuid::new_v4(),
                    endpoint: PostWakeEndpoint::new(noise.floor()),
                    samples: Vec::with_capacity(FRAME_SAMPLES * 63),
                    score,
                    time: unix_ms(),
                });
            }
        }
    }
}

async fn publish_utterances<A: AmpTransport, P: PlaybackOutput, S: ShadowStt + 'static>(
    service: &AudioService<A, P>,
    shadow: &SharedShadow<S>,
    mut pending: mpsc::Receiver<Recording>,
    mut gate: CaptureGate,
    epoch: u64,
    room_id: &str,
) {
    loop {
        let recording = tokio::select! {
            biased;
            _ = gate.closed(epoch) => return,
            recording = pending.recv() => recording,
        };
        let Some(recording) = recording else { return };
        let utterance_id = recording.id;
        let ShadowResult {
            recording,
            text,
            latency_ms,
        } = match shadow.transcribe(gate.clone(), epoch, recording).await {
            Ok(Some(result)) => result,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(room_id, %utterance_id, error = %error.cause(), "Parakeet worker unavailable");
                continue;
            }
        };
        recording.log_endpoint(
            room_id,
            EndpointDecision::Complete,
            latency_ms,
            text.is_ok(),
        );
        let text = match text {
            Ok(text) => text,
            Err(error) => {
                tracing::warn!(
                    room_id, %utterance_id, parakeet_ms = latency_ms,
                    error = %error.cause(), "Parakeet inference failed"
                );
                String::new()
            }
        };
        // The service checks the epoch and sends under the committed ingress gate.
        if gate.epoch() == Some(epoch)
            && let Err(error) = service.publish_utterance(AudioUtterance {
                utterance_id: utterance_id.to_string(),
                room_id: room_id.to_owned(),
                kind: UtteranceKind::Command as i32,
                sample_rate: SAMPLE_RATE,
                pcm_f32le: recording
                    .samples
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect(),
                wake_score: recording.score,
                rms: frame_rms(&recording.samples),
                parakeet_shadow_text: text,
                parakeet_latency_ms: latency_ms,
                captured_at_unix_ms: recording.time,
                capture_epoch: epoch,
            })
        {
            tracing::warn!(room_id, %utterance_id, %error, "utterance publication failed");
        }
    }
}
