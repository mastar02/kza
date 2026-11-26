//! Room supervision and capture ownership across device and inference I/O.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nexa_config::RoomDefinition;
use nexa_proto::v1::{
    AudioUtterance, CaptureControlStatus, MicCaptureState, MicrophoneRoomStatus, UtteranceKind,
};
use tokio::sync::{Semaphore, mpsc, watch};

use crate::ambient_guard::{AmbientGuard, CaptureOutcome};
use crate::arbiter::{ArbitrationOutcome, WakeArbiter, WakeClaim};
use crate::capture_seal::{
    CaptureError, CaptureTransition, SharedCapture, change_capture, running_epoch,
};
use crate::dsp::{FRAME_SAMPLES, NoiseFloorTracker, SAMPLE_RATE, frame_duration, frame_rms};
use crate::endpoint::{EndpointDecision, EndpointTelemetry, PostWakeEndpoint};
use crate::grpc::AudioService;
use crate::ring::{CaptureConsumer, CaptureProducer, capture_ring};
use crate::traits::{
    AmpTransport, AudioError, CaptureSource, PlaybackOutput, ShadowStt, VadEngine, WakeEngine,
};

const FIRST_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);
const STABLE_CAPTURE: Duration = Duration::from_secs(5);
const FRAME_TIMEOUT: Duration = Duration::from_secs(2);
const AMBIENT_MIN_VOICE_FRAMES: usize = 10; // 800 ms of voiced frames.
const AMBIENT_MAX_FRAMES: usize = 150; // 12 s, including pauses.
const NON_WAKE_END_SILENCE_FRAMES: usize = 8; // 640 ms at the capture frame size.
const AMBIENT_COOLDOWN: Duration = Duration::from_secs(3);
const FOLLOW_UP_WINDOW: Duration = Duration::from_secs(12);
const FOLLOW_UP_MIN_VOICE_FRAMES: usize = 3;
const FOLLOW_UP_MAX_FRAMES: usize = 62; // Just under the 5 s command cap.
const WAKE_PREROLL_FRAMES: usize = 10; // 800 ms pre-roll so continuous speech after "Nexa" is not clipped.
const NEAR_WAKE_LOG_MIN: f32 = 0.15;
const NEAR_WAKE_LOG_INTERVAL: Duration = Duration::from_secs(3);

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
    room_closed: watch::Receiver<bool>,
}

impl CaptureGate {
    fn epoch(&self) -> Option<u64> {
        if *self.shutdown.borrow() || *self.room_closed.borrow() {
            return None;
        }
        running_epoch(&self.status.borrow())
    }

    async fn changed(&mut self) -> bool {
        tokio::select! {
            _ = self.room_closed.wait_for(|closed| *closed) => false,
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
    enabled: bool,
}

struct ShadowResult {
    recording: Recording,
    text: Result<String, AudioError>,
    latency_ms: u32,
}

impl<S: ShadowStt + 'static> SharedShadow<S> {
    pub fn new(model: S) -> Self {
        let enabled = model.is_enabled();
        Self {
            model: Arc::new(Mutex::new(model)),
            permit: Arc::new(Semaphore::new(1)),
            enabled,
        }
    }

    async fn transcribe(
        &self,
        mut gate: CaptureGate,
        epoch: u64,
        recording: Recording,
    ) -> Result<Option<ShadowResult>, AudioError> {
        if !self.enabled {
            if gate.epoch() != Some(epoch) {
                return Ok(None);
            }
            return Ok(Some(ShadowResult {
                recording,
                text: Ok(String::new()),
                latency_ms: 0,
            }));
        }
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
            if gate.epoch() != Some(epoch) {
                return Ok(None);
            }
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
    arbiter: WakeArbiter,
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
                arbiter: WakeArbiter::default(),
            },
        }
    }

    /// Share a multi-room wake arbiter across all configured room pipelines.
    #[must_use]
    pub fn with_arbiter(mut self, arbiter: WakeArbiter) -> Self {
        self.processor.arbiter = arbiter;
        self
    }

    pub async fn run<A: AmpTransport, P: PlaybackOutput, S: ShadowStt + 'static>(
        mut self,
        service: Arc<AudioService<A, P>>,
        shadow: Arc<SharedShadow<S>>,
        shutdown: watch::Receiver<bool>,
    ) -> Result<(), CaptureError> {
        let (_room_open, room_closed) = watch::channel(false);
        let mut gate = CaptureGate {
            control: service.capture_controller(),
            status: service.capture_status_receiver(),
            shutdown,
            room_closed,
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
            let (close_room, room_closed) = watch::channel(false);
            let room_gate = CaptureGate {
                room_closed,
                ..gate.clone()
            };
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
                    pump_frames(&mut self.source, producer, room_gate.clone(), epoch).await
                }
                .await;
                // Do not drain buffered PCM once the device has failed or stopped.
                close_room.send_replace(true);
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
            let processing = async {
                let result = process_frames(
                    &mut self.processor,
                    &service,
                    consumer,
                    utterances,
                    room_gate.clone(),
                    epoch,
                )
                .await;
                if result.is_err() {
                    close_room.send_replace(true);
                }
                result
            };
            let publish = publish_utterances(
                &service,
                &shadow,
                pending,
                room_gate.clone(),
                epoch,
                &room_id,
            );
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
    kind: UtteranceKind,
    samples: Vec<f32>,
    score: f32,
    time: u64,
    endpoint: Option<EndpointTelemetry>,
}

impl Recording {
    fn new(kind: UtteranceKind, score: f32) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            kind,
            samples: Vec::new(),
            score,
            time: unix_ms(),
            endpoint: None,
        }
    }

    fn into_utterance(
        self,
        room_id: &str,
        epoch: u64,
        text: String,
        latency_ms: u32,
    ) -> AudioUtterance {
        AudioUtterance {
            utterance_id: self.id.to_string(),
            room_id: room_id.to_owned(),
            kind: self.kind as i32,
            sample_rate: SAMPLE_RATE,
            rms: frame_rms(&self.samples),
            pcm_f32le: self
                .samples
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
            wake_score: self.score,
            parakeet_shadow_text: text,
            parakeet_latency_ms: latency_ms,
            captured_at_unix_ms: self.time,
            capture_epoch: epoch,
        }
    }

    fn log_endpoint(
        &self,
        room_id: &str,
        decision: EndpointDecision,
        parakeet_ms: u32,
        parakeet_ok: bool,
    ) {
        let stats = self.endpoint.unwrap_or_default();
        tracing::info!(
            room_id, utterance_id = %self.id, kind = self.kind.as_str_name(),
            ?decision, duration_ms = frame_duration(self.samples.len()).as_millis() as u64,
            rms = frame_rms(&self.samples),
            level_threshold = stats.level_threshold,
            effective_level_threshold = stats.effective_level_threshold,
            peak_voice_rms = stats.peak_voice_rms,
            tail_rms = stats.last_rms,
            voice_ms = stats.voice_ms,
            vad_voice_ms = stats.vad_voice_ms,
            parakeet_ms, parakeet_ok,
            "utterance endpoint"
        );
    }
}

struct NonWakeSegment {
    recording: Recording,
    voiced: usize,
    quiet: usize,
    started: Instant,
}

impl NonWakeSegment {
    fn new(kind: UtteranceKind, started: Instant) -> Self {
        Self {
            recording: Recording::new(kind, 0.0),
            voiced: 0,
            quiet: 0,
            started,
        }
    }

    fn observe(&mut self, frame: &[f32; FRAME_SAMPLES], speaking: bool) -> bool {
        self.recording.samples.extend_from_slice(frame);
        if speaking {
            self.voiced += 1;
            self.quiet = 0;
        } else {
            self.quiet += 1;
        }
        let cap = match self.recording.kind {
            UtteranceKind::FollowUp => FOLLOW_UP_MAX_FRAMES,
            _ => AMBIENT_MAX_FRAMES,
        };
        self.quiet >= NON_WAKE_END_SILENCE_FRAMES
            || self.recording.samples.len() >= cap * FRAME_SAMPLES
    }

    fn finish(mut self) -> Option<Recording> {
        let minimum = match self.recording.kind {
            // Preserve the command rule that an incomplete, capped clip is not actionable.
            UtteranceKind::FollowUp if self.quiet < NON_WAKE_END_SILENCE_FRAMES => return None,
            UtteranceKind::FollowUp => FOLLOW_UP_MIN_VOICE_FRAMES,
            _ => AMBIENT_MIN_VOICE_FRAMES,
        };
        let samples = &mut self.recording.samples;
        samples.truncate(samples.len() - self.quiet * FRAME_SAMPLES);
        (self.voiced >= minimum).then_some(self.recording)
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
        arbiter,
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
    let mut preroll: VecDeque<[f32; FRAME_SAMPLES]> = VecDeque::with_capacity(WAKE_PREROLL_FRAMES);
    let mut near_wake_logged_at: Option<Instant> = None;
    let mut current: Option<(PostWakeEndpoint, Recording, WakeClaim)> = None;
    let mut non_wake = None::<NonWakeSegment>;
    let mut ambient_demand = service.ambient_demand();
    let mut generation = ambient_demand.borrow_and_update().generation();
    let mut ambient_after = Instant::now();
    let mut follow_up_until = None;
    let zone = zone_id(&room.room_id);
    let mut frame = [0.0; FRAME_SAMPLES];
    let mut filled = 0;
    loop {
        let count = tokio::select! {
            biased;
            _ = gate.closed(epoch) => return Ok(()),
            Ok(()) = ambient_demand.changed() => {
                let next = ambient_demand.borrow_and_update().generation();
                if next != generation {
                    non_wake.take_if(|segment| segment.recording.kind == UtteranceKind::Ambient);
                    generation = next;
                }
                continue;
            }
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
        let mut inference_job = tokio::task::spawn_blocking(move || {
            let mut engines = models
                .lock()
                .map_err(|error| unavailable("lock room models for inference", error))?;
            Ok::<_, AudioError>((
                engines.0.speech_probability(&frame)?,
                engines.1.wake_score(&frame)?,
            ))
        });
        let (speech, score) = tokio::select! {
            biased;
            _ = gate.closed(epoch) => {
                drop((current, non_wake));
                inference_job.await.map_err(|error| unavailable("join room inference worker", error))??;
                return Ok(());
            }
            result = &mut inference_job => result,
        }
        .map_err(|error| unavailable("join room inference worker", error))??;
        let now = Instant::now();
        if let Some((mut endpoint, mut recording, claim)) = current.take() {
            let remaining = endpoint.remaining().as_millis() as usize * SAMPLE_RATE as usize / 1000;
            let used = remaining.min(frame.len());
            recording.samples.extend_from_slice(&frame[..used]);
            claim.observe_rms(rms);
            let decision = endpoint.observe(rms, Some(speech), frame_duration(used));
            if decision == EndpointDecision::Listening {
                current = Some((endpoint, recording, claim));
            } else if decision.dispatches() {
                recording.endpoint = Some(endpoint.telemetry());
                let final_rms = frame_rms(&recording.samples);
                match claim.finish(final_rms, now) {
                    ArbitrationOutcome::Won => {
                        guard.record(now, CaptureOutcome::Accepted);
                        follow_up_until = Some(now + FOLLOW_UP_WINDOW);
                        tokio::select! {
                            biased;
                            _ = gate.closed(epoch) => return Ok(()),
                            result = utterances.send(recording) => result.map_err(|error| unavailable("enqueue utterance", error))?,
                        }
                    }
                    ArbitrationOutcome::Yielded { winner_room } => {
                        tracing::info!(
                            room_id = %room.room_id,
                            utterance_id = %recording.id,
                            winner_room,
                            rms = final_rms,
                            peak_voice_rms = endpoint.telemetry().peak_voice_rms,
                            "utterance yielded to louder room"
                        );
                    }
                }
            } else {
                drop(claim);
                recording.endpoint = Some(endpoint.telemetry());
                recording.log_endpoint(&room.room_id, decision, 0, false);
                guard.record(now, CaptureOutcome::Rejected);
            }
            continue;
        }

        if speech < 0.5 {
            noise.observe(rms);
        }
        if score >= *wake_threshold && guard.allow_wake(now, score) {
            let floor = noise.floor();
            let endpoint = PostWakeEndpoint::new(floor);
            tracing::info!(
                room_id = %room.room_id,
                score,
                threshold = *wake_threshold,
                rms,
                floor,
                level_threshold = endpoint.level_threshold(),
                preroll_frames = preroll.len(),
                "wake word detected"
            );
            non_wake = None;
            follow_up_until = None;
            let mut recording = Recording::new(UtteranceKind::Command, score);
            for prior in preroll.drain(..) {
                recording.samples.extend_from_slice(&prior);
            }
            recording.samples.extend_from_slice(&frame);
            let claim = arbiter.begin(&room.room_id, now, score, rms);
            current = Some((endpoint, recording, claim));
            continue;
        }
        if score >= NEAR_WAKE_LOG_MIN
            && near_wake_logged_at.is_none_or(|at| now.duration_since(at) >= NEAR_WAKE_LOG_INTERVAL)
        {
            near_wake_logged_at = Some(now);
            tracing::info!(
                room_id = %room.room_id,
                score,
                threshold = *wake_threshold,
                rms,
                floor = noise.floor(),
                speech,
                "near-threshold wake score"
            );
        }
        let speaking = service.speaking(&zone).unwrap_or(true);
        if speaking {
            preroll.clear();
        } else {
            if preroll.len() == WAKE_PREROLL_FRAMES {
                preroll.pop_front();
            }
            preroll.push_back(frame);
        }

        let follow_up =
            follow_up_until.is_some_and(|expires| now < expires) && guard.allow_follow_up(now);
        let ambient = generation.is_some()
            && generation == ambient_demand.borrow().generation()
            && now >= ambient_after;
        // Barge-in may stop playback, but the turn's own output must not become a new clip.
        let kind = if speaking {
            None
        } else if follow_up {
            Some(UtteranceKind::FollowUp)
        } else if ambient {
            Some(UtteranceKind::Ambient)
        } else {
            None
        };
        if non_wake.as_ref().map(|segment| segment.recording.kind) != kind {
            non_wake = None;
        }
        let Some(kind) = kind else { continue };
        let voiced = (0.5..=1.0).contains(&speech) && rms >= noise.floor() * 1.8;
        if non_wake.is_none() && !voiced {
            continue;
        }
        let mut segment = non_wake
            .take()
            .unwrap_or_else(|| NonWakeSegment::new(kind, now));
        if !segment.observe(&frame, voiced) {
            non_wake = Some(segment);
            continue;
        }
        let latency_ms = segment.started.elapsed().as_millis();
        let Some(recording) = segment.finish() else {
            continue;
        };
        if kind == UtteranceKind::FollowUp {
            follow_up_until = None; // One candidate per completed wake command.
            tokio::select! {
                biased;
                _ = gate.closed(epoch) => return Ok(()),
                result = utterances.send(recording) => result.map_err(|error| unavailable("enqueue follow-up", error))?,
            }
        } else if let Some(generation) = generation {
            let utterance_id = recording.id;
            let duration_ms = frame_duration(recording.samples.len()).as_millis();
            let utterance = recording.into_utterance(&room.room_id, epoch, String::new(), 0);
            let room_closed = gate.room_closed.borrow();
            if *room_closed || *gate.shutdown.borrow() {
                return Ok(());
            }
            match service.publish_ambient(generation, utterance) {
                Ok(true) => {
                    ambient_after = now + AMBIENT_COOLDOWN;
                    tracing::info!(room_id = %room.room_id, %utterance_id, duration_ms, latency_ms,
                        status = "published", "Ambient segment");
                }
                Ok(false) => (),
                Err(error) => tracing::warn!(room_id = %room.room_id, %utterance_id, latency_ms,
                    status = "publication_failed", code = ?error.code(), "Ambient segment"),
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
        let utterance = recording.into_utterance(room_id, epoch, text, latency_ms);
        let room_closed = gate.room_closed.borrow();
        // The service checks the epoch and sends under the committed ingress gate.
        if !*room_closed
            && !*gate.shutdown.borrow()
            && let Err(error) = service.publish_utterance(utterance)
        {
            tracing::warn!(room_id, %utterance_id, %error, "utterance publication failed");
        }
    }
}
