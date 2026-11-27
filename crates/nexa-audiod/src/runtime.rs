//! Room supervision and capture ownership across device and inference I/O.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::hw::xvf::DspTelemetry;
use crate::ring::{CaptureConsumer, CaptureProducer, capture_ring_with_frames};
use crate::traits::{
    AmpTransport, AudioError, CaptureSource, PlaybackOutput, ShadowStt, VadEngine, WakeEngine,
};

const FIRST_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);
const STABLE_CAPTURE: Duration = Duration::from_secs(5);
const FRAME_TIMEOUT: Duration = Duration::from_secs(2);
const CAPTURE_RING_FRAMES: usize = 24;
const DEAD_ISOCH_ZERO_FRAMES: usize = 75;
const MAX_WATCHDOG_REBOOTS_PER_HOUR: usize = 4;
const WATCHDOG_WINDOW: Duration = Duration::from_secs(3600);
const DSP_PROBE_INTERVAL: Duration = Duration::from_millis(500);
const AMBIENT_MIN_VOICE_FRAMES: usize = 10; // 800 ms of voiced frames.
const AMBIENT_MAX_FRAMES: usize = 150; // 12 s, including pauses.
const NON_WAKE_END_SILENCE_FRAMES: usize = 8; // 640 ms at the capture frame size.
const AMBIENT_SILENCE_COOLDOWN: Duration = Duration::from_millis(500);
const AMBIENT_CAP_COOLDOWN: Duration = Duration::ZERO;
const FOLLOW_UP_WINDOW: Duration = Duration::from_secs(12);
const FOLLOW_UP_MIN_VOICE_FRAMES: usize = 3;
const FOLLOW_UP_MAX_FRAMES: usize = 62; // Just under the 5 s command cap.
const FOLLOW_UP_PREROLL_FRAMES: usize = 4; // 320 ms onset pre-roll (parity with legacy follow_up_preroll_s).
const AMBIENT_PREROLL_FRAMES: usize = 6; // 480 ms onset pre-roll (parity with legacy ambient preroll_ms).
const WAKE_PREROLL_FRAMES: usize = 15; // 1.2 s pre-roll so continuous commands finishing as wake fires are preserved.
const PREROLL_MIN_VOICED_FRAMES: usize = 3; // 240 ms of speech inside pre-roll.
const PREROLL_VAD_THRESHOLD: f32 = 0.35;
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
    follow_up_window: Duration,
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
                follow_up_window: FOLLOW_UP_WINDOW,
                arbiter: WakeArbiter::default(),
            },
        }
    }

    /// Configure the post-command follow-up listening window duration.
    #[must_use]
    pub fn with_follow_up_window(mut self, window: Duration) -> Self {
        self.processor.follow_up_window = window;
        self
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
        let watchdog_reboots = Arc::new(Mutex::new(VecDeque::<Instant>::new()));
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
            let (producer, consumer) = capture_ring_with_frames(CAPTURE_RING_FRAMES);
            let (utterances, pending) = mpsc::channel(1);
            let capture_busy = Arc::new(AtomicBool::new(false));
            let latest_dsp = Arc::new(Mutex::new(None::<DspTelemetry>));
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
                    pump_frames(
                        &mut self.source,
                        producer,
                        room_gate.clone(),
                        epoch,
                        &room_id,
                        Arc::clone(&capture_busy),
                        Arc::clone(&watchdog_reboots),
                        Arc::clone(&latest_dsp),
                    )
                    .await
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
                    Arc::clone(&capture_busy),
                    Arc::clone(&latest_dsp),
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

#[allow(clippy::too_many_arguments)]
async fn pump_frames(
    source: &mut impl CaptureSource,
    mut producer: CaptureProducer,
    mut gate: CaptureGate,
    epoch: u64,
    room_id: &str,
    capture_busy: Arc<AtomicBool>,
    watchdog_reboots: Arc<Mutex<VecDeque<Instant>>>,
    latest_dsp: Arc<Mutex<Option<DspTelemetry>>>,
) -> Result<(), AudioError> {
    let mut samples = [0.0; FRAME_SAMPLES];
    let check_dead_isoch = source.supports_dead_isoch_watchdog();
    let mut zero_streak = 0_usize;
    let mut last_dsp_probe: Option<Instant> = None;
    loop {
        tokio::select! {
            biased;
            _ = gate.closed(epoch) => return Ok(()),
            result = async {
                let count = tokio::time::timeout(FRAME_TIMEOUT, source.read(&mut samples))
                    .await.map_err(|error| unavailable("read capture frame", error))??;
                let samples = samples.get(..count).filter(|frame| !frame.is_empty())
                    .ok_or_else(|| unavailable("read capture frame", "empty or oversized frame"))?;
                if !capture_busy.load(Ordering::Relaxed)
                    && last_dsp_probe.is_none_or(|prev| prev.elapsed() >= DSP_PROBE_INTERVAL)
                {
                    last_dsp_probe = Some(Instant::now());
                    if let Some(telemetry) = source.probe_dsp().await
                        && let Ok(mut slot) = latest_dsp.lock()
                    {
                        *slot = Some(telemetry);
                    }
                }
                if check_dead_isoch {
                    #[allow(clippy::float_cmp)]
                    if samples.iter().all(|&sample| sample == 0.0) {
                        zero_streak = zero_streak.saturating_add(1);
                    } else {
                        zero_streak = 0;
                    }
                    if zero_streak >= DEAD_ISOCH_ZERO_FRAMES
                        && !capture_busy.load(Ordering::Relaxed)
                    {
                        let should_reboot = watchdog_reboots
                            .lock()
                            .ok()
                            .is_some_and(|mut history| {
                                let now = Instant::now();
                                while history
                                    .front()
                                    .is_some_and(|ts| now.duration_since(*ts) >= WATCHDOG_WINDOW)
                                {
                                    history.pop_front();
                                }
                                if history.len() < MAX_WATCHDOG_REBOOTS_PER_HOUR {
                                    history.push_back(now);
                                    true
                                } else {
                                    false
                                }
                            });
                        if should_reboot {
                            tracing::warn!(
                                %room_id,
                                zero_streak,
                                "XVF3800 dead isochronous stream detected; rebooting microphone"
                            );
                            if let Err(error) = source.reboot().await {
                                tracing::warn!(
                                    %room_id,
                                    error = %error.cause(),
                                    "XVF3800 watchdog reboot failed"
                                );
                            }
                        } else {
                            tracing::warn!(
                                %room_id,
                                zero_streak,
                                "XVF3800 dead isochronous stream detected; reboot rate limit reached"
                            );
                        }
                        return Err(AudioError::Unavailable);
                    }
                }
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
    doa_azimuth_rad: f32,
    has_doa: bool,
    doa_stability: f32,
    speech_ratio: f32,
    continues_segment: bool,
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
            doa_azimuth_rad: 0.0,
            has_doa: false,
            doa_stability: 0.0,
            speech_ratio: 0.0,
            continues_segment: false,
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
            doa_azimuth_rad: self.doa_azimuth_rad,
            has_doa: self.has_doa,
            doa_stability: self.doa_stability,
            speech_ratio: self.speech_ratio,
            continues_segment: self.continues_segment,
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
            peak_vad = stats.peak_vad,
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
    total_frames: usize,
    quiet: usize,
    started: Instant,
    doa_sin_sum: f32,
    doa_cos_sum: f32,
    doa_samples: u32,
    continues_segment: bool,
}

impl NonWakeSegment {
    fn with_preroll<'a>(
        kind: UtteranceKind,
        started: Instant,
        preroll: impl IntoIterator<Item = &'a [f32; FRAME_SAMPLES]>,
        continues_segment: bool,
    ) -> Self {
        let mut recording = Recording::new(kind, 0.0);
        for frame in preroll {
            recording.samples.extend_from_slice(frame);
        }
        Self {
            recording,
            voiced: 0,
            total_frames: 0,
            quiet: 0,
            started,
            doa_sin_sum: 0.0,
            doa_cos_sum: 0.0,
            doa_samples: 0,
            continues_segment,
        }
    }

    fn observe(
        &mut self,
        frame: &[f32; FRAME_SAMPLES],
        speaking: bool,
        auto_beam_rad: Option<f32>,
    ) -> bool {
        self.recording.samples.extend_from_slice(frame);
        self.total_frames = self.total_frames.saturating_add(1);
        if speaking {
            self.voiced += 1;
            self.quiet = 0;
            if let Some(angle) = auto_beam_rad
                && angle.is_finite()
            {
                self.doa_sin_sum += angle.sin();
                self.doa_cos_sum += angle.cos();
                self.doa_samples = self.doa_samples.saturating_add(1);
            }
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

    fn finish(mut self) -> Option<(Recording, bool)> {
        let capped = self.quiet < NON_WAKE_END_SILENCE_FRAMES;
        let minimum = match self.recording.kind {
            // Preserve the command rule that an incomplete, capped clip is not actionable.
            UtteranceKind::FollowUp if capped => return None,
            UtteranceKind::FollowUp => FOLLOW_UP_MIN_VOICE_FRAMES,
            _ => AMBIENT_MIN_VOICE_FRAMES,
        };
        let samples = &mut self.recording.samples;
        samples.truncate(samples.len() - self.quiet * FRAME_SAMPLES);
        if self.voiced < minimum {
            return None;
        }
        self.recording.speech_ratio =
            (self.voiced as f32 / self.total_frames.max(1) as f32).clamp(0.0, 1.0);
        self.recording.continues_segment = self.continues_segment;
        if self.doa_samples > 0 {
            self.recording.doa_azimuth_rad = self.doa_sin_sum.atan2(self.doa_cos_sum);
            self.recording.doa_stability = (self.doa_sin_sum.hypot(self.doa_cos_sum)
                / self.doa_samples as f32)
                .clamp(0.0, 1.0);
            self.recording.has_doa = true;
        }
        Some((self.recording, capped))
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_frames<A: AmpTransport, P: PlaybackOutput>(
    processor: &mut RoomProcessor<impl VadEngine + 'static, impl WakeEngine + 'static>,
    service: &AudioService<A, P>,
    mut consumer: CaptureConsumer,
    utterances: mpsc::Sender<Recording>,
    mut gate: CaptureGate,
    epoch: u64,
    capture_busy: Arc<AtomicBool>,
    latest_dsp: Arc<Mutex<Option<DspTelemetry>>>,
) -> Result<(), AudioError> {
    let RoomProcessor {
        room,
        inference,
        wake_threshold,
        follow_up_window,
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
    let mut preroll: VecDeque<([f32; FRAME_SAMPLES], bool)> =
        VecDeque::with_capacity(WAKE_PREROLL_FRAMES);
    let mut near_wake_logged_at: Option<Instant> = None;
    let mut current: Option<(PostWakeEndpoint, Recording, WakeClaim)> = None;
    let mut non_wake = None::<NonWakeSegment>;
    let mut ambient_demand = service.ambient_demand();
    let mut generation = ambient_demand.borrow_and_update().generation();
    let mut ambient_after = Instant::now();
    let mut chain_next_ambient_until: Option<Instant> = None;
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
                        follow_up_until = Some(now + *follow_up_window);
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

        if speech < 0.2 {
            noise.observe(rms);
        }
        if score >= *wake_threshold && guard.allow_wake(now, score) {
            let floor = noise.floor();
            let preroll_voiced_frames = preroll.iter().filter(|(_, voiced)| *voiced).count();
            let endpoint = PostWakeEndpoint::with_preroll_speech(
                floor,
                preroll_voiced_frames >= PREROLL_MIN_VOICED_FRAMES,
            );
            tracing::info!(
                room_id = %room.room_id,
                score,
                threshold = *wake_threshold,
                rms,
                floor,
                level_threshold = endpoint.level_threshold(),
                preroll_frames = preroll.len(),
                preroll_voiced_frames,
                "wake word detected"
            );
            non_wake = None;
            follow_up_until = None;
            let mut recording = Recording::new(UtteranceKind::Command, score);
            for (prior, _) in preroll.drain(..) {
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
        capture_busy.store(speaking || current.is_some(), Ordering::Relaxed);
        if speaking {
            preroll.clear();
        } else {
            if preroll.len() == WAKE_PREROLL_FRAMES {
                preroll.pop_front();
            }
            // Require real Silero VAD speech for pre-roll voicing so non-speech clatter
            // (e.g. dishes/water in cocina) never sets preroll_voiced=true.
            let frame_voiced = (PREROLL_VAD_THRESHOLD..=1.0).contains(&speech);
            preroll.push_back((frame, frame_voiced));
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
        let mut segment = non_wake.take().unwrap_or_else(|| {
            let want = match kind {
                UtteranceKind::FollowUp => FOLLOW_UP_PREROLL_FRAMES,
                _ => AMBIENT_PREROLL_FRAMES,
            };
            let available = preroll.len().saturating_sub(1);
            let start = available.saturating_sub(want);
            let prefix = preroll.range(start..available).map(|(samples, _)| samples);
            let continues = kind == UtteranceKind::Ambient
                && chain_next_ambient_until
                    .take()
                    .is_some_and(|until| now < until);
            NonWakeSegment::with_preroll(kind, now, prefix, continues)
        });
        let auto_beam = latest_dsp
            .lock()
            .ok()
            .and_then(|guard| guard.and_then(|telemetry| telemetry.auto_beam_rad));
        if !segment.observe(&frame, voiced, auto_beam) {
            non_wake = Some(segment);
            continue;
        }
        let latency_ms = segment.started.elapsed().as_millis();
        let Some((recording, capped)) = segment.finish() else {
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
                    if capped {
                        ambient_after = now + AMBIENT_CAP_COOLDOWN;
                        chain_next_ambient_until = Some(now + Duration::from_secs(2));
                    } else {
                        ambient_after = now + AMBIENT_SILENCE_COOLDOWN;
                        chain_next_ambient_until = None;
                    }
                    tracing::info!(room_id = %room.room_id, %utterance_id, duration_ms, latency_ms,
                        capped, status = "published", "Ambient segment");
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
