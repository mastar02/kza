//! Portable AudioService with injected amplifier and playback adapters.

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nexa_proto::v1::audio_service_server::AudioService as AudioServiceRpc;
use nexa_proto::v1::{
    ActorContext, AudioUtterance, CancelPlaybackRequest, CaptureControlStatus, Empty,
    ForgetCarriedOwnersRequest, ForgetCarriedOwnersResponse, HealthResponse, HealthState,
    MicCaptureState, MicrophoneRoomStatus, MicrophonesSnapshot, Permission, PlayAudioChunk,
    PlayAudioReceipt, Role, SetCaptureMuteRequest, SetZoneVolumeRequest,
    SubscribeUtterancesRequest, UtteranceKind, ZoneState, ZonesSnapshot,
};
use tokio::sync::{Mutex as AsyncMutex, broadcast, watch};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::barge_in::BargeIn;
use crate::capture_seal::{CaptureControl, CaptureError, CapturePhase};
use crate::dsp::{frame_duration, frame_rms};
use crate::ma1260::{Command, CommandFrame, LAST_ZONE, percent_to_step, step_to_percent};
use crate::traits::{AmpTransport, AudioError, PlaybackOutput, PlaybackSession, PlaybackSpec};

const PLAYBACK_SAMPLE_RATE: u32 = 24_000;
const UTTERANCE_QUEUE: usize = 16;

#[derive(Debug)]
struct Zone {
    room_id: String,
    amp_zone: u8,
    volume: AsyncMutex<ZoneState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopReason {
    Cancelled,
    BargeIn,
}

struct ActivePlayback {
    token: String,
    stop: watch::Sender<Option<StopReason>>,
    speaking: bool,
    capture_epoch: Option<u64>,
    detector: BargeIn,
}

impl ActivePlayback {
    fn stop(&self, reason: StopReason) -> bool {
        self.stop.send_if_modified(|state| {
            if state.is_some() {
                return false;
            }
            *state = Some(reason);
            true
        })
    }
}

type PlaybackRegistry = Mutex<BTreeMap<String, ActivePlayback>>;

struct PlaybackSlot<'a> {
    registry: &'a PlaybackRegistry,
    zone_id: Option<&'a str>,
}

impl PlaybackSlot<'_> {
    #[allow(clippy::result_large_err)]
    fn retire(mut self) -> Result<Option<StopReason>, Status> {
        let mut active = lock(self.registry)?;
        let playback = self
            .zone_id
            .take()
            .and_then(|id| active.remove(id))
            .ok_or_else(|| Status::internal("playback slot disappeared"))?;
        Ok(*playback.stop.borrow())
    }
}

impl Drop for PlaybackSlot<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.zone_id
            && let Ok(mut active) = self.registry.lock()
        {
            active.remove(id);
        }
    }
}

pub struct AudioService<A, P> {
    capture: Arc<Mutex<CaptureControl>>,
    amp: A,
    output: P,
    zones: BTreeMap<String, Zone>,
    microphones: Mutex<MicrophonesSnapshot>,
    playbacks: PlaybackRegistry,
    utterances: broadcast::Sender<AudioUtterance>,
    started: Instant,
}

impl<A, P> fmt::Debug for AudioService<A, P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioService")
            .field("zone_count", &self.zones.len())
            .finish_non_exhaustive()
    }
}

#[allow(clippy::result_large_err)] // Tonic's public error type carries gRPC metadata.
impl<A: AmpTransport, P: PlaybackOutput> AudioService<A, P> {
    pub fn new(
        capture: Arc<Mutex<CaptureControl>>,
        amp: A,
        output: P,
        zones: impl IntoIterator<Item = ZoneState>,
    ) -> Result<Self, Status> {
        let mut configured = BTreeMap::new();
        for mut zone in zones {
            let amp_zone = u8::try_from(zone.ma1260_zone)
                .ok()
                .filter(|zone| (1..=LAST_ZONE).contains(zone))
                .ok_or_else(|| Status::invalid_argument("invalid amplifier zone"))?;
            if zone.zone_id.is_empty()
                || zone.room_id.is_empty()
                || zone.volume_pct > 100
                || configured.contains_key(&zone.zone_id)
            {
                return Err(Status::invalid_argument("invalid or duplicate audio zone"));
            }
            zone.speaking = false;
            configured.insert(
                zone.zone_id.clone(),
                Zone {
                    room_id: zone.room_id.clone(),
                    amp_zone,
                    volume: AsyncMutex::new(zone),
                },
            );
        }
        Ok(Self {
            capture,
            amp,
            output,
            zones: configured,
            microphones: Mutex::new(MicrophonesSnapshot {
                stale: true,
                ..MicrophonesSnapshot::default()
            }),
            playbacks: Mutex::default(),
            utterances: broadcast::channel(UTTERANCE_QUEUE).0,
            started: Instant::now(),
        })
    }

    pub fn update_microphones(&self, microphones: Vec<MicrophoneRoomStatus>) -> Result<(), Status> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Status::internal("system clock precedes the Unix epoch"))?;
        *lock(&self.microphones)? = MicrophonesSnapshot {
            observed_at_unix_ms: u64::try_from(now.as_millis()).unwrap_or(u64::MAX),
            stale: false,
            microphones,
        };
        Ok(())
    }

    /// Publish only while the capture's original ingress epoch remains open.
    pub fn publish_utterance(&self, utterance: AudioUtterance) -> Result<bool, Status> {
        let capture = lock(&self.capture)?;
        if capture.running_epoch() != Some(utterance.capture_epoch) {
            return Ok(false);
        }
        Ok(self.utterances.send(utterance).is_ok())
    }

    /// Feed audio from the currently admitted capture epoch into matching zones' barge-in detectors.
    /// The room association comes from the configured zones, never from a playback client claim.
    pub fn observe_capture_frame(
        &self,
        room_id: &str,
        epoch: u64,
        frame: &[f32],
    ) -> Result<usize, Status> {
        let rms = frame_rms(frame);
        let duration = frame_duration(frame.len());
        let capture = lock(&self.capture)?;
        if capture.running_epoch() != Some(epoch) {
            return Ok(0);
        }

        let mut active = lock(&self.playbacks)?;
        let mut interrupted = 0;
        for (zone_id, playback) in &mut *active {
            if self.zones[zone_id].room_id != room_id || playback.stop.borrow().is_some() {
                continue;
            }
            if playback.capture_epoch != Some(epoch) {
                playback.capture_epoch = Some(epoch);
                playback.detector = BargeIn::default();
            }
            if playback.detector.observe(playback.speaking, rms, duration)
                && playback.stop(StopReason::BargeIn)
            {
                interrupted += 1;
            }
        }
        Ok(interrupted)
    }

    fn zone(&self, id: &str) -> Result<&Zone, Status> {
        self.zones
            .get(id)
            .ok_or_else(|| Status::not_found("unknown audio zone"))
    }

    fn speaking(&self, zone_id: &str) -> Result<bool, Status> {
        Ok(lock(&self.playbacks)?
            .get(zone_id)
            .is_some_and(|playback| playback.speaking && playback.stop.borrow().is_none()))
    }

    fn reserve_playback<'a>(
        &'a self,
        zone_id: &'a str,
        token: &str,
    ) -> Result<(PlaybackSlot<'a>, watch::Receiver<Option<StopReason>>), Status> {
        self.zone(zone_id)?;
        let mut active = lock(&self.playbacks)?;
        if active.contains_key(zone_id) {
            return Err(Status::already_exists("audio zone already has a playback"));
        }
        let (stop, receiver) = watch::channel(None);
        active.insert(
            zone_id.to_owned(),
            ActivePlayback {
                token: token.to_owned(),
                stop,
                speaking: false,
                capture_epoch: None,
                detector: BargeIn::default(),
            },
        );
        let slot = PlaybackSlot {
            registry: &self.playbacks,
            zone_id: Some(zone_id),
        };
        Ok((slot, receiver))
    }

    async fn play_to_output(
        &self,
        spec: &PlaybackSpec,
        token: &str,
        mut chunks: impl Stream<Item = Result<PlayAudioChunk, Status>> + Unpin + Send,
        first_audio: &mut Option<Instant>,
    ) -> Result<(), Status> {
        let mut session = self.output.open(spec).await.map_err(audio_status)?;
        let mut pcm = Vec::new();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            if chunk.playback_id != spec.playback_id
                || chunk.zone_id != spec.zone_id
                || chunk.sample_rate != spec.sample_rate
                || chunk.priority != spec.priority
                || chunk.barge_in_token != token
            {
                return Err(Status::invalid_argument(
                    "playback metadata changed midstream",
                ));
            }

            let (words, remainder) = chunk.pcm_f32le.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(Status::invalid_argument("playback PCM must be f32le"));
            }
            pcm.clear();
            pcm.extend(words.iter().copied().map(f32::from_le_bytes));
            if pcm.iter().any(|sample| !sample.is_finite()) {
                return Err(Status::invalid_argument(
                    "playback PCM contains non-finite samples",
                ));
            }
            if !pcm.is_empty() {
                if let Some(playback) = lock(&self.playbacks)?.get_mut(&spec.zone_id) {
                    playback.speaking = true;
                }
                session.write(&pcm).await.map_err(audio_status)?;
                first_audio.get_or_insert_with(Instant::now);
            }
            if chunk.is_final {
                if first_audio.is_none() {
                    return Err(Status::invalid_argument("playback stream has no samples"));
                }
                return session.finish().await.map_err(audio_status);
            }
        }
        Err(Status::invalid_argument(
            "playback stream ended before its final chunk",
        ))
    }
}

#[tonic::async_trait]
impl<A: AmpTransport + 'static, P: PlaybackOutput + 'static> AudioServiceRpc
    for AudioService<A, P>
{
    async fn get_health(&self, _: Request<Empty>) -> Result<Response<HealthResponse>, Status> {
        let capture = lock(&self.capture)?.status();
        let (state, detail) = if !capture.supported {
            (HealthState::Degraded, "capture control is unavailable")
        } else {
            match capture.phase() {
                CapturePhase::Running => (HealthState::Ok, "ready"),
                CapturePhase::Muted if capture.total_mute_confirmed => {
                    (HealthState::Ok, "capture muted")
                }
                _ => (
                    HealthState::Degraded,
                    "capture sealed, awaiting confirmation",
                ),
            }
        };
        Ok(Response::new(HealthResponse {
            service: "nexa-audiod".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            state: state as i32,
            uptime_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            detail: detail.into(),
        }))
    }

    async fn get_microphones(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<MicrophonesSnapshot>, Status> {
        let capture = lock(&self.capture)?.status();
        let mut snapshot = lock(&self.microphones)?.clone();
        if capture.ingress_sealed {
            for microphone in &mut snapshot.microphones {
                microphone.state = match capture.phase() {
                    CapturePhase::Muted if capture.total_mute_confirmed => MicCaptureState::Muted,
                    CapturePhase::Unspecified => MicCaptureState::Offline,
                    _ => MicCaptureState::Waiting,
                } as i32;
            }
        }
        Ok(Response::new(snapshot))
    }

    async fn get_zones(&self, _: Request<Empty>) -> Result<Response<ZonesSnapshot>, Status> {
        let mut zones = Vec::with_capacity(self.zones.len());
        for (id, zone) in &self.zones {
            let mut state = zone.volume.lock().await.clone();
            state.speaking = self.speaking(id)?;
            zones.push(state);
        }
        Ok(Response::new(ZonesSnapshot { zones }))
    }

    async fn set_zone_volume(
        &self,
        request: Request<SetZoneVolumeRequest>,
    ) -> Result<Response<ZoneState>, Status> {
        let request = request.into_inner();
        require_permission(request.actor.as_ref(), Permission::HomeControl)?;
        if request.volume_pct > 100 {
            return Err(Status::invalid_argument("volume must be between 0 and 100"));
        }
        let zone = self.zone(&request.zone_id)?;
        let step = percent_to_step(request.volume_pct);
        let command = CommandFrame::new(zone.amp_zone, Command::Volume, step)
            .map_err(|_| Status::internal("invalid configured amplifier zone"))?;
        let mut state = zone.volume.lock().await;
        self.amp.send(command).await.map_err(audio_status)?;
        state.volume_pct = step_to_percent(step);
        let mut response = state.clone();
        response.speaking = self.speaking(&request.zone_id)?;
        Ok(Response::new(response))
    }

    async fn get_capture_control(
        &self,
        _: Request<Empty>,
    ) -> Result<Response<CaptureControlStatus>, Status> {
        Ok(Response::new(lock(&self.capture)?.status()))
    }

    async fn set_capture_mute(
        &self,
        request: Request<SetCaptureMuteRequest>,
    ) -> Result<Response<CaptureControlStatus>, Status> {
        let request = request.into_inner();
        require_permission(request.actor.as_ref(), Permission::AdminManage)?;
        let mut control = lock(&self.capture)?;
        control
            .set_muted(request.muted, request.expected_revision)
            .map_err(capture_status)?;
        Ok(Response::new(control.status()))
    }

    async fn forget_carried_owners(
        &self,
        request: Request<ForgetCarriedOwnersRequest>,
    ) -> Result<Response<ForgetCarriedOwnersResponse>, Status> {
        let request = request.into_inner();
        require_permission(request.actor.as_ref(), Permission::AdminManage)?;
        let mut control = lock(&self.capture)?;
        let forgotten = control
            .forget_carried_owners(request.expected_carried, request.no_capture_running)
            .map_err(capture_status)?;
        let status = control.status();
        drop(control);
        if forgotten != 0 {
            tracing::warn!(
                forgotten_capture_owners = forgotten,
                "operator affirmed that no carried owner is still capturing"
            );
        }
        Ok(Response::new(ForgetCarriedOwnersResponse {
            status: Some(status),
            forgotten_capture_owners: forgotten,
        }))
    }

    type SubscribeUtterancesStream =
        Pin<Box<dyn Stream<Item = Result<AudioUtterance, Status>> + Send>>;

    async fn subscribe_utterances(
        &self,
        request: Request<SubscribeUtterancesRequest>,
    ) -> Result<Response<Self::SubscribeUtterancesStream>, Status> {
        let include_ambient = request.into_inner().include_ambient;
        let capture = Arc::clone(&self.capture);
        let stream =
            BroadcastStream::new(self.utterances.subscribe()).filter_map(
                move |result| match result {
                    Ok(utterance) => {
                        let control = match lock(&capture) {
                            Ok(control) => control,
                            Err(status) => return Some(Err(status)),
                        };
                        (control.running_epoch() == Some(utterance.capture_epoch)
                            && (include_ambient || utterance.kind() != UtteranceKind::Ambient))
                            .then_some(Ok(utterance))
                    }
                    Err(_) => Some(Err(Status::resource_exhausted(
                        "utterance subscriber fell behind",
                    ))),
                },
            );
        Ok(Response::new(Box::pin(stream)))
    }

    async fn play_audio_stream(
        &self,
        request: Request<Streaming<PlayAudioChunk>>,
    ) -> Result<Response<PlayAudioReceipt>, Status> {
        let started = Instant::now();
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("empty playback stream"))?;
        if first.playback_id.is_empty()
            || first.barge_in_token.is_empty()
            || first.sample_rate != PLAYBACK_SAMPLE_RATE
        {
            return Err(Status::invalid_argument("invalid 24 kHz playback header"));
        }
        let spec = PlaybackSpec {
            playback_id: first.playback_id.clone(),
            zone_id: first.zone_id.clone(),
            sample_rate: first.sample_rate,
            priority: first.priority,
        };
        let token = first.barge_in_token.clone();
        let (slot, mut stop) = self.reserve_playback(&spec.zone_id, &token)?;
        let mut first_audio = None;
        tokio::select! {
            biased;
            changed = stop.changed() => {
                changed.map_err(|_| Status::unavailable("playback state closed"))?;
            }
            result = self.play_to_output(
                &spec,
                &token,
                tokio_stream::iter([Ok(first)]).chain(stream),
                &mut first_audio,
            ) => {
                result?;
            }
        }
        let stopped = slot.retire()?;
        Ok(Response::new(PlayAudioReceipt {
            playback_id: spec.playback_id.clone(),
            completed: stopped.is_none(),
            interrupted_by_barge_in: stopped == Some(StopReason::BargeIn),
            ttfa_ms: first_audio.map_or(0, |first| millis_u32(first.duration_since(started))),
            total_ms: millis_u32(started.elapsed()),
        }))
    }

    async fn cancel_playback(
        &self,
        request: Request<CancelPlaybackRequest>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        self.zone(&request.zone_id)?;
        if request.barge_in_token.is_empty() {
            return Err(Status::invalid_argument(
                "playback cancellation token required",
            ));
        }
        if let Some(playback) = lock(&self.playbacks)?.get(&request.zone_id)
            && playback.token == request.barge_in_token
        {
            playback.stop(StopReason::Cancelled);
        }
        Ok(Response::new(Empty {}))
    }
}

#[allow(clippy::result_large_err)]
fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, Status> {
    mutex
        .lock()
        .map_err(|_| Status::internal("audio service state is unavailable"))
}

#[allow(clippy::result_large_err)]
fn require_permission(actor: Option<&ActorContext>, permission: Permission) -> Result<(), Status> {
    // The mode-0600 control-plane UDS supplies verified actor context.
    let actor = actor
        .filter(|actor| !actor.subject.is_empty())
        .ok_or_else(|| Status::unauthenticated("verified actor required"))?;
    let role_allowed = actor.role() == Role::Operator
        || (permission == Permission::HomeControl && actor.role() == Role::Home);
    if role_allowed && actor.permissions.contains(&(permission as i32)) {
        Ok(())
    } else {
        Err(Status::permission_denied(
            "audio control permission required",
        ))
    }
}

fn capture_status(error: CaptureError) -> Status {
    match error {
        CaptureError::Unsupported => Status::unavailable(error.to_string()),
        CaptureError::RevisionConflict { .. } | CaptureError::CarriedCountConflict { .. } => {
            Status::aborted(error.to_string())
        }
        CaptureError::ConfirmationRequired => Status::invalid_argument(error.to_string()),
        CaptureError::Exhausted => Status::resource_exhausted(error.to_string()),
        CaptureError::InvalidState => Status::internal(error.to_string()),
        CaptureError::Busy | CaptureError::Sealed | CaptureError::StaleAcknowledgment => {
            Status::failed_precondition(error.to_string())
        }
    }
}

fn audio_status(error: AudioError) -> Status {
    match error {
        AudioError::Rejected => Status::failed_precondition(error.to_string()),
        AudioError::Unavailable | AudioError::Io(_) => Status::unavailable(error.to_string()),
    }
}

fn millis_u32(duration: Duration) -> u32 {
    u32::try_from(duration.as_millis()).unwrap_or(u32::MAX)
}
