//! Room-local ALSA workers. PCM handles never leave their native I/O threads.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alsa::pcm::{Access, Format, HwParams, PCM, State};
use alsa::{Direction, ValueOr};
use nexa_config::RoomDefinition;
use tokio::sync::{mpsc, oneshot};

use crate::dsp::{FRAME_SAMPLES, SAMPLE_RATE};
use crate::hw::{sysfs, xvf::XvfUsb};
use crate::ring::{CaptureProducer, PushError};
use crate::runtime::{BurstTracker, zone_id};
use crate::traits::{
    AudioError, CaptureCounters, CaptureSource, DspProbe, PlaybackOutput, PlaybackSession,
    PlaybackSpec,
};

const CAPTURE_POLL_MS: u32 = 50;
const PLAYBACK_POLL_MS: u32 = 15;
const IO_TIMEOUT: Duration = Duration::from_millis(500);
const XRUN_BURST_LIMIT: usize = 3;
const XRUN_BURST_WINDOW: Duration = Duration::from_secs(10);
const RING_OVERRUN_RESET_AFTER: Duration = Duration::from_secs(1);
const RING_OVERRUN_LOG_INTERVAL: Duration = Duration::from_secs(1);
const SYSFS: &str = "/sys";

type Reply<T> = oneshot::Sender<Result<T, AudioError>>;
type CaptureStart = (CaptureProducer, Arc<CaptureCounters>);

fn channel_error(operation: &'static str, error: impl std::fmt::Display) -> AudioError {
    tracing::warn!(operation, %error, "ALSA worker communication failed");
    AudioError::Unavailable
}

fn send_reply<T>(reply: Reply<T>, result: Result<T, AudioError>, operation: &'static str) -> bool {
    match reply.send(result) {
        Ok(()) => true,
        Err(Err(error)) => {
            tracing::warn!(operation, error = %error.cause(), "ALSA operation failed after receiver closed");
            false
        }
        Err(Ok(_)) => false, // Expected when an in-flight operation is cancelled.
    }
}

fn drop_playback(pcm: &PCM) {
    if let Err(error) = pcm.drop() {
        tracing::warn!(%error, "ALSA playback buffer discard failed");
    }
}

#[derive(Debug)]
pub struct AlsaCapture {
    room: RoomDefinition,
    begin: Option<oneshot::Sender<CaptureStart>>,
    done: Option<oneshot::Receiver<Result<(), AudioError>>>,
    stop: Option<Arc<AtomicBool>>,
    worker: Option<JoinHandle<()>>,
    device: Option<(String, u32)>,
    port: Option<String>,
}

impl AlsaCapture {
    pub fn new(room: RoomDefinition) -> Self {
        Self {
            room,
            begin: None,
            done: None,
            stop: None,
            worker: None,
            device: None,
            port: None,
        }
    }
}

impl CaptureSource for AlsaCapture {
    fn device(&self) -> (&str, u32) {
        self.device
            .as_ref()
            .map_or(("unknown", 1), |(card, channels)| {
                (card.as_str(), *channels)
            })
    }

    async fn start(&mut self) -> Result<(), AudioError> {
        if self.worker.is_some() {
            return Err(AudioError::Rejected);
        }
        let room = self.room.clone();
        let (ready, initialized) = oneshot::channel();
        let (begin_tx, begin_rx) = oneshot::channel::<CaptureStart>();
        let (done_tx, done_rx) = oneshot::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("alsa-capture".into())
            .spawn(move || {
                let (pcm, port, card, channels) = match open_capture(&room) {
                    Ok(open) => open,
                    Err(error) => {
                        send_reply(ready, Err(error), "open capture");
                        return;
                    }
                };
                if !send_reply(ready, Ok((port, card, channels)), "open capture") {
                    return;
                }
                let Ok((mut producer, counters)) = begin_rx.blocking_recv() else {
                    return;
                };
                if worker_stop.load(Ordering::Acquire) {
                    return;
                }
                if let Err(error) = pcm.start().map_err(alsa_error) {
                    send_reply(done_tx, Err(error), "start capture");
                    return;
                }
                let result = capture_loop(
                    &room.room_id,
                    &pcm,
                    channels as usize,
                    usize::from(room.mic_channel),
                    &mut producer,
                    &counters,
                    &worker_stop,
                );
                send_reply(done_tx, result, "run capture");
                // Closing PCM is the stop barrier, including on disconnect and read error.
            })?;
        self.stop = Some(stop);
        self.begin = Some(begin_tx);
        self.done = Some(done_rx);
        self.worker = Some(worker);
        let (port, card, channels) = initialized
            .await
            .map_err(|error| channel_error("open capture", error))??;
        self.port = Some(port);
        self.device = Some((card, channels));
        Ok(())
    }

    async fn run(
        &mut self,
        producer: CaptureProducer,
        counters: Arc<CaptureCounters>,
    ) -> Result<(), AudioError> {
        let begin = self.begin.take().ok_or(AudioError::Unavailable)?;
        let done = self.done.take().ok_or(AudioError::Unavailable)?;
        begin
            .send((producer, counters))
            .map_err(|_| AudioError::Unavailable)?;
        done.await
            .map_err(|error| channel_error("join capture stream", error))?
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        if let Some(stop) = self.stop.take() {
            stop.store(true, Ordering::Release);
        }
        self.begin.take();
        self.done.take();
        self.device.take();
        if let Some(worker) = self.worker.take() {
            // Joining also confirms PCM was dropped if the worker unwound.
            tokio::task::spawn_blocking(move || {
                if worker.join().is_err() {
                    tracing::warn!("ALSA capture worker panicked");
                }
            })
            .await
            .map_err(|error| channel_error("join capture worker", error))?;
        }
        Ok(())
    }

    fn supports_dead_isoch_watchdog(&self) -> bool {
        true
    }

    async fn reboot(&mut self) -> Result<(), AudioError> {
        let port = if let Some(port) = self.port.clone() {
            port
        } else if let Some(port) = self.room.mic_usb_port.clone() {
            port
        } else if let Some(serial) = self.room.mic_serial.clone() {
            tokio::task::spawn_blocking(move || {
                sysfs::resolve(Path::new(SYSFS), None, Some(&serial)).map(|(port, _)| port)
            })
            .await
            .map_err(|error| channel_error("resolve XVF port for reboot", error))??
        } else {
            return Err(AudioError::Unavailable);
        };
        XvfUsb::new(port).reboot().await
    }

    fn dsp_probe(&self) -> Option<Arc<dyn DspProbe>> {
        self.port
            .clone()
            .map(|port| Arc::new(XvfUsb::new(port)) as Arc<dyn DspProbe>)
    }
}

fn open_capture(room: &RoomDefinition) -> Result<(PCM, String, String, u32), AudioError> {
    let (port, card) = sysfs::resolve(
        Path::new(SYSFS),
        room.mic_usb_port.as_deref(),
        room.mic_serial.as_deref(),
    )?;
    XvfUsb::new(port.clone()).apply(&room.xvf_tuning)?;
    let hw =
        PCM::new(&format!("hw:CARD={card},DEV=0"), Direction::Capture, true).map_err(alsa_error)?;
    let channels = HwParams::any(&hw)
        .and_then(|params| params.get_channels_max())
        .map_err(alsa_error)?;
    drop(hw);
    if u32::from(room.mic_channel) >= channels || !(1..=6).contains(&channels) {
        return Err(AudioError::Rejected);
    }
    let pcm = open_pcm(&card, Direction::Capture, channels, SAMPLE_RATE)?;
    Ok((pcm, port, card, channels))
}

fn open_pcm(card: &str, direction: Direction, channels: u32, rate: u32) -> Result<PCM, AudioError> {
    let pcm =
        PCM::new(&format!("plughw:CARD={card},DEV=0"), direction, true).map_err(alsa_error)?;
    {
        let params = HwParams::any(&pcm).map_err(alsa_error)?;
        params
            .set_access(Access::RWInterleaved)
            .map_err(alsa_error)?;
        params.set_format(Format::s16()).map_err(alsa_error)?;
        params.set_channels(channels).map_err(alsa_error)?;
        params
            .set_rate(rate, ValueOr::Nearest)
            .map_err(alsa_error)?;
        params
            .set_period_time_near(20_000, ValueOr::Nearest)
            .map_err(alsa_error)?;
        let buffer_time = if direction == Direction::Capture {
            320_000
        } else {
            80_000
        };
        params
            .set_buffer_time_near(buffer_time, ValueOr::Nearest)
            .map_err(alsa_error)?;
        pcm.hw_params(&params).map_err(alsa_error)?;
    }
    pcm.prepare().map_err(alsa_error)?;
    Ok(pcm)
}

fn capture_loop(
    room_id: &str,
    pcm: &PCM,
    channels: usize,
    mic_channel: usize,
    producer: &mut CaptureProducer,
    counters: &CaptureCounters,
    stop: &AtomicBool,
) -> Result<(), AudioError> {
    let io = pcm.io_i16().map_err(alsa_error)?;
    let mut interleaved = vec![0_i16; FRAME_SAMPLES * channels];
    let mut mono = [0.0_f32; FRAME_SAMPLES];
    let mut xruns = BurstTracker::new(XRUN_BURST_LIMIT, XRUN_BURST_WINDOW);
    let mut last_progress = Instant::now();
    let mut overrun_since: Option<Instant> = None;
    let mut last_overrun_log: Option<Instant> = None;

    while !stop.load(Ordering::Acquire) {
        match io.readi(&mut interleaved) {
            Ok(frames) if frames > 0 => {
                last_progress = Instant::now();
                let count = frames.min(FRAME_SAMPLES);
                for (dst, frame) in mono[..count]
                    .iter_mut()
                    .zip(interleaved.chunks_exact(channels).take(count))
                {
                    *dst = f32::from(frame[mic_channel]) / 32_768.0;
                }
                match producer.try_push(&mono[..count]) {
                    Ok(()) => overrun_since = None,
                    Err(PushError::Closed) => return Ok(()),
                    Err(PushError::Full) => {
                        counters.ring_overruns.fetch_add(1, Ordering::Relaxed);
                        let now = Instant::now();
                        let started = *overrun_since.get_or_insert(now);
                        if last_overrun_log.is_none_or(|prev| {
                            now.duration_since(prev) >= RING_OVERRUN_LOG_INTERVAL
                        }) {
                            last_overrun_log = Some(now);
                            tracing::warn!(
                                %room_id,
                                dropped_samples = count,
                                "capture ring full; dropping samples"
                            );
                        }
                        if now.duration_since(started) >= RING_OVERRUN_RESET_AFTER {
                            tracing::warn!(
                                %room_id,
                                "capture ring overflow persisted; restarting capture"
                            );
                            return Err(AudioError::Unavailable);
                        }
                    }
                }
            }
            Err(error) if error.errno() == rustix::io::Errno::PIPE.raw_os_error() => {
                counters.xruns.fetch_add(1, Ordering::Relaxed);
                if xruns.record_exceeded(Instant::now()) {
                    tracing::warn!(%room_id, %error, "ALSA capture overrun burst exceeded limit");
                    return Err(alsa_error(error));
                }
                tracing::warn!(%room_id, %error, "ALSA capture overrun; preparing device");
                pcm.prepare().map_err(alsa_error)?;
                pcm.start().map_err(alsa_error)?;
                last_progress = Instant::now();
            }
            Err(error) if !again(error) => return Err(alsa_error(error)),
            _ if last_progress.elapsed() >= IO_TIMEOUT => {
                return Err(std::io::Error::from(std::io::ErrorKind::TimedOut).into());
            }
            _ => {
                pcm.wait(Some(CAPTURE_POLL_MS)).map_err(alsa_error)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct AlsaPlayback {
    rooms: BTreeMap<String, RoomDefinition>,
    workers: Mutex<BTreeMap<String, JoinHandle<()>>>,
}

impl AlsaPlayback {
    pub fn new(rooms: Vec<RoomDefinition>) -> Self {
        Self {
            rooms: rooms
                .into_iter()
                .map(|room| (zone_id(&room.room_id), room))
                .collect(),
            workers: Mutex::default(),
        }
    }
}

#[derive(Debug)]
enum PlaybackRequest {
    Write(Vec<i16>, Reply<()>),
    Finish(Reply<()>),
}

#[derive(Debug)]
pub struct AlsaPlaybackSession {
    sender: mpsc::Sender<PlaybackRequest>,
    cancelled: Arc<AtomicBool>,
    card: String,
}

impl Drop for AlsaPlaybackSession {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl PlaybackOutput for AlsaPlayback {
    type Session = AlsaPlaybackSession;

    async fn open(&self, spec: &PlaybackSpec) -> Result<Self::Session, AudioError> {
        let room = self
            .rooms
            .get(&spec.zone_id)
            .cloned()
            .ok_or(AudioError::Unavailable)?;
        let previous = self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&spec.zone_id);
        if let Some(worker) = previous {
            tokio::task::spawn_blocking(move || {
                if worker.join().is_err() {
                    tracing::warn!("ALSA playback worker panicked");
                }
            })
            .await
            .map_err(|error| channel_error("join prior playback worker", error))?;
        }

        let sample_rate = spec.sample_rate;
        let (sender, mut requests) = mpsc::channel(1);
        let (ready, opened) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut session = AlsaPlaybackSession {
            sender,
            cancelled: Arc::clone(&cancelled),
            card: String::new(),
        };
        let worker = thread::Builder::new()
            .name("alsa-playback".into())
            .spawn(move || {
                let opened = sysfs::alsa_card(
                    Path::new(SYSFS),
                    room.mic_usb_port.as_deref(),
                    room.mic_serial.as_deref(),
                )
                .map_err(AudioError::from)
                .and_then(|card| {
                    open_pcm(&card, Direction::Playback, 1, sample_rate)
                        .inspect_err(|error| tracing::warn!(%card, sample_rate, error = %error.cause(), "ALSA playback open failed"))
                        .map(|pcm| (pcm, card))
                });
                let (pcm, card) = match opened {
                    Ok(opened) => opened,
                    Err(error) => {
                        send_reply(ready, Err(error), "open playback");
                        return;
                    }
                };
                if !send_reply(ready, Ok(card), "open playback") {
                    return;
                }
                while let Some(request) = requests.blocking_recv() {
                    match request {
                        PlaybackRequest::Write(samples, reply) => {
                            let result = write_playback(&pcm, &samples, &cancelled);
                            let failed = result.is_err();
                            if !send_reply(reply, result, "write playback") || failed {
                                break;
                            }
                        }
                        PlaybackRequest::Finish(reply) => {
                            let result = drain_playback(&pcm, &cancelled);
                            if result.is_err() {
                                drop_playback(&pcm);
                            }
                            send_reply(reply, result, "drain playback");
                            return;
                        }
                    }
                }
                drop_playback(&pcm);
            })?;
        self.workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(spec.zone_id.clone(), worker);
        session.card = opened
            .await
            .map_err(|error| channel_error("open playback", error))??;
        Ok(session)
    }
}

impl PlaybackSession for AlsaPlaybackSession {
    fn card(&self) -> &str {
        &self.card
    }

    async fn write(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        let pcm = samples
            .iter()
            .map(|&sample| (sample.clamp(-1.0, 1.0) * 32_767.0) as i16)
            .collect();
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PlaybackRequest::Write(pcm, reply))
            .await
            .map_err(|error| channel_error("request playback write", error))?;
        result
            .await
            .map_err(|error| channel_error("receive playback write result", error))?
    }

    async fn finish(&mut self) -> Result<(), AudioError> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PlaybackRequest::Finish(reply))
            .await
            .map_err(|error| channel_error("request playback drain", error))?;
        result
            .await
            .map_err(|error| channel_error("receive playback drain result", error))?
    }
}

fn write_playback(
    pcm: &PCM,
    mut samples: &[i16],
    cancelled: &AtomicBool,
) -> Result<(), AudioError> {
    let io = pcm.io_i16().map_err(alsa_error)?;
    let mut last_progress = Instant::now();
    while !samples.is_empty() {
        if cancelled.load(Ordering::Acquire) {
            return Err(std::io::Error::from(std::io::ErrorKind::Interrupted).into());
        }
        if last_progress.elapsed() >= IO_TIMEOUT {
            return Err(std::io::Error::from(std::io::ErrorKind::TimedOut).into());
        }
        match io.writei(samples) {
            Ok(count) if count != 0 => {
                samples = &samples[count..];
                last_progress = Instant::now();
            }
            Err(error) if error.errno() == rustix::io::Errno::PIPE.raw_os_error() => {
                tracing::warn!(%error, "ALSA playback underrun; preparing device");
                pcm.prepare().map_err(alsa_error)?;
            }
            Err(error) if !again(error) => return Err(alsa_error(error)),
            _ => {
                pcm.wait(Some(PLAYBACK_POLL_MS)).map_err(alsa_error)?;
            }
        }
    }
    Ok(())
}

fn drain_playback(pcm: &PCM, cancelled: &AtomicBool) -> Result<(), AudioError> {
    let started = Instant::now();
    while !cancelled.load(Ordering::Acquire) && started.elapsed() < IO_TIMEOUT {
        match pcm.drain() {
            Ok(()) => return Ok(()),
            Err(error) if again(error) => thread::sleep(Duration::from_millis(10)),
            Err(error) if pcm.state() == State::Setup => {
                tracing::warn!(%error, "ALSA playback drain returned an error after device stopped");
                return Ok(());
            }
            Err(error) => return Err(alsa_error(error)),
        }
    }
    let kind = if cancelled.load(Ordering::Acquire) {
        std::io::ErrorKind::Interrupted
    } else {
        std::io::ErrorKind::TimedOut
    };
    Err(std::io::Error::from(kind).into())
}

fn again(error: alsa::Error) -> bool {
    error.errno() == rustix::io::Errno::AGAIN.raw_os_error()
}

fn alsa_error(error: alsa::Error) -> AudioError {
    AudioError::Adapter(error.to_string())
}
