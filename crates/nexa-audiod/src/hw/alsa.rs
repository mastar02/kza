//! Room-local ALSA workers. PCM handles never leave their native I/O threads.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alsa::pcm::{Access, Format, HwParams, PCM, State};
use alsa::{Direction, ValueOr};
use nexa_config::RoomDefinition;
use tokio::sync::{mpsc, oneshot};

use crate::dsp::{FRAME_SAMPLES, SAMPLE_RATE};
use crate::hw::{sysfs, xvf::XvfUsb};
use crate::runtime::zone_id;
use crate::traits::{AudioError, CaptureSource, PlaybackOutput, PlaybackSession, PlaybackSpec};

const POLL_MS: u32 = 50;
const IO_TIMEOUT: Duration = Duration::from_millis(500);
const SYSFS: &str = "/sys";

type Reply<T> = oneshot::Sender<Result<T, AudioError>>;

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
struct CaptureRequest {
    length: usize,
    reply: Reply<Vec<f32>>,
}

#[derive(Debug)]
pub struct AlsaCapture {
    room: RoomDefinition,
    sender: Option<mpsc::Sender<CaptureRequest>>,
    worker: Option<JoinHandle<()>>,
    device: Option<(String, u32)>,
}

impl AlsaCapture {
    pub fn new(room: RoomDefinition) -> Self {
        Self {
            room,
            sender: None,
            worker: None,
            device: None,
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
        let (sender, mut requests) = mpsc::channel::<CaptureRequest>(1);
        let (ready, initialized) = oneshot::channel();
        self.worker = Some(thread::Builder::new().name("alsa-capture".into()).spawn(
            move || {
                let (pcm, card, channels) = match open_capture(&room) {
                    Ok(open) => open,
                    Err(error) => {
                        send_reply(ready, Err(error), "open capture");
                        return;
                    }
                };
                if !send_reply(ready, Ok((card, channels)), "open capture") {
                    return;
                }
                let channels = channels as usize;
                let mut interleaved = vec![0; FRAME_SAMPLES * channels];
                while let Some(request) = requests.blocking_recv() {
                    let chunk = &mut interleaved[..request.length * channels];
                    let result = read_capture(&pcm, chunk).map(|frames| {
                        chunk
                            .chunks_exact(channels)
                            .take(frames)
                            .map(|frame| f32::from(frame[usize::from(room.mic_channel)]) / 32_768.0)
                            .collect()
                    });
                    let failed = result.is_err();
                    if !send_reply(request.reply, result, "read capture") || failed {
                        break;
                    }
                }
                // Closing PCM is the stop barrier, including on disconnect and read error.
            },
        )?);
        self.sender = Some(sender);
        self.device = Some(
            initialized
                .await
                .map_err(|error| channel_error("open capture", error))??,
        );
        Ok(())
    }

    async fn read(&mut self, samples: &mut [f32]) -> Result<usize, AudioError> {
        let sender = self.sender.as_ref().ok_or(AudioError::Unavailable)?;
        let (reply, result) = oneshot::channel();
        sender
            .send(CaptureRequest {
                length: samples.len().min(FRAME_SAMPLES),
                reply,
            })
            .await
            .map_err(|error| channel_error("request capture read", error))?;
        let received = result
            .await
            .map_err(|error| channel_error("receive capture frame", error))??;
        samples[..received.len()].copy_from_slice(&received);
        Ok(received.len())
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        self.sender.take();
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
}

fn open_capture(room: &RoomDefinition) -> Result<(PCM, String, u32), AudioError> {
    let (port, card) = sysfs::resolve(
        Path::new(SYSFS),
        room.mic_usb_port.as_deref(),
        room.mic_serial.as_deref(),
    )?;
    XvfUsb::new(port).apply(&room.xvf_tuning)?;
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
    pcm.start().map_err(alsa_error)?;
    Ok((pcm, card, channels))
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

fn read_capture(pcm: &PCM, samples: &mut [i16]) -> Result<usize, AudioError> {
    let io = pcm.io_i16().map_err(alsa_error)?;
    let started = Instant::now();
    loop {
        match io.readi(samples) {
            Ok(count) if count > 0 || samples.is_empty() => return Ok(count),
            Err(error) if !again(error) => return Err(alsa_error(error)),
            _ if started.elapsed() >= IO_TIMEOUT => {
                return Err(std::io::Error::from(std::io::ErrorKind::TimedOut).into());
            }
            _ => {
                pcm.wait(Some(POLL_MS)).map_err(alsa_error)?;
            }
        }
    }
}

#[derive(Debug)]
pub struct AlsaPlayback {
    rooms: BTreeMap<String, RoomDefinition>,
}

impl AlsaPlayback {
    pub fn new(rooms: Vec<RoomDefinition>) -> Self {
        Self {
            rooms: rooms
                .into_iter()
                .map(|room| (zone_id(&room.room_id), room))
                .collect(),
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
        let sample_rate = spec.sample_rate;
        let (sender, mut requests) = mpsc::channel(1);
        let (ready, opened) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut session = AlsaPlaybackSession {
            sender,
            cancelled: Arc::clone(&cancelled),
            card: String::new(),
        };
        thread::Builder::new()
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
                pcm.wait(Some(POLL_MS)).map_err(alsa_error)?;
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
