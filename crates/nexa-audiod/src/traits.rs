//! Dependency injection points for device I/O and inference.

use std::future::Future;

use thiserror::Error;

use crate::ma1260::CommandFrame;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("audio I/O failed")]
    Io(#[from] std::io::Error),
    #[error("audio device or model is unavailable")]
    Unavailable,
    #[error("audio device rejected the command")]
    Rejected,
    #[error("audio adapter failed: {0}")]
    Adapter(String),
}

/// One room's mono f32 capture source at 16 kHz.
pub trait CaptureSource: Send {
    /// Open capture I/O after the runtime has reserved an owner.
    fn start(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;

    /// Read samples into caller-owned storage. A zero count signals end of stream.
    fn read(
        &mut self,
        samples: &mut [f32],
    ) -> impl Future<Output = Result<usize, AudioError>> + Send;

    /// Resolve successfully only after the source has stopped producing capture I/O.
    fn stop(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;
}

/// Synchronous model inference. Call adapters from a bounded blocking executor.
pub trait VadEngine: Send {
    fn reset(&mut self) {}

    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, AudioError>;
}

/// Keep model context local to the room providing the 16 kHz frames.
pub trait WakeEngine: Send {
    fn reset(&mut self) {}

    fn wake_score(&mut self, samples: &[f32]) -> Result<f32, AudioError>;
}

/// Run the single shared shadow recognizer off the real-time capture path.
pub trait ShadowStt: Send {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AudioError>;
}

pub trait AmpTransport: Send + Sync {
    /// Resolve successfully only after the adapter accepts the command explicitly.
    fn send(&self, command: CommandFrame) -> impl Future<Output = Result<(), AudioError>> + Send;
}

pub trait XvfTransport: Send {
    /// Execute one request and read its response using the room's XVF device.
    fn exchange(
        &mut self,
        resource: u8,
        command: u8,
        request: &[u8],
        response: &mut [u8],
    ) -> impl Future<Output = Result<usize, AudioError>> + Send;
}

#[derive(Debug)]
pub struct PlaybackSpec {
    pub playback_id: String,
    pub zone_id: String,
    pub sample_rate: u32,
    pub priority: u32,
}

pub trait PlaybackOutput: Send + Sync {
    type Session: PlaybackSession;

    /// Open a per-stream output. Dropping the future must release any partial open.
    fn open(
        &self,
        spec: &PlaybackSpec,
    ) -> impl Future<Output = Result<Self::Session, AudioError>> + Send;
}

/// A live playback. Dropping it before a successful finish must cancel queued output.
/// Write and finish futures can be dropped when a stream is cancelled.
pub trait PlaybackSession: Send {
    fn write(&mut self, samples: &[f32]) -> impl Future<Output = Result<(), AudioError>> + Send;

    fn finish(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;
}
