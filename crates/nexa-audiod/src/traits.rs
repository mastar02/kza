//! Dependency injection points for device I/O and inference.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use thiserror::Error;

use crate::ma1260::CommandFrame;
use crate::ring::CaptureProducer;

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

impl AudioError {
    pub(crate) fn cause(&self) -> &dyn std::fmt::Display {
        match self {
            Self::Io(error) => error,
            _ => self,
        }
    }
}

/// Shared capture-path counters updated by the capture producer and exposed in health.
#[derive(Debug, Default)]
pub struct CaptureCounters {
    pub xruns: AtomicU64,
    pub ring_overruns: AtomicU64,
}

/// Out-of-band hardware DSP telemetry probe (never called on the audio capture loop).
#[tonic::async_trait]
pub trait DspProbe: Send + Sync {
    /// Poll auto-select beam azimuth in radians (best effort).
    async fn read_azimuth(&self) -> Option<f32>;
}

/// One room's mono f32 capture source at 16 kHz.
pub trait CaptureSource: Send {
    /// Device metadata for an open capture session.
    fn device(&self) -> (&str, u32) {
        ("unknown", 1)
    }

    /// Open capture I/O after the runtime has reserved an owner.
    fn start(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;

    /// Stream mono 16 kHz samples directly into the bounded SPSC capture ring until
    /// EOF, fatal error, or future cancellation.
    fn run(
        &mut self,
        producer: CaptureProducer,
        counters: Arc<CaptureCounters>,
    ) -> impl Future<Output = Result<(), AudioError>> + Send;

    /// Resolve successfully only after the source has stopped producing capture I/O.
    fn stop(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;

    /// Whether this source can suffer dead USB isochronous frames (consecutive exact zeros).
    fn supports_dead_isoch_watchdog(&self) -> bool {
        false
    }

    /// Issue a hardware reboot on supported capture devices (e.g. XVF3800 USB).
    fn reboot(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send {
        async { Ok(()) }
    }

    /// Optional independent handle for out-of-band DSP telemetry (e.g. XVF3800 DoA).
    /// Called after `start()` succeeds.
    fn dsp_probe(&self) -> Option<Arc<dyn DspProbe>> {
        None
    }
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
    fn is_enabled(&self) -> bool {
        true
    }

    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AudioError>;
}

/// Zero-overhead no-op shadow recognizer used when Parakeet is disabled.
#[derive(Debug, Default, Clone, Copy)]
pub struct DisabledShadow;

impl ShadowStt for DisabledShadow {
    fn is_enabled(&self) -> bool {
        false
    }

    fn transcribe(&mut self, _samples: &[f32]) -> Result<String, AudioError> {
        Ok(String::new())
    }
}

pub trait AmpTransport: Send + Sync {
    /// Transport mode exposed in diagnostics.
    fn mode(&self) -> &'static str {
        "unknown"
    }

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
    /// Identifier of the output device.
    fn card(&self) -> &str {
        "unknown"
    }

    fn write(&mut self, samples: &[f32]) -> impl Future<Output = Result<(), AudioError>> + Send;

    fn finish(&mut self) -> impl Future<Output = Result<(), AudioError>> + Send;
}
