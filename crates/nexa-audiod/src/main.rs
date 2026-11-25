//! `nexa-audiod` — Real-time audio edge daemon (Phase 2 target).
//!
//! Responsibilities: per-room ALSA capture with isolated recovery, XVF3800 USB control (`rusb`),
//! MA1260 zone amp control (`tokio-serial`), lock-free playback ring buffers, CPU ONNX inference
//! (`ort`: Silero VAD, OpenWakeWord, single shared Parakeet), and hardware capture mute sealing.

use nexa_proto::sockets;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_target(true).init();
    info!(
        service = "nexa-audiod",
        socket = sockets::AUDIO,
        "nexa-audiod skeleton initialized (Phase 1 foundation)"
    );
}
