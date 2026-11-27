//! Voice turns, per-zone cancellation, and streaming playback.

#![allow(
    clippy::result_large_err,
    reason = "Tonic status carries gRPC metadata and codes across this boundary"
)]

mod dispatch;
pub(crate) mod llm;
mod tts;
mod turns;

pub(crate) use turns::{ActiveTurns, Turn};
