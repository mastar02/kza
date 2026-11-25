//! `nexa-memoryd` — Privacy, semantic memory, voice registry & atomic erasure daemon (Phase 4E target).
//!
//! Exclusive owner of WAL SQLite + embedded vector search (`sqlite-vec`), with zero network egress.

use nexa_proto::sockets;
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_target(true).init();
    info!(
        service = "nexa-memoryd",
        socket = sockets::MEMORY,
        "nexa-memoryd skeleton initialized (Phase 1 foundation)"
    );
}
