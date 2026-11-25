//! `nexa-web` — Independent Rust Axum Control Plane, Auth/RBAC & WebSocket fan-out daemon (Phase 5 target).
//!
//! Completely decoupled from the real-time voice loop; talks to `nexa-audiod`, `nexa-brain`,
//! and `nexa-memoryd` strictly over mode-0600 Unix Domain Sockets.

use nexa_config::{NEXA_PORT_MAX, NEXA_PORT_MIN};
use tracing::info;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_target(true).init();
    info!(
        service = "nexa-web",
        port_range_min = NEXA_PORT_MIN,
        port_range_max = NEXA_PORT_MAX,
        "nexa-web skeleton initialized (Phase 1 foundation)"
    );
}
