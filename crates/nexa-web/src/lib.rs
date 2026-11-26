//! Nexa HTTP/WS control plane. Runtime data is reached through daemon clients.

#[path = "auth/mod.rs"]
pub mod auth;
pub mod clients;
pub mod sys;

mod routes;
#[cfg(feature = "test-mode")]
mod test_mode;
mod ws;

use std::{io, path::PathBuf, sync::Arc};

use auth::{AuthStore, Origin};
use axum::Router;
use clients::Clients;
use sys::SystemMonitor;

#[cfg(feature = "test-mode")]
pub use test_mode::{
    EVALUATOR_TOKEN, HOME_ID, HOME_TOKEN, OPERATOR_TOKEN, TestMode, TestModeError,
};

#[derive(Clone, Debug)]
pub struct WebConfig {
    pub origin: Origin,
    pub allow_insecure_cookies: bool,
    pub private_dir: Option<PathBuf>,
    pub static_dir: Option<PathBuf>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    config: WebConfig,
    auth: AuthStore,
    clients: Clients,
    system: Arc<dyn SystemMonitor>,
    live: ws::Connections,
}

/// Construct a router with isolated authority, daemon clients and system probes.
pub fn build_app(
    mut config: WebConfig,
    auth: AuthStore,
    clients: Clients,
    system: Arc<dyn SystemMonitor>,
) -> io::Result<Router> {
    // Resolve roots once. Per-request walks use openat with O_NOFOLLOW.
    if let Some(path) = &mut config.private_dir
        && let Ok(canonical) = path.canonicalize()
    {
        *path = canonical;
    }
    if let Some(path) = &mut config.static_dir {
        *path = path.canonicalize()?;
        if !path.is_dir()
            || config
                .private_dir
                .as_ref()
                .is_some_and(|private| private.starts_with(&*path) || path.starts_with(private))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid public asset directory",
            ));
        }
    }
    let state = AppState {
        config,
        auth,
        clients,
        system,
        live: ws::Connections::default(),
    };
    Ok(routes::router(state.clone()).merge(ws::router(state)))
}
