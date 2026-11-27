//! Exclusive SQLite + sqlite-vec writer serving the private memory Unix socket.

use std::path::{Path, PathBuf};

use nexa_config::NexaConfig;
use nexa_memoryd::db::{FACT_MODEL, Store};
use nexa_memoryd::service::{BoundSocket, MemoryRpc};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::{FilterExt, filter_fn};
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // Keep dependency diagnostics out of the daemon's log, including at trace level.
    let daemon_targets = filter_fn(|metadata| {
        metadata.target() == "nexa_memoryd" || metadata.target().starts_with("nexa_memoryd::")
    });
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(daemon_targets.and(env_filter)))
        .init();
    let config_dir = config_directory()?;
    let config = NexaConfig::load_file(config_dir.join("nexa.toml"))?;
    if config.ml.embeddings_model != FACT_MODEL {
        return Err("unsupported fact embeddings model".into());
    }
    let mut store = Store::open(&config.paths.data_dir.join("memory"))?;
    // Seed the persisted operator switch from deployment config on first use.
    store.configure_voice_attribution(config.memory.voice_proposals_enabled)?;
    let runtime_dir = std::env::var_os("RUNTIME_DIRECTORY")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_RUNTIME_DIR").map(|dir| Path::new(&dir).join("nexa")))
        .unwrap_or(config.paths.runtime_dir);
    let socket = BoundSocket::bind(&runtime_dir)?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let watchdog = notify_ready()?;
    tracing::info!(
        service = "nexa-memoryd",
        ambient_capture_enabled = config.memory.ambient_enabled,
        "private memory socket ready"
    );

    let rpc = MemoryRpc::new(store, config.memory.ambient_enabled);
    let prune_task = config.memory.ambient_enabled.then(|| {
        let prune_rpc = rpc.clone();
        let retention_days = u32::from(config.memory.automatic_retention_days);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Skip immediate boot tick so startup stays lightweight.
            interval.tick().await;
            loop {
                interval.tick().await;
                if let Err(error) = prune_rpc.prune_expired_ambient(retention_days).await {
                    tracing::warn!(%error, "periodic ambient TTL cleanup failed");
                }
            }
        })
    });

    let result = socket
        .serve(rpc, async {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {},
                _ = terminate.recv() => {},
            }
            notify_stopping();
        })
        .await;
    if let Some(prune_task) = prune_task {
        prune_task.abort();
    }
    if let Some(watchdog) = watchdog {
        watchdog.abort();
    }
    result?;
    Ok(())
}

fn config_directory() -> Result<PathBuf, &'static str> {
    let mut args = std::env::args_os().skip(1);
    match args.next() {
        None => Ok(PathBuf::from("/etc/nexa")),
        Some(flag) if flag == "--config" => {
            let dir = args.next().ok_or("--config requires a directory")?;
            if args.next().is_some() {
                return Err("usage: nexa-memoryd [--config DIRECTORY]");
            }
            Ok(PathBuf::from(dir))
        }
        _ => Err("usage: nexa-memoryd [--config DIRECTORY]"),
    }
}

#[cfg(target_os = "linux")]
fn notify_ready() -> std::io::Result<Option<tokio::task::JoinHandle<()>>> {
    use sd_notify::{NotifyState, notify, watchdog_enabled};

    notify(&[NotifyState::Ready])?;
    Ok(watchdog_enabled().map(|period| {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(period / 2);
            loop {
                interval.tick().await;
                if notify(&[NotifyState::Watchdog]).is_err() {
                    tracing::error!("systemd watchdog notification failed");
                }
            }
        })
    }))
}

#[cfg(target_os = "linux")]
fn notify_stopping() {
    if sd_notify::notify(&[sd_notify::NotifyState::Stopping]).is_err() {
        tracing::error!("systemd stopping notification failed");
    }
}

#[cfg(not(target_os = "linux"))]
fn notify_stopping() {}

#[cfg(not(target_os = "linux"))]
fn notify_ready() -> std::io::Result<Option<tokio::task::JoinHandle<()>>> {
    Ok(None)
}
