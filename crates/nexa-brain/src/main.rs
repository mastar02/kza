//! Entry point for the Nexa voice and control daemon.

use std::{
    env, io,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use nexa_brain::{
    Brain, BrainError, Dependencies, HomeDependencies,
    clients::{UdsAudio, UdsMemory, UdsMl, channel},
    control::BoundControl,
    notify::Notifier,
};
use nexa_config::LoadedBundle;
use nexa_household::SystemClock;
use nexa_proto::{
    sockets,
    v1::{
        audio_service_client::AudioServiceClient, memory_service_client::MemoryServiceClient,
        ml_service_client::MlServiceClient,
    },
};
use tokio::signal::unix::{SignalKind, signal};

#[derive(Debug)]
struct Options {
    config_dir: PathBuf,
    standalone: bool,
}

impl Options {
    fn parse() -> io::Result<Self> {
        let mut args = env::args_os().skip(1);
        let mut config_dir = env::var_os("NEXA_CONFIG_DIR").map(PathBuf::from);
        let mut standalone = false;
        while let Some(argument) = args.next() {
            match argument.to_str() {
                Some("--config-dir" | "--config") => {
                    config_dir = Some(PathBuf::from(args.next().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "--config-dir requires a directory",
                        )
                    })?));
                }
                Some("--standalone-test") => standalone = true,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "usage: nexa-brain [--config-dir PATH] [--standalone-test]",
                    ));
                }
            }
        }
        Ok(Self {
            config_dir: config_dir.unwrap_or_else(|| PathBuf::from("/etc/nexa")),
            standalone,
        })
    }
}

async fn home_token(secrets_path: &Path, standalone: bool) -> io::Result<Option<String>> {
    if let Ok(token) = env::var("HOME_ASSISTANT_TOKEN") {
        return Ok(Some(token.trim().into()));
    }
    match tokio::fs::read_to_string(secrets_path.join("ha-token")).await {
        Ok(token) => Ok(Some(token.trim().into())),
        Err(error) if standalone && error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

async fn run() -> Result<(), BrainError> {
    let options = Options::parse()?;
    let mut term = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let config =
        tokio::task::spawn_blocking(move || LoadedBundle::load_from_dir(options.config_dir))
            .await??;
    let runtime = env::var_os("RUNTIME_DIRECTORY")
        .map(PathBuf::from)
        .or_else(|| env::var_os("XDG_RUNTIME_DIR").map(|dir| Path::new(&dir).join("nexa")))
        .unwrap_or_else(|| config.core.paths.runtime_dir.clone());
    let bound = BoundControl::bind(&runtime.join(sockets::BRAIN))?;
    let token = home_token(&config.core.paths.secrets_dir, options.standalone).await?;
    let home = token
        .as_deref()
        .map(|token| HomeDependencies::new(&config, token))
        .transpose()?;
    let audio = Arc::new(UdsAudio(
        AudioServiceClient::new(channel(&runtime.join(sockets::AUDIO)))
            .max_decoding_message_size(5 * 1024 * 1024),
    ));
    let ml = Arc::new(UdsMl(MlServiceClient::new(channel(
        &runtime.join(sockets::ML),
    ))));
    let memory = (config.core.memory.enabled || config.core.memory.ambient_enabled).then(|| {
        Arc::new(UdsMemory(MemoryServiceClient::new(channel(
            &runtime.join(sockets::MEMORY),
        )))) as Arc<dyn nexa_brain::clients::MemoryClient>
    });
    let brain = Brain::open(
        config,
        Dependencies {
            audio,
            ml,
            memory,
            home,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let notify = Notifier::from_env()?;

    let mut serve = Box::pin(bound.serve(Arc::clone(&brain)));
    let mut jobs = Box::pin(Arc::clone(&brain).run_background());
    notify.send("READY=1")?;
    tracing::info!("nexa-brain listening");
    let server_outcome = tokio::select! {
        result = &mut serve => Some(result),
        _ = &mut jobs => Some(Err(io::Error::other("brain background jobs stopped").into())),
        _ = term.recv() => None,
        _ = interrupt.recv() => None,
        _ = notify.watchdog() => None,
    };
    if notify.send("STOPPING=1").is_err() {
        tracing::warn!("systemd stop notification failed");
    }
    brain.shutdown();
    if let Some(result) = server_outcome {
        return result;
    }
    match tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(serve, jobs).0
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            tracing::warn!("brain shutdown grace period elapsed");
            Ok(())
        }
    }
}

fn main() -> Result<(), BrainError> {
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(run());
    runtime.shutdown_timeout(Duration::from_millis(400));
    result
}
