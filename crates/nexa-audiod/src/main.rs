//! Nexa's per-room audio daemon.

use std::env;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nexa_audiod::arbiter::WakeArbiter;
use nexa_audiod::capture_store::CaptureStore;
use nexa_audiod::grpc::AudioService;
use nexa_audiod::hw::{amp::Ma1260, simulation::Simulation};
use nexa_audiod::runtime::{RoomPipeline, SharedShadow, run_capture_control, zone_id};
use nexa_audiod::socket::{AudioSocket, Notify};
use nexa_audiod::traits::{
    AmpTransport, CaptureSource, DisabledShadow, PlaybackOutput, ShadowStt, VadEngine, WakeEngine,
};
use nexa_config::{LoadedBundle, Ma1260Connection};
use nexa_proto::sockets;
use nexa_proto::v1::{ZoneState, audio_service_server::AudioServiceServer};
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tonic::transport::Server;
use tracing_subscriber::EnvFilter;

const CAPTURE_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);
const GRPC_SHUTDOWN_GRACE: Duration = Duration::from_millis(300);
const WORKER_THREADS: usize = 4;
const MAX_BLOCKING_THREADS: usize = 16;

type DaemonResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
struct Options {
    config: PathBuf,
    socket: Option<PathBuf>,
    simulation: bool,
}

impl Options {
    fn parse() -> io::Result<Option<Self>> {
        let mut args = env::args_os().skip(1);
        let mut options = Self {
            config: env::var_os("NEXA_CONFIG_DIR")
                .map_or_else(|| "/etc/nexa".into(), PathBuf::from),
            socket: None,
            simulation: false,
        };
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--config-dir" | "--config") => {
                    options.config = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or_else(|| invalid("--config-dir requires a directory"))?
                }
                Some("--socket") => {
                    options.socket = Some(
                        args.next()
                            .map(PathBuf::from)
                            .ok_or_else(|| invalid("--socket requires a path"))?,
                    )
                }
                Some("--simulation") => options.simulation = true,
                Some("-h" | "--help") => return Ok(None),
                _ => return Err(invalid("unknown argument (use --help)")),
            }
        }
        Ok(Some(options))
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn main() -> DaemonResult<()> {
    let Some(options) = Options::parse()? else {
        println!("Usage: nexa-audiod [--config-dir DIR] [--simulation] [--socket PATH]");
        return Ok(());
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(WORKER_THREADS)
        .max_blocking_threads(MAX_BLOCKING_THREADS)
        .enable_all()
        .build()?;
    let result = runtime.block_on(run(options));
    runtime.shutdown_timeout(Duration::from_millis(200));
    result
}

async fn run(options: Options) -> DaemonResult<()> {
    let bundle = LoadedBundle::load_from_dir(&options.config)?;
    let socket = options.socket.unwrap_or_else(|| {
        env::var_os("RUNTIME_DIRECTORY")
            .map(PathBuf::from)
            .or_else(|| env::var_os("XDG_RUNTIME_DIR").map(|dir| Path::new(&dir).join("nexa")))
            .unwrap_or_else(|| bundle.core.paths.runtime_dir.clone())
            .join(sockets::AUDIO)
    });
    let arbiter = WakeArbiter::default();
    if options.simulation {
        let rooms = bundle
            .rooms
            .rooms
            .iter()
            .map(|room| {
                RoomPipeline::new(
                    room,
                    Simulation,
                    Simulation,
                    Simulation,
                    bundle.core.audio.wake_threshold,
                )
                .with_follow_up_window(Duration::from_secs(
                    bundle.core.household.voice_choice_window_s.into(),
                ))
                .with_arbiter(arbiter.clone())
            })
            .collect();
        return serve(
            &bundle,
            socket,
            Ma1260Connection::Simulation,
            Simulation,
            rooms,
            DisabledShadow,
        )
        .await;
    }

    #[cfg(all(target_os = "linux", feature = "hardware"))]
    {
        use nexa_audiod::hw::alsa::{AlsaCapture, AlsaPlayback};
        use nexa_audiod::hw::onnx::{OpenWakeWord, Parakeet, Silero, init_cpu};
        use nexa_audiod::traits::AudioError;

        enum HardwareShadow {
            Disabled,
            Parakeet(Box<Parakeet>),
        }

        impl ShadowStt for HardwareShadow {
            fn is_enabled(&self) -> bool {
                matches!(self, Self::Parakeet(_))
            }

            fn transcribe(&mut self, samples: &[f32]) -> Result<String, AudioError> {
                match self {
                    Self::Disabled => Ok(String::new()),
                    Self::Parakeet(model) => model.transcribe(samples),
                }
            }
        }

        init_cpu()?;
        let models = &bundle.core.paths.models_dir;
        let audio = &bundle.core.audio;
        let shadow = if audio.parakeet_model.trim().is_empty() {
            tracing::info!(
                parakeet_mode = "disabled",
                "shadow STT disabled by configuration"
            );
            HardwareShadow::Disabled
        } else {
            tracing::info!(
                parakeet_mode = "enabled",
                "loading Parakeet shadow STT model"
            );
            HardwareShadow::Parakeet(Box::new(Parakeet::open(
                &models.join(&audio.parakeet_model),
            )?))
        };
        let mut rooms = Vec::new();
        for room in &bundle.rooms.rooms {
            let vad = Silero::open(&models.join(&audio.silero_vad_model))?;
            let wake = OpenWakeWord::open(&models.join(&audio.wake_model))?;
            rooms.push(
                RoomPipeline::new(
                    room,
                    AlsaCapture::new(room.clone()),
                    vad,
                    wake,
                    audio.wake_threshold,
                )
                .with_follow_up_window(Duration::from_secs(
                    bundle.core.household.voice_choice_window_s.into(),
                ))
                .with_arbiter(arbiter.clone()),
            );
        }
        let output = AlsaPlayback::new(bundle.rooms.rooms.clone());
        serve(
            &bundle,
            socket,
            audio.ma1260_connection.clone(),
            output,
            rooms,
            shadow,
        )
        .await
    }
    #[cfg(not(all(target_os = "linux", feature = "hardware")))]
    Err(invalid("use --simulation or build the daemon on Linux with --features hardware").into())
}

async fn serve<C, V, W, P, S>(
    bundle: &LoadedBundle,
    socket_path: PathBuf,
    amp: Ma1260Connection,
    output: P,
    rooms: Vec<RoomPipeline<C, V, W>>,
    shadow: S,
) -> DaemonResult<()>
where
    C: CaptureSource + 'static,
    V: VadEngine + 'static,
    W: WakeEngine + 'static,
    P: PlaybackOutput + 'static,
    S: ShadowStt + 'static,
{
    use nexa_audiod::capture_seal::CaptureTransition;

    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let notify = Notify::from_env()?;
    let mut control = CaptureStore::open(&bundle.core.paths.data_dir)?.restore()?;
    let initial = *control.watch_transitions().borrow();
    match initial {
        Some(CaptureTransition::Seal { revision }) => control.acknowledge_seal(revision)?,
        Some(CaptureTransition::Resume { revision }) => control.acknowledge_resumed(revision)?,
        None => (),
    }

    let capture = Arc::new(Mutex::new(control));
    let zones = bundle.rooms.rooms.iter().map(|room| ZoneState {
        zone_id: zone_id(&room.room_id),
        room_id: room.room_id.clone(),
        ma1260_zone: u32::from(room.ma1260_zone),
        ..Default::default()
    });
    let amp = Ma1260::new(amp);
    tracing::info!(
        amp_mode = amp.mode(),
        "configured MA1260 amplifier transport"
    );
    let service = Arc::new(
        AudioService::new(Arc::clone(&capture), amp, output, zones)?
            .with_barge_in(bundle.core.audio.barge_in_enabled),
    );
    let mut socket = AudioSocket::bind(&socket_path).await?;
    let incoming = socket.incoming()?;

    let (stop, shutdown) = watch::channel(false);
    let mut server_stop = shutdown.clone();
    let server = Server::builder()
        .add_service(AudioServiceServer::from_arc(Arc::clone(&service)))
        .serve_with_incoming_shutdown(incoming, async move {
            let _ = server_stop.wait_for(|stopping| *stopping).await;
        });
    tokio::pin!(server);
    notify.send("READY=1")?;
    tracing::info!(socket = %socket_path.display(), "audio socket ready");

    let shadow = Arc::new(SharedShadow::new(shadow));
    let mut workers = JoinSet::new();
    workers.spawn(run_capture_control(Arc::clone(&capture), shutdown.clone()));
    for room in rooms {
        workers.spawn(room.run(Arc::clone(&service), Arc::clone(&shadow), shutdown.clone()));
    }

    let mut server_finished = false;
    let result: DaemonResult<()> = tokio::select! {
        _ = terminate.recv() => Ok(()),
        _ = interrupt.recv() => Ok(()),
        _ = notify.watchdog() => Ok(()),
        result = &mut server => {
            server_finished = true;
            result.map_err(Into::into)
        }
        worker = workers.join_next() => match worker {
            Some(Ok(Err(error))) => Err(error.into()),
            Some(Err(error)) => Err(error.into()),
            _ => Err(io::Error::other("audio worker stopped unexpectedly").into()),
        },
    };
    if let Err(error) = notify.send("STOPPING=1") {
        tracing::warn!(%error, "systemd stop notification failed");
    }
    stop.send_replace(true);
    let stop_server = async {
        // Stream subscribers may live indefinitely. Give in-flight unary calls
        // time to finish, then release the server even if a client is idle.
        if !server_finished
            && let Ok(result) = tokio::time::timeout(GRPC_SHUTDOWN_GRACE, server).await
        {
            result?;
        }
        Ok::<_, tonic::transport::Error>(())
    };
    let stop_workers = async {
        tokio::time::timeout(CAPTURE_SHUTDOWN_GRACE, async {
            while let Some(result) = workers.join_next().await {
                result??;
            }
            Ok::<_, Box<dyn Error + Send + Sync>>(())
        })
        .await
        .map_err(|_| io::Error::other("audio capture workers did not drain"))?
    };
    let (server, workers) = tokio::join!(stop_server, stop_workers);
    result.and(server.map_err(Into::into)).and(workers)
}
