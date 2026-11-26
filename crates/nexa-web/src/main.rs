//! HTTP entry point for the Nexa control plane.

use std::{
    env, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::Parser;
use nexa_config::NexaConfig;
#[cfg(feature = "test-mode")]
use nexa_web::TestMode;
use nexa_web::{
    WebConfig,
    auth::{AuthStore, Origin},
    build_app,
    clients::Clients,
    sys::PlatformMonitor,
};

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Use temporary authority and in-process fixtures on loopback.
    #[cfg(feature = "test-mode")]
    #[arg(long)]
    test_mode: bool,
    /// Local HTTP listener address.
    #[arg(long)]
    bind: Option<SocketAddr>,
    /// Browser origin including scheme, hostname and port.
    #[arg(long)]
    origin: Option<String>,
    /// Drop Secure from session cookies for loopback HTTP.
    #[arg(long)]
    allow_insecure_cookies: bool,
    /// Directory containing the built Vite application.
    #[arg(long)]
    static_dir: Option<PathBuf>,
    /// Runtime configuration file or directory.
    #[arg(long, default_value = "/etc/nexa")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ansi = io::IsTerminal::is_terminal(&io::stdout()) && std::env::var_os("NO_COLOR").is_none();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .with_ansi(ansi)
        .init();
    let args = Args::parse();
    #[cfg(feature = "test-mode")]
    if args.test_mode {
        let bind = args
            .bind
            .unwrap_or(SocketAddr::from(([127, 0, 0, 1], 9500)));
        require_loopback(bind)?;
        let listener = tokio::net::TcpListener::bind(bind).await?;
        let default_origin = format!("http://{}", listener.local_addr()?);
        let origin = Origin::parse(args.origin.as_deref().unwrap_or(&default_origin))?;
        let fixtures = TestMode::new().await?;
        let app = fixtures.app(origin, args.allow_insecure_cookies, args.static_dir)?;
        return serve(listener, app).await;
    }

    let path = if args.config.is_dir() {
        args.config.join("nexa.toml")
    } else {
        args.config
    };
    let config = NexaConfig::load_file(path)?;
    let bind = match args.bind {
        Some(bind) => bind,
        None => format!("{}:{}", config.web.bind_host, config.web.port).parse()?,
    };
    if args.allow_insecure_cookies {
        require_loopback(bind)?;
    }
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let default_origin = format!(
        "https://{}",
        SocketAddr::new(bind.ip(), config.web.proxy_tls_port)
    );
    let origin = Origin::parse(
        args.origin
            .or_else(|| std::env::var("NEXA_WEB_ORIGIN").ok())
            .as_deref()
            .unwrap_or(&default_origin),
    )?;
    let auth = AuthStore::open_in(&config.paths.data_dir)?;
    if let Ok(token) = std::env::var("KZA_DASHBOARD_TOKEN") {
        auth.bootstrap_operator(token).await?;
    }
    let runtime_dir = env::var_os("RUNTIME_DIRECTORY")
        .map(PathBuf::from)
        .or_else(|| env::var_os("XDG_RUNTIME_DIR").map(|dir| Path::new(&dir).join("nexa")))
        .unwrap_or(config.paths.runtime_dir);
    let clients = Clients::uds(&runtime_dir);
    let web = WebConfig {
        origin,
        allow_insecure_cookies: args.allow_insecure_cookies,
        private_dir: config
            .web
            .home_enabled
            .then_some(config.web.private_assets_dir),
        static_dir: args.static_dir,
    };
    let app = build_app(web, auth, clients, Arc::new(PlatformMonitor))?;
    serve(listener, app).await
}

fn require_loopback(bind: SocketAddr) -> io::Result<()> {
    if !bind.ip().is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a loopback listener is required",
        ));
    }
    Ok(())
}

async fn serve(
    listener: tokio::net::TcpListener,
    app: axum::Router,
) -> Result<(), Box<dyn std::error::Error>> {
    let local = listener.local_addr()?;
    #[cfg(target_os = "linux")]
    let _watchdog = notify_systemd()?;
    tracing::info!(address = %local, "nexa-web listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! { _ = interrupt => {}, _ = terminate => {} }
}

#[cfg(target_os = "linux")]
fn notify_systemd() -> io::Result<Option<tokio::task::JoinHandle<()>>> {
    use sd_notify::NotifyState;
    if std::env::var_os("NOTIFY_SOCKET").is_none() {
        return Ok(None);
    }
    sd_notify::notify(false, &[NotifyState::Ready])?;
    let mut micros = 0;
    if !sd_notify::watchdog_enabled(false, &mut micros) || micros < 2 {
        return Ok(None);
    }
    let period = std::time::Duration::from_micros(micros / 2);
    Ok(Some(tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        // The READY notification already marks the initial healthy point.
        interval.tick().await;
        loop {
            interval.tick().await;
            if sd_notify::notify(false, &[NotifyState::Watchdog]).is_err() {
                tracing::warn!("systemd watchdog notification failed");
                return;
            }
        }
    })))
}
