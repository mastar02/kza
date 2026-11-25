//! HTTP entry point for the Nexa control plane.

use std::{io, net::SocketAddr, path::PathBuf, sync::Arc};

use clap::Parser;
use nexa_config::NexaConfig;
use nexa_web::{
    TestMode, WebConfig,
    auth::{AuthStore, Origin},
    build_app,
    clients::Clients,
    sys::PlatformMonitor,
};

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Use temporary authority and in-process fixtures on loopback.
    #[arg(long)]
    test_mode: bool,
    /// Local HTTP listener. Supports port 0 in test mode.
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
    /// Runtime configuration file or directory used outside test mode.
    #[arg(long, default_value = "/etc/nexa")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ansi = io::IsTerminal::is_terminal(&io::stdout()) && std::env::var_os("NO_COLOR").is_none();
    tracing_subscriber::fmt()
        .with_target(true)
        .with_ansi(ansi)
        .init();
    let args = Args::parse();
    let production = if args.test_mode {
        None
    } else {
        let path = if args.config.is_dir() {
            args.config.join("nexa.toml")
        } else {
            args.config
        };
        Some(NexaConfig::load_file(path)?)
    };
    let bind = match (args.bind, &production) {
        (Some(bind), _) => bind,
        (None, Some(config)) => format!("{}:{}", config.web.bind_host, config.web.port).parse()?,
        (None, None) => SocketAddr::from(([127, 0, 0, 1], 9500)),
    };
    if (args.test_mode || args.allow_insecure_cookies) && !bind.ip().is_loopback() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "test mode and insecure cookies require a loopback listener",
        )
        .into());
    }
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let local = listener.local_addr()?;
    let default_origin = match &production {
        Some(config) => format!(
            "https://{}",
            SocketAddr::new(bind.ip(), config.web.proxy_tls_port)
        ),
        None => format!("http://{local}"),
    };
    let configured_origin = production
        .as_ref()
        .and_then(|_| std::env::var("NEXA_WEB_ORIGIN").ok());
    let origin = Origin::parse(
        args.origin
            .or(configured_origin)
            .as_deref()
            .unwrap_or(&default_origin),
    )?;
    let (app, _fixtures) = if let Some(config) = production {
        let auth = AuthStore::open(&config.paths.data_dir.join("auth.db"))?;
        if let Ok(token) = std::env::var("KZA_DASHBOARD_TOKEN") {
            auth.bootstrap_operator(token).await?;
        }
        let clients = Clients::uds(&config.paths.runtime_dir);
        let web = WebConfig {
            origin,
            allow_insecure_cookies: args.allow_insecure_cookies,
            private_dir: config
                .web
                .home_enabled
                .then_some(config.web.private_assets_dir),
            static_dir: args.static_dir,
        };
        (
            build_app(web, auth, clients, Arc::new(PlatformMonitor))?,
            None,
        )
    } else {
        let fixtures = TestMode::new().await?;
        let app = fixtures.app(origin, args.allow_insecure_cookies, args.static_dir)?;
        (app, Some(fixtures))
    };
    #[cfg(target_os = "linux")]
    let _watchdog = notify_systemd()?;
    tracing::info!(address = %local,test_mode = args.test_mode,"nexa-web listening");
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
