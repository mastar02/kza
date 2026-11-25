//! systemd readiness, watchdog, and shutdown datagrams.

use std::{
    env, io,
    os::unix::{
        ffi::OsStrExt,
        net::{SocketAddr, UnixDatagram},
    },
    time::Duration,
};

#[derive(Debug)]
pub struct Notifier {
    socket: Option<(UnixDatagram, SocketAddr)>,
    watchdog: Option<Duration>,
}

impl Notifier {
    pub fn from_env() -> io::Result<Self> {
        let Some(path) = env::var_os("NOTIFY_SOCKET") else {
            return Ok(Self {
                socket: None,
                watchdog: None,
            });
        };
        let bytes = path.as_os_str().as_bytes();
        let address = if let Some(name) = bytes.strip_prefix(b"@") {
            abstract_addr(name)?
        } else {
            SocketAddr::from_pathname(&path)?
        };
        let socket = UnixDatagram::unbound()?;
        socket.set_nonblocking(true)?;
        let watchdog = if env::var_os("WATCHDOG_PID").is_none_or(|pid| {
            pid.to_str().and_then(|pid| pid.parse::<u32>().ok()) == Some(std::process::id())
        }) {
            env::var("WATCHDOG_USEC")
                .ok()
                .and_then(|micros| micros.parse::<u64>().ok())
                .filter(|micros| *micros > 1)
                .map(|micros| Duration::from_micros(micros / 2))
        } else {
            None
        };
        Ok(Self {
            socket: Some((socket, address)),
            watchdog,
        })
    }

    pub fn send(&self, state: &str) -> io::Result<()> {
        if let Some((socket, address)) = &self.socket {
            socket.send_to_addr(state.as_bytes(), address)?;
        }
        Ok(())
    }

    pub async fn watchdog(&self) {
        let Some(period) = self.watchdog else {
            return std::future::pending().await;
        };
        let mut tick = tokio::time::interval(period);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            if self.send("WATCHDOG=1").is_err() {
                tracing::warn!("systemd watchdog notification failed");
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn abstract_addr(name: &[u8]) -> io::Result<SocketAddr> {
    use std::os::linux::net::SocketAddrExt;
    SocketAddr::from_abstract_name(name)
}

#[cfg(not(target_os = "linux"))]
fn abstract_addr(_: &[u8]) -> io::Result<SocketAddr> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "abstract systemd socket requires Linux",
    ))
}
