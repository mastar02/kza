//! Private daemon socket and systemd's Unix datagram notification protocol.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rustix::fs::{FlockOperation, flock};
use tokio::net::{UnixListener, UnixStream};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnixListenerStream;

use crate::capture_store::private_directory;

#[derive(Debug)]
pub struct AudioSocket {
    path: PathBuf,
    device: u64,
    inode: u64,
    listener: Option<UnixListener>,
    _lock: File,
}

impl AudioSocket {
    /// Refuse live sockets and unrelated files. The binding lock serializes
    /// stale cleanup and bind across audiod processes using this path.
    pub async fn bind(path: &Path) -> io::Result<Self> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        private_directory(parent)?;
        let mut lock_path = path.as_os_str().to_owned();
        lock_path.push(".lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32)
            .open(lock_path)?;
        flock(&lock, FlockOperation::NonBlockingLockExclusive)?;
        match path.symlink_metadata() {
            Ok(metadata)
                if metadata.file_type().is_socket()
                    && metadata.uid() == rustix::process::geteuid().as_raw() =>
            {
                match tokio::time::timeout(Duration::from_millis(250), UnixStream::connect(path))
                    .await
                {
                    Ok(Err(error)) if error.kind() == io::ErrorKind::ConnectionRefused => {
                        let current = path.symlink_metadata()?;
                        if current.dev() != metadata.dev() || current.ino() != metadata.ino() {
                            return Err(io::ErrorKind::AddrInUse.into());
                        }
                        fs::remove_file(path)?;
                    }
                    _ => return Err(io::ErrorKind::AddrInUse.into()),
                }
            }
            Ok(_) => return Err(io::ErrorKind::AlreadyExists.into()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
        let listener = UnixListener::bind(path)?;
        let metadata = path.symlink_metadata()?;
        let socket = Self {
            path: path.to_owned(),
            device: metadata.dev(),
            inode: metadata.ino(),
            listener: Some(listener),
            _lock: lock,
        };
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(socket)
    }

    pub fn incoming(
        &mut self,
    ) -> io::Result<impl tokio_stream::Stream<Item = io::Result<UnixStream>> + use<>> {
        let listener = self.listener.take().ok_or(io::ErrorKind::NotConnected)?;
        Ok(
            UnixListenerStream::new(listener).filter_map(|result| match result {
                #[cfg(target_os = "linux")]
                Ok(stream) => {
                    let uid = rustix::process::geteuid().as_raw();
                    if stream
                        .peer_cred()
                        .is_ok_and(|credentials| credentials.uid() == uid)
                    {
                        Some(Ok(stream))
                    } else {
                        tracing::warn!("rejected audio socket connection with unexpected peer UID");
                        None
                    }
                }
                result => Some(result),
            }),
        )
    }
}

impl Drop for AudioSocket {
    fn drop(&mut self) {
        if self.path.symlink_metadata().is_ok_and(|metadata| {
            metadata.file_type().is_socket()
                && metadata.dev() == self.device
                && metadata.ino() == self.inode
        }) && let Err(error) = fs::remove_file(&self.path)
        {
            tracing::warn!(%error, "could not remove audio socket");
        }
    }
}

#[derive(Debug)]
pub struct Notify {
    socket: Option<(UnixDatagram, SocketAddr)>,
    watchdog: Option<Duration>,
}

impl Notify {
    pub fn from_env() -> io::Result<Self> {
        let socket = env::var_os("NOTIFY_SOCKET")
            .filter(|value| !value.is_empty())
            .map(|path| {
                #[cfg(target_os = "linux")]
                let address = {
                    use std::os::linux::net::SocketAddrExt;
                    use std::os::unix::ffi::OsStrExt;
                    match path.as_bytes().strip_prefix(b"@") {
                        Some(name) => SocketAddr::from_abstract_name(name)?,
                        None => SocketAddr::from_pathname(path)?,
                    }
                };
                #[cfg(not(target_os = "linux"))]
                let address = SocketAddr::from_pathname(path)?;
                let socket = UnixDatagram::unbound()?;
                socket.set_nonblocking(true)?;
                Ok::<_, io::Error>((socket, address))
            })
            .transpose()?;
        let own_watchdog = env::var("WATCHDOG_PID")
            .map(|pid| pid.parse::<u32>() == Ok(std::process::id()))
            .unwrap_or(true);
        let watchdog = if socket.is_some() && own_watchdog {
            env::var("WATCHDOG_USEC")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|&micros| micros > 0)
                .map(|micros| Duration::from_micros((micros / 2).max(1)))
        } else {
            None
        };
        Ok(Self { socket, watchdog })
    }

    pub fn send(&self, state: &str) -> io::Result<()> {
        if let Some((socket, address)) = &self.socket {
            socket.send_to_addr(state.as_bytes(), address)?;
        }
        Ok(())
    }

    pub async fn watchdog(&self) {
        if let Some(period) = self.watchdog {
            loop {
                tokio::time::sleep(period).await;
                if let Err(error) = self.send("WATCHDOG=1") {
                    tracing::warn!(%error, "systemd watchdog notification failed");
                }
            }
        } else {
            std::future::pending::<()>().await;
        }
    }
}
