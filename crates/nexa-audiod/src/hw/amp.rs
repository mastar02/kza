//! Serialized MA1260 command/ACK transport over TCP or RS-232.

use std::fmt;
use std::time::Duration;

use nexa_config::Ma1260Connection;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crate::ma1260::{CommandFrame, is_acknowledgment};
use crate::traits::{AmpTransport, AudioError};

const IO_TIMEOUT: Duration = Duration::from_secs(1);

trait AmpStream: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> AmpStream for T {}

pub struct Ma1260 {
    connection: Ma1260Connection,
    stream: Mutex<Option<Box<dyn AmpStream>>>,
}

impl fmt::Debug for Ma1260 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Ma1260").finish_non_exhaustive()
    }
}

impl Ma1260 {
    pub fn new(connection: Ma1260Connection) -> Self {
        Self {
            connection,
            stream: Mutex::default(),
        }
    }

    async fn connect(&self) -> Result<Box<dyn AmpStream>, AudioError> {
        match &self.connection {
            Ma1260Connection::Tcp { host, port } => {
                let stream = TcpStream::connect((host.as_str(), *port)).await?;
                stream.set_nodelay(true)?;
                Ok(Box::new(stream))
            }
            #[cfg(all(target_os = "linux", feature = "hardware"))]
            Ma1260Connection::Serial { device, baud_rate } => {
                use tokio_serial::SerialPortBuilderExt;
                let path = device.to_str().ok_or(AudioError::Rejected)?;
                let stream = tokio_serial::new(path, *baud_rate)
                    .open_native_async()
                    .map_err(|error| AudioError::Adapter(error.to_string()))?;
                Ok(Box::new(stream))
            }
            _ => Err(AudioError::Unavailable),
        }
    }
}

impl AmpTransport for Ma1260 {
    async fn send(&self, command: CommandFrame) -> Result<(), AudioError> {
        if self.connection == Ma1260Connection::Simulation {
            return Ok(());
        }
        let mut cached = self.stream.lock().await;
        let previous = cached.take();
        let reused = previous.is_some();
        let exchange = |previous: Option<Box<dyn AmpStream>>| async move {
            let mut stream = match previous {
                Some(stream) => stream,
                None => self.connect().await?,
            };
            stream.write_all(&command.encode()).await?;
            stream.flush().await?;
            let ack = stream.read_u8().await?;
            if !is_acknowledgment(&[ack]) {
                return Err(AudioError::Rejected);
            }
            Ok(stream)
        };

        // Taking the cached stream also makes cancellation discard a partial transaction.
        match tokio::time::timeout(IO_TIMEOUT, exchange(previous)).await {
            Ok(Ok(stream)) => *cached = Some(stream),
            // The codec uses absolute setters and a query, so reconnect/retry is idempotent.
            Ok(Err(AudioError::Io(_))) | Err(_) if reused => {
                *cached = Some(
                    tokio::time::timeout(IO_TIMEOUT, exchange(None))
                        .await
                        .map_err(|_| AudioError::Unavailable)??,
                );
            }
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(AudioError::Unavailable),
        }
        Ok(())
    }
}
