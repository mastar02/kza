use crate::{CancellationToken, CloudConsentGate, LlmError};
use std::{collections::VecDeque, io, process::Stdio};
use tokio::{
    io::AsyncReadExt,
    process::{ChildStderr, Command},
};

const MAX_STDOUT_BYTES: usize = 4 * 1024 * 1024;
const STDERR_TAIL_BYTES: usize = 4 * 1024;

#[cfg(unix)]
struct ProcessGroup(rustix::process::Pid);

#[cfg(unix)]
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        use rustix::process::{Signal, kill_process_group};
        let _ = kill_process_group(self.0, Signal::KILL);
    }
}

/// Supervise a noninteractive CLI. Pass the cloud gate for commands that use
/// the gateway. The caller owns command arguments and the cancellation deadline.
/// Stdout is limited to 4 MiB. An unsuccessful exit includes an escaped tail of
/// up to 4 KiB of stderr.
pub async fn run_cli(
    mut command: Command,
    cloud: Option<&CloudConsentGate>,
    cancel: &CancellationToken,
) -> Result<Vec<u8>, LlmError> {
    if cancel.is_cancelled() {
        return Err(LlmError::Cancelled);
    }
    if cloud.is_some_and(|gate| !gate.is_allowed()) {
        return Err(LlmError::Consent);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = {
        let _admission = cloud.map(CloudConsentGate::admit).transpose()?;
        command.spawn()?
    };
    #[cfg(unix)]
    let group = child
        .id()
        .and_then(|id| rustix::process::Pid::from_raw(id as i32))
        .map(ProcessGroup);
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing CLI stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing CLI stderr"))?;
    let read_stdout = async {
        let mut output = Vec::new();
        (&mut stdout)
            .take(MAX_STDOUT_BYTES as u64)
            .read_to_end(&mut output)
            .await?;
        if output.len() == MAX_STDOUT_BYTES && stdout.read(&mut [0]).await? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("CLI stdout exceeded {MAX_STDOUT_BYTES} bytes"),
            ));
        }
        Ok(output)
    };
    let done = async {
        let (output, stderr) = tokio::try_join!(read_stdout, read_stderr_tail(stderr))?;
        let status = child.wait().await?;
        if !status.success() {
            let mut message = format!("CLI exited with {status}");
            if !stderr.is_empty() {
                let diagnostic = String::from_utf8_lossy(&stderr);
                message.push_str(". Stderr tail: ");
                message.extend(diagnostic.trim_end().escape_debug());
            }
            return Err(LlmError::Io(io::Error::other(message)));
        }
        Ok(output)
    };
    let withdrawn = async {
        if let Some(gate) = cloud {
            gate.withdrawn.cancelled().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    let result = tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(LlmError::Cancelled),
        _ = withdrawn => Err(LlmError::Consent),
        result = done => result,
    };
    #[cfg(unix)]
    drop(group);
    if result.is_err() && child.id().is_some() {
        let _ = child.kill().await;
    }
    result
}

async fn read_stderr_tail(mut stderr: ChildStderr) -> io::Result<Vec<u8>> {
    let mut tail = VecDeque::with_capacity(STDERR_TAIL_BYTES);
    let mut buffer = [0; STDERR_TAIL_BYTES];
    loop {
        let read = stderr.read(&mut buffer).await?;
        if read == 0 {
            return Ok(tail.into());
        }
        let discard = (tail.len() + read).saturating_sub(STDERR_TAIL_BYTES);
        tail.drain(..discard);
        tail.extend(&buffer[..read]);
    }
}
