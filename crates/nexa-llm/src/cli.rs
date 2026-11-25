use crate::{CancellationToken, CloudConsentGate, LlmError};
use std::{io, process::Stdio};
use tokio::{io::AsyncReadExt, process::Command};

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
        .stderr(Stdio::null())
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
    let mut output = Vec::new();
    let done = async {
        stdout.read_to_end(&mut output).await?;
        let status = child.wait().await?;
        if !status.success() {
            return Err(LlmError::Io(io::Error::other(format!(
                "CLI exited with {status}"
            ))));
        }
        Ok(())
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
        child.kill().await?;
    }
    result.map(|()| output)
}
