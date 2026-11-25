use nexa_llm::{CancellationToken, CloudConsentGate, Endpoint, LlmError, OpenAiClient, run_cli};
use std::{error::Error, sync::Arc, time::Duration};
use tokio::{io::AsyncReadExt, net::TcpListener, process::Command};

type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

#[tokio::test]
async fn cancellation_and_withdrawal_interrupt_pending_http() -> Result {
    for withdraw in [true, false] {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let ep = Endpoint {
            id: "gateway".into(),
            url: format!("http://{}/v1/chat/completions", listener.local_addr()?),
            model: "test".into(),
            cloud: true,
            priority: 1,
            timeout: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(25),
        };
        let gate = Arc::new(CloudConsentGate::new(true));
        let client = OpenAiClient::new(gate.clone())?;
        let cancel = CancellationToken::new();
        let trigger = async {
            let (mut socket, _) = listener.accept().await?;
            socket.read_u8().await?;
            if withdraw {
                gate.withdraw();
            } else {
                cancel.cancel();
            }
            Result::Ok(socket)
        };
        let (socket, result) =
            tokio::join!(trigger, client.stream(&ep, "hola", 1, &cancel, |_| {}));
        let _socket = socket?;
        if withdraw {
            assert!(matches!(result, Err(LlmError::Consent)));
        } else {
            assert!(matches!(result, Err(LlmError::Cancelled)));
        }
    }
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn cli_returns_output_and_reaps_on_withdrawal() -> Result {
    let cancel = CancellationToken::new();
    let mut cmd = Command::new("/bin/echo");
    cmd.arg("hola");
    assert_eq!(run_cli(cmd, None, &cancel).await?, b"hola\n");

    let dir = tempfile::tempdir()?;
    let marker = dir.path().join("pid");
    let gate = CloudConsentGate::new(true);
    let mut cmd = Command::new("/bin/sh");
    cmd.args([
        "-c",
        "echo $$ > pid.tmp && mv pid.tmp pid && exec cat blocked",
        "test",
    ])
    .current_dir(dir.path());
    assert!(
        Command::new("mkfifo")
            .arg(dir.path().join("blocked"))
            .status()
            .await?
            .success()
    );
    let withdraw = async {
        loop {
            match std::fs::read_to_string(&marker) {
                Ok(value) => {
                    let pid = rustix::process::Pid::from_raw(value.trim().parse()?)
                        .ok_or("invalid pid")?;
                    gate.withdraw();
                    return Result::Ok(pid);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tokio::task::yield_now().await
                }
                Err(e) => return Err(e.into()),
            }
        }
    };
    let (pid, result) = tokio::join!(withdraw, run_cli(cmd, Some(&gate), &cancel));
    assert!(matches!(result, Err(LlmError::Consent)));
    assert_eq!(
        rustix::process::test_kill_process(pid?),
        Err(rustix::io::Errno::SRCH)
    );
    assert!(matches!(
        run_cli(Command::new("absent-cli"), Some(&gate), &cancel).await,
        Err(LlmError::Consent)
    ));
    Ok(())
}
