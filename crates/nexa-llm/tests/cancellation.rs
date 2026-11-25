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

#[cfg(unix)]
#[tokio::test]
async fn cli_limits_stdout_and_reaps_child_on_overflow() -> Result {
    let cancel = CancellationToken::new();
    let timeout = Duration::from_secs(10);
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", "head -c 4194304 /dev/zero"]);
    let output = tokio::time::timeout(timeout, run_cli(cmd, None, &cancel)).await??;
    assert_eq!(output.len(), 4 * 1024 * 1024);
    assert!(output.iter().all(|&byte| byte == 0));

    let dir = tempfile::tempdir()?;
    let mut cmd = Command::new("/bin/sh");
    cmd.args([
        "-c",
        "echo $$ > pid && head -c 4194305 /dev/zero && exec sleep 30",
    ])
    .current_dir(dir.path());
    let error = tokio::time::timeout(timeout, run_cli(cmd, None, &cancel))
        .await?
        .err()
        .ok_or("oversized stdout must fail before EOF")?;
    assert!(
        matches!(error, LlmError::Io(ref source) if source.kind() == std::io::ErrorKind::InvalidData)
    );
    assert!(
        error
            .to_string()
            .contains("CLI stdout exceeded 4194304 bytes")
    );

    let pid = std::fs::read_to_string(dir.path().join("pid"))?;
    let pid = rustix::process::Pid::from_raw(pid.trim().parse()?).ok_or("invalid pid")?;
    assert_eq!(
        rustix::process::test_kill_process(pid),
        Err(rustix::io::Errno::SRCH)
    );
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn cli_reports_only_sanitized_stderr_tail_on_failure() -> Result {
    let mut cmd = Command::new("/bin/sh");
    cmd.args([
        "-c",
        "printf 'discarded stderr prefix\\n' >&2; \
         head -c 131072 /dev/zero >&2; \
         printf 'stdout payload'; \
         printf '\\033[31m\\r\\377diagnostic tail\\n' >&2; exit 7",
    ]);
    let cancel = CancellationToken::new();
    let error = tokio::time::timeout(Duration::from_secs(10), run_cli(cmd, None, &cancel))
        .await?
        .err()
        .ok_or("CLI exit 7 must fail")?;
    assert!(matches!(error, LlmError::Io(_)));
    let message = error.to_string();
    assert!(message.contains("exit status: 7"));
    assert!(message.contains(r"\u{1b}[31m\r�diagnostic tail"));
    assert!(message.chars().all(|character| !character.is_control()));
    assert!(!message.contains("discarded stderr prefix"));
    assert!(!message.contains("stdout payload"));
    // The escaped fill bytes take two characters each, and the tail is 4 KiB.
    assert!(message.len() < 2 * 4096 + 128);
    Ok(())
}
