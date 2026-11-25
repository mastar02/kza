use std::{
    error::Error,
    fs, io,
    os::unix::{fs::PermissionsExt, net::UnixDatagram},
    process::{Child, Command},
    time::{Duration, Instant},
};

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn readiness_watchdog_sigterm_and_socket_cleanup() -> Result<(), Box<dyn Error>> {
    let tmp = tempfile::tempdir_in("/tmp")?;
    let config = tmp.path().join("config");
    fs::create_dir(&config)?;
    let runtime = tmp.path().join("r");
    let secrets = tmp.path().join("secrets");
    let data = tmp.path().join("data");
    let toml = include_str!("../../../config/nexa/nexa.toml")
        .replace("/var/lib/nexa", &data.to_string_lossy())
        .replace("/run/user/nexa/nexa", &runtime.to_string_lossy())
        .replace("/etc/nexa/secrets", &secrets.to_string_lossy());
    fs::write(config.join("nexa.toml"), toml)?;
    fs::write(
        config.join("rooms.toml"),
        include_str!("../../../config/nexa/rooms.example.toml"),
    )?;

    let path = tmp.path().join("notify");
    let receiver = UnixDatagram::bind(&path)?;
    receiver.set_read_timeout(Some(Duration::from_secs(8)))?;
    let mut daemon = Daemon(
        Command::new(env!("CARGO_BIN_EXE_nexa-brain"))
            .args(["--standalone-test", "--config-dir"])
            .arg(&config)
            .env("NOTIFY_SOCKET", &path)
            .env("WATCHDOG_USEC", "100000")
            .env_remove("WATCHDOG_PID")
            .env_remove("HOME_ASSISTANT_TOKEN")
            .spawn()?,
    );

    let mut buffer = [0; 128];
    let len = receiver
        .recv(&mut buffer)
        .map_err(|error| io::Error::other(format!("waiting for READY: {error}")))?;
    assert_eq!(&buffer[..len], b"READY=1");
    let brain = runtime.join("brain.sock");
    assert_eq!(fs::metadata(&brain)?.permissions().mode() & 0o777, 0o600);
    assert_eq!(fs::metadata(&runtime)?.permissions().mode() & 0o777, 0o700);
    let len = receiver
        .recv(&mut buffer)
        .map_err(|error| io::Error::other(format!("waiting for WATCHDOG: {error}")))?;
    assert_eq!(&buffer[..len], b"WATCHDOG=1");
    let started = Instant::now();
    receiver.set_read_timeout(Some(Duration::from_secs(3)))?;
    let pid = rustix::process::Pid::from_raw(i32::try_from(daemon.0.id())?)
        .ok_or("invalid daemon PID")?;
    rustix::process::kill_process(pid, rustix::process::Signal::TERM)?;
    loop {
        let len = receiver
            .recv(&mut buffer)
            .map_err(|error| io::Error::other(format!("waiting for STOPPING: {error}")))?;
        if &buffer[..len] == b"STOPPING=1" {
            break;
        }
        if started.elapsed() >= Duration::from_secs(3) {
            return Err(io::Error::other("STOPPING notification missing").into());
        }
    }
    loop {
        if let Some(status) = daemon.0.try_wait()? {
            assert!(status.success(), "{status}");
            break;
        }
        if started.elapsed() >= Duration::from_secs(3) {
            return Err(io::Error::other("daemon shutdown exceeded three seconds").into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(!brain.exists());
    assert!(data.join("household/household.db").is_file());
    Ok(())
}
