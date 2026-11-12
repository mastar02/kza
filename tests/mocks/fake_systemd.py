"""A small systemd-like supervisor for the managed launcher rehearsals (U3b-0d).

It models only the semantics the launcher depends on (design v1 tests.fake_supervisor,
v2 tests_v2.fake_supervisor, v3 N4/N9):

* ExecStart in a new process group, NOTIFY_SOCKET (AF_UNIX datagram) and,
  with WatchdogSec, WATCHDOG_USEC/WATCHDOG_PID for the main process.
* Type=notify: READY=1, STATUS=, STOPPING=1, EXTEND_TIMEOUT_USEC= and WATCHDOG=1,
  with TimeoutStartSec and a watchdog armed only after READY. NotifyAccess=main
  is enforced with SCM_CREDENTIALS where the OS provides them (Linux); on
  macOS the sender cannot be identified and the log records it.
* KillMode=mixed: KillSignal to the main PID only; once the main process is
  gone, or at TimeoutStopSec, SIGKILL to the whole group. A watchdog expiry
  sends WatchdogSignal and uses TimeoutAbortSec.
* Restart=on-failure with RestartSec, RestartPreventExitStatus,
  StartLimitIntervalSec and StartLimitBurst.
* An outer user-manager cap: ``manager_stop(budget)`` marks the system as
  stopping, stops the unit, and SIGKILLs the whole group when the budget ends.
* A ``systemctl`` shim answering the controller's read-only queries from files.

Every transition is appended to ``log`` as (seconds since start, event, detail).
It is validated against real systemd 255 only on the server (not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[2]


def _seconds(value: str) -> float:
    value = value.strip()
    if value in {"", "infinity"}:
        return 0.0
    if value.endswith("min"):
        return float(value[:-3]) * 60
    if value.endswith("ms"):
        return float(value[:-2]) / 1000
    if value.endswith("s"):
        return float(value[:-1])
    return float(value)


def parse_dropins(texts: list[str], base: dict | None = None) -> dict:
    """Overlay drop-in texts (already in lexical order) on base directives.

    ExecStart= and RestartPreventExitStatus= reset; Environment= accumulates;
    every other directive is single-valued (the last one wins).
    """
    directives = {key: (list(value) if isinstance(value, list) else value)
                  for key, value in (base or {}).items()}
    directives.setdefault("ExecStart", [])
    directives.setdefault("Environment", [])
    for text in texts:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", ";", "[")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "ExecStart":
                directives["ExecStart"] = [] if not value else [*directives["ExecStart"], value]
            elif key == "Environment":
                directives["Environment"] = [*directives["Environment"], *shlex.split(value)]
            else:
                directives[key] = value
    return directives


def unit_dropins(unit_dir: Path) -> list[str]:
    """The texts of every *.conf drop-in, in lexical file-name order."""
    if not unit_dir.is_dir():
        return []
    return [path.read_text() for path in sorted(unit_dir.glob("*.conf"))]


def show_properties(directives: dict) -> dict[str, str]:
    """``systemctl show`` style values for the properties the launcher reads."""
    argv = shlex.split(directives["ExecStart"][-1]) if directives["ExecStart"] else []
    exec_start = ("{ path=" + (argv[0] if argv else "") + " ; argv[]=" + shlex.join(argv)
                  + " ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
                  if argv else "")
    values = {"Type": directives.get("Type", "simple"),
              "NotifyAccess": directives.get("NotifyAccess", "none"),
              "KillMode": directives.get("KillMode", "control-group"),
              "RestartPreventExitStatus": directives.get("RestartPreventExitStatus", ""),
              "StandardOutput": directives.get("StandardOutput", "journal"),
              "StandardError": directives.get("StandardError", "inherit"),
              "WorkingDirectory": directives.get("WorkingDirectory", ""),
              "ExecStart": exec_start, "Environment": shlex.join(directives["Environment"])}
    for directive, key, default in (("TimeoutStartSec", "TimeoutStartUSec", "1min 30s"),
                                    ("TimeoutStopSec", "TimeoutStopUSec", "1min 30s"),
                                    ("WatchdogSec", "WatchdogUSec", "0")):
        raw = directives.get(directive)
        values[key] = default if raw is None else (f"{_seconds(raw):g}s" if _seconds(raw) else "0")
    return values


@dataclass
class UnitConfig:
    """The directives the fake supervisor honours."""

    exec_start: list[str]
    environment: dict = field(default_factory=dict)
    working_directory: str | None = None
    type: str = "notify"
    notify_access: str = "main"
    kill_mode: str = "mixed"
    timeout_start: float = 90.0
    timeout_stop: float = 90.0
    timeout_abort: float | None = None
    watchdog: float = 0.0
    restart: str = "on-failure"
    restart_sec: float = 0.2
    restart_prevent: frozenset = frozenset()
    start_limit_interval: float = 300.0
    start_limit_burst: int = 5
    stdout: Path | None = None

    @classmethod
    def from_dropins(cls, unit_dir: Path, *, base: dict | None = None, extra_args: list[str] | None = None,
                     environment: dict | None = None, restart_sec: float | None = None,
                     stdout: Path | None = None, **overrides) -> UnitConfig:
        """Build the unit from the rendered drop-ins (managed.conf), optionally with test overrides."""
        directives = parse_dropins(unit_dropins(unit_dir), base)
        argv = shlex.split(directives["ExecStart"][-1]) + list(extra_args or [])
        env = dict(item.split("=", 1) for item in directives["Environment"] if "=" in item)
        env.update(environment or {})
        values = dict(
            exec_start=argv, environment=env, working_directory=directives.get("WorkingDirectory"),
            type=directives.get("Type", "simple"), notify_access=directives.get("NotifyAccess", "none"),
            kill_mode=directives.get("KillMode", "control-group"),
            timeout_start=_seconds(directives.get("TimeoutStartSec", "90")),
            timeout_stop=_seconds(directives.get("TimeoutStopSec", "90")),
            timeout_abort=_seconds(directives["TimeoutAbortSec"]) if "TimeoutAbortSec" in directives else None,
            watchdog=_seconds(directives.get("WatchdogSec", "0")),
            restart=directives.get("Restart", "no"),
            restart_sec=_seconds(directives.get("RestartSec", "0.1")) if restart_sec is None else restart_sec,
            restart_prevent=frozenset(int(item) for item in directives.get("RestartPreventExitStatus", "").split()
                                      if item.isdigit()),
            start_limit_interval=_seconds(directives.get("StartLimitIntervalSec", "10")),
            start_limit_burst=int(directives.get("StartLimitBurst", "5")), stdout=stdout)
        values.update(overrides)
        return cls(**values)


class FakeSupervisor:
    """Supervise one unit with systemd-like notify, stop, watchdog and restart semantics."""

    def __init__(self, unit: UnitConfig, *, state_dir: Path, shim_dir: Path | None = None) -> None:
        self.unit = unit
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.shim_dir = shim_dir
        self.log: list[tuple[float, str, str]] = []
        self.state = "inactive"
        self.process: subprocess.Popen | None = None
        self.exits: list[int] = []
        self.statuses: list[str] = []
        self.notifications: list[str] = []
        self._t0 = time.monotonic()
        self._lock = threading.RLock()
        self._starts: list[float] = []
        self._socket_dir = Path(tempfile.mkdtemp(prefix="fsd-"))
        if len(str(self._socket_dir)) > 80:
            self._socket_dir = Path(tempfile.mkdtemp(prefix="fsd-", dir="/tmp"))
        self.notify_path = self._socket_dir / "notify"
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._socket.bind(str(self.notify_path))
        self._credentials = hasattr(socket, "SO_PASSCRED")
        if self._credentials:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self._socket.settimeout(0.02)
        self._start_deadline: float | None = None
        self._watchdog_deadline: float | None = None
        self._stop_deadline: float | None = None
        self._cap_deadline: float | None = None
        self._restart_at: float | None = None
        self._stop_requested = False
        self._result = "success"
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="fake-systemd", daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------

    def record(self, event: str, detail: object = "") -> None:
        """Append one transition to the log."""
        self.log.append((round(time.monotonic() - self._t0, 3), event, str(detail)))

    def events(self, name: str | None = None) -> list[tuple[float, str, str]]:
        """Logged transitions, optionally of one kind."""
        with self._lock:
            return [item for item in self.log if name is None or item[1] == name]

    def start(self) -> None:
        """systemctl start: spawn the main process unless the start limit is hit."""
        with self._lock:
            self._stop_requested = False
            self._spawn()

    def stop(self, *, wait: float = 120.0) -> int | None:
        """systemctl stop: KillSignal to the main PID; SIGKILL the group at TimeoutStopSec."""
        with self._lock:
            self._restart_at = None
            self._stop_requested = True
            if self.process is not None and self.process.poll() is None and self.state != "deactivating":
                self._terminate("stop", self.unit.timeout_stop)
        return self.wait_inactive(wait)

    def manager_stop(self, budget: float, *, wait: float = 120.0) -> int | None:
        """The user manager stops (reboot): system 'stopping', unit stop, group SIGKILL at the budget."""
        (self.state_dir / "system_state").write_text("stopping\n")
        with self._lock:
            self.record("manager_stopping", f"budget={budget:g}")
            self._cap_deadline = time.monotonic() + budget
        return self.stop(wait=wait)

    def reboot(self, boot_id_file: Path) -> None:
        """A new boot: fresh boot id, system running again."""
        boot_id_file.write_text(os.urandom(16).hex() + "\n")
        (self.state_dir / "system_state").write_text("running\n")
        with self._lock:
            self._cap_deadline = None
            self.record("reboot")

    def wait_for(self, predicate, timeout: float = 60.0) -> bool:
        """Poll ``predicate(self)`` until it is true or the timeout ends."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if predicate(self):
                    return True
            time.sleep(0.02)
        return False

    def wait_inactive(self, timeout: float = 120.0) -> int | None:
        """Wait until no main process runs and no restart is pending; the last exit code."""
        self.wait_for(lambda unit: unit.state in {"inactive", "failed"} and unit._restart_at is None, timeout)
        return self.exits[-1] if self.exits else None

    def close(self) -> None:
        """Stop the loop, kill anything left and remove the socket."""
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                self._killpg("close")
        self._running = False
        self._thread.join(timeout=5)
        self._socket.close()
        try:
            self.notify_path.unlink()
            self._socket_dir.rmdir()
        except OSError:
            pass

    # -- internals ------------------------------------------------------

    def _environment(self, pid_placeholder: bool = True) -> dict:
        env = dict(self.unit.environment)
        env["NOTIFY_SOCKET"] = str(self.notify_path)
        if self.shim_dir is not None:
            env["PATH"] = str(self.shim_dir) + os.pathsep + env.get("PATH", os.environ.get("PATH", ""))
        return env

    def _spawn(self) -> None:
        now = time.monotonic()
        self._starts = [item for item in self._starts if item > now - self.unit.start_limit_interval]
        if len(self._starts) >= self.unit.start_limit_burst:
            self.state = "failed"
            self.record("start_limit_hit")
            return
        self._starts.append(now)
        env = self._environment()
        stdout = open(self.unit.stdout, "ab") if self.unit.stdout is not None else None  # noqa: SIM115
        if self.unit.watchdog:
            # WATCHDOG_PID must name the main PID: exec through a tiny shell that sets it.
            env["WATCHDOG_USEC"] = str(int(self.unit.watchdog * 1_000_000))
            command = ["/bin/sh", "-c", 'WATCHDOG_PID=$$; export WATCHDOG_PID; exec "$@"', "sh",
                       *self.unit.exec_start]
        else:
            command = list(self.unit.exec_start)
        try:
            self.process = subprocess.Popen(command, env=env, cwd=self.unit.working_directory,
                                            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stdout,
                                            start_new_session=True)
        finally:
            if stdout is not None:
                stdout.close()
        self._result = "success"
        self.state = "activating" if self.unit.type == "notify" else "active"
        self._start_deadline = now + self.unit.timeout_start if self.unit.type == "notify" else None
        self._watchdog_deadline = None
        self._stop_deadline = None
        self.record("start", self.process.pid)

    def _terminate(self, reason: str, timeout: float) -> None:
        self.state = "deactivating"
        self._watchdog_deadline = None
        self._start_deadline = None
        signal_number = signal.SIGTERM
        try:
            if self.unit.kill_mode == "mixed":
                os.kill(self.process.pid, signal_number)
            else:
                os.killpg(self.process.pid, signal_number)
        except OSError:
            pass
        self.record("sigterm_main", reason)
        self._stop_deadline = time.monotonic() + timeout

    def _killpg(self, reason: str) -> None:
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.record("sigkill_group", reason)
        except OSError:
            pass

    def _receive(self) -> None:
        try:
            if self._credentials:
                data, ancillary, _flags, _address = self._socket.recvmsg(4096, socket.CMSG_SPACE(12))
            else:
                data, ancillary = self._socket.recv(4096), []
        except (TimeoutError, BlockingIOError):
            return
        except OSError:
            return
        sender = None
        for level, kind, payload in ancillary:
            if level == socket.SOL_SOCKET and kind == getattr(socket, "SCM_CREDENTIALS", -1):
                sender = struct.unpack("3i", payload[:12])[0]
        with self._lock:
            main = None if self.process is None else self.process.pid
            if self.unit.notify_access == "main" and sender is not None and sender != main:
                self.record("notify_dropped_foreign", sender)
                return
            if self.unit.notify_access == "none":
                self.record("notify_dropped_access_none")
                return
            for line in data.decode("ascii", "replace").splitlines():
                self._notify(line, sender)

    def _notify(self, line: str, sender: int | None) -> None:
        self.notifications.append(line)
        self.record("notify", line if not line.startswith("STATUS=") else "STATUS")
        if line.startswith("STATUS="):
            self.statuses.append(line[len("STATUS="):])
        if line == "READY=1" and self.state == "activating":
            self.state = "active"
            self._start_deadline = None
            if self.unit.watchdog:
                self._watchdog_deadline = time.monotonic() + self.unit.watchdog
            self.record("ready")
        elif line == "WATCHDOG=1" and self.state == "active" and self.unit.watchdog:
            self._watchdog_deadline = time.monotonic() + self.unit.watchdog
        elif line.startswith("EXTEND_TIMEOUT_USEC=") and self.state == "activating":
            extension = int(line.split("=", 1)[1]) / 1_000_000
            self._start_deadline = max(self._start_deadline or 0.0, time.monotonic() + extension)
        elif line == "STOPPING=1":
            self._watchdog_deadline = None

    def _loop(self) -> None:
        while self._running:
            self._receive()
            with self._lock:
                self._tick()

    def _tick(self) -> None:
        now = time.monotonic()
        process = self.process
        if process is not None and self.state in {"activating", "active", "deactivating"}:
            code = process.poll()
            if code is not None:
                self._exited(code)
                return
            if self._cap_deadline is not None and now >= self._cap_deadline:
                self._killpg("manager_cap")
                self._result = "failure"
                self._cap_deadline = None
            elif self._start_deadline is not None and now >= self._start_deadline:
                self.record("start_timeout")
                self._result = "failure"
                self._terminate("start_timeout", self.unit.timeout_stop)
            elif self._watchdog_deadline is not None and now >= self._watchdog_deadline:
                self.record("watchdog_timeout")
                self._result = "failure"
                self._terminate("watchdog", self.unit.timeout_abort or self.unit.timeout_stop)
            elif self._stop_deadline is not None and now >= self._stop_deadline:
                self._killpg("timeout_stop")
                self._result = "failure"
                self._stop_deadline = None
        if self._restart_at is not None and now >= self._restart_at and not self._stop_requested:
            self._restart_at = None
            self.record("restart")
            self._spawn()

    def _exited(self, code: int) -> None:
        self.exits.append(code)
        self.record("exit", code)
        # KillMode=mixed: whatever is left in the group gets SIGKILL now.
        self._killpg("main_exited")
        failure = self._result == "failure" or code != 0
        self.state = "failed" if failure else "inactive"
        self._start_deadline = self._watchdog_deadline = self._stop_deadline = None
        prevented = code in self.unit.restart_prevent
        if (failure and self.unit.restart == "on-failure" and not self._stop_requested and not prevented
                and self._cap_deadline is None):
            self._restart_at = time.monotonic() + self.unit.restart_sec
            self.record("restart_scheduled", code)
        elif prevented:
            self.record("restart_prevented", code)


# --------------------------------------------------------------------------
# systemctl shim


def write_systemctl_shim(directory: Path, state_dir: Path, unit_dir: Path, *, base: dict | None = None,
                         manager_timeout_stop: str = "5s", active_state: str = "inactive") -> Path:
    """Write an executable ``systemctl`` answering the launcher's queries from files.

    ``show user@<uid>.service -p TimeoutStopUSec --value`` reads
    state_dir/manager_timeout_stop; ``is-system-running`` reads
    state_dir/system_state; ``--user show kza-voice.service -p ...`` overlays
    the drop-ins in ``unit_dir`` on ``base``; ``--user daemon-reload`` is
    logged. Every call is appended to state_dir/calls.log with a wall-clock
    nanosecond stamp.
    """
    directory.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "manager_timeout_stop").write_text(manager_timeout_stop + "\n")
    if not (state_dir / "system_state").exists():
        (state_dir / "system_state").write_text("running\n")
    (state_dir / "active_state").write_text(active_state + "\n")
    (state_dir / "config.json").write_text(json.dumps({"unit_dir": str(unit_dir), "base": base or {}}))
    shim = directory / "systemctl"
    shim.write_text("#!" + sys.executable + " -I\n"
                    "import sys\n"
                    f"sys.path.insert(0, {str(REPO)!r})\n"
                    "from tests.mocks.fake_systemd import shim_main\n"
                    f"raise SystemExit(shim_main({str(state_dir)!r}, sys.argv[1:]))\n")
    shim.chmod(0o755)
    return shim


def shim_main(state_dir: str, argv: list[str]) -> int:
    """Entry point of the systemctl shim."""
    state = Path(state_dir)
    with (state / "calls.log").open("a") as log:
        log.write(f"{time.time_ns()} {' '.join(argv)}\n")
    config = json.loads((state / "config.json").read_text())
    if argv[:1] == ["is-system-running"]:
        value = (state / "system_state").read_text().strip()
        sys.stdout.write(value + "\n")
        return 0 if value == "running" else 1
    if argv[:1] == ["show"] and argv[1].startswith("user@"):
        sys.stdout.write((state / "manager_timeout_stop").read_text().strip() + "\n")
        return 0
    if argv[:2] == ["--user", "daemon-reload"]:
        return 0
    if argv[:2] == ["--user", "show"]:
        unit = argv[2]
        names = [argv[index + 1] for index, item in enumerate(argv) if item == "-p"]
        if unit == "kza-voice.service":
            values = show_properties(parse_dropins(unit_dropins(Path(config["unit_dir"])), config["base"]))
            values["ActiveState"] = (state / "active_state").read_text().strip()
        else:
            values = {"TimeoutStopUSec": "1min 30s", "ActiveState": "active"}
        for name in names:
            sys.stdout.write(f"{name}={values.get(name, '')}\n")
        return 0
    sys.stderr.write("fake systemctl: unsupported " + " ".join(argv) + "\n")
    return 2
