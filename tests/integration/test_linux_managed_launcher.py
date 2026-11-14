"""U3b-0e4: the compiled managed launcher under this account's real user systemd (T-C8, T-C10).

The 0d rehearsals ran under tests/mocks/fake_systemd (a process group, no
NotifyAccess, EXTEND only while activating). Here the same compiled
controller (archive X = HEAD plus this working tree, built with
scripts/build_runtime.py) is installed with its own compiled
install-controller and install-launcher, against the real user manager:
install-launcher writes managed.conf beside a production-shaped release.conf
in the stand-in's drop-in directory and verifies the effective unit through
real ``systemctl --user show``. A rehearsal-only drop-in (zz-rehearsal.conf)
then appends the hidden ``--child-command`` to the rendered ExecStart (and
``Restart=no`` where a test drives the restart itself). Units and the
systemctl guard: tests/mocks/real_systemd.py.

Covered here (design: managed-launcher v2/v3, 0d review P12/P13):

- T-C8, real cgroup: a setsid grandchild escapes the process group but not
  the unit cgroup; the controller parses ``0::/user.slice/.../<unit>.service``
  and kills it before the seal (exit record ``stragglers``).
- READY through ``NotifyAccess=main`` only (the child has no NOTIFY_SOCKET),
  a start timeout without READY, a stop requested by the child's own
  STOPPING=1 (systemd arms TimeoutStopSec) with and without
  EXTEND_TIMEOUT_USEC during that stop.
- EXTEND_TIMEOUT_USEC during start: the controller's extension before
  activation outlasts a shorter TimeoutStartSec (R-b on a production-sized
  adopted state is measured in test_linux_adoption_window.py, T-A8).
- T-C10: one headless real src.main (tests/mocks/stop_contract_child.py on
  the slot's compiled modules, synthetic leaves, the real dashboard on a
  private port) stopped with ``systemctl --user stop`` at each start-up phase,
  after READY, and after READY with a dashboard WebSocket open; every stop is
  timed.

The restore-after-crash refusal (design T-C8) runs on the adopted state in
tests/integration/test_linux_adoption_window.py (T-A5).

Skipped only off Linux, without a running user manager, or as root or the
production account kza (real_systemd.unavailable). KZA_0E4_EVIDENCE=<dir>
records the measurements. Each module first sweeps kza-rehearsal-* units a
killed earlier run left behind (real_systemd.sweep_stale); run it under
nohup so a dropped ssh does not kill it mid-test.

Run test_linux_managed_launcher.py and test_linux_adoption_window.py one
pytest at a time, never side by side: they share this account's user manager, and any
``daemon-reload`` re-arms every activating unit's start timer from
TimeoutStartSec, dropping its EXTEND_TIMEOUT_USEC extension (the no-READY
rehearsal's ~90 s start relies on that extension).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time

import pytest
import yaml

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks import real_systemd

pytestmark = pytest.mark.skipif(real_systemd.unavailable() is not None,
                                reason=real_systemd.unavailable() or "available")

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template"
STOP_CONTRACT_CHILD = REPO / "tests" / "mocks" / "stop_contract_child.py"
BASE_UNIT = """[Unit]
Description=KZA U3b-0e4 rehearsal stand-in for kza-voice (test only)

[Service]
Type=simple
Restart=on-failure
RestartSec=10s
TimeoutStartSec=120
Environment=CUDA_VISIBLE_DEVICES=

[Install]
WantedBy={boot}
"""
RELEASE_CONF = ("[Service]\nExecStart=\nExecStart={python} -B -P -m src.main\n"
                "Environment=PYTHONPATH={root}/current\nEnvironment=CONFIG_PATH=config/settings.yaml\n"
                "WorkingDirectory={root}/slots/s/state\n")
STATUS = ("ActiveState", "SubState", "Result", "ExecMainStatus", "ExecMainCode", "StatusText", "NRestarts",
          "MainPID", "ControlGroup")
CLEANUPS: list[dict] = []


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """X: HEAD plus this working tree, compiled by scripts/build_runtime.py (clone overlay)."""
    base = tmp_path_factory.mktemp("linux-launcher-artifact")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    revision = rehearsal.overlay_commit(REPO, clone)
    path = base / "x.tar.gz"
    started = time.monotonic()
    manifest = build_runtime(clone, path, revision)
    assert manifest["git_commit"] == revision
    real_systemd.record("launcher-build", {"build_seconds": round(time.monotonic() - started, 1),
                                           "commit": revision})
    return path


@pytest.fixture(scope="module", autouse=True)
def cleanup_evidence():
    # A killed earlier run (SIGKILL, OOM, dropped ssh) skipped its finalizers: sweep its units first.
    real_systemd.record("launcher-stale-sweep", real_systemd.sweep_stale())
    yield
    real_systemd.record("launcher-unit-cleanup", CLEANUPS)


@pytest.fixture
def systemd(tmp_path):
    manager = real_systemd.UserSystemd(tmp_path / "systemd")
    yield manager
    result = manager.cleanup()
    CLEANUPS.append({key: value for key, value in result.items() if key not in ("before", "after")})
    assert result["rehearsal_units_after"] == [] and result["leftover_paths"] == []
    assert result["guard_breaches"] == [] and result["daemon_reload_exit"] == 0


def _cli(module: Path, root: Path, *arguments: str, env: dict | None = None, timeout: float = 600):
    return subprocess.run([sys.executable, "-I", "-B", str(module), "--root", str(root), "--offline", *arguments],
                          capture_output=True, text=True, timeout=timeout, env=env)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RealDeployment:
    """A temporary managed root whose stand-in unit runs under the real user manager."""

    def __init__(self, archive: Path, tmp_path: Path, systemd: real_systemd.UserSystemd, *,
                 source: Path | None = None, settings: str = rehearsal.SETTINGS, stop_grace: float = 20.0,
                 timeout_stop: float = 60.0, timeout_start: float = 120.0) -> None:
        self.tmp = tmp_path
        self.systemd = systemd
        if source is None:
            source = rehearsal.offline_source(tmp_path, settings)
        staging = tmp_path / "root"
        stage.prepare_slot(archive, staging, "s", source)
        self.root = staging.resolve()
        with stage.AdmissionController(self.root) as admission:  # the bootstrap head (U3b-0e's job in production)
            assert admission.activate("s", managed=True) is None
        self.state = self.root / "slots" / "s" / "state"
        extracted = tmp_path / "extracted"
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
        installed = _cli(extracted / "scripts" / "stage_runtime.pyc", self.root, "install-controller", str(archive))
        assert installed.returncode == 0, installed.stderr
        self.controller = self.root / "controller" / "current" / "scripts" / "stage_runtime.pyc"
        self.unit = systemd.name("voice")
        systemd.write_unit(self.unit, BASE_UNIT.format(boot=systemd.boot))
        self.unit_dir = systemd.dropin_dir(self.unit)
        (self.unit_dir / "release.conf").write_text(RELEASE_CONF.format(python=sys.executable, root=self.root))
        systemd.reload()
        self.console = tmp_path / "logs" / "console.log"
        self.console.parent.mkdir()
        (self.root / launcher.LOGROTATE_NAME).write_text(launcher.render_console_logrotate(str(self.console)))
        result = _cli(self.controller, self.root, "install-launcher", "--unit-dir", str(self.unit_dir),
                      "--template", str(TEMPLATE), "--python", sys.executable, "--stop-grace", f"{stop_grace:g}",
                      "--timeout-start", f"{timeout_start:g}", "--timeout-stop", f"{timeout_stop:g}",
                      "--console-log", str(self.console), env=systemd.environment())
        assert result.returncode == 0, (result.stdout, result.stderr)
        self.install = json.loads(result.stdout)
        assert self.install["verification_mismatches"] == []
        self.managed = (self.unit_dir / "managed.conf").read_text()
        self.phase_file = tmp_path / "phases.txt"

    def child(self, command: list[str], *, restart: bool = True, extra: str = "") -> None:
        """zz-rehearsal.conf: the rendered ExecStart plus --child-command (and Restart=no unless ``restart``)."""
        rendered = launcher._rendered_exec_start(self.managed)
        line = " ".join(shlex.quote(token) for token in [*rendered, "--child-command", json.dumps(command)])
        text = "[Service]\nExecStart=\nExecStart=" + line + "\n" + ("" if restart else "Restart=no\n") + extra
        self.systemd.write_dropin(self.unit, "zz-rehearsal.conf", text)
        self.systemd.reload()

    def launcher_child(self, scenario: str, *, restart: bool = True, **options) -> None:
        execute = options.pop("execute", None)
        script = None
        if execute is not None:
            script = self.tmp / f"exec-{len(list(self.tmp.glob('exec-*')))}.py"
            script.write_text(execute)
        self.child(rehearsal.child_command(self.phase_file, scenario, execute=script, **options), restart=restart)

    def start(self) -> float:
        started = time.monotonic()
        completed = self.systemd.systemctl("start", "--no-block", "kza-voice.service")
        assert completed.returncode == 0, completed.stderr
        return started

    def stop(self, timeout: float = 300) -> tuple[subprocess.CompletedProcess, float]:
        started = time.monotonic()
        completed = self.systemd.systemctl("stop", "kza-voice.service", timeout=timeout)
        return completed, time.monotonic() - started

    def show(self) -> dict:
        return self.systemd.raw_show(self.unit, *STATUS)

    def wait_active(self, timeout: float) -> dict:
        return self.systemd.wait_for(self.unit, lambda shown: shown["ActiveState"] in ("active", "failed")
                                     or (shown["ActiveState"] == "inactive" and shown["Result"] != "success"),
                                     timeout, *STATUS)

    def wait_inactive(self, timeout: float) -> dict:
        return self.systemd.wait_for(self.unit, lambda shown: shown["ActiveState"] in ("inactive", "failed"),
                                     timeout, *STATUS)

    def phases(self):
        return rehearsal.phases(self.phase_file)

    def phase(self, name: str):
        return rehearsal.phase(self.phase_file, name)

    def wait_phase(self, name: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.phase(name) is not None:
                return True
            time.sleep(0.05)
        return False

    def head(self) -> dict:
        return stage.read_admission_head(self.root)

    def exits(self) -> list[dict]:
        directory = self.root / "admission" / "runs"
        return [json.loads(path.read_text()) for path in sorted(directory.glob("*.exit.json"),
                                                                 key=lambda item: item.stat().st_mtime_ns)]

    def console_text(self) -> str:
        return self.console.read_text() if self.console.exists() else ""

    def events(self) -> list[tuple[float, str]]:
        """(epoch seconds, event) of every [Launcher] line in the console log."""
        found = []
        for line in self.console_text().splitlines():
            match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) INFO \S+ \[Launcher\] (.*)$", line)
            if match:
                seconds = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")) + int(match.group(2)) / 1000
                found.append((seconds, match.group(3)))
        return found


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0] if Path(f"/proc/{pid}").exists() else "X"
    return state not in ("Z", "X")


# --- the installed unit, read back from the real manager --------------------------------------------

def test_install_launcher_verifies_the_effective_unit_through_real_systemctl(archive, tmp_path, systemd):
    deployment = RealDeployment(archive, tmp_path, systemd)
    shown = systemd.raw_show(deployment.unit, "Type", "NotifyAccess", "KillMode", "RestartPreventExitStatus",
                             "WorkingDirectory", "TimeoutStopUSec", "TimeoutStartUSec", "DropInPaths",
                             "UnitFileState", "FragmentPath")
    assert shown["Type"] == "notify" and shown["NotifyAccess"] == "main" and shown["KillMode"] == "mixed"
    assert set(shown["RestartPreventExitStatus"].split()) == {"69", "78"}
    assert shown["WorkingDirectory"] == str(deployment.root)
    assert launcher.parse_timespan(shown["TimeoutStopUSec"]) == 60.0
    assert launcher.parse_timespan(shown["TimeoutStartUSec"]) == 120.0
    paths = shown["DropInPaths"].split()
    assert [Path(path).name for path in paths] == ["managed.conf"]  # release.conf moved aside
    assert (deployment.unit_dir / "release.conf.unmanaged").is_file()
    assert deployment.install["drop_in_paths"] == paths
    record = json.loads(next((deployment.root / "admission" / "launcher").glob("install-*.json")).read_text())
    assert record["properties"]["Type"] == "simple"  # the pre-managed unit, read through the guard
    assert record["legacy_exec_start"] == [sys.executable, "-B", "-P", "-m", "src.main"]
    budget = launcher.ServiceManager(executable=str(systemd.shim)).stop_budget(os.getuid())
    real_systemd.record("launcher-installed-unit", {
        "properties": {key: value for key, value in shown.items() if key not in ("DropInPaths", "FragmentPath")},
        "own_user_manager_stop_budget_seconds": budget, "stop_timeout_dropin": deployment.install["stop_timeout_dropin"]})


# --- T-C8: the real cgroup -----------------------------------------------------------------------------

STRAGGLER = """import os, subprocess
process = subprocess.Popen(['sleep', '600'], start_new_session=True, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
open({pidfile!r}, 'w').write(str(process.pid))
"""


def test_tc8_a_setsid_straggler_in_the_unit_cgroup_is_killed_before_the_seal(archive, tmp_path, systemd):
    deployment = RealDeployment(archive, tmp_path, systemd)
    pidfile = tmp_path / "straggler.pid"
    deployment.launcher_child("clean", execute=STRAGGLER.format(pidfile=str(pidfile)))
    started = deployment.start()
    shown = deployment.wait_active(90)
    ready_seconds = time.monotonic() - started
    assert shown["ActiveState"] == "active", (shown, deployment.console_text()[-3000:])
    assert deployment.phase("env") == "clean"  # no NOTIFY_SOCKET or WATCHDOG_*: READY came through main
    straggler = int(pidfile.read_text())
    cgroup = Path(f"/proc/{straggler}/cgroup").read_text()
    assert ("0::" + shown["ControlGroup"]) in cgroup.splitlines()
    assert shown["ControlGroup"].endswith("/" + deployment.unit)
    assert os.getpgid(straggler) == straggler != os.getpgid(int(deployment.phase("pid")))  # escaped the group
    completed, seconds = deployment.stop()
    assert completed.returncode == 0, completed.stderr
    after = deployment.show()
    assert after["Result"] == "success" and after["ExecMainStatus"] == "0"
    exits = deployment.exits()
    assert len(exits) == 1 and exits[0]["stragglers"] == 1 and exits[0]["sealed"] is True
    assert deployment.head()["status"] == "closed"
    assert not _pid_alive(straggler)
    real_systemd.record("tc8-cgroup", {"control_group_suffix": deployment.unit, "stragglers": exits[0]["stragglers"],
                                       "start_to_ready_seconds": round(ready_seconds, 2),
                                       "systemctl_stop_seconds": round(seconds, 2)})


def test_tc8_no_ready_is_a_real_start_timeout_and_the_child_still_closes(archive, tmp_path, systemd):
    deployment = RealDeployment(archive, tmp_path, systemd, timeout_start=30)  # install-launcher's minimum
    deployment.launcher_child("no_ready", restart=False)
    started = deployment.start()
    activating = systemd.wait_for(deployment.unit, lambda shown: deployment.phase("pid") is not None, 60, *STATUS)
    assert activating["ActiveState"] == "activating"  # the child is running; no READY, so not active
    shown = deployment.wait_inactive(180)
    elapsed = time.monotonic() - started
    assert shown["Result"] == "timeout", shown
    # EXTEND_TIMEOUT_USEC=(probe timeout + 30 s) before activation only lengthens the start: systemd
    # times out at max(TimeoutStartSec, that message + its window), never at TimeoutStartSec=30 here.
    window = 60 + 30
    assert window - 5 <= elapsed < window + 30, elapsed
    assert deployment.phase("stop_received") is not None and deployment.phase("receipt") is not None
    assert deployment.head()["status"] == "closed"  # SIGTERM forwarded, receipt, seal
    real_systemd.record("tc8-start-timeout", {"timeout_start": 30, "extend_window_seconds": window,
                                              "seconds_to_inactive": round(elapsed, 2), "result": shown["Result"],
                                              "exec_main_status": shown["ExecMainStatus"]})


def test_a_daemon_reload_drops_the_start_extension_under_real_systemd(archive, tmp_path, systemd):
    """Found by T-A8: systemd 255 re-arms the start timer from TimeoutStartSec on daemon-reload, so an
    EXTEND_TIMEOUT_USEC extension is lost and a start already past TimeoutStartSec times out at once.
    --timeout-start must cover the whole start; EXTEND is not a guarantee against a reload."""
    deployment = RealDeployment(archive, tmp_path, systemd, timeout_start=30)
    deployment.launcher_child("no_ready", restart=False)
    started = deployment.start()
    systemd.wait_for(deployment.unit, lambda shown: deployment.phase("pid") is not None, 60, *STATUS)
    while time.monotonic() - started < 40:  # past TimeoutStartSec=30, inside the controller's 90 s EXTEND
        time.sleep(0.5)
    assert systemd.raw_show(deployment.unit, "ActiveState")["ActiveState"] == "activating"
    reloaded = time.monotonic()
    systemd.reload()
    shown = deployment.wait_inactive(60)
    after_reload = time.monotonic() - reloaded
    assert shown["Result"] == "timeout", shown
    assert after_reload < 15, after_reload  # at once (stop budget aside), not at the EXTEND deadline (~50 s on)
    assert deployment.phase("stop_received") is not None and deployment.head()["status"] == "closed"
    real_systemd.record("tc8-reload-drops-extend", {"timeout_start": 30, "reload_after_seconds": 40,
                                                    "seconds_from_reload_to_inactive": round(after_reload, 2),
                                                    "result": shown["Result"]})


@pytest.mark.parametrize("close_seconds, extend_every, expected", [
    (5.0, 0.0, "sealed"),        # inside TimeoutStopSec
    (50.0, 0.0, "killed"),       # past TimeoutStopSec: systemd kills the unit
    (50.0, 2.0, "sealed"),       # past it, but EXTEND_TIMEOUT_USEC during the stop keeps it
])
def test_tc8_stop_by_notify_arms_the_stop_timeout_and_extend_holds_it(archive, tmp_path, systemd, close_seconds,
                                                                     extend_every, expected):
    deployment = RealDeployment(archive, tmp_path, systemd, stop_grace=5, timeout_stop=35)
    deployment.launcher_child("notify_stop", restart=False, seconds=close_seconds, extend_every=extend_every)
    deployment.start()
    assert deployment.wait_active(90)["ActiveState"] == "active"
    assert deployment.wait_phase("stopping_sent", 30)
    sent = time.monotonic()
    deactivating = systemd.wait_for(deployment.unit, lambda shown: shown["ActiveState"] != "active", 10, *STATUS)
    assert deactivating["ActiveState"] == "deactivating", deactivating  # STOPPING=1 relayed by the controller
    shown = deployment.wait_inactive(close_seconds + 60)
    elapsed = time.monotonic() - sent
    exits = deployment.exits()
    if expected == "sealed":
        assert deployment.phase("self_closed") is not None
        assert shown["Result"] == "exit-code" and shown["ExecMainStatus"] == "75", shown  # child ended, no stop asked
        assert len(exits) == 1 and exits[0]["sealed"] is True and deployment.head()["status"] == "closed"
    else:
        assert shown["Result"] == "timeout", shown
        assert deployment.phase("self_closed") is None and exits == []  # the controller was killed
        assert deployment.head()["status"] == "running"  # the next start recovers it (R-a or R-b)
        assert 34 <= elapsed < 60
    real_systemd.record(f"tc8-stop-by-notify-{close_seconds:g}s-extend-{extend_every:g}", {
        "timeout_stop": 35, "close_seconds": close_seconds, "extend_every": extend_every,
        "seconds_from_stopping_to_inactive": round(elapsed, 2), "result": shown["Result"],
        "exec_main_status": shown["ExecMainStatus"], "head": deployment.head()["status"]})


# --- T-C10: a headless real src.main, stopped at each phase -------------------------------------------

def _headless_settings(state: Path, port: int) -> str:
    from tests.mocks.managed_slot_graph import canonical_template
    from tests.mocks.restore_application_graph import configuration as restore_configuration

    def configuration(root, home, model):
        config = restore_configuration(Path(root), home, model)
        config["dashboard"].update(enabled=True, host="127.0.0.1", port=port)
        return config
    return yaml.safe_dump(canonical_template(state / "data" / "graph", configuration), sort_keys=False)


class HeadlessDeployment(RealDeployment):
    def __init__(self, archive: Path, tmp_path: Path, systemd) -> None:
        self.port = _free_port()
        state = (tmp_path / "root").resolve() / "slots" / "s" / "state"
        source = tmp_path / "offline"
        (source / "config").mkdir(parents=True)
        (source / "data" / "graph").mkdir(parents=True, mode=0o700)
        (source / "config" / "settings.yaml").write_text(_headless_settings(state, self.port))
        super().__init__(archive, tmp_path, systemd, source=source, stop_grace=30, timeout_stop=90)
        assert self.state == state

    def headless(self, scenario: str) -> None:
        self.child([sys.executable, "-I", "-B", str(STOP_CONTRACT_CHILD), "--repo", str(REPO), "--scenario",
                    scenario, "--phase-file", str(self.phase_file), "--graph-root",
                    str(self.state / "data" / "graph"), "--dashboard-port", str(self.port)], restart=False)


async def _open_websocket(port: int, opened: asyncio.Event, release: asyncio.Event):
    import websockets
    from tests.mocks.restore_application_graph import TOKEN

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws",
                                  additional_headers={"Authorization": "Bearer " + TOKEN}) as connection:
        await connection.send(json.dumps({"type": "ping"}))
        assert json.loads(await asyncio.wait_for(connection.recv(), 10)) == {"type": "pong"}
        opened.set()
        try:
            await connection.wait_closed()
        finally:
            release.set()


@pytest.mark.parametrize("scenario, phase", [
    ("entry", "entry"), ("construction", "construction"), ("warmup", "warmup"), ("services", "services"),
    ("pipeline", "pipeline"), ("ready", "steady"), ("websocket", "steady"),
])
def test_tc10_headless_src_main_stops_cleanly_at_each_phase(archive, tmp_path, systemd, scenario, phase):
    deployment = HeadlessDeployment(archive, tmp_path, systemd)
    deployment.headless("ready" if scenario == "websocket" else scenario)
    started = deployment.start()
    assert deployment.wait_phase(phase, 180), deployment.console_text()[-4000:]
    reached = time.monotonic() - started
    shown = deployment.show()
    if phase == "steady":
        shown = deployment.wait_active(60)
        assert shown["ActiveState"] == "active", shown
    else:
        assert shown["ActiveState"] == "activating", shown  # a stop before READY
    websocket = None
    if scenario == "websocket":
        loop = asyncio.new_event_loop()
        opened, released = asyncio.Event(), asyncio.Event()
        import threading
        websocket = threading.Thread(target=loop.run_until_complete,
                                     args=(_open_websocket(deployment.port, opened, released),), daemon=True)
        websocket.start()
        deadline = time.monotonic() + 30
        while not opened.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert opened.is_set(), deployment.console_text()[-3000:]
    completed, seconds = deployment.stop()
    assert completed.returncode == 0, completed.stderr
    shown = deployment.show()
    assert shown["Result"] == "success" and shown["ExecMainStatus"] == "0", (shown, deployment.console_text()[-4000:])
    head = deployment.head()
    assert head["status"] == "closed"
    exits = deployment.exits()
    assert len(exits) == 1 and exits[0]["sealed"] is True and exits[0]["stop_requested"] is True
    receipts = list((deployment.root / "admission" / "receipts").glob("*.json"))
    assert len(receipts) == 1
    console = deployment.console_text()
    expected_phase = "ready" if phase == "steady" else phase
    assert f"[Stop] stop requested by SIGTERM during phase={expected_phase}" in console
    if phase == "steady":
        assert "Dashboard API starting" in console
    if websocket is not None:
        websocket.join(10)
    real_systemd.record(f"tc10-{scenario}", {"phase": phase, "start_to_phase_seconds": round(reached, 2),
                                             "systemctl_stop_seconds": round(seconds, 2),
                                             "stragglers": exits[0]["stragglers"]})
