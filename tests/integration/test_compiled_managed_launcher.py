"""U3b-0d compiled rehearsals: the installed controller under a fake systemd (T-C1..T-C4, T-C6, T-C7, T-C9).

Artifacts: X = HEAD plus this working tree, Y = a trivial descendant, P = the
newest pre-0d ancestor (clone overlay, tests/mocks/launcher_rehearsal). The
controller is installed from X's archive with its own compiled CLI
(install-controller from the extracted archive, then install-launcher from
controller/current) into a temporary root; install-launcher renders the
tracked managed.conf template against a systemctl shim. The fake supervisor
(tests/mocks/fake_systemd) then runs the rendered ExecStart with the
rehearsal-only --child-command and --boot-id-file arguments appended. The
child is tests/mocks/launcher_child.py on the slot's compiled modules.

T-C5 (compaction verification run) waits for unit 0c and is not here.
"""

import json
import os
from pathlib import Path
import signal
import tarfile
import time

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks.runtime_tree import writable_runtime
from tests.mocks import fake_systemd
from tests.mocks import launcher_rehearsal as rehearsal

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template"
BASE_UNIT = {"Type": "simple", "Restart": "on-failure", "RestartSec": "10s", "TimeoutStartSec": "120",
             "StandardOutput": "append:/home/kza/kza/logs/console.log"}
RELEASE_CONF = ("[Service]\nExecStart=\nExecStart={python} -B -P -m src.main\n"
                "Environment=PYTHONPATH={root}/current\nEnvironment=CONFIG_PATH=config/settings.yaml\n"
                "WorkingDirectory={root}/slots/s/state\n")
REGISTER_TARGET = "\n".join([
    "import os, types",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
    "from src.memory.erasure_journal import ErasureParticipant",
    "ledger = fact_ledger.FactLedger(str(state / 'data' / 'privacy.db'))",
    "ledger.initialize()",
    "os.makedirs(state / 'data' / 'clips', exist_ok=True)",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': str(state / 'data' / 'clips')})",
    "ledger._db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "ledger.close()",
])


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    started = time.monotonic()
    built = rehearsal.build_artifacts(tmp_path_factory.mktemp("compiled-launcher"))
    built["build_seconds"] = time.monotonic() - started
    return built


def _cli(module: Path, root: Path, *arguments: str, env: dict | None = None):
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-I", "-B", str(module), "--root", str(root), "--offline", *arguments],
                          capture_output=True, text=True, timeout=180, env=env)


class Deployment:
    """A temporary /home/kza/releases: slot, compiled controller, rendered managed.conf, fake systemd."""

    def __init__(self, artifacts, tmp_path, *, archive="x", settings=rehearsal.SETTINGS, stop_grace=20.0,
                 timeout_stop=60.0, manager_timeout_stop="5s", install_refused=False):
        self.tmp = tmp_path
        self.artifacts = artifacts
        self.root, self.state = rehearsal.managed_root(artifacts["archives"][archive], tmp_path, settings=settings,
                                                       controller=False)
        # I3: extract the archive, install the controller with the archive's own compiled CLI.
        extracted = tmp_path / "extracted"
        with tarfile.open(artifacts["archives"]["x"]) as bundle:
            bundle.extractall(extracted, filter="data")
        installed = _cli(extracted / "scripts" / "stage_runtime.pyc", self.root, "install-controller",
                         str(artifacts["archives"]["x"]))
        assert installed.returncode == 0, installed.stderr
        self.controller = self.root / "controller" / "current" / "scripts" / "stage_runtime.pyc"
        # I4: install-launcher with the installed controller against the systemctl shim.
        self.unit_dir = tmp_path / "kza-voice.service.d"
        self.unit_dir.mkdir()
        import sys
        (self.unit_dir / "release.conf").write_text(RELEASE_CONF.format(python=sys.executable, root=self.root))
        self.sd_state = tmp_path / "sd-state"
        self.shim_dir = tmp_path / "bin"
        fake_systemd.write_systemctl_shim(self.shim_dir, self.sd_state, self.unit_dir, base=BASE_UNIT,
                                          manager_timeout_stop=manager_timeout_stop)
        self.env = {"PATH": str(self.shim_dir) + os.pathsep + os.environ.get("PATH", ""),
                    "HOME": os.environ.get("HOME", str(tmp_path)), "LANG": "C.UTF-8"}
        self.console = tmp_path / "logs" / "console.log"
        self.console.parent.mkdir()
        # Review P5: install-launcher refuses a console log the rotation does not name.
        (self.root / launcher.LOGROTATE_NAME).write_text(launcher.render_console_logrotate(str(self.console)))
        result = _cli(self.controller, self.root, "install-launcher", "--unit-dir", str(self.unit_dir),
                      "--template", str(TEMPLATE), "--python", sys.executable, "--stop-grace", f"{stop_grace:g}",
                      "--timeout-start", "120", "--timeout-stop", f"{timeout_stop:g}",
                      "--console-log", str(self.console), env=self.env)
        self.install_result = result
        if install_refused:
            # Review P2: the compiled install-launcher refuses what serve would refuse with 78.
            # Render managed.conf as an earlier install would have, to rehearse serve's own gate.
            assert result.returncode == 78, result.stderr
            (self.unit_dir / "release.conf").rename(self.unit_dir / "release.conf.unmanaged")
            (self.unit_dir / "managed.conf").write_text(launcher.render_managed_conf(
                TEMPLATE.read_text(), self.root, launcher.LauncherSettings(
                    python=sys.executable, stop_grace=stop_grace, timeout_start=120, timeout_stop=timeout_stop,
                    watchdog_sec=0, console_log=str(self.console))))
        else:
            assert result.returncode == 0, result.stderr
            self.install = json.loads(result.stdout)
            assert self.install["verification_mismatches"] == []
        self.boot_id = tmp_path / "boot_id"
        self.boot_id.write_text("boot-1\n")
        self.phase_file = tmp_path / "phases.txt"
        self.unit = None

    def supervise(self, scenario: str, *, seconds: float = 0.0, execute: str | None = None, **overrides):
        script = None
        if execute is not None:
            script = self.tmp / f"exec-{len(list(self.tmp.glob('exec-*')))}.py"
            script.write_text(execute)
        child = rehearsal.child_command(self.phase_file, scenario, seconds=seconds, execute=script)
        config = fake_systemd.UnitConfig.from_dropins(
            self.unit_dir, base=BASE_UNIT, environment=self.env, restart_sec=0.2, stdout=self.console,
            extra_args=["--child-command", json.dumps(child), "--boot-id-file", str(self.boot_id)], **overrides)
        if self.unit is not None:
            self.unit.close()
        self.unit = fake_systemd.FakeSupervisor(config, state_dir=self.sd_state, shim_dir=self.shim_dir)
        return self.unit

    def phases(self):
        return rehearsal.phases(self.phase_file)

    def head(self):
        return stage.read_admission_head(self.root)

    def console_text(self):
        return self.console.read_text() if self.console.exists() else ""

    def close(self):
        if self.unit is not None:
            self.unit.close()


@pytest.fixture
def deployments():
    created = []
    yield created
    for item in created:
        item.close()


def _deploy(artifacts, tmp_path, deployments, **kwargs):
    deployment = Deployment(artifacts, tmp_path, **kwargs)
    deployments.append(deployment)
    return deployment


def _ready(unit):
    return bool(unit.events("ready"))


# --- T-C1 clean lifecycle --------------------------------------------------------------------------

def test_tc1_clean_lifecycle_restart_and_delegation(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    managed = (deployment.unit_dir / "managed.conf").read_text()
    assert "Type=notify" in managed and not (deployment.unit_dir / "release.conf").exists()
    unit = deployment.supervise("clean", execute=REGISTER_TARGET)
    unit.start()
    assert unit.wait_for(_ready, 60), (unit.log, deployment.console_text()[-3000:])
    assert unit.stop() == 0
    head = deployment.head()
    assert head["status"] == "closed"
    phases = dict(deployment.phases())
    assert phases["stop_received"] == "ready" and phases["mask"] == "empty" and phases["env"] == "clean"
    assert stage._closure_proof(deployment.root, head["binding_id"])["child_pid"] == int(phases["pid"])
    # NotifyAccess=main: every relayed line came from the controller; the child had no NOTIFY_SOCKET.
    assert unit.notifications.count("READY=1") == 1 and "STOPPING=1" in unit.notifications
    assert not unit.events("notify_dropped_foreign")
    # Restart: a new run on the same slot.
    deployment.phase_file.unlink()
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == 0
    second = deployment.head()
    assert second["status"] == "closed" and second["generation"] > head["generation"]
    assert len(rehearsal.receipts(deployment.root)) == 2
    # Delegate to a new slot (Y) and serve it.
    stage.prepare_slot(artifacts["archives"]["y"], deployment.root, "b", deployment.state)
    assert stage.delegate_slot(deployment.root, "b")["targets"]
    assert stage.activate_slot(deployment.root, "b") == "slots/s/runtime"
    deployment.phase_file.unlink()
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == 0
    final = deployment.head()
    assert stage.read_binding(deployment.root, final["binding_id"])["slot"] == "b" and final["status"] == "closed"


# --- T-C2 failures ---------------------------------------------------------------------------------

def test_tc2_child_crash_restarts_once_then_needs_the_operator(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("crash", seconds=0.3)
    unit.start()
    unit.wait_for(lambda item: len(item.exits) >= 2, 120)
    unit.wait_inactive(30)
    assert unit.exits == [68, 69], (unit.log, deployment.console_text()[-3000:])
    assert unit.events("restart_prevented")[0][2] == "69"
    assert deployment.head()["status"] == "unknown"
    recoveries = launcher._recoveries(deployment.root)
    assert [item["class"] for item in recoveries] == ["crash"]


def test_tc2_missing_receipt_is_unknown(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("exit_no_receipt", seconds=0.3)
    unit.start()
    unit.wait_for(lambda item: len(item.exits) >= 2, 120)
    unit.wait_inactive(30)
    assert unit.exits == [68, 69]
    assert "runtime_admission_close_receipt_missing" in deployment.console_text()


def test_tc2_stop_grace_exceeded_is_unknown_and_not_restarted(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, stop_grace=2.0, timeout_stop=40.0)
    unit = deployment.supervise("hang")
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == 68
    assert deployment.head()["status"] == "unknown" and not unit.events("restart")
    assert "stop grace exceeded; SIGKILL" in deployment.console_text()


def test_tc2_watchdog_silence_stops_through_the_controller(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("watchdog_silent", watchdog=1.0, timeout_abort=20.0, start_limit_burst=2)
    unit.start()
    assert unit.wait_for(lambda item: bool(item.events("watchdog_timeout")), 60), unit.log
    unit.wait_for(lambda item: bool(item.exits), 30)
    # WatchdogSignal=SIGTERM reaches the controller only (KillMode=mixed); it orchestrates the stop.
    assert unit.exits[0] == 0 and dict(deployment.phases())["stop_received"] == "ready"
    assert unit.events("restart_scheduled")  # a watchdog failure restarts (on-failure)
    unit.stop()


def test_tc2_watchdog_pings_keep_a_healthy_child_alive(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("clean", watchdog=1.0)
    unit.start()
    assert unit.wait_for(_ready, 60)
    time.sleep(3.0)
    assert not unit.events("watchdog_timeout") and unit.notifications.count("WATCHDOG=1") >= 4
    assert unit.notifications.index("READY=1") < unit.notifications.index("WATCHDOG=1")
    assert unit.stop() == 0


def test_tc2_controller_sigkill_while_running_is_adopted_on_restart(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 60)
    os.kill(unit.process.pid, signal.SIGKILL)
    assert unit.wait_for(lambda item: len(item.events("ready")) >= 2, 90), unit.log
    assert unit.exits[0] == -signal.SIGKILL
    recoveries = launcher._recoveries(deployment.root)
    assert len(recoveries) == 1 and recoveries[0]["head_before"]["status"] == "running"
    assert unit.stop() == 0 and deployment.head()["status"] == "closed"


def test_tc2_controller_sigkill_after_the_receipt_is_sealed_by_r_a(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, stop_grace=3.0, timeout_stop=40.0)
    unit = deployment.supervise("receipt_then_hang")
    unit.start()
    assert unit.wait_for(_ready, 60)
    os.kill(unit.process.pid, signal.SIGTERM)  # the controller forwards; the child writes its receipt
    assert unit.wait_for(lambda item: rehearsal.phase(deployment.phase_file, "receipt") is not None, 30)
    os.kill(unit.process.pid, signal.SIGKILL)  # the controller dies before its seal (child still hangs)
    unit.wait_for(lambda item: len(item.events("ready")) >= 2, 90)
    assert "recovery R-a closed" in deployment.console_text(), deployment.console_text()[-3000:]
    assert launcher._recoveries(deployment.root) == []
    unit.stop()


def test_tc2_supervisor_timeout_stop_kills_a_stuck_controller(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, stop_grace=60.0, timeout_stop=100.0)
    unit = deployment.supervise("hang", timeout_stop=2.0)
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == -signal.SIGKILL
    assert unit.events("sigkill_group")[0][2] == "timeout_stop"
    assert deployment.head()["status"] == "running"  # the controller died before its CAS
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 90), deployment.console_text()[-3000:]
    assert "recovery R-b closed" in deployment.console_text()
    assert unit.stop() == 0


# --- T-C3 restart policy -----------------------------------------------------------------------------

def test_tc3_exit_75_restarts_until_the_start_limit(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    unit = deployment.supervise("self_end", seconds=0.1)
    unit.start()
    assert unit.wait_for(lambda item: bool(item.events("start_limit_hit")), 180), unit.log
    assert set(unit.exits) == {75} and len(unit.exits) == 5  # StartLimitBurst=5 from managed.conf
    assert deployment.head()["status"] == "closed"


def test_tc3_exit_78_never_restarts(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    (deployment.root / "current").unlink()
    (deployment.root / "current").symlink_to("slots/other/runtime")
    unit = deployment.supervise("clean")
    unit.start()
    unit.wait_inactive(60)
    assert unit.exits == [78] and unit.events("restart_prevented")
    assert "runtime_admission_launcher_selection_mismatch" in deployment.console_text()


def test_tc3_exit_70_loop_is_stopped_by_the_start_limit(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    record = deployment.root / "admission" / "controller.json"
    record.unlink()
    record.mkdir()  # an unreadable record: a transient controller error before CAS running
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(lambda item: bool(item.events("start_limit_hit")), 120), unit.log
    assert set(unit.exits) == {70} and len(unit.exits) == 5
    assert deployment.head()["status"] == "closed"


# --- T-C4 rollback -----------------------------------------------------------------------------------

def test_tc4_managed_rollback_to_an_older_launcher_slot_runs(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, archive="y")
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == 0
    stage.prepare_slot(artifacts["archives"]["x"], deployment.root, "older", deployment.state)
    assert stage.activate_slot(deployment.root, "older") == "slots/s/runtime"
    deployment.phase_file.unlink()
    unit = deployment.supervise("clean")
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert unit.stop() == 0
    head = deployment.head()
    assert stage.read_binding(deployment.root, head["binding_id"])["code_commit"] == artifacts["commits"]["x"]


def test_tc4_pre_launcher_artifact_is_refused_78(artifacts, tmp_path, deployments):
    assert "pre" in artifacts["archives"], "the pre-0d artifact must be built"
    deployment = _deploy(artifacts, tmp_path, deployments)
    stage.prepare_slot(artifacts["archives"]["pre"], deployment.root, "pre", deployment.state)
    with pytest.raises(ValueError, match="runtime_admission_launcher_contract_missing"):  # review P10
        stage.activate_slot(deployment.root, "pre")
    # serve's own refusal, rehearsed past the operator check
    assert stage.activate_slot(deployment.root, "pre", launcher_checks=False) == "slots/s/runtime"
    unit = deployment.supervise("clean")
    unit.start()
    unit.wait_inactive(60)
    assert unit.exits == [78] and "runtime_admission_launcher_contract_missing" in deployment.console_text()


def test_tc4_unmanaged_runtime_is_refused_at_selection(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments)
    runtime = writable_runtime(deployment.root / "slots" / "s" / "runtime")  # owner-level tampering
    (runtime / "src" / "core" / "runtime_admission.pyc").unlink()
    unit = deployment.supervise("clean")
    unit.start()
    unit.wait_inactive(60)
    assert unit.exits == [78] and "runtime_admission_candidate_unmanaged" in deployment.console_text()


# --- T-C7 reboot (manager cap; N1 = B fallback and clean modes) -------------------------------------

def test_tc7_reboot_with_the_vendor_5s_budget_then_boot_class_recovery(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, stop_grace=20.0, timeout_stop=60.0,
                         manager_timeout_stop="5s")
    unit = deployment.supervise("slow_stop", seconds=4.0)
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert "reboot budget insufficient (5s)" in unit.statuses
    unit.manager_stop(5.0)
    assert unit.exits[-1] == 68 and deployment.head()["status"] == "unknown"
    console = deployment.console_text()
    assert console.index("stop forwarded") < console.index("system state stopping")  # v3 N9 order
    calls = (deployment.sd_state / "calls.log").read_text()
    assert "is-system-running" in calls
    unit.reboot(deployment.boot_id)
    deployment.phase_file.unlink()
    unit = deployment.supervise("slow_stop", seconds=1.0)
    unit.start()
    assert unit.wait_for(_ready, 90), deployment.console_text()[-3000:]
    assert launcher._recoveries(deployment.root)[-1]["class"] == "boot"
    assert unit.stop() == 0 and deployment.head()["status"] == "closed"


def test_tc7_reboot_with_the_zz_dropin_budget_closes_cleanly(artifacts, tmp_path, deployments):
    deployment = _deploy(artifacts, tmp_path, deployments, stop_grace=20.0, timeout_stop=60.0,
                         manager_timeout_stop="75s")
    unit = deployment.supervise("slow_stop", seconds=2.0)
    unit.start()
    assert unit.wait_for(_ready, 60)
    assert not any("reboot budget" in item for item in unit.statuses)
    unit.manager_stop(75.0)
    assert unit.exits[-1] == 0 and deployment.head()["status"] == "closed"
    assert dict(deployment.phases())["fast_after"] == "True"  # the controller asked for a fast close
    assert not unit.events("sigkill_group") or unit.events("sigkill_group")[0][2] == "main_exited"


# --- T-C9 production-shaped relative paths ------------------------------------------------------------

def test_tc9_relative_model_path_is_refused_at_s3(artifacts, tmp_path, deployments):
    settings = "schema: t\nstt:\n  model: ./models/whisper-v3-turbo\nmemory:\n  db_path: ./data/memory.db\n"
    deployment = _deploy(artifacts, tmp_path, deployments, settings=settings, install_refused=True)
    assert json.loads(deployment.install_result.stderr) == {
        "refused": "runtime_admission_launcher_preflight", "problems": ["path:relative_path:stt.model"]}
    unit = deployment.supervise("clean")
    unit.start()
    unit.wait_inactive(60)
    assert unit.exits == [78]
    assert "runtime_admission_launcher_path:relative_path:stt.model" in deployment.console_text()
    assert deployment.head()["status"] == "closed"


# --- T-C6 measured timings ------------------------------------------------------------------------------

@pytest.mark.parametrize("gigabytes", [0.5, 2.0])
def test_tc6_measured_start_and_stop_timings(artifacts, tmp_path, deployments, gigabytes):
    if gigabytes > 1 and os.environ.get("KZA_LAUNCHER_TIMINGS_2G") != "1":
        pytest.skip("2 GB timing only with KZA_LAUNCHER_TIMINGS_2G=1")
    source_settings = rehearsal.SETTINGS
    chunk = os.urandom(1 << 20)
    total = int(gigabytes * (1 << 30))

    original = rehearsal.offline_source

    def offline(tmp, settings=source_settings):
        source = original(tmp, settings)
        payload = source / "data" / "captures"
        payload.mkdir()
        written, index = 0, 0
        while written < total:
            with (payload / f"part-{index:04d}.bin").open("wb") as out:
                for _ in range(64):
                    out.write(chunk)
            written += 64 << 20
            index += 1
        return source

    rehearsal.offline_source = offline
    try:
        started = time.monotonic()
        deployment = _deploy(artifacts, tmp_path, deployments)
        deploy_seconds = time.monotonic() - started
    finally:
        rehearsal.offline_source = original
    unit = deployment.supervise("clean")
    t0 = time.monotonic()
    unit.start()
    assert unit.wait_for(_ready, 300)
    ready_seconds = time.monotonic() - t0
    t1 = time.monotonic()
    assert unit.stop() == 0
    stop_seconds = time.monotonic() - t1
    measured = {"state_gigabytes": gigabytes, "deploy_prepare_activate_install_seconds": round(deploy_seconds, 2),
                "start_to_ready_seconds": round(ready_seconds, 2), "stop_to_sealed_exit_seconds": round(stop_seconds, 2),
                "build_seconds": round(artifacts["build_seconds"], 1)}
    evidence = os.environ.get("KZA_0D_EVIDENCE")
    if evidence:
        path = Path(evidence) / f"tc6-{gigabytes:g}gb.json"
        path.write_text(json.dumps(measured, indent=2))
    print("T-C6", json.dumps(measured))
