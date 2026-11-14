"""U3b-0e4: the compiled adoption rehearsals on Linux, in runbook order, under a real user systemd.

Design u3b-0e-adoption-design-v3, sub-unit U3b-0e4 (T-A1..T-A18). What the
in-process 0e2/0e3/0e5 tests only simulated is real here:

- every operator step is the archive's compiled CLI (``$R1/scripts/
  stage_runtime.pyc`` before W3, ``controller/current`` after), run as a
  subprocess; tests/mocks/compiled_stage.py only rebinds the 2a models root
  to the fixture's (and moves the wall clock for the day-14 step);
- the stand-ins for kza-voice, kza-llm-fast, kza-code-index, the window
  timers, a reverse-dependency pull-in and the boot target are real units of
  this account's user manager, read through real ``list-unit-files``,
  ``list-units``, ``show`` (EnvironmentFiles, Triggers, WantedBy...) output
  (tests/mocks/real_systemd.py; the guard never lets a production name reach
  systemctl);
- the scratch is the real tmpfs ``/run/user/<uid>`` (the CLI default, no
  test parameter) and the swap report reads the real /proc;
- the managed starts are real systemd starts of the compiled controller
  (Type=notify, READY relayed), with the rehearsal child
  (tests/mocks/launcher_child.py) instead of the full application, whose
  models and devices a rehearsal must not touch.

The legacy fixture is production-shaped (tests/mocks/adoption_legacy, written
by b0866e8's own code): the 63ea8d3 container (its runtime is the compiled
63ea8d3 artifact), hooks.enabled true, audit.db with text, the legacy events
table with markers, metrics files, LoRA adapters, the three unresolved
literal targets, a hot -journal, the symlink container and the 2026-10-29
backups. The legacy service itself is never started (it would load models):
W1's stop is a real ``systemctl stop`` of an inactive unit.

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

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time

import pytest

from scripts import adoption
from scripts import adoption_commands as commands
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import adoption_legacy as fixture
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks import real_systemd

pytestmark = pytest.mark.skipif(real_systemd.unavailable() is not None,
                                reason=real_systemd.unavailable() or "available")

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "tests" / "mocks" / "compiled_stage.py"
TEMPLATE = REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template"
CONTAINER, LEGACY = fixture.C_SLOT, fixture.B_SLOT
P1 = "u3b-0e-P1-D-rehearsal-0e4"
WINDOW = "r1-window-rehearsal-0e4"
P7 = "u3b-0e5-P7-B-rehearsal-0e4"
L4 = "l4-decision-rehearsal-0e4"
DAY = 86400
ERASED = "11111111-1111-4111-8111-111111111111"
STATUS = ("ActiveState", "SubState", "Result", "ExecMainStatus", "StatusText", "InvocationID", "UnitFileState")
CLEANUPS: list[dict] = []


# --- artifacts ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """R1 (HEAD plus this working tree) and the 63ea8d3 legacy artifact, both by scripts/build_runtime.py."""
    if not fixture.available():
        pytest.fail("the legacy commits are missing from this clone")
    base = tmp_path_factory.mktemp("0e4-artifacts")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    revision = rehearsal.overlay_commit(REPO, clone)
    started = time.monotonic()
    archive = base / "r1.tar.gz"
    manifest = build_runtime(clone, archive, revision)
    r1_seconds = time.monotonic() - started
    legacy = base / "63ea8d3.tar.gz"
    legacy_manifest = build_runtime(clone, legacy, "63ea8d3")
    assert legacy_manifest["git_commit"].startswith("63ea8d3")
    extracted = base / "r1"
    with tarfile.open(archive) as bundle:
        bundle.extractall(extracted, filter="data")
    real_systemd.record("window-build", {"r1_build_seconds": round(r1_seconds, 1), "r1_commit": manifest["git_commit"],
                                         "legacy_commit": legacy_manifest["git_commit"]})
    return {"archive": archive, "commit": manifest["git_commit"], "legacy": legacy,
            "r1_stage": extracted / "scripts" / "stage_runtime.pyc"}


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return fixture.b0866e8_tree(tmp_path_factory.mktemp("b0866e8") / "tree")


@pytest.fixture(scope="module", autouse=True)
def cleanup_evidence():
    # A killed earlier run (SIGKILL, OOM, dropped ssh) skipped its finalizers: sweep its units first.
    real_systemd.record("window-stale-sweep", real_systemd.sweep_stale())
    yield
    real_systemd.record("window-unit-cleanup", CLEANUPS)


@pytest.fixture
def systemd(tmp_path):
    manager = real_systemd.UserSystemd(tmp_path / "systemd")
    yield manager
    result = manager.cleanup()
    CLEANUPS.append(result)
    assert result["rehearsal_units_after"] == [] and result["leftover_paths"] == []
    assert result["guard_breaches"] == [] and result["daemon_reload_exit"] == 0


def tree_manifest(root: Path) -> dict:
    """(kind, inode, size, mtime_ns, mode, link, sha256) of every entry under ``root``; atime ignored."""
    found = {}
    for path in [Path(root), *sorted(Path(root).rglob("*"))]:
        info = path.lstat()
        found[path.relative_to(root).as_posix()] = (
            stat.S_IFMT(info.st_mode), info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode,
            os.readlink(path) if path.is_symlink() else None,
            None if path.is_symlink() or not path.is_file() else hashlib.sha256(path.read_bytes()).hexdigest())
    return found


# --- the world ---------------------------------------------------------------------------------------

VOICE_UNIT = """[Unit]
Description=KZA U3b-0e4 rehearsal stand-in for kza-voice (test only)
StartLimitBurst=3
StartLimitIntervalSec=300

[Service]
Type=simple
Restart=on-failure
RestartSec=10s
TimeoutStartSec=120s
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=
StandardOutput=append:{console}
StandardError=append:{console}

[Install]
WantedBy={boot}
"""
RELEASE_CONF = ("[Service]\nExecStart=\nExecStart={python} -B -P -m src.main\n"
                "Environment=PYTHONPATH={root}/current\nEnvironment=CONFIG_PATH=config/settings.yaml\n"
                "WorkingDirectory={root}/slots/{container}/state\n")
LLM_FAST = """[Unit]
Description=KZA U3b-0e4 rehearsal stand-in for kza-llm-fast (test only; never started)

[Service]
Type=simple
EnvironmentFile={env}
EnvironmentFile=-{missing}
ExecStart=/usr/bin/true llama-server -m ${{MODEL_DIR}}/qwen2.5-7b.gguf --port 8101 $LORA_ARGS
"""
CODE_INDEX = """[Unit]
Description=KZA U3b-0e4 rehearsal stand-in for kza-code-index (test only; never started)

[Service]
Type=simple
Environment=CONFIG_PATH=config/settings.yaml
WorkingDirectory={directory}
ExecStart=/usr/bin/true
"""
TIMER = """[Unit]
Description=KZA U3b-0e4 rehearsal window timer (test only)

[Timer]
OnCalendar=2099-01-01
Unit={service}
"""
TIMER_SERVICE = """[Unit]
Description=KZA U3b-0e4 rehearsal timer service (test only)

[Service]
Type=oneshot
{environment}ExecStart=/usr/bin/true
"""
PULL = """[Unit]
Description=KZA U3b-0e4 rehearsal reverse-dependency pull-in (test only; its timer never fires)
Wants={voice}

[Service]
Type=oneshot
ExecStart=/usr/bin/true
"""
HOOKS_CHECK = """import asyncio, json, threading
from contextlib import AsyncExitStack
import yaml
from src.hooks.audit_switch import audit_enabled, refuse_managed_audit
config = yaml.safe_load((state / 'config' / 'settings.yaml').read_text())
refuse_managed_audit(config)  # under the ticket: refuses an enabled audit
from src.main import load_policy_hooks
found = {{}}
def check():
    async def run():
        async with AsyncExitStack() as stack:
            registry = load_policy_hooks(config['hooks'], stack)
            stats = registry.get_stats()
            found['before'] = {{key: value for key, value in stats['before_handler_count'].items() if value}}
            found['after'] = sum(stats['after_handler_count'].values())
    asyncio.run(run())
thread = threading.Thread(target=check)
thread.start()
thread.join()
found['hooks_enabled'] = config['hooks'].get('enabled')
found['audit_enabled'] = audit_enabled(config['hooks'])
found['audit_db'] = (state / 'data' / 'audit.db').exists()
with open({output!r}, 'a') as stream:
    stream.write(json.dumps(found, sort_keys=True) + '\\n')
"""


class Window:
    """One production-shaped legacy root plus its stand-in units in the real user manager."""

    def __init__(self, base: Path, tree: Path, built: dict, systemd: real_systemd.UserSystemd, *,
                 adapters: bool = True) -> None:
        self.base, self.built, self.systemd = base, built, systemd
        self.layout = fixture.build_legacy(base / "legacy", tree, adapters=adapters)
        self.report = fixture.report_of(base / "legacy")
        self.root = base / "legacy" / "releases"
        self.models = self.layout.models_root
        settings = self.layout.legacy_state / "config" / "settings.yaml"
        text = settings.read_text()
        production = re.sub(r"(?m)^hooks:\n  enabled: false", "hooks:\n  enabled: true", text, count=1)
        assert production != text
        settings.write_text(production)  # production runs with hooks.enabled true
        runtime = self.root / "slots" / CONTAINER / "runtime"
        with tarfile.open(built["legacy"]) as bundle:
            bundle.extractall(runtime, filter="data")
        os.chmod(runtime, 0o700)
        (self.root / "current").symlink_to(Path("slots") / CONTAINER / "runtime")
        self._backups()
        self.layout.code_index.mkdir(exist_ok=True)
        self.r1_stage = built["r1_stage"]
        self.controller = self.root / "controller" / "current" / "scripts" / "stage_runtime.pyc"
        self.console = base / "legacy-console.log"
        self.phase_file = base / "phases.txt"
        self.hooks_file = base / "hooks.jsonl"
        self.units_dir = base / "units"
        self.units_dir.mkdir()
        self.timings: dict[str, float] = {}
        names = systemd
        self.voice = names.name("voice")
        self.watchdog_timer, self.backup_timer = names.name("watchdog", "timer"), names.name("backup", "timer")
        self.pull, self.pull_timer = names.name("pull"), names.name("pull", "timer")
        self.timers = (self.watchdog_timer, self.backup_timer)
        self._units()

    def _backups(self) -> None:
        """The 2026-10-29 publication backup: release.conf, config, .env and ten SQLite snapshots."""
        backup = self.root / "backups" / CONTAINER
        (backup / "databases").mkdir(parents=True)
        (backup / "release.conf").write_text("[Service]\nWorkingDirectory=/old\n")
        shutil.copytree(self.layout.legacy_state / "config", backup / "config")
        shutil.copyfile(self.layout.legacy_state / ".env", backup / ".env")
        data = self.layout.legacy_state / "data"
        databases = sorted(path.relative_to(data) for path in data.rglob("*")
                           if path.suffix in (".db", ".sqlite3") and path.is_file() and path.name != "lists.db")
        for relative in databases[:commands.LEGACY_BACKUP_DATABASES]:
            (backup / "databases" / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data / relative, backup / "databases" / relative)

    def llm_env(self, lora: str = "--verbose") -> None:
        (self.units_dir / "llm-fast.env").write_text(f"MODEL_DIR={self.models}\nLORA_ARGS=\"{lora}\"\n")

    def _units(self) -> None:
        sd = self.systemd
        sd.write_unit(sd.boot, "[Unit]\nDescription=KZA U3b-0e4 rehearsal boot target (test only)\n")
        sd.write_unit(self.voice, VOICE_UNIT.format(console=self.console, boot=sd.boot))
        sd.write_dropin(self.voice, "release.conf", RELEASE_CONF.format(python=sys.executable, root=self.root,
                                                                         container=CONTAINER))
        self.llm_env()
        sd.write_unit(sd.name("llm-fast"), LLM_FAST.format(env=self.units_dir / "llm-fast.env",
                                                           missing=self.units_dir / "absent.env"))
        sd.write_unit(sd.name("code-index"), CODE_INDEX.format(directory=self.layout.code_index))
        for timer, environment in ((self.watchdog_timer, f"Environment=PYTHONPATH={self.root}/current\n"),
                                   (self.backup_timer, "")):
            service = timer.replace(".timer", ".service")
            sd.write_unit(timer, TIMER.format(service=service))
            sd.write_unit(service, TIMER_SERVICE.format(environment=environment))
        self.add_pull()
        for verb, unit in (("enable", "kza-voice.service"), ("start", self.watchdog_timer),
                           ("start", self.backup_timer)):
            completed = sd.systemctl(verb, unit)
            assert completed.returncode == 0, (verb, unit, completed.stderr)

    def add_pull(self) -> None:
        """A loaded unit pulling kza-voice in: a service with Wants=kza-voice behind an active timer."""
        self.systemd.write_unit(self.pull, PULL.format(voice=self.voice))
        self.systemd.write_unit(self.pull_timer, TIMER.format(service=self.pull))
        self.systemd.reload()
        assert self.ctl("start", self.pull_timer).returncode == 0

    def drop_pull(self) -> None:
        """Remove the reverse-dependency pull-in (what the runbook requires before going on)."""
        assert self.ctl("stop", self.pull_timer).returncode == 0
        for unit in (self.pull_timer, self.pull):
            (self.systemd.unit_dir / unit).unlink()
        self.systemd.reload()

    def window_stop(self) -> None:
        """W1: kza-voice stopped and disabled (exit 0 only, read back), the window timers stopped."""
        assert self.ctl("stop", "kza-voice.service").returncode == 0
        assert self.ctl("disable", "kza-voice.service").returncode == 0
        assert self.ctl("is-enabled", "kza-voice.service").stdout.strip() == "disabled"
        for timer in self.timers:
            assert self.ctl("stop", timer).returncode == 0

    def install_launcher(self, *, timeout_start: float | None = None) -> dict:
        """W5 (console directory and rotation) and W6 (install-launcher, verified through real systemctl).

        Without ``timeout_start`` the runbook's W6: --timeout-start left at the product default."""
        (self.root / "logs").mkdir(mode=0o700, exist_ok=True)
        console = self.root / "logs" / "console.log"
        (self.root / launcher.LOGROTATE_NAME).write_text(launcher.render_console_logrotate(str(console)))
        started = time.monotonic()
        code, installed, refused = self.launcher_cli(
            "install-launcher", "--unit-dir", str(self.systemd.dropin_dir(self.voice)), "--template", str(TEMPLATE),
            "--python", sys.executable, "--stop-grace", "20",
            *(() if timeout_start is None else ("--timeout-start", f"{timeout_start:g}")),
            "--timeout-stop", "60", "--console-log", str(console))
        self.timings["install_launcher"] = round(time.monotonic() - started, 2)
        assert code == 0 and installed["verification_mismatches"] == [], (installed, refused)
        return installed

    # --- compiled CLIs ---------------------------------------------------------------------------

    def stage(self, *arguments: str, pyc: Path | None = None, wall_offset: float = 0.0, step: str | None = None,
              timeout: float = 1800) -> tuple[int, dict | None, dict | None]:
        """One compiled stage_runtime command: (exit code, stdout JSON, stderr JSON)."""
        environment = self.systemd.environment({"KZA_REHEARSAL_MODELS_ROOT": os.fspath(self.models)})
        if wall_offset:
            environment["KZA_REHEARSAL_WALL_OFFSET"] = str(wall_offset)
        started = time.monotonic()
        completed = subprocess.run([sys.executable, "-I", "-B", str(WRAPPER), str(pyc or self.r1_stage),
                                    "--root", str(self.root), "--offline", *arguments], capture_output=True,
                                   text=True, timeout=timeout, env=environment)
        if step:
            self.timings[step] = round(time.monotonic() - started, 2)
        return completed.returncode, _last_json(completed.stdout), _last_json(completed.stderr)

    def launcher_cli(self, *arguments: str) -> tuple[int, dict | None, dict | None]:
        """A launcher command with the installed controller's own compiled CLI (no wrapper)."""
        completed = subprocess.run([sys.executable, "-I", "-B", str(self.controller), "--root", str(self.root),
                                    "--offline", *arguments], capture_output=True, text=True, timeout=600,
                                   env=self.systemd.environment())
        return completed.returncode, _last_json(completed.stdout), _last_json(completed.stderr)

    # --- systemd ---------------------------------------------------------------------------------

    def ctl(self, *arguments: str) -> subprocess.CompletedProcess:
        return self.systemd.systemctl(*arguments)

    def voice_state(self) -> dict:
        return self.systemd.raw_show(self.voice, *STATUS)

    def fake_boot(self, step: str) -> dict:
        """Start (then stop) the rehearsal boot target: the stand-in must not start (T-A10)."""
        before = self.voice_state()
        completed = self.ctl("start", self.systemd.boot)
        assert completed.returncode == 0, completed.stderr
        time.sleep(1.0)
        after = self.voice_state()
        self.ctl("stop", self.systemd.boot)
        assert after["ActiveState"] in ("inactive", "failed"), (step, after)
        assert after["InvocationID"] == before["InvocationID"], (step, before, after)
        return {"step": step, "active_state": after["ActiveState"], "unit_file_state": after["UnitFileState"],
                "started": after["InvocationID"] != before["InvocationID"]}

    def child(self, scenario: str, *, restart: bool = False, hooks: bool = True, seconds: float = 0.0) -> None:
        """zz-rehearsal.conf after W6: the rendered managed ExecStart plus the rehearsal child."""
        script = None
        if hooks:
            script = self.base / "hooks-check.py"
            script.write_text(HOOKS_CHECK.format(output=str(self.hooks_file)))
        command = rehearsal.child_command(self.phase_file, scenario, execute=script, seconds=seconds)
        managed = (self.systemd.dropin_dir(self.voice) / "managed.conf").read_text()
        import shlex
        line = " ".join(shlex.quote(token) for token in [*launcher._rendered_exec_start(managed),
                                                         "--child-command", json.dumps(command)])
        self.systemd.write_dropin(self.voice, "zz-rehearsal.conf", "[Service]\nExecStart=\nExecStart=" + line + "\n"
                                  + ("" if restart else "Restart=no\n"))
        self.systemd.reload()

    def start_managed(self, timeout: float = 180) -> tuple[dict, float]:
        """Start and wait for this start's own outcome: a new InvocationID, so a previous run's
        'failed' is never read as this one's."""
        previous = self.systemd.raw_show(self.voice, "InvocationID")["InvocationID"]
        started = time.monotonic()
        completed = self.ctl("start", "--no-block", "kza-voice.service")
        assert completed.returncode == 0, completed.stderr
        shown = self.systemd.wait_for(self.voice, lambda s: s["InvocationID"] not in ("", previous) and (
                                      s["ActiveState"] in ("active", "failed")
                                      or (s["ActiveState"] == "inactive" and s["Result"] != "success")),
                                      timeout, *STATUS)
        return shown, time.monotonic() - started

    def stop_managed(self) -> tuple[subprocess.CompletedProcess, float]:
        started = time.monotonic()
        completed = self.ctl("stop", "kza-voice.service")
        return completed, time.monotonic() - started

    def head(self):
        return stage.read_admission_head(self.root)

    def managed_console(self) -> str:
        path = self.root / "logs" / "console.log"
        return path.read_text() if path.exists() else ""


def _sources(conversation: str) -> str:
    """A policy event's sources_json as R1 writes it (capture, conversation, captured_at, expires_at)."""
    return json.dumps([{"capture_id": "capture-0e4", "conversation_id": conversation, "captured_at": 1_799_999_000.0,
                        "expires_at": 9e9}])


def _units_now(w: "Window", plan_sha256: str) -> dict:
    """What commit compares for its 'units' item, read now through the guard (diagnostics)."""
    plan = json.loads((stage.adoption_directory(w.root) / ("plan-" + plan_sha256 + ".json")).read_text())
    manager = launcher.ServiceManager(executable=str(w.systemd.shim))
    timers = tuple(item for item in plan["units"]["units"] if item != launcher.UNIT)
    observed, problems = commands.observe_units(manager, timers, window=True)
    found = {"plan": plan["units"], "now": observed, "problems": problems,
             "plan_dispositions": plan["timer_dispositions"],
             "now_dispositions": commands.timer_dispositions(manager, timers, w.root, commands._layout_of(plan))}
    real_systemd.record("units-diagnostic", found)
    return found


def _last_json(text: str):
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _legacy_trees(window: Window) -> dict:
    return {name: tree_manifest(window.base / "legacy" / name) for name in ("kza", "logs")} | {
        "legacy_state": tree_manifest(window.layout.legacy_state),
        "container": tree_manifest(window.root / "slots" / CONTAINER)}


def _sqlite_bytes(path: Path) -> dict:
    return {item.name: item.read_bytes() for item in path.parent.glob(path.name + "*")}


def _restore(path: Path, saved: dict) -> None:
    for item in path.parent.glob(path.name + "*"):
        item.unlink()
    for name, data in saved.items():
        (path.parent / name).write_bytes(data)


# --- the window, W0..W9, then day 14 (Q5) and a restore after a crash ---------------------------------------

def test_the_window_in_runbook_order_under_real_systemd(tmp_path, tree, built, systemd):
    """T-A1, T-A10, T-A11, T-A12/T-A14, T-A15, T-A16, T-A17, T-A5 and the T-A8 step timings."""
    w = Window(tmp_path.resolve(), tree, built, systemd)
    evidence: dict = {"boots": []}
    common = ("--container-slot", CONTAINER, "--archive", str(built["archive"]), "--slot", "r1",
              "--window-timer", w.watchdog_timer, "--window-timer", w.backup_timer)

    # W0, advisory dry-run with the service's view as production has it: the scratch is the real tmpfs
    # /run/user/<uid> (CLI default). Two expected refusals first: an .env path value into the legacy
    # state and a unit pulling kza-voice in.
    env_file = w.layout.legacy_state / ".env"
    original_env = env_file.read_bytes()
    env_file.write_text(original_env.decode() + f"KZA_EXTRA_DIR={w.layout.legacy_state}/data/extra\n")
    code, report, refused = w.stage("adoption-dry-run", *common, step="dry_run_refused")
    assert code == 78, (report, refused)
    assert "runtime_admission_adoption_env_path:KZA_EXTRA_DIR" in report["problems"]
    assert "runtime_admission_adoption_reverse_dependency" in report["problems"]
    assert report["units"]["reverse_dependencies"] == [w.pull]
    assert report["scratch"] == {"kind": "tmpfs", "removed": True}
    env_file.write_bytes(original_env)  # the reviewed correction, before W1b
    w.drop_pull()
    trees = _legacy_trees(w)
    root_before = tree_manifest(w.root)
    code, report, refused = w.stage("adoption-dry-run", *common, step="dry_run")
    assert code == 0, (report, refused)
    assert tree_manifest(w.root) == root_before and _legacy_trees(w) == trees  # T-D1
    assert report["problems"] == [] and report["warnings"] == [
        "advisory: legacy_training assumed retired (P1=D pending, W1b)"]
    assert report["scratch"] == {"kind": "tmpfs", "removed": True}
    assert report["swap"]["known"] is True
    assert report["units"]["reverse_dependencies"] == []
    assert report["units"]["units"]["kza-voice.service"]["UnitFileState"] == "enabled"
    assert report["timer_dispositions"] == [
        {"unit": w.watchdog_timer, "references": ["managed_code"], "disposition": "review"},
        {"unit": w.backup_timer, "references": [], "disposition": "restart unchanged"}]
    assert report["code_index"]["unit"] == commands.CODE_INDEX_UNIT and report["code_index"]["state"] == "loaded"
    assert report["code_index"]["config_class"] not in ("container", "legacy_state")
    census = report["adoption"]["residue_census"]
    assert census["voice_profile"] == census["voice_consent"] == census["conversation_unmarkable"] == 0
    assert report["legacy_path_census"]["legacy_checkout_targets"] == 0
    assert report["journal_files"] == 1
    scratch_left = [name for name in os.listdir(os.environ["XDG_RUNTIME_DIR"]) if name.startswith(".kza-adoption")]
    assert scratch_left == []
    evidence["dry_run"] = {"timings": report["timings"], "sizes": report["sizes"], "swap": report["swap"],
                           "free": report["free"], "adoption_timings": report["adoption"].get("timings"),
                           "code_index": report["code_index"]}

    # W1: stop and disable kza-voice (exit 0 only), stop the window timers. Nothing starts it from here.
    w.window_stop()
    evidence["boots"].append(w.fake_boot("W1"))

    # W1b (T-A12/T-A14, P1=D): a unit passing --lora under a LoRA root makes retire refuse, nothing deleted.
    w.llm_env(lora=f"--lora {w.models}/lora_adapters/custom_adapter")
    systemd.reload()
    models_before = tree_manifest(w.models)
    code, _out, refused = w.stage("retire-legacy-training", "--decision", P1, "--container-slot", CONTAINER,
                                  "--unit", "kza-llm-fast.service")
    assert code == 78 and refused["refused"].startswith("runtime_admission_retire_unit_under_root"), refused
    assert tree_manifest(w.models) == models_before
    w.llm_env()  # the reviewed unit: no adapter
    systemd.reload()
    code, retired, refused = w.stage("retire-legacy-training", "--decision", P1, "--container-slot", CONTAINER,
                                     "--unit", "kza-llm-fast.service", step="retire")
    assert code == 0, refused
    assert not (w.models / "lora_adapters").exists() and (w.models / "whisper-v3-turbo" / "model.bin").exists()
    record = json.loads((stage.adoption_directory(w.root) / retired["record"]).read_text())
    assert record["units_checked"] == ["kza-llm-fast.service", "kza-voice.service"]
    assert record["units"]["environment_file_keys"] == {"kza-llm-fast.service": ["LORA_ARGS", "MODEL_DIR"]}
    assert record["units"]["discovered"] > 20  # every unit of this real user manager was read
    evidence["retire"] = {"units_discovered": record["units"]["discovered"], "record": retired["record"]}
    evidence["boots"].append(w.fake_boot("W1b"))

    # W2: prepare (tmpfs default scratch, both window timers).
    code, prepared, refused = w.stage("adoption-prepare", *common, "--decision", WINDOW, step="prepare")
    assert code == 0, refused
    plan_sha256 = prepared["plan_sha256"]
    plan = json.loads((stage.adoption_directory(w.root) / ("plan-" + plan_sha256 + ".json")).read_text())
    assert os.readlink(w.root / "current") == f"slots/{CONTAINER}/runtime" and w.head() is None
    evidence["boots"].append(w.fake_boot("W2"))

    # W3: install-controller from the extracted R1 (controller epoch 2).
    code, _out, refused = w.stage("install-controller", str(built["archive"]), step="install_controller")
    assert code == 0, refused
    evidence["boots"].append(w.fake_boot("W3"))

    # W4: a file re-added under a retired root refuses the commit before the CAS; then the commit.
    (w.models / "lora_adapters").mkdir()
    (w.models / "lora_adapters" / "late.safetensors").write_bytes(b"late")
    code, _out, refused = w.stage("adoption-commit", "--confirm", plan_sha256, pyc=w.controller)
    assert code == 78 and refused == {"refused": "runtime_admission_adoption_plan_mismatch:retirement"}, refused
    assert w.head() is None
    shutil.rmtree(w.models / "lora_adapters")
    code, committed, refused = w.stage("adoption-commit", "--confirm", plan_sha256, pyc=w.controller,
                                       step="commit")
    assert code == 0, (refused, _units_now(w, plan_sha256))
    head = w.head()
    assert head["generation"] == 1 and head["status"] == "closed"
    assert os.readlink(w.root / "current") == "slots/r1/runtime"
    adoption_record = json.loads(stage.adoption_record_path(w.root, head).read_text())
    assert retired["record"] in adoption_record["legacy_training"] and P1 in adoption_record["legacy_training"]
    state = w.root / "slots" / "r1" / "state"
    assert not (state / "data" / "audit.db").exists()  # P2=A (T-A15)
    _no_marker_byte(state)
    evidence["boots"].append(w.fake_boot("W4"))

    # T-A10: an explicit legacy start now runs R1 through 'current' without a ticket; G1 refuses (78).
    # So does a start pulled in by another unit (Wants= ignores 'disabled': why W0 refuses a pull-in).
    legacy = _legacy_trees(w)
    starts = {}
    for how in ("explicit", "pull_in"):
        before = w.voice_state()["InvocationID"]
        if how == "explicit":
            assert w.ctl("start", "--no-block", "kza-voice.service").returncode == 0
        else:
            w.add_pull()
            assert w.ctl("start", "--no-block", w.pull).returncode == 0
        shown = systemd.wait_for(w.voice, lambda s: s["InvocationID"] != before and s["ExecMainStatus"] == "78",
                                 60, *STATUS)
        assert shown["ExecMainStatus"] == "78", (how, shown)
        assert w.ctl("stop", "kza-voice.service").returncode == 0  # cancels the on-failure restart
        if how == "pull_in":
            w.drop_pull()
        starts[how] = {"exec_main_status": shown["ExecMainStatus"], "result": shown["Result"]}
    assert w.console.read_text().count("runtime_admission_unmanaged_under_managed_root") >= 2
    assert _legacy_trees(w) == legacy and w.head() == head
    assert w.ctl("is-enabled", "kza-voice.service").stdout.strip() == "disabled"
    evidence["g1_starts"] = starts

    # W5 (console directory, rotation) and W6 (install-launcher, verified through real systemctl).
    installed = w.install_launcher()
    # F1=C: the default TimeoutStartSec (240 s) reaches the real unit and covers a start without EXTEND.
    assert systemd.raw_show(w.voice, "TimeoutStartUSec")["TimeoutStartUSec"] == "4min"
    w.child("clean")
    evidence["boots"].append(w.fake_boot("W6"))
    evidence["stop_timeout_dropin"] = installed["stop_timeout_dropin"]["value_seconds"]

    # W7 is the user's sudo drop-in: not applied here; serve reads this manager's budget (fallback mode).
    # W8: enable, start; restart only the 'restart unchanged' timers (T-A16).
    assert w.ctl("enable", "kza-voice.service").returncode == 0
    assert w.ctl("is-enabled", "kza-voice.service").stdout.strip() == "enabled"
    runtime_before = tree_manifest(w.root / "slots" / "r1" / "runtime")
    shown, seconds = w.start_managed()
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-4000:])
    w.timings["w8_start_to_ready"] = round(seconds, 2)
    dispositions = {item["unit"]: item["disposition"] for item in plan["timer_dispositions"]}
    for timer, disposition in dispositions.items():
        if disposition == "restart unchanged":
            assert w.ctl("start", timer).returncode == 0
    assert systemd.raw_show(w.backup_timer, "ActiveState")["ActiveState"] == "active"
    assert systemd.raw_show(w.watchdog_timer, "ActiveState")["ActiveState"] == "inactive"
    assert dispositions == {w.watchdog_timer: "review", w.backup_timer: "restart unchanged"}

    # W9: a clean stop (receipt, closure seal: DAY 0) and a second start (T-A11).
    completed, stop_seconds = w.stop_managed()
    assert completed.returncode == 0 and w.head()["status"] == "closed"
    w.timings["w9_stop"] = round(stop_seconds, 2)
    shown, seconds = w.start_managed()
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-4000:])
    w.timings["second_start_to_ready"] = round(seconds, 2)
    completed, stop_seconds = w.stop_managed()
    assert completed.returncode == 0 and w.head()["status"] == "closed"
    w.timings["second_stop"] = round(stop_seconds, 2)
    assert tree_manifest(w.root / "slots" / "r1" / "runtime") == runtime_before  # the artifact is read-only
    checks = [json.loads(line) for line in w.hooks_file.read_text().splitlines()]
    assert len(checks) == 2 and all(item == checks[0] for item in checks)
    assert checks[0]["hooks_enabled"] is True and checks[0]["audit_enabled"] is False
    assert checks[0]["audit_db"] is False and checks[0]["after"] == 0
    assert checks[0]["before"].get("before_tts_speak") == 1 and checks[0]["before"].get("before_ha_action", 0) >= 1
    assert not (state / "data" / "audit.db").exists()
    evidence["hooks"] = checks[0]
    exits = sorted((w.root / "admission" / "runs").glob("*.exit.json"))
    assert [json.loads(path.read_text())["sealed"] for path in exits] == [True, True]
    # The runbook's check after W9 (review U3b-0e4 F5), compiled and read-only. The rehearsal child is not
    # the application, so the fixture's queued index deletion stays open here (gap G1): recorded as found.
    code, status, refused = w.launcher_cli("completion-status")
    assert code == 0 and status["head_status"] == "closed", refused
    assert status["index_delete_jobs"] == 1 and status["complete"] is False
    assert set(status["conversation_exempt"].values()) <= {0} and not any(status["resurrected"].values())
    evidence["completion_status_after_w9"] = status

    # Day 14 (T-A17, Q5). First the link check alone: with the b0866e8 state gone and no Q5 record,
    # L4 refuses with legacy_state and changes nothing.
    restore_sources = tmp_path / "restore-sources"
    restore_sources.mkdir()
    for name in ("memory_privacy.db", "events.db"):
        for item in (w.layout.legacy_state / "data").glob(name + "*"):
            shutil.copyfile(item, restore_sources / item.name)
    unit_files = {path.name: path.read_bytes() for path in systemd.dropin_dir(w.voice).iterdir()}
    aside = w.layout.legacy_state.with_name("state.aside")
    os.rename(w.layout.legacy_state, aside)
    code, _out, refused = w.launcher_cli("uninstall-launcher", "--unit-dir", str(systemd.dropin_dir(w.voice)),
                                         "--user-decision", L4, "--phrase", launcher.UNINSTALL_PHRASE,
                                         "--legacy-slot", CONTAINER)
    os.rename(aside, w.layout.legacy_state)
    assert code == 78 and refused["refused"] == "runtime_admission_uninstall_legacy_state", refused
    assert {path.name: path.read_bytes() for path in systemd.dropin_dir(w.voice).iterdir()} == unit_files
    due = 14 * DAY + 3600
    code, dry, refused = w.stage("retire-legacy-state", "--decision", P7, "--container-slot", CONTAINER,
                                 "--dry-run", wall_offset=due, step="q5_dry_run")
    assert code == 0 and dry["eligible"] is True, (dry, refused)
    assert len(dry["backups"]["databases"]) == 10
    code, q5, refused = w.stage("retire-legacy-state", "--decision", P7, "--container-slot", CONTAINER,
                                "--confirm", dry["plan_sha256"], wall_offset=due, step="q5_confirm")
    assert code == 0, refused
    assert not os.path.lexists(w.layout.legacy_state)
    for slot in (CONTAINER, LEGACY):
        code, _out, refused = w.launcher_cli("uninstall-launcher", "--unit-dir", str(systemd.dropin_dir(w.voice)),
                                             "--user-decision", L4, "--phrase", launcher.UNINSTALL_PHRASE,
                                             "--legacy-slot", slot)
        assert code == 78 and refused["refused"] == "runtime_admission_uninstall_legacy_retired", refused
        assert refused["problems"] == [q5["record"]]
    assert {path.name: path.read_bytes() for path in systemd.dropin_dir(w.voice).iterdir()} == unit_files
    shown, _seconds = w.start_managed()  # the managed service is unaffected
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-4000:])
    assert w.stop_managed()[0].returncode == 0 and w.head()["status"] == "closed"
    evidence["q5"] = {"record": q5["record"], "databases": len(dry["backups"]["databases"])}

    # T-A5 / T-C8 (restore after a crash): each restore over the adopted state is refused by automatic R-b
    # under real systemd; the operator's read-only 'adopt --dry-run' names the family.
    w.child("crash", hooks=False, seconds=1.0)
    w.phase_file.unlink()
    shown, _seconds = w.start_managed()
    final = systemd.wait_for(w.voice, lambda s: s["ActiveState"] in ("inactive", "failed"), 60, *STATUS)
    assert w.head()["status"] == "unknown", (shown, final, w.managed_console()[-3000:])
    assert final["ExecMainStatus"] == "68"
    w.child("clean", hooks=False)
    data = state / "data"
    privacy, events = data / "memory_privacy.db", data / "events.db"
    saved = {"privacy": _sqlite_bytes(privacy), "events": _sqlite_bytes(events)}
    variants = {}

    def legacy_privacy():
        _restore(privacy, {item.name: item.read_bytes() for item in restore_sources.glob("memory_privacy.db*")})

    def restored_event():
        with sqlite3.connect(events) as db:
            db.execute("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,sources_json,"
                       "metrics_json,expires_at) VALUES (1,'a','b','c',?,'{}',9e9)",
                       (_sources(ERASED),))
        db.close()

    def restored_fact():
        with sqlite3.connect(privacy) as db:
            db.execute("UPDATE facts SET state='confirmed' WHERE id=?", (w.report["facts"]["deleted"],))
        db.close()

    def restored_voice():
        with sqlite3.connect(privacy) as db:
            db.execute("UPDATE voice_profiles SET deleted_at=NULL WHERE id='voice_deleted_profile'")
        db.close()

    for name, restore in (("legacy_memory_privacy", legacy_privacy), ("conversation", restored_event),
                          ("fact", restored_fact), ("voice_profile", restored_voice)):
        restore()
        began = time.monotonic()
        shown, _seconds = w.start_managed()
        final = systemd.wait_for(w.voice, lambda s: s["ActiveState"] in ("inactive", "failed"), 120, *STATUS)
        seconds = time.monotonic() - began
        code, dry_adopt, refused = w.launcher_cli("adopt", "--dry-run")
        variants[name] = {"exec_main_status": final["ExecMainStatus"], "status_text": final["StatusText"],
                          "seconds": round(seconds, 2), "dry_run": dry_adopt or refused}
        assert final["ExecMainStatus"] == "69", (name, final, w.managed_console()[-3000:])
        assert final["StatusText"].startswith("recovery refused: runtime_admission_recovery_"), final
        assert w.head()["status"] == "unknown"
        _restore(privacy, saved["privacy"])
        _restore(events, saved["events"])
    assert variants["conversation"]["status_text"].endswith("resurrection")
    assert variants["fact"]["status_text"].endswith("resurrection")
    assert variants["voice_profile"]["status_text"].endswith("resurrection")
    # The 2026-10-29 legacy file predates R1's privacy schema: R-b's checks refuse it before any
    # ledger comparison (a restore across the adoption is never adopted, whichever check sees it).
    assert variants["legacy_memory_privacy"]["status_text"].endswith("check_failed"), variants
    assert variants["legacy_memory_privacy"]["dry_run"]["report"]["failures"] == ["runtime_admission_unknown_schema"]
    # Five starts inside StartLimitIntervalSec (the crash and four refusals) exhaust the managed unit's
    # start limit (StartLimitBurst=5 in 300 s, DEPLOYMENT): the operator's reset-failed, after putting
    # the adopted bytes back, clears it.
    limits = systemd.raw_show(w.voice, "StartLimitBurst", "StartLimitIntervalUSec")
    assert limits == {"StartLimitBurst": "5", "StartLimitIntervalUSec": "5min"}, limits
    before, runs = w.voice_state()["InvocationID"], w.managed_console().count("reboot budget")
    sixth = w.ctl("start", "kza-voice.service")  # refused by systemd before anything runs (review F11)
    journal = subprocess.run(["journalctl", "--user", "-u", w.voice, "-n", "5", "-o", "cat", "--no-pager"],
                             capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL).stdout
    assert sixth.returncode != 0 and "Start request repeated too quickly" in journal, (sixth.stderr, journal)
    assert w.voice_state()["InvocationID"] == before and w.managed_console().count("reboot budget") == runs
    assert w.ctl("reset-failed", "kza-voice.service").returncode == 0
    evidence["start_limit"] = dict(limits, sixth_start_exit=sixth.returncode,
                                   journal="Start request repeated too quickly.")
    shown, seconds = w.start_managed()  # the adopted bytes back: automatic R-b closes it once
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-3000:])
    assert launcher._recoveries(w.root)[-1]["mode"] == "automatic"
    assert w.stop_managed()[0].returncode == 0 and w.head()["status"] == "closed"
    evidence["restore_after_crash"] = variants
    evidence["timings"] = w.timings
    real_systemd.record("window", evidence)


def _no_marker_byte(state: Path) -> None:
    """T-A15 (P2=A): no audit or legacy-events marker byte anywhere in the adopted slot state."""
    markers = (b"KZA-LEGACY-EVENTS-MARKER", b"KZA-LEGACY-AUDIT-MARKER")
    for path in state.rglob("*"):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            assert not any(marker in data for marker in markers), path


# --- T-A13: the aborts under real systemd (enable read back) ------------------------------------------

def test_aborts_under_real_systemd(tmp_path, tree, built, systemd):
    """T-A13 and W-abort-1: refusals (legacy changed, wrong head, later ticket), success at generation > 1."""
    w = Window(tmp_path.resolve(), tree, built, systemd, adapters=False)
    w.drop_pull()
    common = ("--container-slot", CONTAINER, "--archive", str(built["archive"]))
    unit_dir = str(systemd.dropin_dir(w.voice))

    window_stop = w.window_stop

    def prepare(slot):
        code, prepared, refused = w.stage("adoption-prepare", *common, "--slot", slot, "--decision", WINDOW,
                                          "--window-timer", w.watchdog_timer, "--window-timer", w.backup_timer)
        assert code == 0, refused
        return prepared["plan_sha256"]

    window_stop()
    # W-abort-1: the marked slot goes, kza-voice is enabled again (exit 0 and is-enabled read back).
    prepare("r1")
    code, aborted, refused = w.stage("adoption-abort", "--prepared-slot", "r1")
    assert code == 0 and aborted["enabled"] is True, (aborted, refused)
    assert not (w.root / "slots" / "r1").exists()
    assert w.ctl("is-enabled", "kza-voice.service").stdout.strip() == "enabled"
    window_stop()
    # First adoption, then W-abort-2 refusals and success.
    plan = prepare("r2")
    assert w.stage("install-controller", str(built["archive"]))[0] == 0
    code, _out, refused = w.stage("adoption-commit", "--confirm", plan, pyc=w.controller)
    assert code == 0, (refused, _units_now(w, plan))
    first = w.head()
    log = w.layout.legacy_state / "logs" / "voice_assistant.log"
    info, original = log.stat(), log.read_bytes()
    log.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    os.utime(log, ns=(info.st_atime_ns, info.st_mtime_ns))
    code, _out, refused = w.stage("adoption-abort", "--unit-dir", unit_dir, pyc=w.controller)
    assert code == 78 and refused == {"refused": "runtime_admission_adoption_abort_legacy_changed"}, refused
    log.write_bytes(original)
    os.utime(log, ns=(info.st_atime_ns, info.st_mtime_ns))
    code, aborted, refused = w.stage("adoption-abort", "--unit-dir", unit_dir, pyc=w.controller)
    assert code == 0 and aborted["enabled"] is True, (aborted, refused)
    assert os.readlink(w.root / "current") == f"slots/{CONTAINER}/runtime" and not (w.root / "slots" / "r2").exists()
    assert w.ctl("is-enabled", "kza-voice.service").stdout.strip() == "enabled"
    assert stage.head_aborted(w.root, first)
    with stage.AdmissionController(w.root) as controller:
        with pytest.raises(ValueError, match="^runtime_admission_adoption_aborted_head$"):
            controller.begin_run()
    # Second adoption on the aborted head (generation 2), aborted again with the controller installed.
    window_stop()
    plan = prepare("r3")
    code, _out, refused = w.stage("adoption-commit", "--confirm", plan, pyc=w.controller)
    assert code == 0, (refused, _units_now(w, plan))
    second = w.head()
    assert second["generation"] == 2
    code, aborted, refused = w.stage("adoption-abort", "--unit-dir", unit_dir, pyc=w.controller)
    assert code == 0 and aborted["enabled"] is True, (aborted, refused)
    # Third adoption: the wrong head (a selection after activation) and a later ticket refuse; L4 only.
    window_stop()
    plan = prepare("r4")
    code, _out, refused = w.stage("adoption-commit", "--confirm", plan, pyc=w.controller)
    assert code == 0, (refused, _units_now(w, plan))
    third = w.head()
    assert third["generation"] == 3
    with stage.AdmissionController(w.root) as controller:
        controller.activate("r4", managed=True)
    code, _out, refused = w.stage("adoption-abort", "--unit-dir", unit_dir, pyc=w.controller)
    assert code == 78 and refused == {"refused": "runtime_admission_adoption_abort_head"}, refused
    w.install_launcher()
    w.child("clean", hooks=False)
    shown, _seconds = w.start_managed()
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-3000:])
    assert w.stop_managed()[0].returncode == 0
    code, _out, refused = w.stage("adoption-abort", "--unit-dir", unit_dir, pyc=w.controller)
    assert code == 78 and refused["refused"] in ("runtime_admission_adoption_abort_head",
                                                 "runtime_admission_adoption_abort_after_run"), refused
    assert (w.root / "slots" / "r4").is_dir() and os.readlink(w.root / "current") == "slots/r4/runtime"
    real_systemd.record("aborts", {"generations": [first["generation"], second["generation"], third["generation"]],
                                   "later_ticket_refusal": refused["refused"]})


# --- T-A18 (P3): residue refused by the compiled prepare -------------------------------------------------

@pytest.mark.parametrize("variant, code", [("voice", "voice_profile"), ("orphan_turn", "conversation")])
def test_unmarkable_residue_aborts_the_compiled_prepare(tmp_path, tree, built, systemd, variant, code):
    w = Window(tmp_path.resolve(), tree, built, systemd, adapters=False)
    w.drop_pull()
    w.window_stop()
    with sqlite3.connect(w.layout.legacy_state / "data" / "memory_privacy.db") as db:
        if variant == "voice":
            db.execute("INSERT INTO voice_fingerprints(id,model_key,dimensions,embedding,duration_s,created_at,"
                       "expires_at,sources_json,suggested_profile,suggested_revision,similarity)"
                       " VALUES ('fp','ecapa',2,x'00',1,1,9e9,'[]','voice_deleted_profile',1,.9)")
        else:
            db.execute("INSERT INTO conversation_turns_v1 VALUES ('orphan','g',?,'unmarked','d')",
                       (_sources(ERASED),))
    db.close()
    code_, report, refused = w.stage("adoption-dry-run", "--container-slot", CONTAINER, "--archive",
                                     str(built["archive"]), "--slot", "r1")
    assert code_ == 78 and "runtime_admission_adoption_residue:" + code in report["problems"], (report, refused)
    code_, _out, refused = w.stage("adoption-prepare", "--container-slot", CONTAINER, "--archive",
                                   str(built["archive"]), "--slot", "r1", "--decision", WINDOW,
                                   "--window-timer", w.watchdog_timer, "--window-timer", w.backup_timer)
    assert code_ == 78 and refused["refused"] == "runtime_admission_adoption_residue:" + code, refused
    assert not (w.root / "slots" / "r1").exists() and w.head() is None


def test_a_conversation_residue_is_requeued_in_the_compiled_commit(tmp_path, tree, built, systemd):
    """T-A18: a restored conversation payload reopens its governing step in the committed state."""
    w = Window(tmp_path.resolve(), tree, built, systemd, adapters=False)
    w.drop_pull()
    w.window_stop()
    with sqlite3.connect(w.layout.legacy_state / "data" / "events.db") as db:
        db.execute("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,sources_json,metrics_json,"
                   "expires_at) VALUES (1,'a','b','c',?,'{}',9e9)", (_sources(ERASED),))
    db.close()
    code, prepared, refused = w.stage("adoption-prepare", "--container-slot", CONTAINER, "--archive",
                                      str(built["archive"]), "--slot", "r1", "--decision", WINDOW,
                                      "--window-timer", w.watchdog_timer, "--window-timer", w.backup_timer)
    assert code == 0, refused
    assert w.stage("install-controller", str(built["archive"]))[0] == 0
    code, _out, refused = w.stage("adoption-commit", "--confirm", prepared["plan_sha256"], pyc=w.controller)
    assert code == 0, (refused, _units_now(w, prepared["plan_sha256"]))
    state = w.root / "slots" / "r1" / "state"
    assert prepared["residue"] == {"conversation_operations_created": 0, "conversation_steps_reopened": 1,
                                   "facts_requeued": 0}
    events = [target["id"] for target in adoption.registered_targets(state) if target["participant"] == "events"]
    with sqlite3.connect((state / "data" / "memory_privacy.db").as_uri() + "?mode=ro", uri=True) as db:
        steps = db.execute("SELECT state, last_error FROM erasure_steps WHERE target_id=?", (events[0],)).fetchall()
    db.close()
    assert steps == [("pending", "interrupted_by_restart")]
    record = json.loads(stage.adoption_record_path(w.root, w.head()).read_text())
    assert record["requeued"] == prepared["residue"]
    real_systemd.record("residue-requeue", {"prepared_residue": prepared["residue"], "events_step": steps[0]})


# --- T-A8: window timings and R-b on a production-sized state ----------------------------------------------

def test_t_a8_window_timings_and_r_b_on_a_production_sized_state(tmp_path, tree, built, systemd):
    """T-A8 and the --timeout-start input: the compiled steps with full-file hashing on a legacy state grown
    to the probe harness's row counts (events, utterances), then a crash and the automatic R-b under real
    systemd with install-launcher's default TimeoutStartSec (240 s, user decision F1=C): the whole start,
    R-b included, must fit in half of it with no help from EXTEND_TIMEOUT_USEC (review U3b-0e4 F2)."""
    w = Window(tmp_path.resolve(), tree, built, systemd, adapters=False)
    w.drop_pull()
    data = w.layout.legacy_state / "data"
    now = 1_800_000_000.0
    text = "texto sintético de una frase ambiental sin datos reales del hogar " * 2
    began = time.monotonic()
    with sqlite3.connect(data / "events.db") as db:
        db.executemany("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,sources_json,"
                       "metrics_json,expires_at) VALUES (?,?,?,?,?,?,?)",
                       ((now - i * 10, "state_change", "light.room" + str(i % 40), "on",
                         _sources("conv" + str(i // 20)), "{}",
                         now + 86400 * 30) for i in range(500_000)))
    db.close()
    with sqlite3.connect(data / "ambient.db") as db:
        db.executemany("INSERT INTO utterances(room_id,t0,t1,text,created_at) VALUES (?,?,?,?,?)",
                       (("room" + str(i % 6), now - i, now - i + 2, text[: 60 + i % 60], now - i)
                        for i in range(250_000)))
    db.close()
    grown = time.monotonic() - began
    common = ("--container-slot", CONTAINER, "--archive", str(built["archive"]), "--slot", "r1",
              "--window-timer", w.watchdog_timer, "--window-timer", w.backup_timer)
    code, report, refused = w.stage("adoption-dry-run", *common, step="dry_run")
    assert code == 0, (report, refused)
    assert report["scratch"] == {"kind": "tmpfs", "removed": True}
    w.window_stop()
    code, prepared, refused = w.stage("adoption-prepare", *common, "--decision", WINDOW, step="prepare")
    assert code == 0, refused
    code, _out, refused = w.stage("install-controller", str(built["archive"]), step="install_controller")
    assert code == 0, refused
    code, _out, refused = w.stage("adoption-commit", "--confirm", prepared["plan_sha256"], pyc=w.controller,
                                  step="commit")
    assert code == 0, (refused, _units_now(w, prepared["plan_sha256"]))
    w.install_launcher()
    w.child("crash", hooks=False, seconds=1.0)
    shown, first_start = w.start_managed()
    assert shown["ActiveState"] in ("active", "failed"), shown
    final = systemd.wait_for(w.voice, lambda s: s["ActiveState"] in ("inactive", "failed"), 120, *STATUS)
    assert final["ExecMainStatus"] == "68" and w.head()["status"] == "unknown", (final, w.managed_console()[-3000:])
    w.child("clean", hooks=False)
    shown, ready = w.start_managed(timeout=900)
    assert shown["ActiveState"] == "active", (shown, w.managed_console()[-4000:])
    events = _launcher_events(w.managed_console())
    budget = [stamp for stamp, name in events if name.startswith("reboot budget")][-1]
    recovered = [(stamp, name) for stamp, name in events if name.startswith("recovery ")][-1]
    assert recovered[1] == "recovery R-b closed", recovered
    started_child = [stamp for stamp, name in events if name == "child started"][-1]
    completed, stop_seconds = w.stop_managed()
    assert completed.returncode == 0 and w.head()["status"] == "closed"
    assert launcher._recoveries(w.root)[-1]["mode"] == "automatic"
    state_bytes = sum(path.lstat().st_size for path in (w.root / "slots" / "r1" / "state").rglob("*")
                      if path.is_file() and not path.is_symlink())
    measured = {"legacy_sizes": report["sizes"], "adopted_state_bytes": state_bytes, "grow_seconds": round(grown, 1),
                "dry_run_report_seconds": report["timings"]["seconds"], "steps": w.timings,
                "first_start_seconds": round(first_start, 2), "timeout_start_seconds": launcher.DEFAULT_TIMEOUT_START,
                "r_b_start_to_ready_seconds": round(ready, 2), "r_b_seconds": round(recovered[0] - budget, 2),
                "recovery_to_child_started_seconds": round(started_child - recovered[0], 2),
                "stop_seconds": round(stop_seconds, 2), "free": report["free"]}
    real_systemd.record("t-a8-production-sized", measured)
    assert ready < launcher.DEFAULT_TIMEOUT_START / 2, measured  # the default covers it without EXTEND


def _launcher_events(console: str) -> list[tuple[float, str]]:
    """(epoch seconds, event) of every [Launcher] line of the managed console log."""
    found = []
    for line in console.splitlines():
        match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) INFO \S+ \[Launcher\] (.*)$", line)
        if match:
            seconds = time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")) + int(match.group(2)) / 1000
            found.append((seconds, match.group(3)))
    return found


# --- production facts (read only) and this host's swap report ------------------------------------------------

def test_production_facts_are_read_only_and_the_swap_report_is_real(tmp_path):
    """The kza user manager's stop budget, /run/user/<kza uid> and the swap the dry run would report."""
    import pwd
    try:
        kza = pwd.getpwnam("kza").pw_uid
    except KeyError:
        kza = None
    budget = launcher.ServiceManager().stop_budget(kza) if kza is not None else None
    mounts = {}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mounts[left.split()[4]] = right.split()[0]
    own = f"/run/user/{os.getuid()}"
    assert commands.scratch_kind(Path(own)) == "tmpfs" and mounts.get(own) == "tmpfs"
    assert commands.scratch_kind(tmp_path) == "other"
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_scratch_not_tmpfs$"):
        commands.checked_scratch(tmp_path, tmpfs_required=True)
    swap = commands.swap_report()
    assert swap["known"] is True
    facts = {"kza_uid": kza, "kza_user_manager_stop_budget_seconds": budget,
             "kza_runtime_dir_fstype": mounts.get(f"/run/user/{kza}") if kza is not None else None,
             "own_runtime_dir_fstype": mounts.get(own), "swap_report": swap,
             "swap_warning_fires": swap["exposure"] != "none",
             "proc_swaps": Path("/proc/swaps").read_text().splitlines()}
    real_systemd.record("production-facts", facts)
    print("production facts", json.dumps(facts, sort_keys=True))
