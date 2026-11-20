"""SC-6b: the release-mode rehearsals, compiled and under this account's real user systemd.

sc-release-design v4 implementation_units SC-6b, items (1)-(17): (1)-(14)
and (16) from SC-6b-A, (15) and (17) from SC-6b-B (rows DR and PA as
implemented by SC-PA-DR; (06-12) also rehearses row PA's regressed source
over an adopted head; the final downtime and disk numbers are
test_linux_sc6b_measurements.py, run alone in a solo window). Every operator step
is the runbook's (docs/runbooks/2026-11-17-release-mode.md, SC-5b) as a
compiled CLI run as a subprocess:

- ``$ST`` is ``controller/current/scripts/stage_runtime.pyc`` (``python -I -B``);
  a T0 runs the extracted NEW's own ``stage_runtime.pyc`` (``tar -xzf`` into a
  new ``mktemp -d``-style directory, after the pinned-sha256 check), and the
  ``T0 / sin T0 / conflicto`` decision is the runbook's own paso-2 snippet,
  read from the runbook file;
- every archive comes from scripts/build_runtime.py: R1's stand-in X4 (this
  tree at controller epoch 4), R2 = X5 (this tree), R2' = X5R2 (revision 2),
  X5R3 (revision 3, the hotfix of (12)), X5C and X5R2C (one controller-closure
  comment: an equal pair with another digest) and the e9669ec legacy artifact
  (L4's slot in (8)); the schema-contract NEWs of (7) are X5/X5R2 with only
  ``release.json``'s ``schema_contract`` rewritten to 2 (what the checks read);
- the stand-in for kza-voice is a real unit of this account's user manager
  (``kza-rehearsal-<token>-voice.service``, test_linux_managed_launcher.
  RealDeployment: compiled install-controller and install-launcher, the
  production managed.conf, Restart=on-failure, RestartPreventExitStatus=69 78);
  only the hidden ``--child-command`` is appended. Starts, stops, enables,
  disables and SIGKILLs are real ``systemctl --user`` calls through the guard
  (tests/mocks/real_systemd.py), which never lets a production unit name
  reach systemctl; every unit is stopped and removed afterwards; (15) kills
  the compiled ``adopt --dry-run`` mid-copy through tests/mocks/compiled_stage.py
  (KZA_REHEARSAL_KILL_AT_STEP=dry-run-mid-copy);
- the child is the rehearsal child (tests/mocks/launcher_child.py, its
  ``--exec`` is a NEW that changes K' at boot, or in (17) an accepted release
  that changes its own state before it dies), except in (14), where it is
  the real compiled application (tools/sc0/sc0_child.py: the real managed
  entry with synthetic leaves, no models, no microphone). Its synthetic
  memory leaf keeps Chroma's default embedding configuration, as
  production's collection does, and supplies the vectors itself
  (tests/mocks/restore_application_graph.SuppliedEmbeddings), so SC-4c's
  isolated Chroma rebuild child rebuilds that store.

Each item writes ``sc6b-<item>.json`` (commands with the paths replaced by
``<root>``/``<tmp>``/``$PY``, exit codes, refusal codes, results, the
per-part times, informative only) and one log per step under
``logs/<item>/`` in KZA_0E4_EVIDENCE. Nothing here touches the kza account,
/home/kza or a production unit. Run it alone, one pytest at a time
(``daemon-reload`` and the time budget; real_systemd's manager lock makes a
concurrent rehearsal wait), under nohup.

Skipped off Linux, without a running user manager, or as root or kza
(real_systemd.module_mark; a collection error
instead under KZA_REQUIRE_REAL_SYSTEMD=1).
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time

import pytest
import yaml

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks.runtime_artifact_cache import cached_build_runtime as build_runtime
from tests.integration import test_linux_managed_launcher as real
from tests.integration.test_linux_adoption_window import _rewrite_release
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks import real_systemd
from tests.mocks import sc4b_states
from tests.mocks.epoch4 import edited_commit, epoch4_commit, epoch4_in_process
from tests.unit.tools.test_sc1b_release_plan import _QUEUE as _QUEUE_IN_PROCESS, _k_retry

pytestmark = real_systemd.module_mark()

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "docs" / "runbooks" / "2026-11-17-release-mode.md"
WRAPPER = REPO / "tests" / "mocks" / "compiled_stage.py"
SC0_CHILD = REPO / "tools" / "sc0" / "sc0_child.py"
SMOKE = 10  # N of the first transition (runbook: N = 10 with T0)
REVISION_1 = b"\nCONTROLLER_REVISION = 1\n"
FUTURE = b"from __future__ import annotations\n"
OWNER_COMPACTION = "src/memory/owner_compaction.py"
LEGACY = launcher.LEGACY_CONTAINER_COMMIT  # R1E: the e9669ec container production serves
STATUS = (*real.STATUS, "InvocationID")
RESTART_SECONDS = 10.0  # managed.conf RestartSec
CLEANUPS: list[dict] = []
OFFLINE_ENV = ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
               "ANONYMIZED_TELEMETRY=False", "TOKENIZERS_PARALLELISM=false", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1")


# --- artifacts --------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """Every NEW of the rehearsals, built by scripts/build_runtime.py in a shared clone."""
    base = tmp_path_factory.mktemp("sc6b-builds")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    x5 = rehearsal.overlay_commit(REPO, clone, message="sc-6b R2 (X5)")
    stage_py = "scripts/stage_runtime.py"
    closure = {OWNER_COMPACTION: [(FUTURE, FUTURE + b"# sc-6b: a controller-closure change\n")]}
    revisions = {
        "x5": x5,
        "x4": epoch4_commit(clone, x5),
        "x5r2": edited_commit(clone, x5, {stage_py: [(REVISION_1, b"\nCONTROLLER_REVISION = 2\n")]},
                              message="sc-6b R2' (revision 2)"),
        "x5r3": edited_commit(clone, x5, {stage_py: [(REVISION_1, b"\nCONTROLLER_REVISION = 3\n")]},
                              message="sc-6b hotfix (revision 3)"),
        "x5c": edited_commit(clone, x5, closure, message="sc-6b revision 1, other closure"),
        "x5r2c": edited_commit(clone, x5, {stage_py: [(REVISION_1, b"\nCONTROLLER_REVISION = 2\n")], **closure},
                               message="sc-6b revision 2, other closure"),
        "legacy": LEGACY,
    }
    found = {}
    for name, revision in revisions.items():
        archive = base / (name + ".tar.gz")
        started = time.monotonic()
        manifest = build_runtime(clone, archive, revision)
        found[name] = {"archive": archive, "manifest": manifest, "seconds": round(time.monotonic() - started, 1)}
    for name, source in (("x5schema", "x5"), ("x5r2schema", "x5r2")):
        archive = base / (name + ".tar.gz")
        _rewrite_release(found[source]["archive"], archive, schema_contract=2)
        found[name] = {"archive": archive, "manifest": stage._read_archive(archive)[0], "seconds": 0.0}
    for name, value in found.items():
        value["sha256"] = _sha256(value["archive"])
        manifest = value["manifest"]
        value["triple"] = list(stage.controller_triple(manifest)) if manifest.get("controller_epoch", 0) >= 5 \
            else [manifest.get("controller_epoch"), manifest.get("controller_revision", 0),
                  manifest.get("controller_sources_sha256")]
    assert found["x4"]["manifest"]["controller_epoch"] == 4 and found["x5"]["manifest"]["controller_epoch"] == 5
    assert found["x5r2"]["triple"][:2] == [5, 2] and found["x5r3"]["triple"][:2] == [5, 3]
    assert found["x5c"]["triple"][:2] == [5, 1] and found["x5c"]["triple"][2] != found["x5"]["triple"][2]
    assert found["x5r2c"]["triple"][:2] == [5, 2] and found["x5r2c"]["triple"][2] != found["x5r2"]["triple"][2]
    assert found["x5schema"]["manifest"]["schema_contract"] == 2
    assert found["legacy"]["manifest"]["git_commit"].startswith(LEGACY)
    real_systemd.record("sc6b-builds", {name: {"commit": value["manifest"]["git_commit"], "sha256": value["sha256"],
                                               "controller_triple": value["triple"],
                                               "schema_contract": value["manifest"].get("schema_contract"),
                                               "build_seconds": value["seconds"]}
                                        for name, value in found.items()})
    return found


@pytest.fixture(scope="module", autouse=True)
def cleanup_evidence():
    # A killed earlier run (SIGKILL, OOM, dropped ssh) skipped its finalizers: sweep its units first.
    real_systemd.record("sc6b-stale-sweep", real_systemd.sweep_stale())
    yield
    real_systemd.record("sc6b-unit-cleanup", CLEANUPS)


@pytest.fixture
def systemd(tmp_path):
    manager = real_systemd.UserSystemd(tmp_path / "systemd")
    yield manager
    result = manager.cleanup()
    CLEANUPS.append({key: value for key, value in result.items() if key not in ("before", "after")})
    assert result["rehearsal_units_after"] == [] and result["leftover_paths"] == []
    assert result["guard_breaches"] == [] and result["daemon_reload_exit"] == 0


def _runbook_block(start: str, end: str) -> str:
    text = RUNBOOK.read_text()
    return text.split(start, 1)[1].split(end, 1)[0]


TRIPLE_CHECK = _runbook_block("<<'PYEOF'\n", "\nPYEOF")  # paso 2, verbatim


def _last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- the record of one item -------------------------------------------------------------------------

class Recorder:
    """Commands, exit codes, refusal codes, results and per-part times of one item (content-free)."""

    def __init__(self, item: str, title: str, tmp_path: Path) -> None:
        self.item, self.title = item, title
        self.steps: list[dict] = []
        self.results: dict = {}
        self.times: dict = {}
        evidence = real_systemd.evidence_directory()
        # Without an evidence directory (the batch suite), keep the step logs out of tmp_path/logs:
        # RealDeployment creates that directory for the unit's console log (batch F suite collision).
        self.logs = (evidence / "logs" if evidence is not None else tmp_path / "rehearsal-logs") / item
        self.logs.mkdir(parents=True, exist_ok=True)
        self.pairs: list[tuple[str, str]] = [(sys.executable, "$PY"), (str(REPO) + "/", "<repo>/"),
                                             (str(tmp_path), "<tmp>")]

    def label(self, path, name: str) -> None:
        self.pairs.append((str(path), name))
        self.pairs.sort(key=lambda pair: -len(pair[0]))

    def scrub(self, text: str) -> str:
        for value, name in self.pairs:
            text = text.replace(value, name)
        return text

    def log(self, name: str, text: str) -> str:
        path = self.logs / f"{len(self.steps):03d}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', name)}.log"
        path.write_text(self.scrub(text))
        return path.name

    def step(self, name: str, argv: list[str], code: int | None, stdout: str, stderr: str, seconds: float,
             **extra) -> dict:
        parsed = _last_json(stdout) or _last_json(stderr) or {}
        entry = {"step": name, "command": [self.scrub(str(token)) for token in argv], "exit": code,
                 "seconds": round(seconds, 3)}
        if isinstance(parsed.get("refused"), str):
            entry["refused"] = parsed["refused"]
        entry.update(extra)
        entry["log"] = self.log(name, "command: " + " ".join(map(str, argv)) + f"\nexit: {code}\n--- stdout\n"
                                + stdout + "\n--- stderr\n" + stderr)
        self.steps.append(entry)
        return entry

    def time(self, name: str, seconds: float) -> None:
        key, count = name, 1
        while key in self.times:
            count += 1
            key = f"{name}#{count}"
        self.times[key] = round(seconds, 3)

    def result(self, **values) -> None:
        self.results.update(values)

    def save(self, outcome: str) -> None:
        real_systemd.record("sc6b-" + self.item, {"item": self.item, "title": self.title, "outcome": outcome,
                                                 "steps": self.steps, "results": self.results,
                                                 "times_informative": self.times})


@pytest.fixture
def record(request, tmp_path):
    holder: dict = {}
    failed_before = request.session.testsfailed

    def make(item: str, title: str) -> Recorder:
        holder["recorder"] = Recorder(item, title, tmp_path)
        return holder["recorder"]

    yield make
    recorder = holder.get("recorder")
    if recorder is not None:  # the call phase's report is logged before this teardown runs
        recorder.save("failed" if request.session.testsfailed > failed_before else "passed")


# --- one production-shaped world --------------------------------------------------------------------

class World:
    """R1's stand-in root (X4, epoch 4) with its real stand-in unit; H closed by R1's own first run."""

    def __init__(self, builds: dict, tmp_path: Path, systemd: real_systemd.UserSystemd, rec: Recorder, *,
                 first: str | None = cw.LEDGER, source: Path | None = None, first_child=None) -> None:
        self.builds, self.tmp, self.systemd, self.rec = builds, tmp_path, systemd, rec
        started = time.monotonic()
        with epoch4_in_process():  # the bootstrap head, as U3b-0e's adoption leaves it (in process)
            self.d = real.RealDeployment(builds["x4"]["archive"], tmp_path, systemd, source=source)
        self.root = self.d.root
        rec.label(self.root, "<root>")
        rec.label(builds["x5"]["archive"].parent, "<builds>")
        self.controller = self.root / "controller" / "current" / "scripts" / "stage_runtime.pyc"
        self.unit = "kza-voice.service"  # the guard maps it to this rehearsal's stand-in
        self.k = 0
        self.extra_env: list[str] = []
        rec.time("world_setup", time.monotonic() - started)
        assert json.loads((self.root / "admission" / "controller.json").read_text())["controller_epoch"] == 4
        if first_child is not None:
            first_child(self)
        else:
            self.serve_clean("h-first-run-r1", execute=first)
        self.h_binding = self.head()["binding_id"]

    # -- reading ----------------------------------------------------------------------------------

    def head(self) -> dict:
        return stage.read_admission_head(self.root)

    def slot_of_head(self) -> str:
        return stage.read_binding(self.root, self.head()["binding_id"])["slot"]

    def witness(self) -> dict:
        return cw.witness(self.root)

    def log(self, compaction_id: str) -> dict:
        return stage.read_compaction_log(self.root, compaction_id)

    def kinds(self, compaction_id: str) -> list[str]:
        return [event["kind"] for event in self.log(compaction_id)["events"]]

    # -- commands ---------------------------------------------------------------------------------

    def run(self, name: str, argv: list[str], *, expect: int | None = 0, env: dict | None = None,
            timeout: float = 1200, stdin: str = "") -> tuple[int, dict, subprocess.CompletedProcess]:
        started = time.monotonic()
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=self.tmp,
                                   env=env if env is not None else self.systemd.environment(), input=stdin)
        seconds = time.monotonic() - started
        self.rec.step(name, argv, completed.returncode, completed.stdout, completed.stderr, seconds)
        parsed = _last_json(completed.stdout) or _last_json(completed.stderr) or {}
        if expect is not None:
            assert completed.returncode == expect, (name, completed.returncode, completed.stdout[-3000:],
                                                    completed.stderr[-3000:])
        return completed.returncode, parsed, completed

    def st(self, name: str, *arguments: str, expect: int | None = 0, pyc: Path | None = None,
           env: dict | None = None) -> dict:
        """``$ST ...``: the installed controller, as the runbook defines $ST."""
        argv = [sys.executable, "-I", "-B", str(pyc or self.controller), "--root", str(self.root), "--offline",
                *arguments]
        return self.run(name, argv, expect=expect, env=env)[1]

    def refused(self, name: str, code: str, *arguments: str, pyc: Path | None = None) -> dict:
        """A refusal with ``code`` (exact, or ``code:<detail>``) that writes nothing under the root."""
        witness = self.witness()
        parsed = self.st(name, *arguments, expect=78, pyc=pyc)
        assert parsed.get("refused") == code or str(parsed.get("refused", "")).startswith(code + ":"), parsed
        assert self.witness() == witness, name
        return parsed

    def ctl(self, name: str, *arguments: str, expect: int | None = 0) -> subprocess.CompletedProcess:
        started = time.monotonic()
        completed = self.systemd.systemctl(*arguments, timeout=300)
        self.rec.step(name, ["systemctl", "--user", *arguments], completed.returncode, completed.stdout,
                      completed.stderr, time.monotonic() - started)
        if expect is not None:
            assert completed.returncode == expect, (name, completed.stderr)
        return completed

    def show(self) -> dict:
        return self.systemd.raw_show(self.d.unit, *STATUS)

    def console(self) -> str:
        return self.d.console_text()

    # -- the unit's child (zz-rehearsal.conf) -------------------------------------------------------

    def child(self, scenario: str, *, execute: str | None = None, restart: bool = True) -> None:
        script = None
        if execute is not None:
            script = self.tmp / f"exec-{len(list(self.tmp.glob('exec-*')))}.py"
            script.write_text(execute)
        if self.d.phase_file.exists():
            self.d.phase_file.unlink()
        command = rehearsal.child_command(self.d.phase_file, scenario, execute=script)
        self.d.child(command, restart=restart, extra=self._environment_lines())

    def app_child(self, *, restart: bool = True, flags: tuple[str, ...] = ()) -> None:
        """(14): the real compiled application (tools/sc0/sc0_child.py) on the slot's compiled modules."""
        if self.d.phase_file.exists():
            self.d.phase_file.unlink()
        self.trace = self.tmp / f"sql-trace-{len(list(self.tmp.glob('sql-trace-*')))}.jsonl"
        command = [sys.executable, "-I", "-B", str(SC0_CHILD), "--repo", str(REPO), "--phase-file",
                   str(self.d.phase_file), "--graph-root", "data/graph", "--dashboard-port", str(self.app_port),
                   "--trace-file", str(self.trace), "--ambient", "real",
                   # The command index at its production store (vectordb.path ./data/chroma_db): SC-4c's
                   # closed list names stores by that path; the synthetic graph would put it at data/graph/index.
                   "--state-store", "vectordb.path=data/chroma_db", *flags]
        self.d.child(command, restart=restart, extra=self._environment_lines())

    def _environment_lines(self) -> str:
        return "".join(f"Environment={item}\n" for item in self.extra_env)

    def phase(self, name: str):
        return self.d.phase(name)

    def begin(self, name: str) -> float:
        """``systemctl --user start --no-block``, then wait until systemd has a new invocation of the unit."""
        previous = self.show()["InvocationID"]
        started = time.monotonic()
        self.ctl(name, "start", "--no-block", self.unit)
        shown = self.systemd.wait_for(self.d.unit, lambda value: value["InvocationID"] not in ("", previous), 60,
                                      *STATUS)
        assert shown["InvocationID"] not in ("", previous), (name, shown)
        return started

    # -- unit operations ----------------------------------------------------------------------------

    def wait_active(self, timeout: float = 240) -> dict:
        return self.systemd.wait_for(self.d.unit, lambda shown: shown["ActiveState"] in ("active", "failed")
                                     or (shown["ActiveState"] == "inactive" and shown["Result"] != "success"),
                                     timeout, *STATUS)

    def wait_settled(self, timeout: float) -> dict:
        """Inactive or failed and still so after RestartSec (a production restart would have begun)."""
        deadline = time.monotonic() + timeout
        while True:
            shown = self.systemd.wait_for(self.d.unit, lambda value: value["ActiveState"] in ("inactive", "failed"),
                                          max(1.0, deadline - time.monotonic()), *STATUS)
            if shown["ActiveState"] not in ("inactive", "failed"):
                return shown
            time.sleep(RESTART_SECONDS + 3)
            again = self.show()
            if again["ActiveState"] in ("inactive", "failed") and again["NRestarts"] == shown["NRestarts"]:
                return again
            if time.monotonic() >= deadline:
                return again

    def serve_clean(self, name: str, *, execute: str | None = None, enable: bool = False,
                    keep_enabled: bool = False, app: bool = False) -> dict:
        """Start the unit, wait for READY (active), stop it: one closed run of the head."""
        if app:
            self.app_child()
        else:
            self.child("clean", execute=execute)
        if enable:
            self.ctl(name + "-enable", "enable", self.unit)
        started = self.begin(name + "-start")
        shown = self.wait_active(300 if app else 180)
        ready = time.monotonic() - started
        assert shown["ActiveState"] == "active", (name, shown, self.console()[-4000:])
        stopped = time.monotonic()
        self.ctl(name + "-stop", "stop", self.unit)
        after = self.d.wait_inactive(120)
        stop_seconds = time.monotonic() - stopped
        assert after["Result"] == "success", (name, after, self.console()[-4000:])
        if enable and not keep_enabled:
            self.ctl(name + "-disable", "disable", self.unit)
        assert self.head()["status"] == "closed", name
        self.rec.time(name + ".start_to_ready", ready)
        self.rec.time(name + ".stop", stop_seconds)
        return {"start_to_ready": ready, "stop": stop_seconds}

    def stop_and_disable(self, name: str = "paso1") -> None:
        """Paso 1: stop and disable (exit 0 each)."""
        self.ctl(name + "-stop", "stop", self.unit)
        self.ctl(name + "-disable", "disable", self.unit)

    def kill(self, name: str) -> None:
        self.ctl(name, "kill", "--signal=SIGKILL", self.unit)

    # -- the runbook's blocks -----------------------------------------------------------------------

    def extract(self, build: str) -> Path:
        """'Verificar y extraer NUEVO': the pinned sha256, then tar -xzf into a new temporary directory."""
        archive = self.builds[build]["archive"]
        assert _sha256(archive) == self.builds[build]["sha256"], "NUEVO distinto: DETENER"
        new = Path(__import__("tempfile").mkdtemp(prefix="new-", dir=self.tmp))
        self.run("extract-" + build, ["tar", "-xzf", str(archive), "-C", str(new)])
        manifest = json.loads((new / "release.json").read_text())
        assert manifest["git_commit"] == self.builds[build]["manifest"]["git_commit"]
        self.rec.label(new, "$NEW")
        return new

    def triple_check(self, new: Path) -> str:
        """Paso 2, the runbook's own snippet: 'T0', 'sin T0' or 'conflicto: no seguir'."""
        installed = self.root / "controller" / "current" / "release.json"
        _code, _parsed, completed = self.run("paso2-triple", [sys.executable, "-I", "-", str(installed),
                                                              str(new / "release.json")],
                                             env=dict(os.environ), stdin=TRIPLE_CHECK)
        return completed.stdout.strip().splitlines()[-1]

    def t0(self, build: str, *, expect: int | None = 0, kill_at: str | None = None,
           new: Path | None = None) -> dict:
        """Paso 3: install-controller from the extracted NEW, never from $ST."""
        new = new or self.extract(build)
        pyc = new / "scripts" / "stage_runtime.pyc"
        archive = str(self.builds[build]["archive"])
        if kill_at is None:
            argv = [sys.executable, "-I", "-B", str(pyc), "--root", str(self.root), "--offline",
                    "install-controller", archive]
            env = None
        else:  # the rehearsal wrapper SIGKILLs the compiled installer at that step (rows T0, T0S)
            argv = [sys.executable, "-I", "-B", str(WRAPPER), str(pyc), "--root", str(self.root), "--offline",
                    "install-controller", archive]
            env = self.systemd.environment({"KZA_REHEARSAL_KILL_AT_STEP": kill_at})
        started = time.monotonic()
        code, parsed, _completed = self.run(f"t0-{build}" + (f"-killed-at-{kill_at}" if kill_at else ""), argv,
                                            expect=expect, env=env)
        self.rec.time(f"t0-{build}" + (f"-{kill_at}" if kill_at else ""), time.monotonic() - started)
        parsed["exit"] = code
        return parsed

    def h_check(self, name: str = "t0-h-check") -> dict:
        """'Comprobar que H sirve con el controlador nuevo, una vez', then stop and disable again."""
        return self.serve_clean(name, enable=True)

    # -- one transition -----------------------------------------------------------------------------

    def plan(self, build: str, k: str, *, expect: int = 0, code: str | None = None,
             accept: str | None = None) -> dict:
        """Paso 4; ``accept`` is --accept-regressed-source's prefix (row PA, 'Después de la aceptación')."""
        arguments = ("compact", "plan", k, "--archive", str(self.builds[build]["archive"]),
                     *(("--accept-regressed-source", accept) if accept is not None else ()))
        if expect == 78:
            return self.refused(f"plan-{k}-{build}", code, *arguments)
        started = time.monotonic()
        planned = self.st(f"plan-{k}-{build}", *arguments)
        self.rec.time(f"plan-{k}", time.monotonic() - started)
        assert planned["plan_sha256"] and planned["untouched_sha256"], planned
        return planned

    def start_arguments(self, build: str, k: str, planned: dict, accept: str | None = None) -> tuple[str, ...]:
        return ("compact", "start", k, "--archive", str(self.builds[build]["archive"]), "--plan-sha256",
                planned["plan_sha256"], "--acknowledge-untouched", planned["untouched_sha256"],
                "--destroy-rollback-states", *(("--accept-regressed-source", accept) if accept is not None else ()))

    def start(self, build: str, k: str, planned: dict, *, accept: str | None = None) -> str:
        h_binding = self.head()["binding_id"]
        started = time.monotonic()
        result = self.st(f"start-{k}-{build}", *self.start_arguments(build, k, planned, accept))
        self.rec.time(f"start-{k}", time.monotonic() - started)
        compaction_id = result["compaction_id"]
        assert result["next"] == "compact verify " + compaction_id
        intent = self.log(compaction_id)["intent"]
        assert intent["mode"] == "release" and intent["retire_prepare_leftovers"] is True, intent
        intent = stage.compaction_log_path(self.root, compaction_id) / "intent.json"
        assert intent == self.root / "admission" / "compaction" / "v1" / compaction_id / "intent.json"
        _code, _parsed, completed = self.run(  # paso 5: the runbook's own read of H's binding
            "paso5-h-binding", [sys.executable, "-I", "-c",
                                'import json,sys; print(json.load(open(sys.argv[1]))["source_binding_id"])',
                                str(intent)], env=dict(os.environ))
        self.confirm_h = completed.stdout.strip()
        assert self.confirm_h == h_binding
        return compaction_id

    def verify(self, compaction_id: str, n: int = SMOKE) -> dict:
        armed = self.st("verify", "compact", "verify", compaction_id, "--smoke-seconds", str(n))
        assert armed["armed"] is True
        return armed

    def v_run(self, name: str, *, scenario: str = "verify", execute: str | None = None, restart: bool = True,
              app: bool = False, kill_after_ready: bool = False, after_kill=None, timeout: float = 400) -> dict:
        """Paso 6: start the (disabled) unit once; serve runs V and stops it itself.

        ``after_kill()`` runs right after the SIGKILL (kill_after_ready), before waiting for the unit to settle.
        """
        if app:
            self.app_child(restart=restart)
        else:
            self.child(scenario, execute=execute, restart=restart)
        started = self.begin(name + "-start")
        if kill_after_ready:
            reached = self.d.wait_phase("ready", 180)
            assert reached, (name, self.console()[-3000:])
            self.kill(name + "-sigkill")
            if after_kill is not None:
                after_kill()
        shown = self.wait_settled(timeout)
        seconds = time.monotonic() - started
        assert shown["ActiveState"] in ("inactive", "failed"), (name, shown, self.console()[-4000:])
        self.rec.time(name + ".start_to_inactive", seconds)
        self.rec.result(**{name + ".unit": {key: shown[key] for key in ("ActiveState", "Result", "ExecMainStatus",
                                                                       "NRestarts")}})
        if shown["ActiveState"] == "failed":
            self.ctl(name + "-reset-failed", "reset-failed", self.unit)
        return shown

    def status(self, compaction_id: str) -> dict:
        witness = self.witness()
        [found] = self.st("paso7-status", "compact", "status", compaction_id)["compactions"]
        assert self.witness() == witness  # status never writes
        return found

    def retire(self, compaction_id: str) -> dict:
        started = time.monotonic()
        retired = self.st("paso8-retire", "compact", "retire", compaction_id)
        self.rec.time("retire", time.monotonic() - started)
        assert retired["phase"] == "complete", retired
        return retired

    def abort_to_h(self, compaction_id: str, *, discard: str | None = None, expect: int = 0,
                   code: str | None = None) -> dict:
        arguments = ["compact", "abort", compaction_id, "--to-h", "--confirm-h", self.confirm_h[:12]]
        if discard is not None:
            arguments += ["--discard-k-changes", discard]
        if expect == 78:
            return self.refused("abort-to-h-refused", code, *arguments)
        started = time.monotonic()
        aborted = self.st("abort-to-h", *arguments)
        self.rec.time("abort_to_h", time.monotonic() - started)
        assert aborted["phase"] == "aborted", aborted
        return aborted

    def back_on_h(self, compaction_id: str, h_binding: str, name: str = "h-after-abort") -> None:
        """After the return to H: H's head, K gone, the log terminal; enable and start: H serves."""
        head = self.head()
        assert head["status"] == "closed" and head["binding_id"] == h_binding
        assert self.log(compaction_id)["terminal"] and self.kinds(compaction_id)[-1] == "aborted"
        assert not any((self.root / "slots").glob("k*"))
        self.serve_clean(name, enable=True)

    def transition(self, build: str, *, name: str, scenario: str = "verify", execute: str | None = None,
                   app: bool = False, h_binding: str | None = None) -> dict:
        """Pasos 0-9 of one ordinary transition that is accepted (no T0 here)."""
        self.k += 1
        k = f"k{self.k}"
        self.st(name + "-paso0-completion-status", "completion-status")
        self.stop_and_disable(name + "-paso1")
        down = time.monotonic()
        planned = self.plan(build, k)
        compaction_id = self.start(build, k, planned)
        self.verify(compaction_id)
        self.v_run(name + "-v", scenario=scenario, execute=execute, app=app)
        status = self.status(compaction_id)
        assert status["verified"] is True and status["armed_run"] == "consumed", status
        assert status["verify_branch_reached"] == "yes", status
        retired = self.retire(compaction_id)
        [started] = [event for event in self.log(compaction_id)["events"] if event["kind"] == "retire_started"]
        assert started["data"]["w"]["class"] == compaction.W_EQUAL
        assert self.slot_of_head() == k
        served = self.serve_clean(name + "-paso9-serve-k-prime", enable=True, keep_enabled=True, app=app)
        self.rec.time(name + ".stop_to_ready_of_k_prime_excluding_paso0", time.monotonic() - down
                      - served["stop"])
        self.rec.result(**{name: {"k": k, "compaction_id": compaction_id, "build": build,
                                  "plan_chroma": planned.get("chroma"), "coverage": planned.get("coverage"),
                                  "destroyed": retired.get("destroyed"), "w_class": started["data"]["w"]["class"]}})
        return {"k": k, "id": compaction_id, "planned": planned}

    def armed(self, build: str, name: str) -> str:
        """Pasos 0-6 up to the armed V (paso 1 included)."""
        self.k += 1
        k = f"k{self.k}"
        self.stop_and_disable(name + "-paso1")
        planned = self.plan(build, k)
        compaction_id = self.start(build, k, planned)
        self.verify(compaction_id)
        return compaction_id


def _first_t0(world: World, build: str = "x5") -> None:
    """The first transition's T0 (R1 -> epoch 5) with the runbook's decision and H check."""
    world.stop_and_disable("t0-paso1")
    new = world.extract(build)
    assert world.triple_check(new) == "T0"
    installed = world.t0(build, new=new)
    assert installed["controller_epoch"] == 5
    assert list(stage.installed_controller_triple(world.root)) == world.builds[build]["triple"]
    world.h_check()


def _sql(path: Path, query: str) -> list:
    with closing(sqlite3.connect(path)) as db:
        return db.execute(query).fetchall()


H_TOMBSTONE = cw.sql(("INSERT INTO conversation_tombstones VALUES ('conv-h', 1.0, 'owner')", ()),
                     ("INSERT INTO conversation_erasures VALUES ('op-h', 'conv-h', 1.0, 'owner')", ()))
DELETE_TOMBSTONES = cw.sql(("DELETE FROM conversation_tombstones", ()))
PA_TOMBSTONE = cw.sql(("INSERT INTO conversation_tombstones VALUES ('conv-pa', 2.0, 'owner')", ()))
H_CREDENTIAL = "\n".join([
    "import sqlite3",
    "from contextlib import closing",
    f"with closing(sqlite3.connect(state / 'data' / 'auth.db')) as db, db:",
    f"    db.execute({sc4b_states.ddl('credentials')!r})",
    "    db.execute('INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)', ('cred-sc6b', 'd' * 64, 'home', 'home',",
    "               'user-sc6b', 1800000000.0, 4102444800.0, 1, '[]'))",
])
REVOKE_CREDENTIAL = "\n".join([
    "import sqlite3",
    "from contextlib import closing",
    "with closing(sqlite3.connect(state / 'data' / 'auth.db')) as db, db:",
    "    db.execute('DELETE FROM credentials')",
])
BOOT_FAILURE = "import os\nos._exit(3)\n"
# SC-1b's queued-deletion planting, run on its own thread: the rehearsal child's --exec runs inside its event loop.
_QUEUE = "\n".join([
    "import threading",
    "_failed = []",
    "def _plant():",
    "    try:",
    *["        " + line for line in _QUEUE_IN_PROCESS.splitlines()],
    "    except BaseException as error:",
    "        _failed.append(error)",
    "_worker = threading.Thread(target=_plant)",
    "_worker.start()",
    "_worker.join()",
    "if _failed:",
    "    raise _failed[0]",
])  # NEW dies at boot, before READY and before its receipt


# --- (1) T0 from the extracted archive, serve R1, R1 -> R2 --------------------------------------------

def test_01_first_transition_t0_then_r1_to_r2(builds, tmp_path, systemd, record):
    rec = record("01", "T0 R2 controller from the extracted archive, serve R1, stop and disable, R1 -> R2")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    times = dict(rec.times)
    rec.result(t0_h_boot_to_ready=times["t0-h-check.start_to_ready"], t0_h_stop=times["t0-h-check.stop"])
    world.transition("x5", name="r1-to-r2")
    assert world.slot_of_head() == "k1"
    assert sorted(os.listdir(world.root / "slots")) == ["k1"]
    served = compaction.served_releases(world.root, stage.read_binding(world.root, world.head()["binding_id"]))
    x5_release = compaction._sha256_bytes(compaction._release_json_bytes(builds["x5"]["manifest"]))
    rec.result(x5_in_served=x5_release in served)
    assert x5_release in served


# --- (2) V boot failure, lossless W abort ----------------------------------------------------------------

def test_02_v_boot_failure_is_aborted_losslessly(builds, tmp_path, systemd, record):
    rec = record("02", "V boot failure, lossless W abort")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-boot-failure")
    shown = world.v_run("v-boot-failure", execute=BOOT_FAILURE)  # production Restart=on-failure
    status = world.status(compaction_id)
    rec.result(status={key: status[key] for key in ("armed_run", "verified", "verify_branch_reached")},
               head_after_v=world.head()["status"], restarts=shown["NRestarts"],
               serve_events=[line.split("[Launcher] ", 1)[1] for line in world.console().splitlines()
                             if "[Launcher] " in line and ("refused" in line or "recovery" in line
                                                            or "exit" in line)][-12:])
    assert status["armed_run"] == "consumed" and status["verified"] is False
    # The NEW dies at boot: serve exits 69 (RestartPreventExitStatus: no restart) and leaves K' open.
    assert shown["ExecMainStatus"] == "69" and shown["NRestarts"] == "0", shown
    retired = world.refused("paso8-retire-refused", "runtime_admission_not_closed", "compact", "retire",
                            compaction_id)
    rec.result(retire_refused=retired["refused"])
    aborted = world.abort_to_h(compaction_id)
    [started] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "abort_started"]
    rec.result(w_class=started["data"]["w"]["class"], abort=aborted.get("phase"))
    assert started["data"]["w"]["class"] == compaction.W_EQUAL
    world.back_on_h(compaction_id, h_binding)


# --- (3) SIGKILL of V without receipt, abort from the open head ------------------------------------------

def test_03_v_sigkilled_without_receipt_is_aborted_from_the_open_head(builds, tmp_path, systemd, record):
    """The whole cgroup SIGKILLed after READY with Restart=no: K' stays open (no receipt); abort --to-h
    recovers it in place (W 'equal') and returns to H."""
    rec = record("03", "SIGKILL of V without receipt, abort from the open head")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-sigkill")
    world.v_run("v-sigkill", restart=False, kill_after_ready=True)
    head = world.head()
    rec.result(head_after_kill=head["status"])
    assert head["status"] != "closed"
    status = world.status(compaction_id)
    rec.result(status={key: status[key] for key in ("armed_run", "verified", "verify_branch_reached")})
    assert status["armed_run"] == "consumed" and status["verified"] is False
    world.refused("paso8-retire-refused", "runtime_admission_not_closed", "compact", "retire", compaction_id)
    world.abort_to_h(compaction_id)
    [started] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "abort_started"]
    rec.result(w_class=started["data"]["w"]["class"])
    assert started["data"]["w"]["class"] == compaction.W_EQUAL
    world.back_on_h(compaction_id, h_binding)


def test_03b_v_sigkilled_under_the_production_restart(builds, tmp_path, systemd, record):
    """Same kill with managed.conf's Restart=on-failure: systemd restarts serve after RestartSec; what the
    restart does over the open K' is recorded, then the return to H (with the AB recheck if it asks)."""
    rec = record("03b", "SIGKILL of V without receipt under Restart=on-failure, then the return to H")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-sigkill-restart")
    k_prime = world.root / "slots" / f"k{world.k}"
    seen: dict = {}

    def before_the_restart() -> None:
        """K' as the kill left it, read during RestartSec (SubState auto-restart, no restart yet)."""
        shown = world.systemd.wait_for(world.d.unit, lambda value: value["SubState"] == "auto-restart"
                                       or value["NRestarts"] not in ("", "0"), 60, *STATUS)
        assert shown["SubState"] == "auto-restart" and shown["NRestarts"] == "0", shown
        seen["k_prime"] = cw.tree(k_prime)
        again = world.show()
        assert again["SubState"] == "auto-restart" and again["NRestarts"] == "0", ("restart began", again)

    shown = world.v_run("v-sigkill-restart", restart=True, kill_after_ready=True, after_kill=before_the_restart)
    status = world.status(compaction_id)
    rec.result(head_after_restart=world.head()["status"], restarts=shown["NRestarts"],
               status={key: status[key] for key in ("armed_run", "verified", "verify_branch_reached")},
               serve_events=[line.split("[Launcher] ", 1)[1] for line in world.console().splitlines()
                             if "[Launcher] " in line][-20:])
    # The one production restart refuses over the open K' (recovery_pending, exit 69, no second restart)
    # and leaves K' exactly as the kill did.
    assert shown["ExecMainStatus"] == "69" and shown["NRestarts"] == "1", shown
    assert any("[Launcher] recovery R-b runtime_admission_compaction_recovery_pending" in line
               for line in world.console().splitlines())
    k_prime_unchanged = cw.tree(k_prime) == seen["k_prime"]
    rec.result(k_prime_unchanged_by_the_restart=k_prime_unchanged)
    assert k_prime_unchanged
    parsed = world.st("abort-to-h-first", "compact", "abort", compaction_id, "--to-h", "--confirm-h",
                      world.confirm_h[:12], expect=None)
    rec.result(first_abort=parsed.get("refused") or parsed.get("phase"))
    if parsed.get("refused") == "runtime_admission_compaction_abort_recheck_required":
        world.abort_to_h(compaction_id)  # row AB: the same command again redoes W0 and W1
    else:
        assert parsed.get("phase") == "aborted", parsed
    world.back_on_h(compaction_id, h_binding)


# --- (4) NEW removes a tombstone at boot ---------------------------------------------------------------

def test_04_new_removing_a_tombstone_is_never_retired_and_abort_returns_to_h(builds, tmp_path, systemd, record):
    rec = record("04", "NEW removes a tombstone at boot: retire refused, abort succeeds")
    world = World(builds, tmp_path, systemd, rec, first=cw.LEDGER + "\n" + H_TOMBSTONE)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-tombstone")
    world.v_run("v-tombstone", execute=DELETE_TOMBSTONES)
    status = world.status(compaction_id)
    assert status["verified"] is True  # V itself was a good run: only W refuses it
    refused = world.refused("paso8-retire-refused", compaction.RETIRE_PROTECTED_CHANGED, "compact", "retire",
                            compaction_id)
    report = refused["report"]["w"]
    rec.result(retire_refused=refused["refused"], w_class=report["class"],
               w_tables=report["tables"].get("data/privacy.db"))
    assert report["class"] == compaction.W_LOSS_ONLY
    assert report["tables"]["data/privacy.db"]["conversation_tombstones"] == {"added": 0, "removed": 1}
    world.abort_to_h(compaction_id)
    world.back_on_h(compaction_id, h_binding)
    state = world.root / "slots" / world.slot_of_head() / "state"
    assert _sql(state / "data" / "privacy.db", "SELECT conversation_id FROM conversation_tombstones") == [("conv-h",)]


# --- (5) NEW revokes a credential at boot: SS, then the SC-D8 A exit -------------------------------------

def test_05_new_revoking_a_credential_stops_safely_until_the_discard(builds, tmp_path, systemd, record):
    rec = record("05", "NEW revokes a credential at boot: safe stop, then the SC-D8 A exit")
    world = World(builds, tmp_path, systemd, rec, first=cw.LEDGER + "\n" + H_CREDENTIAL)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-revoke")
    world.v_run("v-revoke", execute=REVOKE_CREDENTIAL)
    retired = world.refused("paso8-retire-refused", compaction.RETIRE_PROTECTED_CHANGED, "compact", "retire",
                            compaction_id)
    first = world.refused("ss-abort-refused-1", compaction.ABORT_PROTECTED_CHANGED, "compact", "abort",
                          compaction_id, "--to-h", "--confirm-h", world.confirm_h[:12])
    second = world.refused("ss-abort-refused-2", compaction.ABORT_PROTECTED_CHANGED, "compact", "abort",
                           compaction_id, "--to-h", "--confirm-h", world.confirm_h[:12])
    digest = first["report"]["w_sha256"]
    report = first["report"]["w"]
    assert second["report"] == first["report"] and retired["report"]["w_sha256"] == digest
    assert report["class"] == compaction.W_RESTRICTIVE
    assert report["tables"]["data/auth.db"]["credentials"] == {"added": 0, "removed": 1}
    content = json.dumps(first["report"])
    assert "cred-sc6b" not in content and "user-sc6b" not in content  # the report is content-free
    rec.result(w_sha256=digest, w_class=report["class"], verify_branch_reached=report["verify_branch_reached"],
               w_tables=report["tables"].get("data/auth.db"))
    # R5-PRIV-3 and the unit under SS: adopt and its dry run refuse; a start exits 78 without a restart.
    for flag in (("--dry-run",), ("--confirm", "0" * 64)):
        world.refused("ss-adopt" + flag[0], "runtime_admission_compaction_pending", "adopt", *flag)
    world.child("clean")
    world.begin("ss-start-refused")
    shown = world.wait_settled(120)
    rec.result(ss_start={key: shown[key] for key in ("ExecMainStatus", "NRestarts", "Result")})
    assert shown["ExecMainStatus"] == "78" and shown["NRestarts"] == "0", shown
    if shown["ActiveState"] == "failed":
        world.ctl("ss-reset-failed", "reset-failed", world.unit)
    # The runbook's SS state: nothing destroyed or revived, three copies (H, the snapshot, K'), the unit
    # stopped and disabled, so no process runs the scheduled deletions of any copy.
    h_state = Path(stage.read_binding(world.root, h_binding)["state_root"])
    k_state = world.root / "slots" / f"k{world.k}" / "state"
    snapshots = [event["data"] for event in world.log(compaction_id)["events"] if event["kind"] == "snapshot"]
    unit = world.systemd.raw_show(world.d.unit, "ActiveState", "UnitFileState")
    ss_state = {"h": _sql(h_state / "data" / "auth.db", "SELECT id FROM credentials"),
                "k_prime": _sql(k_state / "data" / "auth.db", "SELECT id FROM credentials"),
                "snapshots": [Path(item["path"]).is_dir() and stage._identity(Path(item["path"])) == item["identity"]
                              for item in snapshots],
                "head_slot": world.slot_of_head(), "unit": unit}
    rec.result(ss_state=ss_state)
    assert ss_state == {"h": [("cred-sc6b",)], "k_prime": [], "snapshots": [True], "head_slot": f"k{world.k}",
                        "unit": {"ActiveState": "inactive", "UnitFileState": "disabled"}}, ss_state
    world.refused("ss-discard-wrong-digest", compaction.DISCARD_MISMATCH, "compact", "abort", compaction_id,
                  "--to-h", "--confirm-h", world.confirm_h[:12], "--discard-k-changes", "0" * 12)
    world.abort_to_h(compaction_id, discard=digest[:12])  # SC-D8 A: the person confirmed in writing
    [started] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["discarded"]["w_sha256"] == digest
    rec.result(discarded_counts=started["data"]["discarded"]["counts"])
    world.back_on_h(compaction_id, h_binding)
    state = world.root / "slots" / world.slot_of_head() / "state"
    assert _sql(state / "data" / "auth.db", "SELECT id FROM credentials") == [("cred-sc6b",)]


# --- (6) and (12): rollback R2 -> R2' and back; accepted release killed, R-b, rollback; extended ---------

def test_06_12_rollbacks_and_the_accepted_release_killed(builds, tmp_path, systemd, record):
    rec = record("06-12", "(6) R2 -> R2' and back without T0; (12) accepted release killed without receipt, "
                          "R-b, rollback; extended: adopt with a confirmed regression, hotfix T0 without the H "
                          "check, plan and start only with --accept-regressed-source (row PA), V and retire")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.transition("x5", name="r1-to-r2")  # k1 serves R2 (served: x5)
    world.stop_and_disable("t0-r2prime-paso1")
    new = world.extract("x5r2")
    assert world.triple_check(new) == "T0"
    world.t0("x5r2", new=new)
    world.h_check("t0-r2prime-h-check")
    world.transition("x5r2", name="r2-to-r2prime")  # k2 serves R2'
    # Data go forward: a deletion recorded under R2' (a tombstone and its erasure) must survive the rollback.
    world.serve_clean("r2prime-records-a-deletion", execute=H_TOMBSTONE)
    # (6) the rollback R2' -> R2: a lower triple in served(root), no T0 (install-controller refuses it).
    new = world.extract("x5")
    decision = world.triple_check(new)
    rec.result(rollback_decision=decision)
    assert decision == "sin T0"
    world.stop_and_disable("rollback-t0-refused-paso1")
    witness = world.witness()
    refused = world.t0("x5", new=new, expect=78)
    assert refused["refused"] == "runtime_admission_controller_downgrade" and world.witness() == witness
    world.transition("x5", name="rollback-r2prime-to-r2")  # k3 serves R2 under the revision-2 controller
    privacy = world.root / "slots" / world.slot_of_head() / "state" / "data" / "privacy.db"
    kept = {"tombstones": _sql(privacy, "SELECT conversation_id FROM conversation_tombstones"),
            "erasures": _sql(privacy, "SELECT id, conversation_id FROM conversation_erasures")}
    rec.result(rollback_kept_the_deletion=kept)
    assert kept == {"tombstones": [("conv-h",)], "erasures": [("op-h", "conv-h")]}, kept
    assert list(stage.installed_controller_triple(world.root)) == builds["x5r2"]["triple"]
    world.transition("x5r2", name="forward-r2-to-r2prime")  # k4 serves R2' again: the equal triple, no T0
    # (12) the accepted release (k4, R2') killed without receipt; the production restart runs R-b.
    world.child("clean")
    world.begin("accepted-start")
    assert world.wait_active()["ActiveState"] == "active"
    killed_head = world.head()["binding_id"]
    started = time.monotonic()
    world.kill("accepted-sigkill")
    shown = world.systemd.wait_for(world.d.unit, lambda value: value["ActiveState"] == "active"
                                   and value["NRestarts"] not in ("", "0"), 120, *STATUS)
    rec.time("accepted-kill-to-ready-after-restart", time.monotonic() - started)
    events = [line.split("[Launcher] ", 1)[1] for line in world.console().splitlines() if "[Launcher] " in line]
    rec.result(accepted_kill={"restarts": shown["NRestarts"], "active": shown["ActiveState"],
                              "recovery_events": [event for event in events if event.startswith("recovery")][-3:]})
    assert shown["ActiveState"] == "active", (shown, world.console()[-3000:])
    assert any(event.startswith("recovery") for event in events)
    world.ctl("accepted-stop", "stop", world.unit)
    world.d.wait_inactive(120)
    assert world.head()["status"] == "closed" and world.slot_of_head() == "k4"
    rec.result(head_binding_kept=world.head()["binding_id"] == killed_head)
    assert world.head()["binding_id"] == killed_head
    world.ctl("accepted-disable", "disable", world.unit)
    world.transition("x5", name="rollback-after-r-b")  # k5 serves R2
    k5_binding = world.head()["binding_id"]
    k5_state = world.root / "slots" / "k5" / "state"
    # Extended (R4-OPS-2, row PA with confirmed regressions): k5 records one more tombstone and closes; then
    # it is killed with no restart (the head stays open) and that tombstone goes, so the tombstone row set is
    # older than k5's last closure: exactly the state k5's earlier clean close proved (SC6B-A-12's shape).
    # The operator's adopt confirms that regression; a greater-triple hotfix T0 without the H check; plan and
    # start need --accept-regressed-source with the record's sha256.
    world.serve_clean("pa-records-a-tombstone", execute=PA_TOMBSTONE)
    assert world.head()["binding_id"] != k5_binding
    world.child("clean", restart=False)
    world.begin("pa-start")
    assert world.wait_active()["ActiveState"] == "active"
    world.kill("pa-sigkill")
    world.d.wait_inactive(60)
    rec.result(head_after_pa_kill=world.head()["status"])
    assert world.head()["status"] in ("running", "unknown"), world.head()  # open: no receipt, no R-b
    world.ctl("pa-reset-failed", "reset-failed", world.unit, expect=None)
    world.stop_and_disable("pa-paso1")
    with closing(sqlite3.connect(k5_state / "data" / "privacy.db")) as db, db:
        db.execute("DELETE FROM conversation_tombstones WHERE conversation_id = 'conv-pa'")
    shadows = tmp_path / "adopt-tmpdir"
    shadows.mkdir()
    env = world.systemd.environment({"TMPDIR": str(shadows)})
    dry = world.st("adopt-dry-run", "adopt", "--dry-run", env=env)
    inside = stage.dry_run_directory(world.root)
    leftover = sorted(path.name for path in shadows.glob("kza-adopt-dry-run-*")) + (
        sorted(path.name for path in inside.iterdir()) if inside.is_dir() else [])
    families = sorted({item.split(":", 1)[0] for item in dry["report"]["regressions"]})
    rec.result(adopt_dry_run={"regressions": families, "shadow_left": leftover})
    assert families == ["conversation_tombstones"], dry["report"]
    assert leftover == []  # row DR: the dry run's copy is under the root and removed when it finishes
    unconfirmed = world.st("adopt-confirm-without-regressions", "adopt", "--confirm", dry["head_sha256"],
                           expect=None, env=env)
    rec.result(adopt_unconfirmed=unconfirmed.get("refused"))
    assert unconfirmed.get("refused") == "runtime_admission_recovery_confirmation", unconfirmed
    assert world.head()["status"] in ("running", "unknown")
    adopted = world.st("adopt-confirm", "adopt", "--confirm", dry["head_sha256"], "--confirm-regressions",
                       ",".join(families), env=env)
    head, sha256 = world.head(), adopted["recovery_record_sha256"]
    record = stage._read_recovery_record(world.root, head)
    assert head["status"] == "closed" and stage.read_manual_head(world.root, head)["kind"] == "recovery"
    assert stage._digest(record) == sha256 and record["checks"]["regressed"] == families
    # SC6B-A-12's shape: the adopted head has k5's earlier binding, which k5's earlier clean close proved.
    proved = (world.root / "admission" / "closures" / (head["binding_id"] + ".json")).exists()
    rec.result(adopted_binding_is_k5s=head["binding_id"] == k5_binding, older_closure_proves_it=proved)
    assert head["binding_id"] == k5_binding and proved
    new = world.extract("x5r3")
    assert world.triple_check(new) == "T0"
    installed = world.t0("x5r3", new=new)  # SR-1 alone gates it; no H check (R4-OPS-2)
    assert installed["controller_revision"] == 3
    world.k += 1
    k = f"k{world.k}"
    # Row PA: without the flag plan refuses, naming the record and its families, and writes nothing.
    refused = world.refused("pa-plan-hotfix-without-flag", compaction.REGRESSED_SOURCE_REQUIRED, "compact", "plan",
                            k, "--archive", str(builds["x5r3"]["archive"]))
    rec.result(plan_over_regressed_head=refused["refused"], plan_report=refused.get("report"))
    assert refused["report"] == {"recovery_record_sha256": sha256, "regressed": families}, refused
    world.plan("x5r3", k, expect=78, code=compaction.REGRESSED_SOURCE_MISMATCH,
               accept=("0" if sha256[0] != "0" else "1") + sha256[1:12])
    planned = world.plan("x5r3", k, accept=sha256[:12])
    assert planned["source_recovery_sha256"] == sha256, planned
    world.refused("pa-start-hotfix-without-flag", compaction.REGRESSED_SOURCE_REQUIRED,
                  *world.start_arguments("x5r3", k, planned))
    compaction_id = world.start("x5r3", k, planned, accept=sha256[:12])
    assert world.log(compaction_id)["intent"]["source_recovery_sha256"] == sha256
    world.verify(compaction_id)
    world.v_run("hotfix-v")
    status = world.status(compaction_id)
    assert status["verified"] is True and status["armed_run"] == "consumed", status
    world.retire(compaction_id)
    [begun] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "retire_started"]
    assert begun["data"]["w"]["class"] == compaction.W_EQUAL
    assert world.slot_of_head() == k and not k5_state.exists()
    world.serve_clean("hotfix-serve-k-prime", enable=True)
    privacy = world.root / "slots" / k / "state" / "data" / "privacy.db"
    carried = {"tombstones": _sql(privacy, "SELECT conversation_id FROM conversation_tombstones"),
               "erasures": _sql(privacy, "SELECT id, conversation_id FROM conversation_erasures")}
    rec.result(hotfix_after_adopt={"k": k, "compaction_id": compaction_id, "w_class": begun["data"]["w"]["class"],
                                   "installed_triple": list(stage.installed_controller_triple(world.root))[:2],
                                   "carried_forward": carried})
    # The confirmed regression goes forward as the adopted head had it (no 'conv-pa'); nothing is revived.
    assert carried == {"tombstones": [("conv-h",)], "erasures": [("op-h", "conv-h")]}, carried
    # R5V-2: from the accepted head the flag names no regressed record.
    world.plan("x5r3", f"k{world.k + 1}", expect=78, code=compaction.REGRESSED_SOURCE_MISMATCH, accept=sha256[:12])


# --- (7) and (9): schema mismatch; lower revision and equal pair with another digest -----------------------

def test_07_09_schema_contract_and_revision_refusals(builds, tmp_path, systemd, record):
    rec = record("07-09", "(7) schema mismatch refused at install and plan; (9) lower revision and equal "
                          "pair with another digest refused")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.stop_and_disable("refusals-paso1")
    found = {}
    for build, code in (("x5r2schema", "runtime_admission_controller_schema_contract"),):
        witness = world.witness()
        refused = world.t0(build, expect=78)
        found["install_" + build] = refused.get("refused")
        assert refused.get("refused") == code and world.witness() == witness, refused
    found["plan_x5schema"] = world.plan("x5schema", "k1", expect=78,
                                        code="runtime_admission_compaction_schema_contract")["refused"]
    # (9): revision 2 installed; then revision 1 (lower) and revision 2 with another closure.
    new = world.extract("x5r2")
    assert world.triple_check(new) == "T0"
    world.t0("x5r2", new=new)
    world.h_check("t0-r2-h-check")
    new = world.extract("x5r2c")
    found["paso2_x5r2c"] = world.triple_check(new)
    assert found["paso2_x5r2c"] == "conflicto: no seguir"
    for build, code in (("x5", "runtime_admission_controller_downgrade"),
                        ("x5c", "runtime_admission_controller_downgrade"),
                        ("x5r2c", launcher.CONTROLLER_REVISION_CONFLICT)):
        witness = world.witness()
        refused = world.t0(build, expect=78, new=new if build == "x5r2c" else None)
        found["install_" + build] = refused.get("refused")
        assert refused.get("refused") == code and world.witness() == witness, refused
    for build, code in (("x5r2c", "runtime_admission_compaction_controller_revision_conflict"),
                        ("x5c", compaction.CONTROLLER_BEHIND), ("x5", compaction.CONTROLLER_BEHIND)):
        found["plan_" + build] = world.plan(build, "k1", expect=78, code=code)["refused"]
    rec.result(refusals=found)
    world.h_check("after-refusals-h-serves")


# --- (8) a delegation record written after T0 -------------------------------------------------------------

def _legacy_slot(world: World, builds: dict) -> None:
    """slots/legacy: the e9669ec runtime (its archive extracted) and a state with config/ and data/."""
    runtime = world.root / "slots" / "legacy" / "runtime"
    with tarfile.open(builds["legacy"]["archive"]) as bundle:
        bundle.extractall(runtime, filter="data")
    for name in ("config", "data"):
        (world.root / "slots" / "legacy" / "state" / name).mkdir(parents=True)


def test_08_a_delegation_record_after_t0_refuses_plan_and_serve_and_l4_works(builds, tmp_path, systemd, record):
    rec = record("08", "Delegation record written after T0: plan and serve refuse, L4 uninstall still works")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.stop_and_disable("dl-paso1")
    record_path = world.root / "admission" / "delegations" / ("a" * 64 + ".json")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{}")
    world.plan("x5", "k1", expect=78, code=stage.DELEGATION_FROZEN)
    witness = world.witness()
    world.child("clean")
    world.begin("dl-start-refused")
    shown = world.wait_settled(120)
    rec.result(dl_start={key: shown[key] for key in ("ExecMainStatus", "NRestarts", "Result")})
    assert shown["ExecMainStatus"] == "78" and shown["NRestarts"] == "0", shown
    assert any("refused " + stage.DELEGATION_FROZEN in line for line in world.console().splitlines())
    assert world.witness() == witness
    if shown["ActiveState"] == "failed":
        world.ctl("dl-reset-failed", "reset-failed", world.unit)
    # Row DL: only the record (H unchanged): removing it gives serve back.
    record_path.unlink()
    world.serve_clean("dl-removed-h-serves")
    record_path.write_text("{}")
    _legacy_slot(world, builds)
    (world.d.unit_dir / "zz-rehearsal.conf").unlink()  # production has no rehearsal drop-in
    systemd.reload()
    _code, parsed, _completed = world.run("l4-uninstall-launcher", [
        sys.executable, "-I", "-B", str(world.controller), "--root", str(world.root), "--offline",
        "uninstall-launcher", "--unit-dir", str(world.d.unit_dir), "--user-decision", "sc6b-l4-rehearsal",
        "--phrase", launcher.UNINSTALL_PHRASE, "--legacy-slot", "legacy"])
    rec.result(l4={"mismatches": parsed.get("mismatches"), "record": bool(parsed.get("record"))})
    assert parsed.get("mismatches") == [], parsed
    assert (world.d.unit_dir / "legacy.conf").is_file() and not (world.d.unit_dir / "managed.conf").exists()


# --- (10) a queued audio deletion survives a transition ---------------------------------------------------

def test_10_a_queued_audio_deletion_survives_and_k_removes_its_own_file(builds, tmp_path, systemd, record):
    rec = record("10", "Queued audio deletion survives a transition and is removed under K's path")
    world = World(builds, tmp_path, systemd, rec, first=cw.LEDGER + "\n" + _QUEUE)
    h_state = world.root / "slots" / "s" / "state"
    h_file = h_state / "data" / "ambient_audio" / "queued.flac"
    assert _sql(h_state / "data" / "ambient.db", "SELECT path FROM pending_audio_deletions") == [(str(h_file),)]
    _first_t0(world)
    moved = world.transition("x5", name="queued-audio")
    k_state = world.root / "slots" / moved["k"] / "state"
    k_file = k_state / "data" / "ambient_audio" / "queued.flac"
    assert _sql(k_state / "data" / "ambient.db", "SELECT path FROM pending_audio_deletions") == [(str(k_file),)]
    assert k_file.read_bytes() == b"fLaC-sc1b-queued" and not h_state.exists()
    out = tmp_path / "k-retry.json"
    world.ctl("k-retry-disable", "disable", world.unit)
    world.serve_clean("k-prime-retry", execute=_k_retry(out))
    retried = json.loads(out.read_text())
    rec.result(k_retry=retried, k_rows_after=len(_sql(k_state / "data" / "ambient.db",
                                                        "SELECT path FROM pending_audio_deletions")),
               k_file_after=k_file.exists())
    assert retried == {"removed": 1} and not k_file.exists()
    assert _sql(k_state / "data" / "ambient.db", "SELECT path FROM pending_audio_deletions") == []


# --- (11) a corrected controller installed with K' open, then abort ---------------------------------------

def test_11_a_corrected_controller_with_k_prime_open_then_abort(builds, tmp_path, systemd, record):
    rec = record("11", "Corrected controller (unchanged projection) installed with K' open, then abort")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "ci")
    world.v_run("ci-v", restart=False, kill_after_ready=True)
    assert world.head()["status"] != "closed"
    new = world.extract("x5r2")
    assert world.triple_check(new) == "T0"
    installed = world.t0("x5r2", new=new)  # row CI: no H check, the unit stays stopped and disabled
    assert installed["controller_revision"] == 2
    assert stage.pending_compaction(world.root)["compaction_id"] == compaction_id
    assert world.head()["status"] != "closed"
    status = world.status(compaction_id)
    rec.result(ci_status={key: status[key] for key in ("phase", "armed_run", "verified", "verify_branch_reached")})
    world.abort_to_h(compaction_id)
    world.back_on_h(compaction_id, h_binding, name="h-after-ci-abort")
    assert list(stage.installed_controller_triple(world.root)) == builds["x5r2"]["triple"]


# --- (13) T0 killed after its checks, and between the switch and controller.json ---------------------------

def test_13a_t0_killed_after_its_checks_then_h_serves_and_the_install_is_repeated(builds, tmp_path, systemd,
                                                                                  record):
    rec = record("13a", "T0 killed after its checks (row T0): H serves under the old controller; repeat")
    world = World(builds, tmp_path, systemd, rec)
    world.stop_and_disable("t0-paso1")
    before = (json.loads((world.root / "admission" / "controller.json").read_text()),
              os.readlink(world.root / "controller" / "current"))
    witness = world.witness()
    killed = world.t0("x5", expect=-9, kill_at="checked")
    after = (json.loads((world.root / "admission" / "controller.json").read_text()),
             os.readlink(world.root / "controller" / "current"))
    nothing_written = world.witness() == witness  # row T0: the epoch-5 checks write nothing under the root
    rec.result(killed_exit=killed["exit"], unchanged=after == before, nothing_written=nothing_written)
    assert after == before and nothing_written
    world.serve_clean("h-under-the-old-controller", enable=True)
    installed = world.t0("x5")
    assert installed["controller_epoch"] == 5
    world.h_check()


def test_13b_t0_killed_between_the_switch_and_controller_json_then_repeated(builds, tmp_path, systemd, record):
    rec = record("13b", "T0 killed between the current switch and controller.json (row T0S): repeat install")
    world = World(builds, tmp_path, systemd, rec)
    world.stop_and_disable("t0-paso1")
    world.t0("x5", expect=-9, kill_at="switched")
    record_after = json.loads((world.root / "admission" / "controller.json").read_text())
    current = os.readlink(world.root / "controller" / "current")
    rec.result(controller_json_epoch=record_after["controller_epoch"], current_is_new=current
               == builds["x5"]["manifest"]["git_commit"])
    assert record_after["controller_epoch"] == 4 and current == builds["x5"]["manifest"]["git_commit"]
    witness = world.witness()
    world.child("clean")
    world.begin("t0s-start-refused")
    shown = world.wait_settled(120)
    rec.result(t0s_start={key: shown[key] for key in ("ExecMainStatus", "NRestarts", "Result")})
    assert shown["ExecMainStatus"] == "78" and shown["NRestarts"] == "0", shown
    assert any(stage.CONTROLLER_INSTALL_INCOMPLETE in line for line in world.console().splitlines())
    if shown["ActiveState"] == "failed":
        world.ctl("t0s-reset-failed", "reset-failed", world.unit)
    assert world.witness() == witness
    world.plan("x5", "k1", expect=78, code=stage.CONTROLLER_INSTALL_INCOMPLETE)
    installed = world.t0("x5")  # the same command again
    assert installed["controller_epoch"] == 5
    assert json.loads((world.root / "admission" / "controller.json").read_text())["controller_epoch"] == 5
    world.h_check()


# --- (14) a correct NEW over an H with residue: retire (R5-PRIV-1, R5V-1) ---------------------------------

RESIDUE = ("retired_owner_facts", "retired_user_credential_session", "recorders_crash_running",
           "reminder_action_sending", "wake_clip_fenced", "expired_voice_group_request", "routine_run_interrupted",
           "retention_shortened")


def _residue_code() -> str:
    """H's own run plants the residue (rows an interrupted or older process leaves) with tools/sc0/residue.py;
    recorders_crash_running leaves an active (as after a clean stop), a reserved and an unknown recorder."""
    return "\n".join([
        "import json, sys",
        f"sys.path.append({str(REPO)!r})",
        "import types",
        "import yaml",
        "from src.memory import fact_ledger",
        "_noop = lambda *args, **kwargs: None",  # as tests/mocks/compaction_world.LEDGER: planting, not the graph
        "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
        "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
        "from tools.sc0 import residue",
        "data = state / 'data'",
        "planted = {}",
        f"for name in {list(RESIDUE)!r}:",
        "    planted[name] = residue.STATES[name](data, data / 'graph')",
        "import chromadb",
        "from chromadb.config import Settings",
        "_chroma = chromadb.PersistentClient(path=str(data / 'chroma_db'), settings=Settings(anonymized_telemetry=False))",
        "try:",
        "    _routines = _chroma.get_or_create_collection(name='home_assistant_routines', metadata={'hnsw:space': 'cosine'})",
        "    _routines.add(ids=['r-legacy_0'], documents=['sc6b-legacy-routine'], embeddings=[[0.25, 0.5, 0.75]])",
        "finally:",
        "    _chroma.close()",
        "planted['legacy_chroma_routines'] = {'planted': ['data/chroma_db:home_assistant_routines']}",
        "settings = state / 'config' / 'settings.yaml'",
        "raw = yaml.safe_load(settings.read_text())",
        "for result in planted.values():",
        "    for dotted, value in (result.get('config') or {}).items():",
        "        *parents, leaf = dotted.split('.')",
        "        owner = raw",
        "        for key in parents:",
        "            owner = owner.setdefault(key, {})",
        "        owner[leaf] = value",
        "settings.write_text(yaml.safe_dump(raw, sort_keys=False))",
        "print(json.dumps({name: value.get('planted') for name, value in planted.items()}))",
    ])


def _app_source(tmp_path: Path, port: int) -> Path:
    """The offline source of H: the canonical portable settings (paths as ${KZA_SLOT_BIND_*}), data/graph."""
    from tests.mocks.managed_slot_graph import canonical_template
    from tests.mocks.restore_application_graph import configuration
    source = tmp_path / "app-offline"
    (source / "config").mkdir(parents=True)
    (source / "data" / "graph").mkdir(parents=True, mode=0o700)
    placeholder = (tmp_path / "root" / "slots" / "s" / "state" / "data" / "graph").resolve()

    def portable(root, home, model):
        config = configuration(root, home, model)
        config["dashboard"].update(enabled=True, host="127.0.0.1", port=port)
        return config

    raw = canonical_template(placeholder, portable)
    text = yaml.safe_dump(raw, sort_keys=False)
    assert str(placeholder) not in text
    (source / "config" / "settings.yaml").write_text(text)
    return source


def test_14_a_correct_new_over_h_with_residue_is_retired(builds, tmp_path, systemd, record):
    rec = record("14", "Correct NEW over an H with residue, the real compiled application as H's and NEW's "
                       "child: plan (SC-4c closed list), start (the memory store rebuilt), V and retire")

    port = _free_port()  # the dashboard port is a literal of the state's settings (copied into K)

    def first(world: World) -> None:
        world.app_port = port
        world.extra_env = list(OFFLINE_ENV)
        world.serve_clean("h-first-boot-real-app", app=True)
        world.serve_clean("h-residue-run", execute=_residue_code())

    world = World(builds, tmp_path, systemd, rec, source=_app_source(tmp_path, port), first_child=first)
    h_state = world.root / "slots" / "s" / "state"
    graph = h_state / "data" / "graph"
    counts = {
        "retired_users": len(_sql(graph / "dashboard-auth.db", "SELECT 1 FROM retired_users")),
        "recorders": sorted(row[0] for row in _sql(graph / "privacy.db",
                                                   "SELECT state FROM capture_browser_recorders")),
        "reminder_sending": len(_sql(graph / "reminders.db",
                                     "SELECT 1 FROM reminder_home_actions WHERE status='sending'")),
        "routine_running": len(_sql(graph / "routines.db", "SELECT 1 FROM routine_runs WHERE status='running'"))}
    rec.result(residue_counts=counts)
    # retired_owner_facts and retired_user_credential_session retire one user each; recorders_crash_running
    # leaves an active, a reserved and an unknown recorder; one 'sending' action; one interrupted run.
    assert counts == {"retired_users": 2, "recorders": ["active", "reserved", "unknown"], "reminder_sending": 1,
                      "routine_running": 1}, counts
    _first_t0_app(world)
    world.st("residue-paso0-completion-status", "completion-status")
    world.stop_and_disable("residue-paso1")
    world.k += 1
    planned = world.plan("x5", "k1")
    stores = planned["chroma"]["stores"]
    rec.result(plan_chroma=stores)
    assert stores["data/chroma_db"]["dropped_collections"] == ["home_assistant_routines"]
    assert stores["data/graph/chroma"]["rebuilt"] == 1  # the memory store: SC-4c's isolated rebuild child
    compaction_id = world.start("x5", "k1", planned)
    world.verify(compaction_id)
    world.v_run("residue-v", app=True)
    status = world.status(compaction_id)
    assert status["verified"] is True and status["armed_run"] == "consumed", status
    assert status["verify_branch_reached"] == "yes", status
    head = world.head()  # V's own receipt closed K' (its closure proof reads and checks that receipt)
    assert head["status"] == "closed" and world.slot_of_head() == "k1"
    proof = stage._closure_proof(world.root, head["binding_id"])
    retired = world.retire(compaction_id)
    [begun] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "retire_started"]
    assert begun["data"]["w"]["class"] == compaction.W_EQUAL
    assert world.kinds(compaction_id)[-1] == "complete" and not h_state.exists()  # H and its residue destroyed
    world.serve_clean("residue-serve-k-prime", enable=True, app=True)
    rec.result(retired={"k": "k1", "compaction_id": compaction_id, "destroyed": retired.get("destroyed"),
                        "receipt_sha256": proof["receipt_sha256"], "w_class": begun["data"]["w"]["class"]})


def _first_t0_app(world: World) -> None:
    world.stop_and_disable("t0-paso1")
    new = world.extract("x5")
    assert world.triple_check(new) == "T0"
    world.t0("x5", new=new)
    world.serve_clean("t0-h-check-real-app", enable=True, app=True)


# --- (15) adopt --dry-run killed after acceptance: the next serve removes the shadow (row DR) ---------------

def _accepted_release_killed_without_restart(world: World, name: str) -> dict:
    """The accepted release SIGKILLed after READY with Restart=no: the head stays open (no receipt, no R-b)."""
    world.child("clean", restart=False)
    world.begin(name + "-start")
    assert world.wait_active()["ActiveState"] == "active"
    world.kill(name + "-sigkill")
    world.d.wait_inactive(60)
    head = world.head()
    assert head["status"] in ("running", "unknown"), head
    world.ctl(name + "-reset-failed", "reset-failed", world.unit, expect=None)
    return head


def test_15_a_killed_adopt_dry_run_leaves_its_shadow_under_the_root_until_the_next_serve(builds, tmp_path, systemd,
                                                                                        record):
    """Row DR after acceptance: the compiled ``adopt --dry-run`` SIGKILLed mid-copy leaves its plaintext
    shadow only under <root>/admission/dry-run/<32 hex>/ (0700), nothing in TMPDIR and the head untouched;
    the next lease holder, the unit's serve, removes it before anything else, recovers the head (bounded
    R-b) and serves."""
    rec = record("15", "adopt --dry-run killed mid-copy after acceptance: the shadow stays only under the root "
                       "(row DR); the next serve removes it, recovers the head with R-b and serves")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.transition("x5", name="r1-to-r2")  # k1 accepted: R2 serves
    killed = _accepted_release_killed_without_restart(world, "accepted")
    world.stop_and_disable("dr-paso1")
    tmpdir = tmp_path / "adopt-tmpdir"
    tmpdir.mkdir()
    env = world.systemd.environment({"TMPDIR": str(tmpdir), "KZA_REHEARSAL_KILL_AT_STEP": "dry-run-mid-copy"})
    witness = world.witness()
    world.run("adopt-dry-run-killed-mid-copy", [sys.executable, "-I", "-B", str(WRAPPER), str(world.controller),
                                                "--root", str(world.root), "--offline", "adopt", "--dry-run"],
              expect=-signal.SIGKILL, env=env)
    directory = stage.dry_run_directory(world.root)
    [shadow] = list(directory.iterdir())
    copied = sorted(path.relative_to(shadow / "state").as_posix() for path in (shadow / "state").rglob("*")
                    if path.is_file())
    left = {"shadow_name_is_32_hex": re.fullmatch(r"[0-9a-f]{32}", shadow.name) is not None,
            "modes": {"dry-run": oct(stat.S_IMODE(os.lstat(directory).st_mode)),
                      "shadow": oct(stat.S_IMODE(os.lstat(shadow).st_mode))},
            "files_copied": copied, "tmpdir_entries": sorted(path.name for path in tmpdir.iterdir()),
            "file_modes": sorted({oct(stat.S_IMODE(os.lstat(shadow / "state" / name).st_mode)) for name in copied}),
            "head_unchanged": world.head() == killed}
    rec.result(killed_dry_run=left)
    assert left["shadow_name_is_32_hex"] and left["modes"] == {"dry-run": "0o700", "shadow": "0o700"}, left
    assert left["file_modes"] == ["0o600"], left
    assert len(copied) == 2 and left["tmpdir_entries"] == [] and left["head_unchanged"], left
    # The kill wrote nothing else under the root: only the shadow is new.
    after = world.witness()
    changed = sorted(key for key in set(after) | set(witness) if after.get(key) != witness.get(key))
    rec.result(paths_changed_under_the_root=len(changed),
               all_under_admission_dry_run=all(key.startswith("admission/dry-run") for key in changed))
    assert changed and all(key.startswith("admission/dry-run") for key in changed), changed
    # The next serve (the unit's start: the next lease holder) removes the copy, then recovers and serves.
    before = launcher._recoveries(world.root)
    served = world.serve_clean("serve-after-killed-dry-run", enable=True, keep_enabled=True)
    added = [item for item in launcher._recoveries(world.root) if item not in before]
    rec.result(after_serve={"shadow_left": os.path.lexists(directory),
                            "recoveries_added": [item["mode"] for item in added], "head": world.head()["status"],
                            "start_to_ready": round(served["start_to_ready"], 3)})
    assert not os.path.lexists(directory)
    assert [item["mode"] for item in added] == ["automatic"], added
    assert world.head()["status"] == "closed" and world.slot_of_head() == "k1"
    assert list(tmpdir.iterdir()) == []


# --- (16) a NEW that ignores the verify flag -----------------------------------------------------------------

def test_16_a_new_ignoring_the_verify_flag_is_stopped_at_ready_and_aborted(builds, tmp_path, systemd, record):
    rec = record("16", "NEW ignores the verify flag: serve stops it at READY, not verified, lossless abort")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    h_binding = world.head()["binding_id"]
    compaction_id = world.armed("x5", "v-ignored")
    shown = world.v_run("v-ignored", scenario="verify_ignored")
    status = world.status(compaction_id)
    rec.result(status={key: status[key] for key in ("armed_run", "verified", "verify_branch_reached")},
               unit={key: shown[key] for key in ("ExecMainStatus", "NRestarts", "Result")},
               head=world.head()["status"])
    assert status["verified"] is False and status["verify_branch_reached"] == "no"
    assert world.head()["status"] == "closed" and "verified" not in world.kinds(compaction_id)
    world.refused("paso8-retire-refused", "runtime_admission_compaction_not_verified", "compact", "retire",
                  compaction_id)
    world.abort_to_h(compaction_id)
    [started] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["w"]["class"] == compaction.W_EQUAL
    world.back_on_h(compaction_id, h_binding)


# --- (17) accepted release dying with a regressed R-b: adopt, hotfix with the flag, retire, one more -------

DELETE_PA_TOMBSTONE = cw.sql(("DELETE FROM conversation_tombstones WHERE conversation_id = 'conv-pa'", ()))


def _fact_deleted(stale: Path) -> str:
    """Child code: a synthetic fact stored through the production ledger, a stale copy of privacy.db taken
    outside the state (a backup a restore could later use), then the fact deleted through the review boundary
    (state 'deleted', content and sources scrubbed, a fact_deleted ledger entry)."""
    return "\n".join([
        cw._LEDGER_OPEN,
        "import time, uuid",
        "from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer, MemorySource",
        "now = time.time()",
        "source = MemorySource(str(uuid.uuid4()), str(uuid.uuid4()), now - 60, now + 3600, MemoryOrigin.VOICE)",
        "fact = ledger.store('sc6b synthetic fact', 'fact', 0.9, (source,))",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db')) as live, "
        f"closing(sqlite3.connect({str(stale)!r})) as copy:",
        "    live.backup(copy)",
        "ledger.delete_fact(fact.id, MemoryReviewer('fixture:operator'))",
        "ledger.close()",
    ])


def _fact_restored(stale: Path) -> str:
    """Child code: the dying release restores the fact rows from the stale copy (a resurrected_fact family)."""
    return "\n".join([
        "import sqlite3",
        "from contextlib import closing",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db')) as db, db:",
        f"    db.execute('ATTACH DATABASE ? AS stale', ({str(stale)!r},))",
        "    db.execute('INSERT OR REPLACE INTO facts SELECT * FROM stale.facts')",
        "    db.execute('INSERT OR IGNORE INTO fact_sources SELECT * FROM stale.fact_sources')",
    ])


def _accepted_release_dies(world: World, name: str, execute: str, code: str) -> dict:
    """The accepted release changes its own state at boot and is SIGKILLed after READY under managed.conf's
    Restart=on-failure: the one restart's bounded R-b refuses ``code`` (exit 69, no second restart) and the
    head stays open."""
    shown = world.v_run(name, scenario="clean", execute=execute, kill_after_ready=True)
    lines = [line.split("[Launcher] ", 1)[1] for line in world.console().splitlines() if "[Launcher] " in line]
    found = {"exec_main_status": shown["ExecMainStatus"], "restarts": shown["NRestarts"],
             "status_text": shown["StatusText"], "head": world.head()["status"],
             "recovery_events": [line for line in lines if line.startswith("recovery")][-2:]}
    world.rec.result(**{name + ".r_b": found})
    assert shown["ExecMainStatus"] == "69" and shown["NRestarts"] == "1", found
    assert any(line.startswith(f"recovery R-b {code}") for line in lines), found
    assert found["head"] in ("running", "unknown"), found
    return found


def _confirmed_adopt(world: World, tmp_path: Path) -> tuple[list[str], str, dict]:
    """The runbook's adopt after acceptance (row PA): the dry run (its shadow under the root, row DR), the
    confirmation refused without the families, then adopt confirming exactly the dry run's families."""
    world.stop_and_disable("pa-paso1")
    tmpdir = tmp_path / "adopt-tmpdir"
    tmpdir.mkdir()
    env = world.systemd.environment({"TMPDIR": str(tmpdir)})
    dry = world.st("adopt-dry-run", "adopt", "--dry-run", env=env)
    report = dry["report"]
    families = sorted({item.split(":", 1)[0] for item in report["regressions"]}
                      | {"resurrected_" + name for name, count in report["resurrected"].items() if count})
    inside = stage.dry_run_directory(world.root)
    leftover = sorted(path.name for path in tmpdir.iterdir()) + (
        sorted(path.name for path in inside.iterdir()) if inside.is_dir() else [])
    world.rec.result(adopt_dry_run={"failures": report["failures"], "families": families,
                                    "resurrected": report["resurrected"], "shadow_left": leftover})
    assert report["failures"] == [] and leftover == [], (report, leftover)
    unconfirmed = world.st("adopt-confirm-without-regressions", "adopt", "--confirm", dry["head_sha256"],
                           expect=None, env=env)
    world.rec.result(adopt_unconfirmed=unconfirmed.get("refused"))
    assert unconfirmed.get("refused") == "runtime_admission_recovery_confirmation", unconfirmed
    assert world.head()["status"] in ("running", "unknown")
    adopted = world.st("adopt-confirm", "adopt", "--confirm", dry["head_sha256"], "--confirm-regressions",
                       ",".join(families), env=env)
    head, sha256 = world.head(), adopted["recovery_record_sha256"]
    recovery = stage._read_recovery_record(world.root, head)
    assert head["status"] == "closed" and stage.read_manual_head(world.root, head)["kind"] == "recovery"
    assert stage._digest(recovery) == sha256 and recovery["checks"]["regressed"] == families
    assert recovery["mode"] == "operator", recovery["mode"]
    return families, sha256, head


def _hotfix_t0_without_the_h_check(world: World, build: str) -> None:
    """A greater-triple hotfix over the adopted head: T0, SR-1 alone gates it, no H check (R4-OPS-2)."""
    new = world.extract(build)
    assert world.triple_check(new) == "T0"
    installed = world.t0(build, new=new)
    assert list(stage.installed_controller_triple(world.root)) == world.builds[build]["triple"], installed


def _accepting_transition_to_v(world: World, build: str, families: list[str], sha256: str) -> tuple[str, str]:
    """Row PA: plan and start refuse without --accept-regressed-source (naming the record and its families,
    writing nothing), a foreign prefix refuses, then plan, start (the intent binds the full sha256), V."""
    world.k += 1
    k = f"k{world.k}"
    refused = world.refused("pa-plan-without-flag", compaction.REGRESSED_SOURCE_REQUIRED, "compact", "plan", k,
                            "--archive", str(world.builds[build]["archive"]))
    world.rec.result(plan_over_regressed_head=refused["refused"], plan_report=refused.get("report"))
    assert refused["report"] == {"recovery_record_sha256": sha256, "regressed": families}, refused
    world.plan(build, k, expect=78, code=compaction.REGRESSED_SOURCE_MISMATCH,
               accept=("0" if sha256[0] != "0" else "1") + sha256[1:12])
    planned = world.plan(build, k, accept=sha256[:12])
    assert planned["source_recovery_sha256"] == sha256, planned
    world.refused("pa-start-without-flag", compaction.REGRESSED_SOURCE_REQUIRED,
                  *world.start_arguments(build, k, planned))
    compaction_id = world.start(build, k, planned, accept=sha256[:12])
    assert world.log(compaction_id)["intent"]["source_recovery_sha256"] == sha256
    world.verify(compaction_id)
    world.v_run("hotfix-v")
    status = world.status(compaction_id)
    assert status["verified"] is True and status["armed_run"] == "consumed", status
    return k, compaction_id


def test_17_an_accepted_release_dying_with_a_regressed_r_b_is_adopted_hotfixed_and_followed(builds, tmp_path,
                                                                                           systemd, record):
    """R5-1 and R5V-2 with a non-resurrected family (user answer scpadr_q1_20261118): k1 (accepted, R2)
    records a tombstone and closes; then, under the production restart, it removes that tombstone at boot
    and is SIGKILLed after READY: the restart's R-b refuses runtime_admission_recovery_regression (exit 69,
    one restart). The operator's adopt confirms conversation_tombstones; a greater-triple hotfix (R2', T0
    without the H check); plan and start only with --accept-regressed-source; V; retire (W 'equal'); K'
    serves without the tombstone (nothing revived); then one more ordinary transition completes with no
    flag, which the accepted head refuses."""
    rec = record("17", "Accepted release dying with a regressed R-b: adopt with confirmation, hotfix T0 without "
                       "the H check, plan and start with --accept-regressed-source, V, retire (R5-1), then one "
                       "more ordinary transition completes (R5V-2)")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.transition("x5", name="r1-to-r2")  # k1 accepted: R2 serves
    world.serve_clean("accepted-records-a-tombstone", execute=PA_TOMBSTONE)
    k1_privacy = world.root / "slots" / "k1" / "state" / "data" / "privacy.db"
    assert _sql(k1_privacy, "SELECT conversation_id FROM conversation_tombstones") == [("conv-pa",)]
    _accepted_release_dies(world, "accepted-dies", DELETE_PA_TOMBSTONE, "runtime_admission_recovery_regression")
    families, sha256, adopted = _confirmed_adopt(world, tmp_path)
    assert families == ["conversation_tombstones"], families
    _hotfix_t0_without_the_h_check(world, "x5r2")
    k, compaction_id = _accepting_transition_to_v(world, "x5r2", families, sha256)
    retired = world.retire(compaction_id)
    [begun] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "retire_started"]
    assert begun["data"]["w"]["class"] == compaction.W_EQUAL
    assert world.slot_of_head() == k and not k1_privacy.exists()
    world.serve_clean("hotfix-serve-k-prime", enable=True, keep_enabled=True)
    carried = _sql(world.root / "slots" / k / "state" / "data" / "privacy.db",
                   "SELECT conversation_id FROM conversation_tombstones")
    rec.result(hotfix={"k": k, "compaction_id": compaction_id, "w_class": begun["data"]["w"]["class"],
                       "destroyed": retired.get("destroyed"), "adopted_binding": adopted["binding_id"],
                       "installed_triple": list(stage.installed_controller_triple(world.root))[:2],
                       "tombstones_carried": carried})
    assert carried == [], carried  # the confirmed regression goes forward as adopted; nothing is revived
    # R5V-2: from the accepted head the flag names no regressed record; the next ordinary transition (equal
    # triple, no T0, no flag) completes.
    world.plan("x5r2", f"k{world.k + 1}", expect=78, code=compaction.REGRESSED_SOURCE_MISMATCH, accept=sha256[:12])
    moved = world.transition("x5r2", name="ordinary-after-hotfix")
    intent = world.log(moved["id"])["intent"]
    rec.result(ordinary_after_hotfix={"k": moved["k"], "plan_source_recovery_sha256":
                                      moved["planned"].get("source_recovery_sha256"),
                                      "intent_source_recovery_sha256": intent.get("source_recovery_sha256")})
    assert moved["planned"].get("source_recovery_sha256") is None and intent.get("source_recovery_sha256") is None
    assert world.slot_of_head() == moved["k"] and sorted(os.listdir(world.root / "slots")) == [moved["k"]]


def test_17b_a_resurrected_family_in_the_regressed_source_exits_by_abort_to_h(builds, tmp_path, systemd, record):
    """User answer scpadr_q1_20261118 (the documented limit, runbook «Después de la aceptación»): k1 deletes
    a fact and closes; dying under the production restart it restores the fact's rows from a stale copy of
    privacy.db, and the restart's R-b
    refuses runtime_admission_recovery_resurrection. adopt confirms resurrected_fact; the hotfix plans, starts
    and passes V with --accept-regressed-source, but retire refuses
    runtime_admission_compaction_retire_resurrected (K's baseline is follow-up) and abort --to-h returns
    losslessly (W 'equal') to the adopted H, where completion-status still counts the fact. Keeping the
    service down from there is the operator procedure (runbook «Después de la aceptación»: a written decision
    every 48 hours), not something the system enforces: no product gate refuses to serve that H, and this
    test simply never starts it."""
    rec = record("17b", "Accepted release dying with a resurrected family: adopt with confirmation, hotfix "
                        "with --accept-regressed-source to V, retire refused (retire_resurrected), abort to H")
    world = World(builds, tmp_path, systemd, rec)
    _first_t0(world)
    world.transition("x5", name="r1-to-r2")  # k1 accepted: R2 serves
    stale = tmp_path / "stale-privacy.db"
    world.serve_clean("accepted-deletes-a-fact", execute=_fact_deleted(stale))
    _accepted_release_dies(world, "accepted-dies", _fact_restored(stale), "runtime_admission_recovery_resurrection")
    families, sha256, adopted = _confirmed_adopt(world, tmp_path)
    assert families == ["resurrected_fact"], families
    # Lossless for the adopted H itself: no step from here to the end of the abort may write H's state (the
    # hotfix T0, plan, start, V on K, the refused retire, abort --to-h), and nothing is started on it.
    h_state = Path(stage.read_binding(world.root, adopted["binding_id"])["state_root"])

    def h_tree() -> dict:
        return {key: value for key, value in cw.tree(h_state).items() if not key.endswith(".lock")}

    h_before = h_tree()
    _hotfix_t0_without_the_h_check(world, "x5r2")
    k, compaction_id = _accepting_transition_to_v(world, "x5r2", families, sha256)
    refused = world.refused("paso8-retire-refused", compaction.RETIRE_RESURRECTED, "compact", "retire",
                            compaction_id)
    rec.result(retire_refused=refused["refused"], retire_report=refused.get("report"))
    assert refused["report"]["resurrected"]["fact"] == 1 and sum(refused["report"]["resurrected"].values()) == 1
    world.abort_to_h(compaction_id)
    [started] = [event for event in world.log(compaction_id)["events"] if event["kind"] == "abort_started"]
    head = world.head()
    h_after = h_tree()
    h_changed = sorted(key for key in set(h_before) | set(h_after) if h_before.get(key) != h_after.get(key))
    rec.result(abort={"w_class": started["data"]["w"]["class"], "head": head["status"],
                      "head_is_the_adopted_h": head["binding_id"] == adopted["binding_id"],
                      "h_state_entries": len(h_before), "h_state_changed": h_changed})
    assert len(h_before) > 0
    assert h_changed == [], h_changed  # paths only: the witness itself holds the state's bytes
    assert started["data"]["w"]["class"] == compaction.W_EQUAL
    assert head["status"] == "closed" and head["binding_id"] == adopted["binding_id"]
    assert world.log(compaction_id)["terminal"] and world.kinds(compaction_id)[-1] == "aborted"
    assert not (world.root / "slots" / k).exists()
    # Paso 0 on the H it returned to still counts the resurrected fact. The service stays down by the operator
    # procedure (runbook «Después de la aceptación», a written decision every 48 hours) until K's baseline
    # exists, not by a product gate; the unit check below only records that nothing here started it.
    completion = world.st("h-completion-status", "completion-status")
    unit = world.systemd.raw_show(world.d.unit, "ActiveState", "UnitFileState")
    rec.result(h_completion={"resurrected": completion["resurrected"], "complete": completion["complete"]},
               unit_not_started_here=unit, service_down_by="operator procedure (runbook «Después de la "
                                                            "aceptación»), not enforced by the system")
    assert completion["resurrected"]["fact"] == 1 and completion["complete"] is False, completion
    assert unit == {"ActiveState": "inactive", "UnitFileState": "disabled"}, unit
