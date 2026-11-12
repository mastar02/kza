"""U3b-0d1 compiled rehearsal: real SIGTERMs against the managed entry under real admission.

The working tree (including untracked runtime sources) is committed only in a
temporary shared clone and compiled with scripts/build_runtime.py. Each case
prepares a fresh slot, runs the compiled child through AdmissionController.run
and sends SIGTERM when the child reports a phase. A clean stop must end with
exit 0, a receipt the controller accepts and a closed head.
"""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest
import yaml

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import compiled_restore as compiled
from tests.mocks.managed_slot_graph import _get, _leaves, canonical_template
from tests.mocks.restore_application_graph import configuration as restore_configuration

REPO = Path(__file__).resolve().parents[2]
CHILD = REPO / "tests/mocks/stop_contract_child.py"
MODULE_CHILD = REPO / "tests/mocks/stop_contract_module_child.py"
HOME, MODEL = "http://slot-home.invalid", "http://slot-model.invalid"
RUN_SECONDS = 120


def _git(cwd, *arguments, **kwargs):
    return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True,
                          timeout=120, **kwargs).stdout.strip()


def _overlay_commit(repo: Path, clone: Path) -> str:
    """Commit, in the clone only, the working tree's scripts/ and src/ (tracked and new)."""
    changed = set(_git(repo, "diff", "--name-only", "HEAD", "--", "scripts", "src").split())
    changed |= set(_git(repo, "ls-files", "--others", "--exclude-standard", "--", "scripts", "src").split())
    if not changed:
        return _git(clone, "rev-parse", "HEAD")
    _git(clone, "read-tree", "HEAD")
    for relative in sorted(changed):
        if (repo / relative).exists():
            blob = _git(repo, "hash-object", "-w", "--path", relative, str(repo / relative))
            _git(clone, "update-index", "--add", "--cacheinfo", "100644," + blob + "," + relative)
        else:
            _git(clone, "update-index", "--remove", relative)
    tree = _git(clone, "write-tree")
    environment = dict(os.environ, GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
                       GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid",
                       GIT_AUTHOR_DATE="2026-11-11T00:00:00Z", GIT_COMMITTER_DATE="2026-11-11T00:00:00Z")
    return _git(clone, "commit-tree", tree, "-p", "HEAD", "-m", "u3b-0d1 rehearsal", env=environment)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    task = tmp_path_factory.mktemp("stop-contract-artifact")
    clone = task / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    revision = _overlay_commit(REPO, clone)
    archive = task / "runtime.tar.gz"
    manifest = build_runtime(clone, archive, revision=revision)
    assert manifest["git_commit"] == revision
    assert "src/core/process_lifecycle.pyc" in manifest["files"]
    return archive


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Slot:
    """One prepared, activated slot and the observations of one managed run."""

    def __init__(self, archive: Path, tmp_path: Path, *, entry: str = "harness"):
        self.tmp = tmp_path
        self.entry = entry
        self.port = _free_port()
        self.staging = (tmp_path / "staging")
        self.staging.mkdir()
        self.staging = self.staging.resolve()
        self.root = self.staging / "slots/s/state/data/graph"
        source = tmp_path / "offline"
        for name in ("config", "data"):
            (source / name).mkdir(parents=True)

        def configuration(root, home, model):
            config = restore_configuration(Path(root), home, model)
            config["dashboard"].update(enabled=True, host="127.0.0.1", port=self.port)
            return config

        raw = canonical_template(self.root, configuration)
        (source / "config/settings.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        # Placeholder -> value with the placeholder URLs, for the module-entry child.
        generated = configuration(self.root, HOME, MODEL)
        self.bindings = tmp_path / "bindings.json"
        self.bindings.write_text(json.dumps({
            value[2:-1]: _get(generated, location) for location, value in _leaves(raw)
            if type(value) is str and value.startswith("${KZA_SLOT_BIND_")}))
        stage.prepare_slot(archive, self.staging, "s", source)
        assert stage.activate_slot(self.staging, "s") is None
        self.root.mkdir(mode=0o700)
        self.phase_file = tmp_path / "phases.txt"
        self.phase_file.touch()
        self.fifo = tmp_path / "notify.fifo"
        os.mkfifo(self.fifo, 0o600)
        self.stdout = tmp_path / "child-stdout.log"
        self.stderr = tmp_path / "child-stderr.log"
        self.notify: list[str] = []
        self.signals: list[float] = []

    def phases(self) -> list[tuple[str, str]]:
        return [tuple((line.split(" ", 1) + [""])[:2]) for line in self.phase_file.read_text().splitlines()]

    def phase(self, name: str) -> str | None:
        return next((value for key, value in self.phases() if key == name), None)

    def run(self, scenario: str, trigger, *, count: int = 1, interval: float = 0.0,
            timeout: float = RUN_SECONDS) -> dict:
        """Run the child; ``trigger(slot)`` decides when to send ``count`` SIGTERMs."""
        if self.entry == "module":
            command = [sys.executable, "-I", "-B", str(MODULE_CHILD), "--repo", str(REPO),
                       "--phase-file", str(self.phase_file), "--notify-fifo", str(self.fifo),
                       "--graph-root", str(self.root), "--bindings", str(self.bindings)]
        else:
            command = [sys.executable, "-I", "-B", str(CHILD), "--repo", str(REPO), "--scenario", scenario,
                       "--phase-file", str(self.phase_file), "--notify-fifo", str(self.fifo),
                       "--graph-root", str(self.root), "--dashboard-port", str(self.port)]
        reader = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        outcome: dict = {}

        def execute():
            try:
                with self.stdout.open("w") as out, self.stderr.open("w") as err:
                    outcome["completed"] = stage.run_slot(self.staging, "s", command, timeout=timeout,
                                                          env=compiled._environment(), stdout=out, stderr=err)
            except BaseException as error:
                outcome["error"] = error
            outcome["ended"] = time.monotonic()

        runner = threading.Thread(target=execute)
        runner.start()
        buffer = b""
        try:
            deadline = time.monotonic() + timeout + 15
            while runner.is_alive() and time.monotonic() < deadline:
                try:
                    buffer += os.read(reader, 4096)
                except BlockingIOError:
                    pass
                *lines, buffer = buffer.split(b"\n")
                self.notify.extend(line.decode() for line in lines)
                if not self.signals and trigger(self):
                    pid = int(self.phase("pid"))
                    for number in range(count):
                        if number:
                            # Pending standard signals coalesce; space them out.
                            time.sleep(interval)
                        self.signals.append(time.monotonic())
                        os.kill(pid, signal.SIGTERM)
                time.sleep(0.01)
            runner.join(timeout=30)
            assert not runner.is_alive()
            try:
                buffer += os.read(reader, 4096)
            except BlockingIOError:
                pass
            self.notify.extend(line.decode() for line in buffer.split(b"\n") if line)
        finally:
            os.close(reader)
        # Some fixture imports route logging to stdout; read both streams.
        outcome["log"] = self.stdout.read_text() + self.stderr.read_text()
        return outcome

    def receipts(self) -> list[dict]:
        directory = self.staging / "admission/receipts"
        return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))] if directory.exists() else []


def _at(name):
    return lambda slot: slot.phase(name) is not None


def _steady(slot):
    return slot.phase("steady") is not None and "READY=1" in slot.notify


def _assert_clean(slot, outcome, *, phase, ready, system_ready=None):
    assert "error" not in outcome, (outcome.get("error"), outcome["log"][-4000:])
    assert outcome["completed"].returncode == 0
    head = stage.read_admission_head(slot.staging)
    assert head["status"] == "closed"
    receipts = slot.receipts()
    assert len(receipts) == 1 and receipts[0]["status"] == "closed"
    assert receipts[0]["child_pid"] == int(slot.phase("pid"))
    assert slot.phase("mask") == "empty"
    assert f"[Stop] stop requested by SIGTERM during phase={phase}" in outcome["log"]
    assert "[Stop] closure receipt next; closing logging" in outcome["log"]
    assert slot.phase("returned") is None  # os._exit after the receipt
    assert ("READY=1" in slot.notify) is ready
    assert slot.notify[-1] == "STOPPING=1"
    if ready:
        assert slot.notify.index("READY=1") < slot.notify.index("STOPPING=1")
    if ready if system_ready is None else system_ready:
        assert outcome["log"].index("System ready") < outcome["log"].index("[Stop] stop requested")
    return outcome["ended"] - slot.signals[0]


@pytest.mark.parametrize("scenario, phase", [
    ("entry", "entry"),
    ("construction", "construction"),
    ("warmup", "warmup"),
    ("services", "services"),
    ("pipeline", "pipeline"),
])
def test_sigterm_during_start_up_closes_with_receipt_and_exit_zero(artifact, tmp_path, scenario, phase):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run(scenario, _at(scenario))
    _assert_clean(slot, outcome, phase=phase, ready=False)
    assert slot.phase("steady") is None
    if scenario in {"entry", "construction"}:
        # Construction finishes, then warmup and voice are skipped.
        assert "[Stop] construction finished after a stop; skipping warmup and voice" in outcome["log"]
        assert slot.phase("warmup_entered") is None
    elif scenario in {"services", "pipeline"}:
        assert slot.phase("warmup_entered") is not None


def test_sigterm_in_steady_state_closes_with_receipt_and_exit_zero(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("ready", _steady)
    _assert_clean(slot, outcome, phase="ready", ready=True)


def test_two_sigterms_with_the_real_dashboard_are_one_stop(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("double", _steady, count=2, interval=0.3)
    _assert_clean(slot, outcome, phase="ready", ready=True)
    assert "Dashboard API starting" in outcome["log"]
    assert outcome["log"].count("[Stop] stop requested by") == 1
    assert "[Stop] repeated stop request ignored (count=2)" in outcome["log"]


def test_sigterm_while_capture_is_blocked(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("blocked", lambda slot: slot.phase("blocked") is not None and "READY=1" in slot.notify)
    _assert_clean(slot, outcome, phase="capture_blocked", ready=True, system_ready=False)
    # U1: degraded but serving -> READY plus a capture-blocked STATUS, once.
    assert slot.notify == ["READY=1", "STATUS=capture blocked", "STOPPING=1"]
    assert "System ready" not in outcome["log"]


def test_sigterm_between_ready_and_audio_loop_run(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("audio_start", _at("audio_start"))
    _assert_clean(slot, outcome, phase="ready", ready=True)
    assert slot.phase("steady") is None  # audio_loop.run never started


def test_sigterm_during_module_import_is_deferred(artifact, tmp_path):
    slot = Slot(artifact, tmp_path, entry="module")
    outcome = slot.run("import", _at("importing"))
    _assert_clean(slot, outcome, phase="entry", ready=False)
    # The stop arrived while src.main was still importing src.application.
    assert slot.phase("imported") is not None
    assert slot.phases().index(("imported", "")) > slot.phases().index(("importing", ""))
    assert "[Stop] construction finished after a stop; skipping warmup and voice" in outcome["log"]


def test_receipt_then_exit_without_joining_an_in_flight_executor_job(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("executor", _steady)
    elapsed = _assert_clean(slot, outcome, phase="ready", ready=True)
    # The job sleeps 60 s; asyncio.run would join it after main() returns.
    assert elapsed < 30, elapsed


@pytest.mark.parametrize("scenario", ["sync", "sync_encode"])
def test_sigterm_during_a_command_sync_aborts_it_and_writes_the_receipt(artifact, tmp_path, scenario):
    """U3b-0d2 S2: chroma.aclose (1 s bound) aborts the sync instead of timing out.

    "sync": the cloud call is cancelled. "sync_encode": the owned worker (300
    encodes of 100 ms) stops at its next phrase, so the close drains in time.
    """
    slot = Slot(artifact, tmp_path)
    outcome = slot.run(scenario, lambda slot: slot.phase("syncing") is not None and "READY=1" in slot.notify)
    elapsed = _assert_clean(slot, outcome, phase="ready", ready=True)
    assert slot.phase("sync_ended") == "chroma_sync_aborted"
    # The HTTP reasoner owner waits (unbounded) for its chroma borrower before
    # chroma.aclose runs: an unaborted sync would stretch the stop by its
    # remaining work (about 30 s of encodes) toward stop_grace.
    print(f"stop seconds {scenario}: {elapsed:.2f}")
    assert elapsed < 10, elapsed
    assert "chroma_sync_still_running" not in outcome["log"]


def test_hung_stop_never_writes_a_receipt(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("hung", _steady, timeout=40)
    assert isinstance(outcome.get("error"), subprocess.TimeoutExpired), outcome.get("error")
    assert slot.phase("hanging") is not None
    assert "[Stop] stop requested by SIGTERM during phase=ready" in outcome["log"]
    # The controller's own retirement SIGTERM is a repeat, not a second stop.
    assert "[Stop] repeated stop request ignored" in outcome["log"]
    assert stage.read_admission_head(slot.staging)["status"] == "unknown"
    assert slot.receipts() == []


def test_open_writer_under_the_state_refuses_the_receipt(artifact, tmp_path):
    slot = Slot(artifact, tmp_path)
    outcome = slot.run("writer", _steady)
    error = outcome.get("error")
    assert isinstance(error, subprocess.CalledProcessError) and error.returncode != 0, error
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        assert "[Admission] open writer under the state: data/graph/held-writer.bin" in outcome["log"]
        assert "runtime_admission_open_writer:data/graph/held-writer.bin" in outcome["log"]
    assert stage.read_admission_head(slot.staging)["status"] == "unknown"
    assert slot.receipts() == []
