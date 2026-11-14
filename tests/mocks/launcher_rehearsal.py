"""Helpers for the U3b-0d managed-launcher tests and compiled rehearsals.

Artifacts come from a temporary shared clone, never from the checkout's refs
(the clone-overlay pattern of test_compiled_stop_contract): the working
tree's scripts/ and src/ (tracked and new files) are committed only in the
clone and compiled with scripts/build_runtime.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime

REPO = Path(__file__).resolve().parents[2]
CHILD = REPO / "tests" / "mocks" / "launcher_child.py"
SETTINGS = "schema: launcher-rehearsal\nlogging:\n  level: INFO\nmemory:\n  db_path: ./data/memory.db\n"


def _git(cwd: Path, *arguments: str, env: dict | None = None) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True,
                          timeout=120, env=env).stdout.strip()


def overlay_commit(repo: Path, clone: Path, *, message: str = "u3b-0d rehearsal") -> str:
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
                       GIT_AUTHOR_DATE="2026-11-12T00:00:00Z", GIT_COMMITTER_DATE="2026-11-12T00:00:00Z")
    return _git(clone, "commit-tree", tree, "-p", "HEAD", "-m", message, env=environment)


def build_artifacts(base: Path) -> dict:
    """X = HEAD + working tree; Y = a trivial descendant of X; P = HEAD's parent chain base (pre-0d).

    KZA_LAUNCHER_ARTIFACTS_CACHE (mutant runs only) names a directory with
    prebuilt x/y/pre archives and commits.json, reused instead of rebuilding
    when the mutated code is controller-side only.
    """
    cache = os.environ.get("KZA_LAUNCHER_ARTIFACTS_CACHE")
    if cache:
        directory = Path(cache)
        commits = json.loads((directory / "commits.json").read_text())
        archives = {name: directory / (name + ".tar.gz") for name in ("x", "y", "pre")
                    if (directory / (name + ".tar.gz")).exists()}
        return {"archives": archives, "commits": commits, "clone": None}
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    x = overlay_commit(REPO, clone)
    environment = dict(os.environ, GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
                       GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid",
                       GIT_AUTHOR_DATE="2026-11-12T00:00:01Z", GIT_COMMITTER_DATE="2026-11-12T00:00:01Z")
    y = _git(clone, "commit-tree", x + "^{tree}", "-p", x, "-m", "u3b-0d rehearsal Y", env=environment)
    pre = pre_launcher_revision(clone, x)
    archives = {}
    for name, revision in (("x", x), ("y", y), ("pre", pre)):
        archive = base / (name + ".tar.gz")
        manifest = build_runtime(clone, archive, revision)
        assert manifest["git_commit"] == revision
        archives[name] = archive
    return {"archives": archives, "commits": {"x": x, "y": y, "pre": pre}, "clone": clone}


# The newest commit before U3b-0d (6288e15^). Pinned so the pre-launcher rehearsals never
# skip as the branch grows; a bounded ancestor search silently lost them once it fell out of range.
PRE_LAUNCHER_COMMIT = "40aa05b8d56ffb608e79f3c26b55a9386a9a0605"


def pre_launcher_revision(clone: Path, x: str) -> str:
    """The pinned pre-0d ancestor of X; it must be an ancestor and declare no LAUNCHER_CONTRACT."""
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PRE_LAUNCHER_COMMIT, x], cwd=clone,
                              capture_output=True, timeout=60)
    assert ancestor.returncode == 0, "the pinned pre-0d commit is not an ancestor of the rehearsal commit"
    source = subprocess.run(["git", "show", PRE_LAUNCHER_COMMIT + ":scripts/stage_runtime.py"], cwd=clone,
                            capture_output=True, timeout=60, check=True)
    assert b"\nLAUNCHER_CONTRACT = " not in source.stdout, "the pinned pre-0d commit declares a launcher contract"
    return PRE_LAUNCHER_COMMIT


def offline_source(tmp_path: Path, settings: str = SETTINGS) -> Path:
    """An offline state source with config/ and data/ (no links)."""
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text(settings)
    (source / "data" / "notes.json").write_text('{"v": 1}')
    return source


def managed_root(archive: Path, tmp_path: Path, *, slot: str = "s", settings: str = SETTINGS,
                 controller: bool = True) -> tuple[Path, Path]:
    """Prepare and activate one slot (bootstrap head), then record the controller install."""
    staging = tmp_path / "root"
    stage.prepare_slot(archive, staging, slot, offline_source(tmp_path, settings))
    root = staging.resolve()
    with stage.AdmissionController(root) as admission:  # a managed bootstrap head (U3b-0e's job in production)
        assert admission.activate(slot, managed=True) is None
    if controller:
        launcher.install_controller(root, archive)
    return root, root / "slots" / slot / "state"


def child_command(phase_file: Path, scenario: str, *, seconds: float = 0.0, touch: Path | None = None,
                  execute: Path | None = None, extend_every: float = 0.0) -> list[str]:
    """argv of the rehearsal child (cwd = the slot runtime)."""
    command = [sys.executable, "-I", "-B", str(CHILD), "--scenario", scenario, "--phase-file", str(phase_file),
               "--seconds", str(seconds)]
    if touch is not None:
        command += ["--touch", str(touch)]
    if execute is not None:
        command += ["--exec", str(execute)]
    if extend_every:
        command += ["--extend-every", str(extend_every)]
    return command


def table_ddl(name: str) -> str:
    """CREATE TABLE with the exact admission column contract of one registered table."""
    columns = ", ".join(" ".join(item.split(":")) for item in stage._TABLE_COLUMNS[name].split())
    return f"CREATE TABLE {name} ({columns})"


def phases(path: Path) -> list[tuple[str, str]]:
    """(name, value) lines the rehearsal child wrote."""
    if not path.exists():
        return []
    return [tuple((line.split(" ", 1) + [""])[:2]) for line in path.read_text().splitlines()]


def phase(path: Path, name: str) -> str | None:
    """The first value of one phase, or None."""
    return next((value for key, value in phases(path) if key == name), None)


class FakeManager:
    """ServiceManager double: configurable budget and system state, ordered call log."""

    def __init__(self, budget: float | None = 300.0, state: str = "running", events: list | None = None,
                 properties: dict | None = None) -> None:
        self.budget = budget
        self.state = state
        self.events = events if events is not None else []
        self.properties = dict(properties or {})
        self.reloads = 0

    def stop_budget(self, uid: int) -> float | None:
        self.events.append("budget")
        return self.budget

    def system_state(self) -> str | None:
        self.events.append("system_state")
        return self.state

    def user_show(self, unit: str, properties: tuple[str, ...]) -> dict | None:
        self.events.append("show " + unit)
        return {name: self.properties.get(name, "") for name in properties}

    def daemon_reload(self) -> bool:
        self.reloads += 1
        self.events.append("daemon-reload")
        return True


@dataclass
class ServeResult:
    """One in-process serve run."""

    code: int
    serve: launcher.Serve
    phases: list = field(default_factory=list)
    seconds: float = 0.0


def run_serve(root: Path, command: list[str], *, phase_file: Path, stop_when=None, stop_grace: float = 20.0,
              manager: FakeManager | None = None, recover: str = "bounded", unit_timeout_stop: float = 60.0,
              boot_id_file: Path | None = None, timeout: float = 90.0, environ: dict | None = None,
              wall=time.time) -> ServeResult:
    """Run Serve in this process; ``stop_when(phases)`` decides when to SIGTERM this process.

    After ``timeout`` seconds a still-running Serve is stopped anyway, so a
    run expected to be refused cannot block the test forever.

    The stop is a real SIGTERM to the test process: the controller's own
    handler (installed by Serve on the main thread) must turn it into a byte.
    """
    notifier = launcher.SystemdNotifier(None)
    options = launcher.ServeOptions(root=root, stop_grace=stop_grace, recover=recover,
                                    unit_timeout_stop=unit_timeout_stop, child_command=command,
                                    boot_id_path=str(boot_id_file) if boot_id_file else None, probe_timeout=60)
    if environ is None:
        environ = {key: value for key, value in os.environ.items() if key != "NOTIFY_SOCKET"}
    serve = launcher.Serve(options, manager=manager or FakeManager(), notifier=notifier, environ=environ,
                           wall=wall, straggler_procs=None)
    done = threading.Event()

    def stopper():
        deadline = time.monotonic() + timeout
        while not done.is_set() and time.monotonic() < deadline:
            if stop_when is not None and stop_when(phases(phase_file)):
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.02)
        if not done.is_set():  # fail-safe: a run that should have ended (e.g. a refusal) never hangs the suite
            os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=stopper, daemon=True)
    started = time.monotonic()
    # A late SIGTERM (after Serve restored the handlers) must not kill the test process.
    previous = signal.signal(signal.SIGTERM, lambda *_args: None)
    thread.start()
    try:
        code = serve.run()
    finally:
        done.set()
        thread.join(timeout=5)
        signal.signal(signal.SIGTERM, previous)
    return ServeResult(code, serve, phases(phase_file), time.monotonic() - started)


def receipts(root: Path) -> list[dict]:
    """Every receipt under admission/receipts."""
    directory = root / "admission" / "receipts"
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))] if directory.exists() else []
