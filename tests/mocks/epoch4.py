"""SC-1a test helper: epoch-4 builds of this tree for worlds with owners.

From controller epoch 5 (sc-release-design v4 mechanism.authz3_1) no
delegation record or owner is honoured, so an owner chain can neither be
built nor served by this code as it ships. Owner roots exist only in tests
and rehearsals at epoch <= 4 (mechanism.mode). Tests whose subject is the
validation, serving or compaction of an existing owner chain therefore run
an epoch-4 build of this same tree: the compiled slots and controller come
from a commit whose only difference is the literal ``CONTROLLER_EPOCH = 4``,
and the in-process controller runs with stage_runtime.CONTROLLER_EPOCH = 4
(the ``epoch4_controller`` fixture of tests/conftest.py). Every epoch-5
refusal is tested against the real epoch-5 build (test_sc1a_*).
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import tempfile

EPOCH5_LINE = b"\nCONTROLLER_EPOCH = 5\n"
EPOCH4_LINE = b"\nCONTROLLER_EPOCH = 4\n"
_ENVIRONMENT = {"GIT_AUTHOR_NAME": "rehearsal", "GIT_AUTHOR_EMAIL": "rehearsal@invalid",
                "GIT_COMMITTER_NAME": "rehearsal", "GIT_COMMITTER_EMAIL": "rehearsal@invalid",
                "GIT_AUTHOR_DATE": "2026-11-16T00:00:00Z", "GIT_COMMITTER_DATE": "2026-11-16T00:00:00Z"}


def _git(cwd: Path, *arguments: str, env: dict | None = None, data: bytes | None = None) -> str:
    return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, timeout=120,
                          env=env, input=data).stdout.decode().strip()


def edited_commit(clone: Path, revision: str, edits: dict, *, message: str) -> str:
    """A child commit of ``revision`` (in ``clone``) with each {path: [(old, new), ...]} replaced once."""
    with tempfile.TemporaryDirectory(prefix="edited-index-") as directory:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        _git(clone, "read-tree", revision, env=environment)
        for path, replacements in sorted(edits.items()):
            source = subprocess.run(["git", "show", revision + ":" + path], cwd=clone, check=True,
                                    capture_output=True, timeout=60).stdout
            for old, new in replacements:
                assert source.count(old) == 1, (path, old)
                source = source.replace(old, new)
            blob = _git(clone, "hash-object", "-w", "--stdin", "--path", path, data=source)
            _git(clone, "update-index", "--add", "--cacheinfo", "100644," + blob + "," + path, env=environment)
        tree = _git(clone, "write-tree", env=environment)
    return _git(clone, "commit-tree", tree, "-p", revision, "-m", message, env=dict(os.environ, **_ENVIRONMENT))


def epoch4_commit(clone: Path, revision: str) -> str:
    """A child commit of ``revision`` (in ``clone``) whose stage_runtime declares CONTROLLER_EPOCH = 4."""
    return edited_commit(clone, revision, {"scripts/stage_runtime.py": [(EPOCH5_LINE, EPOCH4_LINE)]},
                         message="sc-1a epoch-4 build")


def epoch4_checkout(base: Path, repo: Path, revision: str) -> tuple[Path, str]:
    """A shared clone of ``repo`` at ``base``/checkout, checked out at the epoch-4 build of ``revision``.

    For rehearsals that read files from a checkout as well as build from its commits.
    """
    clone = base / "checkout"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)],
                   check=True, timeout=120)
    built = epoch4_commit(clone, revision)
    subprocess.run(["git", "checkout", "--quiet", "--detach", built], cwd=clone, check=True, timeout=300)
    return clone.resolve(), built


def installed_worker_command(original):
    """worker_command running the installed controller tree (the epoch-4 build), not this checkout."""
    def worker_command(root, compaction_id):
        argv = original(root, compaction_id)
        installed = Path(root) / "controller" / "current"
        if installed.is_dir():
            argv[argv.index("--root") - 1] = str(installed.resolve())
        return argv
    return worker_command


def apply_epoch4(monkeypatch) -> None:
    """The in-process controller as the epoch-4 build: CONTROLLER_EPOCH = 4, worker from controller/current."""
    from scripts import compaction
    from scripts import stage_runtime as stage
    from src.core import runtime_admission
    monkeypatch.setattr(stage, "CONTROLLER_EPOCH", 4)
    # An in-process worker reaches the controller through runtime_admission's path-loaded companion.
    monkeypatch.setattr(runtime_admission._stage(), "CONTROLLER_EPOCH", 4)
    monkeypatch.setattr(compaction, "worker_command", installed_worker_command(compaction.worker_command))


@contextmanager
def epoch4_in_process(*, old_copy: bool = False):
    """Run the block as the epoch-4 controller; ``old_copy`` also drops require_forward_epoch, as the
    stage_runtime copies of 69a79e6..b0c322d~1 never check it (AUTHZ3-1), and enables their grants."""
    import pytest

    from scripts import stage_runtime as stage
    with pytest.MonkeyPatch.context() as patch:
        apply_epoch4(patch)
        if old_copy:
            patch.setattr(stage, "require_forward_epoch", lambda root, **kwargs: None)
            patch.setattr(stage, "LEGACY_MANAGED_TRANSITIONS", True)
        yield stage
