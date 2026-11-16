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


def epoch4_commit(clone: Path, revision: str) -> str:
    """A child commit of ``revision`` (in ``clone``) whose stage_runtime declares CONTROLLER_EPOCH = 4."""
    source = subprocess.run(["git", "show", revision + ":scripts/stage_runtime.py"], cwd=clone, check=True,
                            capture_output=True, timeout=60).stdout
    assert source.count(EPOCH5_LINE) == 1, "the revision does not declare CONTROLLER_EPOCH = 5"
    blob = _git(clone, "hash-object", "-w", "--stdin", "--path", "scripts/stage_runtime.py",
                data=source.replace(EPOCH5_LINE, EPOCH4_LINE))
    with tempfile.TemporaryDirectory(prefix="epoch4-index-") as directory:
        environment = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        _git(clone, "read-tree", revision, env=environment)
        _git(clone, "update-index", "--add", "--cacheinfo", "100644," + blob + ",scripts/stage_runtime.py",
             env=environment)
        tree = _git(clone, "write-tree", env=environment)
    return _git(clone, "commit-tree", tree, "-p", revision, "-m", "sc-1a epoch-4 build",
                env=dict(os.environ, **_ENVIRONMENT))
