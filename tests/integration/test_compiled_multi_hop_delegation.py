"""U3b-0b compiled rehearsal: release, rollback and roll-forward across real artifacts.

X is the committed HEAD (pre-0b controller code inside the slot); Y is a commit
made only in a temporary shared clone that carries this working tree's
scripts/ and src/ changes (0b code inside the slot). Slots A(X) -> B(Y) ->
C(X, rollback) -> D(Y, roll-forward) each run a real admitted child that
validates its ownership with its own compiled code, after the candidate's own
compiled probe. Delegation uses Y's compiled stage_runtime CLI.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.unit.tools.test_stage_runtime_admission import (
    _CLOSING_CHILD,
    _TARGETS_DDL,
    _command,
    _registration,
    _witness,
)

_INSERT = "INSERT INTO erasure_targets VALUES(?,?,?,?)"


def _git(clone, *arguments, **kwargs):
    return subprocess.run(["git", *arguments], cwd=clone, check=True, capture_output=True, text=True,
                          timeout=120, **kwargs).stdout.strip()


def _working_tree_commit(repo: Path, clone: Path) -> str:
    """Commit, in the clone only, every modified runtime source of the working tree."""
    changed = _git(repo, "diff", "--name-only", "HEAD", "--", "scripts", "src").split()
    assert "scripts/stage_runtime.py" in changed
    _git(clone, "read-tree", "HEAD")
    for relative in changed:
        blob = _git(clone, "hash-object", "-w", "--path", relative, str(repo / relative))
        _git(clone, "update-index", "--add", "--cacheinfo", "100644," + blob + "," + relative)
    tree = _git(clone, "write-tree")
    environment = dict(os.environ, GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
                       GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid",
                       GIT_AUTHOR_DATE="2026-11-11T00:00:00Z", GIT_COMMITTER_DATE="2026-11-11T00:00:00Z")
    return _git(clone, "commit-tree", tree, "-p", "HEAD", "-m", "u3b-0b rehearsal", env=environment)


def _delegate(root: Path, controller_pyc: Path, name: str) -> dict:
    result = subprocess.run([sys.executable, "-B", str(controller_pyc), "--root", str(root), "--offline",
                             "delegate", name], cwd=controller_pyc.parents[1], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _run(root: Path, name: str, rows: list) -> str:
    statements = [["data/privacy.db", statement, row] for statement, row in rows]
    stage.run_slot(root, name, _command(_CLOSING_CHILD, json.dumps(statements)), timeout=60)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    return head["binding_id"]


def _local(state: Path, name: str) -> list:
    directory = state / "data" / ("clips-" + name)
    directory.mkdir()
    return _registration("collectors", 1, {"kind": "wake_clips", "root": str(directory)})


def test_release_rollback_and_roll_forward_across_compiled_artifacts(tmp_path):
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)],
                   check=True, timeout=120)
    x = _git(clone, "rev-parse", "HEAD")
    y = _working_tree_commit(repo, clone)
    archives = {"X": tmp_path / "x.tar.gz", "Y": tmp_path / "y.tar.gz"}
    assert build_runtime(clone, archives["X"], revision=x)["git_commit"] == x
    assert build_runtime(clone, archives["Y"], revision=y)["git_commit"] == y
    built = time.perf_counter()

    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: multi-hop-rehearsal\n")
    root = tmp_path / "staging"
    stage.prepare_slot(archives["X"], root, "a", source)
    root = root.resolve()
    states = {"a": root / "slots/a/state"}
    pathless = _registration("producers", 1, {"kind": "runtime_producers", "protocol": 1})
    owned = {"a": _local(states["a"], "a")}
    bindings = {"a": _run(root, "a", [(_TARGETS_DDL, []), (_INSERT, owned["a"]), (_INSERT, pathless)])}
    outputs, previous = {}, "a"
    for name, release in (("b", "Y"), ("c", "X"), ("d", "Y")):
        stage.prepare_slot(archives[release], root, name, states[previous])
        states[name] = root / "slots" / name / "state"
        outputs[name] = _delegate(root, root / "slots/b/runtime/scripts/stage_runtime.pyc", name)
        owned[name] = _local(states[name], name)
        assert stage.activate_slot(root, name) == "slots/" + previous + "/runtime"
        bindings[name] = _run(root, name, [(_INSERT, owned[name])])
        previous = name

    # Every inherited target names its physical owner; the chain records each hop.
    chains = {}
    for record in stage._delegation_records(root, states["d"]):
        for identity, hops in stage._read_chain(root, record)["chains"].items():
            chains[identity] = (record["source_binding"], [hop["binding_id"] for hop in hops])
    assert chains == {owned["a"][0]: (bindings["a"], [bindings["a"], bindings["b"], bindings["c"]]),
                      owned["b"][0]: (bindings["b"], [bindings["b"], bindings["c"]]),
                      owned["c"][0]: (bindings["c"], [bindings["c"]])}
    policy = json.loads((states["d"] / ".kza-target-ownership.json").read_text())
    assert set(policy["owned_targets"]) == {pathless[0], owned["d"][0]}
    for name, release in (("a", x), ("b", y), ("c", x), ("d", y)):
        assert json.loads((root / "slots" / name / "runtime/release.json").read_text())["git_commit"] == release

    # A cycle back into the rollback slot C (a hop of every inherited chain) is
    # refused with nothing changed.
    head = stage.read_admission_head(root)
    before = _witness(root, *states.values())
    with stage.AdmissionController(root) as controller:
        try:
            controller.authorize_target_delegation(states["c"], head["binding_id"],
                                                   sorted(chains))
        except ValueError as error:
            assert str(error) == "runtime_admission_delegation_cycle"
        else:
            raise AssertionError("cycle accepted")
    assert _witness(root, *states.values()) == before
    elapsed = time.perf_counter() - started
    print(json.dumps({"build_seconds": round(built - started, 1), "total_seconds": round(elapsed, 1),
                      "x": x, "y": y}))
