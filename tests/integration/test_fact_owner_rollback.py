"""U3b-1a rollback rehearsals on compiled slots (design v5 acceptance, review 1a-F2).

Artifacts come from a temporary clone, never from the checkout's refs:
X = the working tree committed in the clone (carries the owner schema), Y = an
empty descendant of X (different code commit, same schema), P = the pre-U3b
parent located with git (the parent of the first commit that registers
fact_owner_schema, through ``git merge-base``). The controller is this
checkout's stage_runtime (U3b-0a probe and the U3b-0b multi-hop delegate).

(1) code rollback: A (X) runs admitted and writes owner rows inside the run;
    B (Y) is prepared from A's closed state, delegated, activated and sees
    the same owner rows.
(2) rejection point: the pre-U3b candidate, prepared from A's closed state and
    delegated, is refused by the probe with nothing changed.
(3) second hop (U3b-0b): after B closes, a rollback slot C (X) prepared from
    B's state is delegated, activated and run, and keeps the same owner rows.
"""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks.slot_rollback import is_managed_capable

REPO = Path(__file__).resolve().parents[2]
PRIVACY = "data/privacy.db"
TIMEOUT = 60

# Harness: inside a managed run the ledger normally opens under the application
# Graph's output-registration coordinator. These children open it directly, so
# the slot's compiled fact_ledger gets a no-op registration hook. Everything
# else (schema, assignment API, admission, closure, copy) is the slot's own code.
_OPEN_DIRECTLY = [
    "import importlib.util, json, os, sys, time, types, uuid",
    "sys.path.insert(0, os.getcwd())",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
]

# Runs inside an admitted child (cwd = the slot's runtime root, compiled code only).
# It writes owner rows through the slot's own compiled FactLedger API, then the
# owned-closure receipt: this proves the admission and copy path of owner rows,
# not application wiring.
_WRITER = "\n".join(_OPEN_DIRECTLY + [
    "from src.memory.erasure_journal import ErasureParticipant",
    "from src.memory.fact_ledger import (FactLedger, MemoryAssigner, MemoryOrigin, MemoryOwnerActor,",
    "    MemoryReviewer, MemorySource)",
    "state = os.path.realpath('data')",
    "os.makedirs(os.path.join(state, 'clips'), exist_ok=True)",
    "ledger = FactLedger(os.path.join(state, 'privacy.db'))",
    "ledger.initialize()",
    "def source():",
    "    now = time.time()",
    "    return MemorySource(str(uuid.uuid4()), str(uuid.uuid4()), now - 1, now + 3600, MemoryOrigin.VOICE)",
    "operator = MemoryAssigner('rehearsal-operator')",
    "allow = {'retired': lambda ids: False, 'active': lambda person: True}",
    "owned = ledger.store('Rehearsal automatic fact', 'fact', .9, (source(),))",
    "proposal = ledger.store('Rehearsal proposal', 'fact', 1.0, (source(),), proposed=True)",
    "household = ledger.confirm(ledger.store('Rehearsal household', 'fact', 1.0, (source(),),",
    "    proposed=True).id, MemoryReviewer('rehearsal-reviewer'))",
    "ledger.assign_owner(owned.id, 'user_rehearsal_a', owned.revision, 0, operator, **allow)",
    "ledger.assign_owner(proposal.id, 'user_rehearsal_b', proposal.revision, 0, operator, **allow)",
    "assert ledger.return_owned(proposal.id, 'user_rehearsal_b', MemoryOwnerActor('home', 'user_rehearsal_b'))",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': os.path.join(state, 'clips')})",
    "# The voice ledger owns this marker row; a bare FactLedger leaves it empty.",
    "ledger._db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "ledger.close()",
    "spec = importlib.util.spec_from_file_location('slot_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
])

# Reads the copied state through the slot's own compiled ledger (initialize
# validates owner shape and values) and reports the rows and counts.
_READER = "\n".join(_OPEN_DIRECTLY + [
    "from src.memory.fact_ledger import FactLedger, MemoryAssigner",
    "ledger = FactLedger(os.path.join(os.path.realpath('data'), 'privacy.db'))",
    "ledger.initialize()",
    "rows = {table: sorted(map(list, ledger._db.execute('SELECT * FROM ' + table)))",
    "        for table in ('fact_owners', 'fact_owner_revisions', 'fact_owner_schema')}",
    "rows['owned_counts'] = ledger.owned_counts(MemoryAssigner('rehearsal-operator')).to_dict()",
    "rows['household'] = sorted(ledger.eligible_ids(scope='household'))",
    "ledger.close()",
    "with open(sys.argv[1], 'x') as output:",
    "    json.dump(rows, output, sort_keys=True)",
    "spec = importlib.util.spec_from_file_location('slot_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
])
_CLOSING = "\n".join([
    "import importlib.util, os",
    "spec = importlib.util.spec_from_file_location('slot_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
])


def _git(clone, *arguments, env=None):
    return subprocess.check_output(["git", *arguments], cwd=clone, text=True, env=env, timeout=120).strip()


def _pre_u3b_parent(clone, x):
    """Parent of the first commit (in X's history) that registers fact_owner_schema, via merge-base."""
    first = _git(clone, "log", "--reverse", "--format=%H", "-S", '"fact_owner_schema"', x,
                 "--", "scripts/stage_runtime.py").split()
    if not first:
        pytest.skip("no commit in X's history registers fact_owner_schema")
    parents = _git(clone, "show", "-s", "--format=%P", first[0]).split()
    if not parents:
        pytest.skip("the owner registration has no parent commit in this clone")
    return _git(clone, "merge-base", x, parents[0])


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Build X, Y (descendant) and P (pre-U3b) from a temporary clone of this checkout."""
    started = time.monotonic()
    if subprocess.run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=REPO,
                      capture_output=True).returncode != 0:
        pytest.skip("git history unavailable")
    base = tmp_path_factory.mktemp("owner-rehearsal-artifacts")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    if _git(clone, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: the pre-U3b parent cannot be located")
    head = _git(REPO, "rev-parse", "HEAD")
    # Stage this checkout's working tree into the clone's object store through a
    # private index: nothing is written to the checkout's index, refs or objects.
    env = dict(os.environ, GIT_INDEX_FILE=str(base / "index"), GIT_DIR=str(clone / ".git"),
               GIT_WORK_TREE=str(REPO), GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
               GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid")
    _git(clone, "read-tree", head, env=env)
    _git(clone, "add", "-A", "--", "src", "scripts", "tools", env=env)
    tree = _git(clone, "write-tree", env=env)
    x = _git(clone, "commit-tree", tree, "-p", head, "-m", "U3b-1a rehearsal X (working tree)", env=env)
    y = _git(clone, "commit-tree", tree, "-p", x, "-m", "U3b-1a rehearsal Y (trivial descendant)", env=env)
    pre = _pre_u3b_parent(clone, x)
    shown = _git(clone, "show", pre + ":scripts/stage_runtime.py")
    assert '"fact_owner_schema"' not in shown and is_managed_capable(clone, pre)
    archives = {}
    for name, revision in (("x", x), ("y", y), ("pre", pre)):
        archive = base / (name + ".tar.gz")
        manifest = build_runtime(clone, archive, revision)
        assert manifest["git_commit"] == revision
        archives[name] = archive
    return {"archives": archives, "commits": {"x": x, "y": y, "pre": pre, "head": head},
            "build_seconds": time.monotonic() - started}


def _child(source, *arguments):
    return [sys.executable, "-I", "-B", "-c", source, *map(str, arguments)]


def _run(root, slot, argv):
    """One admitted child; a failing child reports its own stderr."""
    try:
        stage.run_slot(root, slot, argv, timeout=TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        pytest.fail("admitted child failed: " + (error.stderr or b"").decode(errors="replace")[-4000:])
    assert stage.read_admission_head(root)["status"] == "closed"


def _witness(root, *states):
    """What a refusal must leave byte-identical (as the U3b-0a tests define it)."""
    tree = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {".admission.lock", ".stage.lock"} or relative.startswith("slots/"):
            continue
        if path.is_symlink():
            tree[relative] = ("link", os.readlink(path))
        elif path.is_dir():
            tree[relative] = ("dir",)
        else:
            tree[relative] = ("file", path.read_bytes())
    files = {}
    for state in states:
        for name in (stage._MARKER, ".kza-target-ownership.json"):
            path = state / name
            files[str(path)] = path.read_bytes() if path.exists() else None
    return {"root": tree, "states": files}


def _owner_rows(state):
    with closing(sqlite3.connect((state / PRIVACY).as_uri() + "?mode=ro", uri=True)) as db:
        return {table: sorted(map(list, db.execute("SELECT * FROM " + table)))
                for table in ("fact_owners", "fact_owner_revisions", "fact_owner_schema")}


def _closed_a(artifacts, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: owner-rehearsal\n")
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(artifacts["archives"]["x"], root, "a", source)
    root = root.resolve()
    state = root / "slots" / "a" / "state"
    _run(root, "a", _child(_WRITER))
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    rows = _owner_rows(state)
    assert len(rows["fact_owners"]) == 1 and len(rows["fact_owner_revisions"]) == 2
    assert rows["fact_owner_schema"] == [[1]]
    return root, state, rows


def test_code_rollback_to_a_descendant_slot_keeps_owner_rows(artifacts, tmp_path):
    """Rehearsal (1): different code commit, same schema, one delegated hop."""
    commits = artifacts["commits"]
    assert commits["x"] != commits["y"]
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    manifest = stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    assert manifest["git_commit"] == commits["y"]
    assert json.loads((root / "slots/a/runtime/release.json").read_text())["git_commit"] == commits["x"]
    state_b = root / "slots" / "b" / "state"
    with stage.AdmissionController(root) as controller:
        before = _witness(root, state_a, state_b)
        with pytest.raises(ValueError, match="runtime_admission_foreign_target"):
            controller.activate("b")  # Delegation is a required step.
        assert _witness(root, state_a, state_b) == before
    delegated = stage.delegate_slot(root, "b")
    assert len(delegated["targets"]) == 1
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    assert os.readlink(root / "current") == "slots/b/runtime"
    assert _owner_rows(state_b) == rows
    output = tmp_path / "b-view.json"
    _run(root, "b", _child(_READER, output))
    assert stage.read_admission_head(root)["status"] == "closed"
    view = json.loads(output.read_text())
    assert {table: view[table] for table in rows} == rows
    assert view["owned_counts"] == {"currently_owned": 1, "returned": 1, "ever_assigned_live": 2,
                                    "owned_proposals": 0}
    assert len(view["household"]) == 1
    assert _owner_rows(state_b) == rows


def test_pre_u3b_candidate_is_refused_at_activation_with_nothing_changed(artifacts, tmp_path):
    """Rehearsal (2): prepare from A's closed state, delegate, then activate -> refused by the probe."""
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    manifest = stage.prepare_slot(artifacts["archives"]["pre"], root, "pre", state_a)
    assert manifest["git_commit"] == artifacts["commits"]["pre"]
    state_pre = root / "slots" / "pre" / "state"
    assert stage.delegate_slot(root, "pre")["targets"]
    head = stage.read_admission_head(root)
    before = _witness(root, state_a, state_pre)
    with pytest.raises(ValueError, match="runtime_admission_candidate_rejected: runtime_admission_unknown_schema"):
        stage.activate_slot(root, "pre")
    assert _witness(root, state_a, state_pre) == before
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/a/runtime"
    assert _owner_rows(state_a) == _owner_rows(state_pre) == rows


def test_second_hop_rollback_keeps_owner_rows(artifacts, tmp_path):
    """Rehearsal (3), inverted by U3b-0b: A(X) -> B(Y) -> C(X) is a second hop and works."""
    root, state_a, _ = _closed_a(artifacts, tmp_path)
    stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    stage.delegate_slot(root, "b")
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    _run(root, "b", _child(_CLOSING))
    assert stage.read_admission_head(root)["status"] == "closed"
    state_b = root / "slots" / "b" / "state"
    stage.prepare_slot(artifacts["archives"]["x"], root, "c", state_b)
    state_c = root / "slots" / "c" / "state"
    granted = stage.delegate_slot(root, "c")
    assert granted["targets"] and len(granted["capabilities"]) >= 1
    assert stage.activate_slot(root, "c") == "slots/b/runtime"
    _run(root, "c", _child(_CLOSING))
    assert stage.read_admission_head(root)["status"] == "closed"
    assert _owner_rows(state_c) == _owner_rows(state_b) == _owner_rows(state_a)
    assert os.readlink(root / "current") == "slots/c/runtime"
