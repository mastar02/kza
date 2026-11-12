"""U3b-0d3 rehearsals on compiled slots: deletion ledger and receipt schema 2.

Artifacts come from a temporary shared clone, never from the checkout's refs
(the test_voice_attribution_rollback pattern): X = HEAD plus this working tree
(new files included), committed only in the clone; Y = an empty descendant of
X; P = the parent of the first commit (in X's history) that registers
deletion_ledger_v1 in scripts/stage_runtime.py, i.e. the pre-0d3 controller
contract. P is chosen explicitly, so the schema_compatible_revision caveat of
the U3b-4a-0 review (F1) does not apply here.

(1) rejection point: A (X) records deletions through its own compiled ledger
    and closes with a schema-2 receipt; P prepared from A's closed state and
    delegated is refused by the probe (unknown schema) with nothing changed.
(2) derive and restore keep the chain: B (Y) prepared from A's state,
    delegated and activated; B's own compiled ledger open validates the chain,
    and B's closed binding carries the same constant-size ledger projection.
(3) erasure replay does not append: B's child runs erasure recovery twice and
    repeats the request; the ledger rows stay byte-identical.
(4) manifest: a child that restores an older file after its receipt is
    refused (UNKNOWN), and an older privacy.db restored into a closed state is
    caught offline by the receipt manifest and by forward evolution.
"""

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
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
SETTINGS = "schema: deletion-ledger-rehearsal\nmemory:\n  enabled: true\n"
CONVERSATION = "00000000-0000-4000-8000-00000000d0d3"

_OPEN = [
    "import importlib.util, json, os, sys, time, types",
    "sys.path.insert(0, os.getcwd())",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
    "from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemoryReviewer, MemorySource",
    "state = os.path.realpath('data')",
    "ledger = FactLedger(os.path.join(state, 'privacy.db'))",
    "ledger.initialize()",
    "reviewer = MemoryReviewer('rehearsal-reviewer')",
    "def rows():",
    "    return [list(row) for row in ledger._db.execute('SELECT * FROM deletion_ledger_v1 ORDER BY seq')]",
]
_RECEIPT = [
    "spec = importlib.util.spec_from_file_location('slot_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
]
# A: one explicit fact deletion and one conversation erasure (with a derived fact).
_DELETIONS = "\n".join(_OPEN + [
    "from src.memory.erasure_journal import ErasureParticipant",
    "now = time.time()",
    "import uuid",
    "def source(conversation):",
    "    return MemorySource(str(uuid.uuid4()), conversation, now - 1, now + 3600, MemoryOrigin.VOICE)",
    "fact = ledger.store('Rehearsal fact', 'fact', .9, (source('00000000-0000-4000-8000-00000000c0d3'),))",
    "assert ledger.delete_fact(fact.id, reviewer)",
    "ledger.store('Rehearsal erased fact', 'fact', .9, (source(%r),))" % CONVERSATION,
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': os.path.join(state, 'clips')})",
    "os.makedirs(os.path.join(state, 'clips'), exist_ok=True)",
    "ledger.erasures.request(%r, reviewer)" % CONVERSATION,
    "db = ledger._db",
    "db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "with open(sys.argv[1], 'x') as output:",
    "    json.dump(rows(), output)",
    "ledger.close()",
] + _RECEIPT)
# B (copied state): the slot's own ledger open validates the chain; then the
# erasure replay paths run and the rows must not change.
_REPLAY = "\n".join(_OPEN + [
    "from src.memory.deletion_ledger import verify_chain",
    "before = rows()",
    "chain = verify_chain(ledger._db).to_dict()",
    "ledger.erasures.recover()",
    "ledger.erasures.recover()",
    "ledger.erasures.request(%r, reviewer)" % CONVERSATION,
    "ledger.delete_conversation(%r, reviewer)" % CONVERSATION,
    "with open(sys.argv[1], 'x') as output:",
    "    json.dump({'before': before, 'after': rows(), 'chain': chain}, output)",
    "ledger.close()",
] + _RECEIPT)
# A child that closes with a new file version, then restores the older backup in place.
_RESTORE_AFTER_RECEIPT = "\n".join([
    "import importlib.util, os, shutil, sys",
    "with open(sys.argv[2], 'w') as handle:",
    "    handle.write('{\"v\": 2}')",
] + _RECEIPT + [
    "shutil.copy2(sys.argv[1], sys.argv[2])",
])


def _git(clone, *arguments, env=None):
    return subprocess.check_output(["git", *arguments], cwd=clone, text=True, env=env, timeout=120).strip()


def _pre_0d3_parent(clone, x):
    first = _git(clone, "log", "--reverse", "--format=%H", "-S", '"deletion_ledger_v1"', x,
                 "--", "scripts/stage_runtime.py").split()
    if not first:
        pytest.skip("no commit in X's history registers deletion_ledger_v1")
    parents = _git(clone, "show", "-s", "--format=%P", first[0]).split()
    if not parents:
        pytest.skip("the ledger registration has no parent commit in this clone")
    return _git(clone, "merge-base", x, parents[0])


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Build X (0d3 code), Y (descendant) and P (pre-0d3) from a temporary clone."""
    started = time.monotonic()
    if subprocess.run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=REPO,
                      capture_output=True).returncode != 0:
        pytest.skip("git history unavailable")
    base = tmp_path_factory.mktemp("deletion-ledger-artifacts")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    if _git(clone, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: the pre-0d3 parent cannot be located")
    head = _git(REPO, "rev-parse", "HEAD")
    env = dict(os.environ, GIT_INDEX_FILE=str(base / "index"), GIT_DIR=str(clone / ".git"),
               GIT_WORK_TREE=str(REPO), GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
               GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid")
    _git(clone, "read-tree", head, env=env)
    _git(clone, "add", "-A", "--", "src", "scripts", "tools", env=env)
    tree = _git(clone, "write-tree", env=env)
    x = _git(clone, "commit-tree", tree, "-p", head, "-m", "U3b-0d3 rehearsal X (working tree)", env=env)
    y = _git(clone, "commit-tree", tree, "-p", x, "-m", "U3b-0d3 rehearsal Y (trivial descendant)", env=env)
    pre = _pre_0d3_parent(clone, x)
    assert '"deletion_ledger_v1"' not in _git(clone, "show", pre + ":scripts/stage_runtime.py")
    assert is_managed_capable(clone, pre)
    archives = {}
    for name, revision in (("x", x), ("y", y), ("pre", pre)):
        archive = base / (name + ".tar.gz")
        assert build_runtime(clone, archive, revision)["git_commit"] == revision
        archives[name] = archive
    return {"archives": archives, "commits": {"x": x, "y": y, "pre": pre, "head": head},
            "build_seconds": time.monotonic() - started}


def _child(source, *arguments):
    return [sys.executable, "-I", "-B", "-c", source, *map(str, arguments)]


def _run(root, slot, argv):
    try:
        stage.run_slot(root, slot, argv, timeout=TIMEOUT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        pytest.fail("admitted child failed: " + (error.stderr or b"").decode(errors="replace")[-4000:])
    assert stage.read_admission_head(root)["status"] == "closed"


def _witness(root, *states):
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


def _ledger_rows(state):
    with closing(sqlite3.connect((state / PRIVACY).as_uri() + "?mode=ro", uri=True)) as db:
        return [list(row) for row in db.execute("SELECT * FROM deletion_ledger_v1 ORDER BY seq")]


def _closed_binding(root):
    return stage.read_binding(root, stage.read_admission_head(root)["binding_id"])


def _receipt(root):
    """The closed head's receipt-schema-2 record (beside its unchanged schema-1 receipt)."""
    head = stage.read_admission_head(root)
    run_id = stage._closure_proof(root, head["binding_id"])["running_head"]["run_id"]
    receipt = json.loads((root / "admission" / "receipts" / (run_id + ".json")).read_text())
    assert receipt["schema_version"] == 1
    return stage.read_receipt_state(root, receipt)


def _closed_a(artifacts, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text(SETTINGS)
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(artifacts["archives"]["x"], root, "a", source)
    root = root.resolve()
    state = root / "slots" / "a" / "state"
    output = tmp_path / "a-rows.json"
    _run(root, "a", _child(_DELETIONS, output))
    rows = _ledger_rows(state)
    assert json.loads(output.read_text()) == rows
    assert [row[1] for row in rows] == ["fact_deleted", "conversation_erased", "fact_deleted"]
    receipt = _receipt(root)
    assert receipt["schema_version"] == 2  # X's compiled child writes the closed-state record
    return root, state, rows


def test_pre_0d3_candidate_is_refused_after_the_new_schema(artifacts, tmp_path):
    """Rehearsal (1): the rollback rejection point, with nothing changed."""
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    stage.prepare_slot(artifacts["archives"]["pre"], root, "pre", state_a)
    state_pre = root / "slots" / "pre" / "state"
    assert stage.delegate_slot(root, "pre")["targets"]
    head = stage.read_admission_head(root)
    before = _witness(root, state_a, state_pre)
    with pytest.raises(ValueError, match="runtime_admission_candidate_rejected: runtime_admission_unknown_schema"):
        stage.activate_slot(root, "pre")
    assert _witness(root, state_a, state_pre) == before
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/a/runtime"
    assert _ledger_rows(state_a) == _ledger_rows(state_pre) == rows


def test_copy_keeps_the_chain_and_erasure_replay_never_appends(artifacts, tmp_path):
    """Rehearsals (2) and (3): A(X) -> B(Y); B's own ledger validates the chain and replays erasure."""
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    binding_a = _closed_binding(root)
    projection_a = binding_a["authority"]["projection"]["tables"][PRIVACY]["deletion_ledger_v1"]
    assert projection_a == {"count": 3, "max_seq": 3, "chain_head": rows[-1][5], "chain_valid": True}
    stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    state_b = root / "slots" / "b" / "state"
    assert stage.delegate_slot(root, "b")["targets"]
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    assert _ledger_rows(state_b) == rows
    assert stage.deletion_ledger_regressions(binding_a, state_b) == []
    output = tmp_path / "b-replay.json"
    _run(root, "b", _child(_REPLAY, output))
    report = json.loads(output.read_text())
    assert report["before"] == report["after"] == rows
    assert report["chain"] == projection_a
    binding_b = _closed_binding(root)
    assert binding_b["state_root"] == str(state_b)
    assert binding_b["authority"]["projection"]["tables"][PRIVACY]["deletion_ledger_v1"] == projection_a
    assert _ledger_rows(state_b) == rows


def test_a_file_restored_after_the_receipt_is_refused(artifacts, tmp_path):
    """Rehearsal (4a): X's compiled child restores an older file after its schema-2 receipt."""
    root, state_a, _rows = _closed_a(artifacts, tmp_path)
    target = state_a / "data" / "notes.json"
    target.write_text('{"v": 1}')
    older = tmp_path / "older-notes.json"
    shutil.copy2(target, older)
    with pytest.raises(ValueError, match="runtime_admission_state_manifest_mismatch"):
        stage.run_slot(root, "a", _child(_RESTORE_AFTER_RECEIPT, older, target), timeout=TIMEOUT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert stage.read_admission_head(root)["status"] == "unknown"


def test_an_older_privacy_db_restored_into_the_closed_state_is_detected(artifacts, tmp_path):
    """Rehearsal (4b): the manifest digest and forward evolution both catch an offline restore."""
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    receipt, binding = _receipt(root), _closed_binding(root)
    stage.verify_receipt_state(receipt, state_a)  # the closed state as sealed
    older = tmp_path / "older-privacy.db"
    with closing(sqlite3.connect(state_a / PRIVACY)) as source, closing(sqlite3.connect(older)) as copy:
        source.backup(copy)
    with closing(sqlite3.connect(older)) as db, db:  # the pre-erasure image: only the first deletion
        db.execute("DELETE FROM deletion_ledger_v1 WHERE seq>1")
    shutil.copy2(older, state_a / PRIVACY)
    with pytest.raises(ValueError, match="runtime_admission_state_manifest_mismatch"):
        stage.verify_receipt_state(receipt, state_a)
    assert stage.deletion_ledger_regressions(binding, state_a) == [PRIVACY]
    with stage.AdmissionController(root):  # the existing authority check refuses the restored state too
        with pytest.raises(ValueError, match="privacy_authority_stale"):
            stage.verify_closed_binding(root, stage.read_admission_head(root)["binding_id"])
    assert len(rows) == 3
