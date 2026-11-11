"""U3b-4a-0 rehearsals on compiled slots (design u3b-4 v4), plus the F-CFG probe P0.

Artifacts come from a temporary shared clone, never from the checkout's refs
(the test_fact_owner_rollback / test_compiled_multi_hop_delegation pattern):
X = HEAD plus this working tree, committed only in the clone (R1's code with the
voice stage off: the settings carry no memory.voice_proposals section), Y = an
empty descendant of X (stands in for a later release), P = the parent of the
first commit that registers voice_attribution_schema (the pre-4a controller
contract). 4a-0 has no writers, so admitted children write the rows with SQL
through the slot's own compiled ledger.

(1) code rollback: A (X) writes one row of every kind, aged past every
    window, plus the admin voice switch (G1) on; B (Y) is prepared from A's
    closed state, delegated and activated: the rows survive derivation and
    B's own ledger open byte-identical (R1: nothing time-based at open), and
    the settings bytes never change (G2 d).
(2) rejection point: P, prepared from A's closed state and delegated, is
    refused by the probe with nothing changed.
(3) second hop + G1 data path: B switches voice off in-run; C (X) prepared
    from B's state keeps the switch off.
P0: the voice flag lives in the state's settings.yaml, which the authority
    projection hashes. Editing it (a) in the prepared copy is refused at
    activation (delegation accepts), and (b) in the closed source state before
    prepare is refused at delegation; privacy_authority_stale both times, with
    the admission head and 'current' unchanged. Turning voice on therefore needs
    a controller procedure (F-CFG); turning it off uses the G1 switch.
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
SETTINGS = "schema: voice-attribution-rehearsal\nmemory:\n  enabled: true\n"
VOICE_SECTION = "  voice_proposals:\n    enabled: true\n"
TABLES = ("voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
          "voice_attribution_suspensions", "voice_attribution_control", "voice_attribution_schema")

_OPEN_DIRECTLY = [
    "import importlib.util, json, os, sys, time, types",
    "sys.path.insert(0, os.getcwd())",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
    "from src.memory.fact_ledger import FactLedger",
    "state = os.path.realpath('data')",
    "ledger = FactLedger(os.path.join(state, 'privacy.db'))",
    "ledger.initialize()",
]
_RECEIPT = [
    "spec = importlib.util.spec_from_file_location('slot_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
]
DAY = 86400
PROFILE = "00000000-0000-4000-8000-0000000000a1"
# Runs inside A's admitted child: one row of every kind, all aged past every
# window (attempts 24 h, events 30 d), and the admin switch on (G1).
_WRITER = "\n".join(_OPEN_DIRECTLY + [
    "from src.memory.erasure_journal import ErasureParticipant",
    "now = time.time()",
    "sources = json.dumps([{'capture_id': '00000000-0000-4000-8000-00000000c000', 'conversation_id':",
    "    '00000000-0000-4000-8000-00000000d000', 'captured_at': now - 60 * %d, 'expires_at': now - 59 * %d,"
    % (DAY, DAY),
    "    'origin': 'voice'}], sort_keys=True)",
    "db = ledger._db",
    "db.execute('BEGIN IMMEDIATE')",
    "db.execute(\"INSERT INTO voice_profiles VALUES (?,NULL,'enc_rehearsal',192,NULL,?,?,'rehearsal-reviewer',2,?)\",",
    "           ('%s', sources, now - 60 * %d, now - 50 * %d))" % (PROFILE, DAY, DAY),
    "db.execute(\"INSERT INTO voice_attribution_consents VALUES ('user_rehearsal_a','active',3,?,1,1,?,'user_rehearsal_a')\",",
    "           ('%s', now - 40 * %d))" % (PROFILE, DAY),
    "db.execute(\"INSERT INTO voice_verification_attempts VALUES ('user_rehearsal_a',?,'window_opened')\", (now - 40 * %d,))"
    % DAY,
    "db.execute(\"INSERT INTO voice_verification_attempts VALUES ('user_rehearsal_a',?,'failed')\", (now - 2 * %d,))"
    % DAY,
    "db.execute(\"INSERT INTO voice_attribution_events VALUES ('evt_rehearsal',?,'not_mine',15,?)\",",
    "           ('%s', now - 45 * %d))" % (PROFILE, DAY),
    "db.execute(\"INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')\",",
    "           ('%s', now - 44 * %d))" % (PROFILE, DAY),
    "db.execute(\"INSERT INTO voice_attribution_control VALUES (1,1,1,?,'rehearsal-operator')\", (now - 40 * %d,))"
    % DAY,
    "db.commit()",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': os.path.join(state, 'clips')})",
    "os.makedirs(os.path.join(state, 'clips'), exist_ok=True)",
    "# The voice ledger owns this marker row; a bare FactLedger leaves it empty.",
    "db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "ledger.close()",
] + _RECEIPT)
# Reads the copied state through the slot's own compiled ledger (initialize
# validates shape and values and must not prune) and reports every row.
_READER = "\n".join(_OPEN_DIRECTLY + [
    "rows = {table: sorted(map(list, ledger._db.execute('SELECT * FROM ' + table))) for table in %r}" % (TABLES,),
    "ledger.close()",
    "with open(sys.argv[1], 'x') as output:",
    "    json.dump(rows, output, sort_keys=True)",
] + _RECEIPT)
# G1 data path: an in-run admin write switches voice off (CAS on the revision).
_SWITCH_OFF = "\n".join(_OPEN_DIRECTLY + [
    "db = ledger._db",
    "db.execute('BEGIN IMMEDIATE')",
    "changed = db.execute(\"UPDATE voice_attribution_control SET enabled=0, revision=revision+1, changed_at=?, \"",
    "    \"changed_by='rehearsal-operator' WHERE id=1 AND revision=1\", (time.time(),)).rowcount",
    "assert changed == 1, changed",
    "db.commit()",
    "ledger.close()",
] + _RECEIPT)
_CLOSING = "\n".join(["import importlib.util, os"] + _RECEIPT)


def _git(clone, *arguments, env=None):
    return subprocess.check_output(["git", *arguments], cwd=clone, text=True, env=env, timeout=120).strip()


def _pre_4a_parent(clone, x):
    """Parent of the first commit (in X's history) that registers voice_attribution_schema."""
    first = _git(clone, "log", "--reverse", "--format=%H", "-S", '"voice_attribution_schema"', x,
                 "--", "scripts/stage_runtime.py").split()
    if not first:
        pytest.skip("no commit in X's history registers voice_attribution_schema")
    parents = _git(clone, "show", "-s", "--format=%P", first[0]).split()
    if not parents:
        pytest.skip("the voice-attribution registration has no parent commit in this clone")
    return _git(clone, "merge-base", x, parents[0])


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Build X (R1 code, voice off), Y (descendant) and P (pre-4a) from a temporary clone."""
    started = time.monotonic()
    if subprocess.run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=REPO,
                      capture_output=True).returncode != 0:
        pytest.skip("git history unavailable")
    base = tmp_path_factory.mktemp("voice-attribution-artifacts")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    if _git(clone, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: the pre-4a parent cannot be located")
    head = _git(REPO, "rev-parse", "HEAD")
    # Stage this checkout's working tree into the clone's object store through a
    # private index: nothing is written to the checkout's index, refs or objects.
    env = dict(os.environ, GIT_INDEX_FILE=str(base / "index"), GIT_DIR=str(clone / ".git"),
               GIT_WORK_TREE=str(REPO), GIT_AUTHOR_NAME="rehearsal", GIT_AUTHOR_EMAIL="rehearsal@invalid",
               GIT_COMMITTER_NAME="rehearsal", GIT_COMMITTER_EMAIL="rehearsal@invalid")
    _git(clone, "read-tree", head, env=env)
    _git(clone, "add", "-A", "--", "src", "scripts", "tools", env=env)
    tree = _git(clone, "write-tree", env=env)
    x = _git(clone, "commit-tree", tree, "-p", head, "-m", "U3b-4a-0 rehearsal X (working tree)", env=env)
    y = _git(clone, "commit-tree", tree, "-p", x, "-m", "U3b-4a-0 rehearsal Y (trivial descendant)", env=env)
    pre = _pre_4a_parent(clone, x)
    shown = _git(clone, "show", pre + ":scripts/stage_runtime.py")
    assert '"voice_attribution_schema"' not in shown and is_managed_capable(clone, pre)
    assert '"voice_attribution_schema"' in _git(clone, "show", x + ":scripts/stage_runtime.py")
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


def _rows(state):
    with closing(sqlite3.connect((state / PRIVACY).as_uri() + "?mode=ro", uri=True)) as db:
        return {table: sorted(map(list, db.execute("SELECT * FROM " + table))) for table in TABLES}


def _closed_a(artifacts, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text(SETTINGS)  # no memory.voice_proposals section (G2)
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(artifacts["archives"]["x"], root, "a", source)
    root = root.resolve()
    state = root / "slots" / "a" / "state"
    _run(root, "a", _child(_WRITER))
    rows = _rows(state)
    assert {table: len(value) for table, value in rows.items()} == {
        "voice_attribution_consents": 1, "voice_verification_attempts": 2, "voice_attribution_events": 1,
        "voice_attribution_suspensions": 1, "voice_attribution_control": 1, "voice_attribution_schema": 1}
    assert rows["voice_attribution_control"][0][:3] == [1, 1, 1]
    return root, state, rows


def test_code_rollback_keeps_aged_voice_rows_and_the_settings_bytes(artifacts, tmp_path):
    """Rehearsals (1) and (5): different code commit, one delegated hop, aged rows, no voice section."""
    commits = artifacts["commits"]
    assert commits["x"] != commits["y"]
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    manifest = stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    assert manifest["git_commit"] == commits["y"]
    state_b = root / "slots" / "b" / "state"
    with stage.AdmissionController(root) as controller:
        before = _witness(root, state_a, state_b)
        with pytest.raises(ValueError, match="runtime_admission_foreign_target"):
            controller.activate("b")  # Delegation is a required step.
        assert _witness(root, state_a, state_b) == before
    assert len(stage.delegate_slot(root, "b")["targets"]) == 1
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    assert _rows(state_b) == rows
    output = tmp_path / "b-view.json"
    _run(root, "b", _child(_READER, output))
    assert json.loads(output.read_text()) == rows  # B's own ledger open validated and pruned nothing
    assert _rows(state_b) == rows
    for state in (state_a, state_b):  # G2 (d): the config is never rewritten
        assert (state / "config" / "settings.yaml").read_text() == SETTINGS


def test_pre_4a_candidate_is_refused_at_activation_with_nothing_changed(artifacts, tmp_path):
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
    assert _rows(state_a) == _rows(state_pre) == rows


def test_second_hop_keeps_an_in_run_voice_switch_off(artifacts, tmp_path):
    """Rehearsal (3) + G1 data path: A(X) -> B(Y, switches voice off in-run) -> C(X)."""
    root, state_a, rows = _closed_a(artifacts, tmp_path)
    stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    stage.delegate_slot(root, "b")
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    state_b = root / "slots" / "b" / "state"
    _run(root, "b", _child(_SWITCH_OFF))
    switched = _rows(state_b)
    control = switched["voice_attribution_control"]
    assert len(control) == 1 and control[0][:3] == [1, 0, 2] and control[0][4] == "rehearsal-operator"
    assert {table: switched[table] for table in TABLES if table != "voice_attribution_control"} == {
        table: rows[table] for table in TABLES if table != "voice_attribution_control"}
    stage.prepare_slot(artifacts["archives"]["x"], root, "c", state_b)
    state_c = root / "slots" / "c" / "state"
    granted = stage.delegate_slot(root, "c")
    assert granted["targets"] and len(granted["capabilities"]) >= 1
    assert stage.activate_slot(root, "c") == "slots/b/runtime"
    output = tmp_path / "c-view.json"
    _run(root, "c", _child(_READER, output))
    assert json.loads(output.read_text()) == switched == _rows(state_c)
    assert os.readlink(root / "current") == "slots/c/runtime"


def _attempt(root, name, steps):
    """Run prepare/delegate/activate steps in order; report the first refusal (content-free)."""
    for step, action in steps:
        try:
            action()
        except (ValueError, RuntimeError, OSError) as error:
            return {"refused_at": step, "error": type(error).__name__, "code": str(error)[:200]}
    return {"refused_at": None, "current": os.readlink(root / "current")}


def _record_p0(variant, outcome):
    selected = os.environ.get("KZA_P0_EVIDENCE")
    if selected:
        path = Path(selected)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[variant] = outcome
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def test_p0_editing_the_prepared_copys_voice_flag_is_refused(artifacts, tmp_path):
    """P0 (a): settings.yaml edited in the prepared candidate after prepare.

    Delegation does not read the configuration and succeeds; the activation
    refuses the changed authority (the settings hash is in the projection),
    with the head, 'current' and every marker byte-identical.
    """
    root, state_a, _ = _closed_a(artifacts, tmp_path)
    stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)
    state_b = root / "slots" / "b" / "state"
    (state_b / "config" / "settings.yaml").write_text(SETTINGS + VOICE_SECTION)
    assert stage.delegate_slot(root, "b")["targets"]
    head = stage.read_admission_head(root)
    before = _witness(root, state_a, state_b)
    outcome = _attempt(root, "b", [("activate", lambda: stage.activate_slot(root, "b"))])
    outcome["delegate"] = "accepted"
    _record_p0("a_edit_copy_after_prepare", outcome)
    assert outcome == {"refused_at": "activate", "error": "ValueError", "code": "privacy_authority_stale",
                       "delegate": "accepted"}
    assert _witness(root, state_a, state_b) == before
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/a/runtime"
    assert (state_a / "config" / "settings.yaml").read_text() == SETTINGS


def test_p0_editing_the_source_states_voice_flag_before_prepare_is_refused(artifacts, tmp_path):
    """P0 (b): settings.yaml edited in the closed source state before prepare.

    prepare copies the edited state; delegation re-verifies the closed source
    binding and refuses, with the head and 'current' unchanged.
    """
    root, state_a, _ = _closed_a(artifacts, tmp_path)
    (state_a / "config" / "settings.yaml").write_text(SETTINGS + VOICE_SECTION)
    head = stage.read_admission_head(root)
    outcome = _attempt(root, "b", [
        ("prepare", lambda: stage.prepare_slot(artifacts["archives"]["y"], root, "b", state_a)),
        ("delegate", lambda: stage.delegate_slot(root, "b")),
        ("activate", lambda: stage.activate_slot(root, "b"))])
    _record_p0("b_edit_source_before_prepare", outcome)
    # prepare only copies; delegation re-verifies the closed source binding and refuses.
    assert outcome == {"refused_at": "delegate", "error": "ValueError", "code": "privacy_authority_stale"}
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/a/runtime"


# --- U3b-4a-1: G1 behavioural rehearsal with the real writers ------------------------------------

_OPEN_WITH_SETTINGS = _OPEN_DIRECTLY[:-2] + [
    "import yaml",
    "from src.memory.voice_attribution import (ProfileScore, ScoringSnapshot, VoiceAttributionCapability,",
    "    VoiceProposalSettings)",
    "from src.memory.fact_ledger import MemoryOrigin, MemorySource, MemoryVoiceConsentActor, VoiceSwitchActor",
    "with open(os.path.realpath(os.path.join('config', 'settings.yaml'))) as handle:",
    "    settings = VoiceProposalSettings.from_config(yaml.safe_load(handle))",
    "ledger = FactLedger(os.path.join(state, 'privacy.db'), voice_settings=settings)",
    "ledger.initialize()",
    "switch = VoiceSwitchActor('rehearsal-operator')",
    "allow = {'retired': lambda ids: False, 'active': lambda person: True}",
    "capability = VoiceAttributionCapability('rehearsal')",
    "def report(**values):",
    "    values['switch'] = vars(ledger.voice_switch()) | {'effective': ledger.voice_switch().effective}",
    "    values['owners'] = sorted(map(list, ledger._db.execute(",
    "        'SELECT o.person_id, o.method, r.last_change FROM fact_owners o JOIN fact_owner_revisions r USING(fact_id)')))",
    "    values['consents'] = sorted(map(list, ledger._db.execute(",
    "        'SELECT person_id, state, verified_profile_id FROM voice_attribution_consents')))",
    "    values['control'] = sorted(map(list, ledger._db.execute(",
    "        'SELECT id, enabled, revision, changed_by FROM voice_attribution_control')))",
    "    with open(sys.argv[1], 'x') as output:",
    "        json.dump(values, output, sort_keys=True)",
    "def proposal(text):",
    "    now = time.time()",
    "    source = MemorySource(str(__import__('uuid').uuid4()), str(__import__('uuid').uuid4()), now - 1,",
    "                          now + 47 * 3600, MemoryOrigin.VOICE)",
    "    return ledger.store(text, 'fact', 1.0, (source,), proposed=True)",
    "def scores(top):",
    "    return lambda db: ScoringSnapshot('enc_rehearsal', tuple(ProfileScore(p, 1, .95 if p == top else .3)",
    "        for p in sorted(r[0] for r in db.execute('SELECT id FROM voice_profiles WHERE deleted_at IS NULL'))),",
    "        True, 3.0)",
]
P1, P2 = "00000000-0000-4000-8000-0000000000b1", "00000000-0000-4000-8000-0000000000b2"
# The enabling release (voice on in settings): the operator switches voice on,
# a person opts in and verifies, and one spoken proposal becomes theirs.
_ENABLE = "\n".join(_OPEN_WITH_SETTINGS + [
    "from src.memory.erasure_journal import ErasureParticipant",
    "assert settings.enabled is True",
    "assert ledger.set_voice_switch(True, 0, switch).effective",
    "now = time.time()",
    "db = ledger._db",
    "db.execute('BEGIN IMMEDIATE')",
    "for index, (profile, person) in enumerate(((%r, 'user_rehearsal_a'), (%r, None))):" % (P1, P2),
    "    source = MemorySource('00000000-0000-4000-8000-00000000e00%d' % index,",
    "        '00000000-0000-4000-8000-00000000f00%d' % index, now - 10, now + 47 * 3600, MemoryOrigin.VOICE)",
    "    db.execute(\"INSERT INTO voice_profiles VALUES (?,?,'enc_rehearsal',2,?,?,?,'rehearsal-reviewer',1,NULL)\",",
    "        (profile, 'Rehearsal ' + str(index), bytes(8), json.dumps([source.to_dict()], sort_keys=True), now - 5))",
    "    db.execute('INSERT INTO voice_profile_sources VALUES (?,?)', (profile, source.conversation_id))",
    "    if person is not None:",
    "        db.execute(\"INSERT INTO voice_profile_people VALUES (?,?,1,?,'rehearsal-reviewer')\", (profile, person, now))",
    "db.commit()",
    "me = MemoryVoiceConsentActor('rehearsal-home', 'user_rehearsal_a')",
    "ledger.request_voice_consent(me, 0)",
    "binding = ledger.open_voice_verification(me, 1)",
    "assert ledger.verify_voice_consent(binding, scores(%r), capability, **allow).outcome.value == 'passed'" % P1,
    "assert ledger.attribute_voice_proposal(proposal('Rehearsal voice proposal').id, scores(%r), capability,"
    % P1,
    "    **allow).owner",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': os.path.join(state, 'clips')})",
    "os.makedirs(os.path.join(state, 'clips'), exist_ok=True)",
    "db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "report(stage='enabled')",
    "ledger.close()",
] + _RECEIPT)
# R1 after the rollback, in a state whose settings enable voice: voice is on
# until an operator turns the switch off in-run; then attribution stops.
_R1_SWITCH_OFF = "\n".join(_OPEN_WITH_SETTINGS + [
    "before = ledger.voice_switch()",
    "assert before.effective",
    "ledger.set_voice_switch(False, before.revision, switch)",
    "decision = ledger.attribute_voice_proposal(proposal('Rehearsal after the switch').id, scores(%r),"
    % P1,
    "    capability, **allow)",
    "report(stage='r1', before_effective=before.effective, decision=decision.reason.value)",
    "ledger.close()",
] + _RECEIPT)
_REPORT = "\n".join(_OPEN_WITH_SETTINGS + ["report(stage='later')", "ledger.close()"] + _RECEIPT)


def test_rollback_to_r1_with_voice_on_then_the_operator_switches_it_off(artifacts, tmp_path):
    """G1 (4a-1): A(Y, voice enabled) -> B(X = R1) is voice-on until the in-run switch; C(Y) keeps it off.

    The settings keep voice enabled in every state (F-CFG: only an admitted
    procedure could change them); the protected switch row is the off path.
    """
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    settings = SETTINGS + VOICE_SECTION
    (source / "config" / "settings.yaml").write_text(settings)
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(artifacts["archives"]["y"], root, "a", source)
    root = root.resolve()
    state_a = root / "slots" / "a" / "state"
    _run(root, "a", _child(_ENABLE, tmp_path / "a.json"))
    enabled = json.loads((tmp_path / "a.json").read_text())
    assert enabled["switch"] == {"configured": True, "enabled": True, "revision": 1, "effective": True}
    assert enabled["owners"] == [["user_rehearsal_a", "voice_proposal", "voice_proposed"]]
    assert enabled["consents"] == [["user_rehearsal_a", "active", P1]]

    stage.prepare_slot(artifacts["archives"]["x"], root, "b", state_a)  # rollback to R1 (user decision, Q7)
    state_b = root / "slots" / "b" / "state"
    assert stage.delegate_slot(root, "b")["targets"]
    assert stage.activate_slot(root, "b") == "slots/a/runtime"
    _run(root, "b", _child(_R1_SWITCH_OFF, tmp_path / "b.json"))
    r1 = json.loads((tmp_path / "b.json").read_text())
    assert r1["before_effective"] is True
    assert r1["switch"] == {"configured": True, "enabled": False, "revision": 2, "effective": False}
    assert r1["control"] == [[1, 0, 2, "rehearsal-operator"]]
    assert r1["decision"] == "disabled"
    assert r1["owners"] == enabled["owners"] and r1["consents"] == enabled["consents"]  # person data untouched

    stage.prepare_slot(artifacts["archives"]["y"], root, "c", state_b)
    state_c = root / "slots" / "c" / "state"
    assert stage.delegate_slot(root, "c")["targets"]
    assert stage.activate_slot(root, "c") == "slots/b/runtime"
    _run(root, "c", _child(_REPORT, tmp_path / "c.json"))
    later = json.loads((tmp_path / "c.json").read_text())
    assert later["switch"] == r1["switch"] and later["control"] == r1["control"]
    assert os.readlink(root / "current") == "slots/c/runtime"
    for state in (state_a, state_b, state_c):
        assert (state / "config" / "settings.yaml").read_text() == settings
