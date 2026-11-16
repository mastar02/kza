"""Helpers for the U3b-0c owner-compaction tests and rehearsals.

A world is a real managed root built from X (HEAD plus this working tree,
compiled in a temporary shared clone): a managed bootstrap slot, real closed
runs whose child initializes the production FactLedger (erasure journal,
owner schema, deletion ledger) and registers one collector target per slot,
then delegate hops A -> B -> C in process. Since SC-2 the delegate CLI and
every grant refuse (runtime_admission_delegation_retired), so a world with
hops builds them inside tests.mocks.legacy_transitions (the grant and each
hop's first run, a cross-slot activation); everything after world() runs
with the production flag, so compaction of existing owners is exercised as
an epoch-4 controller runs it (since SC-1a, an epoch-4 build of this tree:
tests/mocks/epoch4.py). The launcher
install record is written as install-launcher would (systemd is not touched
here; the P14 scan is injected and tested in its own modules).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import compaction
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks.epoch4 import epoch4_commit
from tests.mocks.legacy_transitions import legacy_managed_transitions

REPO = rehearsal.REPO
NOW = 1_800_000_000.0

# The closing child: run each code string with ``state`` bound, then write the slot's own receipt.
CHILD = "\n".join([
    "import importlib.util, json, os, sys",
    "sys.path.insert(0, os.getcwd())",
    "from pathlib import Path",
    "state = Path(os.path.realpath('data')).parent",
    "for code in json.loads(sys.argv[1]):",
    "    exec(code, {'state': state})",
    "spec = importlib.util.spec_from_file_location('candidate_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
])
_LEDGER_OPEN = "\n".join([
    "import os, sqlite3, types",
    "from contextlib import closing",
    "from src.memory import fact_ledger",
    "_noop = lambda *args, **kwargs: None",
    "fact_ledger.output_registration = types.SimpleNamespace(before_ledger_open=_noop, ledger_connected=_noop,",
    "    borrowed_ledger=_noop, ledger_closed=_noop, ledger_close_failed=_noop)",
    "ledger = fact_ledger.FactLedger(str(state / 'data' / 'privacy.db'))",
    "ledger.initialize()",
])
# Initialize the production ledger and register this slot's own collector root.
LEDGER = "\n".join([
    _LEDGER_OPEN,
    "from src.memory.erasure_journal import ErasureParticipant",
    "name = 'clips-' + state.parent.name",
    "os.makedirs(state / 'data' / name, exist_ok=True)",
    "ledger.erasures.register_target(ErasureParticipant.COLLECTORS, 1,",
    "    {'kind': 'wake_clips', 'root': str(state / 'data' / name)})",
    "if not ledger._db.execute('SELECT 1 FROM voice_group_schema').fetchone():",
    "    ledger._db.execute('INSERT INTO voice_group_schema VALUES (1)')",
    "ledger.close()",
])


def sql(*statements) -> str:
    """Child code: run (sql, params) pairs on the slot's privacy.db in one transaction."""
    return "\n".join([
        "import sqlite3",
        "from contextlib import closing",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db')) as db, db:",
        *["    db.execute(" + repr(text) + ", " + repr(tuple(parameters)) + ")" for text, parameters in statements],
    ])


def ledger_append(kind: str, subject: str, at: float = NOW) -> str:
    """Child code: one deletion-ledger entry through the production API."""
    return "\n".join([
        "import sqlite3",
        "from contextlib import closing",
        "from src.memory.deletion_ledger import DeletionKind, append_deletion",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db', isolation_level=None)) as db:",
        "    db.execute('BEGIN IMMEDIATE')",
        f"    append_deletion(db, DeletionKind({kind!r}), {subject!r}, {at!r})",
        "    db.execute('COMMIT')",
    ])


def write_file(relative: str, text: str) -> str:
    """Child code: write one file under the state."""
    return f"(state / {relative!r}).parent.mkdir(parents=True, exist_ok=True)\n" \
           f"(state / {relative!r}).write_text({text!r})"


def no_units(root, members):
    """The P14 scan stub of the in-process tests (the scanner has its own unit and real-systemd tests)."""
    return {"units_scanned": 0, "unit_names": [], "logrotate_files": []}


def build_x(base: Path, *, epoch4: bool = False) -> Path:
    """X = HEAD + the working tree's scripts/ and src/, compiled (cached by KZA_COMPACTION_ARTIFACT).

    ``epoch4`` builds the same tree with CONTROLLER_EPOCH = 4 (tests/mocks/epoch4.py), for worlds
    with owners (cached by KZA_COMPACTION_ARTIFACT_EPOCH4).
    """
    cached = os.environ.get("KZA_COMPACTION_ARTIFACT_EPOCH4" if epoch4 else "KZA_COMPACTION_ARTIFACT")
    if cached:
        return Path(cached)
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    revision = rehearsal.overlay_commit(REPO, clone, message="u3b-0c rehearsal")
    if epoch4:
        revision = epoch4_commit(clone, revision)
    archive = base / ("x4.tar.gz" if epoch4 else "x.tar.gz")
    build_runtime(clone, archive, revision)
    return archive


def run(root: Path, slot: str, *codes: str, timeout: float = 60) -> str:
    """One operator run of ``slot`` closing cleanly; returns the closed head's binding id."""
    stage.run_slot(root, slot, [sys.executable, "-I", "-B", "-c", CHILD, json.dumps(list(codes))],
                   timeout=timeout)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    return head["binding_id"]


def delegate(root: Path, slot: str) -> dict:
    """The working tree's delegation in process, with the advisory the pre-SC-2 CLI printed.

    SC-2 retired the grant and the CLI (exit 78); a caller builds legacy
    owners inside tests.mocks.legacy_transitions or under the fixture.
    """
    assert stage.LEGACY_MANAGED_TRANSITIONS, "owner hops need legacy managed transitions (SC-2)"
    result = stage.delegate_slot(root, slot)
    result["compaction_advisory"] = compaction.advisory(Path(root).resolve(), owners=result["owners_after_grant"])
    return {"slot": slot, **result}


def delegate_cli(root: Path, slot: str) -> subprocess.CompletedProcess:
    """The working tree's delegate CLI: refused since SC-2 (exit 78, {"refused": code})."""
    return subprocess.run([sys.executable, "-B", str(REPO / "scripts" / "stage_runtime.py"), "--root", str(root),
                           "--offline", "delegate", slot], cwd=REPO, capture_output=True, text=True, timeout=60)


def install_record(root: Path, contract: int = 3, stamp: str = "20261114T000000Z000000") -> Path:
    """admission/launcher/install-<stamp>.json as install-launcher writes it (content-free subset)."""
    path = root / "admission" / "launcher" / f"install-{stamp}.json"
    stage._atomic_record(path, {"schema_version": 1, "launcher_contract": contract,
                                "logrotate_conf": str(root / "console-logrotate.conf")}, immutable=True)
    return path


def hop(archive: Path, root: Path, source_state: Path, slot: str, *codes: str) -> dict:
    """Copy the closed head's state into ``slot``, delegate, run it closed with LEDGER plus ``codes``.

    The grant and the first run (a cross-slot activation) are legacy
    transitions since SC-2; they run inside legacy_managed_transitions().
    """
    stage.prepare_slot(archive, root, slot, source_state)
    with legacy_managed_transitions():
        output = delegate(root, slot)
        binding = run(root, slot, LEDGER, *codes)
    return {"slot": slot, "state": root / "slots" / slot / "state", "binding": binding, "delegate": output}


def world(archive: Path, tmp_path: Path, *, hops: tuple[str, ...] = ("b", "c"), first: tuple[str, ...] = (),
          env: str | None = None) -> dict:
    """A -> hops..., each a real closed head; the last hop is H. Controller and launcher installed.

    SC-1a: a world with hops has owners, so it needs the epoch-4 build (the x4_archive fixture).
    """
    assert not hops or stage.CONTROLLER_EPOCH == 4, "owner worlds need the epoch-4 build (x4_archive)"
    source = rehearsal.offline_source(tmp_path)
    if env is not None:
        (source / ".env").write_text(env)
        (source / ".env").chmod(0o600)
    staging = tmp_path / "root"
    stage.prepare_slot(archive, staging, "a", source)
    root = staging.resolve()
    with stage.AdmissionController(root) as admission:
        assert admission.activate("a", managed=True) is None
    from scripts import managed_launcher as launcher
    launcher.install_controller(root, archive)
    install_record(root)
    states = {"a": {"slot": "a", "state": root / "slots" / "a" / "state",
                    "binding": run(root, "a", LEDGER, *first)}}
    previous = states["a"]["state"]
    for name in hops:
        states[name] = hop(archive, root, previous, name)
        previous = states[name]["state"]
    return {"root": root, "archive": archive, "states": states, "h": states[hops[-1] if hops else "a"]}


def plan(root: Path, archive: Path, k: str = "k", **options) -> dict:
    return compaction.compact_plan(root, k, archive, scan=options.pop("scan", no_units), **options)


def start(root: Path, archive: Path, k: str = "k", *, planned: dict | None = None, **options) -> dict:
    planned = planned or plan(root, archive, k, retire_prepare_leftovers=options.get("retire_prepare_leftovers",
                                                                                      False))
    return compaction.compact_start(root, k, archive, plan_sha256=planned["plan_sha256"],
                                    acknowledge_untouched=planned["untouched_sha256"],
                                    destroy_rollback_states=True, scan=options.pop("scan", no_units), **options)


def serve(root: Path, tmp_path: Path, name: str, *, scenario: str = "clean", execute: str | None = None,
          timeout: float = 60, wall=None, seconds: float = 0.0):
    """One in-process serve under the fake manager; stopped once the child is READY."""
    phase_file = tmp_path / ("phases-" + name)
    script = None
    if execute is not None:
        script = tmp_path / ("exec-" + name + ".py")
        script.write_text(execute)
    options = {} if wall is None else {"wall": wall}
    return rehearsal.run_serve(root, rehearsal.child_command(phase_file, scenario, execute=script, seconds=seconds),
                               phase_file=phase_file,
                               stop_when=lambda phases: any(key in {"ready", "ready_skipped"} for key, _ in phases),
                               timeout=timeout, **options)


def tree(path: Path) -> dict:
    """Byte-level witness of a tree: files, link texts, directories (and nothing followed)."""
    found = {}
    for directory, names, filenames in os.walk(path, followlinks=False):
        for name in sorted(names + filenames):
            item = Path(directory) / name
            relative = item.relative_to(path).as_posix()
            if item.is_symlink():
                found[relative] = ("link", os.readlink(item))
            elif item.is_dir():
                found[relative] = ("dir",)
            else:
                found[relative] = ("file", item.read_bytes())
    return found


def witness(root: Path) -> dict:
    """Everything under the root except the two lock files every controller entry rewrites."""
    return {key: value for key, value in tree(root).items() if key not in {".admission.lock", ".stage.lock"}}
