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

import atexit
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile

from scripts import adoption
from scripts import compaction
from scripts import stage_runtime as stage
from tests.mocks.runtime_artifact_cache import cached_build_runtime as build_runtime
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


_WORLD_TEMPLATE_CACHE: dict[tuple, dict] = {}
_WORLD_TEMPLATE_DIRS: list[Path] = []


def _cleanup_world_templates() -> None:
    for directory in _WORLD_TEMPLATE_DIRS:
        try:
            if directory.exists():
                stage.make_writable_then_remove(directory)
        except OSError:
            pass


atexit.register(_cleanup_world_templates)


def _copy_tree_preserving(source: Path, destination: Path) -> None:
    """Copy a directory tree preserving symlinks, 0555/0444 modes, mtimes, and SQLite bytes."""
    src_info = os.lstat(source)
    if not stat.S_ISDIR(src_info.st_mode):
        raise ValueError(f"Expected directory source: {source}")
    os.mkdir(destination, 0o700)
    dir_modes: list[tuple[Path, int, int, int]] = [(destination, stat.S_IMODE(src_info.st_mode),
                                                    src_info.st_atime_ns, src_info.st_mtime_ns)]
    stack: list[tuple[Path, Path]] = [(source, destination)]
    while stack:
        src_dir, dst_dir = stack.pop()
        with os.scandir(src_dir) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                src_item = Path(entry.path)
                dst_item = dst_dir / entry.name
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISLNK(info.st_mode):
                    os.symlink(os.readlink(src_item), dst_item)
                elif stat.S_ISDIR(info.st_mode):
                    os.mkdir(dst_item, 0o700)
                    dir_modes.append((dst_item, mode, info.st_atime_ns, info.st_mtime_ns))
                    stack.append((src_item, dst_item))
                elif stat.S_ISREG(info.st_mode):
                    shutil.copyfile(src_item, dst_item)
                    os.utime(dst_item, ns=(info.st_atime_ns, info.st_mtime_ns))
                    os.chmod(dst_item, mode)
                else:
                    raise ValueError(f"Unsupported special file in template: {src_item}")
    for dst_dir, mode, atime_ns, mtime_ns in reversed(dir_modes):
        os.utime(dst_dir, ns=(atime_ns, mtime_ns))
        os.chmod(dst_dir, mode)


def _replace_prefix(value, old_prefix: str, new_prefix: str):
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    if isinstance(value, list):
        return [_replace_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, dict):
        return {key: _replace_prefix(item, old_prefix, new_prefix) for key, item in value.items()}
    return value


def _relocate_privacy_db(db_path: Path, old_root: str, new_root: str) -> dict[str, str]:
    mode = stat.S_IMODE(os.lstat(db_path).st_mode)
    tid_map: dict[str, str] = {}
    with closing(sqlite3.connect(db_path, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT id, participant, version, descriptor FROM erasure_targets").fetchall()
        updated = []
        changed = False
        for old_id, participant, version, raw_desc in rows:
            if old_root in raw_desc:
                new_desc = _replace_prefix(json.loads(raw_desc), old_root, new_root)
                encoded = adoption._canonical(new_desc)
                new_id = adoption.target_id(participant, version, new_desc)
                changed = True
            else:
                encoded = raw_desc
                new_id = old_id
            tid_map[old_id] = new_id
            updated.append((new_id, participant, version, encoded))
        if changed:
            db.execute("DELETE FROM erasure_targets")
            db.executemany("INSERT INTO erasure_targets VALUES (?,?,?,?)", updated)
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ("erasure_steps", "policy_index_deletions", "wake_clips_v1"):
                if table not in tables:
                    continue
                for old_id, new_id in tid_map.items():
                    if old_id != new_id:
                        db.execute(f'UPDATE "{table}" SET target_id=? WHERE target_id=?', (new_id, old_id))
        db.execute("COMMIT")
    os.chmod(db_path, mode)
    return tid_map


def _build_world_uncached(
    archive: Path, tmp_path: Path, *, hops: tuple[str, ...], first: tuple[str, ...], env: str | None,
) -> dict:
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


def _clone_world_template(tpl: dict, tmp_path: Path, archive: Path, hops: tuple[str, ...]) -> dict:
    tpl_root: Path = tpl["root"]
    tmp_path.mkdir(parents=True, exist_ok=True)
    dst_root = (tmp_path / "root").resolve()
    _copy_tree_preserving(tpl_root, dst_root)
    old_root = os.fspath(tpl_root)
    new_root = os.fspath(dst_root)
    slots_order = ("a", *hops)

    # 1. Relocate privacy.db in every slot and collect global target_id mapping.
    tid_map: dict[str, str] = {}
    for slot_name in slots_order:
        db_path = dst_root / "slots" / slot_name / "state" / "data" / "privacy.db"
        if db_path.is_file():
            tid_map.update(_relocate_privacy_db(db_path, old_root, new_root))

    admission_dir = dst_root / "admission"

    # 2. Relocate launcher install record(s).
    launcher_dir = admission_dir / "launcher"
    if launcher_dir.is_dir():
        for path in sorted(launcher_dir.glob("install-*.json")):
            value = _replace_prefix(json.loads(path.read_text()), old_root, new_root)
            path.unlink()
            stage._atomic_record(path, value, immutable=True)

    # 3. Relocate admission/prepared records (slot & state inodes and paths).
    prepared_dir = admission_dir / "prepared"
    if prepared_dir.is_dir():
        staging_to_slot: dict[str, str] = {}
        prepared_entries = []
        for path in sorted(prepared_dir.glob("*.json")):
            val = json.loads(path.read_text())
            prepared_entries.append((path, val))
            if val.get("kind") == "prepared":
                staging_to_slot[val["staging_id"]] = val["slot"]
        for path, val in prepared_entries:
            val = _replace_prefix(val, old_root, new_root)
            if val.get("kind") == "prepared":
                slot_path = dst_root / "slots" / val["slot"]
                val["slot_identity"] = stage._identity(slot_path)
                val["state_identity"] = stage._identity(slot_path / "state")
            elif val.get("kind") == "staging" and val.get("staging_id") in staging_to_slot:
                slot_path = dst_root / "slots" / staging_to_slot[val["staging_id"]]
                val["identity"] = stage._identity(slot_path)
            path.unlink()
            stage._atomic_record(path, val, immutable=True)

    # 4. Load existing admission artifacts to re-chain in exact chronological order.
    bindings_dir = admission_dir / "bindings"
    old_bindings: dict[str, dict] = {}
    for path in sorted(bindings_dir.glob("*.json")):
        old_bindings[path.stem] = json.loads(path.read_text())
        path.unlink()

    closures_dir = admission_dir / "closures"
    old_closures: dict[str, dict] = {}
    if closures_dir.is_dir():
        for path in sorted(closures_dir.glob("*.json")):
            old_closures[path.stem] = json.loads(path.read_text())
            path.unlink()

    delegations_dir = admission_dir / "delegations"
    old_delegations: dict[str, dict] = {}
    if delegations_dir.is_dir():
        for path in sorted(delegations_dir.glob("*.json")):
            old_delegations[path.stem] = json.loads(path.read_text())
            path.unlink()

    chains_dir = admission_dir / "delegation-chains"
    old_chains: dict[str, dict] = {}
    if chains_dir.is_dir():
        for path in sorted(chains_dir.glob("*.json")):
            old_chains[path.stem] = json.loads(path.read_text())
            path.unlink()

    mc_dir = admission_dir / "manual-components"
    heads_dir = mc_dir / "heads"
    old_heads = sorted(
        (json.loads(p.read_text()) for p in heads_dir.glob("*.json")),
        key=lambda item: item["head"]["generation"],
    )
    for path in list(heads_dir.glob("*.json")):
        path.unlink()

    seals_dir = mc_dir / "seals"
    old_seals: dict[str, dict] = {}
    if seals_dir.is_dir():
        for path in sorted(seals_dir.glob("*.json")):
            old_seals[path.stem] = json.loads(path.read_text())
            path.unlink()

    # Map each slot to its (old_pre_binding_id, old_post_binding_id, old_closure).
    slot_runs: dict[str, tuple[str, str, dict]] = {}
    for old_post_bid, closure in old_closures.items():
        slot_name = old_bindings[old_post_bid]["slot"]
        old_pre_bid = closure["running_head"]["binding_id"]
        slot_runs[slot_name] = (old_pre_bid, old_post_bid, closure)

    binding_map: dict[str, str] = {}
    new_bindings: dict[str, dict] = {}
    new_closures_by_old_bid: dict[str, dict] = {}
    close_sha_by_old_bid: dict[str, str] = {}
    rec_map: dict[str, str] = {}
    slot_authority: dict[str, dict] = {}

    def _remap_binding(old_bid: str, *, ownership_sha256: str | None, authority: dict) -> tuple[str, dict]:
        old_b = old_bindings[old_bid]
        slot_name = old_b["slot"]
        slot_runtime = dst_root / "slots" / slot_name / "runtime"
        slot_state = dst_root / "slots" / slot_name / "state"
        new_b = dict(
            old_b,
            runtime_root=os.fspath(slot_runtime),
            state_root=os.fspath(slot_state),
            state_identity=stage._identity(slot_state),
            store_roots={rel: stage._identity(slot_state / rel) for rel in old_b["store_roots"]},
            authority=authority,
            ownership_sha256=ownership_sha256,
        )
        new_bid = stage._digest(new_b)
        stage._atomic_record(bindings_dir / (new_bid + ".json"), new_b, immutable=True)
        binding_map[old_bid] = new_bid
        new_bindings[new_bid] = new_b
        return new_bid, new_b

    for index, slot_name in enumerate(slots_order):
        slot_state = dst_root / "slots" / slot_name / "state"
        old_pre_bid, old_post_bid, old_closure = slot_runs[slot_name]
        run_id = old_closure["running_head"]["run_id"]

        pre_ownership_sha: str | None = None
        sealed_ownership_sha: str | None = None

        if index > 0:
            # Re-chain all delegations targeting this slot.
            old_dest = os.fspath(tpl_root / "slots" / slot_name / "state")
            slot_dels = [
                (old_rid, rec) for old_rid, rec in sorted(old_delegations.items())
                if rec["destination_state"] == old_dest
            ]
            for old_rid, rec in slot_dels:
                new_targets = {}
                for old_tid, target in rec["targets"].items():
                    new_tid = tid_map.get(old_tid, old_tid)
                    new_target = _replace_prefix(copy.deepcopy(target), old_root, new_root)
                    new_target["id"] = new_tid
                    new_targets[new_tid] = new_target
                new_rec = {
                    "schema_version": rec["schema_version"],
                    "destination_state": os.fspath(slot_state),
                    "destination_identity": stage._identity(slot_state),
                    "source_binding": binding_map[rec["source_binding"]],
                    "source_close_sha256": close_sha_by_old_bid[rec["source_binding"]],
                    "targets": new_targets,
                }
                new_rid = stage._digest(new_rec)
                rec_map[old_rid] = new_rid
                stage._atomic_record(delegations_dir / (new_rid + ".json"), new_rec, immutable=True)

                if old_rid in old_chains:
                    chain = old_chains[old_rid]
                    new_chains_map = {}
                    for old_tid, hops_list in chain["chains"].items():
                        new_tid = tid_map.get(old_tid, old_tid)
                        remapped_hops = []
                        for hop_entry in hops_list:
                            old_hbid = hop_entry["binding_id"]
                            hop_state = Path(hop_entry["state_root"].replace(old_root, new_root))
                            remapped_hops.append({
                                "binding_id": binding_map[old_hbid],
                                "close_sha256": close_sha_by_old_bid[old_hbid],
                                "closure": new_closures_by_old_bid[old_hbid],
                                "state_identity": stage._identity(hop_state),
                                "state_root": os.fspath(hop_state),
                                "via": rec_map[hop_entry["via"]] if hop_entry["via"] is not None else None,
                            })
                        new_chains_map[new_tid] = remapped_hops
                    grantor_old_bid = chain["grantor"]["binding_id"]
                    new_chain = {
                        "schema_version": chain["schema_version"],
                        "record_sha256": new_rid,
                        "destination_state": os.fspath(slot_state),
                        "destination_identity": stage._identity(slot_state),
                        "grantor": {
                            "binding_id": binding_map[grantor_old_bid],
                            "close_sha256": close_sha_by_old_bid[grantor_old_bid],
                        },
                        "chains": new_chains_map,
                    }
                    stage._atomic_record(chains_dir / (new_rid + ".json"), new_chain, immutable=True)

            # Re-write .kza-target-ownership.json and seal-intent.
            ownership_path = slot_state / ".kza-target-ownership.json"
            if ownership_path.is_file():
                old_policy = json.loads(ownership_path.read_text())
                new_policy = {
                    "schema_version": old_policy["schema_version"],
                    "delegated_targets": sorted(tid_map.get(t, t) for t in old_policy["delegated_targets"]),
                    "historical_targets": sorted(tid_map.get(t, t) for t in old_policy["historical_targets"]),
                    "owned_targets": sorted(tid_map.get(t, t) for t in old_policy["owned_targets"]),
                    "source_bindings": sorted(binding_map[b] for b in old_policy["source_bindings"]),
                }
                pre_policy = dict(new_policy, owned_targets=[])
                pre_ownership_sha = hashlib.sha256(stage._canonical_json(pre_policy) + b"\n").hexdigest()
                sealed_ownership_sha = hashlib.sha256(stage._canonical_json(new_policy) + b"\n").hexdigest()
                stage._atomic_record(ownership_path, new_policy)
                intent_path = admission_dir / "seal-intents" / (run_id + ".json")
                if intent_path.is_file():
                    intent_val = {
                        "schema_version": 1,
                        "run_id": run_id,
                        "pre_seal_ownership_sha256": pre_ownership_sha,
                        "sealed_ownership_sha256": sealed_ownership_sha,
                    }
                    intent_path.unlink()
                    stage._atomic_record(intent_path, intent_val, immutable=True)

        # Pre-run binding for this slot.
        pre_authority = (
            old_bindings[old_pre_bid]["authority"] if index == 0 else slot_authority[slots_order[index - 1]]
        )
        new_pre_bid, _new_pre_b = _remap_binding(
            old_pre_bid, ownership_sha256=pre_ownership_sha, authority=pre_authority,
        )

        # Post-run binding for this slot.
        post_authority = stage.capture_state_authority(slot_state)
        slot_authority[slot_name] = post_authority
        new_post_bid, new_post_b = _remap_binding(
            old_post_bid, ownership_sha256=sealed_ownership_sha, authority=post_authority,
        )

        # State binding marker.
        stage._atomic_record(
            slot_state / ".kza-state-binding.json",
            {"schema_version": 1, "admission_root": new_root,
             "binding_id": new_post_bid, "state_root": os.fspath(slot_state)},
        )

        # Receipt, receipt-state, and closure.
        receipt_path = admission_dir / "receipts" / (run_id + ".json")
        receipt = dict(json.loads(receipt_path.read_text()), binding_id=new_pre_bid)
        receipt_path.unlink()
        stage._atomic_record(receipt_path, receipt, immutable=True)
        receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

        rs_path = admission_dir / "receipt-states" / (run_id + ".json")
        if rs_path.is_file():
            pre_seal_binding = (
                dict(new_post_b, ownership_sha256=pre_ownership_sha)
                if pre_ownership_sha is not None else new_post_b
            )
            rs_val = dict(
                json.loads(rs_path.read_text()),
                binding_id=new_pre_bid,
                state_manifest_sha256=stage.state_manifest_sha256(slot_state),
                closed_state_sha256=stage.closed_state_sha256(pre_seal_binding),
            )
            rs_path.unlink()
            stage._atomic_record(rs_path, rs_val, immutable=True)

        new_closure = copy.deepcopy(old_closure)
        new_closure["binding_id"] = new_post_bid
        new_closure["receipt_sha256"] = receipt_sha
        new_closure["running_head"]["binding_id"] = new_pre_bid
        stage._atomic_record(closures_dir / (new_post_bid + ".json"), new_closure, immutable=True)
        new_closures_by_old_bid[old_post_bid] = new_closure
        close_sha_by_old_bid[old_post_bid] = stage._digest(new_closure)

    # 5. Re-chain manual-components heads & seals in ascending generation order.
    seal_map: dict[str, str] = {}
    prev_companion: dict | None = None
    final_head: dict | None = None
    for old_comp in old_heads:
        new_head = dict(old_comp["head"], binding_id=binding_map[old_comp["head"]["binding_id"]])
        cap_state = Path(old_comp["capture"]["state_root"].replace(old_root, new_root))
        new_capture = dict(
            old_comp["capture"],
            state_root=os.fspath(cap_state),
            state_identity=stage._identity(cap_state),
        )
        old_seal_id = old_comp["authority_seal"]
        new_seal_id = None
        if old_seal_id is not None:
            if old_seal_id not in seal_map:
                old_seal = old_seals[old_seal_id]
                seal_old_bid = old_seal["binding_id"]
                seal_cap_state = Path(old_seal["capture"]["state_root"].replace(old_root, new_root))
                new_seal = dict(
                    old_seal,
                    binding_id=binding_map[seal_old_bid],
                    capture=dict(
                        old_seal["capture"],
                        state_root=os.fspath(seal_cap_state),
                        state_identity=stage._identity(seal_cap_state),
                    ),
                    closed_head=dict(old_seal["closed_head"], binding_id=binding_map[seal_old_bid]),
                    closure_proof=new_closures_by_old_bid[seal_old_bid],
                )
                remapped_seal_id = stage._digest(new_seal)
                stage._atomic_record(seals_dir / (remapped_seal_id + ".json"), new_seal, immutable=True)
                seal_map[old_seal_id] = remapped_seal_id
            new_seal_id = seal_map[old_seal_id]
        new_comp = dict(
            old_comp,
            authority_seal=new_seal_id,
            capture=new_capture,
            head=new_head,
            previous=stage.manual_reference(prev_companion) if prev_companion is not None else None,
        )
        stage._atomic_record(heads_dir / (stage._digest(new_head) + ".json"), new_comp, immutable=True)
        prev_companion = new_comp
        final_head = new_head

    assert final_head is not None
    stage._atomic_record(admission_dir / "head.json", final_head)

    # 6. Build the return dict with updated state paths, binding ids, and delegate metadata.
    new_states: dict[str, dict] = {}
    for slot_name in slots_order:
        old_entry = tpl["states"][slot_name]
        entry = {
            "slot": slot_name,
            "state": dst_root / "slots" / slot_name / "state",
            "binding": binding_map[old_entry["binding"]],
        }
        if "delegate" in old_entry:
            remapped_delegate = _replace_prefix(copy.deepcopy(old_entry["delegate"]), old_root, new_root)
            if "records" in remapped_delegate and isinstance(remapped_delegate["records"], list):
                remapped_delegate["records"] = [rec_map.get(r, r) for r in remapped_delegate["records"]]
            if "record_sha256" in remapped_delegate and isinstance(remapped_delegate["record_sha256"], str):
                remapped_delegate["record_sha256"] = rec_map.get(
                    remapped_delegate["record_sha256"], remapped_delegate["record_sha256"],
                )
            entry["delegate"] = remapped_delegate
        new_states[slot_name] = entry
    return {"root": dst_root, "archive": archive, "states": new_states, "h": new_states[hops[-1] if hops else "a"]}


def _assert_no_template_paths(clone: dict, needle: str, *, full: bool) -> None:
    """Fail if a clone still names its template (file bytes, SQLite pages, link targets, the dict).

    ``needle`` is the template's random ``kza-cw-tpl-*`` basename, so both the ``/var`` and the
    ``/private/var`` spellings are caught. ``full`` also scans the verbatim slot runtimes; the
    first clone of each template does, later ones scan only ``admission/`` and the slot states.
    """
    marker = needle.encode()
    root: Path = clone["root"]
    assert marker not in repr(clone).encode(), f"template path leaked into the world dict: {needle}"
    if full:
        tops = [root]
    else:
        tops = [root / "admission", *sorted((root / "slots").glob("*/state"))]
    for top in tops:
        for directory, dirnames, filenames in os.walk(top):
            for name in (*dirnames, *filenames):
                path = os.path.join(directory, name)
                if os.path.islink(path):
                    assert marker not in os.fsencode(os.readlink(path)), f"template path in link {path}"
                elif name in filenames and stat.S_ISREG(os.lstat(path).st_mode):
                    with open(path, "rb") as handle:
                        assert marker not in handle.read(), f"template path in {path}"


def world(archive: Path, tmp_path: Path, *, hops: tuple[str, ...] = ("b", "c"), first: tuple[str, ...] = (),
          env: str | None = None) -> dict:
    """A -> hops..., each a real closed head; the last hop is H. Controller and launcher installed.

    SC-1a: a world with hops has owners, so it needs the epoch-4 build (the x4_archive fixture).
    """
    assert not hops or stage.CONTROLLER_EPOCH == 4, "owner worlds need the epoch-4 build (x4_archive)"
    first_tuple = tuple(first)
    hops_tuple = tuple(hops)
    if first_tuple:
        # Fail closed: the codes run against A may write paths into any store (ambient.db's
        # audio_path, for one), which the clone below does not relocate. Never cache them.
        return _build_world_uncached(archive, tmp_path, hops=hops_tuple, first=first_tuple, env=env)
    archive_real = os.path.realpath(archive)
    st = os.lstat(archive_real)
    key = (archive_real, st.st_size, st.st_mtime_ns, stage.CONTROLLER_EPOCH, hops_tuple, env)
    tpl = _WORLD_TEMPLATE_CACHE.get(key)
    if tpl is None or not tpl["root"].is_dir():
        tpl_dir = Path(tempfile.mkdtemp(prefix="kza-cw-tpl-")).resolve()
        _WORLD_TEMPLATE_DIRS.append(tpl_dir)
        tpl = _build_world_uncached(archive, tpl_dir, hops=hops_tuple, first=(), env=env)
        _WORLD_TEMPLATE_CACHE[key] = tpl
        clone = _clone_world_template(tpl, tmp_path, archive, hops_tuple)
        _assert_no_template_paths(clone, tpl_dir.name, full=True)  # The runtimes are copied verbatim.
    else:
        clone = _clone_world_template(tpl, tmp_path, archive, hops_tuple)
        _assert_no_template_paths(clone, tpl["root"].parent.name, full=False)
    return clone


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
