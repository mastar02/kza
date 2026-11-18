"""Owner-state compaction, controller side (U3b-0c).

Owner-compaction design v2 to v5 (data/autonomous-roadmap-20261102/t040-t041/
owner-compaction-design*.json, with every review fix through v5's E1-E3) and the
user's decisions u3b_0c_compaction (P1-P7) and u3b_0c_compaction_v2 (Q1-Q7,
P5_vacuum_new_state). An explicit operator command compacts the managed root
down to one new state K (an identity clone of the closed head H with every
locator moved into K) and, after K has completed one clean managed run,
retires every other proven copy under the root:

* ``compact plan <K> --archive <H archive>``: read-only. Preconditions
  P1-P14, the positive-proof retirement set (v3), the untouched list and its
  digest, content-free coverage counts (members against H), capacity, the
  advisory, H's Chroma collections against the closed list (SC-4c), and the
  plan digest.
* ``compact start <K> ...``: the intent, then C2a (K's runtime from H's
  archive), C2b/C3 in the compaction worker (snapshot, identity clone, D4
  VACUUM on K only, D5 exact check), A1 authorize and A2 activate.
* ``compact verify <id>``: arms the one verification run; the operator starts
  kza-voice and stops it (serve's gate lets exactly that run through).
* ``compact retire <id>``: after the armed run closed cleanly (K'), unlinks
  every member (delete only, P5 = C), writes tombstones and copies-retired
  records, then 'complete'.
* ``compact resume|abort|allow-run|status <id>``.

Everything this module writes under admission/ is content-free: ids,
digests, slot names, absolute slot paths, {device, inode}, sizes and counts.
It never imports ``src`` (G1); the worker phases run as a separate process
with their own capability (src.core.compaction_worker).
"""
from __future__ import annotations

import calendar
from contextlib import closing
import errno
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from uuid import uuid4

logger = logging.getLogger(__name__)

stage = None  # scripts/stage_runtime
launcher = None  # scripts/managed_launcher
units = None  # scripts/compaction_units (P14)


def bind(stage_module, launcher_module, units_module) -> None:
    """Use these implementations (stage_runtime binds by path when compiled)."""
    global stage, launcher, units
    stage, launcher, units = stage_module, launcher_module, units_module


if __package__ == "scripts":  # in-process users (tests)
    from scripts import compaction_units as _units_module  # noqa: E402
    from scripts import managed_launcher as _launcher_module  # noqa: E402
    from scripts import stage_runtime as _stage_module  # noqa: E402

    bind(_stage_module, _launcher_module, _units_module)


class CompactionRefusal(ValueError):
    """A content-free refusal; ``str(error)`` is the code, ``report`` what was found so far."""

    def __init__(self, code: str, report: dict | None = None) -> None:
        super().__init__(code)
        self.report = report


def _refuse(code: str, report: dict | None = None):
    raise CompactionRefusal(code, report)


COMMANDS = ("compact",)
SUBCOMMANDS = ("plan", "start", "verify", "retire", "resume", "abort", "allow-run", "status")
SCHEMA_VERSION = 1
# The user's answers this implementation binds into every intent.
DECISIONS = ("P1=A", "P2=A", "P2b=A", "P3=A", "P4=A", "P5=C", "P5b=A", "P6=A", "P7=A", "Q1=A", "Q2=No",
             "Q3=A", "Q4=inside-/home/kza", "Q5=A", "Q6=A", "Q7=A", "P5_vacuum_new_state=Yes")
OWNER_WARNING = 6  # P1: the advisory recommends compaction at 6 owners
BINDING_MARGIN = 12 * 1024 * 1024  # P9 / M6: K's binding and each projected history archive
FREE_MARGIN = 256 << 20  # P9: free space left over at the measured peak
# P9 (SC-0b): the snapshot manifest holds every row as JSON; this estimate of its size per state byte
# is calibrated, not a bound (measured 2.45 on the synthetic production-size state). The worker
# checks again with the real manifest on disk before it writes K (owner_compaction.free_space_need).
MANIFEST_ESTIMATE_FACTOR = 3
SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
ALLOW_RUN_HOURS = (1, 72)  # Q7
ADVISORY_KINDS = ("conversation_erased", "fact_deleted", "fact_pruned_retired_owner", "voice_profile_deleted",
                  "user_retired")
# Root entries that are never members and never listed as untouched: the managed root itself.
FIXED_ENTRIES = frozenset({"admission", "slots", "current", ".stage.lock", ".admission.lock", "compaction"})
MEMBER_CLASSES = ("own_work", "earlier_compaction", "prepare_leftover", "selection_leftover", "catalog_copy",
                  "managed_slot")
RUNTIME_LINKS = frozenset({"config", "data", ".env"})
SLOT_ENTRIES = frozenset({"runtime", "state"})  # all that prepare_slot makes in a slot (review F6)
_SLOT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_HEX64 = re.compile("[0-9a-f]{64}")
_DEADLINES = ("expires_at", "expires")
_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")
_INTERNAL_TABLES = frozenset({"sqlite_sequence", "sqlite_stat1", "sqlite_stat4"})
_SCHEDULING = frozenset({"erasure_targets", "erasure_steps", "policy_index_deletions", "index_jobs"})
# F7: a dashboard login during the smoke window does not block abort-to-H.
ABORT_EXCLUDED_PROTECTED = frozenset({"sessions", "credential_attempts"})
# SC-1b (sc-release-design v4 mechanism.mode): at controller epoch >= 5 every new log is a
# release-mode log, intent.mode = "release"; a log without 'mode' is a plain-mode log.
RELEASE_MODE = "release"
T0_REQUIRED = "runtime_admission_compaction_t0_required"
CONTROLLER_BEHIND = "runtime_admission_compaction_controller_behind"
# The plain-mode paths (Q6, allow-run, --discard-k-writes) finish only a log without 'mode'.
PLAIN_MODE_ONLY = "runtime_admission_compaction_plain_mode_only"
# SC-4a review RA-1/RB-1 (design note SC4A-N13): SC-4a ships the isolated verification run V, but
# every exit after V (W, abort with W, retire with strict W equality) is SC-4b's. SC-4b ships them
# and sets this to True; with False a release-mode log is never armed (release gate) and its only
# way forward is the lossless plain abort of row A0, which start/resume then name.
RELEASE_EXITS_AVAILABLE = True
RELEASE_EXIT_UNAVAILABLE = "runtime_admission_compaction_release_exit_unavailable"
# SC-4a review RB-2 (SC4A-N14): a release-mode V needs an explicit N within the design's cap
# (downtime_and_disk.formula_check: N = min(60, 600 - T_copy - boots x B - stops), never below 10).
RELEASE_SMOKE_SECONDS_MAX = 60
RELEASE_SMOKE_SECONDS = "runtime_admission_compaction_release_smoke_seconds"
# SC-4b (sc-release-design v4 mechanism.w_abort_predicate, retire): W replaces Q6 in release mode.
# W1's two closed lists (SC-R reviewed): a difference outside them is always restrictive_or_unclear.
W_LOSS = frozenset({"conversation_tombstones", "conversation_erasures", "retired_users", "fact_owner_revisions",
                    "voice_attribution_suspensions", "pending_audio_deletions"})
W_GRANT = frozenset({"credentials", "sessions"})
W_EQUAL, W_LOSS_ONLY, W_RESTRICTIVE = "equal", "loss_only", "restrictive_or_unclear"
W_REPORT_SCHEMA = 1
# W0: reading H failed for a reason that clears (row AT); every other failure is the drift HD.
W0_TRANSIENT_CODES = frozenset({"runtime_admission_writer_active", "runtime_admission_busy",
                                "runtime_admission_sdk_version_unavailable"})
W0_TRANSIENT_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE, errno.EIO, errno.EAGAIN, errno.EINTR, errno.ENOMEM,
                                 errno.EBUSY})
W0_TRANSIENT_SQLITE = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
                                 sqlite3.SQLITE_NOMEM, sqlite3.SQLITE_INTERRUPT})
SDK_CHANGED = "runtime_admission_compaction_sdk_changed"
H_CHANGED = "runtime_admission_compaction_h_changed"
H_TRANSIENT = "runtime_admission_compaction_h_transient"
ABORT_PROTECTED_CHANGED = "runtime_admission_compaction_abort_protected_changed"
DISCARD_MISMATCH = "runtime_admission_compaction_discard_k_changes_mismatch"
RELEASE_MODE_ONLY = "runtime_admission_compaction_release_mode_only"
RETIRE_PROTECTED_CHANGED = "runtime_admission_compaction_retire_protected_changed"
RETIRE_RESURRECTED = "runtime_admission_compaction_retire_resurrected"
K_BINDING_CHANGED = "runtime_admission_compaction_k_binding_changed"
# SC-4b review RA4B-5/RB-7: K' read while a live holder keeps one of its databases locked (or a resource
# errno) is a transient refusal that writes nothing; a fault that persists (EIO, SQLITE_IOERR, damage) stays
# 'unreadable', so SS keeps its exit (SC4B-N19).
K_TRANSIENT = "runtime_admission_compaction_k_transient"
K_TRANSIENT_ERRNOS = W0_TRANSIENT_ERRNOS - {errno.EIO}
K_TRANSIENT_SQLITE = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_NOMEM,
                                sqlite3.SQLITE_INTERRUPT})
# Review RB-3: K' head's closure proof unreadable after V (storage damage, a manual edit): an HD-like stop.
K_CLOSURE_UNREADABLE = "runtime_admission_compaction_k_closure_unreadable"
# Row PA fails closed until its witness exists (SC5B-Q1; runbook «Después de la aceptación»): a head closed
# by a recovery record (R-b or adopt) is not the source of a release-mode transition, even when its binding
# equals one an earlier closure proved (SC-6b review F1). Row PA's closed_witness replaces this refusal.
RECOVERY_HEAD = "runtime_admission_compaction_recovery_head"
# SC-4c closed collection list (sc-release-design v4 start.chroma_allowlist, PA-8, V4R2-C1). The worker
# (src/memory/owner_compaction.py) holds the same list; tests/unit/tools/test_sc4c_chroma_closed_list.py pins both.
CHROMA_STORE = "chroma.sqlite3"
CHROMA_COMMAND_INDEX = "home_assistant_commands"  # with "__pointer" and the "__<version>" staging collections
CHROMA_POLICY_SUFFIX = "_policy_v1"  # SC4C-RV-5 (1): the exact names the runtime makes
CHROMA_COMMAND_NAMES = re.compile(re.escape(CHROMA_COMMAND_INDEX) + r"(__pointer|__[0-9]{14}[0-9a-f]{6})?")
# The named drops, by exact name in their own store only (not copied into K): the command store's routines
# (K's boot recreates them; SC4C-Q6) and the quarantined pre-policy memory collection (SC4C-Q2b).
CHROMA_DROPPED_IN_STORE = frozenset({("data/chroma_db", "home_assistant_routines"),
                                     ("data/memory_db", "user_memories")})
UNLISTED_COLLECTION = "runtime_admission_compaction_unlisted_collection"
# SC4C-RV-5 (4): a store's own files (the pinned SDK's layout); any other file inside a store directory is
# not carried into K (the store is rebuilt), so the plan counts it per store ('other_files').
CHROMA_STORE_FILES = re.compile(r"chroma\.sqlite3(-wal|-shm|-journal)?|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                                r"[0-9a-f]{4}-[0-9a-f]{12}/(header|data_level0|length|link_lists)\.bin|"
                                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/index_metadata\.pickle")
# What the worker inherits; its Chroma rebuild child gets the same (owner_compaction._CHILD_ENVIRONMENT).
WORKER_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SQLITE_TMPDIR", "HOME")
# Review RA4B-1: what a value the selection cannot classify raises (W1 never raises an uncoded error).
_W1_VALUE_ERRORS = (ValueError, TypeError, AttributeError, KeyError, IndexError)


# --- small helpers -----------------------------------------------------------------------------------

def _euid() -> int:
    return os.geteuid()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now(wall) -> float:
    value = float(wall())
    if not math.isfinite(value):
        _refuse("runtime_admission_compaction_clock")
    return value


def checked_root(root) -> Path:
    """The managed root as the operator named it; refuses root, a foreign owner and an unmanaged tree."""
    if _euid() == 0:
        _refuse("runtime_admission_compaction_as_root")
    path = Path(os.path.abspath(os.fspath(root)))
    try:
        info = os.lstat(path)
    except OSError:
        _refuse("runtime_admission_compaction_root")
    if not stat.S_ISDIR(info.st_mode) or path.resolve() != path:
        _refuse("runtime_admission_compaction_root")
    if info.st_uid != _euid():
        _refuse("runtime_admission_compaction_owner")
    if not os.path.lexists(path / "admission" / "head.json"):
        _refuse("runtime_admission_compaction_unmanaged")
    return path


def _identity(path: Path) -> dict:
    info = os.lstat(path)
    return {"device": info.st_dev, "inode": info.st_ino}


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _kind(path: Path) -> str:
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def _walk_census(path: Path) -> dict:
    """Content-free census of one member: files, bytes, links and a digest of (relative path, size)."""
    files = size = links = specials = 0
    listing = []
    device = os.lstat(path).st_dev
    for directory, names, filenames, descriptor in os.fwalk(path, follow_symlinks=False):
        names.sort()
        if os.fstat(descriptor).st_dev != device:
            _refuse("runtime_admission_compaction_mount")
        for name in sorted(filenames):
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = os.path.relpath(os.path.join(directory, name), path)
            if stat.S_ISLNK(info.st_mode):
                links += 1
            elif stat.S_ISREG(info.st_mode):
                files += 1
                size += info.st_size
                listing.append([relative, info.st_size])
            else:
                specials += 1
        for name in names:  # a symlink to a directory is reported by fwalk among names
            if stat.S_ISLNK(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                links += 1
    return {"files": files, "bytes": size, "links": links, "specials": specials,
            "tree_sha256": _sha256_bytes(json.dumps(sorted(listing), separators=(",", ":")).encode())}


def _has_symlink(path: Path) -> bool:
    for _directory, names, filenames, descriptor in os.fwalk(path, follow_symlinks=False):
        for name in names + filenames:
            if stat.S_ISLNK(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                return True
    return False


def _read_records(directory: Path) -> list[tuple[str, dict]]:
    if not _lexists(directory):
        return []
    stage._safe_directory(directory)
    found = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or (name.startswith(".") and not name.startswith(".staging-")):
            continue
        found.append((name, stage._read_json(directory / name)))
    return found


# --- copy catalog (read-only, stdlib) ----------------------------------------------------------------

def catalog_records(root: Path) -> list[dict]:
    """Every admission/copies record's current revision (kind, destination, identity, preparation).

    A stdlib reader of the hash chain src.memory.copy_catalog writes: head ->
    revision file whose sha256 is the head's; then back through every
    previous_sha256. A broken chain refuses (P8 cannot then be proven).
    """
    directory = root / "admission" / "copies"
    records = []
    if not _lexists(directory):
        return records
    stage._safe_directory(directory)
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        entry = directory / name
        try:
            head = stage._read_json(entry / "head.json")
            revision, digest, selected = head["revision"], head["sha256"], None
            while revision:
                path = entry / f"{revision:012d}-{digest}.json"
                raw = path.read_bytes()
                if _sha256_bytes(raw) != digest:
                    raise ValueError("chain")
                event = json.loads(raw)
                if selected is None:
                    selected = dict(event["record"], catalog_head_sha256=head["sha256"])
                if revision == 1:  # the reservation's own file: its mtime dates the reservation (R3-3)
                    selected["reserved_at"] = os.lstat(path).st_mtime
                digest, revision = event["previous_sha256"], revision - 1
        except (OSError, ValueError, KeyError, TypeError):
            _refuse("runtime_admission_compaction_catalog_unreadable")
        records.append(selected)
    return records


def cleanup_claims(root: Path) -> dict:
    """copy_id -> last copy-cleanup claim phase (admission/copy-cleanup/v1/claims)."""
    directory = root / "admission" / "copy-cleanup" / "v1" / "claims"
    phases = {}
    if not _lexists(directory):
        return phases
    for name in sorted(os.listdir(directory)):
        try:
            head = stage._read_json(directory / name / "head.json")
            raw = (directory / name / f"{head['revision']:012d}-{head['sha256']}.json").read_bytes()
            phases[name] = json.loads(raw)["state"]["phase"]
        except (OSError, ValueError, KeyError, TypeError):
            phases[name] = "unknown"
    return phases


# --- the retirement set (design v3 retirement_set_v3, v4 F5/F6) --------------------------------------

def _bindings(root: Path) -> dict:
    found = {}
    for name, record in _read_records(root / "admission" / "bindings"):
        found[name[:-5]] = record
    return found


def _marker_proof(root: Path, name: str, state: Path, bindings: dict) -> dict | None:
    """Proof (a): a binding of this slot plus its valid state marker (v4 F5)."""
    marker = state / stage._MARKER
    try:
        if not stat.S_ISREG(os.lstat(marker).st_mode):
            return None
        value = stage._read_json(marker)
    except (OSError, ValueError):
        return None
    identity = value.get("binding_id")
    binding = bindings.get(identity)
    if (value.get("admission_root") != str(root) or binding is None or binding.get("slot") != name
            or binding.get("state_root") != str(state) or value.get("state_root") != str(state)
            or binding.get("runtime_root") != str(state.parent / "runtime")):
        return None
    return {"proof": "binding", "state_identity": binding["state_identity"],
            "binding_ids": sorted(key for key, item in bindings.items()
                                  if item.get("slot") == name and item.get("state_root") == str(state))}


def _prepare_proof(root: Path, name: str, slot: Path, state: Path, prepared: list) -> dict | None:
    """Proof (b): a 0c prepare record, or its staging record when the slot is the renamed staging dir."""
    slot_identity = _identity(slot)
    for _file, record in prepared:
        if record.get("kind") == "prepared" and record.get("slot") == name and \
                record.get("slot_identity") == slot_identity:
            return {"proof": "prepare_record", "state_identity": record["state_identity"], "binding_ids": []}
    for _file, record in prepared:
        if record.get("kind") == "staging" and record.get("identity") == slot_identity:
            return {"proof": "staging_record", "state_identity": _identity(state), "binding_ids": []}
    return None


def _delegation_proof(root: Path, state: Path) -> dict | None:
    """Proof (c): a delegation record names this state as its destination."""
    stage.require_no_delegation_records(root)  # SC-1a AUTHZ3-1: no record is read at epoch >= 5
    directory = root / "admission" / "delegations"
    for _name, record in _read_records(directory):
        if record.get("destination_state") == str(state):
            return {"proof": "delegation_destination", "state_identity": record["destination_identity"],
                    "binding_ids": []}
    return None


def _earlier_work(root: Path, current: str | None) -> list[dict]:
    """Directories earlier compactions created and recorded by identity (snapshots, scratch)."""
    found = []
    for log in stage.compaction_logs(root):
        if log["compaction_id"] == current:
            continue
        for event in log["events"]:
            if event["kind"] in {"snapshot", "own_work"} and "identity" in event["data"]:
                found.append({"path": event["data"]["path"], "identity": event["data"]["identity"],
                              "compaction_id": log["compaction_id"]})
    return found


def retirement_set(root: Path, *, k_slot: str, retire_prepare_leftovers: bool,
                   compaction_id: str | None = None) -> tuple[list[dict], list[dict]]:
    """(members, untouched) by positive proof only; untouched entries are only lstat'ed (v3).

    A member is destroyed only if admission/ proves the controller made it and
    it still has the recorded identity. Everything else is untouched whatever
    it contains: listed by name and type, never opened (Q4: backups, logs,
    deployments, dashboard-proxy, controller/, archives, unmanaged slots).
    """
    root = Path(root)
    bindings = _bindings(root)
    prepared = _read_records(stage.prepared_directory(root))
    completed_staging = {record.get("staging_id") for _file, record in prepared if record.get("kind") == "prepared"}
    members, untouched = [], []
    copies = catalog_records(root)
    retired = stage.retired_locators(root)

    def keep(relative: str, klass: str) -> None:
        path = root / relative
        untouched.append({"path": relative, "type": _kind(path), "class": klass})

    for name in sorted(os.listdir(root)):
        path = root / name
        if name in FIXED_ENTRIES:
            continue
        if name.startswith(".activate-"):
            if _kind(path) == "directory" and _selection_leftover(path):
                members.append({"class": "selection_leftover", "path": str(path), "identity": _identity(path),
                                "proof": "selection_shape"})
                continue
        keep(name, "unknown")
    slots = root / "slots"
    if _lexists(slots):
        for name in sorted(os.listdir(slots)):
            path = slots / name
            relative = "slots/" + name
            if name == k_slot:
                continue
            if name.startswith(".prepare-"):
                record = next((record for _file, record in prepared if record.get("kind") == "staging"
                               and record.get("identity") == _identity(path)
                               and record.get("staging_id") not in completed_staging), None)
                if record is None:
                    keep(relative, "possible_state_copy")
                elif retire_prepare_leftovers:
                    members.append({"class": "prepare_leftover", "path": str(path), "identity": _identity(path),
                                    "proof": "staging_record"})
                else:
                    keep(relative, "prepare_leftover_kept")
                continue
            if _SLOT.fullmatch(name) is None or _kind(path) != "directory":
                keep(relative, "unknown")
                continue
            if name in retired["slots"] or str(path / "state") in retired["paths"]:
                keep(relative, "retired_locator_reused")  # prepare refuses retired names: made by hand
                continue
            state, runtime = path / "state", path / "runtime"
            if _kind_or_none(state) != "directory" or _kind_or_none(runtime) != "directory":
                keep(relative, "possible_managed_copy" if _lexists(state) else "unknown")
                continue
            proof = (_marker_proof(root, name, state, bindings) or _prepare_proof(root, name, path, state, prepared)
                     or _delegation_proof(root, state))
            if proof is None:
                keep(relative, "possible_managed_copy")
                continue
            if proof["state_identity"] != _identity(state) or _has_symlink(state) or not _runtime_links_ok(runtime):
                keep(relative, "possible_managed_copy")  # identity changed or a link container: never walked
                continue
            extra = _unexpected_entries({"class": "managed_slot", "path": str(path)})
            if extra:
                # Review F6 / r2: the proof covers what the controller made (runtime/ with exactly the
                # release's files and links, state/); anything else was made by hand, so the whole
                # slot is kept. Names only, never contents.
                keep(relative, "unexpected_entries")
                untouched[-1]["names"] = extra[:32]
                continue
            if proof["proof"] == "staging_record" and not retire_prepare_leftovers:
                # Review r2 (F6 residual): a staging record proves the directory, not its state/
                # (no prepare record holds a state identity), so it is retired only with the
                # operator's --retire-prepare-leftovers, like any other prepare leftover.
                keep(relative, "prepare_leftover_kept")
                continue
            members.append({"class": "managed_slot", "path": str(path), "identity": _identity(path),
                            "proof": proof["proof"], "slot": name, "state_root": str(state),
                            "state_identity": proof["state_identity"], "binding_ids": proof["binding_ids"]})
    compaction = root / "compaction"
    if _lexists(compaction):
        recorded = _earlier_work(root, compaction_id)
        for name in sorted(os.listdir(compaction)):
            if name == compaction_id:
                continue
            directory = compaction / name
            for entry in sorted(os.listdir(directory)) if _kind(directory) == "directory" else []:
                path = directory / entry
                match = next((item for item in recorded if item["path"] == str(path)
                              and item["identity"] == _identity(path)), None)
                if match is not None and _kind(path) == "directory":
                    members.append({"class": "earlier_compaction", "path": str(path), "identity": _identity(path),
                                    "proof": "compaction_log:" + match["compaction_id"]})
                else:
                    keep("compaction/" + name + "/" + entry, "unknown")
    untouched += _attach_copies(root, members, copies, k_slot, compaction_id)
    for index, member in enumerate(sorted(members, key=_member_order), start=1):
        member["n"] = index
    members.sort(key=lambda item: item["n"])
    untouched.sort(key=lambda item: item["path"])
    return members, untouched


def _kind_or_none(path: Path) -> str | None:
    return _kind(path) if _lexists(path) else None


def _runtime_links_ok(runtime: Path) -> bool:
    """A slot runtime holds exactly the config, data and .env links (K29), nothing else linked."""
    for directory, names, filenames, descriptor in os.fwalk(runtime, follow_symlinks=False):
        for name in names + filenames:
            if stat.S_ISLNK(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                if os.path.abspath(directory) != os.path.abspath(runtime) or name not in RUNTIME_LINKS:
                    return False
            elif not (stat.S_ISREG(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode)
                      or stat.S_ISDIR(os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode)):
                return False
    return True


def _selection_leftover(path: Path) -> bool:
    """An .activate-* leftover is empty or holds exactly 'current' -> slots/<name>/runtime (v4 F6)."""
    names = os.listdir(path)
    if not names:
        return True
    if names != ["current"] or not os.path.islink(path / "current"):
        return False
    target = os.readlink(path / "current").split("/")
    return len(target) == 3 and target[0] == "slots" and target[2] == "runtime" and _SLOT.fullmatch(target[1])


def _member_order(member: dict) -> tuple:
    return (MEMBER_CLASSES.index(member["class"]), member["path"])


def _attach_copies(root: Path, members: list, copies: list, k_slot: str, compaction_id: str | None = None) -> list:
    """Catalog copies under the root: attached to the member that contains them, or members themselves.

    Copies inside K's state, and inside this compaction's own scratch (its
    snapshots, own work attached at retire), are not plan members.
    """
    kept = []
    skipped = [str(root / "slots" / k_slot / "state")]
    if compaction_id is not None:
        skipped.append(str(root / "compaction" / compaction_id))
    for record in copies:
        destination = record["destination"]
        if any(destination == item or destination.startswith(item + "/") for item in skipped):
            continue
        holder = next((member for member in members if destination == member["path"]
                       or destination.startswith(member["path"].rstrip("/") + "/")), None)
        if holder is not None:
            holder.setdefault("copy_ids", []).append(record["copy_id"])
            continue
        path = Path(destination)
        if not _lexists(path):
            continue
        if record["root_identity"] is not None and record["root_identity"] == _identity(path) \
                and _kind(path) == "directory" and Path(destination).is_relative_to(root):
            member = {"class": "catalog_copy", "path": destination, "identity": _identity(path),
                      "proof": "copy_catalog", "copy_ids": [record["copy_id"]], "allowed_entries": _copy_entries(record)}
            extra = _unexpected_entries(member)
            if extra:
                # Review r2 N7: a catalogued copy holds only the extents its record lists (state/,
                # blobs/, manifest.json...). Anything else beside them (a hand backup in backups/) keeps
                # the whole copy, listed by name.
                kept.append({"path": Path(destination).relative_to(root).as_posix(), "type": "directory",
                             "class": "unexpected_entries", "names": extra[:32]})
                continue
            members.append(member)
    return kept


def _copy_entries(record: dict) -> list[str]:
    """The top-level names a catalogued copy may hold (review r2 N7): its extents' first components
    (an extent '.' adds none) and, once prepared, every name of its preparation inventory."""
    names = set()
    extents = record.get("extents") if type(record.get("extents")) is dict else {}
    relatives = [extent.get("relative") for extent in extents.values() if type(extent) is dict]
    evidence = record.get("prepared_record") if type(record.get("prepared_record")) is dict else {}
    relatives.append(evidence.get("relative"))
    inventory = evidence.get("inventory") if type(evidence.get("inventory")) is dict else {}
    relatives += list(inventory)
    for relative in relatives:
        if type(relative) is str and Path(relative).parts and Path(relative).parts[0] not in {".", ".."}:
            names.add(Path(relative).parts[0])
    return sorted(names)


def untouched_digest(untouched: list[dict]) -> str:
    return _sha256_bytes(json.dumps(untouched, sort_keys=True, separators=(",", ":")).encode())


# --- preconditions ----------------------------------------------------------------------------------

def _privacy_relative(state: Path) -> str:
    """The one database holding the erasure journal (the derivation's privacy store)."""
    found = []
    for path in sorted(state.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in _SQLITE_SUFFIXES:
            with stage._database_image(path) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='erasure_targets'").fetchone():
                    found.append(path.relative_to(state).as_posix())
    if len(found) != 1:
        _refuse("runtime_admission_compaction_journal_ambiguous")
    return found[0]


def collection_kind(name: str, *, policy: bool, store: str) -> str | None:
    """SC-4c: 'rebuilt' (H's own policy memory collection), 'copied' (command index), 'dropped' (the named
    drops: routines only in ``store`` data/chroma_db, the legacy memory collection only in data/memory_db),
    else None."""
    if policy:
        if name.endswith(CHROMA_POLICY_SUFFIX) and len(name) > len(CHROMA_POLICY_SUFFIX):
            return "rebuilt"
    elif CHROMA_COMMAND_NAMES.fullmatch(name):
        return "copied"
    elif (store, name) in CHROMA_DROPPED_IN_STORE:
        return "dropped"
    return None


def chroma_collections(state: Path, privacy_relative: str) -> dict:
    """SC-4c at plan: every collection of every Chroma store of H against the closed list.

    Reads private images of H's chroma.sqlite3 (collection names only, never
    a record) and H's semantic_index targets; an unlisted collection (a
    look-alike of a dropped name, or a dropped name outside its own store,
    among them) or a store with no collection table refuses with
    compaction_unlisted_collection. Returns content-free counts, the dropped
    collections by name only (``dropped_collections``, SC4C-Q2b), and the
    files inside each store that are not Chroma's own and so are not carried
    into K (``other_files``, SC4C-RV-5 (4)).
    """
    policy = set()
    with stage._database_image(state / privacy_relative) as db:
        for (encoded,) in db.execute("SELECT descriptor FROM erasure_targets WHERE participant='semantic_index'"):
            descriptor = json.loads(encoded)
            path = descriptor.get("path") if type(descriptor) is dict else None
            if type(path) is str and Path(path).is_relative_to(state):
                policy.add((Path(path).relative_to(state).as_posix(), descriptor.get("collection")))
    stores, unlisted = {}, []
    for relative, path in _databases(state).items():
        if PurePosixPath(relative).name != CHROMA_STORE:
            continue
        store = PurePosixPath(relative).parent.as_posix()
        with stage._database_image(path) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            names = sorted(row[0] for row in db.execute("SELECT name FROM collections")) \
                if "collections" in tables else None
        if names is None:
            unlisted.append([store, None])
            continue
        counts = {"rebuilt": 0, "copied": 0, "dropped": 0, "other_files": _other_store_files(path.parent),
                  "dropped_collections": []}
        for name in names:
            kind = collection_kind(name, policy=(store, name) in policy, store=store)
            if kind is None:
                unlisted.append([store, name])
            else:
                counts[kind] += 1
                if kind == "dropped":
                    counts["dropped_collections"].append(name)
        stores[store] = counts
    if unlisted:
        _refuse(UNLISTED_COLLECTION, {"collections": unlisted})
    return {"stores": stores}


def _other_store_files(store: Path) -> int:
    """How many entries under one Chroma store are not the SDK's own files (links and special files count)."""
    count = 0
    for directory, names, files in os.walk(store, followlinks=False):
        for name in files + [name for name in names if os.path.islink(os.path.join(directory, name))]:
            path = Path(directory) / name
            relative = path.relative_to(store).as_posix()
            if path.is_symlink() or not path.is_file() or not CHROMA_STORE_FILES.fullmatch(relative):
                count += 1
    return count


def owner_selection(root: Path, head_binding: dict) -> list[dict]:
    """P3: the owner binding and state root of every inherited target, recorded in the intent."""
    validation = stage._target_ownership_validation(root, head_binding)
    owners = {}
    for identity, selected in sorted(validation["delegations"].items()):
        source = selected["source"]
        owners.setdefault(selected["record"]["source_binding"], {
            "binding_id": selected["record"]["source_binding"], "slot": source["slot"],
            "state_root": source["state_root"], "state_identity": source["state_identity"], "targets": []})
        owners[selected["record"]["source_binding"]]["targets"].append(identity)
    return [owners[key] for key in sorted(owners)]


def _config_scan(state: Path, needles: list[str]) -> list[str]:
    """P6 (H4): config/** text and .env bytes naming H's, an owner's or a member's absolute path."""
    hits = []
    candidates = [state / ".env"] if _lexists(state / ".env") else []
    if _lexists(state / "config"):
        candidates += sorted(path for path in (state / "config").rglob("*") if path.is_file())
    encoded = [needle.encode() for needle in needles]
    for path in candidates:
        if path.is_symlink():
            _refuse("runtime_admission_compaction_links")
        raw = path.read_bytes()
        if any(needle in raw for needle in encoded):
            hits.append(path.relative_to(state).as_posix())
    return hits


def _runtime_link_set(runtime: Path) -> dict:
    return {name: os.readlink(runtime / name) for name in sorted(RUNTIME_LINKS)
            if os.path.islink(runtime / name)}


def _release_manifest(archive: Path) -> dict:
    manifest, _files = stage._read_archive(Path(archive))
    return manifest


def _release_json_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2) + "\n").encode()


def launcher_contract_ok(root: Path, head_binding: dict) -> None:
    """P13 (v3, v5 G4): controller installed, launcher contract >= 3 installed and in K's (= H's) release.json."""
    if not _lexists(root / "admission" / "controller.json"):
        _refuse("runtime_admission_compaction_launcher_missing")
    directory = root / "admission" / "launcher"
    records = sorted(directory.glob("install-*.json")) if _lexists(directory) else []
    removed = sorted(directory.glob("uninstall-*.json")) if _lexists(directory) else []
    if not records or (removed and removed[-1].name[len("uninstall-"):] >= records[-1].name[len("install-"):]):
        _refuse("runtime_admission_compaction_launcher_missing")
    install = stage._read_json(records[-1])
    contract = install.get("launcher_contract")
    if type(contract) is not int or contract < 3:
        _refuse("runtime_admission_compaction_launcher_contract")
    if stage.release_integers(head_binding).get("launcher_contract", 0) < 3:
        _refuse("runtime_admission_compaction_launcher_contract")


# --- SC-1b: release mode (sc-release-design v4 mechanism.mode, plan.p6_prime) ------------------------

def release_mode() -> bool:
    """Whether new logs are release-mode logs: keyed on this code's epoch, like the AUTHZ3-1 guard."""
    return stage.CONTROLLER_EPOCH >= stage.RELEASE_MODE_EPOCH


def release_log(log: dict) -> bool:
    """A log planned in release mode (intent.mode == 'release'); install-controller reads the same key.

    A log without 'mode' is a plain-mode log; any other 'mode' value refuses
    (runtime_admission_compaction_log_invalid, review RA-2), so no path treats it as plain.
    """
    try:
        return launcher.release_mode_log(log)
    except ValueError as error:
        _refuse(str(error))


def _require_plain_mode(log: dict) -> None:
    """V4R2-A1: Q6, allow-run and --discard-k-writes only finish a log without 'mode'."""
    if release_log(log):
        _refuse(PLAIN_MODE_ONLY)


def served_releases(root: Path, head_binding: dict) -> set[str]:
    """served(root) (P6'(c)): the archives of complete, verified release-mode logs, plus the head's release.

    Derived from immutable intents and events only (CO-2): no release-history
    record. A plain-mode log, a log completed without 'verified' or one still
    pending or aborted serves nothing.
    """
    served = {head_binding["release_sha256"]}
    for log in stage.compaction_logs(root):
        if (log["terminal"] and log["events"][-1]["kind"] == "complete" and release_log(log)
                and _has(log, "verified")):
            served.add(log["intent"]["archive"]["release_sha256"])
    return served


def _triple_text(triple: tuple) -> str:
    return ".".join(str(item) for item in triple)


def _installed_manifest(root: Path) -> dict:
    """controller/current/release.json, the installed controller's manifest (R4-OPS-4)."""
    try:
        manifest = json.loads((root / "controller" / "current" / "release.json").read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        _refuse("runtime_admission_controller_tampered")
    if type(manifest) is not dict:
        _refuse("runtime_admission_controller_tampered")
    return manifest


def release_admission(root: Path, source: dict, manifest: dict) -> dict:
    """P6'(a)-(d) and (g): may the closed head H move to NEW (``manifest``) without a T0?

    (a) NEW's launcher contract and format epoch meet serve's minima; (b)
    5 <= NEW's controller epoch <= the installed one, so R1 (epoch 4) is never
    NEW (release_epoch), and an epoch above the installed one names its T0
    (t0_required with both triples, R5-3); (c) NEW is in served(root) or its controller triple equals the
    installed one, else t0_required (greater pair), controller_behind (lower
    pair) or a revision conflict (equal pair, other closure), each naming both
    triples (R5-3); (d) H passes serve's slot integers; (g) SR-2: NEW's,
    the installed controller's and H's schema contracts are equal and >= 1.
    Returns the archive fields the plan digest binds (e).
    """
    integers = {}
    for key in ("launcher_contract", "format_epoch", "schema_contract", "controller_epoch"):
        value = manifest.get(key, 0)
        if type(value) is not int or value < 0:
            _refuse("runtime_admission_release_integer")
        integers[key] = value
    if integers["launcher_contract"] < stage.LAUNCHER_CONTRACT:  # (a)
        _refuse("runtime_admission_launcher_contract_missing")
    if integers["format_epoch"] < stage.MIN_FORMAT_EPOCH:
        _refuse("runtime_admission_rollback_format_closed")
    installed_manifest = _installed_manifest(root)
    try:
        installed = stage.controller_triple(installed_manifest)
        triple = stage.controller_triple(manifest)
    except ValueError as error:
        _refuse(str(error))
    detail = ":installed=" + _triple_text(installed) + ",archive=" + _triple_text(triple)
    if triple[0] < stage.RELEASE_MODE_EPOCH:  # (b): R1 (epoch 4) is never NEW (SC-D7 A)
        _refuse("runtime_admission_compaction_release_epoch")
    if triple[0] > installed[0]:  # (b) upper bound: a greater triple, so (c)'s named case (review B3)
        _refuse(T0_REQUIRED + detail)
    release_sha256 = _sha256_bytes(_release_json_bytes(manifest))
    if release_sha256 not in served_releases(root, source) and triple != installed:  # (c)
        if triple[:2] > installed[:2]:
            _refuse(T0_REQUIRED + detail)
        if triple[:2] < installed[:2]:
            _refuse(CONTROLLER_BEHIND + detail)
        _refuse("runtime_admission_compaction_controller_revision_conflict" + detail)  # equal pair (CR2-9)
    try:
        stage._require_launcher_slot(source)  # (d)
        head_contract = stage.release_integers(source)["schema_contract"]
    except ValueError as error:
        _refuse(str(error))
    contracts = {integers["schema_contract"], installed_manifest.get("schema_contract", 0), head_contract}
    if len(contracts) != 1 or type(integers["schema_contract"]) is not int or integers["schema_contract"] < 1:
        _refuse("runtime_admission_compaction_schema_contract")  # (g) SR-2, SC-D4 A
    return {"release_sha256": release_sha256, "git_commit": manifest["git_commit"],
            "schema_contract": integers["schema_contract"], "controller_triple": list(triple)}


def _legacy_slot_names(root: Path) -> set[str]:
    """The two legacy slot names of the committed adoption record (container and legacy state slot).

    Read-only; any record that is not exactly one committed (not aborted)
    adoption record exempts nothing (stricter reading).
    """
    directory = stage.adoption_directory(root)
    if not os.path.isdir(directory) or os.path.islink(directory):
        return set()
    committed = []
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("adoption-") and name.endswith(".json")):
            continue
        try:
            record = stage._read_json(directory / name)
        except (OSError, ValueError):
            return set()
        head = record.get("head")
        if (record.get("kind") != "adoption" or type(head) is not dict
                or stage.adoption_record_path(root, head).name != name):
            return set()
        if not stage.head_aborted(root, head):
            committed.append(record)
    if len(committed) != 1:
        return set()
    record = committed[0]
    layout = record.get("layout")
    legacy = layout.get("legacy_state") if type(layout) is dict else None
    if type(record.get("container_slot")) is not str or type(legacy) is not str:
        return set()
    return {record["container_slot"], Path(legacy).parent.name}


def _unproven_copies(root: Path, untouched: list[dict]) -> list[str]:
    """Release mode: every untouched entry under slots/ but the two legacy slots (plan.other)."""
    exempt = _legacy_slot_names(root)
    return sorted(entry["path"] for entry in untouched if entry["path"].startswith("slots/")
                  and entry["path"].split("/")[1] not in exempt)


def _install_logrotate(root: Path) -> tuple[Path, ...]:
    directory = root / "admission" / "launcher"
    records = sorted(directory.glob("install-*.json")) if _lexists(directory) else []
    if not records:
        return ()
    value = stage._read_json(records[-1]).get("logrotate_conf")
    return (Path(value),) if type(value) is str and value else ()


def controller_location(root: Path, members: list[dict]) -> None:
    """P7: no loaded module comes from slots/ or a member; an installed controller runs from controller/."""
    forbidden = [str(root / "slots")] + [member["path"] for member in members]
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        real = os.path.realpath(filename)
        if any(real == item or real.startswith(item.rstrip("/") + "/") for item in forbidden):
            _refuse("runtime_admission_compaction_controller_location")
    if _lexists(root / "admission" / "controller.json"):
        own = os.path.realpath(stage.__file__)
        if not own.startswith(os.path.realpath(root / "controller") + "/"):
            _refuse("runtime_admission_compaction_controller_location")


def _writers_closed(paths: list[str]) -> None:
    """P5: writers closed in H, every owner and every member state (their .lock/.lease flocks)."""
    for value in paths:
        path = Path(value)
        if _lexists(path) and _kind(path) == "directory":
            try:
                stage._assert_writers_closed(path)
            except ValueError:
                _refuse("runtime_admission_writer_active")


def _state_bytes(state: Path) -> int:
    total = 0
    for path in state.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _tree_bytes(path: Path) -> int:
    """Bytes of the regular files under ``path``, symlinks not followed (K's runtime is H's archive)."""
    total = 0
    for directory, _subdirectories, files in os.walk(path):
        for name in files:
            info = os.lstat(os.path.join(directory, name))
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _largest_sqlite(state: Path) -> int:
    """L: the largest SQLite database under the state, its WAL included (what VACUUM rewrites at once)."""
    largest = 0
    for path in state.rglob("*"):
        if path.suffix in SQLITE_SUFFIXES and path.is_file() and not path.is_symlink():
            wal = Path(str(path) + "-wal")
            size = path.stat().st_size + (wal.stat().st_size if wal.is_file() and not wal.is_symlink() else 0)
            largest = max(largest, size)
    return largest


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def sqlite_temp_directory(environ=None) -> Path | None:
    """Where SQLite's unix VFS puts VACUUM's temporary database (stage_runtime's shared helper)."""
    return stage.sqlite_temp_directory(environ)


def free_space_need(state_bytes: int, manifest_bytes: int, largest_sqlite: int, runtime_bytes: int, *,
                    temp_on_root: bool) -> dict:
    """P9 (SC-0b): free bytes the transition needs at its measured peak, on the root and on SQLite's temp.

    The peak is VACUUM of K's largest database, with H already on disk: the snapshot's SQLite backups
    (S) and its manifest (M), K's state (S) and runtime (R), VACUUM's rollback journal (L) beside the
    database and its temporary database (L) in SQLite's temp directory. Until retire, S + M + S + R
    stay allocated.
    """
    root = 2 * state_bytes + manifest_bytes + largest_sqlite + runtime_bytes + FREE_MARGIN
    if temp_on_root:
        return {"root": root + largest_sqlite, "temp": 0}
    return {"root": root, "temp": largest_sqlite + FREE_MARGIN}


def capacity(root: Path, head_binding: dict, privacy_relative: str) -> dict:
    """P9: projected K binding and history archive at most 12 MiB; free space for the measured peak."""
    binding_bytes = len(stage._canonical_json(head_binding))
    state = Path(head_binding["state_root"])
    archive_bytes = binding_bytes
    with stage._database_image(state / privacy_relative) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in sorted(names):
            if name in {"conversation_tombstones", "conversation_erasures", *_SCHEDULING} \
                    or name.startswith("legacy_quarantine"):
                for row in db.execute("SELECT * FROM " + stage._quote(name)):
                    archive_bytes += len(json.dumps([stage._typed(value) for value in row], default=str))
    for record in stage._delegation_records(root, state):
        archive_bytes += len(stage._canonical_json(record))
    state_bytes = _state_bytes(state)
    free = shutil.disk_usage(root).free
    temp = sqlite_temp_directory()
    temp_on_root = temp is None or _device(temp) == _device(root)
    need = free_space_need(state_bytes, MANIFEST_ESTIMATE_FACTOR * state_bytes, _largest_sqlite(state),
                           _tree_bytes(Path(head_binding["runtime_root"])), temp_on_root=temp_on_root)
    temp_free = None if temp_on_root else shutil.disk_usage(temp).free
    report = {"binding_bytes": binding_bytes, "history_archive_bytes": archive_bytes,
              "state_bytes": state_bytes, "free_bytes": free, "limit_bytes": BINDING_MARGIN,
              "need_bytes": need["root"], "temp_need_bytes": need["temp"], "temp_free_bytes": temp_free}
    if (binding_bytes > BINDING_MARGIN or archive_bytes > BINDING_MARGIN or free < need["root"]
            or (temp_free is not None and temp_free < need["temp"])):
        _refuse("runtime_admission_compaction_capacity", report)
    return report


# --- coverage: content-free classification of member-only rows (design v2 coverage, M3) --------------

def _table_rows(db: sqlite3.Connection, name: str) -> tuple[list[str], dict]:
    info = db.execute("PRAGMA table_info(" + stage._quote(name) + ")").fetchall()
    columns = [row[1] for row in info]
    keys = [row[1] for row in sorted(info, key=lambda item: item[5]) if row[5]]
    rows = {}
    for row in db.execute("SELECT * FROM " + stage._quote(name)):
        value = dict(row)
        key = tuple(value[column] for column in keys) if keys else stage._digest(
            [stage._typed(value[column]) for column in columns])
        rows[key] = value
    return columns, rows


def _databases(state: Path) -> dict:
    found = {}
    if not _lexists(state):
        return found
    for path in sorted(state.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in _SQLITE_SUFFIXES:
            found[path.relative_to(state).as_posix()] = path
    return found


def _head_markers(head_state: Path, markers: dict | None = None) -> dict:
    """The head's soft-deletion markers (and ledger) used to classify 'erased'; merged into ``markers``."""
    markers = markers or {"tombstoned": set(), "deleted_facts": set(), "deleted_profiles": set()}
    for _relative, path in _databases(head_state).items():
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "conversation_tombstones" in names:
                markers["tombstoned"] |= {row[0] for row in db.execute(
                    "SELECT conversation_id FROM conversation_tombstones")}
            if "facts" in names:
                markers["deleted_facts"] |= {row[0] for row in db.execute(
                    "SELECT id FROM facts WHERE state IN ('deleted','quarantined')")}
            if "voice_profiles" in names:
                markers["deleted_profiles"] |= {row[0] for row in db.execute(
                    "SELECT id FROM voice_profiles WHERE deleted_at IS NOT NULL")}
    return markers


def _deadline(row: dict) -> float | None:
    values = [row[name] for name in _DEADLINES if type(row.get(name)) in (int, float)]
    sources = row.get("sources_json")
    if type(sources) is str:
        try:
            parsed = json.loads(sources)
            values += [item["expires_at"] for item in parsed if type(item) is dict
                       and type(item.get("expires_at")) in (int, float)]
        except ValueError:
            pass
    return min(values) if values else None


def _conversations(row: dict) -> set:
    found = set()
    if type(row.get("conversation_id")) is str:
        found.add(row["conversation_id"])
    sources = row.get("sources_json")
    if type(sources) is str:
        try:
            found |= {item.get("conversation_id") for item in json.loads(sources) if type(item) is dict}
        except ValueError:
            pass
    return found - {None}


def classify_row(name: str, row: dict, head_row: dict | None, markers: dict, now: float) -> str:
    """'expired', 'erased', 'superseded', 'scrubbed' or 'unexplained' (design v2 coverage classes)."""
    if head_row is not None:
        text_deadline = row.get("text_expires_at")
        differing = {key for key in row if row[key] != head_row.get(key)}
        if type(text_deadline) in (int, float) and text_deadline <= now and differing and all(
                type(row[key]) is str or row[key] is None for key in differing):
            return "scrubbed"
        return "superseded"
    if _conversations(row) & markers["tombstoned"]:
        return "erased"
    if name == "facts" and row.get("id") in markers["deleted_facts"]:
        return "erased"
    if name in {"fact_sources", "fact_dependencies"} and row.get("fact_id") in markers["deleted_facts"]:
        return "erased"
    if name.startswith("voice_profile") and (row.get("id") in markers["deleted_profiles"]
                                             or row.get("profile_id") in markers["deleted_profiles"]):
        return "erased"
    deadline = _deadline(row)
    if deadline is not None and deadline <= now:
        return "expired"
    return "unexplained"


COVERAGE_CLASSES = ("expired", "erased", "superseded", "scrubbed", "unexplained", "removed_by_k_run")


def coverage(members: list[dict], head_state: Path, *, now: float, k_state: Path | None = None) -> dict:
    """Per member: counts of rows and files not present in H (= K at activation), by class. Numbers only.

    Members are read through private SQLite images (never written). Rows
    identical in H are covered; the scheduling tables and SQLite internals are
    rebuilt per state and not counted. Untouched entries get no counts.

    At retire (review F5, decision D2=A) ``k_state`` is K' after its run, and
    the reference stays H: a row (or file) identical in K' is covered; one
    identical in H but gone from K' was removed by K's own run and is counted
    as 'removed_by_k_run', which never blocks. Everything else is classified
    as at plan time, with K''s erasure markers added to H's.
    """
    head_dbs = _databases(head_state)
    markers = _head_markers(head_state)
    k_dbs = _databases(k_state) if k_state is not None else {}
    k_cache: dict = {}
    k_shas: set = set()
    k_files: dict = {}
    if k_state is not None:
        markers = _head_markers(k_state, markers)
        k_files = _plain_files(k_state)
        k_shas = set(k_files.values())
    head_cache: dict = {}
    head_files = _plain_files(head_state)
    report = {}
    totals = dict.fromkeys(COVERAGE_CLASSES, 0)
    for member in members:
        states = []
        root = Path(member["path"])
        if member["class"] == "managed_slot":
            states = [Path(member["state_root"])]
        elif _lexists(root / "state"):
            states = [root / "state"]
        counts = dict.fromkeys(totals, 0)
        expired_ids = set()
        for state in states:
            for relative, path in _databases(state).items():
                if relative.endswith("chroma.sqlite3"):
                    continue
                try:
                    with stage._database_image(path) as db:
                        names = {row[0] for row in db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'")}
                        for name in sorted(names - _INTERNAL_TABLES - _SCHEDULING):
                            _columns, rows = _table_rows(db, name)
                            head_rows = _head_table(head_cache, head_dbs, relative, name)
                            k_rows = _head_table(k_cache, k_dbs, relative, name) if k_state is not None else None
                            for key, row in rows.items():
                                if k_rows is not None and k_rows.get(key) == row:
                                    continue
                                if head_rows.get(key) == row:
                                    if k_rows is not None:
                                        counts["removed_by_k_run"] += 1
                                    continue
                                klass = classify_row(name, row, head_rows.get(key), markers, now)
                                counts[klass] += 1
                                if klass in {"expired", "erased"} and type(row.get("id")) is str:
                                    expired_ids.add(row["id"])
                except sqlite3.Error:
                    # Review F12: a member file named *.db that SQLite cannot read is a coded refusal.
                    _refuse("runtime_admission_compaction_database_unreadable", {"member": member["n"]})
            for relative, signature in _plain_files(state).items():
                if k_state is not None and (k_files.get(relative) == signature or signature in k_shas):
                    continue  # still in K' (relocated collector targets keep their bytes)
                if head_files.get(relative) == signature:
                    if k_state is not None:
                        counts["removed_by_k_run"] += 1
                    continue
                counts["expired" if Path(relative).stem in expired_ids else "unexplained"] += 1
        report[str(member["n"])] = counts
        for key, value in counts.items():
            totals[key] += value
    return {"members": report, "totals": totals}


def _plain_files(state: Path) -> dict:
    """relative path -> (size, sha256) of a state's non-SQLite files (side files, markers, chroma excluded)."""
    found = {}
    if not _lexists(state):
        return found
    for path in state.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix in _SQLITE_SUFFIXES \
                or path.name.endswith(_SIDE_SUFFIXES) or path.name.startswith(".kza-"):
            continue
        relative = path.relative_to(state).as_posix()
        if "/chroma/" in "/" + relative or relative.startswith("chroma/"):
            continue
        found[relative] = (path.stat().st_size, _file_sha(path))
    return found


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_table(cache: dict, head_dbs: dict, relative: str, name: str) -> dict:
    key = (relative, name)
    if key not in cache:
        path = head_dbs.get(relative)
        rows = {}
        if path is not None:
            with stage._database_image(path) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
                    _columns, rows = _table_rows(db, name)
        cache[key] = rows
    return cache[key]


# --- the advisory (Q1, v4 F4, v5 G3) -----------------------------------------------------------------

def _ledgers(state: Path) -> list[dict]:
    found = []
    for _relative, path in _databases(state).items():
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"deletion_ledger_v1", "deletion_ledger_salt_v1"} <= names:
                salt = db.execute("SELECT salt FROM deletion_ledger_salt_v1 WHERE id=1").fetchone()
                # No salt yet: the ledger exists and has never recorded a deletion.
                subjects = {(row[0], row[1]) for row in db.execute(
                    "SELECT kind,subject_sha256 FROM deletion_ledger_v1")} if salt is not None else set()
                found.append({"salt": salt[0] if salt is not None else None, "subjects": subjects})
    return found


def _live_ids(state: Path) -> dict:
    """Ids a state holds live, per family: facts, voice profiles, conversations, users."""
    ids = {"facts": set(), "voice_profiles": set(), "conversations": set(), "users": set()}
    for _relative, path in _databases(state).items():
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "facts" in names:
                ids["facts"] |= {row[0] for row in db.execute(
                    "SELECT id FROM facts WHERE state NOT IN ('deleted','quarantined')")}
            if "voice_profiles" in names:
                ids["voice_profiles"] |= {row[0] for row in db.execute(
                    "SELECT id FROM voice_profiles WHERE deleted_at IS NULL")}
            for table, column in (("capture_conversations", "id"), ("utterances", "conversation_id")):
                if table in names:
                    ids["conversations"] |= {row[0] for row in db.execute(
                        "SELECT DISTINCT " + column + " FROM " + table + " WHERE " + column + " IS NOT NULL")}
    for path in sorted(state.rglob("*_accounts_v1.json")) if _lexists(state) else []:
        try:
            value = stage._read_json(path)
            ids["users"] |= {user.get("id") for user in value.get("users", []) if type(user) is dict
                             and type(user.get("id")) is str and not user.get("retired_at")}
        except (OSError, ValueError):
            continue
    return ids


_FAMILY_KINDS = {"facts": ("fact_deleted", "fact_pruned_retired_owner"), "voice_profiles": ("voice_profile_deleted",),
                 "conversations": ("conversation_erased",), "users": ("user_retired",)}
_FAMILY_REASON = {"facts": "memory_deleted", "voice_profiles": "voice_profile_deleted",
                  "conversations": "conversation_erased", "users": "user_retired"}


def advisory_counts(head_state: Path, retained: list[Path], *, owners: int) -> dict:
    """compaction_recommended = owners >= 6 or ledger-backed deletions retained elsewhere (v5 G3).

    'deleted' counts ids some retained managed state still holds live whose
    salted digest the head's deletion ledger records with a user-requested
    kind. 'absent' (missing from the head without a ledger entry: natural
    expiry, retention, rollback cuts) is informational only. Without a ledger
    (a head older than U3b-0d3) 'deleted' falls back to the head's soft-delete
    markers and the output says 'ledger_absent'. Ids stay in memory; the
    output holds integers and kinds only.
    """
    ledgers = _ledgers(head_state)
    head_ids = _live_ids(head_state)
    markers = _head_markers(head_state)
    deleted = dict.fromkeys(_FAMILY_KINDS, 0)
    absent = dict.fromkeys(_FAMILY_KINDS, 0)
    for state in retained:
        ids = _live_ids(state)
        for family, values in ids.items():
            for identity in values - head_ids[family]:
                if ledgers:
                    hit = any((kind, launcher.subject_digest(ledger["salt"], identity)) in ledger["subjects"]
                              for ledger in ledgers if ledger["salt"] is not None for kind in _FAMILY_KINDS[family])
                else:
                    hit = ((family == "facts" and identity in markers["deleted_facts"])
                           or (family == "voice_profiles" and identity in markers["deleted_profiles"])
                           or (family == "conversations" and identity in markers["tombstoned"]))
                if hit:
                    deleted[family] += 1
                else:
                    absent[family] += 1
    reasons = []
    if owners >= OWNER_WARNING:
        reasons.append({"kind": "owner_count", "owners": owners})
    for family, count in deleted.items():
        if count:
            reasons.append({"kind": _FAMILY_REASON[family], "pending": count, "retained_states": len(retained)})
    return {"compaction_recommended": bool(reasons), "reasons": reasons, "owners": owners,
            "deleted": deleted, "absent": absent, "ledger": "present" if ledgers else "ledger_absent",
            "retained_states": len(retained)}


def advisory(root: Path, *, owners: int) -> dict:
    """The advisory for delegate's output and 'compact plan' (read-only, never blocks)."""
    root = Path(root)
    head = stage.read_admission_head(root)
    if head is None:
        return {"available": False, "code": "no_head"}
    binding = stage.read_binding(root, head["binding_id"])
    members, _untouched = retirement_set(root, k_slot="", retire_prepare_leftovers=False)
    retained = [Path(member["state_root"]) for member in members
                if member["class"] == "managed_slot" and member["state_root"] != binding["state_root"]]
    return {"available": True, **advisory_counts(Path(binding["state_root"]), retained, owners=owners)}


# --- the plan ----------------------------------------------------------------------------------------

def _plan_value(root: Path, k_slot: str, archive: Path, *, retire_prepare_leftovers: bool, manager_scan,
                wall, compaction_id: str | None = None, named_units: tuple[str, ...] = ()) -> dict:
    """Every read-only check of P1-P14 and the retirement set; the value whose digest the operator acks."""
    stage._slot_name(k_slot)
    release = release_mode()  # SC-1b: the only mode for new logs at epoch >= 5
    if release:
        retire_prepare_leftovers = True  # plan.other: forced on in release mode
    head = stage.read_admission_head(root)
    if head is None or head["status"] != "closed":
        _refuse("runtime_admission_not_closed")
    # SC-1a AUTHZ3-1 (mechanism.authz3_1, V4R2-A1): at controller epoch >= 5 a delegation record
    # anywhere under the root, or a head whose policy lists owners' targets, refuses before the
    # head is verified (whose ownership validation would name a record or a missing one instead).
    try:
        stage.require_no_delegation_records(root)
        stage.require_no_owners(stage.read_binding(root, head["binding_id"]), stage.COMPACTION_OWNERS_PRESENT)
    except ValueError as error:
        _refuse(str(error))
    source = stage.verify_closed_binding(root, head["binding_id"])  # P1, P2
    if stage.head_aborted(root, head):
        _refuse("runtime_admission_adoption_aborted_head")
    if release and recovery_closed(root, head):
        _refuse(RECOVERY_HEAD)
    owners = owner_selection(root, source)  # P3
    if _lexists(root / "slots" / k_slot):
        _refuse("runtime_admission_compaction_k_exists")
    if k_slot in stage.retired_locators(root)["slots"]:
        _refuse("runtime_admission_retired_locator")
    members, untouched = retirement_set(root, k_slot=k_slot, retire_prepare_leftovers=retire_prepare_leftovers,
                                        compaction_id=compaction_id)
    if any(entry["path"] == "slots/" + k_slot for entry in untouched):
        _refuse("runtime_admission_compaction_k_exists")
    if release:
        unproven = _unproven_copies(root, untouched)
        if unproven:
            _refuse("runtime_admission_compaction_release_unproven_copy", {"paths": unproven})
    head_member = next((member for member in members if member.get("state_root") == source["state_root"]), None)
    if head_member is None:
        _refuse("runtime_admission_compaction_head_unproven")
    for owner in owners:
        if not any(member.get("state_root") == owner["state_root"] for member in members):
            _refuse("runtime_admission_compaction_owner_unproven")
    _writers_closed([source["state_root"]] + [owner["state_root"] for owner in owners]
                    + [member.get("state_root") or member["path"] for member in members])  # P5
    manifest = _release_manifest(archive)
    if release:
        archive_fields = release_admission(root, source, manifest)  # P6'(a)-(d), (g)
    else:  # P6 (plain mode, epoch <= 4): K's runtime is H's archive
        if (_sha256_bytes(_release_json_bytes(manifest)) != source["release_sha256"]
                or manifest["git_commit"] != source["code_commit"]):
            _refuse("runtime_admission_compaction_artifact")
        archive_fields = {"release_sha256": source["release_sha256"], "git_commit": manifest["git_commit"]}
    state = Path(source["state_root"])
    needles = [source["state_root"]] + [owner["state_root"] for owner in owners] + \
        [member["path"] for member in members]
    if _config_scan(state, needles):
        _refuse("runtime_admission_compaction_absolute_locator")
    controller_location(root, members)  # P7
    copies = catalog_records(root)  # P8
    claims = cleanup_claims(root)
    abandoned = []
    for record in copies:
        if not Path(record["destination"]).is_relative_to(root):
            _refuse("runtime_admission_compaction_copy_outside_root")
        if claims.get(record["copy_id"]) in {"claimed", "running", "unknown"}:
            _refuse("runtime_admission_compaction_copy_in_progress")
        if record.get("preparation") in {"reserved", "writing"}:
            # Review r2 N10: a copy still being written by a live owner refuses; one whose owner
            # process is gone (the catalog never repairs records) is abandoned, listed by id, and
            # treated as 'unknown' (a member only when its root identity still matches).
            if _owner_alive(record.get("preparation_owner"), record.get("reserved_at")):
                _refuse("runtime_admission_compaction_copy_in_progress")
            abandoned.append(record["copy_id"])
        elif record.get("preparation") not in {"prepared", "failed", "unknown"}:
            _refuse("runtime_admission_compaction_copy_in_progress")
    privacy = _privacy_relative(state)
    chroma = chroma_collections(state, privacy)  # SC-4c: the closed collection list
    room = capacity(root, source, privacy)  # P9
    now = _now(wall)
    covered = coverage(members, state, now=now)  # P10
    launcher_contract_ok(root, source)  # P13
    legacy = launcher.legacy_retirements(root)
    retired = {value["plan_sha256"] for name, value in legacy if name.startswith(launcher.LEGACY_RETIRED_PREFIX)}
    if any(value["plan_sha256"] not in retired for _name, value in legacy):
        _refuse("runtime_admission_compaction_legacy_retirement_pending")  # stricter: one deletion at a time
    scanned = manager_scan(root, members)  # P14
    retained = [Path(member["state_root"]) for member in members
                if member["class"] == "managed_slot" and member["state_root"] != source["state_root"]]
    advice = advisory_counts(state, retained, owners=len(owners))
    value = {"schema_version": SCHEMA_VERSION, "head": head, "source_binding_id": head["binding_id"],
             "k_slot": k_slot, "k_state_root": str(root / "slots" / k_slot / "state"),
             "archive": archive_fields,
             "privacy_relative": privacy, "owners": owners, "members": members,
             "untouched": untouched, "untouched_sha256": untouched_digest(untouched),
             "coverage": covered["totals"], "coverage_members": covered["members"], "capacity": {
                 key: room[key] for key in ("binding_bytes", "history_archive_bytes", "state_bytes", "need_bytes",
                                            "temp_need_bytes")},
             "units": scanned, "advisory": advice, "has_env": _lexists(state / ".env"),
             "retire_prepare_leftovers": retire_prepare_leftovers, "named_units": sorted(set(named_units)),
             "abandoned_copies": sorted(abandoned), "chroma": chroma,
             "decisions": list(DECISIONS)}
    if release:
        value["mode"] = RELEASE_MODE  # P6'(e): the digest and the intent bind it with the archive fields
    return value


def recovery_closed(root: Path, head: dict) -> bool:
    """Whether a closed head was closed by a recovery record (row PA), read only.

    Its manual companion is of kind 'recovery' (R-b or ``adopt``); a
    'compaction_return' head (an abort's return to H) is the H it names,
    repeatedly. A later clean close of the same binding writes a 'closure'
    companion, so a recovered head that served and stopped cleanly is a
    source again.
    """
    seen = set()
    while True:
        kind = stage.read_manual_head(root, head)["kind"]
        if kind != "compaction_return":
            return kind == "recovery"
        seen.add(stage._digest(head))
        head = stage._read_compaction_return(root, head)["h_head"]
        if stage._digest(head) in seen:
            _refuse("runtime_admission_manual_companion_changed")


START_TIME_SLACK = 1.0  # seconds: ps lstart has one-second resolution
PROC_SELF = "/proc/self"  # Linux: /proc; elsewhere (the Mac) ps lstart


def _owner_alive(owner, reserved_at: float | None) -> bool:
    """A catalog record's preparation owner is still the process that reserved it (review r3 R3-3).

    The pid alone is reused: the owner counts as alive only if a process with
    that pid exists and started no later than the reservation (the record's
    first revision file), within START_TIME_SLACK. A later start time, an
    unknown reservation time or no such process is a dead owner: the copy is
    abandoned, so a stale record never blocks the plan.
    """
    pid = owner.get("process_id") if type(owner) is dict else None
    if type(pid) is not int or pid <= 0 or reserved_at is None:
        return False
    started = process_started_at(pid)
    return started is not None and started <= reserved_at + START_TIME_SLACK


def process_started_at(pid: int) -> float | None:
    """A live pid's start as wall-clock seconds (/proc on Linux, ps lstart elsewhere), or None."""
    if os.path.isdir(PROC_SELF):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()
            boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                        if line.startswith("btime "))
        except (OSError, StopIteration, ValueError, IndexError):
            return None
        if len(fields) <= 19 or fields[0] == "Z":
            return None
        return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    try:
        # Review r4 R4-7: ps prints lstart in its TZ; it gets UTC and the result is read as UTC, whatever
        # the caller's TZ (time.mktime would use the caller's zone).
        completed = subprocess.run(["ps", "-o", "stat=,lstart=", "-p", str(pid)], capture_output=True, text=True,
                                   timeout=2, check=False, env={"LC_ALL": "C", "PATH": "/bin:/usr/bin",
                                                                "TZ": "UTC0"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    if completed.returncode != 0 or not output or output.startswith("Z"):
        return None
    try:
        return float(calendar.timegm(time.strptime(output.split(None, 1)[-1].strip(), "%a %b %d %H:%M:%S %Y")))
    except ValueError:
        return None


def plan_digest(value: dict) -> str:
    """The digest the operator acknowledges; volatile measurements (sizes, free space) are left out."""
    stable = {key: item for key, item in value.items() if key not in {"capacity", "advisory"}}
    return stage._digest(stable)


def _default_scan(manager=None, named: tuple[str, ...] = ()):
    def scan(root: Path, members: list[dict]) -> dict:
        roots = [{"path": member["path"], "identity": member["identity"]} for member in members]
        roots += [{"path": member["state_root"], "identity": member["state_identity"]}
                  for member in members if member.get("state_root")]
        try:
            return units.scan_units(units.strict_systemctl(manager or launcher.ServiceManager()), root=root,
                                    members=roots, extra_logrotate=_install_logrotate(root), named=named)
        except units.UnitScanRefusal as refusal:
            _refuse(str(refusal))
    return scan


def compact_plan(root, k_slot: str, archive, *, retire_prepare_leftovers: bool = False, scan=None,
                 wall=time.time, units: tuple[str, ...] = ()) -> dict:
    """``compact plan``: read-only; returns the plan, its digest and the untouched-list digest."""
    root = checked_root(root)
    with stage.authority_memo(), stage.AdmissionController(root):  # SC-0b O1: content-addressed
        if stage.pending_compaction(root) is not None:
            _refuse("runtime_admission_compaction_pending")
        value = _plan_value(root, k_slot, Path(archive), retire_prepare_leftovers=retire_prepare_leftovers,
                            manager_scan=scan or _default_scan(named=tuple(units)), wall=wall,
                            named_units=tuple(units))
    return {**_plan_view(value), "plan_sha256": plan_digest(value)}


def _plan_view(value: dict) -> dict:
    """The operator's content-free view: classes, sizes and counts, never file names inside members."""
    return {"k_slot": value["k_slot"], "owners": len(value["owners"]),
            "members": [{key: member.get(key) for key in ("n", "class", "path", "proof", "slot")}
                        | _walk_census(Path(member["path"])) for member in value["members"]],
            "untouched": value["untouched"], "untouched_sha256": value["untouched_sha256"],
            "abandoned_copies": value.get("abandoned_copies", []),
            "coverage": value["coverage"], "coverage_members": value["coverage_members"],
            "capacity": value["capacity"], "units": value["units"], "advisory": value["advisory"],
            "chroma": value["chroma"]}


# --- start: intent, K's runtime, worker, authorize, activate -----------------------------------------

def compact_start(root, k_slot: str, archive, *, plan_sha256: str, acknowledge_untouched: str,
                  destroy_rollback_states: bool, accept_owner_only_loss: bool = False,
                  retire_prepare_leftovers: bool = False, scan=None, worker=None, wall=time.time,
                  units: tuple[str, ...] = ()) -> dict:
    """``compact start``: P11/P12 acknowledgements, the intent, then C2a-C5 (resumable)."""
    root = checked_root(root)
    if not destroy_rollback_states:
        _refuse("runtime_admission_compaction_not_acknowledged")
    with stage.authority_memo(), stage.AdmissionController(root):
        if stage.pending_compaction(root) is not None:
            _refuse("runtime_admission_compaction_pending")
        value = _plan_value(root, k_slot, Path(archive), retire_prepare_leftovers=retire_prepare_leftovers,
                            manager_scan=scan or _default_scan(named=tuple(units)), wall=wall,
                            named_units=tuple(units))
        if plan_digest(value) != plan_sha256:
            _refuse("runtime_admission_compaction_plan_changed")
        if value["untouched_sha256"] != acknowledge_untouched:
            _refuse("runtime_admission_compaction_plan_changed")
        if value["coverage"]["unexplained"] and not accept_owner_only_loss:
            _refuse("runtime_admission_compaction_owner_only_data", {"coverage": value["coverage"]})
        compaction_id = uuid4().hex
        intent = {**value, "compaction_id": compaction_id, "kind": "compaction_intent", "plan_sha256": plan_sha256,
                  "accept_owner_only_loss": accept_owner_only_loss, "archive_path_sha256": _sha256_bytes(
                      str(Path(archive).resolve()).encode()), "created_at": _now(wall)}
        intent.pop("capacity")
        stage._atomic_record(stage.compaction_log_path(root, compaction_id) / "intent.json", intent, immutable=True)
    logger.info("[Compaction] intent %s written; %d members", compaction_id, len(value["members"]))
    return _continue(root, compaction_id, archive=Path(archive), worker=worker, wall=wall)


def _log(root: Path, compaction_id: str) -> dict:
    return stage.read_compaction_log(root, compaction_id)


def _has(log: dict, kind: str) -> bool:
    return bool(stage.compaction_events(log, kind))


def _last(log: dict, kind: str) -> dict | None:
    events = stage.compaction_events(log, kind)
    return events[-1] if events else None


def _k_runtime(root: Path, log: dict, archive: Path, wall) -> dict:
    """C2a: K from the intent's archive (H's, or NEW's in release mode), with an empty placeholder state."""
    intent = log["intent"]
    compaction_id = log["compaction_id"]
    slot = root / "slots" / intent["k_slot"]
    if not _lexists(slot):
        # Review B1: a resume given another archive than the intent's (H's, after a cut start in
        # release mode) refuses before anything is written, so the next resume with NEW still works.
        if _sha256_bytes(_release_json_bytes(_release_manifest(archive))) != intent["archive"]["release_sha256"]:
            _refuse("runtime_admission_compaction_artifact", {"archive": "not_the_intent_archive"})
        _destroy_own_staging(root, log, wall)
        log = _log(root, compaction_id)
        placeholder = root / "compaction" / compaction_id / "placeholder"
        # Review r2 N9: the path is logged before mkdir, so a cut before 'own_work' leaves a
        # placeholder that resume (and abort, through own_work()) still knows is this compaction's.
        if not any(event["data"].get("path") == str(placeholder)
                   for event in stage.compaction_events(log, "own_work_reserved")):
            log = stage.append_compaction_event(root, log, "own_work_reserved", {"path": str(placeholder)},
                                                at=_now(wall))
        if not any(event["data"]["path"] == str(placeholder) for event in stage.compaction_events(log, "own_work")):
            stage.private_directories(placeholder / "config")
            stage.private_directories(placeholder / "data")
            if sorted(os.listdir(placeholder)) != ["config", "data"] or os.listdir(placeholder / "config") \
                    or os.listdir(placeholder / "data"):
                _refuse("runtime_admission_compaction_identity")  # only ever two empty directories
            log = stage.append_compaction_event(root, log, "own_work", {
                "path": str(placeholder), "identity": _identity(placeholder)}, at=_now(wall))
        stage._prepare_slot(archive, root, intent["k_slot"], placeholder, compaction=compaction_id,
                            env_link=intent["has_env"])
    record = _k_proof(root, log)
    if record is None or record["slot_identity"] != _identity(slot):
        _refuse("runtime_admission_compaction_k_identity")
    source = stage.read_binding(root, intent["source_binding_id"])
    runtime = slot / "runtime"
    # P6'(f): K's release.json is the intent's archive (H's own for a plain-mode log, NEW's in release mode).
    if _sha256_bytes((runtime / "release.json").read_bytes()) != intent["archive"]["release_sha256"]:
        _refuse("runtime_admission_compaction_artifact")
    if _runtime_link_set(runtime) != _runtime_link_set(Path(source["runtime_root"])):
        _refuse("runtime_admission_compaction_links")
    return stage.append_compaction_event(root, log, "k_runtime", {
        "slot": intent["k_slot"], "runtime_root": str(runtime), "state_root": str(slot / "state"),
        "slot_identity": record["slot_identity"], "state_identity": record["state_identity"],
        "release_sha256": record["release_sha256"]}, at=_now(wall))


def _k_proof(root: Path, log: dict) -> dict | None:
    """K's slot as this compaction prepared it: its prepare record, else its staging record (review F2).

    _prepare_slot renames the staging directory to slots/<K> and only then
    writes the prepare record. After a cut in between, slots/<K> still has
    the staging directory's {device, inode}, which its staging record (kind
    staging, this compaction_id, no completed prepare) names.
    """
    intent = log["intent"]
    slot = root / "slots" / intent["k_slot"]
    records = [record for _name, record in _read_records(stage.prepared_directory(root))]
    for record in records:
        if record.get("kind") == "prepared" and record.get("slot") == intent["k_slot"] \
                and record.get("compaction_id") == log["compaction_id"]:
            return record
    completed = {record.get("staging_id") for record in records if record.get("kind") == "prepared"}
    if not _lexists(slot) or _kind(slot) != "directory":
        return None
    for record in records:
        if (record.get("kind") == "staging" and record.get("compaction_id") == log["compaction_id"]
                and record.get("staging_id") not in completed and record.get("identity") == _identity(slot)):
            release = slot / "runtime" / "release.json"
            if not _lexists(release) or _kind(slot / "state") != "directory":
                return None  # never: the rename happens after extraction; refuse rather than guess
            return {"kind": "staging", "slot": intent["k_slot"], "slot_identity": record["identity"],
                    "state_identity": _identity(slot / "state"),
                    "release_sha256": _sha256_bytes(release.read_bytes())}
    return None


def _destroy_own_staging(root: Path, log: dict, wall) -> None:
    """C2a resume: a .prepare-* staging leftover of this compaction's K is removed before retrying."""
    for _name, record in _read_records(stage.prepared_directory(root)):
        if record.get("kind") != "staging":
            continue
        path = Path(record["path"])
        if _lexists(path) and _identity(path) == record["identity"] and path.parent == root / "slots":
            completed = any(item.get("staging_id") == record["staging_id"] and item.get("kind") == "prepared"
                            for _file, item in _read_records(stage.prepared_directory(root)))
            if not completed and record.get("compaction_id") == log["compaction_id"]:
                _unlink_tree(root, path, k_inodes=set(), runtime_links=True)


def worker_command(root: Path, compaction_id: str) -> list[str]:
    """The compaction worker's argv: the controller's own runtime tree, '-I -B', fixed bootstrap."""
    runtime = Path(os.path.realpath(stage.__file__)).parents[1]
    from_guard = [sys.executable, "-I", "-B", "-c", "\n".join([
        "import sys",
        "sys.path.insert(0, sys.argv[1])",
        "from src.core.compaction_worker import main",
        "raise SystemExit(main(sys.argv[2:]))",
    ]), str(runtime), "--root", str(root), "--compaction", compaction_id, "--phase", "materialize"]
    return from_guard


def run_worker(root: Path, compaction_id: str) -> dict:
    """Spawn the worker (G1 capability) and return its JSON line; a refusal is the worker's code."""
    environment = {key: os.environ[key] for key in WORKER_ENVIRONMENT if key in os.environ}
    environment["KZA_COMPACTION_WORKER"] = "1"
    # Private whatever the operator shell's umask (0002 for kza): review S1, same class as F5.
    completed = subprocess.run(worker_command(root, compaction_id), capture_output=True, text=True,
                               env=environment, stdin=subprocess.DEVNULL, check=False,
                               umask=stage.MANAGED_CHILD_UMASK)
    if completed.returncode != 0:
        code = (completed.stderr.strip().splitlines() or ["runtime_admission_compaction_worker_failed"])[-1]
        _refuse(code[:200] if code.startswith(("runtime_admission_", "payload_", "state_", "copy_catalog_"))
                else "runtime_admission_compaction_worker_failed")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _authorize(root: Path, log: dict, wall) -> dict:
    """A1: K's binding and the compaction authority record (design v2 rebinding_and_admission)."""
    intent = log["intent"]
    head = stage.read_admission_head(root)
    if head != intent["head"]:
        _refuse("runtime_admission_head_changed")
    source = stage.verify_closed_binding(root, intent["source_binding_id"])
    source_manual = stage._verify_manual_head(root, head, source)
    candidate = stage._binding_snapshot(root, intent["k_slot"])
    validation = stage._target_ownership_validation(root, candidate)
    if validation["delegations"] or (validation["policy"] or {}).get("delegated_targets"):
        _refuse("runtime_admission_compaction_k_delegated")
    stage._require_manual_equivalent(source_manual["capture"], stage._manual_capture(root, candidate))
    if len(stage._canonical_json(candidate)) > BINDING_MARGIN:  # M6 at A1
        _refuse("runtime_admission_compaction_capacity")
    materialized = _last(log, "materialized")["data"]
    if materialized["state_identity"] != candidate["state_identity"]:
        _refuse("runtime_admission_compaction_k_identity")
    evidence_map = materialized["evidence_map"]
    if candidate["schemas"] != source["schemas"]:  # start.a1_extra (SC-1b): K's schemas are H's
        _refuse("runtime_admission_compaction_schema_changed")
    if not stage._compaction_authority_equivalent(root, source, candidate, evidence_map,
                                                  stage.compaction_source_roots(intent, source)):
        _refuse("runtime_admission_compaction_payload_changed")
    if stage._binding_snapshot(root, source["slot"]) != source or stage.read_admission_head(root) != head:
        _refuse("privacy_authority_stale")
    identity = stage._store_binding(root, candidate)
    record = stage.compaction_authority_record(head, head["binding_id"], candidate, log["compaction_id"],
                                               evidence_map)
    stage._atomic_record(stage.compaction_log_path(root, log["compaction_id"]) / "authority.json", record,
                         immutable=True)
    return stage.append_compaction_event(root, log, "authorized", {"candidate_binding_id": identity}, at=_now(wall))


def _activate(root: Path, controller, log: dict, wall) -> dict:
    """A2: activate K (its own code probes it), size re-checked before the head moves (M6)."""
    intent = log["intent"]
    candidate_id = _last(log, "authorized")["data"]["candidate_binding_id"]
    head = stage.read_admission_head(root)
    if head == intent["head"]:
        candidate = stage._binding_snapshot(root, intent["k_slot"])
        if len(stage._canonical_json(candidate)) > BINDING_MARGIN or stage._digest(candidate) != candidate_id:
            _refuse("runtime_admission_compaction_capacity" if len(stage._canonical_json(candidate))
                    > BINDING_MARGIN else "runtime_admission_compaction_payload_changed")
        controller.activate(intent["k_slot"], expected_head=head, managed=True)
        head = stage.read_admission_head(root)
    if head is None or head["status"] != "closed" or head["binding_id"] != candidate_id:
        _refuse("runtime_admission_head_changed")
    return stage.append_compaction_event(root, log, "activated", {"head": head}, at=_now(wall))


def _destroy_partial_snapshots(root: Path, log: dict, wall) -> dict:
    """C2b resume: a snapshot reserved but never completed is destroyed (logged) before a new one."""
    done = {event["data"]["n"] for event in stage.compaction_events(log, "snapshot")}
    destroyed = {event["data"]["n"] for event in stage.compaction_events(log, "snapshot_destroyed")}
    for event in stage.compaction_events(log, "snapshot_reserved"):
        n = event["data"]["n"]
        if n in done or n in destroyed:
            continue
        path = Path(event["data"]["path"])
        if _lexists(path):
            if path.parent != root / "compaction" / log["compaction_id"]:
                _refuse("runtime_admission_compaction_identity")
            _unlink_tree(root, path, k_inodes=_k_inodes(root, log), runtime_links=False)
        log = stage.append_compaction_event(root, log, "snapshot_destroyed", {"n": n}, at=_now(wall))
    return log


def _next_after_activation(log: dict) -> str:
    """The operator's next command once K is active (review B4): arm the one verification run.

    SC-4a: a release-mode log is armed like a plain one; serve then runs it
    as the isolated verification boot (KZA_RUNTIME_RELEASE_VERIFY). Until
    SC-4b ships V's exits (RELEASE_EXITS_AVAILABLE, SC4A-N13) a release-mode
    log cannot be armed and its only exit is the lossless plain abort.
    """
    if release_log(log) and not RELEASE_EXITS_AVAILABLE:
        return "compact abort " + log["compaction_id"]
    return "compact verify " + log["compaction_id"]


def _continue(root: Path, compaction_id: str, *, archive: Path | None, worker=None, wall=time.time) -> dict:
    """Resume C2a-C5 from the last logged event (every step idempotent or refused)."""
    gate = "compaction:" + compaction_id
    with stage.authority_memo(), stage.AdmissionController(root, gate=gate) as controller:
        log = _log(root, compaction_id)
        if _has(log, "abort_started") or log["terminal"]:
            _refuse("runtime_admission_compaction_phase")
        if not _has(log, "k_runtime"):
            if archive is None:
                _refuse("runtime_admission_compaction_archive_required")
            log = _k_runtime(root, log, archive, wall)
        log = _destroy_partial_snapshots(root, log, wall)
    if not _has(log, "materialized"):
        (worker or run_worker)(root, compaction_id)
    with stage.authority_memo(), stage.AdmissionController(root, gate=gate) as controller:
        log = _log(root, compaction_id)
        if not _has(log, "materialized"):
            _refuse("runtime_admission_compaction_worker_failed")
        if not _has(log, "authorized"):
            log = _authorize(root, log, wall)
        if not _has(log, "activated"):
            log = _activate(root, controller, log, wall)
    return {"compaction_id": compaction_id, "phase": "activated", "next": _next_after_activation(log),
            "materialized": _last(log, "materialized")["data"]["counts"]}


# --- verify, allow-run ------------------------------------------------------------------------------

def compact_verify(root, compaction_id: str, *, smoke_seconds: int | None = None, wall=time.time) -> dict:
    """Arm the one verification run (design v3 V1). The operator then starts kza-voice.

    Plain mode: the operator stops kza-voice after the smoke run; N defaults
    to 120 s. Release mode (SC-4a): serve runs the armed run as the isolated
    verification boot, measures ``smoke_seconds`` from the acknowledged READY
    and stops it itself; the unit stays disabled. A release-mode log needs an
    explicit N of at most RELEASE_SMOKE_SECONDS_MAX (SC4A-N14), and is not
    armed at all until SC-4b ships V's exits (RELEASE_EXITS_AVAILABLE, SC4A-N13).
    """
    root = checked_root(root)
    if smoke_seconds is not None and (type(smoke_seconds) is not int or not 10 <= smoke_seconds <= 3600):
        _refuse("runtime_admission_compaction_smoke_seconds")
    with stage.AdmissionController(root, gate="compaction:" + compaction_id):
        log = _log(root, compaction_id)
        if not _has(log, "activated") or _has(log, "abort_started"):
            _refuse("runtime_admission_compaction_phase")
        head = stage.read_admission_head(root)
        armed, started = stage._armed_run(log)
        if armed is not None:
            stage.record_compaction_verified(root, head, wall=_now(wall))
            log = _log(root, compaction_id)
            return {"compaction_id": compaction_id, "armed": True, "started": started is not None,
                    "verified": _has(log, "verified")}
        if release_log(log):
            if not RELEASE_EXITS_AVAILABLE:
                _refuse(RELEASE_EXIT_UNAVAILABLE)  # SC4A-N13: V has no exit before SC-4b
            if smoke_seconds is None or smoke_seconds > RELEASE_SMOKE_SECONDS_MAX:
                _refuse(RELEASE_SMOKE_SECONDS)  # SC4A-N14: N within SC-D2 A's budget, chosen explicitly
        elif smoke_seconds is None:
            smoke_seconds = 120
        candidate = _last(log, "authorized")["data"]["candidate_binding_id"]
        if head is None or head["status"] != "closed" or head["binding_id"] != candidate:
            _refuse("runtime_admission_head_changed")
        launcher_contract_ok(root, stage.read_binding(root, head["binding_id"]))
        stage.append_compaction_event(root, log, "verify_armed", {
            "binding_id": head["binding_id"], "generation": head["generation"], "smoke_seconds": smoke_seconds},
            at=_now(wall))
    return {"compaction_id": compaction_id, "armed": True, "started": False, "smoke_seconds": smoke_seconds}


def compact_allow_run(root, compaction_id: str, *, reason: str, hours: int, wall=time.time) -> dict:
    """Q7: a logged allow-run with a deadline; releases stay gated and nothing is destroyed."""
    root = checked_root(root)
    if type(reason) is not str or not reason.strip():
        _refuse("runtime_admission_compaction_reason")
    if type(hours) is not int or not ALLOW_RUN_HOURS[0] <= hours <= ALLOW_RUN_HOURS[1]:
        _refuse("runtime_admission_compaction_allow_run_hours")
    with stage.AdmissionController(root, gate="compaction:" + compaction_id):
        log = _log(root, compaction_id)
        _require_plain_mode(log)  # V4R2-A1: allow-run finishes only a plain-mode log
        head = stage.read_admission_head(root)
        stage.record_compaction_verified(root, head, wall=_now(wall))
        log = _log(root, compaction_id)
        if _has(log, "abort_started") or _has(log, "retire_started"):
            _refuse("runtime_admission_compaction_phase")
        if not _has(log, "verified"):
            # Review F4: an allow-run needs K' verified; before that, serve (if the armed run was
            # never consumed) or 'compact abort' are the ways forward.
            _refuse("runtime_admission_compaction_not_verified")
        now = _now(wall)
        until = now + hours * 3600.0
        stage.append_compaction_event(root, log, "allow_run", {
            "reason_sha256": _sha256_bytes(reason.encode()), "hours": hours, "until": until}, at=now)
    return {"compaction_id": compaction_id, "allow_run_until": until, "hours": hours}


# --- the deletion walk (design v2 deletion_walk_M2_P5C) ---------------------------------------------

def _k_inodes(root: Path, log: dict) -> set:
    """{(device, inode)} of every regular file in K's slot (no member file may share one)."""
    found = set()
    slot = root / "slots" / log["intent"]["k_slot"]
    if not _lexists(slot):
        return found
    for _directory, _names, filenames, descriptor in os.fwalk(slot, follow_symlinks=False):
        for name in filenames:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                found.add((info.st_dev, info.st_ino))
    return found


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_tree(root: Path, path: Path, *, k_inodes: set, runtime_links: bool, selection_link: bool = False,
                 unlink_hook=None, remove_top: bool = True) -> int:
    """Bottom-up, dir_fd-relative unlink of one member tree; delete only (P5 = C).

    Never follows a link. Only the slot runtime's own config, data and .env
    links (``runtime_links``) and a selection leftover's top-level 'current'
    link (``selection_link``) are unlinked, as links; any other symlink or
    special file refuses.
    Every entry stays on the member root's device (a mount point refuses).
    Every regular file must have st_nlink == 1 and an inode outside K. No
    overwrite, no truncate, no fstrim. Directories are made owner-writable
    (a read-only runtime) only to unlink their entries. Returns files unlinked.
    """
    top = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    removed = 0
    try:
        device = os.fstat(top).st_dev
        if os.path.ismount(path):
            _refuse("runtime_admission_compaction_mount")
        removed = _unlink_children(top, device, k_inodes, runtime_links, depth=0, unlink_hook=unlink_hook,
                                   name=path.name, selection_link=selection_link)
    finally:
        os.close(top)
    if remove_top:  # a member's quarantine is removed by _retire_member after 'walked' (review r4 R4-6)
        parent = path.parent
        os.rmdir(path)
        _fsync_directory(parent)
    return removed


def _unlink_children(descriptor: int, device: int, k_inodes: set, runtime_links: bool, *, depth: int,
                     unlink_hook, name: str, selection_link: bool = False) -> int:
    removed = 0
    mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
    if mode & stat.S_IRWXU != stat.S_IRWXU:
        os.fchmod(descriptor, mode | stat.S_IRWXU)
    for entry in sorted(os.listdir(descriptor)):
        info = os.stat(entry, dir_fd=descriptor, follow_symlinks=False)
        if info.st_dev != device:
            _refuse("runtime_admission_compaction_mount")
        if stat.S_ISLNK(info.st_mode):
            if not ((runtime_links and depth == 1 and name == "runtime" and entry in RUNTIME_LINKS)
                    or (selection_link and depth == 0 and entry == "current")):
                _refuse("runtime_admission_compaction_symlink")
            os.unlink(entry, dir_fd=descriptor)
        elif stat.S_ISDIR(info.st_mode):
            child = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                if os.fstat(child).st_dev != device:
                    _refuse("runtime_admission_compaction_mount")
                removed += _unlink_children(child, device, k_inodes, runtime_links, depth=depth + 1,
                                            unlink_hook=unlink_hook, name=entry)
                os.fsync(child)
            finally:
                os.close(child)
            os.rmdir(entry, dir_fd=descriptor)
        elif stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1 or (info.st_dev, info.st_ino) in k_inodes:
                _refuse("runtime_admission_compaction_hard_link")
            if unlink_hook is not None:
                unlink_hook(removed)
            os.unlink(entry, dir_fd=descriptor)
            removed += 1
        else:
            _refuse("runtime_admission_compaction_special_file")
    os.fsync(descriptor)
    return removed


def _quarantine(root: Path, compaction_id: str, member: dict) -> Path:
    return root / "compaction" / compaction_id / "retiring" / str(member["n"])


def _tombstone_path(root: Path, member: dict) -> Path:
    key = stage._digest({"path": member["path"], "identity": member["identity"]})
    return stage.tombstone_directory(root) / (key + ".json")


def _tombstone(member: dict, compaction_id: str) -> dict:
    return {"schema_version": 1, "compaction_id": compaction_id, "member": member["n"], "class": member["class"],
            "path": member["path"], "identity": member["identity"], "slot": member.get("slot"),
            "state_root": member.get("state_root"), "state_identity": member.get("state_identity"),
            "binding_ids": member.get("binding_ids", []), "copy_ids": sorted(member.get("copy_ids", [])),
            "status": "destroyed"}


def _accepted_identity(log: dict, member: dict) -> dict:
    accepted = [event for event in stage.compaction_events(log, "identity_accepted")
                if event["data"]["member"] == member["n"]]
    return accepted[-1]["data"]["identity"] if accepted else member["identity"]


def _retire_member(root: Path, log: dict, member: dict, k_inodes: set, wall, unlink_hook=None) -> dict:
    """C7 for one member: identity, 'retiring', same-device rename, walk, tombstone, 'destroyed'."""
    compaction_id = log["compaction_id"]
    path, quarantine = Path(member["path"]), _quarantine(root, compaction_id, member)
    identity = _accepted_identity(log, member)
    tombstone = _tombstone_path(root, member)
    if any(event["data"]["member"] == member["n"] for event in stage.compaction_events(log, "missing_accepted")):
        if _lexists(path) or _lexists(quarantine):
            # Review r4 R4-6: accepted as missing, then put back. The acceptance stands (it is in the
            # log); move it away again (or delete it by hand) and repeat the command.
            _refuse("runtime_admission_compaction_accepted_member_back", {
                "member": member["n"], "path": str(path if _lexists(path) else quarantine)})
        return log  # review r2 N8: logged by the operator; no tombstone, no copies-retired
    done = [event for event in stage.compaction_events(log, "destroyed") if event["data"]["member"] == member["n"]]
    if done:  # resumed after 'destroyed': the copies-retired records are idempotent (immutable, same bytes)
        for copy_id in sorted(member.get("copy_ids", [])):
            _copies_retired(root, copy_id, compaction_id, done[-1]["data"]["tombstone_sha256"])
        return log
    if not _lexists(tombstone):
        at_path = _lexists(path) and _identity(path) == identity
        at_quarantine = _lexists(quarantine) and _identity(quarantine) == identity
        if _lexists(path) and not at_path or _lexists(quarantine) and not at_quarantine:
            _refuse("runtime_admission_compaction_identity", {"member": member["n"]})
        if at_path:
            if not _slot_contents_ok(log, member):
                _refuse("runtime_admission_compaction_identity", {"member": member["n"]})  # review F6
            if not _links_ok(member):
                _refuse("runtime_admission_compaction_symlink", {"member": member["n"]})  # never walked
            if not any(event["data"]["member"] == member["n"]
                       for event in stage.compaction_events(log, "retiring")):
                log = stage.append_compaction_event(root, log, "retiring", {
                    "member": member["n"], "identity": identity}, at=_now(wall))
            stage.private_directories(quarantine.parent)
            try:
                os.rename(path, quarantine)
            except OSError as error:
                if error.errno == errno.EXDEV:
                    _refuse("runtime_admission_compaction_cross_device", {"member": member["n"]})
                raise
            _fsync_directory(path.parent)
            _fsync_directory(quarantine.parent)
            at_quarantine = True
        if at_quarantine and not _renamed(log, member["n"]):
            # Review r3 R3-4: logged after the rename, so 'absent everywhere' later proves the walk ran.
            log = stage.append_compaction_event(root, log, "renamed", {
                "member": member["n"], "identity": identity}, at=_now(wall))
        if not at_path and not at_quarantine and not _renamed(log, member["n"]):
            # Review r2 N8 / r3 R3-4: never renamed by this compaction ('retiring' alone is logged before
            # the rename), so it was moved or deleted by hand: a tombstone would claim a destruction that
            # may not have happened. 'retire|abort --member n --accept-missing' leaves it out, logged.
            _refuse("runtime_admission_compaction_member_missing", {"member": member["n"]})
        if at_quarantine:
            # Review r2 N3: a prepare leftover's runtime links and a selection leftover's 'current' link
            # are unlinked as links (never followed); any other link still refuses.
            _unlink_tree(root, quarantine, k_inodes=k_inodes,
                         runtime_links=member["class"] in {"managed_slot", "prepare_leftover"},
                         selection_link=member["class"] == "selection_leftover", unlink_hook=unlink_hook,
                         remove_top=False)
            if not _walked(log, member["n"]):
                # Review r4 R4-6: logged while the emptied quarantine still exists, so 'absent everywhere'
                # proves the walk only after it; before it, the quarantine was moved out by hand.
                log = stage.append_compaction_event(root, log, "walked", {
                    "member": member["n"], "identity": identity}, at=_now(wall))
            os.rmdir(quarantine)
            _fsync_directory(quarantine.parent)
        elif not _walked(log, member["n"]):
            # 'renamed' and gone from the quarantine before the walk finished: moved out by hand. A
            # tombstone would claim files destroyed that may survive elsewhere; --accept-missing.
            _refuse("runtime_admission_compaction_member_missing", {"member": member["n"]})
        # Absent everywhere (a crash after the last rmdir): nothing with the identity may remain.
        if (_lexists(path) and _identity(path) == identity) or _lexists(quarantine):
            _refuse("runtime_admission_compaction_identity", {"member": member["n"]})
        stage._atomic_record(tombstone, _tombstone(member, compaction_id), immutable=True)
    raw = tombstone.read_bytes()
    log = stage.append_compaction_event(root, log, "destroyed", {
        "member": member["n"], "tombstone_sha256": _sha256_bytes(raw)}, at=_now(wall))
    for copy_id in sorted(member.get("copy_ids", [])):
        _copies_retired(root, copy_id, compaction_id, _sha256_bytes(raw))
    return log


def _renamed(log: dict, number: int) -> bool:
    return any(event["data"]["member"] == number for event in stage.compaction_events(log, "renamed"))


def _walked(log: dict, number: int) -> bool:
    return any(event["data"]["member"] == number for event in stage.compaction_events(log, "walked"))


def _copies_retired(root: Path, copy_id: str, compaction_id: str, tombstone_sha256: str) -> None:
    """M5 / P2b: the catalog copy is terminal 'retired by compaction' (separate immutable record)."""
    head = root / "admission" / "copies" / copy_id / "head.json"
    head_sha = _sha256_bytes(head.read_bytes()) if _lexists(head) else None
    stage._atomic_record(root / "admission" / "copies-retired" / (copy_id + ".json"), {
        "schema_version": 1, "copy_id": copy_id, "catalog_head_sha256": head_sha, "compaction_id": compaction_id,
        "tombstone_sha256": tombstone_sha256}, immutable=True)


K_ABORT_NUMBER = 100000  # K's member number in an abort: above every own-work number (base + event seq)


def _retiring_identity(log: dict, number: int) -> dict | None:
    """The identity a member had when its 'retiring' was logged (resolves it in its quarantine)."""
    found = [event["data"].get("identity") for event in stage.compaction_events(log, "retiring")
             if event["data"]["member"] == number]
    return found[-1] if found else None


def own_work(root: Path, log: dict) -> list[dict]:
    """This compaction's own directories (snapshots, placeholder), recorded by identity in its log.

    Each item's number is fixed by the event that recorded it (review r2 N2):
    len(intent members) + that event's seq, so a resumed retire or abort
    never renumbers. A reservation (snapshot_reserved, own_work_reserved)
    without its completion is listed while it exists at its path, or by the
    identity its 'retiring' logged once it moved into the quarantine.
    """
    base = len(log["intent"]["members"])
    found = []
    done_snapshots = {event["data"]["n"] for event in stage.compaction_events(log, "snapshot")}
    destroyed_snapshots = {event["data"]["n"] for event in stage.compaction_events(log, "snapshot_destroyed")}
    recorded = {event["data"]["path"] for event in stage.compaction_events(log, "own_work")}
    for event in log["events"]:
        number, data = base + event["seq"], event["data"]
        if event["kind"] in {"snapshot", "own_work"}:
            found.append({"class": "own_work", "path": data["path"], "identity": data["identity"],
                          "proof": "compaction_log", "n": number})
            continue
        reserved = (event["kind"] == "snapshot_reserved" and data["n"] not in done_snapshots
                    and data["n"] not in destroyed_snapshots) or \
            (event["kind"] == "own_work_reserved" and data["path"] not in recorded)
        if not reserved:
            continue
        path = Path(data["path"])
        identity = _identity(path) if _lexists(path) else _retiring_identity(log, number)
        if identity is not None:
            found.append({"class": "own_work", "path": str(path), "identity": identity, "proof": "compaction_log",
                          "n": number})
    copies = catalog_records(root)
    for item in found:
        for record in copies:
            if record["destination"] == item["path"] or record["destination"].startswith(item["path"] + "/"):
                item.setdefault("copy_ids", []).append(record["copy_id"])
    return found


def _numbered(log: dict) -> list[dict]:
    members = [dict(member) for member in log["intent"]["members"]]
    members += own_work(Path(log["intent"]["k_state_root"]).parents[2], log)  # stable numbers (r2 N2)
    # Own work and leftovers first, then non-owner slots, owners, and H last.
    head_state = stage.read_binding(Path(log["intent"]["k_state_root"]).parents[2],
                                    log["intent"]["source_binding_id"])["state_root"]
    owners = {owner["state_root"] for owner in log["intent"]["owners"]}

    def order(member):
        state = member.get("state_root")
        rank = 0 if member["class"] != "managed_slot" else 3 if state == head_state else 2 if state in owners else 1
        return (rank, member["n"])
    return sorted(members, key=order)


# --- retire ------------------------------------------------------------------------------------------

def compact_retire(root, compaction_id: str, *, member: int | None = None, accept_identity_change: bool = False,
                   accept_owner_only_loss: str | None = None, accept_missing: bool = False, scan=None,
                   wall=time.time, unlink_hook=None) -> dict:
    """C7/C8: destroy every member after the verified run; tombstones; 'complete'.

    A release-mode log is retired only through the SC-4b gates (_release_retire_gates):
    K' binding (sdk_versions-only rule), closed witness, W0 and W1 exactly 'equal',
    K's whole authority unchanged and no resurrected id.
    """
    root = checked_root(root)
    with stage.authority_memo(), stage.AdmissionController(root, gate="compaction:" + compaction_id):
        log = _log(root, compaction_id)
        if _has(log, "abort_started"):
            _refuse("runtime_admission_compaction_phase")
        head = stage.read_admission_head(root)
        stage.record_compaction_verified(root, head, wall=_now(wall))
        log = _log(root, compaction_id)
        if not _has(log, "verified"):
            _refuse("runtime_admission_compaction_not_verified")  # P3: destroy only after K' closed
        w_record = None
        if release_log(log) and not _has(log, "retire_started"):
            w_record = _release_retire_gates(root, log, head)  # before any other write (R4-OPS-10b)
        if accept_identity_change:
            log = _accept_identity(root, log, member, wall)
        if accept_missing:
            log = _accept_missing(root, log, member, wall)
        if not _has(log, "retire_started"):
            named = tuple(log["intent"].get("named_units", ()))
            report = _pre_retire(root, log, head, scan or _default_scan(named=named), now=_now(wall),
                                 accept_owner_only_loss=accept_owner_only_loss)
            if w_record is not None:
                report["w"] = w_record
            log = stage.append_compaction_event(root, log, "retire_started", report, at=_now(wall))
        return _finish_retire(root, log, wall, unlink_hook=unlink_hook)


def _release_retire_gates(root: Path, log: dict, head: dict) -> dict:
    """SC-4b retire gates of a release-mode log (mechanism.retire.gates); write nothing.

    (1) K' binding: a difference only in sdk_versions refuses transiently
    (sdk_changed, R5-2); any other difference refuses k_binding_changed (route:
    abort --to-h, row RX); then verify_closed_binding (closure witness). (3)
    W0, and W1 'equal' with no allowance; stricter (SC4B-N4): K's whole
    authority digest equals the authorized one, so protected objects only in
    K' (or another schema contract) also refuse. (4) the resurrection scan of
    K' finds nothing (absolute: K's baseline is follow-up, SC4B-N5).
    """
    intent = log["intent"]
    binding = stage.read_binding(root, head["binding_id"])
    if binding["slot"] != intent["k_slot"]:
        _refuse("runtime_admission_head_changed")
    snapshot = stage._binding_snapshot(root, binding["slot"])
    if snapshot != binding:
        changed = sorted(key for key in set(binding) | set(snapshot) if binding.get(key) != snapshot.get(key))
        if changed == ["sdk_versions"]:
            _refuse(SDK_CHANGED, {"bound": binding["sdk_versions"], "installed": snapshot["sdk_versions"]})
        _refuse(K_BINDING_CHANGED, {"code": "privacy_authority_stale", "changed": changed})
    _k_closure_proof(root, head)  # an unreadable proof is the coded HD-like stop (review RB-3)
    stage.verify_closed_binding(root, head["binding_id"])
    found = _w(root, log)
    authorized = stage.read_binding(root, _last(log, "authorized")["data"]["candidate_binding_id"])
    authority_equal = binding["authority"]["digest"] == authorized["authority"]["digest"]
    if found["report"]["class"] != W_EQUAL or not authority_equal:
        _refuse(RETIRE_PROTECTED_CHANGED, {"w": found["report"], "w_sha256": found["sha256"],
                                           "authority_equal": authority_equal})
    resurrected = launcher.resurrection_counts(Path(binding["state_root"]))
    if any(resurrected.values()):
        _refuse(RETIRE_RESURRECTED, {"resurrected": resurrected})
    return {"class": W_EQUAL, "sha256": found["sha256"]}


def _pre_retire(root: Path, log: dict, head: dict, scan, *, now: float,
                accept_owner_only_loss: str | None = None) -> dict:
    """Pre-retire checks (design v1 pre_retire_checks, v4 F2): K' verified and self-contained."""
    intent = log["intent"]
    binding = stage.verify_closed_binding(root, head["binding_id"])
    if binding["slot"] != intent["k_slot"]:
        _refuse("runtime_admission_head_changed")
    validation = stage._target_ownership_validation(root, binding)
    state = Path(binding["state_root"])
    if validation["delegations"]:
        _refuse("runtime_admission_compaction_k_delegated")
    for target in validation["targets"].values():
        for value in stage._descriptor_paths(target["descriptor"]):
            if not Path(value).is_relative_to(state):
                _refuse("runtime_admission_compaction_k_foreign_target")
    members = _numbered(log)
    for item in members:
        path = Path(item["path"])
        if _lexists(path) and _identity(path) != _accepted_identity(log, item):
            _refuse("runtime_admission_compaction_identity", {"member": item["n"]})
    k_inodes = _k_inodes(root, log)
    for item in members:
        path = Path(item["path"])
        if not _lexists(path):
            continue
        for _directory, _names, filenames, descriptor in os.fwalk(path, follow_symlinks=False):
            for name in filenames:
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode) and (info.st_nlink != 1 or (info.st_dev, info.st_ino) in k_inodes):
                    _refuse("runtime_admission_compaction_hard_link", {"member": item["n"]})
    for item in members:
        if not _links_ok(item):
            _refuse("runtime_admission_compaction_symlink", {"member": item["n"]})  # never walked (r2 N3)
        if not _slot_contents_ok(log, item):
            _refuse("runtime_admission_compaction_identity", {"member": item["n"]})  # review F6
    _same_plan(root, log, members)  # review F7
    scanned = scan(root, [item for item in members if _lexists(Path(item["path"]))])  # P14 again
    h_member = next(item for item in intent["members"] if item.get("state_root") == _h_state(root, intent))
    h_state = Path(h_member["state_root"])
    if not _lexists(h_state) or _identity(h_state) != h_member["state_identity"]:
        _refuse("runtime_admission_compaction_identity", {"member": h_member["n"]})
    covered = coverage([item for item in members if _lexists(Path(item["path"]))], h_state, now=now,
                       k_state=state)
    new = {number: counts["unexplained"] - intent["coverage_members"].get(number, {}).get("unexplained", 0)
           for number, counts in covered["members"].items()}
    new = {number: count for number, count in new.items() if count > 0}
    digest = stage._digest({"compaction_id": log["compaction_id"], "unexplained": {
        number: counts["unexplained"] for number, counts in sorted(covered["members"].items())}})
    if new and accept_owner_only_loss != digest:
        # Review F5: the start flag covers the plan's counts only; data first seen at retire needs
        # its own acknowledgement of this coverage (retire --accept-owner-only-loss <sha256>).
        _refuse("runtime_admission_compaction_owner_only_data", {
            "coverage": covered["totals"], "new_unexplained": new, "coverage_sha256": digest})
    return {"members": len(members), "coverage": covered["totals"], "units": scanned.get("units_scanned"),
            "coverage_sha256": digest, "accepted_new_unexplained": sum(new.values())}


def _h_state(root: Path, intent: dict) -> str:
    return stage.read_binding(root, intent["source_binding_id"])["state_root"]


def _runtime_extra(runtime: Path) -> list[str]:
    """Regular files in a slot runtime that its release.json does not list (review r2, F6 residual).

    prepare_slot writes exactly the archive's files, release.json and the
    config/data/.env links; anything else was put there by hand. Names only
    (relative paths), at most 32. An unreadable release.json counts as extra.
    """
    try:
        manifest = json.loads((runtime / "release.json").read_bytes())
        listed = set(manifest["files"])
    except (OSError, ValueError, KeyError, TypeError):
        return ["release.json"]
    extra = []
    for directory, _names, filenames in os.walk(runtime, followlinks=False):
        for name in filenames:
            relative = (Path(directory) / name).relative_to(runtime).as_posix()
            if relative == "release.json" or (relative in RUNTIME_LINKS and os.path.islink(runtime / relative)):
                continue
            if relative not in listed:
                extra.append(relative)
    return sorted(extra)[:32]


def _unexpected_entries(member: dict) -> list[str]:
    """Names in a member that its proof does not cover (review F6, r2 N7): never opened, names only."""
    path = Path(member["path"])
    if not _lexists(path) or _kind(path) != "directory":
        return []
    names = set(os.listdir(path))
    if member["class"] in {"managed_slot", "prepare_leftover"}:
        extra = sorted(names - SLOT_ENTRIES)
        if not extra and member["class"] == "managed_slot" and _kind_or_none(path / "runtime") == "directory":
            extra = ["runtime/" + name for name in _runtime_extra(path / "runtime")]
        return extra
    if member["class"] == "catalog_copy":
        return sorted(names - set(member.get("allowed_entries", [])))
    return []


def _links_ok(member: dict) -> bool:
    """The links a member may hold, checked before anything is logged (review r2 N3): a slot's or a
    prepare leftover's runtime config/data/.env, a selection leftover's single 'current'; nothing else."""
    path = Path(member["path"])
    if not _lexists(path) or _kind(path) != "directory":
        return True
    if member["class"] == "selection_leftover":
        return _selection_leftover(path)
    if member["class"] in {"managed_slot", "prepare_leftover"}:
        if any(os.path.islink(path / name) for name in os.listdir(path)):
            return False
        return not ((_lexists(path / "state") and _has_symlink(path / "state"))
                    or (_lexists(path / "runtime") and not _runtime_links_ok(path / "runtime")))
    return not _has_symlink(path)


def _slot_contents_ok(log: dict, member: dict) -> bool:
    """A member still holds only what its proof covers, and a slot's state/ is the recorded one (F6, r2 N7).

    A member whose identity change was accepted (logged, narrow) is held to
    the entry rule only. Members already moved away are checked by identity.
    """
    path = Path(member["path"])
    if not _lexists(path) or _kind(path) != "directory":
        return True
    if _unexpected_entries(member):
        return False
    if member["class"] != "managed_slot":
        return True
    state = path / "state"
    accepted = any(event["data"]["member"] == member["n"] for event in stage.compaction_events(log, "identity_accepted"))
    return accepted or not _lexists(state) or _identity(state) == member["state_identity"]


def _member_shape(member: dict, identity: dict) -> tuple:
    return (member["class"], member["path"], identity["device"], identity["inode"],
            (member.get("state_identity") or {}).get("inode"), tuple(sorted(member.get("copy_ids", []))))


def _same_plan(root: Path, log: dict, members: list[dict]) -> None:
    """The retirement set and the untouched list, recomputed at retire, are the acknowledged ones (review F7).

    Members whose identity change was accepted are compared by the accepted
    identity, and their own paths are left out of both untouched lists.
    """
    intent = log["intent"]
    current, untouched = retirement_set(root, k_slot=intent["k_slot"],
                                        retire_prepare_leftovers=intent["retire_prepare_leftovers"],
                                        compaction_id=log["compaction_id"])
    accepted = {event["data"]["member"] for event in stage.compaction_events(log, "identity_accepted")}
    missing_ok = {event["data"]["member"] for event in stage.compaction_events(log, "missing_accepted")}
    for item in intent["members"]:
        if item["n"] not in missing_ok and not _lexists(Path(item["path"])):
            # Review r2 N8: moved whole (into backups/, logs/...) or deleted after the plan. The untouched
            # list compares top-level names only, so only this finds it.
            _refuse("runtime_admission_compaction_member_missing", {"member": item["n"]})
    accepted |= missing_ok
    skipped = {Path(item["path"]).relative_to(root).as_posix() for item in intent["members"] if item["n"] in accepted}
    planned = sorted(_member_shape(item, _accepted_identity(log, item)) for item in intent["members"]
                     if item["n"] not in accepted)
    found = sorted(_member_shape(item, item["identity"]) for item in current
                   if Path(item["path"]).relative_to(root).as_posix() not in skipped)
    kept = [entry for entry in untouched if entry["path"] not in skipped]
    acknowledged = [entry for entry in intent["untouched"] if entry["path"] not in skipped]
    if planned != found or kept != acknowledged:
        _refuse("runtime_admission_compaction_plan_changed", {
            "untouched_added": sorted({entry["path"] for entry in kept} - {entry["path"] for entry in acknowledged}),
            "untouched_gone": sorted({entry["path"] for entry in acknowledged} - {entry["path"] for entry in kept}),
            "members_planned": len(planned), "members_found": len(found)})


def _finish_retire(root: Path, log: dict, wall, unlink_hook=None) -> dict:
    k_inodes = _k_inodes(root, log)
    members = _numbered(log)
    for member in members:
        log = _retire_member(root, log, member, k_inodes, wall, unlink_hook=unlink_hook)
    for member in members:
        if _lexists(Path(member["path"])) and _identity(Path(member["path"])) == _accepted_identity(log, member):
            _refuse("runtime_admission_compaction_identity", {"member": member["n"]})
    _drop_scratch(root, log)
    totals = {"members": len(members), "destroyed": len(stage.compaction_events(log, "destroyed"))}
    log = stage.append_compaction_event(root, log, "complete", totals, at=_now(wall))
    return {"compaction_id": log["compaction_id"], "phase": "complete", **totals}


def _accept_missing(root: Path, log: dict, number: int | None, wall, *, members: list | None = None) -> dict:
    """``--member n --accept-missing`` (review r2 N8, r3 R3-1/R3-4): a member gone from its path and never
    renamed by this compaction is left out, logged. No tombstone and no copies-retired: nothing claims it
    destroyed. On retire the members are the plan's; on abort, its own work and K. A member whose
    'retiring' is logged but not its 'renamed' was moved by hand after the cut and may be accepted, and
    so may one 'renamed' but not 'walked' (review r4 R4-6: its quarantine was moved out mid-walk)."""
    members = {item["n"]: item for item in (_numbered(log) if members is None else members)}
    if number not in members:
        _refuse("runtime_admission_compaction_member")
    member = members[number]
    if _lexists(Path(member["path"])) or _lexists(_quarantine(root, log["compaction_id"], member)) \
            or _walked(log, number) or any(event["data"]["member"] == number
                                           for event in stage.compaction_events(log, "destroyed")):
        _refuse("runtime_admission_compaction_member")
    return stage.append_compaction_event(root, log, "missing_accepted", {
        "member": number, "path": member["path"], "identity": member["identity"]}, at=_now(wall))


def _accept_identity(root: Path, log: dict, number: int | None, wall) -> dict:
    """``--member n --accept-identity-change`` (design v2 retire_identity_mismatch), logged and narrow."""
    members = {item["n"]: item for item in _numbered(log)}
    if number not in members:
        _refuse("runtime_admission_compaction_member")
    member = members[number]
    for path in (Path(member["path"]), _quarantine(root, log["compaction_id"], member)):
        if not _lexists(path):
            continue
        k_root = root / "slots" / log["intent"]["k_slot"]
        info = os.lstat(path)
        if (_kind(path) != "directory" or info.st_dev != os.lstat(root).st_dev or path.is_relative_to(k_root)
                or any((value.st_dev, value.st_ino) in _k_inodes(root, log) for value in _file_stats(path))):
            _refuse("runtime_admission_compaction_identity", {"member": number})
        return stage.append_compaction_event(root, log, "identity_accepted", {
            "member": number, "previous": member["identity"], "identity": _identity(path)}, at=_now(wall))
    _refuse("runtime_admission_compaction_member")


def _file_stats(path: Path):
    for _directory, _names, filenames, descriptor in os.fwalk(path, follow_symlinks=False):
        for name in filenames:
            yield os.stat(name, dir_fd=descriptor, follow_symlinks=False)


# --- abort --------------------------------------------------------------------------------------------

def _h_intact(root: Path, intent: dict) -> dict:
    """Predicate (5) without a closed-head requirement: H's binding, marker and closure still hold."""
    binding = stage.read_binding(root, intent["source_binding_id"])
    if stage._binding_snapshot(root, binding["slot"]) != binding:
        _refuse("runtime_admission_compaction_h_changed")
    marker = stage._read_json(Path(binding["state_root"]) / stage._MARKER)
    if marker.get("binding_id") != intent["source_binding_id"]:
        _refuse("runtime_admission_compaction_h_changed")
    stage._closure_proof(root, intent["source_binding_id"])
    return binding


def _privacy_rows(state: Path, relative: str, table: str) -> set:
    path = state / relative
    if not _lexists(path):
        return set()
    with stage._database_image(path) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            return set()
        return {stage._digest([stage._typed(value) for value in row])
                for row in db.execute("SELECT * FROM " + stage._quote(table))}


def abort_to_h_predicate(root: Path, log: dict) -> dict:
    """Design v2 abort_to_h with v3 Q6 and v4 F7: content-free; raises the coded refusal."""
    intent = log["intent"]
    k_state = Path(intent["k_state_root"])
    h_binding = _h_intact(root, intent)  # (5)
    h_state = Path(h_binding["state_root"])
    try:
        stage._assert_writers_closed(k_state)  # (1)
    except ValueError:
        _refuse("runtime_admission_writer_active")
    privacy = intent["privacy_relative"]
    for table in ("conversation_erasures", "conversation_tombstones"):  # (2)
        if _privacy_rows(k_state, privacy, table) != _privacy_rows(h_state, privacy, table):
            _refuse("runtime_admission_compaction_abort_revives")
    h_markers, k_markers = _head_markers(h_state), _head_markers(k_state)  # (4)
    if k_markers["deleted_facts"] - h_markers["deleted_facts"] or \
            k_markers["deleted_profiles"] - h_markers["deleted_profiles"]:
        _refuse("runtime_admission_compaction_abort_revives")
    for relative in _databases(k_state):
        before = _ledger_count(h_state / relative)
        if _ledger_count(k_state / relative) != before:
            _refuse("runtime_admission_compaction_abort_revives")
    authorized = stage.read_binding(root, _last(log, "authorized")["data"]["candidate_binding_id"])  # (3)
    current = stage.capture_state_authority(k_state)
    if _protected_view(current) != _protected_view(authorized["authority"]):
        _refuse("runtime_admission_compaction_abort_protected_changed")
    return {"new_rows": _new_rows(h_state, k_state)}


def _ledger_count(path: Path) -> int | None:
    if not _lexists(path):
        return None
    with stage._database_image(path) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='deletion_ledger_v1'").fetchone():
            return None
        return db.execute("SELECT COUNT(*) FROM deletion_ledger_v1").fetchone()[0]


def _protected_view(authority: dict) -> dict:
    tables = {relative: {name: value for name, value in projection.items() if name not in ABORT_EXCLUDED_PROTECTED}
              for relative, projection in authority["projection"]["tables"].items()}
    return {"tables": tables, "accounts": authority["projection"]["accounts"],
            "configuration": authority["projection"]["configuration"]}


def _new_rows(h_state: Path, k_state: Path) -> dict:
    """K_vs_Kprime: rows K' holds that H does not, per table (counts only)."""
    counts = {}
    h_dbs = _databases(h_state)
    for relative, path in _databases(k_state).items():
        if relative.endswith("chroma.sqlite3"):
            continue
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for name in sorted(names - _INTERNAL_TABLES - _SCHEDULING - ABORT_EXCLUDED_PROTECTED):
                _columns, rows = _table_rows(db, name)
                head = {}
                if relative in h_dbs:
                    with stage._database_image(h_dbs[relative]) as source:
                        if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                          (name,)).fetchone():
                            _columns, head = _table_rows(source, name)
                added = sum(1 for key in rows if key not in head)
                if added:
                    counts[relative + ":" + name] = added
    return counts


# --- SC-4b: W, the release-mode abort predicate (sc-release-design v4 mechanism.w_abort_predicate) ---

def _sqlite_primary(error: sqlite3.Error) -> int | None:
    code = getattr(error, "sqlite_errorcode", None)
    return code & 0xFF if type(code) is int else None


def _sqlite_name(error: sqlite3.Error) -> str:
    code = getattr(error, "sqlite_errorname", None)
    primary = _sqlite_primary(error)
    names = {getattr(sqlite3, name): name for name in dir(sqlite3)
             if name.startswith("SQLITE_") and type(getattr(sqlite3, name)) is int and name.count("_") == 1}
    return names.get(primary, code if type(code) is str else "unknown")


def _error_code(error: BaseException) -> str:
    """A content-free code for an exception: errno or SQLite primary-code name, a coded ValueError's
    code, otherwise 'record_invalid' (never a path or a value)."""
    if isinstance(error, OSError):
        return "errno:" + errno.errorcode.get(error.errno, "unknown")
    if isinstance(error, sqlite3.Error):
        return "sqlite:" + _sqlite_name(error)
    text = str(error)
    return text[:200] if isinstance(error, ValueError) and text.startswith(("runtime_admission_", "privacy_")) \
        else "record_invalid"


def _w0_refusal(error: BaseException) -> CompactionRefusal:
    """W0's mapping of an exception while reading H: a transient refusal (row AT) or the drift HD.

    Transient (R3-OPS-5, R4-PRIV-1, R4-OPS-3): writer_active, busy and
    sdk_version_unavailable; an errno in W0_TRANSIENT_ERRNOS; a SQLite error
    whose primary code is in W0_TRANSIENT_SQLITE (by code, not by class).
    Everything else is h_changed carrying the underlying code (terminal stop
    HD). Both write nothing. The report is content-free (codes and names only).
    """
    if isinstance(error, ValueError) and str(error) in W0_TRANSIENT_CODES:
        return CompactionRefusal(str(error), {"w0": "transient"})
    if isinstance(error, OSError):
        name = errno.errorcode.get(error.errno, "unknown")
        if error.errno in W0_TRANSIENT_ERRNOS:
            return CompactionRefusal(H_TRANSIENT, {"w0": "transient", "errno": name})
    elif isinstance(error, sqlite3.Error) and _sqlite_primary(error) in W0_TRANSIENT_SQLITE:
        return CompactionRefusal(H_TRANSIENT, {"w0": "transient", "sqlite": _sqlite_name(error)})
    return CompactionRefusal(H_CHANGED, {"w0": "drift", "code": _error_code(error)})


def _w0_h(root: Path, intent: dict) -> dict:
    """H intact (_h_intact) with the sdk_versions-only rule; raises whatever reading H raised."""
    binding = stage.read_binding(root, intent["source_binding_id"])
    snapshot = stage._binding_snapshot(root, binding["slot"])
    if snapshot != binding:
        changed = sorted(key for key in set(binding) | set(snapshot) if binding.get(key) != snapshot.get(key))
        if changed == ["sdk_versions"]:  # the shared venv, not H's files (R4-PRIV-1)
            _refuse(SDK_CHANGED, {"w0": "transient", "bound": binding["sdk_versions"],
                                  "installed": snapshot["sdk_versions"]})
        raise ValueError("privacy_authority_stale")
    marker = stage._read_json(Path(binding["state_root"]) / stage._MARKER)
    if marker.get("binding_id") != intent["source_binding_id"]:
        raise ValueError("runtime_admission_marker_mismatch")
    stage._closure_proof(root, intent["source_binding_id"])
    return binding


def w0(root: Path, log: dict) -> dict:
    """W0: no live V child, K' writers closed, H intact; returns {'h_binding', 'k_writers'} or raises
    a coded refusal.

    Every refusal writes nothing, so repeating the command is harmless; a
    transient one clears with its cause, h_changed is the terminal stop HD.
    A K' lock entry that cannot be opened (a symlink, no permission: review
    RB-1) is not a refusal: ``k_writers`` says 'unverifiable' and W1 makes
    the class restrictive_or_unclear, so SS and its SC-D8 A exit apply.
    """
    intent = log["intent"]
    head = stage.read_admission_head(root)
    if head is not None and head["status"] != "closed":
        run = launcher.read_run_record(root, head["run_id"])
        if run is not None and launcher.child_alive(run["child_pid"], run.get("child_start_time")):
            _refuse("runtime_admission_recovery_child_alive", {"w0": "transient"})
    k_writers = {"status": "closed"}
    try:
        stage._assert_writers_closed(Path(intent["k_state_root"]))
    except ValueError:
        _refuse("runtime_admission_writer_active", {"w0": "transient", "state": "k_prime"})
    except OSError as error:
        name = errno.errorcode.get(error.errno, "unknown")
        if error.errno in K_TRANSIENT_ERRNOS:
            _refuse(K_TRANSIENT, {"w0": "transient", "state": "k_prime", "errno": name})
        k_writers = {"status": "unverifiable", "error": "errno:" + name}
    try:
        return {"h_binding": _w0_h(root, intent), "k_writers": k_writers}
    except CompactionRefusal:
        raise
    except (ValueError, OSError, sqlite3.Error, KeyError, TypeError) as error:
        raise _w0_refusal(error) from None


def _w1_recover(path: Path) -> None:
    """Open one K' database read-write without ever creating it (R3-OPS-7): hot-journal rollback and
    WAL checkpoint in place, as R-b's _quick_check does, without its plain connect."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=1.0)) as db:
        db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()


def _w1_regular(path: Path) -> bool:
    """An entry W1 may open: a regular file at its own path, no link on the way (review RA4B-4).

    Checked before the read-write open of _w1_recover, which would follow a
    link in a parent directory to a database outside K' (H's, for one)."""
    return path.resolve(strict=True) == path and _kind(path) == "file"


def _w1_database(k_state: Path, relative: str, expected: dict, h_path: Path) -> dict:
    """W1 for one database K's authorized projection names: its status and per-key differences.

    Raises only the transient K_TRANSIENT (a live holder's lock, a resource
    errno); every other failure is a status ('missing', 'unreadable' with a
    content-free code) or a value the selection cannot classify."""
    path = k_state / relative
    found = {"status": "ok", "tables": {}, "ledger": None, "unclassified": []}
    if not os.path.lexists(path):
        found["status"] = "missing"
        current, names = {}, set()
    else:
        try:
            if not _w1_regular(path):
                raise ValueError("runtime_admission_redirected_database")
            _w1_recover(path)
            with stage._database_image(path) as db:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                current = stage._authority_table_selection(db, names)
                if stage._LEDGER_TABLE in names:
                    try:
                        current[stage._LEDGER_TABLE] = stage._deletion_ledger_projection(db)
                    except _W1_VALUE_ERRORS:
                        current[stage._LEDGER_TABLE] = None  # a broken chain: 'other'
                mixed = dict(current)
                try:
                    stage._mixed_projections(db, names, mixed)
                except _W1_VALUE_ERRORS:  # a value the selection cannot classify: a changed row (RA4B-1)
                    found["unclassified"] = sorted(key for key in expected if key not in current
                                                   and key not in {stage._LEDGER_TABLE, "schema_contract_v1"})
                else:
                    current = mixed
        except sqlite3.Error as error:
            if _sqlite_primary(error) in K_TRANSIENT_SQLITE:
                _refuse(K_TRANSIENT, {"w1": "transient", "state": "k_prime", "database": relative,
                                      "sqlite": _sqlite_name(error)})
            found.update(status="unreadable", error=_error_code(error))
            current, names = {}, set()
        except OSError as error:
            if error.errno in K_TRANSIENT_ERRNOS:
                _refuse(K_TRANSIENT, {"w1": "transient", "state": "k_prime", "database": relative,
                                      "errno": errno.errorcode.get(error.errno, "unknown")})
            found.update(status="unreadable", error=_error_code(error))
            current, names = {}, set()
        except ValueError as error:
            found.update(status="unreadable", error=str(error)[:120] if str(error).startswith("runtime_admission_")
                         else "value_error")
            current, names = {}, set()
        except Exception as error:  # noqa: BLE001 - W1 never raises an uncoded error (RA4B-1)
            found.update(status="unreadable", error="python:" + type(error).__name__)
            current, names = {}, set()
    for key, value in sorted(expected.items()):
        if key == "schema_contract_v1" or key in found["unclassified"]:
            continue  # the contract is not part of W1 (no _schema_contract)
        if key == stage._LEDGER_TABLE:
            found["ledger"] = _w1_ledger(current.get(key), value, h_path)
            continue
        before = _multiset(value["row_hashes"])
        after = _multiset((current.get(key) or {"row_hashes": []})["row_hashes"])
        found["tables"][key] = {"added": _count(after, before), "removed": _count(before, after)}
    return found


def _multiset(values) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count(left: dict, right: dict) -> int:
    return sum(max(0, number - right.get(value, 0)) for value, number in left.items())


def _w1_ledger(current: dict | None, bound: dict, h_path: Path) -> str:
    """'equal', 'strict_prefix' (K' holds a strict prefix of H's chain) or 'other' (a missing or broken
    ledger, an entry H lacks, a fork)."""
    if current is None:
        return "other"
    if current == bound:
        return "equal"
    if current["count"] >= bound["count"]:
        return "other"
    try:
        with stage._database_image(h_path) as db:
            prefix = stage._deletion_ledger_projection(db, upto=current["max_seq"])
    except (ValueError, OSError, sqlite3.Error):
        return "other"
    return "strict_prefix" if prefix == current else "other"


def w1(k_state: Path, h_state: Path, projection: dict, *, verify_branch: str,
       k_writers: dict | None = None) -> dict:
    """W1: K' against K's authorized projection; returns the content-free report and its digest.

    Only the databases, tables and selections the projection names, plus its
    accounts and configuration entries, are compared; objects only in K' are
    ignored. A database missing or unreadable in K' counts as every row of
    it absent; a value the selection cannot classify counts as changed.
    Content-free (counts, relative paths, status words); no clock input, so
    the same K' and H give the same digest. Never raises an uncoded error;
    its only refusal is the transient K_TRANSIENT. ``k_writers`` is W0's
    view of K' locks: 'unverifiable' is restrictive (review RB-1).
    """
    k_writers = k_writers or {"status": "closed"}
    report = {"schema_version": W_REPORT_SCHEMA, "verify_branch_reached": verify_branch, "k_writers": k_writers,
              "databases": {}, "tables": {}, "ledgers": {}, "unclassified": {}, "accounts": {},
              "configuration": {}}
    restrictive, loss = k_writers["status"] != "closed", False
    for relative, expected in sorted(projection["tables"].items()):
        found = _w1_database(k_state, relative, expected, h_state / relative)
        report["databases"][relative] = {key: found[key] for key in ("status", "error") if key in found}
        report["tables"][relative] = found["tables"]
        if found["ledger"] is not None:
            report["ledgers"][relative] = found["ledger"]
        if found["unclassified"]:
            report["unclassified"][relative] = found["unclassified"]
            restrictive = True
        if found["status"] != "ok":
            restrictive = True  # R3-PRIV-3 (stricter, SC4B-N3): a missing or unreadable database never 'loss_only'
        if found["ledger"] == "strict_prefix":
            loss = True
        elif found["ledger"] == "other":
            restrictive = True
        for key, change in found["tables"].items():
            if key in W_LOSS:
                restrictive |= change["added"] > 0
                loss |= change["removed"] > 0
            elif key in W_GRANT:
                restrictive |= change["removed"] > 0
                loss |= change["added"] > 0
            elif change["added"] or change["removed"]:
                restrictive = True
    for relative, expected in sorted(projection["accounts"].items()):
        path = k_state / relative
        try:
            if not os.path.lexists(path):
                current, status = [], "missing"
            elif not _w1_regular(path):
                current, status = [], "unclassifiable"  # a link on the way is never read (RA4B-4)
            else:
                current, status = stage._account_projection(path), "ok"
        except Exception:  # noqa: BLE001 - unreadable, or a record it cannot classify (RA4B-1)
            current, status = [], "unclassifiable"
        before, after = _multiset(expected), _multiset(current)
        change = {"status": status, "added": _count(after, before), "removed": _count(before, after)}
        report["accounts"][relative] = change
        restrictive |= status != "ok" or change["added"] > 0 or change["removed"] > 0
    for relative, expected in sorted(projection["configuration"].items()):
        path = k_state / relative
        try:
            if not os.path.lexists(path):
                status = "missing"
            elif not _w1_regular(path):
                status = "not_regular"  # a link on the way, or not a file (RA4B-4)
            else:
                status = "equal" if hashlib.sha256(path.read_bytes()).hexdigest() == expected else "changed"
        except OSError:
            status = "unreadable"
        report["configuration"][relative] = status
        restrictive |= status != "equal"
    report["class"] = W_RESTRICTIVE if restrictive else W_LOSS_ONLY if loss else W_EQUAL
    return {"report": report, "sha256": stage._digest(report)}


def _w(root: Path, log: dict) -> dict:
    """W0 then W1 for a release-mode log under the lease: {'report', 'sha256'}."""
    checked = w0(root, log)
    authorized = stage.read_binding(root, _last(log, "authorized")["data"]["candidate_binding_id"])
    return w1(Path(log["intent"]["k_state_root"]), Path(checked["h_binding"]["state_root"]),
              authorized["authority"]["projection"], verify_branch=verify_branch_reached(root, log) or "not run",
              k_writers=checked["k_writers"])


def _w_counts(report: dict) -> dict:
    """The non-zero counts of a W report, for the 'abort_started' / 'abort_rechecked' record."""
    tables = {relative: {key: change for key, change in changes.items() if change["added"] or change["removed"]}
              for relative, changes in report["tables"].items()}
    return {"k_writers": {} if report["k_writers"]["status"] == "closed" else report["k_writers"],
            "tables": {relative: changes for relative, changes in tables.items() if changes},
            "databases": {relative: found["status"] for relative, found in report["databases"].items()
                          if found["status"] != "ok"},
            "ledgers": {relative: status for relative, status in report["ledgers"].items() if status != "equal"},
            "unclassified": report["unclassified"],
            "accounts": {relative: change for relative, change in report["accounts"].items()
                         if change["status"] != "ok" or change["added"] or change["removed"]},
            "configuration": {relative: status for relative, status in report["configuration"].items()
                              if status != "equal"}}


def _release_head_is_v(root: Path, log: dict, head: dict) -> bool:
    """A closed head is the armed run's seal: its closure names the armed generation (V1, V3, V4).

    Unlike compaction_verified, this does not need the run to be verified: a
    sealed but unverified K' (row V3) is aborted through W like a verified one.
    A proof that cannot be read refuses K_CLOSURE_UNREADABLE (review RB-3),
    told apart from a readable proof of another run (False: head_moved).
    """
    _armed, started = stage._armed_run(log)
    running = _k_closure_proof(root, head)["running_head"]
    return (started is not None and running["generation"] == started["data"]["generation"]
            and running["binding_id"] == started["data"]["binding_id"])


def _k_closure_proof(root: Path, head: dict) -> dict:
    """K' head's closure proof, or the coded HD-like stop K_CLOSURE_UNREADABLE (review RB-3): after V
    the proof is unreadable only through storage damage or a manual edit under admission/."""
    try:
        return stage._closure_proof(root, head["binding_id"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        _refuse(K_CLOSURE_UNREADABLE, {"code": _error_code(error)})


def _w_checks(root: Path, log: dict, *, confirm_h: str | None, discard_k_changes: str | None,
              verified: bool) -> dict:
    """The release-mode return to H's checks: (the head is V's), H confirmed, W0 (no live child), W1.

    Returns the record for 'abort_started' / 'abort_rechecked' ({'head', 'w'},
    plus 'discarded' when --discard-k-changes matched). ``verified`` False is
    the recheck after the head moved behind the abort (R5-OPS-1). The live V
    child is refused by W0 alone (SC4B-N21), before anything is written.
    """
    intent = log["intent"]
    head = stage.read_admission_head(root)
    if verified and head["status"] == "closed" and not _release_head_is_v(root, log, head):
        _refuse("runtime_admission_compaction_head_moved")
    if confirm_h is None or len(confirm_h) < 12 or not intent["source_binding_id"].startswith(confirm_h):
        _refuse("runtime_admission_compaction_abort_confirmation")
    found = _w(root, log)
    report, digest = found["report"], found["sha256"]
    record = {"head": head, "w": {"class": report["class"], "sha256": digest}}
    if report["class"] == W_RESTRICTIVE:
        if discard_k_changes is None:
            _refuse(ABORT_PROTECTED_CHANGED, {"w": report, "w_sha256": digest})  # safe stop SS
        if len(discard_k_changes) < 12 or not digest.startswith(discard_k_changes.lower()):
            _refuse(DISCARD_MISMATCH, {"w": report, "w_sha256": digest})
        record["discarded"] = {"w_sha256": digest, "counts": _w_counts(report)}  # SC-D8 A
    elif discard_k_changes is not None:
        _refuse(DISCARD_MISMATCH, {"w": report, "w_sha256": digest})  # nothing to discard: stricter (SC4B-N6)
    return record


def _no_delegation_records(root: Path) -> None:
    """AUTHZ3-1 before any abort event (SCR-C-2): a delegation record refuses and nothing is written.

    The return to H refuses runtime_admission_delegation_frozen anyway (its
    ownership validation reads no record at epoch >= 5); checked first, the
    refusal leaves the log uncommitted to the abort, so row DL's exit (remove
    the record, then run the same command again) applies unchanged.
    """
    try:
        stage.require_no_delegation_records(root)
    except ValueError as error:
        _refuse(str(error))


def compact_abort(root, compaction_id: str, *, to_h: bool = False, discard_k_writes: bool = False,
                  confirm_h: str | None = None, member: int | None = None, accept_missing: bool = False,
                  discard_k_changes: str | None = None, wall=time.time) -> dict:
    """Abort before the verification run, or return to H after it (Q6 in plain mode, W in release mode).

    ``member`` with ``accept_missing`` (review r3 R3-1) leaves out one of the
    abort's own items (a snapshot, the placeholder, K) that is gone from its
    path and was never renamed by this compaction, like ``retire
    --accept-missing``: logged, no tombstone, no copies-retired.
    ``discard_k_changes`` (release mode, SC-D8 A) is the 12+ hex prefix of
    the latest W report digest: the person's confirmation that nobody used
    the system during V, the only exit from the safe stop SS.
    """
    root = checked_root(root)
    with stage.AdmissionController(root, gate="compaction:" + compaction_id, allow_open_head=True):
        log = _log(root, compaction_id)
        if log["terminal"] or _has(log, "retire_started"):
            _refuse("runtime_admission_compaction_phase")
        _no_delegation_records(root)  # SCR-C-2: before 'abort_started' or any other abort event
        release = release_log(log)
        if discard_k_writes:
            _require_plain_mode(log)  # no --discard-k-writes in release mode (w_abort_predicate.command)
        if discard_k_changes is not None and not release:
            _refuse(RELEASE_MODE_ONLY)  # --discard-k-changes answers a W report (SC-D8 A)
        if discard_k_changes is not None and not to_h:
            _refuse(DISCARD_MISMATCH)  # a discard is part of the return to H only
        if not _has(log, "abort_started"):
            started = _armed_run_consumed(root, log)
            if started and not to_h:
                _refuse("runtime_admission_compaction_abort_after_run")
            data = {"to_h": bool(to_h), "head": stage.read_admission_head(root)}
            if to_h:
                if not started:
                    _refuse("runtime_admission_compaction_abort_to_h_before_run")
                if _has(log, "allow_run_started"):
                    _refuse("runtime_admission_compaction_abort_after_allow_run")
                if release:  # SC-4b: W replaces Q6 (rows V1-V4, RX, SS)
                    data = {"to_h": True, **_w_checks(root, log, confirm_h=confirm_h,
                                                      discard_k_changes=discard_k_changes, verified=True)}
                else:
                    _q6_checks(root, log, discard_k_writes=discard_k_writes, confirm_h=confirm_h, verified=True)
            log = stage.append_compaction_event(root, log, "abort_started", data, at=_now(wall))
        elif _abort_head_moved(root, log):
            # Review r2 N1: the head moved after 'abort_started' (a K run the gate should have refused):
            # whatever ran is checked again as a return to H (Q6, or W in release mode, R5-OPS-1).
            if not to_h:
                _refuse("runtime_admission_compaction_abort_recheck_required")
            if release:
                record = _w_checks(root, log, confirm_h=confirm_h, discard_k_changes=discard_k_changes,
                                   verified=False)
            else:
                record = {"head": _q6_checks(root, log, discard_k_writes=discard_k_writes, confirm_h=confirm_h,
                                             verified=False)}
            log = stage.append_compaction_event(root, log, "abort_rechecked", record, at=_now(wall))
        if accept_missing:
            log = _accept_missing(root, log, member, wall, members=_abort_members(root, log))
        return _finish_abort(root, log, wall)


def _q6_checks(root: Path, log: dict, *, discard_k_writes: bool, confirm_h: str | None, verified: bool) -> dict:
    """The return to H's checks against the current head: no live child, (the verified head), H confirmed,
    Q6, discard. ``verified`` False is the recheck after a K run behind the abort (review r2 N1).
    Plain mode only: a release-mode log returns to H through W (SC-4b), never through Q6."""
    _require_plain_mode(log)
    intent = log["intent"]
    head = stage.read_admission_head(root)
    if head["status"] != "closed":
        run = launcher.read_run_record(root, head["run_id"])
        if run is not None and launcher.child_alive(run["child_pid"], run.get("child_start_time")):
            _refuse("runtime_admission_recovery_child_alive")
    if verified and head["status"] == "closed" and stage.compaction_verified(root, log, head) is None:
        _refuse("runtime_admission_compaction_head_moved")
    if confirm_h is None or len(confirm_h) < 12 or not intent["source_binding_id"].startswith(confirm_h):
        _refuse("runtime_admission_compaction_abort_confirmation")
    outcome = abort_to_h_predicate(root, log)
    if outcome["new_rows"] and not discard_k_writes:
        _refuse("runtime_admission_compaction_discard_k_writes_required", outcome)
    return head


def _abort_head_moved(root: Path, log: dict) -> bool:
    """The head is neither the one the abort last checked nor the H its own return wrote (review r2 N1)."""
    checked = _last(log, "abort_rechecked") or _last(log, "abort_started")
    head = stage.read_admission_head(root)
    return head != checked["data"].get("head") and not _returned_here(root, log, head)


def _returned_here(root: Path, log: dict, head: dict) -> bool:
    """The head is the closed H this compaction's own return record wrote (review F1).

    compaction_return_to_h writes the return record and the manual head, then
    CASes and publishes 'current'; 'returned' is logged after that. A crash in
    between leaves H closed with this record: the return happened.
    """
    intent = log["intent"]
    if head is None or head["status"] != "closed" or head["binding_id"] != intent["source_binding_id"]:
        return False
    if _has(log, "returned"):
        return True
    path = stage._compaction_return_path(root, head)
    if not _lexists(path):
        return False
    record = stage._read_compaction_return(root, head)
    return record["compaction_id"] == log["compaction_id"] and record["h_head"] == intent["head"]


def _armed_run_consumed(root: Path, log: dict) -> bool:
    """serve's 'unconsumed' rule (review F4): verify_started is logged before begin_run's CAS.

    The armed run counts as started only if a running head of the logged
    generation existed: the head is open, or it closed at that generation or
    later, or the run was verified. A failure before the CAS leaves the head
    closed below that generation: nothing ran, and a plain abort is still the
    'before the run' abort.
    """
    _armed, started = stage._armed_run(log)
    if started is None:
        return False
    head = stage.read_admission_head(root)
    return not (head is not None and head["status"] == "closed"
                and head["generation"] < started["data"]["generation"]
                and head["binding_id"] == started["data"]["binding_id"]
                and stage.compaction_verified(root, log, head) is None)


def _finish_abort(root: Path, log: dict, wall) -> dict:
    intent = log["intent"]
    if _abort_head_moved(root, log):
        _refuse("runtime_admission_compaction_abort_recheck_required")  # review r2 N1: compact abort --to-h
    head = stage.read_admission_head(root)
    if head != intent["head"] and not _returned_here(root, log, head):
        k_bindings = {binding_id for binding_id, binding in _bindings(root).items()
                      if binding.get("slot") == intent["k_slot"]}
        if head is None or head["binding_id"] not in k_bindings:
            _refuse("runtime_admission_head_changed")
        head = stage.compaction_return_to_h(root, head, intent["head"], log["compaction_id"])
    # 'current' names H's runtime before K is destroyed (a cut after the CAS, or inside an
    # activation that published K's selection before its CAS, leaves it naming K).
    h_slot = stage.read_binding(root, intent["source_binding_id"])["slot"]
    if not os.path.islink(root / "current") or os.readlink(root / "current") != "slots/" + h_slot + "/runtime":
        stage._publish_selection(root, h_slot)
    if not _has(log, "returned"):
        log = stage.append_compaction_event(root, log, "returned", {"head": head}, at=_now(wall))
    _destroy_own_staging(root, log, wall)
    # Review r2 N2: every item keeps the number its recording event gave it, and K has a fixed one, so a
    # resumed abort finds a half-walked member in its quarantine instead of renumbering past it.
    work = _abort_members(root, log)
    for member in work:
        log = _retire_member(root, log, member, set(), wall)
    _drop_scratch(root, log)
    destroyed = len({event["data"]["member"] for event in stage.compaction_events(log, "destroyed")})
    log = stage.append_compaction_event(root, log, "aborted", {"destroyed": destroyed,
                                                               "to_h": bool(_last(log, "abort_started")["data"]
                                                                            ["to_h"])}, at=_now(wall))
    return {"compaction_id": log["compaction_id"], "phase": "aborted", "head": head, "destroyed": destroyed}


def _abort_members(root: Path, log: dict) -> list[dict]:
    """An abort's members: this compaction's own work (stable numbers) and K (fixed number), if made."""
    work = own_work(root, log)
    k_member = _k_abort_member(root, log)
    if k_member is not None:
        work.append(k_member)
    return work


def _k_abort_member(root: Path, log: dict) -> dict | None:
    """K as an abort member, at its path or (after a cut mid-walk) in its quarantine; None if never made.

    Review r4 R4-2: once K is made (k_runtime, or its prepare record), K gone
    from slots/<K> with no 'retiring' is still a member: _retire_member
    refuses member_missing and ``--member n --accept-missing`` records it.
    """
    intent = log["intent"]
    k_slot = root / "slots" / intent["k_slot"]
    number = len(intent["members"]) + K_ABORT_NUMBER
    k_event = _last(log, "k_runtime")
    proof = _k_proof(root, log)
    identity = (k_event["data"].get("slot_identity") if k_event else None) or (proof or {}).get("slot_identity") \
        or _retiring_identity(log, number)
    quarantine = root / "compaction" / log["compaction_id"] / "retiring" / str(number)
    if not _lexists(k_slot) and _retiring_identity(log, number) is None and k_event is None and proof is None:
        return None
    if _lexists(k_slot) and (identity is None or _identity(k_slot) != identity):
        _refuse("runtime_admission_compaction_k_identity")
    state_identity = (k_event["data"].get("state_identity") if k_event else None) or \
        (proof or {}).get("state_identity")
    if state_identity is None:
        where = k_slot if _lexists(k_slot) else quarantine
        state_identity = _identity(where / "state") if _lexists(where / "state") else None
    return {"class": "managed_slot", "path": str(k_slot), "identity": identity, "proof": "k_runtime",
            "slot": intent["k_slot"], "state_root": str(k_slot / "state"), "state_identity": state_identity,
            "binding_ids": sorted(key for key, item in _bindings(root).items() if item.get("slot") == intent["k_slot"]),
            "copy_ids": [record["copy_id"] for record in catalog_records(root)
                         if record["destination"] == str(k_slot / "state")], "n": number}


def _drop_scratch(root: Path, log: dict) -> None:
    """Remove the empty scratch directories; anything left in the quarantine refuses (review r2 N2)."""
    scratch = root / "compaction" / log["compaction_id"]
    retiring = scratch / "retiring"
    if _lexists(retiring) and os.listdir(retiring):
        _refuse("runtime_admission_compaction_retiring_residue", {"residue": sorted(os.listdir(retiring))[:20]})
    for leftover in (retiring, scratch):
        if _lexists(leftover) and not os.listdir(leftover):
            os.rmdir(leftover)
            _fsync_directory(leftover.parent)
    if _lexists(scratch):
        _refuse("runtime_admission_compaction_retiring_residue", {"residue": sorted(os.listdir(scratch))[:20]})


# --- resume and status --------------------------------------------------------------------------------

def compact_resume(root, compaction_id: str, *, archive=None, worker=None, scan=None, wall=time.time) -> dict:
    """Continue whichever phase is due: abort, retire, or C2a-C5 (design v2 phases)."""
    root = checked_root(root)
    log = _log(root, compaction_id)
    if log["terminal"]:
        return {"compaction_id": compaction_id, "phase": log["events"][-1]["kind"], "noop": True}
    if _has(log, "abort_started"):
        with stage.AdmissionController(root, gate="compaction:" + compaction_id, allow_open_head=True):
            _no_delegation_records(root)  # SCR-C-2: the resumed abort writes nothing either
            return _finish_abort(root, _log(root, compaction_id), wall)
    if _has(log, "retire_started"):
        with stage.AdmissionController(root, gate="compaction:" + compaction_id):
            return _finish_retire(root, _log(root, compaction_id), wall)
    if _has(log, "activated"):
        return {"compaction_id": compaction_id, "phase": "activated", "noop": True}
    return _continue(root, compaction_id, archive=Path(archive) if archive else None, worker=worker, wall=wall)


def _started_run_id(root: Path, started: dict) -> str | None:
    """The run_id of the run serve began for ``verify_started`` (its immutable run record), or None."""
    runs = root / "admission" / "runs"
    if not _lexists(runs):
        return None
    other = (".exit.json", stage.VERIFY_READY_SUFFIX, stage.VERIFY_SMOKE_STOP_SUFFIX)
    for name in sorted(os.listdir(runs)):
        if not name.endswith(".json") or name.endswith(other):
            continue
        record = launcher.read_run_record(root, name[:-len(".json")])
        if (record is not None and record.get("generation") == started["data"]["generation"]
                and record.get("binding_id") == started["data"]["binding_id"]):
            return record["run_id"]
    return None


def verify_branch_reached(root: Path, log: dict) -> str | None:
    """R5V-5 'verify branch reached' of a release-mode log's armed run, from its records only.

    'yes': serve wrote ready.json, i.e. NEW's code sent VERIFY=<run_id> from its
    verify_only branch before READY. That proves only that the branch was
    reached, not that V was isolated or unused (the SC-D8 A confirmation
    rests on a person's direct knowledge). 'no': the run started without it.
    'unreadable': a run record or a verify record cannot be read (review RB-3;
    status still reports the phase). None: no armed run has started. No clock
    is read.
    """
    _armed, started = stage._armed_run(log)
    if started is None:
        return None
    try:
        run_id = _started_run_id(root, started)
    except (OSError, ValueError):
        return "unreadable"  # review RB-3: an unreadable run record (LauncherRefusal is a ValueError)
    if run_id is None:
        return None
    try:
        ready, _smoke = stage.verify_run_records(root, run_id)
    except (OSError, ValueError):
        return "unreadable"  # review RB-3: status still shows the log's phase
    return "yes" if ready is not None and ready["verify_acknowledged"] is True else "no"


def _armed_run_state(root: Path, log: dict) -> str:
    """Status' view of the armed run (row A0, R3-OPS-4): read-only, no W, no clock."""
    armed, _started = stage._armed_run(log)
    if armed is None:
        return "not armed"
    return "consumed" if _armed_run_consumed(root, log) else "not consumed"


def compact_status(root, compaction_id: str | None = None) -> dict:
    """Read-only, content-free: each log's phase, without a lease.

    An unreadable log is reported by id with its code (review F3), never a
    traceback: every lease-taking command refuses while it exists, and the
    runbook's 'compactación trabada' section says what to do.
    """
    root = checked_root(root)
    directory = stage.compaction_directory(root)
    names = sorted(name for name in os.listdir(directory) if name not in {"retired", "returns"}
                   and not name.startswith(".")) if _lexists(directory) else []
    found = []
    for name in names:
        if compaction_id is not None and name != compaction_id:
            continue
        try:
            if not _lexists(directory / name / "intent.json"):
                continue
            log = stage.read_compaction_log(root, name)
            mode = RELEASE_MODE if release_log(log) else "plain"
            reached = verify_branch_reached(root, log) if mode == RELEASE_MODE else None
            armed = _armed_run_state(root, log) if mode == RELEASE_MODE else None
        except (OSError, ValueError) as error:
            found.append({"compaction_id": name[:64], "invalid": True,
                          "code": str(error)[:200] if str(error).startswith("runtime_admission_")
                          else "runtime_admission_compaction_log_invalid"})
            continue
        found.append({"compaction_id": log["compaction_id"], "terminal": log["terminal"],
                      "phase": log["events"][-1]["kind"] if log["events"] else "intent",
                      "phases": [event["kind"] for event in log["events"]],
                      "members": len(log["intent"]["members"]),
                      "mode": mode, **({"verify_branch_reached": reached or "not run",
                                        "armed_run": armed, "verified": _has(log, "verified")}
                                       if mode == RELEASE_MODE else {})})
    return {"compactions": found}


# --- CLI ------------------------------------------------------------------------------------------------

def add_commands(commands) -> None:
    """Register ``compact`` and its sub-commands on stage_runtime's parser."""
    compact = commands.add_parser("compact", help="Owner-state compaction (U3b-0c)")
    sub = compact.add_subparsers(dest="compact_command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("k_slot")
    plan.add_argument("--archive", type=Path, required=True)
    plan.add_argument("--retire-prepare-leftovers", action="store_true")
    plan.add_argument("--unit", action="append", default=[],
                      help="A user unit that is kza's besides kza-voice, kza-llm-fast and kza-code-index")
    start = sub.add_parser("start")
    start.add_argument("k_slot")
    start.add_argument("--archive", type=Path, required=True)
    start.add_argument("--plan-sha256", required=True)
    start.add_argument("--acknowledge-untouched", required=True)
    start.add_argument("--destroy-rollback-states", action="store_true")
    start.add_argument("--accept-owner-only-loss", action="store_true")
    start.add_argument("--retire-prepare-leftovers", action="store_true")
    start.add_argument("--unit", action="append", default=[])
    for name in ("verify", "retire", "resume", "abort", "allow-run", "status"):
        parser = sub.add_parser(name)
        parser.add_argument("compaction_id", nargs="?" if name == "status" else None)
        if name == "verify":
            parser.add_argument("--smoke-seconds", type=int, default=None,
                                help="Plain mode: default 120. Release mode: required, 10..60 (SC4A-N14)")
        if name == "retire":
            parser.add_argument("--member", type=int)
            parser.add_argument("--accept-identity-change", action="store_true")
            parser.add_argument("--accept-owner-only-loss", metavar="COVERAGE_SHA256")
            parser.add_argument("--accept-missing", action="store_true")
        if name == "resume":
            parser.add_argument("--archive", type=Path)
        if name == "abort":
            parser.add_argument("--to-h", action="store_true")
            parser.add_argument("--discard-k-writes", action="store_true")
            parser.add_argument("--discard-k-changes", metavar="W_SHA256_PREFIX",
                                help="Release mode, safe stop only (SC-D8 A): the latest W report digest")
            parser.add_argument("--confirm-h")
            parser.add_argument("--member", type=int)
            parser.add_argument("--accept-missing", action="store_true")
        if name == "allow-run":
            parser.add_argument("--reason", required=True)
            parser.add_argument("--hours", type=int, required=True)


def run_command(args, *, scan=None, worker=None) -> int:
    """Run one ``compact`` sub-command; prints content-free JSON; 78 on a coded refusal."""
    command = args.compact_command
    try:
        if command == "plan":
            result = compact_plan(args.root, args.k_slot, args.archive,
                                  retire_prepare_leftovers=args.retire_prepare_leftovers, scan=scan,
                                  units=tuple(getattr(args, "unit", ()) or ()))
        elif command == "start":
            result = compact_start(args.root, args.k_slot, args.archive, plan_sha256=args.plan_sha256,
                                   acknowledge_untouched=args.acknowledge_untouched,
                                   destroy_rollback_states=args.destroy_rollback_states,
                                   accept_owner_only_loss=args.accept_owner_only_loss,
                                   retire_prepare_leftovers=args.retire_prepare_leftovers, scan=scan, worker=worker,
                                   units=tuple(getattr(args, "unit", ()) or ()))
        elif command == "verify":
            result = compact_verify(args.root, args.compaction_id, smoke_seconds=args.smoke_seconds)
        elif command == "retire":
            result = compact_retire(args.root, args.compaction_id, member=args.member,
                                    accept_identity_change=args.accept_identity_change,
                                    accept_owner_only_loss=args.accept_owner_only_loss,
                                    accept_missing=getattr(args, "accept_missing", False), scan=scan)
        elif command == "resume":
            result = compact_resume(args.root, args.compaction_id, archive=args.archive, worker=worker, scan=scan)
        elif command == "abort":
            result = compact_abort(args.root, args.compaction_id, to_h=args.to_h,
                                   discard_k_writes=args.discard_k_writes, confirm_h=args.confirm_h,
                                   member=getattr(args, "member", None),
                                   accept_missing=getattr(args, "accept_missing", False),
                                   discard_k_changes=getattr(args, "discard_k_changes", None))
        elif command == "allow-run":
            result = compact_allow_run(args.root, args.compaction_id, reason=args.reason, hours=args.hours)
        else:
            result = compact_status(args.root, args.compaction_id)
    except CompactionRefusal as refusal:
        print(json.dumps({"refused": str(refusal), "report": refusal.report}, sort_keys=True, default=str))
        return 78
    except ValueError as error:
        print(json.dumps({"refused": str(error)[:200]}, sort_keys=True))
        return 78
    except sqlite3.Error:
        # Review F12: coded and content-free (no path, no message text), never a traceback.
        print(json.dumps({"refused": "runtime_admission_compaction_database_unreadable"}, sort_keys=True))
        return 78
    except OSError as error:
        print(json.dumps({"refused": "runtime_admission_compaction_io_error",
                          "errno": errno.errorcode.get(error.errno, "unknown")}, sort_keys=True))
        return 78
    print(json.dumps(result, sort_keys=True, default=str))
    return 0
