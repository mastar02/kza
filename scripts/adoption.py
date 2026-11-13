"""Adoption library (U3b-0e2): a copy of today's production state becomes the first managed state.

Adoption design v3 (data/autonomous-roadmap-20261102/t040-t041/u3b-0e-adoption-design-v3.json),
sections 1-5, with the user's final answers P1-P8 (user-decisions-20261110.json).
This module is the controller side: stdlib only (plus PyYAML for the 2a
editor) and it never imports ``src``, so a G1-guarded root cannot refuse it.
Everything R1's own code must do on the copy (schema-only migration, ledger
seed, residue re-queue, first-start target set) runs in the schema-only child
(:mod:`src.core.schema_only`, phases of :mod:`src.core.adoption_schema`).
The operator commands that drive this library are U3b-0e3.

Source discipline (N7, F9): the source is only read, byte by byte, with
O_RDONLY|O_NOFOLLOW; each SQLite file travels with its -wal and -journal;
every regular file is hashed in the same pass and compared with its copy; no
SQLite connection is ever made to a source. :class:`SourceGuard` installs an
audit hook that refuses any write-mode open, SQLite connect, remove, rename,
mkdir (and similar) on a source path, and counts what it saw.

Refusal codes are content-free: paths name state-relative files, tables and
columns, never values.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import threading
from urllib.parse import unquote

stage = None  # scripts/stage_runtime
launcher = None  # scripts/managed_launcher


def bind(stage_module, launcher_module) -> None:
    """Use these stage_runtime / managed_launcher implementations (the CLI binds by path)."""
    global stage, launcher
    stage, launcher = stage_module, launcher_module


if __package__ == "scripts":  # in-process users (tests)
    from scripts import managed_launcher as _launcher_module  # noqa: E402
    from scripts import stage_runtime as _stage_module  # noqa: E402

    bind(_stage_module, _launcher_module)


class AdoptionRefusal(ValueError):
    """A content-free adoption refusal; ``str(error)`` is the code."""


def _refuse(code: str):
    raise AdoptionRefusal(code)


SCHEMA_VERSION = 1
COPY_MARKER = ".kza-adoption-copy.json"  # = src.core.schema_only.COPY_MARKER (a test pins it)
SQLITE_HEADER = b"SQLite format 3\x00"
SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")
SIDE_SUFFIXES = ("-wal", "-journal", "-shm")
CARRIED_TOP = ("config", "data", ".env")
CONTAINER_LINKS = ("config", "data", ".env", "models", "logs")
SCHEMA_ONLY_ENV = "KZA_ADOPTION_SCHEMA_ONLY"
# = src.core.entry_guard.SCHEMA_ONLY_BOOTSTRAP (a test pins it): the controller never imports src.
SCHEMA_ONLY_BOOTSTRAP = "\n".join([
    "import sys",
    "sys.path.insert(0, sys.argv[1])",
    "from src.core.schema_only import main",
    "raise SystemExit(main(sys.argv[2:]))",
])
_CHILD_ENVIRONMENT = ("LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SQLITE_TMPDIR")

# --- P2=A: what is not carried, and the reviewed database-scoped drop list -------------------------

# data/audit.db (with its -wal/-journal/-shm) stays in the legacy image.
NOT_CARRIED = frozenset({"data/audit.db"})
# The pre-policy 'events' table of events.db, exactly as 8558ff1^ created it.
LEGACY_EVENTS_COLUMNS = ("id", "timestamp", "event_type", "entity_id", "action", "user_id", "user_name",
                         "trigger_phrase", "context_json", "hour", "minute", "weekday", "is_weekend",
                         "time_of_day")
# Each entry is reviewed: (database, table) -> the exact columns it must have.
# The file goes to the scratch, the table is dropped there and the database
# reaches the destination only through VACUUM INTO (review v2 N2).
DROP_LIST = {"data/events.db": {"events": LEGACY_EVENTS_COLUMNS}}

# --- S3 (N11): the hand-written b0866e8 -> R1 schema table, checked against the generated diff -----

# (database, table, column, default as SQLite would store it in each existing row)
COLUMN_ADDITIONS = (
    ("data/reminders.db", "reminders", "scope", "private"),
    ("data/memory_privacy.db", "live_deliveries", "owner_namespace", "legacy-unbound"),
    ("data/memory_privacy.db", "voice_capture_reviews", "audio_metadata_json", None),
    ("data/memory_privacy.db", "voice_capture_reviews", "transcript_metadata_json", None),
)
# Runtime leases R1 rebuilds with owner_namespace (no default) and empties at
# every open (src/dashboard/live_receipts.py _recover); not protected tables.
REBUILT_LEASES = (
    ("data/memory_privacy.db", "live_connections", ("owner_namespace",)),
    ("data/memory_privacy.db", "live_resets", ("owner_namespace",)),
)
# Tables R1's schema-only open adds to an existing database: kind of content.
ADDITIVE_TABLES = {
    "data/memory_privacy.db": {
        "deletion_ledger_v1": "ledger", "deletion_ledger_salt_v1": "salt",
        "fact_owners": "empty", "fact_owner_revisions": "empty", "fact_owner_schema": "marker",
        "voice_attribution_consents": "empty", "voice_verification_attempts": "empty",
        "voice_attribution_events": "empty", "voice_attribution_suspensions": "empty",
        "voice_attribution_control": "empty", "voice_attribution_schema": "marker",
        "voice_capture_review_schema": "marker", "voice_person_schema": "marker",
        "voice_profile_people": "empty",
    },
}
# Targets R1 registers at its first start that b0866e8 never did (reviewed).
NEW_R1_TARGETS = ({"participant": "producers", "version": 1,
                   "descriptor": {"kind": "runtime_producers", "protocol": 1}},)
RESIDUE_REASON = "interrupted_by_restart"  # = adoption_schema.RESIDUE_REASON
INDEX_RECHECK = "index_startup_recheck"  # = IndexCleanupError.RECHECK

PATH_CLASSES = ("legacy_state", "container", "legacy_checkout", "models", "logs", "code_index", "other")
_ABSOLUTE = re.compile(r"(?<![\w.~-])/[^\s\"'\\,;:\]\[{}()<>|]+")


@dataclass(frozen=True)
class SourceLayout:
    """Where the production state lives; every path is absolute.

    Attributes:
        container: The 63ea8d3 slot state, a directory of the five links.
        legacy_state: The b0866e8 slot state the config/data/.env links resolve into.
        models_root: Where the models link resolves (/home/kza/kza/models).
        legacy_checkout: The 2026-10-28 original (/home/kza/kza).
        metrics_root: The legacy metrics directory (/home/kza/logs).
        code_index: The code-index tree (/home/kza/code-index), census only.
    """

    container: Path
    legacy_state: Path
    models_root: Path
    legacy_checkout: Path
    metrics_root: Path
    code_index: Path = Path("/home/kza/code-index")


def _prefixes(path: Path) -> tuple[str, ...]:
    values = {os.path.normpath(os.fspath(path))}
    values.add(os.path.realpath(os.fspath(path)))
    return tuple(sorted(values))


def _under(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value == prefix or value.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def classify(value: str, layout: SourceLayout) -> str:
    """The census class of one absolute path string (never its value)."""
    value = os.path.normpath(value)
    for name, root in (("container", layout.container), ("legacy_state", layout.legacy_state),
                       ("models", layout.models_root), ("legacy_checkout", layout.legacy_checkout),
                       ("logs", layout.metrics_root), ("code_index", layout.code_index)):
        if _under(value, _prefixes(root)):
            return name
    return "other"


def read_source_layout(layout: SourceLayout) -> dict:
    """The five container readlinks; any other target or entry refuses (runtime_admission_adoption_source_layout).

    config, data and .env must resolve to the same names in the legacy state,
    models to the models root, and logs into the legacy state or checkout.
    """
    code = "runtime_admission_adoption_source_layout"
    container = Path(layout.container)
    try:
        if not stat.S_ISDIR(container.lstat().st_mode):
            _refuse(code)
        names = sorted(os.listdir(container))
    except OSError:
        _refuse(code)
    if names != sorted(CONTAINER_LINKS):
        _refuse(code)
    links, resolved = {}, {}
    for name in CONTAINER_LINKS:
        path = container / name
        if not stat.S_ISLNK(path.lstat().st_mode):
            _refuse(code)
        links[name] = os.readlink(path)
        resolved[name] = os.path.realpath(path)
    state = os.path.realpath(layout.legacy_state)
    for name in ("config", "data", ".env"):
        if resolved[name] != os.path.join(state, name):
            _refuse(code)
    if resolved["models"] != os.path.realpath(layout.models_root):
        _refuse(code)
    if not (_under(resolved["logs"], (state,)) or _under(resolved["logs"], _prefixes(layout.legacy_checkout))):
        _refuse(code)
    return {"readlinks": links, "classes": {name: classify(value, layout) for name, value in resolved.items()}}


# --- the source guard (audit hook) -----------------------------------------------------------------

_GUARD_LOCK = threading.Lock()
_GUARD_STATE = {"installed": False, "roots": [], "refused": Counter(), "copies": Counter()}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_PATH_EVENTS = {"os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1), "os.mkdir": (0,), "os.chmod": (0,),
                "os.chown": (0,), "os.truncate": (0,), "os.utime": (0,), "os.link": (0, 1),
                "os.symlink": (1,), "shutil.rmtree": (0,), "shutil.copyfile": (1,), "shutil.move": (0, 1),
                "os.removexattr": (0,), "os.setxattr": (0,)}


def _event_path(value) -> str | None:
    if isinstance(value, (str, bytes, os.PathLike)):
        text = os.fsdecode(value)
        if text.startswith("file:"):
            text = unquote(text[5:].split("?", 1)[0])
        if text in ("", ":memory:"):
            return None
        return os.path.realpath(text)
    return None


def _guarded(path: str | None) -> bool:
    return path is not None and any(_under(path, (root,)) for root in _GUARD_STATE["roots"])


def _audit(event: str, args: tuple) -> None:
    if not _GUARD_STATE["roots"]:
        return
    paths: tuple = ()
    if event == "open" and args:
        mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
        writing = (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
            type(flags) is int and flags & _WRITE_FLAGS)
        if writing:
            paths = (args[0],)
    elif event == "sqlite3.connect" and args:
        path = _event_path(args[0])
        if _guarded(path):
            _GUARD_STATE["refused"][event] += 1
            raise AdoptionRefusal("runtime_admission_adoption_source_access:" + event)
        if path is not None:
            _GUARD_STATE["copies"][event] += 1
        return
    elif event in _PATH_EVENTS:
        paths = tuple(args[index] for index in _PATH_EVENTS[event] if index < len(args))
    for value in paths:
        if _guarded(_event_path(value)):
            _GUARD_STATE["refused"][event] += 1
            raise AdoptionRefusal("runtime_admission_adoption_source_access:" + event)


class SourceGuard:
    """Refuse writes and SQLite connects on source paths for the duration (N7, T-D2).

    Python audit hooks cannot be removed, so the hook is installed once per
    process and acts only while some guard is active. ``refused`` counts the
    refused events; ``connects`` counts SQLite connects seen elsewhere (proof
    the hook observes this process).
    """

    def __init__(self, *roots: Path):
        self.roots = tuple(os.path.realpath(os.fspath(root)) for root in roots)

    def __enter__(self) -> "SourceGuard":
        with _GUARD_LOCK:
            if not _GUARD_STATE["installed"]:
                sys.addaudithook(_audit)
                _GUARD_STATE["installed"] = True
            _GUARD_STATE["roots"] = [*_GUARD_STATE["roots"], *self.roots]
            self._refused, self._copies = Counter(_GUARD_STATE["refused"]), Counter(_GUARD_STATE["copies"])
        return self

    def __exit__(self, *exc) -> None:
        with _GUARD_LOCK:
            roots = list(_GUARD_STATE["roots"])
            for root in self.roots:
                roots.remove(root)
            _GUARD_STATE["roots"] = roots

    @property
    def refused(self) -> int:
        return sum((_GUARD_STATE["refused"] - self._refused).values())

    @property
    def connects(self) -> int:
        return sum((_GUARD_STATE["copies"] - self._copies).values())


def source_roots(layout: SourceLayout) -> tuple[Path, ...]:
    """Every tree the guard protects: the legacy state, container, checkout and metrics."""
    return (layout.legacy_state, layout.container, layout.legacy_checkout, layout.metrics_root)


# --- byte copy with hashes (N7, F9) ----------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path | None, moving: Counter | None = None) -> tuple[list, bytes]:
    """Read ``source`` once (nofollow), hash it, optionally write ``destination``; return entry and header.

    A file that changes while it is read refuses (source_changed); only the
    advisory dry-run passes ``moving``, which counts it instead (U3b-0e3 F2).
    """
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, header = hashlib.sha256(), b""
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            _refuse("runtime_admission_adoption_source_special")
        output = None
        if destination is not None:
            stage.private_directories(destination.parent)
            output = os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")
        try:
            while chunk := stream.read(1 << 20):
                if len(header) < 16:
                    header += chunk[:16 - len(header)]
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        if moving is None:
            _refuse("runtime_admission_adoption_source_changed")
        moving["during_copy"] += 1
    entry = [before.st_ino, before.st_size, before.st_mtime_ns, digest.hexdigest()]
    if destination is not None and _sha256_file(destination) != entry[3]:
        _refuse("runtime_admission_adoption_copy_mismatch")
    return entry, header


def _walk_files(root: Path):
    """Every regular file under ``root`` (lstat only): a symlink or special file refuses."""
    pending = [Path(root)]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for entry in sorted(iterator, key=lambda item: item.name):
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    _refuse("runtime_admission_adoption_source_symlink")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    yield Path(entry.path)
                else:
                    _refuse("runtime_admission_adoption_source_special")


def _side_of(relative: str) -> tuple[str, str]:
    """(main file, suffix) for a -wal/-journal/-shm sibling; (relative, '') otherwise."""
    for suffix in SIDE_SUFFIXES:
        if relative.endswith(suffix):
            return relative[:-len(suffix)], suffix
    return relative, ""


def copy_source(layout: SourceLayout, destination: Path, scratch: Path, moving: Counter | None = None) -> dict:
    """Byte-copy the legacy state's carried trees into ``destination`` (N7, F9, P2=A).

    Every regular file under the legacy state is read once and hashed (the
    full-file source manifest), including what is not carried. Carried files
    (config, data, .env) are written and re-hashed; a database in DROP_LIST
    goes to ``scratch`` instead, a NOT_CARRIED database is only hashed, and
    -shm files are hashed but never copied (SQLite rebuilds them). SQLite
    files are recognised by header as well as suffix.

    Returns:
        ``manifest`` (relative, inode, size, mtime_ns, sha256 per file) and
        content-free counts.
    """
    source = Path(layout.legacy_state)
    entries, report = [], Counter()
    sqlite_files, sides = set(), {}
    files = sorted(_walk_files(source))
    destination, scratch = Path(destination), Path(scratch)
    for path in files:
        relative = path.relative_to(source).as_posix()
        main, suffix = _side_of(relative)
        top = relative.split("/", 1)[0]
        target = None
        if top in CARRIED_TOP and suffix != "-shm":
            if main in NOT_CARRIED:
                report["not_carried_files"] += 1
            elif main in DROP_LIST:
                target = scratch / relative
                report["scratch_files"] += 1
            else:
                target = destination / relative
        elif suffix == "-shm":
            report["shm_not_copied"] += 1
        else:
            report["outside_carried_files"] += 1
        entry, header = _copy_file(path, target, moving)
        entries.append([relative, *entry])
        if suffix:
            sides[relative] = (main, suffix)
        elif header == SQLITE_HEADER or Path(relative).suffix in SQLITE_SUFFIXES:
            sqlite_files.add(relative)
            if header != SQLITE_HEADER and entry[2]:
                report["sqlite_suffix_without_header"] += 1
    for relative, (main, suffix) in sides.items():
        report["sqlite" + suffix] += 1
        if main not in sqlite_files:
            _refuse("runtime_admission_adoption_orphan_journal")
    report["files"] = len(entries)
    report["sqlite_files"] = len(sqlite_files)
    return {"manifest": sorted(entries), "counts": dict(sorted(report.items()))}


def metrics_store_set(layout: SourceLayout, metrics_path: str) -> dict:
    """The DiagnosticStore protocol files of the legacy metrics store (review v1 F10).

    ``metrics_path`` is the legacy configuration value; its policy file is
    ``<stem>_policy_v1<suffix>`` in the metrics root, with '.1', the
    '.sqlite3' catalog (+ -wal/-journal) and '.quarantine-*' siblings.
    A '.diag-*.tmp' sibling refuses; the pre-policy file is listed only.
    """
    original = Path(metrics_path)
    name = original.name if original.stem.endswith("_policy_v1") else original.stem + "_policy_v1" + original.suffix
    root = Path(layout.metrics_root)
    present = sorted(item.name for item in root.iterdir()) if root.is_dir() else []
    if any(item.startswith(".diag-") and item.endswith(".tmp") for item in present):
        _refuse("runtime_admission_adoption_metrics_unsettled")
    wanted = {name, name + ".1", name + ".sqlite3", name + ".sqlite3-wal", name + ".sqlite3-journal"}
    members = [item for item in present if item in wanted
               or item.startswith((name + ".quarantine-", name + ".1.quarantine-"))]
    outside = [item for item in present if item not in members and item.startswith(original.stem)
               and not item.endswith(".lock")]
    for item in members:
        if Path(root / item).is_symlink() or not (root / item).is_file():
            _refuse("runtime_admission_adoption_source_special")
    return {"policy_name": name, "members": members, "outside": outside}


def copy_metrics_store(layout: SourceLayout, metrics_path: str, destination: Path,
                       moving: Counter | None = None) -> dict:
    """Copy the metrics store set to <state>/data/obs/ under the same names; staged must be empty."""
    found = metrics_store_set(layout, metrics_path)
    entries = []
    obs = Path(destination) / "data" / "obs"
    for item in found["members"]:
        entry, _header = _copy_file(Path(layout.metrics_root) / item, obs / item, moving)
        entries.append(["metrics/" + item, *entry])
    catalog = obs / (found["policy_name"] + ".sqlite3")
    if catalog.is_file():
        recover_sqlite_file(catalog)
        with closing(sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True)) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "staged" in names and db.execute("SELECT 1 FROM staged LIMIT 1").fetchone():
                _refuse("runtime_admission_adoption_metrics_unsettled")
    return {"manifest": entries, "members": len(found["members"]), "outside_pre_policy": len(found["outside"]),
            "policy_name": found["policy_name"]}


# --- recovery on the copy ---------------------------------------------------------------------------

def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as stream:
            return stream.read(16) == SQLITE_HEADER
    except OSError:
        return False


def sqlite_files(state: Path) -> list[Path]:
    """Every SQLite main file under ``state``, by header (a test fixture also uses suffixes)."""
    return sorted(path for path in Path(state).rglob("*")
                  if path.is_file() and not path.is_symlink() and not _side_of(path.name)[1] and _is_sqlite(path))


def recover_sqlite_file(path: Path) -> str | None:
    """Recover one copied database in place: roll back a hot journal, checkpoint a WAL.

    Returns 'journal' or 'wal' for what was recovered, None otherwise.
    Refuses a -journal beside a WAL database, a failing quick_check, or any
    -wal/-journal left afterwards.
    """
    path = Path(path)
    journal, wal = Path(str(path) + "-journal"), Path(str(path) + "-wal")
    had_journal = journal.exists() and journal.stat().st_size > 0
    had_wal = wal.exists()
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        rows = db.execute("PRAGMA quick_check").fetchall()
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        if (had_journal and mode == "wal") or (had_wal and mode != "wal"):
            _refuse("runtime_admission_adoption_journal_unexpected")
        if mode == "wal":
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if rows != [("ok",)]:
        _refuse("runtime_admission_adoption_quick_check")
    if journal.exists() and journal.stat().st_size == 0:
        journal.unlink()
    if journal.exists() or wal.exists():
        _refuse("runtime_admission_adoption_journal_left")
    shm = Path(str(path) + "-shm")
    if shm.exists():
        shm.unlink()
    return "journal" if had_journal else "wal" if had_wal else None


def recover_copies(state: Path) -> dict:
    """Recover every copied database; content-free counts of hot journals and WALs."""
    counts = Counter()
    for path in sqlite_files(state):
        counts[recover_sqlite_file(path) or "clean"] += 1
    return {"hot_journals_recovered": counts["journal"], "wal_checkpointed": counts["wal"],
            "databases": sum(counts.values())}


def drop_torn_copies(roots: tuple[Path, ...], moving: Counter) -> None:
    """Advisory only (U3b-0e3 review R2-1): a copy a live writer tore is left out and counted.

    A database whose recovery fails (quick_check, an unexpected or leftover
    journal, or sqlite3.DatabaseError such as 'malformed' or 'not a
    database') is removed from the private copy with its -wal/-journal/-shm
    and counted in ``moving['torn']``; the census continues without it.
    Prepare and commit never call this: there a torn copy refuses.
    """
    for root in roots:
        if not Path(root).is_dir():
            continue
        for path in sqlite_files(Path(root)):
            try:
                recover_sqlite_file(path)
            except (AdoptionRefusal, sqlite3.DatabaseError):
                for suffix in ("", *SIDE_SUFFIXES):
                    Path(str(path) + suffix).unlink(missing_ok=True)
                moving["torn"] += 1


def finish_copies(state: Path) -> None:
    """Checkpoint every database once more (read-only readers may leave a -wal); none may remain."""
    recover_copies(state)
    for path in Path(state).rglob("*"):
        if path.is_file() and _side_of(path.name)[1]:
            _refuse("runtime_admission_adoption_journal_left")


def drop_and_vacuum(scratch: Path, destination: Path) -> dict:
    """P2=A in the scratch: drop each listed table, then VACUUM INTO the destination (N2).

    The scratch files are removed afterwards, so the dropped text never
    touches the destination's disk. SQLite's temporary storage is memory and
    its temporary directory the scratch for the drop (review F5). The
    destination must have freelist 0, no -wal or -journal, a clean
    quick_check and none of the dropped tables.
    """
    report = {}
    for relative, drops in sorted(DROP_LIST.items()):
        source, target = Path(scratch) / relative, Path(destination) / relative
        if not source.exists():
            continue
        recover_sqlite_file(source)
        if target.exists() or target.is_symlink():
            _refuse("runtime_admission_adoption_drop_destination")
        stage.private_directories(target.parent)
        dropped = 0
        with closing(sqlite3.connect(source, isolation_level=None)) as db, _scratch_temp(db, scratch):
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, columns in drops.items():
                if table not in names:
                    continue
                actual = tuple(row[1] for row in db.execute('PRAGMA table_info("' + table + '")'))
                if actual != columns:
                    _refuse("runtime_admission_adoption_drop_shape:" + relative + ":" + table)
                db.execute('DROP TABLE "' + table + '"')
                dropped += 1
            db.execute("VACUUM INTO ?", (str(target),))
        for suffix in ("", *SIDE_SUFFIXES):
            leftover = Path(str(source) + suffix)
            if leftover.exists():
                leftover.unlink()
        os.chmod(target, 0o600)
        with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as db:
            freelist = db.execute("PRAGMA freelist_count").fetchone()[0]
            check = db.execute("PRAGMA quick_check").fetchall()
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if freelist != 0 or check != [("ok",)] or names & set(drops):
            _refuse("runtime_admission_adoption_vacuum:" + relative)
        if any(Path(str(target) + suffix).exists() for suffix in ("-wal", "-journal")):
            _refuse("runtime_admission_adoption_vacuum:" + relative)
        report[relative] = {"tables_dropped": dropped, "temp_store": "memory", "temp_directory": "scratch"}
    return report


@contextmanager
def _scratch_temp(db: sqlite3.Connection, scratch: Path):
    """temp_store=MEMORY and temp_store_directory=<scratch> for one drop; the directory is reset after."""
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA temp_store_directory='" + os.fspath(Path(scratch).resolve()).replace("'", "''") + "'")
    if (db.execute("PRAGMA temp_store").fetchone()[0] != 2
            or db.execute("PRAGMA temp_store_directory").fetchone()[0] != os.fspath(Path(scratch).resolve())):
        _refuse("runtime_admission_adoption_scratch_temp")
    try:
        yield
    finally:
        db.execute("PRAGMA temp_store_directory=''")


def write_copy_marker(state: Path, plan_sha256: str) -> None:
    """The adoption-copy marker the schema-only child requires (0e1 review F1)."""
    if type(plan_sha256) is not str or re.fullmatch("[0-9a-f]{64}", plan_sha256) is None:
        _refuse("runtime_admission_adoption_plan_invalid")
    path = Path(state) / COPY_MARKER
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"schema_version": 1, "plan_sha256": plan_sha256}, stream, sort_keys=True)


# --- contracts ----------------------------------------------------------------------------------------

_INTERNAL = frozenset({"sqlite_sequence", "sqlite_stat1", "sqlite_stat4"})


def _is_chroma(names: set[str]) -> bool:
    return not names.intersection(stage._TABLE_COLUMNS) and {"collections", "segments", "migrations"} <= names


def contract_census(state: Path, *, refuse: bool = True) -> dict:
    """Tables outside the R1 contracts, per database (census); refuses by default.

    An 'events' table in any carried database refuses
    (runtime_admission_adoption_legacy_table): the audit-shaped one is not
    carried and the legacy one was dropped. Any other table outside
    stage_runtime's contracts (global or scoped to this database) refuses
    (runtime_admission_adoption_unknown_table): a further pre-policy text
    table needs its own reviewed DROP_LIST entry.
    """
    state = Path(state)
    outside: dict[str, list[str]] = {}
    for path in sqlite_files(state):
        relative = path.relative_to(state).as_posix()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "events" in names and refuse:
            _refuse("runtime_admission_adoption_legacy_table:" + relative)
        if _is_chroma(names):
            continue
        known = set(stage._TABLE_COLUMNS) | set(stage._DATABASE_TABLE_COLUMNS.get(relative, {})) | _INTERNAL
        unknown = sorted(names - known)
        if unknown:
            outside[relative] = unknown
            if refuse:
                _refuse("runtime_admission_adoption_unknown_table:" + relative + ":" + unknown[0])
    return {"outside_contracts": outside}


def verify_contracts(state: Path) -> int:
    """Every database of the adopted copy passes R1's admission contract (stage_runtime._schema_contract)."""
    state = Path(state)
    checked = 0
    for path in sqlite_files(state):
        relative = path.relative_to(state).as_posix()
        with stage._database_image(path) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            try:
                stage._schema_contract(db, names, relative)
            except ValueError:
                _refuse("runtime_admission_adoption_contract:" + relative)
        checked += 1
    return checked


# --- 2a: the configuration transform ---------------------------------------------------------------

ABSENT = object()
PRODUCTION_MODELS = "/home/kza/kza/models"
PRODUCTION_LOGS = "/home/kza/releases/logs"
PRODUCTION_METRICS = "/home/kza/logs/kza-metrics.jsonl"


@dataclass(frozen=True)
class TransformRow:
    """One 2a row: ``kind`` is 'set', 'tail', 'add' or 'delete'."""

    key: tuple[str, ...]
    kind: str
    sources: tuple
    target: object = None


def transform_table(models_root: str = PRODUCTION_MODELS, logs_root: str = PRODUCTION_LOGS,
                    legacy_metrics: str = PRODUCTION_METRICS) -> tuple[TransformRow, ...]:
    """The hand-written 2a table; ``sources`` are the b0866e8 values each row accepts."""
    return (
        TransformRow(("stt", "model"), "set", ("./models/whisper-v3-turbo",), models_root + "/whisper-v3-turbo"),
        TransformRow(("llm", "model_path"), "tail", ("./models/", "models/"), models_root + "/"),
        TransformRow(("llm", "lora_path"), "set", ("./models/lora_adapters/custom_adapter",), ""),
        TransformRow(("rooms", "wake_word", "porcupine", "keyword_path"), "set", ("models/wakeword/nexa_es.ppn",),
                     models_root + "/wakeword/nexa_es.ppn"),
        TransformRow(("rooms", "wake_word", "porcupine", "model_path"), "set",
                     ("models/wakeword/porcupine_params_es.pv",), models_root + "/wakeword/porcupine_params_es.pv"),
        TransformRow(("logging", "file"), "set", ("./logs/voice_assistant.log",), logs_root + "/voice_assistant.log"),
        TransformRow(("training", "lora", "output_dir"), "set", ("./models/lora_adapters",),
                     "./data/training_outputs/lora"),
        TransformRow(("training", "nightly", "output_dir"), "set", ("./models/lora_adapters/nightly",),
                     "./data/training_outputs/nightly"),
        TransformRow(("training", "nightly", "whisper_finetune", "output_dir"), "set",
                     ("./models/lora_adapters/whisper",), "./data/training_outputs/whisper"),
        TransformRow(("observability", "metrics_path"), "set", (legacy_metrics,), "./data/obs/kza-metrics.jsonl"),
        TransformRow(("paths", "models_root"), "add", (ABSENT,), models_root),
        TransformRow(("speaker_id", "savedir"), "add", (ABSENT,), models_root + "/speaker"),
        TransformRow(("hooks", "audit_sqlite", "enabled"), "add", (ABSENT,), False),
        TransformRow(("hooks", "audit_sqlite_path"), "delete", ("./data/audit.db", ABSENT)),
        TransformRow(("code_index",), "delete", (dict, ABSENT)),
    )


def _lookup(config: dict, key: tuple[str, ...]):
    owner = config
    for part in key:
        if type(owner) is not dict or part not in owner:
            return ABSENT
        owner = owner[part]
    return owner


def _row_target(row: TransformRow, value):
    """The new value for one row given the live value; refuses runtime_admission_adoption_config_unexpected."""
    code = "runtime_admission_adoption_config_unexpected:" + ".".join(row.key)
    if row.kind == "tail":
        if type(value) is str:
            for prefix in row.sources:
                if value.startswith(prefix) and len(value) > len(prefix):
                    return row.target + value[len(prefix):]
            if value.startswith(row.target) and len(value) > len(row.target):
                return value
        _refuse(code)
    if row.kind == "delete":
        if value is ABSENT or any(value == source or (source is dict and type(value) is dict)
                                  for source in row.sources):
            return ABSENT
        _refuse(code)
    if row.kind in ("set", "add"):
        if value == row.target and type(value) is type(row.target):
            return value
        if any(value is source or (source is not ABSENT and value == source) for source in row.sources):
            return row.target
        _refuse(code)
    _refuse(code)


def apply_transform(config: dict, rows: tuple[TransformRow, ...]) -> dict:
    """The dict-level meaning of the table (the editor's post-check reference)."""
    result = json.loads(json.dumps(config))
    for row in rows:
        value = _row_target(row, _lookup(config, row.key))
        owner = result
        for part in row.key[:-1]:
            if value is ABSENT and part not in owner:
                break
            owner = owner.setdefault(part, {})
        else:
            if value is ABSENT:
                owner.pop(row.key[-1], None)
            else:
                owner[row.key[-1]] = value
    return result


_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_][A-Za-z0-9_.-]*|\"[^\"]*\"|'[^']*'):(?P<rest>(?:\s.*)?)$")


def _yaml_scalar(value) -> str:
    if value is True or value is False:
        return "true" if value else "false"
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False)
    _refuse("runtime_admission_adoption_config_value")


def _split_comment(rest: str) -> tuple[str, str]:
    quote = None
    for index, letter in enumerate(rest):
        if quote:
            if letter == quote:
                quote = None
        elif letter in "\"'":
            quote = letter
        elif letter == "#" and (index == 0 or rest[index - 1] in " \t"):
            return rest[:index].rstrip(), "  " + rest[index:]
    return rest.rstrip(), ""


def _yaml_entries(lines: list[str]) -> dict:
    """Map each block-mapping key path to (line index, indent, span end, value text)."""
    entries, stack, skip_indent = {}, [], None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if skip_indent is not None:
            if indent > skip_indent:
                continue
            skip_indent = None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        match = _KEY_LINE.match(line)
        if match is None or stripped.startswith("- "):
            stack.append((indent, None))
            continue
        key = match.group("key").strip("\"'")
        value, _comment = _split_comment(match.group("rest").strip())
        path = tuple(item[1] for item in stack) + (key,)
        if None not in path:
            entries[path] = [index, indent, None, value]
        stack.append((indent, key))
        if value[:1] in ("|", ">"):
            skip_indent = indent
    ordered = sorted(entries.items(), key=lambda item: item[1][0])
    for position, (path, entry) in enumerate(ordered):
        end = len(lines)
        for _other, later in ordered[position + 1:]:
            if later[1] <= entry[1]:
                end = later[0]
                break
        while end > entry[0] + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        entry[2] = end
    return entries


def transform_config_text(text: str, rows: tuple[TransformRow, ...]) -> str:
    """The line-preserving 2a editor, with its post-check (design v3 2a).

    Comment lines and trailing comments are kept and never matched; a row's
    scalar line is rewritten in place, new keys are inserted at the end of
    their parent mapping and deleted subtrees lose exactly their lines.
    Post-check: yaml.safe_load(new) == apply_transform(yaml.safe_load(old)).
    """
    import yaml

    old = yaml.safe_load(text)
    if type(old) is not dict:
        _refuse("runtime_admission_adoption_config_invalid")
    expected = apply_transform(old, rows)
    lines = text.split("\n")
    for row in rows:
        value = _row_target(row, _lookup(old, row.key))
        entries = _yaml_entries(lines)
        entry = entries.get(row.key)
        if value is ABSENT:
            if entry is not None:
                del lines[entry[0]:entry[2]]
            continue
        if entry is not None:
            line = lines[entry[0]]
            match = _KEY_LINE.match(line)
            _value, comment = _split_comment(match.group("rest").strip())
            lines[entry[0]] = match.group("indent") + match.group("key") + ": " + _yaml_scalar(value) + comment
            continue
        # Insert under the deepest existing parent, creating the missing levels.
        depth = len(row.key) - 1
        while depth > 0 and row.key[:depth] not in entries:
            depth -= 1
        if depth == 0:
            position, indent = len(lines), 0
            while position > 0 and not lines[position - 1].strip():
                position -= 1
        else:
            parent = entries[row.key[:depth]]
            children = [item for path, item in entries.items() if len(path) == depth + 1 and path[:depth] == row.key[:depth]]
            indent = children[0][1] if children else parent[1] + 2
            position = parent[2]
        new = []
        for level, part in enumerate(row.key[depth:]):
            prefix = " " * (indent + 2 * level)
            new.append(prefix + part + (":" if level < len(row.key) - depth - 1 else ": " + _yaml_scalar(value)))
        lines[position:position] = new
    result = "\n".join(lines)
    if yaml.safe_load(result) != expected:
        _refuse("runtime_admission_adoption_config_postcheck")
    return result


def transform_digest(rows: tuple[TransformRow, ...]) -> str:
    """Digest of the concrete table (bound by the plan and the seal)."""
    def encode(value):
        return "<absent>" if value is ABSENT else "<mapping>" if value is dict else value
    return hashlib.sha256(json.dumps([[list(row.key), row.kind, [encode(item) for item in row.sources],
                                       encode(row.target)] for row in rows], sort_keys=True).encode()).hexdigest()


def table_digests() -> dict:
    """Digests of the hand-written column, rebuilt-lease, additive and drop tables."""
    def digest(value) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"columns": digest(COLUMN_ADDITIONS), "leases": digest(REBUILT_LEASES),
            "additive": digest(ADDITIVE_TABLES),
            "drops": digest({key: {table: list(columns) for table, columns in value.items()}
                             for key, value in DROP_LIST.items()}),
            "not_carried": digest(sorted(NOT_CARRIED))}


# --- .env scan (N12d, N1) ----------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def env_census(path: Path, layout: SourceLayout) -> dict:
    """Key names whose values are path-like, with their class only; refuses legacy classes.

    KZA_AUDIT_DB present refuses (runtime_admission_adoption_audit_env); a
    path of class legacy_state, container or legacy_checkout refuses and names
    the key (runtime_admission_adoption_env_path:<key>), never the value.
    """
    classes = {}
    if not Path(path).exists():
        return {"keys": classes}
    for line in Path(path).read_text().splitlines():
        match = _ENV_LINE.match(line)
        if match is None or line.lstrip().startswith("#"):
            continue
        key, value = match.group(1), match.group(2).strip().strip("\"'")
        if key == "KZA_AUDIT_DB":
            _refuse("runtime_admission_adoption_audit_env")
        if "/" in value and value.startswith(("/", "./", "../", "~")):
            absolute = os.path.normpath(os.path.join(os.fspath(layout.container), value)) \
                if not value.startswith(("/", "~")) else value
            name = classify(absolute, layout) if absolute.startswith("/") else "other"
            classes[key] = name
            if name in ("legacy_state", "container", "legacy_checkout"):
                _refuse("runtime_admission_adoption_env_path:" + key)
    return {"keys": dict(sorted(classes.items()))}


# --- census and completeness ------------------------------------------------------------------------

def _text_columns(db: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in db.execute('PRAGMA table_info("' + table.replace('"', '""') + '")')]


def path_census(state: Path, layout: SourceLayout) -> dict:
    """Absolute-path strings counted by class in SQLite TEXT cells, JSON/.env/config files (content-free)."""
    state, counts = Path(state), Counter()
    for path in sqlite_files(state):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                for column in _text_columns(db, table):
                    query = 'SELECT "' + column + '" FROM "' + table + '" WHERE typeof("' + column + "\")='text'" \
                            ' AND instr("' + column + "\", '/')>0"
                    for (value,) in db.execute(query):
                        for match in _ABSOLUTE.findall(value):
                            counts[classify(match, layout)] += 1
    for path in sorted(state.rglob("*")):
        if path.is_file() and (path.suffix in (".json", ".jsonl", ".yaml", ".yml") or path.name == ".env"):
            for match in _ABSOLUTE.findall(path.read_text(errors="replace")):
                counts[classify(match, layout)] += 1
    return {name: counts[name] for name in PATH_CLASSES}


def _legacy_needles(layout: SourceLayout) -> tuple[str, ...]:
    return tuple(sorted({*_prefixes(layout.legacy_state), *_prefixes(layout.container)}))


def legacy_strings(state: Path, layout: SourceLayout) -> list[str]:
    """Every (file, table, column) or file still holding a legacy_state/container string (completeness)."""
    state, needles, found = Path(state), _legacy_needles(layout), []
    databases = set(sqlite_files(state))
    for path in sorted(databases):
        relative = path.relative_to(state).as_posix()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                for column in _text_columns(db, table):
                    quoted = '"' + column.replace('"', '""') + '"'
                    clause = " OR ".join("instr(CAST(" + quoted + " AS TEXT), ?)>0" for _ in needles)
                    if db.execute('SELECT 1 FROM "' + table + '" WHERE ' + clause + " LIMIT 1", needles).fetchone():
                        found.append(relative + ":" + table + ":" + column)
    for path in sorted(state.rglob("*")):
        if path.is_file() and path not in databases and not path.is_symlink():
            data = path.read_bytes()
            if any(needle.encode() in data for needle in needles):
                found.append(path.relative_to(state).as_posix())
    return found


# --- legacy_training (P1=D, 2c) ----------------------------------------------------------------------

LEGACY_TRAINING_OUTSIDE = (("lora_adapters", "data/training_outputs/lora"),
                           ("lora_adapters/nightly", "data/training_outputs/nightly"),
                           ("lora_adapters/whisper", "data/training_outputs/whisper"))
_RETIREMENT_KEYS = {"schema_version", "decision_id", "roots"}


def _root_census(root: Path) -> dict:
    """lstat-only: absent, or the regular-file count and size; links and specials refuse."""
    try:
        info = root.lstat()
    except FileNotFoundError:
        return {"exists": False, "files": 0, "bytes": 0}
    if not stat.S_ISDIR(info.st_mode):
        _refuse("runtime_admission_adoption_legacy_training_root")
    files = size = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        for name in names + filenames:
            entry = Path(directory, name).lstat()
            if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                _refuse("runtime_admission_adoption_legacy_training_special")
            if entry.st_dev != info.st_dev:
                _refuse("runtime_admission_adoption_legacy_training_mount")
            if stat.S_ISREG(entry.st_mode):
                files, size = files + 1, size + entry.st_size
    return {"exists": True, "files": files, "bytes": size}


def legacy_training_disposition(layout: SourceLayout, roots: list[str], retirement: dict | None = None,
                                *, assume_retired: bool = False) -> dict:
    """Decide how the outside legacy_training roots are relocated (P1=D, design v3 2c).

    ``roots`` are the registered legacy_training roots outside the legacy
    state. Each must be one of the three lora roots under the models root.
    Absent or empty roots are relocated directly ('no retirement needed');
    roots with files refuse unless ``retirement`` (the retire-legacy-training
    record, U3b-0e3) names them and their absence still holds.

    ``assume_retired`` (advisory dry-run only, U3b-0e3 F2) treats roots with
    files as retired by the pending P1=D step: the disposition is 'assumed'
    and the census is what retire-legacy-training would delete.

    Returns:
        ``mapping`` (old root -> state-relative new root) and the census.
    """
    base = os.path.realpath(os.path.join(layout.models_root))
    allowed = {os.path.join(base, tail): new for tail, new in LEGACY_TRAINING_OUTSIDE}
    census, mapping = {}, {}
    for root in roots:
        real = os.path.normpath(root)
        if real not in allowed and os.path.realpath(real) not in allowed:
            _refuse("runtime_admission_adoption_legacy_training_root")
        key = real if real in allowed else os.path.realpath(real)
        census[key] = _root_census(Path(key))
        mapping[root] = allowed[key]
    if any(value["files"] for value in census.values()):
        if assume_retired and retirement is None:
            return {"mapping": mapping, "retirement": "assumed", "census": census}
        # Files under a root: no retirement can have proven their absence.
        _refuse("runtime_admission_adoption_legacy_training_present")
    if retirement is None:
        return {"mapping": mapping, "retirement": "not_needed", "census": census}
    if (type(retirement) is not dict or set(retirement) != _RETIREMENT_KEYS or retirement["schema_version"] != 1
            or type(retirement["decision_id"]) is not str or not retirement["decision_id"]
            or type(retirement["roots"]) is not list or not retirement["roots"]):
        _refuse("runtime_admission_adoption_retirement_invalid")
    for item in retirement["roots"]:
        if (type(item) is not dict or item.get("root") not in allowed
                or item.get("absence") not in ("enoent", "empty")):
            _refuse("runtime_admission_adoption_retirement_invalid")
        now = _root_census(Path(item["root"]))
        if now["files"] or (item["absence"] == "enoent") == now["exists"]:
            _refuse("runtime_admission_adoption_retirement_absence")
    return {"mapping": mapping, "retirement": retirement["decision_id"], "census": census}


# --- the relocator (design v2 2d, v3 2d; 0d4a review F8) --------------------------------------------

# The explicit remap list: (table, column, how). Nothing else is rewritten.
REMAP = (
    ("erasure_targets", "descriptor", "target"),  # locators, then the id (it hashes the descriptor)
    ("erasure_steps", "target_id", "target_id"),
    ("policy_index_deletions", "target_id", "target_id"),
    ("wake_clips_v1", "target_id", "target_id"),
    ("suggestions_policy_v1", "evidence_json", "evidence"),  # EventDependency target ids
    ("utterances", "audio_path", "path"),
    ("pending_audio_deletions", "path", "path"),
    ("audio_write_intents", "path", "path"),
    ("legacy_quarantine_records_v1", "metadata_json", "json"),
)
_LOCATORS = frozenset({"path", "root", "db_path", "audio_dir"})  # = state_journal_derivation._LOCATORS
_CONTEXT_JSON = "contexts"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def target_id(participant: str, version: int, descriptor: dict) -> str:
    """The erasure target id (= ErasureJournal._register_target's key)."""
    return hashlib.sha256(json.dumps([participant, version, _canonical(descriptor)]).encode()).hexdigest()


def event_target_id(descriptor: dict) -> str:
    """The EventDependency target id (= src.analytics.event_evidence.event_target_id; a test pins it)."""
    return hashlib.sha256(_canonical(descriptor).encode()).hexdigest()


class Relocation:
    """Ordered (old prefix -> new prefix) rules from the legacy layout to the final state root."""

    def __init__(self, layout: SourceLayout, state_root: Path, training: dict, metrics_policy: str | None):
        self.state_root = os.path.normpath(os.fspath(state_root))
        rules = []
        for name in CONTAINER_LINKS:  # de-container first: container/<link> is its real target
            target = os.path.realpath(Path(layout.container) / name)
            for container in _prefixes(layout.container):  # the container itself, never through its links
                rules.append((os.path.join(container, name), target))
        for prefix in _prefixes(layout.legacy_state):
            rules.append((prefix, self.state_root))
        for old, new in training.items():
            for prefix in _prefixes(Path(old)):
                rules.append((prefix, os.path.join(self.state_root, new)))
        if metrics_policy is not None:
            base = Path(layout.metrics_root) / metrics_policy
            for prefix in _prefixes(base):
                rules.append((prefix, os.path.join(self.state_root, "data", "obs", metrics_policy)))
        self.rules = sorted({rule for rule in rules if rule[0] != rule[1]}, key=lambda rule: (-len(rule[0]), rule))

    def path(self, value: str) -> str:
        """The relocated path (rules applied to a fixpoint), or ``value`` itself (review F3).

        Matching is lexical on the stored string: only a matched prefix is
        replaced and the rest is kept byte for byte (no normpath, so '..'
        and '//' are never collapsed). A string no rule matches is returned
        unchanged; a legacy string left this way is caught by the
        completeness scan.
        """
        current = value
        for _round in range(4):
            for old, new in self.rules:
                if current == old or current.startswith(old.rstrip("/") + "/"):
                    current = new + current[len(old):]
                    break
            else:
                return current
        return current

    def string(self, value):
        """``path`` for an absolute-path string; any other value unchanged."""
        return self.path(value) if type(value) is str and value.startswith("/") else value

    def json_value(self, value):
        if isinstance(value, dict):
            return {key: self.json_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.json_value(child) for child in value]
        return self.string(value)

    def descriptor(self, descriptor: dict) -> dict:
        """A target descriptor with its locator keys relocated (= state_journal_derivation._LOCATORS)."""
        return {key: (self.string(value) if key in _LOCATORS else value) for key, value in descriptor.items()}


_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_JSON_KEY_COLON = re.compile(r"\s*:")


def rewrite_json_text(text: str, mapper) -> str:
    """Rewrite the JSON string values ``mapper`` changes, keeping every other byte (review F3).

    Object keys are never mapped. A changed value is re-encoded in its
    token's own style (ensure_ascii or not); indent, key order, spacing and
    every unchanged token stay as they were.
    """
    parts, last = [], 0
    for match in _JSON_STRING.finditer(text):
        if _JSON_KEY_COLON.match(text, match.end()):
            continue
        token = match.group(0)
        value = json.loads(token)
        new = mapper(value)
        if new == value:
            continue
        ascii_style = token == json.dumps(value) and token != json.dumps(value, ensure_ascii=False)
        parts += [text[last:match.start()], json.dumps(new, ensure_ascii=ascii_style)]
        last = match.end()
    return "".join(parts) + text[last:] if parts else text


def _privacy_databases(state: Path) -> list[Path]:
    found = []
    for path in sqlite_files(state):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='erasure_steps'").fetchone():
                found.append(path)
    return found


def _row_counts(state: Path) -> dict:
    counts = {}
    for path in sqlite_files(state):
        relative = path.relative_to(state).as_posix()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                counts[relative + ":" + table] = db.execute('SELECT COUNT(*) FROM "' + table + '"').fetchone()[0]
    return counts


def _foreign_key_violations(state: Path) -> dict:
    found = {}
    for path in sqlite_files(state):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            found[path.relative_to(state).as_posix()] = Counter(
                (row[0], row[2]) for row in db.execute("PRAGMA foreign_key_check"))
    return found


def registered_targets(state: Path) -> list[dict]:
    """The erasure targets of the single journal (more than one journal refuses, 0d4a F4)."""
    journals = _privacy_databases(state)
    if len(journals) != 1:
        _refuse("runtime_admission_adoption_journal_count:" + str(len(journals)))
    with closing(sqlite3.connect(journals[0].as_uri() + "?mode=ro", uri=True)) as db:
        return [{"id": row[0], "participant": row[1], "version": row[2], "descriptor": json.loads(row[3])}
                for row in db.execute("SELECT id, participant, version, descriptor FROM erasure_targets ORDER BY id")]


def _relocate_cell(how: str, value, rules: Relocation, ids: dict, evidence: dict):
    """The relocated value of one REMAP cell; shared by relocate() and the S3 model."""
    if type(value) is not str:
        return value
    if how == "target_id":
        return ids.get(value, value)
    if how == "path":
        return rules.string(value)
    if how == "evidence":
        new = rewrite_json_text(value, lambda item: evidence.get(item, item))
        expected = [dict(item, target_id=evidence.get(item.get("target_id"), item.get("target_id")))
                    if isinstance(item, dict) else item for item in json.loads(value)]
    else:
        new = rewrite_json_text(value, rules.string)
        expected = rules.json_value(json.loads(value))
    if json.loads(new) != expected:
        _refuse("runtime_admission_adoption_relocation_json")
    return new


def relocate(state: Path, state_root: Path, layout: SourceLayout, training: dict,
             metrics_policy: str | None) -> dict:
    """Rewrite every legacy locator of the copy into ``state_root`` (never cutting rows).

    Only the REMAP columns and the context snapshot JSON files change, and
    within them only the strings a rule matches (review F3). Target ids are
    recomputed from the relocated descriptors and every referencing row
    follows; event-evidence ids follow their events target. Afterwards row
    counts are unchanged, no legacy_state/container string remains, and no
    new foreign-key violation exists. S3 models the same change from the
    pre-relocation snapshot (relocation_model, review F1).
    """
    state = Path(state)
    rules = Relocation(layout, state_root, training, metrics_policy)
    before = _row_counts(state)
    violations = _foreign_key_violations(state)
    journals = _privacy_databases(state)
    if len(journals) != 1:
        _refuse("runtime_admission_adoption_journal_count:" + str(len(journals)))
    ids, evidence, changed = {}, {}, Counter()
    with closing(sqlite3.connect(journals[0], isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT id, participant, version, descriptor FROM erasure_targets").fetchall()
        records = {}
        for identity, participant, version, encoded in rows:
            old = json.loads(encoded)
            stage._validate_target_record({"id": identity, "participant": participant, "version": version,
                                           "descriptor": old})
            new = rules.descriptor(old)
            fresh = target_id(participant, version, new)
            record = {"id": fresh, "participant": participant, "version": version, "descriptor": new}
            stage._validate_target_record(record)
            if tuple(stage._descriptor_paths(old)):
                if fresh == identity and any(classify(value, layout) in ("legacy_state", "container")
                                             for value in stage._descriptor_paths(old)):
                    _refuse("runtime_admission_adoption_old_locator_reused")
                for value in stage._descriptor_paths(new):
                    if not _under(value, (rules.state_root,)) and classify(value, layout) in (
                            "legacy_state", "container"):
                        _refuse("runtime_admission_adoption_foreign_relocation")
            if records.setdefault(fresh, record) != record:
                _refuse("runtime_admission_adoption_target_collision")
            ids[identity] = fresh
            if participant == "events":
                evidence[event_target_id(old)] = event_target_id(new)
        if len(records) != len(rows):
            _refuse("runtime_admission_adoption_target_collision")
        db.execute("DELETE FROM erasure_targets")
        db.executemany("INSERT INTO erasure_targets VALUES (?,?,?,?)",
                       [(item["id"], item["participant"], item["version"], _canonical(item["descriptor"]))
                        for item in records.values()])
        changed["erasure_targets"] = sum(1 for old, new in ids.items() if old != new)
        for table, column, how in REMAP[1:4]:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                for old, new in ids.items():
                    if old != new:
                        changed[table] += db.execute(
                            'UPDATE "' + table + '" SET "' + column + '"=? WHERE "' + column + '"=?',
                            (new, old)).rowcount
        db.execute("COMMIT")
    for path in sqlite_files(state):
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            db.execute("BEGIN IMMEDIATE")
            for table, column, how in REMAP[4:]:
                if table not in names:
                    continue
                for rowid, value in db.execute('SELECT rowid, "' + column + '" FROM "' + table + '"').fetchall():
                    new = _relocate_cell(how, value, rules, ids, evidence)
                    if new != value:
                        db.execute('UPDATE "' + table + '" SET "' + column + '"=? WHERE rowid=?', (new, rowid))
                        changed[table] += 1
            db.execute("COMMIT")
    for path in sorted(state.rglob("*.json")):
        relative = path.relative_to(state)
        if _CONTEXT_JSON in relative.parts and path.is_file() and not path.is_symlink():
            raw = path.read_bytes().decode("utf-8")
            new = rewrite_json_text(raw, rules.string)
            if json.loads(new) != rules.json_value(json.loads(raw)):
                _refuse("runtime_admission_adoption_relocation_json")
            if new != raw:
                path.write_bytes(new.encode("utf-8"))  # in place: same inode and mode
                changed["contexts:*.json"] += 1
    if _row_counts(state) != before:
        _refuse("runtime_admission_adoption_relocation_rows")
    for relative, found in _foreign_key_violations(state).items():
        if found != violations.get(relative):  # no new dangling reference (ids follow their target)
            _refuse("runtime_admission_adoption_relocation_foreign_key:" + relative)
    remaining = legacy_strings(state, layout)
    if remaining:
        _refuse("runtime_admission_adoption_relocation_incomplete:" + remaining[0])
    return {"changed": dict(sorted(changed.items())), "targets": len(ids)}


def relocation_model(before: dict, rules: Relocation) -> dict:
    """S3's f for the relocation (review F1): ``before`` with only the REMAP columns relocated.

    The target-id and event-evidence maps are derived here from the
    snapshot's own erasure_targets rows, not taken from relocate(). Every
    non-REMAP column of every table must stay byte-equal; verify_s3 compares
    the result's row digests with the adopted copy's.
    """
    ids, evidence = {}, {}
    for tables in before.values():
        for row in (tables.get("erasure_targets") or {}).get("rows") or []:
            old = json.loads(row["descriptor"])
            new = rules.descriptor(old)
            ids[row["id"]] = target_id(row["participant"], row["version"], new)
            if row["participant"] == "events":
                evidence[event_target_id(old)] = event_target_id(new)
    model = {}
    for relative, tables in before.items():
        model[relative] = dict(tables)
        for table, column, how in REMAP:
            entry = tables.get(table)
            if entry is None:
                continue
            rows = []
            for row in entry["rows"]:
                row = dict(row)
                if how == "target":
                    descriptor = rules.descriptor(json.loads(row[column]))
                    row[column] = _canonical(descriptor)
                    row["id"] = target_id(row["participant"], row["version"], descriptor)
                else:
                    row[column] = _relocate_cell(how, row[column], rules, ids, evidence)
                rows.append(row)
            model[relative][table] = {"columns": entry["columns"], "rows": rows,
                                      "digests": Counter(_row_digest(list(row.values())) for row in rows)}
    return model


def check_target_set(state: Path, expected: list[dict], state_root: Path) -> dict:
    """The relocated targets equal R1's first-start registration (v2 2d post-condition).

    Every registered target must be one R1 registers (or a data-registered
    legacy_quarantine target rooted under ``state_root``); the only R1 targets
    that may be missing are NEW_R1_TARGETS, which R1 adds at its first start.
    """
    actual = {_canonical([item["participant"], item["version"], item["descriptor"]])
              for item in registered_targets(state)}
    wanted = {_canonical([item["participant"], item["version"], item["descriptor"]]) for item in expected}
    extra = []
    for item in sorted(actual - wanted):
        participant, version, descriptor = json.loads(item)
        if not (participant == "collectors" and descriptor.get("kind") == "legacy_quarantine"
                and _under(descriptor.get("root", ""), (os.path.normpath(os.fspath(state_root)),))):
            extra.append(participant + ":" + str(version))
    allowed = {_canonical([item["participant"], item["version"], item["descriptor"]]) for item in NEW_R1_TARGETS}
    missing = [json.loads(item)[0] + ":" + str(json.loads(item)[1]) for item in sorted(wanted - actual - allowed)]
    if extra:
        _refuse("runtime_admission_adoption_target_set:extra:" + extra[0])
    if missing:
        _refuse("runtime_admission_adoption_target_set:missing:" + missing[0])
    return {"registered": len(actual), "added_at_first_start": len(wanted - actual)}


# --- S3: expected = f(source) (N11, v1 F7) -----------------------------------------------------------

def _row_digest(values) -> str:
    return hashlib.sha256(_canonical([stage._typed(value) for value in values]).encode()).hexdigest()


# = [p.value for p in src.memory.erasure_journal.ErasureParticipant] (a test pins it).
ERASURE_PARTICIPANTS = ("producers", "captures", "contexts", "semantic_index", "events", "diagnostics",
                        "collectors", "streams", "suggestions", "voices")
_KEPT = frozenset({"erasure_targets", "erasure_steps", "conversation_erasures", "policy_index_deletions", "conversation_tombstones",
                   "facts", "voice_profiles", "retired_users", "deletion_ledger_v1", "deletion_ledger_salt_v1"})

_REMAP_TABLES = frozenset(table for table, _column, _how in REMAP)


def table_snapshot(state: Path) -> dict:
    """Columns and row digests of every table of every database; full rows of the S3-modelled tables."""
    state, snapshot = Path(state), {}
    for path in sqlite_files(state):
        relative = path.relative_to(state).as_posix()
        tables = {}
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cursor = db.execute('SELECT * FROM "' + table + '"')
                columns = [item[0] for item in cursor.description]
                rows = cursor.fetchall()
                additions = {item[1] for item in COLUMN_ADDITIONS if item[0] == relative}
                kept = table in _KEPT or table in additions or table in _REMAP_TABLES
                tables[table] = {"columns": columns, "digests": Counter(_row_digest(row) for row in rows),
                                 "rows": [dict(zip(columns, row)) for row in rows] if kept else None}
        snapshot[relative] = tables
    return snapshot


def seed_entries_of(before: dict, adoption_time: float) -> dict:
    """The ledger seed f(source) gives, per database: [(kind, subject, at)] in seed order."""
    rank = {"conversation_erased": 0, "fact_deleted": 1, "voice_profile_deleted": 2, "user_retired": 3}
    result = {}
    for relative, tables in before.items():
        entries = []
        for row in (tables.get("conversation_tombstones") or {}).get("rows") or []:
            entries.append(("conversation_erased", row["conversation_id"], float(row["deleted_at"])))
        for row in (tables.get("facts") or {}).get("rows") or []:
            if row["state"] == "deleted":
                entries.append(("fact_deleted", row["id"], float(adoption_time)))
        for row in (tables.get("voice_profiles") or {}).get("rows") or []:
            if row["deleted_at"] is not None:
                entries.append(("voice_profile_deleted", row["id"], float(row["deleted_at"])))
        for row in (tables.get("retired_users") or {}).get("rows") or []:
            entries.append(("user_retired", row["user_id"], float(row["retired_at"])))
        if entries:
            result[relative] = sorted(entries, key=lambda item: (rank[item[0]], item[2], item[1]))
    return result


def _expected_ledger(entries: list, salt: str) -> list[tuple]:
    rows, prev = [], "0" * 64
    for seq, (kind, subject, at) in enumerate(entries, start=1):
        digest = hmac.new(bytes.fromhex(salt), subject.encode(), hashlib.sha256).hexdigest()
        row = stage._ledger_row_sha256(prev, seq, kind, digest, at)
        rows.append((seq, kind, digest, at, prev, row))
        prev = row
    return rows


def verify_s3(before: dict, after: dict, *, adoption_time: float, plan_sha256: str,
              residue: dict, participants: tuple[str, ...] = ERASURE_PARTICIPANTS) -> dict:
    """after == f(before): the schema-only migration, the seed and the re-queue, nothing else.

    f applies, per database and table: (1) the not-carried items are already
    absent from ``before``; (2) ADDITIVE_TABLES appear, empty, as their
    schema marker, or as the seeded ledger and its salt; (3) COLUMN_ADDITIONS
    append their default to every row and REBUILT_LEASES are emptied;
    (4) the ledger seed; (5) the residue re-queue (``residue`` as the
    controller computed it on the seeded copy). Every other table keeps its
    columns and its exact row multiset. Any difference refuses
    (runtime_admission_adoption_s3:<database>:<table>).
    """
    seeds = seed_entries_of(before, adoption_time)
    summary = Counter()
    for relative in sorted(set(before) | set(after)):
        if relative not in before or relative not in after:
            _refuse("runtime_admission_adoption_s3:" + relative + ":database")
        old, new = before[relative], after[relative]
        additions = [item for item in COLUMN_ADDITIONS if item[0] == relative]
        leases = {item[1]: item[2] for item in REBUILT_LEASES if item[0] == relative}
        additive = ADDITIVE_TABLES.get(relative, {})
        for table in sorted(set(old) | set(new)):
            code = "runtime_admission_adoption_s3:" + relative + ":" + table
            if table not in new:
                _refuse(code)
            if table not in old:
                kind = additive.get(table)
                rows = new[table]["digests"]
                if kind == "empty" and not rows:
                    continue
                if kind == "marker" and rows == Counter([_row_digest((1,))]):
                    continue
                if kind in ("ledger", "salt"):
                    continue  # checked below with the seed
                _refuse(code)
            added = [item for item in additions if item[1] == table]
            if table in leases:
                if new[table]["columns"] != old[table]["columns"] + list(leases[table]) or new[table]["digests"]:
                    _refuse(code)
                summary["leases_emptied"] += 1
                continue
            if added:
                if new[table]["columns"] != old[table]["columns"] + [item[2] for item in added]:
                    _refuse(code)
                expected = Counter(_row_digest([*row.values(), *(item[3] for item in added)])
                                   for row in old[table]["rows"])
                if new[table]["digests"] != expected:
                    _refuse(code)
                summary["column_additions"] += len(added)
                continue
            if new[table]["columns"] != old[table]["columns"]:
                _refuse(code)
            if table in ("erasure_steps", "conversation_erasures", "policy_index_deletions"):
                continue  # the residue model below
            if table in ("deletion_ledger_v1", "deletion_ledger_salt_v1") and table in old:
                _refuse(code)  # the seed needs an absent (or empty, new) ledger
            if new[table]["digests"] != old[table]["digests"]:
                _refuse(code)
        if "deletion_ledger_v1" in new:
            salt_rows = new.get("deletion_ledger_salt_v1", {}).get("rows") or []
            entries = seeds.get(relative, [])
            if not entries:
                if new["deletion_ledger_v1"]["digests"] or salt_rows:
                    _refuse("runtime_admission_adoption_s3:" + relative + ":deletion_ledger_v1")
            else:
                if (len(salt_rows) != 1 or salt_rows[0]["id"] != 1
                        or re.fullmatch("[0-9a-f]{64}", str(salt_rows[0]["salt"])) is None):
                    _refuse("runtime_admission_adoption_s3:" + relative + ":deletion_ledger_salt_v1")
                expected = Counter(_row_digest(row) for row in _expected_ledger(entries, salt_rows[0]["salt"]))
                if new["deletion_ledger_v1"]["digests"] != expected:
                    _refuse("runtime_admission_adoption_s3:" + relative + ":deletion_ledger_v1")
                for kind, _subject, _at in entries:
                    summary["seed:" + kind] += 1
        if "erasure_steps" in old:
            summary.update(_verify_residue(relative, old, new, residue, adoption_time, plan_sha256, participants))
    return dict(sorted(summary.items()))


def _verify_residue(relative: str, old: dict, new: dict, residue: dict, adoption_time: float,
                    plan_sha256: str, participants: tuple[str, ...]) -> Counter:
    code = "runtime_admission_adoption_s3:" + relative + ":"
    summary = Counter()
    operations = {row["conversation_id"]: row["id"] for row in old["conversation_erasures"]["rows"]}
    tombstones = {row["conversation_id"] for row in old["conversation_tombstones"]["rows"]}
    created = [row for row in new["conversation_erasures"]["rows"] if row["conversation_id"] not in operations]
    expected_new = {conversation for conversation in residue.get("conversation", {}) if conversation not in operations}
    if ({row["conversation_id"] for row in created} != expected_new or len(created) != len(expected_new)
            or any(row["requested_by"] != "adoption:" + plan_sha256 or row["requested_at"] != adoption_time
                   or row["conversation_id"] not in tombstones for row in created)):
        _refuse(code + "conversation_erasures")
    kept = [row for row in new["conversation_erasures"]["rows"] if row["conversation_id"] in operations]
    if Counter(_row_digest(row.values()) for row in kept) != old["conversation_erasures"]["digests"]:
        _refuse(code + "conversation_erasures")
    summary["operations_created"] = len(created)
    targets = {}
    for row in new.get("erasure_targets", {}).get("rows") or []:
        targets[row["id"]] = row
    reopen = set()
    for conversation, occurrences in residue.get("conversation", {}).items():
        if conversation in operations:
            for _participant, governing in occurrences:
                reopen.update((operations[conversation], target) for target in governing)
    expected_steps = Counter()
    for row in old["erasure_steps"]["rows"]:
        key = (row["operation_id"], row["target_id"])
        if key in reopen and row["state"] != "pending":
            row = dict(row, state="pending", claim_token=None, cursor=None, next_attempt_at=0.0,
                       last_error=RESIDUE_REASON)
            summary["steps_reopened"] += 1
        expected_steps[_row_digest(row.values())] += 1
    registered = _registered(new)
    for row in created:
        for participant in participants:
            found = registered.get(participant, [])
            if found:
                for target, version in found:
                    expected_steps[_row_digest((row["id"], participant, target, version, "pending", 0, 0.0,
                                                None, None, None, None))] += 1
            else:
                expected_steps[_row_digest((row["id"], participant, "unregistered", 0, "pending", 0, 0.0, None,
                                            None, None, "participant_unregistered"))] += 1
    if new["erasure_steps"]["digests"] != expected_steps:
        _refuse(code + "erasure_steps")
    requeue = set(residue.get("fact", set()))
    semantic = [target for target, _version in registered.get("semantic_index", [])]
    expected_index = Counter()
    present = set()
    revisions = {row["id"]: row["revision"] for row in old["facts"]["rows"]}
    for row in old["policy_index_deletions"]["rows"]:
        present.add((row["target_id"], row["fact_id"]))
        if row["fact_id"] in requeue:
            if row["revision"] < revisions[row["fact_id"]]:
                row = dict(row, revision=revisions[row["fact_id"]], state="pending", attempts=0, next_attempt_at=0.0,
                           claim_token=None, last_error=None)
            if row["state"] != "pending":
                row = dict(row, state="pending", claim_token=None, next_attempt_at=0.0, last_error=INDEX_RECHECK)
        expected_index[_row_digest(row.values())] += 1
    for fact in requeue:
        for target in semantic:
            if (target, fact) not in present:
                expected_index[_row_digest((target, fact, revisions[fact], "pending", 0, 0.0, None, None))] += 1
    if new["policy_index_deletions"]["digests"] != expected_index:
        _refuse(code + "policy_index_deletions")
    summary["facts_requeued"] = len(requeue)
    return summary


def _registered(tables: dict) -> dict:
    found = {}
    for row in (tables.get("erasure_targets") or {}).get("rows") or []:
        found.setdefault(row["participant"], []).append((row["id"], row["version"]))
    return found


# --- fingerprints (N7) ------------------------------------------------------------------------------

def file_manifest(root: Path) -> list[list]:
    """(relative, inode, size, mtime_ns, sha256) of every regular file; a link or special refuses."""
    root = Path(root)
    entries = []
    for path in _walk_files(root):
        info = path.lstat()
        entries.append([path.relative_to(root).as_posix(), info.st_ino, info.st_size, info.st_mtime_ns,
                        _sha256_file(path)])
    return sorted(entries)


def manifest_digest(entries: list) -> str:
    """sha256 of the canonical manifest."""
    return hashlib.sha256(_canonical(entries).encode()).hexdigest()


# --- the schema-only child --------------------------------------------------------------------------

def schema_only_command(python: str, runtime_root: Path, *arguments: str) -> list[str]:
    """= src.core.entry_guard.schema_only_command (a test pins the bootstrap)."""
    return [python, "-I", "-B", "-c", SCHEMA_ONLY_BOOTSTRAP, os.fspath(runtime_root), *arguments]


def run_child(python: str, runtime_root: Path, state: Path, phase: str, *, at: float | None = None,
              state_root: Path | None = None, timeout: float = 600.0) -> dict:
    """Run one phase of R1's schema-only child on the copy; its refusal becomes ours.

    The child gets an allow-listed environment (never the controller's
    secrets) with only the capability added, and its working directory is
    the copy's parent, never a managed root.
    """
    arguments = ["--state", os.fspath(state), "--phase", phase]
    if at is not None:
        arguments += ["--at", repr(float(at))]
    if state_root is not None:
        arguments += ["--state-root", os.fspath(state_root)]
    environment = {key: os.environ[key] for key in _CHILD_ENVIRONMENT if key in os.environ}
    environment[SCHEMA_ONLY_ENV] = "1"
    # umask 0o077: what the child creates in the copy (data subdirectories, stores) is private
    # whatever the controller's own umask (0002 on the server).
    result = subprocess.run(schema_only_command(python, runtime_root, *arguments), env=environment,
                            cwd=os.fspath(Path(state).parent), capture_output=True, text=True, timeout=timeout,
                            umask=0o077)
    if result.returncode != 0:
        match = re.findall(r"refused: ([A-Za-z0-9_:.-]+)", result.stderr)
        _refuse(match[-1] if match else "runtime_admission_adoption_child_failed:" + phase)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        _refuse("runtime_admission_adoption_child_output:" + phase)
    value = json.loads(lines[-1])
    if value.get("phase", "schema") != phase or value.get("schema_only") is not True:
        _refuse("runtime_admission_adoption_child_output:" + phase)
    return value


# --- foreign targets (design v1 outside_targets, v3 plan binding; review F2) -------------------------

def descriptor_census(state: Path, layout: SourceLayout) -> dict:
    """Locator classes per erasure participant of the copy (content-free counts, review F2)."""
    census: dict[str, Counter] = {}
    for item in registered_targets(state):
        for value in stage._descriptor_paths(item["descriptor"]):
            census.setdefault(item["participant"], Counter())[classify(value, layout)] += 1
    return {participant: dict(sorted(counts.items())) for participant, counts in sorted(census.items())}


def foreign_targets(state: Path, layout: SourceLayout, metrics_policy: str | None) -> list[list[str]]:
    """(participant, class) of every target locator the relocator has no reviewed rule for.

    Relocated: legacy_state and container locators; the outside
    legacy_training roots (models class, collectors, P1=D); the metrics
    policy file (logs class, exactly that path). Anything else (a
    pre-2026-10-28 /home/kza/kza or /home/kza/app locator, code_index, other)
    is foreign.
    """
    metrics = _prefixes(Path(layout.metrics_root) / metrics_policy) if metrics_policy else ()
    found = set()
    for item in registered_targets(state):
        for value in stage._descriptor_paths(item["descriptor"]):
            kind = classify(value, layout)
            if kind in ("legacy_state", "container"):
                continue
            if (kind == "models" and item["participant"] == "collectors"
                    and item["descriptor"].get("kind") == "legacy_training"):
                continue
            if kind == "logs" and value in metrics:
                continue
            found.add((item["participant"], kind))
    return [list(item) for item in sorted(found)]


# The confirm-foreign set the plan binds. Design v1 lets the operator confirm
# per participant that a foreign locator is "copied in and relocated", but
# never defines the source of that copy (the retained /home/kza/kza tree or
# the 10-28 copy already in the slot state) nor how a relocated target that
# collides with the slot's own target merges its steps. Until a reviewed
# rule exists every confirmation refuses, so the bound set is always empty.
CONFIRM_FOREIGN_SUPPORTED: frozenset = frozenset()


def check_confirm_foreign(confirm_foreign) -> list[str]:
    """The bound confirm-foreign set; any participant without a reviewed rule refuses."""
    items = sorted(set(confirm_foreign or ()))
    if any(item not in CONFIRM_FOREIGN_SUPPORTED for item in items):
        _refuse("runtime_admission_adoption_foreign_confirm_undefined")
    return items


# --- scratch and source stability (review F5, F6) ---------------------------------------------------

SCRATCH_MARKER = ".kza-adoption-scratch.json"


def _mount_type(path: str) -> str | None:
    """The filesystem type of the mount holding ``path`` (Linux mountinfo); None when unknown."""
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    best, kind = -1, None
    for line in lines:
        fields = line.split(" ")
        if "-" not in fields:
            continue
        mount = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4])
        if _under(path, (mount,)) and len(mount) > best:
            best, kind = len(mount), fields[fields.index("-") + 1]
    return kind


def check_scratch(scratch: Path, *, work: Path, state_root: Path, layout: SourceLayout, plan_sha256: str) -> str:
    """The P2=A scratch: private, disjoint from the copy, the slot and the sources, and tmpfs or marked.

    On tmpfs (P4) it must be empty or hold only the scratch marker;
    elsewhere it must hold exactly the marker ``{"schema_version": 1,
    "plan_sha256": <plan>}`` (tests, non-Linux). Returns 'tmpfs' or 'marker'.
    """
    code = "runtime_admission_adoption_scratch"
    scratch = Path(scratch)
    try:
        info = scratch.lstat()
    except OSError:
        _refuse(code)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        _refuse(code)
    real = os.path.realpath(scratch)
    for other in (work, state_root, *source_roots(layout)):
        resolved = os.path.realpath(other)
        if _under(real, (resolved,)) or _under(resolved, (real,)):
            _refuse(code + "_overlap")
    entries = sorted(os.listdir(scratch))
    if _mount_type(real) == "tmpfs" and entries in ([], [SCRATCH_MARKER]):
        return "tmpfs"
    if entries != [SCRATCH_MARKER]:
        _refuse(code + "_unmarked")
    try:
        descriptor = os.open(scratch / SCRATCH_MARKER, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor) as stream:
            marker = json.load(stream)
    except (OSError, ValueError):
        _refuse(code + "_unmarked")
    if marker != {"schema_version": 1, "plan_sha256": plan_sha256}:
        _refuse(code + "_unmarked")
    return "marker"


def write_scratch_marker(scratch: Path, plan_sha256: str) -> None:
    """Mark a non-tmpfs scratch for one plan (tests, non-Linux)."""
    if type(plan_sha256) is not str or re.fullmatch("[0-9a-f]{64}", plan_sha256) is None:
        _refuse("runtime_admission_adoption_plan_invalid")
    descriptor = os.open(Path(scratch) / SCRATCH_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"schema_version": 1, "plan_sha256": plan_sha256}, stream, sort_keys=True)


def source_stat_map(layout: SourceLayout, metrics_path: str) -> dict:
    """(inode, size, mtime_ns) of every source file the manifest covers, re-walked now."""
    found = {}
    root = Path(layout.legacy_state)
    for path in _walk_files(root):
        info = path.lstat()
        found[path.relative_to(root).as_posix()] = [info.st_ino, info.st_size, info.st_mtime_ns]
    for item in metrics_store_set(layout, metrics_path)["members"]:
        info = (Path(layout.metrics_root) / item).lstat()
        found["metrics/" + item] = [info.st_ino, info.st_size, info.st_mtime_ns]
    return found


def source_manifest(layout: SourceLayout, metrics_path: str) -> list:
    """The full-file source manifest recomputed now, in adopt_copy's order (U3b-0e3 commit and abort).

    Every regular file of the legacy state (sorted), then the metrics store
    set members as ``metrics/<name>``; each (relative, inode, size,
    mtime_ns, sha256). Read-only: lstat and O_RDONLY|O_NOFOLLOW only.
    """
    entries = file_manifest(layout.legacy_state)
    root = Path(layout.metrics_root)
    for item in metrics_store_set(layout, metrics_path)["members"]:
        info = (root / item).lstat()
        entries.append(["metrics/" + item, info.st_ino, info.st_size, info.st_mtime_ns, _sha256_file(root / item)])
    return entries


def verify_source_unchanged(layout: SourceLayout, metrics_path: str, manifest: list,
                            moving: Counter | None = None) -> None:
    """Re-walk the sources after the copy: any changed, added or removed file refuses (review F6).

    The advisory dry-run passes ``moving`` and gets the count of differing
    files instead (U3b-0e3 F2: the service may run during it).
    """
    expected = {entry[0]: list(entry[1:4]) for entry in manifest}
    now = source_stat_map(layout, metrics_path)
    if now != expected:
        if moving is None:
            _refuse("runtime_admission_adoption_source_changed")
        moving["after_copy"] = max(moving["after_copy"], sum(
            1 for name in set(now) | set(expected) if now.get(name) != expected.get(name)))


# --- the library pipeline ---------------------------------------------------------------------------

def _legacy_metrics_path(work: Path) -> str:
    import yaml

    config = yaml.safe_load((Path(work) / "config" / "settings.yaml").read_text()) or {}
    return ((config.get("observability") or {}).get("metrics_path")) or PRODUCTION_METRICS


def adopt_copy(layout: SourceLayout, *, work: Path, scratch: Path, state_root: Path, plan_sha256: str,
               adoption_time: float, python: str, runtime_root: Path, retirement: dict | None = None,
               rows: tuple[TransformRow, ...] | None = None, confirm_foreign=(), private: dict | None = None,
               advisory: bool = False) -> dict:
    """Turn a byte copy of the production state into the first managed state, in ``work``.

    ``work`` is the adoption copy (it gets the marker), ``scratch`` the
    P2=A scratch (tmpfs or marked, disjoint from everything else) for the
    DROP_LIST databases, ``state_root`` where the state will be served from
    (descriptors name it). The source is only read, under a SourceGuard, and
    re-walked after the copy; the R1 steps run in the schema-only child.
    Returns a content-free report; any refusal raises AdoptionRefusal with
    the report so far as ``refusal.report`` (the dry-run census). When
    ``private`` is a dict it receives what the operator commands keep
    privately under admission/ and never print: the full source manifest
    (``source_manifest``) and the legacy metrics path (``metrics_path``).

    ``advisory`` (the dry-run only, U3b-0e3 F2) delivers the whole census:
    a moving source is counted (``source_moving``) and the copied bytes are
    used; roots with LoRA files are assumed retired; the soft refusals
    (contracts, foreign targets, relocation, children, residue, S3, target
    set) are collected in ``refusals`` and the pipeline continues. Prepare
    and commit never pass it: every one of those refuses there.
    """
    work, scratch = Path(work), Path(scratch)
    rows = transform_table() if rows is None else rows
    report: dict = {"schema_version": SCHEMA_VERSION, "plan_sha256": plan_sha256}
    private = {} if private is None else private
    try:
        _adopt(report, layout, work, scratch, state_root, plan_sha256, adoption_time, python, runtime_root,
               retirement, rows, confirm_foreign, private, advisory)
    except AdoptionRefusal as refusal:
        refusal.report = report
        raise
    except sqlite3.DatabaseError:  # a corrupt copy is a refusal, never a traceback (review R2-1)
        refusal = AdoptionRefusal("runtime_admission_adoption_sqlite_corrupt")
        refusal.report = report
        raise refusal from None
    return report


def residue_census(residue: dict) -> dict:
    """Content-free counts of resurrection_residue; the 'unmarkable' families refuse at P3=A (design v3 4)."""
    conversation = residue["conversation"]
    return {"conversation": len(conversation),
            "conversation_unmarkable": sum(1 for occurrences in conversation.values()
                                           if any(not governing for _participant, governing in occurrences)),
            "fact": len(residue["fact"]), "voice_profile": len(residue["voice_profile"]),
            "voice_consent": residue["voice_consent"]}


def _adopt(report, layout, work, scratch, state_root, plan_sha256, adoption_time, python, runtime_root,
           retirement, rows, confirm_foreign, private, advisory=False) -> None:
    moving = Counter() if advisory else None

    def soft(call):
        """Run one step: refuse (prepare/commit) or record the refusal and go on (advisory)."""
        try:
            return call()
        except AdoptionRefusal as refusal:
            if not advisory:
                raise
            report.setdefault("refusals", []).append(str(refusal))
            return None

    report["confirm_foreign"] = check_confirm_foreign(confirm_foreign)
    with SourceGuard(*source_roots(layout)) as guard:
        report["layout"] = read_source_layout(layout)
        report["scratch"] = check_scratch(scratch, work=work, state_root=state_root, layout=layout,
                                          plan_sha256=plan_sha256)
        work.mkdir(mode=0o700)
        copied = copy_source(layout, work, scratch, moving)
        metrics_path = _legacy_metrics_path(work)
        metrics = copy_metrics_store(layout, metrics_path, work, moving)
        manifest = copied["manifest"] + metrics["manifest"]
        verify_source_unchanged(layout, metrics_path, manifest, moving)
        private["source_manifest"], private["metrics_path"] = manifest, metrics_path
        report["source_manifest_sha256"] = manifest_digest(manifest)
        report["source_files"] = len(manifest)
        report["copy"] = copied["counts"]
        report["metrics"] = {key: metrics[key] for key in ("members", "outside_pre_policy")}
        if moving is not None:
            drop_torn_copies((work, scratch), moving)
        report["recovery"] = recover_copies(work)
        report["drops"] = drop_and_vacuum(scratch, work)
        write_copy_marker(work, plan_sha256)
        soft(lambda: contract_census(work))
        report["contracts_outside"] = contract_census(work, refuse=False)["outside_contracts"]
        settings_path = work / "config" / "settings.yaml"
        settings_path.write_text(transform_config_text(settings_path.read_text(), rows))
        report["transform_sha256"] = transform_digest(rows)
        report["tables"] = table_digests()
        report["env"] = env_census(work / ".env", layout)
        report["descriptor_classes"] = descriptor_census(work, layout)
        foreign = foreign_targets(work, layout, metrics["policy_name"])
        report["foreign_targets"] = foreign
        if foreign:
            soft(lambda: _refuse("runtime_admission_adoption_foreign_target:" + ":".join(foreign[0])))
        outside = [item["descriptor"]["root"] for item in registered_targets(work)
                   if item["participant"] == "collectors" and item["descriptor"].get("kind") == "legacy_training"
                   and classify(item["descriptor"]["root"], layout) not in ("legacy_state", "container")]
        training = legacy_training_disposition(layout, outside, retirement, assume_retired=advisory)
        report["legacy_training"] = {"retirement": training["retirement"], "roots": len(outside)}
        if training["retirement"] == "assumed":
            report["legacy_training"]["would_retire"] = {
                "files": sum(item["files"] for item in training["census"].values()),
                "bytes": sum(item["bytes"] for item in training["census"].values())}
        report["census_before"] = path_census(work, layout)
        before = table_snapshot(work)  # S3's source: f includes the relocation (review F1)
        report["relocation"] = soft(lambda: relocate(work, state_root, layout, training["mapping"],
                                                     metrics["policy_name"]))
        report["census_after"] = path_census(work, layout)
        expected = soft(lambda: relocation_model(before, Relocation(layout, state_root, training["mapping"],
                                                                    metrics["policy_name"])))
        report["schema"] = soft(lambda: run_child(python, runtime_root, work, "schema"))
        seeded = soft(lambda: run_child(python, runtime_root, work, "seed", at=adoption_time))
        report["seed"] = None if seeded is None else seeded["seeded"]
        residue = launcher.resurrection_residue(work, descriptor_root=Path(state_root))
        report["residue_census"] = residue_census(residue)
        requeued = soft(lambda: run_child(python, runtime_root, work, "residue", at=adoption_time,
                                          state_root=state_root))
        report["residue"] = None if requeued is None else requeued["requeued"]
        report["s3"] = None if expected is None else soft(lambda: verify_s3(
            expected, table_snapshot(work), adoption_time=adoption_time, plan_sha256=plan_sha256, residue=residue))
        report["contracts"] = soft(lambda: verify_contracts(work))
        targets = soft(lambda: run_child(python, runtime_root, work, "targets", state_root=state_root))
        report["targets"] = None if targets is None else soft(
            lambda: check_target_set(work, targets["targets"], state_root))
        finish_copies(work)
        verify_source_unchanged(layout, metrics_path, manifest, moving)
        if moving is not None:
            report["source_moving"] = dict(sorted(moving.items()))
        report["guard"] = {"refused": guard.refused, "connects_observed": guard.connects}
    report["prepared_manifest_sha256"] = manifest_digest(file_manifest(work))
