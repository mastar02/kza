"""Private snapshots, schema initialization and content-free migration accounting."""

import ctypes
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[2]
VOICE_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
FACT_MODEL = "BAAI/bge-m3"
VERSIONS = {"memory": 1, "ambient": 1, "household": 3, "auth": 0}
LOCATIONS = {
    "memory": "memory/memory.db", "ambient": "memory/ambient.db",
    "household": "household/household.db", "auth": "auth/auth.db",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
JSON_FILES = {"users.json", "users_accounts_v1.json", "alerts.json"}
LEDGER_PAIR = ("deletion_ledger_v1", "deletion_ledger_salt_v1")


class MigrationError(Exception):
    """A failure whose message contains no source data."""


def require(condition: bool, error: str) -> None:
    if not condition:
        raise MigrationError(error)


def identifier(value: object) -> str:
    require(isinstance(value, str) and bool(value) and value == value.strip()
            and not any(ord(char) < 32 or ord(char) == 127 for char in value),
            "invalid source identifier")
    return value


def finite(value: object) -> float:
    require(type(value) in (int, float) and math.isfinite(value), "invalid source number")
    return float(value)


def millis(value: object) -> int:
    return round(finite(value) * 1000)


def timestamp(value: object) -> int:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        require(parsed.utcoffset() is not None, "source timestamp has no timezone")
        value = parsed.timestamp()
    return millis(value)


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def decode(value: str) -> object:
    def unique(pairs: list[tuple]) -> dict:
        result = dict(pairs)
        require(len(pairs) == len(result), "duplicate JSON key")
        return result

    return json.loads(value, object_pairs_hook=unique)


def private_open(path: Path) -> BinaryIO:
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def write_json(path: Path, data: object) -> None:
    with private_open(path) as stream:
        stream.write((encode(data) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def files_under(root: Path) -> list[Path]:
    def walk(directory: Path) -> list[Path]:
        files = []
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | getattr(os, "O_NOATIME", 0)
        fd = os.open(directory, flags)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    child = directory / entry.name
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        files.extend(walk(child))
                    else:
                        require(stat.S_ISREG(mode), "source contains a symlink or special file")
                        files.append(child.relative_to(root))
        finally:
            os.close(fd)
        return files

    return sorted(walk(root))


def sqlite_file(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def copy_input(path: Path) -> bool:
    return (sqlite_file(path) or path.name in JSON_FILES or any(
        path.name.endswith(suffix) and sqlite_file(Path(path.name.removesuffix(suffix)))
        for suffix in ("-wal", "-journal")
    ))


def clone_tree(source: Path, clone: Path) -> None:
    # Darwin lacks O_NOATIME. Its native copy-on-write clone preserves timestamps
    # on the original files and directories, including during enumeration.
    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(source), os.fsencode(clone), 0x0001):
        raise MigrationError("source and staging must support a same-volume APFS clone")


def checksum(root: Path, work: Path, *, keep_snapshot: bool = False) -> dict[Path, tuple]:
    """Hash all source bytes while keeping original file and directory timestamps."""
    clone = sys.platform == "darwin"
    snapshot = None
    if clone:
        clone_tree(root, work)
        reading = work
    else:
        require(hasattr(os, "O_NOATIME"), "this platform cannot guarantee unchanged source timestamps")
        reading = root
        if keep_snapshot:
            snapshot = work
            snapshot.mkdir(mode=0o700)
    values = {}
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_NOATIME", 0)
    for relative in files_under(reading):
        original = root / relative
        identity = file_identity(original.lstat())
        with os.fdopen(os.open(reading / relative, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == identity[3],
                    "source changed while reading")
            digest = hashlib.sha256()
            if snapshot is not None and copy_input(relative):
                target = snapshot / relative
                private_parents(target.parent, snapshot)
                with private_open(target) as copy:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        copy.write(chunk)
            else:
                digest = hashlib.file_digest(stream, "sha256")
            require(identity == file_identity(original.lstat()), "source changed while reading")
        values[relative] = digest.hexdigest(), identity
    if clone and keep_snapshot:
        work.chmod(0o700)
        for relative in values:
            path = work / relative
            for parent in path.parents:
                if parent == work:
                    break
                parent.chmod(0o700)
            if copy_input(relative):
                path.chmod(0o600)
            else:
                # SQLite builds a fresh SHM on the clone from committed WAL data.
                path.unlink()
    return values


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_atime_ns)


def private_parents(path: Path, root: Path) -> None:
    for directory in reversed([path, *path.parents][:len(path.relative_to(root).parts)]):
        directory.mkdir(mode=0o700, exist_ok=True)


def quoted(name: str) -> str:
    require(bool(re.fullmatch(r"[a-z_][a-z_0-9]*", name)), "invalid schema identifier")
    return '"' + name + '"'


def _ledger_populated(db: sqlite3.Connection) -> bool:
    return any(db.execute(f"SELECT count(*) FROM {quoted(name)}").fetchone()[0] > 0
               for name in LEDGER_PAIR)


@dataclass
class Counts:
    source: int = 0
    migrated: int = 0
    filtered_tombstoned: int = 0
    filtered_retired: int = 0
    discarded: int = 0
    archived: int = 0


@dataclass
class Catalog:
    """Resolve legacy tables across a private copy of the input tree."""

    tables: dict[str, sqlite3.Connection] = field(default_factory=dict)
    counts: dict[str, Counts] = field(default_factory=dict)

    def add(self, db: sqlite3.Connection, supported: set[str]) -> None:
        names = {name for name, in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")
                 if name in supported}
        pair = names.intersection(LEDGER_PAIR)
        duplicate = "duplicate supported source table"
        require(self.tables.keys().isdisjoint(names - pair), duplicate)

        if pair and not self.tables.keys().isdisjoint(LEDGER_PAIR):
            ledger, salt = LEDGER_PAIR
            existing = self.tables.get(ledger)
            if len(pair) != len(LEDGER_PAIR) or existing is None or self.tables.get(salt) is not existing:
                raise MigrationError(duplicate)
            if _ledger_populated(db):
                require(not _ledger_populated(existing), duplicate)
            else:
                names -= pair

        self.tables.update(dict.fromkeys(names, db))

    def rows(self, table: str) -> list[dict]:
        self.counts.setdefault(table, Counts())
        if table not in self.tables:
            return []
        db = self.tables[table]
        result = [dict(row) for row in db.execute(f"SELECT * FROM {quoted(table)}")]
        self.counts[table].source = len(result)
        return result

    def count(self, table: str, field_name: str, amount: int = 1) -> None:
        counts = self.counts.setdefault(table, Counts())
        setattr(counts, field_name, getattr(counts, field_name) + amount)

    def omit(self, table: str, rows: list[dict]) -> None:
        self.count(table, "discarded", len(rows))


def check(db: sqlite3.Connection, *, foreign_keys: bool = True) -> None:
    require([row[0] for row in db.execute("PRAGMA integrity_check")] == ["ok"],
            "database integrity check failed")
    if foreign_keys:
        require(db.execute("PRAGMA foreign_key_check").fetchone() is None,
                "database foreign key check failed")


def start_source(path: Path) -> sqlite3.Connection:
    if path.stat().st_size == 0:
        raise MigrationError("empty SQLite source")
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def memory_ddl() -> str:
    source = (ROOT / "crates/nexa-memoryd/src/db.rs").read_text()
    ddl = source.split('"CREATE VIRTUAL TABLE fact_vec', 1)[1].split(';"', 1)[0]
    return ("CREATE VIRTUAL TABLE fact_vec" + ddl + ";").format(
        FACT_DIMENSIONS=1024, VOICE_DIMENSIONS=192)


def target_schema(name: str) -> str:
    if name in {"memory", "ambient"}:
        sql = (ROOT / f"crates/nexa-memoryd/src/{name}.sql").read_text()
        return sql + memory_ddl() if name == "memory" else sql.replace("ambient.", "")
    if name == "household":
        schema = ROOT / "crates/nexa-household/src/schema"
        migrations = ("001_scheduling.sql", "002_household.sql", "003_routine_schedules.sql")
        return "\n".join((schema / file).read_text() for file in migrations)
    rust = (ROOT / "crates/nexa-web/src/auth.rs").read_text()
    prefix = "CREATE TABLE IF NOT EXISTS auth_meta"
    return prefix + rust.split(prefix, 1)[1].split(';"', 1)[0] + ";"


def load_vec(db: sqlite3.Connection) -> None:
    path = os.environ.get("NEXA_SQLITE_VEC")
    require(bool(path), "set NEXA_SQLITE_VEC to the sqlite-vec 0.1.7 loadable extension")
    db.enable_load_extension(True)
    try:
        db.load_extension(path, entrypoint="sqlite3_vec_init")
    finally:
        db.enable_load_extension(False)
    version = db.execute("SELECT vec_version()").fetchone()[0]
    require(version in {"v0.1.7", "0.1.7"}, "sqlite-vec version must be 0.1.7")


def start_target(root: Path, name: str) -> sqlite3.Connection:
    path = root / LOCATIONS[name]
    private_parents(path.parent, root)
    with private_open(path):
        pass
    db = sqlite3.connect(path)
    try:
        if name == "memory":
            load_vec(db)
        require(db.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal",
                "cannot enable WAL")
        db.executescript(f"""
            PRAGMA foreign_keys=ON;
            PRAGMA secure_delete=ON;
            PRAGMA synchronous=FULL;
            PRAGMA temp_store=MEMORY;
            PRAGMA user_version={VERSIONS[name]};
        """)
        db.executescript(target_schema(name))
        return db
    except BaseException:
        db.close()
        raise


def validate_target(db: sqlite3.Connection, name: str) -> None:
    db.commit()
    check(db)
    for pragma, value in (("journal_mode", "wal"), ("user_version", VERSIONS[name]),
                          ("foreign_keys", 1), ("secure_delete", 1), ("synchronous", 2)):
        require(db.execute(f"PRAGMA {pragma}").fetchone()[0] == value,
                "destination pragmas differ from Nexa schema")
    require(db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0,
            "destination WAL checkpoint failed")
