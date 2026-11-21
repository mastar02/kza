"""Review a sealed P6 bytecode archive and rehearse the Epoch 2 schema preflight."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile

from scripts import adoption, stage_runtime as stage


SCHEMA_EPOCH = 2
# Epoch 2 pin from tests/unit/tools/test_schema_contract_pin.py.
SCHEMA_DIGEST = "d0b616eaf5ce55c8995913a0063808235c336945a4a97ca8adee90ad65d71dc5"
LEGACY_SLOT = "release_20261029T185008966364Z_e9669ec"
LEGACY_COMMIT = "e9669ec1edb101ca6b190ef8df09314e1ac63562"
FROZEN_SLOT = "release_20261028T145442Z_b0866e8"
L4_PHRASE = "revivir datos borrados"
DECLARATIONS = ("requirements.txt", "requirements-test.txt", "pyproject.toml", "config/settings.yaml")
PRODUCTION_ROOT = Path("/home/kza")

# Loaded with -I -B from the verified archive's temporary runtime.
_PROBE = """
import hashlib
import json
from pathlib import Path
import sys

runtime, root, slot, models = sys.argv[1:]
sys.path.insert(0, runtime)
from scripts import stage_runtime as stage
root = Path(root)
launcher = stage._launcher_module()
commands = stage._adoption_commands_module()
logs = Path(models).parents[1] / "logs"
paths = commands.ProductionPaths(models, str(logs), str(logs / "kza-metrics.jsonl"))
layout, links = commands.source_layout(root, slot, paths)
commands.current_link(root, slot)
_, _, commit = launcher._verify_legacy_slot(root, slot, launcher.LEGACY_COMMITS)
print(json.dumps({
    "schema_contract": stage.SCHEMA_CONTRACT_EPOCH,
    "schema_contract_sha256": hashlib.sha256(stage.schema_contract_document()).hexdigest(),
    "controller_epoch": stage.CONTROLLER_EPOCH,
    "controller_revision": stage.CONTROLLER_REVISION,
    "launcher_contract": stage.LAUNCHER_CONTRACT,
    "legacy_commits": list(launcher.LEGACY_COMMITS),
    "legacy_commit": commit,
    "adoption_source_commit": launcher.LEGACY_CONTAINER_RELEASE,
    "frozen_state_slot": layout.legacy_state.parent.name,
    "legacy_links": len(links["readlinks"]),
    "rollback_phrase": launcher.UNINSTALL_PHRASE,
}))
"""

# Pre-Epoch-2 reminders and an already-sent HA action. Schema-only preserves both.
_LEGACY_SQL = """
CREATE TABLE reminders (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
    trigger_at REAL NOT NULL, created_at REAL NOT NULL, recurrence TEXT,
    recurrence_end REAL, ha_actions TEXT, state TEXT NOT NULL DEFAULT 'active',
    last_fired_at REAL, fire_count INTEGER NOT NULL DEFAULT 0,
    scope TEXT NOT NULL DEFAULT 'private' CHECK (scope IN ('private', 'shared'))
);
CREATE TABLE reminder_home_actions (
    request_id TEXT PRIMARY KEY, body TEXT NOT NULL, status TEXT NOT NULL, reason TEXT
);
INSERT INTO reminders (id, user_id, text, trigger_at, created_at)
VALUES ('r1', 'synthetic-user', 'Synthetic reminder', 500, 20);
INSERT INTO reminder_home_actions VALUES ('a1', '{}', 'sent', NULL);
"""


@dataclass(frozen=True)
class ReviewInputs:
    """Inputs for a single candidate review."""

    repo: Path
    archive: Path
    commit: str
    archive_sha256: str
    runtime_lock: Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _review_path(path: Path) -> Path:
    if path.absolute().is_relative_to(PRODUCTION_ROOT):
        raise ValueError("review inputs must be outside the production tree")
    resolved = path.resolve()
    if resolved.is_relative_to(PRODUCTION_ROOT):
        raise ValueError("review inputs must be outside the production tree")
    return resolved


def _git_blob(repo: Path, commit: str, path: str) -> bytes:
    entry = subprocess.run(
        ["git", "--no-replace-objects", "ls-tree", "-z", commit, "--", path],
        cwd=repo, capture_output=True, timeout=30,
    )
    if entry.returncode or not entry.stdout.startswith((b"100644 blob ", b"100755 blob ")):
        raise ValueError(f"missing or non-regular release declaration at the candidate commit: {path}")
    result = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "blob", f"{commit}:{path}"],
        cwd=repo, capture_output=True, timeout=30,
    )
    if result.returncode or not result.stdout.strip():
        raise ValueError(f"missing or empty release declaration at the candidate commit: {path}")
    return result.stdout


def _check_git_commit(repo: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "-t", commit],
        cwd=repo, capture_output=True, timeout=30,
    )
    if result.returncode or result.stdout != b"commit\n":
        raise ValueError("candidate SHA is not a Git commit in the review checkout")


def _read_lock(path: Path) -> str:
    path = _review_path(path)
    if not path.is_file():
        raise ValueError("runtime dependency lock is missing or is not a regular file")
    data = path.read_bytes()
    if not any(line.strip() and not line.lstrip().startswith(b"#") for line in data.splitlines()):
        raise ValueError("runtime dependency lock is empty")
    return _sha256(data)


def _archive_snapshot(archive: Path, snapshot: Path, expected: str) -> int:
    archive = _review_path(archive)
    if not archive.is_file():
        raise ValueError("runtime archive is missing or is not a regular file")
    with archive.open("rb") as source, snapshot.open("xb") as target:
        shutil.copyfileobj(source, target)
    with snapshot.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("runtime archive SHA-256 differs from the sealed hash")
    return snapshot.stat().st_size


def _read_bundle(snapshot: Path, commit: str) -> tuple[dict, dict[str, bytes]]:
    manifest, files = stage._read_archive(snapshot)
    if manifest["git_commit"] != commit:
        raise ValueError("runtime archive commit differs from the candidate SHA")
    if type(manifest.get("schema_contract")) is not int or manifest["schema_contract"] != SCHEMA_EPOCH:
        raise ValueError("runtime archive must declare schema contract epoch 2")
    if manifest.get("entrypoint") != ["python", "-m", "src.main"]:
        raise ValueError("runtime archive entrypoint differs from src.main")
    magic = bytes.fromhex(manifest["python_magic"])
    for name, data in files.items():
        suffix = Path(name).suffix.lower()
        if Path(name).name.lower().endswith(".py"):
            raise ValueError("runtime archive contains Python source")
        # build_runtime uses unchecked, hash-based PYC for the source-free runtime.
        if suffix == ".pyc" and (len(data) <= 16 or data[:8] != magic + b"\x01\x00\x00\x00"):
            raise ValueError(f"invalid CPython 3.13 bytecode header: {name}")
    stage.controller_triple(manifest, files)
    return manifest, files


def _run_json(command: list[str], scratch: Path, env: dict[str, str], phase: str) -> dict:
    result = subprocess.run(command, cwd=scratch, env=env, capture_output=True, timeout=60, umask=0o077)
    if result.returncode:
        raise ValueError(f"compiled preflight failed: {phase} (exit {result.returncode})")
    value = json.loads(result.stdout)
    if type(value) is not dict:
        raise ValueError(f"compiled preflight returned invalid evidence: {phase}")
    return value


def _check_controller(compiled: dict, manifest: dict) -> None:
    if type(compiled.get("schema_contract")) is not int or compiled["schema_contract"] != SCHEMA_EPOCH:
        raise ValueError("compiled controller does not implement schema contract epoch 2")
    if compiled.get("schema_contract_sha256") != SCHEMA_DIGEST:
        raise ValueError("compiled schema contract digest differs from the Epoch 2 pin")
    for key in ("controller_epoch", "controller_revision", "launcher_contract"):
        if type(manifest.get(key)) is not int or manifest[key] != compiled.get(key):
            raise ValueError(f"compiled controller disagrees with release.json: {key}")
    if (compiled.get("legacy_commits") != [LEGACY_COMMIT[:7]]
            or compiled.get("legacy_commit") != LEGACY_COMMIT
            or compiled.get("adoption_source_commit") != LEGACY_COMMIT
            or compiled.get("frozen_state_slot") != FROZEN_SLOT
            or compiled.get("legacy_links") != 5
            or compiled.get("rollback_phrase") != L4_PHRASE):
        raise ValueError("compiled controller disagrees with the Q4 legacy source or rollback target")


def _sqlite_rows(db: sqlite3.Connection) -> tuple[list, list]:
    if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("synthetic reminders failed SQLite quick_check")
    return db.execute("SELECT * FROM reminders").fetchall(), db.execute(
        "SELECT * FROM reminder_home_actions"
    ).fetchall()


def _synthetic_preflight(scratch: Path, manifest: dict, files: dict[str, bytes], archive_sha256: str) -> dict:
    runtime = scratch / "runtime"
    for name, data in files.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    releases = scratch / "releases"
    frozen = releases / "slots" / FROZEN_SLOT / "state"
    (frozen / "data").mkdir(parents=True)
    (frozen / "config").mkdir()
    (frozen / "logs").mkdir()
    (frozen / ".env").touch()
    (frozen / "config/settings.yaml").write_text("{}\n", encoding="utf-8")
    original = frozen / "data/reminders.db"
    with closing(sqlite3.connect(original)) as db:
        db.executescript(_LEGACY_SQL)
        reminders, actions = _sqlite_rows(db)
    frozen_bytes = original.read_bytes()

    container = releases / "slots" / LEGACY_SLOT
    (container / "runtime").mkdir(parents=True)
    (container / "runtime/release.json").write_text(
        json.dumps({"git_commit": LEGACY_COMMIT, "files": {}}), encoding="utf-8",
    )
    (container / "state").mkdir()
    for name in ("config", "data", "logs", ".env"):
        (container / "state" / name).symlink_to(frozen / name, target_is_directory=name != ".env")
    models = scratch / "legacy-checkout/models"
    models.mkdir(parents=True)
    (container / "state/models").symlink_to(models, target_is_directory=True)
    (releases / "current").symlink_to(container / "runtime", target_is_directory=True)
    compiled = _run_json(
        [sys.executable, "-I", "-B", "-c", _PROBE, str(runtime), str(releases), LEGACY_SLOT, str(models)],
        scratch, {}, "controller and L4 links",
    )
    _check_controller(compiled, manifest)

    copy = scratch / "adoption-copy"
    shutil.copytree(container / "state", copy)
    adoption.write_copy_marker(copy, archive_sha256)
    command = adoption.schema_only_command(sys.executable, runtime, "--state", str(copy), "--phase", "schema")
    result = _run_json(command, scratch, {adoption.SCHEMA_ONLY_ENV: "1"}, "schema-only")
    if result.get("schema_only") is not True or result.get("phase") != "schema":
        raise ValueError("compiled schema-only phase did not complete")

    migrated = copy / "data/reminders.db"
    with closing(sqlite3.connect(migrated.as_uri() + "?mode=ro", uri=True)) as db:
        migrated_reminders, migrated_actions = _sqlite_rows(db)
        columns = [row[1] for row in db.execute("PRAGMA table_info(reminders)")]
        timers = db.execute("SELECT COUNT(*) FROM timers").fetchone()[0]
        preferences = db.execute("SELECT COUNT(*) FROM alert_preferences").fetchone()[0]
    if (columns[-1:] != ["delivery_room"] or migrated_reminders != [row + (None,) for row in reminders]
            or migrated_actions != actions or timers or preferences):
        raise ValueError("Epoch 2 synthetic migration changed protected reminders or actions")
    if (container / "state/data/reminders.db").read_bytes() != frozen_bytes:
        raise ValueError("synthetic frozen rollback database changed")

    return {
        "schema_only": "passed",
        "reminders_preserved": len(reminders),
        "home_actions_preserved": len(actions),
        "timers": timers,
        "alert_preferences": preferences,
        "legacy_current_verified": True,
        "legacy_links_verified": compiled["legacy_links"],
        "frozen_database_unchanged": True,
    }


def rollback_plan() -> dict:
    """Return the fixed Q4 rollback and Q5 retirement operator plan."""
    controller = [
        "<runtime-python>", "-I", "-B", "<installed-controller>/scripts/stage_runtime.pyc",
        "--root", "<releases-root>", "--offline",
    ]
    retire = controller + [
        "retire-legacy-state", "--decision", "<P7-decision-id>", "--container-slot", LEGACY_SLOT,
    ]
    return {
        "method": "L4",
        "legacy_slot": LEGACY_SLOT,
        "legacy_commit": LEGACY_COMMIT,
        "frozen_state_slot": FROZEN_SLOT,
        "day_zero": "ended_at of the first clean managed close (W9)",
        "retire_after_seconds": 14 * 24 * 60 * 60,
        "rollback_deadline": "day_zero + retire_after_seconds, before any Q5 retirement intent",
        "requires": ["recorded user decision", "unit inactive", "legacy integrity and links valid",
                     "no pending release-mode transition or Q5 retirement intent"],
        "effects": ["resumes the frozen legacy state", "can serve erased data again",
                    "managed-state writes are absent from the frozen image"],
        "rollback_argv": controller + [
            "uninstall-launcher", "--unit-dir", "<user-unit-drop-in-dir>",
            "--user-decision", "<L4-decision-id>", "--phrase", L4_PHRASE, "--legacy-slot", LEGACY_SLOT,
        ],
        "restart_requires": ["exit zero", "empty mismatches", "legacy unit uses explicit runtime and frozen state"],
        "retirement_dry_run_argv": retire + ["--dry-run"],
        "retirement_confirm_argv": retire + ["--confirm", "<Q5-plan-sha256>"],
        "retirement_blocked_after_l4": True,
    }


def review_manifest(inputs: ReviewInputs) -> dict:
    """Generate evidence from the sealed archive, committed declarations and synthetic copy."""
    if re.fullmatch(r"[0-9a-f]{40}", inputs.commit) is None:
        raise ValueError("candidate commit must be a full lowercase Git SHA")
    if re.fullmatch(r"[0-9a-f]{64}", inputs.archive_sha256) is None:
        raise ValueError("sealed runtime archive hash must be a lowercase SHA-256")
    if sys.implementation.name != "cpython" or sys.implementation.cache_tag != "cpython-313":
        raise ValueError("release review requires CPython 3.13")
    repo = _review_path(inputs.repo)
    _check_git_commit(repo, inputs.commit)
    lock_sha = _read_lock(inputs.runtime_lock)
    declarations = {path: _sha256(_git_blob(repo, inputs.commit, path)) for path in DECLARATIONS}
    with tempfile.TemporaryDirectory(prefix="kza-p6-review-", dir=_review_path(Path("/tmp"))) as directory:
        scratch = Path(directory)
        snapshot = scratch / "artifact.tar.gz"
        size = _archive_snapshot(inputs.archive, snapshot, inputs.archive_sha256)
        manifest, files = _read_bundle(snapshot, inputs.commit)
        synthetic = _synthetic_preflight(scratch, manifest, files, inputs.archive_sha256)
    return {
        "manifest_version": 1,
        "git_commit": inputs.commit,
        "artifact": {
            "sha256": inputs.archive_sha256,
            "bytes": size,
            "format": manifest["format"],
            "entrypoint": manifest["entrypoint"],
            "python_abi": "cp313",
            "cache_tag": sys.implementation.cache_tag,
            "python_magic": manifest["python_magic"],
            "python_sources": 0,
            "runtime_files": len(files),
        },
        "schema_contract": {"epoch": SCHEMA_EPOCH, "sha256": SCHEMA_DIGEST},
        "controller": {
            "epoch": manifest["controller_epoch"],
            "revision": manifest.get("controller_revision"),
            "sources_sha256": manifest.get("controller_sources_sha256"),
            "launcher_contract": manifest["launcher_contract"],
        },
        "committed_declarations_sha256": declarations,
        "external_runtime_lock_sha256": lock_sha,
        "synthetic_adoption_preflight": synthetic,
        "rollback": rollback_plan(),
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("reviewed manifest contains duplicate keys")
    return value


def _render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Emit the reviewable JSON or recompute and verify an existing review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--commit", required=True, help="Full candidate SHA")
    parser.add_argument("--archive-sha256", required=True, help="SHA-256 from the sealed build evidence")
    parser.add_argument("--runtime-lock", required=True, type=Path, help="External resolved runtime lock")
    parser.add_argument("--verify", type=Path, help="Previously reviewed manifest to verify")
    args = parser.parse_args(argv)

    try:
        actual = _render(review_manifest(ReviewInputs(
            Path(__file__).resolve().parents[1], args.archive, args.commit, args.archive_sha256, args.runtime_lock,
        )))
        if args.verify is not None:
            expected = json.loads(_review_path(args.verify).read_bytes(), object_pairs_hook=_unique_object)
            if _render(expected) != actual:
                raise ValueError("reviewed manifest differs from the recomputed evidence")
        else:
            sys.stdout.write(actual)
    except (OSError, ValueError, TypeError, KeyError, EOFError, tarfile.TarError,
            sqlite3.DatabaseError, subprocess.SubprocessError) as error:
        sys.stderr.write(f"Release review refused: {error}\n")
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
