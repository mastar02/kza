#!/usr/bin/env python3
"""T6.1 command line entry point for the offline KZA to Nexa migration."""

import argparse
import os
import sqlite3
import stat
import sys
import time
from contextlib import ExitStack, closing
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import authority
import household
import memory
from privacy import Ledger, Privacy
from storage import (
    Catalog, FACT_MODEL, JSON_FILES, LOCATIONS, MigrationError, VERSIONS, VOICE_MODEL,
    check, checksum, decode, encode, files_under, private_parents, quoted, require,
    sqlite_file, start_source, start_target, validate_target, write_json,
)

REPORT_NAME = "migration-report.json"


def paths(source: Path, dest: Path, report: Path | None, dry_run: bool) -> tuple:
    require(not source.is_symlink() and source.is_dir(), "source must be a directory")
    require(not dest.is_symlink(), "destination may not be a symlink")
    source, dest = source.resolve(), dest.resolve()
    require(not dest.is_relative_to(source) and not source.is_relative_to(dest),
            "source and destination overlap")
    require(dest.parent.is_dir(), "destination parent must exist")
    require_empty(dest)
    if report is not None:
        require(not report.is_symlink(), "report may not be a symlink")
        report = report.resolve()
        require(not report.is_relative_to(source) and not source.is_relative_to(report),
                "source and report overlap")
        if report.is_relative_to(dest):
            require(not dry_run and report != dest, "report would overlap the dry-run destination")
        else:
            require(report.parent.is_dir() and not dest.is_relative_to(report),
                    "report parent must exist outside the source")
    return source, dest, report


def require_empty(dest: Path) -> None:
    require(not dest.is_symlink() and
            (not dest.exists() or dest.is_dir() and not any(dest.iterdir())),
            "destination exists and is not an empty directory")


def load_snapshot(root: Path, stack: ExitStack) -> tuple[Catalog, list | None, list | None]:
    source = Catalog()
    supported = authority.TABLES | household.TABLES | memory.TABLES
    json_inputs = {}
    for relative in files_under(root):
        path = root / relative
        if sqlite_file(relative):
            db = stack.enter_context(closing(start_source(path)))
            db.executescript("PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;")
            check(db, foreign_keys=False)
            source.add(db, supported)
        elif path.name in JSON_FILES:
            require(path.name not in json_inputs, "duplicate JSON authority")
            json_inputs[path.name] = path
    require(bool(source.tables or json_inputs), "no supported legacy data found")
    users_file = json_inputs.get("users_accounts_v1.json", json_inputs.get("users.json"))
    users = read_export(users_file, "users")
    if "users_accounts_v1.json" in json_inputs:
        users_data = decode(users_file.read_text(encoding="utf-8"))
        require(isinstance(users_data, dict) and users_data.get("schema") == "kza-users-accounts-v1",
                "unsupported account catalog schema")
    return source, users, read_export(json_inputs.get("alerts.json"), "alerts")


def read_export(file: Path | None, key: str) -> list[dict] | None:
    if file is None:
        return None
    payload = decode(file.read_text(encoding="utf-8"))
    rows = payload[key] if isinstance(payload, dict) else payload
    require(type(rows) is list and all(type(row) is dict for row in rows), "invalid JSON export")
    return rows


def database_counts(db: sqlite3.Connection) -> dict[str, int]:
    names = db.execute("PRAGMA table_list")
    return {name: db.execute(f"SELECT count(*) FROM {quoted(name)}").fetchone()[0]
            for schema, name, kind, *_ in names
            if schema == "main" and kind in {"table", "virtual"} and not name.startswith("sqlite_")}


def populate(snapshot: Path, output: Path) -> dict:
    now = time.time()
    with ExitStack() as stack:
        source, users, alerts = load_snapshot(snapshot, stack)
        ledger = Ledger.read(source)
        privacy = Privacy(ledger, memory.tombstones(source), authority.read_retired(source))
        dbs = {name: stack.enter_context(closing(start_target(output, name))) for name in VERSIONS}
        mem, ambient, home, auth = (dbs[name] for name in VERSIONS)
        mem.executemany("INSERT INTO schema_meta VALUES (?,?,?)",
                        ((FACT_MODEL, "fact", 1024), (VOICE_MODEL, "voice", 192)))
        authority.migrate_accounts(source, auth, privacy, users)
        ambient_input = memory.AmbientInputs.read(source, privacy)
        retired_captures = memory.retired_capture_ids(source, privacy)
        aliases, blocked = ambient_input.migrate(source, ambient, privacy, now, retired_captures)
        fact_reasons = memory.migrate_facts(source, mem, privacy, now, retired_captures)
        captures = memory.migrate_voice(source, mem, privacy, aliases, blocked, fact_reasons, now)
        translations = household.migrate_household(source, home, privacy, alerts, now)
        authority.migrate_auth(source, auth, privacy, captures)
        mem.executemany("INSERT INTO conversation_tombstones VALUES (?,?)",
                        sorted(privacy.tombstones.items()))
        source.count("conversation_tombstones", "migrated", source.counts["conversation_tombstones"].source)
        ledger.write(source, mem, output, privacy.tombstones, privacy.retired)
        for row in source.rows("erasure_ops"):
            require(not ledger.legacy, "ambiguous legacy erasure receipt")
            mem.execute("INSERT INTO erasure_ops VALUES (?,?,?,?,?)", tuple(row[key] for key in
                        ("ledger_seq", "erasure_id", "facts_deleted", "vectors_deleted", "captures_deleted")))
            source.count("erasure_ops", "migrated")
        for name, db in dbs.items():
            validate_target(db, name)
        counts = {table: asdict(entry) for table, entry in sorted(source.counts.items())}
        for entry in counts.values():
            require(entry["source"] == sum(value for key, value in entry.items() if key != "source"),
                    "migration accounting does not reconcile")
        return {
            "tables": counts,
            "destination_tables": {name: database_counts(db) for name, db in dbs.items()},
            "destination_schema_versions": VERSIONS,
            "translations": translations,
        }


def migrate(source: Path, dest: Path, dry_run: bool, report_path: Path | None) -> dict:
    source, dest, report_path = paths(source, dest, report_path, dry_run)
    with TemporaryDirectory(prefix=".nexa-migrate-", dir=dest.parent) as temporary:
        staging = Path(temporary)
        snapshot, output = staging / "input", staging / "output"
        output.mkdir(mode=0o700)
        before = checksum(source, snapshot, keep_snapshot=True)
        try:
            report = populate(snapshot, output)
        finally:
            after = checksum(source, staging / "after")
            require(before == after, "source changed during migration")
        report["dry_run"] = dry_run
        # Ordinals pair the two inventories without leaking private filenames.
        report["source_checksums"] = [
            {"file": index, "sha256_before": before[key][0], "sha256_after": after[key][0]}
            for index, key in enumerate(before, start=1)
        ]
        for location in LOCATIONS.values():
            path = output / location
            require(stat.S_IMODE(path.stat().st_mode) == 0o600
                    and stat.S_IMODE(path.parent.stat().st_mode) == 0o700,
                    "destination permissions differ from private schema")
        write_json(output / REPORT_NAME, report)
        if report_path is not None and report_path.is_relative_to(dest):
            output_report = output / report_path.relative_to(dest)
            if output_report != output / REPORT_NAME:
                private_parents(output_report.parent, output)
                write_json(output_report, report)
        elif report_path is not None:
            with TemporaryDirectory(prefix=".nexa-report-", dir=report_path.parent) as temp_report:
                file = Path(temp_report) / REPORT_NAME
                write_json(file, report)
                os.replace(file, report_path)
        if not dry_run:
            require_empty(dest)
            os.rename(output, dest)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = migrate(args.source, args.dest, args.dry_run, args.report)
    except MigrationError as error:
        sys.stderr.write(f"Migration refused: {error}\n")
        return 1
    except (OSError, sqlite3.Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError,
            OverflowError) as error:
        # Raw errors can contain source paths, identifiers or SQL values.
        sys.stderr.write(f"Migration refused: invalid or inaccessible data ({type(error).__name__})\n")
        return 1
    if args.dry_run and args.report is None:
        sys.stdout.write(encode(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
