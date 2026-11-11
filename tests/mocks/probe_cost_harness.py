"""Measure the U3b-0a candidate probe on a production-size synthetic state.

Not collected by pytest. Run from the checkout (CPython 3.13 venv):

    .venv/bin/python -B -m tests.mocks.probe_cost_harness --work DIR --output result.json [--scale N]

DEPLOYMENT.md documents no state sizes, so scale 1 is a reasoned upper estimate
of the protected/ordinary stores a managed activation hashes. Every row is
synthetic; no production data is read. Sizes follow the stores' retention:

- ambient.db  utterances: 48 h TTL x 6 rooms x one utterance per 3.5 s while
  speech is present 16 h/day -> ~250k rows of ~120-char text.
- events.db   events_policy_v1: 500k home-automation events.
- memory.db   20k facts (40% confirmed, i.e. protected projection) + sources.
- voice.db    5k ECAPA fingerprints (192 x float32) and 300 review clips of
  30 s 16 kHz 16-bit audio (~0.96 MB each, ~290 MB of BLOBs).
- diagnostics.db metrics: 300k numeric records.

The probe and the controller each copy every database into memory
(_database_image) and hash protected projections, so wall time scales with
total database bytes. The harness times the controller's _binding_snapshot and
the probe (a fresh interpreter running the candidate's own companion).
"""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import statistics
import sys
import time
from contextlib import closing

REPO = Path(__file__).resolve().parents[2]


def _literal(filename: str, *, assignment: str | None = None, table: str | None = None) -> str:
    tree = ast.parse((REPO / filename).read_text())
    if assignment:
        return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(getattr(target, "id", None) == assignment for target in node.targets))
    pattern = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)?\s+" + table + r"\s*\(", re.I)
    return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                and type(node.value) is str and pattern.search(node.value))


def _rows(count: int, make):
    return (make(index) for index in range(count))


def build_state(source: Path, scale: int) -> dict:
    """Write the synthetic state; return per-database byte sizes."""
    rng = random.Random(20261110)
    data = source / "data"
    (source / "config").mkdir(parents=True)
    data.mkdir()
    (source / "config" / "settings.yaml").write_text("schema: probe-cost\n")
    now = 1_800_000_000.0
    text = "texto sintético de una frase ambiental sin datos reales del hogar " * 2
    sources = json.dumps([{"capture_id": "capture", "conversation_id": "U",
                           "captured_at": now, "expires_at": now + 172800}])

    with closing(sqlite3.connect(data / "ambient.db")) as db, db:
        db.executescript(_literal("src/ambient/store.py", assignment="_SCHEMA"))
        db.execute("ALTER TABLE pending_audio_deletions ADD COLUMN conversation_id TEXT")
        db.executemany("INSERT INTO utterances(room_id,t0,t1,text,created_at,capture_id,conversation_id,"
                       "expires_at) VALUES(?,?,?,?,?,?,?,?)", _rows(250_000 * scale, lambda i: (
                           "room" + str(i % 6), now - i, now - i + 2, text[: 60 + i % 60], now - i,
                           "capture" + str(i), "conv" + str(i // 20), now + 172800 - i)))

    with closing(sqlite3.connect(data / "events.db")) as db, db:
        db.executescript(_literal("src/analytics/event_logger.py", table="events_policy_v1"))
        db.executemany("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,user_id,"
                       "sources_json,metrics_json,expires_at,text_expires_at,hour,minute,weekday,is_weekend,"
                       "time_of_day) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       _rows(500_000 * scale, lambda i: (
                           now - i * 10, "state_change", "light.room" + str(i % 40), "on", "u" + str(i % 4),
                           sources, "{}", now + 86400 * 30, now + 86400, i % 24, i % 60, i % 7, i % 7 > 4,
                           "night")))

    with closing(sqlite3.connect(data / "memory.db")) as db, db:
        db.executescript(_literal("src/memory/fact_ledger.py", assignment="_SCHEMA"))
        facts = 20_000 * scale
        db.executemany("INSERT INTO facts VALUES(?,?,?,?,?,?,?,?,?,?)", _rows(facts, lambda i: (
            "f" + str(i), "hecho sintético " + str(i), "general", 0.8,
            "confirmed" if i % 5 < 2 else "automatic", now - i, None if i % 5 < 2 else now + 86400, 1,
            now - i if i % 5 < 2 else None, "admin" if i % 5 < 2 else None)))
        db.executemany("INSERT INTO fact_sources VALUES(?,?,?,?,?,?,?)", _rows(facts, lambda i: (
            "f" + str(i), "capture" + str(i), "conv" + str(i // 20), now - i, now + 172800, "voice", "[]")))

    with closing(sqlite3.connect(data / "voice.db")) as db, db:
        db.executescript(_literal("src/memory/voice_schema.py", assignment="VOICE_SCHEMA"))
        db.execute("INSERT INTO voice_group_schema VALUES(1)")
        embedding = bytes(rng.getrandbits(8) for _ in range(768))
        db.executemany("INSERT INTO voice_fingerprints(id,model_key,dimensions,embedding,duration_s,created_at,"
                       "expires_at,sources_json) VALUES(?,?,?,?,?,?,?,?)", _rows(5_000 * scale, lambda i: (
                           "fp" + str(i), "ecapa", 192, embedding, 3.0, now - i, now + 172800, sources)))
        clip = os.urandom(960_000)
        db.executemany("INSERT INTO voice_capture_reviews(fingerprint_id,audio_wav,transcript,room_id,duration_s,"
                       "clip_duration_s,audio_metadata_json,transcript_metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                       _rows(300 * scale, lambda i: ("fp" + str(i), clip, "transcripción", "room", 3.0, 30.0,
                                                     "{}", "{}")))

    with closing(sqlite3.connect(data / "diagnostics.db")) as db, db:
        db.executescript(_literal("src/obs/diagnostic_store.py", table="metrics"))
        db.execute("PRAGMA application_id=1264206897")
        db.execute("PRAGMA user_version=2")
        payload = json.dumps({"latency_ms": 123.4, "stage": "stt", "room": "room"})
        db.executemany("INSERT INTO metrics VALUES(?,?,?,?,?,?)", _rows(300_000 * scale, lambda i: (
            "m" + str(i), now - i, now + 86400 * 7, payload, i % 16, len(payload))))
    return {path.name: path.stat().st_size for path in sorted(data.glob("*.db"))}


def _working_tree_archives(work: Path) -> dict:
    """Build X (committed HEAD, pre-0b) and Y (this working tree, committed only in a clone)."""
    import subprocess
    from scripts.build_runtime import build_runtime

    def git(cwd, *arguments, **kwargs):
        return subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True,
                              timeout=120, **kwargs).stdout.strip()

    clone = work / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    git(clone, "read-tree", "HEAD")
    for relative in git(REPO, "diff", "--name-only", "HEAD", "--", "scripts", "src").split():
        blob = git(clone, "hash-object", "-w", "--path", relative, str(REPO / relative))
        git(clone, "update-index", "--add", "--cacheinfo", "100644," + blob + "," + relative)
    environment = dict(os.environ, GIT_AUTHOR_NAME="harness", GIT_AUTHOR_EMAIL="harness@invalid",
                       GIT_COMMITTER_NAME="harness", GIT_COMMITTER_EMAIL="harness@invalid")
    y = git(clone, "commit-tree", git(clone, "write-tree"), "-p", "HEAD", "-m", "harness", env=environment)
    archives = {"X": work / "x.tar.gz", "Y": work / "y.tar.gz"}
    build_runtime(clone, archives["X"], revision=git(clone, "rev-parse", "HEAD"))
    build_runtime(clone, archives["Y"], revision=y)
    return archives


def multi_hop(work: Path, scale: int, repeat: int) -> dict:
    """U3b-0b F2: probe and validation cost for C holding 5 targets from 2 owners.

    A (production-size synthetic state) registers 1 path target and runs closed;
    B is a copy that registers 4 more and runs closed; C is a copy delegated from
    B, so it holds 5 delegated path targets owned by A (1) and B (4). Measures the
    controller's validation pass, the candidate probe for a 0b candidate (Y) and a
    pre-0b candidate (X), counting owner snapshots, and one full managed activate.
    """
    from scripts import stage_runtime as stage
    from tests.unit.tools.test_stage_runtime_admission import (
        _CLOSING_CHILD, _TARGETS_DDL, _command, _registration)

    insert = "INSERT INTO erasure_targets VALUES(?,?,?,?)"

    def local(state, name):
        directory = state / "data" / ("clips-" + name)
        directory.mkdir()
        return _registration("collectors", 1, {"kind": "wake_clips", "root": str(directory)})

    # Authority may only arrive through an admitted run (a first head cannot adopt
    # an unsealed legacy authority), so A's child copies the synthetic databases in.
    child = _CLOSING_CHILD.replace("import importlib.util, json, os, sqlite3, sys",
                                   "import importlib.util, json, os, shutil, sqlite3, sys")
    child = child.replace("module.write_admission_receipt(", "\n".join([
        "for _path in json.loads(sys.argv[2]):",
        "    shutil.copyfile(_path, os.path.join('data', os.path.basename(_path)))",
        "module.write_admission_receipt("]))

    def run(root, name, statements, copies=()):
        stage.run_slot(root, name, _command(child, json.dumps(statements), json.dumps(list(copies))),
                       timeout=stage._PROBE_TIMEOUT_BOUNDS[1])

    archives = _working_tree_archives(work)
    started = time.monotonic()
    sizes = build_state(work / "source", scale)
    # build_state shortens every event's text deadline below its sources' ceiling, so
    # all 500k rows become protected deadline fences and the closed binding exceeds the
    # 16 MB record cap (runtime_admission_unsafe_record, a separate 0a finding). Here the
    # deadline equals the ceiling instead: same bytes, ordinary payload.
    with closing(sqlite3.connect(work / "source" / "data" / "events.db")) as db, db:
        db.execute("UPDATE events_policy_v1 SET text_expires_at=expires_at-86400*30+172800")
    sizes = {path.name: path.stat().st_size for path in sorted((work / "source" / "data").glob("*.db"))}
    built = time.monotonic() - started
    empty = work / "empty"
    (empty / "config").mkdir(parents=True)
    (empty / "data").mkdir()
    (empty / "config" / "settings.yaml").write_text("schema: probe-cost\n")
    root = work / "staging"
    stage.prepare_slot(archives["Y"], root, "a", empty)
    root = root.resolve()
    states = {"a": root / "slots/a/state"}
    run(root, "a", [["data/privacy.db", _TARGETS_DDL, []], ["data/privacy.db", insert, local(states["a"], "a")]],
        copies=[str(path) for path in sorted((work / "source" / "data").glob("*.db"))])
    stage.prepare_slot(archives["Y"], root, "b", states["a"])
    states["b"] = root / "slots/b/state"
    stage.delegate_slot(root, "b")
    run(root, "b", [["data/privacy.db", insert, local(states["b"], "b%d" % index)] for index in range(4)])
    result = {"schema_version": 1, "unit": "U3b-0b/F2", "scale": scale, "python": sys.version.split()[0],
              "database_bytes_per_state": sizes, "total_database_bytes_per_state": sum(sizes.values()),
              "states_on_disk": 3, "state_build_s": round(built, 2),
              "default_timeout_s": stage._PROBE_TIMEOUT_SECONDS, "candidates": {}}
    counts, original = [], stage._binding_snapshot

    def counting(root_, name, **kwargs):
        counts.append(name)
        return original(root_, name, **kwargs)

    for name, release in (("c_x", "X"), ("c", "Y")):
        stage.prepare_slot(archives[release], root, name, states["b"])
        stage.delegate_slot(root, name)
        binding = stage._binding_snapshot(root, name)
        delegated = json.loads((root / "slots" / name / "state/.kza-target-ownership.json").read_text())
        validation, probe = [], []
        for _ in range(repeat):
            stage._binding_snapshot = counting
            counts.clear()
            begin = time.monotonic()
            stage.validate_target_ownership(root, binding)
            validation.append(time.monotonic() - begin)
            stage._binding_snapshot = original
            begin = time.monotonic()
            stage._probe_candidate(root, name, binding, timeout=stage._PROBE_TIMEOUT_BOUNDS[1])
            probe.append(time.monotonic() - begin)
        result["candidates"][name] = {
            "code": "0b (Y)" if release == "Y" else "pre-0b (X = HEAD)",
            "delegated_targets": len(delegated["delegated_targets"]),
            "controller_validation_snapshots": {slot: counts.count(slot) for slot in sorted(set(counts))},
            "controller_validation_s": [round(value, 2) for value in validation],
            "probe_s": [round(value, 2) for value in probe],
            "probe_median_s": round(statistics.median(probe), 2),
            "probe_fraction_of_default_timeout": round(statistics.median(probe) / stage._PROBE_TIMEOUT_SECONDS, 3)}
    stage._binding_snapshot = counting
    counts.clear()
    begin = time.monotonic()
    stage.activate_slot(root, "c")
    result["activate_c_s"] = round(time.monotonic() - begin, 2)
    result["activate_c_controller_snapshots"] = {slot: counts.count(slot) for slot in sorted(set(counts))}
    stage._binding_snapshot = original
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="New directory outside the checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--multi-hop", action="store_true",
                        help="U3b-0b: 2 owners, 5 delegated targets; 0b and pre-0b candidate probes")
    args = parser.parse_args()
    from scripts import stage_runtime as stage
    from scripts.build_runtime import build_runtime

    work = args.work.resolve()
    if work.exists() or work.is_relative_to(REPO):
        raise SystemExit("work directory must be new and outside the checkout")
    work.mkdir(parents=True)
    if args.multi_hop:
        result = multi_hop(work, args.scale, args.repeat)
        args.output.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result))
        return
    archive = work / "runtime.tar.gz"
    manifest = build_runtime(REPO, archive)
    started = time.monotonic()
    sizes = build_state(work / "source", args.scale)
    built = time.monotonic() - started
    root = work / "staging"
    stage.prepare_slot(archive, root, "candidate", work / "source")
    root = root.resolve()
    snapshot, probe = [], []
    binding = None
    for _ in range(args.repeat):
        begin = time.monotonic()
        binding = stage._binding_snapshot(root, "candidate")
        snapshot.append(time.monotonic() - begin)
        begin = time.monotonic()
        stage._probe_candidate(root, "candidate", binding, timeout=stage._PROBE_TIMEOUT_BOUNDS[1])
        probe.append(time.monotonic() - begin)
    result = {"schema_version": 1, "unit": "U3b-0a/R4", "scale": args.scale,
              "candidate_commit": manifest["git_commit"], "python": sys.version.split()[0],
              "database_bytes": sizes, "total_database_bytes": sum(sizes.values()),
              "has_authority": stage._has_authority(binding), "state_build_s": round(built, 2),
              "controller_snapshot_s": [round(value, 3) for value in snapshot],
              "probe_s": [round(value, 3) for value in probe],
              "probe_median_s": round(statistics.median(probe), 3),
              "default_timeout_s": stage._PROBE_TIMEOUT_SECONDS,
              "probe_fraction_of_default_timeout": round(statistics.median(probe) / stage._PROBE_TIMEOUT_SECONDS, 3)}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
