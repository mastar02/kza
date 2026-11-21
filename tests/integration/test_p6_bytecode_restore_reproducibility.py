"""T063: reproduce a committed artifact and reconcile real closed copies."""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
GRAPH = Path("data/graph")
PRIVACY = GRAPH / "privacy.db"
PROTECTED = ("reminders", "timers", "alert_preferences")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], cwd: Path, *, timeout: int = 120, **kwargs) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=timeout, **kwargs,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _build(directory: Path, commit: str) -> tuple[Path, dict]:
    directory.mkdir()
    scratch = directory / "compiler"
    scratch.mkdir()
    archive = directory / "runtime.tar.gz"
    _run(
        [sys.executable, "-I", "-B", str(REPO / "scripts/build_runtime.py"),
         str(archive), "--revision", commit],
        directory, env={**os.environ, "TMPDIR": str(scratch)},
    )
    runtime = directory / "runtime"
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert len(names) == len(set(names))
        assert all(member.isfile() for member in bundle.getmembers())
        bundle.extractall(runtime, filter="data")

    assert not list(runtime.rglob("*.py"))
    manifest_path = runtime / "release.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_commit"] == commit
    assert manifest["schema_contract"] == 2
    assert manifest["python"] == "3.13"
    assert manifest["python_magic"] == importlib.util.MAGIC_NUMBER.hex()
    assert manifest["entrypoint"] == ["python", "-m", "src.main"]
    files = {
        path.relative_to(runtime).as_posix(): _sha(path)
        for path in runtime.rglob("*") if path.is_file() and path != manifest_path
    }
    assert files == manifest["files"]
    bytecode = {name: digest for name, digest in files.items() if name.endswith(".pyc")}
    assert {"src/main.pyc", "scripts/stage_runtime.pyc"} <= bytecode.keys()
    for name in bytecode:
        assert (runtime / name).read_bytes()[:4] == importlib.util.MAGIC_NUMBER
    return archive, {"archive_sha256": _sha(archive), "manifest_sha256": _sha(manifest_path),
                     "bytecode": bytecode}


def _command(entry: Path, runtime: Path, checkout: Path, deps: list[str], mode: str,
             *arguments: Path) -> list[str]:
    return [sys.executable, "-I", "-S", "-B", str(entry), str(runtime), str(checkout),
            json.dumps(deps), mode, *map(str, arguments)]


def _isolated_imports(runtime: Path, checkout: Path, deps: list[Path]):
    """Allow dependency and fixture imports while blocking checkout file reads."""
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path)
    assert not runtime.is_relative_to(checkout)
    island = Path(__file__).resolve().parent

    def refuse_checkout_reads(event: str, args: tuple) -> None:
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(checkout) and not any(path.is_relative_to(dep) for dep in deps):
                raise PermissionError(f"checkout file access from isolated runtime: {path}")

    sys.addaudithook(refuse_checkout_reads)
    sys.path[:0] = [str(path) for path in (runtime, island, *deps)]
    from tests.mocks.compiled_restore_child import ArtifactGuard, _tripwires

    manifest = json.loads((runtime / "release.json").read_text())
    guard = ArtifactGuard(runtime, manifest["files"], "p6", (
        runtime, island, Path(sys.base_prefix).resolve(), *deps,
    ))
    sys.meta_path.insert(0, guard)
    hits, forbidden = _tripwires()
    import torch

    torch.hub.load = forbidden
    torch.load = forbidden
    torch.jit.load = forbidden
    return guard, hits


def _boot() -> None:
    from scripts import stage_runtime
    from src import main
    from src.core import compaction_worker

    assert callable(main.run_entry) and callable(main.build_application)
    assert callable(compaction_worker.main)
    assert stage_runtime.SCHEMA_CONTRACT_EPOCH == 2
    sys.argv = [stage_runtime.__file__, "--help"]
    try:
        stage_runtime.main()
    except SystemExit as result:
        assert result.code == 0
    else:
        raise AssertionError("stage_runtime --help did not run its CLI")


async def _write_protected(graph, mode: str, now: float) -> None:
    from src.reminders.reminder_store import ReminderScope, TimerState
    from src.timers.timer_store import TimerRecord, TimerStore

    reminders = graph.voice_services.reminders._store
    await reminders.create("household", f"Synthetic reminder {mode}", now + 3600,
                           scope=ReminderScope.SHARED)
    timers = TimerStore(reminders)
    await timers.save([TimerRecord(
        id=mode, label=f"Synthetic timer {mode}", actor="household", delivery_room=None,
        state=TimerState.RUNNING, duration_s=300, ends_at=now + 300, remaining_s=None,
        created_at=now, updated_at=now, revision=0,
    )])
    await reminders.set_alert_zones([f"synthetic-{mode}"], expected_revision=0 if mode == "O" else 1)


async def _write_state(mode: str, checkpoint: Path) -> None:
    """Close the real graph after seeding O or erasing the conversation in N."""
    import pytest

    from tests.mocks import application_restore as replay
    from tests.mocks.compiled_restore_child import DataCodec
    from tests.mocks.managed_slot_graph import load_canonical_configuration
    from tests.mocks.memory_graph import Clock
    from tests.mocks.restore_application_graph import application_graph

    state = Path("data").resolve().parent
    codec = DataCodec(checkpoint.parent / "witness-blobs")
    saved = codec.decode(json.loads(checkpoint.read_text())) if mode == "N" else None
    clock = Clock(saved["now"] if saved else time.time())
    async with application_graph(
        state / GRAPH, pytest.MonkeyPatch(), clock,
        configuration_factory=load_canonical_configuration,
    ) as g:
        await _write_protected(g.graph, mode, clock.now)
        if mode == "O":
            subject, survivor = await replay.seed_all(g)
            memory = g.memory.manager.long_term
            preferences = [
                await memory.astore_fact(f"Synthetic preference {label}", "preference", sources=(cohort.source,))
                for label, cohort in (("retired", subject), ("live", survivor))
            ]
            cohorts = (subject, survivor)
            paths = [
                sorted(str(Path(path).relative_to(state)) for path in (await replay.payload(g, cohort))["files"])
                for cohort in cohorts
            ]
            saved = {"subject": subject, "survivor": survivor, "preferences": preferences,
                     "paths": paths, "now": clock.now}
            # The witness contains only identifiers and paths. Its encoder requires no auxiliary blobs.
            checkpoint.write_text(json.dumps(codec.encode(saved)))
            assert not codec.manifest
        else:
            await _retire(g, saved["subject"])
            await replay.assert_subject_erased(g, saved["subject"])
            assert g.privacy.sources_available((saved["survivor"].source,))
        assert g.home["service_calls"] == 0
    if mode == "N":
        checkpoint.with_name("cut-time").write_text(str(clock.now))


async def _retire(graph, subject) -> None:
    from tests.mocks import application_restore as replay
    from tests.mocks.integrated_restore import steps

    operation = await graph.owner.request(subject.source.conversation_id, replay.memory_actor(graph))
    # A published browser delivery still owns a receipt after O's graph closes.
    await replay.clear_browser(graph, subject)
    deadline = time.monotonic() + 25
    while True:
        await graph.owner.reconcile()
        observed = await steps(graph, operation)
        pending = {(row["participant"], row["version"], row["state"], row["last_error"])
                   for row in observed if row["state"] != "complete"
                   and (row["participant"], row["version"]) not in replay.REMAINING_GUARDS}
        if not pending:
            break
        assert time.monotonic() < deadline, pending
        graph.clock.now += 1
        await asyncio.sleep(.05)
    assert {row["participant"] for row in observed} == replay.PARTICIPANTS


def _rows(path: Path, table: str) -> list[dict]:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}"')]


def _by_id(path: Path, table: str) -> dict:
    return {row["id"]: row for row in _rows(path, table)}


def _assert_restored(ordinary, latest, restored: Path, saved: dict, readback: Path) -> None:
    """Check the physical copy against both closed snapshots and the real SDK."""
    import chromadb
    from chromadb.config import Settings

    from src.memory.state_snapshot import collection_records

    old = ordinary.snapshot_root / "state"
    new = latest.snapshot_root / "state"
    retired, live = saved["subject"], saved["survivor"]
    dead_preference, live_preference = saved["preferences"]
    for relative in saved["paths"][0]:
        assert (old / relative).stat().st_size > 0
        assert not (new / relative).exists()
        assert not (restored / relative).exists(), relative
    for relative in saved["paths"][1]:
        assert (new / relative).read_bytes() == (old / relative).read_bytes()
        assert (restored / relative).read_bytes() == (new / relative).read_bytes(), relative

    old_facts, new_facts, restored_facts = [_by_id(root / PRIVACY, "facts") for root in (old, new, restored)]
    for identity in (retired.fact, dead_preference):
        assert old_facts[identity]["content"]
        assert new_facts[identity]["state"] == "deleted" and new_facts[identity]["content"] is None
        assert restored_facts[identity] == new_facts[identity]
    for identity in (live.fact, live_preference):
        assert new_facts[identity]["content"]
        assert restored_facts[identity] == new_facts[identity] == old_facts[identity]
    assert restored_facts[dead_preference]["category"] == restored_facts[live_preference]["category"] == "preference"
    tombstones = [
        {row["conversation_id"] for row in _rows(root / PRIVACY, "conversation_tombstones")}
        for root in (old, new, restored)
    ]
    assert retired.source.conversation_id not in tombstones[0]
    assert retired.source.conversation_id in tombstones[1]
    assert tombstones[2] == tombstones[1]
    assert live.source.conversation_id not in tombstones[2]

    old_captures, new_captures, restored_captures = [
        _by_id(root / GRAPH / "ambient.db", "utterances") for root in (old, new, restored)
    ]
    assert retired.ambient_id in old_captures
    assert retired.ambient_id not in new_captures and retired.ambient_id not in restored_captures
    expected = dict(new_captures[live.ambient_id])
    relative = Path(expected["audio_path"]).relative_to(latest.source_binding["state_root"])
    expected["audio_path"] = str(restored / relative)
    assert restored_captures[live.ambient_id] == expected

    old_clips, new_clips, restored_clips = [
        _by_id(root / PRIVACY, "wake_clips_v1") for root in (old, new, restored)
    ]
    assert set(retired.clip_ids) <= old_clips.keys()
    assert not set(retired.clip_ids) & (new_clips.keys() | restored_clips.keys())
    for identity in live.clip_ids:
        current = dict(restored_clips[identity])
        old_target, relocated_target = new_clips[identity]["target_id"], current.pop("target_id")
        assert old_target != relocated_target
        assert current == {key: value for key, value in new_clips[identity].items() if key != "target_id"}

    for table, retired_ids, live_ids in (
        ("voice_fingerprints", retired.voice_ids, live.voice_ids),
        ("conversation_turns_v1", retired.clip_ids, live.clip_ids),
    ):
        old_rows, new_rows, restored_rows = [_by_id(root / PRIVACY, table) for root in (old, new, restored)]
        assert set(retired_ids) & old_rows.keys()
        assert not set(retired_ids) & (new_rows.keys() | restored_rows.keys())
        kept = set(live_ids) & old_rows.keys()
        assert kept
        assert {identity: restored_rows[identity] for identity in kept} == {
            identity: new_rows[identity] for identity in kept
        }

    reminder_db, = (path for path, image in latest.manifest["databases"].items()
                    if set(PROTECTED) <= image["tables"].keys())
    for table in PROTECTED:
        assert _rows(old / reminder_db, table)
        assert _rows(old / reminder_db, table) != _rows(new / reminder_db, table)
    assert (restored / reminder_db).read_bytes() == (new / reminder_db).read_bytes()

    old_index, = ordinary.manifest["index_exports"]
    new_index, = latest.manifest["index_exports"]
    dead = {retired.fact, dead_preference}
    kept = {live.fact, live_preference}
    assert dead | kept <= {record["id"] for record in old_index["records"]}
    new_records = {record["id"]: record for record in new_index["records"]}
    assert not dead & new_records.keys() and kept <= new_records.keys()
    shutil.copytree(restored / new_index["relative_path"], readback)
    client = chromadb.PersistentClient(path=str(readback), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(new_index["collection"], embedding_function=None)
        observed = {record["id"]: record for record in collection_records(collection)}
        assert not dead & observed.keys()
        assert {identity: observed[identity] for identity in kept} == {
            identity: new_records[identity] for identity in kept
        }
    finally:
        client.close()


def _rehearse(checkout: Path, deps: list[str], task: Path, archive: Path) -> None:
    import yaml

    from src.core.runtime_admission import _stage
    from src.memory import output_registration, payload_derivation
    from src.memory.state_snapshot import snapshot_closed_state, verify_snapshot
    from tests.mocks.compiled_restore_child import DataCodec
    from tests.mocks.managed_slot_graph import canonical_template
    from tests.mocks.restore_application_graph import configuration
    from tests.mocks.slot_reconciliation import fixture_layout

    stage = _stage()
    root = task / "stage"
    source = task / "initial"
    state = root / "slots/candidate/state"
    (source / "config").mkdir(parents=True)
    (source / GRAPH).mkdir(parents=True)
    settings = canonical_template(state / GRAPH, configuration)
    (source / "config/settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    release = stage.prepare_slot(archive, root, "candidate", source)
    assert release["schema_contract"] == stage.SCHEMA_CONTRACT_EPOCH == 2
    empty = task / "empty"
    (empty / "config").mkdir(parents=True)
    (empty / "data").mkdir()
    assert stage.prepare_slot(archive, root, "restored", empty) == release

    checkpoint = task / "cohorts.json"
    images = {}
    with stage.AdmissionController(root) as controller:
        for mode in ("O", "N"):
            command = _command(Path(__file__), state.parent / "runtime", checkout, deps, mode, checkpoint)
            try:
                controller.run("candidate", command, timeout=75, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as error:
                raise AssertionError(error.stderr.decode(errors="replace")) from error
            head = stage.read_admission_head(root)
            assert head["status"] == "closed"
            images[mode] = snapshot_closed_state(controller, head["binding_id"], task / mode)
        assert images["O"].source_binding_id != images["N"].source_binding_id
        inventory = output_registration.read_output_inventory(controller)
        assert len(inventory) == 2
        assert {record["run_id"] for record in inventory} == {
            image.manifest["source_witness"]["proof"]["running_head"]["run_id"]
            for image in images.values()
        }
        assert all(record["version"] == "v4" and record["header"]["schema_version"] == 4
                   and record["phase"] == "observed" and len(record["required_producers"]) == 32
                   for record in inventory)

        destination = root / "slots/restored/state"
        ordinary, latest = images["O"], images["N"]
        layout = fixture_layout(ordinary, latest, destination)
        now = float(checkpoint.with_name("cut-time").read_text())
        plan = payload_derivation.plan_payload_cut(ordinary, latest, now=now, layout=layout)
        result = payload_derivation.materialize_derived_state(plan, destination, controller=controller)
        assert not result.serving
        assert stage.read_admission_head(root) == head
        assert stage.verify_closed_binding(root, head["binding_id"]) == latest.source_binding
        for image in images.values():
            verify_snapshot(image)
        assert output_registration.read_output_inventory(controller) == inventory

    saved = DataCodec(task / "witness-blobs").decode(json.loads(checkpoint.read_text()))
    _assert_restored(ordinary, latest, destination, saved, task / "sdk-readback")
    (task / "restore-proof.json").write_text(json.dumps({
        "commit": release["git_commit"], "schema_contract": 2,
        "journal": "v4", "runs": [record["run_id"] for record in inventory],
        "snapshot_sha256": {mode: image.manifest_sha256 for mode, image in images.items()},
        "plan_sha256": result.plan_sha256, "serving": result.serving,
    }, indent=2) + "\n")


def test_p6_bytecode_and_compatible_restore(tmp_path: Path) -> None:
    """Exercise both T063 deliverables on the exact committed candidate."""
    from tests.mocks.compiled_restore import _environment, copy_island

    assert sys.implementation.cache_tag == "cpython-313" and sys.abiflags == ""
    requested = os.environ.get("KZA_P6_REVISION")
    assert requested is None or re.fullmatch(r"[0-9a-f]{40}", requested)
    commit = _run(["git", "rev-parse", "--verify", (requested or "HEAD") + "^{commit}"], REPO).strip()
    task = tmp_path.resolve()
    assert not task.is_relative_to(REPO)
    archive_a, first = _build(task / "a", commit)
    _, second = _build(task / "b", commit)
    assert first == second
    (task / "build-proof.json").write_text(json.dumps(
        {"commit": commit, "abi": "cp313", "schema_contract": 2, **first}, indent=2,
    ) + "\n")

    island = task / "fixtures"
    copy_island(REPO, island)
    for name in ("managed_slot_graph", "slot_reconciliation"):
        relative = Path("tests/mocks") / f"{name}.py"
        shutil.copyfile(REPO / relative, island / relative)
    entry = island / "p6.py"
    shutil.copyfile(Path(__file__), entry)
    runtime = archive_a.parent / "runtime"
    deps = sorted({sysconfig.get_path(name) or "" for name in ("purelib", "platlib")})
    assert all(Path(path).is_absolute() and Path(path).is_dir() for path in deps)
    env = _environment()
    boot_log = _run(_command(entry, runtime, REPO, deps, "boot"), runtime, env=env)
    (task / "boot.log").write_text(boot_log)
    _run(_command(entry, runtime, REPO, deps, "restore", task, archive_a),
         runtime, env=env, timeout=240)


if __name__ == "__main__":
    runtime_arg, checkout_arg, deps_arg, mode_arg, *other_args = sys.argv[1:]
    dependency_paths = json.loads(deps_arg)
    guard, hits = _isolated_imports(Path(runtime_arg), Path(checkout_arg), [
        Path(path).resolve() for path in dependency_paths
    ])
    from src import main  # Load the real application in each isolated process.
    assert callable(main.run_entry)
    if mode_arg == "boot":
        _boot()
    elif mode_arg == "restore":
        _rehearse(Path(checkout_arg), dependency_paths, *map(Path, other_args))
    else:
        assert mode_arg in {"O", "N"}
        asyncio.run(_write_state(mode_arg, Path(other_args[0])))
    guard.audit()
    assert not hits
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("scripts.", "_kza_runtime_", "_kza_managed_launcher_")):
            origin = Path(module.__file__).resolve()
            assert origin.is_relative_to(runtime_arg)
            assert isinstance(module.__loader__, importlib.machinery.SourcelessFileLoader)
