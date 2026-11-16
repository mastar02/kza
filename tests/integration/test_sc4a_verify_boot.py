"""SC-4a src side: the verify_only graph (sc-release-design v4 v_verification_run, skip_list).

A verification boot constructs, opens, validates and loads the real graph
(main.build_application with the offline leaves SC-0 measured) but starts no
recurring loop, runs no one-shot repair and commits no dashboard-authority
transaction. Each test builds the state with a normal first boot at one
fixed path (registered targets are absolute, like SC-0's fixed staging path),
plants SC-0's residue (tools/sc0/residue.py), takes the protected projection
(stage_runtime.capture_state_authority), runs the verify branch to READY and
through the close, and takes the projection again after the close (R5V-4).

Checked: the structural pins (each listed loop is never started in verify
mode and is started in a normal boot, so the pin discriminates; the boot
prune and the routine scheduler are never reached), the residue canaries
(R3-PRIV-1, R5-PRIV-1, R5V-1, amendment A1: a correct NEW leaves K' equal to K,
so retire would be allowed, not SS), the SQL trace (no write statement on a
projected table except the pinned no-op exemptions; the authority's own
statements are rolled back, c-authority), no 'upgrade' statement over a
contract-1 state, the verify-mode capture close (rows carried unchanged and
counted) and the backstops: a protected write planted after READY in a
construction-spawned task, or in an exit-stack callback, and an exempt
statement that changes a row each leave a non-empty diff (retire refuses).
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import threading
import time

import pytest

from scripts import stage_runtime as stage
from src.core.process_lifecycle import ProcessLifecycle, StopPhase
from src.memory.erasure_journal import ErasureParticipant
from src.memory.runtime_closure import RuntimeClosureWitness
from tests.mocks.application_graph import offline_home
from tests.mocks.producer_erasure_graph import construction_sdk_guards, nominal_factories
from tests.mocks.restore_application_graph import MODEL_ENV, TOKEN, configuration, local_models, sdk_leaves
from tools.sc0 import residue
from tools.sc0.write_inventory import projected_tables

pytestmark = pytest.mark.asyncio

REPO = Path(__file__).resolve().parents[2]
PROJECTED = projected_tables(REPO)
AUTHORITY_DB = "dashboard-auth.db"
# skip_list.no_op_exemption: the only write statements on projected tables a verify boot may run
# besides the authority's rolled-back transactions (c-authority) and schema DDL. Each inserts a
# default or marker row only when absent; (file, function, verb, table).
NO_OP_EXEMPTIONS = frozenset({
    ("src/ambient/control.py", "_load", "INSERT OR IGNORE INTO", "ambient_runtime_control"),
    ("src/memory/capture_control.py", "_load_policy", "INSERT INTO", "capture_runtime_control"),
    ("src/memory/voice_schema.py", "initialize_voice_people_schema", "INSERT INTO", "voice_person_schema"),
    ("src/memory/voice_schema.py", "initialize_voice_review_schema", "INSERT INTO", "voice_capture_review_schema"),
    ("src/memory/voice_schema.py", "initialize_voice_attribution_schema", "INSERT INTO", "voice_attribution_schema"),
    ("src/memory/fact_ledger.py", "initialize_fact_owner_schema", "INSERT INTO", "fact_owner_schema"),
    ("src/users/voice_groups.py", "initialize", "INSERT INTO", "voice_group_schema"),
})
# The residue canaries run in pytest; the 75 s settles and the deferred-probe state need the
# compiled child's timing and run in SC-0's harness with the list on (sc4a-evidence cr7).
CANARIES = ("clean", "retired_owner_facts", "tombstone_without_operation", "retired_user_credential_session",
            "pending_erasure_step", "queued_audio_deletion", "expired_sessions_credentials",
            "recorders_clean_stop_running", "recorders_clean_stop_muted", "recorders_crash_running",
            "recorders_crash_muted", "recorder_orphan_unknown", "reminder_action_sending",
            "diagnostic_quarantine_staged", "deleted_fact_owner_rows", "routine_run_interrupted",
            "owned_fact_expired", "expired_voice_group_request", "wake_clip_fenced",
            "deleted_fact_no_index_row", "retention_shortened")
_WRITE = re.compile(r"^\s*(?:WITH\b.*?\)\s*)?(INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE(?:\s+OR\s+\w+)?"
                    r"|DELETE\s+FROM|ALTER\s+TABLE)\s+[\"'`\[]?([A-Za-z_][\w]*)", re.IGNORECASE | re.DOTALL)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Trace:
    """Every executed write statement on a projected table: (db name, verb, table, src frame)."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str, str, str]] = []
        self._lock = threading.Lock()

    def install(self, patch) -> None:
        real = sqlite3.connect

        def connect(*args, **kwargs):
            connection = real(*args, **kwargs)
            target = kwargs.get("database", args[0] if args else "")
            name = os.path.basename(os.fsdecode(target)) if isinstance(target, (str, bytes, os.PathLike)) else ""

            def trace(statement: str) -> None:
                match = _WRITE.match(statement)
                if match is None or match.group(2) not in PROJECTED:
                    return
                frame = next((f"{item.filename[item.filename.rfind('/src/') + 1:]}:{item.name}"
                              for item in reversed(__import__("traceback").extract_stack())
                              if "/src/" in item.filename), "?")
                with self._lock:
                    self.writes.append((name, re.sub(r"\s+", " ", match.group(1).upper()), match.group(2), frame))

            connection.set_trace_callback(trace)
            return connection

        patch.setattr(sqlite3, "connect", connect)

    def unexpected(self) -> list[tuple]:
        allowed = {(table, verb) for _file, _function, verb, table in NO_OP_EXEMPTIONS}
        return [write for write in self.writes
                if write[0] != AUTHORITY_DB and (write[2], write[1]) not in allowed]


@asynccontextmanager
async def booted(root: Path, *, verify_only: bool, overrides: dict | None = None, trace: Trace | None = None,
                 lifecycle: ProcessLifecycle | None = None, before_build=None):
    """main.build_application over the offline leaves of SC-0's child; the stack closes on exit."""
    import src.main as main_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("KZA_DASHBOARD_TOKEN", TOKEN)
        patch.setenv(MODEL_ENV, "synthetic-local-key")
        factories, _models = nominal_factories(patch, root)
        assert not construction_sdk_guards(patch)
        sdk_leaves(patch, root / "chroma")
        if before_build is not None:
            before_build(patch, main_module)
        async with offline_home() as (home_url, _home), local_models() as (model_url, _requests):
            config = configuration(root, home_url, model_url)
            config["dashboard"].update(enabled=True, host="127.0.0.1", port=_free_port())
            for dotted, value in (overrides or {}).items():
                *parents, leaf = dotted.split(".")
                owner = config
                for key in parents:
                    owner = owner.setdefault(key, {})
                owner[leaf] = value
            if trace is not None:
                trace.install(patch)
            stack = AsyncExitStack()
            try:
                graph = await main_module.build_application(config, shutdown=stack, factories=factories,
                                                            lifecycle=lifecycle, verify_only=verify_only)
                yield graph, stack
            finally:
                await stack.aclose()


async def _until(predicate, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """One normal first boot at the fixed path; each test restores this data/ tree there."""
    fixed = tmp_path_factory.mktemp("sc4a-fixed")
    state = fixed / "state"
    root = state / "data" / "graph"
    root.mkdir(parents=True)
    asyncio.run(_first_boot(root))
    saved = fixed / "template-data"
    shutil.copytree(state / "data", saved, symlinks=True)
    return {"state": state, "root": root, "saved": saved}


async def _first_boot(root: Path) -> None:
    """A normal managed-shape run to READY (audio held, as SC-0's child), then a normal stop."""
    lifecycle = ProcessLifecycle()
    async with booted(root, verify_only=False, lifecycle=lifecycle) as (graph, _stack):
        loop = graph.pipeline.audio_loop

        async def start():
            return None

        async def run():
            await asyncio.Event().wait()

        loop.start, loop.run = start, run
        task = asyncio.create_task(graph.run())
        await _until(lambda: lifecycle.stop.phase in (StopPhase.READY, StopPhase.CAPTURE_BLOCKED)
                     or task.done(), timeout=60)
        assert not task.done()
        await asyncio.sleep(1.0)
        lifecycle.stop.request("test")
        await asyncio.wait_for(task, 30)


def _restore(template: dict) -> tuple[Path, Path]:
    data = template["state"] / "data"
    if data.exists():
        stage.make_writable_then_remove(data)
    shutil.copytree(template["saved"], data, symlinks=True)
    return template["state"], template["root"]


def _diff(before: dict, after: dict) -> list:
    changes = []
    tb, ta = before["projection"]["tables"], after["projection"]["tables"]
    for database in sorted(set(tb) | set(ta)):
        b, a = tb.get(database, {}), ta.get(database, {})
        changes += [(database, table) for table in sorted(set(b) | set(a)) if b.get(table) != a.get(table)]
    for kind in ("accounts", "configuration"):
        xb, xa = before["projection"][kind], after["projection"][kind]
        changes += [(kind, name) for name in sorted(set(xb) | set(xa)) if xb.get(name) != xa.get(name)]
    return changes


async def _verify_run(root: Path, *, overrides=None, trace=None, settle: float = 1.0, plant=None,
                      before_build=None) -> dict:
    """Build in verify mode, run the verify branch to READY, hold, stop, pre_close, close, verify_closed."""
    lifecycle = ProcessLifecycle()
    async with booted(root, verify_only=True, overrides=overrides, trace=trace, lifecycle=lifecycle,
                      before_build=before_build) as (graph, stack):
        witness = RuntimeClosureWitness.capture(graph)
        graph.verify_run_id = "5c" * 16
        if plant is not None:
            plant(graph, stack, lifecycle)
        task = asyncio.create_task(graph.run())
        await _until(lambda: lifecycle.stop.phase is StopPhase.READY or task.done())
        assert not task.done(), task.exception() if task.done() else None
        await asyncio.sleep(settle)
        lifecycle.stop.request("test")
        await asyncio.wait_for(task, 30)
        await witness.pre_close()
    return {"graph": graph, "proof": witness.verify_closed()}


# --- structural pins ---------------------------------------------------------------------------------

def _loops(graph) -> dict:
    """The recurring owners of skip_list (a-*, d-diagnostic-repair, d-boot-retention-clamps loops)."""
    drivers = {participant: graph.erasure_services._drivers[participant].callback.__self__
               for participant in (ErasureParticipant.CAPTURES, ErasureParticipant.EVENTS,
                                   ErasureParticipant.SUGGESTIONS, ErasureParticipant.CONTEXTS,
                                   ErasureParticipant.DIAGNOSTICS, ErasureParticipant.SEMANTIC_INDEX)}
    memory = [reference().__self__ for reference in graph.privacy._borrowers
              if reference() is not None and type(reference().__self__).__name__ == "MemoryServices"]
    clips = graph.producer_coverage.collectors[0]._store
    return {
        "privacy maintenance": graph.privacy.maintenance,
        "voice registry maintain": getattr(graph.voice_registry, "_maintenance", None),
        "wake clip maintain": clips._maintenance,
        "diagnostics emitter": drivers[ErasureParticipant.DIAGNOSTICS].emitter._maintenance,
        "diagnostics history": drivers[ErasureParticipant.DIAGNOSTICS]._maintenance,
        "capture retention": drivers[ErasureParticipant.CAPTURES].retention._maintenance,
        "event retention": drivers[ErasureParticipant.EVENTS].retention._maintenance,
        "suggestion retention": drivers[ErasureParticipant.SUGGESTIONS].retention._maintenance,
        "context retention": drivers[ErasureParticipant.CONTEXTS].retention._maintenance,
        "index maintenance": drivers[ErasureParticipant.SEMANTIC_INDEX].maintenance._maintenance,
        "erasure coordinator": graph.erasure_services._maintenance,
        "live bus retention": graph.pipeline._event_bus._retention._maintenance,
        "memory maintenance": memory[0].maintenance,
    }


async def test_verify_mode_starts_no_listed_loop_and_a_normal_boot_starts_each(template):
    """Structural pin of every listed loop; the normal boot proves each attribute is the real owner."""
    _state, root = _restore(template)
    async with booted(root, verify_only=False) as (graph, _stack):
        await _until(lambda: all(task is not None for task in _loops(graph).values()))
        assert graph.verify_only is False and graph.privacy.ledger.verify_only is False
    _state, root = _restore(template)
    async with booted(root, verify_only=True) as (graph, _stack):
        await asyncio.sleep(0.5)
        assert {name: task for name, task in _loops(graph).items() if task is not None} == {}
        assert graph.verify_only is True and graph.privacy.verify_only and graph.privacy.ledger.verify_only
        assert graph.dashboard_authority._verify_only is True
        assert graph.capture_control.recorders.verify_only is True
        assert graph.privacy.ledger.retired_owner_prune_pending is False
        assert graph.privacy._retired_owner_prune is None  # b-retired-owner-prune: skipped, not deferred


async def test_verify_branch_reaches_no_ingress_or_house_effect_start(template):
    """DashboardAPI.start, ambient, audio, routines (and recover_interrupted), presence, reminders,
    HA sync, nightly trainer and the loop-lag probe are never started; VERIFY precedes READY."""
    _state, root = _restore(template)
    lifecycle = ProcessLifecycle()
    read_fd, write_fd = os.pipe()
    lifecycle.notifier._fd = write_fd
    called = []
    try:
        async with booted(root, verify_only=True, lifecycle=lifecycle) as (graph, _stack):
            def forbid(owner, name):
                async def forbidden(*args, **kwargs):
                    called.append(name)
                    raise AssertionError(name)
                if owner is not None:
                    setattr(owner, name, forbidden)

            forbid(graph, "_start_services")
            forbid(graph.dashboard, "start")
            forbid(graph.ambient_path, "start")
            forbid(graph.pipeline, "run")
            forbid(graph.pipeline.audio_loop, "start")
            forbid(graph.routine_scheduler, "start")
            forbid(graph.routine_scheduler.store, "recover_interrupted")
            forbid(graph.presence_detector, "start")
            forbid(graph.ha_client, "start_state_sync")
            graph.voice_services.start_reminders = lambda: called.append("start_reminders")
            if graph.nightly_trainer is not None:
                graph.nightly_trainer.start_scheduler = lambda: called.append("nightly")
            graph.loop_lag.start = lambda: called.append("loop_lag")
            graph.verify_run_id = "7a" * 16
            task = asyncio.create_task(graph.run())
            await _until(lambda: lifecycle.stop.phase is StopPhase.READY or task.done())
            lifecycle.stop.request("test")
            await asyncio.wait_for(task, 30)
        assert called == []
        assert os.read(read_fd, 4096).split(b"\n")[:2] == [b"VERIFY=" + b"7a" * 16, b"READY=1"]
    finally:
        os.close(read_fd)
        os.close(write_fd)


# --- residue canaries: a correct NEW gives K' == K (retire, not SS) -----------------------------------

@pytest.mark.parametrize("name", CANARIES)
async def test_residue_canary_leaves_k_prime_equal_to_k(template, name):
    state, root = _restore(template)
    planted = residue.STATES[name](state / "data", root)
    before = stage.capture_state_authority(state)
    trace = Trace()
    result = await _verify_run(root, overrides=planted.get("config"), trace=trace)
    after = stage.capture_state_authority(state)
    assert _diff(before, after) == []
    assert trace.unexpected() == []
    assert not any(verb.startswith("ALTER") for _db, verb, _table, _frame in trace.writes)  # schema_rule
    recorders = {"recorders_clean_stop_running": 2, "recorders_clean_stop_muted": 2,
                 "recorders_crash_running": 3, "recorders_crash_muted": 3, "recorder_orphan_unknown": 1}
    assert result["proof"]["capture_permissions_carried"] == recorders.get(name, 0)  # R5V-1: a count


async def test_the_authority_commits_nothing_in_verify_mode(template):
    """c-authority: initialize and the close-path final rotate roll back; file bytes unchanged,
    also with expired sessions/credentials and a retired user's credential present."""
    state, root = _restore(template)
    residue.expired_sessions_credentials(state / "data", root)
    residue.retired_user_credential_session(state / "data", root)
    database = root / AUTHORITY_DB
    sqlite3.connect(database).execute("PRAGMA wal_checkpoint(TRUNCATE)").close()
    witness = {path.name: path.read_bytes() for path in root.glob(AUTHORITY_DB + "*")}
    trace = Trace()
    await _verify_run(root, trace=trace)
    assert {path.name: path.read_bytes() for path in root.glob(AUTHORITY_DB + "*")} == witness
    rolled_back = {(verb, table) for db, verb, table, _frame in trace.writes if db == AUTHORITY_DB}
    assert ("INSERT OR REPLACE INTO", "auth_meta") in rolled_back  # the statements ran, then ROLLBACK


# --- backstops: the diff after the close catches what the list missed (R5V-4) --------------------------

def _protected_write(privacy) -> None:
    with privacy.ledger._transaction() as db:
        db.execute("UPDATE capture_runtime_control SET revision=revision+1 WHERE id=1")


async def test_a_protected_write_after_ready_in_a_construction_spawned_task_changes_k_prime(template):
    state, root = _restore(template)
    before = stage.capture_state_authority(state)

    def plant(graph, stack, lifecycle):
        async def late():
            await _until(lambda: lifecycle.stop.phase is StopPhase.READY)
            await graph.privacy.acall(_protected_write, graph.privacy)
        task = asyncio.create_task(late())
        stack.push_async_callback(asyncio.gather, task, return_exceptions=True)

    await _verify_run(root, plant=plant)
    assert ("data/graph/privacy.db", "capture_runtime_control") in _diff(before, stage.capture_state_authority(state))


async def test_a_protected_write_in_an_exit_stack_callback_changes_k_prime(template):
    state, root = _restore(template)
    before = stage.capture_state_authority(state)

    def plant(graph, stack, lifecycle):
        async def at_close():
            await graph.privacy.acall(_protected_write, graph.privacy)
        stack.push_async_callback(at_close)

    await _verify_run(root, plant=plant)
    assert ("data/graph/privacy.db", "capture_runtime_control") in _diff(before, stage.capture_state_authority(state))


async def test_an_exempt_statement_that_changes_a_row_is_caught(template):
    """no_op_exemption: an H without the default ambient control row gets it in V; K' != K, so retire's
    strict equality refuses (abort with W or SS, never acceptance)."""
    state, root = _restore(template)
    with sqlite3.connect(root / "privacy.db") as db:
        db.execute("DELETE FROM ambient_runtime_control")
    before = stage.capture_state_authority(state)
    trace = Trace()
    await _verify_run(root, trace=trace)
    assert ("data/graph/privacy.db", "ambient_runtime_control") in _diff(before, stage.capture_state_authority(state))
    assert ("privacy.db", "INSERT OR IGNORE INTO", "ambient_runtime_control") in {
        write[:3] for write in trace.writes}


def test_no_op_exemptions_are_the_pinned_sites():
    """Each pinned site exists and inserts only when absent (OR IGNORE, or behind its own guard)."""
    import ast

    for file, function, verb, table in NO_OP_EXEMPTIONS:
        source = (REPO / file).read_text()
        names = {node.name for node in ast.walk(ast.parse(source))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert function in names, (file, function)
        assert re.search(verb.replace(" ", r"\s+") + r"\s+" + table, source, re.IGNORECASE), (file, table)
    assert {table for *_rest, table in NO_OP_EXEMPTIONS} <= set(PROJECTED)
