"""Policy proposals with real event receipts, guarded commits and late workers."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analytics.event_storage import event_descriptor, sweep_events
from src.analytics.event_logger import EventLogger
from src.analytics.pattern_analyzer import PatternAnalyzer, PatternType
from src.analytics.suggestion_engine import SuggestionEngine, SuggestionStatus
from src.memory.fact_ledger import FactDependency
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices
from src.pipeline.request_router import RequestRouter
from scripts.smart_automations import run_command
from tests.integration.test_pattern_evidence import (
    ACTOR,
    DAY,
    START,
    Clock,
    graph,
    seed_days,
)
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


async def engine_for(tmp_path, events, *, llm=None, ha=None):
    return await events.acall(
        SuggestionEngine,
        events,
        suggestions_path=str(tmp_path / "suggestions.json"),
        llm_reasoner=llm,
        ha_client=ha,
    )


def count(engine):
    with sqlite3.connect(engine.store.path) as db:
        return db.execute("SELECT COUNT(*) FROM suggestions_policy_v1").fetchone()[0]


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def generate(tmp_path, events, clock, **kwargs):
    sources = await seed_days(events, clock)
    engine = await engine_for(tmp_path, events, **kwargs)
    suggestions = await events.acall(engine.generate_suggestions)
    assert suggestions
    return engine, suggestions, sources


async def test_legacy_json_stays_untouched_and_never_enters_active_proposals(tmp_path):
    legacy = tmp_path / "suggestions.json"
    original = b'{"suggestions":[{"message":"synthetic unsourced legacy"}]}'
    legacy.write_bytes(original)
    before = legacy.stat()
    async with graph(tmp_path) as (privacy, events, clock):
        engine = await engine_for(tmp_path, events)
        assert await events.acall(engine.get_pending_suggestions) == []
        await seed_days(events, clock)
        suggestions = await events.acall(engine.generate_suggestions)
        assert suggestions and len({s.id for s in suggestions}) == len(suggestions)
        assert (
            legacy.read_bytes() == original
            and legacy.stat().st_mtime_ns == before.st_mtime_ns
        )
        assert engine.store.path.stat().st_mode & 0o777 == 0o600
        targets = await privacy.acall(
            privacy.ledger.erasures.targets, ErasureParticipant.SUGGESTIONS
        )
        assert any(t["descriptor"] == engine.store.descriptor for t in targets)
        assert "_suggestions" not in vars(engine)


async def test_restart_reads_and_expanded_configuration_do_not_renew_proposals(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, events, _):
        engine, proposals, _ = await generate(tmp_path, events, clock)
        dates = {s.id: (s.created_at, s.expires_at) for s in proposals}
        assert all(s.expires_at == START + 34 * DAY for s in proposals)
        clock.now += DAY
        assert {
            s.id: (s.created_at, s.expires_at)
            for s in await events.acall(engine.get_pending_suggestions)
        } == dates
    async with graph(tmp_path, clock=clock, days=180) as (_, events, _):
        engine = await engine_for(tmp_path, events)
        assert {
            s.id: (s.created_at, s.expires_at)
            for s in await events.acall(engine.get_pending_suggestions)
        } == dates
        clock.now = START + 34 * DAY
        assert await events.acall(engine.get_pending_suggestions) == []
        assert await events.acall(engine.generate_suggestions) == []
        assert count(engine) == len(
            proposals
        )  # Physical independent expiry is a later participant.


async def test_valid_metadata_can_generate_after_raw_scrubbing_without_copying_raw_content(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        await seed_days(events, clock)
        clock.now += 3 * DAY
        await privacy.acall(
            sweep_events, event_descriptor(events.db_path), 0, privacy.ledger, 90
        )
        engine = await engine_for(tmp_path, events)
        suggestions = await events.acall(engine.generate_suggestions)
        assert suggestions
        with sqlite3.connect(engine.store.path) as db:
            for row in db.execute(
                "SELECT message,pattern_json,evidence_json,sources_json FROM suggestions_policy_v1"
            ):
                encoded = " ".join(row)
                for secret in (
                    "synthetic private transcript",
                    "synthetic name",
                    "synthetic private context",
                ):
                    assert secret not in encoded


async def test_conversation_revocation_excludes_all_projections_before_physical_cleanup(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        engine, proposals, sources = await generate(tmp_path, events, clock)
        router = RequestRouter(None, None, None, suggestion_engine=engine)
        router._pending_suggestion = proposals[0]
        await privacy.acall(
            privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
        )
        assert not engine.available_now(proposals[0])
        assert await events.acall(engine.get_pending_suggestions) == []
        assert (await events.acall(engine.get_stats))["total_suggestions"] == 0
        assert not (
            await events.acall(engine.respond_to_suggestion, proposals[0].id, True)
        )["success"]
        router.prune_private()
        assert router._pending_proposal is None
        assert count(engine) == len(proposals)
        await router.aclose()


async def test_fact_receipts_constrain_expiry_and_reject_a_later_revision(tmp_path):
    async with graph(tmp_path) as (privacy, events, clock):
        fact = await privacy.acall(
            privacy.ledger.store,
            "synthetic fact",
            "fact",
            0.8,
            (memory_source(now=clock.now),),
        )
        for day in range(5):
            clock.now = START + day * DAY
            source = replace(
                memory_source(now=clock.now),
                fact_dependencies=(
                    FactDependency(fact.id, fact.revision, fact.expires_at),
                ),
            )
            await events.acall(
                events.log, "light.synthetic", "turn_on", sources=(source,)
            )
        engine = await engine_for(tmp_path, events)
        proposals = await events.acall(engine.generate_suggestions)
        assert proposals and all(
            s.expires_at == fact.expires_at == START + 30 * DAY for s in proposals
        )
        await privacy.acall(privacy.ledger.delete_fact, fact.id, ACTOR)
        assert await events.acall(engine.get_pending_suggestions) == []
        assert all(not engine.available_now(s) for s in proposals)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sources_json", "[]"),
        ("evidence_json", "[]"),
        ("pattern_json", "{}"),
        ("expires_at", START + 100 * DAY),
        ("observed_at", START + 3 * DAY),
        ("status", "implemented"),
        ("snooze_until", START + 100 * DAY),
    ],
)
async def test_unknown_or_forged_policy_metadata_is_excluded_and_preserved(
    tmp_path, field, value
):
    async with graph(tmp_path) as (_, events, clock):
        engine, proposals, _ = await generate(tmp_path, events, clock)
        with sqlite3.connect(engine.store.path) as db:
            db.execute("UPDATE suggestions_policy_v1 SET " + field + "=?", (value,))
        assert await events.acall(engine.get_pending_suggestions) == []
        assert count(engine) == len(proposals)


async def test_missing_event_rows_cannot_be_replaced_by_self_reported_lineage(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        engine, proposals, _ = await generate(tmp_path, events, clock)
        with sqlite3.connect(events.db_path) as db:
            db.execute("DELETE FROM events_policy_v1 WHERE id=1")
        assert await events.acall(engine.get_pending_suggestions) == []
        assert not (
            await events.acall(engine.respond_to_suggestion, proposals[0].id, True)
        )["success"]
        assert count(engine) == len(proposals)


async def test_untrusted_pattern_descriptions_and_statistics_are_rebuilt_before_model_prompt(
    tmp_path,
):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        pattern.description, pattern.confidence, pattern.occurrences = (
            "synthetic injected private transcript",
            1,
            9999,
        )
        llm = MagicMock()
        llm.generate.return_value = "Propuesta basada en actividad sintética."
        engine = await engine_for(tmp_path, events, llm=llm)
        engine.pattern_analyzer = SimpleNamespace(
            get_actionable_patterns=lambda _: [pattern]
        )
        proposals = await events.acall(engine.generate_suggestions)
        assert len(proposals) == 1 and proposals[0].pattern.occurrences == 5
        assert (
            "synthetic injected private transcript"
            not in llm.generate.call_args.args[0]
        )
        assert "9999" not in llm.generate.call_args.args[0]


async def test_receipts_cannot_authorize_another_entity_or_an_invalid_pattern(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        pattern.entity_id = "light.unobserved"
        llm = MagicMock()
        engine = await engine_for(tmp_path, events, llm=llm)
        engine.pattern_analyzer = SimpleNamespace(
            get_actionable_patterns=lambda _: [pattern]
        )
        assert await events.acall(engine.generate_suggestions) == []
        llm.generate.assert_not_called()
        assert count(engine) == 0


async def test_recomputed_confidence_must_meet_requested_threshold(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        pattern.confidence = 1
        llm = MagicMock()
        engine = await engine_for(tmp_path, events, llm=llm)
        engine.pattern_analyzer = SimpleNamespace(
            get_actionable_patterns=lambda _: [pattern]
        )
        assert (
            await events.acall(engine.generate_suggestions, min_confidence=0.95) == []
        )
        llm.generate.assert_not_called()
        assert count(engine) == 0


@pytest.mark.parametrize("invalidate", ["marker", "expiry"])
async def test_late_model_output_cannot_commit_after_its_source_is_invalidated(
    tmp_path, invalidate
):
    async with graph(tmp_path) as (privacy, events, clock):
        sources = await seed_days(events, clock)
        started, release = threading.Event(), threading.Event()

        def model(*args, **kwargs):
            started.set()
            assert release.wait(3)
            return "synthetic late wording"

        engine = await engine_for(tmp_path, events, llm=SimpleNamespace(generate=model))
        task = asyncio.create_task(events.acall(engine.generate_suggestions))
        try:
            await until(started.is_set)
            if invalidate == "marker":
                await privacy.acall(
                    privacy.ledger.delete_conversation,
                    sources[0].conversation_id,
                    ACTOR,
                )
            else:
                clock.now = START + 100 * DAY
            release.set()
            assert await task == []
            assert count(engine) == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_model_wait_does_not_renew_creation_or_observation_deadline(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)

        def model(*args, **kwargs):
            clock.now += DAY
            return "synthetic delayed wording"

        engine = await engine_for(tmp_path, events, llm=SimpleNamespace(generate=model))
        proposals = await events.acall(engine.generate_suggestions, max_suggestions=1)
        assert proposals[0].created_at == START + 4 * DAY
        assert proposals[0].expires_at == START + 34 * DAY


async def test_native_proposal_commit_precedes_marker_or_is_rejected(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, events, clock):
        sources = await seed_days(events, clock)
        engine = await engine_for(tmp_path, events)
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        prepared = await events.acall(engine.store.prepare, pattern)
        started, marker_started, release = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = engine.store._connect

        @contextmanager
        def paused():
            with original() as db:
                yield db
                started.set()  # Real INSERT has run, but SQLite has not committed.
                assert release.wait(3)

        monkeypatch.setattr(engine.store, "_connect", paused)
        writer = asyncio.create_task(
            events.acall(engine.store.create, prepared, "synthetic proposal", clock.now)
        )
        marker = None
        try:
            await until(started.is_set)

            def erase():
                marker_started.set()
                return privacy.ledger.delete_conversation(
                    sources[0].conversation_id, ACTOR
                )

            marker = asyncio.create_task(privacy.acall(erase))
            await until(marker_started.is_set)
            assert not marker.done()
            assert not privacy.ledger.conversation_revoked(sources[0].conversation_id)
            assert count(engine) == 0
            release.set()
            record = await writer
            await marker
            assert record is not None and count(engine) == 1
            assert await events.acall(engine.get_pending_suggestions) == []
            with pytest.raises(PermissionError):
                await events.acall(
                    engine.store.create, prepared, "synthetic refill", clock.now
                )
        finally:
            release.set()
            await asyncio.gather(
                *(t for t in (writer, marker) if t is not None), return_exceptions=True
            )


async def test_concurrent_instances_cannot_duplicate_a_live_pattern(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        one, two = (
            await engine_for(tmp_path, events),
            await engine_for(tmp_path, events),
        )
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        results = await asyncio.gather(
            events.acall(one.store.create, pattern, "synthetic one", clock.now),
            events.acall(two.store.create, pattern, "synthetic two", clock.now),
        )
        assert sum(record is not None for record in results) == count(one) == 1


async def test_pending_failed_marker_hides_proposals_without_destroying_or_renewing_them(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        engine, proposals, sources = await generate(tmp_path, events, clock)
        entered, release = threading.Event(), threading.Event()
        original = privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic failed marker")

        privacy.ledger._db = FailedCommit()
        task = asyncio.create_task(
            privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
        )
        try:
            await until(entered.is_set)
            assert all(not engine.available_now(s) for s in proposals)
            assert await events.acall(engine.get_pending_suggestions) == []
            assert count(engine) == len(proposals)
        finally:
            release.set()
            with pytest.raises(OSError, match="synthetic failed marker"):
                await task
            privacy.ledger._db = original
        current = await events.acall(engine.get_pending_suggestions)
        assert {s.id: s.expires_at for s in current} == {
            s.id: s.expires_at for s in proposals
        }


async def test_accepting_is_a_proposal_response_and_never_claims_stub_ha_implementation(
    tmp_path,
):
    async with graph(tmp_path) as (_, events, clock):
        ha = MagicMock()
        engine, proposals, _ = await generate(tmp_path, events, clock, ha=ha)
        result = await events.acall(engine.respond_to_suggestion, proposals[0].id, True)
        assert (
            result["success"] and "borrador" in result["message"] and "yaml" in result
        )
        ha.assert_not_called()
        assert not ha.method_calls
        accepted = (
            await events.acall(engine.store.records, suggestion_id=proposals[0].id)
        )[0]
        assert accepted.status is SuggestionStatus.ACCEPTED
        assert accepted.expires_at == proposals[0].expires_at
        assert not (
            await events.acall(engine.respond_to_suggestion, proposals[0].id, True)
        )["success"]


async def test_snooze_survives_restart_and_is_capped_by_original_expiry(tmp_path):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, events, _):
        engine, proposals, _ = await generate(tmp_path, events, clock)
        proposal = proposals[0]
        result = await events.acall(
            engine.respond_to_suggestion, proposal.id, False, 10000
        )
        assert result["success"]
        row = (await events.acall(engine.store.records, suggestion_id=proposal.id))[0]
        assert row.snooze_until == row.expires_at == proposal.expires_at
    async with graph(tmp_path, clock=clock) as (_, events, _):
        engine = await engine_for(tmp_path, events)
        row = (await events.acall(engine.store.records, suggestion_id=proposal.id))[0]
        assert row.snooze_until == proposal.expires_at
        assert proposal.id not in {
            s.id for s in await events.acall(engine.get_pending_suggestions)
        }
        clock.now = proposal.expires_at
        assert await events.acall(engine.get_pending_suggestions) == []


async def test_voice_delivery_stops_on_metadata_revocation_and_discards_idle_pending(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        engine, _, sources = await generate(tmp_path, events, clock)
        speaking, stopped = asyncio.Event(), asyncio.Event()

        async def speak(*args, **kwargs):
            speaking.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        router = RequestRouter(
            None, SimpleNamespace(aspeak=speak), None, suggestion_engine=engine
        )
        observer = asyncio.create_task(router._maybe_present_suggestion())
        try:
            await speaking.wait()
            assert router._pending_suggestion is not None
            await privacy.acall(
                privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
            )
            async with asyncio.timeout(1):
                await observer
            assert stopped.is_set() and router._pending_suggestion is None
        finally:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()


async def test_cancelled_late_model_keeps_worker_and_lease_until_actual_completion(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        await seed_days(events, clock)
        started, release = threading.Event(), threading.Event()

        def model(*args, **kwargs):
            started.set()
            assert release.wait(3)
            return "synthetic late model"

        engine = await engine_for(tmp_path, events, llm=SimpleNamespace(generate=model))
        observer = asyncio.create_task(events.acall(engine.generate_suggestions))
        contender = PrivacyServices(privacy.ledger.path, clock=clock)
        try:
            await until(started.is_set)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            closing = privacy.begin_close()
            await asyncio.sleep(0)
            assert not closing.done() and privacy.resource_status()["writer_lease"]
            with pytest.raises(RuntimeError, match="privacy_writer_already_active"):
                await contender.acall(contender.initialize)
            release.set()
            await privacy.aclose()
            assert count(engine) == 0
        finally:
            release.set()
            await contender.aclose()


async def test_midnight_pattern_exports_midnight_instead_of_default_noon(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        for day in range(5):
            clock.now = START + day * DAY - 7 * 3600
            await events.acall(events.log, "light.synthetic", "turn_on")
        engine = await engine_for(tmp_path, events)
        proposals = await events.acall(engine.generate_suggestions)
        daily = next(
            p for p in proposals if p.pattern.pattern_type is PatternType.DAILY_TIME
        )
        assert 'at: "00:00:00"' in daily.automation_yaml


async def test_expired_or_revoked_metadata_never_loads_stored_message(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (privacy, events, clock):
        engine, _, sources = await generate(tmp_path, events, clock)
        original = engine.store._connect
        statements = []

        @contextmanager
        def observed():
            with original() as db:
                db.set_trace_callback(statements.append)
                yield db

        monkeypatch.setattr(engine.store, "_connect", observed)
        assert await events.acall(engine.get_pending_suggestions)
        assert any("SELECT message" in statement for statement in statements)
        statements.clear()
        await privacy.acall(
            privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
        )
        assert await events.acall(engine.get_pending_suggestions) == []
        assert not any("SELECT message" in statement for statement in statements)


async def test_event_mutation_waits_until_derivative_commit_and_then_invalidates_it(
    tmp_path, monkeypatch
):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        engine = await engine_for(tmp_path, events)
        pattern = (await events.acall(PatternAnalyzer(events).analyze_all))[0]
        started, editing, release = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = engine.store._connect

        @contextmanager
        def paused():
            with original() as db:
                yield db
                started.set()
                assert release.wait(3)

        monkeypatch.setattr(engine.store, "_connect", paused)
        writer = asyncio.create_task(
            events.acall(engine.store.create, pattern, "synthetic proposal", clock.now)
        )
        editor = None

        def edit():
            with sqlite3.connect(events.db_path) as db:
                db.set_trace_callback(
                    lambda sql: editing.set() if sql == "BEGIN IMMEDIATE" else None
                )
                db.execute("BEGIN IMMEDIATE")
                db.execute("UPDATE events_policy_v1 SET action='turn_off' WHERE id=1")

        try:
            await until(started.is_set)
            editor = asyncio.create_task(asyncio.to_thread(edit))
            await until(editing.is_set)
            assert not editor.done()
            release.set()
            assert await writer is not None
            await editor
            assert await events.acall(engine.get_pending_suggestions) == []
        finally:
            release.set()
            await asyncio.gather(
                *(t for t in (writer, editor) if t is not None), return_exceptions=True
            )


@pytest.mark.parametrize("failure", ["oversized", "error"])
async def test_optional_model_failure_falls_back_without_persisting_or_logging_private_error(
    tmp_path, caplog, failure
):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)

        def model(*args, **kwargs):
            if failure == "oversized":
                return "synthetic model overflow " * 1000
            raise RuntimeError("synthetic private model error")

        engine = await engine_for(tmp_path, events, llm=SimpleNamespace(generate=model))
        proposals = await events.acall(engine.generate_suggestions)
        assert proposals and all(len(s.message) <= 4096 for s in proposals)
        assert "synthetic private model error" not in caplog.text
        assert all("synthetic model overflow" not in s.message for s in proposals)


async def test_policy_paths_are_pinned_and_each_actual_destination_is_registered(
    tmp_path,
):
    async with graph(tmp_path) as (privacy, events, clock):
        await seed_days(events, clock)
        first, second, alias = (
            tmp_path / "first",
            tmp_path / "second",
            tmp_path / "alias",
        )
        first.mkdir()
        second.mkdir()
        alias.symlink_to(first, target_is_directory=True)
        one = await events.acall(
            SuggestionEngine, events, suggestions_path=str(alias / "suggestions.json")
        )
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        assert await events.acall(one.generate_suggestions)
        assert count(one) == 2 and not (second / one.store.path.name).exists()
        two = await events.acall(
            SuggestionEngine, events, suggestions_path=str(alias / "suggestions.json")
        )
        assert await events.acall(two.generate_suggestions)
        targets = await privacy.acall(
            privacy.ledger.erasures.targets, ErasureParticipant.SUGGESTIONS
        )
        assert {t["descriptor"]["path"] for t in targets} == {
            str(one.store.path),
            str(two.store.path),
        }


async def test_conflicting_policy_and_event_locations_are_rejected_before_writer_deadlock(
    tmp_path,
):
    async with graph(tmp_path, name="suggestions.json.policy-v1.sqlite3") as (
        privacy,
        events,
        _,
    ):
        with pytest.raises(ValueError, match="suggestion_store_location_conflict"):
            await engine_for(tmp_path, events)
        conflicting = EventLogger(privacy.ledger.path, source_policy=privacy.ledger)
        try:
            with pytest.raises(
                ValueError, match="event_store_authority_location_conflict"
            ):
                await conflicting.acall(conflicting.initialize)
        finally:
            await conflicting.aclose()


def cli_args():
    return SimpleNamespace(
        command="suggestions", action="list", db_path=None, suggestions_path=None
    )


async def test_shorter_event_policy_applies_before_sweep_and_cannot_expand_on_next_restart(
    tmp_path,
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, events, _):
        engine, proposals, _ = await generate(tmp_path, events, clock)
        clock.now += 3 * DAY
        assert await events.acall(engine.get_pending_suggestions)
    # No periodic sweep runs here: consumers cannot rely on its scheduling.
    async with graph(tmp_path, clock=clock, days=1) as (_, events, _):
        engine = await engine_for(tmp_path, events)
        assert await events.acall(engine.get_pending_suggestions) == []
        assert (await events.acall(events.get_pattern_history)).events == ()
        with sqlite3.connect(events.db_path) as db:
            deadlines = db.execute(
                "SELECT timestamp,expires_at FROM events_policy_v1"
            ).fetchall()
        assert all(expiry == timestamp + DAY for timestamp, expiry in deadlines)
        assert count(engine) == len(proposals)
    async with graph(tmp_path, clock=clock, days=90) as (_, events, _):
        engine = await engine_for(tmp_path, events)
        assert await events.acall(engine.get_pending_suggestions) == []
        assert (await events.acall(events.get_pattern_history)).events == ()


async def test_startup_retention_clamp_preserves_unknown_event_sources(tmp_path):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (_, events, _):
        await seed_days(events, clock)
        with sqlite3.connect(events.db_path) as db:
            db.execute(
                "UPDATE events_policy_v1 SET sources_json='unknown protocol' WHERE id=1"
            )
        clock.now += 3 * DAY
    async with graph(tmp_path, clock=clock, days=1) as (_, events, _):
        assert (await events.acall(events.get_pattern_history)).events == ()
        with sqlite3.connect(events.db_path) as db:
            unknown = db.execute(
                "SELECT sources_json,expires_at FROM events_policy_v1 WHERE id=1"
            ).fetchone()
            known = db.execute(
                "SELECT timestamp,expires_at FROM events_policy_v1 WHERE id>1"
            ).fetchall()
        assert unknown == ("unknown protocol", START + 90 * DAY)
        assert all(expiry == timestamp + DAY for timestamp, expiry in known)


def cli_config(tmp_path):
    return {
        "memory": {"enabled": False, "privacy_db_path": str(tmp_path / "privacy.db")},
        "analytics": {
            "enabled": False,
            "events_db": str(tmp_path / "events.db"),
            "suggestions_path": str(tmp_path / "suggestions.json"),
        },
    }


async def test_cli_uses_same_configured_authority_and_filters_revoked_proposals(
    tmp_path, capsys
):
    clock = Clock()
    async with graph(tmp_path, clock=clock) as (privacy, events, _):
        engine, proposals, sources = await generate(tmp_path, events, clock)
        assert count(engine) == len(proposals)
        await privacy.acall(
            privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR
        )
    args = cli_args()
    capsys.readouterr()
    assert await run_command(args, config=cli_config(tmp_path), clock=clock) == 0
    text = capsys.readouterr().out
    assert "Total: 0" in text and "No hay sugerencias pendientes" in text
    assert "He notado" not in text
    assert not hasattr(args, "event_logger")


async def test_cli_cannot_open_an_alternate_writer_while_runtime_lease_is_live(
    tmp_path,
):
    async with graph(tmp_path) as (_, _, clock):
        config = cli_config(tmp_path)
        config["analytics"]["events_db"] = str(tmp_path / "must-not-be-created.db")
        with pytest.raises(RuntimeError, match="privacy_writer_already_active"):
            await run_command(cli_args(), config=config, clock=clock)
        assert not (tmp_path / "must-not-be-created.db").exists()


async def test_optional_store_read_failure_does_not_fail_completed_command_or_expose_error(
    tmp_path, monkeypatch, caplog
):
    async with graph(tmp_path) as (_, events, clock):
        engine, _, _ = await generate(tmp_path, events, clock)

        def fail(*args, **kwargs):
            raise OSError("synthetic private read error")

        monkeypatch.setattr(engine.store, "records", fail)
        speaker = SimpleNamespace(aspeak=AsyncMock())
        router = RequestRouter(None, speaker, None, suggestion_engine=engine)
        try:
            await router._maybe_present_suggestion()
            speaker.aspeak.assert_not_awaited()
            assert router._pending_suggestion is None
            assert "synthetic private read error" not in caplog.text
        finally:
            await router.aclose()


async def test_voice_response_write_failure_is_reported_without_claiming_acceptance(
    tmp_path, monkeypatch, caplog
):
    async with graph(tmp_path) as (_, events, clock):
        engine, proposals, _ = await generate(tmp_path, events, clock)

        def fail(*args, **kwargs):
            raise OSError("synthetic private write error")

        monkeypatch.setattr(engine.store, "respond", fail)
        router = RequestRouter(None, None, None, suggestion_engine=engine)
        router._pending_suggestion = proposals[0]
        try:
            result = await router._handle_suggestion_response("si")
            assert result["handled"] and not result["success"]
            assert "No pude registrar" in result["response"]
            assert router._pending_suggestion is None
            assert "synthetic private write error" not in caplog.text
            assert (
                await events.acall(engine.store.records, suggestion_id=proposals[0].id)
            )[0].status is SuggestionStatus.PENDING
        finally:
            await router.aclose()
