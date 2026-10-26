"""Real activity receipts, complete denominators and source-aware pattern analysis."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from src.analytics.event_evidence import EventDependency
from src.analytics.event_logger import EventLogger
from src.analytics.event_storage import event_descriptor, sweep_events
from src.analytics.pattern_analyzer import PatternAnalyzer, PatternType
from src.memory.fact_ledger import FactDependency, MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.pipeline.request_router import RequestRouter
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
DAY = 86400
START = datetime(2026, 10, 5, 7).timestamp()  # Monday, in the process timezone.
ACTOR = MemoryReviewer("synthetic-admin")


@dataclass
class Clock:
    now: float = START

    def __call__(self):
        return self.now


@asynccontextmanager
async def graph(tmp_path, *, clock=None, days=90, name="events.db"):
    clock = clock or Clock()
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        events = EventLogger(
            str(tmp_path / name), source_policy=privacy.ledger, retention_days=days
        )
        stack.push_async_callback(events.aclose)
        privacy.add_borrower(events.begin_close)
        await events.acall(events.initialize)
        yield privacy, events, clock


async def seed_days(events, clock, *, count=5):
    sources = []
    for day in range(count):
        clock.now = START + day * DAY
        source = memory_source(now=clock.now)
        sources.append(source)
        await events.acall(
            events.log, "light.synthetic", "turn_on", sources=(source,),
            trigger_phrase="synthetic private transcript", user_name="synthetic name",
            extra_context={"private": "synthetic private context"},
        )
    return sources


def row_count(events):
    with sqlite3.connect(events.db_path) as db:
        return db.execute("SELECT COUNT(*) FROM events_policy_v1").fetchone()[0]


async def test_multiday_patterns_use_metadata_without_loading_or_retaining_raw_text(tmp_path, monkeypatch):
    async with graph(tmp_path) as (_, events, clock):
        sources = await seed_days(events, clock)
        clock.now += 3 * DAY
        # The analyzer must never regress to the Event DTO containing raw text.
        def forbidden(*args, **kwargs):
            raise AssertionError("raw event retrieval entered pattern analysis")

        monkeypatch.setattr(events, "get_events", forbidden)
        monkeypatch.setattr(events, "get_entity_history", forbidden)
        monkeypatch.setattr(events, "get_stats", forbidden)
        monkeypatch.setattr(events, "get_sequences", forbidden)
        patterns = await events.acall(PatternAnalyzer(events).analyze_all)
        assert patterns and all(p.evidence for p in patterns)
        assert all(p.expires_at == START + 90 * DAY for p in patterns)
        history = await events.acall(events.get_pattern_history)
        assert len(history.events) == 5
        encoded = json.dumps([d.to_dict() for d in patterns[0].evidence])
        for secret in ["synthetic", str(tmp_path), *(s.capture_id for s in sources), *(s.conversation_id for s in sources)]:
            assert secret not in encoded
        assert all(not hasattr(e, "trigger_phrase") and not hasattr(e, "sources") for e in history.events)


async def test_raw_scrubbing_keeps_receipts_but_expiry_and_restart_never_renew_them(tmp_path):
    clock = Clock()
    async with graph(tmp_path, clock=clock, days=4) as (privacy, events, _):
        source = memory_source(now=clock.now)
        await events.acall(events.log, "light.synthetic", "turn_on", sources=(source,), trigger_phrase="synthetic raw")
        history = await events.acall(events.get_pattern_history)
        evidence = tuple(e.dependency for e in history.events)
        clock.now += 2 * DAY
        await privacy.acall(sweep_events, event_descriptor(events.db_path), 0, privacy.ledger, 4)
        assert await events.acall(events.pattern_evidence_current, evidence)
        assert (await events.acall(events.get_pattern_history)).events == history.events
        with sqlite3.connect(events.db_path) as db:
            assert db.execute("SELECT trigger_phrase FROM events_policy_v1").fetchone() == (None,)
    async with graph(tmp_path, clock=clock, days=90) as (_, events, _):
        assert await events.acall(events.pattern_evidence_current, evidence)
        clock.now = START + 4 * DAY
        assert not await events.acall(events.pattern_evidence_current, evidence)
        assert (await events.acall(events.get_pattern_history)).events == ()
        assert row_count(events) == 1  # Read filtering precedes physical deletion.


async def test_conversation_marker_invalidates_receipts_before_event_physical_cleanup(tmp_path):
    async with graph(tmp_path) as (privacy, events, clock):
        sources = await seed_days(events, clock)
        patterns = await events.acall(PatternAnalyzer(events).analyze_all)
        evidence = patterns[0].evidence
        assert await events.acall(events.pattern_evidence_current, evidence)
        await privacy.acall(privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR)
        assert row_count(events) == 5
        assert not await events.acall(events.pattern_evidence_current, evidence)


@pytest.mark.parametrize("invalidate", ["delete", "expiry"])
async def test_fact_revisions_and_original_fact_deadlines_constrain_activity_receipts(tmp_path, invalidate):
    async with graph(tmp_path) as (privacy, events, clock):
        source = memory_source(now=clock.now)
        fact = await privacy.acall(privacy.ledger.store, "synthetic fact", "fact", .8, (source,))
        derived = replace(memory_source(now=clock.now), fact_dependencies=(FactDependency(fact.id, fact.revision, fact.expires_at),))
        await events.acall(events.log, "light.synthetic", "turn_on", sources=(derived,))
        evidence = tuple(e.dependency for e in (await events.acall(events.get_pattern_history)).events)
        assert evidence[0].expires_at == fact.expires_at == START + 30 * DAY
        clock.now += 3 * DAY
        assert await events.acall(events.pattern_evidence_current, evidence)
        if invalidate == "delete":
            await privacy.acall(privacy.ledger.delete_fact, fact.id, ACTOR)
        else:
            clock.now = fact.expires_at
        assert not await events.acall(events.pattern_evidence_current, evidence)
        assert row_count(events) == 1


@pytest.mark.parametrize("change", ["entity_id", "action", "timestamp", "expires_at", "sources", "deleted"])
async def test_changed_or_missing_activity_never_resolves_a_stale_receipt(tmp_path, change):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock, count=1)
        evidence = tuple(e.dependency for e in (await events.acall(events.get_pattern_history)).events)
        with sqlite3.connect(events.db_path) as db:
            if change == "sources":
                db.execute("UPDATE events_policy_v1 SET sources_json=?", (json.dumps([memory_source(now=START).to_dict()]),))
            elif change == "deleted":
                db.execute("DELETE FROM events_policy_v1")
            else:
                values = {"entity_id": "light.other", "action": "turn_off", "timestamp": START - 1, "expires_at": START + 91 * DAY}
                db.execute("UPDATE events_policy_v1 SET " + change + "=?", (values[change],))
        assert not await events.acall(events.pattern_evidence_current, evidence)


async def test_receipts_bind_the_actual_store_even_when_row_ids_and_metadata_match(tmp_path):
    async with graph(tmp_path) as (privacy, events, clock):
        source = memory_source(now=clock.now)
        await events.acall(events.log, "light.synthetic", "turn_on", sources=(source,))
        evidence = tuple(e.dependency for e in (await events.acall(events.get_pattern_history)).events)
        other = EventLogger(str(tmp_path / "other.db"), source_policy=privacy.ledger)
        try:
            await other.acall(other.log, "light.synthetic", "turn_on", sources=(source,))
            assert not await other.acall(other.pattern_evidence_current, evidence)
            own = tuple(e.dependency for e in (await other.acall(other.get_pattern_history)).events)
            assert own[0].event_id == evidence[0].event_id
            assert own[0].revision == evidence[0].revision
            assert own[0].target_id != evidence[0].target_id
        finally:
            await other.aclose()


@pytest.mark.parametrize("mutation", [
    {"event_id": True}, {"event_id": 0}, {"event_id": 2**63},
    {"target_id": "unknown"}, {"revision": "g" * 64},
    {"expires_at": float("nan")}, {"expires_at": True},
    {"extra": "synthetic"},
])
async def test_persisted_receipt_protocol_rejects_unknown_values(mutation):
    encoded = EventDependency("a" * 64, 1, "b" * 64, START + DAY).to_dict()
    encoded.update(mutation)
    with pytest.raises(ValueError, match="invalid_event_dependency"):
        EventDependency.from_dict(encoded)


async def test_legacy_rows_and_incomplete_windows_cannot_supply_pattern_confidence(tmp_path, monkeypatch):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        with sqlite3.connect(events.db_path) as db:
            db.execute("CREATE TABLE events (transcript TEXT)")
            db.execute("INSERT INTO events VALUES ('synthetic legacy')")
        history = await events.acall(events.get_pattern_history, limit=4)
        assert not history.complete and history.events == ()
        original = events.get_pattern_history
        monkeypatch.setattr(events, "get_pattern_history", lambda days: original(days, limit=4))
        assert await events.acall(PatternAnalyzer(events).analyze_all) == []
        with sqlite3.connect(events.db_path) as db:
            assert db.execute("SELECT * FROM events").fetchall() == [("synthetic legacy",)]


async def test_weekday_patterns_keep_weekend_comparison_evidence(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        clock.now = START + 5 * DAY + 4 * 3600
        await events.acall(events.log, "light.synthetic", "turn_on")
        patterns = await events.acall(PatternAnalyzer(events).analyze_all)
        weekday = next(p for p in patterns if p.pattern_type is PatternType.WEEKDAY_TIME)
        assert weekday.occurrences == 5
        assert len(weekday.evidence) == 6
        assert len(weekday.to_dict()["event_dependencies"]) == 6


async def test_diagnostic_counters_never_become_automation_patterns(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        for day in range(5):
            clock.now = START + day * DAY
            await events.acall(events.log, "asr_quality:synthetic", "accepted:ok", extra_context={"rms": .2})
        assert len((await events.acall(events.get_pattern_history)).events) == 5
        assert await events.acall(PatternAnalyzer(events).analyze_all) == []


async def test_receipt_validation_checks_later_sql_batches_and_rejects_duplicates(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        for index in range(130):
            clock.now = START + index
            await events.acall(events.log, "light.synthetic", "turn_on")
        evidence = tuple(e.dependency for e in (await events.acall(events.get_pattern_history)).events)
        assert await events.acall(events.pattern_evidence_current, evidence)
        assert not await events.acall(events.pattern_evidence_current, evidence + (evidence[0],))
        assert not await events.acall(events.pattern_evidence_current, ())
        assert not await events.acall(events.pattern_evidence_current, tuple(d.to_dict() for d in evidence))
        with sqlite3.connect(events.db_path) as db:
            db.execute("DELETE FROM events_policy_v1 WHERE id=130")
        assert not await events.acall(events.pattern_evidence_current, evidence)


async def seed_sequence(events, clock, *, count=8, matching=5):
    for day in range(count):
        clock.now = START + day * DAY
        await events.acall(events.log, "switch.trigger", "turn_on")
        if day < matching:
            for second in (10, 20, 30):
                clock.now = START + day * DAY + second
                await events.acall(events.log, "light.target", "turn_on")


async def test_sequence_confidence_counts_distinct_triggers_and_keeps_unmatched_denominator(tmp_path):
    async with graph(tmp_path) as (_, events, clock):
        # An old unmatched trigger is outside the requested 30-day denominator.
        clock.now = START - 40 * DAY
        await events.acall(events.log, "switch.trigger", "turn_on")
        await seed_sequence(events, clock)
        patterns = await events.acall(PatternAnalyzer(events).analyze_all)
        sequence = next(p for p in patterns if p.pattern_type is PatternType.SEQUENCE)
        assert sequence.occurrences == 5  # Not 15 repeated target executions.
        assert sequence.confidence == .62  # 5/8, not 15/9 or 5/9.
        assert len(sequence.evidence) == 8 + 15
        assert 1 not in {d.event_id for d in sequence.evidence}
        # Deleting an unmatched trigger changes the denominator, invalidating
        # this result even though every positive pair remains physically present.
        with sqlite3.connect(events.db_path) as db:
            db.execute("DELETE FROM events_policy_v1 WHERE id=(SELECT MAX(id) FROM events_policy_v1)")
        assert not await events.acall(events.pattern_evidence_current, sequence.evidence)


async def test_sequence_overload_abstains_without_losing_valid_temporal_patterns(tmp_path, monkeypatch):
    async with graph(tmp_path) as (_, events, clock):
        await seed_sequence(events, clock, count=5)
        monkeypatch.setattr(PatternAnalyzer, "MAX_SEQUENCE_COMPARISONS", 3)
        patterns = await events.acall(PatternAnalyzer(events).analyze_all)
        assert patterns
        assert all(p.pattern_type is not PatternType.SEQUENCE for p in patterns)


@pytest.mark.parametrize("invalidate", ["marker", "expiry"])
async def test_authority_is_rechecked_after_pattern_calculation(tmp_path, monkeypatch, invalidate):
    async with graph(tmp_path) as (privacy, events, clock):
        sources = await seed_days(events, clock)
        analyzer = PatternAnalyzer(events)
        original = analyzer._detect_daily_pattern

        def detect(*args):
            result = original(*args)
            assert result is not None
            if invalidate == "marker":
                privacy.ledger.delete_conversation(sources[0].conversation_id, ACTOR)
            else:
                clock.now = START + 90 * DAY
            return result

        monkeypatch.setattr(analyzer, "_detect_daily_pattern", detect)
        assert await events.acall(analyzer.analyze_all) == []
        assert row_count(events) == 5


async def test_cancelled_analysis_keeps_real_native_work_owned_until_close(tmp_path, monkeypatch):
    async with graph(tmp_path) as (_, events, clock):
        await seed_days(events, clock)
        started, release = threading.Event(), threading.Event()
        analyzer = PatternAnalyzer(events)
        original = analyzer._detect_daily_pattern
        loop_thread = threading.get_ident()

        def blocked(*args):
            assert threading.get_ident() != loop_thread
            started.set()
            assert release.wait(3)
            return original(*args)

        monkeypatch.setattr(analyzer, "_detect_daily_pattern", blocked)
        def analyze():
            analyzer.analyze_all()
            return None

        engine = SimpleNamespace(event_logger=events, get_suggestion_to_present=analyze)
        router = RequestRouter(None, None, None, suggestion_engine=engine)
        observer = asyncio.create_task(router._maybe_present_suggestion())
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(.001)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            close = events.begin_close()
            await asyncio.sleep(0)
            assert not close.done()
            assert events.resource_status()["operations"] == 1
        finally:
            release.set()
            await events.aclose()
            await router.aclose()


async def test_voice_suggestion_calls_remain_serialized_after_moving_off_loop(tmp_path):
    async with graph(tmp_path) as (_, events, _):
        engine = SimpleNamespace(event_logger=events)
        router = RequestRouter(None, None, None, suggestion_engine=engine)
        started, release, second_started = threading.Event(), threading.Event(), threading.Event()

        def first():
            started.set()
            assert release.wait(3)
            return "first"

        def second():
            second_started.set()
            return "second"

        one = asyncio.create_task(router._call_suggestion(first))
        two = None
        try:
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(.001)
            two = asyncio.create_task(router._call_suggestion(second))
            async with asyncio.timeout(1):
                while events.resource_status()["operations"] != 2:
                    await asyncio.sleep(.001)
            assert not second_started.is_set()
            release.set()
            assert await asyncio.gather(one, two) == ["first", "second"]
        finally:
            release.set()
            await asyncio.gather(*(t for t in (one, two) if t is not None), return_exceptions=True)
            await router.aclose()
