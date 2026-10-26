"""Actual native ambient lifetimes, idle buffer expiry and metadata lineage."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from weakref import ref

import pytest

from src.ambient.ambient_stt import AmbientSTT
from src.ambient.room_producer import RoomProducer
from src.memory.fact_ledger import MemoryReviewer
from src.memory.producer_context import PRODUCER_CONTEXT
from src.memory.sources import CaptureBoundary
from src.ambient.types import AmbientUtterance
from src.core.owned_work import WorkCloseTimeout
from tests.mocks.ambient_graph import ambient_graph, capture_source, chunk

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer("synthetic-admin")
STAGES = ["stt", "profiles", "speaker", "quality", "doa"]


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


def active_source(transcriber):
    for task in transcriber.snapshot_operations():
        context = task.get_context().get(PRODUCER_CONTEXT)
        if context is not None and context.sources:
            return context.sources[0]
    raise AssertionError("missing actual producer lineage")


async def count_rows(store):
    cursor = await store._db.execute("SELECT COUNT(*) FROM utterances")
    return (await cursor.fetchone())[0]


async def test_real_pipeline_retains_capture_lineage_and_runs_profile_reads_off_loop(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        await g.transcriber._handle_segment("synthetic-a", g.segment())
        rows = await g.store.undistilled_live(require_provenance=True)
        assert len(rows) == 1
        source = g.privacy.scopes._scopes
        assert rows[0]["conversation_id"] in {scope[0] for scope in source.values()}
        assert rows[0]["expires_at"] == rows[0]["t0"] + 48 * 3600
        assert rows[0]["speaker"] == "synthetic-speaker"
        assert all(
            t != threading.get_ident() for name in STAGES for t in g.gates[name].threads
        )
        assert not g.transcriber.snapshot_operations()
        assert g.identifier.embeddings and all(
            r() is None for r in g.identifier.embeddings
        )


@pytest.mark.parametrize("stage", STAGES)
async def test_revocation_cancels_pipeline_but_actual_native_stage_remains_visible(
    tmp_path, stage
):
    async with ambient_graph(tmp_path, stage=stage) as g:
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        await until(g.gates[stage].entered.is_set)
        source = active_source(g.transcriber)
        try:
            await g.privacy.acall(
                g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR
            )
            await asyncio.wait_for(observer, 2)
            assert g.transcriber.resource_status()["segments"] == 0
            assert g.transcriber.resource_status()["operations"] > 0
            assert await count_rows(g.store) == 0
            assert source.conversation_id not in json.dumps(
                g.transcriber.resource_status()
            )
        finally:
            g.gates[stage].release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: not g.transcriber.snapshot_operations())
        assert await count_rows(g.store) == 0
        if stage == "profiles":
            assert g.gates["speaker"].calls == 0


@pytest.mark.parametrize("stage", STAGES)
async def test_native_stage_keeps_original_store_and_writer_lease_after_close_timeout(
    tmp_path, stage
):
    async with ambient_graph(tmp_path, stage=stage) as g:
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        await until(g.gates[stage].entered.is_set)
        try:
            with pytest.raises(WorkCloseTimeout):
                await g.transcriber.stop()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()["writer_lease"]
            assert (
                g.store._db is not None
                and g.transcriber.resource_status()["operations"] > 0
            )
            assert not g.path._closing.done()
        finally:
            g.gates[stage].release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await g.privacy.begin_close()
        assert not observer.cancelled()
        assert g.store._db is None and not g.privacy.resource_status()["writer_lease"]
        assert not g.transcriber.snapshot_operations()


async def test_expired_inflight_native_result_cannot_start_later_stages_or_persist(
    tmp_path,
):
    async with ambient_graph(tmp_path, stage="stt") as g:
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        await until(g.gates["stt"].entered.is_set)
        g.clock.now += 48 * 3600
        try:
            await asyncio.wait_for(observer, 2)
            assert g.transcriber.resource_status()["operations"] > 0
            assert g.gates["speaker"].calls == g.gates["profiles"].calls == 0
        finally:
            g.gates["stt"].release.set()
            await until(lambda: not g.transcriber.snapshot_operations())
        assert await count_rows(g.store) == 0


async def test_cancelled_observer_releases_audio_and_late_stt_result_without_gc_manipulation(
    tmp_path,
):
    async with ambient_graph(tmp_path, stage="stt") as g:
        segment = g.segment()
        audio = ref(segment.audio)
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", segment)
        )
        del segment
        await until(g.gates["stt"].entered.is_set)
        observer.cancel()
        result = await asyncio.gather(observer, return_exceptions=True)
        assert observer.cancelled() and audio() is not None
        assert g.transcriber.resource_status()["operations"] > 0
        g.gates["stt"].release.set()
        await until(lambda: not g.transcriber.snapshot_operations())
        await until(
            lambda: (
                audio() is None and g.engine.results and g.engine.results[0]() is None
            )
        )
        assert (
            result and observer.cancelled()
        )  # Retain the observer/exception through the probe.


async def test_asr_queued_behind_default_executor_rechecks_source_before_native_entry(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            assert release.wait(3)

        blocker = loop.run_in_executor(None, hold)
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        try:
            await until(entered.is_set)
            await until(lambda: bool(g.transcriber._stt.snapshot_operations()))
            source = active_source(g.transcriber)
            await g.privacy.acall(
                g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR
            )
            await asyncio.wait_for(observer, 2)
            assert (
                not blocker.done() and g.transcriber.resource_status()["operations"] > 0
            )
            assert g.gates["stt"].calls == 0
        finally:
            release.set()
            await blocker
            await until(lambda: not g.transcriber.snapshot_operations())
            loop._default_executor = previous
            pool.shutdown(wait=True)
        assert g.gates["stt"].calls == 0 and await count_rows(g.store) == 0


async def test_real_whisper_pinned_delegate_survives_cancelled_outer_observer(
    tmp_path, monkeypatch
):
    from src.stt.whisper_fast import FastWhisperSTT, STTResult

    async with ambient_graph(tmp_path) as g:
        pinned = FastWhisperSTT(device="cpu")
        monkeypatch.setattr(
            pinned, "_load_impl", lambda: setattr(pinned, "_model", object())
        )
        entered, release = threading.Event(), threading.Event()
        identities = []

        def transcribe(audio):
            identities.append(threading.get_ident())
            entered.set()
            assert release.wait(3)
            return STTResult("Synthetic pinned result", 1.0, 0.01, -0.1, 1.0)

        monkeypatch.setattr(pinned, "_transcribe_impl", transcribe)
        g.transcriber._stt = AmbientSTT(pinned, source_policy=g.privacy.ledger)
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            assert (
                identities == [pinned._worker_ident]
                and pinned._worker_ident != threading.get_ident()
            )
            assert g.transcriber._stt.snapshot_operations()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()["writer_lease"]
        finally:
            release.set()
            await g.privacy.begin_close()
            await asyncio.to_thread(pinned._executor.shutdown, wait=True)


@pytest.mark.parametrize("stage", ["factory", "vad"])
async def test_native_room_factory_and_vad_do_not_block_capture_or_disappear_on_close(
    tmp_path, stage
):
    async with ambient_graph(tmp_path, stage=stage) as g:
        await g.path.start()
        g.tap.push("synthetic-a", chunk(), ts=g.clock.now)
        await until(g.gates[stage].entered.is_set)
        ticks = 0
        for _ in range(10):
            await asyncio.sleep(0.002)
            ticks += 1
        assert ticks == 10 and all(
            t != threading.get_ident() for t in g.gates[stage].threads
        )
        try:
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert (
                g.privacy.resource_status()["writer_lease"] and g.store._db is not None
            )
            assert g.transcriber.resource_status()["operations"] > 0
        finally:
            g.gates[stage].release.set()
            await g.privacy.begin_close()
        assert g.store._db is None
        assert not g.transcriber.snapshot_operations()


async def test_idle_preroll_expires_and_releases_audio_without_new_input_or_read(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        await g.path.start()
        audio = chunk(0.0)
        reference = ref(audio)
        g.tap.push("synthetic-a", audio, ts=g.clock.now)
        del audio
        state = g.transcriber._room_states["synthetic-a"]
        await until(
            lambda: state.status()["state_retained"] and not state.snapshot_operations()
        )
        assert reference() is not None
        g.clock.now += 48 * 3600
        await until(lambda: not state.status()["state_retained"])
        await until(lambda: reference() is None)
        assert g.tap.buffered_count() == 0 and g.gates["stt"].calls == 0


async def test_revoking_one_room_prunes_only_its_idle_buffers_and_rejects_renaming_old_audio(
    tmp_path,
):
    async with ambient_graph(tmp_path, rooms=("synthetic-a", "synthetic-b")) as g:
        first = capture_source(g, "synthetic-a")
        second = capture_source(g, "synthetic-b")
        await g.path.start()
        for room in ("synthetic-a", "synthetic-b"):
            g.tap.push(room, chunk(0.0), ts=g.clock.now)
        await until(
            lambda: all(
                state.status()["state_retained"]
                for state in g.transcriber._room_states.values()
            )
        )
        await g.privacy.acall(
            g.privacy.ledger.delete_conversation, first.conversation_id, ACTOR
        )
        await until(
            lambda: (
                not g.transcriber._room_states["synthetic-a"].status()["state_retained"]
            )
        )
        assert g.transcriber._room_states["synthetic-b"].status()["state_retained"]
        await g.transcriber._handle_segment("synthetic-a", g.segment())
        assert g.gates["stt"].calls == 0
        assert not g.privacy.ledger.conversation_revoked(second.conversation_id)
        g.clock.now += 1
        fresh = capture_source(g, "synthetic-a")
        assert fresh.conversation_id != first.conversation_id
        await g.transcriber._handle_segment("synthetic-a", g.segment())
        assert await count_rows(g.store) == 1


async def test_revocation_while_vad_runs_filters_both_output_and_native_state(tmp_path):
    async with ambient_graph(tmp_path, stage="vad") as g:
        source = capture_source(g)
        await g.path.start()
        g.tap.push("synthetic-a", chunk(), ts=g.clock.now)
        await until(g.gates["vad"].entered.is_set)
        await g.privacy.acall(
            g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR
        )
        g.transcriber.prune_private()
        assert g.transcriber.resource_status()["operations"] > 0
        g.gates["vad"].release.set()
        state = g.transcriber._room_states["synthetic-a"]
        await until(
            lambda: (
                not state.status()["state_retained"] and not state.snapshot_operations()
            )
        )
        assert state.ready == [] and g.gates["stt"].calls == 0


@pytest.mark.parametrize("action", ["fence", "close"])
async def test_inflight_callback_cannot_resurrect_a_detached_tap_queue(
    tmp_path, action
):
    async with ambient_graph(tmp_path) as g:
        g.tap.register_room("synthetic-a")
        entered, release = threading.Event(), threading.Event()

        class DelayedDeque(deque):
            def append(self, item):
                entered.set()
                assert release.wait(3)
                super().append(item)

        g.tap._queues["synthetic-a"] = DelayedDeque(maxlen=10)
        timestamp = g.clock.now
        thread = threading.Thread(
            target=lambda: g.tap.push("synthetic-a", chunk(), ts=timestamp)
        )
        thread.start()
        try:
            await until(entered.is_set)
            if action == "fence":
                g.tap.publish_boundary(
                    "synthetic-a", CaptureBoundary(timestamp, timestamp)
                )
            else:
                g.tap.close()
        finally:
            release.set()
            await asyncio.to_thread(thread.join, 2)
        assert not thread.is_alive() and g.tap.drain("synthetic-a") == []
        if action == "close":
            g.tap.register_room("synthetic-a")
            g.tap.push("synthetic-a", chunk(), ts=timestamp + 1)
            assert g.tap.buffered_count() == 0


async def test_ready_segments_retire_at_original_timestamps_and_scope_closes_after_cancel(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        state = RoomProducer(
            "synthetic-a", g.transcriber._segmenter_factory, require_cleanup=True
        )
        now = g.clock.now
        try:
            await state.feed(
                [
                    (now, chunk(), False),
                    (now + 0.08, chunk(0.0), False),
                    (now + 1, chunk(), False),
                    (now + 1.08, chunk(0.0), False),
                ]
            )
            assert len(state.ready) == 2
            state.publish_boundary(CaptureBoundary(now, now + 0.5))
            assert len(state.ready) == 1 and state.ready[0].t0 > now + 0.5
        finally:
            await state.begin_close()
        assert state.ready == [] and not state.status()["state_retained"]


async def test_future_start_and_original_48h_boundaries_do_not_follow_read_times(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        boundary = g.privacy.scopes.buffer_boundary("synthetic-a")
        assert boundary.allows(g.clock.now) and not boundary.allows(g.clock.now - 1)
        g.clock.now += 48 * 3600
        current = g.privacy.scopes.buffer_boundary("synthetic-a")
        assert not current.allows(boundary.started_at)
        assert current.started_at == boundary.started_at
        assert not current.allows(float("nan")) and not current.allows(float("inf"))


async def test_delayed_boundary_publication_and_clock_rollback_cannot_restore_retired_audio(
    tmp_path,
):
    async with ambient_graph(tmp_path) as g:
        segment = g.segment()
        before = g.privacy.scopes.buffer_boundary("synthetic-a")
        g.tap.register_room("synthetic-a")
        g.clock.now += 48 * 3600
        retired = g.privacy.scopes.buffer_boundary("synthetic-a")
        g.tap.publish_boundary("synthetic-a", retired)
        g.tap.publish_boundary("synthetic-a", before)
        g.clock.now -= 24 * 3600
        assert (
            g.privacy.scopes.buffer_boundary("synthetic-a").retired_through
            == retired.retired_through
        )
        g.tap.push("synthetic-a", segment.audio, ts=segment.t0)
        assert g.tap.drain("synthetic-a") == []
        await g.transcriber._handle_segment("synthetic-a", segment)
        assert g.gates["stt"].calls == 0


async def test_native_reset_keeps_writer_lease_until_real_state_cleanup_exits(tmp_path):
    async with ambient_graph(tmp_path) as g:
        await g.path.start()
        g.tap.push("synthetic-a", chunk(0.0), ts=g.clock.now)
        state = g.transcriber._room_states["synthetic-a"]
        await until(
            lambda: state.status()["state_retained"] and not state.snapshot_operations()
        )
        g.gates["reset"].block()
        g.clock.now += 48 * 3600
        await until(g.gates["reset"].entered.is_set)
        try:
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert (
                g.privacy.resource_status()["writer_lease"] and g.store._db is not None
            )
            assert g.transcriber.resource_status()["operations"] > 0
        finally:
            g.gates["reset"].release.set()
            await g.privacy.begin_close()
        assert g.store._db is None


async def test_actual_builder_attaches_all_native_adapters_and_path_to_same_privacy(
    tmp_path,
):
    from src.ambient.transcriber import build_ambient_path

    async with ambient_graph(tmp_path) as g:
        path = build_ambient_path(
            {
                "db_path": str(tmp_path / "built.db"),
                "keep_audio": {"dir": str(tmp_path / "clips")},
            },
            {},
            [],
            None,
            privacy=g.privacy,
        )
        try:
            assert path.transcriber._privacy is g.privacy
            assert path.transcriber._stt._source_policy is g.privacy.ledger
            assert path.transcriber._tagger._source_policy is g.privacy.ledger
            assert any(
                reference() == path.begin_close for reference in g.privacy._borrowers
            )
        finally:
            await path.stop()
        assert path.transcriber._closed and path.tap.buffered_count() == 0


async def test_cancelled_vad_batch_does_not_retain_completed_audio_in_observer_traceback(
    tmp_path,
):
    async with ambient_graph(tmp_path, stage="vad") as g:
        outputs = []
        factory = g.transcriber._segmenter_factory

        def tracked():
            segmenter = factory()
            feed = segmenter.feed

            def observe(**kwargs):
                segments = feed(**kwargs)
                outputs.extend(ref(segment.audio) for segment in segments)
                return segments

            segmenter.feed = observe
            return segmenter

        state = RoomProducer("synthetic-a", tracked, require_cleanup=True)
        now = g.clock.now
        items = [(now, chunk(), False), (now + 0.08, chunk(0.0), False)]
        inputs = [ref(item[1]) for item in items]
        observer = asyncio.create_task(state.feed(items))
        del items
        try:
            await until(g.gates["vad"].entered.is_set)
            observer.cancel()
            result = await asyncio.gather(observer, return_exceptions=True)
            assert observer.cancelled() and state.snapshot_operations()
            g.gates["vad"].release.set()
            await until(lambda: not state.snapshot_operations())
            await until(lambda: outputs and all(r() is None for r in inputs + outputs))
            assert state.ready == [] and result
        finally:
            g.gates["vad"].release.set()
            await state.begin_close()


@pytest.mark.parametrize("phase", ["factory", "feed"])
async def test_native_segmentation_failure_exposes_only_fixed_status_and_discards_buffers(
    tmp_path, phase, caplog
):
    from src.ambient.segmenter import UtteranceSegmenter

    async with ambient_graph(tmp_path) as g:

        def broken(*args, **kwargs):
            raise OSError("Synthetic private audio path")

        def factory():
            if phase == "factory":
                broken()
            return UtteranceSegmenter(broken)

        state = RoomProducer("synthetic-a", factory, require_cleanup=True)
        try:
            await state.feed([(g.clock.now, chunk(), False)])
            assert state.status()["error"] == "ambient_segmentation_failed"
            assert not state.status()["state_retained"] and state.ready == []
            assert "Synthetic private" not in json.dumps(state.status())
            own = [
                r.getMessage()
                for r in caplog.records
                if r.name == "src.ambient.room_producer"
            ]
            assert own and all("Synthetic private" not in message for message in own)
        finally:
            await state.begin_close()


async def test_pending_failed_marker_discards_transient_work_but_preserves_durable_capture(
    tmp_path,
):
    async with ambient_graph(tmp_path, stage="stt") as g:
        observer = asyncio.create_task(
            g.transcriber._handle_segment("synthetic-a", g.segment())
        )
        await until(g.gates["stt"].entered.is_set)
        source = active_source(g.transcriber)
        await g.store.add(
            AmbientUtterance(
                "synthetic-a",
                g.clock.now,
                g.clock.now + 1,
                text="Synthetic durable retry input",
                source="live",
                vad_prob=0.9,
                capture_id=source.capture_id,
                conversation_id=source.conversation_id,
                expires_at=source.expires_at,
            )
        )
        entered, release = threading.Event(), threading.Event()
        original = g.privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("Synthetic failed marker")

        g.privacy.ledger._db = FailedCommit()
        marking = asyncio.create_task(
            g.privacy.acall(
                g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR
            )
        )
        try:
            await until(entered.is_set)
            g.transcriber.prune_private()
            await asyncio.wait_for(observer, 2)
            assert g.transcriber.resource_status()["operations"] > 0
            assert await count_rows(g.store) == 1
            assert await g.store.undistilled_live() == []
        finally:
            release.set()
            with pytest.raises(OSError, match="failed marker"):
                await marking
            g.privacy.ledger._db = original
            g.gates["stt"].release.set()
            await until(lambda: not g.transcriber.snapshot_operations())
        assert len(await g.store.undistilled_live()) == 1
        g.clock.now += 1
        await g.transcriber._handle_segment("synthetic-a", g.segment())
        assert await count_rows(g.store) == 2


async def test_native_lineage_distinguishes_revoked_scope_from_another_live_room(
    tmp_path,
):
    from src.stt.whisper_fast import STTResult
    from tests.mocks.ambient_graph import Gate

    async with ambient_graph(tmp_path, rooms=("synthetic-a", "synthetic-b")) as g:
        gates = {room: Gate() for room in ("synthetic-a", "synthetic-b")}
        for gate in gates.values():
            gate.block()

        def transcribe(audio):
            gates[PRODUCER_CONTEXT.get().room_id].step()
            return STTResult("Synthetic room result", 1.0, 0.01, -0.1, 1.0)

        g.engine.transcribe_with_confidence = transcribe
        observers = {
            room: asyncio.create_task(g.transcriber._handle_segment(room, g.segment()))
            for room in gates
        }
        try:
            await until(
                lambda: (
                    any(gate.entered.is_set() for gate in gates.values())
                    and len(g.transcriber._stt.snapshot_operations()) == 2
                )
            )
            active = next(room for room, gate in gates.items() if gate.entered.is_set())
            waiting = next(room for room in gates if room != active)
            sources = {
                task.get_context().get(PRODUCER_CONTEXT).room_id: task.get_context()
                .get(PRODUCER_CONTEXT)
                .sources[0]
                for task in g.transcriber._stt.snapshot_operations()
            }
            assert (
                len(sources) == 2
                and sources["synthetic-a"].conversation_id
                != sources["synthetic-b"].conversation_id
            )
            await g.privacy.acall(
                g.privacy.ledger.delete_conversation,
                sources[active].conversation_id,
                ACTOR,
            )
            await asyncio.wait_for(observers[active], 2)
            assert not observers[waiting].done()
            cohort = [
                task
                for task in g.transcriber.snapshot_operations()
                if (context := task.get_context().get(PRODUCER_CONTEXT)) is not None
                and any(
                    s.conversation_id == sources[active].conversation_id
                    for s in context.sources
                )
            ]
            assert cohort
            gates[active].release.set()
            await until(lambda: all(task.done() for task in cohort))
            await until(gates[waiting].entered.is_set)
            assert not observers[waiting].done()
            assert g.transcriber.resource_status()["operations"] > 0
        finally:
            for gate in gates.values():
                gate.release.set()
            await asyncio.gather(*observers.values(), return_exceptions=True)
        rows = await g.store.undistilled_live()
        assert len(rows) == 1 and rows[0]["room_id"] == waiting


async def test_concurrent_silero_factories_serialize_cache_load_and_keep_independent_state(
    tmp_path, monkeypatch
):
    import numpy as np
    import torch
    from src.ambient.segmenter import make_silero_predictor

    entered, release = threading.Event(), threading.Event()
    attempted = threading.Event()
    active, peak, models = [0], [0], []

    class Model:
        def __init__(self):
            self.state = 0

        def eval(self):
            pass

        def reset_states(self):
            self.state = 0

        def __call__(self, tensor, sample_rate):
            self.state += 1
            return torch.tensor(self.state / 10)

    def load(**kwargs):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        entered.set()
        assert release.wait(3)
        model = Model()
        models.append(model)
        active[0] -= 1
        return model, None

    monkeypatch.setattr(torch.hub, "load", load)

    def second_factory():
        attempted.set()
        return make_silero_predictor()

    first = asyncio.create_task(asyncio.to_thread(make_silero_predictor))
    second = None
    try:
        await until(entered.is_set)
        second = asyncio.create_task(asyncio.to_thread(second_factory))
        await until(attempted.is_set)
        await asyncio.sleep(0.02)
        assert active[0] == peak[0] == 1
    finally:
        release.set()
        predictors = await asyncio.gather(
            *(task for task in (first, second) if task is not None)
        )
    assert len(models) == 2 and models[0] is not models[1]
    assert predictors[0](np.ones(1024, dtype=np.float32)) == pytest.approx(0.2)
    assert predictors[1](np.ones(1024, dtype=np.float32)) == pytest.approx(0.2)
    predictors[0].reset()
    assert predictors[0](np.ones(512, dtype=np.float32)) == pytest.approx(0.1)
    assert predictors[1](np.ones(512, dtype=np.float32)) == pytest.approx(0.3)


async def test_callback_cannot_combine_old_boundary_with_new_queue(tmp_path):
    async with ambient_graph(tmp_path) as g:
        g.tap.register_room("synthetic-a")
        entered, release = threading.Event(), threading.Event()

        class DelayedBoundaries(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                if threading.current_thread().name == "old-ambient-callback":
                    entered.set()
                    assert release.wait(3)
                return value

        now = g.clock.now
        g.tap._boundaries = DelayedBoundaries(
            {"synthetic-a": CaptureBoundary(None, now - 1)}
        )
        thread = threading.Thread(
            name="old-ambient-callback",
            target=lambda: g.tap.push("synthetic-a", chunk(), ts=now),
        )
        thread.start()
        try:
            await until(entered.is_set)
            g.tap.publish_boundary("synthetic-a", CaptureBoundary(None, now))
        finally:
            release.set()
            await asyncio.to_thread(thread.join, 2)
        assert not thread.is_alive() and g.tap.drain("synthetic-a") == []
        g.tap.push("synthetic-a", chunk(), ts=now + 1)
        assert len(g.tap.drain("synthetic-a")) == 1


@pytest.mark.parametrize("stage", ["stt", "speaker"])
async def test_shared_native_model_serializes_entries_but_revoked_waiter_exits_without_live_room(
    tmp_path, stage
):
    async with ambient_graph(
        tmp_path, stage=stage, rooms=("synthetic-a", "synthetic-b")
    ) as g:
        adapter = g.transcriber._stt if stage == "stt" else g.transcriber._tagger
        observers = {
            room: asyncio.create_task(g.transcriber._handle_segment(room, g.segment()))
            for room in ("synthetic-a", "synthetic-b")
        }
        try:
            await until(g.gates[stage].entered.is_set)
            await until(lambda: len(adapter.snapshot_operations()) == 2)
            # The first entrant holds the model; the other native scope is queued.
            contexts = {
                task.get_context()
                .get(PRODUCER_CONTEXT)
                .room_id: task.get_context().get(PRODUCER_CONTEXT)
                for task in adapter.snapshot_operations()
            }
            locked = adapter._inference_lock
            assert locked.locked()
            # Backend callbacks record their room without keeping private arrays.
            active = (
                g.engine.active_room if stage == "stt" else g.identifier.active_room
            )
            waiting = next(room for room in observers if room != active)
            await asyncio.sleep(0.02)
            assert g.gates[stage].calls == 1
            await g.privacy.acall(
                g.privacy.ledger.delete_conversation,
                contexts[waiting].sources[0].conversation_id,
                ACTOR,
            )
            await asyncio.wait_for(observers[waiting], 2)
            await until(
                lambda: all(
                    task.get_context().get(PRODUCER_CONTEXT).room_id != waiting
                    for task in adapter.snapshot_operations()
                )
            )
            assert not observers[active].done() and g.gates[stage].calls == 1
            assert not observers[waiting].cancelled()
        finally:
            g.gates[stage].release.set()
            await asyncio.gather(*observers.values(), return_exceptions=True)
        rows = await g.store.undistilled_live()
        assert len(rows) == 1 and rows[0]["room_id"] == active
