"""Actual command owners and privacy authority with gated CPU model endpoints."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import threading
from types import SimpleNamespace
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT, producer_context
from src.memory.prompt_context import SourcesInvalidated
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.request_router import RequestRouter
from src.stt.whisper_fast import STTResult
from src.users.speaker_identifier import SpeakerMatch
from tests.mocks.ambient_graph import Clock, Gate
from tests.mocks.responses import response_handler_mock

pytestmark = pytest.mark.asyncio
STAGES = ("primary", "shadow", "profiles", "speaker", "emotion")
ACTOR = MemoryReviewer("synthetic-admin")


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


class Engine:
    def __init__(self, gate, *, text="Synthetic private command", fail=False):
        self.gate, self.text, self.fail = gate, text, fail
        self.results = []
        self.contexts = []

    def transcribe_with_confidence(self, audio, sample_rate=16000):
        self.contexts.append(PRODUCER_CONTEXT.get())
        self.gate.step()
        value = STTResult(self.text, 1., no_speech_prob=.01, avg_logprob=-.1)
        self.results.append(ref(value))
        if self.fail:
            raise RuntimeError("Synthetic private failure payload")
        return value


@dataclass
class Graph:
    privacy: PrivacyServices
    processor: CommandProcessor
    clock: Clock
    gates: dict
    primary: Engine
    shadow: Engine
    embeddings: list
    last_seen: list

    def source(self, room="synthetic-room"):
        return self.privacy.source_for_capture(str(uuid4()), self.clock(), room,
                                               conversation_id=str(uuid4()))

    def launch(self, source, *, pretranscribed=False, parallel=True, audio=None):
        if audio is None:
            audio = np.ones(16000, dtype=np.float32)
        with producer_context("synthetic-room", (source,)):
            return asyncio.create_task(self.processor.process_command(
                audio, use_parallel=parallel,
                pretranscribed_text="Synthetic covered command" if pretranscribed else None))


@asynccontextmanager
async def graph(tmp_path, *, blocked=None, shadow=False, veto=False, fail=False):
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    gates = {name: Gate() for name in STAGES}
    if blocked:
        gates[blocked].block()
    primary, secondary = Engine(gates["primary"], fail=fail), Engine(gates["shadow"])
    embeddings, last_seen = [], []
    user = SimpleNamespace(user_id="synthetic-user", name="Synthetic private name")

    class Registry:
        _version = 1

        def get_all_embeddings(self):
            gates["profiles"].step()
            return {user.user_id: np.ones(192, dtype=np.float32)}

        def get_user(self, user_id):
            return user

        def update_last_seen(self, user_id):
            last_seen.append(user_id)

    class Identifier:
        def identify(self, audio, profiles):
            gates["speaker"].step()
            value = np.ones(192, dtype=np.float32)
            embeddings.append(ref(value))
            return SpeakerMatch(user.user_id, .95, value, True)

    class Emotion:
        def detect(self, audio):
            gates["emotion"].step()
            return SimpleNamespace(emotion="synthetic-private-emotion", processing_time_ms=1.)

    cp = CommandProcessor(primary, speaker_identifier=Identifier(), user_manager=Registry(),
                          emotion_detector=Emotion(), shadow_stt=secondary if shadow else None,
                          shadow_veto=veto, shadow_veto_timeout_s=3,
                          privacy=privacy, close_timeout_s=.03)
    try:
        yield Graph(privacy, cp, clock, gates, primary, secondary, embeddings, last_seen)
    finally:
        for gate in gates.values():
            gate.release.set()
        await cp.begin_close()
        await privacy.begin_close()


@pytest.mark.parametrize("parallel,pretranscribed", [(True, False), (False, False), (True, True), (False, True)])
async def test_all_command_paths_move_models_and_profiles_off_loop(tmp_path, parallel, pretranscribed):
    async with graph(tmp_path) as g:
        source = g.source()
        result = await g.launch(source, pretranscribed=pretranscribed, parallel=parallel)
        assert result.success and result.user.user_id == "synthetic-user"
        assert result.emotion.emotion == "synthetic-private-emotion"
        if pretranscribed:
            assert g.gates["primary"].calls == 0 and result.stt_confidence is None
        else:
            assert result.stt_confidence.avg_logprob == -.1
            assert g.primary.contexts[0].sources == (source,)
        assert all(t != threading.get_ident() for gate in g.gates.values() for t in gate.threads)
        assert g.embeddings and all(value() is None for value in g.embeddings)
        assert not g.processor.snapshot_operations()


@pytest.mark.parametrize("stage", STAGES)
async def test_revocation_aborts_command_but_retains_actual_native_stage(tmp_path, stage):
    async with graph(tmp_path, blocked=stage, shadow=stage == "shadow", veto=stage == "shadow") as g:
        source = g.source()
        observer = g.launch(source)
        await until(g.gates[stage].entered.is_set)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        with pytest.raises(SourcesInvalidated):
            await observer
        await until(lambda: g.processor.resource_status()["commands"] == 0)
        tasks = g.processor.snapshot_operations()
        assert tasks and any(task.get_context().get(PRODUCER_CONTEXT).sources == (source,) for task in tasks)
        assert source.conversation_id not in json.dumps(g.processor.resource_status())
        assert g.processor.get_current_user() is None
        g.gates[stage].release.set()
        await until(lambda: not g.processor.snapshot_operations())
        assert all(value() is None for value in g.primary.results + g.shadow.results + g.embeddings)
        if stage in ("profiles", "speaker"):
            assert not g.last_seen
        if stage == "profiles":
            assert g.gates["speaker"].calls == 0


@pytest.mark.parametrize("stage", STAGES)
async def test_native_command_keeps_privacy_lease_after_bounded_close(tmp_path, stage):
    async with graph(tmp_path, blocked=stage, shadow=stage == "shadow", veto=stage == "shadow") as g:
        observer = g.launch(g.source())
        await until(g.gates[stage].entered.is_set)
        with pytest.raises(WorkCloseTimeout):
            await g.processor.aclose()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()["writer_lease"]
        assert g.processor.resource_status()["native"]["operations"] > 0
        assert not g.processor.begin_close().done()
        g.gates[stage].release.set()
        await asyncio.gather(observer, return_exceptions=True)
        await g.privacy.begin_close()
        assert not g.privacy.resource_status()["writer_lease"]
        assert not g.processor.snapshot_operations()


async def test_cancelled_command_drops_input_and_late_result_without_gc_manipulation(tmp_path):
    async with graph(tmp_path, blocked="primary") as g:
        audio = np.ones(16000, dtype=np.float32)
        audio_ref = ref(audio)
        observer = g.launch(g.source(), audio=audio, parallel=False)
        del audio
        await until(g.gates["primary"].entered.is_set)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await observer
        assert g.processor.resource_status()["native"]["operations"] == 1
        assert audio_ref() is not None
        g.gates["primary"].release.set()
        await until(lambda: not g.processor.snapshot_operations())
        assert caught.value is not None  # Keep the cancelled observer traceback alive.
        assert audio_ref() is None
        assert g.primary.results and all(value() is None for value in g.primary.results)
        assert g.gates["speaker"].calls == 0


async def test_veto_timeout_keeps_native_shadow_visible_and_releases_its_late_result(tmp_path):
    async with graph(tmp_path, blocked="shadow", shadow=True, veto=True) as g:
        g.processor.shadow_veto_timeout_s = .03
        result = await g.launch(g.source())
        assert result.success and not result.stt_vetoed
        assert g.processor.resource_status()["native"]["operations"] == 1
        assert g.processor.resource_status()["shadows"] == 1
        with pytest.raises(WorkCloseTimeout):
            await g.processor.aclose()
        g.gates["shadow"].release.set()
        await g.processor.begin_close()
        assert g.shadow.results and all(value() is None for value in g.shadow.results)


async def test_observer_shadow_is_source_watched_after_primary_response(tmp_path, caplog):
    async with graph(tmp_path, blocked="shadow", shadow=True) as g:
        source = g.source()
        with caplog.at_level(logging.INFO):
            result = await g.launch(source)
            assert result.success
            await until(g.gates["shadow"].entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await until(lambda: not g.processor._shadow_tasks)
            assert g.processor.resource_status()["native"]["operations"] == 1
            g.gates["shadow"].release.set()
            await until(lambda: not g.processor.snapshot_operations())
        assert "[STT-shadow]" not in caplog.text
        assert "Synthetic private" not in caplog.text
        assert g.processor.get_current_user() is None


async def test_expired_queued_model_exits_while_another_room_still_uses_it(tmp_path):
    async with graph(tmp_path, blocked="primary") as g:
        live, retired = g.source("live-room"), g.source("retired-room")
        first = g.launch(live, parallel=False)
        await until(g.gates["primary"].entered.is_set)
        second = g.launch(retired, parallel=False)
        await until(lambda: g.processor.resource_status()["native"]["operations"] == 2)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, retired.conversation_id, ACTOR)
        with pytest.raises(SourcesInvalidated):
            await second
        await until(lambda: g.processor.resource_status()["native"]["operations"] == 1)
        assert not first.done() and g.gates["primary"].calls == 1
        g.gates["primary"].release.set()
        assert (await first).success


async def test_raw_expiry_prevents_late_output_and_idle_identification_reads(tmp_path):
    async with graph(tmp_path, blocked="primary") as g:
        source = g.source()
        observer = g.launch(source)
        await until(g.gates["primary"].entered.is_set)
        g.clock.now = source.expires_at
        with pytest.raises(SourcesInvalidated):
            await observer
        g.gates["primary"].release.set()
        await until(lambda: not g.processor.snapshot_operations())
        fresh = g.source()
        assert (await g.launch(fresh)).success
        g.clock.now = fresh.expires_at
        g.processor.prune_private()
        assert g.processor._current_user is None and g.processor._current_emotion is None


async def test_model_error_does_not_expose_or_retain_private_native_traceback(tmp_path, caplog):
    async with graph(tmp_path, fail=True) as g:
        with caplog.at_level(logging.DEBUG):
            result = await g.launch(g.source(), parallel=False)
        assert not result.success
        assert "Synthetic private failure payload" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records if record.name == "src.pipeline.command_processor")
        assert all(value() is None for value in g.primary.results)


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_real_router_binds_actual_native_lineage_without_semantic_memory(tmp_path, orchestrated):
    async with graph(tmp_path, blocked="primary") as g:
        from unittest.mock import AsyncMock, MagicMock
        dispatch = SimpleNamespace(queue=SimpleNamespace(prune_private=lambda: None), dispatch=AsyncMock())
        ha, chroma = MagicMock(), MagicMock()
        handler = response_handler_mock()
        router = RequestRouter(g.processor, handler, MagicMock(), privacy=g.privacy,
                               orchestrator=dispatch, orchestrator_enabled=orchestrated,
                               ha_client=ha, chroma_sync=chroma)
        event = CommandEvent(np.ones(16000, dtype=np.float32), "synthetic-room", captured_at=g.clock())
        observer = asyncio.create_task(router.process_command(event))
        try:
            await until(g.gates["primary"].entered.is_set)
            context = g.primary.contexts[0]
            source = context.sources[0]
            assert context.room_id == event.room_id and source.capture_id == event.capture_id
            assert source.captured_at == event.captured_at
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            result = await observer
            assert result["intent"] == "cancelled" and not result["success"]
            assert not result["text"] and not result["response"]
            assert not ha.method_calls and not chroma.method_calls and not dispatch.dispatch.called
            assert not handler.respond.called
        finally:
            g.gates["primary"].release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await router.aclose()


async def test_owned_startup_keeps_writer_lease_until_real_model_load_exits(tmp_path):
    from src.pipeline.voice_pipeline import VoicePipeline

    async with graph(tmp_path, blocked="primary") as g:
        g.primary.load = g.gates["primary"].step
        pipeline = VoicePipeline(None, g.processor, None, None, None)
        observer = asyncio.create_task(pipeline.run())
        await until(g.gates["primary"].entered.is_set)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await pipeline.stop()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()["writer_lease"]
        assert g.processor.resource_status()["native"]["operations"] == 1
        g.gates["primary"].release.set()
        await g.privacy.begin_close()
        assert not g.privacy.resource_status()["writer_lease"]


async def test_real_whisper_pinned_delegate_stays_owned_after_command_cancel(tmp_path, monkeypatch):
    from src.stt.whisper_fast import FastWhisperSTT

    async with graph(tmp_path) as g:
        pinned = FastWhisperSTT(device="cpu")
        entered, release = threading.Event(), threading.Event()
        identities, results = [], []
        monkeypatch.setattr(pinned, "_load_impl", lambda: setattr(pinned, "_model", object()))

        def transcribe(audio):
            identities.append(threading.get_ident())
            entered.set()
            assert release.wait(3)
            value = STTResult("Synthetic pinned command", 1.)
            results.append(ref(value))
            return value

        monkeypatch.setattr(pinned, "_transcribe_impl", transcribe)
        cp = CommandProcessor(pinned, privacy=g.privacy, close_timeout_s=.03)
        with producer_context("synthetic-room", (g.source(),)):
            observer = asyncio.create_task(cp.process_command(np.ones(16000, dtype=np.float32)))
        try:
            await until(entered.is_set)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert identities == [pinned._worker_ident] and pinned._worker_ident != threading.get_ident()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert cp.snapshot_operations() and g.privacy.resource_status()["writer_lease"]
        finally:
            release.set()
            await g.privacy.begin_close()
            await asyncio.to_thread(pinned._executor.shutdown, wait=True)
        assert results and all(value() is None for value in results)


@pytest.mark.parametrize("pretranscribed", [False, True])
async def test_real_emotion_adapter_accepts_same_input_in_both_command_paths(tmp_path, pretranscribed):
    from src.users.emotion_detector import EmotionDetector

    async with graph(tmp_path) as g:
        detector = EmotionDetector("synthetic-model", "cpu")

        def predictions(audio, sampling_rate):
            assert sampling_rate == 16000
            g.gates["emotion"].step()
            return [{"label": "neutral", "score": .9}]

        detector._model = predictions
        cp = CommandProcessor(g.primary, emotion_detector=detector, privacy=g.privacy)
        with producer_context("synthetic-room", (g.source(),)):
            result = await cp.process_command(np.ones(16000, dtype=np.float32),
                pretranscribed_text="Synthetic covered command" if pretranscribed else None)
        assert result.emotion.emotion == "neutral"
        assert g.gates["emotion"].calls == 1
        assert g.gates["emotion"].threads[0] != threading.get_ident()
        await cp.aclose()


async def test_pending_failed_marker_discards_transient_command_and_allows_fresh_retry(tmp_path):
    async with graph(tmp_path, blocked="primary") as g:
        source = g.source()
        observer = g.launch(source, parallel=False)
        await until(g.gates["primary"].entered.is_set)
        original = g.privacy.ledger._db
        entered, release = threading.Event(), threading.Event()

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)

            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError("synthetic marker rollback")

        g.privacy.ledger._db = FailedCommit()
        marking = asyncio.create_task(g.privacy.acall(
            g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR))
        try:
            await until(entered.is_set)
            with pytest.raises(SourcesInvalidated):
                await observer
            assert g.processor.resource_status()["native"]["operations"] == 1
        finally:
            release.set()
            with pytest.raises(OSError, match="marker rollback"):
                await marking
            g.privacy.ledger._db = original
            g.gates["primary"].release.set()
            await until(lambda: not g.processor.snapshot_operations())
        assert g.privacy.sources_available((source,))
        assert g.processor.get_current_user() is None
        assert all(value() is None for value in g.primary.results)
        assert (await g.launch(source)).success


async def test_queued_executor_command_rechecks_source_before_native_entry(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    async with graph(tmp_path) as g:
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            assert release.wait(3)

        blocker = loop.run_in_executor(None, hold)
        source = g.source()
        observer = g.launch(source, parallel=False)
        try:
            await until(entered.is_set)
            await until(lambda: bool(g.processor._work._tasks))
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            with pytest.raises(SourcesInvalidated):
                await observer
            assert not blocker.done() and g.processor.snapshot_operations()
            assert g.gates["primary"].calls == 0
        finally:
            release.set()
            await blocker
            await until(lambda: not g.processor.snapshot_operations())
            loop._default_executor = previous
            pool.shutdown(wait=True)
        assert g.gates["primary"].calls == 0


async def test_cancelled_model_waiter_releases_input_while_other_room_stays_live(tmp_path):
    async with graph(tmp_path, blocked="primary") as g:
        first = g.launch(g.source("live-room"), parallel=False)
        await until(g.gates["primary"].entered.is_set)
        audio = np.ones(16000, dtype=np.float32)
        audio_ref = ref(audio)
        second = g.launch(g.source("cancelled-room"), audio=audio, parallel=False)
        del audio
        await until(lambda: g.processor.resource_status()["native"]["operations"] == 2)
        second.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await second
        await until(lambda: g.processor.resource_status()["native"]["operations"] == 1)
        assert audio_ref() is None and caught.value is not None
        assert not first.done() and g.gates["primary"].calls == 1
        g.gates["primary"].release.set()
        assert (await first).success


@pytest.mark.parametrize("stage", ["profiles", "speaker"])
async def test_cancelled_identification_cannot_start_next_stage_or_update_last_seen(tmp_path, stage):
    async with graph(tmp_path, blocked=stage) as g:
        observer = g.launch(g.source())
        await until(g.gates[stage].entered.is_set)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert g.processor.resource_status()["native"]["operations"] > 0
        g.gates[stage].release.set()
        await until(lambda: not g.processor.snapshot_operations())
        assert not g.last_seen and g.processor.get_current_user() is None
        assert all(value() is None for value in g.embeddings)
        if stage == "profiles":
            assert g.gates["speaker"].calls == 0


async def test_privacy_enabled_processor_rejects_missing_lineage_before_model_entry(tmp_path):
    async with graph(tmp_path) as g:
        with pytest.raises(SourcesInvalidated):
            await g.processor.process_command(np.ones(16000, dtype=np.float32))
        assert not g.gates["primary"].calls and not g.processor.snapshot_operations()
