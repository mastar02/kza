"""C0: existing native ownership distinguishes a revoked source from its peer.

This characterizes current public processing/erasure APIs.  It does not provide
a PRODUCERS driver, certify an inventory, or use global capture suspension.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT, producer_context
from src.memory.producer_scopes import current_producer_scope
from src.memory.prompt_context import SourcesInvalidated
from src.pipeline.command_processor import CommandProcessor
from src.stt.whisper_fast import STTResult
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source


pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(4):
        while not predicate():
            await asyncio.sleep(.001)


class NativeStage:
    """A bounded native leaf; its owner Future is the retirement receipt."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.thread = None
        self.sources = None
        self.scope = None

    def transcribe(self, label):
        self.thread = threading.current_thread()
        self.sources = PRODUCER_CONTEXT.get().sources
        scope = current_producer_scope()
        assert scope is not None
        self.scope = scope.snapshot()
        self.entered.set()
        try:
            assert self.release.wait(8), "fixture STT gate was not released"
            return STTResult(f"Synthetic {label} command", 1., no_speech_prob=.01, avg_logprob=-.1)
        finally:
            self.returned.set()


class TwoSourceSTT:
    """Both inputs use the same real CommandProcessor model lock."""

    def __init__(self, subject, unrelated):
        self.subject, self.unrelated = subject, unrelated
        self.calls = []

    def transcribe_with_confidence(self, audio, sample_rate=16000):
        assert sample_rate == 16000 and audio.shape == (16000,)
        value = float(audio[0])
        assert value in (.125, .25) and np.all(audio == value)
        label, stage = ("S", self.subject) if value == .125 else ("U", self.unrelated)
        self.calls.append(label)
        return stage.transcribe(label)


def launch(processor, source, value):
    with producer_context("synthetic-room", (source,)):
        return asyncio.create_task(processor.process_command(
            np.full(16000, value, np.float32), use_parallel=False))


async def test_current_native_retirement_is_scoped_to_the_revoked_conversation(tmp_path):
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=.03)
    subject_stage, unrelated_stage = NativeStage(), NativeStage()
    processor = owner = None
    callers = []
    try:
        await privacy.acall(privacy.initialize)
        subject = memory_source(now=clock.now)
        unrelated = memory_source(now=clock.now)
        subject_original, unrelated_original = subject.to_dict(), unrelated.to_dict()
        model = TwoSourceSTT(subject_stage, unrelated_stage)
        processor = CommandProcessor(model, privacy=privacy, close_timeout_s=.03)
        owner = ErasureCoordinator(privacy, close_timeout_s=.03)
        subject_caller = launch(processor, subject, .125)
        callers.append(subject_caller)
        await until(subject_stage.entered.is_set)
        assert subject_stage.sources == (subject,)
        assert subject_stage.scope == ((subject,), 0)

        unrelated_caller = launch(processor, unrelated, .25)
        callers.append(unrelated_caller)
        await until(lambda: processor.resource_status()["native"]["operations"] == 2)
        # U is admitted to actual owned work behind the same model lock, not
        # a standalone metadata scope manufactured to resemble another room.
        operations = processor.snapshot_operations()
        scopes = [task.get_context().run(current_producer_scope) for task in operations]
        assert all(scope is not None for scope in scopes)
        affinities = [scope.snapshot() for scope in scopes]
        assert set(affinities) == {((subject,), 0), ((unrelated,), 0)}
        assert not unrelated_stage.entered.is_set() and not unrelated_caller.done()
        assert processor.conversation_busy(subject.conversation_id)
        assert processor.conversation_busy(unrelated.conversation_id)

        operation = await owner.request(subject.conversation_id, MemoryReviewer("synthetic-admin"))
        subject_caller.cancel()
        subject_outcome, = await asyncio.gather(subject_caller, return_exceptions=True)
        assert isinstance(subject_outcome, (asyncio.CancelledError, SourcesInvalidated))
        assert not subject_stage.returned.is_set()
        assert processor.conversation_busy(subject.conversation_id)
        assert processor.conversation_busy(unrelated.conversation_id)
        receipt = await privacy.acall(privacy.ledger.erasures.snapshot, operation["id"])
        producer, = [step for step in receipt["steps"] if step["participant"] == "producers"]
        assert producer["state"] != "complete"
        assert producer["last_error"] == "participant_unregistered"
        assert receipt["revoked"] and receipt["state"] != "complete"

        subject_stage.release.set()
        await until(unrelated_stage.entered.is_set)
        # Gate return alone is insufficient: busy() must have reaped S's
        # actual owned task/callback while U's actual native call is retained.
        await until(lambda: not processor.conversation_busy(subject.conversation_id))
        assert subject_stage.returned.is_set()
        assert unrelated_stage.sources == (unrelated,)
        assert unrelated_stage.scope == ((unrelated,), 0)
        assert not unrelated_stage.returned.is_set() and not unrelated_caller.done()
        assert processor.conversation_busy(unrelated.conversation_id)
        assert privacy.resource_status()["writer_lease"] and privacy.ledger._db is not None
        assert not privacy.sources_available((subject,))
        assert privacy.sources_available((unrelated,))
        assert subject.to_dict() == subject_original
        assert unrelated.to_dict() == unrelated_original

        unrelated_stage.release.set()
        result = await unrelated_caller
        assert result.success and result.text == "Synthetic U command"
        assert result.stt_confidence.text == "Synthetic U command"
        await until(lambda: not processor.snapshot_operations())
        assert unrelated_stage.returned.is_set()
        assert not processor.conversation_busy(subject.conversation_id)
        assert not processor.conversation_busy(unrelated.conversation_id)
        assert model.calls == ["S", "U"]
        assert unrelated.to_dict() == unrelated_original
        assert privacy.sources_available((unrelated,))
    finally:
        # Includes setup/assertion failures. Release both leaves before any
        # close and await the real processor/native work, not these Events.
        subject_stage.release.set()
        unrelated_stage.release.set()
        await asyncio.gather(*callers, return_exceptions=True)
        try:
            if owner is not None:
                await owner.begin_close()
        finally:
            try:
                if processor is not None:
                    await processor.begin_close()
            finally:
                await privacy.begin_close()
    assert privacy.ledger._db is None and not privacy.resource_status()["writer_lease"]
