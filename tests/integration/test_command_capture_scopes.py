"""Finite command and capture affinity includes queued arguments and failures."""

import asyncio
from dataclasses import dataclass, field
import json
import threading
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.memory.producer_context import producer_context
from src.memory.prompt_context import SourcesInvalidated
from src.core.owned_work import WorkCloseTimeout
from tests.integration.test_capture_producer_privacy import ACTOR, graph as capture_graph, until
from tests.integration.test_command_producer_privacy import graph as command_graph

pytestmark = pytest.mark.asyncio


@dataclass
class RawInput:
    audio: np.ndarray = field(default_factory=lambda: np.ones(1600, dtype=np.float32))


async def test_queued_auxiliary_input_keeps_caller_lineage_and_cannot_enter_after_revocation(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'auxiliary')
        other = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'other')
        entered, release = threading.Event(), threading.Event()
        calls = []
        def hold():
            entered.set()
            assert release.wait(3)
        first = asyncio.create_task(room.call(hold, sources=(other,)))
        queued = None
        try:
            await until(entered.is_set)
            payload = RawInput()
            reference = ref(payload)
            with producer_context('synthetic-a', (source,)):
                queued = asyncio.create_task(room.call(lambda value: calls.append(value), payload))
            del payload
            await until(lambda: room.resource_status()['controllers'] == 2)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            release.set()
            await first
            with pytest.raises(SourcesInvalidated) as retained:
                await queued
            assert not calls and reference() is None
            assert retained.value.__context__ is None
        finally:
            release.set()
            await asyncio.gather(first, *(tuple([queued]) if queued is not None else ()), return_exceptions=True)


@pytest.mark.parametrize('stage', ['primary', 'shadow', 'profiles', 'speaker', 'emotion'])
@pytest.mark.parametrize('stop', ['cancel', 'revoke'])
async def test_command_native_scope_survives_controller_exit_without_blocking_other_sources(tmp_path, stage, stop):
    async with command_graph(tmp_path, blocked=stage, shadow=stage=='shadow', veto=stage=='shadow') as g:
        source = g.source()
        observer = g.launch(source)
        try:
            await until(g.gates[stage].entered.is_set)
            if stop == 'cancel':
                observer.cancel()
            else:
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: g.processor.resource_status()['commands'] == 0)
            assert g.processor.conversation_busy(source.conversation_id)
            assert not g.processor.conversation_busy(str(uuid4()))
            status = json.dumps(g.processor.resource_status())
            assert source.conversation_id not in status and 'Synthetic private' not in status
        finally:
            g.gates[stage].release.set()
            await until(lambda: not g.processor.snapshot_operations())
        assert not g.processor.conversation_busy(source.conversation_id)
        assert g.processor.resource_status()['source_scopes']['scopes'] == 0


@pytest.mark.parametrize('stop', ['cancel', 'close'])
async def test_direct_emotion_controller_and_native_result_remain_owned(tmp_path, monkeypatch, stop):
    async with command_graph(tmp_path) as g:
        source = g.source()
        entered, release = threading.Event(), threading.Event()
        results = []
        def detect(audio):
            entered.set()
            assert release.wait(3)
            result = RawInput()
            results.append(ref(result))
            return result
        monkeypatch.setattr(g.processor.emotion_detector, 'detect', detect)
        audio = np.ones(1600, dtype=np.float32)
        reference = ref(audio)
        with producer_context('synthetic-room', (source,)):
            observer = asyncio.create_task(g.processor.detect_emotion(audio))
        del audio
        try:
            await until(entered.is_set)
            assert g.processor.resource_status()['emotions'] == 1
            if stop == 'cancel':
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.processor.conversation_busy(source.conversation_id)
            assert not g.processor.conversation_busy(str(uuid4()))
            assert reference() is not None and g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await g.privacy.begin_close()
        await until(lambda: reference() is None and results and results[0]() is None)
        assert not g.processor.snapshot_operations()
        assert g.processor.resource_status()['source_scopes']['scopes'] == 0


@pytest.mark.parametrize('api', ['command', 'emotion'])
async def test_command_scope_capacity_error_cannot_keep_unadmitted_audio(tmp_path, api):
    async with command_graph(tmp_path, blocked='primary') as g:
        g.processor._scopes._capacity = 1
        source = g.source()
        first = g.launch(source, parallel=False)
        try:
            await until(g.gates['primary'].entered.is_set)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            audio = np.ones(1600, dtype=np.float32)
            reference = ref(audio)
            with producer_context('synthetic-room', (g.source(),)):
                operation = g.processor.process_command(audio) if api == 'command' else g.processor.detect_emotion(audio)
                rejected = asyncio.create_task(operation)
            del audio, operation
            with pytest.raises(RuntimeError, match='producer_scope_capacity') as retained:
                await rejected
            assert retained.value is not None and reference() is None
        finally:
            g.gates['primary'].release.set()
            await until(lambda: not g.processor.snapshot_operations())


async def test_busy_query_prunes_idle_identification_and_emotion_after_revocation(tmp_path):
    async with command_graph(tmp_path) as g:
        source = g.source()
        await g.launch(source)
        assert g.processor._current_user is not None and g.processor._current_emotion is not None
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        assert not g.processor.conversation_busy(source.conversation_id)
        assert g.processor._current_user is g.processor._current_emotion is None
        assert g.processor._current_sources == ()


@pytest.mark.parametrize('stage', ['wake', 'vad', 'reset'])
async def test_capture_native_scope_keeps_exact_affinity_after_controller_cancel(tmp_path, stage):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        if stage == 'reset':
            g.feed()
            await room.step()
            source = room.source
            g.gates['synthetic-a']['reset'].block()
            observer = asyncio.create_task(room.call(room._reset, sources=(source,)))
        else:
            g.gates['synthetic-a'][stage].block()
            g.wakes['synthetic-a'].fire = stage == 'vad'
            g.feed()
            observer = asyncio.create_task(room.step())
        try:
            await until(g.gates['synthetic-a'][stage].entered.is_set)
            if stage != 'reset':
                source = room.source
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await until(lambda: room.resource_status()['controllers'] == 0)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            assert room.conversation_busy(source.conversation_id)
            assert not room.conversation_busy(str(uuid4()))
            assert source.conversation_id not in json.dumps(room.resource_status())
        finally:
            g.gates['synthetic-a'][stage].release.set()
            await until(lambda: not room.snapshot_operations())
        assert not room.conversation_busy(source.conversation_id)


async def test_unknown_auxiliary_raw_input_stays_pending_until_native_exit(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        entered, release = threading.Event(), threading.Event()
        def hold(value):
            entered.set()
            assert release.wait(3)
            return value
        payload = RawInput()
        reference = ref(payload)
        observer = asyncio.create_task(room.call(hold, payload))
        del payload
        try:
            await until(entered.is_set)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            assert room.conversation_busy(str(uuid4()))
            assert room.resource_status()['source_scopes']['unknown'] == 1
            assert reference() is not None
        finally:
            release.set()
            await until(lambda: not room.snapshot_operations())
        await until(lambda: reference() is None)
        assert not room.conversation_busy(str(uuid4()))


async def test_revocation_during_auxiliary_callback_rejects_its_late_result(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'auxiliary')
        entered, release = threading.Event(), threading.Event()
        results = []
        def hold():
            entered.set()
            assert release.wait(3)
            result = RawInput()
            results.append(ref(result))
            return result
        observer = asyncio.create_task(room.call(hold, sources=(source,)))
        try:
            await until(entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            release.set()
            with pytest.raises(SourcesInvalidated) as retained:
                await observer
            assert retained.value.__context__ is None
            assert results and results[0]() is None
        finally:
            release.set()
            await asyncio.gather(observer, return_exceptions=True)


async def test_capture_controller_capacity_closes_rejected_inputs_and_cancels_waiters(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'auxiliary')
        entered, release = threading.Event(), threading.Event()
        effects = []
        def hold():
            entered.set()
            assert release.wait(3)
        operations = [asyncio.create_task(room.call(hold, sources=(source,)))]
        try:
            await until(entered.is_set)
            for _ in range(7):
                operations.append(asyncio.create_task(room.call(lambda: effects.append('late'), sources=(source,))))
            await until(lambda: room.resource_status()['controllers'] == 8)
            payload = RawInput()
            reference = ref(payload)
            rejected = asyncio.create_task(room.call(lambda value: effects.append(value), payload, sources=(source,)))
            del payload
            with pytest.raises(RuntimeError, match='capture_controller_capacity') as retained:
                await rejected
            assert retained.value is not None and reference() is None
            closing = room.begin_close()
            await until(lambda: room.resource_status()['controllers'] == 0)
            assert not closing.done() and not effects
        finally:
            release.set()
            await room.begin_close()
            await asyncio.gather(*operations, return_exceptions=True)
        assert not effects and not room.snapshot_operations()


async def test_auxiliary_failure_cannot_escape_with_original_raw_traceback(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'auxiliary')
        def fail(value):
            raise ValueError('synthetic private error', value)
        payload = RawInput()
        reference = ref(payload)
        operation = asyncio.create_task(room.call(fail, payload, sources=(source,)))
        del payload
        retained = None
        try:
            await operation
        except Exception as error:
            retained = error
        assert retained is not None
        assert reference() is None
        assert type(retained) is RuntimeError and str(retained) == 'capture_auxiliary_failed'
        assert retained.__context__ is None


async def test_scope_capacity_rejection_drops_unadmitted_raw_arguments(tmp_path):
    async with capture_graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        room._scopes._capacity = 1
        entered, release = threading.Event(), threading.Event()
        def hold():
            entered.set()
            assert release.wait(3)
        first = asyncio.create_task(room.call(hold))
        try:
            await until(entered.is_set)
            payload = RawInput()
            reference = ref(payload)
            rejected = asyncio.create_task(room.call(lambda value: None, payload))
            del payload
            with pytest.raises(RuntimeError, match='producer_scope_capacity') as retained:
                await rejected
            assert retained.value is not None and reference() is None
        finally:
            release.set()
            await first
