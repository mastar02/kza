"""Ambient VAD state remains owned through failed native reset and retry."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4
from weakref import ref

import pytest

from src.ambient.room_producer import RoomProducer
import src.ambient.room_producer as module
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryOrigin
from tests.integration.test_ambient_producer_privacy import ACTOR, until
from tests.mocks.ambient_graph import ambient_graph, capture_source, chunk

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('trigger', ['expiry', 'revoke'])
async def test_failed_idle_reset_keeps_segmenter_and_privacy_lease_until_actual_retry(tmp_path, monkeypatch, trigger):
    async with ambient_graph(tmp_path) as g:
        failed = [True]
        original = g.transcriber._segmenter_factory
        states, backends = [], []

        def factory():
            state = original()
            predict = state._vad_predict
            class Backend:
                raw = None
                def __call__(self, audio):
                    self.raw = audio.copy()
                    return predict(audio)
                def reset(self):
                    if failed[0]:
                        raise ValueError('Synthetic private recurrent state failure')
                    predict.reset()
                    self.raw = None
            backend = Backend()
            backends.append(backend)  # A supplied backend can also have external owners.
            state._vad_predict = backend
            states.append(ref(state))
            return state

        monkeypatch.setattr(g.transcriber, '_segmenter_factory', factory)
        await g.path.start()
        room = g.transcriber._room_states['synthetic-a']
        source = capture_source(g)
        g.tap.push('synthetic-a', chunk(0.), ts=g.clock.now)
        await until(lambda: bool(states) and room.status()['input_state_retained'] and not room.snapshot_operations())
        raw = ref(backends[0].raw)
        try:
            if trigger == 'expiry':
                g.clock.now += 48*3600
            else:
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            g.transcriber.prune_private()
            await until(lambda: room.status()['error'] == 'ambient_segmenter_reset_failed')
            assert states[0]() is not None
            assert raw() is not None
            assert room.status()['state_retained'] and room.status()['input_state_retained']
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            failed[0] = False
            await g.privacy.begin_close()
        assert states[0]() is None
        assert raw() is None
        assert not room.status()['state_retained']
        assert not g.privacy.resource_status()['writer_lease']
def owned_room(g, factory=None):
    privacy = g.privacy
    def source(room, timestamp):
        return privacy.source_for_capture(str(uuid4()), timestamp, room, origin=MemoryOrigin.AMBIENT)
    def admitted(epoch, room):
        return privacy.scopes.epoch_available(epoch, room, origin=MemoryOrigin.AMBIENT)
    room = RoomProducer('synthetic-a', factory or g.transcriber._segmenter_factory, require_cleanup=True,
        boundary_fn=privacy.scopes.buffer_boundary, source_fn=source, epoch_available=admitted)
    g.transcriber._room_states['synthetic-a'] = room
    g.tap.register_room('synthetic-a')
    room.publish_boundary(privacy.scopes.buffer_boundary('synthetic-a'))
    return room


def stamped(g, items):
    """Use the actual tap's ownership transfer for direct native actor tests."""
    for timestamp, audio, tts_active in items:
        g.tap.push('synthetic-a', audio, ts=timestamp, tts_active=tts_active)
    return g.tap.drain_owned('synthetic-a')


@pytest.mark.parametrize('stop', ['expire', 'revoke', 'close'])
async def test_idle_recurrent_input_has_early_provenance_and_clears_without_new_audio(tmp_path, stop):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        raw = chunk(0.)
        reference = ref(raw)
        await room.feed(stamped(g, [(g.clock.now, raw, False)]))
        del raw
        source = room._source
        assert source is not None and reference() is not None
        assert g.transcriber.conversation_busy(source.conversation_id)
        assert not g.transcriber.conversation_busy(str(uuid4()))
        assert source.origin is MemoryOrigin.AMBIENT
        status = json.dumps(g.transcriber.resource_status())
        assert source.conversation_id not in status and source.capture_id not in status
        if stop == 'close':
            await g.privacy.begin_close()
        else:
            if stop == 'expire':
                g.clock.now += 48*3600
            else:
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            g.transcriber.prune_private()
            await until(lambda: not room.needs_prune() and not room.snapshot_operations())
            assert not g.transcriber.conversation_busy(source.conversation_id)
        assert reference() is None and room._source is None
        assert not room.status()['input_state_retained']


async def test_utterance_close_reset_failure_is_not_hidden_or_followed_by_more_inference(tmp_path):
    async with ambient_graph(tmp_path) as g:
        failed, predictors, models = [True], [], []
        original = g.transcriber._segmenter_factory
        def factory():
            model = original()
            predict = model._vad_predict
            class Backend:
                raw = None
                calls = 0
                def __call__(self, audio):
                    self.calls += 1
                    if self.raw is None:
                        self.raw = audio.copy()
                    return predict(audio)
                def reset(self):
                    if failed[0]:
                        raise ValueError('Synthetic recurrent reset failure.')
                    predict.reset()
                    self.raw = None
            backend = Backend()
            predictors.append(backend)
            models.append(ref(model))
            model._vad_predict = backend
            return model
        room = owned_room(g, factory)
        try:
            await room.feed(stamped(g, [(g.clock.now, chunk(), False), (g.clock.now+.08, chunk(0.), False)]))
            source = room._source
            raw = ref(predictors[0].raw)
            assert source is not None and models[0]().needs_reset
            assert room.status()['reset_pending'] and room.status()['error'] == 'ambient_segmenter_reset_failed'
            assert not room.ready and g.transcriber.conversation_busy(source.conversation_id)
            before = predictors[0].calls
            await room.feed(stamped(g, [(g.clock.now+.16, chunk(), False)]))
            assert predictors[0].calls == before and len(models) == 1
            assert raw() is not None and room._source == source
            failed[0] = False
            await room.prune()
            assert models[0]() is None and raw() is None and room._source is None
            assert not room.status()['reset_pending']
            g.clock.now += 1
            await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
            assert len(models) == 2 and room._source.capture_id != source.capture_id
        finally:
            failed[0] = False
            await g.privacy.begin_close()


async def test_completed_executor_transport_does_not_retain_feed_inputs(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        original = asyncio.to_thread
        transports = []
        async def retained(function, *args, **kwargs):
            result = await original(function, *args, **kwargs)
            if function is module._invoke_room:
                transports.append((function, args, kwargs))
            return result
        monkeypatch.setattr(asyncio, 'to_thread', retained)
        items = [(g.clock.now, chunk(), False), (g.clock.now+.08, chunk(0.), False)]
        references = [ref(item[1]) for item in items]
        observing = asyncio.create_task(room.feed(stamped(g, items)))
        del items
        await observing
        await until(lambda: not room.snapshot_operations())
        assert room.ready and transports
        assert all(value() is None for value in references)
        assert all(args[0].function is None and args[0].args == () for _, args, _ in transports)


async def test_cancelled_queued_native_call_keeps_inputs_until_admitted_work_actually_exits(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        original = asyncio.to_thread
        entered, release = asyncio.Event(), asyncio.Event()
        async def queued(function, *args, **kwargs):
            if function is module._invoke_room:
                entered.set()
                await release.wait()
            return await original(function, *args, **kwargs)
        monkeypatch.setattr(asyncio, 'to_thread', queued)
        audio = chunk()
        reference = ref(audio)
        observer = asyncio.create_task(room.feed(stamped(g, [(g.clock.now, audio, False)])))
        del audio
        errors = []
        try:
            await asyncio.wait_for(entered.wait(), 2)
            observer.cancel()
            try:
                await observer
            except asyncio.CancelledError as error:
                errors.append(error)
            assert errors and reference() is not None and room.snapshot_operations()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await g.privacy.begin_close()
        assert reference() is None and not room.snapshot_operations()
        assert not g.privacy.resource_status()['writer_lease']


async def test_unexpected_native_prune_error_does_not_export_private_traceback(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        references, errors = [], []
        class Payload:
            pass
        def failed(room_id):
            raw = Payload()
            references.append(ref(raw))
            raise ValueError(raw)
        original = room._boundary_fn
        monkeypatch.setattr(room, '_boundary_fn', failed)
        try:
            await room.prune()
        except RuntimeError as error:
            errors.append(error)
        assert errors and str(errors[0]) == 'ambient_room_native_failed'
        assert errors[0].__context__ is None and references[0]() is None
        assert room.status()['reset_pending']
        monkeypatch.setattr(room, '_boundary_fn', original)
        await room.prune()
        assert not room.status()['reset_pending'] and not room.status()['state_retained']


@pytest.mark.parametrize('revoke', ['before-handoff', 'after-persist'])
async def test_recurrent_context_keeps_original_conversation_and_deadline_across_idle_gap(tmp_path, revoke):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        source = room._source
        g.clock.now += 121
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False), (g.clock.now+.08, chunk(), False),
                         (g.clock.now+.16, chunk(0.), False)]))
        assert len(room.ready) == 1
        segment = room.ready.pop()
        assert segment.t0 > source.captured_at and segment.input_source == source
        assert room._source is None  # Actual recurrent reset already succeeded.
        if revoke == 'before-handoff':
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await g.transcriber._handle_segment('synthetic-a', segment)
        rows = await g.store.undistilled_live(require_provenance=True)
        if revoke == 'before-handoff':
            assert not rows and not g.gates['stt'].calls
        else:
            assert len(rows) == 1 and rows[0]['conversation_id'] == source.conversation_id
            assert rows[0]['capture_id'] != source.capture_id
            assert rows[0]['expires_at'] == source.expires_at
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            assert not await g.store.undistilled_live(require_provenance=True)


@pytest.mark.parametrize('conflict', ['origin', 'start', 'expired'])
async def test_handoff_rejects_conflicting_or_expired_buffer_provenance_before_inference(tmp_path, conflict):
    async with ambient_graph(tmp_path) as g:
        source = capture_source(g)
        if conflict == 'expired':
            g.clock.now = source.expires_at
        segment = g.segment()
        if conflict == 'origin':
            source = replace(source, origin=MemoryOrigin.VOICE)
        elif conflict == 'start':
            segment.t0 = source.captured_at-1
        segment.input_source = source
        await g.transcriber._handle_segment('synthetic-a', segment)
        assert not g.gates['stt'].calls
        assert not await g.store.undistilled_live(require_provenance=True)


async def test_cancelled_native_reset_observer_cannot_release_privacy_authority(tmp_path):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        source = room._source
        g.gates['reset'].block()
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        room.publish_boundary(g.privacy.scopes.buffer_boundary('synthetic-a'))
        observer = asyncio.create_task(room.prune())
        errors = []
        try:
            await until(g.gates['reset'].entered.is_set)
            observer.cancel()
            try:
                await observer
            except asyncio.CancelledError as error:
                errors.append(error)
            assert errors and room.snapshot_operations()
            assert g.transcriber.conversation_busy(source.conversation_id)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            g.gates['reset'].release.set()
            await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease']
        assert not room.status()['state_retained']


async def test_close_retains_authority_when_native_reset_transport_is_unavailable(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        audio = chunk(0.)
        reference = ref(audio)
        await room.feed(stamped(g, [(g.clock.now, audio, False)]))
        del audio
        failed = [True]
        original = asyncio.to_thread
        async def transport(function, *args, **kwargs):
            if function is module._invoke_room and failed[0]:
                raise RuntimeError('Synthetic private unavailable executor')
            return await original(function, *args, **kwargs)
        monkeypatch.setattr(asyncio, 'to_thread', transport)
        try:
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            assert reference() is not None and room.status()['reset_pending']
            assert not room.begin_close().done()
        finally:
            failed[0] = False
            await g.privacy.begin_close()
        assert reference() is None and not g.privacy.resource_status()['writer_lease']
