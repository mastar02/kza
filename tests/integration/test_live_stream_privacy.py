"""Actual live queues and retained socket delivery obey capture authority."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.dashboard.live_event_bus import LiveEvent, LiveEventBus, LiveEventType
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')
PRIVATE = 'Synthetic private live utterance'


@asynccontextmanager
async def graph(tmp_path):
    async with AsyncExitStack() as stack:
        clock = Clock(2_000_000_000.)
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=.03)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        bus = LiveEventBus(queue_size=2, source_policy=privacy.ledger, clock=clock, close_timeout_s=.03)
        privacy.add_borrower(bus.begin_close)
        stack.push_async_callback(bus.aclose)
        bus.start()
        yield privacy, bus, clock


def event(source, **payload):
    return LiveEvent(LiveEventType.TURN, payload={'stt': PRIVATE, 'tts':'Synthetic reply', **payload}, sources=(source,))


async def test_unsourced_or_revoked_private_frames_never_enter_policy_queue(tmp_path):
    async with graph(tmp_path) as (privacy, bus, clock):
        _, queue = await bus.subscribe()
        await bus.publish(LiveEvent(LiveEventType.TURN, payload={'stt': PRIVATE}))
        assert queue.empty()
        source = memory_source(now=clock.now)
        await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await bus.publish(event(source))
        assert queue.empty() and bus.resource_status()['rejected'] == 2


async def test_frame_is_detached_from_producer_and_has_original_source_deadline(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        _, queue = await bus.subscribe()
        source = memory_source(now=clock.now-20)
        original = event(source)
        await bus.publish(original)
        original.payload['stt'] = 'Changed after publication'
        admitted = queue.get_nowait()
        frame = admitted.to_frame()
        assert frame['payload']['stt'] == PRIVATE
        assert frame['expires_at'] == source.expires_at
        assert source.conversation_id not in str(frame) and source.capture_id not in str(frame)


async def test_idle_source_expiry_retires_queued_and_already_dequeued_records(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        _, first = await bus.subscribe()
        _, second = await bus.subscribe()
        source = memory_source(now=clock.now, ttl=1)
        await bus.publish(event(source))
        held = first.get_nowait()
        assert held.to_frame()['payload']['stt'] == PRIVATE
        clock.now += 1
        await until(lambda: second.empty() and held.payload == {})
        assert held.to_frame() is None


async def test_conversation_retirement_keeps_other_room_and_never_relabels_old_record(tmp_path):
    async with graph(tmp_path) as (privacy, bus, clock):
        _, queue = await bus.subscribe()
        source, other = (memory_source(now=clock.now) for _ in range(2))
        await bus.publish(event(source))
        await bus.publish(event(other, stt='Other live room'))
        await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await until(lambda: queue.qsize() == 1)
        assert queue.get_nowait().payload['stt'] == 'Other live room'


async def test_blocked_socket_send_is_owned_through_observer_cancellation_and_close_timeout(tmp_path):
    async with graph(tmp_path) as (privacy, bus, clock):
        sub, _ = await bus.subscribe()
        entered, release = asyncio.Event(), asyncio.Event()
        delivered = []
        async def send(frame):
            entered.set()
            await release.wait()
            delivered.append(frame['type'])
        observing = asyncio.create_task(bus.stream(sub, send))
        source = memory_source(now=clock.now)
        await bus.publish(event(source))
        try:
            await until(entered.is_set)
            observing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observing
            with pytest.raises(WorkCloseTimeout):
                await bus.aclose()
            assert bus.resource_status()['active_sends'] == 1
            assert not bus.begin_close().done()
            assert privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await bus.begin_close()
        assert delivered == ['turn'] and bus.resource_status()['active_sends'] == 0


async def test_delivered_last_frame_gets_retirement_control_without_more_events(tmp_path):
    async with graph(tmp_path) as (privacy, bus, clock):
        sub, _ = await bus.subscribe()
        sent = []
        async def send(frame):
            sent.append(frame)
        watching = asyncio.create_task(bus.stream(sub, send))
        source = memory_source(now=clock.now)
        try:
            await bus.publish(event(source))
            await until(lambda: len(sent) == 1)
            await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await until(lambda: len(sent) == 2)
            assert sent[1]['type'] == 'retire'
            assert sent[1]['payload']['event_ids'] == [sent[0]['event_id']]
            assert PRIVATE not in str(sent[1])
        finally:
            await bus.unsubscribe(sub)
            await watching


async def test_actual_pipeline_passes_active_derived_sources_to_live_bus(tmp_path):
    from tests.unit.pipeline.test_voice_pipeline_event_publish import _make_pipeline
    async with graph(tmp_path) as (_, bus, clock):
        _, queue = await bus.subscribe()
        source, derived = (memory_source(now=clock.now) for _ in range(2))
        router = SimpleNamespace(_active_sources=lambda:(source,derived))
        pipeline = _make_pipeline(event_bus=bus, request_router=router)
        await pipeline._publish_turn_event(SimpleNamespace(room_id='synthetic-room'), {'text':PRIVATE,'response':'Synthetic reply'})
        admitted = queue.get_nowait()
        assert admitted.sources == (source,derived)
        assert admitted.payload['stt'] == PRIVATE


@asynccontextmanager
async def asgi_socket(bus, authorization):
    from fastapi import FastAPI
    from src.dashboard.observability import register_observability_routes
    app = FastAPI()
    register_observability_routes(app, event_bus=bus, use_mocks=False)
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    scope = {'type':'websocket','path':'/ws/live','raw_path':b'/ws/live','root_path':'',
        'scheme':'ws','query_string':b'', 'server':('localhost',80),'client':('127.0.0.1',12345),
        'headers':[(b'authorization',authorization.encode())], 'subprotocols':[], 'state':{}}
    await incoming.put({'type':'websocket.connect'})
    task = asyncio.create_task(app(scope, incoming.get, outgoing.put))
    try:
        async with asyncio.timeout(3):
            assert (await outgoing.get())['type'] == 'websocket.accept'
        yield incoming, outgoing
    finally:
        await incoming.put({'type':'websocket.disconnect','code':1000})
        async with asyncio.timeout(3):
            await task


async def test_real_authenticated_asgi_route_serializes_deadline_and_retires_then_disconnects(tmp_path, dashboard_auth):
    import json
    async with graph(tmp_path) as (privacy, bus, clock):
        source = memory_source(now=clock.now)
        async with asgi_socket(bus, dashboard_auth['Authorization']) as (_, outgoing):
            await bus.publish(event(source))
            async with asyncio.timeout(3):
                message = await outgoing.get()
            frame = json.loads(message['text'])
            assert frame['payload']['stt'] == PRIVATE and frame['expires_at'] == source.expires_at
            await privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            async with asyncio.timeout(3):
                retired = json.loads((await outgoing.get())['text'])
            assert retired['type'] == 'retire' and retired['payload']['event_ids'] == [frame['event_id']]
        assert bus.subscriber_count == 0 and bus.resource_status()['active_sends'] == 0


async def test_session_rotation_is_rechecked_before_actual_asgi_send(tmp_path, dashboard_auth, monkeypatch):
    async with graph(tmp_path) as (_, bus, clock):
        async with asgi_socket(bus, dashboard_auth['Authorization']) as (_, outgoing):
            monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-rotated-session')
            await bus.publish(event(memory_source(now=clock.now)))
            async with asyncio.timeout(3):
                message = await outgoing.get()
            assert message['type'] == 'websocket.close' and message['code'] == 1008
        assert bus.subscriber_count == 0


async def test_idle_disconnect_releases_subscriber_without_another_publication(tmp_path):
    async with graph(tmp_path) as (_, bus, _):
        sub, _ = await bus.subscribe()
        inbound = asyncio.Queue()
        sent = []
        async def send(frame):
            sent.append(frame)
        watching = asyncio.create_task(bus.stream(sub, send, receiver=inbound.get))
        await inbound.put({'type':'websocket.disconnect','code':1000})
        async with asyncio.timeout(3):
            await watching
        assert bus.subscriber_count == 0 and not sent


async def test_policy_bus_caps_subscribers_without_blocking_publish(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        for _ in range(16):
            await bus.subscribe()
        with pytest.raises(RuntimeError, match='live_subscriber_capacity'):
            await bus.subscribe()
        assert bus.subscriber_count == 16
        for _ in range(20):
            await bus.publish(event(memory_source(now=clock.now)))
        assert bus.resource_status()['records'] <= 48


async def test_opaque_objects_and_callback_context_never_enter_live_record_or_sender(tmp_path):
    from src.core.logging import LogContext, get_context
    from weakref import ref
    class Payload:
        def __str__(self):
            raise AssertionError('opaque payload formatted')
    async with graph(tmp_path) as (_, bus, clock):
        sub, queue = await bus.subscribe()
        opaque = Payload()
        reference = ref(opaque)
        with LogContext(private=opaque):
            await bus.publish(event(memory_source(now=clock.now), unknown=opaque))
            await bus.publish(event(memory_source(now=clock.now), stt=opaque))
        opaque = None
        assert reference() is None and queue.qsize() == 1
        observed = []
        async def send(frame):
            observed.append((get_context(),frame))
        with LogContext(private=Payload()):
            watching = asyncio.create_task(bus.stream(sub, send))
        try:
            await until(lambda: bool(observed))
            assert observed[0][0] == {} and 'unknown' not in observed[0][1]['payload']
        finally:
            await bus.unsubscribe(sub)
            await watching


async def test_many_valid_strings_cannot_exceed_total_live_payload_budget(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        _, queue = await bus.subscribe()
        await bus.publish(event(memory_source(now=clock.now), **{key:'x'*8192 for key in ('stt','tts','text','body','title')}))
        assert queue.empty() and bus.resource_status()['rejected'] == 1


async def test_idle_authenticated_socket_is_closed_when_session_rotates(tmp_path, dashboard_auth, monkeypatch):
    async with graph(tmp_path) as (_, bus, _):
        async with asgi_socket(bus, dashboard_auth['Authorization']) as (_, outgoing):
            monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-idle-rotation')
            async with asyncio.timeout(3):
                message = await outgoing.get()
            assert message['type'] == 'websocket.close' and message['code'] == 1008
        assert bus.subscriber_count == 0


async def test_actual_socket_close_callback_is_retained_after_payload_has_drained(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        sub, _ = await bus.subscribe()
        closing, release = asyncio.Event(), asyncio.Event()
        async def send(frame):
            pass
        async def close():
            closing.set()
            await release.wait()
        observing = asyncio.create_task(bus.stream(sub, send, close))
        await bus.publish(event(memory_source(now=clock.now)))
        await until(lambda: bus.resource_status()['senders'] == 1)
        try:
            with pytest.raises(WorkCloseTimeout):
                await bus.aclose()
            assert closing.is_set() and bus.resource_status()['active_sends'] == 1
        finally:
            release.set()
            await observing
            await bus.begin_close()
        assert bus.resource_status()['active_sends'] == 0


async def test_pending_failed_commit_retires_transient_frame_without_resurrecting_it(tmp_path):
    import threading
    async with graph(tmp_path) as (privacy, bus, clock):
        _, queue = await bus.subscribe()
        source = memory_source(now=clock.now)
        await bus.publish(event(source))
        retained = queue.get_nowait()
        entered, release = threading.Event(), threading.Event()
        connection = privacy.ledger._db
        class FailedCommit:
            def __getattr__(self, name):
                return getattr(connection, name)
            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError('synthetic_live_commit_failure')
        privacy.ledger._db = FailedCommit()
        deleting = asyncio.create_task(privacy.acall(privacy.ledger.delete_conversation,source.conversation_id,ACTOR))
        try:
            await until(entered.is_set)
            assert retained.payload == {} and retained.to_frame() is None
        finally:
            release.set()
            with pytest.raises(OSError,match='synthetic_live_commit_failure'):
                await deleting
            privacy.ledger._db = connection
        assert privacy.ledger.sources_available((source,))
        assert retained.payload == {} and retained.to_frame() is None


async def test_conversation_index_releases_metadata_when_queue_record_is_released(tmp_path):
    async with graph(tmp_path) as (_, bus, clock):
        _, queue = await bus.subscribe()
        source = memory_source(now=clock.now)
        await bus.publish(event(source))
        retained = queue.get_nowait()
        assert source.conversation_id in bus._retention._by_conversation
        del retained
        assert source.conversation_id not in bus._retention._by_conversation


async def test_missing_authority_fails_closed_even_with_claimed_source():
    bus = LiveEventBus()
    _, queue = await bus.subscribe()
    await bus.publish(event(memory_source()))
    assert queue.empty() and bus.resource_status()['rejected'] == 1
    await bus.aclose()


async def test_revocation_between_projection_and_indexing_cannot_admit_old_payload(tmp_path, monkeypatch):
    import threading
    import src.dashboard.live_retention as module
    async with graph(tmp_path) as (privacy, bus, clock):
        _, queue = await bus.subscribe()
        source = memory_source(now=clock.now)
        project = module.project_live_payload
        outcomes = []
        def project_during_revocation(payload):
            result = project(payload)
            def revoke():
                privacy.ledger.delete_conversation(source.conversation_id,ACTOR)
                outcomes.append(True)
            thread = threading.Thread(target=revoke)
            thread.start()
            thread.join(3)
            assert not thread.is_alive() and outcomes == [True]
            return result
        monkeypatch.setattr(module,'project_live_payload',project_during_revocation)
        await bus.publish(event(source))
        assert queue.empty() and bus.resource_status()['rejected'] == 1


async def test_live_overflow_and_failed_publication_logs_omit_private_context(tmp_path, monkeypatch):
    import logging
    from src.core.logging import LogContext, LogConfig, LogFormat, StructuredFormatter
    import src.pipeline.voice_pipeline as module
    from tests.unit.pipeline.test_voice_pipeline_event_publish import _make_pipeline
    messages = []
    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(StructuredFormatter(LogConfig(format=LogFormat.JSON)).format(record))
    class FailedBus:
        async def publish(self, admitted):
            raise RuntimeError(PRIVATE)
    handler = Capture()
    targets = [logging.getLogger('src.dashboard.live_event_bus'), logging.getLogger('src.pipeline.voice_pipeline')]
    levels = [target.level for target in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(logging.DEBUG)
    monkeypatch.setattr(module,'_PUBLISH_FAILURE_LOGGED',False)
    try:
        async with graph(tmp_path) as (_, bus, clock):
            with LogContext(private=PRIVATE):
                await bus.subscribe()
                for _ in range(3):
                    await bus.publish(event(memory_source(now=clock.now)))
                pipeline = _make_pipeline(event_bus=FailedBus())
                await pipeline._publish_turn_event(None,{'text':PRIVATE})
                await bus.unsubscribe(0)
            assert messages and PRIVATE not in '\n'.join(messages)
            assert any('dropped' in message for message in messages)
            assert any('Live publication deferred' in message for message in messages)
    finally:
        for target, level in zip(targets,levels):
            target.removeHandler(handler)
            target.setLevel(level)
