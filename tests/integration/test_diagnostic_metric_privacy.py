"""Diagnostic sinks receive bounded numeric data without voice payloads."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import threading
import time
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.core.logging import LogContext, LogConfig, LogFormat, StructuredFormatter
from src.core.owned_work import WorkCloseTimeout
from src.obs.metrics_emitter import MetricsEmitter
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio
PRIVATE = 'Synthetic private transcript and household identity'


@asynccontextmanager
async def receiver():
    lines = []
    clients = set()

    async def receive(reader, writer):
        clients.add(writer)
        try:
            while line := await reader.readline():
                lines.append(line.decode())
        finally:
            clients.discard(writer)
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(receive, '127.0.0.1', 0)
    try:
        yield server.sockets[0].getsockname()[1], lines
    finally:
        server.close()
        await server.wait_closed()
        for writer in tuple(clients):
            writer.close()
            await writer.wait_closed()


def request(emitter, **kwargs):
    emitter.emit_request(user_id=PRIVATE, zone_id=PRIVATE, text=PRIVATE,
        intent=PRIVATE, path='fast', success=True,
        timings={'stt': 12., PRIVATE: 3., 'total': float('nan')},
        entity_id=PRIVATE, service=PRIVATE, **kwargs)


async def test_request_wake_and_error_share_private_free_disk_and_tcp_projection(tmp_path):
    legacy = tmp_path/'metrics.jsonl'
    original = json.dumps({'text': PRIVATE})+'\n'
    legacy.write_text(original)
    async with receiver() as (port, lines):
        emitter = MetricsEmitter(str(legacy), logstash_host='127.0.0.1', logstash_port=port)
        try:
            request(emitter)
            emitter.emit_wake(room_id=PRIVATE, matched=False, wake_word=PRIVATE,
                matched_via='rejected', text=PRIVATE, audio_duration_ms=80., wake_stt_ms=20.,
                rejection_reason=PRIVATE)
            emitter.emit_error(module=PRIVATE, error_msg=PRIVATE, context={'private': PRIVATE})
            await emitter.flush()
            await until(lambda: len(lines) == 3)
            documents = [json.loads(line) for line in lines]
            assert {doc['event_type'] for doc in documents} == {'request', 'wake', 'error'}
            assert PRIVATE not in ''.join(lines)
            assert PRIVATE not in emitter.path.read_text()
            assert legacy.read_text() == original
            assert emitter.path != legacy
            request_doc = next(doc for doc in documents if doc['event_type'] == 'request')
            assert request_doc['stt_ms'] == 12.
            assert request_doc['success'] is True
            assert 'total_ms' not in request_doc
            assert all(doc['diagnostic_schema'] == 1 for doc in documents)
        finally:
            await emitter.aclose()


@pytest.mark.parametrize('caller', ['loop', 'native'])
async def test_blocked_metric_io_is_off_loop_and_close_retains_actual_worker(tmp_path, monkeypatch, caller):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), close_timeout_s=.03)
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    original = emitter._write

    def held_write(doc):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(3)
        original(doc)

    monkeypatch.setattr(emitter, '_write', held_write)
    try:
        if caller == 'loop':
            request(emitter)
        else:
            await asyncio.to_thread(request, emitter)
        await until(entered.is_set)
        before = time.perf_counter()
        await asyncio.sleep(.01)
        assert time.perf_counter()-before < .075
        with pytest.raises(WorkCloseTimeout):
            await emitter.aclose()
        assert not emitter.begin_close().done()
    finally:
        release.set()
        await emitter.begin_close()
    assert emitter.resource_status()['pending'] == 0


async def test_metric_transport_cannot_retain_arbitrary_request_context(tmp_path, monkeypatch):
    class Payload:
        pass

    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'))
    entered, release = threading.Event(), threading.Event()
    contexts = []
    original = emitter._write

    def held_write(doc):
        from src.core.logging import get_context
        contexts.append(get_context())
        entered.set()
        assert release.wait(3)
        original(doc)

    monkeypatch.setattr(emitter, '_write', held_write)
    payload = Payload()
    reference = ref(payload)
    try:
        with LogContext(private=payload):
            request(emitter)
        payload = None
        await until(entered.is_set)
        assert reference() is None
        assert contexts == [{}]
    finally:
        release.set()
        await emitter.aclose()


async def test_saturation_drops_metrics_without_unbounded_posted_jobs(tmp_path, monkeypatch):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), capacity=2)
    entered, release = threading.Event(), threading.Event()

    def held_write(doc):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(emitter, '_write', held_write)
    try:
        for _ in range(50):
            request(emitter)
        assert emitter.resource_status()['pending'] == 2
        assert emitter.resource_status()['dropped'] == 48
        await until(entered.is_set)
    finally:
        release.set()
        await emitter.aclose()


async def test_unknown_numeric_objects_are_not_formatted_or_converted(tmp_path):
    class PrivateValue:
        def __str__(self):
            raise AssertionError('private diagnostic value formatted')

        def __float__(self):
            raise AssertionError('private diagnostic value converted')

    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'))
    try:
        emitter.emit_request(user_id=PrivateValue(), zone_id=PrivateValue(), text=PrivateValue(),
            intent=PrivateValue(), path=PrivateValue(), success=True,
            timings={'stt': PrivateValue(), 'total': np.array([1.])})
        await emitter.flush()
        doc = json.loads(emitter.path.read_text())
        assert 'stt_ms' not in doc and 'total_ms' not in doc
    finally:
        await emitter.aclose()


async def test_close_waits_for_a_reserved_native_post_before_it_reaches_the_loop(tmp_path, monkeypatch):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), close_timeout_s=.03)
    entered, release = threading.Event(), threading.Event()
    post = emitter._loop.call_soon_threadsafe

    def held_post(callback, *args, **kwargs):
        if callback == emitter._start:
            entered.set()
            assert release.wait(3)
        return post(callback, *args, **kwargs)

    monkeypatch.setattr(emitter._loop, 'call_soon_threadsafe', held_post)
    posting = asyncio.create_task(asyncio.to_thread(request, emitter))
    try:
        await until(entered.is_set)
        assert emitter.resource_status()['pending'] == 1
        assert emitter.resource_status()['native_operations'] == 0
        with pytest.raises(WorkCloseTimeout):
            await emitter.aclose()
        assert not emitter.begin_close().done()
    finally:
        release.set()
        await posting
        await emitter.begin_close()
    assert emitter.resource_status()['pending'] == 0
    assert PRIVATE not in emitter.path.read_text()


async def test_no_loop_and_closed_emitters_drop_without_file_or_socket_io(tmp_path):
    emitter = await asyncio.to_thread(MetricsEmitter, str(tmp_path/'new'/'metrics.jsonl'))
    request(emitter)
    assert emitter.resource_status()['dropped'] == 1
    assert not emitter.path.parent.exists()
    await emitter.aclose()
    request(emitter)
    assert emitter.resource_status()['dropped'] == 2
    assert not emitter.path.parent.exists()


async def test_lowest_writer_boundary_filters_private_fields_and_opaque_errors(tmp_path, caplog):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'))
    try:
        await asyncio.to_thread(emitter._write, {'event_type': 'error',
            'module': PRIVATE, 'error_msg': PRIVATE, 'context': {'private': PRIVATE},
            'service': PRIVATE, '@timestamp': PRIVATE, PRIVATE: PRIVATE})
        assert PRIVATE not in emitter.path.read_text()
        assert PRIVATE not in caplog.text
    finally:
        await emitter.aclose()


async def test_real_router_and_native_wake_caller_emit_only_numeric_diagnostics(tmp_path):
    from src.orchestrator.dispatcher import PathType
    from src.wakeword.whisper_wake import WhisperWakeDetector, TV_MODE_ENTRY_REJECTS
    from tests.integration.test_prompt_provenance import request_router
    from tests.mocks.responses import response_handler_mock

    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'))
    router = request_router(None, response_handler_mock())
    router.metrics_emitter = emitter
    observations, messages = [], []
    router.latency_monitor = SimpleNamespace(record=lambda **kwargs: observations.append(kwargs))

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(StructuredFormatter(LogConfig(format=LogFormat.JSON)).format(record))

    handler = Capture()
    targets = [logging.getLogger('src.pipeline.request_router'), logging.getLogger('src.wakeword.whisper_wake')]
    previous_levels = [target.level for target in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(logging.DEBUG)
    detector = WhisperWakeDetector(SimpleNamespace(), wake_words=['synthetic'], metrics_emitter=emitter)
    try:
        with LogContext(private=PRIVATE):
            for path in PathType:
                router._log_latency({'latency_ms': 20., 'timings': {'stt': 12., PRIVATE: PRIVATE},
                    'text': PRIVATE, 'user': {'name': PRIVATE}, 'room': {'id': PRIVATE},
                    'intent': 'domotics', 'path': path.value, 'success': True,
                    'action': {'entity_id': PRIVATE, 'service': PRIVATE}})
                await emitter.flush()
            for _ in range(TV_MODE_ENTRY_REJECTS):
                detector._record_reject(PRIVATE)
            await asyncio.to_thread(detector._emit_wake, True, PRIVATE, 'exact', PRIVATE, 80., 20.)
        await emitter.flush()
        documents = [json.loads(line) for line in emitter.path.read_text().splitlines()]
        assert {doc['path'] for doc in documents if doc['event_type'] == 'request'} == {path.value for path in PathType}
        assert PRIVATE not in emitter.path.read_text()
        assert observations and all(row['user'] is None and row['intent'] == 'domotics' for row in observations)
        assert PRIVATE not in repr(observations)
        assert messages and PRIVATE not in '\n'.join(messages)
    finally:
        for target, previous in zip(targets, previous_levels):
            target.removeHandler(handler)
            target.setLevel(previous)
        await emitter.aclose()


@pytest.mark.parametrize('backend', ['parakeet', 'whisper', 'moonshine'])
async def test_stt_backend_logs_keep_timing_without_transcription(backend):
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT, MoonshineSTT

    if backend == 'parakeet':
        stt = ParakeetSTT()
        stt._model = SimpleNamespace(recognize=lambda *args, **kwargs: PRIVATE)
    elif backend == 'whisper':
        stt = FastWhisperSTT(model='synthetic', device='cpu')
        stt._model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([
            SimpleNamespace(text=PRIVATE, no_speech_prob=.1, avg_logprob=-.1, compression_ratio=1.)]), None))
    else:
        stt = MoonshineSTT(device='cpu')
        stt._model = SimpleNamespace(transcribe=lambda *args: PRIVATE)

    def transcribe():
        audio = np.zeros(16000, dtype=np.float32)
        if backend == 'moonshine':
            return stt.transcribe(audio)[0]
        return stt.transcribe_with_confidence(audio).text

    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(StructuredFormatter(LogConfig(format=LogFormat.JSON)).format(record))

    target = logging.getLogger(type(stt).__module__)
    previous = target.level
    handler = Capture()
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        with LogContext(private=PRIVATE):
            assert await asyncio.to_thread(transcribe) == PRIVATE
        assert messages and PRIVATE not in '\n'.join(messages)
        assert any('completed' in message and 'ms' in message for message in messages)
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)
        executor = getattr(stt, '_executor', None)
        if executor is not None:
            await asyncio.to_thread(executor.shutdown)
