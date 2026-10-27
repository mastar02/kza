"""Bounded numeric diagnostics isolated from private voice payloads and contexts."""
from __future__ import annotations

import asyncio
from collections import deque
from contextvars import Context
from enum import StrEnum
import logging
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any

from src.core.owned_work import ClosePhase, OwnedBlockingWork, WorkCloseTimeout
from src.obs.diagnostic_store import DiagnosticRecord, DiagnosticStore
from src.memory.producer_context import PRODUCER_CONTEXT

logger = logging.getLogger(__name__)
_ROTATE_BYTES_DEFAULT = 100 * 1024 * 1024
_TIMINGS = frozenset({'stt', 'speaker', 'speaker_id', 'emotion', 'llm_router',
    'home_assistant', 'tts', 'vector_search', 'router', 'routing', 'llm_stream',
    'routine_check', 'total', 'queue', 'response', 'capture', 'asr', 'inference'})
_NUMBERS = frozenset({*(name+'_ms' for name in _TIMINGS), 'audio_duration_ms',
    'wake_stt_ms', 'fuzzy_ratio', 'no_speech_prob', 'avg_logprob'})
_FLAGS = frozenset({'success', 'used_wake_text', 'early_dispatch', 'matched', 'tv_mode'})
_CODES = {
    'path': frozenset({'fast', 'slow', 'music', 'routine', 'fast_domotics', 'fast_routine',
        'fast_router', 'fast_music', 'slow_music', 'slow_llm', 'sync', 'enrollment',
        'feedback', 'fast_list', 'fast_reminder', 'fast_weather'}),
    'intent': frozenset({'turn_on', 'turn_off', 'toggle', 'set_brightness', 'set_color',
        'set_temperature', 'set_hvac_mode', 'set_position', 'stop', 'scene', 'routine',
        'timer', 'reminder', 'conversation', 'question', 'cancelled', 'unknown', 'domotics',
        'feedback', 'suggestion_response', 'learning', 'enrollment', 'sync', 'intercom',
        'notification_config', 'clarification_requested', 'timeout', 'pending_confirmation',
        'confirmation', 'rejection', 'cancel', 'invalid_target', 'domain_conflict',
        'speaker_auth', 'simple_query', 'queued', 'error', 'music_error', 'weather',
        'unresolved_target', 'gate_rejected', 'llm_rejected'}),
    'matched_via': frozenset({'exact', 'fuzzy', 'alias', 'rejected'}),
    'rejection_reason': frozenset({'no_speech_hallucination', 'high_no_speech_prob',
        'low_avg_logprob', 'implausible_speech_rate', 'multi_wake_hallucination',
        'pathological_repeat', 'tv_stop_phrase', 'no_command_verb',
        'below_fuzzy_threshold', 'no_keyword', 'tv_mode_phantom'}),
    'module': frozenset({'stt', 'wakeword', 'tts', 'router', 'ambient', 'audio',
        'pipeline', 'memory', 'home_assistant'}),
}


class DiagnosticKind(StrEnum):
    REQUEST = 'request'
    WAKE = 'wake'
    ERROR = 'error'


class DiagnosticFailure(StrEnum):
    FAILED = 'diagnostic_retention_failed'
    BUSY = 'diagnostic_writer_busy'
    UNSUPPORTED = 'unsupported_diagnostic_catalog'


def _failure(error):
    # Only fixed errors defined by this module cross the native boundary.
    if type(error) is RuntimeError and len(error.args) == 1 and type(error.args[0]) is str:
        for code in (DiagnosticFailure.BUSY, DiagnosticFailure.UNSUPPORTED):
            if error.args == (code.value,):
                return code
    return DiagnosticFailure.FAILED


def diagnostic_timings(timings: object) -> dict[str, float]:
    """Select bounded finite stages before logging or emitting a timing map."""
    if type(timings) is not dict:
        return {}
    result = {}
    for name in _TIMINGS:
        value = timings.get(name, timings.get(name+'_ms'))
        if type(value) in (int, float) and -1e12 <= value <= 1e12 and math.isfinite(value):
            result[name] = value
    return result


def project_diagnostic(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Copy known scalar fields without stringifying or coercing opaque values."""
    if type(doc) is not dict or type(doc.get('event_type')) is not str:
        return None
    if doc['event_type'] not in DiagnosticKind._value2member_map_:
        return None
    result = {'event_type': doc['event_type'], 'diagnostic_schema': 1}
    for name in _FLAGS:
        value = doc.get(name)
        if type(value) is bool:
            result[name] = value
    for name in _NUMBERS:
        value = doc.get(name)
        if type(value) in (int, float) and -1e12 <= value <= 1e12 and math.isfinite(value):
            result[name] = value
    for name, values in _CODES.items():
        value = doc.get(name)
        if name == 'intent' and type(value) is str:
            for prefix in ('gate_rejected', 'llm_rejected'):
                if value.startswith(prefix+':'):
                    value = prefix
        if name == 'rejection_reason' and type(value) is str and value.startswith('tv_mode_phantom:'):
            value = 'tv_mode_phantom'
        if type(value) is str and value in values:
            result[name] = value
    return result


class MetricsEmitter:
    """Post content-free diagnostics without blocking capture or retaining context.

    Construction performs no I/O. Native callers post sanitized scalar documents
    to the loop on which this owner was constructed. Saturation or a missing loop
    drops a metric. Legacy files stay untouched; this schema uses a new path.
    """

    def __init__(self, path: str = '/home/kza/logs/kza-metrics.jsonl',
                 rotate_bytes: int = _ROTATE_BYTES_DEFAULT, service_name: str = 'kza-voice',
                 logstash_host: str | None = None, logstash_port: int = 5515,
                 *, capacity: int = 8, close_timeout_s: float = 1,
                 retention_days: float = 90, clock=None, policy=None):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError('invalid_metric_capacity')
        self.legacy_path = Path(path)
        self.path = (self.legacy_path if self.legacy_path.stem.endswith('_policy_v1') else
                     self.legacy_path.with_name(self.legacy_path.stem+'_policy_v1'+self.legacy_path.suffix))
        self.rotate_bytes = rotate_bytes
        self.service_name = 'kza-voice'
        self.logstash_host, self.logstash_port = logstash_host, logstash_port
        self._lock, self._sock_lock = threading.Lock(), threading.Lock()
        self._sock: socket.socket | None = None
        self._capacity, self._close_timeout_s = capacity, close_timeout_s
        self._work = OwnedBlockingWork(capacity=capacity, close_timeout_s=close_timeout_s)
        self._clock = clock or time.time
        self._store = DiagnosticStore(self.path, clock=self._clock,
            retention_days=retention_days, rotate_bytes=rotate_bytes, project=project_diagnostic, policy=policy)
        self._retention_work = OwnedBlockingWork(capacity=1, close_timeout_s=close_timeout_s)
        self._close_work = OwnedBlockingWork(capacity=1,close_timeout_s=close_timeout_s)
        self._close_phase = ClosePhase.IDLE
        self._maintenance = None
        self._sweep = None
        self._wake = asyncio.Event()
        self._free = deque(range(capacity))
        self._records: dict[int, DiagnosticRecord] = {}
        self._tasks: set[asyncio.Task] = set()
        self._closed = False
        self._closing = None
        self._dropped = self._failed = 0
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def _enqueue(self, doc, sources=None):
        safe = project_diagnostic(doc)
        doc = None
        if safe is None:
            self._dropped += 1
            return
        try:
            slot = self._free.popleft()
        except IndexError:
            self._dropped += 1
            return
        if self._closed or self._loop is None:
            self._dropped += 1
            self._free.append(slot)
            return
        try:
            sources = PRODUCER_CONTEXT.get().sources if sources is None else sources
            record = self._store.record(safe, sources)
            if record is None:
                self._dropped += 1
                self._free.append(slot)
                return
            self._records[slot] = record
            # A normal thread-safe post captures arbitrary ContextVars as well.
            # This channel deliberately inherits no caller payload or identity.
            self._loop.call_soon_threadsafe(self._start, slot, record, context=Context())
        except Exception:
            self._dropped += 1
            self._records.pop(slot, None)
            self._free.append(slot)

    def _start(self, slot, doc):
        if self._maintenance is None and not self._closed:
            self._maintenance = asyncio.create_task(self._maintain(), context=Context())
        operation = self._work.run(self._invoke_write, doc)
        try:
            task = asyncio.create_task(operation, context=Context())
        except Exception:
            operation.close()
            self._failed += 1
            self._records.pop(slot, None)
            self._free.append(slot)
            return
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            self._records.pop(slot, None)
            self._free.append(slot)
            if done.cancelled() or done.exception() is not None or done.result() is not None:
                self._failed += 1

        task.add_done_callback(finished, context=Context())

    def _invoke_write(self, record):
        try:
            self._write(record)
        except (Exception, asyncio.CancelledError) as error:
            return _failure(error)
        finally:
            record = None

    def _invoke_retention(self):
        try:
            self._store.prune()
        except (Exception, asyncio.CancelledError) as error:
            return _failure(error)

    def _get_sock(self) -> socket.socket | None:
        if not self.logstash_host:
            return None
        if self._sock is None:
            self._sock = socket.create_connection((self.logstash_host, self.logstash_port), timeout=1.5)
            self._sock.settimeout(1.)
        return self._sock

    def _send_tcp(self, line: str, expires_at: float, sources=()) -> None:
        with self._sock_lock:
            try:
                if expires_at <= self._clock() or not self._store.available(sources):
                    return
                connection = self._get_sock()
                if connection is not None and expires_at > self._clock() and self._store.available(sources):
                    connection.sendall((line+'\n').encode('utf-8'))
            except OSError:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None
                logger.warning('Diagnostic TCP delivery deferred')

    def _write(self, doc: dict[str, Any] | DiagnosticRecord) -> None:
        record = doc if isinstance(doc, DiagnosticRecord) else self._store.record(doc)
        doc = None
        if record is None:
            return
        published = self._store.append(record)
        if published is not None and self.logstash_host:
            self._send_tcp(published.line, published.expires_at, record.sources)

    def pending_conversation(self, conversation_id: str) -> int:
        """Include unknown reserved posts and matching actual write/TCP workers."""
        records = tuple(self._records.values())
        unknown = max(0, self._capacity-len(self._free)-len(records))
        return unknown + sum(any(source.conversation_id == conversation_id for source in record.sources)
            for record in records)

    def _invoke_erasure(self, conversation_id):
        try:
            return self._store.erase(conversation_id)
        except (Exception, asyncio.CancelledError) as error:
            return _failure(error)

    async def aerase(self, conversation_id: str) -> int:
        """Wait for actual projection repair; report posts and unverified rows."""
        await self.aprune()
        outcome = await self._retention_work.run(self._invoke_erasure, conversation_id)
        if isinstance(outcome, DiagnosticFailure):
            raise RuntimeError(outcome.value)
        return outcome + self.pending_conversation(conversation_id)

    async def aprune(self) -> None:
        """Reconcile durable deadlines and both local files without another event."""
        if self._sweep is None or self._sweep.done():
            self._sweep = asyncio.create_task(self._retention_work.run(self._invoke_retention), context=Context())
            self._sweep.add_done_callback(lambda done: None if done.cancelled() else done.exception(), context=Context())
        failure = await asyncio.shield(self._sweep)
        if failure is not None:
            raise RuntimeError(failure.value)

    async def start(self) -> None:
        """Repair local retention before starting its idle maintenance loop."""
        if self._closed:
            raise RuntimeError('diagnostics_closed')
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError('diagnostic_loop_mismatch')
        try:
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await self.aprune()
            except TimeoutError:
                raise WorkCloseTimeout('diagnostic_initialization_pending',phase=ClosePhase.RETENTION,
                    diagnostics=self.resource_status()) from None
        finally:
            if self._maintenance is None and not self._closed:
                self._maintenance = asyncio.create_task(self._maintain(), context=Context())

    async def _maintain(self):
        while not self._closed:
            self._wake.clear()
            try:
                await self.aprune()
            except Exception:
                logger.warning('Diagnostic retention deferred')
            try:
                async with asyncio.timeout(1):
                    await self._wake.wait()
            except TimeoutError:
                pass

    def emit_request(self, *, user_id: str, zone_id: str | None, text: str,
                     intent: str | None, path: str | None, success: bool,
                     timings: dict[str, float], audio_duration_ms: float | None = None,
                     entity_id: str | None = None, service: str | None = None,
                     used_wake_text: bool = False, early_dispatch: bool = False, sources=None) -> None:
        """Keep timings and closed diagnostic codes; discard private arguments."""
        del user_id, zone_id, text, entity_id, service
        doc = None
        try:
            doc = {'event_type': DiagnosticKind.REQUEST.value, 'intent': intent,
                'path': path, 'success': success, 'audio_duration_ms': audio_duration_ms,
                'used_wake_text': used_wake_text, 'early_dispatch': early_dispatch}
            doc.update({name+'_ms': value for name, value in diagnostic_timings(timings).items()})
            self._enqueue(doc, sources)
        finally:
            intent = path = timings = doc = sources = None
            success = audio_duration_ms = used_wake_text = early_dispatch = None

    def emit_wake(self, *, room_id: str, matched: bool, wake_word: str | None,
                  matched_via: str, text: str | None, audio_duration_ms: float,
                  wake_stt_ms: float, fuzzy_ratio: float | None = None,
                  rejection_reason: str | None = None, no_speech_prob: float | None = None,
                  avg_logprob: float | None = None, tv_mode: bool = False, sources=None) -> None:
        """Keep detection counters and scores without transcripts or identities."""
        del room_id, wake_word, text
        try:
            self._enqueue({'event_type': DiagnosticKind.WAKE.value, 'matched': matched,
                'matched_via': matched_via, 'audio_duration_ms': audio_duration_ms,
                'rejection_reason': rejection_reason,
                'wake_stt_ms': wake_stt_ms, 'fuzzy_ratio': fuzzy_ratio,
                'no_speech_prob': no_speech_prob, 'avg_logprob': avg_logprob, 'tv_mode': tv_mode}, sources)
        finally:
            matched_via = rejection_reason = sources = None
            matched = audio_duration_ms = wake_stt_ms = fuzzy_ratio = no_speech_prob = avg_logprob = tv_mode = None

    def emit_error(self, *, module: str, error_msg: str, context: dict[str, Any] | None = None, sources=None) -> None:
        """Emit an error count and a closed stage code without arbitrary errors."""
        del error_msg, context
        try:
            self._enqueue({'event_type': DiagnosticKind.ERROR.value, 'module': module}, sources)
        finally:
            module = sources = None

    async def flush(self) -> None:
        """Wait for admitted posts without sealing admission or cancelling I/O."""
        while len(self._free) != self._capacity:
            await asyncio.sleep(.005)
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def resource_status(self) -> dict:
        """Expose counts, never paths, documents, transcripts or identities."""
        return {'closed': self._closed, 'capacity': self._capacity,
            'pending': self._capacity-len(self._free), 'dropped': self._dropped,
            'failed': self._failed, 'native_operations': self._work.status()['operations'],
            'retention': self._store.status(),
            'close_phase':self._close_phase.value,'io':self._work.diagnostics(),
            'retention_io':self._retention_work.diagnostics(),'close_io':self._close_work.diagnostics(),
            'closing': self._closing is not None and not self._closing.done()}

    def _close_socket(self):
        with self._sock_lock:
            connection, self._sock = self._sock, None
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    logger.warning('Diagnostic socket close failed')

    def begin_close(self) -> asyncio.Task:
        """Seal posts, drain actual I/O and close the socket despite observer exit."""
        self._closed = True
        self._wake.set()
        if self._closing is None:
            async def finish():
                self._close_phase = ClosePhase.DRAIN
                await self.flush()
                self._close_phase = ClosePhase.MAINTENANCE
                if self._maintenance is not None:
                    await self._maintenance
                self._close_phase = ClosePhase.WRITES
                await self._work.begin_close()
                self._close_phase = ClosePhase.RETENTION
                await self._retention_work.begin_close()
                self._close_phase = ClosePhase.SOCKET
                await self._close_work.run(self._close_socket)
                self._close_phase = ClosePhase.STORE
                await self._close_work.run(self._store.close)
                await self._close_work.begin_close()
                self._close_phase = ClosePhase.COMPLETE
            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while the retained close task drains actual I/O."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
            except TimeoutError:
                raise WorkCloseTimeout('diagnostic_io_still_running',phase=self._close_phase,
                    diagnostics=self.resource_status()) from None
        closing.result()
        if cancelled:
            raise asyncio.CancelledError
