"""Single-input capture with the same native ownership and provenance as multiroom."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
import time

import numpy as np
import sounddevice as sd

from src.audio import EchoSuppressor
from src.audio.owned_input import OwnedInputStream
from src.ambient.audio_event_detector import AudioEventDetector
from src.conversation import FollowUpMode
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryOrigin, MemorySource
from src.memory.producer_context import producer_context
from src.memory.producer_scopes import ProducerScopes, current_producer_scope
from src.memory.prompt_context import SourcesInvalidated, merge_sources, run_private
from src.pipeline.audio_manager import AudioManager
from src.pipeline.capture_room import CaptureRoom
from src.pipeline.command_event import CommandEvent

logger = logging.getLogger(__name__)
_CALLBACK_OWNER = ContextVar('single_input_callback_owner', default=None)


@dataclass
class CaptureState:
    """Native-worker-only command state, independent of response latency."""

    listening: bool = False
    started_at: float = 0.
    audio: list = field(default_factory=list)


@dataclass(frozen=True)
class AmbientBatch:
    """One analysis window with every original input it includes."""

    audio: np.ndarray
    sources: tuple[MemorySource, ...]


class AudioLoop:
    """Own default-device capture, wake, endpointing and optional ambient events."""

    def __init__(self, audio_manager: AudioManager, echo_suppressor: EchoSuppressor,
                 follow_up: FollowUpMode, sample_rate: int = 16000,
                 ambient_detector=None, *, privacy=None, clock=None,
                 close_timeout_s: float = 1):
        if isinstance(audio_manager, AudioManager) and audio_manager.sample_rate != sample_rate:
            raise ValueError('capture_sample_rate_mismatch')
        if isinstance(ambient_detector, AudioEventDetector) and ambient_detector.sample_rate != sample_rate:
            raise ValueError('ambient_sample_rate_mismatch')
        self.audio_manager, self.echo_suppressor = audio_manager, echo_suppressor
        self.follow_up, self.sample_rate = follow_up, sample_rate
        self.ambient_detector = ambient_detector
        self._privacy, self._clock = privacy, clock or time.time
        self._close_timeout_s = close_timeout_s
        self._state = CaptureState()
        self._ambient_frames = deque()
        self._ambient_samples = 0
        self._ambient_end = None
        self._ambient_ready = deque(maxlen=8)
        self._running = self._closed = False
        self._closing = self._controller = self._holder = None
        self._starting = False
        self._started = False
        self._dispatch_drops = self._analysis_drops = 0
        self._operations: set[asyncio.Task] = set()
        self._dispatches: set[asyncio.Task] = set()
        self._analyses: set[asyncio.Task] = set()
        self._batches: set[asyncio.Task] = set()
        self._scopes = ProducerScopes(self.snapshot_operations, capacity=32)
        self._on_command_callback: Callable[[CommandEvent], Awaitable[dict]] | None = None
        self._on_post_command_callback: Callable[[dict, np.ndarray], Awaitable[None]] | None = None
        self._room = CaptureRoom('', process=self._process_audio, poll=self._poll,
            clear=self._clear_capture, listening=lambda: self._state.listening, privacy=privacy)
        self._input = OwnedInputStream(self._make_stream, self._retire_input)
        if isinstance(ambient_detector, AudioEventDetector) and privacy is not None:
            ambient_detector.attach_privacy(privacy)
        if privacy is not None:
            privacy.add_borrower(self.begin_close)

    def on_command(self, callback: Callable[[CommandEvent], Awaitable[dict]]) -> None:
        """Register the sourced command callback."""
        self._on_command_callback = callback

    def on_post_command(self, callback: Callable[[dict, np.ndarray], Awaitable[None]]) -> None:
        """Register post-processing; capture lineage is inherited by its actual task."""
        self._on_post_command_callback = callback

    def _track(self, task, tasks):
        tasks.add(task)
        def finished(done):
            tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    def _live(self, sources) -> bool:
        return not self._closed and (self._privacy is None or self._privacy.sources_available(sources))

    async def start(self) -> None:
        """Load wake off loop while retaining actual initialization after cancel."""
        if self._closed or self._starting or self._started:
            raise RuntimeError('audio_loop_start_unavailable')
        self._starting = True
        try:
            await self._room.call(self._load_wake)
            if self._closed:
                raise RuntimeError('audio_loop_closed')
            if self.ambient_detector is not None:
                initializing = self._track(asyncio.create_task(self.ambient_detector.initialize()), self._operations)
                await initializing
                await self.ambient_detector.start()
            if self._closed:
                raise RuntimeError('audio_loop_closed')
            self._started = True
        finally:
            self._starting = False

    def _load_wake(self):
        if not self._closed:
            self.audio_manager.load_wake_word()

    def _make_stream(self):
        return sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                              blocksize=1280, callback=self._make_audio_callback())

    def _make_audio_callback(self):
        holder = object()
        self._holder = holder
        def callback(indata, frames, time_info, status):
            ticket = self._room.callbacks.enter(self._room.input_boundary)
            try:
                if ticket is None or not self._room.callbacks.accepting or self._closed or self._holder is not holder:
                    return
                timestamp = self._clock() - frames / self.sample_rate
                self._room.callbacks.stamp(ticket, timestamp, self._room.source)
                if not self._closed and self._holder is holder:
                    self._room.push(timestamp, indata, frames, boundary=ticket.boundary)
            except Exception:
                self._room.callbacks.failed()
            finally:
                indata = None
                if ticket is not None:
                    self._room.callbacks.leave(ticket)
                ticket = None
        return callback

    def _retire_input(self):
        self._room.retire_stream(self._clock())
        self._holder = None

    def _ambient_sources(self):
        return merge_sources(*(sources for sources, _ in tuple(self._ambient_frames)),
                             *(batch.sources for batch in tuple(self._ambient_ready)))

    def _prune_ambient(self):
        if self._closed:
            self._ambient_frames.clear()
            self._ambient_ready.clear()
            self._ambient_end = None
        else:
            self._ambient_frames = deque((sources, audio) for sources, audio in self._ambient_frames if self._live(sources))
            self._ambient_ready = deque((batch for batch in self._ambient_ready if self._live(batch.sources)), maxlen=8)
        self._ambient_samples = sum(len(audio) for _, audio in self._ambient_frames)

    def _append_ambient(self, audio, timestamp):
        self._prune_ambient()
        if type(timestamp) in (int, float):
            if self._ambient_end is not None and timestamp - self._ambient_end > .5:
                self._ambient_frames.clear()
                self._ambient_samples = 0
            self._ambient_end = timestamp + len(audio) / self.sample_rate
        source = self._room.source
        sources = (source,) if source is not None else ()
        if not self._live(sources):
            return
        self._ambient_frames.append((sources, audio.copy()))
        self._ambient_samples += len(audio)
        while self._ambient_samples >= self.sample_rate:
            parts, groups = [], []
            remaining = self.sample_rate
            while remaining:
                lineage, frame = self._ambient_frames.popleft()
                count = min(remaining, len(frame))
                parts.append(frame[:count])
                groups.append(lineage)
                if count < len(frame):
                    self._ambient_frames.appendleft((lineage, frame[count:]))
                remaining -= count
            self._ambient_samples -= self.sample_rate
            self._ambient_ready.append(AmbientBatch(np.concatenate(parts), merge_sources(*groups)))

    def _process_audio(self, indata, frames, time_info, status):
        audio = None
        try:
            self._room.require_live()
            audio = indata[:, 0].copy()
            if not self.echo_suppressor.is_safe_to_listen:
                return
            allowed, reason = self.echo_suppressor.should_process_audio(audio)
            self._room.require_live()
            if not allowed and reason != 'silence':
                return
            if self.ambient_detector is not None:
                self._append_ambient(audio, time_info)
            if self._state.listening:
                self._state.audio.extend(audio)
                return
            detection = self.audio_manager.detect_wake_word(audio)
            self._room.require_live()
            if detection or self.follow_up.is_active and np.sqrt(np.mean(audio ** 2)) > .02 and self.echo_suppressor.is_human_voice(audio):
                self._room.require_live()
                if detection:
                    self.follow_up.start_conversation()
                self._state.started_at = (time_info + frames / self.sample_rate
                    if type(time_info) in (int, float) else self._clock())
                self._state.audio = []
                self._state.listening = True
        finally:
            indata = audio = None

    def _poll(self):
        self._prune_ambient()
        if not self._state.listening:
            return None
        self._room.require_live()
        complete, _, audio, early = self.audio_manager.capture_command_with_vad(
            self._state.audio, self._state.started_at, silence_threshold=.015,
            silence_duration_ms=300, min_speech_ms=300)
        self._room.require_live()
        if complete and audio is not None:
            return CommandEvent(audio=audio, room_id='', capture_complete=early,
                                captured_at=self._state.started_at)
        return None

    def _clear_capture(self):
        self._state = CaptureState()
        self._prune_ambient()
        if self._closed:
            self.audio_manager.close_wake_word()
        else:
            self.audio_manager.reset_wake_word()

    def _take_ambient(self):
        self._prune_ambient()
        result = list(self._ambient_ready)
        self._ambient_ready.clear()
        return result

    async def _invoke_callbacks(self, event, sources):
        result = None
        try:
            if not self._dispatch_live(sources):
                raise SourcesInvalidated('audio_sources_invalidated')
            if self._on_command_callback is not None:
                result = await self._on_command_callback(event)
            else:
                result = {}
            if self._on_post_command_callback is not None:
                if not self._dispatch_live(sources):
                    raise SourcesInvalidated('audio_sources_invalidated')
                await self._on_post_command_callback(result, event.audio)
        finally:
            event = result = None

    async def _dispatch(self, event, sources):
        token = _CALLBACK_OWNER.set(asyncio.current_task())
        try:
            await run_private(self._invoke_callbacks(event, sources), lambda: self._dispatch_live(sources))
        except SourcesInvalidated:
            pass
        except Exception as error:
            logger.error('Audio command callback failed: %s', type(error).__name__)
        finally:
            _CALLBACK_OWNER.reset(token)
            event = None

    def _dispatch_live(self, sources):
        receipt = current_producer_scope()
        selected = receipt.snapshot()[0] if receipt is not None else sources
        return self._live(selected)

    async def _analyze(self, batch):
        token = _CALLBACK_OWNER.set(asyncio.current_task())
        sources = batch.sources
        try:
            operation = self.ambient_detector.analyze_chunk(batch.audio)
            batch = None
            await run_private(operation, lambda: self._live(sources))
        except SourcesInvalidated:
            pass
        except Exception as error:
            logger.warning('Ambient audio callback failed: %s', type(error).__name__)
        finally:
            _CALLBACK_OWNER.reset(token)
            batch = None

    async def _consume(self):
        batch = None
        try:
            while self._running and not self._closed:
                batch = self._track(asyncio.create_task(self._consume_once()), self._batches)
                try:
                    await batch
                except SourcesInvalidated:
                    status = self._room.resource_status()
                    if status['closed'] or status['reset_failed']:
                        raise
                finally:
                    batch = None
                await asyncio.sleep(.05)
        except SourcesInvalidated:
            pass
        except Exception as error:
            logger.error('Audio capture controller stopped: %s', type(error).__name__)
            self.begin_close()
        finally:
            batch = None

    async def _consume_once(self):
        event = step = batches = batch = None
        with self._scopes.operation(()) as handoff:
            try:
                step = self._track(asyncio.create_task(self._room.step(handoff=handoff)), self._operations)
                event = await step
                if event is not None and len(self._dispatches) < 8:
                    sources = ()
                    if self._privacy is not None:
                        sources = (MemorySource(event.capture_id, event.conversation_id,
                            event.captured_at, event.source_expires_at, MemoryOrigin.VOICE),)
                    with self._scopes.operation(sources), producer_context('', sources):
                        self._track(asyncio.create_task(self._dispatch(event, sources)), self._dispatches)
                elif event is not None:
                    self._dispatch_drops += 1
                event = step = None
                if self.ambient_detector is not None:
                    sources = self._ambient_sources()
                    self._scopes.replace_sources(sources)
                    batches = await self._room.call(self._take_ambient, sources=sources)
                    for batch in batches:
                        if len(self._analyses) < 2 and self._live(batch.sources):
                            with self._scopes.operation(batch.sources), producer_context('', batch.sources):
                                self._track(asyncio.create_task(self._analyze(batch)), self._analyses)
                        else:
                            self._analysis_drops += 1
                    batches = batch = None
            finally:
                event = step = batches = batch = None

    async def run(self) -> None:
        """Capture independently of response latency, retaining native resources."""
        if self._closed or self._running:
            raise RuntimeError('audio_loop_run_unavailable')
        self._running = True
        self._controller = asyncio.create_task(self._consume())
        try:
            await self._input.open()
            while self._running:
                await asyncio.sleep(.05)
        finally:
            self.begin_close()

    def begin_close(self) -> asyncio.Task:
        """Reject input and retain all actual driver, model and callback cleanup."""
        self._closed = True
        self._running = False
        if self._closing is None:
            exempt = _CALLBACK_OWNER.get()
            controllers = self._operations | self._dispatches | self._analyses | self._batches
            if self._controller is not None:
                controllers.add(self._controller)
            for task in controllers:
                if task is not exempt:
                    task.cancel()
            drains = [self._room.begin_close(), self._input.begin_close()]
            if isinstance(self.ambient_detector, AudioEventDetector):
                drains.append(self.ambient_detector.begin_close())
            async def finish():
                await asyncio.gather(*controllers, *drains, return_exceptions=True)
                if self.ambient_detector is not None:
                    await self.ambient_detector.stop()
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def stop(self) -> None:
        """Bound close observation; a callback may signal its own retained close."""
        closing = self.begin_close()
        if _CALLBACK_OWNER.get() in self._dispatches | self._analyses:
            return
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(closing)
        except TimeoutError:
            raise WorkCloseTimeout('single_input_still_running') from None

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose finite capture, analysis, callbacks and actual driver operations."""
        tasks = self._operations | self._dispatches | self._analyses | self._batches | set(self._room.snapshot_operations()) | set(self._input.snapshot_operations())
        if isinstance(self.ambient_detector, AudioEventDetector):
            tasks |= set(self.ambient_detector.snapshot_operations())
        return tuple(tasks)

    def conversation_busy(self, conversation_id: str) -> bool:
        """Include finite work, actual input callbacks and local idle buffers."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        if self._scopes.busy(conversation_id, self._privacy.ledger) or self._room.conversation_busy(conversation_id):
            return True
        sources = self._ambient_sources()
        return bool((self._ambient_frames or self._ambient_ready) and not sources) or any(
            source.conversation_id == conversation_id
            or self._privacy.ledger.dependencies_committed_unavailable(source.fact_dependencies) for source in sources)

    async def prune_private(self) -> None:
        """Sweep invalid capture and ambient buffers without waiting for input."""
        await self._room.prune_private()
        if not self._closed:
            await self._room.call(self._prune_ambient, sources=())

    def resource_status(self) -> dict:
        """Report bounded ownership without capture identifiers or audio."""
        return {'closed': self._closed, 'operations': len(self.snapshot_operations()),
                'batches': len(self._batches), 'source_scopes': self._scopes.resource_status(),
                'capture': self._room.resource_status(), 'input': self._input.resource_status(),
                'ambient_samples': self._ambient_samples, 'ambient_ready': len(self._ambient_ready),
                'dispatch_drops': self._dispatch_drops, 'analysis_drops': self._analysis_drops}

    @staticmethod
    def _handle_task_error(task: asyncio.Task) -> None:
        """Report only failure type; never retain a producer's private traceback."""
        if not task.cancelled() and task.exception():
            logger.error('Background audio task failed: %s', type(task.exception()).__name__)
