"""One response owner for generation, bounded sentence delivery and audio."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from src.llm.generation_stream import GenerationStream
from src.core.voice_delivery import (
    DeliveryState, GenerationState, ResponseBudgets, ResponseResult, ResponseStage, ResponseTimedOut,
)
from src.tts.piper_tts import AudioPlaybackError, StreamingAudioPlayer
from src.tts.response_cache import CachedAudio, ResponseCache

logger = logging.getLogger(__name__)


@dataclass
class _Utterance:
    text: str = ""
    pcm: CachedAudio | None = None
    stream: bool = True
    emotion: dict | None = None


@dataclass
class _Session:
    zone_id: str | None
    loop: asyncio.AbstractEventLoop | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    space: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: asyncio.Event = field(default_factory=asyncio.Event)
    queue: deque = field(default_factory=deque)
    done: bool = False
    player: StreamingAudioPlayer | None = None
    delivery: DeliveryState = DeliveryState.COMPLETE
    error: str | None = None
    spoken: int = 0
    peak_queue: int = 0
    filler: str | None = None
    broadcast: bool = False
    delivery_started_at: float | None = None

    def signal(self, event: asyncio.Event) -> None:
        if self.loop is not None:
            try:
                self.loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass


class _DeliveryBlocked(Exception):
    pass


class ResponseCloseTimeout(RuntimeError):
    """Owned voice resources did not exit before the shutdown deadline."""


class ResponseHandler:
    """Serialize the TTS engine and route every PCM source through owned output.

    Async callers use ``aspeak``, ``aplay_earcon`` and ``arespond``. Blocking
    adapters remain for synchronous integrations; none calls sd.play or an
    engine's independent speak method. A cancelled blocking synthesis retains
    the owner slot until it really exits, and cannot start late audio.
    """

    def __init__(
        self, tts, zone_manager=None, llm=None, streaming_enabled: bool = True,
        streaming_buffer_ms: int = 150, streaming_prebuffer_ms: int = 30,
        llm_buffer_preset: str = "balanced", llm_use_filler: bool = True,
        llm_filler_phrases: list | None = None, response_cache: ResponseCache | None = None,
        hooks=None, before_handler_warn_ms: float = 5.0, *,
        sentence_queue_capacity: int = 2, sentence_max_chars: int = 400,
        max_pending_responses: int = 16, playback_stall_timeout_s: float = 1.0,
        playback_drain_timeout_s: float | None = None,
        shutdown_timeout_s: float = 1.0,
        response_budgets: ResponseBudgets | None = None,
    ) -> None:
        if any(type(n) is not int or n <= 0 for n in
               (sentence_queue_capacity, sentence_max_chars, max_pending_responses)):
            raise ValueError("Invalid response queue limits")
        self.tts, self.zone_manager, self.llm = tts, zone_manager, llm
        self.streaming_enabled = streaming_enabled
        self.streaming_buffer_ms, self.streaming_prebuffer_ms = streaming_buffer_ms, streaming_prebuffer_ms
        self.llm_buffer_preset, self.llm_use_filler = llm_buffer_preset, llm_use_filler
        self.llm_filler_phrases = llm_filler_phrases or ["Un momento..."]
        self._response_cache, self.hooks = response_cache, hooks
        self._before_handler_warn_ms = before_handler_warn_ms
        self.sentence_queue_capacity, self.sentence_max_chars = sentence_queue_capacity, sentence_max_chars
        self.max_pending_responses = max_pending_responses
        self.playback_stall_timeout_s, self.playback_drain_timeout_s = playback_stall_timeout_s, playback_drain_timeout_s
        self._ownership = threading.Condition()
        if type(shutdown_timeout_s) not in (int, float) or not math.isfinite(shutdown_timeout_s) or shutdown_timeout_s <= 0:
            raise ValueError("Invalid voice shutdown deadline")
        self.shutdown_timeout_s = shutdown_timeout_s
        self.budgets = response_budgets or ResponseBudgets()
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None
        self._controllers: dict[int, _Session] = {}
        self._workers: dict[int, _Session] = {}
        self._generation_tasks: set[asyncio.Task] = set()
        self._retained_generations: set[asyncio.Task] = set()
        self._abort_tasks: set[asyncio.Task] = set()
        self._owner: _Session | None = None
        self._pending: list[_Session] = []
        self._active_zone_id = None
        self._earcon_audio, self._earcon_sr = None, 24000
        self._generation = GenerationStream(llm)
        self._generation_sources: set[GenerationStream] = {self._generation}

    @property
    def is_speaking(self) -> bool:
        """Report active PCM output, allowing capture during generation/synthesis."""
        with self._ownership:
            session = self._owner
            player = session.player if session is not None else None
        if player is None:
            return False
        return player.is_playing

    @property
    def is_responding(self) -> bool:
        """Whether a response owns generation, synthesis or playback resources."""
        with self._ownership:
            return self._owner is not None

    def set_active_zone(self, zone_id: str) -> None:
        """Set the default only for subsequent sessions."""
        self._active_zone_id = zone_id
        if self.zone_manager is not None:
            self.zone_manager.set_active_zone(zone_id)

    def set_earcon(self, audio, sr: int) -> None:
        """Inject a preloaded PCM earcon."""
        self._earcon_audio, self._earcon_sr = audio, sr

    def _session(self, zone_id, room_context, *, asynchronous: bool) -> _Session:
        if zone_id is None and room_context is not None:
            room = getattr(room_context, "room_id", None)
            zone_id = f"zone_{room}" if room else None
        session = _Session(zone_id if zone_id is not None else self._active_zone_id,
                           asyncio.get_running_loop() if asynchronous else None)
        with self._ownership:
            if self._closed:
                raise RuntimeError("voice_delivery_closed")
            if len(self._pending) >= self.max_pending_responses:
                raise RuntimeError("response_queue_full")
            self._pending.append(session)
            self._controllers[id(session)] = session
        return session

    def _claim(self, session: _Session) -> bool:
        with self._ownership:
            if session.cancelled.is_set():
                if session in self._pending:
                    self._pending.remove(session)
                return False
            if self._owner is None and self._pending and self._pending[0] is session:
                self._pending.pop(0)
                self._owner = session
                self._workers[id(session)] = session
                return True
            return False

    async def _acquire(self, session: _Session) -> bool:
        try:
            while not session.cancelled.is_set():
                if self._claim(session):
                    return True
                await asyncio.sleep(0.005)
            return False
        finally:
            with self._ownership:
                if session in self._pending:
                    self._pending.remove(session)

    def _interrupt(self, session: _Session) -> StreamingAudioPlayer | None:
        with session.condition:
            if session.cancelled.is_set():
                return None
            session.cancelled.set()
            session.queue.clear()
            session.condition.notify_all()
            session.signal(session.space)
            session.signal(session.interrupted)
            return session.player

    async def cancel(self, zone_id: str | None = None) -> bool:
        """Cancel current/pending matching sessions, including before first PCM.

        Cancellation tokens prevent device startup immediately. Device abort
        and close run off-loop; a blocked synthesis continues to own its slot.
        """
        with self._ownership:
            sessions = {**self._controllers, **self._workers}.values()
            targets = [s for s in sessions if not s.cancel_requested.is_set()
                       and (zone_id is None or s.zone_id == zone_id or s.broadcast)]
            for session in targets:
                session.cancel_requested.set()
        if targets:
            await asyncio.gather(*(self._stop_session(session) for session in targets))
        return bool(targets)

    def get_delivery_status(self) -> dict:
        """Report actual retained resources, including cancelled native workers."""
        with self._ownership:
            return {
                "closed": self._closed, "pending": len(self._pending),
                "controllers": len(self._controllers), "workers": len(self._workers),
                "generations": sum(not task.done() for task in self._generation_tasks),
                "blocked_generations": sum(source.sync_busy for source in self._generation_sources),
                "aborts": sum(not task.done() for task in self._abort_tasks),
            }

    def _controller_finished(self, session: _Session) -> None:
        with self._ownership:
            self._controllers.pop(id(session), None)

    async def _close_resources(self) -> None:
        try:
            async with asyncio.timeout(self.shutdown_timeout_s):
                await self.cancel()
                while True:
                    status = self.get_delivery_status()
                    if not any(status[key] for key in ("controllers", "workers", "generations", "blocked_generations", "aborts")):
                        return
                    await asyncio.sleep(0.005)
        except TimeoutError as error:
            raise ResponseCloseTimeout("voice_resources_still_running") from error

    async def aclose(self) -> None:
        """Close admission, cancel and drain owned resources within the deadline.

        Repeated caller cancellation cannot abandon cleanup. A blocked native
        synthesis remains visible and retains its slot after a close timeout;
        it can never start late PCM. A later close may await its real exit.
        """
        with self._ownership:
            self._closed = True
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self._close_resources())
        cancelled = False
        while not self._shutdown_task.done():
            try:
                await asyncio.shield(self._shutdown_task)
            except asyncio.CancelledError:
                cancelled = True
        self._shutdown_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _put(self, session: _Session, utterance: _Utterance) -> None:
        while True:
            session.space.clear()
            with session.condition:
                if session.cancelled.is_set():
                    raise asyncio.CancelledError
                if session.error is not None:
                    raise AudioPlaybackError(session.error)
                if len(session.queue) < self.sentence_queue_capacity:
                    session.queue.append(utterance)
                    session.peak_queue = max(session.peak_queue, len(session.queue))
                    session.condition.notify_all()
                    return
            await session.space.wait()

    def _finish(self, session: _Session) -> None:
        with session.condition:
            session.done = True
            session.condition.notify_all()

    def _render(self, session: _Session, utterance: _Utterance) -> None:
        started = time.perf_counter()
        text, success = utterance.text, False
        if utterance.pcm is None and self.hooks is not None:
            from src.hooks import BlockResult, TtsCall, execute_before_chain
            rewritten = execute_before_chain(
                self.hooks, "before_tts_speak", TtsCall(text=text, voice=None, lang="es", user_id=None, zone_id=session.zone_id),
                warn_ms=self._before_handler_warn_ms,
            )
            if isinstance(rewritten, BlockResult):
                raise _DeliveryBlocked(rewritten.reason)
            if rewritten.zone_id != session.zone_id:
                raise _DeliveryBlocked("tts_origin_zone_changed")
            text = rewritten.text
        try:
            if session.cancelled.is_set():
                return
            if session.zone_id is not None and self.zone_manager is None:
                raise AudioPlaybackError("output_zone_unavailable")
            cached = utterance.pcm
            if cached is None and self._response_cache is not None:
                cached = self._response_cache.get(text)
            if cached is not None:
                source, sr = iter([cached.audio]), cached.sample_rate
            else:
                if utterance.emotion:
                    self._apply_emotion_adjustment(utterance.emotion)
                sr = self.tts.sample_rate
                if utterance.stream and callable(getattr(self.tts, "synthesize_stream", None)):
                    source = iter(self.tts.synthesize_stream(text))
                else:
                    audio, _ = self.tts.synthesize(text)
                    source = iter([audio])
            frames = 0
            source_closed = False

            def close_source():
                nonlocal source_closed
                if not source_closed:
                    source_closed = True
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()

            def guarded_pcm():
                nonlocal frames
                try:
                    while not session.cancelled.is_set():
                        try:
                            chunk = next(source)
                        except StopIteration:
                            return
                        if session.cancelled.is_set():
                            return
                        if chunk is None:
                            raise AudioPlaybackError("tts_returned_no_audio")
                        frames += len(chunk)
                        yield chunk
                finally:
                    close_source()

            def register(player):
                with session.condition:
                    session.player = player
                if session.cancelled.is_set():
                    player.stop()

            pcm = guarded_pcm()
            try:
                if self.zone_manager is not None and session.broadcast:
                    completed = self.zone_manager.play_stream_to_all_zones(
                        pcm, sample_rate=sr, buffer_ms=self.streaming_buffer_ms,
                        prebuffer_ms=self.streaming_prebuffer_ms, on_player=register,
                        cancel_event=session.cancelled, stall_timeout_s=self.playback_stall_timeout_s,
                        drain_timeout_s=self.playback_drain_timeout_s,
                    )
                elif self.zone_manager is not None and session.zone_id is not None:
                    completed = self.zone_manager.play_stream_to_zone(
                        session.zone_id, pcm, sample_rate=sr,
                        buffer_ms=self.streaming_buffer_ms, prebuffer_ms=self.streaming_prebuffer_ms,
                        on_player=register, cancel_event=session.cancelled,
                        stall_timeout_s=self.playback_stall_timeout_s, drain_timeout_s=self.playback_drain_timeout_s,
                    )
                else:
                    player = StreamingAudioPlayer(
                        sample_rate=sr, buffer_ms=self.streaming_buffer_ms, prebuffer_ms=self.streaming_prebuffer_ms,
                        cancel_event=session.cancelled, stall_timeout_s=self.playback_stall_timeout_s,
                        drain_timeout_s=self.playback_drain_timeout_s,
                    )
                    register(player)
                    completed = player.play_stream(pcm)
            finally:
                try:
                    pcm.close()
                finally:
                    close_source()
                    with session.condition:
                        session.player = None
            if not session.cancelled.is_set():
                if not completed or frames == 0:
                    raise AudioPlaybackError("audio_not_delivered")
                success = True
                session.spoken += 1
        finally:
            if self.hooks is not None and utterance.pcm is None:
                from src.hooks import TtsPayload, execute_after_event
                payload = TtsPayload(
                    timestamp=time.time(), text=text, voice=None,
                    latency_ms=(time.perf_counter() - started) * 1000, success=success,
                )
                if session.loop is not None:
                    session.loop.call_soon_threadsafe(execute_after_event, self.hooks, "tts", payload)
                else:
                    execute_after_event(self.hooks, "tts", payload)

    def _worker(self, session: _Session) -> None:
        filler_deadline = time.monotonic() + 0.1
        try:
            while not session.cancelled.is_set():
                with session.condition:
                    while not session.queue and not session.done and not session.cancelled.is_set():
                        remaining = filler_deadline - time.monotonic()
                        if session.filler is not None and remaining <= 0:
                            break
                        session.condition.wait(max(0, remaining) if session.filler is not None else None)
                    if session.cancelled.is_set() or (not session.queue and session.done):
                        return
                    if session.queue:
                        utterance = session.queue.popleft()
                        session.filler = None
                    else:
                        utterance = _Utterance(session.filler, stream=self.streaming_enabled)
                        session.filler = None
                session.signal(session.space)
                if session.delivery_started_at is None:
                    session.delivery_started_at = time.monotonic()
                self._render(session, utterance)
        except _DeliveryBlocked as error:
            session.delivery, session.error = DeliveryState.BLOCKED, str(error)
            self._interrupt(session)
        except Exception as error:
            session.delivery, session.error = DeliveryState.FAILED, str(error)
            logger.warning("[Response] Audio delivery failed: %s", error)
            self._interrupt(session)
        finally:
            if session.cancelled.is_set() and session.delivery is DeliveryState.COMPLETE:
                session.delivery = DeliveryState.CANCELLED
            self._worker_finished(session)

    def _worker_finished(self, session: _Session) -> None:
        with self._ownership:
            if self._owner is session:
                self._owner = None
            self._workers.pop(id(session), None)
            self._ownership.notify_all()
        session.signal(session.space)

    async def asynthesize(self, text: str) -> object:
        """Own startup/cache synthesis without playing or publishing late PCM.

        The worker retains the same engine slot as spoken responses, including
        after caller cancellation. Its real exit is included in aclose.
        """
        session = self._session(None, None, asynchronous=True)

        def synthesize():
            try:
                if not session.cancelled.is_set():
                    return self.tts.synthesize(text)
                return None
            finally:
                self._worker_finished(session)

        try:
            if not await self._acquire(session):
                raise asyncio.CancelledError
            worker = asyncio.create_task(asyncio.to_thread(synthesize))
            worker.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            await self._wait_output(session, worker)
            if session.cancelled.is_set():
                raise asyncio.CancelledError
            return worker.result()
        except asyncio.CancelledError:
            self._interrupt(session)
            raise
        finally:
            self._controller_finished(session)

    @staticmethod
    def _result(session: _Session, text: str, generation=GenerationState.COMPLETE, error=None) -> ResponseResult:
        if session.cancelled.is_set() and session.delivery is DeliveryState.COMPLETE:
            delivery = DeliveryState.CANCELLED
        else:
            delivery = session.delivery
        if delivery is DeliveryState.CANCELLED:
            generation = GenerationState.CANCELLED
        return ResponseResult(text, generation, delivery, error or session.error, spoken_segments=session.spoken)

    async def _wait_output(self, session: _Session, worker: asyncio.Task) -> None:
        await self._wait_output_until(session, worker)

    async def _wait_output_until(self, session: _Session, worker: asyncio.Task,
                                 deadline: float | None = None) -> None:
        interrupted = asyncio.create_task(session.interrupted.wait())
        try:
            timeout = None if deadline is None else max(0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait({worker, interrupted}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise ResponseTimedOut(ResponseStage.DELIVERY)
            if worker.done() or session.delivery in {DeliveryState.FAILED, DeliveryState.BLOCKED}:
                await asyncio.shield(worker)
        finally:
            interrupted.cancel()
            await asyncio.gather(interrupted, return_exceptions=True)

    def _track_generation(self, task: asyncio.Task) -> asyncio.Task:
        with self._ownership:
            self._generation_tasks.add(task)

        def finished(completed):
            with self._ownership:
                self._generation_tasks.discard(completed)
                self._retained_generations.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        return task

    async def _settle_tasks(self, tasks: set[asyncio.Task], timeout: float | None = None) -> bool:
        if not tasks:
            return True
        done, pending = await asyncio.wait(tasks, timeout=self.budgets.cleanup_s if timeout is None else timeout)
        for task in done:
            if not task.cancelled():
                task.exception()
        return not pending

    async def _stop_session(self, session: _Session) -> None:
        player = self._interrupt(session)
        if player is not None:
            task = asyncio.create_task(asyncio.to_thread(player.stop))
            with self._ownership:
                self._abort_tasks.add(task)

            def finished(completed):
                with self._ownership:
                    self._abort_tasks.discard(completed)
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(finished)
            await self._settle_tasks({task})

    async def aspeak(self, text: str, zone_id: str | None = None, stream: bool | None = None,
                     emotion_adjustment: dict | None = None, room_context=None, *,
                     timeout_s: float | None = None) -> ResponseResult:
        """Await one owned utterance without blocking capture or abandoning TTS."""
        if timeout_s is not None and (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("Invalid voice utterance deadline")
        if not text:
            return ResponseResult("", GenerationState.COMPLETE, DeliveryState.EMPTY, speech_handled=False)
        return await self._aspeak_utterance(_Utterance(text=text, stream=self.streaming_enabled if stream is None else stream,
                                                     emotion=emotion_adjustment), zone_id, room_context, timeout_s=timeout_s)

    async def _aspeak_utterance(self, utterance, zone_id, room_context, *, broadcast=False, timeout_s=None) -> ResponseResult:
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        session = self._session(zone_id, room_context, asynchronous=True)
        session.broadcast = broadcast
        try:
            queue_limit = self.budgets.queue_wait_s if deadline is None else min(self.budgets.queue_wait_s, max(0, deadline-loop.time()))
            try:
                async with asyncio.timeout(queue_limit):
                    acquired = await self._acquire(session)
            except TimeoutError as error:
                raise ResponseTimedOut(ResponseStage.QUEUE) from error
            if not acquired:
                return ResponseResult(utterance.text, GenerationState.CANCELLED, DeliveryState.CANCELLED)
            session.queue.append(utterance)
            session.done = True
            worker = asyncio.create_task(asyncio.to_thread(self._worker, session))
            output_deadline = loop.time() + self.budgets.delivery_s
            if deadline is not None:
                output_deadline = min(deadline, output_deadline)
            await self._wait_output_until(session, worker, output_deadline)
            return self._result(session, utterance.text)
        except ResponseTimedOut as error:
            await self._stop_session(session)
            return ResponseResult(utterance.text, GenerationState.COMPLETE, DeliveryState.TIMED_OUT,
                                  str(error), timeout_stage=error.stage)
        except asyncio.CancelledError:
            await self._stop_session(session)
            raise
        finally:
            self._controller_finished(session)

    def speak(self, text: str, zone_id=None, stream=None, emotion_adjustment=None, room_context=None) -> ResponseResult:
        """Blocking adapter for synchronous callers, using the same owner/output."""
        if not text:
            return ResponseResult("", GenerationState.COMPLETE, DeliveryState.EMPTY, speech_handled=False)
        session = self._session(zone_id, room_context, asynchronous=False)
        try:
            with self._ownership:
                while not self._claim(session) and not session.cancelled.is_set():
                    self._ownership.wait(0.02)
            if session.cancelled.is_set():
                return ResponseResult(text, GenerationState.CANCELLED, DeliveryState.CANCELLED)
            session.queue.append(_Utterance(text=text, stream=self.streaming_enabled if stream is None else stream,
                                            emotion=emotion_adjustment))
            session.done = True
            self._worker(session)
            return self._result(session, text)
        finally:
            self._controller_finished(session)

    async def aplay_earcon(self, zone_id=None, room_context=None) -> ResponseResult:
        """Deliver the configured earcon through the same cancellation owner."""
        if self._earcon_audio is None:
            return ResponseResult("", GenerationState.COMPLETE, DeliveryState.EMPTY, speech_handled=False)
        pcm = CachedAudio(self._earcon_audio, self._earcon_sr, len(self._earcon_audio) / self._earcon_sr, "")
        return await self._aspeak_utterance(_Utterance(pcm=pcm), zone_id, room_context)

    def play_earcon(self, zone_id=None, room_context=None) -> ResponseResult:
        """Blocking earcon adapter for synchronous integrations."""
        return asyncio.run(self.aplay_earcon(zone_id, room_context))

    async def aspeak_to_all_zones(self, text: str, stream: bool | None = None) -> ResponseResult:
        """Broadcast through the shared owner and expose its actual delivery."""
        if not text:
            return ResponseResult("", GenerationState.COMPLETE, DeliveryState.EMPTY, speech_handled=False)
        return await self._aspeak_utterance(
            _Utterance(text, stream=self.streaming_enabled if stream is None else stream), None, None, broadcast=True,
        )

    def speak_to_all_zones(self, text: str, stream: bool | None = None) -> ResponseResult:
        """Blocking broadcast compatibility adapter with owned PCM output."""
        return asyncio.run(self.aspeak_to_all_zones(text, stream))

    def _split(self, text: str, final: bool = False) -> tuple[list[str], str]:
        phrases = []
        while text:
            boundary = re.search(r"[.!?;](?:\s|$)|\n" if final else r"[.!?;]\s|\n", text)
            if boundary is not None and boundary.end() <= self.sentence_max_chars:
                end = boundary.end()
            elif len(text) >= self.sentence_max_chars:
                end = text.rfind(" ", 0, self.sentence_max_chars + 1)
                end = end if end > 0 else self.sentence_max_chars
            elif final:
                end = len(text)
            else:
                break
            phrase, text = text[:end], text[end:]
            if phrase.strip():
                phrases.append(phrase.strip())
        return phrases, text

    async def arespond(self, prompt: str, zone_id=None, *, generation: GenerationStream | None = None,
                       max_tokens: int = 1024, temperature: float = 0.7, room_context=None,
                       emotion_adjustment=None, use_filler: bool | None = None,
                       deadline: float | None = None, on_owned: Callable[[], None] | None = None) -> ResponseResult:
        """Own work deadlines and at most one bounded error utterance.

        Queue, provider waiting and output have separate budgets. Downstream
        backpressure does not spend the provider budget. The optional absolute
        deadline caps work after an orchestrator queue; feedback has its own
        budget and is never announced again by the router.
        """
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("Invalid response deadline")
        session = self._session(zone_id, room_context, asynchronous=True)
        loop = asyncio.get_running_loop()
        producer = worker = None
        watches = set()
        resources = set()
        work_stopped = False
        response = ""
        generation_state = GenerationState.INCOMPLETE

        def track(task):
            resources.add(task)
            task.add_done_callback(resources.discard)
            return self._track_generation(task)

        async def produce() -> None:
            nonlocal response, generation_state
            pending = ""
            source = generation if generation is not None else self._generation
            with self._ownership:
                if any(not task.done() for task in self._retained_generations) or any(item.sync_busy for item in self._generation_sources):
                    raise RuntimeError("previous_generation_still_running")
                self._generation_sources = {source}
            tokens = source.astream(prompt, max_tokens=max_tokens, temperature=temperature)
            iterator = tokens.__aiter__()
            read = None
            failed = False
            remaining = self.budgets.generation_s
            try:
                while True:
                    started = loop.time()
                    read = track(asyncio.create_task(anext(iterator)))
                    done, _ = await asyncio.wait({read}, timeout=max(0, remaining))
                    remaining -= loop.time() - started
                    if not done:
                        read.cancel()
                        raise ResponseTimedOut(ResponseStage.GENERATION)
                    try:
                        chunk = read.result()
                    except StopAsyncIteration:
                        break
                    token = chunk.get("token", "")
                    if type(token) is not str:
                        raise ValueError("invalid_stream_text")
                    if len(response) + len(token) > source.max_chars:
                        raise ValueError("generation_text_limit")
                    response += token
                    phrases, pending = self._split(pending + token)
                    for phrase in phrases:
                        await self._put(session, _Utterance(phrase, stream=self.streaming_enabled, emotion=emotion_adjustment))
                if not response.strip():
                    raise RuntimeError("generation_empty")
                generation_state = GenerationState.COMPLETE
                phrases, _ = self._split(pending, final=True)
                for phrase in phrases:
                    await self._put(session, _Utterance(phrase, stream=self.streaming_enabled, emotion=emotion_adjustment))
            except BaseException:
                failed = True
                raise
            finally:
                if read is not None and not read.done() and not read.cancelling():
                    read.cancel()

                async def close_source():
                    if read is not None:
                        await asyncio.gather(read, return_exceptions=True)
                    await tokens.aclose()

                cleanup = track(asyncio.create_task(close_source()))
                try:
                    done, _ = await asyncio.wait({cleanup}, timeout=self.budgets.cleanup_s)
                    if done:
                        cleanup.result()
                    elif not failed:
                        raise ResponseTimedOut(ResponseStage.CLEANUP)
                finally:
                    if not cleanup.done():
                        with self._ownership:
                            self._retained_generations.add(cleanup)
                    self._finish(session)

        async def output_limit():
            while session.delivery_started_at is None:
                await asyncio.sleep(.005)
            await asyncio.sleep(max(0, session.delivery_started_at + self.budgets.delivery_s - loop.time()))
            return ResponseStage.DELIVERY

        async def total_limit():
            await asyncio.sleep(max(0, deadline - loop.time()))
            return ResponseStage.DELIVERY if generation_state is GenerationState.COMPLETE else ResponseStage.GENERATION

        async def stop_work(*, wait_worker=True):
            nonlocal work_stopped
            started = loop.time()
            if producer is not None and not producer.done() and not producer.cancelling():
                producer.cancel()
            await self._stop_session(session)
            work_stopped = True
            while True:
                tasks = {task for task in resources if not task.done()}
                owned = (producer, worker) if wait_worker else (producer,)
                tasks.update(task for task in owned if task is not None and not task.done())
                remaining = self.budgets.cleanup_s - (loop.time()-started)
                if not tasks or remaining <= 0:
                    return
                await self._settle_tasks(tasks, remaining)

        try:
            if on_owned is not None:
                on_owned()
            queue_limit = self.budgets.queue_wait_s if deadline is None else min(self.budgets.queue_wait_s, max(0, deadline-loop.time()))
            try:
                async with asyncio.timeout(queue_limit):
                    acquired = await self._acquire(session)
            except TimeoutError as error:
                raise ResponseTimedOut(ResponseStage.QUEUE) from error
            if not acquired:
                return ResponseResult("", GenerationState.CANCELLED, DeliveryState.CANCELLED)
            if self.llm_use_filler if use_filler is None else use_filler:
                session.filler = self.llm_filler_phrases[0]
            worker = asyncio.create_task(asyncio.to_thread(self._worker, session))
            producer = track(asyncio.create_task(produce()))
            interrupted = asyncio.create_task(session.interrupted.wait())
            output_timer = asyncio.create_task(output_limit())
            watches = {interrupted, output_timer}
            overall_timer = None if deadline is None else asyncio.create_task(total_limit())
            if overall_timer is not None:
                watches.add(overall_timer)
            pending_tasks = {worker, producer, *watches}
            while True:
                done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                # Output failure is not retried through the same failed device.
                if session.delivery in {DeliveryState.FAILED, DeliveryState.BLOCKED}:
                    await stop_work()
                    return self._result(session, response, generation_state, session.error)
                if session.cancelled.is_set():
                    await stop_work(wait_worker=False)
                    return ResponseResult("", GenerationState.CANCELLED, DeliveryState.CANCELLED)
                if producer in done:
                    if producer.cancelled():
                        await stop_work(wait_worker=False)
                        return ResponseResult("", GenerationState.CANCELLED, DeliveryState.CANCELLED)
                    producer.result()
                    pending_tasks.discard(producer)
                if worker in done:
                    await worker
                    if producer.done():
                        producer.result()
                        return self._result(session, response, generation_state)
                    pending_tasks.discard(worker)
                if output_timer in done:
                    raise ResponseTimedOut(output_timer.result())
                if overall_timer is not None and overall_timer in done:
                    raise ResponseTimedOut(overall_timer.result())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await stop_work()
            if session.cancel_requested.is_set():
                return ResponseResult("", GenerationState.CANCELLED, DeliveryState.CANCELLED)
            if generation_state is GenerationState.INCOMPLETE:
                generation_state = GenerationState.FAILED
            stage = error.stage if isinstance(error, ResponseTimedOut) else None
            text = "Lo siento, tarde demasiado. Intenta de nuevo." if stage is not None else "No pude procesar eso ahora mismo, probá de nuevo."
            # A separate cancelled epoch stops old PCM. This is the only error
            # attempt and its queue/synthesis/drain all share the feedback cap.
            try:
                feedback = await self.aspeak(text, zone_id=session.zone_id, timeout_s=self.budgets.feedback_s)
            except RuntimeError as unavailable:
                return ResponseResult(text, generation_state, DeliveryState.FAILED, str(unavailable), timeout_stage=stage)
            if feedback.delivery is DeliveryState.CANCELLED:
                return ResponseResult("", GenerationState.CANCELLED, DeliveryState.CANCELLED)
            if feedback.delivery is DeliveryState.TIMED_OUT:
                return ResponseResult(text, generation_state, DeliveryState.TIMED_OUT,
                                      "response_feedback_timeout", timeout_stage=ResponseStage.FEEDBACK)
            return ResponseResult(text, generation_state, feedback.delivery, str(error),
                                  spoken_segments=session.spoken + feedback.spoken_segments, timeout_stage=stage)
        finally:
            async def cleanup_all():
                if not work_stopped and ((producer is not None and not producer.done()) or (worker is not None and not worker.done())):
                    await stop_work(wait_worker=False)
                for task in watches:
                    task.cancel()
                await asyncio.gather(*watches, return_exceptions=True)

            cleanup = asyncio.create_task(cleanup_all())
            cancelled = False
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
            finally:
                self._controller_finished(session)
            if cancelled:
                raise asyncio.CancelledError

    def speak_with_llm_stream(self, prompt: str, zone_id=None, **kwargs) -> str:
        """Blocking compatibility adapter; async routers use arespond."""
        kwargs.pop("stream", None)
        return asyncio.run(self.arespond(prompt, zone_id, **kwargs)).text

    def _apply_emotion_adjustment(self, adjustment: dict) -> None:
        setter = getattr(self.tts, "set_audio_adjustment", None)
        if callable(setter):
            if "pitch_shift" in adjustment:
                setter(pitch=adjustment["pitch_shift"])
            if "speech_rate" in adjustment:
                setter(speed=adjustment["speech_rate"])
