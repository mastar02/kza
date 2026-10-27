"""Command transcription and identification with ownership of actual native work."""

import asyncio
from dataclasses import dataclass, field
import logging
import threading
import time

import numpy as np

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission
from src.memory.producer_scopes import ProducerScopes
from src.memory.prompt_context import SourcesInvalidated, run_private
from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedCommand:
    """Processed capture, including primary confidence and a distinct veto receipt."""

    text: str
    user: object | None = None
    emotion: object | None = None
    speaker_confidence: float = 0.0
    stt_confidence: STTResult | None = None
    timings: dict = field(default_factory=dict)
    success: bool = False
    stt_vetoed: bool = False


class CommandProcessor:
    """Retain actual model work after cancellation or a shadow observation timeout.

    The router binds capture lineage before admission. Privacy remains active
    independently of semantic memory. Shadow errors/timeouts remain fail-open;
    invalid capture provenance prevents returning any private result.
    """

    def __init__(self, stt, speaker_identifier=None, user_manager=None,
                 emotion_detector=None, sample_rate: int = 16000, shadow_stt=None,
                 shadow_veto: bool = False, shadow_veto_timeout_s: float = 0.8,
                 *, privacy=None, close_timeout_s: float = 1, model_locks=None):
        self.stt, self.speaker_id = stt, speaker_identifier
        self.user_manager, self.emotion_detector = user_manager, emotion_detector
        self.sample_rate, self.shadow_stt = sample_rate, shadow_stt
        self.shadow_veto, self.shadow_veto_timeout_s = shadow_veto, shadow_veto_timeout_s
        self._privacy, self._close_timeout_s = privacy, close_timeout_s
        self._work = OwnedBlockingWork(capacity=8, close_timeout_s=close_timeout_s)
        self._locks = model_locks if model_locks is not None else {}
        for model in (stt, speaker_identifier, emotion_detector, shadow_stt):
            if model is not None:
                self._locks.setdefault(id(model), threading.Lock())
        if (speaker_identifier is not None and getattr(user_manager,'voice_review_managed',False) is True):
            self._locks[id(speaker_identifier)] = user_manager.voice_registry.register_encoder(speaker_identifier)
        self._commands: set[asyncio.Task] = set()
        self._shadow_tasks: set[asyncio.Task] = set()
        self._emotion_tasks: set[asyncio.Task] = set()
        self._scopes = ProducerScopes(self.snapshot_operations, capacity=32)
        self._closed = False
        self._closing: asyncio.Task | None = None
        self._current_user = self._current_emotion = None
        self._current_sources = ()
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._embeddings_cache_version = 0
        self._embeddings_cache_time = 0
        self._embeddings_cache_ttl = 60.0
        if privacy is not None:
            privacy.add_borrower(self.begin_close)

    def _available(self) -> bool:
        return not self._closed and (self._privacy is None or
            self._privacy.sources_available(PRODUCER_CONTEXT.get().sources))

    def _require_live(self) -> None:
        if not self._available():
            raise SourcesInvalidated("command_sources_invalidated")

    def _invoke(self, model, function, args, cancelled, guarded):
        result = None
        invalidated = False

        def require_live():
            self._require_live()
            if cancelled.is_set():
                raise SourcesInvalidated("command_observer_cancelled")

        try:
            with model_admission(self._locks[id(model)], require_live):
                result = function(*args, require_live=require_live) if guarded else function(*args)
                require_live()
                return result
        except SourcesInvalidated:
            invalidated = True
        except Exception as error:
            logger.warning("Command model operation failed: %s", type(error).__name__)
        finally:
            model = function = args = result = cancelled = require_live = None
        # Raise outside except: native exception chains can retain private input.
        if invalidated:
            raise SourcesInvalidated("command_sources_invalidated")
        raise RuntimeError("command_model_failed")

    async def _native(self, model, function, *args, guarded=False):
        cancelled = threading.Event()
        try:
            return await self._work.run(self._invoke, model, function, args, cancelled, guarded)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            model = function = args = cancelled = None

    def load_models(self) -> None:
        """Load models from an owned worker or an explicit synchronous caller."""
        for model in (self.stt, self.shadow_stt, self.speaker_id, self.emotion_detector):
            if model is not None and hasattr(model, "load"):
                with model_admission(self._locks[id(model)], self._require_open):
                    model.load()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("command_processor_closed")

    async def aload_models(self) -> None:
        """Keep actual model warmup owned after startup cancellation."""
        await self._work.run(self.load_models)

    def _track(self, task: asyncio.Task, tasks: set[asyncio.Task]) -> asyncio.Task:
        tasks.add(task)

        def finished(done):
            tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    async def process_command(self, audio: np.ndarray, use_parallel: bool = True,
                              pretranscribed_text: str | None = None) -> ProcessedCommand:
        """Process a capture while preserving ownership of cancelled native work."""
        task = None
        try:
            with self._scopes.operation(PRODUCER_CONTEXT.get().sources):
                self._require_live()
                if len(self._commands) + len(self._emotion_tasks) >= 8:
                    raise RuntimeError("command_processor_capacity")
                task = self._track(asyncio.create_task(run_private(
                    self._process_command(audio, use_parallel, pretranscribed_text),
                    self._available)), self._commands)
                return await task
        finally:
            audio = pretranscribed_text = task = None

    def _start_shadow(self, audio):
        if len(self._shadow_tasks) >= 8:
            return None
        return self._track(asyncio.create_task(run_private(
            self._native(self.shadow_stt, self.shadow_stt.transcribe_with_confidence,
                         audio, self.sample_rate), self._available)), self._shadow_tasks)

    async def _process_command(self, audio, use_parallel, pretranscribed_text):
        result = ProcessedCommand(text="")
        shadow = stt_result = speaker = emotion = text = None
        started = time.perf_counter()
        try:
            if self.shadow_stt is not None and self.shadow_veto:
                shadow = self._start_shadow(audio)
            if pretranscribed_text is not None:
                text = pretranscribed_text
                result.timings.update(stt=0.0, stt_skipped=1)
                speaker, emotion = await self._auxiliary(audio, use_parallel)
            elif use_parallel:
                text, stt_ms, speaker, emotion, stt_result = await self._process_parallel(audio)
                result.timings["stt"] = stt_ms
            else:
                stt_result = await self._primary(audio)
                text = stt_result.text if stt_result is not None else ""
                speaker, emotion = await self._auxiliary(audio, False)
            if stt_result is not None:
                result.stt_confidence = stt_result
                result.timings["stt"] = stt_result.elapsed_ms
            if speaker is not None:
                result.user, result.speaker_confidence, result.timings["speaker_id"] = speaker
            if emotion is not None:
                result.emotion = emotion
                result.timings["emotion"] = emotion.processing_time_ms
            if shadow is not None:
                text = await self._apply_shadow_veto(shadow, text, result)
            self._require_live()
            result.text, result.success = text, bool(text.strip())
            if result.success:
                self._current_user, self._current_emotion = result.user, result.emotion
                self._current_sources = PRODUCER_CONTEXT.get().sources
            result.timings["total"] = (time.perf_counter() - started) * 1000
            logger.info("[CommandProcessor] Command processed: success=%s", result.success)
            if self.shadow_stt is not None and not self.shadow_veto and len(self._shadow_tasks) < 8:
                self._track(asyncio.create_task(self._run_shadow(
                    audio, bool(text.strip()), result.timings.get("stt", 0.0))), self._shadow_tasks)
            return result
        finally:
            if shadow is not None and not shadow.done() and asyncio.current_task().cancelling():
                shadow.cancel()
            audio = pretranscribed_text = result = shadow = stt_result = speaker = emotion = text = None

    @staticmethod
    async def _collect(*operations):
        results = value = None
        try:
            results = await asyncio.gather(*operations, return_exceptions=True)
            for value in results:
                if isinstance(value, BaseException):
                    raise value
            return results
        finally:
            operations = results = value = None

    async def _process_parallel(self, audio: np.ndarray) -> tuple[str, float, tuple | None, object | None, STTResult | None]:
        """Keep the parallel result contract while retaining every actual worker."""
        stt = speaker = emotion = None
        try:
            stt, speaker, emotion = await self._collect(
                self._primary(audio), self._speaker(audio), self._emotion(audio))
            return (stt.text if stt is not None else "",
                    stt.elapsed_ms if stt is not None else 0., speaker, emotion, stt)
        finally:
            audio = stt = speaker = emotion = None

    async def _primary(self, audio):
        try:
            return await self._native(self.stt, self.stt.transcribe_with_confidence, audio, self.sample_rate)
        except SourcesInvalidated:
            raise
        except Exception:
            logger.warning("Primary STT unavailable")
            return None
        finally:
            audio = None

    async def _speaker(self, audio):
        try:
            if (self.speaker_id is not None and getattr(self.user_manager,'voice_review_managed',False) is True):
                self.user_manager.voice_registry.observe(self.speaker_id,audio,PRODUCER_CONTEXT.get().sources,sample_rate=self.sample_rate)
                return None
            if self.speaker_id is not None and self.user_manager is not None:
                return await self._native(self.speaker_id, self._identify_speaker, audio, guarded=True)
        except SourcesInvalidated:
            raise
        except Exception:
            logger.debug("Speaker identification unavailable")
        finally:
            audio = None

    async def _emotion(self, audio):
        try:
            if self.emotion_detector is not None:
                return await self._native(self.emotion_detector, self.emotion_detector.detect, audio)
        except SourcesInvalidated:
            raise
        except Exception:
            logger.debug("Emotion detection unavailable")
        finally:
            audio = None

    async def _auxiliary(self, audio, parallel):
        speaker = None
        try:
            if parallel:
                return await self._collect(self._speaker(audio), self._emotion(audio))
            speaker = await self._speaker(audio)
            return speaker, await self._emotion(audio)
        finally:
            audio = speaker = None

    @staticmethod
    def _log_shadow(primary_present: bool, primary_ms: float, shadow) -> None:
        logger.info("[STT-shadow] primary_present=%s primary_ms=%.0f shadow_present=%s shadow_ms=%.0f",
                    primary_present, primary_ms, bool(shadow.text.strip()), shadow.elapsed_ms)

    async def _apply_shadow_veto(self, shadow_future, text: str, result: ProcessedCommand) -> str:
        """Observe the veto deadline without cancelling the actual shadow task."""
        r = None
        try:
            try:
                r = await asyncio.wait_for(asyncio.shield(shadow_future), self.shadow_veto_timeout_s)
            except SourcesInvalidated:
                raise
            except Exception:
                logger.warning("[STT-veto] shadow unavailable (no veto)")
                return text
            result.timings["stt_shadow"] = r.elapsed_ms
            self._log_shadow(bool(text.strip()), result.timings.get("stt", 0.0), r)
            if text.strip() and not r.text.strip():
                logger.info("[STT-veto] discarded: empty shadow")
                result.stt_vetoed = True
                return ""
            return text
        finally:
            shadow_future = text = result = r = None

    async def _run_shadow(self, audio, primary_present, primary_ms):
        result = None
        try:
            result = await run_private(self._native(self.shadow_stt,
                self.shadow_stt.transcribe_with_confidence, audio, self.sample_rate), self._available)
            self._require_live()
            self._log_shadow(primary_present, primary_ms, result)
        except SourcesInvalidated:
            pass
        except Exception:
            logger.debug("[STT-shadow] unavailable")
        finally:
            audio = result = None

    def _get_cached_embeddings(self) -> dict[str, np.ndarray]:
        now = time.monotonic()
        version = getattr(self.user_manager, "_version", 0)
        if now - self._embeddings_cache_time > self._embeddings_cache_ttl or version != self._embeddings_cache_version or not self._embeddings_cache:
            self._embeddings_cache = self.user_manager.get_all_embeddings()
            self._embeddings_cache_time = now
            self._embeddings_cache_version = version
        return self._embeddings_cache

    def invalidate_embeddings_cache(self) -> None:
        """Discard profile copies after registry changes or native close."""
        self._embeddings_cache = {}
        self._embeddings_cache_time = 0

    def _identify_speaker(self, audio: np.ndarray, *, require_live) -> tuple:
        profiles = match = user = None
        try:
            require_live()
            profiles = self._get_cached_embeddings()
            require_live()
            if not profiles:
                return None, 0.0, 0.0
            started = time.perf_counter()
            match = self.speaker_id.identify(audio, profiles)
            elapsed = (time.perf_counter() - started) * 1000
            require_live()
            if match.is_known and match.user_id:
                user = self.user_manager.get_user(match.user_id)
                require_live()
                if user is not None:
                    self.user_manager.update_last_seen(match.user_id)
                    return user, match.confidence, elapsed
            return None, 0.0, elapsed
        finally:
            audio = profiles = match = user = require_live = None

    def prune_private(self) -> None:
        """Drop idle identification results when their capture becomes unavailable."""
        if self._closed or self._privacy is not None and not self._privacy.sources_available(self._current_sources):
            self._current_user = self._current_emotion = None
            self._current_sources = ()

    def get_current_user(self) -> object | None:
        """Read the last user only while its source remains live."""
        self.prune_private()
        return self._current_user

    def get_current_emotion(self) -> object | None:
        """Read the last emotion only while its source remains live."""
        self.prune_private()
        return self._current_emotion

    async def detect_emotion(self, audio: np.ndarray) -> object | None:
        """Detect emotion off loop under the caller's capture lineage."""
        task = None
        try:
            with self._scopes.operation(PRODUCER_CONTEXT.get().sources):
                self._require_live()
                if len(self._commands) + len(self._emotion_tasks) >= 8:
                    raise RuntimeError('command_processor_capacity')
                task = self._track(asyncio.create_task(run_private(
                    self._emotion(audio), self._available)), self._emotion_tasks)
                return await task
        finally:
            audio = task = None

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose finite controllers and native scopes, including cancelled callers."""
        return tuple(self._commands | self._shadow_tasks | self._emotion_tasks | self._work._tasks)

    def conversation_busy(self, conversation_id: str) -> bool:
        """Prune idle capture associations and query actual finite work affinity."""
        if self._privacy is None:
            raise RuntimeError('privacy_authority_unavailable')
        self.prune_private()
        return self._scopes.busy(conversation_id, self._privacy.ledger)

    def resource_status(self) -> dict:
        """Report ownership counts without capture IDs, text or speaker names."""
        return {"closed": self._closed, "commands": len(self._commands),
                "shadows": len(self._shadow_tasks), "emotions": len(self._emotion_tasks),
                "source_scopes": self._scopes.resource_status(), "native": self._work.status()}

    def begin_close(self) -> asyncio.Task:
        """Reject admission and retain actual native exit before authority release."""
        self._closed = True
        if self._closing is None:
            controllers = self._commands | self._shadow_tasks | self._emotion_tasks
            for task in controllers:
                task.cancel()
            native = self._work.begin_close()

            async def finish():
                await asyncio.gather(*controllers, native, return_exceptions=True)
                self.prune_private()
                self.invalidate_embeddings_cache()
            self._closing = asyncio.create_task(finish())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while keeping unfinished native cleanup owned."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout("command_models_still_running") from None
