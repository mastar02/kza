"""Kokoro-82M TTS with real incremental streaming.

The legacy ``src/tts/piper_tts.py`` (~L833) concatenated every ``KPipeline``
``Result`` and only then sliced the buffer into chunks, so the first audio left
after the whole utterance was synthesised. Here the slot thread forwards each
``Result.audio`` to the event loop as soon as the generator yields it; the gRPC
handler emits one ``TtsAudioChunk`` per ``Result`` and then a trailing empty chunk
with ``is_final=true`` (no lookahead, so the first chunk is never held back).

Cancellation: when the client cancels, the handler sets a stop flag; the slot
thread stops pulling the generator at the next ``Result`` boundary and closes it.
The slot stays busy until that thread really stops, so the next request never
shares the model with an abandoned stream.
"""

import asyncio
import logging
import math
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.slot import InvalidInputError, ModelSlot

logger = logging.getLogger(__name__)

TTS_SLOT_ID = "tts"
KOKORO_SAMPLE_RATE = 24_000
KOKORO_LANG_CODES = frozenset("abefhijpz")
MAX_TEXT_CHARS = 2000
_VOICE_RE = re.compile(r"^[a-z]{2}_[a-z0-9]{1,32}$")
_END = object()


@dataclass(frozen=True)
class SynthesisOptions:
    """Validated synthesis request."""

    text: str
    voice: str
    lang_code: str
    speed: float


def build_synthesis_options(
    *, text: str, voice: str, lang_code: str, speed: float, default_voice: str, default_lang: str
) -> SynthesisOptions:
    """Validate a synthesis request (messages never echo the text).

    Raises:
        InvalidInputError: Empty/oversized text, malformed voice, unknown language
            code or speed outside 0.5..2.0.
    """
    if not text.strip():
        raise InvalidInputError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise InvalidInputError(f"text longer than {MAX_TEXT_CHARS} characters")
    voice = voice or default_voice
    if not _VOICE_RE.match(voice):
        raise InvalidInputError("voice is malformed")
    lang = lang_code or default_lang
    if lang not in KOKORO_LANG_CODES:
        raise InvalidInputError("lang_code is not a Kokoro language")
    speed = speed or 1.0
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise InvalidInputError("speed must be within 0.5..2.0")
    return SynthesisOptions(text, voice, lang, float(speed))


def result_audio(result: Any) -> np.ndarray | None:
    """Extract mono float32 PCM from a ``KPipeline`` ``Result`` (duck-typed tensors)."""
    audio = getattr(result, "audio", None)
    if audio is None:
        return None
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    return array if array.size else None


class KokoroEngine:
    """Real Kokoro adapter: one shared ``KModel`` and one ``KPipeline`` per language.

    Only constructed by :func:`kokoro_loader` on the slot thread; never imported
    by tests.
    """

    def __init__(self, repo_id: str, device: str, default_lang: str) -> None:
        import kokoro  # noqa: PLC0415 - heavy, lazy by design

        self._kokoro = kokoro
        self._repo_id = repo_id
        self._device = device
        self._model = kokoro.KModel(repo_id=repo_id).to(device).eval()
        self._pipelines: dict[str, Any] = {}
        self.pipeline(default_lang)

    def pipeline(self, lang_code: str) -> Any:
        """Return (creating on the slot thread if needed) the pipeline for ``lang_code``."""
        pipe = self._pipelines.get(lang_code)
        if pipe is None:
            pipe = self._kokoro.KPipeline(
                lang_code=lang_code, repo_id=self._repo_id, model=self._model, device=self._device
            )
            self._pipelines[lang_code] = pipe
        return pipe

    def close(self) -> None:
        """Drop pipelines and the model so ``gc`` can release VRAM."""
        self._pipelines.clear()
        self._model = None


def kokoro_loader(repo_id: str, device: str, default_lang: str) -> Callable[[], Any]:
    """Return a zero-arg factory building a :class:`KokoroEngine`."""
    return lambda: KokoroEngine(repo_id, device, default_lang)


class TtsSlot:
    """Streaming synthesis on a single Kokoro slot.

    Args:
        slot: Underlying model slot; its model must expose ``pipeline(lang_code)``
            returning a ``KPipeline``-like callable.
        default_voice: Voice used when the request leaves it empty.
        default_lang: Language used when the request leaves it empty.
    """

    sample_rate = KOKORO_SAMPLE_RATE

    def __init__(self, slot: ModelSlot, *, default_voice: str, default_lang: str) -> None:
        self.slot = slot
        self.default_voice = default_voice
        self.default_lang = default_lang

    def options(self, *, text: str, voice: str, lang_code: str, speed: float) -> SynthesisOptions:
        """Validate a request with this slot's defaults."""
        return build_synthesis_options(
            text=text,
            voice=voice,
            lang_code=lang_code,
            speed=speed,
            default_voice=self.default_voice,
            default_lang=self.default_lang,
        )

    async def stream(self, opts: SynthesisOptions) -> AsyncIterator[np.ndarray]:
        """Yield one PCM array per ``KPipeline`` ``Result`` as soon as it exists.

        Raises:
            SlotUnavailableError: Slot not ready.
            ResourceExhaustedError: Queue full or OOM.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        stop = threading.Event()

        def emit(item: Any) -> bool:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # loop closed: nobody is listening any more
                stop.set()
                return False
            return True

        def produce(engine: Any) -> int:
            produced = 0
            source = None
            try:
                if stop.is_set():
                    return produced
                pipe = engine.pipeline(opts.lang_code)
                source = iter(pipe(opts.text, voice=opts.voice, speed=opts.speed))
                while not stop.is_set():
                    try:
                        result = next(source)
                    except StopIteration:
                        break
                    if stop.is_set():
                        break
                    audio = result_audio(result)
                    if audio is not None and emit(audio):
                        produced += 1
                return produced
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()
                emit(_END)

        future = self.slot.submit(produce)
        finished = False
        try:
            while True:
                item = await queue.get()
                if item is _END:
                    break
                yield item
            finished = True
            await self.slot.track(future)
        finally:
            if not finished:
                stop.set()
                future.add_done_callback(_consume)


def _consume(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()
