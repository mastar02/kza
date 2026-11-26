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
import importlib.metadata
import importlib.util
import inspect
import logging
import math
import pathlib
import re
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.config import KOKORO_LANG_CODES, canonical_kokoro_lang
from nexa_ml.observability import InferenceTiming
from nexa_ml.offline import assert_offline_effective, reject_remote_source
from nexa_ml.slot import InvalidInputError, ModelSlot

logger = logging.getLogger(__name__)

TTS_SLOT_ID = "tts"
KOKORO_SAMPLE_RATE = 24_000
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


def supported_kwargs(factory: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keyword arguments ``factory`` accepts (``None`` values dropped).

    Kokoro 0.7.x has ``KModel(config, model)`` / ``KPipeline(lang_code, model, trf, device)``;
    0.9.x adds ``repo_id`` to both. Inspecting the real signature keeps one adapter
    working across both without passing an unknown keyword.
    """
    present = {k: v for k, v in candidates.items() if v is not None}
    params = inspect.signature(factory).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return present
    return {k: v for k, v in present.items() if k in params}


@dataclass(frozen=True)
class KokoroLocalFiles:
    """Optional local Kokoro layout: ``<dir>/config.json``, ``<dir>/*.pth``, ``<dir>/voices/*.pt``."""

    root: pathlib.Path
    config: pathlib.Path
    weights: pathlib.Path

    @classmethod
    def find(cls, repo_id: str, models_dir: str | None) -> "KokoroLocalFiles | None":
        """Locate a complete local layout (``repo_id`` as a dir, or ``<models_dir>/kokoro``)."""
        candidates = [pathlib.Path(repo_id)]
        if models_dir:
            candidates.append(pathlib.Path(models_dir) / "kokoro")
        for root in candidates:
            config = root / "config.json"
            weights = sorted(root.glob("*.pth"))
            if root.is_dir() and config.is_file() and weights:
                return cls(root, config, weights[0])
        return None

    def voice_path(self, voice: str) -> str | None:
        """Local ``voices/<voice>.pt`` if present."""
        path = self.root / "voices" / f"{voice}.pt"
        return str(path) if path.is_file() else None


# Kokoro builds misaki.en.G2P for English ('a' US, 'b' GB); misaki 0.9.4 (en.py:499-503)
# calls ``spacy.cli.download`` when the spaCy model package is missing.
SPACY_G2P_LANGS = frozenset("ab")
SPACY_EN_MODEL = "en_core_web_sm"  # misaki uses en_core_web_trf only with trf=True (never here)


def spacy_model_installed(name: str) -> bool:
    """Same criterion as misaki 0.9.4 (``spacy.util.is_package``), plus importability.

    Misaki calls ``spacy.cli.download`` whenever the model's *distribution metadata*
    is missing, even if the module itself is importable, so both must be present.
    When spaCy is installed its own ``is_package`` decides; otherwise the identical
    ``importlib.metadata.distribution`` lookup is used.
    """
    if importlib.util.find_spec(name) is None:
        return False
    if importlib.util.find_spec("spacy") is not None:
        import spacy.util  # noqa: PLC0415 - only on the real (offline) Kokoro load path

        return bool(spacy.util.is_package(name))
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def require_local_g2p(lang_code: str, *, offline: bool, installed: Callable[[str], bool]) -> None:
    """Fail closed before ``KPipeline`` would trigger a spaCy model download.

    Raises:
        FileNotFoundError: Offline, English G2P requested and the spaCy model is missing.
    """
    if offline and canonical_kokoro_lang(lang_code) in SPACY_G2P_LANGS and not installed(SPACY_EN_MODEL):
        raise FileNotFoundError(f"spaCy model {SPACY_EN_MODEL} is not installed (offline mode)")


class KokoroEngine:
    """Real Kokoro adapter: one shared ``KModel`` and one ``KPipeline`` per language.

    Constructed only on the slot thread by :func:`kokoro_loader`. With
    ``offline=True`` it refuses URI sources, verifies the Hugging Face client is
    offline (so any cache miss raises instead of downloading) and prefers a
    complete local layout when one exists.

    Args:
        repo_id: Hugging Face repo id or a local directory.
        device: Torch device.
        default_lang: Language whose pipeline is built eagerly.
        models_dir: Optional local model root (``<models_dir>/kokoro``).
        offline: Fail closed instead of reaching the network.
        kokoro_module: Injectable module (tests); defaults to ``import kokoro``.
        spacy_installed: Injectable spaCy-model presence probe (tests).
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        default_lang: str,
        *,
        models_dir: str | None = None,
        offline: bool = True,
        kokoro_module: Any = None,
        spacy_installed: Callable[[str], bool] = spacy_model_installed,
    ) -> None:
        if offline:
            reject_remote_source(repo_id)
            assert_offline_effective()
        self._spacy_installed = spacy_installed
        default_lang = canonical_kokoro_lang(default_lang)
        require_local_g2p(default_lang, offline=offline, installed=spacy_installed)
        if kokoro_module is None:
            import kokoro as kokoro_module  # noqa: PLC0415 - heavy, lazy by design

        self._kokoro = kokoro_module
        self._repo_id = repo_id
        self._device = device
        self._offline = offline
        self._local = KokoroLocalFiles.find(repo_id, models_dir)
        kmodel_cls = kokoro_module.KModel
        model_kwargs = supported_kwargs(
            kmodel_cls,
            {
                "repo_id": None if self._local else repo_id,
                "config": str(self._local.config) if self._local else None,
                "model": str(self._local.weights) if self._local else None,
            },
        )
        default_repo = getattr(kmodel_cls, "REPO_ID", repo_id)
        if self._local is None and "repo_id" not in model_kwargs and repo_id != default_repo:
            raise ValueError("installed kokoro cannot load a non-default repo_id without a local layout")
        self._model = kmodel_cls(**model_kwargs).to(device).eval()
        self._pipelines: dict[str, Any] = {}
        self.pipeline(default_lang)

    def pipeline(self, lang_code: str) -> Any:
        """Return (creating on the slot thread if needed) the pipeline for ``lang_code``.

        Raises:
            ValueError: Non-canonical language code.
            FileNotFoundError: Offline English G2P without the local spaCy model.
        """
        lang_code = canonical_kokoro_lang(lang_code)
        pipe = self._pipelines.get(lang_code)
        if pipe is None:
            require_local_g2p(lang_code, offline=self._offline, installed=self._spacy_installed)
            factory = self._kokoro.KPipeline
            kwargs = supported_kwargs(
                factory,
                {
                    "lang_code": lang_code,
                    "repo_id": self._repo_id if self._local is None else None,
                    "model": self._model,
                    "device": self._device,
                },
            )
            pipe = factory(**kwargs)
            self._pipelines[lang_code] = pipe
        return pipe

    def prepare_voice(self, pipe: Any, voice: str) -> str:
        """Resolve and load (cache) a voice pack; a missing pack is a client error.

        Returns:
            The voice argument to pass to the pipeline (a local ``.pt`` path when
            the local layout has it, otherwise the voice name).

        Raises:
            InvalidInputError: The voice pack is not available locally.
            OfflineViolationError: Offline mode is no longer effective.
        """
        if self._offline:
            assert_offline_effective()
        resolved = (self._local.voice_path(voice) if self._local else None) or voice
        try:
            pipe.load_voice(resolved)
        except (OSError, ValueError, KeyError) as exc:  # HF offline cache miss / unknown voice
            raise InvalidInputError("voice is not available") from exc
        return resolved

    def close(self) -> None:
        """Drop pipelines and the model so ``gc`` can release VRAM."""
        self._pipelines.clear()
        self._model = None


def kokoro_loader(
    repo_id: str, device: str, default_lang: str, *, models_dir: str | None, offline: bool
) -> Callable[[], Any]:
    """Return a zero-arg factory building a :class:`KokoroEngine`."""
    return lambda: KokoroEngine(repo_id, device, default_lang, models_dir=models_dir, offline=offline)


class TtsSlot:
    """Streaming synthesis on a single Kokoro slot.

    Args:
        slot: Underlying model slot; its model must expose ``pipeline(lang_code)``
            returning a ``KPipeline``-like callable and ``prepare_voice(pipe, voice)``.
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

    async def stream(
        self, opts: SynthesisOptions, *, timing: InferenceTiming | None = None
    ) -> AsyncIterator[np.ndarray]:
        """Yield one PCM array per ``KPipeline`` ``Result`` as soon as it exists.

        Raises:
            SlotUnavailableError: Slot not ready.
            ResourceExhaustedError: Queue full or OOM.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        stop = threading.Event()
        timing = timing or InferenceTiming()

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
                voice = engine.prepare_voice(pipe, opts.voice)
                timing.voice = opts.voice  # Log only a voice pack the model could load.
                source = iter(pipe(opts.text, voice=voice, speed=opts.speed))
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
                try:
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()
                finally:
                    emit(_END)  # always: a failing close() must never leave the RPC waiting

        future = self.slot.submit(timing.wrap(self.slot.slot_id, produce))
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
