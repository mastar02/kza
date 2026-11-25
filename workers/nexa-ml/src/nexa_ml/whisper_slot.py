"""faster-whisper STT: one COMMAND slot (v3-turbo) and an AMBIENT pool (N x large-v3).

Each Whisper instance is a :class:`~nexa_ml.slot.ModelSlot` with its own standard
``ThreadPoolExecutor(max_workers=1)``; nothing from the legacy
``pinned_executor.py`` or CPython private APIs is used. ``vad_filter`` is fixed
per mode (COMMAND off, AMBIENT on) because the proto carries no such field.
"""

import asyncio
import logging
import pathlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.audio import parse_device
from nexa_ml.slot import (
    InvalidInputError,
    ModelSlot,
    ResourceExhaustedError,
    SlotState,
    SlotUnavailableError,
)

logger = logging.getLogger(__name__)

COMMAND_SLOT_ID = "stt-command"
AMBIENT_SLOT_PREFIX = "stt-ambient-"
MAX_BEAM_SIZE = 10
MAX_PROMPT_CHARS = 1024
_LANG_RE = re.compile(r"^[a-z]{2,3}$")

# Nexa config aliases -> faster-whisper model ids (faster_whisper.utils._MODELS).
WHISPER_MODEL_IDS = {
    "whisper-v3-turbo": "large-v3-turbo",
    "whisper-large-v3-turbo": "large-v3-turbo",
    "whisper-large-v3": "large-v3",
}


@dataclass(frozen=True)
class TranscribeOptions:
    """Validated decoding options for one request."""

    language: str
    beam_size: int
    vad_filter: bool
    initial_prompt: str | None
    hotwords: str | None


@dataclass(frozen=True)
class Transcription:
    """Aggregated transcription result."""

    text: str
    compression_ratio: float
    avg_logprob: float
    no_speech_prob: float
    latency_ms: int
    model_id: str


def build_options(
    *,
    language: str,
    beam_size: int,
    vad_filter: bool,
    initial_prompt: str,
    hotwords: str,
    default_language: str,
    default_beam: int,
) -> TranscribeOptions:
    """Validate request options, applying slot defaults for zero/empty values.

    Raises:
        InvalidInputError: Bad language code, beam size or oversized prompt/hotwords.
    """
    lang = language or default_language
    if not _LANG_RE.match(lang):
        raise InvalidInputError("language must be an ISO 639 code")
    beam = beam_size or default_beam
    if not 1 <= beam <= MAX_BEAM_SIZE:
        raise InvalidInputError(f"beam_size must be within 1..{MAX_BEAM_SIZE}")
    if len(initial_prompt) > MAX_PROMPT_CHARS or len(hotwords) > MAX_PROMPT_CHARS:
        raise InvalidInputError("initial_prompt/hotwords too long")
    return TranscribeOptions(lang, beam, vad_filter, initial_prompt or None, hotwords or None)


def transcribe_blocking(
    model: Any, audio: np.ndarray, opts: TranscribeOptions, model_id: str
) -> Transcription:
    """Run faster-whisper on the slot thread and aggregate segment statistics.

    Aggregation matches the legacy ``whisper_fast`` contract: max compression
    ratio, mean ``avg_logprob`` and mean ``no_speech_prob``. With no segments the
    text is empty and ``no_speech_prob`` is 1.0.
    """
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "language": opts.language,
        "beam_size": opts.beam_size,
        "initial_prompt": opts.initial_prompt,
        "condition_on_previous_text": False,
        "vad_filter": opts.vad_filter,
    }
    if opts.hotwords:
        kwargs["hotwords"] = opts.hotwords
    segments, _info = model.transcribe(audio, **kwargs)
    seg_list = list(segments)
    text = " ".join(s.text.strip() for s in seg_list).strip()
    if seg_list:
        compression = max(float(s.compression_ratio) for s in seg_list)
        logprob = sum(float(s.avg_logprob) for s in seg_list) / len(seg_list)
        no_speech = sum(float(s.no_speech_prob) for s in seg_list) / len(seg_list)
    else:
        compression, logprob, no_speech = 0.0, 0.0, 1.0
    latency = int((time.perf_counter() - start) * 1000)
    return Transcription(text, compression, logprob, no_speech, latency, model_id)


def resolve_model_source(name: str, models_dir: str | None) -> str:
    """Map a config alias to a local CT2 directory or a faster-whisper model id."""
    candidate = pathlib.Path(name)
    if candidate.is_absolute():
        return name
    if models_dir is not None:
        local = pathlib.Path(models_dir) / name
        if local.is_dir():
            return str(local)
    return WHISPER_MODEL_IDS.get(name, name)


def faster_whisper_loader(
    name: str, device: str, compute_type: str, *, models_dir: str | None, offline: bool
) -> Callable[[], Any]:
    """Return a zero-arg factory creating a ``faster_whisper.WhisperModel``.

    The import is deferred to the slot thread so the worker never imports
    faster-whisper/CTranslate2 unless a real GPU loader actually runs.
    """

    def load() -> Any:
        from faster_whisper import WhisperModel  # noqa: PLC0415 - heavy, lazy by design

        kind, index = parse_device(device)
        return WhisperModel(
            resolve_model_source(name, models_dir),
            device=kind,
            device_index=index,
            compute_type=compute_type,
            download_root=models_dir,
            local_files_only=offline,
        )

    return load


class WhisperSlot:
    """Single Whisper instance (COMMAND or one AMBIENT pool member)."""

    def __init__(
        self, slot: ModelSlot, *, vad_filter: bool, default_beam: int, default_language: str
    ) -> None:
        self.slot = slot
        self.vad_filter = vad_filter
        self.default_beam = default_beam
        self.default_language = default_language

    def options(
        self, *, language: str, beam_size: int, initial_prompt: str, hotwords: str
    ) -> TranscribeOptions:
        """Validate options with this slot's defaults."""
        return build_options(
            language=language,
            beam_size=beam_size,
            vad_filter=self.vad_filter,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            default_language=self.default_language,
            default_beam=self.default_beam,
        )

    async def transcribe(self, audio: np.ndarray, opts: TranscribeOptions) -> Transcription:
        """Transcribe on this slot's executor."""
        name = self.slot.model_name
        return await self.slot.run(lambda model: transcribe_blocking(model, audio, opts, name))


class WhisperPool:
    """AMBIENT pool: N independent Whisper slots, each request served by an idle member.

    Args:
        members: Pool slots (each with its own executor).
        max_waiting: Maximum requests waiting for a free member before
            ``ResourceExhaustedError``.
    """

    def __init__(self, members: list[WhisperSlot], *, max_waiting: int) -> None:
        if not members:
            raise ValueError("pool needs at least one member")
        self.members = members
        self._max_waiting = max_waiting
        self._waiting = 0
        self._cond: asyncio.Condition | None = None
        self._wake_tasks: set[asyncio.Task[None]] = set()
        for member in members:
            member.slot.on_change = self._schedule_wake

    @property
    def template(self) -> WhisperSlot:
        """Member used for option defaults (all members share config)."""
        return self.members[0]

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def _schedule_wake(self) -> None:
        task = asyncio.get_running_loop().create_task(self._wake())
        self._wake_tasks.add(task)
        task.add_done_callback(self._wake_tasks.discard)

    async def _wake(self) -> None:
        cond = self._condition()
        async with cond:
            cond.notify_all()

    def _any_ready(self) -> bool:
        return any(m.slot.state is SlotState.READY for m in self.members)

    def _pick_idle(self) -> WhisperSlot | None:
        for member in self.members:
            if member.slot.state is SlotState.READY and member.slot.inflight == 0:
                return member
        return None

    async def transcribe(self, audio: np.ndarray, opts: TranscribeOptions) -> Transcription:
        """Wait for an idle READY member and transcribe on it.

        Raises:
            SlotUnavailableError: No member is READY.
            ResourceExhaustedError: Too many requests already waiting.
        """
        cond = self._condition()
        async with cond:
            if not self._any_ready():
                raise SlotUnavailableError("ambient pool has no ready member")
            if self._pick_idle() is None:
                if self._waiting >= self._max_waiting:
                    raise ResourceExhaustedError("ambient pool queue is full")
                self._waiting += 1
                try:
                    await cond.wait_for(lambda: self._pick_idle() is not None or not self._any_ready())
                finally:
                    self._waiting -= 1
            member = self._pick_idle()
            if member is None:
                raise SlotUnavailableError("ambient pool has no ready member")
            name = member.slot.model_name
            future = member.slot.submit(lambda model: transcribe_blocking(model, audio, opts, name))
        return await member.slot.track(future)
