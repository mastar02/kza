"""Composition root of model slots: wiring, startup loads, health and reload routing."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexa_ml import __version__
from nexa_ml.config import MlConfig
from nexa_ml.embed_slot import EMBED_SLOT_ID, EmbedSlot, sentence_transformer_loader
from nexa_ml.offline import enforce_offline_env
from nexa_ml.slot import ExecutorKind, InvalidInputError, ModelSlot, SlotSnapshot, SlotState
from nexa_ml.speaker_slot import SPEAKER_SLOT_ID, SpeakerSlot, ecapa_loader
from nexa_ml.tts_slot import TTS_SLOT_ID, TtsSlot, kokoro_loader
from nexa_ml.whisper_slot import (
    AMBIENT_SLOT_PREFIX,
    COMMAND_SLOT_ID,
    WhisperPool,
    WhisperSlot,
    faster_whisper_loader,
)

logger = logging.getLogger(__name__)


class HealthLevel:
    """Aggregate health values (mirrors ``nexa.v1.HealthState`` without importing pb here)."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ModelLoaders:
    """Model factories injected into the runtime (real GPU loaders or test fakes).

    Each factory is called on its slot's lane and returns the model object.
    ``ambient_stt`` receives the pool member index.
    """

    command_stt: Callable[[], Any]
    ambient_stt: Callable[[int], Callable[[], Any]]
    tts: Callable[[], Any]
    speaker: Callable[[], Any]
    embed: Callable[[], Any]


def real_loaders(config: MlConfig) -> ModelLoaders:
    """Build the production loaders (heavy imports stay inside each factory).

    With ``config.offline`` the Hugging Face flags are forced here as well, so any
    caller (``__main__`` or the GPU smoke test) gets the same zero-network guarantee.
    """
    md, offline = config.models_dir, config.offline
    if offline:
        enforce_offline_env()
    else:
        logger.warning("ml.offline=false: model loaders may reach the network")

    def ambient(_index: int) -> Callable[[], Any]:
        return faster_whisper_loader(
            config.ambient_stt_model,
            config.ambient_stt_device,
            config.ambient_stt_compute_type,
            models_dir=md,
            offline=offline,
        )

    return ModelLoaders(
        command_stt=faster_whisper_loader(
            config.command_stt_model,
            config.command_stt_device,
            config.command_stt_compute_type,
            models_dir=md,
            offline=offline,
        ),
        ambient_stt=ambient,
        tts=kokoro_loader(
            config.tts_model, config.tts_device, config.tts_lang_code, models_dir=md, offline=offline
        ),
        speaker=ecapa_loader(
            config.speaker_encoder_model, config.speaker_encoder_device, md, offline=offline
        ),
        embed=sentence_transformer_loader(config.embeddings_model, config.embeddings_device, md, offline),
    )


class MlRuntime:
    """Owns every model slot of the worker.

    Args:
        config: Validated worker configuration.
        loaders: Model factories (DI: real loaders in prod, fakes in tests).
        clock: Monotonic clock in seconds (injectable for tests).
    """

    def __init__(
        self, config: MlConfig, loaders: ModelLoaders, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self._clock = clock
        self._started = clock()
        mp = config.max_pending
        self.command = WhisperSlot(
            ModelSlot(
                COMMAND_SLOT_ID,
                config.command_stt_model,
                config.command_stt_device,
                loaders.command_stt,
                max_pending=mp,
            ),
            vad_filter=config.command_vad_filter,
            default_beam=config.command_beam_size,
            default_language=config.stt_language,
        )
        members = [
            WhisperSlot(
                ModelSlot(
                    f"{AMBIENT_SLOT_PREFIX}{i}",
                    config.ambient_stt_model,
                    config.ambient_stt_device,
                    loaders.ambient_stt(i),
                    max_pending=1,
                ),
                vad_filter=config.ambient_vad_filter,
                default_beam=config.ambient_beam_size,
                default_language=config.stt_language,
            )
            for i in range(config.ambient_stt_workers)
        ]
        self.ambient = WhisperPool(members, max_waiting=mp)
        self.tts = TtsSlot(
            ModelSlot(TTS_SLOT_ID, config.tts_model, config.tts_device, loaders.tts, max_pending=mp),
            default_voice=config.tts_voice,
            default_lang=config.tts_lang_code,
        )
        self.speaker = SpeakerSlot(
            ModelSlot(
                SPEAKER_SLOT_ID,
                config.speaker_encoder_model,
                config.speaker_encoder_device,
                loaders.speaker,
                max_pending=mp,
            )
        )
        self.embed = EmbedSlot(
            ModelSlot(
                EMBED_SLOT_ID,
                config.embeddings_model,
                config.embeddings_device,
                loaders.embed,
                max_pending=mp,
                executor_kind=ExecutorKind.TO_THREAD,
            ),
            max_texts=config.embed_max_texts,
        )
        self._slots: dict[str, ModelSlot] = {
            s.slot_id: s
            for s in (
                self.command.slot,
                *(m.slot for m in members),
                self.tts.slot,
                self.speaker.slot,
                self.embed.slot,
            )
        }

    @property
    def slot_ids(self) -> list[str]:
        """Stable slot identifiers (exposed in ``GetHealth``)."""
        return list(self._slots)

    def uptime_ms(self) -> int:
        """Milliseconds since the runtime was created."""
        return int((self._clock() - self._started) * 1000)

    async def load_all(self) -> None:
        """Load every slot concurrently (each on its own lane); failures stay ``UNAVAILABLE``."""
        results = await asyncio.gather(*(s.load() for s in self._slots.values()), return_exceptions=True)
        failed = [sid for sid, r in zip(self._slots, results, strict=True) if isinstance(r, BaseException)]
        if failed:
            logger.error("Slots failed to load: %s", ", ".join(failed))

    async def reload(self, slot_id: str) -> SlotSnapshot:
        """Reload one slot by id.

        Raises:
            InvalidInputError: Unknown ``slot_id``.
            ResourceExhaustedError: Slot is already loading/reloading.
            SlotUnavailableError: Reload failed; slot stays ``UNAVAILABLE``.
        """
        slot = self._slots.get(slot_id)
        if slot is None:
            raise InvalidInputError("unknown slot_id")
        await slot.reload()
        return slot.snapshot()

    def snapshots(self) -> list[SlotSnapshot]:
        """Health snapshot of every slot (never blocks on model threads)."""
        return [s.snapshot() for s in self._slots.values()]

    def aggregate(self) -> tuple[str, str]:
        """Aggregate health level and a short detail string."""
        states = [s.state for s in self._slots.values()]
        ready = sum(1 for st in states if st is SlotState.READY)
        detail = f"{ready}/{len(states)} slots ready"
        if ready == len(states):
            return HealthLevel.OK, detail
        if ready == 0:
            return HealthLevel.UNAVAILABLE, detail
        return HealthLevel.DEGRADED, detail

    @property
    def version(self) -> str:
        """Worker version string."""
        return __version__

    def close(self) -> None:
        """Release every slot (best effort)."""
        for slot in self._slots.values():
            slot.close()
