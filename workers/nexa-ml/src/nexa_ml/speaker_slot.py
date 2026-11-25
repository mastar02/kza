"""Single deduplicated ECAPA-TDNN speaker encoder (192-d vectors only).

Legacy KZA loaded two ECAPA instances (``main.py`` and the ambient transcriber).
The worker keeps exactly one and returns only the raw embedding: voice identity is
an acoustic signal, never a person, account, role or permission. Matching against
enrolled voices, thresholds and any authority decision live elsewhere.
"""

import logging
import pathlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.slot import ModelSlot

logger = logging.getLogger(__name__)

SPEAKER_SLOT_ID = "speaker"
SPEAKER_EMBEDDING_DIM = 192
MIN_SPEAKER_SECONDS = 0.5


class EmbeddingShapeError(RuntimeError):
    """The encoder returned a vector of the wrong dimension (maps to ``INTERNAL``)."""


@dataclass(frozen=True)
class SpeakerEmbedding:
    """Raw speaker embedding."""

    vector: np.ndarray
    encoder_model_key: str
    latency_ms: int


class EcapaEncoder:
    """Real SpeechBrain ECAPA adapter exposing ``embed(audio) -> np.ndarray``.

    Constructed only on the slot thread by :func:`ecapa_loader`.
    """

    def __init__(self, source: str, device: str, savedir: str) -> None:
        import torchaudio  # noqa: PLC0415 - heavy, lazy by design

        if not hasattr(torchaudio, "list_audio_backends"):  # speechbrain 1.x on torchaudio>=2.9
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

        self._classifier = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": device}
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Encode one mono 16 kHz utterance into its raw (unnormalised) embedding."""
        import torch  # noqa: PLC0415 - heavy, lazy by design

        with torch.inference_mode():
            tensor = torch.from_numpy(audio).unsqueeze(0)
            embedding = self._classifier.encode_batch(tensor)
        return embedding.squeeze().detach().cpu().numpy()

    def close(self) -> None:
        """Drop the classifier reference so ``gc`` can release VRAM."""
        self._classifier = None


def default_savedir(source: str, models_dir: str | None) -> str:
    """Explicit SpeechBrain ``savedir`` (its default is cwd-relative and read-only in prod)."""
    root = pathlib.Path(models_dir) if models_dir else pathlib.Path.home() / ".cache" / "nexa-ml"
    return str(root / "speechbrain" / source.replace("/", "--"))


def ecapa_loader(source: str, device: str, models_dir: str | None) -> Callable[[], Any]:
    """Return a zero-arg factory building the single :class:`EcapaEncoder`."""
    savedir = default_savedir(source, models_dir)
    return lambda: EcapaEncoder(source, device, savedir)


def embed_blocking(encoder: Any, audio: np.ndarray, model_key: str) -> SpeakerEmbedding:
    """Peak-normalise (legacy-compatible) and encode on the slot thread.

    Raises:
        EmbeddingShapeError: Encoder output is not a 192-d vector.
    """
    start = time.perf_counter()
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio = audio / peak
    vector = np.asarray(encoder.embed(audio), dtype=np.float32).reshape(-1)
    if vector.shape != (SPEAKER_EMBEDDING_DIM,):
        raise EmbeddingShapeError(f"speaker embedding has {vector.size} dims")
    return SpeakerEmbedding(vector, model_key, int((time.perf_counter() - start) * 1000))


class SpeakerSlot:
    """The one ECAPA instance of the worker."""

    def __init__(self, slot: ModelSlot) -> None:
        self.slot = slot

    async def embed(self, audio: np.ndarray) -> SpeakerEmbedding:
        """Encode on the slot executor."""
        key = self.slot.model_name
        return await self.slot.run(lambda encoder: embed_blocking(encoder, audio, key))
