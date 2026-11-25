"""Single deduplicated ECAPA-TDNN speaker encoder (192-d vectors only).

Legacy KZA loaded two ECAPA instances (``main.py`` and the ambient transcriber).
The worker keeps exactly one and returns only the raw embedding: voice identity is
an acoustic signal, never a person, account, role or permission. Matching against
enrolled voices, thresholds and any authority decision live elsewhere.
"""

import inspect
import logging
import pathlib
import re
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.offline import OfflineViolationError, assert_offline_effective, reject_remote_source
from nexa_ml.slot import ModelSlot

logger = logging.getLogger(__name__)

SPEAKER_SLOT_ID = "speaker"
SPEAKER_EMBEDDING_DIM = 192
MIN_SPEAKER_SECONDS = 0.5
_ANY_URI_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


class EmbeddingShapeError(RuntimeError):
    """The encoder returned a vector of the wrong dimension (maps to ``INTERNAL``)."""


@dataclass(frozen=True)
class SpeakerEmbedding:
    """Raw speaker embedding."""

    vector: np.ndarray
    encoder_model_key: str
    latency_ms: int


def resolve_ecapa_source(source: str, savedir: str, *, offline: bool) -> str:
    """Pick the directory SpeechBrain loads from, failing closed when offline.

    Offline: URI sources are rejected and the files must already exist locally,
    either in ``source`` (a directory) or in the explicit ``savedir`` cache. The
    returned directory is used as ``source`` *and* as ``pretrained_path`` so the
    hyperparams never point back at the Hub.

    Raises:
        OfflineViolationError: Remote source or online Hugging Face client.
        FileNotFoundError: No local ``hyperparams.yaml`` + ``embedding_model.ckpt``.
    """
    if not offline:
        return source
    reject_remote_source(source)
    assert_offline_effective()
    for candidate in (pathlib.Path(source), pathlib.Path(savedir)):
        if (candidate / "hyperparams.yaml").is_file() and (candidate / "embedding_model.ckpt").is_file():
            reject_remote_manifest(candidate / "hyperparams.yaml")
            return str(candidate)
    raise FileNotFoundError("ECAPA files are not cached locally (offline mode)")


def reject_remote_manifest(hparams: pathlib.Path) -> None:
    """Fail closed if ``hyperparams.yaml`` references any URI (``http://``, ``s3://``...).

    Raises:
        OfflineViolationError: A remote reference is present.
    """
    if _ANY_URI_RE.search(hparams.read_text(encoding="utf-8", errors="replace")):
        raise OfflineViolationError("hyperparams.yaml references a remote URI (offline mode)")


def from_hparams_kwargs(
    local: str, savedir: str, device: str, *, offline: bool, fetching: Any
) -> dict[str, Any]:
    """Build ``EncoderClassifier.from_hparams`` kwargs.

    Offline: ``pretrained_path`` is pinned to the local dir, files are used in place
    (``LocalStrategy.NO_LINK``) and SpeechBrain's own fetcher is told not to use the
    network (``FetchConfig(allow_network=False)``).

    Args:
        local: Local model directory.
        savedir: Explicit SpeechBrain savedir.
        device: Torch device.
        offline: Enforce zero network.
        fetching: ``speechbrain.utils.fetching`` (injectable for tests).

    Raises:
        OfflineViolationError: The installed SpeechBrain cannot disable network fetching.
    """
    kwargs: dict[str, Any] = {"source": local, "savedir": savedir, "run_opts": {"device": device}}
    if not offline:
        return kwargs
    fetch_config_cls = getattr(fetching, "FetchConfig", None)
    local_strategy = getattr(fetching, "LocalStrategy", None)
    if fetch_config_cls is None or "allow_network" not in inspect.signature(fetch_config_cls).parameters:
        raise OfflineViolationError("installed speechbrain has no FetchConfig(allow_network=...)")
    kwargs["overrides"] = {"pretrained_path": local}
    kwargs["fetch_config"] = fetch_config_cls(allow_network=False)
    if local_strategy is not None and hasattr(local_strategy, "NO_LINK"):
        kwargs["local_strategy"] = local_strategy.NO_LINK
    return kwargs


def _load_speechbrain() -> Any:
    """Import SpeechBrain lazily (torchaudio>=2.9 shim included)."""
    import torchaudio  # noqa: PLC0415 - heavy, lazy by design

    if not hasattr(torchaudio, "list_audio_backends"):  # speechbrain 1.x on torchaudio>=2.9
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]
    from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415
    from speechbrain.utils import fetching  # noqa: PLC0415

    return types.SimpleNamespace(EncoderClassifier=EncoderClassifier, fetching=fetching)


class EcapaEncoder:
    """Real SpeechBrain ECAPA adapter exposing ``embed(audio) -> np.ndarray``.

    Constructed only on the slot thread by :func:`ecapa_loader`.
    """

    def __init__(
        self, source: str, device: str, savedir: str, *, offline: bool = True, speechbrain_api: Any = None
    ) -> None:
        local = resolve_ecapa_source(source, savedir, offline=offline)  # fail closed before any import
        api = speechbrain_api if speechbrain_api is not None else _load_speechbrain()
        kwargs = from_hparams_kwargs(local, savedir, device, offline=offline, fetching=api.fetching)
        self._classifier = api.EncoderClassifier.from_hparams(**kwargs)

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
    return str(root / "speechbrain" / source.replace("/", "--").replace(":", "_"))


def ecapa_loader(source: str, device: str, models_dir: str | None, *, offline: bool) -> Callable[[], Any]:
    """Return a zero-arg factory building the single :class:`EcapaEncoder`."""
    savedir = default_savedir(source, models_dir)
    return lambda: EcapaEncoder(source, device, savedir, offline=offline)


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
