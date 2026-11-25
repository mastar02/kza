"""BGE-M3 text embeddings on CPU (1024-d), run through ``asyncio.to_thread``."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexa_ml.slot import InvalidInputError, ModelSlot

logger = logging.getLogger(__name__)

EMBED_SLOT_ID = "embed"
TEXT_EMBEDDING_DIM = 1024
MAX_TEXT_CHARS = 8192


class TextEmbeddingShapeError(RuntimeError):
    """The model returned vectors of the wrong shape (maps to ``INTERNAL``)."""


@dataclass(frozen=True)
class TextEmbeddings:
    """Batch of text embeddings."""

    vectors: np.ndarray
    model_id: str
    latency_ms: int


def sentence_transformer_loader(
    model_id: str, device: str, models_dir: str | None, offline: bool
) -> Callable[[], Any]:
    """Return a zero-arg factory creating the ``SentenceTransformer`` (lazy import)."""

    def load() -> Any:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, lazy

        return SentenceTransformer(model_id, device=device, cache_folder=models_dir, local_files_only=offline)

    return load


def validate_texts(texts: list[str], max_texts: int) -> list[str]:
    """Validate an ``EmbedText`` batch (messages never echo the text).

    Raises:
        InvalidInputError: Empty batch, too many texts, empty or oversized text.
    """
    if not texts:
        raise InvalidInputError("texts is empty")
    if len(texts) > max_texts:
        raise InvalidInputError(f"at most {max_texts} texts per call")
    for text in texts:
        if not text.strip():
            raise InvalidInputError("texts contains an empty entry")
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidInputError(f"text longer than {MAX_TEXT_CHARS} characters")
    return texts


def encode_blocking(model: Any, texts: list[str], model_id: str) -> TextEmbeddings:
    """Encode on a worker thread (same call shape as legacy ``chroma_sync``).

    Raises:
        TextEmbeddingShapeError: Output is not ``(len(texts), 1024)``.
    """
    start = time.perf_counter()
    vectors = np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float32)
    if vectors.shape != (len(texts), TEXT_EMBEDDING_DIM):
        raise TextEmbeddingShapeError(f"text embeddings have shape {vectors.shape}")
    return TextEmbeddings(vectors, model_id, int((time.perf_counter() - start) * 1000))


class EmbedSlot:
    """BGE-M3 slot (``ExecutorKind.TO_THREAD``)."""

    def __init__(self, slot: ModelSlot, *, max_texts: int) -> None:
        self.slot = slot
        self.max_texts = max_texts

    async def embed(self, texts: list[str]) -> TextEmbeddings:
        """Validate and encode a batch."""
        batch = validate_texts(list(texts), self.max_texts)
        model_id = self.slot.model_name
        return await self.slot.run(lambda model: encode_blocking(model, batch, model_id))
