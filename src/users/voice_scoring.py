"""U3b-4b-1: scores from stored embeddings, computed inside the ledger transaction.

No inference happens here. The utterance's embedding is the one the voice
registry already computed on the registered encoder (speaker_id.device) and
stored as a fingerprint; every live confirmed profile's embedding is the
stored mean of its reviewed fingerprints. The score is the raw cosine of the
two stored unit vectors, as in VoiceLedger.put (never SpeakerIdentifier's
(s+1)/2 similarity).
"""

import logging
import math
import sqlite3

import numpy as np

from src.memory.fact_ledger import FactLedger, MemorySource
from src.memory.voice_attribution import ProfileScore, ScoringSnapshot
from src.users.voice_ledger import fingerprint_identity

logger = logging.getLogger(__name__)


class FingerprintUnavailable(LookupError):
    """The capture's fingerprint is missing, expired, foreign or unreadable: no snapshot."""


class FingerprintScorer:
    """``scorer(db)`` for FactLedger.attribute_voice_proposal and verify_voice_consent.

    Reads, in the caller's transaction, the fingerprint whose deterministic id
    is derived from (its model key, the capture's sources), requires those
    sources to be live, and scores it against every live confirmed profile,
    linked or not. A profile of another model key or dimension, or without a
    valid embedding, makes the snapshot incomplete (fail closed). Scores are
    ordered by profile id, so the result does not depend on row order.
    """

    def __init__(self, ledger: FactLedger, fingerprint_id: str, sources: tuple[MemorySource, ...]):
        if type(sources) is not tuple or len(sources) != 1 or type(sources[0]) is not MemorySource:
            raise ValueError("voice_scoring_sources_invalid")
        self._ledger = ledger
        self.fingerprint_id = fingerprint_id
        self._sources = sources

    def __call__(self, db: sqlite3.Connection) -> ScoringSnapshot:
        row = db.execute("SELECT model_key, dimensions, embedding, duration_s, expires_at FROM voice_fingerprints "
                         "WHERE id=?", (self.fingerprint_id,)).fetchone()
        if row is None:
            raise FingerprintUnavailable("voice_fingerprint_unavailable")
        model_key, dimensions, raw, duration_s, expires_at = row
        if (type(model_key) is not str or fingerprint_identity(model_key, self._sources) != self.fingerprint_id
                or not self._ledger._sources_live(db, self._sources) or not self._ledger.clock() < expires_at):
            raise FingerprintUnavailable("voice_fingerprint_unavailable")
        utterance = _unit(raw, dimensions)
        if utterance is None or type(duration_s) not in (int, float) or not math.isfinite(duration_s):
            raise FingerprintUnavailable("voice_fingerprint_unavailable")
        scores, complete = [], True
        for identity, revision, encoded, confirmed_at, confirmed_by, deleted_at, key, size, embedding in db.execute(
                "SELECT id, revision, sources_json, confirmed_at, confirmed_by, deleted_at, model_key, dimensions, "
                "embedding FROM voice_profiles WHERE deleted_at IS NULL ORDER BY id"):
            if not self._ledger.voice_profile_live(encoded, confirmed_at, confirmed_by, deleted_at):
                continue
            profile = _unit(embedding, size) if key == model_key and size == dimensions else None
            score = None if profile is None else float(np.dot(utterance, profile))
            if score is None or not math.isfinite(score):
                complete = False
                continue
            scores.append(ProfileScore(identity, revision, max(-1.0, min(1.0, score))))
        return ScoringSnapshot(model_key, tuple(scores), complete, float(duration_s))


class UnavailableScorer:
    """No fingerprint (dropped, failed or later than embedding_wait_s): every read is unavailable."""

    def __call__(self, db: sqlite3.Connection) -> ScoringSnapshot:
        raise FingerprintUnavailable("voice_fingerprint_unavailable")


def _unit(raw: object, dimensions: object) -> np.ndarray | None:
    """A stored little-endian float32 unit vector of the declared size, or None."""
    if type(raw) is not bytes or type(dimensions) is not int or not 16 <= dimensions <= 1024 or len(raw) != 4 * dimensions:
        return None
    vector = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        return None
    return vector
