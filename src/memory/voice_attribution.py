"""Closed values of the voice-attribution tables (U3b-4a-0) and the pure gate (U3b-4a-1).

The tables carry no CHECK constraints: every value is validated against these
enums when the ledger opens and by every writer (U3b W12, 1a-F4). All values,
including those used only by later units, are enumerated now so that no table
rebuild is ever needed for them.
"""

import logging
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from src.memory.voice_phrase import draw_phrase, normalize_utterance, word_near

logger = logging.getLogger(__name__)


class VoiceConsentState(StrEnum):
    """A person's voice-attribution consent (design u3b-4 v4 consent_semantics)."""

    PENDING_VERIFICATION = "pending_verification"
    ACTIVE = "active"
    REVOKED = "revoked"
    INVALIDATED = "invalidated"
    VERIFICATION_LOCKED = "verification_locked"


class VoiceVerificationOutcome(StrEnum):
    """One verification-attempt row; only 'failed' counts toward the lock."""

    WINDOW_OPENED = "window_opened"
    PASSED = "passed"
    FAILED = "failed"
    FAILED_FOREIGN = "failed_foreign"
    UNAVAILABLE = "unavailable"


class VoiceAttributionEventKind(StrEnum):
    """Content-free misattribution signal."""

    NOT_MINE = "not_mine"


class VoiceSuspensionReason(StrEnum):
    """Why voice attribution is suspended for one profile."""

    NOT_MINE_THRESHOLD = "not_mine_threshold"


# --- U3b-4a-1: settings, gate and capability (pure, no I/O) -----------------------------------


class AttributionReason(StrEnum):
    """Closed reason of one gate decision; only per-reason counters are kept."""

    OWNER = "owner"
    DISABLED = "disabled"
    NO_FINGERPRINT = "no_fingerprint"
    LATE_EMBEDDING = "late_embedding"
    ORIGIN_NOT_VOICE = "origin_not_voice"
    MULTI_SOURCE = "multi_source"
    INCOMPLETE_SCORING_SET = "incomplete_scoring_set"
    TOO_FEW_PROFILES = "too_few_profiles"
    TIE = "tie"
    TOP_NOT_GRANTABLE = "top_not_grantable"
    TOP_FOREIGN = "top_foreign"
    BELOW_SCORE = "below_score"
    BELOW_MARGIN = "below_margin"
    SHORT_SPEECH = "short_speech"
    CONSENT_NOT_ACTIVE = "consent_not_active"
    CONSENT_STALE = "consent_stale"
    SUSPENDED = "suspended"
    PERSON_UNAVAILABLE = "person_unavailable"
    FACT_CHANGED = "fact_changed"
    ERROR = "error"


_COUNTED = frozenset({AttributionReason.TOP_NOT_GRANTABLE, AttributionReason.BELOW_SCORE,
                      AttributionReason.BELOW_MARGIN, AttributionReason.SHORT_SPEECH, AttributionReason.TIE})


def verification_outcome(reason: AttributionReason) -> VoiceVerificationOutcome:
    """Classify a verification decision (design v4 gate.verification_classification_R8).

    Only an owner decision passes. top_foreign is recorded but never counted
    toward the lock; every reason the person cannot influence is 'unavailable'.
    """
    if type(reason) is not AttributionReason:
        raise ValueError("voice_attribution_reason_unknown")
    if reason is AttributionReason.OWNER:
        return VoiceVerificationOutcome.PASSED
    if reason in _COUNTED:
        return VoiceVerificationOutcome.FAILED
    if reason is AttributionReason.TOP_FOREIGN:
        return VoiceVerificationOutcome.FAILED_FOREIGN
    return VoiceVerificationOutcome.UNAVAILABLE


@dataclass(frozen=True)
class VoiceProposalSettings:
    """Read-only view of ``memory.voice_proposals`` (G2).

    A missing section, a missing key or a non-bool ``enabled`` means off.
    Values are copied out of the config; nothing is ever written back. The
    thresholds are placeholders until field evidence (pending_4c).
    """

    enabled: bool = False
    # PLACEHOLDERS, not calibrated: U3b-4c sets them from the P7 field evidence
    # (real XVF3800 captures, the production encoder). Do not tune them locally.
    min_score: float = 0.80  # placeholder
    min_margin: float = 0.10  # placeholder
    min_speech_s: float = 1.5  # placeholder
    embedding_wait_s: float = 2.0  # placeholder
    min_scoring_profiles: int = 2  # placeholder (Q4 A)
    verification_max_failures: int = 3
    verification_lock_window_s: float = 86400.0
    verification_windows_per_day: int = 6
    not_mine_suspend_count: int = 2
    not_mine_window_days: float = 30.0

    _RANGES = {
        "min_score": (float, -1.0, 1.0), "min_margin": (float, 0.0, 2.0), "min_speech_s": (float, 0.0, 30.0),
        "embedding_wait_s": (float, 0.0, 30.0), "min_scoring_profiles": (int, 2, 256),
        "verification_max_failures": (int, 1, 100), "verification_lock_window_s": (float, 3600.0, 2592000.0),
        "verification_windows_per_day": (int, 1, 100), "not_mine_suspend_count": (int, 1, 100),
        "not_mine_window_days": (float, 1.0, 365.0),
    }

    @classmethod
    def from_config(cls, config: object) -> "VoiceProposalSettings":
        """Copy scalars from ``config['memory']['voice_proposals']``; never mutate the config.

        Raises ValueError('voice_proposals_config_invalid') for a present value
        of the wrong type, non-finite or out of range; the caller then leaves
        the voice stage absent.
        """
        memory = config.get("memory") if isinstance(config, Mapping) else None
        section = memory.get("voice_proposals") if isinstance(memory, Mapping) else None
        if section is None:
            return cls()
        if not isinstance(section, Mapping):
            raise ValueError("voice_proposals_config_invalid")
        values: dict = {"enabled": section.get("enabled") is True}
        for name, (kind, low, high) in cls._RANGES.items():
            if name not in section:
                continue
            value = section[name]
            if kind is int:
                valid = type(value) is int and low <= value <= high
            else:
                valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
                value = float(value) if valid else value
            if not valid:
                raise ValueError("voice_proposals_config_invalid")
            values[name] = value
        return cls(**values)


def bucket(value: float) -> int:
    """floor(value * 20) clamped to [-20, 20]; non-finite values are refused."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("voice_attribution_bucket_invalid")
    return max(-20, min(20, math.floor(value * 20)))


@dataclass(frozen=True)
class ProfileScore:
    """Raw cosine of one live profile against the capture, with the revision it was read at."""

    profile_id: str
    revision: int
    score: float


@dataclass(frozen=True)
class ScoringSnapshot:
    """Scores against every live confirmed profile (the scoring set), linked or not."""

    model_key: str
    scores: tuple[ProfileScore, ...]
    complete: bool
    speech_s: float

    def valid(self) -> bool:
        """Shape check; a malformed snapshot is an error, never a decision."""
        return (type(self.model_key) is str and bool(self.model_key) and type(self.scores) is tuple
                and type(self.complete) is bool and type(self.speech_s) in (int, float)
                and math.isfinite(self.speech_s) and self.speech_s >= 0
                and all(type(item) is ProfileScore and type(item.profile_id) is str and item.profile_id
                        and type(item.revision) is int and item.revision >= 1
                        and type(item.score) in (int, float) and math.isfinite(item.score)
                        and -1.0 <= item.score <= 1.0 for item in self.scores)
                and len({item.profile_id for item in self.scores}) == len(self.scores))


@dataclass(frozen=True)
class GateDecision:
    """Content-free result: a reason and, on owner, the granted profile and buckets."""

    reason: AttributionReason
    profile_id: str | None = None
    score_bucket: int | None = None
    margin_bucket: int | None = None

    @property
    def owner(self) -> bool:
        """True only for an owner decision."""
        return self.reason is AttributionReason.OWNER


def evaluate_gate(snapshot: ScoringSnapshot, settings: VoiceProposalSettings,
                  grantable: Callable[[str], AttributionReason | None]) -> GateDecision:
    """Reject-only comparison: an owner only when the unique top is grantable and clears every threshold.

    ``grantable(top_profile_id)`` returns None when the top profile may be
    granted, otherwise the refusal reason (top_foreign, top_not_grantable,
    consent_not_active, consent_stale). Never falls back to the second best.
    """
    if type(snapshot) is not ScoringSnapshot or not snapshot.valid():
        return GateDecision(AttributionReason.ERROR)
    if not snapshot.complete:
        return GateDecision(AttributionReason.INCOMPLETE_SCORING_SET)
    if len(snapshot.scores) < settings.min_scoring_profiles:
        return GateDecision(AttributionReason.TOO_FEW_PROFILES)
    ranked = sorted(snapshot.scores, key=lambda item: item.score, reverse=True)
    top, second = ranked[0], ranked[1]
    if top.score == second.score:
        return GateDecision(AttributionReason.TIE)
    refusal = grantable(top.profile_id)
    if refusal is not None:
        if type(refusal) is not AttributionReason or refusal is AttributionReason.OWNER:
            return GateDecision(AttributionReason.ERROR)
        return GateDecision(refusal)
    if top.score < settings.min_score:
        return GateDecision(AttributionReason.BELOW_SCORE)
    if top.score - second.score < settings.min_margin:
        return GateDecision(AttributionReason.BELOW_MARGIN)
    if snapshot.speech_s < settings.min_speech_s:
        return GateDecision(AttributionReason.SHORT_SPEECH)
    return GateDecision(AttributionReason.OWNER, top.profile_id, bucket(top.score), bucket(top.score - second.score))


@dataclass(frozen=True)
class VoiceAttributionCapability:
    """System capability of the attributor and verifier; not a principal.

    Minted once at the composition root (4b-2). No production code constructs
    it in 4a-1, which an AST pin asserts.
    """

    issuer: str


# --- U3b-4a-2: in-memory verification windows ---------------------------------------------------


@dataclass(frozen=True)
class VerificationWindow:
    """One open window: what it is bound to, the room, the phrase and its deadline.

    The phrase is four words from the fixed list (src/memory/voice_phrase.py,
    U3b-4b-2), single use: a match consumes the window.
    """

    binding: object
    room_id: str
    expires_at: float
    phrase: str | None = None


class VoiceVerificationWindows:
    """Single-use windows kept only in memory: a restart closes them; at most one per person."""

    WINDOW_S = 60.0

    def __init__(self, clock: Callable[[], float], *, draw: Callable[[], str] = draw_phrase):
        self._clock = clock
        self._draw = draw
        self._reserved: Callable[[], Iterable[str]] | None = None
        self._lock = threading.Lock()
        self._windows: dict[str, VerificationWindow] = {}

    def set_reserved(self, names: Callable[[], Iterable[str]] | None) -> None:
        """Install the source of this home's device and area names (U3b-4b review W).

        ``open`` redraws a phrase while any of its words is the same as, one
        edit from or sounds like a word of those names, so a phrase never reads
        as one of this home's devices or rooms.
        """
        self._reserved = names

    def _reserved_words(self) -> frozenset[str]:
        if self._reserved is None:
            return frozenset()
        try:
            return frozenset(word for name in self._reserved()
                             for word in normalize_utterance(name, ("",)).split() if word)
        except Exception as e:  # noqa: BLE001 - the static exclusion still holds
            logger.warning(f"[VoiceWindows] reserved names unavailable ({type(e).__name__})")
            return frozenset()

    def open(self, binding: object, room_id: str) -> VerificationWindow:
        """Replace the person's window with a fresh one bound to ``binding`` and a new phrase.

        The phrase is redrawn while it equals another live window's phrase in
        the same room, so a match always names exactly one window, and while a
        word of it sits near a home device or area name.
        """
        reserved = self._reserved_words()
        with self._lock:
            self._windows.pop(binding.person_id, None)
            now = self._clock()
            taken = {window.phrase for window in self._windows.values()
                     if window.room_id == room_id and window.expires_at > now}
            collisions = 0
            for _ in range(8):
                phrase = self._draw()
                if any(word_near(word, name) for word in phrase.split() for name in reserved):
                    collisions += 1
                    continue
                if phrase not in taken:
                    break
            else:
                raise RuntimeError("memory_voice_phrase_unavailable")
            window = VerificationWindow(binding, room_id, now + self.WINDOW_S, phrase)
            self._windows[binding.person_id] = window
        if collisions:
            logger.info(f"[VoiceWindows] phrase redrawn: home-name collisions={collisions}")
        return window

    def match(self, room_id: str, phrase: str) -> VerificationWindow | None:
        """Consume the live window of ``room_id`` whose phrase equals ``phrase`` (single use).

        ``phrase`` is the normalised transcript. A window of another room, an
        expired window or a different phrase never matches; a consumed window
        cannot match again.
        """
        if type(room_id) is not str or not room_id or type(phrase) is not str or not phrase:
            return None
        with self._lock:
            now = self._clock()
            for person, window in list(self._windows.items()):
                if window.expires_at <= now:
                    del self._windows[person]
                elif window.room_id == room_id and window.phrase == phrase:
                    del self._windows[person]
                    return window
        return None

    def live_phrases(self) -> tuple[str, ...]:
        """The phrases of every live window in any room, without consuming one; empty when none is open."""
        with self._lock:
            if not self._windows:
                return ()
            now = self._clock()
            return tuple(window.phrase for window in self._windows.values() if window.expires_at > now)

    def current(self, person_id: str) -> VerificationWindow | None:
        """The person's live window; an expired one is dropped."""
        with self._lock:
            window = self._windows.get(person_id)
            if window is not None and window.expires_at <= self._clock():
                del self._windows[person_id]
                window = None
            return window

    def close(self, person_id: str) -> None:
        """Close the person's window (revocation, invalidation, retirement)."""
        with self._lock:
            self._windows.pop(person_id, None)

    def close_profile(self, profile_id: str) -> None:
        """Close every window bound to ``profile_id`` (its link, revision or review changed)."""
        with self._lock:
            for person in [person for person, window in self._windows.items()
                           if getattr(window.binding, "profile_id", None) == profile_id]:
                del self._windows[person]

    def clear(self) -> None:
        """Close every window (the voice switch was turned off)."""
        with self._lock:
            self._windows.clear()
