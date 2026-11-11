"""Closed values of the voice-attribution tables (U3b-4a-0).

The tables carry no CHECK constraints: every value is validated against these
enums when the ledger opens and by every writer (U3b W12, 1a-F4). All values,
including those used only by later units, are enumerated now so that no table
rebuild is ever needed for them.
"""

from enum import StrEnum


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
