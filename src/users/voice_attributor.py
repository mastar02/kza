"""U3b-4b-2: attribution of spoken proposals and evaluation of verification phrases, after the reply.

Both run as tracked background tasks, at most MAX_TASKS at a time (overflow
means no owner, or no evaluation). Each waits at most embedding_wait_s for
the capture's fingerprint (VoiceRegistry.await_fingerprint), then runs the
ledger step through PrivacyServices.acall, off the event loop. Errors are
logged by type only and counted per closed reason; no text, person or
profile id is logged. Shutdown cancels the tasks, so nothing is written.

The voice stage is absent unless memory.voice_proposals.enabled is true
(build_voice_attribution returns None), which leaves the router, the
registry and every reply exactly as without this module.
"""

import asyncio
from collections import Counter
from contextvars import Context
from dataclasses import dataclass
import logging
from typing import Callable

from src.memory.fact_ledger import MemoryOrigin, MemorySource
from src.memory.voice_attribution import (
    AttributionReason, VerificationWindow, VoiceAttributionCapability, VoiceProposalSettings,
)
from src.memory.voice_phrase import near_phrase, normalize_utterance
from src.users.voice_scoring import FingerprintScorer, UnavailableScorer

logger = logging.getLogger(__name__)

MAX_TASKS = 4


@dataclass(frozen=True)
class VoiceStageCounts:
    """Per-reason counters only: no text, person or profile ids."""

    attribution: dict
    verification: dict
    overflow: int


class VoiceProposalAttributor:
    """Constructor-injected attributor and verifier; the router holds it only when the stage is on."""

    def __init__(self, privacy, registry, capability: VoiceAttributionCapability,
                 retired: Callable[[frozenset[str]], bool | None], active: Callable[[str], bool | None]):
        if type(capability) is not VoiceAttributionCapability:
            raise PermissionError("memory_voice_capability_denied")
        self._privacy = privacy
        self._ledger = privacy.ledger
        self._registry = registry
        self._capability = capability
        self._retired = retired
        self._active = active
        self._settings: VoiceProposalSettings = privacy.ledger.voice_settings
        self._tasks: set[asyncio.Task] = set()
        self._attribution: Counter = Counter()
        self._verification: Counter = Counter()
        self._overflow = 0
        self._closed = False

    @property
    def windows(self):
        """The ledger's in-memory verification windows (opened by the person's route)."""
        return self._ledger.voice_windows

    def counts(self) -> VoiceStageCounts:
        """Counters by closed reason."""
        return VoiceStageCounts(dict(self._attribution), dict(self._verification), self._overflow)

    @staticmethod
    def channel_source(room_id: object, textual_wake: object, sources: tuple) -> MemorySource | None:
        """The one source of a multi-room room-microphone capture, else None (U3b-4b-2 channels).

        Only a CommandEvent of the multi-room loop qualifies: a non-empty room,
        no textual wake, and exactly one VOICE source. The single-mic loop
        (room ''), raw arrays (room None), the textual wake and ambient
        sources never do.
        """
        if (type(room_id) is not str or not room_id or textual_wake is not False or type(sources) is not tuple
                or len(sources) != 1 or type(sources[0]) is not MemorySource or sources[0].origin is not MemoryOrigin.VOICE):
            return None
        return sources[0]

    def match_phrase(self, text: str, room_id: str, wake_words: tuple[str, ...]) -> VerificationWindow | None:
        """Consume the open window of this room whose phrase is the normalised transcript (on the loop)."""
        phrase = normalize_utterance(text, wake_words)
        if not phrase:
            return None
        return self.windows.match(room_id, phrase)

    def screen(self, text: str, room_id: object, textual_wake: object, sources: tuple,
               wake_words: tuple[str, ...]) -> tuple[VerificationWindow | None, MemorySource | None] | None:
        """Decide on the loop whether a turn is a phrase turn (U3b-4b-2, review L2).

        None (the turn continues untouched) whenever no window is open in any
        room. Otherwise ``(window, source)`` when a qualifying capture speaks
        its room's phrase exactly (the window is consumed), and ``(None,
        None)`` for a near miss: at most eight words matching three of the
        four words of any open phrase, by sameness, one edit or sound, from
        any room, the textual wake or a multi-source capture. A near miss gets
        the same fixed reply, consumes nothing, records no attempt and never
        reaches the gate, the ASR log or the model.
        """
        phrases = self.windows.live_phrases()
        if not phrases:
            return None
        heard = normalize_utterance(text, wake_words)
        if not heard:
            return None
        source = self.channel_source(room_id, textual_wake, sources)
        window = self.windows.match(room_id, heard) if source is not None else None
        if window is not None:
            self._registry.redact_capture(source.capture_id)
            return window, source
        if near_phrase(heard, phrases):
            self._verification["near_miss"] += 1
            return None, None
        return None

    def _spawn(self, coroutine) -> bool:
        if self._closed or len(self._tasks) >= MAX_TASKS:
            coroutine.close()
            self._overflow += 1
            return False
        task = asyncio.create_task(coroutine, context=Context())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    def schedule(self, fact_id: str, source: MemorySource) -> bool:
        """After the reply: try to make an unowned voice proposal its verified speaker's (reject-only)."""
        if type(fact_id) is not str or type(source) is not MemorySource:
            return False
        return self._spawn(self._attribute(fact_id, source))

    def verify(self, window: VerificationWindow, source: MemorySource) -> bool:
        """After the fixed reply: evaluate the consumed window against the utterance."""
        if type(window) is not VerificationWindow or type(source) is not MemorySource:
            return False
        return self._spawn(self._verify(window, source))

    async def _attribute(self, fact_id: str, source: MemorySource) -> None:
        try:
            wait = await self._registry.await_fingerprint(source.capture_id, self._settings.embedding_wait_s)
            if wait.reason is not None:
                self._attribution[wait.reason] += 1
                return
            scorer = FingerprintScorer(self._ledger, wait.fingerprint_id, (source,))
            decision = await self._privacy.acall(self._ledger.attribute_voice_proposal, fact_id, scorer,
                                                 self._capability, retired=self._retired, active=self._active)
            self._attribution[decision.reason.value] += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._attribution[AttributionReason.ERROR.value] += 1
            logger.warning("[VoiceAttribution] attribution failed: %s", type(error).__name__)

    async def _verify(self, window: VerificationWindow, source: MemorySource) -> None:
        late = None
        try:
            wait = await self._registry.await_fingerprint(source.capture_id, self._settings.embedding_wait_s)
            late = wait.completion if wait.reason == "late_embedding" else None
            scorer = (UnavailableScorer() if wait.fingerprint_id is None
                      else FingerprintScorer(self._ledger, wait.fingerprint_id, (source,)))
            result = await self._privacy.acall(self._ledger.verify_voice_consent, window.binding, scorer,
                                               self._capability, retired=self._retired, active=self._active,
                                               review_fingerprint=wait.fingerprint_id)
            self._verification[result.reason.value if wait.reason is None else wait.reason] += 1
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._verification[AttributionReason.ERROR.value] += 1
            logger.warning("[VoiceAttribution] verification failed: %s", type(error).__name__)
        finally:
            if late is not None:
                late.add_done_callback(self._discard_late_clip, context=Context())

    def _discard_late_clip(self, completion: asyncio.Future) -> None:
        """The fingerprint (and its clip with the phrase) landed after evaluation: delete the clip.

        Also after shutdown (review W20): a clip that slipped past the capture's
        redaction is deleted whenever its fingerprint lands.
        """
        if completion.cancelled() or type(completion.result()) is not str:
            return
        identity = completion.result()

        async def discard() -> None:
            try:
                await self._privacy.acall(self._ledger.discard_voice_review, identity, self._capability)
            except Exception as error:
                logger.warning("[VoiceAttribution] clip discard failed: %s", type(error).__name__)

        task = asyncio.create_task(discard(), context=Context())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """Cancel every task: an attribution not yet written is never written."""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


PHRASE_SAFE_WAKE_ENGINES = frozenset({"openwakeword"})


def build_voice_attribution(privacy, registry, authority, *, shutdown, wake_engine: object = "openwakeword",
                            home_names: Callable[[], tuple[str, ...]] | None = None) -> VoiceProposalAttributor | None:
    """Composition root for the voice stage (U3b-4b-2); None leaves every path as without it.

    The stage exists only when memory.voice_proposals.enabled is true (read
    by PrivacyServices with defaults, never written) and a voice registry
    exists. The system capability is minted here and nowhere else (AST pin).
    The whisper wake engine logs every wake transcript, so with it the stage
    stays absent (review E). ``home_names`` gives this home's device and area
    names; a drawn phrase word near one of them is redrawn (review W).
    """
    settings = privacy.ledger.voice_settings
    if type(settings) is not VoiceProposalSettings or settings.enabled is not True or registry is None:
        return None
    if wake_engine not in PHRASE_SAFE_WAKE_ENGINES:
        logger.warning("[VoiceAttribution] voice stage off: the wake engine is not phrase-safe")
        return None
    retired = getattr(authority, "memory_owner_retired", None) or (lambda identities: None)
    active = getattr(authority, "memory_owner_active", None) or (lambda person: None)
    attributor = VoiceProposalAttributor(privacy, registry, VoiceAttributionCapability("system:voice-attribution"),
                                         retired, active)
    privacy.ledger.voice_windows.set_reserved(home_names)
    registry.enable_completions()
    shutdown.push_async_callback(attributor.aclose)
    return attributor
