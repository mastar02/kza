"""Context Compactor — turns into a summary using a background LLM call.

Plan #2 OpenClaw — see docs/superpowers/specs/2026-04-28-openclaw-context-compaction-design.md

Identifier policy strict by construction: HA entity_ids (light.X, scene.Y, etc.)
NEVER pass through the LLM. They are extracted from `preserved_entities` and
surfaced verbatim in `CompactionResult.preserved_ids`.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum

from src.orchestrator.context_manager import ConversationTurn

logger = logging.getLogger(__name__)


# System prompt for the 30B compactor. Constraints:
# - 2-4 sentences: empirically caps token cost at <200 tokens
# - Third person: avoids confusing pronoun chains in summaries
# - NO entity_ids: HA IDs are preserved separately (preserved_ids)
#   so the LLM cannot hallucinate a non-existent ID (anti-rot).
# - JSON envelope: deterministic parsing; _parse_summary tolerates
#   extra prose around the JSON object but rejects unparseable output.
COMPACTOR_SYSTEM_PROMPT = (
    "Sos un compactador de contexto conversacional. Recibís N turnos de "
    "diálogo entre un usuario y un asistente de hogar. Tu tarea: producir "
    "un resumen en 2-4 oraciones en español que capture (a) preferencias "
    "estables del usuario, (b) decisiones tomadas, (c) entidades del hogar "
    "referenciadas en lenguaje natural (NO uses identificadores técnicos). "
    "Usá tercera persona. NO menciones IDs tipo light.X, scene.Y, area.Z — "
    "esos se preservan aparte.\n\n"
    'Devolvé JSON: {"summary": "..."}'
)


class CompactionErrorKind(Enum):
    EMPTY_INPUT = "empty_input"
    TIMEOUT = "timeout"
    REASONER_FAILED = "reasoner_failed"
    PARSE_FAILED = "parse_failed"


class CompactionError(Exception):
    """Raised when compaction cannot produce a usable summary.

    Carries `kind` so callers can apply per-class retry policy
    (e.g., TIMEOUT → eventually retry; PARSE_FAILED → backoff).
    """

    def __init__(
        self,
        kind: CompactionErrorKind,
        message: str,
        *,
        original: Exception | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.original = original


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    preserved_ids: tuple[str, ...]
    compacted_turns_count: int
    model: str
    latency_ms: float

    def __post_init__(self):
        if self.compacted_turns_count <= 0:
            raise ValueError(
                f"compacted_turns_count must be positive, got {self.compacted_turns_count}"
            )
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if not self.summary.strip():
            raise ValueError("summary must be non-empty")


class Compactor:
    """Compacts a list of conversation turns into a short summary.

    The LLM call is async and may take seconds; callers should run this in a
    background task. Identifier policy strict: HA entity_ids are NEVER passed
    to the model — they are extracted from `preserved_entities` and surfaced
    verbatim in the result.
    """

    def __init__(
        self,
        reasoner,  # HttpReasoner-like with async complete(prompt, max_tokens, temperature)
        max_summary_tokens: int = 200,
        timeout_s: float = 30.0,
    ):
        self.reasoner = reasoner
        self.max_summary_tokens = max_summary_tokens
        self.timeout_s = timeout_s

    async def compact(
        self,
        turns: list[ConversationTurn],
        preserved_entities: list[str],
    ) -> CompactionResult:
        """Compact conversation turns into a summary.

        Args:
            turns: List of conversation turns to compact.
            preserved_entities: List of HA entity_ids to preserve verbatim.

        Returns:
            CompactionResult with summary, preserved IDs, and latency.

        Raises:
            CompactionError: If turns is empty, timeout occurs, reasoner fails,
                or LLM output is unparseable / produces invalid summary
                (PARSE_FAILED).
        """
        if not turns:
            raise CompactionError(
                CompactionErrorKind.EMPTY_INPUT, "No turns to compact"
            )

        prompt = self._build_prompt(turns)
        start = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                self.reasoner.complete(
                    prompt=prompt,
                    max_tokens=self.max_summary_tokens,
                    temperature=0.3,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise CompactionError(
                CompactionErrorKind.TIMEOUT,
                f"Compactor timeout after {self.timeout_s}s",
                original=e,
            ) from e
        except Exception as e:
            raise CompactionError(
                CompactionErrorKind.REASONER_FAILED,
                f"Compactor reasoner error: {e}",
                original=e,
            ) from e
        latency_ms = (time.perf_counter() - start) * 1000

        summary = self._parse_summary(text)
        preserved_ids = tuple(sorted(set(preserved_entities)))
        model = getattr(self.reasoner, "_resolved_model", None) or "unknown"

        logger.info(
            f"[Compactor] turns={len(turns)} summary_chars={len(summary)} "
            f"preserved_ids={len(preserved_ids)} latency={latency_ms:.0f}ms"
        )

        try:
            return CompactionResult(
                summary=summary,
                preserved_ids=preserved_ids,
                compacted_turns_count=len(turns),
                model=model,
                latency_ms=latency_ms,
            )
        except ValueError as e:
            raise CompactionError(
                CompactionErrorKind.PARSE_FAILED,
                f"Compactor produced invalid result: {e}",
                original=e,
            ) from e

    def _build_prompt(self, turns: list[ConversationTurn]) -> str:
        """Build the LLM prompt from turn contents only.

        Note: turn.entities (HA entity_ids) are intentionally NOT included.
        Identifier policy strict — IDs are preserved out-of-band in
        CompactionResult.preserved_ids.
        """
        lines = [COMPACTOR_SYSTEM_PROMPT, "", "Turnos a compactar:"]
        for i, turn in enumerate(turns, start=1):
            lines.append(f"{i}. [{turn.role}] {turn.content}")
        lines.append("")
        lines.append("Resumen JSON:")
        return "\n".join(lines)

    def _parse_summary(self, text: str) -> str:
        """Extract summary from LLM response (JSON or substring fallback).

        Raises:
            CompactionError(PARSE_FAILED): If no parseable JSON summary found.
        """
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("summary"), str):
                return data["summary"].strip()
        except json.JSONDecodeError:
            pass
        # Fallback: try to extract first JSON object substring
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if isinstance(data, dict) and isinstance(data.get("summary"), str):
                return data["summary"].strip()
        except (ValueError, json.JSONDecodeError):
            pass
        # Final: refuse to propagate LLM garbage into long-term memory.
        logger.warning(
            f"[Compactor] JSON parse failed, refusing to persist garbage ({len(text)} chars)"
        )
        raise CompactionError(
            CompactionErrorKind.PARSE_FAILED,
            "Compactor produced no parseable JSON summary",
        )
