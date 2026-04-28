"""Context Compactor — turns into a summary using a background LLM call."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from src.orchestrator.context_manager import ConversationTurn

logger = logging.getLogger(__name__)


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


class CompactionError(Exception):
    """Raised when compaction cannot produce a usable summary."""


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    summary: str
    preserved_ids: list[str]
    compacted_turns_count: int
    model: str = "unknown"
    latency_ms: float = 0.0


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
            CompactionError: If turns is empty, timeout occurs, or reasoner fails.
        """
        if not turns:
            raise CompactionError("No turns to compact")

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
            raise CompactionError(f"Compactor timeout after {self.timeout_s}s") from e
        except Exception as e:
            raise CompactionError(f"Compactor reasoner error: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        summary = self._parse_summary(text)
        preserved_ids = sorted(set(preserved_entities))
        model = getattr(self.reasoner, "_resolved_model", None) or "unknown"

        logger.info(
            f"[Compactor] turns={len(turns)} summary_chars={len(summary)} "
            f"preserved_ids={len(preserved_ids)} latency={latency_ms:.0f}ms"
        )

        return CompactionResult(
            summary=summary,
            preserved_ids=preserved_ids,
            compacted_turns_count=len(turns),
            model=model,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, turns: list[ConversationTurn]) -> str:
        """Build the LLM prompt from conversation turns."""
        lines = [COMPACTOR_SYSTEM_PROMPT, "", "Turnos a compactar:"]
        for i, turn in enumerate(turns, start=1):
            lines.append(f"{i}. [{turn.role}] {turn.content}")
        lines.append("")
        lines.append("Resumen JSON:")
        return "\n".join(lines)

    def _parse_summary(self, text: str) -> str:
        """Extract summary from LLM response (JSON or fallback)."""
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
        # Final fallback: return text literal trimmed
        logger.warning(f"[Compactor] JSON parse failed, using literal text ({len(text)} chars)")
        return text
