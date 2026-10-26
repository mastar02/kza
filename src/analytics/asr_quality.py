"""Medición de calidad del command path STT.

Traduce el outcome de una captura post-wake al EventLogger privado. Usa
entity_id="asr_quality:<room>" y action="<outcome>:<reason>". Las consultas
del logger aplican procedencia, vencimiento de texto y revocaciones; no se
consulta directamente la tabla legacy.

Fail-open: si event_logger es None o falla, no se rompe el pipeline de voz.
"""
from __future__ import annotations

import asyncio
import logging

from src.analytics.event_logger import EventLogger
from src.memory.fact_ledger import MemorySource

logger = logging.getLogger(__name__)


def log_asr_outcome(
    event_logger,
    room_id: str,
    outcome: str,
    reason: str,
    text: str,
    signals: dict,
    wake_score: float,
    rms: float,
    *, sources: tuple[MemorySource, ...] = (),
) -> None:
    """Registrar el resultado de una captura del command path.

    Embudo real del gate:
        denominator = accepted + gate_rejected

    Args:
        event_logger: EventLogger o None (None = no-op, fail-open).
        room_id: room donde ocurrió la captura.
        outcome: Uno de los siguientes valores reales:
            - ``accepted``: superó el CommandGate y pasa al orchestrator.
            - ``gate_rejected``: rechazado en el gate (incluye STT vacío).
            - ``low_confidence``: el 7B rechazó por confianza baja (post-gate).
            - ``unverified_intent``: el 7B rechazó por intent no verificado (post-gate).
            - ``earcon_fired``: el earcon sonó realmente (emitido junto al outcome
              anterior, no en lugar de él).
        reason: sub-motivo (ej: empty, empty_after_norm, high_compression, ok).
        text: transcripción (se trunca a 60 chars).
        signals: dict de señales STT (compression_ratio, etc.).
        wake_score: score del wake que abrió la captura.
        rms: energía RMS de la captura.
    """
    if event_logger is None:
        return
    try:
        event_logger.log(
            entity_id=f"asr_quality:{room_id}",
            action=f"{outcome}:{reason}",
            trigger_phrase=(text or "")[:60],
            extra_context={
                "wake_score": wake_score,
                "rms": rms,
                "compression_ratio": signals.get("compression_ratio"),
                "no_speech_prob": signals.get("no_speech_prob"),
                "avg_logprob": signals.get("avg_logprob"),
            },
            **({"sources": sources} if sources else {}),
        )
    except Exception as error:
        logger.debug("ASR metric write skipped: %s", type(error).__name__)


async def alog_asr_outcome(event_logger, *args, **kwargs) -> None:
    """Keep diagnostic I/O off-loop and bound its observation during voice."""
    if event_logger is None:
        return
    try:
        if isinstance(event_logger, EventLogger):
            async with asyncio.timeout(event_logger.observation_timeout_s):
                await event_logger.acall(log_asr_outcome, event_logger, *args, **kwargs)
        else:
            await asyncio.to_thread(log_asr_outcome, event_logger, *args, **kwargs)
    except Exception as error:
        logger.debug("ASR metric observation deferred: %s", type(error).__name__)
