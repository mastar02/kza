"""Medición de calidad del command path STT.

Traduce el outcome de una captura post-wake a un evento del EventLogger
existente (data/events.db). NO toca el schema: usa entity_id="asr_quality:<room>"
y action="<outcome>:<reason>" para poder hacer luego:
    SELECT action, count(*) FROM events
    WHERE entity_id LIKE 'asr_quality:%' GROUP BY action;

Fail-open: si event_logger es None o falla, no se rompe el pipeline de voz.
"""
from __future__ import annotations

import logging

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
        )
    except Exception as e:  # nunca romper el pipeline de voz por la métrica
        logger.debug(f"[asr_quality] log falló (ignorado): {e}")
