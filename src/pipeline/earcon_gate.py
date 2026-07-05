"""Decide si suena el earcon 'no entendí'.

Regla "humano plausible": el earcon NUNCA puede sonarle a la TV. Suena solo si
el wake fue fuerte Y hubo energía real Y el motivo del reject es de
no-comprensión (no de ruido/eco). Lógica pura, un solo lugar testeable.
"""
from __future__ import annotations

# Motivos que indican TV/eco/ruido → JAMÁS earcon (aunque el wake sea fuerte).
_NOISE_PREFIXES = ("noise_phrase", "filler_word", "word_repetition", "missing_wake", "prompt_echo")


def should_play_earcon(reason: str, wake_score: float, rms: float, cfg: dict) -> bool:
    """True si corresponde reproducir el earcon para este reject.

    Args:
        reason: AcceptanceDecision.reason o el intent de reject del router
            (ej: 'empty', "high_compression:3.4>2.2", 'low_confidence:0.42').
        wake_score: score del wake que abrió la captura.
        rms: energía RMS de la captura.
        cfg: {enabled, min_wake_score, min_rms, reasons}.
    """
    if not cfg.get("enabled", False):
        return False
    if any(reason.startswith(p) for p in _NOISE_PREFIXES):
        return False
    if wake_score < cfg.get("min_wake_score", 0.55):
        return False
    if rms < cfg.get("min_rms", 0.02):
        return False
    allowed = cfg.get("reasons", ())
    return any(reason.startswith(r) for r in allowed)
