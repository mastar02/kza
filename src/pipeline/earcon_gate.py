"""Decide si suena el earcon 'no entendí'.

Regla "humano plausible": el earcon NUNCA puede sonarle a la TV. Suena solo si
el wake fue fuerte Y hubo energía real Y el motivo del reject es de
no-comprensión (no de ruido/eco). Lógica pura, un solo lugar testeable.
"""
from __future__ import annotations

# Motivos que indican TV/eco/ruido → JAMÁS earcon (aunque el wake sea fuerte).
_NOISE_PREFIXES = ("noise_phrase", "filler_word", "word_repetition", "missing_wake", "prompt_echo")

# Motivos con evidencia POSITIVA de habla real, que por eso saltean el piso de
# RMS. El piso existe para que el earcon no le suene a la TV distante; cuando
# el motor primario ya produjo una transcripción, la duda sobre "¿había habla?"
# no existe. Caso real: 2026-07-25 18:19, wake 0.87 y rms 0.0105 (los comandos
# de este mic viven en 0.010-0.013, por debajo del min_rms 0.02 de prod) — el
# veto se comió un "prendé la luz" sin un solo sonido de vuelta.
_REAL_SPEECH_PREFIXES = ("stt_veto",)


def should_play_earcon(reason: str, wake_score: float, rms: float, cfg: dict) -> bool:
    """True si corresponde reproducir el earcon para este reject.

    Args:
        reason: AcceptanceDecision.reason o el intent de reject del router
            (ej: 'empty', "high_compression:3.4>2.2", 'low_confidence:0.42',
            'stt_veto').
        wake_score: score del wake que abrió la captura.
        rms: energía RMS de la captura.
        cfg: {enabled, min_wake_score, min_rms, reasons, exclude_reasons}.
            `exclude_reasons` es el escape hatch para apagar un motivo que de
            otro modo entra por default (hoy sólo `stt_veto`).
    """
    if not cfg.get("enabled", False):
        return False
    if any(reason.startswith(p) for p in _NOISE_PREFIXES):
        return False
    if any(reason.startswith(p) for p in cfg.get("exclude_reasons", ())):
        return False
    if wake_score < cfg.get("min_wake_score", 0.55):
        return False

    real_speech = any(reason.startswith(p) for p in _REAL_SPEECH_PREFIXES)
    if not real_speech and rms < cfg.get("min_rms", 0.02):
        return False

    # Los motivos con evidencia de habla real no dependen de estar listados en
    # `reasons`: si dependieran, el arreglo quedaría inerte hasta editar la
    # config de producción.
    if real_speech:
        return True
    allowed = cfg.get("reasons", ())
    return any(reason.startswith(r) for r in allowed)
