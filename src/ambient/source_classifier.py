"""SourceClassifier — etiqueta la fuente de cada utterance (spec §4).

Orden de reglas (la primera que aplica gana):
  1. during_tts            → "self"   (nuestro TTS sonando; no va al RAG)
  2. speaker conocido      → "live"   (la persona manda sobre la dirección)
  3. sin DoA o inestable   → "unknown"
  4. DoA ≈ tv_azimuth      → "tv"     (requiere tv_azimuth calibrado en Fase 2)
  5. DoA estable no-TV     → "live" (o "unknown" si require_known_speaker_for_live)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ambient.doa import angular_distance


@dataclass
class SourceClassifierConfig:
    """Umbrales del clasificador de fuente (settings.yaml → ambient.classifier)."""

    tv_azimuth: float | None = None        # rad relativo; None = sin calibrar
    tv_tolerance_rad: float = 0.35
    min_stability: float = 0.6
    require_known_speaker_for_live: bool = False


class SourceClassifier:
    """Reglas declarativas — sin estado, trivialmente testeable."""

    def __init__(self, config: SourceClassifierConfig):
        self._cfg = config

    def classify(
        self,
        *,
        speaker: str,
        azimuth: float | None,
        stability: float,
        during_tts: bool,
    ) -> str:
        """Clasificar la fuente de una utterance.

        Aplica las 5 reglas del docstring del módulo en orden; la primera
        que matchea gana.

        Returns:
            Uno de SOURCE_VALUES: "live" | "tv" | "self" | "unknown".
        """
        cfg = self._cfg
        if during_tts:
            return "self"
        if speaker != "unknown":
            return "live"
        if azimuth is None or stability < cfg.min_stability:
            return "unknown"
        if (
            cfg.tv_azimuth is not None
            and angular_distance(azimuth, cfg.tv_azimuth) <= cfg.tv_tolerance_rad
        ):
            return "tv"
        return "unknown" if cfg.require_known_speaker_for_live else "live"
