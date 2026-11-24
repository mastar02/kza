"""Alias histórico de ``scripts/train_nexa.py`` (ambos entrenan el wake word «nexa»).

Consolidado en Fase 4 de la rearquitectura de entrenamiento para eliminar las
376 líneas duplicadas manteniendo su propio espacio global de módulo (requerido
por ``test_legacy_wakeword_capture_authority`` al parchar ``module.ROOT``).
"""
from __future__ import annotations

from pathlib import Path

_SOURCE = Path(__file__).resolve().with_name("train_nexa.py")
exec(compile(_SOURCE.read_text(encoding="utf-8"), str(_SOURCE), "exec"), globals())  # noqa: S102 -- delegates verbatim to sibling scripts/train_nexa.py
