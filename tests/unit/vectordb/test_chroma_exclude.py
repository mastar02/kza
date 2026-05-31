"""Exclusión configurable de entidades del vector search (ChromaSync).

Motiva: migración de luces a Zigbee2MQTT con patrón grupo-por-habitación —
las bombillas miembro (light.escritorio1/2/3) no deben indexarse para voz;
solo el grupo (light.escritorio). Filtro por lista explícita + patrones regex,
configurable desde settings.yaml (vectordb.exclude_entities/exclude_patterns).
"""
import pytest

from src.vectordb.chroma_sync import ChromaSync


def _make(**kw):
    return ChromaSync(
        chroma_path="/tmp/x",
        embedder_model="m",
        embedder_device="cpu",
        **kw,
    )


def test_default_excludes_hogar():
    cs = _make()
    assert cs._is_excluded("light.hogar") is True
    assert cs._is_excluded("light.escritorio") is False


def test_excluded_by_explicit_list():
    cs = _make(excluded_entities=["light.hogar", "light.test"])
    assert cs._is_excluded("light.test") is True
    assert cs._is_excluded("light.hogar") is True
    assert cs._is_excluded("light.escritorio") is False


def test_excluded_by_pattern_members_not_group():
    cs = _make(excluded_patterns=[r"^light\.escritorio\d+$"])
    # miembros (terminan en dígito) → excluidos
    assert cs._is_excluded("light.escritorio1") is True
    assert cs._is_excluded("light.escritorio2") is True
    assert cs._is_excluded("light.escritorio3") is True
    # el grupo (sin dígito) → se mantiene
    assert cs._is_excluded("light.escritorio") is False
    # otras habitaciones → no afectadas
    assert cs._is_excluded("light.living") is False


def test_list_and_patterns_combined():
    cs = _make(
        excluded_entities=["light.hogar"],
        excluded_patterns=[r"^light\.cuarto\d+$"],
    )
    assert cs._is_excluded("light.hogar") is True
    assert cs._is_excluded("light.cuarto1") is True
    assert cs._is_excluded("light.cuarto") is False


def test_invalid_pattern_is_ignored_not_crash():
    # un patrón malformado no debe tumbar el sync; se ignora.
    cs = _make(excluded_patterns=[r"[unclosed", r"^light\.bano\d+$"])
    assert cs._is_excluded("light.bano1") is True
    assert cs._is_excluded("light.bano") is False
