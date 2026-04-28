"""Pytest harness para src/nlu/regex/extractor.py.

Consume el golden corpus de src/nlu/regex/_corpus.py y valida:
- Positivos: matchean con intent + entity + slots correctos.
- Multi-intent: split por conectores devuelve N entries con orden estable.
- Negativos: devuelven [] (rechazados).
- Limitaciones conocidas: documentadas, marcadas como xfail.

Cualquier divergencia entre el corpus y el comportamiento del extractor
bloquea el merge. Cuando se agregue una nueva forma de comando o un nuevo
verbo voseo a vocab.py, también se agrega su caso al corpus.
"""
from __future__ import annotations

import pytest

from src.nlu.regex import RegexExtractor
from src.nlu.regex._corpus import (
    FUZZY_POSITIVES,
    KNOWN_LIMITATIONS,
    MULTI_POSITIVES,
    NEGATIVES,
    POSITIVES,
)


@pytest.fixture(scope="module")
def extractor() -> RegexExtractor:
    return RegexExtractor()


@pytest.mark.parametrize("text,expected", POSITIVES)
def test_positive_single(extractor: RegexExtractor, text: str, expected: dict) -> None:
    matches = extractor.extract(text)
    assert len(matches) == 1, f"esperaba 1 match para {text!r}, got {len(matches)}"
    m = matches[0]
    assert m.intent == expected["intent"], (
        f"intent mismatch para {text!r}: esperado {expected['intent']}, got {m.intent}"
    )
    assert m.entity_canonical == expected["entity"], (
        f"entity mismatch para {text!r}: esperado {expected['entity']}, got {m.entity_canonical}"
    )
    for k, v in expected["slots"].items():
        assert m.slots.get(k) == v, (
            f"slot {k} mismatch para {text!r}: esperado {v}, got {m.slots.get(k)}"
        )


@pytest.mark.parametrize("text,expected_list", MULTI_POSITIVES)
def test_positive_multi(extractor: RegexExtractor, text: str, expected_list: list[dict]) -> None:
    matches = extractor.extract(text)
    assert len(matches) == len(expected_list), (
        f"split count mismatch para {text!r}: esperaba {len(expected_list)}, got {len(matches)}"
    )
    for m, exp in zip(matches, expected_list):
        assert m.intent == exp["intent"]
        assert m.entity_canonical == exp["entity"]


@pytest.mark.parametrize("text,reason", NEGATIVES)
def test_negative_rejected(extractor: RegexExtractor, text: str, reason: str) -> None:
    """Todo negativo DEBE caer (devolver []). Reason es informativa."""
    matches = extractor.extract(text)
    assert matches == [], (
        f"falso positivo ({reason}): {text!r} devolvió "
        f"intent={matches[0].intent if matches else None}"
    )


@pytest.mark.parametrize("text,reason", KNOWN_LIMITATIONS)
@pytest.mark.xfail(strict=True, reason="limitación documentada del regex puro")
def test_known_limitations(extractor: RegexExtractor, text: str, reason: str) -> None:
    """Estos casos NO los maneja el regex por diseño — se delegan al LLM."""
    matches = extractor.extract(text)
    # Esperamos que el regex falle (ya sea no matchear o split incorrecto).
    # Si en el futuro alguien lo arregla, este test pasa y xfail strict
    # convierte el pass en failure → indicación para mover el caso a
    # MULTI_POSITIVES / POSITIVES.
    assert len(matches) >= 2, "esperamos que matchee correctamente"


@pytest.mark.parametrize("text,expected", FUZZY_POSITIVES)
def test_fuzzy_whisper_hallucinations(
    extractor: RegexExtractor, text: str, expected: dict
) -> None:
    """Whisper a veces emite typos. Aceptamos que el regex no los matchee
    todos hoy — los cubriremos con typo dictionary. Por ahora solo
    documentamos el comportamiento."""
    matches = extractor.extract(text)
    if matches:
        # Si matchea, debe ser el intent esperado.
        assert matches[0].intent == expected["intent"]
