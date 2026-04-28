"""Regex-based command extractor para KZA — fast path determinístico.

Submódulos:
    vocab       — listas controladas (verbos voseo, entidades whitelisted, cardinales, colores)
    blockers    — detectores de negación, pasado, modo, pregunta, etc.
    normalize   — strip de wake word, lowercase, whitespace
    extractor   — RegexExtractor.extract() → list[RegexMatch]
    _corpus     — golden test corpus (consumido por tests/unit/nlu/test_regex_extractor.py)
"""
from .extractor import RegexExtractor, RegexMatch
from .normalize import normalize

__all__ = ["RegexExtractor", "RegexMatch", "normalize"]
