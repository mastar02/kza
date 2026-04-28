"""Dev runner para el golden corpus del RegexExtractor.

Ejecuta el corpus con output legible en consola — útil para iteración rápida
durante el desarrollo del extractor o el corpus mismo. Para CI usá pytest:
    pytest tests/unit/nlu/test_regex_extractor.py

Uso:
    python -m scripts.run_regex_corpus
"""
from __future__ import annotations

import sys
from collections import defaultdict

from src.nlu.regex import RegexExtractor
from src.nlu.regex._corpus import (
    FUZZY_POSITIVES,
    KNOWN_LIMITATIONS,
    MULTI_POSITIVES,
    NEGATIVES,
    POSITIVES,
)


def main() -> int:
    ext = RegexExtractor()
    total = 0
    passed = 0
    failures: list[str] = []

    print("=" * 78)
    print("POSITIVOS — el extractor DEBE matchear")
    print("=" * 78)
    for text, expected in POSITIVES:
        total += 1
        matches = ext.extract(text)
        if not matches:
            failures.append(f"[FAIL] no_match: {text!r}")
            print(f"  ✗ {text!r}")
            continue
        m = matches[0]
        intent_ok = m.intent == expected["intent"]
        ent_ok = m.entity_canonical == expected["entity"]
        slots_ok = all(m.slots.get(k) == v for k, v in expected["slots"].items())
        if intent_ok and ent_ok and slots_ok:
            passed += 1
            print(f"  ✓ {text!r}")
        else:
            print(f"  ✗ {text!r}  expected={expected}  got=intent={m.intent} entity={m.entity_canonical} slots={m.slots}")
            failures.append(f"[FAIL] mismatch: {text!r}")

    print("\nMULTI-INTENT")
    print("=" * 78)
    for text, expected_list in MULTI_POSITIVES:
        total += 1
        matches = ext.extract(text)
        if len(matches) != len(expected_list):
            print(f"  ✗ {text!r} → {len(matches)} matches, esperaba {len(expected_list)}")
            failures.append(f"[FAIL] multi_count: {text!r}")
            continue
        ok = all(
            m.intent == exp["intent"] and m.entity_canonical == exp["entity"]
            for m, exp in zip(matches, expected_list)
        )
        if ok:
            passed += 1
            print(f"  ✓ {text!r}")
        else:
            print(f"  ✗ {text!r}")
            failures.append(f"[FAIL] multi_content: {text!r}")

    print("\nFUZZY (Whisper hallucinations) — best-effort, no obligatorio")
    print("=" * 78)
    for text, expected in FUZZY_POSITIVES:
        matches = ext.extract(text)
        status = "✓ matched" if matches else "○ skipped"
        print(f"  {status}: {text!r}")

    print("\nLIMITACIONES CONOCIDAS — el regex falla por diseño, LLM lo maneja")
    print("=" * 78)
    for text, reason in KNOWN_LIMITATIONS:
        matches = ext.extract(text)
        if len(matches) < 2:
            print(f"  ✓ documentado ({reason}): {text!r} → {len(matches)} match")
        else:
            print(f"  ! ahora SÍ funciona ({reason}): {text!r} — mover a MULTI_POSITIVES")

    print("\nNEGATIVOS — el extractor NO debe matchear")
    print("=" * 78)
    by_reason: dict[str, list[str]] = defaultdict(list)
    for text, reason in NEGATIVES:
        total += 1
        matches = ext.extract(text)
        if not matches:
            passed += 1
        else:
            by_reason[reason].append(text)
            print(f"  ✗ FALSE POSITIVE ({reason}): {text!r} → intent={matches[0].intent}")
            failures.append(f"[FAIL-FP] {reason}: {text!r}")

    print()
    print("=" * 78)
    print(f"RESUMEN: {passed}/{total} ({100 * passed / total:.1f}%)")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
