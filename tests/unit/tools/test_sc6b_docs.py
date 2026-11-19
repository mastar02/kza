"""SC-6b finding SC6B-B-M1: the docs carry the measured downtime numbers and the first transition's N.

The solo-window measurement of 2026-11-19 (production-size H) gave T_copy
163-165 s and 14.7 s of stops with T0; with the W8 boot B ~ 146 s the first
managed transition projects D ~ 660 s and the design formula would refuse its
N. The user accepted the one-time overrun (SC6B-B-M1, 2026-11-19): the
operator still passes --smoke-seconds 10. These tests pin that the docs no
longer carry the stale T_copy of 146 s, state what the operator passes, and
that the tool accepts that N in release mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pytest

from scripts import compaction
from scripts import stage_runtime as stage

REPO = Path(__file__).resolve().parents[3]
RUNBOOK = REPO / "docs/runbooks/2026-11-17-release-mode.md"
DEPLOYMENT = REPO / "docs/architecture/DEPLOYMENT.md"
COMPACTION_RUNBOOK = REPO / "docs/runbooks/2026-11-14-owner-compaction.md"
DECISION = "SC6B-B-M1"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` up to the next heading of the same or a higher level."""
    level = heading.split(" ", 1)[0]
    body = text.split("\n" + heading + "\n", 1)[1]
    stop = re.search(r"\n#{1,%d} " % len(level), body)
    return body[:stop.start()] if stop else body


def _bullet(text: str, lead: str) -> str:
    """The flattened top-level bullet that starts with ``lead``."""
    body = text.split("\n- " + lead, 1)[1]
    stop = re.search(r"\n(- |\n|#)", body)
    return _flat(lead + (body[:stop.start()] if stop else body))


class _Parsed(Exception):
    pass


def _parse(monkeypatch, tokens: list[str]) -> argparse.Namespace:
    """stage_runtime.main()'s own parser, nothing run."""
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["stage_runtime.py", "--root", "/nonexistent", "--offline", *tokens])
    try:
        stage.main()
    except _Parsed as parsed:
        return parsed.args[0]
    except SystemExit as exited:
        pytest.fail(f"stage_runtime rejects {tokens!r} (exit {exited.code})")
    pytest.fail(f"stage_runtime ran {tokens!r} instead of parsing it")


@pytest.mark.parametrize("path", [RUNBOOK, DEPLOYMENT, COMPACTION_RUNBOOK], ids=lambda path: path.name)
def test_no_doc_keeps_the_stale_t_copy_of_146_seconds(path):
    flat = _flat(path.read_text())
    assert "T_copy ≈ 146 s" not in flat
    assert "T_copy 163–165 s" in flat
    assert DECISION in flat


def test_the_release_runbook_budget_states_the_measured_values_and_the_one_time_overrun():
    text = RUNBOOK.read_text()
    assert "(`morning_questions_20261119`)" in _flat(text.split("## Compuerta", 1)[0])
    budget = _bullet(text, "**Presupuesto de bajada (SC-D2 A).**")
    for fact in ("Medido por SC-6b a tamaño de producción (2026-11-19): T_copy 163–165 s y paradas 14,7 s con T0",
                 "D ≈ 660 s (unos 11 minutos) con N = 10",
                 "**Primera transición (decisión del usuario SC6B-B-M1, 2026-11-19):** puede pasar los 10 minutos "
                 "una sola vez y usa N = 10 (`--smoke-seconds 10`)",
                 "600 − 164,8 − 3 × 146 − 14,7 = −17,5 y rechazaría ese N (menos de 10)",
                 "Las transiciones posteriores siguen dentro de los 600 s.",
                 "En cada transición, anotar las horas de la parada y del `READY`",
                 "si la primera pasa los ≈ 11 minutos proyectados, o una posterior pasa los 10, se anota y el diseño "
                 "se revisa antes de la próxima transición"):
        assert fact in budget, fact


def test_step_6_states_what_the_operator_passes_and_the_tool_parses_it(monkeypatch):
    step = _flat(_section(RUNBOOK.read_text(), "### Paso 6: corrida de verificación V"))
    assert "**Primera transición, con T0:** `--smoke-seconds 10`." in step
    assert "La fórmula da −17,5 y rechazaría ese N; el operador pasa igual 10 por la decisión SC6B-B-M1" in step
    assert "**Transiciones posteriores:** con los valores de hoy la fórmula da ≈ 135, así que `--smoke-seconds 60`" in step
    for n in (10, 60):
        args = _parse(monkeypatch, ["compact", "verify", "c-1", "--smoke-seconds", str(n)])
        assert args.smoke_seconds == n
        assert n <= compaction.RELEASE_SMOKE_SECONDS_MAX


def test_the_tool_accepts_n_10_and_refuses_9_before_any_release_check(monkeypatch):
    """compact_verify's own range is 10..3600 (release mode caps it at 60 inside the lease): 10 passes it."""

    class _Reached(Exception):
        pass

    def lease(*args, **kwargs):
        raise _Reached()

    monkeypatch.setattr(compaction, "checked_root", lambda root: Path(root))
    monkeypatch.setattr(compaction.stage, "AdmissionController", lease)
    with pytest.raises(compaction.CompactionRefusal, match="^runtime_admission_compaction_smoke_seconds$"):
        compaction.compact_verify("/nonexistent", "c-1", smoke_seconds=9)
    with pytest.raises(_Reached):
        compaction.compact_verify("/nonexistent", "c-1", smoke_seconds=10)
    assert 10 <= compaction.RELEASE_SMOKE_SECONDS_MAX


def test_the_first_transition_is_rechecked_against_the_b_that_w8_measures():
    """SC6B-B-M1-RV1: the accepted overrun (~660 s) assumed B ~ 146 s; W8's B is recomputed before the first transition."""
    text = RUNBOOK.read_text()
    budget = _bullet(text, "**Presupuesto de bajada (SC-D2 A).**")
    assert ("Antes de la primera transición, con el B que mida W8, se recalcula D ≈ 222,4 + 3 × B con N = 10"
            in budget)
    assert "si pasa de unos 660 s, no se sigue bajo SC6B-B-M1 y se vuelve a preguntar al usuario" in budget
    assert "Con B ≤ 136,8 s la fórmula misma ya da N ≥ 10." in budget
    step = _flat(_section(text, "### Paso 6: corrida de verificación V"))
    assert ("Antes, con el B que mida W8, recalcular D ≈ 222,4 + 3 × B: si pasa de unos 660 s, no seguir y volver "
            "a preguntar al usuario.") in step
    n_rule = _bullet(DEPLOYMENT.read_text(), "**N del modo release (SC4A-N14).**")
    assert "Antes de esa transición, D se recalcula con el B que mida W8: D ≈ 222,4 + 3 × B con N = 10" in n_rule
    assert "Si pasa de unos 660 s, la transición no sigue bajo SC6B-B-M1 y se vuelve a preguntar al usuario" in n_rule
    # The numbers: A's measured D without its three stub boots, and the B limits they imply.
    measured_d, stub_boots, t_copy, stops = 242.6, 20.2, 164.8, 14.7
    rest = round(measured_d - stub_boots, 1)
    assert rest == 222.4
    assert round(rest + 3 * 146, 1) == 660.4
    assert round((600 - t_copy - stops - 10) / 3, 1) == 136.8


@pytest.mark.parametrize("path", [RUNBOOK, DEPLOYMENT, COMPACTION_RUNBOOK], ids=lambda path: path.name)
def test_it_is_n_that_the_formula_refuses(path):
    """SC6B-B-M1-RV4: no pronoun that makes the tool or the transition the formula's object."""
    flat = _flat(path.read_text())
    for ambiguous in ("la fórmula no la calcula", "fórmula la rechazaría", "y la rechazaría"):
        assert ambiguous not in flat, ambiguous
    assert "rechazaría ese N" in flat


def test_the_owner_compaction_runbook_keeps_the_recompute_with_w8_b_caveat():
    """SC6B-B-M1-RV2: 'posteriores, 60' holds with B ~ 146 s only; the step points to the release runbook."""
    flat = _flat(COMPACTION_RUNBOOK.read_text())
    assert "transiciones posteriores, 60. Esos N valen con B ≈ 146 s: si W8 mide otro B, se recalculan con ese B" in flat
    assert "(paso 6 y «Presupuesto de bajada» en el runbook del modo release)" in flat


def test_the_sc0b_copy_duration_points_to_the_sc6b_measurement():
    """SC6B-B-M1-RV3: the SC-0b paragraph (145-146 s) tells the reader that SC-6b measured 163-165 s."""
    text = DEPLOYMENT.read_text()
    paragraph = _flat(text.split("\nDuración y espacio (SC-0 y SC-0b", 1)[1].split("\n\n", 1)[0])
    assert "desde SC-0b tarda 145-146 s" in paragraph
    assert ("SC-6b midió la misma copia (plan + start) en 163–165 s el 2026-11-19: ese es el T_copy que usan el "
            "presupuesto de bajada y la fórmula de N (ver «Medición de bajada y disco (SC-6b)»)") in paragraph


def test_deployment_carries_the_per_path_table_and_the_accepted_overrun():
    text = DEPLOYMENT.read_text()
    n_rule = _bullet(text, "**N del modo release (SC4A-N14).**")
    for fact in ("La herramienta sólo controla 10 ≤ N ≤ 60; no calcula la fórmula.",
                 "T_copy 163–165 s, paradas 14,7 s con T0",
                 "600 − 164,8 − 3 × 146 − 14,7 = −17,5: la fórmula rechazaría ese N (menos de 10)",
                 "El operador pasa igual `--smoke-seconds 10` por la decisión del usuario SC6B-B-M1 (2026-11-19)"):
        assert fact in n_rule, fact
    section = _section(text, "### Medición de bajada y disco (SC-6b) — 2026-11-19")
    rows = [line for line in section.splitlines() if line.startswith("| ") and not line.startswith("| Camino")]
    assert rows == [
        "| A: primera, con T0 | 3 (20,2 s) | 242,6 s | 164,8 s | 14,7 s | 924 / 1243 MB | 13,1 MB | ≈ 660 s |",
        "| B: posterior, sin T0 | 2 (13,3 s) | 225,5 s | 163,2 s | 9,5 s | 883 / 1216 MB | 0,3 MB | ≈ 504 s |",
        "| C: abort sin pérdida | 2 (13,4 s) | 204,2 s | 162,7 s | 9,7 s | 882 / 1216 MB | 0,4 MB | ≈ 483 s |",
    ]
    flat = _flat(section)
    assert ("**Presupuesto superado una vez, aceptado (SC6B-B-M1, respuesta del usuario del 2026-11-19, "
            "`morning_questions_20261119`).**") in flat
    assert "las transiciones posteriores siguen dentro de 600 s" in flat
    # The section sits right after the SC-5b procedure, before rows PA and DR.
    assert (text.index("### Procedimiento del modo release (SC-5b)")
            < text.index("### Medición de bajada y disco (SC-6b)")
            < text.index("### Filas PA y DR (SC-PA-DR)"))
