"""SC-5a: the pre-R1 docs state that delegation is retired and claim no release mode yet.

sc-release-design v3 implementation_units SC-5a tests: relative markdown links
of the four touched docs resolve; DEPLOYMENT and the runbooks say epoch 4 and
that no managed transition exists after R1 until release mode ships.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = ("docs/architecture/DEPLOYMENT.md", "docs/runbooks/2026-11-13-r1-adoption-window.md",
        "docs/runbooks/2026-11-14-owner-compaction.md", "docs/runbooks/2026-11-05-mac-development-handoff.md")
_LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("relative", DOCS)
def test_relative_links_resolve(relative):
    document = REPO / relative
    for target in _LINK.findall(document.read_text()):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = (document.parent / target.split("#", 1)[0]).resolve()
        if path.is_relative_to(REPO / "data"):
            continue  # local evidence, gitignored: absent from clones by design
        assert path.exists(), (relative, target)


def test_deployment_states_the_retirement_and_the_gap():
    text = (REPO / DOCS[0]).read_text()
    assert "### Delegación retirada desde SC-2" in text
    assert "runtime_admission_delegation_retired" in text
    assert "runtime_admission_transition_requires_release_mode" in text
    assert "### Sin transiciones gestionadas después de R1 hasta el modo release" in text
    assert "SC-D5" in text and "`controller_epoch == 4`" in text
    assert "`controller_epoch == 3`" not in text
    # No claim that release mode exists, and no grant instructions left for operators.
    for claim in ("modo release está disponible", "modo release disponible", "repetí `delegate <slot>`"):
        assert claim not in text


_GAP = "Sin transiciones gestionadas después de R1 hasta el modo release"


def _paragraphs(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text)]


def test_every_managed_rollback_mention_is_marked_unavailable():
    """SC-2 review R1: no paragraph offers a managed rollback without pointing to the gap section."""
    text = (REPO / DOCS[0]).read_text()
    mentions = [block for block in _paragraphs(text) if re.search(r"[Rr]ollback gestionado", block)]
    assert len(mentions) >= 3
    for block in mentions:
        assert _GAP in block or block.startswith("### " + _GAP), block[:160]


def test_install_controller_minimum_is_epoch_4_everywhere():
    """SC-2 review R2: DEPLOYMENT and the compaction runbook agree on the D4 exit path."""
    for relative in (DOCS[0], DOCS[2]):
        text = " ".join((REPO / relative).read_text().split())
        assert "época 3 o posterior" not in text, relative
    assert "`install-controller` (época 4 o posterior desde SC-2" in " ".join((REPO / DOCS[0]).read_text().split())


def test_deployment_states_the_epoch_lockout_residual():
    """SC-2 review SC2-R3: the lockout needs a readable controller.json."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "Ese bloqueo necesita un `admission/controller.json` legible" in text
    assert "repararlo sólo con `controller/current` o `install-controller`" in text
    assert "no correr el `stage_runtime` de ningún slot" in text


def test_deployment_states_the_schema_contract_and_its_exclusions():
    """SC-2 review SC2-R1: the extended vocabularies, their status and the excluded literal sets."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "#### Contrato de esquema (época 1)" in (REPO / DOCS[0]).read_text()
    assert "enmienda propuesta del diseño (sin confirmar)" in text and "antes de SC-6a" in text
    exclusions = [block for block in _paragraphs((REPO / DOCS[0]).read_text())
                  if block.startswith("Quedan fuera los conjuntos cerrados que `src` escribe como literales sin"
                                      " nombre importable")]
    assert len(exclusions) == 1
    for excluded in ("capture_runtime_control.mode", "capture_browser_recorders.kind", "routine_runs.source",
                     "legacy_quarantine_records_v1.state", "reminders.recurrence"):
        assert excluded in exclusions[0], excluded


def test_r1_day14_says_no_rollback_remains_and_checks_sc_d5():
    """SC-2 review R3: the day-14 section itself warns, so an operator reading only it sees SC-D5."""
    text = (REPO / DOCS[1]).read_text()
    section = text.split("## Día 14: retiro Q5", 1)[1].split("\n0. ", 1)[0]
    flat = " ".join(section.split())
    assert "tampoco hay rollback gestionado" in flat and "no queda ningún rollback" in flat
    assert "SC-D5" in flat and "opción B" in flat


def test_runbooks_name_epoch_4():
    r1 = (REPO / DOCS[1]).read_text()
    assert "`controller_epoch` 4" in r1 and "época 4)" in r1 and "SC-6a" in r1
    assert "`controller_epoch` 3, que" not in r1
    compaction = (REPO / DOCS[2]).read_text()
    assert "`controller_epoch == 4`" in compaction and "`controller_epoch == 3`" not in compaction
