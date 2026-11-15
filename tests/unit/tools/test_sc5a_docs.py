"""SC-5a: the pre-R1 docs state that delegation is retired and claim no release mode yet.

sc-release-design v3 implementation_units SC-5a tests: relative markdown links
of the four touched docs resolve; DEPLOYMENT and the runbooks say epoch 4 and
that no managed transition exists after R1 until release mode ships.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

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
    """SC-2 review R2 and C1: DEPLOYMENT, the R1 runbook and the compaction runbook agree on epoch 4."""
    for relative in (DOCS[0], DOCS[1], DOCS[2]):
        text = " ".join((REPO / relative).read_text().split())
        assert "época 3 o posterior" not in text, relative
        assert "Exige época de controlador 2" not in text, relative
    assert "`install-controller` (época 4 o posterior desde SC-2" in " ".join((REPO / DOCS[0]).read_text().split())
    w3 = [line for line in (REPO / DOCS[1]).read_text().splitlines() if line.startswith("- **W3**:")]
    assert w3 == ["- **W3**: `$ST install-controller r1.tar.gz`. El archivo debe declarar `controller_epoch` 4,"
                  " exactamente la época que comprueba `adoption-commit`."]


def test_run_is_not_offered_for_older_slots_under_a_managed_head():
    """SC-2 review C2: run() goes through activate, so older lineages run only in legacy test roots."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "`run()` sigue aceptando slots anteriores" not in text
    assert ("Desde SC-2, bajo un head gestionado `run()` sólo corre el slot del propio head; correr linajes"
            " viejos queda para raíces de prueba con `LEGACY_MANAGED_TRANSITIONS`.") in text


def test_deployment_names_the_other_head_moves_and_that_they_grant_nothing():
    """SC-2 review AUTHZ-2: adoption-commit after an aborted head and plain compaction's K remain."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "la única transición gestionada entre slots sea el modo release" in text
    assert "la única transición gestionada sea el modo release" not in text
    bullet = text.split("- Siguen moviendo el head, por diseño:", 1)[1].split(" - ", 1)[0]
    assert "`adoption-commit` cuando el head es un head de adopción abortado" in bullet
    assert "el K de una compactación simple" in bullet and "A1 rechaza un K delegado" in bullet
    # SC-2 review RV3-1: compaction's abort also moves the head, back to H.
    assert ("el regreso a H de `compact abort --to-h` (`compaction_return_to_h`, bajo la compuerta de su"
            " compactación)") in bullet
    assert "Ninguno concede dueños ni autoridad derivada." in bullet


def test_every_compaction_pending_list_with_delegate_names_the_retirement():
    """SC-2 review RV3-2: delegate refuses with delegation_retired before reaching the compaction gate."""
    checked = 0
    for relative in (DOCS[0], DOCS[2]):
        for block in _paragraphs((REPO / relative).read_text()):
            if "runtime_admission_compaction_pending" in block and "delegate" in block:
                checked += 1
                assert ("desde SC-2, `delegate` rechaza siempre con `runtime_admission_delegation_retired`"
                        in block), (relative, block[:160])
    assert checked >= 2


# SC-2 review AUTHZ3-1: the epoch lockout lives in the old code, so it covers only copies that have it.
_LOCKOUT_FROM = "b0c322d"  # first stage_runtime with require_forward_epoch
_DELEGATE_FROM = "69a79e6"  # first stage_runtime with delegate_slot (U3b-0a)
_FALSE_LOCKOUT_CLAIMS = ("ninguna concesión es posible", "Así un controlador viejo no puede volver a conceder",
                         "una copia de época 3 ya no toma el lease", "así que no puede volver a conceder")


def _stage_runtime_at(revision: str) -> str:
    result = subprocess.run(["git", "show", f"{revision}:scripts/stage_runtime.py"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.fail(f"{revision} is not available in this clone: {result.stderr[:200]}")
    return result.stdout


def test_the_documented_unlocked_range_matches_the_history():
    """The range the docs name: delegate exists from 69a79e6, the forward-epoch check only from b0c322d."""
    assert "def delegate_slot(" not in _stage_runtime_at(_DELEGATE_FROM + "~1")
    assert "def delegate_slot(" in _stage_runtime_at(_DELEGATE_FROM)
    before = _stage_runtime_at(_LOCKOUT_FROM + "~1")
    assert "def delegate_slot(" in before and "require_forward_epoch" not in before
    assert "\nCONTROLLER_EPOCH = 3\n" in before
    assert "def require_forward_epoch(" in _stage_runtime_at(_LOCKOUT_FROM)


def test_the_old_copy_lockout_is_scoped_to_copies_with_the_check():
    """AUTHZ3-1: no doc claims an old controller copy can no longer grant; each names the unlocked range."""
    for relative in DOCS:
        text = " ".join((REPO / relative).read_text().split())
        for claim in _FALSE_LOCKOUT_CLAIMS:
            assert claim not in text, (relative, claim)
    deployment = " ".join((REPO / DOCS[0]).read_text().split())
    assert ("Una copia de `stage_runtime` que ya contiene `require_forward_epoch` (desde b0c322d, la corrección F8"
            " de U3b-0c, todavía época 3; el 26de262 fijado incluido)") in deployment
    assert ("Toda copia de `stage_runtime` de 69a79e6 (U3b-0a, donde entra `delegate`) a b0c322d~1 no comprueba"
            " la época") in deployment
    assert "en cada ubicación de transferencia, no sólo los R1 de época 3" in deployment
    assert "Barrera estructural diferida a SC-1 (decisión técnica del 2026-11-15)" in deployment
    assert "sin confirmar" not in deployment
    for relative in (DOCS[1], DOCS[2], DOCS[3]):
        text = " ".join((REPO / relative).read_text().split())
        assert "69a79e6" in text and "b0c322d~1" in text and "AUTHZ3-1" in text, relative
    for relative in (DOCS[1], DOCS[3]):  # the SC-6a cleanup and rehearsal addition
        text = " ".join((REPO / relative).read_text().split())
        assert "bajo `yo` y en cada ubicación de transferencia" in text, relative
        assert "de una copia de ese rango sobre una raíz de época 4" in text, relative


def test_deployment_states_the_epoch_lockout_residual():
    """SC-2 review SC2-R3: the lockout needs a readable controller.json."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "Ese bloqueo necesita un `admission/controller.json` legible" in text
    assert "repararlo sólo con `controller/current` o `install-controller`" in text
    assert "no correr el `stage_runtime` de ningún slot" in text


def test_deployment_states_the_schema_contract_and_its_exclusions():
    """SC-2 review SC2-R1 and AUTHZ-1: the vocabularies, the rule, their status and the excluded sets."""
    text = " ".join((REPO / DOCS[0]).read_text().split())
    assert "#### Contrato de esquema (época 1)" in (REPO / DOCS[0]).read_text()
    assert "enmienda del diseño aceptada el 2026-11-15 como decisión técnica" in text
    assert "revocable hasta que SC-6a construya R1" in text
    assert "la regla es la relevancia para la privacidad, no el conjunto protegido" in text
    for pinned in ("`ErasureState`", "`ErasureError`", "`IndexCleanupError`", "`MemoryOrigin`"):
        assert pinned in text, pinned
    exclusions = [block for block in _paragraphs((REPO / DOCS[0]).read_text())
                  if block.startswith("Quedan fuera los conjuntos cerrados que `src` escribe como literales sin"
                                      " nombre importable")]
    assert len(exclusions) == 1
    for excluded in ("capture_runtime_control.mode", "capture_browser_recorders.kind", "routine_runs.source",
                     "legacy_quarantine_records_v1.state", "reminders.recurrence",
                     "policy_index_deletions.state", "facts.category", "voice_groups.state"):
        assert excluded in exclusions[0], excluded
    assert "subir la época a mano" in exclusions[0]


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
