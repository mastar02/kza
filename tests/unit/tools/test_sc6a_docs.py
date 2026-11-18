"""SC-6a: the R1 runbook carries the mandatory pinned-archive step before W1 (PS-13).

sc-release-design v3 epochs.r1_rebuild: 'sha256sum "$R1_ARCHIVE"' must equal
the pinned value, else stop. The block itself is run by the rehearsals
(tests/mocks/r1_runbook.py); here its place and shape are pinned.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from tests.mocks import r1_runbook

REPO = Path(__file__).resolve().parents[3]


def test_the_runbook_has_the_pinned_archive_step_before_any_command_and_w1():
    text = r1_runbook.runbook_text()
    section = text.index(r1_runbook.HEADING)
    assert section < text.index("## Antes de la ventana") < text.index("- **W1**. Ejecutar:")
    assert section < text.index("$ST adoption-dry-run")
    block = r1_runbook.check_block(text)
    assert set(r1_runbook.pinned(text)) == {"R1_ARCHIVE", "R1_SHA256", "R1_COMMIT"}
    assert ('if [ "$(sha256sum "$R1_ARCHIVE" | cut -d\' \' -f1)" = "$R1_SHA256" ]; then echo "'
            + r1_runbook.OK + '"; else echo "' + r1_runbook.STOP + '"; false; fi') in block.splitlines()
    flat = " ".join(text.split())
    assert "Si no imprime `R1 fijado: OK` (sale con 1), **el procedimiento se detiene**" in flat
    assert "El valor vive en un commit de documentación posterior al commit construido" in flat
    assert "Sólo cuenta el sha256 del archivo" in flat
    # W1 is preceded by the same check, with the same archive.
    window = text.split("\n## Ventana\n", 1)[1]
    pre = window.index("- **W1-pre** (obligatorio): repetir la verificación de «Archivo R1 fijado»")
    assert pre < window.index("- **W1**. Ejecutar:")
    assert "Sin `R1 fijado: OK`, detener: no hay W1." in " ".join(window.split())
    # Re-review RR-4: the W1-pre sentence that keeps $ST from the extraction (or repeats it) is pinned.
    assert ("`$ST` sigue siendo el de «Extraer R1 verificado» en esta misma sesión; si no, repetir el bloque "
            "`PY=`/`C=`, «Archivo R1 fijado» y ese bloque (otro `mktemp -d`).") in " ".join(window.split())


def test_r1_and_st_come_from_a_new_extraction_of_the_verified_archive():
    """SC6A-R2: right after the check the runbook extracts "$R1_ARCHIVE" into a new mktemp -d and
    defines $ST only there; every command passes "$R1_ARCHIVE", never a relative r1.tar.gz."""
    text = r1_runbook.runbook_text()
    check = text.index(r1_runbook.HEADING)
    extract = text.index(r1_runbook.EXTRACT_HEADING)
    assert check < extract < text.index("## Antes de la ventana") < text.index("$ST adoption-dry-run")
    lines = r1_runbook.extract_block(text).splitlines()
    # Re-review RR-2: the block defines $PY itself, so a new session that repeats only the two blocks works.
    assert lines[0] == r1_runbook.RUNBOOK_PY
    assert lines[1] == "unset R1 ST"
    assert lines[2].startswith("R1=$(mktemp -d)")
    assert lines[3] == 'tar -xzf "$R1_ARCHIVE" -C "$R1"'
    assert lines[4].startswith('if [ "$("$PY" -I -c ') and '"$R1/release.json")" = "$R1_COMMIT" ]; then ST=' in lines[4]
    # $R1 is assigned nowhere else, and $ST only here and in the day-14 block (from controller/current).
    assigned = [line for line in text.splitlines() if re.match(r"\s*(R1|ST)=", line)]
    assert assigned == [lines[2], r1_runbook.day14_block(text).splitlines()[2]], assigned
    assert text.count('ST="$PY -I -B $R1/scripts/stage_runtime.pyc') == 1 and "R1=<" not in text
    # Every command takes the verified archive by its variable.
    flat = " ".join(text.split())
    for command in ('$ST adoption-dry-run --container-slot $C --archive "$R1_ARCHIVE" --slot',
                    '- **W2**: `$ST adoption-prepare --container-slot $C --archive "$R1_ARCHIVE" --slot',
                    '- **W3**: `$ST install-controller "$R1_ARCHIVE"`.'):
        assert command in flat, command
    assert flat.count("r1.tar.gz") == 1 and "R1_ARCHIVE=<ruta del r1.tar.gz transferido>" in flat
    assert "Sin `R1 extraído: OK`, detener: `$ST` queda sin definir." in flat
    assert ("Si la sesión se pierde, repetir el bloque `PY=`/`C=` de arriba (los comandos de la ventana usan "
            "`$C`), «Archivo R1 fijado» y este bloque (otro `mktemp -d`).") in flat


CONTROLLER_ST = ('ST="$PY -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc '
                 '--root /home/kza/releases --offline"')


def test_day14_defines_its_commands_from_the_installed_controller():
    """Re-review RR-1: day 14 runs 14 days after W9, in a new session where the window's extracted $ST is
    gone. The section defines PY, C and ST from controller/current before any step that runs $ST, as W4 and
    completion-status do; DEPLOYMENT's «Cómo se corre» does the same."""
    text = r1_runbook.runbook_text()
    assert r1_runbook.day14_block(text).splitlines() == [
        r1_runbook.RUNBOOK_PY, "C=release_20261029T185008966364Z_e9669ec", CONTROLLER_ST]
    section = text.split(r1_runbook.DAY14_HEADING, 1)[1].split("\n## ", 1)[0]
    assert section.index(CONTROLLER_ST) < section.index("\n0. ") < section.index("$ST retire-legacy-state")
    flat = " ".join(section.split())
    assert "El día 14 se corre en una sesión nueva de `kza`, no en la de la ventana" in flat
    assert "Desde W3 sólo se corre `controller/current`" in flat
    for command in ("$ST retire-legacy-state --decision <id P7> --container-slot $C --dry-run",
                    "$ST retire-legacy-state --decision <id P7> --container-slot $C --confirm <plan_sha256>"):
        assert command in flat, command
    deployment = (REPO / "docs/architecture/DEPLOYMENT.md").read_text()
    how = deployment.split("Cómo se corre (bajo `kza`, nunca como root)", 1)[1].split("\n- ", 1)[0]
    block = re.findall(r"```bash\n(.*?)```", how, flags=re.S)[0].splitlines()
    assert block[:2] == [r1_runbook.RUNBOOK_PY, CONTROLLER_ST], block
    assert all(line.startswith(("$ST retire-legacy-state", "systemctl --user")) for line in block[2:]), block


def _git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *arguments], cwd=REPO, capture_output=True, text=True, timeout=60)


def test_the_pin_names_an_epoch4_commit_of_this_history_and_the_docs_agree():
    """The pinned values are a sha256 and a commit of this history whose controller is epoch 4 with
    schema contract 1; DEPLOYMENT and the handoff name the same values. (The pin itself lives in a
    later docs commit: an archive cannot contain its own hash.)"""
    pin = r1_runbook.pinned()
    assert re.fullmatch(r"[0-9a-f]{64}", pin["R1_SHA256"]), pin
    assert re.fullmatch(r"[0-9a-f]{40}", pin["R1_COMMIT"]), pin
    assert pin["R1_ARCHIVE"] == "<ruta del r1.tar.gz transferido>"  # the operator fills the path, never the pin
    commit = pin["R1_COMMIT"]
    if _git("cat-file", "-e", commit + "^{commit}").returncode != 0:
        pytest.fail(commit + " is not available in this clone")
    assert _git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
    source = _git("show", commit + ":scripts/stage_runtime.py").stdout
    assert "\nCONTROLLER_EPOCH = 4\n" in source and "\nSCHEMA_CONTRACT_EPOCH = 1\n" in source
    assert "runtime_admission_delegation_retired" in source
    short = commit[:7]
    for relative in ("docs/architecture/DEPLOYMENT.md", "docs/runbooks/2026-11-05-mac-development-handoff.md"):
        flat = " ".join((REPO / relative).read_text().split())
        assert pin["R1_SHA256"] in flat and short in flat, relative
    assert f"/home/yo/projects/releases-r1/r1-{short}.tar.gz" in " ".join(r1_runbook.runbook_text().split())


def test_the_superseded_81ed568_archive_is_recorded_not_pinned():
    """R1-B re-pin: SC-6a's archive (81ed568) was kept beside the new one under a superseded name;
    the runbook, DEPLOYMENT and the handoff record that name, and the pin is no longer it."""
    pin = r1_runbook.pinned()
    assert pin["R1_COMMIT"] != "81ed5680fa46d0b5b48e8e6eb9e4a36da5f15ed0"
    assert pin["R1_SHA256"] != "3f57bd197f2f1633f32683616abefd3db038466d7a966d9925b24c84fc342a0e"
    for relative in ("docs/runbooks/2026-11-13-r1-adoption-window.md", "docs/architecture/DEPLOYMENT.md",
                     "docs/runbooks/2026-11-05-mac-development-handoff.md"):
        flat = " ".join((REPO / relative).read_text().split())
        assert "r1-81ed568.tar.gz.superseded-r1b" in flat, relative
    handoff = " ".join((REPO / "docs/runbooks/2026-11-05-mac-development-handoff.md").read_text().split())
    assert "El runbook de R1 lo fija y exige" not in handoff
    assert "El runbook de R1 lo fijó hasta ese re-fijado" in handoff


def test_the_scheduling_gate_states_what_is_done_and_leaves_scheduling_to_the_user():
    """The U3b-0e5 gate (independent review, compiled T-A17 on Linux) is stated as met with its
    references; it is not the only thing left: the SC-6a fix-round re-review and the R1-B1/R1-B2
    round-1 independent review stay open before the window, then operator steps; no document
    declares the window scheduled."""
    text = r1_runbook.runbook_text()
    gate = " ".join(text.split("## Compuerta de programación (P7, Q5)", 1)[1].split("**Paso del día 14**", 1)[0].split())
    assert "Faltan su revisión independiente" not in gate
    assert "Estado (2026-11-15): esta compuerta está cumplida." in gate
    for reference in ("u3b-0e5-independent-review.json", "`round_2`", "`review_round`", "268c852",
                      "tests/integration/test_linux_adoption_window.py", "/home/yo/projects/runs/full-r1b",
                      "/home/yo/projects/runs/sc6a"):
        assert reference in gate, reference
    assert "Este runbook no autoriza ni programa la ventana: la programa el usuario, como operador de `kza`." in gate
    for step in ("W7", "drop-in sudo", "arranque real hasta READY", "`r1-c6462e5.tar.gz`"):
        assert step in gate, step
    open_reviews = ("re-revisión de la ronda de arreglos de SC-6a", "cbb509c", "9ec7d95",
                    "revisión independiente de la ronda 1 de arreglos de R1-B1/R1-B2",
                    "visto bueno visual del panel de R1-B2 sigue con el usuario")
    handoff = " ".join((REPO / "docs/runbooks/2026-11-05-mac-development-handoff.md").read_text().split())
    deployment = " ".join((REPO / "docs/architecture/DEPLOYMENT.md").read_text().split())
    assert "Esta compuerta no es lo único pendiente antes de la ventana." in gate
    assert "`t040-t041/r1b1-evidence.json`" in gate and "`r1b2.review_round_1`" in gate
    assert "Si alguna de esas revisiones cambia código, R1 se vuelve a construir y a fijar." in gate
    assert "La ventana de R1 sigue sin programarse. Su «Compuerta de programación» (runbook de R1) está cumplida" in handoff
    assert "La compuerta no es lo único pendiente antes de la ventana. Siguen abiertas" in handoff
    assert "la ventana la programa el usuario, como operador, y sigue sin programarse." in deployment
    assert "La compuerta no es lo único pendiente antes de la ventana: siguen abiertas" in deployment
    for name, flat in (("runbook", gate), ("handoff", handoff), ("DEPLOYMENT", deployment)):
        for phrase in open_reviews:
            assert phrase in flat, (name, phrase)
        for stale in ("pasos del operador, no de desarrollo", "Lo que queda son pasos del operador",
                      "queda son pasos del operador"):
            assert stale not in flat, (name, stale)
    assert "Falta el T-A17 compilado en Linux" not in handoff
    assert "Faltaba el T-A17 compilado en Linux (U3b-0e4); ya pasó en el server" in handoff
    assert "Pendiente, antes de la ventana de R1: revisión independiente de la ronda 1" in handoff
    for claim in ("ventana programada", "ventana autorizada", "la ventana está programada"):
        assert claim not in gate.lower(), claim


def test_r1e_the_container_is_e9669ec_and_day14_retires_production_s_legacy_entries():
    """R1E (batch_g_questions_20261118 "R1 base"): both C= lines name the slot 'current' serves since
    2026-10-29 15:50. R1E-2 (r1e_questions_20261118): L4 serves only e9669ec, and day 14 retires the
    frozen state with its three report files, both backup sets and the 63ea8d3 and 2c5adb4 slots."""
    text = r1_runbook.runbook_text()
    assert re.findall(r"(?m)^C=(\S+)$", text) == ["release_20261029T185008966364Z_e9669ec"] * 2
    flat = " ".join(text.split())
    for needle in ("runtime_admission_adoption_container_commit", "runtime_admission_uninstall_legacy_commit",
                   "runtime_admission_legacy_retire_slot_unexpected", "runtime_admission_legacy_retire_superseded",
                   "runtime_admission_uninstall_legacy_retired", "R1E-Q1", "R1E-Q2/Q4/Q5", "R1E-Q3",
                   "`activation-report.json`", "`live-report.json`", "`prepare-report.json`",
                   "`backups/release_20261029T174830709997Z_63ea8d3/databases/`", "`backups/$C/databases/`",
                   "el slot `release_20261029T174830709997Z_63ea8d3` entero",
                   "slot `release_20261028T144002Z_2c5adb4` entero", "veinte copias",
                   "L4 sólo sirve el contenedor de e9669ec", "`release_unproven_copy`",
                   "$R/slots/release_20261028T144002Z_2c5adb4 # todos: No such file or directory"):
        assert needle in flat, needle
    for stale in ("L4 hacia 63ea8d3", "contenedor de 63ea8d3", "Falta que el usuario confirme",
                  "hacen falta además dos decisiones del usuario", "los dos slots con"):
        assert stale not in flat, stale
