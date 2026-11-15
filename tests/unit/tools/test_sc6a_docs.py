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
