"""SC-6a: the R1 runbook carries the mandatory pinned-archive step before W1 (PS-13).

sc-release-design v3 epochs.r1_rebuild: 'sha256sum "$R1_ARCHIVE"' must equal
the pinned value, else stop. The block itself is run by the rehearsals
(tests/mocks/r1_runbook.py); here its place and shape are pinned.
"""

from __future__ import annotations

from pathlib import Path

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
