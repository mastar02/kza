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


def test_runbooks_name_epoch_4():
    r1 = (REPO / DOCS[1]).read_text()
    assert "`controller_epoch` 4" in r1 and "época 4)" in r1 and "SC-6a" in r1
    assert "`controller_epoch` 3, que" not in r1
    compaction = (REPO / DOCS[2]).read_text()
    assert "`controller_epoch == 4`" in compaction and "`controller_epoch == 3`" not in compaction
