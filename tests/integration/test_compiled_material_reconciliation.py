"""Real old/new bytecode, material ordinary rollback, and protected forward recovery."""
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.mocks.slot_reconciliation import exercise_material_reconciliation


def test_compiled_material_rollback_and_forward_preserve_latest_authority(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    selected = os.environ.get("KZA_RECONCILIATION_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else
        Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = exercise_material_reconciliation(tmp_path, evidence.resolve(), repo, new_revision=revision)
    assert proof["complete"] is True
