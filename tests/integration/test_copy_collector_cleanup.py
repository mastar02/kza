"""Current F authority cleans only S collectors in retained mixed O/O copies."""
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.mocks.copy_cleanup_scenario import exercise_collector_copy_cleanup


def test_current_authority_cleans_mixed_collector_copies_and_drains_cancelled_native(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    selected = os.environ.get("KZA_COPY_CLEANUP_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else
        Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = exercise_collector_copy_cleanup(tmp_path, evidence.resolve(), repo, new_revision=revision)
    assert proof["complete"] is True and proof["current_F_preserved"] is True
    assert proof["catalog_copy_count"] == 8 and proof["aggregate_copy_complete"] is False
    assert proof["F1_delivered"] is False and proof["goal_complete"] is False
    assert proof["cleanup"]["cancelled"]["cancelled_observer"] is True
    assert proof["cleanup"]["cancelled"]["native_returned"] is True
