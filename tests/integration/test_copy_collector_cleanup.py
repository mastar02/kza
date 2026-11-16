"""Current F authority cleans only S collectors in retained mixed O/O copies."""
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.mocks.epoch4 import epoch4_checkout
from tests.mocks.copy_cleanup_scenario import exercise_collector_copy_cleanup


def test_current_authority_cleans_mixed_collector_copies_and_drains_cancelled_native(tmp_path, tmp_path_factory, epoch4_controller, legacy_managed_transitions):
    # SC-2: the material reconciliation world (derived-state grants and runs of derived slots under
    # the managed head) is a legacy world; it runs under the legacy_managed_transitions fixture.
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    # SC-1a: no epoch-5 controller honours a delegation, so the new release and the in-process
    # controller are that revision's epoch-4 build, from a shared checkout outside the task tree.
    repo, revision = epoch4_checkout(tmp_path_factory.mktemp("epoch4"), repo, revision)
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
