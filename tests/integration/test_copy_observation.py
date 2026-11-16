"""Real existing copies remain retrospective/non-serving after observation."""
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.mocks.epoch4 import epoch4_checkout
from tests.mocks.copy_observation_scenario import exercise_copy_observation


def test_compiled_existing_copy_observations_preserve_current_authority_without_grants(tmp_path, tmp_path_factory, epoch4_controller, legacy_managed_transitions):
    # SC-2: the material reconciliation world (derived-state grants and runs of derived slots under
    # the managed head) is a legacy world; it runs under the legacy_managed_transitions fixture.
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    # SC-1a: no epoch-5 controller honours a delegation, so the new release and the in-process
    # controller are that revision's epoch-4 build, from a shared checkout outside the task tree.
    repo, revision = epoch4_checkout(tmp_path_factory.mktemp("epoch4"), repo, revision)
    selected = os.environ.get("KZA_COPY_OBSERVATION_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else
        Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = exercise_copy_observation(tmp_path, evidence, repo, new_revision=revision)
    assert proof["complete"] is True and proof["observed_root_count"] == 3
    assert proof["current_F_unchanged"] is True and proof["original_catalog_unchanged"] is True
    assert proof["reader"]["retrospective"] is True and proof["reader"]["non_serving"] is True
    assert proof["aggregate_copy_complete"] is False and proof["F1_delivered"] is False
    assert proof["goal_complete"] is False
