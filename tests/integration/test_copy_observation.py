"""Real existing copies remain retrospective/non-serving after observation."""
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from tests.mocks.copy_observation_scenario import exercise_copy_observation


def test_compiled_existing_copy_observations_preserve_current_authority_without_grants(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
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
