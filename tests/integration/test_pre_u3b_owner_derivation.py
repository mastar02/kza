"""Pre-U3b managed state derived by current code: lookup index and owner schema additions.

The old side is slot_rollback.managed_revision (the earliest managed-capable
ancestor, chosen with git; 80b47f5 today), which predates both the erasure
lookup index (014b422) and the U3b owner schema (58d7b5c). N is derived from
O's closed state by HEAD code and then run. A payload cut with ordinary O and
latest N is then derived and authorized into R's slot: only the known
additive-schema rule (payload_derivation._privacy_schema_added) admits O's
older privacy schema there.

The scenario never activates R. Phase R cannot run on this old side: every state
derived or closed by U3b code carries the owner schema, and the U3b rule is
that pre-owner code refuses such a state at activation
(runtime_admission_unknown_schema; tests/integration/test_fact_owner_rollback.py).
The full O/N/R/F rehearsal uses a schema-compatible old side instead
(test_compiled_material_reconciliation).
"""
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from scripts.stage_runtime import AdmissionController
from src.memory import output_registration
from tests.mocks.slot_reconciliation import exercise_pre_u3b_derivation, privacy_schema_rows

PRIVACY = "data/graph/privacy.db"
INDEX = {"type": "index", "name": "idx_erasure_target_participant_id", "tbl_name": "erasure_targets",
    "sql": "CREATE INDEX idx_erasure_target_participant_id ON erasure_targets(participant, id)"}
OWNER_TABLES = ("fact_owners", "fact_owner_revisions", "fact_owner_schema")


def test_pre_u3b_state_derives_with_one_lookup_index_and_the_empty_owner_schema(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    revision = os.environ.get("KZA_RECONCILIATION_REVISION") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    selected = os.environ.get("KZA_RECONCILIATION_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else
        Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = exercise_pre_u3b_derivation(tmp_path, evidence.resolve(), repo, new_revision=revision)
    assert proof["complete"] is True
    assert proof["code_pair"]["O"] != proof["code_pair"]["N"]
    images = {branch: json.loads((tmp_path / "closed-snapshots" / branch / "manifest.json").read_text())
        for branch in ("O", "N")}
    schema = {branch: images[branch]["databases"][PRIVACY]["schema"] for branch in images}
    # O predates both additions.
    assert not any(row["name"] == INDEX["name"] for row in schema["O"])
    assert not any(row["tbl_name"] in OWNER_TABLES for row in schema["O"])
    # Every later state carries exactly one copy of the index: the derived
    # candidate as authorized, and N's closed image after its run.
    assert privacy_schema_rows(proof["derived"]["state_root"], INDEX["name"]) == [INDEX]
    assert [row for row in schema["N"] if row["name"] == INDEX["name"]] == [INDEX]
    assert privacy_schema_rows(proof["derived_cut"]["state_root"], INDEX["name"]) == [INDEX]
    # The stage-1 rule admitted the derivation although the derived ledger gained
    # the empty owner schema that O's authority never had.
    assert {"fact_owners", "fact_owner_revisions"} <= set(proof["derived"]["authority_tables"])
    for table in OWNER_TABLES:
        assert sum(1 for row in schema["N"] if row["type"] == "table" and row["name"] == table) == 1
    for table in ("fact_owners", "fact_owner_revisions"):
        assert images["N"]["databases"][PRIVACY]["tables"][table]["rows"] == []
    # Output registration (ef8bc6d) postdates the old side: O's admitted run
    # journals nothing; N's run writes exactly one observed V4 journal.
    runs = {}
    for mode in ("seed-O", "advance-N"):
        parent = json.loads((evidence / (mode + "-parent-receipt.json")).read_text())
        closure = tmp_path / "staging/admission/closures" / (parent["closed_binding_id"] + ".json")
        runs[mode] = json.loads(closure.read_text())["running_head"]
    assert runs["seed-O"]["run_id"] != runs["advance-N"]["run_id"]
    with AdmissionController(tmp_path / "staging") as controller:
        records = output_registration.read_output_inventory(controller)
    assert [record["run_id"] for record in records] == [runs["advance-N"]["run_id"]]
    (record,) = records
    assert record["phase"] == "observed"
    assert record["version"] == "v4" and record["header"]["schema_version"] == 4
    assert (record["header"]["binding_id"], record["header"]["generation"]) == (
        runs["advance-N"]["binding_id"], runs["advance-N"]["generation"])
