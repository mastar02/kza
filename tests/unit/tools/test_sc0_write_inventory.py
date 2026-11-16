"""SC-0's writer inventory sees every table capture_state_authority projects (SC-0b review F1).

The design's partial-projection rule (Q4) and SC-4a's skip list rely on
tools/sc0/write_inventory.projected_tables(); a refactor of the capture that
hides a projected table from it must fail here, not silently shrink the list.
"""
from pathlib import Path

from scripts import stage_runtime as stage
from tools.sc0 import write_inventory

REPO = Path(__file__).resolve().parents[3]

# Measured at eec1792 (before the capture body moved into _capture_state_authority): 61 tables.
PARTIAL = {"conversation_turns_v1", "events_policy_v1", "fact_dependencies", "fact_sources", "facts",
           "policy_index_deletions", "sqlite_sequence", "stt_corrections_v1", "suggestions_policy_v1",
           "utterances", "voice_fingerprints", "voice_group_lineage", "voice_group_members", "voice_groups",
           "wake_clips_v1"}
SCHEMA_MARKERS = {"fact_owner_schema", "voice_attribution_schema", "voice_capture_review_schema",
                  "voice_group_schema", "voice_person_schema"}


def test_projected_tables_are_pinned():
    kinds = write_inventory.projected_tables(REPO)
    by_kind = {kind: {name for name, value in kinds.items() if value == kind}
               for kind in ("partial", "full", "schema_marker")}
    assert set(kinds.values()) == set(by_kind)
    assert by_kind["partial"] == PARTIAL
    assert by_kind["schema_marker"] == SCHEMA_MARKERS == set(stage._SCHEMA_MARKERS)
    assert by_kind["full"] == set(stage._PROTECTED_TABLES)
    assert len(kinds) == 61


def test_the_inventory_reads_the_capture_body_not_only_its_memo_wrapper():
    """fact_dependencies, policy_index_deletions and voice_group_lineage appear only in the body
    (since SC-4b, in its shared selection _authority_table_selection)."""
    kinds = write_inventory.projected_tables(REPO)
    for name in ("fact_dependencies", "policy_index_deletions", "voice_group_lineage"):
        assert kinds[name] == "partial", name
