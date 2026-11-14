"""U3b-0c identity-clone planning units (owner-compaction design v2 D2, D3, H4, R2; T-U11).

The managed, real-snapshot path (snapshot, journal qualification, VACUUM,
D5 on a real K) runs in tests/unit/tools/test_stage_runtime_compaction.py.
These tests pin the planner's pure rules on manifest-shaped inputs.
"""
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from scripts import stage_runtime as stage
from src.analytics.event_evidence import event_target_id, project_pattern_event
from src.memory import owner_compaction as oc

A, B, H = "/r/slots/a/state", "/r/slots/b/state", "/r/slots/h/state"
K = Path("/r/slots/k/state")


def test_many_to_one_relocation_keeps_each_relative_path():
    roots = (A, B, H)
    assert oc.relocate_path(A + "/data/clips-a", roots, K) == str(K / "data/clips-a")
    assert oc.relocate_path(B + "/data/clips-b", roots, K) == str(K / "data/clips-b")
    assert oc.relocate_path(H, roots, K) == str(K)
    for value in ("/elsewhere/data", A + "x/data", "/r/slots/a"):
        with pytest.raises(oc.CompactionCloneError, match="target_foreign"):
            oc.relocate_path(value, roots, K)


def test_descriptor_relocation_is_limited_to_the_journal_locator_keys():
    roots = (A,)
    assert oc._relocate_descriptor({"kind": "wake_clips", "root": A + "/data/c"}, roots, K) == {
        "kind": "wake_clips", "root": str(K / "data/c")}
    assert oc._relocate_descriptor({"kind": "runtime_producers", "protocol": 1}, roots, K) == {
        "kind": "runtime_producers", "protocol": 1}
    for descriptor in ({"kind": "x", "nested": {"root": A + "/d"}}, {"kind": "x", "base_dir": A + "/d"},
                       {"kind": "x", "roots": [A + "/d"]}):
        with pytest.raises(oc.CompactionCloneError, match="target_unsupported"):
            oc._relocate_descriptor(descriptor, roots, K)


def _snapshot(tables):
    return SimpleNamespace(manifest={"databases": {"data/privacy.db": {"tables": {
        name: {"rows": rows} for name, rows in tables.items()}}}})


def test_absolute_locators_outside_the_relocated_columns_refuse():
    roots = (A, H)
    oc._check_locators(_snapshot({"utterances": [{"id": 1, "audio_path": A + "/data/audio/1.wav"}],
                                  "erasure_targets": [{"id": "t", "descriptor": json.dumps({"root": A})}]}), roots)
    with pytest.raises(oc.CompactionCloneError, match="absolute_locator:data/privacy.db:facts.content"):
        oc._check_locators(_snapshot({"facts": [{"id": "f", "content": "see " + H + "/config"}]}), roots)
    with pytest.raises(oc.CompactionCloneError, match="absolute_locator"):
        oc._check_locators(_snapshot({"utterances": [{"id": 1, "audio_path": "x" + A + "/a.wav"}]}), roots)


def test_evidence_json_is_rewritten_only_through_the_map():
    old, new = "a" * 64, "b" * 64
    text = '[{"target_id": "' + old + '", "event_id": 1}]'
    assert stage.compaction_evidence_json(text, {"c" * 64: new}) == text  # byte-identical when untouched
    assert json.loads(stage.compaction_evidence_json(text, {old: new})) == [{"target_id": new, "event_id": 1}]


def _events_case(entry_target, *, parent_changed=False, parent=True):
    """H with one events target and one suggestion depending on its event 7 (rules a-c)."""
    old_descriptor = {"kind": "events_policy", "path": H + "/data/events.db"}
    new_descriptor = {"kind": "events_policy", "path": str(K / "data/events.db")}
    old_id, new_id = event_target_id(old_descriptor), event_target_id(new_descriptor)
    row = {"id": 7, "timestamp": 100.0, "event_type": "command", "entity_id": "light.x", "action": "turn_on",
           "user_id": None, "expires_at": 500.0, "sources_json": "[]"}
    dependency = project_pattern_event(row, old_id).dependency.to_dict()
    if parent_changed:
        row = dict(row, action="turn_off")
    target_id = {"inventory": old_id, "unknown": "d" * 64}[entry_target]
    evidence = json.dumps([dict(dependency, target_id=target_id)])
    snapshot = SimpleNamespace(manifest={"databases": {
        "data/privacy.db": {"tables": {
            "erasure_targets": {"rows": [{"id": "t1", "participant": "events", "version": 1,
                                          "descriptor": json.dumps(old_descriptor)}]},
            "suggestions_policy_v1": {"rows": [{"id": "s1", "evidence_json": evidence}]}}},
        "data/events.db": {"tables": {"events_policy_v1": {"rows": [row] if parent else []}}}}})
    relocations = {"t1": new_descriptor}
    return snapshot, relocations, {old_id: new_id}


@pytest.mark.parametrize("entry_target, parent, parent_changed, refused", [
    ("inventory", True, False, False),   # (a) the remapped dependency projects exactly
    ("unknown", True, False, False),     # (b) a target unknown to H's inventory is kept as it is
    ("inventory", False, False, False),  # (c) the parent expired: kept, as the live app keeps it
    ("inventory", True, True, True),     # the parent exists with a different dependency: event-id reuse
])
def test_suggestion_evidence_rules(entry_target, parent, parent_changed, refused):
    snapshot, relocations, evidence_map = _events_case(entry_target, parent=parent, parent_changed=parent_changed)
    check = lambda: oc._check_evidence(snapshot, "data/privacy.db", relocations, evidence_map, K)
    if refused:
        with pytest.raises(oc.CompactionCloneError, match="event_lineage_changed"):
            check()
    else:
        check()
