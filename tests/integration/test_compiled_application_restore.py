"""The committed artifact must preserve the actual public restore boundary."""

from tests.mocks.compiled_restore import exercise_compiled_restore


def test_committed_bytecode_graph_preserves_current_authority_after_replay(tmp_path):
    proof = exercise_compiled_restore(tmp_path)
    assert proof["accepted_within_scope"] is True
    assert proof["artifact_unchanged"] is True
    assert proof["fixture_and_child_inputs_unchanged"] is True
    assert proof["actual_child_waits_returned"] is True
    assert set(proof["phases"]) == {"generate", "replay"}
    assert all(item["accepted_within_scope"] is True and item["actual_wait_returned"] is True
        and item["containment_only"] is False for item in proof["phases"].values())
