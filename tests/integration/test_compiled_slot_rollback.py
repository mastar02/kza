"""A compiled two-root rehearsal retains latest privacy authority on activation."""
from tests.mocks.slot_rollback import exercise_slot_rollback


def test_compiled_slot_rollback_preserves_latest_privacy_authority(tmp_path):
    exercise_slot_rollback(tmp_path)
