"""SC-2 test helper: build a legacy managed world with the retired transitions.

SC-2 (sc-release-design v3 delegation_off) refuses every delegation grant and
the activation of another slot under a managed head unless
stage_runtime.LEGACY_MANAGED_TRANSITIONS is True. Tests whose subject is the
validation, serving or compaction of an EXISTING owner chain build that chain
inside ``legacy_managed_transitions()`` and then exercise it with the flag
back at its production value (False), so the code under test runs exactly as
an epoch-4 controller would. Whole tests that stay in a legacy world use the
``legacy_managed_transitions`` fixture of tests/conftest.py instead.
"""

from __future__ import annotations

from contextlib import contextmanager

from scripts import stage_runtime as stage


@contextmanager
def legacy_managed_transitions():
    """Set LEGACY_MANAGED_TRANSITIONS for the block and restore the previous value."""
    previous = stage.LEGACY_MANAGED_TRANSITIONS
    stage.LEGACY_MANAGED_TRANSITIONS = True
    try:
        yield stage
    finally:
        stage.LEGACY_MANAGED_TRANSITIONS = previous
