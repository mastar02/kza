"""Explicitly synthetic, timestamped state source with controllable read ownership."""

import asyncio
from copy import deepcopy
from datetime import timedelta

from src.core.task_outcome import CURRENT_REQUEST_ID
from src.home_assistant.ha_client import HAAreaAssignments, HAStatesSnapshot


class CapabilitySource:
    def __init__(self, clock):
        self.clock = clock
        self.states = [{"entity_id": "light.fixture", "state": "on",
                        "attributes": {"supported_color_modes": ["brightness"]}}]
        self.state_reads = self.area_reads = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.failed = False
        self.known_time = True
        self.ttl_s = 6
        self.previous = HAStatesSnapshot(False, None, "ha_states_unavailable")

    async def get_state_snapshot(self):
        assert CURRENT_REQUEST_ID.get() is None
        self.state_reads += 1
        self.entered.set()
        await self.release.wait()
        now = self.clock()
        if self.failed:
            from dataclasses import replace
            return replace(self.previous, available=False, reason="ha_states_unavailable", attempted_at=now)
        observed = now if self.known_time else None
        self.previous = HAStatesSnapshot(True, observed, None, tuple(deepcopy(self.states)),
            observed + timedelta(seconds=self.ttl_s) if observed else None, now)
        return self.previous

    async def get_area_assignments(self):
        self.area_reads += 1
        now = self.clock()
        return HAAreaAssignments(True, now, None, expires_at=now+timedelta(seconds=60), attempted_at=now)
