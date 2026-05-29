import asyncio
import time

import pytest

# Skip if chromadb isn't installed locally (same pattern as other vectordb tests).
# ChromaSync imports chromadb at module top-level; server always has it.
pytest.importorskip("chromadb")

from src.vectordb.chroma_sync import ChromaSync


class _SlowSyncChroma(ChromaSync):
    """Override search_command con un sleep síncrono para probar el offload."""

    def __init__(self):
        # No llamamos super().__init__ — no necesitamos chroma real para este test.
        self._search_calls = []

    def search_command(self, query, threshold=0.65, **kwargs):
        time.sleep(0.05)  # simula el encode CPU bloqueante (~48ms)
        self._search_calls.append((query, threshold, kwargs))
        return {"entity_id": "light.living", "similarity": 0.9}


@pytest.mark.asyncio
async def test_asearch_command_runs_off_event_loop():
    chroma = _SlowSyncChroma()

    ticks = 0

    async def _ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1

    result, _ = await asyncio.gather(
        chroma.asearch_command("prendé la luz", 0.65, service_filter="turn_on"),
        _ticker(),
    )

    assert result == {"entity_id": "light.living", "similarity": 0.9}
    assert ticks == 5, "el event loop quedó bloqueado durante el encode"
    assert chroma._search_calls == [
        (
            "prendé la luz",
            0.65,
            {
                "service_filter": "turn_on",
                "query_slots": None,
                "hint_entities": None,
                "prefer_area": None,
            },
        )
    ]
