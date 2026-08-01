"""Syncing a dead entity writes a permanently impoverished index."""
import pytest

from scripts.sync_ha_to_chroma import select_syncable


def test_dead_entities_are_separated_from_live_ones():
    entities = [
        {"entity_id": "light.grupo_living", "state": "off"},
        {"entity_id": "light.grupo_cuarto", "state": "unavailable"},
        {"entity_id": "light.grupo_bano", "state": "unknown"},
    ]
    live, dead = select_syncable(entities)
    assert [e["entity_id"] for e in live] == ["light.grupo_living"]
    assert sorted(e["entity_id"] for e in dead) == [
        "light.grupo_bano", "light.grupo_cuarto",
    ]


def test_all_live_returns_no_dead():
    entities = [{"entity_id": "light.grupo_living", "state": "on"}]
    live, dead = select_syncable(entities)
    assert len(live) == 1 and dead == []
