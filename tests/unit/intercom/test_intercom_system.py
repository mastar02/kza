"""Tests for IntercomSystem — regression coverage for Zone field contract."""
import pytest
from unittest.mock import MagicMock

from src.audio.zone_manager import Zone, ZoneManager


@pytest.mark.asyncio
async def test_intercom_loads_zones_without_crashing_on_missing_fields():
    """Regression 2026-04-24: Zone didn't expose media_player_entity,
    speaker_entity, tts_target → FeatureManager.start logged AttributeError
    non-fatal on every restart. Fix makes them optional (default None).

    This test drives IntercomSystem._load_zones() and asserts it completes
    without AttributeError and stores the zone with None media_player.
    """
    from src.intercom.intercom_system import IntercomSystem

    zone = Zone(
        id="zone_1",
        name="Living Room",
        mic_device_index=0,
        ma1260_zone=1,
    )
    zm = MagicMock(spec=ZoneManager)
    zm.get_all_zones.return_value = {"zone_1": zone}

    intercom = IntercomSystem(zone_manager=zm, ha_client=None)

    await intercom._load_zones()

    assert "zone_1" in intercom._zones
    assert intercom._zones["zone_1"]["media_player"] is None


def test_zone_accepts_optional_intercom_fields():
    """Zone should allow setting intercom fields without errors."""
    z = Zone(
        id="zone_1",
        name="Living",
        mic_device_index=0,
        ma1260_zone=1,
        media_player_entity="media_player.living",
        speaker_entity="switch.speaker_living",
        tts_target="tts.piper",
    )
    assert z.media_player_entity == "media_player.living"
    assert z.speaker_entity == "switch.speaker_living"
    assert z.tts_target == "tts.piper"


def test_zone_defaults_intercom_fields_to_none():
    """Without explicit values, fields default to None (backward compatible)."""
    z = Zone(id="z", name="Z", mic_device_index=0, ma1260_zone=1)
    assert z.media_player_entity is None
    assert z.speaker_entity is None
    assert z.tts_target is None
