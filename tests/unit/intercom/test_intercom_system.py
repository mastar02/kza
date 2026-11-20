"""Intercom routing, playback receipts and optional audio zone metadata."""

from unittest.mock import MagicMock

import pytest

from src.audio.zone_manager import Zone, ZoneManager
from src.core.task_outcome import DispatchState, EffectState, TaskStatus
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.intercom.intercom_system import AnnouncementPriority, IntercomSystem, ZoneDelivery
from src.nlu.household_intents import HouseholdFamily, classify_household_intent


@pytest.mark.asyncio
async def test_intercom_loads_zones_without_crashing_on_missing_fields():
    """Regression 2026-04-24: Zone didn't expose media_player_entity,
    speaker_entity, tts_target → FeatureManager.start logged AttributeError
    non-fatal on every restart. Fix makes them optional (default None).

    This test drives IntercomSystem._load_zones() and asserts it completes
    without AttributeError and stores the zone with None media_player.
    """
    zone = Zone(
        id="zone_1",
        name="Living Room",
        mic_device_index=0,
        ma1260_zone=1,
    )
    zm = MagicMock(spec=ZoneManager)
    zm.get_all_zones.return_value = {"zone_1": zone}

    intercom = IntercomSystem(zone_manager=zm)

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


@pytest.mark.parametrize(("text", "message", "zones"), [
    ("Avisá en el fogón que Llegó Ana", "Llegó Ana", ["zone_cocina"]),
    ("Decile al Salón: Llegó Ana", "Llegó Ana", ["zone_sala"]),
    ("Decí en toda la casa que Llegó Ana", "Llegó Ana", ["zone_cocina", "zone_sala"]),
    ("Anunciá: Ana vino al Salón", "Ana vino al Salón", ["zone_cocina", "zone_sala"]),
    ("Anunciá que Ana está en la cocina", "Ana está en la cocina", ["zone_cocina", "zone_sala"]),
    ("Anunciá: Vení, y apagá la luz", "Vení, y apagá la luz", ["zone_cocina", "zone_sala"]),
    ("Avisá en kitchen que Llegó Ana", "Llegó Ana", []),
    ("Avisá en otra sala que Llegó Ana", "Llegó Ana", []),
    ("Anunciá en patio que Llegó Ana", "Llegó Ana", []),
    ("Anunciá en el fogón", "", ["zone_cocina"]),
])
def test_voice_resolves_only_configured_room_labels(text, message, zones):
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina", "fogón"), "zone_sala": ("Salón",)})
    intercom.register_zone("patio", "Patio", media_player="media_player.patio")
    intent = classify_household_intent(text)
    assert intent.family is HouseholdFamily.INTERCOM
    assert intercom._parse_announcement_command(intent.complement) == (message, zones)


async def test_broadcast_keeps_confirmed_failed_and_uncertain_zones_without_voice_escalation():
    attempts = []

    async def speak(message, zone):
        attempts.append((message, zone))
        if zone == "zone_pasillo":
            return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.TIMED_OUT, output_started=True)
        if zone == "zone_sala":
            return ResponseResult(message, GenerationState.FAILED, DeliveryState.FAILED)
        return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    intercom = IntercomSystem(speak, room_catalog={
        "zone_cocina": ("Cocina",), "zone_sala": ("Sala",), "zone_pasillo": ("Pasillo",), "zone_patio": ("Patio",),
    })
    for room in ("cocina", "sala", "pasillo"):
        intercom.register_zone(room, room, media_player=f"media_player.{room}")

    await intercom.start()
    try:
        missing, _, rejected = await intercom.announce_from_voice("en un lugar desconocido que Vengan")
        assert (missing.status, missing.dispatch, rejected) == (TaskStatus.NEEDS_INPUT, DispatchState.NOT_SENT, None)
        assert attempts == [] and intercom.get_history() == []

        outcome, response, announcement = await intercom.announce_from_voice("en toda la casa que Hay un incendio")
        assert announcement.priority is AnnouncementPriority.NORMAL
        assert announcement.zone_receipts == {
            "zone_cocina": ZoneDelivery.CONFIRMED, "zone_sala": ZoneDelivery.FAILED,
            "zone_pasillo": ZoneDelivery.UNCERTAIN, "zone_patio": ZoneDelivery.FAILED,
        }
        assert (outcome.status, outcome.dispatch, outcome.effect) == (
            TaskStatus.UNCERTAIN, DispatchState.PARTIAL, EffectState.PARTIAL,
        )
        assert not outcome.task_completed and "1 de 4" in response
        assert attempts == [("Hay un incendio", zone) for zone in ("zone_cocina", "zone_sala", "zone_pasillo")]
        assert intercom.get_history() == [announcement]
    finally:
        await intercom.stop()
