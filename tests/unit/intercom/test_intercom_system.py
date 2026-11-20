"""Intercom routing, playback receipts and optional audio zone metadata."""

import asyncio
from itertools import permutations
from unittest.mock import MagicMock

import pytest

from src.audio.zone_manager import Zone, ZoneManager
from src.core.task_outcome import DispatchState, EffectState, TaskStatus
from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.intercom.intercom_system import AnnouncementPriority, IntercomSystem, ZoneDelivery
from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.household_intents import HouseholdFamily, _quote_spans, classify_household_intent


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
    ("Avisa, en la cocina: Llego Ana", "Llego Ana", ["zone_cocina"]),
    ("Avisa: en la cocina: Llego Ana", "Llego Ana", ["zone_cocina"]),
    ('Anunciá en la cocina: "En la sala: llegó Ana."', '"En la sala: llegó Ana."', ["zone_cocina"]),
    ("Decile al Salón: Llegó Ana", "Llegó Ana", ["zone_sala"]),
    ("Decí en toda la casa que Llegó Ana", "Llegó Ana", ["zone_cocina", "zone_sala"]),
    ("Anunciá: Ana vino al Salón", "Ana vino al Salón", ["zone_cocina", "zone_sala"]),
    ("Anunciá que Ana está en la cocina", "Ana está en la cocina", ["zone_cocina", "zone_sala"]),
    ("Anunciá: Vení, y apagá la luz", "Vení, y apagá la luz", ["zone_cocina", "zone_sala"]),
    ("Anuncia: El pedido esta listo.", "El pedido esta listo.", ["zone_cocina", "zone_sala"]),
    ("Anunciá: «Esperen, por favor.», por favor", "«Esperen, por favor.»", ["zone_cocina", "zone_sala"]),
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


@pytest.mark.parametrize(("opening", "closing"), [("«", "»"), ('"', '"'), ("'", "'")])
@pytest.mark.parametrize("trailing_filler", ["", ", por favor"])
@pytest.mark.parametrize("body", [
    "En la cocina: se corto el agua.",
    "En la cocina se corto el agua.",
    "En la cocina: esperen, por favor.",
    "En la cocina esperen, por favor.",
    "En la cocina:\nesperen.",
    "En la cocina:\nesperen, por favor.",
])
async def test_explicit_quote_preserves_literal_text_and_broadcasts(opening, closing, body, trailing_filler):
    attempts = []

    async def speak(message, zone):
        attempts.append((message, zone))
        return ResponseResult(message, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    intercom = IntercomSystem(speak, room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    intercom.register_zone("cocina", "Cocina")
    intercom.register_zone("sala", "Sala")
    await intercom.start()
    try:
        message = f"{opening}{body}{closing}"
        intent = classify_household_intent(f"Anuncia: {message}{trailing_filler}")
        assert intent.family is HouseholdFamily.INTERCOM
        outcome, _, announcement = await intercom.announce_from_voice(intent.complement)
        assert outcome.task_completed
        assert announcement.receipt()["zones"] == {"zone_cocina": "confirmed", "zone_sala": "confirmed"}
        assert attempts == [(message, "zone_cocina"), (message, "zone_sala")]
    finally:
        await intercom.stop()


@pytest.mark.parametrize("message", [
    "«En la cocina: «esperen», por favor.»",
    "«En la cocina:\n«esperen», por favor.»",
    "«En la cocina: «esperen», dijo Ana, por favor.»",
    '"En la cocina: "esperen", por favor."',
    "'En la cocina: \"esperen\", por favor.'",
])
def test_nested_quote_keeps_outer_message_and_stays_a_broadcast(message):
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    text = f"Anunciá: {message}, por favor"
    assert CommandAcceptanceGate().evaluate(text).accept
    intent = classify_household_intent(text)
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize("separator", [" ", "\n"])
@pytest.mark.parametrize("trailing_filler", ["", ", por favor"])
def test_nested_quote_starting_with_ellipsis_preserves_its_outer_message(quote, separator, trailing_filler):
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    message = f"{quote}En la cocina:{separator}Deciles {quote}... esperen{quote}, por favor.{quote}"
    text = f"Anunciá: {message}{trailing_filler}"
    assert CommandAcceptanceGate().evaluate(text).accept
    intent = classify_household_intent(text)
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


@pytest.mark.parametrize(("depth", "inner_padding"), [
    (1, ("", "")), (2, ("", "")), (3, ("", "")),
    (2, ("\n", " ")), (3, ("\n", " ")),
])
@pytest.mark.parametrize(("opening", "closing"), [('"', '"'), ("'", "'"), ("«", "»")])
@pytest.mark.parametrize("separator", [" ", "\n"])
@pytest.mark.parametrize("ellipsis", ["", "... "])
@pytest.mark.parametrize("trailing_filler", ["", ", por favor"])
def test_balanced_quote_depth_preserves_all_literal_layers(depth, inner_padding, opening, closing, separator, ellipsis,
                                                          trailing_filler):
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    layers = []
    body = f"En la cocina:{separator}{ellipsis}esperen, por favor."
    if depth > 1:
        instruction = f"{ellipsis}esperen"
        if depth == 3:
            layers.append(f"{opening}ya{closing}")
            instruction += f" a que diga {layers[0]}"
        leading, trailing = inner_padding
        layers.insert(0, f"{opening}{leading}{instruction}, por favor{trailing}{closing}")
        body = f"En la cocina:{separator}Deciles {layers[0]}, por favor."
    message = f"{opening}{body}{closing}"
    assert [message[start:end] for start, end in _quote_spans(message)] == [message, *layers]
    text = f"Anunciá: {message}{trailing_filler}"
    assert CommandAcceptanceGate().evaluate(text).accept
    intent = classify_household_intent(text)
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


@pytest.mark.parametrize(("message", "quoted_spans"), [
    ('"hola ", "chau"', ['"hola "', '"chau"']),
    ('"hola ", por favor', ['"hola "']),
    ("'hola ',d'Artagnan vio a O'Neil", ["'hola '"]),
])
def test_ambiguous_quote_closure_requires_two_future_delimiters(message, quoted_spans):
    assert [message[start:end] for start, end in _quote_spans(message)] == quoted_spans


@pytest.mark.parametrize("quote", ['"', "'"])
@pytest.mark.parametrize(("prefix", "separator", "fragments"), [
    ("Anunciá: ", "... y ", ("En la cocina: esperen ", "chau ", "por favor")),
    ("Anunciá: ", "... y ", ("hola ", "chau")),
    ("Anunciá ", ", ", ("hola ", "chau")),
    ("Anunciá: ", "... y ", ("hola ", "... chau")),
    ("Anunciá: ", "… y ", ("hola ", "chau")),
    ("Anunciá: ", "... ", ("En la cocina: esperen ", "por favor")),
    ("Anunciá: ", "… ", ("En la cocina: esperen ", "por favor")),
])
def test_sibling_quotes_stay_separate_and_preserve_broadcast(quote, prefix, separator, fragments):
    siblings = [f"{quote}{fragment}{quote}" for fragment in fragments]
    message = separator.join(siblings)
    assert [message[start:end] for start, end in _quote_spans(message)] == siblings

    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    text = f"{prefix}{message}, por favor"
    assert CommandAcceptanceGate().evaluate(text).accept
    intent = classify_household_intent(text)
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


def test_ellipsis_can_separate_siblings_with_different_quote_types():
    siblings = ['"En la cocina: esperen "', "«chau»", '"por favor"']
    message = "... y ".join(siblings)
    assert [message[start:end] for start, end in _quote_spans(message)] == siblings


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quote_starting_on_the_next_line_stays_literal(quote):
    message = f"{quote}\nEn la cocina: esperen, por favor.{quote}"
    intent = classify_household_intent(f"Anunciá: {message}, por favor")
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


@pytest.mark.parametrize(("outer", "inner"), [('"', "'"), ("'", '"')])
@pytest.mark.parametrize("separator", ["", "\n"])
@pytest.mark.parametrize("trailing_filler", ["", ", por favor"])
@pytest.mark.parametrize("all_contiguous", [False, True])
def test_contiguous_mixed_quote_openings_keep_all_three_layers(
    outer, inner, separator, trailing_filler, all_contiguous,
):
    if all_contiguous:
        innermost = "«En la cocina: esperen, por favor.»"
        body = f"{separator}{innermost}"
    else:
        innermost = "«ya»"
        body = f"En la cocina: esperen a que diga {innermost}, por favor"
    middle = f"{inner}{body}{inner}"
    message = f"{outer}{separator}{middle}, dijo Ana.{outer}"
    assert [message[start:end] for start, end in _quote_spans(message)] == [message, middle, innermost]
    text = f"Anunciá: {message}{trailing_filler}"
    assert CommandAcceptanceGate().evaluate(text).accept
    intent = classify_household_intent(text)
    intercom = IntercomSystem(room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    assert intercom._parse_announcement_command(intent.complement) == (message, ["zone_cocina", "zone_sala"])


@pytest.mark.parametrize("quotes", list(permutations([('"', '"'), ("'", "'"), ("«", "»")])))
@pytest.mark.parametrize("outer_gap", ["", "\n", "... ", "… ", "...\n", "…\n"])
@pytest.mark.parametrize("inner_gap", ["", "\n", "... ", "… ", "...\n", "…\n"])
async def test_ellipsis_between_mixed_quote_openings_keeps_full_broadcast(quotes, outer_gap, inner_gap):
    (outer_open, outer_close), (middle_open, middle_close), (inner_open, inner_close) = quotes
    inner = f"{inner_open}En la cocina: esperen{inner_close}"
    middle = f"{middle_open}{inner_gap}{inner}{middle_close}"
    message = f"{outer_open}{outer_gap}{middle}, por favor.{outer_close}"
    attempts = []

    async def speak(text, zone):
        attempts.append((text, zone))
        return ResponseResult(text, GenerationState.COMPLETE, DeliveryState.COMPLETE)

    intercom = IntercomSystem(speak, room_catalog={"zone_cocina": ("Cocina",), "zone_sala": ("Sala",)})
    intercom.register_zone("cocina", "Cocina")
    intercom.register_zone("sala", "Sala")
    await intercom.start()
    try:
        text = f"Anuncia: {message}"
        assert CommandAcceptanceGate().evaluate(text).accept
        intent = classify_household_intent(text)
        assert intent.family is HouseholdFamily.INTERCOM
        outcome, _, announcement = await intercom.announce_from_voice(intent.complement)
        assert outcome.task_completed
        assert announcement.receipt()["zones"] == {"zone_cocina": "confirmed", "zone_sala": "confirmed"}
        assert attempts == [(message, "zone_cocina"), (message, "zone_sala")]
        assert [message[start:end] for start, end in _quote_spans(message)] == [message, middle, inner]
    finally:
        await intercom.stop()


@pytest.mark.parametrize("empty", ['""', "''"])
@pytest.mark.parametrize("suffix", ["", " ", ", ", "...", "…", "»", "... («", "… —-"])
def test_empty_straight_quotes_are_not_treated_as_nested_openers(empty, suffix):
    assert _quote_spans(empty + suffix) == []


@pytest.mark.parametrize("text", [
    "Avisa, en el fogon: Llego Ana",
    "Avisa, en el ambiente: Llego Ana",
])
async def test_punctuated_unknown_or_ambiguous_destination_never_broadcasts(text):
    speak = MagicMock(side_effect=AssertionError("clarifications have no playback"))
    intercom = IntercomSystem(speak, room_catalog={
        "zone_cocina": ("Cocina", "ambiente"), "zone_sala": ("Sala", "ambiente"),
    })
    intercom.register_zone("cocina", "Cocina")
    intercom.register_zone("sala", "Sala")
    await intercom.start()
    try:
        intent = classify_household_intent(text)
        outcome, response, announcement = await intercom.announce_from_voice(intent.complement)
        assert outcome.status is TaskStatus.NEEDS_INPUT
        assert "sala configurada" in response and announcement is None
        speak.assert_not_called()
        assert intercom.get_history() == []
    finally:
        await intercom.stop()


def test_gate_accepts_noise_words_in_intercom_payload_and_preserves_other_rejections():
    gate = CommandAcceptanceGate()
    assert gate.evaluate("Anuncia: El pedido esta listo.").accept
    assert gate.evaluate("Anunciá: ¡Gracias por ver!").accept
    assert gate.evaluate("Anunciá: «En la cocina: El pedido está listo.»").accept
    rejected = gate.evaluate("Gracias por ver, anunciá que llegó Ana")
    assert not rejected.accept and rejected.reason == "noise_phrase:'gracias por ver'"


@pytest.mark.parametrize("shared_delivery", [DeliveryState.COMPLETE, DeliveryState.FAILED])
async def test_voice_broadcast_keeps_room_receipts_for_one_attempt_per_physical_output(shared_delivery):
    attempts = []

    async def speak(message, zone):
        attempts.append((message, zone))
        delivery = shared_delivery if zone == "zone_cocina" else DeliveryState.COMPLETE
        return ResponseResult(message, GenerationState.COMPLETE, delivery)

    zones = ZoneManager([
        Zone("cocina", "Cocina", 0, 1), Zone("comedor", "Comedor", 1, 1), Zone("sala", "Sala", 2, 2),
    ], ma1260_controller=object())
    intercom = IntercomSystem(speak, zones, room_catalog={
        "zone_cocina": ("Cocina",), "zone_comedor": ("Comedor",), "zone_sala": ("Sala",),
    })
    await intercom.start()
    try:
        outcome, _, announcement = await intercom.announce_from_voice("en toda la casa: ¡Llegó Ana!")
        shared = "confirmed" if shared_delivery is DeliveryState.COMPLETE else "failed"
        assert announcement.receipt()["zones"] == {
            "zone_cocina": shared, "zone_comedor": shared, "zone_sala": "confirmed",
        }
        assert outcome.task_completed is (shared_delivery is DeliveryState.COMPLETE)
        assert attempts == [("¡Llegó Ana!", "zone_cocina"), ("¡Llegó Ana!", "zone_sala")]
    finally:
        await intercom.stop()


async def test_shutdown_settles_active_and_pending_programmatic_announcements():
    started = asyncio.Event()

    async def speak(message, zone):
        started.set()
        await asyncio.Future()

    intercom = IntercomSystem(speak)
    intercom.register_zone("cocina", "Cocina")
    await intercom.start()
    active = await intercom.announce("Activo")
    await started.wait()
    pending = await intercom.announce("Pendiente")
    await intercom.stop()

    await intercom._announcement_queue.join()
    await active.wait_for_delivery()
    await pending.wait_for_delivery()
    assert active.zone_receipts == {"cocina": ZoneDelivery.UNCERTAIN}
    assert pending.zone_receipts == {"cocina": ZoneDelivery.NOT_SENT}
    assert not active.delivery_confirmed and not pending.delivery_confirmed


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
