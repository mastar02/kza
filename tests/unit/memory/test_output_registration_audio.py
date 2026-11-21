"""Exercise the audio codecs against the real V4 journal and native input owner."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import threading
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.core import runtime_admission
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.output_registration_audio import (
    AUDIO_GRAMMAR_SPECS,
    AudioOwner,
    bind_audio_family_publishers,
    resolve_audio_family_plans,
)
from src.pipeline import audio_loop as single
from src.pipeline.audio_manager import AudioManager
from tests.mocks.producer_erasure_graph import CPUInputBackend
from tests.unit.memory.test_delegated_output_inventory import (  # noqa: F401
    _write,
    decoder,
    delegated_decoder_world,
)
from tests.unit.memory.test_output_publisher_seam import (  # noqa: F401
    _compose_legacy_fixture,
    _journal_bytes,
    local_grammar_table,
)

PRODUCERS = (
    "audio_input", "audio_health", "audio_output", "presence_hardware", "capture", "collectors", "voices",
)
NOMINAL_OWNERS = {
    "audio_input": {"audio_manager", "audio_loop", "owned_input_stream"},
    "audio_health": {"multi_room_audio_loop"},
    "audio_output": {"voice_services", "response_handler", "tts_model_owner"},
    "presence_hardware": {"room_context_manager"},
    "capture": {"capture_eraser", "capture_control"},
    "collectors": {"collector_eraser", "wake_clip_store", "conversation_collector"},
    "voices": {"voice_registry", "voice_ledger", "voice_enrollment"},
}
CONFIG = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}


@pytest.fixture
def audio_writer(decoder, local_grammar_table, monkeypatch):  # noqa: F811
    @asynccontextmanager
    async def open_writer(selection=None):
        ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
        state = decoder["state"]
        admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)}, str(state / "privacy.db"),
                                                        runtime_admission._config_digest(CONFIG))
        monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
        checks = []
        # The controller fixture is synthetic. All journal records, decoding,
        # disk execution and close checks still run through the real coordinator.
        monkeypatch.setattr(admission, "_revalidate_live", lambda: None)
        monkeypatch.setattr(admission, "revalidate", lambda: checks.append("full"))
        async with AsyncExitStack() as stack:
            admission.install_on(stack)
            writer = v2.OutputRegistrationV2(admission, CONFIG, key=base._KEY, journal_format=v4.V4_FORMAT,
                extra_grammars=AUDIO_GRAMMAR_SPECS, publisher_selection=selection)
            admission._output_registration = writer
            await writer._run(writer._start)
            try:
                yield writer, checks
            finally:
                await writer._disk.aclose()

    return open_writer


def aggregate(producer, plan):
    value = {
        "owner_counts": {role: int(role in NOMINAL_OWNERS[producer]) for role in plan["roles"]},
        "owner_state": "drained",
        "owner_replaced": False,
    }
    if producer == "presence_hardware":
        value["amplifier_mode"] = "absent"
    return value


@pytest.mark.parametrize("enabled", [True, False])
async def test_real_journal_composes_and_reads_every_audio_row(audio_writer, decoder, enabled):  # noqa: F811
    selection = dict.fromkeys(PRODUCERS, enabled)
    async with audio_writer(selection) as (writer, checks):
        bindings = await bind_audio_family_publishers(writer, CONFIG, selection=selection)
        assert checks == ["full"]
        assert tuple(bindings) == (PRODUCERS if enabled else ())
        plans = resolve_audio_family_plans(CONFIG, writer.admission.state_root)
        if enabled:
            header, _, states, _ = v4.chain(writer._directory)
            assert {producer: states[producer]["phase"] for producer in PRODUCERS} == dict.fromkeys(PRODUCERS, "intent")
            for producer in PRODUCERS:
                await writer.attach_publisher(producer)
            states = v4.chain(writer._directory)[2]
            assert all(states[producer]["phase"] == "attached" for producer in PRODUCERS)

        _compose_legacy_fixture(writer)
        await writer.composed()
        writer._handles.clear()  # Synthetic settled legacy slots are irrelevant to the audio close.
        writer._accounts_disabled = writer._configuration_disabled = False
        if enabled:
            for producer in PRODUCERS:
                await writer.publisher_opening(producer)
            states = v4.chain(writer._directory)[2]
            assert all(states[producer]["phase"] == "opening" for producer in PRODUCERS)
            for producer in PRODUCERS:
                await writer.publisher_drained(producer, aggregate(producer, plans[producer]))

        header, _, states, _ = v4.chain(writer._directory)
        assert dict(header["grammars"]) == dict(v4.WRITER_GRAMMARS, **{p: f"{p}_publisher/1" for p in PRODUCERS})
        assert tuple(spec.producer for spec in AUDIO_GRAMMAR_SPECS) == PRODUCERS
        for producer in PRODUCERS:
            state = states[producer]
            assert state["producer"] == producer
            assert state["phase"] == ("drained" if enabled else "disabled")
            assert state["plan"] == plans[producer]
            assert state["aggregate"] == (aggregate(producer, plans[producer]) if enabled else None)
        await writer.aclose()
        _write(decoder["root"] / "admission/receipts" / f"{header['run_id']}.json", {
            "run_id": header["run_id"], "binding_id": header["binding_id"], "generation": header["generation"],
            "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True),
        })
        files = _journal_bytes(writer)
        inventory, = (record for record in v2.read_inventory(decoder["controller"]) if record["version"] == "v4")
        assert inventory["phase"] == "observed" and inventory["stores"] == states
        composition, = inventory["material_observations"]
        assert composition["kind"] == "composition"
        assert composition["instrumented"] == sorted(v2._SELECTED | header["grammars"].keys())
        assert set(PRODUCERS) <= set(inventory["instrumented_producers"])
        for key in ("partial_producers", "uncovered_producers"):
            assert set(PRODUCERS).intersection(inventory[key]) == (set() if enabled else set(PRODUCERS))
        assert not inventory["global_inventory_complete"] and not inventory["aggregate_copy_complete"]
        assert _journal_bytes(writer) == files


def test_resolver_pins_reviewed_destinations_without_creating_them(tmp_path):
    root = tmp_path / "absent"
    health = root / "data/audio_health.json"
    plans = resolve_audio_family_plans({"rooms": {"stream_watchdog": {"health_path": str(health)}}}, root)
    assert not root.exists()
    assert plans["audio_health"]["destinations"] == {"health_file": str(health)}
    assert plans["capture"]["destinations"] == {
        "ambient_db": str(root / "data/ambient.db"), "keep_audio_dir": str(root / "data/ambient_audio"),
    }
    assert plans["capture"]["references"] == {"capture_runtime_control": "privacy_authority"}
    assert plans["voices"]["references"] == {"review_clips": "privacy_authority", "enrollment": "resource_only"}
    assert plans["collectors"]["destinations"] == {
        "wake_clips": str(root / "data/wakeword_training/captured"),
        "stt_corrections": str(root / "data/stt_corrections"),
        "conversations": str(root / "data/conversations"),
        "exports": str(root / "data/conversation_exports"),
    }
    for producer in ("audio_input", "audio_output", "presence_hardware"):
        assert set(plans[producer]) == {"kind", "roles"} and plans[producer]["kind"] == "resource_only"


def test_enabled_collector_cannot_be_omitted_from_receipt(tmp_path):
    config = {"training": {"stt_corrections": {"enabled": True}}}
    plan = resolve_audio_family_plans(config, tmp_path)["collectors"]
    spec = next(spec for spec in AUDIO_GRAMMAR_SPECS if spec.producer == "collectors")
    value = aggregate("collectors", plan)
    event = {"kind": spec.event_kind, "phase": "drained", "plan": plan, "aggregate": value}
    header = {"state_root": str(tmp_path)}
    prior = {"phase": "opening", "plan": plan}
    assert plan["enabled_owners"] == {"stt_correction_collector": True, "wake_clip_writer": False}
    with pytest.raises(RuntimeError, match="audio_publisher_owner_counts"):
        spec.decode_event(event, prior, True, header)
    value["owner_counts"]["stt_correction_collector"] = 1
    assert spec.decode_event(event, prior, True, header)["aggregate"] == value


def test_presence_receipt_distinguishes_simulation_from_absent_amplifier(tmp_path):
    plan = resolve_audio_family_plans({}, tmp_path)["presence_hardware"]
    spec = next(spec for spec in AUDIO_GRAMMAR_SPECS if spec.producer == "presence_hardware")
    value = aggregate("presence_hardware", plan)
    value["owner_counts"]["ma1260"] = 1
    event = {"kind": spec.event_kind, "phase": "drained", "plan": plan, "aggregate": value}
    prior = {"phase": "opening", "plan": plan}
    header = {"state_root": str(tmp_path)}
    with pytest.raises(RuntimeError, match="audio_publisher_aggregate"):
        spec.decode_event(event, prior, True, header)
    value["amplifier_mode"] = "simulation"
    assert spec.decode_event(event, prior, True, header)["aggregate"] == value


@pytest.mark.parametrize("path", ["../data/audio_health.json", "/foreign/data/audio_health.json",
                                  "data/other.json", "data/tmp/../audio_health.json", "data/<object at 0x1>"])
def test_resolver_rejects_unreviewed_paths(tmp_path, path):
    with pytest.raises(RuntimeError, match="audio_publisher_destination"):
        resolve_audio_family_plans({"rooms": {"stream_watchdog": {"health_path": path}}}, tmp_path)


@pytest.mark.parametrize("field", ["raw_audio", "user_embedding", "repr"])
@pytest.mark.parametrize("phase", ["intent", "drained"])
def test_each_grammar_rejects_sensitive_fields_at_both_boundaries(tmp_path, phase, field):
    plans = resolve_audio_family_plans({}, tmp_path)
    header = {"state_root": str(tmp_path)}
    for spec in AUDIO_GRAMMAR_SPECS:
        plan = plans[spec.producer]
        event = {"kind": spec.event_kind, "phase": phase, "plan": dict(plan)}
        prior = None
        if phase == "drained":
            prior = {"phase": "opening", "plan": plan}
            event["aggregate"] = aggregate(spec.producer, plan)
        event["plan" if phase == "intent" else "aggregate"][field] = "synthetic-private-marker"
        with pytest.raises(RuntimeError, match="(audio_publisher_(plan|aggregate)|publisher_metadata)"):
            spec.decode_event(event, prior, True, header)


@pytest.mark.parametrize("fault", ["extra_field", "string_value", "alternate_path"])
async def test_invalid_intent_never_reaches_native_open(audio_writer, fault):
    async with audio_writer() as (writer, _):
        root = writer.admission.state_root
        plan = resolve_audio_family_plans({}, root)["audio_health"]
        if fault == "extra_field":
            plan["destinations"]["extra"] = str(root / "data/extra.json")
        elif fault == "string_value":
            plan["kind"] = "<object at 0x1>"
        else:
            plan["destinations"]["health_file"] = str(root / "data/another.json")
        target = root / "native-open"
        before = _journal_bytes(writer)

        with pytest.raises(RuntimeError, match="publisher_plan"):
            await writer.publisher_intent("audio_health", plan)
            target.touch()
        assert not target.exists() and _journal_bytes(writer) == before


async def test_failed_durable_intent_stops_the_original_factory(audio_writer, monkeypatch):
    async with audio_writer() as (writer, _):
        before = _journal_bytes(writer)
        constructed = []

        def fail(event):
            assert event["kind"] == "audio_input_publisher" and event["phase"] == "intent"
            raise OSError("synthetic journal write failure")

        monkeypatch.setattr(writer, "_record", fail)
        with pytest.raises(OSError, match="synthetic journal write failure"):
            await bind_audio_family_publishers(writer, CONFIG)
            constructed.append(AudioManager())
        assert not constructed and _journal_bytes(writer) == before


@pytest.mark.parametrize("invalid", ["replacement", "extra_count", "bool_count", "false_state", "empty"])
async def test_terminal_metadata_fails_closed(audio_writer, invalid):
    async with audio_writer() as (writer, _):
        plan = resolve_audio_family_plans(CONFIG, writer.admission.state_root)["audio_input"]
        handle = await writer.publisher_intent("audio_input", plan)
        await handle.attached()
        await handle.opening()
        value = aggregate("audio_input", plan)
        if invalid == "replacement":
            value["owner_replaced"] = True
        elif invalid == "extra_count":
            value["owner_counts"]["user_embedding"] = 1
        elif invalid == "bool_count":
            value["owner_counts"]["audio_loop"] = True
        elif invalid == "false_state":
            value["owner_state"] = "<object at 0x1>"
        else:
            value["owner_counts"] = dict.fromkeys(value["owner_counts"], 0)
        before = _journal_bytes(writer)
        with pytest.raises(RuntimeError, match="audio_publisher_(aggregate|owner_counts)"):
            await handle.drained(value)
        assert _journal_bytes(writer) == before


async def prepare_input(writer, *, extra=False):
    selection = {producer: producer == "audio_input" for producer in PRODUCERS}
    bindings = await bind_audio_family_publishers(writer, CONFIG, selection=selection)
    manager = AudioManager()
    loop = single.AudioLoop(manager, None, None)
    slots = {"audio_manager": manager}
    binding = bindings["audio_input"]

    def read_graph():
        graph = {"audio_manager": (AudioOwner(slots["audio_manager"], loop._closing),),
                 "audio_loop": (AudioOwner(loop),), "owned_input_stream": (AudioOwner(loop._input),)}
        if extra:
            graph["unexpected"] = (AudioOwner(manager),)
        return graph

    return binding, loop, slots, read_graph


@pytest.mark.parametrize("fault", ["replace_before_open", "replace_after_open", "unknown_type", "unknown_role", "wrong_parent"])
async def test_binding_rejects_live_graph_drift(audio_writer, fault):
    selection = {producer: producer == "audio_input" for producer in PRODUCERS}
    async with audio_writer(selection) as (writer, _):
        binding, loop, slots, read_graph = await prepare_input(writer, extra=fault == "unknown_role")
        try:
            if fault in {"unknown_type", "unknown_role", "wrong_parent"}:
                if fault == "unknown_type":
                    slots["audio_manager"] = object()
                if fault == "wrong_parent":
                    slots["audio_manager"] = AudioManager()
                with pytest.raises(RuntimeError, match="audio_publisher_owner_(type|roles|graph)"):
                    await binding.attached(read_graph)
                return
            await binding.attached(read_graph)
            if fault == "replace_after_open":
                await binding.opening()
            prior = slots["audio_manager"]
            slots["audio_manager"] = AudioManager()
            before = _journal_bytes(writer)
            with pytest.raises(RuntimeError, match="audio_publisher_owner_replaced"):
                if fault == "replace_before_open":
                    await binding.opening()
                else:
                    await binding.drained()
            slots["audio_manager"] = prior
            with pytest.raises(RuntimeError, match="audio_publisher_owner_replaced"):
                binding.require_current()
            assert binding.owner_replaced and _journal_bytes(writer) == before
        finally:
            await loop.begin_close()


async def test_binding_observes_original_strict_input_close_after_waiter_cancellation(audio_writer, monkeypatch):
    selection = {producer: producer == "audio_input" for producer in PRODUCERS}
    async with audio_writer(selection) as (writer, _):
        binding, loop, _, read_graph = await prepare_input(writer)
        sdk = CPUInputBackend()
        monkeypatch.setattr(single.sd, "InputStream", sdk.InputStream)
        entered, release = threading.Event(), threading.Event()
        try:
            await binding.attached(read_graph)
            await binding.opening()
            await loop._input.open()
            stream = sdk.stream
            close = stream.close

            def hold_close(*, ignore_errors):
                entered.set()
                assert release.wait(5)
                close(ignore_errors=ignore_errors)

            stream.close = hold_close
            with pytest.raises(RuntimeError, match="audio_publisher_owner_drain_unconfirmed"):
                await binding.drained()
            assert loop._closing is None and stream.closes == 0

            receipt = loop.begin_close()
            observer = asyncio.create_task(binding.drained())
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                assert not observer.done() and not receipt.done() and not loop._input.retired
                assert v4.chain(writer._directory)[2]["audio_input"]["phase"] == "opening"
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert not receipt.done() and not receipt.cancelled()
            finally:
                release.set()
                await asyncio.gather(observer, return_exceptions=True)
            await binding.drained()
            assert receipt.done() and loop._input.retired and stream.pointer is None
            assert (stream.starts, stream.stops, stream.closes) == (1, 1, 1)
            state = v4.chain(writer._directory)[2]["audio_input"]
            assert state["phase"] == "drained" and state["aggregate"] == aggregate("audio_input", state["plan"])
            await writer.aclose()
        finally:
            release.set()
            await loop.begin_close()
