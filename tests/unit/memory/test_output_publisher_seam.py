"""Generic V4 journal phases with real writes and a synthetic admission lease."""
from contextlib import AsyncExitStack
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import pickle
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.core import runtime_admission
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.output_publisher import ALL_MATRIX_PRODUCERS, GrammarSpec, publisher_grammar, publisher_plan
from tests.unit.memory.test_delegated_output_inventory import (  # noqa: F401
    _write,
    decoder,
    delegated_decoder_world,
)
from tests.unit.memory.test_output_registration_v4 import _record, _v4_run

AUDIO = replace(publisher_grammar("audio_input", roles=("single_input", "multi_room")),
                grammar_id="audio_input_publisher/99")
OFFLINE = replace(publisher_grammar("offline_copies"), grammar_id="offline_copies_publisher/99")
SPECS = (AUDIO, OFFLINE)


@pytest.fixture
def local_grammar_table(monkeypatch):
    entries = dict(v4._GRAMMARS)
    monkeypatch.setattr(v4, "_GRAMMAR_ENTRIES", entries)
    monkeypatch.setattr(v4, "_GRAMMARS", MappingProxyType(entries))


@pytest.fixture
async def writer(monkeypatch, decoder, local_grammar_table):  # noqa: F811
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}
    ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(decoder["state"])},
        str(decoder["state"] / "privacy.db"), runtime_admission._config_digest(config))
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    async with AsyncExitStack() as stack:
        admission.install_on(stack)
        coordinator = v2.OutputRegistrationV2(admission, config, key=base._KEY, journal_format=v4.V4_FORMAT,
            extra_grammars=SPECS, publisher_selection={OFFLINE.producer: False})
        admission._output_registration = coordinator
        await coordinator._run(coordinator._start)
        try:
            yield coordinator
        finally:
            await coordinator._disk.aclose()


def _journal_bytes(writer):
    return {path.name: path.read_bytes() for path in writer._directory.iterdir()}


def _compose_legacy_fixture(writer):
    """Provide settled legacy bindings. Only the generic legs are exercised here."""
    writer._accounts_disabled = writer._configuration_disabled = True
    writer._roles.update(dict.fromkeys((
        "privacy_authority", "routines", "events.eraser", "events.retention",
        "suggestions.eraser", "suggestions.retention",
    ), SimpleNamespace(phase="active")))
    writer._handles.update(dict.fromkeys(("lists", "reminders")))


async def test_generic_phases_composition_and_inventory(writer, decoder):  # noqa: F811
    _compose_legacy_fixture(writer)
    plan = publisher_plan("transport_session", roles=("single_input", "multi_room"))
    publisher = await writer.publisher_intent(AUDIO.producer, plan)
    plan["roles"].clear()  # The persisted intent is detached from the caller's dict.
    attached = await writer.attach_publisher(AUDIO.producer)
    assert attached.producer == publisher.producer
    await writer.publisher_disabled(OFFLINE.producer, publisher_plan("declared_root"))
    await writer.composed()
    await attached.opening()
    await publisher.drained({})

    header, _, states, _ = v4.chain(writer._directory)
    assert header["producers"] == list(ALL_MATRIX_PRODUCERS) and len(ALL_MATRIX_PRODUCERS) == 32
    assert header["grammars"] == dict(v4.WRITER_GRAMMARS, **{s.producer: s.grammar_id for s in SPECS})
    assert states[AUDIO.producer]["plan"]["roles"] == ["multi_room", "single_input"]
    assert states[AUDIO.producer]["aggregate"] == {}
    assert states[OFFLINE.producer]["phase"] == "disabled"

    # Join the run to the shared synthetic closed-controller fixture.
    _write(decoder["root"] / "admission/receipts" / f"{header['run_id']}.json", {
        "run_id": header["run_id"], "binding_id": header["binding_id"], "generation": header["generation"],
        "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True),
    })
    before = _journal_bytes(writer)
    value = _record(decoder)
    assert value["phase"] == "observed" and value["stores"] == states
    assert AUDIO.producer not in value["uncovered_producers"]
    assert AUDIO.producer not in value["partial_producers"]
    assert OFFLINE.producer in value["uncovered_producers"]
    assert OFFLINE.producer in value["partial_producers"]
    assert set(value["instrumented_producers"]) >= {AUDIO.producer, OFFLINE.producer}
    composition, = value["material_observations"]
    assert composition["kind"] == "composition"
    assert composition["instrumented"] == sorted(v2._SELECTED | header["grammars"].keys())
    assert not value["global_inventory_complete"] and not value["aggregate_copy_complete"]
    assert _journal_bytes(writer) == before


async def test_composition_rejects_intent_without_attachment(writer):
    _compose_legacy_fixture(writer)
    await writer.publisher_intent(AUDIO.producer,
        publisher_plan("resource_only", roles=("multi_room", "single_input")))
    await writer.publisher_disabled(OFFLINE.producer, publisher_plan("declared_root"))
    with pytest.raises(v2.OutputRegistrationError, match="composition_incomplete"):
        await writer.composed()
    assert all(base._read(path)[0]["kind"] != "composition" for path in writer._directory.glob("[0-9]*.json"))


async def test_drain_without_opening_does_not_publish(writer):
    publisher = await writer.publisher_intent(
        AUDIO.producer, publisher_plan("resource_only", roles=("multi_room", "single_input")),
    )
    await publisher.attached()
    before = _journal_bytes(writer)
    with pytest.raises(RuntimeError, match="transition"):
        await publisher.drained({})
    assert _journal_bytes(writer) == before


async def test_failure_before_intent_never_opens_native_destination(writer):
    target = Path(writer.admission.state_root) / "unopened"
    before = _journal_bytes(writer)

    async def construct():
        await writer.publisher_intent(AUDIO.producer, {"kind": "file_tree", "roles": ["unreviewed"]})
        os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))

    with pytest.raises(RuntimeError, match="publisher_roles"):
        await construct()
    assert not target.exists()
    assert _journal_bytes(writer) == before


def test_unknown_grammar_fails_closed(decoder):  # noqa: F811
    directory = _v4_run(decoder, "4" * 32, [], grammars={AUDIO.producer: AUDIO.grammar_id})
    with pytest.raises(v4.UnsupportedProducerGrammar):
        v4.chain(directory)
    with pytest.raises(v4.UnsupportedProducerGrammar):
        v4.refuse_unknown_grammars(directory.parent)
    value = _record(decoder)
    assert value["phase"] == "unknown" and value["reason"] == "unsupported_producer_grammar"
    assert "stores" not in value


def test_registered_grammar_cannot_replace_an_id_or_skip_common_validation(local_grammar_table):
    def permissive_decoder(event, prior, selected, header):
        return {}

    spec = GrammarSpec(AUDIO.grammar_id, AUDIO.producer, AUDIO.event_kind, permissive_decoder)
    v4.register_grammars((spec,))
    v4.register_grammars((spec,))
    with pytest.raises(RuntimeError, match="publisher_grammar_conflict"):
        v4.register_grammars((OFFLINE, AUDIO))
    assert OFFLINE.grammar_id not in v4._GRAMMARS
    resolved = v4.declared_grammars({"grammars": {spec.producer: spec.grammar_id}})
    event = {"kind": spec.event_kind, "phase": "intent", "plan": {"kind": "sqlite", "oauth_token": "secret"}}
    with pytest.raises(RuntimeError, match="publisher_metadata"):
        resolved[spec.producer].decode_event(event, None, True, {})


@pytest.mark.parametrize("kind", (
    "sqlite", "file_tree", "atomic_json", "transport_session", "resource_only",
    "external_effect", "declared_root", "readonly_input",
))
def test_plan_kinds_are_supported(kind):
    plan = publisher_plan(kind)
    event = {"kind": OFFLINE.event_kind, "phase": "intent", "plan": plan}
    assert OFFLINE.decode_event(event, None, True, {})["plan"] == plan


@pytest.mark.parametrize("metadata", (
    {"api_token": "sensitive"}, {"secret": "sensitive"}, {"audio": b"raw"},
    {"embedding": [1, 2]}, {"repr": "<opaque object>"}, {"descriptor": object()},
    {"unreviewed": "free text"},
))
def test_generic_grammar_rejects_private_and_unreviewed_metadata(metadata):
    event = {"kind": OFFLINE.event_kind, "phase": "intent", "plan": {"kind": "resource_only", **metadata}}
    with pytest.raises(RuntimeError, match="publisher_(plan|metadata)"):
        OFFLINE.decode_event(event, None, True, {})


def test_grammar_spec_is_immutable_and_non_serializable():
    with pytest.raises(FrozenInstanceError):
        AUDIO.producer = "offline_copies"
    with pytest.raises(TypeError, match="output_journal_grammar_not_serializable"):
        pickle.dumps(AUDIO)
