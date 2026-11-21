"""Exercise all memory codecs through the real V4 journal and nominal owners."""
import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
from hashlib import sha256
import json
import logging
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.application import HttpReasonerOwner, SpeakerModelOwner, STTExecutorOwner, TTSModelOwner, WakeModelOwner
from src.core import runtime_admission
from src.llm.reasoner import FastRouter, HttpReasoner
from src.llm.router_factory import build_llm_router
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.index_erasure import PolicyIndexEraser
from src.memory.memory_manager import MemoryManager
from src.memory.output_registration_memory import (
    MEMORY_GRAMMAR_SPECS,
    bind_memory_family_publishers,
    memory_family_selection,
    resolve_memory_family_plans,
)
from src.memory.privacy_services import PrivacyServices
from src.memory.services import MemoryServices
from src.nlu.llm_router import LLMCommandRouter
from src.obs.diagnostic_erasure import DiagnosticServices, diagnostic_projection_path
from src.obs.metrics_emitter import MetricsEmitter
from src.orchestrator.compactor import Compactor
from src.orchestrator.context_erasure import ContextEraser
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.context_persister import ContextPersister
from src.vectordb.chroma_sync import ChromaSync
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
    "house_vector_index", "personal_memory", "contexts", "model_resources",
    "reasoner_transports", "diagnostics", "logging",
)
_SQLITE_CONNECT = sqlite3.connect


def _digest(text):
    return sha256(str(text).encode()).hexdigest()


def _config(root, enabled):
    return {
        "dashboard": {"enabled": False}, "analytics": {"enabled": False},
        "vectordb": {"path": str(root / "house-index")},
        "memory": {"enabled": enabled, "chroma_path": str(root / "personal-index"),
                   "preferences_path": str(root / "prefs.json"), "privacy_db_path": str(root / "privacy.db")},
        "orchestrator": {"enabled": enabled, "context": {
            "compaction": {"enabled": enabled},
            "persistence": {"enabled": enabled, "base_path": str(root / "contexts")},
        }},
        "reasoner": {"http_base_url": "http://127.0.0.1:8101/v1",
                     "destination": "local" if enabled else "cloud", "cloud": {"consent": False}},
        "router": {"enabled": enabled, "destination": "local"},
        "llm": {"failover": {"endpoints": [{"id": "fast_router_7b", "kind": "fast_router", "priority": 1}],
                             "cooldowns": {"persist_path": str(root / "cooldown.json")}}},
        "nlu": {"llm_router": {"enabled": enabled}},
        "observability": {"enabled": enabled, "metrics_path": str(root / "metrics.jsonl"),
                          "logstash_host": "collector.internal", "logstash_port": 5516},
        "logging": {"file": str(root.parent / "declared.log")},
    }


@pytest.fixture(params=[True, False], ids=["enabled", "history_only"])
async def world(request, monkeypatch, decoder, local_grammar_table):  # noqa: F811
    root = decoder["state"]
    config = _config(root, request.param)
    plans = resolve_memory_family_plans(config, root, console_log=root.parent / "console.log")
    ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
    binding = {"state_root": str(root)}
    admission = runtime_admission.RuntimeAdmission(
        ticket, binding, str(root / "privacy.db"), runtime_admission._config_digest(config),
    )
    # The external launcher lease is synthetic; run the real RuntimeAdmission
    # revalidation and the real writer, reader, locks and owner lifecycles.
    checks, snapshot = [], object()

    def take_snapshot(actual_root, actual_ticket):
        assert actual_root == root and actual_ticket == ticket
        checks.append("full")
        return snapshot

    def validate(actual_root, actual_ticket, actual_snapshot):
        assert actual_root == root and actual_ticket == ticket and actual_snapshot is snapshot
        return binding

    admission_stage = runtime_admission._stage()
    monkeypatch.setattr(admission_stage, "admission_validation_snapshot", take_snapshot)
    monkeypatch.setattr(admission_stage, "validate_admission_live", validate)
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    admission._validation_snapshot = snapshot
    # The privacy authority predates these family bindings. The synthetic
    # legacy observations below exercise only the new family's composition.
    privacy = PrivacyServices(str(root / "privacy.db"))
    with monkeypatch.context() as m:
        m.setattr(sqlite3, "connect", _SQLITE_CONNECT)
        await asyncio.to_thread(privacy.initialize)
    async with AsyncExitStack() as stack:
        stack.push_async_callback(privacy.aclose)
        admission.install_on(stack)
        writer = v2.OutputRegistrationV2(
            admission, config, key=base._KEY, journal_format=v4.V4_FORMAT,
            extra_grammars=MEMORY_GRAMMAR_SPECS, publisher_selection=memory_family_selection(plans),
        )
        admission._output_registration = writer
        await writer._run(writer._start)
        _compose_legacy_fixture(writer)
        try:
            yield writer, plans, checks, config, decoder, privacy
        finally:
            await writer._disk.aclose()
            admission._output_registration = None


def _read(writer, closed_world):
    header, _, _, _ = v4.chain(writer._directory)
    _write(closed_world["root"] / "admission/receipts" / f"{header['run_id']}.json", {
        "run_id": header["run_id"], "binding_id": header["binding_id"], "generation": header["generation"],
        "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True),
    })
    before = _journal_bytes(writer)
    result, = (value for value in v2.read_inventory(closed_world["controller"]) if value["version"] == "v4")
    assert _journal_bytes(writer) == before
    return result


async def test_all_seven_publishers_compose_and_read_real_owner_evidence(world):
    writer, plans, checks, config, closed_world, privacy = world
    root = writer.admission.state_root
    enabled = config["memory"]["enabled"]
    publishers = await bind_memory_family_publishers(writer, plans)
    assert checks == ["full"] * 8
    header, _, states, _ = v4.chain(writer._directory)
    assert tuple(plans) == PRODUCERS
    assert {producer: header["grammars"][producer] for producer in PRODUCERS} == {
        producer: f"{producer}_publisher/1" for producer in PRODUCERS
    }
    assert states["house_vector_index"]["phase"] == "intent" and not (root / "house-index").exists()

    chroma = ChromaSync(config["vectordb"]["path"], "test-model", "cpu")
    contexts = ContextEraser(privacy)
    eraser = PolicyIndexEraser(privacy, {"path": str(root / "personal-index"), "collection": "user_memories_policy_v1"})
    diagnostics = DiagnosticServices(privacy)
    models = {
        "stt_executors": STTExecutorOwner(), "tts_models": TTSModelOwner(),
        "speaker_models": SpeakerModelOwner(), "wake_models": WakeModelOwner(),
    }
    owners = {
        "house_vector_index": {"chroma_sync": chroma},
        "personal_memory": {"policy_index_eraser": eraser},
        "contexts": {"context_eraser": contexts},
        "model_resources": models,
        "diagnostics": {"diagnostic_services": diagnostics},
        "logging": {"bootstrap_logging": logging.getLogger()},
    }
    async with AsyncExitStack() as resources:
        resources.push_async_callback(chroma.aclose)
        resources.push_async_callback(contexts.begin_close)
        resources.push_async_callback(eraser.begin_close)
        resources.push_async_callback(diagnostics.begin_close)
        for owner in models.values():
            resources.push_async_callback(owner.aclose)
        if enabled:
            manager = MemoryManager(None, config["memory"]["preferences_path"], privacy=privacy)
            memory = MemoryServices(manager)
            resources.push_async_callback(memory.aclose)
            owners["personal_memory"].update(memory_services=memory, long_term_memory=manager.long_term)

            persister = ContextPersister(root / "contexts", source_policy=privacy.ledger)
            context_manager = ContextManager(persister=persister, source_policy=privacy.ledger)
            resources.push_async_callback(context_manager.aclose)
            owners["contexts"].update(context_manager=context_manager, context_persister=persister)

            emitter = MetricsEmitter(str(root / "metrics.jsonl"), policy=privacy.ledger)
            diagnostics.emitter = emitter  # DiagnosticServices owns its drain.
            owners["diagnostics"]["metrics_emitter"] = emitter

            http = HttpReasonerOwner()
            fast = FastRouter(destination="local")
            reasoner = HttpReasoner(base_url=config["reasoner"]["http_base_url"], destination="local")
            http.add_model(reasoner)
            command = LLMCommandRouter(fast_router=fast, source_policy=privacy.ledger)
            resources.push_async_callback(http.aclose)
            resources.push_async_callback(fast.aclose)
            resources.push_async_callback(command.aclose)
            owners["reasoner_transports"] = {
                "http_reasoners": http, "fast_router": fast, "compactor": Compactor(reasoner=reasoner),
                "llm_router": build_llm_router(config["llm"]["failover"], {"fast_router_7b": fast}),
                "llm_command_router": command,
            }

        # The Chroma SDK has yet to open. Neither construction nor the reserve
        # and attach steps may create its store.
        for producer, instances in owners.items():
            for role, owner in instances.items():
                await publishers[producer].attach(role, owner)
                if plans[producer]["role_modes"][role] != "lazy":
                    await publishers[producer].opening(role, owner)
                    await publishers[producer].opened(role, owner)
            assert publishers[producer].matches(instances)
        assert chroma._client is None and not (root / "house-index").exists()
        await writer.composed()
        with pytest.raises(RuntimeError, match="publisher_owner"):
            await publishers["house_vector_index"].drained(owners["house_vector_index"])

    for producer, instances in owners.items():
        await publishers[producer].drained(instances)

    value = _read(writer, closed_world)
    assert value["phase"] == "observed"
    assert set(value["instrumented_producers"]) >= set(PRODUCERS)
    assert not value["global_inventory_complete"] and not value["aggregate_copy_complete"]
    assert all(value["stores"][p]["plan"] == plans[p] for p in PRODUCERS)
    house_drain = value["stores"]["house_vector_index"]["aggregate"]["owners"]["chroma_sync"]
    assert house_drain["observation"] == "lazy_unopened" and len(house_drain["instance_sha256"]) == 64
    for producer in owners:
        assert value["stores"][producer]["phase"] == "drained"
        assert producer not in value["uncovered_producers"]
    if not enabled:
        assert "reasoner_transports" in value["uncovered_producers"]
        assert "reasoner_transports" in value["partial_producers"]
        assert value["stores"]["reasoner_transports"]["phase"] == "disabled"
        assert plans["reasoner_transports"]["boundaries"]["cooldown_mode"] == "history_only"
        for producer, role in (("personal_memory", "memory_services"), ("contexts", "context_manager"),
                               ("diagnostics", "metrics_emitter")):
            assert value["stores"][producer]["aggregate"]["owners"][role] == {
                "observation": "disabled", "instance_sha256": None,
            }


async def test_foreign_or_replaced_owner_cannot_open_or_drain(world, monkeypatch):
    writer, plans, _, config, *_ = world
    publisher = (await bind_memory_family_publishers(writer, plans))["house_vector_index"]
    path = config["vectordb"]["path"]
    chroma, replacement = (ChromaSync(path, "test-model", "cpu") for _ in range(2))
    before = _journal_bytes(writer)
    with pytest.raises(RuntimeError, match="publisher_owner"):
        await publisher.attach("chroma_sync", object())
    assert _journal_bytes(writer) == before
    await publisher.attach("chroma_sync", chroma)
    with pytest.raises(RuntimeError, match="publisher_owner"):
        await publisher.opening("chroma_sync", replacement)
    assert not publisher.matches({"chroma_sync": replacement})
    assert v4.chain(writer._directory)[2]["house_vector_index"]["phase"] == "attached"
    assert chroma._client is None and replacement._client is None

    observed = []

    def refuse_construct(owner):
        phase = v4.chain(writer._directory)[2]["house_vector_index"]["phase"]
        observed.append((owner, phase))
        raise RuntimeError("injected SDK constructor failure")

    monkeypatch.setattr(ChromaSync, "_construct", refuse_construct)
    await publisher.opening("chroma_sync", chroma)
    with pytest.raises(RuntimeError, match="chroma_sync_initialization_failed"):
        await asyncio.to_thread(chroma.initialize)
    assert observed == [(chroma, "opening")]
    await chroma.aclose()
    await replacement.aclose()
    # A close after an unconfirmed native opening must leave the row pending.
    with pytest.raises(RuntimeError, match="publisher_owner"):
        await publisher.drained({"chroma_sync": chroma})
    with pytest.raises(RuntimeError, match="publisher_owner"):
        await publisher.drained({"chroma_sync": replacement})
    assert v4.chain(writer._directory)[2]["house_vector_index"]["phase"] == "opening"


async def test_rejects_whole_plan_before_owner_construction(world):
    writer, plans, checks, *_ = world
    plans["logging"]["boundaries"]["stderr"] = "https://person:credential@collector.invalid"
    before = _journal_bytes(writer)
    with pytest.raises(RuntimeError, match="publisher_plan"):
        await bind_memory_family_publishers(writer, plans)
        pytest.fail("native constructors are downstream of successful reservation")
    assert checks == [] and before == _journal_bytes(writer)


@pytest.mark.parametrize(("field", "value"), [
    ("x_api_key", "private-value"), ("user_embedding", {"v": 42}), ("note", "<SDK object at 0x1>"),
    ("instance_sha256", "https://person:credential@collector.invalid"), ("observation", "raw prompt"),
])
async def test_closed_aggregate_rejects_unreviewed_metadata(world, field, value):
    writer, plans, *_ = world
    publisher = await writer.publisher_intent("house_vector_index", plans["house_vector_index"])
    await publisher.attached()
    await publisher.opening()
    before = _journal_bytes(writer)
    evidence = {"observation": "drained", "instance_sha256": "a" * 64, field: value}
    with pytest.raises(RuntimeError, match="publisher_(aggregate|metadata)"):
        await publisher.drained({"owners": {"chroma_sync": evidence}})
    assert _journal_bytes(writer) == before


def test_resolvers_preserve_config_and_separate_destinations(tmp_path, monkeypatch):
    root = tmp_path / "not-created"
    config = _config(root, True)
    config["stt"] = {"model": "./models/whisper", "shadow_engine": "parakeet", "initial_prompt": "test-only-phrase"}
    config["speaker_id"] = {"model": "speechbrain/spkrec-ecapa-voxceleb", "savedir": "model-cache"}
    config["rooms"] = {"enabled": True, "office": {"mic_device_index": 0}, "wake_word": {
        "engine": "porcupine", "porcupine": {"access_key_env": "TEST_ONLY_ACCESS_KEY"},
    }}
    config["reasoner"].update(http_base_url="https://person:credential@collector.invalid/v1?key=private",
                              api_key_env="TEST_ONLY_API_KEY", destination="cloud")
    config["ambient"] = {"enabled": True, "stt": {"engine": "parakeet"}}
    config["reasoner"]["cloud"]["consent"] = True
    frozen = deepcopy(config)

    def no_io(*args, **kwargs):
        pytest.fail("a pure resolver must not probe paths")

    with monkeypatch.context() as m:
        for method in ("resolve", "exists", "stat", "mkdir", "open"):
            m.setattr(Path, method, no_io)
        plans = resolve_memory_family_plans(config, root, console_log=tmp_path / "console.log")

    assert config == frozen and not root.exists()
    serialized = json.dumps(plans)
    for private in ("credential", "collector.invalid", "collector.internal", "TEST_ONLY", "test-only-phrase", str(root)):
        assert private not in serialized
    primary = plans["reasoner_transports"]["boundaries"]["primary"]
    assert primary == {"mode": "blocked", "protocol": "unresolved", "processing": "cloud",
                       "port": None, "origin_sha256": None}
    transport = plans["reasoner_transports"]["boundaries"]
    assert transport["compaction"]["processing"] == "local" and transport["compaction"]["port"] == 8101
    assert transport["cooldown_path_sha256"] == _digest(root / "cooldown.json")
    assert transport["egress_authority"] == "original_owner"

    model = plans["model_resources"]["boundaries"]
    assert model["stt_primary"]["input_kind"] == "readonly_input"
    assert model["stt_shadow"]["sdk_cache"] == model["stt_ambient"]["sdk_cache"] == "unconfirmed"
    assert model["speaker_primary"]["cache_sha256"] == _digest(root / "model-cache/speechbrain_spkrec-ecapa-voxceleb")
    assert model["wake_room"]["input_kind"] == model["wake_parameters"]["input_kind"] == "readonly_input"
    assert model["wake_fallback"]["mode"] == "lazy" and model["tts_secondary"]["mode"] == "disabled"

    diagnostics = plans["diagnostics"]["boundaries"]
    jsonl = diagnostic_projection_path(root / "metrics.jsonl")
    assert diagnostics["jsonl_sha256"] == _digest(jsonl)
    assert diagnostics["rotation_sha256"] == _digest(root / "metrics_policy_v1.jsonl.1")
    assert diagnostics["sqlite_sha256"] == _digest(root / "metrics_policy_v1.jsonl.sqlite3")
    assert diagnostics["logstash"]["status"] == "external_collector_unconfirmed"
    logging_plan = plans["logging"]["boundaries"]
    assert logging_plan["stderr"] == logging_plan["stdout"] == "inherited"
    assert logging_plan["console_redirection_sha256"] == _digest(tmp_path / "console.log")
    assert logging_plan["diagnostic_forwarder"] == diagnostics["logstash"]


@pytest.mark.parametrize(("mode", "withdrawn", "expected"), [
    ("http", False, "configured"), ("http", True, "blocked"),
    ("hermes_cli", False, "configured"), ("hermes_cli", True, "blocked"),
])
def test_cloud_transport_metadata_uses_literal_consent(tmp_path, mode, withdrawn, expected):
    config = {"reasoner": {"http_base_url": "http://127.0.0.1:8210/v1", "mode": mode, "cloud": {"consent": True}}}
    plan = resolve_memory_family_plans(config, tmp_path, cloud_withdrawn=withdrawn)["reasoner_transports"]
    primary = plan["boundaries"]["primary"]
    assert primary["processing"] == "cloud" and primary["mode"] == expected
    assert primary["port"] == (8210 if mode == "http" else None)
    config["reasoner"]["cloud"]["consent"] = "true"
    assert resolve_memory_family_plans(config, tmp_path)["reasoner_transports"]["boundaries"]["primary"]["mode"] == "blocked"
