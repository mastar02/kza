"""Exercise the real application composition, V4 journal and callsite contract."""
import ast
from collections import Counter
from contextlib import AsyncExitStack
from io import StringIO
import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from src import main
from src.application import ApplicationFactories
from src.core import runtime_admission
from src.memory import output_registration as base
from src.memory import output_registration_v4 as v4
from src.memory.output_census import _census
from src.memory.output_publisher import ALL_MATRIX_PRODUCERS, GrammarSpec
from tests.mocks.application_graph import offline_config, offline_home


ROOT = Path(__file__).resolve().parents[2]
_CALLS = {
    "apply_boot_consent_decision", "attach", "basicConfig", "bind", "connect", "initialize",
    "load", "load_earcon", "mkdir", "open", "open_stream", "record_negative_samples",
    "record_positive_samples", "register", "router_factory", "start", "test_connection", "train",
    "write_bytes", "write_text", "atomic_write",
}


def _callsites(source: str) -> Counter[str]:
    sites = Counter()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        callable_name = ast.unparse(node.func)
        name = callable_name.split(".")[-1]
        if isinstance(node.func, ast.BoolOp):
            sites[callable_name] += 1  # Original DI constructor choice.
        elif name in _CALLS or name.startswith(("build_", "create_")) or name[:1].isupper():
            sites[callable_name] += 1
        elif callable_name == "asyncio.to_thread" and node.args:
            worker = ast.unparse(node.args[0])
            if worker.startswith(("build_", "create_")):
                sites[worker] += 1
    return sites


def test_composition_and_offline_callsites_are_classified():
    manifest = json.loads(Path(__file__).with_name("runtime_output_callsites.json").read_text())
    assert set(manifest) == {"src/main.py", "src/application.py", "scripts/train_wakeword.py"}
    for relative, mapping in manifest.items():
        assert all(row is None or row in ALL_MATRIX_PRODUCERS for row in mapping.values())
        observed = _callsites((ROOT / relative).read_text())
        pinned = Counter(callable_name.partition("#")[0] for callable_name in mapping)
        assert not observed - pinned, f"Unclassified output callsites in {relative}: {observed - pinned}"
        assert not pinned - observed, f"Stale output callsites in {relative}: {pinned - observed}"


def test_a_new_constructor_or_file_write_changes_the_callsite_census():
    assert _callsites("sink = NewProducer(); sink.write_bytes(data)") == {
        "NewProducer": 1, "sink.write_bytes": 1,
    }


@pytest.fixture
def install_authority(tmp_path, monkeypatch):
    """Supply synthetic ticket provenance, retaining all real owner and journal operations."""
    state = tmp_path / "state"
    root = tmp_path / "controller"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    (root / "admission").mkdir(parents=True)
    monkeypatch.chdir(state)
    monkeypatch.setattr(base, "_ACTIVE", set())
    monkeypatch.setattr(runtime_admission, "_installed_runs", set())

    # Isolate pytest's handlers while closing the actual handler of this run.
    handler = logging.StreamHandler(StringIO())
    log = logging.getLogger()
    log.addHandler(handler)

    def close_handler():
        handler.close()
        log.removeHandler(handler)

    from src.memory import runtime_output_publishers
    monkeypatch.setattr(runtime_output_publishers, "close_logging", close_handler)

    def install(config, *, runtime_root=None):
        ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
        binding = {"state_root": str(state)}
        if runtime_root is not None:
            binding["runtime_root"] = str(runtime_root)
        admission = runtime_admission.RuntimeAdmission(ticket, binding,
            config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
        admission._validation_snapshot = object()
        monkeypatch.setattr(admission, "revalidate", lambda: None)
        monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
        monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
        monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
        monkeypatch.setattr(main, "verify_runtime_admission", lambda config: admission)
        return admission

    try:
        yield state, install
    finally:
        log.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("multi_room", [False, True])
async def test_managed_graph_declares_all_32_and_observes_original_closes(install_authority, monkeypatch, multi_room):
    state, install = install_authority
    async with offline_home() as (url, home):
        config = offline_config(state / "data", url)
        config["dashboard"]["enabled"] = False
        config["observability"]["enabled"] = False
        config["tts"]["engine"] = "dual"
        runtime_root = None
        if multi_room:
            config["memory"]["enabled"] = True
            config["rooms"] = {
                "enabled": True, "test_room": {"mic_device_index": 0, "mic_usb_port": "3-1", "ma1260_zone": 1},
                "wake_word": {"verification": {"mode": "off"}, "xvf_tuning": {
                    "apply_on_start": True, "params": {"PP_AGCMAXGAIN": [16.0]},
                }},
            }
            config["presence"] = {"enabled": True}
            runtime_root = state.parent / "runtime"
            runtime_root.mkdir()
            (runtime_root / "release.json").write_text(json.dumps({
                "format": "kza-cpython-bytecode-v1", "python": "3.13", "git_commit": "a" * 40,
                "python_magic": "0" * 8, "entrypoint": ["python", "-m", "src.main"],
                "external_paths": ["config", "data", "models", "logs"],
                "files": {"src/main.pyc": "a" * 64, "src/__init__.pyc": "b" * 64},
            }))
        admission = install(config, runtime_root=runtime_root)
        # Stage owns the controller receipt. The in-process synthetic admission
        # verifies the graph and its drains before reaching this boundary.
        from scripts import stage_runtime as stage
        receipts = []
        monkeypatch.setattr(runtime_admission, "_stage", lambda: stage)
        monkeypatch.setattr(stage, "write_admission_receipt", lambda ticket, resources: receipts.append(resources))
        leaves = ApplicationFactories(preflight=lambda: None)
        async with AsyncExitStack() as stack:
            graph = await main.build_application(config, shutdown=stack, factories=leaves,
                                                 verify_only=True, register_runtime_outputs=True)
            writer = admission._output_registration
            header, _, states, _ = v4.chain(writer._directory)
            assert len(header["grammars"]) == 25 and header["selected"]["audio_health"] is multi_room
            assert set(states) >= set(ALL_MATRIX_PRODUCERS)
            old_input = graph.fallback_audio_loop._input
            graph.fallback_audio_loop.pause_admission()
            await graph.fallback_audio_loop.drain_pause()
            graph.fallback_audio_loop.resume_admission()
            assert old_input.retired and graph.fallback_audio_loop._input is not old_input
        header, _, states, _ = v4.chain(writer._directory)
        assert states["audio_input"]["phase"] == "drained"
        assert states["audio_input"]["aggregate"]["owner_replaced"] is False
        assert states["home_integrations"]["aggregate"]["owners"]["ha_client"]["observation"] == "drained"
        assert states["offline_copies"]["aggregate"]["observation"] == "catalog_unresolved"
        if multi_room:
            assert states["audio_health"]["phase"] == "drained"
            assert states["presence_hardware"]["aggregate"]["owner_counts"]["xvf_controller"] == 1
            assert states["presence_hardware"]["aggregate"]["amplifier_mode"] == "simulation"
            assert states["build_forensic_outputs"]["aggregate"] == {"observation": "declared_manifests_observed"}
        assert receipts and home["service_calls"] == 0
        events = (base._read(path)[0] for path in sorted(writer._directory.glob("[0-9]*.json")))
        composition, = tuple(event for event in events if event["kind"] == "composition")
        census = _census({"phase": "observed", "version": "v4", "run_id": header["run_id"],
                          "header": header, "stores": states, "material_observations": (composition,)})
        assert census.registered_producers == 32
        assert census.uncovered_producers == ("offline_copies",)
        assert not census.global_inventory_complete


async def test_failed_intent_never_opens_native(install_authority, monkeypatch):
    state, install = install_authority
    async with offline_home() as (url, home):
        admission = install(offline_config(state / "data", url))

        def fail_record(self, event):
            assert event["phase"] == "intent"
            raise OSError("journal unavailable")

        from src.memory.output_registration_v2 import OutputRegistrationV2
        monkeypatch.setattr(OutputRegistrationV2, "_record", fail_record)
        with pytest.raises(RuntimeError, match="epoch_incomplete"):
            async with AsyncExitStack() as stack:
                await main.build_application(offline_config(state / "data", url), shutdown=stack,
                                             verify_only=True, register_runtime_outputs=True)
        assert admission._output_registration._failed
        assert home["reads"] == 0 and not list((state / "data").glob("*.db"))


async def test_attach_failure_executes_real_close_once(install_authority, monkeypatch):
    from src.home_assistant.ha_client import HomeAssistantClient
    from src.memory.output_registration_v2 import OutputRegistrationV2

    state, install = install_authority
    original_record, original_close = OutputRegistrationV2._record, HomeAssistantClient.close
    closed = []

    def fail_attachment(self, event):
        if event["kind"] == "home_integrations_publisher" and event["phase"] == "attached":
            raise OSError("attach journal unavailable")
        return original_record(self, event)

    async def observe_close(self):
        await original_close(self)
        closed.append(self)

    monkeypatch.setattr(OutputRegistrationV2, "_record", fail_attachment)
    monkeypatch.setattr(HomeAssistantClient, "close", observe_close)
    async with offline_home() as (url, home):
        config = offline_config(state / "data", url)
        config["dashboard"]["enabled"] = False
        install(config)
        with pytest.raises(RuntimeError, match="epoch_incomplete"):
            async with AsyncExitStack() as stack:
                await main.build_application(config, shutdown=stack, verify_only=True, register_runtime_outputs=True)
        client, = closed
        assert client._close_task.done() and client._close_task.exception() is None
        assert home["reads"] == 0


def test_reader_resolves_runtime_grammars_and_refuses_unknown_versions(tmp_path):
    declared = dict(v4.WRITER_GRAMMARS) | {spec.producer: spec.grammar_id for spec in v4.ALL_RUNTIME_EXTRA_GRAMMARS}
    assert len(v4.ALL_RUNTIME_EXTRA_GRAMMARS) == 23
    assert set(v4.declared_grammars({"grammars": declared})) == declared.keys()
    bad = dict(declared, audio_input="audio_input_publisher/97")
    run = tmp_path / ("1" * 32)
    run.mkdir()
    (run / "epoch.json").write_bytes(base._encoded({"grammars": bad}))
    with pytest.raises(v4.UnsupportedProducerGrammar):
        v4.refuse_unknown_grammars(tmp_path)


@pytest.mark.parametrize("metadata", [
    {"raw_audio": 1}, {"user_embedding": "x"}, {"x_api_key": "x"},
    {"levels": {"session_token": 1}}, {"note": "a repr <object at 0x1>"},
    {"note": "a repr <src.fixture.Owner object at 0x1>"},
])
def test_common_metadata_filter_precedes_a_permissive_family_decoder(metadata):
    grammar = v4._Grammar("audio_input_publisher/98", "audio_input", "audio_input_publisher",
        lambda event, prior, selected, header: v4._extra_event(
            GrammarSpec("audio_input_publisher/98", "audio_input", "audio_input_publisher",
                        lambda *args: pytest.fail("family decoder received private metadata")),
            event, prior, selected, header))
    event = {"kind": grammar.event_kind, "phase": "intent", "plan": {"kind": "resource_only", **metadata}}
    with pytest.raises(RuntimeError, match="publisher_metadata"):
        grammar.decode_event(event, None, True, {})
