"""Offline family projections through real V4 journals and read-only catalogs."""

from contextlib import AsyncExitStack
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.audio.native_capture_authority import ManualVoiceCommandAuthorityClient, ZoneSetupAuthorityClient
from src.core import runtime_admission
from src.core.resource_retirement import ResourceSession
from src.memory import copy_catalog
from src.memory import output_registration as base
from src.memory import output_registration_v2 as v2
from src.memory import output_registration_v4 as v4
from src.memory.output_registration_offline import (
    OFFLINE_GRAMMAR_SPECS,
    ForensicDeclarations,
    OfflineCommand,
    bind_offline_family_publishers,
    observe_build_forensic_outputs,
    observe_offline_copies,
    resolve_offline_family_plans,
)
from src.training.habit_dataset_generator import HabitDatasetGenerator
from src.training.nightly_trainer import NightlyConfig, NightlyTrainer
from tests.unit.audio.test_native_capture_authority_client import NativeProtocol
from tests.unit.memory.test_copy_catalog import record  # noqa: F401
from tests.unit.memory.test_copy_observation import observation, published  # noqa: F401
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
from tests.unit.memory.test_output_registration_v4 import _record
from tests.unit.tools.test_calibration_authority_client import ADMIN_TOKEN, CAPABILITY, URL

_SPECS = {spec.producer: spec for spec in OFFLINE_GRAMMAR_SPECS}


def _release_manifest():
    return {"format": "kza-cpython-bytecode-v1", "python": "3.13", "python_magic": "f30d0d0a",
            "git_commit": "a" * 40, "entrypoint": ["python", "-m", "src.main"],
            "external_paths": ["config", "data", "models", "logs"],
            "files": dict.fromkeys(("src/main.pyc", "src/__init__.pyc"), "b" * 64)}


def _checkpoint(runtime, state):
    return {"schema_version": 1, "kind": "kza-compiled-restore-state", "source_commit": "a" * 40,
            "release_sha256": "c" * 64, "helper_manifest_sha256": "d" * 64,
            "runtime_root": str(runtime), "state_root": str(state),
            "payload": {"private_fixture": "FORBIDDEN_CHECKPOINT_VALUE"}, "blob_manifest": {}}


@pytest.fixture
def manifests(tmp_path):
    root = tmp_path.resolve()
    release = root / "build/release.json"
    checkpoint = root / "forensics/checkpoint.json"
    _write(release, _release_manifest())
    _write(checkpoint, _checkpoint(release.parent, root / "current"))
    return ForensicDeclarations((release,), (checkpoint,))


@pytest.fixture
def make_writer(monkeypatch, decoder, local_grammar_table):  # noqa: F811
    """Synthetic admission witness with the same real journal writer as the seam."""
    async def make(stack, config, plans):
        ticket = dict(decoder["header"], admission_root=str(decoder["root"]), run_id=uuid4().hex, generation=3)
        admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(decoder["state"])},
            str(decoder["state"] / "privacy.db"), runtime_admission._config_digest(config))
        monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
        monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
        admission.install_on(stack)
        writer = v2.OutputRegistrationV2(admission, config, key=base._KEY, journal_format=v4.V4_FORMAT,
            extra_grammars=OFFLINE_GRAMMAR_SPECS,
            publisher_selection={producer: plan.selected for producer, plan in plans.items()})
        admission._output_registration = writer
        stack.push_async_callback(writer._disk.aclose)
        await writer._run(writer._start)
        return writer

    return make


def _close_receipt(world, writer):
    header, _, _, _ = v4.chain(writer._directory)
    _write(world["root"] / "admission/receipts" / f"{header['run_id']}.json", {
        "run_id": header["run_id"], "binding_id": header["binding_id"], "generation": header["generation"],
        "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True),
    })


@pytest.mark.parametrize("enabled", (True, False))
async def test_offline_family_journal_composition_and_inventory(make_writer, decoder, manifests, enabled):  # noqa: F811
    state, admission = decoder["state"], decoder["root"]
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}, "training": {
        "nightly": {"enabled": enabled, "data_dir": "data/PRIVATE_DATASETS"}, "habits": {"enabled": False}}}
    dataset = state / "data/PRIVATE_DATASETS/preexisting.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"PRIVATE_DATASET_CONTENT")
    plans = resolve_offline_family_plans(config, state, admission_root=admission,
        capture_command=OfflineCommand("enroll_voice", paths=("data/recorded", "data/profile")),
        export_command=OfflineCommand("setup_spotify", paths=("PRIVATE_OAUTH_LOCATOR",)),
        forensic=manifests)
    training = plans["training"].scope
    assert plans["training"].selected is enabled
    assert training.nightly_data == dataset.parent
    assert len(training.legacy_adapters) == 2
    assert training.habits_data == state / "data/habit_training"
    assert plans["offline_exports_training"].scope.paths == ()
    assert "PRIVATE" not in repr(plans)

    async with AsyncExitStack() as stack:
        writer = await make_writer(stack, config, plans)
        _compose_legacy_fixture(writer)
        publishers = await bind_offline_family_publishers(writer, plans)
        enrollment = ResourceSession(error_prefix="enrollment")
        forensic = await observe_build_forensic_outputs(plans["build_forensic_outputs"])
        copies = await observe_offline_copies(plans["offline_copies"])
        owners = {
            "offline_copies": (copies,),
            "offline_capture_tools": (enrollment,),
            "offline_exports_training": (plans["offline_exports_training"].scope,),
            "build_forensic_outputs": (forensic,),
        }
        if enabled:
            habits = HabitDatasetGenerator(data_dir=str(training.habits_data))
            trainer = NightlyTrainer(config=NightlyConfig(data_dir=str(training.nightly_data),
                conversations_dir=str(training.conversations), output_dir=str(training.nightly_output)),
                habit_generator=habits)
            owners["training"] = (trainer, habits)
            with pytest.raises(RuntimeError, match="offline_owner"):
                await publishers.attach("training", trainer, object())

        for producer, attached in owners.items():
            await publishers.attach(producer, *attached)
            assert publishers.matches_attached_owners(producer, *attached)
        if enabled:
            trainer.habit_generator = None
            assert not publishers.matches_attached_owners("training", trainer, habits)
            trainer.habit_generator = habits
        assert not publishers.matches_attached_owners("offline_capture_tools",
            ResourceSession(error_prefix="enrollment"))
        await writer.composed()
        for producer, attached in owners.items():
            await publishers.opening(producer, *attached)

        with pytest.raises(RuntimeError, match="offline_enrollment_unsettled"):
            await publishers.drained("offline_capture_tools", enrollment)
        with enrollment:
            pass
        if enabled:
            trainer.stop_scheduler()
        for producer, attached in owners.items():
            await publishers.drained(producer, *attached)

        _, _, states, _ = v4.chain(writer._directory)
        assert set(_SPECS) == {"training", "offline_copies", "offline_capture_tools",
                               "offline_exports_training", "build_forensic_outputs"}
        assert [spec.grammar_id for spec in OFFLINE_GRAMMAR_SPECS] == [
            f"{producer}_publisher/1" for producer in _SPECS]
        for producer, plan in plans.items():
            assert states[producer]["phase"] == ("drained" if plan.selected else "disabled")
            assert states[producer]["plan"] == plan.journal_plan()
        assert states["offline_exports_training"]["plan"]["boundary"] == "protected_oauth"
        assert forensic.code_roots == (manifests.code_manifests[0].parent,)
        assert forensic.private_roots == (manifests.private_checkpoints[0].parent,)
        assert set(binding["state_root"] for binding in copies.bindings) == {str(state), str(decoder["old"])}

        _close_receipt(decoder, writer)
        before = _journal_bytes(writer)
        result = _record(decoder)
        assert result["phase"] == "observed" and result["stores"] == states
        assert result["material_observations"][0]["kind"] == "composition"
        assert set(_SPECS) <= set(result["instrumented_producers"])
        assert ("training" in result["partial_producers"]) is not enabled
        assert not result["global_inventory_complete"] and not result["aggregate_copy_complete"]
        assert not result["capability"]
        for raw in before.values():
            assert b"PRIVATE_" not in raw and b"FORBIDDEN_CHECKPOINT_VALUE" not in raw
        assert _journal_bytes(writer) == before
        assert dataset.read_bytes() == b"PRIVATE_DATASET_CONTENT"


async def test_readonly_copies_keep_failed_derivations_and_legacy_slots(tmp_path, record):  # noqa: F811
    root = tmp_path.resolve()
    legacy = root / "slots/old"
    legacy.mkdir(parents=True)
    record["kind"] = "derived_state"
    record["source_snapshot_refs"] = [{"source_binding_id": record["source_binding_ids"][0],
        "snapshot_root": str(root / "unread-snapshot"), "root_identity": record["parent_identity"],
        "manifest_sha256": "e" * 64}]
    extent = {"relative": "PRIVATE_EXTENT.sqlite", "family": "managed_payload", "representation": "sqlite"}
    chain = root / "admission/copies" / record["copy_id"]
    chain.mkdir(parents=True)
    head = copy_catalog._append(chain, record, None)
    failed = dict(record, preparation="failed", error_code="SyntheticFailure",
        extents={copy_catalog._sha(copy_catalog._json(extent)): extent})
    copy_catalog._append(chain, failed, head)
    malformed = chain.parent / "unrecognized"
    malformed.symlink_to(root, target_is_directory=True)
    before = {path.name: path.read_bytes() for path in chain.iterdir()}

    plans = resolve_offline_family_plans({}, root, admission_root=root)
    inventory = await observe_offline_copies(plans["offline_copies"])
    assert inventory.slots == (legacy,)
    observed, = inventory.copies
    assert {key: observed[key] for key in failed} == failed
    assert observed["capability"] is False
    assert inventory.unknown_entries == (malformed,)
    assert "PRIVATE_EXTENT" not in repr(inventory)
    assert not (root / ".admission.lock").exists() and not (root / "unread-snapshot").exists()
    assert not (root / "copy").exists()
    assert {path.name: path.read_bytes() for path in chain.iterdir()} == before


@pytest.mark.parametrize("tool", ("test_voice_command", "zone_setup", "diagnose_wakeword"))
def test_native_cli_plan_validates_original_origin_and_device(tmp_path, tool):
    def resolve(origin, devices):
        return resolve_offline_family_plans({}, tmp_path,
            capture_command=OfflineCommand(tool, device_indices=devices, authority_origin=origin))

    plan = resolve("https://EXAMPLE.test:443", (9,))["offline_capture_tools"]
    assert plan.scope.authority_origin == "https://example.test:443"
    assert plan.journal_plan()["boundary"] == "delegated_target"
    assert "example.test" not in repr(plan)
    with pytest.raises(ValueError, match="calibration_authority_origin_invalid"):
        resolve("https://user:PRIVATE@example.test/", (9,))
    with pytest.raises(ValueError, match="offline_capture_device"):
        resolve("http://localhost", (True,))


@pytest.mark.parametrize("keyword", ("capture_command", "export_command"))
def test_unknown_cli_tool_is_refused(tmp_path, keyword):
    with pytest.raises(ValueError, match="offline_tool"):
        resolve_offline_family_plans({}, tmp_path, **{keyword: OfflineCommand("new_cli")})


@pytest.mark.parametrize("tool", ("inventory_home", "sync_ha_to_chroma", "stage_legacy_memory",
                                  "train_assistant", "train_wakeword", "train_custom_wake"))
def test_export_tools_keep_effective_destination_out_of_journal(tmp_path, tool):
    plan = resolve_offline_family_plans({}, tmp_path,
        export_command=OfflineCommand(tool, ("PRIVATE_ROOT",)))["offline_exports_training"]
    assert plan.scope.paths == (tmp_path / "PRIVATE_ROOT",)
    assert plan.journal_plan()["tool"] == tool
    assert "PRIVATE_ROOT" not in str(plan.journal_plan()) + repr(plan)


async def test_native_capture_requires_the_original_observed_retirement(make_writer, decoder):  # noqa: F811
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}
    plans = resolve_offline_family_plans(config, decoder["state"], capture_command=OfflineCommand(
        "test_voice_command", device_indices=(0,), authority_origin="http://localhost"))
    producer = "offline_capture_tools"
    async with AsyncExitStack() as stack:
        writer = await make_writer(stack, config, plans)
        publishers = await bind_offline_family_publishers(writer, plans)
        authority = ManualVoiceCommandAuthorityClient("http://localhost", "FIXTURE_CREDENTIAL")
        wrong = ZoneSetupAuthorityClient("http://localhost", "FIXTURE_CREDENTIAL")
        try:
            with pytest.raises(RuntimeError, match="offline_owner"):
                await publishers.attach(producer, wrong)
            await publishers.attach(producer, authority)
            with pytest.raises(RuntimeError, match="offline_opening"):
                await publishers.opening(producer, wrong)
            await publishers.opening(producer, authority)
            authority.close()
            with pytest.raises(RuntimeError, match="offline_capture_unsettled"):
                await publishers.drained(producer, authority)
            assert v4.chain(writer._directory)[2][producer]["phase"] == "opening"
            assert all(b"FIXTURE_CREDENTIAL" not in raw for raw in _journal_bytes(writer).values())
        finally:
            authority.close()
            wrong.close()


async def test_native_capture_records_observed_retirement_without_credentials(make_writer, decoder):  # noqa: F811
    config = {"dashboard": {"enabled": False}, "analytics": {"enabled": False}}
    plans = resolve_offline_family_plans(config, decoder["state"], capture_command=OfflineCommand(
        "test_voice_command", device_indices=(0,), authority_origin=URL))
    protocol = NativeProtocol()
    protocol.namespace = "manual-voice-command"
    protocol.reply_kind = "manual_voice_command"
    producer = "offline_capture_tools"
    async with AsyncExitStack() as stack:
        writer = await make_writer(stack, config, plans)
        publishers = await bind_offline_family_publishers(writer, plans)
        authority = ManualVoiceCommandAuthorityClient(URL, ADMIN_TOKEN, session_factory=protocol.session)
        try:
            await publishers.attach(producer, authority)
            await publishers.opening(producer, authority)
            authority.register()
            authority.retire()
        finally:
            authority.close()
        await publishers.drained(producer, authority)
        assert protocol.calls == ["register", "retire"]
        assert v4.chain(writer._directory)[2][producer]["aggregate"] == {"observation": "native_retired"}
        for raw in _journal_bytes(writer).values():
            assert all(value.encode() not in raw for value in (URL, ADMIN_TOKEN, CAPABILITY))


async def test_declared_audit_observation_uses_metadata_after_payload_disappears(tmp_path, observation):  # noqa: F811
    root, payload, directory, _ = observation
    plans = resolve_offline_family_plans({}, tmp_path.resolve(),
        forensic=ForensicDeclarations(private_observations=(directory,)))
    with pytest.raises(ValueError, match="offline_manifest_invalid"):
        await observe_build_forensic_outputs(plans["build_forensic_outputs"])
    published(observation)
    (payload / "retained.bin").unlink()
    inventory = await observe_build_forensic_outputs(plans["build_forensic_outputs"])
    assert inventory.private_roots == (payload,)
    assert inventory.code_roots == ()
    assert not (root / ".admission.lock").exists()


async def test_code_and_forensic_manifests_cannot_be_swapped_nested_or_symlinked(tmp_path, manifests):
    root = tmp_path.resolve()
    release, = manifests.code_manifests
    checkpoint, = manifests.private_checkpoints

    async def observe(code, private):
        plans = resolve_offline_family_plans({}, root, forensic=ForensicDeclarations(code, private))
        return await observe_build_forensic_outputs(plans["build_forensic_outputs"])

    with pytest.raises(ValueError, match="offline_manifest_invalid"):
        await observe((checkpoint,), (release,))
    nested = release.parent / "private/checkpoint.json"
    _write(nested, _checkpoint(release.parent, root / "current"))
    with pytest.raises(ValueError, match="offline_manifest_root_overlap"):
        await observe((release,), (nested,))
    linked = checkpoint.parent / "linked.json"
    linked.symlink_to(checkpoint)
    with pytest.raises(ValueError, match="offline_manifest_invalid"):
        await observe((), (linked,))
    manifest = _release_manifest()
    manifest["files"]["data/PRIVATE_COPY"] = "f" * 64
    _write(release, manifest)
    with pytest.raises(ValueError, match="offline_manifest_invalid"):
        await observe((release,), ())


@pytest.mark.parametrize("producer", tuple(_SPECS))
@pytest.mark.parametrize("extra", ({"raw_audio": "PRIVATE"}, {"user_embedding": "PRIVATE"},
                                   {"x_api_key": "PRIVATE"}, {"note": "<object at 0x1>"}))
def test_each_grammar_rejects_unreviewed_shapes(tmp_path, producer, extra):
    spec = _SPECS[producer]
    plans = resolve_offline_family_plans({"training": {"nightly": {"enabled": True}}}, tmp_path,
        admission_root=tmp_path, capture_command=OfflineCommand("enroll_voice", ("wav", "profile")),
        export_command=OfflineCommand("setup_spotify"),
        forensic=ForensicDeclarations((tmp_path / "release.json",)))
    plan = plans[producer].journal_plan()
    header = {"selected": {producer: True}}
    event = {"kind": spec.event_kind, "phase": "intent", "plan": plan}
    with pytest.raises(RuntimeError, match="offline_plan"):
        spec.decode_event(dict(event, plan={**plan, **extra}), None, True, header)
    prior = None
    for phase in ("intent", "attached", "opening"):
        prior = spec.decode_event(dict(event, phase=phase), prior, True, header)
    with pytest.raises(RuntimeError, match="offline_aggregate"):
        spec.decode_event(dict(event, phase="drained", aggregate=extra), prior, True, header)
    with pytest.raises(RuntimeError, match="offline_aggregate"):
        spec.decode_event(dict(event, phase="drained", aggregate={"observation": "PRIVATE"}), prior, True, header)
