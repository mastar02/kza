"""Isolated bytecode-only D replay child; never imported by its parent harness."""
from __future__ import annotations

# The bootstrap prefix must remain stdlib-only.
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType

CONTROL_FIELDS = frozenset({"schema_version", "executable_literal", "parent_prefix", "base_prefix",
    "python_version", "cache_tag", "python_magic_hex", "dependency_roots", "checkout_root",
    "runtime_root", "release_sha256", "source_commit", "island_root", "island_files", "state_root", "evidence_root"})
GUARDS = [{"participant": "diagnostics", "version": 1},
          {"participant": "streams", "version": 2}, {"participant": "voices", "version": 2}]


def _require(condition, code):
    if not condition:
        raise AssertionError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value, *, exists=True):
    path = Path(value)
    _require(path.is_absolute(), "relative_control_path")
    resolved = path.resolve(strict=exists)
    _require(path == resolved and not path.is_symlink(), "noncanonical_control_path")
    return resolved


def _atomic_json(path, document):
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha(raw)


class ArtifactGuard(importlib.abc.MetaPathFinder):
    """Return only the hash-inspected sourceless spec, never a source fallback."""
    def __init__(self, runtime, files, mode, allowed):
        self.runtime, self.files, self.mode, self.allowed = runtime, files, mode, allowed
        self.imports = {}

    def inspect(self, fullname, spec):
        _require(spec is not None and isinstance(spec.loader, importlib.machinery.SourcelessFileLoader),
                 "src_loader_not_sourceless")
        path = _canonical(spec.origin)
        _require(path.is_file() and path.suffix == ".pyc" and path.is_relative_to(self.runtime),
                 "src_origin_outside_artifact")
        relative, raw = path.relative_to(self.runtime).as_posix(), path.read_bytes()
        _require(self.files.get(relative) == _sha(raw), "src_manifest_hash_mismatch")
        _require(raw[:4] == importlib.util.MAGIC_NUMBER, "src_magic_mismatch")
        self.imports[fullname] = {"relative_origin": relative, "sha256": _sha(raw),
            "loader": type(spec.loader).__name__, "phase": self.mode}
        return path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src" or fullname.startswith("src."):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            self.inspect(fullname, spec)
            return spec
        return None

    def audit(self):
        for entry in sys.path:
            _require(bool(entry), "implicit_cwd_on_sys_path")
            path = Path(entry).resolve()
            _require(any(path == root or path.is_relative_to(root) for root in self.allowed), "unplanned_sys_path")
        for name, module in tuple(sys.modules.items()):
            if name == "src" or name.startswith("src."):
                path = self.inspect(name, module.__spec__)
                _require(Path(module.__file__).resolve() == path, "src_file_spec_disagree")
        _require("src.main" in self.imports and "src.application" in self.imports, "public_graph_not_imported")
        return dict(sorted(self.imports.items()))


def _tripwires():
    hits = []
    def forbidden(*args, **kwargs):
        hits.append("native_or_model_loader")
        raise AssertionError("forbidden_native_or_model_loader")
    def module(name, package=False):
        value = ModuleType(name)
        value.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
        if package:
            value.__path__ = []
        sys.modules[name] = value
        return value
    sound = module("sounddevice")
    sound.PortAudioError = type("PortAudioError", (Exception,), {})
    sound.CallbackStop = type("CallbackStop", (Exception,), {})
    sound.CallbackAbort = type("CallbackAbort", (Exception,), {})
    sound.CallbackFlags = type("CallbackFlags", (), {})
    for name in ("InputStream", "RawInputStream", "OutputStream", "RawOutputStream", "Stream",
                 "play", "rec", "stop", "query_devices", "query_hostapis", "check_input_settings"):
        setattr(sound, name, forbidden)
    module("sentence_transformers", package=True).SentenceTransformer = forbidden
    module("onnx_asr").load_model = forbidden
    module("openwakeword", package=True).Model = forbidden
    module("speechbrain", package=True)
    module("speechbrain.inference", package=True)
    speaker = module("speechbrain.inference.speaker")
    speaker.EncoderClassifier = type("EncoderClassifier", (), {"from_hparams": staticmethod(forbidden)})
    module("torchaudio").list_audio_backends = lambda: []
    return hits, forbidden


def _bootstrap(args):
    control = json.loads(_canonical(args.control).read_text())
    _require(type(control) is dict and set(control) == CONTROL_FIELDS and control["schema_version"] == 1,
             "invalid_control_schema")
    commit = control["source_commit"]
    _require(type(commit) is str and len(commit) == 40 and set(commit) <= set("0123456789abcdef"), "unresolved_source_commit")
    _require(sys.executable == control["executable_literal"], "wrong_literal_interpreter")
    _require(list(sys.version_info[:3]) == control["python_version"] and sys.version_info[:2] == (3, 13),
             "wrong_python_version")
    _require(sys.implementation.cache_tag == control["cache_tag"]
             and importlib.util.MAGIC_NUMBER.hex() == control["python_magic_hex"], "wrong_python_abi")
    _require(sys.flags.isolated and sys.flags.no_site and sys.flags.no_user_site
             and sys.flags.ignore_environment and sys.dont_write_bytecode, "interpreter_not_isolated")
    _require(not any(name == "src" or name.startswith("src.") for name in sys.modules), "src_preloaded")
    runtime, state, evidence = _canonical(args.runtime), _canonical(args.state_root), _canonical(args.evidence_root)
    _require(str(runtime) == control["runtime_root"] and str(state) == control["state_root"]
             and str(evidence) == control["evidence_root"], "control_path_disagreement")
    checkout, island = _canonical(control["checkout_root"]), _canonical(control["island_root"])
    _require(not runtime.is_relative_to(checkout) and not island.is_relative_to(checkout)
             and not state.is_relative_to(checkout), "fixture_inside_checkout")
    _require(not (island / "src").exists() and not list((runtime / "src").rglob("*.py")), "src_source_present")
    _require(not list(island.rglob("conftest.py")) and not list(island.rglob(".env")), "private_fixture_copy")
    for relative, digest in control["island_files"].items():
        path = _canonical(island / relative)
        _require(path.is_relative_to(island) and _sha(path.read_bytes()) == digest, "fixture_hash_mismatch")
    own = Path(__file__).resolve().relative_to(island).as_posix()
    _require(own in control["island_files"], "child_not_hashlisted")
    release_path = runtime / "release.json"
    _require(_sha(release_path.read_bytes()) == control["release_sha256"], "release_hash_mismatch")
    release = json.loads(release_path.read_text())
    _require(release["git_commit"] == commit and release["python_magic"] == control["python_magic_hex"]
             and release["format"] == "kza-cpython-bytecode-v1", "release_candidate_mismatch")
    for relative, digest in release["files"].items():
        path = _canonical(runtime / relative)
        _require(path.is_relative_to(runtime) and _sha(path.read_bytes()) == digest, "artifact_hash_mismatch")
    dependencies = [_canonical(value) for value in control["dependency_roots"]]
    _require(dependencies and all(path.name == "site-packages" and ".venv" in path.parts for path in dependencies),
             "unplanned_dependency_roots")
    base = _canonical(control["base_prefix"])
    guard = ArtifactGuard(runtime, release["files"], args.mode, [runtime, island, base, *dependencies])
    sys.meta_path.insert(0, guard)
    hits, forbidden = _tripwires()
    sys.path[:0] = [str(runtime), str(island), *(str(path) for path in dependencies)]
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true"})
    # CPU numerical dependency is allowed; its loaders are guarded before D imports.
    import torch
    torch.hub.load = forbidden
    torch.load = forbidden
    torch.jit.load = forbidden
    return control, release, guard, hits


def _dto_specs():
    from src.memory.erasure_journal import ErasureParticipant, ErasureStep
    from src.memory.fact_ledger import MemorySource
    from tests.mocks.application_restore import ReplayWitness
    from tests.mocks.integrated_restore import Cohort, DataSnapshot
    from tests.mocks.restore_preservation import ManualSentinels
    specs = {
        "DataSnapshot": (DataSnapshot, ("data", "root_identities", "diagnostic_files", "index_path", "index_collection", "sdk_version")),
        "ReplayWitness": (ReplayWitness, ("root", "schemas", "root_identities", "index_path", "index_collection", "sdk_version", "index_metadata", "index_lineage")),
        "Cohort": (Cohort, ("source", "fact", "ambient_id", "archive_path", "clip_ids", "event_ids", "suggestion_ids", "voice_ids", "user", "browser_client")),
        "ManualSentinels": (ManualSentinels, ("profile_id", "profile_revision", "link_revision", "user_id", "user", "fact_id", "group_request", "actor_subject", "voice_ids", "suggestion_ids", "rows", "config_consent", "deleted_profile_id", "home_token")),
        "ErasureStep": (ErasureStep, ("operation_id", "conversation_id", "participant", "target_id", "version", "descriptor", "attempts", "claim_token", "cursor")),
    }
    return specs, MemorySource, ErasureParticipant


class DataCodec:
    """Closed data protocol preserving tuples and exact binary payloads."""
    def __init__(self, directory, manifest=None):
        self.directory = directory
        self.manifest = {} if manifest is None else manifest
        self.used = set()
        self.specs, self.source, self.participant = _dto_specs()

    def encode(self, value, depth=0):
        _require(depth <= 100, "checkpoint_depth")
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            _require(math.isfinite(value), "checkpoint_nonfinite")
            return value
        if type(value) is bytes:
            _require(len(value) <= 16 * 1024 * 1024, "checkpoint_blob_large")
            digest = _sha(value)
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = self.directory / (digest + ".bin")
            if path.exists():
                _require(not path.is_symlink() and path.read_bytes() == value, "checkpoint_blob_conflict")
            else:
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
            self.manifest[digest] = len(value)
            return {"k": "bytes", "sha256": digest, "size": len(value)}
        if isinstance(value, Path):
            _require(value.is_absolute(), "checkpoint_relative_path")
            return {"k": "path", "v": str(value)}
        if type(value) in (tuple, list):
            return {"k": "tuple" if type(value) is tuple else "list",
                    "v": [self.encode(item, depth + 1) for item in value]}
        if type(value) is dict:
            _require(all(type(key) is str for key in value), "checkpoint_dict_key")
            return {"k": "dict", "v": [[key, self.encode(item, depth + 1)] for key, item in value.items()]}
        if type(value) is self.source:
            return {"k": "dto", "n": "MemorySource", "v": self.encode(value.to_dict(), depth + 1)}
        for name, (kind, fields) in self.specs.items():
            if type(value) is kind:
                data = {field: getattr(value, field) for field in fields}
                if name == "ErasureStep":
                    data["participant"] = value.participant.value
                return {"k": "dto", "n": name, "v": self.encode(data, depth + 1)}
        raise TypeError("unsupported_checkpoint_type")

    def decode(self, value, depth=0):
        _require(depth <= 100, "checkpoint_depth")
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            _require(math.isfinite(value), "checkpoint_nonfinite")
            return value
        _require(type(value) is dict and "k" in value, "checkpoint_tag")
        tag = value["k"]
        if tag == "bytes":
            _require(set(value) == {"k", "sha256", "size"}, "checkpoint_blob_fields")
            digest, size = value["sha256"], value["size"]
            _require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
                     and type(size) is int and 0 <= size <= 16 * 1024 * 1024
                     and self.manifest.get(digest) == size, "checkpoint_blob_manifest")
            path = _canonical(self.directory / (digest + ".bin"))
            raw = path.read_bytes()
            _require(len(raw) == size and _sha(raw) == digest, "checkpoint_blob_hash")
            self.used.add(digest)
            return raw
        if tag in ("tuple", "list", "dict", "path"):
            _require(set(value) == {"k", "v"}, "checkpoint_container_fields")
            if tag == "path":
                _require(type(value["v"]) is str and Path(value["v"]).is_absolute(), "checkpoint_path")
                return Path(value["v"])
            _require(type(value["v"]) is list and len(value["v"]) <= 100000, "checkpoint_container")
            if tag == "dict":
                result = {}
                for pair in value["v"]:
                    _require(type(pair) is list and len(pair) == 2 and type(pair[0]) is str
                             and pair[0] not in result, "checkpoint_duplicate_key")
                    result[pair[0]] = self.decode(pair[1], depth + 1)
                return result
            result = [self.decode(item, depth + 1) for item in value["v"]]
            return tuple(result) if tag == "tuple" else result
        _require(tag == "dto" and set(value) == {"k", "n", "v"}, "checkpoint_dto_tag")
        fields = self.decode(value["v"], depth + 1)
        _require(type(fields) is dict, "checkpoint_dto_fields")
        if value["n"] == "MemorySource":
            _require(set(fields) <= {"capture_id", "conversation_id", "captured_at", "expires_at", "origin", "fact_dependencies"}, "checkpoint_source_fields")
            return self.source.from_dict(fields)
        _require(value["n"] in self.specs, "checkpoint_dto_unknown")
        kind, expected = self.specs[value["n"]]
        _require(set(fields) == set(expected), "checkpoint_dto_fields")
        if value["n"] == "ErasureStep":
            fields["participant"] = self.participant(fields["participant"])
            _require(type(fields["version"]) is int and type(fields["attempts"]) is int, "checkpoint_claim_types")
        return kind(**fields)


PAYLOAD_FIELDS = frozenset({"snapshot", "witness", "subject", "foreign", "manual", "operation",
    "old_claim", "clock_now", "foreign_before", "original_u_source", "old_reader_namespace", "g1_close_proof"})


def _checkpoint(args, control, payload=None):
    path = Path(args.checkpoint)
    excluded = [Path(control[name]) for name in ("state_root", "runtime_root", "island_root", "checkout_root")]
    _require(path.is_absolute() and path.parent.resolve() == path.parent and not path.is_symlink()
             and all(not path.is_relative_to(root) for root in excluded), "checkpoint_location")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    pins = {"schema_version": 1, "kind": "kza-compiled-restore-state",
        "source_commit": control["source_commit"], "runtime_root": args.runtime,
        "state_root": args.state_root, "release_sha256": control["release_sha256"],
        "helper_manifest_sha256": _sha(json.dumps(control["island_files"], sort_keys=True).encode())}
    if payload is not None:
        _require(set(payload) == PAYLOAD_FIELDS and not path.exists(), "checkpoint_payload_schema")
        codec = DataCodec(path.parent / "blobs")
        encoded = codec.encode(payload)
        _require(sum(codec.manifest.values()) <= 64 * 1024 * 1024, "checkpoint_blobs_large")
        return _atomic_json(path, {**pins, "payload": encoded, "blob_manifest": codec.manifest})
    _require(path.stat().st_size <= 64 * 1024 * 1024, "checkpoint_json_large")
    raw = path.read_bytes()
    document = json.loads(raw)
    _require(type(document) is dict and set(document) == set(pins) | {"payload", "blob_manifest"}
             and all(document[key] == value for key, value in pins.items()), "checkpoint_pin_mismatch")
    manifest = document["blob_manifest"]
    _require(type(manifest) is dict and len(manifest) <= 10000
             and all(type(size) is int and 0 <= size <= 16 * 1024 * 1024 for size in manifest.values())
             and sum(manifest.values()) <= 64 * 1024 * 1024, "checkpoint_blob_manifest")
    codec = DataCodec(path.parent / "blobs", manifest)
    decoded = codec.decode(document["payload"])
    _require(type(decoded) is dict and set(decoded) == PAYLOAD_FIELDS
             and codec.used == set(codec.manifest), "checkpoint_payload_schema")
    return decoded, _sha(raw)


def _observed_guards(steps):
    found = [guard for guard in GUARDS if any(step["participant"] == guard["participant"]
        and step["version"] == guard["version"] and step["state"] != "complete" for step in steps)]
    _require(found == GUARDS, "unresolved_guard_missing")
    return found


async def _generate(args, control, guard):
    import pytest
    from src.memory.erasure_journal import ErasureParticipant
    from tests.mocks import application_restore as replay
    from tests.mocks import integrated_restore as data_helpers
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.producer_erasure_graph import NativeGate as ProducerGate
    from tests.mocks.restore_application_graph import application_graph
    from tests.mocks.restore_preservation import assert_manual_sentinels, seed_manual_sentinels
    clock = Clock(time.time())
    subject_gate, unrelated_gate = ProducerGate(), ProducerGate()
    patch = pytest.MonkeyPatch()
    marker = len(MemoryClient._created)
    try:
        async with application_graph(Path(args.state_root), patch, clock,
                subject_gate=subject_gate, unrelated_gate=unrelated_gate) as first:
            guard.audit()
            _graph_contract(first, replay)
            subject, foreign = await replay.seed_all(first)
            claims = []
            actual_claim = first.privacy.ledger.erasures.claim
            def observe_claim(row):
                result = actual_claim(row)
                if result is not None and result.participant is ErasureParticipant.PRODUCERS:
                    claims.append(result)
                return result
            patch.setattr(first.privacy.ledger.erasures, "claim", observe_claim)
            saved, witness = await replay.snapshot_payload(first, subject)
            sources = await replay.manual_voice_inputs(first, foreign)
            manual = await seed_manual_sentinels(first, sources=sources, voice_ids=tuple(foreign.voice_ids),
                suggestion_ids=tuple(foreign.suggestion_ids), root_lease=replay.root_lease(first), delete_profile=True)
            foreign_before = await replay.payload(first, foreign)
            original_u = foreign.source.to_dict()
            operation = await replay.erase_with_real_producer(first, subject, foreign, subject_gate, unrelated_gate)
            await replay.clear_browser(first, subject)
            await replay.settle(first, operation)
            selected = await data_helpers.steps(first, operation)
            _require(any(step["participant"] == "producers" and step["version"] == 1
                         and step["state"] == "complete" for step in selected), "producer_receipt_missing")
            old_claim = next(item for item in reversed(claims) if item.operation_id == operation["id"])
            await replay.assert_subject_erased(first, subject)
            await replay.assert_foreign_unchanged(first, foreign, foreign_before)
            await assert_manual_sentinels(first, manual)
            _require(first.home["service_calls"] == 0, "unexpected_HA_call")
            guards = _observed_guards(selected)
            namespace = first.reader.namespace
            _require(first.graph.producer_erasure.inventory.verify().valid, "G1_inventory_invalid")
            guard.audit()
        _require(first.client.closed and not first.privacy.resource_status()["writer_lease"], "G1_close_incomplete")
        _require(subject_gate.returned.is_set() and unrelated_gate.returned.is_set(), "G1_native_exit_missing")
        MemoryClient.close_since(marker)
        payload = {"snapshot": saved, "witness": witness, "subject": subject, "foreign": foreign,
            "manual": manual, "operation": operation, "old_claim": old_claim, "clock_now": clock.now,
            "foreign_before": foreign_before, "original_u_source": original_u,
            "old_reader_namespace": namespace, "g1_close_proof": {"sdk_clients_closed": True,
                "writer_lease_released": True, "native_gates_returned": True}}
        checkpoint_sha = _checkpoint(args, control, payload)
        return {"checkpoint_sha256": checkpoint_sha, "three_guards": guards,
            "domain_oracles": {"ten_compiled_drivers": True, "same_authority": True,
                "native_source_retirement": True, "S_physically_erased": True,
                "U_exact": True, "manual_exact": True, "no_aggregate_complete": True}}
    finally:
        subject_gate.release.set()
        unrelated_gate.release.set()
        patch.undo()


def _graph_contract(bundle, replay):
    _require(bundle.owner is bundle.graph.erasure_services and bundle.owner.privacy is bundle.privacy,
             "authority_graph_disagreement")
    _require({part.value for part in bundle.owner._drivers} == replay.PARTICIPANTS,
             "participant_inventory_mismatch")
    _require(bundle.graph.producer_erasure.inventory.verify().valid, "producer_inventory_invalid")


async def _replay(args, control, guard):
    import pytest
    from src.memory.erasure_journal import ErasureError, ErasureParticipant, ErasureResult
    from src.memory.prompt_context import SourcesInvalidated
    from src.users.voice_ledger import VoiceLedger
    from tests.mocks import application_restore as replay
    from tests.mocks import integrated_restore as data_helpers
    from tests.mocks.memory_graph import Clock, MemoryClient
    from tests.mocks.restore_application_graph import application_graph
    from tests.mocks.restore_preservation import assert_manual_sentinels

    payload, checkpoint_sha = _checkpoint(args, control)
    _require(payload["g1_close_proof"] == {"sdk_clients_closed": True,
        "writer_lease_released": True, "native_gates_returned": True}, "G1_close_proof_missing")
    saved, witness = payload["snapshot"], payload["witness"]
    subject, foreign, manual = payload["subject"], payload["foreign"], payload["manual"]
    operation, old_claim = payload["operation"], payload["old_claim"]
    root = Path(args.state_root)
    clock = Clock(payload["clock_now"])
    # Only data DTOs survived G1. No application or SDK client exists at replay.
    prepared = await asyncio.to_thread(replay.restore_payload, root, saved, witness)
    _require(prepared.preservation is not None, "offline_preservation_missing")
    reader_checks = []

    async def before_voices(privacy):
        def selected_reviews():
            return {name: [row for row in data_helpers.rows(root / "privacy.db", name)
                if row.get("id", row.get("fingerprint_id")) in subject.voice_ids]
                for name in ("voice_fingerprints", "voice_capture_reviews")}
        before = await privacy.acall(selected_reviews)

        def reject_stale_claim():
            with privacy.ledger._lock:
                selected = data_helpers.stored_steps(root, operation)
                producer, = [step for step in selected
                    if step["participant"] == "producers" and step["version"] == 1]
                _require(not privacy.ledger.erasures.is_current_claim(old_claim), "old_claim_current")
                _require(not privacy.ledger.erasures.finish(old_claim, ErasureResult(True)), "old_claim_accepted")
                _require(data_helpers.stored_steps(root, operation) == selected, "old_claim_mutated_journal")
                return producer

        producer = await privacy.acall(reject_stale_claim)
        _require(producer["state"] == "pending" and producer["claim_token"] is None,
                 "recovered_producer_not_pending")
        _require(producer["last_error"] in {ErasureError.RESTARTED.value, ErasureError.UNAVAILABLE.value},
                 "recovery_reason_missing")
        _require(bool(before["voice_fingerprints"]) and bool(before["voice_capture_reviews"]),
                 "raw_restored_voice_missing")
        _require(all(row["audio_wav"] and row["transcript"] == "Synthetic reviewed clip"
            for row in before["voice_capture_reviews"]), "raw_restored_voice_changed")
        visible = await privacy.acall(VoiceLedger(privacy.ledger).snapshot)
        _require(not set(subject.voice_ids) & {row["id"] for row in visible["fingerprints"]},
                 "reader_exposed_restored_subject")
        _require(await privacy.acall(selected_reviews) == before, "reader_removed_payload_to_hide")
        reader_checks.append(privacy)

    patch = pytest.MonkeyPatch()
    deleting, unlinking = data_helpers.NativeGate(), data_helpers.NativeGate()
    marker = len(MemoryClient._created)
    try:
        data_helpers.install_restore_gates(patch, subject, deleting, unlinking)
        async with AsyncExitStack() as stack:
            building = asyncio.create_task(stack.enter_async_context(application_graph(
                root, patch, clock, gates=(deleting, unlinking), before_voices=before_voices)))
            observing = None
            try:
                await data_helpers.until(lambda: deleting.entered.is_set()
                    or (building.done() and building.exception() is not None))
                if building.done() and building.exception() is not None:
                    building.result()
                _require(deleting.entered.is_set() and bool(deleting.threads)
                         and not deleting.returned.is_set(), "semantic_native_gate_missing")
                selected = await asyncio.to_thread(data_helpers.stored_steps, root, operation)
                _require(any(step["participant"] == "semantic_index" and step["state"] != "complete"
                    for step in selected), "semantic_receipt_before_native_return")
                target = building
                if building.done():
                    ready = building.result()
                    eraser = ready.owner._drivers[ErasureParticipant.SEMANTIC_INDEX].callback.__self__
                    pending = set(ready.memory.manager.long_term._operations) | set(eraser._work._tasks)
                    target = next(task for task in pending if not task.done())
                entered = asyncio.Event()

                async def observe():
                    entered.set()
                    return await asyncio.shield(target)

                observing = asyncio.create_task(observe())
                await entered.wait()
                _require(not observing.done(), "observer_not_waiting")
                observing.cancel()
                await asyncio.gather(observing, return_exceptions=True)
                _require(not deleting.returned.is_set(), "cancel_forged_native_return")
                deleting.release.set()
                await data_helpers.until(lambda: unlinking.entered.is_set()
                    or (building.done() and building.exception() is not None))
                if building.done() and building.exception() is not None:
                    building.result()
                _require(unlinking.entered.is_set() and bool(unlinking.threads)
                         and not unlinking.returned.is_set(), "capture_native_gate_missing")
                _require(Path(subject.archive_path).read_bytes() == saved.data["files"][str(subject.archive_path)],
                         "archive_gone_before_native_return")
                selected = await asyncio.to_thread(data_helpers.stored_steps, root, operation)
                _require(any(step["participant"] == "captures" and step["state"] != "complete"
                    for step in selected), "capture_receipt_before_native_return")
                unlinking.release.set()
                second = await building
                _graph_contract(second, replay)
                before_calls = list(second.models.leaf.calls)
                with pytest.raises(SourcesInvalidated, match="command_sources_invalidated"):
                    await replay.launch(second.graph, subject.source, .125)
                _require(second.models.leaf.calls == before_calls, "revoked_source_reached_model")
                _require(not second.processor.conversation_busy(subject.source.conversation_id),
                         "revoked_source_left_job")
                _require(reader_checks == [second.privacy], "reader_used_other_authority")
                _require(second.reader.namespace == payload["old_reader_namespace"], "reader_namespace_changed")
                await replay.clear_browser(second, subject)
                await replay.settle(second, operation)
                _require(deleting.returned.is_set() and unlinking.returned.is_set(), "G2_native_exit_missing")
                await replay.assert_subject_erased(second, subject)
                with pytest.raises(LookupError, match="voice_capture_unavailable"):
                    await second.voices.acall(second.voices.ledger.review_capture,
                        subject.voice_ids[0], replay.voice_actor(second))
                await replay.assert_foreign_unchanged(second, foreign, payload["foreign_before"], rehydrated=True)
                await assert_manual_sentinels(second, manual)
                _require(foreign.source.to_dict() == payload["original_u_source"], "U_source_changed")
                _require(second.home["service_calls"] == 0, "unexpected_HA_call")
                guards = _observed_guards(await data_helpers.steps(second, operation))
                guard.audit()
            finally:
                deleting.release.set()
                unlinking.release.set()
                if observing is not None:
                    await asyncio.gather(observing, return_exceptions=True)
                await asyncio.gather(building, return_exceptions=True)
        _require(second.client.closed and not second.privacy.resource_status()["writer_lease"],
                 "G2_close_incomplete")
        MemoryClient.close_since(marker)
        _require(_sha(Path(args.checkpoint).read_bytes()) == checkpoint_sha, "checkpoint_rewritten")
        return {"checkpoint_sha256": checkpoint_sha, "three_guards": guards,
            "domain_oracles": {"ten_compiled_drivers": True, "same_authority": True,
                "data_only_offline_restore": True, "stale_claim_rejected_without_mutation": True,
                "reader_filters_raw_present": True, "cancel_observer_waits_real_native": True,
                "revoked_source_rejected_before_model": True, "S_physically_erased": True,
                "U_rehydrated_exact": True, "manual_exact": True, "no_aggregate_complete": True}}
    finally:
        deleting.release.set()
        unlinking.release.set()
        patch.undo()


def _main():
    parser = argparse.ArgumentParser(description="Compiled application restore child")
    parser.add_argument("--mode", required=True, choices=("generate", "replay"))
    parser.add_argument("--control", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    control, release, guard, hits = _bootstrap(args)
    runner = _generate if args.mode == "generate" else _replay
    summary = asyncio.run(runner(args, control, guard))
    origins = guard.audit()
    _require(not hits, "forbidden_native_or_model_call")
    for relative, digest in release["files"].items():
        _require(_sha(_canonical(Path(args.runtime) / relative).read_bytes()) == digest,
                 "artifact_changed_during_phase")
    _require(_sha((Path(args.runtime) / "release.json").read_bytes()) == control["release_sha256"],
             "release_changed_during_phase")
    result = {"schema_version": 1, "accepted_within_scope": True, "mode": args.mode,
        "source_commit": control["source_commit"], "runtime_root": args.runtime,
        "state_root": args.state_root, "pid": os.getpid(), "writer_lease_released": True,
        "sdk_clients_closed": True, "native_gates_returned": True, "zero_HA_service_calls": True,
        "module_origins": origins, "sys_flags": {
            "isolated": int(sys.flags.isolated), "ignore_environment": int(sys.flags.ignore_environment),
            "no_user_site": int(sys.flags.no_user_site), "no_site": int(sys.flags.no_site),
            "dont_write_bytecode": int(sys.dont_write_bytecode)}, **summary}
    _atomic_json(Path(args.evidence_root) / (args.mode + "-result.json"), result)
    print(json.dumps({"mode": args.mode, "accepted_within_scope": True,
        "source_commit": control["source_commit"], "checkpoint_sha256": summary["checkpoint_sha256"]},
        sort_keys=True))


if __name__ == "__main__":
    _main()
