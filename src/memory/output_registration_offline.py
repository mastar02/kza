"""T024 offline publisher plans and read-only destination observations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING

from src.audio import native_capture_authority as capture
from src.core.paths import TRAINING_OUTPUT_LORA, TRAINING_OUTPUT_NIGHTLY, TRAINING_OUTPUT_WHISPER
from src.core.runtime_admission import _stage
from src.memory import output_registration as base
from src.memory.output_publisher import GenericOutputPublisher, GrammarSpec, publisher_grammar, publisher_plan

if TYPE_CHECKING:
    from src.core.runtime_admission import RuntimeAdmission
    from src.memory.output_registration_v2 import OutputRegistrationV2


_CAPTURE_OWNERS = {
    "test_voice_command": capture.ManualVoiceCommandAuthorityClient,
    "zone_setup": capture.ZoneSetupAuthorityClient,
    "diagnose_wakeword": capture.WakewordDiagnosticsAuthorityClient,
    "train_wakeword": capture.WakewordTrainingAuthorityClient,
    "train_nexa": capture.WakewordTrainingAuthorityClient,
    "train_che_bo": capture.WakewordTrainingAuthorityClient,
    "acoustic_calibration": capture.CalibrationAuthorityClient,
}
_CAPTURE_BOUNDARIES = dict.fromkeys(_CAPTURE_OWNERS, "delegated_target") | {"enroll_voice": "recorded_input"}


def _capture_retired(client: capture.NativeCaptureAuthorityClient) -> bool:
    return client._retired and client._closed and not client._uncertain


class OfflineCaptureOwners:
    """Retain every lazy client of a single delegated capture invocation.

    The original LiveWakewordCapture closes the stream, drains admitted work
    and retires each remote capability. Standalone commands have no local V4
    admission and use this scope to observe those exact owner receipts.
    """

    def __init__(self, tool: str, origin: str, factory: Callable[[], capture.NativeCaptureAuthorityClient]) -> None:
        self._nominal = _CAPTURE_OWNERS[tool]
        self._origin = capture._origin(origin)
        self._factory = factory
        self._clients: list[capture.NativeCaptureAuthorityClient] = []

    def create(self) -> capture.NativeCaptureAuthorityClient:
        """Attach a fresh nominal client before passing it to the recorder."""
        client = self._factory()
        base._require(not any(client is previous for previous in self._clients), "offline_capture_duplicate_client")
        try:
            base._require(type(client) is self._nominal and client._origin == self._origin
                          and client._capability is None and not client._closed, "offline_capture_owner")
        except BaseException:
            client.close()
            raise
        self._clients.append(client)
        return client

    def require_retired(self) -> None:
        """Verify the original native-retirement receipt for every client."""
        base._require(all(_capture_retired(client) for client in self._clients), "offline_capture_unsettled")


_CAPTURE_MODES = {
    "train_wakeword": ("record", "test"),
    "train_nexa": ("record-positive", "record-negative", "test"),
    "train_che_bo": ("record-positive", "record-negative", "test"),
    "acoustic_calibration": ("capture",),
}
_EXPORT_BOUNDARIES = {
    "inventory_home": "private_export",
    "sync_ha_to_chroma": "ha_index",
    "setup_spotify": "protected_oauth",
    "stage_legacy_memory": "legacy_staging",
    "train_assistant": "training_export",
    "train_wakeword": "wakeword_training",
    "train_custom_wake": "wakeword_training",
}
_EXPORT_MODES = {"train_wakeword": ("train",)}
_ROW_LAYOUTS = {
    "training": ("file_tree", ("habit_dataset_generator", "nightly_trainer")),
    "offline_copies": ("readonly_input", ("admission_controller", "copy_reservation", "derivation", "snapshot")),
    "offline_capture_tools": ("resource_only", tuple(_CAPTURE_BOUNDARIES)),
    "offline_exports_training": ("declared_root", tuple(_EXPORT_BOUNDARIES)),
    "build_forensic_outputs": ("declared_root", ("code_artifacts", "private_forensics")),
}


def _journal_plan(producer: str, tool: str = "none", mode: str | None = None) -> dict:
    kind, roles = _ROW_LAYOUTS[producer]
    if producer == "training":
        details = {"prior_outputs": "retain", "sdk_work": "external_private", "child_work": "owner_scoped"}
    elif producer == "offline_copies":
        details = {"admission_catalogs": "readonly", "legacy_roots": "retain"}
    elif producer == "offline_capture_tools":
        details = {"tool": tool, "boundary": "none" if tool == "none" else _CAPTURE_BOUNDARIES[tool]}
    elif producer == "offline_exports_training":
        details = {"tool": tool, "boundary": "none" if tool == "none" else _EXPORT_BOUNDARIES[tool]}
    else:
        details = {"code": "release_manifest", "forensics": "private_manifest"}
    if mode is not None:
        details["mode"] = mode
    return publisher_plan(kind, roles=roles, **details)


def _offline_grammar(producer: str) -> GrammarSpec:
    tool_labels = {
        "offline_capture_tools": _CAPTURE_BOUNDARIES,
        "offline_exports_training": _EXPORT_BOUNDARIES,
    }.get(producer, {})
    tool_modes = _CAPTURE_MODES if producer == "offline_capture_tools" else _EXPORT_MODES
    observations = {
        "training": {"instance_reported", "blocked_instance"},
        "offline_copies": {"catalog_observed", "catalog_unresolved"},
        "offline_exports_training": {"invocation_reported"},
        "build_forensic_outputs": {"declared_manifests_observed"},
    }

    def plan_shape(plan: dict, header: dict) -> None:
        tool = plan.get("tool", "none")
        base._require(type(tool) is str and (tool == "none" or tool in tool_labels), "offline_tool")
        if tool_labels:
            base._require(header.get("selected", {}).get(producer) is (tool != "none"),
                          "offline_tool_selection")
        mode = plan.get("mode")
        allowed_modes = tool_modes.get(tool)
        if allowed_modes is not None:
            base._require(type(mode) is str and mode in allowed_modes, "offline_tool_mode")
        else:
            base._require("mode" not in plan, "offline_tool_mode")
        base._require(plan == _journal_plan(producer, tool, mode), "offline_plan")

    def aggregate_shape(aggregate: dict, plan: dict, header: dict) -> None:
        if producer == "offline_capture_tools":
            allowed = {"native_retired"} if plan["tool"] in _CAPTURE_OWNERS else {"resource_session_reported"}
        else:
            allowed = observations[producer]
        observation = aggregate.get("observation")
        base._require(set(aggregate) == {"observation"} and type(observation) is str
                      and observation in allowed, "offline_aggregate")

    return publisher_grammar(producer, roles=_ROW_LAYOUTS[producer][1], plan_shape=plan_shape,
                             aggregate_shape=aggregate_shape)


OFFLINE_GRAMMAR_SPECS = tuple(_offline_grammar(producer) for producer in _ROW_LAYOUTS)


@dataclass(frozen=True, slots=True)
class TrainingOutputs:
    """Private constructor paths and historical training obligations."""

    nightly_data: Path = field(repr=False)
    nightly_output: Path = field(repr=False)
    habits_data: Path = field(repr=False)
    conversations: Path = field(repr=False)
    exports: Path = field(repr=False)
    legacy_adapters: tuple[Path, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class OfflineCommand:
    """Effective CLI arguments after parsing, excluding authentication material.

    For capture, paths are the declared input/output locators and device indices
    describe the selected native devices. For exports, pass effective output
    paths from the original CLI. Spotify uses the protected destination label
    regardless of its local locator. For the wakeword training CLIs, pass the
    parsed subcommand as mode. Calibration capture uses mode "capture" by default.
    """

    tool: str = field(repr=False)
    paths: tuple[str | Path, ...] = field(default=(), repr=False)
    device_indices: tuple[int, ...] = field(default=(), repr=False)
    authority_origin: str | None = field(default=None, repr=False)
    mode: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ForensicDeclarations:
    """Explicit release manifests, checkpoints and admission observation dirs."""

    code_manifests: tuple[Path, ...] = field(default=(), repr=False)
    private_checkpoints: tuple[Path, ...] = field(default=(), repr=False)
    private_observations: tuple[Path, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class OfflineFamilyPlan:
    """One private resolved scope and its closed V4 projection."""

    producer: str
    selected: bool
    state_root: Path = field(repr=False)
    scope: TrainingOutputs | Path | OfflineCommand | ForensicDeclarations | None = field(repr=False)

    def journal_plan(self) -> dict:
        """Return fixed labels suitable for a V4 event."""
        if isinstance(self.scope, OfflineCommand):
            return _journal_plan(self.producer, self.scope.tool, self.scope.mode)
        return _journal_plan(self.producer)


def _path(root: Path, value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not os.fspath(value) or "\0" in os.fspath(value):
        raise ValueError("offline_path")
    # Lexical resolution keeps plan construction independent of disk and cwd.
    return Path(os.path.normpath(root / value))


def _absolute_root(value: str | Path, error: str) -> Path:
    if not isinstance(value, (str, Path)) or "\0" in os.fspath(value):
        raise ValueError(error)
    path = Path(value)
    if not path.is_absolute() or str(path) != os.path.normpath(path):
        raise ValueError(error)
    return path


def _command(root: Path, command: OfflineCommand | None, *, capture_family: bool) -> OfflineCommand | None:
    if command is None:
        return None
    boundaries = _CAPTURE_BOUNDARIES if capture_family else _EXPORT_BOUNDARIES
    if type(command.tool) is not str or command.tool not in boundaries:
        raise ValueError("offline_tool")
    modes = _CAPTURE_MODES if capture_family else _EXPORT_MODES
    allowed_modes = modes.get(command.tool, (None,))
    mode = command.mode
    if command.tool == "acoustic_calibration" and mode is None:
        mode = "capture"
    if (mode is not None and type(mode) is not str) or mode not in allowed_modes:
        raise ValueError("offline_tool_mode")
    if command.tool == "setup_spotify":
        return OfflineCommand(command.tool)
    paths = tuple(_path(root, value) for value in command.paths)
    origin = command.authority_origin
    devices = command.device_indices
    if capture_family and command.tool in _CAPTURE_OWNERS:
        origin = capture._origin(origin)
        if (type(devices) is not tuple or not devices and command.tool != "train_wakeword"
                or any(type(device) is not int or device < 0 for device in devices)):
            raise ValueError("offline_capture_device")
        if command.tool != "zone_setup" and len(devices) > 1:
            raise ValueError("offline_capture_device")
    elif origin is not None or devices or not paths:
        raise ValueError("offline_cli_scope")
    return OfflineCommand(command.tool, paths, devices, origin, mode)


def resolve_offline_family_plans(
    config: Mapping,
    state_root: Path,
    *,
    admission_root: Path | None = None,
    capture_command: OfflineCommand | None = None,
    export_command: OfflineCommand | None = None,
    forensic: ForensicDeclarations = ForensicDeclarations(),
) -> dict[str, OfflineFamilyPlan]:
    """Resolve the five rows from effective config and explicitly declared inputs.

    Relative training paths use the managed state root. Pass already-resolved
    absolute CLI paths for commands whose base directory is outside that root.
    The nightly switch constructs both runtime owners today. All historical
    paths survive either selection. SDK locations remain external obligations.
    """
    root = _absolute_root(state_root, "offline_state_root")
    training = config.get("training", {})
    nightly = training.get("nightly", {})
    training_scope = TrainingOutputs(
        _path(root, nightly.get("data_dir", "./data/nightly_training")),
        _path(root, nightly.get("output_dir", TRAINING_OUTPUT_NIGHTLY)),
        _path(root, training.get("habits", {}).get("data_dir", "./data/habit_training")),
        _path(root, training.get("conversations_dir", "./data/conversations")),
        _path(root, training.get("exports_dir", "./data/conversation_exports")),
        (
            _path(root, training.get("lora", {}).get("output_dir", TRAINING_OUTPUT_LORA)),
            _path(root, nightly.get("whisper_finetune", {}).get("output_dir", TRAINING_OUTPUT_WHISPER)),
        ),
    )
    admission = None if admission_root is None else _absolute_root(admission_root, "offline_admission_root")
    capture_scope = _command(root, capture_command, capture_family=True)
    export_scope = _command(root, export_command, capture_family=False)
    forensic_scope = ForensicDeclarations(
        tuple(_path(root, path) for path in forensic.code_manifests),
        tuple(_path(root, path) for path in forensic.private_checkpoints),
        tuple(_path(root, path) for path in forensic.private_observations),
    )
    return {
        "training": OfflineFamilyPlan("training", bool(nightly.get("enabled", False)), root, training_scope),
        "offline_copies": OfflineFamilyPlan("offline_copies", admission is not None, root, admission),
        "offline_capture_tools": OfflineFamilyPlan("offline_capture_tools", capture_scope is not None, root, capture_scope),
        "offline_exports_training": OfflineFamilyPlan("offline_exports_training", export_scope is not None, root, export_scope),
        "build_forensic_outputs": OfflineFamilyPlan("build_forensic_outputs",
            bool(forensic_scope.code_manifests or forensic_scope.private_checkpoints
                 or forensic_scope.private_observations), root, forensic_scope),
    }


@dataclass(frozen=True, slots=True)
class _AdmissionScope:
    state_root: Path = field(repr=False)
    admission_root: Path = field(repr=False)
    binding_id: str = field(repr=False)
    run_id: str = field(repr=False)


def _admission_scope(admission: RuntimeAdmission) -> _AdmissionScope:
    state = _absolute_root(admission.state_root, "offline_state_root")
    root = _absolute_root(admission._ticket["admission_root"], "offline_admission_root")
    binding_id = admission._ticket["binding_id"]
    base._require(_hex(binding_id, 64), "offline_binding_id")
    return _AdmissionScope(state, root, binding_id, admission.run_id)


def _require_copy_scope(plan: OfflineFamilyPlan, scope: _AdmissionScope) -> None:
    base._require(plan.state_root == scope.state_root, "offline_state_root")
    base._require(plan.producer == "offline_copies" and plan.scope == scope.admission_root,
                  "offline_admission_root")


@dataclass(frozen=True, eq=False, slots=True)
class OfflineCopyInventory:
    """Private admission metadata, including failed and unregistered obligations."""

    _scope: _AdmissionScope = field(repr=False)
    _active_binding_observed: bool = field(repr=False)
    slots: tuple[Path, ...] = field(repr=False)
    bindings: tuple[dict, ...] = field(repr=False)
    copies: tuple[dict, ...] = field(repr=False)
    unknown_entries: tuple[Path, ...] = field(repr=False)


def _directory_entries(path: Path, *, optional: bool = False) -> tuple[Path, ...]:
    try:
        return tuple(sorted(_stage()._safe_directory(path).iterdir()))
    except FileNotFoundError:
        if not optional or path.resolve(strict=False) != path:
            raise
        return ()


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("offline_manifest_file")


def _read_copy_inventory(scope: _AdmissionScope) -> OfflineCopyInventory:
    from src.memory.copy_catalog import _recover_record

    stage = _stage()
    root = scope.admission_root
    active_binding_observed = False
    sections = ("slots", "admission/bindings", "admission/copies")
    slots, bindings, copies, unknown = [], [], [], []
    try:
        stage._safe_directory(root)
    except (OSError, ValueError):
        return OfflineCopyInventory(scope, False, (), (), (), (root,))
    for section, collected in zip(sections, (slots, bindings, copies), strict=True):
        directory = root / section
        try:
            # Slots and bindings precede admission. The copy catalog is lazy.
            entries = _directory_entries(directory, optional=section == "admission/copies")
        except (OSError, ValueError):
            unknown.append(directory)
            continue
        for entry in entries:
            try:
                if section == "slots":
                    stage._slot_name(entry.name)
                    stage._safe_directory(entry)
                    collected.append(entry)
                elif section == "admission/bindings":
                    if entry.suffix != ".json":
                        raise ValueError("offline_binding_file")
                    _regular(entry)
                    binding = stage.read_binding(root, entry.stem)
                    collected.append(binding)
                    if entry.stem == scope.binding_id and binding["state_root"] == str(scope.state_root):
                        active_binding_observed = True
                else:
                    if not _hex(entry.name, 32):
                        raise ValueError("offline_copy_entry")
                    stage._safe_directory(entry)
                    for record in _directory_entries(entry):
                        _regular(record)
                    collected.append(_recover_record(entry))
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                unknown.append(entry)
    return OfflineCopyInventory(scope, active_binding_observed, tuple(slots), tuple(bindings),
                                tuple(copies), tuple(unknown))


async def observe_offline_copies(plan: OfflineFamilyPlan, coordinator: OutputRegistrationV2) -> OfflineCopyInventory:
    """Read slots, original bindings and copy chains without taking a lease.

    Copy records retain declared destination, snapshot references and extents.
    This scan retains unresolved entries and checks the active ticket binding.
    It never visits copy payload roots.
    """
    scope = _admission_scope(coordinator.admission)
    _require_copy_scope(plan, scope)
    return await asyncio.to_thread(_read_copy_inventory, scope)


@dataclass(frozen=True, eq=False, slots=True)
class BuildForensicInventory:
    """The separate roots declared by release and private checkpoint manifests."""

    declarations: ForensicDeclarations = field(repr=False)
    code_roots: tuple[Path, ...] = field(repr=False)
    private_roots: tuple[Path, ...] = field(repr=False)


def _read_manifest(path: Path) -> dict:
    limit = 64 * 1024 * 1024
    if path.parent.resolve(strict=True) != path.parent:
        raise ValueError("offline_manifest_location")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("offline_manifest_file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("offline_manifest_file")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("offline_manifest_schema")
    return value


def _hex(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and re.fullmatch("[a-f0-9]+", value) is not None


def _code_manifest(value: dict) -> bool:
    files = value.get("files")
    return (value.get("format") == "kza-cpython-bytecode-v1" and value.get("python") == "3.13"
            and _hex(value.get("git_commit"), 40) and _hex(value.get("python_magic"), 8)
            and value.get("entrypoint") == ["python", "-m", "src.main"]
            and value.get("external_paths") == ["config", "data", "models", "logs"]
            and type(files) is dict and {"src/main.pyc", "src/__init__.pyc"} <= files.keys()
            and all(_code_entry(name) and _hex(digest, 64) for name, digest in files.items()))


def _code_entry(name: str) -> bool:
    path = Path(name)
    return (bool(path.parts) and path.parts[0] in {"src", "scripts", "tools"} and ".." not in path.parts
            and path.as_posix() == name and "\0" not in name
            and (path.suffix == ".pyc" or path.parts[0] == "src" and path.suffix == ".csv"
                 or path.is_relative_to("src/dashboard/frontend") and path.suffix in {
                     ".html", ".css", ".js", ".jsx", ".json", ".svg", ".txt", ".png", ".jpg",
                     ".ico", ".woff", ".woff2", ".ttf", ".wav",
                 }))


def _private_checkpoint(value: dict, root: Path) -> bool:
    fields = {"schema_version", "kind", "source_commit", "runtime_root", "state_root",
              "release_sha256", "helper_manifest_sha256", "payload", "blob_manifest"}
    if (value.keys() != fields or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "kza-compiled-restore-state" or not _hex(value["source_commit"], 40)
            or not _hex(value["release_sha256"], 64) or not _hex(value["helper_manifest_sha256"], 64)
            or type(value["payload"]) is not dict or type(value["blob_manifest"]) is not dict):
        return False
    for field_name in ("runtime_root", "state_root"):
        locator = value[field_name]
        if type(locator) is not str or not Path(locator).is_absolute():
            return False
        owner_root = _path(root, locator)
        if root.is_relative_to(owner_root) or owner_root.is_relative_to(root):
            return False
    return True


def _private_observation_root(directory: Path) -> Path:
    if tuple(directory.parts[-4:-1]) != ("admission", "copy-observations", "v1"):
        raise ValueError("offline_observation_location")
    stage = _stage()
    stage._safe_directory(directory)
    for record in _directory_entries(directory):
        _regular(record)
    intent, event, _ = stage._read_copy_observation(directory, directory.parents[3])
    if event["phase"] != "observed":
        raise ValueError("offline_observation_unfinished")
    return Path(intent["observed_root"])


def _read_forensic_inventory(declarations: ForensicDeclarations) -> BuildForensicInventory:
    code = tuple(dict.fromkeys(path.parent for path in declarations.code_manifests))
    try:
        observations = tuple(_private_observation_root(path) for path in declarations.private_observations)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise ValueError("offline_manifest_invalid") from None
    private = tuple(dict.fromkeys(tuple(path.parent for path in declarations.private_checkpoints) + observations))
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a in code for b in private):
        raise ValueError("offline_manifest_root_overlap")
    try:
        for manifest in declarations.code_manifests:
            if not _code_manifest(_read_manifest(manifest)):
                raise ValueError
        for checkpoint in declarations.private_checkpoints:
            if not _private_checkpoint(_read_manifest(checkpoint), checkpoint.parent):
                raise ValueError
    except (OSError, ValueError, RecursionError):
        raise ValueError("offline_manifest_invalid") from None
    return BuildForensicInventory(declarations, code, private)


async def observe_build_forensic_outputs(plan: OfflineFamilyPlan) -> BuildForensicInventory:
    """Read explicitly named manifests and keep code and private roots disjoint."""
    if (plan.producer != "build_forensic_outputs" or not plan.selected
            or not isinstance(plan.scope, ForensicDeclarations)):
        raise ValueError("offline_forensic_plan")
    return await asyncio.to_thread(_read_forensic_inventory, plan.scope)


def _matches_training(scope: TrainingOutputs, state_root: Path, owners: tuple[object, ...]) -> bool:
    from src.training.habit_dataset_generator import HabitDatasetGenerator
    from src.training.nightly_trainer import NightlyTrainer

    if len(owners) != 2:
        return False
    trainer, habits = owners
    return (type(trainer) is NightlyTrainer and type(habits) is HabitDatasetGenerator
            and trainer.habit_generator is habits
            and _path(state_root, trainer.config.data_dir) == scope.nightly_data
            and _path(state_root, trainer.config.output_dir) == scope.nightly_output
            and _path(state_root, trainer.config.conversations_dir) == scope.conversations
            and _path(state_root, habits.data_dir) == scope.habits_data)


def _matches_plan(plan: OfflineFamilyPlan, owners: tuple[object, ...], admitted: _AdmissionScope) -> bool:
    if not owners:
        return False
    scope = plan.scope
    if isinstance(scope, TrainingOutputs):
        return _matches_training(scope, plan.state_root, owners)
    if plan.producer == "offline_capture_tools" and isinstance(scope, OfflineCommand) and scope.tool in _CAPTURE_OWNERS:
        return (len({id(owner) for owner in owners}) == len(owners)
                and all(type(owner) is _CAPTURE_OWNERS[scope.tool] and owner._origin == scope.authority_origin
                        for owner in owners))
    if len(owners) != 1:
        return False
    owner, = owners
    if plan.producer == "offline_copies":
        return type(owner) is OfflineCopyInventory and owner._scope == admitted
    if plan.producer == "build_forensic_outputs":
        return type(owner) is BuildForensicInventory and owner.declarations is scope
    if plan.producer == "offline_exports_training":
        return owner is scope
    if plan.producer == "offline_capture_tools" and isinstance(scope, OfflineCommand):
        from src.core.resource_retirement import ResourceSession

        return scope.tool == "enroll_voice" and type(owner) is ResourceSession and owner.error_prefix == "enrollment"
    return False


def _observation(plan: OfflineFamilyPlan, owners: tuple[object, ...]) -> str:
    owner = owners[0]
    match plan.producer:
        case "training":
            return "blocked_instance" if owner.training_blocked else "instance_reported"
        case "offline_copies":
            bound_states = {binding["state_root"] for binding in owner.bindings}
            if (not owner._active_binding_observed or owner.unknown_entries
                    or any(str(slot / "state") not in bound_states for slot in owner.slots)
                    or any(record["preparation"] != "prepared" or not record.get("finished")
                           for record in owner.copies)):
                return "catalog_unresolved"
            return "catalog_observed"
        case "offline_capture_tools":
            if plan.scope.tool == "enroll_voice":
                base._require(owner._sealed, "offline_enrollment_unsettled")
                return "resource_session_reported"
            base._require(all(_capture_retired(client) for client in owners), "offline_capture_unsettled")
            return "native_retired"
        case "offline_exports_training":
            return "invocation_reported"
        case "build_forensic_outputs":
            return "declared_manifests_observed"
    raise ValueError("offline_producer")


class OfflineFamilyPublishers:
    """Keep the exact attached owner identities in memory for P2-I1.

    Only the lifecycle labels are durable. I1 must compare the actual graph or
    invocation against the original owners. Catalog/manifest observations and
    a CLI declaration are scoped evidence, never a global coverage predicate.
    """

    def __init__(
        self, plans: Mapping[str, OfflineFamilyPlan], publishers: Mapping[str, GenericOutputPublisher],
        admitted: _AdmissionScope,
    ) -> None:
        self._plans = dict(plans)
        self._publishers = dict(publishers)
        self._admitted = admitted
        self._owners: dict[str, tuple[object, ...]] = {}
        self._opening: set[str] = set()
        self._drained: set[str] = set()
        self._lock = asyncio.Lock()

    def matches_attached_owners(self, producer: str, *owners: object) -> bool:
        """Check ordered identity and the nominal contract against I1's owners."""
        attached = self._owners.get(producer)
        return (attached is not None and len(attached) == len(owners)
                and all(first is second for first, second in zip(attached, owners, strict=True))
                and _matches_plan(self._plans[producer], owners, self._admitted))

    async def attach(self, producer: str, *owners: object) -> None:
        """Associate the constructed owners after intent and before native work."""
        async with self._lock:
            base._require(producer in self._publishers and producer not in self._owners, "offline_attach")
            base._require(_matches_plan(self._plans[producer], owners, self._admitted), "offline_owner")
            await self._publishers[producer].attached()
            self._owners[producer] = owners

    async def opening(self, producer: str, *owners: object) -> None:
        """Record opening before handing control to the original owner."""
        async with self._lock:
            base._require(self.matches_attached_owners(producer, *owners)
                          and producer not in self._opening, "offline_opening")
            await self._publishers[producer].opening()
            self._opening.add(producer)

    async def drained(self, producer: str, *owners: object) -> None:
        """Record a terminal observation after the original owner has returned."""
        async with self._lock:
            base._require(self.matches_attached_owners(producer, *owners) and producer in self._opening
                          and producer not in self._drained, "offline_drained")
            observation = _observation(self._plans[producer], owners)
            await self._publishers[producer].drained({"observation": observation})
            self._drained.add(producer)


async def bind_offline_family_publishers(
    coordinator: OutputRegistrationV2, plans: Mapping[str, OfflineFamilyPlan],
) -> OfflineFamilyPublishers:
    """Publish intents/disabled rows before construction by the I1 caller.

    Construct the coordinator with OFFLINE_GRAMMAR_SPECS and use each resolved
    plan's selected flag in publisher_selection. Pass the actual admission root
    to the resolver. The original owner controls
    authorization, native opening and resource retirement. The returned binder
    checks the identities on subsequent lifecycle observations.
    """
    plans = dict(plans)
    base._require(plans.keys() == _ROW_LAYOUTS.keys(), "offline_rows")
    admitted = _admission_scope(coordinator.admission)
    _require_copy_scope(plans["offline_copies"], admitted)
    shapes = {}
    header = {"selected": coordinator._selected}
    for spec in OFFLINE_GRAMMAR_SPECS:
        plan = plans[spec.producer]
        base._require(plan.state_root == admitted.state_root, "offline_state_root")
        base._require(plan.producer == spec.producer and coordinator._selected.get(spec.producer) is plan.selected
                      and coordinator._writer_grammars.get(spec.producer) == spec.grammar_id,
                      "offline_selection")
        shape = plan.journal_plan()
        spec.decode_event({"kind": spec.event_kind, "phase": "intent" if plan.selected else "disabled",
                           "plan": shape}, None, plan.selected, header)
        shapes[spec.producer] = shape

    await coordinator._run(coordinator.admission.revalidate)
    publishers = {}
    for producer, shape in shapes.items():
        if plans[producer].selected:
            publishers[producer] = await coordinator.publisher_intent(producer, shape)
        else:
            await coordinator.publisher_disabled(producer, shape)
    return OfflineFamilyPublishers(plans, publishers, admitted)
