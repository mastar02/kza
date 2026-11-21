"""T024 offline publisher plans and read-only destination observations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
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
    from src.memory.output_registration_v2 import OutputRegistrationV2


_CAPTURE_OWNERS = {
    "test_voice_command": capture.ManualVoiceCommandAuthorityClient,
    "zone_setup": capture.ZoneSetupAuthorityClient,
    "diagnose_wakeword": capture.WakewordDiagnosticsAuthorityClient,
}
_CAPTURE_BOUNDARIES = dict.fromkeys(_CAPTURE_OWNERS, "delegated_target") | {"enroll_voice": "recorded_input"}
_EXPORT_BOUNDARIES = {
    "inventory_home": "private_export",
    "sync_ha_to_chroma": "ha_index",
    "setup_spotify": "protected_oauth",
    "stage_legacy_memory": "legacy_staging",
    "train_assistant": "training_export",
    "train_wakeword": "wakeword_training",
    "train_custom_wake": "wakeword_training",
}
_ROW_LAYOUTS = {
    "training": ("file_tree", ("habit_dataset_generator", "nightly_trainer")),
    "offline_copies": ("readonly_input", ("admission_controller", "copy_reservation", "derivation", "snapshot")),
    "offline_capture_tools": ("resource_only", tuple(_CAPTURE_BOUNDARIES)),
    "offline_exports_training": ("declared_root", tuple(_EXPORT_BOUNDARIES)),
    "build_forensic_outputs": ("declared_root", ("code_artifacts", "private_forensics")),
}


def _journal_plan(producer: str, tool: str = "none") -> dict:
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
    return publisher_plan(kind, roles=roles, **details)


def _offline_grammar(producer: str) -> GrammarSpec:
    tool_labels = {
        "offline_capture_tools": _CAPTURE_BOUNDARIES,
        "offline_exports_training": _EXPORT_BOUNDARIES,
    }.get(producer, {})
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
        base._require(plan == _journal_plan(producer, tool), "offline_plan")

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

    state_root: Path = field(repr=False)
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
    regardless of its local locator.
    """

    tool: str = field(repr=False)
    paths: tuple[str | Path, ...] = field(default=(), repr=False)
    device_indices: tuple[int, ...] = field(default=(), repr=False)
    authority_origin: str | None = field(default=None, repr=False)


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
    scope: TrainingOutputs | Path | OfflineCommand | ForensicDeclarations | None = field(repr=False)

    def journal_plan(self) -> dict:
        """Return fixed labels suitable for a V4 event."""
        tool = self.scope.tool if isinstance(self.scope, OfflineCommand) else "none"
        return _journal_plan(self.producer, tool)


def _path(root: Path, value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not os.fspath(value) or "\0" in os.fspath(value):
        raise ValueError("offline_path")
    # Lexical resolution keeps plan construction independent of disk and cwd.
    return Path(os.path.normpath(root / value))


def _command(root: Path, command: OfflineCommand | None, *, live: bool) -> OfflineCommand | None:
    if command is None:
        return None
    boundaries = _CAPTURE_BOUNDARIES if live else _EXPORT_BOUNDARIES
    if type(command.tool) is not str or command.tool not in boundaries:
        raise ValueError("offline_tool")
    if command.tool == "setup_spotify":
        return OfflineCommand(command.tool)
    paths = tuple(_path(root, value) for value in command.paths)
    origin = command.authority_origin
    devices = command.device_indices
    if live and command.tool in _CAPTURE_OWNERS:
        origin = capture._origin(origin)
        if (type(devices) is not tuple or not devices
                or any(type(device) is not int or device < 0 for device in devices)):
            raise ValueError("offline_capture_device")
        if command.tool != "zone_setup" and len(devices) != 1:
            raise ValueError("offline_capture_device")
    elif origin is not None or devices or not paths:
        raise ValueError("offline_cli_scope")
    return OfflineCommand(command.tool, paths, devices, origin)


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
    root = Path(state_root)
    if not root.is_absolute():
        raise ValueError("offline_state_root")
    training = config.get("training", {})
    nightly = training.get("nightly", {})
    training_scope = TrainingOutputs(
        root,
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
    admission = None if admission_root is None else _path(root, admission_root)
    capture_scope = _command(root, capture_command, live=True)
    export_scope = _command(root, export_command, live=False)
    forensic_scope = ForensicDeclarations(
        tuple(_path(root, path) for path in forensic.code_manifests),
        tuple(_path(root, path) for path in forensic.private_checkpoints),
        tuple(_path(root, path) for path in forensic.private_observations),
    )
    return {
        "training": OfflineFamilyPlan("training", bool(nightly.get("enabled", False)), training_scope),
        "offline_copies": OfflineFamilyPlan("offline_copies", admission is not None, admission),
        "offline_capture_tools": OfflineFamilyPlan("offline_capture_tools", capture_scope is not None, capture_scope),
        "offline_exports_training": OfflineFamilyPlan("offline_exports_training", export_scope is not None, export_scope),
        "build_forensic_outputs": OfflineFamilyPlan("build_forensic_outputs",
            bool(forensic_scope.code_manifests or forensic_scope.private_checkpoints
                 or forensic_scope.private_observations), forensic_scope),
    }


@dataclass(frozen=True, eq=False, slots=True)
class OfflineCopyInventory:
    """Private admission metadata, including failed and unregistered obligations."""

    root: Path = field(repr=False)
    slots: tuple[Path, ...] = field(repr=False)
    bindings: tuple[dict, ...] = field(repr=False)
    copies: tuple[dict, ...] = field(repr=False)
    unknown_entries: tuple[Path, ...] = field(repr=False)


def _directory_entries(path: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(_stage()._safe_directory(path).iterdir()))
    except FileNotFoundError:
        if path.resolve(strict=False) != path:
            raise ValueError("offline_directory") from None
        return ()


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("offline_manifest_file")


def _read_copy_inventory(root: Path) -> OfflineCopyInventory:
    from src.memory.copy_catalog import _recover_record

    stage = _stage()
    sections = ("slots", "admission/bindings", "admission/copies")
    slots, bindings, copies, unknown = [], [], [], []
    try:
        stage._safe_directory(root)
    except (OSError, ValueError):
        return OfflineCopyInventory(root, (), (), (), (root,))
    for section, collected in zip(sections, (slots, bindings, copies), strict=True):
        directory = root / section
        try:
            entries = _directory_entries(directory)
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
                    collected.append(stage.read_binding(root, entry.stem))
                else:
                    if not _hex(entry.name, 32):
                        raise ValueError("offline_copy_entry")
                    stage._safe_directory(entry)
                    for record in _directory_entries(entry):
                        _regular(record)
                    collected.append(_recover_record(entry))
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                unknown.append(entry)
    return OfflineCopyInventory(root, tuple(slots), tuple(bindings), tuple(copies), tuple(unknown))


async def observe_offline_copies(plan: OfflineFamilyPlan) -> OfflineCopyInventory:
    """Read slots, original bindings and copy chains without taking a lease.

    Copy records retain declared destination, snapshot references and extents.
    This scan also retains unresolved entries. It never visits copy payload roots.
    """
    if plan.producer != "offline_copies" or not isinstance(plan.scope, Path):
        raise ValueError("offline_copy_plan")
    return await asyncio.to_thread(_read_copy_inventory, plan.scope)


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


def _matches_training(scope: TrainingOutputs, owners: tuple[object, ...]) -> bool:
    from src.training.habit_dataset_generator import HabitDatasetGenerator
    from src.training.nightly_trainer import NightlyTrainer

    if len(owners) != 2:
        return False
    trainer, habits = owners
    return (type(trainer) is NightlyTrainer and type(habits) is HabitDatasetGenerator
            and trainer.habit_generator is habits
            and _path(scope.state_root, trainer.config.data_dir) == scope.nightly_data
            and _path(scope.state_root, trainer.config.output_dir) == scope.nightly_output
            and _path(scope.state_root, trainer.config.conversations_dir) == scope.conversations
            and _path(scope.state_root, habits.data_dir) == scope.habits_data)


def _matches_plan(plan: OfflineFamilyPlan, owners: tuple[object, ...]) -> bool:
    if not owners:
        return False
    scope = plan.scope
    if isinstance(scope, TrainingOutputs):
        return _matches_training(scope, owners)
    if len(owners) != 1:
        return False
    owner, = owners
    if plan.producer == "offline_copies":
        return type(owner) is OfflineCopyInventory and owner.root == scope
    if plan.producer == "build_forensic_outputs":
        return type(owner) is BuildForensicInventory and owner.declarations is scope
    if plan.producer == "offline_exports_training":
        return owner is scope
    if plan.producer == "offline_capture_tools" and isinstance(scope, OfflineCommand):
        if scope.tool == "enroll_voice":
            from src.core.resource_retirement import ResourceSession

            return type(owner) is ResourceSession and owner.error_prefix == "enrollment"
        return type(owner) is _CAPTURE_OWNERS[scope.tool] and owner._origin == scope.authority_origin
    return False


def _observation(plan: OfflineFamilyPlan, owner: object) -> str:
    match plan.producer:
        case "training":
            return "blocked_instance" if owner.training_blocked else "instance_reported"
        case "offline_copies":
            bound_states = {binding["state_root"] for binding in owner.bindings}
            if (owner.unknown_entries or any(str(slot / "state") not in bound_states for slot in owner.slots)
                    or any(record["preparation"] != "prepared" or not record.get("finished")
                           for record in owner.copies)):
                return "catalog_unresolved"
            return "catalog_observed"
        case "offline_capture_tools":
            if plan.scope.tool == "enroll_voice":
                base._require(owner._sealed, "offline_enrollment_unsettled")
                return "resource_session_reported"
            base._require(owner._retired and owner._closed and not owner._uncertain, "offline_capture_unsettled")
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
    ) -> None:
        self._plans = dict(plans)
        self._publishers = dict(publishers)
        self._owners: dict[str, tuple[object, ...]] = {}
        self._opening: set[str] = set()
        self._drained: set[str] = set()
        self._lock = asyncio.Lock()

    def matches_attached_owners(self, producer: str, *owners: object) -> bool:
        """Check ordered identity and the nominal contract against I1's owners."""
        attached = self._owners.get(producer)
        return (attached is not None and len(attached) == len(owners)
                and all(first is second for first, second in zip(attached, owners, strict=True))
                and _matches_plan(self._plans[producer], owners))

    async def attach(self, producer: str, *owners: object) -> None:
        """Associate the constructed owners after intent and before native work."""
        async with self._lock:
            base._require(producer in self._publishers and producer not in self._owners, "offline_attach")
            base._require(_matches_plan(self._plans[producer], owners), "offline_owner")
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
            observation = _observation(self._plans[producer], owners[0])
            await self._publishers[producer].drained({"observation": observation})
            self._drained.add(producer)


async def bind_offline_family_publishers(
    coordinator: OutputRegistrationV2, plans: Mapping[str, OfflineFamilyPlan],
) -> OfflineFamilyPublishers:
    """Publish intents/disabled rows before construction by the I1 caller.

    Construct the coordinator with OFFLINE_GRAMMAR_SPECS and use each resolved
    plan's selected flag in publisher_selection. The original owner controls
    authorization, native opening and resource retirement. The returned binder
    checks the identities on subsequent lifecycle observations.
    """
    base._require(plans.keys() == _ROW_LAYOUTS.keys(), "offline_rows")
    shapes = {}
    header = {"selected": coordinator._selected}
    for spec in OFFLINE_GRAMMAR_SPECS:
        plan = plans[spec.producer]
        base._require(plan.producer == spec.producer and coordinator._selected.get(spec.producer) is plan.selected
                      and coordinator._writer_grammars.get(spec.producer) == spec.grammar_id,
                      "offline_selection")
        shape = plan.journal_plan()
        spec.decode_event({"kind": spec.event_kind, "phase": "intent" if plan.selected else "disabled",
                           "plan": shape}, None, plan.selected, header)
        shapes[spec.producer] = shape

    publishers = {}
    for producer, shape in shapes.items():
        if plans[producer].selected:
            publishers[producer] = await coordinator.publisher_intent(producer, shape)
        else:
            await coordinator.publisher_disabled(producer, shape)
    return OfflineFamilyPublishers(plans, publishers)
