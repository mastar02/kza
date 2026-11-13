"""Versioned output observations bound to original native resource owners."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import weakref
from weakref import WeakKeyDictionary
from typing import Callable, Iterator, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.runtime_admission import RuntimeAdmission
    from src.lists.list_store import ListStore
    from src.reminders.reminder_store import ReminderStore
    from src.memory.erasure_journal import ErasureParticipant, TargetRegistrySnapshot

from src.memory import output_registration as base
from src.core.owned_work import current_native_execution, validate_native_execution

logger = logging.getLogger(__name__)

_CALL: ContextVar[tuple | None] = ContextVar("output_call", default=None)
_OPERATION: ContextVar[object | None] = ContextVar("output_operation", default=None)

_MAX_VERSIONS = 32
_MAX_OPEN_EXTENTS = 512
_MAX_NATIVE_WITNESSES = 128
_SELECTED = frozenset({"lists", "reminders", "privacy_authority", "authority_sessions", "routines", "events", "suggestions"})
_PHASES = {
    "intent": {None}, "attached": {"intent"}, "opening": {"attached"},
    "active": {"opening", "attached"}, "closed": {"active"},
    "native_failed_then_actually_closed": {"attached", "opening"},
    "disabled_never_opened": {None}, "unresolved": {None, "attached"},
}


class OutputRegistrationError(RuntimeError):
    """A rejected output boundary cannot become a successful closure receipt."""


class RegistryChanged(RuntimeError):
    """Original source target observation changed before native admission."""


class OutputNotReady(RuntimeError):
    """Original close ownership exists but output admission is not ready yet."""


@dataclass(eq=False)
class _Owner:
    instance: object
    producer: str
    role: str
    key: str
    path: str | None
    work: object | None
    phase: str = "waiting"
    snapshot: object | None = None
    grants: dict | None = None
    registry_state: tuple | None = None
    native_failed: bool = False
    final_work: object | None = None
    final_args: tuple | None = None
    final_execution: object | None = None
    final_sequence: asyncio.Task | None = None
    native_identity: dict | None = None
    initialized_execution: object | None = None


@dataclass(eq=False, frozen=True)
class _HistoricalGrant:
    entry: object
    plan_json: bytes
    scope_json: bytes | None


@dataclass(eq=False, frozen=True)
class _Operation:
    # Weak: this lives in the native thread's copied Context, which a failed
    # call's traceback can keep alive (to_thread frame). A strong binding here
    # closes Context -> binding -> task -> exception -> traceback -> Context.
    execution: weakref.ref
    owner: _Owner
    snapshot: object
    grants: tuple


@dataclass(eq=False, frozen=True)
class ConstructorReservation:
    """An actual constructor dispatch reservation, never a serialized grant."""
    coordinator: OutputRegistrationV2
    events: object
    path: str

    def __reduce_ex__(self, protocol: int):
        raise TypeError("output_constructor_not_serializable")


def _managed() -> bool:
    from src.core.runtime_admission import _verified_admissions
    return bool(base._ACTIVE or any(_verified_admissions.values()))


def _unmanaged_path(path: str | Path) -> None:
    from src.core.runtime_admission import _validate_path
    if _managed() or _validate_path(path, initial=False) is not None:
        _reject("required")


def _coordinator(owner: object | None = None) -> OutputRegistrationV2 | None:
    if owner is not None:
        value = getattr(owner, "_output_coordinator", None)
        if isinstance(value, OutputRegistrationV2):
            return value
    from src.core.runtime_admission import _verified_admissions
    values = [a._output_registration for a, installed in _verified_admissions.items()
              if installed and isinstance(a._output_registration, OutputRegistrationV2)]
    if len(values) == 1:
        return values[0]
    return None


def _reject(reason: str, coordinator: OutputRegistrationV2 | None = None):
    if coordinator is not None:
        coordinator._failed = True
    raise OutputRegistrationError("output_registration_" + reason)


def _namespace(root: Path, *, reserve_entries: int = 0) -> int:
    """One bounded budget covers every known and unknown journal version."""
    count, total = reserve_entries, 0
    base._require(count <= base.MAX_NAMESPACE_ENTRIES, "capacity")
    if not root.exists():
        return 0
    base._directory(root)
    pending = [(root, 0)]
    versions = 0
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                base._require(count <= base.MAX_NAMESPACE_ENTRIES, "capacity")
                info = entry.stat(follow_symlinks=False)
                if depth == 0:
                    versions += 1
                    base._require(versions <= _MAX_VERSIONS, "capacity")
                if stat.S_ISDIR(info.st_mode):
                    base._require(depth < 3, "namespace_changed")
                    pending.append((Path(entry.path), depth + 1))
                else:
                    base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= base.MAX_RECORD_BYTES, "namespace_changed")
                    total += info.st_size
                    base._require(total <= base.MAX_TOTAL_BYTES, "capacity")
    return total


def _plan(producer: str, role: str, path: str | None, state: Path) -> dict:
    if path is None:
        return {"kind": "worker", "producer": producer, "role": role}
    supplied = Path(path).absolute()
    base._require(not supplied.is_symlink(), "destination")
    physical = supplied.resolve(strict=False)
    base._require(physical.is_relative_to(state) and physical != state, "destination")
    ancestor = physical.parent
    missing = []
    while not ancestor.exists():
        missing.append(ancestor.name)
        ancestor = ancestor.parent
    identity = base._directory(ancestor)
    existing = None
    if physical.exists():
        info = physical.lstat()
        base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination")
        existing = base._identity(physical)
    for suffix in ("-wal", "-shm", "-journal", ".lock", ".lease"):
        sidecar = Path(str(physical) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            info = sidecar.lstat()
            base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination")
    lease = ".lock" if producer == "privacy_authority" else ".lease" if producer == "authority_sessions" else None
    return {"kind": "sqlite", "producer": producer, "role": role, "path": str(physical),
            "configured_path": str(supplied), "parent_identity": identity,
            "ancestor": str(ancestor), "missing_parents": list(reversed(missing)),
            "existing_identity": existing, "before_first_open": True,
            "creation_observation": "observed_existing" if existing is not None else "planned_absent",
            "sidecars": [str(physical) + s for s in ("-wal", "-shm", "-journal")],
            "leases": [] if lease is None else [str(physical) + lease],
            "classification": "protected_authority" if producer in {"privacy_authority", "authority_sessions", "routines"} else "mixed_source_policy"}


def _canonical_path(value: object) -> Path:
    base._require(type(value) is str, "plan")
    path = Path(value)
    base._require(path.is_absolute() and str(path) == value and ".." not in path.parts, "plan")
    return path


def _plan_shape(plan: dict, state: Path) -> None:
    base._require(type(plan) is dict and plan.get("producer") in _SELECTED, "plan")
    producer = plan["producer"]
    if "kind" not in plan:
        base._require(producer in {"lists", "reminders"}, "plan")
        base._plan_shape(plan, producer, state)
        return
    role = plan.get("role")
    if plan["kind"] == "worker":
        base._require(set(plan) == {"kind", "producer", "role"} and (
            producer in {"events", "suggestions"} and role in {"eraser", "retention"}
            or producer in {"authority_sessions", "events", "suggestions"} and role == "primary"
        ), "plan")
        return
    base._require(set(plan) == {"kind", "producer", "role", "path", "configured_path", "parent_identity", "ancestor", "missing_parents", "existing_identity", "before_first_open", "creation_observation", "sidecars", "leases", "classification"}
        and plan["kind"] == "sqlite" and producer in _SELECTED - {"lists", "reminders"}
        and (role == "primary" or role == "historical" and producer in {"events", "suggestions"}), "plan")
    path, ancestor = _canonical_path(plan["path"]), _canonical_path(plan["ancestor"])
    configured = _canonical_path(plan["configured_path"])
    base._require(path.is_relative_to(state) and path != state and ancestor.is_relative_to(state)
        and path.parent.is_relative_to(ancestor) and configured.name == path.name
        and type(plan["missing_parents"]) is list
        and plan["missing_parents"] == list(path.parent.relative_to(ancestor).parts)
        and (plan["existing_identity"] is None or not plan["missing_parents"])
        and plan["creation_observation"] == ("observed_existing" if plan["existing_identity"] is not None else "planned_absent")
        and plan["before_first_open"] is True and base._identity_shape(plan["parent_identity"])
        and (plan["existing_identity"] is None or base._identity_shape(plan["existing_identity"]))
        and plan["sidecars"] == [str(path) + suffix for suffix in ("-wal", "-shm", "-journal")], "plan")
    lease = ".lock" if producer == "privacy_authority" else ".lease" if producer == "authority_sessions" else None
    base._require(plan["leases"] == ([] if lease is None else [str(path) + lease])
        and plan["classification"] == ("protected_authority" if producer in {"privacy_authority", "authority_sessions", "routines"} else "mixed_source_policy"), "plan")


def _owner_event_shape(event: dict, prior: dict | None, selected: dict) -> None:
    plan, phase, native = event["plan"], event["phase"], event["native_identity"]
    producer, role = plan["producer"], plan.get("role", "primary")
    base._require(type(event["owner"]) is str and event["owner"] == producer + (
        "." + role if role in {"eraser", "retention"} else ""), "owner")
    if plan.get("kind") == "worker":
        base._require(native is None, "native_identity")
        if role == "primary":
            base._require(selected.get(producer) is False and phase == "disabled_never_opened" and prior is None, "owner_phase")
        else:
            base._require(phase in {"intent", "attached", "active", "closed", "unresolved"}
                and (phase != "active" or prior is not None and prior["phase"] == "attached")
                and (phase != "unresolved" or prior is not None and prior["phase"] == "attached"), "owner_phase")
        return
    base._require(role == "primary" and phase not in {"disabled_never_opened"}
        and (producer in {"lists", "reminders"} or selected.get(producer) is True), "owner_phase")
    if phase in {"active", "closed"}:
        base._require(base._identity_shape(native), "native_identity")
        if phase == "active":
            base._require(prior is not None and prior["phase"] == "opening"
                and (plan["existing_identity"] is None or native == plan["existing_identity"]), "native_identity")
        else:
            base._require(prior is not None and prior["phase"] == "active" and native == prior["native_identity"], "native_identity")
    else:
        base._require(native is None, "native_identity")
        if phase == "native_failed_then_actually_closed":
            base._require(prior is not None and prior["phase"] == "opening", "owner_phase")
        if phase == "unresolved":
            base._require(prior is not None and prior["phase"] == "attached", "owner_phase")


def _physical_shape(path: str | Path, plan: dict, *, expected_identity: dict | None = None,
                    allow_absent: bool = True) -> None:
    """Recheck the pinned physical extent immediately before its native open."""
    supplied = Path(path)
    base._require(not supplied.is_symlink() and str(supplied.resolve(strict=False)) == plan["path"], "destination_changed")
    base._require(base._directory(Path(plan["ancestor"])) == plan["parent_identity"], "destination_changed")
    try:
        info = supplied.lstat()
    except FileNotFoundError:
        base._require(allow_absent, "destination_changed")
    else:
        base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination_changed")
        identity = {"device": info.st_dev, "inode": info.st_ino}
        base._require(expected_identity is None or identity == expected_identity, "destination_changed")
    for name in (*plan["sidecars"], *plan["leases"]):
        try:
            info = Path(name).lstat()
        except FileNotFoundError:
            continue
        base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination_changed")


def _target_record(entry: object) -> dict:
    return {"id": entry.id, "participant": entry.participant.value,
            "version": entry.version, "descriptor": entry.descriptor}


def _scope_shape(scope: dict, state: Path) -> Path:
    """Check the tagged observation's shape; this grants no execution authority."""
    base._require(type(scope) is dict and set(scope) == {
        "scope_schema_version", "source_binding", "delegation_sha256", "source_close_sha256",
        "source_state_root", "source_state_identity"}
        and type(scope["scope_schema_version"]) is int and scope["scope_schema_version"] == 1
        and all(type(scope[key]) is str and re.fullmatch("[0-9a-f]{64}", scope[key])
                for key in ("source_binding", "delegation_sha256", "source_close_sha256"))
        and base._identity_shape(scope["source_state_identity"]), "delegated_scope")
    root = _canonical_path(scope["source_state_root"])
    base._require(root != state, "delegated_scope")
    return root


def _historical_plan(admission: object, entry: object) -> tuple[dict, bytes | None, bytes | None]:
    """Plan an exact local target or one selected by verified delegation."""
    descriptor, producer = entry.descriptor, entry.participant.value
    path, state = Path(descriptor["path"]).absolute(), admission.state_root
    if path.is_relative_to(state):
        return _plan(producer, "historical", path, state), None, None
    scope_json, policy_json = admission._plan_historical_target(_target_record(entry))
    scope = json.loads(scope_json)
    source = _scope_shape(scope, state)
    plan = _plan(producer, "historical", path, source)
    base._require(plan["path"] == descriptor["path"], "delegated_scope")
    return plan, scope_json, policy_json


def _policy_shape(event: dict) -> bytes:
    base._require(set(event) == {"kind", "scope_schema_version", "ownership_sha256", "policy",
        "revision", "previous_sha256"} and event["kind"] == "delegation_policy"
        and type(event["scope_schema_version"]) is int and event["scope_schema_version"] == 1
        and type(event["ownership_sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", event["ownership_sha256"]), "delegated_policy")
    policy = event["policy"]
    base._require(type(policy) is dict and set(policy) == {"schema_version", "source_bindings",
        "owned_targets", "delegated_targets", "historical_targets"}
        and type(policy["schema_version"]) is int and policy["schema_version"] == 1
        and all(type(policy[key]) is list for key in ("source_bindings", "owned_targets", "delegated_targets", "historical_targets")), "delegated_policy")
    raw = base._encoded(policy)
    base._require(base._sha(raw) == event["ownership_sha256"], "delegated_policy")
    return raw


def _observation_shape(event: dict, state: Path, owners: dict, targets: set) -> None:
    common = {"kind", "revision", "previous_sha256"}
    kind = event["kind"]
    if kind == "delegation_policy":
        _policy_shape(event)
        return
    if kind == "delegated_target":
        # V2's explicit extension. The old target form remains current-state
        # only, and older readers reject this discriminator as unknown.
        source = _scope_shape(event["scope"], state)
        local = dict(event)
        del local["scope"]
        local["kind"] = "target"
        base._require(event.get("status") in {"attached", "final_registry_observation"}
            and "plan" in event and event.get("owner") in owners, "delegated_scope")
        owner = owners[event["owner"]]["plan"]
        if event["status"] == "attached":
            producer = event["plan"].get("producer")
            base._require(owner.get("role") in {"eraser", "retention"}
                and owner.get("producer") in {"events", "suggestions"}
                and (producer == "events" or owner["producer"] == producer == "suggestions")
                and event.get("borrowed_reader") is (owner["producer"] == "suggestions" and producer == "events"), "delegated_scope")
        else:
            base._require(event["owner"] == "privacy_authority"
                and event["physical_status"] == "not_dispatched_by_final_observation", "delegated_scope")
        _observation_shape(local, source, owners, targets)
        return
    if kind == "composition":
        base._require(set(event) == common | {"instrumented", "global_inventory_complete", "aggregate_copy_complete"}
            and event["instrumented"] == sorted(_SELECTED)
            and event["global_inventory_complete"] is False and event["aggregate_copy_complete"] is False, "event")
        return
    base._require(event.get("owner") in owners, "event_owner")
    if kind == "registry":
        fields = common | {"owner", "digest", "status"}
        base._require(type(event["digest"]) is str and re.fullmatch("[0-9a-f]{64}", event["digest"]), "event")
        if event["status"] == "final_observation":
            fields.add("target_count")
            base._require(type(event["target_count"]) is int and 0 <= event["target_count"] <= 256, "event")
        base._require(set(event) == fields and event["status"] in {"complete", "overflow", "corrupt", "unavailable", "final_observation"}, "event")
    elif kind == "target":
        base._require(type(event["id"]) is str and re.fullmatch("[0-9a-f]{64}", event["id"]), "event")
        fields = common | {"owner", "id", "status"}
        if event["status"] == "attached":
            fields |= {"plan", "borrowed_reader"}
            base._require(type(event["borrowed_reader"]) is bool, "event")
        elif event["status"] == "final_registry_observation":
            fields |= {"participant", "version", "representation_status", "dispatch_coverage", "opaque_digest", "physical_status"}
            if "plan" in event:
                fields.add("plan")
            base._require(event["participant"] in {"events", "suggestions"}
                and type(event["version"]) is int and event["version"] > 0
                and event["representation_status"] in {"registered", "unsupported_version"}
                and event["dispatch_coverage"] == "observed_only"
                and event["physical_status"] in {"not_dispatched_by_final_observation", "unbound_or_unsupported", "nonphysical_or_unsupported"}, "event")
        else:
            fields.add("opaque_digest")
            base._require(event["status"] in {"unsupported", "unbound_or_unsupported"}, "event")
        base._require(set(event) == fields, "event")
        if "opaque_digest" in event:
            value = event["opaque_digest"]
            base._require(value is None or type(value) is str and re.fullmatch("[0-9a-f]{64}", value), "event")
        if "plan" in event:
            _plan_shape(event["plan"], state)
            base._require(event["plan"]["producer"] in {"events", "suggestions"} and event["plan"].get("role") == "historical", "event")
            targets.add((event["owner"], event["plan"]["path"]))
    elif kind == "physical":
        base._require(set(event) == common | {"owner", "path", "state", "identity"}
            and (event["owner"], event["path"]) in targets
            and event["state"] in {"file_absent", "file_existing_uninspected", "representation_absent", "supported", "representation_corrupt_or_unsupported"}
            and (event["identity"] is None or base._identity_shape(event["identity"])), "event")
    else:
        raise RuntimeError("output_registration_event")


def chain(directory: Path) -> tuple[dict, dict, dict, int]:
    """Read v2 only; original v1 parser remains immutable and separately selected."""
    namespace = directory.parents[2]
    _namespace(namespace)
    header, raw_header = base._read(directory / "epoch.json")
    base._require(header.get("schema_version") == 2 and header.get("run_id") == directory.name
        and re.fullmatch("[0-9a-f]{32}", directory.name)
        and header.get("producers") == list(base.PRODUCERS)
        and header.get("namespace_identity") == base._directory(directory)
        and base._identity_shape(header.get("state_identity"))
        and all(type(header.get(k)) is str and re.fullmatch("[0-9a-f]{64}", header[k])
                for k in ("binding_id", "config_sha256"))
        and type(header.get("generation")) is int and header["generation"] > 0
        and type(header.get("selected")) is dict
        and set(header["selected"]) == {"privacy_authority", "authority_sessions", "routines", "events", "suggestions"}
        and all(type(v) is bool for v in header["selected"].values())
        and set(header) == {"schema_version", "run_id", "generation", "binding_id", "config_sha256", "state_root", "state_identity", "namespace_identity", "producers", "selected"}, "header")
    state = Path(header["state_root"])
    base._require(state.is_absolute() and str(state) == header["state_root"] and ".." not in state.parts, "header")
    head, raw_head = base._read(directory / "HEAD.json")
    base._require(set(head) == {"revision", "sha256"} and type(head["revision"]) is int
                  and 0 <= head["revision"] <= base.MAX_EVENTS, "head")
    paths = list(directory.iterdir())
    by_revision = {}
    for path in paths:
        match = re.fullmatch(r"(\d{12})-([0-9a-f]{64})\.json", path.name)
        if match:
            revision = int(match[1])
            base._require(revision not in by_revision, "orphan")
            by_revision[revision] = path
    base._require(len(paths) == head["revision"] + 2 and len(by_revision) == head["revision"], "orphan")
    previous, total, states = base._sha(raw_header), len(raw_header) + len(raw_head), {}
    targets, policy_seen = set(), False
    for revision in range(1, head["revision"] + 1):
        path = by_revision[revision]
        event, raw = base._read(path)
        total += len(raw)
        digest = base._sha(raw)
        base._require(path.name == f"{revision:012d}-{digest}.json" and event["revision"] == revision
                      and event["previous_sha256"] == previous, "chain")
        if event["kind"] == "owner":
            base._require(set(event) == {"kind", "owner", "phase", "plan", "native_identity", "revision", "previous_sha256"}, "event")
            key, phase, plan = event["owner"], event["phase"], event["plan"]
            _plan_shape(plan, state)
            prior = states.get(key)
            base._require(phase in _PHASES and (None if prior is None else prior["phase"]) in _PHASES[phase], "transition")
            base._require(prior is None or prior["plan"] == plan, "plan_changed")
            _owner_event_shape(event, prior, header["selected"])
            states[key] = {"phase": phase, "plan": plan, "native_identity": event["native_identity"],
                           "revision": revision, "sha256": digest}
        else:
            if event["kind"] == "delegation_policy":
                base._require(not policy_seen, "delegated_policy")
                policy_seen = True
            if event["kind"] == "delegated_target":
                base._require(policy_seen, "delegated_policy")
            _observation_shape(event, state, states, targets)
        previous = digest
    base._require(previous == head["sha256"] and total <= base.MAX_RUN_BYTES, "head")
    base._require(base._read(directory / "HEAD.json")[1] == raw_head
                  and base._read(directory / "epoch.json")[1] == raw_header, "journal_changed")
    return header, head, states, total


@dataclass(frozen=True, eq=False)
class _JournalFormat:
    """Fixed private journal encoding policy; never a coordinator or authority."""
    schema_version: int
    namespace_name: str
    known_versions: frozenset
    selected_producers: frozenset
    decode_chain: Callable

    def __reduce_ex__(self, protocol: int):
        raise TypeError("output_journal_format_not_serializable")


V2_FORMAT = _JournalFormat(schema_version=2, namespace_name="v2", known_versions=frozenset({"v1", "v2"}),
                           selected_producers=_SELECTED, decode_chain=chain)


def _known_format(value: object) -> _JournalFormat:
    if value is V2_FORMAT:
        return value
    from src.memory.output_registration_v3 import V3_FORMAT
    from src.memory.output_registration_v4 import V4_FORMAT
    if value is not V3_FORMAT and value is not V4_FORMAT:
        raise RuntimeError("output_registration_format")
    return value


class OutputRegistrationV2(base.DestinationRegistrationCoordinator):
    """Observe five additional real owner families without taking their close."""

    def __init__(self, admission: RuntimeAdmission, config: dict, *, key: object,
                 journal_format: _JournalFormat | None = None) -> None:
        super().__init__(admission, config, key=key)
        self._format = _known_format(V2_FORMAT if journal_format is None else journal_format)
        self._directory = (self._directory.parents[2] / self._format.namespace_name / "runs"
                           / admission._ticket["run_id"])
        self._namespace = self._directory.parents[2]
        self._roles: dict[str, _Owner] = {}
        self._by_instance: dict[int, _Owner] = {}
        self._targets: dict[str, dict] = {}
        self._observations: dict[tuple, tuple] = {}
        self._delegation_policy_json: bytes | None = None
        self._open_extents: dict[int, tuple] = {}
        self._opening_extents = 0
        self._dispatches = WeakKeyDictionary()
        self._dispatch_lock = threading.RLock()
        self._reservations: dict[int, ConstructorReservation] = {}
        self._constructor_returns: dict[int, object] = {}
        self._expected.update({
            "privacy_authority": os.fspath(config.get("memory", {}).get("privacy_db_path", "./data/memory_privacy.db")),
            "authority_sessions": os.fspath(config.get("dashboard", {}).get("auth", {}).get("db_path", "./data/dashboard_auth.db")),
            "routines": os.fspath(config.get("routines", {}).get("db_path", "./data/routines.db")),
            "events": os.fspath(config.get("analytics", {}).get("events_db", "./data/events.db")),
            "suggestions": os.fspath(config.get("analytics", {}).get("suggestions_path", "./data/suggestions.json")),
        })
        self._selected = {"authority_sessions": bool(config.get("dashboard", {}).get("enabled", True)),
                          "events": bool(config.get("analytics", {}).get("enabled", True)),
                          "suggestions": bool(config.get("analytics", {}).get("enabled", True)),
                          "privacy_authority": True, "routines": True}
        self._accounts: _AccountPublisher | None = None
        self._accounts_disabled = False
        self._configuration: _ConfigurationPublisher | None = None
        self._configuration_disabled = False
        # OQ-8: the boot consent decision, computed once in _start and written
        # unchanged on this run's first configuration record.
        reasoner = config.get("reasoner")
        cloud = reasoner.get("cloud") if type(reasoner) is dict else None
        self._consent_granted = type(cloud) is dict and cloud.get("consent") is True
        self._prior_decision = None
        self._effective_decision: dict | None = None
        self._settled_decision: dict | None = None
        if self._format.schema_version in {3, 4}:
            self._expected["accounts"] = os.fspath(config.get("speaker_id", {}).get("users_path", "./data/users.json"))
            self._selected["accounts"] = self._selected["authority_sessions"]
        if self._format.schema_version == 4:
            # main constructs ConfigurationChanges exactly when the dashboard is
            # enabled under an admission; the destination is never configured.
            self._expected["configuration"] = str(admission.state_root / "config" / "settings.yaml")
            self._selected["configuration"] = self._selected["authority_sessions"]

    @property
    def prior_decision(self):
        """The reader's PriorConsentDecision for this run (None before _start or below V4)."""
        return self._prior_decision

    @property
    def effective_decision(self) -> dict | None:
        """Exactly the decision this run records; non-None closes the boot gate."""
        return self._effective_decision

    @property
    def settled_decision(self) -> dict | None:
        """A prior decision this boot superseded because consent was not granted."""
        return self._settled_decision

    def _decide_consent(self, runs: Path, run_id: str) -> None:
        from src.memory.output_registration_v4 import prior_consent_decision
        prior = prior_consent_decision(runs, run_id)
        self._prior_decision = prior
        if not self._consent_granted:
            # Settling: no grant in the effective config clears the marker.
            self._effective_decision = None
            if prior.decision is not None:
                self._settled_decision = prior.decision
                source = prior.decision["source"]
                logger.warning(
                    "Consentimiento de nube: el arranque sin consentimiento asentó el retiro previo. "
                    "Cloud consent marker settled by a boot without consent (superseded basis=%s, "
                    "source run=%s)", prior.decision["basis"], source["run_id"])
        else:
            self._effective_decision = prior.decision

    def _current(self) -> None:
        # Metadata carries no opening grant. Full validation is performed at
        # explicit admission boundaries, never beneath this journal lock.
        base._require(self.admission._output_registration is self and not self._failed, "coordinator")
        self.admission._revalidate_live()
        self._current_pins()
        if self._namespace.exists():
            for version in self._namespace.iterdir():
                base._require(version.name in self._format.known_versions and version.is_dir()
                    and (version / "runs").is_dir(), "unsupported_version")

    async def register_store(self, store: ListStore | ReminderStore, producer: str) -> base.DestinationRegistration:
        """Keep inherited list/reminder attachment behind full validation."""
        await self._run(self.admission.revalidate)
        return await super().register_store(store, producer)

    def _opening(self, handle: base.DestinationRegistration, store: object,
                 producer: str, path: str) -> None:
        # Both inherited async SQLite owners open outside OwnedBlockingWork.
        # Validate before entering the original coordinator critical section.
        self.admission.revalidate()
        super()._opening(handle, store, producer, path)

    def _preflight(self, execution: object) -> None:
        """Validate a recognized original callback before it acquires native locks."""
        roles = tuple(self._roles.values())
        allowed = tuple(binding for binding in roles
            if binding.work is execution.work and self._call_allowed(binding, execution))
        allowed += self._account_owners(execution)
        final = tuple(binding for binding in roles
            if binding.final_work is execution.work and self._call_allowed(binding, execution)
            and execution.parent_sequence is binding.final_sequence
            and execution.parent_sequence is not None)
        # Ordinary already-open privacy reads/writes and owner cleanup keep
        # their existing behavior; they acquire no connection-opening grant.
        if not allowed and not final:
            return
        try:
            with self._dispatch_lock:
                for previous in tuple(self._dispatches):
                    if previous.task.done():
                        del self._dispatches[previous]
                if len(self._dispatches) >= _MAX_NATIVE_WITNESSES:
                    _reject("native_capacity", self)
            if final:
                for binding in final:
                    self._close_binding(binding.instance)
                    if binding.instance._closing is None or binding.instance._closing.done():
                        _reject("close_owner", self)
            else:
                self._live()
            self.admission._preflight_native(execution)
            if not final:
                self._current()
            with self._dispatch_lock:
                if len(self._dispatches) >= _MAX_NATIVE_WITNESSES:
                    _reject("native_capacity", self)
                self._dispatches[execution] = allowed + final
        except BaseException:
            self._failed = True
            raise

    def _dispatch(self, binding: _Owner, execution: object) -> None:
        with self._dispatch_lock:
            admitted = self._dispatches.get(execution, ())
        if not any(item is binding for item in admitted):
            _reject("native_owner", self)
        self.admission._revalidate_native(execution)

    def _start(self):
        with self._lock:
            self._current()
            ticket = self.admission._ticket
            root = Path(ticket["admission_root"]) / "admission"
            base._directory(root)
            size = _namespace(self._namespace, reserve_entries=6)
            if self._namespace.exists():
                for version in self._namespace.iterdir():
                    base._require(version.name in self._format.known_versions and version.is_dir() and (version / "runs").is_dir(), "unsupported_version")
                    base._require(not (version / "runs" / ticket["run_id"]).exists(), "run_exists")
            grammars = None
            if self._format.schema_version >= 4:
                from src.memory.output_registration_v4 import WRITER_GRAMMARS, refuse_unknown_grammars
                # Never start beside a run whose declared grammar this release cannot read.
                refuse_unknown_grammars(self._namespace / self._format.namespace_name / "runs")
                grammars = dict(WRITER_GRAMMARS)
                # OQ-8: read the authoritative decision before this run exists.
                self._decide_consent(self._namespace / self._format.namespace_name / "runs", ticket["run_id"])
            header = {"schema_version": self._format.schema_version, "run_id": ticket["run_id"], "generation": ticket["generation"],
                      "binding_id": ticket["binding_id"], "config_sha256": self.admission._config_digest,
                      "state_root": str(self.admission.state_root), "state_identity": base._identity(self.admission.state_root),
                      "namespace_identity": {"device": 2**64 - 1, "inode": 2**64 - 1},
                      "producers": list(base.PRODUCERS), "selected": self._selected}
            if grammars is not None:
                header["grammars"] = grammars
            raw = base._encoded(header)
            reservation = len(raw) + 2 * len(base._encoded({"revision": 0, "sha256": "0"*64}))
            base._require(reservation <= base.MAX_RUN_BYTES and size + reservation <= base.MAX_TOTAL_BYTES, "capacity")
            for part in ("output-inventory", self._format.namespace_name, "runs", ticket["run_id"]):
                child = root / part
                if not child.exists():
                    child.mkdir(mode=0o700)
                    base._fsync(root)
                base._directory(child)
                root = child
            self._directory_identity = base._directory(root)
            header["namespace_identity"] = self._directory_identity
            self._epoch = header
            raw = base._encoded(header)
            base._write_new(root / "epoch.json", raw)
            self._epoch_identity = base._identity(root / "epoch.json")
            base._head(root, None, {"revision": 0, "sha256": base._sha(raw)})

    def _record(self, value: dict) -> None:
        with self._lock:
            self._current()
            _, head, _, size = self._format.decode_chain(self._directory)
            value = dict(value, revision=head["revision"] + 1, previous_sha256=head["sha256"])
            raw = base._encoded(value)
            digest = base._sha(raw)
            next_head = {"revision": value["revision"], "sha256": digest}
            base._require(value["revision"] <= base.MAX_EVENTS and size + len(raw) + len(base._encoded(next_head)) <= base.MAX_RUN_BYTES, "capacity")
            base._require(_namespace(self._namespace, reserve_entries=2) + len(raw) + len(base._encoded(next_head)) <= base.MAX_TOTAL_BYTES, "capacity")
            base._write_new(self._directory / f"{value['revision']:012d}-{digest}.json", raw)
            self._current()
            base._head(self._directory, head, next_head)

    def _append(self, producer, phase, plan):
        with self._lock:
            _, _, states, _ = self._format.decode_chain(self._directory)
            prior = states.get(producer)
            base._require(phase in _PHASES and (None if prior is None else prior["phase"]) in _PHASES[phase], "transition")
            native = None
            if phase in {"active", "closed"} and plan.get("path"):
                path = Path(plan["path"])
                info = path.lstat()
                base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "native_identity")
                native = base._identity(path)
                expected = plan.get("existing_identity") if phase == "active" else prior["native_identity"]
                base._require(expected is None or native == expected, "native_identity")
            event = {"kind": "owner", "owner": producer, "phase": phase, "plan": plan, "native_identity": native}
            _plan_shape(plan, self.admission.state_root)
            _owner_event_shape(event, prior, self._selected)
            self._record(event)

    def _live(self) -> None:
        from src.core.runtime_admission import _verified_admissions
        if self._failed or self.admission._output_registration is not self or _verified_admissions.get(self.admission) is not True:
            _reject("coordinator", self)

    def _nominal(self, owner: object, producer: str, role: str):
        from src.memory.privacy_services import PrivacyServices
        from src.dashboard.authority import DashboardAuthority
        from src.routines.routine_store import RoutineStore
        from src.analytics.event_logger import EventLogger
        from src.analytics.erasure import EventEraser
        from src.analytics.retention import EventRetention
        from src.analytics.suggestion_store import SuggestionStore
        from src.analytics.suggestion_erasure import SuggestionEraser
        from src.analytics.suggestion_retention import SuggestionRetention
        expected = {("privacy_authority", "primary"): PrivacyServices, ("authority_sessions", "primary"): DashboardAuthority,
                    ("routines", "primary"): RoutineStore, ("events", "primary"): EventLogger,
                    ("events", "eraser"): EventEraser, ("events", "retention"): EventRetention,
                    ("suggestions", "primary"): SuggestionStore, ("suggestions", "eraser"): SuggestionEraser,
                    ("suggestions", "retention"): SuggestionRetention}
        if type(owner) is not expected.get((producer, role)):
            _reject("owner", self)

    async def attach(self, owner: object, producer: str, *, role: str = "primary") -> None:
        self._nominal(owner, producer, role)
        key = producer if role == "primary" else producer + "." + role
        if key in self._roles or id(owner) in self._by_instance:
            _reject("owner_duplicate", self)
        path = None
        if role == "primary":
            path = os.fspath(owner.ledger.path if producer == "privacy_authority" else owner.path if producer == "authority_sessions" else owner._db_path if producer == "routines" else owner.db_path)
            if str(Path(path).absolute()) != str(Path(self._expected[producer]).absolute()):
                _reject("configured_path", self)
        work = getattr(owner, "_disk", None) if (producer, role) == ("suggestions", "eraser") else getattr(owner, "_work", None)
        binding = _Owner(owner, producer, role, key, path, work, grants={})
        # Retain exact original owner before the first fallible await.
        self._roles[key] = self._by_instance[id(owner)] = binding
        owner._output_coordinator = self
        if work is not None:
            work._install_native_preflight(self, self._preflight)
        await self._run(self.admission.revalidate)
        plan = await self._run(_plan, producer, role, path, self.admission.state_root)
        handle = base.DestinationRegistration(self, owner, key, path or "")
        self._handles[key] = (handle, plan)
        await self._run(self._append, key, "intent", plan)
        await self._run(self._append, key, "attached", plan)
        owner._destination_registration = handle
        binding.phase = "attached"
        if producer == "privacy_authority":
            owner.ledger._output_coordinator = self
            owner.ledger._output_privacy_owner = owner
        if role != "primary":
            await self.refresh(owner)
            await self._run(self._append, key, "active", plan)
            binding.phase = "active"

    def _owner(self, owner: object) -> _Owner:
        self._live()
        binding = self._by_instance.get(id(owner))
        if binding is None or binding.instance is not owner:
            _reject("attachment", self)
        return binding

    def _call_allowed(self, binding: _Owner, execution: object) -> bool:
        function = execution.function
        if binding.producer == "privacy_authority":
            return getattr(function, "__self__", None) is binding.instance and getattr(function, "__func__", None) is type(binding.instance).initialize
        if binding.producer == "authority_sessions":
            return getattr(function, "__self__", None) is binding.instance and getattr(function, "__func__", None) is type(binding.instance)._transaction
        if binding.role == "primary":
            event_owner = binding.instance if binding.producer == "events" else binding.instance.events
            call = _CALL.get()
            return call is not None and call[0] is event_owner and call[1] is function
        from src.analytics.retention import sweep_events
        from src.analytics.suggestion_retention import sweep_suggestions
        from src.analytics.suggestion_erasure import cleanup_suggestions
        if binding.producer == "events" and binding.role == "eraser":
            return getattr(function, "__self__", None) is binding.instance and getattr(function, "__func__", None) is type(binding.instance)._clean
        expected = {("events", "retention"): sweep_events,
                    ("suggestions", "retention"): sweep_suggestions,
                    ("suggestions", "eraser"): cleanup_suggestions}
        return function is expected.get((binding.producer, binding.role))

    def _native(self, binding: _Owner, path: str | Path, *, history: bool = False):
        execution = current_native_execution()
        final_rotation = (binding.final_execution is execution and binding.final_work is not None
            and validate_native_execution(execution, work=binding.final_work))
        try:
            if final_rotation:
                self._dispatch(binding, execution)
                self._close_binding(binding.instance)
            else:
                self._current()
        except BaseException:
            self._failed = True
            raise
        work = binding.work
        allowed = work is not None and validate_native_execution(execution, work=work)
        if not allowed and binding.final_work is not None:
            allowed = validate_native_execution(execution, work=binding.final_work)
        if allowed:
            allowed = self._call_allowed(binding, execution)
        if not allowed or binding.phase not in {"attached", "opening", "active"}:
            _reject("native_owner", self)
        self._dispatch(binding, execution)
        plan = self._handles[binding.key][1]
        physical = str(Path(path).resolve(strict=False))
        if history:
            self._admit_history(binding, physical, execution, path)
        else:
            expected = binding.native_identity or plan["existing_identity"]
            try:
                _plan_shape(plan, self.admission.state_root)
                _physical_shape(path, plan, expected_identity=expected, allow_absent=expected is None)
            except (OSError, RuntimeError, ValueError):
                _reject("destination_changed", self)
        if binding.phase == "attached":
            try:
                self._append(binding.key, "opening", plan)
            except BaseException:
                self._failed = True
                raise
            binding.phase = "opening"
        return execution

    def before_native(self, owner: object, path: str | Path) -> object:
        return self._native(self._owner(owner), path)

    def ready_native(self, owner: object) -> None:
        """Capture preparation inside native initialization, never publish active."""
        binding = self._owner(owner)
        execution = current_native_execution()
        if not validate_native_execution(execution, work=binding.work) or binding.phase != "opening" or not self._call_allowed(binding, execution):
            _reject("initialization_owner", self)
        binding.initialized_execution = execution

    async def initializer_returned(self, owner: object) -> None:
        """Publish after native return while the original worker retains sequence."""
        binding = self._owner(owner)
        execution = binding.initialized_execution
        if execution is None or execution.retained_task is not asyncio.current_task() or execution.retained_task not in binding.work._tasks:
            _reject("initialization_owner", self)
        if not execution.task.done() or execution.task.cancelled() or execution.task.exception() is not None:
            _reject("initialization_pending", self)
        await self._run(self._append, binding.key, "active", self._handles[binding.key][1])
        binding.phase = "active"
        binding.native_identity = base._identity(Path(self._handles[binding.key][1]["path"]))
        if binding.producer == "privacy_authority":
            owner._ready = True
        elif binding.producer == "events":
            owner._initialized = True
        else:
            _reject("initialization_owner", self)

    async def ready_async(self, owner: object) -> None:
        binding = self._owner(owner)
        expected = owner._initializing if binding.producer == "authority_sessions" else self._initializers.get(binding.key)
        if expected is not asyncio.current_task() or binding.phase != "opening":
            _reject("initialization_owner", self)
        if binding.producer == "routines" and (owner._db is None or owner._db._connection is None or owner._work is None or owner._work._connection is not owner._db or not owner._lock.locked()):
            _reject("initialization_owner", self)
        await self._run(self._append, binding.key, "active", self._handles[binding.key][1])
        binding.phase = "active"
        binding.native_identity = base._identity(Path(self._handles[binding.key][1]["path"]))

    async def routine_opening(self, owner: object) -> None:
        binding = self._owner(owner)
        if not owner._lock.locked() or owner._db is not None:
            _reject("initialization_owner", self)
        plan = self._handles[binding.key][1]
        def opening() -> None:
            self.admission.revalidate()
            self._current()
            _physical_shape(owner._db_path, plan, expected_identity=plan["existing_identity"],
                allow_absent=plan["existing_identity"] is None)
            self._append(binding.key, "opening", plan)
        await self._run(opening)
        binding.phase = "opening"
        self._initializers[binding.key] = asyncio.current_task()

    def _close_binding(self, owner: object) -> _Owner:
        from src.core.runtime_admission import _verified_admissions
        binding = self._by_instance.get(id(owner))
        if (binding is None or binding.instance is not owner or binding.producer != "authority_sessions"
                or binding.phase != "active" or owner._lease_fd is None or not owner._ready
                or self.admission._output_registration is not self
                or _verified_admissions.get(self.admission) is not True):
            _reject("close_owner", self)
        return binding

    def bind_final_worker(self, owner: object, work: object, args: tuple) -> None:
        """Bind only the original close task's final rotation, even if metadata failed."""
        from src.core.owned_work import OwnedBlockingWork
        binding = self._close_binding(owner)
        if (owner._closing is not asyncio.current_task() or binding.final_work is not None
                or type(work) is not OwnedBlockingWork or work.status()["closed"] or work.status()["operations"]
                or not binding.work.status()["closed"] or binding.work.status()["operations"]
                or args != (owner._observed_digest, owner._observed_epoch)):
            _reject("close_owner", self)
        binding.final_work, binding.final_args = work, args
        work._install_native_preflight(self, self._preflight)

    def dashboard_operation(self, owner: object, action: str, args: tuple, guard: object) -> None:
        """A final worker may only persist the exact already-owned rotation fence."""
        binding = self._by_instance.get(id(owner))
        execution = current_native_execution()
        if binding is not None and binding.final_work is not None and validate_native_execution(execution, work=binding.final_work):
            self._close_binding(owner)
            if (owner._closing is None or owner._closing.done()
                    or getattr(execution.function, "__self__", None) is not owner
                    or getattr(execution.function, "__func__", None) is not type(owner)._transaction
                    or execution.parent_sequence is None or execution.parent_sequence is not binding.final_sequence
                    or action != "rotate" or args != binding.final_args or guard is not None):
                _reject("close_operation", self)
            binding.final_execution = execution

    def native_owner(self, owner: object) -> _Owner:
        binding = self._by_instance.get(id(owner))
        execution = current_native_execution()
        if (binding is not None and binding.final_execution is execution
                and binding.final_work is not None and validate_native_execution(execution, work=binding.final_work)):
            return self._close_binding(owner)
        return self._owner(owner)

    async def closed_owner(self, owner: object, *, connection: object | None = None) -> None:
        binding = self._by_instance.get(id(owner))
        if binding is None:
            # An original owner may already be installed when attachment
            # fails. Its native close must still drain; the epoch stays failed.
            self._failed = True
            return
        if owner._closing is not asyncio.current_task():
            _reject("close_owner", self)
        if binding.producer == "routines" and (owner._db is not None or owner._work is not None or (connection is not None and connection._connection is not None)):
            _reject("native_close_pending", self)
        if binding.work is not None:
            status = binding.work.status()
            if not status["closed"] or status["operations"]:
                _reject("native_close_pending", self)
        executor = getattr(owner, "_executor", None)
        if executor is not None and any(thread.is_alive() for thread in tuple(executor._threads)):
            _reject("native_close_pending", self)
        if binding.final_work is not None and (not binding.final_work.status()["closed"] or binding.final_work.status()["operations"]):
            _reject("native_close_pending", self)
        if any(item[0] is binding for item in self._open_extents.values()):
            _reject("native_close_pending", self)
        if binding.producer == "privacy_authority" and (owner.ledger._db is not None or owner._lease is not None):
            _reject("native_close_pending", self)
        if binding.producer == "authority_sessions" and owner._lease_fd is not None:
            _reject("native_close_pending", self)
        if self._failed:
            return
        if binding.key not in self._handles:
            self._failed = True
            return
        phase = ("closed" if binding.phase == "active" else
                 "native_failed_then_actually_closed" if binding.phase == "opening" else "unresolved")
        await self._run(self._append, binding.key, phase, self._handles[binding.key][1])
        binding.phase = phase
        # A SuggestionStore borrows EventLogger's actual drain, not a new close.
        if binding.producer == "events" and binding.role == "primary":
            for borrowed in tuple(self._roles.values()):
                if borrowed.producer == "suggestions" and borrowed.role == "primary" and borrowed.work is binding.work:
                    terminal = ("closed" if borrowed.phase == "active" else
                                "native_failed_then_actually_closed" if borrowed.phase == "opening" else "unresolved")
                    await self._run(self._append, borrowed.key, terminal, self._handles[borrowed.key][1])
                    borrowed.phase = terminal

    async def reserve_constructor(self, events: object, path: str) -> ConstructorReservation:
        binding = self._owner(events)
        if binding.producer != "events" or binding.role != "primary" or binding.phase != "active":
            _reject("constructor_owner", self)
        from src.analytics.suggestion_storage import suggestion_descriptor
        if str(Path(path).absolute()) != str(Path(self._expected["suggestions"]).absolute()):
            _reject("configured_path", self)
        await self._run(self.admission.revalidate)
        plan = await self._run(_plan, "suggestions", "primary", suggestion_descriptor(path)["path"], self.admission.state_root)
        await self._run(self._append, "suggestions", "intent", plan)
        reservation = ConstructorReservation(self, events, path)
        self._reservations[id(reservation)] = reservation
        self._constructor_plan = plan
        return reservation

    def constructor_enter(self, store: object, reservation: ConstructorReservation | None) -> None:
        self._nominal(store, "suggestions", "primary")
        if type(reservation) is not ConstructorReservation or self._reservations.get(id(reservation)) is not reservation or reservation.coordinator is not self:
            _reject("constructor_reservation", self)
        parent = self._owner(reservation.events)
        from src.analytics.suggestion_engine import SuggestionEngine
        execution = current_native_execution()
        if not validate_native_execution(execution, work=parent.work, function=SuggestionEngine):
            _reject("constructor_owner", self)
        self._dispatch(parent, execution)
        binding = _Owner(store, "suggestions", "primary", "suggestions", self._constructor_plan["path"], parent.work, grants={})
        # Only the original reserved constructor can join its actual child to
        # this already-validated invocation; no late historical grants change.
        with self._dispatch_lock:
            self._dispatches[execution] = self._dispatches[execution] + (binding,)
        self._roles["suggestions"] = self._by_instance[id(store)] = binding
        store._output_coordinator = self
        handle = base.DestinationRegistration(self, store, "suggestions", reservation.path)
        store._destination_registration = handle
        self._handles["suggestions"] = (handle, self._constructor_plan)
        try:
            self._append("suggestions", "attached", self._constructor_plan)
        except BaseException:
            self._failed = True
            raise
        binding.phase = "attached"
        self._constructor_returns[id(reservation)] = store

    async def refresh(self, owner: object) -> TargetRegistrySnapshot:
        binding = self._owner(owner)
        privacy = owner.privacy
        try:
            snapshot = await privacy.acall(privacy.ledger.erasures.target_snapshot)
        except BaseException:
            self._failed = True
            raise
        if snapshot.ledger is not privacy.ledger or snapshot.status != "complete":
            await self._run(self._record, {"kind": "registry", "owner": binding.key, "digest": snapshot.digest, "status": snapshot.status})
            _reject("registry_unavailable", self)
        if binding.snapshot is not None and binding.snapshot.digest == snapshot.digest:
            return snapshot
        grants = {}
        for entry in snapshot.entries:
            descriptor = entry.descriptor
            producer = "events" if entry.participant.value == "events" else "suggestions"
            if entry.status != "registered" or not isinstance(descriptor, dict) or "path" not in descriptor:
                await self._run(self._record, {"kind": "target", "owner": binding.key, "id": entry.id,
                    "status": "unsupported", "opaque_digest": entry.opaque_digest})
                continue
            try:
                from src.analytics.event_storage import validate_event_descriptor
                from src.analytics.suggestion_storage import validate_suggestion_descriptor
                (validate_event_descriptor if producer == "events" else validate_suggestion_descriptor)(descriptor)
                plan, scope_json, policy_json = await self._disk.run(_historical_plan, self.admission, entry)
            except (ValueError, RuntimeError, OSError):
                await self._run(self._record, {"kind": "target", "owner": binding.key, "id": entry.id,
                    "status": "unbound_or_unsupported", "opaque_digest": entry.opaque_digest})
                continue
            if binding.producer == "events" and producer != "events":
                continue
            if plan["path"] in grants and grants[plan["path"]].entry.id != entry.id:
                _reject("target_path_conflict", self)
            grants[plan["path"]] = _HistoricalGrant(entry, base._encoded(plan), scope_json)
            record = {"kind": "target", "owner": binding.key, "id": entry.id,
                "status": "attached", "plan": plan, "borrowed_reader": binding.producer == "suggestions" and producer == "events"}
            if scope_json is not None:
                await self._run(self._record_historical_policy, policy_json)
                record.update(kind="delegated_target", scope=json.loads(scope_json))
            await self._run(self._record, record)
        await self._run(self._record, {"kind": "registry", "owner": binding.key, "digest": snapshot.digest, "status": "complete"})
        binding.registry_state = (snapshot, tuple(grants.items()))
        binding.grants, binding.snapshot = grants, snapshot
        return snapshot

    def _record_historical_policy(self, policy_json: bytes) -> None:
        """Publish the exact running policy once; no event becomes a grant."""
        with self._lock:
            base._require(type(policy_json) is bytes
                and base._sha(policy_json) == self.admission._binding["ownership_sha256"], "delegated_policy")
            if self._delegation_policy_json is not None:
                base._require(self._delegation_policy_json == policy_json, "delegated_policy")
                return
            policy = json.loads(policy_json)
            base._require(base._encoded(policy) == policy_json, "delegated_policy")
            self._record({"kind": "delegation_policy", "scope_schema_version": 1,
                "ownership_sha256": base._sha(policy_json), "policy": policy})
            self._delegation_policy_json = policy_json

    def _join(self, binding: _Owner, physical: str, registry_state: tuple, execution: object) -> _Operation:
        previous, pairs = registry_state
        grants = dict(pairs)
        if physical not in grants:
            _reject("target_unregistered", self)
        grant = grants[physical]
        self._history_scope(grant, execution)
        ledger = binding.instance.privacy.ledger
        with ledger._transaction() as db:
            snapshot = ledger.erasures._target_snapshot_db(db, previous.participants, limit=256)
            if snapshot.status != "complete" or snapshot.digest != previous.digest or snapshot.lifecycle_generation != previous.lifecycle_generation:
                raise RegistryChanged("output_registry_changed")
            expected = grant.entry
            if not any(e.id == expected.id and e.participant is expected.participant and e.version == expected.version
                       and e.descriptor_json == expected.descriptor_json for e in snapshot.entries):
                _reject("target_unregistered", self)
            admission = _Operation(weakref.ref(execution), binding, previous, pairs)
        return admission

    def _history_scope(self, grant: _HistoricalGrant, execution: object) -> Path:
        if grant.scope_json is None:
            return self.admission.state_root
        try:
            observed, policy_json = self.admission._native_historical_target(execution, _target_record(grant.entry))
            base._require(observed == grant.scope_json, "delegated_scope")
            base._require(policy_json == self._delegation_policy_json, "delegated_policy")
            return _scope_shape(json.loads(observed), self.admission.state_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            _reject("target_unregistered", self)

    def _history_path(self, path: str | Path, grant: _HistoricalGrant, execution: object) -> None:
        source = self._history_scope(grant, execution)
        plan = json.loads(grant.plan_json)
        try:
            _plan_shape(plan, source)
            base._require(str(Path(path).absolute()) == plan["path"] == grant.entry.descriptor["path"], "destination_changed")
            _physical_shape(path, plan, expected_identity=plan["existing_identity"], allow_absent=True)
        except (OSError, RuntimeError, ValueError):
            _reject("destination_changed", self)

    def _admit_history(self, binding: _Owner, physical: str, execution: object, supplied: str | Path) -> _Operation:
        previous = _OPERATION.get()
        if isinstance(previous, _Operation) and execution is not None and previous.execution() is execution:
            if previous.owner is not binding or not any(path == physical for path, _ in previous.grants):
                _reject("target_unregistered", self)
            admission = previous
        else:
            registry_state = binding.registry_state
            if registry_state is None:
                raise OutputNotReady("output_registration_waiting")
            admission = self._join(binding, physical, registry_state, execution)
            _OPERATION.set(admission)
        self._history_path(supplied, dict(admission.grants)[physical], execution)
        return admission

    def observation_entry(self, path: str | Path, kind: str) -> _Owner:
        admission = _OPERATION.get()
        execution = current_native_execution()
        physical = str(Path(path).resolve(strict=False))
        if not isinstance(admission, _Operation) or execution is None or admission.execution() is not execution:
            _reject("target_operation", self)
        binding = admission.owner
        if not validate_native_execution(execution, work=binding.work) or not self._call_allowed(binding, execution):
            _reject("target_operation", self)
        if not any(p == physical for p, _ in admission.grants) or (kind == "suggestions" and binding.producer != kind):
            _reject("target_operation", self)
        return binding

    def history_entry(self, path: str | Path, kind: str) -> tuple[_Owner, object]:
        execution = current_native_execution()
        candidates = [binding for binding in self._roles.values() if binding.role != "primary"
                      and validate_native_execution(execution, work=binding.work)]
        if len(candidates) != 1:
            _reject("native_owner", self)
        binding = candidates[0]
        if binding.phase != "active":
            raise OutputNotReady("output_registration_waiting")
        if kind == "suggestions" and binding.producer != "suggestions":
            _reject("native_role", self)
        return binding, self._native(binding, path, history=True)

    def observe(self, binding: _Owner, path: str | Path, state: str) -> None:
        key = binding.key, str(Path(path).resolve(strict=False))
        identity = base._identity(Path(path)) if Path(path).exists() else None
        with self._lock:
            previous = self._observations.get(key)
            if previous == (state, identity):
                return
            # Opening the same physical file is not a new lifetime or a loss
            # of its already-observed representation. Ordinary SQL work must
            # not emit two journal fsyncs for an unchanged database.
            if state == "file_existing_uninspected" and previous is not None and previous[1] == identity:
                return
            try:
                self._record({"kind": "physical", "owner": binding.key, "path": key[1], "state": state, "identity": identity})
                self._observations[key] = (state, identity)
            except BaseException:
                self._failed = True
                raise

    async def composed(self, graph: object | None = None) -> None:
        self._live()
        if graph is not None and "accounts" in self._selected:
            # Re-pin the composed public graph to the exact account owner. The
            # ApplicationGraph exposes the publisher only through its dashboard.
            publisher, missing = self._accounts, object()
            dashboard = getattr(graph, "dashboard", missing)
            if publisher is not None:
                pinned = (dashboard is publisher.dashboard
                          and getattr(dashboard, "user_manager", None) is publisher.manager)
            else:
                pinned = self._accounts_disabled and dashboard is None
            if not pinned:
                _reject("account_owner", self)
        if graph is not None and "configuration" in self._selected and (
                self._configuration is not None or self._configuration_disabled):
            # Re-pin the composed public graph to the exact configuration owner
            # through the real ApplicationGraph fields; never a synthetic alias.
            publisher, missing = self._configuration, object()
            changes = getattr(graph, "configuration_changes", missing)
            authority = getattr(graph, "dashboard_authority", missing)
            if publisher is not None:
                sessions = self._roles.get("authority_sessions")
                pinned = (publisher.phase == "attached" and changes is publisher.owner
                          and authority is publisher.authority and authority is publisher.owner._authority
                          and sessions is not None and sessions.instance is authority
                          and publisher.owner._graph is graph)
            else:
                pinned = changes is None and authority is None
            if not pinned:
                _reject("configuration_owner", self)
        for producer, selected in self._selected.items():
            if producer == "configuration":
                publisher = self._configuration
                if (selected and (publisher is None or publisher.phase != "attached")
                        or not selected and (publisher is not None or not self._configuration_disabled)):
                    _reject("composition_incomplete", self)
                continue
            if producer == "accounts":
                publisher = self._accounts
                if (selected and (publisher is None or publisher.phase not in {"attached", "opening"})
                        or not selected and (publisher is not None or not self._accounts_disabled)):
                    _reject("composition_incomplete", self)
                continue
            binding = self._roles.get(producer)
            if not selected:
                if binding is not None:
                    _reject("selection_changed", self)
                plan = {"kind": "worker", "producer": producer, "role": "primary"}
                await self._run(self._append, producer, "disabled_never_opened", plan)
            elif binding is None or (binding.phase != "active" and not (
                producer == "routines" and binding.phase == "native_failed_then_actually_closed"
            )):
                _reject("composition_incomplete", self)
        for key in ("events.eraser", "events.retention", "suggestions.eraser", "suggestions.retention"):
            if key not in self._roles or self._roles[key].phase != "active":
                _reject("composition_incomplete", self)
        for key in ("lists", "reminders"):
            if key not in self._handles:
                _reject("composition_incomplete", self)
        await self._run(self._record, {"kind": "composition", "instrumented": sorted(self._format.selected_producers),
            "global_inventory_complete": False, "aggregate_copy_complete": False})

    # ------------------------------------------------------------------
    # V3 account publisher: the original UserManager atomic JSON writer,
    # observed through the original DashboardAPI.admin_work owner.
    # ------------------------------------------------------------------

    async def attach_accounts(self, manager: object, dashboard: object) -> None:
        """Bind the exact dashboard publisher before serving; never take its close."""
        from src.users.user_manager import UserManager
        from src.dashboard.api import DashboardAPI
        work = getattr(dashboard, "admin_work", None)
        if (self._format.schema_version not in {3, 4} or self._selected.get("accounts") is not True
                or self._accounts is not None or self._accounts_disabled
                or type(manager) is not UserManager or type(dashboard) is not DashboardAPI
                or dashboard.user_manager is not manager or manager.voice_review_managed is not True
                or work is None or getattr(dashboard.app.state, "dashboard_admin_work", None) is not work):
            _reject("account_owner", self)
        if str(Path(manager.legacy_path).absolute()) != str(Path(self._expected["accounts"]).absolute()):
            _reject("configured_path", self)
        publisher = _AccountPublisher(manager, dashboard, work)
        # Retain the exact publisher and install the observer before any await.
        self._accounts = publisher
        manager._output_coordinator = self
        work._install_native_preflight(self, self._preflight)
        from src.memory.output_registration_v3 import account_plan
        await self._run(self.admission.revalidate)
        def planned() -> dict:
            publisher.pin.open_at(manager.data_path)  # Before the plan observes its identity.
            plan = account_plan(manager.data_path, manager.legacy_path, manager.legacy_path,
                                self.admission.state_root)
            try:
                publisher.pin.require(plan["initial_identity"])
            except (OSError, RuntimeError, ValueError):
                _reject("destination_changed", self)
            return plan
        plan = await self._run(planned)
        await self._run(self._append_account, "intent", plan)
        await self._run(self._append_account, "attached", plan)
        publisher.plan, publisher.expected_identity = plan, plan["initial_identity"]
        publisher.phase = "attached"

    async def accounts_disabled(self, manager: object) -> None:
        """Record an existing but unselected publisher; no worker or absence claim."""
        from src.users.user_manager import UserManager
        if (self._format.schema_version not in {3, 4} or self._selected.get("accounts") is not False
                or self._accounts is not None or self._accounts_disabled or type(manager) is not UserManager):
            _reject("account_owner", self)
        if str(Path(manager.legacy_path).absolute()) != str(Path(self._expected["accounts"]).absolute()):
            _reject("configured_path", self)

        def publish() -> None:
            from src.memory.output_registration_v3 import account_event_shape
            with self._lock:
                _, _, states, _ = self._format.decode_chain(self._directory)
                event = {"kind": "account_publisher", "phase": "disabled",
                         "configured_path": str(Path(manager.legacy_path).absolute()),
                         "path": str(Path(manager.data_path).absolute().resolve(strict=False))}
                account_event_shape(event, states.get("accounts"), False, self.admission.state_root)
                self._record(event)
        await self._run(publish)
        self._accounts_disabled = True

    def _append_account(self, phase: str, plan: dict, aggregate: dict | None = None) -> None:
        from src.memory.output_registration_v3 import account_event_shape
        with self._lock:
            _, _, states, _ = self._format.decode_chain(self._directory)
            event = {"kind": "account_publisher", "phase": phase, "plan": plan}
            if aggregate is not None:
                event["aggregate"] = aggregate
            account_event_shape(event, states.get("accounts"), self._selected.get("accounts"),
                                self.admission.state_root)
            self._record(event)

    def _account_physical(self, publisher: _AccountPublisher, manager: object) -> None:
        """Cooperative gate: exact canonical leaf and its current own generation."""
        plan = publisher.plan
        try:
            supplied = Path(manager.data_path)
            base._require(plan is not None and not supplied.is_symlink()
                and str(supplied.absolute().resolve(strict=False)) == plan["path"], "destination_changed")
            base._require(base._directory(Path(plan["ancestor"])) == plan["parent_identity"], "destination_changed")
            try:
                info = supplied.lstat()
            except FileNotFoundError:
                base._require(publisher.expected_identity is None, "destination_changed")
            else:
                base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and {"device": info.st_dev, "inode": info.st_ino} == publisher.expected_identity
                    and publisher.pin.holds(info), "destination_changed")
        except (OSError, RuntimeError, ValueError):
            _reject("destination_changed", self)

    def _account_owners(self, execution: object) -> tuple:
        """Recognize only an installed exact account invocation or delete step."""
        publisher = self._accounts
        if publisher is None or execution.work is not publisher.work:
            return ()
        from src.dashboard.native_work import _invoke_admin
        from src.users.user_manager import UserManager
        with self._dispatch_lock:
            invocation = publisher.invocations.get(execution)
            if invocation is not None:
                if execution.function is not _invoke_admin or invocation.admitted:
                    return ()
                invocation.admitted = True
                return (publisher,)
            parent = execution.parent_sequence
            step = None if parent is None else publisher.steps.get(parent)
            function = execution.function
            if (step is None or step.execution is not None
                    or getattr(function, "__self__", None) is not publisher.manager
                    or getattr(function, "__func__", None) is not UserManager.remove_user
                    or not _delete_sequence(publisher, parent)):
                return ()
            step.execution = execution
            return (publisher,)

    async def account_publisher_drained(self, dashboard: object) -> None:
        """Called by the original DashboardAPI.stop task after its stack exited."""
        publisher = self._accounts
        if publisher is None or publisher.dashboard is not dashboard:
            self._failed = True
            return
        if asyncio.current_task() is not dashboard._stop_task:
            _reject("close_owner", self)
        status = publisher.work.status()
        if not status["closed"] or status["operations"] or publisher.active is not None:
            _reject("native_close_pending", self)
        if self._failed or publisher.uncertain or publisher.phase not in {"attached", "opening"}:
            # A poisoned epoch keeps its original drain but never a clean terminal.
            self._failed = True
            return

        def publish() -> None:
            self._account_physical(publisher, publisher.manager)
            aggregate = {"attempts": publisher.attempts, "committed_replacements": publisher.committed,
                         "clean_failed_attempts": publisher.clean_failed,
                         "last_committed_identity": publisher.expected_identity,
                         "native_drain_returned": True, "open_temp_count": 0, "cleanup_uncertain": False}
            self._append_account("drained", publisher.plan, aggregate)
        await self._run(publish)
        publisher.phase = "drained"

    # ------------------------------------------------------------------
    # V4 configuration publisher: the original ConfigurationChanges raw-YAML
    # writer of <state_root>/config/settings.yaml, observed at its single
    # shutdown write window. The coordinator never calls _commit, cancels
    # the writer or deletes a file.
    # ------------------------------------------------------------------

    async def attach_configuration(self, owner: object, authority: object) -> None:
        """Bind the exact installed owner before any write; record intent and attached."""
        from src.core.state_configuration import ConfigurationChanges
        from src.dashboard.authority import DashboardAuthority
        sessions = self._roles.get("authority_sessions")
        if (self._format.schema_version != 4 or self._selected.get("configuration") is not True
                or self._configuration is not None or self._configuration_disabled
                or type(owner) is not ConfigurationChanges or type(authority) is not DashboardAuthority
                or owner._authority is not authority or sessions is None or sessions.instance is not authority
                or owner._admission is not self.admission or not owner._installed or owner._closed
                or owner._graph is not None or owner._pending is not None):
            _reject("configuration_owner", self)
        publisher = _ConfigurationPublisher(owner, authority)
        # Retain the exact owner before the first fallible await.
        self._configuration = publisher
        owner._output_coordinator = self
        from src.memory.output_registration_v4 import configuration_path, configuration_plan

        def planned() -> dict:
            # Pinned before the plan observes it: the recorded identity stays this inode.
            publisher.pin.open_at(configuration_path(Path(self.admission.state_root)))
            plan = configuration_plan(self.admission.state_root)
            try:
                publisher.pin.require(plan["initial_identity"])
            except (OSError, RuntimeError, ValueError):
                _reject("destination_changed", self)
            return plan
        plan = await self._run(planned)
        if plan["path"] != self._expected["configuration"]:
            _reject("configured_path", self)
        try:
            projected = self.admission._binding["authority"]["projection"]["configuration"]["config/settings.yaml"]
        except (KeyError, TypeError):
            projected = None
        if plan["initial_sha256"] != projected:
            # Admission verified this projection; a difference is an unregistered writer.
            _reject("configuration_binding", self)
        await self._run(self._append_configuration, "intent", plan)
        await self._run(self._append_configuration, "attached", plan)
        publisher.plan = plan
        publisher.phase = "attached"

    async def configuration_disabled(self) -> None:
        """Record the unselected publisher; the existing file stays an obligation."""
        if (self._format.schema_version != 4 or self._selected.get("configuration") is not False
                or self._configuration is not None or self._configuration_disabled):
            _reject("configuration_owner", self)

        def publish() -> None:
            from src.memory.output_registration_v4 import configuration_event_shape_v2
            with self._lock:
                _, _, states, _ = self._format.decode_chain(self._directory)
                event = {"kind": "configuration_publisher", "phase": "disabled",
                         "path": self._expected["configuration"], "decision": self._effective_decision}
                configuration_event_shape_v2(event, states.get("configuration"), False,
                                             self.admission.state_root, self._epoch["state_identity"])
                self._record(event)
        await self._run(publish)
        self._configuration_disabled = True

    def _append_configuration(self, phase: str, plan: dict, aggregate: dict | None = None) -> None:
        from src.memory.output_registration_v4 import actor_withdrawal, configuration_event_shape_v2
        with self._lock:
            _, _, states, _ = self._format.decode_chain(self._directory)
            prior = states.get("configuration")
            event = {"kind": "configuration_publisher", "phase": phase, "plan": plan,
                     "decision": self._effective_decision}
            if phase == "withdrawn":
                event["withdrawal"] = actor_withdrawal(plan)
            elif phase in {"opening", "drained"}:
                event["withdrawal"] = None if prior is None else prior["withdrawal"]
            if aggregate is not None:
                event["aggregate"] = aggregate
            configuration_event_shape_v2(event, prior, self._selected.get("configuration"),
                                         self.admission.state_root, self._epoch["state_identity"])
            self._record(event)

    async def configuration_withdrawn(self, owner: object) -> None:
        """Durably record this run's actor withdrawal (record and HEAD fsynced) before returning."""
        try:
            self._live()
            publisher = self._configuration
            if publisher is None or publisher.owner is not owner or publisher.phase != "attached":
                _reject("configuration_owner", self)
            await self._run(self._append_configuration, "withdrawn", publisher.plan)
            publisher.phase = "withdrawn"
        except BaseException:
            self._failed = True
            raise

    async def configuration_opening(self, owner: object) -> _ConfigurationCommit:
        """Record opening before the writer Task and before any file side effect."""
        try:
            self._live()
            publisher = self._configuration
            # OQ-8 (M5): a shutdown write opens only after a durable actor withdrawal.
            if (publisher is None or publisher.owner is not owner or publisher.phase != "withdrawn"
                    or publisher.extent is not None or owner._pending is None):
                _reject("configuration_owner", self)

            def publish() -> None:
                from src.memory.output_registration_v4 import configuration_plan
                with self._lock:
                    # The physical file must still be the attached one.
                    if (configuration_plan(self.admission.state_root) != publisher.plan
                            or not publisher.pin.holds_path(publisher.plan["path"])):
                        _reject("destination_changed", self)
                    self._append_configuration("opening", publisher.plan)
            await self._run(publish)
            publisher.phase = "opening"
            publisher.extent = _ConfigurationCommit(self, publisher)
            return publisher.extent
        except BaseException:
            self._failed = True
            raise

    async def configuration_drained(self, owner: object, extent: _ConfigurationCommit | None,
                                    outcome: str | None, *, graph_failed: bool = False) -> None:
        """Called once by the owner's exit after its writer returned (or never started)."""
        if self._failed:
            return  # A poisoned epoch keeps its original exit but never a clean terminal.
        publisher = self._configuration
        if publisher is None or publisher.owner is not owner:
            self._failed = True
            if graph_failed:
                return  # Keep the original construction error visible.
            _reject("required", self)
        if publisher.uncertain or publisher.phase not in {"attached", "withdrawn", "opening"}:
            self._failed = True
            return
        plan = publisher.plan
        from src.memory.output_registration_v4 import CONFIGURATION_OUTCOMES_V2 as CONFIGURATION_OUTCOMES
        if extent is None:
            pending = owner._pending is not None
            if (publisher.phase not in {"attached", "withdrawn"} or publisher.extent is not None
                    or outcome not in {"no_pending", "not_attempted_graph_failed", "refused_before_temporary"}
                    or (outcome == "no_pending") == pending):
                _reject("configuration_outcome", self)
            attempts, identity, digest, temporary = 0, plan["initial_identity"], plan["initial_sha256"], None
        else:
            if (publisher.phase != "opening" or extent is not publisher.extent or extent.coordinator is not self
                    or extent.state not in {"issued", "finished"}):
                _reject("configuration_outcome", self)
            if extent.state == "issued":
                outcome = "refused_before_temporary"  # The owner refused before admitting the extent.
            elif extent.confirmed:
                outcome = "committed"
            else:
                outcome = "clean_failed"
            attempts = extent.attempts
            identity = extent.committed_identity if extent.confirmed else plan["initial_identity"]
            digest = extent.committed_sha256 if extent.confirmed else plan["initial_sha256"]
            temporary = extent.temporary
        if publisher.phase not in CONFIGURATION_OUTCOMES[outcome]:
            _reject("configuration_outcome", self)

        def publish() -> None:
            from src.memory.output_registration_v4 import configuration_observation
            observed = configuration_observation(plan)
            if (observed != (identity, digest) or not publisher.pin.holds_path(plan["path"])
                    or base._directory(Path(plan["parent"])) != plan["parent_identity"]
                    or (temporary is not None and os.path.lexists(temporary))):
                _reject("destination_changed", self)
            aggregate = {"outcome": outcome, "attempts": attempts, "last_committed_identity": identity,
                         "last_committed_sha256": digest, "writer_returned": True, "open_temp_count": 0,
                         "cleanup_uncertain": False}
            self._append_configuration("drained", plan, aggregate)
        await self._run(publish)
        publisher.phase = "drained"

    async def aclose(self) -> None:
        if self._closing is None:
            async def finish():
                tasks = {getattr(h.store, "_closing", None) for h, _ in self._handles.values()}
                publisher = self._accounts
                if publisher is not None:
                    # Observe the exact original DashboardAPI.stop owner; never call it.
                    tasks.add(publisher.dashboard._stop_task)
                try:
                    await asyncio.gather(*(asyncio.shield(t) for t in tasks if t is not None))
                    if self._open_extents or self._opening_extents:
                        _reject("native_close_pending", self)
                    if publisher is not None and (publisher.dashboard._stop_task is None or publisher.active is not None):
                        _reject("native_close_pending", self)
                    if not self._failed:
                        _, _, states, _ = await self._run(self._format.decode_chain, self._directory)
                        for key in self._handles:
                            base._require(states.get(key, {}).get("phase") in {"closed", "native_failed_then_actually_closed"}, "epoch_incomplete")
                        if publisher is not None:
                            base._require(states.get("accounts", {}).get("phase") == "drained", "epoch_incomplete")
                        if self._accounts_disabled:
                            base._require(states.get("accounts", {}).get("phase") == "disabled", "epoch_incomplete")
                        if self._configuration is not None:
                            base._require(states.get("configuration", {}).get("phase") == "drained", "epoch_incomplete")
                        if self._configuration_disabled:
                            base._require(states.get("configuration", {}).get("phase") == "disabled", "epoch_incomplete")
                finally:
                    # Every destination pin lives until the epoch closes, poisoned or not.
                    for pinned in (publisher, self._configuration):
                        if pinned is not None:
                            pinned.pin.release()
                    await self._disk.aclose()
                base._require(not self._failed, "epoch_incomplete")
            self._closing = asyncio.create_task(finish(), context=Context())
            self._closing.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(self._closing)
        except TimeoutError as error:
            from src.core.owned_work import WorkCloseTimeout
            raise WorkCloseTimeout("output_registration_close_pending", phase=base.ClosePhase.DRAIN, diagnostics=self._disk.diagnostics()) from error


async def attach_output(owner: object, producer: str, *, role: str = "primary") -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        await coordinator.attach(owner, producer, role=role)
    elif _managed():
        _reject("coordinator")


def before_native(owner: object, path: str | Path) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        coordinator.before_native(owner, path)
    else:
        _unmanaged_path(path)


def borrowed_ledger(ledger: object) -> None:
    coordinator = _coordinator(ledger)
    if coordinator is None:
        if _managed():
            _reject("required")
        return
    owner = getattr(ledger, "_output_privacy_owner", None)
    binding = coordinator._owner(owner)
    if owner.ledger is not ledger or binding.phase not in {"opening", "active"} or ledger._db is None:
        _reject("ledger_lifetime", coordinator)


def native_ready(owner: object) -> bool:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        coordinator.ready_native(owner)
        return True
    return False


async def async_ready(owner: object) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        await coordinator.ready_async(owner)


async def owner_closed(owner: object, *, connection: object | None = None) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        await coordinator.closed_owner(owner, connection=connection)


def before_ledger_open(ledger: object) -> None:
    owner = getattr(ledger, "_output_privacy_owner", None)
    if owner is None:
        _unmanaged_path(ledger.path)
        return
    before_native(owner, ledger.path)


async def before_routine_open(owner: object) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        await coordinator.routine_opening(owner)
    else:
        await asyncio.to_thread(_unmanaged_path, owner._db_path)


def final_worker(owner: object, work: object, args: tuple) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        coordinator.bind_final_worker(owner, work, args)


async def reserve_suggestions(events: object, path: str) -> ConstructorReservation | None:
    coordinator = _coordinator(events)
    return None if coordinator is None else await coordinator.reserve_constructor(events, path)


def enter_suggestions(store: object, reservation: ConstructorReservation | None) -> None:
    coordinator = _coordinator()
    if coordinator is not None:
        coordinator.constructor_enter(store, reservation)
    elif _managed():
        _reject("required")


async def run_registered(owner: object, work: object, function: Callable, *args: Any, **kwargs: Any) -> Any:
    coordinator = _coordinator(owner)
    if coordinator is None:
        if _managed():
            _reject("required")
        return await work.run(function, *args, **kwargs)
    binding = coordinator._owner(owner)
    if binding.phase != "active":
        raise OutputNotReady("output_registration_waiting")
    for attempt in range(2):
        await coordinator.refresh(owner)
        try:
            return await work.run(function, *args, **kwargs)
        except RegistryChanged:
            if attempt:
                _reject("registry_unstable", coordinator)


def target_rows(ledger: object, participant: ErasureParticipant) -> list[dict]:
    if not _managed():
        return ledger.erasures.targets(participant)
    coordinator = _coordinator(ledger)
    if coordinator is None:
        _reject("ledger_owner")
    snapshot = ledger.erasures.target_snapshot()
    if snapshot.status != "complete":
        _reject("registry_unavailable", coordinator)
    return [{"id": e.id, "version": e.version, "descriptor": e.descriptor}
            for e in snapshot.entries if e.participant is participant and e.descriptor is not None]


@contextmanager
def native_connection(owner: object | None, path: str | Path, connect: Callable, *args: Any, kind: str | None = None, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Keep the original per-operation close policy and observe its real return."""
    coordinator = _coordinator(owner)
    binding = execution = None
    if coordinator is not None:
        if owner is None:
            binding, execution = coordinator.history_entry(path, kind)
        else:
            binding = coordinator.native_owner(owner)
            execution = coordinator._native(binding, path)
        with coordinator._lock:
            if len(coordinator._open_extents) + coordinator._opening_extents >= _MAX_OPEN_EXTENTS:
                _reject("native_capacity", coordinator)
            coordinator._opening_extents += 1
    else:
        _unmanaged_path(path)
    try:
        connection = connect(*args, **kwargs)
    except BaseException:
        if coordinator is not None:
            with coordinator._lock:
                coordinator._opening_extents -= 1
        raise
    if coordinator is not None:
        with coordinator._lock:
            coordinator._opening_extents -= 1
            coordinator._open_extents[id(connection)] = (binding, execution, connection)
    try:
        if coordinator is not None and owner is None:
            coordinator.observe(binding, path, "file_existing_uninspected")
        yield connection
    except (ValueError, sqlite3.DatabaseError):
        if coordinator is not None and owner is None:
            coordinator.observe(binding, path, "representation_corrupt_or_unsupported")
        raise
    finally:
        try:
            connection.close()
        except BaseException:
            if coordinator is not None:
                coordinator._failed = True
            raise
        else:
            if coordinator is not None:
                coordinator._open_extents.pop(id(connection), None)


def observe_history(path: str | Path, kind: str, state: str) -> None:
    coordinator = _coordinator()
    if coordinator is not None:
        binding = coordinator.observation_entry(path, kind)
        coordinator.observe(binding, path, state)
    elif _managed():
        _reject("required")


def _read_scope_record(path: Path) -> tuple[dict, bytes]:
    """Read new scope metadata once through stable nofollow directory FDs."""
    path = _canonical_path(str(path))
    limit = 16 * 1024 * 1024  # Preserve the original stage metadata limit.
    descriptors, parents = [], []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    def directory_identity(info: os.stat_result) -> tuple:
        base._require(stat.S_ISDIR(info.st_mode), "delegated_record")
        return info.st_dev, info.st_ino, info.st_mode
    def file_identity(info: os.stat_result) -> tuple:
        base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "delegated_record")
        return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)
    try:
        current = os.open(path.anchor, directory_flags)
        descriptors.append(current)
        root_identity = directory_identity(os.fstat(current))
        for name in path.parts[1:-1]:
            child = os.open(name, directory_flags, dir_fd=current)
            descriptors.append(child)
            identity = directory_identity(os.fstat(child))
            base._require(directory_identity(os.stat(name, dir_fd=current, follow_symlinks=False)) == identity, "delegated_record")
            parents.append((current, name, child, identity))
            current = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        descriptors.append(descriptor)
        before = file_identity(os.fstat(descriptor))
        base._require(before[4] <= limit, "delegated_record")
        chunks, size = [], 0
        while size <= limit:
            block = os.read(descriptor, min(65536, limit + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
        raw = b"".join(chunks)
        base._require(len(raw) <= limit and len(raw) == before[4]
            and file_identity(os.fstat(descriptor)) == before
            and file_identity(os.stat(path.name, dir_fd=current, follow_symlinks=False)) == before,
            "delegated_record")
        for parent, name, child, identity in parents:
            base._require(directory_identity(os.fstat(child)) == identity
                and directory_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == identity, "delegated_record")
        base._require(directory_identity(os.fstat(descriptors[0])) == root_identity
            and directory_identity(os.stat(path.anchor, follow_symlinks=False)) == root_identity, "delegated_record")
        value = json.loads(raw)
        base._require(type(value) is dict, "delegated_record")
        return value, raw
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_delegated_observation(stage: object, root: Path, binding: dict, event: dict, policy_json: bytes) -> bytes:
    """Join a closed observation to this controller's actual immutable records."""
    state, scope = Path(binding["state_root"]), event["scope"]
    source_root = _scope_shape(scope, state)
    policy = json.loads(policy_json)
    base._require(set(policy) == {"schema_version", "source_bindings", "owned_targets",
        "delegated_targets", "historical_targets"}
        and type(policy["schema_version"]) is int and policy["schema_version"] == 1
        and all(type(policy[key]) is list for key in ("source_bindings", "owned_targets", "delegated_targets", "historical_targets"))
        and hashlib.sha256(policy_json).hexdigest() == binding["ownership_sha256"]
        and event["id"] in policy["delegated_targets"] and event["id"] not in policy["owned_targets"]
        and scope["source_binding"] in policy["source_bindings"], "delegated_scope")
    record, _ = _read_scope_record(root / "admission/delegations" / (scope["delegation_sha256"] + ".json"))
    base._require(set(record) == {"schema_version", "destination_state", "destination_identity",
        "source_binding", "source_close_sha256", "targets"} and record["schema_version"] == 1
        and stage._digest(record) == scope["delegation_sha256"]
        and record["destination_state"] == str(state)
        and record["destination_identity"] == binding["state_identity"] == stage._identity(stage._safe_directory(state))
        and record["source_binding"] == scope["source_binding"]
        and record["source_close_sha256"] == scope["source_close_sha256"], "delegated_scope")
    target = record["targets"][event["id"]]
    stage._validate_target_record(target)
    plan = event["plan"]
    base._require(set(target) == {"id", "participant", "version", "descriptor"}
        and target["id"] == event["id"] and target["version"] == 1
        and target["participant"] == plan["producer"] and plan["role"] == "historical"
        and target["descriptor"]["path"] == plan["path"] == plan["configured_path"], "delegated_scope")
    if event["status"] == "final_registry_observation":
        base._require(event["participant"] == target["participant"] and event["version"] == target["version"], "delegated_scope")
    source, _ = _read_scope_record(root / "admission/bindings" / (scope["source_binding"] + ".json"))
    stage._validate_binding_record(scope["source_binding"], source)
    base._require(source["state_root"] == str(source_root)
        and source["state_identity"] == scope["source_state_identity"] == stage._identity(stage._safe_directory(source_root)), "delegated_scope")
    marker, _ = _read_scope_record(source_root / stage._MARKER)
    base._require(marker.get("binding_id") == scope["source_binding"], "delegated_scope")
    proof, _ = _read_scope_record(root / "admission/closures" / (scope["source_binding"] + ".json"))
    running = stage._validate_closure_header(scope["source_binding"], proof)
    base._require(type(running["run_id"]) is str and re.fullmatch("[0-9a-f]{32}", running["run_id"]), "delegated_scope")
    receipt, receipt_raw = _read_scope_record(root / "admission/receipts" / (running["run_id"] + ".json"))
    stage._validate_closure_receipt(proof, receipt)
    stage._validate_closure_digest(proof, hashlib.sha256(receipt_raw).hexdigest())
    base._require(stage._digest(proof) == scope["source_close_sha256"], "delegated_scope")
    _plan_shape(plan, source_root)
    return stage._canonical_json({"policy": policy, "record": record, "source": source,
                                  "marker": marker, "proof": proof})


def read_inventory(controller: object) -> tuple[dict, ...]:
    from src.memory.copy_catalog import _controller_lease
    stage, root, head, _, _ = _controller_lease(controller)
    namespace = root / "admission/output-inventory"
    if not namespace.exists():
        return ()
    identity = base._directory(namespace)
    _namespace(namespace)
    versions = sorted(namespace.iterdir())
    results, seen = [], {}
    from src.memory.output_registration_v3 import chain as chain_v3
    from src.memory.output_registration_v4 import UnsupportedProducerGrammar, chain as chain_v4
    decoders = {"v1": base._chain_v1, "v2": chain, "v3": chain_v3, "v4": chain_v4}
    for version in versions:
        if version.name not in decoders or not version.is_dir() or not (version / "runs").is_dir():
            results.append({"version": version.name, "phase": "unknown", "reason": "unsupported_version",
                "capability": False, "global_inventory_complete": False, "aggregate_copy_complete": False})
            continue
        runs = version / "runs"
        runs_identity, run_entries = base._directory(runs), sorted(runs.iterdir())
        for directory in run_entries:
            value = {"run_id": directory.name, "version": version.name, "phase": "unknown", "capability": False,
                     "global_inventory_complete": False, "aggregate_copy_complete": False,
                     "required_producers": list(base.PRODUCERS), "uncovered_producers": list(base.PRODUCERS)}
            recovered = {}
            try:
                decode = decoders[version.name]
                header, journal, states, _ = decode(directory)
                binding = stage.read_binding(root, header["binding_id"])
                receipt = stage._read_json(root / "admission/receipts" / (header["run_id"] + ".json"))
                base._require(binding["state_root"] == header["state_root"] and binding["state_identity"] == header["state_identity"]
                    and receipt["run_id"] == header["run_id"] and receipt["binding_id"] == header["binding_id"]
                    and receipt["generation"] == header["generation"] and receipt["status"] == "closed"
                    and set(receipt["resources"]) == stage._CLOSED_RESOURCES
                    and all(v is True for v in receipt["resources"].values()), "run_receipt")
                raw_config = binding["authority"]["projection"]["configuration"].get("config/settings.yaml")
                base._require(type(raw_config) is str and re.fullmatch("[0-9a-f]{64}", raw_config), "configuration_binding")
                configuration = states.get("configuration") if version.name == "v4" else None
                if configuration is not None and configuration["plan"] is not None:
                    # The attach-time file must be the one the admission binding projected.
                    base._require(configuration["plan"]["initial_sha256"] == raw_config, "configuration_binding")
                recovered.update(header=header, journal=journal, stores=states, phase="observed",
                    configuration={"raw_sha256": binding["authority"]["projection"]["configuration"].get("config/settings.yaml"),
                                   "effective_digest": "observed_only"})
                if version.name == "v1":
                    recovered["uncovered_producers"] = [p for p in base.PRODUCERS if p not in states or states[p]["phase"] != "closed"]
                else:
                    recovered["instrumented_producers"] = sorted({s.get("producer") or s["plan"]["producer"] for s in states.values()})
                    recovered["uncovered_producers"] = [p for p in base.PRODUCERS if p not in {"lists", "reminders"} or states.get(p, {}).get("phase") != "closed"]
                    selected = _SELECTED | ({"accounts"} if version.name == "v3" else set()) | (
                        set(header["grammars"]) if version.name == "v4" else set())
                    recovered["partial_producers"] = [p for p in sorted(selected - {"lists", "reminders"})]
                    material = []
                    for path in sorted(directory.iterdir()):
                        if re.fullmatch(r"\d{12}-[0-9a-f]{64}\.json", path.name):
                            event, _ = base._read(path)
                            if event["kind"] in {"target", "delegated_target", "delegation_policy", "physical", "registry", "composition"}:
                                material.append(event)
                    policies = [event for event in material if event["kind"] == "delegation_policy"]
                    policy_json = _policy_shape(policies[0]) if policies else None
                    if policy_json is not None:
                        base._require(base._sha(policy_json) == binding["ownership_sha256"], "delegated_policy")
                    delegated = [(event, _read_delegated_observation(stage, root, binding, event, policy_json))
                        for event in material if event["kind"] == "delegated_target"]
                    recovered["material_observations"] = tuple(material)
                    base._require(decode(directory)[:3] == (header, journal, states), "journal_changed")
                    for event, observation in delegated:
                        base._require(_read_delegated_observation(stage, root, binding, event, policy_json) == observation, "authority_changed")
            except UnsupportedProducerGrammar:
                # Typed and distinct: a newer grammar is never decoded as another one.
                value["reason"] = "unsupported_producer_grammar"
            except (ValueError, RuntimeError, OSError, KeyError, TypeError):
                value["reason"] = "invalid_or_unfinished_history"
            else:
                value.update(recovered)
            if directory.name in seen:
                for conflicting in (value, seen[directory.name]):
                    run_id, journal_version = conflicting["run_id"], conflicting["version"]
                    conflicting.clear()
                    conflicting.update(run_id=run_id, version=journal_version, phase="unknown",
                        reason="conflicting_run_versions", capability=False, global_inventory_complete=False,
                        aggregate_copy_complete=False, required_producers=list(base.PRODUCERS),
                        uncovered_producers=list(base.PRODUCERS))
            seen[directory.name] = value
            results.append(value)
        base._require(base._directory(runs) == runs_identity and sorted(runs.iterdir()) == run_entries, "journal_changed")
    base._require(base._directory(namespace) == identity and sorted(namespace.iterdir()) == versions
                  and _controller_lease(controller)[2] == head, "authority_changed")
    return tuple(results)


@contextmanager
def event_call(owner: object, function: Callable) -> Iterator[None]:
    """Bind the exact callable admitted through the original EventLogger API."""
    token = _CALL.set((owner, function))
    try:
        yield
    finally:
        _CALL.reset(token)


def borrowed_events(owner: object) -> None:
    coordinator = _coordinator(owner)
    if coordinator is None:
        if _managed():
            _reject("required")
        return
    binding = coordinator._owner(owner)
    if binding.phase != "active":
        _reject("initialization_incomplete", coordinator)


def ledger_connected(ledger: object, connection: object) -> None:
    coordinator = _coordinator(ledger)
    if coordinator is not None:
        owner = ledger._output_privacy_owner
        binding = coordinator._owner(owner)
        execution = coordinator.before_native(owner, ledger.path)
        coordinator._open_extents[id(connection)] = (binding, execution, connection)


def ledger_closed(ledger: object, connection: object) -> None:
    coordinator = _coordinator(ledger)
    if coordinator is not None:
        previous = coordinator._open_extents.get(id(connection))
        if previous is None or previous[2] is not connection:
            _reject("ledger_close_owner", coordinator)
        owner = ledger._output_privacy_owner
        execution = current_native_execution()
        allowed = validate_native_execution(execution, work=owner._close_work) or validate_native_execution(execution, work=owner._work)
        if not allowed or getattr(execution.function, "__self__", None) is not owner or getattr(execution.function, "__name__", None) not in {"initialize", "_close_resources"}:
            _reject("ledger_close_owner", coordinator)
        try:
            connection.in_transaction
        except sqlite3.ProgrammingError:
            pass
        else:
            _reject("ledger_close_pending", coordinator)
        coordinator._open_extents.pop(id(connection))


def ledger_close_failed(ledger: object, connection: object) -> None:
    coordinator = _coordinator(ledger)
    if coordinator is not None:
        coordinator._failed = True


def history_before(path: str | Path, kind: str) -> None:
    coordinator = _coordinator()
    if coordinator is not None:
        coordinator.history_entry(path, kind)
    else:
        _unmanaged_path(path)


def target_groups(ledger: object) -> tuple[list[dict], list[dict]]:
    from src.memory.erasure_journal import ErasureParticipant
    if not _managed():
        return ledger.erasures.targets(ErasureParticipant.SUGGESTIONS), ledger.erasures.targets(ErasureParticipant.EVENTS)
    snapshot = ledger.erasures.target_snapshot()
    coordinator = _coordinator(ledger)
    if coordinator is None or snapshot.status != "complete":
        _reject("registry_unavailable", coordinator)
    def rows(participant):
        return [{"id": e.id, "version": e.version, "descriptor": e.descriptor}
                for e in snapshot.entries if e.participant is participant and e.descriptor is not None]
    return rows(ErasureParticipant.SUGGESTIONS), rows(ErasureParticipant.EVENTS)


async def construct_suggestions(events: object, function: Callable, **kwargs: Any) -> object:
    coordinator = _coordinator(events)
    if coordinator is None:
        if _managed():
            _reject("required")
        return await events.acall(function, **kwargs)
    reservation = await coordinator.reserve_constructor(events, kwargs["suggestions_path"])
    async def sequence():
        with event_call(events, function):
            engine = await events._work.native_step(function, output_reservation=reservation, **kwargs)
        store = coordinator._constructor_returns.get(id(reservation))
        binding = coordinator._owner(store)
        if engine.store is not store or binding.phase != "opening" or asyncio.current_task() not in events._work._tasks:
            _reject("constructor_return", coordinator)
        await coordinator._run(coordinator._append, "suggestions", "active", coordinator._constructor_plan)
        binding.phase = "active"
        binding.native_identity = base._identity(Path(coordinator._constructor_plan["path"]))
        del coordinator._reservations[id(reservation)]
        del coordinator._constructor_returns[id(reservation)]
        return engine
    return await events._work.run_async(sequence)


async def final_registry(privacy: object) -> None:
    coordinator = _coordinator(privacy)
    if coordinator is None:
        return
    if privacy._closing is not asyncio.current_task():
        _reject("close_owner", coordinator)
    snapshot = await privacy._close_work.run(privacy.ledger.erasures.target_snapshot)
    if snapshot.ledger is not privacy.ledger or snapshot.status != "complete":
        _reject("registry_unavailable", coordinator)
    for entry in snapshot.entries:
        record = {"kind": "target", "owner": "privacy_authority",
            "id": entry.id, "participant": entry.participant.value, "version": entry.version,
            "status": "final_registry_observation", "representation_status": entry.status,
            "dispatch_coverage": "observed_only", "opaque_digest": entry.opaque_digest}
        descriptor = entry.descriptor
        if entry.status == "registered" and isinstance(descriptor, dict) and "path" in descriptor:
            try:
                from src.analytics.event_storage import validate_event_descriptor
                from src.analytics.suggestion_storage import validate_suggestion_descriptor
                validate = validate_event_descriptor if entry.participant.value == "events" else validate_suggestion_descriptor
                validate(descriptor)
                record["plan"], scope_json, policy_json = await coordinator._disk.run(_historical_plan, coordinator.admission, entry)
                if scope_json is not None:
                    await coordinator._run(coordinator._record_historical_policy, policy_json)
                    record.update(kind="delegated_target", scope=json.loads(scope_json))
                record["physical_status"] = "not_dispatched_by_final_observation"
            except (ValueError, RuntimeError, OSError):
                record["physical_status"] = "unbound_or_unsupported"
        else:
            record["physical_status"] = "nonphysical_or_unsupported"
        await coordinator._run(coordinator._record, record)
    await coordinator._run(coordinator._record, {"kind": "registry", "owner": "privacy_authority",
        "digest": snapshot.digest, "status": "final_observation", "target_count": len(snapshot.entries)})


async def composed(coordinator: OutputRegistrationV2 | None, graph: object | None = None) -> None:
    if coordinator is not None:
        await coordinator.composed(graph)


async def initialize_owned(owner: object, work: object, function: Callable, *args: Any, **kwargs: Any) -> Any:
    """Keep native return and active publication in the original admitted owner."""
    coordinator = _coordinator(owner)
    if coordinator is None:
        return await work.run(function, *args, **kwargs)
    binding = coordinator._owner(owner)
    if binding.phase == "active":
        return await work.run(function, *args, **kwargs)
    if binding.phase != "attached" or getattr(function, "__self__", None) is not owner or getattr(function, "__func__", None) is not type(owner).initialize:
        _reject("initialization_owner", coordinator)
    async def sequence() -> Any:
        result = await work.native_step(function, *args, **kwargs)
        await coordinator.initializer_returned(owner)
        return result
    return await work.run_async(sequence)


def dashboard_operation(owner: object, action: str, args: tuple, guard: object) -> None:
    coordinator = _coordinator(owner)
    if coordinator is not None:
        coordinator.dashboard_operation(owner, action, args, guard)


async def run_final_rotation(owner: object, work: object, args: tuple) -> Any:
    """Dispatch the existing final worker only from its original retained close."""
    coordinator = _coordinator(owner)
    if coordinator is None:
        return await work.run(owner._transaction, "rotate", args, None)
    binding = coordinator._close_binding(owner)
    if owner._closing is not asyncio.current_task() or binding.final_work is not work or binding.final_args != args:
        _reject("close_owner", coordinator)
    async def sequence() -> Any:
        binding.final_sequence = asyncio.current_task()
        return await work.native_step(owner._transaction, "rotate", args, None)
    return await work.run_async(sequence)


# ----------------------------------------------------------------------
# Account publisher observations. Private provenance only: route session
# guards and before_save remain the authority; nothing here grants it.
# ----------------------------------------------------------------------

class _DestinationPin:
    """Hold a publisher destination's current inode open across its before/after checks.

    ext4 hands a freed inode number to the next allocation at once, so a
    recorded (st_dev, st_ino) names the destination only while that inode
    stays allocated. While the pin is open it does; a replacement at the name
    then always has another number, and ``holds`` compares the name's lstat
    with the pinned inode itself. Only the atomic JSON/YAML publishers are
    pinned: closing an extra descriptor on a SQLite file would drop the
    process's POSIX locks on it. After a publish the pin is the writer's own
    temporary descriptor (read-write from mkstemp); nothing writes through it.
    """

    __slots__ = ("descriptor",)

    def __init__(self) -> None:
        self.descriptor = None

    def __reduce_ex__(self, protocol: int):
        raise TypeError("destination_pin_not_serializable")

    def open_at(self, path: str | Path) -> None:
        """Pin whatever is at ``path`` now (nothing when absent or unopenable); then ``require``."""
        self.release()
        try:
            self.descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            self.descriptor = None

    def require(self, identity: dict | None) -> None:
        """The pin is exactly the regular file ``identity`` names; None requires nothing pinned."""
        if identity is None:
            base._require(self.descriptor is None, "destination_changed")
            return
        base._require(self.descriptor is not None, "destination_changed")
        held = os.fstat(self.descriptor)
        base._require(stat.S_ISREG(held.st_mode) and (held.st_dev, held.st_ino) == _pair(identity),
                      "destination_changed")

    def adopt(self, descriptor: int) -> None:
        """Take over the writer's own temporary pin as the destination pin after its replace.

        The path is never reopened, so nothing at the name between the replace
        and this point can be pinned instead, and no open can fail after a
        durable publish. The previous generation's pin is closed.
        """
        previous, self.descriptor = self.descriptor, descriptor
        if previous is not None:
            try:
                os.close(previous)
            except OSError:
                pass  # The descriptor is released either way; the new pin is already in place.

    def holds(self, info: os.stat_result) -> bool:
        """``info`` (an lstat of the name) is the pinned inode.

        While the pin is open its inode cannot be freed, so its number cannot
        name another file: (st_dev, st_ino) alone decides. Size, mtime and
        ctime come from the same inode and add nothing on such a filesystem;
        they are a belt-and-braces check for one whose st_ino is not unique
        (and what the tests' simulated number reuse exercises).
        """
        if self.descriptor is None:
            return False
        held = os.fstat(self.descriptor)
        return ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                == (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns, held.st_ctime_ns))

    def holds_path(self, path: str | Path) -> bool:
        try:
            return self.holds(os.lstat(path))
        except OSError:
            return False

    def release(self) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is not None:
            os.close(descriptor)


@dataclass(eq=False)
class _AccountPublisher:
    manager: object
    dashboard: object
    work: object
    phase: str = "waiting"
    plan: dict | None = None
    expected_identity: dict | None = None
    pin: _DestinationPin = field(default_factory=_DestinationPin)
    attempts: int = 0
    committed: int = 0
    clean_failed: int = 0
    uncertain: bool = False
    active: object | None = None
    # Consumed adapter entries: weak keys, no payload, pruned after task end.
    entries: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)
    sequences: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)
    # Ephemeral payload observations; removed in the original adapter finally.
    invocations: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class _AccountInvocation:
    execution: object
    manager: object
    function: object
    current: object
    args: tuple
    kwargs: dict
    admitted: bool = False
    saves: int = 0

    def __reduce_ex__(self, protocol: int):
        raise TypeError("account_invocation_not_serializable")


@dataclass(eq=False, slots=True)
class _AccountSequence:
    function: object
    args: tuple | None


@dataclass(eq=False, slots=True)
class _DeleteStep:
    manager: object
    user_id: object
    validator: object
    execution: object | None = None
    consumed: bool = False
    saves: int = 0

    def __reduce_ex__(self, protocol: int):
        raise TypeError("account_step_not_serializable")


def _prune_done(mapping: WeakKeyDictionary, task_of: Callable) -> None:
    for key in tuple(mapping):
        if task_of(key).done():
            del mapping[key]


def _publisher() -> tuple[OutputRegistrationV2 | None, _AccountPublisher | None]:
    coordinator = _coordinator()
    return coordinator, None if coordinator is None else coordinator._accounts


def _delete_sequence(publisher: _AccountPublisher, task: asyncio.Task) -> bool:
    from src.dashboard.admin import _delete_account
    from src.dashboard.native_work import _invoke_admin_sequence
    sequence = publisher.sequences.get(task)
    return (sequence is not None and sequence.function is _delete_account and sequence.args is not None
            and publisher.work._retained_sequence_function(task) is _invoke_admin_sequence)


class _AdminEntry:
    """One consumed _invoke_admin entry; carries payload only for a real mutator."""

    __slots__ = ("coordinator", "publisher", "invocation")

    def __init__(self, coordinator, publisher, invocation) -> None:
        self.coordinator, self.publisher, self.invocation = coordinator, publisher, invocation

    def admit(self) -> None:
        """Run the sole full preflight after the original guard, outside catalog locks."""
        if self.invocation is not None:
            self.coordinator._preflight(self.invocation.execution)

    def release(self) -> None:
        invocation, publisher, coordinator = self.invocation, self.publisher, self.coordinator
        self.coordinator = self.publisher = self.invocation = None
        if invocation is None:
            return
        with coordinator._dispatch_lock:
            if publisher.invocations.get(invocation.execution) is invocation:
                del publisher.invocations[invocation.execution]
        invocation.execution = invocation.manager = invocation.function = invocation.current = None
        invocation.args, invocation.kwargs = (), {}


def account_admin_entry(current: Callable, function: Callable, args: tuple, kwargs: dict) -> _AdminEntry | None:
    """Consume the one adapter entry of the actual admin_work native invocation."""
    coordinator, publisher = _publisher()
    if publisher is None:
        return None
    execution = current_native_execution()
    if execution is None or execution.work is not publisher.work:
        return None
    from src.dashboard.native_work import _invoke_admin
    from src.users.user_manager import UserManager
    invocation = None
    with coordinator._dispatch_lock:
        if execution.function is not _invoke_admin:
            _reject("account_invocation", coordinator)
        _prune_done(publisher.entries, lambda item: item.task)
        if execution in publisher.entries:
            _reject("account_invocation", coordinator)
        if len(publisher.entries) >= _MAX_NATIVE_WITNESSES:
            _reject("native_capacity", coordinator)
        publisher.entries[execution] = True
        if (getattr(function, "__self__", None) is publisher.manager
                and getattr(function, "__func__", None) in (UserManager.add_user, UserManager.update_user)):
            invocation = _AccountInvocation(execution, publisher.manager, function, current, args, kwargs)
            publisher.invocations[execution] = invocation
    return _AdminEntry(coordinator, publisher, invocation)


class _SequenceEntry:
    __slots__ = ("coordinator", "entry")

    def __init__(self, coordinator, entry) -> None:
        self.coordinator, self.entry = coordinator, entry

    def release(self) -> None:
        coordinator, entry = self.coordinator, self.entry
        self.coordinator = self.entry = None
        with coordinator._dispatch_lock:
            # The consumed marker stays until the real task ends; payload does not.
            entry.function, entry.args = None, None


def account_admin_sequence(function: Callable, args: tuple) -> _SequenceEntry | None:
    """Consume the one entry of the exact retained admin sequence task."""
    coordinator, publisher = _publisher()
    task = asyncio.current_task()
    if publisher is None or task is None or task not in publisher.work._tasks:
        return None
    try:
        reason, entry = _sequence_entry(coordinator, publisher, task, function, args)
    finally:
        # A refusal is stored by this very task: keep no frame -> task cycle.
        task = function = args = publisher = None
    if entry is None:
        _reject(reason, coordinator)
    return entry


def _sequence_entry(coordinator, publisher, task, function, args) -> tuple[str, _SequenceEntry | None]:
    from src.dashboard.admin import _delete_account
    from src.dashboard.native_work import _invoke_admin_sequence
    with coordinator._dispatch_lock:
        if publisher.work._retained_sequence_function(task) is not _invoke_admin_sequence:
            return "account_sequence", None
        _prune_done(publisher.sequences, lambda item: item)
        if task in publisher.sequences:
            return "account_sequence", None
        if len(publisher.sequences) >= _MAX_NATIVE_WITNESSES:
            return "native_capacity", None
        delete = function is _delete_account
        entry = _AccountSequence(function if delete else None, args if delete else None)
        publisher.sequences[task] = entry
    return "", _SequenceEntry(coordinator, entry)


class _StepEntry:
    __slots__ = ("coordinator", "publisher", "task", "step")

    def __init__(self, coordinator, publisher, task, step) -> None:
        self.coordinator, self.publisher, self.task, self.step = coordinator, publisher, task, step

    def release(self) -> None:
        coordinator, publisher, task, step = self.coordinator, self.publisher, self.task, self.step
        self.coordinator = self.publisher = self.task = self.step = None
        with coordinator._dispatch_lock:
            if publisher.steps.get(task) is step:
                del publisher.steps[task]
            step.manager = step.user_id = step.validator = step.execution = None


def account_delete_step(work: object, manager: object, user_id: object, validator: object) -> _StepEntry | None:
    """Observe the original remove_user native_step of the retained delete sequence."""
    coordinator, publisher = _publisher()
    if publisher is None or work is not publisher.work:
        return None
    task = asyncio.current_task()
    try:
        entry = _delete_step(coordinator, publisher, task, work, manager, user_id, validator)
    finally:
        task = work = manager = user_id = validator = publisher = None
    if entry is None:
        _reject("account_delete_step", coordinator)
    return entry


def _delete_step(coordinator, publisher, task, work, manager, user_id, validator) -> _StepEntry | None:
    from functools import partial
    from src.dashboard.native_work import require_current
    with coordinator._dispatch_lock:
        sequence = None if task is None else publisher.sequences.get(task)
        arguments = None if sequence is None else sequence.args
        if (task is None or not _delete_sequence(publisher, task) or task in publisher.steps
                or manager is not publisher.manager or arguments is None or len(arguments) != 7
                or arguments[0] is not work or arguments[4] is not manager or arguments[5] is not user_id
                or type(validator) is not partial or validator.func is not require_current
                or len(validator.args) != 1 or validator.args[0] is not arguments[1] or validator.keywords):
            return None
        step = _DeleteStep(manager, user_id, validator)
        publisher.steps[task] = step
    return _StepEntry(coordinator, publisher, task, step)


def before_account_remove(manager: object, user_id: object, requesting_user: object, before_save: object) -> None:
    """Compare actual remove_user locals with the pending original step; no authority."""
    coordinator = _coordinator(manager)
    publisher = None if coordinator is None else coordinator._accounts
    if publisher is None or publisher.manager is not manager:
        return
    execution = current_native_execution()
    if execution is None:
        return
    with coordinator._dispatch_lock:
        step = next((item for item in publisher.steps.values() if item.execution is execution), None)
        if step is None:
            return
        if (step.consumed or user_id is not step.user_id or requesting_user is not None
                or before_save is not step.validator):
            _reject("account_delete_step", coordinator)
        step.consumed = True


def _pinned_stat(info: os.stat_result, pin: int | None) -> bool:
    """``info`` (an lstat of the own temporary's name) is the writer's pinned own inode.

    The writer's pin keeps the inode allocated, so (st_dev, st_ino) decides;
    size, mtime and ctime are belt-and-braces for a filesystem whose st_ino
    is not unique (and what the tests' simulated number reuse exercises).
    """
    if pin is None:
        return False
    held = os.fstat(pin)
    return ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == (held.st_dev, held.st_ino, held.st_size, held.st_mtime_ns, held.st_ctime_ns))


def _directory_pair(path: Path) -> tuple | None:
    info = os.lstat(path)
    return (info.st_dev, info.st_ino) if stat.S_ISDIR(info.st_mode) else None


class _AccountSave:
    """Bounded in-memory extent of one original UserManager._save attempt."""

    __slots__ = ("coordinator", "publisher", "execution", "confirmed")

    def __init__(self, coordinator, publisher, execution) -> None:
        self.coordinator, self.publisher, self.execution = coordinator, publisher, execution
        self.confirmed = False

    def before_replace(self, temporary: str, own: tuple | None, parent: tuple | None, final: tuple | None,
                       pin: int | None) -> None:
        """Recheck live execution, destination generation and the own temp inode."""
        coordinator, publisher = self.coordinator, self.publisher
        try:
            coordinator._dispatch(publisher, self.execution)
            coordinator._account_physical(publisher, publisher.manager)
            info = os.lstat(temporary)
            base._require(own is not None and final is not None and parent is not None
                and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == own and (info.st_size, info.st_mtime_ns) == final
                and _pinned_stat(info, pin)
                and _directory_pair(Path(temporary).parent) == parent, "temporary_changed")
        except (OSError, RuntimeError, ValueError):
            coordinator._failed = True
            raise OutputRegistrationError("output_registration_temporary_changed") from None

    def replaced(self, own: tuple, parent: tuple | None, pin: int) -> None:
        """After original RAM publication: join the destination to the own temp inode.

        On success the writer's own pin (``pin``, still holding ``own``) becomes
        the destination pin; on a refusal the writer keeps and closes it.
        """
        coordinator, publisher = self.coordinator, self.publisher
        try:
            info = os.lstat(publisher.manager.data_path)
            held = os.fstat(pin)
            base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == own == (held.st_dev, held.st_ino)
                and _directory_pair(Path(publisher.manager.data_path).parent) == parent, "destination_changed")
        except (OSError, RuntimeError, ValueError):
            publisher.pin.release()
            publisher.uncertain = True
            coordinator._failed = True
            raise OutputRegistrationError("output_registration_destination_changed") from None
        publisher.expected_identity = {"device": own[0], "inode": own[1]}
        publisher.committed += 1
        self.confirmed = True
        publisher.pin.adopt(pin)  # Last: nothing after it can fail, so ownership moves exactly once.

    def finish(self, *, published: bool, cleanup_proven: bool) -> None:
        """Release the extent; unproven cleanup or unconfirmed publication poisons."""
        coordinator, publisher = self.coordinator, self.publisher
        with coordinator._dispatch_lock:
            if publisher.active is self:
                publisher.active = None
            if not self.confirmed:
                if published or not cleanup_proven:
                    publisher.uncertain = True
                    coordinator._failed = True
                else:
                    publisher.clean_failed += 1
        self.coordinator = self.publisher = self.execution = None


def before_account_save(manager: object, before_save: object) -> _AccountSave | None:
    """Admit one observed managed save before mkdir/mkstemp; None when unmanaged."""
    coordinator = _coordinator(manager)
    if coordinator is None:
        _unmanaged_path(manager.data_path)
        return None
    from src.memory.output_registration_v3 import MAX_ACCOUNT_COUNT
    publisher = coordinator._accounts
    execution = current_native_execution()
    try:
        if (publisher is None or publisher.manager is not manager or execution is None
                or publisher.phase not in {"attached", "opening"}):
            _reject("account_publisher", coordinator)
        with coordinator._dispatch_lock:
            record = publisher.invocations.get(execution)
            if record is not None:
                allowed = (record.admitted and record.manager is manager
                           and record.kwargs.get("before_save") is before_save)
            else:
                record = next((item for item in publisher.steps.values() if item.execution is execution), None)
                allowed = record is not None and record.consumed and record.validator is before_save
            if not allowed or record.saves:
                _reject("account_invocation", coordinator)
            if publisher.active is not None or publisher.uncertain:
                _reject("native_capacity", coordinator)
            if publisher.attempts >= MAX_ACCOUNT_COUNT:
                _reject("capacity", coordinator)
            record.saves = 1
        coordinator._dispatch(publisher, execution)
        coordinator._account_physical(publisher, manager)
        if publisher.phase == "attached":
            coordinator._append_account("opening", publisher.plan)
            publisher.phase = "opening"
        extent = _AccountSave(coordinator, publisher, execution)
        with coordinator._dispatch_lock:
            publisher.active = extent
            publisher.attempts += 1
        return extent
    except BaseException:
        coordinator._failed = True
        raise


def account_publisher(dashboard: object) -> OutputRegistrationV2 | None:
    """Return the coordinator observing this exact dashboard's publisher, if any."""
    manager = getattr(dashboard, "user_manager", None)
    coordinator = None if manager is None else _coordinator(manager)
    publisher = None if coordinator is None else coordinator._accounts
    return coordinator if publisher is not None and publisher.dashboard is dashboard else None


@dataclass(eq=False)
class _ConfigurationPublisher:
    owner: object
    authority: object
    phase: str = "waiting"
    plan: dict | None = None
    uncertain: bool = False
    extent: _ConfigurationCommit | None = None
    pin: _DestinationPin = field(default_factory=_DestinationPin)


def _pair(identity: dict) -> tuple[int, int]:
    return identity["device"], identity["inode"]


class _ConfigurationCommit:
    """Single-use in-memory extent of the one ConfigurationChanges._commit attempt.

    Its methods run on the writer thread inside the owner's lock. They use
    only lstat and in-memory state, never the coordinator journal lock; the
    loop reads the extent only after the writer Task has completed.
    """

    __slots__ = ("coordinator", "publisher", "state", "attempts", "temporary", "own", "confirmed",
                 "committed_identity", "committed_sha256", "reason")

    def __init__(self, coordinator: OutputRegistrationV2, publisher: _ConfigurationPublisher) -> None:
        self.coordinator, self.publisher = coordinator, publisher
        self.state, self.attempts, self.temporary, self.own = "issued", 0, None, None
        self.confirmed, self.committed_identity, self.committed_sha256 = False, None, None
        self.reason = None

    @property
    def failure(self) -> str | None:
        """The registration refusal this extent recorded on the writer thread, if any."""
        return None if self.reason is None else "output_registration_" + self.reason

    def __reduce_ex__(self, protocol: int):
        raise TypeError("configuration_extent_not_serializable")

    def _poison(self, reason: str) -> None:
        self.reason = self.reason or reason
        self.publisher.uncertain = True
        self.coordinator._failed = True
        raise OutputRegistrationError("output_registration_" + reason)

    def before_temporary(self, owner: object, path: Path, roots: tuple, identity: tuple, digest: str) -> None:
        """Admit the attempt before mkstemp: same owner, live epoch, the attached file."""
        coordinator, publisher = self.coordinator, self.publisher
        plan = publisher.plan
        if (self.state != "issued" or owner is not publisher.owner or coordinator._failed
                or coordinator.admission._output_registration is not coordinator
                or publisher.phase != "opening" or publisher.uncertain
                or str(path) != plan["path"]
                or tuple(roots) != (_pair(plan["state_identity"]), _pair(plan["parent_identity"]))
                or tuple(identity) != _pair(plan["initial_identity"]) or digest != plan["initial_sha256"]
                or not publisher.pin.holds_path(plan["path"])):
            self.state = "refused"
            self._poison("configuration_extent")
        self.state, self.attempts = "temporary", 1

    def before_replace(self, temporary: str, own: tuple | None, parent: tuple | None, final: tuple | None,
                       pin: int | None) -> None:
        """Recheck the own temp inode, its directory and the still-original destination."""
        plan = self.publisher.plan
        try:
            base._require(self.state == "temporary" and not self.coordinator._failed, "temporary_changed")
            info = os.lstat(temporary)
            base._require(own is not None and final is not None and parent is not None
                and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == own and (info.st_size, info.st_mtime_ns) == final
                and _pinned_stat(info, pin)
                and str(Path(temporary).parent) == plan["parent"]
                and Path(temporary).name.startswith(plan["temporary_scope"]["prefix"])
                and parent == _pair(plan["parent_identity"])
                and _directory_pair(Path(temporary).parent) == parent, "temporary_changed")
            destination = os.lstat(plan["path"])
            base._require(stat.S_ISREG(destination.st_mode) and destination.st_nlink == 1
                and (destination.st_dev, destination.st_ino) == _pair(plan["initial_identity"])
                and self.publisher.pin.holds(destination), "temporary_changed")
        except (OSError, RuntimeError, ValueError):
            self.state = "refused"
            self._poison("temporary_changed")
        self.state, self.temporary, self.own = "replacing", temporary, own

    def replaced(self, own: tuple, sha256: str, pin: int) -> None:
        """Immediately after os.replace: join the destination to the own temp inode.

        On success the writer's own pin (``pin``, still holding ``own``) becomes
        the destination pin; on a refusal the writer keeps and closes it.
        """
        plan = self.publisher.plan
        try:
            base._require(self.state == "replacing" and own == self.own, "destination_changed")
            info = os.lstat(plan["path"])
            held = os.fstat(pin)
            base._require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and (info.st_dev, info.st_ino) == own == (held.st_dev, held.st_ino)
                and _directory_pair(Path(plan["parent"])) == _pair(plan["parent_identity"]), "destination_changed")
        except (OSError, RuntimeError, ValueError):
            self.publisher.pin.release()
            self.state = "published"
            self._poison("destination_changed")
        self.state, self.confirmed = "published", True
        self.committed_identity, self.committed_sha256 = {"device": own[0], "inode": own[1]}, sha256
        self.publisher.pin.adopt(pin)  # Last: nothing after it can fail, so ownership moves exactly once.

    def finish(self, *, temporary: str | None, published: bool, durable: bool, cleanup_proven: bool) -> None:
        """Release the extent; an unconfirmed or undurable publication, or unproven cleanup, poisons."""
        self.temporary = temporary if temporary is not None else self.temporary
        if published and not (self.confirmed and durable) or not published and not cleanup_proven:
            self.reason = self.reason or ("publication_unconfirmed" if published else "cleanup_unproven")
            self.publisher.uncertain = True
            self.coordinator._failed = True
        self.confirmed = self.confirmed and published and durable
        self.state = "finished"


async def configuration_opening(owner: object) -> _ConfigurationCommit | None:
    """Admit the owner's single write window; None when unmanaged."""
    if not _managed():
        return None
    coordinator = _coordinator(owner)
    if coordinator is None or coordinator._configuration is None or coordinator._configuration.owner is not owner:
        _reject("required", coordinator)
    return await coordinator.configuration_opening(owner)


async def configuration_withdrawn(owner: object) -> bool:
    """Durably record the owner's actor withdrawal; False only for a truly unmanaged run.

    Mirrors configuration_opening: a managed run with no coordinator or no
    attached publisher for this owner refuses (and poisons), never reports
    an unmanaged success.
    """
    if not _managed():
        return False
    coordinator = _coordinator(owner)
    if coordinator is None or coordinator._configuration is None or coordinator._configuration.owner is not owner:
        _reject("required", coordinator)
    await coordinator.configuration_withdrawn(owner)
    return True


def configuration_withdrawal_unconfirmed(owner: object) -> None:
    """Poison the owner's epoch: the gate closed but its marker is not confirmed."""
    coordinator = _coordinator(owner)
    if coordinator is not None:
        coordinator._failed = True


async def configuration_drained(owner: object, extent: _ConfigurationCommit | None, outcome: str | None,
                                *, graph_failed: bool = False) -> None:
    """Record the owner's terminal after its writer returned; a no-op when unmanaged."""
    if not _managed():
        return
    coordinator = _coordinator(owner)
    if coordinator is None:
        if graph_failed:
            return
        _reject("required")
    await coordinator.configuration_drained(owner, extent, outcome, graph_failed=graph_failed)
