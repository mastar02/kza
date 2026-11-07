"""Bind managed application construction and closure to a controller lease."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, AsyncExitStack, nullcontext
from contextvars import Context
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import threading
from types import MethodType, ModuleType
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from src.memory.output_registration import DestinationRegistrationCoordinator

MARKER = ".kza-state-binding.json"
TICKET_ENV = "KZA_RUNTIME_ADMISSION"
_MAX_NATIVE_WITNESSES = 128
_guard = threading.RLock()
_companions: dict[str, ModuleType] = {}
_verified_runs: set[tuple[str, str]] = set()
_installed_runs: set[str] = set()
# Exact factory provenance; installed entries stay managed until process exit.
_verified_admissions: dict[object, bool] = {}


def _companion(relative: str) -> ModuleType:
    with _guard:
        if relative not in _companions:
            own = Path(__file__).resolve()
            suffix = ".pyc" if own.suffix == ".pyc" else ".py"
            path = own.parents[2] / (relative + suffix)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("runtime_admission_trusted_module_missing")
            name = "_kza_runtime_" + hashlib.sha256(relative.encode()).hexdigest()[:16] + "_" + str(os.getpid())
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("runtime_admission_controller_loader_missing")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            _companions[relative] = module
        return _companions[relative]


def _stage() -> ModuleType:
    return _companion("scripts/stage_runtime")


def diagnostic_history_validation_gate(path: str | Path) -> AbstractContextManager[object]:
    """Serialize a historical writer with the actual companion's full probes.

    This is scheduling only: neither a marker, ticket nor this gate validates
    authority or permits a writer. The original operation must keep its own
    policy and native lifetime checks. Call only inside its blocking worker,
    before opening the store, and retain the gate through its original close.

    Args:
        path: The original historical diagnostic destination.

    Returns:
        The actual full-validator mutex for a managed process or destination;
        an inert context for an unmanaged operation. Installed provenance stays
        sticky even if the ticket or marker is later removed.
    """
    with _guard:
        installed = any(run in _installed_runs for run, _ in _verified_runs)
    if installed or TICKET_ENV in os.environ or _marker_for(path) is not None:
        return _stage()._ADMISSION_VALIDATION_LOCK
    return nullcontext()


def _marker_for(path: str | Path) -> Path | None:
    if os.fspath(path) == ":memory:":
        return None
    resolved = Path(path).resolve(strict=False)
    for parent in (resolved.parent, *resolved.parent.parents):
        marker = parent / MARKER
        try:
            info = marker.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or marker.is_symlink():
            raise RuntimeError("runtime_admission_marker_not_regular")
        return parent
    return None


def _ticket() -> dict | None:
    raw = os.environ.get(TICKET_ENV)
    if raw is None:
        return None
    if not raw or len(raw.encode()) > 16384:
        raise RuntimeError("runtime_admission_ticket_invalid")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("runtime_admission_ticket_invalid") from error
    if type(value) is not dict:
        raise RuntimeError("runtime_admission_ticket_invalid")
    return value


def _validate_path(path: str | Path, *, initial: bool) -> tuple[dict, dict] | None:
    state = _marker_for(path)
    ticket = _ticket()
    if state is None:
        if ticket is not None:
            raise RuntimeError("runtime_admission_state_unbound")
        return None
    if ticket is None:
        raise RuntimeError("runtime_admission_ticket_missing")
    with _guard:
        admitted = [item for item, installed in _verified_admissions.items()
            if installed and item._ticket == ticket and item.state_root == state]
    if len(admitted) == 1 and not initial:
        admission = admitted[0]
        admission.revalidate()
        binding = admission._binding
    else:
        binding = _stage().validate_admission_ticket(state, ticket, check_state=initial)
    return ticket, binding


def _config_digest(config: dict) -> str:
    try:
        raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise RuntimeError("runtime_admission_config_not_canonical") from error
    return hashlib.sha256(raw).hexdigest()


def _check_effective_config(config: dict, state: Path) -> str:
    import yaml
    from src.core.settings_schema import check_unresolved_env_vars, resolve_env_vars, validate_settings

    path = state / "config/settings.yaml"
    if path.is_symlink() or path.resolve(strict=False) != path or not path.is_file():
        raise RuntimeError("runtime_admission_canonical_config_missing")
    if path.stat().st_size > 4 * 1024 * 1024:
        raise RuntimeError("runtime_admission_config_too_large")
    persisted = yaml.safe_load(path.read_text())
    if type(persisted) is not dict:
        raise RuntimeError("runtime_admission_config_invalid")
    expected = resolve_env_vars(persisted)
    check_unresolved_env_vars(expected)
    expected = validate_settings(expected)
    actual = _config_digest(config)
    if actual != _config_digest(expected):
        raise RuntimeError("runtime_admission_effective_config_changed")
    return actual


def _check_serving_paths(config: dict, state: Path) -> None:
    keys = {"db_path", "privacy_db_path", "chroma_path", "preferences_path", "data_path",
        "data_dir", "conversations_dir", "exports_dir", "metrics_path", "events_db",
        "suggestions_path", "base_path", "control_selection_path", "output_dir",
        "manifest_path", "ledger_path", "storage_path"}
    special = {("vectordb", "path"), ("ambient", "keep_audio", "dir"),
        ("rooms", "wake_word", "capture_clips", "dir")}

    def inspect(value, prefix=()):
        if type(value) is dict:
            for name, child in value.items():
                path = (*prefix, name)
                if (name in keys or path in special) and isinstance(child, (str, os.PathLike)) and child:
                    if not Path(child).resolve(strict=False).is_relative_to(state):
                        raise RuntimeError("runtime_admission_foreign_serving_store:" + ".".join(path))
                elif isinstance(child, (dict, list)):
                    inspect(child, path)
        elif type(value) is list:
            for number, child in enumerate(value):
                inspect(child, (*prefix, str(number)))
    inspect(config)


async def _construction_return(task: asyncio.Task):
    """Consume an actual owned return before propagating observer cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancelled and not isinstance(error, asyncio.CancelledError):
            raise BaseExceptionGroup("runtime admission construction failed during cancellation",
                                     [error, asyncio.CancelledError()]) from None
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class _DiagnosticConstruction:
    """Serialize final admission with the nominal historical writer's lease."""

    def __init__(self, graph):
        from src.core.owned_work import OwnedBlockingWork
        from src.memory.erasure_coordinator import CleanupDriver, ErasureCoordinator
        from src.memory.erasure_journal import ErasureParticipant
        from src.obs.diagnostic_erasure import DiagnosticServices

        self.graph, self.privacy = graph, graph.privacy
        self.coordinator = graph.erasure_services
        if (type(self.coordinator) is not ErasureCoordinator
                or self.coordinator.privacy is not self.privacy):
            raise RuntimeError("runtime_admission_diagnostic_owner_changed")
        self.participant = ErasureParticipant.DIAGNOSTICS
        self.driver = self.coordinator._drivers.get(self.participant)
        if type(self.driver) is not CleanupDriver or self.driver.versions != frozenset({2}):
            raise RuntimeError("runtime_admission_diagnostic_owner_changed")
        self.callback = self.driver.callback
        self.owner = getattr(self.callback, "__self__", None)
        if (type(self.callback) is not MethodType or type(self.owner) is not DiagnosticServices
                or self.callback.__func__ is not DiagnosticServices.clean
                or self.owner.privacy is not self.privacy
                or type(self.owner._lock) is not asyncio.Lock
                or type(self.owner._disk) is not OwnedBlockingWork):
            raise RuntimeError("runtime_admission_diagnostic_owner_changed")
        self.lock, self.disk = self.owner._lock, self.owner._disk
        self.verify()

    def verify(self):
        if (self.graph.privacy is not self.privacy
                or self.graph.erasure_services is not self.coordinator
                or self.coordinator.privacy is not self.privacy
                or self.coordinator.journal is not self.privacy.ledger.erasures
                or self.coordinator._drivers.get(self.participant) is not self.driver
                or self.driver.callback is not self.callback
                or self.owner.privacy is not self.privacy
                or self.owner._lock is not self.lock or self.owner._disk is not self.disk
                or self.coordinator._closed or self.owner._closed or self.disk.status()["closed"]):
            raise RuntimeError("runtime_admission_diagnostic_owner_changed")
        self.privacy.require_available()


class RuntimeAdmission:
    """Observe an actual owned-stack drain before recording closure."""

    def __init__(self, ticket: dict, binding: dict, privacy_path: str, config_digest: str):
        self._ticket = json.loads(json.dumps(ticket))
        self._binding = json.loads(json.dumps(binding))
        self._privacy_path = privacy_path
        self._config_digest = config_digest
        self._graph = None
        self._installed = False
        self._shutdown: AsyncExitStack | None = None
        self._closure_witness = None
        self._closure_prepared = False
        self._closure_failed = False
        self._stop_original = None
        self._stop_observer = None
        self._diagnostic_construction = None
        self._construction_revalidation: asyncio.Task | None = None
        self._output_registration = None
        self._validation_lock = threading.RLock()
        self._validation_snapshot = None
        self._native_validations = WeakKeyDictionary()

    @property
    def state_root(self) -> Path:
        """Return the canonical state container selected by this admission."""
        return Path(self._binding["state_root"])

    def install_on(self, shutdown: AsyncExitStack) -> None:
        """Register first so this observer runs after all resource exits."""
        if not isinstance(shutdown, AsyncExitStack):
            raise TypeError("runtime_admission_requires_owned_stack")
        with _guard:
            run = self._ticket["run_id"]
            if self._installed or run in _installed_runs:
                raise RuntimeError("runtime_admission_run_already_installed")
            _installed_runs.add(run)
            self._installed = True
            self._shutdown = shutdown
            if self in _verified_admissions:
                _verified_admissions[self] = True
        shutdown.push_async_exit(self._after_owned_close)

    async def begin_destination_registration(self, config: dict) -> DestinationRegistrationCoordinator:
        """Register producer destinations without replacing source/close authority."""
        from src.memory.output_registration import begin_registration
        return await begin_registration(self, config, self._shutdown)

    def attach_graph(self, graph) -> None:
        """Attach the actual public graph after successful construction."""
        module = sys.modules.get("src.application")
        expected = None if module is None else getattr(module, "ApplicationGraph", None)
        if self._graph is not None or expected is None or type(graph) is not expected:
            raise RuntimeError("runtime_admission_public_graph_required")
        if Path(graph.privacy.ledger.path).resolve() != Path(self._privacy_path).resolve():
            raise RuntimeError("runtime_admission_privacy_path_changed")
        if _config_digest(graph.config) != self._config_digest:
            raise RuntimeError("runtime_admission_effective_config_changed")
        if self._shutdown is None:
            raise RuntimeError("runtime_admission_stack_not_installed")
        diagnostics = _DiagnosticConstruction(graph)
        original = graph._stop_services
        if (getattr(original, "__self__", None) is not graph
                or getattr(original, "__func__", None) is not expected._stop_services):
            raise RuntimeError("runtime_admission_stop_owner_changed")
        self._stop_original = original
        self._stop_observer = self._observe_service_stop
        graph._stop_services = self._stop_observer
        self._closure_witness = _companion("src/memory/runtime_closure").RuntimeClosureWitness.capture(graph)
        self._diagnostic_construction = diagnostics
        self._graph = graph
        self._shutdown.push_async_exit(self._before_owned_close)

    async def _prepare_closure(self) -> None:
        if self._closure_failed:
            raise RuntimeError("runtime_admission_closure_unconfirmed")
        if self._closure_prepared:
            return
        if self._closure_witness is None:
            self._closure_failed = True
            raise RuntimeError("runtime_admission_closure_witness_missing")
        try:
            await self._closure_witness.pre_close()
        except BaseException:
            self._closure_failed = True
            raise
        self._closure_prepared = True

    async def _observe_service_stop(self, *args, **kwargs) -> None:
        # Observe live identities before Graph.run retires them. Always attempt
        # the original cleanup, even when observation itself fails.
        failures = []
        try:
            await self._prepare_closure()
        except BaseException as error:
            failures.append(error)
        try:
            await self._stop_original(*args, **kwargs)
        except BaseException as error:
            self._closure_failed = True
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("runtime admission stop unconfirmed", failures)

    def revalidate(self) -> None:
        """Complete the original full validation before a new native invocation."""
        with self._validation_lock:
            snapshot = _stage().admission_validation_snapshot(self.state_root, self._ticket)
            binding = _stage().validate_admission_live(self.state_root, self._ticket, snapshot)
            if binding != self._binding:
                raise RuntimeError("runtime_admission_binding_changed")
            self._validation_snapshot = snapshot

    def _revalidate_live(self, snapshot: object | None = None) -> None:
        """Check live authority without acquiring SQL, writer probes, or full mutex."""
        observation = snapshot
        if observation is None:
            from src.core.owned_work import current_native_execution
            execution = current_native_execution()
            with _guard:
                observation = (self._native_validations.get(execution)
                    if execution is not None else None)
            if observation is None:
                observation = self._validation_snapshot
        if observation is None:
            raise RuntimeError("runtime_admission_validation_missing")
        binding = _stage().validate_admission_live(self.state_root, self._ticket, observation)
        if binding != self._binding:
            raise RuntimeError("runtime_admission_binding_changed")

    def _preflight_native(self, execution: object) -> None:
        from src.core.owned_work import validate_native_execution
        if _verified_admissions.get(self) is not True or not validate_native_execution(execution):
            raise RuntimeError("runtime_admission_native_owner")
        with self._validation_lock:
            with _guard:
                for previous in tuple(self._native_validations):
                    if previous.task.done():
                        del self._native_validations[previous]
                if len(self._native_validations) >= _MAX_NATIVE_WITNESSES:
                    raise RuntimeError("runtime_admission_native_capacity")
            self.revalidate()
            snapshot = self._validation_snapshot
            with _guard:
                self._native_validations[execution] = snapshot

    def _revalidate_native(self, execution: object) -> None:
        from src.core.owned_work import validate_native_execution
        if _verified_admissions.get(self) is not True or not validate_native_execution(execution):
            raise RuntimeError("runtime_admission_native_owner")
        with _guard:
            snapshot = self._native_validations.get(execution)
        if snapshot is None:
            raise RuntimeError("runtime_admission_native_owner")
        self._revalidate_live(snapshot)

    def _historical_target_scope(self, snapshot: object, target: dict) -> tuple[bytes, bytes]:
        """Project one previously verified delegated target, never a root grant."""
        stage = _stage()
        try:
            stage._validate_target_record(target)
            ownership = json.loads(snapshot.ownership_json)
            selected = ownership["delegations"][target["id"]]
            record, source = selected["record"], selected["source"]
            if (target["participant"] not in {"events", "suggestions"}
                    or type(ownership["policy"]["schema_version"]) is not int
                    or ownership["policy"]["schema_version"] != 1
                    or ownership["targets"].get(target["id"]) != target
                    or record["targets"].get(target["id"]) != target
                    or target["id"] not in ownership["policy"]["delegated_targets"]
                    or record["source_binding"] not in ownership["policy"]["source_bindings"]
                    or record["destination_state"] != str(self.state_root)
                    or record["destination_identity"] != self._binding["state_identity"]
                    or stage._digest(source) != record["source_binding"]
                    or selected["marker"].get("binding_id") != record["source_binding"]
                    or stage._digest(selected["proof"]) != record["source_close_sha256"]):
                raise RuntimeError("runtime_admission_target_scope")
            path, root = Path(target["descriptor"]["path"]), Path(source["state_root"])
            if root == self.state_root or not path.is_relative_to(root) or path == root:
                raise RuntimeError("runtime_admission_target_scope")
            policy_json = stage._canonical_json(ownership["policy"]) + b"\n"
            if hashlib.sha256(policy_json).hexdigest() != self._binding["ownership_sha256"]:
                raise RuntimeError("runtime_admission_target_scope")
            scope_json = stage._canonical_json({
                "scope_schema_version": 1, "source_binding": record["source_binding"],
                "delegation_sha256": stage._digest(record),
                "source_close_sha256": record["source_close_sha256"],
                "source_state_root": source["state_root"],
                "source_state_identity": source["state_identity"],
            })
            return scope_json, policy_json
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError("runtime_admission_target_scope") from exc

    def _plan_historical_target(self, target: dict) -> tuple[bytes, bytes]:
        """Return a live planning observation without granting native access."""
        if _verified_admissions.get(self) is not True:
            raise RuntimeError("runtime_admission_target_scope")
        snapshot = self._validation_snapshot
        if snapshot is None:
            raise RuntimeError("runtime_admission_validation_missing")
        self._revalidate_live(snapshot)
        return self._historical_target_scope(snapshot, target)

    def _native_historical_target(self, execution: object, target: dict) -> tuple[bytes, bytes]:
        """Join one exact target to this actual invocation's issued snapshot."""
        from src.core.owned_work import validate_native_execution
        if _verified_admissions.get(self) is not True or not validate_native_execution(execution):
            raise RuntimeError("runtime_admission_native_owner")
        with _guard:
            snapshot = self._native_validations.get(execution)
        if snapshot is None:
            raise RuntimeError("runtime_admission_native_owner")
        self._revalidate_live(snapshot)
        return self._historical_target_scope(snapshot, target)

    async def revalidate_constructed(self) -> None:
        """Keep historical repair excluded until final native validation returns.

        This neither closes maintenance nor suppresses foreign writer locks.
        The construction Task owns the lock independently of its observers.
        """
        guard = self._diagnostic_construction
        if self._graph is None or guard is None:
            raise RuntimeError("runtime_admission_public_graph_required")
        current = asyncio.current_task()
        if current is self._construction_revalidation or current in guard.owner._operations:
            raise RuntimeError("runtime_admission_construction_reentrant")
        guard.verify()
        previous = self._construction_revalidation
        if previous is None or previous.done() and not previous.cancelled() and previous.exception() is None:
            async def execute():
                async with guard.lock:
                    guard.verify()
                    drain = asyncio.create_task(guard.disk.drain_admitted(), context=Context())
                    await _construction_return(drain)
                    guard.verify()
                    validation = asyncio.create_task(asyncio.to_thread(self.revalidate), context=Context())
                    await _construction_return(validation)
                    guard.verify()
            self._construction_revalidation = asyncio.create_task(execute(), context=Context())
        await _construction_return(self._construction_revalidation)

    async def _after_owned_close(self, exc_type, exc, traceback) -> bool:
        # The stack passes the outcome of every earlier resource exit. Partial
        # factories, observer timeouts and failed drains receive no receipt.
        if exc_type is not None or self._graph is None or not self._installed:
            return False
        if self._closure_failed or self._graph._stop_services is not self._stop_observer:
            raise RuntimeError("runtime_admission_closure_unconfirmed")
        status = self._graph.privacy.resource_status()
        if status["writer_lease"] or status["closing"] or status["close_phase"] != "complete":
            raise RuntimeError("runtime_admission_privacy_drain_unconfirmed")
        if self._closure_witness is None:
            raise RuntimeError("runtime_admission_closure_witness_missing")
        proof = await asyncio.to_thread(self._closure_witness.verify_closed)
        required = ("bindings_unchanged", "model_owners_closed", "graph_closed",
            "privacy_closed", "sdk_closed", "native_closed")
        if (type(proof) is not dict or type(proof.get("schema_version")) is not int
                or proof["schema_version"] != 1
                or any(proof.get(name) is not True for name in required)):
            raise RuntimeError("runtime_admission_closure_proof_invalid")
        if _config_digest(self._graph.config) != self._config_digest:
            raise RuntimeError("runtime_admission_effective_config_changed")
        await asyncio.to_thread(self.revalidate)
        resources = {name: proof[name] for name in
            ("graph_closed", "privacy_closed", "sdk_closed", "native_closed")}
        await asyncio.to_thread(_stage().write_admission_receipt, self._ticket, resources)
        return False

    async def _before_owned_close(self, exc_type, exc, traceback) -> bool:
        await self._prepare_closure()
        return False


def verify_runtime_admission(config: dict, *, runtime_root: Path | None = None) -> RuntimeAdmission | None:
    """Validate managed state before any application resources are constructed.

    Args:
        config: The same explicit central configuration used by the graph.
        runtime_root: Exact runtime selected by a trusted compatibility launcher.

    Returns:
        A lifetime observer, or None for an unmanaged development configuration.
    """
    path = os.fspath(config.get("memory", {}).get("privacy_db_path", "./data/memory_privacy.db"))
    verified = _validate_path(path, initial=True)
    if verified is None:
        return None
    ticket, binding = verified
    own_runtime = Path(__file__).resolve().parents[2] if runtime_root is None else Path(runtime_root).resolve()
    if own_runtime != Path(binding["runtime_root"]).resolve():
        raise RuntimeError("runtime_admission_code_root_changed")
    state = Path(binding["state_root"]).resolve()
    digest = _check_effective_config(config, state)
    _check_serving_paths(config, state)
    # Detect a private-file change during decoding before the first constructor.
    _stage().validate_admission_ticket(state, ticket, check_state=True)
    admission = RuntimeAdmission(ticket, binding, path, digest)
    with _guard:
        _verified_runs.add((ticket["run_id"], ticket["binding_id"]))
        _verified_admissions[admission] = False
    return admission


def verify_privacy_admission(path: str | Path) -> None:
    """Recheck the original lease; entered initializers reuse their full preflight."""
    from src.core.owned_work import current_native_execution
    from src.memory.output_registration_v2 import OutputRegistrationV2
    execution = current_native_execution()
    ticket = _ticket()
    with _guard:
        admitted = [item for item, installed in _verified_admissions.items()
            if installed and item._ticket == ticket
            and type(item._output_registration) is OutputRegistrationV2]
        key = None if ticket is None else (ticket.get("run_id"), ticket.get("binding_id"))
        initial = key not in _verified_runs
    if len(admitted) == 1:
        admission = admitted[0]
        if initial or _marker_for(path) != admission.state_root:
            raise RuntimeError("runtime_admission_state_unbound")
        # A managed V2 initializer is already under the privacy owner's lock.
        # Missing/replayed dispatch proof fails here; it never starts a scan.
        admission._revalidate_native(execution)
        binding = admission._binding
    else:
        verified = _validate_path(path, initial=initial)
        if verified is None:
            return
        ticket, binding = verified
    if Path(__file__).resolve().parents[2] != Path(binding["runtime_root"]).resolve():
        raise RuntimeError("runtime_admission_code_root_changed")
    with _guard:
        _verified_runs.add((ticket["run_id"], ticket["binding_id"]))


def seal_runtime_configuration(config: dict) -> dict:
    """Detach managed configuration from mutable caller-owned dictionaries."""
    return copy.deepcopy(config) if os.environ.get(TICKET_ENV) is not None else config


async def build_compatible_application(
    builder: Callable[..., Awaitable[object]], config: dict, *, shutdown: AsyncExitStack,
    runtime_root: Path, factories=None,
) -> object:
    """Apply actual admission and drain observation to compatible older code."""
    module = sys.modules.get("src.main")
    if (module is None or getattr(builder, "__module__", None) != "src.main"
            or hasattr(module, "verify_runtime_admission")):
        raise RuntimeError("runtime_admission_compatible_older_builder_required")
    config = seal_runtime_configuration(config)
    admission = await asyncio.to_thread(verify_runtime_admission, config, runtime_root=runtime_root)
    if admission is None:
        raise RuntimeError("runtime_admission_compatible_state_unbound")
    admission.install_on(shutdown)
    graph = await builder(config, shutdown=shutdown, factories=factories)
    admission.attach_graph(graph)
    await admission.revalidate_constructed()
    return graph
