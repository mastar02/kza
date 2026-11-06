"""Bind managed application construction and closure to a controller lease."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
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

if TYPE_CHECKING:
    from src.memory.output_registration import DestinationRegistrationCoordinator

MARKER = ".kza-state-binding.json"
TICKET_ENV = "KZA_RUNTIME_ADMISSION"
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
        """Check the live head/lease while allowing legitimate row writes."""
        binding = _stage().validate_admission_ticket(
            Path(self._binding["state_root"]), self._ticket, check_state=False)
        if binding != self._binding:
            raise RuntimeError("runtime_admission_binding_changed")

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
    """Recheck a managed lease immediately before private SQLite can open."""
    ticket = _ticket()
    with _guard:
        key = None if ticket is None else (ticket.get("run_id"), ticket.get("binding_id"))
        initial = key not in _verified_runs
    verified = _validate_path(path, initial=initial)
    if verified is not None:
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
