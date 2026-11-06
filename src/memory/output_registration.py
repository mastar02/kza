"""Destination metadata for managed producers; never source or close authority."""
from __future__ import annotations

import asyncio
from contextvars import Context
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import TYPE_CHECKING, NoReturn

from src.core.owned_work import ClosePhase, OwnedBlockingWork, WorkCloseTimeout

if TYPE_CHECKING:
    from contextlib import AsyncExitStack
    from src.core.runtime_admission import RuntimeAdmission
    from src.lists.list_store import ListStore
    from src.reminders.reminder_store import ReminderStore

PRODUCERS = (
    "authority_sessions", "configuration", "accounts", "privacy_authority",
    "house_vector_index", "personal_memory", "diagnostics", "events", "suggestions",
    "contexts", "capture", "collectors", "voices", "lists", "reminders", "routines",
    "household_features", "home_integrations", "dashboard_transport", "audio_input",
    "audio_health", "audio_output", "model_resources", "reasoner_transports",
    "presence_hardware", "latency", "training", "logging", "offline_copies",
    "offline_capture_tools", "offline_exports_training", "build_forensic_outputs",
)
MAX_RECORD_BYTES = 65536
MAX_EVENTS = 65536
MAX_RUN_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_NAMESPACE_ENTRIES = 16384
_KEY = object()
_ACTIVE = set()
_PHASES = {"intent": None, "attached": "intent", "opening": "attached",
           "active": "opening", "closed": "active"}


def _require(value, reason):
    if not value:
        raise RuntimeError("output_registration_" + reason)


def _encoded(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= MAX_RECORD_BYTES, "record_size")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(path):
    info = path.lstat()
    _require(not stat.S_ISLNK(info.st_mode), "redirected")
    return {"device": info.st_dev, "inode": info.st_ino}


def _identity_shape(value):
    return (type(value) is dict and set(value) == {"device", "inode"}
        and all(type(n) is int and n >= 0 for n in value.values()))


def _namespace_size(root, *, reserve_entries=0):
    """Bound both traversal cost and retained metadata; never prune history."""
    _require(reserve_entries <= MAX_NAMESPACE_ENTRIES, "capacity")
    if not root.exists():
        return 0
    _directory(root)
    size, count = 0, reserve_entries
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                _require(count <= MAX_NAMESPACE_ENTRIES, "capacity")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _require(depth == 0, "namespace_changed")
                    pending.append((Path(entry.path), depth + 1))
                else:
                    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                        and info.st_size <= MAX_RECORD_BYTES, "namespace_changed")
                    size += info.st_size
                    _require(size <= MAX_TOTAL_BYTES, "capacity")
    return size


def _directory(path):
    _require(path.is_absolute() and path.resolve(strict=True) == path
             and path.is_dir() and not path.is_symlink(), "directory")
    return _identity(path)


def _read(path):
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_size <= MAX_RECORD_BYTES, "record_file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require((info.st_dev, info.st_ino) == (before.st_dev, before.st_ino), "record_changed")
        raw = stream.read(MAX_RECORD_BYTES + 1)
    _require(len(raw) <= MAX_RECORD_BYTES, "record_size")
    value = json.loads(raw)
    _require(_encoded(value) == raw, "record_encoding")
    return value, raw


def _fsync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, raw):
    _require(len(raw) <= MAX_RECORD_BYTES, "record_size")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync(path.parent)


def _head(directory, previous, value):
    if previous is not None:
        _require(_read(directory / "HEAD.json")[0] == previous, "cas")
    else:
        _require(not (directory / "HEAD.json").exists(), "cas")
    temporary = directory / "HEAD.next"
    _write_new(temporary, _encoded(value))
    os.replace(temporary, directory / "HEAD.json")
    _fsync(directory)


def _chain_v1(directory):
    _namespace_size(directory.parent)
    _directory(directory)
    header, header_raw = _read(directory / "epoch.json")
    _require(set(header) == {"schema_version", "run_id", "generation", "binding_id",
        "config_sha256", "state_root", "state_identity", "namespace_identity", "producers"}
        and type(header["schema_version"]) is int and header["schema_version"] == 1
        and header["run_id"] == directory.name and re.fullmatch("[0-9a-f]{32}", directory.name)
        and type(header["generation"]) is int and header["generation"] > 0
        and all(type(header[k]) is str and re.fullmatch("[0-9a-f]{64}", header[k])
            for k in ("binding_id", "config_sha256"))
        and type(header["state_root"]) is str and Path(header["state_root"]).is_absolute()
        and str(Path(header["state_root"])) == header["state_root"]
        and ".." not in Path(header["state_root"]).parts
        and _identity_shape(header["state_identity"])
        and _identity_shape(header["namespace_identity"])
        and header["producers"] == list(PRODUCERS)
        and header["namespace_identity"] == _identity(directory), "header")
    head, head_raw = _read(directory / "HEAD.json")
    _require(set(head) == {"revision", "sha256"} and type(head["revision"]) is int
        and 0 <= head["revision"] <= MAX_EVENTS, "head")
    previous = _sha(_encoded(header))
    states, total = {}, len(header_raw) + len(head_raw)
    expected = {"epoch.json", "HEAD.json"}
    for revision in range(1, head["revision"] + 1):
        paths = list(directory.glob(f"{revision:012d}-*.json"))
        _require(len(paths) == 1, "chain")
        path = paths[0]
        event, raw = _read(path)
        total += len(raw)
        digest = _sha(raw)
        _require(path.name == f"{revision:012d}-{digest}.json"
            and set(event) == {"revision", "previous_sha256", "producer", "phase", "plan", "native_identity"}
            and type(event["revision"]) is int and event["revision"] == revision
            and event["previous_sha256"] == previous, "chain")
        name, phase, plan = event["producer"], event["phase"], event["plan"]
        _require(name in ("lists", "reminders") and phase in _PHASES, "event")
        prior = states.get(name)
        _require((None if prior is None else prior["phase"]) == _PHASES[phase], "transition")
        _plan_shape(plan, name, Path(header["state_root"]))
        _require(prior is None or prior["plan"] == plan, "plan_changed")
        native_identity = event["native_identity"]
        _require(_identity_shape(native_identity) if phase in ("active", "closed")
            else native_identity is None, "native_identity")
        if phase == "active":
            _require(plan["existing_identity"] is None or native_identity == plan["existing_identity"], "native_identity")
        if phase == "closed":
            _require(native_identity == prior["native_identity"], "native_identity")
        states[name] = {"phase": phase, "plan": plan, "native_identity": native_identity,
            "revision": revision, "sha256": digest}
        previous = digest
        expected.add(path.name)
    _require(head["sha256"] == previous and total <= MAX_RUN_BYTES, "head")
    _require({p.name for p in directory.iterdir()} == expected, "orphan")
    _require(_read(directory / "HEAD.json")[1] == head_raw
        and _read(directory / "epoch.json")[1] == header_raw, "journal_changed")
    return header, head, states, total


def _plan_shape(plan, producer, state):
    _require(type(plan) is dict and set(plan) == {"producer", "path", "parent_identity",
        "existing_identity", "sidecars", "classification", "before_first_open"}
        and plan["producer"] == producer and plan["classification"] == "protected_household_tasks"
        and plan["before_first_open"] is True and type(plan["path"]) is str, "plan")
    path = Path(plan["path"])
    _require(path.is_absolute() and str(path) == plan["path"] and ".." not in path.parts
        and path.is_relative_to(state) and path != state
        and plan["sidecars"] == [str(path) + suffix for suffix in ("-wal", "-shm", "-journal")], "plan")
    _require(_identity_shape(plan["parent_identity"])
        and (plan["existing_identity"] is None or _identity_shape(plan["existing_identity"])), "plan")


def _plan(producer, path, state):
    supplied = Path(path).absolute()
    _require(not supplied.is_symlink(), "destination")
    path = supplied.resolve(strict=False)
    _require(path.is_relative_to(state), "destination")
    parent = _directory(path.parent)
    existing = None
    if path.exists() or path.is_symlink():
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination")
        existing = _identity(path)
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() or sidecar.is_symlink():
            info = sidecar.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "destination")
    value = {"producer": producer, "path": str(path), "parent_identity": parent,
        "existing_identity": existing, "sidecars": [str(path) + s for s in ("-wal", "-shm", "-journal")],
        "classification": "protected_household_tasks", "before_first_open": True}
    _plan_shape(value, producer, state)
    return value


def _nominal(store, producer):
    from src.lists.list_store import ListStore
    from src.reminders.reminder_store import ReminderStore
    return type(store) is {"lists": ListStore, "reminders": ReminderStore}.get(producer)


def _chain(directory):
    header, _ = _read(directory / "epoch.json")
    if header.get("schema_version") == 1:
        return _chain_v1(directory)
    if header.get("schema_version") == 2:
        from src.memory.output_registration_v2 import chain
        return chain(directory)
    raise RuntimeError("output_registration_schema")


@dataclass(frozen=True)
class DestinationRegistration:
    """Instance-bound registration; not a policy permission or cleanup receipt."""
    coordinator: DestinationRegistrationCoordinator
    store: ListStore | ReminderStore
    producer: str
    path: str

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("output_registration_not_serializable")


class DestinationRegistrationCoordinator:
    """Own only bounded journal work; original stores own their native work."""

    def __init__(self, admission: RuntimeAdmission, config: dict, *, key: object) -> None:
        _require(key is _KEY and admission._installed, "actual_admission")
        self.admission = admission
        self._expected = {name: str(Path(config.get(name, {}).get("db_path", f"./data/{name}.db")).absolute())
                          for name in ("lists", "reminders")}
        self._disk = OwnedBlockingWork(capacity=64, isolate_context=True)
        self._lock = threading.RLock()
        self._handles = {}
        self._initializers = {}
        self._connections = {}
        self._epoch = None
        self._epoch_identity = None
        self._failed = False
        self._closing = None
        self._directory = Path(admission._ticket["admission_root"]) / "admission/output-inventory/v1/runs" / admission._ticket["run_id"]
        self._directory_identity = None

    def _current(self):
        _require(self.admission._output_registration is self and not self._failed, "coordinator")
        self.admission.revalidate()
        if self._directory_identity is not None:
            _require(_directory(self._directory) == self._directory_identity, "namespace_changed")
        if self._epoch_identity is not None:
            path = self._directory / "epoch.json"
            _require(_identity(path) == self._epoch_identity
                and _read(path)[0] == self._epoch, "epoch_changed")

    def _start(self):
        with self._lock:
            self._current()
            ticket = self.admission._ticket
            _require(re.fullmatch("[0-9a-f]{32}", ticket["run_id"]), "run_id")
            root = Path(ticket["admission_root"]) / "admission"
            _directory(root)
            header = {"schema_version": 1, "run_id": ticket["run_id"], "generation": ticket["generation"],
                "binding_id": ticket["binding_id"], "config_sha256": self.admission._config_digest,
                "state_root": str(self.admission.state_root), "state_identity": _identity(self.admission.state_root),
                "namespace_identity": {"device": 2**64 - 1, "inode": 2**64 - 1},
                "producers": list(PRODUCERS)}
            # Validate encoding and reserve header plus HEAD and atomic-temp
            # overhead before creating even the first namespace directory.
            reservation = len(_encoded(header)) + 2 * len(_encoded({"revision": 0, "sha256": "0" * 64}))
            _require(reservation <= MAX_RUN_BYTES
                and _namespace_size(self._directory.parent, reserve_entries=4) + reservation <= MAX_TOTAL_BYTES, "capacity")
            for part in ("output-inventory", "v1", "runs"):
                child = root / part
                if not child.exists():
                    child.mkdir(mode=0o700)
                    _fsync(root)
                _directory(child)
                root = child
            _require(not self._directory.exists(), "run_exists")
            self._directory.mkdir(mode=0o700)
            _fsync(root)
            self._directory_identity = _directory(self._directory)
            header["namespace_identity"] = self._directory_identity
            raw = _encoded(header)
            self._epoch = header
            _write_new(self._directory / "epoch.json", raw)
            self._epoch_identity = _identity(self._directory / "epoch.json")
            _head(self._directory, None, {"revision": 0, "sha256": _sha(raw)})

    async def _run(self, function, *args):
        try:
            return await self._disk.run(function, *args)
        except BaseException:
            self._failed = True
            raise

    def _append(self, producer, phase, plan):
        with self._lock:
            self._current()
            _, head, states, total = _chain(self._directory)
            previous = states.get(producer)
            _require(producer in ("lists", "reminders") and phase in _PHASES
                and (None if previous is None else previous["phase"]) == _PHASES[phase], "transition")
            _plan_shape(plan, producer, self.admission.state_root)
            _require(previous is None or previous["plan"] == plan, "plan_changed")
            native_identity = None
            if phase in ("active", "closed"):
                path = Path(plan["path"])
                info = path.lstat()
                _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "native_identity")
                native_identity = _identity(path)
                expected = plan["existing_identity"] if phase == "active" else previous["native_identity"]
                _require(expected is None or expected == native_identity, "native_identity")
            revision = head["revision"] + 1
            raw = _encoded({"revision": revision, "previous_sha256": head["sha256"],
                "producer": producer, "phase": phase, "plan": plan, "native_identity": native_identity})
            _require(revision <= MAX_EVENTS and total + len(raw) <= MAX_RUN_BYTES, "capacity")
            next_head = _encoded({"revision": revision, "sha256": _sha(raw)})
            size = _namespace_size(self._directory.parent, reserve_entries=2)
            _require(size + len(raw) + len(next_head) <= MAX_TOTAL_BYTES
                and total + len(raw) + len(next_head) <= MAX_RUN_BYTES, "capacity")
            digest = _sha(raw)
            _write_new(self._directory / f"{revision:012d}-{digest}.json", raw)
            self._current()
            _head(self._directory, head, {"revision": revision, "sha256": digest})

    async def register_store(
        self, store: ListStore | ReminderStore, producer: str,
    ) -> DestinationRegistration:
        """Persist intent and exact attachment before the actual store open."""
        _require(_nominal(store, producer) and producer not in self._handles
            and store._db is None and self._closing is None, "store")
        path = str(Path(store._db_path).absolute())
        _require(path == self._expected[producer], "configured_path")
        plan = await self._run(_plan, producer, path, self.admission.state_root)
        await self._run(self._append, producer, "intent", plan)
        handle = DestinationRegistration(self, store, producer, path)
        await self._run(self._append, producer, "attached", plan)
        self._handles[producer] = (handle, plan)
        store._destination_registration = handle
        return handle

    def _check(self, handle, store, producer, path):
        self._current()
        _require(type(handle) is DestinationRegistration and handle.coordinator is self
            and handle.store is store and _nominal(store, producer)
            and handle.producer == producer and handle.path == str(Path(path).absolute())
            and self._handles.get(producer, (None,))[0] is handle
            and store._destination_registration is handle, "attachment")
        plan = self._handles[producer][1]
        supplied = Path(handle.path)
        _require(not supplied.is_symlink() and str(supplied.resolve(strict=False)) == plan["path"]
            and _directory(Path(plan["path"]).parent) == plan["parent_identity"], "destination_changed")
        return plan

    def _opening(self, handle, store, producer, path):
        with self._lock:
            plan = self._check(handle, store, producer, path)
            _require(_plan(producer, path, self.admission.state_root) == plan, "destination_changed")
            self._append(producer, "opening", plan)

    async def opening(
        self, handle: DestinationRegistration, store: ListStore | ReminderStore,
        producer: str, path: str,
    ) -> None:
        """Check actual running authority and original destination before connect."""
        _require(store._lifecycle.locked() and store._db is None
            and producer not in self._initializers, "initialization_owner")
        await self._run(self._opening, handle, store, producer, path)
        self._initializers[producer] = asyncio.current_task()

    async def active(
        self, handle: DestinationRegistration, store: ListStore | ReminderStore,
    ) -> None:
        """Record initialization only from its original task after native return."""
        _require(self._initializers.get(handle.producer) is asyncio.current_task()
            and store._lifecycle.locked() and store._db is not None
            and store._work is not None and store._db._connection is not None, "initialization_owner")
        self._connections[handle.producer] = store._db
        def publish():
            plan = self._check(handle, store, handle.producer, store._db_path)
            self._append(handle.producer, "active", plan)
        try:
            await self._run(publish)
        except BaseException:
            # Stop new consumers synchronously. The original store still owns
            # this connection and its retained close task drains existing work.
            store._work.begin_close()
            raise
        finally:
            self._initializers.pop(handle.producer, None)

    async def closed(
        self, handle: DestinationRegistration, store: ListStore | ReminderStore,
    ) -> None:
        """Observe the original retained task after real native close."""
        _require(store._closing is asyncio.current_task() and store._lifecycle.locked()
            and store._db is None and store._work is None, "close_owner")
        connection = self._connections.get(handle.producer)
        _require(connection is None or connection._connection is None, "native_close_pending")
        if self._failed:
            return  # Failed epoch stays incomplete; no synthetic closure event.
        def publish():
            plan = self._check(handle, store, handle.producer, store._db_path)
            _, _, states, _ = _chain(self._directory)
            if states[handle.producer]["phase"] == "active":
                self._append(handle.producer, "closed", plan)
        await self._run(publish)

    async def aclose(self) -> None:
        """Drain metadata after existing store callbacks; never own their close."""
        if self._closing is None:
            def verify_observations():
                self._current()
                _, _, states, _ = _chain(self._directory)
                _require(all(states.get(name, {}).get("phase") == "closed"
                    for name in self._handles), "epoch_incomplete")
            async def finish():
                pending = [h.store._closing for h, _ in self._handles.values() if h.store._closing is not None]
                try:
                    if pending:
                        await asyncio.gather(*(asyncio.shield(task) for task in pending))
                    if not self._failed:
                        await self._run(verify_observations)
                finally:
                    await self._disk.aclose()
                _require(not self._failed, "epoch_incomplete")
                # Installed managed provenance remains sticky through process exit.
            self._closing = asyncio.create_task(finish(), context=Context())
            self._closing.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(self._closing)
        except TimeoutError as error:
            raise WorkCloseTimeout("output_registration_close_pending", phase=ClosePhase.DRAIN,
                                   diagnostics=self._disk.diagnostics()) from error


async def begin_registration(
    admission: RuntimeAdmission, config: dict, shutdown: AsyncExitStack | None,
) -> DestinationRegistrationCoordinator:
    """Install a coordinator only on the actual verified installed admission."""
    from src.core.runtime_admission import RuntimeAdmission, _config_digest, _verified_runs, _verified_admissions
    _require(type(admission) is RuntimeAdmission and admission._installed
        and _verified_admissions.get(admission) is True
        and admission._shutdown is shutdown and admission._output_registration is None
        and (admission._ticket["run_id"], admission._ticket["binding_id"]) in _verified_runs
        and _config_digest(config) == admission._config_digest, "actual_admission")
    admission.revalidate()
    from src.memory.output_registration_v2 import OutputRegistrationV2
    coordinator = OutputRegistrationV2(admission, config, key=_KEY)
    admission._output_registration = coordinator
    _ACTIVE.add(coordinator)
    shutdown.push_async_callback(coordinator.aclose)
    await coordinator._run(coordinator._start)
    return coordinator


async def before_store_open(
    store: ListStore | ReminderStore, producer: str, path: str,
    handle: DestinationRegistration | None,
) -> None:
    """Reject omitted managed registration independently of the optional handle."""
    from src.core.runtime_admission import _validate_path, _verified_admissions
    if _ACTIVE or any(_verified_admissions.values()):
        _require(type(handle) is DestinationRegistration and handle.coordinator in _ACTIVE, "required")
        await handle.coordinator.opening(handle, store, producer, path)
        return
    managed = await asyncio.to_thread(_validate_path, path, initial=False)
    if managed is None:
        _require(handle is None, "unmanaged_handle")
        return
    _require(type(handle) is DestinationRegistration, "required")
    await handle.coordinator.opening(handle, store, producer, path)


async def store_open_returned(store: ListStore | ReminderStore) -> None:
    """Publish initialization only after the original store finishes it."""
    handle = store._destination_registration
    if handle is not None:
        await handle.coordinator.active(handle, store)


async def store_close_returned(store: ListStore | ReminderStore) -> None:
    """Called inside the store's retained task after native close returned."""
    handle = store._destination_registration
    if handle is not None:
        await handle.coordinator.closed(handle, store)


def read_output_inventory(controller: object) -> tuple[dict, ...]:
    """Recover immutable v1 and v2 observations without cross-version promotion."""
    from src.memory.output_registration_v2 import read_inventory
    return read_inventory(controller)
