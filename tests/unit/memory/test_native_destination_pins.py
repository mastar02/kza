"""Native SQLite destinations are pinned from plan until epoch close (review F4).

Each native destination is named by (st_dev, st_ino) between its plan and its
opens. ext4 hands a freed inode number to the next allocation at once, so a
same-uid delete and recreate at the path could carry the recorded number.
On Linux an O_PATH descriptor holds the inode: the substitute then always has
another number (asserted on the real filesystem below), and closing the pin
keeps SQLite's POSIX locks. macOS has no O_PATH and never reuses a file id.

The reuse assertions are real only on Linux/ext4; on macOS they hold because
APFS never reuses ids. The Linux-only cases need O_PATH itself.
"""
from __future__ import annotations

from contextlib import closing, nullcontext
import fcntl
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap

import aiosqlite
import pytest

from src.analytics.event_logger import EventLogger
from src.dashboard.authority import DashboardAuthority
from src.lists.list_store import ListStore
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.routines.routine_store import RoutineStore
from tests.unit.memory.test_output_registration_families import (
    assert_failed_registry_cleanup, managed,
)

LINUX_PIN = pytest.mark.skipif(base.NATIVE_PIN_FLAGS is None, reason="O_PATH pins exist on Linux only")
_CONNECT = sqlite3.connect  # The substituting party's own connector, never the forbidden seam.


def _sqlite_file(path: Path) -> None:
    with closing(_CONNECT(path)) as database:
        database.execute("CREATE TABLE IF NOT EXISTS existing(value TEXT)")
        database.commit()


def _recreate(path: Path) -> os.stat_result:
    """Same-uid delete and recreate at the path; returns the substitute's lstat."""
    os.unlink(path)
    _sqlite_file(path)
    return os.lstat(path)


def _forbid_connect(monkeypatch) -> list:
    connects = []

    def forbidden(*args, **kwargs):
        connects.append(True)
        raise AssertionError("a recreated destination reached the connector")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(aiosqlite, "connect", forbidden)
    return connects


def _family(config: dict, family: str):
    if family == "privacy_authority":
        store = PrivacyServices(config["memory"]["privacy_db_path"])
        return store, Path(store.ledger.path), store.aclose, lambda: store.acall(store.initialize)
    if family == "authority_sessions":
        store = DashboardAuthority(config["dashboard"]["auth"]["db_path"])
        return store, store.path, store.aclose, store.initialize
    if family == "routines":
        store = RoutineStore(config["routines"]["db_path"])
        return store, Path(store._db_path), store.close, store.initialize
    store = EventLogger(config["analytics"]["events_db"])
    return store, store.db_path, store.aclose, lambda: store.acall(store.initialize)


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["privacy_authority", "authority_sessions", "routines", "events"])
async def test_a_recreated_destination_never_passes_between_attach_and_the_native_open(
        tmp_path, monkeypatch, family):
    body_completed = False
    expected_cleanup = family == "privacy_authority"
    cleanup_context = (pytest.raises(registration.OutputRegistrationError,
        match="^output_registration_registry_unavailable$") if expected_cleanup else nullcontext())
    with cleanup_context as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            store, path, close, initialize = _family(config, family)
            stack.push_async_callback(close)
            _sqlite_file(path)
            original = os.lstat(path)
            await registration.attach_output(store, family)
            substitute = _recreate(path)
            # The pin holds the planned inode, so its number cannot be reused.
            assert (substitute.st_dev, substitute.st_ino) != (original.st_dev, original.st_ino)
            connects = _forbid_connect(monkeypatch)
            with pytest.raises(RuntimeError, match="destination_changed"):
                await initialize()
            assert connects == [] and owner._failed
            body_completed = True
    assert body_completed
    if expected_cleanup:
        assert_failed_registry_cleanup(owner, store, cleanup.value)


@pytest.mark.asyncio
async def test_a_recreated_list_destination_never_passes_between_register_and_open(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        path = Path(config["lists"]["db_path"])
        _sqlite_file(path)
        original = os.lstat(path)
        store = ListStore(str(path))
        stack.push_async_callback(store.close)
        await owner.register_store(store, "lists")
        substitute = _recreate(path)
        assert (substitute.st_dev, substitute.st_ino) != (original.st_dev, original.st_ino)
        connects = _forbid_connect(monkeypatch)
        with pytest.raises(RuntimeError, match="destination_changed"):
            await store.initialize()
        assert connects == []
        owner._failed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [True, False], ids=["existing", "created_by_open"])
async def test_a_recreated_events_destination_never_passes_on_reopen(tmp_path, monkeypatch, existing):
    """Per-operation connections close between calls; the pin outlives them."""
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = EventLogger(config["analytics"]["events_db"])
        stack.push_async_callback(store.aclose)
        if existing:
            _sqlite_file(store.db_path)
        await registration.attach_output(store, "events")
        await store.acall(store.initialize)
        recorded = owner._roles["events"].native_identity
        assert recorded == base._identity(store.db_path)
        await store.acall(store.get_stats)  # An ordinary reopen of the recorded inode passes.
        substitute = _recreate(store.db_path)
        assert {"device": substitute.st_dev, "inode": substitute.st_ino} != recorded
        connects = _forbid_connect(monkeypatch)
        with pytest.raises(RuntimeError, match="destination_changed"):
            await store.acall(store.get_stats)
        assert connects == [] and owner._failed


@pytest.mark.asyncio
async def test_a_recreated_historical_target_is_rejected_before_connect(tmp_path, monkeypatch):
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_storage import event_descriptor
    from src.analytics.retention import EventRetention
    from src.memory.erasure_coordinator import ErasureCoordinator
    from src.memory.erasure_journal import ErasureParticipant
    monkeypatch.setattr(EventRetention, "start", lambda self: None)
    body_completed = False
    with pytest.raises(registration.OutputRegistrationError,
            match="^output_registration_registry_unavailable$") as cleanup:
        async with managed(tmp_path, monkeypatch) as (owner, stack, config):
            privacy = await build_privacy_services(config, shutdown=stack)
            erasures = ErasureCoordinator(privacy)
            stack.push_async_callback(erasures.aclose)
            path = Path(config["analytics"]["events_db"])
            _sqlite_file(path)
            original = os.lstat(path)
            driver = await build_event_erasure(config, privacy=privacy, owner=erasures)
            descriptor = event_descriptor(path)
            snapshot = await privacy.acall(privacy.ledger.erasures.target_snapshot)
            identity = next(e.id for e in snapshot.entries
                            if e.participant is ErasureParticipant.EVENTS and e.descriptor == descriptor)
            actual_worker = driver.retention._work._worker
            substitutes = []

            async def recreated_after_refresh(function, *args, **kwargs):
                substitutes.append(_recreate(path))
                return await actual_worker(function, *args, **kwargs)

            monkeypatch.setattr(driver.retention._work, "_worker", recreated_after_refresh)
            connects = _forbid_connect(monkeypatch)
            outcome = await driver.retention._run({"id": identity, "version": 1, "descriptor": descriptor}, 0)
            assert len(substitutes) == 1
            assert (substitutes[0].st_dev, substitutes[0].st_ino) != (original.st_dev, original.st_ino)
            assert outcome.error is not None and owner._failed
            assert connects == []
            body_completed = True
    assert body_completed
    assert_failed_registry_cleanup(owner, privacy, cleanup.value)


# ----------------------------------------------------------------------
# Linux: the O_PATH pin itself
# ----------------------------------------------------------------------

@LINUX_PIN
@pytest.mark.asyncio
async def test_a_swap_between_the_pin_and_the_plan_is_refused_at_attach(tmp_path, monkeypatch):
    """The plan must observe the pinned inode; a swap after the pin poisons the attach."""
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        store = RoutineStore(config["routines"]["db_path"])
        stack.push_async_callback(store.close)
        path = Path(store._db_path)
        _sqlite_file(path)
        plan = registration._plan

        def swapped(producer, role, supplied, state):
            if producer == "routines":
                _recreate(path)
            return plan(producer, role, supplied, state)

        monkeypatch.setattr(registration, "_plan", swapped)
        with pytest.raises(RuntimeError, match="destination_changed"):
            await registration.attach_output(store, "routines")
        assert owner._failed


@LINUX_PIN
@pytest.mark.asyncio
async def test_every_native_pin_is_an_o_path_descriptor_released_at_epoch_close(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        _sqlite_file(Path(config["routines"]["db_path"]))
        routines = RoutineStore(config["routines"]["db_path"])
        events = EventLogger(config["analytics"]["events_db"])  # Created by its first open.
        stack.push_async_callback(routines.close)
        stack.push_async_callback(events.aclose)
        await registration.attach_output(routines, "routines")
        await registration.attach_output(events, "events")
        await routines.initialize()
        await events.acall(events.initialize)
        pins = dict(owner._native_pins)
        assert {"routines", "events"} <= set(pins)
        for key, path in (("routines", Path(routines._db_path)), ("events", events.db_path)):
            descriptor = pins[key].descriptor
            assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_PATH
            held, named = os.fstat(descriptor), os.lstat(path)
            assert (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)
    assert owner._native_pins == {}
    assert all(pin.descriptor is None for pin in pins.values())


_PROBE = textwrap.dedent("""
    import sqlite3, sys
    database = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
    try:
        database.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as error:
        print("locked" if "locked" in str(error) else "error:" + str(error))
    else:
        database.execute("ROLLBACK")
        print("acquired")
""")


def _other_process_writer(path: Path) -> str:
    result = subprocess.run([sys.executable, "-I", "-c", _PROBE, str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="POSIX record locks")
def test_closing_an_ordinary_extra_descriptor_drops_a_held_sqlite_lock(tmp_path):
    """Why a SQLite destination cannot take an O_RDONLY pin (Linux and macOS alike)."""
    path = tmp_path / "locked.db"
    _sqlite_file(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        assert _other_process_writer(path) == "locked"
        os.close(os.open(path, os.O_RDONLY | os.O_NOFOLLOW))
        assert _other_process_writer(path) == "acquired"
        holder.execute("ROLLBACK")


@LINUX_PIN
def test_closing_an_o_path_pin_keeps_a_held_sqlite_lock(tmp_path):
    path = tmp_path / "locked.db"
    _sqlite_file(path)
    with closing(sqlite3.connect(path, isolation_level=None)) as holder:
        holder.execute("BEGIN IMMEDIATE")
        assert _other_process_writer(path) == "locked"
        pin = base.NativePin()
        pin.open_at(path)
        assert pin.descriptor is not None
        pin.require(base._identity(path))
        pin.release()
        assert _other_process_writer(path) == "locked"
        holder.execute("ROLLBACK")
    assert _other_process_writer(path) == "acquired"


@LINUX_PIN
def test_an_o_path_pin_prevents_inode_number_reuse(tmp_path):
    """Twenty delete-and-recreate rounds at a pinned path never reuse the pinned number."""
    path = tmp_path / "pinned.db"
    for _ in range(20):
        _sqlite_file(path)
        pin = base.NativePin()
        pin.open_at(path)
        original = os.lstat(path)
        substitute = _recreate(path)
        assert (substitute.st_dev, substitute.st_ino) != (original.st_dev, original.st_ino)
        with pytest.raises(RuntimeError, match="destination_changed"):
            pin.require(base._identity(path))
        pin.release()
        os.unlink(path)


@LINUX_PIN
def test_a_pin_never_follows_a_symlink(tmp_path):
    target = tmp_path / "target.db"
    _sqlite_file(target)
    link = tmp_path / "link.db"
    link.symlink_to(target)
    pin = base.NativePin()
    pin.open_at(link)
    with pytest.raises(RuntimeError, match="destination_changed"):
        pin.require(base._identity(target))
    pin.release()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS/APFS contract")
def test_macos_has_no_o_path_and_never_reuses_a_freed_file_id(tmp_path):
    """The reason macOS takes no pin: APFS allocates file ids monotonically."""
    assert not hasattr(os, "O_PATH") and base.NATIVE_PIN_FLAGS is None
    freed = set()
    for _ in range(200):
        path = tmp_path / "probe.db"
        path.write_bytes(b"x")
        info = os.lstat(path)
        assert (info.st_dev, info.st_ino) not in freed
        freed.add((info.st_dev, info.st_ino))
        os.unlink(path)
    pin = base.NativePin()
    pin.open_at(tmp_path)
    assert pin.descriptor is None
    pin.require({"device": 1, "inode": 1})  # Nothing to hold, nothing to refuse.
