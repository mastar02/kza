"""Journal/store seam tests; synthetic admission is not compiled authority proof."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
from pathlib import Path
import threading
from uuid import uuid4

import aiosqlite
import pytest

from src.core import runtime_admission
from src.core.owned_work import WorkCloseTimeout
from src.lists.list_store import ListStore
from src.memory import output_registration as registration
from src.reminders.reminder_store import ReminderStore


@asynccontextmanager
async def coordinator(tmp_path, monkeypatch, *, defaults=False):
    """Mock only running-ticket validation; real journal/store workers run below."""
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    (state / "data").mkdir(parents=True)
    config = {name: {"db_path": str(state / "data" / (name + ".db"))}
              for name in ("lists", "reminders")}
    if defaults:
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "data").symlink_to(state / "data", target_is_directory=True)
        monkeypatch.chdir(runtime)
        config = {}
    ticket = {"admission_root": str(root), "run_id": uuid4().hex,
              "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        str(state / "privacy.db"), runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(registration, "_ACTIVE", set())
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    stack = AsyncExitStack()
    admission.install_on(stack)
    try:
        owner = await admission.begin_destination_registration(config)
        yield owner, stack, config
    finally:
        try:
            await stack.__aexit__(RuntimeError, RuntimeError("unit seam has no graph receipt"), None)
        except RuntimeError as error:
            if str(error) != "output_registration_epoch_incomplete":
                raise


def store_for(producer, config, stack):
    store = {"lists": ListStore, "reminders": ReminderStore}[producer](config[producer]["db_path"])
    stack.push_async_callback(store.close)
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_real_store_lifetime_records_once_without_private_content(tmp_path, monkeypatch, producer):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        await owner.register_store(store, producer)
        assert not Path(store._db_path).exists()
        await store.initialize()
        sentinel = "private-content-that-must-never-enter-output-inventory"
        for _ in range(40):
            if producer == "lists":
                await store.create_list(sentinel, "user", "synthetic-user")
            else:
                await store.create("synthetic-user", sentinel, 2000000000.0)
        assert all(sentinel.encode() not in p.read_bytes() for p in owner._directory.iterdir())
        header, head, states, _ = registration._chain(owner._directory)
        assert head["revision"] == 4 and states[producer]["phase"] == "active"
        assert len(header["producers"]) == 32
        assert states[producer]["plan"]["existing_identity"] is None
        await store.close()
        assert registration._chain(owner._directory)[2][producer]["phase"] == "closed"
        assert registration._chain(owner._directory)[1]["revision"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
@pytest.mark.parametrize("revision", [1, 2])
async def test_intent_or_attachment_publication_failure_prevents_connect(tmp_path, monkeypatch, producer, revision):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        original = registration._head
        connects = []
        def reject(directory, previous, value):
            if value["revision"] == revision:
                raise OSError("injected journal publication fault")
            return original(directory, previous, value)
        monkeypatch.setattr(registration, "_head", reject)
        monkeypatch.setattr(aiosqlite, "connect", lambda *a, **k: connects.append(True))
        with pytest.raises(OSError):
            await owner.register_store(store, producer)
        before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        with pytest.raises(RuntimeError, match="required"):
            await store.initialize()
        assert connects == [] and store._db is None and not Path(store._db_path).exists()
        assert not list(Path(store._db_path).parent.glob(Path(store._db_path).name + "*"))
        assert before == {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        assert store._destination_registration is None
        with pytest.raises(RuntimeError, match="orphan"):
            registration._chain(owner._directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_active_publication_failure_keeps_original_native_owner(tmp_path, monkeypatch, producer):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        await owner.register_store(store, producer)
        original = registration._head
        def reject(directory, previous, value):
            if value["revision"] == 4:
                raise OSError("injected active publication fault")
            return original(directory, previous, value)
        monkeypatch.setattr(registration, "_head", reject)
        with pytest.raises(OSError):
            await store.initialize()
        database = store._db
        assert database is not None and store._work is not None and Path(store._db_path).exists()
        closes = []
        close = database.close
        async def actual_close():
            closes.append(True)
            await close()
        monkeypatch.setattr(database, "close", actual_close)
        with pytest.raises(RuntimeError):
            await store.initialize()
        with pytest.raises(RuntimeError):
            await store._work.write(lambda db: db.execute("CREATE TABLE forbidden(value)"))
        with pytest.raises(RuntimeError, match="epoch_incomplete"):
            await stack.__aexit__(OSError, OSError("active publication failed"), None)
        await store.close()
        assert closes == [True] and store._db is None and store._closing.done()
        with pytest.raises(RuntimeError, match="orphan"):
            registration._chain(owner._directory)


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
@pytest.mark.parametrize("mutation", ["none", "foreign_instance", "path", "none_foreign_no_ticket"])
async def test_managed_registration_cannot_be_omitted_replayed_or_downgraded(tmp_path, monkeypatch, producer, mutation):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        if mutation != "none":
            handle = await owner.register_store(store, producer)
            if mutation == "foreign_instance":
                store = store_for(producer, config, stack)
                store._destination_registration = handle
            else:
                store._db_path = str(tmp_path / "unmanaged-looking.db")
                if mutation == "none_foreign_no_ticket":
                    store._destination_registration = None
                    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
        connects = []
        monkeypatch.setattr(aiosqlite, "connect", lambda *a, **k: connects.append(True))
        with pytest.raises(RuntimeError, match="output_registration_"):
            await store.initialize()
        assert not connects and store._db is None


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_changed_authority_is_rejected_before_native_connect(tmp_path, monkeypatch, producer):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        await owner.register_store(store, producer)
        def stale():
            raise ValueError("runtime_admission_claim_stale")
        monkeypatch.setattr(owner.admission, "revalidate", stale)
        with pytest.raises(ValueError, match="claim_stale"):
            await store.initialize()
        assert store._db is None and not Path(store._db_path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
@pytest.mark.parametrize("observer", ["cancel", "timeout"])
async def test_only_retained_native_close_return_can_publish_closed(tmp_path, monkeypatch, producer, observer):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        await owner.register_store(store, producer)
        await store.initialize()
        entered, release = threading.Event(), threading.Event()
        execute = store._db._execute
        closes = []
        async def held_execute(function, *args, **kwargs):
            if getattr(function, "__name__", None) != "close":
                return await execute(function, *args, **kwargs)
            def held():
                entered.set()
                assert release.wait(5)
                closes.append(True)
                return function(*args, **kwargs)
            return await execute(held)
        monkeypatch.setattr(store._db, "_execute", held_execute)
        task = asyncio.create_task(store.close())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            retained = store._closing
            assert retained is not None and not retained.done()
            if observer == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(WorkCloseTimeout):
                    await task
            assert registration._chain(owner._directory)[2][producer]["phase"] == "active"
            assert not closes and not retained.done()
            release.set()
            await retained
            assert store._closing is retained and closes == [True]
            assert registration._chain(owner._directory)[2][producer]["phase"] == "closed"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_unmanaged_default_stores_remain_usable_without_certification(tmp_path, monkeypatch, producer):
    monkeypatch.setattr(registration, "_ACTIVE", set())
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    store = {"lists": ListStore, "reminders": ReminderStore}[producer](str(tmp_path / "local.db"))
    try:
        await store.initialize()
        assert store._destination_registration is None
    finally:
        await store.close()
    assert not (tmp_path / "admission").exists()


@pytest.mark.asyncio
async def test_existing_database_is_observed_not_claimed_created(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        path = Path(config["lists"]["db_path"])
        path.touch()
        identity = registration._identity(path)
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        plan = registration._chain(owner._directory)[2]["lists"]["plan"]
        assert plan["existing_identity"] == identity and plan["before_first_open"] is True
        await store.initialize()
        await store.close()


@pytest.mark.asyncio
async def test_capacity_refusal_leaves_destination_unopened(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        monkeypatch.setattr(registration, "MAX_EVENTS", 0)
        with pytest.raises(RuntimeError, match="capacity"):
            await owner.register_store(store, "lists")
        assert not Path(store._db_path).exists() and store._db is None


def test_record_size_limit_is_shared_by_writer_and_reader(tmp_path):
    overhead = len(registration._encoded({"value": ""}))
    value = {"value": "x" * (registration.MAX_RECORD_BYTES - overhead)}
    raw = registration._encoded(value)
    assert len(raw) == registration.MAX_RECORD_BYTES and raw.endswith(b"\n")
    path = tmp_path / "boundary.json"
    path.write_bytes(raw)
    assert registration._read(path)[0] == value
    with pytest.raises(RuntimeError, match="record_size"):
        registration._encoded({"value": value["value"] + "x"})
    path.write_bytes(raw + b" ")
    with pytest.raises(RuntimeError, match="record_file"):
        registration._read(path)


def test_metadata_and_forged_controller_do_not_grant_inventory_access():
    with pytest.raises((RuntimeError, ValueError, AttributeError)):
        registration.read_output_inventory({"root": "/tmp", "status": "closed"})


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["lists", "reminders"])
async def test_lifecycle_events_require_the_actual_store_task(tmp_path, monkeypatch, producer):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for(producer, config, stack)
        handle = await owner.register_store(store, producer)
        with pytest.raises(RuntimeError, match="initialization_owner"):
            await owner.active(handle, store)
        await store.initialize()
        with pytest.raises(RuntimeError, match="close_owner"):
            await registration.store_close_returned(store)
        assert registration._chain(owner._directory)[2][producer]["phase"] == "active"
        assert await store._work.read(lambda db: db.execute("SELECT 1").fetchone())


@pytest.mark.asyncio
async def test_closed_managed_context_cannot_be_downgraded(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        await store.initialize()
        await store.close()
        await owner.aclose()
        monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
        foreign = ListStore(str(tmp_path / "outside.db"))
        with pytest.raises(RuntimeError, match="required"):
            await foreign.initialize()
        assert not Path(foreign._db_path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", ["MAX_TOTAL_BYTES", "MAX_RUN_BYTES", "MAX_NAMESPACE_ENTRIES"])
async def test_capacity_failure_preserves_existing_native_owner(tmp_path, monkeypatch, bound):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        await store.initialize()
        before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        database = store._db
        closes = []
        native_close = database.close
        async def actual_close():
            closes.append(True)
            await native_close()
        monkeypatch.setattr(database, "close", actual_close)
        monkeypatch.setattr(registration, bound, 0)
        other = store_for("reminders", config, stack)
        with pytest.raises(RuntimeError):
            await owner.register_store(other, "reminders")
        assert other._db is None and not Path(other._db_path).exists()
        assert store._db is database and database._connection is not None and closes == []
        assert before == {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        await store.close()
        assert store._db is None and closes == [True]


@pytest.mark.asyncio
async def test_epoch_replacement_rejects_open_even_with_same_content(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        path = owner._directory / "epoch.json"
        replacement = owner._directory / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with pytest.raises(RuntimeError, match="epoch_changed"):
            await store.initialize()
        assert store._db is None and not Path(store._db_path).exists()


@pytest.mark.asyncio
async def test_startup_capacity_failure_stays_managed_without_creating_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(registration, "MAX_TOTAL_BYTES", 0)
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    with pytest.raises(RuntimeError, match="capacity"):
        async with coordinator(tmp_path, monkeypatch):
            raise AssertionError("Startup unexpectedly succeeded")
    assert registration._ACTIVE
    assert not (tmp_path / "admission-root/admission/output-inventory").exists()
    store = ListStore(str(tmp_path / "outside.db"))
    with pytest.raises(RuntimeError, match="required"):
        await store.initialize()
    assert store._db is None and not Path(store._db_path).exists()


@pytest.mark.asyncio
async def test_installed_provenance_rejects_downgrade_before_begin_returns(tmp_path, monkeypatch):
    # The earlier begin revalidation is the distinct fault: no coordinator
    # namespace or _ACTIVE membership may be needed to detect managed use.
    async def fail_before_coordinator(admission, config):
        assert runtime_admission._verified_admissions.get(admission) is True
        raise RuntimeError("injected_before_coordinator")
    monkeypatch.setattr(runtime_admission.RuntimeAdmission, "begin_destination_registration", fail_before_coordinator)
    with pytest.raises(RuntimeError, match="injected_before_coordinator"):
        async with coordinator(tmp_path, monkeypatch):
            raise AssertionError("Unexpected successful begin")
    assert not registration._ACTIVE and any(runtime_admission._verified_admissions.values())
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    store = ReminderStore(str(tmp_path / "outside.db"))
    with pytest.raises(RuntimeError, match="required"):
        await store.initialize()
    assert store._db is None and not Path(store._db_path).exists()


@pytest.mark.asyncio
async def test_pending_work_capacity_keeps_real_owner_until_shutdown(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        await store.initialize()
        before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        assert owner._disk._capacity == 64
        monkeypatch.setattr(owner._disk, "_capacity", 1)
        entered, release = threading.Event(), threading.Event()
        def held():
            entered.set()
            assert release.wait(5)
        pending = asyncio.create_task(owner._run(held))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            other = store_for("reminders", config, stack)
            with pytest.raises(RuntimeError, match="capacity"):
                await owner.register_store(other, "reminders")
            assert other._db is None and store._db is not None
            assert before == {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        finally:
            release.set()
            await pending
        await store.close()
        assert store._db is None


@pytest.mark.asyncio
async def test_event_reserves_atomic_temporary_entry_before_mutation(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        store = store_for("lists", config, stack)
        await owner.register_store(store, "lists")
        before = {p.name: p.read_bytes() for p in owner._directory.iterdir()}
        count = len(list(owner._directory.parent.rglob("*")))
        monkeypatch.setattr(registration, "MAX_NAMESPACE_ENTRIES", count + 1)
        with pytest.raises(RuntimeError, match="capacity"):
            await store.initialize()
        assert store._db is None and not Path(store._db_path).exists()
        assert before == {p.name: p.read_bytes() for p in owner._directory.iterdir()}


@pytest.mark.asyncio
async def test_managed_builder_defaults_pin_physical_state_through_runtime_link(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch, defaults=True) as (owner, stack, config):
        for producer, store_type in (("lists", ListStore), ("reminders", ReminderStore)):
            store = store_type(f"./data/{producer}.db")
            stack.push_async_callback(store.close)
            await owner.register_store(store, producer)
            await store.initialize()
            state = registration._chain(owner._directory)[2][producer]
            assert state["plan"]["path"] == str(tmp_path / "state/data" / (producer + ".db"))
            assert state["native_identity"] == registration._identity(Path(state["plan"]["path"]))
            await store.close()


@pytest.mark.asyncio
async def test_cloned_admission_cannot_begin_registration(tmp_path, monkeypatch):
    async with coordinator(tmp_path, monkeypatch) as (owner, stack, config):
        original = owner.admission
        clone = runtime_admission.RuntimeAdmission(original._ticket, original._binding,
            original._privacy_path, original._config_digest)
        # Copied attributes and nominal type do not create factory provenance.
        clone._installed = True
        clone._shutdown = stack
        monkeypatch.setattr(clone, "revalidate", lambda: None)
        with pytest.raises(RuntimeError, match="actual_admission"):
            await registration.begin_registration(clone, config, stack)
        assert clone._output_registration is None
