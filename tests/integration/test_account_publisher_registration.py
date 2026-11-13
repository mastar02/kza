"""Managed account publisher through original dashboard routes and owners.

Like the family registration tests, admission validation and the verified
instance registry are synthetic seams. Native execution provenance, HTTP
routes, session guards, UserManager publication, credential retirement and the
original DashboardAPI.stop drain are real. Compiled ancestry is separate.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
import io
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from uuid import uuid4
import wave

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.core import runtime_admission
from src.dashboard import admin as admin_routes
from src.dashboard import native_work
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from src.users import user_manager as catalog
from src.users.user_manager import PermissionLevel, UserManager
from tests.mocks.inode_reuse import report_inode_reuse

PRIVATE = ("Synthetic Alba", "Synthetic Bruno", "Synthetic Carla")


def _voice_links():
    """Managed review policy only; credential retirement stays the real authority."""
    return SimpleNamespace(ledger=SimpleNamespace(people=SimpleNamespace(
        user_links_cleared=lambda identities: True)))


@asynccontextmanager
async def account_graph(tmp_path, monkeypatch, token, *, attach=True, speaker=None):
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    data = state / "data"
    data.mkdir(parents=True)
    config = {
        "memory": {"privacy_db_path": str(data / "privacy.db")},
        "dashboard": {"auth": {"db_path": str(data / "authority.db")}},
        "routines": {"db_path": str(data / "routines.db")},
        "analytics": {"events_db": str(data / "events.db"), "suggestions_path": str(data / "suggestions.json")},
        "speaker_id": {"users_path": str(data / "users.json")},
        **{name: {"db_path": str(data / (name + ".db"))} for name in ("lists", "reminders")},
    }
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    owner = await admission.begin_destination_registration(config)
    graph = SimpleNamespace(owner=owner, stack=stack, config=config, data=data, closed=False)
    try:
        users = None
        authority = DashboardAuthority(config["dashboard"]["auth"]["db_path"],
                                       account_provider=lambda: users.active_accounts())
        stack.push_async_callback(authority.aclose)
        await registration.attach_output(authority, "authority_sessions")
        users = UserManager(config["speaker_id"]["users_path"], voice_registry=_voice_links(),
                            retirement_proof=authority.users_retired)
        await authority.initialize()
        api = DashboardAPI(authority=authority, user_manager=users, speaker_identifier=speaker,
                           speaker_model_lock=None if speaker is None else threading.Lock())
        stack.push_async_callback(api.stop)
        if attach:
            await owner.attach_accounts(users, api)
        graph.__dict__.update(api=api, users=users, authority=authority)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://testserver") as client:
            assert (await client.post("/api/admin/auth/login", json={"token": token})).status_code == 200
            graph.client = client
            yield graph
    finally:
        await close(graph)


async def close(graph):
    if graph.closed:
        return
    graph.closed = True
    try:
        await graph.stack.__aexit__(RuntimeError, RuntimeError("unit has no compiled graph receipt"), None)
    except RuntimeError as error:
        if str(error) != "output_registration_epoch_incomplete":
            raise


def token(dashboard_auth):
    return dashboard_auth["Authorization"].split(" ", 1)[1]


def account_events(owner):
    return [event for event in (base._read(p)[0] for p in sorted(owner._directory.glob("[0-9]*.json")))
            if event["kind"] == "account_publisher"]


def journal_bytes(owner):
    return b"".join(p.read_bytes() for p in sorted(owner._directory.iterdir()))


def identity(path):
    info = os.lstat(path)
    return {"device": info.st_dev, "inode": info.st_ino}


def temporaries(graph):
    return sorted(p.name for p in graph.data.glob(".accounts-*"))


async def create(graph, name, level="guest"):
    return await graph.client.post("/api/admin/users", json={"name": name, "permission_level": level})


def assert_poisoned_close(graph):
    owner = graph.owner
    assert owner._failed
    assert graph.api._stop_task is not None and graph.api._stop_task.done()
    work = graph.api.admin_work.status()
    assert work["closed"] and work["operations"] == 0
    assert owner._closing.done() and str(owner._closing.exception()) == "output_registration_epoch_incomplete"
    assert [event["phase"] for event in account_events(owner)][-1] != "drained"


# A-T01, A-T09, A-T12, A-ID01
async def test_authenticated_create_update_delete_publish_registered_file_after_opening(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        owner, users = graph.owner, graph.users
        path = Path(users.data_path)
        assert owner._directory.parents[1].name == "v4"
        assert base._chain(owner._directory)[0]["selected"]["accounts"] is True
        plan = account_events(owner)[0]["plan"]
        assert [event["phase"] for event in account_events(owner)] == ["intent", "attached"]
        assert plan["path"] == str(path.resolve()) and plan["initial_identity"] is None
        assert plan["creation_observation"] == "planned_absent" and plan["before_first_publish"] is True
        assert plan["temporary_scope"] == {"directory": str(path.parent.resolve()), "prefix": ".accounts-",
                                           "allocator": "mkstemp"}
        allocations, descriptors = [], []
        original_mkstemp, original_fdopen = catalog.tempfile.mkstemp, catalog.FileIO

        def mkstemp(*args, **kwargs):
            phases = [event["phase"] for event in account_events(owner)]
            allocations.append((phases[-1], original_mkstemp(*args, **kwargs)))
            return allocations[-1][1]

        def fdopen(descriptor, *args, **kwargs):
            if len(descriptors) < len(allocations) and descriptor == allocations[-1][1][0]:
                # Independent observation of the real allocated tempfile FD.
                info = os.fstat(descriptor)
                descriptors.append({"device": info.st_dev, "inode": info.st_ino})
            return original_fdopen(descriptor, *args, **kwargs)

        monkeypatch.setattr(catalog.tempfile, "mkstemp", mkstemp)
        monkeypatch.setattr(catalog, "FileIO", fdopen)
        response = await create(graph, PRIVATE[0])
        assert response.status_code == 200, response.text
        created = response.json()["user_id"]
        assert [phase for phase, _ in allocations] == ["opening"]
        assert identity(path) == descriptors[-1] == users._output_coordinator._accounts.expected_identity
        revision = base._chain(owner._directory)[1]["revision"]
        for name in PRIVATE[1:]:
            assert (await create(graph, name, "adult")).status_code == 200
            assert identity(path) == descriptors[-1]
        update = await graph.client.put(f"/api/admin/users/{created}/permissions", json={"permission_level": "teen"})
        assert update.status_code == 200
        assert identity(path) == descriptors[-1] and len(descriptors) == 4
        # Public replacements advance only the own generation; no per-save journal event.
        assert base._chain(owner._directory)[1]["revision"] == revision
        deleted = await graph.client.delete(f"/api/admin/users/{created}")
        assert deleted.status_code == 200, deleted.text
        assert created not in users._users and identity(path) == descriptors[-1]
        on_disk = UserManager(str(graph.data / "users.json"), voice_registry=_voice_links())
        assert set(on_disk._users) == set(users._users) and len(users._users) == 2
        assert temporaries(graph) == []
        publisher = owner._accounts
        # Payload observations end with their real callbacks; only markers remain.
        assert publisher.invocations == {} and publisher.steps == {} and publisher.active is None
        assert all(entry.function is None and entry.args is None for entry in publisher.sequences.values())
        await graph.api.stop()
        events = account_events(owner)
        assert [event["phase"] for event in events] == ["intent", "attached", "opening", "drained"]
        assert events[-1]["aggregate"] == {
            "attempts": 5, "committed_replacements": 5, "clean_failed_attempts": 0,
            "last_committed_identity": descriptors[-1], "native_drain_returned": True,
            "open_temp_count": 0, "cleanup_uncertain": False}
        raw = journal_bytes(owner)
        for private in (*PRIVATE, "teen", "adult", created, "permission_level", "preferences"):
            assert private.encode() not in raw
        await close(graph)
        assert owner._closing.done() and owner._closing.exception() is None
        assert base._chain(owner._directory)[2]["accounts"]["phase"] == "drained"


# A-T02
@pytest.mark.parametrize("form", ["recursive_unrelated_entry", "direct_foreign_callback", "wrong_manager"])
async def test_unobserved_or_recursive_admin_entry_cannot_mint_account_publication(
    tmp_path, monkeypatch, dashboard_auth, form,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        work, users = graph.api.admin_work, graph.users
        mkstemp_calls = []
        monkeypatch.setattr(catalog.tempfile, "mkstemp",
                            lambda *a, **k: mkstemp_calls.append(a) or (_ for _ in ()).throw(AssertionError("write")))
        kwargs = {"name": PRIVATE[0], "permission_level": PermissionLevel.GUEST}
        with pytest.raises(registration.OutputRegistrationError):
            if form == "recursive_unrelated_entry":
                # The outer entry is consumed although its callback is not a mutator.
                await native_work.run_admin_work(work, lambda: True, native_work._invoke_admin,
                                                 lambda: True, users.add_user, (), kwargs)
            elif form == "direct_foreign_callback":
                await work.run(users.add_user, **kwargs)
            else:
                other = UserManager(str(graph.data / "users.json"), voice_registry=_voice_links())
                await native_work.run_admin_work(work, lambda: True, other.add_user, **kwargs)
        assert mkstemp_calls == [] and not Path(users.data_path).exists() and temporaries(graph) == []
        assert users._users == {}
        await close(graph)
        assert_poisoned_close(graph)


# A-T03
async def test_foreign_sequence_cannot_enter_original_delete_adapter(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        assert (await create(graph, PRIVATE[0])).status_code == 200
        work, users = graph.api.admin_work, graph.users
        before = Path(users.data_path).read_bytes()
        user_id = next(iter(users._users))
        validator = partial(native_work.require_current, lambda: True)

        async def foreign():
            return await native_work._invoke_admin_sequence(lambda: True, admin_routes._delete_account, (
                work, lambda: True, graph.authority, None, users, user_id, None), {})

        with pytest.raises(registration.OutputRegistrationError):
            await work.run_async(foreign)
        with pytest.raises(registration.OutputRegistrationError):
            registration.account_delete_step(work, users, user_id, validator)
        assert Path(users.data_path).read_bytes() == before and user_id in users._users
        await close(graph)
        assert_poisoned_close(graph)


async def test_delete_argument_substitution_is_refused_before_catalog_write(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        for name in PRIVATE[:2]:
            assert (await create(graph, name)).status_code == 200
        users, work = graph.users, graph.api.admin_work
        target, other = list(users._users)
        before = Path(users.data_path).read_bytes()
        original = work.native_step

        async def substituted(function, *args, **kwargs):
            if getattr(function, "__func__", None) is UserManager.remove_user:
                args = (other,) + args[1:]
            return await original(function, *args, **kwargs)

        monkeypatch.setattr(work, "native_step", substituted)
        response = await graph.client.delete(f"/api/admin/users/{target}")
        assert response.status_code == 503
        assert Path(users.data_path).read_bytes() == before and set(users._users) == {target, other}
        await close(graph)
        assert_poisoned_close(graph)


# A-T03 positive: original sequence survives begin_close and observer cancellation.
async def test_original_delete_sequence_completes_after_begin_close_and_observer_cancel(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        assert (await create(graph, PRIVATE[0])).status_code == 200
        users, authority = graph.users, graph.authority
        user_id = next(iter(users._users))
        entered, release = asyncio.Event(), asyncio.Event()
        original = authority.retire_user

        async def retire(actor, identity):
            entered.set()
            await release.wait()
            return await original(actor, identity)

        monkeypatch.setattr(authority, "retire_user", retire)
        request = asyncio.create_task(graph.client.delete(f"/api/admin/users/{user_id}"))
        await asyncio.wait_for(entered.wait(), 5)
        drain = graph.api.admin_work.begin_close()
        for _ in range(2):
            request.cancel()
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(request, return_exceptions=True)
        await asyncio.wait_for(asyncio.shield(drain), 5)
        assert user_id not in users._users
        assert user_id not in UserManager(str(graph.data / "users.json"), voice_registry=_voice_links())._users
        await graph.api.stop()
        assert account_events(graph.owner)[-1]["aggregate"]["committed_replacements"] == 2
        await close(graph)
        assert graph.owner._closing.exception() is None


# A-T05
async def test_attachment_failure_refuses_first_save_while_original_stop_drains(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth), attach=False) as graph:
        original = base._head

        def fail(directory, previous, value):
            if any(event["phase"] == "intent" for event in account_events(graph.owner)):
                raise OSError("synthetic attached fault")
            return original(directory, previous, value)

        monkeypatch.setattr(base, "_head", fail)
        with pytest.raises(OSError):
            await graph.owner.attach_accounts(graph.users, graph.api)
        monkeypatch.setattr(base, "_head", original)
        mkdirs = []
        monkeypatch.setattr(catalog.tempfile, "mkstemp", lambda *a, **k: mkdirs.append(a))
        assert (await create(graph, PRIVATE[0])).status_code == 503
        assert mkdirs == [] and not Path(graph.users.data_path).exists()
        await close(graph)
        assert_poisoned_close(graph)


@pytest.mark.parametrize("change", ["removed_handle", "moved_path", "hardlink", "replaced_inode", "symlink",
                                    "rewritten_at_name", "rewritten_with_reused_inode_number"])
async def test_handle_path_or_external_identity_change_stays_managed_and_rejects(
    tmp_path, monkeypatch, dashboard_auth, change,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        assert (await create(graph, PRIVATE[0])).status_code == 200
        users = graph.users
        path = Path(users.data_path)
        expected = identity(path)
        if change == "removed_handle":
            del users._output_coordinator
            response = await graph.client.post("/api/admin/users", json={"name": PRIVATE[1]})
            assert response.status_code == 200  # Global installed coordinator still observes.
            assert identity(path) != expected
            await close(graph)
            assert graph.owner._closing.exception() is None
            return
        if change == "moved_path":
            users.data_path = graph.data / "moved_accounts_v1.json"
        elif change == "hardlink":
            os.link(path, graph.data / "external-link")
        elif change == "replaced_inode":
            copy = graph.data / "external-copy"
            copy.write_bytes(path.read_bytes())
            os.replace(copy, path)
        elif change.startswith("rewritten"):
            # Same bytes, unlinked and written again at the name: on ext4 the number is reused.
            original, raw = os.lstat(path), path.read_bytes()
            path.unlink()
            path.write_bytes(raw)
            if change == "rewritten_with_reused_inode_number":
                report_inode_reuse(monkeypatch, os.lstat(path), original)
        else:
            outside = graph.data / "outside.json"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
        before = path.read_bytes()
        calls = []
        monkeypatch.setattr(catalog.tempfile, "mkstemp", lambda *a, **k: calls.append(a))
        assert (await create(graph, PRIVATE[1])).status_code == 503
        assert calls == [] and path.read_bytes() == before and len(users._users) == 1
        assert not (graph.data / "moved_accounts_v1.json").exists()
        await close(graph)
        assert_poisoned_close(graph)
        assert graph.owner._accounts.pin.descriptor is None  # Released by the epoch's close.


# A-T06
async def test_revoked_lease_during_real_temp_write_cleans_attempt_and_keeps_file_and_ram(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        assert (await create(graph, PRIVATE[0])).status_code == 200
        users, path = graph.users, Path(graph.users.data_path)
        before, members = path.read_bytes(), dict(users._users)
        entered, release = threading.Event(), threading.Event()
        original = catalog.json.dump

        def dump(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(catalog.json, "dump", dump)
        pending = asyncio.create_task(create(graph, PRIVATE[1]))
        assert await asyncio.to_thread(entered.wait, 5)
        assert len(temporaries(graph)) == 1
        assert (await graph.client.post("/api/admin/auth/logout")).status_code == 200
        pending.cancel()
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await graph.api.admin_work.drain_admitted()
        assert path.read_bytes() == before and users._users.keys() == members.keys()
        assert temporaries(graph) == []
        await graph.api.stop()
        aggregate = account_events(graph.owner)[-1]["aggregate"]
        assert (aggregate["attempts"], aggregate["committed_replacements"], aggregate["clean_failed_attempts"]) == (2, 1, 1)
        await close(graph)
        assert graph.owner._closing.exception() is None


# A-T07 / A-ID04 managed variants
@pytest.mark.parametrize("fault", ["fdopen", "fdopen_canary", "wrapper", "inner_fstat"])
async def test_managed_descriptor_faults_close_once_and_never_claim_clean_uncertainty(
    tmp_path, monkeypatch, dashboard_auth, fault,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        allocated, closes, calls = [], [], {"fstat": 0}
        original_mkstemp, original_close, original_fstat = catalog.tempfile.mkstemp, catalog.os.close, catalog.os.fstat
        original_fdopen = catalog.FileIO

        def mkstemp(*args, **kwargs):
            allocated.append(original_mkstemp(*args, **kwargs))
            return allocated[-1]

        # One allocation only: later journal descriptors may reuse the number.
        def close_descriptor(descriptor):
            if allocated and descriptor == allocated[-1][0] and not closes:
                closes.append(descriptor)
            return original_close(descriptor)

        def fdopen(descriptor, *args, **kwargs):
            if not allocated or descriptor != allocated[-1][0] or calls.get("fdopen"):
                return original_fdopen(descriptor, *args, **kwargs)
            calls["fdopen"] = 1
            if fault == "fdopen_canary":
                canary = graph.data / "canary"
                canary.write_bytes(b"canary")
                os.replace(canary, allocated[-1][1])
            raise OSError("synthetic fdopen failure")

        def wrapper(*args, **kwargs):
            raise OSError("synthetic wrapper failure")

        def fstat(descriptor):
            if allocated and descriptor == allocated[-1][0]:
                calls["fstat"] += 1
            if fault == "inner_fstat" and allocated and descriptor == allocated[-1][0] and calls["fstat"] == 1:
                raise OSError("synthetic fstat failure")
            return original_fstat(descriptor)

        monkeypatch.setattr(catalog.tempfile, "mkstemp", mkstemp)
        monkeypatch.setattr(catalog.os, "close", close_descriptor)
        monkeypatch.setattr(catalog.os, "fstat", fstat)
        if fault in {"fdopen", "fdopen_canary"}:
            # First transfer stage: FileIO construction over the raw descriptor.
            monkeypatch.setattr(catalog, "FileIO", fdopen)
        elif fault == "wrapper":
            # Second stage: FileIO owns the descriptor; wrapper construction fails.
            monkeypatch.setattr(catalog, "TextIOWrapper", wrapper)
        assert (await create(graph, PRIVATE[0])).status_code == 503
        descriptor, temporary = allocated[-1]
        with pytest.raises(OSError):
            os.fstat(descriptor)  # The original allocation is not leaked.
        assert not Path(graph.users.data_path).exists() and graph.users._users == {}
        if fault in {"fdopen", "wrapper"}:
            assert closes == ([descriptor] if fault == "fdopen" else []) and not Path(temporary).exists()
            await graph.api.stop()
            aggregate = account_events(graph.owner)[-1]["aggregate"]
            assert (aggregate["attempts"], aggregate["clean_failed_attempts"]) == (1, 1)
            await close(graph)
            assert graph.owner._closing.exception() is None
            return
        if fault == "fdopen_canary":
            assert closes == [descriptor] and Path(temporary).read_bytes() == b"canary"
        else:
            # Transferred: only the original file object closes the descriptor.
            assert closes == [] and Path(temporary).exists()
        await close(graph)
        assert_poisoned_close(graph)


# A-T08 / A-ID02
async def test_destination_substituted_after_replace_keeps_approved_ram_and_poisons(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        users = graph.users
        path = Path(users.data_path)
        original = catalog.os.replace
        replaced = []

        def replace(source, destination):
            original(source, destination)
            if Path(destination) == path:
                replaced.append(destination)
                canary = graph.data / "canary"
                canary.write_bytes(b"canary-account-file")
                original(canary, destination)

        monkeypatch.setattr(catalog.os, "replace", replace)
        assert (await create(graph, PRIVATE[0])).status_code == 503
        assert len(replaced) == 1 and path.read_bytes() == b"canary-account-file"
        assert [user.name for user in users._users.values()] == [PRIVATE[0]] and users._version == 1
        assert graph.owner._accounts.expected_identity is None
        await close(graph)
        assert_poisoned_close(graph)


# A-ID03
async def test_temp_name_substituted_before_final_gate_never_replaces_or_deletes_canary(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        users = graph.users
        replaced, allocated = [], []
        original_replace, original_mkstemp, original_dump = catalog.os.replace, catalog.tempfile.mkstemp, catalog.json.dump
        monkeypatch.setattr(catalog.tempfile, "mkstemp",
                            lambda *a, **k: allocated.append(original_mkstemp(*a, **k)) or allocated[-1])

        def dump(*args, **kwargs):
            original_dump(*args, **kwargs)
            canary = graph.data / "canary"
            canary.write_bytes(b"temp-canary")
            original_replace(canary, allocated[-1][1])

        monkeypatch.setattr(catalog.json, "dump", dump)
        path = Path(users.data_path)

        def replace(source, destination):
            if Path(destination) == path:
                replaced.append(source)
            return original_replace(source, destination)

        monkeypatch.setattr(catalog.os, "replace", replace)
        assert (await create(graph, PRIVATE[0])).status_code == 503
        assert replaced == [] and not Path(users.data_path).exists() and users._users == {}
        assert Path(allocated[-1][1]).read_bytes() == b"temp-canary"
        with pytest.raises(OSError):
            os.fstat(allocated[-1][0])
        await close(graph)
        assert_poisoned_close(graph)


# A-T10
async def test_stop_owner_survives_repeated_cancel_until_publisher_drain_returns(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        entered, release = threading.Event(), threading.Event()
        original = catalog.json.dump

        def dump(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(catalog.json, "dump", dump)
        pending = asyncio.create_task(create(graph, PRIVATE[0]))
        assert await asyncio.to_thread(entered.wait, 5)
        stopping = asyncio.create_task(graph.api.stop())
        await asyncio.sleep(0)
        for _ in range(2):
            stopping.cancel()
            await asyncio.sleep(0)
        assert not graph.api._stop_task.done()
        assert [event["phase"] for event in account_events(graph.owner)][-1] == "opening"
        release.set()
        assert (await pending).status_code == 200
        with pytest.raises(asyncio.CancelledError):
            await stopping
        await asyncio.shield(graph.api._stop_task)
        events = account_events(graph.owner)
        assert events[-1]["phase"] == "drained" and events[-1]["aggregate"]["committed_replacements"] == 1
        await close(graph)
        assert graph.owner._closing.exception() is None


async def test_eager_task_factory_keeps_original_stop_task_identity(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        loop = asyncio.get_running_loop()
        previous = loop.get_task_factory()
        loop.set_task_factory(asyncio.eager_task_factory)
        try:
            await graph.api.stop()
        finally:
            loop.set_task_factory(previous)
        assert account_events(graph.owner)[-1]["phase"] == "drained"
        assert account_events(graph.owner)[-1]["aggregate"]["attempts"] == 0
        await close(graph)
        assert graph.owner._closing.exception() is None


# Census oracles: read-only construction and RAM-only last_seen stay outside the gate.
async def test_read_only_construction_and_last_seen_never_touch_account_disk_or_journal(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        assert (await create(graph, PRIVATE[0])).status_code == 200
        users = graph.users
        before, journal = Path(users.data_path).read_bytes(), journal_bytes(graph.owner)
        user_id = next(iter(users._users))
        snapshot = users.active_accounts()
        await asyncio.to_thread(users.update_last_seen, user_id)
        reader = UserManager(str(graph.data / "users.json"), voice_registry=_voice_links())
        assert set(reader._users) == {user_id} and users.active_accounts() is snapshot
        assert users.set_voice_embedding(user_id, object()) is False
        assert Path(users.data_path).read_bytes() == before and journal_bytes(graph.owner) == journal
        assert not graph.owner._failed


def _wav(seconds=1.0, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x01\x00" * int(seconds * rate))
    return buffer.getvalue()


# Legacy voice enrollment is outside account_admin_entry's admitted functions
# (add_user/update_user). With V3 accounts attached, the managed review guard
# must refuse the route before any native work or catalog publication.
async def test_legacy_enrollment_route_is_refused_before_saving_with_accounts_attached(
    tmp_path, monkeypatch, dashboard_auth,
):
    speaker = SimpleNamespace(calls=[], create_enrollment_embedding=lambda arrays: (
        speaker.calls.append(len(arrays)), np.ones(192, dtype=np.float32))[1])
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth), speaker=speaker) as graph:
        owner, users = graph.owner, graph.users
        assert users.voice_review_managed is True and owner._accounts.manager is users
        assert (await create(graph, PRIVATE[0])).status_code == 200
        user_id = next(iter(users._users))
        path = Path(users.data_path)
        before, journal, ram = path.read_bytes(), journal_bytes(owner), dict(users._users)
        events = account_events(owner)
        calls, original = [], users.set_voice_embedding

        def spy(*args, **kwargs):
            calls.append(args[0] if args else kwargs.get("user_id"))
            return original(*args, **kwargs)

        monkeypatch.setattr(users, "set_voice_embedding", spy)
        files = [("samples", (f"s{index}.wav", _wav(), "audio/wav")) for index in range(3)]
        response = await graph.client.post(f"/api/admin/users/{user_id}/enroll", files=files)
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "voice_profile_review_required"
        assert calls == [] and speaker.calls == []
        assert graph.api.admin_work.status()["operations"] == 0
        assert path.read_bytes() == before and users._users == ram
        assert journal_bytes(owner) == journal and account_events(owner) == events
        assert not owner._failed and temporaries(graph) == []
        await graph.api.stop()
        await close(graph)
        assert owner._closing.done() and owner._closing.exception() is None
        assert [event["phase"] for event in account_events(owner)] == ["intent", "attached", "opening", "drained"]
        assert base._chain(owner._directory)[2]["accounts"]["phase"] == "drained"


# The native catalog edit itself refuses managed voice publication: inside the
# real admin work, set_voice_embedding changes nothing and never saves.
async def test_managed_set_voice_embedding_inside_admin_work_returns_false_without_write(
    tmp_path, monkeypatch, dashboard_auth,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        owner, users = graph.owner, graph.users
        assert (await create(graph, PRIVATE[0])).status_code == 200
        user_id = next(iter(users._users))
        path = Path(users.data_path)
        before, journal, ram = path.read_bytes(), journal_bytes(owner), dict(users._users)
        events, saves, original = account_events(owner), [], users._save

        def save(*args, **kwargs):
            saves.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(users, "_save", save)
        changed = await native_work.run_admin_work(graph.api.admin_work, lambda: True, users.set_voice_embedding,
                                                   user_id, np.ones(192, dtype=np.float32))
        assert changed is False and saves == []
        assert path.read_bytes() == before and users._users == ram
        assert users._users[user_id].voice_embedding is None
        assert journal_bytes(owner) == journal and account_events(owner) == events
        assert not owner._failed and temporaries(graph) == []
        await graph.api.stop()
        await close(graph)
        assert owner._closing.done() and owner._closing.exception() is None
        assert base._chain(owner._directory)[2]["accounts"]["phase"] == "drained"


# Graph re-pin: the composed ApplicationGraph exposes the publisher only through
# its real ``dashboard`` field (it has no user_manager); the pin reads that field.
@pytest.mark.parametrize("shape", ["same_dashboard", "foreign_dashboard", "no_dashboard", "swapped_manager"])
async def test_composition_repins_graph_dashboard_to_attached_publisher(
    tmp_path, monkeypatch, dashboard_auth, shape,
):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        users = graph.users
        dashboard = {"same_dashboard": graph.api, "swapped_manager": graph.api, "no_dashboard": None,
                     "foreign_dashboard": DashboardAPI(user_manager=users)}[shape]
        if shape == "swapped_manager":
            monkeypatch.setattr(graph.api, "user_manager",
                                UserManager(str(graph.data / "users.json"), voice_registry=_voice_links()))
        public = SimpleNamespace(dashboard=dashboard)
        expected = ("output_registration_composition_incomplete" if shape == "same_dashboard"
                    else "output_registration_account_owner")
        # Other families are not composed in this fixture; only the pin is observed.
        with pytest.raises(registration.OutputRegistrationError) as error:
            await registration.composed(graph.owner, public)
        assert str(error.value) == expected and graph.owner._failed
        if shape == "swapped_manager":
            monkeypatch.setattr(graph.api, "user_manager", users)
        await close(graph)
        assert_poisoned_close(graph)


def test_application_graph_still_exposes_only_the_dashboard_field_the_pin_reads():
    """Drift guard: a rename must fail here, not only in the compiled graph run."""
    import dataclasses
    from src.application import ApplicationGraph
    names = {field.name for field in dataclasses.fields(ApplicationGraph)}
    assert "dashboard" in names and "user_manager" not in names


# ----------------------------------------------------------------------
# Dashboard disabled: the real manager exists but no managed writer does.
# ----------------------------------------------------------------------

@asynccontextmanager
async def disabled_graph(tmp_path, monkeypatch):
    root, state = tmp_path / "admission-root", tmp_path / "state"
    (root / "admission").mkdir(parents=True)
    data = state / "data"
    data.mkdir(parents=True)
    config = {
        "memory": {"privacy_db_path": str(data / "privacy.db")},
        "dashboard": {"enabled": False, "auth": {"db_path": str(data / "authority.db")}},
        "routines": {"db_path": str(data / "routines.db")},
        "analytics": {"enabled": False, "events_db": str(data / "events.db"),
                      "suggestions_path": str(data / "suggestions.json")},
        "speaker_id": {"users_path": str(data / "users.json")},
        **{name: {"db_path": str(data / (name + ".db"))} for name in ("lists", "reminders")},
    }
    ticket = {"admission_root": str(root), "run_id": uuid4().hex, "binding_id": "a" * 64, "generation": 1}
    admission = runtime_admission.RuntimeAdmission(ticket, {"state_root": str(state)},
        config["memory"]["privacy_db_path"], runtime_admission._config_digest(config))
    monkeypatch.setattr(admission, "revalidate", lambda: None)
    admission._validation_snapshot = object()
    monkeypatch.setattr(admission, "_revalidate_live", lambda snapshot=None: None)
    monkeypatch.setattr(runtime_admission, "_verified_runs", {(ticket["run_id"], ticket["binding_id"])})
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: False})
    monkeypatch.setattr(base, "_ACTIVE", set())
    stack = AsyncExitStack()
    admission.install_on(stack)
    owner = await admission.begin_destination_registration(config)
    users = UserManager(config["speaker_id"]["users_path"], voice_registry=_voice_links())
    graph = SimpleNamespace(owner=owner, stack=stack, config=config, data=data, users=users, closed=False)
    try:
        yield graph
    finally:
        await close(graph)


def accounts_state(owner):
    from src.memory import output_registration_v4 as registration_v4
    return registration_v4.chain(owner._directory)[2].get("accounts")


async def test_disabled_publisher_is_recorded_and_composition_pins_the_same_manager(tmp_path, monkeypatch):
    async with disabled_graph(tmp_path, monkeypatch) as graph:
        owner, users = graph.owner, graph.users
        assert owner._selected["accounts"] is False
        await owner.accounts_disabled(users)
        state = accounts_state(owner)
        assert state["phase"] == "disabled" and state["plan"] is None
        assert state["path"] == str(Path(users.data_path).resolve())
        assert state["configured_path"] == str(Path(users.legacy_path).absolute())
        assert base._chain(owner._directory)[2]["accounts"] == state
        assert [event["phase"] for event in account_events(owner)] == ["disabled"]
        assert not Path(users.data_path).exists() and not owner._failed
        # The pin passes; composition then stops only at the uncomposed families.
        with pytest.raises(registration.OutputRegistrationError,
                           match="^output_registration_composition_incomplete$"):
            await registration.composed(owner, SimpleNamespace(dashboard=None))


async def test_disabled_publisher_rejects_composed_graph_with_a_dashboard(tmp_path, monkeypatch):
    async with disabled_graph(tmp_path, monkeypatch) as graph:
        owner = graph.owner
        await owner.accounts_disabled(graph.users)
        present = DashboardAPI(user_manager=graph.users)
        journal = journal_bytes(owner)
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_account_owner$"):
            await registration.composed(owner, SimpleNamespace(dashboard=present))
        assert owner._failed and journal_bytes(owner) == journal
        await close(graph)
        assert str(owner._closing.exception()) == "output_registration_epoch_incomplete"


async def test_disabled_then_attach_and_duplicate_disabled_are_rejected(tmp_path, monkeypatch):
    for second in ("attach", "disabled_again"):
        async with disabled_graph(tmp_path / second, monkeypatch) as graph:
            owner, users = graph.owner, graph.users
            await owner.accounts_disabled(users)
            journal = journal_bytes(owner)
            with pytest.raises(registration.OutputRegistrationError, match="^output_registration_account_owner$"):
                if second == "attach":
                    await owner.attach_accounts(users, DashboardAPI(user_manager=users))
                else:
                    await owner.accounts_disabled(users)
            assert owner._failed and owner._accounts is None and journal_bytes(owner) == journal
            assert accounts_state(owner)["phase"] == "disabled"


async def test_disabled_then_attach_is_refused_by_the_disabled_marker_itself(tmp_path, monkeypatch):
    """Isolate the order guard: selection is flipped so only the marker can refuse."""
    async with disabled_graph(tmp_path, monkeypatch) as graph:
        owner, users = graph.owner, graph.users
        await owner.accounts_disabled(users)
        monkeypatch.setitem(owner._selected, "accounts", True)
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_account_owner$"):
            await owner.attach_accounts(users, DashboardAPI(user_manager=users))
        assert owner._accounts is None and owner._failed


async def test_disabled_after_attach_is_rejected(tmp_path, monkeypatch, dashboard_auth):
    async with account_graph(tmp_path, monkeypatch, token(dashboard_auth)) as graph:
        owner = graph.owner
        journal = journal_bytes(owner)
        with pytest.raises(registration.OutputRegistrationError, match="^output_registration_account_owner$"):
            await owner.accounts_disabled(graph.users)
        assert owner._failed and not owner._accounts_disabled and journal_bytes(owner) == journal
        await close(graph)
        assert_poisoned_close(graph)


@pytest.mark.parametrize("malformation,code", [
    ("extra_key", "account_event"), ("relative_path", "plan"), ("selected_header", "account_event")])
async def test_malformed_disabled_event_is_rejected_by_v3_chain(tmp_path, monkeypatch, malformation, code):
    from src.memory import output_registration_v4 as registration_v4
    async with disabled_graph(tmp_path, monkeypatch) as graph:
        owner = graph.owner
        await owner.accounts_disabled(graph.users)
        directory = owner._directory
        event, = account_events(owner)
        local = {key: value for key, value in event.items() if key not in {"revision", "previous_sha256"}}
        if malformation == "extra_key":
            local["names"] = ["Synthetic"]
        elif malformation == "relative_path":
            local["path"] = "data/users_accounts_v1.json"
        else:
            # Same event, but under a header that selects the publisher.
            header, _ = base._read(directory / "epoch.json")
            header["selected"]["accounts"] = True
            (directory / "epoch.json").write_bytes(base._encoded(header))
        previous = base._sha((directory / "epoch.json").read_bytes())
        for path in directory.glob("[0-9]*.json"):
            path.unlink()
        raw = base._encoded(dict(local, revision=1, previous_sha256=previous))
        (directory / f"{1:012d}-{base._sha(raw)}.json").write_bytes(raw)
        (directory / "HEAD.json").write_bytes(base._encoded({"revision": 1, "sha256": base._sha(raw)}))
        owner._failed = True  # Preserve the malformed evidence while closing.
        expected = "^output_registration_" + code + "$"  # "plan" is the canonical-path check.
        with pytest.raises(RuntimeError, match=expected):
            registration_v4.chain(directory)
        with pytest.raises(RuntimeError, match=expected):
            base._chain(directory)
