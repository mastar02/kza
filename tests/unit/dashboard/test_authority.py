"""Synthetic server authority; no environment credentials or household data."""

import asyncio
from contextlib import closing
from contextvars import ContextVar
import sqlite3
import threading

import pytest
import pytest_asyncio

from src.dashboard.authority import (
    DashboardAuthority, DashboardPermission as Permission, DashboardRole as Role,
    EvaluationModality as Modality, EvaluationSample, legacy_lease, source_digest,
)


@pytest_asyncio.fixture
async def authority(tmp_path):
    now, token = [1_700_000_000.0], ["synthetic-administrative-token-A"]
    service = DashboardAuthority(tmp_path / "private" / "auth.db", session_ttl_s=60,
                                 clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await service.initialize()
        yield service, now, token
    finally:
        await service.aclose()


def sample(now, **changes):
    values = {"fingerprint_id": "vf_" + "a" * 64,
              "source_digests": frozenset({source_digest({"capture_id": "synthetic-source", "expires_at": now + 20})}),
              "modalities": frozenset({Modality.AUDIO_PREVIEW}), "expires_at": now + 20}
    return EvaluationSample(**(values | changes))


async def issue(fixture, role=Role.HOME, **changes):
    service, now, token = fixture
    values = {"subject": "synthetic-member", "role": role, "expires_at": now[0] + 120}
    if role is Role.EVALUATOR:
        values["samples"] = (sample(now[0]),)
    return await service.issue_credential(service.resolve(token[0]), **(values | changes))


@pytest.mark.asyncio
@pytest.mark.parametrize("role,personal", [(Role.OPERATOR, False), (Role.HOME, False),
                                         (Role.HOME, True), (Role.EVALUATOR, False)])
async def test_server_roles_do_not_inherit_admin_personal_or_review_authority(authority, role, personal):
    service, now, root = authority
    credential = await issue(authority, role, **({"user_id": "synthetic-user"} if personal else {}))
    lease = service.resolve("Bearer " + credential.token)
    assert lease.principal.role is role
    assert lease.principal.user_id == ("synthetic-user" if personal else None)
    if role is Role.OPERATOR:
        assert lease.current(frozenset(Permission))
    else:
        for permission in (Permission.ADMIN_MANAGE, Permission.AUTH_MANAGE, Permission.MEMORY_REVIEW,
                           Permission.VOICES_REVIEW, Permission.CONVERSATIONS_READ, Permission.ROUTINES_MANAGE):
            assert not lease.current(frozenset({permission}))
        assert (Permission.PERSONAL_TASKS in lease.principal.permissions) is personal
        assert lease.current(frozenset({Permission.HOME_CONTROL})) is (role is Role.HOME)
        assert lease.current(frozenset({Permission.ROUTINES_EXECUTE})) is (role is Role.HOME)
        with pytest.raises(PermissionError):
            await service.issue_credential(lease, subject="forbidden", role=Role.OPERATOR, expires_at=now[0] + 30)


@pytest.mark.asyncio
async def test_private_storage_and_representations_never_retain_raw_secrets(authority):
    service, _, root = authority
    credential = await issue(authority)
    session = await service.login(credential.token)
    root_session = await service.login(root[0])
    for path in service.path.parent.iterdir():
        assert path.stat().st_mode & 0o077 == 0
        content = path.read_bytes()
        for secret in (root[0], credential.token, session.token, root_session.token):
            assert secret.encode() not in content
    for value in (credential, session, root_session, session.lease, credential.principal):
        assert credential.token not in repr(value) and session.token not in repr(value)


@pytest.mark.asyncio
async def test_explicit_header_selects_identity_without_cookie_fallback(authority):
    service, _, root = authority
    privileged = await service.login(root[0])
    home = await issue(authority)
    assert service.resolve(None, privileged.token).principal.role is Role.OPERATOR
    assert service.resolve(home.token, privileged.token).principal.role is Role.HOME
    assert service.resolve("Bearer invalid", privileged.token) is None
    assert service.resolve("", privileged.token) is None
    assert service.resolve(None, root[0]) is None
    assert service.resolve(None, home.token) is None
    assert service.resolve("Bearer inválido", privileged.token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_lifetime", [20, 120])
async def test_server_expiry_is_minimum_of_session_and_credential_and_never_restored_by_clock_rollback(
    authority, credential_lifetime,
):
    service, now, _ = authority
    credential = await issue(authority, expires_at=now[0] + credential_lifetime)
    session = await service.login(credential.token)
    deadline = now[0] + min(60, credential_lifetime)
    assert session.lease.principal.expires_at == deadline
    now[0] = deadline
    assert service.resolve(None, session.token) is None
    assert session.lease.current() is False
    now[0] -= 100
    assert service.resolve(None, session.token) is None
    assert session.lease.current() is False


@pytest.mark.asyncio
async def test_observed_expiry_is_persisted_before_restart_even_if_clock_moves_back(authority):
    service, now, token = authority
    credential = await issue(authority, expires_at=now[0] + 10)
    session = await service.login(credential.token)
    now[0] += 10
    await service.synchronize()
    await service.aclose()
    now[0] -= 20
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await replacement.initialize()
        assert replacement.resolve(credential.token) is None
        assert replacement.resolve(None, session.token) is None
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
async def test_logout_revokes_only_this_session_and_survives_restart(authority):
    service, now, token = authority
    first, second = await service.login(token[0]), await service.login(token[0])
    assert await service.logout(first.lease) is True
    assert first.lease.current() is False
    assert service.resolve(None, first.token) is None
    assert service.resolve(None, second.token) is not None
    assert await service.logout(service.resolve(token[0])) is False
    await service.aclose()
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await replacement.initialize()
        assert replacement.resolve(None, first.token) is None
        assert replacement.resolve(None, second.token) is not None
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
async def test_credential_revocation_retires_every_derived_session_and_survives_restart(authority):
    service, now, root = authority
    credential = await issue(authority)
    one, two = await service.login(credential.token), await service.login(credential.token)
    assert await service.revoke_credential(service.resolve(root[0]), credential.identity) is True
    assert not one.lease.current() and not two.lease.current()
    assert service.resolve(credential.token) is None
    assert await service.revoke_credential(service.resolve(root[0]), credential.identity) is False
    await service.aclose()
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: root[0])
    try:
        await replacement.initialize()
        assert replacement.resolve(credential.token) is None
        assert replacement.resolve(None, one.token) is None
        assert replacement.resolve(None, two.token) is None
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_between_rotations", [False, True])
async def test_admin_rotation_a_b_a_never_resurrects_old_sessions_or_leases(authority, persist_between_rotations):
    service, now, token = authority
    original, session = service.resolve(token[0]), await service.login(token[0])
    old_token = token[0]
    token[0] = "synthetic-administrative-token-B"
    assert original.current() is False and session.lease.current() is False
    if persist_between_rotations:
        await service.synchronize()
        assert service.resolve(token[0]) is not None
    token[0] = old_token
    await service.synchronize()
    assert service.resolve(old_token) is not None
    assert service.resolve(None, session.token) is None
    assert original.current() is False
    await service.aclose()
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await replacement.initialize()
        assert replacement.resolve(old_token) is not None
        assert replacement.resolve(None, session.token) is None
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
async def test_evaluation_requires_exact_complete_sources_expiry_and_separate_modalities(authority):
    service, now, _ = authority
    selection = sample(now[0])
    credential = await issue(authority, Role.EVALUATOR, samples=(selection,))
    lease = service.resolve(credential.token)
    assert lease.sample(selection.fingerprint_id, selection.source_digests) == selection
    assert Modality.TRANSCRIPT_FULL not in lease.sample(selection.fingerprint_id).modalities
    assert lease.sample("vf_" + "b" * 64) is None
    assert lease.sample(selection.fingerprint_id, frozenset()) is None
    assert lease.sample(selection.fingerprint_id, selection.source_digests | {"b" * 64}) is None
    assert selection.fingerprint_id not in repr(lease)
    now[0] = selection.expires_at
    assert lease.current() is True
    assert lease.sample(selection.fingerprint_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"role": "operator"}, {"role": Role.EVALUATOR, "samples": ()},
    {"role": Role.OPERATOR, "user_id": "unbound-user"}, {"expires_at": float("inf")},
    {"expires_at": 0}, {"expires_at": 1_700_000_000 + 31 * 86400},
])
async def test_incomplete_or_client_like_grants_are_rejected(authority, changes):
    with pytest.raises(ValueError):
        await issue(authority, **changes)


@pytest.mark.asyncio
async def test_another_authority_or_standalone_legacy_lease_cannot_mint_grants(authority, tmp_path):
    service, now, token = authority
    other = DashboardAuthority(tmp_path / "other.db", clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await other.initialize()
        for actor in (other.resolve(token[0]), legacy_lease(token[0], expected=lambda: token[0])):
            with pytest.raises(PermissionError):
                await service.issue_credential(actor, subject="foreign", role=Role.HOME, expires_at=now[0] + 10)
    finally:
        await other.aclose()


@pytest.mark.asyncio
async def test_cancelled_login_still_owns_commit_and_publication_in_empty_context(authority, monkeypatch):
    service, _, root = authority
    entered, release = threading.Event(), threading.Event()
    private_context = ContextVar("synthetic_private_request", default=None)
    observations, original = [], service._transaction
    def transaction(action, args, guard):
        if action == "login":
            observations.append(private_context.get())
            entered.set()
            assert release.wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    mark = private_context.set("synthetic-request-only")
    task = asyncio.create_task(service.login(root[0]))
    private_context.reset(mark)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._tasks and not service._sessions
    finally:
        release.set()
        await asyncio.gather(*tuple(service._tasks), return_exceptions=True)
    assert observations == [None]
    assert len(service._sessions) == 1
    with closing(sqlite3.connect(service.path)) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_revocation_seals_admission_before_delayed_sqlite_commit(authority, monkeypatch):
    service, _, root = authority
    credential = await issue(authority)
    session = await service.login(credential.token)
    entered, release = threading.Event(), threading.Event()
    original = service._transaction
    def transaction(action, args, guard):
        if action == "revoke_credential":
            entered.set()
            assert release.wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    task = asyncio.create_task(service.revoke_credential(service.resolve(root[0]), credential.identity))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not session.lease.current()
        assert service.resolve(credential.token) is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(*tuple(service._tasks), return_exceptions=True)
    assert service.resolve(None, session.token) is None
    with closing(sqlite3.connect(service.path)) as db:
        assert db.execute("SELECT count(*) FROM credentials").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_second_writer_cannot_open_same_authority_or_disturb_first(authority):
    service, now, token = authority
    other = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        with pytest.raises(BlockingIOError):
            await other.initialize()
        assert other.resolve(token[0]) is None
        assert service.resolve(token[0]) is not None
    finally:
        await other.aclose()


@pytest.mark.asyncio
async def test_failed_persistence_closes_authority_without_publishing_success(authority, monkeypatch):
    service, _, token = authority
    original = service._transaction
    def transaction(action, args, guard):
        if action == "login":
            raise sqlite3.OperationalError("synthetic-storage-failure")
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    with pytest.raises(sqlite3.OperationalError):
        await service.login(token[0])
    assert not service.configured
    assert service.resolve(token[0]) is None
    assert not service._sessions


@pytest.mark.asyncio
async def test_evaluation_sample_expiry_also_survives_restart_with_clock_rollback(authority):
    service, now, token = authority
    selection = sample(now[0])
    credential = await issue(authority, Role.EVALUATOR, samples=(selection,))
    now[0] = selection.expires_at
    await service.synchronize()
    await service.aclose()
    now[0] -= 10
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: token[0])
    try:
        await replacement.initialize()
        lease = replacement.resolve(credential.token)
        assert lease is not None and lease.current()
        assert lease.sample(selection.fingerprint_id) is None
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("self_revision", [False, True])
async def test_permission_reduction_keeps_credential_valid_and_retires_old_sessions_and_leases(authority, self_revision):
    service, now, root = authority
    credential = await issue(authority, Role.OPERATOR)
    old_lease = service.resolve(credential.token)
    session = await service.login(credential.token)
    actor = old_lease if self_revision else service.resolve(root[0])
    assert await service.revise_credential(actor, credential.identity, role=Role.HOME,
                                           expires_at=now[0] + 90)
    assert not old_lease.current() and not session.lease.current()
    assert service.resolve(None, session.token) is None
    replacement = service.resolve(credential.token)
    assert replacement.principal.role is Role.HOME
    assert replacement.principal.revision == old_lease.principal.revision + 1
    assert replacement.current(frozenset({Permission.HOME_READ}))
    assert not replacement.current(frozenset({Permission.VOICES_REVIEW}))
    assert not replacement.current(frozenset({Permission.AUTH_MANAGE}))


@pytest.mark.asyncio
async def test_expired_actor_cannot_commit_a_login_held_in_native_queue(authority, monkeypatch):
    service, now, _ = authority
    credential = await issue(authority, expires_at=now[0] + 5)
    entered, release = threading.Event(), threading.Event()
    original = service._transaction
    def transaction(action, args, guard):
        if action == "login":
            entered.set()
            assert release.wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    task = asyncio.create_task(service.login(credential.token))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        now[0] += 5
    finally:
        release.set()
    with pytest.raises(PermissionError):
        await task
    assert not service._sessions
    with closing(sqlite3.connect(service.path)) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    assert service.configured


@pytest.mark.asyncio
async def test_commit_to_publication_gap_remains_owned_after_caller_cancel(authority, monkeypatch):
    service, _, root = authority
    committed, release = asyncio.Event(), asyncio.Event()
    original = service._work.run
    async def run(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if args[0] == "login":
            committed.set()
            await release.wait()
        return result
    monkeypatch.setattr(service._work, "run", run)
    task = asyncio.create_task(service.login(root[0]))
    try:
        await asyncio.wait_for(committed.wait(), 3)
        with closing(sqlite3.connect(service.path)) as db:
            assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
        assert not service._sessions
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._tasks
    finally:
        release.set()
        await asyncio.gather(*tuple(service._tasks), return_exceptions=True)
    assert len(service._sessions) == 1


@pytest.mark.asyncio
async def test_close_retains_writer_lease_until_native_drain_even_if_operation_observer_is_cancelled(
    authority, monkeypatch,
):
    service, now, root = authority
    entered, release = threading.Event(), threading.Event()
    original = service._transaction
    def transaction(action, args, guard):
        if action == "login":
            entered.set()
            assert release.wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    caller = asyncio.create_task(service.login(root[0]))
    other = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: root[0])
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        operation = next(iter(service._tasks))
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        observer = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not observer.done()
        assert service._lease_fd is not None
        with pytest.raises(BlockingIOError):
            await other.initialize()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert service._lease_fd is not None
    finally:
        release.set()
        await service.aclose()
        await other.aclose()
    replacement = DashboardAuthority(service.path, clock=lambda: now[0], expected_token=lambda: root[0])
    try:
        await replacement.initialize()
        assert replacement.resolve(root[0]) is not None
        assert not replacement._sessions
    finally:
        await replacement.aclose()


def test_standalone_legacy_binding_stays_retired_after_observed_rotation():
    token = ["synthetic-legacy-A"]
    lease = legacy_lease(token[0], expected=lambda: token[0])
    assert lease.current()
    token[0] = "synthetic-legacy-B"
    assert not lease.current()
    token[0] = "synthetic-legacy-A"
    assert not lease.current()
    assert legacy_lease(token[0], expected=lambda: token[0]).current()


@pytest.mark.asyncio
async def test_concurrent_role_changes_keep_old_permission_closed_until_last_commit(authority, monkeypatch):
    service, now, root = authority
    credential = await issue(authority, Role.OPERATOR)
    old = service.resolve(credential.token)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    calls, original = [], service._transaction
    def transaction(action, args, guard):
        if action == "revise":
            index = len(calls)
            calls.append(index)
            entered[index].set()
            assert release[index].wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(service, "_transaction", transaction)
    first = asyncio.create_task(service.revise_credential(service.resolve(root[0]), credential.identity,
                                                          role=Role.HOME, expires_at=now[0] + 90))
    second = None
    try:
        assert await asyncio.to_thread(entered[0].wait, 3)
        assert not old.current() and service.resolve(credential.token) is None
        second = asyncio.create_task(service.revise_credential(service.resolve(root[0]), credential.identity,
                          role=Role.EVALUATOR, expires_at=now[0] + 90, samples=(sample(now[0]),)))
        await asyncio.sleep(0)
        release[0].set()
        assert await asyncio.to_thread(entered[1].wait, 3)
        assert not old.current() and service.resolve(credential.token) is None
    finally:
        for signal in release:
            signal.set()
        results = await asyncio.gather(*([first, second] if second is not None else [first]), return_exceptions=True)
    assert results == [True, True]
    current = service.resolve(credential.token)
    assert current.principal.role is Role.EVALUATOR
    assert current.principal.revision == 3
    assert current.sample("vf_" + "a" * 64) is not None
    assert not current.current(frozenset({Permission.HOME_CONTROL}))
