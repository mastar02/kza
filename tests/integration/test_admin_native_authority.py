"""Synthetic HTTP administration retains native effects and original leases."""

import asyncio
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import gc
import io
import os
import subprocess
import threading
from types import SimpleNamespace
import wave
import weakref

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.alerts.alert_manager import Alert, AlertManager, AlertPriority, AlertType
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.dashboard.native_work import run_admin_work
from src.llm.cooldown import CooldownManager
from src.llm.metrics import LLMMetricsTracker
from src.llm.types import ErrorKind
from src.users.user_manager import PermissionLevel, UserManager
from tests.mocks.memory_graph import Clock


@pytest.fixture
async def admin_graph(tmp_path, dashboard_auth):
    clock = Clock()
    users = UserManager(str(tmp_path / "users.json"))
    user, _ = users.add_user("Synthetic", permission_level=PermissionLevel.ADULT)
    cooldown = CooldownManager(tmp_path / "cooldown.json")
    cooldown.record_failure("synthetic", ErrorKind.BILLING)
    alerts = AlertManager()
    alert = Alert("synthetic", AlertType.SECURITY, AlertPriority.MEDIUM, "Synthetic")
    alerts._history.append(alert)
    speaker = SimpleNamespace(create_enrollment_embedding=lambda arrays: np.ones(8, dtype=np.float32))
    lock = threading.Lock()
    router = SimpleNamespace(_cd=cooldown, _endpoints=[SimpleNamespace(id="synthetic", kind="local")])
    metrics = LLMMetricsTracker()
    metrics.record("synthetic", tokens=8, elapsed_ms=20)
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / "auth.db", clock=clock, session_ttl_s=60,
                                       account_provider=users.active_accounts)
        users._retirement_proof = authority.users_retired
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        api = DashboardAPI(authority=authority, user_manager=users, speaker_identifier=speaker,
                           speaker_model_lock=lock, alert_manager=alerts, llm_router=router, llm_metrics=metrics)
        stack.push_async_callback(api.stop)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://testserver") as client:
            token = dashboard_auth["Authorization"].split(" ", 1)[1]
            assert (await client.post("/api/admin/auth/login", json={"token": token})).status_code == 200
            yield SimpleNamespace(api=api, client=client, users=users, user=user, cooldown=cooldown,
                                  alerts=alerts, alert=alert, speaker=speaker, lock=lock,
                                  clock=clock, authority=authority, metrics=metrics)


async def wait_event(event):
    assert await asyncio.to_thread(event.wait, 3)


def samples():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(3200))
    return [("samples", (f"sample-{n}.wav", stream.getvalue(), "audio/wav")) for n in range(3)]


async def test_admin_people_projection_contains_only_scalars_and_denies_household(admin_graph, monkeypatch):
    graph = admin_graph
    import src.users.user_manager as catalog

    def forbidden(*args, **kwargs):
        raise AssertionError('private account projection attempted')

    monkeypatch.setattr(catalog, 'deepcopy', forbidden)
    monkeypatch.setattr(graph.users, 'get_all_users', forbidden)
    response = await graph.client.get('/api/admin/users')
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.json() == [{'id': graph.user.user_id, 'name': 'Synthetic',
                                'permission_level': 'adult', 'is_active': True}]
    credential = await graph.authority.issue_credential(graph.authority.resolve(os.environ['KZA_DASHBOARD_TOKEN']),
                    subject='synthetic-household', role=DashboardRole.HOME, expires_at=graph.clock.now + 120)
    assert (await graph.client.post('/api/admin/auth/login', json={'token': credential.token})).status_code == 200
    monkeypatch.setattr(graph.users, 'project_accounts', forbidden)
    assert (await graph.client.get('/api/admin/users')).status_code == 403


async def test_admin_people_catalog_unavailable_is_not_an_empty_list(admin_graph):
    graph = admin_graph
    graph.users._load_error = True
    response = await graph.client.get('/api/admin/users')
    assert response.status_code == 503
    assert response.json()['detail'] == 'user_catalog_unavailable'


async def test_last_active_admin_is_protected_before_credential_retirement(admin_graph, monkeypatch):
    graph = admin_graph
    graph.users.update_user(graph.user.user_id, permission_level=PermissionLevel.ADMIN)

    async def forbidden(*args):
        raise AssertionError('protected account reached retirement')

    monkeypatch.setattr(graph.authority, 'retire_user', forbidden)
    response = await graph.client.delete(f'/api/admin/users/{graph.user.user_id}')
    assert response.status_code == 409
    assert graph.users.get_user(graph.user.user_id) is not None


async def test_closed_shared_owner_rejects_delete_before_any_native_work(admin_graph, monkeypatch):
    graph = admin_graph
    await graph.api.admin_work.aclose()

    def forbidden(*args, **kwargs):
        raise AssertionError('closed owner dispatched a native read')

    monkeypatch.setattr(graph.users, 'check_removal', forbidden)
    response = await graph.client.delete(f'/api/admin/users/{graph.user.user_id}')
    assert response.status_code == 503
    assert response.json()['detail']['credentials_retired'] is False


async def test_admin_delete_rejected_retirement_never_calls_catalog_remove(admin_graph, monkeypatch):
    graph = admin_graph
    calls = []

    async def denied(*args):
        raise PermissionError('synthetic retirement denied')

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('catalog remove attempted')

    monkeypatch.setattr(graph.authority, 'retire_user', denied)
    monkeypatch.setattr(graph.users, 'remove_user', forbidden)
    response = await graph.client.delete(f'/api/admin/users/{graph.user.user_id}')
    assert response.status_code == 403
    assert response.json()['detail']['credentials_retired'] is False
    assert calls == []
    assert graph.users.get_user(graph.user.user_id) is not None


async def test_cancelled_admin_delete_keeps_both_retirement_and_catalog_commit_owned(admin_graph, monkeypatch):
    graph = admin_graph
    entered, release = threading.Event(), threading.Event()
    transaction = graph.authority._transaction
    root = graph.authority.resolve(os.environ['KZA_DASHBOARD_TOKEN'])
    credential = await graph.authority.issue_credential(root, subject='synthetic-personal', role=DashboardRole.HOME,
                    expires_at=graph.clock.now + 120, user_id=graph.user.user_id)
    session = await graph.authority.login(credential.token)

    def held(action, args, guard):
        if action == 'retire_user':
            entered.set()
            assert release.wait(5)
        return transaction(action, args, guard)

    monkeypatch.setattr(graph.authority, '_transaction', held)
    closing = None
    request = asyncio.create_task(graph.client.delete(f'/api/admin/users/{graph.user.user_id}'))
    try:
        await wait_event(entered)
        assert not session.lease.current()
        assert graph.users.get_user(graph.user.user_id) is not None
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        closing = graph.api.admin_work.begin_close()
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    if closing is not None:
        await closing
    assert graph.users.get_user(graph.user.user_id) is None
    assert graph.authority.users_retired(frozenset({graph.user.user_id}))
    assert graph.authority.resolve(credential.token) is None


async def test_delete_catalog_failure_reports_retirement_and_retry_completes(admin_graph, monkeypatch):
    graph = admin_graph
    import src.users.user_manager as catalog
    replace = catalog.os.replace

    def failed(*args):
        raise OSError('synthetic private path secret')

    monkeypatch.setattr(catalog.os, 'replace', failed)
    response = await graph.client.delete(f'/api/admin/users/{graph.user.user_id}')
    assert response.status_code == 503
    assert response.json()['detail'] == {'code': 'user_catalog_unavailable', 'credentials_retired': True}
    assert 'private path' not in response.text
    assert graph.users.get_user(graph.user.user_id) is not None
    monkeypatch.setattr(catalog.os, 'replace', replace)
    assert (await graph.client.delete(f'/api/admin/users/{graph.user.user_id}')).status_code == 200
    assert graph.users.get_user(graph.user.user_id) is None


@pytest.mark.parametrize("resource", ["catalog", "cooldown"])
@pytest.mark.parametrize("cancel_observer", [False, True])
async def test_retirement_before_replace_preserves_actual_file_ram_and_other_writer(
    admin_graph, monkeypatch, dashboard_auth, resource, cancel_observer,
):
    graph = admin_graph
    path = graph.users.data_path if resource == "catalog" else graph.cooldown.persistence_path
    original_bytes, version = path.read_bytes(), graph.users._version
    entered, release = threading.Event(), threading.Event()
    fsync = os.fsync

    def flush(descriptor):
        fsync(descriptor)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(os, "fsync", flush)
    endpoint = "/api/admin/users" if resource == "catalog" else "/api/llm/endpoints/synthetic/clear-cooldown"
    pending = asyncio.create_task(graph.client.post(endpoint, json={"name": "Denied"}))
    try:
        await wait_event(entered)
        if cancel_observer:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
    finally:
        release.set()
    if not cancel_observer:
        assert (await pending).status_code == 403
    await graph.api.admin_work.drain_admitted()
    assert path.read_bytes() == original_bytes
    assert graph.users._version == version
    assert graph.users.get_user_by_name("Denied") is None
    assert graph.cooldown.get_state("synthetic").error_count == 1
    assert not list(path.parent.glob(".accounts-*"))
    assert not list(path.parent.glob(".cooldown-*"))
    # A later independent administrator remains able to publish its own state.
    response = await graph.client.post(endpoint, json={"name": "Approved"}, headers=dashboard_auth)
    assert response.status_code == 200
    if resource == "catalog":
        assert graph.users.get_user_by_name("Approved") is not None
        assert graph.users._version == version + 1
    else:
        assert CooldownManager(path).get_state("synthetic").error_count == 0


@pytest.mark.parametrize("resource", ["catalog", "cooldown"])
async def test_cancelled_observer_after_replace_cannot_retire_approved_publication(
    admin_graph, monkeypatch, resource,
):
    graph = admin_graph
    entered, release = threading.Event(), threading.Event()
    replace = os.replace
    version = graph.users._version

    def commit(source, destination):
        replace(source, destination)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(os, "replace", commit)
    endpoint = "/api/admin/users" if resource == "catalog" else "/api/llm/endpoints/synthetic/clear-cooldown"
    pending = asyncio.create_task(graph.client.post(endpoint, json={"name": "Committed"}))
    stopper = None
    try:
        await wait_event(entered)
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
        stopper = asyncio.create_task(graph.api.stop())
        await asyncio.sleep(0)
        stopper.cancel()
        await asyncio.sleep(0)
        stopper.cancel()
        await asyncio.sleep(0)
        assert not stopper.done()
        assert graph.api.admin_work.status()["operations"] == 1
    finally:
        release.set()
        if stopper is not None:
            await asyncio.gather(stopper, return_exceptions=True)
    assert graph.api.admin_work.status()["operations"] == 0
    if resource == "catalog":
        assert graph.users.get_user_by_name("Committed") is not None
        assert graph.users._version == version + 1
        assert UserManager(str(graph.users.data_path)).get_user_by_name("Committed") is not None
    else:
        assert graph.cooldown.get_state("synthetic").error_count == 0
        assert CooldownManager(graph.cooldown.persistence_path).get_state("synthetic").error_count == 0


@pytest.mark.parametrize("resource", ["users", "endpoints", "metrics"])
async def test_private_read_expiring_while_waiting_for_native_lock_has_no_projection(
    admin_graph, monkeypatch, dashboard_auth, resource,
):
    graph = admin_graph
    import src.users.user_manager as user_module

    entered, release = threading.Event(), threading.Event()
    attempted = threading.Event()
    lock = {"users": graph.users._lock, "endpoints": graph.cooldown._lock,
            "metrics": graph.metrics._lock}[resource]
    projections = []
    metadata = user_module.AccountMetadata

    def project(*args, **kwargs):
        projections.append(True)
        return metadata(*args, **kwargs)

    if resource == "users":
        monkeypatch.setattr(user_module, "AccountMetadata", project)
        name, target = "project_accounts", graph.users
    elif resource == "endpoints":
        name, target = "is_available", graph.cooldown
    else:
        name, target = "snapshot", graph.metrics
        data = graph.metrics._data

        class Observed(dict):
            def get(self, key, default=None):
                projections.append(True)
                return super().get(key, default)

        monkeypatch.setattr(graph.metrics, "_data", Observed(data))
    original = getattr(target, name)

    def read(*args, **kwargs):
        attempted.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, read)

    def hold():
        with lock:
            entered.set()
            assert release.wait(5)

    blocker = asyncio.create_task(asyncio.to_thread(hold))
    pending = None
    try:
        await wait_event(entered)
        endpoint = "/api/users" if resource == "users" else "/api/llm/endpoints"
        pending = asyncio.create_task(graph.client.get(endpoint))
        await wait_event(attempted)
        graph.clock.now += 61
    finally:
        release.set()
        await blocker
    assert (await pending).status_code == 403
    assert projections == []
    response = await graph.client.get(endpoint, headers=dashboard_auth)
    assert response.status_code == 200
    assert len(response.json()) == 1
    if resource in {"users", "metrics"}:
        assert projections == [True]


@pytest.mark.parametrize("phase", ["waiting_model", "inference", "cancelled_observer"])
async def test_enrollment_shares_native_model_admission_and_retains_only_live_publication(
    admin_graph, monkeypatch, phase,
):
    graph = admin_graph
    import src.dashboard.admin as admin_module

    entered, release = threading.Event(), threading.Event()
    admission = admin_module.model_admission
    calls = []
    version, original = graph.users._version, graph.users.data_path.read_bytes()

    @contextmanager
    def admitted(lock, validator):
        assert lock is graph.lock
        if phase == "waiting_model":
            entered.set()
        with admission(lock, validator):
            yield

    def embed(arrays):
        calls.append([array.shape for array in arrays])
        assert graph.lock.locked()
        if phase != "waiting_model":
            entered.set()
            assert release.wait(5)
        return np.ones(8, dtype=np.float32)

    monkeypatch.setattr(admin_module, "model_admission", admitted)
    graph.speaker.create_enrollment_embedding = embed
    if phase == "waiting_model":
        graph.lock.acquire()
    pending = asyncio.create_task(graph.client.post(f"/api/admin/users/{graph.user.user_id}/enroll", files=samples()))
    try:
        await wait_event(entered)
        if phase == "cancelled_observer":
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            assert graph.api.admin_work.status()["operations"] == 1
        else:
            assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
    finally:
        release.set()
        if phase == "waiting_model":
            graph.lock.release()
    if phase != "cancelled_observer":
        assert (await pending).status_code == 403
    await graph.api.admin_work.drain_admitted()
    current = graph.users.get_user(graph.user.user_id)
    assert len(calls) == (0 if phase == "waiting_model" else 1)
    if phase == "cancelled_observer":
        assert np.array_equal(current.voice_embedding, np.ones(8, dtype=np.float32))
        assert graph.users._version == version + 1
        assert UserManager(str(graph.users.data_path)).get_user(graph.user.user_id).voice_embedding is not None
    else:
        assert current.voice_embedding is None
        assert graph.users.data_path.read_bytes() == original
        assert graph.users._version == version


async def test_restart_retired_before_native_entry_launches_no_process(admin_graph, monkeypatch):
    graph = admin_graph
    import src.dashboard.admin as admin_module

    entered, release = asyncio.Event(), asyncio.Event()
    worker = graph.api.admin_work._worker
    calls = []

    async def queued(*args, **kwargs):
        entered.set()
        await release.wait()
        return await worker(*args, **kwargs)

    monkeypatch.setattr(graph.api.admin_work, "_worker", queued)
    monkeypatch.setattr(admin_module.shutil, "which", lambda name: "/synthetic/systemctl")
    monkeypatch.setattr(admin_module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    pending = asyncio.create_task(graph.client.post("/api/admin/services/kza-voice/restart"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
    finally:
        release.set()
    assert (await pending).status_code == 403
    assert calls == []


async def test_started_process_survives_observer_cancel_and_logout_in_empty_context(admin_graph, monkeypatch):
    graph = admin_graph
    import src.dashboard.admin as admin_module

    entered, release = threading.Event(), threading.Event()
    marker = ContextVar("synthetic_request_secret", default=None)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs, marker.get()))
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(admin_module.shutil, "which", lambda name: "/synthetic/systemctl")
    monkeypatch.setattr(admin_module.subprocess, "run", run)
    token = marker.set("synthetic-secret")
    try:
        pending = asyncio.create_task(graph.client.post("/api/admin/services/kza-voice/restart"))
    finally:
        marker.reset(token)
    try:
        await wait_event(entered)
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
        assert graph.api.admin_work.status()["operations"] == 1
    finally:
        release.set()
    await graph.api.admin_work.drain_admitted()
    assert calls == [(["systemctl", "--user", "restart", "kza-voice"],
                      {"capture_output": True, "text": True, "timeout": 15}, None)]


async def test_started_process_timeout_is_uncertain_and_never_retried(admin_graph, monkeypatch):
    import src.dashboard.admin as admin_module

    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(admin_module.shutil, "which", lambda name: "/synthetic/systemctl")
    monkeypatch.setattr(admin_module.subprocess, "run", timeout)
    response = await admin_graph.client.post("/api/admin/services/kza-voice/restart")
    assert response.status_code == 504
    assert response.json()["detail"] == {"reason": "restart_timeout", "effect": "uncertain"}
    assert len(calls) == 1


async def test_process_launch_error_is_reported_without_exception_payload_or_retry(admin_graph, monkeypatch):
    import src.dashboard.admin as admin_module

    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        raise OSError("synthetic_private_process_path")

    monkeypatch.setattr(admin_module.shutil, "which", lambda name: "/synthetic/systemctl")
    monkeypatch.setattr(admin_module.subprocess, "run", fail)
    response = await admin_graph.client.post("/api/admin/services/kza-voice/restart")
    assert response.status_code == 503
    assert response.json()["detail"] == {"reason": "restart_process_unavailable", "effect": "uncertain"}
    assert "synthetic_private_process_path" not in response.text
    assert len(calls) == 1


@pytest.mark.parametrize("rate,seconds,channels", [(4000, 1, 1), (16000, 31, 1), (16000, 1, 3)])
def test_invalid_wav_dimensions_are_rejected_before_allocating_decoded_audio(monkeypatch, rate, seconds, channels):
    import soundfile as sf
    from src.dashboard.admin import _decode_wav_bytes

    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(rate * seconds * channels * 2))
    allocations = []

    def read(*args, **kwargs):
        allocations.append(True)
        return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr(sf.SoundFile, "read", read)
    with pytest.raises(ValueError, match="audio_sample_invalid"):
        _decode_wav_bytes(stream.getvalue())
    assert allocations == []


def test_native_wav_conversion_is_finite_detached_mono_at_16khz():
    import soundfile as sf
    from src.dashboard.admin import _decode_wav_bytes

    stream = io.BytesIO()
    stereo = np.column_stack((np.full(800, 0.25), np.full(800, 0.75)))
    sf.write(stream, stereo, 8000, format="WAV", subtype="FLOAT")
    result = _decode_wav_bytes(stream.getvalue())
    assert result.shape == (1600,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.5)
    assert not np.shares_memory(result, stereo)
    invalid = io.BytesIO()
    sf.write(invalid, np.full(800, np.nan), 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError, match="audio_sample_invalid"):
        _decode_wav_bytes(invalid.getvalue())


@pytest.mark.parametrize("sample", ["empty_after_resample", "overflow_after_downmix"])
async def test_conversion_cannot_admit_empty_or_nonfinite_audio_to_native_model(admin_graph, sample):
    import soundfile as sf

    calls = []
    admin_graph.speaker.create_enrollment_embedding = lambda arrays: calls.append(True)
    stream = io.BytesIO()
    if sample == "empty_after_resample":
        sf.write(stream, np.zeros(1, dtype=np.float32), 192000, format="WAV", subtype="FLOAT")
    else:
        sf.write(stream, np.full((2, 2), np.finfo(np.float32).max, dtype=np.float32),
                 16000, format="WAV", subtype="FLOAT")
    uploads = [("samples", (f"sample-{n}.wav", stream.getvalue(), "audio/wav")) for n in range(3)]
    previous = admin_graph.users.data_path.read_bytes()
    response = await admin_graph.client.post(f"/api/admin/users/{admin_graph.user.user_id}/enroll", files=uploads)
    assert response.status_code == 400
    assert calls == []
    assert admin_graph.users.data_path.read_bytes() == previous
    assert admin_graph.users.get_user(admin_graph.user.user_id).voice_embedding is None


async def test_dashboard_stop_observes_server_after_native_drain_despite_repeated_cancel(admin_graph):
    graph = admin_graph
    entered, release = threading.Event(), threading.Event()
    server_release = asyncio.Event()
    graph.api._server = SimpleNamespace(should_exit=False)
    server = asyncio.create_task(server_release.wait())
    graph.api._serve_task = server

    def block():
        entered.set()
        assert release.wait(5)

    work = asyncio.create_task(graph.api.admin_work.run(block))
    closing = concurrent = None
    try:
        await wait_event(entered)
        closing = asyncio.create_task(graph.api.stop())
        concurrent = asyncio.create_task(graph.api.stop())
        await asyncio.sleep(0)
        assert graph.api._server.should_exit
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0)
        assert not closing.done()
        assert not concurrent.done()
        release.set()
        await work
        await graph.api.admin_work.drain_admitted()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not concurrent.done()
        assert not server.done()
    finally:
        release.set()
        server_release.set()
        await asyncio.gather(work, return_exceptions=True)
        results = await asyncio.gather(*(task for task in (closing, concurrent) if task), return_exceptions=True)
        await server
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] is None
    assert graph.api._stop_task.done()
    await graph.api.stop()


async def test_cancelled_observer_traceback_cannot_keep_completed_native_payload(admin_graph):
    class Payload:
        pass

    entered, release = threading.Event(), threading.Event()
    payload = Payload()
    reference = weakref.ref(payload)

    def native(value):
        try:
            assert isinstance(value, Payload)
            entered.set()
            assert release.wait(5)
        finally:
            value = None

    pending = asyncio.create_task(run_admin_work(admin_graph.api.admin_work, lambda: True, native, payload))
    retained_error = None
    try:
        await wait_event(entered)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError as error:
            retained_error = error
        payload = None
        assert reference() is not None
    finally:
        release.set()
        await admin_graph.api.admin_work.drain_admitted()
    await asyncio.sleep(0)
    gc.collect()
    assert retained_error is not None
    assert reference() is None


@pytest.mark.parametrize("action", ["ack", "dismiss"])
async def test_alert_waiting_for_lock_rechecks_original_lease_before_any_field_change(
    admin_graph, monkeypatch, dashboard_auth, action,
):
    graph = admin_graph
    previous = datetime(2020, 1, 1, tzinfo=timezone.utc)
    graph.alert.processed_at = previous
    entered = asyncio.Event()
    original = graph.alerts.mark_processed

    async def process(*args, **kwargs):
        entered.set()
        return await original(*args, **kwargs)

    monkeypatch.setattr(graph.alerts, "mark_processed", process)
    await graph.alerts._lock.acquire()
    method = "POST" if action == "ack" else "DELETE"
    endpoint = "/api/admin/alerts/synthetic/ack" if action == "ack" else "/api/admin/alerts/synthetic"
    pending = asyncio.create_task(graph.client.request(method, endpoint))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert (await graph.client.post("/api/admin/auth/logout")).json()["session_revoked"]
    finally:
        graph.alerts._lock.release()
    assert (await pending).status_code == 403
    assert graph.alert.processed is False
    assert graph.alert.processed_at == previous
    assert (await graph.client.request(method, endpoint, headers=dashboard_auth)).status_code == 200
    assert graph.alert.processed is True
    assert (graph.alert.processed_at != previous) is (action == "ack")


@pytest.mark.parametrize("resource", ["catalog", "cooldown", "enrollment"])
async def test_persistence_error_never_publishes_success_or_ram(admin_graph, monkeypatch, resource):
    graph = admin_graph
    path = graph.cooldown.persistence_path if resource == "cooldown" else graph.users.data_path
    previous, version = path.read_bytes(), graph.users._version

    def fail(*args):
        raise PermissionError("synthetic_private_path")

    monkeypatch.setattr(os, "replace", fail)
    endpoint = "/api/admin/users" if resource == "catalog" else "/api/llm/endpoints/synthetic/clear-cooldown"
    if resource == "enrollment":
        response = await graph.client.post(f"/api/admin/users/{graph.user.user_id}/enroll", files=samples())
    else:
        response = await graph.client.post(endpoint, json={"name": "Failed"})
    assert response.status_code == 503
    assert "synthetic_private_path" not in response.text
    assert path.read_bytes() == previous
    assert graph.users._version == version
    assert graph.users.get_user_by_name("Failed") is None
    assert graph.users.get_user(graph.user.user_id).voice_embedding is None
    assert graph.cooldown.get_state("synthetic").error_count == 1
