"""Tests para /api/admin/* — auth, users CRUD, enroll, alerts, services."""

import io
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import threading
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI

from src.dashboard.admin import register_admin_routes
from src.users.user_manager import UserManager
from tests.mocks.dashboard_sessions import session_client


TEST_TOKEN = "test-secret-token-1234567890"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", TEST_TOKEN)


@pytest.fixture
def fake_user_manager(tmp_path):
    """Exercise actual catalog publication with synthetic private storage."""
    return UserManager(str(tmp_path / "users.json"))


@pytest.fixture
def fake_speaker_identifier():
    si = MagicMock()
    si.create_enrollment_embedding.return_value = np.zeros(192, dtype=np.float32)
    return si


@pytest.fixture
def fake_alert_manager():
    from src.alerts.alert_manager import Alert, AlertManager, AlertPriority, AlertType
    alert = Alert(alert_id="al_test", alert_type=AlertType.SECURITY,
                  priority=AlertPriority.MEDIUM, message="test")
    manager = AlertManager()
    manager._history.append(alert)
    return manager


@pytest.fixture
def _client(tmp_path):
    with ExitStack() as stack:
        sequence = []
        def create(**kwargs):
            def app_factory(authority):
                app = FastAPI()
                app.state.dashboard_authority = authority
                register_admin_routes(app, **kwargs)
                return app
            path = tmp_path / f"app-{len(sequence)}" / "auth.db"
            sequence.append(path)
            return stack.enter_context(session_client(app_factory, path))
        yield create


def _bearer():
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    """PCM 16-bit mono WAV en memoria."""
    n = int(sr * seconds)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 32767 * 0.3).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------- Auth ----------------

def test_admin_endpoint_without_token_returns_401(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users", json={"name": "X", "permission_level": "adult"})
    assert r.status_code == 401


def test_admin_endpoint_with_bad_token_returns_401(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "X", "permission_level": "adult"},
               headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_admin_endpoint_with_correct_token_passes(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "Tomás", "permission_level": "adult"},
               headers=_bearer())
    assert r.status_code == 200
    assert r.json()["name"] == "Tomás"


def test_login_sets_cookie(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/auth/login", json={"token": TEST_TOKEN})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "kza_dashboard_token" in r.cookies


def test_login_with_wrong_token_returns_401(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/auth/login", json={"token": "wrong"})
    assert r.status_code == 401


def test_cookie_grants_access_to_admin_endpoints(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    c.post("/api/admin/auth/login", json={"token": TEST_TOKEN})
    r = c.get("/api/admin/auth/whoami")
    assert r.status_code == 200


# ---------------- Users CRUD ----------------

def test_create_user_with_invalid_permission_level(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "X", "permission_level": "superadmin"},
               headers=_bearer())
    assert r.status_code == 400


def test_create_duplicate_user_returns_409(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    c.post("/api/admin/users",
           json={"name": "Ana", "permission_level": "adult"}, headers=_bearer())
    r = c.post("/api/admin/users",
               json={"name": "Ana", "permission_level": "adult"}, headers=_bearer())
    assert r.status_code == 409


def test_delete_user(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    cr = c.post("/api/admin/users",
                json={"name": "Bob", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    r = c.delete(f"/api/admin/users/{uid}", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_delete_unknown_user_returns_404(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    r = c.delete("/api/admin/users/nonexistent", headers=_bearer())
    assert r.status_code == 404


def test_update_permissions(fake_user_manager, _client):
    c = _client(user_manager=fake_user_manager)
    cr = c.post("/api/admin/users",
                json={"name": "Eva", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    r = c.put(f"/api/admin/users/{uid}/permissions",
              json={"permission_level": "admin"}, headers=_bearer())
    assert r.status_code == 200
    assert r.json()["permission_level"] == "ADMIN"


# ---------------- Enrollment ----------------

def test_enroll_requires_3_samples(fake_user_manager, fake_speaker_identifier, _client):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    cr = c.post("/api/admin/users",
                json={"name": "Lía", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    files = [("samples", ("s1.wav", _wav_bytes(), "audio/wav"))]
    r = c.post(f"/api/admin/users/{uid}/enroll", files=files, headers=_bearer())
    assert r.status_code == 400


def test_enroll_with_3_samples_creates_embedding(fake_user_manager, fake_speaker_identifier, _client):
    observed = []
    def embed(arrays):
        assert len(arrays) == 3 and all(isinstance(array, np.ndarray) for array in arrays)
        assert all(array.dtype == np.float32 for array in arrays)
        observed.append(tuple(len(array) for array in arrays))
        return np.zeros(192, dtype=np.float32)
    fake_speaker_identifier.create_enrollment_embedding.side_effect = embed
    client = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    created = client.post("/api/admin/users",
        json={"name": "Lía", "permission_level": "adult"}, headers=_bearer())
    identity = created.json()["user_id"]
    files = [("samples", (f"s{i}.wav", _wav_bytes(), "audio/wav")) for i in range(3)]
    response = client.post(f"/api/admin/users/{identity}/enroll", files=files, headers=_bearer())
    assert response.status_code == 200 and response.json()["samples"] == 3
    assert response.json()["embedding_dim"] == 192
    assert len(observed) == 1 and len(observed[0]) == 3
    fake_speaker_identifier.create_enrollment_embedding.assert_called_once()
    assert fake_speaker_identifier.create_enrollment_embedding.call_args[0][0] == []
    np.testing.assert_array_equal(fake_user_manager.get_user(identity).voice_embedding,
                                  np.zeros(192, dtype=np.float32))


def test_enrollment_rechecks_original_session_after_native_work(
    fake_user_manager, fake_speaker_identifier, _client,
):
    client = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    created = client.post("/api/admin/users",
        json={"name": "Synthetic", "permission_level": "adult"}, headers=_bearer())
    identity = created.json()["user_id"]
    before_file = fake_user_manager.data_path.read_bytes()
    before_version = fake_user_manager._version
    user = fake_user_manager.get_user(identity)
    before = user.voice_embedding
    assert client.post("/api/admin/auth/login", json={"token": TEST_TOKEN}).status_code == 200
    entered, release = threading.Event(), threading.Event()
    def embed(audio):
        entered.set()
        assert release.wait(5)
        return np.zeros(192, dtype=np.float32)
    fake_speaker_identifier.create_enrollment_embedding.side_effect = embed
    files = [("samples", (f"s{i}.wav", _wav_bytes(), "audio/wav")) for i in range(3)]
    authority = client.app.state.dashboard_authority
    lease = authority.resolve(None, client.cookies.get("kza_dashboard_token"))
    with ThreadPoolExecutor(max_workers=1) as workers:
        request = workers.submit(client.post, f"/api/admin/users/{identity}/enroll", files=files)
        try:
            assert entered.wait(3)
            assert client.portal.call(authority.logout, lease) is True
        finally:
            release.set()
        assert request.result(timeout=5).status_code == 403
    assert user.voice_embedding is before
    assert fake_user_manager.get_user(identity).voice_embedding is None
    assert fake_user_manager.data_path.read_bytes() == before_file
    assert fake_user_manager._version == before_version


def test_enroll_unknown_user_returns_404(fake_user_manager, fake_speaker_identifier, _client):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    files = [("samples", (f"s{i}.wav", _wav_bytes(), "audio/wav")) for i in range(3)]
    r = c.post("/api/admin/users/nonexistent/enroll", files=files, headers=_bearer())
    assert r.status_code == 404


def test_enroll_with_invalid_audio_returns_400(fake_user_manager, fake_speaker_identifier, _client):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    cr = c.post("/api/admin/users",
                json={"name": "Z", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    files = [("samples", (f"s{i}.wav", b"not-a-wav", "audio/wav")) for i in range(3)]
    r = c.post(f"/api/admin/users/{uid}/enroll", files=files, headers=_bearer())
    assert r.status_code == 400


# ---------------- Alerts ----------------

def test_alert_ack(fake_alert_manager, _client):
    c = _client(alert_manager=fake_alert_manager)
    r = c.post("/api/admin/alerts/al_test/ack", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["alert_id"] == "al_test"


def test_alert_ack_unknown_returns_404(fake_alert_manager, _client):
    c = _client(alert_manager=fake_alert_manager)
    r = c.post("/api/admin/alerts/al_unknown/ack", headers=_bearer())
    assert r.status_code == 404


def test_alert_dismiss(fake_alert_manager, _client):
    c = _client(alert_manager=fake_alert_manager)
    r = c.delete("/api/admin/alerts/al_test", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["dismissed"] is True


# ---------------- Services ----------------

def test_restart_service_not_in_allowlist_returns_403(_client):
    c = _client()
    r = c.post("/api/admin/services/sshd/restart", headers=_bearer())
    assert r.status_code == 403


def test_restart_service_calls_systemctl(monkeypatch, _client):
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/systemctl")

    class _R:
        returncode = 0
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())

    c = _client()
    r = c.post("/api/admin/services/kza-voice/restart", headers=_bearer())
    assert r.status_code == 200, r.json()
    assert r.json()["service"] == "kza-voice"


def test_restart_service_systemctl_failure_returns_500(monkeypatch, _client):
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/systemctl")

    class _R:
        returncode = 1
        stderr = "Failed to start"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _R())

    c = _client()
    r = c.post("/api/admin/services/kza-voice/restart", headers=_bearer())
    assert r.status_code == 500


# ---------------- Fail closed without a configured token ----------------

def test_no_token_configured_rejects_without_creating_user(monkeypatch, fake_user_manager, _client):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "Open", "permission_level": "adult"})
    assert r.status_code == 503
    assert fake_user_manager._version == 0 and not fake_user_manager.data_path.exists()
    assert fake_user_manager._users == {}
