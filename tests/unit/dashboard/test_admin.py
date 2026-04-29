"""Tests para /api/admin/* — auth, users CRUD, enroll, alerts, services."""

import io
import os
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dashboard.admin import register_admin_routes
from src.users.user_manager import PermissionLevel, User


TEST_TOKEN = "test-secret-token-1234567890"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", TEST_TOKEN)


@pytest.fixture
def fake_user_manager():
    um = MagicMock()
    um._users = {}

    def add_user(name, permission_level, voice_embedding=None, requesting_user=None):
        if any(u.name == name for u in um._users.values()):
            return None, f"Ya existe usuario {name}"
        uid = f"user_{name.lower()}_1"
        u = User(user_id=uid, name=name, permission_level=permission_level,
                 voice_embedding=voice_embedding)
        um._users[uid] = u
        return u, "created"

    def get_user(uid):
        return um._users.get(uid)

    def remove_user(uid, requesting_user=None):
        if uid in um._users:
            del um._users[uid]
            return True, "removed"
        return False, "no existe"

    um.add_user.side_effect = add_user
    um.get_user.side_effect = get_user
    um.remove_user.side_effect = remove_user
    um._save = MagicMock()
    return um


@pytest.fixture
def fake_speaker_identifier():
    si = MagicMock()
    si.create_enrollment_embedding.return_value = np.zeros(192, dtype=np.float32)
    return si


@pytest.fixture
def fake_alert_manager():
    from src.alerts.alert_manager import Alert, AlertPriority, AlertType
    a = Alert(alert_id="al_test", alert_type=AlertType.SECURITY,
              priority=AlertPriority.MEDIUM, message="test")
    am = MagicMock()
    am.get_alert.side_effect = lambda i: a if i == "al_test" else None
    return am


def _client(**kwargs):
    app = FastAPI()
    register_admin_routes(app, **kwargs)
    return TestClient(app)


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

def test_admin_endpoint_without_token_returns_401(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users", json={"name": "X", "permission_level": "adult"})
    assert r.status_code == 401


def test_admin_endpoint_with_bad_token_returns_401(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "X", "permission_level": "adult"},
               headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_admin_endpoint_with_correct_token_passes(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "Tomás", "permission_level": "adult"},
               headers=_bearer())
    assert r.status_code == 200
    assert r.json()["name"] == "Tomás"


def test_login_sets_cookie(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/auth/login", json={"token": TEST_TOKEN})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "kza_dashboard_token" in r.cookies


def test_login_with_wrong_token_returns_401(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/auth/login", json={"token": "wrong"})
    assert r.status_code == 401


def test_cookie_grants_access_to_admin_endpoints(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    c.post("/api/admin/auth/login", json={"token": TEST_TOKEN})
    r = c.get("/api/admin/auth/whoami")
    assert r.status_code == 200


# ---------------- Users CRUD ----------------

def test_create_user_with_invalid_permission_level(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "X", "permission_level": "superadmin"},
               headers=_bearer())
    assert r.status_code == 400


def test_create_duplicate_user_returns_409(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    c.post("/api/admin/users",
           json={"name": "Ana", "permission_level": "adult"}, headers=_bearer())
    r = c.post("/api/admin/users",
               json={"name": "Ana", "permission_level": "adult"}, headers=_bearer())
    assert r.status_code == 409


def test_delete_user(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    cr = c.post("/api/admin/users",
                json={"name": "Bob", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    r = c.delete(f"/api/admin/users/{uid}", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_delete_unknown_user_returns_404(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    r = c.delete("/api/admin/users/nonexistent", headers=_bearer())
    assert r.status_code == 404


def test_update_permissions(fake_user_manager):
    c = _client(user_manager=fake_user_manager)
    cr = c.post("/api/admin/users",
                json={"name": "Eva", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    r = c.put(f"/api/admin/users/{uid}/permissions",
              json={"permission_level": "admin"}, headers=_bearer())
    assert r.status_code == 200
    assert r.json()["permission_level"] == "ADMIN"


# ---------------- Enrollment ----------------

def test_enroll_requires_3_samples(fake_user_manager, fake_speaker_identifier):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    cr = c.post("/api/admin/users",
                json={"name": "Lía", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    files = [("samples", ("s1.wav", _wav_bytes(), "audio/wav"))]
    r = c.post(f"/api/admin/users/{uid}/enroll", files=files, headers=_bearer())
    assert r.status_code == 400


def test_enroll_with_3_samples_creates_embedding(fake_user_manager, fake_speaker_identifier):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    cr = c.post("/api/admin/users",
                json={"name": "Lía", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    files = [
        ("samples", (f"s{i}.wav", _wav_bytes(), "audio/wav"))
        for i in range(3)
    ]
    r = c.post(f"/api/admin/users/{uid}/enroll", files=files, headers=_bearer())
    assert r.status_code == 200, r.json()
    assert r.json()["samples"] == 3
    assert r.json()["embedding_dim"] == 192
    fake_speaker_identifier.create_enrollment_embedding.assert_called_once()
    call_arg = fake_speaker_identifier.create_enrollment_embedding.call_args[0][0]
    assert len(call_arg) == 3
    assert all(isinstance(a, np.ndarray) for a in call_arg)
    assert all(a.dtype == np.float32 for a in call_arg)


def test_enroll_unknown_user_returns_404(fake_user_manager, fake_speaker_identifier):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    files = [("samples", (f"s{i}.wav", _wav_bytes(), "audio/wav")) for i in range(3)]
    r = c.post("/api/admin/users/nonexistent/enroll", files=files, headers=_bearer())
    assert r.status_code == 404


def test_enroll_with_invalid_audio_returns_400(fake_user_manager, fake_speaker_identifier):
    c = _client(user_manager=fake_user_manager, speaker_identifier=fake_speaker_identifier)
    cr = c.post("/api/admin/users",
                json={"name": "Z", "permission_level": "adult"}, headers=_bearer())
    uid = cr.json()["user_id"]
    files = [("samples", (f"s{i}.wav", b"not-a-wav", "audio/wav")) for i in range(3)]
    r = c.post(f"/api/admin/users/{uid}/enroll", files=files, headers=_bearer())
    assert r.status_code == 400


# ---------------- Alerts ----------------

def test_alert_ack(fake_alert_manager):
    c = _client(alert_manager=fake_alert_manager)
    r = c.post("/api/admin/alerts/al_test/ack", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["alert_id"] == "al_test"


def test_alert_ack_unknown_returns_404(fake_alert_manager):
    c = _client(alert_manager=fake_alert_manager)
    r = c.post("/api/admin/alerts/al_unknown/ack", headers=_bearer())
    assert r.status_code == 404


def test_alert_dismiss(fake_alert_manager):
    c = _client(alert_manager=fake_alert_manager)
    r = c.delete("/api/admin/alerts/al_test", headers=_bearer())
    assert r.status_code == 200
    assert r.json()["dismissed"] is True


# ---------------- Services ----------------

def test_restart_service_not_in_allowlist_returns_403():
    c = _client()
    r = c.post("/api/admin/services/sshd/restart", headers=_bearer())
    assert r.status_code == 403


def test_restart_service_calls_systemctl(monkeypatch):
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


def test_restart_service_systemctl_failure_returns_500(monkeypatch):
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


# ---------------- Public mode (sin token configurado) ----------------

def test_no_token_configured_means_open_access(monkeypatch, fake_user_manager):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    c = _client(user_manager=fake_user_manager)
    r = c.post("/api/admin/users",
               json={"name": "Open", "permission_level": "adult"})
    assert r.status_code == 200
