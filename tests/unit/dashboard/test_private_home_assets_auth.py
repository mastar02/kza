"""Security tests for authenticated private NEXO home assets and Uvicorn proxy trust."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample

PRIVATE_DIR = Path(__file__).resolve().parents[3] / "src/dashboard/frontend/home/nexo/private"


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_private_nexo_assets_require_auth_and_home_read_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "operator-bootstrap-token")
    authority = DashboardAuthority(tmp_path / "auth.db")
    expires_at = time.time() + 3600.0
    try:
        await authority.synchronize()
        actor = authority.resolve("Bearer operator-bootstrap-token")
        assert actor is not None
        home_cred = await authority.issue_credential(
            actor,
            subject="hogar",
            role=DashboardRole.HOME,
            expires_at=expires_at,
        )
        eval_sample = EvaluationSample(
            fingerprint_id="vf_" + "a" * 64,
            source_digests=frozenset({"b" * 64}),
            modalities=frozenset({EvaluationModality.AUDIO_PREVIEW}),
            expires_at=expires_at,
        )
        eval_cred = await authority.issue_credential(
            actor,
            subject="eval",
            role=DashboardRole.EVALUATOR,
            expires_at=expires_at,
            samples=(eval_sample,),
        )

        casa_exists = (PRIVATE_DIR / "casa.glb").is_file()
        plan_exists = (PRIVATE_DIR / "house-plan.js").is_file()

        with TestClient(DashboardAPI(home_enabled=True, authority=authority).app) as client:
            for path in ("/home/nexo/private/casa.glb", "/home/nexo/private/house-plan.js", "/home/nexo/private"):
                unauth_get = client.get(path)
                assert unauth_get.status_code == 401, path
                assert unauth_get.headers.get("cache-control") == "no-store", path

                unauth_head = client.head(path)
                assert unauth_head.status_code == 401, path
                assert unauth_head.headers.get("cache-control") == "no-store", path

                eval_get = client.get(path, headers=_auth_headers(eval_cred.token))
                assert eval_get.status_code == 403, path
                assert eval_get.headers.get("cache-control") == "no-store", path

            # Home role has home:read and receives private assets with no-store.
            casa = client.get("/home/nexo/private/casa.glb", headers=_auth_headers(home_cred.token))
            assert casa.status_code == (200 if casa_exists else 404)
            assert casa.headers.get("cache-control") == "no-store"
            if casa_exists:
                assert casa.headers.get("content-type", "").startswith("model/gltf-binary")

            plan = client.get("/home/nexo/private/house-plan.js", headers=_auth_headers(home_cred.token))
            assert plan.status_code == (200 if plan_exists else 404)
            assert plan.headers.get("cache-control") == "no-store"
            if plan_exists:
                assert "javascript" in plan.headers.get("content-type", "")

            # Path traversal cannot escape private/ even with valid home:read.
            for traversal in (
                "/home/nexo/private/..%2fhouse.js",
                "/home/nexo/private/%2e%2e/house.js",
                "/home/nexo/private/missing-file.glb",
            ):
                resp = client.get(traversal, headers=_auth_headers(home_cred.token))
                assert resp.status_code == 404, traversal
                assert resp.headers.get("cache-control") == "no-store", traversal
    finally:
        await authority.aclose()


def test_private_nexo_assets_blocked_when_home_disabled_or_auth_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    with TestClient(DashboardAPI(home_enabled=False).app) as client:
        for path in ("/home/nexo/private/casa.glb", "/home/nexo/private/house-plan.js"):
            unconfigured = client.get(path)
            assert unconfigured.status_code == 503, path
            assert unconfigured.headers.get("cache-control") == "no-store", path

    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "operator-secret")
    with TestClient(DashboardAPI(home_enabled=False).app) as client:
        for path in ("/home/nexo/private/casa.glb", "/home/nexo/private/house-plan.js"):
            # Unauthenticated still gets 401; authenticated gets 404 when home is disabled.
            assert client.get(path).status_code == 401, path
            authed = client.get(path, headers=_auth_headers("operator-secret"))
            assert authed.status_code == 404, path
            assert authed.headers.get("cache-control") == "no-store", path


def test_uvicorn_config_pins_loopback_proxy_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _FakeConfig:
        def __init__(self, app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

    class _FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        async def serve(self):
            return None

    fake_uvicorn = SimpleNamespace(Config=_FakeConfig, Server=_FakeServer)
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)

    api = DashboardAPI(host="127.0.0.1", port=9500)
    asyncio.run(api.start())

    assert captured["kwargs"]["proxy_headers"] is True
    assert captured["kwargs"]["forwarded_allow_ips"] == "127.0.0.1"
