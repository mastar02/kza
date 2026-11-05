"""Native calibration leases share durable retirement without browser identity."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_browser_owners import ORIGIN, browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled


pytestmark = pytest.mark.asyncio
NATIVE = "/api/capture/native/calibration"
BROWSER = "/api/capture/recorders"


async def test_native_reservation_and_namespace_do_not_grant_browser_capture(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "native.sqlite"))
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    await recorders.initialize()
    issued = []
    try:
        principal = actor()
        browser = await recorders.register(principal)
        native = await recorders.register(principal, kind="calibration")
        issued.extend(((browser["token"], "browser"), (native["token"], "calibration")))
        assert "kind" not in browser and native["kind"] == "calibration"
        assert native["capture_allowed"] is False
        assert recorders.pending(kind="browser") == 1
        assert recorders.pending(kind="calibration") == 1
        for token, wrong_kind in ((browser["token"], "calibration"), (native["token"], "browser")):
            with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                await recorders.activate(token, principal, kind=wrong_kind)
            with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                await recorders.poll(token, principal, kind=wrong_kind)
            with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                await recorders.retire(token, kind=wrong_kind)
        assert recorders.pending(kind="browser") == recorders.pending(kind="calibration") == 1
        activated = await recorders.activate(native["token"], principal, kind="calibration")
        assert activated["state"] == "active" and activated["capture_allowed"] is True
    finally:
        for token, kind in issued:
            await recorders.retire(token, kind=kind)
        await recorders.aclose()
        await privacy.aclose()


async def test_native_original_actor_cannot_be_replaced_but_can_retire(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "lease.sqlite"))
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    await recorders.initialize()
    current = True
    principal = actor(lambda: current)
    token = None
    try:
        issued = await recorders.register(principal, kind="calibration")
        token = issued["token"]
        assert (await recorders.activate(token, principal, kind="calibration"))["capture_allowed"]
        current = False
        replacement = actor()
        state = await recorders.poll(token, replacement, kind="calibration")
        assert state["state"] == "active"
        assert state["retirement_required"] and not state["capture_allowed"]
        with pytest.raises(ValueError, match="capture_recorder_actor_retired"):
            await recorders.activate(token, replacement, kind="calibration")
        assert recorders.pending(kind="calibration") == 1
        assert await recorders.retire(token, kind="calibration") == {"retired": True}
        assert recorders.pending() == 0
    finally:
        if token is not None:
            await recorders.retire(token, kind="calibration")
        await recorders.aclose()
        await privacy.aclose()


async def test_existing_two_column_rows_migrate_as_browser_without_losing_barrier(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "legacy.sqlite"))
    await privacy.acall(privacy.initialize)
    token = "a" * 43

    def seed():
        with privacy.ledger._transaction() as db:
            db.execute("CREATE TABLE capture_browser_recorders (token_hash TEXT PRIMARY KEY, state TEXT NOT NULL)")
            db.execute("INSERT INTO capture_browser_recorders VALUES (?, 'active')", (CaptureRecorders._key(token),))

    await privacy.acall(seed)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    try:
        await recorders.initialize()
        state = await recorders.poll(token, actor())
        assert state["state"] == "unknown" and not state["capture_allowed"]
        assert recorders.pending(kind="browser") == 1
        assert recorders.pending(kind="calibration") == 0
        with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
            await recorders.retire(token, kind="calibration")
        assert recorders.pending() == 1
        await recorders.retire(token)
    finally:
        await recorders.retire(token)
        await recorders.aclose()
        await privacy.aclose()


async def test_native_active_restart_is_unknown_until_capability_retirement(tmp_path):
    path = str(tmp_path / "restart.sqlite")
    first = PrivacyServices(path)
    await first.acall(first.initialize)
    recorders = CaptureRecorders(first, capture_allowed=lambda: True)
    await recorders.initialize()
    principal = actor()
    reserved = await recorders.register(principal, kind="calibration")
    active = await recorders.register(principal, kind="calibration")
    await recorders.activate(active["token"], principal, kind="calibration")
    try:
        await recorders.aclose()
        assert recorders.pending(kind="calibration") == 2
    finally:
        await first.aclose()
    second = PrivacyServices(path)
    await second.acall(second.initialize)
    recovered = CaptureRecorders(second, capture_allowed=lambda: True)
    draining = None
    try:
        await recovered.initialize()
        state = await recovered.poll(active["token"], actor(), kind="calibration")
        assert state["state"] == "unknown" and not state["capture_allowed"]
        assert recovered.pending(kind="browser") == 0
        draining = asyncio.create_task(recovered.drain())
        await until(lambda: recovered.pending() == 1)
        assert not draining.done()
        assert (await recovered.poll(reserved["token"], actor(), kind="calibration"))["state"] == "retired"
        await recovered.retire(active["token"], kind="calibration")
        await asyncio.wait_for(asyncio.shield(draining), 2)
        assert recovered.pending() == 0
    finally:
        await recovered.retire(active["token"], kind="calibration")
        if draining is not None:
            await asyncio.gather(draining, return_exceptions=True)
        await recovered.aclose()
        await second.aclose()


async def test_native_http_retirement_after_logout_is_not_browser_or_unmute(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            logged = await client.post("/api/admin/auth/login", headers=ORIGIN,
                json={"token": dashboard_auth["Authorization"].split(" ", 1)[1]})
            assert logged.status_code == 200
            registered = await client.post(NATIVE + "/register", headers=ORIGIN, json={})
            assert registered.status_code == 200
            assert registered.headers["Cache-Control"] == "no-store"
            assert registered.json()["kind"] == "calibration"
            token = registered.json()["token"]
            try:
                state = g.control.snapshot()
                assert state["pending_native_clients"] == 1
                assert state["pending_browser_recorders"] == 0
                active = await client.post(NATIVE + "/activate", headers=ORIGIN, json={"token": token})
                assert active.status_code == 200 and active.json()["capture_allowed"] is True
                assert (await client.post("/api/admin/auth/logout", headers=ORIGIN)).status_code == 200
                assert (await client.post(NATIVE + "/poll", headers=ORIGIN, json={"token": token})).status_code == 401
                g.control.request(True, g.control.snapshot()["revision"], actor())
                await until(lambda: g.control.snapshot()["phase"] == "muting")
                assert not g.control.snapshot()["total_mute_confirmed"]
                assert g.control.snapshot()["pending_native_clients"] == 1
                assert g.control.snapshot()["pending_browser_recorders"] == 0
                assert (await client.post(NATIVE + "/retire", json={"token": token})).status_code == 403
                assert (await client.post(NATIVE + "/retire", headers={"Origin": "https://other.invalid"},
                                          json={"token": token})).status_code == 403
                assert (await client.post(BROWSER + "/retire", headers=ORIGIN, json={"token": token})).status_code == 409
                assert g.control.snapshot()["pending_native_clients"] == 1
                retired = await client.post(NATIVE + "/retire", headers=ORIGIN, json={"token": token})
                assert retired.status_code == 200 and retired.json() == {"retired": True}
                muted = await capture_phase(g.control, "muted")
                assert muted["total_mute_confirmed"] and muted["persisted_muted"]
                assert muted["pending_native_clients"] == muted["pending_browser_recorders"] == 0
                denied = await client.post(NATIVE + "/activate", headers=ORIGIN, json={"token": token})
                assert denied.status_code == 401
            finally:
                await g.control.recorders.retire(token, kind="calibration")


async def test_native_registration_kind_and_policy_are_not_client_claims(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            logged = await client.post("/api/admin/auth/login", headers=ORIGIN,
                json={"token": dashboard_auth["Authorization"].split(" ", 1)[1]})
            assert logged.status_code == 200
            for body in ({"kind": "browser"}, {"kind": "calibration"}, {"muted": False}, {"capture_allowed": True}):
                rejected = await client.post(NATIVE + "/register", headers=ORIGIN, json=body)
                assert rejected.status_code == 422
            assert g.control.snapshot()["pending_native_clients"] == 0
            assert g.control.snapshot()["pending_browser_recorders"] == 0
            g.control.request(True, g.control.snapshot()["revision"], actor())
            await capture_phase(g.control, "muted")
            denied = await client.post(NATIVE + "/register", headers=ORIGIN, json={})
            assert denied.status_code == 409
            assert g.control.snapshot()["total_mute_confirmed"]
