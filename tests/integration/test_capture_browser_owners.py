"""Actual global/browser and ASGI upload owners with synthetic input leaves."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, UploadFile
import httpx
import pytest
from starlette.datastructures import FormData

from src.dashboard.admin import register_admin_routes
from src.dashboard.auth import require_dashboard_access
from src.dashboard.authority import DashboardAuthority
from src.dashboard.capture_browser_api import register_capture_browser_routes
from src.dashboard.capture_ingress import CaptureIngressMiddleware
from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled


pytestmark = pytest.mark.asyncio
ORIGIN = {"Origin": "http://test"}


@asynccontextmanager
async def browser_app(control, tmp_path, *, admin=True):
    authority = DashboardAuthority(str(tmp_path / "browser-authority.sqlite"))
    await authority.initialize()
    app = FastAPI(dependencies=[Depends(require_dashboard_access)])
    app.state.dashboard_authority = authority
    if admin:
        register_admin_routes(app)
        control.bind_enrollment_work(app.state.dashboard_admin_work)
    register_capture_browser_routes(app, control)
    app.add_middleware(CaptureIngressMiddleware, control=control)
    try:
        yield app
    finally:
        work = getattr(app.state, "dashboard_admin_work", None)
        if work is not None:
            await asyncio.shield(work.begin_close())
        await authority.aclose()


async def test_active_browser_retirement_after_logout_cannot_unmute(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            login = await client.post("/api/admin/auth/login", headers=ORIGIN,
                                      json={"token": dashboard_auth["Authorization"].split(" ", 1)[1]})
            assert login.status_code == 200
            registered = await client.post("/api/capture/recorders/register", headers=ORIGIN)
            assert registered.status_code == 200
            token = registered.json()["token"]
            active = await client.post("/api/capture/recorders/activate", headers=ORIGIN, json={"token": token})
            assert active.status_code == 200 and active.json()["capture_allowed"] is True
            try:
                logout = await client.post("/api/admin/auth/logout", headers=ORIGIN)
                assert logout.status_code == 200
                denied = await client.post("/api/capture/recorders/poll", headers=ORIGIN, json={"token": token})
                assert denied.status_code == 401
                g.control.request(True, g.control.snapshot()["revision"], actor())
                await until(lambda: g.control.snapshot()["pending_browser_recorders"] > 0)
                assert g.control.snapshot()["total_mute_confirmed"] is False
                forged = await client.post("/api/capture/recorders/retire", headers=ORIGIN,
                                            json={"token": token, "muted": False})
                assert forged.status_code == 422
                no_origin = await client.post("/api/capture/recorders/retire", json={"token": token})
                assert no_origin.status_code == 403
                pending = g.control.snapshot()["pending_browser_recorders"]
                assert pending > 0
                foreign = await client.post("/api/capture/recorders/retire",
                    headers={"Origin": "https://other.invalid"}, json={"token": token})
                assert foreign.status_code == 403
                assert g.control.snapshot()["pending_browser_recorders"] == pending
                assert g.control.snapshot()["total_mute_confirmed"] is False
                retired = await client.post("/api/capture/recorders/retire", headers=ORIGIN, json={"token": token})
                assert retired.status_code == 200 and retired.json() == {"retired": True}
                snapshot = await capture_phase(g.control, "muted")
                assert snapshot["total_mute_confirmed"] is True
                assert snapshot["persisted_muted"] is True
                denied = await client.post("/api/capture/recorders/activate", headers=ORIGIN, json={"token": token})
                assert denied.status_code == 401
            finally:
                await g.control.recorders.retire(token)


async def test_restart_unknown_owner_is_not_a_stop_ack_or_shutdown_block(tmp_path):
    path = str(tmp_path / "browser-private.sqlite")
    first = PrivacyServices(path)
    await first.acall(first.initialize)
    registry = CaptureRecorders(first, capture_allowed=lambda: True)
    await registry.initialize()
    principal = actor()
    reserved = await registry.register(principal)
    issued = await registry.register(principal)
    await registry.activate(issued["token"], principal)
    try:
        await asyncio.wait_for(registry.aclose(), 2)
        assert registry.pending() == 2
    finally:
        await first.aclose()
    second = PrivacyServices(path)
    await second.acall(second.initialize)
    recovered = CaptureRecorders(second, capture_allowed=lambda: True)
    try:
        await asyncio.wait_for(recovered.initialize(), 2)
        status = await recovered.poll(issued["token"], actor())
        assert status["state"] == "unknown" and status["capture_allowed"] is False
        assert recovered.pending() == 2
        draining = asyncio.create_task(recovered.drain())
        try:
            await until(lambda: recovered.pending() == 1)
            assert not draining.done()
            assert (await recovered.poll(reserved["token"], actor()))["state"] == "retired"
            await recovered.retire(issued["token"])
            await asyncio.wait_for(asyncio.shield(draining), 2)
            assert recovered.pending() == 0
        finally:
            await recovered.retire(issued["token"])
            await asyncio.gather(draining, return_exceptions=True)
    finally:
        await recovered.aclose()
        await second.aclose()


async def test_cancelled_upload_observer_does_not_release_actual_parser_cleanup(
        tmp_path, monkeypatch, dashboard_auth):
    entered, release = asyncio.Event(), asyncio.Event()
    closed_files = []
    original = FormData.close

    async def held_close(form):
        await original(form)
        closed_files.extend(value.file.closed for _, value in form.multi_items()
                            if isinstance(value, UploadFile) or hasattr(value, "file"))
        entered.set()
        await release.wait()

    monkeypatch.setattr(FormData, "close", held_close)
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path, admin=False) as app:
        @app.post("/api/admin/users/synthetic/enroll")
        async def enroll(sample: UploadFile = File(...)):
            assert await sample.read() == b"synthetic PCM bytes"
            return {"ok": True}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test",
                                    headers={**dashboard_auth, **ORIGIN}) as client:
            observer = asyncio.create_task(client.post("/api/admin/users/synthetic/enroll",
                files={"sample": ("synthetic.wav", b"synthetic PCM bytes", "audio/wav")}))
            owner = None
            try:
                await asyncio.wait_for(entered.wait(), 3)
                assert closed_files and all(closed_files)
                lease, = g.control._uploads
                owner = lease.task
                assert owner is not observer and not owner.done()
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert not owner.done() and owner.cancelling() == 0
                g.control.request(True, g.control.snapshot()["revision"], actor())
                await until(lambda: g.control.snapshot()["phase"] == "muting")
                assert g.control.snapshot()["total_mute_confirmed"] is False
                assert lease in g.control._uploads and not owner.done()
                release.set()
                await asyncio.wait_for(asyncio.shield(owner), 3)
                snapshot = await capture_phase(g.control, "muted")
                assert snapshot["total_mute_confirmed"] is True
                assert lease not in g.control._uploads and owner.done()
            finally:
                release.set()
                await asyncio.gather(observer, return_exceptions=True)
                if owner is not None:
                    await asyncio.gather(owner, return_exceptions=True)


async def test_mute_interrupts_actual_prebuffer_without_starting_the_parser(
        tmp_path, monkeypatch, dashboard_auth):
    reading, left = asyncio.Event(), asyncio.Event()
    parser_calls = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first bounded bytes"
            reading.set()
            try:
                await asyncio.Event().wait()
            finally:
                left.set()

    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path, admin=False) as app:
        @app.post("/api/admin/users/synthetic/enroll")
        async def enroll(sample: UploadFile = File(...)):
            parser_calls.append(sample)
            return {"ok": True}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test",
                                    headers={**dashboard_auth, **ORIGIN}) as client:
            observer = asyncio.create_task(client.post("/api/admin/users/synthetic/enroll",
                content=Body(), headers={"content-type": "multipart/form-data; boundary=synthetic"}))
            await asyncio.wait_for(reading.wait(), 3)
            lease, = g.control._uploads
            owner = lease.task
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert not owner.done()
            g.control.request(True, g.control.snapshot()["revision"], actor())
            await asyncio.wait_for(left.wait(), 3)
            await asyncio.wait_for(asyncio.shield(owner), 3)
            assert parser_calls == []
            assert (await capture_phase(g.control, "muted"))["total_mute_confirmed"] is True
            assert lease not in g.control._uploads
