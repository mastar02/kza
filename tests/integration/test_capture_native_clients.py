"""Native calibration leases share durable retirement without browser identity."""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from src.memory.capture_recorders import CaptureRecorders
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_browser_owners import ORIGIN, browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled


pytestmark = pytest.mark.asyncio
NATIVE = "/api/capture/native/calibration"
TRAINING = "/api/capture/native/wakeword-training"
DIAGNOSTICS = "/api/capture/native/wakeword-diagnostics"
BROWSER = "/api/capture/recorders"


async def test_native_reservation_and_namespace_do_not_grant_browser_capture(tmp_path):
    privacy = PrivacyServices(str(tmp_path / "native.sqlite"))
    await privacy.acall(privacy.initialize)
    recorders = CaptureRecorders(privacy, capture_allowed=lambda: True)
    await recorders.initialize()
    issued = []
    try:
        principal = actor()
        for kind in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics"):
            item = await recorders.register(principal, kind=kind)
            issued.append((item["token"], kind))
            if kind == "browser":
                assert "kind" not in item
            else:
                assert item["kind"] == kind
            assert item["capture_allowed"] is False
            assert recorders.pending(kind=kind) == 1
        for token, actual_kind in issued:
            for wrong_kind in ("browser", "calibration", "wakeword_training", "wakeword_diagnostics"):
                if wrong_kind == actual_kind:
                    continue
                with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                    await recorders.activate(token, principal, kind=wrong_kind)
                with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                    await recorders.poll(token, principal, kind=wrong_kind)
                with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                    await recorders.retire(token, kind=wrong_kind)
        assert all(recorders.pending(kind=kind) == 1 for _, kind in issued)
        for token, kind in issued:
            activated = await recorders.activate(token, principal, kind=kind)
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


async def test_training_and_calibration_share_ack_barrier_but_not_capabilities(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            credentials = {"token": dashboard_auth["Authorization"].split(" ", 1)[1]}
            assert (await client.post("/api/admin/auth/login", headers=ORIGIN, json=credentials)).status_code == 200
            tokens = []
            try:
                for route, kind in ((NATIVE, "calibration"), (TRAINING, "wakeword_training")):
                    reserved = await client.post(route + "/register", headers=ORIGIN, json={})
                    assert reserved.status_code == 200 and reserved.json()["kind"] == kind
                    assert reserved.json()["capture_allowed"] is False
                    assert reserved.headers["Cache-Control"] == "no-store"
                    token = reserved.json()["token"]
                    tokens.append((route, kind, token))
                    active = await client.post(route + "/activate", headers=ORIGIN, json={"token": token})
                    assert active.status_code == 200 and active.json()["capture_allowed"] is True
                state = g.control.snapshot()
                assert state["pending_native_clients"] == 2 and state["pending_browser_recorders"] == 0
                for source_route, _, token in tokens:
                    wrong_route = TRAINING if source_route == NATIVE else NATIVE
                    for action in ("activate", "poll", "retire"):
                        denied = await client.post(wrong_route + "/" + action, headers=ORIGIN, json={"token": token})
                        assert denied.status_code == 409
                assert g.control.snapshot()["pending_native_clients"] == 2

                assert (await client.post("/api/admin/auth/logout", headers=ORIGIN)).status_code == 200
                for route, _, token in tokens:
                    for action in ("activate", "poll"):
                        assert (await client.post(route + "/" + action, headers=ORIGIN, json={"token": token})).status_code == 401
                # A new valid session is not the originally registered actor.
                assert (await client.post("/api/admin/auth/login", headers=ORIGIN, json=credentials)).status_code == 200
                for route, _, token in tokens:
                    poll = await client.post(route + "/poll", headers=ORIGIN, json={"token": token})
                    assert poll.status_code == 200 and poll.json()["capture_allowed"] is False
                    assert poll.json()["retirement_required"] is True
                    assert (await client.post(route + "/activate", headers=ORIGIN, json={"token": token})).status_code == 409
                assert (await client.post("/api/admin/auth/logout", headers=ORIGIN)).status_code == 200

                g.control.request(True, g.control.snapshot()["revision"], actor())
                await until(lambda: g.control.snapshot()["phase"] == "muting")
                assert not g.control.snapshot()["total_mute_confirmed"]
                training_token = tokens[1][2]
                assert (await client.post(TRAINING + "/retire", json={"token": training_token})).status_code == 403
                assert (await client.post(TRAINING + "/retire", headers={"Origin": "https://other.invalid"},
                                          json={"token": training_token})).status_code == 403
                assert (await client.post(TRAINING + "/retire", headers=ORIGIN,
                                          json={"token": training_token, "muted": False})).status_code == 422
                assert g.control.snapshot()["pending_native_clients"] == 2
                calibration_token = tokens[0][2]
                assert (await client.post(NATIVE + "/retire", headers=ORIGIN,
                                          json={"token": calibration_token})).status_code == 200
                assert g.control.snapshot()["pending_native_clients"] == 1
                assert not g.control.snapshot()["total_mute_confirmed"]
                retired = await client.post(TRAINING + "/retire", headers=ORIGIN, json={"token": training_token})
                assert retired.status_code == 200 and retired.json() == {"retired": True}
                state = await capture_phase(g.control, "muted")
                assert state["total_mute_confirmed"] and state["persisted_muted"]
                assert state["pending_native_clients"] == state["pending_browser_recorders"] == 0
            finally:
                for _, kind, token in tokens:
                    await g.control.recorders.retire(token, kind=kind)


async def test_restart_keeps_training_unknown_until_its_own_retirement(tmp_path):
    path = str(tmp_path / "both-native.sqlite")
    first = PrivacyServices(path)
    await first.acall(first.initialize)
    registry = CaptureRecorders(first, capture_allowed=lambda: True)
    await registry.initialize()
    principal = actor()
    calibration = await registry.register(principal, kind="calibration")
    training = await registry.register(principal, kind="wakeword_training")
    reserved = await registry.register(principal, kind="wakeword_training")
    await registry.activate(calibration["token"], principal, kind="calibration")
    await registry.activate(training["token"], principal, kind="wakeword_training")
    try:
        await registry.aclose()
        assert registry.pending(kind="calibration") == 1 and registry.pending(kind="wakeword_training") == 2
    finally:
        await first.aclose()

    second = PrivacyServices(path)
    await second.acall(second.initialize)
    recovered = CaptureRecorders(second, capture_allowed=lambda: True)
    draining = None
    try:
        await recovered.initialize()
        for entry, kind in ((calibration, "calibration"), (training, "wakeword_training")):
            state = await recovered.poll(entry["token"], actor(), kind=kind)
            assert state["state"] == "unknown" and not state["capture_allowed"]
        draining = asyncio.create_task(recovered.drain())
        await until(lambda: recovered.pending() == 2)
        assert not draining.done()
        assert (await recovered.poll(reserved["token"], actor(), kind="wakeword_training"))["state"] == "retired"
        await recovered.retire(calibration["token"], kind="calibration")
        assert recovered.pending(kind="wakeword_training") == 1 and not draining.done()
        with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
            await recovered.retire(training["token"], kind="calibration")
        assert recovered.pending() == 1
        await recovered.retire(training["token"], kind="wakeword_training")
        await asyncio.wait_for(asyncio.shield(draining), 2)
        assert recovered.pending() == 0
    finally:
        await recovered.retire(calibration["token"], kind="calibration")
        await recovered.retire(training["token"], kind="wakeword_training")
        if draining is not None:
            await asyncio.gather(draining, return_exceptions=True)
        await recovered.aclose()
        await second.aclose()


async def test_training_registration_rejects_client_kind_and_muted_admission(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            logged = await client.post("/api/admin/auth/login", headers=ORIGIN,
                json={"token": dashboard_auth["Authorization"].split(" ", 1)[1]})
            assert logged.status_code == 200
            for body in ({"kind": "calibration"}, {"kind": "wakeword_training"}, {"muted": False}, {"capture_allowed": True}):
                rejected = await client.post(TRAINING + "/register", headers=ORIGIN, json=body)
                assert rejected.status_code == 422 and rejected.headers["Cache-Control"] == "no-store"
            assert g.control.snapshot()["pending_native_clients"] == 0
            g.control.request(True, g.control.snapshot()["revision"], actor())
            await capture_phase(g.control, "muted")
            denied = await client.post(TRAINING + "/register", headers=ORIGIN, json={})
            assert denied.status_code == 409 and denied.headers["Cache-Control"] == "no-store"
            assert g.control.snapshot()["total_mute_confirmed"]


async def test_diagnostics_has_its_own_capability_and_three_native_ack_barrier(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            credentials = {"token": dashboard_auth["Authorization"].split(" ", 1)[1]}
            assert (await client.post("/api/admin/auth/login", headers=ORIGIN, json=credentials)).status_code == 200
            tokens = []
            try:
                for body in ({"kind": "calibration"}, {"kind": "wakeword_training"}, {"kind": "wakeword_diagnostics"}, {"capture_allowed": True}):
                    denied = await client.post(DIAGNOSTICS + "/register", headers=ORIGIN, json=body)
                    assert denied.status_code == 422 and denied.headers["Cache-Control"] == "no-store"
                for route, kind in ((NATIVE, "calibration"), (TRAINING, "wakeword_training"),
                                    (DIAGNOSTICS, "wakeword_diagnostics")):
                    issued = await client.post(route + "/register", headers=ORIGIN, json={})
                    assert issued.status_code == 200 and issued.json()["kind"] == kind
                    assert issued.json()["capture_allowed"] is False
                    token = issued.json()["token"]
                    tokens.append((route, kind, token))
                    active = await client.post(route + "/activate", headers=ORIGIN, json={"token": token})
                    assert active.status_code == 200 and active.json()["capture_allowed"] is True
                    assert active.json()["kind"] == kind
                assert g.control.snapshot()["pending_native_clients"] == 3
                assert g.control.snapshot()["pending_browser_recorders"] == 0
                for source, _, token in tokens:
                    for target, _, _peer_token in tokens:
                        if target == source:
                            continue
                        for action in ("activate", "poll", "retire"):
                            response = await client.post(target + "/" + action, headers=ORIGIN, json={"token": token})
                            assert response.status_code == 409
                assert g.control.snapshot()["pending_native_clients"] == 3

                diagnostic_token = tokens[2][2]
                assert (await client.post("/api/admin/auth/logout", headers=ORIGIN)).status_code == 200
                assert (await client.post(DIAGNOSTICS + "/register", headers=ORIGIN, json={})).status_code == 401
                assert (await client.post(DIAGNOSTICS + "/poll", headers=ORIGIN,
                                          json={"token": diagnostic_token})).status_code == 401
                assert (await client.post("/api/admin/auth/login", headers=ORIGIN, json=credentials)).status_code == 200
                poll = await client.post(DIAGNOSTICS + "/poll", headers=ORIGIN, json={"token": diagnostic_token})
                assert poll.status_code == 200 and poll.json()["capture_allowed"] is False
                assert poll.json()["retirement_required"] is True
                assert (await client.post(DIAGNOSTICS + "/activate", headers=ORIGIN,
                                          json={"token": diagnostic_token})).status_code == 409
                assert (await client.post("/api/admin/auth/logout", headers=ORIGIN)).status_code == 200
                g.control.request(True, g.control.snapshot()["revision"], actor())
                await until(lambda: g.control.snapshot()["phase"] == "muting")
                assert not g.control.snapshot()["total_mute_confirmed"]
                assert (await client.post(DIAGNOSTICS + "/retire",
                                          json={"token": diagnostic_token})).status_code == 403
                assert (await client.post(DIAGNOSTICS + "/retire", headers={"Origin": "https://other.invalid"},
                                          json={"token": diagnostic_token})).status_code == 403
                assert (await client.post(DIAGNOSTICS + "/retire", headers=ORIGIN,
                                          json={"token": diagnostic_token, "muted": False})).status_code == 422
                for expected, (route, _kind, token) in zip((2, 1, 0), tokens):
                    retired = await client.post(route + "/retire", headers=ORIGIN, json={"token": token})
                    assert retired.status_code == 200 and retired.json() == {"retired": True}
                    assert g.control.snapshot()["pending_native_clients"] == expected
                    assert g.control.snapshot()["pending_browser_recorders"] == 0
                    if expected:
                        assert not g.control.snapshot()["total_mute_confirmed"]
                state = await capture_phase(g.control, "muted")
                assert state["total_mute_confirmed"] and state["persisted_muted"]
                denied = await client.post(DIAGNOSTICS + "/register", headers=ORIGIN, json={})
                assert denied.status_code == 401
            finally:
                for _, kind, token in tokens:
                    await g.control.recorders.retire(token, kind=kind)


async def test_restart_preserves_three_native_unknown_kinds_until_separate_receipts(tmp_path):
    path = str(tmp_path / "three-native.sqlite")
    first = PrivacyServices(path)
    await first.acall(first.initialize)
    registry = CaptureRecorders(first, capture_allowed=lambda: True)
    issued = []
    try:
        await registry.initialize()
        principal = actor()
        for kind in ("calibration", "wakeword_training", "wakeword_diagnostics"):
            item = await registry.register(principal, kind=kind)
            issued.append((item["token"], kind))
            await registry.activate(item["token"], principal, kind=kind)
        reserved = await registry.register(principal, kind="wakeword_diagnostics")
        await registry.aclose()
        assert registry.pending(kind="wakeword_diagnostics") == 2
    finally:
        try:
            await registry.aclose()
        finally:
            await first.aclose()

    second = PrivacyServices(path)
    await second.acall(second.initialize)
    recovered = CaptureRecorders(second, capture_allowed=lambda: True)
    draining = None
    try:
        await recovered.initialize()
        for token, kind in issued:
            state = await recovered.poll(token, actor(), kind=kind)
            assert state["state"] == "unknown" and state["capture_allowed"] is False
        assert recovered.pending(kind="browser") == 0
        draining = asyncio.create_task(recovered.drain())
        await until(lambda: recovered.pending() == 3)
        assert not draining.done()
        assert (await recovered.poll(reserved["token"], actor(), kind="wakeword_diagnostics"))["state"] == "retired"
        for expected, (token, kind) in zip((2, 1, 0), issued):
            wrong_kind = "calibration" if kind != "calibration" else "wakeword_diagnostics"
            with pytest.raises(ValueError, match="capture_recorder_kind_denied"):
                await recovered.retire(token, kind=wrong_kind)
            assert recovered.pending() == expected + 1
            await recovered.retire(token, kind=kind)
            assert recovered.pending() == expected
            if expected:
                assert not draining.done()
        await asyncio.wait_for(asyncio.shield(draining), 2)
        assert recovered.pending() == 0
    finally:
        for token, kind in issued:
            await recovered.retire(token, kind=kind)
        if draining is not None:
            await asyncio.gather(draining, return_exceptions=True)
        await recovered.aclose()
        await second.aclose()


@pytest.mark.parametrize("role", [DashboardRole.HOME, DashboardRole.EVALUATOR])
async def test_diagnostics_does_not_turn_household_or_voice_permission_into_admin(tmp_path, monkeypatch, dashboard_auth, role):
    async with controlled(tmp_path, monkeypatch) as g, browser_app(g.control, tmp_path) as app:
        before = g.control.snapshot()
        now = time.time()
        samples = (EvaluationSample("vf_" + "a" * 64, frozenset({"b" * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        authority = app.state.dashboard_authority
        grant = await authority.issue_credential(
            authority.resolve(dashboard_auth["Authorization"]),
            subject="synthetic-diagnostic-member", role=role, expires_at=now + 120, samples=samples,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            logged = await client.post("/api/admin/auth/login", headers=ORIGIN, json={
                "token": grant.token, "role": "operator", "permissions": ["admin:manage"],
                "user_id": "synthetic-untrusted-person",
            })
            assert logged.status_code == 200
            for action, body in (("register", {}), ("activate", {"token": "a" * 43}),
                                 ("poll", {"token": "a" * 43})):
                denied = await client.post(DIAGNOSTICS + "/" + action, headers=ORIGIN, json=body)
                assert denied.status_code == 403
        assert g.control.snapshot() == before
