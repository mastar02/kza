"""Routine proposal review on the real dashboard, scheduler and erasure graph."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole
from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.routines.models import ScheduledRoutine, parse_routine_spec
from src.routines.proposals import ProposalOperation
from tests.integration.test_capability_application import configuration
from tests.integration.test_dashboard_sessions import leaves as application_leaves
from tests.integration.test_routine_home_denial import login
from tests.integration.test_routine_session_authority import definition
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio
leaves = application_leaves
PATH = "/api/routines/proposals"


@asynccontextmanager
async def application(config, leaves, dashboard_auth, *, buffer_http=False):
    async with AsyncExitStack() as stack:
        graph = await build_application(config, shutdown=stack, factories=leaves)
        if buffer_http:
            @graph.dashboard.app.middleware("http")
            async def buffer(request, call_next):
                return await call_next(request)

        async with AsyncClient(
            transport=ASGITransport(app=graph.dashboard.app), base_url="http://testserver",
        ) as client:
            token = dashboard_auth["Authorization"].split(" ", 1)[1]
            assert (await client.post("/api/admin/auth/login", json={"token": token})).status_code == 200
            yield graph, client


@pytest.fixture
async def review(leaves, tmp_path, dashboard_auth, request):
    transport = TaskTransport()
    async with transport.serve() as url:
        config, selection = configuration(tmp_path, url)
        async with application(config, leaves, dashboard_auth, buffer_http=getattr(request, "param", False)) as (graph, client):
            tray = graph.dashboard.routine_proposals
            assert tray is graph.producer_coverage.context_erasure.routine_proposals
            yield SimpleNamespace(
                graph=graph, tray=tray, scheduler=graph.routine_scheduler,
                client=client, transport=transport, selection=selection,
            )


def propose(review, operation, *, source=None, name="Synthetic proposal", target=None, expires_at=None):
    spec = parse_routine_spec(definition(name)) if operation in {"create", "update"} else None
    return review.tray.submit(
        ProposalOperation(operation), conversation_id=source or str(uuid4()), spec=spec,
        routine_id=target.routine_id if target else None,
        expected_revision=target.revision if target else None, expires_at=expires_at,
    )


async def confirm(review, proposal_id):
    return await review.client.post(f"{PATH}/{proposal_id}/confirm")


async def test_cookie_and_routines_manage_required_for_every_review_route(review, dashboard_auth):
    proposal_id = propose(review, "create")
    route = f"{PATH}/{proposal_id}"
    routes = [("GET", PATH), ("GET", route), ("POST", route + "/confirm"),
              ("POST", route + "/reject")]
    async with AsyncClient(
        transport=ASGITransport(app=review.graph.dashboard.app), base_url="http://testserver",
    ) as anonymous:
        for method, path in routes:
            assert (await anonymous.request(method, path)).status_code == 401
    cookie_as_bearer = {"Authorization": f"Bearer {review.client.cookies.get(COOKIE_NAME)}"}
    for headers in (dashboard_auth, cookie_as_bearer):
        for method, path in routes:
            response = await review.client.request(method, path, headers=headers)
            assert (response.status_code, response.json()["detail"]) == (403, "routine_proposals_require_session")
    async with login(review.graph, dashboard_auth, DashboardRole.HOME) as household:
        for method, path in routes:
            response = await household.request(method, path)
            assert (response.status_code, response.json()["detail"]) == (403, "dashboard_permission_denied")
    assert (await review.client.get(route)).status_code == 200
    assert await review.scheduler.get_all_routines() == []
    assert review.transport.sends == []


async def test_create_preview_and_confirm_once_without_disclosing_source(review):
    source = str(uuid4())
    proposal_id = propose(review, "create", source=source)
    assert propose(review, "create", source=source) == proposal_id
    listed = await review.client.get(PATH)
    assert listed.status_code == 200 and listed.headers["Cache-Control"] == "no-store"
    assert listed.json()["cleared_on_restart"] is True
    [entry] = listed.json()["proposals"]
    assert (entry["proposal_id"], entry["operation"], entry["source"]) == (proposal_id, "create", "household_voice")
    assert datetime.fromisoformat(entry["expires_at"]) - datetime.fromisoformat(entry["created_at"]) == timedelta(hours=48)
    preview = await review.client.get(f"{PATH}/{proposal_id}")
    assert preview.status_code == 200 and preview.headers["Cache-Control"] == "no-store"
    assert preview.json()["diff"] == {"before": None, "after": definition("Synthetic proposal")}
    assert preview.json()["stale"] is False
    for response in (listed, preview):
        assert source not in response.text
        assert "conversation_id" not in response.text and "transcript" not in response.text and "person_id" not in response.text
    assert await review.scheduler.get_all_routines() == []
    assert review.transport.sends == []

    response = await confirm(review, proposal_id)
    assert response.status_code == 200
    receipt = response.json()
    assert receipt == {
        "proposal_id": proposal_id, "status": "confirmed", "routine_id": receipt["routine_id"], "revision": 1,
    }
    assert (await confirm(review, proposal_id)).json() == receipt
    assert propose(review, "create", source=source) == proposal_id
    assert (await review.client.get(PATH)).json()["proposals"] == []
    [saved] = await review.scheduler.get_all_routines()
    principal = (await review.client.get("/api/admin/auth/whoami")).json()
    assert saved.routine_id == receipt["routine_id"] and saved.created_by == principal["subject"]
    assert await review.scheduler.get_latest_run(saved.routine_id) is None
    assert review.transport.sends == []


async def test_update_disable_enable_and_delete_preview_and_cas(review):
    saved = await review.scheduler.register_routine(
        parse_routine_spec(definition("Synthetic seed")), actor="synthetic-seed",
    )
    for operation, enabled, name in (
        ("update", True, "Synthetic proposal"), ("disable", False, "Synthetic proposal"),
        ("enable", True, "Synthetic proposal"), ("delete", None, None),
    ):
        proposal_id = propose(review, operation, target=saved)
        stale_id = propose(review, operation, target=saved)
        route = f"{PATH}/{proposal_id}"
        preview = (await review.client.get(route)).json()
        assert preview["expected_revision"] == preview["current_revision"] == saved.revision
        assert preview["stale"] is False
        assert preview["diff"] == {
            "before": definition(saved.spec.name, enabled=saved.spec.enabled),
            "after": definition(name, enabled=enabled) if name is not None else None,
        }
        assert await review.scheduler.get_routine(saved.routine_id) == saved

        response = await confirm(review, proposal_id)
        assert response.status_code == 200
        assert response.json()["routine_id"] == saved.routine_id
        assert (await confirm(review, proposal_id)).json() == response.json()
        stale_preview = (await review.client.get(f"{PATH}/{stale_id}")).json()
        assert stale_preview["stale"] is True
        conflict = await confirm(review, stale_id)
        assert (conflict.status_code, conflict.json()["detail"]) == (409, "revision_conflict")
        current = await review.scheduler.get_routine(saved.routine_id)
        if name is None:
            assert current is None and response.json()["revision"] is None
        else:
            assert current.revision == saved.revision + 1
            assert (current.spec.name, current.spec.enabled) == (name, enabled)
            saved = current
    assert review.transport.sends == []


async def test_reject_is_idempotent_and_cannot_be_confirmed_or_resubmitted(review):
    source = str(uuid4())
    proposal_id = propose(review, "create", source=source)
    route = f"{PATH}/{proposal_id}"
    response = await review.client.post(route + "/reject")
    assert response.status_code == 200 and response.json()["status"] == "rejected"
    assert (await review.client.post(route + "/reject")).json() == response.json()
    assert propose(review, "create", source=source) == proposal_id
    assert (await review.client.get(route)).status_code == 404
    assert (await review.client.get(PATH)).json()["proposals"] == []
    conflict = await confirm(review, proposal_id)
    assert (conflict.status_code, conflict.json()["detail"]) == (409, "proposal_rejected")
    assert await review.scheduler.get_all_routines() == []
    assert review.transport.sends == []


async def test_create_update_and_enable_require_current_casa_selection(review):
    before = await review.scheduler.register_routine(
        parse_routine_spec(definition("Synthetic disabled", enabled=False)), actor="synthetic-seed",
    )
    create = propose(review, "create")
    update = propose(review, "update", target=before)
    enable = propose(review, "enable", target=before)
    selection = json.loads(review.selection.read_text())
    review.selection.write_text(json.dumps({**selection, "allowed_entities": []}))
    for proposal_id in (create, update, enable):
        assert (await review.client.get(f"{PATH}/{proposal_id}")).status_code == 200
        response = await confirm(review, proposal_id)
        assert (response.status_code, response.json()["detail"]) == (422, "invalid_routine")
        assert await review.scheduler.get_all_routines() == [before]
    review.selection.write_text(json.dumps(selection))
    assert (await confirm(review, enable)).status_code == 200
    assert (await review.scheduler.get_routine(before.routine_id)).spec.enabled
    assert review.transport.sends == []


async def test_ttl_never_extends_on_replay_and_context_sweep_physically_purges(review, monkeypatch):
    ledger = review.graph.privacy.ledger
    now = [ledger.clock()]
    monkeypatch.setattr(ledger, "clock", lambda: now[0])
    source = str(uuid4())
    short = propose(review, "create", source=source, expires_at=now[0] + 10)
    long = propose(review, "create", expires_at=now[0] + 7 * 24 * 60 * 60)
    by_id = {item["proposal_id"]: item for item in (await review.client.get(PATH)).json()["proposals"]}
    assert datetime.fromisoformat(by_id[long]["expires_at"]) - datetime.fromisoformat(by_id[long]["created_at"]) == timedelta(hours=48)
    now[0] += 5
    assert propose(review, "create", source=source, expires_at=now[0] + 20) == short
    now[0] += 5
    driver = review.graph.producer_coverage.context_erasure
    await driver._sweep_runtime()
    assert short not in review.tray._entries
    assert [item["proposal_id"] for item in (await review.client.get(PATH)).json()["proposals"]] == [long]
    assert (await review.client.get(f"{PATH}/{short}")).status_code == 404
    assert (await confirm(review, short)).status_code == 404
    assert await review.scheduler.get_all_routines() == []


def hold_provisional_write(review, monkeypatch):
    work = review.scheduler.store._work
    original = work.write
    entered, release = threading.Event(), threading.Event()

    async def gated_write(function, *, guard=None, publish=None):
        def provisional(database):
            result = function(database)
            entered.set()
            assert release.wait(5)
            return result
        return await original(provisional, guard=guard, publish=publish)

    monkeypatch.setattr(work, "write", gated_write)
    return entered, release


@pytest.mark.parametrize("retire", ["session", "source", "expiry"])
async def test_commit_guard_rolls_back_a_provisional_write(review, monkeypatch, retire):
    source = str(uuid4())
    privacy = review.graph.privacy
    now = [privacy.ledger.clock()]
    if retire == "expiry":
        monkeypatch.setattr(privacy.ledger, "clock", lambda: now[0])
    proposal_id = propose(review, "create", source=source, expires_at=now[0] + 60)
    entered, release = hold_provisional_write(review, monkeypatch)
    confirming = asyncio.create_task(confirm(review, proposal_id))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if retire == "session":
            assert (await review.client.post("/api/admin/auth/logout")).json()["session_revoked"]
        elif retire == "source":
            await review.graph.erasure_services.request(source, MemoryReviewer("synthetic-reviewer"))
        else:
            now[0] += 60
    finally:
        release.set()
    denied = await confirming
    assert denied.status_code == 403
    assert await review.scheduler.get_all_routines() == []
    assert review.transport.sends == []


async def test_pending_revocation_blocks_commit_before_erasure_publish(review, monkeypatch):
    source = str(uuid4())
    proposal_id = propose(review, "create", source=source)
    ledger = review.graph.privacy.ledger
    original = ledger._publish_revocation_admission
    revoking, publish = threading.Event(), threading.Event()
    writing, commit = hold_provisional_write(review, monkeypatch)

    def gated_revocation():
        original()
        revoking.set()
        assert publish.wait(5)

    monkeypatch.setattr(ledger, "_publish_revocation_admission", gated_revocation)
    confirming = asyncio.create_task(confirm(review, proposal_id))
    try:
        assert await asyncio.to_thread(writing.wait, 3)
        erasing = asyncio.create_task(review.graph.erasure_services.request(source, MemoryReviewer("synthetic-reviewer")))
        assert await asyncio.to_thread(revoking.wait, 3)
        assert ledger.conversation_revoked(source) and not ledger.conversation_committed_revoked(source)
        assert (await review.client.get(f"{PATH}/{proposal_id}")).status_code == 404
        assert proposal_id in review.tray._entries
        commit.set()
        assert (await confirming).status_code == 403
    finally:
        commit.set()
        publish.set()
    await erasing
    assert await review.scheduler.get_all_routines() == []


async def test_registered_erasure_removes_pending_and_terminal_proposals(review):
    graph = review.graph
    await graph.producer_coverage.context_erasure.retention.begin_close()
    source = str(uuid4())
    pending = propose(review, "create", source=source)
    rejected = propose(review, "create", source=source, name="Synthetic rejected")
    assert (await review.client.post(f"{PATH}/{rejected}/reject")).status_code == 200
    survivor = propose(review, "create")
    await graph.erasure_services.request(source, MemoryReviewer("synthetic-reviewer"))
    async with asyncio.timeout(3):
        while pending in review.tray._entries or rejected in review.tray._entries:
            await graph.erasure_services.reconcile()
            await asyncio.sleep(.001)
    assert survivor in review.tray._entries
    assert (await confirm(review, pending)).status_code == 404
    assert (await review.client.post(f"{PATH}/{rejected}/reject")).status_code == 404
    with pytest.raises(PermissionError, match="proposal_source_unavailable"):
        propose(review, "create", source=source)
    assert await review.scheduler.get_all_routines() == []


@pytest.mark.parametrize(("view", "review", "cancel_observer"), [
    ("list", False, False), ("preview", False, False),
    ("list", True, True), ("preview", True, False),
], indirect=["review"])
async def test_erasure_between_headers_and_body_retires_the_owned_proposal_delivery(
    review, view, cancel_observer,
):
    app = review.graph.dashboard.app
    source = str(uuid4())
    proposal_id = propose(review, "create", source=source, name="Synthetic private diff")
    survivor = propose(review, "create", name="Synthetic surviving proposal")
    route = PATH if view == "list" else f"{PATH}/{proposal_id}"
    headers_sent, release = asyncio.Event(), asyncio.Event()
    messages = []

    async def gated(scope, receive, send):
        async def actual(message):
            messages.append(message.copy())
            await send(message)
            if message["type"] == "http.response.start":
                headers_sent.set()
                await release.wait()

        await app(scope, receive, actual)

    async with AsyncClient(
        transport=ASGITransport(app=gated), base_url="http://testserver", cookies=review.client.cookies,
    ) as reader:
        pending = asyncio.create_task(reader.get(route))
        try:
            await asyncio.wait_for(headers_sent.wait(), 3)
            assert messages[0]["status"] == 200
            if cancel_observer:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            owner = review.graph.erasure_services
            receipt = await owner.request(source, MemoryReviewer("synthetic-reviewer"))
            async with asyncio.timeout(3):
                while True:
                    await owner.reconcile()
                    status = await owner.privacy.acall(owner.journal.snapshot, receipt["id"])
                    if any(
                        step["participant"] == "contexts" and step["state"] == "pending"
                        and step["pending_count"] == 1 for step in status["steps"]
                    ):
                        break
                    await asyncio.sleep(.001)
            assert proposal_id not in review.tray._entries
            release.set()
            if not cancel_observer:
                # With headers already sent, ASGITransport reports an incomplete response.
                with pytest.raises(AssertionError):
                    await pending
            async with asyncio.timeout(3):
                while review.tray._deliveries:
                    await asyncio.sleep(.001)
            assert [message["type"] for message in messages] == ["http.response.start"]
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)

    listed = await review.client.get(PATH)
    assert listed.status_code == 200
    assert [entry["proposal_id"] for entry in listed.json()["proposals"]] == [survivor]
    assert (await review.client.get(f"{PATH}/{proposal_id}")).status_code == 404


async def test_cancelled_confirm_after_native_commit_replays_the_single_receipt(review, monkeypatch):
    proposal_id = propose(review, "create")
    route = f"{PATH}/{proposal_id}"
    database = review.scheduler.store._db
    original = database._execute
    committed, release = asyncio.Event(), asyncio.Event()

    async def gated_result(function, *args, **kwargs):
        result = await original(function, *args, **kwargs)
        if isinstance(result, ScheduledRoutine) and not committed.is_set():
            committed.set()
            await release.wait()
        return result

    monkeypatch.setattr(database, "_execute", gated_result)
    first = asyncio.create_task(confirm(review, proposal_id))
    try:
        await asyncio.wait_for(committed.wait(), 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        duplicate = await confirm(review, proposal_id)
        assert (duplicate.status_code, duplicate.json()["detail"]) == (409, "proposal_confirming")
        reject = await review.client.post(route + "/reject")
        assert (reject.status_code, reject.json()["detail"]) == (409, "proposal_confirming")
    finally:
        release.set()
    async with asyncio.timeout(3):
        while (receipt := await confirm(review, proposal_id)).status_code == 409:
            assert receipt.json()["detail"] == "proposal_confirming"
            await asyncio.sleep(.001)
    [saved] = await review.scheduler.get_all_routines()
    assert receipt.status_code == 200 and receipt.json()["routine_id"] == saved.routine_id
    assert saved.revision == 1 and review.transport.sends == []


async def test_restart_empties_the_tray_but_keeps_confirmed_definitions(leaves, tmp_path, dashboard_auth):
    async with TaskTransport().serve() as url:
        config, _ = configuration(tmp_path, url)
        async with application(config, leaves, dashboard_auth) as (graph, client):
            review = SimpleNamespace(tray=graph.dashboard.routine_proposals, client=client)
            pending = propose(review, "create")
            accepted = propose(review, "create", name="Synthetic approved")
            response = await confirm(review, accepted)
            assert response.status_code == 200
            routine_id = response.json()["routine_id"]
        async with application(config, leaves, dashboard_auth) as (restarted, client):
            assert (await client.get(PATH)).json() == {"proposals": [], "cleared_on_restart": True}
            assert (await client.get(f"{PATH}/{pending}")).status_code == 404
            assert (await client.post(f"{PATH}/{accepted}/confirm")).status_code == 404
            [saved] = await restarted.routine_scheduler.get_all_routines()
            assert saved.routine_id == routine_id
