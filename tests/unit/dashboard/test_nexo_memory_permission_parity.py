"""U3b-3: the NEXO client maps every memory route it uses exactly as the server requires.

The effective server requirement is auth._route_permissions plus the handler's own
permission. The handler permission is read from the registered memory_api routes (R4):
each route's dependencies are run against a recording session_guard, so a handler that
changes its permission (or drops its guard) fails this test. The shipped state.js must
ask for the same set, and must leave every other memory path unmapped (refused locally).
"""

import asyncio
import json
from pathlib import Path
import shutil
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
import pytest
from starlette.requests import Request
from starlette.routing import Match

from src.dashboard import memory_api
from src.dashboard.auth import _route_permissions

ROOT = Path(__file__).resolve().parents[3]
FACT = "fact_" + "a" * 64
CURSOR = "k1." + "0" * 16 + "." + FACT

# (method, path) for every memory route the Memoria and Mis recuerdos views call.
USED = [
    ("GET", "/api/memory"),
    ("GET", f"/api/memory?facts_after={CURSOR}"),
    ("GET", f"/api/memory?proposals_after={CURSOR}"),
    ("POST", f"/api/memory/proposals/{FACT}/confirm"),
    ("POST", f"/api/memory/proposals/{FACT}/reject"),
    ("DELETE", f"/api/memory/facts/{FACT}"),
    ("POST", f"/api/memory/facts/{FACT}/owner"),
    ("GET", "/api/memory/returned"),
    ("GET", f"/api/memory/returned?after={CURSOR}"),
    ("GET", "/api/memory/owners/counts"),
    ("GET", "/api/memory/mine"),
    ("GET", f"/api/memory/mine?after={CURSOR}"),
    ("DELETE", f"/api/memory/mine/facts/{FACT}"),
    ("POST", f"/api/memory/mine/facts/{FACT}/return"),
]
# A documented snapshot of what the handlers require today; the live read below must agree with it.
HANDLER = {
    "/api/memory": set(), "/api/memory/proposals/{fact_id}/confirm": {"memory:review"},
    "/api/memory/proposals/{fact_id}/reject": {"memory:review"}, "/api/memory/facts/{fact_id}": {"memory:review"},
    "/api/memory/facts/{fact_id}/owner": {"memory:assign"}, "/api/memory/returned": {"memory:assign"},
    "/api/memory/owners/counts": {"memory:assign"}, "/api/memory/mine": {"memory:own:read"},
    "/api/memory/mine/facts/{fact_id}": {"memory:own:write"}, "/api/memory/mine/facts/{fact_id}/return": {"memory:own:write"},
}
# Paths the server maps but no NEXO view uses (or no route serves): the client must refuse them.
UNUSED = [
    ("GET", "/api/memory/facts"), ("GET", "/api/memory/proposals"), ("GET", f"/api/memory/facts/{FACT}/owner"),
    ("DELETE", f"/api/memory/facts/{FACT}/owner"), ("POST", f"/api/memory/facts/{FACT}/owner/x"),
    ("POST", "/api/memory/facts//owner"), ("POST", "/api/memory/returned"), ("DELETE", "/api/memory/owners/counts"),
    ("PUT", "/api/memory/mine"), ("POST", "/api/memory/mine"), ("GET", "/api/memory/mine/unknown"),
    ("GET", f"/api/memory/mine/facts/{FACT}"), ("POST", f"/api/memory/mine/facts/{FACT}/owner"),
    ("PUT", f"/api/memory/mine/facts/{FACT}/return"), ("GET", "/api/memory/minefoo"),
    ("DELETE", "/api/memory/mine/facts/fact_x"), ("POST", f"/api/memory/mine/facts/{FACT}/unassign"),
]


def scope(method: str, path: str) -> dict:
    return {"type": "http", "method": method, "path": path.split("?")[0], "headers": [], "query_string": b""}


def registered_routes() -> list[APIRoute]:
    app = FastAPI()
    memory_api.register_memory_routes(app, manager=None)
    return [route for route in app.routes if isinstance(route, APIRoute)]


def route_for(routes: list[APIRoute], method: str, path: str) -> APIRoute:
    matches = [route for route in routes if route.matches(scope(method, path))[0] is Match.FULL]
    assert len(matches) == 1, (method, path, [route.path for route in matches])
    return matches[0]


def handler_permissions(route: APIRoute, monkeypatch) -> set[str]:
    """Run the route's own dependencies with a session_guard that records what it is asked for."""
    asked: list[frozenset] = []

    def recording_guard(_request, permissions, _principal):
        asked.append(frozenset(permissions))
        raise PermissionError("recorded")

    monkeypatch.setattr(memory_api, "session_guard", recording_guard)
    for dependency in route.dependant.dependencies:
        call = dependency.call
        if call is memory_api.require_dashboard_session:
            continue
        assert call.__module__ == memory_api.__name__, call
        with pytest.raises(HTTPException) as refused:
            asyncio.run(call(Request(scope(*sorted(route.methods)[:1], route.path)), object()))
        assert refused.value.status_code == 403
    return set().union(*asked)


def server(route: APIRoute, method: str, path: str, monkeypatch) -> list[str]:
    return sorted(set(_route_permissions(Request(scope(method, path)))) | handler_permissions(route, monkeypatch))


def client(routes: list[tuple[str, str]]) -> list:
    node = shutil.which("node")
    assert node, "Node is required to verify the shipped NEXO permission map"
    script = ("const s=require('./src/dashboard/frontend/home/nexo/state.js');"
              "const routes=" + json.dumps(routes) + ";"
              "console.log(JSON.stringify(routes.map(([m,p])=>{const v=s.memoryPermission(p.split('?')[0],m);"
              "return v===null?null:(Array.isArray(v)?v:[v]).slice().sort();})));")
    result = subprocess.run([node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_handler_permissions_are_read_from_memory_api(monkeypatch):
    routes = registered_routes()
    live = {route_for(routes, method, path).path: handler_permissions(route_for(routes, method, path), monkeypatch)
            for method, path in USED}
    assert live == HANDLER


def test_the_live_read_detects_handler_drift(monkeypatch):
    """A handler asking for another permission changes what this test reads (R4)."""
    route = route_for(registered_routes(), "GET", "/api/memory/mine")

    async def drifted(request: Request, principal: object):
        try:
            memory_api.session_guard(request, frozenset({"memory:own:write"}), principal)
        except PermissionError as error:
            raise HTTPException(403) from error
    drifted.__module__ = memory_api.__name__
    route.dependant.dependencies[0].call = drifted
    assert handler_permissions(route, monkeypatch) == {"memory:own:write"} != HANDLER["/api/memory/mine"]


def test_every_used_memory_route_requires_exactly_the_server_permissions(monkeypatch):
    routes = registered_routes()
    mapped = client([[method, path] for method, path in USED])
    expected = [server(route_for(routes, method, path), method, path, monkeypatch) for method, path in USED]
    assert mapped == expected


def test_unused_memory_routes_stay_unmapped_in_the_client():
    assert client([[method, path] for method, path in UNUSED]) == [None] * len(UNUSED)
