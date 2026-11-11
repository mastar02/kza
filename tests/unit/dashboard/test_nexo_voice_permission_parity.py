"""U3b-4a-3: the NEXO client maps every voice route it uses exactly as the server requires.

The effective server requirement is auth._route_permissions plus the handler's own guard,
read from the registered memory_voice_api routes: each route's dependencies (and the switch
handler, whose guard is in its body) run against a recording session_guard. The shipped
state.js must ask for the same set and leave every other voice path unmapped. The Spanish copy
the client repeats must equal the server's.
"""

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
import pytest
from starlette.requests import Request
from starlette.routing import Match

from src.dashboard import memory_api, memory_voice_api
from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardPermission

ROOT = Path(__file__).resolve().parents[3]
PERSON = "user_" + "a" * 32
PROFILE = "00000000-0000-4000-8000-00000000000a"

USED = [
    ("GET", "/api/memory/mine/voice"),
    ("POST", "/api/memory/mine/voice/consent"),
    ("DELETE", "/api/memory/mine/voice/consent"),
    ("POST", "/api/memory/mine/voice/verification"),
    ("GET", "/api/memory/voice/consents"),
    ("POST", f"/api/memory/voice/consents/{PERSON}/reopen"),
    ("GET", "/api/memory/voice/switch"),
    ("POST", "/api/memory/voice/switch"),
    ("POST", f"/api/voices/profiles/{PROFILE}/attribution/resume"),
]
HANDLER = {
    ("GET", "/api/memory/mine/voice"): {"memory:own:voice"},
    ("POST", "/api/memory/mine/voice/consent"): {"memory:own:voice"},
    ("DELETE", "/api/memory/mine/voice/consent"): {"memory:own:voice"},
    ("POST", "/api/memory/mine/voice/verification"): {"memory:own:voice"},
    ("GET", "/api/memory/voice/consents"): {"memory:assign"},
    ("POST", "/api/memory/voice/consents/{person_id}/reopen"): {"memory:assign"},
    ("GET", "/api/memory/voice/switch"): {"memory:assign"},
    ("POST", "/api/memory/voice/switch"): {"admin:manage", "memory:assign"},
    ("POST", "/api/voices/profiles/{profile_id}/attribution/resume"): {"memory:assign", "voices:review", "admin:manage"},
}
UNUSED = [
    ("PUT", "/api/memory/mine/voice"), ("POST", "/api/memory/mine/voice"), ("GET", "/api/memory/mine/voice/consent"),
    ("GET", "/api/memory/mine/voice/verification"), ("DELETE", "/api/memory/mine/voice/verification"),
    ("GET", "/api/memory/mine/voice/unknown"), ("POST", "/api/memory/voice/consents"),
    ("GET", f"/api/memory/voice/consents/{PERSON}/reopen"), ("POST", "/api/memory/voice/consents/a b/reopen"),
    ("POST", f"/api/memory/voice/consents/{PERSON}"), ("DELETE", "/api/memory/voice/switch"),
    ("POST", "/api/voices/profiles/not-a-uuid/attribution/resume"), ("GET", f"/api/voices/profiles/{PROFILE}/attribution/resume"),
    ("POST", f"/api/voices/profiles/{PROFILE}/attribution"), ("POST", f"/api/voices/profiles/{PROFILE.upper()}/attribution/resume"),
    ("GET", "/api/memory/voice"), ("POST", "/api/memory/mine/voice/consent/x"),
]


def scope(method: str, path: str) -> dict:
    return {"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}


def registered_routes() -> list[APIRoute]:
    app = FastAPI()
    memory_voice_api.register_memory_voice_routes(app, manager=None)
    return [route for route in app.routes if isinstance(route, APIRoute)]


def route_for(routes, method, path) -> APIRoute:
    matches = [route for route in routes if route.matches(scope(method, path))[0] is Match.FULL]
    assert len(matches) == 1, (method, path, [route.path for route in matches])
    return matches[0]


def handler_permissions(route: APIRoute, monkeypatch) -> set[str]:
    """Run the route's own guards with a session_guard that records what it is asked for."""
    asked: list[frozenset] = []

    def recording_guard(_request, permissions, _principal):
        asked.append(frozenset(permissions))
        raise PermissionError("recorded")

    monkeypatch.setattr(memory_voice_api, "session_guard", recording_guard)
    principal = SimpleNamespace(permissions=frozenset(DashboardPermission), user_id=PERSON, subject="synthetic", role=None)
    method = sorted(route.methods)[0]
    calls = [dependency.call for dependency in route.dependant.dependencies
             if dependency.call is not memory_voice_api.require_dashboard_session]
    if not calls:  # The switch write mints its actor in the handler body.
        calls = [route.endpoint]
    for call in calls:
        assert call.__module__ == memory_voice_api.__name__, call
        with pytest.raises(HTTPException) as refused:
            asyncio.run(call(Request(scope(method, route.path)), principal))
        assert refused.value.status_code == 403
    return set().union(*asked)


def client(routes) -> list:
    node = shutil.which("node")
    assert node, "Node is required to verify the shipped NEXO permission map"
    script = ("const s=require('./src/dashboard/frontend/home/nexo/state.js');"
              "const routes=" + json.dumps(routes) + ";"
              "console.log(JSON.stringify(routes.map(([m,p])=>{const v=s.memoryPermission(p,m);"
              "return v===null?null:(Array.isArray(v)?v:[v]).slice().sort();})));")
    result = subprocess.run([node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_handler_guards_are_read_from_memory_voice_api(monkeypatch):
    routes = registered_routes()
    live = {(method, route_for(routes, method, path).path): handler_permissions(route_for(routes, method, path), monkeypatch)
            for method, path in USED}
    assert live == HANDLER


def test_every_used_voice_route_requires_exactly_the_server_permissions(monkeypatch):
    routes = registered_routes()
    expected = [sorted(set(_route_permissions(Request(scope(method, path))))
                       | handler_permissions(route_for(routes, method, path), monkeypatch)) for method, path in USED]
    assert client([[method, path] for method, path in USED]) == expected


def test_unused_voice_routes_stay_unmapped_in_the_client():
    assert client([[method, path] for method, path in UNUSED]) == [None] * len(UNUSED)


def _client_constants() -> dict:
    script = ("const p=require('./src/dashboard/frontend/home/nexo/my-voice-owner.js');"
              "const a=require('./src/dashboard/frontend/home/nexo/voice-consents-owner.js');"
              "const m=require('./src/dashboard/frontend/home/nexo/my-memory-owner.js');"
              "console.log(JSON.stringify({opt:p.OPT_IN_DISCLOSURE,reenrol:p.REENROLMENT_NOTICE,insufficient:p.INSUFFICIENT_PROFILES_MESSAGE,"
              "states:p.STATE_MESSAGES,voices:a.VOICES_DISCLOSURE,multi:a.MULTI_PROFILE_NOTICE,voiceReturn:m.VOICE_RETURN_NOTICE,"
              "ret:m.RETURN_NOTICE,max:a.MAX_CONSENTS}));")
    result = subprocess.run([shutil.which("node"), "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_the_client_copy_equals_the_server_copy():
    from src.memory.fact_ledger import VOICE_CONSENT_LISTING_MAX
    copy = _client_constants()
    assert copy == {"opt": memory_voice_api.OPT_IN_DISCLOSURE, "reenrol": memory_voice_api.REENROLMENT_NOTICE,
                    "insufficient": memory_voice_api.INSUFFICIENT_PROFILES_MESSAGE, "states": memory_voice_api.STATE_MESSAGES,
                    "voices": memory_voice_api.VOICES_DISCLOSURE, "multi": memory_voice_api.MULTI_PROFILE_NOTICE,
                    "voiceReturn": memory_api.VOICE_RETURN_NOTICE, "ret": memory_api.RETURN_NOTICE, "max": VOICE_CONSENT_LISTING_MAX}


def test_the_views_mapping_keeps_voice_inside_the_existing_views():
    """No new navigation entry: the person's section lives in 'Mis recuerdos', the administration in 'Voces'."""
    source = (ROOT / "src/dashboard/frontend/home/nexo/state.js").read_text(encoding="utf-8")
    assert "mymemory: ['memory:own:read']" in source and "voices: ['voices:read', 'voices:review']" in source
    assert "voice-attribution" not in source and "myvoice" not in source
