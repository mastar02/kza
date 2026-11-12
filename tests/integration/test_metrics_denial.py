"""U3b-0d2 S7: /api/metrics (latency, loop lag, uptime) requires diagnostics:read.

Real graph, real logins and the real route: anonymous and household sessions
are refused; only an operator reads. Evaluator and person-bound household
grants need seeded samples or people, so their role grants are checked directly.
"""

from contextlib import AsyncExitStack
import time

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.requests import Request

from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardRole, _permissions
from src.main import build_application
from tests.integration.test_capability_application import leaves as capability_leaves
from tests.mocks.application_graph import offline_config
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio

PERSON = "user_" + "a" * 32
DENIED = {"detail": "dashboard_permission_denied"}
UNAUTHENTICATED = {"detail": "auth required"}


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize("method,expected", [
    ("GET", {"diagnostics:read"}), ("HEAD", {"diagnostics:read"}), ("POST", {"admin:manage"}),
])
async def test_metrics_route_permission_is_explicit(method, expected):
    request = Request({"type": "http", "method": method, "path": "/api/metrics", "headers": [],
                       "query_string": b""})
    assert set(_route_permissions(request)) == expected


@pytest.mark.parametrize("role,user_id,allowed", [
    (DashboardRole.OPERATOR, None, True), (DashboardRole.HOME, None, False),
    (DashboardRole.HOME, PERSON, False), (DashboardRole.EVALUATOR, None, False),
])
async def test_only_operators_hold_diagnostics_read(role, user_id, allowed):
    """A person-bound household grant adds personal permissions, never diagnosis."""
    assert ("diagnostics:read" in _permissions(role, user_id)) is allowed


ROWS = [
    # actor, role, user_id, method, status, body
    ("anonymous", None, None, "GET", 401, UNAUTHENTICATED),
    ("home", DashboardRole.HOME, None, "GET", 403, DENIED),
    # No write route exists: the router answers 405 before any dependency, with no data.
    ("home", DashboardRole.HOME, None, "POST", 405, {"detail": "Method Not Allowed"}),
    ("anonymous", None, None, "POST", 405, {"detail": "Method Not Allowed"}),
    ("operator", DashboardRole.OPERATOR, None, "GET", 200, None),
]


async def test_metrics_denial_matrix(tmp_path, leaves, dashboard_auth):
    transport = TaskTransport()
    async with transport.serve() as url, AsyncExitStack() as stack:
        graph = await build_application(offline_config(tmp_path, url), shutdown=stack, factories=leaves)
        authority = graph.dashboard_authority
        admin = authority.resolve(dashboard_auth["Authorization"])
        for actor, role, user_id, method, status, body in ROWS:
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app),
                                   base_url="http://testserver") as client:
                if role is not None:
                    kwargs = {"user_id": user_id} if user_id else {}
                    grant = await authority.issue_credential(admin, subject=f"synthetic-{actor}", role=role,
                                                             expires_at=time.time() + 120, **kwargs)
                    login = await client.post("/api/admin/auth/login", json={"token": grant.token})
                    assert login.status_code == 200, actor
                response = await client.request(method, "/api/metrics")
            assert response.status_code == status, (actor, method, response.text)
            if body is not None:
                assert response.json() == body, (actor, method)
            elif status == 200:
                data = response.json()
                assert "loop_lag" in data and data["loop_lag"]["warn_ms"] == 200.0, data
