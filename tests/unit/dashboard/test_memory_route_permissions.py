"""U3b-2 route permission mapping for memory: the person's branch comes first."""

import ast
from pathlib import Path

import pytest
from starlette.requests import Request

from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardRole, _permissions

ROOT = Path(__file__).resolve().parents[3]
FACT = "fact_" + "a" * 64
PERSON = "user_" + "a" * 32


def permissions(method: str, path: str) -> set[str]:
    return set(_route_permissions(Request({"type": "http", "method": method, "path": path, "headers": [],
                                           "query_string": b""})))


@pytest.mark.parametrize("method,path,expected", [
    ("GET", "/api/memory/mine", {"memory:own:read"}),
    ("HEAD", "/api/memory/mine", {"memory:own:read"}),
    ("OPTIONS", "/api/memory/mine", {"memory:own:read"}),
    ("GET", "/api/memory/mine/", {"memory:own:read"}),
    ("GET", "/api/memory/mine/unknown", {"memory:own:read"}),
    ("GET", "/api/memory/mine/../facts", {"memory:own:read"}),
    ("DELETE", f"/api/memory/mine/facts/{FACT}", {"memory:own:write"}),
    ("POST", f"/api/memory/mine/facts/{FACT}/return", {"memory:own:write"}),
    ("POST", f"/api/memory/mine/facts/{FACT}/owner", {"memory:own:write"}),
    ("PUT", "/api/memory/mine", {"memory:own:write"}),
    ("POST", f"/api/memory/facts/{FACT}/owner", {"memory:assign"}),
    ("GET", f"/api/memory/facts/{FACT}/owner", {"memory:assign", "memory:read"}),
    ("GET", "/api/memory/returned", {"memory:assign", "memory:read"}),
    ("GET", "/api/memory/owners/counts", {"memory:assign", "memory:read"}),
    ("POST", f"/api/memory/facts/{FACT}/owner/x", {"memory:review"}),
    # Review R3: the owner branch requires exactly /api/memory/facts/{id}/owner (5 slashes).
    ("POST", f"/api/memory/facts/x/{FACT}/owner", {"memory:review"}),
    ("POST", "/api/memory/facts//owner", {"memory:assign"}),  # empty id: still the operator-only branch
    ("GET", f"/api/memory/facts/x/{FACT}/owner", {"memory:read"}),
    ("GET", "/api/memory/minefoo", {"memory:read"}),
    ("GET", "/api/memory/MINE", {"memory:read"}),
    ("GET", "/api/memory", {"memory:read"}),
    ("GET", "/api/memory/facts", {"memory:read"}),
    ("DELETE", f"/api/memory/facts/{FACT}", {"memory:review"}),
    ("POST", f"/api/memory/proposals/{FACT}/confirm", {"memory:review"}),
])
def test_memory_route_permissions(method, path, expected):
    assert permissions(method, path) == expected


@pytest.mark.parametrize("role,user_id,mine,assign", [
    (DashboardRole.HOME, PERSON, True, False),
    (DashboardRole.HOME, None, False, False),
    (DashboardRole.OPERATOR, None, False, True),
    (DashboardRole.EVALUATOR, None, False, False),
])
def test_only_a_person_bound_household_reaches_mine_and_only_operators_assign(role, user_id, mine, assign):
    granted = _permissions(role, user_id)
    for method, path in (("GET", "/api/memory/mine"), ("DELETE", f"/api/memory/mine/facts/{FACT}"),
                         ("POST", f"/api/memory/mine/facts/{FACT}/return")):
        assert (permissions(method, path) <= granted) is mine
    for method, path in (("POST", f"/api/memory/facts/{FACT}/owner"), ("GET", "/api/memory/returned"),
                         ("GET", "/api/memory/owners/counts")):
        assert (permissions(method, path) <= granted) is assign


def test_the_person_branch_precedes_the_generic_memory_branch():
    tree = ast.parse((ROOT / "src/dashboard/auth.py").read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_route_permissions")
    order = [ast.unparse(n.test) for n in function.body if isinstance(n, ast.If)]
    mine = next(i for i, test in enumerate(order) if "'/api/memory/mine'" in test or '"/api/memory/mine"' in test)
    owner = next(i for i, test in enumerate(order) if "/owner" in test)
    generic = next(i for i, test in enumerate(order) if test.startswith("path == '/api/memory' or")
                   or test.startswith('path == "/api/memory" or'))
    assert mine < owner < generic
