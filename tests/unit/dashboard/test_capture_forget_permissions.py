"""R1-B2: forgetting lost recordings is an administrative permission only.

The route needs ``admin:manage`` at the session layer and again when the
capture controller is minted; no household or evaluator role holds it, and a
recognized voice or person never selects a role (DashboardRole docstring).
"""

import pytest
from starlette.requests import Request

from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardPrincipal, DashboardRole, _permissions
from src.memory.capture_control import CaptureController


def _required(method: str, path: str) -> set[str]:
    return set(_route_permissions(Request({"type": "http", "method": method, "path": path,
                                           "headers": [], "query_string": b""})))


@pytest.mark.parametrize("method", ["POST", "GET"])
def test_forget_route_requires_admin_manage_only(method):
    assert _required(method, "/api/capture/control/forget-carried") == {"admin:manage"}


@pytest.mark.parametrize("role,user_id", [
    (DashboardRole.HOME, None), (DashboardRole.HOME, "user_" + "a" * 32),
    (DashboardRole.EVALUATOR, None),
])
def test_no_household_or_evaluator_role_can_mint_the_controller(role, user_id):
    permissions = _permissions(role, user_id)
    assert "admin:manage" not in permissions
    with pytest.raises(PermissionError):
        CaptureController.from_dashboard(DashboardPrincipal("synthetic-member", permissions),
                                         lambda: True)


def test_only_the_operator_role_holds_the_permission():
    assert "admin:manage" in _permissions(DashboardRole.OPERATOR, None)
    controller = CaptureController.from_dashboard(
        DashboardPrincipal("synthetic-operator", _permissions(DashboardRole.OPERATOR, None)),
        lambda: True)
    controller.require_current()
