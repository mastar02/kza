"""T040/T041 U4: the legacy home page and the old root dashboard are retired.

NEXO (``/``) keeps Rutinas and Memoria; the technical panel stays reachable only
by direct URL. Retiring static files changes no route permission.
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from src.dashboard.api import DashboardAPI
from src.dashboard.auth import _route_permissions
from src.dashboard.authority import _HOME_PERMISSIONS, DashboardPermission

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "src/dashboard/frontend"

RETIRED_URLS = (
    "/home/legacy.html", "/home/home.css", "/home/routines.js", "/home/memory.js",
    "/index.html", "/dashboard.js", "/style.css",
)
RETIRED_FILES = (
    "home/legacy.html", "home/home.css", "home/routines.js", "home/memory.js",
    "index.html", "dashboard.js", "style.css",
)
# References to a retired file or to the globals only those files defined.
RETIRED_REFERENCE = re.compile(
    r"legacy\.html|home\.css|home/routines\.js|home/memory\.js|(?<![\w}])/index\.html"
    r"|dashboard/frontend/index\.html|(?<!\w)dashboard\.js|frontend/style\.css"
    r"|RoutinesUI|MemoryUI|additional-tools|Micrófonos y diagnóstico"
)
SCANNED_DIRS = ("src", "tests", "scripts", "tools", "config")
SCANNED_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".jsx", ".html", ".css", ".yaml", ".yml", ".json", ".sh", ".toml"}
# Files that name the retired paths only to assert their absence.
NEGATIVE_TESTS = {
    "tests/unit/dashboard/test_legacy_views_retired.py",
    "tests/unit/tools/test_stage_runtime.py",
    "tests/integration/test_memory_services.py",
    "tests/frontend/browser_nexo_home.py",
    "tests/frontend/browser_nexo_routines.py",
}


def test_retired_files_are_gone_and_shared_library_stays():
    assert [name for name in RETIRED_FILES if (FRONTEND / name).exists()] == []
    for kept in ("home/index.html", "home/home.js", "home/request-id.js", "home/nexo/routines.js", "home/nexo/memory.js"):
        assert (FRONTEND / kept).is_file(), kept


def test_retired_urls_answer_404_and_nexo_keeps_serving(monkeypatch):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    with TestClient(DashboardAPI(home_enabled=True).app) as client:
        for path in RETIRED_URLS:
            assert client.get(path).status_code == 404, path
        for path in ("/home/home.js", "/home/request-id.js", "/home/nexo/routines.js", "/home/nexo/memory.js", "/obs/"):
            assert client.get(path).status_code == 200, path
        shell = client.get("/")
        assert shell.status_code == 200
        for token in ("legacy.html", 'href="/obs', "technical-link", "additional-tools", "/index.html", "home.css"):
            assert token not in shell.text, token


def test_root_with_home_disabled_explains_instead_of_serving_the_retired_grid(monkeypatch):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    with TestClient(DashboardAPI(home_enabled=False).app) as client:
        root = client.get("/")
        assert root.status_code == 404
        assert root.headers["cache-control"] == "no-store"
        assert "dashboard.home.enabled" in root.text and "<a " not in root.text
        for path in RETIRED_URLS:
            assert client.get(path).status_code == 404, path
        assert client.get("/obs/").status_code == 200


def test_no_source_references_the_retired_views():
    hits = []
    for directory in SCANNED_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT).as_posix()
            if (not path.is_file() or path.suffix not in SCANNED_SUFFIXES or relative in NEGATIVE_TESTS
                    or "vendor" in path.parts or "__pycache__" in path.parts or "node_modules" in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits += [f"{relative}: {match.group(0)}" for match in RETIRED_REFERENCE.finditer(text)]
    assert hits == []


def _permissions(method: str, path: str) -> frozenset[str]:
    return _route_permissions(Request({"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}))


def test_retirement_changes_no_permission():
    assert {member.value for member in DashboardPermission} == {
        "home:read", "home:control", "tasks:read", "tasks:write", "tasks:personal", "routines:execute",
        "routines:manage", "diagnostics:read", "conversations:read", "memory:read", "memory:review",
        "voices:read", "voices:review", "evaluation:read", "ambient:control", "admin:manage", "auth:manage",
        # Added by U3b-2 (per-person memory), not by the legacy retirement.
        "memory:assign", "memory:own:read", "memory:own:write",
    }
    assert _HOME_PERMISSIONS == {"home:read", "home:control", "tasks:read", "tasks:write", "routines:execute"}
    expected = {
        ("GET", "/api/home/microphones"): {"diagnostics:read"},
        ("GET", "/api/routines"): {"home:read"},
        ("POST", "/api/routines"): {"routines:manage"},
        ("POST", "/api/routines/r1/execute"): {"routines:manage"},
        ("GET", "/api/routines/r1/runs/x"): {"routines:manage"},
        ("GET", "/api/memory"): {"memory:read"},
        ("POST", "/api/memory/proposals/p1/confirm"): {"memory:review"},
        ("DELETE", "/api/memory/facts/f1"): {"memory:review"},
        ("GET", "/api/health"): {"diagnostics:read"},
    }
    assert {key: set(_permissions(*key)) for key in expected} == expected


def test_private_apis_behind_retired_views_still_require_a_session(dashboard_auth):
    with TestClient(DashboardAPI(home_enabled=True).app) as client:
        for path in ("/api/home/microphones", "/api/routines", "/api/memory", "/api/health"):
            assert client.get(path).status_code == 401, path
