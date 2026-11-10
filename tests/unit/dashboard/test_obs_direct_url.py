"""NEXO no longer links to the technical panel; /obs/ stays reachable by direct URL."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from src.dashboard.api import DashboardAPI

ROOT = Path(__file__).resolve().parents[3]
HOME = ROOT / "src/dashboard/frontend/home"
OBS = ROOT / "src/dashboard/frontend/obs"

# A navigable reference to the technical panel: href attributes, property
# assignments, location changes or window.open, but not module imports.
OBS_LINK = re.compile(r"""(href\s*=\s*["']?/obs|\.href\s*=\s*["'`]/obs|location[^;\n]*["'`]/obs|open\(\s*["'`]/obs)""")


def _nexo_files():
    files = [HOME / "index.html"]
    files += sorted(path for path in (HOME / "nexo").rglob("*") if path.suffix in {".js", ".css", ".html"})
    return files


def test_nexo_sources_have_no_technical_panel_link():
    hits = []
    for path in _nexo_files():
        text = path.read_text(encoding="utf-8")
        for token in ("technical-link", "demo-badge", "Panel técnico"):
            if token in text:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
        hits += [f"{path.relative_to(ROOT)}: {match.group(0)}" for match in OBS_LINK.finditer(text)]
    assert hits == []


def test_nexo_still_imports_the_shared_privacy_modules_from_obs():
    app = (HOME / "nexo/app.js").read_text(encoding="utf-8")
    for module in ("live-privacy.js", "conversation-channel.js", "voice-privacy.js", "voice-channel.js"):
        assert f"import '../../obs/src/{module}';" in app


def test_served_nexo_shell_has_no_obs_link_while_obs_stays_reachable(monkeypatch):
    monkeypatch.delenv("KZA_DASHBOARD_TOKEN", raising=False)
    with TestClient(DashboardAPI(home_enabled=True).app) as client:
        shell = client.get("/")
        assert shell.status_code == 200
        assert "technical-link" not in shell.text and not OBS_LINK.search(shell.text)
        panel = client.get("/obs/")
        assert panel.status_code == 200
        assert "<title>KZA — Panel técnico</title>" in panel.text
        assert client.get("/obs/src/obs-data.js").status_code == 200
        for retired in ("/obs/src/mocks.jsx", "/obs/tweaks-panel.jsx"):
            assert client.get(retired).status_code == 404


def test_obs_panel_ships_no_example_data():
    for path in OBS.rglob("*"):
        if "vendor" in path.parts or path.suffix not in {".js", ".jsx", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("window.MOCKS", "mocks.jsx", "tweaks-panel", "192.168.1.2", "kza@a3f9c2e", "turn_8821"):
            assert token not in text, f"{path.relative_to(ROOT)}: {token}"
