"""Manual Chromium smoke against the real UI and a synthetic local HA graph.

Run as yo: python -m tests.frontend.browser_typed_controls
No real HA, microphone, model or production port is used. Requires Chromium
and the development environment's existing websockets/uvicorn packages.
"""

import asyncio
import base64
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from unittest.mock import AsyncMock

import aiohttp
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.home_assistant.ha_client import HAAreaAssignments, HAStatesSnapshot
from src.routines.factory import build_routine_scheduler
from tests.integration.test_typed_home_operations import typed_graph


class Page:
    """Small sequential CDP client using an existing development dependency."""

    def __init__(self, connection):
        self.connection = connection
        self.sequence = 0
        self.errors = []

    async def command(self, method, params=None):
        self.sequence += 1
        identity = self.sequence
        await self.connection.send(json.dumps({"id": identity, "method": method, "params": params or {}}))
        while True:
            result = json.loads(await asyncio.wait_for(self.connection.recv(), 15))
            if result.get("method") == "Runtime.exceptionThrown":
                self.errors.append(result["params"])
            if result.get("id") == identity:
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result.get("result", {})

    async def evaluate(self, expression):
        result = await self.command("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    async def wait_for(self, expression):
        for _ in range(100):
            if await self.evaluate(expression):
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Browser condition timed out: " + expression)


async def main():
    """Render and exercise typed controls with actual browser events and HTTP."""
    os.environ["KZA_DASHBOARD_TOKEN"] = "synthetic-browser-session"
    ha, actions = typed_graph.__wrapped__()
    ha.get_state_snapshot = AsyncMock(side_effect=lambda: HAStatesSnapshot(True, datetime.now(UTC), None, tuple(ha.states.values())))
    ha.get_area_assignments = AsyncMock(return_value=HAAreaAssignments(True, datetime.now(UTC), None))
    # Chromium's Snap uses a private /tmp namespace. A disposable profile in
    # its own user data directory is visible to the driver and the browser.
    browser_data = Path.home() / "snap/chromium/common"
    browser_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kza-browser-", dir=browser_data) as work:
        directory = Path(work)
        scheduler = await build_routine_scheduler(db_path=str(directory / "routines.db"), timezone="America/Montevideo", ha_client=ha, home_actions=actions)
        dashboard = DashboardAPI(home_enabled=True, ha_client=ha, home_actions=actions, routine_scheduler=scheduler)

        @dashboard.app.middleware("http")
        async def fixture_cookie(request, call_next):
            response = await call_next(request)
            if request.url.path == "/":
                response.set_cookie("kza_dashboard_token", "synthetic-browser-session", httponly=True, samesite="strict")
            return response

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level="warning"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        browser = None
        try:
            for _ in range(100):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            profile = directory / "chromium"
            with (directory / "chromium.log").open("w") as output:
                browser = subprocess.Popen([
                    "chromium", "--headless", "--disable-gpu", "--no-first-run",
                    "--no-default-browser-check", "--remote-debugging-port=0", f"--user-data-dir={profile}",
                    f"http://127.0.0.1:{port}/",
                ], stdout=output, stderr=output)
                active_port = profile / "DevToolsActivePort"
                for _ in range(200):
                    if active_port.exists():
                        break
                    if browser.poll() is not None:
                        raise RuntimeError((directory / "chromium.log").read_text()[-2000:])
                    await asyncio.sleep(0.05)
                debugging = int(active_port.read_text().splitlines()[0])
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{debugging}/json/list") as response:
                        pages = await response.json()
                target = next(page for page in pages if page["type"] == "page")
                async with connect(target["webSocketDebuggerUrl"], max_size=None) as connection:
                    page = Page(connection)
                    await page.command("Runtime.enable")
                    await page.wait_for("document.querySelector('[aria-label=\"Temperatura objetivo (°C) de climate.test_ac\"]') !== null")
                    await page.evaluate("""(() => {
                        const input = document.querySelector('[aria-label="Temperatura objetivo (°C) de climate.test_ac"]');
                        input.value = '23.5'; input.dispatchEvent(new Event('input', {bubbles:true}));
                        input.closest('.typed-controls').querySelector('button').click();
                    })()""")
                    await page.wait_for("document.body.textContent.includes('Temperatura objetivo · Confirmado por HA')")
                    assert ha.sends[-1][3] == {"temperature": 23.5}
                    await page.evaluate("""(() => {
                        const input = document.querySelector('[aria-label="Temperatura de color (K) de light.test_lamp"]');
                        input.value = '2700'; input.dispatchEvent(new Event('input', {bubbles:true}));
                        const brightness = document.querySelector('[aria-label="Brillo opcional (%) de light.test_lamp"]');
                        brightness.value = '30'; brightness.dispatchEvent(new Event('input', {bubbles:true}));
                        input.closest('.typed-controls').querySelector('button').click();
                    })()""")
                    await page.wait_for("document.body.textContent.includes('Temperatura de color · Confirmado por HA')")
                    assert ha.sends[-1][3] == {"brightness_pct": 30, "color_temp_kelvin": 2700}
                    house_image = await page.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                    house_path = Path("/tmp/kza-typed-controls-house.png")
                    house_path.write_bytes(base64.b64decode(house_image["data"]))
                    house_path.chmod(0o600)
                    await page.evaluate("document.querySelector('[data-view=\"routines\"]').click()")
                    await page.wait_for("[...document.querySelectorAll('button')].some(button => button.textContent === 'Crear rutina')")
                    await page.evaluate("[...document.querySelectorAll('button')].find(button => button.textContent === 'Crear rutina').click()")
                    await page.evaluate("""(() => {
                        const name = document.querySelector('[data-routine-field="Nombre"]');
                        name.value = 'Synthetic typed routine'; name.dispatchEvent(new Event('input', {bubbles:true}));
                        [...document.querySelectorAll('button')].find(button => button.textContent === 'Agregar acción').click();
                        const entity = document.querySelector('[data-routine-field="Dispositivo de la acción 1"]');
                        entity.value = 'climate.test_ac'; entity.dispatchEvent(new Event('change', {bubbles:true}));
                        const operation = document.querySelector('[data-routine-field="Operación 1"]');
                        operation.value = 'set_temperature'; operation.dispatchEvent(new Event('change', {bubbles:true}));
                        const temperature = document.querySelector('[data-routine-field="Temperatura objetivo (°C) de la acción 1"]');
                        temperature.value = '24.5'; temperature.dispatchEvent(new Event('input', {bubbles:true}));
                        [...document.querySelectorAll('button')].find(button => button.textContent === 'Guardar').click();
                    })()""")
                    await page.wait_for("document.body.textContent.includes('Guardada')")
                    stored = await scheduler.get_all_routines()
                    assert len(stored) == 1 and stored[0].spec.actions[0].temperature == 24.5
                    assert len(ha.sends) == 2  # Saving the routine cannot execute it.
                    assert not page.errors, page.errors
                    screenshot = await page.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                    output_image = Path("/tmp/kza-typed-controls-browser.png")
                    output_image.write_bytes(base64.b64decode(screenshot["data"]))
                    output_image.chmod(0o600)
                    print(json.dumps({"passed": True, "typed_sends": len(ha.sends), "stored_temperature": 24.5, "screenshot": str(output_image), "javascript_exceptions": len(page.errors)}))
        finally:
            if browser is not None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, timeout=10)
            server.should_exit = True
            await serving
            await scheduler.stop()
            await scheduler.store.close()
            await actions.close()


if __name__ == "__main__":
    asyncio.run(main())
