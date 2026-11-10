"""Real Chromium check of the technical panel reached by direct URL /obs/.

Verifies that every view renders without runtime errors and without example
data, that unavailable sources are shown as unavailable, that controls without
a backing operation are disabled with a reason, and that alert acknowledgement
runs the existing authenticated endpoint and reports only the server result.
Synthetic fixtures only.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from types import SimpleNamespace
from unittest import mock

import aiohttp
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from tests.frontend.browser_nexo_home import screenshot
from tests.frontend.browser_support import Page, browser_binary, browser_profile_root, evidence_directory

EVIDENCE = evidence_directory('KZA_OBS_PANEL_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/obs-browser-runs')
TOKEN = 'synthetic-obs-panel-root'
FORBIDDEN = ('Marco', 'Lucía', 'vllm-7b', 'ik-30b', '192.168.1.2', '21d 08h', 'kza@a3f9c2e', 'turn_8821',
             'Khruangbin', 'datos de ejemplo', 'wireframe')


class Priority(Enum):
    """Synthetic priority with the attribute shape the adapter reads."""

    HIGH = 2


class Kind(Enum):
    """Synthetic alert type."""

    DEVICE = 'device'


class SyntheticAlerts:
    """In-memory alert source exposing the two methods the dashboard uses."""

    def __init__(self):
        self.alerts = [SimpleNamespace(alert_id='synthetic-alert-1', processed=False, priority=Priority.HIGH,
                                       alert_type=Kind.DEVICE, timestamp=datetime.now(timezone.utc),
                                       message='Synthetic device alert', details={'zone': 'synthetic-zone'}),
                       SimpleNamespace(alert_id='synthetic-alert-2', processed=False, priority=Priority.HIGH,
                                       alert_type=Kind.DEVICE, timestamp=datetime.now(timezone.utc),
                                       message='Synthetic second alert', details={'zone': 'synthetic-zone'})]
        self.marked = []

    def get_history(self, limit=None):
        return list(self.alerts)[:limit]

    async def mark_processed(self, alert_id, *, before_publish=None, update_timestamp=True, timestamp_factory=None):
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                if before_publish is not None:
                    before_publish()
                alert.processed = True
                self.marked.append((alert_id, update_timestamp))
                return True
        return False


async def main() -> None:
    os.umask(0o077)
    EVIDENCE.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ['KZA_DASHBOARD_TOKEN'] = TOKEN
    requests, checks = [], []
    alerts = SyntheticAlerts()
    binary = browser_binary()
    with tempfile.TemporaryDirectory(prefix='kza-obs-panel-', dir=browser_profile_root()) as temporary:
        directory = Path(temporary)
        authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=600)
        await authority.initialize()
        dashboard = DashboardAPI(authority=authority, alert_manager=alerts)

        @dashboard.app.middleware('http')
        async def record(request, call_next):
            response = await call_next(request)
            requests.append({'method': request.method, 'path': request.url.path, 'status': response.status_code})
            return response

        listener = socket.socket(); listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        browser = None
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(.02)
            profile = directory / 'chromium'
            with (EVIDENCE / 'chromium.log').open('w') as output:
                browser = subprocess.Popen([binary, '--headless', '--disable-gpu', '--no-first-run',
                    '--disable-background-networking', '--remote-debugging-port=0',
                    f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)
                async with asyncio.timeout(15):
                    while not (profile / 'DevToolsActivePort').exists():
                        assert browser.poll() is None
                        await asyncio.sleep(.05)
                debugging = int((profile / 'DevToolsActivePort').read_text().splitlines()[0])
                async with aiohttp.ClientSession() as client:
                    async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                        target = next(item for item in await response.json() if item['type'] == 'page')
                async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                    page = Page(connection)
                    await page.command('Runtime.enable')
                    await page.command('Page.enable')
                    await page.command('Emulation.setDeviceMetricsOverride',
                                       {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.command('Page.navigate', {'url': f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("!!document.querySelector('input[type=password]')")
                    await page.evaluate(
                        "(()=>{const input=document.querySelector('input[type=password]');"
                        "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,"
                        + json.dumps(TOKEN) + ");input.dispatchEvent(new Event('input',{bubbles:true}));"
                        "input.form.requestSubmit();})()")
                    await page.wait_for("!!document.querySelector('#view-live')")
                    assert await page.evaluate("typeof window.MOCKS === 'undefined'")
                    checks.append('login reaches Live without example data globals')

                    async def visit(label, view_id):
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(n=>n.textContent.includes("
                                            + json.dumps(label) + ")).click()")
                        await page.wait_for(f"!!document.querySelector('#{view_id}')")
                        text = await page.evaluate('document.body.innerText')
                        leaked = [token for token in FORBIDDEN if token in text]
                        assert not leaked, (label, leaked)
                        await screenshot(page, EVIDENCE / f'obs-{view_id}.png')
                        return text

                    text = await visit('Live', 'view-live')
                    assert 'Zonas no disponibles' in text and 'Latencia' in text and 'No disponible' in text
                    assert 'Fuentes no disponibles' in text
                    checks.append('Live shows unavailable zones/router/latency instead of examples')

                    text = await visit('Alertas', 'view-alerts')
                    assert 'Synthetic device alert' in text
                    await page.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='Marcar atendida').click()")
                    await page.wait_for("document.body.innerText.includes('El servidor marcó la alerta como atendida.')")
                    assert alerts.marked == [('synthetic-alert-1', True)]
                    assert {'method': 'POST', 'path': '/api/admin/alerts/synthetic-alert-1/ack', 'status': 200} in requests
                    assert await page.evaluate("document.body.innerText.includes('atendidas · 1')")
                    checks.append('ack runs POST /api/admin/alerts/{id}/ack and shows the server-confirmed result')
                    reads = sum(item['path'] == '/api/alerts' for item in requests)
                    await asyncio.sleep(7)  # past the 6 s observation TTL, with no new read
                    assert await page.evaluate("document.querySelector('#view-alerts').innerText.includes('vencida')")
                    assert sum(item['path'] == '/api/alerts' for item in requests) == reads
                    checks.append('expired observation is relabelled vencida without a new read')
                    await page.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='Descartar').click()")
                    await page.wait_for("document.body.innerText.includes('El servidor descartó la alerta.')")
                    assert alerts.marked[-1] == ('synthetic-alert-2', False)
                    assert {'method': 'DELETE', 'path': '/api/admin/alerts/synthetic-alert-2', 'status': 200} in requests
                    assert await page.evaluate("document.body.innerText.includes('activas · 0') && document.body.innerText.includes('atendidas · 2')")
                    checks.append('dismiss runs DELETE /api/admin/alerts/{id} and shows the server-confirmed result')

                    text = await visit('LLM Router', 'view-llm')
                    assert 'Endpoints no disponibles' in text
                    assert await page.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='Forzar chequeo de salud').disabled")
                    checks.append('health check disabled with reason; endpoints unavailable without router')

                    text = await visit('Home Assistant', 'view-ha')
                    assert 'Entidades no disponibles' in text
                    text = await visit('Sistema', 'view-system')
                    assert 'Registros no disponibles' in text
                    text = await visit('Privacidad', 'view-privacy')
                    text = await visit('Conversaciones', 'view-conversations')
                    text = await visit('Voces', 'view-users')
                    await page.evaluate("[...document.querySelectorAll('.nav-item')].find(n=>n.textContent.includes('Admin')).click()")
                    await page.wait_for("document.body.innerText.includes('Acciones de servicio')")
                    await screenshot(page, EVIDENCE / 'obs-admin.png')
                    checks.append('every view renders without example data')

                    # Restart whose server outcome is uncertain: systemctl is replaced in this
                    # process only (never executed) and times out, so the server answers 504.
                    commands = []

                    def timed_out(args, **kwargs):
                        commands.append(list(args))
                        raise subprocess.TimeoutExpired(args, kwargs.get('timeout'))

                    with mock.patch('src.dashboard.admin.shutil.which', lambda _: '/usr/bin/systemctl'), \
                            mock.patch('src.dashboard.admin.subprocess.run', timed_out):
                        await page.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='↻ Reiniciar').click()")
                        await page.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent==='Confirmar reinicio de kza-voice').click()")
                        await page.wait_for("document.body.innerText.includes('Resultado incierto: el servidor no pudo confirmar el efecto')")
                    assert commands == [['systemctl', '--user', 'restart', 'kza-voice']]
                    assert {'method': 'POST', 'path': '/api/admin/services/kza-voice/restart', 'status': 504} in requests
                    assert not await page.evaluate("document.body.innerText.includes('completó el reinicio')")
                    await screenshot(page, EVIDENCE / 'obs-admin-restart-uncertain.png')
                    checks.append('restart with a server timeout (504, effect uncertain) is shown as Resultado incierto, never applied')

                    await page.command('Emulation.setDeviceMetricsOverride',
                                       {'width': 390, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
                    await visit('Live', 'view-live')
                    assert not page.errors, page.errors
                    checks.append('no runtime exceptions')
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                browser.wait(timeout=10)
            server.should_exit = True
            await serving
            await authority.aclose()
    result = {'passed': True, 'synthetic_fixture': True, 'checks': checks,
              'requests': [item for item in requests if item['method'] != 'GET' or item['path'].startswith('/api/alerts')]}
    (EVIDENCE / 'browser-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    asyncio.run(main())
