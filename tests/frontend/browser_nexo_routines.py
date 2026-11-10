"""Real Chromium, real logins and a real routine scheduler for the NEXO Rutinas view.

Synthetic credentials stay in process memory. Evidence never serializes a
credential, a cookie or a request body; only methods, paths and counts.
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
import socket

import aiohttp
import uvicorn
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from src.rooms.room_context import RoomConfig
from src.routines.factory import build_routine_scheduler
from src.routines.models import RunStatus, parse_routine_spec
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_support import Page, _press_tab, audit_view, browser_binary, browser_profile_root, evidence_directory, install_login_session, press_enter
from tests.integration.test_typed_home_operations import typed_graph


EVIDENCE = evidence_directory('KZA_NEXO_ROUTINES_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/u2-routines-browser-runs')
WIDTHS = (320, 390, 768, 1024, 1280, 1440)
TZ = 'America/Montevideo'
INSTRUMENT = r"""(() => {
  const fetcher = window.fetch.bind(window);
  window.__qaRules = [];
  window.fetch = async (input, options = {}) => {
    const path = String(input), method = (options.method || 'GET').toUpperCase();
    const index = window.__qaRules.findIndex(rule => rule.method === method && path.startsWith(rule.path));
    const rule = index >= 0 ? window.__qaRules.splice(index, 1)[0] : null;
    const response = await fetcher(input, options);
    // The server has answered in both modes: 'hold' delays delivery, 'drop' loses it.
    if (rule?.mode === 'hold') await new Promise(resolve => { window.__qaRelease = resolve; window.__qaHeld = path; });
    if (rule?.mode === 'drop') { await response.text(); throw new TypeError('Synthetic lost response'); }
    return response;
  };
})()"""


def definition(name: str, *, enabled: bool = True) -> dict:
    return {'name': name, 'description': '', 'enabled': enabled, 'timezone': TZ, 'schedule': None, 'conditions': [],
            'actions': [{'entity_id': 'light.test_lamp', 'operation': 'turn_on'}]}


def run_count(database: Path, routine_id: str | None = None) -> int:
    with sqlite3.connect(database) as connection:
        if routine_id is None:
            return connection.execute('SELECT COUNT(*) FROM routine_runs').fetchone()[0]
        return connection.execute('SELECT COUNT(*) FROM routine_runs WHERE routine_id = ?', (routine_id,)).fetchone()[0]


def key(value: str) -> str:
    return json.dumps(f'[data-key="{value}"]')


async def main() -> None:
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-routines-bootstrap'
    evidence = EVIDENCE
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    requests: list[dict] = []
    checks: list[str] = []

    class DiagnosticPage(Page):
        async def wait_for(self, expression):
            try:
                for _ in range(400):
                    if await self.evaluate('!!(' + expression + ')'):
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError('Browser condition timed out: ' + expression)
            except Exception:
                diagnostic = await self.evaluate("({text:document.body.innerText.slice(0,4000),dialog:document.querySelector('dialog[open]')?.innerText||null})")
                (evidence / 'browser-failure.json').write_text(json.dumps({'condition': expression, 'document': diagnostic, 'errors': self.errors,
                                                                          'requests': requests[-40:]}, indent=2, ensure_ascii=False) + '\n')
                await screenshot(self, evidence / 'browser-failure.png')
                raise

    binary = browser_binary()
    browser_data = browser_profile_root(mode=0o700)
    ha, actions = typed_graph.__wrapped__()
    for entity, name in [('light.test_lamp', 'Luz principal'), ('climate.test_ac', 'Aire acondicionado'), ('cover.test_blind', 'Persiana')]:
        ha.states[entity]['attributes']['friendly_name'] = name
    with tempfile.TemporaryDirectory(prefix='kza-nexo-routines-', dir=browser_data) as temporary:
        directory = Path(temporary)
        database = directory / 'routines.db'
        scheduler = await build_routine_scheduler(db_path=str(database), timezone=TZ, ha_client=ha, home_actions=actions)
        authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=600)
        await authority.initialize()
        root = authority.resolve('Bearer synthetic-routines-bootstrap')
        assert root is not None
        expires = time.time() + 900
        operator = await authority.issue_credential(root, subject='synthetic-operator', role=DashboardRole.OPERATOR, expires_at=expires)
        operator_b = await authority.issue_credential(root, subject='synthetic-operator-b', role=DashboardRole.OPERATOR, expires_at=expires)
        household = await authority.issue_credential(root, subject='synthetic-home', role=DashboardRole.HOME, expires_at=expires)
        selection = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
        evaluator = await authority.issue_credential(root, subject='synthetic-evaluator', role=DashboardRole.EVALUATOR, expires_at=expires, samples=(selection,))
        rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp', default_cover='cover.test_blind'),
                 'kitchen': RoomConfig('kitchen', 'Cocina', 'la cocina', default_climate='climate.test_ac')}
        dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms, routine_scheduler=scheduler)

        async def unavailable_live_channel(websocket):
            await websocket.close(code=1013)

        for route in ['/ws/voices', '/ws/evaluation']:
            dashboard.app.router.routes.insert(0, WebSocketRoute(route, unavailable_live_channel))

        @dashboard.app.middleware('http')
        async def record_request(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith('/api/'):
                requests.append({'method': request.method, 'path': request.url.path, 'status': response.status_code})
            return response

        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        dashboard._server, dashboard._serve_task = server, serving
        browser = None
        try:
            for _ in range(150):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            assert server.started
            profile = directory / 'chromium'
            with (evidence / 'chromium.log').open('w') as output:
                browser = await asyncio.to_thread(subprocess.Popen, [
                    binary, '--headless', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                    '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update',
                    '--disable-sync', '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank',
                ], stdout=output, stderr=output)
                active_port = profile / 'DevToolsActivePort'
                for _ in range(200):
                    if active_port.exists():
                        break
                    assert browser.poll() is None, 'Chromium exited; see private browser log'
                    await asyncio.sleep(0.05)
                debugging = int(active_port.read_text().splitlines()[0])
                async with aiohttp.ClientSession() as client:
                    async with client.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                        targets = await response.json()
                target = next(item for item in targets if item['type'] == 'page')
                async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                    page = DiagnosticPage(connection)
                    await page.command('Runtime.enable')
                    await page.command('Page.enable')
                    await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': INSTRUMENT})
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.command('Page.navigate', {'url': f'http://127.0.0.1:{port}/'})
                    await page.wait_for("document.querySelector('#login-panel') && document.querySelector('#login-error').textContent.length>0")

                    async def click(selector: str) -> None:
                        await page.evaluate(f'document.querySelector({json.dumps(selector)}).click()')

                    async def press(selector: str) -> None:
                        # Card controls are disabled while a read or write for the routine is in flight.
                        await page.wait_for(f'(e=>e&&!e.disabled)(document.querySelector({json.dumps(selector)}))')
                        await click(selector)

                    async def fill(field: str, value) -> None:
                        await page.evaluate(f"(()=>{{const e=document.querySelector('#routine-dialog '+{key(field)});e.value={json.dumps(value)};"
                                            "e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()")

                    async def rule(method: str, path: str, mode: str) -> None:
                        await page.evaluate(f'window.__qaRules.push({json.dumps({"method": method, "path": path, "mode": mode})})')

                    def card(routine_id: str, inner: str = '') -> str:
                        return json.dumps(f'#routines-list [data-routine-id="{routine_id}"] {inner}'.strip())

                    async def ids() -> list[str]:
                        return [item.routine_id for item in await scheduler.get_all_routines()]

                    # Operator: navigation, empty state, failed save focus, preview, create.
                    await login(page, operator.token, 'Administración')
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,voices,people,actions,routines,memory,access'")
                    assert await page.evaluate("document.querySelectorAll('a[href*=\"legacy.html\"]').length===0")
                    await click('#app-nav [data-view=routines]')
                    await page.wait_for("location.hash==='#rutinas' && !document.querySelector('#routines-view').hidden && !document.querySelector('.routines-empty').hidden")
                    assert await page.evaluate("document.querySelector('.routines-empty').textContent.includes('Todavía no hay rutinas')")
                    checks.append('nav entry and #rutinas hash; empty state from a real empty store')
                    await click('#routines-create')
                    await page.wait_for("document.querySelector('#routine-dialog').open && document.activeElement?.dataset.key==='name'")
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for("document.activeElement?.id==='routine-errors' && document.querySelector('#routine-errors').textContent.includes('Escribí un nombre')")
                    checks.append('failed save moves focus to the validation summary')
                    await fill('name', 'Noche tranquila')
                    await fill('mode', 'scheduled')
                    assert await page.evaluate("document.querySelector('#routine-dialog [data-key=hour]').value==='' && document.querySelector('#routine-dialog [data-key=minute]').value===''"), 'no invented time'
                    assert await page.evaluate("[...document.querySelectorAll('#routine-dialog .routine-day')].every(b=>b.getAttribute('aria-pressed')==='false' && b.getAttribute('aria-label').length>4)")
                    await fill('hour', '22')
                    await fill('minute', '30')
                    await click('#routine-dialog [data-key="day:0"]')
                    await click('#routine-dialog [data-key="day:2"]')
                    await click('#routine-dialog [data-key="condition:add"]')
                    await fill('condition:0:entity', 'cover.test_blind')
                    await fill('condition:0:state', 'closed')
                    await click('#routine-dialog [data-key="action:add"]')
                    await fill('action:0:entity', 'light.test_lamp')
                    await fill('action:0:operation', 'turn_on')
                    await click('#routine-dialog [data-key="action:add"]')
                    await fill('action:1:entity', 'cover.test_blind')
                    await fill('action:1:operation', 'open_cover')
                    await click('#routine-dialog [data-key="preview"]')
                    await page.wait_for("document.querySelector('#routine-dialog .routine-preview')?.textContent.includes('Vista previa · sin guardar ni ejecutar')")
                    assert await page.evaluate("document.querySelector('#routine-dialog .routine-preview').textContent.includes('Lu, Mi · 22:30')")
                    assert await ids() == [], 'preview writes nothing'
                    checks.append('preview normalizes without writing')
                    audits = {}
                    audits['rutinas-editor'] = await audit_view(page, evidence, 'rutinas-editor', (
                        ('#routine-dialog [data-key=name]', None), ('#routine-dialog [data-key="day:6"]', None), ('#routine-dialog [data-key="action:1:up"]', None),
                        ('#routine-dialog [data-key=preview]', None), ('#routine-dialog [data-key=save]', None)), screenshot, widths=WIDTHS)
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for("!document.querySelector('#routine-dialog').open && document.querySelectorAll('#routines-list .routine-card').length===1")
                    [main_id] = await ids()
                    assert await page.evaluate(f"document.querySelector({card(main_id)}).textContent.includes('Revisión 1') && document.querySelector({card(main_id)}).textContent.includes('Lu, Mi · 22:30 · Montevideo')")
                    assert await page.evaluate("document.querySelector('#routines-live').textContent.includes('Revisión 1')")
                    checks.append('create through the dialog persists revision 1; live region announces it')

                    # Edit (remove the condition so runs are not skipped), disable/enable.
                    await press(f'{json.loads(card(main_id))} [data-key="{main_id}:edit"]')
                    await page.wait_for("document.querySelector('#routine-dialog').open")
                    await click('#routine-dialog [data-key="condition:0:remove"]')
                    await fill('name', 'Noche')
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for(f"!document.querySelector('#routine-dialog').open && document.querySelector({card(main_id)}).textContent.includes('Revisión 2')")
                    assert await page.evaluate(f"document.activeElement?.dataset.key==='{main_id}:edit'"), 'focus returns to the invoking control'
                    await press(f'[data-key="{main_id}:toggle"]')
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Desactivada') && document.querySelector({card(main_id)}).textContent.includes('Revisión 3')")
                    await press(f'[data-key="{main_id}:toggle"]')
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Revisión 4')")
                    checks.append('edit, disable and enable through CAS revisions')

                    # Execute, then a lost execute response survives a reload and is consulted, not resent.
                    sends = len(ha.sends)
                    await press(f'[data-key="{main_id}:execute"]')
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Resultado de tu intento') && document.querySelector({card(main_id, '[data-run-status=succeeded]')})")
                    assert run_count(database, main_id) == 1 and len(ha.sends) == sends + 2
                    await rule('POST', f'/api/routines/{main_id}/execute', 'drop')
                    await press(f'[data-key="{main_id}:execute"]')
                    await page.wait_for(f"document.querySelector({card(main_id, '[data-routine-notice]')})?.textContent.startsWith('Resultado pendiente de consulta. No se pudo contactar')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['state'] == 'unresolved' and stored['request_id']
                    assert run_count(database, main_id) == 2, ('the server committed the lost send', run_count(database, main_id), requests[-6:])
                    await page.command('Page.reload')
                    await page.wait_for(f"!document.querySelector('#routines-view').hidden && document.querySelector({card(main_id)})?.textContent.includes('Resultado pendiente de consulta')")
                    assert await page.evaluate(f"!document.querySelector('[data-key=\"{main_id}:execute\"]')"), 'no fresh one-click execute while unresolved'
                    await press(f'[data-key="{main_id}:consult"]')
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Resultado de tu intento') && !document.querySelector({card(main_id)}).textContent.includes('Resultado pendiente')")
                    assert run_count(database, main_id) == 2, 'consult never resends'
                    latest = await scheduler.get_latest_run(main_id)
                    assert await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}')).run_id") == latest.run_id
                    checks.append('lost execute response: reload keeps the request identity, consult finds the same run, one run on the server')

                    # Leaving the view during an execute never aborts it.
                    await rule('POST', f'/api/routines/{main_id}/execute', 'hold')
                    await press(f'[data-key="{main_id}:execute"]')
                    await page.wait_for('window.__qaHeld')
                    await click('#app-nav [data-view=home]')
                    await page.wait_for("!document.querySelector('#home-view').hidden")
                    await page.evaluate('window.__qaRelease()')
                    await click('#app-nav [data-view=routines]')
                    await page.wait_for(f"document.querySelector({card(main_id)})?.textContent.includes('Resultado de tu intento') && !document.querySelector({card(main_id)}).textContent.includes('pendiente')")
                    assert run_count(database, main_id) == 3
                    checks.append('navigating away mid-execute keeps the write; exactly one more run')

                    # Another account takes over the shared session cookie (another tab), then this tab reloads.
                    await rule('POST', f'/api/routines/{main_id}/execute', 'drop')
                    await press(f'[data-key="{main_id}:execute"]')
                    await page.wait_for(f"document.querySelector({card(main_id, '[data-routine-notice]')})?.textContent.startsWith('Resultado pendiente de consulta')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['subject'] == 'synthetic-operator' and stored['state'] == 'unresolved'
                    runs_before_switch = run_count(database, main_id)
                    await install_login_session(page, f'http://127.0.0.1:{port}', operator_b.token)
                    marker = len(requests)
                    await page.command('Page.reload')
                    await page.wait_for(f"!document.querySelector('#routines-view').hidden && document.querySelector({card(main_id)})")
                    await page.wait_for("document.querySelector('.routine-session-notice') && !document.querySelector('.routine-session-notice').hidden")
                    assert await page.evaluate(f"sessionStorage.getItem('kza:routine-attempt:{main_id}')===null")
                    assert await page.evaluate(f"!document.querySelector({card(main_id)}).textContent.includes('Resultado pendiente') && !document.querySelector('[data-key=\"{main_id}:retry\"]')")
                    await asyncio.sleep(0.5)
                    assert not any(item['method'] == 'POST' or item['path'].endswith('/attempts') for item in requests[marker:]), 'no consult or resend of the other account identity'
                    assert run_count(database, main_id) == runs_before_switch
                    checks.append("account switch under the same tab: the other account's pending identity is dropped on reload, never consulted or resent; a reminder is shown")

                    # R2b: the tab is only hidden (no reload) while another tab switches the shared cookie back to A.
                    await rule('POST', f'/api/routines/{main_id}/execute', 'drop')
                    await press(f'[data-key="{main_id}:execute"]')
                    await page.wait_for(f"document.querySelector({card(main_id, '[data-routine-notice]')})?.textContent.startsWith('Resultado pendiente de consulta')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['subject'] == 'synthetic-operator-b' and stored['state'] == 'unresolved'
                    runs_before_takeover = run_count(database, main_id)
                    await page.evaluate("""(()=>{window.__qaHidden=true;
                      Object.defineProperty(document,'hidden',{configurable:true,get:()=>window.__qaHidden});
                      Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>window.__qaHidden?'hidden':'visible'});
                      document.dispatchEvent(new Event('visibilitychange'));})()""")
                    await install_login_session(page, f'http://127.0.0.1:{port}', operator.token)
                    marker = len(requests)
                    await page.evaluate("window.__qaHidden=false;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for("document.querySelector('.routine-session-notice') && !document.querySelector('.routine-session-notice').hidden && !document.querySelector('#routines-view').hidden")
                    await page.wait_for(f"document.querySelector({card(main_id)}) && sessionStorage.getItem('kza:routine-attempt:{main_id}')===null")
                    await asyncio.sleep(0.5)
                    after = requests[marker:]
                    assert after and after[0]['path'] == '/api/admin/auth/whoami', after[:4]
                    assert not any(item['method'] == 'POST' and item['path'] != '/api/admin/auth/login' or item['path'].endswith('/attempts') or '/runs/' in item['path'] for item in after), after
                    assert run_count(database, main_id) == runs_before_takeover, 'nothing resent under the other account'
                    assert await page.evaluate(f"document.querySelector('.routine-session-notice').textContent.includes('Revisá el último resultado') && !document.querySelector('[data-key=\"{main_id}:retry\"]')")
                    await page.command('Page.reload')
                    await page.wait_for(f"!document.querySelector('#routines-view').hidden && document.querySelector({card(main_id)})")
                    checks.append('hidden tab (no reload) while the cookie switches account: on return whoami is asked first, nothing is retried or consulted, the identity is dropped, the reminder shows and the session is re-read')

                    # Lost PUT: resolved by comparison, not by a second write.
                    await press(f'[data-key="{main_id}:edit"]')
                    await page.wait_for("document.querySelector('#routine-dialog').open")
                    await fill('name', 'Noche larga')
                    await rule('PUT', f'/api/routines/{main_id}', 'drop')
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for(f"!document.querySelector('#routine-dialog').open && document.querySelector({card(main_id)}).textContent.includes('Revisión 5') && document.querySelector({card(main_id)}).textContent.includes('El cambio se aplicó.')")
                    assert sum(1 for item in requests if item['method'] == 'PUT') == 2
                    checks.append('lost PUT resolves as applied after a list read; no second PUT')

                    # Lost DELETE on a second routine.
                    doomed = await scheduler.register_routine(parse_routine_spec(definition('Para borrar')), actor='synthetic-seed')
                    await click('#routines-refresh')
                    await page.wait_for(f"document.querySelector({card(doomed.routine_id)})")
                    await rule('DELETE', f'/api/routines/{doomed.routine_id}', 'drop')
                    await press(f'[data-key="{doomed.routine_id}:delete"]')
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open && document.activeElement?.dataset.key==='confirm-cancel'")
                    await click('#routine-confirm-dialog [data-key="confirm-accept"]')
                    await page.wait_for(f"!document.querySelector({card(doomed.routine_id)}) && document.querySelector('#routines-live').textContent==='La rutina se borró.'")
                    assert doomed.routine_id not in await ids()
                    checks.append('delete asks in a dialog; lost DELETE resolves as applied')

                    # Uncertain latest run (scheduled, no local attempt) requires confirmation before repeating.
                    risky = await scheduler.register_routine(parse_routine_spec(definition('Incierta')), actor='synthetic-seed')
                    claim = await scheduler.store.claim_run(risky.routine_id, json.dumps(['seed']), actor='synthetic-schedule', source='schedule')
                    await scheduler.store.mark_action_sending(claim.run.run_id, 0)
                    await scheduler.store.finish_run(claim.run.run_id, RunStatus.UNCERTAIN)
                    await click('#routines-refresh')
                    await page.wait_for(f"document.querySelector({card(risky.routine_id, '[data-run-status=uncertain]')})")
                    await press(f'[data-key="{risky.routine_id}:execute"]')
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open && document.querySelector('#routine-confirm-dialog').textContent.includes('Luz principal · Incierta')")
                    await screenshot(page, evidence / 'uncertain-confirm.png', full=False)
                    audits['rutinas-confirmacion'] = await audit_view(page, evidence, 'rutinas-confirmacion', (
                        ('#routine-confirm-dialog [data-key=confirm-cancel]', None), ('#routine-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                    await click('#routine-confirm-dialog [data-key="confirm-cancel"]')
                    await page.wait_for("!document.querySelector('#routine-confirm-dialog').open")
                    assert run_count(database, risky.routine_id) == 1, 'cancel sends nothing'
                    await press(f'[data-key="{risky.routine_id}:execute"]')
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open")
                    await click('#routine-confirm-dialog [data-key="confirm-accept"]')
                    await page.wait_for(f"document.querySelector({card(risky.routine_id)}).textContent.includes('Resultado de tu intento')")
                    assert run_count(database, risky.routine_id) == 2
                    checks.append('uncertain latest run gates a new execute behind an explicit confirmation')

                    # 409 from a concurrent change keeps the draft and focuses the explanation.
                    await press(f'[data-key="{main_id}:edit"]')
                    await page.wait_for("document.querySelector('#routine-dialog').open")
                    await fill('name', 'Mi borrador')
                    current = await scheduler.get_routine(main_id)
                    await scheduler.update_routine(main_id, parse_routine_spec({**definition('Cambio de otra sesión')}), expected_revision=current.revision, actor='synthetic-other')
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for("document.activeElement?.id==='routine-errors' && document.querySelector('#routine-errors').textContent.includes('cambió en otra sesión')")
                    assert await page.evaluate("document.querySelector('#routine-dialog [data-key=name]').value==='Mi borrador' && !!document.querySelector('#routine-dialog [data-key=reload]')")
                    await click('#routine-dialog [data-key="reload"]')
                    await page.wait_for("document.querySelector('#routine-dialog').open && document.querySelector('#routine-dialog [data-key=name]').value==='Cambio de otra sesión'")
                    checks.append('409 keeps the draft, focuses the reason and offers discard-and-reload')

                    # A list refresh never steals focus: editor field and card button.
                    await page.evaluate("(()=>{const e=document.querySelector('#routine-dialog [data-key=name]');e.focus();e.value='Escribiendo';e.dispatchEvent(new Event('input',{bubbles:true}));e.setSelectionRange(3,3);})()")
                    await page.evaluate("document.querySelector('#routines-refresh').click()")
                    await asyncio.sleep(0.4)
                    assert await page.evaluate("document.activeElement?.dataset.key==='name' && document.activeElement.value==='Escribiendo' && document.activeElement.selectionStart===3")
                    await click('#routine-dialog [data-key="close"]')
                    await page.wait_for("!document.querySelector('#routine-dialog').open")
                    await page.evaluate(f"document.querySelector('[data-key=\"{main_id}:toggle\"]').focus()")
                    current = await scheduler.get_routine(main_id)
                    await scheduler.update_routine(main_id, parse_routine_spec(definition('Renombrada en el servidor')), expected_revision=current.revision, actor='synthetic-other')
                    await page.evaluate("document.querySelector('#routines-refresh').click()")
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Renombrada en el servidor')")
                    assert await page.evaluate(f"document.activeElement?.dataset.key==='{main_id}:toggle'"), 'focus survives a rebuilt card'
                    checks.append('forced refresh keeps focus in the editor field (with caret) and on a rebuilt card button')

                    # N1: 'Actualizar' from the keyboard keeps focus while the list is read and after it.
                    await rule('GET', '/api/routines', 'hold')
                    await page.evaluate("window.__qaHeld=null;document.querySelector('#routines-refresh').focus()")
                    await press_enter(page)
                    await page.wait_for("window.__qaHeld && document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='true'")
                    assert await page.evaluate("document.activeElement?.id==='routines-refresh'"), 'focus kept while loading'
                    await page.evaluate('window.__qaRelease()')
                    await page.wait_for("document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='false'")
                    await asyncio.sleep(0.2)
                    assert await page.evaluate("document.activeElement?.id==='routines-refresh'"), 'focus kept after the refresh'
                    checks.append('N1: keyboard Enter on Actualizar keeps focus on the button during and after the read')

                    audits['rutinas'] = await audit_view(page, evidence, 'rutinas', (
                        ('#app-nav [data-view=routines]', None), ('#rail-nav [data-view=access]', None), ('#routines-create', None),
                        ('#routines-refresh', None), (f'[data-key="{main_id}:execute"]', None), (f'[data-key="{main_id}:delete"]', None)), screenshot, widths=WIDTHS)
                    checks.append('operator rail with Rutinas: last view reachable and hit-testable at 320 px')
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 320, 'height': 760, 'deviceScaleFactor': 2, 'mobile': True})
                    await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    await asyncio.sleep(0.2)
                    rail = await page.evaluate("""(()=>{const r=document.querySelector('#rail-nav'),b=[...r.querySelectorAll('[data-view]')];
                      const active=r.querySelector('.active').getBoundingClientRect(),box=r.getBoundingClientRect();
                      return {cue:r.classList.contains('more-start')||r.classList.contains('more-end'),mask:getComputedStyle(r).maskImage!=='none',
                              sizes:b.map(x=>[Math.round(x.getBoundingClientRect().width),Math.round(x.getBoundingClientRect().height)]),
                              activeVisible:active.left>=box.left-1&&active.right<=box.right+1}})()""")
                    assert rail['cue'] and rail['mask'] and rail['activeVisible'], rail
                    assert all(width >= 44 and height >= 44 for width, height in rail['sizes']), rail
                    # Start and scrolled-middle states: the end fade lies over a partly visible next item.
                    edge = """(()=>{const r=document.querySelector('#rail-nav'),box=r.getBoundingClientRect();
                      const cut=[...r.querySelectorAll('[data-view]')].map(x=>x.getBoundingClientRect()).find(x=>x.left<box.right-1&&x.right>box.right+1);
                      return {start:r.classList.contains('more-start'),end:r.classList.contains('more-end'),left:r.scrollLeft,
                              peek:cut?Math.round(box.right-cut.left):null,cutWidth:cut?Math.round(cut.width):null}})()"""
                    await page.evaluate("document.querySelector('#rail-nav').scrollLeft=0")
                    await asyncio.sleep(0.2)
                    start_state = await page.evaluate(edge)
                    assert start_state['end'] and not start_state['start'], start_state
                    assert start_state['peek'] is not None and 16 <= start_state['peek'] <= start_state['cutWidth'] - 16, start_state
                    await screenshot(page, evidence / 'rail-320-start.png', full=False)
                    await page.evaluate("(()=>{const r=document.querySelector('#rail-nav');r.scrollLeft=(r.scrollWidth-r.clientWidth)/2})()")
                    await asyncio.sleep(0.2)
                    middle_state = await page.evaluate(edge)
                    assert middle_state['start'] and middle_state['end'] and middle_state['left'] > 0, middle_state
                    await screenshot(page, evidence / 'rail-320-middle.png', full=False)
                    assert (evidence / 'rail-320-start.png').read_bytes() != (evidence / 'rail-320-middle.png').read_bytes(), 'rail screenshots differ'
                    # Keyboard focus on a rail item: the ring is drawn inside the button and the item is scrolled clear of the fades.
                    await page.evaluate("document.querySelector('#rail-nav').scrollLeft=0")
                    await page.evaluate("document.querySelectorAll('#rail-nav [data-view]')[3].focus()")
                    await _press_tab(page)
                    await asyncio.sleep(0.3)
                    ring = await page.evaluate("""(()=>{const r=document.querySelector('#rail-nav'),e=document.activeElement,box=r.getBoundingClientRect(),own=e.getBoundingClientRect(),style=getComputedStyle(e);
                      return {rail:r.contains(e),visible:e.matches(':focus-visible'),offset:style.outlineOffset,width:style.outlineWidth,
                              inside:own.left>=box.left+20&&own.right<=box.right-20&&own.top>=box.top&&own.bottom<=box.bottom}})()""")
                    assert ring == {'rail': True, 'visible': True, 'offset': '-3px', 'width': '2px', 'inside': True}, ring
                    await screenshot(page, evidence / 'rail-320-focus.png', full=False)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    checks.append('320 px rail: start and scrolled-middle states differ, the end fade covers a partly visible item, keyboard focus ring drawn inside and clear of the fades')

                    # 503 keeps the last data and says so.
                    saved_scheduler, dashboard.scheduler = dashboard.scheduler, None
                    await click('#routines-refresh')
                    await page.wait_for("document.querySelector('#routines-freshness').textContent.includes('Rutinas no disponibles') && document.querySelector('#routines-freshness').textContent.includes('últimos datos')")
                    assert await page.evaluate('document.querySelectorAll("#routines-list .routine-card").length') == 2
                    dashboard.scheduler = saved_scheduler
                    checks.append('503 shows unavailability and keeps the last observation')

                    # Logout wipes pending identities; a household session sees routines read-only.
                    assert await page.evaluate("Object.keys(sessionStorage).some(k=>k.startsWith('kza:routine-attempt:'))")
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    assert await page.evaluate("!Object.keys(sessionStorage).some(k=>k.startsWith('kza:routine-attempt:'))")
                    checks.append('logout wipes routine attempt keys')
                    before = [(item.routine_id, item.revision) for item in await scheduler.get_all_routines()]
                    runs_before = run_count(database)
                    await login(page, household.token, 'Acceso de hogar')
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,actions,routines'")
                    await click('#app-nav [data-view=routines]')
                    await page.wait_for("document.querySelectorAll('#routines-list .routine-card').length===2")
                    assert await page.evaluate("document.querySelectorAll('[data-routine-write]').length===0 && document.querySelector('#routines-create').hidden && !document.querySelector('#routine-dialog').open")
                    assert await page.evaluate("document.querySelector('.routines-permission').textContent.includes('requiere un acceso de administración')")
                    denied = await page.evaluate(f"""Promise.all([
                      fetch('/api/routines',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({json.dumps(definition('Intento del hogar'))})}}).then(r=>r.status),
                      fetch('/api/routines/{main_id}/execute',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{request_id:'household'}})}}).then(r=>r.status),
                      fetch('/api/routines/{main_id}/disable',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{expected_revision:1}})}}).then(r=>r.status)])""")
                    assert denied == [403, 403, 403], denied
                    assert [(item.routine_id, item.revision) for item in await scheduler.get_all_routines()] == before and run_count(database) == runs_before
                    audits['rutinas-hogar'] = await audit_view(page, evidence, 'rutinas-hogar', (
                        ('#app-nav [data-view=routines]', None), ('#routines-refresh', None)), screenshot, widths=WIDTHS)
                    checks.append('household: read-only view, zero write controls, server 403 for create/execute/disable, store unchanged')
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    await login(page, evaluator.token, 'Evaluación temporal')
                    await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                    assert await page.evaluate("!document.querySelector('[data-view=routines]')")
                    await page.evaluate("location.hash='#rutinas'")
                    await asyncio.sleep(0.3)
                    assert await page.evaluate("document.querySelector('#routines-view').hidden && !document.querySelector('#routines-view').children.length")
                    assert await page.evaluate("fetch('/api/routines').then(r=>r.status)") == 403
                    checks.append('evaluator: no Rutinas entry, #rutinas falls back, server 403')
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    assert page.errors == [], page.errors

                    result = {'passed': True, 'synthetic_fixture': True, 'actual_chromium_http_login': True, 'real_routine_scheduler': True,
                              'roles': ['operator', 'home', 'evaluator'], 'widths': list(WIDTHS), 'checks': checks,
                              'runs_on_server': run_count(database), 'home_effects': len(ha.sends),
                              'a11y_evidence': [f'a11y-{view}.json' for view in audits],
                              'a11y_findings': {view: len(audit['findings']) for view, audit in audits.items()},
                              'requests': len(requests), 'javascript_exceptions': len(page.errors)}
                    (evidence / 'browser-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
                    print(json.dumps(result, ensure_ascii=False))
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, 10)
            await dashboard.stop()
            await serving
            await authority.aclose()
            await scheduler.store.close()


if __name__ == '__main__':
    asyncio.run(main())
