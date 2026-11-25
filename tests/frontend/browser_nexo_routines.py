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
from dataclasses import replace
from pathlib import Path
import socket
from uuid import uuid4

import aiohttp
import uvicorn
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from src.memory.fact_ledger import FactLedger
from src.rooms.room_context import RoomConfig
from src.routines.factory import build_routine_scheduler
from src.routines.models import RunStatus, parse_routine_spec
from src.routines.proposals import ProposalOperation, RoutineProposalTray
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_support import (
    FixtureRows, choose_option,
    assert_frame_consistent,
    VIEWPORT_HEIGHTS, WCAG_SCRIPT, Page, _press_tab, audit_view, browser_binary, press_escape,
    browser_profile_root, evidence_directory, install_login_session, press_enter,
)
from tests.integration.test_typed_home_operations import typed_graph


EVIDENCE = evidence_directory('KZA_NEXO_ROUTINES_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/u2-routines-browser-runs')
WIDTHS = (320, 390, 768, 1024, 1280, 1440)
CAPTURE_WIDTHS = (1440, 768, 390)
ZOOM_WIDTHS = (320, 390)
TZ = 'America/Montevideo'
PROPOSALS = '/api/routines/proposals'
INSTRUMENT = r"""(() => {
  const fetcher = window.fetch.bind(window);
  window.__qaRules = [];
  window.__qaProposalRequests = [];
  window.fetch = async (input, options = {}) => {
    const path = String(input), method = (options.method || 'GET').toUpperCase();
    if (path.startsWith('/api/routines/proposals')) window.__qaProposalRequests.push({ path, method });
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


def field_key(value: str) -> str:
    return json.dumps(f'[data-key="{value}"]')


async def capture_routines(page: Page, evidence: Path, state: str) -> list[dict]:
    """Capture a stable Rutinas state and measure it with 200% text."""
    theme = "document.querySelector('#theme-slot select')"
    previous = await page.evaluate(f'{theme}.value')

    async def choose_theme(value: str) -> None:
        await page.evaluate(f"""(() => {{
            const select = {theme};
            select.value = {json.dumps(value)};
            select.dispatchEvent(new Event('change', {{bubbles: true}}));
        }})()""")
        await page.evaluate("""(async () => {
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            await Promise.allSettled(document.getAnimations()
                .filter(animation => animation instanceof CSSTransition)
                .map(animation => animation.finished));
        })()""")

    zoom_checks = []
    try:
        for value in ('claro', 'oscuro'):
            await choose_theme(value)
            assert await page.evaluate('document.documentElement.dataset.theme') == value
            for width in CAPTURE_WIDTHS:
                await page.command('Emulation.setDeviceMetricsOverride', {
                    'width': width, 'height': VIEWPORT_HEIGHTS[width],
                    'deviceScaleFactor': 1, 'mobile': width <= 390,
                })
                await screenshot(page, evidence / f'rutinas-{state}-{value}-{width}.png',
                                 full=state in ('con-datos', 'no-disponible', 'propuestas'))
                if state == 'con-datos':
                    await assert_frame_consistent(page, '#routines-view', value)
                if width == 768:
                    await assert_routines_contrast(page, state, value)
            for width in ZOOM_WIDTHS:
                await page.command('Emulation.setDeviceMetricsOverride', {
                    'width': width, 'height': VIEWPORT_HEIGHTS[width],
                    'deviceScaleFactor': 1, 'mobile': True,
                })
                await page.evaluate(WCAG_SCRIPT)
                await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                shot = f'rutinas-{state}-{value}-{width}-text200.png'
                try:
                    zoom = await page.evaluate('window.__kzaWcag.beginTextZoom()')
                    await screenshot(page, evidence / shot, full=zoom['dialog'] is None)
                finally:
                    await page.evaluate('window.__kzaWcag.endTextZoom()')
                zoom_checks.append({'state': state, 'theme': value, 'width': width, 'screenshot': shot, **zoom})
    finally:
        await choose_theme(previous)
        await page.command('Emulation.setDeviceMetricsOverride', {
            'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False,
        })
    return zoom_checks


async def assert_routines_contrast(page: Page, state: str, theme: str) -> None:
    """Require AA for Rutinas and its dialogs using the shared luminance audit."""
    await page.evaluate(WCAG_SCRIPT)
    contrast = await page.evaluate("""(() => {
        const modal = document.querySelector('#routine-dialog[open], #routine-confirm-dialog[open]');
        const scope = modal || document.querySelector('#routines-view');
        const findings = window.__kzaWcag.scan([], 24, 44, []).filter(item => item.issue === 'low_contrast'
            && item.element && scope.contains(document.querySelector(item.element)));
        const probe = document.createElement('span');
        probe.style.color = 'var(--nexo-ground)';
        scope.append(probe);
        const shellMatches = getComputedStyle(probe).color === getComputedStyle(document.body).backgroundColor;
        probe.remove();
        return {findings, shellMatches};
    })()""")
    assert contrast['shellMatches'], f'Rutinas {state}, {theme}: expected the NEXO shell palette'
    assert contrast['findings'] == [], f'Rutinas {state}, {theme}: {contrast["findings"]}'


async def assert_no_device_ids(page: Page, selector: str) -> None:
    """Keep device IDs out of visible text and accessible labels, including options."""
    matches = await page.evaluate(r"""(selector => {
        const root = document.querySelector(selector);
        const labels = ['aria-label', 'aria-description', 'aria-valuetext', 'title', 'alt', 'label'];
        return [...root.querySelectorAll('*')].flatMap(el => [
            ...[...el.childNodes].filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent),
            ...labels.map(label => el.getAttribute(label) || ''),
        ]).flatMap(text => text.match(/\b[a-z][a-z0-9_]*\.[a-z0-9_]+\b/g) || []);
    })(""" + json.dumps(selector) + ')')
    assert matches == [], f'Device identifiers in Rutinas labels or text: {matches}'


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
                diagnostic = await self.evaluate("({text:document.body?.innerText?.slice(0,4000)||'',dialog:document.querySelector('dialog[open]')?.innerText||null})")
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
        operator_b = await authority.issue_credential(root, subject='synthetic-operator', role=DashboardRole.OPERATOR, expires_at=expires)
        household = await authority.issue_credential(root, subject='synthetic-home', role=DashboardRole.HOME, expires_at=expires)
        selection = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
        evaluator = await authority.issue_credential(root, subject='synthetic-evaluator', role=DashboardRole.EVALUATOR, expires_at=expires, samples=(selection,))
        rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp', default_cover='cover.test_blind'),
                 'kitchen': RoomConfig('kitchen', 'Cocina', 'la cocina', default_climate='climate.test_ac')}
        ledger = FactLedger(str(directory / 'facts.db'))
        ledger.initialize()
        tray = RoutineProposalTray(ledger)
        dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms,
                                 routine_scheduler=scheduler, routine_proposals=tray)

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
                        if field.endswith(':entity'):
                            await choose_option(page, '#routine-dialog ' + json.loads(field_key(field)), value)
                            return
                        await page.evaluate(f"(()=>{{const e=document.querySelector('#routine-dialog '+{field_key(field)});e.value={json.dumps(value)};"
                                            "e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()")

                    async def rule(method: str, path: str, mode: str) -> None:
                        await page.evaluate(f'window.__qaRules.push({json.dumps({"method": method, "path": path, "mode": mode})})')

                    routine_names: dict[str, tuple[str, ...]] = {}
                    proposal_rows = FixtureRows('#routine-proposals .routine-proposal', '.row-title')

                    def card_selector(identity: str, inner: str = '') -> str:
                        if identity not in routine_names:
                            return proposal_rows.selector(identity, inner)
                        labels = ','.join(f'[aria-label={json.dumps("Rutina " + name, ensure_ascii=False)}]'
                                          for name in routine_names[identity])
                        return f'#routines-list .routine-card:is({labels}) {inner}'.strip()

                    def card(identity: str, inner: str = '') -> str:
                        return json.dumps(card_selector(identity, inner))

                    def control(identity: str, action: str) -> str:
                        return card_selector(identity, f'[data-key="{action}"]')

                    def key(value: str) -> str:
                        return json.dumps(control(*value.rsplit(':', 1)))

                    async def ids() -> list[str]:
                        return [item.routine_id for item in await scheduler.get_all_routines()]

                    # Operator: navigation, empty state, failed save focus, preview, create.
                    await login(page, operator.token, 'Administración')
                    assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,routines,tasks,activity,people,memory,access'")
                    assert await page.evaluate("document.querySelectorAll('a[href*=\"legacy.html\"]').length===0")
                    await rule('GET', '/api/routines', 'hold')
                    await click('#app-nav [data-view=routines]')
                    await page.wait_for("window.__qaHeld === '/api/routines' && document.querySelector('#routines-freshness')?.textContent.includes('Consultando rutinas')")
                    zoom_checks = await capture_routines(page, evidence, 'cargando')
                    await page.evaluate('window.__qaRelease()')
                    await page.wait_for("location.hash==='#rutinas' && !document.querySelector('#routines-view').hidden && !document.querySelector('.routines-empty').hidden")
                    assert await page.evaluate("document.querySelector('.routines-empty').textContent.includes('Todavía no hay rutinas')")
                    zoom_checks += await capture_routines(page, evidence, 'vacio')
                    checks.append('nav entry and #rutinas hash; empty state from a real empty store')
                    await press('#routines-create')
                    await page.wait_for("document.querySelector('#routine-dialog').open && document.activeElement?.dataset.key==='name'")
                    await press_escape(page)
                    await page.wait_for("!document.querySelector('#routine-dialog').open && document.activeElement.id==='routines-create'")
                    await press('#routines-create')
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
                    await fill('condition:0:entity', 'Persiana')
                    await fill('condition:0:state', 'closed')
                    assert await page.evaluate("""(() => {
                        const field = document.querySelector('[data-key="condition:0:state"]');
                        const help = document.getElementById(field.getAttribute('aria-describedby'));
                        return field.type === 'text'
                            && help?.textContent === 'Cerrado = closed · Abierto = open · Encendido = on · Apagado = off';
                    })()""")
                    await click('#routine-dialog [data-key="action:add"]')
                    await fill('action:0:entity', 'Luz principal')
                    await fill('action:0:operation', 'turn_on')
                    await click('#routine-dialog [data-key="action:add"]')
                    await fill('action:1:entity', 'Persiana')
                    await fill('action:1:operation', 'open_cover')
                    await click('#routine-dialog [data-key="preview"]')
                    await page.wait_for("document.querySelector('#routine-dialog .routine-preview')?.textContent.includes('Vista previa · sin guardar ni ejecutar')")
                    assert await page.evaluate("document.querySelector('#routine-dialog .routine-preview').textContent.includes('Lu, Mi · 22:30')")
                    assert await ids() == [], 'preview writes nothing'
                    await assert_no_device_ids(page, '#routine-dialog')
                    checks.append('preview normalizes without writing')
                    zoom_checks += await capture_routines(page, evidence, 'editor')
                    audits = {}
                    audits['rutinas-editor'] = await audit_view(page, evidence, 'rutinas-editor', (
                        ('#routine-dialog [data-key=name]', None), ('#routine-dialog [data-key="day:6"]', None), ('#routine-dialog [data-key="action:1:up"]', None),
                        ('#routine-dialog [data-key=preview]', None), ('#routine-dialog [data-key=save]', None)), screenshot, widths=WIDTHS, text_zoom=True)
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for("!document.querySelector('#routine-dialog').open && document.querySelectorAll('#routines-list .routine-card').length===1")
                    [main_id] = await ids()
                    routine_names[main_id] = ('Noche tranquila', 'Noche', 'Noche larga', 'Cambio de otra sesión',
                                              'Renombrada en el servidor', 'Rutina revisada por voz')
                    saved = await scheduler.get_routine(main_id)
                    assert saved is not None and saved.spec.conditions[0].state == 'closed'
                    assert await page.evaluate(f"document.querySelector({card(main_id)}).textContent.includes('Versión 1') && document.querySelector({card(main_id)}).textContent.includes('Lu, Mi · 22:30 · Montevideo')")
                    assert await page.evaluate("document.querySelector('#routines-live').textContent.includes('Versión 1')")
                    checks.append('create through the dialog persists revision 1; live region announces it')
                    checks.append('condition help translates known states; save preserves the exact Home Assistant value')

                    # Edit (remove the condition so runs are not skipped), disable/enable.
                    await press(control(main_id, 'edit'))
                    await page.wait_for("document.querySelector('#routine-dialog').open")
                    await click('#routine-dialog [data-key="condition:0:remove"]')
                    await fill('name', 'Noche')
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for(f"!document.querySelector('#routine-dialog').open && document.querySelector({card(main_id)}).textContent.includes('Versión 2')")
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(main_id + ':edit')})"), 'focus returns to the invoking control'
                    await press(control(main_id, 'toggle'))
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Desactivada') && document.querySelector({card(main_id)}).textContent.includes('Versión 3')")
                    assert await page.evaluate(f"document.querySelector({key(main_id + ':toggle')}).getAttribute('aria-checked')==='false'")
                    await press(control(main_id, 'toggle'))
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Versión 4')")
                    checks.append('edit, disable and enable through CAS revisions')
                    assert await page.evaluate(f"document.querySelector({key(main_id + ':toggle')}).getAttribute('aria-checked')==='true'")
                    tree = await page.command('Accessibility.getFullAXTree')
                    assert any(node.get('role', {}).get('value') == 'switch'
                               and node.get('name', {}).get('value') == 'Activa o desactivada: Noche'
                               for node in tree['nodes'])
                    checks.append('the switch exposes the saved state and a stable accessible name')

                    # Execute, then a lost execute response survives a reload and is consulted, not resent.
                    sends = len(ha.sends)
                    await press(control(main_id, 'execute'))
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Resultado de tu intento') && document.querySelector({card(main_id, '[data-run-status=succeeded]')})")
                    assert run_count(database, main_id) == 1 and len(ha.sends) == sends + 2
                    await rule('POST', f'/api/routines/{main_id}/execute', 'drop')
                    await press(control(main_id, 'execute'))
                    await page.wait_for(f"document.querySelector({card(main_id, '.routine-notice')})?.textContent.startsWith('Resultado pendiente de consulta. No se pudo contactar')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['state'] == 'unresolved' and stored['request_id']
                    assert run_count(database, main_id) == 2, ('the server committed the lost send', run_count(database, main_id), requests[-6:])
                    await page.command('Page.reload')
                    await page.wait_for(f"document.querySelector('#routines-view')?.hidden === false && document.querySelector({card(main_id)})?.textContent.includes('Resultado pendiente de consulta')")
                    assert await page.evaluate(f"!document.querySelector({key(main_id + ':execute')})"), 'no fresh one-click execute while unresolved'
                    await press(control(main_id, 'consult'))
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Resultado de tu intento') && !document.querySelector({card(main_id)}).textContent.includes('Resultado pendiente')")
                    assert run_count(database, main_id) == 2, 'consult never resends'
                    latest = await scheduler.get_latest_run(main_id)
                    assert await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}')).run_id") == latest.run_id
                    checks.append('lost execute response: reload keeps the request identity, consult finds the same run, one run on the server')

                    # Leaving the view during an execute never aborts it.
                    await rule('POST', f'/api/routines/{main_id}/execute', 'hold')
                    await press(control(main_id, 'execute'))
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
                    await press(control(main_id, 'execute'))
                    await page.wait_for(f"document.querySelector({card(main_id, '.routine-notice')})?.textContent.startsWith('Resultado pendiente de consulta')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['actor'] == operator.principal.routine_actor and stored['state'] == 'unresolved'
                    runs_before_switch = run_count(database, main_id)
                    await install_login_session(page, f'http://127.0.0.1:{port}', operator_b.token)
                    marker = len(requests)
                    await page.command('Page.reload')
                    await page.wait_for(f"document.querySelector('#routines-view')?.hidden === false && document.querySelector({card(main_id)})")
                    await page.wait_for("document.querySelector('.routine-session-notice') && !document.querySelector('.routine-session-notice').hidden")
                    assert await page.evaluate(f"sessionStorage.getItem('kza:routine-attempt:{main_id}')===null")
                    assert await page.evaluate(f"!document.querySelector({card(main_id)}).textContent.includes('Resultado pendiente') && !document.querySelector({key(main_id + ':retry')})")
                    await asyncio.sleep(0.5)
                    assert not any(item['method'] == 'POST' or item['path'].endswith('/attempts') for item in requests[marker:]), 'no consult or resend of the other account identity'
                    assert run_count(database, main_id) == runs_before_switch
                    checks.append("account switch under the same tab: the other account's pending identity is dropped on reload, never consulted or resent; a reminder is shown")

                    # R2b: the tab is only hidden (no reload) while another tab switches the shared cookie back to A.
                    await rule('POST', f'/api/routines/{main_id}/execute', 'drop')
                    await press(control(main_id, 'execute'))
                    await page.wait_for(f"document.querySelector({card(main_id, '.routine-notice')})?.textContent.startsWith('Resultado pendiente de consulta')")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['actor'] == operator_b.principal.routine_actor and stored['state'] == 'unresolved'
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
                    assert await page.evaluate(f"document.querySelector('.routine-session-notice').textContent.includes('Revisá el último resultado') && !document.querySelector({key(main_id + ':retry')})")
                    await page.command('Page.reload')
                    await page.wait_for(f"document.querySelector('#routines-view')?.hidden === false && document.querySelector({card(main_id)})")
                    checks.append('hidden tab (no reload) while the cookie switches account: on return whoami is asked first, nothing is retried or consulted, the identity is dropped, the reminder shows and the session is re-read')

                    # Lost PUT: resolved by comparison, not by a second write.
                    await press(control(main_id, 'edit'))
                    await page.wait_for("document.querySelector('#routine-dialog').open")
                    await fill('name', 'Noche larga')
                    await rule('PUT', f'/api/routines/{main_id}', 'drop')
                    await click('#routine-dialog [data-key="save"]')
                    await page.wait_for(f"!document.querySelector('#routine-dialog').open && document.querySelector({card(main_id)}).textContent.includes('Versión 5') && document.querySelector({card(main_id)}).textContent.includes('El cambio se aplicó.')")
                    assert sum(1 for item in requests if item['method'] == 'PUT') == 2
                    checks.append('lost PUT resolves as applied after a list read; no second PUT')

                    # Lost DELETE on a second routine.
                    doomed = await scheduler.register_routine(parse_routine_spec(definition('Para borrar')), actor='synthetic-seed')
                    routine_names[doomed.routine_id] = (doomed.spec.name,)
                    await click('#routines-refresh')
                    await page.wait_for(f"document.querySelector({card(doomed.routine_id)})")
                    await rule('DELETE', f'/api/routines/{doomed.routine_id}', 'drop')
                    await press(control(doomed.routine_id, 'delete'))
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open && document.activeElement?.dataset.key==='confirm-cancel'")
                    await click('#routine-confirm-dialog [data-key="confirm-accept"]')
                    await page.wait_for(f"!document.querySelector({card(doomed.routine_id)}) && document.querySelector('#routines-live').textContent==='La rutina se borró.'")
                    assert doomed.routine_id not in await ids()
                    checks.append('delete asks in a dialog; lost DELETE resolves as applied')

                    # Uncertain latest run (scheduled, no local attempt) requires confirmation before repeating.
                    risky = await scheduler.register_routine(parse_routine_spec(definition('Incierta')), actor='synthetic-seed')
                    routine_names[risky.routine_id] = (risky.spec.name, 'Actualizada durante la edición')
                    claim = await scheduler.store.claim_run(risky.routine_id, json.dumps(['seed']), actor='synthetic-schedule', source='schedule')
                    await scheduler.store.mark_action_sending(claim.run.run_id, 0)
                    await scheduler.store.finish_run(claim.run.run_id, RunStatus.UNCERTAIN)
                    await click('#routines-refresh')
                    await page.wait_for(f"document.querySelector({card(risky.routine_id, '[data-run-status=uncertain]')})")
                    await press(control(risky.routine_id, 'execute'))
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open && document.querySelector('#routine-confirm-dialog').textContent.includes('Luz principal · Incierta')")
                    await assert_no_device_ids(page, '#routine-confirm-dialog')
                    zoom_checks += await capture_routines(page, evidence, 'confirmacion')
                    await screenshot(page, evidence / 'uncertain-confirm.png', full=False)
                    audits['rutinas-confirmacion'] = await audit_view(page, evidence, 'rutinas-confirmacion', (
                        ('#routine-confirm-dialog [data-key=confirm-cancel]', None), ('#routine-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                    await press_escape(page)
                    await page.wait_for(f"!document.querySelector('#routine-confirm-dialog').open && "
                                        f"document.activeElement === document.querySelector({json.dumps(control(risky.routine_id, 'execute'))})")
                    assert run_count(database, risky.routine_id) == 1, 'cancel sends nothing'
                    await press(control(risky.routine_id, 'execute'))
                    await page.wait_for("document.querySelector('#routine-confirm-dialog').open")
                    await click('#routine-confirm-dialog [data-key="confirm-accept"]')
                    await page.wait_for(f"document.querySelector({card(risky.routine_id)}).textContent.includes('Resultado de tu intento')")
                    assert run_count(database, risky.routine_id) == 2
                    checks.append('uncertain latest run gates a new execute behind an explicit confirmation')

                    # 409 from a concurrent change keeps the draft and focuses the explanation.
                    await press(control(main_id, 'edit'))
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
                    # WEB-11: complete a held refresh with a non-collapsed selection in the field reached by Tab.
                    await page.wait_for("document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='false'")
                    await fill('description', 'Descripción sintética')
                    card_title = f"document.querySelector({card(risky.routine_id, 'h3')})?.textContent"
                    renamed = await scheduler.update_routine(
                        risky.routine_id, replace(risky.spec, name='Actualizada durante la edición'),
                        expected_revision=risky.revision, actor='synthetic-other')
                    await rule('GET', '/api/routines', 'hold')
                    await page.evaluate('window.__qaHeld=null')
                    await click('#routines-refresh')
                    await page.wait_for("window.__qaHeld==='/api/routines'")
                    await page.evaluate("document.querySelector('#routine-dialog [data-key=name]').focus()")
                    await _press_tab(page)
                    await page.evaluate("document.querySelector('#routine-dialog [data-key=description]').setSelectionRange(2, 7)")
                    editor_focus = """(e => ({tag: e.tagName, id: e.id || null, key: e.dataset.key || null, dialog: e.closest('dialog')?.id,
                        selectionStart: e.selectionStart, selectionEnd: e.selectionEnd}))(document.activeElement)"""
                    focus_before = await page.evaluate(editor_focus)
                    assert focus_before == {'tag': 'INPUT', 'id': None, 'key': 'description', 'dialog': 'routine-dialog',
                                            'selectionStart': 2, 'selectionEnd': 7}, focus_before
                    title_before = await page.evaluate(card_title)
                    assert title_before == risky.spec.name and title_before != renamed.spec.name, title_before
                    await page.evaluate('window.__qaRelease()')
                    await page.wait_for("document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='false'")
                    await page.wait_for(f"{card_title} === {json.dumps(renamed.spec.name)}")
                    focus_after = await page.evaluate(editor_focus)
                    assert focus_after == focus_before, (focus_before, focus_after)
                    web11 = {'before': focus_before, 'after': focus_after, 'refreshed_card': risky.routine_id,
                             'title_before': title_before, 'title_after': renamed.spec.name}
                    checks.append('WEB-11: the editor field reached by real Tab keeps both selection endpoints after refresh')

                    await click('#routine-dialog [data-key="close"]')
                    await page.wait_for("!document.querySelector('#routine-dialog').open")
                    await page.evaluate(f"document.querySelector({key(main_id + ':toggle')}).focus()")
                    current = await scheduler.get_routine(main_id)
                    await scheduler.update_routine(main_id, parse_routine_spec(definition('Renombrada en el servidor')), expected_revision=current.revision, actor='synthetic-other')
                    await page.evaluate("document.querySelector('#routines-refresh').click()")
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Renombrada en el servidor')")
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(main_id + ':toggle')})"), 'focus survives a rebuilt card'
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

                    # Wait for the risky run to settle before snapshotting the view's Tab sequence.
                    risky_execute = control(risky.routine_id, 'execute')
                    await page.wait_for(f'(e => e && !e.disabled)(document.querySelector({json.dumps(risky_execute)}))')
                    zoom_checks += await capture_routines(page, evidence, 'con-datos')
                    audits['rutinas'] = await audit_view(page, evidence, 'rutinas', (
                        ('#app-nav [data-view=routines]', None), ('#rail-nav [data-view=access]', None), ('#routines-create', None),
                        ('#routines-refresh', None), (control(main_id, 'execute'), None),
                        (control(main_id, 'consult'), None), (control(main_id, 'edit'), None),
                        (control(main_id, 'toggle'), None), (control(main_id, 'delete'), None),
                        (risky_execute, None)), screenshot, widths=WIDTHS, text_zoom=True)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 320, 'height': 760, 'deviceScaleFactor': 2, 'mobile': True})
                    await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    nav_bounds = await page.evaluate("""(() => {
                        const nav = document.querySelector('#rail-nav');
                        const bounds = nav.getBoundingClientRect();
                        return {
                            pageFits: document.documentElement.scrollWidth <= innerWidth,
                            links: [...nav.querySelectorAll('[data-section]')].map(link => {
                                const box = link.getBoundingClientRect();
                                return {width: box.width, height: box.height,
                                        visible: box.left >= bounds.left && box.right <= bounds.right
                                            && box.top >= 0 && box.bottom <= innerHeight,
                                        hit: document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2) === link};
                            }),
                        };
                    })()""")
                    assert nav_bounds['pageFits'] and len(nav_bounds['links']) == 7, nav_bounds
                    assert all(item['width'] >= 44 and item['height'] >= 44 and item['visible'] and item['hit']
                               for item in nav_bounds['links']), nav_bounds
                    await screenshot(page, evidence / 'navigation-320.png', full=False)
                    await page.evaluate("document.querySelector('#rail-nav [data-section=memory]').focus()")
                    await _press_tab(page)
                    ring = await page.evaluate("""(() => {
                        const link = document.activeElement, style = getComputedStyle(link);
                        return {section: link.dataset.section, visible: link.matches(':focus-visible'),
                                offset: style.outlineOffset, width: style.outlineWidth};
                    })()""")
                    assert ring == {'section': 'system', 'visible': True, 'offset': '-3px', 'width': '3px'}, ring
                    await screenshot(page, evidence / 'navigation-320-focus.png', full=False)
                    await press_enter(page)
                    await page.wait_for("location.hash==='#accesos' && !document.querySelector('#access-view').hidden")
                    assert await page.evaluate("document.activeElement.matches('#rail-nav [data-section=system][aria-current=true]')")
                    await _press_tab(page, backward=True)
                    assert await page.evaluate("document.activeElement.dataset.section==='memory'")
                    await click('#rail-nav [data-section=routines]')
                    await page.wait_for("document.querySelector('#routines-list .routine-card')")
                    # Re-entering Rutinas starts a list read that keeps Actualizar aria-disabled; wait for it to settle.
                    await page.wait_for("document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='false'")
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    checks.append('320 px single navigation: controls fit, 44 px targets, Tab and Shift+Tab follow section order, Enter opens Sistema and retains focus')

                    # 503 keeps the last data and says so.
                    saved_scheduler, dashboard.scheduler = dashboard.scheduler, None
                    # Actualizar ignores clicks while a list read is in flight (aria-disabled), and re-entering Rutinas
                    # can start one more read after the wait above. Click only when enabled and retry until the 503
                    # notice shows; the 20 s wait below still fails closed if it never does.
                    for _ in range(80):
                        if await page.evaluate("document.querySelector('#routines-freshness').textContent.includes('Rutinas no disponibles')"):
                            break
                        if await page.evaluate("document.querySelector('#routines-refresh').getAttribute('aria-disabled')==='false'"):
                            await click('#routines-refresh')
                        await asyncio.sleep(0.25)
                    await page.wait_for("document.querySelector('#routines-freshness').textContent.includes('Rutinas no disponibles') && document.querySelector('#routines-freshness').textContent.includes('últimos datos')")
                    assert await page.evaluate('document.querySelectorAll("#routines-list .routine-card").length') == 2
                    zoom_checks += await capture_routines(page, evidence, 'no-disponible')
                    dashboard.scheduler = saved_scheduler
                    checks.append('503 shows unavailability and keeps the last observation')
                    await click('#routines-refresh')
                    await page.wait_for("!document.querySelector('#routines-freshness').classList.contains('is-error')")

                    # Real voice-tray entries. Review is read-only; Enter, Tab and Enter decide.
                    previous = await scheduler.get_routine(main_id)
                    assert previous is not None
                    runs_before_proposals, effects_before_proposals = run_count(database), len(ha.sends)
                    update = tray.submit(ProposalOperation.UPDATE, conversation_id=str(uuid4()),
                                         routine_id=main_id, expected_revision=previous.revision,
                                         spec=replace(previous.spec, name='Rutina revisada por voz'))
                    rejected_definition = definition('Rutina para rechazar')
                    rejected_definition['conditions'] = [{'entity_id': 'cover.test_blind', 'state': 'open'}]
                    discard = tray.submit(ProposalOperation.CREATE, conversation_id=str(uuid4()),
                                          spec=parse_routine_spec(rejected_definition))
                    await proposal_rows.bind(page, update, 'Editar rutina')
                    await proposal_rows.bind(page, discard, 'Crear rutina')
                    await click('[data-key="proposals-refresh"]')
                    await page.wait_for(f"document.querySelector({key(update + ':review')}) && document.querySelector({key(discard + ':review')})")
                    tree = await page.command('Accessibility.getFullAXTree')
                    assert any(node.get('role', {}).get('value') == 'region' and
                               node.get('name', {}).get('value') == 'Propuestas por voz' for node in tree['nodes'])
                    await page.evaluate(f"document.querySelector({key(update + ':review')}).focus()")
                    await press_enter(page)
                    before_diff = card(update, '[data-proposal-diff="before"]')
                    after_diff = card(update, '[data-proposal-diff="after"]')
                    await page.wait_for(f"document.querySelector({before_diff})?.textContent.includes({json.dumps(previous.spec.name)}) && "
                                        f"document.querySelector({after_diff})?.textContent.includes('Rutina revisada por voz')")
                    assert await scheduler.get_routine(main_id) == previous
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(update + ':review')})")
                    zoom_checks += await capture_routines(page, evidence, 'propuestas')
                    await assert_no_device_ids(page, '#routines-view')
                    tree = await page.command('Accessibility.getFullAXTree')
                    names = {node.get('name', {}).get('value') for node in tree['nodes'] if node.get('role', {}).get('value') == 'button'}
                    assert {'Confirmar propuesta', 'Descartar'} <= names
                    await _press_tab(page)
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(update + ':confirm')})")
                    await rule('POST', f'{PROPOSALS}/{update}/confirm', 'drop')
                    await press_enter(page)
                    await page.wait_for(f"document.querySelector({key(update + ':confirm')})?.textContent==='Reintentar confirmación'")
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(update + ':confirm')})")
                    assert await page.evaluate("document.querySelector('#routine-proposals').textContent.includes('Resultado incierto')")
                    await press_enter(page)
                    await page.wait_for("document.querySelector('#routine-proposals').textContent.includes('Propuesta confirmada')")
                    updated = await scheduler.get_routine(main_id)
                    assert updated is not None and updated.revision == previous.revision + 1 and updated.spec.name == 'Rutina revisada por voz'
                    assert await page.evaluate("document.activeElement?.dataset.key==='proposals-refresh'")
                    assert [item['status'] for item in requests if item['method'] == 'POST' and
                            item['path'] == f'{PROPOSALS}/{update}/confirm'] == [200, 200]

                    await page.evaluate(f"document.querySelector({key(discard + ':review')}).focus()")
                    await press_enter(page)
                    await page.wait_for(f"document.querySelector({key(discard + ':reject')})")
                    rejected_condition = card(discard, '[data-proposal-diff="after"] ul li')
                    assert await page.evaluate(
                        f"document.querySelector({rejected_condition})?.textContent.endsWith(' · Abierto')"
                    ), 'Expected Abierto in the proposed condition'
                    checks.append('proposed condition open is rendered as Abierto in the after panel')
                    await _press_tab(page)
                    await _press_tab(page)
                    assert await page.evaluate(f"document.activeElement===document.querySelector({key(discard + ':reject')})")
                    await press_enter(page)
                    await page.wait_for("document.querySelector('#routine-proposals').textContent.includes('Propuesta rechazada')")
                    assert await page.evaluate("document.activeElement?.dataset.key==='proposals-refresh'")
                    assert [item['status'] for item in requests if item['method'] == 'POST' and
                            item['path'] == f'{PROPOSALS}/{discard}/reject'] == [200]
                    assert len(await scheduler.get_all_routines()) == 2
                    assert run_count(database) == runs_before_proposals and len(ha.sends) == effects_before_proposals
                    checks.append('voice proposals: named region and controls, read-only before/after, keyboard confirm and replay after lost reply, keyboard reject, no run or HA effects')
                    checks.append('device IDs are absent from visible text, options and accessible labels')
                    tray.submit(ProposalOperation.CREATE, conversation_id=str(uuid4()),
                                spec=parse_routine_spec(definition('Propuesta reservada para revisión')))

                    # Logout wipes pending identities. Household can run and consult its own results.
                    assert await page.evaluate("Object.keys(sessionStorage).some(k=>k.startsWith('kza:routine-attempt:'))")
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    assert await page.evaluate("!Object.keys(sessionStorage).some(k=>k.startsWith('kza:routine-attempt:'))")
                    checks.append('logout wipes routine attempt keys')
                    before = [(item.routine_id, item.revision) for item in await scheduler.get_all_routines()]
                    runs_before = run_count(database)
                    operator_run = await scheduler.get_latest_run(main_id)
                    assert operator_run is not None
                    proposal_marker = await page.evaluate('window.__qaProposalRequests.length')
                    request_marker = len(requests)
                    await login(page, household.token, 'Acceso de hogar')
                    assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,routines,tasks,activity'")
                    await click('#app-nav [data-view=routines]')
                    await page.wait_for("document.querySelectorAll('#routines-list .routine-card').length===2")
                    assert await page.evaluate("document.querySelector('[data-routine-write=execute]') && !document.querySelector('[data-routine-write]:not([data-routine-write=execute])') && document.querySelector('#routines-create').hidden && !document.querySelector('#routine-dialog').open")
                    assert await page.evaluate("document.querySelector('.routines-permission').textContent.includes('consultar tus resultados')")
                    assert await page.evaluate(f"!document.querySelector({card(main_id, '.routine-run')})")
                    denied = await page.evaluate(f"""Promise.all([
                      fetch('/api/routines',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({json.dumps(definition('Intento del hogar'))})}}).then(r=>r.status),
                      fetch('/api/routines/{main_id}/disable',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{expected_revision:1}})}}).then(r=>r.status),
                      fetch('/api/routines/{main_id}/runs/{operator_run.run_id}').then(r=>r.status)])""")
                    assert denied == [403, 403, 404], denied
                    await press(control(main_id, 'execute'))
                    await page.wait_for(f"document.querySelector({card(main_id)}).textContent.includes('Resultado de tu intento') && document.querySelector({card(main_id, '[data-run-status=succeeded]')})")
                    stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:routine-attempt:{main_id}'))")
                    assert stored['actor'] == household.principal.routine_actor and stored['state'] == 'terminal'
                    assert stored['run_id'] != operator_run.run_id
                    marker = len(requests)
                    await press(control(main_id, 'consult'))
                    await page.wait_for(f"!document.querySelector({key(main_id + ':consult')}).disabled")
                    assert any(item['method'] == 'GET' and item['path'] == f"/api/routines/{main_id}/runs/{stored['run_id']}"
                               for item in requests[marker:])
                    assert [(item.routine_id, item.revision) for item in await scheduler.get_all_routines()] == before
                    assert run_count(database) == runs_before + 1
                    audits['rutinas-hogar'] = await audit_view(page, evidence, 'rutinas-hogar', (
                        ('#app-nav [data-view=routines]', None), ('#routines-refresh', None),
                        (control(main_id, 'execute'), None), (control(main_id, 'consult'), None)), screenshot, widths=WIDTHS)
                    checks.append('both themes match the NEXO shell and pass scoped visible contrast for every captured state')
                    assert await page.evaluate("!document.querySelector('#routine-proposals')")
                    assert await page.evaluate('window.__qaProposalRequests.length') == proposal_marker
                    assert not any(item['path'].startswith(PROPOSALS) for item in requests[request_marker:])
                    checks.append('household: execute and consult own run, create/disable return 403, operator run returns 404, one new run without definition changes')
                    checks.append('household without routines:manage neither sees nor requests voice proposals')
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    proposal_marker = await page.evaluate('window.__qaProposalRequests.length')
                    request_marker = len(requests)
                    await login(page, evaluator.token, 'Evaluación temporal')
                    await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                    assert await page.evaluate("!document.querySelector('[data-view=routines]')")
                    await page.evaluate("location.hash='#rutinas'")
                    await asyncio.sleep(0.3)
                    assert await page.evaluate("document.querySelector('#routines-view').hidden && !document.querySelector('#routines-view').children.length")
                    assert await page.evaluate("fetch('/api/routines').then(r=>r.status)") == 403
                    assert await page.evaluate("!document.querySelector('#routine-proposals')")
                    assert await page.evaluate('window.__qaProposalRequests.length') == proposal_marker
                    assert not any(item['path'].startswith(PROPOSALS) for item in requests[request_marker:])
                    checks.append('evaluator: no Rutinas entry, #rutinas falls back, server 403')
                    checks.append('evaluator neither sees nor requests voice proposals')
                    await click('#logout')
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    assert page.errors == [], page.errors

                    zoom_evidence = evidence / 'text-zoom-rutinas.json'
                    zoom_evidence.write_text(json.dumps(zoom_checks, indent=2) + '\n')
                    zoom_failures = [
                        f'{item["state"]}/{item["theme"]}/{item["width"]}px '
                        f'(document={item["document"]}, dialog={item["dialog"]})'
                        for item in zoom_checks if item['document'] or item['dialog']
                    ]
                    zoom_failures += [
                        f'{view}/{width}px (+{item["overflow_px"]}px)'
                        for view, audit in audits.items() for width in ZOOM_WIDTHS
                        for item in audit['widths'][str(width)]['wcag_findings']
                        if item['criterion'] == '1.4.4' and item['issue'] == 'increased_horizontal_overflow'
                    ]
                    if not zoom_failures:
                        checks.append('no horizontal overflow at 200% text at 320/390px')
                    result = {'passed': not zoom_failures, 'synthetic_fixture': True, 'actual_chromium_http_login': True, 'real_routine_scheduler': True,
                              'roles': ['operator', 'home', 'evaluator'], 'widths': list(WIDTHS), 'checks': checks, 'web11_focus': web11,
                              'runs_on_server': run_count(database), 'home_effects': len(ha.sends),
                              'a11y_evidence': [f'a11y-{view}.json' for view in audits] + [zoom_evidence.name],
                              'a11y_findings': {view: len(audit['findings']) for view, audit in audits.items()},
                              'text_zoom_failures': zoom_failures,
                              'requests': len(requests), 'javascript_exceptions': len(page.errors)}
                    (evidence / 'browser-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
                    print(json.dumps(result, ensure_ascii=False))
                    assert not zoom_failures, f'Rutinas 200% text horizontal overflow. See {zoom_evidence.name} and a11y reports.'
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, 10)
            await dashboard.stop()
            await serving
            await tray.aclose()
            ledger.close()
            await authority.aclose()
            await scheduler.store.close()


if __name__ == '__main__':
    asyncio.run(main())
