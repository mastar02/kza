"""Real Chromium, real logins and real memory services for the NEXO Memoria view.

U3b-3 adds the operator owner side: walked pagination, owner labels and counts,
the assign dialog (permanence, D5, A16 notice, 409 dependents), reassign,
unassign, 'Devueltos', the prune flag and the account-removal disclosure.

Synthetic credentials stay in process memory. Evidence never serializes a
credential, a cookie, a request body or fact text; only methods, paths,
statuses and counts.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import aiohttp
import uvicorn
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard import memory_api
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from src.memory.fact_ledger import FactDependency, MemoryAssigner, MemoryReviewer
from src.memory.services import build_memory_services
from src.rooms.room_context import RoomConfig
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_support import (
    FixtureRows, choose_option,
    Page, _press_tab, assert_frame_consistent, audit_view, browser_binary, browser_profile_root, press_escape,
    evidence_directory, install_login_session, press_enter,
)
from tests.integration.test_typed_home_operations import typed_graph
from tests.mocks.memory_graph import MemoryClient

# Copy shipped by memory.js (R1).
DEPENDENTS_SCOPE = ('Se eliminan los recuerdos derivados que existan al momento de asignar, que pueden no ser exactamente los de esta '
                    'lista si cambian mientras decidís. Si la lista cambió, el aviso final dice cuáles difieren, con el texto que ves acá; '
                    'los que no figuran en esta vista solo se cuentan.')


EVIDENCE = evidence_directory('KZA_NEXO_MEMORY_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/u3-memory-browser-runs')
WIDTHS = (320, 390, 768, 1024, 1280, 1440)
ZOOM_WIDTHS = (320, 390)
ACTIVE_CONTROL = "({tag: document.activeElement.tagName, key: document.activeElement.dataset.key || null})"
INSTRUMENT = r"""(() => {
  const fetcher = window.fetch.bind(window);
  window.__qaRules = [];
  window.fetch = async (input, options = {}) => {
    const path = String(input), method = (options.method || 'GET').toUpperCase();
    const index = window.__qaRules.findIndex(rule => rule.method === method && path.startsWith(rule.path));
    const rule = index >= 0 ? window.__qaRules.splice(index, 1)[0] : null;
    // 'lose': the request reaches the server but the client gives up at once (a timeout).
    if (rule?.mode === 'lose') { fetcher(input, options).catch(() => {}); throw new TypeError('Synthetic lost request'); }
    const response = await fetcher(input, options);
    // The server has answered in both modes: 'hold' delays delivery, 'drop' loses it.
    if (rule?.mode === 'hold') await new Promise(resolve => { window.__qaRelease = resolve; window.__qaHeld = path; });
    if (rule?.mode === 'hold-drop') {
      await new Promise(resolve => { window.__qaRelease = resolve; window.__qaHeld = path; });
      await response.text();
      throw new TypeError('Synthetic lost response');
    }
    if (rule?.mode === 'drop') { await response.text(); throw new TypeError('Synthetic lost response'); }
    return response;
  };
})()"""
# Text-only enlargement: every element inside the scope gets its computed font size doubled
# (WCAG 1.4.4 style 200% text), then reflow, clipping and hit-testing are measured.
TEXT_ZOOM = r"""((scopes) => {
  const roots = scopes.map(selector => document.querySelector(selector)).filter(Boolean);
  const elements = roots.flatMap(root => [root, ...root.querySelectorAll('*')]);
  const sizes = elements.map(el => [el, parseFloat(getComputedStyle(el).fontSize), getComputedStyle(el).lineHeight]);
  for (const [el, size, line] of sizes) {
    el.style.fontSize = (size * 2) + 'px';
    if (line.endsWith('px')) el.style.lineHeight = (parseFloat(line) * 2) + 'px';
  }
  return elements.length;
})"""
TEXT_ZOOM_CHECK = r"""((scopes, keys) => {
  const width = window.innerWidth, roots = scopes.map(selector => document.querySelector(selector)).filter(Boolean);
  const shown = el => el.getClientRects().length > 0 && el.checkVisibility({visibilityProperty: true});
  const offenders = [], clipped = [], blocked = [];
  for (const root of roots) for (const el of [root, ...root.querySelectorAll('*')]) {
    if (!shown(el)) continue;
    const rect = el.getBoundingClientRect();
    if (rect.right > width + 1 || rect.left < -1) offenders.push(el.tagName.toLowerCase() + '.' + [...el.classList].join('.'));
    const style = getComputedStyle(el);
    if (['hidden', 'clip'].includes(style.overflowX) && el.scrollWidth > el.clientWidth + 1 && el.textContent.trim()) clipped.push(el.tagName.toLowerCase() + '.' + [...el.classList].join('.'));
  }
  for (const selector of keys) {
    const el = document.querySelector(selector);
    if (!el) { blocked.push(selector + ': missing'); continue; }
    el.scrollIntoView({block: 'center', inline: 'nearest'});
    const rect = el.getBoundingClientRect(), hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    if (!hit || !(hit === el || el.contains(hit))) blocked.push(selector + ': not hit-testable');
    if (rect.width < 44 || rect.height < 44) blocked.push(selector + ': below 44 px');
  }
  return {scrollWidth: document.documentElement.scrollWidth, innerWidth: width, offenders: offenders.slice(0, 20), clipped: clipped.slice(0, 20), blocked};
})"""


MEMORY_THEME_CHECK = r"""(() => {
    const root = document.querySelector('dialog[open], #memory-view:not([hidden]), #my-memory-view:not([hidden])');
    const dialog = document.querySelector('dialog[open]');
    const scope = dialog || root;
    const rgb = color => color.match(/[\d.]+/g).map(Number);
    const luminance = color => rgb(color).slice(0, 3).map(channel => {
      const value = channel / 255;
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    const background = element => {
      for (let current = element; current; current = current.parentElement) {
        const fill = getComputedStyle(current).backgroundColor;
        if (rgb(fill)[3] !== 0) return fill;
      }
      return getComputedStyle(document.body).backgroundColor;
    };
    const problems = [];
    const expected = document.documentElement.dataset.theme === 'claro' ? 'light' : 'dark';
    if (getComputedStyle(scope).colorScheme !== expected) problems.push('wrong color scheme');
    const elements = [...scope.querySelectorAll('h1,h2,h3,p,li,label>span,summary,button,select')];
    if (!dialog) elements.push(...document.querySelectorAll('#rail-nav button, .home-picker, #session-label'));
    for (const element of elements) {
      if (!element.getClientRects().length || !element.textContent.trim() || element.matches(':disabled')) continue;
      const style = getComputedStyle(element);
      if (scope.contains(element) && Number.parseFloat(style.fontSize) < 14) problems.push(element.className + ': text < 14 px');
      const colors = [luminance(style.color), luminance(background(element))].sort((a, b) => b - a);
      if ((colors[0] + 0.05) / (colors[1] + 0.05) < 4.5) problems.push(element.className + ': contrast < 4.5');
    }
    return [...new Set(problems)];
})()"""


async def capture_memory_state(page: Page, evidence: Path, name: str, selector: str, *,
                               frame_view: str | None = None) -> None:
    """Capture one real view state at the three redesign widths and both saved themes."""
    theme_selector = '#theme-slot select'
    previous = await page.evaluate(f"document.querySelector('{theme_selector}').value")

    async def theme(value: str) -> None:
        active = await page.evaluate(f"""(value => {{
          const select = document.querySelector('{theme_selector}');
          select.value = value;
          select.dispatchEvent(new Event('change'));
          return document.documentElement.dataset.theme;
        }})({json.dumps(value)})""")
        assert value == 'auto' or value == active, (value, active)

    try:
        for value in ('claro', 'oscuro'):
            await theme(value)
            await page.evaluate('new Promise(resolve => setTimeout(resolve, 220))')
            for width in (1440, 768, 390):
                await page.command('Emulation.setDeviceMetricsOverride', {
                    'width': width, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': width < 768,
                })
                await page.evaluate(f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block:'center'}})")
                await screenshot(page, evidence / f'e6-{name}-{value}-{width}.png', full=False, scroll_top=False)
                if frame_view is not None:
                    await assert_frame_consistent(page, frame_view, value)
                    await screenshot(page, evidence / f'e6-{name}-marco-{value}-{width}.png')
                problems = await page.evaluate(MEMORY_THEME_CHECK)
                assert not problems, (name, value, width, problems)
    finally:
        await theme(previous)
        await page.command('Emulation.setDeviceMetricsOverride', {
            'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False,
        })


async def open_memory_hash(page: Page, base: str, route: str, label: str, expected_view: str) -> None:
    """Open a fresh document at a memory route, then wait for the authorized destination."""
    url = f'{base}/?memory-route={route}#{route}'
    await page.command('Page.navigate', {'url': url})
    await page.wait_for(f"""location.search === '?memory-route={route}' &&
        document.querySelector('#session-label')?.textContent === {json.dumps(label)} &&
        document.querySelector('#{expected_view}')?.hidden === false &&
        document.querySelector('#view-transition-status')?.hidden === true""")


async def main() -> None:
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-memory-bootstrap'
    evidence = EVIDENCE
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    requests: list[dict] = []
    checks: list[str] = []
    gates: dict[tuple[str, str], asyncio.Event] = {}

    class DiagnosticPage(Page):
        async def wait_for(self, expression):
            last_evaluation_error = None
            try:
                for _ in range(400):
                    try:
                        if await self.evaluate('!!(' + expression + ')'):
                            return
                    except RuntimeError as exc:
                        # Right after Page.reload the new document may not have the element yet.
                        # Retry and retain the latest error for the timeout diagnostic.
                        last_evaluation_error = str(exc)
                    await asyncio.sleep(0.05)
                raise AssertionError('Browser condition timed out: ' + expression)
            except Exception:
                try:
                    diagnostic = await self.evaluate("({text:document.body?.innerText?.slice(0,3000)||'',dialog:document.querySelector('dialog[open]')?.innerText||null})")
                except RuntimeError as exc:
                    diagnostic = {'evaluation_error': str(exc)}
                (evidence / 'browser-failure.json').write_text(json.dumps({'condition': expression, 'document': diagnostic, 'errors': self.errors,
                                                                          'requests': requests[-40:],
                                                                          'last_evaluation_error': last_evaluation_error}, indent=2, ensure_ascii=False) + '\n')
                await screenshot(self, evidence / 'browser-failure.png')
                raise

    binary = browser_binary()
    browser_data = browser_profile_root(mode=0o700)
    ha, actions = typed_graph.__wrapped__()
    with tempfile.TemporaryDirectory(prefix='kza-nexo-memory-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            client = MemoryClient(directory / 'chroma')
            stack.push_async_callback(asyncio.to_thread, client.close)
            services = await build_memory_services({'memory': {'privacy_db_path': str(directory / 'privacy.db'),
                                                    'preferences_path': str(directory / 'preferences.json')}},
                                                   shutdown=stack, chroma_client=client)
            manager, memory = services.manager, services.manager.long_term
            # Real account catalog: assignment checks the person is an active, non-retired account.
            users = UserManager(str(directory / 'users.json'), retirement_proof=lambda identities: authority.users_retired(identities))
            person_a, _ = users.add_user('Persona sintética A', PermissionLevel.ADULT)
            person_b, _ = users.add_user('Persona sintética B', PermissionLevel.ADULT)
            assert person_a is not None and person_b is not None
            authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=900, account_provider=users.active_accounts)
            await authority.initialize()
            stack.push_async_callback(authority.aclose)
            root = authority.resolve('Bearer synthetic-memory-bootstrap')
            assert root is not None
            expires = time.time() + 1200
            operator = await authority.issue_credential(root, subject='synthetic-operator', role=DashboardRole.OPERATOR, expires_at=expires)
            operator_b = await authority.issue_credential(root, subject='synthetic-operator-b', role=DashboardRole.OPERATOR, expires_at=expires)
            household = await authority.issue_credential(root, subject='synthetic-home', role=DashboardRole.HOME, expires_at=expires)
            home_a = await authority.issue_credential(root, subject='synthetic-home-a', role=DashboardRole.HOME, expires_at=expires, user_id=person_a.user_id)
            selection = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
            evaluator = await authority.issue_credential(root, subject='synthetic-evaluator', role=DashboardRole.EVALUATOR, expires_at=expires, samples=(selection,))
            rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp')}
            dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms, memory_manager=manager, user_manager=users)

            def source(*parents):
                value = manager.source_for_capture(str(uuid4()), time.time(), 'synthetic-room')
                return replace(value, fact_dependencies=tuple(FactDependency(p.id, p.revision, p.expires_at) for p in parents)) if parents else value

            fixture_texts = {}

            async def propose(text):
                identity = (await memory.acall(memory.propose, text, (source(),)))['id']
                fixture_texts[identity] = text
                return identity

            async def automatic(text, *parents):
                identity = await memory.astore_fact(text, 'fact', sources=(source(*parents),))
                fixture_texts[identity] = text
                return identity

            async def get(identity, review=True):
                return await memory.acall(memory.ledger.get, identity, review=review)

            def owner_row(identity):
                row = memory.ledger._db.execute('SELECT person_id FROM fact_owners WHERE fact_id=?', (identity,)).fetchone()
                return None if row is None else row[0]

            async def owner_of(identity):
                return await memory.acall(owner_row, identity)

            async def seed_assign(identity, person):
                fact = await get(identity)
                return await memory.acall(memory.ledger.assign_owner, identity, person, fact.revision, 0, MemoryAssigner('synthetic-seed-operator'),
                                          retired=authority.memory_owner_retired, active=authority.memory_owner_active)

            async def unavailable_live_channel(websocket):
                await websocket.close(code=1013)

            for route in ['/ws/voices', '/ws/evaluation']:
                dashboard.app.router.routes.insert(0, WebSocketRoute(route, unavailable_live_channel))

            @dashboard.app.middleware('http')
            async def record_request(request, call_next):
                gate = gates.get((request.method, request.url.path))
                if gate is not None:
                    await gate.wait()
                response = await call_next(request)
                if request.url.path.startswith('/api/'):
                    # Query parameter names only (never cursor values).
                    requests.append({'method': request.method, 'path': request.url.path, 'status': response.status_code,
                                     'query': sorted(request.query_params.keys())})
                return response

            listener = socket.socket()
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
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
                    async with aiohttp.ClientSession() as http:
                        async with http.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            targets = await response.json()
                    target = next(item for item in targets if item['type'] == 'page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = DiagnosticPage(connection)
                        await page.command('Runtime.enable')
                        await page.command('Page.enable')
                        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': INSTRUMENT})
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        await page.command('Page.navigate', {'url': base + '/'})
                        await page.wait_for("document.querySelector('#login-panel') && document.querySelector('#login-error').textContent.length>0")

                        async def click(selector: str) -> None:
                            await page.evaluate(f'document.querySelector({json.dumps(selector)}).click()')

                        async def press(selector: str) -> None:
                            await page.wait_for(f'(e=>e&&!e.disabled)(document.querySelector({json.dumps(selector)}))')
                            await click(selector)

                        async def rule(method: str, path: str, mode: str) -> None:
                            await page.evaluate(f'window.__qaRules.push({json.dumps({"method": method, "path": path, "mode": mode})})')

                        fixture_rows = FixtureRows('#memory-view .memory-card', '.memory-content')

                        def card(identity: str, inner: str = '') -> str:
                            return json.dumps(f'#memory-view ' + fixture_rows.selector(identity, inner))

                        def key(value: str) -> str:
                            identity, action = value.rsplit(':', 1)
                            return fixture_rows.selector(identity, f'[data-key="{action}"]')

                        def notice(identity: str) -> str:
                            return f"(document.querySelector({card(identity, '.memory-notice')})?.textContent||'')"

                        async def set_hidden(hidden: bool) -> None:
                            await page.evaluate("""(hidden=>{if(!window.__qaHiddenInstalled){window.__qaHiddenInstalled=true;window.__qaHidden=false;
                              Object.defineProperty(document,'hidden',{configurable:true,get:()=>window.__qaHidden});
                              Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>window.__qaHidden?'hidden':'visible'});}
                              window.__qaHidden=hidden;document.dispatchEvent(new Event('visibilitychange'));})(""" + json.dumps(hidden) + ')')

                        def count(method: str, path: str) -> int:
                            return sum(1 for item in requests if item['method'] == method and item['path'] == path)

                        # Operator A: navigation and honest copy.
                        await login(page, operator.token, 'Administración')
                        assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,routines,tasks,activity,people,memory,access'")
                        marker = len(requests)
                        await rule('GET', '/api/memory', 'hold-drop')
                        await click('#app-nav [data-view=memory]')
                        await page.wait_for("window.__qaHeld==='/api/memory' && document.querySelector('#memory-freshness').textContent.startsWith('Consultando')")
                        await capture_memory_state(page, evidence, 'memoria-cargando', '#memory-freshness')
                        await page.evaluate('window.__qaRelease()')
                        await page.wait_for("document.querySelector('#memory-freshness').classList.contains('is-error')")
                        await capture_memory_state(page, evidence, 'memoria-error', '#memory-freshness')
                        await click('#memory-refresh')
                        await page.wait_for("document.querySelector('#memory-freshness').textContent.startsWith('Consultada') && "
                                            "document.querySelectorAll('#memory-view .memory-empty:not([hidden])').length === 3")
                        await capture_memory_state(page, evidence, 'memoria-vacio', '#memory-proposals-title')

                        # Synthetic household memory.
                        confirm_id = await propose('Prefiero el café sin azúcar. Dato sintético de prueba.')
                        stale_id = await propose('La reunión semanal es los martes. Propuesta sintética.')
                        held_id = await propose('Riego las plantas los domingos. Propuesta sintética.')
                        bidi_text = 'Texto sintético ‮odatled‬ con control bidi'
                        bidi_id = await automatic(bidi_text)
                        parent_id = await automatic('Juego al fútbol los jueves. Dato sintético raíz.')
                        parent = await get(parent_id, review=False)
                        child_id = await automatic('Los jueves llego tarde a casa. Derivado sintético.', parent)
                        grandchild_id = await automatic('Los jueves cenamos tarde. Derivado sintético de segundo nivel.', await get(child_id, review=False))
                        dropped_id = await automatic('Uso la bicicleta para ir al trabajo. Dato sintético.')
                        lost_id = await automatic('Mi color favorito es el verde. Dato sintético.')
                        hidden_id = await automatic('Leo antes de dormir. Dato sintético.')
                        long_text = 'Dato sintético largo. ' + 'Esta frase se repite para probar el recorte de textos extensos. ' * 12
                        long_id = await automatic(long_text)
                        n5_proposal_id = await propose('Escucho radio por la mañana. Propuesta sintética para descartar.')
                        n5_parent_id = await automatic('Tengo un perro llamado Sol. Dato sintético raíz para N5.')
                        # U3b-3 owner flows.
                        plain_id = await automatic('Me gusta el té de menta. Dato sintético para asignar.')
                        owner_parent_id = await automatic('Practico guitarra los lunes. Dato sintético raíz con derivado.')
                        owner_child_id = await automatic('Los lunes no cocino. Derivado sintético del dato asignado.', await get(owner_parent_id, review=False))
                        lost_assign_id = await automatic('Camino por la rambla los sábados. Dato sintético para la asignación perdida.')
                        returned_keep_id = await automatic('Tomo mate amargo. Dato sintético que la persona devuelve y se reasigna.')
                        returned_drop_id = await automatic('Colecciono estampillas. Dato sintético que la persona devuelve y se elimina.')
                        for identity in (returned_keep_id, returned_drop_id):
                            await seed_assign(identity, person_a.user_id)

                        labels = [(identity, text[:60]) for identity, text in fixture_texts.items()]
                        await fixture_rows.bind_many(page, labels, prefix=True)

                        await click('#memory-refresh')
                        await page.wait_for(f"location.hash==='#memoria' && !document.querySelector('#memory-view').hidden && document.querySelector({card(confirm_id)})")
                        opened = [item['path'] for item in requests[marker:] if item['path'] in ('/api/admin/auth/whoami', '/api/memory')]
                        assert opened[:2] == ['/api/admin/auth/whoami', '/api/memory'], opened
                        await capture_memory_state(page, evidence, 'memoria-con-datos', '#memory-proposals .memory-card',
                                                   frame_view='#memory-view')
                        assert await page.evaluate("document.querySelector('#section-views').hidden && !document.querySelector('[data-view=mymemory]')")
                        text = await page.evaluate("document.querySelector('#memory-view').innerText")
                        assert 'KZA guarda automáticamente datos que deduce de lo que escucha en el ambiente; ese texto puede procesarse en la nube' in text and 'como máximo 30 días' in text
                        assert 'En esta versión KZA no usa estos recuerdos para responder' in text
                        assert 'La memoria es compartida por todo el hogar: las cuentas de administración ven y revisan los recuerdos de todos' in text
                        # Copy that was false for releases built from HEAD (u3b-design-review factual finding b).
                        for phrase in ('se usa en las respuestas', 'usa en sus respuestas', 'Cualquier persona que hable con KZA', 'deduce de las conversaciones'):
                            assert phrase not in text, phrase
                        assert 'Sólo tu confirmación' not in text and 'Solo tu confirmación' not in text and 'borrado total' not in text.lower()
                        checks.append('nav entry and #memoria hash; whoami precedes the first memory read; F4 copy (ambient source, cloud processing, 30 days, not used in answers, shared and visible to administration); no legacy ownership phrase and no use-in-answers claim')
                        bidi = await page.evaluate(f"(()=>{{const e=document.querySelector({card(bidi_id, '.memory-content')});return {{dir:e.dir,bidi:getComputedStyle(e).unicodeBidi,text:e.textContent}}}})()")
                        assert bidi == {'dir': 'auto', 'bidi': 'isolate', 'text': bidi_text}, bidi
                        checks.append('F15: untrusted text with U+202E rendered verbatim with dir=auto and unicode-bidi:isolate')
                        assert await page.evaluate(f"document.querySelector({card(long_id, '.memory-content')}).textContent.endsWith('…')")
                        await page.evaluate(f"document.querySelector('{key(long_id + ':expand')}').focus()")
                        await press(key(long_id + ':expand'))
                        await page.wait_for(f"!document.querySelector({card(long_id, '.memory-content')}).textContent.endsWith('…') && document.activeElement===document.querySelector({json.dumps(key(long_id + ':expand'))})")
                        checks.append('long text is clamped with an expandable control that keeps focus')

                        # WEB-11: the owner timer removes a newer proposal while the older card has keyboard focus.
                        await page.wait_for("document.querySelector('#memory-refresh').getAttribute('aria-disabled')==='false'")
                        now = time.time()
                        short_source = manager.source_for_capture(str(uuid4()), now, 'web-11-room', expires_at=now + 8)
                        expiring_label = 'Propuesta sintética de corta duración.'
                        expiring_id = (await memory.acall(memory.propose, expiring_label, (short_source,)))['id']
                        await fixture_rows.bind(page, expiring_id, expiring_label)
                        await click('#memory-refresh')
                        await page.wait_for(f"document.querySelector({card(expiring_id)})")
                        await page.evaluate(f"document.querySelector('{key(confirm_id + ':confirm')}').focus()")
                        await _press_tab(page)
                        focus_before = await page.evaluate(ACTIVE_CONTROL)
                        assert await page.evaluate(f"document.activeElement===document.querySelector({json.dumps(key(confirm_id + ':reject'))})")
                        assert focus_before == {'tag': 'BUTTON', 'key': 'reject'}, focus_before
                        assert await page.evaluate(f"document.querySelector({card(expiring_id)})?.compareDocumentPosition(document.activeElement) & Node.DOCUMENT_POSITION_FOLLOWING")
                        reads_before_expiry = count('GET', '/api/memory')
                        await page.wait_for(f"!document.querySelector({card(expiring_id)})")
                        assert count('GET', '/api/memory') == reads_before_expiry
                        focus_after = await page.evaluate(ACTIVE_CONTROL)
                        assert await page.evaluate(f"document.activeElement===document.querySelector({json.dumps(key(confirm_id + ':reject'))})")
                        assert focus_after == focus_before, (focus_before, focus_after)
                        web11 = {'expired_card': expiring_id, 'before': focus_before, 'after': focus_after}
                        checks.append('WEB-11: real Tab reaches the proposal reject button and focus survives another proposal expiring')

                        # Confirm: the card moves to Recuerdos guardados and focus follows it.
                        await page.evaluate(f"document.querySelector('{key(confirm_id + ':confirm')}').focus()")
                        await press(key(confirm_id + ':confirm'))
                        await page.wait_for(f"document.querySelector('#memory-facts {key(confirm_id + ':delete')}') && {notice(confirm_id)}.includes('conservado')")
                        assert (await get(confirm_id)).confirmed_by == 'synthetic-operator'
                        assert await page.evaluate(f"document.querySelector({card(confirm_id)}).contains(document.activeElement)"), 'focus follows the moved card'
                        assert await page.evaluate(f"document.querySelector({card(confirm_id)}).textContent.includes('Confirmado desde esta cuenta')")
                        checks.append('confirm through the view: ledger confirmed_by is the session subject; the card moves and focus follows it')

                        # Stale reject: confirmed by another reviewer after the snapshot (409 provisional, re-read agrees).
                        await memory.acall(memory.confirm, stale_id, MemoryReviewer('synthetic-other-reviewer'))
                        await press(key(stale_id + ':reject'))
                        await page.wait_for(f"{notice(stale_id)}==='Otra sesión ya la confirmó; no se descartó.'")
                        assert (await get(stale_id)).state.value == 'confirmed'
                        assert not await page.evaluate(f"Object.keys(sessionStorage).includes('kza:memory-op:{stale_id}')")
                        checks.append('F3: a stale reject (409) is reported after the re-read agrees; the confirmation elsewhere is preserved')

                        # Delete with the cascade listed; cancel sends nothing.
                        await press(key(parent_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open && document.activeElement?.dataset.key==='confirm-cancel'")
                        dialog = await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        assert 'Es definitivo: no se puede deshacer' in dialog and 'Los jueves llego tarde a casa' in dialog and 'Los jueves cenamos tarde' in dialog, dialog
                        assert 'se limpian en segundo plano; si la limpieza queda pendiente se indica en esta vista' in dialog and 'borrado total' not in dialog.lower()
                        audits = {}
                        await capture_memory_state(page, evidence, 'memoria-eliminar', '#memory-confirm-title')
                        audits['memoria-eliminar'] = await audit_view(page, evidence, 'memoria-eliminar', (
                            ('#memory-confirm-dialog [data-key=confirm-cancel]', None), ('#memory-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await press_escape(page)
                        await page.wait_for(f"!document.querySelector('#memory-confirm-dialog').open && document.activeElement===document.querySelector({json.dumps(key(parent_id + ':delete'))})")
                        assert count('DELETE', f'/api/memory/facts/{parent_id}') == 0
                        await press(key(parent_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(parent_id)}) && !document.querySelector({card(child_id)}) && !document.querySelector({card(grandchild_id)})")
                        assert [await get(item) for item in (parent_id, child_id, grandchild_id)] == [None, None, None]
                        assert await page.evaluate("document.querySelector('#memory-view').contains(document.activeElement)"), 'focus stays in the view'
                        checks.append('F6: the dialog lists the derived memories; cancel sends nothing and returns focus; accept deletes the cascade and the re-read removes all three cards')

                        # Lost DELETE response after the commit: unresolved, then a reload resolves it from a read.
                        await rule('DELETE', f'/api/memory/facts/{dropped_id}', 'drop')
                        await press(key(dropped_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"document.querySelector('#memory-live').textContent==='Comprobado: ya no está guardado.' && !document.querySelector({card(dropped_id)})")
                        assert await get(dropped_id) is None and count('DELETE', f'/api/memory/facts/{dropped_id}') == 1
                        checks.append('lost DELETE response after commit: resolved by a read that started after the loss; one DELETE on the server')

                        # F2: the server is still working when the client gives up; the read sees the fact and must not say "no se eliminó".
                        gate = asyncio.Event()
                        gates[('DELETE', f'/api/memory/facts/{lost_id}')] = gate
                        await rule('DELETE', f'/api/memory/facts/{lost_id}', 'lose')
                        await press(key(lost_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"{notice(lost_id)}==='Todavía figura guardado; puede que el cambio no se haya aplicado.'")
                        assert await page.evaluate(f"document.querySelector({card(lost_id, '[data-memory-marker]')})?.textContent.startsWith('Sin confirmar')")
                        assert not await page.evaluate("document.querySelector('#memory-view').innerText.includes('no se eliminó')")
                        stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:memory-op:{lost_id}'))")
                        assert stored['subject'] == 'synthetic-operator' and stored['op'] == 'delete' and set(stored) <= {'op', 'subject', 'sent_at', 'state', 'reason'}
                        assert await get(lost_id) is not None
                        gate.set()
                        for _ in range(100):
                            if await get(lost_id) is None:
                                break
                            await asyncio.sleep(0.05)
                        assert await get(lost_id) is None
                        await click('#memory-refresh')
                        await page.wait_for(f"document.querySelector('#memory-live').textContent==='Comprobado: ya no está guardado.' && !document.querySelector({card(lost_id)})")
                        assert count('DELETE', f'/api/memory/facts/{lost_id}') == 1
                        del gates[('DELETE', f'/api/memory/facts/{lost_id}')]
                        checks.append('F2: a held server write while the client lost the request: the re-read keeps it unresolved with honest copy; the later commit is reported only after a new read')

                        # Leaving the view never aborts a write.
                        held_path = f'/api/memory/proposals/{held_id}/confirm'
                        await rule('POST', held_path, 'hold')
                        await press(key(held_id + ':confirm'))
                        # The earlier held GET leaves its marker behind until another response is held.
                        await page.wait_for(f'window.__qaHeld === {json.dumps(held_path)}')
                        await click('#app-nav [data-view=home]')
                        await page.wait_for("!document.querySelector('#home-view').hidden && !document.querySelector('#memory-view').children.length")
                        await page.evaluate('window.__qaRelease()')
                        await click('#app-nav [data-view=memory]')
                        await page.wait_for(f"document.querySelector('#memory-facts {key(held_id + ':delete')}')")
                        assert (await get(held_id)).state.value == 'confirmed' and count('POST', f'/api/memory/proposals/{held_id}/confirm') == 1
                        checks.append('navigating away mid-confirm keeps the write; exactly one POST; the private view was emptied while away')

                        # F8: hiding the tab with the delete dialog open drops all fact text and sends nothing.
                        await press(key(hidden_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await set_hidden(True)
                        await page.wait_for("!document.querySelector('#memory-confirm-dialog').open")
                        body = await page.evaluate('document.body.innerText + document.body.textContent')
                        assert 'Leo antes de dormir' not in body and 'Dato sintético' not in body
                        await set_hidden(False)
                        await page.wait_for(f"document.querySelector({card(hidden_id)})")
                        assert count('DELETE', f'/api/memory/facts/{hidden_id}') == 0 and await get(hidden_id) is not None
                        checks.append('F8: a hidden tab closes and empties the delete dialog; no fact text left in the DOM; nothing sent')

                        # N5: a read lands while the delete dialog is open and adds a derived memory: accepting re-asks with the new list.
                        # Server-side gate: the re-read after the reject is answered only after the new derived memory exists.
                        gate = asyncio.Event()
                        gates[('GET', '/api/memory')] = gate
                        await press(key(n5_proposal_id + ':reject'))
                        await page.wait_for("document.querySelector('#memory-live').textContent.startsWith('Propuesta descartada')")
                        await press(key(n5_parent_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        assert 'No hay recuerdos derivados visibles' in await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        n5_child_id = await automatic('Paseo a Sol cada tarde. Derivado sintético agregado con el diálogo abierto.', await get(n5_parent_id, review=False))
                        await fixture_rows.bind(page, n5_child_id, fixture_texts[n5_child_id])
                        gate.set(); del gates[('GET', '/api/memory')]
                        await page.wait_for(f"document.querySelector({card(n5_child_id)})")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open && document.querySelector('#memory-confirm-dialog .memory-changed')")
                        dialog = await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        assert 'Paseo a Sol cada tarde' in dialog and 'cambió mientras decidías' in dialog, dialog
                        assert count('DELETE', f'/api/memory/facts/{n5_parent_id}') == 0, 'nothing sent for the outdated list'
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(n5_parent_id)}) && !document.querySelector({card(n5_child_id)})")
                        assert await get(n5_parent_id) is None and await get(n5_child_id) is None
                        checks.append('N5: a read that changes the derived list while the dialog is open re-asks with the new list before anything is sent')

                        # Accessibility and reflow of the list, plus a 200% text pass.
                        audits['memoria'] = await audit_view(page, evidence, 'memoria', (
                            ('#app-nav [data-view=memory]', None), ('#rail-nav [data-view=access]', None), ('#memory-refresh', None),
                            (key(hidden_id + ':delete'), None), (key(hidden_id + ':assign'), None), (key(confirm_id + ':sources'), None)), screenshot, widths=WIDTHS, text_zoom=True)
                        zoom = {}
                        for width in ZOOM_WIDTHS:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
                            await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                            enlarged = await page.evaluate(TEXT_ZOOM + "(['#memory-view','#rail-nav'])")
                            measured = await page.evaluate(TEXT_ZOOM_CHECK + "(['#memory-view'], " + json.dumps(['#memory-refresh', key(hidden_id + ':delete'), key(hidden_id + ':assign'), key(confirm_id + ':sources')]) + ')')
                            await screenshot(page, evidence / f'memoria-text200-{width}.png')
                            problems = ([f'document scrolls horizontally {measured["scrollWidth"]}>{measured["innerWidth"]}'] if measured['scrollWidth'] > measured['innerWidth'] else []) \
                                + [f'overflows: {item}' for item in measured['offenders']] + [f'clipped: {item}' for item in measured['clipped']] + measured['blocked']
                            zoom[str(width)] = {'elements_enlarged': enlarged, 'measurement': measured, 'findings': problems, 'screenshot': f'memoria-text200-{width}.png'}
                            await page.command('Page.reload')
                            await page.wait_for(f"!document.querySelector('#memory-view').hidden && document.querySelector({card(hidden_id)})")
                        (evidence / 'text-zoom-memoria.json').write_text(json.dumps({'method': 'computed font-size and px line-height of every element in #memory-view and #rail-nav doubled (text-only 200%), viewport at 320 and 390 CSS px', 'widths': zoom}, indent=2, ensure_ascii=False) + '\n')
                        zoom_findings = [f'{width}px: {item}' for width, value in zoom.items() for item in value['findings']]
                        assert not zoom_findings, zoom_findings
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        checks.append('audit_view at 320/390/768/1024/1280/1440 (list and delete dialog) and a 200% text pass at 320/390 without overflow, clipping or unreachable controls')

                        # N1: 'Actualizar' from the keyboard keeps focus while the read is in flight and after it.
                        await rule('GET', '/api/memory', 'hold')
                        await page.evaluate("window.__qaHeld=null;document.querySelector('#memory-refresh').focus()")
                        await press_enter(page)
                        await page.wait_for("window.__qaHeld && document.querySelector('#memory-refresh').getAttribute('aria-disabled')==='true'")
                        assert await page.evaluate("document.activeElement?.id==='memory-refresh'"), 'focus kept while loading'
                        await press_enter(page)
                        await page.evaluate('window.__qaRelease()')
                        await page.wait_for("document.querySelector('#memory-refresh').getAttribute('aria-disabled')==='false'")
                        await asyncio.sleep(0.2)
                        assert await page.evaluate("document.activeElement?.id==='memory-refresh'"), 'focus kept after the refresh'
                        checks.append('N1: keyboard Enter on Actualizar keeps focus on the button during and after the read; a second Enter while loading is ignored')

                        # F5: hidden tab while another tab switches the shared cookie to B, no markers: whoami first, nothing written.
                        await set_hidden(True)
                        await install_login_session(page, base, operator_b.token)
                        marker = len(requests)
                        await set_hidden(False)
                        await page.wait_for("document.querySelector('#session-label').textContent==='Administración' && !document.querySelector('#memory-view').hidden && document.querySelector('#memory-facts .memory-card')")
                        after = requests[marker:]
                        assert after and after[0]['path'] == '/api/admin/auth/whoami', after[:4]
                        assert not any(item['method'] in ('POST', 'DELETE') for item in after), after
                        assert await page.evaluate(f"document.querySelector({card(confirm_id)}).textContent.includes('Confirmado desde otra cuenta de administración')"), 'attribution follows the new account'
                        checks.append('F5: tab return after an account switch with no markers: whoami first, re-bootstrap, attribution re-evaluated for the new account, nothing written')

                        # B leaves an unresolved delete (server held, client lost); A takes the cookie back while the tab is hidden.
                        gate = asyncio.Event()
                        gates[('DELETE', f'/api/memory/facts/{hidden_id}')] = gate
                        await rule('DELETE', f'/api/memory/facts/{hidden_id}', 'lose')
                        await press(key(hidden_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"{notice(hidden_id)}.startsWith('Todavía figura guardado')")
                        assert (await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:memory-op:{hidden_id}'))"))['subject'] == 'synthetic-operator-b'
                        await set_hidden(True)
                        await install_login_session(page, base, operator.token)
                        marker = len(requests)
                        await set_hidden(False)
                        await page.wait_for("document.querySelector('.memory-session-notice') && !document.querySelector('.memory-session-notice').hidden && document.querySelector('.memory-session-notice').textContent.includes('sesión anterior')")
                        assert await page.evaluate(f"sessionStorage.getItem('kza:memory-op:{hidden_id}')===null")
                        after = requests[marker:]
                        assert after[0]['path'] == '/api/admin/auth/whoami', after[:4]
                        assert not any(item['method'] in ('POST', 'DELETE') for item in after), after
                        gate.set()
                        await asyncio.sleep(0.3)
                        del gates[('DELETE', f'/api/memory/facts/{hidden_id}')]
                        assert count('DELETE', f'/api/memory/facts/{hidden_id}') == 1, 'never resent under the other account'
                        checks.append("F5/F11: B's unresolved delete is never resent or resolved under A; the key is wiped and the session-end reminder is shown")

                        # U3b-3: owner labels and counts; the assign dialog states permanence, D5 and the server's A16 notice.
                        await click('#memory-refresh')
                        await page.wait_for(f"document.querySelector({card(plain_id)}) && (document.querySelector('#memory-owner-counts')?.textContent||'').includes('Asignados a personas: 2 · Devueltos: 0')")
                        assert await page.evaluate(f"document.querySelector({card(plain_id, '.memory-owner')}).textContent") == 'Sin asignar'
                        assert await page.evaluate(f"document.querySelector({card(returned_keep_id, '.memory-owner')}).textContent.startsWith('Asignado a Persona sintética A')")
                        await page.evaluate(f"document.querySelector('{key(plain_id + ':assign')}').focus()")
                        await press(key(plain_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open && document.activeElement?.id==='memory-assign-person'")
                        dialog = await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        assert 'queda confirmado y no vence' in dialog and 'La persona va a poder verlo, eliminarlo o devolverlo' in dialog, dialog
                        assert 'Si el recuerdo habla de otra persona, asignarlo se lo muestra a quien elijas.' in dialog, dialog
                        assert memory_api.ASSIGN_NOTICE in dialog, 'the A16 notice from GET /api/memory is shown before acting'
                        options = await page.evaluate("[...document.querySelectorAll('#memory-assign-person option')].map(o=>[o.value,o.textContent])")
                        assert options == [['0', 'Persona sintética A'], ['1', 'Persona sintética B']], options
                        await capture_memory_state(page, evidence, 'memoria-asignar', '#memory-confirm-title')
                        audits['memoria-asignar'] = await audit_view(page, evidence, 'memoria-asignar', (
                            ('#memory-assign-person', None), ('#memory-confirm-dialog [data-key=confirm-cancel]', None), ('#memory-confirm-dialog [data-key=confirm-accept]', None)),
                            screenshot, widths=WIDTHS)
                        await press_escape(page)
                        await page.wait_for(f"!document.querySelector('#memory-confirm-dialog').open && document.activeElement===document.querySelector({json.dumps(key(plain_id + ':assign'))})")
                        assert count('POST', f'/api/memory/facts/{plain_id}/owner') == 0 and await owner_of(plain_id) is None
                        await press(key(plain_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await choose_option(page, '#memory-assign-person', 'Persona sintética A')
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"(document.querySelector({card(plain_id, '.memory-owner')})?.textContent||'').startsWith('Asignado a Persona sintética A') && document.querySelector('#memory-live').textContent.startsWith('Recuerdo asignado.')")
                        stored = await get(plain_id)
                        assert await owner_of(plain_id) == person_a.user_id and stored.state.value == 'confirmed' and stored.expires_at is None
                        assert memory_api.ASSIGN_NOTICE in await page.evaluate("document.querySelector('#memory-live').textContent"), 'the receipt notice is announced'
                        assert count('POST', f'/api/memory/facts/{plain_id}/owner') == 1
                        assert await page.evaluate("!Object.keys(sessionStorage).some(k=>k.startsWith('kza:memory-op:'))")
                        checks.append('U3b-3 assign: owner label and counts; dialog states permanence, the D5 sentence and the server A16 notice before acting; cancel sends nothing and returns focus; accept assigns, confirms without expiry and announces the receipt notice')

                        # 409 has_dependents: the dialog lists labels and states; cancel deletes nothing; accept deletes them and assigns.
                        await press(key(owner_parent_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await choose_option(page, '#memory-assign-person', 'Persona sintética B')
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for("document.querySelector('#memory-assign-dependents li .memory-content')?.textContent.startsWith('Los lunes no cocino')")
                        dialog = await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        assert memory_api.DEPENDENTS_NOTICE in dialog and 'automático' in dialog and 'Los lunes no cocino' in dialog, dialog
                        leaked_id = await page.evaluate(r"""(() => {
                          const root = document.querySelector('#memory-confirm-dialog');
                          return [root.textContent, ...[root, ...root.querySelectorAll('*')].flatMap(node =>
                            [...node.attributes].filter(attr => attr.name === 'title' || attr.name.startsWith('aria-')).map(attr => attr.value))]
                            .some(value => /fact_[0-9a-f]{64}/i.test(value));
                        })()""")
                        assert not leaked_id, 'technical memory ID in dialog text or accessibility attributes'
                        # R1: before confirming, the dialog says the server decides the set at apply time and reports only a count.
                        assert DEPENDENTS_SCOPE in dialog and await page.evaluate("Boolean(document.querySelector('#memory-assign-dependents-scope'))"), dialog
                        assert await owner_of(owner_parent_id) is None and await get(owner_child_id) is not None
                        await capture_memory_state(page, evidence, 'memoria-derivados', '#memory-confirm-title')
                        audits['memoria-derivados'] = await audit_view(page, evidence, 'memoria-derivados', (
                            ('#memory-confirm-dialog [data-key=confirm-cancel]', None), ('#memory-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await press_escape(page)
                        await page.wait_for("!document.querySelector('#memory-confirm-dialog').open")
                        assert await get(owner_child_id) is not None and await owner_of(owner_parent_id) is None
                        assert count('POST', f'/api/memory/facts/{owner_parent_id}/owner') == 1
                        await press(key(owner_parent_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await choose_option(page, '#memory-assign-person', 'Persona sintética B')
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for("document.querySelector('#memory-assign-dependents')")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(owner_child_id)}) && (document.querySelector({card(owner_parent_id, '.memory-owner')})?.textContent||'').startsWith('Asignado a Persona sintética B')")
                        assert await get(owner_child_id) is None and await owner_of(owner_parent_id) == person_b.user_id
                        assert count('POST', f'/api/memory/facts/{owner_parent_id}/owner') == 3
                        checks.append('U3b-3 409 has_dependents: labels and states listed with the server notice; IDs reconciled from the dialog-local map; cancel deletes nothing; the confirmed resend deletes the dependents and assigns')

                        # Reassign and unassign.
                        await press(key(plain_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        assert await page.evaluate("[...document.querySelectorAll('#memory-assign-person option')].map(o=>[o.value,o.textContent])") == [['0', 'Persona sintética B']], 'the current owner is not offered'
                        assert 'deja de verlo en Mis recuerdos' in await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"(document.querySelector({card(plain_id, '.memory-owner')})?.textContent||'').startsWith('Asignado a Persona sintética B')")
                        assert await owner_of(plain_id) == person_b.user_id
                        await press(key(plain_id + ':unassign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        assert 'Persona sintética B deja de verlo en Mis recuerdos' in await page.evaluate("document.querySelector('#memory-confirm-dialog').innerText")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"document.querySelector({card(plain_id, '.memory-owner')})?.textContent==='Sin asignar · la administración quitó la asignación'")
                        assert await owner_of(plain_id) is None and (await get(plain_id)).state.value == 'confirmed'
                        checks.append('U3b-3 reassign offers only other active people and says the current owner loses it; unassign keeps the memory confirmed and labels it')

                        # Lost assignment response after the commit: resolved by a later read.
                        await rule('POST', f'/api/memory/facts/{lost_assign_id}/owner', 'drop')
                        await press(key(lost_assign_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for("document.querySelector('#memory-live').textContent==='Comprobado: el recuerdo quedó asignado y confirmado.'")
                        assert await owner_of(lost_assign_id) == person_a.user_id and count('POST', f'/api/memory/facts/{lost_assign_id}/owner') == 1
                        checks.append('U3b-3 lost assignment response: unresolved, then settled by a read that started after the loss; one POST')

                        # 'Devueltos': person A says 'No es mío' through the real route; the operator reassigns one and deletes the other.
                        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as person_http:
                            async with person_http.post(base + '/api/admin/auth/login', json={'token': home_a.token}) as response:
                                assert response.status == 200
                            for identity in (returned_keep_id, returned_drop_id):
                                async with person_http.post(base + f'/api/memory/mine/facts/{identity}/return') as response:
                                    assert response.status == 200 and (await response.json())['returned'] is True
                        await click('#memory-refresh')
                        await page.wait_for(f"document.querySelector('#memory-returned {key(returned_keep_id + ':assign')}') && document.querySelector('#memory-returned {key(returned_drop_id + ':delete')}') && (document.querySelector('#memory-owner-counts')?.textContent||'').includes('Devueltos: 2')")
                        assert await page.evaluate(f"document.querySelectorAll({card(returned_keep_id)}).length") == 1, 'each memory is shown once'
                        assert 'Una persona marcó estos recuerdos como «No es mío»' in await page.evaluate("document.querySelector('#memory-view').innerText")
                        assert await page.evaluate(f"document.querySelector({card(returned_keep_id, '.memory-owner')}).textContent") == 'Devuelto por la persona («No es mío») · sin asignar'
                        assert any(item['path'] == '/api/memory/returned' and item['status'] == 200 for item in requests)
                        audits['memoria-devueltos'] = await audit_view(page, evidence, 'memoria-devueltos', (
                            (key(returned_keep_id + ':assign'), None), (key(returned_drop_id + ':delete'), None)), screenshot, widths=WIDTHS)
                        await press(key(returned_keep_id + ':assign'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await choose_option(page, '#memory-assign-person', 'Persona sintética B')
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"document.querySelector({card(returned_keep_id)})?.parentElement.id==='memory-facts' && !document.querySelector({card(returned_keep_id)})?.closest('#memory-returned')")
                        assert await owner_of(returned_keep_id) == person_b.user_id
                        await press(key(returned_drop_id + ':delete'))
                        await page.wait_for("document.querySelector('#memory-confirm-dialog').open")
                        await click('#memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(returned_drop_id)}) && (document.querySelector('#memory-owner-counts')?.textContent||'').includes('Devueltos: 0')")
                        assert await get(returned_drop_id) is None
                        checks.append("U3b-3 Devueltos: 'No es mío' through the real person route lands in the paginated queue with its count; reassign moves it out; delete removes it")

                        # retired_owner_prune_pending (synthetic flag on the real ledger attribute) is shown and clears.
                        memory.ledger.retired_owner_prune_pending = True
                        await click('#memory-refresh')
                        await page.wait_for("!document.querySelector('#memory-prune-pending').hidden && document.querySelector('#memory-prune-pending').textContent.includes('cuentas ya retiradas')")
                        memory.ledger.retired_owner_prune_pending = False
                        await click('#memory-refresh')
                        await page.wait_for("document.querySelector('#memory-prune-pending').hidden")
                        checks.append('U3b-3 retired_owner_prune_pending from GET /api/memory/owners/counts is shown as a status and cleared on the next read')

                        # Pagination (release requirement): 100 newer facts push old ones past the first page; the view walks the cursors.
                        for index in range(100):
                            await automatic(f'Dato sintético de relleno número {index}.')
                        marker = len(requests)
                        await click('#memory-refresh')
                        await page.wait_for("document.querySelectorAll('#memory-facts .memory-card').length===100 && (document.querySelector('#memory-facts + .memory-more button')?.textContent||'').startsWith('Mostrar más recuerdos')")
                        walked = [item for item in requests[marker:] if item['path'] == '/api/memory']
                        assert any(item['query'] == ['facts_after'] for item in walked), walked
                        text = await page.evaluate("document.querySelector('#memory-view').innerText")
                        assert 'Se muestran los 100 más recientes' not in text and 'no se pueden revisar ni eliminar desde acá' not in text
                        assert not await page.evaluate(f"document.querySelector({card(confirm_id)})")
                        await page.evaluate("document.querySelector('#memory-facts + .memory-more button').focus()")
                        await press('#memory-facts + .memory-more button')
                        await page.wait_for(f"document.querySelector({card(confirm_id)}) && document.querySelector('#memory-facts').contains(document.activeElement)")
                        assert await page.evaluate(f"!!document.querySelector('{key(confirm_id + ':delete')}')"), 'the old confirmed memory can be reviewed again'
                        checks.append('U3b-3 pagination: the read walks facts_after cursors; older confirmed memories are reachable through Mostrar más (focus moves to the first new card); the 100-cap copy is gone')

                        # Account removal discloses the deletion of assigned memories; cancel removes nothing.
                        await click('#app-nav [data-view=people]')
                        await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===2")
                        # Person cards no longer carry the account id in the DOM: pick the card by its visible name.
                        await page.evaluate("[...document.querySelectorAll('#people-grid [data-person]')].find(card => card.textContent.includes('Persona sintética B')).click()")
                        await page.wait_for("document.querySelector('#person-detail .identity-danger')")
                        await click('#person-detail .identity-danger')
                        await page.wait_for("document.querySelector('#person-dialog').open")
                        removal = await page.evaluate("document.querySelector('#person-dialog').innerText")
                        assert 'También se eliminan, sin posibilidad de recuperarlos, los recuerdos asignados a esta persona en Memoria' in removal, removal
                        await click('#person-dialog .modal-actions .identity-secondary')
                        await page.wait_for("!document.querySelector('#person-dialog').open")
                        assert not any(item['method'] == 'DELETE' and item['path'].startswith('/api/admin/users/') for item in requests)
                        assert await owner_of(plain_id) is None and await owner_of(owner_parent_id) == person_b.user_id
                        checks.append('U3b-3 account removal dialog discloses that the person\'s assigned memories are deleted irreversibly; cancel sends nothing')
                        await click('#app-nav [data-view=memory]')
                        await page.wait_for("document.querySelectorAll('#memory-facts .memory-card').length===100")

                        # Logout wipes markers; HOME and evaluator never reach memory.
                        await click('#logout')
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        assert await page.evaluate("!Object.keys(sessionStorage).some(k=>k.startsWith('kza:memory-op:'))")
                        assert await page.evaluate("!document.querySelector('#memory-view').children.length")
                        await login(page, household.token, 'Acceso de hogar')
                        assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,routines,tasks,activity'")
                        marker = len(requests)
                        await page.evaluate("location.hash='#memoria'")
                        await page.command('Page.reload')
                        await page.wait_for("document.querySelector('#session-label').textContent==='Acceso de hogar' && !document.querySelector('#home-view').hidden")
                        await asyncio.sleep(0.3)
                        assert not any(item['path'].startswith('/api/memory') for item in requests[marker:]), requests[marker:]
                        assert await page.evaluate("document.querySelector('#memory-view').hidden && !document.querySelector('#memory-view').children.length")
                        assert await page.evaluate("fetch('/api/memory').then(r=>r.status)") == 403
                        checks.append('F10 household: no Memoria entry; #memoria falls back to Hogar with no memory request; server 403')
                        await click('#logout')
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        await login(page, home_a.token, 'Acceso de hogar')
                        assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,routines,tasks,activity,mymemory'")
                        marker = len(requests)
                        await page.evaluate("location.hash='#memoria'")
                        await page.command('Page.reload')
                        await page.wait_for("document.querySelector('#session-label').textContent==='Acceso de hogar' && !document.querySelector('#home-view').hidden")
                        await asyncio.sleep(0.3)
                        assert not any(item['path'] == '/api/memory' or item['path'].startswith('/api/memory/') and not item['path'].startswith('/api/memory/mine') for item in requests[marker:]), requests[marker:]
                        assert await page.evaluate("document.querySelector('#memory-view').hidden && !document.querySelector('#memory-view').children.length")
                        for path in ('/api/memory', '/api/memory/returned', '/api/memory/owners/counts'):
                            assert await page.evaluate(f"fetch({json.dumps(path)}).then(r=>r.status)") == 403, path
                        status = await page.evaluate(f"fetch('/api/memory/facts/{plain_id}/owner',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{person_id:{json.dumps(person_a.user_id)},expected_fact_revision:1,expected_owner_revision:0}})}}).then(r=>r.status)")
                        assert status == 403 and await owner_of(plain_id) is None
                        body = await page.evaluate('document.body.innerText')
                        assert 'Practico guitarra' not in body and 'Persona sintética B' not in body
                        checks.append('U3b-3 household bound to a person: Mis recuerdos only, no Memoria entry or operator DOM; operator memory routes and assignment answer 403 and change nothing')
                        await click('#logout')
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        await login(page, evaluator.token, 'Evaluación temporal')
                        await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                        assert await page.evaluate("!document.querySelector('[data-view=memory]')")
                        assert await page.evaluate("fetch('/api/memory').then(r=>r.status)") == 403
                        checks.append('evaluator: no Memoria entry; server 403')
                        await click('#logout')
                        await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                        assert page.errors == [], page.errors

                        result = {'passed': True, 'synthetic_fixture': True, 'actual_chromium_http_login': True, 'real_memory_services': True,
                                  'roles': ['operator', 'operator-b', 'home', 'home-person-a', 'evaluator'], 'widths': list(WIDTHS), 'text_zoom_widths': list(ZOOM_WIDTHS),
                                  'checks': checks, 'web11_focus': web11, 'a11y_evidence': [f'a11y-{view}.json' for view in audits] + ['text-zoom-memoria.json'],
                                  'a11y_findings': {view: len(audit['findings']) for view, audit in audits.items()},
                                  'text_zoom_findings': len(zoom_findings), 'requests': len(requests), 'javascript_exceptions': len(page.errors)}
                        (evidence / 'browser-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
                        print(json.dumps(result, ensure_ascii=False))
            finally:
                for gate in gates.values():
                    gate.set()
                if browser is not None and browser.poll() is None:
                    browser.terminate()
                    await asyncio.to_thread(browser.wait, 10)
                await dashboard.stop()
                await serving


if __name__ == '__main__':
    asyncio.run(main())
