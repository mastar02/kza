"""Exercise real Activity sources and authority in Chromium with synthetic household data."""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import time

import aiohttp
import pytest
import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard import authority as grants
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardPermission, DashboardRole, EvaluationModality, EvaluationSample
from src.reminders.reminder_scheduler import ReminderScheduler, notice_body, notice_request_id
from src.reminders.reminder_store import ReminderScope, ReminderStore
from src.rooms.room_context import RoomConfig
from src.routines.factory import build_routine_scheduler
from src.routines.models import RunStatus, parse_routine_spec
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_nexo_home import NOTICE_LIST, screenshot
from tests.frontend.browser_nexo_memory import TEXT_ZOOM, TEXT_ZOOM_CHECK
from tests.frontend.browser_support import (
    Page, assert_frame_consistent, audit_current_accessibility, audit_view, browser_binary, evidence_directory, install_login_session, press_enter,
    temporary_browser_profile,
)
from tests.integration.test_typed_home_operations import typed_graph


ROUTINES = '/api/routines'
ACTIONS = '/api/home/actions'
READY = "document.querySelector('#activity-refresh')?.getAttribute('aria-disabled') === 'false'"
CONTROLS = (('#activity-refresh', 'Actualizar'),)
BLOCK_IDS = ('activity-actions', 'activity-notices', 'activity-routines')
INSTRUMENT = """(() => {
  const trace = window.__activityTrace = {api: [], openNotices: 0, reconcile: 0, requests: []};
  const fetch = window.fetch.bind(window);
  window.fetch = (input, options) => {
    const path = new URL(input, location.href).pathname;
    trace.requests.push(path);
    if (path === '/api/home/actions' && options?.method === 'POST')
      trace.actionId = JSON.parse(options.body).request_id;
    return fetch(input, options);
  };
  function wrapOnLoad(name, install) {
    let assigned;
    Object.defineProperty(window, name, {
      get: () => assigned,
      set(value) { assigned = value; install(value); }
    });
  }
  wrapOnLoad('NexoState', state => {
    const create = state.createScopedApi;
    state.createScopedApi = options => {
      const api = create(options);
      return (path, init) => { trace.api.push(path); return api(path, init); };
    };
  });
  wrapOnLoad('NexoNoticesOwner', module => {
    const create = module.createNoticesOwner;
    module.createNoticesOwner = options => {
      const owner = create(options), open = owner.open;
      owner.open = () => { trace.openNotices++; return open(); };
      return owner;
    };
  });
  wrapOnLoad('HomeUI', module => {
    const create = module.createActionController;
    module.createActionController = options => {
      const owner = create(options), reconcile = owner.reconcile;
      owner.reconcile = () => { trace.reconcile++; return reconcile(); };
      return owner;
    };
  });
})()"""


async def set_width(page: Page, width: int) -> None:
    """Use the same CSS pixel widths as the other NEXO harnesses."""
    await page.command('Emulation.setDeviceMetricsOverride', {
        'width': width, 'height': 960, 'deviceScaleFactor': 1, 'mobile': width <= 390,
    })


async def set_theme(page: Page, theme: str) -> None:
    """Change the theme through its actual UI control."""
    await page.evaluate(f"""(() => {{
      const control = document.querySelector('#theme-slot select');
      control.value = {json.dumps(theme)};
      control.dispatchEvent(new Event('change'));
    }})()""")
    assert await page.evaluate(f'document.documentElement.dataset.theme === {json.dumps(theme)}')


async def click(page: Page, selector: str) -> None:
    """Activate an existing UI control."""
    await page.evaluate(f'document.querySelector({json.dumps(selector)}).click()')


async def activate_as(page: Page, base: str, token: str, label: str, url: str = '/#actividad') -> None:
    """Use a real HTTP session and load the requested route in a fresh document."""
    await install_login_session(page, base, token)
    destination = base + url
    await page.command('Page.navigate', {'url': destination})
    await page.wait_for(f"location.href === {json.dumps(destination)} && document.querySelector('#login-panel')?.hidden && "
                        f"document.querySelector('#session-label').textContent === {json.dumps(label)}")


async def seed_run(scheduler, name: str, actor: str, source: str):
    """Persist an execution through the same store read by the real list route."""
    spec = parse_routine_spec({'name': name, 'description': '', 'enabled': True,
        'timezone': 'America/Montevideo', 'schedule': None, 'conditions': [],
        'actions': [{'entity_id': 'light.test_lamp', 'operation': 'turn_off'}]})
    routine = await scheduler.store.save(spec, actor=actor)
    claim = await scheduler.store.claim_run(routine.routine_id, 'synthetic-occurrence-' + name,
        actor=actor, source=source, scheduled_for=datetime.now(timezone.utc) if source == 'scheduled' else None)
    await scheduler.store.finish_run(claim.run.run_id, RunStatus.SKIPPED)
    return routine.routine_id, claim.run.run_id


async def assert_absent_sources(page: Page, requests: list[str], marker: int, missing: set[str]) -> dict:
    """Read both sides of the boundary even when scopedApi prevents transport."""
    trace = await page.evaluate('window.__activityTrace')
    dom = await page.evaluate("""({
      headings: [...document.querySelectorAll('#activity-view h2')].map(node => node.id),
      text: document.querySelector('#activity-view').textContent
    })""")
    checks = {
        'action': ('activity-actions', 'Acciones', ACTIONS, trace['reconcile']),
        'notice': ('activity-notices', 'Avisos', NOTICE_LIST, trace['openNotices']),
        'routine': ('activity-routines', 'Rutinas', ROUTINES, 0),
    }
    for source in missing:
        heading, label, path, calls = checks[source]
        assert heading not in dom['headings'], f'{source}: forbidden DOM block'
        assert label not in dom['text'] and 'permiso' not in dom['text'], f'{source}: forbidden fallback text'
        assert calls == 0, f'{source}: forbidden owner call ({calls})'
        assert path not in trace['requests'] and path not in requests[marker:], f'{source}: forbidden network request {path}'
        assert not any(value.split('?')[0] == path for value in trace['api']), f'{source}: forbidden scoped API call {path}'
    return {'missing': sorted(missing), 'headings': dom['headings'],
            'source_spies': {'reconcile': trace['reconcile'], 'open_notices': trace['openNotices']},
            'scoped_paths': sorted(set(trace['api'])), 'network_paths': sorted(set(requests[marker:]))}


async def capture_activity(page: Page, evidence: Path) -> None:
    """Capture both themes and assert accessibility, contrast and enlarged text reflow."""
    for theme in ('oscuro', 'claro'):
        await set_theme(page, theme)
        await assert_frame_consistent(page, '#activity-view', theme)
        audit = await audit_view(page, evidence, 'actividad', CONTROLS, screenshot,
                                 widths=(1440, 768, 390), fresh=READY, theme=theme)
        for width, outcome in audit['widths'].items():
            candidates = [item for item in outcome['wcag_findings'] if item.get('criterion') in ('1.4.3', '1.4.11')
                          and item.get('element') and item.get('level') != 'review']
            failures = await page.evaluate("""(findings => {
              const activity = document.querySelector('#activity-view');
              return findings.filter(item => activity.contains(document.querySelector(item.element)));
            })(""" + json.dumps(candidates) + ')')
            assert not failures, (theme, width, failures)
        geometry = await page.evaluate("""(() => {
          const scope = document.querySelector('#activity-view');
          return [...scope.querySelectorAll('*')].filter(el => el.childNodes.length &&
            [...el.childNodes].some(node => node.nodeType === Node.TEXT_NODE && node.textContent.trim()) &&
            el.getClientRects().length && parseFloat(getComputedStyle(el).fontSize) < 14).map(el => el.outerHTML);
        })()""")
        assert not geometry, ('Activity text under 14 px', geometry)
        x, y = await page.evaluate("""(() => {
          const control = document.querySelector('#activity-refresh');
          control.scrollIntoView({block: 'center'});
          const rect = control.getBoundingClientRect();
          return [rect.x + rect.width / 2, rect.y + rect.height / 2];
        })()""")
        await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'x': x, 'y': y})
        assert await page.evaluate("""(() => {
          const control = document.querySelector('#activity-refresh');
          return control.matches(':hover') &&
            getComputedStyle(control).color === getComputedStyle(document.querySelector('#activity-title')).color;
        })()""")
        await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'x': 0, 'y': 0})
        await page.evaluate("document.querySelector('#activity-refresh').focus()")
        assert await page.evaluate("getComputedStyle(document.activeElement).outlineWidth === '3px'")
        await press_enter(page)
        await page.wait_for(READY)
        assert await page.evaluate("document.activeElement.id === 'activity-refresh'")
        await set_width(page, 320)
        await page.evaluate(TEXT_ZOOM + "(['#activity-view'])")
        resized = await page.evaluate(TEXT_ZOOM_CHECK + "(['#activity-view'], ['#activity-refresh'])")
        resized['titleLines'] = await page.evaluate("""(() => {
          const range = document.createRange();
          range.selectNodeContents(document.querySelector('#activity-title'));
          return range.getClientRects().length;
        })()""")
        (evidence / f'actividad-{theme}-320-texto-200.json').write_text(json.dumps(resized, indent=2) + '\n')
        assert resized['scrollWidth'] <= resized['innerWidth'] + 1 and not any(
            resized[key] for key in ('offenders', 'clipped', 'blocked')), (theme, resized)
        assert resized['titleLines'] == 1, (theme, 'Activity title wraps inside a word at 200%', resized)
        try:
            await screenshot(page, evidence / f'actividad-{theme}-320-texto-200.png')
            await audit_current_accessibility(page, evidence, 'actividad-text200')
        finally:
            await page.evaluate("document.querySelectorAll('#activity-view [style]').forEach(el => el.removeAttribute('style'))")
    await set_theme(page, 'oscuro')
    await set_width(page, 1440)


async def test_nexo_activity_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify account and source boundaries, real projection, logout and reflow."""
    evidence = evidence_directory('KZA_NEXO_ACTIVITY_EVIDENCE', 'data/nexo-activity-browser-runs')
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    bootstrap = 'synthetic-activity-bootstrap'
    monkeypatch.setenv('KZA_DASHBOARD_TOKEN', bootstrap)
    ha, actions = typed_graph.__wrapped__()
    ha.states['light.test_lamp']['attributes']['friendly_name'] = 'Luz principal de prueba'
    rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp')}
    store = ReminderStore(str(tmp_path / 'reminders.db'))
    users = UserManager(str(tmp_path / 'users.json'), retirement_proof=lambda ids: authority.users_retired(ids))
    authority = DashboardAuthority(tmp_path / 'auth.db', session_ttl_s=600, account_provider=users.active_accounts)
    requests = []
    release_routines = asyncio.Event()
    unavailable_routines = stale_notices = False

    async with AsyncExitStack() as stack:
        for resource, close in ((authority, authority.aclose), (store, store.close)):
            stack.push_async_callback(close)
            await resource.initialize()
        scheduler = await build_routine_scheduler(db_path=str(tmp_path / 'routines.db'),
            timezone='America/Montevideo', ha_client=ha, home_actions=actions)
        stack.push_async_callback(scheduler.stop)
        dashboard = DashboardAPI(authority=authority, user_manager=users, ha_client=ha, home_actions=actions,
            home_rooms=rooms, routine_scheduler=scheduler, reminder_scheduler=ReminderScheduler(store, None, None, ha, home_actions=actions))
        stack.push_async_callback(dashboard.stop)
        root = authority.resolve(bootstrap)
        person, _ = users.add_user('Persona sintética', PermissionLevel.ADULT)
        expires = time.time() + 900
        operator = await authority.issue_credential(root, subject='activity-operator', role=DashboardRole.OPERATOR, expires_at=expires)
        visitor = await authority.issue_credential(root, subject='activity-visitor', role=DashboardRole.HOME, expires_at=expires)
        linked = await authority.issue_credential(root, subject='activity-linked', role=DashboardRole.HOME,
                                                   expires_at=expires, user_id=person.user_id)
        sample = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
        evaluator = await authority.issue_credential(root, subject='activity-evaluator', role=DashboardRole.EVALUATOR,
                                                      expires_at=expires, samples=(sample,))

        private_ids = []
        for name, credential, origin in [('Rutina propia visitante', visitor, 'manual'),
                ('Rutina propia vinculada', linked, 'manual'), ('Rutina programada de prueba', operator, 'scheduled')]:
            private_ids.extend(await seed_run(scheduler, name, credential.principal.routine_actor, origin))
            private_ids.extend((credential.principal.subject, credential.principal.routine_actor))
            if credential.principal.account_binding:
                private_ids.append(credential.principal.account_binding)
        triggered = time.time()
        for index, text, delivery in [(1, 'Aviso pendiente de prueba', None), (2, 'Aviso incierto de prueba', 'uncertain'),
                                       (3, 'Aviso confirmado oculto', 'confirmed')]:
            reminder = await store.create('household', text, triggered - 60 + index, scope=ReminderScope.SHARED)
            private_ids.append(reminder.id)
            if delivery:
                key = notice_request_id(reminder.id, reminder.trigger_at, 1)
                assert await store.claim_home_action(key, notice_body(reminder.id, reminder.trigger_at, 1, triggered - 1, 'living'))
                await store.finish_home_action(key, delivery, 'delivery_cancelled' if delivery == 'uncertain' else None)
                assert await store.mark_fired(reminder.id)
        await store.create(person.user_id, 'Aviso personal oculto', triggered + 30, scope=ReminderScope.PRIVATE)

        async def unavailable_live_channel(websocket):
            await websocket.close(code=1013)

        dashboard.app.router.routes.insert(0, WebSocketRoute('/ws/evaluation', unavailable_live_channel))

        @dashboard.app.middleware('http')
        async def observe(request, call_next):
            path = request.url.path
            if path.startswith('/api/'):
                requests.append(path)
            if path == ROUTINES:
                await release_routines.wait()
                if unavailable_routines:
                    return JSONResponse({'detail': 'synthetic-internal-error'}, status_code=503)
            response = await call_next(request)
            if path == NOTICE_LIST and stale_notices:
                response.headers['X-KZA-Observation-State'] = 'stale'
            return response

        listener = stack.enter_context(socket.socket())
        listener.bind(('127.0.0.1', 0))
        base = f'http://127.0.0.1:{listener.getsockname()[1]}'
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))

        async def stop_server() -> None:
            release_routines.set()
            server.should_exit = True
            await serving

        stack.push_async_callback(stop_server)
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(.02)
        profile = await stack.enter_async_context(temporary_browser_profile('kza-activity-'))
        output = stack.enter_context((evidence / 'chromium.log').open('w'))
        browser = subprocess.Popen([browser_binary(), '--headless', '--disable-gpu', '--use-angle=swiftshader',
            '--enable-unsafe-swiftshader', '--no-first-run', '--disable-background-networking',
            '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'], stdout=output, stderr=output)

        async def stop_browser() -> None:
            if browser.poll() is None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, 10)

        stack.push_async_callback(stop_browser)
        port_file = profile / 'DevToolsActivePort'
        async with asyncio.timeout(15):
            while not port_file.exists():
                assert browser.poll() is None, 'Chromium exited before opening CDP'
                await asyncio.sleep(.05)
        async with aiohttp.ClientSession() as client:
            async with client.get(f'http://127.0.0.1:{port_file.read_text().splitlines()[0]}/json/list') as response:
                target = next(item for item in await response.json() if item['type'] == 'page')
        async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
            page = Page(connection)
            await page.command('Runtime.enable')
            await page.command('Page.enable')
            await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': INSTRUMENT})
            await set_width(page, 1440)
            try:
                await activate_as(page, base, operator.token, 'Administración', '/')
                await page.wait_for("!!document.querySelector('#room-controls button[data-operation=turn_on]:enabled')")
                await click(page, '#room-controls button[data-operation=turn_on]')
                await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=turn_on]')?.dataset.status === 'confirmed'")
                action_id = await page.evaluate('window.__activityTrace.actionId')
                assert action_id
                private_ids.append(action_id)
                await page.evaluate("sessionStorage.setItem('kza:routine-pending-ended', '1')")
                await click(page, '#rail-nav [data-section=activity]')
                await page.wait_for("document.querySelector('#activity-routines') && document.querySelector('#activity-view').textContent.includes('Consultando rutinas')")
                assert await page.evaluate("location.hash === '#actividad'")
                await screenshot(page, evidence / 'actividad-cargando.png')
                release_routines.set()
                await page.wait_for(READY)
                assert await page.evaluate("[...document.querySelectorAll('#activity-view h2')].map(el => el.id)") == list(BLOCK_IDS)
                assert await page.evaluate("sessionStorage.getItem('kza:routine-pending-ended') === '1'")
                rendered = await page.evaluate("document.querySelector('#activity-view').outerHTML")
                for text in ('Luz principal de prueba', 'Sala de estar', 'Confirmado por Home Assistant',
                             'solo las acciones que enviaste desde esta pestaña', 'No hay un historial en el servidor',
                             'Aviso pendiente de prueba', 'Aviso incierto de prueba', 'Rutina propia visitante',
                             'Rutina propia vinculada', 'Rutina programada de prueba', 'Programación', 'Sin hora registrada'):
                    assert text in rendered, text
                for hidden in [*private_ids, 'light.test_lamp', 'synthetic-occurrence', 'delivery_cancelled',
                               'Aviso confirmado oculto', 'Aviso personal oculto', 'Simulado']:
                    assert hidden and hidden not in rendered, repr(hidden)
                assert await page.evaluate("document.querySelectorAll('#activity-view [role=status]').length === 1")
                assert await page.evaluate("document.querySelectorAll('#activity-view li time[datetime]').length === 4")
                await capture_activity(page, evidence)

                unavailable_routines = True
                await click(page, '#activity-refresh')
                await page.wait_for(f"{READY} && document.querySelector('#activity-view').textContent.includes('solo parte de la actividad')")
                assert await page.evaluate("document.querySelector('#activity-routines').parentElement.textContent.includes('No se pudieron consultar las rutinas.')")
                assert await page.evaluate("document.querySelector('#activity-view').textContent.includes('Aviso pendiente de prueba') && document.querySelector('#activity-view').textContent.includes('Confirmado por Home Assistant')")
                await screenshot(page, evidence / 'actividad-parcial.png')
                unavailable_routines, stale_notices = False, True
                await click(page, '#activity-refresh')
                await page.wait_for(f"{READY} && document.querySelector('#activity-view').textContent.includes('pueden estar desactualizados')")
                stale_notices = False
                await click(page, '#activity-refresh')
                await page.wait_for(f"{READY} && document.querySelector('#activity-view').textContent.includes('Consulta actualizada')")

                await click(page, '#logout')
                await page.wait_for("!document.querySelector('#login-panel').hidden && document.querySelector('#login-error').textContent === 'Sesión cerrada.'")
                assert await page.evaluate("document.querySelector('#activity-view').childElementCount === 0")
                assert await page.evaluate("!document.querySelector('#rail-nav [data-section=activity]')")

                for credential, name, extras in [(visitor, 'Rutina propia visitante', []),
                                                  (linked, 'Rutina propia vinculada', ['mymemory'])]:
                    await activate_as(page, base, credential.token, 'Acceso de hogar', f'/?case={credential.principal.subject}#actividad')
                    await page.wait_for(READY)
                    menu = await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(el => el.dataset.view)")
                    assert menu == ['home', 'routines', 'tasks', 'activity', *extras]
                    assert await page.evaluate("[...document.querySelectorAll('#activity-view h2')].map(el => el.id)") == list(BLOCK_IDS)
                    assert await page.evaluate("document.querySelector('#activity-routines').parentElement.querySelector('.row-title').textContent") == name
                    assert await page.evaluate("document.querySelector('#activity-routines').parentElement.querySelectorAll('li').length === 1")
                    assert await page.evaluate("document.querySelector('#activity-actions').parentElement.textContent.includes('Todavía no enviaste acciones')")

                await activate_as(page, base, evaluator.token, 'Evaluación temporal')
                await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(el => el.dataset.view)") == ['evaluation']
                assert await page.evaluate("document.querySelector('#activity-view').hidden && !document.querySelector('#activity-view').children.length")

                negative = {}
                for name, denied in [('action', DashboardPermission.HOME_CONTROL),
                                     ('notice', DashboardPermission.TASKS_READ),
                                     ('routine', DashboardPermission.ROUTINES_EXECUTE),
                                     ('view', DashboardPermission.HOME_READ)]:
                    with monkeypatch.context() as limited:
                        limited.setattr(grants, '_HOME_PERMISSIONS', grants._HOME_PERMISSIONS - {denied})
                        marker = len(requests)
                        await activate_as(page, base, visitor.token, 'Acceso de hogar', f'/?limited={name}#actividad')
                        if name == 'view':
                            await page.wait_for("!document.querySelector('#tasks-view').hidden")
                            assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(el => el.dataset.view)") == ['tasks']
                            assert await page.evaluate("document.querySelector('#activity-view').hidden && !document.querySelector('#activity-view').children.length")
                            missing = {'action', 'notice', 'routine'}
                        else:
                            await page.wait_for(READY)
                            await click(page, '#activity-refresh')
                            await page.wait_for(READY)
                            missing = {name}
                        negative[name] = await assert_absent_sources(page, requests, marker, missing)

                assert not page.errors
                (evidence / 'result.json').write_text(json.dumps({'passed': True, 'theme_widths': [1440, 768, 390],
                    'text_zoom': {'width': 320, 'percentage': 200}, 'partial_and_stale': True,
                    'logout_clears_dom_and_receipts': True, 'roles': 4, 'negative': negative}, indent=2) + '\n')
            except Exception:
                await screenshot(page, evidence / 'browser-failure.png')
                raise


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
