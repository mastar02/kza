"""Exercise NEXO task scopes and writes in Chromium against the real local API."""

import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import aiohttp
import pytest
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, OwnerType
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.rooms.room_context import RoomConfig
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_nexo_access import click_text, escape
from tests.frontend.browser_nexo_home import screenshot
from tests.frontend.browser_support import (
    Page, _press_tab, audit_modal_focus, audit_modal_return, audit_view, audit_wcag,
    browser_binary, browser_profile_root, evidence_directory, install_login_session,
)
from tests.integration.test_typed_home_operations import typed_graph


READY = "document.querySelector('#tasks-refresh')?.getAttribute('aria-disabled') === 'false'"
MODAL = '#tasks-confirm-dialog'
WIDTHS = (320, 390, 1440)


async def activate(page: Page, opener: str) -> None:
    await page.evaluate(f"(() => {{ const action = document.querySelector({json.dumps(opener)}); action.focus(); action.click(); }})()")
    await page.wait_for(f"document.querySelector('{MODAL}').open")


async def fill_editor(page: Page, scope: str, label: str, value: str, reminder: bool = False) -> None:
    selector = f'#tasks-view [data-scope={scope}]'
    await click_text(page, selector + ' summary', label)
    await page.evaluate(f"""(() => {{
      const form = document.querySelector({json.dumps(selector + ' details[open] form')});
      form.querySelector('input[type=text]').value = {json.dumps(value)};
      if ({json.dumps(reminder)}) {{
        const date = new Date(Date.now() + 3600000);
        form.querySelector('input[type=datetime-local]').value =
          new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
      }}
      form.requestSubmit();
    }})()""")
    await page.wait_for(f"{READY} && document.querySelector({json.dumps(selector)}).innerText.includes({json.dumps(value)})")


@pytest.mark.asyncio
async def test_nexo_tasks_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify real household and personal requests, mutations and accessibility."""
    evidence = evidence_directory('KZA_NEXO_TASKS_EVIDENCE', 'data/autonomous-roadmap-20261102/t035/nexo-tasks-browser-runs')
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    bootstrap = 'synthetic-nexo-tasks-root'
    monkeypatch.setenv('KZA_DASHBOARD_TOKEN', bootstrap)
    users = UserManager(str(tmp_path / 'users.json'), retirement_proof=lambda ids: authority.users_retired(ids))
    authority = DashboardAuthority(tmp_path / 'auth.db', session_ttl_s=300, account_provider=users.active_accounts)
    lists = ListStore(str(tmp_path / 'lists.db'))
    reminders = ReminderStore(str(tmp_path / 'reminders.db'))
    ha, actions = typed_graph.__wrapped__()
    rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp')}
    requests = []

    async with AsyncExitStack() as stack:
        for resource, close in ((authority, authority.aclose), (lists, lists.close), (reminders, reminders.close)):
            stack.push_async_callback(close)
            await resource.initialize()
        person, _ = await asyncio.to_thread(users.add_user, 'Persona sintética', PermissionLevel.ADULT)
        assert person is not None
        root = authority.resolve(bootstrap)
        assert root is not None
        expires = time.time() + 600
        visitor = await authority.issue_credential(root, subject='visitor', role=DashboardRole.HOME, expires_at=expires)
        linked = await authority.issue_credential(root, subject='linked', role=DashboardRole.HOME, expires_at=expires,
                                                  user_id=person.user_id)
        operator = await authority.issue_credential(root, subject='operator', role=DashboardRole.OPERATOR, expires_at=expires)
        reminder_manager = ReminderManager(reminders, home_actions=actions)
        dashboard = DashboardAPI(authority=authority, user_manager=users, list_manager=ListManager(lists),
                                 reminder_manager=reminder_manager,
                                 ha_client=ha, home_actions=actions, home_rooms=rooms)
        stack.push_async_callback(dashboard.stop)

        @dashboard.app.middleware('http')
        async def record(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith(('/api/lists', '/api/reminders')):
                requests.append((request.method, request.url.path, dict(request.query_params), response.status_code))
            return response

        shared = await lists.create_list('Lista del hogar', OwnerType.SHARED, None)
        await lists.add_item(shared.id, 'Arroz de prueba')
        private = await lists.create_list('Lista privada de prueba', OwnerType.USER, person.user_id)
        await lists.add_item(private.id, 'Elemento privado', added_by=person.user_id)
        trigger = int(time.time()) + 3600
        action = await reminders.create('household', 'Apagar luz de prueba <img src=x>', trigger,
            scope=ReminderScope.SHARED, ha_actions=[{'domain': 'light', 'service': 'turn_off',
                                                    'entity_id': 'light.test_lamp'}])
        plain = await reminders.create('household', 'Recordatorio sin acción', trigger, scope=ReminderScope.SHARED)
        await reminders.create(person.user_id, 'Recordatorio privado de prueba', trigger)
        listener = stack.enter_context(socket.socket())
        listener.bind(('127.0.0.1', 0))
        base = f'http://127.0.0.1:{listener.getsockname()[1]}'
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))

        async def stop_server() -> None:
            server.should_exit = True
            await serving

        stack.push_async_callback(stop_server)
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(.02)

        profile = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='kza-tasks-', dir=browser_profile_root())))
        output = stack.enter_context((evidence / 'chromium.log').open('w'))
        browser = subprocess.Popen([browser_binary(), '--headless', '--disable-gpu', '--no-first-run',
            '--disable-background-networking', '--remote-debugging-port=0', f'--user-data-dir={profile}', 'about:blank'],
            stdout=output, stderr=output)

        async def stop_browser() -> None:
            if browser.poll() is None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, 10)

        stack.push_async_callback(stop_browser)
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
            await install_login_session(page, base, visitor.token)
            await page.command('Page.navigate', {'url': base + '/#tareas'})
            await page.wait_for(f"{READY} && !!document.querySelector('#tasks-shared-title')")
            assert await page.evaluate("""(() => {
              const view = document.querySelector('#tasks-view'), text = view.innerText;
              return !view.hidden && !text.includes('Mis tareas') && !text.includes('prueba privada') &&
                text.includes('Lista del hogar') && text.includes('Arroz de prueba') &&
                text.includes('Apagar luz de prueba <img src=x>') && !view.querySelector('img') &&
                text.includes('light/turn_off') && text.includes('light.test_lamp') &&
                text.includes('Sala de estar') && !text.includes('Ámbito · Personal') &&
                view.querySelectorAll('.tasks-actions').length === 1;
            })()""")
            assert await page.evaluate(f"Date.parse(document.querySelector('#tasks-view time').dateTime) / 1000 === {trigger}")
            assert {q.get('scope') for _, _, q, _ in requests} == {None, 'shared'}

            opener = f'[data-task-key="shared:reminder:{action.id}"]'
            visitor_audit = await audit_view(page, evidence, 'tareas-visitante', (
                ('#app-nav [data-view=tasks]', None), ('#tasks-refresh', None),
                ('#tasks-view summary', 'Nueva lista · compartida del hogar'), (opener, None),
            ), screenshot, widths=WIDTHS, restore=(1440, 900, False), fresh=READY)
            assert await page.evaluate("""[...document.querySelectorAll('#tasks-view button, #tasks-view summary')]
              .filter(control => control.getClientRects().length)
              .every(control => { const box = control.getBoundingClientRect(); return box.width >= 44 && box.height >= 44; })""")
            for width in WIDTHS:
                await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900,
                                                                        'deviceScaleFactor': 1, 'mobile': width <= 390})
                await activate(page, opener)
                report = await audit_modal_focus(page, MODAL, initial='#tasks-confirm-cancel')
                assert await page.evaluate("""(() => {
                  const dialog = document.querySelector('#tasks-confirm-dialog');
                  return dialog.querySelector('#tasks-confirm-target').textContent === 'Apagar luz de prueba <img src=x>' &&
                    [...dialog.querySelectorAll('button')].every(button => {
                      const box = button.getBoundingClientRect(); return box.width >= 44 && box.height >= 44;
                    });
                })()""")
                await escape(page)
                await page.wait_for(f"!document.querySelector('{MODAL}').open")
                await audit_modal_return(page, evidence, f'tareas-eliminar-{width}', report, dialog=MODAL, opener=opener)
                assert report['findings'] == []
            assert not any(method == 'DELETE' for method, _, _, _ in requests)

            await activate(page, opener)
            await click_text(page, MODAL + ' button', 'Eliminar')
            await page.wait_for(f"{READY} && document.querySelector('#tasks-feedback').innerText === 'Recordatorio eliminado.'")
            assert (await reminders.get_by_id(action.id, shared_only=True)).state is ReminderState.CANCELLED
            delete_index = requests.index(('DELETE', f'/api/reminders/{action.id}', {}, 200))
            assert ('GET', '/api/reminders', {'scope': 'shared'}, 200) in requests[delete_index + 1:]
            assert await page.evaluate("document.activeElement.id === 'tasks-refresh' && !document.querySelector('.tasks-actions')")

            await activate(page, f'[data-task-key="shared:reminder:{plain.id}"]')
            assert await reminders.cancel(plain.id, shared_only=True)
            await click_text(page, MODAL + ' button', 'Eliminar')
            await page.wait_for(f"{READY} && document.querySelector('#tasks-feedback').innerText.includes('cambió')")
            assert ('DELETE', f'/api/reminders/{plain.id}', {}, 409) in requests
            assert await page.evaluate("document.querySelector('#tasks-view').innerText.includes('No hay recordatorios activos.')")

            await fill_editor(page, 'shared', 'Añadir elemento', 'Elemento compartido nuevo')
            item = next(row for row in await lists.get_items(shared.id) if row.text == 'Elemento compartido nuevo')
            await activate(page, f'[data-task-key="shared:item:{shared.id}:{item.id}"]')
            await click_text(page, MODAL + ' button', 'Eliminar')
            await page.wait_for(f"{READY} && document.querySelector('#tasks-feedback').innerText === 'Elemento eliminado.'")
            assert ('DELETE', f'/api/lists/{shared.id}/items/{item.id}', {}, 200) in requests

            dashboard.reminder_manager = None
            await page.evaluate("document.querySelector('#tasks-refresh').click()")
            await page.wait_for(f"{READY} && document.querySelector('#tasks-status').innerText.includes('no está disponible')")
            assert await page.evaluate("!document.querySelector('#tasks-view .tasks-scope') && document.querySelector('#tasks-status').role === 'status'")
            dashboard.reminder_manager = reminder_manager
            await page.evaluate("document.querySelector('#tasks-refresh').click()")
            await page.wait_for(f"{READY} && !!document.querySelector('#tasks-shared-title')")

            await fill_editor(page, 'shared', 'Nueva lista · compartida del hogar', 'Lista compartida nueva')
            await fill_editor(page, 'shared', 'Nuevo recordatorio · compartida del hogar', 'Recordatorio compartido nuevo', reminder=True)
            assert not any(query.get('scope') == 'personal' for _, _, query, _ in requests)

            # Another tab can change the cookie without changing this tab's presentation.
            await install_login_session(page, base, linked.token)
            await page.evaluate("document.querySelector('#tasks-refresh').click()")
            await page.wait_for(f"{READY} && document.querySelector('#tasks-personal-title')?.textContent === 'Mis tareas'")
            assert await page.evaluate("""(() => {
              const shared = document.querySelector('#tasks-view [data-scope=shared]').innerText;
              const personal = document.querySelector('#tasks-view [data-scope=personal]').innerText;
              return personal.includes('Lista privada de prueba') && personal.includes('Elemento privado') &&
                personal.includes('Recordatorio privado de prueba') && !shared.includes('privado');
            })()""")
            personal_audit = await audit_view(page, evidence, 'tareas-personales', (
                ('#app-nav [data-view=tasks]', None), ('#tasks-refresh', None),
                ('#tasks-view summary', 'Nueva lista · personal'),
                ('#tasks-view summary', 'Nuevo recordatorio · personal'),
            ), screenshot, widths=WIDTHS, restore=(1440, 900, False), fresh=READY)
            personal_panel = '#tasks-view [data-scope=personal]'
            await click_text(page, personal_panel + ' summary', 'Nuevo recordatorio · personal')
            form = personal_panel + ' details[open] form'
            form_selector = json.dumps(form)
            form_audit = {}
            for width in WIDTHS:
                await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900,
                                                                        'deviceScaleFactor': 1, 'mobile': width <= 390})
                assert await page.evaluate(f"""(() => {{
                  const controls = [...document.querySelector({form_selector}).elements];
                  return document.documentElement.scrollWidth <= innerWidth &&
                    controls.every(control => {{
                      const box = control.getBoundingClientRect(); return box.width >= 44 && box.height >= 44;
                    }});
                }})()""")
                await page.evaluate(f"document.querySelector({form_selector}).elements[0].focus()")
                await _press_tab(page)
                assert await page.evaluate("document.activeElement.type === 'datetime-local'")
                # Each part of the native date and time field can receive a Tab
                # while the document's activeElement remains the same.
                for _ in range(10):
                    await _press_tab(page)
                    if await page.evaluate(f"document.activeElement === document.querySelector({form_selector}).elements[2]"):
                        break
                else:
                    pytest.fail('El teclado no llegó al botón Guardar')
                await screenshot(page, evidence / f'tareas-formulario-{width}.png')
                findings = await audit_wcag(page)
                form_findings = await page.evaluate(f"""{json.dumps(findings)}.filter(item =>
                    item.element && document.querySelector(item.element)?.closest({form_selector}))""")
                form_audit[str(width)] = {'keyboard_reached_submit': True, 'findings': form_findings}
            (evidence / 'a11y-tareas-formulario.json').write_text(json.dumps(form_audit, indent=2) + '\n', encoding='utf-8')
            assert all(not result['findings'] for result in form_audit.values())
            await click_text(page, personal_panel + ' summary', 'Nuevo recordatorio · personal')
            await fill_editor(page, 'personal', 'Nueva lista · personal', 'Lista personal nueva')
            await fill_editor(page, 'personal', 'Nuevo recordatorio · personal', 'Recordatorio personal nuevo', reminder=True)
            own = await lists.get_lists_for_user(person.user_id)
            assert any(row.name == 'Lista personal nueva' and row.owner_id == person.user_id for row in own)
            own_reminders = await reminders.get_active_for_user(person.user_id)
            assert any(row.text == 'Recordatorio personal nuevo' and row.scope is ReminderScope.PRIVATE for row in own_reminders)

            await activate(page, '#tasks-view [data-scope=personal] [data-task-key*=":reminder:"]')
            await install_login_session(page, base, operator.token)
            previous = len(requests)
            await click_text(page, MODAL + ' button', 'Eliminar')
            await page.wait_for(f"{READY} && document.querySelector('#session-label').textContent === 'Administración'")
            assert await page.evaluate("""!document.querySelector('#tasks-personal-title') &&
                !document.querySelector('#tasks-confirm-dialog').open &&
                !document.querySelector('#tasks-confirm-dialog').children.length &&
                !document.querySelector('#tasks-view').innerText.includes('privada')""")
            assert all(method != 'DELETE' and query.get('scope') != 'personal' for method, _, query, _ in requests[previous:])
            assert all('user_id' not in query for _, _, query, _ in requests)
            assert not page.errors
            (evidence / 'result.json').write_text(json.dumps({
                'widths': sorted(visitor_audit['widths']), 'personal_widths': sorted(personal_audit['widths']),
                'visitor_and_operator_are_shared_only': True, 'authenticated_personal_creation': True,
                'cancel_and_conflict_reconciled': True, 'service_unavailable_and_recovered': True,
                'list_item_add_and_delete': True, 'account_change_drops_private_content': True,
            }, indent=2) + '\n', encoding='utf-8')
