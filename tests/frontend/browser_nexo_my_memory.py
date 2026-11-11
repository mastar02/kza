"""Real Chromium, real logins and real memory services for NEXO 'Mis recuerdos' (U3b-3).

A household credential bound to person A sees only A's assigned memories through
GET /api/memory/mine; 'No es mío' and delete go through the real person routes.
Negative cases: a household credential without a person, B, an evaluator and an
operator; a voice profile linked to A grants nothing. Evidence never serializes
a credential, a cookie, a request body or fact text: only methods, paths,
query parameter names, statuses and counts.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import json
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
from src.users.user_manager import PermissionLevel, UserManager
from src.users.voice_ledger import VoiceReviewer
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_nexo_memory import INSTRUMENT, TEXT_ZOOM, TEXT_ZOOM_CHECK
from tests.frontend.browser_nexo_voices import person_link
from tests.frontend.browser_support import Page, audit_view, browser_binary, browser_profile_root, evidence_directory, install_login_session
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.voice_registry import ACTOR, graph as voice_graph, seed as voice_seed

# Copy shipped by my-memory.js / my-memory-owner.js (R3, R6).
CASCADE = ('Se elimina además todo lo que dependa de este recuerdo al momento de borrarlo, aunque haya aparecido después de abrir '
           'este aviso. No se informa cuántos datos derivados se eliminaron.')
RESTORED_STILL = ('Todavía figura entre tus recuerdos. La operación quedó de una sesión anterior y no se reenvía sola: si querés, '
                  'hacela de nuevo.')


EVIDENCE = evidence_directory('KZA_NEXO_MY_MEMORY_EVIDENCE', 'data/autonomous-roadmap-20261102/u3b/u3b-3-dev/my-memory-browser-runs')
WIDTHS = (320, 390, 768, 1024, 1280, 1440)
ZOOM_WIDTHS = (320, 390)


async def main() -> None:
    evidence = EVIDENCE
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    requests: list[dict] = []
    checks: list[str] = []
    gates: dict[tuple[str, str], asyncio.Event] = {}

    class DiagnosticPage(Page):
        async def wait_for(self, expression):
            try:
                for _ in range(400):
                    if await self.evaluate('!!(' + expression + ')'):
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError('Browser condition timed out: ' + expression)
            except Exception:
                diagnostic = await self.evaluate("({text:document.body.innerText.slice(0,3000),dialog:document.querySelector('dialog[open]')?.innerText||null})")
                (evidence / 'browser-failure.json').write_text(json.dumps({'condition': expression, 'document': diagnostic, 'errors': self.errors,
                                                                          'requests': requests[-40:]}, indent=2, ensure_ascii=False) + '\n')
                await screenshot(self, evidence / 'browser-failure.png')
                raise

    binary = browser_binary()
    browser_data = browser_profile_root(mode=0o700)
    with tempfile.TemporaryDirectory(prefix='kza-nexo-my-memory-', dir=browser_data) as temporary:
        directory = Path(temporary)
        (directory / 'voice').mkdir()
        async with AsyncExitStack() as stack:
            users = UserManager(str(directory / 'users.json'), retirement_proof=lambda identities: authority.users_retired(identities))
            person_a, _ = users.add_user('Persona sintética A', PermissionLevel.ADULT)
            person_b, _ = users.add_user('Persona sintética B', PermissionLevel.ADULT)
            assert person_a is not None and person_b is not None
            authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=900, account_provider=users.active_accounts,
                                           expected_token=lambda: 'synthetic-my-memory-root')
            await authority.initialize()
            stack.push_async_callback(authority.aclose)
            root = authority.resolve('synthetic-my-memory-root')
            assert root is not None
            expires = time.time() + 1200
            home_a = await authority.issue_credential(root, subject='synthetic-home-a', role=DashboardRole.HOME, expires_at=expires, user_id=person_a.user_id)
            home_b = await authority.issue_credential(root, subject='synthetic-home-b', role=DashboardRole.HOME, expires_at=expires, user_id=person_b.user_id)
            shared = await authority.issue_credential(root, subject='synthetic-home-shared', role=DashboardRole.HOME, expires_at=expires)
            operator = await authority.issue_credential(root, subject='synthetic-operator', role=DashboardRole.OPERATOR, expires_at=expires)
            selection = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), expires)
            evaluator = await authority.issue_credential(root, subject='synthetic-evaluator', role=DashboardRole.EVALUATOR, expires_at=expires, samples=(selection,))

            # A reviewed voice profile linked to person A (real voice ledger and link CAS): it must grant nothing.
            _privacy, _owner, voices, voice_clock = await stack.enter_async_context(voice_graph(directory / 'voice'))
            _sources, identities = await voice_seed(voices, voice_clock)
            profile = await voices.acall(voices.ledger.confirm, identities, 'Voz sintética de A', str(uuid4()), ACTOR)
            linker_session = await authority.login(operator.token)
            linker = VoiceReviewer.from_dashboard(linker_session.lease.principal,
                                                  lambda: linker_session.lease.current(frozenset({'voices:review', 'admin:manage'})), person_authority=authority)
            await voices.acall(voices.ledger.people.set_person, profile['id'], person_a.user_id, 1, 0, linker)
            assert await voices.acall(person_link, voices, profile['id']) == (person_a.user_id, 1)

            client = MemoryClient(directory / 'chroma')
            stack.push_async_callback(asyncio.to_thread, client.close)
            services = await build_memory_services({'memory': {'privacy_db_path': str(directory / 'memory-privacy.db'),
                                                    'preferences_path': str(directory / 'preferences.json')}},
                                                   shutdown=stack, chroma_client=client)
            manager, memory = services.manager, services.manager.long_term
            dashboard = DashboardAPI(authority=authority, memory_manager=manager, user_manager=users)

            def source(*parents):
                value = manager.source_for_capture(str(uuid4()), time.time(), 'synthetic-room')
                return replace(value, fact_dependencies=tuple(FactDependency(p.id, p.revision, p.expires_at) for p in parents)) if parents else value

            async def get(identity, review=True):
                return await memory.acall(memory.ledger.get, identity, review=review)

            async def automatic(text, *parents):
                return await memory.astore_fact(text, 'preference', sources=(source(*parents),))

            async def confirmed(text):
                proposal = (await memory.acall(memory.propose, text, (source(),)))['id']
                await memory.acall(memory.confirm, proposal, MemoryReviewer('synthetic-seed-reviewer'))
                return proposal

            async def assign(identity, person):
                fact = await get(identity)
                current = await memory.acall(lambda: memory.ledger._db.execute(
                    'SELECT revision FROM fact_owner_revisions WHERE fact_id=?', (identity,)).fetchone())
                return await memory.acall(memory.ledger.assign_owner, identity, person, fact.revision, 0 if current is None else current[0],
                                          MemoryAssigner('synthetic-seed-operator'), retired=authority.memory_owner_retired, active=authority.memory_owner_active)

            def owner_row(identity):
                row = memory.ledger._db.execute('SELECT person_id FROM fact_owners WHERE fact_id=?', (identity,)).fetchone()
                change = memory.ledger._db.execute('SELECT last_change FROM fact_owner_revisions WHERE fact_id=?', (identity,)).fetchone()
                return (None if row is None else row[0], None if change is None else change[0])

            async def owner_of(identity):
                return await memory.acall(owner_row, identity)

            # Synthetic memories: A's own, B's own, and household memories nobody owns.
            texts = {}

            async def own(name, text, person, *, parent=None):
                identity = await (automatic(text) if parent is None else automatic(text, parent))
                await assign(identity, person)
                texts[name] = text
                return identity

            a_return = await own('a_return', 'Mi cumpleaños es en abril. Recuerdo sintético de A para devolver.', person_a.user_id)
            a_lost = await own('a_lost', 'Prefiero dormir con la ventana abierta. Recuerdo sintético de A con respuesta perdida.', person_a.user_id)
            a_held = await own('a_held', 'Tomo café con leche. Recuerdo sintético de A con el servidor demorado.', person_a.user_id)
            a_hidden = await own('a_hidden', 'Leo novelas policiales. Recuerdo sintético de A para ocultar la pestaña.', person_a.user_id)
            a_leave = await own('a_leave', 'Voy al gimnasio temprano. Recuerdo sintético de A para salir de la vista.', person_a.user_id)
            bidi_text = 'Texto sintético de A ‮odatled‬ con control bidi'
            a_bidi = await own('a_bidi', bidi_text, person_a.user_id)
            # A confirmed parent with an unowned derived memory: deleting it says derived data goes too.
            a_parent = await confirmed('Trabajo desde casa los viernes. Recuerdo sintético de A con un derivado.')
            await assign(a_parent, person_a.user_id)
            texts['a_parent'] = 'Trabajo desde casa los viernes'
            hidden_child = await automatic('Los viernes almuerzo en casa. Derivado sintético sin dueño.', await get(a_parent, review=False))
            b_fact = await own('b_fact', 'Mi clave del portón termina en siete. Recuerdo privado sintético de B.', person_b.user_id)
            household_fact = await automatic('La heladera hace ruido de noche. Dato sintético del hogar sin dueño.')
            foreign_texts = ['Recuerdo privado sintético de B', 'Dato sintético del hogar sin dueño', 'Derivado sintético sin dueño']

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
                profile_dir = directory / 'chromium'
                with (evidence / 'chromium.log').open('w') as output:
                    browser = await asyncio.to_thread(subprocess.Popen, [
                        binary, '--headless', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
                        '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update',
                        '--disable-sync', '--remote-debugging-port=0', f'--user-data-dir={profile_dir}', 'about:blank',
                    ], stdout=output, stderr=output)
                    active_port = profile_dir / 'DevToolsActivePort'
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

                        def card(identity: str, inner: str = '') -> str:
                            return json.dumps(f'#my-memory-view [data-my-memory-id="{identity}"] {inner}'.strip())

                        def key(value: str) -> str:
                            return f'#my-memory-view [data-key="{value}"]'

                        def notice(identity: str) -> str:
                            return f"(document.querySelector({json.dumps(f'#my-memory-view [data-memory-notice=\"{identity}\"]')})?.textContent||'')"

                        async def set_hidden(hidden: bool) -> None:
                            await page.evaluate("""(hidden=>{if(!window.__qaHiddenInstalled){window.__qaHiddenInstalled=true;window.__qaHidden=false;
                              Object.defineProperty(document,'hidden',{configurable:true,get:()=>window.__qaHidden});
                              Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>window.__qaHidden?'hidden':'visible'});}
                              window.__qaHidden=hidden;document.dispatchEvent(new Event('visibilitychange'));})(""" + json.dumps(hidden) + ')')

                        def count(method: str, path: str) -> int:
                            return sum(1 for item in requests if item['method'] == method and item['path'] == path)

                        async def dom_text() -> str:
                            return await page.evaluate('document.body.innerText + "\\n" + document.body.textContent')

                        async def logout() -> None:
                            await click('#logout')
                            await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                        # Person A: navigation, whoami first, own memories only, honest copy.
                        await login(page, home_a.token, 'Acceso de hogar')
                        assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,actions,routines,mymemory'")
                        marker = len(requests)
                        await click('#app-nav [data-view=mymemory]')
                        await page.wait_for(f"location.hash==='#mis-recuerdos' && !document.querySelector('#my-memory-view').hidden && document.querySelector({card(a_return)})")
                        opened = [item['path'] for item in requests[marker:] if item['path'] in ('/api/admin/auth/whoami', '/api/memory/mine')]
                        # U3b-4a-3: the voice section confirms the account on its own, so a second whoami may precede the list.
                        assert opened[0] == '/api/admin/auth/whoami' and '/api/memory/mine' in opened, opened
                        assert not any(item['path'] == '/api/memory' or item['path'].startswith('/api/memory/') and not item['path'].startswith('/api/memory/mine')
                                       for item in requests[marker:]), requests[marker:]
                        text = await page.evaluate("document.querySelector('#my-memory-view').innerText")
                        for phrase in ('Recuerdos que la administración del hogar te asignó. Ves solo los tuyos.',
                                       'Las cuentas de administración del hogar ven todos los recuerdos, también estos, y pueden eliminarlos o reasignarlos.',
                                       'En esta versión KZA no usa estos recuerdos para responder.',
                                       'Si un recuerdo no es tuyo, usá «No es mío»: vuelve a la administración y no se borra.'):
                            assert phrase in text, phrase
                        assert 'borrado total' not in text.lower() and 'usa en sus respuestas' not in text
                        ids = await page.evaluate("[...document.querySelectorAll('#my-memory-view [data-my-memory-id]')].map(n=>n.dataset.myMemoryId)")
                        assert set(ids) == {a_return, a_lost, a_held, a_hidden, a_leave, a_bidi, a_parent}, ids
                        body = await dom_text()
                        assert not any(phrase in body for phrase in foreign_texts), 'no other person\'s or household memory in the DOM'
                        assert b_fact not in body and household_fact not in body and hidden_child not in body
                        assert await page.evaluate(f"document.querySelector({card(a_return)}).textContent.includes('Confirmado por la administración · no vence')")
                        assert await page.evaluate(f"document.querySelector({card(a_return)}).textContent.includes('Asignado por la administración')")
                        assert not await page.evaluate("!!document.querySelector('[data-view=memory]') || document.querySelector('#memory-view').children.length > 0")
                        checks.append('person A: Mis recuerdos entry and #mis-recuerdos hash; whoami precedes /api/memory/mine; only A\'s memories; no B or household content in the DOM; privacy copy (what is visible, administration sees everything, not used in answers, No es mío is not a deletion); no operator view or request')
                        bidi = await page.evaluate(f"(()=>{{const e=document.querySelector({card(a_bidi, '.memory-content')});return {{dir:e.dir,bidi:getComputedStyle(e).unicodeBidi,text:e.textContent}}}})()")
                        assert bidi == {'dir': 'auto', 'bidi': 'isolate', 'text': bidi_text}, bidi
                        checks.append('untrusted text with U+202E rendered verbatim with dir=auto and unicode-bidi:isolate')

                        # 'No es mío': the dialog states the notice before acting; cancel sends nothing; accept returns it.
                        await page.evaluate(f"document.querySelector('{key(a_return + ':return')}').focus()")
                        await press(key(a_return + ':return'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open && document.activeElement?.dataset.key==='confirm-cancel'")
                        dialog = await page.evaluate("document.querySelector('#my-memory-confirm-dialog').innerText")
                        assert memory_api.RETURN_NOTICE in dialog and 'Se devuelve a la administración del hogar' in dialog, dialog
                        audits = {}
                        audits['mis-recuerdos-no-es-mio'] = await audit_view(page, evidence, 'mis-recuerdos-no-es-mio', (
                            ('#my-memory-confirm-dialog [data-key=confirm-cancel]', None), ('#my-memory-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await click('#my-memory-confirm-dialog [data-key="confirm-cancel"]')
                        await page.wait_for(f"!document.querySelector('#my-memory-confirm-dialog').open && document.activeElement?.dataset.key==='{a_return}:return'")
                        assert count('POST', f'/api/memory/mine/facts/{a_return}/return') == 0
                        await press(key(a_return + ':return'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(a_return)}) && document.querySelector('#my-memory-return-notice').textContent==={json.dumps(memory_api.RETURN_NOTICE)} && !document.querySelector('#my-memory-return-notice').hidden")
                        assert await owner_of(a_return) == (None, 'owner_returned') and (await get(a_return)).state.value == 'confirmed'
                        assert count('POST', f'/api/memory/mine/facts/{a_return}/return') == 1
                        assert await page.evaluate("document.querySelector('#my-memory-view').contains(document.activeElement)"), 'focus stays in the view'
                        checks.append("No es mío: the confirmation states it is not deleted and goes to administration before acting; cancel sends nothing and returns focus; accept returns it (owner row gone, last_change owner_returned, still confirmed) and the notice stays visible")

                        # Delete: irreversible dialog with the hidden-dependents sentence and erasure limits; cascade server-side.
                        await press(key(a_parent + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open && document.activeElement?.dataset.key==='confirm-cancel'")
                        dialog = await page.evaluate("document.querySelector('#my-memory-confirm-dialog').innerText")
                        assert 'Es definitivo: no se puede deshacer.' in dialog and 'También se eliminan datos derivados de este recuerdo que no se muestran acá.' in dialog, dialog
                        assert 'No borra la conversación de origen' in dialog and 'Los viernes almuerzo' not in dialog
                        # R6: always said, whatever the visible cascade: deleted at apply time, no count reported.
                        assert CASCADE in dialog and await page.evaluate("Boolean(document.querySelector('#my-memory-delete-cascade'))"), dialog
                        audits['mis-recuerdos-eliminar'] = await audit_view(page, evidence, 'mis-recuerdos-eliminar', (
                            ('#my-memory-confirm-dialog [data-key=confirm-cancel]', None), ('#my-memory-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await click('#my-memory-confirm-dialog [data-key="confirm-cancel"]')
                        await page.wait_for("!document.querySelector('#my-memory-confirm-dialog').open")
                        assert count('DELETE', f'/api/memory/mine/facts/{a_parent}') == 0
                        await press(key(a_parent + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"!document.querySelector({card(a_parent)}) && document.querySelector('#my-memory-live').textContent.startsWith('Recuerdo eliminado')")
                        assert await get(a_parent) is None and await get(hidden_child) is None
                        checks.append('delete own: the dialog says it is irreversible, discloses hidden derived data without showing it and the erasure limits; cancel sends nothing; accept deletes the memory and its derived data')

                        # Lost DELETE response after the commit: resolved only by a later read.
                        await rule('DELETE', f'/api/memory/mine/facts/{a_lost}', 'drop')
                        await press(key(a_lost + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"document.querySelector('#my-memory-live').textContent==='Comprobado: ya no figura entre tus recuerdos.' && !document.querySelector({card(a_lost)})")
                        assert await get(a_lost) is None and count('DELETE', f'/api/memory/mine/facts/{a_lost}') == 1
                        checks.append('lost DELETE response after commit: resolved by a read that started after the loss; one DELETE on the server')

                        # F2: the server is still working when the client gives up: the read must not say anything was not applied.
                        gate = asyncio.Event()
                        gates[('DELETE', f'/api/memory/mine/facts/{a_held}')] = gate
                        await rule('DELETE', f'/api/memory/mine/facts/{a_held}', 'lose')
                        await press(key(a_held + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"{notice(a_held)}==='Todavía figura entre tus recuerdos; puede que el cambio no se haya aplicado.'")
                        stored = await page.evaluate(f"JSON.parse(sessionStorage.getItem('kza:my-memory-op:{a_held}'))")
                        assert stored['subject'] == 'synthetic-home-a' and stored['op'] == 'delete' and set(stored) <= {'op', 'subject', 'sent_at', 'state', 'reason'}
                        assert not any('sintético' in value for value in await page.evaluate('Object.values(sessionStorage)'))
                        assert 'no se eliminó' not in await page.evaluate("document.querySelector('#my-memory-view').innerText")
                        gate.set()
                        for _ in range(100):
                            if await get(a_held) is None:
                                break
                            await asyncio.sleep(0.05)
                        assert await get(a_held) is None
                        await click('#my-memory-refresh')
                        await page.wait_for(f"document.querySelector('#my-memory-live').textContent==='Comprobado: ya no figura entre tus recuerdos.' && !document.querySelector({card(a_held)})")
                        assert count('DELETE', f'/api/memory/mine/facts/{a_held}') == 1
                        del gates[('DELETE', f'/api/memory/mine/facts/{a_held}')]
                        checks.append('F2: a held server write while the client lost the request stays unresolved with honest copy and a content-free marker; the later commit is reported only after a new read')

                        # Leaving the view never aborts a write.
                        await rule('POST', f'/api/memory/mine/facts/{a_leave}/return', 'hold')
                        await page.evaluate('window.__qaHeld=null')
                        await press(key(a_leave + ':return'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for('window.__qaHeld')
                        await click('#app-nav [data-view=home]')
                        await page.wait_for("!document.querySelector('#home-view').hidden && !document.querySelector('#my-memory-view').children.length")
                        await page.evaluate('window.__qaRelease()')
                        await click('#app-nav [data-view=mymemory]')
                        await page.wait_for(f"!document.querySelector('#my-memory-view').hidden && document.querySelector({card(a_hidden)}) && !document.querySelector({card(a_leave)})")
                        assert await owner_of(a_leave) == (None, 'owner_returned') and count('POST', f'/api/memory/mine/facts/{a_leave}/return') == 1
                        checks.append('navigating away mid-return keeps the write; exactly one POST; the private view was emptied while away')

                        # F8: hiding the tab with a dialog open drops all fact text and sends nothing.
                        await press(key(a_hidden + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await set_hidden(True)
                        await page.wait_for("!document.querySelector('#my-memory-confirm-dialog').open")
                        assert 'Recuerdo sintético' not in await dom_text() and 'Leo novelas' not in await dom_text()
                        await set_hidden(False)
                        await page.wait_for(f"document.querySelector({card(a_hidden)})")
                        assert count('DELETE', f'/api/memory/mine/facts/{a_hidden}') == 0 and await get(a_hidden) is not None
                        checks.append('F8: a hidden tab closes and empties the dialog; no memory text left in the DOM; nothing sent')

                        # R3: a marker left by an earlier page (maybe another login of the same subject) is never
                        # 'Comprobado' and never resent with one click; the person may act again through the dialog.
                        await page.evaluate(f"sessionStorage.setItem('kza:my-memory-op:{a_hidden}', JSON.stringify({{op:'delete',subject:'synthetic-home-a',sent_at:1,state:'unresolved'}}))")
                        await page.command('Page.reload')
                        await page.wait_for(f"{notice(a_hidden)}==={json.dumps(RESTORED_STILL)}")
                        assert await page.evaluate(f"!document.querySelector({json.dumps(key(a_hidden + ':retry'))}) && Boolean(document.querySelector({json.dumps(key(a_hidden + ':delete'))}))")
                        assert 'Comprobado' not in await page.evaluate("document.querySelector('#my-memory-view').innerText")
                        assert await page.evaluate(f"sessionStorage.getItem('kza:my-memory-op:{a_hidden}')===null")
                        assert count('DELETE', f'/api/memory/mine/facts/{a_hidden}') == 0 and await get(a_hidden) is not None
                        checks.append('R3: a marker restored after a reload is settled by the read without Comprobado or a one-click retry; nothing is resent and the dialog actions stay available')

                        # Accessibility and reflow of the list, plus a 200% text pass.
                        audits['mis-recuerdos'] = await audit_view(page, evidence, 'mis-recuerdos', (
                            ('#app-nav [data-view=mymemory]', None), ('#my-memory-refresh', None), (key(a_hidden + ':return'), None),
                            (key(a_hidden + ':delete'), None)), screenshot, widths=WIDTHS)
                        zoom = {}
                        for width in ZOOM_WIDTHS:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
                            await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                            enlarged = await page.evaluate(TEXT_ZOOM + "(['#my-memory-view','#rail-nav'])")
                            measured = await page.evaluate(TEXT_ZOOM_CHECK + "(['#my-memory-view'], " + json.dumps(['#my-memory-refresh', key(a_hidden + ':return'), key(a_hidden + ':delete')]) + ')')
                            await screenshot(page, evidence / f'mis-recuerdos-text200-{width}.png')
                            problems = ([f'document scrolls horizontally {measured["scrollWidth"]}>{measured["innerWidth"]}'] if measured['scrollWidth'] > measured['innerWidth'] else []) \
                                + [f'overflows: {item}' for item in measured['offenders']] + [f'clipped: {item}' for item in measured['clipped']] + measured['blocked']
                            zoom[str(width)] = {'elements_enlarged': enlarged, 'measurement': measured, 'findings': problems, 'screenshot': f'mis-recuerdos-text200-{width}.png'}
                            await page.command('Page.reload')
                            await page.wait_for(f"!document.querySelector('#my-memory-view').hidden && document.querySelector({card(a_hidden)})")
                        (evidence / 'text-zoom-mis-recuerdos.json').write_text(json.dumps({'method': 'computed font-size and px line-height of every element in #my-memory-view and #rail-nav doubled (text-only 200%), viewport at 320 and 390 CSS px', 'widths': zoom}, indent=2, ensure_ascii=False) + '\n')
                        zoom_findings = [f'{width}px: {item}' for width, value in zoom.items() for item in value['findings']]
                        assert not zoom_findings, zoom_findings
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                        checks.append('audit_view at 320/390/768/1024/1280/1440 (list, No es mío and delete dialogs) and a 200% text pass at 320/390 without overflow, clipping or unreachable controls')

                        # Pagination: 105 more memories for A; the read walks ?after= and 'Mostrar más' reveals the rest.
                        for index in range(105):
                            await own(f'fill{index}', f'Recuerdo sintético de relleno de A número {index}.', person_a.user_id)
                        marker = len(requests)
                        await click('#my-memory-refresh')
                        await page.wait_for("document.querySelectorAll('#my-memory-facts .memory-card').length===100 && (document.querySelector('#my-memory-view .memory-more button')?.textContent||'').startsWith('Mostrar más recuerdos (quedan 7)')")
                        walked = [item for item in requests[marker:] if item['path'] == '/api/memory/mine']
                        assert [item['query'] for item in walked] == [[], ['after']], walked
                        await page.evaluate("document.querySelector('#my-memory-view .memory-more button').focus()")
                        await press('#my-memory-view .memory-more button')
                        await page.wait_for(f"document.querySelectorAll('#my-memory-facts .memory-card').length===107 && document.querySelector({card(a_hidden)}) && document.querySelector('#my-memory-facts').contains(document.activeElement)")
                        checks.append('pagination: /api/memory/mine is walked through its after cursor (107 memories, 2 pages); Mostrar más reveals the rest and moves focus to the first new card')

                        # Account switch while hidden with an unresolved delete: whoami first, nothing resent under B, A's content gone.
                        gate = asyncio.Event()
                        gates[('DELETE', f'/api/memory/mine/facts/{a_hidden}')] = gate
                        await rule('DELETE', f'/api/memory/mine/facts/{a_hidden}', 'lose')
                        await press(key(a_hidden + ':delete'))
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        await click('#my-memory-confirm-dialog [data-key="confirm-accept"]')
                        await page.wait_for(f"{notice(a_hidden)}.startsWith('Todavía figura')")
                        await set_hidden(True)
                        await install_login_session(page, base, home_b.token)
                        marker = len(requests)
                        await set_hidden(False)
                        await page.wait_for(f"document.querySelector('#session-label').textContent==='Acceso de hogar' && document.querySelector({card(b_fact)})")
                        after = requests[marker:]
                        assert after and after[0]['path'] == '/api/admin/auth/whoami', after[:4]
                        assert not any(item['method'] in ('POST', 'DELETE') for item in after), after
                        assert await page.evaluate(f"sessionStorage.getItem('kza:my-memory-op:{a_hidden}')===null")
                        body = await dom_text()
                        assert 'Recuerdo sintético de A' not in body and 'Recuerdo sintético de relleno de A' not in body, "A's memories are gone under B"
                        assert await page.evaluate("[...document.querySelectorAll('#my-memory-view [data-my-memory-id]')].map(n=>n.dataset.myMemoryId)") == [b_fact]
                        gate.set()
                        await asyncio.sleep(0.3)
                        del gates[('DELETE', f'/api/memory/mine/facts/{a_hidden}')]
                        assert count('DELETE', f'/api/memory/mine/facts/{a_hidden}') == 1, "never resent under B"
                        checks.append("F5/F11: an account switch in another tab while hidden: whoami first, re-bootstrap as B, A's unresolved delete is never resent and its key is wiped; only B's memory in the DOM")
                        await logout()
                        assert await page.evaluate("!Object.keys(sessionStorage).some(k=>k.startsWith('kza:my-memory-op:'))")
                        assert await page.evaluate("!document.querySelector('#my-memory-view').children.length")

                        # Negative: household account without a person (voice profile linked to A grants nothing).
                        await login(page, shared.token, 'Acceso de hogar')
                        assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(n=>n.dataset.view).join(',')==='home,actions,routines'")
                        marker = len(requests)
                        await page.evaluate("location.hash='#mis-recuerdos'")
                        await page.command('Page.reload')
                        await page.wait_for("document.querySelector('#session-label').textContent==='Acceso de hogar' && !document.querySelector('#home-view').hidden")
                        await asyncio.sleep(0.3)
                        assert not any(item['path'].startswith('/api/memory') for item in requests[marker:]), requests[marker:]
                        assert await page.evaluate("document.querySelector('#my-memory-view').hidden && !document.querySelector('#my-memory-view').children.length")
                        for path in ('/api/memory/mine', '/api/memory', '/api/memory/returned'):
                            assert await page.evaluate(f"fetch({json.dumps(path)}).then(r=>r.status)") == 403, path
                        assert await page.evaluate(f"fetch('/api/memory/mine/facts/{a_hidden}',{{method:'DELETE'}}).then(r=>r.status)") == 403
                        assert 'sintético de A' not in await dom_text()
                        assert await voices.acall(person_link, voices, profile['id']) == (person_a.user_id, 1)
                        assert await get(a_hidden) is None or await owner_of(a_hidden) == (person_a.user_id, 'assigned')
                        checks.append('household without a person, with a reviewed voice profile linked to A in the voice ledger: no Mis recuerdos entry, #mis-recuerdos falls back to Hogar with no memory request; /api/memory/mine, operator routes and a delete answer 403')
                        await logout()

                        # Negative: evaluator and operator never get Mis recuerdos.
                        await login(page, evaluator.token, 'Evaluación temporal')
                        await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                        assert await page.evaluate("!document.querySelector('[data-view=mymemory]') && !document.querySelector('[data-view=memory]')")
                        assert await page.evaluate("fetch('/api/memory/mine').then(r=>r.status)") == 403
                        await logout()
                        await login(page, operator.token, 'Administración')
                        assert await page.evaluate("!document.querySelector('[data-view=mymemory]') && !!document.querySelector('[data-view=memory]')")
                        assert await page.evaluate("fetch('/api/memory/mine').then(r=>r.json().then(b=>[r.status,b.detail]))") == [403, 'dashboard_permission_denied']
                        await logout()
                        checks.append('evaluator: no Mis recuerdos or Memoria entry, /api/memory/mine 403; operator: Memoria but no Mis recuerdos, /api/memory/mine 403 dashboard_permission_denied')
                        assert page.errors == [], page.errors

                        result = {'passed': True, 'synthetic_fixture': True, 'actual_chromium_http_login': True, 'real_memory_services': True,
                                  'real_voice_link': True,
                                  'roles': ['home-person-a', 'home-person-b', 'home-without-person', 'evaluator', 'operator'],
                                  'widths': list(WIDTHS), 'text_zoom_widths': list(ZOOM_WIDTHS),
                                  'checks': checks, 'a11y_evidence': [f'a11y-{view}.json' for view in audits] + ['text-zoom-mis-recuerdos.json'],
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
