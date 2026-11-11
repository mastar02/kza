"""Real Chromium, real logins, one shared privacy ledger for NEXO 'Memoria por voz' (U3b-4a-3).

The memory routes and the voice registry share one PrivacyServices ledger, as in production
(services.manager.long_term.ledger is privacy.ledger). The settings flag is on for this run;
the G1 switch starts off and the operator turns it on through Voces. Verification outcomes are
injected through the ledger (the scorer and the phrase belong to U3b-4b). Evidence never
serializes a credential, a cookie, a request body, a phrase or fact text: only methods, paths,
query parameter names, statuses and counts.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import aiohttp
import uvicorn
from starlette.responses import JSONResponse
from websockets.asyncio.client import connect

from src.audio.microphone_status import (
    BindingState, CaptureSnapshot, CaptureState, ConnectionState, MicSnapshot, MicrophonesSnapshot,
)
from src.dashboard import memory_api, memory_voice_api
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryOwnerActor, MemoryVoiceConsentActor
from src.memory.privacy_services import PrivacyServices
from src.memory.services import build_memory_services
from src.memory.voice_attribution import ProfileScore, ScoringSnapshot, VoiceAttributionCapability, VoiceProposalSettings
from src.rooms.room_context import RoomConfig
from src.users.user_manager import PermissionLevel
from src.users.voice_ledger import VoiceReviewer
from src.users.voice_registry import build_voice_registry
from tests.frontend.browser_nexo_home import login, screenshot
from tests.frontend.browser_nexo_memory import INSTRUMENT, TEXT_ZOOM, TEXT_ZOOM_CHECK
from tests.frontend.browser_support import Page, audit_view, browser_binary, browser_profile_root, evidence_directory
from tests.integration.test_typed_home_operations import typed_graph
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.voice_registry import ACTOR, seed as voice_seed

EVIDENCE = evidence_directory('KZA_NEXO_MY_VOICE_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/u3b-4a-3-dev/browser-runs')
WIDTHS = (320, 390, 768, 1024, 1280, 1440)
ZOOM_WIDTHS = WIDTHS  # Z: 200 % text at every width
CAP = VoiceAttributionCapability('synthetic-browser')
# TEXT_ZOOM doubles the computed size, so each width starts from the stylesheet sizes again (otherwise it compounds).
TEXT_ZOOM_RESET = r"""((scopes) => { for (const root of scopes.map(selector => document.querySelector(selector)).filter(Boolean))
  for (const el of [root, ...root.querySelectorAll('*')]) { el.style.fontSize = ''; el.style.lineHeight = ''; } })"""
ROOM_UNAVAILABLE = 'Esa habitación no está escuchando ahora. Elegí otra o probá más tarde. Cuenta como uno de los intentos de hoy.'
NO_PHRASE = 'La frase para decir en voz alta todavía no está disponible en esta versión. La ventana se cierra sola.'


def mic_snapshot() -> MicrophonesSnapshot:
    """A fresh snapshot where only 'living' captures (the kitchen has no microphone)."""
    mic = MicSnapshot('synthetic-mic', 'living', 'Sala de estar', None, None, None, None, None, None, None,
                      ConnectionState.CONNECTED, BindingState.MATCHED, CaptureSnapshot(CaptureState.RECEIVING))
    return MicrophonesSnapshot(time.time(), 60.0, False, (mic,), ('kitchen',))


async def main() -> None:
    evidence = EVIDENCE
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    requests: list[dict] = []
    checks: list[str] = []
    # S: 'lose' commits the write and then drops its answer (502); 'down' refuses a read (503) without reaching the route.
    faults: dict[tuple[str, str], str] = {}

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
    ha, actions = typed_graph.__wrapped__()
    with tempfile.TemporaryDirectory(prefix='kza-nexo-my-voice-', dir=browser_data) as temporary:
        directory = Path(temporary)
        async with AsyncExitStack() as stack:
            clock = Clock(time.time())
            privacy = PrivacyServices(str(directory / 'privacy.db'), clock=clock, voice_settings=VoiceProposalSettings(enabled=True))
            stack.push_async_callback(privacy.aclose)
            await privacy.acall(privacy.initialize)
            privacy.start()
            owner = ErasureCoordinator(privacy)
            stack.push_async_callback(owner.aclose)
            bus = await build_live_services({}, privacy=privacy, owner=owner, shutdown=stack)
            voices = await build_voice_registry(privacy=privacy, owner=owner, shutdown=stack)
            client = MemoryClient(directory / 'chroma')
            stack.push_async_callback(asyncio.to_thread, client.close)
            services = await build_memory_services({'memory': {'privacy_db_path': str(directory / 'privacy.db'), 'chroma_path': str(directory / 'chroma'),
                                                    'preferences_path': str(directory / 'preferences.json')}},
                                                   shutdown=stack, chroma_client=client, privacy=privacy)
            manager, memory = services.manager, services.manager.long_term
            ledger = memory.ledger
            assert ledger is privacy.ledger and voices.ledger.people.ledger is ledger, 'one ledger, as in production'
            authority = None
            users = await voices.load_accounts(str(directory / 'legacy-users.json'),
                                               retirement_proof=lambda identities: authority.users_retired(identities) and voices.ledger.people.user_links_cleared(identities))
            people = {}
            for key in 'abc':
                person, _ = await voices.acall(users.add_user, f'Persona sintética {key.upper()}', PermissionLevel.ADULT)
                assert person is not None
                people[key] = person.user_id
            authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=900, account_provider=lambda: users.active_accounts(),
                                           expected_token=lambda: 'synthetic-my-voice-root')
            await authority.initialize()
            stack.push_async_callback(authority.aclose)
            root = authority.resolve('synthetic-my-voice-root')
            expires = time.time() + 1500
            homes = {key: await authority.issue_credential(root, subject=f'synthetic-home-{key}', role=DashboardRole.HOME, expires_at=expires, user_id=people[key])
                     for key in 'abc'}
            operator = await authority.issue_credential(root, subject='synthetic-operator', role=DashboardRole.OPERATOR, expires_at=expires)

            # Reviewed voice profiles (real voice ledger) and person links (real link CAS).
            linker_session = await authority.login(operator.token)
            linker = VoiceReviewer.from_dashboard(linker_session.lease.principal,
                                                  lambda: linker_session.lease.current(frozenset({'voices:review', 'admin:manage'})), person_authority=authority)
            profiles = {}
            for key, person, name in (('a', people['a'], 'Voz sintética de A'), ('b', people['b'], 'Voz sintética de B'),
                                      ('c1', people['c'], 'Voz sintética de C uno'), ('c2', people['c'], 'Voz sintética de C dos'), ('x', None, 'Voz sintética sin persona')):
                _sources, identities = await voice_seed(voices, clock, ttl=7200)
                profile = await voices.acall(voices.ledger.confirm, identities, name, str(uuid4()), ACTOR)
                profiles[key] = profile['id']
                if person is not None:
                    await voices.acall(voices.ledger.people.set_person, profile['id'], person, 1, 0, linker)

            def scores(top):
                return lambda db: ScoringSnapshot('enc_synthetic', tuple(
                    ProfileScore(row[0], row[1], .95 if row[0] == top else .3) for row in db.execute(
                        'SELECT id, revision FROM voice_profiles WHERE deleted_at IS NULL')), True, 3.0)

            scores_for = scores

            allow = {'retired': authority.memory_owner_retired, 'active': authority.memory_owner_active}

            async def proposal(text, top):
                source = manager.source_for_capture(str(uuid4()), clock.now, 'living')
                fact = await memory.acall(memory.propose, text, (source,))
                decision = await memory.acall(ledger.attribute_voice_proposal, fact['id'], scores_for(top), CAP, **allow)
                assert decision.owner, decision
                return fact['id']

            async def verify(person, top):
                window = ledger.voice_windows.current(person)
                assert window is not None, 'a window is open'
                return await memory.acall(ledger.verify_voice_consent, window.binding, scores_for(top), CAP, **allow)

            def table(sql, *args):
                return [tuple(row) for row in ledger._db.execute(sql, args).fetchall()]

            async def rows(sql, *args):
                return await memory.acall(table, sql, *args)

            rooms = {'living': RoomConfig('living', 'Sala de estar', 'la sala', default_light='light.test_lamp'),
                     'kitchen': RoomConfig('kitchen', 'Cocina', 'la cocina', default_climate='climate.test_ac')}
            dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms, memory_manager=manager,
                                     user_manager=users, voice_registry=voices, event_bus=bus, microphone_snapshot_provider=mic_snapshot)

            @dashboard.app.middleware('http')
            async def record_request(request, call_next):
                fault = faults.get((request.method, request.url.path))
                if fault == 'down':
                    response = JSONResponse({'detail': 'synthetic_unavailable'}, status_code=503)
                else:
                    response = await call_next(request)
                    if fault == 'lose':
                        response = JSONResponse({'detail': 'synthetic_lost_answer'}, status_code=502)
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
                            await page.wait_for(f"(e=>e&&!e.disabled&&e.getAttribute('aria-disabled')!=='true')(document.querySelector({json.dumps(selector)}))")
                            await click(selector)

                        def text_of(selector: str) -> str:
                            return f"(document.querySelector({json.dumps(selector)})?.textContent||'')"

                        def count(method: str, path: str) -> int:
                            return sum(1 for item in requests if item['method'] == method and item['path'] == path)

                        async def logout() -> None:
                            await click('#logout')
                            await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                        async def open_my_memory() -> None:
                            await click('#app-nav [data-view=mymemory]')
                            await page.wait_for("!document.querySelector('#my-memory-view').hidden && document.querySelector('#my-voice-status')?.textContent.length>0")

                        async def reopen_my_memory() -> None:
                            await click('#app-nav [data-view=home]')
                            await page.wait_for("!document.querySelector('#home-view').hidden")
                            await open_my_memory()

                        async def accept(dialog: str, key: str) -> None:
                            await page.wait_for(f"document.querySelector('{dialog}').open")
                            await click(f'{dialog} [data-key="{key}"]')
                            await page.wait_for(f"!document.querySelector('{dialog}').open")

                        async def text_zoom(view: str, scopes: list[str], keys: list[str]) -> list[str]:
                            found = []
                            for width in ZOOM_WIDTHS:
                                await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 844, 'deviceScaleFactor': 1, 'mobile': width < 768})
                                await page.evaluate(TEXT_ZOOM_RESET + '(' + json.dumps(scopes + ['#rail-nav']) + ')')
                                await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                                enlarged = await page.evaluate(TEXT_ZOOM + '(' + json.dumps(scopes + ['#rail-nav']) + ')')
                                measured = await page.evaluate(TEXT_ZOOM_CHECK + '(' + json.dumps(scopes) + ', ' + json.dumps(keys) + ')')
                                await screenshot(page, evidence / f'{view}-text200-{width}.png')
                                problems = ([f'document scrolls horizontally {measured["scrollWidth"]}>{measured["innerWidth"]}'] if measured['scrollWidth'] > measured['innerWidth'] else []) \
                                    + [f'overflows: {item}' for item in measured['offenders']] + [f'clipped: {item}' for item in measured['clipped']] + measured['blocked']
                                (evidence / f'text-zoom-{view}-{width}.json').write_text(json.dumps({'elements_enlarged': enlarged, 'measurement': measured, 'findings': problems}, indent=2, ensure_ascii=False) + '\n')
                                found += [f'{view} {width}px: {item}' for item in problems]
                            await page.evaluate(TEXT_ZOOM_RESET + '(' + json.dumps(scopes + ['#rail-nav']) + ')')
                            await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                            return found

                        audits, zoom_findings = {}, []

                        # Operator: the G1 switch through Voces (settings flag on, switch row off).
                        await login(page, operator.token, 'Administración')
                        await click('#app-nav [data-view=voices]')
                        await page.wait_for("!document.querySelector('#voices-view').hidden && " + text_of('#voice-attribution-switch') + ".includes('Interruptor apagado')")
                        panel = await page.evaluate("document.querySelector('#voice-attribution').innerText")
                        assert memory_voice_api.VOICES_DISCLOSURE in panel and 'Todavía nadie pidió inscribirse.' in panel, panel
                        await press('#voice-attribution [data-key=voice-switch]')
                        await accept('#voice-attribution-confirm-dialog', 'voice-admin-accept')
                        await page.wait_for(text_of('#voice-attribution-live') + "==='Memoria por voz encendida.'")
                        assert (await memory.acall(ledger.voice_switch)).effective is True
                        checks.append('operator: Voces shows the voice disclosure and the G1 switch; turning it on goes through a dialog and the real route')
                        await logout()

                        # Person A: opt-in, a refused room, a window without a phrase, then an injected pass.
                        await login(page, homes['a'].token, 'Acceso de hogar')
                        marker = len(requests)
                        await open_my_memory()
                        opened = [item['path'] for item in requests[marker:]]
                        assert opened.index('/api/admin/auth/whoami') < opened.index('/api/memory/mine/voice'), opened
                        await page.wait_for(text_of('#my-voice-status') + ".includes('No estás inscripto')")
                        disclosure = await page.evaluate("document.querySelector('#my-voice-disclosure').textContent")
                        assert '48 horas en la transcripción ambiental' in disclosure and 'evaluador autorizado' in disclosure and 'Tu voz no te da permisos ni acceso' in disclosure
                        await press('#my-memory-view [data-key=voice-request]')
                        await page.wait_for("document.querySelector('#my-voice-confirm-dialog').open && document.activeElement?.dataset.key==='voice-confirm-cancel'")
                        assert memory_voice_api.OPT_IN_DISCLOSURE in await page.evaluate("document.querySelector('#my-voice-confirm-dialog').innerText")
                        audits['voz-inscripcion'] = await audit_view(page, evidence, 'voz-inscripcion', (
                            ('#my-voice-confirm-dialog [data-key=voice-confirm-cancel]', None), ('#my-voice-confirm-dialog [data-key=voice-confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await accept('#my-voice-confirm-dialog', 'voice-confirm-accept')
                        await page.wait_for(text_of('#my-voice-status') + ".startsWith('Inscripción pedida') && document.querySelector('#my-voice-room')")
                        options = await page.evaluate("[...document.querySelectorAll('#my-voice-room option')].map(o=>[o.value,o.textContent])")
                        assert options == [['living', 'Sala de estar'], ['kitchen', 'Cocina']], options
                        await page.evaluate("(s=>{s.value='kitchen';s.dispatchEvent(new Event('change'))})(document.querySelector('#my-voice-room'))")
                        await press('#my-memory-view [data-key=voice-window]')
                        await page.wait_for(text_of('#my-voice-notice') + '===' + json.dumps(ROOM_UNAVAILABLE))
                        assert len(await rows('SELECT 1 FROM voice_verification_attempts WHERE person_id=?', people['a'])) == 1, 'the room refusal counts'
                        await page.evaluate("(s=>{s.value='living';s.dispatchEvent(new Event('change'))})(document.querySelector('#my-voice-room'))")
                        await press('#my-memory-view [data-key=voice-window]')
                        await page.wait_for(text_of('#my-voice-window') + ".includes('Ventana abierta en Sala de estar') && " + text_of('#my-voice-window') + '.includes(' + json.dumps(NO_PHRASE) + ')')
                        stored = json.dumps(await page.evaluate('Object.fromEntries(Object.entries(sessionStorage))'))
                        assert 'living' not in stored and 'Sala' not in stored, 'no room or phrase in sessionStorage'
                        assert (await verify(people['a'], profiles['a'])).outcome.value == 'passed'
                        await reopen_my_memory()
                        await page.wait_for(text_of('#my-voice-status') + "==='Inscripción activa.' && " + text_of('#my-voice-message') + '.startsWith(' + json.dumps(memory_voice_api.STATE_MESSAGES['active']) + ')')
                        assert await page.evaluate(text_of('#my-voice-notice') + "===''"), 'no stale window notice once the window is gone'
                        checks.append('person A: whoami before the status; the opt-in dialog carries the full disclosure (48 h ambient transcript, evaluators, no permissions); request, a refused room counted against the daily limit, a window without a phrase (R1), no room or phrase in sessionStorage; an injected pass shows the active message')

                        # Pending voice proposals: content-free entries with exactly «Descartar» and «No es mío».
                        discard = await proposal('Recuerdo de voz sintético para descartar', profiles['a'])
                        give_back = await proposal('Recuerdo de voz sintético que no es mío', profiles['a'])
                        await reopen_my_memory()
                        await page.wait_for(f"document.querySelector('[data-my-memory-id=\"{discard}\"]') && document.querySelector('[data-my-memory-id=\"{give_back}\"]')")
                        body = await page.evaluate('document.body.innerText + document.body.textContent')
                        assert 'Recuerdo de voz sintético' not in body, 'no content before confirmation (Q5)'
                        labels = await page.evaluate(f"[...document.querySelectorAll('[data-my-memory-id=\"{discard}\"] button')].map(b=>b.textContent)")
                        assert labels == ['Descartar', 'No es mío'], labels
                        card = await page.evaluate(f"document.querySelector('[data-my-memory-id=\"{discard}\"]').innerText")
                        assert 'Propuesta de voz pendiente de revisión' in card and 'hora aproximada' in card and ':00' in card, card
                        audits['mis-recuerdos-voz'] = await audit_view(page, evidence, 'mis-recuerdos-voz', (
                            ('#app-nav [data-view=mymemory]', None), ('#my-memory-view [data-key=voice-revoke]', None),
                            (f'#my-memory-view [data-key="{discard}:discard"]', None), (f'#my-memory-view [data-key="{discard}:return"]', None)), screenshot, widths=WIDTHS)
                        zoom_findings += await text_zoom('mis-recuerdos-voz', ['#my-memory-view'], ['#my-memory-view [data-key=voice-revoke]', f'#my-memory-view [data-key="{discard}:discard"]'])
                        await reopen_my_memory()
                        await page.wait_for(f"document.querySelector('[data-my-memory-id=\"{discard}\"]')")
                        await press(f'#my-memory-view [data-key="{discard}:discard"]')
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        assert 'Se va a borrar esta propuesta. La administración no la va a ver.' in await page.evaluate("document.querySelector('#my-memory-confirm-dialog').innerText")
                        await accept('#my-memory-confirm-dialog', 'confirm-accept')
                        await page.wait_for(f"!document.querySelector('[data-my-memory-id=\"{discard}\"]') && " + text_of('#my-memory-live') + "==='Propuesta descartada.'")
                        assert await rows('SELECT state, content FROM facts WHERE id=?', discard) == [('deleted', None)]
                        assert await rows('SELECT COUNT(*) FROM voice_attribution_events') == [(0,)], 'a discard counts nothing (Q6)'
                        await press(f'#my-memory-view [data-key="{give_back}:return"]')
                        await page.wait_for("document.querySelector('#my-memory-confirm-dialog').open")
                        dialog = await page.evaluate("document.querySelector('#my-memory-confirm-dialog').innerText")
                        assert memory_api.VOICE_RETURN_NOTICE in dialog and 'Si no fuiste vos' in dialog, dialog
                        audits['voz-no-es-mio'] = await audit_view(page, evidence, 'voz-no-es-mio', (
                            ('#my-memory-confirm-dialog [data-key=confirm-cancel]', None), ('#my-memory-confirm-dialog [data-key=confirm-accept]', None)), screenshot, widths=WIDTHS)
                        await accept('#my-memory-confirm-dialog', 'confirm-accept')
                        await page.wait_for(text_of('#my-memory-return-notice') + '===' + json.dumps(memory_api.VOICE_RETURN_NOTICE))
                        assert await rows('SELECT COUNT(*) FROM voice_attribution_events') == [(1,)]
                        checks.append('pending entries: no content in the DOM before confirmation; exactly «Descartar» and «No es mío»; hour-rounded times; «Descartar» deletes with no event; «No es mío» states and then shows the N5 notice and counts one event')
                        # A second 'No es mío' on A's profile (through the ledger) suspends it for the resume flow below.
                        second = await proposal('Otro recuerdo de voz sintético', profiles['a'])
                        assert await memory.acall(ledger.return_owned, second, people['a'], MemoryOwnerActor('synthetic-home-a', people['a']))
                        assert await rows('SELECT profile_id FROM voice_attribution_suspensions') == [(profiles['a'],)]
                        await logout()

                        # Person C has two linked voices: the re-enrolment notice.
                        await login(page, homes['c'].token, 'Acceso de hogar')
                        await open_my_memory()
                        await press('#my-memory-view [data-key=voice-request]')
                        await accept('#my-voice-confirm-dialog', 'voice-confirm-accept')
                        await page.wait_for(text_of('#my-voice-notice') + '===' + json.dumps(memory_voice_api.REENROLMENT_NOTICE))
                        checks.append('person C with two linked voices: the request is refused with the re-enrolment notice')
                        await logout()

                        # Person B, S: the request commits but its answer is lost and the status read fails, so the
                        # marker stays unresolved; a reload (pagehide, wipe false) restores it and the read settles it
                        # without «Comprobado» and without resending.
                        await login(page, homes['b'].token, 'Acceso de hogar')
                        await open_my_memory()
                        faults.update({('POST', '/api/memory/mine/voice/consent'): 'lose', ('GET', '/api/memory/mine/voice'): 'down'})
                        await press('#my-memory-view [data-key=voice-request]')
                        await accept('#my-voice-confirm-dialog', 'voice-confirm-accept')
                        await page.wait_for(text_of('#my-voice-notice') + ".startsWith('Sin confirmar.')")
                        stored = await page.evaluate("sessionStorage.getItem('kza:my-voice-op:consent')")
                        assert stored and sorted(json.loads(stored)) == ['expected_revision', 'op', 'sent_at', 'subject'], 'a content-free marker'
                        faults.clear()
                        posts = count('POST', '/api/memory/mine/voice/consent')
                        await page.command('Page.reload')
                        await page.wait_for(text_of('#my-voice-notice') + '===' + json.dumps('Tu estado cambió, pero no podemos confirmar que haya sido por este pedido. Revisalo.'))
                        assert 'Comprobado' not in await page.evaluate("document.querySelector('#my-memory-view').innerText")
                        assert await page.evaluate("sessionStorage.getItem('kza:my-voice-op:consent')===null and sessionStorage.getItem('kza:my-voice-op-ended')===null".replace(' and ', '&&'))
                        assert count('POST', '/api/memory/mine/voice/consent') == posts, 'never resent'
                        assert await rows('SELECT state FROM voice_attribution_consents WHERE person_id=?', people['b']) == [('pending_verification',)]
                        await page.wait_for("document.querySelector('#my-voice-room')")
                        checks.append('S (person): a request whose answer was lost and whose status read failed leaves a content-free marker; after a reload the restored marker is settled by the read without «Comprobado», never resent, then dropped')
                        # Person B: three failures (injected) lock the verification.
                        actor_b = MemoryVoiceConsentActor('synthetic-home-b', people['b'])
                        for _ in range(3):
                            assert await memory.acall(ledger.open_voice_window, actor_b, 1, 'living', lambda room: True) is not None
                            await verify(people['b'], profiles['x'])
                        await reopen_my_memory()
                        await page.wait_for(text_of('#my-voice-status') + "==='Verificación bloqueada.' && " + text_of('#my-voice-message') + '===' + json.dumps(memory_voice_api.STATE_MESSAGES['verification_locked']))
                        locked = await page.evaluate("[...document.querySelectorAll('#my-memory-view .memory-voice button')].filter(b=>!b.closest('[hidden]')).map(b=>b.textContent)")
                        assert locked == ['Darme de baja…'], locked
                        checks.append('person B: three failed verifications lock it; the view shows the locked message and only «Darme de baja»')
                        await logout()

                        # Operator: consents with the verified profile, multi-profile notice, Voces notes, reopen and resume.
                        await login(page, operator.token, 'Administración')
                        await click('#app-nav [data-view=voices]')
                        await page.wait_for(f"document.querySelector('[data-voice-consent=\"{people['b']}\"]') && document.querySelector('[data-voice-suspended=\"{profiles['a']}\"]')")
                        panel = await page.evaluate("document.querySelector('#voice-attribution').innerText")
                        assert 'Persona sintética A' in panel and 'Verificada con «Voz sintética de A»' in panel, panel
                        assert memory_voice_api.MULTI_PROFILE_NOTICE in panel and 'Persona sintética C' in panel, panel
                        await page.wait_for(f"(document.querySelector('[data-voice-profile=\"{profiles['a']}\"]')?.textContent||'').includes('Verificada por Persona sintética A')")
                        assert 'Propuestas por voz en pausa' in await page.evaluate(f"document.querySelector('[data-voice-profile=\"{profiles['a']}\"]').textContent")
                        assert not any(word in panel.lower() for word in ('puntaje', 'score', 'mesa verde'))
                        audits['voces-memoria-por-voz'] = await audit_view(page, evidence, 'voces-memoria-por-voz', (
                            ('#voice-attribution [data-key=voice-switch]', None), (f'#voice-attribution [data-key="reopen:{people["b"]}"]', None),
                            (f'#voice-attribution [data-key="resume:{profiles["a"]}"]', None)), screenshot, widths=WIDTHS)
                        zoom_findings += await text_zoom('voces-memoria-por-voz', ['#voice-attribution'], ['#voice-attribution [data-key=voice-switch]', f'#voice-attribution [data-key="reopen:{people["b"]}"]'])
                        await click('#app-nav [data-view=home]')
                        await page.wait_for("!document.querySelector('#home-view').hidden")
                        await click('#app-nav [data-view=voices]')
                        await page.wait_for(f"document.querySelector('#voice-attribution [data-key=\"reopen:{people['b']}\"]')")
                        reopen_path = f"/api/memory/voice/consents/{people['b']}/reopen"
                        faults.update({('POST', reopen_path): 'lose', ('GET', '/api/memory/voice/consents'): 'down'})
                        await press(f'#voice-attribution [data-key="reopen:{people["b"]}"]')
                        await page.wait_for("document.querySelector('#voice-attribution-confirm-dialog').open")
                        assert 'Persona sintética B va a poder volver a verificar su voz' in await page.evaluate("document.querySelector('#voice-attribution-confirm-dialog').innerText")
                        await accept('#voice-attribution-confirm-dialog', 'voice-admin-accept')
                        admin_key = f"kza:voice-admin-op:reopen:{people['b']}"
                        await page.wait_for(f"sessionStorage.getItem({json.dumps(admin_key)}) && " + text_of('#voice-attribution-live') + ".startsWith('Sin confirmar.')")
                        faults.clear()
                        await page.command('Page.reload')
                        await page.wait_for(f"(document.querySelector('[data-voice-admin-notice=\"reopen:{people['b']}\"]')?.textContent||'').length>0 && sessionStorage.getItem({json.dumps(admin_key)})===null")
                        settled = await page.evaluate(f"document.querySelector('[data-voice-admin-notice=\"reopen:{people['b']}\"]').textContent")
                        assert 'Comprobado' not in settled and 'Comprobado' not in await page.evaluate("document.querySelector('#voice-attribution').innerText"), settled
                        assert count('POST', reopen_path) == 1, 'never resent'
                        assert await rows('SELECT state FROM voice_attribution_consents WHERE person_id=?', people['b']) == [('pending_verification',)]
                        checks.append('S (administration): a reopen whose answer was lost survives a reload as a content-free marker; the next read settles it without «Comprobado» and without resending')
                        await page.wait_for(f"document.querySelector('#voice-attribution [data-key=\"resume:{profiles['a']}\"]')")
                        await press(f'#voice-attribution [data-key="resume:{profiles["a"]}"]')
                        await accept('#voice-attribution-confirm-dialog', 'voice-admin-accept')
                        await page.wait_for(text_of('#voice-attribution-live') + "==='Se reanudaron las propuestas por voz de ese perfil.'")
                        assert await rows('SELECT COUNT(*) FROM voice_attribution_suspensions') == [(0,)]
                        await press('#voice-attribution [data-key=voice-switch]')
                        await accept('#voice-attribution-confirm-dialog', 'voice-admin-accept')
                        await page.wait_for(text_of('#voice-attribution-live') + "==='Memoria por voz apagada.'")
                        checks.append('operator: consents list with the verified profile and names, the multi-profile notice, «Verificada por» and the pause next to the profile in Voces, no scores; reopen and resume through dialogs and the real routes; the switch turned off')
                        await logout()

                        # Switch off: B (pending again) sees the status and only the revocation.
                        await login(page, homes['b'].token, 'Acceso de hogar')
                        await open_my_memory()
                        await page.wait_for(text_of('#my-voice-status') + ".includes('desactivada en este hogar')")
                        labels = await page.evaluate("[...document.querySelectorAll('#my-memory-view .memory-voice button')].filter(b=>!b.closest('[hidden]')).map(b=>b.textContent)")
                        assert labels == ['Darme de baja…'], labels
                        assert await page.evaluate("!document.querySelector('#my-voice-room')")
                        await press('#my-memory-view [data-key=voice-revoke]')
                        await accept('#my-voice-confirm-dialog', 'voice-confirm-accept')
                        await page.wait_for(text_of('#my-voice-status') + ".startsWith('Te diste de baja.')")
                        assert await rows('SELECT state FROM voice_attribution_consents WHERE person_id=?', people['b']) == [('revoked',)]
                        checks.append('switch off: the person sees the status and only «Darme de baja»; the revocation works')
                        assert await page.evaluate("fetch('/api/memory/voice/consents').then(r=>r.status)") == 403
                        await logout()
                        await login(page, operator.token, 'Administración')
                        assert await page.evaluate("!document.querySelector('[data-view=mymemory]')")
                        assert await page.evaluate("fetch('/api/memory/mine/voice').then(r=>r.json().then(b=>[r.status,b.detail]))") == [403, 'dashboard_permission_denied']
                        await logout()
                        checks.append('negatives: a household account gets 403 on the consents list; the operator has no Mis recuerdos and gets 403 on the person status')
                        assert not zoom_findings, zoom_findings
                        assert page.errors == [], page.errors

                        result = {'passed': True, 'synthetic_fixture': True, 'actual_chromium_http_login': True, 'shared_privacy_ledger': True,
                                  'roles': ['home-person-a', 'home-person-b', 'home-person-c', 'operator'], 'widths': list(WIDTHS), 'text_zoom_widths': list(ZOOM_WIDTHS),
                                  'checks': checks, 'a11y_evidence': [f'a11y-{view}.json' for view in audits],
                                  'a11y_findings': {view: len(audit['findings']) for view, audit in audits.items()},
                                  'text_zoom_findings': len(zoom_findings), 'requests': len(requests), 'javascript_exceptions': len(page.errors),
                                  'voice_route_calls': {key: count(*key.split(' ', 1)) for key in ('GET /api/memory/mine/voice', 'POST /api/memory/mine/voice/consent',
                                                                                                    'POST /api/memory/mine/voice/verification', 'DELETE /api/memory/mine/voice/consent',
                                                                                                    'GET /api/memory/voice/consents', 'POST /api/memory/voice/switch')}}
                        (evidence / 'browser-result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
                        print(json.dumps(result, ensure_ascii=False))
            finally:
                if browser is not None and browser.poll() is None:
                    browser.terminate()
                    await asyncio.to_thread(browser.wait, 10)
                await dashboard.stop()
                await serving


if __name__ == '__main__':
    asyncio.run(main())
