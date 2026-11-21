"""Real Chromium check of the technical panel reached by direct URL /obs/.

Verifies that every view renders without runtime errors and without example
data, that unavailable sources are shown as unavailable, that controls without
a backing operation are disabled with a reason, and that alert acknowledgement
runs the existing authenticated endpoint and reports only the server result.
Keyboard navigation (WEB-03, WEB-05 /obs/) is driven with real key events: Tab
order over the nine entries, Enter and Space, a visible focus ring, the hash
written per view, Back/Forward and reload restoring it, focus moved to the view
region and the active view announced. The responsive audit adds the mobile
disclosure, populated tables, forms and media at five widths. The login audit
checks the native modal, focus return and error announcement. Synthetic fixtures only.
"""

import asyncio
import base64
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
from starlette.responses import JSONResponse
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from src.dashboard.live_event_bus import LiveEventBus
from src.dashboard.live_receipts import LiveReceiptStore
from src.memory.privacy_services import PrivacyServices
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_support import (Page, _press_tab, browser_binary, browser_profile_root, evidence_directory,
                                            press_enter)

EVIDENCE = evidence_directory('KZA_OBS_PANEL_EVIDENCE', 'data/autonomous-roadmap-20261102/t040-t041/obs-browser-runs')
TOKEN = 'synthetic-obs-panel-root'
FORBIDDEN = ('Marco', 'Lucía', 'vllm-7b', 'ik-30b', '192.168.1.2', '21d 08h', 'kza@a3f9c2e', 'turn_8821',
             'Khruangbin', 'datos de ejemplo', 'wireframe')
OBS_FETCH_SPY = r"""(() => {
    const fetch = window.fetch.bind(window);
    window.__obsUserWrites = [];
    window.__obsAuthEvents = 0;
    window.addEventListener('kza:auth-required', () => window.__obsAuthEvents++);
    window.fetch = (input, options) => {
        const method = (options?.method ?? input.method ?? 'GET').toUpperCase();
        const path = new URL(input.url ?? input, location.href).pathname;
        if (['POST', 'DELETE'].includes(method) && /^\/api\/admin\/users(?:\/|$)/.test(path)) {
            window.__obsUserWrites.push({method, path});
        }
        return fetch(input, options);
    };
})()"""
# Navigation entries in document order: (hash id, visible label, readiness condition).
NAV = (('live', 'Live', "!!document.querySelector('#view-live')"),
       ('conversations', 'Conversaciones', "!!document.querySelector('#view-conversations h1')"),
       ('privacy', 'Privacidad', "!!document.querySelector('#view-privacy')"),
       ('alerts', 'Alertas', "!!document.querySelector('#view-alerts')"),
       ('ha', 'Home Assistant', "!!document.querySelector('#view-ha')"),
       ('llm', 'LLM Router', "!!document.querySelector('#view-llm')"),
       ('users', 'Voces', "!!document.querySelector('#view-users h1')"),
       ('system', 'Sistema', "!!document.querySelector('#view-system')"),
       ('admin', 'Admin', "document.body?.innerText.includes('Acciones de servicio')"))
# The focused element, its focus ring and the ring's contrast against the
# background it is drawn on (WCAG 1.4.11 asks for at least 3:1).
FOCUS_PROBE = r"""(()=>{
  const el=document.activeElement, items=[...document.querySelectorAll('.nav-item')];
  const rgb=value=>(value.match(/[\d.]+/g)||[]).slice(0,3).map(Number);
  const lum=([r,g,b])=>{const f=c=>{c/=255;return c<=0.03928?c/12.92:((c+0.055)/1.055)**2.4};return 0.2126*f(r)+0.7152*f(g)+0.0722*f(b)};
  const behind=node=>{for(let n=node.parentElement;n;n=n.parentElement){const c=getComputedStyle(n).backgroundColor,m=c.match(/[\d.]+/g);
    if(m&&(m.length<4||+m[3]>0))return c}return getComputedStyle(document.body).backgroundColor};
  const style=getComputedStyle(el), a=lum(rgb(style.outlineColor)), b=lum(rgb(behind(el)));
  const rest=items.find(item=>item!==el), restStyle=rest&&getComputedStyle(rest);
  return {nav:items.indexOf(el), tag:el.tagName.toLowerCase(), text:(el.textContent||'').trim().slice(0,40),
          focusVisible:el.matches(':focus-visible'), outlineStyle:style.outlineStyle, outlineWidth:parseFloat(style.outlineWidth)||0,
          outlineColor:style.outlineColor, contrast:Math.round((Math.max(a,b)+0.05)/(Math.min(a,b)+0.05)*100)/100,
          unfocusedOutline:restStyle?restStyle.outlineStyle:null};
})()"""
NAV_STATE = ("(()=>{const main=document.querySelector('main.main');return {hash:location.hash,"
             "current:[...document.querySelectorAll('.nav-item[aria-current=page]')].map(n=>n.children[1].textContent),"
             "crumbs:document.querySelector('.crumbs b')?.textContent,focusMain:document.activeElement===main,"
             "mainLabel:main?.getAttribute('aria-label'),announced:document.querySelector('.sr-only[role=status]')?.textContent}})()")
WIDTHS = (320, 390, 768, 1280, 1440)
TABLES = {
    'admin': {'Usuarios administrados'},
    'conversations': {'Turnos de conversación'},
    'ha': {'Entidades de Home Assistant', 'Acciones disparadas'},
    'llm': {'Endpoints del LLM Router'},
    'system': {'Procesos de Synthetic GPU', 'Servicios del sistema'},
    'users': {'Tabla de muestras de voz'},
}
LAYOUT_PROBE = r"""(() => {
  const bounds = el => el.getBoundingClientRect();
  const visible = selector => [...document.querySelectorAll(selector)].filter(el => el.getClientRects().length);
  const root = document.documentElement;
  const fields = 'input:not([type=checkbox]):not([type=radio]), select, textarea';
  const review = '#view-users section[aria-label="Escuchar muestra"]';
  const reviewCard = document.querySelector(review);
  return {
    width: window.innerWidth, client: root.clientWidth,
    page: Math.max(root.scrollWidth, document.body.scrollWidth),
    targets: visible(`button, ${fields}`).filter(el => bounds(el).width < 44 || bounds(el).height < 44)
      .map(el => el.getAttribute('aria-label') || el.textContent.trim()),
    fields: visible(fields).filter(el => parseFloat(getComputedStyle(el).fontSize) < 16)
      .map(el => el.outerHTML),
    grids: visible('.grid-2, .grid-3, .grid-4').map(el => {
      const columns = Number(el.className.match(/grid-(\d)/)[1]);
      return [columns, getComputedStyle(el).gridTemplateColumns.split(' ').length];
    }),
    tables: visible('table.k').map(el => {
      const region = el.parentElement;
      return {name: region.getAttribute('aria-label'), role: region.getAttribute('role'),
              tabindex: region.tabIndex, overflow: getComputedStyle(region).overflowX};
    }),
    voice: reviewCard ? {
      bounds: [bounds(reviewCard).left, bounds(reviewCard).right],
      captures: visible('#view-users [data-capture-id]').length,
      audio: [...reviewCard.querySelectorAll('audio')].map(el => ({
        hasPreview: el.src.startsWith('blob:'), paused: el.paused, controls: el.controls,
        autoplay: el.autoplay, preload: el.preload,
      })),
      buttons: visible(`${review} button`).map(el => ({
        name: el.textContent.trim(), left: bounds(el).left, right: bounds(el).right,
        width: bounds(el).width, height: bounds(el).height, disabled: el.disabled, tabindex: el.tabIndex,
      })),
    } : null,
    overlays: visible('.admin-overlay > .card, dialog.login-dialog:modal').map(el => [bounds(el).left, bounds(el).right]),
  };
})()"""

# Seed layout-specific rows at the stores' normal delivery boundaries, after the
# authenticated HTTP and operation checks have run against the real routes.
LAYOUT_OBSERVATIONS = r"""(() => {
  const store = window.__kzaObs, observation = {source: 'real'};
  const longId = 'synthetic_' + 'layout_'.repeat(20);
  store.accept('HA_ENTITIES', observation, [{id: `sensor.${longId}`, domain: 'sensor', name: 'Synthetic sensor', state: 'on', score: .9}]);
  store.accept('HA_ACTIONS', observation, [{id: 'synthetic-action', service: 'light.turn_on', target: longId, ok: true, lat_ms: 100}]);
  store.accept('LLM_ENDPOINTS', observation, [{id: 'synthetic-model', role: 'local', url: `http://synthetic.test/${longId}`, state: 'healthy'}]);
  store.accept('GPUS', observation, [{id: 0, name: 'Synthetic GPU', util: 10, procs: [{name: longId, vram: 2}]}]);
  store.accept('SERVICES', observation, [{name: 'kza-voice', status: 'active'}]);
})()"""
LAYOUT_CONVERSATION = r"""(() => {
  const wall = Date.now(), now = wall / 1000;
  return window.__kzaConversationState.receive([{
    id: 'turn_1_00000000-0000-0000-0000-000000000001', ts: '12:00', user: 'Synthetic speaker',
    zone: 'Synthetic room', path: 'command', stt: 'Synthetic conversation phrase', tts: '', intent: '', target: '',
    observed_at: now, expires_at: now + 60,
  }], {wall, mono: performance.now()});
})()"""
LAYOUT_VOICE = r"""(async () => {
  const wav = await encodeWav(new Int16Array(16000), 16000).arrayBuffer();
  const audio_base64 = btoa(String.fromCharCode(...new Uint8Array(wav)));
  const state = window.__kzaVoiceState;
  for (let i = 0; i < 4; i++) {
    await window.__kzaVoiceController.settled();
    if (state.profiles.status() !== 'loading' && state.captures.status() !== 'loading') break;
  }
  const wall = Date.now(), now = wall / 1000;
  const started = {wall, mono: performance.now()};
  const id = 'vf_' + 'a'.repeat(64);
  const timing = {observed_at: now, expires_at: now + 60};
  const captures = state.captures.receive([{
    id, ...timing, created_at: now, duration_s: 1, review_available: true, room_id: null,
    suggested_profile: null, similarity: null, group_id: '00000000-0000-0000-0000-000000000001',
    group_revision: 1, group_size: 1, group_state: 'unknown', group_reason: 'unreviewed',
    group_similarity: null, group_expires_at: timing.expires_at,
  }], started);
  return captures && state.review.receive([{
    id, ...timing, room_id: 'Synthetic room', transcript: 'Synthetic audio phrase',
    duration_s: 1, clip_duration_s: 1, clip_truncated: false, transcript_scope: 'full_segment',
    audio_mime: 'audio/wav', audio_base64, audio_metadata: null, transcript_metadata: null,
  }], started);
})()"""


async def _press(page, key: str, code: str, keycode: int, *, text: str | None = None, modifiers: int = 0) -> None:
    """Dispatch one real key press (down and up) to the focused element."""
    down = {'type': 'keyDown' if text else 'rawKeyDown', 'key': key, 'code': code, 'modifiers': modifiers,
            'windowsVirtualKeyCode': keycode, 'nativeVirtualKeyCode': keycode}
    if text:
        down['text'] = text
    await page.command('Input.dispatchKeyEvent', down)
    await page.command('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': key, 'code': code, 'modifiers': modifiers,
                                                  'windowsVirtualKeyCode': keycode, 'nativeVirtualKeyCode': keycode})


async def press_space(page) -> None:
    """Activate the focused control with a real Space key."""
    await _press(page, ' ', 'Space', 32, text=' ')


async def press_shift_tab(page) -> None:
    """Move focus backwards with a real Shift+Tab."""
    await _press(page, 'Tab', 'Tab', 9, modifiers=8)


def assert_ring(probe: dict) -> None:
    """The focused control matches :focus-visible and draws a visible, contrasting ring."""
    assert probe['focusVisible'], probe
    assert probe['outlineStyle'] not in ('none', 'hidden') and probe['outlineWidth'] >= 2, probe
    assert probe['contrast'] >= 3, probe


def expected_state(index: int) -> dict:
    """Navigation state once the entry at ``index`` has been opened by navigation."""
    view, label, _ = NAV[index]
    return {'hash': f'#{view}', 'current': [label], 'crumbs': label, 'focusMain': True, 'mainLabel': label,
            'announced': f'Vista activa: {label}'}


async def history_step(page, delta: int) -> None:
    """Traverse session history like the browser's Back (-1) and Forward (+1) buttons."""
    history = await page.command('Page.getNavigationHistory')
    entry = history['entries'][history['currentIndex'] + delta]
    await page.command('Page.navigateToHistoryEntry', {'entryId': entry['id']})


async def resize(page: Page, width: int) -> None:
    """Emulate the viewport and mobile layout below the panel breakpoint."""
    await page.command('Emulation.setDeviceMetricsOverride',
                       {'width': width, 'height': 844, 'deviceScaleFactor': 1, 'mobile': width < 768})


async def check_layout(page: Page, width: int, view: str, *, names=(), overlay: bool = False) -> dict:
    """Check actual geometry and accessible names in Chromium at one width."""
    probe = await page.evaluate(LAYOUT_PROBE)
    assert probe['width'] == width and probe['page'] <= probe['client'], (width, view, probe)
    assert not probe['targets'] and not probe['fields'], (width, view, probe)
    assert all(actual == (1 if width < 768 else configured) for configured, actual in probe['grids']), (width, view, probe)
    assert len(probe['overlays']) == int(overlay), (width, view, probe)
    assert all(0 <= left < right <= probe['client'] for left, right in probe['overlays']), (width, view, probe)
    regions = TABLES.get(view, set())
    assert {item['name'] for item in probe['tables']} == regions, (width, view, probe)
    for item in probe['tables']:
        assert item['role'] == 'region' and item['tabindex'] == 0 and item['overflow'] == 'auto', item
    if view == 'users':
        voice = probe['voice']
        assert voice and voice['captures'] == 1, (width, probe)
        assert voice['audio'] == [{'hasPreview': True, 'paused': True, 'controls': False,
                                   'autoplay': False, 'preload': 'auto'}], (width, voice)
        assert len(voice['buttons']) == 1, (width, voice)
        play = voice['buttons'][0]
        left, right = voice['bounds']
        assert 0 <= left <= play['left'] < play['right'] <= right <= probe['client'], (width, voice)
        assert play['width'] >= 44 and play['height'] >= 44, (width, play)
        assert play['name'] == 'Reproducir' and play['tabindex'] == 0 and not play['disabled'], (width, play)
    tree = (await page.command('Accessibility.getFullAXTree'))['nodes']
    exposed = {(item.get('role', {}).get('value'), item.get('name', {}).get('value'))
               for item in tree if not item.get('ignored')}
    if view == 'login':
        assert ('dialog', 'Acceso admin') in exposed, (width, exposed)
        assert ('button', 'Abrir acceso admin') not in exposed, (width, exposed)
    if view == 'users':
        assert ('region', 'Escuchar muestra') in exposed, (width, exposed)
        assert ('button', 'Reproducir') in exposed, (width, exposed)
    for name in regions:
        assert ('region', name) in exposed, (width, view, name)
    exposed_names = {name for _, name in exposed}
    for name in names:
        assert name in exposed_names, (width, view, name)
    return probe


async def screenshot(page: Page, path: Path, *, full: bool = True) -> None:
    """Capture the current panel without waiting for display refresh callbacks."""
    await page.evaluate('window.scrollTo(0, 0)')
    result = await page.command('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': full})
    path.write_bytes(base64.b64decode(result['data']))
    path.chmod(0o600)


async def set_login_token(page: Page, token: str) -> None:
    """Fill the controlled field via its native setter and React input event."""
    await page.evaluate(
        "(()=>{const input=document.querySelector('dialog:modal input[type=password]');"
        "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,"
        + json.dumps(token) + ");input.dispatchEvent(new Event('input',{bubbles:true}));})()"
    )


async def login_keyboard(page: Page, login_release: asyncio.Event, requests: list[dict]) -> None:
    """Verify the modal's key loop, return focus and single pending POST."""
    password_focused = "document.activeElement.matches('dialog:modal input[type=password]')"
    assert await page.evaluate(password_focused)
    assert not await page.evaluate("!!document.querySelector('dialog:modal [role=alert]')")
    assert await page.evaluate(
        "(()=>{const dialog=document.querySelector('dialog:modal'), field=dialog.querySelector('input');"
        "return dialog.getAttribute('aria-modal')==='true' && field.labels.length===1 && "
        "field.labels[0].textContent==='Token de administración' && "
        "document.getElementById(dialog.getAttribute('aria-describedby')).textContent.trim()==="
        "'Token configurado en KZA_DASHBOARD_TOKEN.';})()"
    )
    await press_shift_tab(page)
    assert await page.evaluate("document.activeElement.textContent==='Cerrar'")
    await _press_tab(page)
    assert await page.evaluate(password_focused)
    await _press_tab(page)
    assert await page.evaluate("document.activeElement.textContent==='Entrar'")
    await _press(page, 'Escape', 'Escape', 27)
    await page.wait_for("!document.querySelector('dialog:modal') && document.activeElement.textContent==='Abrir acceso admin'")
    await press_enter(page)
    await page.wait_for(password_focused)

    await set_login_token(page, 'synthetic-invalid')
    await page.evaluate("(()=>{const form=document.querySelector('dialog:modal form');"
                        "form.requestSubmit();form.requestSubmit();})()")
    await page.wait_for("document.querySelector('dialog:modal [role=status]')?.textContent==='Enviando…'")
    assert await page.evaluate("document.querySelector('dialog:modal button[type=submit]').disabled")
    login_release.set()
    await page.wait_for("document.querySelector('dialog:modal [role=alert]')?.textContent==="
                        "'La credencial es inválida o la sesión venció.'")
    assert not await page.evaluate("document.querySelector('dialog:modal button[type=submit]').disabled")
    assert [entry['status'] for entry in requests if entry['path'] == '/api/admin/auth/login'] == [401]


async def mobile_keyboard(page: Page) -> None:
    """Exercise disclosure, Escape and history with real key events at 320 px."""
    await page.evaluate("location.hash = 'live'")
    await page.wait_for(NAV[0][2])
    await page.wait_for("document.activeElement.matches('main')")
    toggle = "document.querySelector('.menu-toggle')"
    menu = f"document.getElementById({toggle}.getAttribute('aria-controls'))"
    collapsed = f"{toggle}.getAttribute('aria-expanded') === 'false' && !{menu}.getClientRects().length"
    expanded = f"{toggle}.getAttribute('aria-expanded') === 'true' && {menu}.getClientRects().length"
    assert await page.evaluate(collapsed)
    await press_shift_tab(page)
    assert await page.evaluate(f"document.activeElement === {toggle}")
    assert_ring(await page.evaluate(FOCUS_PROBE))
    await _press_tab(page)
    assert await page.evaluate("document.activeElement.matches('.topbar button')")
    await press_shift_tab(page)
    await press_space(page)
    await page.wait_for(expanded)
    await check_layout(page, 320, 'live', names=('Menú', 'Secciones del panel técnico'))
    await screenshot(page, EVIDENCE / 'obs-mobile-menu-320.png', full=False)
    await _press_tab(page)
    probe = await page.evaluate(FOCUS_PROBE)
    assert probe['nav'] == 0, probe
    assert_ring(probe)
    await _press(page, 'Escape', 'Escape', 27)
    await page.wait_for(collapsed)
    assert await page.evaluate(f"document.activeElement === {toggle}")
    assert_ring(await page.evaluate(FOCUS_PROBE))

    await press_enter(page)
    await page.wait_for(expanded)
    await _press_tab(page)
    await press_enter(page)  # Selecting the current view also closes and moves focus.
    await page.wait_for(collapsed)
    assert await page.evaluate(NAV_STATE) == expected_state(0)
    await press_shift_tab(page)
    await press_enter(page)
    await page.wait_for(expanded)
    for _ in range(4):
        await _press_tab(page)
    assert (await page.evaluate(FOCUS_PROBE))['nav'] == 3
    await press_space(page)
    await page.wait_for(NAV[3][2])
    await page.wait_for(collapsed)
    assert await page.evaluate(NAV_STATE) == expected_state(3)

    await press_shift_tab(page)
    await press_enter(page)
    await page.wait_for(expanded)
    await history_step(page, -1)
    await page.wait_for(NAV[0][2])
    await page.wait_for(collapsed)
    assert await page.evaluate(NAV_STATE) == expected_state(0)
    await history_step(page, +1)
    await page.wait_for(NAV[3][2])
    await page.wait_for("document.activeElement.matches('main')")
    assert await page.evaluate(NAV_STATE) == expected_state(3)


async def responsive_layout(page: Page) -> dict:
    """Check all nine views and the enrolment overlay at each requested width."""
    await page.evaluate(LAYOUT_OBSERVATIONS)
    await resize(page, 320)
    await mobile_keyboard(page)
    result = {}
    for width in WIDTHS:
        await resize(page, width)
        visibility = await page.evaluate("[...document.querySelectorAll('.menu-toggle, .nav')].map(el => !!el.getClientRects().length)")
        assert visibility == [width < 768, width >= 768], (width, visibility)
        result[width] = {}
        for view, _, ready in NAV:
            await page.evaluate(f"location.hash = '{view}'")
            await page.wait_for(ready)
            names = ('Actualizar datos',)
            if view == 'conversations':
                assert await page.evaluate(LAYOUT_CONVERSATION)
                names += ('Todas las voces', 'Todos los recorridos')
            if view == 'users':
                assert await page.evaluate(LAYOUT_VOICE)
                names += ('Nombre del perfil',)
            if view == 'admin':
                await page.wait_for("!!document.querySelector('.user-actions button')")
                names += ('Enrolar voz de Synthetic account',)
            result[width][view] = await check_layout(page, width, view, names=names)
            if width == 390 and view == 'live':
                await screenshot(page, EVIDENCE / 'obs-view-live-390.png', full=False)

        if width == 320:
            await page.evaluate("[...document.querySelectorAll('main .content button')]"
                                ".find(button => button.textContent === 'cerrar sesión').focus()")
            await _press_tab(page)
            assert await page.evaluate("document.activeElement.matches('.table-scroll')")
            assert_ring(await page.evaluate(FOCUS_PROBE))
            assert await page.evaluate("document.activeElement.scrollWidth > document.activeElement.clientWidth")
            await _press_tab(page)
            await page.wait_for("document.activeElement.matches('.user-actions button') && "
                                "document.activeElement.closest('.table-scroll').scrollLeft > 0")
            assert_ring(await page.evaluate(FOCUS_PROBE))

        await page.evaluate("document.querySelector('.user-actions button').focus()")
        await press_enter(page)
        await page.wait_for("!!document.querySelector('[role=dialog]')")
        result[width]['enrollment'] = await check_layout(page, width, 'admin', overlay=True,
                                                         names=('Cerrar enrolamiento', '🎙 grabar muestra 1'))
        await page.evaluate("document.querySelector('[aria-label=\"Cerrar enrolamiento\"]').click()")
        await page.wait_for("!document.querySelector('[role=dialog]')")
    return result


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
    faults = {}
    alerts_hold: tuple[asyncio.Event, asyncio.Event] | None = None
    login_release = asyncio.Event()
    alerts = SyntheticAlerts()
    binary = browser_binary()
    with tempfile.TemporaryDirectory(prefix='kza-obs-panel-', dir=browser_profile_root()) as temporary:
        directory = Path(temporary)
        authority = DashboardAuthority(directory / 'auth.db', session_ttl_s=600)
        await authority.initialize()
        users = UserManager(str(directory / 'users.json'))
        users.add_user('Synthetic account', permission_level=PermissionLevel.ADULT)
        # A long cell places the enrolment column beyond the narrow viewport.
        users.add_user('Synthetic' + 'W' * 48, permission_level=PermissionLevel.ADULT)
        privacy = PrivacyServices(str(directory / 'privacy.db'))
        await privacy.acall(privacy.initialize)
        receipts = LiveReceiptStore(privacy.ledger)
        bus = LiveEventBus(source_policy=privacy.ledger, clock=privacy.ledger.clock, receipts=receipts)
        privacy.add_borrower(bus.begin_close)
        bus.start()
        dashboard = DashboardAPI(authority=authority, alert_manager=alerts, user_manager=users, event_bus=bus)

        @dashboard.app.middleware('http')
        async def record(request, call_next):
            if request.url.path == '/api/admin/auth/login':
                await login_release.wait()
            if request.method == 'GET' and request.url.path == '/api/alerts' and alerts_hold is not None:
                started, release = alerts_hold
                started.set()
                await release.wait()
            status = faults.get(request.url.path)
            response = JSONResponse({}, status_code=status) if status else await call_next(request)
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
                    await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': OBS_FETCH_SPY})
                    await page.command('Emulation.setDeviceMetricsOverride',
                                       {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.command('Page.navigate', {'url': f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("!!document.querySelector('dialog:modal input[type=password]')")
                    login = {}
                    for width in WIDTHS:
                        await resize(page, width)
                        login[width] = await check_layout(page, width, 'login', overlay=True,
                                                          names=('Acceso admin', 'Token de administración', 'Entrar', 'Cerrar'))
                    try:
                        await login_keyboard(page, login_release, requests)
                    finally:
                        login_release.set()
                    checks.append('first visit opens a clean named modal, wraps Tab, restores focus and sends one invalid POST')
                    await set_login_token(page, TOKEN)
                    await press_enter(page)
                    await page.wait_for("!!document.querySelector('#view-live')")
                    assert await page.evaluate("document.activeElement===document.querySelector('main.main')")
                    assert [entry['status'] for entry in requests if entry['path'] == '/api/admin/auth/login'] == [401, 200]
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
                    assert 'WS en vivo (live/conv/voces)' in text
                    checks.append('Live shows unavailable zones/router/latency instead of examples and scopes WS pulse')

                    text = await visit('Alertas', 'view-alerts')
                    assert 'Synthetic device alert' in text
                    acknowledge = "[...document.querySelectorAll('button')].find(b=>b.textContent==='Marcar atendida')"
                    ack_path = '/api/admin/alerts/synthetic-alert-1/ack'
                    faults[ack_path] = 504
                    await page.evaluate(f"{acknowledge}.click()")
                    await page.wait_for(f"{acknowledge}?.disabled && document.body.innerText.includes('Resultado incierto')")
                    assert not alerts.marked
                    del faults[ack_path]
                    await page.evaluate("document.querySelector('.topbar button').click()")
                    await page.wait_for(f"{acknowledge}?.disabled === false")
                    assert sum(item['path'] == ack_path for item in requests) == 1
                    checks.append('a fresh alerts read enables a new explicit acknowledgement without an automatic retry')
                    await page.evaluate(f"{acknowledge}.click()")
                    await page.wait_for("document.body.innerText.includes('El servidor marcó la alerta como atendida.')")
                    assert alerts.marked == [('synthetic-alert-1', True)]
                    assert {'method': 'POST', 'path': ack_path, 'status': 200} in requests
                    assert await page.evaluate("document.body.innerText.includes('atendidas · 1')")
                    checks.append('ack runs POST /api/admin/alerts/{id}/ack and shows the server-confirmed result')

                    # WEB-09: expiry marks vencida without extending expires_at locally,
                    # polls ONLY the active view (/api/alerts), and renews to vigente on server reply.
                    await page.wait_for("!document.querySelector('.topbar button').disabled")
                    before_expires = await page.evaluate("window.__kzaObs.entry('ALERTS').expiresAt")
                    assert before_expires
                    before_requests = len(requests)
                    poll_started, poll_release = asyncio.Event(), asyncio.Event()
                    alerts_hold = (poll_started, poll_release)
                    try:
                        await asyncio.wait_for(poll_started.wait(), timeout=10)
                        await page.wait_for("document.querySelector('#view-alerts').innerText.includes('vencida')")
                        assert await page.evaluate("window.__kzaObs.freshness('ALERTS')") == 'vencida'
                        assert await page.evaluate("window.__kzaObs.entry('ALERTS').expiresAt") == before_expires
                        inactive_paths = {'/api/zones', '/api/llm/endpoints', '/api/ha/entities',
                                          '/api/ha/actions', '/api/system/gpus', '/api/system/services'}
                        inactive_hits = [item for item in requests[before_requests:] if item['path'] in inactive_paths]
                        assert not inactive_hits, inactive_hits
                    finally:
                        alerts_hold = None
                        poll_release.set()
                    await page.wait_for(
                        "document.querySelector('#view-alerts').innerText.includes('vigente') && "
                        f"window.__kzaObs.entry('ALERTS').expiresAt !== {json.dumps(before_expires)}"
                    )
                    after_expires = await page.evaluate("window.__kzaObs.entry('ALERTS').expiresAt")
                    assert after_expires > before_expires
                    checks.append('expired active-view observation marks vencida without extending expires_at and renews end-to-end')

                    # WEB-09: hiding the tab stops the poller (relabels vencida with zero fetches)
                    # and returning to visible immediately revalidates.
                    hidden_expires = after_expires
                    reads_before_hide = sum(item['path'] == '/api/alerts' for item in requests)
                    await page.evaluate("""(() => {
                        Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
                        Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
                        document.dispatchEvent(new Event('visibilitychange'));
                    })()""")
                    try:
                        await asyncio.sleep(6.5)  # past the 6 s observation TTL while hidden
                        assert await page.evaluate("document.querySelector('#view-alerts').innerText.includes('vencida')")
                        assert await page.evaluate("window.__kzaObs.entry('ALERTS').expiresAt") == hidden_expires
                        assert sum(item['path'] == '/api/alerts' for item in requests) == reads_before_hide
                    finally:
                        await page.evaluate("""(() => {
                            delete document.visibilityState;
                            delete document.hidden;
                            document.dispatchEvent(new Event('visibilitychange'));
                        })()""")
                    await page.wait_for(
                        "document.querySelector('#view-alerts').innerText.includes('vigente') && "
                        f"window.__kzaObs.entry('ALERTS').expiresAt !== {json.dumps(hidden_expires)}"
                    )
                    assert sum(item['path'] == '/api/alerts' for item in requests) == reads_before_hide + 1
                    checks.append('hidden tab stops polling without extending expires_at and resumes revalidation when visible')

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
                    assert await page.evaluate("""(() => {
                        const bars = Object.fromEntries([...document.querySelectorAll('#view-system .h-bar')]
                            .map(bar => [bar.querySelector('h2')?.textContent, bar.querySelector('.meta')?.textContent || '']));
                        return bars.GPUs?.includes('fuente ') && bars.Servicios?.includes('fuente ');
                    })()""")
                    checks.append('System view shows SourceNote provenance for both GPUs and Services')
                    text = await visit('Privacidad', 'view-privacy')
                    text = await visit('Conversaciones', 'view-conversations')
                    text = await visit('Voces', 'view-users')
                    await page.evaluate("[...document.querySelectorAll('.nav-item')].find(n=>n.textContent.includes('Admin')).click()")
                    await page.wait_for("document.body.innerText.includes('Acciones de servicio')")
                    await page.wait_for("!!document.querySelector('.user-actions button')")
                    assert await page.evaluate("""(() => {
                        const region = document.querySelector('[aria-label="Usuarios administrados"]');
                        const card = region.closest('.card');
                        return card.textContent.includes('Las personas se crean y se retiran en NEXO › Personas')
                            && [...region.querySelectorAll('thead th')].slice(0, 4).map(th => th.textContent)
                                .join(',') === 'id,nombre,permiso,voz'
                            && region.textContent.includes('Synthetic account')
                            && !card.querySelector('form, input, select, [aria-label^="Eliminar usuario"]')
                            && region.querySelector('button').getAttribute('aria-label') === 'Enrolar voz de Synthetic account';
                    })()""")
                    await page.evaluate("document.querySelector('.user-actions button').click()")
                    await page.wait_for("!!document.querySelector('.admin-overlay [role=dialog]')")
                    await page.evaluate("""document.querySelector('[aria-label="Cerrar enrolamiento"]').click()""")
                    await page.wait_for("!document.querySelector('.admin-overlay')")
                    assert await page.evaluate('window.__obsUserWrites') == []
                    assert not [item for item in requests if item['method'] in {'POST', 'DELETE'}
                                and (item['path'] == '/api/admin/users' or item['path'].startswith('/api/admin/users/'))]
                    await screenshot(page, EVIDENCE / 'obs-admin.png')
                    checks.append('Admin lists users, opens and closes voice enrollment, and performs no admin-user writes')

                    restart = "document.querySelector('.admin-service-row:first-child button')"
                    restart_path = '/api/admin/services/kza-voice/restart'

                    async def request_restart():
                        await page.evaluate(f"{restart}.click()")
                        await page.wait_for(f"{restart}.textContent === 'Confirmar reinicio de kza-voice'")
                        await page.evaluate(f"{restart}.click()")

                    for status, message in ((409, 'Conflicto: el estado cambió'), (403, 'no tiene permiso para esta operación')):
                        auth_events = await page.evaluate('window.__obsAuthEvents')
                        faults[restart_path] = status
                        await request_restart()
                        await page.wait_for(f"document.body.innerText.includes({json.dumps(message)})")
                        assert {'method': 'POST', 'path': restart_path, 'status': status} in requests
                        assert await page.evaluate('window.__obsAuthEvents') == auth_events
                        assert not await page.evaluate('!!document.querySelector("dialog:modal")')
                    del faults[restart_path]
                    checks.append('409 reports a conflict and 403 reports denial without reopening login')

                    # Restart whose server outcome is uncertain: systemctl is replaced in this
                    # process only (never executed) and times out, so the server answers 504.
                    commands = []

                    def timed_out(args, **kwargs):
                        commands.append(list(args))
                        raise subprocess.TimeoutExpired(args, kwargs.get('timeout'))

                    with mock.patch('src.dashboard.admin.shutil.which', lambda _: '/usr/bin/systemctl'), \
                            mock.patch('src.dashboard.admin.subprocess.run', timed_out):
                        await request_restart()
                        await page.wait_for("document.body.innerText.includes('Resultado incierto: el servidor no pudo confirmar el efecto')")
                        await page.wait_for(f"{restart}.disabled && {restart}.textContent.includes('Reiniciar')")
                        sent = sum(item['path'] == restart_path for item in requests)
                        await page.evaluate(f"{restart}.click()")
                        await visit('Live', 'view-live')
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(n=>n.textContent.includes('Admin')).click()")
                        await page.wait_for(f"{restart}?.disabled")
                        await page.evaluate(f"{restart}.click()")
                        assert sum(item['path'] == restart_path for item in requests) == sent
                    assert commands == [['systemctl', '--user', 'restart', 'kza-voice']]
                    assert {'method': 'POST', 'path': '/api/admin/services/kza-voice/restart', 'status': 504} in requests
                    assert not await page.evaluate("document.body.innerText.includes('completó el reinicio')")
                    await screenshot(page, EVIDENCE / 'obs-admin-restart-uncertain.png')
                    checks.append('504 leaves restart disabled across navigation and blocks repeated clicks')

                    keyboard = await keyboard_navigation(page, port, checks)

                    responsive = await responsive_layout(page)
                    assert await page.evaluate('window.__obsUserWrites') == []
                    assert not [item for item in requests if item['method'] in {'POST', 'DELETE'}
                                and (item['path'] == '/api/admin/users' or item['path'].startswith('/api/admin/users/'))]
                    checks.append('fetch and server spies see zero admin-user writes while visiting and using the read-only list')
                    checks.append('responsive layout, accessible names and populated table regions at 320/390/768/1280/1440 px')
                    checks.append('mobile keyboard disclosure, Escape, focus return, hash history and keyboard table scrolling')
                    faults['/api/zones'] = 403
                    await page.evaluate("document.querySelector('.topbar button').click()")
                    await page.wait_for("document.body.innerText.includes('Fuentes sin permiso')")
                    assert await page.evaluate("!document.querySelector('dialog:modal') && "
                                               "window.__kzaObs.status('ZONES')==='sin permiso' && "
                                               "window.__kzaObs.can('admin:manage')")
                    checks.append('403 on zones marks the collection without expiring the technical session')

                    await page.evaluate("location.hash='live'")
                    await page.wait_for(NAV[0][2])
                    faults['/api/admin/auth/whoami'] = 503
                    await page.evaluate("location.hash='admin'")
                    await page.wait_for("document.querySelector('main [role=alert]')?.textContent==="
                                        "'La autoridad de acceso no está disponible.'")
                    assert not await page.evaluate("!!document.querySelector('dialog:modal')")
                    checks.append('503 on the Admin session check reports unavailable without opening login')

                    # WEB-09: WS reconnect to 'conectado' after disconnect rehydrates active/expired inventories.
                    del faults['/api/admin/auth/whoami']
                    del faults['/api/zones']
                    await page.evaluate("location.hash='alerts'")
                    await page.wait_for(NAV[3][2])
                    await page.wait_for("window.__kzaObs.channel('ws') === 'conectado'")
                    alerts_before_ws = sum(item['path'] == '/api/alerts' for item in requests)
                    zones_before_ws = sum(item['path'] == '/api/zones' for item in requests)
                    # Mark ZONES as an expired real observation so reconnect reconciles both active (ALERTS) and expired (ZONES).
                    await page.evaluate(
                        "window.__kzaObs.accept('ZONES', {source:'real', expiresAt:'2026-01-01T00:00:00Z'}, "
                        "[{id:'stale'}], window.__kzaObs.beginRead())"
                    )
                    assert await page.evaluate("window.__kzaObs.freshness('ZONES')") == 'vencida'
                    await page.evaluate("window.__kzaLiveWs.close()")
                    await page.wait_for("window.__kzaObs.channel('ws') !== 'conectado'")
                    await page.wait_for("window.__kzaObs.channel('ws') === 'conectado'")
                    async with asyncio.timeout(5):
                        while (sum(item['path'] == '/api/alerts' for item in requests) <= alerts_before_ws
                               or sum(item['path'] == '/api/zones' for item in requests) <= zones_before_ws):
                            await asyncio.sleep(.05)
                    checks.append('WS reconnection to conectado rehydrates active and expired inventories')

                    # WEB-09: 401 during deadline polling opens login and stops all further polling.
                    faults['/api/alerts'] = 401
                    async with asyncio.timeout(10):
                        while not await page.evaluate("!!document.querySelector('dialog:modal input[type=password]')"):
                            await asyncio.sleep(.05)
                    del faults['/api/alerts']
                    alerts_after_401 = sum(item['path'] == '/api/alerts' for item in requests)
                    await asyncio.sleep(1.5)
                    assert sum(item['path'] == '/api/alerts' for item in requests) == alerts_after_401
                    checks.append('401 during polling opens login and stops further polling')

                    assert not page.errors, page.errors
                    checks.append('no runtime exceptions')
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                browser.wait(timeout=10)
            server.should_exit = True
            await serving
            await bus.aclose()
            await privacy.aclose()
            await authority.aclose()
    result = {'passed': True, 'synthetic_fixture': True, 'checks': checks, 'keyboard': keyboard,
              'responsive': {width: {'login': login[width], **responsive[width]} for width in WIDTHS},
              'requests': [item for item in requests if item['method'] != 'GET' or item['path'].startswith('/api/alerts')]}
    (EVIDENCE / 'browser-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({**result, 'responsive': list(responsive)}))


async def keyboard_navigation(page: Page, port: int, checks: list) -> dict:
    """Drive the side navigation with real keys from the Admin view left by the restart check."""
    record = {'tab_order': [], 'enter': [], 'space': [], 'history': [], 'controls': []}
    ready = dict((view, condition) for view, _, condition in NAV)

    async def opened(index: int) -> dict:
        await page.wait_for(NAV[index][2])
        await page.wait_for("document.activeElement===document.querySelector('main.main')")
        state = await page.evaluate(NAV_STATE)
        assert state == expected_state(index), (NAV[index], state)
        return state

    # A view change by keyboard clears the operation notice, as a click did before.
    assert await page.evaluate("window.__kzaObs.notice() === null")
    uncertain = await page.evaluate("window.__kzaObs.operationOutcome('restartService', 'kza-voice')")
    assert uncertain['state'] == 'uncertain', uncertain
    await page.evaluate(f"window.__kzaObs.setNotice({json.dumps(uncertain)})")
    await page.wait_for("document.body.innerText.includes('Resultado incierto')")
    await page.evaluate("document.querySelectorAll('.nav-item')[8].focus()")
    await press_shift_tab(page)
    probe = await page.evaluate(FOCUS_PROBE)
    assert probe['nav'] == 7, probe
    await press_enter(page)
    await opened(7)
    assert not await page.evaluate("document.body.innerText.includes('Resultado incierto')")
    checks.append('opening a view with the keyboard clears the previous operation notice')
    # Back/Forward goes through hashchange, not the nav handler: it must clear the notice too.
    await page.evaluate(f"window.__kzaObs.setNotice({json.dumps(uncertain)})")
    await page.wait_for("document.body.innerText.includes('Resultado incierto')")
    await history_step(page, -1)
    await opened(8)
    assert not await page.evaluate("document.body.innerText.includes('Resultado incierto')")
    checks.append('Back/Forward to another view clears the previous operation notice')

    # Reload keeps the view named by the hash; nothing is focused on load.
    await page.evaluate("window.__obsBeforeReload=true")
    await page.command('Page.reload')
    await page.wait_for(f"!window.__obsBeforeReload && ({ready['admin']})")
    assert await page.evaluate("location.hash==='#admin'&&document.activeElement===document.body")
    assert await page.evaluate("[...document.querySelectorAll('.nav-item[aria-current=page]')].map(n=>n.children[1].textContent).join()") == 'Admin'
    assert await page.evaluate("document.querySelector('.sr-only[role=status]')?.textContent") == ''
    checks.append('reload with #admin reopens Admin without moving focus or announcing')

    # Real Tab from the start of the document: the nine entries in order, each with a visible ring.
    for index in range(len(NAV) + 1):
        await _press_tab(page)
        probe = await page.evaluate(FOCUS_PROBE)
        record['tab_order'].append(probe)
        if index < len(NAV):
            assert probe['nav'] == index and probe['text'].startswith(NAV[index][1]), (index, probe)
            assert probe['unfocusedOutline'] == 'none', probe
            assert_ring(probe)
            if NAV[index][0] == 'alerts':
                await screenshot(page, EVIDENCE / 'obs-focus-nav-alerts.png', full=False)
        else:
            assert probe['nav'] == -1 and probe['tag'] == 'button' and probe['text'] == 'Actualizar datos', probe
            assert_ring(probe)
    checks.append('Tab reaches the nine entries in document order with a 2 px :focus-visible ring of contrast >= 3:1, then the topbar button')

    # Enter and Space each open every view; the neighbour is focused and the target reached with a real (Shift+)Tab.
    for key, press in (('enter', press_enter), ('space', press_space)):
        for index in range(len(NAV)):
            if index:
                await page.evaluate(f"document.querySelectorAll('.nav-item')[{index - 1}].focus()")
                await _press_tab(page)
            else:
                await page.evaluate("document.querySelectorAll('.nav-item')[1].focus()")
                await press_shift_tab(page)
            probe = await page.evaluate(FOCUS_PROBE)
            assert probe['nav'] == index, (key, index, probe)
            assert_ring(probe)
            await press(page)
            record[key].append(await opened(index))
            if key == 'enter' and NAV[index][0] == 'alerts':
                await screenshot(page, EVIDENCE / 'obs-keyboard-alerts.png', full=False)
        checks.append(f'{key.capitalize()} opens each of the nine views: aria-current, hash, focus on the named view region, announcement')

    # Back and Forward restore the view from the hash, move focus and announce it.
    for delta, index in ((-1, 7), (-1, 6), (+1, 7), (+1, 8)):
        await history_step(page, delta)
        record['history'].append({'delta': delta, **await opened(index)})
    history = await page.command('Page.getNavigationHistory')
    start = max(i for i, entry in enumerate(history['entries']) if entry['url'] == f'http://127.0.0.1:{port}/obs/')
    await page.command('Page.navigateToHistoryEntry', {'entryId': history['entries'][start]['id']})
    await page.wait_for(ready['live'])
    await page.wait_for("document.activeElement===document.querySelector('main.main')")
    state = await page.evaluate(NAV_STATE)
    assert state == {**expected_state(0), 'hash': ''}, state
    record['history'].append({'to': 'entry without hash', **state})
    await history_step(page, +1)
    await page.wait_for("location.hash!==''")
    state = await page.evaluate(NAV_STATE)
    index = next(i for i, (view, _, _) in enumerate(NAV) if f'#{view}' == state['hash'])
    record['history'].append({'delta': 1, **await opened(index)})
    checks.append('Back and Forward restore the view named by the hash (Back to the entry without hash opens Live), with focus and announcement')

    # The read-only user table and enrolment button remain reachable by keyboard.
    await page.evaluate("document.querySelectorAll('.nav-item')[7].focus()")
    await _press_tab(page)
    await press_enter(page)
    await opened(8)
    await page.wait_for("!!document.querySelector('.user-actions button')")
    for selector in ('.topbar button', 'main .content .page-head button',
                     '[aria-label="Usuarios administrados"]', '.user-actions button'):
        await _press_tab(page)
        probe = await page.evaluate(FOCUS_PROBE)
        assert await page.evaluate(f"document.activeElement.matches({json.dumps(selector)})"), probe
        assert_ring(probe)
        record['controls'].append(probe)
    await press_shift_tab(page)
    probe = await page.evaluate(FOCUS_PROBE)
    assert await page.evaluate("document.activeElement.getAttribute('aria-label') === 'Usuarios administrados'"), probe
    assert_ring(probe)
    record['controls'].append(probe)
    await screenshot(page, EVIDENCE / 'obs-focus-admin-table.png', full=False)
    checks.append('the table region and buttons in Admin show the focus ring with the keyboard')
    (EVIDENCE / 'keyboard-navigation.json').write_text(json.dumps(record, indent=2) + '\n')
    return {'tab_order': [probe['text'] for probe in record['tab_order']],
            'contrast_min': min(probe['contrast'] for probe in record['tab_order'] + record['controls']),
            'enter_views': len(record['enter']), 'space_views': len(record['space']), 'history_steps': len(record['history'])}


if __name__ == '__main__':
    asyncio.run(main())
