"""Real Chromium, individual HTTP logins and synthetic HA for the NEXO shell."""

import asyncio
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import aiohttp
import uvicorn
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import (
    DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample,
)
from src.rooms.room_context import RoomConfig
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_typed_controls import Page
from tests.integration.test_typed_home_operations import typed_graph


INSTRUMENT = """(() => {
  const fetcher=window.fetch.bind(window);
  window.__qaDropNext=false;window.__qaDelayReceipt=false;window.__qaFailLogout=false;window.__qaHoldLogout=false;
  window.fetch=async(input,options={})=>{
    const path=String(input);
    if(path==='/api/admin/auth/logout'&&window.__qaFailLogout)throw new TypeError('Synthetic offline logout');
    if(path==='/api/admin/auth/logout'&&window.__qaHoldLogout)await new Promise(resolve=>window.__qaReleaseLogout=resolve);
    const response=await fetcher(input,options);
    if(path==='/api/admin/users'&&(!options.method||options.method==='GET')&&window.__qaHoldPeople){window.__qaHoldPeople=false;await new Promise(resolve=>window.__qaReleasePeople=resolve);}
    if(path==='/api/admin/users'&&options.method==='POST'&&window.__qaHoldPeopleMutation){window.__qaHoldPeopleMutation=false;await new Promise(resolve=>window.__qaReleasePeopleMutation=resolve);}
    if(path.startsWith('/api/admin/users/')&&options.method==='PUT'&&window.__qaDropPeopleMutation){window.__qaDropPeopleMutation=false;await response.text();throw new TypeError('Synthetic lost account receipt');}
    if(path==='/api/admin/users'&&(!options.method||options.method==='GET')&&window.__qaPeopleUnavailable){window.__qaPeopleUnavailable=false;return new Response(JSON.stringify({detail:'user_catalog_unavailable'}),{status:503,headers:{'Content-Type':'application/json'}});}
    if(path==='/api/home/actions'&&options.method==='POST'&&window.__qaDropNext){window.__qaDropNext=false;await response.text();throw new TypeError('Synthetic lost response');}
    if(path.startsWith('/api/home/actions?')&&window.__qaDelayReceipt){window.__qaDelayReceipt=false;return new Response(JSON.stringify({status:'in_progress'}),{status:200,headers:{'Content-Type':'application/json'}});}
    return response;
  };
  const request=window.requestAnimationFrame.bind(window),cancel=window.cancelAnimationFrame.bind(window);
  const frames=new Set();window.__qaFrames=frames;
  window.requestAnimationFrame=callback=>{let id;id=request(time=>{frames.delete(id);callback(time)});frames.add(id);return id};
  window.cancelAnimationFrame=id=>{frames.delete(id);cancel(id)};
})()"""


async def login(page: Page, token: str, label: str) -> None:
    await page.wait_for("!document.querySelector('#login-panel').hidden")
    await page.evaluate("document.querySelector('#token').value=" + json.dumps(token) + ";document.querySelector('#login-form').requestSubmit()")
    await page.wait_for("document.querySelector('#session-label').textContent===" + json.dumps(label))
    assert await page.evaluate("document.querySelector('#token').value==='' && document.cookie===''")


async def screenshot(page: Page, path: Path, *, full: bool = True, scroll_top: bool = True) -> None:
    if scroll_top:
        await page.evaluate('window.scrollTo(0,0)')
    await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
    if await page.evaluate("!!document.querySelector('#house-canvas canvas')"):
        assert await page.evaluate("(()=>{const canvas=document.querySelector('#house-canvas canvas'),gl=canvas.getContext('webgl2'),pixel=new Uint8Array(4);gl.readPixels(Math.floor(canvas.width/2),Math.floor(canvas.height/2),1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);return pixel[3]>0})()"), 'Illustration must remain visible between renders'
    result = await page.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": full})
    path.write_bytes(base64.b64decode(result["data"]))
    path.chmod(0o600)


async def main() -> None:
    os.environ["KZA_DASHBOARD_TOKEN"] = "synthetic-nexo-bootstrap"
    evidence = Path("data/autonomous-roadmap-20261102/t047/people-browser")
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)

    class DiagnosticPage(Page):
        async def wait_for(self, expression):
            try:
                for _ in range(400):
                    if await self.evaluate('!!(' + expression + ')'):
                        return
                    await asyncio.sleep(0.05)
                raise AssertionError('Browser condition timed out: ' + expression)
            except Exception:
                diagnostic = await self.evaluate("({text:document.body.innerText,ready:document.readyState,canvas:!!document.querySelector('#house-canvas canvas')})")
                (evidence / 'browser-failure.json').write_text(json.dumps({'condition': expression, 'document': diagnostic, 'errors': self.errors, 'requests': requests}, indent=2) + '\n')
                await screenshot(self, evidence / 'browser-failure.png')
                raise
    browser_data = Path.home() / "snap/chromium/common"
    browser_data.mkdir(mode=0o700, parents=True, exist_ok=True)
    ha, actions = typed_graph.__wrapped__()
    for entity, name in [("light.test_lamp", "Luz principal"), ("climate.test_ac", "Aire acondicionado"), ("cover.test_blind", "Persiana")]:
        ha.states[entity]["attributes"]["friendly_name"] = name
    ha.states['light.test_lamp']['attributes'].update(brightness=166, color_temp_kelvin=2700)
    requests = []
    with tempfile.TemporaryDirectory(prefix="kza-nexo-", dir=browser_data) as temporary:
        directory = Path(temporary)
        users = UserManager(str(directory / 'users.json'), retirement_proof=lambda identities: authority.users_retired(identities))
        admin, _ = users.add_user('Administración sintética', PermissionLevel.ADMIN)
        users.add_user('Persona de prueba', PermissionLevel.ADULT)
        unsafe_name, _ = users.add_user('Prueba <img src=x onerror=window.__personXss=true>', PermissionLevel.ADULT)
        authority = DashboardAuthority(directory / "auth.db", session_ttl_s=120, account_provider=users.active_accounts)
        await authority.initialize()
        root = authority.resolve("Bearer synthetic-nexo-bootstrap")
        assert root is not None
        home = await authority.issue_credential(root, subject="synthetic-home", role=DashboardRole.HOME, expires_at=time.time() + 180)
        operator = await authority.issue_credential(root, subject="synthetic-operator", role=DashboardRole.OPERATOR, expires_at=time.time() + 180)
        selection = EvaluationSample("vf_" + "a" * 64, frozenset({"b" * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), time.time() + 180)
        evaluator = await authority.issue_credential(root, subject="synthetic-evaluator", role=DashboardRole.EVALUATOR, expires_at=time.time() + 180, samples=(selection,))
        rooms = {
            "living": RoomConfig("living", "Sala de estar", "la sala", default_light="light.test_lamp", default_cover="cover.test_blind"),
            "kitchen": RoomConfig("kitchen", "Cocina", "la cocina", default_climate="climate.test_ac"),
        }
        dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms, user_manager=users)

        async def unavailable_live_channel(websocket):
            # This fixture has household services only; private live services are exercised separately.
            await websocket.close(code=1013)

        for route in ['/ws/voices', '/ws/evaluation']:
            dashboard.app.router.routes.insert(0, WebSocketRoute(route, unavailable_live_channel))

        @dashboard.app.middleware("http")
        async def record_request(request, call_next):
            if request.url.path.startswith("/api/"):
                requests.append({"method": request.method, "path": request.url.path,
                                 "request_id": request.query_params.get("request_id")})
            return await call_next(request)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level="warning"))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        dashboard._server, dashboard._serve_task = server, serving
        browser = None
        try:
            for _ in range(150):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            assert server.started
            profile = directory / "chromium"
            with (evidence / "chromium-first.log").open("w") as output:
                browser = await asyncio.to_thread(subprocess.Popen, [
                    "chromium", "--headless", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--no-first-run", "--no-default-browser-check", "--disable-background-networking", "--disable-component-update",
                    "--disable-sync", "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank",
                ], stdout=output, stderr=output)
                active_port = profile / "DevToolsActivePort"
                for _ in range(200):
                    if active_port.exists():
                        break
                    assert browser.poll() is None, "Chromium exited; see private browser log"
                    await asyncio.sleep(0.05)
                debugging = int(active_port.read_text().splitlines()[0])
                async with aiohttp.ClientSession() as client:
                    async with client.get(f"http://127.0.0.1:{debugging}/json/list") as response:
                        targets = await response.json()
                target = next(target for target in targets if target["type"] == "page")
                async with connect(target["webSocketDebuggerUrl"], max_size=None) as connection:
                    page = DiagnosticPage(connection)
                    await page.command("Runtime.enable")
                    await page.command("Page.enable")
                    await page.command("Page.addScriptToEvaluateOnNewDocument", {"source": INSTRUMENT})
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 1425, "height": 1180, "deviceScaleFactor": 1, "mobile": False})
                    await page.command("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                    await page.wait_for("document.querySelector('#login-panel') && document.querySelector('#login-error').textContent.length>0")
                    assert [request["path"] for request in requests] == ["/api/admin/auth/whoami"]

                    marker = len(requests)
                    await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true})")
                    await login(page, home.token, "Acceso de hogar")
                    await asyncio.sleep(0.1)
                    assert not any(request['path'] == '/api/home/catalog' for request in requests[marker:])
                    await page.evaluate("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for("document.querySelector('#room-list [data-room-id=living]') && document.querySelector('#room-controls [data-operation=turn_on]')")
                    await page.wait_for("document.querySelector('#map-loading').hidden")
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(node=>node.dataset.view).join(',')==='home,actions'")
                    assert not any(request["path"] in {"/api/users", "/api/voices", "/api/home/microphones", "/api/conversations"} for request in requests[marker:])
                    await screenshot(page, evidence / "home-desktop.png")
                    webgl = await page.evaluate("!!document.querySelector('#house-canvas canvas')")

                    await page.evaluate("document.querySelector('#room-controls button[data-operation=turn_on]').click()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=turn_on]').dataset.status==='confirmed'")
                    assert len(ha.sends) == 1 and ha.sends[0][2] == "light.test_lamp"
                    await page.evaluate("(()=>{const details=document.querySelector('#room-controls .advanced-control');details.open=true;const input=details.querySelector('[name=color_temp_kelvin]');input.value='3000';input.closest('form').requestSubmit()})()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=color_temp_kelvin]').dataset.status==='confirmed'")
                    assert ha.sends[-1][3] == {'color_temp_kelvin': 3000}
                    await page.evaluate("document.querySelector('#layer-tabs [data-layer=climate]').click();document.querySelector('#room-list [data-room-id=kitchen]').click()")
                    await page.wait_for("document.querySelector('#room-controls input[name=temperature]')")
                    assert await page.evaluate("document.querySelector('#room-controls .device-value').textContent==='22 °C · objetivo'")
                    await page.evaluate("(()=>{const input=document.querySelector('#room-controls input[name=temperature]');input.focus();input.value='23.5';document.querySelector('#refresh').click()})()")
                    ha.states['climate.test_ac']['attributes']['temperature'] = 21
                    await page.wait_for("document.querySelector('#room-controls .device-value').textContent==='21 °C · objetivo'")
                    assert await page.evaluate("document.activeElement.name==='temperature' && document.activeElement.value==='23.5'")
                    await page.evaluate("(()=>{const input=document.querySelector('#room-controls input[name=temperature]');input.value='23.5';input.closest('form').requestSubmit()})()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=set_temperature]').dataset.status==='confirmed'")
                    assert ha.sends[-1][2:] == ("climate.test_ac", {"temperature": 23.5})
                    assert ha.states["light.test_lamp"]["state"] == "on"
                    await screenshot(page, evidence / "climate-desktop.png")

                    before = sum(request["path"] == "/api/home/actions" and request["method"] == "POST" for request in requests)
                    await page.evaluate("(()=>{window.__qaDropNext=true;window.__qaDelayReceipt=true;const input=document.querySelector('#room-controls input[name=temperature]');input.value='24';input.closest('form').requestSubmit()})()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=set_temperature]').dataset.status==='confirmed' && !window.__qaDropNext")
                    await asyncio.sleep(0.2)
                    assert sum(request["path"] == "/api/home/actions" and request["method"] == "POST" for request in requests) == before + 1

                    await page.evaluate("document.querySelector('#view-top').click()")
                    assert await page.evaluate("document.querySelector('#view-top').classList.contains('active') && !document.querySelector('#view-3d').classList.contains('active') && document.querySelector('#view-top').getAttribute('aria-pressed')==='true'")
                    await page.evaluate("document.querySelector('#reset-view').click()")
                    assert await page.evaluate("document.querySelector('#view-3d').classList.contains('active') && document.querySelector('#view-3d').getAttribute('aria-pressed')==='true'")
                    assert any(request["path"] == "/api/home/actions" and request["method"] == "GET" and request["request_id"] for request in requests)
                    await page.evaluate("document.querySelector('#refresh').click()")
                    await asyncio.sleep(0.1)
                    assert sum(request["path"] == "/api/home/actions" and request["method"] == "POST" for request in requests) == before + 1

                    await page.evaluate("document.querySelector('#app-nav [data-view=actions]').click()")
                    await page.wait_for("!document.querySelector('#actions-view').hidden")
                    assert await page.evaluate("!document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    await page.evaluate("history.back()")
                    await page.wait_for("!document.querySelector('#home-view').hidden && document.querySelector('#map-loading').hidden")
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
                    await asyncio.sleep(0.2)
                    assert await page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
                    assert await page.evaluate("getComputedStyle(document.querySelector('#session-label')).display!=='none'")
                    await screenshot(page, evidence / "home-mobile.png", full=False)
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 320, "height": 760, "deviceScaleFactor": 1, "mobile": True})
                    await asyncio.sleep(0.1)
                    assert await page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")

                    await page.evaluate("window.__qaHoldLogout=true;document.querySelector('#logout').click()")
                    await page.wait_for("!!window.__qaReleaseLogout")
                    assert await page.evaluate("document.querySelector('#nexo-workspace').hidden && !document.querySelector('#room-controls').children.length && !document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    logins = sum(request['path'] == '/api/admin/auth/login' for request in requests)
                    await page.evaluate("document.querySelector('#token').value=" + json.dumps(operator.token) + ";document.querySelector('#login-form').requestSubmit()")
                    await asyncio.sleep(0.1)
                    assert sum(request['path'] == '/api/admin/auth/login' for request in requests) == logins
                    await page.evaluate("window.__qaHoldLogout=false;window.__qaReleaseLogout()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    marker = len(requests)
                    # This household-only fixture also covers hidden evaluator bootstrap.
                    await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:true})")
                    await login(page, evaluator.token, "Evaluación temporal")
                    await page.wait_for("!document.querySelector('#evaluation-view').hidden && !!document.querySelector('#app-nav [data-view=evaluation]')")
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].every(node=>node.dataset.view==='evaluation') && !document.querySelector('#house-canvas canvas')")
                    assert not any(request["path"].startswith('/api/home/') or request["path"] in {"/api/users", "/api/voices", "/api/conversations"} for request in requests[marker:])
                    await page.evaluate("document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    await page.evaluate("delete document.hidden")
                    await login(page, operator.token, "Administración")
                    await page.wait_for("document.querySelector('#app-nav [data-view=voices]') && document.querySelector('#room-list').children.length>0")
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].map(node=>node.dataset.view).join(',')==='home,voices,people,actions,access'")
                    assert not any(request['path'] == '/api/admin/users' for request in requests)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.evaluate("document.querySelector('#app-nav [data-view=people]').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3")
                    assert await page.evaluate("!document.querySelector('#people-view img') && !window.__personXss && document.querySelector('#people-grid').textContent.includes('<img')")
                    assert await page.evaluate("!document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    await page.evaluate("window.__qaReleasePeople=null;window.__qaHoldPeople=true;document.querySelector('#refresh-people').click()")
                    await page.wait_for("typeof window.__qaReleasePeople==='function'")
                    users.update_user(unsafe_name.user_id, name='Álex de prueba')
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelector('#people-grid').textContent.includes('Álex de prueba')")
                    await page.evaluate('window.__qaReleasePeople()')
                    assert await page.evaluate("!document.querySelector('#people-grid').textContent.includes('<img')")
                    marker = len(requests)
                    await page.evaluate("window.__qaHoldPeopleMutation=true;document.querySelector('#add-person').click();document.querySelector('#person-form [name=personName]').value='Persona creada en navegador';document.querySelector('#person-form').requestSubmit()")
                    await page.wait_for("typeof window.__qaReleasePeopleMutation==='function'")
                    await page.evaluate("document.querySelector('#app-nav [data-view=actions]').click();document.querySelector('#app-nav [data-view=people]').click()")
                    assert await page.evaluate("document.querySelector('#add-person').disabled && document.querySelector('#people-status').textContent.includes('Esperando')")
                    assert sum(item['path'] == '/api/admin/users' and item['method'] == 'POST' for item in requests[marker:]) == 1
                    await page.evaluate('window.__qaReleasePeopleMutation()')
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===4 && document.querySelector('#person-detail h3').textContent==='Persona creada en navegador'")
                    created = users.get_user_by_name('Persona creada en navegador')
                    assert created is not None
                    assert sum(item['path'] == '/api/admin/users' and item['method'] == 'POST' for item in requests[marker:]) == 1
                    await page.evaluate("window.__qaDropPeopleMutation=true;document.querySelector('#person-detail .identity-text-link').click();document.querySelector('#person-form [name=permission_level]').value='child';document.querySelector('#person-form').requestSubmit()")
                    await page.wait_for("document.querySelector('#people-status').textContent.includes('No se confirmó el resultado')")
                    assert await page.evaluate("document.querySelector('#add-person').disabled && !document.querySelectorAll('#people-grid [data-person]').length")
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===4")
                    await page.evaluate("[...document.querySelectorAll('#people-grid [data-person]')].find(button=>button.textContent.includes('Persona creada en navegador')).click()")
                    await page.wait_for("document.querySelector('#person-detail .person-role-info')?.textContent.includes('Niño/a')")
                    assert users.get_user(created.user_id).permission_level is PermissionLevel.CHILD
                    assert sum(item['method'] == 'PUT' for item in requests[marker:]) == 1
                    personal = await authority.issue_credential(root, subject='synthetic-personal-persona', role=DashboardRole.HOME,
                        expires_at=time.time() + 120, user_id=created.user_id)
                    personal_session = await authority.login(personal.token)
                    assert personal_session.lease.principal.role is DashboardRole.HOME
                    await page.evaluate("document.querySelector('#person-detail .identity-danger').click()")
                    await page.wait_for("document.querySelector('#person-dialog').open")
                    assert not any(item['method'] == 'DELETE' for item in requests[marker:])
                    assert personal_session.lease.current()
                    await page.evaluate("document.querySelector('#person-dialog .identity-secondary').click();document.querySelector('#person-detail .identity-danger').click();(()=>{const button=document.querySelector('#confirm-person-removal');button.click();button.click()})()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3 && document.querySelector('#people-status').textContent.includes('Persona retirada')")
                    assert users.get_user(created.user_id) is None
                    assert authority.resolve(personal.token) is None and not personal_session.lease.current()
                    assert sum(item['method'] == 'DELETE' for item in requests[marker:]) == 1
                    await page.evaluate("document.querySelector('[data-person=' + CSS.escape(" + json.dumps(admin.user_id) + ") + ']').click();document.querySelector('#person-detail .identity-danger').click();document.querySelector('#confirm-person-removal').click()")
                    await page.wait_for("document.querySelector('#people-status').textContent.includes('último administrador')")
                    assert users.get_user(admin.user_id) is not None
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3")
                    await screenshot(page, evidence / 'people-desktop.png')
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 390, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                    await screenshot(page, evidence / 'people-mobile.png', full=False)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 320, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.evaluate("document.querySelector('#add-person').click()")
                    assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth && document.querySelector("#person-dialog").getBoundingClientRect().right<=window.innerWidth')
                    await page.evaluate("document.querySelector('#person-form [name=personName]').value='Borrador preservado';window.__qaPeopleUnavailable=true;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for("document.querySelector('#person-dialog .person-dialog-status').textContent.includes('no está disponible')")
                    assert await page.evaluate("document.querySelector('#person-dialog').open && document.querySelector('#person-form button[type=submit]').disabled && document.querySelector('#person-form [name=personName]').value==='Borrador preservado'")
                    await page.evaluate("document.querySelector('#person-dialog .icon-button').click()")
                    assert await page.evaluate("document.querySelector('#add-person').disabled && !document.querySelectorAll('#people-grid [data-person]').length && !document.querySelector('#people-grid').textContent.includes('Todavía no hay')")
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3")
                    await page.evaluate("document.querySelector('#add-person').click();document.querySelector('#person-form [name=personName]').value='Borrador privado sintético';window.__qaReleasePeople=null;window.__qaHoldPeople=true;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for("typeof window.__qaReleasePeople==='function'")
                    await page.evaluate("document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    assert await page.evaluate("!document.querySelector('#person-dialog').open && !document.querySelector('#person-dialog').children.length && !document.querySelector('#people-view').children.length")
                    await login(page, home.token, 'Acceso de hogar')
                    await page.evaluate('window.__qaReleasePeople()')
                    await page.wait_for("document.querySelector('#room-list').children.length>0")
                    assert await page.evaluate("document.querySelector('#session-label').textContent==='Acceso de hogar' && !document.querySelector('#people-view').children.length && !document.querySelector('#app-nav [data-view=people]')")
                    await page.evaluate("document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    await login(page, operator.token, 'Administración')
                    await page.evaluate("window.__qaFailLogout=true;document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent.includes('No pudimos confirmar')")
                    assert await page.evaluate("document.querySelector('#nexo-workspace').hidden && !document.querySelector('#room-controls').children.length")
                    marker = len(requests)
                    await page.command("Page.reload")
                    await page.wait_for("document.querySelector('#login-error').textContent.includes('cierre anterior')")
                    assert len(requests) == marker
                    await page.evaluate("document.querySelector('#retry-logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    short = await authority.issue_credential(root, subject="synthetic-short", role=DashboardRole.HOME, expires_at=time.time() + 3)
                    await login(page, short.token, "Acceso de hogar")
                    await page.wait_for("document.querySelector('#room-list').children.length>0")
                    await page.wait_for("document.querySelector('#nexo-workspace').hidden")
                    assert await page.evaluate("!document.querySelector('#room-controls').children.length && window.__qaFrames.size===0")
                    assert page.errors == [], page.errors
                    result = {"passed": True, "synthetic_fixture": True, "actual_chromium_http_login": True,
                              "roles": ["home", "evaluator", "operator"], "home_effects": len(ha.sends),
                              "webgl_illustration": webgl, "individual_sessions": True,
                              "checks": ["bootstrap no private prefetch", "hidden bootstrap sends no catalog", "role permissions before requests", "independent light/climate typed effects", "focused drafts preserve current observed labels", "lost POST and in_progress receipt reconcile by poll without repost", "camera visual and ARIA agree", "visible model between renders", "navigation releases renderer/RAF", "320px and 390px no horizontal overflow", "logout clears before pending HTTP and blocks crossed login", "failed logout stays hidden across reload", "session expires before fresh catalog"],
                              "people_checks": ['real CRUD once per gesture', 'mutation stays pending across leave/reentry without repost', 'crossed GET keeps latest result', 'lost mutation receipt requires GET without repost', 'account level does not select dashboard role', 'cancelled confirmation has no effect', 'deletion retires every personal session', 'last admin conflict', '503 is not empty catalog', 'open dialog reports failed refresh and disables submit while preserving draft', 'names are DOM text', '320/390 reflow', 'logout clears dialog/draft/accounts before late GET under another login'],
                              "screenshots": ["home-desktop.png", "climate-desktop.png", "home-mobile.png", "people-desktop.png", "people-mobile.png"]}
                    (evidence / "browser-result.json").write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps(result))
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                await asyncio.to_thread(browser.wait, 10)
            await dashboard.stop()
            await serving
            await authority.aclose()


if __name__ == "__main__":
    asyncio.run(main())
