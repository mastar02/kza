"""Real Chromium, individual HTTP logins, synthetic HA and shared notices for NEXO."""

import argparse
import asyncio
import base64
from dataclasses import replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from unittest.mock import patch

import aiohttp
import uvicorn
from starlette.routing import WebSocketRoute
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import (
    DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample,
)
from src.home_assistant import capability_service
from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
from src.home_assistant.effect_plans import scene_effects_from_states
from src.home_assistant.home_actions import HomeActionService
from src.reminders.reminder_scheduler import ReminderScheduler, notice_body, notice_request_id
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.rooms.room_context import RoomConfig
from src.users.user_manager import PermissionLevel, UserManager
from tests.frontend.browser_support import (
    assert_frame_consistent, assert_nexo_presentation,
    Page, audit_current_accessibility, audit_modal_focus, audit_modal_return, audit_status, audit_view,
    browser_binary,
    browser_profile_root, dispatch_key, evidence_directory, press_enter, press_escape, press_key, reduced_motion_house,
)
from tests.integration.test_typed_home_operations import typed_graph


# Keep UI audits independent of host scheduling. The held-read scenario
# advances the browser clock past this real deadline and checks expiry explicitly.
CATALOG_TTL_S = 60

INSTRUMENT = """(() => {
  const wallTime = Date.now;
  window.__qaCatalogClockOffset = 0;
  Date.now = () => wallTime() + window.__qaCatalogClockOffset;
  const fetcher=window.fetch.bind(window);
  window.__qaDropNext=false;window.__qaDelayReceipt=false;window.__qaFailLogout=false;window.__qaHoldLogout=false;
  window.__qaLogin503=false;window.__qaLogin403=false;window.__qaLoginNetwork=false;window.__qaLoginTimeout=false;
  window.fetch=async(input,options={})=>{
    const path=String(input);
    if(path==='/api/admin/auth/login'&&window.__qaLogin503){window.__qaLogin503=false;return new Response(JSON.stringify({ok:false,detail:'dashboard_authority_unavailable'}),{status:503,headers:{'Content-Type':'application/json'}});}
    if(path==='/api/admin/auth/login'&&window.__qaLogin403){window.__qaLogin403=false;return new Response(JSON.stringify({detail:'origin denied'}),{status:403,headers:{'Content-Type':'application/json'}});}
    if(path==='/api/admin/auth/login'&&window.__qaLoginNetwork){window.__qaLoginNetwork=false;throw new TypeError('Failed to fetch');}
    if(path==='/api/admin/auth/login'&&window.__qaLoginTimeout){window.__qaLoginTimeout=false;throw new DOMException('The operation timed out.','TimeoutError');}
    if(path==='/api/admin/auth/logout'&&window.__qaFailLogout)throw new TypeError('Synthetic offline logout');
    if(path==='/api/admin/auth/logout'&&window.__qaHoldLogout)await new Promise(resolve=>window.__qaReleaseLogout=resolve);
    if(path==='/api/home/catalog'&&window.__qaHoldCatalog){window.__qaHoldCatalog=false;await new Promise(resolve=>window.__qaReleaseCatalog=resolve);}
    if(path==='/api/reminders/notices'&&window.__qaHoldNotices){window.__qaHoldNotices=false;await new Promise(resolve=>window.__qaReleaseNotices=resolve);}
    if(path==='/api/reminders/notices'&&window.__qaFailNotices)throw new TypeError('Synthetic offline notices');
    const evaluatorToken=path==='/api/reminders/notices/dismiss'&&window.__qaNoticeEvaluatorToken;
    if(evaluatorToken){
      delete window.__qaNoticeEvaluatorToken;
      // This intercept runs after the app's whoami and before the POST reaches the server.
      const login=await fetcher('/api/admin/auth/login',{method:'POST',credentials:'same-origin',cache:'no-store',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({token:evaluatorToken})});
      if(!login.ok)throw new Error('Synthetic evaluator cookie switch failed');
    }
    const response=await fetcher(input,options);
    if(path==='/home/nexo/private/casa.glb'&&window.__qaHoldModel){
      window.__qaHoldModel=false;
      await new Promise(resolve=>window.__qaReleaseModel=resolve);
    }
    if(evaluatorToken)window.__qaNoticeWriteStatus=response.status;
    if(path==='/api/reminders/notices'&&window.__qaHoldNoticeResponse){
      window.__qaHoldNoticeResponse=false;
      await response.clone().arrayBuffer();
      await new Promise(resolve=>window.__qaReleaseNoticeResponse=resolve);
    }
    if(path==='/api/reminders/notices/dismiss'&&window.__qaDropNoticeDismiss){window.__qaDropNoticeDismiss=false;await response.text();throw new TypeError('Synthetic lost notice receipt');}
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


# The two fixture rooms named Cocina remain distinct by their ordered rows.
LIVING_CHOICE = '#room-list .room-row:nth-child(1) .room-choice'
KITCHEN_CHOICE = '#room-list .room-row:nth-child(2) .room-choice'
KITCHEN_POWER = '#room-list .room-row:nth-child(2) .room-power'
ANNEX_CHOICE = '#room-list .room-row:nth-child(4) .room-choice'
KITCHEN_PIN = "[...document.querySelectorAll('#room-pins .room-pin')].find(pin=>pin.textContent==='Cocina')"

HOME_CONTROLS = (
    ('#section-views [data-view=home]', None), ('#section-views [data-view=actions]', None), ('#refresh', None),
    ('#view-3d', None), ('#view-top', None), ('#zoom-in', None), ('#zoom-out', None), ('#reset-view', None),
    ('#rotate-left', None), ('#rotate-right', None),
    ('#expand-map', None), ('#global-scenes button', 'Lectura'), ('#layer-tabs [data-layer=lights]', None),
    (LIVING_CHOICE, None),
    (KITCHEN_POWER, None),
    ('#room-controls input[name=temperature]', None), ('#room-controls .thermostat-step', None), ('#logout', None),
)
# U4: no surface links to the retired legacy views or the technical panel.
LEGACY_LINKS = ("document.querySelectorAll('a[href*=\"legacy.html\"], a[href^=\"/obs\"], a[href=\"/index.html\"], "
                "a[href=\"/dashboard.js\"], #technical-link, #additional-tools').length")
# Entity controls are gated by expiring observations: audit them only while the page shows data as current.
HOME_FRESH = ("!document.querySelector('#connection-status').classList.contains('is-stale') && "
              "[...document.querySelectorAll('.device-card')].every(card => card.dataset.observation === 'current')")
# Records every entity card that turns stale; without a held request, any entry is a refresh-timing flicker.
STALE_MONITOR = ("(()=>{window.__qaStale=[];new MutationObserver(records=>{for(const record of records){const card=record.target;"
                 "if(card.dataset.observation==='stale'&&record.oldValue!=='stale')window.__qaStale.push({device:card.querySelector('.device-name')?.textContent,"
                 "at:Math.round(performance.now()),status:document.querySelector('#connection-status').textContent})}})"
                 ".observe(document.body,{subtree:true,attributes:true,attributeFilter:['data-observation'],attributeOldValue:true});return true})()")
ACTION_CONTROLS = (('#section-views [data-view=home]', None), ('#action-list button', None), ('#refresh', None), ('#logout', None))
PEOPLE_CONTROLS = (('#app-nav [data-view=people]', None), ('#add-person', None), ('#refresh-people', None),
                   ('#people-grid [data-person]', None), ('#logout', None))
NOTICE_LIST = '/api/reminders/notices'
NOTICE_DISMISS = NOTICE_LIST + '/dismiss'
NOTICES_READY = "document.querySelector('#home-notices')?.dataset.noticeState==='ready'"
NOTICES_EMPTY = "document.querySelector('#home-notices')?.dataset.noticeState==='empty'"
PENDING_DISMISS = "#home-notices [data-notice-status=pending] [data-notice-dismiss]"
UNCERTAIN_DISMISS = "#home-notices [data-notice-status=uncertain] [data-notice-dismiss]"
PRIVATE_NOTICE = 'AVISO_PRIVADO_SINTETICO'


async def screenshot(page: Page, path: Path, *, full: bool = True, scroll_top: bool = True) -> None:
    await assert_nexo_presentation(page)
    if scroll_top:
        await page.evaluate('window.scrollTo(0,0)')
    await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
    if await page.evaluate("!!document.querySelector('#house-canvas canvas') && !document.querySelector('#house-canvas + .house-fallback')"):
        assert await page.evaluate("(()=>{const canvas=document.querySelector('#house-canvas canvas'),gl=canvas.getContext('webgl2'),pixel=new Uint8Array(4);gl.readPixels(Math.floor(canvas.width/2),Math.floor(canvas.height/2),1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);return pixel[3]>0})()"), 'The house plan must remain visible between renders'
    result = await page.command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": full})
    path.write_bytes(base64.b64decode(result["data"]))
    path.chmod(0o600)


async def check_floor_pixels(page: Page) -> dict:
    """Measure two interior floor patches in the initial desktop 3D view in both themes."""
    readings = {}
    for theme in ('claro', 'oscuro'):
        readings[theme] = await page.evaluate(r"""(async theme => {
            const selector = document.querySelector('#theme-slot select');
            selector.value = theme;
            selector.dispatchEvent(new Event('change'));
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            const canvas = document.querySelector('#house-canvas canvas');
            const gl = canvas.getContext('webgl2');
            const style = getComputedStyle(canvas.closest('.map-viewport'));
            const [center, edge] = ['--nexo-segment', '--nexo-ground'].map(property => {
                const rgb = parseInt(style.getPropertyValue(property).trim().slice(1), 16);
                return [rgb >> 16, (rgb >> 8) & 255, rgb & 255];
            });
            const pixel = new Uint8Array(4);
            // Rectangles over the near left wood floor and the living-room tiles.
            const patches = [
                {x: [.375, .392, .409], y: [.655, .675, .695]},
                {x: [.56, .58, .60], y: [.71, .725, .74]},
            ];
            return patches.map(({x, y}) => {
                let luminance = 0, visible = 0, distanceFromBackdrop = 0;
                for (const u of x) for (const v of y) {
                    gl.readPixels(Math.floor(u * canvas.width), Math.floor((1 - v) * canvas.height),
                        1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
                    const luma = .2126 * pixel[0] + .7152 * pixel[1] + .0722 * pixel[2];
                    luminance += luma;
                    if (pixel[3] >= 250 && luma >= 32) visible++;
                    // Same ellipse as the CSS backdrop, from its center to the 75% stop.
                    const mix = Math.min(1, Math.hypot(u - .5, v - .5) * Math.SQRT2 / .75);
                    distanceFromBackdrop += Math.hypot(...center.map((channel, index) =>
                        channel + (edge[index] - channel) * mix - pixel[index]));
                }
                const count = x.length * y.length;
                return {mean_luminance: luminance / count, visible, count, mean_backdrop_distance: distanceFromBackdrop / count};
            });
        })(""" + json.dumps(theme) + ")")
    await page.evaluate("localStorage.removeItem('nexo-theme')")
    assert all(
        patch['mean_luminance'] >= 60 and patch['visible'] >= 7 and patch['mean_backdrop_distance'] >= 15
        for patches in readings.values() for patch in patches
    ), f'3D floor must be lit and distinguishable from both theme backdrops: {readings}'
    return readings


async def check_context_recovery(page: Page, evidence: Path) -> None:
    """Lose the real WebGL context with a loaded model and with a held response."""
    paused = """(() => {
        const status = document.querySelector('#house-canvas + .house-fallback[role=status]');
        return status?.textContent.includes('El plano 3D se pausó') && status.getBoundingClientRect().height > 0
            && document.querySelector('#room-pins').hidden
            && ['view-3d', 'view-top', 'zoom-in', 'zoom-out', 'rotate-left', 'rotate-right', 'reset-view']
                .every(id => document.getElementById(id).disabled)
            && document.querySelector('#house-canvas canvas').getContext('webgl2').isContextLost();
    })()"""
    ready = """!document.querySelector('#house-canvas + .house-fallback') && !document.querySelector('#room-pins').hidden
        && !document.querySelector('#view-top').disabled"""
    for model_held in (False, True):
        if model_held:
            await page.evaluate("document.querySelector('#section-views [data-view=actions]').click()")
            await page.wait_for("!document.querySelector('#actions-view').hidden && !document.querySelector('#house-canvas canvas')")
            await page.evaluate("window.__qaHoldModel=true;document.querySelector('#section-views [data-view=home]').click()")
            await page.wait_for("!!window.__qaReleaseModel && document.querySelector('#room-pins').children.length===0")
        assert await page.evaluate("""(() => {
            const gl = document.querySelector('#house-canvas canvas').getContext('webgl2');
            window.__qaContext = gl.getExtension('WEBGL_lose_context');
            if (!window.__qaContext) return false;
            window.__qaContext.loseContext();
            return true;
        })()"""), 'Chromium must expose WEBGL_lose_context'
        await page.wait_for(paused)
        if model_held:
            await page.evaluate("window.__qaReleaseModel();delete window.__qaReleaseModel")
            await page.wait_for("document.querySelector('#room-pins').children.length===3")
            assert await page.evaluate(paused), 'A model loaded during context loss must leave the camera disabled'
        else:
            tree = await page.command('Accessibility.getFullAXTree')
            nodes = {node['nodeId']: node for node in tree['nodes']}
            assert any(
                not node['ignored'] and node.get('role', {}).get('value') == 'status'
                and any('El plano 3D se pausó' in nodes[child].get('name', {}).get('value', '')
                        for child in node.get('childIds', []) if child in nodes)
                for node in nodes.values()
            ), 'The paused viewer must expose its message in the browser accessibility tree'
            await screenshot(page, evidence / 'hogar-contexto-perdido.png')
        await page.evaluate("window.__qaContext.restoreContext();delete window.__qaContext")
        await page.wait_for(ready)


async def check_mobile_pins(page: Page) -> dict:
    """Check that mobile pin targets fit without overlap at normal and double text size."""
    layouts = await page.evaluate(r"""(async () => {
        const container = document.querySelector('#room-pins');
        const settle = () => new Promise(resolve => setTimeout(resolve, 120));
        const measure = () => {
            const bounds = container.getBoundingClientRect();
            const pins = [...container.querySelectorAll('.room-pin')];
            const rects = pins.map(pin => pin.getBoundingClientRect());
            const targets = pins.map((pin, i) => {
                const {left, right, top, bottom} = rects[i];
                // Projection adds a subpixel translation. Measure the same target without
                // that translation so DOMRect float cancellation cannot turn 44px into 43.9999px.
                const translation = pin.style.translate;
                pin.style.translate = 'none';
                const {width, height} = pin.getBoundingClientRect();
                pin.style.translate = translation;
                const name = document.createRange();
                name.selectNodeContents(pin.querySelector('.tag').firstChild);
                return {room: pin.textContent, x: left - bounds.left, y: top - bounds.top, width, height,
                    name_lines: name.getClientRects().length,
                    inside: left >= bounds.left && right <= bounds.right && top >= bounds.top && bottom <= bounds.bottom,
                    visible: getComputedStyle(pin).visibility === 'visible'};
            });
            const overlap = rects.some((a, i) => rects.slice(i + 1)
                .some(b => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom));
            return {targets, overlap};
        };
        await settle();
        const normal = measure();
        const sizes = [...container.querySelectorAll('*')].map(node => {
            const style = getComputedStyle(node);
            return [node, parseFloat(style.fontSize), style.lineHeight];
        });
        let doubled;
        try {
            for (const [node, font, line] of sizes) {
                node.style.fontSize = 2 * font + 'px';
                if (line.endsWith('px')) node.style.lineHeight = 2 * parseFloat(line) + 'px';
            }
            await settle();
            doubled = measure();
        } finally {
            for (const [node] of sizes) {
                node.style.removeProperty('font-size');
                node.style.removeProperty('line-height');
            }
            await settle();
        }
        return {normal, doubled};
    })()""")
    for layout in layouts.values():
        assert len(layout['targets']) == 3 and not layout['overlap'], layouts
        assert all(pin['visible'] and pin['inside'] and pin['width'] >= 44 and pin['height'] >= 44
                   for pin in layout['targets']), layouts
    assert next(pin for pin in layouts['normal']['targets'] if pin['room'] == 'Escalera')['name_lines'] == 1, layouts
    return layouts


async def check_notices(page: Page, evidence: Path, store: ReminderStore, requests: list[dict]) -> dict:
    """Exercise real shared HTTP reads, committed dismissals and degraded states at 390px."""
    await page.wait_for(NOTICES_EMPTY)
    await audit_status(page, evidence, 'avisos', 'empty', '#home-notices .notice-status')
    now = time.time()
    long_notice = 'Aviso compartido <img src=x> ' + 'reflow' * 24
    pending = await store.create('synthetic-owner', long_notice, now - 10, scope=ReminderScope.SHARED)
    uncertain = await store.create('synthetic-owner', 'Entrega cortada de prueba', now - 8, scope=ReminderScope.SHARED)
    await store.create('synthetic-owner', PRIVATE_NOTICE, now - 4, scope=ReminderScope.PRIVATE)
    key = notice_request_id(uncertain.id, uncertain.trigger_at, 1)
    assert await store.claim_home_action(key, notice_body(uncertain.id, uncertain.trigger_at, 1, now - 4, 'living'))
    await store.finish_home_action(key, 'uncertain', 'delivery_cancelled')
    assert await store.mark_fired(uncertain.id)

    await page.evaluate("document.querySelector('#home-notices button').click()")
    await page.wait_for(f"{NOTICES_READY} && document.querySelectorAll('#home-notices .notice-row').length===2")
    assert await page.evaluate(
        "(()=>{const panel=document.querySelector('#home-notices');"
        "return panel.textContent.includes(" + json.dumps(long_notice) + ")&&!panel.querySelector('img')"
        "&&!panel.textContent.includes(" + json.dumps(PRIVATE_NOTICE) + ")"
        "&&panel.querySelectorAll('time[datetime]').length===2"
        "&&Math.abs(Date.parse(panel.querySelector('[data-notice-status=pending] time').dateTime)-"
        + json.dumps(pending.trigger_at * 1000) + ")<1"
        "&&panel.textContent.includes('Ambiente no informado')&&panel.textContent.includes('Entrega incierta · Cortado')})()"
    )
    audit = await audit_view(page, evidence, 'avisos', (
        ('#home-notices button', 'Actualizar avisos'), (PENDING_DISMISS, 'Descartar'), (UNCERTAIN_DISMISS, 'Descartar'),
    ), screenshot, widths=(390,), restore=(390, 844, True), fresh=NOTICES_READY)

    def post_count() -> int:
        return sum(item['method'] == 'POST' and item['path'] == NOTICE_DISMISS for item in requests)

    control = 'document.querySelector(' + json.dumps(PENDING_DISMISS) + ')'
    before = post_count()
    await page.evaluate(
        f"window.__qaHoldNotices=true;document.querySelector('#home-notices button').click();{control}.focus()"
    )
    await page.wait_for("typeof window.__qaReleaseNotices==='function'")
    await page.wait_for("document.querySelector('#home-notices').dataset.noticeState==='stale'")
    assert await page.evaluate(f"document.activeElement==={control} && {control}.getAttribute('aria-disabled')==='true'")
    await press_enter(page)
    assert post_count() == before
    await audit_status(page, evidence, 'avisos', 'stale', '#home-notices .notice-status')
    await page.evaluate("window.__qaFailNotices=true;window.__qaReleaseNotices()")
    await page.wait_for("document.querySelector('#home-notices .notice-status').textContent.includes('contactar')")
    assert await page.evaluate("document.querySelector('#home-notices .notice-status').textContent.includes('desactualizada')")
    await audit_status(page, evidence, 'avisos', 'offline', '#home-notices .notice-status')
    await page.evaluate("window.__qaFailNotices=false;document.querySelector('#home-notices button').click()")
    await page.wait_for(f"{NOTICES_READY} && {control}.getAttribute('aria-disabled')==='false'")
    assert await page.evaluate(f"document.activeElement==={control}")
    await press_enter(page)
    await page.wait_for("document.querySelector('#home-notices .notice-feedback').textContent==='Aviso descartado.'")
    assert await page.evaluate(f"!{control} && document.activeElement===document.querySelector('#home-notices button')")
    await audit_status(page, evidence, 'avisos', 'dismissed', '#home-notices .notice-feedback')
    assert (await store.get_by_id(pending.id)).state is ReminderState.FIRED
    assert post_count() == before + 1

    uncertain_control = 'document.querySelector(' + json.dumps(UNCERTAIN_DISMISS) + ')'
    await page.wait_for(f"{NOTICES_READY} && {uncertain_control}.getAttribute('aria-disabled')==='false'")
    await page.evaluate(f"window.__qaDropNoticeDismiss=true;{uncertain_control}.focus()")
    await press_enter(page)
    await page.wait_for(f"{NOTICES_EMPTY} && document.querySelector('#home-notices .notice-feedback').dataset.kind==='uncertain'")
    assert await page.evaluate("document.querySelector('#home-notices .notice-feedback').textContent.includes('No se pudo confirmar')")
    assert await store.get_home_action_receipt(key) is None
    assert post_count() == before + 2
    return audit


# T042-U5 Q24: computed text and field floors; neither text nor field values are serialized.
Q24_WIDTHS = (320, 390, 1440)
Q24_PROBE = """(()=>{
  const shown=el=>el.getClientRects().length>0&&el.checkVisibility({visibilityProperty:true,opacityProperty:true});
  const name=el=>el.localName+(el.id?'#'+el.id:'')+(typeof el.className==='string'&&el.className.trim()?'.'+el.className.trim().split(/\\s+/).join('.'):'');
  const text=[],fields=[];let textNodes=0,fieldCount=0;
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  for(let node;node=walker.nextNode();){
    const el=node.parentElement;
    if(!el||!node.textContent.trim()||el.closest('script,style,noscript,svg,[aria-hidden=true]')||!shown(el))continue;
    textNodes++;const size=parseFloat(getComputedStyle(el).fontSize);
    if(size<14)text.push({element:name(el),size});
  }
  for(const el of document.querySelectorAll('input,select,textarea')){
    if(el.matches('[type=hidden],[type=button],[type=submit],[type=file]')||!shown(el))continue;
    fieldCount++;const size=parseFloat(getComputedStyle(el).fontSize);
    if(size<16)fields.push({element:name(el),size});
  }
  return {textNodes,fieldCount,text:text.slice(0,15),fields:fields.slice(0,15)};
})()"""


async def audit_q24(page: Page, evidence: Path, label: str, *, restore: tuple[int, int, bool] = (1425, 1180, False)) -> dict:
    """Assert the 14 px text floor and 16 px mobile field floor."""
    report, problems = {}, []
    for width in Q24_WIDTHS:
        await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': width <= 390})
        await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
        await asyncio.sleep(0.1)
        found = await page.evaluate(Q24_PROBE)
        if label == 'hogar':
            fit = await page.evaluate("""(() => {
                const bad = [];
                const buttons = [...document.querySelectorAll('#room-controls .touch-preset')];
                if (!buttons.length) bad.push('Faltan los controles rápidos de clima');
                for (const button of buttons) {
                    const rect = button.getBoundingClientRect();
                    const range = document.createRange();
                    range.selectNodeContents(button);
                    if ([...range.getClientRects()].some(text => text.left < rect.left || text.right > rect.right
                        || text.top < rect.top || text.bottom > rect.bottom)) bad.push(button.textContent);
                }
                const stats = document.querySelector('.house-stats');
                stats.scrollIntoView({block:'center'});
                for (const stat of stats.querySelectorAll('dt, dd')) {
                    const rect = stat.getBoundingClientRect();
                    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    if (hit !== stat) bad.push(stat.textContent);
                }
                return bad;
            })()""")
            problems += [f'{width}px overlapping or missing Casa content: {item}' for item in fit]
            found['casa_fit'] = fit
        report[str(width)] = found
        assert found['textNodes'] > 0, f'Q24 {label} {width}px: the probe saw no text'
        problems += [f'{width}px text {item["size"]}px: {item["element"]}' for item in found['text']]
        if width <= 390:
            problems += [f'{width}px field {item["size"]}px: {item["element"]}' for item in found['fields']]
    await page.command('Emulation.setDeviceMetricsOverride', {'width': restore[0], 'height': restore[1], 'deviceScaleFactor': 1, 'mobile': restore[2]})
    (evidence / f'q24-{label}.json').write_text(json.dumps({'view': label, 'widths': report, 'problems': problems}, indent=2) + '\n')
    assert not problems, f'Q24 {label} floors violated: ' + '; '.join(problems[:10])
    return report


ROTATE_STEP = math.pi / 12
HOUSE_YAW = "parseFloat(document.querySelector('#house-canvas').dataset.yaw)"


async def check_live_map(page: Page) -> None:
    """Select a bound room with a pin and the floor, using actual pointer events."""
    await page.wait_for("document.querySelectorAll('#room-pins button').length===3")
    assert await page.evaluate(f"""[...document.querySelectorAll('#room-pins button')].every(pin=>{{
      const r=pin.getBoundingClientRect();
      return pin.type==='button'&&pin.tabIndex===-1&&r.width>=44&&r.height>=44&&!pin.querySelector('img');
    }}) && !document.querySelector('#live-badge').hidden
       && document.querySelector('{ANNEX_CHOICE} strong').textContent==='Cocina'
       && [...document.querySelectorAll('#room-pins button')].map(pin=>pin.textContent).sort().join() === 'Cocina,Escalera,Sala de estar'""")
    await page.evaluate("document.querySelector('#map-stage').scrollIntoView({block:'center'})")
    point = await page.evaluate(f"""(()=>{{const r=({KITCHEN_PIN}).getBoundingClientRect();
      return {{x:r.left+r.width/2,y:r.top+r.height/2}}}})()""")
    mouse = {'button': 'left', 'clickCount': 1, **point}
    await page.command('Input.dispatchMouseEvent', {'type': 'mousePressed', 'buttons': 1, **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'buttons': 0, **mouse})
    await page.wait_for(f"document.querySelector('{KITCHEN_CHOICE}').getAttribute('aria-pressed')==='true'")

    await page.evaluate(f"""document.querySelector('{LIVING_CHOICE}').click();
      document.querySelector('#view-top').click()""")
    await asyncio.sleep(1.2)
    point = await page.evaluate(f"""(()=>{{const pins=document.querySelector('#room-pins');
      const r=({KITCHEN_PIN}).getBoundingClientRect();
      pins.style.visibility='hidden';return {{x:r.left+r.width/2,y:r.top+r.height/2}}}})()""")
    mouse = {'button': 'left', 'clickCount': 1, **point}
    await page.command('Input.dispatchMouseEvent', {'type': 'mousePressed', 'buttons': 1, **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'buttons': 1, **{**mouse, 'x': point['x'] + 5}})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'buttons': 1, **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'buttons': 0, **mouse})
    assert await page.evaluate(f"document.querySelector('{LIVING_CHOICE}').getAttribute('aria-pressed')==='true'")
    await page.command('Input.dispatchMouseEvent', {'type': 'mousePressed', 'buttons': 1, **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'buttons': 0, **mouse})
    await page.wait_for(f"document.querySelector('{KITCHEN_CHOICE}').getAttribute('aria-pressed')==='true'")
    await page.evaluate(f"""document.querySelector('#room-pins').style.visibility='';
      document.querySelector('{LIVING_CHOICE}').click();document.querySelector('#reset-view').click()""")


async def _key(page: Page, key: str, code: str, key_code: int, text: str | None = None) -> None:
    extra = {'text': text} if text else {}
    await dispatch_key(page, 'keyDown' if text else 'rawKeyDown', key, code, key_code, **extra)
    await dispatch_key(page, 'keyUp', key, code, key_code)


async def audit_rotation(page: Page, evidence: Path, remount) -> dict:
    """Q26 (WCAG 2.5.7): rotate buttons are a single-pointer alternative to dragging the 3D model."""
    async def yaw() -> float:
        return await page.evaluate(HOUSE_YAW)

    async def press(selector: str, how: str) -> None:
        assert await page.evaluate(f"(()=>{{const el=document.querySelector({json.dumps(selector)});el.focus();return document.activeElement===el}})()"), selector
        if how == 'Enter':
            await press_enter(page)
        else:
            await _key(page, ' ', 'Space', 32, text=' ')
        await asyncio.sleep(0.15)

    observed = {}
    await page.evaluate("document.querySelector('#reset-view').click()")
    await asyncio.sleep(0.15)
    home_yaw = await yaw()
    # Targets: named, enabled and at least 44 x 44 CSS px on phone and desktop.
    for width, mobile in ((320, True), (1425, False)):
        await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': mobile})
        await asyncio.sleep(0.15)
        for selector, label in (('#rotate-left', 'Rotar a la izquierda'), ('#rotate-right', 'Rotar a la derecha')):
            box = await page.evaluate(f"(()=>{{const el=document.querySelector('{selector}'),r=el.getBoundingClientRect();return {{w:r.width,h:r.height,disabled:el.disabled,name:el.getAttribute('aria-label'),type:el.type,visible:el.checkVisibility()}}}})()")
            assert box['w'] >= 44 and box['h'] >= 44 and box['visible'] and not box['disabled'], (width, selector, box)
            assert box['name'] == label and box['type'] == 'button', box
    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
    await asyncio.sleep(0.15)
    # Tab order follows the visual order: zoom, rotate left, rotate right, reset.
    await page.evaluate("document.querySelector('#zoom-out').focus()")
    order = []
    for _ in range(3):
        await _key(page, 'Tab', 'Tab', 9)
        order.append(await page.evaluate('document.activeElement.id'))
    assert order == ['rotate-left', 'rotate-right', 'reset-view'], order
    # Enter and Space turn the yaw by one fixed step each way.
    for selector, how, steps in (('#rotate-left', 'Enter', 1), ('#rotate-left', 'Space', 2), ('#rotate-right', 'Enter', 1), ('#rotate-right', 'Space', 0)):
        await press(selector, how)
        got = await yaw()
        assert abs(got - (home_yaw + steps * ROTATE_STEP)) < 1e-3, (selector, how, got, home_yaw, steps)
    observed['keyboard'] = {'order': order, 'step_rad': ROTATE_STEP}
    # Dragging still works and reset restores the default view.
    center = await page.evaluate("(()=>{const r=document.querySelector('#house-canvas canvas').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()")
    mouse = {'y': center['y'], 'button': 'left', 'clickCount': 1}
    await page.command('Input.dispatchMouseEvent', {'type': 'mousePressed', 'buttons': 1, 'x': center['x'], **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'buttons': 1, 'x': center['x'] + 50, **mouse})
    await page.command('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'buttons': 0, 'x': center['x'] + 50, **mouse})
    await asyncio.sleep(0.15)
    dragged = await yaw()
    assert abs(dragged - (home_yaw - 50 * 0.006)) < 1e-3, ('drag', dragged, home_yaw)
    await press('#reset-view', 'Enter')
    assert abs(await yaw() - home_yaw) < 1e-3
    observed['drag_yaw'] = dragged
    # Reduced motion applies the step at once; default motion still eases it.
    for preference in ('reduce', 'no-preference'):
        await page.command('Emulation.setEmulatedMedia', {'features': [{'name': 'prefers-reduced-motion', 'value': preference}]})
        await page.evaluate("document.querySelector('#reset-view').click()")
        await asyncio.sleep(1.2)
        await page.evaluate("window.__kzaFrame0=document.querySelector('#house-canvas canvas').toDataURL();document.querySelector('#rotate-left').click()")
        await asyncio.sleep(0.15)
        await page.evaluate("window.__kzaFrame1=document.querySelector('#house-canvas canvas').toDataURL()")
        await asyncio.sleep(1.2)
        result = await page.evaluate("({moved:window.__kzaFrame1!==window.__kzaFrame0,settledEarly:window.__kzaFrame1===document.querySelector('#house-canvas canvas').toDataURL()})")
        observed[preference] = result
        if preference == 'reduce':
            assert result['moved'] and result['settledEarly'], f'Q26 reduced motion: the rotation must render at once, without easing {result}'
        else:
            assert not result['settledEarly'], f'Q26 default motion: the rotation is expected to ease {result}'
    await page.command('Emulation.setEmulatedMedia', {'features': []})
    await page.evaluate("document.querySelector('#reset-view').click()")
    # Expanded map on short phones: all six tools keep a 44 px box inside the clipped stage and pass a real hit test.
    observed['expanded_tools'] = {}
    for width, height in ((320, 520), (390, 520), (390, 540)):
        await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': True})
        await page.evaluate("document.querySelector('#expand-map').click()")
        await asyncio.sleep(0.2)
        tools = await page.evaluate("""(()=>{const stage=document.querySelector('#map-stage').getBoundingClientRect();
          return ['zoom-in','zoom-out','rotate-left','rotate-right','reset-view','expand-map'].map(id=>{
            const el=document.getElementById(id),r=el.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2,hit=document.elementFromPoint(x,y);
            const w=Math.min(r.right,stage.right)-Math.max(r.left,stage.left),h=Math.min(r.bottom,stage.bottom)-Math.max(r.top,stage.top);
            return {id,w:r.width,h:r.height,visibleW:w,visibleH:h,hit:!!hit&&(hit===el||el.contains(hit))}})})()""")
        observed['expanded_tools'][f'{width}x{height}'] = tools
        await page.evaluate("document.querySelector('#expand-map').click()")
        for tool in tools:
            assert tool['w'] >= 44 and tool['h'] >= 44 and tool['visibleW'] >= 44 and tool['visibleH'] >= 44 and tool['hit'], (width, height, tool)
    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
    await asyncio.sleep(0.15)
    # Without WebGL the tools are disabled, not left as dead controls.
    await page.evaluate("window.__qaGetContext=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(){return null}")
    await page.evaluate("document.querySelector('#section-views [data-view=actions]').click()")
    await page.wait_for("!document.querySelector('#actions-view').hidden && !document.querySelector('#house-canvas canvas')")
    await page.evaluate("document.querySelector('#section-views [data-view=home]').click()")
    await page.wait_for("!document.querySelector('#home-view').hidden && document.querySelector('#map-loading').hidden && !!document.querySelector('#house-canvas + .house-fallback')")
    fallback = await page.evaluate("""(()=>{const state=Object.fromEntries(['rotate-left','rotate-right'].map(id=>{
      const el=document.querySelector('#'+id);el.focus();el.click();return [id,{disabled:el.disabled,focused:document.activeElement===el}]}));
      return {state,canvas:!!document.querySelector('#house-canvas canvas'),yaw:document.querySelector('#house-canvas').dataset.yaw??null}})()""")
    assert fallback['canvas'] is False and fallback['yaw'] is None, fallback
    assert all(item['disabled'] and not item['focused'] for item in fallback['state'].values()), fallback
    observed['fallback'] = fallback
    await page.evaluate("HTMLCanvasElement.prototype.getContext=window.__qaGetContext;delete window.__qaGetContext")
    await remount()
    (evidence / 'q26-rotation.json').write_text(json.dumps(observed, indent=2) + '\n')
    return observed


async def check_without_plan(page: Page, evidence: Path, token: str, ha) -> None:
    """Casa keeps its room choices and device actions when the model is missing."""
    await page.wait_for("document.querySelector('#login-error')?.textContent.length > 0"
                        " && !document.querySelector('#token').disabled")
    await login(page, token, "Acceso de hogar")
    await page.wait_for("""document.querySelector('#map-loading').hidden
        && document.querySelector('#house-canvas + .house-fallback[role=status]')?.textContent
           === 'No se pudo mostrar el plano. Podés usar la lista de ambientes.'
        && document.querySelector('#room-list').children.length === 4""")
    assert await page.evaluate("""!document.querySelector('#house-canvas canvas')
        && !document.querySelector('#room-pins').children.length
        && ['view-3d', 'view-top', 'zoom-in', 'zoom-out', 'rotate-left', 'rotate-right', 'reset-view']
            .every(id => document.getElementById(id).disabled)""")
    for selector, heading in ((KITCHEN_CHOICE, "Cocina"), (LIVING_CHOICE, "Sala de estar")):
        await page.evaluate(f"document.querySelector('{selector}').focus()")
        await press_enter(page)
        await page.wait_for(
            "document.querySelector('#room-controls .selected-room-heading')?.textContent==="
            + json.dumps(heading)
        )
    await page.evaluate("document.querySelector('#room-controls button[data-operation=turn_on]').click()")
    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=turn_on]')?.dataset.status==='confirmed'")
    assert len(ha.sends) == 1 and ha.sends[0][1:3] == ('turn_on', 'light.test_lamp')

    screenshots = []
    for theme in ('claro', 'oscuro'):
        await page.evaluate(f"""(() => {{
            const select = document.querySelector('#theme-slot select');
            select.value = '{theme}'; select.dispatchEvent(new Event('change'));
        }})()""")
        for width, height in ((1440, 1000), (768, 1024), (390, 844)):
            await page.command('Emulation.setDeviceMetricsOverride', {
                'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': width == 390,
            })
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            name = f'hogar-sin-plano-{theme}-{width}.png'
            await screenshot(page, evidence / name)
            screenshots.append(name)
    assert page.errors == [], page.errors
    result = {"passed": True, "webgl_house": False, "room_choices": 4, "confirmed_light_actions": 1,
              "uncaught_exceptions": len(page.errors), "screenshots": screenshots}
    (evidence / "browser-result.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


async def main(*, without_plan: bool = False, missing_model: bool = False) -> None:
    os.environ["KZA_DASHBOARD_TOKEN"] = "synthetic-nexo-bootstrap"
    evidence = evidence_directory("KZA_NEXO_HOME_EVIDENCE", "data/autonomous-roadmap-20261102/t047/people-browser-runs")
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
    binary = browser_binary()
    browser_data = browser_profile_root(mode=0o700)
    ha, actions = typed_graph.__wrapped__()
    read_ha = ha.get_state_snapshot

    async def catalog_fixture_snapshot():
        snapshot = await read_ha()
        return replace(snapshot, expires_at=snapshot.observed_at + timedelta(seconds=CATALOG_TTL_S))

    ha.get_state_snapshot = catalog_fixture_snapshot
    for entity, name in [("light.test_lamp", "Luz principal"), ("climate.test_ac", "Aire acondicionado")]:
        ha.states[entity]["attributes"]["friendly_name"] = name
    ha.states['light.test_lamp']['attributes'].update(brightness=166, color_temp_kelvin=2700)
    ha.states['climate.test_ac']['attributes']['hvac_action'] = 'cooling'
    ha.states['sensor.test_temperature'] = {'entity_id': 'sensor.test_temperature', 'state': '24.5',
                                           'attributes': {'unit_of_measurement': '°C'}}
    reading_state = {'entity_id': 'scene.test_reading', 'state': 'unknown',
                     'attributes': {'entity_id': ['light.test_lamp']}}
    grants = await actions.policy.snapshot_grants()
    reading = EntityGrant('scene.test_reading', frozenset({'activate_scene'}), display_name='Lectura',
                          scene_effects=scene_effects_from_states({'light.test_lamp': {'state': 'on'}}))
    actions = HomeActionService(ha, policy=HomeControlPolicy((*grants, reading)))
    requests = []
    with (tempfile.TemporaryDirectory(prefix="kza-nexo-", dir=browser_data) as temporary,
          patch.object(capability_service, 'STATE_OBSERVATION_TTL_S', CATALOG_TTL_S)):
        directory = Path(temporary)
        users = UserManager(str(directory / 'users.json'), retirement_proof=lambda identities: authority.users_retired(identities))
        admin, _ = users.add_user('Administración sintética', PermissionLevel.ADMIN)
        users.add_user('Persona de prueba', PermissionLevel.ADULT)
        unsafe_name, _ = users.add_user('Prueba <img src=x onerror=window.__personXss=true>', PermissionLevel.ADULT)
        authority = DashboardAuthority(directory / "auth.db", session_ttl_s=300, account_provider=users.active_accounts)
        await authority.initialize()
        root = authority.resolve("Bearer synthetic-nexo-bootstrap")
        assert root is not None
        home = await authority.issue_credential(root, subject="synthetic-home", role=DashboardRole.HOME, expires_at=time.time() + 600)
        operator = await authority.issue_credential(root, subject="synthetic-operator", role=DashboardRole.OPERATOR, expires_at=time.time() + 600)
        selection = EvaluationSample("vf_" + "a" * 64, frozenset({"b" * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), time.time() + 600)
        evaluator = await authority.issue_credential(root, subject="synthetic-evaluator", role=DashboardRole.EVALUATOR, expires_at=time.time() + 600, samples=(selection,))
        rooms = {
            "living": RoomConfig("living", "Sala de estar", "la sala", default_light="light.test_lamp",
                                 default_cover="cover.test_blind", temperature_sensor="sensor.test_temperature"),
            "kitchen": RoomConfig("kitchen", "Cocina", "la cocina", default_climate="climate.test_ac"),
            "escalera": RoomConfig("escalera", "Escalera", "la escalera"),
            "annex": RoomConfig("annex", "Cocina", "el anexo"),
        }
        notice_store = ReminderStore(str(directory / 'reminders.db'))
        await notice_store.initialize()
        reminder_scheduler = ReminderScheduler(notice_store, None, None, ha, home_actions=actions)
        dashboard = DashboardAPI(authority=authority, ha_client=ha, home_actions=actions, home_rooms=rooms,
                                 user_manager=users, reminder_scheduler=reminder_scheduler)

        async def unavailable_live_channel(websocket):
            # This fixture has household services only; private live services are exercised separately.
            await websocket.close(code=1013)

        for route in ['/ws/voices', '/ws/evaluation']:
            dashboard.app.router.routes.insert(0, WebSocketRoute(route, unavailable_live_channel))

        action_posts = []

        @dashboard.app.middleware("http")
        async def record_request(request, call_next):
            if request.method == "POST" and request.url.path == "/api/home/actions":
                body = await request.json()
                action_posts.append({key: value for key, value in body.items() if key != "request_id"})
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
                    binary, "--headless", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
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
                    if without_plan or missing_model:
                        for name, status in (('casa.glb', 404), ('house-plan.js', 200 if missing_model else 404)):
                            async with client.get(f'http://127.0.0.1:{port}/home/nexo/private/{name}') as unauth:
                                assert unauth.status == 401, (name, unauth.status)
                            async with client.get(
                                f'http://127.0.0.1:{port}/home/nexo/private/{name}',
                                headers={'Authorization': f'Bearer {home.token}'},
                            ) as response:
                                assert response.status == status, (name, response.status)
                target = next(target for target in targets if target["type"] == "page")
                async with connect(target["webSocketDebuggerUrl"], max_size=None) as connection:
                    page = DiagnosticPage(connection)
                    await page.command("Runtime.enable")
                    await page.command("Page.enable")
                    await page.command("Page.addScriptToEvaluateOnNewDocument", {"source": INSTRUMENT})
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 1425, "height": 1180, "deviceScaleFactor": 1, "mobile": False})
                    await page.command("Emulation.setEmulatedMedia", {
                        "features": [{"name": "prefers-color-scheme", "value": "dark"}],
                    })
                    await page.command("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                    if without_plan or missing_model:
                        await check_without_plan(page, evidence, home.token, ha)
                        return
                    await page.wait_for("document.querySelector('#login-panel') && document.querySelector('#login-error').textContent.length>0")
                    assert await page.evaluate("""document.documentElement.dataset.theme === 'oscuro'
                        && document.body.classList.contains('night')
                        && document.querySelector('#theme-slot select')?.value === 'auto'
                        && getComputedStyle(document.documentElement).getPropertyValue('--nexo-ground').trim() === '#14171A'""")
                    assert [request["path"] for request in requests] == ["/api/admin/auth/whoami"]
                    assert await page.evaluate("document.querySelector('#token').getAttribute('aria-describedby')==='login-guidance' && document.querySelector('#login-guidance').textContent.includes('administrador del hogar') && document.querySelector('#login-guidance').textContent.includes('evaluación temporal') && document.querySelector('#login-guidance').textContent.includes('Home Assistant no funciona')")
                    await page.evaluate("document.querySelector('.skip-link').focus()")
                    await press_enter(page)
                    assert await page.evaluate("document.activeElement.id === 'main'"), 'Enter on the skip link must reach main'
                    await page.evaluate("history.replaceState({}, '', location.pathname); document.activeElement.blur()")
                    for theme in ('claro', 'oscuro'):
                        await page.evaluate("(value=>{const select=document.querySelector('#theme-slot select');"
                                            "select.value=value;select.dispatchEvent(new Event('change'))})(" + json.dumps(theme) + ")")
                        for width in (1440, 768, 390):
                            await page.command('Emulation.setDeviceMetricsOverride',
                                               {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': width == 390})
                            await screenshot(page, evidence / f'login-{theme}-{width}.png', full=False)
                            if width != 768:
                                await audit_current_accessibility(page, evidence, 'sesion')
                            await page.evaluate("document.querySelector('#help-button').focus()")
                            await press_enter(page)
                            assert await page.evaluate("document.querySelector('#info-dialog').open")
                            await screenshot(page, evidence / f'help-{theme}-{width}.png', full=False)
                            if width != 768:
                                focus = await audit_current_accessibility(
                                    page, evidence, 'ayuda', '#info-dialog', initial='#dialog-close')
                                await press_escape(page)
                                await page.wait_for("!document.querySelector('#info-dialog').open")
                                await audit_modal_return(page, evidence, f'ayuda-retorno-{theme}-{width}', focus,
                                                         dialog='#info-dialog', opener='#help-button')
                            else:
                                await page.evaluate("document.querySelector('#dialog-done').click()")
                                assert await page.evaluate("!document.querySelector('#info-dialog').open")
                    await page.command('Emulation.setDeviceMetricsOverride',
                                       {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    await screenshot(page, evidence / 'login-guidance.png')
                    await audit_q24(page, evidence, 'sesion')

                    # T045-U3: login error classification in real Chromium (401, 403, 503, timeout, network).
                    login_error_checks = {}
                    raw_english = ("Failed to fetch", "AbortError", "TimeoutError", "NetworkError", "TypeError", "origin denied")
                    for label, hook, token_value, required_terms, forbidden_terms in [
                        ("401", "", "synthetic-invalid-credential", ("credencial", "válida"), ()),
                        ("503", "window.__qaLogin503=true;", home.token, ("no está disponible",), ("credencial", "inválida")),
                        ("403", "window.__qaLogin403=true;", home.token, ("origen",), ("credencial", "inválida")),
                        ("timeout", "window.__qaLoginTimeout=true;", home.token, ("tiempo",), ("credencial", "inválida")),
                        ("network", "window.__qaLoginNetwork=true;", home.token, ("conectar",), ("credencial", "inválida")),
                    ]:
                        await page.evaluate(
                            f"document.querySelector('#login-error').textContent='';{hook}"
                            f"document.querySelector('#token').value={json.dumps(token_value)};"
                            f"document.querySelector('#login-form').requestSubmit()"
                        )
                        await page.wait_for("document.querySelector('#login-error').textContent.length>0 && !document.querySelector('#token').disabled")
                        await assert_nexo_presentation(page)
                        msg = await page.evaluate("document.querySelector('#login-error').textContent")
                        assert await page.evaluate("document.querySelector('#nexo-workspace').hidden && !document.querySelector('#login-panel').hidden && document.querySelector('#token').value===''"), label
                        for term in required_terms:
                            assert term in msg.lower(), f"login {label} missing {term!r}: {msg!r}"
                        for term in (*forbidden_terms, *raw_english):
                            assert term.lower() not in msg.lower(), f"login {label} leaked {term!r}: {msg!r}"
                        login_error_checks[label] = msg

                    marker = len(requests)
                    await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,get:()=>true})")
                    await login(page, home.token, "Acceso de hogar")
                    await asyncio.sleep(0.1)
                    assert not any(request['path'] == '/api/home/catalog' for request in requests[marker:])
                    assert not any(request['path'] == NOTICE_LIST for request in requests[marker:])
                    assert await page.evaluate("document.querySelector('#global-scenes').hidden")
                    await page.evaluate("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for(f"document.querySelector('{LIVING_CHOICE}') && document.querySelector('#room-controls [data-operation=turn_on]')")
                    await page.wait_for("document.querySelector('#map-loading').hidden")
                    assert await page.evaluate(r"""(() => {
                        const view = document.querySelector('#home-view');
                        const panel = view.querySelector('#room-controls');
                        return panel.querySelector('.selected-room-heading').textContent === 'Sala de estar'
                            && panel.querySelector('.room-temperature').textContent.includes('24.5 °C')
                            && panel.querySelector('.rows .row[data-domain=cover] .device-name').textContent === 'Persiana · Sala de estar'
                            && panel.querySelector('.rows .row[data-domain=light] .sw[aria-checked=false]')
                            && view.querySelector('#global-scenes').hidden
                            && !document.querySelector('.hero, .weather, .catalog-note, footer')
                            && !/tu casa, en sintonía|control local|ilustración|tu espacio\. tu control/i.test(document.body.innerText)
                            && !/\b(?:light|climate|cover|scene|sensor)\.test_/i.test(view.innerText)
                            && ![...view.querySelectorAll('[aria-label]')].some(node => /\.test_/.test(node.getAttribute('aria-label')));
                    })()""")
                    ha.states['scene.test_reading'] = reading_state
                    await page.evaluate("document.querySelector('#refresh').click()")
                    await page.wait_for("!document.querySelector('#global-scenes').hidden")
                    assert await page.evaluate("""[...document.querySelectorAll('#global-scenes button')].map(button=>button.textContent).join() === 'Lectura'""")
                    assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(node=>node.dataset.view).join(',')==='home,routines,tasks,activity'")
                    assert await page.evaluate("""(() => {
                        const nav = document.querySelector('#app-nav');
                        return document.querySelectorAll('nav').length === 1
                            && nav.contains(document.querySelector('#theme-slot'))
                            && nav.querySelector('#rail-nav [aria-current=true]')?.textContent === 'Casa'
                            && [...nav.querySelectorAll('#section-views button')].map(node => node.textContent).join() === 'Vista de la casa,Acciones'
                            && nav.querySelector('#section-views [aria-current=page]')?.dataset.view === 'home';
                    })()""")
                    assert not any(request["path"] in {"/api/users", "/api/voices", "/api/home/microphones", "/api/conversations"} for request in requests[marker:])
                    await screenshot(page, evidence / "home-desktop.png")
                    webgl = await page.evaluate("!!document.querySelector('#house-canvas canvas')")
                    assert webgl, 'The live pin and raycast require WebGL in this harness'
                    floor_pixels = await check_floor_pixels(page)
                    await check_live_map(page)

                    await page.evaluate("document.querySelector('#room-controls button[data-operation=turn_on]').click()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=turn_on]').dataset.status==='confirmed'")
                    assert len(ha.sends) == 1 and ha.sends[0][2] == "light.test_lamp"
                    await page.evaluate("(()=>{const details=document.querySelector('#room-controls .advanced-control');details.open=true;const input=details.querySelector('[name=color_temp_kelvin]');input.value='3000';input.closest('form').requestSubmit()})()")
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=color_temp_kelvin]').dataset.status==='confirmed'")
                    assert ha.sends[-1][3] == {'color_temp_kelvin': 3000}
                    ha.states['light.test_lamp']['attributes'].pop('brightness')
                    await page.wait_for("document.querySelector('#room-list [role=switch][data-operation=turn_off]')?.getAttribute('aria-checked')==='true'")
                    await page.evaluate("document.querySelector('#room-list [role=switch][data-operation=turn_off]').click()")
                    await page.wait_for("""document.querySelector('#room-list [role=switch][data-operation=turn_on]')?.getAttribute('aria-checked')==='false'
                      && document.querySelector('#room-controls .device-value').textContent==='Apagada'""")
                    assert ha.sends[-1][1:3] == ('turn_off', 'light.test_lamp')
                    assert await page.evaluate("""(()=>{const slider=document.querySelector('#room-controls input[name=brightness_pct]');
                      return slider.type==='range' && slider.value===slider.min && slider.closest('label').querySelector('output').textContent===slider.min+' %';})()""")
                    await page.evaluate("""(()=>{const slider=document.querySelector('#room-controls input[name=brightness_pct]');
                      slider.value='34';slider.focus();})()""")
                    await press_key(page, 'ArrowRight', 39)
                    await page.wait_for("document.querySelector('#room-controls [data-receipt][data-operation=brightness]').dataset.status==='confirmed'")
                    assert ha.sends[-1][2:] == ('light.test_lamp', {'brightness_pct': 35})
                    await page.evaluate("document.querySelector('#global-scenes [data-operation=activate_scene]').click()")
                    await page.wait_for("document.querySelector('#global-scenes [data-receipt]').dataset.status==='confirmed'")
                    assert ha.sends[-1][1:3] == ('turn_on', 'light.test_lamp')
                    await page.evaluate(f"document.querySelector('#layer-tabs [data-layer=climate]').click();document.querySelector('{KITCHEN_CHOICE}').click()")
                    await page.wait_for("document.querySelector('#room-controls input[name=temperature]')")
                    assert await page.evaluate("document.querySelector('#room-controls .device-value').textContent==='22 °C · objetivo'")
                    assert await page.evaluate(f"""document.querySelector('#room-controls .device-current').textContent==='31 °C · actual'
                      && document.querySelector('#room-controls .room-temperature').textContent.includes('Enfriando')
                      && ({KITCHEN_PIN})?.getAttribute('aria-pressed')==='true'""")
                    for label, target in (('+', 22.5), ('−', 22)):
                        await page.evaluate(f"[...document.querySelectorAll('#room-controls .thermostat-step')].find(button=>button.textContent==={json.dumps(label)}).click()")
                        await page.wait_for(f"document.querySelector('#room-controls .device-value').textContent==='{target} °C · objetivo'")
                        assert ha.sends[-1][2:] == ('climate.test_ac', {'temperature': target})
                    await page.evaluate("(()=>{const input=document.querySelector('#room-controls input[name=temperature]');input.focus();input.value='23.5';document.querySelector('#refresh').click()})()")
                    ha.states['climate.test_ac']['attributes']['temperature'] = 21
                    await page.wait_for("document.querySelector('#room-controls .device-value').textContent==='21 °C · objetivo'")
                    assert await page.evaluate("document.activeElement.name==='temperature' && document.activeElement.value==='23.5'")

                    # Genuine staleness: while a catalog request is held past the observation expiry, the
                    # climate controls are disabled and marked; fresh data re-enables them and returns focus and draft.
                    await page.evaluate("(()=>{window.__qaHeldInput=document.querySelector('#room-controls input[name=temperature]');window.__qaHoldCatalog=true;window.__qaHeldInput.focus();window.__qaHeldInput.value='22.5';document.querySelector('#refresh').click()})()")
                    await page.wait_for("!!window.__qaReleaseCatalog")
                    await page.evaluate(f"window.__qaCatalogClockOffset = {(CATALOG_TTL_S + 1) * 1000}")
                    stale_state = "(()=>{const input=window.__qaHeldInput,card=input.closest('.device-card');return input.isConnected&&input.disabled&&card.dataset.observation==='stale'&&[...card.querySelectorAll('[data-control]')].every(control=>control.disabled)&&document.querySelector('#connection-status').classList.contains('is-stale')})()"
                    for _ in range(240):
                        if await page.evaluate(stale_state):
                            break
                        await asyncio.sleep(0.05)
                    else:
                        raise AssertionError('held catalog did not expire the climate observation')
                    await audit_status(page, evidence, 'hogar', 'stale', '#connection-status')
                    held_focus = await page.evaluate("document.activeElement===window.__qaHeldInput?'retained':document.activeElement===document.body?'body':String(document.activeElement?.id||document.activeElement?.tagName)")
                    await page.evaluate("window.__qaCatalogClockOffset=0;window.__qaReleaseCatalog();delete window.__qaReleaseCatalog")
                    await page.wait_for("(()=>{const input=window.__qaHeldInput;return input.isConnected&&!input.disabled&&input.closest('.device-card').dataset.observation==='current'&&!document.querySelector('#connection-status').classList.contains('is-stale')})()")
                    assert await page.evaluate("document.activeElement===window.__qaHeldInput && window.__qaHeldInput.value==='22.5'"), 'focus and draft must return with fresh data'
                    staleness_check = {'fixture_ttl_s': CATALOG_TTL_S, 'clock_advance_s': CATALOG_TTL_S + 1,
                                       'focus_while_stale': held_focus, 'focus_restored': True, 'draft_preserved': True}
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

                    assert action_posts == [
                        {'entity_id': 'light.test_lamp', 'operation': 'turn_on'},
                        {'entity_id': 'light.test_lamp', 'operation': 'color_temp_kelvin', 'color_temp_kelvin': 3000},
                        {'entity_id': 'light.test_lamp', 'operation': 'turn_off'},
                        {'entity_id': 'light.test_lamp', 'operation': 'brightness', 'brightness_pct': 35},
                        {'entity_id': 'scene.test_reading', 'operation': 'activate_scene'},
                        *({'entity_id': 'climate.test_ac', 'operation': 'set_temperature', 'temperature': target}
                          for target in (22.5, 22, 23.5, 24)),
                    ]

                    await page.evaluate("document.querySelector('#view-top').click()")
                    assert await page.evaluate("document.querySelector('#view-top').classList.contains('active') && !document.querySelector('#view-3d').classList.contains('active') && document.querySelector('#view-top').getAttribute('aria-pressed')==='true'")
                    await page.evaluate("document.querySelector('#reset-view').click()")
                    assert await page.evaluate("document.querySelector('#view-3d').classList.contains('active') && document.querySelector('#view-3d').getAttribute('aria-pressed')==='true'")
                    assert any(request["path"] == "/api/home/actions" and request["method"] == "GET" and request["request_id"] for request in requests)
                    await page.evaluate("document.querySelector('#refresh').click()")
                    await asyncio.sleep(0.1)
                    assert sum(request["path"] == "/api/home/actions" and request["method"] == "POST" for request in requests) == before + 1

                    await page.evaluate("document.querySelector('#section-views [data-view=actions]').click()")
                    await page.wait_for("!document.querySelector('#actions-view').hidden")
                    assert await page.evaluate("!document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    assert await page.evaluate("!document.querySelector('#home-notices')")
                    assert await page.evaluate(STALE_MONITOR)
                    audits = {'acciones': await audit_view(page, evidence, 'acciones', ACTION_CONTROLS, screenshot, fresh=HOME_FRESH)}
                    await page.evaluate("history.back()")
                    await page.wait_for("!document.querySelector('#home-view').hidden && document.querySelector('#map-loading').hidden")

                    async def remount_house():
                        await page.evaluate("document.querySelector('#section-views [data-view=actions]').click()")
                        await page.wait_for("!document.querySelector('#actions-view').hidden && !document.querySelector('#house-canvas canvas')")
                        await page.evaluate("document.querySelector('#section-views [data-view=home]').click()")
                        await page.wait_for("!document.querySelector('#home-view').hidden && document.querySelector('#map-loading').hidden && !!document.querySelector('#house-canvas canvas') && !document.querySelector('#view-top').disabled")

                    motion = await reduced_motion_house(page, remount_house)
                    audits['hogar'] = await audit_view(page, evidence, 'hogar', HOME_CONTROLS, screenshot, widths=(320, 1280, 1425), fresh=HOME_FRESH, theme='oscuro')
                    stale_episodes = await page.evaluate("window.__qaStale")
                    (evidence / 'stale-episodes.json').write_text(json.dumps({'episodes': stale_episodes, 'staleness_check': staleness_check}, indent=2) + '\n')
                    assert stale_episodes == [], f'entity observations lapsed between catalog refreshes: {stale_episodes[:5]}'

                    # Supplemental captures check freshness per viewport without extending the functional monitor.
                    themed_audits = {}
                    mobile_pins = {}

                    async def themed_screenshot(page: Page, path: Path, *, full: bool = True) -> None:
                        await screenshot(page, path, full=full)
                        active_theme = await page.evaluate("document.documentElement.dataset.theme")
                        await assert_frame_consistent(page, '#home-view', active_theme)
                        if path.stem.endswith('-390'):
                            mobile_pins[active_theme] = await check_mobile_pins(page)

                    assert await page.evaluate("""(() => {
                        const selector = document.querySelector('#theme-slot select');
                        const panel = document.querySelector('#room-controls');
                        const palette = () => [getComputedStyle(panel).color, getComputedStyle(panel).backgroundColor];
                        const dark = palette();
                        selector.value = 'claro'; selector.dispatchEvent(new Event('change'));
                        const light = palette();
                        const day = document.documentElement.dataset.theme === 'claro'
                            && !document.body.classList.contains('night') && localStorage.getItem('nexo-theme') === 'claro';
                        selector.value = 'oscuro'; selector.dispatchEvent(new Event('change'));
                        localStorage.removeItem('nexo-theme');
                        return day && dark.every((color, index) => color !== light[index])
                            && palette().every((color, index) => color === dark[index]);
                    })()""")
                    for theme in ('oscuro', 'claro'):
                        await page.evaluate("""(theme => {
                            const selector = document.querySelector('#theme-slot select');
                            selector.value = theme; selector.dispatchEvent(new Event('change'));
                        })(""" + json.dumps(theme) + ")")
                        themed_audits[theme] = await audit_view(
                            page, evidence, 'hogar', HOME_CONTROLS, themed_screenshot,
                            widths=(1440, 768, 390), fresh=HOME_FRESH, theme=theme,
                        )
                    audits['hogar']['widths'].update(themed_audits['oscuro']['widths'])
                    (evidence / 'a11y-hogar.json').write_text(
                        json.dumps(audits['hogar'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    assert await page.evaluate("""(() => {
                        const selector = document.querySelector('#theme-slot select');
                        selector.value = 'oscuro'; selector.dispatchEvent(new Event('change'));
                        localStorage.removeItem('nexo-theme');
                        return document.documentElement.dataset.theme === 'oscuro'
                            && document.body.classList.contains('night');
                    })()""")
                    # Q26 takes several seconds without catalog refreshes, so it runs after the freshness monitor is closed.
                    await audit_q24(page, evidence, 'hogar')
                    rotation = await audit_rotation(page, evidence, remount_house)
                    await check_context_recovery(page, evidence)
                    await page.evaluate("document.querySelector('#section-views [data-view=actions]').click()")
                    await page.wait_for("!document.querySelector('#actions-view').hidden && document.querySelector('#action-list button')")
                    await audit_q24(page, evidence, 'acciones')
                    await remount_house()
                    legacy_links = {'home-1425': await page.evaluate(LEGACY_LINKS)}
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
                    await asyncio.sleep(0.2)
                    assert await page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
                    assert await page.evaluate("getComputedStyle(document.querySelector('#session-label')).display!=='none'")
                    await screenshot(page, evidence / "home-mobile.png", full=False)
                    prior_ha_effects = len(ha.sends)
                    audits['avisos'] = await check_notices(page, evidence, notice_store, requests)
                    assert len(ha.sends) == prior_ha_effects
                    await page.command("Emulation.setDeviceMetricsOverride", {"width": 320, "height": 760, "deviceScaleFactor": 1, "mobile": True})
                    await asyncio.sleep(0.1)
                    assert await page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")

                    late_notice = await notice_store.create('synthetic-owner', 'Aviso antes de cerrar sesión',
                                                            time.time() - 2, scope=ReminderScope.SHARED)
                    await page.evaluate("document.querySelector('#home-notices button').click()")
                    await page.wait_for(f"{NOTICES_READY} && document.querySelector('#home-notices').textContent.includes('Aviso antes de cerrar sesión')")
                    await page.evaluate("window.__qaHoldNoticeResponse=true;document.querySelector('#home-notices button').click()")
                    await page.wait_for("typeof window.__qaReleaseNoticeResponse==='function'")
                    await page.evaluate("window.__qaHoldLogout=true;document.querySelector('#logout').click()")
                    await page.wait_for("!!window.__qaReleaseLogout")
                    assert await page.evaluate("document.querySelector('#nexo-workspace').hidden && !document.querySelector('#room-controls').children.length && !document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    assert await page.evaluate("!document.querySelector('#home-notices') && !document.body.textContent.includes('Aviso antes de cerrar sesión')")
                    logins = sum(request['path'] == '/api/admin/auth/login' for request in requests)
                    await page.evaluate("document.querySelector('#token').value=" + json.dumps(operator.token) + ";document.querySelector('#login-form').requestSubmit()")
                    await asyncio.sleep(0.1)
                    assert sum(request['path'] == '/api/admin/auth/login' for request in requests) == logins
                    await page.evaluate("window.__qaHoldLogout=false;window.__qaReleaseLogout()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    marker = len(requests)
                    await page.evaluate("Object.defineProperty(document,'hidden',{configurable:true,value:false})")
                    await login(page, evaluator.token, "Evaluación temporal")
                    await page.wait_for("!document.querySelector('#evaluation-view').hidden && !!document.querySelector('#app-nav [data-view=evaluation]')")
                    assert await page.evaluate(r"""(() => {
                        const view = document.querySelector('#evaluation-view');
                        const fallback = view.querySelector('.page-head');
                        return fallback?.checkVisibility()
                            && fallback.innerText.trim().replace(/\s+/g, ' ') === 'Evaluación Sin lectura confirmada.'
                            && !!view.querySelector('#evaluation-content:empty')
                            && view.querySelectorAll('[role=status]').length === 1;
                    })()"""), 'Unconfirmed Evaluation must show no session or reading data'
                    assert await page.evaluate("[...document.querySelectorAll('#app-nav [data-view]')].every(node=>node.dataset.view==='evaluation') && !document.querySelector('#house-canvas canvas')")
                    await page.evaluate("new Promise(resolve=>{window.__qaReleaseNoticeResponse();requestAnimationFrame(()=>requestAnimationFrame(resolve))})")
                    assert await page.evaluate("document.hidden===false && document.querySelector('#home-view').hidden && !document.querySelector('#home-notices,.notice-row,[data-notice-dismiss]') && !document.body.textContent.includes('Aviso antes de cerrar sesión')")
                    assert not any(item['path'] in {NOTICE_LIST, NOTICE_DISMISS} for item in requests[marker:])
                    assert not any(request["path"].startswith('/api/home/') or request["path"] in {"/api/users", "/api/voices", "/api/conversations"} for request in requests[marker:])
                    assert not any(request["path"].startswith('/api/evaluation/') for request in requests[marker:])
                    await page.evaluate("document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    await login(page, home.token, "Acceso de hogar")
                    await page.wait_for(f"{NOTICES_READY} && !!document.querySelector('{PENDING_DISMISS}')")
                    marker = len(requests)
                    await page.evaluate(f"window.__qaNoticeEvaluatorToken={json.dumps(evaluator.token)};"
                                        f"document.querySelector('{PENDING_DISMISS}').click()")
                    await page.wait_for("window.__qaNoticeWriteStatus===403 && document.querySelector('#session-label').textContent==='Evaluación temporal'")
                    await page.wait_for("!document.querySelector('#evaluation-view').hidden")
                    assert sum(item['path'] == NOTICE_DISMISS and item['method'] == 'POST' for item in requests[marker:]) == 1
                    assert (await notice_store.get_by_id(late_notice.id)).state is ReminderState.ACTIVE
                    assert await notice_store.get_home_action_receipt(notice_request_id(late_notice.id, late_notice.trigger_at, 1)) is None
                    assert await page.evaluate("!document.querySelector('#home-notices,.notice-row,[data-notice-dismiss]')")
                    await page.evaluate("document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    await page.evaluate("delete document.hidden")
                    await login(page, operator.token, "Administración")
                    await page.wait_for("document.querySelector('#rail-nav [data-section=people]') && document.querySelector('#room-list').children.length>0")
                    await page.wait_for(f"{NOTICES_READY} && !!document.querySelector('{PENDING_DISMISS}')")
                    assert await page.evaluate("[...document.querySelectorAll('#rail-nav [data-view]')].map(node=>node.dataset.view).join(',')==='home,routines,tasks,activity,people,memory,access'")
                    assert not any(request['path'] == '/api/admin/users' for request in requests)
                    # D1: the technical panel stays reachable only by direct URL, never linked from NEXO.
                    assert await page.evaluate("document.querySelectorAll('a[href^=\"/obs\"], a[href*=\"//\"][href*=\"/obs\"], #technical-link').length===0")
                    for width in (320, 390, 768, 1024, 1280, 1440):
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': width < 768})
                        legacy_links[f'operator-{width}'] = await page.evaluate(LEGACY_LINKS)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 1425, 'height': 1180, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.evaluate("document.querySelector('#app-nav [data-view=people]').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3")
                    assert await page.evaluate("""(() => {
                        const tabs = document.querySelector('#section-views');
                        return [...tabs.children].map(tab => tab.textContent).join() === 'Lista de personas,Voces'
                            && tabs.querySelector('[aria-current=page]')?.dataset.view === 'people';
                    })()""")
                    assert await page.evaluate("!document.querySelector('#people-view img') && !window.__personXss && document.querySelector('#people-grid').textContent.includes('<img')")
                    assert await page.evaluate("!document.querySelector('#house-canvas canvas') && window.__qaFrames.size===0")
                    await page.evaluate("window.__qaReleasePeople=null;window.__qaHoldPeople=true;document.querySelector('#refresh-people').click()")
                    await page.wait_for("typeof window.__qaReleasePeople==='function'")
                    await audit_status(page, evidence, 'personas', 'loading', '#people-status')
                    users.update_user(unsafe_name.user_id, name='Álex de prueba')
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelector('#people-grid').textContent.includes('Álex de prueba')")
                    await page.evaluate('window.__qaReleasePeople()')
                    assert await page.evaluate("!document.querySelector('#people-grid').textContent.includes('<img')")
                    await page.evaluate("document.querySelector('#add-person').focus()")
                    await press_enter(page)
                    await page.wait_for("document.querySelector('#person-dialog').open")
                    await audit_view(page, evidence, 'personas-editor', (('#person-form [name=personName]', None),),
                                     screenshot, widths=(1440, 768, 390))
                    await press_escape(page)
                    await page.wait_for("!document.querySelector('#person-dialog').open && document.activeElement.id === 'add-person'")
                    marker = len(requests)
                    await page.evaluate("window.__qaHoldPeopleMutation=true;document.querySelector('#add-person').click();document.querySelector('#person-form [name=personName]').value='Persona creada en navegador';document.querySelector('#person-form').requestSubmit()")
                    await page.wait_for("typeof window.__qaReleasePeopleMutation==='function'")
                    await page.evaluate("document.querySelector('#rail-nav [data-section=home]').click();document.querySelector('#section-views [data-view=actions]').click();document.querySelector('#rail-nav [data-section=people]').click()")
                    assert await page.evaluate("document.querySelector('#add-person').disabled && document.querySelector('#people-status').textContent.includes('Esperando')")
                    assert sum(item['path'] == '/api/admin/users' and item['method'] == 'POST' for item in requests[marker:]) == 1
                    await page.evaluate('window.__qaReleasePeopleMutation()')
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===4 && document.querySelector('#person-detail h2').textContent==='Persona creada en navegador'")
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
                    await page.evaluate("(()=>{const trigger=document.querySelector('#person-detail .identity-danger');trigger.focus();trigger.click()})()")
                    await page.wait_for("document.querySelector('#person-dialog').open")
                    removal_focus = await audit_modal_focus(page, '#person-dialog', initial='#person-dialog .identity-secondary')
                    assert not any(item['method'] == 'DELETE' for item in requests[marker:])
                    assert personal_session.lease.current()
                    await audit_view(page, evidence, 'personas-confirmacion', (('#person-dialog .identity-secondary', None),),
                                     screenshot, widths=(1440, 768, 390))
                    await press_escape(page)
                    await page.wait_for("!document.querySelector('#person-dialog').open")
                    await audit_modal_return(page, evidence, 'personas-retirar', removal_focus,
                                             dialog='#person-dialog', opener='#person-detail .identity-danger')
                    await page.evaluate("document.querySelector('#person-detail .identity-danger').click();(()=>{const button=document.querySelector('#confirm-person-removal');button.click();button.click()})()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3 && document.querySelector('#people-status').textContent.includes('Persona retirada')")
                    assert users.get_user(created.user_id) is None
                    assert authority.resolve(personal.token) is None and not personal_session.lease.current()
                    assert sum(item['method'] == 'DELETE' for item in requests[marker:]) == 1
                    await page.evaluate("""[...document.querySelectorAll('#people-grid [data-person]')]
                        .find(card => card.querySelector('.row-title').textContent === 'Administración sintética').click();
                        document.querySelector('#person-detail .identity-danger').click();
                        document.querySelector('#confirm-person-removal').click()""")
                    await page.wait_for("document.querySelector('#people-status').textContent.includes('último administrador')")
                    assert users.get_user(admin.user_id) is not None
                    await page.evaluate("document.querySelector('#refresh-people').click()")
                    await page.wait_for("document.querySelectorAll('#people-grid [data-person]').length===3")
                    audits['personas'] = await audit_view(page, evidence, 'personas', PEOPLE_CONTROLS, screenshot)
                    await audit_q24(page, evidence, 'personas')
                    await screenshot(page, evidence / 'people-desktop.png')
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 390, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                    await screenshot(page, evidence / 'people-mobile.png', full=False)
                    await page.command('Emulation.setDeviceMetricsOverride', {'width': 320, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
                    await page.evaluate("document.querySelector('#add-person').click()")
                    assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth && document.querySelector("#person-dialog").getBoundingClientRect().right<=window.innerWidth')
                    await page.evaluate("document.querySelector('#person-form [name=personName]').value='Borrador preservado';window.__qaPeopleUnavailable=true;document.dispatchEvent(new Event('visibilitychange'))")
                    await page.wait_for("document.querySelector('#person-dialog .person-dialog-status').textContent.includes('no está disponible')")
                    await audit_status(page, evidence, 'personas', '503', '#person-dialog .person-dialog-status')
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
                    legacy_links['operator-relogin-1425'] = await page.evaluate(LEGACY_LINKS)
                    assert not any(legacy_links.values()), legacy_links
                    await page.evaluate("window.__qaFailLogout=true;document.querySelector('#logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent.includes('No pudimos confirmar')")
                    assert await page.evaluate("document.querySelector('#nexo-workspace').hidden && !document.querySelector('#room-controls').children.length")
                    marker = len(requests)
                    await page.command("Page.reload")
                    await page.wait_for("document.querySelector('#login-error')?.textContent.includes('cierre anterior')")
                    assert len(requests) == marker
                    await page.evaluate("document.querySelector('#retry-logout').click()")
                    await page.wait_for("document.querySelector('#login-error').textContent==='Sesión cerrada.'")

                    # Leave time for the real login and first catalog, then observe expiry while the view is mounted.
                    short = await authority.issue_credential(root, subject="synthetic-short", role=DashboardRole.HOME, expires_at=time.time() + 8)
                    await login(page, short.token, "Acceso de hogar")
                    await page.wait_for("document.querySelector('#room-list').children.length>0")
                    await page.wait_for("document.querySelector('#nexo-workspace').hidden")
                    assert await page.evaluate("!document.querySelector('#room-controls').children.length && window.__qaFrames.size===0")
                    assert page.errors == [], page.errors
                    result = {"passed": True, "synthetic_fixture": True, "actual_chromium_http_login": True,
                              "roles": ["home", "evaluator", "operator"], "home_effects": len(ha.sends),
                              "webgl_house": webgl, "webgl_context_recovery": True, "webgl_floor_pixels": floor_pixels,
                              "individual_sessions": True,
                              "live_home_checks": ["exact bindings including escalera; unbound room stays in list",
                                                   "pin and floor pointer selection; movement over 4 px does not select",
                                                   "quick room switch, off-state brightness slider, global scene receipt",
                                                   "observed climate temperature and one-touch setpoint steps"],
                              "checks": ["bootstrap no private prefetch", "hidden bootstrap sends no catalog", "role permissions before requests", "independent light/climate typed effects", "focused drafts preserve current observed labels", "lost POST and in_progress receipt reconcile by poll without repost", "camera visual and ARIA agree", "visible model between renders", "navigation releases renderer/RAF", "320px and 390px no horizontal overflow", "operator session has no /obs link", "logout clears before pending HTTP and blocks crossed login", "failed logout stays hidden across reload", "session expires before fresh catalog", "held catalog past expiry disables and marks entity controls, fresh data restores focus and draft", "no entity observation lapses between refreshes during audits"],
                              "staleness": {**staleness_check, "stale_episodes_during_audits": len(stale_episodes)},
                              "notice_checks": ["real shared-only list and dismiss", "initial and post-dismiss empty state", "pending and uncertain delivery labels",
                                                "no markup in notice text", "stale and offline states block writes", "keyboard focus holds through refresh and returns after removal",
                                                "lost committed receipt stays uncertain", "visible evaluator has no notice requests or DOM after a late response",
                                                "logout clears before late response", "server rejects evaluator cookie introduced after whoami"],
                              "people_checks": ['real CRUD once per gesture', 'mutation stays pending across leave/reentry without repost', 'crossed GET keeps latest result', 'lost mutation receipt requires GET without repost', 'account level does not select dashboard role', 'cancelled confirmation has no effect', 'deletion retires every personal session', 'last admin conflict', '503 is not empty catalog', 'open dialog reports failed refresh and disables submit while preserving draft', 'names are DOM text', '320/390 reflow', 'logout clears dialog/draft/accounts before late GET under another login'],
                              "screenshots": ["home-desktop.png", "climate-desktop.png", "home-mobile.png", "people-desktop.png", "people-mobile.png"],
                              "theme_screenshots": {theme: [width['screenshot'] for width in audit['widths'].values()]
                                                    for theme, audit in themed_audits.items()},
                              "mobile_pin_layout": mobile_pins,
                              "reflow_widths": {view: sorted(audit['widths'], key=int) for view, audit in audits.items()},
                              "a11y_checks": ["no horizontal overflow per C08 width", "key controls visible, enabled and hit-testable",
                                              "accessible names from the browser accessibility tree", "Tab reaches every control in document order",
                                              ":focus-visible with a visible indicator, not obscured", "44x44 CSS px touch targets at 320/390",
                                              "html lang es", "prefers-reduced-motion settles the 3D camera at once",
                                              "Q24 text 14px and phone field 16px floors at 320/390/1440",
                                              "Q26 rotate buttons: Tab, Enter, Space, drag, reset, reduced motion, no-WebGL fallback"],
                              "a11y_evidence": [f'a11y-{view}.json' for view in audits], "reduced_motion": motion,
                              "rotation_q26": rotation,
                              "login_error_checks": login_error_checks,
                              "legacy_links": legacy_links}
                    (evidence / "browser-result.json").write_text(json.dumps(result, indent=2) + '\n')
                    print(json.dumps(result))
        finally:
            if browser is not None and browser.poll() is None:
                browser.terminate()
                try:
                    await asyncio.to_thread(browser.wait, 10)
                except subprocess.TimeoutExpired:
                    browser.kill()
                    await asyncio.to_thread(browser.wait, 5)
            await dashboard.stop()
            await serving
            await authority.aclose()
            await notice_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--without-plan', action='store_true', help='Serve a checkout without either private file')
    variants.add_argument('--missing-model', action='store_true', help='Serve a checkout with plan definitions only')
    args = parser.parse_args()
    asyncio.run(main(without_plan=args.without_plan, missing_model=args.missing_model))
