"""Chromium live ticker privacy smoke with real HTTP/WebSocket and synthetic input."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import aiohttp
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.dashboard import system_monitor
from src.dashboard.live_event_bus import LiveEvent, LiveEventBus, LiveEventType
from src.dashboard.live_receipts import LiveReceiptStore
from src.analytics.event_logger import EventLogger
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.frontend.browser_typed_controls import Page
from tests.mocks.memory_sources import memory_source


async def main():
    """Verify rendered retirement, idle expiry, disconnect and session cleanup."""
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-live-browser'
    browser_data = Path.home()/'snap/chromium/common'
    browser_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-live-browser-', dir=browser_data) as work:
        directory = Path(work)
        privacy = PrivacyServices(str(directory/'privacy.db'))
        await privacy.acall(privacy.initialize)
        receipts = LiveReceiptStore(privacy.ledger)
        bus = LiveEventBus(source_policy=privacy.ledger, clock=privacy.ledger.clock, receipts=receipts)
        privacy.add_borrower(bus.begin_close)
        bus.start()
        events = EventLogger(str(directory/'events.db'), source_policy=privacy.ledger)
        await events.acall(events.initialize)
        dashboard = DashboardAPI(event_bus=bus, event_logger=events, observability_use_mocks=False)
        # Keep mocked UI service adapters independent of host hardware probing.
        original_gpu, original_services = system_monitor.gpu_snapshot, system_monitor.services_snapshot
        system_monitor.gpu_snapshot, system_monitor.services_snapshot = lambda:None, lambda:None
        @dashboard.app.middleware('http')
        async def fixture_cookie(request, call_next):
            response = await call_next(request)
            if request.url.path == '/':
                response.set_cookie('kza_dashboard_token','synthetic-live-browser',httponly=True,samesite='strict')
            return response
        listener = socket.socket()
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app,log_level='warning'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        browser = None
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    await asyncio.sleep(.02)
            profile = directory/'chromium'
            with (directory/'chromium.log').open('w') as log:
                browser = subprocess.Popen(['chromium','--headless','--disable-gpu','--no-first-run',
                    '--remote-debugging-port=0',f'--user-data-dir={profile}',f'http://127.0.0.1:{port}/'],stdout=log,stderr=log)
                async with asyncio.timeout(15):
                    while not (profile/'DevToolsActivePort').exists():
                        if browser.poll() is not None:
                            raise RuntimeError((directory/'chromium.log').read_text()[:2000])
                        await asyncio.sleep(.05)
                debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                        target = next(item for item in await response.json() if item['type']=='page')
                async with connect(target['webSocketDebuggerUrl'],max_size=None) as connection:
                    page = Page(connection)
                    await page.command('Runtime.enable')
                    await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState === 1")
                    async def show(text, ttl=48*3600):
                        source = memory_source(now=time.time(),ttl=ttl)
                        await bus.publish(LiveEvent(LiveEventType.TURN,payload={'stt':text,'success':True},sources=(source,)))
                        await page.wait_for('document.body.textContent.includes('+json.dumps(text)+')')
                        return source
                    private = 'Synthetic private browser turn'
                    source = await show(private)
                    assert await receipts.pending(source.conversation_id) == 1
                    await page.evaluate('window.__liveCommit = window.__kzaLiveState.committed; window.__kzaLiveState.committed = () => {};')
                    await privacy.acall(privacy.ledger.delete_conversation,source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await page.wait_for('!document.body.textContent.includes('+json.dumps(private)+')')
                    assert await page.evaluate('window.__kzaLiveState.snapshot() === null')
                    await asyncio.sleep(.05)
                    assert await receipts.pending(source.conversation_id) == 1
                    await page.evaluate('window.__kzaLiveState.committed = window.__liveCommit; delete window.__liveCommit; window.__kzaLiveState.committed(null);')
                    async with asyncio.timeout(3):
                        while await receipts.pending(source.conversation_id):
                            await asyncio.sleep(.01)
                    await show('Synthetic expiring browser turn',ttl=1)
                    await page.wait_for("!document.body.textContent.includes('Synthetic expiring browser turn')")
                    leaving = await show('Synthetic leaving turn')
                    await page.evaluate("window.dispatchEvent(new Event('pagehide'))")
                    await page.wait_for("!document.body.textContent.includes('Synthetic leaving turn')")
                    async with asyncio.timeout(3):
                        while await receipts.pending(leaving.conversation_id):
                            await asyncio.sleep(.01)
                    await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState === 1")
                    await show('Synthetic disconnect turn')
                    await page.evaluate('window.__kzaLiveWs.close()')
                    await page.wait_for("!document.body.textContent.includes('Synthetic disconnect turn')")
                    await page.wait_for('window.__kzaLiveWs?.readyState === 1')
                    rest_text = 'Synthetic expiring REST conversation'
                    rest_source = memory_source(now=time.time(), ttl=5)
                    await events.alog('light.synthetic', 'turn_on', trigger_phrase=rest_text,
                        user_name='Synthetic REST voice', sources=(rest_source,))
                    await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState === 1")
                    assert await page.evaluate('!JSON.stringify(window.MOCKS.CONVERSATIONS).includes('+json.dumps(rest_text)+')')
                    await page.evaluate("[...document.querySelectorAll('.nav-item')].find(node => node.textContent.includes('Conversaciones')).click()")
                    await page.wait_for('document.body.textContent.includes('+json.dumps(rest_text)+')')
                    await page.evaluate("const filter = document.querySelector('[aria-label=\"Todas las voces\"]'); filter.value = 'Synthetic REST voice'; filter.dispatchEvent(new Event('change'))")
                    await page.wait_for('!document.body.textContent.includes('+json.dumps(rest_text)+')')
                    assert await page.evaluate("window.__kzaConversationState.snapshot().length === 0 && !document.body.textContent.includes('Synthetic REST voice')")
                    async with asyncio.timeout(3):
                        while await receipts.pending(rest_source.conversation_id):
                            await asyncio.sleep(.01)
                    retired_source = memory_source(now=time.time(), ttl=60)
                    retired_text = 'Synthetic revoked REST conversation'
                    await events.alog('light.synthetic','turn_on',trigger_phrase=retired_text,sources=(retired_source,))
                    await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("window.__kzaConversationState?.snapshot().some(row => row.stt === 'Synthetic revoked REST conversation')")
                    await page.evaluate("[...document.querySelectorAll('.nav-item')].find(node => node.textContent.includes('Conversaciones')).click()")
                    await page.wait_for('document.body.textContent.includes('+json.dumps(retired_text)+')')
                    await page.evaluate('window.__restCommit = window.__kzaConversationState.committed; window.__kzaConversationState.committed = () => {};')
                    await privacy.acall(privacy.ledger.delete_conversation,retired_source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await page.wait_for('!document.body.textContent.includes('+json.dumps(retired_text)+')')
                    assert await page.evaluate('window.__kzaConversationState.snapshot().length === 0')
                    await asyncio.sleep(.05)
                    assert await receipts.pending(retired_source.conversation_id)==1
                    await page.evaluate('window.__kzaConversationState.committed = window.__restCommit; delete window.__restCommit; window.__kzaConversationState.committed(null);')
                    async with asyncio.timeout(3):
                        while await receipts.pending(retired_source.conversation_id):
                            await asyncio.sleep(.01)
                    await events.alog('light.synthetic', 'turn_on', trigger_phrase='Synthetic session REST conversation',
                        sources=(memory_source(now=time.time(), ttl=60),))
                    await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                    await page.wait_for("!!document.querySelector('#view-live') && window.__kzaLiveWs?.readyState === 1")
                    await page.wait_for("window.__kzaConversationState.snapshot().some(row => row.stt === 'Synthetic session REST conversation')")
                    await show('Synthetic session turn')
                    await page.command('Network.clearBrowserCookies')
                    await page.evaluate('window.__kzaLiveWs.close()')
                    await page.wait_for("!document.querySelector('#view-live') && !document.body.textContent.includes('Synthetic session turn')")
                    assert await page.evaluate('window.__kzaConversationState.snapshot().length === 0')
                    assert not page.errors, page.errors
                    print(json.dumps({'passed':True,'actual_chromium_http_ws_and_react':True,
                        'rendered_conversation_retirement':True,'rendered_idle_expiry':True,
                        'disconnect_clears_ticker':True,'lost_session_clears_private_dom':True,
                        'javascript_exceptions':len(page.errors),'synthetic_sources_and_service_adapters':True,
                        'browser_ack_after_dom_and_cache_clear':True,
                        'browser_ack_waits_for_explicit_dom_commit':True,
                        'pagehide_handler_clears_and_acknowledges_owned_view':True,
                        'rest_conversation_original_deadline_clears_cache_dom_and_voice_filter':True,
                        'rest_conversation_payload_excluded_from_global_mock_cache':True,
                        'session_loss_clears_unmounted_rest_conversation_cache':True,
                        'rest_revocation_clears_before_original_expiry':True,
                        'rest_ack_waits_for_explicit_dom_commit':True,
                        'global_streams_acceptance':False}))
        finally:
            if browser is not None:
                browser.terminate()
                await asyncio.to_thread(browser.wait,timeout=10)
            server.should_exit = True
            await serving
            await bus.aclose()
            await events.aclose()
            await privacy.aclose()
            system_monitor.gpu_snapshot, system_monitor.services_snapshot = original_gpu, original_services


if __name__ == '__main__':
    asyncio.run(main())
