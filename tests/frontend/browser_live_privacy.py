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
        bus = LiveEventBus(source_policy=privacy.ledger, clock=privacy.ledger.clock)
        privacy.add_borrower(bus.begin_close)
        bus.start()
        dashboard = DashboardAPI(event_bus=bus)
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
                    await privacy.acall(privacy.ledger.delete_conversation,source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await page.wait_for('!document.body.textContent.includes('+json.dumps(private)+')')
                    assert await page.evaluate('window.__kzaLiveState.snapshot() === null')
                    await show('Synthetic expiring browser turn',ttl=1)
                    await page.wait_for("!document.body.textContent.includes('Synthetic expiring browser turn')")
                    await show('Synthetic disconnect turn')
                    await page.evaluate('window.__kzaLiveWs.close()')
                    await page.wait_for("!document.body.textContent.includes('Synthetic disconnect turn')")
                    await page.wait_for('window.__kzaLiveWs?.readyState === 1')
                    await show('Synthetic session turn')
                    await page.command('Network.clearBrowserCookies')
                    await page.evaluate('window.__kzaLiveWs.close()')
                    await page.wait_for("!document.querySelector('#view-live') && !document.body.textContent.includes('Synthetic session turn')")
                    assert not page.errors, page.errors
                    print(json.dumps({'passed':True,'actual_chromium_http_ws_and_react':True,
                        'rendered_conversation_retirement':True,'rendered_idle_expiry':True,
                        'disconnect_clears_ticker':True,'lost_session_clears_private_dom':True,
                        'javascript_exceptions':len(page.errors),'synthetic_sources_and_service_adapters':True,
                        'browser_erasure_ack_or_global_streams_acceptance':False}))
        finally:
            if browser is not None:
                browser.terminate()
                await asyncio.to_thread(browser.wait,timeout=10)
            server.should_exit = True
            await serving
            await bus.aclose()
            await privacy.aclose()
            system_monitor.gpu_snapshot, system_monitor.services_snapshot = original_gpu, original_services


if __name__ == '__main__':
    asyncio.run(main())
