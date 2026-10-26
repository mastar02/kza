"""Chromium review smoke with disposable SQLite/Chroma and synthetic facts."""

import asyncio
import base64
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from uuid import uuid4

import aiohttp
import uvicorn
from websockets.asyncio.client import connect

from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer, MemorySource
from src.memory.services import build_memory_services
from tests.frontend.browser_typed_controls import Page
from tests.mocks.memory_graph import MemoryClient


async def main():
    """Exercise real browser events, authenticated review and logout cleanup."""
    os.environ['KZA_DASHBOARD_TOKEN'] = 'synthetic-memory-browser'
    browser_data = Path.home()/'snap/chromium/common'
    browser_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-memory-browser-', dir=browser_data) as work:
        directory = Path(work)
        async with AsyncExitStack() as stack:
            services = await build_memory_services({'memory': {
                'privacy_db_path': str(directory/'privacy.db'),
                'preferences_path': str(directory/'preferences.json'),
            }}, shutdown=stack, chroma_client=MemoryClient(directory/'chroma'))
            memory = services.manager.long_term

            def source():
                now = time.time()
                return MemorySource(str(uuid4()), str(uuid4()), now, now+48*3600, MemoryOrigin.VOICE)

            first = await memory.acall(memory.propose, 'Prefiero cocinar temprano. Dato sintético de prueba.', (source(),))
            second = await memory.acall(memory.propose, 'Una segunda propuesta sintética.', (source(),))
            await memory.astore_fact('La planta de prueba se riega los martes.', 'fact', sources=(source(),))
            dashboard = DashboardAPI(memory_manager=services.manager, home_enabled=True)

            @dashboard.app.middleware('http')
            async def session_fixture(request, call_next):
                response = await call_next(request)
                response.headers['X-KZA-Demo'] = 'true'
                if request.url.path == '/':
                    response.set_cookie('kza_dashboard_token', 'synthetic-memory-browser', httponly=True, samesite='strict')
                return response

            listener = socket.socket(); listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='warning'))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            browser = None
            try:
                while not server.started:
                    await asyncio.sleep(.02)
                profile = directory/'chromium'
                with (directory/'chromium.log').open('w') as log:
                    browser = subprocess.Popen(['chromium', '--headless', '--disable-gpu', '--no-first-run',
                        '--remote-debugging-port=0', f'--user-data-dir={profile}',
                        f'http://127.0.0.1:{port}/'], stdout=log, stderr=log)
                    for _ in range(200):
                        if (profile/'DevToolsActivePort').exists():
                            break
                        if browser.poll() is not None:
                            raise RuntimeError((directory/'chromium.log').read_text()[:3000])
                        await asyncio.sleep(.05)
                    debugging = int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f'http://127.0.0.1:{debugging}/json/list') as response:
                            target = next(item for item in await response.json() if item['type'] == 'page')
                    async with connect(target['webSocketDebuggerUrl'], max_size=None) as connection:
                        page = Page(connection); await page.command('Runtime.enable')
                        await page.command('Emulation.setDeviceMetricsOverride', {'width': 1280, 'height': 900, 'deviceScaleFactor': 1, 'mobile': False})
                        await page.wait_for("document.querySelector('[data-view=memory]') && !document.querySelector('#workspace').hidden")
                        await page.evaluate("document.querySelector('[data-view=memory]').click()")
                        await page.wait_for("document.querySelectorAll('[data-memory-id]').length === 3")
                        first_selector = '[data-memory-id="'+first['id']+'"]'
                        second_selector = '[data-memory-id="'+second['id']+'"]'
                        await page.evaluate(f"[...document.querySelector({json.dumps(first_selector)}).querySelectorAll('button')].find(b => b.textContent === 'Conservar sin vencimiento').click()")
                        await page.wait_for(f"document.querySelector({json.dumps(first_selector)})?.textContent.includes('Confirmado manualmente')")
                        assert memory.ledger.get(first['id']).confirmed_by == 'dashboard-admin'
                        # A second actor confirms after the browser's snapshot.
                        await memory.acall(memory.confirm, second['id'], MemoryReviewer('other-test-reviewer'))
                        await page.evaluate(f"[...document.querySelector({json.dumps(second_selector)}).querySelectorAll('button')].find(b => b.textContent === 'Descartar propuesta').click()")
                        await page.wait_for("document.body.textContent.includes('cambió o venció')")
                        assert memory.ledger.get(second['id']).state.value == 'confirmed'
                        await page.evaluate("document.querySelector('#refresh').click()")
                        await page.wait_for("document.querySelectorAll('[data-memory-id]').length === 3")
                        screenshots = []
                        for name, width, height, mobile in [('desktop',1280,900,False),('mobile',390,844,True)]:
                            await page.command('Emulation.setDeviceMetricsOverride', {'width':width,'height':height,'deviceScaleFactor':1,'mobile':mobile})
                            screenshot = await page.command('Page.captureScreenshot', {'format':'png','captureBeyondViewport':True})
                            destination = Path('/tmp/kza-memory-review-'+name+'.png')
                            destination.write_bytes(base64.b64decode(screenshot['data'])); destination.chmod(0o600)
                            screenshots.append(str(destination))
                            assert not await page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
                        await page.evaluate(f"[...document.querySelector({json.dumps(first_selector)}).querySelectorAll('button')].find(b => b.textContent === 'Eliminar recuerdo').click()")
                        await page.wait_for(f"!document.querySelector({json.dumps(first_selector)})")
                        assert memory.ledger.get(first['id']) is None
                        await page.command('Network.clearBrowserCookies')
                        await page.evaluate("document.querySelector('#refresh').click()")
                        await page.wait_for("document.querySelector('#workspace').hidden && document.querySelector('#content').children.length === 0")
                        assert not page.errors, page.errors
                        print(json.dumps({'passed':True, 'confirmation':True, 'stale_rejection_preserved_confirmation':True,
                            'fact_deleted':True, 'logout_cleared_private_dom':True,
                            'javascript_exceptions':len(page.errors), 'screenshots':screenshots}))
            finally:
                if browser is not None:
                    browser.terminate(); await asyncio.to_thread(browser.wait, timeout=10)
                server.should_exit = True; await serving


if __name__ == '__main__':
    asyncio.run(main())
