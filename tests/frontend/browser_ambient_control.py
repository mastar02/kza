"""Actual authenticated privacy UI, retained ASR pause, and non-voice resume."""

import asyncio
from contextlib import AsyncExitStack
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

from src.ambient.control import build_ambient_control
from src.dashboard.api import DashboardAPI
from src.dashboard import system_monitor
from src.dashboard.live_erasure import build_live_services
from src.memory.erasure_coordinator import ErasureCoordinator
from tests.frontend.browser_typed_controls import Page
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.ambient_graph import ambient_graph,Clock


async def main():
    """Exercise the real page with synthetic audio and an actually held native ASR."""
    token='synthetic-ambient-control-browser'
    os.environ['KZA_DASHBOARD_TOKEN']=token
    browser_data=Path.home()/'snap/chromium/common'
    browser_data.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kza-ambient-browser-',dir=browser_data) as directory:
        work=Path(directory)
        async with ambient_graph(work,stage='stt',clock=Clock(time.time())) as g,AsyncExitStack() as stack:
            control=await build_ambient_control(privacy=g.privacy,path=g.path,shutdown=stack)
            await g.transcriber.start()
            g.clock.now+=.2
            owner=ErasureCoordinator(g.privacy)
            stack.push_async_callback(owner.aclose)
            bus=await build_live_services({},privacy=g.privacy,owner=owner,shutdown=stack)
            dashboard=DashboardAPI(ambient_control=control,event_bus=bus,observability_use_mocks=False)
            gpu,services=system_monitor.gpu_snapshot,system_monitor.services_snapshot
            system_monitor.gpu_snapshot,system_monitor.services_snapshot=lambda:None,lambda:None
            listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
            server=uvicorn.Server(uvicorn.Config(dashboard.app,log_level='warning'))
            serving=asyncio.create_task(server.serve(sockets=[listener]))
            browser=processing=None
            try:
                await until(lambda:server.started)
                profile=work/'chromium'
                with (work/'chromium.log').open('w') as log:
                    browser=subprocess.Popen(['chromium','--headless','--disable-gpu','--no-first-run',
                        '--remote-debugging-port=0',f'--user-data-dir={profile}'],stdout=log,stderr=log)
                    async with asyncio.timeout(15):
                        while not (profile/'DevToolsActivePort').exists():
                            if browser.poll() is not None:
                                raise RuntimeError((work/'chromium.log').read_text()[:2000])
                            await asyncio.sleep(.05)
                    debug=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                    async with aiohttp.ClientSession() as client:
                        async with client.get(f'http://127.0.0.1:{debug}/json/list') as response:
                            target=next(item for item in await response.json() if item['type']=='page')
                    async with connect(target['webSocketDebuggerUrl'],max_size=None) as ws:
                        page=Page(ws)
                        await page.command('Runtime.enable')
                        await page.command('Network.enable')
                        await page.command('Network.setCookie',{'name':'kza_dashboard_token','value':token,
                            'url':f'http://127.0.0.1:{port}/','httpOnly':True,'sameSite':'Strict'})
                        await page.command('Network.setBlockedURLs',{'urls':['https://unpkg.com/*','https://fonts.googleapis.com/*','https://fonts.gstatic.com/*']})
                        await page.command('Page.navigate',{'url':f'http://127.0.0.1:{port}/obs/'})
                        await page.wait_for("document.querySelector('#view-live') && window.__kzaLiveWs?.readyState===1")
                        await page.evaluate("[...document.querySelectorAll('.nav-item')].find(item=>item.textContent==='Privacidad').click()")
                        await page.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental habilitada')")
                        processing=asyncio.create_task(g.transcriber._handle_segment('synthetic-a',g.segment()))
                        await until(g.gates['stt'].entered.is_set)
                        await page.evaluate("[...document.querySelectorAll('button')].find(item=>item.textContent==='Pausar captura ambiental').click()")
                        await page.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Pausando: esperando el trabajo iniciado')")
                        assert control.snapshot()['phase']=='pausing' and control.snapshot()['pending_operations']>0
                        assert await page.evaluate("[...document.querySelectorAll('button')].find(item=>item.textContent==='Reactivar captura ambiental').disabled")
                        g.gates['stt'].release.set()
                        await processing
                        await page.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental pausada')")
                        cursor=await g.store._db.execute('SELECT COUNT(*) FROM utterances')
                        assert (await cursor.fetchone())[0]==0
                        await page.evaluate("[...document.querySelectorAll('button')].find(item=>item.textContent==='Reactivar captura ambiental').click()")
                        await page.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental habilitada')")
                        g.clock.now+=.2
                        await g.transcriber._handle_segment('synthetic-a',g.segment())
                        assert len(await g.store.undistilled_live(require_provenance=True))==1
                        await page.evaluate("[...document.querySelectorAll('button')].find(item=>item.textContent==='Pausar captura ambiental').click()")
                        await page.wait_for("document.querySelector('#view-privacy')?.textContent.includes('Captura ambiental pausada')")
                        assert len(await g.store.undistilled_live(require_provenance=True))==1
                        os.environ['KZA_DASHBOARD_TOKEN']='synthetic-ambient-token-rotated'
                        await page.wait_for("!document.querySelector('#view-privacy') || [...document.querySelectorAll('#view-privacy button')].every(item=>item.disabled)")
                        assert control.snapshot()['phase']=='paused'
                        assert page.errors==[]
                        print(json.dumps({'passed':True,'actual_chromium_http_ws_and_react':True,
                            'ui_pause_pending_until_actual_native_asr_exit':True,'late_transcript_not_persisted':True,
                            'reactivation_via_authenticated_ui_without_voice':True,'old_history_preserved':True,
                            'revoked_session_cannot_reactivate':True,'javascript_exceptions':0,'external_bootstrap_assets_blocked':True,
                            'synthetic_models_and_sources':True,'total_mute_hardware_or_all_producers_acceptance':False}))
            finally:
                g.gates['stt'].release.set()
                if processing is not None:
                    await asyncio.gather(processing,return_exceptions=True)
                if browser is not None:
                    browser.terminate();await asyncio.to_thread(browser.wait,timeout=10)
                server.should_exit=True
                await serving
                system_monitor.gpu_snapshot,system_monitor.services_snapshot=gpu,services


if __name__=='__main__':
    asyncio.run(main())
