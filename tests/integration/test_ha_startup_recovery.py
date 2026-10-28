"""HA failure at startup degrades control while local services keep ownership."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

from aiohttp import web
import pytest

from src.home_assistant.ha_client import HomeAssistantClient, HAConnectionState

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def endpoint(status):
    state = {'ready':False,'reads':0,'ws_attempts':0,'service_calls':0}
    async def api(request):
        return web.json_response({'message':'API running'}) if state['ready'] else web.Response(status=status)
    async def states(request):
        state['reads'] += 1
        return web.json_response([{'entity_id':'light.synthetic','state':'off','attributes':{}}]) if state['ready'] else web.Response(status=status)
    async def websocket(request):
        state['ws_attempts'] += 1
        if not state['ready']:
            return web.Response(status=status)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({'type':'auth_required'})
        auth = await ws.receive_json()
        assert auth == {'type':'auth','access_token':'synthetic-token'}
        await ws.send_json({'type':'auth_ok'})
        async for msg in ws:
            body = msg.json()
            if body.get('type') == 'subscribe_events':
                await ws.send_json({'id':body['id'],'type':'result','success':True})
        return ws
    async def service(request):
        state['service_calls'] += 1
        return web.json_response([])
    app = web.Application()
    app.add_routes([web.get('/api/',api),web.get('/api/states',states),web.get('/api/websocket',websocket),
        web.post('/api/services/{domain}/{service}',service)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner,'127.0.0.1',0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f'http://127.0.0.1:{port}',state
    finally:
        await runner.cleanup()


@pytest.mark.parametrize('status',[403,503])
async def test_main_continues_local_construction_after_real_ha_startup_rejection(monkeypatch,status):
    import src.main as module
    async with endpoint(status) as (url,state):
        client = HomeAssistantClient(url,'synthetic-token')
        monkeypatch.setattr(module,'load_dotenv',lambda:None)
        monkeypatch.setattr(module,'load_config',lambda _: {'home_assistant':{'url':url,'token':'synthetic-token'}})
        monkeypatch.setattr(module,'HomeAssistantClient',lambda **kwargs:client)
        monkeypatch.setattr(module,'_preflight_vram',lambda:None)
        def reached(_):
            raise RuntimeError('local_model_factory_reached')
        monkeypatch.setattr(module,'create_stt',reached)
        with pytest.raises(RuntimeError,match='^local_model_factory_reached$'):
            async with AsyncExitStack() as stack:
                await module._run_main(stack)
        health = client.get_health_status()
        assert health.state is (HAConnectionState.AUTH_ERROR if status==403 else HAConnectionState.DISCONNECTED)
        assert client._session.closed and state['service_calls']==0


@pytest.mark.parametrize('status',[403,503])
async def test_deferred_state_sync_recovers_through_same_client_and_existing_backoff(status):
    async with endpoint(status) as (url,state):
        client = HomeAssistantClient(url,'synthetic-token')
        try:
            assert not await client.test_connection()
            # The bounded warmup failure does not permanently disable the
            # retained reconnect loop. Use its real network calls and backoff.
            await client.start_state_sync()
            task = client._state_subscribe_task
            assert task is not None and not task.done()
            async with asyncio.timeout(3):
                while state['reads']<1:
                    await asyncio.sleep(.001)
            state['ready']=True
            async with asyncio.timeout(8):
                while client.get_entity_state_cached('light.synthetic') is None or client._ws_events is None:
                    await asyncio.sleep(.005)
            assert client._state_subscribe_task is task and not task.done()
            assert client.get_health_status().state is HAConnectionState.CONNECTED
            assert state['service_calls']==0
        finally:
            await client.close()
        assert client._state_subscribe_task is None and client._session.closed
