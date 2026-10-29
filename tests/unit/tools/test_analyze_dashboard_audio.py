"""The live diagnostic uses authenticated pages without exporting raw content."""

import base64
import asyncio
from io import BytesIO
import json
import time
from uuid import uuid4

from aiohttp import web
import numpy as np
import pytest
import soundfile as sf

from tools.analyze_dashboard_audio import analyze_dashboard, measure_preview
from tools.analyze_audio_quality import summarize


def preview():
    output = BytesIO()
    signal = .1 * np.sin(2 * np.pi * 997 * np.arange(16000) / 16000)
    sf.write(output, signal, 16000, format='WAV', subtype='PCM_16')
    return {'audio_mime': 'audio/wav', 'audio_base64': base64.b64encode(output.getvalue()).decode(),
            'expires_at': time.time() + 60, 'room_id': 'synthetic-room', 'transcript': 'private synthetic transcript'}


def test_preview_metrics_export_no_audio_transcript_or_identity():
    row = measure_preview(preview())
    assert row['measurements'][0]['samples'] == 16000
    assert row['kind'] == 'current_dashboard_preview'
    assert not {'audio_base64', 'transcript', 'id'} & row.keys()
    limits = ' '.join(summarize([row])['limits'])
    assert 'authenticated API' in limits and 'from filenames' not in limits
    expired = preview()
    expired['expires_at'] = time.time() - 1
    with pytest.raises(ValueError):
        measure_preview(expired)


@pytest.mark.asyncio
@pytest.mark.parametrize('limit',[None,1])
async def test_live_client_paginates_handles_expiry_and_waits_to_ack_a_held_body(limit):
    connection, reset, nonce = str(uuid4()), str(uuid4()), str(uuid4())
    first, second = 'vf_' + 'a' * 64, 'vf_' + 'b' * 64
    requests, acknowledgements, sockets = [], [], []
    held_ticket = str(uuid4())
    held_nonce = str(uuid4())
    body_released = False
    async def socket(request):
        assert request.headers.get('Authorization') == 'Bearer synthetic-diagnostic'
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sockets.append(ws)
        hello = await ws.receive_json()
        assert hello['type'] == 'live_hello' and hello['version'] == 3
        await ws.send_json({'type': 'live_reset', 'payload': {'id': reset, 'nonce': nonce}})
        assert await ws.receive_json() == {'type': 'live_reset_ack', 'id': reset, 'nonce': nonce}
        await ws.send_json({'type': 'rest_ready', 'connection_id': connection})
        async for incoming in ws:
            if incoming.type == web.WSMsgType.TEXT:
                document = json.loads(incoming.data)
                for receipt in document['receipts']:
                    if receipt['ticket'] == held_ticket:
                        assert body_released, 'Client retired a still-owned HTTP body'
                    acknowledgements.append(receipt)
        return ws
    async def get(request):
        nonlocal body_released
        assert request.method == 'GET'
        assert request.headers.get('Authorization') == 'Bearer synthetic-diagnostic'
        assert request.headers.get('X-KZA-Live-Connection') == connection
        requests.append(request.path_qs)
        if request.path == '/api/voices':
            page = request.query.get('cursor')
            capture = first if not page else second
            result = {'snapshot': {'receipt': {'ticket': str(uuid4()), 'nonce': str(uuid4())},
                                  'rows': [{'id': capture, 'review_available': True}],
                                  'next_cursor': first if not page else None}}
            return web.json_response(result)
        if second in request.path:
            return web.Response(status=404)
        # A retire notification can arrive before the preview HTTP body ends.
        # Its receipt must remain pending until that body has been released.
        await sockets[0].send_json({'type': 'retire', 'payload': {'receipts': [
            {'ticket': held_ticket, 'nonce': held_nonce, 'event_id': str(uuid4())}]}})
        await asyncio.sleep(.05)
        assert not any(item['ticket'] == held_ticket for item in acknowledgements)
        result = {'snapshot': {'receipt': {'ticket': held_ticket, 'nonce': held_nonce}, 'rows': [preview()]}}
        body = json.dumps(result).encode()
        response = web.StreamResponse(headers={'Content-Type': 'application/json'})
        await response.prepare(request)
        await response.write(body[:100])
        await asyncio.sleep(.01)
        body_released = True
        await response.write(body[100:])
        await response.write_eof()
        return response
    app = web.Application()
    app.router.add_get('/ws/live', socket)
    app.router.add_get('/api/voices', get)
    app.router.add_get('/api/voices/captures/{identity}/review', get)
    runner = web.AppRunner(app)
    await runner.setup()
    listener = web.TCPSite(runner, '127.0.0.1', 0)
    await listener.start()
    port = listener._server.sockets[0].getsockname()[1]
    try:
        report = await analyze_dashboard('synthetic-diagnostic', base_url=f'http://127.0.0.1:{port}',
                                         max_captures=limit, timings=True)
        assert report['listed_captures'] == (1 if limit else 2)
        assert report['outcomes'] == ({'analyzed': 1} if limit else {'analyzed': 1, 'unavailable_or_expired': 1})
        assert report['summary']['decoded_files'] == 1
        encoded = json.dumps(report)
        assert 'private synthetic transcript' not in encoded and 'audio_base64' not in encoded
        assert first not in encoded and second not in encoded
        assert ('/api/voices?cursor=' + first in requests) == (limit is None)
        assert any(item['ticket'] == held_ticket for item in acknowledgements)
        timing = next(row for row in report['request_timings'] if row['kind']=='review')
        assert timing['headers_ms'] >= 40 and timing['body_ms'] >= 5
        assert timing['total_ms'] >= timing['headers_ms'] + timing['body_ms']
        assert timing['bytes'] > 32000
        assert 'synthetic-diagnostic' not in encoded
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('limit',[0,-1,4097,True])
async def test_invalid_sample_limit_is_rejected_before_network_access(limit):
    with pytest.raises(ValueError,match='invalid_capture_limit'):
        await analyze_dashboard('synthetic-diagnostic',max_captures=limit)


@pytest.mark.asyncio
async def test_live_client_reads_actual_dashboard_voice_api_and_receipt_protocol(tmp_path, monkeypatch):
    import socket

    import uvicorn

    from src.dashboard.api import DashboardAPI
    from tests.mocks.memory_sources import memory_source
    from tests.mocks.voice_registry import AUDIO, Encoder, idle, web_graph

    monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'synthetic-real-api-diagnostic')
    async with web_graph(tmp_path) as (_, _, voices, clock, bus):
        encoder = Encoder()
        voices.register_encoder(encoder)
        for _ in range(3):
            assert voices.observe(encoder, AUDIO, (memory_source(now=clock.now, ttl=60),),
                                  transcript='Private real-api synthetic phrase', room_id='synthetic-room')
        await idle(voices)
        dashboard = DashboardAPI(event_bus=bus, voice_registry=voices, observability_use_mocks=False)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(dashboard.app, log_level='critical', lifespan='off'))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if serving.done():
                        serving.result()
                    await asyncio.sleep(.01)
            report = await analyze_dashboard('synthetic-real-api-diagnostic', base_url=f'http://127.0.0.1:{port}')
            assert report['listed_captures'] == 3
            assert report['outcomes'] == {'analyzed': 3}
            assert report['summary']['decoded_files'] == 3
            assert 'Private real-api synthetic phrase' not in json.dumps(report)
        finally:
            server.should_exit = True
            async with asyncio.timeout(5):
                await serving
