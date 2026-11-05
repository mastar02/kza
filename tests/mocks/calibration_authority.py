"""Synthetic positive HTTP grant for K; actual loopback server for authority tests."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import socket
from urllib.parse import urlsplit

import requests
from requests.adapters import BaseAdapter
import uvicorn


class PositiveCalibrationAdapter(BaseAdapter):
    """Replace only HTTP transport; exercise the shipped client and monitor."""

    base_url = 'http://127.0.0.1:18765'

    def __init__(self):
        super().__init__()
        self.token = 'a' * 43
        self.active = False
        self.retired = False
        self.calls = []

    def send(self, request, **kwargs):
        parsed = urlsplit(request.url)
        if (request.method != 'POST' or parsed.scheme != 'http'
                or parsed.netloc != '127.0.0.1:18765'
                or not parsed.path.startswith('/api/capture/native/calibration/')):
            raise RuntimeError('unexpected_calibration_test_transport')
        action = parsed.path.rsplit('/', 1)[-1]
        body = json.loads(request.body or '{}')
        self.calls.append(action)
        if action == 'register':
            assert body == {}
            payload = {'token': self.token, 'state': 'reserved', 'capture_allowed': False,
                'kind': 'calibration', 'generation': 0}
        else:
            assert body == {'token': self.token}
            if action == 'activate':
                self.active = True
            elif action == 'retire':
                self.retired = True
                self.active = False
            elif action != 'poll':
                raise RuntimeError('unexpected_calibration_test_action')
            if action == 'retire':
                payload = {'retired': True}
            else:
                payload = {'state': 'active' if self.active else 'retired',
                    'capture_allowed': self.active, 'retirement_required': not self.active,
                    'kind': 'calibration', 'generation': 0}
        response = requests.Response()
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        response._content = json.dumps(payload).encode()
        response.url = request.url
        response.request = request
        return response

    def close(self):
        # Transport disposal is deliberately not a remote retirement receipt.
        pass


def positive_authority(monkeypatch):
    """Provide K's explicit positive policy leaf, never a fake client object."""
    from tools.calibration_authority import CalibrationAuthorityClient

    adapter = PositiveCalibrationAdapter()

    session = requests.Session()
    session.trust_env = False
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return CalibrationAuthorityClient(adapter.base_url, 'synthetic-policy-token',
        session_factory=lambda: session)


@asynccontextmanager
async def serve_capture_app(app):
    """Serve the actual ASGI registry on an ephemeral loopback listener."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', lifespan='off'))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if serving.done():
                    serving.result()
                await asyncio.sleep(.001)
        yield f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        try:
            async with asyncio.timeout(5):
                await asyncio.shield(serving)
        finally:
            listener.close()
