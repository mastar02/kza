"""Retain aiohttp's internal WS close invocation and connection-lost receipt.

The SDK calls ``self.close`` from receive paths and clears cancelled ping
references. Capture these owners before auth, including for borrowed sessions.
Private SDK fields used here are covered by real-aiohttp loopback tests.
"""

import asyncio
from contextvars import Context

import aiohttp


def _observe_done(future):
    if not future.cancelled():
        future.exception()


class WebSocketRetirement:
    def __init__(self, ws, retain, purpose):
        self._ws = ws
        self._retain = retain
        self._purpose = purpose
        self._task = None
        self._sdk_tasks = set()
        self._response = ws._response
        connection = self._response.connection
        protocol = connection.protocol if connection is not None else ws._writer.protocol
        if protocol is None:
            raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable")
        self._protocol = protocol
        self._lost = protocol.closed
        if self._lost is None and not protocol._connection_lost_called:
            raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable")
        if self._lost is not None:
            self._lost.add_done_callback(_observe_done, context=Context())
        self._original_close = ws.close
        self._cancel_heartbeat = ws._cancel_heartbeat
        # No await separates capture from installation: SDK callbacks cannot
        # begin a close or discard a cancelled ping between these steps.
        ws._cancel_heartbeat = self._track_heartbeat_cancel
        ws.close = self.close

    def _track_heartbeat_cancel(self):
        task = self._ws._ping_task
        if task is not None:
            self._sdk_tasks.add(task)

            def finished(done):
                _observe_done(done)
                self._sdk_tasks.discard(done)

            task.add_done_callback(finished, context=Context())
        self._cancel_heartbeat()

    def request_abort(self):
        self._protocol.abort()

    async def close(self, **options):
        if self._task is None:
            self._task = self._retain(self._finish(options), self._purpose)
        return await asyncio.shield(self._task)

    async def _finish(self, options):
        result = await self._original_close(**options)
        # A SDK-originated closed flag can make the original close return
        # immediately. Request this response's physical close, never its session.
        self._response.close()
        await self._response.wait_for_close()
        while self._sdk_tasks:
            pending = tuple(task for task in self._sdk_tasks if not task.done())
            for task in tuple(self._sdk_tasks):
                if task.done():
                    _observe_done(task)
                    self._sdk_tasks.discard(task)
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
        if self._lost is not None:
            try:
                await asyncio.shield(self._lost)
            except aiohttp.ClientConnectionError:
                # Exceptional connection_lost is still an actual notification.
                if not self._lost.done() or self._lost.cancelled():
                    raise
        if self._ws.closed is not True:
            raise aiohttp.ClientConnectionError("ha_socket_close_unconfirmed")
        return result


def is_native_websocket(ws):
    # Mock spec.__class__ spoofing must not turn a leaf into a real SDK adapter.
    return aiohttp.ClientWebSocketResponse in type(ws).__mro__


def capture_websocket_retirement(ws, retain, purpose):
    if not is_native_websocket(ws):
        return None
    try:
        return WebSocketRetirement(ws, retain, purpose)
    except (AttributeError, TypeError) as error:
        raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable") from error
