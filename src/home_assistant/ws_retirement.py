"""Retain aiohttp's internal WS close invocation and connection-lost receipt.

The SDK calls ``self.close`` from receive paths and clears cancelled ping
references. Capture these owners before auth, including for borrowed sessions.
Private SDK fields used here are covered by real-aiohttp loopback tests.
"""

import asyncio
from collections.abc import Callable, Coroutine
from contextvars import Context
from typing import Any

import aiohttp


def _observe_done(future):
    if not future.cancelled():
        future.exception()


class WebSocketRetirement:
    """Own the close and its bounded observation for a single native socket."""

    def __init__(
        self, ws: aiohttp.ClientWebSocketResponse,
        retain: Callable[[Coroutine[Any, Any, Any], str], asyncio.Task],
        purpose: str, *, observe_s: float, abort_join_s: float,
        pending: Callable[[], None], isolate: Callable[[asyncio.Task], None],
    ) -> None:
        self._ws = ws
        self._retain = retain
        self._purpose = purpose
        self._observe_s = observe_s
        self._abort_join_s = abort_join_s
        self._pending = pending
        self._isolate = isolate
        self._task: asyncio.Task | None = None
        self._supervisor: asyncio.Task | None = None
        self._sdk_tasks: set[asyncio.Task] = set()
        self._response = ws._response
        connection = self._response.connection
        protocol = connection.protocol if connection is not None else ws._writer.protocol
        if protocol is None:
            raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable")
        self._protocol = protocol
        self._lost = protocol.closed
        connection_lost = protocol._connection_lost_called
        self._original_close = ws.close
        self._cancel_heartbeat = ws._cancel_heartbeat
        self._set_closed = ws._set_closed
        ping = ws._ping_task
        if (type(connection_lost) is not bool
                or (not connection_lost if self._lost is None else not isinstance(self._lost, asyncio.Future))
                or ping is not None and not isinstance(ping, asyncio.Task)
                or not all(map(callable, (protocol.abort, self._set_closed, self._cancel_heartbeat,
                                          self._original_close, self._response.close, self._response.wait_for_close)))):
            raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable")
        if self._lost is not None:
            self._lost.add_done_callback(_observe_done, context=Context())
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

    @property
    def pending(self) -> bool:
        """Return whether the SDK close still owns unfinished work."""
        return self._task is not None and not self._task.done()

    def request_abort(self) -> None:
        """Abort the captured protocol, which belongs to this socket only."""
        self._protocol.abort()

    async def close(self, **options) -> bool:
        """Join the bounded supervisor, even for SDK-originated closes."""
        if self._supervisor is None:
            retain = self._retain
            self._task = retain(self._finish(options), self._purpose)
            self._supervisor = retain(self._supervise(self._task), self._purpose)
            self._retain = None
        return await asyncio.shield(self._supervisor)

    async def _supervise(self, owner: asyncio.Task) -> bool:
        try:
            done, _ = await asyncio.wait((owner,), timeout=self._observe_s)
            if done:
                return owner.result()
            self._pending()
            self.request_abort()
            done, _ = await asyncio.wait((owner,), timeout=self._abort_join_s)
            if done:
                return owner.result()
            if (not self._protocol._connection_lost_called
                    or self._lost is not None and (not self._lost.done() or self._lost.cancelled())):
                raise aiohttp.ClientConnectionError("ha_socket_abort_unconfirmed")

            # Fence SDK callbacks and detach the old response from the
            # connector before admitting any successor in the same session.
            self._set_closed()
            self._response.close()
            self._isolate(owner)
            owner.cancel()
            return False
        finally:
            self._pending = self._isolate = None

    async def _finish(self, options: dict) -> bool:
        try:
            return await self._original_close(**options)
        finally:
            # The closed flag can precede SDK exit. Join these owners even
            # when the original close raises or is cancelled.
            self._response.close()
            try:
                await self._response.wait_for_close()
            finally:
                await self._join_sdk()

    async def _join_sdk(self) -> None:
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


def is_native_websocket(ws):
    # Mock spec.__class__ spoofing must not turn a leaf into a real SDK adapter.
    return aiohttp.ClientWebSocketResponse in type(ws).__mro__


def capture_websocket_retirement(ws, retain, purpose, **options) -> WebSocketRetirement | None:
    """Install retirement tracking before the SDK can begin an automatic close."""
    if not is_native_websocket(ws):
        return None
    try:
        return WebSocketRetirement(ws, retain, purpose, **options)
    except (AttributeError, TypeError) as error:
        raise aiohttp.ClientConnectionError("ha_sdk_retirement_unavailable") from error
