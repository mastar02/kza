"""Retirable live records and actual WebSocket ownership under capture authority."""

import asyncio
from collections import deque
from contextvars import Context
from dataclasses import dataclass, field
from enum import StrEnum
import math
import time
from uuid import uuid4
from weakref import WeakSet, finalize, ref

from src.core.owned_work import WorkCloseTimeout
from src.dashboard.live_browser import LiveBrowserSession
from src.memory.fact_ledger import FactDependency, MemorySource


class LiveDeliveryResult(StrEnum):
    STOPPED = 'live_delivery_stopped'
    FAILED = 'live_delivery_failed'


def _text(value, limit=8192):
    return value if type(value) is str and len(value) <= limit else None


def project_live_payload(payload: object) -> dict | None:
    """Copy closed scalar fields without serializing arbitrary caller objects."""
    if type(payload) is not dict:
        return None
    output = {}
    for key in ('id', 'zone', 'stt', 'tts', 'intent', 'path', 'text', 'title', 'body', 'endpoint_id', 'until', 'priority', 'type'):
        value = payload.get(key)
        if value is None or _text(value) is not None:
            if key in payload:
                output[key] = value
        else:
            return None
    user = payload.get('user')
    if type(user) is dict:
        user = user.get('name')
    if user is not None and _text(user, 256) is None:
        return None
    if 'user' in payload:
        output['user'] = user
    for key in ('latency_ms', 'confidence', 'duration_ms', 'step'):
        value = payload.get(key)
        if value is None:
            continue
        if type(value) not in (int, float) or not -1e12 <= value <= 1e12 or not math.isfinite(value):
            return None
        output[key] = value
    if 'success' in payload:
        if type(payload['success']) is not bool:
            return None
        output['success'] = payload['success']
    if sum(len(value.encode('utf-8')) for value in output.values() if type(value) is str) > 32768:
        return None
    return output


@dataclass(eq=False, repr=False)
class RetainedLiveEvent:
    """Only an owned payload is mutable; retirement cannot be reversed."""

    type: object
    ts: str
    sources: tuple[MemorySource, ...]
    expires_at: float
    _policy: object
    _data: dict | None = field(repr=False)
    event_id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def payload(self) -> dict:
        """Expose a detached view; a consumer cannot mutate sibling queues."""
        if not self.available():
            self.retire()
            return {}
        return dict(self._data or {})

    def available(self) -> bool:
        """Read capture authority without I/O or a renewed deadline."""
        return self._data is not None and self._policy.sources_available(self.sources)

    def retire(self) -> None:
        """Drop the raw projection while preserving its causal metadata."""
        self._data = None

    def to_frame(self) -> dict | None:
        """Recheck at serialization, including a record already dequeued."""
        if not self.available():
            self.retire()
            return None
        return {'type': self.type.value, 'payload': self.payload, 'ts': self.ts,
            'event_id': self.event_id, 'expires_at': self.expires_at}


@dataclass(frozen=True)
class LastLiveFrame:
    """Remember what the ticker holds without retaining its private payload."""

    event_id: str
    sources: tuple[MemorySource, ...]


@dataclass
class LiveSubscriber:
    """One sender, one delivered metadata receipt, and a bounded control box."""

    queue: asyncio.Queue
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    retired: set[str] = field(default_factory=set)
    last: LastLiveFrame | None = None
    active: RetainedLiveEvent | None = field(default=None, repr=False)
    closed: bool = False
    guard: object | None = field(default=None, repr=False)
    browser: LiveBrowserSession | None = field(default=None, repr=False)


class LiveRetention:
    """Own bounded queues and socket sends after an observer stops waiting."""

    def __init__(self, queues, policy, *, clock=None, capacity=256, max_subscribers=16, close_timeout_s=1, receipts=None):
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError('invalid_live_queue_capacity')
        if type(max_subscribers) is not int or not 1 <= max_subscribers <= 64:
            raise ValueError('invalid_live_subscriber_capacity')
        self.queues, self.policy = queues, policy
        self.receipts = receipts
        self.clock = clock or time.time
        self.max_subscribers = max_subscribers
        self._record_capacity = max_subscribers*(capacity+1)
        self._records = WeakSet()
        self._by_conversation: dict[str, WeakSet] = {}
        self._scan = deque()
        self._states: dict[int, LiveSubscriber] = {}
        self._senders: dict[int, asyncio.Task] = {}
        self._loop = self._maintenance = self._closing = None
        self._closed = False
        self._active = self.rejected = 0
        self._close_timeout_s = close_timeout_s
        self._wake = asyncio.Event()
        policy.add_revocation_observer(self._revoke)

    def _revoke(self, conversations):
        for conversation in conversations:
            for record in tuple(self._by_conversation.get(conversation, ())):
                record.retire()
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.prune, context=Context())

    def admit(self, event) -> RetainedLiveEvent | None:
        """Validate and detach before publishing a single raw reference."""
        sources = event.sources
        if (self._closed or type(sources) is not tuple or not sources or len(sources) > 256
                or any(type(source) is not MemorySource or any(type(dep) is not FactDependency
                    for dep in source.fact_dependencies) for source in sources)
                or sum(len(source.fact_dependencies) for source in sources) > 256
                or not self.policy.sources_available(sources)
                or len(self._records) >= self._record_capacity):
            self.rejected += 1
            return None
        payload = project_live_payload(event.payload)
        if payload is None or type(event.ts) is not str or len(event.ts) > 32:
            self.rejected += 1
            return None
        record = RetainedLiveEvent(event.type, event.ts, sources,
            min(source.expires_at for source in sources), self.policy, payload)
        self._records.add(record)
        conversations = frozenset(source.conversation_id for source in sources)
        for conversation in conversations:
            self._by_conversation.setdefault(conversation, WeakSet()).add(record)
        finalize(record, self._forget, self._by_conversation, conversations)
        if not self.policy.sources_available(sources):
            record.retire()
            self.rejected += 1
            return None
        return record

    @staticmethod
    def _forget(index, conversations):
        for conversation in conversations:
            if not any(index.get(conversation, ())):
                index.pop(conversation, None)

    def attach(self, sub_id, queue):
        if self._closed or len(self._states) >= self.max_subscribers:
            raise RuntimeError('live_subscriber_capacity')
        self._states[sub_id] = LiveSubscriber(queue)

    def notify(self, sub_id):
        state = self._states.get(sub_id)
        if state is not None:
            state.wake.set()

    @staticmethod
    def _drain(queue):
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
            queue.task_done()
        return items

    def prune(self) -> None:
        """Retire idle queues and notify clients whose last frame was revoked."""
        records = tuple(self._records)
        now = self.clock()
        for record in records:
            if record.expires_at <= now or not self.policy.authority_available:
                record.retire()
        if not self._scan:
            self._scan.extend(ref(record) for record in records)
        # Conversation fences use the index immediately. Other source/fact
        # changes are swept in bounded batches and rechecked at serialization.
        for _ in range(min(32, len(self._scan))):
            record = self._scan.popleft()()
            if record is not None and record._data is not None and not record.available():
                record.retire()
        for state in tuple(self._states.values()):
            if state.browser is not None and state.browser.timed_out():
                state.closed = True
            if state.guard is not None:
                try:
                    allowed = state.guard() is True
                except Exception:
                    allowed = False
                if not allowed:
                    state.closed = True
            for record in self._drain(state.queue):
                if not state.closed and record._data is not None:
                    state.queue.put_nowait(record)
            if state.last is not None and not self.policy.sources_available(state.last.sources):
                state.retired.add(state.last.event_id)
            if state.active is not None and not state.active.available():
                state.retired.add(state.active.event_id)
            if state.retired or state.closed:
                state.wake.set()

    async def _send(self, sender, frame):
        self._active += 1
        try:
            await sender(frame)
        finally:
            frame = None
            self._active -= 1

    async def _watch_disconnect(self, state, receiver):
        message = None
        try:
            while not state.closed:
                message = await receiver()
                disconnected = type(message) is not dict or message.get('type') == 'websocket.disconnect'
                if disconnected:
                    return
                if state.guard is not None and state.guard() is not True:
                    return
                if state.browser is not None:
                    handling = state.browser.handle(message)
                    message = None
                    await handling
                    handling = None
                message = None
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            message = receiver = None
            state.closed = True
            state.wake.set()

    async def _dispatch(self, state, sender, close, receiver):
        frame = record = None
        watching = (asyncio.create_task(self._watch_disconnect(state, receiver), context=Context())
            if receiver is not None else None)
        try:
            while not state.closed:
                state.wake.clear()
                if state.browser is not None:
                    reset = state.browser.reset_frame()
                    if reset is not None:
                        await self._send(sender,reset)
                        reset = None
                        continue
                    if not state.browser.ready:
                        await state.wake.wait()
                        continue
                if state.retired:
                    tokens = sorted(state.retired)
                    # Remove only the selected controls: revocation may add new
                    # metadata while the actual socket callback is suspended.
                    control = (state.browser.retirement_frame(tokens) if state.browser is not None else
                        {'type':'retire','payload':{'event_ids':tokens},'ts':''})
                    await self._send(sender,control)
                    control = None
                    state.retired.difference_update(tokens)
                    if state.last is not None and state.last.event_id in tokens:
                        state.last = None
                    continue
                if state.queue.empty():
                    await state.wake.wait()
                    continue
                record = state.queue.get_nowait()
                state.queue.task_done()
                state.active = record
                ticket = None
                if state.browser is not None:
                    ticket = await state.browser.before_frame(record)
                    if ticket is None or state.closed:
                        frame = record = state.active = None
                        continue
                frame = record.to_frame()
                if frame is None and ticket is not None:
                    state.retired.add(record.event_id)
                if frame is not None and ticket is not None:
                    frame['receipt'] = ticket.to_dict()
                if frame is not None:
                    await self._send(sender, frame)
                    state.last = LastLiveFrame(record.event_id, record.sources)
                    if not record.available():
                        state.retired.add(record.event_id)
                frame = record = state.active = None
            return LiveDeliveryResult.STOPPED
        except (Exception, asyncio.CancelledError):
            return LiveDeliveryResult.FAILED
        finally:
            frame = record = state.active = None
            if watching is not None:
                watching.cancel()
                await asyncio.gather(watching, return_exceptions=True)
            if state.browser is not None:
                state.browser.seal()
            if close is not None:
                self._active += 1
                try:
                    await close()
                except (Exception, asyncio.CancelledError):
                    pass
                finally:
                    self._active -= 1
            if state.browser is not None:
                try:
                    await state.browser.finish()
                except (Exception,asyncio.CancelledError):
                    pass
            sender = close = receiver = watching = None

    async def stream(self, sub_id, sender, close=None, receiver=None, guard=None, *, browser=False):
        """Shield actual socket ownership; observer cancellation does not stop it."""
        if self._closed or sub_id not in self._states or sub_id in self._senders:
            raise RuntimeError('live_subscriber_unavailable')
        state = self._states[sub_id]
        state.guard = guard
        if browser:
            if self.receipts is None or receiver is None:
                raise RuntimeError('live_browser_receipts_unavailable')
            state.browser = LiveBrowserSession(self.receipts,state.wake)
        task = asyncio.create_task(self._dispatch(state, sender, close, receiver), context=Context())
        self._senders[sub_id] = task
        def finished(done):
            self._senders.pop(sub_id, None)
            self._states.pop(sub_id, None)
            self.queues.pop(sub_id, None)
            self._drain(state.queue)
            state.last = None
            state.guard = None
            state.retired.clear()
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def detach(self, sub_id):
        state = self._states.get(sub_id)
        if state is None:
            return
        state.closed = True
        state.wake.set()
        self._drain(state.queue)
        task = self._senders.get(sub_id)
        if task is None:
            self._states.pop(sub_id, None)
            self.queues.pop(sub_id, None)
            return
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(task)
        except TimeoutError:
            raise WorkCloseTimeout('live_delivery_still_running') from None

    def start(self) -> None:
        """Maintain deadlines without another publication or subscriber read."""
        if self._maintenance is not None or self._closed:
            return
        self._loop = asyncio.get_running_loop()
        async def maintain():
            while not self._closed:
                self._wake.clear()
                self.prune()
                try:
                    async with asyncio.timeout(.05):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        self._maintenance = asyncio.create_task(maintain(), context=Context())

    def pending_conversation(self, conversation_id: str) -> int:
        """Count actual raw send callbacks, independently of durable client ACKs."""
        self.prune()
        return sum(state.active is not None and any(source.conversation_id == conversation_id
            for source in state.active.sources) for state in self._states.values())

    def status(self) -> dict:
        """Expose counts without private payloads, paths, identities or sources."""
        return {'closed':self._closed, 'active_sends':self._active, 'rejected':self.rejected,
            'records':len(self._records), 'senders':len(self._senders),
            'subscribers':len(self._states), 'subscriber_capacity':self.max_subscribers}

    def begin_close(self) -> asyncio.Task:
        """Seal publication and keep actual sends and close callbacks owned."""
        self._closed = True
        self._wake.set()
        for record in tuple(self._records):
            record.retire()
        for state in self._states.values():
            state.closed = True
            state.wake.set()
            self._drain(state.queue)
        if self._closing is None:
            async def finish():
                await asyncio.gather(*tuple(self._senders.values()), return_exceptions=True)
                if self._maintenance is not None:
                    await self._maintenance
                self._states.clear()
                self.queues.clear()
                self._scan.clear()
                self._by_conversation.clear()
                if self.receipts is not None:
                    await self.receipts.begin_close()
            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing

    async def aclose(self) -> None:
        """Bound observation while the actual transport scope remains retained."""
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('live_delivery_still_running') from None
