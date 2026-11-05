"""Durable administrative control over actual reusable capture owners."""

from __future__ import annotations

import asyncio
from contextvars import Context
from dataclasses import dataclass, field
from typing import Callable

from src.core.owned_work import OwnedBlockingWork, WorkCloseTimeout
from src.dashboard.authority import DashboardPrincipal
from src.memory.fact_ledger import MemoryOrigin


@dataclass(frozen=True)
class CaptureController:
    """Server-minted authority bound to the original administrative lease."""

    subject: str
    _current: Callable[[], bool] = field(repr=False, compare=False)

    @classmethod
    def from_dashboard(cls, principal: DashboardPrincipal, current: Callable[[], bool]):
        if type(principal) is not DashboardPrincipal or 'admin:manage' not in principal.permissions:
            raise PermissionError('capture_control_denied')
        return cls(principal.subject, current)

    def require_current(self) -> None:
        if self._current() is not True:
            raise PermissionError('capture_control_denied')


class CaptureUploadLease:
    """Retain reception, parsing and cleanup independently of its HTTP observer."""

    def __init__(self, control, current):
        self._control, self._current = control, current
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._retired = False

    def current(self) -> bool:
        return (not self._retired and not self.stop.is_set()
                and self._control.capture_allowed and self._current() is True)

    def attach_task(self, task: asyncio.Task) -> None:
        if self.task is not None or not isinstance(task, asyncio.Task):
            raise RuntimeError('capture_upload_owner_invalid')
        self.task = task
        task.add_done_callback(self._finished, context=Context())

    def _finished(self, task):
        if not task.cancelled():
            task.exception()
        self.retire()

    def retire(self) -> None:
        if self.task is not None and not self.task.done() and asyncio.current_task() is not self.task:
            raise RuntimeError('capture_upload_still_owned')
        self._retired = True
        self._control._uploads.discard(self)
        self._control._uploads_changed.set()


class CaptureControl:
    """Persistent intent is distinct from confirmed native and browser exit."""

    def __init__(self, privacy, audio_loop, command_processor, *, ambient_control=None,
                 voices=None, collectors=(), native_drains=(), native_supported=True):
        from src.ambient.control import AmbientControl
        from src.memory.capture_recorders import CaptureRecorders
        from src.pipeline.audio_loop import AudioLoop
        from src.pipeline.command_processor import CommandProcessor
        from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
        from src.training.conversation_collector import ConversationCollector
        from src.training.stt_correction_collector import STTCorrectionCollector
        from src.users.voice_registry import VoiceRegistry

        self.privacy, self.audio_loop, self.command_processor = privacy, audio_loop, command_processor
        self.ambient_control, self.voices = ambient_control, voices
        self.collectors = tuple(c for c in collectors if c is not None)
        self._native_drains = tuple(native_drains)
        self._native_supported = native_supported is True
        self._supported = isinstance(audio_loop, (AudioLoop, MultiRoomAudioLoop))
        self._supported &= isinstance(command_processor, CommandProcessor)
        self._supported &= ambient_control is None or isinstance(ambient_control, AmbientControl)
        self._supported &= voices is None or isinstance(voices, VoiceRegistry)
        self._supported &= all(isinstance(c, (ConversationCollector, STTCorrectionCollector))
                               for c in self.collectors)
        self._supported &= self._native_supported
        self._participants = tuple((part, method) for part, method, nominal in (
            (audio_loop, 'drain_pause', (AudioLoop, MultiRoomAudioLoop)),
            (command_processor, 'drain_pause', (CommandProcessor,)),
            (ambient_control, 'drain_capture', (AmbientControl,)),
            *((c, 'drain_pause', (ConversationCollector, STTCorrectionCollector))
              for c in self.collectors),
        ) if isinstance(part, nominal))
        for part in (audio_loop, command_processor):
            bound = getattr(part, '_privacy', None)
            if bound is not None and bound is not privacy:
                raise ValueError('capture_control_authority_mismatch')
        if ambient_control is not None and ambient_control.privacy is not privacy:
            raise ValueError('capture_control_ambient_authority_mismatch')
        if voices is not None and voices.privacy is not privacy:
            raise ValueError('capture_control_voice_authority_mismatch')
        self._reason = object()
        self._revision = 0
        self._target = self._persisted = True
        self._mode = 'muted'
        self._phase = 'initializing'
        self._sealed = True
        self._ready = self._closed = self._policy_valid = False
        self._operation = self._initialization = self._closing = None
        self._error_code = None
        self._seal_failed = False
        self._close_signal = asyncio.Event()
        self._uploads_changed = asyncio.Event()
        self._uploads: set[CaptureUploadLease] = set()
        self._enrollment_work: OwnedBlockingWork | None = None
        # The privacy owner retains this borrower before closing its executor
        # and ledger. Its public work admission may already be sealed while an
        # admitted transition still needs to compensate an uncertain commit.
        self._policy_work = OwnedBlockingWork(worker=privacy._offload, capacity=2)
        self.recorders = CaptureRecorders(privacy, capture_allowed=lambda: self.capture_allowed)

    @property
    def capture_allowed(self) -> bool:
        """A short read also safe for original native commit guards."""
        return self._ready and not self._closed and not self._sealed and not self._target

    def bind_enrollment_work(self, work: OwnedBlockingWork) -> None:
        if not isinstance(work, OwnedBlockingWork) or (self._enrollment_work is not None
                                                      and self._enrollment_work is not work):
            raise ValueError('capture_enrollment_owner_invalid')
        self._enrollment_work = work

    def admit_upload(self, current: Callable[[], bool]) -> CaptureUploadLease:
        if current() is not True:
            raise PermissionError('capture_control_denied')
        if not self.capture_allowed or len(self._uploads) >= 2:
            raise RuntimeError('capture_upload_unavailable')
        lease = CaptureUploadLease(self, current)
        self._uploads.add(lease)
        return lease

    def _seal_origins(self) -> None:
        self._sealed = True
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            for part in (self.privacy.scopes, self.voices):
                if part is not None:
                    try:
                        part.suspend_origin(origin, reason=self._reason)
                    except Exception:
                        self._seal_failed = True

    def _seal(self) -> None:
        self._seal_origins()
        self.recorders.seal()
        for lease in tuple(self._uploads):
            lease.stop.set()
        for part, drain_method in self._participants:
            method = 'pause_capture' if drain_method == 'drain_capture' else 'pause_admission'
            try:
                getattr(part, method)()
            except Exception:
                self._seal_failed = True

    def _release_origins(self) -> None:
        for origin in (MemoryOrigin.AMBIENT, MemoryOrigin.VOICE):
            if self.voices is not None:
                self.voices.resume_origin(origin, reason=self._reason)
            self.privacy.scopes.resume_origin(origin, reason=self._reason)
        self._sealed = False

    def _load_policy(self):
        with self.privacy.ledger._transaction() as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                "AND name='capture_runtime_control'").fetchone() is not None
            if not exists:
                db.execute("""CREATE TABLE capture_runtime_control (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    mode TEXT NOT NULL CHECK(mode IN ('running','muted','resuming')),
                    revision INTEGER NOT NULL CHECK(revision>=0))""")
                # Preserve the capture-enabled default only for a new schema.
                # A missing row in an established schema is corruption.
                db.execute("INSERT INTO capture_runtime_control VALUES(1,'running',0)")
            row = db.execute('SELECT mode,revision FROM capture_runtime_control WHERE id=1').fetchone()
            if (row is None or row['mode'] not in ('running', 'muted', 'resuming')
                    or type(row['revision']) is not int or row['revision'] < 0):
                raise RuntimeError('capture_control_state_invalid')
            return row['mode'], row['revision']

    def _write_mode(self, mode, revision, actor):
        with self.privacy.ledger._transaction() as db:
            actor.require_current()
            if self._closed:
                raise RuntimeError('capture_control_closed')
            row = db.execute('SELECT revision FROM capture_runtime_control WHERE id=1').fetchone()
            if row is None or row['revision'] != revision:
                raise ValueError('capture_control_revision_conflict')
            db.execute('UPDATE capture_runtime_control SET mode=?,revision=revision+1 WHERE id=1',
                       (mode,))
            actor.require_current()
            return revision + 1

    def _force_muted(self):
        with self.privacy.ledger._transaction() as db:
            row = db.execute('SELECT mode,revision FROM capture_runtime_control WHERE id=1').fetchone()
            if row is None:
                raise RuntimeError('capture_control_state_invalid')
            if row['mode'] != 'muted':
                db.execute("UPDATE capture_runtime_control SET mode='muted',revision=revision+1 WHERE id=1")
                return row['revision'] + 1
            return row['revision']

    @staticmethod
    def _spawn(operation):
        task = asyncio.create_task(operation, context=Context())
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception(),
                               context=Context())
        return task

    async def initialize(self) -> None:
        if self._initialization is not None or self._closed:
            raise RuntimeError('capture_control_already_initialized')
        self._seal_origins()
        self._initialization = self._spawn(self._initialize())
        await asyncio.shield(self._initialization)

    async def _initialize(self):
        try:
            mode, revision = await self._policy_work.run(self._load_policy)
            await self.recorders.initialize()
            self._mode, self._revision = mode, revision
            self._target = self._persisted = mode != 'running'
            self._policy_valid = True
            if self._closed:
                return
            self._ready = True
            if self._target:
                self._seal()
                self._phase = 'muting'
                self._operation = self._spawn(self._restore())
            else:
                self._release_origins()
                self.recorders.resume()
                self._phase = 'running'
        except (Exception, asyncio.CancelledError):
            self._seal()
            self._phase = 'failed'
            self._error_code = 'capture_control_state_invalid'

    async def _restore(self):
        try:
            await self._drain_server()
            await self._drain_recorders()
            if not self._closed:
                self._phase = 'muted'
        except (Exception, asyncio.CancelledError):
            self._seal()
            self._phase = 'failed'
            self._error_code = 'capture_retirement_unconfirmed'

    def request(self, muted: bool, expected_revision: int, actor: CaptureController) -> dict:
        if type(actor) is not CaptureController:
            raise PermissionError('capture_control_denied')
        actor.require_current()
        if self._closed or not self._ready or not self._policy_valid or not self._supported:
            raise RuntimeError('capture_control_unavailable')
        if (type(muted) is not bool or type(expected_revision) is not int
                or expected_revision < 0 or expected_revision != self._revision):
            raise ValueError('capture_control_revision_conflict')
        if self._operation is not None and not self._operation.done():
            raise ValueError('capture_control_busy')
        self._target = muted
        self._error_code = None
        self._seal_failed = False
        self._seal()
        self._phase = 'muting' if muted else 'resuming'
        self._operation = self._spawn(self._transition(muted, expected_revision, actor))
        return self.snapshot()

    async def _transition(self, muted, revision, actor):
        try:
            mode = 'muted' if muted else 'resuming'
            self._revision = await self._policy_work.run(self._write_mode, mode, revision, actor)
            self._mode, self._persisted = mode, True
            await self._drain_server()
            await self._drain_recorders()
            actor.require_current()
            if self._closed:
                raise RuntimeError('capture_control_closed')
            if muted:
                self._phase = 'muted'
                return
            await self._resume_server()
            actor.require_current()
            self._revision = await self._policy_work.run(self._write_mode, 'running', self._revision, actor)
            actor.require_current()
            if self._closed:
                raise RuntimeError('capture_control_closed')
            self._release_origins()
            self.recorders.resume()
            self._mode, self._persisted = 'running', False
            self._phase = 'running'
        except (Exception, asyncio.CancelledError):
            self._seal()
            self._phase = 'failed'
            self._error_code = 'capture_transition_unconfirmed'
            # A worker can fail after SQLite committed but before its observer
            # received the receipt. Every failed transition restores durable
            # mute, including that uncertain commit boundary.
            try:
                self._revision = await self._policy_work.run(self._force_muted)
                self._mode, self._persisted = 'muted', True
            except (Exception, asyncio.CancelledError):
                self._error_code = 'capture_policy_unconfirmed'
        finally:
            actor = None

    async def _drain_uploads(self):
        while self._uploads:
            leases = tuple(self._uploads)
            tasks = tuple(lease.task for lease in leases if lease.task is not None)
            if tasks:
                await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
            else:
                self._uploads_changed.clear()
                if self._uploads:
                    await self._uploads_changed.wait()

    async def _drain_server(self, *, closing=False):
        if not closing and (not self._supported or self._seal_failed):
            raise RuntimeError('capture_participant_unsupported')

        async def drain_part(part, method):
            if getattr(part, '_closed', False):
                await asyncio.shield(part.begin_close())
            else:
                await getattr(part, method)()

        results = await asyncio.gather(*(drain_part(part, method) for part, method in self._participants),
                                       self._drain_uploads(), return_exceptions=True)
        native = list(self._native_drains) if not closing or self._native_supported else []
        native.append(self._policy_work.drain_admitted)
        if self.voices is not None:
            native.append(self.voices._work.drain_admitted)
        if self._enrollment_work is not None:
            native.append(self._enrollment_work.drain_admitted)
        results.extend(await asyncio.gather(*(call() for call in native), return_exceptions=True))
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            if isinstance(failures[0], Exception):
                raise failures[0]
            raise RuntimeError('capture_retirement_unconfirmed')

    async def _drain_recorders(self):
        waiting = self._spawn(self.recorders.drain())
        closing = asyncio.create_task(self._close_signal.wait(), context=Context())
        try:
            await asyncio.wait((waiting, closing), return_when=asyncio.FIRST_COMPLETED)
            if self._closed:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                raise RuntimeError('capture_control_closed')
            await asyncio.shield(waiting)
        finally:
            closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)

    async def _resume_server(self):
        self.command_processor.resume_admission()
        if self.ambient_control is not None:
            self.ambient_control.resume_capture()
        for part in self.collectors:
            part.resume_admission()
        self.audio_loop.resume_admission()
        from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop
        if isinstance(self.audio_loop, MultiRoomAudioLoop):
            await self.audio_loop.drain_resume()
        else:
            opening = getattr(self.audio_loop, '_reopening', None)
            if opening is not None:
                await asyncio.shield(opening)

    def snapshot(self) -> dict:
        tasks = set()
        for part in (self.audio_loop, self.command_processor):
            snapshot = getattr(part, 'snapshot_operations', None)
            if callable(snapshot):
                tasks.update(snapshot())
        pending = sum(not task.done() for task in tasks) + len(self._uploads)
        pending += self._policy_work.status()['operations']
        pending += int(self._operation is not None and not self._operation.done())
        if self.ambient_control is not None:
            pending += self.ambient_control.snapshot()['pending_operations']
        if self.voices is not None:
            pending += sum(self.voices.pending_origin(origin)
                           for origin in (MemoryOrigin.VOICE, MemoryOrigin.AMBIENT))
        for part in self.collectors:
            pending += len(part._jobs) + len(part._active)
        if self._enrollment_work is not None:
            pending += self._enrollment_work.status()['operations']
        recorders_pending = self.recorders.pending()
        browsers = self.recorders.pending(kind='browser')
        native_clients = (self.recorders.pending(kind='calibration')
                          + self.recorders.pending(kind='wakeword_training')
                          + self.recorders.pending(kind='wakeword_diagnostics'))
        return {'revision': self._revision, 'target_muted': self._target,
                'persisted_muted': self._persisted, 'phase': self._phase,
                'ingress_sealed': self._sealed, 'supported': bool(self._supported),
                'pending_operations': pending + recorders_pending,
                'pending_browser_recorders': browsers, 'pending_native_clients': native_clients,
                'error_code': self._error_code,
                'total_mute_confirmed': (self._phase == 'muted' and self._persisted
                                         and self._sealed and self._supported and not pending
                                         and not recorders_pending)}

    def begin_close(self) -> asyncio.Task:
        self._closed = True
        self._close_signal.set()
        self._seal()
        if self._closing is None:
            async def finish():
                if self._initialization is not None:
                    await asyncio.gather(asyncio.shield(self._initialization), return_exceptions=True)
                if self._operation is not None:
                    await asyncio.gather(asyncio.shield(self._operation), return_exceptions=True)
                policy_closing = self._policy_work.begin_close()
                results = await asyncio.gather(self._drain_server(closing=True), self.recorders.aclose(),
                                               asyncio.shield(policy_closing),
                                               return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                self._phase = 'closed'
            self._closing = self._spawn(finish())
        return self._closing

    async def aclose(self) -> None:
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('capture_control_still_running') from None


async def build_capture_control(*, privacy, audio_loop, command_processor,
                                ambient_control=None, voices=None, collectors=(),
                                native_drains=(), native_supported=True, shutdown) -> CaptureControl:
    control = CaptureControl(privacy, audio_loop, command_processor,
                             ambient_control=ambient_control, voices=voices,
                             collectors=collectors, native_drains=native_drains,
                             native_supported=native_supported)
    privacy.add_borrower(control.begin_close)
    shutdown.push_async_callback(control.aclose)
    await control.initialize()
    return control
