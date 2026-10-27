"""Authenticated, durable ambient admission with actual native pause completion."""

import asyncio
from collections.abc import Callable
from contextvars import Context
from dataclasses import dataclass,field

from src.core.owned_work import WorkCloseTimeout
from src.dashboard.auth import DashboardPrincipal
from src.memory.fact_ledger import MemoryOrigin


@dataclass(frozen=True)
class AmbientController:
    """A dashboard capability, independent of acoustic names and voice commands."""

    subject: str
    _current: Callable[[],bool] = field(repr=False,compare=False)

    @classmethod
    def from_dashboard(cls, principal: DashboardPrincipal, current: Callable[[],bool]) -> 'AmbientController':
        """Mint only from the server's explicit ambient-control permission."""
        if type(principal) is not DashboardPrincipal or 'ambient:control' not in principal.permissions:
            raise PermissionError('ambient_control_denied')
        return cls(principal.subject,current)

    def require_current(self) -> None:
        """A queued transition cannot outlive its credential."""
        if self._current() is not True:
            raise PermissionError('ambient_control_denied')


class AmbientControl:
    """Seal immediately; acknowledge pause only after retained work and reset exit."""

    def __init__(self, privacy, path, *, voices=None):
        if path.transcriber._privacy is not privacy or path.store.source_policy is not privacy.ledger:
            raise ValueError('ambient_control_authority_mismatch')
        if voices is not None and voices.privacy is not privacy:
            raise ValueError('ambient_control_voice_authority_mismatch')
        self.privacy,self.path,self.voices = privacy,path,voices
        self._revision = 0
        self._target = self._persisted = True
        self._phase = 'initializing'
        self._ready = self._closed = False
        self._operation = self._closing = None
        self._seal()

    def _seal(self):
        if self.voices is not None:
            self.voices.suspend_origin(MemoryOrigin.AMBIENT)
        self.privacy.scopes.suspend_origin(MemoryOrigin.AMBIENT)
        self.path.transcriber.pause_admission()
        if self.path.distiller is not None:
            self.path.distiller.set_paused(True)

    def _unseal(self):
        if self._closed:
            raise RuntimeError('ambient_control_closed')
        self.privacy.scopes.resume_origin(MemoryOrigin.AMBIENT)
        if self.voices is not None:
            self.voices.resume_origin(MemoryOrigin.AMBIENT)
        self.path.transcriber.resume_admission()
        if self.path.distiller is not None:
            self.path.distiller.set_paused(False)

    def _load(self):
        with self.privacy.ledger._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS ambient_runtime_control (
                id INTEGER PRIMARY KEY CHECK(id=1),paused INTEGER NOT NULL CHECK(paused IN (0,1)),
                revision INTEGER NOT NULL CHECK(revision>=0))''')
            db.execute('INSERT OR IGNORE INTO ambient_runtime_control VALUES (1,0,0)')
            row = db.execute('SELECT paused,revision FROM ambient_runtime_control WHERE id=1').fetchone()
            if row is None or row['paused'] not in (0,1) or type(row['revision']) is not int or row['revision']<0:
                raise RuntimeError('ambient_control_state_invalid')
            return bool(row['paused']),row['revision']

    def _write(self, paused, revision, actor):
        with self.privacy.ledger._transaction() as db:
            actor.require_current()
            row = db.execute('SELECT revision FROM ambient_runtime_control WHERE id=1').fetchone()
            if row is None or row['revision']!=revision:
                raise ValueError('ambient_control_revision_conflict')
            db.execute('UPDATE ambient_runtime_control SET paused=?,revision=revision+1 WHERE id=1',(int(paused),))
            actor.require_current()
            return revision+1

    def _restore_paused(self):
        with self.privacy.ledger._transaction() as db:
            db.execute('UPDATE ambient_runtime_control SET paused=1,revision=revision+1 WHERE id=1')
            return db.execute('SELECT revision FROM ambient_runtime_control WHERE id=1').fetchone()[0]

    def _start(self, operation):
        self._operation = asyncio.create_task(operation,context=Context())
        def finished(done):
            if not done.cancelled():
                done.exception()
        self._operation.add_done_callback(finished,context=Context())
        return self._operation

    async def initialize(self) -> None:
        """Load private policy while admission stays sealed, before capture starts."""
        if self._operation is not None or self._closed:
            raise RuntimeError('ambient_control_already_initialized')
        await asyncio.shield(self._start(self._initialize()))

    async def _initialize(self):
        paused,revision = await self.privacy.acall(self._load)
        self._revision,self._target,self._persisted = revision,paused,paused
        await self._drain()
        if self._closed:
            raise RuntimeError('ambient_control_closed')
        if not paused:
            self._unseal()
        self._ready = True
        self._phase = 'paused' if paused else 'running'

    def request(self, paused: bool, revision: int, actor: AmbientController) -> dict:
        """Admit one exact-revision transition; return immediately with its phase."""
        if type(actor) is not AmbientController:
            raise PermissionError('ambient_control_denied')
        actor.require_current()
        if self._closed or not self._ready:
            raise RuntimeError('ambient_control_unavailable')
        if self._operation is not None and not self._operation.done():
            raise ValueError('ambient_control_busy')
        if type(paused) is not bool or type(revision) is not int or revision!=self._revision:
            raise ValueError('ambient_control_revision_conflict')
        self._target = paused
        self._phase = 'pausing' if paused else 'resuming'
        self._seal()
        self._start(self._transition(paused,revision,actor))
        return self.snapshot()

    async def _transition(self, paused, revision, actor):
        committed_resume = False
        try:
            if paused:
                self._revision = await self.privacy.acall(self._write,True,revision,actor)
                self._persisted = True
            await self._drain()
            actor.require_current()
            if not paused:
                self._revision = await self.privacy.acall(self._write,False,revision,actor)
                committed_resume = True
                self._persisted = False
                actor.require_current()
                self._unseal()
            self._phase = 'paused' if paused else 'running'
        except (Exception,asyncio.CancelledError):
            self._seal()
            self._phase = 'failed'
            if committed_resume:
                try:
                    self._revision = await self.privacy.acall(self._restore_paused)
                    self._persisted = True
                except (Exception,asyncio.CancelledError):
                    pass
        finally:
            actor = None

    async def _drain(self):
        await self.path.transcriber.drain_pause()
        if self.path.distiller is not None:
            while self.path.distiller.snapshot_operations():
                await asyncio.gather(*self.path.distiller.snapshot_operations(),return_exceptions=True)
                await asyncio.sleep(0)
        if self.voices is not None:
            while self.voices.pending_origin(MemoryOrigin.AMBIENT):
                await asyncio.sleep(.005)

    def snapshot(self) -> dict:
        """Project counts and policy only; no household, source or media metadata."""
        operations = set(self.path.transcriber.snapshot_operations())
        if self.path.distiller is not None:
            operations.update(self.path.distiller.snapshot_operations())
        pending = sum(not task.done() for task in operations)+int(self.path.tap.ingress_pending())
        if self.voices is not None:
            pending+=self.voices.pending_origin(MemoryOrigin.AMBIENT)
        return {'revision':self._revision,'target_paused':self._target,'persisted_paused':self._persisted,
            'phase':self._phase,'admission_paused':self.path.transcriber._paused,
            'pending_operations':pending,'command_capture_affected':False,'total_mute_supported':False}

    def begin_close(self) -> asyncio.Task:
        """Retain transitions and real native drain after observer cancellation."""
        self._closed = True
        self._seal()
        if self._closing is None:
            async def finish():
                if self._operation is not None:
                    await asyncio.gather(asyncio.shield(self._operation),return_exceptions=True)
                await self._drain()
                self._phase = 'closed'
            self._closing = asyncio.create_task(finish(),context=Context())
        return self._closing

    async def aclose(self) -> None:
        """A bounded observer never releases unfinished native ownership."""
        try:
            async with asyncio.timeout(1):
                await asyncio.shield(self.begin_close())
        except TimeoutError:
            raise WorkCloseTimeout('ambient_control_still_running') from None


async def build_ambient_control(*, privacy, path, voices=None, shutdown) -> AmbientControl:
    """Install the shared control before any runtime producer can start."""
    control = AmbientControl(privacy,path,voices=voices)
    privacy.add_borrower(control.begin_close)
    shutdown.push_async_callback(control.aclose)
    await control.initialize()
    return control
