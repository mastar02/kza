"""Local numeric diagnostic cleanup without claiming unaudited channels erased."""

import asyncio
from contextvars import Context
import logging
from pathlib import Path

from src.core.owned_work import OwnedBlockingWork
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.obs.diagnostic_store import DiagnosticStore
from src.obs.metrics_emitter import DiagnosticFailure, MetricsEmitter, project_diagnostic

logger = logging.getLogger(__name__)


def diagnostic_descriptor(path: str) -> dict:
    """Resolve the versioned local projection before registering its owner."""
    original = Path(path)
    target = original if original.stem.endswith('_policy_v1') else original.with_name(
        original.stem+'_policy_v1'+original.suffix)
    if target.is_symlink():
        raise ValueError('diagnostic_target_symlink')
    return {'kind': 'numeric_catalog', 'path': str(target.resolve())}


def validate_diagnostic_descriptor(descriptor: dict) -> None:
    """Accept only the registered local numeric catalog protocol."""
    if (type(descriptor) is not dict or set(descriptor) != {'kind', 'path'}
            or descriptor['kind'] != 'numeric_catalog' or type(descriptor['path']) is not str
            or not Path(descriptor['path']).is_absolute()
            or not Path(descriptor['path']).stem.endswith('_policy_v1')):
        raise ValueError('unsupported_diagnostic_target')


class DiagnosticServices:
    """Keep one emitter and one serialized historical worker, with no FD pool."""

    def __init__(self, privacy, *, retention_days=90, rotate_bytes=100*1024*1024):
        self.privacy = privacy
        self.emitter: MetricsEmitter | None = None
        self._retention_days, self._rotate_bytes = retention_days, rotate_bytes
        self._targets: dict[str, dict] = {}
        self._history: set[str] = set()
        self._disk = OwnedBlockingWork(capacity=1)
        self._lock = asyncio.Lock()
        self._operations: set[asyncio.Task] = set()
        self._maintenance = self._closing = None
        self._closed = False
        self._wake = asyncio.Event()

    def _invoke_history(self, path, conversation):
        try:
            return self._history_operation(path, conversation)
        except (Exception, asyncio.CancelledError):
            return DiagnosticFailure.FAILED

    def _history_operation(self, path, conversation):
        store = None
        try:
            store = DiagnosticStore(Path(path), clock=self.privacy.ledger.clock,
                retention_days=self._retention_days, rotate_bytes=self._rotate_bytes,
                project=project_diagnostic, policy=self.privacy.ledger)
            if conversation is not None:
                return store.erase(conversation)
            store.prune()
            return 0
        except (Exception, asyncio.CancelledError):
            return DiagnosticFailure.FAILED
        finally:
            if store is not None:
                store.close()

    async def _run_history(self, path, conversation=None):
        # An observer cannot release the lock while its actual native task runs.
        async def execute():
            async with self._lock:
                outcome = await self._disk.run(self._invoke_history, path, conversation)
                if isinstance(outcome, DiagnosticFailure):
                    raise RuntimeError(outcome.value)
                return outcome
        task = asyncio.create_task(execute(), context=Context())
        self._operations.add(task)
        def finished(done):
            self._operations.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished, context=Context())
        return await asyncio.shield(task)

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Acknowledge the exact local target after physical and native cleanup."""
        if self._closed:
            raise RuntimeError('diagnostic_services_closed')
        if step.participant is not ErasureParticipant.DIAGNOSTICS or step.version != 2:
            raise ValueError('unsupported_diagnostic_target')
        validate_diagnostic_descriptor(step.descriptor)
        if self._targets.get(step.target_id) != step.descriptor:
            raise ValueError('unregistered_diagnostic_target')
        self.privacy.require_available()
        if not self.privacy.ledger.conversation_committed_revoked(step.conversation_id):
            raise RuntimeError('diagnostic_erasure_requires_committed_authority')
        path = step.descriptor['path']
        if self.emitter is not None and str(self.emitter.path) == path:
            pending = await self.emitter.aerase(step.conversation_id)
        else:
            pending = await self._run_history(path, step.conversation_id)
        return ErasureResult(pending == 0, pending_count=pending)

    def start(self) -> None:
        """Sweep historical TTL with one worker, independent of optional emission."""
        if self._maintenance is not None:
            return
        async def maintain():
            while not self._closed:
                self._wake.clear()
                for path in sorted(self._history):
                    if self._closed:
                        break
                    try:
                        await self._run_history(path)
                    except Exception:
                        logger.warning('Historical diagnostic repair deferred')
                try:
                    async with asyncio.timeout(1):
                        await self._wake.wait()
                except TimeoutError:
                    pass
        self._maintenance = asyncio.create_task(maintain(), context=Context())

    def begin_close(self) -> asyncio.Task:
        """Retain current TCP and historical disk ownership until actual exit."""
        self._closed = True
        self._wake.set()
        if self._closing is None:
            emission = self.emitter.begin_close() if self.emitter is not None else None
            async def finish():
                if self._maintenance is not None:
                    await self._maintenance
                await asyncio.gather(*tuple(self._operations), return_exceptions=True)
                await self._disk.begin_close()
                if emission is not None:
                    await emission
            self._closing = asyncio.create_task(finish(), context=Context())
        return self._closing


async def build_diagnostic_services(config: dict, *, privacy, owner) -> DiagnosticServices:
    """Register cleanup before opening catalogs or admitting conversation metrics."""
    settings = config.get('observability', {})
    descriptor = await privacy.acall(diagnostic_descriptor,
        settings.get('metrics_path', '/home/kza/logs/kza-metrics.jsonl'))
    await privacy.acall(privacy.ledger.erasures.register_target,
        ErasureParticipant.DIAGNOSTICS, 2, descriptor)
    # The local protocol cannot certify application/SDK logs, unversioned legacy
    # or remote collectors. Keep that missing audit durable and visibly blocked.
    await privacy.acall(privacy.ledger.erasures.register_target,
        ErasureParticipant.DIAGNOSTICS, 1, {'kind': 'unresolved_diagnostic_inventory'})
    services = DiagnosticServices(privacy,
        retention_days=config.get('analytics', {}).get('retention_days', 90),
        rotate_bytes=settings.get('rotate_bytes', 100*1024*1024))
    owner.register(ErasureParticipant.DIAGNOSTICS, services.clean,
        versions=frozenset({2}), begin_close=services.begin_close)
    targets = await privacy.acall(privacy.ledger.erasures.targets, ErasureParticipant.DIAGNOSTICS)
    for target in targets:
        if target['version'] != 2:
            continue
        try:
            validate_diagnostic_descriptor(target['descriptor'])
        except ValueError:
            continue
        services._targets[target['id']] = target['descriptor']
        services._history.add(target['descriptor']['path'])
    if settings.get('enabled', True):
        emitter = MetricsEmitter(descriptor['path'], policy=privacy.ledger, clock=privacy.ledger.clock,
            retention_days=services._retention_days, rotate_bytes=services._rotate_bytes,
            logstash_host=settings.get('logstash_host'), logstash_port=settings.get('logstash_port', 5515))
        services.emitter = emitter
        privacy.add_borrower(emitter.begin_close)
        services._history.discard(descriptor['path'])
        try:
            await emitter.start()
        except Exception:
            logger.warning('Diagnostic retention unavailable; local repair will retry')
    services.start()
    return services
