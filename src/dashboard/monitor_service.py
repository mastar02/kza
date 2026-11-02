"""Bounded on-demand probes whose actual workers outlive cancelled HTTP waiters."""

import asyncio
from contextvars import Context
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging

from src.core.owned_work import OwnedBlockingWork
from src.core.observation import Observation, ObservationSource
from src.dashboard import system_monitor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeSample:
    """A native read and the metadata fixed once by its actual owner."""

    data: list[dict] | None = field(repr=False)
    observation: Observation


class MonitorService:
    """Share simultaneous probes and drain native work before dashboard shutdown."""

    def __init__(self) -> None:
        self._work = OwnedBlockingWork(capacity=2)
        self._tasks: dict[str, asyncio.Task] = {}
        self._closed = False
        self._closing: asyncio.Task | None = None
        self._last: dict[str, Observation] = {}

    async def sample(self, component: str) -> ProbeSample:
        """Read one known probe outside the event loop without retaining caller context."""
        probes = {'gpus': system_monitor.gpu_snapshot, 'services': system_monitor.services_snapshot}
        if component not in probes:
            raise ValueError('unknown_monitor_probe')
        if self._closed:
            raise RuntimeError('monitors_closed')
        task = self._tasks.get(component)
        if task is None or task.done():
            task = asyncio.create_task(self._read(component, probes[component]), context=Context())
            self._tasks[component] = task

            def finished(done):
                if self._tasks.get(component) is done:
                    self._tasks.pop(component, None)
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(finished, context=Context())
        try:
            return await asyncio.shield(task)
        finally:
            task = None

    async def _read(self, component, function) -> ProbeSample:
        previous = self._last.get(component)

        def capture():
            attempted = datetime.now(UTC)
            try:
                data = function()
            except Exception as error:
                logger.warning("Monitor probe unavailable [%s]: %s", component, type(error).__name__)
                data = None
            source = ObservationSource.GPU_PROBE if component == 'gpus' else ObservationSource.SERVICE_PROBE
            if data is None and previous is not None:
                observation = previous.failed(attempted_at=attempted)
            else:
                observation = Observation.from_read(source, available=data is not None,
                    observed_at=datetime.now(UTC) if data is not None else None, ttl_s=6, attempted_at=attempted)
            return ProbeSample(data, observation)

        sample = await self._work.run(capture)
        self._last[component] = sample.observation
        return sample

    async def aclose(self) -> None:
        """Close admission and retain the drain through repeated caller cancellation."""
        self._closed = True
        if self._closing is None:
            self._closing = asyncio.create_task(self._drain(), context=Context())
        await asyncio.shield(self._closing)

    async def _drain(self) -> None:
        await asyncio.gather(*tuple(self._tasks.values()), return_exceptions=True)
        await self._work.aclose()
