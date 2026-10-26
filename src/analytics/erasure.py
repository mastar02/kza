"""Bounded cleanup of registered policy event databases, without legacy edits."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial

from src.core.owned_work import OwnedBlockingWork
from src.analytics.event_storage import (
    event_descriptor,
    event_sources,
    existing_event_database,
    validate_event_descriptor,
)
from src.analytics.retention import EventRetention
from src.memory.erasure_journal import ErasureParticipant, ErasureResult, ErasureStep
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.privacy_services import PrivacyServices


class EventEraser:
    """Scan source metadata in batches; revoked writers cannot refill old IDs."""

    def __init__(self, privacy: PrivacyServices, *, retention_days: float = 90):
        self.privacy = privacy
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="event-erasure"
        )
        self._work = OwnedBlockingWork(worker=self._offload)
        self._closing: asyncio.Task | None = None
        self.retention = EventRetention(privacy, retention_days=retention_days)

    async def _offload(self, function, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, copy_context().run, partial(function, *args, **kwargs)
        )

    def _clean(self, step: ErasureStep) -> ErasureResult:
        ledger = self.privacy.ledger
        descriptor = step.descriptor
        if step.participant is not ErasureParticipant.EVENTS or step.version != 1:
            raise ValueError("unsupported_event_target")
        validate_event_descriptor(descriptor)
        targets = ledger.erasures.targets(ErasureParticipant.EVENTS)
        if not any(
            row["id"] == step.target_id
            and row["version"] == step.version
            and row["descriptor"] == descriptor
            for row in targets
        ):
            raise ValueError("unregistered_event_target")
        if not ledger.authority_available or not ledger.conversation_revoked(
            step.conversation_id
        ):
            raise RuntimeError("event_erasure_authority_unavailable")
        cursor = step.cursor
        if cursor is not None and (
            not cursor.isascii()
            or not cursor.isdecimal()
            or len(cursor) > 19
            or not 0 < int(cursor) <= 2**63 - 1
        ):
            raise ValueError("invalid_event_cursor")
        with existing_event_database(descriptor) as db:
            if db is None:
                return ErasureResult(True, pending_count=0)
            rows = db.execute(
                """SELECT id,sources_json,
                (trigger_phrase IS NOT NULL OR user_name IS NOT NULL OR context_json IS NULL OR context_json!=metrics_json)
                FROM events_policy_v1 WHERE id>? ORDER BY id LIMIT 128""",
                (int(cursor) if cursor else 0,),
            ).fetchall()
            ids = []
            for row_id, encoded, raw in rows:
                sources = event_sources(encoded, raw)
                if any(
                    source.conversation_id == step.conversation_id for source in sources
                ) or ledger.sources_committed_unavailable(sources):
                    ids.append((row_id,))
            db.executemany("DELETE FROM events_policy_v1 WHERE id=?", ids)
        complete = len(rows) < 128
        return ErasureResult(
            complete,
            str(rows[-1][0]) if not complete else None,
            0 if complete else None,
        )

    async def clean(self, step: ErasureStep) -> ErasureResult:
        """Retain actual SQLite completion if the coordinator observer is cancelled."""
        return await self._work.run(self._clean, step)

    def begin_close(self) -> asyncio.Task:
        """Drain actual batches before shutting down their private native pool."""
        if self._closing is None:
            retention = self.retention.begin_close()
            drain = self._work.begin_close()

            async def finish():
                await retention
                await drain
                await asyncio.to_thread(self._executor.shutdown, wait=True)

            self._closing = asyncio.create_task(finish())
        return self._closing


async def build_event_erasure(
    config: dict, *, privacy: PrivacyServices, owner: ErasureCoordinator
) -> EventEraser:
    """Connect historical event cleanup even when analytics is disabled."""
    descriptor = await privacy.acall(
        event_descriptor,
        config.get("analytics", {}).get("events_db", "./data/events.db"),
    )
    await privacy.acall(
        privacy.ledger.erasures.register_target,
        ErasureParticipant.EVENTS,
        1,
        descriptor,
    )
    driver = EventEraser(
        privacy, retention_days=config.get("analytics", {}).get("retention_days", 90)
    )
    owner.register(
        ErasureParticipant.EVENTS,
        driver.clean,
        versions=frozenset({1}),
        begin_close=driver.begin_close,
    )
    driver.retention.start()
    return driver
