"""Construction boundary for persistent routines using Casa's shared dependencies."""

import asyncio

from src.home_assistant.home_actions import HomeActionService
from src.routines.routine_executor import RoutineExecutor
from src.routines.routine_scheduler import RoutineScheduler
from src.routines.routine_store import RoutineStore


async def build_routine_scheduler(
    *, db_path: str, timezone: str, ha_client, home_actions: HomeActionService,
) -> RoutineScheduler:
    """Initialize storage and construct a scheduler without starting its loop.

    Args:
        db_path: SQLite database owned by the routines subsystem.
        timezone: Supported local timezone for scheduled occurrences.
        ha_client: Shared HA client owned by the application.
        home_actions: Shared Casa action service owned by the application.

    Returns:
        A constructed scheduler whose loop the application starts explicitly.

    Raises:
        BaseException: Construction or cancellation errors after closing the
            owned store. Shared HA and action dependencies remain application-owned.
    """
    store = RoutineStore(db_path)
    try:
        await store.initialize()
        executor = RoutineExecutor(ha_client=ha_client, home_actions=home_actions, store=store)
        return RoutineScheduler(executor, ha_client=ha_client, store=store, timezone=timezone)
    except BaseException:
        # Keep ownership until close completes even if cancellation repeats.
        cleanup = asyncio.create_task(store.close())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError
        raise
