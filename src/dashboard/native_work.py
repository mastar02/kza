"""Shared native ownership and original-authority fences for dashboard work."""

import asyncio
from collections.abc import Callable
from functools import partial

from fastapi import FastAPI

from src.core.owned_work import OwnedBlockingWork


class NativeAuthorityRetired(PermissionError):
    """The original dashboard authority no longer permits this native action."""


def require_current(current: Callable[[], bool]) -> None:
    """Adapt a detached boolean lease to a raising native publication guard."""
    if current() is not True:
        raise NativeAuthorityRetired("dashboard_authority_retired")


def _invoke_admin(current: Callable[[], bool], function: Callable, args: tuple, kwargs: dict):
    """Release private inputs from failed worker tracebacks after completion."""
    try:
        require_current(current)
        return function(*args, **kwargs)
    finally:
        current = function = None
        args, kwargs = (), {}


async def run_admin_work(owner: OwnedBlockingWork, current: Callable[[], bool],
                         function: Callable, *args, **kwargs):
    """Check original authority without retaining cancelled observer inputs."""
    try:
        return await owner.run(_invoke_admin, current, function, args, kwargs)
    finally:
        owner = current = function = None
        args, kwargs = (), {}


async def _invoke_admin_sequence(current: Callable[[], bool], function: Callable,
                                 args: tuple, kwargs: dict):
    """Retain only detached inputs through the complete admitted sequence."""
    try:
        require_current(current)
        return await function(*args, **kwargs)
    finally:
        current = function = None
        args, kwargs = (), {}


async def run_admin_sequence(owner: OwnedBlockingWork, current: Callable[[], bool],
                             function: Callable, *args, **kwargs):
    """Use one shared admission for native and asynchronous administrative steps."""
    try:
        return await owner.run_async(_invoke_admin_sequence, current, function, args, kwargs)
    finally:
        owner = current = function = None
        args, kwargs = (), {}


async def drain_admin_work(owner: OwnedBlockingWork) -> None:
    """Finish the real drain before permitting dependent models to close."""
    task = owner.begin_close()
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


def ensure_admin_work(app: FastAPI, owner: OwnedBlockingWork | None = None) -> OwnedBlockingWork:
    """Use one bounded owner for administrative jobs and private projections."""
    existing = getattr(app.state, "dashboard_admin_work", None)
    if existing is not None:
        if owner is not None and owner is not existing:
            raise ValueError("dashboard_admin_owner_mismatch")
        return existing
    owner = owner if owner is not None else OwnedBlockingWork(isolate_context=True)
    app.state.dashboard_admin_work = owner
    app.add_event_handler("shutdown", partial(drain_admin_work, owner))
    return owner
