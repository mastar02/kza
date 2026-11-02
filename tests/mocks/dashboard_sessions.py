"""Test clients that actually own their private authority on one portal loop."""

from contextlib import AsyncExitStack, asynccontextmanager, contextmanager

from fastapi.testclient import TestClient

from src.dashboard.authority import DashboardAuthority


@contextmanager
def session_client(app_factory, path, **kwargs):
    """Build an explicitly injected test app and drain it before its authority."""
    authority = DashboardAuthority(path)
    result = app_factory(authority)
    app, cleanup = result if isinstance(result, tuple) else (result, None)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with AsyncExitStack() as stack:
            stack.push_async_callback(authority.aclose)
            if cleanup is not None:
                stack.push_async_callback(cleanup)
            await authority.initialize()
            async with original(application) as state:
                yield state

    app.router.lifespan_context = lifespan
    with TestClient(app, **kwargs) as client:
        yield client
