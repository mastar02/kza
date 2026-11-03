"""Bounded close observations preserve the actual initialization return owner."""

import asyncio
from contextlib import AsyncExitStack
import threading

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from tests.integration.test_privacy_authority import configuration


pytestmark = pytest.mark.asyncio


async def _wait_native(event: threading.Event) -> None:
    async with asyncio.timeout(3):
        while not event.is_set():
            await asyncio.sleep(.001)


async def test_repeated_close_observations_keep_initialized_native_return_owned(
    tmp_path, monkeypatch,
):
    initialized = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    instances = []
    initialize = PrivacyServices.initialize

    def held_native_return(self):
        # Acquire the real SQLite connection and writer lease before either gate.
        initialize(self)
        instances.append(self)
        initialized.set()
        assert release_first.wait(3)
        second_entered.set()
        assert release_second.wait(3)

    monkeypatch.setattr(PrivacyServices, "initialize", held_native_return)
    stack = AsyncExitStack()
    building = asyncio.create_task(
        build_privacy_services(configuration(tmp_path), shutdown=stack),
    )
    privacy = owner = None
    try:
        await _wait_native(initialized)
        privacy = instances[0]
        assert privacy.ledger._db is not None
        assert privacy.resource_status()["writer_lease"]

        building.cancel()
        with pytest.raises(asyncio.CancelledError):
            await building

        privacy._close_timeout_s = .02
        owner = privacy.begin_close()
        with pytest.raises(WorkCloseTimeout):
            await stack.aclose()
        assert privacy.begin_close() is owner
        assert not owner.done()
        assert privacy.resource_status()["operations"] == 1
        assert privacy.resource_status()["writer_lease"]
        assert privacy.ledger._db is not None

        release_first.set()
        await _wait_native(second_entered)
        with pytest.raises(WorkCloseTimeout):
            await privacy.aclose()
        assert privacy.begin_close() is owner
        assert not owner.done()
        assert privacy.resource_status()["operations"] == 1
        assert privacy.resource_status()["writer_lease"]
        assert privacy.ledger._db is not None
    finally:
        release_first.set()
        release_second.set()
        if not building.done():
            building.cancel()
        await asyncio.gather(building, return_exceptions=True)
        if privacy is None and instances:
            privacy = instances[0]
        if privacy is not None:
            retained = privacy.begin_close()
            if owner is not None:
                assert retained is owner
            owner = retained
            await asyncio.shield(owner)
        await stack.aclose()

    assert privacy is not None and owner is not None
    assert privacy.begin_close() is owner
    assert privacy.ledger._db is None
    status = privacy.resource_status()
    assert status["operations"] == 0
    assert not status["writer_lease"]
    assert not status["closing"]
