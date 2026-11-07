"""Real dashboard/native lifetime under the existing synthetic admission fixture.

The fixture stubs admission verification. This test does not prove compiled
authority ancestry, SDK close failure, or global inventory completeness.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import errno
import json
import os
import sqlite3
import threading

import pytest

from src.core.owned_work import current_native_execution, validate_native_execution
from src.dashboard.authority import DashboardAuthority
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration
from tests.unit.memory.test_output_registration_families import managed


@pytest.mark.asyncio
async def test_cancelled_poisoned_final_rotation_keeps_actual_worker_and_lease(tmp_path, monkeypatch):
    async with managed(tmp_path, monkeypatch) as (owner, stack, config):
        token = {"value": "synthetic-before-rotation"}
        store = DashboardAuthority(
            config["dashboard"]["auth"]["db_path"],
            expected_token=lambda: token["value"],
        )
        stack.push_async_callback(store.aclose)
        await registration.attach_output(store, "authority_sessions")
        await store.initialize()
        original_work, original_lease = store._work, store._lease_fd
        previous_epoch, previous_digest = store._persisted_epoch, store._persisted_digest
        binding = owner._roles["authority_sessions"]
        assert original_lease is not None
        assert "privacy_authority" not in owner._roles

        # Change real persisted metadata, then let the original public path
        # discover the stale epoch and poison its coordinator before any open.
        epoch_path = owner._directory / "epoch.json"
        epoch = json.loads(epoch_path.read_text())
        digest = epoch["config_sha256"]
        epoch["config_sha256"] = ("1" if digest[0] == "0" else "0") + digest[1:]
        epoch_path.write_bytes(base._encoded(epoch))
        poisoned_journal = {path.name: path.read_bytes() for path in owner._directory.iterdir()}
        token["value"] = "synthetic-after-rotation"

        entered, release, actually_closed = threading.Event(), threading.Event(), threading.Event()
        connect_calls, close_entries = [], []
        actual_connect = sqlite3.connect

        class HeldFinalClose(sqlite3.Connection):
            def close(self):
                execution = current_native_execution()
                close_entries.append({
                    "execution": execution,
                    "actual_final_work": execution is not None and execution.work is binding.final_work,
                    "actual_original_method": execution is not None
                    and getattr(execution.function, "__self__", None) is store
                    and getattr(execution.function, "__func__", None) is DashboardAuthority._transaction,
                    "retained_sequence": None if execution is None else execution.parent_sequence,
                    "valid_inside_native_thread": validate_native_execution(execution, work=binding.final_work),
                })
                entered.set()
                try:
                    if not release.wait(5):
                        raise TimeoutError("test did not release original final native close")
                finally:
                    # Exactly the original owner's close calls actual SQLite
                    # close, including bounded cleanup if a test assertion fails.
                    super().close()
                    actually_closed.set()

        def observed_connect(*args, **kwargs):
            connect_calls.append(args[0])
            if binding.final_work is not None:
                kwargs = dict(kwargs, factory=HeldFinalClose)
            return actual_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", observed_connect)
        with pytest.raises(RuntimeError, match="^output_registration_epoch_changed$"):
            await store.synchronize()
        assert owner._failed and connect_calls == []
        assert store._ready and store._lease_fd == original_lease

        observer = asyncio.create_task(store.aclose())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            retained_close, final_work = store._closing, binding.final_work
            assert retained_close is not None and retained_close is not observer and not retained_close.done()
            assert final_work is not None and final_work is not original_work
            assert original_work.status()["closed"] and original_work.status()["operations"] == 0
            assert final_work.status()["operations"] == 1
            assert len(close_entries) == 1 and len(connect_calls) == 1
            observed = close_entries[0]
            assert observed["actual_final_work"] and observed["actual_original_method"]
            assert observed["valid_inside_native_thread"]
            assert observed["retained_sequence"] is binding.final_sequence
            assert not binding.final_sequence.done()
            assert len(owner._open_extents) == 1 and not actually_closed.is_set()

            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert store._closing is retained_close and not retained_close.done()
            assert final_work.status()["operations"] == 1
            assert store._lease_fd == original_lease
            os.fstat(original_lease)  # The actual lease FD is still open.
            assert len(owner._open_extents) == 1 and not actually_closed.is_set()
            assert {path.name: path.read_bytes() for path in owner._directory.iterdir()} == poisoned_journal
        finally:
            release.set()
            if not observer.done():
                observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            await asyncio.wait_for(store.aclose(), 3)

        assert actually_closed.is_set() and len(close_entries) == 1
        assert store._closing.done() and store._lease_fd is None and not owner._open_extents
        assert not store._closing.cancelled() and store._closing.exception() is None
        assert binding.final_sequence.done() and not binding.final_sequence.cancelled()
        assert binding.final_sequence.exception() is None
        with pytest.raises(OSError) as lease_closed:
            os.fstat(original_lease)
        assert lease_closed.value.errno == errno.EBADF
        assert binding.final_work.status()["closed"] and binding.final_work.status()["operations"] == 0
        assert owner._failed
        assert {path.name: path.read_bytes() for path in owner._directory.iterdir()} == poisoned_journal
        with closing(actual_connect(f"file:{store.path}?mode=ro", uri=True)) as database:
            persisted = dict(database.execute("SELECT key,value FROM auth_meta"))
        assert int(persisted["legacy_epoch"]) > previous_epoch
        assert int(persisted["legacy_epoch"]) >= store._observed_epoch
        assert persisted["legacy_digest"] == store._observed_digest != previous_digest

    # Dashboard-only attachment has no PrivacyServices final_registry path.
    # The unchanged fixture permits exactly its incomplete synthetic epoch;
    # inspect the retained coordinator result and metadata drain explicitly.
    assert owner._closing.done() and not owner._closing.cancelled()
    error = owner._closing.exception()
    assert type(error) is RuntimeError and str(error) == "output_registration_epoch_incomplete"
    assert owner._disk.status()["closed"] and owner._disk.status()["operations"] == 0
    assert not owner._open_extents and owner._opening_extents == 0
