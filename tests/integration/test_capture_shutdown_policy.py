"""Policy compensation stays owned while its privacy authority is closing."""
from __future__ import annotations

import asyncio
import threading

import pytest

from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled

pytestmark = pytest.mark.asyncio


async def test_privacy_shutdown_after_running_commit_preserves_durable_mute(tmp_path, monkeypatch):
    committed, release, returned = threading.Event(), threading.Event(), threading.Event()
    workers = []
    async with controlled(tmp_path, monkeypatch) as g:
        # The production builder registers this exact borrower before loading.
        g.graph.privacy.add_borrower(g.control.begin_close)
        g.control.request(True, g.control.snapshot()['revision'], actor())
        await capture_phase(g.control, 'muted')
        write = g.control._write_mode

        def held(mode, revision, controller):
            result = write(mode, revision, controller)
            if mode == 'running':
                workers.append(threading.current_thread())
                committed.set()
                assert release.wait(5), 'real committed policy writer was not released'
                returned.set()
            return result

        monkeypatch.setattr(g.control, '_write_mode', held)
        closing = None
        try:
            g.control.request(False, g.control.snapshot()['revision'], actor())
            await until(committed.is_set)
            closing = g.graph.privacy.begin_close()
            assert len(workers) == 1 and workers[0].ident is not None
            assert not returned.is_set() and not closing.done()
            assert g.control.snapshot()['ingress_sealed']
            assert not g.control.snapshot()['total_mute_confirmed']
            assert g.graph.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            if closing is not None:
                async with asyncio.timeout(5):
                    await asyncio.shield(closing)
        assert returned.is_set() and closing.done()
        assert not g.graph.privacy.resource_status()['writer_lease']
        assert g.control.snapshot()['ingress_sealed']
        assert not g.control.snapshot()['total_mute_confirmed']

    # Reopen through actual initialization, without repairing policy rows.
    async with controlled(tmp_path, monkeypatch) as rebuilt:
        mode, _ = await rebuilt.graph.privacy.acall(rebuilt.control._load_policy)
        assert mode == 'muted'
        state = rebuilt.control.snapshot()
        assert state['target_muted'] and state['persisted_muted']
        assert state['ingress_sealed'] and state['total_mute_confirmed']
