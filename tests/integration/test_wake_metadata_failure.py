"""Real Wake teardown failure keeps privacy; child exit contains failed handles."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from src.main import build_application
from src.wakeword.detector import WakeWordDetector
from tests.integration import test_application_graph as application_fixtures
from tests.integration.test_wake_graph_retirement import configured
from tests.mocks.application_graph import offline_home

ROOT = Path(__file__).resolve().parents[2]


def _child(state_dir, receipt_path):
    sys.path.insert(0, str(ROOT))
    state = Path(state_dir)
    state.mkdir()
    patch = pytest.MonkeyPatch()
    providers = application_fixtures.leaves.__wrapped__(patch, state)

    async def scenario():
        assert all(n not in sys.modules for n in ('openwakeword', 'onnxruntime'))
        async with offline_home() as (url, observations):
            stack = AsyncExitStack()
            await stack.__aenter__()
            graph = await build_application(
                configured(state, url), shutdown=stack,
                factories=replace(providers, wake_detector=None),
            )
            detector = graph.pipeline.audio_loop.room_streams['living'].wake_detector
            assert type(detector) is WakeWordDetector
            assert graph.wake_models._models[id(detector)] is detector
            callback_calls, resets, disposal_entries, disposal_returns = [], [], [], []

            def callback(*args):
                callback_calls.append(1)

            class SyntheticBackend:
                def reset(self):
                    resets.append(1)

            backend = SyntheticBackend()
            detector.on_detection(callback)
            detector._oww_model = backend
            assert not detector._loaded and detector._wake_close is None
            original_dispose = detector._wake_release_resources
            original_condition = detector._wake_condition
            original_begin_close = detector.begin_close
            selected_tasks = []

            def dispose():
                disposal_entries.append(threading.current_thread())
                result = original_dispose()
                disposal_returns.append(threading.current_thread())
                return result

            def observed_begin_close():
                task = original_begin_close()
                selected_tasks.append(task)
                return task

            class ElectionControl(BaseException):
                pass

            class ConditionLeaf:
                faults = 0

                def __enter__(self):
                    return original_condition.__enter__()

                def __exit__(self, *args):
                    result = original_condition.__exit__(*args)
                    selected = detector._wake_close
                    if (self.faults == 0 and selected is not None
                            and selected.owner is threading.current_thread()
                            and selected.result is None and not disposal_entries):
                        self.faults += 1
                        raise ElectionControl()
                    return result

                def __getattr__(self, name):
                    return getattr(original_condition, name)

            fault = ConditionLeaf()
            detector._wake_condition = fault
            detector._wake_release_resources = dispose
            detector.begin_close = observed_begin_close
            # The real stack drains actual borrowers before selecting the model.
            closing = asyncio.create_task(stack.aclose())
            results = await asyncio.gather(
                asyncio.wait_for(asyncio.shield(closing), 8), return_exceptions=True,
            )
            assert closing.done() and not closing.cancelled()
            assert isinstance(results[0], BaseException) and closing.exception() is not None
            assert selected_tasks and detector._wake_closing is selected_tasks[0]
            assert all(task is selected_tasks[0] for task in selected_tasks)
            actual = graph.wake_models._closing
            assert actual is not None and graph.wake_models.begin_close() is actual
            with pytest.raises(RuntimeError, match='^wake_close_failed$'):
                await asyncio.wait_for(asyncio.shield(actual), 3)
            for task in (actual, selected_tasks[0]):
                assert task.done() and not task.cancelled()
                error = task.exception()
                assert type(error) is RuntimeError and error.args == ('wake_close_failed',)
            assert fault.faults == 1 and not disposal_entries and not disposal_returns
            assert detector._oww_model is backend
            assert graph.wake_models._models[id(detector)] is detector
            assert graph.privacy.resource_status()['writer_lease']
            assert graph.privacy.ledger._db is not None
            assert graph.privacy.ledger._db.execute('SELECT 1').fetchone()[0] == 1
            assert observations['service_calls'] == 0 and not callback_calls
            assert all(n not in sys.modules for n in ('openwakeword', 'onnxruntime'))
            return {
                'real_factory_stack_close_failed': True,
                'selected_original_begin_close_invoked': True,
                'actual_model_and_wake_owner_tasks_failed': True,
                'synthetic_backend_reference_retained': True,
                'disposal_entries': len(disposal_entries),
                'disposal_returns': len(disposal_returns),
                'borrower_reset_leaf_invocations': len(resets),
                'privacy_writer_lease_retained': True,
                'privacy_ledger_usable': True,
                'callback_invocations': len(callback_calls),
                'service_calls': observations['service_calls'],
                'SDK_imported': False,
                'process_exit_is_containment_only': True,
            }

    proof = asyncio.run(scenario())
    Path(receipt_path).write_text(json.dumps(proof, indent=2)+'\n')
    os._exit(0)


@pytest.mark.usefixtures('dashboard_auth')
def test_real_wake_election_failure_keeps_backend_and_privacy_authority(tmp_path):
    receipt = tmp_path/'wake-graph-failure.json'
    command = (
        'import sys; from tests.integration.test_wake_metadata_failure import _child; '
        '_child(sys.argv[1], sys.argv[2])'
    )
    result = subprocess.run(
        [sys.executable, '-c', command, str(tmp_path/'failed-wake-graph'), str(receipt)],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=30, check=False,
    )
    (tmp_path/'child.stdout.log').write_text(result.stdout)
    (tmp_path/'child.stderr.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    proof = json.loads(receipt.read_text())
    assert proof['real_factory_stack_close_failed']
    assert proof['selected_original_begin_close_invoked']
    assert proof['actual_model_and_wake_owner_tasks_failed']
    assert proof['synthetic_backend_reference_retained']
    assert proof['disposal_entries'] == proof['disposal_returns'] == 0
    assert proof['privacy_writer_lease_retained'] and proof['privacy_ledger_usable']
    assert proof['callback_invocations'] == proof['service_calls'] == 0
    assert not proof['SDK_imported'] and proof['process_exit_is_containment_only']
