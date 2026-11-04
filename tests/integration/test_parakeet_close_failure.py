"""A failed real graph close must retain privacy; child exit contains fixtures."""
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
from src.stt.parakeet_stt import ParakeetSTT
from src.stt.whisper_fast import create_stt
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import OfflineSTT, offline_config, offline_home

ROOT = Path(__file__).resolve().parents[2]


def _child(state_dir, receipt_path):
    # The actual fixture changes cwd; preserve this checkout for later imports.
    sys.path.insert(0, str(ROOT))
    monkeypatch = pytest.MonkeyPatch()
    state = Path(state_dir)
    state.mkdir()
    providers = application_fixtures.leaves.__wrapped__(monkeypatch, state)

    async def scenario():
        assert 'onnx_asr' not in sys.modules

        def stt_factory(config):
            return create_stt(config) if config.get('engine') == 'parakeet' else OfflineSTT()

        async with offline_home() as (url, observations):
            config = offline_config(state, url)
            config.setdefault('stt', {}).update(
                engine='parakeet', shadow_engine=None,
                parakeet_model='synthetic-parakeet',
            )
            stack = AsyncExitStack()
            await stack.__aenter__()
            graph = await build_application(
                config, shutdown=stack, factories=replace(providers, stt=stt_factory),
            )
            model = graph.models.stt
            assert type(model) is ParakeetSTT
            assert graph.stt_executors._models[id(model)] is model
            sentinel = object()
            model._model = sentinel
            disposals = []
            original_dispose = model._parakeet_release_resources
            original_condition = model._parakeet_condition
            first_result = {}

            def dispose():
                disposals.append(threading.current_thread())
                return original_dispose()

            class ElectionControl(BaseException):
                pass

            class ConditionLeaf:
                faults = 0

                def __enter__(self):
                    return original_condition.__enter__()

                def __exit__(self, *args):
                    result = original_condition.__exit__(*args)
                    selected = model._parakeet_close
                    if (self.faults == 0 and selected is not None
                            and selected.owner is first and selected.result is None):
                        self.faults += 1
                        raise ElectionControl()
                    return result

                def __getattr__(self, name):
                    return getattr(original_condition, name)

            def first_close():
                try:
                    model.close()
                except BaseException as error:
                    first_result['fixed_failure'] = (
                        type(error) is RuntimeError
                        and error.args == ('parakeet_close_failed',)
                    )
                else:
                    first_result['returned'] = True

            fault = ConditionLeaf()
            model._parakeet_condition = fault
            model._parakeet_release_resources = dispose
            first = threading.Thread(target=first_close, daemon=True)
            first.start()
            await asyncio.to_thread(first.join, 3)
            assert not first.is_alive() and first_result == {'fixed_failure': True}
            assert fault.faults == 1 and not disposals
            assert model._model is sentinel

            closing = asyncio.create_task(stack.aclose())
            results = await asyncio.gather(
                asyncio.wait_for(asyncio.shield(closing), 8), return_exceptions=True,
            )
            assert closing.done() and not closing.cancelled()
            assert isinstance(results[0], BaseException)
            assert closing.exception() is not None
            actual = graph.stt_executors._closing
            model_actual = model._parakeet_closing
            assert actual is not None and model_actual is not None
            assert graph.stt_executors.begin_close() is actual
            assert model.begin_close() is model_actual
            for operation in (actual, model_actual):
                assert operation.done() and not operation.cancelled()
                error = operation.exception()
                assert type(error) is RuntimeError
                assert error.args == ('parakeet_close_failed',)
            assert model._model is sentinel and not disposals
            assert graph.stt_executors._models[id(model)] is model
            assert graph.privacy.resource_status()['writer_lease']
            assert graph.privacy.ledger._db is not None
            assert graph.privacy.ledger._db.execute('SELECT 1').fetchone()[0] == 1
            assert observations['service_calls'] == 0
            assert 'onnx_asr' not in sys.modules
            return {
                'real_factory_stack_close_failed': True,
                'actual_first_thread_joined': True,
                'model_and_stt_tasks_terminal_failed': True,
                'model_reference_retained': True,
                'disposals': len(disposals),
                'privacy_writer_lease_retained': True,
                'privacy_ledger_db_retained': True,
                'service_calls': observations['service_calls'],
                'sdk_imported': False,
                'process_exit_is_containment_only': True,
            }

    receipt = asyncio.run(scenario())
    Path(receipt_path).write_text(json.dumps(receipt, indent=2)+'\n')
    # Do not force privacy close, reset a token, or claim native retirement.
    os._exit(0)


@pytest.mark.usefixtures('dashboard_auth')
def test_real_factory_stack_failure_retains_parakeet_and_privacy_authority(tmp_path):
    receipt = tmp_path/'graph-failure-receipt.json'
    command = (
        'import sys; from tests.integration.test_parakeet_close_failure import _child; '
        '_child(sys.argv[1], sys.argv[2])'
    )
    result = subprocess.run(
        [sys.executable, '-c', command, str(tmp_path/'failed-graph'), str(receipt)],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=30, check=False,
    )
    (tmp_path/'graph-child.stdout.log').write_text(result.stdout)
    (tmp_path/'graph-child.stderr.log').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    proof = json.loads(receipt.read_text())
    assert proof['real_factory_stack_close_failed']
    assert proof['actual_first_thread_joined']
    assert proof['model_and_stt_tasks_terminal_failed']
    assert proof['model_reference_retained'] and proof['disposals'] == 0
    assert proof['privacy_writer_lease_retained'] and proof['privacy_ledger_db_retained']
    assert proof['service_calls'] == 0 and not proof['sdk_imported']
    assert proof['process_exit_is_containment_only']
