"""Memory close cannot starve native retirement callbacks on CPython 3.13."""

import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('mode', ['result', 'error', 'cancel', 'native'])
async def test_memory_close_retires_ready_callbacks_and_keeps_actual_native_ownership(tmp_path, mode):
    # The watchdog is outside the affected event loop, so a regression fails
    # within a bound even when that loop cannot execute its own timeout.
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'tests.mocks.memory_close', mode, str(tmp_path),
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(3):
            stdout, stderr = await process.communicate()
    except TimeoutError:
        pytest.fail('memory close starved its retirement callbacks')
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert json.loads(stdout) == {'mode': mode, 'passed': True, 'operations': 0, 'writer_lease': False}
