"""Deterministic ready-queue crossings and an actual retained native worker."""

import asyncio
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

from src.core.owned_work import WorkCloseTimeout
from src.memory.long_term import LongTermMemory, _invoke_memory
from src.memory.privacy_services import PrivacyServices


async def exercise(mode: str, root: Path) -> dict:
    """Close real memory/SQLite ownership across a controlled I/O completion."""
    privacy = PrivacyServices(str(root/'privacy.db'), close_timeout_s=.02)
    await privacy.acall(privacy.initialize)
    memory = LongTermMemory(None, privacy=privacy)
    try:
        if mode == 'native':
            entered, release = threading.Event(), threading.Event()

            def native():
                entered.set()
                assert release.wait(2)
                # The shared authority must remain open through actual work.
                return privacy.ledger.status()

            observer = asyncio.create_task(memory.acall(native))
            try:
                async with asyncio.timeout(1):
                    while not entered.is_set():
                        await asyncio.sleep(.001)
                try:
                    await privacy.aclose()
                except WorkCloseTimeout:
                    pass
                else:
                    raise AssertionError('native work was closed prematurely')
                assert memory.resource_status()['operations'] == 1
                assert privacy.resource_status()['writer_lease']
            finally:
                release.set()
                await observer
            await privacy.aclose()
            assert not privacy.resource_status()['writer_lease']
        else:
            gate = asyncio.get_running_loop().create_future()
            entered, heartbeat = asyncio.Event(), asyncio.Event()
            original = asyncio.to_thread

            def operation():
                if mode == 'error':
                    raise ValueError('synthetic native error')
                if mode == 'cancel':
                    raise asyncio.CancelledError
                return 'synthetic result'

            async def controlled(function, *args, **kwargs):
                if function is _invoke_memory and args[0].function is operation:
                    entered.set()
                    await gate
                    return function(*args, **kwargs)
                return await original(function, *args, **kwargs)

            with patch('asyncio.to_thread', controlled):
                observer = asyncio.create_task(memory.acall(operation))
                await entered.wait()
                # The actual operation task is queued before the close task;
                # its retirement callback will be queued after the close task.
                gate.set_result(None)
                closing = memory.begin_close()
                asyncio.get_running_loop().call_soon(heartbeat.set)
                await memory.aclose()
                result, = await asyncio.gather(observer, return_exceptions=True)
                assert closing.done() and heartbeat.is_set()
                if mode == 'error':
                    assert isinstance(result, ValueError)
                elif mode == 'cancel':
                    assert isinstance(result, asyncio.CancelledError)
                else:
                    assert result == 'synthetic result'
            assert privacy.resource_status()['writer_lease']
        assert memory.resource_status()['operations'] == 0
    finally:
        await privacy.aclose()
    assert not privacy.resource_status()['writer_lease']
    return {'mode': mode, 'passed': True, 'operations': 0, 'writer_lease': False}


if __name__ == '__main__':
    print(json.dumps(asyncio.run(exercise(sys.argv[1], Path(sys.argv[2])))))
