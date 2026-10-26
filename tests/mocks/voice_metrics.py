"""Standalone synthetic voice measurement; no model or hardware initialization."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from tests.mocks.llm_http import event
from tests.mocks.voice_graph import voice_graph


async def measure_long_response(monkeypatch):
    loop = asyncio.get_running_loop()
    phrases = [f'Oración número {i} suficientemente larga.' for i in range(100)]
    lags = []
    owner, engine, backend, endpoint, _ = voice_graph(monkeypatch,
        parts=[.2, *(event(text + ' ') for text in phrases), event(finish='stop')])
    try:
        for _ in range(10):
            stopped = asyncio.Event()

            async def heartbeat():
                while not stopped.is_set():
                    start = loop.time()
                    await asyncio.sleep(.005)
                    lags.append(max(0, (loop.time()-start-.005)*1000))

            monitor = asyncio.create_task(heartbeat())
            try:
                result = await asyncio.wait_for(owner.arespond('synthetic', zone_id='zone_origin'), 10)
            finally:
                stopped.set()
                await monitor
            assert result.success and result.spoken_segments == 100
            assert result.text == ''.join(text + ' ' for text in phrases)
            assert engine.calls == phrases and engine.peak_active == 1
            assert endpoint.body.closed and endpoint.client.is_closed()
            assert all(output.closed and output.finished.is_set() for output in backend.outputs)
            np.testing.assert_array_equal(backend.heard()[:, 0], np.concatenate([engine.audio(t) for t in phrases]))
            # Observation sinks belong to the test, not the application. Reuse
            # one live owner and release each verified run's captured PCM.
            engine.calls.clear()
            backend.outputs.clear()
    finally:
        await owner.aclose()
    ordered = sorted(lags)
    p95 = ordered[int(.95 * (len(ordered)-1))]
    metrics = {
        'repetitions': 10, 'sentences_per_run': 100, 'samples_per_run': 2400000,
        'provider_wait_ms': 200, 'heartbeat_interval_ms': 5, 'heartbeat_samples': len(lags),
        'loop_lag_p95_ms': p95, 'loop_lag_max_ms': max(lags), 'exact_pcm_runs': 10,
    }
    return metrics


if __name__ == '__main__':
    from pytest import MonkeyPatch
    with MonkeyPatch.context() as patch:
        Path(sys.argv[1]).write_text(json.dumps(asyncio.run(measure_long_response(patch))))
