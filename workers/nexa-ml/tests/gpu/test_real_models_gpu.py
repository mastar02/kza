"""Server-only smoke with the real models (``pytest -m gpu``); never runs in PR CI.

Requires the ``gpu`` extra, CUDA and pre-cached weights (offline). Reports RTF and
first-TTS-chunk latency; VRAM budgets are checked with ``nvidia-smi`` by the operator.
"""

import os
import pathlib
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("torch")
pytest.importorskip("faster_whisper")

from nexa_ml.config import load_config  # noqa: E402
from nexa_ml.runtime import MlRuntime, real_loaders  # noqa: E402

CONFIG = pathlib.Path(os.environ.get("NEXA_ML_CONFIG", "/etc/nexa/nexa.toml"))


@pytest.fixture
async def runtime():
    rt = MlRuntime(load_config(CONFIG), real_loaders(load_config(CONFIG)))
    await rt.load_all()
    try:
        yield rt
    finally:
        rt.close()


async def test_real_slots_load_and_answer(runtime: MlRuntime) -> None:
    level, detail = runtime.aggregate()
    assert level == "ok", detail
    audio = np.zeros(16_000 * 2, dtype=np.float32)
    opts = runtime.command.options(language="", beam_size=0, initial_prompt="", hotwords="")
    start = time.perf_counter()
    await runtime.command.transcribe(audio, opts)
    print(f"command RTF={(time.perf_counter() - start) / 2.0:.3f}")

    tts_opts = runtime.tts.options(text="Hola, encendí la luz de la cocina.", voice="", lang_code="", speed=0)
    start = time.perf_counter()
    first = None
    async for _chunk in runtime.tts.stream(tts_opts):
        first = first or time.perf_counter() - start
    assert first is not None and first < 2.0
    print(f"tts first chunk={first * 1000:.0f} ms")

    spk = await runtime.speaker.embed(
        np.random.default_rng(0).standard_normal(16_000).astype(np.float32) * 0.1
    )
    assert spk.vector.shape == (192,)
    emb = await runtime.embed.embed(["luz cocina"])
    assert emb.vectors.shape == (1, 1024)
