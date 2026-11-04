"""Real reusable lifecycle wrappers with actor-free synthetic model leaves."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from src.stt.parakeet_stt import ParakeetSTT
from src.tts.piper_tts import KokoroTTS


def capture_stt(*, recognize=None):
    class ASRLeaf:
        def recognize(self, audio, *, language):
            if recognize is not None:
                return recognize(audio, language=language)
            return "enciende la luz"

    model = ParakeetSTT()
    model._model = ASRLeaf()
    return model


def capture_tts():
    class PipelineLeaf:
        def __call__(self, text, *, voice):
            yield SimpleNamespace(audio=np.zeros(240, dtype=np.float32))

    model = KokoroTTS(device="cpu")
    model._model = PipelineLeaf()
    return model


def capture_factories(factories):
    """Keep graph/store/HTTP fixtures; replace only foreign speech wrappers."""
    return replace(factories, stt=lambda _: capture_stt(), tts=lambda _: capture_tts())
