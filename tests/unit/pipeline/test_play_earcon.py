import numpy as np
from src.pipeline.response_handler import ResponseHandler
from src.tts import piper_tts
from tests.mocks.audio_output import AudioBackend
from tests.mocks.speech import PCMEngine


def test_play_earcon_plays_loaded_asset(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, 'sd', backend)
    handler = ResponseHandler(PCMEngine())
    expected = np.ones(4800, dtype=np.float32)
    handler.set_earcon(expected, 24000)
    assert handler.play_earcon().success
    np.testing.assert_array_equal(backend.heard()[:, 0], expected)


def test_play_earcon_noop_when_no_asset(monkeypatch):
    backend = AudioBackend()
    monkeypatch.setattr(piper_tts, 'sd', backend)
    handler = ResponseHandler(PCMEngine())
    assert not handler.play_earcon().speech_handled
    assert backend.outputs == []
