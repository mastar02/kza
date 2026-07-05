import numpy as np
from src.pipeline.response_handler import ResponseHandler


def _make_handler():
    # ResponseHandler con un earcon pre-cargado y un zone_manager fake que
    # registra qué se reprodujo.
    h = ResponseHandler.__new__(ResponseHandler)
    h._earcon_audio = np.zeros(4800, dtype=np.float32)  # 200ms @ 24k
    h._earcon_sr = 24000
    h._active_zone_id = "zone_living"
    h.zone_manager = None
    h.played = []

    def _fake_play_array(audio, sr, zone_id):
        h.played.append((len(audio), sr, zone_id))

    h._play_earcon_array = _fake_play_array
    return h


def test_play_earcon_plays_loaded_asset():
    h = _make_handler()
    ResponseHandler.play_earcon(h, zone_id="zone_cocina")
    assert h.played == [(4800, 24000, "zone_cocina")]


def test_play_earcon_noop_when_no_asset():
    h = _make_handler()
    h._earcon_audio = None
    ResponseHandler.play_earcon(h, zone_id="zone_cocina")
    assert h.played == []
