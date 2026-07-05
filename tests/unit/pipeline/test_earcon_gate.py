from src.pipeline.earcon_gate import should_play_earcon


CFG = dict(enabled=True, min_wake_score=0.55, min_rms=0.02,
           reasons=("empty", "empty_after_norm", "high_compression", "low_confidence"))


def test_fires_on_empty_with_strong_wake_and_energy():
    assert should_play_earcon("empty", wake_score=0.81, rms=0.05, cfg=CFG) is True


def test_silent_on_noise_phrase_even_with_strong_wake():
    # TV/eco: NUNCA earcon.
    assert should_play_earcon("noise_phrase:'gracias por ver'", wake_score=0.9,
                              rms=0.1, cfg=CFG) is False


def test_silent_on_filler():
    assert should_play_earcon("filler_word:'si'", wake_score=0.9, rms=0.1, cfg=CFG) is False


def test_silent_on_weak_wake():
    assert should_play_earcon("empty", wake_score=0.41, rms=0.05, cfg=CFG) is False


def test_silent_on_low_energy():
    assert should_play_earcon("empty", wake_score=0.81, rms=0.005, cfg=CFG) is False


def test_high_compression_prefix_matches():
    assert should_play_earcon("high_compression:3.40>2.2", wake_score=0.81,
                              rms=0.05, cfg=CFG) is True


def test_disabled_never_fires():
    cfg = {**CFG, "enabled": False}
    assert should_play_earcon("empty", wake_score=0.99, rms=0.5, cfg=cfg) is False
