"""Smoke: el gate se construye desde un dict de config."""
from src.nlu.command_gate import CommandAcceptanceGate


def test_gate_from_config_dict():
    cfg = {"enforce_confidence": True, "max_no_speech_prob": 0.5, "min_avg_logprob": -1.0}
    g = CommandAcceptanceGate(wake_words=("nexa",), **cfg)
    assert g._enforce_confidence is True
    assert g._max_no_speech_prob == 0.5
    assert g._min_avg_logprob == -1.0
