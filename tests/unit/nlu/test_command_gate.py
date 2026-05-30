"""Tests: CommandAcceptanceGate."""
from src.nlu.command_gate import CommandAcceptanceGate, AcceptanceDecision


def _gate(**kw):
    return CommandAcceptanceGate(wake_words=("nexa", "alexa"), **kw)


def test_accepts_real_command():
    d = _gate().evaluate("nexa prendé la luz del escritorio")
    assert d.accept is True
    assert d.reason == "ok"


def test_rejects_empty():
    assert _gate().evaluate("").accept is False


def test_rejects_noise_phrase():
    d = _gate().evaluate("nexa suscribite al canal de youtube")
    assert d.accept is False
    assert "noise_phrase" in d.reason


def test_rejects_filler_word():
    assert _gate().evaluate("gracias").accept is False


def test_rejects_word_repetition():
    assert _gate().evaluate("nexa nexa nexa nexa").accept is False


def test_rejects_missing_wake():
    d = _gate().evaluate("prendé la luz del escritorio")
    assert d.accept is False
    assert "missing_wake" in d.reason


def test_accept_has_empty_signals():
    d = _gate().evaluate("nexa prendé la luz")
    assert d.signals == {}


def test_reject_has_empty_signals():
    d = _gate().evaluate("")
    assert d.signals == {}


def test_fail_open_on_internal_error(monkeypatch):
    import src.nlu.command_gate as m
    monkeypatch.setattr(m, "_normalize", lambda t: 1 / 0)
    d = _gate().evaluate("nexa prendé la luz")
    assert d.accept is True
    assert d.reason == "gate_error"
