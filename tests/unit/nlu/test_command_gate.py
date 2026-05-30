"""Tests: CommandAcceptanceGate."""
from src.nlu.command_gate import CommandAcceptanceGate, AcceptanceDecision
from src.stt.whisper_fast import STTResult


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


def test_accept_signals_carry_null_confidence():
    d = _gate().evaluate("nexa prendé la luz")
    assert d.signals == {"no_speech_prob": None, "avg_logprob": None}


def test_reject_signals_carry_null_confidence():
    d = _gate().evaluate("")
    assert d.signals == {"no_speech_prob": None, "avg_logprob": None}


def test_fail_open_on_internal_error(monkeypatch):
    import src.nlu.command_gate as m
    monkeypatch.setattr(m, "_normalize", lambda t: 1 / 0)
    d = _gate().evaluate("nexa prendé la luz")
    assert d.accept is True
    assert d.reason == "gate_error"


def _conf(no_speech, logprob):
    return STTResult("x", 1.0, no_speech_prob=no_speech, avg_logprob=logprob)


def test_low_confidence_rejected_when_enforcing():
    g = _gate(enforce_confidence=True, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.9, -2.0))
    assert d.accept is False
    assert "low_confidence" in d.reason


def test_low_confidence_shadow_accepts_but_flags():
    g = _gate(enforce_confidence=False, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.9, -2.0))
    assert d.accept is True                       # shadow -> pasa
    assert d.signals.get("would_reject")          # pero lo marca


def test_high_confidence_accepts():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("nexa prendé la luz", _conf(0.1, -0.3))
    assert d.accept is True


def test_none_confidence_not_penalized():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("nexa prendé la luz", None)
    assert d.accept is True


def test_hard_rule_wins_over_confidence():
    g = _gate(enforce_confidence=True)
    d = g.evaluate("", _conf(0.9, -2.0))
    assert d.accept is False
    assert d.reason == "empty"


def test_fail_open_on_exception():
    g = _gate(enforce_confidence=True)
    class Boom:
        @property
        def no_speech_prob(self): raise ValueError("boom")
        avg_logprob = -0.1
    d = g.evaluate("nexa prendé la luz", Boom())
    assert d.accept is True
    assert d.reason == "gate_error"


def test_only_nsp_exceeded_flags():
    g = _gate(enforce_confidence=True, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.9, -0.3))  # nsp bad, alp ok
    assert d.accept is False
    assert "no_speech" in d.reason and "avg_logprob" not in d.reason


def test_only_alp_exceeded_flags():
    g = _gate(enforce_confidence=True, max_no_speech_prob=0.6, min_avg_logprob=-1.2)
    d = g.evaluate("nexa prendé la luz", _conf(0.1, -2.0))  # nsp ok, alp bad
    assert d.accept is False
    assert "avg_logprob" in d.reason and "no_speech" not in d.reason
