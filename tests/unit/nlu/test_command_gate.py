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
    assert d.signals == {
        "no_speech_prob": None, "avg_logprob": None, "compression_ratio": None,
    }


def test_reject_signals_carry_null_confidence():
    d = _gate().evaluate("")
    assert d.signals == {
        "no_speech_prob": None, "avg_logprob": None, "compression_ratio": None,
    }


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


def test_accepts_activa_la_routine_command():
    """'activá la rutina/alarma/escena' NO debe rechazarse como ruido.

    El noise phrase 'activa la' (substring) era over-broad: false-rejectaba
    comandos válidos. Con engine=openwakeword el gate corre con wake_words=()
    (sin missing_wake), así que el noise phrase era el único filtro que los
    bloqueaba. (2026-06-02.)
    """
    g = CommandAcceptanceGate(wake_words=())  # prod openwakeword
    for cmd in ("activá la rutina de la mañana", "activá la alarma", "activá la escena cine"):
        d = g.evaluate(cmd)
        assert d.accept is True, f"rechazó {cmd!r}: {d.reason}"


def test_still_rejects_youtube_noise_without_wake():
    """Las otras noise phrases siguen rechazando (no rompimos el gate)."""
    g = CommandAcceptanceGate(wake_words=())
    assert g.evaluate("suscribite al canal de youtube").accept is False


# ============================================================
# Guard de compression_ratio (2026-06-04) — anti-alucinación validado
# (openai/whisper #2378): texto repetitivo ("la luz la luz la luz...")
# comprime alto; un comando real corto nunca llega a 2.2 (overhead zlib).
# A diferencia de no_speech_prob/avg_logprob (muertos en turbo — N-2 del
# reporte XVF3800), esta señal SÍ separa → enforce flag propio.
# ============================================================

def _conf_cr(cr, text="x"):
    return STTResult(text, 1.0, compression_ratio=cr)


def test_high_compression_rejected_when_enforcing():
    g = _gate(enforce_compression_ratio=True, max_compression_ratio=2.2)
    d = g.evaluate("nexa prendé la luz la luz la luz la luz", _conf_cr(3.5))
    assert d.accept is False
    assert "high_compression" in d.reason


def test_high_compression_shadow_accepts_but_flags():
    # Default: shadow (mismo patrón que enforce_confidence — calibrar antes).
    g = _gate(max_compression_ratio=2.2)
    d = g.evaluate("nexa prendé la luz la luz la luz la luz", _conf_cr(3.5))
    assert d.accept is True
    assert "high_compression" in d.signals.get("would_reject", "")


def test_normal_compression_accepts():
    g = _gate(enforce_compression_ratio=True, max_compression_ratio=2.2)
    d = g.evaluate("nexa prendé la luz del escritorio", _conf_cr(1.4))
    assert d.accept is True
    assert d.reason == "ok"


def test_none_compression_not_penalized():
    g = _gate(enforce_compression_ratio=True, max_compression_ratio=2.2)
    d = g.evaluate("nexa prendé la luz", STTResult("x", 1.0))
    assert d.accept is True


def test_compression_signal_surfaced():
    g = _gate()
    d = g.evaluate("nexa prendé la luz", _conf_cr(1.7))
    assert d.signals.get("compression_ratio") == 1.7


def test_hard_rule_wins_over_compression():
    # hard rule (empty) tiene prioridad sobre compression como reason
    g = _gate(enforce_compression_ratio=True, max_compression_ratio=2.2)
    d = g.evaluate("", _conf_cr(9.9))
    assert d.accept is False
    assert d.reason == "empty"


# --- BoH-es: blocklist + prompt_echo (spec 2026-07-05) ---

REAL_PROMPT = (
    "Esto es un asistente de voz llamado Nexa que controla luces, aire "
    "acondicionado, persianas y música en el escritorio, el living, la cocina, "
    "el baño y el hall. Habla rioplatense con voseo: prendé, apagá, subí, "
    "bajá, poné."
)


def test_rejects_aplausos_as_noise():
    d = _gate().evaluate("¡Aplausos!")
    assert d.accept is False
    assert d.reason.startswith("noise_phrase")


def test_prompt_echo_rejects_prompt_fragment():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("Esto es un asistente de voz.")
    assert d.accept is False
    assert d.reason == "prompt_echo"


def test_prompt_echo_rejects_slightly_garbled_fragment():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("Esto es un asistente de vos")
    assert d.accept is False
    assert d.reason == "prompt_echo"


def test_prompt_echo_does_not_reject_real_commands():
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    for cmd in (
        "nexa prendé la luz del escritorio",
        "nexa subí el volumen en el living",
        "activá la escena lectura",
        "apagá el aire acondicionado del living",
    ):
        d = g.evaluate(cmd)
        assert d.accept is True, f"false-reject: {cmd!r} → {d.reason}"


def test_prompt_echo_skips_short_texts():
    # <4 palabras jamás dispara prompt_echo aunque estén en el prompt
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("en el living")
    assert d.reason != "prompt_echo"


def test_prompt_echo_inactive_without_prompt():
    d = CommandAcceptanceGate().evaluate("Esto es un asistente de voz.")
    assert d.accept is True


def test_prompt_echo_does_not_reject_multiroom_enumeration():
    # Regresión review 2026-07-05: con ratio 0.8 esto daba prompt_echo (0.879, margen 0.021)
    g = CommandAcceptanceGate(initial_prompt=REAL_PROMPT)
    d = g.evaluate("luces del escritorio el living la cocina el baño y el hall")
    assert d.accept is True


def test_prompt_echo_survives_long_prompt_sentences():
    # Regresión review 2026-07-05: con autojunk=True (default), en oraciones
    # >=200 chars difflib purga los chars frecuentes y find_longest_match
    # devuelve size 0 → eco verbatim aceptado en silencio. Fixture all-prose
    # a propósito: un fixture con dígitos deja anclas que sobreviven la purga
    # y el test pasaría aun sin el fix.
    nombres = (
        "principal", "de visitas", "del fondo", "de arriba", "de abajo",
        "del pasillo", "de servicio", "de invitados", "del altillo",
        "de la terraza", "del garage", "de la entrada", "del sótano",
        "de almacenaje", "de trabajo", "de descanso", "de juego",
    )
    rooms = ", ".join(f"el cuarto {n} con la lampara {n} y la persiana {n}" for n in nombres)
    long_prompt = f"Esto es un asistente de voz que controla todas las luces y persianas en {rooms}. Puedes hablarle en español."
    g = CommandAcceptanceGate(initial_prompt=long_prompt)
    d = g.evaluate("el cuarto de visitas con la lampara de visitas y la persiana de visitas")
    assert d.accept is False
    assert d.reason == "prompt_echo"
