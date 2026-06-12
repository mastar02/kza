"""Tests del AmbientGuard — compuerta acústica integral (spec 2026-06-05).

Máquina de estados NORMAL → STRICT → COOLDOWN por habitación, alimentada por
la tasa de capturas rechazadas. Reloj inyectado: cero sleeps.
"""
import numpy as np
import pytest

from src.pipeline.ambient_guard import (
    AmbientGuard,
    AmbientGuardConfig,
    GuardState,
    classify_outcome,
)
from src.pipeline.multi_room_audio_loop import compute_wake_vad


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_guard(clock=None, **overrides) -> AmbientGuard:
    cfg = AmbientGuardConfig(
        enabled=True,
        strict_entry_rejects=3,
        strict_entry_window_s=60.0,
        strict_exit_quiet_s=120.0,
        strict_wake_score=0.65,
        strict_min_rms=0.0,
        strict_min_spenergy=0.0,
        cooldown_entry_rejects=3,
        cooldown_entry_window_s=60.0,
        cooldown_duration_s=30.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return AmbientGuard(config=cfg, time_fn=clock or FakeClock())


class TestPassiveDefaults:
    def test_disabled_accepts_everything(self):
        guard = AmbientGuard()  # config default: enabled=False
        d = guard.on_wake("escritorio", score=0.01, rms=0.0)
        assert d.accept is True
        assert d.reason == "disabled"

    def test_disabled_never_escalates(self):
        guard = AmbientGuard()
        for _ in range(50):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_disabled_follow_up_allowed(self):
        guard = AmbientGuard()
        assert guard.follow_up_allowed("escritorio") is True


class TestNormalState:
    def test_accepts_any_score_in_normal(self):
        guard = make_guard()
        d = guard.on_wake("escritorio", score=0.41, rms=0.001)
        assert d.accept is True
        assert d.state is GuardState.NORMAL

    def test_follow_up_allowed_in_normal(self):
        guard = make_guard()
        assert guard.follow_up_allowed("escritorio") is True


class TestEscalationToStrict:
    def test_rejects_within_window_escalate(self):
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_rejects_outside_window_do_not_escalate(self):
        clock = FakeClock()
        guard = make_guard(clock=clock)
        guard.on_capture_result("escritorio", "noise")
        clock.advance(61.0)
        guard.on_capture_result("escritorio", "noise")
        clock.advance(61.0)
        guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_accepted_and_other_fail_do_not_escalate(self):
        guard = make_guard()
        for outcome in ("accepted", "other_fail", "accepted", "other_fail",
                        "accepted", "other_fail"):
            guard.on_capture_result("escritorio", outcome)
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_all_reject_kinds_count(self):
        guard = make_guard()
        for outcome in ("noise", "empty", "timeout"):
            guard.on_capture_result("escritorio", outcome)
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestStrictState:
    def _strict_guard(self, clock=None, **overrides):
        guard = make_guard(clock=clock, **overrides)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        return guard

    def test_low_score_rejected_in_strict(self):
        guard = self._strict_guard()
        d = guard.on_wake("escritorio", score=0.50, rms=0.05)
        assert d.accept is False
        assert d.reason == "strict_score"
        assert d.state is GuardState.STRICT

    def test_high_score_accepted_in_strict(self):
        guard = self._strict_guard()
        d = guard.on_wake("escritorio", score=0.80, rms=0.05)
        assert d.accept is True

    def test_strict_min_rms_enforced_when_configured(self):
        guard = self._strict_guard(strict_min_rms=0.02)
        d = guard.on_wake("escritorio", score=0.80, rms=0.01)
        assert d.accept is False
        assert d.reason == "strict_rms"

    def test_strict_min_spenergy_enforced_when_configured(self):
        guard = self._strict_guard(strict_min_spenergy=50.0)
        d = guard.on_wake("escritorio", score=0.80, rms=0.05, spenergy_peak=10.0)
        assert d.accept is False
        assert d.reason == "strict_spenergy"

    def test_spenergy_none_fails_open(self):
        # Sin lectura del chip (fail-open del controller) NO se bloquea voz.
        guard = self._strict_guard(strict_min_spenergy=50.0)
        d = guard.on_wake("escritorio", score=0.80, rms=0.05, spenergy_peak=None)
        assert d.accept is True

    def test_follow_up_blocked_in_strict(self):
        guard = self._strict_guard()
        assert guard.follow_up_allowed("escritorio") is False

    def test_exit_to_normal_after_quiet(self):
        clock = FakeClock()
        guard = self._strict_guard(clock=clock)
        clock.advance(121.0)  # > strict_exit_quiet_s sin rechazos
        assert guard.state_for("escritorio") is GuardState.NORMAL

    def test_guard_rejection_keeps_strict_alive(self):
        # Los rechazos del propio guard (TV sigue disparando wakes con score
        # bajo) refrescan el quiet timer → STRICT no expira mientras haya TV.
        clock = FakeClock()
        guard = self._strict_guard(clock=clock)
        clock.advance(100.0)
        guard.on_wake("escritorio", score=0.50, rms=0.05)  # rechazo del guard
        clock.advance(100.0)  # 200s desde la escalada, pero 100s desde el último rechazo
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_accepted_command_does_not_exit_strict(self):
        # Un comando real exitoso con TV de fondo NO saca de STRICT (la TV
        # sigue ahí); la salida es solo por quiet sostenido.
        guard = self._strict_guard()
        guard.on_capture_result("escritorio", "accepted")
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestCooldown:
    def _cooldown_guard(self, clock):
        guard = make_guard(clock=clock)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → COOLDOWN
        assert guard.state_for("escritorio") is GuardState.COOLDOWN
        return guard

    def test_capture_rejects_in_strict_escalate_to_cooldown(self):
        self._cooldown_guard(FakeClock())

    def test_everything_rejected_during_cooldown(self):
        guard = self._cooldown_guard(FakeClock())
        d = guard.on_wake("escritorio", score=0.99, rms=0.5)
        assert d.accept is False
        assert d.reason == "cooldown"

    def test_cooldown_expires_to_strict(self):
        clock = FakeClock()
        guard = self._cooldown_guard(clock)
        clock.advance(31.0)  # > cooldown_duration_s
        assert guard.state_for("escritorio") is GuardState.STRICT

    def test_guard_rejections_do_not_escalate_to_cooldown(self):
        # Rechazos a nivel guard (strict_score) son gratis: no gastan
        # Whisper/router → NO cuentan para COOLDOWN.
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(10):
            guard.on_wake("escritorio", score=0.50, rms=0.05)
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestPerRoomIsolation:
    def test_rooms_have_independent_state(self):
        guard = make_guard()
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        assert guard.state_for("living") is GuardState.NORMAL
        d = guard.on_wake("living", score=0.41, rms=0.01)
        assert d.accept is True


class TestClassifyOutcome:
    def test_success_is_accepted(self):
        assert classify_outcome({"success": True, "text": "prende la luz"}) == "accepted"

    def test_empty_text(self):
        assert classify_outcome({"success": False, "text": ""}) == "empty"
        assert classify_outcome({"success": False, "text": None}) == "empty"
        assert classify_outcome({}) == "empty"

    def test_gate_and_llm_rejections_are_noise(self):
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "gate_rejected"}) == "noise"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "llm_rejected:tv_phrase"}) == "noise"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "low_confidence:0.40"}) == "noise"

    def test_unavailable_and_timeout_are_timeout(self):
        # El LLMRouter produce rejection_reason="unavailable" en timeout/error
        # local → intent "llm_rejected:unavailable" debe clasificar timeout,
        # NO noise (por eso este check va ANTES del de llm_rejected).
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "llm_rejected:unavailable"}) == "timeout"
        assert classify_outcome(
            {"success": False, "text": "x", "intent": "timeout"}) == "timeout"

    def test_real_command_downstream_failure_is_other_fail(self):
        # Voz real que falló en HA: NO debe escalar el guard.
        assert classify_outcome(
            {"success": False, "text": "prende la luz", "intent": "domotics"}) == "other_fail"

    def test_unverified_intent_is_other_fail_not_noise(self):
        # unverified_intent = pasó CommandGate + el 7B dio is_command con
        # confianza alta; solo el verbo quedó garbleado por el STT far-field.
        # Eso es señal de humano real, no de TV — NO debe escalar el guard
        # (caso real 2026-06-11 17:57: 'Nexa, a perder a luz' rechazado y el
        # guard entró a STRICT 2s después, dejando sordo el sistema al usuario).
        assert classify_outcome(
            {"success": False, "text": "Nexa, a perder a luz.",
             "intent": "unverified_intent:turn_off"}) == "other_fail"


class TestConfigInvariant:
    def test_cooldown_clamped_when_exceeds_quiet(self):
        # cooldown_duration_s >= strict_exit_quiet_s would chain
        # COOLDOWN→STRICT→NORMAL in a single _refresh call. __post_init__
        # clamps it to strict_exit_quiet_s / 2 without raising.
        cfg = AmbientGuardConfig(
            enabled=True,
            cooldown_duration_s=150.0,
            strict_exit_quiet_s=30.0,
        )
        assert cfg.cooldown_duration_s == 15.0  # clamped to 30/2

    def test_clamped_cooldown_exits_to_strict_not_normal(self):
        # After clamp, COOLDOWN must still exit to STRICT (not jump to NORMAL).
        clock = FakeClock()
        cfg = AmbientGuardConfig(
            enabled=True,
            strict_entry_rejects=3,
            strict_entry_window_s=60.0,
            strict_exit_quiet_s=30.0,
            cooldown_duration_s=150.0,  # will be clamped to 15.0
            cooldown_entry_rejects=3,
            cooldown_entry_window_s=60.0,
        )
        guard = AmbientGuard(config=cfg, time_fn=clock)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → COOLDOWN
        assert guard.state_for("escritorio") is GuardState.COOLDOWN
        clock.advance(16.0)  # > clamped cooldown (15s), < strict_exit_quiet (30s)
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestCooldownRefreshesQuietTimer:
    def test_reject_during_cooldown_refreshes_quiet_timer(self):
        # Stale async capture results during COOLDOWN must refresh last_reject_at
        # so that STRICT survives after COOLDOWN expires.
        clock = FakeClock(t=1000.0)
        # Use strict_exit_quiet_s=120, cooldown_duration_s=30
        guard = make_guard(clock=clock)
        # Escalate to COOLDOWN
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → COOLDOWN

        cooldown_entry_t = clock.t  # t=1000

        # Advance to t+20: still in COOLDOWN; report a reject (stale async result)
        clock.advance(20.0)  # t=1020
        guard.on_capture_result("escritorio", "noise")  # refreshes last_reject_at to t=1020

        # Advance past cooldown (30s from entry): t+31 → exits to STRICT
        clock.t = cooldown_entry_t + 31.0  # t=1031
        assert guard.state_for("escritorio") is GuardState.STRICT

        # 119s after the in-COOLDOWN reject → still STRICT (quiet < 120s)
        clock.t = 1020.0 + 119.0  # t=1139
        assert guard.state_for("escritorio") is GuardState.STRICT

        # 120s+ after the in-COOLDOWN reject → NORMAL
        clock.t = 1020.0 + 120.0  # t=1140
        assert guard.state_for("escritorio") is GuardState.NORMAL


class TestPerRoomCooldownIndependence:
    def test_escritorio_cooldown_does_not_affect_living(self):
        clock = FakeClock()
        guard = make_guard(clock=clock)
        # Escalate escritorio to COOLDOWN
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → COOLDOWN
        assert guard.state_for("escritorio") is GuardState.COOLDOWN
        # living stays NORMAL and accepts any wake
        assert guard.state_for("living") is GuardState.NORMAL
        d = guard.on_wake("living", score=0.41, rms=0.01)
        assert d.accept is True
        assert d.state is GuardState.NORMAL


class TestFullCycleReEntry:
    def test_normal_strict_quiet_normal_then_strict_again(self):
        clock = FakeClock()
        guard = make_guard(clock=clock)
        # First cycle: NORMAL → STRICT → (quiet) → NORMAL
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        clock.advance(121.0)  # quiet > 120s
        assert guard.state_for("escritorio") is GuardState.NORMAL
        # Second cycle: 3 more rejects → STRICT again
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT


class TestStrictBoundaryScore:
    def test_exact_strict_wake_score_is_accepted(self):
        # Guard check is `score < strict_wake_score`, so score == threshold
        # must be ACCEPTED.
        clock = FakeClock()
        guard = make_guard(clock=clock, strict_wake_score=0.65)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # → STRICT
        assert guard.state_for("escritorio") is GuardState.STRICT
        d = guard.on_wake("escritorio", score=0.65, rms=0.05)
        assert d.accept is True
        assert d.reason == "ok"


class TestPostSuccessGrace:
    """Caso real 2026-06-06: 'apagá' pasó STRICT (wake 0.77) pero el 'prendé'
    encadenado salió 0.40-0.59 < 0.72 → rechazado. Un comando ACEPTADO es
    evidencia fuerte de usuario real → gracia post-éxito: follow_up permitido
    en STRICT por strict_follow_up_grace_s tras el último accepted."""

    def _strict_guard(self, clock):
        guard = make_guard(clock=clock, strict_follow_up_grace_s=12.0)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        assert guard.state_for("escritorio") is GuardState.STRICT
        return guard

    def test_follow_up_allowed_within_grace_after_accept(self):
        clock = FakeClock()
        guard = self._strict_guard(clock)
        assert guard.follow_up_allowed("escritorio") is False
        guard.on_capture_result("escritorio", "accepted")
        assert guard.follow_up_allowed("escritorio") is True
        clock.advance(11.0)
        assert guard.follow_up_allowed("escritorio") is True

    def test_grace_expires(self):
        clock = FakeClock()
        guard = self._strict_guard(clock)
        guard.on_capture_result("escritorio", "accepted")
        clock.advance(13.0)
        assert guard.follow_up_allowed("escritorio") is False

    def test_grace_zero_disables(self):
        clock = FakeClock()
        guard = make_guard(clock=clock, strict_follow_up_grace_s=0.0)
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")
        guard.on_capture_result("escritorio", "accepted")
        assert guard.follow_up_allowed("escritorio") is False

    def test_grace_does_not_apply_in_cooldown(self):
        clock = FakeClock()
        guard = self._strict_guard(clock)
        guard.on_capture_result("escritorio", "accepted")
        for _ in range(3):
            guard.on_capture_result("escritorio", "noise")  # STRICT → COOLDOWN
        assert guard.state_for("escritorio") is GuardState.COOLDOWN
        assert guard.follow_up_allowed("escritorio") is False

    def test_noise_does_not_open_grace(self):
        clock = FakeClock()
        guard = self._strict_guard(clock)
        # other_fail tampoco abre gracia (no fue acción confirmada)
        guard.on_capture_result("escritorio", "other_fail")
        assert guard.follow_up_allowed("escritorio") is False


def make_strict_guard(clock=None, **overrides):
    """Guard ya escalado a STRICT, para probar on_wake con wake_vad."""
    guard = make_guard(clock=clock, **overrides)
    for _ in range(3):  # strict_entry_rejects=3
        guard.on_capture_result("escritorio", "noise")
    assert guard.state_for("escritorio") is GuardState.STRICT
    return guard


class TestVadAdaptiveOnWake:
    def test_enforce_high_vad_accepts_mid_score(self):
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_hi=0.70)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.80)
        assert d.accept is True

    def test_enforce_low_vad_rejects_mid_score(self):
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_lo=0.30)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.10)
        assert d.accept is False
        assert d.reason == "strict_score"

    def test_enforce_vad_none_uses_hard(self):
        guard = make_strict_guard(strict_vad_adaptive=True, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=None)
        assert d.accept is False

    def test_shadow_decides_with_hard_despite_high_vad(self):
        guard = make_strict_guard(strict_vad_adaptive=False, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_hi=0.70)
        d = guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.80)
        assert d.accept is False
        assert d.reason == "strict_score"

    def test_wake_vad_ignored_in_normal(self):
        guard = make_guard(strict_vad_adaptive=True)
        d = guard.on_wake("escritorio", score=0.41, rms=0.05, wake_vad=0.10)
        assert d.accept is True

    def test_shadow_emits_log(self, caplog):
        import logging
        guard = make_strict_guard(strict_vad_adaptive=False, strict_wake_score=0.72,
                                  strict_wake_score_min=0.50, strict_vad_hi=0.70)
        with caplog.at_level(logging.INFO, logger="src.pipeline.ambient_guard"):
            guard.on_wake("escritorio", score=0.60, rms=0.05, wake_vad=0.80)
        assert any("AmbientGuard-vadshadow" in r.message for r in caplog.records)


class TestEffectiveThreshold:
    def test_vad_none_returns_hard(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(None) == 0.72

    def test_vad_at_or_below_lo_returns_hard(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(0.30) == 0.72
        assert guard._effective_strict_threshold(0.10) == 0.72

    def test_vad_at_or_above_hi_returns_soft(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(0.70) == 0.50
        assert guard._effective_strict_threshold(0.95) == 0.50

    def test_vad_midpoint_interpolates_linear(self):
        guard = make_guard(strict_wake_score=0.72, strict_wake_score_min=0.50,
                           strict_vad_lo=0.30, strict_vad_hi=0.70)
        assert guard._effective_strict_threshold(0.50) == pytest.approx(0.61)


class TestVadAdaptiveConfig:
    def test_new_fields_have_shadow_defaults(self):
        cfg = AmbientGuardConfig()
        assert cfg.strict_vad_adaptive is False  # shadow primero
        assert cfg.strict_wake_score_min == 0.50
        assert cfg.strict_vad_lo == 0.30
        assert cfg.strict_vad_hi == 0.70

    def test_clamps_lo_below_hi(self):
        cfg = AmbientGuardConfig(strict_vad_lo=0.8, strict_vad_hi=0.4)
        assert cfg.strict_vad_lo < cfg.strict_vad_hi

    def test_clamps_min_not_above_hard(self):
        cfg = AmbientGuardConfig(strict_wake_score=0.65, strict_wake_score_min=0.90)
        assert cfg.strict_wake_score_min <= cfg.strict_wake_score


class TestComputeWakeVad:
    def test_uses_predictor_max_over_chunk(self):
        calls = []
        def fake_predict(mono):
            calls.append(mono)
            return 0.83
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, fake_predict) == 0.83
        assert len(calls) == 1

    def test_none_predictor_returns_none(self):
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, None) is None

    def test_predictor_error_returns_none(self):
        def boom(mono):
            raise RuntimeError("silero down")
        audio = np.zeros(1280, dtype=np.float32)
        assert compute_wake_vad(audio, boom) is None  # fail-safe
