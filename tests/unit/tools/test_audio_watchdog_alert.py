"""Tests: notify_ha() no debe propagar errores de red, y el bucle principal
del poller no debe morir por una vuelta fallida — un watchdog que se cae
cuando lo que vigila no responde es peor que no tener watchdog (su silencio
se lee como "todo bien"), justo el patrón de fallo silencioso que la task
de 2026-07-31 existe para eliminar.
"""
import requests
import pytest

from tools import audio_watchdog_alert


class _StopLoop(BaseException):
    """Sentinel para cortar el `while True` de main() en el test.

    Hereda de BaseException (no de Exception) a propósito: el `except
    Exception` del bucle es justamente lo que se está probando que NO
    atrapa cualquier cosa — si heredara de Exception, quedaría atrapado
    ahí y el test no distinguiría "el bucle sobrevive" de "el bucle nunca
    terminó por otra razón".
    """


def test_notify_ha_connection_error_returns_false_without_raising(monkeypatch):
    def raise_connection_error(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("boom de prueba")

    monkeypatch.setattr(audio_watchdog_alert.requests, "post", raise_connection_error)

    ok = audio_watchdog_alert.notify_ha("http://127.0.0.1:1", "tok", "t", "m")

    assert ok is False


def test_notify_ha_timeout_returns_false_without_raising(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise requests.exceptions.Timeout("boom de prueba")

    monkeypatch.setattr(audio_watchdog_alert.requests, "post", raise_timeout)

    ok = audio_watchdog_alert.notify_ha("http://127.0.0.1:1", "tok", "t", "m")

    assert ok is False


def test_notify_ha_success_still_returns_true(monkeypatch):
    class _Resp:
        status_code = 200

    monkeypatch.setattr(audio_watchdog_alert.requests, "post", lambda *a, **kw: _Resp())

    ok = audio_watchdog_alert.notify_ha("http://127.0.0.1:1", "tok", "t", "m")

    assert ok is True


def test_loop_survives_failing_iteration_and_continues(monkeypatch):
    calls = []

    def fake_check_once(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("vuelta 1 rota a propósito")
        raise _StopLoop  # confirma que la vuelta 2 ocurrió, y corta el bucle

    monkeypatch.setattr(audio_watchdog_alert, "check_once", fake_check_once)
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "dummy-token")

    with pytest.raises(_StopLoop):
        audio_watchdog_alert.main(
            ["--health-path", "/no/existe/audio_health.json", "--interval-s", "0"]
        )

    assert len(calls) == 2


def test_once_mode_reports_not_sano_on_unexpected_error(monkeypatch):
    def fake_check_once(*_args, **_kwargs):
        raise RuntimeError("vuelta rota")

    monkeypatch.setattr(audio_watchdog_alert, "check_once", fake_check_once)
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "dummy-token")

    exit_code = audio_watchdog_alert.main(
        ["--health-path", "/no/existe/audio_health.json", "--once"]
    )

    assert exit_code == 1
