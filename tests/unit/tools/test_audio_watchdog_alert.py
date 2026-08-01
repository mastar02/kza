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


# Un vigilante no puede tratar "no pude leer el snapshot" como "todo bien":
# un JSON corrupto, con forma incorrecta, o con permisos que el usuario del
# poller no puede leer, antes caían en `except Exception: deaf = []` y
# reportaban sano para siempre. Estos tres casos prueban que ahora todos
# cuentan como anomalía (deaf no-vacío).


def test_check_once_corrupt_json_is_anomaly_not_sano(tmp_path, monkeypatch):
    health_path = tmp_path / "audio_health.json"
    health_path.write_text("{esto no es json valido")

    monkeypatch.setattr(audio_watchdog_alert, "notify_ha", lambda *a, **kw: True)

    deaf = audio_watchdog_alert.check_once(str(health_path), 120.0, "http://x", "tok")

    assert deaf


def test_check_once_wrong_shaped_json_is_anomaly_not_sano(tmp_path, monkeypatch):
    health_path = tmp_path / "audio_health.json"
    health_path.write_text("[1, 2, 3]")  # forma incorrecta: lista, no dict

    monkeypatch.setattr(audio_watchdog_alert, "notify_ha", lambda *a, **kw: True)

    deaf = audio_watchdog_alert.check_once(str(health_path), 120.0, "http://x", "tok")

    assert deaf


def test_check_once_permission_error_is_anomaly_not_sano(tmp_path, monkeypatch):
    health_path = tmp_path / "audio_health.json"
    health_path.write_text('{"wall": 1, "rooms": {}}')
    real_open = open

    def deny_only_health_path(path, *args, **kwargs):
        if str(path) == str(health_path):
            # Simula un poller corriendo bajo otro usuario: el snapshot
            # queda 0600 propiedad de kza y este proceso no puede leerlo.
            raise PermissionError("simulando otro usuario")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(audio_watchdog_alert, "open", deny_only_health_path, raising=False)
    monkeypatch.setattr(audio_watchdog_alert, "notify_ha", lambda *a, **kw: True)

    deaf = audio_watchdog_alert.check_once(str(health_path), 120.0, "http://x", "tok")

    assert deaf


def test_once_mode_exit_code_1_for_corrupt_snapshot(tmp_path, monkeypatch):
    """Repro end-to-end exacto del review: JSON corrupto en --once debe dar
    EXIT=1 (sordera/anomalía), nunca EXIT=0 (sano)."""
    health_path = tmp_path / "audio_health.json"
    health_path.write_text("{esto no es json valido")

    monkeypatch.setattr(audio_watchdog_alert, "notify_ha", lambda *a, **kw: True)
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "dummy-token")

    exit_code = audio_watchdog_alert.main(["--health-path", str(health_path), "--once"])

    assert exit_code == 1


def test_sys_path_insert_does_not_grow_unbounded(monkeypatch, tmp_path):
    """check_once() se llama muchas veces en el bucle real; sys.path no debe
    ganar una entrada por vuelta (medido en el review: 200 vueltas → 201
    entradas duplicadas).

    Se compara contra el punto de partida (`before`), no contra un tope
    absoluto: bajo la suite completa, pytest mismo puede dejar _REPO_ROOT en
    sys.path más de una vez por su propia mecánica de import/collection
    (verificado: ya aparece duplicado ANTES de que este test llame a
    check_once por primera vez). Eso no es lo que este test audita — lo que
    importa es que 50 vueltas no agreguen NINGUNA entrada nueva sobre lo que
    ya había, sin importar cuál sea ese punto de partida."""
    health_path = tmp_path / "audio_health.json"
    health_path.write_text('{"wall": 99999999999, "rooms": {}}')
    monkeypatch.setattr(audio_watchdog_alert, "notify_ha", lambda *a, **kw: True)

    before = list(audio_watchdog_alert.sys.path)
    before_count = before.count(audio_watchdog_alert._REPO_ROOT)
    for _ in range(50):
        audio_watchdog_alert.check_once(str(health_path), 120.0, "http://x", "tok")
    after = audio_watchdog_alert.sys.path

    assert after.count(audio_watchdog_alert._REPO_ROOT) == before_count
    assert len(after) == len(before)
