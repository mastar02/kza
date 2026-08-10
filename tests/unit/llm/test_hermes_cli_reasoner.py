"""Tests for HermesCliReasoner — subprocess-backed reasoner (hermes -z --provider openai-codex).

No hay binario `hermes` real disponible en CI/dev — todo mockea subprocess.Popen (para _run,
que necesita el process group real para el timeout) y subprocess.run (para load(), que es un
chequeo simple sin ese requisito).
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.hermes_reasoner import HermesCliReasoner


def _completed(stdout="", stderr="", returncode=0):
    """Para tests de load(), que usa subprocess.run (chequeo simple, sin timeout de proceso)."""
    return subprocess.CompletedProcess(
        args=["hermes"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _fake_popen(stdout="", stderr="", returncode=0, pid=4242):
    """Mock de un subprocess.Popen para _run() — communicate() devuelve (stdout, stderr)."""
    proc = MagicMock()
    proc.pid = pid
    proc.returncode = returncode
    proc.communicate.return_value = (stdout, stderr)
    return proc


def _fake_popen_writing_usage_file(stdout="respuesta", usage_content=None):
    """side_effect de subprocess.Popen que escribe --usage-file antes de volver,
    igual que haría el binario real. usage_content=None simula que hermes NO
    llegó a escribir el archivo (falla antes de ese punto)."""

    def _side_effect(cmd, **kwargs):
        idx = cmd.index("--usage-file")
        path = cmd[idx + 1]
        if usage_content is not None:
            Path(path).write_text(usage_content)
        else:
            Path(path).unlink(missing_ok=True)
        return _fake_popen(stdout=stdout)

    return _side_effect


def test_init_defaults():
    r = HermesCliReasoner()
    assert r.binary_path == "hermes"
    assert r.provider == "openai-codex"
    assert r.model is None
    assert r.timeout_s == 90.0


def test_init_overrides():
    r = HermesCliReasoner(
        binary_path="/opt/hermes/bin/hermes",
        provider="openai-codex",
        model="gpt-5.1-codex",
        timeout_s=45.0,
    )
    assert r.binary_path == "/opt/hermes/bin/hermes"
    assert r.model == "gpt-5.1-codex"
    assert r.timeout_s == 45.0


@patch("src.llm.hermes_reasoner.subprocess.run")
def test_load_ok_when_provider_ready(mock_run):
    mock_run.return_value = _completed(stdout="openai-codex: ready (oauth)\n")
    r = HermesCliReasoner()
    r.load()  # no debe lanzar
    args, kwargs = mock_run.call_args
    assert args[0] == ["hermes", "auth", "status"]


@patch("src.llm.hermes_reasoner.subprocess.run")
def test_load_uses_explicit_utf8_encoding(mock_run):
    """No locale-dependent decoding — systemd --user con LANG sin setear no
    garantiza UTF-8 para la salida es-AR de `hermes auth status`."""
    mock_run.return_value = _completed(stdout="openai-codex: ready (oauth)\n")
    r = HermesCliReasoner()
    r.load()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert "text" not in kwargs


@patch("src.llm.hermes_reasoner.subprocess.run")
def test_load_raises_when_provider_not_listed(mock_run):
    mock_run.return_value = _completed(stdout="nous: ready\nxai: ready\n")
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="openai-codex"):
        r.load()


@patch("src.llm.hermes_reasoner.subprocess.run")
def test_load_raises_when_auth_command_fails(mock_run):
    mock_run.return_value = _completed(stderr="hermes: command not found", returncode=127)
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="openai-codex"):
        r.load()


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_returns_stripped_stdout(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="  la luz del living está prendida  \n")
    r = HermesCliReasoner()
    assert r._run("¿está prendida la luz?") == "la luz del living está prendida"


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_builds_correct_command_and_starts_new_session(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="ok")
    r = HermesCliReasoner(model="gpt-5.1-codex")
    r._run("hola")
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == "hermes"
    assert cmd[1] == "-z"
    assert cmd[2] == "hola"
    assert "--provider" in cmd and cmd[cmd.index("--provider") + 1] == "openai-codex"
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5.1-codex"
    assert "--usage-file" in cmd
    # encoding= (no text=True) porque solo encoding= fija explícitamente UTF-8 —
    # text=True solo decodifica con el encoding del locale, que bajo systemd
    # --user con LANG sin setear puede no ser UTF-8-safe.
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert "text" not in kwargs
    # start_new_session=True es lo que permite matar el process group entero
    # en timeout (Step 3) — sin esto, os.killpg no tiene un grupo propio que matar.
    assert kwargs["start_new_session"] is True


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_omits_model_flag_when_unset(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="ok")
    r = HermesCliReasoner()
    r._run("hola")
    cmd = mock_popen.call_args[0][0]
    assert "-m" not in cmd


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_raises_on_nonzero_exit_with_stderr_in_message(mock_popen):
    mock_popen.return_value = _fake_popen(stderr="429 rate limit exceeded", returncode=1)
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="rate limit"):
        r._run("hola")


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_raises_on_empty_stdout_even_with_exit_zero(mock_popen):
    """exit=0 con stdout vacío/whitespace es una forma de falla silenciosa —
    el patrón que este proyecto evita explícitamente en todos lados (ver
    feedback_proxies_mentirosos): un proxy (returncode) dice "éxito" mientras
    la realidad (sin texto de respuesta) dice lo contrario."""
    mock_popen.return_value = _fake_popen(stdout="   \n", stderr="", returncode=0)
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="empty"):
        r._run("hola")


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_empty_stdout_still_cleans_up_usage_file(mock_popen):
    written_path = {}

    def _side_effect(cmd, **kwargs):
        idx = cmd.index("--usage-file")
        written_path["path"] = cmd[idx + 1]
        Path(written_path["path"]).write_text(json.dumps({"tokens": {"total": 1}}))
        return _fake_popen(stdout="", stderr="", returncode=0)

    mock_popen.side_effect = _side_effect
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="empty"):
        r._run("hola")

    assert not Path(written_path["path"]).exists()


@patch("src.llm.hermes_reasoner.os.killpg")
@patch("src.llm.hermes_reasoner.os.getpgid", return_value=9999)
@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_raises_on_timeout_with_timeout_in_message(mock_popen, mock_getpgid, mock_killpg):
    proc = _fake_popen()
    proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["hermes"], timeout=90.0)
    mock_popen.return_value = proc
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="timed out"):
        r._run("hola")


@patch("src.llm.hermes_reasoner.os.killpg")
@patch("src.llm.hermes_reasoner.os.getpgid", return_value=9999)
@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_kills_process_group_on_timeout(mock_popen, mock_getpgid, mock_killpg):
    """El bug que este mecanismo previene: un hermes colgado sin fallback deja el
    slow path sin reasoner Y un proceso zombie. proc.kill() no alcanza si hermes
    forkea hijos — hace falta matar el process group completo (os.killpg)."""
    import signal
    proc = _fake_popen(pid=4242)
    proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["hermes"], timeout=90.0)
    mock_popen.return_value = proc

    r = HermesCliReasoner()
    with pytest.raises(RuntimeError):
        r._run("hola")

    mock_getpgid.assert_called_once_with(4242)
    mock_killpg.assert_called_once_with(9999, signal.SIGKILL)
    proc.wait.assert_called_once()


@patch("src.llm.hermes_reasoner.os.killpg", side_effect=ProcessLookupError)
@patch("src.llm.hermes_reasoner.os.getpgid", return_value=9999)
@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_timeout_survives_process_already_dead(mock_popen, mock_getpgid, mock_killpg):
    """Race: el proceso termina solo justo antes del killpg — no debe romper el timeout handling."""
    proc = _fake_popen()
    proc.communicate.side_effect = subprocess.TimeoutExpired(cmd=["hermes"], timeout=90.0)
    mock_popen.return_value = proc
    r = HermesCliReasoner()
    with pytest.raises(RuntimeError, match="timed out"):
        r._run("hola")


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_populates_last_metrics_from_usage_file(mock_popen):
    mock_popen.side_effect = _fake_popen_writing_usage_file(
        usage_content=json.dumps({"tokens": {"total": 123}})
    )
    r = HermesCliReasoner()
    text = r._run("hola")

    assert text == "respuesta"
    assert r._last_metrics["tokens"] == 123
    assert r._last_metrics["ms"] > 0


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_deletes_temp_usage_file_after_reading(mock_popen):
    written_path = {}

    def _side_effect(cmd, **kwargs):
        idx = cmd.index("--usage-file")
        written_path["path"] = cmd[idx + 1]
        Path(written_path["path"]).write_text(json.dumps({"tokens": {"total": 1}}))
        return _fake_popen(stdout="ok")

    mock_popen.side_effect = _side_effect
    r = HermesCliReasoner()
    r._run("hola")

    assert not Path(written_path["path"]).exists()


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_calls_metrics_tracker_when_attached(mock_popen):
    mock_popen.side_effect = _fake_popen_writing_usage_file(
        usage_content=json.dumps({"tokens": {"total": 42}})
    )
    r = HermesCliReasoner()
    tracker = MagicMock()
    r._metrics_tracker = tracker
    r._endpoint_id = "hermes_codex"
    r._run("hola")

    assert tracker.record.call_count == 1
    call_args = tracker.record.call_args[0]
    assert call_args[0] == "hermes_codex"
    assert call_args[1] == 42
    assert call_args[2] >= 0  # ms


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_survives_missing_usage_file(mock_popen):
    # hermes no llega a escribir el archivo (falla antes) — no debe romper la respuesta
    mock_popen.side_effect = _fake_popen_writing_usage_file(
        stdout="respuesta igual", usage_content=None
    )
    r = HermesCliReasoner()
    text = r._run("hola")

    assert text == "respuesta igual"
    assert r._last_metrics is None


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_survives_malformed_usage_file(mock_popen):
    mock_popen.side_effect = _fake_popen_writing_usage_file(
        stdout="respuesta igual", usage_content="esto no es json{{{"
    )
    r = HermesCliReasoner()
    text = r._run("hola")

    assert text == "respuesta igual"
    assert r._last_metrics is None


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_survives_usage_file_without_tokens_key(mock_popen):
    mock_popen.side_effect = _fake_popen_writing_usage_file(
        stdout="respuesta igual", usage_content=json.dumps({"model": "gpt-5.1-codex"})
    )
    r = HermesCliReasoner()
    text = r._run("hola")

    assert text == "respuesta igual"
    assert r._last_metrics["tokens"] == 0


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_run_resets_stale_metrics_when_later_run_has_no_usage_file(mock_popen):
    """Bug que esto previene: una corrida exitosa puebla _last_metrics con
    números reales; si una corrida POSTERIOR tiene --usage-file faltante o
    malformado (_record_usage degrada en silencio, por diseño), sin este
    reset _last_metrics seguiría reportando las métricas de la corrida
    VIEJA como si fueran de la corrida actual — __call__ las usa tal cual
    para el dict `usage`."""
    r = HermesCliReasoner()

    mock_popen.side_effect = _fake_popen_writing_usage_file(
        stdout="primera respuesta", usage_content=json.dumps({"tokens": {"total": 999}})
    )
    r._run("primera")
    assert r._last_metrics["tokens"] == 999

    mock_popen.side_effect = _fake_popen_writing_usage_file(
        stdout="segunda respuesta", usage_content=None
    )
    r._run("segunda")
    assert r._last_metrics is None


def test_has_drop_in_interface():
    # _process_llm_request hace hasattr(self.llm, 'generate_stream') / .generate
    assert hasattr(HermesCliReasoner, "generate")
    assert hasattr(HermesCliReasoner, "generate_stream")
    assert hasattr(HermesCliReasoner, "complete")


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_call_returns_choices_shape(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="la luz está prendida")
    r = HermesCliReasoner()
    result = r("¿está prendida la luz?")
    assert result["choices"][0]["text"] == "la luz está prendida"
    assert "usage" in result


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_generate_returns_text(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="hola mundo")
    r = HermesCliReasoner()
    assert r.generate("hi") == "hola mundo"


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_generate_stream_yields_single_chunk_with_full_text(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="respuesta completa")
    r = HermesCliReasoner()
    out = list(r.generate_stream("hi"))
    assert len(out) == 1
    assert out[0]["token"] == "respuesta completa"
    assert out[0]["text"] == "respuesta completa"
    assert out[0]["token_count"] == 1


def test_generate_stream_is_a_plain_sync_generator_not_async():
    # el orchestrator hace `for chunk in llm.generate_stream(prompt)` sin await
    import inspect
    assert inspect.isgeneratorfunction(HermesCliReasoner.generate_stream)
    assert not inspect.isasyncgenfunction(HermesCliReasoner.generate_stream)


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_complete_is_async_and_returns_text(mock_popen):
    mock_popen.return_value = _fake_popen(stdout="respuesta async")
    r = HermesCliReasoner()
    result = asyncio.run(r.complete("hi"))
    assert result == "respuesta async"


@patch("src.llm.hermes_reasoner.subprocess.Popen")
def test_complete_does_not_block_event_loop(mock_popen):
    # simula un proceso "lento" — complete() tiene que ceder el loop mientras
    # communicate() bloquea en el thread secundario (asyncio.to_thread).
    # La prueba mide wall-clock time: si complete() fuera bloqueante (sin
    # asyncio.to_thread), el gather tomaría ~0.05s [subprocess] + ~0.03s [ticker]
    # = 0.08s+ serializados. Con asyncio.to_thread, ambos corren concurrentemente,
    # elapsed ≈ max(0.05, 0.03) ≈ 0.05s. Asertamos elapsed < 0.075s para capturar
    # regresiones donde se remueva asyncio.to_thread accidentalmente.
    import time as time_mod

    def slow_communicate(timeout=None):
        time_mod.sleep(0.05)
        return ("ok", "")

    def make_slow_proc(cmd, **kwargs):
        proc = _fake_popen(stdout="ok")
        proc.communicate.side_effect = slow_communicate
        return proc

    mock_popen.side_effect = make_slow_proc

    async def run_concurrently():
        r = HermesCliReasoner()
        ticks = []

        async def ticker():
            for _ in range(3):
                await asyncio.sleep(0.01)
                ticks.append(1)

        results = await asyncio.gather(r.complete("hi"), ticker())
        return results, ticks

    start = time.perf_counter()
    results, ticks = asyncio.run(run_concurrently())
    elapsed = time.perf_counter() - start

    assert results[0] == "ok"
    assert len(ticks) == 3  # ticker ran while complete() waited on thread
    # Validate actual concurrency via wall-clock time: sequential execution would
    # take ~0.08s, concurrent ~0.05s. Assert we're close to concurrent (< 0.075s).
    assert elapsed < 0.075, f"expected concurrent execution (~0.05s), got {elapsed:.3f}s"


def test_get_info():
    r = HermesCliReasoner(model="gpt-5.1-codex")
    info = r.get_info()
    assert info["mode"] == "hermes_cli"
    assert info["provider"] == "openai-codex"
    assert info["model"] == "gpt-5.1-codex"


def test_lora_stubs_are_noop():
    r = HermesCliReasoner()
    r.load_lora("/some/path")  # no debe lanzar
    r.unload_lora()  # no debe lanzar


def test_hermes_cli_reasoner_importable_from_package():
    from src.llm import HermesCliReasoner as FromPackage
    assert FromPackage is HermesCliReasoner
