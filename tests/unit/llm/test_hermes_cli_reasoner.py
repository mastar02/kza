"""Tests for HermesCliReasoner — subprocess-backed reasoner (hermes -z --provider openai-codex).

No hay binario `hermes` real disponible en CI/dev — todo mockea subprocess.Popen (para _run,
que necesita el process group real para el timeout) y subprocess.run (para load(), que es un
chequeo simple sin ese requisito).
"""

import subprocess
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
    assert kwargs["text"] is True
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
