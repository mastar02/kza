# Hermes CLI Reasoner (Pieza 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar el backend del reasoner cloud del slow path (hoy gateway `:8200` → MiniMax)
por Hermes Agent autenticado con la cuenta ChatGPT del usuario (`hermes auth add openai-codex`),
invocado como subproceso vía `hermes -z`, sin cambiar nada del flujo de prompt/contexto del
orchestrator.

**Architecture:** Clase nueva `HermesCliReasoner` (`src/llm/hermes_reasoner.py`), duck-typed
idéntica a `HttpReasoner`/`LLMReasoner` (mismo patrón "drop-in" del proyecto). Por dentro llama
`hermes -z "<prompt>" --provider openai-codex --usage-file <tmp>` vía `subprocess.run` (sync,
envuelto en `asyncio.to_thread` para el path async — mismo patrón que ya usa
`HttpReasoner.complete()`). Sin streaming real: un solo chunk. Gate de privacidad extendido para
un modo sin URL (`reasoner.mode == "hermes_cli"`, cloud incondicional). Compactor también usa
`HermesCliReasoner` cuando el gate lo permite.

**Tech Stack:** Python 3.13, `subprocess` (sync, sin dependencias nuevas) + `asyncio.to_thread`
para el único call site async (`complete()`) — mismo uso de `asyncio.to_thread` que ya tiene
`HttpReasoner.complete()` (`src/llm/reasoner.py`), aunque ahí envuelve una llamada HTTP, no un
subproceso. La combinación específica "subprocess sync envuelto en `asyncio.to_thread`" no tiene
precedente previo en el codebase — `src/training/nightly_trainer.py`, que sí corre subprocesos
desde código async, usa el API nativo `asyncio.create_subprocess_exec` en vez de este patrón.
pytest + `unittest.mock` (`asyncio_mode = auto` en `pytest.ini`, no hace falta
`@pytest.mark.asyncio` pero el codebase lo usa igual por claridad).

## Global Constraints

- Sin imports relativos — `from src.llm.hermes_reasoner import HermesCliReasoner`.
- `async/await` para toda operación I/O; el subproceso corre sync pero se envuelve en
  `asyncio.to_thread` en el path que consume el event loop (`complete()`).
- Docstrings Google-style en clase y métodos públicos; type hints en firmas públicas.
- `logger = logging.getLogger(__name__)`; nunca `print()`.
- No modificar `config/settings.yaml` para activar `mode: "hermes_cli"` en producción — el plan
  solo agrega el código y la config de referencia (comentada), el flip es un paso operativo manual
  posterior al smoke test (spec §10, fuera de este plan).
- Todo cambio en `src/llm/cloud_consent.py` mantiene el orden gate→fallback documentado — no
  reordenar, no reimplementar esa lógica inline en otro archivo.
- Spec de referencia: `docs/superpowers/specs/2026-08-10-hermes-cli-reasoner-design.md`.

---

### Task 1: `HermesCliReasoner` — construcción, `load()`, mecanismo `_run()` (éxito/error/timeout)

**Files:**
- Create: `src/llm/hermes_reasoner.py`
- Test: `tests/unit/llm/test_hermes_cli_reasoner.py`

**Interfaces:**
- Consumes: nada de otras tasks de este plan (primera pieza de código nueva).
- Produces: `HermesCliReasoner.__init__(self, binary_path: str = "hermes", provider: str =
  "openai-codex", model: str | None = None, timeout_s: float = 90.0)`; `load(self) -> None`
  (síncrono, lanza `RuntimeError` si `hermes auth status` no reporta el provider listo);
  `_run(self, prompt: str) -> str` (síncrono, corre el subproceso, devuelve el texto de
  respuesta o lanza `RuntimeError` con el motivo — timeout o exit≠0 — en el mensaje, para que
  `src/llm/error_classifier.py` (matching por texto: "timeout", "429", "rate limit", etc.) lo
  clasifique sin cambios).

- [ ] **Step 1: Escribir los tests que fallan**

```python
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
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.llm.hermes_reasoner'`

- [ ] **Step 3: Implementar `HermesCliReasoner` — init, load(), _run()**

```python
"""
Hermes CLI Reasoner
Drop-in reasoner que invoca Hermes Agent (Nous Research) autenticado vía
OpenAI Codex OAuth (`hermes auth add openai-codex`) como subproceso, en vez
de un cliente HTTP. Ver docs/superpowers/specs/2026-08-10-hermes-cli-reasoner-design.md.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class HermesCliReasoner:
    """Reasoner que delega en el CLI de Hermes Agent (`hermes -z`).

    Drop-in de HttpReasoner/LLMReasoner — misma interfaz (`__call__`, `generate`,
    `generate_stream`, `complete`, `get_info`) para que el orchestrator y el
    LLMRouter no distingan de qué backend viene la respuesta. Por dentro no hay
    HTTP: cada llamada arranca un subproceso `hermes -z "<prompt>" --provider
    openai-codex`, que devuelve el texto final en stdout ("nothing else on
    stdout or stderr" según la doc del CLI) y nada de streaming real.
    """

    def __init__(
        self,
        binary_path: str = "hermes",
        provider: str = "openai-codex",
        model: str | None = None,
        timeout_s: float = 90.0,
    ):
        """
        Args:
            binary_path: Ruta al binario `hermes` (override si el entorno de
                systemd --user no hereda el PATH del shell que corrió el instalador).
            provider: Provider de Hermes a usar (`--provider`). Siempre
                "openai-codex" en este proyecto — la cuota es la suscripción
                ChatGPT del usuario, autenticada por fuera con
                `hermes auth add openai-codex`.
            model: Pin opcional de modelo (`-m`). None = default de Hermes/Codex.
            timeout_s: Timeout duro del subproceso. Más alto que el HTTP
                equivalente (60s en HttpReasoner) porque suma el arranque del
                proceso `hermes`.
        """
        self.binary_path = binary_path
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self._last_metrics: dict | None = None
        self._metrics_tracker = None
        self._endpoint_id: str | None = None

    def load(self):
        """Chequeo de boot: falla ruidoso si `hermes` o el auth de Codex no están listos.

        Corre `hermes auth status` (síncrono, timeout corto — es un chequeo
        único al arranque, no el hot path) y valida que el provider configurado
        aparece en la salida. Heurística por texto, no hay flag de filtro
        documentado para `auth status` — si esto da falsos negativos/positivos
        contra el binario real, el error real de todas formas sale fuerte en el
        primer `_run()` (mismo espíritu que `_resolve_api_key` en reasoner.py:
        un deploy mal configurado se ve al boot, no como una falla opaca después).

        Usa `subprocess.run` simple, NO `Popen` + process-group kill como
        `_run()`: es un comando corto (`auth status`, no `-z`) que no arranca
        el agente completo, con su propio timeout acotado — el riesgo de
        huérfanos que justifica el mecanismo de `_run()` no aplica acá.
        """
        try:
            result = subprocess.run(
                [self.binary_path, "auth", "status"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                f"No se pudo correr '{self.binary_path} auth status': {e}"
            ) from e
        if result.returncode != 0 or self.provider not in result.stdout:
            raise RuntimeError(
                f"Hermes auth no está lista para provider={self.provider!r}. "
                f"Correr '{self.binary_path} auth add {self.provider}' en el server. "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )
        logger.info(f"HermesCliReasoner OK — provider={self.provider}")

    def _build_cmd(self, prompt: str, usage_path: str) -> list[str]:
        """Arma el argv de `hermes -z`. Sin shell=True — sin riesgo de injection."""
        cmd = [
            self.binary_path, "-z", prompt,
            "--provider", self.provider,
            "--usage-file", usage_path,
        ]
        if self.model:
            cmd += ["-m", self.model]
        return cmd

    def _run(self, prompt: str) -> str:
        """Corre `hermes -z` síncrono y devuelve el texto de respuesta.

        Usa Popen (no subprocess.run) porque el timeout necesita matar el
        process group ENTERO, no solo el proceso hijo directo: `hermes` puede
        forkear hijos propios (el sandboxed terminal backend, por ejemplo), y
        `Popen.kill()` no los alcanza — dejaría procesos huérfanos colgados
        del slow path indefinidamente. `start_new_session=True` pone a
        `hermes` en su propio process group para que `os.killpg` lo pueda
        matar entero.

        Lanza RuntimeError con el motivo en el mensaje (stderr del proceso, o
        "timed out") — src/llm/error_classifier.py clasifica por texto
        (rate-limit/timeout/auth/etc.), así que no hace falta un tipo de
        excepción especial para que el LLMRouter reaccione igual que con
        HttpReasoner.

        Args:
            prompt: Texto de entrada.

        Returns:
            Texto de respuesta (stdout del subproceso, trimeado).
        """
        fd, usage_path = tempfile.mkstemp(suffix=".json", prefix="hermes-usage-")
        os.close(fd)
        try:
            cmd = self._build_cmd(prompt, usage_path)
            start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                self._kill_process_group(proc)
                raise RuntimeError(f"hermes -z timed out after {self.timeout_s}s")
            elapsed_ms = (time.perf_counter() - start) * 1000
            if proc.returncode != 0:
                raise RuntimeError(
                    f"hermes -z failed (exit={proc.returncode}): {stderr.strip()}"
                )
            self._record_usage(usage_path, elapsed_ms)
            return stdout.strip()
        finally:
            Path(usage_path).unlink(missing_ok=True)

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """Mata el process group completo tras un timeout.

        `proc.kill()` solo mata al hijo directo — si `hermes` forkeó hijos
        propios, sobreviven como huérfanos. `ProcessLookupError` cubre la
        carrera donde el proceso ya terminó solo justo antes de este call.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def _record_usage(self, usage_path: str, elapsed_ms: float) -> None:
        """Best-effort: parsea --usage-file para métricas. Nunca rompe la respuesta.

        El schema exacto del JSON no está confirmado contra el binario real
        (solo documentado como "cost, tokens, model, provider, session_id,
        completed/failed") — cualquier forma inesperada degrada a no-op con un
        debug log, no una excepción. Verificar contra un `hermes -z` real en el
        server y ajustar las claves leídas acá si hace falta (ver Task 8).
        """
        try:
            with open(usage_path) as f:
                usage = json.load(f)
            tokens = (
                usage.get("tokens", {}).get("total")
                if isinstance(usage.get("tokens"), dict)
                else usage.get("total_tokens") or usage.get("tokens") or 0
            )
            tokens = tokens or 0
            self._last_metrics = {"tokens": tokens, "ms": elapsed_ms}
            if self._metrics_tracker is not None and self._endpoint_id and tokens:
                self._metrics_tracker.record(self._endpoint_id, tokens, elapsed_ms)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.debug(f"No se pudo parsear --usage-file de hermes -z: {e}")
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm/hermes_reasoner.py tests/unit/llm/test_hermes_cli_reasoner.py
git commit -m "feat(llm): HermesCliReasoner — construcción, load() y mecanismo _run()"
```

---

### Task 2: Métricas — parseo de `--usage-file` (casos borde)

**Files:**
- Modify: `tests/unit/llm/test_hermes_cli_reasoner.py`
- Modify: `src/llm/hermes_reasoner.py` (solo si algún test de este task revela un bug de Task 1;
  `_record_usage` ya está implementado ahí)

**Interfaces:**
- Consumes: `HermesCliReasoner._record_usage`, `HermesCliReasoner._last_metrics`,
  `HermesCliReasoner._metrics_tracker`, `HermesCliReasoner._endpoint_id` (Task 1).
- Produces: nada nuevo — este task solo blinda `_record_usage` con más casos.

- [ ] **Step 1: Escribir los tests que fallan (agregar al archivo existente)**

```python
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
```

Agregar al tope del archivo de test los imports que falten: `import json` y
`from pathlib import Path`.

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v -k "usage_file or metrics_tracker"`
Expected: probablemente ya PASAN contra la implementación de Task 1 (es defensiva desde el
arranque) — si alguno falla, ese es el bug real a corregir en `_record_usage`.

- [ ] **Step 3: Ajustar `_record_usage` si algún test falló, si no, no hay cambios de código**

(Sin placeholder: si Step 2 dio todo verde, este step es un no-op explícito — anotarlo en el
commit como "sin cambios de código, solo cobertura".)

- [ ] **Step 4: Correr la suite completa del archivo**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v`
Expected: PASS (18 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/llm/test_hermes_cli_reasoner.py
git commit -m "test(llm): cubrir casos borde de --usage-file en HermesCliReasoner"
```

---

### Task 3: Interfaz drop-in completa (`__call__`, `generate`, `generate_stream`, `complete`, `get_info`) + export

**Files:**
- Modify: `src/llm/hermes_reasoner.py`
- Modify: `src/llm/__init__.py`
- Modify: `tests/unit/llm/test_hermes_cli_reasoner.py`

**Interfaces:**
- Consumes: `HermesCliReasoner._run(self, prompt: str) -> str` (Task 1).
- Produces: `__call__(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9, top_k=40,
  repeat_penalty=1.1, stop=None) -> dict` (shape `{"choices": [{"text": str}], "usage": {...}}`,
  igual que `HttpReasoner.__call__`); `generate(self, prompt, max_tokens=1024, temperature=0.7) ->
  str`; `generate_stream(self, prompt, max_tokens=1024, temperature=0.7, **_ignored)` (generador
  SÍNCRONO — el orchestrator lo consume con `for chunk in llm.generate_stream(prompt)`, sin
  `await`, ver `src/orchestrator/dispatcher.py:2112`); `async complete(self, prompt,
  max_tokens=512, temperature=0.7, **_ignored) -> str`; `get_info(self) -> dict`; `load_lora`,
  `unload_lora` (stubs, mismo patrón que `HttpReasoner`). `src.llm.HermesCliReasoner` importable
  desde el paquete.

- [ ] **Step 1: Escribir los tests que fallan**

```python
import asyncio


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
    # communicate() bloquea en el thread secundario (asyncio.to_thread)
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

    results, ticks = asyncio.run(run_concurrently())
    assert results[0] == "ok"
    assert len(ticks) == 3  # el ticker corrió mientras complete() esperaba el thread


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
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v -k "drop_in or call_returns or generate or complete or get_info or lora or importable"`
Expected: FAIL — `AttributeError`/`ImportError` (los métodos y el export todavía no existen)

- [ ] **Step 3: Implementar la interfaz drop-in en `src/llm/hermes_reasoner.py`**

Agregar al final de la clase `HermesCliReasoner` (después de `_record_usage`):

```python
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
    ) -> dict:
        """Completions-style: {choices: [{text: ...}], usage: {...}}.

        max_tokens/temperature/top_p/top_k/repeat_penalty/stop no se pueden
        pasar a `hermes -z` (no hay flags documentados para eso) — se
        aceptan solo por compat de firma con HttpReasoner/LLMReasoner, no
        se usan. El control de longitud/temperatura queda del lado de la
        config de Hermes, no de KZA.
        """
        text = self._run(prompt)
        tokens = self._last_metrics["tokens"] if self._last_metrics else 0
        return {
            "choices": [{"text": text}],
            "usage": {"prompt_tokens": 0, "completion_tokens": tokens},
        }

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generar solo el texto (drop-in de LLMReasoner.generate/HttpReasoner.generate)."""
        return self(prompt, max_tokens=max_tokens, temperature=temperature)["choices"][0]["text"]

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, **_ignored):
        """Streaming "falso": un solo chunk con la respuesta completa.

        `hermes -z` no da streaming token-por-token ("final response text
        out, nothing else"). El orchestrator ya tolera reasoners sin
        streaming real (cae a `generate()` si `generate_stream` no está, y
        acá optamos por implementarlo igual para que el pooling de
        `_process_llm_request` — que asume UN reasoner con streaming
        opcional — no necesite un branch nuevo).

        Yields:
            Un único dict {"token", "text", "token_count": 1} con la
            respuesta completa.
        """
        text = self._run(prompt)
        yield {"token": text, "text": text, "token_count": 1}

    async def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **_ignored) -> str:
        """API unificada para LLMRouter — async, retorna texto plano.

        `_run` es síncrono (subprocess.run bloqueante) — se envuelve en
        `asyncio.to_thread` para no bloquear el event loop mientras el
        subproceso corre, mismo patrón que el path sin idle-watchdog de
        `HttpReasoner.complete()`.
        """
        import asyncio
        return await asyncio.to_thread(self._run, prompt)

    def get_info(self) -> dict:
        return {"mode": "hermes_cli", "provider": self.provider, "model": self.model}

    def load_lora(self, *a, **kw):
        logger.warning("HermesCliReasoner no soporta LoRA (el binario es externo).")

    def unload_lora(self):
        pass
```

- [ ] **Step 4: Exportar en `src/llm/__init__.py`**

En `src/llm/__init__.py`, cambiar:

```python
from src.llm.reasoner import LLMReasoner, FastRouter, HttpReasoner
```

por:

```python
from src.llm.reasoner import LLMReasoner, FastRouter, HttpReasoner
from src.llm.hermes_reasoner import HermesCliReasoner
```

y agregar `"HermesCliReasoner",` a la lista `__all__` (junto a `"HttpReasoner"`).

- [ ] **Step 5: Correr los tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_hermes_cli_reasoner.py -v`
Expected: PASS (28 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm/hermes_reasoner.py src/llm/__init__.py tests/unit/llm/test_hermes_cli_reasoner.py
git commit -m "feat(llm): interfaz drop-in completa de HermesCliReasoner + export del paquete"
```

---

### Task 4: Gate de privacidad — branch `hermes_cli` en `resolve_reasoner_gate`

**Files:**
- Modify: `src/llm/cloud_consent.py`
- Test: `tests/unit/llm/test_cloud_consent.py`

**Interfaces:**
- Consumes: `cloud_reasoner_allowed(reasoner_config: dict) -> bool` (ya existe).
- Produces: `resolve_reasoner_gate(reasoner_config, reasoner_mode, default_local_url)` con un
  branch nuevo para `reasoner_mode == "hermes_cli"` que devuelve `(gate_allowed, None)` — sin
  URL, gateado solo por `reasoner_config["cloud"]["consent"]`, fail-closed por default.

- [ ] **Step 1: Escribir los tests que fallan (agregar a `tests/unit/llm/test_cloud_consent.py`)**

```python
def test_hermes_cli_mode_blocked_without_consent():
    """hermes_cli es cloud incondicional — sin URL que evaluar, solo consent."""
    cfg = {"cloud": {"consent": False}}
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False


def test_hermes_cli_mode_allowed_with_consent():
    cfg = {"cloud": {"consent": True}}
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is True


def test_hermes_cli_mode_returns_no_url():
    """No hay http_base_url que resolver para este modo — a diferencia de mode='http'."""
    cfg = {"cloud": {"consent": True}}
    _, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert resolved is None


def test_hermes_cli_mode_ignores_leftover_http_base_url():
    """Si reasoner_config todavía tiene http_base_url (config vieja sin limpiar),
    el gate de hermes_cli no lo usa para nada — is_cloud_endpoint nunca se llama.
    """
    cfg = {
        "http_base_url": "http://127.0.0.1:8200/v1",  # no debería importar acá
        "cloud": {"consent": False},
    }
    gate_allowed, resolved = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False  # si is_cloud_endpoint se colara, esto daría True (fail-open)
    assert resolved is None


def test_hermes_cli_mode_defaults_to_blocked_when_consent_key_missing():
    """Fail-closed: sin la key cloud.consent, hermes_cli no se habilita solo."""
    cfg = {}
    gate_allowed, _ = resolve_reasoner_gate(cfg, "hermes_cli", DEFAULT_LOCAL_LLM_GATEWAY)
    assert gate_allowed is False
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_cloud_consent.py -v -k hermes_cli`
Expected: FAIL — `test_hermes_cli_mode_returns_no_url` y las demás dan `resolved` = la URL de
fallback local (el branch cae hoy en el `else` genérico que sí devuelve una URL) o el
`gate_allowed` no es el esperado, según el caso.

- [ ] **Step 3: Implementar el branch en `resolve_reasoner_gate`**

En `src/llm/cloud_consent.py`, modificar `resolve_reasoner_gate`:

```python
def resolve_reasoner_gate(
    reasoner_config: dict, reasoner_mode: str, default_local_url: str
) -> tuple[bool, str | None]:
    """Evalúa el gate de consent para CUALQUIER ``reasoner.mode``.

    [... docstring existente sin cambios ...]

    Con ``mode="hermes_cli"`` no hay cliente HTTP ni ``base_url`` — Hermes
    corre como subproceso, siempre sale de la máquina hacia la cuenta
    ChatGPT del usuario. Se trata como cloud incondicional: el gate depende
    SOLO de ``reasoner.cloud.consent``, sin pasar por ``is_cloud_endpoint``
    (que asume una URL parseable y no aplica acá). Devuelve ``None`` como
    segundo elemento de la tupla — no hay URL que un caller pueda usar.
    """
    if reasoner_mode == "http":
        return resolve_http_reasoner_base_url(reasoner_config, default_local_url)
    if reasoner_mode == "hermes_cli":
        return cloud_reasoner_allowed(reasoner_config), None
    return (
        cloud_reasoner_allowed(reasoner_config),
        reasoner_config.get("http_base_url", default_local_url),
    )
```

Actualizar la firma también en el type hint de retorno (`tuple[bool, str | None]`) y en cualquier
docstring que prometa `str` no-opcional para el segundo elemento.

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_cloud_consent.py -v`
Expected: PASS (todos, incluyendo los 26 preexistentes + 5 nuevos)

- [ ] **Step 5: Commit**

```bash
git add src/llm/cloud_consent.py tests/unit/llm/test_cloud_consent.py
git commit -m "fix(llm): gate de privacidad para reasoner.mode=hermes_cli (cloud incondicional)"
```

---

### Task 5: Wiring del reasoner principal en `main.py`

**Files:**
- Modify: `src/main.py:305-394` (bloque de construcción del reasoner)
- Modify: `tests/unit/llm/test_main_cloud_client_wiring.py`

**Interfaces:**
- Consumes: `HermesCliReasoner` (Task 3), `resolve_reasoner_gate` con branch `hermes_cli` (Task 4).
- Produces: `llm` (variable local de `main.py`) es una instancia de `HermesCliReasoner` cuando
  `reasoner_config["mode"] == "hermes_cli"` y el gate lo permite; `None` si no.

- [ ] **Step 1: Escribir el test que falla (agregar a `test_main_cloud_client_wiring.py`)**

```python
from src.llm.hermes_reasoner import HermesCliReasoner


def test_hermes_cli_reasoner_construction_from_config():
    """Mismo patrón que test_httpreasoner_cloud_construction_from_config, para el modo nuevo."""
    reasoner_cfg = {
        "mode": "hermes_cli",
        "hermes_binary_path": "/opt/hermes/bin/hermes",
        "hermes_provider": "openai-codex",
        "hermes_model": None,
        "hermes_timeout_s": 90,
    }
    r = HermesCliReasoner(
        binary_path=reasoner_cfg.get("hermes_binary_path", "hermes"),
        provider=reasoner_cfg.get("hermes_provider", "openai-codex"),
        model=reasoner_cfg.get("hermes_model"),
        timeout_s=reasoner_cfg.get("hermes_timeout_s", 90.0),
    )
    assert r.binary_path == "/opt/hermes/bin/hermes"
    assert r.provider == "openai-codex"
    assert r.model is None
    assert r.timeout_s == 90


def test_hermes_cli_reasoner_construction_uses_defaults_when_keys_missing():
    reasoner_cfg = {"mode": "hermes_cli"}
    r = HermesCliReasoner(
        binary_path=reasoner_cfg.get("hermes_binary_path", "hermes"),
        provider=reasoner_cfg.get("hermes_provider", "openai-codex"),
        model=reasoner_cfg.get("hermes_model"),
        timeout_s=reasoner_cfg.get("hermes_timeout_s", 90.0),
    )
    assert r.binary_path == "hermes"
    assert r.timeout_s == 90.0
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_main_cloud_client_wiring.py -v`
Expected: en realidad este test específico PASA ya (construye `HermesCliReasoner` directo, no
pasa por `main.py`) — confirma que Task 3 dejó la clase lista. Lo que falta es el wiring real en
`main.py`, que no tiene test unitario directo (mismo patrón que el resto del archivo: la
selección de rama en `main()` se verifica manualmente / vía smoke test, no con un test que
importe `main()` completo).

- [ ] **Step 3: Implementar el branch en `src/main.py`**

En `src/main.py`, reemplazar el bloque `if reasoner_mode == "http": ... else: ...` (líneas
325-394) por un `if/elif/else` con la rama nueva en el medio:

```python
    if reasoner_mode == "http":
        # [... bloque existente sin cambios ...]
        if not gate_allowed:
            logger.warning(
                "Reasoner cloud bloqueado por falta de consent — slow path sin reasoner. "
                "Setear reasoner.cloud.consent=true en settings.yaml para activarlo."
            )
            llm = None
        else:
            llm = HttpReasoner(
                base_url=reasoner_config.get("http_base_url", DEFAULT_LOCAL_LLM_GATEWAY),
                model=reasoner_config.get("http_model"),
                timeout=reasoner_config.get("http_timeout", 120),
                idle_timeout_s=reasoner_config.get("idle_timeout_s"),
                api_style=reasoner_config.get("api_style", "completions"),
                api_key_env=reasoner_config.get("api_key_env"),
                verify_ssl=reasoner_config.get("verify_ssl", True),
            )
            try:
                llm.load()
                info = llm.get_info()
                logger.info(f"LLM reasoner (cloud) vía HTTP → {info['base_url']} modelo={info['model']}")
            except Exception as e:
                logger.error(f"HttpReasoner cloud no contactable: {e}. llm=None — failover via LLMRouter")
                llm = None
    elif reasoner_mode == "hermes_cli":
        # Reasoner cloud vía subproceso (hermes -z --provider openai-codex),
        # sin base_url — ver resolve_reasoner_gate branch hermes_cli
        # (src/llm/cloud_consent.py) y docs/superpowers/specs/
        # 2026-08-10-hermes-cli-reasoner-design.md.
        from src.llm.hermes_reasoner import HermesCliReasoner
        if not gate_allowed:
            logger.warning(
                "Reasoner cloud (hermes_cli) bloqueado por falta de consent — slow path sin reasoner. "
                "Setear reasoner.cloud.consent=true en settings.yaml para activarlo."
            )
            llm = None
        else:
            llm = HermesCliReasoner(
                binary_path=reasoner_config.get("hermes_binary_path", "hermes"),
                provider=reasoner_config.get("hermes_provider", "openai-codex"),
                model=reasoner_config.get("hermes_model"),
                timeout_s=reasoner_config.get("hermes_timeout_s", 90.0),
            )
            try:
                llm.load()
                info = llm.get_info()
                logger.info(f"LLM reasoner (cloud) vía Hermes CLI → provider={info['provider']} modelo={info['model']}")
            except Exception as e:
                logger.error(f"HermesCliReasoner no disponible: {e}. llm=None — failover via LLMRouter")
                llm = None
    else:
        model_path = reasoner_config.get("model_path")
        if not model_path or not Path(model_path).exists():
            logger.warning(f"Modelo LLM local no encontrado: {model_path}")
            llm = None
        else:
            llm = LLMReasoner(
                model_path=model_path,
                lora_path=reasoner_config.get("lora_path"),
                lora_scale=reasoner_config.get("lora_scale", 1.0),
                n_ctx=reasoner_config.get("n_ctx", 32768),
                n_threads=reasoner_config.get("n_threads", 24),
                n_batch=reasoner_config.get("n_batch", 512),
                n_gpu_layers=reasoner_config.get("n_gpu_layers", 0),
                chat_format=reasoner_config.get("chat_format", "chatml"),
                rope_freq_base=reasoner_config.get("rope_freq_base", 1000000.0),
                rope_freq_scale=reasoner_config.get("rope_freq_scale", 1.0),
            )
            llm.load()
            logger.info("LLM 72B cargado en proceso (mode=local)")
```

- [ ] **Step 4: Correr la suite completa de `tests/unit/llm/`**

Run: `.venv/bin/python -m pytest tests/unit/llm/ -v`
Expected: PASS (todos)

- [ ] **Step 5: Verificar que `main.py` sigue siendo válido (import sintáctico)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/main.py').read())"`
Expected: sin salida (sin `SyntaxError`)

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/unit/llm/test_main_cloud_client_wiring.py
git commit -m "feat(main): wiring de reasoner.mode=hermes_cli para el reasoner principal"
```

---

### Task 6: Wiring del compactor en `main.py` (también vía Hermes)

**Files:**
- Modify: `src/main.py:1192-1242` (bloque del compactor)
- Modify: `tests/unit/llm/test_main_cloud_client_wiring.py`

**Interfaces:**
- Consumes: `HermesCliReasoner` (Task 3), `gate_allowed`/`reasoner_mode` ya calculados más arriba
  en `main.py` (Task 5).
- Produces: `compaction_reasoner` es una instancia de `HermesCliReasoner` cuando
  `reasoner_mode == "hermes_cli"` y `gate_allowed`; sigue siendo `HttpReasoner` (comportamiento
  actual, sin cambios) para `mode == "http"`; con `hermes_cli` y gate bloqueado, `compactor =
  None` (no degrada a local silenciosamente — decisión: si no hay consent, tampoco hay
  compactación vía Hermes ni tiene sentido forzar un segundo subproceso solo para eso cuando el
  reasoner principal ya está apagado por la misma razón).

- [ ] **Step 1: Escribir el test que falla**

```python
def test_hermes_cli_compaction_reasoner_construction():
    """El compactor con mode=hermes_cli usa la misma clase que el reasoner principal."""
    reasoner_cfg = {
        "mode": "hermes_cli",
        "hermes_binary_path": "hermes",
        "hermes_provider": "openai-codex",
        "hermes_model": None,
        "hermes_timeout_s": 90,
    }
    compaction_reasoner = HermesCliReasoner(
        binary_path=reasoner_cfg.get("hermes_binary_path", "hermes"),
        provider=reasoner_cfg.get("hermes_provider", "openai-codex"),
        model=reasoner_cfg.get("hermes_model"),
        timeout_s=reasoner_cfg.get("hermes_timeout_s", 90.0),
    )
    assert isinstance(compaction_reasoner, HermesCliReasoner)
    assert compaction_reasoner.provider == "openai-codex"
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `.venv/bin/python -m pytest tests/unit/llm/test_main_cloud_client_wiring.py -v`
Expected: PASA igual que en Task 5 (construcción directa, no ejercita `main.py`) — el punto de
este task es el wiring real, verificado por lectura + Step 5 abajo.

- [ ] **Step 3: Implementar el branch en el bloque del compactor de `src/main.py`**

Reemplazar el bloque `if compaction_cfg.get("enabled", False): ...` (líneas 1197-1242):

```python
    if compaction_cfg.get("enabled", False):
        from src.orchestrator import Compactor

        if reasoner_mode == "hermes_cli":
            # Sin base_url que resolver — resolve_compaction_endpoint asume HTTP
            # (ver su docstring) y no aplica acá. El compactor usa la MISMA clase
            # y config que el reasoner principal cuando el gate lo permite; si no,
            # no hay compactación (no tiene sentido forzar un segundo subproceso
            # hermes solo para esto cuando el reasoner principal ya está apagado
            # por falta de consent).
            from src.llm.hermes_reasoner import HermesCliReasoner
            if not gate_allowed:
                logger.warning(
                    "[main] Compactor (hermes_cli) deshabilitado — reasoner.cloud.consent=false."
                )
                compactor = None
            else:
                compaction_reasoner = HermesCliReasoner(
                    binary_path=reasoner_config.get("hermes_binary_path", "hermes"),
                    provider=reasoner_config.get("hermes_provider", "openai-codex"),
                    model=reasoner_config.get("hermes_model"),
                    timeout_s=compaction_cfg.get("timeout_s", 30.0),
                )
                try:
                    compaction_reasoner.load()
                    compactor = Compactor(
                        reasoner=compaction_reasoner,
                        max_summary_tokens=compaction_cfg.get("max_summary_tokens", 200),
                        timeout_s=compaction_cfg.get("timeout_s", 30.0),
                    )
                    logger.info(
                        f"[main] Compactor enabled vía Hermes CLI (threshold={compaction_threshold}, "
                        f"keep_recent={keep_recent_turns})"
                    )
                except Exception as e:
                    logger.error(
                        f"[main] Compactor disabled — could not load Hermes reasoner: {e}",
                        exc_info=True,
                    )
                    compactor = None
        else:
            from src.llm.cloud_consent import resolve_compaction_endpoint
            from src.llm.reasoner import HttpReasoner

            # [... bloque existente para mode="http"/"local", sin cambios ...]
            compaction_endpoint = resolve_compaction_endpoint(
                compaction_cfg, reasoner_config, gate_allowed
            )
            compaction_reasoner = HttpReasoner(
                base_url=compaction_endpoint.base_url,
                model=compaction_endpoint.model,
                api_style=compaction_endpoint.api_style,
                api_key_env=compaction_endpoint.api_key_env,
                timeout=compaction_cfg.get("timeout_s", 30.0),
            )
            try:
                compaction_reasoner.load()
                compactor = Compactor(
                    reasoner=compaction_reasoner,
                    max_summary_tokens=compaction_cfg.get("max_summary_tokens", 200),
                    timeout_s=compaction_cfg.get("timeout_s", 30.0),
                )
                logger.info(
                    f"[main] Compactor enabled (threshold={compaction_threshold}, "
                    f"keep_recent={keep_recent_turns})"
                )
            except Exception as e:
                logger.error(
                    f"[main] Compactor disabled — could not load reasoner: {e}",
                    exc_info=True,
                )
                compactor = None
```

Nota importante: el bloque `else` de arriba es **idéntico** al código actual (líneas 1198-1242
sin tocar) — solo se lo re-indenta un nivel adentro del nuevo `if reasoner_mode == "hermes_cli":
... else: ...`. No cambia comportamiento para `mode="http"` (test guard:
`test_compaction_inherits_cloud_endpoint_when_gate_allows` y los demás de
`test_cloud_consent.py`, que no tocan `main.py` pero sí `resolve_compaction_endpoint`, siguen
pasando sin cambios).

- [ ] **Step 4: Correr la suite completa de `tests/unit/llm/`**

Run: `.venv/bin/python -m pytest tests/unit/llm/ -v`
Expected: PASS (todos)

- [ ] **Step 5: Verificar sintaxis de `main.py`**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/main.py').read())"`
Expected: sin salida

- [ ] **Step 6: Commit**

```bash
git add src/main.py tests/unit/llm/test_main_cloud_client_wiring.py
git commit -m "feat(main): compactor también vía Hermes CLI cuando reasoner.mode=hermes_cli"
```

---

### Task 7: Config de referencia en `settings.yaml` (comentada, sin flip de `mode`)

**Files:**
- Modify: `config/settings.yaml:387-418`

**Interfaces:**
- Consumes: nada de código — solo documentación operativa de las keys que Tasks 1-6 ya soportan.
- Produces: bloque de config comentado listo para copiar/pegar y descomentar en el server después
  del smoke test — `mode` en producción sigue siendo `"http"` (MiniMax) hasta ese paso manual,
  fuera de este plan.

- [ ] **Step 1: Agregar el bloque comentado después del bloque de rollback existente**

En `config/settings.yaml`, después de la línea 418 (`#   idle_timeout_s: 8.0`) y antes de la línea
420 (`# LLM Failover...`), agregar:

```yaml
  # --- Hermes CLI (Pieza 1, 2026-08-10) — NO activar sin correr el smoke test
  # del plan primero (docs/superpowers/plans/2026-08-10-hermes-cli-reasoner.md §10-11).
  # Bootstrap manual UNA VEZ en el server antes de activar:
  #   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
  #   hermes auth add openai-codex   # device-code flow, completar desde otro browser
  # Para activar: comentar el bloque "mode: http" de arriba y descomentar esto.
  # mode: "hermes_cli"
  # hermes_binary_path: "hermes"       # ruta absoluta si systemd --user no hereda PATH
  # hermes_provider: "openai-codex"
  # hermes_model: null                  # sin pin — default de Hermes/Codex
  # hermes_timeout_s: 90                # suma el arranque del proceso hermes sobre HTTP
  # cloud:
  #   consent: true                     # misma semántica que el bloque http de arriba
```

- [ ] **Step 2: Verificar que el YAML sigue siendo válido**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('config/settings.yaml'))"`
Expected: sin salida (sin `yaml.YAMLError`)

- [ ] **Step 3: Verificar que `mode` efectivo sigue siendo "http" (no se activó por accidente)**

Run: `.venv/bin/python -c "import yaml; c = yaml.safe_load(open('config/settings.yaml')); assert c['reasoner']['mode'] == 'http', c['reasoner']['mode']; print('OK: mode sigue en http')"`
Expected: `OK: mode sigue en http`

- [ ] **Step 4: Commit**

```bash
git add config/settings.yaml
git commit -m "docs(config): bloque de referencia comentado para reasoner.mode=hermes_cli"
```

---

### Task 8: Suite completa + checklist de smoke test manual

**Files:** ninguno nuevo — verificación final.

**Interfaces:** N/A (task de verificación, no de código).

- [ ] **Step 1: Correr la suite completa del proyecto**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: todos los tests preexistentes siguen en verde + los ~36 nuevos de este plan (0 fallos,
0 nuevos xfail inesperados).

- [ ] **Step 2: Correr solo `tests/unit/llm/` con `-v` para revisión rápida del diff de cobertura**

Run: `.venv/bin/python -m pytest tests/unit/llm/ -v --tb=short`
Expected: PASS, y contar que aparecen los archivos/tests nuevos: `test_hermes_cli_reasoner.py`
(28 tests) + los agregados a `test_cloud_consent.py` (5) y `test_main_cloud_client_wiring.py` (3).

- [ ] **Step 3: Verificar que `hermes -z --help` real coincide con los flags asumidos (Task 1)**

Esto requiere el binario `hermes` instalado y autenticado — **no es parte de CI**, es un paso
manual en el server antes de activar `mode: hermes_cli` en producción (ver Task 7 Step 1).
Documentar acá qué verificar:

```bash
ssh kza "hermes -z --help 2>&1 | grep -E '\-\-provider|\-\-usage-file|\-m,|--model'"
```

Expected: confirma que `-z`, `--provider`, `--usage-file`, `-m/--model` existen tal como los usa
`HermesCliReasoner._build_cmd` (Task 1, Step 3). Si algún flag cambió de nombre, ajustar
`_build_cmd` y sus tests antes de seguir — **no activar en prod con flags sin verificar contra el
binario real.**

- [ ] **Step 4: Smoke test de calidad (manual, en el server, DESPUÉS de Step 3)**

No forma parte de este plan de implementación (es el paso de rollout de la spec §10) — queda
anotado acá como el siguiente paso fuera de este documento: comparar un set chico de consultas
es-AR de domótica contra la respuesta actual de MiniMax antes de flippear `mode` en el
`settings.yaml` del server.
