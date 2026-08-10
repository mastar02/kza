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
                capture_output=True, encoding="utf-8", errors="replace", timeout=10,
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

        Lanza RuntimeError con el motivo en el mensaje (stderr del proceso,
        "timed out", o exit=0 con stdout vacío/solo-whitespace — degradación
        silenciosa que este proyecto evita explícitamente, ver
        feedback_proxies_mentirosos) — src/llm/error_classifier.py clasifica
        por texto (rate-limit/timeout/auth/etc.), así que no hace falta un
        tipo de excepción especial para que el LLMRouter reaccione igual que
        con HttpReasoner.

        Args:
            prompt: Texto de entrada.

        Returns:
            Texto de respuesta (stdout del subproceso, trimeado).
        """
        self._last_metrics = None
        fd, usage_path = tempfile.mkstemp(suffix=".json", prefix="hermes-usage-")
        os.close(fd)
        try:
            cmd = self._build_cmd(prompt, usage_path)
            start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as e:
                self._kill_process_group(proc)
                raise RuntimeError(f"hermes -z timed out after {self.timeout_s}s") from e
            elapsed_ms = (time.perf_counter() - start) * 1000
            if proc.returncode != 0:
                raise RuntimeError(
                    f"hermes -z failed (exit={proc.returncode}): {stderr.strip()}"
                )
            self._record_usage(usage_path, elapsed_ms)
            if not stdout.strip():
                raise RuntimeError(
                    f"hermes -z returned empty output (exit=0), stderr={stderr.strip()!r}"
                )
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
        """Retorna información del reasoner para logging/debugging."""
        return {"mode": "hermes_cli", "provider": self.provider, "model": self.model}

    def load_lora(self, *a, **kw):
        """Stub: HermesCliReasoner no soporta LoRA (el binario es externo)."""
        logger.warning("HermesCliReasoner no soporta LoRA (el binario es externo).")

    def unload_lora(self):
        """Stub: no-op."""
        pass
