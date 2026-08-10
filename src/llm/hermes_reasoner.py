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


def warn_if_router_timeout_misaligned(
    hermes_timeout_s: float, router_timeout_s: float | None
) -> bool:
    """Loguea un ERROR si el timeout externo del LLMRouter es menor al del subproceso hermes.

    `HermesCliReasoner.timeout_s` (default 90s) gobierna el `proc.communicate(timeout=...)`
    síncrono DENTRO de `_run()` — cuando vence, `_kill_process_group` mata el process
    group entero, sin huérfanos. Pero cuando `complete()` corre bajo `LLMRouter`
    (`src/main.py`, branch `hermes_cli`), hay un SEGUNDO timeout por afuera: el
    `asyncio.wait_for(...)` que `LLMRouter` envuelve alrededor de la task
    `asyncio.to_thread`-wrapped, con el valor de `llm.failover.endpoints[reasoner_cloud]
    .timeout_s` en `settings.yaml` — una key independiente, editable por separado.

    Si ese timeout externo es MENOR al de `_run()`, `asyncio.wait_for` cancela la task
    ANTES de que el timeout propio del subproceso llegue a dispararse. Cancelar una task
    de `asyncio.to_thread` NO mata el thread nativo de abajo (asyncio no tiene forma de
    interrumpirlo) — así que el proceso `hermes` real sigue corriendo huérfano hasta SU
    PROPIO (más largo) timeout, Y el thread que ocupa queda retenido del pool compartido
    de `asyncio.to_thread` ese tiempo extra, potencialmente hambreando otras llamadas
    `asyncio.to_thread` de partes no relacionadas del codebase.

    Deliberadamente solo un log — NO levanta excepción ni frena el boot: es una
    advertencia de misconfiguración, no un fallo duro (el reasoner puede andar bien la
    mayoría de las corridas, que no llegan al timeout). Ver checklist item 1 en
    `config/settings.yaml` (bloque comentado `hermes_cli`) para el mismo aviso en forma
    de instrucción manual — esta función es la versión enforcida en código.

    Args:
        hermes_timeout_s: `timeout_s` configurado en el `HermesCliReasoner` (subproceso).
        router_timeout_s: `timeout_s` del entry `reasoner_cloud` en
            `llm.failover.endpoints` (el `asyncio.wait_for` externo de `LLMRouter`).
            `None` si `LLMRouter`/ese entry no está configurado — nada que chequear.

    Returns:
        `True` si logueó el aviso (config desalineada), `False` si está OK o no aplica.
    """
    if router_timeout_s is not None and router_timeout_s < hermes_timeout_s:
        logger.error(
            "Config desalineada: llm.failover.endpoints[reasoner_cloud].timeout_s "
            f"({router_timeout_s}s) < reasoner.hermes_timeout_s ({hermes_timeout_s}s). "
            "El asyncio.wait_for externo del LLMRouter puede cancelar la llamada a "
            "HermesCliReasoner ANTES de que el subproceso 'hermes' termine solo — "
            "cancelar una task de asyncio.to_thread NO mata el thread/subproceso de "
            "abajo, así que 'hermes' queda corriendo huérfano hasta SU PROPIO timeout, "
            "y el thread pool compartido de asyncio.to_thread queda hambreado ese "
            "tiempo extra para el resto del codebase. Fix: subir "
            "llm.failover.endpoints[reasoner_cloud].timeout_s a >= "
            f"{hermes_timeout_s}s en config/settings.yaml."
        )
        return True
    return False


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

    def _run(self, prompt: str) -> tuple[str, int]:
        """Corre `hermes -z` síncrono y devuelve (texto de respuesta, tokens).

        Usa Popen (no subprocess.run) porque el timeout necesita matar el
        process group ENTERO, no solo el proceso hijo directo: `hermes` puede
        forkear hijos propios (el sandboxed terminal backend, por ejemplo), y
        `Popen.kill()` no los alcanza — dejaría procesos huérfanos colgados
        del slow path indefinidamente. `start_new_session=True` pone a
        `hermes` en su propio process group para que `os.killpg` lo pueda
        matar entero. `stdin=DEVNULL` evita que un prompt interactivo
        (approval/auth) del agente se cuelgue leyendo el stdin heredado del
        proceso padre en vez de fallar rápido — y, si KZA corre interactivo
        desde una terminal (dev), que el hijo robe el teclado del operador.

        Lanza RuntimeError con el motivo en el mensaje (stderr del proceso,
        "timed out", "could not start", o exit=0 con stdout vacío/solo-whitespace
        — degradación silenciosa que este proyecto evita explícitamente, ver
        feedback_proxies_mentirosos) — src/llm/error_classifier.py clasifica
        por texto (rate-limit/timeout/auth/etc.), así que no hace falta un
        tipo de excepción especial para que el LLMRouter reaccione igual que
        con HttpReasoner. Esto incluye el propio `Popen()`: si el binario no
        existe o no es ejecutable, Python tira FileNotFoundError/OSError
        crudo, no RuntimeError — sin envolverlo acá esa excepción se escapa
        del contrato de esta clase y error_classifier la cae en PERMANENT
        (no failover-worthy) en vez de darle al LLMRouter la chance de rotar
        a fast_router_7b.

        Args:
            prompt: Texto de entrada.

        Returns:
            Tupla (texto de respuesta trimeado, tokens de la corrida — 0 si
            --usage-file no se pudo leer/parsear). Devolver los tokens acá
            (en vez de solo dejarlos en self._last_metrics) es lo que permite
            a __call__ leer el valor de SU PROPIA corrida sin una carrera si
            dos llamadas pisan la misma instancia concurrentemente.
        """
        self._last_metrics = None
        fd, usage_path = tempfile.mkstemp(suffix=".json", prefix="hermes-usage-")
        os.close(fd)
        try:
            cmd = self._build_cmd(prompt, usage_path)
            start = time.perf_counter()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
            except OSError as e:
                raise RuntimeError(f"hermes -z could not start: {e}") from e
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
            tokens = self._record_usage(usage_path, elapsed_ms)
            if not stdout.strip():
                raise RuntimeError(
                    f"hermes -z returned empty output (exit=0), stderr={stderr.strip()!r}"
                )
            return stdout.strip(), tokens
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

    def _record_usage(self, usage_path: str, elapsed_ms: float) -> int:
        """Best-effort: parsea --usage-file para métricas. Nunca rompe la respuesta.

        El schema exacto del JSON no está confirmado contra el binario real
        (solo documentado como "cost, tokens, model, provider, session_id,
        completed/failed") — cualquier forma inesperada (incluyendo un
        "tokens" que no es dict/int/str-numérico) degrada a no-op con un
        debug log, no una excepción. Verificar contra un `hermes -z` real en
        el server y ajustar las claves leídas acá si hace falta (ver Task 8).

        `encoding="utf-8", errors="replace"` en el open() — mismo motivo que
        los dos subprocess calls de este módulo (systemd --user con LANG sin
        setear no garantiza UTF-8 para lo que --usage-file escribió). Sin
        esto, un UnicodeDecodeError no cae en el except tuple (no es
        json.JSONDecodeError) y se escapa de _run(), tirando abajo una
        respuesta por lo demás exitosa — por eso ValueError (superclase de
        UnicodeDecodeError, y también lo que tira int() sobre un string no
        numérico) está en el tuple.

        Dos try/except separados a propósito: el primero (leer+parsear el
        archivo) es la degradación silenciosa "by design" documentada arriba
        — logger.debug. El segundo (self._metrics_tracker.record(...)) es un
        problema DISTINTO y más accionable (el tracker en sí está roto) — si
        compartiera el except del primero, un fallo del tracker se loguearía
        como si el usage-file no se hubiese podido parsear, lo cual es
        engañoso para debug. logger.warning, no silencioso.

        Returns:
            Tokens de esta corrida (int, 0 si no se pudo determinar).
        """
        tokens = 0
        try:
            with open(usage_path, encoding="utf-8", errors="replace") as f:
                usage = json.load(f)
            raw_tokens = (
                usage.get("tokens", {}).get("total")
                if isinstance(usage.get("tokens"), dict)
                else usage.get("total_tokens") or usage.get("tokens") or 0
            )
            tokens = int(raw_tokens or 0)
            self._last_metrics = {"tokens": tokens, "ms": elapsed_ms}
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as e:
            logger.debug(f"No se pudo parsear --usage-file de hermes -z: {e}")
            return 0
        try:
            if self._metrics_tracker is not None and self._endpoint_id and tokens:
                self._metrics_tracker.record(self._endpoint_id, tokens, elapsed_ms)
        except Exception as e:
            logger.warning(f"metrics tracker .record() falló (usage-file sí se parseó OK): {e}")
        return tokens

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

        Lee los tokens del valor LOCAL que devuelve _run(), no de
        self._last_metrics — self._last_metrics es un atributo de instancia
        compartido; si dos llamadas pisan la misma instancia de
        HermesCliReasoner concurrentemente (main.py la registra en el dict
        de clients de LLMRouter), releer el atributo acá podría devolver los
        tokens de la OTRA corrida. HttpReasoner.__call__ evita esto mismo
        armando su dict `usage` desde una variable local (`resp.usage`).
        """
        text, tokens = self._run(prompt)
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

        NOTA: este método es completamente síncrono — el call site real
        (`MultiUserOrchestrator._process_llm_request`, `src/orchestrator/
        dispatcher.py`: `for chunk in self.llm.generate_stream(prompt):`,
        sin `await`) bloquea el event loop ENTERO durante toda la duración
        del subproceso `hermes -z` (hasta `timeout_s`, default 90s).
        `HttpReasoner.generate_stream()` — el sibling que esta clase
        reemplaza en drop-in — documenta la misma limitación en su propio
        docstring, pero acá la ventana de bloqueo es MAYOR: HttpReasoner
        al menos cede el loop en los reads incrementales de la red
        (streaming real), mientras que acá no hay nada incremental debajo —
        es una única llamada bloqueante de punta a punta.

        Yields:
            Un único dict {"token", "text", "token_count": 1} con la
            respuesta completa.
        """
        text, _tokens = self._run(prompt)
        yield {"token": text, "text": text, "token_count": 1}

    async def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **_ignored) -> str:
        """API unificada para LLMRouter — async, retorna texto plano.

        `_run` es síncrono (subprocess.run bloqueante) — se envuelve en
        `asyncio.to_thread` para no bloquear el event loop mientras el
        subproceso corre, mismo patrón que el path sin idle-watchdog de
        `HttpReasoner.complete()`.
        """
        import asyncio
        text, _tokens = await asyncio.to_thread(self._run, prompt)
        return text

    def get_info(self) -> dict:
        """Retorna información del reasoner para logging/debugging."""
        return {"mode": "hermes_cli", "provider": self.provider, "model": self.model}

    def load_lora(self, *a, **kw):
        """Stub: HermesCliReasoner no soporta LoRA (el binario es externo)."""
        logger.warning("HermesCliReasoner no soporta LoRA (el binario es externo).")

    def unload_lora(self):
        """Stub: no-op."""
        pass
