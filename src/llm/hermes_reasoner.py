"""
Hermes CLI Reasoner
Drop-in reasoner que invoca Hermes Agent (Nous Research) autenticado vía
OpenAI Codex OAuth (`hermes auth add openai-codex`) como subproceso, en vez
de un cliente HTTP. Ver docs/superpowers/specs/2026-08-10-hermes-cli-reasoner-design.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import Context
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from src.core.egress import EgressPolicy

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


_HERMES_PROCESS_LOCK = threading.Lock()
_HERMES_PROCESS_OWNERS: set[_HermesProcessOwner] = set()


@dataclass(eq=False)
class _HermesProcessOwner:
    """Retain one CLI process until its known retirement receipts complete."""

    usage_path: str
    proc: subprocess.Popen | None = None
    stdout: TextIO | None = None
    stderr: TextIO | None = None
    pgid: int | None = None
    phase_codes: list[str] = field(default_factory=list)
    launch_called: bool = False
    wait_confirmed: bool = False
    signal_confirmed: bool = False
    stdout_closed: bool = False
    stderr_closed: bool = False
    usage_removed: bool = False
    cleanup_started: bool = False
    cleanup_finished: bool = False
    retired: bool = False
    reasoner: _HermesLifecycle | None = None

    def __post_init__(self) -> None:
        with _HERMES_PROCESS_LOCK:
            _HERMES_PROCESS_OWNERS.add(self)

    def adopt(self, proc: subprocess.Popen) -> None:
        """Record the returned process before communicating with it."""
        self.proc = proc
        # start_new_session=True makes the child's PID its initial group ID.
        self.pgid = proc.pid
        self.stdout = proc.stdout
        self.stderr = proc.stderr

    def _failed(
        self, phase: str, error: BaseException, controls: list[BaseException]
    ) -> None:
        self.phase_codes.append(phase)
        if not isinstance(error, Exception):
            controls.append(error)

    def retire(self, *, terminate: bool) -> list[BaseException]:
        """Attempt each known receipt once, retaining any uncertain resources."""
        if self.cleanup_started:
            return []
        self.cleanup_started = True
        controls: list[BaseException] = []
        proc = self.proc
        if proc is None:
            if self.launch_called:
                self.phase_codes.append("launch_unconfirmed")
        else:
            if terminate:
                try:
                    os.killpg(self.pgid, signal.SIGKILL)
                except ProcessLookupError:
                    self.signal_confirmed = True
                except BaseException as error:
                    self._failed("group_signal_failed", error, controls)
                else:
                    self.signal_confirmed = True

            if not self.wait_confirmed:
                try:
                    if terminate and not self.signal_confirmed:
                        # A failed signal does not justify an unbounded wait.
                        proc.wait(timeout=0)
                    else:
                        proc.wait()
                except BaseException as error:
                    self._failed("process_wait_failed", error, controls)
                else:
                    self.wait_confirmed = True

            for name in ("stdout", "stderr"):
                pipe = getattr(self, name)
                if pipe is None:
                    self.phase_codes.append(f"{name}_unconfirmed")
                    continue
                try:
                    pipe.close()
                except BaseException as error:
                    self._failed(f"{name}_close_failed", error, controls)
                else:
                    setattr(self, f"{name}_closed", True)

        process_retired = (
            (proc is None and not self.launch_called)
            or (
                proc is not None
                and self.wait_confirmed
                and self.stdout_closed
                and self.stderr_closed
                and (not terminate or self.signal_confirmed)
            )
        )
        if process_retired and not self.phase_codes:
            try:
                Path(self.usage_path).unlink(missing_ok=True)
            except BaseException as error:
                self._failed("usage_unlink_failed", error, controls)
            else:
                self.usage_removed = True

        if self.usage_removed and not self.phase_codes:
            with _HERMES_PROCESS_LOCK:
                _HERMES_PROCESS_OWNERS.remove(self)
            self.retired = True
        self.cleanup_finished = True
        return controls


@dataclass(eq=False)
class _HermesCall:
    asynchronous: bool = False
    thread: threading.Thread | None = None
    on_stack: bool = False
    native_started: bool = False
    native_finished: bool = False
    job_returned: bool = False
    uncertain: bool = False
    observer_attached: bool = False
    task: asyncio.Task | None = None
    operation: object | None = None
    value: object = None
    error: BaseException | None = None
    process_owners: set[_HermesProcessOwner] = field(default_factory=set)


class _HermesLifecycle:
    """Own admitted CLI invocations without owning the borrowed thread pool."""

    def _hermes_initialize(self) -> None:
        self._hermes_condition = threading.Condition()
        self._hermes_local = threading.local()
        self._hermes_calls: set[_HermesCall] = set()
        self._hermes_process_owners: set[_HermesProcessOwner] = set()
        self._hermes_sealed = False
        self._hermes_closing: asyncio.Task | None = None
        self._hermes_close_started = False
        self._hermes_close_uncertain = False
        self._hermes_close_operation = None

    def _hermes_current(self) -> _HermesCall | None:
        thread = threading.current_thread()
        with self._hermes_condition:
            return next(
                (call for call in self._hermes_calls
                 if call.on_stack and call.thread is thread), None,
            )

    def _hermes_collect_locked(self) -> None:
        for call in tuple(self._hermes_calls):
            complete = call.native_finished and call.job_returned
            if call.asynchronous:
                complete = complete and call.task is not None and call.task.done()
            confirmed = all(owner.retired for owner in call.process_owners)
            if complete and confirmed and not call.uncertain:
                self._hermes_calls.discard(call)
                self._hermes_process_owners.difference_update(call.process_owners)

    @contextmanager
    def _hermes_enter(self, call: _HermesCall):
        previous = getattr(self._hermes_local, "call", None)
        try:
            with self._hermes_condition:
                call.thread = threading.current_thread()
                call.native_started = True
                call.on_stack = True
            self._hermes_local.call = call
            yield call
        finally:
            call.on_stack = False
            try:
                self._hermes_local.call = previous
                with self._hermes_condition:
                    self._hermes_condition.notify_all()
            except BaseException:
                call.uncertain = True
                raise

    @contextmanager
    def _hermes_admit(self):
        parent = self._hermes_current()
        if parent is not None:
            yield parent
            return
        call = _HermesCall()
        try:
            with self._hermes_condition:
                if self._hermes_sealed:
                    raise RuntimeError("hermes_closed")
                self._hermes_calls.add(call)
            with self._hermes_enter(call):
                yield call
        finally:
            call.on_stack = False
            call.native_finished = True
            call.job_returned = True
            with self._hermes_condition:
                self._hermes_collect_locked()
                self._hermes_condition.notify_all()

    def _hermes_track_process(self, owner: _HermesProcessOwner) -> None:
        call = self._hermes_current()
        if call is None:
            raise RuntimeError("hermes_admission_missing")
        with self._hermes_condition:
            self._hermes_process_owners.add(owner)
            call.process_owners.add(owner)

    def _hermes_outcome(
        self, call: _HermesCall, value=None, error: BaseException | None = None
    ) -> None:
        with self._hermes_condition:
            if call.observer_attached:
                call.value, call.error = value, error

    def _hermes_invoke(self, call: _HermesCall, function, args, kwargs) -> None:
        value = None
        try:
            with self._hermes_enter(call):
                try:
                    value = function(*args, **kwargs)
                except BaseException as error:
                    self._hermes_outcome(call, error=error)
                else:
                    self._hermes_outcome(call, value=value)
                finally:
                    value = function = None
                    args, kwargs = (), {}
        except BaseException as error:
            call.uncertain = True
            self._hermes_outcome(call, error=error)
        finally:
            value = function = None
            args, kwargs = (), {}
            call.on_stack = False
            call.native_finished = True
            with self._hermes_condition:
                self._hermes_condition.notify_all()

    async def _hermes_execute(self, call: _HermesCall, function, args, kwargs) -> None:
        try:
            await asyncio.to_thread(self._hermes_invoke, call, function, args, kwargs)
            call.job_returned = True
        except BaseException as error:
            call.uncertain = True
            self._hermes_outcome(call, error=error)
        finally:
            function = None
            args, kwargs = (), {}

    def _hermes_job_done(self, call: _HermesCall, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            call.uncertain = True
        with self._hermes_condition:
            self._hermes_collect_locked()
            self._hermes_condition.notify_all()

    async def _hermes_async(self, function, *args, **kwargs):
        call = _HermesCall(asynchronous=True, observer_attached=True)
        try:
            with self._hermes_condition:
                if self._hermes_sealed:
                    raise RuntimeError("hermes_closed")
                self._hermes_calls.add(call)
            call.operation = self._hermes_execute(call, function, args, kwargs)
            try:
                call.task = asyncio.create_task(call.operation, context=Context())
                call.operation = None
                call.task.add_done_callback(
                    lambda done: self._hermes_job_done(call, done), context=Context(),
                )
            except BaseException:
                call.uncertain = True
                raise
            await asyncio.shield(call.task)
            if call.error is not None:
                raise call.error
            if call.uncertain:
                raise RuntimeError("hermes_retirement_unconfirmed")
            return call.value
        except BaseException:
            if call.task is None:
                call.uncertain = True
            raise
        finally:
            with self._hermes_condition:
                call.observer_attached = False
                call.value = call.error = None
                self._hermes_collect_locked()
                self._hermes_condition.notify_all()
            function = None
            args, kwargs = (), {}

    def _hermes_require_close_caller(self) -> None:
        if self._hermes_current() is not None:
            raise RuntimeError("hermes_close_from_worker")
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if current is not None and current is self._hermes_closing:
            raise RuntimeError("hermes_close_from_owner")

    def request_close(self) -> None:
        """Seal new invocations without waiting or performing external IO."""
        with self._hermes_condition:
            self._hermes_sealed = True
            self._hermes_condition.notify_all()

    def _hermes_pending_locked(self) -> bool:
        for call in self._hermes_calls:
            if call.native_started and not call.native_finished:
                return True
            if call.asynchronous:
                if call.task is not None and not call.task.done():
                    return True
                if call.task is None and not call.uncertain:
                    return True
            elif not call.native_finished:
                return True
        return False

    def _hermes_failed_locked(self) -> bool:
        if self._hermes_close_uncertain or any(call.uncertain for call in self._hermes_calls):
            return True
        if any(not owner.retired for owner in self._hermes_process_owners):
            return True
        with _HERMES_PROCESS_LOCK:
            return any(owner.reasoner is self and not owner.retired
                       for owner in _HERMES_PROCESS_OWNERS)

    async def _hermes_finish(self) -> None:
        while True:
            with self._hermes_condition:
                self._hermes_collect_locked()
                pending = self._hermes_pending_locked()
                failed = self._hermes_failed_locked()
            if not pending:
                if failed:
                    raise RuntimeError("hermes_retirement_unconfirmed")
                return
            try:
                await asyncio.sleep(.01)
            except asyncio.CancelledError:
                # This owner continues observing actual native work.
                continue

    def begin_close(self) -> asyncio.Task:
        """Return the one actual retirement owner after original-caller checks."""
        self._hermes_require_close_caller()
        asyncio.get_running_loop()
        self.request_close()
        if self._hermes_closing is None:
            if self._hermes_close_started:
                raise RuntimeError("hermes_retirement_unconfirmed")
            self._hermes_close_started = True
            self._hermes_close_operation = self._hermes_finish()
            try:
                self._hermes_closing = asyncio.create_task(
                    self._hermes_close_operation, context=Context(),
                )
                self._hermes_close_operation = None
            except BaseException:
                self._hermes_close_uncertain = True
                raise
        return self._hermes_closing

    async def aclose(self) -> None:
        """Join actual retirement despite cancellation of this observer."""
        closing = self.begin_close()
        cancelled = failure = None
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError as error:
                cancelled = cancelled or error
            except Exception:
                break
        try:
            closing.result()
        except BaseException as error:
            failure = error
        if failure is not None and cancelled is not None:
            raise BaseExceptionGroup("hermes_close_observation", [failure, cancelled])
        if failure is not None:
            raise failure
        if cancelled is not None:
            raise cancelled

    def wait_closed(self) -> None:
        """Wait off-loop for admitted work; never join an active caller itself."""
        self._hermes_require_close_caller()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("hermes_close_requires_async")
        self.request_close()
        with self._hermes_condition:
            while True:
                self._hermes_collect_locked()
                if not self._hermes_pending_locked():
                    if self._hermes_failed_locked():
                        raise RuntimeError("hermes_retirement_unconfirmed")
                    return
                self._hermes_condition.wait(.05)

    def close(self) -> None:
        """Wait synchronously outside an event loop for actual retirement."""
        self.wait_closed()


class HermesCliReasoner(_HermesLifecycle):
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
        cloud_consent: bool = False,
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
        self._hermes_initialize()
        self.binary_path = binary_path
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self.policy = EgressPolicy(cloud_consent=cloud_consent)
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
        with self._hermes_admit():
            self.policy.require_allowed()
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
        with self._hermes_admit():
            self.policy.require_allowed()
            self._last_metrics = None
            fd, usage_path = tempfile.mkstemp(suffix=".json", prefix="hermes-usage-")
            os.close(fd)
            owner = _HermesProcessOwner(usage_path)
            primary: BaseException | None = None
            try:
                owner.reasoner = self
                self._hermes_track_process(owner)
                cmd = self._build_cmd(prompt, usage_path)
                start = time.perf_counter()
                owner.launch_called = True
                try:
                    owner.proc = subprocess.Popen(
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
                proc = owner.proc
                owner.adopt(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout_s)
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError(f"hermes -z timed out after {self.timeout_s}s") from e
                elapsed_ms = (time.perf_counter() - start) * 1000
                # A successful communicate returns only after its actual wait.
                owner.wait_confirmed = True
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
            except BaseException as error:
                primary = error
                raise
            finally:
                controls: list[BaseException] = []
                try:
                    controls = owner.retire(terminate=not owner.wait_confirmed)
                except BaseException as error:
                    # The strong registry remains the owner even if metadata fails.
                    owner.phase_codes.append("retirement_unconfirmed")
                    if not isinstance(error, Exception):
                        controls.append(error)
                if controls:
                    errors = ([primary] if primary is not None else []) + [
                        error for error in controls if error is not primary
                    ]
                    if len(errors) == 1:
                        raise errors[0]
                    raise BaseExceptionGroup("hermes_process_cleanup_controls", errors)
                if primary is None and not owner.retired:
                    raise RuntimeError("hermes_process_retirement_failed")

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
        with self._hermes_admit():
            text, tokens = self._run(prompt)
            return {
                "choices": [{"text": text}],
                "usage": {"prompt_tokens": 0, "completion_tokens": tokens},
            }

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generar solo el texto (drop-in de LLMReasoner.generate/HttpReasoner.generate)."""
        with self._hermes_admit():
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
        with self._hermes_admit():
            text, _tokens = self._run(prompt)
        yield {"token": text, "text": text, "token_count": 1}

    async def aload(self) -> None:
        """Load through an owned native receipt without blocking the event loop."""
        return await self._hermes_async(self.load)

    async def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, **_ignored) -> str:
        """Run one owned native invocation and return its complete text.

        Cancellation detaches this observer while actual native work remains
        retained until its real callable and process retirement finish.
        """
        text, _tokens = await self._hermes_async(self._run, prompt)
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
