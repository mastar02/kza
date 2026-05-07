"""
LLM Reasoner Module
Modelo grande para razonamiento profundo con soporte LoRA.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _resolve_api_key(base_url: str) -> str:
    """Pick the bearer token env var matching the endpoint's port.

    :8100 → vLLM (VLLM_API_KEY). :8200 → llama-server (LLAMA_API_KEY).
    Unknown/unparseable port → tries VLLM_API_KEY then LLAMA_API_KEY (covers
    proxied deploys without an explicit port). When no env var is set at all,
    falls back to "not-used" so local-dev against unauthenticated endpoints
    keeps working — but logs a warning so a misconfigured prod deploy (missing
    EnvironmentFile, wrong systemd User=) is visible at startup instead of
    surfacing later as an opaque 401.
    """
    port = urlparse(base_url).port
    if port == 8100:
        key = os.getenv("VLLM_API_KEY")
        if key:
            return key
    elif port == 8200:
        key = os.getenv("LLAMA_API_KEY")
        if key:
            return key
    else:
        key = os.getenv("VLLM_API_KEY") or os.getenv("LLAMA_API_KEY")
        if key:
            return key
        logger.warning(
            "API key resolution: could not determine endpoint kind from base_url=%r "
            "(port=%s) and no VLLM_API_KEY/LLAMA_API_KEY in env; using 'not-used' sentinel",
            base_url, port,
        )
        return "not-used"
    logger.warning(
        "API key resolution: %s env var not set for %s; using 'not-used' sentinel — "
        "requests will fail if endpoint enforces auth",
        "VLLM_API_KEY" if port == 8100 else "LLAMA_API_KEY",
        base_url,
    )
    return "not-used"


# Modelos recomendados para 128GB RAM
RECOMMENDED_MODELS = {
    "llama-3.3-70b-q8": {
        "filename": "Llama-3.3-70B-Instruct-Q8_0.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF",
        "ram_gb": 70,
        "context": 131072,
        "description": "Mejor opcion - casi sin perdida de calidad"
    },
    "qwen2.5-72b-q8": {
        "filename": "Qwen2.5-72B-Instruct-Q8_0.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF",
        "ram_gb": 72,
        "context": 131072,
        "description": "Excelente en espanol y razonamiento"
    },
    "llama-3.1-70b-q8": {
        "filename": "Meta-Llama-3.1-70B-Instruct-Q8_0.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "ram_gb": 70,
        "context": 131072,
        "description": "Muy bueno, contexto largo nativo"
    },
    "llama-3.1-70b-q4": {
        "filename": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "ram_gb": 45,
        "context": 131072,
        "description": "Opcion mas ligera si necesitas mas RAM libre"
    }
}


class LLMReasoner:
    """
    LLM para razonamiento profundo (CPU) con soporte LoRA.

    Optimizado para 128GB RAM con modelos Q8_0 para maxima calidad.
    """

    def __init__(
        self,
        model_path: str,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
        n_ctx: int = 32768,
        n_threads: int = 24,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        chat_format: str = "llama-3",
        rope_freq_base: float = 500000.0,
        rope_freq_scale: float = 1.0
    ):
        """
        Args:
            model_path: Ruta al modelo GGUF
            lora_path: Ruta al adapter LoRA (opcional)
            lora_scale: Escala del adapter (1.0 = full, 0.5 = medio)
            n_ctx: Tamano del contexto (32K por defecto, hasta 128K)
            n_threads: Threads de CPU (24 para Threadripper)
            n_batch: Tamano del batch para prompt processing
            n_gpu_layers: Capas en GPU (0 = solo CPU)
            chat_format: Formato de chat (llama-3, chatml, etc)
            rope_freq_base: Base para RoPE scaling (contexto largo)
            rope_freq_scale: Escala RoPE
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.rope_freq_base = rope_freq_base
        self.rope_freq_scale = rope_freq_scale

        self._model = None
        self._lora_loaded = False

    def load(self):
        """Cargar modelo llama.cpp con LoRA opcional"""
        from llama_cpp import Llama

        logger.info(f"Cargando LLM: {self.model_path}")
        logger.info(f"  n_ctx={self.n_ctx}, n_threads={self.n_threads}, n_batch={self.n_batch}")

        start = time.time()

        # Configuracion base
        model_kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "rope_freq_base": self.rope_freq_base,
            "rope_freq_scale": self.rope_freq_scale,
            "verbose": False,
            "chat_format": self.chat_format
        }

        # Agregar LoRA si existe
        if self.lora_path and Path(self.lora_path).exists():
            model_kwargs["lora_path"] = self.lora_path
            model_kwargs["lora_scale"] = self.lora_scale
            logger.info(f"  LoRA adapter: {self.lora_path} (scale={self.lora_scale})")
            self._lora_loaded = True
        elif self.lora_path:
            logger.warning(f"  LoRA no encontrado: {self.lora_path}")

        self._model = Llama(**model_kwargs)

        elapsed = time.time() - start

        # Mostrar info del modelo
        model_size_gb = Path(self.model_path).stat().st_size / (1024**3)
        logger.info(f"LLM cargado en {elapsed:.1f}s ({model_size_gb:.1f}GB)")

        if self._lora_loaded:
            logger.info(f"  LoRA activo: personalidad adaptada")

    def load_lora(self, lora_path: str, scale: float = 1.0):
        """
        Cargar o cambiar adapter LoRA en caliente.

        Args:
            lora_path: Ruta al adapter
            scale: Escala (1.0 = completo)
        """
        if self._model is None:
            self.lora_path = lora_path
            self.lora_scale = scale
            return

        # llama-cpp-python soporta cambio de LoRA en runtime
        try:
            self._model.set_lora(lora_path, scale)
            self._lora_loaded = True
            logger.info(f"LoRA cargado: {lora_path}")
        except AttributeError:
            logger.warning("Esta version de llama-cpp no soporta set_lora. Recarga el modelo.")
            self.lora_path = lora_path
            self.lora_scale = scale
            self._model = None
            self.load()

    def unload_lora(self):
        """Descargar adapter LoRA"""
        if self._model and self._lora_loaded:
            try:
                self._model.set_lora(None, 0)
                self._lora_loaded = False
                logger.info("LoRA descargado")
            except (AttributeError, TypeError):
                logger.warning("No se pudo descargar LoRA dinamicamente")

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None
    ) -> dict:
        """
        Generar respuesta.

        Args:
            prompt: Prompt de entrada
            max_tokens: Maximo de tokens a generar
            temperature: Temperatura (0 = deterministico, 0.7 = creativo)
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repeat_penalty: Penalidad por repeticion
            stop: Tokens de parada

        Returns:
            Resultado de llama.cpp
        """
        if self._model is None:
            self.load()

        start = time.perf_counter()

        result = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        tokens_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        logger.debug(
            f"LLM: {elapsed_ms:.0f}ms | "
            f"prompt={prompt_tokens} comp={completion_tokens} | "
            f"{tokens_per_sec:.1f} t/s"
        )

        return result

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generar solo el texto de respuesta"""
        result = self(prompt, max_tokens=max_tokens, temperature=temperature)
        return result["choices"][0]["text"]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None
    ):
        """
        Generar respuesta en streaming (token por token).

        Yields:
            dict con 'token' y 'text' acumulado
        """
        if self._model is None:
            self.load()

        start = time.perf_counter()
        accumulated_text = ""
        token_count = 0

        for chunk in self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            echo=False,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            accumulated_text += token
            token_count += 1

            yield {
                "token": token,
                "text": accumulated_text,
                "token_count": token_count
            }

        elapsed_ms = (time.perf_counter() - start) * 1000
        tokens_per_sec = token_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        logger.debug(
            f"LLM Stream: {elapsed_ms:.0f}ms | "
            f"{token_count} tokens | "
            f"{tokens_per_sec:.1f} t/s"
        )

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: str | None = None
    ) -> str:
        """
        Chat con formato de mensajes (usa chat template del modelo).

        Args:
            messages: Lista de {"role": "user|assistant|system", "content": "..."}
            max_tokens: Tokens maximos
            temperature: Temperatura
            system_prompt: System prompt adicional (opcional)
        """
        if self._model is None:
            self.load()

        # Agregar system prompt si se proporciona y no hay uno en messages
        if system_prompt and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": system_prompt}] + messages

        start = time.perf_counter()

        result = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        response = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", 0)

        logger.debug(f"Chat: {elapsed_ms:.0f}ms, {tokens} tokens")

        return response

    def get_info(self) -> dict:
        """Obtener informacion del modelo cargado"""
        return {
            "model_path": self.model_path,
            "lora_path": self.lora_path if self._lora_loaded else None,
            "lora_active": self._lora_loaded,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "chat_format": self.chat_format,
            "loaded": self._model is not None
        }

    @staticmethod
    def list_recommended_models() -> dict:
        """Listar modelos recomendados para 128GB RAM"""
        return RECOMMENDED_MODELS


class HttpReasoner:
    """
    Cliente HTTP al 72B (llama-cpp-python server bajo kza-72b.service, :8200).
    Drop-in de LLMReasoner con la misma signature de __call__ — retorna dict
    tipo `{"choices": [{"text": ...}], "usage": {...}}` para backward compat.

    El service es "siempre caliente" (Q1=A). El consumo va por HTTP para
    evitar doble-carga del modelo en el proceso de KZA.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8200/v1",
        model: str | None = None,           # si None, usa el primero que liste el server
        timeout: float = 120.0,
        idle_timeout_s: float | None = None,  # watchdog para streams (72B en CPU se cuelga)
        **_ignored_legacy,
    ):
        # Nota: el fallback simple base_url→fallback_base_url (commit 074160b)
        # fue removido en favor de LLMRouter (plan #1 OpenClaw 2026-04-28).
        # LLMRouter da chain por request, cooldown exponencial y idle watchdog;
        # supera al load-time fallback. Args legacy (fallback_*) se ignoran.
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.idle_timeout_s = idle_timeout_s
        self._client = None
        self._resolved_model = None
        self._resolved_base_url = None
        self._last_metrics: dict | None = None
        self._metrics_tracker = None
        self._endpoint_id: str | None = None

    def _try_connect(self, base_url: str, preferred_model: str | None) -> tuple[object, str]:
        """Conectar a un endpoint OpenAI-compat y resolver el model id.

        Returns (client, resolved_model_id). Raises if endpoint doesn't respond.
        """
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=_resolve_api_key(base_url), timeout=self.timeout)
        models = client.models.list()
        ids = [m.id for m in models.data]
        if preferred_model and preferred_model in ids:
            return client, preferred_model
        if ids:
            if preferred_model:
                logger.warning(
                    f"Modelo '{preferred_model}' no está en {base_url}; uso '{ids[0]}'"
                )
            return client, ids[0]
        raise RuntimeError(f"Endpoint {base_url} no lista ningún modelo")

    def load(self):
        try:
            self._client, self._resolved_model = self._try_connect(self.base_url, self.model)
            self._resolved_base_url = self.base_url
            logger.info(
                f"HttpReasoner OK → {self.base_url} (modelo: {self._resolved_model})"
            )
        except Exception as e:
            logger.error(f"HttpReasoner no pudo contactar {self.base_url}: {e}")
            raise

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
        """Completions-style: devuelve {choices: [{text: ...}], usage: {...}}."""
        if self._client is None:
            self.load()
        start = time.perf_counter()
        resp = self._client.completions.create(
            model=self._resolved_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"HttpReasoner ({elapsed_ms:.0f}ms) {len(prompt)}chars → {len(resp.choices[0].text)}chars")
        return {
            "choices": [{"text": resp.choices[0].text}],
            "usage": {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            },
        }

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **_ignored,
    ) -> str:
        """API unificada para LLMRouter — async, retorna texto plano.

        Si `idle_timeout_s` está configurado, usa stream con watchdog: aborta
        si el 72B no emite chunks por N segundos. Sin watchdog, hace request
        no-streaming convencional.
        """
        import asyncio

        if self._client is None:
            self.load()

        if not self.idle_timeout_s:
            def _call():
                t0 = time.perf_counter()
                resp = self._client.completions.create(
                    model=self._resolved_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    tokens = getattr(usage, "completion_tokens", 0) or 0
                    self._last_metrics = {"tokens": tokens, "ms": elapsed_ms}
                    if self._metrics_tracker is not None and self._endpoint_id and tokens > 0:
                        self._metrics_tracker.record(self._endpoint_id, tokens, elapsed_ms)
                return resp.choices[0].text
            return await asyncio.to_thread(_call)

        # Path con stream + idle_watchdog
        from src.llm.idle_watchdog import idle_watchdog

        def _open_stream():
            return self._client.completions.create(
                model=self._resolved_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        sync_stream = await asyncio.to_thread(_open_stream)

        async def _async_iter():
            it = iter(sync_stream)
            while True:
                try:
                    chunk = await asyncio.to_thread(next, it)
                except StopIteration:
                    return
                yield chunk

        text_parts: list[str] = []
        async for chunk in idle_watchdog(_async_iter(), self.idle_timeout_s):
            try:
                delta = chunk.choices[0].text or ""
            except (AttributeError, IndexError):
                delta = ""
            text_parts.append(delta)
        return "".join(text_parts)

    # Métodos no aplicables (LoRA, GGUF load) — stubs para backward compat
    def load_lora(self, *a, **kw):
        logger.warning("HttpReasoner no carga LoRA (el service es externo, coordinación manual).")

    def unload_lora(self):
        pass

    def get_info(self) -> dict:
        return {
            "mode": "http",
            "base_url": self._resolved_base_url or self.base_url,
            "model": self._resolved_model,
        }


class FastRouter:
    """
    Router rápido — cliente HTTP al vLLM compartido (:8100, usuario infra).

    Reemplaza la carga local de Qwen 7B por consumo HTTP del servicio compartido.
    Interfaz idéntica a la versión local (`FastRouterLocal`) para drop-in.
    Ver Notion KZA página 8 §4 — catálogo de modelos compartidos.
    """

    SYSTEM_PROMPT_PREFIX = """Eres KZA, un asistente de hogar inteligente ultra-rápido.
Tu trabajo es clasificar y responder consultas de domótica de forma eficiente.

REGLAS:
- Consultas simples (luces, temperatura, estado): responde directamente
- Consultas complejas (creatividad, conocimiento especializado): responde [DEEP]
- Sé conciso y natural en español

"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100/v1",
        model: str = "qwen2.5-7b-awq",
        timeout: float = 30.0,
        # Backward-compat: ignorar kwargs de la versión local (device, gpu_*, lora_*).
        **_ignored,
    ):
        self.base_url = base_url
        self.model_name = model
        self.timeout = timeout
        self._client = None
        self._available = False
        self._last_metrics: dict | None = None
        # Para metrics tracker (pegados desde main.py post-construcción).
        self._metrics_tracker = None
        self._endpoint_id: str | None = None
        if _ignored:
            logger.debug(f"FastRouter (HTTP): ignorando kwargs legacy {list(_ignored.keys())}")

    def load(self):
        """Inicializar cliente OpenAI + verificar que el modelo existe en el catálogo."""
        from openai import OpenAI

        logger.info(f"Conectando al vLLM compartido en {self.base_url} (modelo: {self.model_name})")
        self._client = OpenAI(base_url=self.base_url, api_key=_resolve_api_key(self.base_url), timeout=self.timeout)

        try:
            models = self._client.models.list()
            ids = [m.id for m in models.data]
            if self.model_name not in ids:
                logger.warning(
                    f"Modelo '{self.model_name}' no está en el catálogo vLLM. "
                    f"Disponibles: {ids}. Ver Notion KZA pág 8 §4.4 para pedir uno nuevo."
                )
            else:
                self._available = True
                logger.info(f"Router HTTP listo — modelo '{self.model_name}' disponible")
        except Exception as e:
            logger.error(f"No se pudo contactar vLLM compartido: {e}")
            self._available = False

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        **_ignored,
    ) -> str:
        """API unificada para LLMRouter — async, single prompt, retorna texto.

        A diferencia de `generate`, NO swallowea excepciones — las propaga para
        que el LLMRouter pueda clasificarlas (rate-limit, timeout, etc.) y
        rotar al siguiente endpoint.

        Side effect: setea `self._last_metrics = {"tokens": int, "ms": float}`
        tras cada call exitosa para que el LLMRouter pueda samplearlo y forwardar
        a un MetricsTracker. No-op si el endpoint no devuelve `usage`.
        """
        import asyncio

        def _call():
            if self._client is None:
                self.load()
            t0 = time.perf_counter()
            resp = self._client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            usage = getattr(resp, "usage", None)
            if usage is not None:
                tokens = getattr(usage, "completion_tokens", 0) or 0
                self._last_metrics = {"tokens": tokens, "ms": elapsed_ms}
                if self._metrics_tracker is not None and self._endpoint_id and tokens > 0:
                    self._metrics_tracker.record(self._endpoint_id, tokens, elapsed_ms)
            return resp.choices[0].text

        return await asyncio.to_thread(_call)

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.3,
        extra_body: dict | None = None,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generar respuestas en batch (secuencial por limitación de API OpenAI).

        extra_body: dict opcional para parámetros vLLM-specific (ej. guided_json,
        guided_choice, guided_regex). Se reenvía tal cual al endpoint OpenAI-compat.

        stop: secuencias que cortan la generación. Sin esto, modelos chat-tuned
        invocados con /completions tienden a continuar inventando vueltas del
        prompt ("Texto: ...\nCategoría: ..."), lo que ensucia el output. Medido
        +18 puntos accuracy en classify (benchmarks/router/REPORT.md).
        """
        if self._client is None:
            self.load()

        start = time.perf_counter()
        results: list[str] = []
        kwargs = {"extra_body": extra_body} if extra_body else {}
        if stop:
            kwargs["stop"] = stop
        for prompt in prompts:
            try:
                resp = self._client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                results.append(resp.choices[0].text)
            except Exception as e:
                logger.error(f"Router HTTP error: {e}")
                results.append("")
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Router HTTP ({elapsed_ms:.0f}ms, {len(prompts)} prompts)")
        return results

    def load_lora(self, lora_path: str):
        """LoRA no soportado vía HTTP en vLLM compartido. Requiere coordinación con infra (R3)."""
        logger.warning(
            "FastRouter HTTP no carga LoRA directamente. El vLLM compartido es de infra; "
            "para activar un adapter KZA, abrir pedido según Notion pág 8 §4.4."
        )

    def unload_lora(self):
        """No-op en modo HTTP."""
        pass

    # Patrones que un modelo chat-tuned tiende a regenerar tras cumplir la
    # tarea. Sin estos stops el output se contamina con vueltas inventadas
    # del prompt. Comparten estructura, los listo una sola vez.
    _ROUTER_STOP = ["\n\n", "Texto:", "Pregunta:", "Consulta:", "Categoría:"]

    def classify(self, text: str, options: list[str]) -> str:
        """Clasificar texto en una de las opciones (usa prefix cache)"""
        options_str = ", ".join(options)

        # Usar prefix cacheado
        prompt = f"""{self.SYSTEM_PROMPT_PREFIX}Clasifica en: {options_str}
Texto: {text}
Categoría:"""

        results = self.generate([prompt], max_tokens=20, stop=self._ROUTER_STOP)
        return results[0].strip()

    def should_use_deep_reasoning(self, text: str) -> bool:
        """Decidir si requiere razonamiento profundo (usa prefix cache)"""
        prompt = f"""{self.SYSTEM_PROMPT_PREFIX}¿Requiere razonamiento complejo? Responde SIMPLE o COMPLEJO.
Pregunta: {text}
Respuesta:"""

        results = self.generate([prompt], max_tokens=10, stop=self._ROUTER_STOP)
        return "COMPLEJO" in results[0].upper()

    def get_cache_stats(self) -> dict:
        """Stats del cliente. Prefix caching lo maneja vLLM server-side (infra)."""
        return {
            "mode": "http",
            "base_url": self.base_url,
            "model": self.model_name,
            "available": self._available,
            "system_prompt_tokens": len(self.SYSTEM_PROMPT_PREFIX.split()),
        }

    def classify_and_respond(
        self,
        text: str,
        context: str = "",
        max_tokens: int = 256
    ) -> tuple[bool, str]:
        """
        Clasificar y responder en UNA SOLA inferencia con prefix caching.

        Optimizaciones:
        - Single inference: ahorra ~20ms vs clasificar + responder por separado
        - Prefix caching: el system prompt está cacheado en KV-cache (~40-80ms ahorro)

        Args:
            text: Texto del usuario
            context: Contexto adicional (historial, etc.)
            max_tokens: Máximo de tokens para la respuesta

        Returns:
            (needs_deep_reasoning, response)
            - Si needs_deep_reasoning=True, response está vacío (usar LLM 70B)
            - Si needs_deep_reasoning=False, response contiene la respuesta del router
        """
        # Usar el SYSTEM_PROMPT_PREFIX cacheado como prefijo
        # vLLM detecta el prefijo común y reutiliza su KV-cache
        context_line = f"Contexto: {context}\n" if context else ""
        prompt = f"""{self.SYSTEM_PROMPT_PREFIX}Consulta: {text}
{context_line}
Si requiere razonamiento complejo responde [DEEP], si no responde directamente:"""

        # Acá NO usamos los stops de classify/reasoning porque la respuesta
        # libre puede contener "Pregunta:" o cambios de línea legítimos.
        # Solo cortamos en patrones que indican que el modelo regenera prompt.
        results = self.generate(
            [prompt],
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["\n\nConsulta:", "\n\nTexto:"],
        )
        response = results[0].strip()

        if response.startswith("[DEEP]") or "[DEEP]" in response[:20]:
            return True, ""

        return False, response
