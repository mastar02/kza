"""
LLM Reasoner Module
Modelo grande para razonamiento profundo con soporte LoRA.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


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


class FastRouter:
    """
    Modelo rapido para clasificacion/routing (GPU).

    Optimizaciones:
    - Prefix caching: cachea KV-cache del system prompt (~40-80ms ahorro)
    - Single inference: classify_and_respond() combina clasificación + respuesta
    """

    # System prompt cacheado (prefijo común para todas las consultas)
    SYSTEM_PROMPT_PREFIX = """Eres KZA, un asistente de hogar inteligente ultra-rápido.
Tu trabajo es clasificar y responder consultas de domótica de forma eficiente.

REGLAS:
- Consultas simples (luces, temperatura, estado): responde directamente
- Consultas complejas (creatividad, conocimiento especializado): responde [DEEP]
- Sé conciso y natural en español

"""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:2",
        gpu_memory_utilization: float = 0.85,
        enable_prefix_caching: bool = True,
        enable_lora: bool = False,
        lora_path: str | None = None,
        max_lora_rank: int = 32,
    ):
        self.model_name = model
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self._lora_path = lora_path
        self._lora_active = False
        self._llm = None
        self._original_cuda_visible = None
        self._prefix_cached = False

    def _get_device_index(self) -> int:
        """Extraer indice de GPU del string device (e.g., 'cuda:2' -> 2)"""
        if ":" in self.device:
            return int(self.device.split(":")[-1])
        return 0

    def load(self):
        """
        Cargar modelo con vLLM.

        vLLM no acepta parametro 'device' directamente. En su lugar,
        usa CUDA_VISIBLE_DEVICES para seleccionar la GPU.

        Optimización: enable_prefix_caching=True para cachear KV-cache
        del system prompt y ahorrar ~40-80ms por inferencia.
        """
        import os

        self._original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

        device_idx = self._get_device_index()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

        logger.info(f"Cargando Router: {self.model_name} en GPU {device_idx}")
        logger.info(f"  Prefix caching: {self.enable_prefix_caching}")
        start = time.time()

        from vllm import LLM

        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_prefix_caching": self.enable_prefix_caching,
        }

        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
            logger.info(f"  LoRA enabled: max_rank={self.max_lora_rank}")

        self._llm = LLM(**llm_kwargs)

        if self._original_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._original_cuda_visible
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        elapsed = time.time() - start
        logger.info(f"Router cargado en {elapsed:.1f}s")

        # Load LoRA adapter if configured
        if self.enable_lora and self._lora_path:
            self.load_lora(self._lora_path)

        # Warmup: pre-cachear el system prompt
        if self.enable_prefix_caching:
            self._warmup_prefix_cache()

    def _warmup_prefix_cache(self):
        """
        Pre-cachear el system prompt para que inferencias futuras sean más rápidas.

        El primer call con el prefix lo cachea en KV-cache de vLLM.
        Calls subsecuentes con el mismo prefix reusan el cache (~40-80ms ahorro).
        """
        try:
            t_warmup = time.perf_counter()

            # Hacer una inferencia dummy con el system prompt para cachearlo
            warmup_prompt = self.SYSTEM_PROMPT_PREFIX + "Consulta: hola\nRespuesta:"

            from vllm import SamplingParams
            params = SamplingParams(max_tokens=5, temperature=0.1)
            _ = self._llm.generate([warmup_prompt], params)

            self._prefix_cached = True
            warmup_ms = (time.perf_counter() - t_warmup) * 1000
            logger.info(f"Router prefix cache warmup: {warmup_ms:.0f}ms")

        except Exception as e:
            logger.warning(f"Router prefix warmup falló (no crítico): {e}")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> list[str]:
        """Generar respuestas en batch"""
        if self._llm is None:
            self.load()

        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )

        start = time.perf_counter()

        generate_kwargs = {"prompts": prompts, "sampling_params": params}

        if self._lora_active and self._lora_path:
            from vllm.lora.request import LoRARequest
            generate_kwargs["lora_request"] = LoRARequest(
                "nightly_adapter", 1, self._lora_path
            )

        outputs = self._llm.generate(**generate_kwargs)

        elapsed_ms = (time.perf_counter() - start) * 1000
        lora_tag = " +LoRA" if self._lora_active else ""
        logger.debug(f"Router{lora_tag} ({elapsed_ms:.0f}ms, {len(prompts)} prompts)")

        return [o.outputs[0].text for o in outputs]

    def load_lora(self, lora_path: str):
        """Cargar adapter LoRA (hot-swap, no requiere reiniciar vLLM)."""
        if not Path(lora_path).exists():
            logger.warning(f"LoRA adapter not found: {lora_path}")
            return

        self._lora_path = lora_path
        self._lora_active = True
        logger.info(f"Router LoRA loaded: {lora_path}")

    def unload_lora(self):
        """Desactivar adapter LoRA."""
        self._lora_active = False
        logger.info("Router LoRA unloaded")

    def classify(self, text: str, options: list[str]) -> str:
        """Clasificar texto en una de las opciones (usa prefix cache)"""
        options_str = ", ".join(options)

        # Usar prefix cacheado
        prompt = f"""{self.SYSTEM_PROMPT_PREFIX}Clasifica en: {options_str}
Texto: {text}
Categoría:"""

        results = self.generate([prompt], max_tokens=20)
        return results[0].strip()

    def should_use_deep_reasoning(self, text: str) -> bool:
        """Decidir si requiere razonamiento profundo (usa prefix cache)"""
        prompt = f"""{self.SYSTEM_PROMPT_PREFIX}¿Requiere razonamiento complejo? Responde SIMPLE o COMPLEJO.
Pregunta: {text}
Respuesta:"""

        results = self.generate([prompt], max_tokens=10)
        return "COMPLEJO" in results[0].upper()

    def get_cache_stats(self) -> dict:
        """Obtener estadísticas del prefix cache"""
        return {
            "prefix_caching_enabled": self.enable_prefix_caching,
            "prefix_cached": self._prefix_cached,
            "system_prompt_tokens": len(self.SYSTEM_PROMPT_PREFIX.split())  # Aproximado
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

        results = self.generate([prompt], max_tokens=max_tokens, temperature=0.7)
        response = results[0].strip()

        if response.startswith("[DEEP]") or "[DEEP]" in response[:20]:
            return True, ""

        return False, response
