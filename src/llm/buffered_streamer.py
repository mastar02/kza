"""
Buffered LLM Streamer
Sistema de buffering inteligente entre LLM y TTS para conversacion fluida.

Estrategias:
1. Filler inicial: Respuesta instantanea mientras el LLM piensa
2. Pre-buffer: Acumular N tokens antes de empezar TTS
3. Sentence buffer: Enviar oraciones completas al TTS
4. Streaming continuo: TTS habla mientras LLM genera
"""

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Callable, Generator, Optional

logger = logging.getLogger(__name__)


class BufferStrategy(Enum):
    """Estrategias de buffering disponibles"""
    SENTENCE = "sentence"      # Buffer por oraciones completas
    TOKEN_COUNT = "token"      # Buffer por cantidad de tokens
    TIME_BASED = "time"        # Buffer por tiempo transcurrido
    HYBRID = "hybrid"          # Combinacion inteligente


@dataclass
class BufferConfig:
    """Configuracion del buffer LLM -> TTS"""
    # Filler phrases
    use_filler: bool = True
    filler_phrases: list[str] = field(default_factory=lambda: [
        "Dejame ver...",
        "Un momento...",
        "Mmm...",
        "Pensando...",
        "Dame un segundo...",
    ])
    filler_delay_ms: int = 100  # Delay antes de filler (evita filler en respuestas rapidas)

    # Pre-buffer
    min_tokens_before_speech: int = 15  # ~3-4 palabras antes de empezar
    min_chars_before_speech: int = 40   # Alternativa basada en caracteres

    # Sentence detection
    sentence_delimiters: str = ".!?;"   # Caracteres que terminan oracion
    comma_as_pause: bool = True         # Usar coma como punto de pausa
    min_sentence_length: int = 20       # Minimo de caracteres para considerar oracion

    # Timing
    max_buffer_time_ms: int = 3000      # Forzar flush despues de N ms sin delimitador
    inter_sentence_pause_ms: int = 150  # Pausa entre oraciones

    # Strategy
    strategy: BufferStrategy = BufferStrategy.HYBRID


@dataclass
class StreamingState:
    """Estado interno del streaming"""
    started: bool = False
    filler_spoken: bool = False
    first_speech_time: float = None
    total_tokens: int = 0
    total_chars: int = 0
    sentences_spoken: int = 0
    buffer_flushes: int = 0
    start_time: float = None


class BufferedLLMStreamer:
    """
    Streamer con buffering inteligente entre LLM y TTS.

    Permite que el LLM genere a 6-10 tok/s y el TTS hable fluidamente
    acumulando texto antes de enviarlo.
    """

    def __init__(
        self,
        tts_engine,
        config: BufferConfig = None,
        on_sentence_ready: Callable[[str], None] = None,
        on_stream_complete: Callable[[str, dict], None] = None
    ):
        """
        Args:
            tts_engine: Motor TTS con metodo speak() o speak_stream()
            config: Configuracion del buffer
            on_sentence_ready: Callback cuando una oracion esta lista
            on_stream_complete: Callback al terminar (texto_completo, stats)
        """
        self.tts = tts_engine
        self.config = config or BufferConfig()
        self.on_sentence_ready = on_sentence_ready
        self.on_stream_complete = on_stream_complete

        self._state = StreamingState()
        self._buffer = ""
        self._full_response = ""
        self._sentence_queue: Queue[str] = Queue()
        self._stop_event = threading.Event()
        self._tts_thread: Optional[threading.Thread] = None

    def _speak_filler(self):
        """Reproducir frase de relleno mientras el LLM piensa"""
        if not self.config.use_filler or self._state.filler_spoken:
            return

        # Pequeno delay para evitar filler en respuestas muy rapidas
        time.sleep(self.config.filler_delay_ms / 1000)

        # Si ya llego contenido, no usar filler
        if self._state.total_tokens > self.config.min_tokens_before_speech:
            return

        filler = random.choice(self.config.filler_phrases)
        self._state.filler_spoken = True

        logger.debug(f"[Filler] {filler}")

        # Hablar filler (bloqueante pero corto)
        if hasattr(self.tts, 'speak'):
            self.tts.speak(filler, blocking=True)

    def _is_sentence_end(self, text: str) -> bool:
        """Detectar si el texto termina en fin de oracion"""
        text = text.rstrip()
        if not text:
            return False

        # Verificar delimitadores principales
        if text[-1] in self.config.sentence_delimiters:
            return len(text) >= self.config.min_sentence_length

        # Verificar coma como pausa (si esta habilitado y hay suficiente texto)
        if self.config.comma_as_pause and text[-1] == ',':
            return len(text) >= self.config.min_sentence_length * 1.5

        return False

    def _extract_complete_sentences(self) -> list[str]:
        """Extraer oraciones completas del buffer"""
        sentences = []

        # Patron para dividir por delimitadores manteniendo el delimitador
        pattern = f'([{re.escape(self.config.sentence_delimiters)}])'
        if self.config.comma_as_pause:
            pattern = f'([{re.escape(self.config.sentence_delimiters)},])'

        parts = re.split(pattern, self._buffer)

        # Reconstruir oraciones
        current = ""
        for i, part in enumerate(parts):
            current += part
            # Si es un delimitador y hay suficiente texto
            if part in self.config.sentence_delimiters or (self.config.comma_as_pause and part == ','):
                if len(current.strip()) >= self.config.min_sentence_length:
                    sentences.append(current.strip())
                    current = ""

        # Lo que queda va de vuelta al buffer
        self._buffer = current

        return sentences

    def _should_flush_buffer(self) -> bool:
        """Decidir si forzar flush del buffer"""
        if not self._buffer.strip():
            return False

        # Flush por tiempo maximo
        if self._state.start_time:
            elapsed_ms = (time.time() - self._state.start_time) * 1000
            buffer_age = elapsed_ms - (self._state.first_speech_time or elapsed_ms)
            if buffer_age > self.config.max_buffer_time_ms:
                return True

        return False

    def _tts_worker(self):
        """Worker thread que procesa la cola de oraciones para TTS"""
        while not self._stop_event.is_set():
            try:
                sentence = self._sentence_queue.get(timeout=0.1)
                if sentence is None:  # Senal de fin
                    break

                # Callback opcional
                if self.on_sentence_ready:
                    self.on_sentence_ready(sentence)

                # Hablar la oracion
                logger.debug(f"[TTS] {sentence[:50]}...")

                if hasattr(self.tts, 'speak_stream'):
                    self.tts.speak_stream(sentence)
                elif hasattr(self.tts, 'speak'):
                    self.tts.speak(sentence, blocking=True)

                self._state.sentences_spoken += 1

                # Pausa entre oraciones
                if self.config.inter_sentence_pause_ms > 0:
                    time.sleep(self.config.inter_sentence_pause_ms / 1000)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error en TTS worker: {e}")

    def _start_tts_worker(self):
        """Iniciar thread de TTS"""
        if self._tts_thread is None or not self._tts_thread.is_alive():
            self._stop_event.clear()
            self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self._tts_thread.start()

    def _queue_sentence(self, sentence: str):
        """Agregar oracion a la cola de TTS"""
        if sentence.strip():
            self._sentence_queue.put(sentence)
            self._state.buffer_flushes += 1

    def process_stream(
        self,
        token_generator: Generator[dict, None, None],
        speak_filler: bool = None
    ) -> str:
        """
        Procesar stream de tokens del LLM y enviar a TTS con buffering.

        Args:
            token_generator: Generador que yield {'token': str, 'text': str}
            speak_filler: Override para usar filler (None = usar config)

        Returns:
            Texto completo generado
        """
        self._state = StreamingState(start_time=time.time())
        self._buffer = ""
        self._full_response = ""
        self._sentence_queue = Queue()

        use_filler = speak_filler if speak_filler is not None else self.config.use_filler

        # Iniciar worker de TTS
        self._start_tts_worker()

        # Thread para filler (no bloquea el procesamiento)
        if use_filler:
            filler_thread = threading.Thread(target=self._speak_filler, daemon=True)
            filler_thread.start()

        try:
            for chunk in token_generator:
                token = chunk.get("token", "")
                self._buffer += token
                self._full_response += token
                self._state.total_tokens += 1
                self._state.total_chars += len(token)

                # Registrar tiempo del primer token
                if self._state.first_speech_time is None and self._state.total_tokens == 1:
                    self._state.first_speech_time = time.time()

                # Verificar si hay oraciones completas para enviar
                if self.config.strategy in [BufferStrategy.SENTENCE, BufferStrategy.HYBRID]:
                    sentences = self._extract_complete_sentences()
                    for sentence in sentences:
                        self._queue_sentence(sentence)

                # Verificar flush por tiempo
                elif self._should_flush_buffer():
                    self._queue_sentence(self._buffer)
                    self._buffer = ""

            # Flush del buffer restante
            if self._buffer.strip():
                self._queue_sentence(self._buffer)
                self._buffer = ""

        finally:
            # Senal de fin para el worker
            self._sentence_queue.put(None)

            # Esperar que termine el TTS
            if self._tts_thread and self._tts_thread.is_alive():
                self._tts_thread.join(timeout=30)

            self._stop_event.set()

        # Stats finales
        stats = self._get_stats()
        logger.info(
            f"[Buffer] {stats['total_tokens']} tokens, "
            f"{stats['sentences_spoken']} oraciones, "
            f"{stats['tokens_per_second']:.1f} t/s"
        )

        if self.on_stream_complete:
            self.on_stream_complete(self._full_response, stats)

        return self._full_response

    def _get_stats(self) -> dict:
        """Obtener estadisticas del streaming"""
        elapsed = time.time() - self._state.start_time if self._state.start_time else 0
        return {
            "total_tokens": self._state.total_tokens,
            "total_chars": self._state.total_chars,
            "sentences_spoken": self._state.sentences_spoken,
            "buffer_flushes": self._state.buffer_flushes,
            "filler_used": self._state.filler_spoken,
            "elapsed_seconds": elapsed,
            "tokens_per_second": self._state.total_tokens / elapsed if elapsed > 0 else 0
        }

    def stream_and_speak(
        self,
        llm,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_filler: bool = None
    ) -> str:
        """
        Metodo conveniente: genera con LLM y habla con buffering.

        Args:
            llm: LLMReasoner con metodo generate_stream()
            prompt: Prompt para el LLM
            max_tokens: Tokens maximos
            temperature: Temperatura
            use_filler: Usar filler phrase

        Returns:
            Texto completo generado
        """
        if not hasattr(llm, 'generate_stream'):
            # Fallback a generacion normal sin streaming
            logger.warning("LLM no soporta streaming, usando generacion normal")
            response = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if hasattr(self.tts, 'speak'):
                self.tts.speak(response)
            return response

        token_gen = llm.generate_stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return self.process_stream(token_gen, speak_filler=use_filler)


class ConversationStreamer:
    """
    Streamer optimizado para conversaciones multi-turno.
    Mantiene contexto y ajusta buffering segun el tipo de respuesta.
    """

    def __init__(
        self,
        llm,
        tts,
        config: BufferConfig = None
    ):
        self.llm = llm
        self.tts = tts
        self.config = config or BufferConfig()
        self._streamer = BufferedLLMStreamer(tts, config)
        self._conversation_history: list[dict] = []

    def add_system_prompt(self, prompt: str):
        """Agregar system prompt al inicio de la conversacion"""
        self._conversation_history = [{"role": "system", "content": prompt}]

    def respond(
        self,
        user_input: str,
        use_filler: bool = True,
        max_tokens: int = 512
    ) -> str:
        """
        Responder a input del usuario con streaming + buffering.

        Args:
            user_input: Texto del usuario
            use_filler: Usar filler mientras piensa
            max_tokens: Tokens maximos

        Returns:
            Respuesta del asistente
        """
        # Agregar turno del usuario
        self._conversation_history.append({"role": "user", "content": user_input})

        # Construir prompt con historia
        prompt = self._build_prompt()

        # Generar y hablar con buffering
        response = self._streamer.stream_and_speak(
            self.llm,
            prompt,
            max_tokens=max_tokens,
            use_filler=use_filler
        )

        # Agregar respuesta al historial
        self._conversation_history.append({"role": "assistant", "content": response})

        return response

    def _build_prompt(self) -> str:
        """Construir prompt con historial de conversacion"""
        parts = []
        for msg in self._conversation_history:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"\nUsuario: {content}")
            elif role == "assistant":
                parts.append(f"\nAsistente: {content}")

        parts.append("\nAsistente:")
        return "".join(parts)

    def clear_history(self):
        """Limpiar historial manteniendo system prompt"""
        system = [m for m in self._conversation_history if m["role"] == "system"]
        self._conversation_history = system

    def get_history(self) -> list[dict]:
        """Obtener historial de conversacion"""
        return self._conversation_history.copy()


def create_buffered_streamer(
    tts,
    preset: str = "balanced"
) -> BufferedLLMStreamer:
    """
    Factory para crear streamer con configuracion preestablecida.

    Presets:
        - "fast": Minimo buffering, respuesta rapida pero puede entrecortarse
        - "balanced": Balance entre latencia y fluidez (recomendado para 32B)
        - "smooth": Mas buffering, muy fluido pero mayor latencia inicial
        - "slow_llm": Optimizado para LLMs lentos (2-4 tok/s)
    """
    presets = {
        "fast": BufferConfig(
            use_filler=False,
            min_tokens_before_speech=5,
            min_chars_before_speech=15,
            max_buffer_time_ms=1500,
            inter_sentence_pause_ms=50
        ),
        "balanced": BufferConfig(
            use_filler=True,
            min_tokens_before_speech=15,
            min_chars_before_speech=40,
            max_buffer_time_ms=3000,
            inter_sentence_pause_ms=150
        ),
        "smooth": BufferConfig(
            use_filler=True,
            min_tokens_before_speech=30,
            min_chars_before_speech=80,
            max_buffer_time_ms=5000,
            inter_sentence_pause_ms=200
        ),
        "slow_llm": BufferConfig(
            use_filler=True,
            filler_delay_ms=50,
            min_tokens_before_speech=25,
            min_chars_before_speech=60,
            max_buffer_time_ms=4000,
            inter_sentence_pause_ms=250,
            comma_as_pause=True  # Mas puntos de pausa para LLM lento
        )
    }

    config = presets.get(preset, presets["balanced"])
    return BufferedLLMStreamer(tts, config)
