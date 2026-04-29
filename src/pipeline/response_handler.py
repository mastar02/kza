"""
Response Handler Module
Gestiona síntesis de voz, streaming de audio y enrutamiento a zonas.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from src.llm.buffered_streamer import (
    BufferedLLMStreamer,
    BufferConfig,
    create_buffered_streamer
)
from src.tts.response_cache import CachedAudio, ResponseCache

logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Gestiona respuestas de audio: TTS, streaming, enrutamiento a zonas.

    Responsabilidades:
    - TTS (Text-to-Speech)
    - Streaming de audio a zonas
    - LLM buffered streaming (para LLMs lentos)
    - Ajuste de respuesta según emoción detectada
    """

    def __init__(
        self,
        tts,
        zone_manager=None,
        llm=None,
        streaming_enabled: bool = True,
        streaming_buffer_ms: int = 150,
        streaming_prebuffer_ms: int = 30,
        llm_buffer_preset: str = "balanced",
        llm_use_filler: bool = True,
        llm_filler_phrases: list = None,
        response_cache: ResponseCache | None = None,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
        before_handler_warn_ms: float = 5.0,
    ):
        """
        Inicializar ResponseHandler.

        Args:
            tts: TTS service (ej: PiperTTS)
            zone_manager: ZoneManager para enrutamiento de audio
            llm: LLM reasoner para generación de respuestas
            streaming_enabled: Habilitar streaming de audio
            streaming_buffer_ms: Buffer size para streaming
            streaming_prebuffer_ms: Prebuffer para streaming
            llm_buffer_preset: Preset de buffering para LLM
            llm_use_filler: Usar frases de relleno mientras LLM genera
            llm_filler_phrases: Frases de relleno custom
            response_cache: Cache TTS pre-generado (S2). Si se provee,
                `speak()` consulta el cache antes del TTS live; hit → playback
                directo del ndarray (~5-10ms). Miss → camino normal.
            hooks: Optional HookRegistry instance (plan #3 OpenClaw). When set,
                before_ha_action / before_tts_speak hooks fire on each invocation
                and after-events emit at pipeline checkpoints. Backward-compat:
                None → no hook calls, behavior identical to baseline.
            before_handler_warn_ms: Threshold (ms) for logging slow before-handlers
                in execute_before_chain. Default 5.0ms.
        """
        self.tts = tts
        self.zone_manager = zone_manager
        self.llm = llm

        self.streaming_enabled = streaming_enabled
        self.streaming_buffer_ms = streaming_buffer_ms
        self.streaming_prebuffer_ms = streaming_prebuffer_ms

        self.llm_buffer_preset = llm_buffer_preset
        self.llm_use_filler = llm_use_filler
        self.llm_filler_phrases = llm_filler_phrases

        self._response_cache = response_cache
        self.hooks = hooks  # plan #3 OpenClaw — HookRegistry or None
        self._before_handler_warn_ms = before_handler_warn_ms
        self._llm_streamer: BufferedLLMStreamer | None = None
        self._active_zone_id = None

        # Barge-in state (S3). `is_speaking` se consulta desde el audio
        # callback de MultiRoomAudioLoop. `_current_stream` guarda el handle
        # de sounddevice (OutputStream o StreamingAudioPlayer) activo para
        # poder cortarlo en <200ms. `_playback_task` cubre cualquier task
        # asyncio que envuelva el playback.
        self._is_speaking: bool = False
        self._current_stream = None
        self._playback_task: asyncio.Task | None = None

    @property
    def is_speaking(self) -> bool:
        """True si hay un TTS activo. Consultado por barge-in checker."""
        return self._is_speaking

    async def cancel(self) -> bool:
        """
        Interrumpir el TTS actual lo más rápido posible.

        Cierra el stream de sounddevice (si hay uno expuesto en
        `_current_stream`), cancela `_playback_task` si existe, y como
        fallback llama `sd.stop()` para cortar cualquier playback inline
        (`sd.play`) que no esté expuesto como stream.

        Idempotent: llamadas múltiples no fallan; si no hay TTS activo,
        retorna False sin efectos laterales.

        Returns:
            True si había algo activo que cancelar, False en caso contrario.
        """
        if not self._is_speaking:
            return False

        logger.info("⏹  TTS cancel (barge-in)")
        # Marcamos primero para evitar race con otro cancel concurrente.
        self._is_speaking = False

        # 1. Cerrar el stream explícito (si un método TTS lo registró).
        stream = self._current_stream
        if stream is not None:
            try:
                # sounddevice OutputStream expone stop/close; el
                # StreamingAudioPlayer de PiperTTS expone solo stop().
                if hasattr(stream, "stop"):
                    stream.stop()
                if hasattr(stream, "close"):
                    stream.close()
            except Exception as e:
                logger.warning(f"cancel(): error cerrando stream: {e}")
            finally:
                self._current_stream = None

        # 2. Cancelar task de playback si hay uno registrado.
        task = self._playback_task
        if task is not None and not task.done():
            task.cancel()
        self._playback_task = None

        # 3. Fallback: sd.stop() corta cualquier sd.play() inline que no
        # esté expuesto como stream (e.g. `PiperTTS.speak`, `_playback_cached`
        # con sd.play). No rompe si sounddevice no está inicializado.
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            logger.debug(f"cancel(): sd.stop() no-op: {e}")

        return True

    def set_active_zone(self, zone_id: str):
        """Establecer zona activa para respuestas."""
        self._active_zone_id = zone_id
        if self.zone_manager:
            self.zone_manager.set_active_zone(zone_id)
        logger.info(f"Zona activa: {zone_id}")

    def speak(
        self,
        text: str,
        zone_id: str = None,
        stream: bool = None,
        emotion_adjustment: dict = None,
        room_context=None,
    ):
        """
        Sintetizar y reproducir texto.

        Args:
            text: Texto a sintetizar
            zone_id: ID de zona específica (None = usar zona activa)
            stream: Usar streaming (None = usar config, True/False = forzar)
            emotion_adjustment: Ajustes según emoción detectada
            room_context: RoomContext para resolver zona automáticamente
        """
        if not text:
            return

        # Plan #3 OpenClaw — before_tts_speak chain
        _hook_t0 = None
        if self.hooks is not None:
            from src.hooks import (
                BlockResult,
                TtsCall,
                execute_before_chain,
            )

            _hook_t0 = time.perf_counter()
            tts_call = TtsCall(
                text=text,
                voice=None,
                lang="es",
                user_id=None,
                zone_id=zone_id,
            )
            result = execute_before_chain(
                self.hooks, "before_tts_speak", tts_call,
                warn_ms=self._before_handler_warn_ms,
            )
            if isinstance(result, BlockResult):
                logger.info(
                    f"[TTS BLOCKED] rule={result.rule_name}: {result.reason}"
                )
                return
            # rewrite chain may have modified text (and possibly other fields)
            text = result.text
            zone_id = result.zone_id

        # Resolve zone from room context if available
        if room_context and hasattr(room_context, 'room_id') and not zone_id:
            zone_id = f"zone_{room_context.room_id}"

        # Aplicar ajustes de emoción si están disponibles
        if emotion_adjustment:
            self._apply_emotion_adjustment(emotion_adjustment)

        # Determinar si usar streaming
        use_streaming = stream if stream is not None else self.streaming_enabled

        # Enrutar a zona
        target_zone = zone_id or self._active_zone_id

        # Barge-in state on: desde aquí cualquier consulta a `is_speaking`
        # devuelve True. Se limpia en `finally` aunque haya excepciones.
        self._is_speaking = True
        try:
            # Cache hit (S2): playback directo si hay match. No aplicamos
            # emotion adjustment a respuestas cacheadas (el pitch/rate vive en
            # la síntesis live); son acks cortos pre-generados, el tradeoff
            # vale la pena.
            if self._response_cache is not None:
                cached = self._response_cache.get(text)
                if cached is not None:
                    t0 = time.perf_counter()
                    self._playback_cached(cached, target_zone)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info(
                        f"TTS cache HIT: {text!r} "
                        f"(playback={elapsed_ms:.1f}ms, audio={cached.duration_s*1000:.0f}ms)"
                    )
                    return

            if self.zone_manager and target_zone:
                self._speak_to_zone(text, target_zone, use_streaming)
            else:
                # Fallback: TTS directo
                self._speak_direct(text, use_streaming)
        finally:
            # `cancel()` pudo haberlo apagado antes; idempotent.
            self._is_speaking = False
            self._current_stream = None

        # Plan #3 OpenClaw — emit after-event TTS
        if self.hooks is not None:
            try:
                from src.hooks import TtsPayload, execute_after_event

                latency_ms = (
                    (time.perf_counter() - _hook_t0) * 1000.0
                    if _hook_t0 is not None
                    else 0.0
                )
                execute_after_event(
                    self.hooks,
                    "tts",
                    TtsPayload(
                        timestamp=time.time(),
                        text=text,
                        voice=None,
                        latency_ms=latency_ms,
                        success=True,
                    ),
                )
            except Exception as e:
                logger.warning(f"after_event tts emit failed: {e}")

    def _playback_cached(self, cached: CachedAudio, zone_id: str = None):
        """Reproducir audio cacheado reutilizando el mismo mecanismo que TTS live.

        Si hay zone_manager + zone_id, va por `play_to_zone` (mismo path que
        el TTS normal). En fallback directo usa `sounddevice.sd.play`, igual
        que `PiperTTS.speak`/`KokoroTTS.speak`.
        """
        if self.zone_manager and zone_id:
            self.zone_manager.play_to_zone(
                zone_id=zone_id,
                audio_data=cached.audio,
                sample_rate=cached.sample_rate,
                block=True,
            )
            return

        # Fallback directo: mismo mecanismo que *TTS.speak() — sounddevice.
        try:
            import sounddevice as sd
            sd.play(cached.audio, samplerate=cached.sample_rate)
            sd.wait()
        except Exception as e:
            logger.warning(f"Playback cacheado falló, fallback TTS live: {e}")
            # Si falla el playback cacheado, caer al TTS normal para no perder
            # el ack al usuario.
            self._speak_direct(cached.text, self.streaming_enabled)

    def _speak_to_zone(
        self,
        text: str,
        zone_id: str,
        use_streaming: bool
    ):
        """
        Sintetizar y reproducir en una zona específica.

        Args:
            text: Texto a sintetizar
            zone_id: ID de la zona
            use_streaming: Usar streaming de audio
        """
        if use_streaming and hasattr(self.tts, 'synthesize_stream'):
            # Streaming: menor latencia
            audio_generator = self.tts.synthesize_stream(text)
            self.zone_manager.play_stream_to_zone(
                zone_id=zone_id,
                audio_generator=audio_generator,
                sample_rate=self.tts.sample_rate,
                buffer_ms=self.streaming_buffer_ms,
                prebuffer_ms=self.streaming_prebuffer_ms
            )
            logger.debug(
                f"Streaming enviado a zona: {zone_id} "
                f"(buffer={self.streaming_buffer_ms}ms)"
            )
        else:
            # Síntesis completa, luego reproducir
            audio_data, _ = self.tts.synthesize(text)
            if audio_data is not None:
                self.zone_manager.play_to_zone(
                    zone_id=zone_id,
                    audio_data=audio_data,
                    sample_rate=self.tts.sample_rate,
                    block=True
                )

    def _speak_direct(self, text: str, use_streaming: bool):
        """
        Sintetizar y reproducir directamente (sin zone manager).

        Args:
            text: Texto a sintetizar
            use_streaming: Usar streaming
        """
        if use_streaming and hasattr(self.tts, 'speak_stream'):
            self.tts.speak_stream(
                text,
                buffer_ms=self.streaming_buffer_ms,
                prebuffer_ms=self.streaming_prebuffer_ms
            )
        else:
            self.tts.speak(text)

    def speak_with_llm_stream(
        self,
        prompt: str,
        zone_id: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_filler: bool = None,
        emotion_adjustment: dict = None,
        room_context=None,
    ) -> str:
        """
        Generar respuesta con LLM y hablar con buffering inteligente.

        Permite que el LLM genere a velocidad lenta (6-10 tok/s) mientras
        el TTS habla de forma fluida usando buffering por oraciones.

        Args:
            prompt: Prompt para el LLM
            zone_id: Zona de destino
            max_tokens: Tokens máximos
            temperature: Temperatura
            use_filler: Override para usar filler
            emotion_adjustment: Ajustes según emoción
            room_context: RoomContext para resolver zona automáticamente

        Returns:
            Texto completo de la respuesta
        """
        # Resolve zone from room context if available
        if room_context and hasattr(room_context, 'room_id') and not zone_id:
            zone_id = f"zone_{room_context.room_id}"

        # Aplicar ajustes de emoción
        if emotion_adjustment:
            self._apply_emotion_adjustment(emotion_adjustment)

        # Inicializar streamer si es necesario
        self._init_llm_streamer()

        # Usar filler por defecto según config
        filler = use_filler if use_filler is not None else self.llm_use_filler

        # Establecer zona activa si se proporciona
        if zone_id:
            self.set_active_zone(zone_id)

        # Barge-in state on durante todo el streaming del LLM (puede durar
        # varios segundos). El fallback que llama a `self.speak()` reusa el
        # mismo flag — idempotent, no problema.
        self._is_speaking = True
        try:
            # Verificar si el LLM soporta streaming
            if self.llm and hasattr(self.llm, 'generate_stream'):
                return self._llm_streamer.stream_and_speak(
                    self.llm,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_filler=filler
                )
            else:
                # Fallback: generar completo y luego hablar
                logger.warning("LLM no soporta streaming, usando modo normal")
                response = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                self.speak(response, zone_id=zone_id)
                return response
        finally:
            self._is_speaking = False
            self._current_stream = None

    def _init_llm_streamer(self):
        """Inicializar el streamer con buffering para el LLM."""
        if self._llm_streamer is not None:
            return

        # Crear configuración personalizada si hay filler phrases custom
        if self.llm_filler_phrases:
            config = BufferConfig(
                use_filler=self.llm_use_filler,
                filler_phrases=self.llm_filler_phrases
            )
            self._llm_streamer = BufferedLLMStreamer(self.tts, config)
        else:
            # Usar preset
            self._llm_streamer = create_buffered_streamer(
                self.tts,
                preset=self.llm_buffer_preset
            )
            self._llm_streamer.config.use_filler = self.llm_use_filler

        logger.info(
            f"LLM Streamer inicializado "
            f"(preset={self.llm_buffer_preset}, filler={self.llm_use_filler})"
        )

    def _apply_emotion_adjustment(self, emotion_adjustment: dict):
        """
        Aplicar ajustes de emoción al TTS.

        Args:
            emotion_adjustment: Dict con adjustments (pitch_shift, speech_rate, etc.)
        """
        if not hasattr(self.tts, 'set_audio_adjustment'):
            return

        try:
            # Aplicar ajustes al TTS si lo soporta
            if "pitch_shift" in emotion_adjustment:
                self.tts.set_audio_adjustment(
                    pitch=emotion_adjustment["pitch_shift"]
                )
            if "speech_rate" in emotion_adjustment:
                self.tts.set_audio_adjustment(
                    speed=emotion_adjustment["speech_rate"]
                )

            logger.debug(
                f"Emotion adjustment aplicado: {emotion_adjustment.get('emotional_tone', 'unknown')}"
            )
        except Exception as e:
            logger.warning(f"Error aplicando emotion adjustment: {e}")

    def speak_to_all_zones(self, text: str, stream: bool = None):
        """
        Sintetizar y reproducir en todas las zonas.

        Args:
            text: Texto a sintetizar
            stream: Usar streaming
        """
        self._is_speaking = True
        try:
            if not self.zone_manager:
                self._speak_direct(text, stream if stream is not None else self.streaming_enabled)
                return

            use_streaming = stream if stream is not None else self.streaming_enabled

            # Para todas las zonas, usar síntesis completa (más simple que streaming)
            audio_data, _ = self.tts.synthesize(text)
            if audio_data is not None:
                self.zone_manager.play_to_all_zones(
                    audio_data=audio_data,
                    sample_rate=self.tts.sample_rate
                )
                logger.debug(f"Audio reproducido en todas las zonas")
        finally:
            self._is_speaking = False
            self._current_stream = None
