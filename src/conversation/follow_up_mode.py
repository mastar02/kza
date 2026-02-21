"""
Follow-Up Mode - Conversación sin Wake Word
Permite conversación natural sin repetir el wake word después de cada interacción.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Estados de la conversación"""
    IDLE = "idle"                    # Esperando wake word
    ACTIVE = "active"                # Conversación activa (sin need wake word)
    LISTENING = "listening"          # Escuchando activamente
    PROCESSING = "processing"        # Procesando respuesta
    SPEAKING = "speaking"            # KZA hablando
    WAITING_FOLLOWUP = "waiting"     # Esperando posible follow-up


@dataclass
class ConversationContext:
    """Contexto de la conversación activa"""
    user_id: Optional[str] = None
    started_at: float = 0
    last_interaction: float = 0
    turn_count: int = 0
    topic: Optional[str] = None
    entities: dict = field(default_factory=dict)  # Entidades mencionadas
    pending_question: bool = False  # KZA hizo una pregunta
    last_response_type: str = ""    # question, statement, command_result


class FollowUpMode:
    """
    Gestiona el modo de conversación continua sin wake word.

    Características:
    - Ventana de follow-up configurable (default 8 segundos)
    - Extiende automáticamente si KZA hace una pregunta
    - Detecta fin de conversación por silencio o despedida
    - Soporta múltiples usuarios (cada uno su contexto)
    """

    # Frases que indican fin de conversación
    END_PHRASES = [
        "gracias", "ok gracias", "eso es todo", "nada más",
        "listo", "perfecto", "vale", "está bien", "bye",
        "adiós", "hasta luego", "nos vemos", "chao"
    ]

    # Frases que indican continuación
    CONTINUE_PHRASES = [
        "y también", "otra cosa", "además", "espera",
        "una pregunta", "oye", "ah", "eh"
    ]

    def __init__(
        self,
        follow_up_window: float = 8.0,      # Segundos para follow-up
        question_window: float = 15.0,       # Ventana extendida si KZA preguntó
        max_conversation_time: float = 300,  # 5 min máximo por conversación
        whisper_detection: bool = True       # Detectar susurros
    ):
        self.follow_up_window = follow_up_window
        self.question_window = question_window
        self.max_conversation_time = max_conversation_time
        self.whisper_detection = whisper_detection

        self._state = ConversationState.IDLE
        self._context: Optional[ConversationContext] = None
        self._timeout_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_timeout: Optional[Callable] = None
        self._on_conversation_end: Optional[Callable] = None

        # Audio level tracking para whisper detection
        self._recent_audio_levels: list[float] = []
        self._whisper_threshold = 0.3  # Relativo al nivel normal

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def is_active(self) -> bool:
        """¿Está en modo conversación activa?"""
        return self._state in [
            ConversationState.ACTIVE,
            ConversationState.LISTENING,
            ConversationState.PROCESSING,
            ConversationState.SPEAKING,
            ConversationState.WAITING_FOLLOWUP
        ]

    @property
    def needs_wake_word(self) -> bool:
        """¿Necesita wake word para el próximo comando?"""
        return self._state == ConversationState.IDLE

    @property
    def context(self) -> Optional[ConversationContext]:
        return self._context

    def start_conversation(self, user_id: str = None, triggered_by: str = "wake_word"):
        """Iniciar nueva conversación"""
        now = time.time()

        self._context = ConversationContext(
            user_id=user_id,
            started_at=now,
            last_interaction=now,
            turn_count=0
        )

        self._transition_to(ConversationState.ACTIVE)
        logger.info(f"Conversación iniciada para {user_id or 'unknown'} via {triggered_by}")

        # Iniciar timer de timeout
        self._schedule_timeout()

    def on_user_speech(self, text: str, user_id: str = None, audio_level: float = 1.0):
        """
        Llamar cuando el usuario habla.

        Args:
            text: Texto transcrito
            user_id: ID del usuario (si identificado)
            audio_level: Nivel de audio normalizado (para whisper detection)
        """
        if not self._context:
            return

        now = time.time()
        self._context.last_interaction = now
        self._context.turn_count += 1

        # Track audio level para whisper detection
        self._recent_audio_levels.append(audio_level)
        if len(self._recent_audio_levels) > 10:
            self._recent_audio_levels.pop(0)

        # Verificar si es despedida
        text_lower = text.lower().strip()
        if self._is_end_phrase(text_lower):
            logger.info(f"Detectada frase de despedida: {text}")
            self._schedule_end_conversation(delay=2.0)  # Pequeño delay para responder
            return

        # Verificar si continúa
        if self._is_continue_phrase(text_lower):
            self._extend_timeout(extra=5.0)

        # Resetear timeout
        self._schedule_timeout()
        self._transition_to(ConversationState.PROCESSING)

    def on_kza_response(
        self,
        response_type: str = "statement",
        asked_question: bool = False,
        entities: dict = None
    ):
        """
        Llamar cuando KZA responde.

        Args:
            response_type: "question", "statement", "command_result", "clarification"
            asked_question: Si KZA hizo una pregunta
            entities: Entidades mencionadas en la respuesta
        """
        if not self._context:
            return

        self._context.last_response_type = response_type
        self._context.pending_question = asked_question

        if entities:
            self._context.entities.update(entities)

        self._transition_to(ConversationState.SPEAKING)

    def on_kza_finished_speaking(self):
        """Llamar cuando KZA termina de hablar"""
        if not self._context:
            return

        self._transition_to(ConversationState.WAITING_FOLLOWUP)

        # Ajustar timeout según si hizo pregunta
        if self._context.pending_question:
            self._schedule_timeout(window=self.question_window)
        else:
            self._schedule_timeout()

    def end_conversation(self, reason: str = "manual"):
        """Terminar conversación"""
        if self._context:
            duration = time.time() - self._context.started_at
            turns = self._context.turn_count
            logger.info(f"Conversación terminada: {reason}, {turns} turnos, {duration:.1f}s")

            if self._on_conversation_end:
                self._on_conversation_end(self._context, reason)

        self._cancel_timeout()
        self._context = None
        self._transition_to(ConversationState.IDLE)

    def is_whisper(self, audio_level: float) -> bool:
        """Detectar si el audio es un susurro"""
        if not self.whisper_detection or not self._recent_audio_levels:
            return False

        avg_level = sum(self._recent_audio_levels) / len(self._recent_audio_levels)
        return audio_level < avg_level * self._whisper_threshold

    def get_follow_up_window(self) -> float:
        """Obtener ventana de follow-up actual"""
        if self._context and self._context.pending_question:
            return self.question_window
        return self.follow_up_window

    def should_accept_speech(self, has_wake_word: bool = False) -> bool:
        """
        ¿Debería aceptar este audio como comando?

        Args:
            has_wake_word: Si el audio contiene el wake word

        Returns:
            True si debería procesar el comando
        """
        # Siempre aceptar si tiene wake word
        if has_wake_word:
            return True

        # En modo activo, aceptar sin wake word
        if self.is_active:
            # Verificar que no ha expirado el timeout
            if self._context:
                elapsed = time.time() - self._context.last_interaction
                window = self.get_follow_up_window()
                if elapsed < window:
                    return True

        return False

    # ==================== Internal ====================

    def _transition_to(self, new_state: ConversationState):
        """Transicionar a nuevo estado"""
        old_state = self._state
        self._state = new_state

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

        logger.debug(f"Conversation state: {old_state.value} -> {new_state.value}")

    def _schedule_timeout(self, window: float = None):
        """Programar timeout de conversación"""
        self._cancel_timeout()

        timeout = window or self.get_follow_up_window()
        self._timeout_task = asyncio.create_task(self._timeout_handler(timeout))

    def _extend_timeout(self, extra: float):
        """Extender el timeout actual"""
        self._cancel_timeout()
        window = self.get_follow_up_window() + extra
        self._timeout_task = asyncio.create_task(self._timeout_handler(window))

    def _cancel_timeout(self):
        """Cancelar timeout pendiente"""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    async def _timeout_handler(self, timeout: float):
        """Handler de timeout"""
        try:
            await asyncio.sleep(timeout)

            # Verificar si la conversación excedió el máximo
            if self._context:
                duration = time.time() - self._context.started_at
                if duration >= self.max_conversation_time:
                    self.end_conversation("max_time")
                    return

            logger.debug(f"Follow-up timeout después de {timeout}s")

            if self._on_timeout:
                self._on_timeout()

            self.end_conversation("timeout")

        except asyncio.CancelledError:
            pass

    def _schedule_end_conversation(self, delay: float):
        """Programar fin de conversación con delay"""
        async def delayed_end():
            await asyncio.sleep(delay)
            self.end_conversation("user_ended")

        asyncio.create_task(delayed_end())

    def _is_end_phrase(self, text: str) -> bool:
        """Verificar si es frase de despedida"""
        return any(phrase in text for phrase in self.END_PHRASES)

    def _is_continue_phrase(self, text: str) -> bool:
        """Verificar si indica continuación"""
        return any(phrase in text for phrase in self.CONTINUE_PHRASES)

    # ==================== Callbacks ====================

    def on_state_change(self, callback: Callable[[ConversationState, ConversationState], None]):
        """Registrar callback para cambios de estado"""
        self._on_state_change = callback

    def on_timeout(self, callback: Callable[[], None]):
        """Registrar callback para timeout"""
        self._on_timeout = callback

    def on_conversation_end(self, callback: Callable[[ConversationContext, str], None]):
        """Registrar callback para fin de conversación"""
        self._on_conversation_end = callback

    def get_status(self) -> dict:
        """Obtener estado completo"""
        status = {
            "state": self._state.value,
            "is_active": self.is_active,
            "needs_wake_word": self.needs_wake_word,
            "follow_up_window": self.get_follow_up_window()
        }

        if self._context:
            status["context"] = {
                "user_id": self._context.user_id,
                "turn_count": self._context.turn_count,
                "duration": time.time() - self._context.started_at,
                "pending_question": self._context.pending_question
            }

        return status
