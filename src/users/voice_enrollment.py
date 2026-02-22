"""
Voice Enrollment Module
Workflow para registrar nuevos usuarios y sus voces.

Proceso de enrollment:
1. Admin dice "agregar nueva persona"
2. Sistema pide nombre
3. Sistema pide nivel de permiso
4. Sistema solicita 3-5 muestras de voz
5. Sistema crea embedding promediado
6. Usuario registrado
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from .speaker_identifier import SpeakerIdentifier
from .user_manager import PermissionLevel, User, UserManager

logger = logging.getLogger(__name__)


class EnrollmentState(Enum):
    """Estados del proceso de enrollment"""
    IDLE = auto()
    WAITING_NAME = auto()
    WAITING_PERMISSION = auto()
    COLLECTING_SAMPLES = auto()
    CONFIRMING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class EnrollmentSession:
    """Sesión activa de enrollment"""
    state: EnrollmentState = EnrollmentState.IDLE
    requesting_user: User | None = None
    new_user_name: str | None = None
    permission_level: PermissionLevel | None = None
    voice_samples: list[np.ndarray] = field(default_factory=list)
    required_samples: int = 3
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


# Frases de permiso y sus niveles
PERMISSION_PHRASES = {
    "admin": PermissionLevel.ADMIN,
    "administrador": PermissionLevel.ADMIN,
    "adulto": PermissionLevel.ADULT,
    "adult": PermissionLevel.ADULT,
    "teenager": PermissionLevel.TEEN,
    "adolescente": PermissionLevel.TEEN,
    "teen": PermissionLevel.TEEN,
    "niño": PermissionLevel.CHILD,
    "niña": PermissionLevel.CHILD,
    "child": PermissionLevel.CHILD,
    "invitado": PermissionLevel.GUEST,
    "guest": PermissionLevel.GUEST,
}


class VoiceEnrollment:
    """
    Gestiona el proceso de enrollment de voz.

    Flujo conversacional:
    User: "Agregar nueva persona"
    Bot: "¿Cómo se llama?"
    User: "María"
    Bot: "¿Qué nivel de permiso? Admin, adulto, adolescente, niño o invitado"
    User: "Adulto"
    Bot: "María, di 3 frases diferentes para aprender tu voz"
    María: "Hola, soy María" (muestra 1)
    Bot: "Bien, continúa (2/3)"
    María: "Me gusta el café" (muestra 2)
    Bot: "Una más (3/3)"
    María: "Prende la luz del living" (muestra 3)
    Bot: "¡Listo! María registrada como adulto"
    """

    def __init__(
        self,
        user_manager: UserManager,
        speaker_identifier: SpeakerIdentifier,
        session_timeout: float = 120.0  # 2 minutos
    ):
        self.user_manager = user_manager
        self.speaker_id = speaker_identifier
        self.session_timeout = session_timeout

        self._session: EnrollmentSession | None = None

    @property
    def is_active(self) -> bool:
        """Verificar si hay una sesión de enrollment activa"""
        if self._session is None:
            return False

        # Verificar timeout
        if time.time() - self._session.last_activity > self.session_timeout:
            self._cancel_session("Sesión expirada por inactividad")
            return False

        return self._session.state not in [
            EnrollmentState.IDLE,
            EnrollmentState.COMPLETED,
            EnrollmentState.CANCELLED
        ]

    def handle(
        self,
        text: str,
        audio: np.ndarray | None = None,
        current_user: User | None = None
    ) -> dict:
        """
        Procesar entrada durante enrollment.

        Args:
            text: Texto transcrito
            audio: Audio original (para extraer embedding)
            current_user: Usuario actual (para verificar permisos)

        Returns:
            {
                "handled": bool,
                "response": str,
                "state": EnrollmentState,
                "completed_user": User o None
            }
        """
        result = {
            "handled": False,
            "response": "",
            "state": EnrollmentState.IDLE,
            "completed_user": None
        }

        text_lower = text.lower().strip()

        # Verificar si es comando de inicio
        if self._is_enrollment_trigger(text_lower):
            return self._start_enrollment(current_user)

        # Si no hay sesión activa, no manejar
        if not self.is_active:
            return result

        # Actualizar actividad
        self._session.last_activity = time.time()

        # Comando de cancelación
        if self._is_cancel_command(text_lower):
            return self._cancel_session("Enrollment cancelado")

        # Procesar según estado
        state = self._session.state

        if state == EnrollmentState.WAITING_NAME:
            return self._handle_name_input(text)

        elif state == EnrollmentState.WAITING_PERMISSION:
            return self._handle_permission_input(text_lower)

        elif state == EnrollmentState.COLLECTING_SAMPLES:
            return self._handle_voice_sample(audio)

        elif state == EnrollmentState.CONFIRMING:
            return self._handle_confirmation(text_lower)

        return result

    def _is_enrollment_trigger(self, text: str) -> bool:
        """Detectar si es comando para iniciar enrollment"""
        triggers = [
            "agregar persona",
            "agregar usuario",
            "nueva persona",
            "nuevo usuario",
            "registrar persona",
            "registrar usuario",
            "añadir persona",
            "add user",
            "add person"
        ]
        return any(trigger in text for trigger in triggers)

    def _is_cancel_command(self, text: str) -> bool:
        """Detectar comando de cancelación"""
        cancel_phrases = ["cancelar", "cancel", "salir", "exit", "olvídalo", "no importa"]
        return any(phrase in text for phrase in cancel_phrases)

    def _start_enrollment(self, requesting_user: User | None) -> dict:
        """Iniciar proceso de enrollment"""
        # Verificar permisos
        if requesting_user is not None:
            check = self.user_manager.check_action_permission(requesting_user, "add_user")
            if not check.allowed:
                return {
                    "handled": True,
                    "response": self.user_manager.format_permission_denied_message(
                        requesting_user, check
                    ),
                    "state": EnrollmentState.IDLE,
                    "completed_user": None
                }

        # Crear sesión
        self._session = EnrollmentSession(
            state=EnrollmentState.WAITING_NAME,
            requesting_user=requesting_user
        )

        logger.info("Enrollment iniciado")

        return {
            "handled": True,
            "response": "¿Cómo se llama la nueva persona?",
            "state": EnrollmentState.WAITING_NAME,
            "completed_user": None
        }

    def _handle_name_input(self, text: str) -> dict:
        """Procesar nombre del nuevo usuario"""
        # Limpiar nombre
        name = text.strip().title()

        # Validar
        if len(name) < 2:
            return {
                "handled": True,
                "response": "Ese nombre es muy corto. ¿Cómo se llama?",
                "state": EnrollmentState.WAITING_NAME,
                "completed_user": None
            }

        # Verificar si ya existe
        existing = self.user_manager.get_user_by_name(name)
        if existing:
            return {
                "handled": True,
                "response": f"Ya existe alguien llamado {name}. Usa otro nombre.",
                "state": EnrollmentState.WAITING_NAME,
                "completed_user": None
            }

        self._session.new_user_name = name
        self._session.state = EnrollmentState.WAITING_PERMISSION

        logger.info(f"Enrollment: nombre = {name}")

        return {
            "handled": True,
            "response": f"Perfecto, {name}. ¿Qué nivel de permiso tendrá? "
                       "Di: administrador, adulto, adolescente, niño o invitado.",
            "state": EnrollmentState.WAITING_PERMISSION,
            "completed_user": None
        }

    def _handle_permission_input(self, text: str) -> dict:
        """Procesar nivel de permiso"""
        # Buscar nivel en el texto
        permission_level = None

        for phrase, level in PERMISSION_PHRASES.items():
            if phrase in text:
                permission_level = level
                break

        if permission_level is None:
            return {
                "handled": True,
                "response": "No entendí el nivel. Di: administrador, adulto, adolescente, niño o invitado.",
                "state": EnrollmentState.WAITING_PERMISSION,
                "completed_user": None
            }

        # Verificar que el solicitante puede asignar ese nivel
        if self._session.requesting_user:
            if permission_level >= self._session.requesting_user.permission_level:
                if permission_level == PermissionLevel.ADMIN:
                    return {
                        "handled": True,
                        "response": "Solo un administrador puede crear otro administrador.",
                        "state": EnrollmentState.WAITING_PERMISSION,
                        "completed_user": None
                    }

        self._session.permission_level = permission_level
        self._session.state = EnrollmentState.COLLECTING_SAMPLES

        level_name = permission_level.name.lower()
        name = self._session.new_user_name

        logger.info(f"Enrollment: permiso = {level_name}")

        return {
            "handled": True,
            "response": f"{name}, voy a aprender tu voz. "
                       f"Di {self._session.required_samples} frases diferentes. "
                       "Empieza cuando quieras.",
            "state": EnrollmentState.COLLECTING_SAMPLES,
            "completed_user": None
        }

    def _handle_voice_sample(self, audio: np.ndarray | None) -> dict:
        """Procesar muestra de voz"""
        if audio is None or len(audio) < 8000:  # Mínimo 0.5 segundos
            return {
                "handled": True,
                "response": "No escuché bien. Intenta de nuevo con una frase más larga.",
                "state": EnrollmentState.COLLECTING_SAMPLES,
                "completed_user": None
            }

        # Agregar muestra
        self._session.voice_samples.append(audio)
        current = len(self._session.voice_samples)
        required = self._session.required_samples

        logger.info(f"Enrollment: muestra {current}/{required}")

        if current < required:
            remaining = required - current
            return {
                "handled": True,
                "response": f"Bien. {remaining} más.",
                "state": EnrollmentState.COLLECTING_SAMPLES,
                "completed_user": None
            }

        # Tenemos todas las muestras, crear embedding
        return self._complete_enrollment()

    def _complete_enrollment(self) -> dict:
        """Finalizar enrollment y crear usuario"""
        try:
            # Crear embedding promediado
            embedding = self.speaker_id.create_enrollment_embedding(
                self._session.voice_samples
            )

            # Crear usuario
            user, message = self.user_manager.add_user(
                name=self._session.new_user_name,
                permission_level=self._session.permission_level,
                voice_embedding=embedding,
                requesting_user=self._session.requesting_user
            )

            if user is None:
                return {
                    "handled": True,
                    "response": f"Error: {message}",
                    "state": EnrollmentState.CANCELLED,
                    "completed_user": None
                }

            self._session.state = EnrollmentState.COMPLETED

            level_names_es = {
                PermissionLevel.GUEST: "invitado",
                PermissionLevel.CHILD: "niño",
                PermissionLevel.TEEN: "adolescente",
                PermissionLevel.ADULT: "adulto",
                PermissionLevel.ADMIN: "administrador"
            }
            level_es = level_names_es.get(user.permission_level, "usuario")

            logger.info(f"Enrollment completado: {user.name} ({level_es})")

            return {
                "handled": True,
                "response": f"¡Listo! {user.name} registrado como {level_es}. "
                           f"Ya puedo reconocer tu voz.",
                "state": EnrollmentState.COMPLETED,
                "completed_user": user
            }

        except Exception as e:
            logger.error(f"Error en enrollment: {e}")
            return {
                "handled": True,
                "response": f"Hubo un error al registrar. Intenta de nuevo.",
                "state": EnrollmentState.CANCELLED,
                "completed_user": None
            }
        finally:
            self._session = None

    def _handle_confirmation(self, text: str) -> dict:
        """Manejar confirmación"""
        # Por ahora no usado, pero reservado para flujo con confirmación
        if "sí" in text or "si" in text or "confirmar" in text:
            return self._complete_enrollment()
        elif "no" in text:
            return self._cancel_session("Enrollment cancelado")

        return {
            "handled": True,
            "response": "Di 'sí' para confirmar o 'no' para cancelar.",
            "state": EnrollmentState.CONFIRMING,
            "completed_user": None
        }

    def _cancel_session(self, reason: str) -> dict:
        """Cancelar sesión de enrollment"""
        logger.info(f"Enrollment cancelado: {reason}")
        self._session = None

        return {
            "handled": True,
            "response": reason,
            "state": EnrollmentState.CANCELLED,
            "completed_user": None
        }

    def get_session_info(self) -> dict | None:
        """Obtener información de la sesión actual"""
        if self._session is None:
            return None

        return {
            "state": self._session.state.name,
            "name": self._session.new_user_name,
            "permission": self._session.permission_level.name if self._session.permission_level else None,
            "samples_collected": len(self._session.voice_samples),
            "samples_required": self._session.required_samples,
            "elapsed_seconds": time.time() - self._session.started_at
        }
