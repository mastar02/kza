"""
Voice Routine Handler - LLM-Based
Manejo avanzado de rutinas por comandos de voz usando LLM para NLU robusto.

FIX: Reescrito para usar LLM en lugar de regex frágiles.
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class VoiceRoutineIntent:
    """Intención detectada para rutinas"""
    intent: str  # create, delete, list, execute, enable, disable, edit, none
    routine_name: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""
    extracted_data: dict = field(default_factory=dict)


# Prompt template for intent classification
INTENT_CLASSIFICATION_PROMPT = """Eres un asistente de domótica que clasifica comandos de voz sobre rutinas.

Comandos del usuario: "{text}"

Clasifica la intención del usuario. Responde SOLO con JSON válido:

{{
  "intent": "create|execute|list|delete|enable|disable|status|none",
  "routine_name": "nombre de la rutina mencionada o null",
  "confidence": 0.0-1.0,
  "is_routine_related": true/false
}}

Intenciones:
- "create": Usuario quiere crear/programar una nueva rutina o automatización
- "execute": Usuario quiere ejecutar/activar una rutina existente
- "list": Usuario quiere ver/listar sus rutinas
- "delete": Usuario quiere eliminar/borrar una rutina
- "enable": Usuario quiere activar/habilitar una rutina pausada
- "disable": Usuario quiere desactivar/pausar una rutina
- "status": Usuario pregunta por el estado de una rutina
- "none": No es un comando de rutina

JSON:"""


# Prompt for extracting routine components
ROUTINE_EXTRACTION_PROMPT = """Extrae los componentes de esta rutina de domótica:

Comando: "{text}"

Extrae triggers (cuándo), conditions (si aplica), y actions (qué hacer).

Responde SOLO con JSON válido:

{{
  "name": "nombre sugerido para la rutina",
  "triggers": [
    {{
      "type": "time|presence_enter|presence_leave|sunrise|sunset|device_state|voice",
      "config": {{}}
    }}
  ],
  "conditions": [
    {{
      "type": "time_range|device_state|presence",
      "config": {{}}
    }}
  ],
  "actions": [
    {{
      "type": "ha_service",
      "domain": "light|switch|climate|scene|script|media_player",
      "service": "turn_on|turn_off|toggle|set_temperature|...",
      "entity_id": "light.sala|climate.casa|scene.pelicula|...",
      "data": {{}}
    }}
  ]
}}

Ejemplos de triggers:
- time: {{"at": "07:00:00", "days": ["mon","tue","wed","thu","fri"]}}
- presence_enter: {{"zone": "home"}}
- presence_leave: {{"zone": "home"}}
- sunrise: {{"offset_minutes": 0}}
- sunset: {{"offset_minutes": 30}}
- device_state: {{"entity_id": "binary_sensor.puerta", "state": "on"}}

Ejemplos de actions:
- Encender luz: {{"type": "ha_service", "domain": "light", "service": "turn_on", "entity_id": "light.sala"}}
- Ajustar brillo: {{"type": "ha_service", "domain": "light", "service": "turn_on", "entity_id": "light.sala", "data": {{"brightness_pct": 50}}}}
- Temperatura: {{"type": "ha_service", "domain": "climate", "service": "set_temperature", "entity_id": "climate.casa", "data": {{"temperature": 22}}}}
- Escena: {{"type": "ha_scene", "entity_id": "scene.pelicula"}}
- Spotify: {{"type": "spotify_play", "mood": "relajante"}}
- TTS: {{"type": "tts_speak", "text": "Bienvenido a casa"}}

JSON:"""


# PRIORIDAD: Velocidad > precisión
# Fast-path extenso para evitar latencia de LLM en la mayoría de casos
FAST_PATTERNS = {
    "execute": [
        "rutina ", "ejecuta ", "activa rutina", "modo ",
        "pon la rutina", "lanza rutina", "inicia rutina",
        "corre rutina", "arranca ", "activa ", "pon modo"
    ],
    "list": [
        "mis rutinas", "qué rutinas", "cuáles rutinas",
        "lista de rutinas", "muestra rutinas", "tengo rutinas",
        "ver rutinas", "dime las rutinas", "rutinas activas"
    ],
    "create": [
        "crea rutina", "nueva rutina", "crear rutina",
        "programa rutina", "automatiza", "configura rutina",
        "quiero que cuando", "cada vez que", "cuando llegue",
        "a las ", "al amanecer", "al atardecer"
    ],
    "delete": [
        "elimina rutina", "borra rutina", "quita rutina",
        "elimina la rutina", "borra la rutina"
    ],
    "enable": [
        "activa la rutina", "habilita rutina", "enciende rutina",
        "reactiva rutina"
    ],
    "disable": [
        "desactiva rutina", "pausa rutina", "apaga rutina",
        "deshabilita rutina", "para rutina"
    ],
    "status": [
        "estado de rutina", "cómo está la rutina", "status rutina"
    ],
}


class VoiceRoutineHandler:
    """
    Handler de rutinas por voz usando LLM para NLU robusto.

    Características:
    - Detección de intención basada en LLM (no regex frágiles)
    - Extracción de entidades con contexto
    - Manejo de conversación multi-turno
    - Fast-path para comandos obvios (baja latencia)
    - Fallback graceful si LLM no está disponible
    """

    CONFIRM_WORDS = ["sí", "si", "confirmo", "dale", "ok", "correcto", "perfecto", "adelante", "hazlo"]
    CANCEL_WORDS = ["no", "cancela", "cancelar", "olvídalo", "déjalo", "para"]

    def __init__(
        self,
        routine_scheduler,
        routine_manager=None,
        llm_client=None,
        fast_mode: bool = True
    ):
        """
        Args:
            routine_scheduler: RoutineScheduler para gestionar rutinas
            routine_manager: RoutineManager opcional (legacy)
            llm_client: Cliente LLM para NLU (debe tener método generate o __call__)
            fast_mode: Usar fast-path para comandos obvios
        """
        self.scheduler = routine_scheduler
        self.manager = routine_manager
        self.llm = llm_client
        self.fast_mode = fast_mode

        # Estado de conversación
        self._pending_routine: Optional[dict] = None
        self._awaiting_confirmation: bool = False
        self._conversation_context: list[dict] = []
        self._last_intent: Optional[VoiceRoutineIntent] = None

        # Cache de intents recientes para evitar re-clasificar
        self._intent_cache: dict[str, VoiceRoutineIntent] = {}
        self._cache_ttl = 30  # segundos

    async def detect_intent(self, text: str) -> VoiceRoutineIntent:
        """
        Detectar intención del comando de voz.

        Usa LLM para clasificación robusta con fast-path para comandos obvios.
        """
        text_lower = text.lower().strip()

        # 1. Fast-path para comandos obvios (baja latencia)
        if self.fast_mode:
            fast_intent = self._fast_classify(text_lower)
            if fast_intent:
                return fast_intent

        # 2. Check cache
        cache_key = text_lower[:100]
        if cache_key in self._intent_cache:
            cached = self._intent_cache[cache_key]
            # Cache hit
            return cached

        # 3. Usar LLM para clasificación
        if self.llm:
            intent = await self._classify_with_llm(text)
            if intent.confidence > 0.5:
                self._intent_cache[cache_key] = intent
                return intent

        # 4. Fallback: no es comando de rutina
        return VoiceRoutineIntent(
            intent="none",
            confidence=0.0,
            raw_text=text
        )

    def _fast_classify(self, text: str) -> Optional[VoiceRoutineIntent]:
        """
        Clasificación rápida para comandos obvios.
        PRIORIDAD: Velocidad - evita latencia de LLM en el 90% de casos.
        """
        # Execute rutina
        for pattern in FAST_PATTERNS.get("execute", []):
            if pattern in text:
                name = self._extract_routine_name(text)
                if name:
                    return VoiceRoutineIntent(
                        intent="execute",
                        routine_name=name,
                        confidence=0.95,
                        raw_text=text
                    )

        # List rutinas
        for pattern in FAST_PATTERNS.get("list", []):
            if pattern in text:
                return VoiceRoutineIntent(
                    intent="list",
                    confidence=0.95,
                    raw_text=text
                )

        # Create rutina
        for pattern in FAST_PATTERNS.get("create", []):
            if pattern in text:
                return VoiceRoutineIntent(
                    intent="create",
                    confidence=0.90,
                    raw_text=text
                )

        # Delete rutina
        for pattern in FAST_PATTERNS.get("delete", []):
            if pattern in text:
                name = self._extract_routine_name(text)
                return VoiceRoutineIntent(
                    intent="delete",
                    routine_name=name,
                    confidence=0.95,
                    raw_text=text
                )

        # Enable rutina
        for pattern in FAST_PATTERNS.get("enable", []):
            if pattern in text:
                name = self._extract_routine_name(text)
                return VoiceRoutineIntent(
                    intent="enable",
                    routine_name=name,
                    confidence=0.95,
                    raw_text=text
                )

        # Disable rutina
        for pattern in FAST_PATTERNS.get("disable", []):
            if pattern in text:
                name = self._extract_routine_name(text)
                return VoiceRoutineIntent(
                    intent="disable",
                    routine_name=name,
                    confidence=0.95,
                    raw_text=text
                )

        # Status rutina
        for pattern in FAST_PATTERNS.get("status", []):
            if pattern in text:
                name = self._extract_routine_name(text)
                return VoiceRoutineIntent(
                    intent="status",
                    routine_name=name,
                    confidence=0.95,
                    raw_text=text
                )

        return None

    def _extract_routine_name(self, text: str) -> Optional[str]:
        """Extraer nombre de rutina del texto de forma rápida"""
        # Patrones comunes para extraer nombre
        patterns = [
            ("rutina ", None),
            ("modo ", None),
            ("la rutina ", None),
            ("rutina llamada ", None),
        ]

        for pattern, _ in patterns:
            if pattern in text:
                parts = text.split(pattern)
                if len(parts) > 1:
                    # Tomar lo que viene después y limpiar
                    name = parts[1].strip().strip("'\"")
                    # Cortar en palabras clave que no son parte del nombre
                    for stop in [" que ", " para ", " porque ", " y "]:
                        if stop in name:
                            name = name.split(stop)[0]
                    return name.strip() if name else None

        return None

    async def _classify_with_llm(self, text: str) -> VoiceRoutineIntent:
        """Clasificar intención usando LLM"""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(text=text)

        try:
            # Llamar al LLM
            if hasattr(self.llm, 'generate'):
                response = await self._call_llm_async(
                    self.llm.generate,
                    prompt,
                    max_tokens=150,
                    temperature=0.1
                )
            elif callable(self.llm):
                response = await self._call_llm_async(
                    self.llm,
                    prompt,
                    max_tokens=150,
                    temperature=0.1
                )
            else:
                logger.warning("LLM client no tiene método generate ni es callable")
                return VoiceRoutineIntent(intent="none", confidence=0.0, raw_text=text)

            # Parsear respuesta JSON
            result = self._extract_json(response)

            if result and result.get("is_routine_related", False):
                return VoiceRoutineIntent(
                    intent=result.get("intent", "none"),
                    routine_name=result.get("routine_name"),
                    confidence=float(result.get("confidence", 0.7)),
                    raw_text=text,
                    extracted_data=result
                )

        except Exception as e:
            logger.error(f"Error en clasificación LLM: {e}")

        return VoiceRoutineIntent(intent="none", confidence=0.0, raw_text=text)

    async def _call_llm_async(self, func, *args, **kwargs) -> str:
        """Llamar función LLM de forma async"""
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            # Ejecutar en thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

        # Extraer texto de diferentes formatos de respuesta
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Formato OpenAI/LMStudio
            if "choices" in result:
                return result["choices"][0].get("text", "") or result["choices"][0].get("message", {}).get("content", "")
            # Formato directo
            return result.get("text", "") or result.get("content", "")

        return str(result)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extraer JSON de respuesta de LLM"""
        try:
            # Buscar JSON en el texto
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.debug(f"Error parseando JSON: {e}")

        return None

    async def handle(self, text: str, user_id: str = None) -> dict:
        """
        Manejar comando de voz de rutina.

        Returns:
            {
                "handled": bool,
                "response": str,
                "success": bool,
                "needs_confirmation": bool,
                "data": dict  # datos adicionales
            }
        """
        result = {
            "handled": False,
            "response": "",
            "success": False,
            "needs_confirmation": False,
            "data": {}
        }

        # Verificar si esperamos confirmación
        if self._awaiting_confirmation:
            return await self._handle_confirmation(text, result, user_id)

        # Detectar intención
        intent = await self.detect_intent(text)
        self._last_intent = intent

        if intent.intent == "none" or intent.confidence < 0.5:
            return result

        result["handled"] = True
        result["data"]["intent"] = intent.intent
        result["data"]["confidence"] = intent.confidence

        # Dispatch al handler apropiado
        handlers = {
            "create": self._handle_create,
            "execute": self._handle_execute,
            "list": self._handle_list,
            "delete": self._handle_delete,
            "enable": self._handle_enable,
            "disable": self._handle_disable,
            "status": self._handle_status,
        }

        handler = handlers.get(intent.intent)
        if handler:
            return await handler(intent, result, user_id)

        return result

    # ==================== Handlers ====================

    async def _handle_create(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Manejar creación de rutina usando LLM para extracción"""
        logger.info(f"Creando rutina desde voz: {intent.raw_text}")

        # Usar LLM para extraer componentes
        routine_data = await self._extract_routine_with_llm(intent.raw_text)

        if not routine_data or not routine_data.get("triggers") or not routine_data.get("actions"):
            result["response"] = (
                "No pude entender completamente la rutina. "
                "¿Puedes decirme cuándo quieres que se active y qué quieres que haga?"
            )
            self._conversation_context.append({
                "role": "assistant",
                "content": result["response"],
                "partial_routine": routine_data
            })
            return result

        # Guardar para confirmación
        self._pending_routine = routine_data
        self._pending_routine["created_by"] = user_id or "voice"
        self._awaiting_confirmation = True

        # Generar resumen legible
        summary = await self._summarize_routine(routine_data)

        result["response"] = f"Voy a crear esta rutina: {summary}. ¿Confirmas?"
        result["success"] = True
        result["needs_confirmation"] = True
        result["data"]["routine_preview"] = routine_data

        return result

    async def _extract_routine_with_llm(self, text: str) -> dict:
        """Extraer componentes de rutina usando LLM"""
        if not self.llm:
            logger.warning("LLM no disponible para extracción de rutina")
            return {}

        prompt = ROUTINE_EXTRACTION_PROMPT.format(text=text)

        try:
            if hasattr(self.llm, 'generate'):
                response = await self._call_llm_async(
                    self.llm.generate,
                    prompt,
                    max_tokens=500,
                    temperature=0.2
                )
            elif callable(self.llm):
                response = await self._call_llm_async(
                    self.llm,
                    prompt,
                    max_tokens=500,
                    temperature=0.2
                )
            else:
                return {}

            result = self._extract_json(response)

            if result:
                # Validar estructura mínima
                if "triggers" in result and "actions" in result:
                    return result

        except Exception as e:
            logger.error(f"Error extrayendo rutina con LLM: {e}")

        return {}

    async def _summarize_routine(self, routine_data: dict) -> str:
        """Generar resumen legible de la rutina"""
        triggers = routine_data.get("triggers", [])
        actions = routine_data.get("actions", [])
        name = routine_data.get("name", "")

        parts = []

        # Nombre
        if name:
            parts.append(f"'{name}'")

        # Triggers
        trigger_descs = []
        for t in triggers:
            t_type = t.get("type", "")
            config = t.get("config", {})

            if t_type == "time":
                time_at = config.get("at", "")[:5]
                days = config.get("days", [])
                day_str = "todos los días" if not days or len(days) == 7 else f"{len(days)} días"
                trigger_descs.append(f"a las {time_at} ({day_str})")
            elif t_type == "presence_enter":
                trigger_descs.append("cuando llegues a casa")
            elif t_type == "presence_leave":
                trigger_descs.append("cuando salgas")
            elif t_type == "sunrise":
                offset = config.get("offset_minutes", 0)
                if offset:
                    trigger_descs.append(f"al amanecer +{offset}min")
                else:
                    trigger_descs.append("al amanecer")
            elif t_type == "sunset":
                trigger_descs.append("al atardecer")
            else:
                trigger_descs.append(t_type)

        if trigger_descs:
            parts.append(", ".join(trigger_descs))

        # Actions
        action_descs = []
        for a in actions:
            a_type = a.get("type", "")
            domain = a.get("domain", "")
            service = a.get("service", "")
            entity = a.get("entity_id", "")
            data = a.get("data", {})

            if domain == "light":
                room = entity.split(".")[-1].replace("_", " ") if entity else "luces"
                if service == "turn_on":
                    brightness = data.get("brightness_pct")
                    if brightness:
                        action_descs.append(f"encender {room} al {brightness}%")
                    else:
                        action_descs.append(f"encender {room}")
                else:
                    action_descs.append(f"apagar {room}")
            elif domain == "climate":
                temp = data.get("temperature")
                if temp:
                    action_descs.append(f"poner clima a {temp}°")
            elif domain == "scene" or a_type == "ha_scene":
                scene = entity.split(".")[-1].replace("_", " ") if entity else "escena"
                action_descs.append(f"activar escena {scene}")
            elif a_type == "spotify_play":
                mood = a.get("mood", "música")
                action_descs.append(f"poner {mood}")
            elif a_type == "tts_speak":
                action_descs.append("anunciar mensaje")
            else:
                action_descs.append(f"{service} {entity}" if service else str(a_type))

        if action_descs:
            parts.append("→ " + ", ".join(action_descs))

        return " ".join(parts)

    async def _handle_execute(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Ejecutar rutina por nombre"""
        routine_name = intent.routine_name

        if not routine_name:
            result["response"] = "¿Qué rutina quieres ejecutar?"
            return result

        # Buscar y ejecutar
        routines = self.scheduler.get_all_routines()

        # Búsqueda fuzzy por nombre
        routine = self._find_routine_by_name(routine_name, routines)

        if routine:
            # Verificar permisos
            if hasattr(routine, 'allowed_users') and routine.allowed_users:
                if user_id and user_id not in routine.allowed_users:
                    result["response"] = f"No tienes permiso para ejecutar '{routine.name}'"
                    return result

            # Ejecutar
            execution_result = await self.scheduler.execute_by_name(
                routine.name,
                context={"user_id": user_id, "trigger": "voice"}
            )

            if execution_result:
                result["response"] = f"Listo, ejecuté '{routine.name}'"
                result["success"] = True
            else:
                result["response"] = f"Hubo un problema ejecutando '{routine.name}'"
        else:
            # Sugerir rutinas similares
            similar = self._find_similar_routines(routine_name, routines, limit=3)
            if similar:
                names = ", ".join(r.name for r in similar)
                result["response"] = f"No encontré '{routine_name}'. ¿Quisiste decir: {names}?"
            else:
                result["response"] = f"No encontré la rutina '{routine_name}'"

        return result

    def _find_routine_by_name(self, name: str, routines: list) -> Optional[Any]:
        """Buscar rutina por nombre (exacto o parcial)"""
        name_lower = name.lower()

        # Búsqueda exacta
        for r in routines:
            if r.name.lower() == name_lower:
                return r

        # Búsqueda parcial
        for r in routines:
            if name_lower in r.name.lower() or r.name.lower() in name_lower:
                return r

        return None

    def _find_similar_routines(self, name: str, routines: list, limit: int = 3) -> list:
        """Encontrar rutinas con nombres similares"""
        name_lower = name.lower()
        scored = []

        for r in routines:
            r_name = r.name.lower()
            # Score simple basado en coincidencia de palabras
            name_words = set(name_lower.split())
            r_words = set(r_name.split())
            common = len(name_words & r_words)

            if common > 0 or any(w in r_name for w in name_words):
                scored.append((r, common))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:limit]]

    async def _handle_list(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Listar rutinas del usuario"""
        routines = self.scheduler.get_all_routines()

        # Filtrar por usuario si aplica
        if user_id:
            user_routines = [
                r for r in routines
                if not hasattr(r, 'allowed_users') or not r.allowed_users or user_id in r.allowed_users
            ]
        else:
            user_routines = routines

        if user_routines:
            enabled = [r for r in user_routines if r.enabled]
            disabled = [r for r in user_routines if not r.enabled]

            # Limitar para respuesta por voz
            names = [r.name for r in enabled[:5]]

            response = f"Tienes {len(user_routines)} rutinas"
            if enabled:
                response += f". Activas: {', '.join(names)}"
                if len(enabled) > 5:
                    response += f" y {len(enabled) - 5} más"
            if disabled:
                response += f". {len(disabled)} están pausadas"

            result["response"] = response
            result["data"]["routines"] = [{"name": r.name, "enabled": r.enabled} for r in user_routines]
        else:
            result["response"] = "No tienes rutinas creadas. ¿Quieres que creemos una?"

        result["success"] = True
        return result

    async def _handle_delete(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Eliminar rutina"""
        routine_name = intent.routine_name

        if not routine_name:
            result["response"] = "¿Qué rutina quieres eliminar?"
            return result

        routines = self.scheduler.get_all_routines()
        routine = self._find_routine_by_name(routine_name, routines)

        if routine:
            # Verificar permisos (solo owner puede eliminar)
            if hasattr(routine, 'owner_user_id') and routine.owner_user_id:
                if user_id and user_id != routine.owner_user_id:
                    result["response"] = f"Solo el creador puede eliminar '{routine.name}'"
                    return result

            self.scheduler.unregister_routine(routine.routine_id)
            result["response"] = f"Eliminé la rutina '{routine.name}'"
            result["success"] = True
        else:
            result["response"] = f"No encontré la rutina '{routine_name}'"

        return result

    async def _handle_enable(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Habilitar rutina"""
        routine_name = intent.routine_name

        if not routine_name:
            result["response"] = "¿Qué rutina quieres activar?"
            return result

        routines = self.scheduler.get_all_routines()
        routine = self._find_routine_by_name(routine_name, routines)

        if routine:
            self.scheduler.enable_routine(routine.routine_id)
            result["response"] = f"Activé la rutina '{routine.name}'"
            result["success"] = True
        else:
            result["response"] = f"No encontré la rutina '{routine_name}'"

        return result

    async def _handle_disable(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Deshabilitar rutina"""
        routine_name = intent.routine_name

        if not routine_name:
            result["response"] = "¿Qué rutina quieres pausar?"
            return result

        routines = self.scheduler.get_all_routines()
        routine = self._find_routine_by_name(routine_name, routines)

        if routine:
            self.scheduler.disable_routine(routine.routine_id)
            result["response"] = f"Pausé la rutina '{routine.name}'"
            result["success"] = True
        else:
            result["response"] = f"No encontré la rutina '{routine_name}'"

        return result

    async def _handle_status(
        self,
        intent: VoiceRoutineIntent,
        result: dict,
        user_id: str
    ) -> dict:
        """Estado de rutina"""
        routine_name = intent.routine_name

        if not routine_name:
            # Estado general
            routines = self.scheduler.get_all_routines()
            enabled = sum(1 for r in routines if r.enabled)
            result["response"] = f"Tienes {len(routines)} rutinas, {enabled} activas"
            result["success"] = True
            return result

        routines = self.scheduler.get_all_routines()
        routine = self._find_routine_by_name(routine_name, routines)

        if routine:
            status = "activa" if routine.enabled else "pausada"
            last = routine.last_executed
            last_str = last.strftime("%H:%M") if last else "nunca"
            count = routine.execution_count

            result["response"] = (
                f"'{routine.name}' está {status}. "
                f"Se ha ejecutado {count} veces, última: {last_str}"
            )
            result["success"] = True
            result["data"]["routine_status"] = {
                "name": routine.name,
                "enabled": routine.enabled,
                "execution_count": count,
                "last_executed": last_str
            }
        else:
            result["response"] = f"No encontré la rutina '{routine_name}'"

        return result

    async def _handle_confirmation(
        self,
        text: str,
        result: dict,
        user_id: str
    ) -> dict:
        """Manejar confirmación de rutina pendiente"""
        text_lower = text.lower().strip()
        result["handled"] = True

        if any(word in text_lower for word in self.CONFIRM_WORDS):
            # Crear la rutina
            from .routine_scheduler import ScheduledRoutine
            import uuid

            routine = ScheduledRoutine(
                routine_id=f"voice_{uuid.uuid4().hex[:8]}",
                name=self._pending_routine.get("name", f"Rutina {datetime.now().strftime('%H%M')}"),
                triggers=self._pending_routine.get("triggers", []),
                conditions=self._pending_routine.get("conditions", []),
                actions=self._pending_routine.get("actions", []),
                created_at=datetime.now().isoformat(),
                created_by=self._pending_routine.get("created_by", "voice"),
                owner_user_id=user_id
            )

            self.scheduler.register_routine(routine)

            result["response"] = f"Listo, creé la rutina '{routine.name}'"
            result["success"] = True
            result["data"]["routine_id"] = routine.routine_id

            self._reset_state()

        elif any(word in text_lower for word in self.CANCEL_WORDS):
            self._reset_state()
            result["response"] = "OK, cancelé la creación"

        else:
            # Podría ser una modificación
            result["response"] = "¿Confirmas la creación? Di sí o no."
            result["needs_confirmation"] = True

        return result

    def _reset_state(self):
        """Resetear estado de conversación"""
        self._pending_routine = None
        self._awaiting_confirmation = False
        self._conversation_context = []

    def clear_cache(self):
        """Limpiar cache de intents"""
        self._intent_cache = {}

    @property
    def is_awaiting_confirmation(self) -> bool:
        """¿Está esperando confirmación?"""
        return self._awaiting_confirmation

    @property
    def pending_routine(self) -> Optional[dict]:
        """Rutina pendiente de confirmación"""
        return self._pending_routine
