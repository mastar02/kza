"""
Command Learner Module
Permite enseñar nuevos comandos a la IA de forma interactiva.

Ejemplo:
    User: "Enseñar comando"
    IA: "¿Qué frase quieres que aprenda?"
    User: "Modo película"
    IA: "¿Qué debe hacer cuando digas 'modo película'?"
    User: "Apagar luces del living y prender la tele"
    IA: "Entendido. ¿Quieres que ejecute esas acciones ahora para confirmar?"
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LearningState(Enum):
    """Estados del proceso de aprendizaje"""
    IDLE = auto()
    WAITING_PHRASE = auto()
    WAITING_ACTION = auto()
    WAITING_CONFIRMATION = auto()
    COMPLETED = auto()


@dataclass
class LearningSession:
    """Sesión activa de aprendizaje"""
    state: LearningState = LearningState.IDLE
    trigger_phrase: Optional[str] = None
    action_description: Optional[str] = None
    parsed_actions: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


@dataclass
class CustomCommand:
    """Comando personalizado aprendido"""
    id: str
    trigger_phrases: list[str]  # Múltiples formas de activar
    action_description: str
    actions: list[dict]  # Lista de acciones HA
    created_at: float
    created_by: Optional[str] = None  # Usuario que lo creó
    times_used: int = 0
    last_used: Optional[float] = None


class CommandLearner:
    """
    Sistema de aprendizaje de comandos personalizados.

    Permite a los usuarios enseñar nuevos comandos de forma conversacional.
    Los comandos se guardan en ChromaDB para búsqueda semántica.
    """

    LEARNING_TRIGGERS = [
        "enseñar comando",
        "aprender comando",
        "nuevo comando",
        "crear comando",
        "teach command",
        "learn command"
    ]

    def __init__(
        self,
        chroma_sync,
        ha_client,
        llm_reasoner,
        commands_file: str = "./data/custom_commands.json"
    ):
        self.chroma = chroma_sync
        self.ha = ha_client
        self.llm = llm_reasoner
        self.commands_file = Path(commands_file)

        self._session: Optional[LearningSession] = None
        self._commands: dict[str, CustomCommand] = {}

        self._load_commands()

    def _load_commands(self):
        """Cargar comandos personalizados desde archivo"""
        if self.commands_file.exists():
            try:
                with open(self.commands_file) as f:
                    data = json.load(f)

                for cmd_data in data.get("commands", []):
                    cmd = CustomCommand(
                        id=cmd_data["id"],
                        trigger_phrases=cmd_data["trigger_phrases"],
                        action_description=cmd_data["action_description"],
                        actions=cmd_data["actions"],
                        created_at=cmd_data["created_at"],
                        created_by=cmd_data.get("created_by"),
                        times_used=cmd_data.get("times_used", 0),
                        last_used=cmd_data.get("last_used")
                    )
                    self._commands[cmd.id] = cmd

                logger.info(f"Cargados {len(self._commands)} comandos personalizados")

            except Exception as e:
                logger.error(f"Error cargando comandos: {e}")

    def _save_commands(self):
        """Guardar comandos a archivo"""
        self.commands_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "commands": [
                {
                    "id": cmd.id,
                    "trigger_phrases": cmd.trigger_phrases,
                    "action_description": cmd.action_description,
                    "actions": cmd.actions,
                    "created_at": cmd.created_at,
                    "created_by": cmd.created_by,
                    "times_used": cmd.times_used,
                    "last_used": cmd.last_used
                }
                for cmd in self._commands.values()
            ],
            "last_updated": time.time()
        }

        with open(self.commands_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @property
    def is_learning(self) -> bool:
        """Verificar si hay una sesión de aprendizaje activa"""
        return self._session is not None and self._session.state != LearningState.IDLE

    def is_learning_trigger(self, text: str) -> bool:
        """Verificar si el texto es un trigger de aprendizaje"""
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in self.LEARNING_TRIGGERS)

    def handle(self, text: str, user_name: str = None) -> dict:
        """
        Procesar entrada durante el aprendizaje.

        Returns:
            {
                "handled": bool,
                "response": str,
                "state": LearningState,
                "command_created": CustomCommand o None
            }
        """
        result = {
            "handled": False,
            "response": "",
            "state": LearningState.IDLE,
            "command_created": None
        }

        text_lower = text.lower().strip()

        # Verificar si es trigger de inicio
        if self.is_learning_trigger(text_lower):
            return self._start_learning()

        # Si no hay sesión activa, no manejar
        if not self.is_learning:
            return result

        # Comando de cancelación
        if self._is_cancel(text_lower):
            return self._cancel_learning()

        # Procesar según estado
        state = self._session.state

        if state == LearningState.WAITING_PHRASE:
            return self._handle_phrase_input(text, user_name)

        elif state == LearningState.WAITING_ACTION:
            return self._handle_action_input(text)

        elif state == LearningState.WAITING_CONFIRMATION:
            return self._handle_confirmation(text_lower)

        return result

    def _is_cancel(self, text: str) -> bool:
        """Detectar cancelación"""
        return any(word in text for word in ["cancelar", "cancel", "salir", "no importa"])

    def _start_learning(self) -> dict:
        """Iniciar sesión de aprendizaje"""
        self._session = LearningSession(state=LearningState.WAITING_PHRASE)

        logger.info("Sesión de aprendizaje iniciada")

        return {
            "handled": True,
            "response": "¿Qué frase quieres que aprenda? Por ejemplo: 'modo película' o 'buenas noches'",
            "state": LearningState.WAITING_PHRASE,
            "command_created": None
        }

    def _handle_phrase_input(self, text: str, user_name: str = None) -> dict:
        """Procesar la frase trigger"""
        phrase = text.strip()

        if len(phrase) < 3:
            return {
                "handled": True,
                "response": "Esa frase es muy corta. Dime una frase más descriptiva.",
                "state": LearningState.WAITING_PHRASE,
                "command_created": None
            }

        # Verificar si ya existe
        existing = self._find_similar_command(phrase)
        if existing:
            return {
                "handled": True,
                "response": f"Ya existe un comando similar: '{existing.trigger_phrases[0]}'. "
                           "Dime otra frase o di 'cancelar'.",
                "state": LearningState.WAITING_PHRASE,
                "command_created": None
            }

        self._session.trigger_phrase = phrase
        self._session.state = LearningState.WAITING_ACTION

        logger.info(f"Aprendizaje: frase = '{phrase}'")

        return {
            "handled": True,
            "response": f"Perfecto. Cuando digas '{phrase}', ¿qué quieres que haga? "
                       "Describe las acciones, por ejemplo: 'apagar luces y prender la tele'",
            "state": LearningState.WAITING_ACTION,
            "command_created": None
        }

    def _handle_action_input(self, text: str) -> dict:
        """Procesar la descripción de acciones"""
        description = text.strip()

        if len(description) < 5:
            return {
                "handled": True,
                "response": "Necesito más detalle. ¿Qué dispositivos quieres controlar?",
                "state": LearningState.WAITING_ACTION,
                "command_created": None
            }

        self._session.action_description = description

        # Usar LLM para parsear las acciones a comandos de HA
        actions = self._parse_actions_with_llm(description)
        self._session.parsed_actions = actions

        if not actions:
            return {
                "handled": True,
                "response": "No pude identificar acciones específicas. "
                           "Intenta ser más específico, por ejemplo: 'apagar luz del living'",
                "state": LearningState.WAITING_ACTION,
                "command_created": None
            }

        # Formatear acciones para confirmar
        actions_text = self._format_actions_for_display(actions)
        self._session.state = LearningState.WAITING_CONFIRMATION

        logger.info(f"Aprendizaje: acciones = {actions}")

        return {
            "handled": True,
            "response": f"Entendido. Cuando digas '{self._session.trigger_phrase}', haré:\n"
                       f"{actions_text}\n\n¿Está correcto? Di 'sí' para guardar o 'no' para cancelar.",
            "state": LearningState.WAITING_CONFIRMATION,
            "command_created": None
        }

    def _handle_confirmation(self, text: str) -> dict:
        """Manejar confirmación"""
        if any(word in text for word in ["sí", "si", "yes", "correcto", "guardar", "ok"]):
            return self._save_new_command()

        elif any(word in text for word in ["no", "cancelar", "cancel"]):
            return self._cancel_learning()

        return {
            "handled": True,
            "response": "Di 'sí' para guardar el comando o 'no' para cancelar.",
            "state": LearningState.WAITING_CONFIRMATION,
            "command_created": None
        }

    def _parse_actions_with_llm(self, description: str) -> list[dict]:
        """Usar LLM para convertir descripción en acciones de HA"""
        if self.llm is None:
            return self._parse_actions_simple(description)

        # Obtener entidades disponibles
        entities = self.ha.get_domotics_entities()
        entities_summary = self._summarize_entities(entities)

        prompt = f"""Convierte esta descripción en acciones de Home Assistant.

Descripción: "{description}"

Entidades disponibles:
{entities_summary}

Responde SOLO con JSON, sin explicación:
[
  {{"domain": "light", "service": "turn_off", "entity_id": "light.living_room"}},
  {{"domain": "media_player", "service": "turn_on", "entity_id": "media_player.tv"}}
]

Si no puedes identificar acciones, responde: []
"""

        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.1)

            # Extraer JSON de la respuesta
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                actions = json.loads(json_match.group())
                return actions

        except Exception as e:
            logger.error(f"Error parseando acciones con LLM: {e}")

        return self._parse_actions_simple(description)

    def _parse_actions_simple(self, description: str) -> list[dict]:
        """Parseo simple sin LLM"""
        actions = []
        desc_lower = description.lower()

        # Patrones simples
        if "apagar" in desc_lower or "apaga" in desc_lower:
            if "luz" in desc_lower or "luces" in desc_lower:
                actions.append({
                    "domain": "light",
                    "service": "turn_off",
                    "entity_id": "light.living_room"  # Default
                })

        if "prender" in desc_lower or "prende" in desc_lower or "encender" in desc_lower:
            if "luz" in desc_lower or "luces" in desc_lower:
                actions.append({
                    "domain": "light",
                    "service": "turn_on",
                    "entity_id": "light.living_room"
                })
            if "tele" in desc_lower or "tv" in desc_lower:
                actions.append({
                    "domain": "media_player",
                    "service": "turn_on",
                    "entity_id": "media_player.tv"
                })

        return actions

    def _summarize_entities(self, entities: list[dict]) -> str:
        """Resumir entidades para el prompt"""
        by_domain = {}
        for e in entities[:50]:  # Limitar
            domain = e["entity_id"].split(".")[0]
            if domain not in by_domain:
                by_domain[domain] = []
            name = e.get("attributes", {}).get("friendly_name", e["entity_id"])
            by_domain[domain].append(f"  - {e['entity_id']}: {name}")

        lines = []
        for domain, items in by_domain.items():
            lines.append(f"{domain}:")
            lines.extend(items[:5])  # Max 5 por dominio

        return "\n".join(lines)

    def _format_actions_for_display(self, actions: list[dict]) -> str:
        """Formatear acciones para mostrar al usuario"""
        lines = []
        for action in actions:
            service = action.get("service", "?")
            entity = action.get("entity_id", "?")

            # Traducir servicios
            service_es = {
                "turn_on": "Encender",
                "turn_off": "Apagar",
                "toggle": "Alternar",
                "open_cover": "Abrir",
                "close_cover": "Cerrar"
            }.get(service, service)

            lines.append(f"  • {service_es}: {entity}")

        return "\n".join(lines)

    def _save_new_command(self) -> dict:
        """Guardar el nuevo comando"""
        cmd_id = f"custom_{int(time.time())}"

        # Generar variaciones de la frase
        variations = self._generate_phrase_variations(self._session.trigger_phrase)

        command = CustomCommand(
            id=cmd_id,
            trigger_phrases=variations,
            action_description=self._session.action_description,
            actions=self._session.parsed_actions,
            created_at=time.time()
        )

        self._commands[cmd_id] = command
        self._save_commands()

        # Agregar a ChromaDB para búsqueda semántica
        self._add_to_vector_db(command)

        self._session = None

        logger.info(f"Comando guardado: {command.trigger_phrases[0]}")

        return {
            "handled": True,
            "response": f"¡Listo! He aprendido el comando '{command.trigger_phrases[0]}'. "
                       "Puedes usarlo cuando quieras.",
            "state": LearningState.COMPLETED,
            "command_created": command
        }

    def _generate_phrase_variations(self, phrase: str) -> list[str]:
        """Generar variaciones de la frase trigger"""
        variations = [phrase]

        # Variaciones simples
        phrase_lower = phrase.lower()

        # Agregar con/sin artículos
        if not phrase_lower.startswith("el ") and not phrase_lower.startswith("la "):
            variations.append(f"el {phrase}")
            variations.append(f"la {phrase}")

        # Agregar como comando
        variations.append(f"activar {phrase}")
        variations.append(f"ejecutar {phrase}")

        return list(set(variations))[:5]  # Max 5 variaciones

    def _add_to_vector_db(self, command: CustomCommand):
        """Agregar comando a ChromaDB para búsqueda"""
        try:
            # Agregar cada variación
            for phrase in command.trigger_phrases:
                self.chroma.add_custom_command(
                    phrase=phrase,
                    command_id=command.id,
                    actions=command.actions,
                    description=command.action_description
                )
        except Exception as e:
            logger.error(f"Error agregando a vector DB: {e}")

    def _find_similar_command(self, phrase: str) -> Optional[CustomCommand]:
        """Buscar comando similar existente"""
        phrase_lower = phrase.lower()

        for cmd in self._commands.values():
            for trigger in cmd.trigger_phrases:
                if trigger.lower() == phrase_lower:
                    return cmd

        return None

    def _cancel_learning(self) -> dict:
        """Cancelar sesión de aprendizaje"""
        self._session = None

        return {
            "handled": True,
            "response": "Aprendizaje cancelado.",
            "state": LearningState.IDLE,
            "command_created": None
        }

    def get_custom_commands(self) -> list[dict]:
        """Obtener lista de comandos personalizados"""
        return [
            {
                "id": cmd.id,
                "trigger": cmd.trigger_phrases[0],
                "all_triggers": cmd.trigger_phrases,
                "description": cmd.action_description,
                "actions_count": len(cmd.actions),
                "times_used": cmd.times_used
            }
            for cmd in self._commands.values()
        ]

    def execute_custom_command(self, command_id: str) -> bool:
        """Ejecutar un comando personalizado"""
        if command_id not in self._commands:
            return False

        cmd = self._commands[command_id]

        success = True
        for action in cmd.actions:
            try:
                result = self.ha.call_service(
                    action["domain"],
                    action["service"],
                    action["entity_id"],
                    action.get("data")
                )
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Error ejecutando acción: {e}")
                success = False

        # Actualizar estadísticas
        cmd.times_used += 1
        cmd.last_used = time.time()
        self._save_commands()

        return success

    def delete_command(self, command_id: str) -> bool:
        """Eliminar un comando personalizado"""
        if command_id not in self._commands:
            return False

        del self._commands[command_id]
        self._save_commands()

        logger.info(f"Comando eliminado: {command_id}")
        return True
