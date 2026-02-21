"""
Personality Manager Module
Gestiona la personalidad, nombre y comportamiento de la IA.

Permite personalizar:
- Nombre de la IA
- Tono de respuestas (formal, casual, técnico)
- Idioma preferido
- Frases características
- Respuestas a saludos
- Reglas de comportamiento
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Personality:
    """Configuración de personalidad de la IA"""

    # Identidad
    name: str = "Jarvis"
    description: str = "Asistente de hogar inteligente"

    # Tono y estilo
    tone: str = "friendly"  # formal, friendly, casual, technical
    language: str = "es"    # es, en, etc.
    use_emojis: bool = False
    verbose: bool = False   # Respuestas largas vs concisas

    # Frases personalizadas
    greeting_responses: list[str] = field(default_factory=lambda: [
        "¡Hola! ¿En qué puedo ayudarte?",
        "¿Qué necesitas?",
        "A tu servicio."
    ])

    farewell_responses: list[str] = field(default_factory=lambda: [
        "¡Hasta luego!",
        "Que tengas buen día.",
        "Aquí estaré si me necesitas."
    ])

    confirmation_phrases: list[str] = field(default_factory=lambda: [
        "Listo",
        "Hecho",
        "Ya está"
    ])

    error_phrases: list[str] = field(default_factory=lambda: [
        "No pude hacerlo",
        "Hubo un problema",
        "Algo salió mal"
    ])

    # Comportamiento
    always_confirm_actions: bool = False
    explain_actions: bool = True
    proactive_suggestions: bool = True

    # Reglas personalizadas
    custom_rules: list[str] = field(default_factory=list)

    # Conocimiento específico
    household_info: dict = field(default_factory=dict)


TONE_TEMPLATES = {
    "formal": {
        "system_prefix": "Eres {name}, un asistente profesional y cortés.",
        "instruction_style": "Por favor, responde de manera formal y respetuosa.",
        "example_response": "Con gusto. He encendido la iluminación del salón."
    },
    "friendly": {
        "system_prefix": "Eres {name}, un asistente amigable y servicial.",
        "instruction_style": "Responde de forma natural y cercana, como un amigo.",
        "example_response": "¡Listo! Ya prendí la luz del living."
    },
    "casual": {
        "system_prefix": "Eres {name}, un asistente relajado y casual.",
        "instruction_style": "Responde de forma muy informal y relajada.",
        "example_response": "Dale, luz prendida."
    },
    "technical": {
        "system_prefix": "Eres {name}, un asistente técnico y preciso.",
        "instruction_style": "Proporciona respuestas técnicas y detalladas.",
        "example_response": "Ejecutado: light.turn_on en light.living_room. Estado confirmado: on."
    },
    "butler": {
        "system_prefix": "Eres {name}, un mayordomo británico elegante y sofisticado.",
        "instruction_style": "Responde con elegancia y formalidad británica.",
        "example_response": "Muy bien, señor. La iluminación ha sido activada en el salón principal."
    }
}


class PersonalityManager:
    """
    Gestiona la personalidad de la IA.

    Permite:
    - Cambiar nombre y tono
    - Personalizar respuestas
    - Agregar reglas de comportamiento
    - Generar system prompts personalizados
    """

    def __init__(self, config_path: str = "./data/personality.json"):
        self.config_path = Path(config_path)
        self.personality = Personality()
        self._load()

    def _load(self):
        """Cargar configuración de personalidad"""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)

                self.personality = Personality(
                    name=data.get("name", "Jarvis"),
                    description=data.get("description", "Asistente de hogar inteligente"),
                    tone=data.get("tone", "friendly"),
                    language=data.get("language", "es"),
                    use_emojis=data.get("use_emojis", False),
                    verbose=data.get("verbose", False),
                    greeting_responses=data.get("greeting_responses", self.personality.greeting_responses),
                    farewell_responses=data.get("farewell_responses", self.personality.farewell_responses),
                    confirmation_phrases=data.get("confirmation_phrases", self.personality.confirmation_phrases),
                    error_phrases=data.get("error_phrases", self.personality.error_phrases),
                    always_confirm_actions=data.get("always_confirm_actions", False),
                    explain_actions=data.get("explain_actions", True),
                    proactive_suggestions=data.get("proactive_suggestions", True),
                    custom_rules=data.get("custom_rules", []),
                    household_info=data.get("household_info", {})
                )

                logger.info(f"Personalidad cargada: {self.personality.name} ({self.personality.tone})")

            except Exception as e:
                logger.error(f"Error cargando personalidad: {e}")

    def _save(self):
        """Guardar configuración"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.personality.name,
            "description": self.personality.description,
            "tone": self.personality.tone,
            "language": self.personality.language,
            "use_emojis": self.personality.use_emojis,
            "verbose": self.personality.verbose,
            "greeting_responses": self.personality.greeting_responses,
            "farewell_responses": self.personality.farewell_responses,
            "confirmation_phrases": self.personality.confirmation_phrases,
            "error_phrases": self.personality.error_phrases,
            "always_confirm_actions": self.personality.always_confirm_actions,
            "explain_actions": self.personality.explain_actions,
            "proactive_suggestions": self.personality.proactive_suggestions,
            "custom_rules": self.personality.custom_rules,
            "household_info": self.personality.household_info
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Personalidad guardada")

    def set_name(self, name: str):
        """Cambiar nombre de la IA"""
        self.personality.name = name
        self._save()
        logger.info(f"Nombre cambiado a: {name}")

    def set_tone(self, tone: str):
        """Cambiar tono de respuestas"""
        if tone not in TONE_TEMPLATES:
            raise ValueError(f"Tono no válido. Opciones: {list(TONE_TEMPLATES.keys())}")

        self.personality.tone = tone
        self._save()
        logger.info(f"Tono cambiado a: {tone}")

    def add_custom_rule(self, rule: str):
        """Agregar regla de comportamiento personalizada"""
        self.personality.custom_rules.append(rule)
        self._save()

    def remove_custom_rule(self, index: int):
        """Eliminar regla por índice"""
        if 0 <= index < len(self.personality.custom_rules):
            removed = self.personality.custom_rules.pop(index)
            self._save()
            return removed
        return None

    def set_household_info(self, key: str, value: str):
        """Agregar información del hogar"""
        self.personality.household_info[key] = value
        self._save()

    def build_system_prompt(self, context: dict = None) -> str:
        """
        Generar system prompt basado en la personalidad.

        Args:
            context: Contexto adicional (hora, estado del hogar, etc.)

        Returns:
            System prompt personalizado
        """
        p = self.personality
        tone_template = TONE_TEMPLATES.get(p.tone, TONE_TEMPLATES["friendly"])

        # Construir prompt base
        prompt_parts = []

        # Identidad
        identity = tone_template["system_prefix"].format(name=p.name)
        prompt_parts.append(identity)
        prompt_parts.append(p.description)

        # Instrucciones de estilo
        prompt_parts.append(tone_template["instruction_style"])

        # Idioma
        if p.language == "es":
            prompt_parts.append("Responde siempre en español.")
        elif p.language == "en":
            prompt_parts.append("Always respond in English.")

        # Verbosidad
        if p.verbose:
            prompt_parts.append("Proporciona respuestas detalladas y explicativas.")
        else:
            prompt_parts.append("Sé conciso y directo en tus respuestas.")

        # Emojis
        if p.use_emojis:
            prompt_parts.append("Puedes usar emojis ocasionalmente para ser más expresivo.")
        else:
            prompt_parts.append("No uses emojis en tus respuestas.")

        # Comportamiento
        if p.always_confirm_actions:
            prompt_parts.append("Siempre confirma las acciones antes de ejecutarlas.")

        if p.explain_actions:
            prompt_parts.append("Explica brevemente lo que haces cuando ejecutas comandos.")

        if p.proactive_suggestions:
            prompt_parts.append("Puedes hacer sugerencias proactivas cuando sea útil.")

        # Reglas personalizadas
        if p.custom_rules:
            prompt_parts.append("\nReglas especiales:")
            for rule in p.custom_rules:
                prompt_parts.append(f"- {rule}")

        # Información del hogar
        if p.household_info:
            prompt_parts.append("\nInformación del hogar:")
            for key, value in p.household_info.items():
                prompt_parts.append(f"- {key}: {value}")

        # Contexto adicional
        if context:
            if "time" in context:
                prompt_parts.append(f"\nHora actual: {context['time']}")
            if "user" in context:
                prompt_parts.append(f"Usuario actual: {context['user']}")

        return "\n".join(prompt_parts)

    def get_random_response(self, response_type: str) -> str:
        """Obtener respuesta aleatoria de un tipo"""
        import random

        p = self.personality
        responses = {
            "greeting": p.greeting_responses,
            "farewell": p.farewell_responses,
            "confirmation": p.confirmation_phrases,
            "error": p.error_phrases
        }

        if response_type in responses and responses[response_type]:
            return random.choice(responses[response_type])

        return ""

    def get_greeting(self) -> str:
        """Obtener saludo"""
        return self.get_random_response("greeting")

    def get_confirmation(self) -> str:
        """Obtener confirmación"""
        return self.get_random_response("confirmation")

    def get_error_message(self) -> str:
        """Obtener mensaje de error"""
        return self.get_random_response("error")

    def get_config(self) -> dict:
        """Obtener configuración actual"""
        p = self.personality
        return {
            "name": p.name,
            "description": p.description,
            "tone": p.tone,
            "language": p.language,
            "use_emojis": p.use_emojis,
            "verbose": p.verbose,
            "custom_rules_count": len(p.custom_rules),
            "available_tones": list(TONE_TEMPLATES.keys())
        }

    def interactive_setup(self) -> dict:
        """
        Retorna preguntas para configuración interactiva.
        Para usar con el pipeline de voz.
        """
        return {
            "questions": [
                {
                    "key": "name",
                    "prompt": "¿Cómo quieres que me llame?",
                    "current": self.personality.name
                },
                {
                    "key": "tone",
                    "prompt": f"¿Qué tono prefieres? Opciones: {', '.join(TONE_TEMPLATES.keys())}",
                    "current": self.personality.tone
                },
                {
                    "key": "verbose",
                    "prompt": "¿Prefieres respuestas detalladas o concisas?",
                    "current": "detalladas" if self.personality.verbose else "concisas"
                }
            ]
        }

    def apply_setting(self, key: str, value: str) -> str:
        """Aplicar una configuración y retornar confirmación"""
        if key == "name":
            self.set_name(value)
            return f"De acuerdo, ahora me llamo {value}"

        elif key == "tone":
            try:
                self.set_tone(value.lower())
                return f"Tono cambiado a {value}"
            except ValueError as e:
                return str(e)

        elif key == "verbose":
            self.personality.verbose = value.lower() in ["detalladas", "sí", "si", "true"]
            self._save()
            return "Preferencia de verbosidad actualizada"

        elif key == "rule":
            self.add_custom_rule(value)
            return f"Regla agregada: {value}"

        elif key == "household":
            # Formato: "key=value"
            if "=" in value:
                k, v = value.split("=", 1)
                self.set_household_info(k.strip(), v.strip())
                return f"Información guardada: {k} = {v}"
            return "Formato inválido. Usa: key=value"

        return "Configuración no reconocida"
