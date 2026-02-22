"""
Fact Extractor Module
Usa LLM para extraer hechos importantes de conversaciones.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """Analiza esta interacción y extrae hechos importantes sobre el usuario.

Interacción:
Usuario: {user_input}
Asistente: {assistant_response}

Extrae hechos en las siguientes categorías:
- personal: Información sobre el usuario (nombre, familia, trabajo)
- preference: Preferencias (temperatura preferida, horarios, gustos)
- pattern: Patrones de comportamiento (rutinas, hábitos)
- fact: Hechos generales mencionados

Responde SOLO con un JSON array. Si no hay hechos relevantes, responde con [].
Cada hecho debe tener: content, category, confidence (0-1)

Ejemplo:
[
  {{"content": "El usuario prefiere la temperatura a 22 grados", "category": "preference", "confidence": 0.9}},
  {{"content": "Trabaja desde casa los viernes", "category": "pattern", "confidence": 0.8}}
]

JSON:"""


class FactExtractor:
    """
    Extrae hechos de conversaciones usando LLM.

    Identifica y categoriza información relevante para memoria a largo plazo:
    - Preferencias del usuario (temperatura, horarios)
    - Información personal (nombres, trabajo)
    - Patrones de comportamiento (rutinas)
    """

    def __init__(self, llm, min_confidence: float = 0.6):
        """
        Args:
            llm: Instancia de LLMReasoner para extracción
            min_confidence: Confianza mínima para guardar un hecho
        """
        self.llm = llm
        self.min_confidence = min_confidence

        # Patrones para extracción rápida sin LLM
        self._quick_patterns = {
            "preference": [
                (r"(?:me gusta|prefiero) (?:que )?(?:la |el )?(?:temperatura|aire).*?(\d+)", "temperature_preference"),
                (r"(?:pon|ponme) (?:la |el )?(?:temperatura|aire).*?(\d+)", "temperature_preference"),
                (r"(?:me despierto|me levanto) (?:a las? )?(\d{1,2}(?::\d{2})?)", "wake_time"),
                (r"(?:me acuesto|me duermo) (?:a las? )?(\d{1,2}(?::\d{2})?)", "sleep_time"),
            ],
            "pattern": [
                (r"(?:todos los |cada )(\w+) (?:hago|suelo|acostumbro)", "weekly_pattern"),
                (r"(?:cuando llego|al llegar) (?:a casa|del trabajo)", "arrival_pattern"),
                (r"(?:cuando salgo|al salir) (?:de casa|al trabajo)", "departure_pattern"),
            ]
        }

    def extract(
        self,
        user_input: str,
        assistant_response: str
    ) -> list[dict]:
        """
        Extraer hechos de una interacción.

        Args:
            user_input: Lo que dijo el usuario
            assistant_response: Respuesta del asistente

        Returns:
            Lista de hechos extraídos con formato:
            [{"content": str, "category": str, "confidence": float}]
        """
        facts = []

        # 1. Extracción rápida con patrones (sin LLM)
        quick_facts = self._quick_extract(user_input)
        facts.extend(quick_facts)

        # 2. Extracción profunda con LLM si el input es significativo
        if self._should_use_llm(user_input):
            try:
                llm_facts = self._llm_extract(user_input, assistant_response)
                facts.extend(llm_facts)
            except Exception as e:
                logger.debug(f"LLM extraction failed: {e}")

        # Filtrar por confianza mínima
        facts = [f for f in facts if f.get("confidence", 0) >= self.min_confidence]

        # Deduplicar
        seen = set()
        unique_facts = []
        for fact in facts:
            key = fact["content"].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        return unique_facts

    def _quick_extract(self, text: str) -> list[dict]:
        """Extracción rápida usando patrones regex"""
        facts = []
        text_lower = text.lower()

        for category, patterns in self._quick_patterns.items():
            for pattern, label in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    fact_content = self._format_quick_fact(label, value, text)
                    if fact_content:
                        facts.append({
                            "content": fact_content,
                            "category": category,
                            "confidence": 0.85,
                            "extraction_method": "pattern"
                        })

        return facts

    def _format_quick_fact(self, label: str, value: str, original: str) -> str | None:
        """Formatear un hecho extraído por patrón"""
        templates = {
            "temperature_preference": f"El usuario prefiere la temperatura a {value} grados",
            "wake_time": f"El usuario se despierta a las {value}",
            "sleep_time": f"El usuario se acuesta a las {value}",
            "weekly_pattern": f"El usuario tiene actividad los {value}",
            "arrival_pattern": "El usuario tiene una rutina de llegada a casa",
            "departure_pattern": "El usuario tiene una rutina de salida"
        }
        return templates.get(label)

    def _should_use_llm(self, text: str) -> bool:
        """Determinar si vale la pena usar LLM para extraer"""
        # Usa LLM si el texto tiene información personal/contextual
        indicators = [
            "me llamo", "mi nombre", "trabajo", "vivo",
            "mi familia", "mi esposa", "mi esposo", "mis hijos",
            "me gusta", "prefiero", "siempre", "nunca",
            "todos los días", "normalmente", "usualmente"
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in indicators)

    def _llm_extract(
        self,
        user_input: str,
        assistant_response: str
    ) -> list[dict]:
        """Extracción usando LLM"""
        prompt = EXTRACTION_PROMPT.format(
            user_input=user_input,
            assistant_response=assistant_response
        )

        response = self.llm.generate(
            prompt,
            max_tokens=500,
            temperature=0.1  # Baja temperatura para salida estructurada
        )

        # Parsear JSON de la respuesta
        try:
            # Limpiar respuesta
            response = response.strip()

            # Encontrar JSON array en la respuesta
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())

                # Validar estructura
                valid_facts = []
                for fact in facts:
                    if isinstance(fact, dict) and "content" in fact and "category" in fact:
                        fact["extraction_method"] = "llm"
                        fact["confidence"] = fact.get("confidence", 0.7)
                        valid_facts.append(fact)

                return valid_facts

        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse LLM response as JSON: {e}")

        return []

    def extract_from_command(self, command: str, entity_used: str) -> list[dict]:
        """
        Extraer patrones de comandos de domótica.

        Args:
            command: Comando del usuario
            entity_used: Entidad de HA que se usó

        Returns:
            Lista de patrones detectados
        """
        facts = []

        # Detectar patrones de uso
        from datetime import datetime
        hour = datetime.now().hour

        # Inferir preferencias por horario
        if "luz" in command.lower() or "light" in entity_used:
            if 6 <= hour <= 8:
                facts.append({
                    "content": f"El usuario prende luces por la mañana ({hour}:00)",
                    "category": "pattern",
                    "confidence": 0.6
                })
            elif 18 <= hour <= 22:
                facts.append({
                    "content": f"El usuario prende luces por la noche ({hour}:00)",
                    "category": "pattern",
                    "confidence": 0.6
                })

        if "clima" in entity_used or "climate" in entity_used:
            # Extraer temperatura si está en el comando
            temp_match = re.search(r'(\d+)\s*(?:grados|°)', command.lower())
            if temp_match:
                temp = temp_match.group(1)
                facts.append({
                    "content": f"El usuario configuró el aire a {temp} grados",
                    "category": "preference",
                    "confidence": 0.75
                })

        return facts
