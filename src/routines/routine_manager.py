"""
Routine Manager Module
Creación y gestión de rutinas/automatizaciones por voz
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Tipos de triggers para automatizaciones"""
    TIME = "time"                    # "a las 7am"
    STATE = "state"                  # "cuando se abra la puerta"
    ZONE = "zone"                    # "cuando llegue a casa"
    SUN = "sun"                      # "al atardecer"
    NUMERIC_STATE = "numeric_state"  # "cuando la temperatura suba de 25"
    WEBHOOK = "webhook"
    EVENT = "event"


@dataclass
class RoutineTrigger:
    type: TriggerType
    config: dict
    description: str


@dataclass
class RoutineAction:
    entity_id: str
    domain: str
    service: str
    data: dict
    description: str


@dataclass
class Routine:
    id: str
    name: str
    description: str
    triggers: list[RoutineTrigger]
    conditions: list[dict]
    actions: list[RoutineAction]
    created_at: str
    created_by_voice: bool = True


class RoutineManager:
    """Gestión de rutinas por voz"""
    
    # Patrones para detectar intención
    CREATE_PATTERNS = [
        "crea una rutina", "nueva rutina", "crear rutina",
        "crea una automatización", "nueva automatización",
        "programa que", "quiero que cuando", "haz que cuando",
        "configura que", "automatiza", "cada vez que"
    ]
    
    DELETE_PATTERNS = [
        "elimina la rutina", "borra la rutina", "quita la rutina",
        "elimina la automatización", "cancela la rutina",
        "desactiva la rutina"
    ]
    
    LIST_PATTERNS = [
        "qué rutinas tengo", "lista de rutinas", "muestra las rutinas",
        "cuáles son mis rutinas", "mis automatizaciones"
    ]
    
    CONFIRM_PATTERNS = ["sí", "si", "confirmo", "dale", "ok", "correcto", "perfecto"]
    CANCEL_PATTERNS = ["no", "cancela", "cancelar", "olvídalo"]
    
    def __init__(self, ha_client, chroma_sync, llm_reasoner):
        self.ha_client = ha_client
        self.chroma = chroma_sync
        self.llm = llm_reasoner
        
        # Estado de conversación
        self.pending_routine: Optional[dict] = None
        self.awaiting_confirmation: bool = False
    
    def detect_intent(self, text: str) -> dict:
        """Detectar si el usuario quiere gestionar una rutina"""
        text_lower = text.lower()
        
        for pattern in self.CREATE_PATTERNS:
            if pattern in text_lower:
                return {"intent": "create", "text": text}
        
        for pattern in self.DELETE_PATTERNS:
            if pattern in text_lower:
                return {"intent": "delete", "text": text}
        
        for pattern in self.LIST_PATTERNS:
            if pattern in text_lower:
                return {"intent": "list", "text": text}
        
        return {"intent": None, "text": text}
    
    async def handle(self, text: str) -> dict:
        """
        Manejar comando de rutina
        
        Returns:
            {
                "handled": bool,
                "response": str,
                "success": bool
            }
        """
        result = {
            "handled": False,
            "response": "",
            "success": False
        }
        
        # Verificar si estamos esperando confirmación
        if self.awaiting_confirmation:
            return await self._handle_confirmation(text, result)
        
        # Detectar intención
        intent = self.detect_intent(text)
        
        if intent["intent"] == "create":
            return await self._handle_create(text, result)
        
        elif intent["intent"] == "delete":
            return await self._handle_delete(text, result)
        
        elif intent["intent"] == "list":
            return await self._handle_list(result)
        
        return result
    
    async def _handle_confirmation(self, text: str, result: dict) -> dict:
        """Manejar confirmación de rutina pendiente"""
        text_lower = text.lower()
        result["handled"] = True
        
        if any(word in text_lower for word in self.CONFIRM_PATTERNS):
            # Crear la rutina
            routine = self._build_routine(self.pending_routine)
            success, error = self._create_in_ha(routine)
            
            if success:
                # Guardar en ChromaDB
                self.chroma.save_routine(asdict(routine))
                result["response"] = f"Listo, creé la rutina '{routine.name}'"
                result["success"] = True
            else:
                result["response"] = f"No pude crear la rutina: {error}"
            
            self._reset_state()
        
        elif any(word in text_lower for word in self.CANCEL_PATTERNS):
            self._reset_state()
            result["response"] = "OK, cancelé la creación de la rutina"
        
        else:
            # No entendió, preguntar de nuevo
            result["response"] = "¿Confirmas la creación de la rutina? Di sí o no."
        
        return result
    
    async def _handle_create(self, text: str, result: dict) -> dict:
        """Manejar creación de rutina"""
        result["handled"] = True
        
        logger.info("Extrayendo rutina con LLM...")
        
        # Obtener entidades disponibles
        entities = self.ha_client.get_domotics_entities()
        
        # Extraer con LLM
        routine_data = self._extract_routine_with_llm(text, entities)
        
        if not routine_data:
            result["response"] = "No pude entender la rutina. ¿Puedes reformularla?"
            return result
        
        # Validar entidades
        is_valid, errors = self._validate_entities(routine_data, entities)
        
        if not is_valid:
            result["response"] = f"Hay problemas: {', '.join(errors)}. ¿Puedes ser más específico?"
            return result
        
        # Guardar para confirmación
        self.pending_routine = routine_data
        self.awaiting_confirmation = True
        
        # Generar resumen
        triggers_desc = ", ".join([t["description"] for t in routine_data["triggers"]])
        actions_desc = ", ".join([a["description"] for a in routine_data["actions"]])
        
        result["response"] = (
            f"Voy a crear la rutina '{routine_data['name']}': "
            f"{triggers_desc}, entonces {actions_desc}. ¿Confirmas?"
        )
        result["success"] = True
        
        return result
    
    async def _handle_delete(self, text: str, result: dict) -> dict:
        """Manejar eliminación de rutina"""
        result["handled"] = True
        
        routine = self.chroma.search_routine(text)
        
        if routine:
            # Eliminar de HA
            self.ha_client.delete_automation(routine["routine_id"])
            # Eliminar de ChromaDB
            self.chroma.delete_routine(routine["routine_id"])
            
            result["response"] = f"Eliminé la rutina '{routine['name']}'"
            result["success"] = True
        else:
            result["response"] = "No encontré esa rutina"
        
        return result
    
    async def _handle_list(self, result: dict) -> dict:
        """Listar rutinas"""
        result["handled"] = True
        
        routines = self.chroma.list_routines()
        
        if routines:
            names = [r["name"] for r in routines]
            result["response"] = f"Tienes {len(routines)} rutinas: {', '.join(names)}"
        else:
            result["response"] = "No tienes rutinas creadas por voz"
        
        result["success"] = True
        return result
    
    def _extract_routine_with_llm(self, text: str, entities: list[dict]) -> Optional[dict]:
        """Usar LLM para extraer componentes de la rutina"""
        
        # Resumen de entidades
        entities_summary = [
            {
                "entity_id": e["entity_id"],
                "name": e["attributes"].get("friendly_name", e["entity_id"]),
                "domain": e["entity_id"].split(".")[0]
            }
            for e in entities[:100]
        ]
        
        prompt = f"""Extrae información de este comando de voz para crear una rutina de domótica.

ENTIDADES DISPONIBLES:
{json.dumps(entities_summary, indent=2, ensure_ascii=False)}

COMANDO:
"{text}"

Devuelve JSON con:
{{
    "name": "nombre corto",
    "description": "descripción",
    "triggers": [
        {{
            "type": "time|state|zone|sun|numeric_state",
            "config": {{}},
            "description": "descripción del trigger"
        }}
    ],
    "conditions": [],
    "actions": [
        {{
            "entity_id": "...",
            "domain": "...",
            "service": "...",
            "data": {{}},
            "description": "..."
        }}
    ]
}}

REGLAS:
- type "time": config {{"at": "07:00:00"}}
- type "zone": config {{"entity_id": "person.X", "zone": "zone.home", "event": "enter|leave"}}
- type "sun": config {{"event": "sunrise|sunset"}}
- type "state": config {{"entity_id": "...", "to": "on|off"}}
- type "numeric_state": config {{"entity_id": "...", "above": N}}
- Para temperatura: service "set_temperature", data {{"temperature": N}}

Solo JSON:"""

        result = self.llm(prompt, max_tokens=1500, temperature=0.1)
        response = result["choices"][0]["text"]
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError as e:
            logger.warning(f"Error parseando JSON: {e}")
        
        return None
    
    def _validate_entities(self, routine_data: dict, entities: list[dict]) -> tuple[bool, list[str]]:
        """Validar que las entidades existen"""
        errors = []
        entity_ids = {e["entity_id"] for e in entities}
        
        for action in routine_data.get("actions", []):
            if action["entity_id"] not in entity_ids:
                # Intentar buscar en ChromaDB
                match = self.chroma.search_command(action["description"], threshold=0.5)
                if match:
                    action["entity_id"] = match["entity_id"]
                    action["domain"] = match["domain"]
                else:
                    errors.append(f"Entidad no encontrada: {action['entity_id']}")
        
        return len(errors) == 0, errors
    
    def _build_routine(self, data: dict) -> Routine:
        """Construir objeto Routine desde datos"""
        return Routine(
            id=f"voice_{uuid.uuid4().hex[:8]}",
            name=data["name"],
            description=data["description"],
            triggers=[
                RoutineTrigger(
                    type=TriggerType(t["type"]),
                    config=t["config"],
                    description=t["description"]
                )
                for t in data["triggers"]
            ],
            conditions=data.get("conditions", []),
            actions=[
                RoutineAction(
                    entity_id=a["entity_id"],
                    domain=a["domain"],
                    service=a["service"],
                    data=a.get("data", {}),
                    description=a["description"]
                )
                for a in data["actions"]
            ],
            created_at=datetime.now().isoformat()
        )
    
    def _create_in_ha(self, routine: Routine) -> tuple[bool, str]:
        """Crear automatización en Home Assistant"""
        
        # Convertir a formato HA
        automation_config = {
            "id": routine.id,
            "alias": routine.name,
            "description": routine.description,
            "trigger": [],
            "condition": routine.conditions,
            "action": [],
            "mode": "single"
        }
        
        # Convertir triggers
        for trigger in routine.triggers:
            ha_trigger = self._convert_trigger(trigger)
            if ha_trigger:
                automation_config["trigger"].append(ha_trigger)
        
        # Convertir acciones
        for action in routine.actions:
            ha_action = {
                "service": f"{action.domain}.{action.service}",
                "target": {"entity_id": action.entity_id}
            }
            if action.data:
                ha_action["data"] = action.data
            automation_config["action"].append(ha_action)
        
        return self.ha_client.create_automation(routine.id, automation_config)
    
    def _convert_trigger(self, trigger: RoutineTrigger) -> Optional[dict]:
        """Convertir trigger a formato Home Assistant"""
        
        if trigger.type == TriggerType.TIME:
            return {
                "platform": "time",
                "at": trigger.config.get("at", "07:00:00")
            }
        
        elif trigger.type == TriggerType.STATE:
            return {
                "platform": "state",
                "entity_id": trigger.config.get("entity_id"),
                "to": trigger.config.get("to")
            }
        
        elif trigger.type == TriggerType.ZONE:
            return {
                "platform": "zone",
                "entity_id": trigger.config.get("entity_id", "person.owner"),
                "zone": trigger.config.get("zone", "zone.home"),
                "event": trigger.config.get("event", "enter")
            }
        
        elif trigger.type == TriggerType.SUN:
            ha_trigger = {
                "platform": "sun",
                "event": trigger.config.get("event", "sunset")
            }
            if "offset" in trigger.config:
                ha_trigger["offset"] = trigger.config["offset"]
            return ha_trigger
        
        elif trigger.type == TriggerType.NUMERIC_STATE:
            ha_trigger = {
                "platform": "numeric_state",
                "entity_id": trigger.config.get("entity_id")
            }
            if "above" in trigger.config:
                ha_trigger["above"] = trigger.config["above"]
            if "below" in trigger.config:
                ha_trigger["below"] = trigger.config["below"]
            return ha_trigger
        
        return None
    
    def _reset_state(self):
        """Resetear estado de conversación"""
        self.pending_routine = None
        self.awaiting_confirmation = False
