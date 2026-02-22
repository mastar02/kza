"""
Habit Dataset Generator - Generador de Datasets de Hábitos

Puente entre los módulos de aprendizaje (PatternLearner, EventLogger,
ConversationCollector) y el NightlyTrainer.

Genera datasets JSONL enriquecidos que enseñan al modelo:
1. Preferencias del usuario ("Mastar prefiere el AC a 22°C")
2. Patrones temporales ("A las 7am, encender luces del escritorio")
3. Hábitos contextuales ("Cuando Mastar llega a casa, enciende living")
4. Preferencias de entidades ("Cuando dice 'la luz', se refiere a light.living")
5. Secuencias de acciones ("Después de encender la luz, ajusta el AC")

Flujo:
    PatternLearner ──┐
    EventLogger ──────┼──► HabitDatasetGenerator ──► JSONL ──► NightlyTrainer
    ConversationCollector ─┘
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HabitExample:
    """Un ejemplo de entrenamiento generado a partir de hábitos"""
    instruction: str      # Pregunta o comando del usuario
    input: str            # Contexto adicional (hora, usuario, etc.)
    output: str           # Respuesta esperada del asistente
    category: str         # "preference", "temporal", "sequence", "entity_alias", "contextual"
    source: str           # "pattern", "event", "conversation", "synthetic"
    confidence: float     # Confianza en el ejemplo (0-1)
    metadata: dict = field(default_factory=dict)


@dataclass
class UserProfile:
    """Perfil de hábitos de un usuario"""
    user_id: str
    user_name: str = "Usuario"
    preferred_temperatures: dict = field(default_factory=dict)  # room -> temp
    preferred_brightness: dict = field(default_factory=dict)    # room -> brightness
    wake_time: str | None = None      # "07:00"
    sleep_time: str | None = None     # "23:00"
    entity_aliases: dict = field(default_factory=dict)  # alias -> entity_id
    frequent_commands: list = field(default_factory=list)
    room_preferences: dict = field(default_factory=dict)  # time_of_day -> room


class HabitDatasetGenerator:
    """
    Genera datasets de entrenamiento a partir de hábitos aprendidos.

    Integra datos de:
    - PatternLearner: patrones repetitivos detectados
    - EventLogger: eventos de domótica con contexto temporal
    - ConversationCollector: conversaciones marcadas como buenas

    Produce:
    - JSONL con ejemplos en formato Alpaca para QLoRA
    - Perfiles de usuario enriquecidos
    - Datos sintéticos que refuerzan patrones detectados
    """

    # Templates para generación de ejemplos
    TEMPORAL_TEMPLATES = [
        ("¿Qué debería hacer a las {time}?",
         "Basándome en tus costumbres, a las {time} normalmente {action_desc}. ¿Quieres que lo haga?"),
        ("¿Qué hago normalmente a esta hora?",
         "A esta hora ({time}) sueles {action_desc}."),
        ("Es hora de {action_verb}",
         "Entendido, {action_response}"),
    ]

    PREFERENCE_TEMPLATES = [
        ("Pon el aire",
         "Ajusto el aire de {room} a {temp}°C, como te gusta."),
        ("¿A qué temperatura pongo el aire de {room}?",
         "Normalmente lo ponés a {temp}°C en {room}."),
        ("Configurá la temperatura",
         "Pongo el aire de {room} a {temp}°C. ¿Está bien?"),
    ]

    SEQUENCE_TEMPLATES = [
        ("Ya llegué a casa",
         "Bienvenido. Enciendo las luces de {room} y ajusto el aire a {temp}°C como siempre."),
        ("Es hora de dormir",
         "Apago las luces de {rooms}, bajo las persianas y ajusto el aire a {temp}°C. Buenas noches."),
        ("Buenos días",
         "Buenos días. Abro las persianas de {room} y enciendo la cafetera. Son las {time}."),
    ]

    CONTEXT_TEMPLATES = [
        ("Prende la luz",
         "Enciendo {entity_name}.", "Basado en la habitación detectada"),
        ("Apagá todo",
         "Apago las luces de {room} y el aire acondicionado.", ""),
    ]

    def __init__(
        self,
        data_dir: str = "./data/habit_training",
        pattern_learner=None,
        event_logger=None,
        conversation_collector=None,
        min_confidence: float = 0.6,
        synthetic_multiplier: int = 3,
    ):
        """
        Args:
            data_dir: Directorio para datasets generados
            pattern_learner: PatternLearner instance
            event_logger: EventLogger instance
            conversation_collector: ConversationCollector instance
            min_confidence: Confianza mínima para incluir ejemplos
            synthetic_multiplier: Variaciones sintéticas por patrón real
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.pattern_learner = pattern_learner
        self.event_logger = event_logger
        self.conversation_collector = conversation_collector
        self.min_confidence = min_confidence
        self.synthetic_multiplier = synthetic_multiplier

        # Perfiles de usuario
        self._user_profiles: dict[str, UserProfile] = {}

        # Cache de ejemplos generados
        self._examples: list[HabitExample] = []

        # Cargar perfiles existentes
        self._load_profiles()

    def _load_profiles(self):
        """Cargar perfiles de usuario persistidos"""
        profiles_file = self.data_dir / "user_profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file) as f:
                    data = json.load(f)
                for user_id, profile_data in data.items():
                    self._user_profiles[user_id] = UserProfile(
                        user_id=user_id, **profile_data
                    )
                logger.info(f"Cargados {len(self._user_profiles)} perfiles de usuario")
            except Exception as e:
                logger.warning(f"Error cargando perfiles: {e}")

    def _save_profiles(self):
        """Persistir perfiles de usuario"""
        profiles_file = self.data_dir / "user_profiles.json"
        data = {}
        for user_id, profile in self._user_profiles.items():
            data[user_id] = {
                "user_name": profile.user_name,
                "preferred_temperatures": profile.preferred_temperatures,
                "preferred_brightness": profile.preferred_brightness,
                "wake_time": profile.wake_time,
                "sleep_time": profile.sleep_time,
                "entity_aliases": profile.entity_aliases,
                "frequent_commands": profile.frequent_commands,
                "room_preferences": profile.room_preferences,
            }
        with open(profiles_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # PROFILE BUILDING - Construir perfiles a partir de datos
    # =========================================================================

    def build_user_profiles(self, events_db=None) -> dict[str, UserProfile]:
        """
        Construir perfiles de usuario analizando eventos históricos.

        Args:
            events_db: Ruta a la base SQLite de eventos (opcional)

        Returns:
            Perfiles de usuario generados
        """
        if self.event_logger:
            self._build_profiles_from_events()

        if self.pattern_learner:
            self._enrich_profiles_from_patterns()

        self._save_profiles()
        return self._user_profiles

    def _build_profiles_from_events(self):
        """Extraer preferencias de usuario del historial de eventos"""
        if not self.event_logger:
            return

        try:
            # Obtener eventos de los últimos 30 días
            events = self.event_logger.query_recent(days=30)
        except Exception:
            events = []

        # Agrupar por usuario
        user_events = defaultdict(list)
        for event in events:
            user_id = event.get("user_id")
            if user_id:
                user_events[user_id].append(event)

        for user_id, events_list in user_events.items():
            if user_id not in self._user_profiles:
                self._user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    user_name=events_list[0].get("user_name", user_id)
                )

            profile = self._user_profiles[user_id]

            # Extraer preferencias de temperatura
            climate_events = [
                e for e in events_list
                if "climate" in e.get("entity_id", "")
            ]
            self._extract_temperature_preferences(profile, climate_events)

            # Extraer horarios de despertar/dormir
            self._extract_schedule(profile, events_list)

            # Extraer comandos frecuentes
            self._extract_frequent_commands(profile, events_list)

            # Extraer preferencias de habitación por momento del día
            self._extract_room_preferences(profile, events_list)

    def _extract_temperature_preferences(
        self, profile: UserProfile, climate_events: list
    ):
        """Extraer temperaturas preferidas por habitación"""
        room_temps = defaultdict(list)

        for event in climate_events:
            entity_id = event.get("entity_id", "")
            context = event.get("context_json", {})
            if isinstance(context, str):
                try:
                    context = json.loads(context)
                except (json.JSONDecodeError, TypeError):
                    context = {}

            temp = context.get("temperature")
            if temp is not None:
                # Extraer nombre de habitación del entity_id
                room = entity_id.split(".")[-1].replace("_ac", "").replace("_clima", "")
                room_temps[room].append(float(temp))

        for room, temps in room_temps.items():
            if temps:
                # Temperatura más frecuente (moda)
                from collections import Counter
                temp_counts = Counter(round(t) for t in temps)
                profile.preferred_temperatures[room] = temp_counts.most_common(1)[0][0]

    def _extract_schedule(self, profile: UserProfile, events: list):
        """Extraer horarios de despertar y dormir"""
        morning_events = []
        night_events = []

        for event in events:
            hour = event.get("hour", 0)
            entity_id = event.get("entity_id", "")

            # Luces encendidas temprano = despertar
            if 5 <= hour <= 9 and "light" in entity_id and "turn_on" in event.get("action", ""):
                morning_events.append(hour * 60 + event.get("minute", 0))

            # Luces apagadas tarde = dormir
            if 21 <= hour <= 2 and "light" in entity_id and "turn_off" in event.get("action", ""):
                night_events.append(hour * 60 + event.get("minute", 0))

        if morning_events:
            avg_wake = sum(morning_events) / len(morning_events)
            profile.wake_time = f"{int(avg_wake // 60):02d}:{int(avg_wake % 60):02d}"

        if night_events:
            avg_sleep = sum(night_events) / len(night_events)
            profile.sleep_time = f"{int(avg_sleep // 60):02d}:{int(avg_sleep % 60):02d}"

    def _extract_frequent_commands(self, profile: UserProfile, events: list):
        """Extraer comandos más frecuentes"""
        command_counts = defaultdict(int)

        for event in events:
            trigger = event.get("trigger_phrase", "")
            if trigger:
                command_counts[trigger] += 1

        # Top 10 comandos
        sorted_commands = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)
        profile.frequent_commands = [
            {"command": cmd, "count": count}
            for cmd, count in sorted_commands[:10]
        ]

    def _extract_room_preferences(self, profile: UserProfile, events: list):
        """Extraer en qué habitación suele estar el usuario por momento del día"""
        time_room = defaultdict(lambda: defaultdict(int))

        for event in events:
            hour = event.get("hour", 0)
            entity_id = event.get("entity_id", "")

            # Determinar período del día
            if 6 <= hour < 12:
                period = "mañana"
            elif 12 <= hour < 18:
                period = "tarde"
            elif 18 <= hour < 22:
                period = "noche"
            else:
                period = "madrugada"

            # Extraer habitación del entity_id
            room = entity_id.split(".")[-1].split("_")[0]
            if room:
                time_room[period][room] += 1

        for period, rooms in time_room.items():
            if rooms:
                profile.room_preferences[period] = max(rooms, key=rooms.get)

    def _enrich_profiles_from_patterns(self):
        """Enriquecer perfiles con patrones detectados"""
        if not self.pattern_learner:
            return

        patterns = self.pattern_learner._patterns
        for pattern_id, pattern in patterns.items():
            user_id = pattern.user_id
            if not user_id:
                continue

            if user_id not in self._user_profiles:
                self._user_profiles[user_id] = UserProfile(user_id=user_id)

            profile = self._user_profiles[user_id]

            # Enriquecer con datos típicos del patrón
            if pattern.typical_data:
                entity_room = pattern.entity_id.split(".")[-1].split("_")[0]
                if "temperature" in pattern.typical_data:
                    profile.preferred_temperatures[entity_room] = pattern.typical_data["temperature"]

    # =========================================================================
    # DATASET GENERATION - Generar ejemplos de entrenamiento
    # =========================================================================

    def generate_dataset(
        self,
        days: int = 30,
        include_synthetic: bool = True
    ) -> list[HabitExample]:
        """
        Generar dataset completo de hábitos.

        Args:
            days: Días de historial a considerar
            include_synthetic: Incluir variaciones sintéticas

        Returns:
            Lista de ejemplos de entrenamiento
        """
        self._examples = []

        # 1. Ejemplos de patrones temporales
        if self.pattern_learner:
            self._generate_from_patterns()

        # 2. Ejemplos de eventos (preferencias)
        if self.event_logger:
            self._generate_from_events(days)

        # 3. Ejemplos de conversaciones de calidad
        if self.conversation_collector:
            self._generate_from_conversations()

        # 4. Ejemplos de perfiles (preferencias aprendidas)
        self._generate_from_profiles()

        # 5. Variaciones sintéticas
        if include_synthetic:
            self._generate_synthetic_variations()

        # Filtrar por confianza
        self._examples = [
            ex for ex in self._examples
            if ex.confidence >= self.min_confidence
        ]

        logger.info(
            f"Dataset generado: {len(self._examples)} ejemplos "
            f"({self._count_by_category()})"
        )

        return self._examples

    def _generate_from_patterns(self):
        """Generar ejemplos a partir de patrones detectados"""
        if not self.pattern_learner:
            return

        for pattern_id, pattern in self.pattern_learner._patterns.items():
            if pattern.dismissed:
                continue

            time_str = pattern.typical_time.strftime("%H:%M")
            entity_name = pattern.entity_id.split(".")[-1].replace("_", " ")
            action_verb = self._action_to_verb(pattern.action_type)
            action_desc = f"{action_verb} {entity_name}"
            days_desc = self._days_description(pattern.days_of_week)

            # Template 1: "¿Qué debería hacer a las X?"
            self._examples.append(HabitExample(
                instruction=f"¿Qué debería hacer a las {time_str}?",
                input=f"Día: {days_desc}",
                output=f"Basándome en tus costumbres, a las {time_str} normalmente sueles {action_desc}. ¿Quieres que lo haga?",
                category="temporal",
                source="pattern",
                confidence=pattern.confidence,
                metadata={
                    "pattern_id": pattern_id,
                    "entity_id": pattern.entity_id,
                    "user_id": pattern.user_id,
                }
            ))

            # Template 2: Comando directo
            self._examples.append(HabitExample(
                instruction=f"{action_verb.capitalize()} {entity_name}",
                input=f"Hora: {time_str}",
                output=f"Listo, {action_desc}.",
                category="temporal",
                source="pattern",
                confidence=pattern.confidence * 0.9,
                metadata={"pattern_id": pattern_id}
            ))

            # Template 3: Confirmación proactiva
            if pattern.confidence >= 0.8:
                self._examples.append(HabitExample(
                    instruction="",  # Sin instrucción (proactivo)
                    input=f"Son las {time_str}, {days_desc}. Patrón detectado: {action_desc}",
                    output=f"Son las {time_str}. ¿Quieres que {action_desc}?",
                    category="contextual",
                    source="pattern",
                    confidence=pattern.confidence * 0.8,
                    metadata={"pattern_id": pattern_id, "proactive": True}
                ))

    def _generate_from_events(self, days: int = 30):
        """Generar ejemplos a partir de eventos del historial"""
        if not self.event_logger:
            return

        try:
            events = self.event_logger.query_recent(days=days)
        except Exception:
            return

        # Detectar secuencias (acciones que siempre ocurren juntas)
        sequences = self._detect_action_sequences(events)

        for seq in sequences:
            if seq["confidence"] < self.min_confidence:
                continue

            actions_desc = " y ".join(seq["actions_desc"])

            self._examples.append(HabitExample(
                instruction=seq.get("trigger", ""),
                input=f"Hora: {seq.get('typical_time', '')}",
                output=f"Entendido. {actions_desc}.",
                category="sequence",
                source="event",
                confidence=seq["confidence"],
                metadata={"sequence": seq["actions"]}
            ))

    def _detect_action_sequences(
        self, events: list, max_gap_seconds: int = 120
    ) -> list[dict]:
        """
        Detectar secuencias de acciones que siempre ocurren juntas.

        Ejemplo: Siempre que enciende la luz del living, después ajusta el AC.
        """
        sequences = []
        sequence_counts = defaultdict(int)
        sequence_details = defaultdict(list)

        # Agrupar eventos por sesión (gap < max_gap_seconds)
        sessions = []
        current_session = []

        sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))

        for event in sorted_events:
            ts = event.get("timestamp", 0)
            if current_session and (ts - current_session[-1].get("timestamp", 0)) > max_gap_seconds:
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = []
            current_session.append(event)

        if len(current_session) >= 2:
            sessions.append(current_session)

        # Contar secuencias de pares
        for session in sessions:
            for i in range(len(session) - 1):
                action_a = f"{session[i].get('entity_id', '')}:{session[i].get('action', '')}"
                action_b = f"{session[i+1].get('entity_id', '')}:{session[i+1].get('action', '')}"
                pair = (action_a, action_b)
                sequence_counts[pair] += 1
                sequence_details[pair].append(session[i].get("hour", 0))

        # Filtrar secuencias frecuentes
        for pair, count in sequence_counts.items():
            if count < 3:
                continue

            action_a, action_b = pair
            entity_a = action_a.split(":")[0].split(".")[-1].replace("_", " ")
            entity_b = action_b.split(":")[0].split(".")[-1].replace("_", " ")
            service_a = action_a.split(":")[-1].replace("_", " ")
            service_b = action_b.split(":")[-1].replace("_", " ")

            hours = sequence_details[pair]
            avg_hour = sum(hours) / len(hours) if hours else 0

            sequences.append({
                "actions": [action_a, action_b],
                "actions_desc": [
                    f"{service_a} {entity_a}",
                    f"{service_b} {entity_b}"
                ],
                "count": count,
                "confidence": min(count / 10, 1.0),
                "typical_time": f"{int(avg_hour):02d}:00",
                "trigger": f"{service_a} {entity_a}",
            })

        return sequences

    def _generate_from_conversations(self):
        """Generar ejemplos de conversaciones marcadas como buenas"""
        if not self.conversation_collector:
            return

        for conv in self.conversation_collector._conversations:
            for turn in conv.turns:
                # Solo turnos marcados como buenos o corregidos
                if turn.quality.value not in ("good", "corrected"):
                    continue

                response = turn.correction if turn.correction else turn.assistant_response

                self._examples.append(HabitExample(
                    instruction=turn.user_input,
                    input=f"Usuario: {turn.user_name or conv.user_name or ''}",
                    output=response,
                    category="preference",
                    source="conversation",
                    confidence=0.9 if turn.quality.value == "corrected" else 0.8,
                    metadata={
                        "intent": turn.intent,
                        "quality": turn.quality.value,
                    }
                ))

    def _generate_from_profiles(self):
        """Generar ejemplos a partir de perfiles de usuario"""
        for user_id, profile in self._user_profiles.items():
            user_name = profile.user_name

            # Preferencias de temperatura
            for room, temp in profile.preferred_temperatures.items():
                room_display = room.replace("_", " ")

                self._examples.append(HabitExample(
                    instruction=f"Poné el aire de {room_display}",
                    input=f"Usuario: {user_name}",
                    output=f"Ajusto el aire de {room_display} a {temp}°C, como te gusta.",
                    category="preference",
                    source="profile",
                    confidence=0.85,
                    metadata={"user_id": user_id, "room": room, "temp": temp}
                ))

                self._examples.append(HabitExample(
                    instruction=f"¿A qué temperatura me gusta el aire en {room_display}?",
                    input=f"Usuario: {user_name}",
                    output=f"Normalmente lo ponés a {temp}°C en {room_display}.",
                    category="preference",
                    source="profile",
                    confidence=0.85,
                    metadata={"user_id": user_id}
                ))

            # Horario de despertar/dormir
            if profile.wake_time:
                self._examples.append(HabitExample(
                    instruction="Buenos días",
                    input=f"Usuario: {user_name}, Hora: {profile.wake_time}",
                    output=f"Buenos días {user_name}. Son las {profile.wake_time}. ¿Quieres que abra las persianas y encienda las luces?",
                    category="contextual",
                    source="profile",
                    confidence=0.75,
                    metadata={"user_id": user_id, "wake_time": profile.wake_time}
                ))

            if profile.sleep_time:
                self._examples.append(HabitExample(
                    instruction="Buenas noches",
                    input=f"Usuario: {user_name}, Hora: {profile.sleep_time}",
                    output=f"Buenas noches {user_name}. Apago las luces, bajo las persianas y ajusto el aire para dormir.",
                    category="contextual",
                    source="profile",
                    confidence=0.75,
                    metadata={"user_id": user_id, "sleep_time": profile.sleep_time}
                ))

            # Preferencias de habitación por momento
            for period, room in profile.room_preferences.items():
                room_display = room.replace("_", " ")
                self._examples.append(HabitExample(
                    instruction=f"Prende la luz",
                    input=f"Usuario: {user_name}, Período: {period}",
                    output=f"Enciendo la luz de {room_display}.",
                    category="contextual",
                    source="profile",
                    confidence=0.7,
                    metadata={"user_id": user_id, "period": period, "room": room}
                ))

    def _generate_synthetic_variations(self):
        """Generar variaciones sintéticas de los ejemplos reales"""
        real_examples = list(self._examples)

        # Variaciones de frases para comandos comunes
        command_variations = {
            "prendé": ["encendé", "prende", "enciende", "dale a"],
            "apagá": ["apaga", "desactivá", "desactiva"],
            "subí": ["sube", "abrí", "abre"],
            "bajá": ["baja", "cerrá", "cierra"],
            "poné": ["pon", "configura", "ajusta", "setea"],
        }

        synthetic_count = 0
        for example in real_examples:
            if synthetic_count >= len(real_examples) * self.synthetic_multiplier:
                break

            # Generar variaciones del instruction
            instruction_lower = example.instruction.lower()
            for base, variants in command_variations.items():
                if base in instruction_lower:
                    for variant in variants[:2]:  # Max 2 variaciones por base
                        new_instruction = instruction_lower.replace(base, variant)
                        if new_instruction != instruction_lower:
                            self._examples.append(HabitExample(
                                instruction=new_instruction.capitalize(),
                                input=example.input,
                                output=example.output,
                                category=example.category,
                                source="synthetic",
                                confidence=example.confidence * 0.85,
                                metadata={**example.metadata, "original": example.instruction}
                            ))
                            synthetic_count += 1

    # =========================================================================
    # EXPORT - Exportar dataset
    # =========================================================================

    def export_jsonl(
        self,
        output_path: str = None,
        max_examples: int = None
    ) -> str:
        """
        Exportar dataset en formato JSONL (Alpaca) para entrenamiento.

        Args:
            output_path: Ruta de salida (auto-generada si None)
            max_examples: Máximo de ejemplos a exportar

        Returns:
            Ruta al archivo JSONL generado
        """
        if not self._examples:
            self.generate_dataset()

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.data_dir / f"habits_{timestamp}.jsonl")

        examples_to_export = self._examples
        if max_examples and len(examples_to_export) > max_examples:
            # Priorizar por confianza
            examples_to_export = sorted(
                examples_to_export, key=lambda x: x.confidence, reverse=True
            )[:max_examples]

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples_to_export:
                entry = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Dataset exportado: {output_path} ({len(examples_to_export)} ejemplos)")
        return output_path

    def export_with_metadata(self, output_path: str = None) -> str:
        """Exportar dataset con metadata completa (para análisis)"""
        if not self._examples:
            self.generate_dataset()

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.data_dir / f"habits_full_{timestamp}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for example in self._examples:
                entry = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output,
                    "category": example.category,
                    "source": example.source,
                    "confidence": example.confidence,
                    "metadata": example.metadata,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return output_path

    # =========================================================================
    # INTEGRATION WITH NIGHTLY TRAINER
    # =========================================================================

    def prepare_for_nightly_training(self, nightly_data_dir: str = None) -> dict:
        """
        Preparar dataset de hábitos e integrarlo con los datos del NightlyTrainer.

        Returns:
            Dict con path al dataset y estadísticas
        """
        # Construir perfiles actualizados
        self.build_user_profiles()

        # Generar dataset
        examples = self.generate_dataset(include_synthetic=True)

        if not examples:
            return {
                "success": False,
                "reason": "No hay suficientes datos de hábitos",
                "examples": 0
            }

        # Exportar a directorio del nightly trainer
        if not nightly_data_dir:
            nightly_data_dir = "./data/nightly_training"

        Path(nightly_data_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path(nightly_data_dir) / f"habits_{timestamp}.jsonl")
        self.export_jsonl(output_path)

        stats = self.get_stats()

        return {
            "success": True,
            "dataset_path": output_path,
            "examples": len(examples),
            "stats": stats,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _action_to_verb(self, action_type: str) -> str:
        """Convertir tipo de acción a verbo"""
        verbs = {
            "light_on": "encender la luz de",
            "light_off": "apagar la luz de",
            "climate_set": "ajustar el aire de",
            "cover_open": "abrir las persianas de",
            "cover_close": "cerrar las persianas de",
            "switch_on": "encender",
            "switch_off": "apagar",
            "fan_on": "encender el ventilador de",
            "fan_off": "apagar el ventilador de",
            "media_play": "reproducir en",
            "media_pause": "pausar en",
        }
        return verbs.get(action_type, action_type.replace("_", " "))

    def _days_description(self, days: list[int]) -> str:
        """Descripción legible de días de la semana"""
        if len(days) == 7:
            return "todos los días"
        if days == [0, 1, 2, 3, 4]:
            return "de lunes a viernes"
        if days == [5, 6]:
            return "fines de semana"
        names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        return ", ".join(names[d] for d in days)

    def _count_by_category(self) -> str:
        """Contar ejemplos por categoría"""
        counts = defaultdict(int)
        for ex in self._examples:
            counts[ex.category] += 1
        return ", ".join(f"{cat}={count}" for cat, count in sorted(counts.items()))

    def get_stats(self) -> dict:
        """Obtener estadísticas del generador"""
        category_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for ex in self._examples:
            category_counts[ex.category] += 1
            source_counts[ex.source] += 1

        return {
            "total_examples": len(self._examples),
            "by_category": dict(category_counts),
            "by_source": dict(source_counts),
            "user_profiles": len(self._user_profiles),
            "avg_confidence": (
                sum(ex.confidence for ex in self._examples) / len(self._examples)
                if self._examples else 0
            ),
        }
