"""
Morning Briefing - Briefings Proactivos Personalizados
Genera resúmenes matutinos personalizados por usuario.

"Buenos días Gabriel. Son las 7:30. Hoy tienes 3 reuniones,
la primera a las 9. El tráfico está fluido, deberías salir a las 8:15.
Hace 18 grados y lloverá por la tarde. ¿Quieres que prepare tu café?"
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import Optional, Callable, Any
from enum import Enum

logger = logging.getLogger(__name__)


class BriefingSection(Enum):
    """Secciones del briefing"""
    GREETING = "greeting"
    TIME_DATE = "time_date"
    WEATHER = "weather"
    CALENDAR = "calendar"
    TRAFFIC = "traffic"
    TASKS = "tasks"
    NEWS = "news"
    HOME_STATUS = "home_status"
    REMINDERS = "reminders"
    BIRTHDAYS = "birthdays"
    CUSTOM = "custom"


@dataclass
class BriefingConfig:
    """Configuración de briefing por usuario"""
    enabled: bool = True
    trigger_time: dtime = dtime(7, 30)
    trigger_on_presence: bool = True  # Trigger cuando detecte presencia en la mañana
    sections: list[BriefingSection] = field(default_factory=lambda: [
        BriefingSection.GREETING,
        BriefingSection.TIME_DATE,
        BriefingSection.WEATHER,
        BriefingSection.CALENDAR,
        BriefingSection.TRAFFIC,
        BriefingSection.TASKS,
        BriefingSection.HOME_STATUS
    ])
    max_duration_seconds: int = 60  # Máximo duración del briefing hablado
    include_suggestions: bool = True
    voice_style: str = "friendly"  # friendly, formal, brief


@dataclass
class UserBriefingData:
    """Datos recopilados para el briefing de un usuario"""
    user_id: str
    user_name: str
    generated_at: datetime = None

    # Clima
    weather_temp: Optional[float] = None
    weather_condition: str = ""
    weather_forecast: str = ""

    # Calendario
    events_today: list[dict] = field(default_factory=list)
    next_event: Optional[dict] = None

    # Tráfico
    commute_time_minutes: Optional[int] = None
    traffic_status: str = "normal"  # normal, moderate, heavy
    suggested_departure: Optional[str] = None

    # Tareas
    tasks_due_today: list[str] = field(default_factory=list)
    overdue_tasks: list[str] = field(default_factory=list)

    # Hogar
    home_alerts: list[str] = field(default_factory=list)
    devices_status: dict = field(default_factory=dict)

    # Personalizado
    reminders: list[str] = field(default_factory=list)
    birthdays: list[str] = field(default_factory=list)
    custom_sections: list[dict] = field(default_factory=list)


class MorningBriefing:
    """
    Generador de briefings matutinos personalizados.

    Características:
    - Configurable por usuario
    - Múltiples triggers: hora fija, detección de presencia
    - Integración con calendario, clima, tráfico
    - Generación de texto natural con LLM
    - Respuestas concisas optimizadas para voz
    """

    # Plantillas de saludo por hora
    GREETINGS = {
        (5, 9): "Buenos días",
        (9, 12): "Buen día",
        (12, 14): "Buenas tardes",
        (14, 20): "Buenas tardes",
        (20, 24): "Buenas noches",
        (0, 5): "Buenas noches"
    }

    def __init__(
        self,
        weather_provider=None,
        calendar_provider=None,
        traffic_provider=None,
        task_provider=None,
        ha_client=None,
        llm_client=None
    ):
        # Proveedores de datos
        self.weather = weather_provider
        self.calendar = calendar_provider
        self.traffic = traffic_provider
        self.tasks = task_provider
        self.ha = ha_client
        self.llm = llm_client

        # Configuraciones por usuario
        self._user_configs: dict[str, BriefingConfig] = {}

        # Historial de briefings entregados
        self._delivered_today: dict[str, datetime] = {}

        # Callbacks
        self._on_briefing_ready: Optional[Callable] = None

    def configure_user(self, user_id: str, config: BriefingConfig):
        """Configurar briefing para un usuario"""
        self._user_configs[user_id] = config
        logger.info(f"Briefing configurado para {user_id}")

    def get_user_config(self, user_id: str) -> BriefingConfig:
        """Obtener configuración de usuario (o default)"""
        return self._user_configs.get(user_id, BriefingConfig())

    async def should_deliver_briefing(self, user_id: str, trigger: str = "presence") -> bool:
        """
        ¿Debería entregar briefing ahora?

        Args:
            user_id: ID del usuario
            trigger: "presence", "time", "manual"
        """
        config = self.get_user_config(user_id)

        if not config.enabled:
            return False

        now = datetime.now()

        # Verificar si ya se entregó hoy
        if user_id in self._delivered_today:
            last_delivery = self._delivered_today[user_id]
            if last_delivery.date() == now.date():
                return False

        # Verificar hora (solo entre 5am y 11am para briefing matutino)
        if now.hour < 5 or now.hour > 11:
            return False

        if trigger == "presence" and not config.trigger_on_presence:
            return False

        if trigger == "time":
            # Verificar si es la hora configurada (±5 min)
            trigger_dt = datetime.combine(now.date(), config.trigger_time)
            diff = abs((now - trigger_dt).total_seconds())
            if diff > 300:  # 5 minutos de tolerancia
                return False

        return True

    async def generate_briefing(self, user_id: str, user_name: str = None) -> UserBriefingData:
        """Generar datos del briefing para un usuario"""
        data = UserBriefingData(
            user_id=user_id,
            user_name=user_name or user_id,
            generated_at=datetime.now()
        )

        config = self.get_user_config(user_id)

        # Recopilar datos en paralelo para velocidad
        tasks = []

        if BriefingSection.WEATHER in config.sections and self.weather:
            tasks.append(self._fetch_weather(data))

        if BriefingSection.CALENDAR in config.sections and self.calendar:
            tasks.append(self._fetch_calendar(data, user_id))

        if BriefingSection.TRAFFIC in config.sections and self.traffic:
            tasks.append(self._fetch_traffic(data, user_id))

        if BriefingSection.TASKS in config.sections and self.tasks:
            tasks.append(self._fetch_tasks(data, user_id))

        if BriefingSection.HOME_STATUS in config.sections and self.ha:
            tasks.append(self._fetch_home_status(data))

        # Ejecutar todas las tareas en paralelo
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return data

    async def _fetch_weather(self, data: UserBriefingData):
        """Obtener datos del clima"""
        try:
            if hasattr(self.weather, 'get_current'):
                weather = await self.weather.get_current()
                data.weather_temp = weather.get("temperature")
                data.weather_condition = weather.get("condition", "")
                data.weather_forecast = weather.get("forecast", "")
        except Exception as e:
            logger.warning(f"Error obteniendo clima: {e}")

    async def _fetch_calendar(self, data: UserBriefingData, user_id: str):
        """Obtener eventos del calendario"""
        try:
            if hasattr(self.calendar, 'get_today_events'):
                events = await self.calendar.get_today_events(user_id)
                data.events_today = events or []
                if events:
                    # Ordenar por hora y obtener el próximo
                    sorted_events = sorted(events, key=lambda e: e.get("start", ""))
                    data.next_event = sorted_events[0]
        except Exception as e:
            logger.warning(f"Error obteniendo calendario: {e}")

    async def _fetch_traffic(self, data: UserBriefingData, user_id: str):
        """Obtener estado del tráfico"""
        try:
            if hasattr(self.traffic, 'get_commute'):
                traffic = await self.traffic.get_commute(user_id)
                data.commute_time_minutes = traffic.get("duration_minutes")
                data.traffic_status = traffic.get("status", "normal")

                # Calcular hora de salida sugerida
                if data.next_event and data.commute_time_minutes:
                    event_time = data.next_event.get("start")
                    if event_time:
                        # Sugerir salir con margen
                        buffer_minutes = 15
                        total_minutes = data.commute_time_minutes + buffer_minutes
                        # Simplificación - en producción parsear el datetime
                        data.suggested_departure = f"{total_minutes} minutos antes"

        except Exception as e:
            logger.warning(f"Error obteniendo tráfico: {e}")

    async def _fetch_tasks(self, data: UserBriefingData, user_id: str):
        """Obtener tareas pendientes"""
        try:
            if hasattr(self.tasks, 'get_due_today'):
                data.tasks_due_today = await self.tasks.get_due_today(user_id) or []
            if hasattr(self.tasks, 'get_overdue'):
                data.overdue_tasks = await self.tasks.get_overdue(user_id) or []
        except Exception as e:
            logger.warning(f"Error obteniendo tareas: {e}")

    async def _fetch_home_status(self, data: UserBriefingData):
        """Obtener estado del hogar"""
        try:
            if hasattr(self.ha, 'get_alerts'):
                data.home_alerts = await self.ha.get_alerts() or []

            if hasattr(self.ha, 'get_summary'):
                data.devices_status = await self.ha.get_summary() or {}
        except Exception as e:
            logger.warning(f"Error obteniendo estado del hogar: {e}")

    def generate_text(self, data: UserBriefingData, style: str = "friendly") -> str:
        """
        Generar texto del briefing para TTS.

        Args:
            data: Datos recopilados
            style: "friendly", "formal", "brief"
        """
        config = self.get_user_config(data.user_id)
        sections = []

        # Saludo
        if BriefingSection.GREETING in config.sections:
            greeting = self._get_greeting()
            sections.append(f"{greeting} {data.user_name}.")

        # Hora y fecha
        if BriefingSection.TIME_DATE in config.sections:
            now = datetime.now()
            day_name = self._get_day_name(now.weekday())
            sections.append(f"Son las {now.strftime('%H:%M')} del {day_name} {now.day}.")

        # Clima
        if BriefingSection.WEATHER in config.sections and data.weather_temp:
            weather_text = f"Hace {data.weather_temp:.0f} grados"
            if data.weather_condition:
                weather_text += f", {data.weather_condition}"
            if data.weather_forecast:
                weather_text += f". {data.weather_forecast}"
            sections.append(weather_text + ".")

        # Calendario
        if BriefingSection.CALENDAR in config.sections:
            if data.events_today:
                n_events = len(data.events_today)
                if n_events == 1:
                    sections.append(f"Tienes un evento hoy.")
                else:
                    sections.append(f"Tienes {n_events} eventos hoy.")

                if data.next_event:
                    event_name = data.next_event.get("title", "evento")
                    event_time = data.next_event.get("start_time", "")
                    sections.append(f"El primero es {event_name} a las {event_time}.")
            else:
                if style != "brief":
                    sections.append("No tienes eventos programados hoy.")

        # Tráfico
        if BriefingSection.TRAFFIC in config.sections and data.commute_time_minutes:
            if data.traffic_status == "heavy":
                sections.append(
                    f"Hay tráfico pesado. El viaje tomará unos {data.commute_time_minutes} minutos."
                )
            elif data.traffic_status == "moderate":
                sections.append(
                    f"Tráfico moderado, unos {data.commute_time_minutes} minutos de viaje."
                )
            else:
                sections.append(f"El tráfico está fluido.")

            if data.suggested_departure:
                sections.append(f"Te sugiero salir {data.suggested_departure}.")

        # Tareas
        if BriefingSection.TASKS in config.sections:
            if data.overdue_tasks:
                n_overdue = len(data.overdue_tasks)
                sections.append(f"Tienes {n_overdue} tareas atrasadas.")

            if data.tasks_due_today:
                n_tasks = len(data.tasks_due_today)
                if n_tasks == 1:
                    sections.append(f"Una tarea para hoy: {data.tasks_due_today[0]}.")
                elif n_tasks <= 3:
                    tasks_str = ", ".join(data.tasks_due_today)
                    sections.append(f"Tareas para hoy: {tasks_str}.")
                else:
                    sections.append(f"Tienes {n_tasks} tareas para hoy.")

        # Estado del hogar
        if BriefingSection.HOME_STATUS in config.sections and data.home_alerts:
            if len(data.home_alerts) == 1:
                sections.append(f"Alerta del hogar: {data.home_alerts[0]}.")
            else:
                sections.append(f"Hay {len(data.home_alerts)} alertas en el hogar.")

        # Reminders y cumpleaños
        if BriefingSection.BIRTHDAYS in config.sections and data.birthdays:
            sections.append(f"Hoy es cumpleaños de {', '.join(data.birthdays)}.")

        if BriefingSection.REMINDERS in config.sections and data.reminders:
            for reminder in data.reminders[:2]:  # Máximo 2
                sections.append(f"Recordatorio: {reminder}.")

        # Generar sugerencia final si está habilitado
        if config.include_suggestions and style == "friendly":
            suggestions = self._generate_suggestions(data)
            if suggestions:
                sections.append(suggestions)

        return " ".join(sections)

    def _get_greeting(self) -> str:
        """Obtener saludo según hora del día"""
        hour = datetime.now().hour
        for (start, end), greeting in self.GREETINGS.items():
            if start <= hour < end:
                return greeting
        return "Hola"

    def _get_day_name(self, weekday: int) -> str:
        """Obtener nombre del día"""
        days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        return days[weekday]

    def _generate_suggestions(self, data: UserBriefingData) -> str:
        """Generar sugerencia proactiva"""
        suggestions = []

        # Si hay reunión pronto y tráfico pesado
        if data.next_event and data.traffic_status == "heavy":
            suggestions.append("deberías salir pronto")

        # Si hace frío
        if data.weather_temp and data.weather_temp < 15:
            suggestions.append("lleva abrigo")

        # Si hay tareas atrasadas
        if len(data.overdue_tasks) > 3:
            suggestions.append("revisa tus tareas pendientes")

        if suggestions:
            return "Te sugiero que " + " y ".join(suggestions) + "."

        return ""

    async def deliver_briefing(self, user_id: str, user_name: str = None) -> str:
        """
        Generar y entregar briefing completo.

        Returns:
            Texto del briefing para TTS
        """
        config = self.get_user_config(user_id)

        # Generar datos
        data = await self.generate_briefing(user_id, user_name)

        # Generar texto
        text = self.generate_text(data, style=config.voice_style)

        # Marcar como entregado
        self._delivered_today[user_id] = datetime.now()

        # Callback
        if self._on_briefing_ready:
            self._on_briefing_ready(user_id, text, data)

        logger.info(f"Briefing generado para {user_id}: {len(text)} caracteres")

        return text

    async def generate_with_llm(self, data: UserBriefingData) -> str:
        """Generar briefing usando LLM para texto más natural"""
        if not self.llm:
            return self.generate_text(data)

        prompt = f"""Genera un briefing matutino amigable y conciso para {data.user_name}.

Datos:
- Hora: {datetime.now().strftime('%H:%M')}
- Clima: {data.weather_temp}°, {data.weather_condition}
- Eventos hoy: {len(data.events_today)}
- Próximo evento: {data.next_event.get('title') if data.next_event else 'ninguno'}
- Tráfico: {data.traffic_status}
- Tareas pendientes: {len(data.tasks_due_today)}
- Alertas hogar: {len(data.home_alerts)}

Genera un briefing de máximo 4 oraciones, natural y útil.
No uses formato de lista, habla de forma conversacional.
Briefing:"""

        try:
            if hasattr(self.llm, 'generate'):
                result = await self.llm.generate(prompt, max_tokens=200, temperature=0.7)
                return result.get("text", self.generate_text(data))
        except Exception as e:
            logger.warning(f"Error generando briefing con LLM: {e}")

        return self.generate_text(data)

    # ==================== Callbacks ====================

    def on_briefing_ready(self, callback: Callable[[str, str, UserBriefingData], None]):
        """Registrar callback cuando el briefing está listo"""
        self._on_briefing_ready = callback

    # ==================== Estado ====================

    def get_status(self) -> dict:
        """Obtener estado del sistema de briefings"""
        return {
            "configured_users": list(self._user_configs.keys()),
            "delivered_today": {
                user_id: dt.isoformat()
                for user_id, dt in self._delivered_today.items()
            }
        }

    def reset_daily(self):
        """Resetear historial diario (llamar a medianoche)"""
        self._delivered_today = {}
        logger.info("Historial de briefings reseteado")
