"""
Pattern Alerts Module
Genera alertas basadas en desviaciones de patrones normales.

Proporciona:
- Detección de desviaciones de rutinas
- Alertas de comportamiento anómalo
- Análisis de patrones de actividad

Uso:
    from src.alerts import PatternAlerts

    pattern_alerts = PatternAlerts(alert_manager)

    # Detectar desviación de rutina
    await pattern_alerts.check_routine_deviation(
        activity="despertar",
        expected_time_utc="07:00",
        actual_time_utc="10:30"
    )

    # Detectar comportamiento anómalo
    await pattern_alerts.check_unusual_activity(
        activity_type="home_energy_usage",
        deviation_percent=150.0
    )
"""

from datetime import datetime
from typing import Optional

from src.core.logging import get_logger
from .alert_manager import Alert, AlertManager, AlertPriority, AlertType

logger = get_logger(__name__)


class PatternAlerts:
    """Gestor de alertas de patrones anómalos"""

    # Actividades normales y sus horarios típicos (hora UTC)
    TYPICAL_ACTIVITIES = {
        "despertar": {"hour": 7, "tolerance_hours": 1},
        "desayuno": {"hour": 8, "tolerance_hours": 1},
        "salida": {"hour": 9, "tolerance_hours": 1},
        "comida": {"hour": 13, "tolerance_hours": 1},
        "regreso": {"hour": 18, "tolerance_hours": 1},
        "cena": {"hour": 20, "tolerance_hours": 1},
        "dormir": {"hour": 23, "tolerance_hours": 2},
    }

    # Umbrales de desviación para diferentes métricas
    DEVIATION_THRESHOLDS = {
        "energy_usage": 120.0,           # % de normal
        "water_usage": 150.0,             # % de normal
        "temperature": 3.0,               # grados C
        "activity_level": 180.0,          # % de normal
    }

    def __init__(self, alert_manager: AlertManager):
        """
        Inicializar PatternAlerts.

        Args:
            alert_manager: Instancia de AlertManager
        """
        self.alert_manager = alert_manager
        logger.debug("PatternAlerts initialized")

    async def check_routine_deviation(
        self,
        activity: str,
        expected_time_utc: str,
        actual_time_utc: str,
        tolerance_minutes: int = 60,
    ) -> Optional[Alert]:
        """
        Detectar desviación de rutina esperada.

        Args:
            activity: Tipo de actividad (ej: "despertar", "comida")
            expected_time_utc: Hora esperada (formato "HH:MM")
            actual_time_utc: Hora real (formato "HH:MM")
            tolerance_minutes: Minutos de tolerancia antes de alerta

        Returns:
            Alert si se creó, None si está dentro de tolerancia
        """
        try:
            expected = datetime.strptime(expected_time_utc, "%H:%M")
            actual = datetime.strptime(actual_time_utc, "%H:%M")

            # Calcular diferencia en minutos
            diff_minutes = abs((actual - expected).total_seconds() / 60)

            if diff_minutes <= tolerance_minutes:
                # Dentro de tolerancia
                return None

            # Desviación significativa
            hours_diff = diff_minutes / 60
            message = (
                f"Desviación de rutina: {activity} "
                f"{hours_diff:.1f}h fuera del horario normal"
            )

            alert = await self.alert_manager.create_alert(
                alert_type=AlertType.PATTERN,
                priority=AlertPriority.MEDIUM,
                message=message,
                details={
                    "activity": activity,
                    "expected_time": expected_time_utc,
                    "actual_time": actual_time_utc,
                    "deviation_minutes": round(diff_minutes, 1),
                },
            )

            if alert:
                logger.info(
                    f"Routine deviation: {activity} ({diff_minutes:.1f} min) (id={alert.alert_id})"
                )

            return alert

        except ValueError as e:
            logger.error(f"Invalid time format: {e}")
            return None

    async def check_unusual_activity(
        self,
        activity_type: str,
        current_value: float,
        normal_baseline: float,
        threshold_percent: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Detectar actividad anómala basada en desviación de baseline.

        Args:
            activity_type: Tipo de actividad (ej: "energy_usage", "temperature")
            current_value: Valor actual
            normal_baseline: Valor normal de baseline
            threshold_percent: Umbral de desviación % (si no especificado usa default)

        Returns:
            Alert si se creó, None si está dentro de tolerancia
        """
        # Usar umbral específico o default
        threshold = threshold_percent or self.DEVIATION_THRESHOLDS.get(activity_type)

        if threshold is None:
            logger.warning(f"No threshold defined for activity: {activity_type}")
            return None

        if normal_baseline == 0:
            logger.warning("Cannot calculate deviation with zero baseline")
            return None

        # Calcular desviación porcentual
        deviation_percent = (current_value / normal_baseline) * 100

        # Si threshold es porcentaje (>100 típicamente), comparar directamente
        if threshold > 100:
            # Ej: threshold=120 significa alertar si >120% de normal
            if deviation_percent <= threshold:
                return None
        else:
            # Ej: threshold=3.0 para temperatura significa ±3 grados
            if abs(current_value - normal_baseline) <= threshold:
                return None

        # Actividad anómala detectada
        message = (
            f"Actividad anómala: {activity_type} "
            f"en {deviation_percent:.0f}% de normal"
        )

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.PATTERN,
            priority=AlertPriority.MEDIUM,
            message=message,
            details={
                "activity_type": activity_type,
                "current_value": round(current_value, 2),
                "baseline": round(normal_baseline, 2),
                "deviation_percent": round(deviation_percent, 1),
            },
        )

        if alert:
            logger.info(
                f"Unusual activity: {activity_type} ({deviation_percent:.0f}%) (id={alert.alert_id})"
            )

        return alert

    async def check_sleep_pattern_anomaly(
        self,
        sleep_duration_hours: float,
        expected_hours: float = 8.0,
        deviation_threshold: float = 2.0,
    ) -> Optional[Alert]:
        """
        Detectar anomalías en patrón de sueño.

        Args:
            sleep_duration_hours: Horas dormidas
            expected_hours: Horas esperadas (default: 8)
            deviation_threshold: Horas de desviación para alertar

        Returns:
            Alert si se creó
        """
        deviation = abs(sleep_duration_hours - expected_hours)

        if deviation <= deviation_threshold:
            return None

        if sleep_duration_hours < expected_hours:
            message = f"Sueño insuficiente: {sleep_duration_hours:.1f}h (esperadas {expected_hours}h)"
            priority = AlertPriority.MEDIUM
        else:
            message = f"Sueño excesivo: {sleep_duration_hours:.1f}h (esperadas {expected_hours}h)"
            priority = AlertPriority.LOW

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.PATTERN,
            priority=priority,
            message=message,
            details={
                "sleep_hours": round(sleep_duration_hours, 1),
                "expected_hours": expected_hours,
                "deviation_hours": round(deviation, 1),
            },
        )

        if alert:
            logger.info(
                f"Sleep anomaly: {sleep_duration_hours:.1f}h (expected {expected_hours}h) (id={alert.alert_id})"
            )

        return alert

    async def check_activity_gap(
        self,
        zone: str,
        last_activity_hours_ago: float,
        max_gap_hours: float = 4.0,
    ) -> Optional[Alert]:
        """
        Detectar períodos sin actividad esperada en una zona.

        Args:
            zone: Zona a monitorear
            last_activity_hours_ago: Hace cuántas horas fue la última actividad
            max_gap_hours: Máximo gap permitido antes de alerta

        Returns:
            Alert si se creó
        """
        if last_activity_hours_ago <= max_gap_hours:
            return None

        message = (
            f"Sin actividad en {zone} por {last_activity_hours_ago:.1f}h "
            f"(máximo esperado: {max_gap_hours}h)"
        )

        alert = await self.alert_manager.create_alert(
            alert_type=AlertType.PATTERN,
            priority=AlertPriority.MEDIUM,
            message=message,
            details={
                "zone": zone,
                "last_activity_hours": round(last_activity_hours_ago, 1),
                "max_gap_hours": max_gap_hours,
            },
        )

        if alert:
            logger.info(
                f"Activity gap: {zone} ({last_activity_hours_ago:.1f}h) (id={alert.alert_id})"
            )

        return alert
