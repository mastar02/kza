# Sistema de Alertas Proactivas KZA

Sistema central de gestión de alertas con soporte para deduplicación, notificaciones de voz y handlers por tipo de alerta.

## Estructura

```
src/alerts/
├── __init__.py                      # Exports principales
├── alert_manager.py                 # Gestor central de alertas
├── security_alerts.py               # Alertas de seguridad
├── pattern_alerts.py                # Alertas de patrones anómalos
├── device_alerts.py                 # Alertas de dispositivos
├── alert_scheduler.py               # Scheduler para verificaciones periódicas
├── integration_example.py           # Ejemplo de integración con main.py
├── complete_integration_demo.py     # Demo completa y funcional
├── VOICE_COMMANDS_GUIDE.md          # Guía de comandos de voz
├── example_usage.py                 # Ejemplo de uso básico
└── README.md                        # Este archivo
```

## Características Principales

### 1. AlertManager (Gestor Central)

El `AlertManager` es el componente central que:

- **Define prioridades**: CRITICAL, HIGH, MEDIUM, LOW
- **Define tipos**: SECURITY, PATTERN, DEVICE, REMINDER, WELLNESS
- **Gestiona deduplicación**: Evita alertas duplicadas con cooldown configurable (default: 5 min)
- **Notificaciones de voz**: Anuncias alertas críticas/altas por TTS
- **Registro de handlers**: Permite handlers por tipo de alerta
- **Historial persistente**: Mantiene historial con límite configurable
- **Resumen de pendientes**: Proporciona estado de alertas no procesadas

#### Uso Básico

```python
from src.alerts import AlertManager, AlertPriority, AlertType

# Crear manager con callback TTS
def tts_callback(message):
    print(f"[TTS] {message}")

alert_manager = AlertManager(
    cooldown_seconds=300,      # 5 minutos
    max_history=1000,
    tts_callback=tts_callback
)

# Registrar handler para alertas de seguridad
async def security_handler(alert):
    print(f"Alerta de seguridad: {alert.message}")

alert_manager.register_handler(AlertType.SECURITY, security_handler)

# Crear alerta
alert = await alert_manager.create_alert(
    alert_type=AlertType.SECURITY,
    priority=AlertPriority.CRITICAL,
    message="Puerta abierta inesperadamente",
    details={"zone": "entrada"}
)
```

### 2. SecurityAlerts (Alertas de Seguridad)

Detecta y alerta sobre eventos de seguridad:

#### Métodos Disponibles

- `check_door_status()`: Puertas abiertas/cerradas
- `check_unusual_motion()`: Movimiento en zonas sin ocupación esperada
- `check_door_sequence()`: Secuencias anormales de apertura/cierre
- `check_access_denied()`: Intentos de acceso denegado

#### Ejemplo

```python
from src.alerts import SecurityAlerts

security = SecurityAlerts(alert_manager)

# Detectar puerta abierta
await security.check_door_status(
    zone="entrada",
    is_open=True,
    expected_open=False
)

# Detectar movimiento inusual
await security.check_unusual_motion(
    zone="sala",
    motion_detected=True,
    expected_occupancy=False
)
```

### 3. PatternAlerts (Alertas de Patrones)

Detecta desviaciones de patrones normales de actividad:

#### Métodos Disponibles

- `check_routine_deviation()`: Actividades fuera de horario normal
- `check_unusual_activity()`: Deviaciones en métricas (energía, temperatura, etc.)
- `check_sleep_pattern_anomaly()`: Anomalías en patrón de sueño
- `check_activity_gap()`: Períodos sin actividad esperada

#### Ejemplo

```python
from src.alerts import PatternAlerts

patterns = PatternAlerts(alert_manager)

# Detectar desviación de rutina
await patterns.check_routine_deviation(
    activity="despertar",
    expected_time_utc="07:00",
    actual_time_utc="10:00",
    tolerance_minutes=60
)

# Detectar uso anómalo de energía
await patterns.check_unusual_activity(
    activity_type="energy_usage",
    current_value=250.0,
    normal_baseline=100.0,
    threshold_percent=120.0
)
```

### 4. DeviceAlerts (Alertas de Dispositivos)

Monitorea el estado de dispositivos inteligentes:

#### Métodos Disponibles

- `check_battery_level()`: Batería baja (umbrales por tipo)
- `check_connectivity()`: Dispositivos offline
- `check_signal_strength()`: Señal débil
- `check_firmware_update()`: Actualizaciones disponibles
- `check_response_time()`: Dispositivos respondiendo lentamente
- `check_multiple_devices()`: Verificación batch

#### Ejemplo

```python
from src.alerts import DeviceAlerts

devices = DeviceAlerts(alert_manager)

# Verificar batería baja
await devices.check_battery_level(
    device_name="sensor_temp_sala",
    battery_percent=15,
    device_type="sensor_temperature"
)

# Verificar conectividad
await devices.check_connectivity(
    device_name="luz_entrada",
    is_online=False,
    device_type="light"
)

# Verificar múltiples dispositivos
devices_status = [
    {"name": "sensor_1", "type": "sensor", "battery_percent": 50},
    {"name": "sensor_2", "type": "sensor", "is_online": False},
]
await devices.check_multiple_devices(devices_status)
```

## Deduplicación

El sistema evita alertas duplicadas automáticamente:

- **Cooldown configurable**: Por defecto 5 minutos (300 segundos)
- **Clave de deduplicación**: Basada en tipo de alerta + mensaje
- **Limpieza**: Método `clear_cooldowns()` para resetear

```python
# Las alertas idénticas en menos de 5 min retornan None
alert1 = await alert_manager.create_alert(...)  # Crea alerta
alert2 = await alert_manager.create_alert(...)  # Retorna None (deduplicada)

# Resetear cooldowns si es necesario
alert_manager.clear_cooldowns()
```

## Notificaciones de Voz

Alertas CRITICAL y HIGH generan notificaciones de voz automáticamente:

```python
# Callback TTS recibe mensaje formateado
async def tts_callback(message: str):
    await speak(message)

alert_manager = AlertManager(tts_callback=tts_callback)

# Esta alerta disparará notificación de voz
await alert_manager.create_alert(
    alert_type=AlertType.SECURITY,
    priority=AlertPriority.CRITICAL,  # Genera TTS
    message="Intruso detectado"
)
```

## Handlers

Registra funciones que se ejecutan cuando se crean alertas:

```python
# Handler async
async def security_handler(alert):
    print(f"Alerta: {alert.message}")
    await send_notification(alert)

# Handler sync
def log_alert(alert):
    logger.warning(f"Alert: {alert.message}")

alert_manager.register_handler(AlertType.SECURITY, security_handler)
alert_manager.register_handler(AlertType.DEVICE, log_alert)

# Se ejecutan automáticamente al crear alertas
alert = await alert_manager.create_alert(
    alert_type=AlertType.SECURITY,
    ...
)  # Ejecuta security_handler automáticamente
```

## Historial y Resumen

### Obtener Historial

```python
# Últimas N alertas (más recientes primero)
recent = alert_manager.get_history(limit=10)

for alert in recent:
    print(f"{alert.priority.name}: {alert.message}")
```

### Resumen de Pendientes

```python
summary = alert_manager.get_pending_summary()
# {
#   "total_pending": 5,
#   "by_type": {"security": 2, "device": 3},
#   "by_priority": {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 2}
# }
```

### Estadísticas

```python
stats = alert_manager.get_stats()
# {
#   "total_alerts": 100,
#   "pending_alerts": 5,
#   "processed_alerts": 95,
#   "dedup_keys": 20,
#   "by_type": {...},
#   "by_priority": {...}
# }
```

## Procesar Alertas

```python
# Marcar como procesada
await alert_manager.mark_processed(alert_id="abc123")

# Obtener alerta por ID
alert = alert_manager.get_alert("abc123")
```

## AlertScheduler (Verificaciones Periódicas)

El `AlertScheduler` ejecuta verificaciones automáticas de alertas en intervalos configurables:

### Características

- **Verificaciones periódicas**: Seguridad (1 min), Patrones (5 min), Dispositivos (10 min)
- **Ejecución asincrónica**: Background tasks sin bloquear la aplicación
- **Manejo de errores**: Continúa ejecutándose aunque haya errores
- **Handlers personalizables**: Registra tus propios handlers de verificación
- **Control dinámico**: Habilita/deshabilita tipos de verificación en tiempo real

### Uso Básico

```python
from src.alerts import AlertScheduler, SecurityAlerts, PatternAlerts, DeviceAlerts

# Crear scheduler
scheduler = AlertScheduler(
    alert_manager=alert_manager,
    security_interval=60.0,        # 1 minuto
    pattern_interval=300.0,        # 5 minutos
    device_interval=600.0,         # 10 minutos
)

# Registrar tus propias verificaciones
async def my_security_check():
    """Tu lógica de verificación de seguridad"""
    # ... obtener datos de Home Assistant ...
    # ... crear alertas si es necesario ...
    pass

scheduler.register_security_handler(my_security_check)

# Iniciar scheduler (ejecuta indefinidamente)
scheduler_task = asyncio.create_task(scheduler.start())

# Más tarde, detener
await scheduler.stop()
```

### Métodos Disponibles

```python
# Registrar verificaciones
scheduler.register_security_handler(async_func)
scheduler.register_pattern_handler(async_func)
scheduler.register_device_handler(async_func)

# Controlar ejecución
await scheduler.start()              # Iniciar (bloquea hasta stop)
await scheduler.stop()               # Detener

# Configuración dinámica
scheduler.set_check_enabled(CheckType.SECURITY, enabled=True)
scheduler.set_check_interval(CheckType.SECURITY, 30.0)  # 30 segundos

# Obtener estado
status = scheduler.get_status()
# {
#   "running": True,
#   "checks": {
#     "security": {"enabled": True, "interval_seconds": 60.0, ...},
#     ...
#   },
#   "registered_handlers": {"security": 2, "pattern": 1, "device": 0}
# }
```

### Ejemplo Completo con Scheduler

```python
from src.home_assistant.ha_client import HomeAssistantClient

async def main():
    # Crear componentes
    alert_manager = AlertManager(tts_callback=tts_service.speak)
    security = SecurityAlerts(alert_manager)
    scheduler = AlertScheduler(alert_manager)
    ha_client = HomeAssistantClient(url=ha_url, token=ha_token)

    # Crear handlers que usan Home Assistant
    async def check_doors():
        zones = ["entrada", "garaje", "cocina"]
        for zone in zones:
            is_open = await ha_client.get_state(f"binary_sensor.puerta_{zone}")
            await security.check_door_status(
                zone=zone,
                is_open=is_open == "on",
                expected_open=False
            )

    async def check_motion():
        zones = ["dormitorio", "sala"]
        for zone in zones:
            motion = await ha_client.get_state(f"binary_sensor.movimiento_{zone}")
            await security.check_unusual_motion(
                zone=zone,
                motion_detected=motion == "on",
                expected_occupancy=True
            )

    # Registrar handlers
    scheduler.register_security_handler(check_doors)
    scheduler.register_security_handler(check_motion)

    # Iniciar scheduler en background
    scheduler_task = asyncio.create_task(scheduler.start())

    # Tu aplicación continúa normalmente...
    try:
        await your_main_application()
    finally:
        # Limpiar antes de salir
        await scheduler.stop()
        await scheduler_task
```

## Comandos de Voz para Alertas

El sistema soporta comandos de voz para consultar y controlar alertas:

### Comandos Disponibles

- **"¿Tengo alertas pendientes?"** → Resumen de alertas
- **"Mostrar alertas críticas"** → Listar alertas críticas
- **"Silenciar alertas"** → Pausar notificaciones
- **"Borrar alertas"** → Limpiar historial
- **"Estado de alertas"** → Información del sistema

### Integración en el Router de Comandos

```python
from src.alerts.integration_example import handle_alert_voice_commands

async def route_command(command: str) -> str:
    # Intenta procesar como comando de alertas
    response = await handle_alert_voice_commands(command, alert_manager)

    if response:
        return response

    # Si no, procesar normalmente
    # ... resto del routing ...
```

Ver `VOICE_COMMANDS_GUIDE.md` para una guía completa de integración.

## Valores por Defecto

### DeviceAlerts - Umbrales de Batería

```python
{
    "sensor_default": 20,        # % mínimo
    "sensor_door": 15,
    "sensor_motion": 20,
    "sensor_temperature": 20,
    "sensor_humidity": 20,
    "remote": 10,
    "light": 15,
    "switch": 15,
}
```

### PatternAlerts - Umbrales de Desviación

```python
{
    "energy_usage": 120.0,       # % de normal
    "water_usage": 150.0,
    "temperature": 3.0,         # grados C
    "activity_level": 180.0,    # % de normal
}
```

## Testing

Se incluyen tests completos para todos los módulos:

```bash
# Ejecutar todos los tests de alertas
pytest tests/unit/alerts/ -v

# Ejecutar test específico
pytest tests/unit/alerts/test_alert_manager.py -v
```

### Cobertura de Tests

- **AlertManager**: 21 tests
- **SecurityAlerts**: 18 tests
- **PatternAlerts**: 17 tests
- **DeviceAlerts**: 18 tests

Total: 74 tests, todos pasando

## Integración con el Proyecto

El sistema de alertas está diseñado para integrarse con:

- **Home Assistant**: Estados de dispositivos y sensores
- **TTS Service**: Notificaciones de voz
- **Routines**: Detección de desviaciones de patrones
- **Logging**: Sistema de logging estructurado del proyecto

## Ejemplo Completo

Ver `example_usage.py` para un ejemplo completo de uso de todos los componentes.

## Configuración en config/settings.yaml

La sección `alerts` en el YAML permite configurar:

```yaml
alerts:
  enabled: true

  # Configuración general
  general:
    cooldown_seconds: 300        # Deduplicación
    max_history: 1000            # Historial máximo
    voice_notifications: true    # Alertas críticas por TTS

  # Scheduler
  scheduler:
    enabled: true
    security_interval_seconds: 60
    pattern_interval_seconds: 300
    device_interval_seconds: 600

  # Tipos habilitados
  enabled_types:
    security: true
    pattern: true
    device: true
    reminder: false
    wellness: false

  # Configuración específica por tipo
  security:
    zones:
      entrada:
        name: "Entrada principal"
        alert_on_open: true
        alert_on_motion: true
      # ... más zonas ...

  devices:
    battery_thresholds:
      sensor_default: 20
      sensor_motion: 20
      # ...
```

## Ejemplos de Uso

### 1. Ejemplo Básico (example_usage.py)

Demuestra el uso de todos los componentes en modo simple:

```bash
python src/alerts/example_usage.py
```

### 2. Demo Completa (complete_integration_demo.py)

Demostración completa con simuladores de Home Assistant y TTS:

```bash
python src/alerts/complete_integration_demo.py
```

### 3. Integración en main.py (integration_example.py)

Ver `integration_example.py` para cómo integrar en tu aplicación principal.

## Notas de Diseño

1. **Async-first**: Todos los métodos principales son async
2. **Thread-safe**: Usa asyncio.Lock para operaciones concurrentes
3. **Type hints**: Código con type hints completos
4. **Logging**: Integrado con el sistema de logging estructurado del proyecto
5. **Sin dependencias externas**: Solo usa librerías stdlib y del proyecto
6. **Modular**: Cada tipo de alerta es independiente y extensible
7. **Configurable**: Todos los parámetros se pueden ajustar sin cambiar código

## Próximos Pasos

Para integrar el sistema de alertas completamente:

1. **En config/settings.yaml**: Verificar y ajustar configuración de alertas
2. **En main.py**:
   - Inicializar AlertManager
   - Registrar handlers según necesidades
   - Iniciar scheduler si está habilitado
3. **En el router de comandos de voz**:
   - Integrar `handle_alert_voice_commands()`
   - Ver `VOICE_COMMANDS_GUIDE.md` para opciones de integración
4. **En handlers personalizados**:
   - Registrar acciones a tomar cuando se crean alertas
   - Ejemplo: enviar notificaciones, activar automáticas HA, logging, etc.

Ver `VOICE_COMMANDS_GUIDE.md` para detalles de integración de comandos de voz.
