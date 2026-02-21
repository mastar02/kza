# Resumen de Integración del Sistema de Alertas

## Cambios Realizados

### 1. Nuevos Archivos Creados

#### `src/alerts/alert_scheduler.py` (390 líneas)
- Sistema de scheduler para ejecutar verificaciones periódicas
- Intervalos configurables: Seguridad (1 min), Patrones (5 min), Dispositivos (10 min)
- Ejecución asincrónica en background
- Manejo robusto de errores
- Control dinámico de verificaciones

**Características principales:**
- `AlertScheduler`: Clase principal del scheduler
- `CheckType`: Enum para tipos de verificación
- `CheckConfig`: Dataclass para configuración de chequeos

#### `src/alerts/integration_example.py` (520 líneas)
- Ejemplos de cómo integrar todo el sistema
- Funciones de inicialización reutilizables
- Manejo de comandos de voz para alertas
- Documentación de cómo agregar al main.py

**Funciones principales:**
- `initialize_alert_system()`: Inicializa AlertManager, SecurityAlerts, PatternAlerts, DeviceAlerts, AlertScheduler
- `register_alert_handlers()`: Registra handlers para procesar alertas
- `start_alert_scheduler()`: Inicia el scheduler en background
- `handle_alert_voice_commands()`: Procesa comandos de voz

**Comandos de voz soportados:**
1. "¿Tengo alertas pendientes?" → Resumen
2. "Mostrar alertas críticas" → Listar críticas
3. "Silenciar alertas" → Pausar notificaciones
4. "Borrar alertas" → Limpiar historial
5. "Estado de alertas" → Info del sistema

#### `src/alerts/complete_integration_demo.py` (480 líneas)
- Demo funcional completa con simuladores
- Simula Home Assistant Client
- Simula servicio de TTS
- Pruebas de todos los componentes

**Para ejecutar:**
```bash
python src/alerts/complete_integration_demo.py
```

#### `src/alerts/VOICE_COMMANDS_GUIDE.md` (300+ líneas)
- Guía completa de comandos de voz
- Múltiples opciones de integración
- Ejemplos de código
- Troubleshooting

### 2. Archivos Modificados

#### `config/settings.yaml`
Agregada nueva sección `alerts` con:
- Configuración general (cooldown, max_history, voice_notifications)
- Scheduler (intervalos de verificación)
- Tipos habilitados (security, pattern, device, reminder, wellness)
- Configuración específica por tipo
- Umbrales y parámetros ajustables

```yaml
alerts:
  enabled: true
  general:
    cooldown_seconds: 300
    max_history: 1000
    voice_notifications: true
  scheduler:
    enabled: true
    security_interval_seconds: 60
    pattern_interval_seconds: 300
    device_interval_seconds: 600
  # ... más config ...
```

#### `src/alerts/__init__.py`
Agregadas exportaciones para:
- `AlertScheduler`
- `CheckType`
- `CheckConfig`

#### `src/alerts/README.md`
Actualizado con:
- Nueva estructura de archivos
- Sección completa sobre AlertScheduler
- Sección sobre comandos de voz
- Configuración en settings.yaml
- Ejemplos de uso
- Próximos pasos de integración

## Archivos Existentes (Sin Cambios)

Los siguientes archivos permanecen sin cambios pero son parte integral:
- `src/alerts/alert_manager.py` - Gestor central
- `src/alerts/security_alerts.py` - Alertas de seguridad
- `src/alerts/pattern_alerts.py` - Alertas de patrones
- `src/alerts/device_alerts.py` - Alertas de dispositivos
- `src/alerts/example_usage.py` - Ejemplo básico

## Cómo Integrar en tu Aplicación

### Paso 1: Verificar Configuración

En `config/settings.yaml`, asegúrate de que la sección `alerts` esté configurada correctamente:

```yaml
alerts:
  enabled: true
  general:
    cooldown_seconds: 300
    max_history: 1000
    voice_notifications: true
  scheduler:
    enabled: true
    security_interval_seconds: 60
    pattern_interval_seconds: 300
    device_interval_seconds: 600
```

### Paso 2: Integrar en main.py

```python
# En src/main.py, agregar después de crear otros componentes:

from src.alerts.integration_example import (
    initialize_alert_system,
    register_alert_handlers,
    start_alert_scheduler,
)

async def main():
    # ... cargar configuración ...
    config = load_config(config_path)

    if config.get('alerts', {}).get('enabled'):
        # Inicializar sistema de alertas
        alert_manager, scheduler, security, patterns, devices = initialize_alert_system(
            config=config['alerts'],
            tts_callback=tts_service.speak  # Tu servicio TTS
        )

        # Registrar handlers
        register_alert_handlers(alert_manager, ha_client)

        # Iniciar scheduler
        if config['alerts']['scheduler']['enabled']:
            scheduler_task = await start_alert_scheduler(scheduler)

    # ... resto de tu aplicación ...

    # Antes de salir:
    # await scheduler.stop()
```

### Paso 3: Integrar Comandos de Voz

En tu router de comandos (ej: `src/pipeline/voice_router.py`):

```python
from src.alerts.integration_example import handle_alert_voice_commands

async def route_command(command: str) -> str:
    # Intentar procesar como comando de alertas
    response = await handle_alert_voice_commands(command, self.alert_manager)

    if response:
        return response

    # Si no es comando de alertas, procesar normalmente
    # ... rest of routing ...
```

### Paso 4: Registrar Handlers Personalizados

```python
async def my_security_handler(alert):
    """Tu lógica de seguridad personalizada"""
    logger.warning(f"Security alert: {alert.message}")
    # Aquí puedes:
    # - Enviar notificaciones
    # - Activar automatizaciones en Home Assistant
    # - Registrar en base de datos
    # - etc.

alert_manager.register_handler(AlertType.SECURITY, my_security_handler)
```

### Paso 5 (Opcional): Registrar Verificaciones Personalizadas

```python
async def check_my_devices():
    """Verificación personalizada de dispositivos"""
    # Obtener datos de Home Assistant
    devices = await ha_client.get_devices()

    # Crear alertas si es necesario
    for device in devices:
        if device['battery_low']:
            await device_alerts.check_battery_level(
                device_name=device['name'],
                battery_percent=device['battery'],
                device_type=device['type']
            )

scheduler.register_device_handler(check_my_devices)
```

## Archivos de Referencia

### Para Entender el Sistema
1. Leer: `src/alerts/README.md` - Documentación completa
2. Ejecutar: `python src/alerts/example_usage.py` - Ejemplo básico
3. Ejecutar: `python src/alerts/complete_integration_demo.py` - Demo completa

### Para Integrar Comandos de Voz
1. Leer: `src/alerts/VOICE_COMMANDS_GUIDE.md` - Guía detallada
2. Ver: `src/alerts/integration_example.py` - Función `handle_alert_voice_commands()`

### Para Integrar en main.py
1. Ver: `src/alerts/integration_example.py` - Funciones reutilizables
2. Sección: `# SECCIÓN 5: Integración en main.py` del archivo

## Flujo de Funcionamiento

```
┌─────────────────────────────────────────────────────────────┐
│ main.py                                                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Cargar config/settings.yaml                              │
│  2. initialize_alert_system()                                │
│     ├── AlertManager(config['alerts']['general'])            │
│     ├── SecurityAlerts(alert_manager)                        │
│     ├── PatternAlerts(alert_manager)                         │
│     ├── DeviceAlerts(alert_manager)                          │
│     └── AlertScheduler(alert_manager, config['alerts'])      │
│  3. register_alert_handlers()                                │
│  4. await start_alert_scheduler() → background task          │
│  5. await main_app()  // Tu aplicación                       │
│                                                               │
│ Voice Pipeline                                                │
│ ├─ async handle_alert_voice_commands(command)               │
│ └─ Devuelve respuesta al usuario                            │
│                                                               │
│ AlertScheduler (background)                                  │
│ ├─ Ejecuta cada 60s:  Security Checks                        │
│ ├─ Ejecuta cada 300s: Pattern Checks                         │
│ └─ Ejecuta cada 600s: Device Checks                          │
│                                                               │
│ AlertManager                                                  │
│ ├─ create_alert()                                            │
│ ├─ Deduplicación automática                                  │
│ ├─ Ejecuta handlers registrados                              │
│ ├─ Notificación de voz (CRITICAL/HIGH)                       │
│ └─ Registra en historial                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Ejemplos de Casos de Uso

### Caso 1: Alertar si Puerta Abierta Inesperadamente
```python
# En scheduler o comando manual:
await security.check_door_status(
    zone="entrada",
    is_open=True,
    expected_open=False
)
# Resultado: Alerta CRITICAL con TTS

# Handler ejecuta automáticamente:
# - Log de seguridad
# - Posible: activar grabación de cámara en HA
```

### Caso 2: Notificar Batería Baja de Sensor
```python
# En scheduler periódico:
devices_status = await ha_client.get_all_devices_status()
await devices.check_multiple_devices(devices_status)

# Handler ejecuta:
# - Log de mantenimiento
# - Posible: agregar a lista de tareas
```

### Caso 3: Usuario Pide Resumen de Alertas
```python
# Usuario dice: "¿Tengo alertas pendientes?"
response = await handle_alert_voice_commands(
    "¿Tengo alertas pendientes?",
    alert_manager
)
# Respuesta: "Tienes 3 alertas pendientes. 1 crítica. 2 importantes."
# TTS la reproduce
```

## Consideraciones Importantes

1. **TTS Callback**: Debe ser async o sync, el sistema lo detecta automáticamente
2. **Home Assistant Client**: Necesitas pasar tu instancia existente de HAClient
3. **Configuración**: Todos los intervalos son ajustables en config/settings.yaml
4. **Errores**: Si hay un error en un handler, el scheduler continúa ejecutándose
5. **Performance**: El scheduler usa asyncio, no bloquea la aplicación principal

## Testing

Para probar la integración:

```bash
# Ejecutar la demo completa
python src/alerts/complete_integration_demo.py

# Ejecutar ejemplo básico
python src/alerts/example_usage.py

# En tu aplicación, después de inicializar:
# 1. Crear algunas alertas manualmente
# 2. Procesar comandos de voz
# 3. Verificar que el scheduler está ejecutándose
# 4. Consultar estado del sistema
```

## Troubleshooting

### Problema: El scheduler no crea alertas
- Verificar que los handlers estén registrados correctamente
- Asegurar que los datos de Home Assistant estén disponibles
- Revisar logs para errores

### Problema: Comandos de voz no funcionan
- Verificar que `alert_manager` está inyectado en el pipeline
- Asegurar que `AlertManager` tiene alertas en el historial
- Comprobar que las palabras clave coinciden

### Problema: TTS no se escucha en alertas críticas
- Verificar que `tts_callback` está configurado en AlertManager
- Asegurar que el callback es async o sync (el sistema lo detecta)
- Revisar que la alerta tiene prioridad CRITICAL o HIGH

## Referencias Rápidas

| Archivo | Propósito | Cuándo Usar |
|---------|-----------|------------|
| `integration_example.py` | Funciones reutilizables | Para integrar en main.py |
| `complete_integration_demo.py` | Demo completa | Para entender el flujo |
| `VOICE_COMMANDS_GUIDE.md` | Integración de voz | Para agregar comandos de voz |
| `alert_scheduler.py` | Scheduler | Para verificaciones periódicas |
| `README.md` | Documentación completa | Referencia general |
| `example_usage.py` | Ejemplo básico | Punto de partida |

## Próximos Pasos Recomendados

1. ✅ Revisar `config/settings.yaml` - Verificar valores por defecto
2. ✅ Ejecutar `complete_integration_demo.py` - Entender el flujo
3. ✅ Leer `VOICE_COMMANDS_GUIDE.md` - Preparar integración de voz
4. ✅ Copiar código de `integration_example.py` a `main.py`
5. ✅ Registrar handlers personalizados según necesidades
6. ✅ Registrar verificaciones en scheduler
7. ✅ Integrar comandos de voz en router
8. ✅ Probar en aplicación real

---

**Fecha de Creación**: Febrero 3, 2025
**Versión**: 1.0
**Estado**: Listo para integración
