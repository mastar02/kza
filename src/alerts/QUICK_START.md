# Quick Start: Sistema de Alertas

Esta es una guía rápida para empezar con el sistema de alertas. Para más detalles, ver la documentación completa en `README.md`.

## 5 Minutos: Lo Básico

### 1. Ejecutar la Demo

```bash
cd /sessions/great-epic-planck/mnt/kza
python src/alerts/complete_integration_demo.py
```

Esto te mostrará:
- Alertas de seguridad (puertas)
- Alertas de dispositivos (batería)
- Alertas de patrones (energía)
- Cómo el sistema deduplica alertas
- Cómo procesa alertas

### 2. Entender la Estructura

El sistema tiene 5 componentes:

```
AlertManager (Centro)
├── SecurityAlerts (Puertas, movimiento)
├── PatternAlerts (Comportamiento anómalo)
├── DeviceAlerts (Estado dispositivos)
└── AlertScheduler (Verificaciones periódicas)
```

### 3. Conceptos Clave

- **AlertManager**: Gestor central que deduplica, notifica y registra alertas
- **Handlers**: Funciones que se ejecutan cuando se crea una alerta
- **TTS**: Notificaciones de voz para alertas críticas/altas
- **Scheduler**: Ejecuta verificaciones cada 1/5/10 minutos
- **Deduplicación**: Evita alertas duplicadas por 5 minutos (configurable)

## 15 Minutos: Integración Básica

### Paso 1: Verificar Configuración

En `config/settings.yaml`:

```yaml
alerts:
  enabled: true
  general:
    cooldown_seconds: 300      # 5 minutos
    voice_notifications: true   # Alertas de voz
  scheduler:
    enabled: true
    security_interval_seconds: 60        # Cada 1 min
    pattern_interval_seconds: 300        # Cada 5 min
    device_interval_seconds: 600         # Cada 10 min
```

### Paso 2: Copiar a main.py

En `src/main.py`, agregar en la función `main()`:

```python
# Agregar imports
from src.alerts.integration_example import (
    initialize_alert_system,
    register_alert_handlers,
    start_alert_scheduler,
)

async def main():
    # ... código existente ...

    # Inicializar sistema de alertas
    if config.get('alerts', {}).get('enabled'):
        alert_manager, scheduler, security, patterns, devices = initialize_alert_system(
            config=config['alerts'],
            tts_callback=tts_service.speak  # Tu servicio TTS
        )

        # Registrar handlers
        register_alert_handlers(alert_manager, ha_client)

        # Iniciar scheduler
        if config['alerts']['scheduler']['enabled']:
            scheduler_task = await start_alert_scheduler(scheduler)

    # ... resto de código ...

    # Al final, antes de salir:
    # await scheduler.stop()
```

### Paso 3: Listo

El sistema ya está funcionando. Las alertas se crearán automáticamente cuando:
- Se abra una puerta inesperadamente
- Se detecte movimiento en zona sin ocupación
- Batería de sensor esté baja
- Dispositivo esté offline
- Etc.

## 30 Minutos: Agregar Comandos de Voz

En tu router de comandos (`src/pipeline/voice_router.py` o similar):

```python
from src.alerts.integration_example import handle_alert_voice_commands

async def route_command(command: str) -> Optional[str]:
    # Intentar procesar como comando de alertas
    response = await handle_alert_voice_commands(command, self.alert_manager)

    if response:
        return response

    # Si no es comando de alertas, procesamiento normal
    # ... rest of routing ...
```

Ahora estos comandos funcionan:
- "¿Tengo alertas pendientes?"
- "Mostrar alertas críticas"
- "Silenciar alertas"
- "Borrar alertas"
- "Estado de alertas"

## Comandos Útiles

```bash
# Ejecutar la demo
python src/alerts/complete_integration_demo.py

# Ejecutar ejemplo básico
python src/alerts/example_usage.py

# Verificar que los archivos existen
ls -la src/alerts/alert_scheduler.py
ls -la src/alerts/integration_example.py

# Ver la configuración
grep -A 50 "^alerts:" config/settings.yaml
```

## Archivos Clave

| Archivo | Descripción | Para Qué |
|---------|------------|----------|
| `alert_scheduler.py` | Scheduler periódico | Verificaciones automáticas |
| `integration_example.py` | Funciones reutilizables | Integrar en main.py |
| `complete_integration_demo.py` | Demo completa | Entender el flujo |
| `VOICE_COMMANDS_GUIDE.md` | Guía de voz | Agregar comandos de voz |
| `README.md` | Documentación completa | Referencia |

## Ejemplos Rápidos

### Crear una alerta manualmente

```python
alert = await alert_manager.create_alert(
    alert_type=AlertType.SECURITY,
    priority=AlertPriority.CRITICAL,
    message="Puerta abierta",
    details={"zone": "entrada"}
)
```

### Procesar una alerta

```python
await alert_manager.mark_processed(alert_id)
```

### Obtener resumen

```python
summary = alert_manager.get_pending_summary()
print(f"Alertas pendientes: {summary['total_pending']}")
```

### Habilitar/deshabilitar scheduler

```python
scheduler.set_check_enabled(CheckType.SECURITY, enabled=True)
scheduler.set_check_interval(CheckType.SECURITY, 30.0)  # Cada 30s
```

## Troubleshooting Rápido

**P: El scheduler no se inicia**
R: Verificar que `alerts.scheduler.enabled: true` en YAML

**P: No recibo notificaciones de voz**
R: Asegurar que TTS callback está pasado a `initialize_alert_system()`

**P: Los comandos de voz no funcionan**
R: Verificar que `handle_alert_voice_commands()` está en el router

**P: Quiero cambiar los intervalos del scheduler**
R: Editar en `config/settings.yaml` la sección `alerts.scheduler`

## Siguiente: Personalización

Una vez que todo funciona:

1. Registrar tus propios handlers
2. Registrar verificaciones personalizadas en el scheduler
3. Agregar más comandos de voz
4. Ajustar umbrales en `config/settings.yaml`

Ver `INTEGRATION_SUMMARY.md` para casos de uso avanzados.

---

**Tiempo Total Estimado**: 30 minutos de setup + personalización
