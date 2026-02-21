# Guía: Comandos de Voz para Alertas

Este documento explica cómo integrar los comandos de voz para el sistema de alertas en tu aplicación.

## Comandos Disponibles

El sistema de alertas soporta los siguientes comandos de voz:

### 1. Resumen de Alertas Pendientes
**Frases de activación:**
- "¿Tengo alertas pendientes?"
- "¿Cuántas alertas tengo?"
- "Mostrar alertas pendientes"
- "Dame un resumen de alertas"

**Respuesta del sistema:**
```
Tienes 3 alertas pendientes. 1 crítica. 2 importantes.
```

### 2. Mostrar Alertas Críticas
**Frases de activación:**
- "Mostrar alertas críticas"
- "¿Cuáles son mis alertas críticas?"
- "Alertas críticas"

**Respuesta del sistema:**
```
Tienes 1 alerta crítica: Movimiento detectado en dormitorio sin ocupación esperada.
```

### 3. Silenciar Alertas
**Frases de activación:**
- "Silenciar alertas"
- "Pausar alertas"
- "Snooze alertas"
- "Silenciar alertas por 1 hora" (futuro)

**Respuesta del sistema:**
```
Alertas silenciadas. Los cooldowns han sido reiniciados.
```

### 4. Borrar Alertas
**Frases de activación:**
- "Borrar alertas"
- "Limpiar alertas"
- "Eliminar alertas"
- "Borra el historial de alertas"

**Respuesta del sistema:**
```
Historial de alertas eliminado.
```

### 5. Estado del Sistema de Alertas
**Frases de activación:**
- "Estado de alertas"
- "Información de alertas"
- "Estadísticas de alertas"

**Respuesta del sistema:**
```
Sistema de alertas: 25 alertas en total, 3 pendientes, 22 procesadas.
```

## Integración en el Pipeline de Comandos

### Opción 1: Router Centralizado

Si usas un router de intents, agregar en tu módulo de routing:

```python
# En src/pipeline/voice_router.py o similar

from src.alerts import handle_alert_voice_commands

class VoiceRouter:
    def __init__(self, alert_manager):
        self.alert_manager = alert_manager

    async def route_command(self, command: str, intent: str) -> Optional[str]:
        """
        Enrutar comando a handler apropiado.
        """

        # Verificar si es un comando de alertas
        if intent == "system:alerts" or self._is_alert_command(command):
            response = await handle_alert_voice_commands(
                command,
                self.alert_manager
            )

            if response:
                return response

        # ... continuar con otros intents ...

    def _is_alert_command(self, command: str) -> bool:
        """Detectar si es un comando de alertas"""
        alert_keywords = [
            "alerta", "alertas", "crítica", "críticas",
            "silenciar", "snooze", "pausar",
            "borrar", "limpiar", "eliminar",
            "estado", "información", "estadísticas"
        ]
        return any(kw in command.lower() for kw in alert_keywords)
```

### Opción 2: Integración en VoicePipeline

Si tienes un pipeline centralizado:

```python
# En src/pipeline/voice_pipeline.py

from src.alerts import handle_alert_voice_commands

class VoicePipeline:
    def __init__(self, ..., alert_manager=None):
        # ... otros componentes ...
        self.alert_manager = alert_manager

    async def process_command(self, command: str) -> str:
        """
        Procesar comando de voz completo.
        """

        # 1. Primero, intentar comandos del sistema (alertas, etc.)
        if self.alert_manager:
            alert_response = await handle_alert_voice_commands(
                command,
                self.alert_manager
            )

            if alert_response:
                return alert_response

        # 2. Luego, procesar comandos de domótica, etc.
        # ... procesamiento existente ...
```

### Opción 3: Integración en Orchestrator

Si usas un orquestador de múltiples usuarios:

```python
# En src/orchestrator/voice_orchestrator.py

from src.alerts import handle_alert_voice_commands

class VoiceOrchestrator:
    async def process_user_command(self, user_id: str, command: str) -> str:
        """
        Procesar comando de usuario (multi-user safe).
        """

        # Comandos del sistema (alertas) son globales
        alert_response = await handle_alert_voice_commands(
            command,
            self.alert_manager
        )

        if alert_response:
            return alert_response

        # ... procesamiento normal por usuario ...
```

## Configuración de Intents en LLM/Router

Si usas un clasificador de intents basado en LLM:

```python
# En tu configuración de LLM o router

INTENT_DEFINITIONS = {
    "system:alerts": {
        "description": "Comandos relacionados con alertas del sistema",
        "examples": [
            "¿Tengo alertas pendientes?",
            "Mostrar alertas críticas",
            "Silenciar alertas",
            "Estado de alertas"
        ],
        "handler": "handle_alert_voice_commands"
    },
    # ... otros intents ...
}
```

## Integración en Vectordb (Search-based)

Si usas búsqueda vectorial para detectar comandos:

```python
# Agregar estos ejemplos a tu colección de comandos en Chroma

commands_to_add = [
    {
        "command": "¿Tengo alertas pendientes?",
        "intent": "system:alerts",
        "handler": "summary_alerts",
        "description": "Mostrar resumen de alertas pendientes"
    },
    {
        "command": "Mostrar alertas críticas",
        "intent": "system:alerts",
        "handler": "critical_alerts",
        "description": "Listar alertas críticas"
    },
    {
        "command": "Silenciar alertas",
        "intent": "system:alerts",
        "handler": "snooze_alerts",
        "description": "Pausar notificaciones de alertas"
    },
    {
        "command": "Borrar alertas",
        "intent": "system:alerts",
        "handler": "clear_alerts",
        "description": "Limpiar historial de alertas"
    },
    {
        "command": "Estado de alertas",
        "intent": "system:alerts",
        "handler": "alert_status",
        "description": "Obtener información del sistema de alertas"
    }
]

# En tu inicialización:
await chroma_sync.add_commands(commands_to_add)
```

## Ejemplo Completo de Integración en Main

```python
# En src/main.py

from src.alerts import (
    AlertManager, SecurityAlerts, PatternAlerts, DeviceAlerts,
    AlertScheduler, handle_alert_voice_commands
)
from src.alerts.integration_example import (
    initialize_alert_system,
    register_alert_handlers,
    start_alert_scheduler
)

async def main():
    # ... cargar configuración y otros componentes ...

    # Inicializar sistema de alertas
    alert_manager, scheduler, security, patterns, devices = initialize_alert_system(
        config=config['alerts'],
        tts_callback=tts_service.speak  # Tu servicio de TTS
    )

    # Registrar handlers
    register_alert_handlers(alert_manager, ha_client)

    # Iniciar scheduler
    scheduler_task = await start_alert_scheduler(scheduler)

    # Integrar en pipeline
    voice_pipeline.alert_manager = alert_manager
    voice_pipeline.handle_alert_command = handle_alert_voice_commands

    # ... resto de la aplicación ...

    # Antes de salir:
    await scheduler.stop()
```

## Extensión: Agregar más Comandos

Para agregar nuevos comandos de voz para alertas:

1. Editar `src/alerts/integration_example.py` en la función `handle_alert_voice_commands()`
2. Agregar nueva rama `elif` con las palabras clave
3. Implementar la lógica
4. Registrar el comando en tu vectordb si aplica

Ejemplo:

```python
# En handle_alert_voice_commands()

elif any(x in command_lower for x in ["marcar como leído", "procesada"]):
    # Obtener la alerta más reciente
    history = alert_manager.get_history(limit=1)
    if history:
        await alert_manager.mark_processed(history[0].alert_id)
        return f"Alerta {history[0].alert_id} marcada como procesada."
    return "No hay alertas para procesar."
```

## Pruebas

Ejemplo de código para probar los comandos:

```python
import asyncio
from src.alerts import AlertManager, handle_alert_voice_commands

async def test_voice_commands():
    alert_manager = AlertManager()

    # Crear algunas alertas de prueba
    await alert_manager.create_alert(
        alert_type=AlertType.SECURITY,
        priority=AlertPriority.CRITICAL,
        message="Puerta abierta inesperadamente"
    )

    # Probar comandos
    test_commands = [
        "¿Tengo alertas pendientes?",
        "Mostrar alertas críticas",
        "Estado de alertas",
        "Silenciar alertas",
    ]

    for cmd in test_commands:
        response = await handle_alert_voice_commands(cmd, alert_manager)
        print(f"Command: {cmd}")
        print(f"Response: {response}\n")

asyncio.run(test_voice_commands())
```

## Mensajes de Respuesta Personalizados

Los mensajes de respuesta se pueden personalizar editando `handle_alert_voice_commands()`:

```python
# Cambiar el formato de respuesta
if total == 0:
    return "Ninguna alerta pendiente, todo bien."  # Antes: "No tienes alertas pendientes."

# Agregar más contexto
response = f"Tengo {total} alerta"
response += "s" if total != 1 else ""
response += " registrada"
response += "s" if total != 1 else ""
response += " en el sistema."
```

## Consideraciones de Diseño

1. **Seguridad**: Los comandos de alertas no requieren autenticación especial pero sí pueden filtrar por usuario
2. **Prioridad**: Los comandos de alertas son críticos y deberían tener alta prioridad en la cola de procesamiento
3. **Latencia**: El handler de alertas es rápido (no requiere LLM) así que puede procesarse en paralelo
4. **Context-awareness**: Los comandos pueden adaptarse al contexto del usuario (ej: "mis alertas" vs "alertas del sistema")

## Troubleshooting

**Problema**: El comando no se reconoce
- **Solución**: Verificar que el intento esté registrado en el router y que las palabras clave coincidan

**Problema**: El handler no se llama
- **Solución**: Asegurar que `alert_manager` esté inyectado correctamente en el pipeline

**Problema**: Respuesta vacía o genérica
- **Solución**: Verificar que `AlertManager` tenga alertas en el historial

## Referencias

- `src/alerts/integration_example.py` - Ejemplo completo de integración
- `src/alerts/alert_manager.py` - Documentación de AlertManager
- `config/settings.yaml` - Configuración del sistema de alertas
