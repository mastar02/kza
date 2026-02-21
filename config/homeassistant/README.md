# Integración KZA con Home Assistant

## Instalación Rápida

### 1. Copiar archivos de configuración

```bash
# Desde la carpeta de KZA
cp config/homeassistant/*.yaml /path/to/homeassistant/config/
```

### 2. Editar configuration.yaml

Agregar las siguientes líneas:

```yaml
# KZA Integration
input_text: !include kza_helpers.yaml
script: !include kza_scripts.yaml
automation: !include kza_automations.yaml

# Helper para briefing (crear manualmente o agregar)
input_boolean:
  kza_briefing_delivered_today:
    name: "KZA Briefing entregado hoy"
    initial: false
    icon: mdi:weather-sunny

# Dashboard (opcional - modo YAML)
lovelace:
  mode: yaml
  dashboards:
    kza:
      mode: yaml
      filename: kza_dashboard.yaml
      title: KZA
      icon: mdi:robot
      show_in_sidebar: true
```

### 3. Reiniciar Home Assistant

```bash
# Verificar configuración
ha core check

# Reiniciar
ha core restart
```

## Sensores Disponibles

| Entidad | Descripción |
|---------|-------------|
| `sensor.kza_state` | Estado general (idle, speaking, etc.) |
| `binary_sensor.kza_listening` | ¿Está escuchando? |
| `binary_sensor.kza_speaking` | ¿Está hablando? |
| `binary_sensor.kza_conversation_active` | ¿Conversación activa? |
| `binary_sensor.kza_dnd` | No Molestar activo |
| `sensor.kza_active_timers` | Cantidad de timers activos |
| `sensor.kza_presence` | Personas en casa |
| `sensor.kza_alerts` | Alertas activas |
| `sensor.kza_echo_suppressor` | Estado del echo suppressor |
| `sensor.kza_intercom` | Zonas de intercom |
| `sensor.kza_features` | Features activos |

## Servicios Disponibles

### kza.announce
Hacer un anuncio por voz.

```yaml
service: kza.announce
data:
  message: "La cena está lista"
  zones: "cocina,comedor"  # Opcional, vacío = todas
  priority: "normal"  # normal, high, emergency
```

### kza.set_timer
Crear un timer con nombre.

```yaml
service: kza.set_timer
data:
  name: "pasta"
  minutes: 10
  zone: "cocina"  # Opcional
```

### kza.cancel_timer
Cancelar un timer.

```yaml
service: kza.cancel_timer
data:
  name: "pasta"  # "todos" para cancelar todos
```

### kza.set_dnd
Activar/desactivar No Molestar.

```yaml
service: kza.set_dnd
data:
  user_id: "gabriel"  # Opcional
  enabled: true
```

### kza.speak
Hacer que KZA diga algo.

```yaml
service: kza.speak
data:
  message: "Hola, bienvenido a casa"
  zone: "entrada"  # Opcional
```

## Scripts Incluidos

| Script | Descripción |
|--------|-------------|
| `script.kza_cancel_all_timers` | Cancela todos los timers |
| `script.kza_morning_briefing` | Inicia briefing matutino |
| `script.kza_toggle_dnd` | Toggle No Molestar |
| `script.kza_timer_pasta` | Timer 10 min para pasta |
| `script.kza_timer_arroz` | Timer 18 min para arroz |
| `script.kza_timer_huevos` | Timer 7 min para huevos |
| `script.kza_timer_te` | Timer 4 min para té |

## Automatizaciones Incluidas

| Automatización | Descripción |
|----------------|-------------|
| `kza_dnd_night_on` | Activa DND a las 23:00 |
| `kza_dnd_morning_off` | Desactiva DND a las 07:00 |
| `kza_announce_arrival` | Anuncia cuando alguien llega |
| `kza_announce_house_empty` | Anuncia cuando todos se van |
| `kza_alert_notification` | Notifica si hay >3 alertas |
| `kza_morning_briefing_auto` | Briefing automático en la mañana |

## Personalización

### Cambiar sensores en automatizaciones

Las automatizaciones usan sensores de ejemplo. Edita `kza_automations.yaml` para usar tus sensores reales:

```yaml
# Cambiar:
entity_id: binary_sensor.cocina_motion
# Por tu sensor real:
entity_id: binary_sensor.mi_sensor_de_cocina
```

### Agregar más presets de timer

En `kza_scripts.yaml`, duplica un script existente:

```yaml
kza_timer_cafe:
  alias: "KZA: Timer Café (3 min)"
  icon: mdi:coffee
  sequence:
    - service: kza.set_timer
      data:
        name: "cafe"
        minutes: 3
```

## Tarjetas Recomendadas (HACS)

Para mejor visualización, instala estas tarjetas desde HACS:

- **Mushroom Cards**: Para los entity cards modernos
- **Timer Bar Card**: Para visualizar timers
- **Mini Graph Card**: Para gráficos de latencia

## Troubleshooting

### Los sensores no aparecen

1. Verifica que KZA esté corriendo
2. Revisa los logs: `tail -f /var/log/kza/kza.log`
3. Verifica la conexión a HA

### Los servicios no funcionan

1. Reinicia HA después de agregar los servicios
2. Verifica que el dominio `kza` esté registrado
3. Revisa Developer Tools > Services

### El dashboard no carga

1. Verifica que `mode: yaml` esté configurado
2. Revisa la sintaxis YAML: `yamllint kza_dashboard.yaml`
3. Borra caché del navegador
