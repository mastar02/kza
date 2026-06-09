# Análisis de Integración: OpenClaw + KZA

## 📊 Resumen Ejecutivo

**OpenClaw** es un asistente personal de IA que corre en dispositivos propios, con capacidad multi-canal (WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Teams, etc.) y control por voz.

**Objetivo:** Integrar las capacidades de OpenClaw en KZA para tener una IA local que se comunique de forma natural a través de múltiples canales de mensajería.

---

## 🏗️ Arquitectura de OpenClaw

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CANALES DE ENTRADA                               │
├─────────────────────────────────────────────────────────────────────────┤
│ WhatsApp │ Telegram │ Slack │ Discord │ Signal │ iMessage │ Teams │ IRC │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          GATEWAY (Node.js)                               │
│                     ws://127.0.0.1:18789                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ • WebSocket Control Plane                                                │
│ • Session Management                                                     │
│ • Channel Routing                                                        │
│ • Security (DM pairing, allowlists)                                      │
│ • Hooks & Events                                                         │
│ • Cron Jobs                                                              │
│ • Webhooks                                                               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT (Pi Runtime)                               │
├─────────────────────────────────────────────────────────────────────────┤
│ • LLM Interaction (Anthropic/OpenAI)                                     │
│ • Tool Streaming                                                         │
│ • Session Context                                                        │
│ • Skills System                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            TOOLS                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ Browser │ Canvas │ Nodes │ Cron │ Sessions │ Discord/Slack Actions      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura de Código de OpenClaw

### Core (src/)
| Directorio | Propósito | Archivos Clave |
|------------|-----------|----------------|
| `agents/` | Runtime de agentes LLM | ~100+ archivos |
| `gateway/` | WebSocket control plane | server.impl.ts, server-http.ts |
| `channels/` | Abstracción de canales | registry.ts, dock.ts |
| `telegram/` | Bot de Telegram (grammY) | bot.ts, send.ts |
| `discord/` | Bot de Discord (discord.js) | Integración directa |
| `slack/` | Bot de Slack (Bolt) | Socket Mode |
| `signal/` | Signal (signal-cli) | REST API |
| `imessage/` | iMessage (imsg) | macOS only |
| `whatsapp/` | Normalización | Extensions tiene la implementación |
| `tts/` | Text-to-Speech | ElevenLabs integration |
| `browser/` | Control de navegador | CDP automation |
| `hooks/` | Sistema de hooks | Eventos personalizables |
| `cron/` | Tareas programadas | Wakeups automáticos |

### Extensions (extensions/)
| Extension | Descripción | Tecnología |
|-----------|-------------|------------|
| `whatsapp/` | WhatsApp Web | Baileys library |
| `msteams/` | Microsoft Teams | Bot Framework |
| `matrix/` | Matrix Protocol | Matrix SDK |
| `googlechat/` | Google Chat | Chat API |
| `bluebubbles/` | iMessage (recomendado) | BlueBubbles server |
| `voice-call/` | Llamadas de voz | VoIP integration |
| `zalo/` | Zalo messenger | Custom API |

### Skills (skills/)
```
skills/
├── 1password/      # Integración con 1Password
├── apple-notes/    # Notas de Apple
├── apple-reminders/# Recordatorios
├── discord/        # Comandos de Discord
├── github/         # GitHub CLI
├── notion/         # Notion API
└── ...             # 50+ skills disponibles
```

---

## 🔌 Implementación de Canales

### WhatsApp (via Baileys)
```typescript
// extensions/whatsapp/src/channel.ts
export const whatsappPlugin: ChannelPlugin<ResolvedWhatsAppAccount> = {
  id: "whatsapp",
  capabilities: {
    chatTypes: ["direct", "group"],
    polls: true,
    reactions: true,
    media: true,
  },
  outbound: {
    deliveryMode: "gateway",
    textChunkLimit: 4000,
    pollMaxOptions: 12,
  },
  // ... configuración de seguridad, routing, etc.
};
```

### Telegram (via grammY)
```typescript
// src/telegram/bot.ts
- Maneja mensajes entrantes
- Soporta grupos con mention gating
- Media handling (fotos, audio, video)
- Inline buttons para interacción
```

### Sistema de Routing
```typescript
// src/routing/
- Channel routing por cuenta
- DM policy (pairing/open)
- Group allowlists
- Mention gating para grupos
```

---

## 🎯 Plan de Integración con KZA

### Opción A: Bridge Completo (Recomendado)
Crear un bridge Python que conecte con OpenClaw Gateway.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KZA (Python)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Voice Pipeline │ Home Assistant │ LLM Local │ Echo Suppressor           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ WebSocket
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OpenClaw Gateway (Node.js)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ WhatsApp │ Telegram │ Slack │ Discord │ Signal │ iMessage │ Teams       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ventajas:**
- Usa OpenClaw tal cual (mantenido por la comunidad)
- Actualizaciones automáticas de canales
- Menos código que mantener

**Desventajas:**
- Dependencia de Node.js
- Latencia adicional (WebSocket hop)

### Opción B: Port Nativo a Python
Re-implementar los canales críticos en Python.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KZA (Python)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Voice │ HA │ LLM │ WhatsApp │ Telegram │ Discord │ Signal               │
│                    (neonize)   (aiogram)  (py-cord) (signal-cli)        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ventajas:**
- Todo en Python
- Sin dependencias externas
- Menor latencia

**Desventajas:**
- Más código que mantener
- Actualizaciones manuales
- Re-inventar la rueda

### Opción C: Híbrido (Más Práctico)
Canales críticos nativos + OpenClaw para el resto.

---

## 📋 Componentes a Integrar (Prioridad)

### Alta Prioridad
| Componente | Método | Biblioteca Python |
|------------|--------|-------------------|
| **WhatsApp** | Nativo | `neonize` o `whatsapp-web.py` |
| **Telegram** | Nativo | `aiogram` o `python-telegram-bot` |
| **Discord** | Nativo | `py-cord` o `discord.py` |

### Media Prioridad
| Componente | Método | Notas |
|------------|--------|-------|
| Slack | Bridge | Bolt SDK existe en Python |
| Signal | Bridge | Requiere signal-cli |
| Teams | Bridge | Complejo, mejor bridge |

### Baja Prioridad
| Componente | Método | Notas |
|------------|--------|-------|
| iMessage | Bridge | macOS only |
| Matrix | Bridge | Nicho específico |
| IRC | Nativo | Simple con `irc3` |

---

## 🛠️ Estructura Propuesta para KZA

```
src/
├── channels/
│   ├── __init__.py
│   ├── base.py              # ChannelBase abstract class
│   ├── registry.py          # Channel registration
│   ├── routing.py           # Message routing
│   ├── whatsapp/
│   │   ├── __init__.py
│   │   ├── client.py        # WhatsApp Web client
│   │   ├── handlers.py      # Message handlers
│   │   └── media.py         # Media processing
│   ├── telegram/
│   │   ├── __init__.py
│   │   ├── bot.py           # Telegram bot
│   │   ├── handlers.py
│   │   └── commands.py
│   ├── discord/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   └── handlers.py
│   └── bridge/
│       ├── __init__.py
│       └── openclaw.py      # OpenClaw WebSocket bridge
├── gateway/
│   ├── __init__.py
│   ├── server.py            # WebSocket server
│   ├── sessions.py          # Session management
│   └── security.py          # Auth & allowlists
```

---

## 🔧 Implementación Sugerida

### Fase 1: Base (1-2 semanas)
1. Crear `ChannelBase` abstract class
2. Implementar `ChannelRegistry`
3. Crear sistema de routing básico
4. Integrar con `VoicePipeline` existente

### Fase 2: Canales Core (2-3 semanas)
1. Implementar Telegram (más simple)
2. Implementar WhatsApp (más demanda)
3. Implementar Discord

### Fase 3: Gateway (1 semana)
1. WebSocket server para control remoto
2. Integración con Home Assistant
3. Dashboard de canales

### Fase 4: Features Avanzados (2+ semanas)
1. Skills system adaptado
2. OpenClaw bridge para canales adicionales
3. Voice-to-channel (hablar y enviar a WhatsApp)

---

## 📊 Comparación de Bibliotecas Python

### WhatsApp
| Biblioteca | Estado | Método | Notas |
|------------|--------|--------|-------|
| `neonize` | Activo | Web API | Fork de Baileys |
| `whatsapp-web.py` | Activo | Web API | Bien documentado |
| `yowsup` | Abandonado | ❌ | No usar |

### Telegram
| Biblioteca | Estado | Async | Notas |
|------------|--------|-------|-------|
| `aiogram` | Activo | ✅ | Más moderno |
| `python-telegram-bot` | Activo | ✅ | Más popular |

### Discord
| Biblioteca | Estado | Async | Notas |
|------------|--------|-------|-------|
| `py-cord` | Activo | ✅ | Fork de discord.py |
| `discord.py` | Activo | ✅ | Original |

---

## 🎤 Integración con Voice Pipeline

El flujo quedaría así:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ENTRADA                                                              │
├──────────────────────────────────────────────────────────────────────┤
│  Voz (Micrófono) │ WhatsApp │ Telegram │ Discord │ Home Assistant    │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PROCESAMIENTO KZA                                                    │
├──────────────────────────────────────────────────────────────────────┤
│  1. STT (si es voz)                                                   │
│  2. Router/Clasificador                                               │
│  3. LLM 70B (razonamiento)                                            │
│  4. Home Assistant (domótica)                                         │
│  5. TTS (respuesta de voz)                                            │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SALIDA                                                               │
├──────────────────────────────────────────────────────────────────────┤
│  Parlante │ WhatsApp │ Telegram │ Discord │ Home Assistant Dashboard │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Recomendación Final

**Para tu caso de uso**, recomiendo:

1. **Fase Inicial:** Implementar Telegram y WhatsApp nativos en Python
   - Telegram es el más fácil de implementar y testear
   - WhatsApp es el más usado

2. **Fase Siguiente:** Agregar Discord
   - Bueno para comunidades/equipos

3. **Largo Plazo:** OpenClaw Bridge para canales adicionales
   - No reinventar la rueda para Signal, Teams, Matrix

4. **Hardware:** Con tu setup (Threadripper + 4x RTX 3070):
   - Puedes correr KZA + OpenClaw Gateway + LLM local sin problemas
   - El overhead de Node.js es mínimo (~100MB RAM)

---

## ⚠️ Consideraciones Importantes

1. **Seguridad:** OpenClaw implementa DM pairing para evitar spam. Implementar algo similar.

2. **Rate Limits:** WhatsApp y Telegram tienen límites. Respetar.

3. **Terms of Service:**
   - WhatsApp prohíbe oficialmente bots no-business
   - Usar con precaución para uso personal

4. **Mantenimiento:** Las APIs de mensajería cambian frecuentemente.

---

## 📚 Referencias

- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [OpenClaw Docs](https://docs.openclaw.ai)
- [Baileys (WhatsApp)](https://github.com/WhiskeySockets/Baileys)
- [grammY (Telegram)](https://grammy.dev)
- [discord.py](https://discordpy.readthedocs.io)
- [aiogram](https://docs.aiogram.dev)
