# Update para Notion KZA página 8 — "Proyectos registrados" → entrada KZA

**Pegá esto reemplazando/actualizando la entrada de KZA en la sección "Proyectos registrados" de la página 8 del Notion.**

## kza (UID 1000)

**Dueño**: Gabriel. `/home/kza/`. Asistente de voz domótico local.

### Componentes
- **kza-72b.service** (nativo, excepción R10 #3 pre-existente del LLM): Qwen2.5-72B Q8_0 vía `llama-cpp-python` server en `127.0.0.1:8200`, OpenAI API compat. Siempre caliente (~37 GB RSS + ~37 GB page cache mmap). Unit: `~/.config/systemd/user/kza-72b.service`.
- **kza-voice.service** (nativo, **excepción R10 #3**): voice pipeline completo (wake word + STT + SpeakerID + Emotion + TTS + orchestrator + HA client + Chroma in-process). Depende de `kza-72b.service`. Unit: `~/.config/systemd/user/kza-voice.service`. Habilitado + disabled hasta cut-over final.
- **Quadlets preparados** (no arrancados hasta que el código soporte HttpClient): `kza-internal.network` + `kza-chroma.container` (imagen `docker.io/chromadb/chroma:0.5.23`, `PublishPort=127.0.0.1:9500:8000`).

### Justificación excepción R10 #3 (voice pipeline nativo)
- Dispositivos USB 2886:0018 (ReSpeaker Mic Array v2.0) + extensores RJ45 por habitación; passthrough a contenedor requiere `--device=/dev/snd` + grupos `audio`.
- MA1260 serial (`/dev/ttyUSB0`) para routing de zonas.
- Latencia objetivo <300 ms end-to-end; overhead CDI + contenedor agregaría ~50-80 ms en el path crítico.
- Precedente: vLLM en `infra` también es nativo por motivos similares.

### Consumos del shared
- **vLLM 7B** (infra `:8100`) — router/clasificador + auto-indexación de comandos HA a ChromaDB.
- **Home Assistant REST** (`:8123`) — domótica.

### Modelos operados (referencia para catálogo §4.1 si se quieren agregar como disponibles)
| Modelo | Rol | Ubicación | VRAM/RAM | Notas |
|---|---|---|---|---|
| Whisper large-v3-turbo (CT2) | STT | cuda:0 | ~800 MB | `./models/whisper-v3-turbo/` |
| BGE-M3 | Embeddings multilingüe (dim=1024) | cuda:0 | ~1.5 GB | Reemplaza antigua posición en cuda:1 (no cabía con vLLM) |
| Kokoro-82M | TTS | cuda:0 | ~1 GB | 347 MB disco |
| ECAPA-TDNN | SpeakerID | cuda:0 | ~500 MB | |
| wav2vec2 emotion | Detección emociones | cuda:0 | ~1 GB | |
| Qwen2.5-72B-Instruct Q8_0 | Razonamiento profundo (slow path) | CPU | ~37 GB RSS | vía `kza-72b.service :8200`; no compartido con otros proyectos |

### Puertos publicados (sub-rango KZA: 9500-9599)
- `127.0.0.1:9500` — kza-chroma.container (pendiente habilitación)
- `127.0.0.1:8200` — kza-72b.service (local-only, sin LAN exposure)

### VRAM / RAM declarados
- **cuda:0**: ~7.5 GB (STT 0.8 + TTS 1 + SpeakerID 0.5 + Emotion 1 + BGE-M3 1.5 + buffers) — `queda ~0.5 GB margen`.
- **RAM**: ~45 GB (72B ~37 GB + pipeline ~8 GB).

### Estado (2026-04-21)
- Voice pipeline funcional end-to-end (validado con `scripts/test_voice_command.py`).
- ChromaDB con **640 frases** de 10 grupos de luces con capabilities (onoff/brightness/color_temp/color).
- NLU con intent classifier léxico + slot extractor (brightness_pct, rgb_color, color_temp_kelvin) integrado a `RequestRouter`.
- `LastActionTracker` para comandos ambiguos (Q6 C+B: toggle implícito con TTL 60s, pregunta fallback).
- Migración a layout estándar `/home/kza/{app,data,secrets,logs,.config/…}` completada.
- Pendiente: contenedorización de Chroma, enable `kza-voice.service` como cut-over final.
