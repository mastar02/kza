# Voice optimization — 6 sprints post-early-dispatch

**Fecha**: 2026-04-24
**Contexto**: KZA pipeline operativo con wake word "nexa" (Whisper + fuzzy fonético),
early dispatch por streaming parser, speaker filter deshabilitado temporalmente.
Este doc coordina 6 mejoras independientes, cada una con su plan detallado.

## Sprints

| # | Sprint | Effort | Depende | Archivos tocados |
|---|---|---|---|---|
| S1 | Model warmup startup | 🟢 30m | — | `src/main.py` |
| S2 | TTS pre-generated responses | 🟡 2-3h | S1 | `src/tts/response_cache.py` (new), `response_handler.py`, `main.py`, `settings.yaml` |
| S3 | Barge-in (interrumpir TTS) | 🟡 3-4h | S2 | `response_handler.py`, `multi_room_audio_loop.py`, `echo_suppressor.py`, `settings.yaml` |
| S4 | Confidence-based confirmation | 🟡 2-3h | — | `command_grammar.py`, `request_router.py`, `settings.yaml` |
| S5 | Endpointing neural | 🟠 4-6h | — | `whisper_wake.py`, `streaming_whisper_wake.py`, `multi_room_audio_loop.py`, `settings.yaml` |
| S6 | HA state prefetch cache | 🔴 4-6h | — | `ha_client.py`, `dispatcher.py`, `action_context.py`, `settings.yaml` |

Docs individuales:
- [S1](2026-04-24-s1-model-warmup.md)
- [S2](2026-04-24-s2-tts-response-cache.md)
- [S3](2026-04-24-s3-barge-in.md)
- [S4](2026-04-24-s4-confidence-confirmation.md)
- [S5](2026-04-24-s5-endpointing-neural.md)
- [S6](2026-04-24-s6-ha-state-prefetch.md)

## Matriz de conflictos

|  | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|
| S1 | — | ⚠️ `main.py` | | | | |
| S2 | ⚠️ `main.py` | — | ⚠️ `response_handler.py` | | | |
| S3 | | ⚠️ `response_handler.py` | — | | ⚠️ `loop.py` | |
| S4 | | | | — | | |
| S5 | | | ⚠️ `loop.py` | | — | |
| S6 | | | | | | — |

**TODOS** tocan `config/settings.yaml`, pero en secciones distintas. Merge conflicts
son resolvables automáticamente o triviales de editar a mano (cada sprint edita su
propio sub-árbol YAML).

## Estrategia de paralelización

### Ronda 1 (paralelo, 3 agentes)
- **S1** (main.py additions post-`load()`)
- **S4** (independiente, solo parser + router)
- **S6** (independiente, solo HA client + dispatcher)

Agentes en worktrees isolados → 3 branches → merge secuencial al main.

### Ronda 2 (después que S1 mergeó)
- **S2** (depende del pattern de warmup para pre-generar TTS)

### Ronda 3 (después que S2 mergeó)
- **S3** (requiere el cache de S2 para restart cleanly el TTS interrumpido)
- **S5** (independiente, pero toca `multi_room_audio_loop.py` que S3 también toca)

S3 y S5 pueden ir paralelos si se coordinan en `multi_room_audio_loop.py` — S3
agrega detección barge-in en audio_callback, S5 cambia el VAD check en
`_check_vad_completion`. Secciones distintas del archivo.

## Criterios comunes

Cada sprint debe:
1. **No romper tests existentes** — correr `pytest tests/unit/nlu/ tests/unit/wakeword/` antes del commit.
2. **Agregar tests propios** bajo `tests/unit/<module>/`.
3. **Feature flag en config** — cada mejora es opt-in vía `settings.yaml`, default `false` excepto donde se indique.
4. **Logs claros** en INFO level cuando se activa (una sola vez al startup).
5. **Commit con mensaje descriptivo** siguiendo el patrón `feat(area): descripción`.
6. **Update del plan** marcando completado al final de `docs/superpowers/plans/<sprint>.md`.

## Protocolo de despliegue

Post-merge al main, un solo deploy:
```bash
ssh kza@192.168.1.2 "cd /home/kza/app && git pull --ff-only origin main && systemctl --user restart kza-voice.service"
```

Validación post-deploy:
```bash
ssh kza@192.168.1.2 "journalctl --user -u kza-voice.service --since '30 seconds ago' -n 50"
```
