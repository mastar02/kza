# Wake textual "nexa" sobre el stream ambient — Diseño (Etapa A)

**Fecha:** 2026-07-05
**Estado:** Aprobado (pendiente plan)
**Contexto:** el wake acústico (openwakeword nexa.onnx) scorea 0.41-0.59 con la
acústica post-mudanza del mic (2026-07-02) — funciona sin margen; el fix de
fondo (re-entrenar con el dataset WakeClipWriter) sigue pendiente. Mientras
tanto, el ambient path (Fase 2, shadow desde 2026-06-06) ya transcribe todo
con Parakeet en CPU. Este diseño agrega un canal de disparo TEXTUAL sobre esas
transcripciones: red de seguridad cuando el acústico no dispara.

## Requisitos del usuario (2026-07-05)

1. **Sin "nexa" no se ejecuta nada**, y todo disparo pasa por el routing
   fast/slow existente (gate → grammar/LLMRouter → dispatcher). El canal
   textual NO bypassa ninguna defensa.
2. **Detección de "nexa" fiable.**
3. **Escala a multi-room** cuando haya más micrófonos.
4. Modo inicial: **ejecutar desde el día 1** (sin fase shadow; auditable por
   logs).
5. El uso "inteligente" del mic array (beams por ángulo del hablante) es la
   **Etapa B** — spec separada, requiere flasheo 6ch del XVF3800.

## Arquitectura

```
AmbientTranscriber._handle_segment (existente)
  └─ utterance {text, room_id, source, speaker, azimuth, t0, t1}
        │  source == 'tv' → NO dispara (compuerta anti-TV vía source_classifier)
        ▼
  TextualWakeDetector.match(text) — NUEVO (src/ambient/textual_wake.py)
        │  sin match "nexa" → nada
        ▼
  dedup: ¿el command path procesó algo en esta room hace < dedup_window_s?
        │  sí → skip (el wake acústico ya lo tomó)
        ▼
  CommandEvent pretranscripto (texto completo de la utterance, room_id)
        → request_router.process_command()   ← el MISMO camino de siempre
```

### Matcher (calibrado con datos reales de ambient.db, 3.869 utterances)

- Variantes observadas de "nexa" en Parakeet: `nexa` (19), `next up` (7,
  artefacto del language-ID inglés), `lexa` (1).
- Regla v1: sobre texto normalizado (lowercase, sin acentos/puntuación),
  token exacto en `{"nexa", "next up"}` O token con edit-distance ≤ 1 contra
  `"nexa"` (cubre neza/nesa/mexa/lexa). `"alexa"` queda afuera (distancia 2)
  — a propósito: es la otra asistente.
- Config: `ambient.textual_wake.variants` (lista) y `max_edit_distance` en
  settings.yaml — recalibrable sin código.

### Dedup contra el wake acústico

- El detector consulta el timestamp del último comando procesado por room
  (el request_router lo registra; ver plan para el mecanismo exacto).
- `dedup_window_s: 8.0` (configurable). Si el acústico disparó hace <8s en
  esa room, el textual se abstiene — evita doble ejecución del mismo comando
  (el canal textual llega 1-3s más tarde por diseño).

### Fiabilidad ejecutando desde el día 1

- Defensas heredadas completas: CommandGate (blocklist + prompt_echo +
  filler), grammar fastpath (conf ≥ 0.75), LLMRouter (descarta no-comandos).
- `source == 'tv'` corta el caso "la TV dijo nexa" ANTES del matcher.
- Todo disparo (ejecute o no) se loguea `[TextualWake]` con texto, room,
  speaker, source y decisión — auditoría por journal desde el primer día.
- Kill-switch: `ambient.textual_wake.enabled: false` (independiente del
  ambient path).

### Multi-room

Heredado del ambient path: cada utterance trae su `room_id`; el CommandEvent
se emite con esa room. Rooms nuevas del ambient obtienen wake textual sin
código adicional.

## Latencia y GPU (decisión 2026-07-05)

- Latencia total del canal: ~1.5-3s, dominada por el cierre de utterance del
  segmentador (~0.5-1s de silencio) + cola del worker; la inferencia Parakeet
  CPU es ~100-300ms. Rol: red de seguridad, NO reemplaza el fast path
  acústico (<300ms).
- **V1 en CPU.** Promover Parakeet a GPU recortaría solo ~0.2s (cuda:0 tiene
  ~6.4GB libres medidos hoy con el sistema caliente — dato que corrige la
  nota vieja de "VRAM apretadísima") pero introduce contención con Whisper en
  el fast path. Knob documentado: `ambient.stt.device` — decisión diferida.
- Lever real de latencia (posterior, con datos de `[TextualWake]`): cierre
  adaptativo del segmentador.
- GPU para Parakeet se reevalúa en la **Etapa B** (beams paralelos ×N
  transcripciones), donde el throughput sí importa. Regla de siempre:
  reasignación de GPU se discute antes de ejecutarla.

## Etapa B (spec separada, futura)

Firmware 6ch del XVF3800 (canales raw + DoA fino) → beamforming por software
hacia el azimuth del hablante activo → transcripción paralela de 2+ beams con
voces simultáneas → mejor beam alimenta ambient y command path. Prerequisitos:
flasheo coordinado (kza-voice parado — regla: jamás tocar USB del XVF con el
servicio vivo), re-validación SPENERGY/AGC sobre los canales nuevos, decisión
de GPU. Los seams de A ya están listos: `room_id` + `azimuth` viajan en cada
utterance.

## Manejo de errores

| Falla | Comportamiento |
|-------|----------------|
| Matcher lanza excepción | fail-open del ambient: la utterance se persiste igual, sin disparo; error logueado |
| request_router falla en el dispatch textual | mismo manejo que un comando acústico (result con success=False); no tumba el transcriber |
| Doble disparo acústico+textual | dedup_window_s lo corta; si igual pasara, el dedup existente del router (dedup_window_ms) es la segunda barrera |
| Parakeet transcribe el comando garbled | las defensas del router deciden (igual que cualquier comando); el textual no agrega riesgo nuevo |

## Testing

- Matcher: variantes reales (nexa/next up/neza/lexa), exclusión de "alexa",
  normalización, edge cases (texto vacío, "nexa" como substring de otra
  palabra NO matchea — token-level).
- Dedup: dentro/fuera de ventana, por room independiente.
- Integración: utterance con "nexa apagá la luz" → CommandEvent al router
  (mock); utterance source='tv' con "nexa" → NO dispara; utterance sin nexa
  → NO dispara.
- Fail-open: matcher que lanza → utterance persiste, sin crash.

## Qué NO toca

- El wake acústico y su pipeline (siguen idénticos; el textual es aditivo).
- El segmentador/clasificador del ambient (v1 solo CONSUME utterances).
- GPUs, VRAM, modelos residentes.
- El slow path cloud: el canal textual produce CommandEvents que siguen las
  mismas reglas de consent existentes; las utterances ambient NO comando
  siguen sin salir del server (TTL 48h local), como hasta hoy.

## Fuera de alcance (v1)

- Etapa B completa (beams/DoA).
- Speaker-ID como gate del disparo textual (enforcement sigue opt-in global
  `security.require_known_speaker_for_actions`, default off — cuando se
  active, aplica a ambos canales por igual vía el router).
- Cierre adaptativo del segmentador (tuning posterior con datos).
- Matching sobre transcripciones parciales (requiere tocar el segmentador).
