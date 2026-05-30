# Command Acceptance Gate — Diseño

**Fecha:** 2026-05-30
**Estado:** Aprobado (brainstorming) → pendiente plan de implementación
**Relacionado:** [[project_escritorio_light_phantom_toggles_2026-05-29]], [[project_wake_tv_filter_pipeline_regression]]

## Contexto y objetivo

Tras arreglar los bugs de la luz fantasma (initial_prompt recortado + guarda de
conflicto de dominio), queda la defensa de fondo: **evitar que audio que no es un
comando real del usuario termine ejecutando una acción**. Hoy esa defensa vive
dispersa en `_is_noise_text` (`src/pipeline/request_router.py:73`) como un pilón
de reglas substring ad-hoc, y **descarta una señal valiosa**: la confianza que el
STT calcula por segmento (`no_speech_prob`, `avg_logprob`) y que `whisper_fast`
tira (`whisper_fast.py:130` solo usa `s.text`).

Objetivo: consolidar la aceptación de comandos en un **gate dedicado, testeable y
extensible**, que sume la señal de confianza del STT, con un rollout que calibra
los umbrales con data real antes de enforcer.

## Threat model y scope (honesto)

El usuario quiere atacar tres fuentes de audio falso:

| Fuente | ¿La ataca este gate? |
|---|---|
| **Alucinaciones de Whisper** (#3) | **SÍ, directo.** La confianza del STT (`no_speech_prob` alto / `avg_logprob` bajo) es exactamente la señal. Causa raíz del incidente 2026-05-29. |
| **TV/media reproduciendo voz** (#1) | **Parcial.** Solo si el audio transcribe con baja calidad. No hay entidad `media_player.*` en HA (331 entidades, solo `remote.universal_remote`) → el gate por contexto HA NO es viable hoy. |
| **Voz humana ambiente** (#2) | **No.** Habla real y limpia → confianza alta → indistinguible de un comando real. Techo: speaker-ID (enrollment, diferido). |

**Decisión de scope:** el gate clava #3 y consolida/mejora la estructura. #1 y #2
quedan para speaker-ID; el gate deja el **slot listo** para esa señal. NO se
implementa speaker-ID ni gate por HA media (no hay entidad) en este trabajo.

**Comportamiento ante captura dudosa:** descarte **silencioso** (sin TTS, sin
diálogo de confirmación). Decisión del usuario.

## Arquitectura y data flow

```
whisper_fast: transcribe()           (sin cambios, compat 2-tupla)
              transcribe_with_confidence() → STTResult(text, elapsed_ms,
                                                        no_speech_prob, avg_logprob)
        ↓ (solo el path de comando)
command_processor.process_command → ProcessedCommand.stt_confidence
        ↓
request_router (chokepoint, hoy _is_noise_text línea 436)
        → CommandAcceptanceGate.evaluate(text, stt_confidence)  (wake_words en __init__)
        → AcceptanceDecision(accept, reason, signals)
              accept → sigue (gate LLM / LLMCommandRouter / dispatcher)
              reject → descarte silencioso: result intent="gate_rejected",
                       success=False, response="", return temprano (sin TTS)
```

## Componentes

### 1. STT surfacea la confianza (`src/stt/whisper_fast.py`)

faster-whisper expone por `Segment`: `avg_logprob` y `no_speech_prob`. Hoy se
descartan. Cambios:

- Nuevo `@dataclass STTResult`: `text: str`, `elapsed_ms: float`,
  `no_speech_prob: float | None`, `avg_logprob: float | None`. `None` = desconocido
  (audio vacío / sin segmentos) → el gate lo trata como "no penalizar".
- Refactor: la lógica de transcripción vive en un helper interno que devuelve
  `STTResult`. Agregación entre segmentos: **media** de `avg_logprob` y **media**
  de `no_speech_prob` (sin segmentos → ambos `None`).
- `transcribe(audio, sample_rate) -> (text, elapsed_ms)` **NO cambia su firma**
  (wrapper sobre el helper) → los ~6 callers existentes (multi_room_audio_loop,
  providers/factory, early-dispatch/streaming internos, warmup) quedan intactos.
- Nuevo `transcribe_with_confidence(audio, sample_rate) -> STTResult` → lo usa
  solo `command_processor` en el path de comando.
- `MoonshineSTT` (factory alternativo): agregar `transcribe_with_confidence`
  que devuelve `STTResult` con confianza `None` (no expone esos campos) para
  cumplir la interfaz sin romper.

### 2. `ProcessedCommand` carga la confianza (`src/pipeline/command_processor.py`)

- Nuevo campo `stt_confidence: "STTConfidence | None" = None` (o tupla
  `(no_speech_prob, avg_logprob)`; se decide en el plan). Default `None`.
- `process_command`: el path que transcribe (no el `pretranscribed_text`, que
  viene del wake/early-dispatch y no tiene confianza propia) usa
  `transcribe_with_confidence` y puebla `result.stt_confidence`. El path
  `pretranscribed_text` deja `None` (el gate no penaliza confianza desconocida).

### 3. `CommandAcceptanceGate` (nuevo `src/nlu/command_gate.py`)

- `@dataclass(frozen=True) AcceptanceDecision`: `accept: bool`, `reason: str`,
  `signals: dict` (para logging/telemetría: incluye los valores evaluados).
- `class CommandAcceptanceGate`:
  - `__init__(self, wake_words, enforce_confidence, max_no_speech_prob, min_avg_logprob)`.
  - `evaluate(text, stt_confidence) -> AcceptanceDecision`.
  - **Hard rules (enforce siempre)** — migradas tal cual desde `_is_noise_text`:
    vacío / vacío-tras-normalizar, `_NOISE_PHRASES` (TV/YouTube + eco TTS),
    filler words, repetición extrema, **wake ausente**. Mantienen su semántica y
    sus tests actuales.
  - **Confidence rules (nuevas)** — si hay `stt_confidence` (no `None`):
    `no_speech_prob > max_no_speech_prob` **o** `avg_logprob < min_avg_logprob`
    → candidato a rechazo. Sujeto a `enforce_confidence` (ver rollout).
  - El orden: hard rules primero (rechazo inmediato si matchean); luego
    confidence.

### 4. Wiring en `request_router.py`

- `RequestRouter.__init__` recibe un `command_gate` inyectado (DI por
  constructor, patrón del proyecto). `main.py` lo construye desde config.
- Reemplazar las dos llamadas a `_is_noise_text` (líneas 436 y 714) por
  `self.command_gate.evaluate(...)`. `_is_noise_text` y sus constantes se
  **mueven** a `command_gate.py` (no quedan duplicadas).
- Backward-compat: si `command_gate is None` (no configurado), comportamiento =
  hard rules equivalentes a hoy (o un gate default). Sin gate inyectado el
  pipeline no debe romperse.

## Rollout y calibración (la clave del "bien atacado")

Las hard rules ya están validadas (shippean enforce desde el día 1). Las
**confidence rules arrancan en shadow mode**:

- `enforce_confidence: false` (default): el gate **evalúa y LOGuea** qué
  rechazaría por confianza, con los valores reales (`no_speech_prob`,
  `avg_logprob`, `text`), pero **NO descarta** — el comando sigue. Cero riesgo de
  tirar comandos reales mientras juntamos data de la casa real.
- Tras revisar los logs y validar umbrales, flippear `enforce_confidence: true`.

**Logging de calibración:** en cada comando (aceptado o no), el gate emite a INFO
una línea estructurada con `no_speech_prob`, `avg_logprob`, `decision`, `reason`,
`would_reject` (en shadow) y `text[:60]`. Permite tunear thresholds con `grep` +
percentiles sobre tráfico real.

## Config (`config/settings.yaml`, bloque nuevo `command_gate`)

```yaml
command_gate:
  enforce_confidence: false        # shadow mode hasta calibrar
  max_no_speech_prob: 0.60         # > esto = sospechoso (conservador, a calibrar)
  min_avg_logprob: -1.20           # < esto = sospechoso (conservador, a calibrar)
  # Hard rules (noise phrases, filler, repetición, wake ausente) siempre enforce.
```

## Manejo de errores

- El gate **nunca** debe tumbar el pipeline. Si `evaluate` lanza, se loguea y se
  **acepta** el comando (fail-open) — preferimos un fantasma ocasional a perder
  todo el control de voz.
- `stt_confidence is None` (audio vacío, Moonshine, pretranscribed) → confidence
  rules no aplican (no penalizan). Las hard rules sí.

## Testing

- **Unit del gate** (`tests/unit/nlu/test_command_gate.py`): cada hard rule
  (migración de los casos de `_is_noise_text`), cada confidence rule, shadow vs
  enforce (shadow → accept + `would_reject` en signals; enforce → reject),
  `None` confidence → no penaliza, fail-open ante excepción.
- **STT confidence** (`tests/unit/stt/`): `transcribe_with_confidence` agrega
  bien la confianza con segments mockeados (`no_speech_prob`/`avg_logprob`);
  sin segmentos → `None`; `transcribe()` 2-tupla sigue intacto.
- **Integración** (`tests/unit/pipeline/`): `request_router` acepta→pasa,
  reject-enforce→descarta (intent="gate_rejected", sin TTS), reject-shadow→pasa.

## Fuera de scope (futuro, slots dejados)

- **Speaker-ID en el gate** (resuelve #1/#2 de verdad): cuando haya enrollment,
  el gate suma una hard/confidence rule de speaker. El flag voice-auth
  (`security.require_known_speaker_for_actions`) ya existe; convergerán.
- **Gate por HA media context**: cuando exista entidad `media_player.*` de la TV
  en HA, sumar "si hay media sonando en la zona, subir la vara".
- **Energía/SNR de audio**: descartado como señal central (los fantasmas del
  2026-05-29 eran fuertes, -13 a -22 dBFS → energía no discrimina).
- **Calibración automática** de thresholds desde los logs (hoy manual).

## Resumen de archivos tocados

| Archivo | Cambio |
|---|---|
| `src/stt/whisper_fast.py` | `STTResult` dataclass, helper interno, `transcribe_with_confidence()`; `transcribe()` sin cambios |
| `src/pipeline/command_processor.py` | campo `stt_confidence` en `ProcessedCommand`; usar `transcribe_with_confidence` en el path de comando |
| `src/nlu/command_gate.py` (nuevo) | `CommandAcceptanceGate`, `AcceptanceDecision`, constantes migradas de `_is_noise_text` |
| `src/pipeline/request_router.py` | inyectar `command_gate`, reemplazar `_is_noise_text` (×2), mover constantes |
| `src/main.py` | construir `CommandAcceptanceGate` desde config e inyectarlo |
| `config/settings.yaml` | bloque `command_gate` |
| `tests/unit/nlu/`, `tests/unit/stt/`, `tests/unit/pipeline/` | unit + integración |
