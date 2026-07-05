# Umbral de wake adaptativo por vad en STRICT — diseño

**Fecha**: 2026-06-09 · **Estado**: spec aprobado, pendiente plan de implementación
**Origen**: análisis noche 2026-06-08 — la conversación humana sostenida dejó sordo al asistente.

## Problema

El `AmbientGuard` (spec 2026-06-05) escala a estado STRICT cuando hay "ambiente
hostil" (≥3 capturas rechazadas en 180s) y en STRICT exige wake `score ≥ 0.72`.
Fue calibrado contra TV de fondo. Pero **no distingue TV de conversación humana**:
ambas generan capturas rechazadas y escalan a STRICT igual.

Medido la noche del 2026-06-08 (conversación sostenida 18-23h, ~1390 utterances
ambient):
- **46 entradas a STRICT**, **188 wakes rechazados** por `strict_score`.
- **Solo 3 comandos** lograron procesarse en toda la noche.
- Los "Nexa" reales del usuario, far-field, disparaban el wake a 0.42-0.65 →
  bloqueados por el umbral 0.72 antes de transcribirse (no aparecen ni en los
  logs del router).

**Causa raíz**: en ambiente de voz-humana, STRICT endurece el wake justo cuando
la voz far-field del usuario es más débil → mata sus comandos. En ambiente-TV,
STRICT es correcto (la TV no da comandos).

## Objetivo

Que el guard **module la severidad del wake en STRICT según si el audio del
trigger es voz clara o no**, en vez de un umbral fijo. Voz clara cercana →
umbral más bajo (el comando pasa); audio pobre/lejano (TV, ruido) → umbral alto
(se mantiene la protección).

**No-objetivos** (YAGNI): distinguir "comando" de "charla" (imposible sin la
wake word — la wake word ES la señal de intención); DoA/speaker-ID (no hay
enrollment); tocar el command path; cambiar entrada/salida de STRICT o COOLDOWN.

## Señal: Silero en vivo sobre el audio del wake

Decisión: la señal de modulación es `wake_vad` = prob de Silero corrida sobre el
**mismo buffer que disparó el wake**, síncrona en el audio thread (~0.46ms/chunk,
costo despreciable). Frente a usar el estado reciente del ambient path:
- **Más fresca**: mide la captura puntual, sin la latencia del ambient
  (segmento cierra a 700ms de silencio + Parakeet).
- **Sin acople**: el guard no depende de que el ambient path haya transcrito.
- Silero ya está cargado en el proceso (lo usa el ambient segmenter).

### Limitación conocida (verificada con datos, no asumida)

`vad_prob` separa **voz vs silencio/ruido**, NO **TV-con-diálogo vs voz-humana**:
la noche del 08 una línea de TV en inglés ("Arthur has documents I'm going to
show") dio `vad 0.66`, tan alto como la conversación humana. Implicaciones de
diseño:
1. El escalado es **parcial**: `0.72 → strict_wake_score_min` (~0.5), **nunca
   hasta el threshold base 0.4**. Así la TV-con-diálogo todavía necesita un wake
   decente (la Fase 0 mostró que la TV raramente excede wakes fuertes).
2. Los breakpoints se **calibran con datos reales** (modo shadow abajo), no a ojo.

## Mecanismo

En `AmbientGuard.on_wake`, solo en estado **STRICT**, el umbral fijo
`strict_wake_score` se reemplaza por un umbral interpolado:

```
def _effective_strict_threshold(wake_vad):
    lo, hi = strict_vad_lo, strict_vad_hi          # p.ej. 0.3, 0.7
    hard, soft = strict_wake_score, strict_wake_score_min  # 0.72, 0.50
    if wake_vad <= lo:   return hard
    if wake_vad >= hi:   return soft
    frac = (wake_vad - lo) / (hi - lo)
    return hard - frac * (hard - soft)             # lineal hard→soft
```

- `wake_vad ≤ strict_vad_lo` → `strict_wake_score` (0.72): TV/ruido lejano.
- `wake_vad ≥ strict_vad_hi` → `strict_wake_score_min` (0.50): voz clara.
- intermedio: lineal.

NORMAL y COOLDOWN **no cambian**. Las compuertas `strict_min_rms`/
`strict_min_spenergy` (hoy muertas, 0.0) quedan como están.

### Compat de firma

`on_wake(room_id, score, rms, spenergy_peak=None)` gana un parámetro opcional
`wake_vad: float | None = None`. Si es `None` (caller no lo provee, o fallo al
computarlo) → se usa el umbral fijo `strict_wake_score` (fail-safe: comportamiento
actual, nunca más permisivo por falta de señal).

## Modo shadow → enforce (patrón del proyecto)

`strict_vad_adaptive` (bool, **default `false` = shadow**):
- **shadow** (`false`): el guard computa el umbral adaptativo y **loguea**
  `[AmbientGuard-vadshadow] room=X wake_vad=.. score=.. umbral_fijo=0.72
  umbral_adaptativo=Y cambiaria=sí/no` pero **decide con el umbral fijo**. Cero
  efecto en producción; junta datos (incluidos los comandos reales del usuario).
- **enforce** (`true`): decide con el umbral adaptativo.

Con unos días de shadow se calibran `strict_vad_lo`/`strict_vad_hi`/
`strict_wake_score_min` contra los `wake_vad` reales de los comandos del usuario
vs los de TV, y recién ahí se flipea a enforce. Evita adivinar umbrales (riesgo
explícito registrado en la memoria del proyecto).

## Config (settings.yaml → rooms.ambient_guard)

```yaml
    strict_vad_adaptive: false        # shadow primero
    strict_wake_score_min: 0.50       # umbral con voz clara (wake_vad alto)
    strict_vad_lo: 0.30               # ≤ esto → umbral duro 0.72
    strict_vad_hi: 0.70               # ≥ esto → umbral blando 0.50
    # strict_wake_score: 0.72 (ya existe) = umbral duro / TV
```

## Integración

`multi_room_audio_loop`: en el punto donde hoy llama `guard.on_wake(...)` tras un
trigger, corre Silero sobre el buffer del wake y pasa `wake_vad`. El predictor
Silero se obtiene del mismo factory que usa el ambient segmenter
(`make_silero_predictor`), inyectado o lazy. Fail-open: si Silero falla →
`wake_vad=None` → umbral fijo.

## Testing

`AmbientGuard` ya tiene reloj inyectable y tests sin sleeps. Casos nuevos:
- `wake_vad ≤ lo` en STRICT → umbral 0.72 (rechaza score 0.6).
- `wake_vad ≥ hi` en STRICT → umbral 0.50 (acepta score 0.6).
- interpolación: `wake_vad` intermedio → umbral entre 0.50 y 0.72.
- `wake_vad=None` → umbral fijo 0.72 (fail-safe).
- shadow (`strict_vad_adaptive=false`): decisión = umbral fijo, pero loguea.
- NORMAL/COOLDOWN ignoran `wake_vad`.

## Archivos

- `src/pipeline/ambient_guard.py`: `_effective_strict_threshold`, `on_wake`
  acepta `wake_vad`, config nuevos campos + `__post_init__` (clamp lo<hi,
  soft≤hard).
- `src/pipeline/multi_room_audio_loop.py`: computar `wake_vad` con Silero y pasarlo.
- `config/settings.yaml`: bloque `ambient_guard` (4 campos nuevos).
- `tests/unit/pipeline/test_ambient_guard*.py`: casos de escalado + shadow.
