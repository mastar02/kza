# S5 — Endpointing neural

**Effort**: 🟠 4-6 h
**Depende**: early dispatch (ya implementado)
**Branch sugerido**: `feat/s5-endpointing-neural`

## Objetivo

Reemplazar el endpointing basado en threshold de silencio (silero-VAD + 300ms)
por un classifier neural que distingue "pausa de pensamiento" (no cortar) de
"final de turno" (cortar YA). Reduce latencia percibida y falsos cortes.

## Contexto

Hoy en `multi_room_audio_loop._check_vad_completion`:
- `silence_duration_ms=300` por defecto → esperamos 300ms de silencio antes de cerrar.
- Si el user hace una pausa natural mid-frase, cerramos antes y perdemos contexto.
- Si el user habla con un ritmo variable, el threshold no adapta.

Google/Apple usan "endpointer" dedicado: un pequeño modelo ML que ve los últimos
N ms y predice "turno terminó" con probabilidad. Opciones:

1. **silero-vad-streaming** (upgrade del actual) — gratis, misma API pero con señales más finas.
2. **Nvidia NeMo VAD** (~50 MB, preciso) — requiere nemo + TensorRT.
3. **webrtc-vad** (fast, ligero) — solo VAD binaria, no turno.
4. **Conformer-CTC endpointer** (~5 MB, custom-trained) — complejo.
5. **Heurística combinada con parser**: aprovechar `parse_partial_command.ready_to_dispatch()` como señal positiva de "turno completo" (early exit legítimo).

**Recomendación**: **opción 1 + opción 5** combinadas. Bajo esfuerzo, alto impacto.

## Arquitectura

```
audio_chunk ──► silero-vad (refinado, prob de voz)
                │
                ├─ prob > 0.7 → voz activa, resetear silence_run
                ├─ prob < 0.3 → silencio fuerte, incrementar silence_run
                └─ 0.3 ≤ prob ≤ 0.7 → ambiguo, incrementar lento
                │
                ├─ parse_partial_command(current_text).ready_to_dispatch()
                │   ├─ True → cerrar inmediatamente (endpoint reached)
                │   └─ False → esperar más señal
                │
                └─ silence_run_ms >= adaptive_threshold → cerrar
                    donde adaptive_threshold:
                      - 150ms si parser ya tiene intent+entity (end cerca)
                      - 300ms si solo wake+intent (esperar entity)
                      - 500ms si solo wake (esperar verbo)
```

## Archivos a modificar

### Investigación previa (agente debe hacer esto antes de codear)

```bash
pip index versions silero-vad         # última versión
pip index versions speech-recognition # para comparar
```

Verificar en `torch.hub` si `silero_vad` actual ya expone `prob` directamente.
El módulo actual devuelve `torch.Tensor` con prob por chunk.

### `src/wakeword/whisper_wake.py`

Ya tiene `_is_speech(chunk)` que devuelve bool. Reemplazar por `_voice_prob(chunk)
-> float`:

```python
def _voice_prob(self, chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    if rms < self.min_rms:
        return 0.0
    if self._vad is not None:
        try:
            tensor = self._torch.from_numpy(chunk.astype(np.float32))
            with self._torch.no_grad():
                return float(self._vad(tensor, SAMPLE_RATE).item())
        except Exception:
            pass
    return 0.5 if rms > self.min_rms else 0.0


def _is_speech(self, chunk: np.ndarray) -> bool:
    """Legacy wrapper: True si prob >= threshold."""
    return self._voice_prob(chunk) >= self.vad_threshold
```

### `src/pipeline/multi_room_audio_loop.py`

Extender `_check_vad_completion` con endpointing adaptativo:

```python
def __init__(
    self,
    ...,
    endpointing_enabled: bool = True,
    endpointing_short_ms: int = 150,  # silence if parser ready
    endpointing_medium_ms: int = 300, # silence if parser intent-only
    endpointing_long_ms: int = 500,   # silence if no progress
    endpointing_stt=None,  # optional: para parse_partial check
):
    ...
    self.endpointing_enabled = endpointing_enabled
    self.endpointing_short_ms = endpointing_short_ms
    self.endpointing_medium_ms = endpointing_medium_ms
    self.endpointing_long_ms = endpointing_long_ms
    self._endpointing_stt = endpointing_stt
```

En `_check_vad_completion`:

```python
def _check_vad_completion(self, rs: RoomStream) -> tuple[bool, np.ndarray | None]:
    elapsed = time.time() - rs.command_start_time
    elapsed_ms = elapsed * 1000

    if elapsed_ms < self.min_speech_ms:
        return False, None
    if not rs.audio_buffer:
        return False, None

    samples_per_ms = self.sample_rate // 1000
    silence_samples = int(self.silence_duration_ms * samples_per_ms)
    recent = rs.audio_buffer[-silence_samples:]

    if recent:
        recent_array = np.array(recent, dtype=np.float32)
        rms = float(np.sqrt(np.mean(recent_array ** 2)))

        # NEW: endpointing adaptativo
        if self.endpointing_enabled:
            threshold_ms = self._adaptive_endpoint_threshold(rs)
            silence_needed = int(threshold_ms * samples_per_ms)
            recent_adaptive = rs.audio_buffer[-silence_needed:]
            if len(recent_adaptive) >= silence_needed:
                rms_adaptive = float(np.sqrt(np.mean(
                    np.array(recent_adaptive, dtype=np.float32) ** 2
                )))
                if rms_adaptive < self.silence_threshold:
                    audio_data = np.array(rs.audio_buffer, dtype=np.float32)
                    logger.debug(
                        f"Adaptive endpoint in {rs.room_id}: "
                        f"threshold={threshold_ms}ms rms={rms_adaptive:.3f}"
                    )
                    return True, audio_data

        # Fallback a silence check clásico
        if rms < self.silence_threshold:
            audio_data = np.array(rs.audio_buffer, dtype=np.float32)
            return True, audio_data

    if elapsed >= self.command_duration:
        audio_data = np.array(rs.audio_buffer, dtype=np.float32)
        return True, audio_data

    return False, None


def _adaptive_endpoint_threshold(self, rs: RoomStream) -> int:
    """
    Decide cuánto silencio esperar antes de cerrar, basado en el estado
    actual del parser streaming (si hay).
    """
    if rs.early_command is not None and rs.early_command.ready_to_dispatch():
        return self.endpointing_short_ms
    # Si hay partial transcript con intent pero sin entity, esperar más.
    # Sin info de parser, default al medium.
    return self.endpointing_medium_ms
```

### `config/settings.yaml`

```yaml
rooms:
  ...
  endpointing:
    enabled: true
    short_ms: 150       # si parser ya está ready_to_dispatch
    medium_ms: 300      # si parser parcial
    long_ms: 500        # si sin señal del parser
```

### `src/main.py`

```python
endpointing_cfg = rooms_config.get("endpointing", {})
multi_room_loop = MultiRoomAudioLoop(
    ...
    endpointing_enabled=endpointing_cfg.get("enabled", True),
    endpointing_short_ms=endpointing_cfg.get("short_ms", 150),
    endpointing_medium_ms=endpointing_cfg.get("medium_ms", 300),
    endpointing_long_ms=endpointing_cfg.get("long_ms", 500),
)
```

## Validación

1. Comando con pausa: "nexa... apagá... la luz del escritorio"
   - Antes: cerraba en la primera pausa (300ms silencio) → STT obtiene "nexa".
   - Después: parser no está ready, endpoint espera 500ms → captura todo.

2. Comando rápido: "nexa apagá la luz"
   - Antes: cerraba 300ms post-"luz".
   - Después: parser ready_to_dispatch → cierre inmediato (150ms silencio) →
     ganancia de ~150ms en latencia percibida.

3. Comando largo: "nexa poné la luz del living en color azul al 80 por ciento"
   - Parser no está ready hasta `luz del living` (intent + entity + room)
   - Post-"por ciento" pausa → cierre rápido (150ms).

## Test unitario

`tests/unit/pipeline/test_endpointing.py`:
```python
import numpy as np
from unittest.mock import MagicMock
from src.nlu.command_grammar import PartialCommand

def test_adaptive_threshold_ready_command():
    rs = MagicMock()
    rs.early_command = PartialCommand(intent="turn_off", entity="light")
    loop = MultiRoomAudioLoop(...)
    assert loop._adaptive_endpoint_threshold(rs) == 150

def test_adaptive_threshold_no_ready():
    rs = MagicMock()
    rs.early_command = None
    loop = MultiRoomAudioLoop(...)
    assert loop._adaptive_endpoint_threshold(rs) == 300
```

## Edge cases

- **Falso ready_to_dispatch**: si el parser confunde "la luz" como entity pero el
  user iba a decir "la luz del jardín" (no configurado), endpoint corto = corte
  prematuro. Mitigable con threshold intermedio para rooms conocidos vs unknown.
- **Silero v4 vs v5**: API puede cambiar. Pin versión en requirements.
- **Thread safety**: `rs.early_command` lo escribe el worker task; `_check_vad_completion`
  lo lee en polling loop. Ambos en mismo event loop → OK sin lock.
- **Sin parser**: si `early_dispatch=false`, `rs.early_command` siempre None →
  siempre usa `medium_ms` → comportamiento equivalente al actual.

## Commit message sugerido

```
feat(pipeline): endpointing adaptativo con señal del parser streaming

Silero-VAD binario + silencio fijo de 300ms cortan mid-frase si el user
hace pausa de pensamiento. El parser streaming ya sabe si tenemos comando
completo — aprovecharlo para decidir cuánto silencio esperar.

- MultiRoomAudioLoop: _adaptive_endpoint_threshold:
    - parser ready_to_dispatch → 150ms silence (cerrar YA)
    - parser parcial o ausente → 300ms (actual)
    - sin señal → 500ms (esperar más)
- WhisperWakeDetector: _voice_prob(chunk) además del _is_speech binario
  (preparado para endpointers más sofisticados en el futuro).
- config/settings.yaml: rooms.endpointing.{enabled, short, medium, long}_ms.

Ganancia típica: -150ms en comandos completos, +200ms captura correcta
en comandos con pausas.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Checklist

- [ ] `WhisperWakeDetector._voice_prob` agregado + `_is_speech` wrapper
- [ ] `MultiRoomAudioLoop`: constructor + endpointing thresholds
- [ ] `_adaptive_endpoint_threshold` usando `rs.early_command`
- [ ] `_check_vad_completion` llama al adaptativo antes del clásico
- [ ] `settings.yaml` rooms.endpointing
- [ ] Tests adaptive threshold + endpoint trigger
- [ ] Regression tests
- [ ] Commit + push
