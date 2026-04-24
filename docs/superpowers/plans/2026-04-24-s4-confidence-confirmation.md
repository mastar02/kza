# S4 — Confidence-based confirmation

**Effort**: 🟡 2-3 h
**Depende**: nada
**Branch sugerido**: `feat/s4-confidence-confirmation`

## Objetivo

Si un comando tiene confidence baja o apunta a una acción sensible (cerrar
persianas de noche, apagar sistema), pedir confirmación antes de ejecutar.
Guardrail contra falsos positivos y ejecuciones ambiguas.

## Arquitectura

```
PartialCommand (ya existe) extendido con confidence: float ∈ [0, 1]
  │
  └─ RequestRouter:
     ├─ confidence >= threshold Y (intent reversible) → ejecutar normal
     ├─ confidence >= threshold Y intent sensible      → ejecutar + audit log
     ├─ confidence <  threshold Y intent reversible    → ejecutar con "heurística" + speak confirmación post-facto ("Apagué la luz por las dudas, avisame si no querías")
     └─ confidence <  threshold Y intent sensible      → pedir confirmación ("¿Cerrar las persianas del living?") + modo listening corto (5s)
```

## Confidence computation

En `parse_partial_command`, calcular un score compuesto:

```python
confidence = min(
    wake_score,       # fuzzy_ratio de wake word detectado (0-1)
    intent_score,     # 1.0 si intent encontrado, 0.0 si None
    entity_score,     # 1.0 si entity encontrada, 0.0 si None
) * boost_if_room * boost_if_slots
```

Donde `boost_if_room` = 1.1 si `room != None`, else 1.0 (clamp a 1.0).
`boost_if_slots` = 1.05 si hay slots, else 1.0.

Si falta alguno de los componentes básicos (wake, intent, entity), confidence
= 0 → parser no está `ready_to_dispatch`.

## Archivos a modificar

### `src/nlu/command_grammar.py`

Agregar campo al dataclass:

```python
@dataclass
class PartialCommand:
    ...
    confidence: float = 0.0

    def ready_to_dispatch(self) -> bool:
        return self.intent is not None and self.entity is not None

    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        return self.confidence >= threshold
```

Calcularlo en `parse_partial_command`:

```python
def parse_partial_command(text: str) -> PartialCommand:
    ...
    pc.confidence = _compute_confidence(pc)
    return pc


def _compute_confidence(pc: PartialCommand) -> float:
    # Base: requieren intent y entity.
    if pc.intent is None or pc.entity is None:
        return 0.0
    # Base score = 0.7 (mínimo si intent + entity detectados).
    score = 0.7
    # Wake word presente → +0.15
    if pc.has_wake:
        score += 0.15
    # Room presente → +0.10
    if pc.room is not None:
        score += 0.10
    # Slots presentes → +0.05
    if pc.slots:
        score += 0.05
    return min(score, 1.0)
```

### `src/nlu/__init__.py` — actualizar exports si necesario.

### Nuevo helper: `src/nlu/sensitive_actions.py`

```python
"""
Lista de combinaciones intent+entity consideradas "sensibles" — requieren
confirmación o rationalizan audit logs extra.
"""
from __future__ import annotations

SENSITIVE_COMBOS: set[tuple[str, str]] = {
    # (intent, entity)
    ("turn_off", "climate"),     # apagar aire en invierno
    ("set_cover_position", "cover"),  # persianas (ambiguous up/down)
    ("turn_off", "media_player"),      # cortar entretenimiento
    # Extender con más combos según use case.
}


def is_sensitive(intent: str | None, entity: str | None) -> bool:
    if not intent or not entity:
        return False
    return (intent, entity) in SENSITIVE_COMBOS
```

### `src/pipeline/request_router.py`

Importar y usar:

```python
from src.nlu.sensitive_actions import is_sensitive
from src.nlu.command_grammar import parse_partial_command
```

En `process_command` (o en la rama orchestrated/legacy después de tener `text`):

```python
pc = parse_partial_command(text)
threshold = self.confidence_threshold  # nuevo campo

if pc.confidence < threshold:
    if is_sensitive(pc.intent, pc.entity):
        # Pedir confirmación explícita.
        question = self._build_confirmation_question(pc)
        await self.response_handler.speak(question)
        # Modo listening corto esperando sí/no (via follow_up).
        logger.info(
            f"Low confidence ({pc.confidence:.2f}) + sensitive — pidiendo confirmación"
        )
        # Marcamos el comando como pending; el próximo utterance ("sí"/"no")
        # lo resuelve. Implementación simple: dejar que FollowUpMode lo maneje.
        result["pending_confirmation"] = True
        result["pending_pc"] = pc
        return result
    else:
        # Reversible (ej. luz): ejecutar igual, logeando la incertidumbre.
        logger.info(f"Low confidence ({pc.confidence:.2f}) — ejecutando por ser reversible")
```

Agregar `confidence_threshold` al constructor.

### `src/main.py`

```python
confidence_cfg = config.get("orchestrator", {}).get("confidence", {})
request_router = RequestRouter(
    ...
    confidence_threshold=confidence_cfg.get("threshold", 0.75),
)
```

### `config/settings.yaml`

```yaml
orchestrator:
  ...
  confidence:
    # Debajo de este threshold, comandos sensibles piden confirmación;
    # los reversibles se ejecutan pero logean la incertidumbre.
    threshold: 0.75
    confirmation_timeout_s: 5   # cuánto esperamos la respuesta sí/no
```

## Helper: construir pregunta de confirmación

```python
def _build_confirmation_question(self, pc: PartialCommand) -> str:
    action_verb = {
        "turn_off": "apagar",
        "turn_on": "prender",
        "set_cover_position": "mover",
    }.get(pc.intent, "hacer")
    entity_name = {
        "light": "la luz",
        "climate": "el aire",
        "cover": "las persianas",
        "fan": "el ventilador",
        "media_player": "la música",
    }.get(pc.entity, "el dispositivo")
    if pc.room:
        room_name = {
            "escritorio": "del escritorio",
            "living": "del living",
            "cocina": "de la cocina",
            "bano": "del baño",
            "hall": "del hall",
            "cuarto": "del cuarto",
        }.get(pc.room, "")
        return f"¿Querés {action_verb} {entity_name} {room_name}? Decí sí o no."
    return f"¿Querés {action_verb} {entity_name}? Decí sí o no."
```

## Validación

1. Comando ambiguo: decí "nexa apagá" (solo verbo, sin entidad). Whisper puede
   captar variantes. Verificar que el parser no dispare (confidence < 0.75).
2. Comando sensible con confidence bajo: decí algo que suene como "cerrá las
   persianas del living" pero con ruido. Debería preguntar antes de ejecutar.
3. Comando claro: "nexa prendé la luz del escritorio" → confidence alta,
   ejecuta sin preguntar.

## Test unitario

`tests/unit/nlu/test_confidence.py`:
```python
import pytest
from src.nlu.command_grammar import parse_partial_command
from src.nlu.sensitive_actions import is_sensitive


def test_confidence_full_command_high():
    pc = parse_partial_command("nexa apagá la luz del escritorio al 50 por ciento")
    assert pc.confidence >= 0.95


def test_confidence_no_room_lower():
    pc = parse_partial_command("nexa apagá la luz")
    assert 0.8 <= pc.confidence < 0.95


def test_confidence_zero_when_incomplete():
    pc = parse_partial_command("nexa")
    assert pc.confidence == 0.0


@pytest.mark.parametrize("intent,entity,expected", [
    ("turn_off", "climate", True),
    ("turn_on", "light", False),
    ("set_cover_position", "cover", True),
    (None, "light", False),
])
def test_sensitive_actions(intent, entity, expected):
    assert is_sensitive(intent, entity) == expected
```

## Edge cases

- **Confidence calibration**: los scores son heurísticos, no probabilidades
  reales. Si el user siente que pregunta demasiado, bajar threshold a 0.65.
- **Confirmation loop infinito**: si el "sí" del usuario también tiene baja
  confidence, NO recursivarlo. Aceptar cualquier `sí`/`dale`/`ok` que caiga en
  listening mode post-confirm sin volver a chequear confidence.
- **Timeout de confirmación**: si el usuario no responde en 5s, cancelar +
  speak "OK, no hago nada".

## Commit message sugerido

```
feat(nlu): confidence score + confirmation para comandos sensibles

Guardrail estilo Alexa "Did you mean X?" para comandos con baja confianza
o acciones de impacto (cerrar persianas, apagar aire).

- command_grammar.PartialCommand.confidence: score compuesto de
  wake/intent/entity/room/slots. 0.0 si falta básico, 1.0 si completo.
- nlu/sensitive_actions.py: SENSITIVE_COMBOS con (intent, entity) que
  requieren confirmación explícita.
- RequestRouter: check confidence < threshold. Sensible → pregunta +
  listening corto. Reversible → ejecuta + log de incertidumbre.
- settings.yaml: orchestrator.confidence.{threshold, timeout}.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Checklist

- [ ] `PartialCommand.confidence` + `is_high_confidence()` + `_compute_confidence`
- [ ] Nuevo `src/nlu/sensitive_actions.py`
- [ ] `RequestRouter.process_command` check confidence + confirmación
- [ ] `_build_confirmation_question` helper
- [ ] `settings.yaml` orchestrator.confidence
- [ ] Tests confidence + sensitive
- [ ] Regression tests
- [ ] Commit + push
