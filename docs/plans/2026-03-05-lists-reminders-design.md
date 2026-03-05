# Listas y Recordatorios — Diseño

**Fecha:** 2026-03-05
**Estado:** Aprobado
**Enfoque:** B — Dos módulos separados (`src/lists/` + `src/reminders/`)

## Resumen

Sistema completo de productividad por voz: listas genéricas (compras, tareas, etc.) con soporte personal/compartido, y recordatorios con recurrencia y acciones HA encadenadas. Triple interfaz: voz, API REST, y entidades HA.

## Requisitos

- Listas genéricas con nombre (compras, oficina, hogar, etc.)
- Ownership: personales por usuario (speaker ID) + compartidas por hogar
- Recordatorios one-shot y recurrentes (daily, weekdays, weekly, monthly)
- Delivery: TTS en zona del usuario (presencia BLE) + acciones HA encadenadas
- Interfaces: voz, API REST (dashboard), entidades HA (todo platform)
- Persistencia: SQLite (`./data/lists.db`)

## Modelo de Datos

### SQLite Schema

```sql
CREATE TABLE lists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_type TEXT NOT NULL,      -- "user" | "shared"
    owner_id TEXT,                 -- user_id si personal, NULL si shared
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE list_items (
    id TEXT PRIMARY KEY,
    list_id TEXT NOT NULL REFERENCES lists(id),
    text TEXT NOT NULL,
    completed INTEGER DEFAULT 0,
    added_by TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE reminders (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    trigger_at REAL NOT NULL,
    created_at REAL NOT NULL,
    recurrence TEXT,               -- "daily", "weekly:1", "monthly:15", "weekdays"
    recurrence_end REAL,
    ha_actions TEXT,               -- JSON array
    state TEXT DEFAULT 'active',   -- active | fired | cancelled
    last_fired_at REAL,
    fire_count INTEGER DEFAULT 0
);
```

### Dataclasses

```python
@dataclass
class ListItem:
    id: str
    list_id: str
    text: str
    completed: bool = False
    added_by: str | None = None
    created_at: float = 0.0

@dataclass
class UserList:
    id: str
    name: str
    owner_type: str  # "user" | "shared"
    owner_id: str | None
    items: list[ListItem] = field(default_factory=list)

@dataclass
class Reminder:
    id: str
    user_id: str
    text: str
    trigger_at: float
    recurrence: str | None = None
    ha_actions: list[dict] | None = None
    state: str = "active"
```

## Arquitectura de Módulos

```
src/lists/
├── __init__.py
├── list_manager.py       # CRUD listas + items (~200 líneas)
├── list_store.py         # SQLite persistence (~150 líneas)
└── ha_sync.py            # Sync con HA todo platform (~150 líneas)

src/reminders/
├── __init__.py
├── reminder_manager.py   # CRUD + delivery (TTS + HA actions) (~250 líneas)
├── reminder_scheduler.py # Asyncio loop, heapq para próximo disparo (~200 líneas)
├── recurrence.py         # Parser y cálculo de próxima fecha (~100 líneas)
└── reminder_store.py     # SQLite persistence (~150 líneas)
```

## Flujo de Datos

```
Voz: "agrega leche a la lista de compras"
  -> Router 7B clasifica como FAST_LIST
  -> ListManager.add_item("compras", "leche", user_id)
  -> ha_sync -> HA todo.add_item (si sync activo)
  -> TTS: "Listo, agregue leche"

Voz: "recuerdame a las 6 sacar la basura y enciende la luz del patio"
  -> Router 7B clasifica como FAST_REMINDER
  -> ReminderManager.create(text, trigger_at, ha_actions=[{light.turn_on, patio}])
  -> ReminderScheduler inserta en heapq
  -> A las 18:00:
      -> PresenceDetector -> zona del usuario
      -> TTS en esa zona: "Oye, recuerda sacar la basura"
      -> ha_client.call_service("light", "turn_on", "light.patio")

Voz: "todos los lunes recuerdame poner la ropa"
  -> ReminderManager.create(text, trigger_at=proximo_lunes_8am, recurrence="weekly:1")
  -> Cada lunes se dispara y recalcula next trigger
```

## Dispatcher Integration

Dos nuevos PathType:

```python
class PathType(StrEnum):
    # ... existentes ...
    FAST_LIST = "fast_list"
    FAST_REMINDER = "fast_reminder"
```

Ambos fast path (<300ms) — operaciones locales sin LLM.

## Comandos de Voz

### Listas

| Comando | Accion |
|---------|--------|
| "agrega leche a la lista de compras" | add_item("compras", "leche") |
| "quita el pan de la lista" | remove_item("compras", "pan") |
| "que hay en la lista de compras?" | get_items("compras") -> TTS lee items |
| "vacia la lista de compras" | clear_list("compras") |
| "crea una lista de la oficina" | create_list("oficina", owner_type="user") |
| "crea una lista compartida del hogar" | create_list("hogar", owner_type="shared") |
| "borra la lista de la oficina" | delete_list("oficina") |

### Recordatorios

| Comando | Accion |
|---------|--------|
| "recuerdame a las 5 sacar la basura" | one-shot hoy 17:00 |
| "recuerdame manana a las 8 ir al dentista" | one-shot manana 08:00 |
| "todos los lunes recuerdame poner la ropa" | recurrence="weekly:1" |
| "cada dia a las 7 recuerdame tomar la pastilla" | recurrence="daily" |
| "de lunes a viernes a las 9 recuerdame el standup" | recurrence="weekdays" |
| "el 15 de cada mes recuerdame pagar la luz" | recurrence="monthly:15" |
| "que recordatorios tengo?" | lista activos -> TTS |
| "cancela el recordatorio de la basura" | fuzzy match -> cancel |
| "que tengo pendiente hoy?" | reminders hoy + items sin completar |

## Recurrence Engine

Formato simplificado (4 tipos):

```python
class RecurrenceType(StrEnum):
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"       # weekly:1 = lunes, weekly:5 = viernes
    MONTHLY = "monthly"     # monthly:15 = dia 15
```

Una sola funcion: `next_trigger(last_trigger, recurrence) -> float`

## Delivery de Recordatorios

1. `PresenceDetector.get_user_zone(user_id)` -> zona actual
2. Si zona encontrada: TTS en zona + ejecutar ha_actions
3. Si no detectado: retry cada 5 min (max 3 intentos)
4. Despues de 3 intentos: marcar "missed"
5. Al detectar presencia: avisar "Tenias un recordatorio pendiente: {texto}"
6. Si recurrente: calcular next_trigger, actualizar DB
7. Si one-shot: marcar state="fired"

## API REST

```
GET    /api/lists                    # Todas las listas
GET    /api/lists/{id}/items         # Items de una lista
POST   /api/lists/{id}/items         # Agregar item
DELETE /api/lists/{id}/items/{item}  # Quitar item

GET    /api/reminders                # Todos los recordatorios
POST   /api/reminders                # Crear recordatorio
DELETE /api/reminders/{id}           # Cancelar recordatorio
```

## HA Integration

- Cada UserList se refleja como entidad `todo.kza_<nombre>` via HA todo platform
- Sync bidireccional: KZA -> HA y HA -> KZA (via WebSocket subscription)
- KZA es fuente de verdad; si sync falla, KZA sigue funcionando
- Recordatorios NO se sincronizan a HA

## Config (settings.yaml)

```yaml
lists:
  db_path: "./data/lists.db"
  default_list_name: "compras"
  ha_sync_enabled: true
  ha_entity_prefix: "todo.kza"

reminders:
  max_retries: 3
  retry_interval_seconds: 300
  missed_reminder_on_arrival: true
  default_time: "09:00"
  tts_prefix: "Oye, recuerda"
```

## Error Handling

| Escenario | Comportamiento |
|-----------|---------------|
| SQLite write falla | Log error, retry 1x, responder "No pude guardar" |
| HA sync falla | Log warning, continuar sin sync |
| TTS falla al disparar | Retry en 30s, max 2 intentos |
| Hora ambigua ("a las 5") | Asumir PM si <7, AM si >=7. Confirmar por voz |
| Recurrencia acumulando | Cleanup: eliminar fired/cancelled sin recurrencia despues de 30 dias |

## Testing

```
tests/unit/lists/
├── test_list_manager.py
├── test_list_store.py
└── test_ha_sync.py

tests/unit/reminders/
├── test_reminder_manager.py
├── test_reminder_scheduler.py
├── test_recurrence.py
└── test_reminder_store.py

tests/integration/
└── test_lists_reminders_e2e.py
```

Estimacion: ~80-100 tests nuevos.

## DI en main.py

```python
list_store = ListStore(config["lists"]["db_path"])
list_manager = ListManager(list_store, ha_client, config["lists"])
reminder_store = ReminderStore(config["lists"]["db_path"])
reminder_scheduler = ReminderScheduler(
    reminder_store, tts, presence_detector, ha_client, config["reminders"]
)
reminder_manager = ReminderManager(reminder_store, reminder_scheduler, config["reminders"])

dispatcher.register_list_manager(list_manager)
dispatcher.register_reminder_manager(reminder_manager)
asyncio.create_task(reminder_scheduler.start())
```

## Decisiones de Diseno

1. **SQLite sobre JSON**: recurrencia y queries temporales ("que tengo hoy?") se benefician de SQL
2. **Dos modulos separados**: lista (dato estatico) != recordatorio (evento temporal), concerns distintos
3. **Recurrencia simplificada**: 4 tipos cubren 95% de casos reales, extensible sin romper
4. **KZA como fuente de verdad**: HA es espejo, no depender de HA para funcionalidad core
5. **Fuzzy match con difflib**: para pocos items, no gastar GPU en embeddings
6. **Sleep hasta next trigger**: mas eficiente que polling periodico del AlertScheduler
