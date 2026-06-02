# Escritorio: grupos `light.grupo_*` + escenas indexadas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KZA adopta los 8 `light.grupo_*` por prefijo, indexa 5 escenas "modo" por el fast-path vectorial, y deja la config del escritorio apuntando a entidades reales.

**Architecture:** Tres cambios en `scripts/sync_ha_to_chroma.py` (importabilidad lazy, `is_group_entity` por prefijo, indexado de escenas curadas con `domain="scene"`) + correcciones en `config/settings.yaml`. Las escenas se indexan como comandos genéricos (`call_service_ws("scene","turn_on","scene.x")`), reusando el path probado de luces. Cero cambios en `src/main.py`.

**Tech Stack:** Python 3.13 (venv `/Users/yo/Documents/kza/.venv`), pytest, ChromaDB, BGE-M3.

**Spec:** `docs/superpowers/specs/2026-05-31-kza-adopt-grupos-escenas-design.md`

**Worktree:** `/Users/yo/Documents/kza-wt-escritorio` (branch `feat/escritorio-grupos-escenas`, base `origin/main` 224912d).

**Nota de entorno:** correr pytest con `/Users/yo/Documents/kza/.venv/bin/python -m pytest` (el `python3` del sistema es 3.9 y rompe). Las pruebas unitarias NO acceden a HA (funciones puras + YAML). El smoke `--dry-run` contra HA real va en la fase de verificación/deploy (necesita token).

---

### Task 1: Hacer el módulo importable sin `.env` (lazy load)

**Problema:** `scripts/sync_ha_to_chroma.py:45-52` lee `.env` y `os.environ["HOME_ASSISTANT_*"]` a nivel de módulo → `import` crashea sin `.env` (KeyError/FileNotFoundError), impide TDD de las funciones puras. Refactor mínimo: tolerar `.env` ausente y usar `.get`. Runtime idéntico cuando `.env` existe.

**Files:**
- Modify: `scripts/sync_ha_to_chroma.py:45-52`
- Test: `tests/unit/scripts/test_sync_ha_to_chroma.py` (crear) + `tests/unit/scripts/__init__.py` (crear vacío)

- [ ] **Step 1: Crear el test de importabilidad (falla)**

`tests/unit/scripts/__init__.py`: archivo vacío.

`tests/unit/scripts/test_sync_ha_to_chroma.py`:
```python
"""Tests de las funciones puras de scripts/sync_ha_to_chroma.py (sin acceso a HA)."""
import importlib


def test_module_imports_without_env(monkeypatch):
    # Sin HOME_ASSISTANT_* y sin .env, el import no debe crashear.
    monkeypatch.delenv("HOME_ASSISTANT_URL", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    mod = importlib.import_module("scripts.sync_ha_to_chroma")
    assert hasattr(mod, "is_group_entity")
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Correr el test (debe fallar)**

Run: `cd /Users/yo/Documents/kza-wt-escritorio && /Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -q`
Expected: FAIL (FileNotFoundError `.env` o KeyError `HOME_ASSISTANT_URL` al importar).

- [ ] **Step 3: Refactor lazy/tolerante**

Reemplazar `scripts/sync_ha_to_chroma.py:45-52` (el bloque `for line in (ROOT/".env")...` + las 2 asignaciones `HA_URL`/`HA_TOKEN`) por:
```python
_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)

HA_URL = os.environ.get("HOME_ASSISTANT_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")
```

- [ ] **Step 4: Correr el test (debe pasar)**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add tests/unit/scripts/ scripts/sync_ha_to_chroma.py
git commit -m "refactor(sync): carga lazy de .env/HA token (habilita TDD de funciones puras)"
```

---

### Task 2: `is_group_entity` por prefijo `light.grupo_` + borrar `KNOWN_GROUPS`

**Files:**
- Modify: `scripts/sync_ha_to_chroma.py` (borrar `KNOWN_GROUPS` 96-105; reescribir `is_group_entity` 119-120)
- Test: `tests/unit/scripts/test_sync_ha_to_chroma.py`

- [ ] **Step 1: Test (falla)** — añadir a `test_sync_ha_to_chroma.py`:
```python
from scripts.sync_ha_to_chroma import is_group_entity


def test_is_group_entity_prefix():
    assert is_group_entity("light.grupo_cocina", "Cocina") is True
    assert is_group_entity("light.grupo_escritorio", "Escritorio") is True
    # Modelo viejo Hue y Z2M one-off NO son grupos del modelo nuevo:
    assert is_group_entity("light.cocina", "Cocina") is False
    assert is_group_entity("light.escritorio_2", "Escritorio 4") is False
    assert is_group_entity("light.hogar", "Hogar") is False
    # Bombita individual:
    assert is_group_entity("light.l1", "L1") is False
```

- [ ] **Step 2: Correr (falla)** — `light.cocina`/`light.escritorio_2` darían True con el `KNOWN_GROUPS` viejo.
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py::test_is_group_entity_prefix -q`
Expected: FAIL.

- [ ] **Step 3: Implementar** — borrar el bloque `KNOWN_GROUPS = { ... }` (96-105) y reescribir:
```python
def is_group_entity(entity_id: str, friendly_name: str) -> bool:
    return entity_id.startswith("light.grupo_")
```
(Dejar `GROUP_PREFIX_MAP` y `decode_individual` intactos — son del feature de bombita individual.)

- [ ] **Step 4: Correr (pasa)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -q`
Expected: PASS (todos).

- [ ] **Step 5: Commit**
```bash
git add scripts/sync_ha_to_chroma.py tests/unit/scripts/test_sync_ha_to_chroma.py
git commit -m "feat(sync): detectar grupos por prefijo light.grupo_ (8 cuartos, borra KNOWN_GROUPS)"
```

---

### Task 3: `build_scene_specs()` — escenas curadas (anti-colisión)

**Files:**
- Modify: `scripts/sync_ha_to_chroma.py` (añadir `SceneSpec`, `SCENE_ALLOWLIST`, `SCENE_PHRASES`, `build_scene_specs` cerca de `CommandSpec`, ~línea 145)
- Test: `tests/unit/scripts/test_sync_ha_to_chroma.py`

- [ ] **Step 1: Test (falla)** — añadir:
```python
from scripts.sync_ha_to_chroma import build_scene_specs

_ROOM_WORDS = {"cocina", "living", "escritorio", "baño", "bano", "hall",
               "cuarto", "balcón", "balcon", "escalera", "pasillo"}


def test_build_scene_specs_shape():
    specs = build_scene_specs()
    ids = {s.entity_id for s in specs}
    assert ids == {"scene.cine", "scene.lectura", "scene.calida",
                   "scene.fria", "scene.relax"}
    for s in specs:
        assert len(s.phrases) >= 3
        assert s.value_label and "." not in s.value_label


def test_scene_phrases_no_room_collision():
    # Las frases de escena NO deben mencionar un cuarto (evita pisar el
    # color_temp por-cuarto: 'poné la cocina fría' → grupo_cocina, no scene.fria).
    for s in build_scene_specs():
        for phrase in s.phrases:
            low = phrase.lower()
            assert not any(room in low.split() for room in _ROOM_WORDS), \
                f"frase de escena con cuarto: {phrase!r}"
```

- [ ] **Step 2: Correr (falla)** — `ImportError: build_scene_specs`.
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -k scene -q`
Expected: FAIL.

- [ ] **Step 3: Implementar** — insertar tras la definición de `CommandSpec` (~línea 145):
```python
@dataclass
class SceneSpec:
    """Escena global 'modo' indexada como comando (Approach B)."""
    entity_id: str          # scene.cine
    value_label: str        # cine
    phrases: list[str]


SCENE_ALLOWLIST = ["cine", "lectura", "calida", "fria", "relax"]

SCENE_PHRASES = {
    "cine":    ["modo cine", "poné modo cine", "activá la escena cine",
                "ponela en cine", "escena cine"],
    "lectura": ["modo lectura", "escena lectura", "activá lectura",
                "poné modo lectura", "luz de lectura"],
    "calida":  ["modo cálido", "escena cálida", "poné todo cálido",
                "luz cálida en toda la casa", "modo cálida"],
    "fria":    ["modo fresco", "escena fría", "poné todo frío",
                "luz fría en toda la casa", "modo frío"],
    "relax":   ["modo relax", "escena relax", "activá relax",
                "poné modo relax", "ponela en relax"],
}


def build_scene_specs() -> list[SceneSpec]:
    """Specs de las 5 escenas globales 'modo' con frases curadas (sin LLM)."""
    return [
        SceneSpec(entity_id=f"scene.{name}", value_label=name,
                  phrases=list(SCENE_PHRASES[name]))
        for name in SCENE_ALLOWLIST
    ]
```

- [ ] **Step 4: Correr (pasa)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -k scene -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/sync_ha_to_chroma.py tests/unit/scripts/test_sync_ha_to_chroma.py
git commit -m "feat(sync): build_scene_specs — 5 escenas modo con frases curadas (anti-colisión)"
```

---

### Task 4: Construir docs de escena (`domain="scene"`) + cablear en `main()`

**Files:**
- Modify: `scripts/sync_ha_to_chroma.py` (añadir `build_scene_documents`; bloque de indexado en `main()` tras el loop de luces, antes del `logger.info("Done...")`)
- Test: `tests/unit/scripts/test_sync_ha_to_chroma.py`

- [ ] **Step 1: Test (falla)** — añadir:
```python
import json as _json
from scripts.sync_ha_to_chroma import build_scene_documents, build_scene_specs


def test_build_scene_documents_metadata():
    docs = build_scene_documents(build_scene_specs())
    # Un doc por frase (5 escenas × ≥3 frases):
    assert len(docs) >= 15
    by_eid = {}
    for doc_id, phrase, meta in docs:
        assert meta["domain"] == "scene"
        assert meta["service"] == "turn_on"
        assert meta["entity_id"].startswith("scene.")
        assert meta["capability"] == "scene"
        assert _json.loads(meta["service_data"]) == {}
        assert meta["is_group"] is False
        assert isinstance(phrase, str) and phrase
        assert doc_id.endswith(tuple("0123456789"))
        by_eid.setdefault(meta["entity_id"], 0)
        by_eid[meta["entity_id"]] += 1
    assert by_eid["scene.cine"] >= 3
    # cache_key estable entre llamadas (idempotencia incremental):
    docs2 = build_scene_documents(build_scene_specs())
    assert [d[0] for d in docs] == [d[0] for d in docs2]
```

- [ ] **Step 2: Correr (falla)** — `ImportError: build_scene_documents`.
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -k scene_documents -q`
Expected: FAIL.

- [ ] **Step 3: Implementar `build_scene_documents`** — tras `build_scene_specs`:
```python
def build_scene_documents(specs: list[SceneSpec]) -> list[tuple[str, str, dict]]:
    """(doc_id, phrase, metadata) por frase. Metadata genérica → call_service_ws("scene","turn_on")."""
    out: list[tuple[str, str, dict]] = []
    for spec in specs:
        key = cache_key(spec.entity_id, spec.value_label, None, "scene", "activate")
        for j, phrase in enumerate(spec.phrases):
            meta = {
                "entity_id": spec.entity_id,
                "friendly_name": spec.value_label,
                "area": "",
                "domain": "scene",
                "service": "turn_on",
                "capability": "scene",
                "value_label": spec.value_label,
                "service_data": json.dumps({}),
                "is_group": False,
                "cache_key": key,
            }
            out.append((f"{key}_{j}", phrase, meta))
    return out
```

- [ ] **Step 4: Correr (pasa)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -k scene_documents -q`
Expected: PASS.

- [ ] **Step 5: Cablear en `main()`** — insertar antes de `logger.info(f"Done. Stats: {stats}")` (~línea 526):
```python
    # ── Escenas globales 'modo' (Approach B: curadas, sin LLM) ──────────────
    if not args.only_individual and not args.entity:
        scene_added = 0
        for doc_id, phrase, meta in build_scene_documents(build_scene_specs()):
            if meta["cache_key"] in existing_keys and not args.force:
                continue
            logger.info(f"[scene] {meta['entity_id']} ← {phrase!r}")
            if args.dry_run:
                continue
            collection.add(
                ids=[doc_id],
                embeddings=[embedder.encode(phrase).tolist()],
                documents=[phrase],
                metadatas=[meta],
            )
            scene_added += 1
        logger.info(f"Escenas indexadas: {scene_added}")
```

- [ ] **Step 6: Smoke de integración (mock collection/embedder)** — añadir test:
```python
def test_scene_indexing_block_persists(monkeypatch):
    """build_scene_documents → collection.add con embeddings (sin tocar HA/LLM)."""
    docs = build_scene_documents(build_scene_specs())
    added = []

    class FakeColl:
        def add(self, ids, embeddings, documents, metadatas):
            added.append((ids[0], metadatas[0]["domain"], metadatas[0]["service"]))

    class FakeEmb:
        def encode(self, p):
            class _A:
                def tolist(self_): return [0.0, 0.1, 0.2]
            return _A()

    coll, emb = FakeColl(), FakeEmb()
    for doc_id, phrase, meta in docs:
        coll.add(ids=[doc_id], embeddings=[emb.encode(phrase).tolist()],
                 documents=[phrase], metadatas=[meta])
    assert added and all(d == "scene" and s == "turn_on" for _, d, s in added)
```
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_sync_ha_to_chroma.py -q`
Expected: PASS (todos).

- [ ] **Step 7: Commit**
```bash
git add scripts/sync_ha_to_chroma.py tests/unit/scripts/test_sync_ha_to_chroma.py
git commit -m "feat(sync): indexar 5 escenas modo con domain=scene en main() (Approach B)"
```

---

### Task 5: Config real del escritorio + `default_light` → `grupo_*`

**Files:**
- Modify: `config/settings.yaml` (rooms: living, hall, cocina, escritorio, bano)
- Test: `tests/unit/scripts/test_settings_rooms_entities.py` (crear)

- [ ] **Step 1: Test (falla)** — `tests/unit/scripts/test_settings_rooms_entities.py`:
```python
"""La config de rooms apunta a entidades reales (verificado contra /api/states 2026-05-31)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _rooms():
    cfg = yaml.safe_load((ROOT / "config/settings.yaml").read_text())
    return cfg["rooms"]


def test_default_light_uses_grupo():
    r = _rooms()
    assert r["living"]["default_light"] == "light.grupo_living"
    assert r["cocina"]["default_light"] == "light.grupo_cocina"
    assert r["bano"]["default_light"] == "light.grupo_bano"
    assert r["escritorio"]["default_light"] == "light.grupo_escritorio"
    assert r["hall"]["default_light"] == "light.grupo_pasillo"  # hall≈pasillo


def test_escritorio_real_sensors():
    e = _rooms()["escritorio"]
    assert e["motion_sensor"] == "binary_sensor.escritorio_motion"
    assert e["temperature_sensor"] == "sensor.blink_escritorio_temperature"


def test_escritorio_no_phantom_devices():
    # climate/media/tts inexistentes en HA → no deben estar activos (comentados).
    e = _rooms()["escritorio"]
    assert "default_climate" not in e
    assert "default_media_player" not in e
    assert "tts_speaker" not in e
```

- [ ] **Step 2: Correr (falla)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_settings_rooms_entities.py -q`
Expected: FAIL.

- [ ] **Step 3: Editar `config/settings.yaml`** — aplicar exactamente:

`living` (~1115): `default_light: "light.living"` → `default_light: "light.grupo_living"`

`hall` (~1134): `default_light: "light.hall"` → `default_light: "light.grupo_pasillo"   # hall≈pasillo; light.hall no existe en HA (2026-05-31)`

`cocina` (~1149): `default_light: "light.cocina"` → `default_light: "light.grupo_cocina"`

`bano` (~1191): `default_light: "light.bano"` → `default_light: "light.grupo_bano"`

`escritorio` (~1173-1186): reemplazar el bloque:
```yaml
    default_light: "light.escritorio_2"   # grupo Z2M 2026-05-31 (slug light.escritorio reservado por huérfano Hue; volver a light.escritorio al liberarlo)
    default_climate: "climate.escritorio_ac"
    default_media_player: "media_player.escritorio_monitor"
    motion_sensor: "binary_sensor.motion_escritorio"
    temperature_sensor: "sensor.temperature_escritorio"
    tts_speaker: "media_player.escritorio_speaker"
```
por:
```yaml
    default_light: "light.grupo_escritorio"   # grupo HA uniforme (= Alexa); 2026-05-31
    motion_sensor: "binary_sensor.escritorio_motion"
    temperature_sensor: "sensor.blink_escritorio_temperature"   # de la cámara Blink (no hay sensor dedicado)
    # TODO: el escritorio NO tiene estos devices físicamente (no existen en HA al 2026-05-31).
    # Descomentar y apuntar a entidades reales cuando se agreguen:
    # default_climate: "climate.escritorio_ac"
    # default_media_player: "media_player.escritorio_monitor"
    # tts_speaker: "media_player.escritorio_speaker"
```

- [ ] **Step 4: Correr (pasa)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/test_settings_rooms_entities.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add config/settings.yaml tests/unit/scripts/test_settings_rooms_entities.py
git commit -m "fix(config): default_light→grupo_* (5 rooms) + escritorio a entidades reales (comenta fantasmas)"
```

---

### Task 6: Verificación local + smoke `--dry-run`

- [ ] **Step 1: Suite completa de las funciones tocadas**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/scripts/ -q`
Expected: PASS (todos).

- [ ] **Step 2: Regresión del sync/nlu adyacente (no romper lo existente)**
Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/vectordb tests/unit/nlu -q`
Expected: PASS (o el baseline previo sin regresión nueva).

- [ ] **Step 3: Smoke `--dry-run` contra HA real** (necesita token; correr en server o exportando `HOME_ASSISTANT_URL`/`HOME_ASSISTANT_TOKEN`):
```bash
HOME_ASSISTANT_URL=http://localhost:8123 HOME_ASSISTANT_TOKEN=<tok> \
  /Users/yo/Documents/kza/.venv/bin/python scripts/sync_ha_to_chroma.py --dry-run 2>&1 | tail -40
```
Expected: lista los 8 `light.grupo_*` como GROUP + `[scene]` para las 5 escenas. 0 `light.living`/`light.escritorio_2` seleccionados.

---

## Deploy (fase aparte, con OK del usuario)

1. Parar `kza-voice` (libera lock de chroma) — `ssh kza ...`.
2. Llevar la rama al server (`git fetch` de la laptop / merge a main + el server pullea — confirmar mecánica).
3. **Preflight VRAM** cuda:1 (>1500MB libres) antes de cargar el embedder.
4. Re-sync `--wipe` con la key sourceada: `source /home/kza/secrets/llama-api-key.env` + `--vllm-url http://127.0.0.1:8101/v1 --vllm-model <gguf Q4_K_M> --embedder-device cpu --wipe`.
5. Relanzar `kza-voice`; verificar 2/2 streams.
6. Validación por voz: "prendé el escritorio", "poné la cocina fría" (→ grupo_cocina), "modo cine", "modo lectura", "modo relax".
7. (Lado-HA, posterior, con OK) deshabilitar los grupos Hue.

---

## Self-Review

- **Spec coverage:** Parte 1 → Task 2 (+ smoke Task 6). Parte 2 (escenas B) → Tasks 3-4. Parte 3 (config) → Task 5. Importabilidad para TDD → Task 1. Verificación/deploy → Task 6 + sección Deploy. ✓
- **Placeholder scan:** sin TBD/TODO de plan (los `# TODO` en settings.yaml son contenido intencional). ✓
- **Type consistency:** `SceneSpec(entity_id, value_label, phrases)` usado consistente en Tasks 3-4; `build_scene_specs`/`build_scene_documents`/`cache_key`/`is_group_entity` con firmas estables. ✓
