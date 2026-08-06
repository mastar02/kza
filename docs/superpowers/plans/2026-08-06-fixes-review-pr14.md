# Fixes de la Review Profunda del PR #14 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar los defectos que la review profunda del PR #14 (2026-08-06, cinco agentes) encontró vivos en main: un misroute que come comandos domóticos, observabilidad nula del clima sin datos, huérfanos FLAC que sobreviven al TTL, bloqueos del event loop en el voice path, un exit code mentiroso en el tool de WER, y los gaps de tests que dejarían regresar todo lo anterior.

**Architecture:** Siete tareas independientes, ordenadas por urgencia: las dos primeras corrigen comportamiento visible en producción HOY (el misroute y la ceguera del clima), las dos siguientes endurecen la campaña `keep_audio` en curso (event loop y huérfanos), después tools, tests y limpieza de comentarios. Cada tarea es commiteable y deployable por separado.

**Tech Stack:** Python 3.13, pytest, aiosqlite, soundfile, asyncio.

## Global Constraints

- Python de tests: `/Users/yo/Documents/kza/.venv/bin/python -m pytest` (el python3 del sistema es 3.9 y rompe).
- `async/await` para todo I/O; NUNCA bloquear el event loop (CLAUDE.md).
- Sin imports relativos; `from src.modulo import X`.
- `logging.getLogger(__name__)`, nunca `print()` en `src/` (en `tools/` el print es precedente aceptado).
- Mensajes de voz en español; código y logs en inglés (los módulos de `src/ambient/` ya loguean en español — seguir la convención local del archivo que se toca).
- Config solo en `config/settings.yaml`.
- TDD: test rojo antes de la implementación, en cada tarea.
- ⚠️ La campaña `keep_audio` corre en producción hasta ~2026-08-08 09:19. Este plan NO toca el server; el deploy se coordina aparte (el settings.yaml del server tiene drift declarado de una línea).

---

### Task 1: Los fragmentos de cláusula de clima dejan de comerse comandos domóticos

El bug crítico de la review: `WEATHER_KEYWORDS` contiene cláusulas sueltas (`"hace calor"`, `"hace frío"`, `"llueve"`, `"lloviendo"`, `"va a llover"`) que corren ANTES que la domótica, y el guard de adyacencia solo veta verbos pegados a sustantivos de clima. `"prendé la luz que hace frío"` → `FAST_WEATHER`: escuchás el pronóstico y la luz no prende (verificado contra main ejecutando `_classify_request`).

**Fix:** partir `WEATHER_KEYWORDS` en dos tiers. Las consultas explícitas (`"qué tiempo hace"`, `"pronóstico"`, `"está el clima"`, …) rutean como hoy. Los *fragmentos de cláusula* solo rutean a clima si NO hay ningún verbo de `DOMOTICS_KEYWORDS` en el texto, o si el texto es pregunta (mismo segundo signal `is_question` que ya usa el guard). Efecto colateral deliberado: los 4 casos del xfail estricto `test_climate_commands_with_interposed_adverb_misroute_to_weather` pasan a rutear bien (todos contienen un fragmento + un verbo), así que el xfail se convierte en test positivo.

**Files:**
- Modify: `src/orchestrator/dispatcher.py` (lista `WEATHER_KEYWORDS` ~línea 417-431; bloque de clima en `_classify_request` ~línea 730-737)
- Test: `tests/unit/orchestrator/test_dispatcher_world_routing.py`

**Interfaces:**
- Produces: atributo de clase `RequestDispatcher.WEATHER_CLAUSE_FRAGMENTS: frozenset[str]` (subconjunto literal de `WEATHER_KEYWORDS`).
- Consumes: `_kw_match(keyword, text_lower)` (ya existe, módulo-level en dispatcher.py:91).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/unit/orchestrator/test_dispatcher_world_routing.py`, después de `test_domotics_climate_adjacency_guard_finding_3`:

```python
@pytest.mark.parametrize("text", [
    # Verificados contra main en la review del PR #14 (2026-08-06): todos
    # ruteaban FAST_WEATHER — el usuario escuchaba el pronóstico y la
    # acción pedida nunca se ejecutaba (fallo domótico silencioso). El
    # discriminante: la cláusula de clima es JUSTIFICACIÓN del comando,
    # no consulta. Regla: un fragmento de WEATHER_CLAUSE_FRAGMENTS solo
    # rutea a clima si no hay ningún verbo domótico en el texto (o si es
    # pregunta).
    "prendé la luz que hace frío",
    "apagá la luz, hace calor",
    "prendé el ventilador que hace calor",
    "prendé la estufa, hace frío",
    "cerrá la persiana que llueve",
    "cerrá las ventanas que está lloviendo",
    "prendé la luz del living porque va a llover",
    "encender la luz, hace frío",
    "apagá todo, hace calor",
])
def test_non_climate_commands_with_weather_clause_stay_domotics(dispatcher, text):
    path, _ = dispatcher._classify_request(text)
    assert path == PathType.FAST_DOMOTICS


def test_weather_clause_fragments_is_a_subset_of_weather_keywords():
    # El tier de fragmentos no puede inventar keywords: si alguien saca un
    # fragmento de WEATHER_KEYWORDS y olvida el frozenset, esto lo detecta.
    assert RequestDispatcher.WEATHER_CLAUSE_FRAGMENTS <= set(
        RequestDispatcher.WEATHER_KEYWORDS
    )
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_world_routing.py -k "weather_clause or stay_domotics" -v`
Expected: FAIL — los 9 casos dan `FAST_WEATHER`, y el subset test da `AttributeError: WEATHER_CLAUSE_FRAGMENTS`.

- [ ] **Step 3: Implementar el tier de fragmentos**

En `src/orchestrator/dispatcher.py`, inmediatamente después del cierre de la lista `WEATHER_KEYWORDS` (línea ~431):

```python
    # Subconjunto de WEATHER_KEYWORDS que son FRAGMENTOS DE CLÁUSULA, no
    # consultas completas: aparecen igual de seguido como justificación
    # colgada de un comando ("prendé la estufa, hace frío") que como
    # consulta de clima. Review PR #14 (2026-08-06): con un tier único
    # esos comandos ruteaban FAST_WEATHER y la acción nunca se ejecutaba
    # (fallo domótico silencioso, verificado contra main). Un fragmento
    # solo rutea a clima si NINGÚN verbo de DOMOTICS_KEYWORDS aparece en
    # el texto, o si el texto es pregunta ("¿tengo que prender el clima o
    # hace calor afuera?" sigue siendo clima). Las consultas explícitas
    # ("qué tiempo hace", "pronóstico", "está el clima") NO llevan veto:
    # "está el clima bien, no hace falta prender nada" sigue en clima
    # aunque "prende" matchee como substring de "prender".
    WEATHER_CLAUSE_FRAGMENTS = frozenset({
        "hace calor", "hace frío", "hace frio",
        "llueve", "va a llover", "lloviendo",
    })
```

Y reemplazar el bloque de clima en `_classify_request` (líneas 734-737, dejando intactos `is_question` y `climate_command_adjacent` de arriba):

```python
        if not climate_command_adjacent:
            domotics_verb = any(
                _kw_match(k, text_lower) for k in self.DOMOTICS_KEYWORDS
            )
            for keyword in self.WEATHER_KEYWORDS:
                if keyword in text_lower:
                    if (not is_question and domotics_verb
                            and keyword in self.WEATHER_CLAUSE_FRAGMENTS):
                        # "prendé la luz que hace frío": la cláusula es
                        # justificación de un comando — dejar que el loop
                        # de DOMOTICS_KEYWORDS de abajo lo capture.
                        continue
                    return PathType.FAST_WEATHER, Priority.HIGH
```

- [ ] **Step 4: Correr los tests nuevos y verificar que pasan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_world_routing.py -k "weather_clause or stay_domotics" -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Convertir el xfail que ahora XPASSea**

Correr el archivo entero: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/orchestrator/test_dispatcher_world_routing.py -v`. Los 4 casos de `test_climate_commands_with_interposed_adverb_misroute_to_weather` ahora fallan con XPASS(strict). Convertirlo en test positivo: borrar el decorador `@pytest.mark.xfail(...)` completo, renombrar a `test_climate_commands_with_interposed_adverb_route_to_domotics`, y reemplazar el docstring por:

```python
    """Comandos de AC con una palabra entre el verbo y el sustantivo.

    Misroute conocido desde 2026-08-04 (commit a513108 + revert ef651e2):
    el guard de adyacencia no los cubría y ganaba el keyword de clima.
    Cerrado por el veto de WEATHER_CLAUSE_FRAGMENTS (review PR #14,
    2026-08-06): todos estos casos contienen un fragmento de cláusula +
    un verbo domótico, así que el fragmento ya no captura y el loop de
    DOMOTICS_KEYWORDS resuelve.
    """
```

El assert ya es `assert path == PathType.FAST_DOMOTICS` — queda igual.

- [ ] **Step 6: Correr el archivo completo + los tests del dispatcher**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/orchestrator/ -q`
Expected: PASS, 0 xfailed en `test_dispatcher_world_routing.py` (los 4 se convirtieron), sin regresiones en el resto.

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator/dispatcher.py tests/unit/orchestrator/test_dispatcher_world_routing.py
git commit -m "fix(dispatcher): las cláusulas de clima ya no se comen comandos domóticos"
```

---

### Task 2: FAST_WEATHER observable + "pasado mañana"

Dos defectos del mismo handler. (1) Un `weather_entity` mal configurado es invisible para siempre: la rama "hoy" es un lookup de cache que devuelve `None` sin log, `describe_current(None)` da `NO_DATA`, y el handler marca `success=True` — cero señales. (2) `"pasado mañana" in text` matchea `"mañana"` como substring y responde el pronóstico de mañana etiquetado "Mañana:" — confidently wrong.

**Decisión de diseño:** `success=True` para el degradado honesto NO cambia (los tests existentes lo pinean a propósito: success=False está reservado a excepciones). La observabilidad entra por un contador en `_stats` + un warning rate-limited.

**Files:**
- Modify: `src/orchestrator/dispatcher.py` (`__init__` ~línea 544 `_stats`; `_handle_weather` ~línea 1335-1395)
- Modify: `src/world/weather.py` (`_DIA_INDEX` línea 85)
- Test: `tests/unit/orchestrator/test_dispatcher_world_routing.py`, `tests/unit/world/test_weather.py`

**Interfaces:**
- Produces: `self._stats["weather_no_data"]: int`; `_DIA_INDEX` acepta `"pasado mañana"` / `"pasado manana"` → índice 2.
- Consumes: `describe_forecast(forecast, dia)` ya capitaliza `dia` para la frase hablada.

- [ ] **Step 1: Tests que fallan — pasado mañana en weather.py**

En `tests/unit/world/test_weather.py`:

```python
def test_describe_forecast_pasado_manana_uses_index_2():
    forecast = [
        {"condition": "sunny", "temperature": 20, "templow": 10},
        {"condition": "rainy", "temperature": 18, "templow": 9},
        {"condition": "cloudy", "temperature": 15, "templow": 7},
    ]
    out = describe_forecast(forecast, "pasado mañana")
    assert out.startswith("Pasado mañana:")
    assert "nublado" in out and "entre 7 y 15 grados" in out


def test_describe_forecast_pasado_manana_short_forecast_is_honest():
    # Solo 2 días de pronóstico: índice 2 no existe → NO_FORECAST, no IndexError.
    forecast = [{"condition": "sunny", "temperature": 20},
                {"condition": "rainy", "temperature": 18}]
    assert describe_forecast(forecast, "pasado mañana") == NO_FORECAST
```

(Verificar que el archivo ya importa `describe_forecast` y `NO_FORECAST`; si no, agregarlos al import de `src.world.weather`.)

- [ ] **Step 2: Tests que fallan — dispatcher: detección de día + contador + warning**

En `tests/unit/orchestrator/test_dispatcher_world_routing.py`:

```python
async def test_pasado_manana_requests_day_after_tomorrow(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.call_service_with_response = AsyncMock(return_value={
        "service_response": {"weather.forecast_home": {"forecast": [
            {"condition": "sunny", "temperature": 20, "templow": 10},
            {"condition": "rainy", "temperature": 18, "templow": 9},
            {"condition": "cloudy", "temperature": 15, "templow": 7},
        ]}}
    })
    result = await d._handle_weather("qué tiempo hace pasado mañana", Priority.HIGH)
    assert result.response.startswith("Pasado mañana:")


async def test_no_data_increments_stat_and_warns(dispatcher_with_async_ha, caplog):
    d = dispatcher_with_async_ha
    d.ha.get_entity_state_cached = MagicMock(return_value=None)
    with caplog.at_level("WARNING"):
        result = await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert result.success is True          # el degradado honesto NO es fallo
    assert result.response == NO_DATA
    assert d._stats["weather_no_data"] == 1
    assert any("weather_entity" in r.message for r in caplog.records)


async def test_data_present_does_not_touch_the_no_data_stat(dispatcher_with_async_ha):
    d = dispatcher_with_async_ha
    d.ha.get_entity_state_cached = MagicMock(return_value={
        "state": "sunny", "attributes": {"temperature": 22.0},
    })
    result = await d._handle_weather("qué tiempo hace", Priority.HIGH)
    assert "22 grados" in result.response
    assert d._stats["weather_no_data"] == 0
```

- [ ] **Step 3: Correr y verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/world/test_weather.py tests/unit/orchestrator/test_dispatcher_world_routing.py -k "pasado_manana or no_data_increments or does_not_touch" -v`
Expected: FAIL (los 5).

- [ ] **Step 4: Implementar**

`src/world/weather.py` línea 85:

```python
_DIA_INDEX = {"hoy": 0, "mañana": 1, "manana": 1, "pasado mañana": 2, "pasado manana": 2}
```

(`describe_forecast` ya maneja `index >= len(forecast)` → `NO_FORECAST`, y `dia.capitalize()` produce "Pasado mañana".)

`src/orchestrator/dispatcher.py`:

1. En el dict `self._stats = {` (~línea 544) agregar la key `"weather_no_data": 0,`.
2. En `__init__`, junto a las demás inicializaciones de estado: `self._last_weather_nodata_warn = float("-inf")`.
3. Verificar que el módulo importa `time` (si no, agregarlo al bloque stdlib).
4. En `_handle_weather`, reemplazar la detección de `dia` (línea 1361):

```python
        if "pasado mañana" in text_lower or "pasado manana" in text_lower:
            dia = "pasado mañana"
        elif "mañana" in text_lower or "manana" in text_lower:
            dia = "mañana"
        else:
            dia = None
```

5. Antes del `return DispatchResult(...)` final, después del try/except:

```python
        # Observabilidad (review PR #14, 2026-08-06): un weather_entity mal
        # configurado, un boot sin prefetch o un WS muerto degradan TODOS a
        # la misma respuesta honesta con success=True — sin esto son
        # invisibles para siempre: cero logs, stats de éxito. El contador
        # separa "habló el clima" de "habló la disculpa"; el warning (rate-
        # limited, no spamear si el sensor está caído un fin de semana)
        # apunta directo a la config.
        from src.world.weather import NO_FORECAST
        if success and response in (NO_DATA, NO_FORECAST):
            self._stats["weather_no_data"] += 1
            now = time.monotonic()
            if now - self._last_weather_nodata_warn > 300:
                self._last_weather_nodata_warn = now
                logger.warning(
                    "FAST_WEATHER answered honestly with no data "
                    "(entity=%s, dia=%s, count=%d) — if chronic, check that "
                    "home_assistant.weather_entity exists in HA",
                    self.weather_entity, dia, self._stats["weather_no_data"],
                )
```

(El import local de `NO_FORECAST` puede fundirse con el import local de `NO_DATA`/`describe_*` que ya está al principio del método.)

- [ ] **Step 5: Correr todo lo tocado**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/world/ tests/unit/orchestrator/test_dispatcher_world_routing.py -q`
Expected: PASS. Los tests existentes de honest-fallback (`success is True`) siguen verdes.

- [ ] **Step 6: Actualizar el comentario de `weather_entity` en settings.yaml**

En `config/settings.yaml` (~línea 46-51), el comentario dice que la respuesta degradada es `"No tengo el dato del clima ahora mismo"` SIEMPRE — falso para la rama de pronóstico. Reemplazar el bloque por:

```yaml
  # Entidad de HA de la que sale el clima hablado (path FAST_WEATHER del
  # dispatcher: "qué tiempo hace" lee su estado cacheado, "...mañana"/"...
  # pasado mañana" le piden weather.get_forecasts). Si el nombre real de la
  # entidad en HA no es este, el asistente responde el degradado honesto
  # SIEMPRE ("No tengo el dato del clima..." para hoy, "No tengo el
  # pronóstico..." para el resto), igual que un sensor caído. Señales para
  # distinguirlo (review PR #14): el contador _stats["weather_no_data"] y
  # el warning rate-limited "FAST_WEATHER answered honestly with no data".
  weather_entity: "weather.forecast_home"
```

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator/dispatcher.py src/world/weather.py config/settings.yaml \
        tests/unit/world/test_weather.py tests/unit/orchestrator/test_dispatcher_world_routing.py
git commit -m "fix(weather): sin-datos observable (stat + warning) y soporte de pasado mañana"
```

---

### Task 3: AudioArchiver — chequeo de disco fuera del event loop + contadores

`_has_room()` corre sincrónico en el loop (Path.exists + shutil.disk_usage = syscalls stat/statvfs) una vez por segmento archivado, ANTES del `to_thread` — violación de CLAUDE.md activa exactamente durante la campaña. Además los fallos se loguean pero no se cuentan: tras una noche de campaña nadie puede responder "¿qué fracción de segmentos no se archivó?" sin grepear el journal, y los fallos de archivado sesgan el dataset (correlacionan con presión de disco).

**Files:**
- Modify: `src/ambient/audio_archive.py`
- Test: `tests/unit/ambient/test_audio_archive.py`

**Interfaces:**
- Produces: `AudioArchiver.stats: dict` con keys `"written"`, `"skipped_disk"`, `"failed"` (ints). `_write_sync(path, mono) -> bool` (antes `-> None`): False = piso de disco, True = escribió.
- Consumes: nada nuevo; `write()` conserva su firma y contrato (`str | None`).

- [ ] **Step 1: Tests que fallan**

En `tests/unit/ambient/test_audio_archive.py` (seguir el estilo de los tests existentes del archivo):

```python
def test_piso_de_disco_corta_en_el_thread_y_cuenta_skipped(tmp_path):
    # min_free_bytes imposible (1 exabyte): _has_room da False sin monkeypatch.
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                         min_free_bytes=10**18)
    audio = np.full(1600, 0.1, dtype=np.float32)
    path = asyncio.run(arch.write("cocina", 1, audio))
    assert path is None
    assert list(tmp_path.rglob("*.flac")) == []
    assert arch.stats == {"written": 0, "skipped_disk": 1, "failed": 0}


def test_escritura_ok_cuenta_written(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    audio = np.full(1600, 0.1, dtype=np.float32)
    path = asyncio.run(arch.write("cocina", 2, audio))
    assert path is not None and path.endswith("cocina/2.flac")
    assert arch.stats == {"written": 1, "skipped_disk": 0, "failed": 0}


def test_fallo_de_escritura_cuenta_failed(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    audio = np.zeros((0,), dtype=np.float32)   # audio vacío → ValueError interna
    path = asyncio.run(arch.write("cocina", 3, audio))
    assert path is None
    assert arch.stats["failed"] == 1
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/test_audio_archive.py -v`
Expected: FAIL con `AttributeError: stats` en los 3 nuevos; los preexistentes verdes.

- [ ] **Step 3: Implementar**

En `src/ambient/audio_archive.py`:

1. En `__init__`, agregar: `self.stats = {"written": 0, "skipped_disk": 0, "failed": 0}`.
2. Reescribir `write` y `_write_sync` (el chequeo de disco se muda al thread):

```python
    async def write(
        self, room_id: str, utt_id: int, audio: np.ndarray
    ) -> str | None:
        """Guardar el audio de una utterance.

        Args:
            room_id: Habitación de origen (define el subdirectorio).
            utt_id: rowid de la utterance; da el nombre del archivo.
            audio: (n_samples, n_channels) float32, o (n_samples,) mono.

        Returns:
            La ruta del archivo escrito, o None si está deshabilitado, no
            hay disco por encima del piso, o falló la escritura. Los tres
            casos se distinguen en ``self.stats`` (skipped_disk / failed;
            deshabilitado no cuenta) y los dos últimos se loguean — el
            rechazo por piso de disco como máximo una vez por hora.
        """
        if not self.enabled:
            return None
        try:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            if mono.size == 0:
                raise ValueError("audio vacío")
            path = self.base_dir / room_id / f"{utt_id}.flac"
            # El chequeo de disco (stat/statvfs) va DENTRO del thread: en el
            # loop bloqueaba el voice path una vez por segmento (review
            # PR #14, 2026-08-06).
            written = await asyncio.to_thread(self._write_sync, path, mono)
            if not written:
                self.stats["skipped_disk"] += 1
                return None
            self.stats["written"] += 1
            return str(path)
        except Exception as e:
            self.stats["failed"] += 1
            logger.warning(
                "AudioArchiver: no se pudo guardar %s/%d (%s)", room_id, utt_id, e
            )
            return None

    def _write_sync(self, path: Path, mono: np.ndarray) -> bool:
        """Escritura bloqueante — corre en un hilo aparte.

        Returns:
            False si el disco está debajo del piso (no se escribió nada),
            True si el archivo quedó en disco.
        """
        import soundfile as sf

        if not self._has_room():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), mono, self.sample_rate, format="FLAC")
        return True
```

3. Fixes de docstring del módulo (hallazgos del comment-analyzer): en el docstring top-level, cambiar "no se pudo guardar un wav" por "no se pudo guardar un FLAC", y "cualquier fallo ... devuelve None y se loguea" por "cualquier fallo devuelve None; los fallos de escritura se loguean siempre, el rechazo por piso de disco a lo sumo una vez por hora, y enabled=False retorna None en silencio".

- [ ] **Step 4: Correr los tests del módulo + transcriber (consumidor)**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/test_audio_archive.py tests/unit/ambient/test_transcriber.py -q`
Expected: PASS completo (el contrato externo de `write()` no cambió).

- [ ] **Step 5: Commit**

```bash
git add src/ambient/audio_archive.py tests/unit/ambient/test_audio_archive.py
git commit -m "fix(ambient): chequeo de disco del archiver fuera del event loop + stats"
```

---

### Task 4: Purga — unlinks en thread, barrido de huérfanos, rowcount honesto

Tres defectos de la misma zona. (1) `purge_expired` unlinkea FLACs sincrónico en el loop (potencialmente miles por pasada al volumen de campaña). (2) El DELETE commitea ANTES de los unlinks: un unlink fallido deja el archivo fuera del alcance del TTL para siempre; y un crash entre `AudioArchiver.write()` y `set_audio_path()` deja un FLAC sin fila que nadie barre — audio crudo del hogar fuera de la política de retención (defecto de privacidad). (3) El log de purga cuenta *intentos* de unlink como si fueran borrados, y `set_audio_path` con id inexistente commitea "bien" sin tocar ninguna fila.

**Fix:** unlinks batcheados vía `asyncio.to_thread`; sweep de huérfanos (FLACs sin fila viva Y más viejos que el TTL — la guarda de mtime protege la carrera write→set_audio_path); log con conteo real; warning en `set_audio_path` con rowcount 0.

**Files:**
- Modify: `src/ambient/store.py` (`__init__` línea 76, `set_audio_path` línea 137-150, `purge_expired` línea 229-259; agregar `import asyncio` al header si falta)
- Modify: `src/ambient/transcriber.py` (`build_ambient_path`, wiring del store ~línea 433)
- Test: `tests/unit/ambient/test_store.py`

**Interfaces:**
- Produces: `AmbientStore.__init__(db_path, retention_hours, audio_dir: str | None = None)`. Métodos privados `_unlink_batch(paths: list[str]) -> int` y `_sweep_orphans(live_ids: set[int], cutoff: float) -> int` (síncronos, corren en thread).
- Consumes: convención de nombre de archivo `<utt_id>.flac` (el sweep parsea `f.stem` como int).

- [ ] **Step 1: Tests que fallan**

En `tests/unit/ambient/test_store.py`, siguiendo el estilo de creación/init/close de los tests vecinos del archivo:

```python
def test_purga_barre_huerfanos_fuera_del_ttl_y_respeta_lo_vivo(tmp_path):
    async def inner():
        audio_dir = tmp_path / "ambient_audio"
        (audio_dir / "cocina").mkdir(parents=True)
        store = AmbientStore(db_path=str(tmp_path / "a.db"),
                             retention_hours=1.0, audio_dir=str(audio_dir))
        await store.init()
        old = time.time() - 7200  # 2h > TTL de 1h

        # Huérfano real: archivo sin fila, más viejo que el TTL → se barre.
        orphan = audio_dir / "cocina" / "999.flac"
        orphan.write_bytes(b"x")
        os.utime(orphan, (old, old))

        # Carrera write→set_audio_path: archivo sin fila pero FRESCO → vive.
        fresh = audio_dir / "cocina" / "998.flac"
        fresh.write_bytes(b"x")

        # Archivo viejo PERO con fila viva → vive (la fila manda).
        utt_id = await store.add(AmbientUtterance(
            room_id="cocina", t0=time.time(), t1=time.time() + 1.0, text="hola",
        ))
        ref = audio_dir / "cocina" / f"{utt_id}.flac"
        ref.write_bytes(b"x")
        os.utime(ref, (old, old))
        await store.set_audio_path(utt_id, str(ref))

        await store.purge_expired()
        assert not orphan.exists()
        assert fresh.exists()
        assert ref.exists()
        await store.close()
    asyncio.run(inner())


def test_purga_sin_audio_dir_no_barre_nada(tmp_path):
    # Compat: el store sin audio_dir (default) purga filas igual que antes.
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1.0)
        await store.init()
        await store.add(AmbientUtterance(
            room_id="cocina", t0=time.time() - 7200, t1=time.time() - 7199,
            text="vieja",
        ))
        assert await store.purge_expired() == 1
        await store.close()
    asyncio.run(inner())


def test_set_audio_path_con_id_inexistente_loguea_warning(tmp_path, caplog):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"))
        await store.init()
        with caplog.at_level("WARNING"):
            await store.set_audio_path(424242, "/no/existe.flac")
        assert any("424242" in r.message for r in caplog.records)
        await store.close()
    asyncio.run(inner())
```

(Agregar `import os` y `import time` al header del test si faltan. Si los nombres reales de init/teardown del store difieren de `init()`/`close()`, usar los de los tests vecinos.)

- [ ] **Step 2: Correr y verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/test_store.py -v -k "huerfanos or sin_audio_dir or inexistente"`
Expected: FAIL — `TypeError: unexpected keyword argument 'audio_dir'` en el primero; el warning ausente en el tercero.

- [ ] **Step 3: Implementar en store.py**

1. `__init__`:

```python
    def __init__(
        self,
        db_path: str = "./data/ambient.db",
        retention_hours: float = 12.0,
        audio_dir: str | None = None,
    ):
```
con `self.audio_dir = Path(audio_dir) if audio_dir else None` y docstring del parámetro: "Raíz de los FLACs de keep_audio. Si se pasa, la purga además barre huérfanos (archivos sin fila viva y más viejos que el TTL): un unlink fallido o un crash entre write() y set_audio_path() dejaban audio crudo del hogar fuera de la política de retención para siempre (review PR #14)."

2. `set_audio_path` — capturar el cursor y avisar si no tocó filas:

```python
        cur = await self._db.execute(
            "UPDATE utterances SET audio_path=? WHERE id=?", (path, utt_id)
        )
        await self._db.commit()
        if cur.rowcount == 0:
            logger.warning(
                "AmbientStore.set_audio_path: id %d no existe — %s queda "
                "huérfano hasta el sweep de la purga", utt_id, path,
            )
```

3. `purge_expired` — reemplazar desde el loop de unlinks hasta el log:

```python
        unlinked = await asyncio.to_thread(self._unlink_batch, doomed)
        orphans = 0
        if self.audio_dir is not None:
            id_cur = await self._db.execute("SELECT id FROM utterances")
            live_ids = {row["id"] for row in await id_cur.fetchall()}
            orphans = await asyncio.to_thread(self._sweep_orphans, live_ids, cutoff)
        if cur.rowcount or orphans:
            logger.info(
                "AmbientStore purga: %d utterances borradas (TTL %.1fh), "
                "%d/%d audios borrados, %d huérfanos barridos",
                cur.rowcount, self.retention_hours, unlinked, len(doomed), orphans,
            )
        return cur.rowcount
```

4. Métodos nuevos (síncronos — corren vía `to_thread`):

```python
    def _unlink_batch(self, paths: list[str]) -> int:
        """Borrar archivos best-effort. Devuelve cuántos se borraron de verdad."""
        ok = 0
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
                ok += 1
            except OSError as e:
                logger.warning(
                    "AmbientStore purga: no se pudo borrar %s: %s", path, e
                )
        return ok

    def _sweep_orphans(self, live_ids: set[int], cutoff: float) -> int:
        """Borrar FLACs sin fila viva y más viejos que el TTL.

        La guarda de mtime protege la carrera write()→set_audio_path(): un
        FLAC recién escrito todavía sin puntero NO es huérfano. Un archivo
        cuyo unlink falló en una purga anterior (fila ya borrada) sí lo es,
        y cae acá en el siguiente ciclo.
        """
        if self.audio_dir is None or not self.audio_dir.is_dir():
            return 0
        swept = 0
        for f in self.audio_dir.rglob("*.flac"):
            try:
                utt_id = int(f.stem)
            except ValueError:
                continue  # archivo ajeno a la convención <id>.flac — no tocar
            try:
                if utt_id in live_ids or f.stat().st_mtime >= cutoff:
                    continue
                f.unlink(missing_ok=True)
                swept += 1
            except OSError as e:
                logger.warning(
                    "AmbientStore sweep: no se pudo borrar %s: %s", f, e
                )
        return swept
```

5. Verificar que el header tiene `import asyncio`.

- [ ] **Step 4: Wiring en build_ambient_path**

En `src/ambient/transcriber.py`, `build_ambient_path`: mover la línea `ka_cfg = ambient_cfg.get("keep_audio", {}) or {}` (hoy ~455, justo antes del `AudioArchiver`) ARRIBA de la creación del store (~433), y pasar el dir:

```python
    ka_cfg = ambient_cfg.get("keep_audio", {}) or {}
    store = AmbientStore(
        db_path=ambient_cfg.get("db_path", "./data/ambient.db"),
        retention_hours=ambient_cfg.get("retention_hours", 12.0),
        audio_dir=ka_cfg.get("dir", "./data/ambient_audio"),
    )
```

(No duplicar `ka_cfg`: la definición original de más abajo se elimina; el `AudioArchiver` usa la de arriba.)

- [ ] **Step 5: Correr todo el módulo ambient**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/ -q`
Expected: PASS. Ojo con los tests existentes que pineen el texto exacto del log de purga ("%d audios") — actualizarlos al formato nuevo si los hay.

- [ ] **Step 6: Commit**

```bash
git add src/ambient/store.py src/ambient/transcriber.py tests/unit/ambient/test_store.py
git commit -m "fix(ambient): purga sin bloquear el loop + barrido de FLACs huérfanos"
```

---

### Task 5: Tools — exit code honesto, snapshot corrupto ruidoso, markers sin drift

Tres hardening del kit de medición, todos de la familia "proxies mentirosos". (1) `ambient_wer.py` sale con exit 0 en un reporte que él mismo marca "NO CONFIABLE — NO USAR"; su hermano `ambient_groundtruth.py` estableció exit 2 para éxito parcial en este mismo PR. (2) Un `hypotheses.json` corrupto es indistinguible de uno ausente: `load_snapshot` traga el `JSONDecodeError` y cae en silencio a la DB (que purga a las 48h) — el operador no sabe que tiene que recuperar el snapshot del backup. (3) `_validate` en groundtruth hardcodea `{UNINTELLIGIBLE, "[tv]", "[media]"}` en vez de importar el set de `src/ambient/wer.py`, y los marcadores INLINE (`"dijo algo [ininteligible] y se fue"`) se puntúan como palabras sin aviso.

**Files:**
- Modify: `tools/ambient_wer.py` (docstring exit codes, `load_snapshot` ~línea 197-214, `main` ~final)
- Modify: `tools/ambient_groundtruth.py` (`_validate` ~línea 425-427)
- Test: `tests/unit/tools/test_ambient_wer_report.py`, `tests/unit/tools/test_ambient_groundtruth.py`

**Interfaces:**
- Produces: `ambient_wer.main()` → `SystemExit(2)` cuando `rep["confiable"]` es False; `load_snapshot(path) -> tuple[dict, dict, str | None]` (tercer elemento: descripción del error de parseo, None si no hubo).
- Consumes: `src.ambient.wer.bucket_of`, `is_excluded`, y el set de marcadores que `wer.py` exporte (verificar el nombre exacto con `grep -n "MARKERS\|UNINTELLIGIBLE" src/ambient/wer.py tools/ambient_groundtruth.py`).

- [ ] **Step 1: Tests que fallan — exit 2 y snapshot corrupto**

En `tests/unit/tools/test_ambient_wer_report.py` (usar los helpers/estilo del archivo; `main` se invoca con `monkeypatch.setattr(sys, "argv", ...)`):

```python
def test_reporte_no_confiable_sale_con_exit_2(tmp_path, monkeypatch):
    from src.ambient.wer import bucket_of
    b_alto, b_bajo = bucket_of(0.9), bucket_of(0.1)
    (tmp_path / "groundtruth.json").write_text(
        json.dumps({"1": "hola che"}), encoding="utf-8")
    (tmp_path / "hypotheses.json").write_text(json.dumps({
        "utterances": {"1": {"text": "hola che", "vad_prob": 0.9}},
        # Volumen real concentrado en un bucket sin ningún par evaluado →
        # cobertura muy por debajo del piso → confiable=False.
        "volumes": {b_alto: 1, b_bajo: 99},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py", "--groundtruth", str(tmp_path / "groundtruth.json"),
        "--out", str(tmp_path / "rep.json"),
    ])
    with pytest.raises(SystemExit) as ex:
        ambient_wer.main()
    assert ex.value.code == 2


def test_snapshot_corrupto_es_error_duro_no_fallback_silencioso(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "groundtruth.json").write_text(
        json.dumps({"1": "hola"}), encoding="utf-8")
    (tmp_path / "hypotheses.json").write_text("{trunc", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "ambient_wer.py", "--groundtruth", str(tmp_path / "groundtruth.json"),
        "--db", str(tmp_path / "nonexistent.db"),
    ])
    with pytest.raises(SystemExit) as ex:
        ambient_wer.main()
    assert ex.value.code == 1
    assert "corrupto" in capsys.readouterr().err
```

- [ ] **Step 2: Test que falla — aviso de marcador inline en groundtruth**

En `tests/unit/tools/test_ambient_groundtruth.py`, junto a los tests de `_validate` (usa el helper `_escribir_set` ya definido en el archivo):

```python
def test_validate_avisa_marcador_inline(tmp_path, capsys):
    """Referencia con "[ininteligible]" INCRUSTADO en una frase más larga:
    el scorer lo bracket-stripea y lo puntúa como la palabra 'ininteligible'
    (sustitución/deleción falsa). _validate debe avisar nombrando el id,
    SIN invalidar el set (sigue siendo usable, solo distorsiona ese par)."""
    path = _escribir_set(tmp_path, {
        "1": "hola",
        "2": "dijo algo [ininteligible] y se fue",
    })
    _validate(path)          # no lanza
    assert "INLINE" in capsys.readouterr().out
```

- [ ] **Step 3: Correr y verificar que fallan**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/tools/ -v -k "exit_2 or corrupto or inline"`
Expected: FAIL (exit 0 en el primero, fallback silencioso en el segundo, sin aviso en el tercero).

- [ ] **Step 4: Implementar en ambient_wer.py**

1. `load_snapshot` — separar "no existe" (fallback esperado) de "corrupto" (error duro):

```python
def load_snapshot(path: Path) -> tuple[dict[str, dict], dict[str, int], str | None]:
    """Leer el ``hypotheses.json`` que escribió el export.

    Returns:
        ``(utterances, volumes, error)``. ``error`` es None si el archivo no
        existe (fallback a DB esperado) o si se leyó bien; si el archivo
        EXISTE pero no parsea, ``error`` describe el problema — el snapshot
        de una campaña se escribe una sola vez, y confundir "corrupto" con
        "ausente" manda al operador a una DB que purga a las 48h en vez de
        al backup del archivo.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}, {}, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, {}, f"snapshot corrupto: {e}"
    utts = data.get("utterances") or {}
    vols = {k: int(v) for k, v in (data.get("volumes") or {}).items()}
    return utts, vols, None
```

2. En `main`, en el call site: `snap_utts, snap_vols, snap_error = load_snapshot(snap_path)` y, ANTES del fallback a DB:

```python
    if snap_error:
        print(f"ERROR: {snap_path} — {snap_error}. NO se cae a la DB (purga "
              f"48h, ventana corrida): recuperá el snapshot del backup.",
              file=sys.stderr)
        raise SystemExit(1)
```

3. Al final de `main`, después de guardar el reporte:

```python
    if not rep["confiable"]:
        # exit 0 acá sería el mismo proxy mentiroso que el export ya cerró:
        # un `--validate && ambient_wer && publicar` encadenado leería un
        # agregado inusable como éxito.
        raise SystemExit(2)
```

4. Docstring del módulo, bloque "Exit codes": cambiar la línea del 0 por "0 Reporte construido y agregado confiable." y agregar "2 Reporte construido pero NO CONFIABLE (mirá `motivos` en el JSON) — no publicar el agregado."

5. Actualizar cualquier test existente de `load_snapshot` a la firma de 3 elementos.

- [ ] **Step 5: Implementar en ambient_groundtruth.py**

1. Línea 52: ampliar el import a `from src.ambient.wer import UNINTELLIGIBLE, bucket_of, is_excluded`.
2. Línea ~427: reemplazar el set hardcodeado — el conteo `marcadores` pasa de `if v.strip().lower() in {UNINTELLIGIBLE, "[tv]", "[media]"}` a `if is_excluded(v)` (así `wer.MEDIA_MARKERS` tiene un solo dueño y el validador no puede driftear del scorer).
3. Después del bloque de prints de conteo (`print(f"  marcadores excluidos   : {marcadores}")`), agregar el aviso inline:

```python
    inline = [uid for uid, v in data.items()
              if isinstance(v, str) and "[" in v and not is_excluded(v)]
    if inline:
        print(f"  AVISO: {len(inline)} referencia(s) con marcador INLINE — el "
              f"scorer las puntúa como palabras; un marcador vale solo como "
              f"referencia COMPLETA. Ids: {inline[:10]}")
```

- [ ] **Step 6: Correr los tests de tools completos**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/tools/ -q`
Expected: PASS (incluidos los preexistentes de exit codes del export).

- [ ] **Step 7: Commit**

```bash
git add tools/ambient_wer.py tools/ambient_groundtruth.py tests/unit/tools/
git commit -m "fix(tools): exit 2 en reporte no confiable + snapshot corrupto ruidoso + aviso inline"
```

---

### Task 6: Gaps de tests — solo tests, cero producción

Los tres gaps de riesgo alto del pr-test-analyzer. (a) Nada pinea el orden `persist → wake → archive` que arregló `8cf4a6a`: revertirlo (archivar antes del wake, o dejar que un UPDATE roto se trague el "nexa") deja todo verde. (b) El test de "keep_audio apagado" prueba `archiver=None`, pero producción usa archiver PRESENTE con `enabled=False` — borrar la mitad `.enabled` del gate `self._archiver is not None and self._archiver.enabled` pasa desapercibido. (c) Ningún test asegura `?return_response=true` en la URL — es la razón de ser del método: sin el param, HA devuelve 200 con la lista de changed-states, el dispatcher lo degrada a NO_FORECAST y el pronóstico queda roto para siempre con todos los tests verdes.

**Files:**
- Test: `tests/unit/ambient/test_transcriber.py`
- Test: `tests/unit/home_assistant/test_ha_client_call_service_with_response.py`

**Interfaces:**
- Consumes: `FakeStore`, `FakeAmbientSTT`, `_make`, `_seg`, `EmptySTT` (ya definidos en test_transcriber.py); `_FakeResponseCtx` (ya definido en el test de ha_client). `AmbientTranscriber._textual_wake.maybe_dispatch(room_id, text, source, speaker, audio=...)` y `self._stt.asr_mono(audio)` (contratos existentes de producción).

- [ ] **Step 1: Test de orden persist→wake→archive (con archiver roto)**

En `tests/unit/ambient/test_transcriber.py`, junto a los tests de archiver:

```python
def test_orden_persist_wake_archive_y_el_wake_sobrevive_archiver_roto(tmp_path):
    """Pinea el orden que fijó 8cf4a6a: el archivado (instrumentación) va
    DESPUÉS del wake textual (cara al usuario), y un UPDATE de audio_path
    roto jamás se traga un 'nexa' real. Antes de este test, revertir ese
    orden dejaba la suite verde."""
    events = []

    class OrderStore(FakeStore):
        async def add(self, utt):
            events.append("persist")
            return await super().add(utt)

        async def set_audio_path(self, utt_id, path):
            events.append("archive")
            raise RuntimeError("UPDATE roto")

    class WakeCapableSTT(FakeAmbientSTT):
        # maybe_dispatch recibe audio=self._stt.asr_mono(...): sin esto el
        # AttributeError se comería el dispatch y el test mediría otra cosa.
        def asr_mono(self, audio):
            return audio[:, 0] if audio.ndim == 2 else audio

    class RecordingWake:
        async def maybe_dispatch(self, room_id, text, source, speaker, audio=None):
            events.append("dispatch")

    store = OrderStore()
    tap, tr = _make(store)
    tr._stt = WakeCapableSTT()
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    tr._textual_wake = RecordingWake()

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert events == ["persist", "dispatch", "archive"]
```

- [ ] **Step 2: Test del wiring real con enabled=False**

```python
def test_segmento_sin_texto_NO_se_persiste_con_archiver_deshabilitado(tmp_path):
    """El wiring real de producción (build_ambient_path) SIEMPRE construye
    el archiver y pasa enabled=False cuando keep_audio está apagado — el
    default de la casa. El test vecino con archiver=None no cubre la mitad
    `.enabled` del gate: borrarla haría persistir una fila text_empty por
    cada segmento vacío del ambient, siempre-on, hasta el TTL."""
    store = FakeStore()
    tap, tr = _make(store)
    tr._stt = EmptySTT()
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=False)

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added == []
    assert list(tmp_path.rglob("*.flac")) == []
```

- [ ] **Step 3: Test de la URL con return_response**

En `tests/unit/home_assistant/test_ha_client_call_service_with_response.py`, dentro de `TestCallServiceWithResponse`:

```python
    @pytest.mark.asyncio
    async def test_the_url_carries_return_response_true(self, client):
        """El query param ES la razón de ser del método: sin él HA devuelve
        200 con la lista de changed-states en vez de service_response, el
        dispatcher degrada a NO_FORECAST y el pronóstico queda roto para
        siempre con todos los tests verdes."""
        ctx = _FakeResponseCtx(200)
        ctx._response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        await client.call_service_with_response(
            "weather", "get_forecasts", "weather.forecast_home"
        )

        url = mock_session.post.call_args.args[0]
        assert url.endswith("/api/services/weather/get_forecasts?return_response=true")
```

- [ ] **Step 4: Correr los tres y verificar que pasan YA (pinean comportamiento vigente)**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py tests/unit/home_assistant/test_ha_client_call_service_with_response.py -q`
Expected: PASS. Después, verificación de que muerden: revertir mentalmente (o con `git stash`-style edición temporal) el orden wake/archive en `_handle_segment` y confirmar que el test 1 se pone rojo; restaurar.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/ambient/test_transcriber.py tests/unit/home_assistant/test_ha_client_call_service_with_response.py
git commit -m "test(review-pr14): pinear orden wake/archive, gate enabled=False y return_response"
```

---

### Task 7: Invariante text_empty + logs de ha_client + comentarios que desinforman

Batch final de fixes chicos. (1) `AmbientUtterance` puede representar `text_empty=True, text="prendé la luz"` — el invariante vive solo en disciplina del único call-site; un `__post_init__` de tres líneas lo cierra. (2) `call_service_with_response` descarta el body de error de HA en non-200 (el perfil exacto del incidente de Chroma: el body del 400 tenía el diagnóstico) y su except genérico loguea en español sin contexto ni stack. (3) Cuatro comentarios que desinforman, hallados por el comment-analyzer.

**Files:**
- Modify: `src/ambient/types.py` (~línea 39-70)
- Modify: `src/home_assistant/ha_client.py` (líneas 512-518 y 534-537)
- Modify: `src/ambient/wer.py` (comentario línea ~106)
- Modify: `src/nlu/climate_intent.py` (docstring de `ClimateIntentClassifier` ~línea 120)
- Modify: `config/settings.yaml` (~línea 429)
- Test: `tests/unit/ambient/test_store.py` (o test_types si existe), `tests/unit/home_assistant/test_ha_client_call_service_with_response.py`

**Interfaces:**
- Produces: `AmbientUtterance.__post_init__` que levanta `ValueError` si `text_empty and text`.
- Consumes: nada nuevo.

- [ ] **Step 1: Test que falla — invariante text_empty**

En `tests/unit/ambient/test_store.py` (o el archivo de tests de types del módulo si existe):

```python
def test_text_empty_con_texto_es_un_bug_de_construccion():
    """text_empty es DERIVADO de text (review PR #14): una fila 'vacía' con
    texto destilable —o la inversa— solo puede salir de un call-site nuevo
    mal escrito. Mejor un ValueError en el constructor que una fila
    fantasma en ambient.db."""
    with pytest.raises(ValueError):
        AmbientUtterance(room_id="cocina", t0=0.0, t1=1.0,
                         text="prendé la luz", text_empty=True)


def test_text_vacio_sin_flag_sigue_siendo_legal():
    # Las filas legacy (pre-migración) tienen text="" y text_empty=False;
    # el invariante solo prohíbe la dirección peligrosa.
    u = AmbientUtterance(room_id="cocina", t0=0.0, t1=1.0, text="")
    assert u.text_empty is False
```

- [ ] **Step 2: Correr y verificar que el primero falla**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/ -k "text_empty_con_texto or sin_flag" -v`
Expected: el primero FAIL (no levanta), el segundo PASS.

- [ ] **Step 3: Implementar `__post_init__`**

En `src/ambient/types.py`, al final de `AmbientUtterance`:

```python
    def __post_init__(self):
        # text_empty es derivado de text: la única fuente legal es el
        # transcriber persistiendo un segmento que el STT devolvió vacío.
        # Un segundo call-site (backfill, fixture) que los haga divergir
        # produciría o una fila "vacía" con texto destilable o un fantasma
        # sin marcar — mejor reventar acá (review PR #14, 2026-08-06).
        if self.text_empty and self.text:
            raise ValueError(
                f"text_empty=True con text no vacío ({self.text[:30]!r})"
            )
```

Run de nuevo: ambos PASS. Después la suite de ambient completa (`pytest tests/unit/ambient/ -q`) — ningún test legítimo construye la combinación prohibida.

- [ ] **Step 4: ha_client — body del error y except genérico**

En `src/home_assistant/ha_client.py`:

1. Rama non-200 (líneas 512-518) — capturar el body truncado (defensivo: los mocks de los tests existentes no definen `.text`):

```python
                if resp.status != 200:
                    try:
                        body_snippet = (await resp.text())[:300]
                    except Exception:
                        body_snippet = "<unreadable body>"
                    logger.warning(
                        f"Error {resp.status}: {domain}.{service} on {entity_id} "
                        f"({elapsed:.0f}ms): {body_snippet}"
                    )
                    self._record_error(RuntimeError(f"HTTP {resp.status}"), label)
                    return None
```

2. Except genérico (líneas 534-537) — inglés, contexto y stack:

```python
        except Exception as exc:  # noqa: BLE001 - logged and surfaced as None
            logger.error(
                f"Unexpected error in {label} on {entity_id}: {exc}",
                exc_info=True,
            )
            self._record_error(exc, label)
            return None
```

3. Test nuevo en `TestCallServiceWithResponse` (y ajustar el test del 500 si pineaba el texto viejo del log):

```python
    @pytest.mark.asyncio
    async def test_non_200_logs_the_ha_error_body(self, client, caplog):
        """El body del 400/500 de HA trae el POR QUÉ (entidad inexistente,
        servicio sin response). Tirarlo repite el incidente de Chroma: el
        diagnóstico estaba en el body de un 400 silencioso."""
        ctx = _FakeResponseCtx(400)
        ctx._response.text = AsyncMock(
            return_value='{"message": "Service weather.get_forecasts does not support response"}'
        )
        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        import logging
        with caplog.at_level(logging.WARNING):
            result = await client.call_service_with_response(
                "weather", "get_forecasts", "weather.bad_entity"
            )

        assert result is None
        assert any("does not support response" in r.message for r in caplog.records)
```

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/home_assistant/test_ha_client_call_service_with_response.py -q` → PASS.

- [ ] **Step 4b: Type hints faltantes en transcriber.py (CLAUDE.md exige hints en parámetros públicos)**

En `src/ambient/transcriber.py`:

1. Línea ~44, firma de `AmbientTranscriber.__init__`: cambiar `archiver=None` por `archiver: "AudioArchiver | None" = None`. El módulo ya tiene un bloque `if TYPE_CHECKING:` — agregar ahí `from src.ambient.audio_archive import AudioArchiver` si no está (verificar con `grep -n "TYPE_CHECKING" src/ambient/transcriber.py`; si el bloque no existiera, crearlo debajo de los imports).
2. Línea ~252, `_archive_audio`: cambiar `audio` por `audio: np.ndarray` (el mismo valor ya está anotado `np.ndarray` en `audio_archive.py:43`; `numpy` ya está importado en el módulo).

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py -q` → PASS (cambio solo de anotaciones).

- [ ] **Step 5: Los cuatro comentarios**

1. `src/ambient/wer.py` ~línea 106: el comentario `# d[i][j] = costo; op[i][j] = operación elegida para llegar ahí` describe una matriz `op` que no existe. Reemplazar por: `# d[i][j] = costo; la operación se reconstruye en el backtrace comparando costos`.
2. `src/nlu/climate_intent.py`, docstring de `ClimateIntentClassifier` (~línea 120): agregar como primera línea después del summary: `NO-GO 2026-08-04 — código muerto preservado como instrumento; ver el header del módulo antes de cablear esto a nada.` (el code-search sirve la clase sin el header del módulo — este es el agujero que cierra).
3. `config/settings.yaml` ~línea 429: el comentario dice `priority=1 slow (MiniMax-M2.7-highspeed cloud)` pero el endpoint `reasoner_cloud` de abajo declara `priority: 2`. Corregir el número en el comentario a `priority=2` (mismo patrón de rot que este PR ya corrigió más abajo en 793e14b).
4. `src/home_assistant/ha_client.py` ~línea 489: `"(verified 2026-08-04, ha_client.py:160)"` — el número de línea hardcodeado rota con cualquier edición. Cambiar a `"(verified 2026-08-04, see _ensure_session)"`.

- [ ] **Step 6: Suite completa + commit**

Run: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/unit/ambient/ tests/unit/home_assistant/ tests/unit/nlu/ -q`
Expected: PASS.

```bash
git add src/ambient/types.py src/ambient/wer.py src/ambient/transcriber.py \
        src/nlu/climate_intent.py src/home_assistant/ha_client.py \
        config/settings.yaml tests/unit/ambient/ tests/unit/home_assistant/
git commit -m "fix(review-pr14): invariante text_empty, type hints, logs de ha_client y comentarios rotos"
```

---

## Cierre

- [ ] Suite completa: `/Users/yo/Documents/kza/.venv/bin/python -m pytest tests/ -q` → verde.
- [ ] `python tools/smoke_test.py` (dry-run del fast path — no cubre LLM ni audio).
- [ ] Deploy coordinado al server según `docs/superpowers/plans/` runbook de deploy (memoria `reference_runbook_deploy_main_2026-08-03`): recordar el drift declarado de `settings.yaml` (`keep_audio.enabled: true` hasta el fin de la campaña ~08-08) — un `git pull` sobre el server va a conflictuar con esa línea; stashear/reaplicar el drift o esperar al cierre de campaña.
- [ ] Los hallazgos de la review que este plan NO ataca (decisión explícita): la conflación `success=True`/NO_DATA en `_stats` más allá del contador nuevo (basta por ahora), el `Protocol` para el router de `climate_intent` (código muerto), el `WerResult` frozen/property (se toca solo si el tema WER reaparece), y la mezcla de idiomas en docstrings preexistentes.
