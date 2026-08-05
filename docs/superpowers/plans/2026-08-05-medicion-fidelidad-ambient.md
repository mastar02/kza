# Medición de fidelidad del ambient — Pieza A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el instrumento que permite medir la fidelidad real de la transcripción ambient — persistencia de audio, set de ground truth humano ciego, y runner de WER por bucket de `vad_prob`.

**Architecture:** Tres piezas independientes. (1) El ambient path gana persistencia opcional de audio FLAC, detrás de un flag apagado por default, que persiste **también los segmentos cuyo texto salió vacío** — sin eso la tasa de deleción es invisible. (2) Una herramienta offline arma un set estratificado y muestra el audio sin revelar la salida del modelo. (3) Un core puro de WER en `src/` + un CLI en `tools/` reporta por bucket y agregado re-ponderado.

**Tech Stack:** Python 3.13, asyncio, aiosqlite, soundfile (FLAC), numpy, pytest. **Sin dependencias nuevas** — todas están en `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-08-05-fidelidad-transcripcion-ambient-design.md`

## Global Constraints

- Python 3.13. Tests con el venv del proyecto: `/Users/yo/Documents/kza/.venv/bin/python -m pytest` (el `python3` del sistema es 3.9 y rompe en dataclasses).
- **Prohibido agregar dependencias.** `soundfile>=0.12.1`, `numpy>=1.24.0`, `aiosqlite` ya están.
- `async/await` para todo I/O. La escritura de audio va por `asyncio.to_thread` — nunca bloquear el event loop.
- Inyección de dependencias por constructor. Sin imports relativos: siempre `from src.modulo import Clase`.
- Toda config nueva va en `config/settings.yaml`. Prohibido crear archivos de config.
- `logger = logging.getLogger(__name__)` en `src/`. En `tools/` el reporte a stdout con `print` es correcto (patrón de `tools/harvest_hallucinations.py`).
- Docstrings Google-style en clases y métodos públicos. Type hints en firmas públicas.
- **No tocar** el command path, el wake, el dispatcher ni `src/main.py`.
- `keep_audio.enabled` arranca en `false`. Con el flag apagado el comportamiento actual no cambia en absoluto — hay tests de regresión que lo verifican.
- Formato de audio: **FLAC mono 16 kHz** (lossless; el TTL es corto y el propósito es medir y re-transcribir).
- Normalización de WER: minúsculas, colapso de espacios, quitar puntuación de borde. **Se conservan acentos y ñ.**

---

### Task 1: Esquema del store — `audio_path`, `text_empty`, purga de archivos

**Files:**
- Modify: `src/ambient/types.py:39-63` (dataclass `AmbientUtterance`)
- Modify: `src/ambient/store.py:1-13` (docstring), `:26-62` (schema + migraciones), `:94-126` (`add`), `:174-185` (`undistilled_live`), `:201-219` (`purge_expired`)
- Test: `tests/unit/ambient/test_store.py`

**Interfaces:**
- Consumes: nada (primera tarea).
- Produces: `AmbientUtterance.text_empty: bool`, `AmbientUtterance.audio_path: str | None`; `AmbientStore.set_audio_path(utt_id: int, path: str) -> None`; `AmbientStore.purge_expired() -> int` (ahora borra también archivos).

- [ ] **Step 1: Escribir los tests que fallan**

Agregar al final de `tests/unit/ambient/test_store.py`:

```python
def test_audio_path_roundtrip(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        uid = await store.add(_utt(time.time()))
        await store.set_audio_path(uid, "data/ambient_audio/escritorio/1.flac")
        rows = await store.utterances_between("escritorio", 0, time.time() + 10)
        assert rows[0]["audio_path"] == "data/ambient_audio/escritorio/1.flac"
        await store.close()
    _run(inner())


def test_text_empty_default_es_cero(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        uid = await store.add(_utt(time.time()))
        rows = await store.utterances_between("escritorio", 0, time.time() + 10)
        assert rows[0]["text_empty"] == 0
        assert rows[0]["audio_path"] is None
        assert uid > 0
        await store.close()
    _run(inner())


def test_undistilled_live_excluye_text_empty(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        now = time.time()
        await store.add(_utt(now, text="con texto", vad_prob=0.9))
        await store.add(_utt(now, text="", vad_prob=0.9, text_empty=True))
        rows = await store.undistilled_live(limit=10)
        assert [r["text"] for r in rows] == ["con texto"]
        await store.close()
    _run(inner())


def test_purge_borra_el_archivo_de_audio(tmp_path):
    async def inner():
        audio = tmp_path / "viejo.flac"
        audio.write_bytes(b"fake")
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        uid = await store.add(_utt(time.time() - 7200))
        await store.set_audio_path(uid, str(audio))
        n = await store.purge_expired()
        assert n == 1
        assert not audio.exists()
        await store.close()
    _run(inner())


def test_purge_sobrevive_archivo_faltante(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        uid = await store.add(_utt(time.time() - 7200))
        await store.set_audio_path(uid, str(tmp_path / "no_existe.flac"))
        n = await store.purge_expired()   # no debe lanzar
        assert n == 1
        await store.close()
    _run(inner())
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_store.py -v -k "audio_path or text_empty or purge_borra or purge_sobrevive"`
Expected: FAIL — `TypeError: AmbientUtterance.__init__() got an unexpected keyword argument 'text_empty'` y `AttributeError: 'AmbientStore' object has no attribute 'set_audio_path'`

- [ ] **Step 3: Agregar los campos al DTO**

En `src/ambient/types.py`, dentro de `AmbientUtterance`, después de `distilled: bool = False`:

```python
    # Medición de fidelidad (spec 2026-08-05). text_empty marca los segmentos
    # que el STT transcribió como vacío: hoy se descartan antes de persistir,
    # y por eso la tasa de deleción es invisible. Solo se insertan cuando
    # ambient.keep_audio está activo (campaña de medición).
    text_empty: bool = False
    audio_path: str | None = None   # ruta del FLAC, o None si no se archivó
```

- [ ] **Step 4: Actualizar schema, migraciones y `add`**

En `src/ambient/store.py`, el docstring del módulo dice "Solo texto — jamás audio". Reemplazar esa línea por:

```python
Solo texto por default. Con ``ambient.keep_audio`` activo se guarda además el
FLAC del segmento (columna ``audio_path``) para poder medir WER y re-transcribir;
la purga por TTL borra fila y archivo juntos.
```

En `_SCHEMA`, agregar antes de `created_at REAL NOT NULL`:

```sql
  text_empty INTEGER NOT NULL DEFAULT 0,
  audio_path TEXT,
```

En `_MIGRATIONS`, agregar:

```python
    # Medición de fidelidad (2026-08-05): la DB de prod ya existe.
    "text_empty": "ALTER TABLE utterances ADD COLUMN text_empty INTEGER NOT NULL DEFAULT 0",
    "audio_path": "ALTER TABLE utterances ADD COLUMN audio_path TEXT",
```

En `add()`, cambiar el INSERT para incluir las dos columnas:

```python
        cur = await self._db.execute(
            """INSERT INTO utterances
               (room_id, t0, t1, text, speaker, speaker_confidence, azimuth,
                azimuth_stability, source, confidence, no_speech_prob,
                vad_prob, lang, lang_prob, lang_ok, during_tts, distilled,
                text_empty, audio_path, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                utt.room_id, utt.t0, utt.t1, utt.text, utt.speaker,
                utt.speaker_confidence, utt.azimuth, utt.azimuth_stability,
                utt.source, utt.confidence, utt.no_speech_prob,
                utt.vad_prob, utt.lang, utt.lang_prob,
                None if utt.lang_ok is None else int(utt.lang_ok),
                int(utt.during_tts), int(utt.distilled),
                int(utt.text_empty), utt.audio_path,
                time.time(),
            ),
        )
```

- [ ] **Step 5: Agregar `set_audio_path` y excluir vacías del distiller**

En `src/ambient/store.py`, después de `add()`:

```python
    async def set_audio_path(self, utt_id: int, path: str) -> None:
        """Registrar la ruta del FLAC archivado para una utterance.

        Se llama después de escribir el archivo, porque el nombre depende del
        id que devuelve add(). Un fallo de escritura deja audio_path en NULL.

        Args:
            utt_id: rowid devuelto por add().
            path: ruta del archivo, tal como la devolvió el AudioArchiver.
        """
        await self._db.execute(
            "UPDATE utterances SET audio_path=? WHERE id=?", (path, utt_id)
        )
        await self._db.commit()
```

En `undistilled_live()`, agregar la condición a la query base:

```python
        query = (
            "SELECT * FROM utterances WHERE distilled=0 "
            "AND source NOT IN ('self','tv') "
            "AND COALESCE(text_empty, 0) = 0 "
            "AND COALESCE(vad_prob, 0) >= ? "
        )
```

Y agregar al docstring de `undistilled_live`, después del párrafo sobre `min_vad_prob`:

```
        Excluye ``text_empty=1``: son filas que existen solo para medir la tasa
        de deleción (segmentos sin texto) y no tienen nada que destilar.
```

- [ ] **Step 6: Hacer que la purga borre los archivos**

Reemplazar el cuerpo de `purge_expired()` por:

```python
        cutoff = time.time() - self.retention_hours * 3600
        cur = await self._db.execute(
            "SELECT audio_path FROM utterances WHERE t0 < ? AND audio_path IS NOT NULL",
            (cutoff,),
        )
        doomed = [row["audio_path"] for row in await cur.fetchall()]
        cur = await self._db.execute(
            "DELETE FROM utterances WHERE t0 < ?", (cutoff,)
        )
        await self._db.commit()
        for path in doomed:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("AmbientStore purga: no se pudo borrar %s: %s", path, e)
        if cur.rowcount:
            logger.info(
                "AmbientStore purga: %d utterances borradas (TTL %.1fh), %d audios",
                cur.rowcount, self.retention_hours, len(doomed),
            )
        return cur.rowcount
```

`Path` ya está importado en `store.py:18`.

- [ ] **Step 7: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_store.py -v`
Expected: PASS — todos, incluidos los preexistentes (regresión del schema).

- [ ] **Step 8: Commit**

```bash
git add src/ambient/types.py src/ambient/store.py tests/unit/ambient/test_store.py
git commit -m "feat(ambient): columnas audio_path/text_empty + purga de audio en el store"
```

---

### Task 2: `AudioArchiver` — escritura FLAC best-effort

**Files:**
- Create: `src/ambient/audio_archive.py`
- Test: `tests/unit/ambient/test_audio_archive.py`

**Interfaces:**
- Consumes: nada del código nuevo (independiente de Task 1).
- Produces: `AudioArchiver(base_dir: str, enabled: bool = False, sample_rate: int = 16000, min_free_bytes: int = 1_000_000_000)` con atributo público `.enabled: bool` y método `async def write(room_id: str, utt_id: int, audio: np.ndarray) -> str | None`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/ambient/test_audio_archive.py`:

```python
"""Tests: AudioArchiver — persistencia FLAC best-effort del ambient."""
import asyncio

import numpy as np
import soundfile as sf

from src.ambient.audio_archive import AudioArchiver


def _run(coro):
    return asyncio.run(coro)


def _audio(n: int = 1600, ch: int = 2) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((n, ch)).astype(np.float32) * 0.1


def test_escribe_flac_y_devuelve_ruta(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    path = _run(arch.write("escritorio", 42, _audio()))
    assert path is not None
    assert path.endswith("escritorio/42.flac")
    data, sr = sf.read(path)
    assert sr == 16000
    assert data.ndim == 1          # se guarda mono
    assert len(data) == 1600


def test_deshabilitado_no_escribe_nada(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=False)
    assert _run(arch.write("escritorio", 1, _audio())) is None
    assert list(tmp_path.iterdir()) == []


def test_audio_mono_1d_tambien_funciona(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True)
    mono = _audio(ch=1).reshape(-1)
    path = _run(arch.write("cocina", 3, mono))
    data, _ = sf.read(path)
    assert len(data) == 1600


def test_fallo_de_escritura_devuelve_none_sin_lanzar(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path / "no" / "existe"), enabled=True)
    # base_dir se crea solo; forzamos el fallo con un audio inválido
    assert _run(arch.write("escritorio", 1, np.array([], dtype=np.float32))) is None


def test_disco_lleno_desactiva_la_escritura(tmp_path):
    arch = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                         min_free_bytes=10**18)  # piso imposible de cumplir
    assert _run(arch.write("escritorio", 1, _audio())) is None
    assert not (tmp_path / "escritorio").exists()
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_audio_archive.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.ambient.audio_archive'`

- [ ] **Step 3: Implementar el módulo**

Crear `src/ambient/audio_archive.py`:

```python
"""AudioArchiver — persistencia FLAC del audio de cada segmento del ambient.

Existe SOLO para la campaña de medición de fidelidad (spec 2026-08-05): sin el
audio no hay referencia posible y no se puede calcular WER ni re-transcribir.
Apagado por default.

Best-effort por diseño: cualquier fallo (disco lleno, permisos, I/O, audio
inválido) devuelve None y se loguea. NUNCA propaga — el pipeline de voz no se
cae porque no se pudo guardar un wav.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DISK_WARN_INTERVAL_S = 3600.0


class AudioArchiver:
    """Escribe el audio de un segmento a FLAC mono, best-effort."""

    def __init__(
        self,
        base_dir: str,
        enabled: bool = False,
        sample_rate: int = 16000,
        min_free_bytes: int = 1_000_000_000,
    ):
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.min_free_bytes = min_free_bytes
        self._last_disk_warn = 0.0

    async def write(
        self, room_id: str, utt_id: int, audio: np.ndarray
    ) -> str | None:
        """Guardar el audio de una utterance.

        Args:
            room_id: Habitación de origen (define el subdirectorio).
            utt_id: rowid de la utterance; da el nombre del archivo.
            audio: (n_samples, n_channels) float32, o (n_samples,) mono.

        Returns:
            La ruta del archivo escrito, o None si está deshabilitado o falló.
        """
        if not self.enabled:
            return None
        if not self._has_room():
            return None
        try:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            if mono.size == 0:
                raise ValueError("audio vacío")
            path = self.base_dir / room_id / f"{utt_id}.flac"
            await asyncio.to_thread(self._write_sync, path, mono)
            return str(path)
        except Exception as e:
            logger.warning(
                "AudioArchiver: no se pudo guardar %s/%d (%s)", room_id, utt_id, e
            )
            return None

    def _write_sync(self, path: Path, mono: np.ndarray) -> None:
        """Escritura bloqueante — corre en un hilo aparte."""
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), mono, self.sample_rate, format="FLAC")

    def _has_room(self) -> bool:
        """¿Queda espacio por encima del piso configurado?"""
        try:
            probe = self.base_dir if self.base_dir.exists() else self.base_dir.parent
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free = shutil.disk_usage(probe).free
        except OSError:
            return True  # si no se puede medir, no bloquear la escritura
        if free >= self.min_free_bytes:
            return True
        now = time.time()
        if now - self._last_disk_warn > _DISK_WARN_INTERVAL_S:
            self._last_disk_warn = now
            logger.warning(
                "AudioArchiver: %d MB libres < piso %d MB — audio no archivado",
                free // 1_000_000, self.min_free_bytes // 1_000_000,
            )
        return False
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_audio_archive.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/audio_archive.py tests/unit/ambient/test_audio_archive.py
git commit -m "feat(ambient): AudioArchiver FLAC best-effort con piso de disco"
```

---

### Task 3: Wiring en el transcriber + config

**Files:**
- Modify: `src/ambient/transcriber.py:30-64` (`__init__`), `:134-185` (`_handle_segment`), `~:290-390` (builder)
- Modify: `config/settings.yaml` (bloque `ambient:`, después de `db_path`)
- Test: `tests/unit/ambient/test_transcriber.py`

**Interfaces:**
- Consumes: `AmbientUtterance.text_empty` / `.audio_path` y `AmbientStore.set_audio_path` (Task 1); `AudioArchiver.write` y `.enabled` (Task 2).
- Produces: `AmbientTranscriber.__init__(..., archiver=None)` — parámetro keyword opcional al final.

- [ ] **Step 1: Escribir los tests que fallan**

El archivo ya tiene los fakes (`FakeAmbientSTT`, `FakeTagger`, `FakeDoA`, `FakeStore`) y el helper `_make(store, tv_azimuth=2.5)` en `tests/unit/ambient/test_transcriber.py:18-68`. **Reutilizarlos**, no duplicar.

Primero, agregar `set_audio_path` a `FakeStore` (línea ~36) — el transcriber la va a llamar:

```python
class FakeStore:
    def __init__(self):
        self.added = []
        self.audio_paths = {}     # utt_id → path registrado

    async def add(self, utt):
        self.added.append(utt)
        return len(self.added)

    async def set_audio_path(self, utt_id, path):
        self.audio_paths[utt_id] = path

    async def purge_expired(self):
        return 0
```

Agregar los imports que falten al tope del archivo:

```python
from src.ambient.audio_archive import AudioArchiver
from src.ambient.types import RawSegment
```

Y agregar estos tests al final del archivo. Llaman `_handle_segment` directo (más simple y determinista que manejar el tap):

```python
def _seg(vad: float = 0.8) -> RawSegment:
    return RawSegment(t0=100.0, t1=102.0,
                      audio=np.full((1600, 6), 0.2, dtype=np.float32),
                      vad_prob=vad)


class EmptySTT:
    async def transcribe(self, audio):
        return STTResult(text="", elapsed_ms=5.0)


def test_segmento_sin_texto_se_persiste_si_hay_archiver(tmp_path):
    """Sin esto la tasa de deleción es invisible: el modo de falla más
    importante del ambient es transcribir habla real como vacío."""
    store = FakeStore()
    tap, tr = _make(store)
    tr._stt = EmptySTT()
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert len(store.added) == 1
    assert store.added[0].text == ""
    assert store.added[0].text_empty is True
    assert store.audio_paths[1].endswith("escritorio/1.flac")


def test_segmento_sin_texto_NO_se_persiste_sin_archiver():
    """Regresión: con keep_audio apagado el comportamiento es el de hoy."""
    store = FakeStore()
    tap, tr = _make(store)
    tr._stt = EmptySTT()

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added == []


def test_segmento_con_texto_archiva_el_audio(tmp_path):
    store = FakeStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added[0].text == "hola che"
    assert store.added[0].text_empty is False
    assert store.audio_paths[1].endswith("escritorio/1.flac")


def test_fallo_del_archiver_no_rompe_la_utterance(tmp_path):
    store = FakeStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True,
                                 min_free_bytes=10**18)   # nunca hay lugar

    asyncio.run(tr._handle_segment("escritorio", _seg()))

    assert store.added[0].text == "hola che"
    assert store.audio_paths == {}      # no se registró ninguna ruta


def test_update_fallido_borra_el_archivo(tmp_path):
    """Una fila sin audio_path deja el archivo fuera del alcance de la purga
    por TTL — huérfano permanente. Se borra en el momento."""
    class BrokenStore(FakeStore):
        async def set_audio_path(self, utt_id, path):
            raise RuntimeError("db caída")

    store = BrokenStore()
    tap, tr = _make(store)
    tr._archiver = AudioArchiver(base_dir=str(tmp_path), enabled=True)

    asyncio.run(tr._handle_segment("escritorio", _seg()))   # no propaga

    assert not (tmp_path / "escritorio" / "1.flac").exists()
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py -v -k "archiver or sin_texto or archiva"`
Expected: FAIL — `TypeError: AmbientTranscriber.__init__() got an unexpected keyword argument 'archiver'`

- [ ] **Step 3: Aceptar el archiver en el constructor**

En `src/ambient/transcriber.py`, agregar el parámetro al final de la firma de `__init__` (después de `quality_fn`):

```python
        archiver=None,
```

Y en el cuerpo, después de `self._quality_fn = quality_fn`:

```python
        # AudioArchiver opcional (spec 2026-08-05). None = comportamiento
        # histórico exacto: no se guarda audio y los segmentos sin texto se
        # descartan sin persistir.
        self._archiver = archiver
```

- [ ] **Step 4: Persistir los segmentos sin texto y archivar el audio**

En `_handle_segment`, reemplazar:

```python
            stt_result = await self._stt.transcribe(seg.audio)
            text = stt_result.text.strip()
            if not text:
                return
```

por:

```python
            stt_result = await self._stt.transcribe(seg.audio)
            text = stt_result.text.strip()
            archiving = self._archiver is not None and self._archiver.enabled
            if not text:
                # Sin archiver: comportamiento histórico (descartar y salir).
                # Con archiver: se persiste una fila mínima + el audio, porque
                # la tasa de deleción —habla real transcripta como vacío— es
                # el modo de falla que ninguna otra señal deja ver.
                if not archiving:
                    return
                empty = AmbientUtterance(
                    room_id=room_id, t0=seg.t0, t1=seg.t1, text="",
                    source="self" if seg.during_tts else "unknown",
                    vad_prob=seg.vad_prob, during_tts=seg.during_tts,
                    text_empty=True,
                )
                empty_id = await self._store.add(empty)
                await self._archive_audio(room_id, empty_id, seg.audio)
                return
```

Nota para quien implemente: en la rama de texto vacío **no** se llaman el tagger ni el DoA. Son las operaciones caras del path (GPU y GCC-PHAT) y no aportan nada a una fila que solo existe para medir; llamarlas agregaría carga real a la máquina de producción.

Después, reemplazar la línea `await self._store.add(utt)` por:

```python
            utt_id = await self._store.add(utt)
            if archiving:
                await self._archive_audio(room_id, utt_id, seg.audio)
```

Y agregar el helper como método de `AmbientTranscriber`, justo después de `_handle_segment`:

```python
    async def _archive_audio(self, room_id: str, utt_id: int, audio) -> None:
        """Guardar el audio del segmento y apuntar la fila al archivo.

        Si el UPDATE falla, borra el archivo antes de propagar: una fila con
        audio_path NULL deja el archivo fuera del alcance de la purga por TTL,
        o sea un huérfano permanente en disco.

        Args:
            room_id: Habitación de origen.
            utt_id: rowid devuelto por store.add().
            audio: Audio multicanal del segmento.
        """
        path = await self._archiver.write(room_id, utt_id, audio)
        if not path:
            return
        try:
            await self._store.set_audio_path(utt_id, path)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
```

`Path` tiene que estar importado en `transcriber.py`; si no lo está, agregar `from pathlib import Path` al bloque de imports de stdlib. El `raise` final lo atrapa el `try/except` que ya envuelve `_handle_segment`, así que un fallo de DB sigue sin matar el worker de la habitación.

- [ ] **Step 5: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_transcriber.py -v`
Expected: PASS — los nuevos y todos los preexistentes.

- [ ] **Step 6: Agregar la config**

En `config/settings.yaml`, dentro del bloque `ambient:`, inmediatamente después de la línea `db_path: "./data/ambient.db"`:

```yaml
  # Persistencia de audio para medir fidelidad (spec 2026-08-05). OFF por
  # default: se prende SOLO durante la campaña de medición y se apaga al
  # terminar. Sin el audio no hay referencia posible → no hay WER, y no se
  # puede re-transcribir el histórico cuando mejore el modelo.
  # Costo: ~190 MB/día (2 salas, 13,7% de habla), ~380 MB en la ventana de 48h.
  # ⚠️ Guarda audio crudo de TODO lo que se hable, incluidas terceras personas.
  keep_audio:
    enabled: false
    dir: "./data/ambient_audio"
    min_free_mb: 1000        # piso de disco; debajo de esto no archiva
```

- [ ] **Step 7: Construir el archiver en el builder**

En `src/ambient/transcriber.py`, en el builder (junto al resto de los `from src.ambient...` locales), agregar el import:

```python
    from src.ambient.audio_archive import AudioArchiver
```

Antes de la construcción del `AmbientTranscriber`, agregar:

```python
    ka_cfg = ambient_cfg.get("keep_audio", {}) or {}
    archiver = AudioArchiver(
        base_dir=ka_cfg.get("dir", "./data/ambient_audio"),
        enabled=bool(ka_cfg.get("enabled", False)),
        min_free_bytes=int(ka_cfg.get("min_free_mb", 1000)) * 1_000_000,
    )
```

Y pasarlo al constructor agregando `archiver=archiver,` a la lista de argumentos que ya incluye `tagger=tagger,`.

- [ ] **Step 8: Correr la suite completa del ambient**

Run: `.venv/bin/python -m pytest tests/unit/ambient/ -v`
Expected: PASS — sin regresiones.

- [ ] **Step 9: Commit**

```bash
git add src/ambient/transcriber.py config/settings.yaml tests/unit/ambient/test_transcriber.py
git commit -m "feat(ambient): archivar audio y persistir segmentos sin texto (keep_audio)"
```

---

### Task 4: Core de WER — normalización, distancia de edición, buckets

**Files:**
- Create: `src/ambient/wer.py`
- Test: `tests/unit/ambient/test_wer.py`

**Interfaces:**
- Consumes: nada (independiente de las tareas 1-3).
- Produces: `VAD_BUCKETS: list[tuple[float, float]]`; `bucket_of(vad: float | None) -> str`; `normalize_words(text: str) -> list[str]`; `@dataclass WerResult(subs, ins, dels, ref_words, wer)`; `score(reference: str, hypothesis: str) -> WerResult`; `UNINTELLIGIBLE`, `MEDIA_MARKERS`; `is_excluded(reference: str) -> bool`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/ambient/test_wer.py`:

```python
"""Tests: core de WER para medir fidelidad del ambient."""
from src.ambient.wer import (
    UNINTELLIGIBLE, bucket_of, is_excluded, normalize_words, score,
)


def test_identidad_da_wer_cero():
    r = score("prendé la luz del escritorio", "prendé la luz del escritorio")
    assert r.wer == 0.0
    assert (r.subs, r.ins, r.dels) == (0, 0, 0)


def test_una_sustitucion_en_cuatro_palabras():
    r = score("prendé la luz roja", "prendé la luz azul")
    assert r.subs == 1
    assert r.ref_words == 4
    assert r.wer == 0.25


def test_delecion_se_cuenta_como_delecion():
    r = score("prendé la luz del escritorio", "prendé la luz")
    assert r.dels == 2
    assert r.ins == 0


def test_insercion_se_cuenta_como_insercion():
    r = score("prendé la luz", "prendé la luz del escritorio")
    assert r.ins == 2
    assert r.dels == 0


def test_hipotesis_vacia_es_delecion_total():
    r = score("prendé la luz", "")
    assert r.dels == 3
    assert r.wer == 1.0


def test_referencia_vacia_con_hipotesis_es_alucinacion():
    r = score("", "¡Gracias por ver el video!")
    assert r.ref_words == 0
    assert r.ins == 5
    assert r.wer == 1.0     # convención: ref vacía + hyp no vacía = 1.0


def test_ambas_vacias_es_cero():
    assert score("", "").wer == 0.0


def test_normalizacion_conserva_acentos_y_enie():
    # "apaga" != "apagá": el acento es señal real de calidad en español
    assert normalize_words("¡Apagá!") == ["apagá"]
    assert score("apagá", "apaga").wer == 1.0
    assert normalize_words("el año") == ["el", "año"]


def test_normalizacion_baja_caso_y_saca_puntuacion_de_borde():
    assert normalize_words("  Hola,  QUE tal.  ") == ["hola", "que", "tal"]


def test_buckets_cubren_el_rango():
    assert bucket_of(0.05) == "0.00-0.20"
    assert bucket_of(0.20) == "0.20-0.35"
    assert bucket_of(0.99) == "0.80-1.00"
    assert bucket_of(None) == "sin_vad"


def test_marcadores_excluidos_del_wer():
    assert is_excluded(UNINTELLIGIBLE)
    assert is_excluded("[tv]")
    assert not is_excluded("prendé la luz")
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_wer.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.ambient.wer'`

- [ ] **Step 3: Implementar el core**

Crear `src/ambient/wer.py`:

```python
"""Core de WER para medir la fidelidad de la transcripción ambient.

Sin dependencias externas: la distancia de edición a nivel palabra son 25
líneas y el proyecto no incorpora jiwer para eso.

La normalización define el número que se reporta, así que está acotada y
testeada: minúsculas, colapso de espacios, puntuación de borde fuera.
**Los acentos y la ñ se conservan** — son señal real de calidad en español y
quitarlos inflaría artificialmente el resultado.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Marcadores que el humano puede escribir en la referencia. Se excluyen del
# WER y se reportan aparte: son el techo real del audio, no error del modelo.
UNINTELLIGIBLE = "[ininteligible]"
MEDIA_MARKERS = frozenset({"[tv]", "[media]"})

# Buckets de vad_prob — los mismos del análisis del 2026-08-04 sobre ambient.db.
VAD_BUCKETS: list[tuple[float, float]] = [
    (0.00, 0.20), (0.20, 0.35), (0.35, 0.50),
    (0.50, 0.65), (0.65, 0.80), (0.80, 1.01),
]

_PUNCT = "¡!¿?.,;:…\"'()[]{}—–-«»"
_WS_RE = re.compile(r"\s+")


def bucket_of(vad: float | None) -> str:
    """Etiqueta del bucket de vad_prob al que pertenece un valor."""
    if vad is None:
        return "sin_vad"
    for lo, hi in VAD_BUCKETS:
        if lo <= vad < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "sin_vad"


def normalize_words(text: str) -> list[str]:
    """Tokenizar para comparar: minúsculas, sin puntuación de borde.

    Conserva acentos y ñ a propósito.

    Args:
        text: Texto crudo (referencia o hipótesis).

    Returns:
        Lista de palabras normalizadas; [] si no queda nada.
    """
    out = []
    for raw in _WS_RE.split(text.strip().lower()):
        w = raw.strip(_PUNCT)
        if w:
            out.append(w)
    return out


def is_excluded(reference: str) -> bool:
    """¿La referencia es un marcador que no se puntúa (ininteligible/media)?"""
    r = reference.strip().lower()
    return r == UNINTELLIGIBLE or r in MEDIA_MARKERS


@dataclass
class WerResult:
    """Descomposición del error de una comparación referencia/hipótesis."""

    subs: int
    ins: int
    dels: int
    ref_words: int
    wer: float


def score(reference: str, hypothesis: str) -> WerResult:
    """Comparar una hipótesis contra su referencia humana.

    Convención para el caso degenerado: si la referencia está vacía y la
    hipótesis no, el WER es 1.0 (alucinación total). Si ambas están vacías,
    0.0. Sin esto, dividir por cero al normalizar.

    Args:
        reference: Lo que se dijo realmente (transcripción humana).
        hypothesis: Lo que produjo el modelo.

    Returns:
        WerResult con sustituciones, inserciones, deleciones y WER.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    subs, ins, dels = _edit_ops(ref, hyp)
    if not ref:
        return WerResult(subs, ins, dels, 0, 0.0 if not hyp else 1.0)
    return WerResult(subs, ins, dels, len(ref), (subs + ins + dels) / len(ref))


def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein a nivel palabra con backtrace de operaciones.

    Returns:
        (sustituciones, inserciones, deleciones) del alineamiento óptimo.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = costo; op[i][j] = operación elegida para llegar ahí
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j - 1],   # sustitución
                    d[i][j - 1],       # inserción
                    d[i - 1][j],       # deleción
                )
    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    return subs, ins, dels
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/ambient/test_wer.py -v`
Expected: PASS — 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/wer.py tests/unit/ambient/test_wer.py
git commit -m "feat(ambient): core de WER con normalizacion que conserva acentos"
```

---

### Task 5: Kit de ground truth ciego

**Files:**
- Create: `tools/ambient_groundtruth.py`
- Test: `tests/unit/tools/test_ambient_groundtruth.py`

**Interfaces:**
- Consumes: `bucket_of`, `VAD_BUCKETS` de `src.ambient.wer` (Task 4); la columna `audio_path` (Task 1).
- Produces: `sample_stratified(rows: list[dict], per_bucket: int, seed: int) -> list[dict]`; `render_html(items: list[dict]) -> str`; CLI `--export` / `--validate`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/tools/test_ambient_groundtruth.py` (crear el `__init__.py` del directorio si no existe):

```python
"""Tests: armado del set de ground truth ciego."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.ambient_groundtruth import render_html, sample_stratified


def _rows(n_per_bucket: int = 20):
    """Filas sintéticas repartidas por bucket de vad."""
    out, uid = [], 0
    for centro in (0.10, 0.27, 0.42, 0.57, 0.72, 0.90):
        for _ in range(n_per_bucket):
            uid += 1
            out.append({
                "id": uid, "room_id": "escritorio", "vad_prob": centro,
                "text": f"texto {uid}", "audio_path": f"/tmp/{uid}.flac",
            })
    return out


def test_asignacion_igual_por_bucket_no_proporcional():
    # 6 buckets con volúmenes MUY distintos: el bucket alto tiene 3 filas
    rows = [r for r in _rows(50) if r["vad_prob"] != 0.90]
    rows += [r for r in _rows(3) if r["vad_prob"] == 0.90]
    sel = sample_stratified(rows, per_bucket=7, seed=1)
    from src.ambient.wer import bucket_of
    from collections import Counter
    got = Counter(bucket_of(r["vad_prob"]) for r in sel)
    assert got["0.00-0.20"] == 7
    # el bucket escaso aporta lo que tiene, sin romper
    assert got["0.80-1.00"] == 3


def test_es_reproducible_con_la_misma_semilla():
    rows = _rows()
    a = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=42)]
    b = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=42)]
    c = [r["id"] for r in sample_stratified(rows, per_bucket=5, seed=43)]
    assert a == b
    assert a != c


def test_ignora_filas_sin_audio():
    rows = _rows(5)
    for r in rows:
        r["audio_path"] = None
    assert sample_stratified(rows, per_bucket=7, seed=1) == []


def test_el_html_NO_filtra_la_salida_del_modelo():
    """Ciego por diseño: ver la hipótesis ancla al transcriptor humano."""
    items = [{"id": 1, "room_id": "escritorio", "vad_prob": 0.5,
              "audio": "1.flac", "text": "GRACIAS POR VER EL VIDEO"}]
    html = render_html(items)
    assert "GRACIAS POR VER EL VIDEO" not in html
    assert "1.flac" in html
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_ambient_groundtruth.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tools.ambient_groundtruth'`

- [ ] **Step 3: Implementar la herramienta**

Crear `tools/ambient_groundtruth.py`:

```python
#!/usr/bin/env python3
"""Arma el set de ground truth para medir la fidelidad del ambient.

CIEGO POR DISEÑO: el HTML de transcripción muestra el audio y un campo vacío,
nunca lo que transcribió el modelo. Ver la hipótesis ancla al transcriptor
humano y contamina la referencia — es la lección directa del eval de clima,
donde medir sobre el mismo set con el que se construyó la solución dio 95,5%
y el held-out limpio reveló el fallo que canceló el proyecto.

Uso:
    # 1. exportar el set desde la DB (en el server, o sobre una copia)
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --export \\
        --db data/ambient.db --out /tmp/gt --per-bucket 7

    # 2. abrir /tmp/gt/index.html, escuchar y transcribir, guardar el JSON

    # 3. validar el JSON completado
    PYTHONPATH=. python3 tools/ambient_groundtruth.py --validate /tmp/gt/groundtruth.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import UNINTELLIGIBLE, bucket_of


def sample_stratified(
    rows: list[dict], per_bucket: int, seed: int
) -> list[dict]:
    """Muestrear ``per_bucket`` filas de CADA bucket de vad_prob.

    Asignación IGUAL, no proporcional: una proporcional le daría al bucket
    0.80-1.00 apenas 2 de 42 muestras (283 de 5.939 filas en prod), que es
    justo el bucket donde queremos saber si ya estamos en ~95%.

    Args:
        rows: Filas de utterances (necesitan ``vad_prob`` y ``audio_path``).
        per_bucket: Cuántas tomar de cada bucket (si hay menos, toma todas).
        seed: Semilla del muestreo, para reproducibilidad.

    Returns:
        Filas seleccionadas, ordenadas por bucket y luego por id.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("audio_path"):
            continue          # sin audio no se puede transcribir a mano
        by_bucket[bucket_of(r.get("vad_prob"))].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for bucket in sorted(by_bucket):
        pool = sorted(by_bucket[bucket], key=lambda r: r["id"])
        out.extend(rng.sample(pool, min(per_bucket, len(pool))))
    return out


def render_html(items: list[dict]) -> str:
    """HTML autocontenido para transcribir a ciegas.

    NO incluye el campo ``text`` de los items: el transcriptor no debe ver la
    hipótesis del modelo.

    Args:
        items: Dicts con ``id``, ``room_id``, ``vad_prob`` y ``audio``
            (nombre de archivo relativo al HTML).

    Returns:
        Documento HTML completo, sin red ni dependencias.
    """
    rows = "\n".join(
        f'<li><b>#{it["id"]}</b> <small>{it["room_id"]} · vad '
        f'{it["vad_prob"]:.2f}</small><br>'
        f'<audio controls src="{it["audio"]}"></audio><br>'
        f'<textarea data-id="{it["id"]}" rows="2" cols="70" '
        f'placeholder="lo que escuchás; vacío si no hay habla"></textarea></li>'
        for it in items
    )
    return f"""<!doctype html><meta charset="utf-8">
<title>Ground truth ambient</title>
<style>body{{font:15px system-ui;max-width:60rem;margin:2rem auto;padding:0 1rem}}
li{{margin-bottom:1.5rem}} textarea{{font:14px ui-monospace,monospace}}</style>
<h1>Transcripción de referencia</h1>
<p>Escuchá y escribí <b>exactamente</b> lo que se dice. Dejá vacío si no hay
habla. Usá <code>{UNINTELLIGIBLE}</code> si hay voz pero no se entiende, y
<code>[tv]</code> si la fuente es un parlante.</p>
<p><b>No vas a ver lo que transcribió el modelo</b> — verlo anclaría tu
referencia y arruinaría la medición.</p>
<ol>{rows}</ol>
<button onclick="save()">Guardar groundtruth.json</button>
<script>
function save() {{
  const out = {{}};
  document.querySelectorAll('textarea').forEach(t => out[t.dataset.id] = t.value);
  const b = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'groundtruth.json'; a.click();
}}
</script>"""


def _export(db_path: str, out_dir: str, per_bucket: int, seed: int) -> None:
    """Exportar el set: copia los FLAC y genera index.html + metadata."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT id, room_id, vad_prob, text, audio_path FROM utterances "
        "WHERE audio_path IS NOT NULL"
    )]
    sel = sample_stratified(rows, per_bucket, seed)
    if not sel:
        print("ERROR: ninguna utterance tiene audio archivado. "
              "¿Está prendido ambient.keep_audio?", file=sys.stderr)
        raise SystemExit(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for r in sel:
        name = f'{r["id"]}.flac'
        try:
            shutil.copy(r["audio_path"], out / name)
        except OSError as e:
            print(f"  aviso: no se pudo copiar {r['audio_path']}: {e}", file=sys.stderr)
            continue
        items.append({"id": r["id"], "room_id": r["room_id"],
                      "vad_prob": r["vad_prob"] or 0.0, "audio": name})
    (out / "index.html").write_text(render_html(items), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(
        {"seed": seed, "per_bucket": per_bucket, "db": db_path,
         "ids": [i["id"] for i in items]}, indent=2), encoding="utf-8")
    print(f"{len(items)} utterances exportadas a {out}/")
    print(f"Abrí {out}/index.html, transcribí, y guardá el groundtruth.json ahí mismo.")


def _validate(path: str) -> None:
    """Verificar que el JSON completado sea usable."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vacias = [k for k, v in data.items() if v.strip() == ""]
    print(f"{len(data)} referencias; {len(vacias)} marcadas como sin habla")
    print("OK — usable por tools/ambient_wer.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--db", default="data/ambient.db")
    ap.add_argument("--out", default="/tmp/gt")
    ap.add_argument("--per-bucket", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--validate", metavar="JSON")
    args = ap.parse_args()
    if args.export:
        _export(args.db, args.out, args.per_bucket, args.seed)
    elif args.validate:
        _validate(args.validate)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_ambient_groundtruth.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add tools/ambient_groundtruth.py tests/unit/tools/
git commit -m "feat(tools): kit de ground truth ciego para el ambient"
```

---

### Task 6: Runner de WER — reporte por bucket y agregado re-ponderado

**Files:**
- Create: `tools/ambient_wer.py`
- Test: `tests/unit/tools/test_ambient_wer_report.py`

**Interfaces:**
- Consumes: `score`, `bucket_of`, `is_excluded` de `src.ambient.wer` (Task 4); el `groundtruth.json` de Task 5.
- Produces: `build_report(pairs: list[dict], volumes: dict[str, int]) -> dict` — `pairs` con claves `id`, `vad_prob`, `reference`, `hypothesis`; `volumes` con el conteo real por bucket en la DB.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/unit/tools/test_ambient_wer_report.py`:

```python
"""Tests: reporte de WER por bucket con agregado re-ponderado."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.ambient_wer import build_report


def test_wer_por_bucket():
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "prendé la luz",
         "hypothesis": "prendé la luz"},
        {"id": 2, "vad_prob": 0.10, "reference": "prendé la luz",
         "hypothesis": "so we got a border"},
    ]
    rep = build_report(pairs, volumes={"0.80-1.00": 100, "0.00-0.20": 100})
    assert rep["buckets"]["0.80-1.00"]["wer"] == 0.0
    assert rep["buckets"]["0.00-0.20"]["wer"] > 1.0   # 3 dels + 5 ins sobre 3 ref


def test_agregado_reponderado_por_volumen_real():
    """El set tiene asignación igual por bucket; el agregado debe pesar por
    el volumen real de la DB, si no sobre-representa los buckets altos."""
    pairs = [
        {"id": 1, "vad_prob": 0.90, "reference": "a b c d", "hypothesis": "a b c d"},
        {"id": 2, "vad_prob": 0.10, "reference": "a b c d", "hypothesis": "x y z w"},
    ]
    # el bucket malo es 9x más frecuente en la DB real
    rep = build_report(pairs, volumes={"0.80-1.00": 100, "0.00-0.20": 900})
    assert rep["wer_simple"] == 0.5                 # promedio plano
    assert rep["wer_reponderado"] == 0.9            # 0.0*0.1 + 1.0*0.9


def test_delecion_e_alucinacion_se_reportan_aparte():
    # ojo: vad=0.45 cae en "0.35-0.50"; vad=0.50 caería en "0.50-0.65"
    pairs = [
        {"id": 1, "vad_prob": 0.45, "reference": "hay habla real", "hypothesis": ""},
        {"id": 2, "vad_prob": 0.45, "reference": "", "hypothesis": "¡Gracias!"},
    ]
    rep = build_report(pairs, volumes={"0.35-0.50": 10})
    assert rep["deleciones_totales"] == 1
    assert rep["alucinaciones"] == 1


def test_marcadores_se_excluyen_del_wer():
    pairs = [
        {"id": 1, "vad_prob": 0.45, "reference": "[ininteligible]", "hypothesis": "algo"},
        {"id": 2, "vad_prob": 0.45, "reference": "hola", "hypothesis": "hola"},
    ]
    rep = build_report(pairs, volumes={"0.35-0.50": 10})
    assert rep["excluidas"] == 1
    assert rep["buckets"]["0.35-0.50"]["n"] == 1
    assert rep["buckets"]["0.35-0.50"]["wer"] == 0.0
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_ambient_wer_report.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'tools.ambient_wer'`

- [ ] **Step 3: Implementar el runner**

Crear `tools/ambient_wer.py`:

```python
#!/usr/bin/env python3
"""Mide el WER del ambient contra el ground truth humano.

Reporta SIEMPRE por bucket de vad_prob, y el agregado de dos formas: plano y
re-ponderado por el volumen real de cada bucket en la DB. El plano NO es el WER
del sistema — el set tiene asignación igual por bucket y sobre-representa los
buckets altos a propósito.

Uso:
    PYTHONPATH=. python3 tools/ambient_wer.py \\
        --db data/ambient.db --groundtruth /tmp/gt/groundtruth.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from src.ambient.wer import bucket_of, is_excluded, score


def build_report(pairs: list[dict], volumes: dict[str, int]) -> dict:
    """Construir el reporte de WER a partir de los pares referencia/hipótesis.

    Args:
        pairs: Dicts con ``id``, ``vad_prob``, ``reference``, ``hypothesis``.
        volumes: Conteo real de utterances por bucket en la DB, para el
            agregado re-ponderado.

    Returns:
        Dict con ``buckets``, ``wer_simple``, ``wer_reponderado``,
        ``deleciones_totales``, ``alucinaciones`` y ``excluidas``.
    """
    per_bucket: dict[str, list[float]] = defaultdict(list)
    dels_totales = alucinaciones = excluidas = 0
    for p in pairs:
        ref, hyp = p["reference"], p["hypothesis"]
        if is_excluded(ref):
            excluidas += 1
            continue
        r = score(ref, hyp)
        per_bucket[bucket_of(p.get("vad_prob"))].append(r.wer)
        if ref.strip() and not hyp.strip():
            dels_totales += 1
        if not ref.strip() and hyp.strip():
            alucinaciones += 1
    buckets = {
        b: {"n": len(v), "wer": sum(v) / len(v)}
        for b, v in sorted(per_bucket.items()) if v
    }
    todas = [w for v in per_bucket.values() for w in v]
    total_vol = sum(volumes.get(b, 0) for b in buckets) or 1
    reponderado = sum(
        st["wer"] * volumes.get(b, 0) for b, st in buckets.items()
    ) / total_vol
    return {
        "buckets": buckets,
        "wer_simple": (sum(todas) / len(todas)) if todas else 0.0,
        "wer_reponderado": reponderado,
        "deleciones_totales": dels_totales,
        "alucinaciones": alucinaciones,
        "excluidas": excluidas,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/ambient.db")
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    refs = json.loads(Path(args.groundtruth).read_text(encoding="utf-8"))
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row
    pairs = []
    for uid, reference in refs.items():
        row = db.execute(
            "SELECT text, vad_prob FROM utterances WHERE id=?", (int(uid),)
        ).fetchone()
        if row is None:
            print(f"  aviso: la utterance {uid} ya no está en la DB (purgada)")
            continue
        pairs.append({"id": int(uid), "vad_prob": row["vad_prob"],
                      "reference": reference, "hypothesis": row["text"]})

    volumes: dict[str, int] = defaultdict(int)
    for row in db.execute("SELECT vad_prob FROM utterances"):
        volumes[bucket_of(row["vad_prob"])] += 1

    rep = build_report(pairs, dict(volumes))
    print(f"\npares evaluados: {len(pairs)}  (excluidas: {rep['excluidas']})\n")
    print(f"  {'bucket':<14}{'n':>4}{'WER':>9}{'vol DB':>9}")
    for b, st in rep["buckets"].items():
        print(f"  {b:<14}{st['n']:>4}{st['wer']:>9.3f}{volumes.get(b, 0):>9}")
    print(f"\n  WER plano (del set)          : {rep['wer_simple']:.3f}")
    print(f"  WER re-ponderado (sistema)   : {rep['wer_reponderado']:.3f}")
    print(f"  deleciones (habla → vacío)   : {rep['deleciones_totales']}")
    print(f"  alucinaciones (vacío → texto): {rep['alucinaciones']}")

    out = args.out or f"data/wer_report_{Path(args.groundtruth).stem}.json"
    Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreporte guardado en {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/python -m pytest tests/unit/tools/test_ambient_wer_report.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Correr la suite completa (sin regresiones)**

Run: `.venv/bin/python -m pytest tests/unit/ambient/ tests/unit/tools/ -q`
Expected: PASS. La suite completa (`pytest tests/`) tiene un baseline conocido de fallos preexistentes — compararlo contra el baseline, no contra cero.

- [ ] **Step 6: Commit**

```bash
git add tools/ambient_wer.py tests/unit/tools/test_ambient_wer_report.py
git commit -m "feat(tools): runner de WER por bucket con agregado reponderado"
```

---

## Después del plan: la campaña de medición

El código no da números por sí solo. La secuencia operativa, que **se coordina con el usuario** porque toca una máquina que usa el hogar a diario:

1. Deploy de las tareas 1-3 al server, con `keep_audio.enabled: false` (sin cambio de comportamiento).
2. Verificar espacio libre (`df -h /home`) y prender `keep_audio.enabled: true`. Reiniciar `kza-voice`.
3. Dejar correr ≥48 h.
4. `tools/ambient_groundtruth.py --export` → el usuario transcribe a ciegas.
5. `tools/ambient_wer.py` → números por bucket.
6. **Apagar `keep_audio` y verificar el borrado del audio.**
7. Con esos números se escribe el plan de la Pieza B (fusión multi-mic) — y recién ahí se sabe si `vad_prob` es un criterio de selección válido.

## Notas de riesgo para quien implemente

- La DB de producción ya existe y tiene 5.939 filas: las migraciones de la Task 1 corren vía `ALTER TABLE` sobre ella. **Hacer backup de `ambient.db` antes del deploy.**
- `keep_audio` guarda audio crudo de todo lo que se hable en la casa, incluidas terceras personas. TTL de 48 h; el paso 6 no es opcional.
- El único cambio que toca el hot path del ambient es la Task 3. Si algo sale mal en producción, `keep_audio.enabled: false` restaura el comportamiento exacto de hoy sin revertir código.
