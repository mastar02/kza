"""AmbientStore — persistencia TTL de utterances (data/ambient.db).

Política del spec: destilar-y-descartar. El texto crudo vive acá
retention_hours y la purga lo borra; los hechos destilados viven en la
memoria de largo plazo (ChromaDB). Solo texto por default. Con
``ambient.keep_audio`` activo se guarda además el FLAC del segmento
(columna ``audio_path``) para poder medir WER y re-transcribir; la purga
por TTL borra fila y archivo juntos.

Los lectores aplican el vencimiento sin depender de la purga. Las capturas
nuevas conservan un ID global y una fecha límite que no se extiende al ampliar
la configuración. El inicio del audio manda; una fecha futura queda limitada
por la inserción. Las filas legacy conservan sus metadatos sin inventar IDs.
La purga registra los archivos pendientes en la misma transacción que retira
sus filas; un unlink fallido se reintenta incluso después de reiniciar.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import aiosqlite

from src.ambient.types import SOURCE_VALUES, AmbientUtterance

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  room_id TEXT NOT NULL,
  t0 REAL NOT NULL,
  t1 REAL NOT NULL,
  text TEXT NOT NULL,
  speaker TEXT NOT NULL DEFAULT 'unknown',
  speaker_confidence REAL NOT NULL DEFAULT 0,
  azimuth REAL,
  azimuth_stability REAL NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'unknown',
  confidence REAL,
  no_speech_prob REAL,
  vad_prob REAL,
  lang TEXT,
  lang_prob REAL,
  lang_ok INTEGER,
  during_tts INTEGER NOT NULL DEFAULT 0,
  distilled INTEGER NOT NULL DEFAULT 0,
  text_empty INTEGER NOT NULL DEFAULT 0,
  audio_path TEXT,
  capture_id TEXT,
  expires_at REAL,
  conversation_id TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_room_time ON utterances(room_id, t0);
CREATE INDEX IF NOT EXISTS idx_utt_distill ON utterances(distilled, source);
CREATE TABLE IF NOT EXISTS pending_audio_deletions (
  path TEXT PRIMARY KEY,
  requested_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);
CREATE TABLE IF NOT EXISTS capture_conversations (
  id TEXT PRIMARY KEY, room_id TEXT NOT NULL, started_at REAL NOT NULL, last_seen_at REAL NOT NULL
);
"""

# Columnas agregadas después del deploy Fase 2 (la DB de prod ya existía):
# init() las agrega vía ALTER TABLE si faltan. SQLite no soporta IF NOT
# EXISTS en ADD COLUMN → se inspecciona PRAGMA table_info.
_MIGRATIONS: dict[str, str] = {
    "vad_prob": "ALTER TABLE utterances ADD COLUMN vad_prob REAL",
    # Idioma del texto (flag-no-drop, 2026-06-13). La DB de prod ya tiene
    # vad_prob pero no estas tres → se agregan en orden sin perder filas.
    "lang": "ALTER TABLE utterances ADD COLUMN lang TEXT",
    "lang_prob": "ALTER TABLE utterances ADD COLUMN lang_prob REAL",
    "lang_ok": "ALTER TABLE utterances ADD COLUMN lang_ok INTEGER",
    # Medición de fidelidad (2026-08-05): la DB de prod ya existe.
    "text_empty": "ALTER TABLE utterances ADD COLUMN text_empty INTEGER NOT NULL DEFAULT 0",
    "audio_path": "ALTER TABLE utterances ADD COLUMN audio_path TEXT",
    "capture_id": "ALTER TABLE utterances ADD COLUMN capture_id TEXT",
    "expires_at": "ALTER TABLE utterances ADD COLUMN expires_at REAL",
    "conversation_id": "ALTER TABLE utterances ADD COLUMN conversation_id TEXT",
}

_LIVE = "t0 > :cutoff AND created_at > :cutoff AND (expires_at IS NULL OR expires_at > :now)"
_EXPIRED = "t0 <= :cutoff OR created_at <= :cutoff OR expires_at <= :now"


@dataclass(frozen=True)
class _AudioDeletion:
    path: str
    removed: bool = False
    error: str | None = None


class AmbientStore:
    """CRUD async sobre la tabla utterances, con TTL."""

    def __init__(
        self,
        db_path: str = "./data/ambient.db",
        retention_hours: float = 48.0,
        audio_dir: str | None = None,
        *, clock: Callable[[], float] | None = None,
        conversation_idle_s: float = 120,
    ):
        """Inicializar el store.

        Args:
            db_path: Ruta al archivo SQLite.
            retention_hours: TTL en horas para la purga.
            audio_dir: Raíz de los FLACs de keep_audio. Si se pasa, la purga
                además barre huérfanos (archivos sin fila viva y más viejos
                que el TTL): un unlink fallido o un crash entre write() y
                set_audio_path() dejaban audio crudo del hogar fuera de la
                política de retención para siempre (review PR #14).
            clock: Reloj de segundos epoch para vencimiento y pruebas.
        """
        if type(retention_hours) not in (int, float) or not math.isfinite(retention_hours) or not 0 < retention_hours <= 48:
            raise ValueError("Capture retention must be positive and at most 48 hours")
        self.db_path = db_path
        if type(conversation_idle_s) not in (int, float) or not math.isfinite(conversation_idle_s) or not 0 < conversation_idle_s <= 3600:
            raise ValueError("invalid_capture_conversation_idle")
        self.conversation_idle_s = conversation_idle_s
        self.retention_hours = retention_hours
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self._db: aiosqlite.Connection | None = None
        self._clock = time.time if clock is None else clock
        self._write_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self.expiry_changed = asyncio.Event()
        self._next_expiry: float | None = None

    async def init(self) -> None:
        """Crear/abrir la DB y aplicar el schema."""
        await asyncio.to_thread(Path(self.db_path).parent.mkdir, parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        cur = await self._db.execute("PRAGMA table_info(utterances)")
        existing = {row["name"] for row in await cur.fetchall()}
        for col, ddl in _MIGRATIONS.items():
            if col not in existing:
                await self._db.execute(ddl)
                logger.info("AmbientStore migración: columna %s agregada", col)
        await self._db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_utt_capture ON utterances(capture_id) WHERE capture_id IS NOT NULL")
        await self._db.commit()
        logger.info("AmbientStore listo (%s, TTL %.1fh)", self.db_path, self.retention_hours)

    async def close(self) -> None:
        """Cerrar la conexión."""
        async with self._cleanup_lock, self._write_lock:
            if self._db is not None:
                await self._db.close()
                self._db = None

    def _retention_parameters(self) -> dict:
        now = self._clock()
        return {"now": now, "cutoff": now - self.retention_hours * 3600}

    async def next_expiration_delay(self) -> float:
        """Time until the earliest capture deadline or a pending retry."""
        cur = await self._db.execute(
            "SELECT MIN(min(COALESCE(expires_at, t0+:ttl), t0+:ttl, created_at+:ttl)) AS deadline FROM utterances",
            {"ttl": self.retention_hours * 3600},
        )
        self._next_expiry = (await cur.fetchone())["deadline"]
        delay = 3600.0 if self._next_expiry is None else max(0.0, self._next_expiry - self._clock())
        if (await self.audio_cleanup_status())["pending"]:
            delay = min(delay, 60.0)
        return min(delay, 3600.0)

    async def _mutate(self, operation):
        """Keep a SQLite transaction atomic and owned through cancellation."""
        async def transaction():
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                result = await operation()
                await self._db.commit()
                return result
            except BaseException:
                await self._db.rollback()
                raise

        async with self._write_lock:
            return await self._finish_owned(transaction())

    async def _finish_owned(self, operation):
        task = asyncio.create_task(operation)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def add(self, utt: AmbientUtterance) -> int:
        """Insertar utterance; devuelve el id asignado.

        Args:
            utt: Utterance a persistir.

        Returns:
            rowid (int) recién creado.

        Raises:
            ValueError: si utt.source no está en SOURCE_VALUES.
        """
        if utt.source not in SOURCE_VALUES:
            raise ValueError(f"source inválido: {utt.source!r} (∉ {SOURCE_VALUES})")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (utt.t0, utt.t1)) or utt.t1 < utt.t0:
            raise ValueError("Capture timestamps must be finite and ordered")
        now = self._clock()
        expires_at = min(utt.t0, now) + self.retention_hours * 3600
        utt.expires_at = expires_at

        async def insert():
            if utt.conversation_id is None:
                cur = await self._db.execute(
                    "SELECT id FROM capture_conversations WHERE room_id=? AND started_at<=? AND last_seen_at>=? AND started_at>=? "
                    "ORDER BY last_seen_at DESC LIMIT 1", (utt.room_id, utt.t0, utt.t0-self.conversation_idle_s, utt.t0-3600),
                )
                conversation = await cur.fetchone()
                utt.conversation_id = conversation["id"] if conversation else str(uuid4())
            await self._db.execute(
                "INSERT INTO capture_conversations VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET last_seen_at=max(last_seen_at,excluded.last_seen_at)",
                (utt.conversation_id, utt.room_id, utt.t0, utt.t1),
            )
            cur = await self._db.execute(
                """INSERT INTO utterances
                   (room_id, t0, t1, text, speaker, speaker_confidence, azimuth,
                    azimuth_stability, source, confidence, no_speech_prob,
                    vad_prob, lang, lang_prob, lang_ok, during_tts, distilled,
                    text_empty, audio_path, created_at, capture_id, expires_at, conversation_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    utt.room_id, utt.t0, utt.t1, utt.text, utt.speaker,
                    utt.speaker_confidence, utt.azimuth, utt.azimuth_stability,
                    utt.source, utt.confidence, utt.no_speech_prob,
                    utt.vad_prob, utt.lang, utt.lang_prob,
                    None if utt.lang_ok is None else int(utt.lang_ok),
                    int(utt.during_tts), int(utt.distilled),
                    int(utt.text_empty), utt.audio_path,
                    now, utt.capture_id, expires_at, utt.conversation_id,
                ),
            )
            if self._next_expiry is None or expires_at < self._next_expiry:
                self._next_expiry = expires_at
                self.expiry_changed.set()
            return cur.lastrowid
        return await self._mutate(insert)

    async def set_audio_path(self, utt_id: int, path: str) -> bool:
        """Registrar la ruta del FLAC archivado para una utterance.

        Se llama después de escribir el archivo, porque el nombre depende del
        id que devuelve add(). Un fallo de escritura deja audio_path en NULL.

        Args:
            utt_id: rowid devuelto por add().
            path: ruta del archivo, tal como la devolvió el AudioArchiver.

        Returns:
            True si la captura sigue vigente. False registra el archivo para
            borrado durable sin recrear una captura vencida o ausente.
        """
        async def attach():
            parameters = self._retention_parameters() | {"id": utt_id, "path": path}
            cur = await self._db.execute(
                "UPDATE utterances SET audio_path=:path WHERE id=:id AND " + _LIVE, parameters,  # nosec B608 -- Predicate is constant; all values use bound parameters.
            )
            if cur.rowcount == 0:
                await self._queue_audio_deletion(path)
            return cur.rowcount > 0

        attached = await self._mutate(attach)
        if not attached:
            logger.warning(
                "AmbientStore.set_audio_path: id %d missing or expired; cleanup queued", utt_id,
            )
            await self._retry_audio_deletions()
        return attached

    async def read_audio(self, utt_id: int) -> bytes | None:
        """Read live clip bytes, rechecking expiry after asynchronous disk I/O."""
        async def live_path():
            cur = await self._db.execute(
                "SELECT audio_path FROM utterances WHERE id=:id AND " + _LIVE,  # nosec B608 -- Predicate is constant; all values use bound parameters.
                self._retention_parameters() | {"id": utt_id},
            )
            row = await cur.fetchone()
            return row["audio_path"] if row else None

        path = await live_path()
        if not path:
            return None

        def read():
            return self._audio_path(path).read_bytes()

        try:
            data = await asyncio.to_thread(read)
        except (OSError, ValueError):
            return None
        return data if await live_path() == path else None

    async def utterances_between(
        self, room_id: str, t0: float, t1: float
    ) -> list[dict]:
        """Utterances de una room cuyo inicio cae en [t0, t1].

        Args:
            room_id: Identificador de habitación.
            t0: Límite inferior de timestamp de audio (inclusive).
            t1: Límite superior de timestamp de audio (inclusive).

        Returns:
            Lista de dicts ordenada por t0 ascendente.
        """
        cur = await self._db.execute(
            "SELECT * FROM utterances WHERE room_id=:room AND t0>=:start AND t0<=:end AND " + _LIVE + " ORDER BY t0",  # nosec B608 -- Predicate is constant; all values use bound parameters.
            self._retention_parameters() | {"room": room_id, "start": t0, "end": t1},
        )
        return [dict(r) for r in await cur.fetchall()]

    async def undistilled_live(
        self, limit: int = 200, min_vad_prob: float = 0.0, spanish_only: bool = False,
        require_provenance: bool = False,
    ) -> list[dict]:
        """Lote para el Distiller: utterances destilables sin destilar.

        Selecciona por blacklist de fuente en vez de whitelist 'live': sin
        enrollment ni DoA estable el classifier marca TODO 'unknown' (medido
        en prod: 1833/1833), así que filtrar por source='live' devolvería
        cero. Se incluye 'unknown' (el grueso real) y 'live'; se excluyen
        'self' (nuestro TTS) y 'tv' (cuando el DoA lo marque).

        La compuerta de calidad real es ``min_vad_prob`` (Silero): la señal
        anti-alucinación validada sobre ambient.db (cruce español-hogar /
        bleed-TV ≈ 0.45). ``no_speech_prob`` quedó muerto (degenerado ~0).

        Excluye ``text_empty=1``: son filas que existen solo para medir la
        tasa de deleción (segmentos sin texto) y no tienen nada que destilar.

        Args:
            limit: Máximo de filas a devolver.
            min_vad_prob: Umbral inferior de vad_prob (vad NULL cuenta como 0).
            spanish_only: si True, excluye utterances marcadas no-conservables
                (``lang_ok=0``); conserva las conservables (``lang_ok=1``) y las
                sin clasificar (``lang_ok NULL`` → COALESCE 1, filas viejas no se
                pierden). Es el filtro flag-no-drop: el no-español persiste en la
                DB (audit/re-train) pero no se destila a memoria.

        Returns:
            Lista de dicts ordenada por t0 ascendente.
        """
        query = (
            "SELECT * FROM utterances WHERE distilled=0 "  # nosec B608 -- Predicate is constant; all values use bound parameters.
            "AND source NOT IN ('self','tv') "
            "AND COALESCE(text_empty, 0) = 0 "
            "AND COALESCE(vad_prob, 0) >= :vad AND " + _LIVE + " "
        )
        params = self._retention_parameters() | {"vad": min_vad_prob, "limit": limit}
        if spanish_only:
            query += "AND COALESCE(lang_ok, 1) = 1 "
        if require_provenance:
            query += "AND capture_id IS NOT NULL AND conversation_id IS NOT NULL AND expires_at IS NOT NULL "
        query += "ORDER BY t0 LIMIT :limit"
        cur = await self._db.execute(query, params)
        return [dict(r) for r in await cur.fetchall()]

    async def mark_distilled(self, ids: list[int]) -> None:
        """Marcar utterances como destiladas (no se reprocesarán).

        Args:
            ids: Lista de rowids a marcar.
        """
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        async def mark():
            await self._db.execute(
                f"UPDATE utterances SET distilled=1 WHERE id IN ({marks})", ids  # nosec B608 -- Only placeholders are interpolated.
            )
        await self._mutate(mark)

    async def spanish_keep_rate(self, since_s: float | None = None) -> dict:
        """Tasa de español conservable (``lang_ok``) — métrica de producto del ambient.

        El 24/7 existe para saber si el ASR detecta bien el español, no para
        destilar charla. ``rate`` es ok / (ok + not_ok); las filas sin clasificar
        (``lang_ok`` NULL, DBs viejas) se cuentan aparte y no entran al rate.

        Args:
            since_s: Si se pasa, solo utterances con ``t0 >= since_s``.

        Returns:
            Dict con n, ok, not_ok, unclassified, rate (float|None), by_room.
        """
        where = "WHERE " + _LIVE
        params = self._retention_parameters()
        if since_s is not None:
            where += " AND t0 >= :since"
            params["since"] = since_s
        cur = await self._db.execute(
            f"SELECT "
            f"COUNT(*) AS n, "
            f"SUM(CASE WHEN lang_ok = 1 THEN 1 ELSE 0 END) AS ok, "
            f"SUM(CASE WHEN lang_ok = 0 THEN 1 ELSE 0 END) AS not_ok, "
            f"SUM(CASE WHEN lang_ok IS NULL THEN 1 ELSE 0 END) AS unclassified "
            f"FROM utterances {where}",  # nosec B608 -- where es constante o 'WHERE t0 >= ?'
            params,
        )
        row = await cur.fetchone()
        n = int(row["n"] or 0)
        ok = int(row["ok"] or 0)
        not_ok = int(row["not_ok"] or 0)
        unclassified = int(row["unclassified"] or 0)
        classified = ok + not_ok
        rate = (ok / classified) if classified else None
        room_cur = await self._db.execute(
            f"SELECT room_id, "
            f"COUNT(*) AS n, "
            f"SUM(CASE WHEN lang_ok = 1 THEN 1 ELSE 0 END) AS ok, "
            f"SUM(CASE WHEN lang_ok = 0 THEN 1 ELSE 0 END) AS not_ok, "
            f"SUM(CASE WHEN lang_ok IS NULL THEN 1 ELSE 0 END) AS unclassified "
            f"FROM utterances {where} GROUP BY room_id",  # nosec B608
            params,
        )
        by_room = {}
        for r in await room_cur.fetchall():
            r_ok = int(r["ok"] or 0)
            r_not = int(r["not_ok"] or 0)
            r_class = r_ok + r_not
            by_room[r["room_id"]] = {
                "n": int(r["n"] or 0),
                "ok": r_ok,
                "not_ok": r_not,
                "unclassified": int(r["unclassified"] or 0),
                "rate": (r_ok / r_class) if r_class else None,
            }
        return {
            "n": n,
            "ok": ok,
            "not_ok": not_ok,
            "unclassified": unclassified,
            "rate": rate,
            "by_room": by_room,
        }

    async def purge_expired(self) -> int:
        """Borrar utterances cuyo audio es más viejo que retention_hours.

        El plazo usa el inicio del audio, la inserción como límite para fechas
        futuras y el vencimiento persistido de las capturas nuevas. Las rutas
        se registran para cleanup antes de eliminar las filas, atómicamente.

        Returns:
            Número de filas borradas.
        """
        parameters = self._retention_parameters()

        async def expire():
            cur = await self._db.execute(
                "SELECT audio_path FROM utterances WHERE (" + _EXPIRED + ") AND audio_path IS NOT NULL", parameters,  # nosec B608 -- Predicate is constant; all values use bound parameters.
            )
            for row in await cur.fetchall():
                await self._queue_audio_deletion(row["audio_path"])
            cur = await self._db.execute("DELETE FROM utterances WHERE " + _EXPIRED, parameters)  # nosec B608 -- Predicate is constant; all values use bound parameters.
            await self._db.execute("DELETE FROM capture_conversations WHERE id NOT IN (SELECT conversation_id FROM utterances WHERE conversation_id IS NOT NULL)")
            return cur.rowcount

        deleted = await self._mutate(expire)
        unlinked = await self._retry_audio_deletions()
        orphans = 0
        if self.audio_dir is not None:
            id_cur = await self._db.execute("SELECT id FROM utterances")
            live_ids = {row["id"] for row in await id_cur.fetchall()}
            orphans = await asyncio.to_thread(self._sweep_orphans, live_ids, parameters["cutoff"])
        if deleted or unlinked or orphans:
            logger.info("AmbientStore expiry: rows=%d audio=%d orphans=%d", deleted, unlinked, orphans)
        return deleted

    async def _queue_audio_deletion(self, path: str) -> None:
        # Called only inside the same transaction that removes the reference.
        await self._db.execute(
            "INSERT OR IGNORE INTO pending_audio_deletions(path, requested_at) VALUES (?, ?)",
            (path, self._clock()),
        )
        self.expiry_changed.set()

    async def discard_audio(self, path: str) -> None:
        """Durably discard an unaccepted or cancelled archive, including retries."""
        await self._mutate(lambda: self._queue_audio_deletion(path))
        await self._retry_audio_deletions()

    async def audio_cleanup_status(self) -> dict:
        """Report pending file removal without exposing clip paths or content."""
        cur = await self._db.execute(
            "SELECT COUNT(*) AS pending, COALESCE(SUM(attempts), 0) AS attempts, "
            "MIN(requested_at) AS oldest_request FROM pending_audio_deletions"
        )
        return dict(await cur.fetchone())

    async def _retry_audio_deletions(self) -> int:
        async def attempt():
            cur = await self._db.execute(
                "SELECT path FROM pending_audio_deletions ORDER BY attempts, requested_at, path LIMIT 256"
            )
            paths = [row["path"] for row in await cur.fetchall()]
            if not paths:
                return 0
            outcomes = await asyncio.to_thread(self._unlink_outcomes, paths)

            async def acknowledge():
                for result in outcomes:
                    if result.error is None:
                        await self._db.execute("DELETE FROM pending_audio_deletions WHERE path=?", (result.path,))
                    else:
                        await self._db.execute(
                            "UPDATE pending_audio_deletions SET attempts=attempts+1, last_error=? WHERE path=?",
                            (result.error, result.path),
                        )

            await self._mutate(acknowledge)
            return sum(result.removed for result in outcomes)

        async with self._cleanup_lock:
            return await self._finish_owned(attempt())

    def _audio_path(self, path: str) -> Path:
        candidate = Path(path)
        if self.audio_dir is not None:
            root = self.audio_dir.resolve()
            resolved = candidate.resolve()
            if resolved == root or not resolved.is_relative_to(root) or candidate.suffix != ".flac":
                raise ValueError("audio_path_outside_archive")
        return candidate

    def _unlink_outcomes(self, paths: list[str]) -> list[_AudioDeletion]:
        outcomes = []
        for path in paths:
            try:
                self._audio_path(path).unlink(missing_ok=False)
                outcomes.append(_AudioDeletion(path, removed=True))
            except FileNotFoundError:
                outcomes.append(_AudioDeletion(path))
            except (OSError, ValueError) as error:
                logger.warning("AmbientStore audio cleanup failed: %s", type(error).__name__)
                outcomes.append(_AudioDeletion(path, error=type(error).__name__))
        return outcomes

    def _unlink_batch(self, paths: list[str]) -> int:
        """Count actual removals; a missing file is already reconciled."""
        return sum(result.removed for result in self._unlink_outcomes(paths))

    def _sweep_orphans(self, live_ids: set[int], cutoff: float) -> int:
        """Borrar FLACs sin fila viva y más viejos que el TTL.

        La guarda de mtime protege la carrera write()→set_audio_path(): un
        FLAC recién escrito todavía sin puntero NO es huérfano. Un archivo
        cuyo unlink acaba de fallar en `_unlink_batch` (fila ya borrada) es
        huérfano YA en este mismo `purge_expired()` — el sweep corre
        DESPUÉS de `_unlink_batch` en la misma llamada, así que lo reintenta
        de inmediato; no hace falta esperar al próximo ciclo de purga.
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
                # Guarda de seguridad (M4): `utt_id in live_ids` es correcta
                # entre purgas porque `utterances.id` es AUTOINCREMENT — SQLite
                # nunca reusa un rowid mientras la tabla exista, así que un id
                # que ya no está en live_ids no puede "revivir" con otro
                # contenido. Límite conocido: si la DB se borra y se recrea
                # desde cero (no un ALTER, un archivo nuevo), AUTOINCREMENT
                # reinicia en 1 y un FLAC viejo puede quedar sombreado por una
                # fila nueva homónima (mismo id, contenido distinto) — ese
                # caso no lo cubre esta guarda.
                if utt_id in live_ids or f.stat().st_mtime >= cutoff:
                    continue
                swept += self._unlink_batch([str(f)])
            except OSError as e:
                logger.warning(
                    "AmbientStore sweep: no se pudo borrar/inspeccionar %s: %s", f, e
                )
        return swept
