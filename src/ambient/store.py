"""AmbientStore — persistencia TTL de utterances (data/ambient.db).

Política del spec: destilar-y-descartar. El texto crudo vive acá
retention_hours y la purga lo borra; los hechos destilados viven en la
memoria de largo plazo (ChromaDB). Solo texto — jamás audio.

Nota de diseño (purga):
  purge_expired() borra por `t0 < cutoff`, NO por `created_at`.
  Semántica: el TTL aplica a la ANTIGÜEDAD DEL AUDIO, no al momento
  de inserción. En operación normal ambos son casi iguales; en tests
  y en replays difieren. created_at sigue siendo time.time() del INSERT
  para auditoría/debugging.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

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
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_room_time ON utterances(room_id, t0);
CREATE INDEX IF NOT EXISTS idx_utt_distill ON utterances(distilled, source);
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
}


class AmbientStore:
    """CRUD async sobre la tabla utterances, con TTL."""

    def __init__(self, db_path: str = "./data/ambient.db", retention_hours: float = 12.0):
        self.db_path = db_path
        self.retention_hours = retention_hours
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Crear/abrir la DB y aplicar el schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        cur = await self._db.execute("PRAGMA table_info(utterances)")
        existing = {row["name"] for row in await cur.fetchall()}
        for col, ddl in _MIGRATIONS.items():
            if col not in existing:
                await self._db.execute(ddl)
                logger.info("AmbientStore migración: columna %s agregada", col)
        await self._db.commit()
        logger.info("AmbientStore listo (%s, TTL %.1fh)", self.db_path, self.retention_hours)

    async def close(self) -> None:
        """Cerrar la conexión."""
        if self._db is not None:
            await self._db.close()
            self._db = None

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
        cur = await self._db.execute(
            """INSERT INTO utterances
               (room_id, t0, t1, text, speaker, speaker_confidence, azimuth,
                azimuth_stability, source, confidence, no_speech_prob,
                vad_prob, lang, lang_prob, lang_ok, during_tts, distilled,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                utt.room_id, utt.t0, utt.t1, utt.text, utt.speaker,
                utt.speaker_confidence, utt.azimuth, utt.azimuth_stability,
                utt.source, utt.confidence, utt.no_speech_prob,
                utt.vad_prob, utt.lang, utt.lang_prob,
                None if utt.lang_ok is None else int(utt.lang_ok),
                int(utt.during_tts), int(utt.distilled),
                time.time(),
            ),
        )
        await self._db.commit()
        return cur.lastrowid

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
            "SELECT * FROM utterances WHERE room_id=? AND t0>=? AND t0<=? ORDER BY t0",
            (room_id, t0, t1),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def undistilled_live(
        self, limit: int = 200, min_vad_prob: float = 0.0, spanish_only: bool = False
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
            "SELECT * FROM utterances WHERE distilled=0 "
            "AND source NOT IN ('self','tv') "
            "AND COALESCE(vad_prob, 0) >= ? "
        )
        params: list = [min_vad_prob]
        if spanish_only:
            query += "AND COALESCE(lang_ok, 1) = 1 "
        query += "ORDER BY t0 LIMIT ?"
        params.append(limit)
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
        await self._db.execute(
            f"UPDATE utterances SET distilled=1 WHERE id IN ({marks})", ids
        )
        await self._db.commit()

    async def purge_expired(self) -> int:
        """Borrar utterances cuyo audio es más viejo que retention_hours.

        La purga filtra por `t0` (antigüedad del audio), no por `created_at`
        (momento de inserción). Esto garantiza TTL correcto tanto en operación
        normal como en replays/tests donde t0 puede diferir mucho de created_at.

        Returns:
            Número de filas borradas.
        """
        cutoff = time.time() - self.retention_hours * 3600
        cur = await self._db.execute(
            "DELETE FROM utterances WHERE t0 < ?", (cutoff,)
        )
        await self._db.commit()
        if cur.rowcount:
            logger.info("AmbientStore purga: %d utterances borradas (TTL %.1fh)",
                        cur.rowcount, self.retention_hours)
        return cur.rowcount
