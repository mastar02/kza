"""Tests: AmbientStore — SQLite TTL para utterances ambientales."""
import asyncio
import time

from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance


def _utt(t0: float, text: str = "hola", source: str = "live", **kw) -> AmbientUtterance:
    return AmbientUtterance(
        room_id=kw.pop("room_id", "escritorio"), t0=t0, t1=t0 + 2.0,
        text=text, source=source, **kw,
    )


def _run(coro):
    return asyncio.run(coro)


def test_add_and_query_between(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "ambient.db"), retention_hours=12)
        await store.init()
        now = time.time()
        await store.add(_utt(now - 10, text="primera"))
        await store.add(_utt(now - 5, text="segunda"))
        await store.add(_utt(now - 5, text="otra room", room_id="living"))

        rows = await store.utterances_between("escritorio", now - 7, now)
        assert [r["text"] for r in rows] == ["segunda"]
        await store.close()
    _run(inner())


def test_undistilled_live_and_mark(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        now = time.time()
        id_live = await store.add(_utt(now, text="dato útil", source="live"))
        await store.add(_utt(now, text="ruido tele", source="tv"))
        await store.add(_utt(now, text="yo mismo", source="self"))

        batch = await store.undistilled_live(limit=10)
        assert [r["id"] for r in batch] == [id_live]

        await store.mark_distilled([id_live])
        assert await store.undistilled_live(limit=10) == []
        await store.close()
    _run(inner())


def test_purge_expired(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        now = time.time()
        await store.add(_utt(now - 7200, text="vieja"))   # 2h: expira
        await store.add(_utt(now - 60, text="fresca"))
        deleted = await store.purge_expired()
        assert deleted == 1
        rows = await store.utterances_between("escritorio", 0, now + 10)
        assert [r["text"] for r in rows] == ["fresca"]
        await store.close()
    _run(inner())


def test_vad_prob_roundtrip(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=12)
        await store.init()
        now = time.time()
        await store.add(_utt(now, text="con vad", vad_prob=0.42))
        await store.add(_utt(now, text="sin vad"))  # default None
        rows = await store.utterances_between("escritorio", now - 1, now + 1)
        by_text = {r["text"]: r["vad_prob"] for r in rows}
        assert by_text["con vad"] == 0.42
        assert by_text["sin vad"] is None
        await store.close()
    _run(inner())


def test_init_migrates_old_schema_adding_vad_prob(tmp_path):
    # La DB de prod (deploy Fase 2 del 2026-06-06) ya existe SIN vad_prob:
    # init() debe agregar la columna sin perder filas.
    import sqlite3

    db_path = str(tmp_path / "legacy.db")
    legacy_schema = """
    CREATE TABLE utterances (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      room_id TEXT NOT NULL, t0 REAL NOT NULL, t1 REAL NOT NULL,
      text TEXT NOT NULL, speaker TEXT NOT NULL DEFAULT 'unknown',
      speaker_confidence REAL NOT NULL DEFAULT 0, azimuth REAL,
      azimuth_stability REAL NOT NULL DEFAULT 0,
      source TEXT NOT NULL DEFAULT 'unknown', confidence REAL,
      no_speech_prob REAL, during_tts INTEGER NOT NULL DEFAULT 0,
      distilled INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
    );
    """
    conn = sqlite3.connect(db_path)
    conn.executescript(legacy_schema)
    conn.execute(
        "INSERT INTO utterances (room_id, t0, t1, text, created_at) "
        "VALUES ('escritorio', 100.0, 102.0, 'fila vieja', 100.0)"
    )
    conn.commit()
    conn.close()

    async def inner():
        store = AmbientStore(db_path=db_path, retention_hours=12)
        await store.init()  # debe migrar sin romper
        rows = await store.utterances_between("escritorio", 0, 200)
        assert [r["text"] for r in rows] == ["fila vieja"]
        assert rows[0]["vad_prob"] is None
        # y aceptar inserts con la columna nueva
        await store.add(_utt(150.0, text="fila nueva", vad_prob=0.8))
        rows = await store.utterances_between("escritorio", 0, 200)
        assert {r["text"]: r["vad_prob"] for r in rows} == {
            "fila vieja": None, "fila nueva": 0.8,
        }
        await store.close()
    _run(inner())


def test_add_validates_source(tmp_path):
    async def inner():
        store = AmbientStore(db_path=str(tmp_path / "a.db"), retention_hours=1)
        await store.init()
        try:
            await store.add(_utt(time.time(), source="martian"))
            raised = False
        except ValueError:
            raised = True
        assert raised
        await store.close()
    _run(inner())
