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
