"""Tests for ContextPersister — atomic JSON snapshot per user."""

import json
from pathlib import Path

import pytest

from src.orchestrator.context_persister import ContextPersister, PERSISTED_VERSION
from src.orchestrator.context_manager import UserContext


@pytest.fixture
def tmp_persister(tmp_path: Path) -> ContextPersister:
    return ContextPersister(base_path=tmp_path / "contexts")


def _ctx(user_id: str = "u1", **overrides) -> UserContext:
    base = dict(
        user_id=user_id,
        user_name="Juan",
        compacted_summary="Resumen previo.",
        preserved_ids=["light.escritorio"],
        session_count=2,
    )
    base.update(overrides)
    return UserContext(**base)


class TestContextPersister:
    def test_save_creates_directory_and_file(self, tmp_path: Path):
        persister = ContextPersister(base_path=tmp_path / "contexts")
        persister.save(_ctx())
        assert (tmp_path / "contexts" / "u1.json").exists()

    def test_save_load_roundtrip(self, tmp_persister):
        ctx = _ctx(user_id="alice", compacted_summary="Hola.", preserved_ids=["a.b"])
        tmp_persister.save(ctx)

        data = tmp_persister.load("alice")
        assert data["user_id"] == "alice"
        assert data["compacted_summary"] == "Hola."
        assert data["preserved_ids"] == ["a.b"]
        assert data["version"] == PERSISTED_VERSION
        assert "last_seen" in data

    def test_exists(self, tmp_persister):
        assert tmp_persister.exists("noone") is False
        tmp_persister.save(_ctx(user_id="bob"))
        assert tmp_persister.exists("bob") is True

    def test_load_missing_returns_none(self, tmp_persister):
        assert tmp_persister.load("ghost") is None

    def test_load_corrupt_returns_none_and_logs(self, tmp_persister, caplog):
        path = tmp_persister.base_path / "broken.json"
        tmp_persister.base_path.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json")

        result = tmp_persister.load("broken")
        assert result is None
        assert any("corrupt" in rec.message.lower() or "json" in rec.message.lower()
                   for rec in caplog.records)

    def test_save_atomic_no_partial_file(self, tmp_persister, monkeypatch):
        ctx = _ctx(user_id="atomic", compacted_summary="v1")
        tmp_persister.save(ctx)
        original_path = tmp_persister.base_path / "atomic.json"
        original = original_path.read_text()

        # Provocar fallo durante write
        import os
        real_replace = os.replace

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)

        ctx2 = _ctx(user_id="atomic", compacted_summary="v2_FAILED")
        with pytest.raises(OSError):
            tmp_persister.save(ctx2)

        # El archivo original no debe haber sido pisado
        assert original_path.read_text() == original
        # Y no debe quedar .tmp huérfano viable como contexto
        assert not any(
            p.name == "atomic.json"
            for p in tmp_persister.base_path.iterdir()
            if p.read_text() != original
        )

    def test_user_id_with_path_separator_rejected(self, tmp_persister):
        ctx = _ctx(user_id="../etc/passwd")
        with pytest.raises(ValueError):
            tmp_persister.save(ctx)

    def test_load_version_mismatch_returns_none(self, tmp_persister, caplog):
        base = tmp_persister.base_path
        base.mkdir(parents=True, exist_ok=True)
        (base / "vmismatch.json").write_text(json.dumps({
            "version": 999,  # future schema
            "user_id": "vmismatch",
            "user_name": "X",
            "last_seen": 0.0,
            "session_count": 1,
            "compacted_summary": "ok",
            "preserved_ids": [],
        }))
        result = tmp_persister.load("vmismatch")
        assert result is None
        assert any("version mismatch" in rec.message.lower() for rec in caplog.records)

    def test_load_corrupt_quarantines_file(self, tmp_persister):
        base = tmp_persister.base_path
        base.mkdir(parents=True, exist_ok=True)
        target = base / "qcorrupt.json"
        target.write_text("{ not json")

        result = tmp_persister.load("qcorrupt")
        assert result is None
        # Original file should be moved out of the way
        assert not target.exists()
        # Quarantine file should exist with .corrupt- prefix
        quarantines = list(base.glob("qcorrupt.json.corrupt-*"))
        assert len(quarantines) == 1
