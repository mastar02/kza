"""Tests de las funciones puras de scripts/sync_ha_to_chroma.py (sin acceso a HA)."""
import importlib


def test_module_imports_without_env(monkeypatch):
    # Sin HOME_ASSISTANT_* y sin .env, el import no debe crashear.
    monkeypatch.delenv("HOME_ASSISTANT_URL", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    mod = importlib.import_module("scripts.sync_ha_to_chroma")
    assert hasattr(mod, "is_group_entity")
    assert hasattr(mod, "main")
