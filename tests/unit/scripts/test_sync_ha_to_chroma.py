"""Tests de las funciones puras de scripts/sync_ha_to_chroma.py (sin acceso a HA)."""
import importlib


def test_module_imports_without_env(monkeypatch):
    # Sin HOME_ASSISTANT_* y sin .env, el import no debe crashear.
    monkeypatch.delenv("HOME_ASSISTANT_URL", raising=False)
    monkeypatch.delenv("HOME_ASSISTANT_TOKEN", raising=False)
    mod = importlib.import_module("scripts.sync_ha_to_chroma")
    assert hasattr(mod, "is_group_entity")
    assert hasattr(mod, "main")


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
