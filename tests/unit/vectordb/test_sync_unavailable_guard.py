"""Syncing a dead entity writes a permanently impoverished index."""
import sys

import pytest

import scripts.sync_ha_to_chroma as sync_mod
from scripts.sync_ha_to_chroma import cache_key, select_syncable


def test_dead_entities_are_separated_from_live_ones():
    entities = [
        {"entity_id": "light.grupo_living", "state": "off"},
        {"entity_id": "light.grupo_cuarto", "state": "unavailable"},
        {"entity_id": "light.grupo_bano", "state": "unknown"},
    ]
    live, dead = select_syncable(entities)
    assert [e["entity_id"] for e in live] == ["light.grupo_living"]
    assert sorted(e["entity_id"] for e in dead) == [
        "light.grupo_bano", "light.grupo_cuarto",
    ]


def test_all_live_returns_no_dead():
    entities = [{"entity_id": "light.grupo_living", "state": "on"}]
    live, dead = select_syncable(entities)
    assert len(live) == 1 and dead == []


_BASE_KEY = dict(entity_id="light.grupo_cuarto", friendly_name="Cuarto",
                 area="cuarto", capability="onoff", value="on")


def test_cache_key_changes_when_entity_recovers():
    """Revivir (muerta → viva) tiene que invalidar la key.

    Es lo que fuerza la reindexación: si la vitalidad no entrara en la key, el
    sync incremental vería la misma de siempre y saltearía la entidad, dejando
    el índice empobrecido que se generó mientras estaba caída.
    """
    key_unavailable = cache_key(**_BASE_KEY, state="unavailable")
    key_on = cache_key(**_BASE_KEY, state="on")
    assert key_unavailable != key_on
    # Determinismo: mismo estado -> misma key.
    assert cache_key(**_BASE_KEY, state="on") == key_on


@pytest.mark.parametrize("vivo_a, vivo_b", [
    ("on", "off"),            # el caso que rompía el cache: las luces conmutan
    ("on", "playing"),
    ("off", None),            # sin estado != muerta
    ("unknown_pero_vivo", "idle"),
])
def test_cache_key_is_stable_across_live_states(vivo_a, vivo_b):
    """Dos estados VIVOS distintos deben dar la MISMA key.

    Meter el estado crudo en la key costaba carísimo y no entregaba lo que
    prometía: `"on"` y `"off"` producían keys distintas, así que el sync
    incremental dejaba de acertar casi siempre (las luces conmutan entre
    corridas → ~100-120 llamadas al LLM en la primera corrida post-deploy). Y
    encima no lograba el objetivo: `collection.add()` NO borra los documentos
    viejos, así que los documentos empobrecidos que el cambio existía para
    invalidar sobrevivían igual.

    El bucket grueso vivo/muerta conserva el objetivo real —que la recuperación
    invalide la key— sin destruir el cache incremental.
    """
    assert cache_key(**_BASE_KEY, state=vivo_a) == cache_key(**_BASE_KEY, state=vivo_b)


@pytest.mark.parametrize("muerta_a, muerta_b", [("unavailable", "unknown")])
def test_cache_key_is_stable_across_dead_states(muerta_a, muerta_b):
    """Los estados que `select_syncable` considera muertos comparten bucket."""
    assert cache_key(**_BASE_KEY, state=muerta_a) == cache_key(
        **_BASE_KEY, state=muerta_b
    )


def test_cache_key_bucket_matches_what_select_syncable_excludes():
    """El bucket y el criterio de exclusión tienen que ser el mismo.

    Si `select_syncable` excluyera un estado que el bucket cuenta como vivo (o
    al revés), habría entidades cuya recuperación no invalida su key: volverían
    a indexarse con el documento empobrecido y el bug volvería en silencio.
    """
    estados = ["unavailable", "unknown", "on", "off", "idle", None]
    for estado in estados:
        excluida_del_sync = select_syncable([{"entity_id": "x", "state": estado}])[0] == []
        key_de_muerta = cache_key(**_BASE_KEY, state=estado) == cache_key(
            **_BASE_KEY, state="unavailable"
        )
        assert excluida_del_sync == key_de_muerta, (
            f"state={estado!r}: select_syncable y el bucket de cache_key no coinciden"
        )


def _fake_entity(entity_id: str, state: str, friendly_name: str | None = None):
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": {"friendly_name": friendly_name or entity_id},
    }


def test_main_aborts_with_exit_2_before_touching_chroma_when_entity_unavailable(monkeypatch):
    """El guard real de main() debe abortar (exit 2) ANTES de tocar Chroma.

    Los tests de arriba ejercitan select_syncable() en aislamiento; este
    ejercita el guard tal como vive en main(), con HA mockeado. Si alguien
    borra el sys.exit(2) del guard, este test debe fallar.
    """
    fake_states = [
        _fake_entity("light.grupo_living", "on", "Living"),
        _fake_entity("light.grupo_cuarto", "unavailable"),
    ]
    monkeypatch.setattr(sync_mod, "ha_get", lambda path: fake_states)
    monkeypatch.setattr(sync_mod, "resolve_areas", lambda entity_ids: {})

    class _BoomChroma:
        """Si el guard no abortara, main() haría `import chromadb` y tocaría
        el índice local a continuación. Lo bloqueamos con un error claro en
        vez de arriesgar tocar `data/chroma_db` real desde un test unitario."""

        def __getattr__(self, name):
            raise AssertionError(
                "el guard debería haber abortado antes de llegar a chromadb"
            )

    monkeypatch.setitem(sys.modules, "chromadb", _BoomChroma())
    monkeypatch.setattr(sys, "argv", ["sync_ha_to_chroma.py"])

    with pytest.raises(SystemExit) as excinfo:
        sync_mod.main()
    assert excinfo.value.code == 2


def test_main_allow_unavailable_still_logs_excluded_entities(monkeypatch, capsys):
    """Con --allow-unavailable no debe abortar, pero las entidades excluidas
    tienen que quedar logueadas igual (antes el log solo pasaba en la rama de
    abort, así que con el flag activo el operador no se enteraba de qué
    entidad quedó afuera y sin ninguna cobertura en el índice)."""
    fake_states = [_fake_entity("light.grupo_cuarto", "unavailable")]
    monkeypatch.setattr(sync_mod, "ha_get", lambda path: fake_states)
    monkeypatch.setattr(sync_mod, "resolve_areas", lambda entity_ids: {})
    monkeypatch.setattr(
        sys, "argv", ["sync_ha_to_chroma.py", "--allow-unavailable"],
    )

    # Todas las entidades quedan afuera (única y unavailable) -> selected
    # termina vacío -> main() retorna por el early-return de "Nada que
    # indexar", sin llegar a tocar Chroma. Así el test verifica el logueo
    # "siempre" sin necesitar stubear el resto del pipeline (LLM/embedder).
    sync_mod.main()

    err = capsys.readouterr().err
    assert "light.grupo_cuarto" in err
    assert "unavailable" in err
    assert "EXCLUIDAS" in err
