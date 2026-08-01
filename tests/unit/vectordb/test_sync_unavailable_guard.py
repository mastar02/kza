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


def test_cache_key_does_not_take_state():
    """`cache_key` no recibe `state`: la exclusión de muertas la hace
    `select_syncable` ANTES de que se calcule ninguna key (ver main()), así
    que para cuando `cache_key` corre, `state` siempre es "vivo" — un bucket
    vivo/muerto acá sería una rama que nunca se alcanza. Meter el estado
    crudo en la key en cambio sí tenía costo real: "on"/"off" producían keys
    distintas, así que el sync incremental dejaba de acertar en cada corrida
    (las luces conmutan de estado) y encima no invalidaba nada, porque
    `collection.add()` no borra los documentos viejos.
    """
    assert cache_key(**_BASE_KEY) == cache_key(**_BASE_KEY)
    # Mismos datos -> misma key, determinístico, sin ningún parámetro de estado.
    otra = dict(_BASE_KEY, value="off")
    assert cache_key(**_BASE_KEY) != cache_key(**otra)


def test_cache_key_format_is_pinned_against_reintroducing_a_state_suffix():
    """Golden hash: fija el FORMATO exacto de la key, no solo su determinismo.

    Este es el test que mata la mutación real que este PR corrige: reintroducir
    `state`/`vitality` en `cache_key` (con cualquier default, incluso uno que
    nunca dispara la rama "dead") cambia el string crudo que se hashea y por lo
    tanto la key completa. Ese cambio de formato es justo lo que rompía el sync
    incremental en producción: el primer deploy post-cambio no matchea NINGÚN
    `cache_key` existente en Chroma, reprocesa todo (~100-120 llamadas al LLM) y
    -como `collection.add()` no borra- duplica cada documento (~613) en vez de
    reemplazarlo. `test_cache_key_does_not_take_state` de arriba NO alcanza a
    detectar esto (compara la función contra sí misma, así que sigue siendo
    consistente aunque el formato cambie); este test ata el hash a un valor
    conocido para que cualquier cambio de formato, no solo una inconsistencia
    interna, haga fallar la suite.
    """
    assert cache_key(**_BASE_KEY) == "b9272532f87e5f0a"


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
