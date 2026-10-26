"""Exclusión de entidades en QUERY-time del vector search (2026-06-04).

Bug real: el sync del 31-05 indexó 76 docs genéricos de light.hogar
("Prendé la iluminación.") contradiciendo el diseño del runtime (default
exclude + global-scope por keywords hardcoded en el dispatcher). Resultado:
"Nexa, prende la luz." desde el escritorio prendió TODA la casa.
search_command confiaba en que los excluidos no estuvieran indexados —
ahora los filtra también al consultar (defensa contra índices contaminados).
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("chromadb")


def _fake_query_result(rows):
    return {
        "ids": [[f"id_{i}" for i in range(len(rows))]],
        "distances": [[d for d, _, _ in rows]],
        "metadatas": [[m for _, m, _ in rows]],
        "documents": [[doc for _, _, doc in rows]],
    }


def _make_chroma(rows, excluded=None):
    from src.vectordb.chroma_sync import ChromaSync

    cs = ChromaSync.__new__(ChromaSync)
    # `commands` es @property sin setter → setear el atributo interno real
    cs._commands_collection = MagicMock()
    cs._commands_collection.query = MagicMock(return_value=_fake_query_result(rows))
    cs._embedder = MagicMock()
    cs._embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.0] * 8))
    cs._excluded_entities = set(excluded if excluded is not None else ["light.hogar"])
    cs._excluded_patterns = []
    return cs


def _meta(entity_id, area="", service="turn_on"):
    return {
        "entity_id": entity_id, "domain": "light", "service": service,
        "area": area, "friendly_name": entity_id,
    }


# El caso real de prod: light.hogar (excluido) gana por similitud con doc
# genérico; el grupo del escritorio está apenas abajo y sobre threshold.
# El candidato no excluido pertenece a otra habitación. Sin área explícita
# permite comprobar el filtro; con área solicitada también debe rechazarse.
_PROD_ROWS = [
    (0.10, _meta("light.hogar"), "Prendé la iluminación."),
    (0.46, _meta("light.grupo_cocina", area="Cocina"), "Prendé la luz de la cocina."),
]


class TestExcludedEntitiesQueryTime:
    def test_excluded_global_and_other_room_cannot_replace_requested_area(self):
        cs = _make_chroma(_PROD_ROWS)
        r = cs.search_command("prende la luz", threshold=0.65, prefer_area="Escritorio")
        assert r is None
        assert cs._commands_collection.query.call_count == 1
        assert cs._commands_collection.query.call_args.kwargs["where"] == {"area": "Escritorio"}

    def test_excluded_top_candidate_skipped_without_prefer_area(self):
        cs = _make_chroma(_PROD_ROWS)
        r = cs.search_command("prende la luz", threshold=0.65)
        assert r is not None
        assert r["entity_id"] == "light.grupo_cocina"

    def test_all_candidates_excluded_returns_none(self):
        rows = [(0.10, _meta("light.hogar"), "Prendé la iluminación.")]
        cs = _make_chroma(rows)
        assert cs.search_command("prende la luz", threshold=0.65) is None

    def test_no_exclusions_keeps_top(self):
        cs = _make_chroma(_PROD_ROWS, excluded=[])
        r = cs.search_command("prende la luz", threshold=0.65)
        assert r["entity_id"] == "light.hogar"
