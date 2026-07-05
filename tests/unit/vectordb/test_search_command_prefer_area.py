"""Tests for search_command prefer_area kwarg.

Reproduce el bug de 2026-05-03 01:25 / 08:08: con texto fantasma
"Nexa bajá la luz al cincuenta por ciento" (sin mencionar room) el
vector search elegía light.cuarto porque tenía un doc contaminado
"Prendé las luces, tú." sin contexto de room. Tras este fix, cuando
el caller pasa prefer_area derivado del mic, los candidatos cuyo
metadata.area matchea reciben un bonus que re-puntúa la elección
sin descartar ningún candidato.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

# Skip the whole module if chromadb isn't installed locally — these tests
# exercise ChromaSync.search_command directly which imports chromadb at
# module top. Server has it; dev laptop may not.
pytest.importorskip("chromadb")


def _fake_query_result(rows: list[tuple[float, dict, str]]) -> dict:
    """Build the dict that ChromaDB.query returns from (distance, meta, doc)."""
    return {
        "ids": [[f"id_{i}" for i in range(len(rows))]],
        "distances": [[d for d, _, _ in rows]],
        "metadatas": [[m for _, m, _ in rows]],
        "documents": [[doc for _, _, doc in rows]],
    }


def _make_chroma_with_results(rows):
    """Build a ChromaSync-compatible mock; rows is list of (distance, meta, doc)."""
    from src.vectordb.chroma_sync import ChromaSync

    cs = ChromaSync.__new__(ChromaSync)
    # `commands` es @property sin setter → setear el atributo interno real
    cs._commands_collection = MagicMock()
    cs._commands_collection.query = MagicMock(return_value=_fake_query_result(rows))
    cs._embedder = MagicMock()
    cs._embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.0] * 8))
    # search_command filtra excluidos en query-time (2026-06-04) — el mock
    # bypasea __init__, así que hay que poblar el estado de exclusiones.
    cs._excluded_entities = {"light.hogar"}
    cs._excluded_patterns = []
    return cs


class TestPreferAreaSignature:
    def test_search_command_accepts_prefer_area(self):
        """ChromaSync.search_command must accept prefer_area kwarg."""
        import inspect
        from src.vectordb.chroma_sync import ChromaSync
        sig = inspect.signature(ChromaSync.search_command)
        assert "prefer_area" in sig.parameters, (
            f"prefer_area kwarg not in search_command. "
            f"Found: {list(sig.parameters.keys())}"
        )
        assert sig.parameters["prefer_area"].default is None


class TestPreferAreaScoring:
    """Reproduce real scenario from log 2026-05-03."""

    def _phantom_rows(self):
        """Rows simulando la respuesta real de Chroma para el texto fantasma.

        light.cuarto gana por margen pequeño porque su doc contaminado
        ('Prendé las luces, tú.' sin room) es más cercano semánticamente a
        'bajá la luz al cincuenta por ciento' que el doc específico de
        light.escritorio ('Prendé la luz del escritorio.').
        """
        return [
            # distance, metadata, document
            (0.30, {
                "entity_id": "light.cuarto",
                "domain": "light",
                "service": "turn_on",
                "friendly_name": "Cuarto",
                "area": "Cuarto",
                "service_data": "{}",
                "capability": "onoff",
                "value_label": "prender",
            }, "Prendé las luces, tú."),
            (0.34, {
                "entity_id": "light.escritorio",
                "domain": "light",
                "service": "turn_on",
                "friendly_name": "Escritorio",
                "area": "Escritorio",
                "service_data": "{}",
                "capability": "onoff",
                "value_label": "prender",
            }, "Prendé la luz del escritorio."),
            (0.40, {
                "entity_id": "light.living",
                "domain": "light",
                "service": "turn_on",
                "friendly_name": "Living",
                "area": "Living",
                "service_data": "{}",
                "capability": "onoff",
                "value_label": "prender",
            }, "Prendé la luz del living."),
        ]

    def test_without_prefer_area_chroma_picks_cuarto_baseline(self):
        """Sin prefer_area, gana light.cuarto (replica el bug actual)."""
        cs = _make_chroma_with_results(self._phantom_rows())
        result = cs.search_command(
            "bajá la luz al cincuenta por ciento", threshold=0.5,
        )
        assert result is not None
        assert result["entity_id"] == "light.cuarto", (
            "Baseline without prefer_area should still match light.cuarto "
            "(this is the bug we are fixing)."
        )

    def test_with_prefer_area_escritorio_re_ranks_to_escritorio(self):
        """Con prefer_area='Escritorio', el bonus re-puntúa y gana escritorio."""
        cs = _make_chroma_with_results(self._phantom_rows())
        result = cs.search_command(
            "bajá la luz al cincuenta por ciento",
            threshold=0.5,
            prefer_area="Escritorio",
        )
        assert result is not None
        assert result["entity_id"] == "light.escritorio", (
            f"prefer_area='Escritorio' should boost light.escritorio over "
            f"light.cuarto. Got: {result['entity_id']}"
        )

    def test_prefer_area_does_not_force_match_below_threshold(self):
        """El bonus no debe sobrepasar al threshold base.

        Si TODOS los candidatos están bajo threshold, el bonus no debe
        rescatar artificialmente uno con area-match. Devolvemos None igual.
        """
        rows = [
            (1.5, {  # similarity ≈ 0.25 — bien debajo de threshold 0.5
                "entity_id": "light.escritorio",
                "domain": "light", "service": "turn_on",
                "friendly_name": "Escritorio", "area": "Escritorio",
                "service_data": "{}", "capability": "onoff",
                "value_label": "prender",
            }, "Prendé la luz del escritorio."),
        ]
        cs = _make_chroma_with_results(rows)
        result = cs.search_command(
            "qué hora es", threshold=0.5, prefer_area="Escritorio",
        )
        assert result is None, (
            "prefer_area boost must not rescue results below the base threshold."
        )

    def test_prefer_area_respects_service_filter(self):
        """prefer_area no debe pisar al service_filter (sigue siendo where clause)."""
        cs = _make_chroma_with_results(self._phantom_rows())
        cs.search_command(
            "bajá la luz", threshold=0.5,
            service_filter="turn_off",
            prefer_area="Escritorio",
        )
        # Verify the where clause includes service_filter — con el two-pass
        # (2026-06-04) la query del pase de área lo lleva dentro del $and.
        call_kwargs = cs._commands_collection.query.call_args.kwargs
        where = call_kwargs.get("where") or {}
        clauses = where.get("$and", [where])
        assert {"service": "turn_off"} in clauses, (
            f"service_filter must still propagate in the where clause. "
            f"Got where={where}"
        )

    def test_prefer_area_unknown_area_falls_back_to_baseline(self):
        """Si prefer_area no matchea ningún candidato, usar ranking original."""
        cs = _make_chroma_with_results(self._phantom_rows())
        result = cs.search_command(
            "bajá la luz al cincuenta por ciento",
            threshold=0.5,
            prefer_area="HabitacionInexistente",
        )
        # Sin candidatos con esa area, gana el top distance original.
        assert result["entity_id"] == "light.cuarto"


class TestContextBoostStrength:
    """Fix 2026-06-04: con boost 0.15, el garble far-field del STT ('prender a
    luz', 'la luz de la vida') matcheaba docs de OTRAS rooms con gap > 0.15 y
    prendía living/balcón desde el escritorio. El boost debe ser contundente:
    si hay candidato del área del mic sobre threshold, gana."""

    def test_garble_gap_over_old_boost_still_picks_mic_area(self):
        # living sim=0.88 vs escritorio sim=0.72 (gap 0.16 > 0.15 viejo)
        rows = [
            (0.24, {"entity_id": "light.grupo_living", "domain": "light",
                    "service": "turn_on", "area": "Living",
                    "friendly_name": "living"}, "Prendé la luz del living."),
            (0.56, {"entity_id": "light.grupo_escritorio", "domain": "light",
                    "service": "turn_on", "area": "Escritorio",
                    "friendly_name": "escritorio"}, "Prendé la luz del escritorio."),
        ]
        cs = _make_chroma_with_results(rows)
        r = cs.search_command("prender a luz", threshold=0.65, prefer_area="Escritorio")
        assert r is not None
        assert r["entity_id"] == "light.grupo_escritorio"

    def test_boost_still_does_not_rescue_below_threshold(self):
        # El gate de threshold sigue sobre la similarity BASE: un doc del área
        # con sim 0.50 no se rescata por boost.
        rows = [
            (1.0, {"entity_id": "light.grupo_escritorio", "domain": "light",
                   "service": "turn_on", "area": "Escritorio",
                   "friendly_name": "escritorio"}, "Prendé la luz del escritorio."),
        ]
        cs = _make_chroma_with_results(rows)
        assert cs.search_command("xyz", threshold=0.65, prefer_area="Escritorio") is None


def _make_chroma_two_pass(area_rows, global_rows):
    """Mock cuyo query devuelve area_rows si el where filtra por area,
    global_rows si no — simula el filtrado real de ChromaDB."""
    from src.vectordb.chroma_sync import ChromaSync

    cs = ChromaSync.__new__(ChromaSync)
    cs._commands_collection = MagicMock()

    def _query(**kwargs):
        where = kwargs.get("where") or {}
        clauses = where.get("$and", [where]) if where else []
        has_area = any("area" in c for c in clauses)
        return _fake_query_result(area_rows if has_area else global_rows)

    cs._commands_collection.query = MagicMock(side_effect=_query)
    cs._embedder = MagicMock()
    cs._embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.0] * 8))
    cs._excluded_entities = {"light.hogar"}
    cs._excluded_patterns = []
    return cs


def _row(dist, entity, area, doc, service="turn_on"):
    return (dist, {"entity_id": entity, "domain": "light", "service": service,
                   "area": area, "friendly_name": entity}, doc)


class TestAreaFirstTwoPassSearch:
    """Fix 2026-06-04 ronda 3: 'prende la luz' desde el escritorio prendía el
    PASILLO — el top-10 global se llena de docs cortos de otras rooms y ningún
    doc del área del mic entra al pool → el boost no tiene a quién boostear
    (verificado con ranking real: top-10 sin Escritorio, pasillo sim=0.945).
    Pase 1: SOLO docs del área preferida; si hay match ≥ threshold, gana.
    Pase 2 (fallback): query global actual con boost."""

    # El caso real: top global lleno de pasillo/living, escritorio fuera.
    _GLOBAL = [
        _row(0.11, "light.grupo_pasillo", "Pasillo", "prende la luz fría"),
        _row(0.23, "light.grupo_balcon", "Balcón", "Prendé la luz del balcón"),
        _row(0.25, "light.grupo_living", "Living", "Prendé la luz con tono cálida"),
    ]
    _AREA = [
        _row(0.40, "light.grupo_escritorio", "Escritorio", "Vos prenderás la luz del escritorio."),
    ]

    def test_generic_query_picks_mic_area_even_outside_global_top(self):
        cs = _make_chroma_two_pass(self._AREA, self._GLOBAL)
        r = cs.search_command("prende la luz", threshold=0.65, prefer_area="Escritorio")
        assert r is not None
        assert r["entity_id"] == "light.grupo_escritorio"

    def test_area_below_threshold_falls_back_to_global(self):
        area_weak = [_row(0.90, "light.grupo_escritorio", "Escritorio", "doc lejano")]  # sim 0.55
        cs = _make_chroma_two_pass(area_weak, self._GLOBAL)
        r = cs.search_command("prende la luz", threshold=0.65, prefer_area="Escritorio")
        assert r is not None
        assert r["entity_id"] == "light.grupo_pasillo"  # mejor global

    def test_area_pass_empty_falls_back_to_global(self):
        cs = _make_chroma_two_pass([], self._GLOBAL)
        r = cs.search_command("prende la luz", threshold=0.65, prefer_area="Escritorio")
        assert r is not None
        assert r["entity_id"] == "light.grupo_pasillo"

    def test_no_prefer_area_single_global_pass(self):
        cs = _make_chroma_two_pass(self._AREA, self._GLOBAL)
        r = cs.search_command("prende la luz", threshold=0.65)
        assert r["entity_id"] == "light.grupo_pasillo"
        # una sola query (sin pase de área)
        assert cs._commands_collection.query.call_count == 1
