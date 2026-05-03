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
    cs.commands = MagicMock()
    cs.commands.query = MagicMock(return_value=_fake_query_result(rows))
    cs.embedder = MagicMock()
    cs.embedder.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.0] * 8))
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
        # Verify the where clause includes service_filter
        call_kwargs = cs.commands.query.call_args.kwargs
        assert call_kwargs.get("where") == {"service": "turn_off"}, (
            f"service_filter must still propagate as where clause. "
            f"Got where={call_kwargs.get('where')}"
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
