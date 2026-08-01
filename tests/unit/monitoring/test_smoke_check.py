"""Smoke check de la cadena texto → entidad → payload, sin ejecutar nada.

Nace del 2026-07-30: "prendé la luz" murió porque el LLM metió un slot
`entity` en el service_data y HA rechaza el payload ENTERO ante una clave
desconocida. El fallo era invisible hasta que el usuario dijo "no funcionó".

Estos checks corren en dry-run: validan que un comando canónico resuelve a una
entidad viva y produce un payload que HA aceptaría, sin llamar al servicio.
"""

import pytest

from src.home_assistant.ha_client import HomeAssistantClient
from src.monitoring.smoke_check import (
    SmokeResult,
    check_phrase,
    entity_problem,
    indexed_entity_ids,
    indexed_entity_ids_or_problem,
    invalid_payload_keys,
)


class TestInvalidPayloadKeys:
    def test_clean_payload_has_no_problems(self):
        assert invalid_payload_keys({"brightness_pct": 50}) == []

    def test_empty_payload_is_valid(self):
        assert invalid_payload_keys({}) == []

    def test_reports_the_key_ha_would_reject(self):
        """El caso exacto del incidente."""
        assert invalid_payload_keys({"entity": "light.x"}) == ["entity"]

    def test_reports_every_offender_sorted(self):
        bad = invalid_payload_keys(
            {"entity": "light.x", "brightness_pct": 50, "zzz": 1}
        )
        assert bad == ["entity", "zzz"]


class TestEntityProblem:
    def test_healthy_entity_has_no_problem(self):
        assert entity_problem("light.grupo_living", {"light.grupo_living": "off"}) is None

    def test_missing_entity_is_reported(self):
        problem = entity_problem("light.fantasma", {"light.grupo_living": "off"})
        assert problem is not None
        assert "no existe" in problem

    def test_unavailable_entity_is_reported(self):
        """Una entidad unavailable acepta la llamada y no hace nada."""
        problem = entity_problem("light.grupo_bano", {"light.grupo_bano": "unavailable"})
        assert problem is not None
        assert "unavailable" in problem

    @pytest.mark.parametrize(
        "state", ["unknown", "on", "off", "idle", "2026-07-31T12:00:00"]
    )
    def test_unknown_is_not_a_problem(self, state):
        """`unknown` no bloquea nada: HA ejecuta la llamada igual.

        Las escenas viven en `unknown` hasta la primera activación
        (`Scene.state` = timestamp de la última, o `None`), y están indexadas
        en Chroma. Marcarlas como problema haría que el smoke test reporte
        rojo por entidades perfectamente accionables.
        """
        assert entity_problem("scene.fria", {"scene.fria": state}) is None

    @pytest.mark.parametrize(
        "state",
        ["unavailable", "unknown", "on", "off", "idle", "2026-07-31T12:00:00"],
    )
    def test_agrees_with_the_dispatcher_precheck(self, state):
        """El veredicto del smoke test y el del dispatcher deben coincidir.

        Son los dos únicos lugares que deciden "esta entidad está muerta", y
        divergieron: el precheck contaba `unknown` como muerta y el smoke test
        no, así que el diagnóstico decía "las escenas están sanas" mientras el
        dispatcher las rechazaba. Este test ata los dos criterios: si alguien
        cambia uno solo, esto falla.
        """
        ha = HomeAssistantClient.__new__(HomeAssistantClient)
        ha._state_cache = {"scene.fria": {"state": state}}

        smoke_says_dead = entity_problem("scene.fria", {"scene.fria": state}) is not None
        dispatcher_says_dead = ha.is_entity_available("scene.fria") is False

        assert smoke_says_dead == dispatcher_says_dead


class TestCheckPhrase:
    HA = {"light.grupo_living": "off", "light.grupo_bano": "unavailable"}

    def _resolved(self, **over):
        base = {
            "entity_id": "light.grupo_living",
            "domain": "light",
            "service": "turn_on",
            "data": {},
            "similarity": 0.82,
        }
        base.update(over)
        return base

    def test_happy_path(self):
        r = check_phrase("prendé la luz del living", self._resolved(), self.HA)
        assert r.ok is True
        assert r.stage == "ok"
        assert r.entity_id == "light.grupo_living"

    def test_unresolved_phrase_fails_at_vector_search(self):
        """Chroma vacío o stale: pasó el 27/7 cuando el --wipe dejó solo escenas."""
        r = check_phrase("prendé la luz del living", None, self.HA)
        assert r.ok is False
        assert r.stage == "vector_search"

    def test_dead_entity_fails_at_entity_stage(self):
        r = check_phrase(
            "prendé la luz del baño",
            self._resolved(entity_id="light.grupo_bano"),
            self.HA,
        )
        assert r.ok is False
        assert r.stage == "entity"
        assert "unavailable" in r.detail

    def test_bad_payload_fails_at_payload_stage(self):
        """El bug de hoy: habría sido detectado antes que por vos."""
        r = check_phrase(
            "prendé la luz del living",
            self._resolved(data={"entity": "light.grupo_living"}),
            self.HA,
        )
        assert r.ok is False
        assert r.stage == "payload"
        assert "entity" in r.detail

    def test_entity_checked_before_payload(self):
        """Con dos problemas, reporta el más temprano: es el que hay que arreglar."""
        r = check_phrase(
            "prendé la luz del baño",
            self._resolved(entity_id="light.grupo_bano", data={"entity": "x"}),
            self.HA,
        )
        assert r.stage == "entity"

    def test_result_carries_payload_for_reporting(self):
        r = check_phrase(
            "poné la luz del living al 50",
            self._resolved(data={"brightness_pct": 50}),
            self.HA,
        )
        assert r.ok is True
        assert r.service_data == {"brightness_pct": 50}


class TestIndexedEntityIds:
    """La cobertura del smoke test viene de lo que el índice expone, no de
    una lista escrita a mano (2026-07-31): cuarto/balcón/escalera no eran
    default_light de ninguna room y quedaban invisibles pese a resolver por
    voz con similitud 0.92-1.00."""

    def test_indexed_entities_are_derived_from_chroma_not_a_list(self):
        """Coverage must come from what is addressable, not from a hardcoded list."""
        fake_collection = type("C", (), {
            "get": staticmethod(lambda **kw: {"metadatas": [
                {"entity_id": "light.grupo_living"},
                {"entity_id": "light.grupo_cuarto"},
                {"entity_id": "light.grupo_living"},  # duplicado
            ]})
        })()
        assert indexed_entity_ids(fake_collection) == [
            "light.grupo_cuarto", "light.grupo_living",
        ]

    def test_empty_collection_returns_empty_list(self):
        fake_collection = type("C", (), {
            "get": staticmethod(lambda **kw: {"metadatas": []})
        })()
        assert indexed_entity_ids(fake_collection) == []

    def test_metadata_without_entity_id_is_skipped(self):
        """Metadata de rutinas/docs sin entity_id no debe romper ni colarse."""
        fake_collection = type("C", (), {
            "get": staticmethod(lambda **kw: {"metadatas": [
                {"entity_id": "light.grupo_living"},
                {"description": "algo sin entity_id"},
                None,
            ]})
        })()
        assert indexed_entity_ids(fake_collection) == ["light.grupo_living"]


class TestIndexedEntityIdsOrProblem:
    """Review 2026-08-01: el try/except que envolvía `indexed_entity_ids` en
    `tools/smoke_test.py` tenía cobertura cero — mutación probó que borrar el
    `fallos += 1` de ese except sobrevivía la suite entera (80/80 verde). Se
    movió la lógica acá, a una función pura, para que un smoke test roto de
    ESTA forma (Chroma explota al leer el índice) quede detectado por un
    test en vez de depender de que el CLI nunca se toque mal."""

    def test_healthy_collection_returns_ids_and_no_problem(self):
        fake_collection = type("C", (), {
            "get": staticmethod(lambda **kw: {"metadatas": [
                {"entity_id": "light.grupo_living"},
            ]})
        })()
        ids, problem = indexed_entity_ids_or_problem(fake_collection)
        assert ids == ["light.grupo_living"]
        assert problem is None

    def test_broken_collection_reports_a_problem_instead_of_raising(self):
        """El caso que el except protegía: Chroma caído no debe tirar traceback
        ni, peor, quedar sin contarse como fallo. Mata la mutación de borrar
        el `fallos += 1`: si `problem` volviera `None` acá, el caller nunca
        se enteraría de que el índice está roto."""

        class ColeccionRota:
            def get(self, **kw):
                raise RuntimeError("chroma no disponible")

        ids, problem = indexed_entity_ids_or_problem(ColeccionRota())
        assert ids == []
        assert problem is not None
        assert "chroma no disponible" in problem
