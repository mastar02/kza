"""Smoke check de la cadena texto → entidad → payload, sin ejecutar nada.

Nace del 2026-07-30: "prendé la luz" murió porque el LLM metió un slot
`entity` en el service_data y HA rechaza el payload ENTERO ante una clave
desconocida. El fallo era invisible hasta que el usuario dijo "no funcionó".

Estos checks corren en dry-run: validan que un comando canónico resuelve a una
entidad viva y produce un payload que HA aceptaría, sin llamar al servicio.
"""

import pytest

from src.monitoring.smoke_check import (
    SmokeResult,
    check_phrase,
    entity_problem,
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
