"""Los slots que van a Home Assistant deben ser claves válidas de service_data.

Incidente 2026-07-30: "prendé la luz" en el escritorio falló con

    {'code': 'invalid_format', 'message': "extra keys not allowed @ data['entity']"}

HA rechaza el service_data ENTERO si trae una clave que no conoce, así que un
solo slot espurio anula el comando completo. El slot venía del LLM router: a
diferencia del extractor por regex —que solo puede emitir las 5 claves
canónicas— el LLM devuelve JSON libre, y `merge_service_data` hacía
`merged.update(query_slots)` sin validar nada.
"""

import pytest

from src.nlu.slot_extractor import merge_service_data


class TestQuerySlotsWhitelist:
    def test_unknown_slot_is_dropped(self):
        """El caso exacto del incidente: 'entity' no es service_data de HA."""
        merged = merge_service_data({}, {"entity": "light.grupo_escritorio"})
        assert "entity" not in merged

    def test_unknown_slot_does_not_take_down_valid_ones(self):
        """Un slot espurio no puede costar el comando entero."""
        merged = merge_service_data(
            {}, {"entity": "light.grupo_escritorio", "brightness_pct": 40}
        )
        assert merged == {"brightness_pct": 40}

    @pytest.mark.parametrize(
        "slot,value",
        [
            ("brightness_pct", 73),
            ("rgb_color", [255, 0, 0]),
            ("color_temp_kelvin", 2700),
            ("effect", "colorloop"),
            ("volume_pct", 30),
        ],
    )
    def test_valid_slots_survive(self, slot, value):
        assert merge_service_data({}, {slot: value}) == {slot: value}

    def test_preset_from_chroma_is_preserved(self):
        """El preset viene de nuestro propio sync: no se filtra."""
        merged = merge_service_data({"brightness_pct": 50}, {})
        assert merged == {"brightness_pct": 50}

    def test_user_slot_still_wins_over_preset(self):
        """El filtro no puede romper la precedencia ya existente."""
        merged = merge_service_data({"brightness_pct": 50}, {"brightness_pct": 73})
        assert merged == {"brightness_pct": 73}

    def test_color_exclusivity_still_holds(self):
        """rgb_color y color_temp_kelvin siguen siendo mutuamente excluyentes."""
        merged = merge_service_data(
            {"color_temp_kelvin": 2700}, {"rgb_color": [0, 255, 0]}
        )
        assert merged == {"rgb_color": [0, 255, 0]}
