"""Tests: merge_service_data service-aware (review 2026-08-09).

HA rechaza el service_data ENTERO ante una clave fuera de schema — no la
ignora — así que un slot espurio anula el comando completo. El merge filtra
los slots del usuario contra el schema del service de destino.
"""
from src.nlu.slot_extractor import merge_service_data


class TestServiceAwareMerge:
    def test_turn_on_descarta_volume(self):
        # "poné las luces fuerte": 'fuerte' vive en BRIGHTNESS_WORDS y en
        # VOLUME_WORDS → el extractor emite los dos; volume_pct en el data de
        # light.turn_on mataba el comando entero.
        out = merge_service_data(
            {}, {"brightness_pct": 90, "volume_pct": 90}, service="turn_on"
        )
        assert out == {"brightness_pct": 90}

    def test_turn_off_no_admite_slots(self):
        # "apagá la luz, está muy fuerte": brightness residual en turn_off
        # hacía que HA rechace el apagado (schema: transition/flash).
        out = merge_service_data(
            {}, {"brightness_pct": 90}, service="turn_off"
        )
        assert out == {}

    def test_sin_service_conserva_comportamiento_legacy(self):
        out = merge_service_data({}, {"volume_pct": 30})
        assert out == {"volume_pct": 30}

    def test_color_ambiguo_gana_rgb_dentro_de_los_slots_del_usuario(self):
        # 'amarilla' vive en COLOR_MAP y TEMP_WORDS_K → el extractor emite
        # rgb_color Y color_temp_kelvin; HA los trata como excluyentes y
        # rechazaba el comando entero. El color explícito gana.
        out = merge_service_data(
            {},
            {"rgb_color": [255, 255, 0], "color_temp_kelvin": 2700},
            service="turn_on",
        )
        assert out == {"rgb_color": [255, 255, 0]}

    def test_color_del_usuario_pisa_temp_del_preset(self):
        out = merge_service_data(
            {"color_temp_kelvin": 3000}, {"rgb_color": [255, 0, 0]},
            service="turn_on",
        )
        assert out == {"rgb_color": [255, 0, 0]}

    def test_temp_del_usuario_pisa_color_del_preset(self):
        out = merge_service_data(
            {"rgb_color": [1, 2, 3]}, {"color_temp_kelvin": 2700},
            service="turn_on",
        )
        assert out == {"color_temp_kelvin": 2700}
