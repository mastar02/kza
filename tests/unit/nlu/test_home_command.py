"""Full-phrase extended intents preserve parameters and decline ambiguity."""

import pytest

from src.nlu.home_command import parse_home_command


@pytest.mark.parametrize("text,domain,target,operation,parameters", [
    ("Nexa, poné el aire del salón a 22,5 grados Celsius", "climate", "del salon", "set_temperature", {"temperature": 22.5}),
    ("¿Podés poner el termostato a veintidós coma cinco grados?", "climate", "", "set_temperature", {"temperature": 22.5}),
    ("Por favor ajustá la temperatura a treinta y cinco °C", "climate", "", "set_temperature", {"temperature": 35}),
    ("Poné el aire en modo frío", "climate", "", "set_hvac_mode", {"hvac_mode": "cool"}),
    ("Dejá el aire del salón en calor", "climate", "del salon", "set_hvac_mode", {"hvac_mode": "heat"}),
    ("Poné el aire en seco", "climate", "", "set_hvac_mode", {"hvac_mode": "dry"}),
    ("Poné el aire en ventilación", "climate", "", "set_hvac_mode", {"hvac_mode": "fan_only"}),
    ("Poné el aire en automático", "climate", "", "set_hvac_mode", {"hvac_mode": "auto"}),
    ("Apagá el aire", "climate", "", "set_hvac_mode", {"hvac_mode": "off"}),
    ("Abrí la cortina", "cover", "", "open_cover", {}),
    ("Cerrá las persianas del salón", "cover", "del salon", "close_cover", {}),
    ("Detené la persiana", "cover", "", "stop_cover", {}),
    ("Bajá la persiana del salón al treinta por ciento", "cover", "del salon", "set_cover_position", {"position": 30}),
    ("Poné la persiana al 0%", "cover", "", "set_cover_position", {"position": 0}),
    ("Poné la luz del salón cálida al treinta por ciento", "light", "del salon", "color_temp_kelvin", {"color_temp_kelvin": 2700, "brightness_pct": 30}),
    ("Encendé la luz en 4000 kelvin", "light", "", "color_temp_kelvin", {"color_temp_kelvin": 4000}),
    ("Poné la luz fría por favor", "light", "", "color_temp_kelvin", {"color_temp_kelvin": 6500}),
])
def test_complete_extended_imperatives(text, domain, target, operation, parameters):
    command = parse_home_command(text, wake_words=("nexa",))
    assert (command.domain, command.target, command.operation, command.parameters, command.reason) == (domain, target, operation, parameters, None)


@pytest.mark.parametrize("text", [
    "Prendé el aire", "Poné el aire a dos tres grados", "Poné el aire a cien cinco grados",
    "Poné el aire a 70°F", "Poné el aire a -20 grados", "Poné el aire a 2e1 grados",
    "Poné el aire a veintidós coma cinco seis grados", "Poné el aire a 22 grados y apagá la luz",
    "Poné la persiana al -10%", "Bajá la persiana al 101%", "Poné la persiana al 30,5%",
    "Poné la luz cálida al -30%", "Poné la luz cálida al dos tres por ciento",
    "Poné la luz cálida al 0%", "Poné la luz a 2700 kelvin y apagá el aire",
])
def test_incomplete_or_malformed_claim_never_supplies_guessed_parameters(text):
    command = parse_home_command(text)
    # The shared semantic blocker may reject before this parser claims it.
    assert command is None or command.reason == "incomplete_home_command"


@pytest.mark.parametrize("text", [
    "Prendé la luz", "Bajá la luz al 30%", "Qué temperatura hace", "La luz es cálida",
    "No pongas el aire a 22 grados", "¿Apagaste el aire?", "Ayer puse el aire a 22 grados",
    "Poné el aire a 22 grados mañana", "Si hace calor poné el aire a 22 grados",
    "Me dijo que apague el aire", "¿Está abierta la persiana?",
    "Dejé el aire a 22 grados", "Configuré el aire a 22 grados", "Ajusté el aire a 22 grados",
    "Activé el aire", "Desactivé el aire", "Paré la persiana",
])
def test_other_paths_or_non_immediate_speech_do_not_produce_a_typed_effect(text):
    assert parse_home_command(text) is None
