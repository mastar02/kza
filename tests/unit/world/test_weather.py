from src.world.weather import describe_current


def _state(condition="partlycloudy", temp=12.7, **attrs):
    return {
        "entity_id": "weather.forecast_home",
        "state": condition,
        "attributes": {"temperature": temp, "temperature_unit": "°C", **attrs},
    }


def test_describes_temperature_and_condition_in_spanish():
    # temp=12.7 rounds to 13 for speech (see test_rounds_temperature_for_speech).
    out = describe_current(_state())
    assert "13" in out
    assert "grados" in out.lower()
    assert "nublado" in out.lower()


def test_rounds_temperature_for_speech():
    # 12.7 spoken as "13 grados": decimals are noise out loud.
    assert "13 grados" in describe_current(_state(temp=12.7))
    assert "12 grados" in describe_current(_state(temp=12.4))


def test_unknown_condition_falls_back_without_inventing():
    out = describe_current(_state(condition="hurricane_of_frogs"))
    assert "13 grados" in out
    assert "rana" not in out.lower()


def test_unavailable_entity_says_so():
    out = describe_current({"state": "unavailable", "attributes": {}})
    assert "no" in out.lower()
    assert "grados" not in out.lower()


def test_missing_entity_says_so():
    # HA restarted, entity not synced yet: never invent a number.
    out = describe_current(None)
    assert "no" in out.lower()
    assert "grados" not in out.lower()


def test_missing_temperature_attribute_does_not_crash():
    out = describe_current({"state": "cloudy", "attributes": {}})
    assert "nublado" in out.lower()
    assert "None" not in out


def test_includes_humidity_when_present():
    out = describe_current(_state(humidity=81))
    assert "81" in out
