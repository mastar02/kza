import pytest
from src.nlu.slot_extractor import extract_volume, SLOT_VOLUME, extract_slots


@pytest.mark.parametrize("text,expected", [
    ("subí el volumen al 40", 40),
    ("ponelo al 40%", 40),
    ("volumen 80", 80),
    ("más fuerte", 90),
    ("bajito", 20),
    ("ponelo bien fuerte", 90),
    ("la luz al 50", None),          # 'al N' sin contexto de volumen → no es volumen
])
def test_extract_volume(text, expected):
    assert extract_volume(text) == expected


def test_volume_in_extract_slots():
    slots = extract_slots("subí el volumen al 30")
    assert slots.get(SLOT_VOLUME) == 30
