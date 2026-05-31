import pytest
from src.nlu.slot_extractor import extract_volume, SLOT_VOLUME, extract_slots


@pytest.mark.parametrize("text,expected", [
    ("subí el volumen al 40", 40),
    ("ponelo al 40%", 40),
    ("volumen 80", 80),
    ("ponelo más fuerte", 90),       # antes "más fuerte" (sin contexto) → ahora con verbo
    ("ponelo bajito", 20),           # antes "bajito" (sin contexto) → ahora con verbo
    ("ponelo bien fuerte", 90),
    ("la luz al 50", None),          # brillo, sin contexto de volumen → None
])
def test_extract_volume(text, expected):
    assert extract_volume(text) == expected


def test_volume_in_extract_slots():
    slots = extract_slots("subí el volumen al 30")
    assert slots.get(SLOT_VOLUME) == 30


@pytest.mark.parametrize("text", [
    "bajito la luz",
    "más alto el brillo",
    "poneme más alta la luz",
    "bajá la luz",
])
def test_no_false_positive_on_brightness(text):
    assert extract_volume(text) is None
