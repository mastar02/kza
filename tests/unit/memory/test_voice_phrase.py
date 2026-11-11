"""U3b-4b-2: the verification phrase word list, its normaliser and single-use window matching (R7)."""

import ast
from pathlib import Path
import random
import re
import unicodedata
from types import SimpleNamespace

import pytest
import yaml

from src.memory.voice_attribution import VoiceVerificationWindows
from src.memory.voice_phrase import (
    PHRASE_LENGTH, PHRASE_WORDS, VERIFICATION_REPLY, draw_phrase, edit_distance, near_phrase, normalize_utterance,
    phonetic_key, phrase_overlap, word_near,
)
from src.nlu.home_command import parse_home_command
from src.nlu.command_grammar import parse_default_light_target, parse_partial_command
from src.nlu.regex import vocab
from src.nlu.regex.vocab import CARDINALS

ROOT = Path(__file__).resolve().parents[3]
MEMORY_REGEX = r"(?:record[aá]|acordate|acu[eé]rdate|guard[aá] en (?:la )?memoria)\s+que\s+(.{1,2000})"


def _plain(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", value.casefold()) if not unicodedata.combining(char))


def _excluded_vocabulary() -> set[str]:
    """Every word the command lane, rooms and wake know: NLU and router literals, configured rooms, numbers."""
    words = set()
    sources = [*sorted((ROOT / "src/nlu").rglob("*.py")), ROOT / "src/pipeline/request_router.py",
               ROOT / "src/home_assistant/voice_resolver.py", ROOT / "src/ambient/textual_wake.py"]
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                words |= set(re.findall(r"[a-z]{3,}", _plain(node.value)))
    config = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))

    def strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from strings(key)
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    for section in ("rooms", "wake_word", "home_assistant", "spotify"):
        for value in strings(config.get(section)):
            words |= set(re.findall(r"[a-z]{3,}", _plain(value)))
    words |= {_plain(word) for word in CARDINALS}
    words |= {"nexa", "alexa", "kza", "recorda", "acordate", "acuerdate", "guarda", "memoria", "recuerdo", "olvida",
              "hora", "minuto", "segundo", "dia", "noche", "tarde", "manana", "semana", "lunes", "martes", "miercoles",
              "jueves", "viernes", "sabado", "domingo", "rojo", "azul", "verde", "amarillo", "blanco", "negro",
              "naranja", "rosa", "violeta", "luz", "lampara", "tele", "television", "radio", "musica", "cancion",
              "puerta", "ventana", "enchufe", "alarma", "sirena", "reloj", "vela", "cuarto", "sala", "cocina", "bano",
              "dormitorio", "living", "comedor", "patio", "jardin", "garage", "oficina", "pasillo", "escalera",
              "terraza", "balcon", "planta", "piso", "medio", "cero", "mil", "cien"}
    return words


def test_the_list_has_256_distinct_plain_lowercase_words():
    assert len(PHRASE_WORDS) == 256 and len(set(PHRASE_WORDS)) == 256
    assert all(word.isascii() and word.isalpha() and word == word.lower() and len(word) >= 3 for word in PHRASE_WORDS)
    assert all(normalize_utterance(word) == word for word in PHRASE_WORDS)


def test_no_word_is_a_command_room_device_number_time_colour_wake_or_memory_word():
    clash = sorted(set(PHRASE_WORDS) & _excluded_vocabulary())
    assert clash == []


_ALPHABET = "abcdefghijklmnopqrstuvwxyzñ"
# Common Spanish ASR confusions: b/v, s/z/soft c, ll/y/l, silent h, qu/k, rr/r, ñ/n, g/j.
_CONFUSIONS = (("b", "v"), ("v", "b"), ("s", "z"), ("z", "s"), ("ce", "se"), ("ci", "si"), ("se", "ce"), ("si", "ci"),
               ("ll", "y"), ("y", "ll"), ("ll", "l"), ("h", ""), ("qu", "k"), ("ca", "ka"), ("rr", "r"), ("r", "rr"),
               ("n", "ñ"), ("nz", "ñ"), ("g", "j"), ("j", "g"))
_COMMAND_FORMS = (
    "pon pone poner pongo ponga apaga apago apagar apague prende prendo prender prenda enciende enciendo encender "
    "activa activar desactiva desactivar abre abro abrir cierra cierro cerrar sube subo subir baja bajo bajar para "
    "paro parar deten detener frena cambia cambiar ajusta ajustar configura deja dejar llama llamar toca tocar "
    "reproduce reproducir pausa pausar segui seguir luz luces lampara lamparas foco focos aire clima climatizador "
    "termostato calefaccion temperatura persiana persianas cortina cortinas estor escena tele television radio musica "
    "volumen ventilador enchufe puerta ventana alarma timbre sirena reloj hora minuto segundo dia noche tarde manana "
    "semana hoy ayer grados kelvin calido frio tibio neutro hola chau gracias espera cancion tema siguiente anterior"
).split()


def _misheard(word: str) -> set[str]:
    """The word under one common confusion (or none), then under at most one more edit."""
    confused = {word, "h" + word, word + "s", word.removesuffix("s")}
    for before, after in _CONFUSIONS:
        start = 0
        while (index := word.find(before, start)) != -1:
            confused.add(word[:index] + after + word[index + len(before):])
            start = index + 1
    variants = set()
    for value in confused:
        variants.add(value)
        for index in range(len(value) + 1):
            variants |= {value[:index] + char + value[index:] for char in _ALPHABET}
        for index in range(len(value)):
            variants.add(value[:index] + value[index + 1:])
            variants |= {value[:index] + char + value[index + 1:] for char in _ALPHABET}
    return {_plain(value) for value in variants}


def _command_and_device_vocabulary() -> set[str]:
    words = set(_excluded_vocabulary()) | set(_COMMAND_FORMS)
    for verbs in vocab.IMPERATIVES_VOSEO.values():
        words |= {_plain(verb) for verb in verbs}
    for entity in vocab.ALL_ENTITIES:
        words |= {_plain(word) for alias in (entity.canonical, *entity.aliases) for word in alias.split()}
    words |= {_plain(word) for word in (*vocab.COLOR_TEMP_WORDS, *vocab.COLOR_RGB_WORDS, *vocab.WAKE_WORDS)}
    return {word for word in words if len(word) >= 3}


def test_no_word_is_one_mishearing_away_from_a_command_or_device_word():
    """Review W: one edit and one common Spanish ASR confusion, against the command and device vocabulary."""
    known = _command_and_device_vocabulary()
    assert {"foco", "prendo", "nueve", "brillo", "planta", "color", "mil"} <= known
    clashes = {word: sorted(_misheard(word) & known) for word in PHRASE_WORDS if _misheard(word) & known}
    assert clashes == {}
    for replaced in ("foca", "nube", "manzana", "olla", "abrigo", "campana", "tetera", "llave"):
        assert replaced not in PHRASE_WORDS
    assert _misheard("foca") & known and _misheard("nube") & known, "the check would have caught the old words"


def test_word_nearness_is_sameness_one_edit_or_sound():
    assert edit_distance("foca", "foco") == 1 and edit_distance("pino", "prendo") == 3
    assert edit_distance("ajedrez", "tiburon", 2) == 3, "bounded"
    assert word_near("foco", "foca") and word_near("tiburon", "tiburon")
    assert word_near("baye", "valle") and word_near("serbesa", "cerveza") and word_near("kilo", "quilo")
    assert phonetic_key("hacha") == phonetic_key("acha") and phonetic_key("lluvia") == phonetic_key("yubia")
    assert not word_near("pino", "prendo") and not word_near("sol", "sal"), "short words need the same sound"
    assert phrase_overlap(["foco", "foco", "x"], ["foca", "foca"]) == 2
    assert phrase_overlap(["foco"], ["foca", "foca"]) == 1, "a heard word counts once"
    phrase = "foca pino ajedrez tiburon"
    assert near_phrase("prendo foco ajedrez tiburon", [phrase]), "foca->foco, pino->prendo: three of four"
    assert near_phrase("prendo el foco del ajedrez tiburon", [phrase])
    assert not near_phrase("prendo foco ajedrez", [phrase]), "two of four"
    assert not near_phrase("prendo el foco", [phrase]) and not near_phrase("", [phrase])
    assert not near_phrase("prendo foco ajedrez tiburon", []), "no open phrase, no near miss"
    assert not near_phrase("uno dos tres cuatro cinco seis foco ajedrez tiburon", [phrase]), "more than eight words"


@pytest.mark.parametrize("seed", range(4))
def test_no_word_or_sampled_phrase_is_a_home_command_or_a_memory_request(seed):
    chooser = random.Random(seed)
    phrases = [*PHRASE_WORDS, *(draw_phrase(lambda words: chooser.choice(words)) for _ in range(500))]
    for phrase in phrases:
        for spoken in (phrase, "nexa " + phrase, phrase.capitalize() + "."):
            assert parse_home_command(spoken, wake_words=("nexa",)) is None, spoken
            partial = parse_partial_command(spoken)
            assert partial.intent is None and partial.domain is None, spoken
            assert parse_default_light_target(spoken) is None, spoken
            assert re.fullmatch(MEMORY_REGEX, spoken, re.IGNORECASE) is None, spoken


def test_a_phrase_is_four_words_drawn_with_the_given_choice():
    picks = iter(["gato", "perro", "barco", "nube"])
    assert draw_phrase(lambda words: next(picks)) == "gato perro barco nube"
    assert PHRASE_LENGTH == 4 and len(draw_phrase().split()) == 4
    assert all(word in PHRASE_WORDS for word in draw_phrase().split())


def test_normalisation_drops_the_wake_word_case_accents_and_punctuation():
    assert normalize_utterance("Nexa, Gató perro... ¡Barco NUBE!", ("nexa",)) == "gato perro barco nube"
    assert normalize_utterance("  gato   perro\tbarco nube ") == "gato perro barco nube"
    assert normalize_utterance("Pingüino, canción y CAÑA") == "pinguino cancion y cana", "mid-word marks never split a word"
    assert normalize_utterance("nexa") == "" and normalize_utterance(None) == ""
    assert normalize_utterance("nexagato perro barco nube", ("nexa",)) == "nexagato perro barco nube"
    assert VERIFICATION_REPLY == "Recibí la frase. El resultado aparece en Mis recuerdos."


class Clock:
    value = 1000.0

    def __call__(self):
        return self.value


def binding(person, profile="profile"):
    return SimpleNamespace(person_id=person, profile_id=profile)


def test_windows_match_by_room_and_phrase_once():
    clock = Clock()
    phrases = iter(["gato perro barco nube", "pino roca isla lago", "gato perro barco nube", "taza olla plato jarra"])
    windows = VoiceVerificationWindows(clock, draw=lambda: next(phrases))
    first = windows.open(binding("a"), "living")
    second = windows.open(binding("b"), "living")
    third = windows.open(binding("c"), "kitchen")  # the same phrase is allowed in another room
    assert (first.phrase, second.phrase, third.phrase) == ("gato perro barco nube", "pino roca isla lago", "gato perro barco nube")
    assert windows.match("kitchen", "pino roca isla lago") is None, "another room's phrase"
    assert windows.match("living", "pino roca isla lago") is second
    assert windows.match("living", "pino roca isla lago") is None, "single use"
    assert windows.current("a") is first and windows.current("c") is third
    assert windows.match("living", "gato perro barco") is None and windows.match("", "gato perro barco nube") is None
    clock.value += 60.0
    assert windows.match("living", "gato perro barco nube") is None, "expired at exactly 60 s"
    assert windows.current("a") is None


def test_a_phrase_already_open_in_the_room_is_redrawn():
    clock = Clock()
    phrases = iter(["gato perro barco nube", "gato perro barco nube", "pino roca isla lago"])
    windows = VoiceVerificationWindows(clock, draw=lambda: next(phrases))
    windows.open(binding("a"), "living")
    assert windows.open(binding("b"), "living").phrase == "pino roca isla lago"
    stuck = VoiceVerificationWindows(clock, draw=lambda: "gato perro barco nube")
    stuck.open(binding("a"), "living")
    with pytest.raises(RuntimeError, match="memory_voice_phrase_unavailable"):
        stuck.open(binding("b"), "living")
    assert stuck.current("b") is None


def test_a_phrase_with_a_word_near_a_home_name_is_redrawn_and_only_the_count_is_logged(caplog):
    caplog.set_level("DEBUG")
    clock = Clock()
    phrases = iter(["tiburon ajedrez trineo vitral", "delfin ajedrez trineo vitral", "mosaico ajedrez trineo vitral"])
    windows = VoiceVerificationWindows(clock, draw=lambda: next(phrases))
    windows.set_reserved(lambda: ("Lámpara Tiburón", "living", "Delfín del patio"))
    assert windows.open(binding("a"), "living").phrase == "mosaico ajedrez trineo vitral"
    assert [record.getMessage() for record in caplog.records] == ["[VoiceWindows] phrase redrawn: home-name collisions=2"]
    near = VoiceVerificationWindows(clock, draw=lambda: "tiburona ajedrez trineo vitral")
    near.set_reserved(lambda: ("tiburon",))
    with pytest.raises(RuntimeError, match="memory_voice_phrase_unavailable"):
        near.open(binding("a"), "living")


def test_unavailable_home_names_fall_back_to_the_static_list():
    clock = Clock()

    def broken():
        raise ConnectionError("ha down")

    windows = VoiceVerificationWindows(clock, draw=lambda: "tiburon ajedrez trineo vitral")
    windows.set_reserved(broken)
    assert windows.open(binding("a"), "living").phrase == "tiburon ajedrez trineo vitral"


def test_live_phrases_list_every_room_without_consuming():
    clock = Clock()
    phrases = iter(["gato perro barco nube", "pino roca isla lago"])
    windows = VoiceVerificationWindows(clock, draw=lambda: next(phrases))
    assert windows.live_phrases() == ()
    first = windows.open(binding("a"), "living")
    windows.open(binding("b"), "kitchen")
    assert sorted(windows.live_phrases()) == ["gato perro barco nube", "pino roca isla lago"]
    assert windows.current("a") is first, "listing consumes nothing"
    clock.value += 60.0
    assert windows.live_phrases() == ()
