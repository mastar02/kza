"""Deterministic extended controls with complete parameter and target evidence."""

from dataclasses import dataclass, field
import math
import re
import unicodedata

from src.nlu.command_grammar import non_immediate_action_reason
from src.nlu.regex.vocab import CARDINALS
from src.nlu.slot_extractor import TEMP_WORDS_K


def normalize_target(value: str) -> str:
    """Normalize aliases for exact comparison, without fuzzy room substitution."""
    value = "".join(char for char in unicodedata.normalize("NFD", value.casefold()) if not unicodedata.combining(char))
    return " ".join(value.split())


@dataclass(frozen=True)
class HomeCommand:
    """An extended control claim; a reason requests clarification without effects."""

    domain: str
    target: str
    operation: str | None = None
    parameters: dict = field(default_factory=dict)
    reason: str | None = None


_AIR = r"(?:aire(?: acondicionado)?|clima|climatizador|termostato|calefaccion|temperatura)"
_COVER = r"(?:persianas?|cortinas?|estores?)"
_LIGHT = r"(?:luces|luz|lamparas?|iluminacion)"
_SET = r"(?:pone|poner|pon|ponga|poneme|ajusta|ajustar|ajuste|deja|dejar|deje|configura|configurar|configure|cambia|cambiar|cambie)"
_OFF = r"(?:apaga|apagar|apague|desactiva|desactivar|desactive)"
_ON = r"(?:prende|prender|prenda|enciende|encende|encender|encienda|activa|activar|active)"
_OPEN = r"(?:abri|abre|abrir|abra|subi|sube|subir|suba|levanta|levantar)"
_CLOSE = r"(?:cerra|cierra|cerrar|cierre|baja|bajar|baje)"
_STOP = r"(?:para|parar|pare|detene|detener|deten|detenga|frena|frenar|frene)"
_ARTICLE = r"(?:(?:el|la|los|las)\s+)?"
_MODES = {"apagado": "off", "apagada": "off", "off": "off", "frio": "cool", "fria": "cool", "refrigeracion": "cool", "calor": "heat", "calefaccion": "heat", "seco": "dry", "deshumidificacion": "dry", "deshumidificar": "dry", "ventilacion": "fan_only", "solo ventilador": "fan_only", "automatico": "auto", "automatica": "auto", "auto": "auto"}
_CARDINAL_VALUES = {normalize_target(word): number for word, number in CARDINALS.items()}


def _number(raw: str) -> float | None:
    raw = raw.strip()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", raw):
        number = float(raw.replace(",", "."))
        return number if math.isfinite(number) else None
    def cardinal(text: str) -> int | None:
        if text in _CARDINAL_VALUES:
            return _CARDINAL_VALUES[text]
        words = text.split()
        if len(words) == 3 and words[1] == "y":
            words.pop(1)
        if len(words) == 2:
            tens, units = (_CARDINAL_VALUES.get(word) for word in words)
            if tens in {30, 40, 50, 60, 70, 80, 90} and units is not None and 1 <= units <= 9:
                return tens + units
        return None

    parts = re.split(r"\s+(?:coma|punto|con)\s+", raw)
    whole = cardinal(parts[0])
    if whole is None or len(parts) > 2:
        return None
    if len(parts) == 1:
        return whole
    fraction = cardinal(parts[1])
    if fraction is None:
        return None
    return whole + fraction / (10 ** len(str(fraction)))


def parse_home_command(text: str, *, wake_words: tuple[str, ...] = ()) -> HomeCommand | None:
    """Parse one full extended imperative without using a model's guessed slots.

    None leaves ordinary lights, weather, conversations and other product paths
    to the existing router. A claimed but incomplete extended control must not
    fall through to vector retrieval or an inferred absolute setting.
    """
    if non_immediate_action_reason(text):
        return None
    raw = text.strip().strip("¿?¡! .;")
    for _ in range(3):
        before = raw
        for wake in wake_words:
            raw = re.sub(r"^" + re.escape(wake) + r"\b[\s,.:]*", "", raw, count=1, flags=re.IGNORECASE)
        if before == raw:
            break
    value = normalize_target(raw)
    value = re.sub(r"^(?:por favor[, ]+)?(?:(?:podes|puedes|podrias|podria)\s+)?", "", value)
    value = re.sub(r"[, ]+por favor$", "", value)
    head = re.match(r"^(\w+)\s+" + _ARTICLE + r"(" + _AIR + "|" + _COVER + "|" + _LIGHT + r")\b", value)
    if head is None:
        return None
    noun = head.group(2)
    domain = "climate" if re.fullmatch(_AIR, noun) else "cover" if re.fullmatch(_COVER, noun) else "light"
    known_verbs = "|".join((_SET, _OFF, _ON, _OPEN, _CLOSE, _STOP))
    if re.fullmatch(known_verbs, head.group(1)) is None:
        return None
    if domain == "light" and not re.search(r"\b(?:kelvin|\d+\s*k|calid[ao]|tibi[ao]|neutr[ao]|fri[ao])\b", value):
        return None
    invalid = HomeCommand(domain, value[head.start(2):], reason="incomplete_home_command")
    if domain == "climate":
        match = re.fullmatch(_SET + r"\s+" + _ARTICLE + _AIR + r"(?P<target>.*?)\s+(?:a|en)\s+(?P<value>.+?)(?:\s*(?:grados?(?: celsius| centigrados)?|°\s*c|celsius))?", value)
        if match and (number := _number(match["value"])) is not None:
            return HomeCommand(domain, match["target"].strip(), "set_temperature", {"temperature": number})
        match = re.fullmatch(_SET + r"\s+" + _ARTICLE + _AIR + r"(?P<target>.*?)\s+(?:en\s+)?(?:modo\s+)?(?P<mode>" + "|".join(sorted(_MODES, key=len, reverse=True)) + ")", value)
        if match:
            return HomeCommand(domain, match["target"].strip(), "set_hvac_mode", {"hvac_mode": _MODES[match["mode"]]})
        match = re.fullmatch(_OFF + r"\s+" + _ARTICLE + _AIR + r"(?P<target>.*)", value)
        if match:
            return HomeCommand(domain, match["target"].strip(), "set_hvac_mode", {"hvac_mode": "off"})
        return invalid
    if domain == "cover":
        match = re.fullmatch("(?:" + _SET + "|" + _OPEN + "|" + _CLOSE + r")\s+" + _ARTICLE + _COVER + r"(?P<target>.*?)\s+(?:al?|hasta el)\s+(?P<value>.+?)\s*(?:%|por ciento|porciento)", value)
        if match:
            number = _number(match["value"])
            if number is not None and float(number).is_integer() and 0 <= number <= 100:
                return HomeCommand(domain, match["target"].strip(), "set_cover_position", {"position": int(number)})
            return invalid
        match = re.fullmatch("(?P<verb>" + _OPEN + "|" + _CLOSE + "|" + _STOP + r")\s+" + _ARTICLE + _COVER + r"(?P<target>.*)", value)
        if match:
            operation = "stop_cover" if re.fullmatch(_STOP, match["verb"]) else "open_cover" if re.fullmatch(_OPEN, match["verb"]) else "close_cover"
            return HomeCommand(domain, match["target"].strip(), operation)
        return invalid
    # Color and optional brightness belong to the same light intent.
    match = re.fullmatch("(?:" + _SET + "|" + _ON + r")\s+" + _ARTICLE + _LIGHT + r"(?P<target>.*?)\s+(?:en\s+|a\s+)?(?P<color>\d{1,5}\s*(?:k|kelvin)|calid[ao]|tibi[ao]|neutr[ao]|fri[ao])(?:\s+(?:y\s+)?al?\s+(?P<brightness>.+?)\s*(?:%|por ciento|porciento))?", value)
    if not match:
        return invalid
    color = match["color"]
    numeric = re.fullmatch(r"(\d+)\s*(?:k|kelvin)", color)
    kelvin = int(numeric[1]) if numeric else {normalize_target(word): k for word, k in TEMP_WORDS_K.items()}.get(color)
    if kelvin is None:
        kelvin = {"tibia": 2700, "tibio": 2700}.get(color)
    if kelvin is None:
        return invalid
    parameters = {"color_temp_kelvin": kelvin}
    if match["brightness"] is not None:
        brightness = _number(match["brightness"])
        if brightness is None or not float(brightness).is_integer() or not 0 < brightness <= 100:
            return invalid
        parameters["brightness_pct"] = int(brightness)
    return HomeCommand(domain, match["target"].strip(), "color_temp_kelvin", parameters)
