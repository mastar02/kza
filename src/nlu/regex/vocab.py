"""Vocabulario controlado para el RegexExtractor de KZA.

Decisión de diseño clave: VOSEO-ONLY en imperativos.

En español rioplatense el voseo entrega imperativos morfológicamente
unívocos (`apagá`, `prendé`, `bajá`, `subí`, `poné`...) que casi nunca
aparecen en otros contextos sintácticos. El tuteo (`apaga`, `prende`)
es ambiguo con la 3ra persona del indicativo y genera falsos positivos
en transcripciones de TV / narrativa.

Imperativos tuteo NO van al regex: caen al LLM gate por defecto.

Origen del catálogo de entidades: snapshot de Home Assistant
(domain=light, friendly_name) tomado el 2026-04-28. Lista cerrada,
versionada, revisable. Cuando se agregue una nueva luz/dispositivo a
HA, se agrega acá explícitamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ─── Imperativos voseo por intent ───────────────────────────────────────────
#
# Cada imperativo es la forma EXACTA del voseo (verbo + acento agudo final).
# Los clíticos (`-lo`, `-la`, `-le`, `-los`, `-las`) se manejan como sufijo
# opcional en el regex (ver extractor.py).
#
# Reglas de pertenencia: solo entran formas que el hablante usa REALMENTE.
# No se incluyen variantes literarias o regionales raras.

IMPERATIVES_VOSEO: dict[str, tuple[str, ...]] = {
    "turn_off":     ("apagá", "cortá", "desactivá"),
    "turn_on":      ("prendé", "encendé", "activá"),
    "toggle":       ("cambiá",),
    "set_brightness_down": ("bajá", "reducí"),
    "set_brightness_up":   ("subí", "aumentá", "levantá"),
    "set_brightness_abs":  ("poné", "ponele"),  # con slot %
    "set_volume_down":     ("bajá",),
    "set_volume_up":       ("subí", "aumentá"),
    "set_volume_abs":      ("poné", "ponele"),
    "set_temperature_down": ("bajá", "reducí"),
    "set_temperature_up":   ("subí", "aumentá"),
    "set_temperature_abs":  ("poné", "ponele"),
    "media_pause":  ("pausá", "frená", "parálo", "pará"),
    "media_play":   ("reproducí", "tocá", "sonále", "sonalo"),
    "media_stop":   ("frená", "detené"),
    "media_next":   ("siguiente", "próxima", "saltá"),  # "siguiente" no es imperativo pero va con "tema"/"canción"
    "media_previous": ("anterior", "volvé"),
    "open":         ("abrí",),
    "close":        ("cerrá",),
    "scene_activate": ("activá",),  # "activá la escena X"
}


# Lista plana de TODOS los verbos voseo permitidos — útil para construir el
# regex del verbo en una sola alternation.
ALL_VOSEO_VERBS: tuple[str, ...] = tuple(
    sorted({v for verbs in IMPERATIVES_VOSEO.values() for v in verbs})
)


# Verbos polisémicos: el mismo voseo puede mapear a distintos intents según
# la entidad. Resolución se hace en el extractor combinando verbo + entidad.
# Ej: "bajá" + "luz" → set_brightness_down; "bajá" + "volumen" → set_volume_down.
POLISEMIC_VERBS: frozenset[str] = frozenset({
    "bajá", "subí", "poné", "ponele", "aumentá", "reducí", "levantá",
})


# ─── Entidades — whitelist derivada de Home Assistant ───────────────────────
#
# Cada entry mapea ALIAS HABLADOS → entity_id canónico de HA.
# El alias es lo que el usuario dice; el id de HA es lo que termina en el
# dispatcher. Aceptamos plurales y variantes con/sin artículo.

@dataclass(frozen=True)
class EntityAlias:
    """Mapeo entre forma hablada y entity_id de HA."""
    canonical: str            # forma normalizada interna (ej "escritorio")
    ha_entity_id: str | None  # entity_id de HA (ej "light.escritorio") o None si se resuelve por contexto
    domain: str               # "light", "media_player", "climate", etc.
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # Si la entidad NO requiere "luz" delante (ej "apagá el escritorio"),
    # marcar como standalone. Si requiere "luz del X", standalone=False.
    standalone: bool = False


# Luces (domain=light). Snapshot HA 2026-04-28: 42 luces con friendly_names
# Escritorio, Living, Cocina, Baño, Cuarto, Hogar, Pasillo, Balcón, Escalera.
# Las "led_living*" e individuales (e1, b2, ...) son sub-luces — el grupo
# room-level es lo que el usuario nombra.

LIGHT_ENTITIES: tuple[EntityAlias, ...] = (
    EntityAlias("escritorio", "light.escritorio", "light",
                aliases=("escritorios",), standalone=True),
    EntityAlias("living", "light.living", "light",
                aliases=("livin", "el living"), standalone=True),
    EntityAlias("cocina", "light.cocina", "light",
                aliases=("la cocina",), standalone=True),
    EntityAlias("bano", "light.bano", "light",
                aliases=("baño", "banio", "el baño"), standalone=True),
    EntityAlias("cuarto", "light.cuarto", "light",
                aliases=("dormitorio", "habitación"), standalone=True),
    EntityAlias("hogar", "light.hogar", "light",
                aliases=("toda la casa", "todas las luces"), standalone=False),
    EntityAlias("pasillo", "light.pasillo", "light",
                aliases=(), standalone=True),
    EntityAlias("balcon", "light.balcon", "light",
                aliases=("balcón",), standalone=True),
    EntityAlias("escalera", "light.escalera", "light",
                aliases=("escaleras",), standalone=True),
    EntityAlias("led_living", "light.led_living", "light",
                aliases=("led del living", "leds del living"), standalone=False),
)


# Climate / aire acondicionado (no apareció en HA snapshot pero sí en logs
# como entidad mencionada — usuario dice "aire" → ahí mapeo a un climate.*
# que se resuelve por contexto en HA. Por ahora mantenemos alias hablado
# y el dispatcher resuelve).
CLIMATE_ALIASES: tuple[EntityAlias, ...] = (
    EntityAlias("aire", None, "climate",
                aliases=("aire acondicionado", "aa", "el aire"), standalone=True),
    EntityAlias("temperatura", None, "climate",
                aliases=("la temperatura", "temp"), standalone=True),
)


# Media player aliases (volumen, música).
MEDIA_ALIASES: tuple[EntityAlias, ...] = (
    EntityAlias("volumen", None, "media_player",
                aliases=("el volumen",), standalone=True),
    EntityAlias("musica", None, "media_player",
                aliases=("música", "la música"), standalone=True),
)


# Cobertores (persiana, cortina).
COVER_ALIASES: tuple[EntityAlias, ...] = (
    EntityAlias("persiana", None, "cover",
                aliases=("persianas", "la persiana"), standalone=True),
    EntityAlias("cortina", None, "cover",
                aliases=("cortinas", "la cortina"), standalone=True),
)


# Concatenación final: lista plana de todas las entidades reconocibles.
ALL_ENTITIES: tuple[EntityAlias, ...] = (
    *LIGHT_ENTITIES,
    *CLIMATE_ALIASES,
    *MEDIA_ALIASES,
    *COVER_ALIASES,
)


def build_alias_index() -> dict[str, EntityAlias]:
    """Diccionario alias_normalizado → EntityAlias para lookup rápido.

    Ejemplo:
        {"escritorio": <EntityAlias>, "escritorios": <EntityAlias>,
         "living": <EntityAlias>, "livin": <EntityAlias>, ...}
    """
    idx: dict[str, EntityAlias] = {}
    for ent in ALL_ENTITIES:
        idx[ent.canonical] = ent
        for alias in ent.aliases:
            idx[alias.lower()] = ent
    return idx


ALIAS_INDEX = build_alias_index()


# ─── Slots numéricos: cardinales hablados → int ─────────────────────────────
#
# Whisper transcribe "cincuenta" como palabra, no como "50". Convertimos.
# Cubrimos 0-100 que es el rango usable en domótica (brightness, %, °C).

CARDINALS: dict[str, int] = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintidós": 22,
    "veintitres": 23, "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25,
    "veintiseis": 26, "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}


def parse_cardinal(text: str) -> int | None:
    """Convertir un fragmento como 'cincuenta', 'cuarenta y cinco', '70' a int.

    Soporta:
    - Dígitos directos: "50", "70" → 50, 70
    - Cardinales simples: "cincuenta" → 50
    - Cardinales compuestos con 'y': "cuarenta y cinco" → 45
    - Compuestos sin 'y' (Whisper a veces los omite): "cuarenta cinco" → 45

    Devuelve None si no se puede parsear.
    """
    text = text.strip().lower()
    if not text:
        return None
    # 1) Dígitos directos
    if text.isdigit():
        n = int(text)
        return n if 0 <= n <= 100 else None
    # 2) Cardinal simple
    if text in CARDINALS:
        return CARDINALS[text]
    # 3) Compuesto con 'y'
    parts = text.replace(" y ", " ").split()
    total = 0
    for p in parts:
        if p not in CARDINALS:
            return None
        total += CARDINALS[p]
    return total if 0 <= total <= 100 else None


# ─── Modificadores de color / temperatura de luz ────────────────────────────

COLOR_TEMP_WORDS: dict[str, int] = {
    # nombre → kelvin aproximado
    "cálida": 2700,
    "calida": 2700,
    "tibia": 3000,
    "neutra": 4000,
    "neutral": 4000,
    "blanca": 5000,
    "fría": 6500,
    "fria": 6500,
}

COLOR_RGB_WORDS: dict[str, tuple[int, int, int]] = {
    "roja": (255, 0, 0),       "rojo": (255, 0, 0),
    "azul": (0, 0, 255),
    "verde": (0, 255, 0),
    "amarilla": (255, 255, 0), "amarillo": (255, 255, 0),
    "naranja": (255, 165, 0),
    "rosa": (255, 105, 180),
    "violeta": (138, 43, 226), "morada": (128, 0, 128), "morado": (128, 0, 128),
}


# ─── Wake words a stripear antes del match ──────────────────────────────────
#
# Ya filtrados por el wake detector pero pueden quedar al inicio del texto.

WAKE_WORDS: tuple[str, ...] = ("nexa", "kaza", "alexa", "nexia", "nexta", "casa")
