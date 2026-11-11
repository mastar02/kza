"""Verification phrase for the voice opt-in (U3b-4b-2): a fixed word list, a draw and a normaliser.

Pure, no I/O. The phrase is four words drawn with ``secrets.choice`` from
PHRASE_WORDS. The list holds only everyday nouns (animals, food, nature,
objects, characters) and excludes, by construction and by test, every
command, room, device, number, time, colour, wake and memory word, so a
phrase can never read as a home command or an "acordate que" request. The
exclusion also covers likely mis-hearings: no word is one edit or one common
Spanish ASR confusion (b/v, s/z/c, ll/y, h, qu/k, g/j, rr/r, ñ/n, plural s)
away from that vocabulary. Words are stored without accents or ñ, the form
``normalize_utterance`` gives the STT output, so they compare exactly.

``word_near`` and ``phrase_overlap`` let the router recognise a near miss of
an open window's phrase (U3b-4b review L2) and the windows reject a drawn word
that sits near a home device or area name.
"""

import re
import secrets
import unicodedata

PHRASE_LENGTH = 4
# The one reply to a matched phrase, whatever the evaluation later decides (U3b-4b-2).
VERIFICATION_REPLY = "Recibí la frase. El resultado aparece en Mis recuerdos."

PHRASE_WORDS: tuple[str, ...] = (
    # Animals
    "ornitorrinco", "ardilla", "ballena", "rinoceronte", "aguila", "camello", "canguro", "caracol", "jabali", "mapache",
    "pinguino", "cisne", "cocodrilo", "conejo", "cordero", "cuervo", "elefante", "gaviota", "gallina", "leopardo",
    "ganso", "golondrina", "gorila", "tiburon", "gusano", "hormiga", "jirafa", "lagarto", "langosta",
    "liebre", "lobo", "loro", "mariposa", "medusa", "mosca", "oveja", "paloma", "calamar",
    "salamandra", "camaleon", "escarabajo", "libelula", "pulpo", "cangrejo", "sapo", "serpiente", "tigre", "hipopotamo",
    "tortuga", "delfin", "zorro", "ostra", "murcielago", "bisonte", "castor", "erizo", "avestruz",
    "lechuza", "nutria", "pelicano", "armadillo",
    # Food
    "arandano", "cereza", "ciruela", "damasco", "durazno", "frambuesa", "frutilla", "granada", "guayaba",
    "mandarina", "nispero", "membrillo", "nuez", "papaya", "pistacho", "castana",
    "aceituna", "almendra", "avena", "mostaza", "canela", "cebolla", "oregano", "harina", "tomillo", "oliva",
    "pimienta", "jengibre", "tomate", "zanahoria", "lenteja", "garbanzo", "trigo", "lechuga", "avellana",
    "pepino", "batata", "calabaza", "espinaca", "perejil", "albahaca", "vainilla",
    # Nature
    "meseta", "bosque", "cascada", "repollo", "peninsula", "desierto", "archipielago", "glaciar", "isla",
    "estepa", "laguna", "llanura", "tundra", "orilla", "piedra", "meandro", "pradera", "fiordo",
    "selva", "barranco", "manantial", "rama", "semilla", "sendero", "girasol", "helecho", "musgo", "cactus",
    "quebrada", "roble", "altiplano", "palmera", "hierba", "espiga", "estrella", "cordillera", "caverna", "brisa",
    "niebla", "granizo", "trueno", "tormenta", "arroyo", "pantano", "acantilado", "higuera",
    # Objects
    "ancla", "anillo", "barco", "pergamino", "botella", "telescopio", "canasta", "carpeta", "castillo", "sonajero",
    "rompecabezas", "cuaderno", "cuchara", "escoba", "espejo", "globo", "guitarra", "hamaca", "abanico", "ladrillo",
    "libro", "tintero", "maleta", "barrilete", "martillo", "mochila", "moneda", "paraguas", "pelota", "peine",
    "pincel", "pluma", "regalo", "sombrero", "cascabel", "tijera", "tambor", "ajedrez", "trompeta", "mosaico",
    "zapato", "bufanda", "guante", "chaqueta", "brujula", "tornillo", "vitral", "cuerda", "submarino",
    "medalla", "corona", "escudo", "espada", "flecha", "velero", "canoa", "bicicleta",
    "helicoptero", "tractor", "carreta", "carrusel", "tenedor", "cuchillo", "plato", "acordeon", "embudo",
    # Places, stories and characters
    "aventura", "bandera", "biblioteca", "enigma", "zorzal", "leyenda", "misterio", "poema", "tesoro", "acuario",
    "trineo", "planetario", "molino", "granja", "aldea", "pueblo", "mercado", "museo", "teatro", "gaucho",
    "desfile", "carnaval", "orquesta", "payaso", "pirata", "vaquero", "marinero", "cartero", "panadero", "astronauta",
    "duende", "gigante", "unicornio", "picaflor", "alfarero", "caballero", "princesa", "explorador", "inventor",
    "detective", "jinete",
)


def draw_phrase(choice=secrets.choice) -> str:
    """Four words drawn independently with a cryptographic choice."""
    return " ".join(choice(PHRASE_WORDS) for _ in range(PHRASE_LENGTH))


def normalize_utterance(text: str, wake_words: tuple[str, ...] = ()) -> str:
    """Casefold, strip accents and punctuation, drop a leading wake word, collapse spaces.

    Applied to the transcript before any comparison with an open window, so
    "Nexa, gato perro... ¡Caballo conejo!" reads as "gato perro caballo conejo".
    """
    if type(text) is not str:
        return ""
    value = "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                    if not unicodedata.combining(char))
    value = " ".join(re.sub(r"[^\w\s]|_", " ", value).split())
    for word in wake_words or ("nexa",):
        wake = normalize_utterance(word, ("",)) if word else ""
        if wake and (value == wake or value.startswith(wake + " ")):
            value = value[len(wake):].strip()
            break
    return value


NEAR_MISS_MIN_WORDS = 3
NEAR_MISS_MAX_WORDS = 8
_NEAR_EDIT_MIN_LENGTH = 4


def edit_distance(left: str, right: str, bound: int = 2) -> int:
    """Levenshtein distance, cut off at ``bound + 1`` once it can only grow past ``bound``."""
    if abs(len(left) - len(right)) > bound:
        return bound + 1
    previous = list(range(len(right) + 1))
    for row, char in enumerate(left, 1):
        current = [row]
        for column, other in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (char != other)))
        if min(current) > bound:
            return bound + 1
        previous = current
    return min(previous[-1], bound + 1)


_PHONETIC_RULES = (
    ("ch", "\x00"), ("qu", "k"), ("ll", "y"), ("h", ""), ("v", "b"), ("z", "s"), ("w", "u"), ("x", "ks"),
    ("ce", "se"), ("ci", "si"), ("c", "k"), ("ge", "je"), ("gi", "ji"), ("\x00", "ch"),
)


def phonetic_key(word: str) -> str:
    """A coarse Spanish sound key: silent h, b/v, s/z/soft c, ll/y, qu/hard c/k, soft g/j, doubles collapsed."""
    value = normalize_utterance(word, ("",)).replace("ñ", "n")
    for before, after in _PHONETIC_RULES:
        value = value.replace(before, after)
    collapsed = []
    for char in value:
        if not collapsed or collapsed[-1] != char:
            collapsed.append(char)
    return "".join(collapsed).removesuffix("s") or value


def word_near(heard: str, word: str) -> bool:
    """True when ``heard`` is ``word``, one edit from it (both of four or more letters) or sounds the same."""
    if heard == word:
        return True
    if min(len(heard), len(word)) >= _NEAR_EDIT_MIN_LENGTH and edit_distance(heard, word, 1) <= 1:
        return True
    return phonetic_key(heard) == phonetic_key(word)


def phrase_overlap(heard_words, phrase_words) -> int:
    """How many phrase words a distinct heard word sits near (greedy, each heard word used once)."""
    remaining = list(heard_words)
    count = 0
    for word in phrase_words:
        for index, heard in enumerate(remaining):
            if word_near(heard, word):
                del remaining[index]
                count += 1
                break
    return count


def near_phrase(normalized: str, phrases) -> bool:
    """True when a normalised utterance of at most eight words matches three of the four words of any phrase."""
    heard = normalized.split()
    if not heard or len(heard) > NEAR_MISS_MAX_WORDS:
        return False
    return any(phrase_overlap(heard, phrase.split()) >= NEAR_MISS_MIN_WORDS for phrase in phrases)
