"""Golden corpus para el RegexExtractor de KZA.

Cada entry es (texto, expected) donde expected es:
- None  → el regex DEBE rechazar (caer al LLM gate / reasoner)
- dict  → el regex DEBE matchear con ese intent + entity + slots

El corpus se mantiene VERSIONADO con el código del extractor: un fallo en
algún caso bloquea el merge. Cuando se agregue una nueva forma de comando
o un nuevo verbo voseo a vocab.py, también se agrega su caso acá.

Origen:
- POSITIVOS: extraídos de logs reales de kza-voice (`journalctl`) sobre 7 días
  de uso, deduplicados, normalizados al imperativo voseo canónico.
- NEGATIVOS: variantes adversariales construidas a partir de los positivos
  (negación, pretérito, infinitivo, 3ra persona, conversacional, entity
  no whitelisted), más casos puros de log que llegaron al pipeline pero
  no eran comandos.
"""
from __future__ import annotations


# =============================================================================
# POSITIVOS — el extractor DEBE matchear
# =============================================================================

POSITIVES: list[tuple[str, dict]] = [
    # turn_off — apagá / cortá / desactivá
    ("Nexa apagá la luz del escritorio",
        {"intent": "turn_off", "entity": "escritorio", "slots": {}}),
    ("Nexa apagá la luz del living",
        {"intent": "turn_off", "entity": "living", "slots": {}}),
    ("Nexa apagá la luz de la cocina",
        {"intent": "turn_off", "entity": "cocina", "slots": {}}),
    ("Nexa apagá la luz del baño",
        {"intent": "turn_off", "entity": "bano", "slots": {}}),
    ("Nexa apagá la luz del cuarto",
        {"intent": "turn_off", "entity": "cuarto", "slots": {}}),
    ("Nexa apagá la luz",
        {"intent": "turn_off", "entity": None, "slots": {}}),
    ("Nexa apagá el escritorio",
        {"intent": "turn_off", "entity": "escritorio", "slots": {}}),
    ("Nexa apagá la cocina",
        {"intent": "turn_off", "entity": "cocina", "slots": {}}),
    ("Nexa cortá la luz del escritorio",
        {"intent": "turn_off", "entity": "escritorio", "slots": {}}),
    ("apagá la luz",
        {"intent": "turn_off", "entity": None, "slots": {}}),

    # turn_on — prendé / encendé / activá
    ("Nexa prendé la luz del escritorio",
        {"intent": "turn_on", "entity": "escritorio", "slots": {}}),
    ("Nexa prendé la luz del living",
        {"intent": "turn_on", "entity": "living", "slots": {}}),
    ("Nexa prendé el escritorio",
        {"intent": "turn_on", "entity": "escritorio", "slots": {}}),
    ("Nexa encendé la luz del escritorio",
        {"intent": "turn_on", "entity": "escritorio", "slots": {}}),
    ("Nexa encendé la luz",
        {"intent": "turn_on", "entity": None, "slots": {}}),

    # set_brightness — bajá/subí + entidad luz, sin slot %
    ("Nexa bajá la luz del escritorio",
        {"intent": "set_brightness", "entity": "escritorio",
         "slots": {"direction": "down"}}),
    ("Nexa subí la luz del escritorio",
        {"intent": "set_brightness", "entity": "escritorio",
         "slots": {"direction": "up"}}),
    ("Nexa bajá la luz del living",
        {"intent": "set_brightness", "entity": "living",
         "slots": {"direction": "down"}}),
    ("Nexa bajá la luz del hogar",
        {"intent": "set_brightness", "entity": "hogar",
         "slots": {"direction": "down"}}),

    # set_brightness — con slot %
    ("Nexa bajá la luz al cincuenta por ciento",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 50}}),
    ("Nexa bajá la luz al cincuenta",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 50}}),
    ("Nexa poné la luz al cincuenta por ciento",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 50}}),
    ("Nexa poné la luz al setenta",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 70}}),
    ("Nexa poné la luz al treinta por ciento",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 30}}),
    ("Nexa subí la luz al ochenta",
        {"intent": "set_brightness", "entity": None,
         "slots": {"brightness_pct": 80, "direction": "up"}}),
    ("Nexa bajá la luz del escritorio al cuarenta por ciento",
        {"intent": "set_brightness", "entity": "escritorio",
         "slots": {"brightness_pct": 40}}),

    # set_temperature
    ("Nexa bajá la temperatura del aire",
        {"intent": "set_temperature", "entity": "temperatura",
         "slots": {"direction": "down"}}),
    ("Nexa subí la temperatura del aire",
        {"intent": "set_temperature", "entity": "temperatura",
         "slots": {"direction": "up"}}),
    ("Nexa bajá la temperatura",
        {"intent": "set_temperature", "entity": "temperatura",
         "slots": {"direction": "down"}}),

    # set_volume
    ("Nexa subí el volumen",
        {"intent": "set_volume", "entity": "volumen",
         "slots": {"direction": "up"}}),
    ("Nexa bajá el volumen",
        {"intent": "set_volume", "entity": "volumen",
         "slots": {"direction": "down"}}),
    ("Nexa poné el volumen al cincuenta",
        {"intent": "set_volume", "entity": "volumen",
         "slots": {"value": 50}}),

    # media controls
    ("Nexa pausá la música",
        {"intent": "media_pause", "entity": "musica", "slots": {}}),
    ("Nexa frená la música",
        {"intent": "media_pause", "entity": "musica", "slots": {}}),
    ("Nexa reproducí la música",
        {"intent": "media_play", "entity": "musica", "slots": {}}),
    ("Nexa tocá música",
        {"intent": "media_play", "entity": "musica", "slots": {}}),

    # open / close (requieren domain=cover explícito)
    ("Nexa abrí la persiana",
        {"intent": "open", "entity": "persiana", "slots": {}}),
    ("Nexa cerrá la persiana",
        {"intent": "close", "entity": "persiana", "slots": {}}),

    # toggle
    ("Nexa cambiá la luz del escritorio",
        {"intent": "toggle", "entity": "escritorio", "slots": {}}),
]


# Multi-intent — el extractor devuelve list[RegexMatch] con N entries.
MULTI_POSITIVES: list[tuple[str, list[dict]]] = [
    ("Nexa apagá la luz del escritorio y prendé el aire del living", [
        {"intent": "turn_off", "entity": "escritorio", "slots": {}},
        {"intent": "turn_on", "entity": "aire", "slots": {}},
    ]),
    ("Nexa apagá el escritorio y prendé la cocina", [
        {"intent": "turn_off", "entity": "escritorio", "slots": {}},
        {"intent": "turn_on", "entity": "cocina", "slots": {}},
    ]),
    ("Nexa apagá la luz también pausá la música", [
        {"intent": "turn_off", "entity": None, "slots": {}},
        {"intent": "media_pause", "entity": "musica", "slots": {}},
    ]),
]


# Limitaciones conocidas del regex (caen al LLM por diseño).
# Estos NO deben matchear el regex — el LLM los maneja con contexto.
KNOWN_LIMITATIONS: list[tuple[str, str]] = [
    ("Nexa apagá la luz del escritorio y de la cocina",
        "ellipsis_verb_omission"),
]


# Whisper hallucinations — fuzzy match aceptable (no obligatorio, mejora futura).
FUZZY_POSITIVES: list[tuple[str, dict]] = [
    ("Nexa abagá la luz del escritorio",  # apagá → abagá (Whisper traga 'p')
        {"intent": "turn_off", "entity": "escritorio", "slots": {}}),
    ("Nexa aprendé la luz del escritorio",  # prendé → aprendé
        {"intent": "turn_on", "entity": "escritorio", "slots": {}}),
    ("Nexa apagá la luz del vivitorio",  # escritorio → vivitorio
        {"intent": "turn_off", "entity": "escritorio", "slots": {}}),
]


# =============================================================================
# NEGATIVOS — el extractor NO debe matchear
# =============================================================================

NEGATIVES: list[tuple[str, str]] = [
    # ── Negación ──
    ("Nexa no apagués la luz", "negation"),
    ("Nexa no prendas la luz todavía", "negation"),
    ("no apagués nada", "negation"),
    ("Nexa nunca prendas esa luz", "negation"),

    # ── Pasado (1ra y 2da persona) ──
    ("Nexa ya apagué la luz", "past_tense"),
    ("Nexa ya prendí el aire", "past_tense"),
    ("Nexa apagaste la luz?", "past_tense"),
    ("Nexa prendiste la luz", "past_tense"),
    ("Nexa apagué la luz hace un rato", "past_tense"),

    # ── 3ra persona (NO es comando, es narrativa/reporte) ──
    ("La casa apaga la luz del escritorio.", "third_person"),
    ("Nexa habla de la luz del escritorio.", "third_person"),
    ("Nexa prendió esta gallera de la casa", "third_person"),
    ("ella apaga las luces cuando entra", "third_person"),
    ("Lucas prendió la luz del living", "third_person"),

    # ── Infinitivo (Whisper a veces emite infinitivo, no es imperativo) ──
    ("Nexa apagar la luz del escritorio.", "infinitive"),
    ("Nexa encender la luz del escritorio.", "infinitive"),
    ("Nexa prender el aire", "infinitive"),

    # ── Subjuntivo / condicional ──
    ("Nexa ojalá apagaras la luz", "subjunctive"),
    ("Nexa si prendés la luz me molesta", "conditional"),
    ("Nexa cuando apagués la luz avisame", "conditional"),
    ("Nexa si querés apagame la luz", "conditional"),

    # ── Pregunta ──
    ("Nexa apagaste la luz del escritorio?", "question"),
    ("Nexa qué luces están prendidas?", "question"),
    ("Nexa cómo prendo la luz?", "question"),
    ("¿Nexa apagaste la luz?", "question"),

    # ── Reporte de habla ──
    ("Nexa le dije que apagara la luz", "report_of_speech"),
    ("Nexa Lucas comentó que prendió la luz", "report_of_speech"),
    ("Nexa preguntó si apagamos la luz", "report_of_speech"),

    # ── Conversacional con verbo pero sin imperativo claro ──
    ("Nexa cambia la base, no sé, sí.", "conversational"),
    ("Nexa la mejor opción. Espera.", "no_verb"),
    ("Nexa cerrado, Lucas vos tenés contexto ahí.", "conversational"),
    ("Nexa parte de la lampa hoy está más caro", "conversational"),
    ("Nexa de cambiarle el código que viene de NexCore.", "conversational"),

    # ── Entidad inexistente en HA whitelist ──
    ("Nexa bajá la luz del mundo", "unknown_entity"),
    ("Nexa bajá la luz de la iniciación", "unknown_entity"),
    ("Nexa bajá la luz del encarnadero", "unknown_entity"),
    ("Nexa bajá la vista", "unknown_entity"),
    ("Nexa bajá la cierre", "unknown_entity"),

    # ── Wake word repetido sin comando ──
    ("Nexa", "incomplete"),
    ("Nexa, Nexa, Nexa.", "incomplete"),
    ("Nexa.", "incomplete"),

    # ── Comandos truncados / sin entidad clara ──
    ("Nexa bajita.", "incomplete"),
    ("Nexa quisi.", "noise"),

    # ── Ruido conversacional fuerte que sobrevive al verbo+entidad ──
    # NOTA: estos casos están en el límite; el regex matchea el comando inicial
    # antes de la coma. Confiamos en que el LLM gate posterior con texto
    # completo los rechace por contexto. Por eso NO están listados aquí —
    # son responsabilidad del gate, no del regex.

    ("Nexa bajá la luz al cincuenta por ciento, si no, si no, si no, si no.",
        "noise"),

    # ── Trailing question mark / muletilla final ──
    ("Nexa bajá la luz al cincuenta por ciento, ¿no?", "trailing_question"),
    ("Nexa subí la luz del escritorio, ¿no?", "trailing_question"),

    # ── 1ra persona plural / nosotros (no comando directo) ──
    ("Comandamos una mesa en el sotano.", "first_person_plural"),

    # ── Imperativo válido pero entidad fuera del scope domótico ──
    ("Nexa poné la mano", "out_of_scope"),
    ("Nexa cerrá la boca", "out_of_scope"),
]
