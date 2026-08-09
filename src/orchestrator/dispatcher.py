"""
Request Dispatcher
Enruta peticiones al path correcto (Fast Path vs Slow Path).

Fast Path (paralelo):
- Domotica via vector search
- Consultas simples via Router 7B
- Rutinas predefinidas
- Respuestas < 1 segundo

Slow Path (serializado):
- Razonamiento profundo con LLM 32B/70B
- Conversaciones multi-turno
- Peticiones complejas
- Cola priorizada

Ejemplo:
    dispatcher = RequestDispatcher(
        chroma_sync=chroma,
        router=router_7b,
        llm=llm_32b,
        context_manager=context_manager,
        priority_queue=queue
    )

    # Procesar peticion - automaticamente va al path correcto
    result = await dispatcher.dispatch(
        user_id="user_123",
        text="Prende la luz",
        zone_id="living"
    )
"""

import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

from src.core.logging import get_logger
from src.nlu.slot_extractor import merge_service_data
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.priority_queue import (
    Priority,
    Request,
    PriorityRequestQueue
)
from src.world.weather import DEFAULT_ENTITY as DEFAULT_WEATHER_ENTITY

logger = get_logger(__name__)


# Techo del POST a `weather.get_forecasts` (rama "mañana" de _handle_weather).
# Explícito y propio de este path a propósito: heredar el timeout de sesión
# (`home_assistant.timeout`, hoy 2.0s) ataba la ventana de mudez del clima a
# una config que se toca por motivos ajenos al clima. Es ~7x el presupuesto de
# 300ms del fast path, y está bien: esta rama NO es cacheada, hace un salto de
# red real y no puede entrar en 300ms. Lo que se acota acá es cuánto puede
# tardar el usuario en escuchar el degradado honesto.
WEATHER_FORECAST_TIMEOUT_S = 2.0

# Ventana del rate-limit del warning "FAST_WEATHER answered honestly with no
# data" (M8, review PR #15). No evita el spam de un fin de semana entero con
# el sensor caído — a 300s son ~288 warnings/día igual — el objetivo real es
# no repetir el warning por CADA request individual mientras el problema
# sigue vivo, no acotar el volumen diario total.
_WEATHER_NODATA_WARN_INTERVAL_S = 300.0


# Stems cortos que colisionan como SUBcadena dentro de palabras comunes y
# disparaban acciones fantasma sobre charla ambiente (2026-06-02):
#   baja ∈ traBAJAmos · pon ∈ suPONgo/PONen · sube ∈ suBEstimar ·
#   olvida ∈ inOLVIDAble
# Para estos exigimos límite de palabra, PERO admitimos el sufijo enclítico
# rioplatense (vocal temática opcional + pronombre): bajame, bajale, ponele,
# ponelo, olvidate... siguen ruteando. El resto de los keywords usa substring
# (preserva morfología: prende ∈ prender, apaga ∈ apagar). La preposición
# "para" se removió de CANCEL_KEYWORDS (demasiado común incluso como palabra
# entera; el comando real es "pará"). Ver review adversarial 2026-06-02 +
# project_nexa_command_detection_rootcause_2026-06-02.
_BOUNDARY_KEYWORDS = frozenset({
    "pon", "baja", "sube", "olvida", "olvidá",
    # Infinitivos (2026-06-04): como substring colisionan (bajar∈trabajar,
    # poner∈suponer, cerrar∈encerrar) → exigen límite de palabra.
    "encender", "cerrar", "abrir", "subir", "bajar", "poner",
})
# \bkw(e?<clítico>)?\b — la "e?" cubre la vocal temática del voseo (poné→ponele,
# olvidá→olvidate). "ponen"/"pone"/"ponemos"/"trabajamos"/"subestimar"/
# "inolvidable" NO terminan en un clítico pegado al stem → no matchean.
_ENCLITICS = "me|te|se|le|lo|la|nos|les|los|las"
_BOUNDARY_RE = {
    kw: re.compile(rf"\b{re.escape(kw)}(?:e?(?:{_ENCLITICS}))?\b")
    for kw in _BOUNDARY_KEYWORDS
}


def _kw_match(keyword: str, text_lower: str) -> bool:
    """True si `keyword` aparece en `text_lower`.

    Para los stems cortos ambiguos exige límite de palabra (admitiendo el
    enclítico rioplatense); para el resto usa substring (preserva variantes
    morfológicas como prende∈prender).
    """
    rx = _BOUNDARY_RE.get(keyword)
    if rx is not None:
        return rx.search(text_lower) is not None
    return keyword in text_lower


# Mapeo zone_id (ej: "zone_escritorio") → metadata.area en Chroma (ej:
# "Escritorio"). Acoplado a `home_assistant.area` de HA: si agregás un room
# nuevo allá, agregalo acá. Mantener sincronizado con
# src/rooms/room_context.create_default_rooms().
_ZONE_TO_AREA: dict[str, str] = {
    "zone_living": "Living",
    "zone_escritorio": "Escritorio",
    "zone_hall": "Hall",
    "zone_cocina": "Cocina",
    "zone_bano": "Baño",
    "zone_cuarto": "Cuarto",
}

# Aliases de room reconocidos para detección literal en el texto del usuario
# (decisión 1-B: room hablado pisa al mic). Cada key es el alias normalizado
# (lowercase + sin acentos) → area canónica. Mantener sincronizado con
# RoomConfig.aliases en src/rooms/room_context.create_default_rooms().
_ROOM_ALIASES_TO_AREA: dict[str, str] = {
    # Living — incluye los destrozos far-field del STT (2026-07-30). Es la
    # única habitación con nombre en inglés y la única que se rompe: medido
    # sobre 4 días de log, "del living" 12 veces contra "del libby" 9. Sin
    # estos alias el comando pierde el prefer_area y cae a la zona del mic
    # (pedías el living y prendía el escritorio). Mantener alineado con
    # ROOM_ALIASES de src/nlu/command_grammar.py.
    "living": "Living", "sala": "Living", "salon": "Living",
    "libby": "Living", "livin": "Living", "libin": "Living",
    # Escritorio
    "escritorio": "Escritorio", "oficina": "Escritorio", "estudio": "Escritorio",
    # Hall
    "hall": "Hall", "pasillo": "Hall", "entrada": "Hall",
    # Cocina
    "cocina": "Cocina", "kitchen": "Cocina",
    # Baño
    "bano": "Baño", "bathroom": "Baño",
    # Cuarto / Dormitorio
    "cuarto": "Cuarto", "dormitorio": "Cuarto", "habitacion": "Cuarto",
}


# Sustantivos que indican un dominio NO-luz explícito. Si el vector search
# devuelve una entidad `light.*` pero el texto pide uno de estos dominios, es
# un misfire (típicamente light.escritorio como fallback de la zona del mic):
# rechazamos el match de luz y dejamos que caiga al router/slow path. Bug
# fantasma 2026-05-29 (ver project_escritorio_light_phantom_toggles_2026-05-29).
_NON_LIGHT_DOMAIN_NOUNS: dict[str, str] = {
    "volumen": "media_player",
    "volume": "media_player",
    "temperatura": "climate",
    "termostato": "climate",
    "calefaccion": "climate",
    "aire": "climate",
    "clima": "climate",
    "grados": "climate",
}

# Sustantivos de _NON_LIGHT_DOMAIN_NOUNS que son termostato/AC. Usado por el
# guard en _classify_request que evita que un comando de domotica ("prendé
# el clima", "poné el clima en 22") se clasifique como clima hablado.
# Hallazgo 2026-08-04: "el clima" bare en WEATHER_KEYWORDS se comia esos
# comandos porque la rama de clima corre antes que DOMOTICS_KEYWORDS.
_CLIMATE_DOMAIN_NOUNS = frozenset(
    noun for noun, domain in _NON_LIGHT_DOMAIN_NOUNS.items() if domain == "climate"
)


def _strip_accents(text: str) -> str:
    """Quitar diacríticos (NFD + descartar combining marks).

    Helper compartido: antes había tres copias de este bloque inline
    (con `import re as _re` / `import unicodedata as _ud` locales pese a
    que ambos ya están importados a nivel de módulo) en lo que hoy son
    `_conflicting_domain`, `_resolve_prefer_area` y el guard de
    `_classify_request` (cleanup 2026-08-04, review de Finding 3).
    """
    norm = unicodedata.normalize("NFD", text)
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


# Sustantivos de luz: si aparecen, confiamos en el match light.* aunque haya
# un sustantivo no-luz (ej: 'poné la luz' nunca debe ser rechazado).
# Nota: el match se hace sobre texto normalizado SIN acentos → entradas sin tilde.
_LIGHT_NOUNS: tuple[str, ...] = (
    "luz", "luces", "lampara", "foco", "luminaria", "veladora",
)

# Respuesta inmediata y honesta cuando se rechaza un match de luz por conflicto
# de dominio. NO se rerutea al slow path: ese camino no resuelve el dominio
# (no hay entidad climate; el LLM del slow path no tiene tool-calling a HA) y,
# ante un fallo del reasoner, cuelga 5s y miente con un timeout falso (la cola
# no notifica el fail — issue C). Devolvemos feedback accionable al instante.
_DOMAIN_CONFLICT_RESPONSE: dict[str, str] = {
    "climate": "Todavía no tengo control de temperatura configurado.",
    "media_player": "No tengo cómo cambiar el volumen en esta zona todavía.",
}
_DOMAIN_CONFLICT_DEFAULT = "Ese comando no es para una luz, no pude ejecutarlo."


def _conflicting_domain(text: str, matched_domain: str) -> str | None:
    """Detectar misfire de dominio en el fast path.

    Devuelve el dominio en conflicto (para logging) o None si no hay conflicto.
    Conservador: solo dispara cuando el match es ``light.*``, el texto contiene
    un sustantivo no-luz explícito (volumen/temperatura/...) y NO contiene
    ningún sustantivo de luz (si el usuario dijo 'luz', confiamos en el match).
    El match es accent-insensitive con word-boundaries para no pegar substrings.

    Args:
        text: Texto del usuario (post-STT).
        matched_domain: Dominio de la entidad devuelta por el vector search.

    Returns:
        El dominio esperado por el texto (ej: "climate") si hay conflicto, o None.
    """
    if matched_domain != "light" or not text:
        return None
    norm = _strip_accents(text.lower())
    if any(re.search(rf"\b{re.escape(noun)}\b", norm) for noun in _LIGHT_NOUNS):
        return None
    for noun, domain in _NON_LIGHT_DOMAIN_NOUNS.items():
        if re.search(rf"\b{re.escape(noun)}\b", norm):
            return domain
    return None


def _resolve_prefer_area(text: str, zone_id: str | None) -> str | None:
    """Decidir qué area pasar como prefer_area al vector search.

    Prioridad:
        1. Si el texto menciona como token literal un alias conocido
           (regex con word-boundaries, accent-insensitive) → ese area.
        2. Si no, traducir zone_id del mic via _ZONE_TO_AREA.
        3. Si tampoco, None (sin restricción).

    El check de alias usa regex con \\b para no matchear substrings
    accidentales (ej: 'baño' en 'rebañar'). Es deliberadamente conservador:
    solo aliases enumerados, no embeddings — implementa el trade-off "B"
    discutido en sesión 2026-05-03.
    """
    if text:
        norm = _strip_accents(text.lower())
        for alias, area in _ROOM_ALIASES_TO_AREA.items():
            if re.search(rf"\b{re.escape(alias)}\b", norm):
                return area
    if zone_id:
        return _ZONE_TO_AREA.get(zone_id)
    return None


def _log_fire_and_reconcile_exception(task: "asyncio.Task") -> None:
    """Done-callback for the fire-and-forget HA dispatch task.

    Logs any exception so it doesn't disappear into 'Task exception was
    never retrieved' at GC time.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            f"[Dispatcher] _fire_and_reconcile_ha task failed: "
            f"{type(exc).__name__}: {exc}",
            exc_info=exc,
        )


class PathType(StrEnum):
    """Tipo de path para procesar la peticion"""
    FAST_DOMOTICS = "fast_domotics"       # Vector search + HA
    FAST_ROUTINE = "fast_routine"          # Rutinas predefinidas
    FAST_ROUTER = "fast_router"            # Router 7B para respuestas simples
    FAST_MUSIC = "fast_music"              # Spotify - búsqueda directa
    SLOW_MUSIC = "slow_music"              # Spotify - interpretación con LLM
    SLOW_LLM = "slow_llm"                  # LLM grande para razonamiento
    SYNC = "sync"                          # Comandos de sincronizacion
    ENROLLMENT = "enrollment"              # Registro de usuarios
    FEEDBACK = "feedback"                  # Feedback sobre respuestas
    FAST_LIST = "fast_list"                # List CRUD
    FAST_REMINDER = "fast_reminder"        # Reminder CRUD
    FAST_WEATHER = "fast_weather"           # Clima desde HA, sin red externa


@dataclass
class DispatchResult:
    """Resultado del dispatch"""
    path: PathType
    priority: Priority
    success: bool
    response: str
    intent: str = None
    action: dict = None
    timings: dict = field(default_factory=dict)
    error: str = None
    was_queued: bool = False
    queue_position: int = None
    user_id: str = None
    zone_id: str = None

    def to_dict(self) -> dict:
        return {
            "path": self.path.value,
            "priority": self.priority.name,
            "success": self.success,
            "response": self.response,
            "intent": self.intent,
            "timings": self.timings,
            "was_queued": self.was_queued,
            "queue_position": self.queue_position
        }


class RequestDispatcher:
    """
    Dispatcher que enruta peticiones al path optimo.

    Arquitectura:
                        ┌─────────────────┐
                        │    DISPATCH     │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
        ┌───────────┐    ┌───────────────┐   ┌───────────┐
        │ FAST PATH │    │  SLOW PATH    │   │  SPECIAL  │
        │ (paralelo)│    │ (serializado) │   │  COMMANDS │
        ├───────────┤    ├───────────────┤   ├───────────┤
        │• Domotica │    │• LLM Queue    │   │• Sync     │
        │• Router 7B│    │• Context/user │   │• Enroll   │
        │• Rutinas  │    │• Buffering    │   │• Feedback │
        └───────────┘    └───────────────┘   └───────────┘
    """

    # Palabras clave para detectar intents rapidos.
    # Incluye variantes voseo rioplatense (prendé/apagá/subí/bajá/abrí/cerrá/
    # poné/cambiá/activá/desactivá) porque `in text_lower` es substring-match
    # y la tilde rompe el match contra "prende"/"apaga"/etc.
    DOMOTICS_KEYWORDS = [
        "prende", "prendé", "enciende", "encendé",
        "apaga", "apagá",
        "sube", "subí", "baja", "bajá",
        "abre", "abrí", "cierra", "cerrá",
        "pon", "poné", "cambia", "cambiá",
        "activa", "activá", "desactiva", "desactivá",
        # Infinitivos NO derivables del prefijo conjugado (2026-06-04):
        # enciende≠encender (diptongación e→ie), cierra≠cerrar, abre≠abrir,
        # y los boundary-stems no alcanzan al infinitivo (\bbaja\b ∌ bajar).
        # Whisper produce infinitivos seguido ("encender la luz"). Todos van
        # con word-boundary (ver _BOUNDARY_KEYWORDS): bajar∈traBAJAR,
        # poner∈suPONER, cerrar∈enCERRAR colisionan como substring.
        "encender", "cerrar", "abrir", "subir", "bajar", "poner",
    ]

    SYNC_KEYWORDS = [
        "sincroniza", "sincronizá", "actualiza", "actualizá",
        "refresca", "refrescá", "sync",
    ]

    ENROLLMENT_KEYWORDS = [
        "agregar persona", "agregar usuario", "nueva persona",
        "registrar", "registrá", "add user",
    ]

    CANCEL_KEYWORDS = [
        "cancela", "cancelá", "olvida", "olvidá",
        "pará", "detente", "detené",
        "cancel", "stop",
    ]  # "para" (preposición) removido 2026-06-02 — solo "pará" cancela

    LIST_KEYWORDS = [
        "lista de", "agrega", "agregale", "quita", "quitale",
        "qué hay en la lista", "vacía la lista", "vaciala",
        "crea una lista", "borra la lista", "lista compartida",
    ]

    REMINDER_KEYWORDS = [
        "recuérdame", "recuerdame", "recordatorio",
        "avísame", "avisame", "qué tengo pendiente",
        "que tengo pendiente", "qué recordatorios",
        "que recordatorios", "todos los lunes",
        "todos los días", "todos los dias",
        "cada día", "cada dia", "cada lunes",
        "cada martes", "de lunes a viernes",
        "cancela el recordatorio",
    ]

    # Música - Fast path (búsqueda directa)
    MUSIC_DIRECT_KEYWORDS = [
        "pon música de", "música de", "canciones de", "reproduce",
        "playlist", "pausa", "pausá", "siguiente canción", "anterior",
        "qué suena", "qué está sonando",
        # Media-control lexemes — checked before DOMOTICS_KEYWORDS so verbs
        # like "subí" or "poné" that also appear in domotics are resolved by
        # the more-specific media noun, not by the generic verb.
        "música", "musica",        # "poné música", "pausá la música"
        "volumen",                 # "subí/bajá el volumen"
        "canción", "cancion",      # "siguiente/anterior canción"
        "reproducí", "reproduci",  # "reproducí la playlist"
    ]

    # Música - Slow path (requiere interpretación)
    MUSIC_CONTEXT_KEYWORDS = [
        "música para", "algo para", "algo tranquilo", "algo alegre",
        "música relajante", "música mientras", "ambiente"
    ]

    # Clima. Frases de DOS palabras a proposito: "temperatura"/"clima" solos
    # mapean a `climate` (el termostato/AC) en _NON_LIGHT_DOMAIN_NOUNS, y
    # "poné la temperatura en 22" / "prendé el clima" tienen que seguir
    # yendo a domotica. Lo que separa una de otra es el complemento, no el
    # sustantivo — por eso NO hay una entrada bare "el clima" acá: ya está
    # cubierta por "está el clima"/"esta el clima", y una entrada bare se
    # comía comandos de AC ("prendé el clima" -> hallazgo 2026-08-04, ver
    # _CLIMATE_DOMAIN_NOUNS y el guard en _classify_request).
    WEATHER_KEYWORDS = [
        "qué tiempo hace", "que tiempo hace",
        "está el clima", "esta el clima",
        "temperatura hace", "temperatura hay",
        "temperatura afuera", "grados hay afuera", "grados hace",
        "llueve", "va a llover", "lloviendo",
        "pronóstico", "pronostico",
        # Finding 3 (review 2026-08-04): sin esto, una pregunta como "¿tengo
        # que prender el clima o hace calor afuera?" no matcheaba ningún
        # WEATHER_KEYWORDS y caía al loop de DOMOTICS_KEYWORDS de abajo, que
        # matchea "prende" como substring de "prender" -> fast_domotics
        # incorrecto. "hace calor"/"hace frío" son vocabulario de clima
        # genérico, no colisionan con ningún comando de domótica.
        "hace calor", "hace frío", "hace frio",
    ]

    # Subconjunto de WEATHER_KEYWORDS que son FRAGMENTOS DE CLÁUSULA, no
    # consultas completas: aparecen igual de seguido como justificación
    # colgada de un comando ("prendé la estufa, hace frío") que como
    # consulta de clima. Review PR #14 (2026-08-06): con un tier único
    # esos comandos ruteaban FAST_WEATHER y la acción nunca se ejecutaba
    # (fallo domótico silencioso, verificado contra main). Un fragmento
    # solo rutea a clima si NINGÚN verbo de DOMOTICS_KEYWORDS aparece en
    # el texto, o si el texto es pregunta ("¿tengo que prender el clima o
    # hace calor afuera?" sigue siendo clima). Las consultas explícitas
    # ("qué tiempo hace", "pronóstico", "está el clima") NO llevan veto:
    # "está el clima bien, no hace falta prender nada" sigue en clima
    # aunque "prende" matchee como substring de "prender".
    WEATHER_CLAUSE_FRAGMENTS = frozenset({
        "hace calor", "hace frío", "hace frio",
        "llueve", "va a llover", "lloviendo",
    })

    # Marcadores de no-imperativo (Finding I1, review PR #15). El veto de
    # WEATHER_CLAUSE_FRAGMENTS de arriba asume que "fragmento + verbo
    # domótico sin '?'" es un comando con una cláusula de justificación
    # colgada ("prendé la estufa, hace frío"). Esa asunción se rompe cuando
    # el STT no transcribió la puntuación de una pregunta/observación real:
    # "puedo abrir las ventanas o va a llover", "tengo que prender el clima
    # o hace calor afuera", "no hace falta prender nada, hace calor" NO son
    # comandos aunque tengan verbo domótico + fragmento de clima sin "?".
    # La última es la clase más dura — negación como cláusula de necesidad,
    # documentada como el fallo que canceló el ruteo de clima por modelo
    # (NO-GO 2026-08-04): "no hace falta" niega el comando, no lo justifica.
    #
    # Si el texto contiene alguno de estos hints, el veto NO aplica — el
    # fragmento sigue ruteando a FAST_WEATHER en vez de caer al loop de
    # DOMOTICS_KEYWORDS. Asimetría de costo a propósito: sin hint y sin "?",
    # sesgar a domótica arriesga ejecutar una acción no pedida (fantasma);
    # con hint, sesgar a clima arriesga como mucho contestar con el
    # pronóstico una pregunta que no se hizo — mucho más barato. Solo se usa
    # DENTRO del veto (ver _classify_request): NO toca `is_question` global
    # ni el guard de adyacencia de más abajo — ensancharlo ahí reabriría
    # Finding 3 (preguntas de clima que el guard de adyacencia debe seguir
    # ignorando).
    _NON_COMMAND_HINTS = frozenset({
        "tengo que", "tenemos que", "hay que", "puedo", "podemos",
        "conviene", "no hace falta", "no hay que",
    })

    # Guard de adyacencia verbo-sustantivo (Finding 3, review 2026-08-04,
    # reemplaza el guard anterior "verbo en cualquier lado + sustantivo en
    # cualquier lado"). Ese guard viejo capturaba preguntas de clima
    # genuinas que solo mencionaban un verbo de domótica en otra cláusula
    # (ej: "¿tengo que prender el clima o hace calor afuera?"). Este exige
    # que el verbo esté INMEDIATAMENTE antes del sustantivo climático (con
    # a lo sumo un determinante de por medio: "prendé EL clima", "poné LA
    # temperatura"), que es la forma literal de los 6 casos Critical/
    # collision reales. No aflojar a "en cualquier lugar de la oración":
    # eso reintroduce Finding 3.
    #
    # Se combina en _classify_request con un segundo signal independiente
    # (bail-out por "?"/"¿") en vez de confiar solo en que "prender"/
    # "activar" no sean literales en DOMOTICS_KEYWORDS (los infinitivos NO
    # curados) — esa ausencia hoy ayuda a los casos híbridos, pero es
    # accidental: si algún día se agregan esos infinitivos a
    # DOMOTICS_KEYWORDS (mismo patrón que encender/cerrar/abrir/subir/
    # bajar/poner), la adyacencia sola dejaría de alcanzar y el signal de
    # interrogación sigue cubriendo.
    _DOMOTICS_VERBS_STRIPPED = sorted({_strip_accents(v) for v in DOMOTICS_KEYWORDS})
    _DOMOTICS_CLIMATE_ADJACENCY_RE = re.compile(
        r"\b(?:" + "|".join(re.escape(v) for v in _DOMOTICS_VERBS_STRIPPED) + r")\b"
        r"\s+(?:el|la|los|las)?\s*"
        r"\b(?:" + "|".join(re.escape(n) for n in sorted(_CLIMATE_DOMAIN_NOUNS)) + r")\b"
    )

    def __init__(
        self,
        chroma_sync,
        ha_client,
        routine_manager,
        router=None,
        llm=None,
        tts=None,
        context_manager: ContextManager = None,
        priority_queue: PriorityRequestQueue = None,
        buffered_streamer=None,
        music_dispatcher=None,
        list_manager=None,
        reminder_manager=None,
        response_handler=None,
        vector_threshold: float = 0.65,
        use_router_for_simple: bool = True,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
        before_handler_warn_ms: float = 5.0,
        require_known_speaker_for_actions: bool = False,
        unavailable_precheck_enabled: bool = True,
        weather_entity: str = DEFAULT_WEATHER_ENTITY,
    ):
        """
        Args:
            chroma_sync: Sincronizador de ChromaDB
            ha_client: Cliente de Home Assistant
            routine_manager: Gestor de rutinas
            router: Router 7B (opcional, para respuestas simples)
            llm: LLM grande para razonamiento
            tts: Motor TTS
            context_manager: Gestor de contextos por usuario
            priority_queue: Cola priorizada para slow path
            buffered_streamer: Streamer con buffering para TTS
            music_dispatcher: Dispatcher de música/Spotify
            vector_threshold: Umbral de similitud para vector search
            use_router_for_simple: Usar router 7B para preguntas simples
            hooks: Optional HookRegistry instance (plan #3 OpenClaw). When set,
                before_ha_action / before_tts_speak hooks fire on each invocation
                and after-events emit at pipeline checkpoints. Backward-compat:
                None → no hook calls, behavior identical to baseline.
            before_handler_warn_ms: Threshold (ms) for logging slow before-handlers
                in execute_before_chain. Default 5.0ms — anything above eats into
                the 300ms fast path budget.
            unavailable_precheck_enabled: Kill switch para el precheck de
                `is_entity_available` en `_fire_and_reconcile_ha` (default True).
                Ver comentario en `config/settings.yaml:home_assistant.
                unavailable_precheck_enabled` para el escenario de recuperación
                que justifica poder apagarlo.
            weather_entity: Entidad de HA que expone el clima (default
                `src.world.weather.DEFAULT_ENTITY`), usada por FAST_WEATHER.
                Se configura en `config/settings.yaml:home_assistant.
                weather_entity` y llega por MultiUserOrchestrator.
        """
        self.chroma = chroma_sync
        self.ha = ha_client
        self.routines = routine_manager
        self.router = router
        self.llm = llm
        self.tts = tts
        self.context_manager = context_manager or ContextManager()
        self.queue = priority_queue or PriorityRequestQueue()
        self.streamer = buffered_streamer
        self.music = music_dispatcher
        self.vector_threshold = vector_threshold
        self.use_router = use_router_for_simple
        self.list_manager = list_manager
        self.reminder_manager = reminder_manager
        self.response_handler = response_handler
        self.weather_entity = weather_entity
        self.hooks = hooks  # plan #3 OpenClaw — HookRegistry or None
        self._before_handler_warn_ms = before_handler_warn_ms
        # Voice-auth opcional (default OFF): exige speaker enrolado para ejecutar
        # acciones de domótica. Todos los disparos fantasma son User=unknown, así
        # que activarlo los corta — pero también bloquea invitados no enrolados.
        # Requiere speaker ID confiable. Ver project_escritorio_light_phantom_toggles.
        self._require_known_for_actions = require_known_speaker_for_actions
        # Kill switch del precheck de disponibilidad (default True — activo).
        # Apagalo si un reinicio de Z2M/HA deja el state_cache stale por más
        # de lo que tarda en sanar (WS state_changed, o el snapshot REST de
        # home_assistant.state_prefetch.full_refresh_interval_s) y eso empieza
        # a retener comandos contra dispositivos que en realidad ya volvieron.
        self._unavailable_precheck_enabled = unavailable_precheck_enabled

        # Estadisticas
        self._stats = {
            "total_requests": 0,
            "fast_path": 0,
            "slow_path": 0,
            "music_requests": 0,
            "weather_no_data": 0,
            "by_path": {p: 0 for p in PathType}
        }

        # Callback para respuestas del slow path
        self._slow_path_callbacks: dict[str, Callable] = {}

        # Rate-limiting para el warning del weather_no_data (review PR #14)
        self._last_weather_nodata_warn = float("-inf")

    async def dispatch(
        self,
        user_id: str,
        text: str,
        user_name: str = None,
        zone_id: str = None,
        permission_level: int = 3,
        on_response: Callable[[DispatchResult], None] = None,
        timeout: float = 5.0,
        service_filter: str | None = None,
        query_slots: dict | None = None,
    ) -> DispatchResult:
        """
        Procesar una peticion, enrutando al path correcto.

        Args:
            user_id: ID del usuario
            text: Texto de la peticion
            user_name: Nombre del usuario
            zone_id: Zona de origen (ej: "zone_escritorio")
            permission_level: Nivel de permisos
            on_response: Callback cuando hay respuesta (para slow path)
            timeout: Timeout maximo
            service_filter: Service HA del intent (ej: "turn_off",
                "set_brightness"). Si viene, se propaga al vector search
                para evitar matches por antónimos. Migración del path
                legacy → orchestrated (bug 2026-05-03 — el orchestrated
                descartaba el intent del LLMRouter).
            query_slots: Slots NLU (brightness_pct, rgb_color, etc.) que
                sobrescriben el preset del comando matcheado. También
                portado desde el legacy.
        """
        start_time = time.perf_counter()
        self._stats["total_requests"] += 1

        # Normalizar texto
        text = text.strip()
        text_lower = text.lower()

        # Obtener/crear contexto del usuario
        ctx = self.context_manager.get_or_create(
            user_id=user_id,
            user_name=user_name,
            zone_id=zone_id,
            permission_level=permission_level
        )

        # 1. Detectar comandos especiales
        special_result = await self._check_special_commands(text_lower, user_id, ctx)
        if special_result:
            special_result.timings["total"] = (time.perf_counter() - start_time) * 1000
            return special_result

        # 2. Detectar intent y prioridad
        path, priority = self._classify_request(text_lower, service_filter=service_filter)

        # 3. Enrutar al path correcto
        if path == PathType.FAST_MUSIC:
            # Música - búsqueda directa
            result = await self._fast_music_path(text, user_id)
            self._stats["fast_path"] += 1
            self._stats["music_requests"] += 1

        elif path == PathType.SLOW_MUSIC:
            # Música - requiere interpretación con LLM
            result = await self._slow_music_path(text, user_id)
            self._stats["slow_path"] += 1
            self._stats["music_requests"] += 1

        elif path in [PathType.FAST_DOMOTICS, PathType.FAST_ROUTINE, PathType.FAST_ROUTER]:
            # Fast path - procesar inmediatamente
            result = await self._fast_path(
                text=text,
                path=path,
                user_id=user_id,
                zone_id=zone_id,
                permission_level=permission_level,
                service_filter=service_filter,
                query_slots=query_slots,
            )
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_LIST:
            result = await self._fast_list_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_REMINDER:
            result = await self._fast_reminder_path(text, user_id, zone_id)
            self._stats["fast_path"] += 1

        elif path == PathType.FAST_WEATHER:
            result = await self._handle_weather(text, priority)
            self._stats["fast_path"] += 1

        else:
            # Slow path - encolar para LLM
            result = await self._slow_path(
                text=text,
                user_id=user_id,
                user_name=user_name or ctx.user_name,
                zone_id=zone_id,
                priority=priority,
                on_response=on_response,
                timeout=timeout
            )
            self._stats["slow_path"] += 1

        # Actualizar estadisticas
        self._stats["by_path"][path] += 1

        # Agregar timings
        result.timings["total"] = (time.perf_counter() - start_time) * 1000
        result.user_id = user_id
        result.zone_id = zone_id

        return result

    def _classify_request(
        self, text_lower: str, service_filter: str | None = None
    ) -> tuple[PathType, Priority]:
        """
        Clasificar peticion para determinar path y prioridad.

        Args:
            text_lower: Texto del usuario en minúsculas.
            service_filter: Clasificación previa del grammar/router (turn_on/
                turn_off). Si viene, manda sobre el keyword-matching: el
                upstream ya decidió que es domótica con alta confianza —
                re-adivinar por keywords mandaba "encender la luz" (infinitivo
                sin keyword) al SLOW_LLM → timeout 5s (bug 2026-06-04).

        Returns:
            (PathType, Priority)
        """
        # ⚠️ Contrato con request_router: el mapeo intent→service_filter de
        # allá (incluye set/set_brightness/set_color→"turn_on", review
        # 2026-08-09) cuenta con que ESTE par literal gana el fast path. Si
        # el mapeo emite un valor nuevo fuera del par, filtra el vector
        # search pero pierde FAST_DOMOTICS en silencio — extender acá también.
        if service_filter in ("turn_on", "turn_off"):
            return PathType.FAST_DOMOTICS, Priority.HIGH

        # Detectar música - contexto complejo (slow path)
        if self.music:
            for keyword in self.MUSIC_CONTEXT_KEYWORDS:
                if keyword in text_lower:
                    return PathType.SLOW_MUSIC, Priority.MEDIUM

            # Detectar música - búsqueda directa (fast path)
            for keyword in self.MUSIC_DIRECT_KEYWORDS:
                if keyword in text_lower:
                    return PathType.FAST_MUSIC, Priority.HIGH

        # Detect lists
        for keyword in self.LIST_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_LIST, Priority.HIGH

        # Detect reminders
        for keyword in self.REMINDER_KEYWORDS:
            if keyword in text_lower:
                return PathType.FAST_REMINDER, Priority.HIGH

        # Clima -> fuente local en HA. Va ANTES de domotica porque
        # "qué temperatura hace" comparte sustantivo con el termostato; el
        # complemento ("hace"/"afuera") es lo que desambigua. Va DESPUES de
        # musica/listas/recordatorios, que son mas especificas.
        #
        # Guard de defensa en profundidad (hallazgo 2026-08-04, endurecido
        # por Finding 3 de la re-review): un verbo de domotica INMEDIATAMENTE
        # ANTES de un sustantivo de clima explicito (temperatura/termostato/
        # calefaccion/aire/clima/grados, ver _DOMOTICS_CLIMATE_ADJACENCY_RE)
        # es SIEMPRE domotica — nunca una consulta hablada de clima — aunque
        # algun WEATHER_KEYWORDS futuro matchee sin complemento. Segundo
        # signal independiente: si el texto es una pregunta ("?"/"¿") nunca
        # es un comando, así que el guard nunca dispara. No relajar a "verbo
        # y sustantivo en cualquier lado de la oración": eso reintroduce
        # Finding 3 (preguntas de clima que solo mencionan un verbo de
        # domótica en otra cláusula, ej. "¿tengo que prender el clima o hace
        # calor afuera?").
        is_question = "?" in text_lower or "¿" in text_lower
        climate_command_adjacent = not is_question and bool(
            self._DOMOTICS_CLIMATE_ADJACENCY_RE.search(_strip_accents(text_lower))
        )
        if not climate_command_adjacent:
            domotics_verb = any(
                _kw_match(k, text_lower) for k in self.DOMOTICS_KEYWORDS
            )
            # Finding I1 (review PR #15): ver _NON_COMMAND_HINTS. Un hint de
            # no-imperativo presente desactiva el veto de más abajo — el
            # fragmento + verbo domótico deja de leerse como "comando con
            # justificación colgada" y vuelve a leerse como lo que
            # probablemente es: una pregunta u observación sin "?"
            # transcrita.
            non_command_hint = any(h in text_lower for h in self._NON_COMMAND_HINTS)
            for keyword in self.WEATHER_KEYWORDS:
                if keyword in text_lower:
                    if (not is_question and domotics_verb
                            and keyword in self.WEATHER_CLAUSE_FRAGMENTS
                            and not non_command_hint):
                        # "prendé la luz que hace frío": la cláusula es
                        # justificación de un comando — dejar que el loop
                        # de DOMOTICS_KEYWORDS de abajo lo capture. Con un
                        # _NON_COMMAND_HINTS presente ("no hace falta
                        # prender nada, hace calor") el veto NO aplica: cae
                        # al `return` de abajo y rutea FAST_WEATHER.
                        continue
                    return PathType.FAST_WEATHER, Priority.HIGH

        # Detectar domotica por keywords
        for keyword in self.DOMOTICS_KEYWORDS:
            if _kw_match(keyword, text_lower):
                return PathType.FAST_DOMOTICS, Priority.HIGH

        # Detectar rutinas
        if any(word in text_lower for word in ["rutina", "automatiza", "cuando"]):
            return PathType.FAST_ROUTINE, Priority.MEDIUM

        # Si tenemos router, preguntas simples van por fast path
        if self.router and self.use_router:
            # Preguntas muy simples
            if self._is_simple_query(text_lower):
                return PathType.FAST_ROUTER, Priority.MEDIUM

        # Todo lo demas va al slow path
        return PathType.SLOW_LLM, Priority.LOW

    def _is_simple_query(self, text_lower: str) -> bool:
        """Detectar si es una pregunta simple que el router puede manejar"""
        simple_patterns = [
            "que hora es",
            "que dia es",
            "como esta el clima",
            "que temperatura",
            "hola", "buenos dias", "buenas tardes", "buenas noches"
        ]
        return any(pattern in text_lower for pattern in simple_patterns)

    async def _check_special_commands(
        self,
        text_lower: str,
        user_id: str,
        ctx
    ) -> DispatchResult | None:
        """Verificar comandos especiales"""

        # Confirmacion pendiente - check BEFORE cancel keywords
        # so "no cancela" is treated as rejection, not as a cancel command
        if ctx.pending_confirmation:
            if any(word in text_lower for word in ["si", "confirma", "acepto", "ok"]):
                # Procesar confirmacion
                confirmation = ctx.pending_confirmation
                self.context_manager.clear_pending_confirmation(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTINE,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Confirmado",
                    intent="confirmation",
                    action=confirmation
                )
            elif any(word in text_lower for word in ["no", "cancela", "rechaza"]):
                self.context_manager.clear_pending_confirmation(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTINE,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Cancelado",
                    intent="rejection"
                )

        # Comando de cancelacion
        for keyword in self.CANCEL_KEYWORDS:
            if _kw_match(keyword, text_lower):
                cancelled = self.queue.cancel_user_request(user_id)
                return DispatchResult(
                    path=PathType.FAST_ROUTER,
                    priority=Priority.HIGH,
                    success=True,
                    response="Cancelado" if cancelled else "No hay nada que cancelar",
                    intent="cancel"
                )

        # Comando de sincronizacion
        for keyword in self.SYNC_KEYWORDS:
            if keyword in text_lower:
                return DispatchResult(
                    path=PathType.SYNC,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Sincronizando comandos...",
                    intent="sync"
                )

        # Comando de enrollment
        for keyword in self.ENROLLMENT_KEYWORDS:
            if keyword in text_lower:
                return DispatchResult(
                    path=PathType.ENROLLMENT,
                    priority=Priority.MEDIUM,
                    success=True,
                    response="Iniciando registro de usuario...",
                    intent="enrollment"
                )

        return None

    async def _fast_path(
        self,
        text: str,
        path: PathType,
        user_id: str,
        zone_id: str,
        permission_level: int,
        service_filter: str | None = None,
        query_slots: dict | None = None,
    ) -> DispatchResult:
        """
        Procesar peticion por fast path (paralelo, sin cola).
        """
        timings = {}

        if path == PathType.FAST_DOMOTICS:
            # Resolver prefer_area: 1) si el texto menciona literal un alias
            # de room conocido, ese gana (decisión 1-B); 2) si no, derivar
            # del zone_id del mic. Bug 2026-05-03: sin esto, el text-only
            # vector search elegía light.cuarto para queries genéricas
            # desde el escritorio.
            prefer_area = _resolve_prefer_area(text, zone_id)

            # Buscar comando en vector DB
            t0 = time.perf_counter()
            command = await self.chroma.asearch_command(
                text,
                self.vector_threshold,
                service_filter=service_filter,
                query_slots=query_slots,
                prefer_area=prefer_area,
            )
            timings["vector_search"] = (time.perf_counter() - t0) * 1000

            # Path dedicado para comandos globales ("toda la casa", "hogar").
            # light.hogar fue excluido del vector search (generaba FPs al
            # matchear cualquier "prendé la luz"), así que lo resolvemos acá
            # solo cuando el texto tiene keywords explícitas de scope global.
            tl = text.lower()
            global_kw = ("toda la casa", "todas las luces", "todo el hogar",
                         "del hogar", "en toda la", "la casa entera")
            if any(kw in tl for kw in global_kw):
                svc = "turn_off" if any(v in tl for v in ("apaga", "apagá", "apagar")) else "turn_on"
                logger.info(f"Global scope detected → {svc}@light.hogar")
                # Este bloque bypassa el vector search, que es donde se
                # mergean los query_slots (chroma_sync → merge_service_data).
                # Sin este merge, "luces de toda la casa al 50%" prendía todo
                # al brillo anterior y descartaba el 50% en silencio (review
                # 2026-08-09; alcanzable desde que set/set_brightness rutean
                # FAST_DOMOTICS). Solo aplica a turn_on: brightness/color no
                # son service_data de turn_off.
                data = (
                    merge_service_data({}, query_slots or {})
                    if svc == "turn_on"
                    else {}
                )
                command = {
                    "entity_id": "light.hogar",
                    "domain": "light",
                    "service": svc,
                    "description": "toda la casa",
                    "data": data,
                }

            # Guarda de conflicto de dominio (bug fantasma 2026-05-29): si el
            # vector search devolvió una luz pero el texto pide explícitamente
            # otro dominio (volumen→media_player, temperatura→climate), es un
            # misfire — light.escritorio es el fallback de la zona del mic. NO
            # disparamos la luz y devolvemos una respuesta inmediata y honesta
            # (no se rerutea al slow path: no resolvería el dominio y, si el
            # reasoner falla, cuelga 5s con un timeout falso — ver nota en
            # _DOMAIN_CONFLICT_RESPONSE).
            if command and command.get("domain") == "light":
                conflict = _conflicting_domain(text, "light")
                if conflict:
                    logger.info(
                        f"[Dispatcher] Domain conflict: text requests {conflict!r} "
                        f"but vector match is {command.get('entity_id')!r}; rejecting "
                        f"light action, returning immediate response"
                    )
                    return DispatchResult(
                        path=path,
                        priority=Priority.HIGH,
                        success=False,
                        response=_DOMAIN_CONFLICT_RESPONSE.get(
                            conflict, _DOMAIN_CONFLICT_DEFAULT
                        ),
                        intent="domain_conflict",
                        timings=timings,
                    )

            # Voice-auth opcional (default OFF): si está activo y el speaker no
            # está enrolado, NO ejecutamos la acción de domótica (todos los
            # disparos fantasma son User=unknown). Defensa adicional, complementa
            # el fix del initial_prompt + la guarda de dominio.
            if command and self._require_known_for_actions and (
                not user_id or user_id == "unknown"
            ):
                logger.info(
                    f"[Dispatcher] Speaker desconocido (user_id={user_id!r}) + "
                    f"require_known_speaker_for_actions → acción {command.get('service')!r}"
                    f"@{command.get('entity_id')!r} rechazada"
                )
                return DispatchResult(
                    path=path,
                    priority=Priority.HIGH,
                    success=False,
                    response="No te reconozco, no puedo hacer eso.",
                    intent="speaker_auth",
                    timings=timings,
                )

            if command:
                # 2026-04-28: removido el skip idempotent por cache. La integración
                # Hue↔HA tiene lag (a veces segundos) entre el estado real y el cache
                # vía WS. Con el skip, una luz físicamente prendida pero con cache="off"
                # ignoraba el "apagá" silenciosamente. La HA call es idempotent
                # server-side, así que mandarla siempre no tiene costo y garantiza
                # que el estado real se sincronice.
                # 2026-04-26: fire-and-forget. El usuario valida visualmente las
                # acciones de domótica (no quiere TTS ack). El HA call corre en
                # background; si falla, _fire_and_reconcile_ha habla el error.
                # Eso baja `home_assistant` de ~155ms a ~0ms en el camino crítico.
                _ha_task = asyncio.create_task(self._fire_and_reconcile_ha(command))
                _ha_task.add_done_callback(_log_fire_and_reconcile_exception)
                timings["home_assistant"] = 0.0

                return DispatchResult(
                    path=path,
                    priority=Priority.HIGH,
                    success=True,
                    response=command["description"],
                    intent="domotics",
                    action={**command, "fire_and_forget": True},
                    timings=timings
                )

            # No encontrado en vector DB, intentar con router
            if self.router:
                path = PathType.FAST_ROUTER

        if path == PathType.FAST_ROUTER and self.router:
            # Usar router para respuesta rápida. Soporta tanto LLMRouter
            # (async, candidate chain) como FastRouter suelto (sync, batch).
            t0 = time.perf_counter()
            try:
                if hasattr(self.router, "complete"):
                    result = await self.router.complete(text, max_tokens=128)
                    response = result.text if hasattr(result, "text") else str(result)
                    if hasattr(result, "endpoint_id"):
                        timings["router_endpoint"] = result.endpoint_id
                else:
                    response = self.router.generate([text], max_tokens=128)[0]
                timings["router"] = (time.perf_counter() - t0) * 1000

                return DispatchResult(
                    path=path,
                    priority=Priority.MEDIUM,
                    success=True,
                    response=response.strip(),
                    intent="simple_query",
                    timings=timings
                )
            except Exception as e:
                logger.warning(f"Router fallo, pasando a slow path: {e}")

        if path == PathType.FAST_ROUTINE:
            # Procesar rutina
            t0 = time.perf_counter()
            routine_result = await self.routines.handle(text)
            timings["routine"] = (time.perf_counter() - t0) * 1000

            if routine_result["handled"]:
                return DispatchResult(
                    path=path,
                    priority=Priority.MEDIUM,
                    success=routine_result["success"],
                    response=routine_result["response"],
                    intent="routine",
                    timings=timings
                )

        # Fallback a slow path
        return await self._slow_path(
            text=text,
            user_id=user_id,
            user_name=None,
            zone_id=zone_id,
            priority=Priority.LOW,
            on_response=None,
            timeout=5.0
        )

    async def _slow_path(
        self,
        text: str,
        user_id: str,
        user_name: str,
        zone_id: str,
        priority: Priority,
        on_response: Callable,
        timeout: float
    ) -> DispatchResult:
        """
        Procesar peticion por slow path (cola priorizada).
        """
        timings = {}

        # Crear evento para esperar respuesta
        response_event = asyncio.Event()
        result_holder = {"result": None}

        def on_complete(request: Request):
            result_holder["result"] = request.result
            try:
                response_event.set()
            except RuntimeError:
                pass

        def on_cancel(request: Request):
            result_holder["result"] = "Cancelado"
            try:
                response_event.set()
            except RuntimeError:
                pass

        def on_fail(request: Request):
            # Issue C: despertar al waiter ante un fallo del reasoner en vez de
            # dejarlo colgado hasta el timeout (que mentía con "tardé demasiado").
            result_holder["failed"] = True
            result_holder["error"] = request.error
            try:
                response_event.set()
            except RuntimeError:
                pass

        # Encolar peticion
        t0 = time.perf_counter()
        request = self.queue.enqueue(
            user_id=user_id,
            text=text,
            priority=priority,
            user_name=user_name,
            zone_id=zone_id,
            on_complete=on_complete,
            on_cancel=on_cancel,
            on_fail=on_fail
        )
        timings["queue"] = (time.perf_counter() - t0) * 1000

        position = self.queue.get_position(request.request_id)

        # Si hay callback, notificar que esta en cola
        if on_response and position and position > 1:
            on_response(DispatchResult(
                path=PathType.SLOW_LLM,
                priority=priority,
                success=True,
                response=f"Un momento, hay {position - 1} peticion(es) antes",
                intent="queued",
                was_queued=True,
                queue_position=position,
                timings=timings
            ))

        # Esperar respuesta o timeout
        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            request.cancel()
            return DispatchResult(
                path=PathType.SLOW_LLM,
                priority=priority,
                success=False,
                response="Lo siento, tarde demasiado. Intenta de nuevo.",
                intent="timeout",
                error="timeout",
                was_queued=True,
                timings=timings
            )

        timings["llm"] = request.processing_time * 1000 if request.processing_time else 0

        # Issue C: el reasoner falló (red/cloud caído/excepción). Despertamos por
        # on_fail (no por timeout), así que respondemos un error accionable de
        # inmediato en vez del falso "tardé demasiado".
        if result_holder.get("failed"):
            logger.warning(
                f"[Dispatcher] Slow path falló (reasoner): "
                f"{result_holder.get('error')}"
            )
            return DispatchResult(
                path=PathType.SLOW_LLM,
                priority=priority,
                success=False,
                response="No pude procesar eso ahora mismo, probá de nuevo.",
                intent="error",
                error=result_holder.get("error"),
                was_queued=True,
                queue_position=position,
                timings=timings
            )

        response = result_holder["result"]

        return DispatchResult(
            path=PathType.SLOW_LLM,
            priority=priority,
            success=request.status.name == "COMPLETED",
            response=response or "Sin respuesta",
            intent="conversation",
            was_queued=True,
            queue_position=position,
            timings=timings
        )

    async def _fast_music_path(self, text: str, user_id: str) -> DispatchResult:
        """
        Procesar comando de música por fast path (búsqueda directa).
        """
        timings = {}

        if not self.music:
            return DispatchResult(
                path=PathType.FAST_MUSIC,
                priority=Priority.HIGH,
                success=False,
                response="Spotify no está configurado",
                intent="music_error",
                timings=timings
            )

        t0 = time.perf_counter()

        # Obtener preferencias del usuario si existen
        user_prefs = None
        ctx = self.context_manager.get(user_id)
        if ctx and hasattr(ctx, 'music_preferences'):
            user_prefs = ctx.music_preferences

        # Procesar comando de música
        result = await self.music.process(text, user_preferences=user_prefs)
        timings["spotify"] = (time.perf_counter() - t0) * 1000

        return DispatchResult(
            path=PathType.FAST_MUSIC,
            priority=Priority.HIGH,
            success=result.success,
            response=result.response,
            intent=f"music_{result.intent.value}",
            action=result.details,
            timings=timings
        )

    async def _slow_music_path(self, text: str, user_id: str) -> DispatchResult:
        """
        Procesar comando de música por slow path (requiere LLM para interpretar).
        Ejemplo: "Pon música para una cena romántica a la luz de las velas"
        """
        timings = {}

        if not self.music:
            return DispatchResult(
                path=PathType.SLOW_MUSIC,
                priority=Priority.MEDIUM,
                success=False,
                response="Spotify no está configurado",
                intent="music_error",
                timings=timings
            )

        t0 = time.perf_counter()

        # Obtener preferencias del usuario
        user_prefs = None
        ctx = self.context_manager.get(user_id)
        if ctx and hasattr(ctx, 'music_preferences'):
            user_prefs = ctx.music_preferences

        # Procesar - el MusicDispatcher usará LLM internamente si es necesario
        result = await self.music.process(text, user_preferences=user_prefs)
        timings["spotify_with_llm"] = (time.perf_counter() - t0) * 1000

        return DispatchResult(
            path=PathType.SLOW_MUSIC,
            priority=Priority.MEDIUM,
            success=result.success,
            response=result.response,
            intent=f"music_{result.intent.value}",
            action={"interpreted_mood": result.details.get("interpreted_as")},
            timings=timings
        )

    async def _fast_list_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle list commands via ListManager."""
        if not self.list_manager:
            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=False, response="Listas no configuradas",
            )

        text_lower = text.lower()
        try:
            if any(w in text_lower for w in ["qué hay", "que hay", "dime la lista", "lee la lista"]):
                list_name = self._extract_list_name(text_lower)
                items = await self.list_manager.get_items(user_id, list_name)
                if not items:
                    response = "La lista está vacía"
                else:
                    item_texts = ", ".join(i.text for i in items)
                    response = f"En la lista tienes: {item_texts}"
            elif any(w in text_lower for w in ["vacía", "vacia", "limpia"]):
                list_name = self._extract_list_name(text_lower)
                await self.list_manager.clear_list(user_id, list_name)
                response = "Listo, vacié la lista"
            elif any(w in text_lower for w in ["borra la lista", "elimina la lista"]):
                list_name = self._extract_list_name(text_lower)
                if list_name and await self.list_manager.delete_list(user_id, list_name):
                    response = f"Borré la lista {list_name}"
                else:
                    response = "No encontré esa lista"
            elif any(w in text_lower for w in ["crea una lista", "nueva lista"]):
                shared = "compartida" in text_lower
                list_name = self._extract_list_name(text_lower)
                if list_name:
                    await self.list_manager.create_list(user_id, list_name, shared=shared)
                    response = f"Creé la lista {list_name}"
                else:
                    response = "No entendí el nombre de la lista"
            elif any(w in text_lower for w in ["quita", "quitale", "elimina", "tacha"]):
                item_text = self._extract_item_text(text_lower, removing=True)
                list_name = self._extract_list_name(text_lower)
                if item_text and await self.list_manager.remove_item(user_id, item_text, list_name):
                    response = f"Quité {item_text}"
                else:
                    response = "No encontré ese artículo en la lista"
            elif any(w in text_lower for w in ["agrega", "agregale", "añade", "pon"]):
                item_text = self._extract_item_text(text_lower, removing=False)
                list_name = self._extract_list_name(text_lower)
                if item_text:
                    await self.list_manager.add_item(user_id, item_text, list_name)
                    response = f"Agregué {item_text}"
                else:
                    response = "No entendí qué agregar"
            else:
                # Fallback: list all lists
                lists = await self.list_manager.get_all_lists(user_id)
                if lists:
                    names = ", ".join(lst.name for lst in lists)
                    response = f"Tienes estas listas: {names}"
                else:
                    response = "No tienes listas creadas"

            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=True, response=response,
            )
        except Exception as e:
            logger.error("List command error: %s", e)
            return DispatchResult(
                path=PathType.FAST_LIST, priority=Priority.HIGH,
                success=False, response="Hubo un error con la lista",
            )

    async def _fast_reminder_path(self, text: str, user_id: str, zone_id: str = None) -> DispatchResult:
        """Handle reminder commands via ReminderManager."""
        if not self.reminder_manager:
            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=False, response="Recordatorios no configurados",
            )

        text_lower = text.lower()
        try:
            if any(w in text_lower for w in ["qué recordatorios", "que recordatorios", "mis recordatorios"]):
                active = await self.reminder_manager.get_active(user_id)
                if not active:
                    response = "No tienes recordatorios activos"
                else:
                    lines = [self.reminder_manager.format_for_voice(r) for r in active[:5]]
                    response = "Tus recordatorios: " + ". ".join(lines)
            elif any(w in text_lower for w in ["qué tengo pendiente", "que tengo pendiente"]):
                today = await self.reminder_manager.get_today(user_id)
                if not today:
                    response = "No tienes nada pendiente hoy"
                else:
                    lines = [self.reminder_manager.format_for_voice(r) for r in today]
                    response = "Pendiente hoy: " + ". ".join(lines)
            elif "cancela" in text_lower and "recordatorio" in text_lower:
                import re
                match = re.search(r'recordatorio\s+(?:de\s+)?(.+)', text_lower)
                search_text = match.group(1).strip() if match else text_lower
                if await self.reminder_manager.cancel_by_text(user_id, search_text):
                    response = "Recordatorio cancelado"
                else:
                    response = "No encontré ese recordatorio"
            else:
                response = "Entendido, pero necesito el Router para interpretar la hora. Usa la API por ahora."

            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=True, response=response,
            )
        except Exception as e:
            logger.error("Reminder command error: %s", e)
            return DispatchResult(
                path=PathType.FAST_REMINDER, priority=Priority.HIGH,
                success=False, response="Hubo un error con el recordatorio",
            )

    async def _handle_weather(self, text: str, priority: Priority) -> DispatchResult:
        """Answer from HA. Two branches with very different costs.

        - "hoy" (default): cached read via `get_entity_state_cached`, no
          network hop — fits inside the 300ms fast path budget.
        - "mañana"/"pasado mañana": a REAL POST to `weather.get_forecasts`,
          awaited inline. NOT cached; it necessarily blows the fast path
          budget, and WEATHER_FORECAST_TIMEOUT_S is what bounds by how much.

        Nunca propaga una excepción. Un turno de voz mudo es el peor
        degradado posible: desde acá una excepción atraviesa `dispatch()`,
        `MultiUserOrchestrator.process()`, `request_router` y `voice_pipeline`
        sin que nadie la atrape, y el usuario no escucha NADA — ni respuesta,
        ni error, ni beep. Los FALLOS sí hablan (ver request_router.py).

        Args:
            text: Texto del usuario (ya identificado como consulta de clima).
            priority: Prioridad con la que se clasificó la petición.

        Returns:
            DispatchResult siempre hablable; `success=False` solo si hubo
            una excepción inesperada (que además se loguea a ERROR).
        """
        from src.world.weather import NO_DATA, describe_current, describe_forecast

        text_lower = text.lower()
        if "pasado mañana" in text_lower or "pasado manana" in text_lower:
            dia = "pasado mañana"
        elif "mañana" in text_lower or "manana" in text_lower:
            dia = "mañana"
        else:
            dia = None

        try:
            if dia:
                payload = await self.ha.call_service_with_response(
                    "weather", "get_forecasts", self.weather_entity,
                    {"type": "daily"},
                    timeout=WEATHER_FORECAST_TIMEOUT_S,
                )
                # `or {}` en CADA nivel, no `.get(clave, {})`: el default de
                # `dict.get` solo aplica cuando la CLAVE FALTA, no cuando la
                # clave existe con valor None. HA puede responder 200 con
                # {"service_response": null} (el servicio corrió y no tiene
                # nada que devolver) y ahí `.get("service_response", {})`
                # devuelve None -> AttributeError en el `.get` siguiente.
                service_response = (payload or {}).get("service_response") or {}
                entity_block = service_response.get(self.weather_entity) or {}
                forecast = entity_block.get("forecast") or []
                response = describe_forecast(forecast, dia)
            else:
                response = describe_current(
                    self.ha.get_entity_state_cached(self.weather_entity)
                )
            success = True
        except Exception as exc:  # noqa: BLE001 - red de seguridad anti-mudo
            logger.error(
                "Weather handler failed (entity=%s, dia=%s): %s",
                self.weather_entity, dia, exc, exc_info=True,
            )
            response, success = NO_DATA, False

        # Observabilidad (review PR #14, 2026-08-06): un weather_entity mal
        # configurado, un boot sin prefetch o un WS muerto degradan TODOS a
        # la misma respuesta honesta con success=True — sin esto son
        # invisibles para siempre: cero logs, stats de éxito. El contador
        # separa "habló el clima" de "habló la disculpa"; el warning
        # (rate-limited a _WEATHER_NODATA_WARN_INTERVAL_S, no repetir por
        # CADA request mientras el problema sigue vivo) apunta a la config.
        #
        # CRÍTICO: esto NO puede levantar una excepción. El método nunca
        # propaga excepciones; una falla acá atravesaría dispatch() y dejaría
        # al usuario escuchando NADA. El bloque está protegido para reforzar
        # la garantía anti-mudo: si observabilidad falla, se loguea y listo.
        try:
            from src.world.weather import NO_FORECAST
            if success and response in (NO_DATA, NO_FORECAST):
                self._stats["weather_no_data"] += 1
                now = time.monotonic()
                if now - self._last_weather_nodata_warn > _WEATHER_NODATA_WARN_INTERVAL_S:
                    self._last_weather_nodata_warn = now
                    logger.warning(
                        "FAST_WEATHER answered honestly with no data "
                        "(entity=%s, dia=%s, count=%d) — if chronic, check that "
                        "home_assistant.weather_entity exists in HA and that "
                        "the integration serves the requested horizon",
                        self.weather_entity, dia, self._stats["weather_no_data"],
                    )
        except Exception as obs_exc:  # noqa: BLE001 - no silenciar por observabilidad
            logger.error(
                "weather_no_data observability failed (response unaffected): %s",
                obs_exc, exc_info=True,
            )

        return DispatchResult(
            path=PathType.FAST_WEATHER, priority=priority,
            success=success, response=response, intent="weather",
        )

    def _extract_list_name(self, text: str) -> str | None:
        """Extract list name from text like 'la lista de compras' or 'la lista del hogar'."""
        import re
        # "la lista de X" / "la lista del X" / "a la lista X"
        match = re.search(r'(?:la lista (?:de(?:l)?|)\s+)(\w[\w\s]*?)(?:\s*$|[,.])', text)
        if match:
            return match.group(1).strip()
        # "lista compartida X"
        match = re.search(r'lista compartida\s+(?:de(?:l)?\s+)?(\w[\w\s]*?)(?:\s*$|[,.])', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_item_text(self, text: str, removing: bool = False) -> str | None:
        """Extract item text from commands like 'agrega leche a la lista'."""
        import re
        if removing:
            # "quita X de la lista"
            match = re.search(r'(?:quita|quitale|elimina|tacha)\s+(?:el |la |los |las )?(.+?)(?:\s+de la lista|\s*$)', text)
        else:
            # "agrega X a la lista" or just "agrega X"
            match = re.search(r'(?:agrega|agregale|añade|pon)\s+(.+?)(?:\s+a la lista|\s+en la lista|\s*$)', text)
        if match:
            item = match.group(1).strip()
            # Remove trailing list name reference
            item = re.sub(r'\s+(?:de|a|en)\s+la\s+lista.*$', '', item)
            return item if item else None
        return None

    async def dispatch_batch(
        self,
        requests: list[dict]
    ) -> list[DispatchResult]:
        """
        Procesar multiples peticiones en paralelo.

        Args:
            requests: Lista de {"user_id", "text", ...}

        Returns:
            Lista de resultados
        """
        tasks = [
            self.dispatch(**req)
            for req in requests
        ]
        return await asyncio.gather(*tasks)

    def get_queue_status(self) -> dict:
        """Obtener estado de la cola"""
        return self.queue.get_stats()

    def get_stats(self) -> dict:
        """Obtener estadisticas del dispatcher"""
        return {
            **self._stats,
            "queue": self.queue.get_stats(),
            "contexts": self.context_manager.get_stats()
        }

    def notify_user_waiting(
        self,
        user_id: str,
        zone_id: str,
        other_user_name: str
    ):
        """
        Notificar a un usuario que debe esperar.

        Usado cuando llega una peticion y hay otra en proceso.
        """
        message = f"Un momento, estoy respondiendo a {other_user_name}"
        if self.tts:
            # TODO: Enviar a zona especifica
            self.tts.speak(message)
        return message

    async def _fire_and_reconcile_ha(self, command: dict) -> None:
        """Ejecutar el call a HA en background y reportar solo en caso de error.

        Llamado desde `_fast_path` con `asyncio.create_task` para no bloquear
        el camino crítico del usuario. La latencia hacia HA + dispositivo
        sigue existiendo, pero ya no la paga el usuario en el TTS.

        Reconciliación criterio α (sesión 2026-04-26):
        - Si `call_service_ws` retorna False o lanza excepción → hablar el error
          usando `command.description`. La integración HA/WS ya tiene su propio
          timeout y fallback REST, así que llegar acá significa fallo real.
        - Si retorna True → silencio (el usuario valida visualmente).
        """
        domain = command.get("domain")
        service = command.get("service")
        entity_id = command.get("entity_id")
        description = command.get("description") or entity_id or "esa acción"

        # Plan #3 OpenClaw — before_ha_action chain (block / rewrite)
        call = None
        rewritten_data = command.get("data")
        if self.hooks is not None:
            from src.hooks import (
                HaActionCall,
                BlockResult,
                HaActionDispatchedPayload,
                HaActionBlockedPayload,
                execute_before_chain,
                execute_after_event,
            )
            call = HaActionCall(
                entity_id=entity_id or "",
                domain=domain or "",
                service=service or "",
                service_data=rewritten_data or {},
                user_id=command.get("user_id"),
                user_name=command.get("user_name"),
                zone_id=command.get("zone_id"),
                timestamp=time.time(),
            )
            result = execute_before_chain(
                self.hooks, "before_ha_action", call,
                warn_ms=self._before_handler_warn_ms,
            )
            if isinstance(result, BlockResult):
                logger.info(
                    f"[HA-CALL BLOCKED] {domain}.{service}@{entity_id} "
                    f"by rule={result.rule_name}: {result.reason}"
                )
                if self.response_handler is not None:
                    try:
                        self.response_handler.speak(result.reason or "No puedo hacer eso")
                    except Exception as e:
                        logger.warning(f"No pude hablar block reason: {e}")
                execute_after_event(
                    self.hooks,
                    "ha_action_blocked",
                    HaActionBlockedPayload(
                        timestamp=time.time(), call=call, block=result,
                    ),
                )
                return
            # Apply rewrite (if any) back to local vars
            domain = result.domain
            service = result.service
            entity_id = result.entity_id
            rewritten_data = result.service_data

        # HA acepta la llamada a una entidad unavailable y la filtra en
        # silencio (helpers/service.py:720), devolviendo success=true. Sin este
        # chequeo, "prendé la luz del cuarto" con la bombilla caída produce
        # silencio absoluto: ni voz, ni earcon, ni luz.
        # Falla ABIERTO ante None: sin dato en cache, llamamos igual.
        # Kill switch: self._unavailable_precheck_enabled (default True). Un
        # reinicio de Z2M/HA (recurrente, más con la migración Hue→Z2M en
        # curso) deja entidades genuinamente unavailable; si el WS de events
        # se cae en silencio en esa ventana, el cache no sana hasta el
        # snapshot REST (home_assistant.state_prefetch.full_refresh_interval_s,
        # 300s) y este precheck retendría comandos contra dispositivos que ya
        # volvieron. Apagar el flag en config/settings.yaml si eso pasa.
        if (
            self._unavailable_precheck_enabled
            and entity_id
            and self.ha.is_entity_available(entity_id) is False
        ):
            logger.warning(
                f"[HA-UNAVAILABLE] {domain}.{service}@{entity_id} "
                f"({description}) — no se envía la llamada"
            )
            if self.response_handler is not None:
                try:
                    self.response_handler.play_earcon(zone_id=command.get("zone_id"))
                except Exception as e:
                    logger.warning(f"No pude reproducir earcon de entidad caída: {e}")
            # Plan #3 OpenClaw — el precheck es una tercera forma de retener
            # un comando (además del before_ha_action block de arriba), y sin
            # emitir este evento el comando quedaba fuera del audit trail
            # (src/policies/audit_sqlite.py) — invisible en data/audit.db.
            if self.hooks is not None and call is not None:
                from dataclasses import replace
                from src.hooks import (
                    BlockResult,
                    HaActionBlockedPayload,
                    execute_after_event,
                )
                final_call = replace(
                    call,
                    domain=domain or "",
                    service=service or "",
                    entity_id=entity_id or "",
                    service_data=rewritten_data or {},
                )
                execute_after_event(
                    self.hooks,
                    "ha_action_blocked",
                    HaActionBlockedPayload(
                        timestamp=time.time(),
                        call=final_call,
                        block=BlockResult(
                            reason=f"{description}: entidad no disponible",
                            rule_name="entity_unavailable",
                        ),
                    ),
                )
            return

        t0 = time.perf_counter()
        err: str | None = None
        try:
            success = await self.ha.call_service_ws(
                domain, service, entity_id, rewritten_data,
            )
            dt = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[HA-CALL] {domain}.{service}@{entity_id} "
                f"success={success} took={dt:.0f}ms"
            )
        except Exception as e:
            err = str(e)
            logger.error(
                f"Reconcile error en {domain}.{service}@{entity_id}: {e}"
            )
            success = False

        # Plan #3 OpenClaw — emit after_event with dispatch result
        if self.hooks is not None and call is not None:
            from dataclasses import replace
            from src.hooks import (
                HaActionDispatchedPayload,
                execute_after_event,
            )
            final_call = replace(
                call,
                domain=domain or "",
                service=service or "",
                entity_id=entity_id or "",
                service_data=rewritten_data or {},
            )
            execute_after_event(
                self.hooks,
                "ha_action_dispatched",
                HaActionDispatchedPayload(
                    timestamp=time.time(),
                    call=final_call,
                    success=success,
                    error=err,
                ),
            )

        if success:
            return

        # Fallo real. Earcon (no frase) en la zona donde habló el usuario:
        # decisión 2026-07-25. La regla "domótica silenciosa" cubre los ÉXITOS
        # que el usuario valida visualmente; un fallo mudo es justo lo que hizo
        # invisible el bug del primer comando post-idle.
        logger.warning(
            f"[HA-FAIL] {domain}.{service}@{entity_id} ({description}) "
            f"err={err or 'success=False'}"
        )
        if self.response_handler is not None:
            try:
                self.response_handler.play_earcon(zone_id=command.get("zone_id"))
            except Exception as e:
                logger.warning(f"No pude reproducir earcon de fallo HA: {e}")
        else:
            logger.warning(
                f"HA fire-and-forget falló en {domain}.{service}@{entity_id} "
                f"sin response_handler — usuario no fue notificado"
            )


class MultiUserOrchestrator:
    """
    Orquestador completo para multiples usuarios.

    Coordina todos los componentes:
    - Context Manager
    - Priority Queue
    - Request Dispatcher
    - Request Processor

    Ejemplo:
        orchestrator = MultiUserOrchestrator(
            chroma_sync=chroma,
            ha_client=ha,
            routine_manager=routines,
            router=router_7b,
            llm=llm_32b,
            tts=tts
        )

        await orchestrator.start()

        # Procesar peticion
        result = await orchestrator.process(
            user_id="juan",
            text="Explícame la relatividad",
            zone_id="living"
        )

    Plan #2 OpenClaw kwargs (opcionales):
    - compactor: Compactor instance — si presente, dispara compactación
      en background al alcanzar `compaction_threshold` turnos
    - persister: ContextPersister — si presente, snapshot a disk al
      expirar contextos + hidratación al volver el usuario
    - compaction_threshold, keep_recent_turns: parámetros de la heurística
    Pasar `persister=...` también cambia start() del thread daemon legacy
    al async cleanup loop.
    """

    def __init__(
        self,
        chroma_sync,
        ha_client,
        routine_manager,
        router=None,
        llm=None,
        tts=None,
        speaker_identifier=None,
        user_manager=None,
        music_dispatcher=None,
        list_manager=None,
        reminder_manager=None,
        response_handler=None,
        max_context_history: int = 10,
        context_timeout: float = 300,
        auto_cancel_previous: bool = True,
        # Plan #2 OpenClaw
        compactor=None,
        persister=None,
        compaction_threshold: int = 6,
        keep_recent_turns: int = 3,
        # Plan #3 OpenClaw — plugin hooks
        hooks=None,
        before_handler_warn_ms: float = 5.0,
        require_known_speaker_for_actions: bool = False,
        unavailable_precheck_enabled: bool = True,
        weather_entity: str = DEFAULT_WEATHER_ENTITY,
    ):
        """Initialize the multi-user orchestrator.

        Args:
            response_handler: ResponseHandler para dar feedback al usuario. Se
                forwardea al RequestDispatcher; sin él, un fallo de HA en el
                fire-and-forget muere en un WARNING y el usuario no se entera
                (incidente 2026-07-25).
            hooks: Optional HookRegistry instance (plan #3 OpenClaw). When set,
                before_ha_action / before_tts_speak hooks fire on each invocation
                and after-events emit at pipeline checkpoints. Backward-compat:
                None → no hook calls, behavior identical to baseline.
            before_handler_warn_ms: Threshold (ms) for logging slow before-handlers.
                Forwarded to RequestDispatcher.
            unavailable_precheck_enabled: Kill switch del precheck de
                `is_entity_available` (default True). Forwarded to RequestDispatcher.
            weather_entity: Entidad de HA que expone el clima. Viene de
                `config/settings.yaml:home_assistant.weather_entity`. Sin este
                forward el dispatcher se quedaba siempre con el literal por
                defecto y, si la entidad de HA se llamaba distinto, el
                asistente contestaba "No tengo el dato del clima ahora mismo"
                para siempre — indistinguible de un sensor caído.
        """
        self._hooks = hooks  # plan #3 OpenClaw — exposed for log_hook_stats()
        # Componentes principales
        self.chroma = chroma_sync
        self.ha = ha_client
        self.routines = routine_manager
        self.router = router
        self.llm = llm
        self.tts = tts
        self.speaker_id = speaker_identifier
        self.user_manager = user_manager
        self.music = music_dispatcher

        # Plan #2 OpenClaw — track persister so start/stop can pick the right cleanup
        self._persister = persister

        # Inicializar subsistemas
        self._context_manager = ContextManager(
            max_history=max_context_history,
            inactive_timeout=context_timeout,
            compactor=compactor,
            persister=persister,
            compaction_threshold=compaction_threshold,
            keep_recent_turns=keep_recent_turns,
        )

        self._queue = PriorityRequestQueue(
            auto_cancel_previous=auto_cancel_previous
        )

        self._cancel_manager = self._queue  # Para acceso desde VoicePipeline

        # Public aliases for start()/stop() methods
        self.context_manager = self._context_manager
        self.queue = self._queue

        self.dispatcher = RequestDispatcher(
            chroma_sync=chroma_sync,
            ha_client=ha_client,
            routine_manager=routine_manager,
            router=router,
            llm=llm,
            tts=tts,
            context_manager=self._context_manager,
            priority_queue=self._queue,
            music_dispatcher=music_dispatcher,
            list_manager=list_manager,
            reminder_manager=reminder_manager,
            response_handler=response_handler,
            hooks=hooks,
            before_handler_warn_ms=before_handler_warn_ms,
            require_known_speaker_for_actions=require_known_speaker_for_actions,
            unavailable_precheck_enabled=unavailable_precheck_enabled,
            weather_entity=weather_entity,
        )

        self._running = False
        self._processor_task = None
        # Plan #2: track async cleanup task if persister is set
        self._async_cleanup_task = None

    async def start(self):
        """Iniciar el orquestador.

        Si fue construido con persister != None, el cleanup corre como
        asyncio task (necesario para snapshot de contextos al expirar).
        Si no, usa el thread daemon legacy (sin snapshot persistido).
        """
        if self._running:
            return

        self._running = True

        # Iniciar limpieza de contextos: async loop si hay persister, thread si no
        if self._persister is not None:
            self._async_cleanup_task = asyncio.create_task(
                self.context_manager.start_cleanup_loop_async()
            )
        else:
            self.context_manager.start_cleanup_thread()

        # Iniciar procesador de cola
        self._processor_task = asyncio.create_task(self._process_queue())

        logger.info("MultiUserOrchestrator iniciado")

    async def stop(self):
        """Detener el orquestador.

        Cancela peticiones pendientes y detiene el cleanup. Si el cleanup
        es async (persister != None), espera a que termine la última
        iteración. Si es thread, lo detiene cooperativamente.

        Excepciones del cleanup task durante stop se loguean (ERROR) pero
        no propagan — la parada del orquestador no debe fallar por
        problemas terminales.
        """
        self._running = False

        # Cancelar peticiones pendientes
        self.queue.cancel_all()

        # Detener cleanup (async si hay persister, thread en otro caso)
        if self._persister is not None:
            self.context_manager.stop_cleanup_loop_async()
            if self._async_cleanup_task:
                try:
                    await self._async_cleanup_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(
                        f"[Orchestrator] cleanup task failed during stop: {e}",
                        exc_info=True,
                    )
        else:
            self.context_manager.stop_cleanup_thread()

        # Detener procesador
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("MultiUserOrchestrator detenido")

    async def process(
        self,
        user_id: str,
        text: str,
        audio: any = None,
        zone_id: str = None,
        on_response: Callable = None,
        service_filter: str | None = None,
        query_slots: dict | None = None,
    ) -> DispatchResult:
        """
        Procesar una peticion de usuario.

        Args:
            user_id: ID del usuario (o None para identificar por voz)
            text: Texto transcrito
            audio: Audio original (para speaker ID si user_id es None)
            zone_id: Zona de origen
            on_response: Callback para respuestas
            service_filter: Service HA del intent clasificado por el
                LLMRouter (turn_on, turn_off, set_brightness, etc.).
                Se propaga al dispatcher para filtrar el vector search.
            query_slots: Slots NLU del LLMRouter (brightness_pct,
                rgb_color, color_temp_kelvin). Sobrescriben preset.
        """
        # Identificar usuario si no se proporciono
        if user_id is None and audio is not None and self.speaker_id:
            user_id, user_name = await self._identify_speaker(audio)
        else:
            user_name = None
            if self.user_manager and user_id:
                user = self.user_manager.get_user(user_id)
                if user:
                    user_name = user.name

        user_id = user_id or "unknown"
        permission_level = 0

        if self.user_manager and user_id != "unknown":
            user = self.user_manager.get_user(user_id)
            if user:
                permission_level = user.permission_level.value

        # Dispatch
        return await self.dispatcher.dispatch(
            user_id=user_id,
            text=text,
            user_name=user_name,
            zone_id=zone_id,
            permission_level=permission_level,
            on_response=on_response,
            service_filter=service_filter,
            query_slots=query_slots,
        )

    async def _identify_speaker(self, audio) -> tuple[str, str]:
        """Identificar usuario por voz"""
        if not self.speaker_id or not self.user_manager:
            return None, None

        embeddings = self.user_manager.get_all_embeddings()
        if not embeddings:
            return None, None

        match = self.speaker_id.identify(audio, embeddings)
        if match.is_known and match.user_id:
            user = self.user_manager.get_user(match.user_id)
            if user:
                return user.user_id, user.name

        return None, None

    async def _process_queue(self):
        """Procesar peticiones de la cola"""
        while self._running:
            try:
                request = await self.queue.dequeue_async(timeout=1.0)
                if request is None:
                    continue

                # Notificar a otros usuarios en espera
                await self._notify_waiting_users(request)

                # Procesar con LLM
                await self._process_llm_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error en proceso de cola", data={"error": str(e)})

    async def _notify_waiting_users(self, current_request: Request):
        """Notificar a usuarios en espera que deben esperar"""
        queue_status = self.queue.get_queue_status()

        for queued in queue_status:
            if queued["user_id"] != current_request.user_id:
                # Notificar que debe esperar
                ctx = self.context_manager.get(queued["user_id"])
                if ctx and ctx.zone_id and self.tts:
                    message = f"Un momento, estoy con {current_request.user_name}"
                    # TODO: Enviar a zona especifica
                    logger.debug(f"Notificando a {queued['user_id']}: {message}")

    async def _process_llm_request(self, request: Request):
        """Procesar peticion con el LLM"""
        try:
            # Construir prompt con contexto
            prompt = self.context_manager.build_prompt(
                request.user_id,
                request.text
            )

            # Generar respuesta
            if hasattr(self.llm, 'generate_stream'):
                # Streaming con verificacion de cancelacion
                response_parts = []
                for chunk in self.llm.generate_stream(prompt):
                    if request.is_cancelled:
                        logger.info(
                            "Request cancelado",
                            request_id=request.request_id,
                            user_id=request.user_id
                        )
                        return

                    # Verificar interrupciones de mayor prioridad
                    if self.queue.interrupt_for_priority(request.priority):
                        request.cancel()
                        logger.info(
                            "Request interrumpido por mayor prioridad",
                            request_id=request.request_id,
                            user_id=request.user_id
                        )
                        return

                    response_parts.append(chunk.get("token", ""))

                response = "".join(response_parts)
            else:
                response = self.llm.generate(prompt)

            # Agregar al contexto
            self.context_manager.add_turn(
                request.user_id, "user", request.text
            )
            self.context_manager.add_turn(
                request.user_id, "assistant", response
            )

            # Completar
            request.complete(response)

        except Exception as e:
            logger.error(
                "Error procesando request",
                request_id=request.request_id,
                user_id=request.user_id,
                data={"error": str(e)}
            )
            request.fail(str(e))

        finally:
            self.queue.clear_current()

    def get_stats(self) -> dict:
        """Obtener estadisticas completas"""
        return self.dispatcher.get_stats()

    def log_hook_stats(self) -> None:
        """Log a one-line HookRegistry stats dump. Intended for periodic
        invocation from a cleanup loop or HealthAggregator integration.

        Plan #3 OpenClaw I6 — counters are otherwise dead-end. Wiring this
        method to a periodic schedule (e.g. piggybacking on the context
        cleanup loop) is a follow-up task; today it's only callable on demand.
        """
        if self._hooks is None:
            return
        stats = self._hooks.get_stats()
        if stats["handler_failures"] > 0 or stats["slow_handler_count"] > 0:
            logger.info(
                f"[HookRegistry] handler_failures={stats['handler_failures']} "
                f"slow_handler_count={stats['slow_handler_count']} "
                f"after_in_flight={stats['after_tasks_in_flight']} "
                f"recent_errors={stats['handler_recent_errors']}"
            )
