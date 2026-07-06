"""Textual wake channel: "nexa" safety net over the ambient transcription stream.

El wake acústico (openwakeword) puede fallar en condiciones far-field (spec
2026-07-05, "compuerta acústica"). Este módulo agrega una red de seguridad
puramente textual: si una utterance del ambient path (ya transcripta, ni de TV
ni eco de la propia TTS del asistente) contiene una mención de "nexa" —exacta
o a corta distancia de edición—, se
re-despacha como comando con el texto ya transcripto (`CommandEvent.wake_text`),
evitando un segundo pase de STT.

Variantes calibradas contra `ambient.db` (auditoría 2026-07-05, utterances
reales etiquetadas como voz humana): "nexa" (19 ocurrencias), "next up" (7 —
mishearing típico en inglés, bigrama), "lexa" (1). El default
`variants=("nexa", "next up")` cubre las dos variantes con volumen; "lexa" no
se agrega como variante propia porque ya cae dentro de max_edit_distance=1 de
"nexa" (sustitución l↔n).

Falso positivo aceptado (DECISIÓN 2026-07-05, v1): "anexa" está a
edit-distance 1 de "nexa" (inserción de una 'a': a-n-e-x-a vs n-e-x-a), así
que dispara igual que "nexa". Es un falso positivo raro en habla
conversacional del hogar ("anexa el documento" no es un patrón común
hablado); se acepta para v1 y se pinnea con test en vez de ocultarlo (ver
test_textual_wake.py). Si el volumen de falsos positivos en producción lo
justifica, la mitigación futura es una excepción explícita para "anexa" —NO
bajar max_edit_distance a 0, que perdería la variante real "lexa".

Por el mismo mecanismo por-token, "next" (sin el "up") también matchea "nexa"
por sí solo (distancia 1, sustitución t↔a), pero es un FALSO POSITIVO
común: el STT ambient (Parakeet) emite inglés spurious con regularidad.
Mismo caso con "nena" (vocativo rioplatense muy común, sustitución x↔n) y
"nexo" (palabra española común, sustitución a↔o): ambas dispararían un
comando fantasma en conversación normal del hogar. Mitigación (v1): denylist
de palabras comunes —inglés o español— (`_FUZZY_DENYLIST`: "next", "nena",
"nexo") que son edit-distance ≤1 de un token base, excluidas SOLO del fuzzy
per-token. El bigram "next up" se evalúa en el pase exacto, que NO consulta
el denylist, así que sigue matcheando; los tokens exactos ("nexa" sin fuzzy)
tampoco consultan el denylist. Ver también test_textual_wake.py.

"alexa" NO matchea: edit-distance("nexa", "alexa") = 2 (dos inserciones),
por encima del default max_edit_distance=1 — verificado con test.

`normalize_text` duplica deliberadamente el algoritmo de
`src.nlu.command_gate._normalize` (lowercase, sin acentos, sin puntuación,
espacios colapsados): el módulo `ambient` no debe importar `src.nlu` (frontera
de arquitectura — el ambient path es upstream e independiente del stack NLU).
Mantener ambas implementaciones en sync si el algoritmo cambia.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable

import numpy as np

from src.pipeline.command_event import CommandEvent

logger = logging.getLogger(__name__)

_DEFAULT_VARIANTS: tuple[str, ...] = ("nexa", "next up")

# Palabras comunes (inglés o español) que están a edit-distance ≤1 de "nexa"
# pero son falsos positivos — excluidas del fuzzy per-token (las bigramas
# exactas como "next up" y los tokens exactos se siguen evaluando sin
# consultar esta lista):
# - "next": spurious del STT ambient Parakeet (emite inglés sobre far-field).
# - "nena": vocativo rioplatense extremadamente común ("nena, apagá la luz"
#   dicho a una persona, no al asistente) — sustitución x↔n de "nexa".
# - "nexo": palabra española común ("el nexo entre ambos") — sustitución
#   a↔o de "nexa".
_FUZZY_DENYLIST: frozenset[str] = frozenset({"next", "nena", "nexo"})


def normalize_text(text: str) -> str:
    """Normalizar texto: minúsculas, sin acentos ni puntuación, espacios colapsados.

    Duplicado deliberado de `src.nlu.command_gate._normalize` — ver docstring
    del módulo para la justificación de la frontera de arquitectura.

    Args:
        text: Texto crudo (con o sin acentos/puntuación).

    Returns:
        Texto normalizado: minúsculas, sin diacríticos, sin puntuación,
        espacios colapsados y sin bordes.
    """
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _within_edit_distance(a: str, b: str, max_dist: int) -> bool:
    """True si la distancia de edición (Levenshtein) entre a y b es <= max_dist.

    Early-exit por diferencia de largo: una edición individual (insert/delete)
    cambia el largo en a lo sumo 1, así que si `abs(len(a) - len(b)) > max_dist`
    ya no puede existir un camino de <= max_dist ediciones — evita correr la
    DP para la enorme mayoría de tokens de una utterance (que no se parecen
    en nada a "nexa").

    Args:
        a: Primer token (ya normalizado).
        b: Segundo token (ya normalizado).
        max_dist: Distancia máxima aceptada (inclusive).

    Returns:
        True si distance(a, b) <= max_dist.
    """
    if abs(len(a) - len(b)) > max_dist:
        return False
    if a == b:
        return True
    if max_dist <= 0:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1] <= max_dist


def matches_wake(
    text: str,
    variants: tuple[str, ...] = _DEFAULT_VARIANTS,
    max_edit_distance: int = 1,
) -> bool:
    """Detectar si el texto normalizado contiene una mención de wake "nexa".

    Match a nivel token: alguna variante de una sola palabra matchea exacto
    un token, alguna variante con espacio ("next up") matchea exacto un
    bigrama de tokens consecutivos, o algún token individual está a
    edit-distance <= max_edit_distance de alguna variante de una sola
    palabra (el "token base", p. ej. "nexa").

    Args:
        text: Texto crudo de la utterance (se normaliza internamente).
        variants: Variantes conocidas del wake word. Las que contienen un
            espacio se tratan como bigramas (match exacto únicamente); las
            de una sola palabra habilitan además el match fuzzy.
        max_edit_distance: Distancia de edición máxima para el match fuzzy
            de variantes de una sola palabra. 0 = solo match exacto.

    Returns:
        True si el texto contiene el wake word (exacto o fuzzy).
    """
    norm = normalize_text(text)
    if not norm:
        return False
    tokens = norm.split()
    if not tokens:
        return False

    norm_variants = tuple(v.strip().lower() for v in variants if v and v.strip())
    single_variants = [v for v in norm_variants if " " not in v]
    bigram_variants = [v for v in norm_variants if " " in v]

    if set(tokens) & set(single_variants):
        return True

    if bigram_variants and len(tokens) >= 2:
        bigrams = {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}
        if bigrams & set(bigram_variants):
            return True

    for token in tokens:
        if token in _FUZZY_DENYLIST:
            continue
        for base in single_variants:
            if _within_edit_distance(token, base, max_edit_distance):
                return True

    return False


class TextualWakeDetector:
    """Detecta "nexa" en utterances del ambient y despacha un CommandEvent.

    Lógica pura + dedup dual: se abstiene si el wake acústico ya disparó
    recientemente en la room (`last_acoustic_command_ts_fn`) o si el propio
    canal textual ya disparó hace menos de `dedup_window_s` en esa room
    (evita duplicar la ejecución de un comando cuando el ambient re-transcribe
    fragmentos solapados de la misma utterance).

    Fail-safe: una excepción de `dispatch_fn` se loguea (ERROR) y se traduce
    en `return False` — nunca se re-propaga; el worker del ambient no debe
    morir por esto.
    """

    def __init__(
        self,
        dispatch_fn: Callable[[CommandEvent], Awaitable[dict]],
        last_acoustic_command_ts_fn: Callable[[str], float],
        enabled: bool = True,
        dedup_window_s: float = 8.0,
        variants: tuple[str, ...] = _DEFAULT_VARIANTS,
        max_edit_distance: int = 1,
        now_fn: Callable[[], float] = time.monotonic,
    ):
        """Configurar el detector.

        Args:
            dispatch_fn: `process_command` del router (inyectado), recibe el
                CommandEvent construido y devuelve el dict de resultado.
            last_acoustic_command_ts_fn: Timestamp monotónico del último
                dispatch ACÚSTICO por room (0.0 = nunca), para el dedup
                cruzado con el wake acústico.
            enabled: Kill-switch del canal completo.
            dedup_window_s: Ventana de deduplicación (segundos) tanto contra
                el último dispatch acústico como contra el propio último
                dispatch textual, por room.
            variants: Variantes del wake word (ver `matches_wake`).
            max_edit_distance: Distancia de edición máxima para el match
                fuzzy (ver `matches_wake`).
            now_fn: Reloj inyectable (default `time.monotonic`) — permite
                FakeClock en tests, cero sleeps.
        """
        self._dispatch_fn = dispatch_fn
        self._last_acoustic_command_ts_fn = last_acoustic_command_ts_fn
        self._enabled = enabled
        self._dedup_window_s = dedup_window_s
        self._variants = variants
        self._max_edit_distance = max_edit_distance
        self._now_fn = now_fn
        # Último dispatch TEXTUAL propio por room (0.0 = nunca) — dedup contra
        # re-transcripciones solapadas de la misma utterance.
        self._last_dispatch_ts: dict[str, float] = {}

    async def maybe_dispatch(
        self,
        room_id: str,
        text: str,
        source: str,
        speaker: str | None,
        audio: np.ndarray,
    ) -> bool:
        """Evaluar una utterance y despachar un comando si corresponde.

        Reglas en orden (la primera que aplica decide):
        1. Canal deshabilitado -> no dispara.
        2. `source in {"tv", "self"}` -> no dispara (log INFO). "tv" = audio
           de un televisor de fondo; "self" = eco de la propia TTS del
           asistente reproduciéndose (`SourceClassifier` durante `during_tts`).
        3. Sin match del wake word -> no dispara (sin log, caso común).
        4. Wake acústico disparó hace menos de `dedup_window_s` en esta
           room -> no dispara (log INFO).
        5. Este mismo canal ya disparó hace menos de `dedup_window_s` en
           esta room -> no dispara (log INFO).
        6. Si no aplica ninguna de las anteriores: construye un
           `CommandEvent` con el texto ya transcripto y lo despacha.

        Args:
            room_id: Habitación de origen de la utterance.
            text: Texto ya transcripto por el ambient STT.
            source: Clasificación de la fuente ("tv", "self", "live", etc.) —
                "tv" y "self" se tratan especial acá (skip, ver Reglas).
            speaker: Hablante identificado, o None. Solo se usa para logging.
            audio: Audio del segmento (misma vista usada para el ASR
                ambient), se adjunta al CommandEvent sin copiar.

        Returns:
            True si se despachó un comando, False en cualquier otro caso
            (incluida una excepción de `dispatch_fn`, que se traga).
        """
        if not self._enabled:
            return False

        if source in {"tv", "self"}:
            decision = "source_tv" if source == "tv" else "source_self"
            logger.info(
                f"[TextualWake] skip room={room_id} source={source} "
                f"speaker={speaker} decision={decision} text={text!r}"
            )
            return False

        if not matches_wake(
            text, variants=self._variants, max_edit_distance=self._max_edit_distance
        ):
            return False

        now = self._now_fn()

        last_acoustic = self._last_acoustic_command_ts_fn(room_id)
        if last_acoustic > 0.0 and (now - last_acoustic) < self._dedup_window_s:
            logger.info(
                f"[TextualWake] skip room={room_id} source={source} "
                f"speaker={speaker} decision=dedup_acoustic text={text!r}"
            )
            return False

        last_self = self._last_dispatch_ts.get(room_id, 0.0)
        if last_self > 0.0 and (now - last_self) < self._dedup_window_s:
            logger.info(
                f"[TextualWake] skip room={room_id} source={source} "
                f"speaker={speaker} decision=dedup_self text={text!r}"
            )
            return False

        event = CommandEvent(audio=audio, room_id=room_id, wake_text=text, wake_score=1.0)
        try:
            await self._dispatch_fn(event)
        except Exception as e:
            logger.error(
                f"[TextualWake] dispatch error room={room_id} text={text!r}: {e}"
            )
            return False

        logger.info(
            f"[TextualWake] DISPARO room={room_id} text={text!r} "
            f"source={source} speaker={speaker}"
        )
        self._last_dispatch_ts[room_id] = now
        return True
