"""
WhisperWakeDetector — wake word basado en STT continuo.

Alternativa a OpenWakeWord cuando no tenemos modelos custom entrenados.
Usa el Whisper ya cargado en cuda:0 para transcribir cada utterance y
matchear la palabra de activación contra la transcripción.

Flujo por chunk (~80ms del multi_room_audio_loop):
  1. VAD (silero) clasifica chunk como speech/silence.
  2. Si in_speech → acumula audio en buffer interno.
  3. Si silence_end_ms de silencio post-speech → transcribe buffer con Whisper.
  4. Si la transcripción contiene alguna wake word → trigger.

Interfaz compatible con WakeWordDetector (detect/predict/get_active_models).
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_MS = 80  # tamaño esperado del chunk del caller

# Verbos/keywords que esperamos en un comando real post-wake. Si el texto
# matcheó la wake pero no tiene ninguno de estos → probablemente es TV
# hablando + Whisper transcribió algo que sonó como "Nexa". Rechazar.
_COMMAND_VERB_RE = re.compile(
    r"\b("
    r"prend\w*|encend\w*|apag\w*|enci\w*|"      # on/off (cubre prendé, apagá, encienden…)
    r"sub\w*|baj\w*|"                            # up/down/dimming (subí, bajá, suban…)
    r"pon\w*|cambi\w*|"                          # set/change
    r"abr\w*|cerr\w*|clos\w*|open\w*|"           # cover
    r"activ\w*|desactiv\w*|"                     # activate
    r"temperatura|brillo|volumen|"               # properties (sustantivos)
    r"luz al|luces al|al cincuent|al treint|"    # % dimming (frases)
    r"por ciento|maximo|minimo"
    r")\b", re.IGNORECASE,
)

# Frases típicas de TV/streaming que NUNCA son un comando real.
# Substring match en el texto normalizado.
_TV_STOP_PHRASES = (
    "suscribe", "suscrib",          # suscríbete
    "campanita",
    "gracias por ver",
    "dale like", "dale lie", "dale mega like",
    "canal de youtube",
    "activa la",                    # "activa la campanita"
)


def _normalize(text: str) -> str:
    """Lowercase + quitar acentos + colapsar espacios. Para match robusto."""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Prefijos coalescidos ↔ forma real. Whisper a veces pega la 'a' final
# de "nexa" al inicio del verbo siguiente: "nexa prendé" → "nexa aprende".
# Mapeo conservador — solo prefijos de verbos de domótica conocidos para
# evitar reinterpretar palabras legítimas.
_COALESCED_VERB_PREFIXES: tuple[tuple[str, str], ...] = (
    ("aprend", "prend"),    # nexa aprende → nexa prende
    ("aencend", "encend"),  # nexa aencendé → nexa encendé
    ("abaj", "baj"),        # nexa abajá → nexa bajá
    ("asub", "sub"),        # nexa asubí → nexa subí
    # Nota: "apag" NO se lista porque ya es verbo válido ("nexa apagá"),
    # y "abr" tampoco porque "abr" (abrí) es legítimo tras "nexa".
)


def _decoalesce_post_wake(norm_text: str, wake_norm: str) -> str:
    """Corregir el pegado del wake con el verbo siguiente.

    Whisper ocasionalmente produce 'nexa aprende' cuando el usuario dijo
    'nexa, prendé' — la 'a' final del wake se pega al inicio del verbo.
    Detectamos y re-segmentamos las combinaciones conocidas.

    Args:
        norm_text: texto ya pasado por `_normalize` (lowercase, sin acentos).
        wake_norm: wake word en forma normalizada (ej: 'nexa').

    Returns:
        Texto corregido, o el original si no había coalescing detectable.
    """
    if not norm_text or not wake_norm:
        return norm_text
    words = norm_text.split()
    if len(words) < 2 or words[0] != wake_norm:
        return norm_text
    second = words[1]
    for coalesced, real in _COALESCED_VERB_PREFIXES:
        if second.startswith(coalesced):
            words[1] = real + second[len(coalesced):]
            return " ".join(words)
    return norm_text


def _decoalesce_original_text(text: str, wake_norm: str) -> str:
    """Aplicar decoalesce al texto ORIGINAL (con acentos y puntuación).

    Complemento de `_decoalesce_post_wake`: ese retorna una versión
    normalizada. Esta versión hace el mismo fix sobre el texto original
    para poder propagarlo al router como `pretranscribed_text`, de modo
    que el NLU vea 'prendé' en vez de 'aprendé'.

    Args:
        text: texto tal cual lo devolvió Whisper.
        wake_norm: wake word normalizada (ej: 'nexa').

    Returns:
        Texto original con el verbo post-wake re-segmentado, o igual si
        no aplicaba ningún mapeo.
    """
    if not text or not wake_norm:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    first_norm = _normalize(words[0])
    if first_norm != wake_norm:
        return text
    second_norm = _normalize(words[1])
    for coalesced, real in _COALESCED_VERB_PREFIXES:
        if coalesced == real:
            continue
        if second_norm.startswith(coalesced):
            # Reemplazar el prefijo en la palabra original preservando el
            # sufijo (que puede tener acentos: 'aprendé' → 'prendé').
            words[1] = re.sub(
                r"^" + re.escape(coalesced),
                real,
                words[1],
                count=1,
                flags=re.IGNORECASE,
            )
            return " ".join(words)
    return text


def _phonetic_es(word: str) -> str:
    """
    Codifica una palabra a una representación fonética simplificada del
    español rioplatense. Hace "nexa" y "next" colapsar a /neks.../ mientras
    "nena" /nena/ queda bien separada.

    Transformaciones:
      - NFD + quitar acentos
      - v/w → b (mismo fonema bilabial)
      - h muda eliminada
      - qu/q → k
      - c antes de e/i/y → s (seseo rioplatense), resto → k
      - g antes de e/i → j
      - x → ks (explicita el cluster)
      - z → s (seseo)
      - ll → y, ñ → ny
      - colapsa consonantes repetidas

    La palabra clave "nexa" es especialmente discriminable porque /ks/ es raro
    en ataques o codas del español común. Todas las palabras que no tienen ese
    cluster van a quedar distantes en Levenshtein sobre esta representación.
    """
    t = unicodedata.normalize("NFD", word.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[vw]", "b", t)
    t = re.sub(r"h", "", t)
    t = re.sub(r"qu", "k", t)
    t = re.sub(r"q", "k", t)
    t = re.sub(r"c([eiy])", r"s\1", t)
    t = re.sub(r"c", "k", t)
    t = re.sub(r"g([ei])", r"j\1", t)
    t = re.sub(r"x", "ks", t)
    t = re.sub(r"z", "s", t)
    t = t.replace("ll", "y").replace("ñ", "ny")
    t = re.sub(r"([bdfgjklmnpqrstxy])\1+", r"\1", t)
    return t


class WhisperWakeDetector:
    """
    Wake word detector STT-based.

    Args:
        whisper_stt: instancia FastWhisperSTT ya cargada (compartida con el pipeline).
        wake_words: lista de palabras/frases a detectar (ej. ["nexa"]). Match insensible
            a mayúsculas/acentos.
        silence_end_ms: silencio consecutivo que cierra la utterance.
        min_utterance_ms: descarta utterances más cortas (ruido, click).
        max_utterance_s: fuerza transcripción si la utterance se alarga.
        pre_roll_ms: audio previo al inicio de VAD-speech (evita cortar la primera sílaba).
        vad_threshold: umbral de silero-vad (0-1). Más bajo = más permisivo.
        language: idioma para Whisper (ej "es").
    """

    def __init__(
        self,
        whisper_stt,
        wake_words: list[str],
        silence_end_ms: int = 500,
        min_utterance_ms: int = 250,
        max_utterance_s: float = 3.5,
        pre_roll_ms: int = 200,
        vad_threshold: float = 0.7,
        min_rms: float = 0.025,  # bajado de 0.04 para tolerar voz a distancia media
        require_start: bool = True,
        language: str = "es",
        speaker_identifier=None,
        speaker_embedding: Optional[np.ndarray] = None,
        speaker_threshold: float = 0.65,
        speaker_min_audio_s: float = 0.8,
        fuzzy_threshold: float = 0.75,
        fuzzy_start_words: int = 3,
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
        metrics_emitter=None,
        room_id: Optional[str] = None,
        follow_up_window_s: float = 4.0,
        follow_up_max_words: int = 3,
    ):
        """
        vad_threshold=0.7 (estricto) filtra voces lejanas/TV a volumen medio.
        min_rms=0.04 descarta chunks de baja energía (voz de TV atenuada).
        require_start=True exige que la wake word esté en las primeras 3 palabras de la
        utterance — un noticiero diciendo "nexa" en medio de una oración no triggerea.

        Speaker filter (opcional): si se pasan speaker_identifier + speaker_embedding,
        antes de llamar Whisper se compara la utterance completa contra el embedding de
        referencia. Si cosine_sim < speaker_threshold, la utterance se descarta
        silenciosamente (log DEBUG). Esto filtra TV/otras voces y evita transcripciones
        inútiles — reduce CPU/GPU significativamente en ambiente ruidoso.
        """
        self.whisper = whisper_stt
        self.wake_words_norm = [_normalize(w) for w in wake_words]
        self.wake_words_phon = [_phonetic_es(w) for w in self.wake_words_norm]
        self.silence_end_ms = silence_end_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_s = max_utterance_s
        self.pre_roll_ms = pre_roll_ms
        self.vad_threshold = vad_threshold
        self.min_rms = min_rms
        self.require_start = require_start
        self.language = language
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_start_words = fuzzy_start_words
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.metrics_emitter = metrics_emitter
        self.room_id = room_id or "unknown"

        self.speaker_identifier = speaker_identifier
        self.speaker_embedding = speaker_embedding
        self.speaker_threshold = speaker_threshold
        self.speaker_min_audio_s = speaker_min_audio_s
        self._speaker_filter_active = (
            speaker_identifier is not None and speaker_embedding is not None
        )

        # Follow-up window: cuando una utterance contiene solo el wake (≤ N
        # palabras) y no tiene verbo de comando, no rechazamos — armamos una
        # ventana donde la próxima utterance con verbo se trata como comando
        # implícito (sin requerir wake repetido). Resuelve el caso clásico
        # del usuario que dice "Nexa..." con pausa antes del comando.
        self.follow_up_window_s = follow_up_window_s
        self.follow_up_max_words = follow_up_max_words
        self._follow_up_until: float = 0.0

        self._vad = None
        self._loaded = False

        # State entre llamadas a detect()
        self._in_speech = False
        self._utterance: list[np.ndarray] = []
        self._pre_roll = deque(maxlen=max(1, pre_roll_ms // CHUNK_MS))
        self._silence_run_ms = 0
        self._utt_start_t = 0.0

        # Audio de la utterance que matcheó (post-wake-word recortado).
        # Si existe, multi_room_audio_loop lo usa como "command audio" en vez
        # de capturar audio nuevo (que probablemente ya pasó con el wake).
        self._pending_command_audio: Optional[np.ndarray] = None
        self._pending_text: Optional[str] = None

    def load(self):
        if self._loaded:
            return
        logger.info("WhisperWakeDetector: cargando silero-vad...")
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad = model
            self._torch = torch
        except Exception as e:
            logger.error(f"No pude cargar silero-vad: {e}. "
                         f"Fallback a VAD trivial por RMS (menos preciso).")
            self._vad = None
        self._loaded = True
        filter_str = (
            f" +speaker_filter(dim={self.speaker_embedding.shape[0]}, "
            f"threshold={self.speaker_threshold})"
            if self._speaker_filter_active else ""
        )
        logger.info(
            f"WhisperWakeDetector listo. Wake words: {self.wake_words_norm} "
            f"(engine=whisper+vad{filter_str})"
        )

    def _speaker_match(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Compara el audio de la utterance completa contra el embedding de referencia.

        Returns:
            (pass, similarity) — pass=True si la utterance debe procesarse.

        - Si no hay filter activo → (True, 1.0).
        - Si audio < speaker_min_audio_s → (True, 0.0) — embedding poco confiable
          con audio corto; dejamos pasar y filtramos por substring match.
        - Caso normal → (sim >= threshold, sim).
        """
        if not self._speaker_filter_active:
            return True, 1.0
        dur_s = len(audio) / SAMPLE_RATE
        if dur_s < self.speaker_min_audio_s:
            return True, 0.0
        try:
            emb = self.speaker_identifier.get_embedding(audio)
            sim = self.speaker_identifier.compute_similarity(
                emb, self.speaker_embedding,
            )
        except Exception as e:
            logger.warning(f"Speaker match falló ({e}); dejando pasar la utterance.")
            return True, 0.0
        passed = sim >= self.speaker_threshold
        return passed, sim

    def get_active_models(self) -> list[str]:
        return [f"whisper:{w}" for w in self.wake_words_norm]

    def _voice_prob(self, chunk: np.ndarray) -> float:
        """
        Devuelve probabilidad [0.0, 1.0] de que el chunk contenga voz humana.

        Doble gate:
          - RMS mínimo (filtra voz de TV lejana/atenuada) → 0.0 si debajo.
          - Silero-VAD (confirma que es voz humana) → probabilidad exacta.

        Preparado para endpointers adaptativos (ver S5): el caller puede usar
        esta señal continua para decidir thresholds dinámicos en vez del bool
        binario de `_is_speech`.
        """
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        prob = 0.0
        if rms >= self.min_rms and self._vad is not None:
            # Silero-VAD v4/v5 requiere exactamente 512 samples @16kHz
            # (chunks del pipeline son 1280 = 80ms → partir en 2-3 sub-chunks
            # y tomar el max prob).
            sub_size = 512
            audio = chunk.astype(np.float32)
            max_sub = 0.0
            try:
                for i in range(0, len(audio) - sub_size + 1, sub_size):
                    sub = audio[i:i + sub_size]
                    tensor = self._torch.from_numpy(sub)
                    with self._torch.no_grad():
                        p = float(self._vad(tensor, SAMPLE_RATE).item())
                    if p > max_sub:
                        max_sub = p
                prob = max_sub
            except Exception as e:
                if not hasattr(self, "_dbg_silero_err"):
                    logger.warning(f"Silero VAD error: {e} — fallback RMS-only")
                    self._dbg_silero_err = True
                prob = 0.5 if rms > self.min_rms else 0.0
        elif rms >= self.min_rms:
            prob = 0.5

        if not hasattr(self, "_dbg_max_rms"):
            self._dbg_max_rms = 0.0
            self._dbg_max_prob = 0.0
            self._dbg_last_t = time.time()
            self._dbg_count = 0
        self._dbg_max_rms = max(self._dbg_max_rms, rms)
        self._dbg_max_prob = max(self._dbg_max_prob, prob)
        self._dbg_count += 1
        now = time.time()
        if now - self._dbg_last_t >= 2.0:
            import math
            dbfs = 20 * math.log10(self._dbg_max_rms + 1e-9)
            logger.info(
                f"🎤 DBG[{self._dbg_count}ch/2s] max_rms={self._dbg_max_rms:.4f} "
                f"({dbfs:.1f}dBFS) max_prob={self._dbg_max_prob:.2f} "
                f"gates(rms>={self.min_rms} vad>={self.vad_threshold})"
            )
            self._dbg_max_rms = 0.0
            self._dbg_max_prob = 0.0
            self._dbg_last_t = now
            self._dbg_count = 0
        return prob

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """
        True si el chunk tiene voz suficientemente fuerte. Wrapper binario
        sobre `_voice_prob` usando `self.vad_threshold` como umbral.
        """
        return self._voice_prob(chunk) >= self.vad_threshold

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """Devuelve {palabra: score} con score 1.0 en trigger, 0.0 en resto."""
        result: dict[str, float] = {w: 0.0 for w in self.wake_words_norm}
        match = self._process_chunk(audio_chunk)
        if match is not None:
            result[match] = 1.0
        return result

    def detect(self, audio_chunk: np.ndarray) -> Optional[tuple[str, float]]:
        """Compatible con WakeWordDetector.detect."""
        match = self._process_chunk(audio_chunk)
        if match is not None:
            return (match, 1.0)
        return None

    def _process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        if not self._loaded:
            self.load()

        # Asumimos float32 normalizado [-1, 1]; si viene int16 o escala int16-float, normalizar
        chunk = audio_chunk
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        max_abs = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        if max_abs > 1.5:  # viene en escala int16
            chunk = chunk / 32768.0

        is_speech = self._is_speech(chunk)

        if not self._in_speech:
            self._pre_roll.append(chunk)
            if is_speech:
                self._in_speech = True
                self._utt_start_t = time.time()
                self._utterance = list(self._pre_roll)
                self._utterance.append(chunk)
                self._silence_run_ms = 0
            return None

        # In speech
        self._utterance.append(chunk)
        if is_speech:
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += CHUNK_MS

        utt_dur_s = time.time() - self._utt_start_t
        end_of_utterance = (
            self._silence_run_ms >= self.silence_end_ms
            or utt_dur_s >= self.max_utterance_s
        )
        if not end_of_utterance:
            return None

        # Close utterance
        audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._utterance = []
        self._pre_roll.clear()
        self._silence_run_ms = 0

        dur_ms = len(audio) * 1000 / SAMPLE_RATE
        if dur_ms < self.min_utterance_ms:
            return None

        if self._speaker_filter_active:
            passed, sim = self._speaker_match(audio)
            if not passed:
                logger.info(
                    f"Speaker filter REJECT (sim={sim:.3f} < {self.speaker_threshold}) "
                    f"— skip Whisper ({dur_ms:.0f}ms utterance)"
                )
                return None
            logger.info(f"Speaker filter PASS (sim={sim:.3f}) — procede a Whisper")

        match, text = self._transcribe_and_match(audio, dur_ms)
        if match and text:
            # Guardar texto completo para que el pipeline lo use como
            # pretranscribed_text (evita 2do Whisper que a veces alucina).
            self._pending_text = text
            # Recortar el audio desde después del wake word (aprox)
            # El wake word "nexa" pronunciado ocupa ~400-500ms. Asumimos offset conservador.
            # Si el wake fue primera palabra, restamos ~500ms. Si fue en medio, el pipeline
            # STT re-transcribirá el audio completo y el intent classifier lo resolverá igual.
            norm = _normalize(text)
            words = norm.split()
            wake_idx = next((i for i, w in enumerate(words) if match in w), 0)
            # Offset en samples: aprox 400ms por palabra pre-wake + 500ms del wake mismo
            offset_s = (wake_idx + 1) * 0.4
            offset_samples = int(offset_s * SAMPLE_RATE)
            if offset_samples < len(audio):
                self._pending_command_audio = audio[offset_samples:].copy()
            else:
                # Wake al final sin comando inline → mantener todo el audio como backup
                self._pending_command_audio = audio.copy()
        return match

    def pop_pending_text(self) -> Optional[str]:
        """Consumir el texto completo transcrito por el wake detector.

        El caller (MultiRoomAudioLoop) lo usa como `pretranscribed_text` en el
        CommandEvent para evitar que un segundo Whisper re-transcriba el mismo
        audio (que a veces alucina).
        """
        t = self._pending_text
        self._pending_text = None
        return t

    def pop_pending_command_audio(self) -> Optional[np.ndarray]:
        """Consumir el audio recortado de la utterance del wake. El caller lo pasa al STT."""
        audio = self._pending_command_audio
        self._pending_command_audio = None
        return audio

    def _transcribe_and_match(self, audio: np.ndarray, dur_ms: float) -> tuple[Optional[str], str]:
        """Retorna (wake_word_matched | None, texto_completo_transcripto)."""
        t0 = time.time()
        try:
            model = getattr(self.whisper, "_model", None) or self.whisper
            segments, _ = model.transcribe(
                audio, language=self.language,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt,
                vad_filter=False,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as e:
            logger.error(f"WhisperWake transcribe error: {e}")
            return (None, "")
        stt_ms = (time.time() - t0) * 1000
        if not text:
            return (None, "")

        norm = _normalize(text)
        # Fix coalescing Whisper: 'nexa aprende' → 'nexa prende' antes de
        # aplicar las reglas de TV stop / command verb.
        # Aplicamos el mismo fix al `text` original para que el pending_text
        # que va al router/NLU también esté corregido (el NLU lee el texto
        # original con acentos, no el norm).
        for wake_norm in self.wake_words_norm:
            norm_fixed = _decoalesce_post_wake(norm, wake_norm)
            if norm_fixed != norm:
                text_fixed = _decoalesce_original_text(text, wake_norm)
                logger.info(
                    f"Decoalesced post-wake: {norm!r} → {norm_fixed!r} "
                    f"(text: {text!r} → {text_fixed!r})"
                )
                norm = norm_fixed
                text = text_fixed
                break
        logger.info(f"WhisperWake [{dur_ms:.0f}ms→{stt_ms:.0f}ms]: {norm!r}")

        # Paso 0: si estamos dentro de una ventana de follow-up post-wake-solo,
        # aceptar comandos sin wake explícito. El usuario dijo "Nexa..." y
        # ahora completa con el comando real.
        now = time.time()
        if now < self._follow_up_until:
            has_wake = any(w in norm for w in self.wake_words_norm)
            has_verb = bool(_COMMAND_VERB_RE.search(norm))
            is_tv = any(p in norm for p in _TV_STOP_PHRASES)
            if has_verb and not has_wake and not is_tv:
                self._follow_up_until = 0.0  # consumir
                canonical = self.wake_words_norm[0]
                synthesized = f"{canonical} {text}"
                logger.info(
                    f"🔥 Follow-up command capturado (ventana {self.follow_up_window_s}s): "
                    f"{text!r} → {synthesized!r}"
                )
                self._emit_wake(
                    True, canonical, "follow_up", synthesized, dur_ms, stt_ms,
                )
                return (canonical, synthesized)

        # Paso 1: substring match contra los aliases configurados (exact match en norm).
        # TV stop-words: frases que jamás son comandos legítimos.
        for phrase in _TV_STOP_PHRASES:
            if phrase in norm:
                logger.info(f"Wake rejected — TV stop phrase {phrase!r} en: {text!r}")
                self._emit_wake(
                    False, None, "rejected", text, dur_ms, stt_ms,
                    rejection_reason="tv_stop_phrase",
                )
                return (None, text)

        for w in self.wake_words_norm:
            if w in norm:
                if self.require_start:
                    first_words = " ".join(norm.split()[:self.fuzzy_start_words])
                    if w not in first_words:
                        logger.debug(f"Wake word '{w}' encontrada pero no al inicio — skip")
                        continue
                # Exigir verbo de comando post-wake: filtra exact matches
                # de "Nexa" cuando la TV dice algo que Whisper oyó como Nexa
                # pero el contenido siguiente no es un comando domótica.
                if not _COMMAND_VERB_RE.search(norm):
                    self._maybe_arm_follow_up(norm, text)
                    logger.info(
                        f"Wake rejected — sin verbo de comando post-wake: {text!r}"
                    )
                    self._emit_wake(
                        False, w, "rejected", text, dur_ms, stt_ms,
                        rejection_reason="no_command_verb",
                    )
                    return (None, text)
                logger.info(f"🔥 Wake word '{w}' detectado en: {text!r}")
                self._emit_wake(True, w, "exact", text, dur_ms, stt_ms)
                return (w, text)
        # Paso 2: fuzzy match fonético (Levenshtein sobre codificación española).
        # "nexa" /neksa/ vs "next" /nekst/ ≈ 0.80 (match); vs "nena" /nena/ ≈ 0.67
        # (rechazado). El cluster /ks/ separa los verdaderos positivos de FPs.
        best_ratio = 0.0
        if self.wake_words_phon and self.fuzzy_threshold > 0:
            words = norm.split()[:self.fuzzy_start_words]
            best_word = ""
            best_phon = ""
            best_target = ""
            for word in words:
                if len(word) < 3:
                    continue
                word_phon = _phonetic_es(word)
                for wake_phon in self.wake_words_phon:
                    r = SequenceMatcher(None, word_phon, wake_phon).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_word = word
                        best_phon = word_phon
                        best_target = wake_phon
            if best_ratio >= self.fuzzy_threshold:
                # Fuzzy match con threshold alto todavía puede agarrar TV
                # random. Requerir verbo de comando para ejecutar.
                if not _COMMAND_VERB_RE.search(norm):
                    self._maybe_arm_follow_up(norm, text)
                    logger.info(
                        f"Fuzzy wake rejected — sin verbo de comando: {text!r} "
                        f"(best {best_word!r} ratio={best_ratio:.2f})"
                    )
                    self._emit_wake(
                        False, None, "rejected", text, dur_ms, stt_ms,
                        fuzzy_ratio=best_ratio,
                        rejection_reason="no_command_verb",
                    )
                    return (None, text)
                canonical = self.wake_words_norm[0]
                logger.info(
                    f"🔥 Wake word fuzzy match: '{best_word}' /{best_phon}/ ~ "
                    f"/{best_target}/ (ratio={best_ratio:.2f}) en: {text!r}"
                )
                self._emit_wake(
                    True, canonical, "fuzzy", text, dur_ms, stt_ms,
                    fuzzy_ratio=best_ratio,
                )
                return (canonical, text)
        self._emit_wake(
            False, None, "rejected", text, dur_ms, stt_ms,
            fuzzy_ratio=best_ratio if best_ratio > 0 else None,
            rejection_reason="below_fuzzy_threshold" if best_ratio > 0 else "no_keyword",
        )
        return (None, text)

    def _maybe_arm_follow_up(self, norm: str, text: str) -> None:
        """Si la utterance es solo wake (≤ N palabras) y no tiene verbo,
        armar la ventana de follow-up para aceptar el comando que viene.

        No arma cuando la utterance es larga (>N palabras): eso suele ser
        TV o user diciendo algo no-comando ("Nexa, ¿dónde estás?"); abrir
        la ventana en esos casos invitaría falsos positivos.
        """
        if self.follow_up_window_s <= 0:
            return
        word_count = len(norm.split())
        if word_count > self.follow_up_max_words:
            return
        self._follow_up_until = time.time() + self.follow_up_window_s
        logger.info(
            f"🎤 Follow-up armed por {self.follow_up_window_s}s — "
            f"esperando comando tras wake-only: {text!r}"
        )

    def _emit_wake(
        self,
        matched: bool,
        wake_word: Optional[str],
        matched_via: str,
        text: str,
        dur_ms: float,
        stt_ms: float,
        fuzzy_ratio: Optional[float] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        if not self.metrics_emitter:
            return
        try:
            self.metrics_emitter.emit_wake(
                room_id=self.room_id,
                matched=matched,
                wake_word=wake_word,
                matched_via=matched_via,
                text=text,
                audio_duration_ms=dur_ms,
                wake_stt_ms=stt_ms,
                fuzzy_ratio=fuzzy_ratio,
                rejection_reason=rejection_reason,
            )
        except Exception as e:
            logger.warning(f"MetricsEmitter emit_wake failed: {e}")
