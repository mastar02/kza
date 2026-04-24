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


def _normalize(text: str) -> str:
    """Lowercase + quitar acentos + colapsar espacios. Para match robusto."""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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

        self.speaker_identifier = speaker_identifier
        self.speaker_embedding = speaker_embedding
        self.speaker_threshold = speaker_threshold
        self.speaker_min_audio_s = speaker_min_audio_s
        self._speaker_filter_active = (
            speaker_identifier is not None and speaker_embedding is not None
        )

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
        if rms < self.min_rms:
            return 0.0
        if self._vad is not None:
            try:
                tensor = self._torch.from_numpy(chunk.astype(np.float32))
                with self._torch.no_grad():
                    return float(self._vad(tensor, SAMPLE_RATE).item())
            except Exception:
                pass
        # Fallback si VAD falla: señal media si hay energía suficiente.
        return 0.5 if rms > self.min_rms else 0.0

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
        logger.info(f"WhisperWake [{dur_ms:.0f}ms→{stt_ms:.0f}ms]: {norm!r}")
        # Paso 1: substring match contra los aliases configurados (exact match en norm).
        for w in self.wake_words_norm:
            if w in norm:
                if self.require_start:
                    first_words = " ".join(norm.split()[:self.fuzzy_start_words])
                    if w not in first_words:
                        logger.debug(f"Wake word '{w}' encontrada pero no al inicio — skip")
                        continue
                logger.info(f"🔥 Wake word '{w}' detectado en: {text!r}")
                return (w, text)
        # Paso 2: fuzzy match fonético (Levenshtein sobre codificación española).
        # "nexa" /neksa/ vs "next" /nekst/ ≈ 0.80 (match); vs "nena" /nena/ ≈ 0.67
        # (rechazado). El cluster /ks/ separa los verdaderos positivos de FPs.
        if self.wake_words_phon and self.fuzzy_threshold > 0:
            words = norm.split()[:self.fuzzy_start_words]
            best_ratio = 0.0
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
                canonical = self.wake_words_norm[0]
                logger.info(
                    f"🔥 Wake word fuzzy match: '{best_word}' /{best_phon}/ ~ "
                    f"/{best_target}/ (ratio={best_ratio:.2f}) en: {text!r}"
                )
                return (canonical, text)
        return (None, text)
