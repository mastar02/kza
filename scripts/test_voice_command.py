"""
End-to-end test: mic (ReSpeaker) -> Whisper STT -> Chroma search -> HA action.

Uso:
  python scripts/test_voice_command.py                 # loop interactivo, Enter para grabar
  python scripts/test_voice_command.py --duration 4    # grabá 4s (default: 5)
  python scripts/test_voice_command.py --device 9      # mic device index (default: 9)
  python scripts/test_voice_command.py --no-execute    # solo transcribe + search, no llama HA

Press-to-talk (sin wake word) para validación inicial del pipeline.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("voice_test")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Cargar .env
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

HA_URL = os.environ["HOME_ASSISTANT_URL"].rstrip("/")
HA_TOKEN = os.environ["HOME_ASSISTANT_TOKEN"]


# Regex de intent: verbos en español para turn_on/turn_off.
# Incluye tú (apaga/prende), vos (apagá/prendé), infinitivo (apagar/prender).
_RE_TURN_OFF = re.compile(r"\b(apag[aá]r?|cort[aá]r?|corte[a-z]*)\b", re.IGNORECASE)
_RE_TURN_ON = re.compile(r"\b(prend[eé]r?|encend[eé]r?|enciendo|ilumin[aá]r?|activ[aá]r?)\b", re.IGNORECASE)


def classify_intent(text: str) -> str | None:
    """Clasifica intent por léxico (antes del vector search). Retorna 'turn_on', 'turn_off', o None."""
    if _RE_TURN_OFF.search(text):
        return "turn_off"
    if _RE_TURN_ON.search(text):
        return "turn_on"
    return None


def call_ha_service(domain: str, service: str, entity_id: str) -> dict:
    """Invocar servicio HA vía REST. Retorna lista de estados afectados."""
    data = json.dumps({"entity_id": entity_id}).encode()
    req = urllib.request.Request(
        f"{HA_URL}/api/services/{domain}/{service}",
        data=data,
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=10).read()
    return json.loads(resp)


def record(duration: float, device: int, fs: int = 16000):
    """Grabar audio del ReSpeaker (6ch). Devuelve mono (canal 0: processed/beamformed)."""
    import numpy as np
    import sounddevice as sd
    logger.info(f"Grabando {duration}s en device {device}...")
    audio_6ch = sd.rec(int(duration * fs), samplerate=fs, channels=6, device=device, dtype="float32")
    sd.wait()
    # Canal 0 del ReSpeaker v2.0 6ch = beamformed + AEC processed (ASR-ready)
    mono = audio_6ch[:, 0]
    rms = float(np.sqrt(np.mean(mono**2)))
    logger.info(f"Grabación: {len(mono)} samples, RMS={rms:.4f}")
    if rms < 0.001:
        logger.warning("¡Audio casi en silencio! Verificá ganancia del mic.")
    return mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--device", type=int, default=9)
    ap.add_argument("--whisper-model", default=str(ROOT / "models/whisper-v3-turbo"))
    ap.add_argument("--chroma-path", default=str(ROOT / "data/chroma_db"))
    ap.add_argument("--threshold", type=float, default=0.55, help="Similitud mínima (0-1)")
    ap.add_argument("--no-execute", action="store_true", help="No llamar HA, solo transcribir+buscar")
    ap.add_argument("--once", action="store_true", help="Una sola iteración (no loop)")
    ap.add_argument("--embedder-device", default="cuda:0",
                    help="Device para BGE-M3. Default cuda:0 (cuda:1 está ocupado por vLLM)")
    args = ap.parse_args()

    # 1) Whisper (cuda:0)
    logger.info(f"Cargando Whisper desde {args.whisper_model} (cuda:0)...")
    from faster_whisper import WhisperModel
    stt = WhisperModel(args.whisper_model, device="cuda", device_index=0, compute_type="float16")
    logger.info("Whisper cargado")

    # 2) BGE-M3 embedder
    logger.info(f"Cargando BGE-M3 ({args.embedder_device})...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("BAAI/bge-m3", device=args.embedder_device)

    # 3) Chroma
    import chromadb
    client = chromadb.PersistentClient(path=args.chroma_path)
    col = client.get_collection("home_assistant_commands")
    count = col.count()
    logger.info(f"Chroma: {count} frases indexadas")
    if count == 0:
        logger.error("Chroma vacío. Corré primero: scripts/sync_ha_to_chroma.py --entity light.escritorio")
        return 1

    # 4) Loop
    while True:
        try:
            input("\n>>> Enter para grabar (Ctrl-C para salir)...")
        except (KeyboardInterrupt, EOFError):
            logger.info("Saliendo.")
            return 0

        t_total = time.time()

        # Record
        audio = record(args.duration, args.device)

        # STT
        t0 = time.time()
        segments, info = stt.transcribe(audio, language="es", vad_filter=True, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        t_stt = (time.time() - t0) * 1000
        logger.info(f"STT [{t_stt:.0f}ms]: {text!r}")

        if not text:
            logger.warning("Transcripción vacía. Reintentá.")
            if args.once: return 1
            continue

        # Clasificar intent por léxico (turn_on/turn_off) para filtrar en Chroma.
        # Dense embeddings no distinguen polaridad de verbo — el filtro léxico resuelve eso.
        intent = classify_intent(text)
        if intent:
            logger.info(f"Intent léxico: {intent}")

        # Vector search (con filtro de service si hay intent)
        t0 = time.time()
        q_emb = embedder.encode(text).tolist()
        where = {"service": intent} if intent else None
        res = col.query(
            query_embeddings=[q_emb],
            n_results=3,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        t_search = (time.time() - t0) * 1000

        if not res["ids"][0]:
            logger.warning(f"Sin resultados en Chroma [{t_search:.0f}ms]")
            if args.once: return 1
            continue

        # Log top-3 para visibilidad
        logger.info(f"Vector search [{t_search:.0f}ms] — top 3:")
        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i]
            sim = 1 - (dist / 2)
            meta = res["metadatas"][0][i]
            doc = res["documents"][0][i]
            logger.info(f"  [{i+1}] sim={sim:.3f} | {meta['entity_id']} {meta['service']} | {doc!r}")

        top_dist = res["distances"][0][0]
        top_sim = 1 - (top_dist / 2)
        top_meta = res["metadatas"][0][0]

        if top_sim < args.threshold:
            logger.warning(f"Similitud {top_sim:.3f} < threshold {args.threshold}. No ejecutando.")
            if args.once: return 1
            continue

        # Execute
        entity_id = top_meta["entity_id"]
        domain = top_meta["domain"]
        service = top_meta["service"]
        logger.info(f"MATCH: {domain}.{service}({entity_id})")

        if args.no_execute:
            logger.info("(--no-execute activo, no llamo HA)")
        else:
            t0 = time.time()
            try:
                call_ha_service(domain, service, entity_id)
                t_ha = (time.time() - t0) * 1000
                logger.info(f"HA OK [{t_ha:.0f}ms]: {service} sobre {entity_id}")
            except Exception as ex:
                logger.error(f"HA FAIL: {ex}")

        t_end = (time.time() - t_total) * 1000
        logger.info(f"Total end-to-end: {t_end:.0f}ms")

        if args.once:
            return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
