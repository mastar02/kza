"""
Enrollment de voz — genera el embedding ECAPA de referencia para speaker filter.

Lee WAVs de un directorio (por default `data/wakeword_training/nexa/positive/`),
extrae embeddings con SpeakerIdentifier, los promedia L2-normalized y guarda el
resultado en `data/users/{user_id}_voice.npy` + metadata .json.

El embedding se usa por el WhisperWakeDetector para filtrar voz del TV/otros
antes de llamar Whisper. Ver docs/superpowers/plans/2026-04-23-wake-word-roadmap.md
(Fase 1).

Uso:
    python -m scripts.enroll_voice --user gabriel
    python -m scripts.enroll_voice --user gabriel --dir data/wakeword_training/nexa/positive
    python -m scripts.enroll_voice --user gabriel --threshold-report  # sólo métricas, no guarda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import wave
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.users.speaker_identifier import SpeakerIdentifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("enroll_voice")

SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray | None:
    """Devuelve audio float32 mono @ 16kHz; None si el archivo es inválido."""
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getframerate() != SAMPLE_RATE:
                logger.warning("%s: sample_rate=%d (esperado %d), skip",
                               path.name, wf.getframerate(), SAMPLE_RATE)
                return None
            n = wf.getnframes()
            raw = wf.readframes(n)
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
    except (wave.Error, EOFError) as e:
        logger.warning("%s: WAV inválido (%s), skip", path.name, e)
        return None

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        logger.warning("%s: sampwidth=%d no soportado, skip", path.name, sampwidth)
        return None

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


def intra_user_consistency(embeddings: list[np.ndarray]) -> float:
    """Promedio de cosine similarity pairwise (excluye diagonal)."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for a, b in combinations(embeddings, 2):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        sims.append(float(np.dot(a, b) / (na * nb)))
    return float(np.mean(sims)) if sims else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="gabriel",
                        help="user_id (default: gabriel)")
    parser.add_argument("--dir", default="data/wakeword_training/nexa/positive",
                        help="Directorio con WAVs del usuario")
    parser.add_argument("--output-dir", default="data/users",
                        help="Directorio destino del embedding (default: data/users)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="Duración mínima por sample (seg) — ECAPA degrada <1s")
    parser.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb",
                        help="Modelo ECAPA (HuggingFace ID)")
    parser.add_argument("--device", default="cuda:0",
                        help="cuda:0 | cuda:1 | cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sólo mostrar stats, no guardar embedding")
    args = parser.parse_args()

    wav_dir = (ROOT / args.dir).resolve()
    if not wav_dir.is_dir():
        logger.error("No existe el directorio: %s", wav_dir)
        return 1

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        logger.error("Sin WAVs en %s", wav_dir)
        return 1
    logger.info("Encontrados %d WAVs en %s", len(wavs), wav_dir)

    min_samples = int(args.min_duration * SAMPLE_RATE)
    audios: list[np.ndarray] = []
    skipped_short = 0
    for w in wavs:
        audio = load_wav(w)
        if audio is None:
            continue
        if len(audio) < min_samples:
            skipped_short += 1
            continue
        audios.append(audio)
    if not audios:
        logger.error("Ningún audio válido después de filtros (min_duration=%.1fs)",
                     args.min_duration)
        return 1
    logger.info("Audios válidos: %d (descartados por cortos: %d)",
                len(audios), skipped_short)

    logger.info("Cargando SpeakerIdentifier (%s @ %s)...", args.model, args.device)
    t0 = time.time()
    spk = SpeakerIdentifier(model_name=args.model, device=args.device)
    spk.load()
    logger.info("Loaded en %.1fs", time.time() - t0)

    logger.info("Extrayendo embeddings...")
    t0 = time.time()
    embeddings = []
    for i, audio in enumerate(audios):
        emb = spk.get_embedding(audio)
        embeddings.append(emb)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d embeddings (%.2fs acumulado)",
                        i + 1, len(audios), time.time() - t0)
    logger.info("Embeddings extraídos en %.1fs", time.time() - t0)

    dim = embeddings[0].shape[0]
    logger.info("Embedding dim = %d", dim)

    consistency = intra_user_consistency(embeddings)
    logger.info("Consistencia intra-user (mean pairwise cos sim): %.3f", consistency)
    if consistency < 0.70:
        logger.warning("Consistencia baja (<0.70) — verifique que las muestras sean "
                       "del mismo speaker o considere re-grabar.")

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    logger.info("Embedding promedio L2-normalizado (shape=%s, norm=%.4f)",
                mean_emb.shape, float(np.linalg.norm(mean_emb)))

    if args.dry_run:
        logger.info("dry-run — no se guarda nada.")
        return 0

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{args.user}_voice.npy"
    meta_path = out_dir / f"{args.user}_voice.json"

    np.save(emb_path, mean_emb.astype(np.float32))
    metadata = {
        "user_id": args.user,
        "model": args.model,
        "device": args.device,
        "embedding_dim": int(dim),
        "n_samples": len(embeddings),
        "n_skipped_short": skipped_short,
        "min_duration_s": args.min_duration,
        "intra_consistency": round(consistency, 4),
        "source_dir": str(wav_dir.relative_to(ROOT)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Guardado: %s", emb_path)
    logger.info("Metadata: %s", meta_path)
    logger.info("Listo. Usar en config con rooms.wake_word.speaker_filter.embedding_path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
