"""
Entrenamiento de wake word custom (Fase 3 del roadmap).

Pipeline:
  1. Generar ~10k positives sintéticos con piper-sample-generator (voces españolas).
  2. Combinar con 50 reales grabadas (`data/wakeword_training/nexa/positive/`).
  3. Augmentar con audiomentations (pitch shift, time stretch, noise, reverb, EQ).
  4. Negatives = 80 hard negatives + subset de MUSAN (speech/music/noise).
  5. Extraer mel features + Google speech embeddings (via openwakeword helpers).
  6. Entrenar head classification (Keras, CNN chico).
  7. Exportar a ONNX → `models/wakeword/nexa.onnx`.

Pre-requisitos (ver scripts/setup_custom_wake_deps.sh):
  - piper-sample-generator + piper-tts
  - tensorflow == 2.x
  - audiomentations
  - openwakeword >= 0.6
  - MUSAN dataset en /home/kza/data/musan/

Uso:
    python -m scripts.train_custom_wake generate      # 1+2
    python -m scripts.train_custom_wake features      # 5
    python -m scripts.train_custom_wake train         # 6
    python -m scripts.train_custom_wake export        # 7
    python -m scripts.train_custom_wake all           # full pipeline

NOTA: este script asume correr en el server con cuda:0 libre.
Pausar kza-voice.service antes:
    systemctl --user stop kza-voice.service
    python -m scripts.train_custom_wake all
    systemctl --user start kza-voice.service
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_custom_wake")

WAKE_WORD = "nexa"
SAMPLE_RATE = 16000
CLIP_DURATION_S = 1.5

POSITIVE_REAL_DIR = ROOT / "data" / "wakeword_training" / WAKE_WORD / "positive"
NEGATIVE_REAL_DIR = ROOT / "data" / "wakeword_training" / WAKE_WORD / "negative"

OUT_ROOT = ROOT / "data" / "wakeword_custom"
SYNTHETIC_DIR = OUT_ROOT / "synthetic_positives"
AUGMENTED_DIR = OUT_ROOT / "augmented_positives"
NEGATIVES_DIR = OUT_ROOT / "negatives"
FEATURES_DIR = OUT_ROOT / "features"
MODEL_DIR = ROOT / "models" / "wakeword"

MUSAN_DEFAULT = Path("/home/kza/data/musan")


def check_deps() -> bool:
    missing = []
    for pkg in ("piper", "tensorflow", "audiomentations", "openwakeword", "librosa"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Dependencias faltantes: %s", ", ".join(missing))
        logger.error("Instalar con: bash scripts/setup_custom_wake_deps.sh")
        return False
    return True


def cmd_generate(args) -> int:
    """1 + 2: generar positives sintéticos con piper + combinar con reales."""
    if not check_deps():
        return 1

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Piper synthesis ----
    logger.info("Generando %d positives sintéticos con piper-sample-generator...",
                args.n_synthetic)
    from piper_sample_generator import generate_samples  # lazy import

    generate_samples(
        text=WAKE_WORD,
        max_samples=args.n_synthetic,
        output_dir=str(SYNTHETIC_DIR),
        model=args.piper_model,
        batch_size=args.piper_batch,
        noise_scales=[0.333, 0.500, 0.667],
        length_scales=[0.9, 1.0, 1.1, 1.25],
    )
    logger.info("Piper listo: %d WAVs en %s",
                len(list(SYNTHETIC_DIR.glob("*.wav"))), SYNTHETIC_DIR)

    # ---- 2. Augmentation ----
    logger.info("Augmentando reales + sintéticos con audiomentations...")
    from audiomentations import (
        Compose, AddGaussianNoise, PitchShift, TimeStretch,
        Gain, AddBackgroundNoise,
    )
    import soundfile as sf
    import librosa

    augment = Compose([
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    ])

    # Agregar background noise (TV/music/speech de MUSAN) como augment extra
    musan_path = Path(args.musan) if args.musan else MUSAN_DEFAULT
    noise_augment = None
    if musan_path.exists():
        noise_augment = Compose([
            AddBackgroundNoise(
                sounds_path=str(musan_path),
                min_snr_in_db=3, max_snr_in_db=30,
                p=0.8,
            ),
        ])
        logger.info("Background noise augmentation activa: %s", musan_path)
    else:
        logger.warning("MUSAN no encontrado en %s; skip background noise", musan_path)

    def _augment_and_save(src: Path, dest_prefix: str, n_variants: int = 3):
        try:
            audio, sr = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            logger.debug("No pude leer %s: %s", src, e)
            return
        for i in range(n_variants):
            out = augment(samples=audio.copy(), sample_rate=SAMPLE_RATE)
            if noise_augment:
                out = noise_augment(samples=out, sample_rate=SAMPLE_RATE)
            # Pad o trim a CLIP_DURATION_S exacto
            target = int(CLIP_DURATION_S * SAMPLE_RATE)
            if len(out) > target:
                out = out[:target]
            elif len(out) < target:
                out = np.pad(out, (0, target - len(out)))
            path = AUGMENTED_DIR / f"{dest_prefix}_{i}.wav"
            sf.write(str(path), out, SAMPLE_RATE)

    logger.info("Augmentando %d reales...", len(list(POSITIVE_REAL_DIR.glob("*.wav"))))
    for i, w in enumerate(sorted(POSITIVE_REAL_DIR.glob("*.wav"))):
        _augment_and_save(w, f"real_{i:04d}", n_variants=args.augment_real)

    logger.info("Augmentando %d sintéticos...",
                len(list(SYNTHETIC_DIR.glob("*.wav"))))
    synth_files = sorted(SYNTHETIC_DIR.glob("*.wav"))
    for i, w in enumerate(synth_files):
        _augment_and_save(w, f"synth_{i:05d}", n_variants=args.augment_synth)

    logger.info("Total augmented positives: %d",
                len(list(AUGMENTED_DIR.glob("*.wav"))))

    # ---- 3. Negatives ----
    logger.info("Preparando negatives (hard negatives reales + MUSAN)...")
    for i, w in enumerate(sorted(NEGATIVE_REAL_DIR.glob("*.wav"))):
        shutil.copy(w, NEGATIVES_DIR / f"hard_{i:04d}.wav")

    if musan_path.exists():
        musan_speech = list((musan_path / "speech").rglob("*.wav"))
        musan_music = list((musan_path / "music").rglob("*.wav"))
        musan_noise = list((musan_path / "noise").rglob("*.wav"))
        logger.info("MUSAN: %d speech + %d music + %d noise",
                    len(musan_speech), len(musan_music), len(musan_noise))

        n_musan_samples = min(args.n_negatives_musan,
                              len(musan_speech) + len(musan_music) + len(musan_noise))
        rng = random.Random(42)
        pool = musan_speech + musan_music + musan_noise
        rng.shuffle(pool)
        for i, src in enumerate(pool[:n_musan_samples]):
            try:
                audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True,
                                         duration=CLIP_DURATION_S)
            except Exception:
                continue
            target = int(CLIP_DURATION_S * SAMPLE_RATE)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            out = NEGATIVES_DIR / f"musan_{i:05d}.wav"
            sf.write(str(out), audio[:target], SAMPLE_RATE)
        logger.info("Negatives MUSAN: %d agregados", n_musan_samples)

    logger.info("Total negatives: %d", len(list(NEGATIVES_DIR.glob("*.wav"))))
    return 0


def cmd_features(args) -> int:
    """Extraer mel features + Google speech embeddings (openwakeword)."""
    if not check_deps():
        return 1
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    import openwakeword.utils as oww_utils

    audio_features = oww_utils.AudioFeatures()

    def _extract_dir(wav_dir: Path, label: int, out_path: Path) -> int:
        feats = []
        files = sorted(wav_dir.glob("*.wav"))
        logger.info("Extrayendo features: %s (%d files, label=%d)",
                    wav_dir.name, len(files), label)
        import soundfile as sf
        for i, f in enumerate(files):
            try:
                audio, sr = sf.read(str(f))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio.astype(np.float32),
                                             orig_sr=sr, target_sr=SAMPLE_RATE)
                # openwakeword AudioFeatures espera int16
                audio_i16 = (audio * 32768.0).astype(np.int16)
                features = audio_features.embed_clip(audio_i16)
                feats.append(features)
            except Exception as e:
                logger.debug("Skip %s: %s", f.name, e)
            if (i + 1) % 500 == 0:
                logger.info("  %d/%d", i + 1, len(files))
        if not feats:
            return 0
        X = np.array(feats, dtype=np.float32)
        y = np.full(len(X), label, dtype=np.int32)
        np.savez(str(out_path), X=X, y=y)
        logger.info("  Guardado: %s  shape=%s", out_path, X.shape)
        return len(X)

    n_pos = _extract_dir(AUGMENTED_DIR, label=1,
                          out_path=FEATURES_DIR / "positives.npz")
    n_neg = _extract_dir(NEGATIVES_DIR, label=0,
                          out_path=FEATURES_DIR / "negatives.npz")
    logger.info("Features: %d positives, %d negatives", n_pos, n_neg)
    return 0


def cmd_train(args) -> int:
    """Entrenar CNN head de clasificación sobre openwakeword features."""
    if not check_deps():
        return 1

    import tensorflow as tf
    from tensorflow.keras import layers, models

    pos = np.load(FEATURES_DIR / "positives.npz")
    neg = np.load(FEATURES_DIR / "negatives.npz")
    X = np.concatenate([pos["X"], neg["X"]], axis=0)
    y = np.concatenate([pos["y"], neg["y"]], axis=0)
    logger.info("Dataset: X=%s y=%s (pos=%d, neg=%d)",
                X.shape, y.shape, int(y.sum()), int((y == 0).sum()))

    # Shuffle + split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Model: CNN simple
    input_shape = X.shape[1:]
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="prec"),
                 tf.keras.metrics.Recall(name="rec")],
    )
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_prec", mode="max",
                                         patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    saved = MODEL_DIR / f"{WAKE_WORD}_keras"
    model.save(saved)
    logger.info("Modelo Keras guardado: %s", saved)
    return 0


def cmd_export(args) -> int:
    """Exportar el modelo Keras a ONNX."""
    import tf2onnx
    from tensorflow.keras.models import load_model

    saved = MODEL_DIR / f"{WAKE_WORD}_keras"
    model = load_model(saved)
    onnx_path = MODEL_DIR / f"{WAKE_WORD}.onnx"
    input_signature = [
        # openwakeword expects (batch, time_steps, feat_dim)
        ...
    ]
    # Simplificado: usar spec directo
    import tensorflow as tf
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec,
                                                opset=13,
                                                output_path=str(onnx_path))
    logger.info("ONNX exportado: %s", onnx_path)
    meta = MODEL_DIR / f"{WAKE_WORD}.json"
    meta.write_text(json.dumps({
        "wake_word": WAKE_WORD,
        "sample_rate": SAMPLE_RATE,
        "clip_duration_s": CLIP_DURATION_S,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    logger.info("Listo. Config: rooms.wake_word.engine=openwakeword, model=%s",
                WAKE_WORD)
    return 0


def cmd_all(args) -> int:
    for step in (cmd_generate, cmd_features, cmd_train, cmd_export):
        logger.info("=" * 60)
        logger.info("STEP: %s", step.__name__)
        logger.info("=" * 60)
        rc = step(args)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="Generar positivos+negativos")
    p.add_argument("--n-synthetic", type=int, default=10000)
    p.add_argument("--piper-model", default="es_ES-sharvard-medium")
    p.add_argument("--piper-batch", type=int, default=32)
    p.add_argument("--augment-real", type=int, default=20,
                   help="Variantes por real (50 reales × 20 = 1000)")
    p.add_argument("--augment-synth", type=int, default=2,
                   help="Variantes por synth (10k × 2 = 20k)")
    p.add_argument("--n-negatives-musan", type=int, default=5000)
    p.add_argument("--musan", default=None, help="Path a MUSAN (default /home/kza/data/musan)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("features", help="Extraer features con openwakeword")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="Entrenar CNN")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="Exportar ONNX")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("all", help="Pipeline completo generate→features→train→export")
    p.add_argument("--n-synthetic", type=int, default=10000)
    p.add_argument("--piper-model", default="es_ES-sharvard-medium")
    p.add_argument("--piper-batch", type=int, default=32)
    p.add_argument("--augment-real", type=int, default=20)
    p.add_argument("--augment-synth", type=int, default=2)
    p.add_argument("--n-negatives-musan", type=int, default=5000)
    p.add_argument("--musan", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
