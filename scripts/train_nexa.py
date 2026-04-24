"""
Sesión de training para wake word 'nexa'.

Nexa elegida por estructura acústica óptima:
  - Cluster /ks/ en medio → peak de energía alto en espectrograma
  - Vocales /e/+/a/ contrastantes (frontal vs central)
  - Pronunciación consistente (sin /r/ vibrante que se "traga" al acelerar)
  - Zero false positives en español cotidiano

Negativas incluyen "alexa", "exa" sola, fragmentos /ks/ (taxi, examen) para
forzar que el modelo requiera la firma completa y no dispare con parciales.

Usa el ReSpeaker XVF-3000 (device 9, 6 canales, tomamos canal 0 procesado).

Comandos:
  record-positive   — graba positivas
  record-negative   — graba negativas (incluye hard negatives acústicos)
  train             — entrena el modelo ONNX
  deploy            — copia .onnx al path de producción y actualiza settings
  test              — loop de detección en vivo con threshold configurable
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _voice_service_running() -> bool:
    """Check whether kza-voice.service is currently active."""
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "kza-voice.service"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "active"


def _stop_voice_service() -> bool:
    """Stop kza-voice.service so the mic (device 9) is free. Returns True if it was running."""
    if not _voice_service_running():
        return False
    print("   [kza-voice] Parando kza-voice.service para liberar el mic...")
    subprocess.run(["systemctl", "--user", "stop", "kza-voice.service"], check=True)
    time.sleep(0.5)
    print("   [kza-voice] Servicio parado. El mic está libre.")
    return True


def _start_voice_service() -> None:
    """Restart kza-voice.service after recording is done."""
    print("   [kza-voice] Reiniciando kza-voice.service...")
    subprocess.run(["systemctl", "--user", "start", "kza-voice.service"], check=True)
    print("   [kza-voice] Servicio reiniciado.")

WAKE_WORD_NAME = "nexa"
# Wake word único: "nexa" → el modelo aprende la firma acústica exacta.
# Variá vos la entonación/velocidad/distancia al grabar las 50 positivas.
POSITIVE_PROMPTS = [
    "nexa",
]
# Hard negatives para /ks/ y parciales:
# - Palabras con /ks/ similar pero distintas (taxi, texto, examen)
# - Fragmentos ("exa" solo) para que no dispare con partes
# - "alexa" para que NO confunda si hay un Echo cerca
NEGATIVE_PROMPTS = [
    "exa",                                   # fragmento final — no debe disparar
    "ne",                                    # fragmento inicial
    "alexa",                                  # confusor crítico /ks/+/a/ final
    "taxi",                                   # /ks/ en medio, distinto contexto
    "examen",                                 # /ks/ en contexto
    "texto",                                  # /ks/ en contexto
    "excelente",                              # /ks/ en contexto
    "(silencio — no digas nada)",
    "buen día, ¿cómo andás?",
    "la luz del living está prendida",
    "tengo que salir más tarde",
    "pasame el control remoto",
    "¿qué hora es?",
    "no tengo ganas de cocinar",
    "vamos al parque",
    "mañana llueve",
    "prendé la luz del escritorio",          # comandos del asistente (post wake word)
    "apagá la cocina",
    "qué temperatura hace",
    "me voy a dormir",
]

SAMPLE_RATE = 16000
SAMPLE_DURATION_S = 2.0
MIC_DEVICE = 9  # ReSpeaker 4 Mic Array v2.0 — ch0 = beamformed + AEC


def record_sample(duration_s: float = SAMPLE_DURATION_S) -> tuple[np.ndarray, float]:
    """Graba SAMPLE_DURATION_S del ReSpeaker (6 canales → ch0 procesado)."""
    import sounddevice as sd
    samples = int(SAMPLE_RATE * duration_s)
    audio_6ch = sd.rec(samples, samplerate=SAMPLE_RATE, channels=6, device=MIC_DEVICE, dtype="float32")
    sd.wait()
    mono = audio_6ch[:, 0].copy()
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return mono, rms


def save_wav(path: Path, audio_float32: np.ndarray) -> None:
    pcm = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def cmd_record_positive(args):
    out_dir = ROOT / "data/wakeword_training" / WAKE_WORD_NAME / "positive"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    start_idx = len(existing)
    target = args.count

    print(f"\n🎤 GRABACIÓN POSITIVAS ({WAKE_WORD_NAME})")
    print(f"   Existentes: {start_idx}. Objetivo: grabar {target} más.")
    print(f"   Dispositivo: {MIC_DEVICE} (ReSpeaker)")
    print(f"   Duración por muestra: {SAMPLE_DURATION_S}s")
    print(f"   Frase: '{POSITIVE_PROMPTS[0]}' — decila cada vez variando:")
    print(f"     - Entonación (neutra / enfática / rápida / lenta)")
    print(f"     - Distancia al mic (30-80 cm)")
    print(f"     - Volumen (normal / bajo / fuerte)")
    print(f"   Esta variación hace que el modelo generalice mejor.")
    print()

    service_was_running = _stop_voice_service()
    try:
        input("Enter para empezar (Ctrl-C para salir)...")

        recorded = 0
        for i in range(target):
            prompt = POSITIVE_PROMPTS[i % len(POSITIVE_PROMPTS)]
            print(f"\n[{i + 1}/{target}]  →  decí: '{prompt}'")
            for c in (3, 2, 1):
                print(f"   {c}...", end="", flush=True)
                time.sleep(0.5)
            print(" 🎙️  ¡HABLA!")
            audio, rms = record_sample()
            if rms < 0.01:
                print(f"   ⚠️ RMS={rms:.4f} — muy bajo, descartado. Intentá más fuerte o más cerca.")
                continue
            fname = f"positive_{start_idx + recorded + 1:04d}.wav"
            save_wav(out_dir / fname, audio)
            print(f"   ✅ {fname} (RMS={rms:.3f})")
            recorded += 1
            time.sleep(0.8)

        print(f"\n✅ Grabadas {recorded}/{target} positivas. Total en disco: {start_idx + recorded}")
    finally:
        if service_was_running:
            _start_voice_service()


def cmd_record_negative(args):
    out_dir = ROOT / "data/wakeword_training" / WAKE_WORD_NAME / "negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    start_idx = len(existing)
    target = args.count

    print(f"\n🎤 GRABACIÓN NEGATIVAS")
    print(f"   Existentes: {start_idx}. Objetivo: grabar {target} más.")
    print(f"   Decí frases VARIADAS que NO sean el wake word.")
    print(f"   Incluir silencio y ruido ambiente ayuda a reducir falsos positivos.")
    print()

    service_was_running = _stop_voice_service()
    try:
        input("Enter para empezar (Ctrl-C para salir)...")

        recorded = 0
        for i in range(target):
            prompt = random.choice(NEGATIVE_PROMPTS)
            print(f"\n[{i + 1}/{target}]  →  decí algo como: '{prompt}' (o improvisá)")
            for c in (3, 2, 1):
                print(f"   {c}...", end="", flush=True)
                time.sleep(0.5)
            print(" 🎙️  ¡HABLA!")
            audio, rms = record_sample()
            fname = f"negative_{start_idx + recorded + 1:04d}.wav"
            save_wav(out_dir / fname, audio)
            print(f"   ✅ {fname} (RMS={rms:.3f})")
            recorded += 1
            time.sleep(0.8)

        print(f"\n✅ Grabadas {recorded}/{target} negativas. Total: {start_idx + recorded}")
    finally:
        if service_was_running:
            _start_voice_service()


def cmd_train(args):
    from src.wakeword.trainer import WakeWordTrainer
    trainer = WakeWordTrainer(data_dir="./data/wakeword_training", models_dir="./models/wakeword")
    print(f"Entrenando '{WAKE_WORD_NAME}' ({args.epochs} epochs)...")
    model_path = trainer.train(
        wake_word_name=WAKE_WORD_NAME,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment_data=True,
    )
    if model_path:
        print(f"\n✅ Modelo entrenado: {model_path}")
    else:
        print("\n❌ Entrenamiento falló. Verificá que tenés ≥30 positivas y ≥30 negativas.")


def cmd_deploy(args):
    """Copia el .onnx entrenado al path del detector + update settings."""
    src = ROOT / "models/wakeword" / f"{WAKE_WORD_NAME}.onnx"
    if not src.exists():
        print(f"❌ No encuentro {src} — ¿entrenaste ya?")
        return
    print(f"✅ Modelo custom: {src}")
    print(f"   El detector lo levanta vía custom_models_dir='./models/wakeword' + model='{WAKE_WORD_NAME}'.")
    print(f"\nEditá config/settings.yaml:")
    print(f"  wake_word.model: '{WAKE_WORD_NAME}'")
    print(f"  rooms.wake_word.model: '{WAKE_WORD_NAME}'")
    print(f"\nLuego: systemctl --user restart kza-voice.service")


def cmd_test(args):
    """
    Loop interactivo con threshold configurable.
    Imprime en tiempo real: RMS del audio + score máximo por modelo.
    Con --verbose imprime cada chunk; sin --verbose solo los "interesantes" (RMS alto
    o score > 0.05, para no inundar la consola con silencio).
    """
    from src.wakeword.detector import WakeWordDetector
    import sounddevice as sd
    import queue

    det = WakeWordDetector(
        models=[WAKE_WORD_NAME] if args.custom else ["hey_jarvis"],
        threshold=args.threshold,
        custom_models_dir="./models/wakeword",
    )
    det.load()
    print(f"Threshold: {args.threshold}. Modelos activos: {det.get_active_models() if hasattr(det,'get_active_models') else 'hey_jarvis'}")
    print(f"Hablá al mic del escritorio. El countdown muestra RMS y score en vivo.")

    service_was_running = _stop_voice_service()
    try:
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].astype(np.float32).copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=1280, channels=6,
            device=MIC_DEVICE, dtype="float32", callback=cb,
        ):
            print("Listening... Ctrl-C para salir\n")
            max_score_window = 0.0
            try:
                while True:
                    chunk = q.get()
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    scores = det.predict(chunk)  # {model: float}
                    best_model = max(scores, key=scores.get) if scores else "?"
                    best_score = scores.get(best_model, 0.0)
                    max_score_window = max(max_score_window, best_score)

                    triggered = best_score >= args.threshold
                    show = args.verbose or triggered or rms > 0.02 or best_score > 0.05

                    if show:
                        bar = "█" * int(best_score * 30)
                        marker = " 🔥 TRIGGER" if triggered else ""
                        print(f"  RMS={rms:.3f}  {best_model}={best_score:.3f} {bar:<30}{marker}")

            except KeyboardInterrupt:
                print(f"\nMax score observado en la sesión: {max_score_window:.3f}")
    finally:
        if service_was_running:
            _start_voice_service()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record-positive"); p.add_argument("--count", type=int, default=100)
    p = sub.add_parser("record-negative"); p.add_argument("--count", type=int, default=150)
    p = sub.add_parser("train"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=32)
    sub.add_parser("deploy")
    p = sub.add_parser("test"); p.add_argument("--threshold", type=float, default=0.35); p.add_argument("--custom", action="store_true"); p.add_argument("--verbose", action="store_true", help="Imprime todos los chunks (default: solo cuando RMS>0.02 o score>0.05)")
    args = ap.parse_args()

    handlers = {
        "record-positive": cmd_record_positive,
        "record-negative": cmd_record_negative,
        "train": cmd_train,
        "deploy": cmd_deploy,
        "test": cmd_test,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
