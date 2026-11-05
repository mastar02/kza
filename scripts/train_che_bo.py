"""
Sesión local de entrenamiento para el wake word «nexa».

La grabación y la prueba en vivo requieren la concesión administrativa original
mediante --dashboard-url URL y --capture-token-env VARIABLE (alias --token-env).
Ejemplo: record-positive --dashboard-url "$KZA_DASHBOARD_URL" --count 50.
La variable predeterminada de credencial es KZA_DASHBOARD_TOKEN.

Se conservan seis canales a 16000 Hz, dispositivo 9 y una copia del canal 0.
Las positivas descartan RMS < 0.01; las negativas conservan también el silencio.
La captura no detiene ni reinicia servicios: su gestión queda separada.

Comandos:
  record-positive   — graba positivas
  record-negative   — graba negativas, incluidos hard negatives acústicos
  train             — entrena el modelo ONNX con muestras existentes
  deploy            — muestra instrucciones; no copia ni publica el modelo
  test              — detección en vivo con threshold configurable

train y deploy no requieren permiso de captura. Ambos nombres de script son
aliases del mismo wake word «nexa»; no se infiere otro modelo por el filename.
"""

from __future__ import annotations

import argparse
import random
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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


def _authority_factory(args):
    import os
    from src.audio.native_capture_authority import WakewordTrainingAuthorityClient

    if not args.capture_url or not args.capture_token_env:
        raise RuntimeError("legacy_wakeword_authority_required")
    token = os.environ.get(args.capture_token_env)
    if not token:
        raise RuntimeError("legacy_wakeword_authority_required")
    return lambda: WakewordTrainingAuthorityClient(args.capture_url, token)


def _raise_capture_errors(primary, cleanup):
    if primary is not None and cleanup is not None:
        raise BaseExceptionGroup("legacy_wakeword_failed", [primary, cleanup]) from None
    error = primary if primary is not None else cleanup
    if error is not None:
        raise error from None


def _drive_capture(authority_factory, function):
    from src.wakeword.legacy_capture import LegacyWakewordCapture, fresh_error
    capture = LegacyWakewordCapture(authority_factory)
    primary = cleanup = result = None
    try:
        capture.start()
        with capture.operation("main"):
            try:
                result = function(capture)
            except BaseException as error:
                primary = fresh_error(error)
    except BaseException as error:
        primary = fresh_error(error)
    finally:
        function = authority_factory = None
        try:
            capture.finish()
        except BaseException as error:
            cleanup = fresh_error(error)
    if primary is not None or cleanup is not None:
        result = None
        _raise_capture_errors(primary, cleanup)
    return result


def _record_sample(capture, duration_s=SAMPLE_DURATION_S):
    from src.wakeword.legacy_capture import fresh_error
    import sounddevice as sd
    audio_6ch = mono = stream = None
    primary = cleanup = None
    key = None
    with capture.operation("record"):
        try:
            key, stream = capture.open_stream(lambda: sd.InputStream(
                samplerate=SAMPLE_RATE, channels=6, device=MIC_DEVICE, dtype="float32"))
            try:
                capture.require_current()
                audio_6ch, _overflowed = stream.read(int(SAMPLE_RATE * duration_s))
            except BaseException as error:
                primary = fresh_error(error)
            finally:
                stream = None
                close_key, key = key, None
                try:
                    capture.close_stream(close_key)
                except BaseException as error:
                    cleanup = fresh_error(error)
            if primary is not None or cleanup is not None:
                audio_6ch = mono = _overflowed = None
                _raise_capture_errors(primary, cleanup)
            capture.require_current()
            mono = audio_6ch[:, 0].copy()
            rms = float(np.sqrt(np.mean(mono ** 2)))
            return mono, rms
        finally:
            audio_6ch = mono = stream = _overflowed = None
            if key is not None:
                capture.close_stream(key)


def _test(capture, args):
    from src.wakeword.detector import WakeWordDetector
    from src.wakeword.legacy_capture import fresh_error
    import sounddevice as sd
    import queue
    det = chunk = q = _stream = None
    key = None
    primary = cleanup = None
    try:
        det = capture.own_detector(lambda: WakeWordDetector(
            models=[WAKE_WORD_NAME] if args.custom else ["hey_jarvis"],
            threshold=args.threshold, custom_models_dir="./models/wakeword"))
        capture.publish(print, f"Threshold: {args.threshold}. Modelos activos: {det.get_active_models() if hasattr(det, 'get_active_models') else 'hey_jarvis'}")
        capture.publish(print, "Hablá al mic del escritorio. El countdown muestra RMS y score en vivo.")
        q = capture.own_queue(queue.Queue())

        def cb(indata, frames, time_info, status, _queue=q):
            chunk = indata[:, 0].astype(np.float32).copy()
            try:
                capture.publish(_queue.put, chunk)
            finally:
                indata = chunk = None

        key, _stream = capture.open_stream(lambda: sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=1280, channels=6,
            device=MIC_DEVICE, dtype="float32", callback=capture.callback(cb)))
        _stream = None
        capture.publish(print, "Listening... Ctrl-C para salir\n")
        max_score_window = 0.0
        try:
            while True:
                chunk = capture.next_chunk(q)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                scores = det.predict(chunk)
                best_model = max(scores, key=scores.get) if scores else "?"
                best_score = scores.get(best_model, 0.0)
                max_score_window = max(max_score_window, best_score)
                triggered = best_score >= args.threshold
                show = args.verbose or triggered or rms > 0.02 or best_score > 0.05
                if show:
                    bar = "█" * int(best_score * 30)
                    marker = " 🔥 TRIGGER" if triggered else ""
                    capture.publish(print, f"  RMS={rms:.3f}  {best_model}={best_score:.3f} {bar:<30}{marker}")
                chunk = None
        except KeyboardInterrupt:
            capture.publish(print, f"\nMax score observado en la sesión: {max_score_window:.3f}")
    except BaseException as error:
        primary = fresh_error(error)
    finally:
        if key is not None:
            close_key, key = key, None
            try:
                capture.close_stream(close_key)
            except BaseException as error:
                cleanup = fresh_error(error)
        det = chunk = q = _stream = None
    _raise_capture_errors(primary, cleanup)


def _record_positive(capture, args):
    try:
        out_dir = ROOT / 'data/wakeword_training' / WAKE_WORD_NAME / 'positive'
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob('*.wav'))
        start_idx = len(existing)
        target = args.count
        capture.publish(print, f'\n🎤 GRABACIÓN POSITIVAS ({WAKE_WORD_NAME})')
        capture.publish(print, f'   Existentes: {start_idx}. Objetivo: grabar {target} más.')
        capture.publish(print, f'   Dispositivo: {MIC_DEVICE} (ReSpeaker)')
        capture.publish(print, f'   Duración por muestra: {SAMPLE_DURATION_S}s')
        capture.publish(print, f"   Frase: '{POSITIVE_PROMPTS[0]}' — decila cada vez variando:")
        capture.publish(print, '     - Entonación (neutra / enfática / rápida / lenta)')
        capture.publish(print, '     - Distancia al mic (30-80 cm)')
        capture.publish(print, '     - Volumen (normal / bajo / fuerte)')
        capture.publish(print, '   Esta variación hace que el modelo generalice mejor.')
        capture.publish(print)
        if not capture.wait_for_enter('Enter para empezar (Ctrl-C para salir)...'):
            return
        recorded = 0
        for i in range(target):
            prompt = POSITIVE_PROMPTS[i % len(POSITIVE_PROMPTS)]
            capture.publish(print, f"\n[{i + 1}/{target}]  →  decí: '{prompt}'")
            for c in (3, 2, 1):
                capture.publish(print, f'   {c}...', end='', flush=True)
                capture.pause(0.5)
            capture.publish(print, ' 🎙️  ¡HABLA!')
            audio, rms = _record_sample(capture)
            if rms < 0.01:
                capture.publish(print, f'   ⚠️ RMS={rms:.4f} — muy bajo, descartado. Intentá más fuerte o más cerca.')
                continue
            fname = f'positive_{start_idx + recorded + 1:04d}.wav'
            capture.publish(save_wav, out_dir / fname, audio)
            capture.publish(print, f'   ✅ {fname} (RMS={rms:.3f})')
            recorded += 1
            capture.pause(0.8)
        capture.publish(print, f'\n✅ Grabadas {recorded}/{target} positivas. Total en disco: {start_idx + recorded}')
    finally:
        audio = None

def _record_negative(capture, args):
    try:
        out_dir = ROOT / 'data/wakeword_training' / WAKE_WORD_NAME / 'negative'
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob('*.wav'))
        start_idx = len(existing)
        target = args.count
        capture.publish(print, '\n🎤 GRABACIÓN NEGATIVAS')
        capture.publish(print, f'   Existentes: {start_idx}. Objetivo: grabar {target} más.')
        capture.publish(print, '   Decí frases VARIADAS que NO sean el wake word.')
        capture.publish(print, '   Incluir silencio y ruido ambiente ayuda a reducir falsos positivos.')
        capture.publish(print)
        if not capture.wait_for_enter('Enter para empezar (Ctrl-C para salir)...'):
            return
        recorded = 0
        for i in range(target):
            prompt = random.choice(NEGATIVE_PROMPTS)
            capture.publish(print, f"\n[{i + 1}/{target}]  →  decí algo como: '{prompt}' (o improvisá)")
            for c in (3, 2, 1):
                capture.publish(print, f'   {c}...', end='', flush=True)
                capture.pause(0.5)
            capture.publish(print, ' 🎙️  ¡HABLA!')
            audio, rms = _record_sample(capture)
            fname = f'negative_{start_idx + recorded + 1:04d}.wav'
            capture.publish(save_wav, out_dir / fname, audio)
            capture.publish(print, f'   ✅ {fname} (RMS={rms:.3f})')
            recorded += 1
            capture.pause(0.8)
        capture.publish(print, f'\n✅ Grabadas {recorded}/{target} negativas. Total: {start_idx + recorded}')
    finally:
        audio = None

def record_sample(duration_s: float = SAMPLE_DURATION_S, *, authority_factory=None) -> tuple[np.ndarray, float]:
    """Capture a six-channel sample under an explicit original training grant."""
    return _drive_capture(authority_factory, lambda capture: _record_sample(capture, duration_s))


def save_wav(path: Path, audio_float32: np.ndarray) -> None:
    pcm = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def cmd_record_positive(args):
    return _drive_capture(_authority_factory(args), lambda capture: _record_positive(capture, args))


def cmd_record_negative(args):
    return _drive_capture(_authority_factory(args), lambda capture: _record_negative(capture, args))


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
    print("\nEditá config/settings.yaml:")
    print(f"  wake_word.model: '{WAKE_WORD_NAME}'")
    print(f"  rooms.wake_word.model: '{WAKE_WORD_NAME}'")
    print("\nLuego: systemctl --user restart kza-voice.service")


def cmd_test(args):
    return _drive_capture(_authority_factory(args), lambda capture: _test(capture, args))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record-positive")
    p.add_argument("--count", type=int, default=100)
    p = sub.add_parser("record-negative")
    p.add_argument("--count", type=int, default=150)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    sub.add_parser("deploy")
    p = sub.add_parser("test")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--custom", action="store_true")
    p.add_argument("--verbose", action="store_true", help="Imprime todos los chunks (default: solo cuando RMS>0.02 o score>0.05)")
    for name in ("record-positive", "record-negative", "test"):
        parser = sub.choices[name]
        parser.add_argument("--dashboard-url", "--capture-url", dest="capture_url", required=True)
        parser.add_argument("--capture-token-env", "--token-env", default="KZA_DASHBOARD_TOKEN")
    args = ap.parse_args(argv)
    handlers = {"record-positive": cmd_record_positive, "record-negative": cmd_record_negative,
                "train": cmd_train, "deploy": cmd_deploy, "test": cmd_test}
    handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
