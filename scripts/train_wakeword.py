#!/usr/bin/env python3
"""
KZA Voice - Wake Word Training CLI
Graba muestras y entrena wake words personalizados.
"""

import argparse
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))
import src  # noqa: E402,F401 -- G1 (U3b-0e): refuses an unmanaged run under a managed root (exit 78)


def _authority_factory(args: argparse.Namespace):
    """Keep the explicitly selected origin and operator credential per command."""
    import os

    from src.audio.native_capture_authority import WakewordTrainingAuthorityClient

    url = getattr(args, "dashboard_url", None)
    token_env = getattr(args, "token_env", "KZA_DASHBOARD_TOKEN")
    token = os.environ.get(token_env, "")
    if not url or not token:
        raise RuntimeError("wakeword_capture_authority_required")

    def create() -> WakewordTrainingAuthorityClient:
        return WakewordTrainingAuthorityClient(url, token)

    return create


def cmd_record(args: argparse.Namespace) -> None:
    """Grabar muestras de entrenamiento con una autoridad explícita."""
    authority_factory = _authority_factory(args)
    from src.wakeword.recorder import WakeWordRecorder

    print(f"\n🎤 Grabación de muestras para: '{args.name}'")
    print("=" * 50)

    recorder = WakeWordRecorder(
        wake_word_name=args.name,
        output_dir=args.output_dir,
        authority_factory=authority_factory,
    )

    stats = recorder.get_stats()
    print(f"Muestras existentes: {stats['positive_samples']} positivas, {stats['negative_samples']} negativas")
    print()

    if args.positive:
        print(f"📢 Grabando {args.count} muestras POSITIVAS")
        print(f"   Deberás decir: '{args.name}'")
        print()
        input("Presiona Enter para comenzar...")

        recorded = recorder.record_positive_samples(n=args.count)
        print(f"\n✅ Grabadas {recorded} muestras positivas")

    if args.negative:
        print(f"\n📢 Grabando {args.count} muestras NEGATIVAS")
        print("   Di cualquier cosa EXCEPTO la palabra de activación")
        print()
        input("Presiona Enter para comenzar...")

        recorded = recorder.record_negative_samples(n=args.count)
        print(f"\n✅ Grabadas {recorded} muestras negativas")

    # Mostrar estado final
    stats = recorder.get_stats()
    print("\n📊 Estado actual:")
    print(f"   Positivas: {stats['positive_samples']}")
    print(f"   Negativas: {stats['negative_samples']}")
    print(f"   Listo para entrenar: {'✅' if stats['ready_for_training'] else '❌ (necesitas mínimo 30 de cada)'}")


def cmd_train(args):
    """Entrenar modelo de wake word"""
    from src.wakeword.trainer import WakeWordTrainer
    from src.wakeword.recorder import WakeWordRecorder

    print(f"\n🧠 Entrenando modelo para: '{args.name}'")
    print("=" * 50)

    # Verificar datos
    recorder = WakeWordRecorder(wake_word_name=args.name, output_dir=args.data_dir)
    stats = recorder.get_stats()

    if not stats['ready_for_training']:
        print("❌ Datos insuficientes:")
        print(f"   Positivas: {stats['positive_samples']} (mínimo 30)")
        print(f"   Negativas: {stats['negative_samples']} (mínimo 30)")
        print(f"\nUsa: python {sys.argv[0]} record --name '{args.name}' --positive --negative")
        return 1

    print("📊 Datos de entrenamiento:")
    print(f"   Positivas: {stats['positive_samples']}")
    print(f"   Negativas: {stats['negative_samples']}")
    print()

    trainer = WakeWordTrainer(
        data_dir=args.data_dir,
        models_dir=args.output_dir
    )

    print("⏳ Entrenando (esto puede tomar varios minutos)...")
    model_path = trainer.train(
        wake_word_name=args.name,
        epochs=args.epochs
    )

    if model_path:
        print("\n✅ Modelo entrenado exitosamente!")
        print(f"   Path: {model_path}")
        print("\nPara usar este wake word, actualiza config/settings.yaml:")
        print("   wake_word:")
        print(f"     model: '{args.name}'")
        return 0
    else:
        print("\n❌ Error en entrenamiento")
        return 1


def cmd_test(args: argparse.Namespace) -> None:
    """Test detection under an explicit live capture authority."""
    authority = _authority_factory(args)()
    from src.wakeword.live_capture import LiveWakewordCapture

    with LiveWakewordCapture(authority) as live:
        from src.wakeword.detector import WakeWordDetector
        import sounddevice as sd

        live.publish(print, f"\n🎤 Probando wake word: '{args.model}'")
        live.publish(print, "=" * 50)
        live.publish(print, f"Umbral: {args.threshold}")
        live.publish(print, "Presiona Ctrl+C para terminar\n")
        detector = live.own_detector(lambda: WakeWordDetector(
            models=[args.model], threshold=args.threshold,
        ))
        live.publish(print, f"Modelos activos: {detector.get_active_models()}")
        live.publish(print, "\n🎧 Escuchando...\n")
        detection_count = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal detection_count
            audio = indata[:, 0].copy()
            try:
                live.require_current()
                result = detector.detect(audio)
                live.require_current()
                if result:
                    model_name, confidence = result
                    detection_count += 1
                    live.publish(print, f"  🔔 [{detection_count}] Detectado: {model_name} (conf: {confidence:.2f})")
            finally:
                audio.fill(0)

        live.open_stream(lambda: sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", blocksize=1280,
            callback=live.callback(audio_callback),
        ))
        try:
            live.publish(print, "Di la palabra de activación...")
            while True:
                live.require_current()
                sd.sleep(100)
        except KeyboardInterrupt:
            live.publish(print, f"\n\n📊 Resumen: {detection_count} detecciones")

def cmd_list(args):
    """Listar modelos disponibles"""
    from src.wakeword.detector import WakeWordDetector
    from src.wakeword.trainer import WakeWordTrainer

    print("\n📋 Modelos de Wake Word Disponibles")
    print("=" * 50)

    # Pre-entrenados
    print("\n🏭 Pre-entrenados (OpenWakeWord):")
    for model in WakeWordDetector.PRETRAINED_MODELS:
        print(f"   • {model}")

    # Personalizados
    trainer = WakeWordTrainer()
    custom = trainer.list_trained_models()

    if custom:
        print("\n🎨 Personalizados (entrenados):")
        for model in custom:
            size = model.get('size_kb', 0)
            samples = model.get('positive_samples', '?')
            print(f"   • {model['name']} ({size:.1f} KB, {samples} muestras)")
    else:
        print("\n🎨 Personalizados: ninguno")
        print("   Entrena uno con: python train_wakeword.py record --name 'mi_palabra'")


def main():
    parser = argparse.ArgumentParser(
        description="KZA Wake Word Training Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Grabar muestras positivas y negativas
  python train_wakeword.py record --name "oye casa" --positive --negative --count 50 --dashboard-url http://localhost:8000

  # Solo grabar más muestras positivas
  python train_wakeword.py record --name "oye casa" --positive --count 20 --dashboard-url http://localhost:8000

  # Entrenar modelo
  python train_wakeword.py train --name "oye casa" --epochs 100

  # Probar detección
  python train_wakeword.py test --model "oye_casa" --threshold 0.5 --dashboard-url http://localhost:8000

  # Listar modelos disponibles
  python train_wakeword.py list
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando")

    # Record
    record_parser = subparsers.add_parser("record", help="Grabar muestras de entrenamiento")
    record_parser.add_argument("--name", required=True, help="Nombre del wake word")
    record_parser.add_argument("--positive", action="store_true", help="Grabar muestras positivas")
    record_parser.add_argument("--negative", action="store_true", help="Grabar muestras negativas")
    record_parser.add_argument("--count", type=int, default=50, help="Número de muestras (default: 50)")
    record_parser.add_argument("--output-dir", default="./data/wakeword_training", help="Directorio de salida")

    record_parser.add_argument("--dashboard-url", required=True, help="Origen del Dashboard que controla la captura")
    record_parser.add_argument("--token-env", default="KZA_DASHBOARD_TOKEN", help="Variable de entorno con la credencial administrativa")

    # Train
    train_parser = subparsers.add_parser("train", help="Entrenar modelo de wake word")
    train_parser.add_argument("--name", required=True, help="Nombre del wake word")
    train_parser.add_argument("--epochs", type=int, default=100, help="Épocas de entrenamiento (default: 100)")
    train_parser.add_argument("--data-dir", default="./data/wakeword_training", help="Directorio de datos")
    train_parser.add_argument("--output-dir", default="./models/wakeword", help="Directorio de modelos")

    # Test
    test_parser = subparsers.add_parser("test", help="Probar detección de wake word")
    test_parser.add_argument("--model", required=True, help="Modelo a probar")
    test_parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de detección (default: 0.5)")

    test_parser.add_argument("--dashboard-url", required=True, help="Origen del Dashboard que controla la captura")
    test_parser.add_argument("--token-env", default="KZA_DASHBOARD_TOKEN", help="Variable de entorno con la credencial administrativa")

    # List
    subparsers.add_parser("list", help="Listar modelos disponibles")

    args = parser.parse_args()

    if args.command == "record":
        if not args.positive and not args.negative:
            print("Especifica --positive y/o --negative")
            return 1
        return cmd_record(args)

    elif args.command == "train":
        return cmd_train(args)

    elif args.command == "test":
        return cmd_test(args)

    elif args.command == "list":
        return cmd_list(args)

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
