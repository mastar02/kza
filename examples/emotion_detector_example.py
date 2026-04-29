"""
Ejemplo de uso del EmotionDetector.

Muestra cómo:
1. Inicializar el detector
2. Detectar emociones en audio
3. Acceder a los resultados y ajustes de respuesta
"""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from users.emotion_detector import EmotionDetector


def example_detect_emotion():
    """Ejemplo básico de detección de emoción"""

    # Inicializar detector en cuda:0 (pipeline de audio consolidado, comparte
    # GPU con speaker_identifier por diseño)
    detector = EmotionDetector(
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cuda:0",
        sample_rate=16000
    )

    # Simular audio (en real, esto vendría de micrófono/archivo)
    # 1 segundo de audio a 16kHz
    sample_audio = np.random.randn(16000).astype(np.float32)

    # Cargar modelo (lazy loading - solo la primera vez)
    detector.load()

    # Detectar emoción
    result = detector.detect(sample_audio)

    print(f"Emoción detectada: {result.emotion}")
    print(f"Confianza: {result.confidence:.2%}")
    print(f"Arousal (energía): {result.arousal:.2f}")
    print(f"Valence (positivo/negativo): {result.valence:.2f}")
    print(f"Tiempo de procesamiento: {result.processing_time_ms:.1f}ms")

    # Acceder a ajustes para adaptar respuesta
    adjustments = result.response_adjustment
    print(f"\nAjustes para respuesta:")
    print(f"  - Pitch shift: {adjustments['pitch_shift']:+.1f}")
    print(f"  - Velocidad: {adjustments['speech_rate']:.1f}x")
    print(f"  - Energía: {adjustments['energy']:.1f}x")
    print(f"  - Tono emocional: {adjustments['emotional_tone']}")


def example_batch_detection():
    """Ejemplo de detección en lote"""

    detector = EmotionDetector(
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cuda:0",
    )
    detector.load()

    # Simular múltiples audios
    audio_samples = [
        np.random.randn(16000).astype(np.float32),
        np.random.randn(16000).astype(np.float32),
        np.random.randn(16000).astype(np.float32),
    ]

    # Detectar todas a la vez
    results = detector.batch_detect(audio_samples)

    for i, result in enumerate(results, 1):
        print(f"Audio {i}: {result.emotion} (conf: {result.confidence:.2%})")


def example_emotion_description():
    """Ejemplo de descripción de emociones en español"""

    detector = EmotionDetector(
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cuda:0",
    )

    emotions = ["happy", "sad", "angry", "fearful", "neutral", "surprised"]

    print("Emociones en español:")
    for emotion in emotions:
        description = detector.get_emotion_description(emotion)
        characteristics = detector.EMOTION_CHARACTERISTICS[emotion]
        print(f"  {emotion:10} -> {description:12} (arousal: {characteristics['arousal']:.1f}, valence: {characteristics['valence']:.1f})")


def example_error_handling():
    """Ejemplo de manejo de errores graceful"""

    detector = EmotionDetector(
        model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device="cuda:0",
    )

    # Audio vacío o corrupto
    empty_audio = np.array([], dtype=np.float32)

    try:
        result = detector.detect(empty_audio)
        print(f"Resultado: {result.emotion} (fallback neutral)")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EmotionDetector - Ejemplos")
    print("=" * 60)

    print("\n1. DETECCIÓN BÁSICA DE EMOCIÓN")
    print("-" * 60)
    # example_detect_emotion()  # Requiere transformers + CUDA
    print("(Requiere transformers instalado)")

    print("\n2. DETECCIÓN EN LOTE")
    print("-" * 60)
    # example_batch_detection()  # Requiere transformers + CUDA
    print("(Requiere transformers instalado)")

    print("\n3. DESCRIPCIÓN DE EMOCIONES")
    print("-" * 60)
    example_emotion_description()

    print("\n4. MANEJO DE ERRORES")
    print("-" * 60)
    # example_error_handling()  # Requiere transformers + CUDA
    print("(Requiere transformers instalado)")
