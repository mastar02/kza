# EmotionDetector - Módulo de Detección de Emociones

## Descripción

El módulo `EmotionDetector` detecta emociones en audio usando modelos de speech emotion recognition basados en wav2vec2. Permite adaptar respuestas del sistema según el estado emocional del usuario.

## Características

- **Detección de 6 emociones**: happy, sad, angry, fearful, neutral, surprised
- **Métricas de emoción**:
  - Arousal (energía/activación): 0-1
  - Valence (positivo/negativo): 0-1
  - Confianza en predicción: 0-1

- **Ajustes de respuesta automáticos**:
  - Pitch shift (altura del tono)
  - Speech rate (velocidad de habla)
  - Energy (intensidad)
  - Emotional tone (tono emocional)

- **Lazy loading**: El modelo se carga solo cuando se necesita
- **Error handling graceful**: Retorna "neutral" si hay errores
- **GPU optimizado**: Usa cuda:1 (compartido con speaker_identifier)
- **Batch processing**: Detecta emociones en múltiples audios

## Instalación

Las dependencias necesarias están en `requirements.txt`:

```bash
# Core dependencies
transformers>=4.40.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
```

Para instalación GPU (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers torchaudio
```

## Uso Básico

### Inicialización

```python
from users.emotion_detector import EmotionDetector

# Con valores por defecto (CUDA:1)
detector = EmotionDetector()

# Con configuración personalizada
detector = EmotionDetector(
    model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    device="cuda:1",
    sample_rate=16000
)
```

### Detección de Emoción

```python
import numpy as np

# Audio como numpy array (float32, mono, 16kHz)
audio = np.array([...], dtype=np.float32)

# Cargar modelo (lazy loading)
detector.load()

# Detectar emoción
result = detector.detect(audio)

# Acceder a resultados
print(f"Emoción: {result.emotion}")
print(f"Confianza: {result.confidence:.2%}")
print(f"Arousal: {result.arousal:.2f}")
print(f"Valence: {result.valence:.2f}")
print(f"Es confiable: {result.is_confident}")
```

### Ajustes de Respuesta

```python
# Obtener ajustes para adaptar respuesta
adjustments = result.response_adjustment

print(f"Pitch shift: {adjustments['pitch_shift']}")      # -0.5 a 0.5
print(f"Speech rate: {adjustments['speech_rate']}")      # 0.5 a 2.0
print(f"Energy: {adjustments['energy']}")                # 0.5 a 2.0
print(f"Tone: {adjustments['emotional_tone']}")          # cheerful, sympathetic, etc
```

### Detección en Lote

```python
# Detectar múltiples audios
audio_samples = [audio1, audio2, audio3]
results = detector.batch_detect(audio_samples)

for result in results:
    print(f"{result.emotion} - {result.confidence:.2%}")
```

## Arquitectura

```
Audio Input (16kHz, float32)
    ↓
wav2vec2 Embeddings
    ↓
Emotion Classifier
    ↓
EmotionResult
├─ emotion: str
├─ confidence: float
├─ arousal: float
├─ valence: float
├─ all_emotions: dict
└─ response_adjustment: dict
```

## Modelos Soportados

### Recomendado
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (Recomendado)
  - Entrenado en multiple speech sources
  - MSP-Podcast dataset
  - 6 emociones: happy, sad, angry, fearful, neutral, surprised

### Alternativas
- `facebook/wav2vec2-large-xlsr-53` + clasificador custom
- Modelos más ligeros para inference rápido

## Resultados (EmotionResult)

### Propiedades

```python
@dataclass
class EmotionResult:
    emotion: str              # happy, sad, angry, fearful, neutral, surprised
    confidence: float         # 0-1 (>= 0.7 es confiable)
    arousal: float           # 0-1 (energía/activación)
    valence: float           # 0-1 (positivo:1, negativo:0)
    all_emotions: dict       # {emotion: confidence} para todas
    processing_time_ms: float

    @property
    def is_confident(self) -> bool:
        """¿La predicción es confiable? (>= 0.7)"""

    @property
    def response_adjustment(self) -> dict:
        """Ajustes de respuesta recomendados"""
```

## Mapeo de Emociones

| Emoción | Arousal | Valence | Ajustes |
|---------|---------|---------|---------|
| happy | 0.8 | 0.9 | pitch +0.2, speed 1.1x, energy 1.2x |
| sad | 0.3 | 0.1 | pitch -0.2, speed 0.9x, energy 0.8x |
| angry | 0.9 | 0.2 | pitch +0.1, speed 1.2x, energy 1.5x |
| fearful | 0.7 | 0.2 | pitch +0.15, speed 1.3x, energy 0.7x |
| surprised | 0.8 | 0.6 | pitch +0.3, speed 1.2x, energy 1.3x |
| neutral | 0.5 | 0.5 | sin cambios |

## Manejo de Errores

El módulo maneja errores gracefully y retorna "neutral" como fallback:

```python
try:
    result = detector.detect(audio)
except Exception as e:
    # Retorna neutral como fallback automáticamente
    logger.error(f"Error detectando emoción: {e}")
```

## Performance

- Tiempo de carga del modelo: ~5-10s (primera vez)
- Tiempo de inferencia: ~50-200ms por audio
- Uso de GPU: ~2-4GB VRAM
- Compatible con cuda:1 (compartido)

## Tests

Run all tests:
```bash
pytest tests/unit/users/test_emotion_detector.py -v
```

Coverage de tests: 35 test cases
- 8 tests para EmotionResult
- 9 tests para inicialización y carga
- 13 tests para detección
- 2 tests para batch processing
- 3 tests para utilidades

## Integración con Otras Partes

### Speaker Identifier
Ambos usan GPU cuda:1:
```python
speaker_id = SpeakerIdentifier(device="cuda:1")
emotion_detector = EmotionDetector(device="cuda:1")
# Comparten memoria de GPU eficientemente
```

### TTS (Text-to-Speech)
Adaptar TTS usando ajustes de emoción:
```python
emotion_result = detector.detect(user_audio)
adjustments = emotion_result.response_adjustment

# Usar adjustments en TTS
tts.synthesize(
    text=response,
    pitch=adjustments['pitch_shift'],
    speed=adjustments['speech_rate'],
    energy=adjustments['energy'],
    emotional_tone=adjustments['emotional_tone']
)
```

### Rutinas Personalizadas
Adaptar rutinas según emoción detectada:
```python
result = detector.detect(audio)

if result.emotion == "sad":
    # Ejecutar rutina de consuelo
    routine_manager.execute("comfort_routine")
elif result.emotion == "angry":
    # Ejecutar rutina de calma
    routine_manager.execute("calm_routine")
```

## Ejemplos

Ver `/examples/emotion_detector_example.py` para ejemplos completos.

## Notas Técnicas

1. **Audio Input**: Debe ser float32, mono, 16kHz
2. **Normalización**: Se normaliza automáticamente si valores > 1.0
3. **Lazy Loading**: Modelo se carga en primer `detect()` o `load()`
4. **Caché**: No hay caché de modelos persistente (se recarga cada sesión)
5. **Device Sharing**: GPU cuda:1 compartida con speaker_identifier

## Limitaciones

- Requiere audio de buena calidad (SNR > 20dB recomendado)
- Entrenado principalmente con inglés
- Mejor desempeño con audio > 1 segundo
- Emociones con arousal similar pueden ser confundidas
- Emociones de fondo en múltiples hablantes pueden afectar

## Referencias

- Modelo: [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
- Dataset: MSP-Podcast
- Paper: "The MSP-Podcast Corpus: A Large and Diverse Multi-Speaker Corpus for Speech Emotion Recognition"
