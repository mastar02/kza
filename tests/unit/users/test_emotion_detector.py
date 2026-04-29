"""
Tests for Emotion Detector Module.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from users.emotion_detector import EmotionDetector, EmotionResult
from tests.factories import make_emotion_detector


class TestEmotionResult:
    """Test suite for EmotionResult dataclass"""

    def test_emotion_result_creation(self):
        """Test creating an EmotionResult"""
        result = EmotionResult(
            emotion="happy",
            confidence=0.95,
            arousal=0.8,
            valence=0.9,
            all_emotions={"happy": 0.95, "sad": 0.05},
            processing_time_ms=50.0,
        )

        assert result.emotion == "happy"
        assert result.confidence == 0.95
        assert result.arousal == 0.8
        assert result.valence == 0.9
        assert result.processing_time_ms == 50.0

    def test_is_confident_threshold(self):
        """Test is_confident property"""
        # Confident
        result = EmotionResult(
            emotion="happy",
            confidence=0.75,
            arousal=0.8,
            valence=0.9,
            all_emotions={},
            processing_time_ms=0.0,
        )
        assert result.is_confident is True

        # Not confident
        result = EmotionResult(
            emotion="neutral",
            confidence=0.65,
            arousal=0.5,
            valence=0.5,
            all_emotions={},
            processing_time_ms=0.0,
        )
        assert result.is_confident is False

    def test_response_adjustment_happy(self):
        """Test response adjustment for happy emotion"""
        result = EmotionResult(
            emotion="happy",
            confidence=0.9,
            arousal=0.8,
            valence=0.9,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        assert adjustment["pitch_shift"] == 0.2
        assert adjustment["speech_rate"] == 1.1
        assert adjustment["energy"] == 1.2
        assert adjustment["emotional_tone"] == "cheerful"

    def test_response_adjustment_sad(self):
        """Test response adjustment for sad emotion"""
        result = EmotionResult(
            emotion="sad",
            confidence=0.85,
            arousal=0.3,
            valence=0.1,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        assert adjustment["pitch_shift"] == -0.2
        assert adjustment["speech_rate"] == 0.9
        assert adjustment["energy"] == 0.8
        assert adjustment["emotional_tone"] == "sympathetic"

    def test_response_adjustment_angry(self):
        """Test response adjustment for angry emotion"""
        result = EmotionResult(
            emotion="angry",
            confidence=0.88,
            arousal=0.9,
            valence=0.2,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        assert adjustment["pitch_shift"] == 0.1
        assert adjustment["speech_rate"] == 1.2
        assert adjustment["energy"] == 1.5
        assert adjustment["emotional_tone"] == "firm"

    def test_response_adjustment_fearful(self):
        """Test response adjustment for fearful emotion"""
        result = EmotionResult(
            emotion="fearful",
            confidence=0.82,
            arousal=0.7,
            valence=0.2,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        assert adjustment["pitch_shift"] == 0.15
        assert adjustment["speech_rate"] == 1.3
        assert adjustment["energy"] == 0.7
        assert adjustment["emotional_tone"] == "calm_reassuring"

    def test_response_adjustment_surprised(self):
        """Test response adjustment for surprised emotion"""
        result = EmotionResult(
            emotion="surprised",
            confidence=0.80,
            arousal=0.8,
            valence=0.6,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        assert adjustment["pitch_shift"] == 0.3
        assert adjustment["speech_rate"] == 1.2
        assert adjustment["energy"] == 1.3
        assert adjustment["emotional_tone"] == "engaging"

    def test_response_adjustment_neutral(self):
        """Test response adjustment for neutral emotion"""
        result = EmotionResult(
            emotion="neutral",
            confidence=0.70,
            arousal=0.5,
            valence=0.5,
            all_emotions={},
            processing_time_ms=0.0,
        )

        adjustment = result.response_adjustment

        # Neutral should have no special adjustments
        assert adjustment["pitch_shift"] == 0.0
        assert adjustment["speech_rate"] == 1.0
        assert adjustment["energy"] == 1.0
        assert adjustment["emotional_tone"] == "neutral"


class TestEmotionDetectorInit:
    """Test suite for EmotionDetector initialization"""

    def test_init_requires_device(self):
        """Constructor must reject missing device — config is the source of truth."""
        with pytest.raises(TypeError):
            EmotionDetector(model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

    def test_init_requires_model_name(self):
        """Constructor must reject missing model_name."""
        with pytest.raises(TypeError):
            EmotionDetector(device="cpu")

    def test_init_with_factory_defaults(self):
        """Factory provides CPU defaults for behavior-focused tests."""
        detector = make_emotion_detector()

        assert detector.model_name == "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        assert detector.device == "cpu"
        assert detector.sample_rate == 16000
        assert detector._model is None
        assert detector._processor is None

    def test_init_custom_device(self):
        """Test initialization with custom device"""
        detector = make_emotion_detector(device="cuda:0")
        assert detector.device == "cuda:0"

    def test_init_cpu_device(self):
        """Test initialization with CPU device"""
        detector = make_emotion_detector(device="cpu")
        assert detector.device == "cpu"


class TestEmotionDetectorLoad:
    """Test suite for model loading"""

    def test_load_model(self):
        """Test loading the emotion model with mocked pipeline"""
        detector = make_emotion_detector()

        with patch.object(detector, "_load_wav2vec2") as mock_load:
            detector.load()

            assert detector._model_type is not None or detector._model is None
            mock_load.assert_called_once()

    def test_load_model_already_loaded(self):
        """Test loading when model is already loaded"""
        detector = make_emotion_detector()
        detector._model = MagicMock()  # Mark as already loaded

        with patch.object(detector, "_load_wav2vec2") as mock_load:
            detector.load()
            # Should return early without calling _load_wav2vec2
            mock_load.assert_not_called()

    def test_load_model_import_error(self):
        """Test handling ImportError when loading model"""
        detector = make_emotion_detector()

        with patch.object(detector, "_load_wav2vec2") as mock_load:
            mock_load.side_effect = ImportError("transformers not found")

            with pytest.raises(ImportError):
                detector.load()

    def test_get_device_id_cpu(self):
        """Test device ID extraction for CPU"""
        detector = make_emotion_detector(device="cpu")
        assert detector._get_device_id() == -1

    def test_get_device_id_cuda(self):
        """Test device ID extraction for CUDA"""
        detector = make_emotion_detector(device="cuda:1")
        assert detector._get_device_id() == 1

        detector = make_emotion_detector(device="cuda:0")
        assert detector._get_device_id() == 0

    def test_get_device_id_default(self):
        """Test device ID extraction with invalid format"""
        detector = make_emotion_detector(device="invalid")
        # Should default to 0
        assert detector._get_device_id() == 0


class TestEmotionDetectorDetect:
    """Test suite for emotion detection"""

    @pytest.fixture
    def mock_detector(self):
        """Create a detector with mocked model"""
        detector = make_emotion_detector()
        detector._model = MagicMock()
        detector._model_type = "wav2vec2_pipeline"
        return detector

    def test_detect_happy(self, mock_detector, sample_audio):
        """Test detecting happy emotion"""
        mock_detector._model.return_value = [
            {"label": "happy", "score": 0.95},
            {"label": "sad", "score": 0.03},
            {"label": "angry", "score": 0.02},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "happy"
        assert result.confidence == 0.95
        assert result.arousal == 0.8
        assert result.valence == 0.9
        assert result.is_confident is True

    def test_detect_sad(self, mock_detector, sample_audio):
        """Test detecting sad emotion"""
        mock_detector._model.return_value = [
            {"label": "sad", "score": 0.88},
            {"label": "neutral", "score": 0.12},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "sad"
        assert result.confidence == 0.88
        assert result.arousal == 0.3
        assert result.valence == 0.1

    def test_detect_angry(self, mock_detector, sample_audio):
        """Test detecting angry emotion"""
        mock_detector._model.return_value = [
            {"label": "angry", "score": 0.92},
            {"label": "fearful", "score": 0.08},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "angry"
        assert result.confidence == 0.92
        assert result.arousal == 0.9
        assert result.valence == 0.2

    def test_detect_fearful(self, mock_detector, sample_audio):
        """Test detecting fearful emotion"""
        mock_detector._model.return_value = [
            {"label": "fearful", "score": 0.80},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "fearful"
        assert result.confidence == 0.80
        assert result.arousal == 0.7
        assert result.valence == 0.2

    def test_detect_surprised(self, mock_detector, sample_audio):
        """Test detecting surprised emotion"""
        mock_detector._model.return_value = [
            {"label": "surprised", "score": 0.85},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "surprised"
        assert result.confidence == 0.85
        assert result.arousal == 0.8
        assert result.valence == 0.6

    def test_detect_neutral(self, mock_detector, sample_audio):
        """Test detecting neutral emotion"""
        mock_detector._model.return_value = [
            {"label": "neutral", "score": 0.75},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.emotion == "neutral"
        assert result.confidence == 0.75
        assert result.arousal == 0.5
        assert result.valence == 0.5

    def test_detect_audio_format_conversion(self, mock_detector):
        """Test audio format conversion"""
        # Create audio with different dtype
        audio = np.ones(16000, dtype=np.float64)

        mock_detector._model.return_value = [
            {"label": "neutral", "score": 0.5},
        ]

        result = mock_detector.detect(audio)

        assert result.emotion == "neutral"
        # Verify model was called
        mock_detector._model.assert_called_once()

    def test_detect_audio_normalization(self, mock_detector):
        """Test audio normalization"""
        # Create audio with values > 1.0
        audio = np.ones(16000, dtype=np.float32) * 2.0

        mock_detector._model.return_value = [
            {"label": "happy", "score": 0.8},
        ]

        result = mock_detector.detect(audio)

        assert result.emotion == "happy"
        # Verify model was called
        mock_detector._model.assert_called_once()

    def test_detect_loads_model_if_needed(self, sample_audio):
        """Test that model is loaded on first detect call"""
        detector = make_emotion_detector()

        with patch.object(detector, "load") as mock_load:
            detector._model = MagicMock()
            detector._model_type = "wav2vec2_pipeline"
            detector._model.return_value = [
                {"label": "neutral", "score": 0.5},
            ]

            detector.detect(sample_audio)

            # Model was already set, so load shouldn't be called
            mock_load.assert_not_called()

    def test_detect_error_graceful_fallback(self, mock_detector, sample_audio):
        """Test graceful error handling"""
        mock_detector._model.side_effect = Exception("Model error")

        result = mock_detector.detect(sample_audio)

        # Should return neutral as fallback
        assert result.emotion == "neutral"
        assert result.confidence == 0.0
        assert "neutral" in result.all_emotions

    def test_detect_empty_predictions_fallback(self, mock_detector, sample_audio):
        """Test handling empty predictions"""
        mock_detector._model.return_value = []

        result = mock_detector.detect(sample_audio)

        # Should return neutral as fallback
        assert result.emotion == "neutral"

    def test_detect_processing_time(self, mock_detector, sample_audio):
        """Test that processing time is recorded"""
        mock_detector._model.return_value = [
            {"label": "happy", "score": 0.9},
        ]

        result = mock_detector.detect(sample_audio)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 1000  # Should be less than 1 second

    def test_detect_all_emotions_in_result(self, mock_detector, sample_audio):
        """Test that all emotions are included in result"""
        mock_detector._model.return_value = [
            {"label": "happy", "score": 0.6},
            {"label": "sad", "score": 0.2},
            {"label": "angry", "score": 0.1},
            {"label": "neutral", "score": 0.1},
        ]

        result = mock_detector.detect(sample_audio)

        assert "happy" in result.all_emotions
        assert "sad" in result.all_emotions
        assert "angry" in result.all_emotions
        assert "neutral" in result.all_emotions


class TestEmotionDetectorBatch:
    """Test suite for batch detection"""

    @pytest.fixture
    def mock_detector(self):
        """Create a detector with mocked model"""
        detector = make_emotion_detector()
        detector._model = MagicMock()
        detector._model_type = "wav2vec2_pipeline"
        return detector

    def test_batch_detect(self, mock_detector):
        """Test batch detection"""
        audio_samples = [
            np.zeros(16000, dtype=np.float32),
            np.ones(16000, dtype=np.float32),
            np.random.rand(16000).astype(np.float32),
        ]

        mock_detector._model.return_value = [
            [{"label": "happy", "score": 0.9}],
            [{"label": "sad", "score": 0.85}],
            [{"label": "angry", "score": 0.88}],
        ]

        results = mock_detector.batch_detect(audio_samples)

        assert len(results) == 3
        assert results[0].emotion == "happy"
        assert results[1].emotion == "sad"
        assert results[2].emotion == "angry"

    def test_batch_detect_empty_list(self, mock_detector):
        """Test batch detection with empty list"""
        results = mock_detector.batch_detect([])
        assert results == []


class TestEmotionDetectorUtils:
    """Test suite for utility methods"""

    def test_emotion_characteristics(self):
        """Test emotion characteristics mapping"""
        detector = make_emotion_detector()

        assert detector.EMOTION_CHARACTERISTICS["happy"]["arousal"] == 0.8
        assert detector.EMOTION_CHARACTERISTICS["happy"]["valence"] == 0.9

        assert detector.EMOTION_CHARACTERISTICS["sad"]["arousal"] == 0.3
        assert detector.EMOTION_CHARACTERISTICS["sad"]["valence"] == 0.1

        assert detector.EMOTION_CHARACTERISTICS["angry"]["arousal"] == 0.9
        assert detector.EMOTION_CHARACTERISTICS["angry"]["valence"] == 0.2

    def test_get_emotion_description(self):
        """Test emotion description translation"""
        detector = make_emotion_detector()

        assert detector.get_emotion_description("happy") == "Alegre"
        assert detector.get_emotion_description("sad") == "Triste"
        assert detector.get_emotion_description("angry") == "Enojado"
        assert detector.get_emotion_description("fearful") == "Asustado"
        assert detector.get_emotion_description("neutral") == "Neutral"
        assert detector.get_emotion_description("surprised") == "Sorprendido"
        assert detector.get_emotion_description("unknown") == "Desconocida"

    def test_create_neutral_result(self):
        """Test creating neutral fallback result"""
        detector = make_emotion_detector()
        import time

        start = time.perf_counter()
        result = detector._create_neutral_result(start)

        assert result.emotion == "neutral"
        assert result.confidence == 0.0
        assert result.arousal == 0.5
        assert result.valence == 0.5
        assert result.all_emotions["neutral"] == 1.0
        assert result.processing_time_ms >= 0
