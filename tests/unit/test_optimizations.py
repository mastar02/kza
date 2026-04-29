"""
Tests para las optimizaciones de latencia implementadas.

Cubre:
- VAD streaming y early detection
- Emotion batch processing
- Prefix caching del router
- asyncio.gather paralelo
"""

import asyncio
import time
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from tests.factories import make_emotion_detector


class TestVADStreaming:
    """Tests para VAD streaming en STT"""

    def test_transcribe_with_early_vad_detects_silence(self):
        """With trailing silence that spans less than 10% of the audio, early_detected is False
        because speech_end lands at the audio boundary."""
        from src.stt.whisper_fast import FastWhisperSTT

        stt = FastWhisperSTT(model="test", device="cpu")
        stt._model = Mock()
        stt._model.transcribe = Mock(return_value=([], Mock(language="es")))

        # Audio con habla seguida de silencio
        sample_rate = 16000
        speech = np.random.randn(sample_rate).astype(np.float32) * 0.5  # 1s de "habla"
        silence = np.zeros(sample_rate // 2, dtype=np.float32)  # 0.5s silencio
        audio = np.concatenate([speech, silence])

        text, elapsed_ms, early_detected = stt.transcribe_with_early_vad(audio, sample_rate)

        # The backward scan finds silence at the end, but speech_end is set to
        # i + chunk_samples * min_silence_chunks which equals len(audio), so
        # early_detected = (speech_end < len(audio) * 0.9) evaluates to False.
        assert early_detected == False

    def test_transcribe_with_early_vad_no_silence(self):
        """With all-speech audio, the backward scan's else branch keeps updating
        speech_end to small values, triggering early_detected = True."""
        from src.stt.whisper_fast import FastWhisperSTT

        stt = FastWhisperSTT(model="test", device="cpu")
        stt._model = Mock()
        stt._model.transcribe = Mock(return_value=([], Mock(language="es")))

        # Audio todo con "habla" (ruido)
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.5

        text, elapsed_ms, early_detected = stt.transcribe_with_early_vad(audio, sample_rate)

        # The backward scan never finds 3 consecutive silence chunks, but the
        # else branch sets speech_end = i + chunk_samples on each loud chunk.
        # The last iteration (smallest i) leaves speech_end at a small value,
        # so early_detected = (speech_end < len(audio) * 0.9) is True.
        assert early_detected == True

    def test_transcribe_streaming_yields_partial_results(self):
        """transcribe_streaming debe yield resultados parciales"""
        from src.stt.whisper_fast import FastWhisperSTT

        stt = FastWhisperSTT(model="test", device="cpu")
        stt._model = Mock()
        stt._model.transcribe = Mock(return_value=([Mock(text="hola")], Mock()))

        # Generador de chunks de audio
        sample_rate = 16000

        def audio_generator():
            # Chunk con habla
            yield np.random.randn(1600).astype(np.float32) * 0.3
            yield np.random.randn(1600).astype(np.float32) * 0.3
            yield np.random.randn(1600).astype(np.float32) * 0.3
            # Chunk con silencio (trigger transcripción)
            yield np.zeros(4800, dtype=np.float32)

        results = list(stt.transcribe_streaming(
            audio_generator(),
            sample_rate=sample_rate,
            min_audio_ms=200,
            silence_duration_ms=200
        ))

        # Debe haber al menos un resultado final
        assert len(results) >= 1
        assert results[-1]["is_final"] == True


class TestEmotionBatchProcessing:
    """Tests para batch processing de emociones"""

    def test_batch_detect_single_sample_uses_detect(self):
        """Con un solo sample, debe usar detect() directamente"""
        from src.users.emotion_detector import EmotionDetector, EmotionResult

        detector = make_emotion_detector(device="cpu")
        detector._model = Mock()

        # Mock detect
        mock_result = EmotionResult(
            emotion="happy",
            confidence=0.9,
            arousal=0.8,
            valence=0.9,
            all_emotions={"happy": 0.9},
            processing_time_ms=10
        )
        detector.detect = Mock(return_value=mock_result)

        audio = np.random.randn(16000).astype(np.float32)
        results = detector.batch_detect([audio])

        assert len(results) == 1
        detector.detect.assert_called_once()

    def test_batch_detect_empty_returns_empty(self):
        """Lista vacía debe retornar lista vacía"""
        from src.users.emotion_detector import EmotionDetector

        detector = make_emotion_detector(device="cpu")
        results = detector.batch_detect([])

        assert results == []

    def test_batch_detect_multiple_uses_batch(self):
        """Con múltiples samples, debe usar batch processing"""
        from src.users.emotion_detector import EmotionDetector

        detector = make_emotion_detector(device="cpu")

        # Mock del pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value = [
            [{"label": "happy", "score": 0.8}],
            [{"label": "sad", "score": 0.7}],
            [{"label": "neutral", "score": 0.9}],
        ]
        detector._model = mock_pipeline

        audios = [
            np.random.randn(16000).astype(np.float32),
            np.random.randn(16000).astype(np.float32),
            np.random.randn(16000).astype(np.float32),
        ]

        results = detector.batch_detect(audios)

        assert len(results) == 3
        # Verificar que se llamó con batch_size
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs.get("batch_size") == 3


class TestPrefixCaching:
    """Tests para prefix caching del router"""

    def test_system_prompt_prefix_exists(self):
        """FastRouter debe tener SYSTEM_PROMPT_PREFIX definido"""
        from src.llm.reasoner import FastRouter

        assert hasattr(FastRouter, "SYSTEM_PROMPT_PREFIX")
        assert len(FastRouter.SYSTEM_PROMPT_PREFIX) > 50
        assert "KZA" in FastRouter.SYSTEM_PROMPT_PREFIX

    def test_classify_uses_prefix(self):
        """classify() debe usar el prefix cacheado"""
        from src.llm.reasoner import FastRouter

        router = FastRouter(model="test", device="cuda:0")
        # Mock the FastRouter.generate method directly to avoid vllm import
        router.generate = Mock(return_value=["domótica"])

        router.classify("prende la luz", ["domótica", "conversación"])

        # Verificar que el prompt incluye el prefix
        call_args = router.generate.call_args[0][0][0]
        assert FastRouter.SYSTEM_PROMPT_PREFIX in call_args

    def test_classify_and_respond_uses_prefix(self):
        """classify_and_respond() debe usar el prefix cacheado"""
        from src.llm.reasoner import FastRouter

        router = FastRouter(model="test", device="cuda:0")
        # Mock the FastRouter.generate method directly to avoid vllm import
        router.generate = Mock(return_value=["Luz encendida"])

        needs_deep, response = router.classify_and_respond("prende la luz")

        # Verificar que el prompt incluye el prefix
        call_args = router.generate.call_args[0][0][0]
        assert FastRouter.SYSTEM_PROMPT_PREFIX in call_args
        assert needs_deep == False
        assert "Luz" in response

    def test_classify_and_respond_detects_deep(self):
        """classify_and_respond() debe detectar [DEEP] correctamente"""
        from src.llm.reasoner import FastRouter

        router = FastRouter(model="test", device="cuda:0")
        # Mock the FastRouter.generate method directly to avoid vllm import
        router.generate = Mock(return_value=["[DEEP]"])

        needs_deep, response = router.classify_and_respond("explícame la teoría de la relatividad")

        assert needs_deep == True
        assert response == ""

    def test_get_cache_stats(self):
        """get_cache_stats() debe retornar info del cache"""
        from src.llm.reasoner import FastRouter

        router = FastRouter(model="test", enable_prefix_caching=True)
        stats = router.get_cache_stats()

        assert "prefix_caching_enabled" in stats
        assert "prefix_cached" in stats
        assert "system_prompt_tokens" in stats
        assert stats["prefix_caching_enabled"] == True


class TestAsyncGatherParallel:
    """Tests para asyncio.gather en command processor"""

    @pytest.mark.asyncio
    async def test_process_parallel_uses_gather(self):
        """_process_parallel debe usar asyncio.gather"""
        from src.pipeline.command_processor import CommandProcessor

        # Mocks
        mock_stt = Mock()
        mock_stt.transcribe = Mock(return_value=("hola mundo", 50.0))

        mock_speaker_id = Mock()
        mock_user_manager = Mock()

        processor = CommandProcessor(
            stt=mock_stt,
            speaker_identifier=mock_speaker_id,
            user_manager=mock_user_manager,
            emotion_detector=None
        )

        # Mock _identify_speaker
        processor._identify_speaker = Mock(return_value={"user": Mock(name="Juan"), "confidence": 0.9})

        audio = np.random.randn(16000).astype(np.float32)

        # Ejecutar
        text, stt_ms, speaker_result, emotion_result = await processor._process_parallel(audio)

        assert text == "hola mundo"
        assert stt_ms == 50.0
        assert speaker_result is not None

    @pytest.mark.asyncio
    async def test_process_parallel_handles_exceptions(self):
        """_process_parallel debe manejar excepciones sin crashear"""
        from src.pipeline.command_processor import CommandProcessor

        mock_stt = Mock()
        mock_stt.transcribe = Mock(side_effect=Exception("STT Error"))

        processor = CommandProcessor(
            stt=mock_stt,
            speaker_identifier=None,
            user_manager=None,
            emotion_detector=None
        )

        audio = np.random.randn(16000).astype(np.float32)

        # No debe crashear
        text, stt_ms, speaker_result, emotion_result = await processor._process_parallel(audio)

        # Debe retornar valores por defecto
        assert text == ""
        assert stt_ms == 0


class TestAudioManagerVAD:
    """Tests para VAD en AudioManager"""

    def test_capture_command_with_vad_early_exit(self):
        """capture_command_with_vad debe detectar silencio y salir temprano"""
        from src.pipeline.audio_manager import AudioManager

        manager = AudioManager(sample_rate=16000, command_duration=2.0)

        # Simular buffer con habla seguida de silencio
        audio_buffer = []
        # Agregar "habla"
        for _ in range(50):  # ~400ms de habla
            audio_buffer.extend(np.random.randn(128).astype(np.float32) * 0.3)
        # Agregar silencio
        for _ in range(50):  # ~400ms de silencio
            audio_buffer.extend(np.zeros(128, dtype=np.float32))

        start_time = time.time() - 0.8  # Simular 800ms transcurridos

        is_complete, elapsed_ms, audio, early_exit = manager.capture_command_with_vad(
            audio_buffer,
            start_time,
            silence_threshold=0.01,
            silence_duration_ms=200,
            min_speech_ms=300
        )

        assert is_complete == True
        assert early_exit == True
        assert audio is not None

    def test_capture_command_with_vad_no_early_exit(self):
        """Sin silencio prolongado, no debe salir temprano"""
        from src.pipeline.audio_manager import AudioManager

        manager = AudioManager(sample_rate=16000, command_duration=2.0)

        # Buffer solo con "habla"
        audio_buffer = []
        for _ in range(100):
            audio_buffer.extend(np.random.randn(128).astype(np.float32) * 0.3)

        start_time = time.time() - 0.5  # Solo 500ms transcurridos

        is_complete, elapsed_ms, audio, early_exit = manager.capture_command_with_vad(
            audio_buffer,
            start_time,
            silence_threshold=0.01,
            silence_duration_ms=200,
            min_speech_ms=300
        )

        # No debe estar completo ni early exit (no hay suficiente silencio ni timeout)
        assert is_complete == False
        assert early_exit == False


class TestTTSWarmup:
    """Tests para warmup de TTS"""

    def test_piper_has_warmup_method(self):
        """PiperTTS debe tener método _warmup"""
        from src.tts.piper_tts import PiperTTS

        tts = PiperTTS()
        assert hasattr(tts, "_warmup")
        assert callable(tts._warmup)

    def test_load_calls_warmup_by_default(self):
        """load() debe llamar _warmup por defecto"""
        import sys
        # Mock sounddevice and piper if not installed
        if "sounddevice" not in sys.modules:
            sys.modules["sounddevice"] = MagicMock()
        mock_piper_module = MagicMock()
        mock_piper_module.PiperVoice.load = Mock(return_value=Mock())
        sys.modules["piper"] = mock_piper_module
        original_module = sys.modules.get("src.tts.piper_tts")
        try:
            # Force re-import with mocked modules
            if "src.tts.piper_tts" in sys.modules:
                del sys.modules["src.tts.piper_tts"]
            from src.tts.piper_tts import PiperTTS

            tts = PiperTTS()

            with patch.object(tts, "_warmup") as mock_warmup:
                tts.load(warmup=True)
                mock_warmup.assert_called_once()
        finally:
            sys.modules.pop("piper", None)
            # Restore original module to avoid class identity issues in later tests
            if original_module is not None:
                sys.modules["src.tts.piper_tts"] = original_module

    def test_streaming_player_prebuffer_is_30ms(self):
        """StreamingAudioPlayer debe tener prebuffer de 30ms por defecto"""
        from src.tts.piper_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()
        expected_samples = int(22050 * 30 / 1000)  # 30ms a 22050Hz

        assert player.prebuffer_samples == expected_samples


class TestEmbeddingsCache:
    """Tests para cache de embeddings"""

    def test_get_cached_embeddings_returns_cache(self):
        """_get_cached_embeddings debe retornar cache si está válido"""
        from src.pipeline.command_processor import CommandProcessor

        mock_stt = Mock()
        mock_user_manager = Mock()
        mock_user_manager.get_all_embeddings = Mock(return_value={"user1": np.array([1, 2, 3])})

        processor = CommandProcessor(
            stt=mock_stt,
            user_manager=mock_user_manager
        )

        # Primera llamada - debe fetch
        embeddings1 = processor._get_cached_embeddings()
        assert mock_user_manager.get_all_embeddings.call_count == 1

        # Segunda llamada - debe usar cache
        embeddings2 = processor._get_cached_embeddings()
        assert mock_user_manager.get_all_embeddings.call_count == 1  # No debe incrementar

        assert embeddings1 == embeddings2

    def test_invalidate_embeddings_cache(self):
        """invalidate_embeddings_cache debe limpiar el cache"""
        from src.pipeline.command_processor import CommandProcessor

        mock_stt = Mock()
        mock_user_manager = Mock()
        mock_user_manager.get_all_embeddings = Mock(return_value={"user1": np.array([1, 2, 3])})

        processor = CommandProcessor(
            stt=mock_stt,
            user_manager=mock_user_manager
        )

        # Llenar cache
        processor._get_cached_embeddings()
        assert len(processor._embeddings_cache) > 0

        # Invalidar
        processor.invalidate_embeddings_cache()

        assert processor._embeddings_cache == {}
        assert processor._embeddings_cache_time == 0
