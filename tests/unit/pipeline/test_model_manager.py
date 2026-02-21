"""
Tests for ModelManager - GPU model lifecycle management.

These tests ensure that:
1. Models can be loaded and unloaded correctly
2. VRAM is properly managed
3. State transitions are correct
4. Callbacks are triggered appropriately
5. The system fails gracefully when models aren't available

CRITICAL: These tests run without actual GPUs using mocks.
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Mock system-level modules BEFORE any imports
sys.modules['sounddevice'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['pyaudio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()

import pytest
import asyncio

from tests.mocks.mock_gpu import (
    MockCUDA, MockTorchModule, MockWhisperModel,
    MockTTSModel, MockSpeakerIDModel, MockEmbeddingsModel
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_torch():
    """Mock torch module"""
    mock = MagicMock()
    mock.cuda = MockCUDA
    mock.cuda.is_available.return_value = True
    mock.cuda.device_count.return_value = 4
    return mock


@pytest.fixture
def model_manager_config():
    """Test configuration for ModelManager"""
    from src.pipeline.model_manager import ModelManagerConfig
    return ModelManagerConfig(
        tts_gpu=0,
        embeddings_gpu=1,
        speaker_id_gpu=1,
        router_gpu=2,
        router_enabled=True,
        stt_gpu=3,
        emotion_gpu=3,
        emotion_enabled=True
    )


@pytest.fixture
def model_manager(model_manager_config):
    """Create ModelManager with mocked dependencies"""
    from src.pipeline.model_manager import ModelManager
    return ModelManager(config=model_manager_config)


# ============================================================
# Initialization Tests
# ============================================================

class TestModelManagerInit:
    """Tests for ModelManager initialization"""

    def test_init_creates_model_registry(self, model_manager):
        """ModelManager should initialize with correct model registry"""
        assert "tts" in model_manager.models
        assert "stt" in model_manager.models
        assert "embeddings" in model_manager.models
        assert "speaker_id" in model_manager.models
        assert "router" in model_manager.models
        assert "emotion" in model_manager.models

    def test_init_models_start_unloaded(self, model_manager):
        """All models should start in UNLOADED state"""
        from src.pipeline.model_manager import ModelState

        for name, info in model_manager.models.items():
            assert info.state == ModelState.UNLOADED, f"{name} should be UNLOADED"
            assert info.model_instance is None

    def test_init_assigns_correct_gpus(self, model_manager):
        """Models should be assigned to correct GPUs"""
        assert model_manager.models["tts"].gpu_id == 0
        assert model_manager.models["embeddings"].gpu_id == 1
        assert model_manager.models["speaker_id"].gpu_id == 1
        assert model_manager.models["router"].gpu_id == 2
        assert model_manager.models["stt"].gpu_id == 3
        assert model_manager.models["emotion"].gpu_id == 3

    def test_init_without_router(self):
        """Can initialize without router model"""
        from src.pipeline.model_manager import ModelManager, ModelManagerConfig

        config = ModelManagerConfig(router_enabled=False)
        manager = ModelManager(config)

        assert "router" not in manager.models

    def test_init_without_emotion(self):
        """Can initialize without emotion model"""
        from src.pipeline.model_manager import ModelManager, ModelManagerConfig

        config = ModelManagerConfig(emotion_enabled=False)
        manager = ModelManager(config)

        assert "emotion" not in manager.models


# ============================================================
# State Management Tests
# ============================================================

class TestModelStateManagement:
    """Tests for model state transitions"""

    def test_state_callback_on_change(self, model_manager_config):
        """State change callback should be triggered"""
        from src.pipeline.model_manager import ModelManager, ModelState

        callback_calls = []

        def on_state_change(model_name, state):
            callback_calls.append((model_name, state))

        manager = ModelManager(
            config=model_manager_config,
            on_state_change=on_state_change
        )

        # Simulate state change
        manager._notify_state_change("tts", ModelState.LOADING)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("tts", ModelState.LOADING)

    def test_is_all_loaded_false_initially(self, model_manager):
        """is_all_loaded should return False when no models loaded"""
        assert model_manager.is_all_loaded() is False

    def test_is_all_unloaded_true_initially(self, model_manager):
        """is_all_unloaded should return True initially"""
        assert model_manager.is_all_unloaded() is True


# ============================================================
# VRAM Tracking Tests
# ============================================================

class TestVRAMTracking:
    """Tests for VRAM usage tracking"""

    def test_vram_usage_empty_initially(self, model_manager):
        """VRAM usage should be empty when no models loaded"""
        usage = model_manager.get_vram_usage()
        assert usage == {}

    def test_status_includes_vram_info(self, model_manager):
        """Status should include VRAM information"""
        status = model_manager.get_status()

        assert "total_vram_mb" in status
        assert status["total_vram_mb"] == 0  # No models loaded

        assert "models" in status
        for name, info in status["models"].items():
            assert "vram_mb" in info
            assert "gpu" in info


# ============================================================
# Load/Unload Tests (with mocks)
# ============================================================

class TestModelLoading:
    """Tests for model loading with mocked dependencies"""

    @pytest.mark.asyncio
    async def test_load_model_changes_state(self, model_manager):
        """Loading a model should change its state"""
        from src.pipeline.model_manager import ModelState

        # Mock the actual loader to avoid real model loading
        with patch.object(model_manager, '_load_tts', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = MockTTSModel()

            success = await model_manager._load_model("tts")

            assert success is True
            assert model_manager.models["tts"].state == ModelState.LOADED
            assert model_manager.models["tts"].model_instance is not None

    @pytest.mark.asyncio
    async def test_load_model_failure_sets_error_state(self, model_manager):
        """Failed model loading should set ERROR state"""
        from src.pipeline.model_manager import ModelState

        with patch.object(model_manager, '_load_tts', new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = RuntimeError("GPU out of memory")

            success = await model_manager._load_model("tts")

            assert success is False
            assert model_manager.models["tts"].state == ModelState.ERROR
            assert "GPU out of memory" in model_manager.models["tts"].error_message

    @pytest.mark.asyncio
    async def test_load_unknown_model_fails(self, model_manager):
        """Loading unknown model should fail gracefully"""
        success = await model_manager._load_model("nonexistent_model")
        assert success is False

    @pytest.mark.asyncio
    async def test_load_already_loaded_model_is_noop(self, model_manager):
        """Loading already loaded model should be a no-op"""
        from src.pipeline.model_manager import ModelState

        # Pre-set model as loaded
        model_manager.models["tts"].state = ModelState.LOADED
        model_manager.models["tts"].model_instance = MockTTSModel()

        success = await model_manager._load_model("tts")

        assert success is True  # Should succeed without reloading


class TestModelUnloading:
    """Tests for model unloading"""

    @pytest.mark.asyncio
    async def test_unload_model_clears_instance(self, model_manager):
        """Unloading should clear model instance"""
        from src.pipeline.model_manager import ModelState

        # Pre-set model as loaded
        model_manager.models["tts"].state = ModelState.LOADED
        model_manager.models["tts"].model_instance = MockTTSModel()

        success = await model_manager._unload_model("tts")

        assert success is True
        assert model_manager.models["tts"].state == ModelState.UNLOADED
        assert model_manager.models["tts"].model_instance is None

    @pytest.mark.asyncio
    async def test_unload_already_unloaded_is_noop(self, model_manager):
        """Unloading already unloaded model should be a no-op"""
        from src.pipeline.model_manager import ModelState

        model_manager.models["tts"].state = ModelState.UNLOADED

        success = await model_manager._unload_model("tts")

        assert success is True

    @pytest.mark.asyncio
    async def test_unload_all_clears_all_models(self, model_manager):
        """unload_all should clear all models"""
        from src.pipeline.model_manager import ModelState

        # Pre-set some models as loaded
        for name in ["tts", "stt", "embeddings"]:
            model_manager.models[name].state = ModelState.LOADED
            model_manager.models[name].model_instance = MockTorchModule()

        # Mock CUDA cache clearing
        with patch.object(model_manager, '_clear_cuda_cache'):
            results = await model_manager.unload_all()

        # All should be unloaded now
        assert model_manager.is_all_unloaded()
        assert all(results.values())


# ============================================================
# Integration Tests
# ============================================================

class TestLoadUnloadCycle:
    """Tests for complete load/unload cycles"""

    @pytest.mark.asyncio
    async def test_full_cycle_for_nightly_training(self, model_manager):
        """Simulate the nightly training cycle: load → use → unload → reload"""
        from src.pipeline.model_manager import ModelState

        # Mock all loaders
        async def mock_loader():
            return MockTorchModule()

        for method in ['_load_tts', '_load_stt', '_load_embeddings',
                       '_load_speaker_id', '_load_router', '_load_emotion']:
            with patch.object(model_manager, method, new_callable=AsyncMock) as m:
                m.return_value = MockTorchModule()

        # Phase 1: Load all (daytime operation)
        with patch.object(model_manager, '_clear_cuda_cache'):
            # Manually set models as loaded for this test
            for name in model_manager.models:
                model_manager.models[name].state = ModelState.LOADED
                model_manager.models[name].model_instance = MockTorchModule()

            assert model_manager.is_all_loaded()

            # Phase 2: Unload for training
            await model_manager.unload_all()
            assert model_manager.is_all_unloaded()

            # Phase 3: Training would happen here...

            # Phase 4: Reload for morning
            for name in model_manager.models:
                model_manager.models[name].state = ModelState.LOADED
                model_manager.models[name].model_instance = MockTorchModule()

            assert model_manager.is_all_loaded()


# ============================================================
# Concurrency Tests
# ============================================================

class TestConcurrency:
    """Tests for concurrent operations"""

    @pytest.mark.asyncio
    async def test_no_concurrent_loading(self, model_manager):
        """Should prevent concurrent load_all calls"""
        model_manager._loading = True

        results = await model_manager.load_all()

        assert results == {}  # Should return empty, not try to load

    @pytest.mark.asyncio
    async def test_no_concurrent_unloading(self, model_manager):
        """Should prevent concurrent unload_all calls"""
        model_manager._unloading = True

        results = await model_manager.unload_all()

        assert results == {}  # Should return empty


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_get_model_returns_none_for_unloaded(self, model_manager):
        """get_model should return None for unloaded models"""
        result = model_manager.get_model("tts")
        assert result is None

    def test_get_model_returns_none_for_unknown(self, model_manager):
        """get_model should return None for unknown models"""
        result = model_manager.get_model("nonexistent")
        assert result is None

    def test_callback_error_doesnt_crash(self, model_manager_config):
        """Callback errors should not crash the manager"""
        from src.pipeline.model_manager import ModelManager, ModelState

        def bad_callback(model_name, state):
            raise RuntimeError("Callback error")

        manager = ModelManager(
            config=model_manager_config,
            on_state_change=bad_callback
        )

        # Should not raise
        manager._notify_state_change("tts", ModelState.LOADING)
