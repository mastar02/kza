"""
Pytest configuration and shared fixtures.

This file provides:
- Mock GPU components for testing without CUDA hardware
- Mock Home Assistant client
- Sample data fixtures
- Configuration fixtures
- Global hook registry cleanup (Plan #3 OpenClaw)

Run tests with: pytest tests/ -v
Run safety tests only: pytest tests/safety/ -v
Run with coverage: pytest tests/ --cov=src --cov-report=html
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ==================== Pytest Configuration ====================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "safety: marks tests as safety-critical"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skipped without CUDA)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip GPU tests if torch.cuda is not available
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# ==================== Hook Cleanup (Plan #3 OpenClaw) ====================

@pytest.fixture(autouse=True)
def _clear_global_hook_registry():
    """Plan #3 OpenClaw — clear the global hook registry between tests.

    Decorators in src/policies/* register handlers into _global_registry at
    import time (which only happens once per process). Without this fixture,
    tests that import policy modules pollute global state for tests that run
    afterwards. This autouse fixture clears registrations (but NOT in-flight
    asyncio tasks, which are intentionally preserved per registry.clear()
    contract).
    """
    try:
        from src.hooks.registry import _global_registry
    except ImportError:
        # src.hooks not on path for some test environments — fine
        yield
        return

    _global_registry.clear()
    yield
    _global_registry.clear()


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_ha_client():
    """Mock Home Assistant client"""
    from tests.mocks.mock_ha_client import MockHomeAssistantClient
    return MockHomeAssistantClient()


@pytest.fixture
def mock_llm():
    """Mock LLM reasoner"""
    from tests.mocks.mock_llm import MockLLMReasoner
    return MockLLMReasoner()


@pytest.fixture
def mock_stt():
    """Mock Speech-to-Text"""
    stt = MagicMock()
    stt.load = MagicMock()
    stt.transcribe = MagicMock(return_value=("prende la luz", 50.0))
    return stt


@pytest.fixture
def mock_tts():
    """Mock Text-to-Speech"""
    tts = MagicMock()
    tts.speak = MagicMock()
    tts.synthesize = MagicMock(return_value=(MagicMock(), 30.0))
    return tts


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB sync"""
    chroma = MagicMock()
    chroma.initialize = MagicMock()
    chroma.get_stats = MagicMock(return_value={"commands_phrases": 100, "routines": 5})
    chroma.search_command = MagicMock(return_value={
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.living",
        "description": "Prendiendo luz del living",
        "similarity": 0.95
    })
    chroma.sync_commands = MagicMock(return_value=100)
    return chroma


@pytest.fixture
def mock_routine_manager():
    """Mock routine manager"""
    manager = MagicMock()
    manager.handle = AsyncMock(return_value={
        "handled": False,
        "response": "",
        "success": False
    })
    return manager


# ==================== Sample Data Fixtures ====================

@pytest.fixture
def sample_entities():
    """Sample Home Assistant entities"""
    return [
        {
            "entity_id": "light.living_room",
            "state": "off",
            "attributes": {
                "friendly_name": "Luz del Living",
                "brightness": 0
            }
        },
        {
            "entity_id": "climate.bedroom",
            "state": "cool",
            "attributes": {
                "friendly_name": "Aire del Dormitorio",
                "temperature": 22,
                "current_temperature": 24
            }
        },
        {
            "entity_id": "cover.blinds",
            "state": "closed",
            "attributes": {
                "friendly_name": "Persianas",
                "current_position": 0
            }
        }
    ]


@pytest.fixture
def sample_audio():
    """Sample audio data (silence)"""
    import numpy as np
    return np.zeros(16000, dtype=np.float32)  # 1 second of silence


@pytest.fixture
def sample_commands():
    """Sample voice commands for testing"""
    return [
        "prende la luz del living",
        "apaga el aire del dormitorio",
        "sube las persianas",
        "pon el aire a 22 grados",
        "sincroniza los comandos",
        "crea una rutina para cuando llegue a casa"
    ]


# ==================== Configuration Fixtures ====================

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "home_assistant": {
            "url": "http://localhost:8123",
            "token": "test_token",
            "timeout": 2.0
        },
        "stt": {
            "model": "distil-whisper/distil-small.en",
            "device": "cpu",
            "language": "es"
        },
        "tts": {
            "engine": "piper",
            "piper": {
                "model": "test_model.onnx"
            }
        },
        "vector_db": {
            "path": "/tmp/test_chroma",
            "search_threshold": 0.65
        },
        "latency_targets": {
            "total_ms": 300,
            "stt_ms": 150,
            "tts_ms": 80
        }
    }


# ==================== GPU Mock Fixtures ====================

@pytest.fixture
def mock_cuda():
    """Mock CUDA module"""
    from tests.mocks.mock_gpu import MockCUDA
    return MockCUDA


@pytest.fixture
def mock_torch():
    """Mock torch module with CUDA"""
    from tests.mocks.mock_gpu import create_mock_torch
    return create_mock_torch()


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for tests without GPU"""
    from src.pipeline.model_manager import ModelManager, ModelManagerConfig, ModelState

    manager = MagicMock(spec=ModelManager)
    manager.models = {
        "tts": MagicMock(state=ModelState.LOADED, gpu_id=0, vram_mb=2000),
        "stt": MagicMock(state=ModelState.LOADED, gpu_id=3, vram_mb=2500),
        "embeddings": MagicMock(state=ModelState.LOADED, gpu_id=1, vram_mb=500),
        "speaker_id": MagicMock(state=ModelState.LOADED, gpu_id=1, vram_mb=1500),
        "router": MagicMock(state=ModelState.LOADED, gpu_id=2, vram_mb=6000),
        "emotion": MagicMock(state=ModelState.LOADED, gpu_id=3, vram_mb=500),
    }
    manager.is_all_loaded.return_value = True
    manager.is_all_unloaded.return_value = False
    manager.get_vram_usage.return_value = {0: 2000, 1: 2000, 2: 6000, 3: 3000}
    manager.unload_all = AsyncMock(return_value={k: True for k in manager.models})
    manager.load_all = AsyncMock(return_value={k: True for k in manager.models})
    manager.reload_all = AsyncMock(return_value={k: True for k in manager.models})

    return manager


# ==================== Training Fixtures ====================

@pytest.fixture
def temp_training_dirs():
    """Create temporary directories for training tests"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dirs = {
            "output": base / "output",
            "data": base / "data",
            "contexts": base / "contexts",
            "conversations": base / "conversations",
        }

        for d in dirs.values():
            d.mkdir(parents=True)

        yield dirs


@pytest.fixture
def nightly_config(temp_training_dirs):
    """NightlyTrainer config for testing"""
    from src.training.nightly_trainer import NightlyConfig

    return NightlyConfig(
        training_hour=3,
        training_minute=0,
        min_samples_to_train=5,
        gpus=[0],
        use_distributed=False,
        use_qlora=True,
        base_model="test-model",
        epochs=1,
        batch_size=2,
        output_dir=str(temp_training_dirs["output"]),
        data_dir=str(temp_training_dirs["data"]),
        contexts_dir=str(temp_training_dirs["contexts"]),
        conversations_dir=str(temp_training_dirs["conversations"]),
    )


# ==================== Safety Test Fixtures ====================

@pytest.fixture
def permission_levels():
    """Permission level constants for safety tests"""
    return {
        "GUEST": 0,
        "CHILD": 1,
        "TEEN": 2,
        "ADULT": 3,
        "ADMIN": 4,
    }


@pytest.fixture
def domain_permissions():
    """Required permission levels for domains"""
    return {
        "light": 1,
        "switch": 1,
        "media_player": 1,
        "climate": 2,
        "cover": 2,
        "fan": 2,
        "lock": 3,
        "alarm_control_panel": 3,
        "camera": 3,
    }


@pytest.fixture
def mock_user_admin():
    """Mock admin user for testing"""
    return {
        "id": "user_admin",
        "name": "Admin",
        "permission_level": 4,
        "voice_embedding": [0.1] * 192,
    }


@pytest.fixture
def mock_user_child():
    """Mock child user for testing"""
    return {
        "id": "user_child",
        "name": "Niño",
        "permission_level": 1,
        "voice_embedding": [0.4] * 192,
    }


# ==================== Alert Fixtures ====================

@pytest.fixture
def mock_alert_manager():
    """Mock AlertManager"""
    manager = MagicMock()
    manager.add_alert = MagicMock(return_value="alert_001")
    manager.get_active_alerts.return_value = []
    manager.acknowledge_alert = MagicMock(return_value=True)
    return manager


# ==================== Context Fixtures ====================

@pytest.fixture
def sample_user_context():
    """Sample user context for testing"""
    import time

    return {
        "user_id": "user_001",
        "user_name": "Test User",
        "zone_id": "living_room",
        "conversation_history": [
            {
                "role": "user",
                "content": "prende la luz",
                "timestamp": time.time() - 60,
            },
            {
                "role": "assistant",
                "content": "Luz encendida.",
                "timestamp": time.time() - 55,
            },
        ],
        "preferences": {"default_volume": 50},
    }


# ==================== Home Assistant Fixtures ====================

@pytest.fixture
def sample_ha_entities():
    """Extended sample Home Assistant entities"""
    return [
        # Lights
        {"entity_id": "light.living_room", "state": "off", "attributes": {"friendly_name": "Luz Living"}},
        {"entity_id": "light.bedroom", "state": "on", "attributes": {"friendly_name": "Luz Dormitorio", "brightness": 200}},

        # Climate
        {"entity_id": "climate.main", "state": "cool", "attributes": {"friendly_name": "Aire Principal", "temperature": 22}},

        # Covers
        {"entity_id": "cover.blinds", "state": "closed", "attributes": {"friendly_name": "Persianas"}},
        {"entity_id": "cover.garage_door", "state": "closed", "attributes": {"friendly_name": "Puerta Garage"}},

        # Security
        {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "Cerradura Principal"}},
        {"entity_id": "alarm_control_panel.home", "state": "armed_away", "attributes": {"friendly_name": "Alarma"}},

        # Media
        {"entity_id": "media_player.spotify", "state": "paused", "attributes": {"friendly_name": "Spotify"}},
    ]
