"""
Mock GPU modules for testing without actual GPUs.

Allows running the full test suite in CI/CD environments
without CUDA hardware.
"""

from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class MockCUDADeviceProperties:
    """Mock CUDA device properties"""
    name: str = "Mock RTX 3070"
    total_memory: int = 8 * 1024 ** 3  # 8GB


class MockTensor:
    """Mock PyTorch tensor"""

    def __init__(self, data=None, shape=(1,), dtype="float32"):
        self.shape = shape
        self.dtype = dtype
        self._data = data if data is not None else np.zeros(shape)

    def cpu(self):
        return self

    def cuda(self, device=None):
        return self

    def to(self, device):
        return self

    def numpy(self):
        return self._data

    def __len__(self):
        return self.shape[0]


class MockCUDA:
    """Mock torch.cuda module"""

    _is_available = True
    _device_count = 4
    _memory_allocated = {}
    _memory_reserved = {}

    @classmethod
    def is_available(cls) -> bool:
        return cls._is_available

    @classmethod
    def device_count(cls) -> int:
        return cls._device_count

    @classmethod
    def get_device_properties(cls, device: int) -> MockCUDADeviceProperties:
        return MockCUDADeviceProperties(
            name=f"Mock RTX 3070 #{device}",
            total_memory=8 * 1024 ** 3
        )

    @classmethod
    def memory_allocated(cls, device: int = 0) -> int:
        return cls._memory_allocated.get(device, 0)

    @classmethod
    def memory_reserved(cls, device: int = 0) -> int:
        return cls._memory_reserved.get(device, 0)

    @classmethod
    def empty_cache(cls):
        cls._memory_allocated.clear()
        cls._memory_reserved.clear()

    @classmethod
    def synchronize(cls):
        pass

    @classmethod
    def set_available(cls, available: bool):
        """For testing scenarios without GPU"""
        cls._is_available = available

    @classmethod
    def set_device_count(cls, count: int):
        cls._device_count = count

    @classmethod
    def device(cls, device_id: int):
        """Context manager for device"""
        class DeviceContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return DeviceContext()


class MockTorchModule:
    """Mock PyTorch module (nn.Module-like)"""

    def __init__(self):
        self._device = "cpu"
        self._parameters = {}

    def to(self, device):
        self._device = device
        return self

    def cpu(self):
        self._device = "cpu"
        return self

    def cuda(self, device=None):
        self._device = f"cuda:{device}" if device else "cuda:0"
        return self

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def parameters(self):
        return iter(self._parameters.values())

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, *args, **kwargs):
        return MockTensor()


class MockWhisperModel(MockTorchModule):
    """Mock Whisper STT model"""

    def __init__(self, model_name: str = "base"):
        super().__init__()
        self.model_name = model_name

    def transcribe(self, audio, language="es"):
        return {
            "text": "prende la luz del living",
            "language": "es",
            "segments": []
        }


class MockTTSModel(MockTorchModule):
    """Mock TTS model"""

    def __init__(self, model_name: str = "vits"):
        super().__init__()
        self.model_name = model_name

    def synthesize(self, text: str) -> np.ndarray:
        # Return 1 second of silence at 22050 Hz
        return np.zeros(22050, dtype=np.float32)


class MockSpeakerIDModel(MockTorchModule):
    """Mock Speaker Identification model"""

    def __init__(self):
        super().__init__()
        self._embeddings = {}

    def encode_batch(self, audio) -> MockTensor:
        return MockTensor(shape=(1, 192))  # ECAPA-TDNN embedding size

    def verify(self, audio1, audio2) -> float:
        return 0.85  # Similarity score


class MockEmbeddingsModel(MockTorchModule):
    """Mock sentence embeddings model"""

    def __init__(self, model_name: str = "bge-small"):
        super().__init__()
        self.model_name = model_name

    def encode(self, texts, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(texts), 384).astype(np.float32)


class MockVLLM:
    """Mock vLLM for Router model"""

    def __init__(self, model: str, **kwargs):
        self.model = model

    def generate(self, prompts, **kwargs):
        class Output:
            def __init__(self):
                self.outputs = [type('obj', (object,), {'text': 'domotics'})()]

        return [Output() for _ in prompts]

    def shutdown(self):
        pass


class MockEmotionModel(MockTorchModule):
    """Mock emotion detection model"""

    EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        # Return mock logits for emotion classification
        logits = np.zeros((1, len(self.EMOTIONS)))
        logits[0, 0] = 2.0  # neutral highest
        return type('obj', (object,), {'logits': MockTensor(logits)})()


# ============================================================
# Patch helpers for tests
# ============================================================

def patch_torch_cuda():
    """Patch torch.cuda with mock"""
    return patch.dict('sys.modules', {
        'torch.cuda': MagicMock(**{
            'is_available': MockCUDA.is_available,
            'device_count': MockCUDA.device_count,
            'get_device_properties': MockCUDA.get_device_properties,
            'memory_allocated': MockCUDA.memory_allocated,
            'memory_reserved': MockCUDA.memory_reserved,
            'empty_cache': MockCUDA.empty_cache,
            'synchronize': MockCUDA.synchronize,
            'device': MockCUDA.device,
        })
    })


def create_mock_torch():
    """Create a complete mock torch module"""
    mock_torch = MagicMock()
    mock_torch.cuda = MockCUDA
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.Tensor = MockTensor
    return mock_torch
