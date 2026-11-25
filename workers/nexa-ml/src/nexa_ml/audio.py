"""PCM payload validation shared by STT and speaker slots (no audio ever logged)."""

import numpy as np

from nexa_ml.slot import InvalidInputError

WHISPER_SAMPLE_RATE = 16_000


def decode_pcm_f32le(
    payload: bytes,
    sample_rate: int,
    *,
    expected_rate: int = WHISPER_SAMPLE_RATE,
    min_seconds: float = 0.0,
    max_seconds: float,
) -> np.ndarray:
    """Decode mono little-endian float32 PCM into a writable ``float32`` array.

    Args:
        payload: Raw ``pcm_f32le`` bytes.
        sample_rate: Declared sample rate of the payload.
        expected_rate: Only accepted rate (the worker never resamples).
        min_seconds: Minimum accepted duration.
        max_seconds: Maximum accepted duration.

    Returns:
        A contiguous, writable ``float32`` array in native byte order.

    Raises:
        InvalidInputError: Wrong rate, empty/misaligned payload, out-of-range
            duration or non-finite samples. Messages never include sample data.
    """
    if sample_rate != expected_rate:
        raise InvalidInputError(f"sample_rate must be {expected_rate}")
    if not payload:
        raise InvalidInputError("pcm_f32le is empty")
    if len(payload) % 4:
        raise InvalidInputError("pcm_f32le length must be a multiple of 4 bytes")
    samples = len(payload) // 4
    seconds = samples / expected_rate
    if seconds > max_seconds:
        raise InvalidInputError(f"audio longer than {max_seconds:.0f} s")
    if seconds < min_seconds:
        raise InvalidInputError(f"audio shorter than {min_seconds:.2f} s")
    audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(audio).all():
        raise InvalidInputError("pcm_f32le contains non-finite samples")
    return audio


def parse_device(device: str) -> tuple[str, int]:
    """Split ``cuda:N`` into ``("cuda", N)``; ``cpu`` becomes ``("cpu", 0)``."""
    kind, _, index = device.partition(":")
    return kind, int(index) if index else 0
