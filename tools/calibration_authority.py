"""Compatibility facade for the reviewed native calibration authority client."""
from src.audio.native_capture_authority import (
    CalibrationAuthorityClient as CalibrationAuthorityClient,
    _CAPABILITY as _CAPABILITY,
    _UNCONFIRMED as _UNCONFIRMED,
    _UNCONFIRMED_LOCK as _UNCONFIRMED_LOCK,
    _origin as _origin,
    requests as requests,
    sys as sys,
    threading as threading,
)

__all__ = ["CalibrationAuthorityClient"]
