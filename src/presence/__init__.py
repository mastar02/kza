"""
Presence Detection Module
Detecta presencia de personas usando BLE, WiFi y otros sensores.
"""

from src.presence.ble_scanner import BLEScanner, BLEDevice
from src.presence.presence_detector import (
    PresenceDetector,
    PresenceState,
    RoomOccupancy
)

__all__ = [
    "BLEScanner",
    "BLEDevice",
    "PresenceDetector",
    "PresenceState",
    "RoomOccupancy"
]
