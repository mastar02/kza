"""
Room Context Module
Gestión de contexto por habitación fusionando micrófono + Bluetooth.
"""

from src.rooms.room_context import (
    RoomConfig,
    RoomContext,
    RoomContextManager,
    ContextSource,
    create_default_rooms,
    auto_detect_microphones,
    auto_detect_bt_adapters,
    list_bt_adapters,
    resolve_bt_adapter,
)

__all__ = [
    "RoomConfig",
    "RoomContext",
    "RoomContextManager",
    "ContextSource",
    "create_default_rooms",
    "auto_detect_microphones",
    "auto_detect_bt_adapters",
    "list_bt_adapters",
    "resolve_bt_adapter",
]
