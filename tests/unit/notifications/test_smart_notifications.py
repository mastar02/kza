"""Voice broadcast attempts are unique per notification and output zone."""

from types import SimpleNamespace

import pytest

from src.audio.zone_manager import Zone, ZoneManager
from src.notifications.smart_notifications import Notification, NotificationChannel, NotificationPriority
from src.pipeline.feature_manager import build_feature_manager


@pytest.mark.parametrize("first_delivery_ok", [True, False])
async def test_broadcast_attempts_each_zone_once_even_with_aliases_and_failed_output(first_delivery_ok):
    attempts = []

    async def speak(message, *, zone_id):
        attempts.append((message, zone_id))
        return first_delivery_ok or zone_id != "zone_pasillo"

    zone_manager = ZoneManager([Zone("pasillo", "Pasillo", 0, 1), Zone("cocina", "Cocina", 1, 2)])
    response = SimpleNamespace(aspeak=speak)
    user_zones = {"one": None, "two": "pasillo", "three": "zone_cocina"}
    users = SimpleNamespace(get_all_users=lambda: [SimpleNamespace(user_id=user_id) for user_id in user_zones])
    config = {"alerts": {"enabled": False}, "tts": {"streaming": {"default_output_zone": "zone_pasillo"}}}
    manager = build_feature_manager(config, response_handler=response, zone_manager=zone_manager,
                                    user_manager=users).notifications
    manager.presence = SimpleNamespace(is_user_home=lambda _: True, get_user_zone=user_zones.__getitem__)
    notice = Notification("notice", "", "Llegó el pedido.", priority=NotificationPriority.HIGH)

    await manager._deliver_notification(notice)

    assert attempts == [("Llegó el pedido.", "zone_pasillo"), ("Llegó el pedido.", "zone_cocina")]
    assert notice.channels_used == [NotificationChannel.VOICE] * (2 if first_delivery_ok else 1)
    assert notice.delivered
