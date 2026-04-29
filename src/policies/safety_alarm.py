"""Safety policy: nunca desarmar la alarma entre las 22:00 y las 07:00.

Plan #3 OpenClaw — use case 1.
"""

from datetime import datetime

from src.hooks import before_ha_action, BlockResult


@before_ha_action(priority=10)
def proteger_alarma_de_noche(call):
    if call.entity_id == "alarm_control_panel.casa" and call.service == "alarm_disarm":
        h = datetime.now().hour
        if h >= 22 or h < 7:
            return BlockResult(
                reason="No puedo desarmar la alarma a esta hora",
                rule_name="proteger_alarma_de_noche",
            )
    return None
