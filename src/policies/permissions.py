"""Multi-user permission gating: niños no pueden controlar dominios sensibles.

Plan #3 OpenClaw — use case 2 (block puro, sin require_approval).
"""

from src.hooks import before_ha_action, BlockResult


# user_ids identificados como niños (por SpeakerID enrollment)
CHILD_USER_IDS: set[str] = {"niño1", "niño2"}

# Dominios HA que sólo pueden controlar adultos
ADULT_ONLY_DOMAINS: set[str] = {"climate", "lock", "alarm_control_panel"}


@before_ha_action(priority=5)  # priority=5: corre ANTES que safety_alarm (priority=10)
def chicos_sin_dominios_adultos(call):
    if call.user_id in CHILD_USER_IDS and call.domain in ADULT_ONLY_DOMAINS:
        return BlockResult(
            reason="No tenés permiso para eso",
            rule_name="chicos_sin_dominios_adultos",
        )
    return None
