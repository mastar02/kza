"""
Sensitive action registry for confirmation gating.

Algunas combinaciones de intent+entity tienen impacto que no queremos
ejecutar "a ciegas" si la confidence es baja: cerrar persianas de noche,
apagar el aire en invierno, cortar música en una reunión, etc.

El RequestRouter consulta `is_sensitive(intent, entity)` y, si devuelve True
con confidence < threshold, pide confirmación en vez de ejecutar.

Extender `SENSITIVE_COMBOS` con más tuplas según evolucione el use case.
"""
from __future__ import annotations

# Combos sensibles: (intent, entity_domain).
#   - turn_off + climate → apagar aire/calefacción (puede dejar la casa fría).
#   - set_cover_position + cover → persianas (ambiguo arriba/abajo, y de noche
#     puede comprometer privacidad/temperatura).
#   - turn_off + media_player → cortar música/entretenimiento activo.
SENSITIVE_COMBOS: set[tuple[str, str]] = {
    ("turn_off", "climate"),
    ("set_cover_position", "cover"),
    ("turn_off", "media_player"),
}


def is_sensitive(intent: str | None, entity: str | None) -> bool:
    """
    ¿Requiere este comando confirmación explícita cuando la confidence es baja?

    Args:
        intent: Intent detectado (ej: "turn_on", "turn_off", "set_cover_position").
        entity: Entity domain detectado (ej: "light", "climate", "cover").

    Returns:
        True si `(intent, entity)` está en `SENSITIVE_COMBOS`; False si alguno
        de los dos es None o el combo no está registrado.
    """
    if not intent or not entity:
        return False
    return (intent, entity) in SENSITIVE_COMBOS
