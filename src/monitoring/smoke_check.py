"""Smoke check en dry-run de la cadena texto → entidad → payload.

Verifica que un comando canónico llegaría a ejecutarse, SIN ejecutarlo:
resuelve la frase contra el índice vectorial, comprueba que la entidad exista
y esté viva en Home Assistant, y valida que el `service_data` resultante sea
algo que HA vaya a aceptar.

Por qué existe (2026-07-30): tres fallos distintos llegaron a producción sin
que nada avisara, y en los tres el diagnóstico empezó con el usuario diciendo
"no funciona":

- un slot `entity` inyectado por el LLM hacía que HA rechazara el payload
  entero (``extra keys not allowed @ data['entity']``);
- un ``--wipe`` de ChromaDB sin la API key dejó el índice con solo escenas, y
  todas las luces dejaron de resolver;
- entidades que quedan ``unavailable`` aceptan la llamada y no hacen nada.

Los tres son detectables sin tocar el hogar. Este módulo es la parte pura y
testeable; ``tools/smoke_test.py`` la cablea contra el sistema real.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.nlu.slot_extractor import VALID_SERVICE_DATA_SLOTS

# Etapas de la cadena, en el orden en que se recorren. Se reporta la PRIMERA
# que falla: las siguientes son consecuencia, no causa.
STAGE_OK = "ok"
STAGE_VECTOR_SEARCH = "vector_search"
STAGE_ENTITY = "entity"
STAGE_PAYLOAD = "payload"


@dataclass
class SmokeResult:
    """Resultado de verificar una frase canónica."""

    phrase: str
    ok: bool
    stage: str
    detail: str
    entity_id: str | None = None
    service_data: dict | None = None
    similarity: float | None = None


def invalid_payload_keys(service_data: dict | None) -> list[str]:
    """Claves del payload que HA rechazaría, ordenadas.

    HA responde ``invalid_format`` y descarta el service_data COMPLETO ante
    una sola clave desconocida, así que cualquier sobrante anula el comando.
    """
    return sorted(set(service_data or {}) - VALID_SERVICE_DATA_SLOTS)


def indexed_entity_ids(collection) -> list[str]:
    """Los entity_id realmente direccionables por voz, según el índice.

    La cobertura del smoke test tiene que salir de lo que el sistema expone
    (ChromaDB), no de una lista escrita a mano: el `default_light` de cada
    room dejaba fuera entidades vivas del índice (2026-07-31: cuarto/balcón/
    escalera no eran `default_light` de ninguna room), así que el smoke test
    salía verde para ellas pese a resolver por voz con similitud 0.92-1.00.

    Args:
        collection: colección de Chroma con comandos indexados (o cualquier
            objeto con un método `.get(include=[...])` que devuelva
            `{"metadatas": [...]}` — así queda testeable sin ChromaDB real).
    """
    got = collection.get(include=["metadatas"])
    ids = {
        m.get("entity_id")
        for m in (got.get("metadatas") or [])
        if m and m.get("entity_id")
    }
    return sorted(ids)


def indexed_entity_ids_or_problem(collection) -> tuple[list[str], str | None]:
    """Como `indexed_entity_ids`, pero nunca deja escapar una excepción.

    Un tool de diagnóstico que explota con un traceback sin manejar es en sí
    mismo un fallo silencioso a medias: el operador ve un stacktrace en vez
    de un veredicto claro, y si el caller no lo recablea con cuidado el
    smoke test puede terminar contando eso como "no hay problemas" (nunca
    llegó a incrementar el contador de fallos porque la excepción cortó el
    flujo antes). Envolver el try/except acá, en vez de en `tools/smoke_test.py`,
    lo vuelve testeable: `tools/` no se importa desde los tests.

    Args:
        collection: mismo contrato que `indexed_entity_ids`.

    Returns:
        `(entity_ids, problema)`. Si la colección responde bien, `problema`
        es `None`. Si `collection.get(...)` levanta cualquier excepción,
        `entity_ids` queda vacío y `problema` describe la falla — el caller
        debe tratar un `problema` no-`None` como un fallo del smoke test,
        igual que hace con `entity_problem`.
    """
    try:
        return indexed_entity_ids(collection), None
    except Exception as e:  # noqa: BLE001 — diagnóstico, no debe propagar
        return [], f"no pude leer el índice de Chroma: {e}"


def entity_problem(entity_id: str, ha_states: dict[str, str]) -> str | None:
    """Describe por qué `entity_id` no serviría, o None si está sana.

    Args:
        entity_id: entidad resuelta por el vector search.
        ha_states: mapa entity_id → state tal como lo reporta HA.
    """
    if entity_id not in ha_states:
        return f"{entity_id} no existe en Home Assistant"
    state = ha_states[entity_id]
    if state == "unavailable":
        return (
            f"{entity_id} está unavailable — HA acepta la llamada y no pasa nada"
        )
    return None


def check_phrase(
    phrase: str,
    resolved: dict | None,
    ha_states: dict[str, str],
) -> SmokeResult:
    """Recorrer la cadena para una frase, sin ejecutar la acción.

    Args:
        phrase: comando canónico, tal como lo diría el usuario.
        resolved: dict del vector search (entity_id/domain/service/data), o
            None si no resolvió.
        ha_states: mapa entity_id → state de HA.

    Returns:
        SmokeResult con la PRIMERA etapa que falla (las posteriores son
        consecuencia de esa), o stage="ok".
    """
    if not resolved or not resolved.get("entity_id"):
        return SmokeResult(
            phrase=phrase,
            ok=False,
            stage=STAGE_VECTOR_SEARCH,
            detail="no resolvió a ninguna entidad (¿índice vacío o stale?)",
        )

    entity_id = resolved["entity_id"]
    service_data = resolved.get("data") or {}
    similarity = resolved.get("similarity")

    problem = entity_problem(entity_id, ha_states)
    if problem is not None:
        return SmokeResult(
            phrase=phrase,
            ok=False,
            stage=STAGE_ENTITY,
            detail=problem,
            entity_id=entity_id,
            service_data=service_data,
            similarity=similarity,
        )

    bad_keys = invalid_payload_keys(service_data)
    if bad_keys:
        return SmokeResult(
            phrase=phrase,
            ok=False,
            stage=STAGE_PAYLOAD,
            detail=(
                f"HA rechazaría el payload por {bad_keys} "
                f"(descarta el service_data entero)"
            ),
            entity_id=entity_id,
            service_data=service_data,
            similarity=similarity,
        )

    service = resolved.get("service", "?")
    return SmokeResult(
        phrase=phrase,
        ok=True,
        stage=STAGE_OK,
        detail=f"{service} → {entity_id}",
        entity_id=entity_id,
        service_data=service_data,
        similarity=similarity,
    )
