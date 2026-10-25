"""Shared voice hooks, single-attempt execution, receipts and local feedback."""

import logging
import time
import uuid

from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionResult, HomeActionService

logger = logging.getLogger(__name__)


class VoiceHomeActionExecutor:
    """Keep both voice routing modes behind the same final action boundary."""

    def __init__(self, ha_client, actions: HomeActionService, *, response_handler=None,
                 hooks=None, before_handler_warn_ms: float = 5.0, unavailable_precheck_enabled: bool = True) -> None:
        self.ha = ha_client
        self.home_actions = actions
        self.response_handler = response_handler
        self.hooks = hooks
        self._before_handler_warn_ms = before_handler_warn_ms
        self._unavailable_precheck_enabled = unavailable_precheck_enabled

    async def execute(self, command: dict) -> HomeActionResult:
        """Apply hooks and return the shared service receipt, with local failure feedback."""
        request_id = "voice:" + uuid.uuid4().hex
        domain = command.get("domain")
        service = command.get("service")
        entity_id = command.get("entity_id")
        description = command.get("description") or entity_id or "esa acción"

        # Plan #3 OpenClaw — before_ha_action chain (block / rewrite)
        call = None
        rewritten_data = command.get("data")
        if self.hooks is not None:
            from src.hooks import (
                HaActionCall,
                BlockResult,
                HaActionDispatchedPayload,
                HaActionBlockedPayload,
                execute_before_chain,
                execute_after_event,
            )
            call = HaActionCall(
                entity_id=entity_id or "",
                domain=domain or "",
                service=service or "",
                service_data=rewritten_data or {},
                user_id=command.get("user_id"),
                user_name=command.get("user_name"),
                zone_id=command.get("zone_id"),
                timestamp=time.time(),
            )
            result = execute_before_chain(
                self.hooks, "before_ha_action", call,
                warn_ms=self._before_handler_warn_ms,
            )
            if isinstance(result, BlockResult):
                logger.info(
                    f"[HA-CALL BLOCKED] {domain}.{service}@{entity_id} "
                    f"by rule={result.rule_name}: {result.reason}"
                )
                if self.response_handler is not None:
                    try:
                        self.response_handler.speak(result.reason or "No puedo hacer eso")
                    except Exception as e:
                        logger.warning(f"No pude hablar block reason: {e}")
                execute_after_event(
                    self.hooks,
                    "ha_action_blocked",
                    HaActionBlockedPayload(
                        timestamp=time.time(), call=call, block=result,
                    ),
                )
                return HomeActionResult(request_id, entity_id, "failed", "hook_blocked")
            # Apply rewrite (if any) back to local vars
            domain = result.domain
            service = result.service
            entity_id = result.entity_id
            rewritten_data = result.service_data

        # A negative cache entry is refreshed for prompt feedback. Regardless
        # of this optional precheck, the shared service enforces fresh evidence.
        unavailable = (
            self._unavailable_precheck_enabled
            and entity_id
            and self.ha.is_entity_available(entity_id) is False
        )
        if unavailable:
            # Refresh a negative cache entry before suppressing a recovered device.
            try:
                fresh = await self.ha.get_entity_state_fresh(entity_id)
            except Exception:
                fresh = None
            if isinstance(fresh, dict) and fresh.get("state") not in (None, "unavailable"):
                unavailable = False
        if unavailable:
            logger.warning(
                f"[HA-UNAVAILABLE] {domain}.{service}@{entity_id} "
                f"({description}) — no se envía la llamada"
            )
            if self.response_handler is not None:
                try:
                    self.response_handler.play_earcon(zone_id=command.get("zone_id"))
                except Exception as e:
                    logger.warning(f"No pude reproducir earcon de entidad caída: {e}")
            # Plan #3 OpenClaw — el precheck es una tercera forma de retener
            # un comando (además del before_ha_action block de arriba), y sin
            # emitir este evento el comando quedaba fuera del audit trail
            # (src/policies/audit_sqlite.py) — invisible en data/audit.db.
            if self.hooks is not None and call is not None:
                from dataclasses import replace
                from src.hooks import (
                    BlockResult,
                    HaActionBlockedPayload,
                    execute_after_event,
                )
                final_call = replace(
                    call,
                    domain=domain or "",
                    service=service or "",
                    entity_id=entity_id or "",
                    service_data=rewritten_data or {},
                )
                execute_after_event(
                    self.hooks,
                    "ha_action_blocked",
                    HaActionBlockedPayload(
                        timestamp=time.time(),
                        call=final_call,
                        block=BlockResult(
                            reason=f"{description}: entidad no disponible",
                            rule_name="entity_unavailable",
                        ),
                    ),
                )
            return HomeActionResult(request_id, entity_id, "failed", "entity_unavailable")

        t0 = time.perf_counter()
        err: str | None = None
        try:
            receipt = await self.home_actions.execute_service(
                domain, service, entity_id, rewritten_data,
                actor=HomeActor.household(), origin=ActionOrigin.VOICE, request_id=request_id,
            )
            success = receipt.status == "confirmed"
            err = receipt.reason
            dt = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[HA-CALL] {domain}.{service}@{entity_id} "
                f"success={success} took={dt:.0f}ms"
            )
        except Exception as e:
            err = str(e)
            logger.error(
                f"Reconcile error en {domain}.{service}@{entity_id}: {e}"
            )
            success = False
            receipt = HomeActionResult(request_id, entity_id, "uncertain", "action_exception")

        # Plan #3 OpenClaw — emit after_event with dispatch result
        if self.hooks is not None and call is not None:
            from dataclasses import replace
            from src.hooks import (
                HaActionDispatchedPayload,
                execute_after_event,
            )
            final_call = replace(
                call,
                domain=domain or "",
                service=service or "",
                entity_id=entity_id or "",
                service_data=rewritten_data or {},
            )
            execute_after_event(
                self.hooks,
                "ha_action_dispatched",
                HaActionDispatchedPayload(
                    timestamp=time.time(),
                    call=final_call,
                    success=success,
                    error=err,
                ),
            )

        if success:
            return receipt

        # Fallo real. Earcon (no frase) en la zona donde habló el usuario:
        # decisión 2026-07-25. La regla "domótica silenciosa" cubre los ÉXITOS
        # que el usuario valida visualmente; un fallo mudo es justo lo que hizo
        # invisible el bug del primer comando post-idle.
        logger.warning(
            f"[HA-FAIL] {domain}.{service}@{entity_id} ({description}) "
            f"err={err or 'success=False'}"
        )
        if self.response_handler is not None:
            try:
                self.response_handler.play_earcon(zone_id=command.get("zone_id"))
            except Exception as e:
                logger.warning(f"No pude reproducir earcon de fallo HA: {e}")
        else:
            logger.warning(
                f"HA fire-and-forget falló en {domain}.{service}@{entity_id} "
                f"sin response_handler — usuario no fue notificado"
            )
        return receipt

