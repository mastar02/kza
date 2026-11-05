"""Explicit Dashboard action authority for a manual voice-command client.

The capture capability never enters this transport. Receipts describe HA state,
not physical completion; an unknown transport outcome is never retried here.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import requests

from src.audio.native_capture_authority import _origin
from src.core.task_outcome import DispatchState, EffectState
from src.home_assistant.control_policy import ActionOrigin
from src.home_assistant.home_actions import request_from_service


@dataclass(frozen=True)
class ManualVoiceActionResult:
    """Keep dispatch and observed effect separate from the public status."""

    request_id: str
    entity_id: str
    status: str
    dispatch: str
    effect: str
    observation_scope: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _ActionLease:
    """Identify one admitted synchronous callable without an increment gap."""

    thread: threading.Thread


class ManualVoiceActionClient:
    """Own a separate Session using an explicitly supplied home:control actor.

    The enclosing capture owner must await execute and close before retiring its
    capability. A POST without a terminal receipt leaves remote work unconfirmed,
    even after local Session close. Failed known close receipts retain handles.
    No second POST or close retry is inferred safe.
    """

    def __init__(self, url: str, token: str, *, timeout_s: float = 10.0,
                 session_factory: Callable[[], requests.Session] = requests.Session) -> None:
        try:
            self._origin = _origin(url)
        except ValueError:
            raise ValueError("manual_voice_action_origin_invalid") from None
        if (type(token) is not str or not token or len(token) > 4096
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("manual_voice_action_credential_invalid")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("manual_voice_action_timing_invalid")
        self._credential: str | None = token
        self._timeout = timeout_s
        self._condition = threading.Condition()
        self._active: _ActionLease | None = None
        self._closing = self._closed = self._close_attempted = False
        self._response: requests.Response | None = None
        self._remote_unconfirmed = False
        self._session = session_factory()
        self._session.trust_env = False

    @staticmethod
    def _unknown(request_id: str, entity_id: str, reason: str) -> ManualVoiceActionResult:
        return ManualVoiceActionResult(request_id, entity_id, "uncertain", "unknown", "unknown",
                                       reason=reason)

    @staticmethod
    def _receipt(payload: object, request_id: str, entity_id: str) -> ManualVoiceActionResult:
        if not isinstance(payload, dict):
            raise ValueError("manual_voice_action_receipt_invalid")
        status, dispatch, effect = (payload.get(key) for key in ("status", "dispatch", "effect"))
        scope = payload.get("observation_scope")
        if (payload.get("request_id") != request_id or payload.get("entity_id") != entity_id
                or type(status) is not str or status not in {"confirmed", "failed", "uncertain", "in_progress"}
                or type(dispatch) is not str or dispatch not in {value.value for value in DispatchState}
                or type(effect) is not str or effect not in {value.value for value in EffectState}
                or scope not in {None, "ha_state"}
                or (status == "confirmed" and (dispatch, effect, scope) != ("accepted", "observed", "ha_state"))):
            raise ValueError("manual_voice_action_receipt_invalid")
        # Never retain an arbitrary server detail, response body, or traceback.
        return ManualVoiceActionResult(request_id, entity_id, status, dispatch, effect, scope,
                                       None if status == "confirmed" else "manual_voice_action_unconfirmed")

    def execute(self, domain: str, service: str, entity_id: str, *,
                request_id: str) -> ManualVoiceActionResult:
        """Send one validated intent once, returning only bounded receipt fields."""
        try:
            intent = request_from_service(domain, service, entity_id, None,
                                          request_id=request_id, origin=ActionOrigin.HTTP)
        except (ValueError, TypeError):
            raise ValueError("manual_voice_action_intent_invalid") from None
        lease = _ActionLease(threading.current_thread())
        control = None
        try:
            with self._condition:
                if self._closing:
                    raise RuntimeError("manual_voice_action_closed")
                if self._active is not None:
                    raise RuntimeError("manual_voice_action_busy")
                self._active = lease
            result = self._unknown(request_id, entity_id, "manual_voice_action_transport_unknown")
            try:
                # The server owns a shielded action Task after POST admission.
                self._remote_unconfirmed = True
                self._response = self._session.post(self._origin + "/api/home/actions",
                    json={"entity_id": intent.entity_id, "operation": intent.operation,
                          "request_id": intent.request_id},
                    headers={"Origin": self._origin, "Authorization": "Bearer " + self._credential},
                    timeout=self._timeout, allow_redirects=False)
                if self._response.status_code == 200:
                    result = self._receipt(self._response.json(), request_id, entity_id)
                    if result.status in {"confirmed", "failed", "uncertain"}:
                        self._remote_unconfirmed = False
                elif self._response.status_code in {401, 403, 409, 422}:
                    reason = {401: "manual_voice_action_auth_required", 403: "manual_voice_action_denied",
                              409: "manual_voice_action_request_conflict", 422: "manual_voice_action_invalid"}
                    if self._response.status_code in {401, 403}:
                        # Delivery may be denied after the server admitted an action.
                        result = self._unknown(request_id, entity_id, reason[self._response.status_code])
                    else:
                        self._remote_unconfirmed = False
                        result = ManualVoiceActionResult(request_id, entity_id, "failed", "not_sent",
                                                        "not_observed", reason=reason[self._response.status_code])
            except Exception:
                result = self._unknown(request_id, entity_id, "manual_voice_action_transport_unknown")
            except BaseException as error:
                control = (KeyboardInterrupt() if isinstance(error, KeyboardInterrupt) else
                           SystemExit(error.code if type(error.code) is int else 1)
                           if isinstance(error, SystemExit) else RuntimeError("manual_voice_action_interrupted"))
            finally:
                if self._response is not None:
                    try:
                        self._response.close()
                    except BaseException as error:
                        with self._condition:
                            self._closing = True
                        result = self._unknown(request_id, entity_id, "manual_voice_action_close_unconfirmed")
                        if control is None and isinstance(error, (KeyboardInterrupt, SystemExit)):
                            control = (KeyboardInterrupt() if isinstance(error, KeyboardInterrupt) else
                                       SystemExit(error.code if type(error.code) is int else 1))
                    else:
                        self._response = None
        finally:
            with self._condition:
                if self._active is lease:
                    if self._remote_unconfirmed:
                        self._closing = True
                    self._active = None
                    self._condition.notify_all()
        if control is not None:
            raise control from None
        return result

    def close(self) -> None:
        """Wait for the admitted request and require the actual Session close return."""
        with self._condition:
            if self._active is not None and self._active.thread is threading.current_thread():
                raise RuntimeError("manual_voice_action_close_from_worker")
            self._closing = True
            while self._active is not None:
                self._condition.wait(.05)
            if self._closed:
                return
            if self._close_attempted:
                raise RuntimeError("manual_voice_action_close_unconfirmed")
            self._close_attempted = True
        failed = False
        try:
            self._session.close()
        except BaseException:
            failed = True
        if failed or self._response is not None or self._remote_unconfirmed:
            raise RuntimeError("manual_voice_action_close_unconfirmed")
        self._credential = None
        self._closed = True
