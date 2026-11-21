"""Room-scoped media control using Home Assistant observations and service receipts."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import re
from time import time

from src.home_assistant.control_policy import ActorKind, HomeActionDenied, HomeActor
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallOutcome, ServiceCallStatus
from src.home_assistant.operations import finite_number
from src.home_assistant.state_projection import EntityObservation


class MediaOperation(StrEnum):
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    VOLUME_SET = "volume_set"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    VOLUME_MUTE = "volume_mute"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"


# MediaPlayerEntityFeature bits. PLAY_MEDIA starts new content and is not PLAY.
_PLAYER_SERVICES = {
    MediaOperation.PLAY: (16384, "media_play"),
    MediaOperation.PAUSE: (1, "media_pause"),
    MediaOperation.STOP: (4096, "media_stop"),
    MediaOperation.VOLUME_SET: (4, "volume_set"),
    MediaOperation.VOLUME_UP: (1024, "volume_up"),
    MediaOperation.VOLUME_DOWN: (1024, "volume_down"),
    MediaOperation.VOLUME_MUTE: (8, "volume_mute"),
    MediaOperation.TURN_ON: (128, "turn_on"),
    MediaOperation.TURN_OFF: (256, "turn_off"),
}


@dataclass(frozen=True)
class RemoteGrant:
    """Associated remote with individually authorized power services and volume keys."""

    entity_id: str
    turn_on: bool = False
    turn_off: bool = False
    volume_up_command: str | None = None
    volume_down_command: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"remote\.[a-z0-9_]+", self.entity_id):
            raise ValueError("invalid_remote_entity")
        if type(self.turn_on) is not bool or type(self.turn_off) is not bool:
            raise ValueError("invalid_remote_power_grant")
        for command in (self.volume_up_command, self.volume_down_command):
            if command is not None and (
                type(command) is not str or not command.strip()
            ):
                raise ValueError("invalid_remote_command")

    def service(self, op: MediaOperation) -> tuple[str, dict] | None:
        if (
            op is MediaOperation.TURN_ON
            and self.turn_on
            or op is MediaOperation.TURN_OFF
            and self.turn_off
        ):
            return op.value, {}
        command = {
            MediaOperation.VOLUME_UP: self.volume_up_command,
            MediaOperation.VOLUME_DOWN: self.volume_down_command,
        }.get(op)
        return ("send_command", {"command": command}) if command else None


@dataclass(frozen=True)
class MediaSelection:
    """One player selected by trusted room configuration."""

    room: str
    entity_id: str
    remote: RemoteGrant | None = None


@dataclass(frozen=True)
class MediaState:
    """Current observation. Missing and historical attributes are omitted."""

    state: str
    supported_features: int = 0
    volume_level: float | None = None
    is_volume_muted: bool | None = None
    media_title: str | None = None
    app_name: str | None = None


def _authorize(actor: HomeActor, permission: str) -> None:
    if not isinstance(actor, HomeActor):
        raise HomeActionDenied("home_actor_required")
    actor.require_current()
    if actor.kind is not ActorKind.HOUSEHOLD and permission not in actor.permissions:
        raise HomeActionDenied("media_permission_denied")


def _parameters(op: MediaOperation, value: float | bool | None) -> dict:
    if op is MediaOperation.VOLUME_SET:
        if not finite_number(value) or not 0 <= value <= 1:
            raise ValueError("invalid_volume_level")
        return {"volume_level": value}
    if op is MediaOperation.VOLUME_MUTE:
        if type(value) is not bool:
            raise ValueError("invalid_volume_mute")
        return {"is_volume_muted": value}
    if value is not None:
        raise ValueError("unexpected_media_value")
    return {}


class MediaController:
    """Read and control only the configured players and their associated remotes."""

    def __init__(
        self,
        ha_client: HomeAssistantClient,
        selections: Iterable[MediaSelection],
        *,
        max_staleness_s: float,
        clock: Callable[[], float] = time,
    ) -> None:
        if not finite_number(max_staleness_s) or max_staleness_s <= 0:
            raise ValueError("invalid_max_staleness")
        self._ha = ha_client
        self._clock = clock
        self._max_age = max_staleness_s
        self._players: dict[tuple[str, str], MediaSelection] = {}
        for selection in selections:
            key = (selection.room, selection.entity_id)
            if (
                not selection.room.strip()
                or key in self._players
                or not re.fullmatch(r"media_player\.[a-z0-9_]+", selection.entity_id)
            ):
                raise ValueError("invalid_media_selection")
            self._players[key] = selection

    def _selection(self, room: str, entity_id: str) -> MediaSelection:
        try:
            return self._players[room, entity_id]
        except KeyError:
            raise HomeActionDenied("media_entity_not_selected") from None

    def selected_players(self, actor: HomeActor, *, room: str | None = None) -> tuple[MediaSelection, ...]:
        """Return explicit selections for a current shared reader."""
        _authorize(actor, "home:read")
        return tuple(player for player in self._players.values() if room is None or player.room == room)

    async def _observe(self, actor: HomeActor, entity_id: str) -> EntityObservation:
        actor.require_current()
        try:
            observation = await self._ha.read_entity_observation(entity_id)
        finally:
            actor.require_current()
        if observation.entity_id != entity_id:
            raise ValueError("media_observation_target_mismatch")
        return observation

    def _fresh(self, observation: EntityObservation, now: float) -> bool:
        return (
            observation.current(now) and now - observation.observed_at <= self._max_age
        )

    def _state(self, observation: EntityObservation) -> MediaState:
        now = self._clock()
        if not self._fresh(observation, now):
            if observation.reason in (
                None,
                "observation_expired",
                "clock_before_observation",
            ):
                return MediaState("stale")
            return MediaState("unavailable")
        if not observation.exists:
            return MediaState("unavailable")
        raw = observation.state
        state = raw["state"]
        if state in ("unknown", "unavailable"):
            return MediaState(state)
        if state not in (
            "playing",
            "paused",
            "idle",
            "off",
            "standby",
            "on",
            "buffering",
        ):
            return MediaState("unknown")
        attrs = raw["attributes"]
        bits = attrs.get("supported_features")
        volume = attrs.get("volume_level")
        mute = attrs.get("is_volume_muted")
        title, app = attrs.get("media_title"), attrs.get("app_name")
        return MediaState(
            state,
            bits if type(bits) is int and bits >= 0 else 0,
            volume if finite_number(volume) and 0 <= volume <= 1 else None,
            mute if type(mute) is bool else None,
            title if isinstance(title, str) else None,
            app if isinstance(app, str) else None,
        )

    async def read(self, actor: HomeActor, room: str, entity_id: str) -> MediaState:
        """Return the selected player's current state and optional HA attributes."""
        _authorize(actor, "home:read")
        self._selection(room, entity_id)
        return self._state(await self._observe(actor, entity_id))

    async def control(
        self,
        actor: HomeActor,
        room: str,
        entity_id: str,
        operation: MediaOperation,
        *,
        value: float | bool | None = None,
    ) -> ServiceCallOutcome:
        """Attempt one authorized service and return HA transport certainty.

        Raises:
            HomeActionDenied: The actor, selection, observation or capability denies dispatch.
            ValueError: The operation or its argument is invalid.
        """
        _authorize(actor, "home:control")
        selection = self._selection(room, entity_id)
        op = MediaOperation(operation)
        data = _parameters(op, value)
        player = await self._observe(actor, entity_id)
        state = self._state(player)
        if state.state in ("unknown", "unavailable", "stale"):
            raise HomeActionDenied(f"media_{state.state}")
        bit, service = _PLAYER_SERVICES[op]
        domain = "media_player"
        observations = [player]
        if not state.supported_features & bit:
            remote = selection.remote
            fallback = remote.service(op) if remote else None
            if fallback is None:
                raise HomeActionDenied(f"media_unsupported_{op.value}")
            domain, entity_id = "remote", remote.entity_id
            service, data = fallback
            observation = await self._observe(actor, entity_id)
            remote_state = self._state(observation).state
            if remote_state not in ("on", "off"):
                raise HomeActionDenied(f"remote_{remote_state}")
            observations.append(observation)

        def current() -> bool:
            actor.require_current()
            now = self._clock()
            return all(self._fresh(observation, now) for observation in observations)

        actor.require_current()
        receipt = await self._ha.call_service_once(domain, service, entity_id, data, guard=current)
        try:
            actor.require_current()
        except HomeActionDenied:
            if receipt.status in (ServiceCallStatus.ACCEPTED, ServiceCallStatus.UNCERTAIN):
                return ServiceCallOutcome(ServiceCallStatus.UNCERTAIN, "authority_lost_after_dispatch")
            if receipt.status is ServiceCallStatus.NOT_SENT:
                raise
        return receipt
