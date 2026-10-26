"""Resolve spoken targets within current explicit selection and room references."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from src.home_assistant.home_actions import HomeActionService
from src.nlu.home_command import HomeCommand, normalize_target
from src.rooms.room_context import RoomConfig


@dataclass(frozen=True)
class VoiceTarget:
    """One exact entity, or a reason to ask without sending an effect."""

    entity_id: str | None
    reason: str | None = None


class HomeVoiceResolver:
    """Use configured room defaults; no vector or name-fragment fallback."""

    def __init__(self, actions: HomeActionService, rooms: Mapping[str, RoomConfig] | None = None) -> None:
        self.actions = actions
        self.rooms = dict(rooms) if isinstance(rooms, Mapping) else {}

    async def resolve(self, command: HomeCommand, *, room_id: str | None) -> VoiceTarget:
        """Resolve an explicit room or microphone default against current grants.

        An explicit unknown/ambiguous qualifier never falls back to the mic or
        to a single remaining entity in the house. Without any room evidence,
        one selected domain entity is sufficient; more than one needs a choice.
        """
        grants = await self.actions.policy.snapshot_grants()
        candidates = {grant.entity_id for grant in grants if grant.direct_control and grant.entity_id.startswith(command.domain + ".")}
        if not candidates:
            return VoiceTarget(None, "no_selected_target")
        if command.domain == "scene":
            target = normalize_target(command.target)
            matches = []
            for grant in grants:
                if grant.entity_id in candidates and grant.scene_effects:
                    aliases = (grant.entity_id.partition(".")[2].replace("_", " "), grant.display_name, *grant.aliases)
                    if target in {normalize_target(alias) for alias in aliases if alias}:
                        matches.append(grant.entity_id)
            return VoiceTarget(matches[0]) if len(matches) == 1 else VoiceTarget(None, "unknown_or_ambiguous_scene")
        target = normalize_target(command.target)
        explicit = bool(target)
        target = re.sub(r"^(?:del|de la|de los|de las|de|en el|en la|en|el|la|los|las)\s+", "", target)
        matches = []
        if explicit:
            for config in self.rooms.values():
                aliases = (config.room_id, config.name, config.display_name, *config.aliases)
                if target in {normalize_target(re.sub(r"^(?:el|la|los|las)\s+", "", alias, flags=re.IGNORECASE)) for alias in aliases}:
                    matches.append(config)
            if len(matches) != 1:
                return VoiceTarget(None, "unknown_or_ambiguous_room")
            config = matches[0]
        elif room_id is not None:
            config = self.rooms.get(room_id)
            if config is None:
                return VoiceTarget(None, "unknown_microphone_room")
        else:
            return VoiceTarget(next(iter(candidates))) if len(candidates) == 1 else VoiceTarget(None, "ambiguous_target")
        selected = getattr(config, "default_" + command.domain, None)
        if selected not in candidates:
            return VoiceTarget(None, "room_target_not_selected")
        return VoiceTarget(selected)
