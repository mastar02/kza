"""Extract one scheduled routine definition from a bounded local grammar."""

from enum import StrEnum
import re

from src.core.household_clock import SUPPORTED_HOUSEHOLD_TIMEZONE
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionRequest, HomeActionService
from src.home_assistant.operations import action_parameters
from src.nlu.command_grammar import security_order
from src.nlu.home_command import HomeCommand, normalize_target, parse_home_command
from src.nlu.regex.vocab import CARDINALS
from src.reminders.voice_request import period_hours
from src.routines.models import RoutineSpec, parse_routine_spec


class ProposalIssue(StrEnum):
    """Reasons a voice definition could not be extracted."""

    TIME = "time"
    TARGET = "target"


class VoiceProposalError(ValueError):
    """The utterance needs clarification or names an unsupported effect."""

    def __init__(self, issue: ProposalIssue) -> None:
        super().__init__(issue.value)
        self.issue = issue


_ON = r"prend(?:e|er|a)|enc(?:iende|ende|ender|ienda)"
_OFF = r"apag(?:a|ar|ue)"
_POWER = re.compile(rf"(?P<verb>{_ON}|{_OFF}) (?:(?:el|la|los|las) )?(?P<target>.+)")
_ACTION_HEAD = re.compile(
    rf"\b(?:{_ON}|{_OFF}|activ\w*|desactiv\w*|abr\w*|cierr\w*|cerr\w*|pon\w*|ajust\w*|sub\w*|baj\w*)\b"
)
_TIME_LEAD = re.compile(r"\b(?:a|cuando (?:sea|sean)) las?\b")
_CLOCK = re.compile(r"(?P<hour>[0-9]{1,2}|[a-z]+)(?::(?P<minute>[0-9]{2}))?"
                    r"(?: de la (?P<period>manana|tarde|noche|madrugada)| ?(?P<meridiem>[ap]\.? ?m\.?))?")
_DAYS = {"todos los dias": tuple(range(7)), "de lunes a viernes": tuple(range(5)),
         "los fines de semana": (5, 6)}
_DAYS.update({f"los {day}": (index,) for index, day in
              enumerate(("lunes", "martes", "miercoles", "jueves", "viernes", "sabados", "domingos"))})
_SECURITY_TARGET = re.compile(r"\b(?:alarmas?|alarm_control_panel|cerraduras?|lock|cerrojos?|candados?)\b")
_DEVICE_NOUN = re.compile(r"(?P<noun>luces|luz|lamparas?|enchufes?|interruptor(?:es)?) (?P<name>.+)")
_TARGET_LEAD = re.compile(r"^(?:de (?:la|los|las) |del? |en (?:el |la )?)")


def _schedule_and_action(text: str) -> tuple[dict, str]:
    time_leads = list(_TIME_LEAD.finditer(text))
    if len(time_leads) != 1:
        raise VoiceProposalError(ProposalIssue.TIME)
    lead = time_leads[0]
    before, after = text[:lead.start()].strip(" ,"), text[lead.end():].strip()
    if not before or before in _DAYS:
        action_start = _ACTION_HEAD.search(after)
        if action_start is None:
            raise VoiceProposalError(ProposalIssue.TARGET)
        clock, action = after[:action_start.start()].strip(" ,"), after[action_start.start():]
        day_before = before
    else:
        action, clock = before, after
        day_before = next((day for day in _DAYS if action.endswith(" " + day)), "")
        if day_before:
            action = action.removesuffix(" " + day)
    day_after = next((day for day in _DAYS if clock.endswith(" " + day)), "")
    if day_before and day_after:
        raise VoiceProposalError(ProposalIssue.TIME)
    if day_after:
        clock = clock.removesuffix(" " + day_after)
    match = _CLOCK.fullmatch(clock)
    if match is None:
        raise VoiceProposalError(ProposalIssue.TIME)
    raw_hour = match["hour"]
    hour = int(raw_hour) if raw_hour.isdecimal() else CARDINALS.get(raw_hour, 1 if raw_hour == "una" else -1)
    minute = int(match["minute"] or 0)
    period = match["period"] or (re.sub(r"[. ]", "", match["meridiem"]) if match["meridiem"] else None)
    explicit_clock = len(raw_hour) == 2 and raw_hour.isdecimal() and match["minute"] is not None
    if period and 1 <= hour <= 12:
        hours = period_hours(hour, period)
        if len(hours) != 1:
            raise VoiceProposalError(ProposalIssue.TIME)
        hour = next(iter(hours))
    elif period or (1 <= hour <= 12 and not explicit_clock):
        raise VoiceProposalError(ProposalIssue.TIME)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise VoiceProposalError(ProposalIssue.TIME)
    days = _DAYS[day_before or day_after or "todos los dias"]
    return {"at": f"{hour:02d}:{minute:02d}", "weekdays": list(days)}, action


def _command(action: str) -> HomeCommand:
    extended = parse_home_command(action)
    if extended is not None:
        return extended
    match = _POWER.fullmatch(action)
    if match is None:
        raise VoiceProposalError(ProposalIssue.TARGET)
    target = match["target"]
    domain = ""
    device = _DEVICE_NOUN.fullmatch(target)
    if device is not None:
        domain = "light" if device["noun"].startswith(("lu", "lampara")) else "switch"
        target = device["name"]
    return HomeCommand(domain, target, "turn_off" if match["verb"].startswith("apag") else "turn_on")


def _selection_names(*aliases: str) -> set[str]:
    names = {normalize_target(alias) for alias in aliases if alias}
    shorter = {_TARGET_LEAD.sub("", match["name"]) for name in names
               if (match := _DEVICE_NOUN.fullmatch(name)) is not None}
    return names | shorter


class VoiceProposalExtractor:
    """Build a strict definition from the reviewed selection and a single voice clause."""

    def __init__(self, home_actions: HomeActionService) -> None:
        self._home_actions = home_actions

    async def extract(self, complement: str) -> RoutineSpec:
        """Resolve a complete action and local schedule without touching HA or a model."""
        text = normalize_target(complement.lstrip(" ,:¿?¡!").rstrip(" ,¿?¡!."))
        if len(text) > 512 or security_order(text) or _SECURITY_TARGET.search(text):
            raise VoiceProposalError(ProposalIssue.TARGET)
        text = re.sub(r"^(?:para que |para |que )", "", text)
        schedule, action = _schedule_and_action(text)
        command = _command(action)
        target = _TARGET_LEAD.sub("", command.target)
        if not target or command.reason is not None:
            raise VoiceProposalError(ProposalIssue.TARGET)

        matches = []
        for grant in await self._home_actions.policy.snapshot_grants():
            domain, _, slug = grant.entity_id.partition(".")
            if command.domain and domain != command.domain:
                continue
            if target in _selection_names(grant.entity_id, slug.replace("_", " "), grant.display_name or "",
                                          *grant.aliases):
                matches.append(grant.entity_id)
        if len(matches) != 1:
            raise VoiceProposalError(ProposalIssue.TARGET)
        entity_id = matches[0]
        spec = parse_routine_spec({
            "name": f"Rutina de {entity_id} ({schedule['at']})", "description": "", "enabled": True,
            "timezone": SUPPORTED_HOUSEHOLD_TIMEZONE, "schedule": schedule, "conditions": [],
            "actions": [{"entity_id": entity_id, "operation": command.operation, **command.parameters}],
        })
        effect, = spec.actions
        await self._home_actions.validate_selection(HomeActionRequest(
            effect.entity_id, str(effect.operation), "voice-routine-proposal", origin=ActionOrigin.ROUTINE,
            **action_parameters(effect),
        ), actor=HomeActor.household())
        return spec
