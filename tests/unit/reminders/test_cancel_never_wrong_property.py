"""Generated property: a spoken reminder cancel never cancels a reminder another reading could mean.

Independent verification of T032-U1. A deterministic grid (no randomness,
no new dependency) of 2-3 shared reminders that share a base text, where a
text may embed a clock (every spoken format) and/or a day word and where
due times are chosen so that a text clock coincides with another
reminder's due time. Cancel phrases are built from each reminder's text
and/or its due time in every format, with and without a period of the day.

The oracle is written from the user decisions, not from the implementation:
a reminder is *consistent* with a phrase when some plausible reading selects
it, each clock or day phrase read either as the due time or as words of the
text (clocks compared by the wall times they can mean; user answer
T032-U1-RR-P1), or when the reminder's text holds every spoken word (user
answer T032-U1-Q1). Accepted defaults: "12 de la noche" is 00:00, 1-4 "de
la noche" is early morning, a bare "de mañana" is the morning (Q2, Q3).

Expected: exactly one consistent reminder -> it may be cancelled (asking is
acceptable and counted as over-ask); more than one -> ask, cancel nothing.
Any cancelled reminder that is not the single consistent one is a
counterexample. Every follow-up reference the implementation proposes must
be consistent with exactly its own reminder.

Set ``T032_PROP_REPORT`` to a path prefix to write the counts and classes
as JSON.
"""

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import permutations
from zoneinfo import ZoneInfo

import pytest

from src.reminders.reminder_manager import CancelMatch, ReminderManager
from src.reminders.reminder_store import Reminder, ReminderScope, ReminderState
from tests.mocks.household_dispatch import CountedLeaves

MVD = ZoneInfo("America/Montevideo")
NOW = datetime(2026, 11, 19, 0, 30, tzinfo=MVD)  # a Thursday
TODAY = NOW.date()
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
HOUR_WORDS = {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete", 8: "ocho",
              9: "nueve", 10: "diez", 11: "once", 12: "doce"}
# Function words only: hour and minute words ("una", "media") and periods are content.
STOP = frozenset({"de", "del", "el", "la", "las", "los", "lo", "a", "al", "para", "y", "por", "en", "con", "que",
                  "un"})
TOKEN = re.compile(r"\d{1,2}[:.]\d{2}|\d+|[^\W\d_]+")
BASE = "tomar pastillas"
SUPER = "tomar pastillas rosadas"


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def words(text: str) -> frozenset[str]:
    return frozenset(w for w in TOKEN.findall(fold(text)) if w not in STOP)


def walls(hour: int, minute: int, period: str | None = None) -> frozenset[tuple[int, int]]:
    """Wall times a spoken clock can mean (user decisions Q2: 12 and 1-4 "de la noche" are after midnight)."""
    if hour > 12:
        return frozenset({(hour, minute)})
    am, pm = (hour % 12, minute), (hour % 12 + 12, minute)
    if period in ("mañana", "madrugada"):
        return frozenset({am})
    if period == "noche":
        return frozenset({am} if hour in (12, 1, 2, 3, 4) else {pm})
    if period in ("tarde", "mediodía"):
        return frozenset({pm})
    return frozenset({am, pm})


@dataclass(frozen=True)
class Clock:
    text: str  # without the preposition: "las 8 y media", "la una del mediodía"
    walls: frozenset
    family: str


@dataclass(frozen=True)
class Day:
    text: str
    dates: frozenset = frozenset()
    morning: bool = False

    @property
    def kind(self) -> str:
        if self.morning:
            return "de mañana(morning)"
        return re.sub("|".join(WEEKDAYS), "<wd>", self.text)


PERIODS = (("mañana", "de la mañana", range(5, 12)), ("madrugada", "de la madrugada", range(1, 6)),
           ("tarde", "de la tarde", range(1, 9)), ("noche", "de la noche", (8, 9, 10, 11, 12, 1, 2, 3, 4)),
           ("mediodía", "del mediodía", (12, 1)))


def clock_forms(h24: int, minute: int) -> list[Clock]:
    """Every spoken clock this grid uses for a wall time, each with the wall times it can mean."""
    h12 = h24 % 12 or 12
    art = "la" if h12 == 1 else "las"
    plain = []
    if minute == 0:
        plain += [Clock(f"{art} {h12}", walls(h12, 0), "H"), Clock(f"{art} {HOUR_WORDS[h12]}", walls(h12, 0), "word")]
    else:
        plain += [Clock(f"{art} {h12}:{minute:02d}", walls(h12, minute), "H:MM"),
                  Clock(f"{art} {h12}.{minute:02d}", walls(h12, minute), "H.MM")]
        if h12 < 10:
            plain.append(Clock(f"{art} {h12:02d}:{minute:02d}", walls(h12, minute), "0H:MM"))
        if minute in (15, 30):
            word = "media" if minute == 30 else "cuarto"
            plain.append(Clock(f"{art} {h12} y {word}", walls(h12, minute), f"y {word}"))
    forms = list(plain)
    if h24 > 12:
        forms.append(Clock(f"las {h24}" + (f":{minute:02d}" if minute else ""), walls(h24, minute), "24h"))
    for period, suffix, hours in PERIODS:
        meant = walls(h12, minute, period)
        if h12 not in hours or (h24, minute) not in meant:
            continue
        for p in plain:
            if p.family in ("H", "H:MM", "y media", "y cuarto"):
                forms.append(Clock(f"{p.text} {suffix}", meant, f"{p.family}+{period}"))
    return forms


def weekday_dates(weekday: int) -> frozenset[date]:
    return frozenset(TODAY + timedelta(days=k) for k in range(14) if (TODAY + timedelta(days=k)).weekday() == weekday)


def day_forms(day: date) -> list[Day]:
    """Spoken days for a due date, the most natural first (never "de mañana": Q3)."""
    offset = (day - TODAY).days
    out = []
    if offset == 0:
        out += [Day("de hoy", frozenset({day})), Day("hoy", frozenset({day})), Day("para hoy", frozenset({day}))]
    elif offset == 1:
        out += [Day("mañana", frozenset({day})), Day("para mañana", frozenset({day}))]
    elif offset == 2:
        out += [Day("de pasado mañana", frozenset({day})), Day("pasado mañana", frozenset({day}))]
    name = WEEKDAYS[day.weekday()]
    out += [Day(f"del {name}", weekday_dates(day.weekday())), Day(f"el {name}", weekday_dates(day.weekday()))]
    return out


MORNING = Day("de mañana", morning=True)
TEXT_DAYS = {
    "hoy": (Day("hoy", frozenset({TODAY})), Day("de hoy", frozenset({TODAY}))),
    "mañana": (Day("mañana", frozenset({TODAY + timedelta(1)})), Day("para mañana", frozenset({TODAY + timedelta(1)}))),
    "pasado mañana": (Day("pasado mañana", frozenset({TODAY + timedelta(2)})),
                      Day("de pasado mañana", frozenset({TODAY + timedelta(2)}))),
    "el lunes": (Day("el lunes", weekday_dates(0)), Day("del lunes", weekday_dates(0))),
    "del miércoles": (Day("del miércoles", weekday_dates(2)), Day("el miércoles", weekday_dates(2))),
}


def _c(text: str, hour: int, minute: int, period: str | None, family: str) -> Clock:
    return Clock(text, walls(hour, minute, period), family)


TEXT_CLOCKS = (
    _c("las 2", 2, 0, None, "H"), _c("la una", 1, 0, None, "word"), _c("las dos", 2, 0, None, "word"),
    _c("las 8:30", 8, 30, None, "H:MM"), _c("las 8.30", 8, 30, None, "H.MM"), _c("las 08:30", 8, 30, None, "0H:MM"),
    _c("las 8 y media", 8, 30, None, "y media"), _c("las 8 y cuarto", 8, 15, None, "y cuarto"),
    _c("las 20", 20, 0, None, "24h"), _c("las 20:30", 20, 30, None, "24h"),
    _c("las 2 de la tarde", 2, 0, "tarde", "H+tarde"), _c("las 12 del mediodía", 12, 0, "mediodía", "H+mediodía"),
    _c("la una del mediodía", 1, 0, "mediodía", "word+mediodía"), _c("las 8 de la noche", 8, 0, "noche", "H+noche"),
    _c("las 12 de la noche", 12, 0, "noche", "H+noche"), _c("las 2 de la noche", 2, 0, "noche", "H+noche"),
    _c("las 8 de la mañana", 8, 0, "mañana", "H+mañana"),
)
FEW_CLOCKS = tuple(c for c in TEXT_CLOCKS if c.text in ("las 2", "la una", "las 8 y media", "las 20",
                                                         "las 2 de la tarde", "las 12 de la noche"))


@dataclass(frozen=True)
class Spec:
    base: str
    clock: Clock | None
    day: str | None
    due: datetime

    @property
    def text(self) -> str:
        return " ".join([self.base] + ([f"a {self.clock.text}"] if self.clock else []) + ([self.day] if self.day else []))

    @property
    def feature(self) -> str:
        parts = ([f"text-clock:{self.clock.family}"] if self.clock else []) + ([f"text-day:{self.day}"] if self.day else [])
        return "+".join(parts) or ("superset" if self.base == SUPER else "plain")


@dataclass(frozen=True)
class Phrase:
    spoken: str  # the reference as the dispatcher passes it: "de tomar pastillas a las 2"
    base: str
    clocks: tuple = ()
    days: tuple = ()
    shape: tuple = ()

    def with_clock(self, prep: str, clock: Clock, origin: str) -> "Phrase":
        return Phrase(f"{self.spoken} {prep} {clock.text}", self.base, self.clocks + (clock,), self.days,
                      self.shape + (f"{origin}-clock:{clock.family}",))

    def with_day(self, day: Day, origin: str) -> "Phrase":
        return Phrase(f"{self.spoken} {day.text}", self.base, self.clocks, self.days + (day,),
                      self.shape + (f"{origin}-day:{day.kind}",))


def at(days: int, hour: int, minute: int = 0) -> datetime:
    day = TODAY + timedelta(days=days)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=MVD)


def consistent(spec: Spec, phrase: Phrase) -> bool:
    """Some plausible reading of the phrase selects this reminder (independent of the implementation)."""
    text_words = words(spec.text)
    if words(phrase.spoken) <= text_words:  # Q1: the text holds every spoken word
        return True
    if not words(phrase.base) <= text_words:
        return False
    due_hm = (spec.due.hour, spec.due.minute)
    for clock in phrase.clocks:
        as_due = due_hm in clock.walls
        as_text = (spec.clock is not None and bool(spec.clock.walls & clock.walls)) or words(clock.text) <= text_words
        if not (as_due or as_text):
            return False
    for day in phrase.days:
        as_due = spec.due.hour < 12 if day.morning else spec.due.date() in day.dates
        if not (as_due or words(day.text) <= text_words):
            return False
    return True


def reminder_sets() -> list[tuple[str, tuple[Spec, ...]]]:
    """The deterministic grid of shared reminder sets."""
    out = []
    # A: one text holds a clock; the other (plain or superset) is due at one of its wall times or elsewhere.
    for tc in TEXT_CLOCKS:
        meant = sorted(tc.walls)
        for due1 in (at(1, 9, 45), at(1, *meant[0])):
            for base1, base2, offsets in ((BASE, BASE, (1, 4)), (SUPER, BASE, (1,)), (BASE, SUPER, (1,))):
                for wall in meant + [(17, 0)]:
                    for offset in offsets:
                        out.append(("A", (Spec(base1, tc, None, due1), Spec(base2, None, None, at(offset, *wall)))))
    # B: one text holds a day word; the other is due on that day or another.
    for td in TEXT_DAYS:
        for due1 in (at(0, 18), at(1, 8)):
            for offset in (0, 1, 2, 4, 6):
                for hour in (8, 18):
                    out.append(("B", (Spec(BASE, None, td, due1), Spec(BASE, None, None, at(offset, hour)))))
    # C: three reminders: a clock text, a day text, and a plain one.
    for tc in FEW_CLOCKS:
        meant = sorted(tc.walls)
        for td in ("mañana", "el lunes", "hoy"):
            for due2 in (at(1, *meant[0]), at(4, 17)):
                for due3 in (at(1, *meant[0]), at(4, *meant[-1]), at(0, 17)):
                    out.append(("C", (Spec(BASE, tc, None, at(1, 9, 45)), Spec(BASE, None, td, due2),
                                      Spec(BASE, None, None, due3))))
    # D: a text holds both a clock and a day.
    for tc in FEW_CLOCKS:
        meant = sorted(tc.walls)
        for td in ("mañana", "el lunes", "del miércoles"):
            for due2 in (at(1, *meant[0]), at(4, *meant[0]), at(6, *meant[-1]), at(1, 17)):
                out.append(("D", (Spec(BASE, tc, td, at(1, 9, 45)), Spec(BASE, None, None, due2))))
    # E: crossed clocks (each due at the other's text clock); F: the same text twice.
    for tc1, tc2 in permutations(FEW_CLOCKS, 2):
        out.append(("E", (Spec(BASE, tc1, None, at(1, *sorted(tc2.walls)[-1])),
                          Spec(BASE, tc2, None, at(1, *sorted(tc1.walls)[-1])))))
    for tc in FEW_CLOCKS:
        meant = sorted(tc.walls)
        out.append(("F", (Spec(BASE, tc, None, at(1, *meant[0])), Spec(BASE, tc, None, at(1, 17)))))
    # G: crossed days (each due on the day the other's text names).
    offsets = {"hoy": 0, "mañana": 1, "el lunes": 4, "del miércoles": 6}
    for td1, td2 in permutations(offsets, 2):
        out.append(("G", (Spec(BASE, None, td1, at(offsets[td2], 9)), Spec(BASE, None, td2, at(offsets[td1], 9)))))
    return out


def _restated(clock: Clock) -> list[Clock]:
    """The text's clock and every other way to say a wall time it can mean."""
    seen = {clock.text: clock}
    for wall in sorted(clock.walls):
        for form in clock_forms(*wall):
            seen.setdefault(form.text, form)
    return list(seen.values())


def phrases_for(specs: tuple[Spec, ...]) -> list[Phrase]:
    out: dict[str, Phrase] = {}

    def add(phrase: Phrase) -> None:
        out.setdefault(phrase.spoken, phrase)

    wall_times = sorted({(s.due.hour, s.due.minute) for s in specs}
                        | {w for s in specs if s.clock for w in s.clock.walls})
    tails = []
    for wall in wall_times:
        for form in clock_forms(*wall):
            tails.append(("de", form))
            if form.family in ("H", "H:MM", "word") or "+" in form.family:
                tails.append(("a", form))
    due_tails = [("de", form) for s in specs for form in clock_forms(s.due.hour, s.due.minute)
                 if form.family in ("H", "H:MM") or "+" in form.family]
    dates = sorted({s.due.date() for s in specs})
    all_days = [d for day in dates for d in day_forms(day)] + [MORNING]
    all_days += [d for s in specs if s.day for d in TEXT_DAYS[s.day]]
    primary_days = [day_forms(day)[0] for day in dates] + [day_forms(day)[-1] for day in dates]
    for base in sorted({s.base for s in specs} | {BASE}):
        bare = Phrase(f"de {base}", base)
        add(bare)
        for prep, form in tails:
            timed = bare.with_clock(prep, form, "tail")
            add(timed)
            for day in primary_days:
                add(timed.with_day(day, "tail"))
        for day in all_days:
            add(bare.with_day(day, "tail"))
    for spec in specs:
        if spec.clock is None and spec.day is None:
            continue
        for clock in ([None] if spec.clock is None else _restated(spec.clock)):
            for day in ((None,) if spec.day is None else TEXT_DAYS[spec.day]):
                body = Phrase(f"de {spec.base}", spec.base)
                if clock is not None:
                    body = body.with_clock("a", clock, "text")
                if day is not None:
                    body = body.with_day(day, "text")
                add(body)
                for prep, form in due_tails:
                    add(body.with_clock(prep, form, "tail"))
                for extra in primary_days:
                    add(body.with_day(extra, "tail"))
    return list(out.values())


class _Store:
    """Only what ReminderManager reads: the scoped active reminders in trigger order."""

    def __init__(self, reminders: list[Reminder]) -> None:
        self._reminders = reminders

    async def get_active_for_user(self, user_id, *, guard=None, shared_only=False):
        return sorted((r for r in self._reminders if r.state is ReminderState.ACTIVE
                       and (not shared_only or r.scope is ReminderScope.SHARED)), key=lambda r: r.trigger_at)


def _reminders(specs: tuple[Spec, ...]) -> list[Reminder]:
    return [Reminder(id=f"r{i}", user_id="household", text=s.text, trigger_at=s.due.timestamp(),
                     created_at=NOW.timestamp(), scope=ReminderScope.SHARED) for i, s in enumerate(specs)]


def _manager(store) -> ReminderManager:
    return ReminderManager(store=store, config={}, clock=lambda: NOW.timestamp())


def _describe(specs: tuple[Spec, ...]) -> list[str]:
    return [f"r{i} '{s.text}' due {s.due:%a %d %H:%M}" for i, s in enumerate(specs)]


class Tally:
    """Counts and counterexample classes, each with its shortest repro."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.classes: dict[tuple, dict] = {}
        self.class_counts: Counter = Counter()

    def counterexample(self, key: tuple, repro: dict) -> None:
        self.counts["counterexamples"] += 1
        self.class_counts[key] += 1
        size = (len(repro["reminders"]), len(repro["said"]))
        best = self.classes.get(key)
        if best is None or size < (len(best["reminders"]), len(best["said"])):
            self.classes[key] = repro

    def report(self, name: str) -> dict:
        classes = [{"class": " | ".join(key), "count": self.class_counts[key], **self.classes[key]}
                   for key in sorted(self.classes, key=lambda k: -self.class_counts[k])]
        data = {"counts": dict(self.counts), "classes": classes}
        prefix = os.environ.get("T032_PROP_REPORT")
        if prefix:
            with open(f"{prefix}-{name}.json", "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
        return data

    def message(self, data: dict) -> str:
        lines = [json.dumps(data["counts"])]
        for item in data["classes"][:40]:
            lines.append(f"[{item['count']}] {item['class']}: said {item['said']!r} with {item['reminders']} -> "
                         f"{item['got']}; consistent {item['consistent']}")
        return "\n".join(lines)


def _judge(tally: Tally, kind: str, specs, phrase: Phrase, cancelled: set[int], asked: bool, said: str) -> None:
    oracle = {i for i, s in enumerate(specs) if consistent(s, phrase)}
    tally.counts["cases"] += 1
    tally.counts[f"oracle_{min(len(oracle), 2)}"] += 1
    if cancelled:
        if len(cancelled) == 1 and oracle == cancelled:
            tally.counts["correct_cancel"] += 1
            return
        (wrong,) = sorted(cancelled)[:1]
        reason = "cancelled with another reading open" if wrong in oracle else "cancelled an inconsistent reminder"
        others = sorted(specs[i].feature for i in oracle - {wrong})
        key = (reason, kind, " ".join(phrase.shape) or "base", f"cancelled {specs[wrong].feature}",
               "also " + ",".join(others))
        tally.counterexample(key, {"said": said, "reminders": _describe(specs),
                                   "got": f"cancelled {sorted(cancelled)}", "consistent": sorted(oracle)})
    elif len(oracle) == 1:
        tally.counts["over_ask" if asked else "miss_not_found"] += 1
    else:
        tally.counts["correct_no_cancel"] += 1


GRID = reminder_sets()


def test_the_oracle_reads_the_user_decisions():
    """The oracle is not vacuous: the RR-P1, FV3-1/2, Q1 and FV3-3 shapes come out as the decisions say."""
    two = Spec(BASE, TEXT_CLOCKS[0], None, at(1, 9)), Spec(BASE, None, None, at(1, 14))
    said = Phrase("de tomar pastillas de las 2", BASE, (clock_forms(14, 0)[0],))
    assert [consistent(s, said) for s in two] == [True, True]
    said = Phrase("de tomar pastillas de las 2 de la tarde", BASE, (Clock("las 2 de la tarde", walls(2, 0, "tarde"), ""),))
    assert [consistent(s, said) for s in two] == [True, True]
    said = Phrase("de tomar pastillas de las 9 de la mañana", BASE, (Clock("las 9 de la mañana", walls(9, 0, "mañana"), ""),))
    assert [consistent(s, said) for s in two] == [True, False]
    night = Spec(BASE, TEXT_CLOCKS[8], None, at(1, 9)), Spec(BASE, None, None, at(1, 20))  # "a las 20"
    said = Phrase("de tomar pastillas de las 8 de la noche", BASE, (Clock("las 8 de la noche", walls(8, 0, "noche"), ""),))
    assert [consistent(s, said) for s in night] == [True, True]
    bread = Spec(BASE, None, "mañana", at(0, 18)), Spec(BASE, None, None, at(1, 8))
    assert [consistent(s, Phrase("de tomar pastillas de mañana", BASE, days=(MORNING,))) for s in bread] == [True, True]
    assert [consistent(s, Phrase("de tomar pastillas", BASE)) for s in bread] == [True, True]
    assert not consistent(Spec(BASE, None, None, at(1, 9)), Phrase("de tomar pastillas de la una", BASE,
                                                                     (Clock("la una", walls(1, 0), ""),)))
    assert len(GRID) > 600


async def test_reminder_manager_never_cancels_a_reminder_another_reading_could_mean():
    tally = Tally()
    for kind, specs in GRID:
        manager = _manager(_Store(_reminders(specs)))
        for phrase in phrases_for(specs):
            resolution = await manager.resolve(None, phrase.spoken, shared_only=True)
            cancelled = {int(resolution.reminder.id[1:])} if resolution.match is CancelMatch.UNIQUE else set()
            _judge(tally, kind, specs, phrase, cancelled, resolution.match is CancelMatch.AMBIGUOUS, phrase.spoken)
    data = tally.report("manager")
    assert tally.counts["correct_cancel"] > 1000 and tally.counts["correct_no_cancel"] > 1000, tally.message(data)
    assert tally.counts["counterexamples"] == 0, tally.message(data)


MINIMAL_REPROS = (
    # Crossed clocks: "a la una" is r1's text and r0's due time, "de las 2" the reverse.
    ("crossed clocks", (Spec(BASE, TEXT_CLOCKS[0], None, at(1, 13)), Spec(BASE, TEXT_CLOCKS[1], None, at(1, 14))),
     Phrase("de tomar pastillas", BASE).with_clock("a", TEXT_CLOCKS[1], "text")
     .with_clock("de", clock_forms(14, 0)[0], "tail")),
    # Crossed days: "el lunes" is r0's text and r1's due day, "mañana" the reverse.
    ("crossed days", (Spec(BASE, None, "el lunes", at(1, 9)), Spec(BASE, None, "mañana", at(4, 9))),
     Phrase("de tomar pastillas", BASE).with_day(TEXT_DAYS["el lunes"][0], "text")
     .with_day(TEXT_DAYS["mañana"][0], "tail")),
    # Review FV3-3 (pinned known limit): a bare "de mañana" is the morning (Q3) or a text word.
    ("de mañana, day word", (Spec(BASE, None, "mañana", at(0, 18)), Spec(BASE, None, None, at(1, 8))),
     Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")),
    ("de mañana, clock period", (Spec(BASE, TEXT_CLOCKS[16], None, at(1, 9, 45)), Spec(BASE, None, None, at(1, 8))),
     Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")),
)


@pytest.mark.parametrize(("name", "specs", "phrase"), MINIMAL_REPROS, ids=[r[0] for r in MINIMAL_REPROS])
async def test_minimal_repro_asks_which(name, specs, phrase):
    assert [consistent(s, phrase) for s in specs] == [True, True]
    resolution = await _manager(_Store(_reminders(specs))).resolve(None, phrase.spoken, shared_only=True)
    assert resolution.match is CancelMatch.AMBIGUOUS, (phrase.spoken, _describe(specs), resolution.reminder)


_TIME = (r"(?P<art>las?) (?P<h>\d{1,2})(?::(?P<m>\d{2}))? "
         r"(?P<per>de la mañana|de la tarde|de la noche|de la madrugada|del mediodía)")
_DAY = r"(?P<day>de hoy|de pasado mañana|del (?:" + "|".join(WEEKDAYS) + "))"
_TAIL = re.compile(rf"(?: de (?P<time>{_TIME}))?(?: {_DAY})?")


def parse_reference(reference: str, specs: tuple[Spec, ...]) -> Phrase | None:
    """A proposed reference ("<text or base> [de <time>] [<day>]") as the oracle's phrase."""
    split = next(((spec, body, found) for spec in specs for body in (spec.text, spec.base)
                  if reference.startswith(body) and (found := _TAIL.fullmatch(reference[len(body):]))), None)
    if split is None:
        return None
    spec, body, found = split
    phrase = Phrase(f"de {spec.base}", spec.base)
    if body == spec.text:
        if spec.clock is not None:
            phrase = phrase.with_clock("a", spec.clock, "text")
        if spec.day is not None:
            phrase = phrase.with_day(TEXT_DAYS[spec.day][0], "text")
    if found.group("time"):
        period = found.group("per").split()[-1]
        clock = Clock(found.group("time"), walls(int(found.group("h")), int(found.group("m") or 0), period), "proposal")
        phrase = phrase.with_clock("de", clock, "tail")
    if found.group("day"):
        text = found.group("day")
        if text == "de hoy":
            day = Day(text, frozenset({TODAY}))
        elif text == "de pasado mañana":
            day = Day(text, frozenset({TODAY + timedelta(2)}))
        else:
            day = Day(text, weekday_dates(WEEKDAYS.index(text.split()[-1])))
        phrase = phrase.with_day(day, "tail")
    if phrase.spoken != f"de {reference}":
        return None
    return phrase


async def test_every_proposed_reference_names_exactly_its_own_reminder():
    tally = Tally()
    for kind, specs in GRID:
        reminders = _reminders(specs)
        manager = _manager(_Store(reminders))
        for with_time in (False, True):
            references = await manager.voice_references(None, tuple(reminders), with_time=with_time, shared_only=True)
            for rid, reference in references.items():
                own = int(rid[1:])
                tally.counts["references"] += 1
                phrase = parse_reference(reference, specs)
                if phrase is None:
                    tally.counterexample(("unparsed proposal", kind), {"said": reference, "reminders": _describe(specs),
                                                                       "got": rid, "consistent": []})
                    continue
                oracle = {i for i, s in enumerate(specs) if consistent(s, phrase)}
                if oracle != {own}:
                    key = ("proposed reference has another reading", kind, " ".join(phrase.shape) or "base",
                           f"names {specs[own].feature}",
                           "also " + ",".join(sorted(specs[i].feature for i in oracle - {own})))
                    tally.counterexample(key, {"said": f"de {reference}", "reminders": _describe(specs),
                                               "got": f"proposed for {rid}", "consistent": sorted(oracle)})
                resolution = await manager.resolve(None, f"de {reference}", shared_only=True)
                assert resolution.match is CancelMatch.UNIQUE and resolution.reminder.id == rid, reference
    data = tally.report("references")
    assert tally.counts["references"] > 500, tally.message(data)
    assert tally.counts["counterexamples"] == 0, tally.message(data)


VERBS = ("cancelá el", "borrá el", "eliminá el", "sacá el", "quitá el", "anulá el", "cancela el", "olvidate del",
         "cancelame el")
_PROPOSAL = re.compile(r"por ejemplo: cancelá el recordatorio de (.+?)\.(?:\s|$)")
# Every phrase shape above, sampled with a fixed stride, plus fixed shapes from earlier review rounds.
SAMPLE_STRIDE = 151


def _dispatch_sample() -> list[tuple[str, tuple[Spec, ...], Phrase]]:
    cases, k = [], 0
    for kind, specs in GRID:
        for phrase in phrases_for(specs):
            if k % SAMPLE_STRIDE == 0:
                cases.append((kind, specs, phrase))
            k += 1
    una, dos = TEXT_CLOCKS[1], TEXT_CLOCKS[0]
    lunes, manana = TEXT_DAYS["el lunes"][0], TEXT_DAYS["mañana"][0]
    fixed = (
        ("FV3-3", (Spec(BASE, None, "mañana", at(0, 18)), Spec(BASE, None, None, at(1, 8))),
         Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")),
        ("E", (Spec(BASE, FEW_CLOCKS[3], None, at(1, 14)), Spec(BASE, FEW_CLOCKS[4], None, at(1, 20))),
         Phrase("de tomar pastillas", BASE).with_clock("a", FEW_CLOCKS[3], "text")
         .with_clock("de", clock_forms(14, 0)[0], "tail")),
        # Crossed clocks: "a la una" text of one / due of the other, "de las 2" the reverse.
        ("E", (Spec(BASE, dos, None, at(1, 13)), Spec(BASE, una, None, at(1, 14))),
         Phrase("de tomar pastillas", BASE).with_clock("a", una, "text").with_clock("de", dos, "tail")),
        # Crossed days: "el lunes" text of one / due of the other, "mañana" the reverse.
        ("G", (Spec(BASE, None, "el lunes", at(1, 9)), Spec(BASE, None, "mañana", at(4, 9))),
         Phrase("de tomar pastillas", BASE).with_day(lunes, "text").with_day(manana, "tail")),
    )
    return cases + list(fixed)


async def test_dispatcher_sample_never_cancels_a_reminder_another_reading_could_mean(tmp_path):
    leaves = CountedLeaves(tmp_path)
    await leaves.start()
    leaves.reminder_manager = _manager(leaves.reminder_store)
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    tally = Tally()
    try:
        for n, (kind, specs, phrase) in enumerate(_dispatch_sample()):
            created = [await manager.create("household", s.text, s.due.timestamp(), scope=ReminderScope.SHARED)
                       for s in specs]
            index = {r.id: i for i, r in enumerate(created)}

            async def active() -> set[str]:
                return {r.id for r in await manager.get_active(None, shared_only=True)} & index.keys()

            prefix = "Nexa, " if n % 3 == 0 else ""
            said = f"{prefix}{VERBS[n % len(VERBS)]} recordatorio {phrase.spoken}{'.' if n % 2 else ''}"
            before = await active()
            result = await dispatcher.dispatch(user_id="unknown", text=said)
            gone = {index[i] for i in before - await active()}
            asked = result.response.startswith(("Encontré ", "Hay "))
            tally.counts["claimed" if result.path.value == "fast_reminder" else "unclaimed"] += 1
            _judge(tally, kind, specs, phrase, gone, asked, said)
            proposal = _PROPOSAL.search(result.response) if not gone else None
            if proposal is not None:
                tally.counts["proposals"] += 1
                follow = parse_reference(proposal.group(1), specs)
                before = await active()
                again = await dispatcher.dispatch(user_id="unknown", text=f"cancelá el recordatorio de {proposal.group(1)}")
                named = {index[i] for i in before - await active()}
                if follow is None:
                    tally.counterexample(("unparsed proposal", kind), {"said": proposal.group(1),
                                                                       "reminders": _describe(specs),
                                                                       "got": again.response, "consistent": []})
                else:
                    oracle = {i for i, s in enumerate(specs) if consistent(s, follow)}
                    if len(named) != 1 or oracle != named:
                        key = ("proposed command", kind, " ".join(follow.shape),
                               f"cancelled {sorted(specs[i].feature for i in named)}",
                               "consistent " + ",".join(sorted(specs[i].feature for i in oracle)))
                        tally.counterexample(key, {"said": proposal.group(1), "reminders": _describe(specs),
                                                   "got": f"cancelled {sorted(named)}", "consistent": sorted(oracle)})
            for reminder_id in await active():
                assert await leaves.reminder_store.cancel(reminder_id, shared_only=True)
    finally:
        await leaves.close()
    data = tally.report("dispatcher")
    assert tally.counts["cases"] > 300 and tally.counts["correct_cancel"] > 30, tally.message(data)
    assert tally.counts["counterexamples"] == 0, tally.message(data)
