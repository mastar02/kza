"""Generated property: a spoken reminder cancel never cancels a reminder unless it is the only candidate.

Independent verification of T032-U1. A deterministic grid (no randomness,
no new dependency) of 2-3 shared reminders that share a base text, where a
text may embed a clock (every spoken format) and/or a day word and where
due times are chosen so that a text clock coincides with another
reminder's due time. Cancel phrases are built from each reminder's text
and/or its due time in every format, with and without a period of the day.

The oracle is written from the user decisions, not from the implementation.
Oracle change (2026-11-18, user decision t032_design_20261118 in
t040-t041/user-decisions-20261110.json, "Elegir por número"): the earlier
oracle read each clock or day phrase as a due time or as words of a text
(T032-U1-Q1, RR-P1); it is replaced, and every grid is kept. A reminder is
now a *candidate* for a phrase when its text holds the phrase's base content
words, where only words that some active reminder's text holds narrow and
the time phrases said around the base never do. Nexa never guesses by the
spoken time: exactly one candidate -> it is cancelled; more -> it asks,
listing them by number; none -> nothing is found.

Expected: a cancel only when the candidate set is exactly that one reminder
(a wrong cancel is a counterexample); one candidate is cancelled directly
(asking is a counterexample unless the base holds a number, which the
implementation reads as a possible clock and never narrows by); when it
asks, every candidate is listed. Through the dispatcher, the numbered
answer cancels exactly the listed reminder of that number, and it is a
candidate.

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


def candidates(texts: list[str], base: str) -> set[int]:
    """The reminders a cancel naming ``base`` may mean (user decision t032_design_20261118).

    Written from the decision, not from the implementation: only the base's
    content words narrow, and only those some reminder's text holds; the
    time phrases said around the base never do. No known word: no candidate.
    """
    vocabulary = frozenset().union(*(words(text) for text in texts))
    known = words(base) & vocabulary
    return {i for i, text in enumerate(texts) if known <= words(text)} if known else set()


NUMBER_WORDS = frozenset(HOUR_WORDS.values()) | {"veinte"}


def has_number(base: str) -> bool:
    """The base holds a number, which the implementation may read as a clock and never narrows by."""
    return any(w.isdigit() or w in NUMBER_WORDS for w in TOKEN.findall(fold(base)))


def oracle(specs, phrase) -> set[int]:
    return candidates([s.text for s in specs], phrase.base)


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
        self.data: dict = {}

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
                         f"{item['got']}; candidates {item['candidates']}")
        return "\n".join(lines)


def _judge(tally: Tally, kind: str, specs, phrase, cancelled: set[int], asked: bool, said: str, *,
           listed: set[int] | None = None, shape: str | None = None, describe=None) -> None:
    """Count one case; a wrong cancel, a needless question or an unlisted candidate is a counterexample."""
    expected = oracle(specs, phrase)
    shape = (" ".join(phrase.shape) or "base") if shape is None else shape
    describe = describe or _describe
    tally.counts["cases"] += 1
    tally.counts[f"oracle_{min(len(expected), 2)}"] += 1
    repro = {"said": said, "reminders": describe(specs), "candidates": sorted(expected)}
    if cancelled:
        if len(cancelled) == 1 and expected == cancelled:
            tally.counts["correct_cancel"] += 1
            return
        (wrong,) = sorted(cancelled)[:1]
        reason = "cancelled with another candidate" if wrong in expected else "cancelled a non-candidate"
        others = sorted(specs[i].feature for i in expected - {wrong})
        tally.counterexample((reason, kind, shape, f"cancelled {specs[wrong].feature}", "also " + ",".join(others)),
                             {**repro, "got": f"cancelled {sorted(cancelled)}"})
    elif len(expected) == 1 and not (asked and has_number(phrase.base)):
        tally.counterexample(("one candidate not cancelled", kind, shape),
                             {**repro, "got": "asked" if asked else "not found"})
    elif expected and not asked:
        tally.counterexample(("candidates not asked", kind, shape), {**repro, "got": "not found"})
    elif asked and listed is not None and (not expected <= listed
                                           or (listed - expected and not has_number(phrase.base))):
        tally.counterexample(("listed set is not the candidate set", kind, shape),
                             {**repro, "got": f"listed {sorted(listed)}"})
    else:
        tally.counts["over_ask_number_base" if len(expected) == 1 else "correct_no_cancel"] += 1


GRID = reminder_sets()


def test_the_oracle_reads_the_user_decisions():
    """The oracle is not vacuous: no time phrase narrows, a distinct word does, an unknown word is ignored."""
    two = Spec(BASE, TEXT_CLOCKS[0], None, at(1, 9)), Spec(BASE, None, None, at(1, 14))
    said = Phrase("de tomar pastillas", BASE).with_clock("a", TEXT_CLOCKS[0], "text")
    assert oracle(two, said) == {0, 1}  # the text's own clock, said as such, never narrows
    assert oracle(two, said.with_clock("de", clock_forms(9, 0)[0], "tail")) == {0, 1}  # nor its due time
    bread = Spec(BASE, None, "mañana", at(0, 18)), Spec(BASE, None, None, at(1, 8))
    assert oracle(bread, Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")) == {0, 1}
    pink = Spec(SUPER, None, None, at(1, 14)), Spec(BASE, TEXT_CLOCKS[0], None, at(1, 9))
    assert oracle(pink, Phrase(f"de {SUPER}", SUPER)) == {0}
    assert oracle(pink, Phrase(f"de {BASE}", BASE)) == {0, 1}
    assert oracle(pink, Phrase("de tomar pastillas azules", "tomar pastillas azules")) == {0, 1}
    assert oracle(pink, Phrase("de comprar pan", "comprar pan")) == set()
    assert has_number("tomar 10 gotas") and has_number("tomar veinte gotas") and not has_number(SUPER)
    assert len(GRID) > 600


def _cancelled(resolution) -> tuple[set[int], bool, set[int]]:
    """Cancelled index, whether it asked, and the indices it would list."""
    cancelled = {int(resolution.reminder.id[1:])} if resolution.match is CancelMatch.UNIQUE else set()
    asked = resolution.match in (CancelMatch.AMBIGUOUS, CancelMatch.NO_REFERENCE)
    return cancelled, asked, {int(r.id[1:]) for r in resolution.candidates}


async def test_reminder_manager_cancels_only_the_single_candidate():
    tally = Tally()
    for kind, specs in GRID:
        manager = _manager(_Store(_reminders(specs)))
        for phrase in phrases_for(specs):
            cancelled, asked, listed = _cancelled(await manager.resolve(None, phrase.spoken, shared_only=True))
            _judge(tally, kind, specs, phrase, cancelled, asked, phrase.spoken, listed=listed)
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
    # Review FV3-3: a bare "de mañana" is never read as a time nor as the text's word.
    ("de mañana, day word", (Spec(BASE, None, "mañana", at(0, 18)), Spec(BASE, None, None, at(1, 8))),
     Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")),
    ("de mañana, clock period", (Spec(BASE, TEXT_CLOCKS[16], None, at(1, 9, 45)), Spec(BASE, None, None, at(1, 8))),
     Phrase("de tomar pastillas", BASE).with_day(MORNING, "tail")),
)


@pytest.mark.parametrize(("name", "specs", "phrase"), MINIMAL_REPROS, ids=[r[0] for r in MINIMAL_REPROS])
async def test_minimal_repro_asks_which(name, specs, phrase):
    assert oracle(specs, phrase) == {0, 1}
    resolution = await _manager(_Store(_reminders(specs))).resolve(None, phrase.spoken, shared_only=True)
    assert resolution.match is CancelMatch.AMBIGUOUS, (phrase.spoken, _describe(specs), resolution.reminder)
    assert {r.id for r in resolution.candidates} == {"r0", "r1"}


VERBS = ("cancelá el", "borrá el", "eliminá el", "sacá el", "quitá el", "anulá el", "cancela el", "olvidate del",
         "cancelame el")
# The numbered list of an asking reply ("1: tomar pastillas a las 14:00; 2: ...; ¿cuál?").
_LISTED = re.compile(r"(\d): (.+?)(?=; \d: |; ¿)")
ORDINAL = {1: ("primero", "primera"), 2: ("segundo", "segunda"), 3: ("tercero", "tercera")}
CARDINAL = {1: "uno", 2: "dos", 3: "tres"}
ANSWERS = (lambda k: f"el {ORDINAL[k][0]}", lambda k: f"la {ORDINAL[k][1]}", lambda k: f"el {k}",
           lambda k: CARDINAL[k], lambda k: f"Nexa, el {CARDINAL[k]}", lambda k: f"cancelá el {ORDINAL[k][0]}")
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


async def _dispatch_round(tmp_path, sample, describe, judge, report: str) -> Tally:
    """Say each sampled cancel through the dispatcher, then answer a listed number; judge both turns."""
    leaves = CountedLeaves(tmp_path)
    await leaves.start()
    leaves.reminder_manager = _manager(leaves.reminder_store)
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    tally = Tally()
    try:
        for n, (kind, specs, phrase) in enumerate(sample):
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
            listed = dict(_LISTED.findall(result.response)) if asked else {}
            labels = {i: {manager.voice_label(r), f"{manager.voice_label(r)} {manager.spoken_day_reference(r)}"}
                      for i, r in enumerate(created)}
            named_by = {int(k): {i for i, own in labels.items() if label in own} for k, label in listed.items()}
            judge(tally, kind, specs, phrase, gone, asked, said, listed=set().union(*named_by.values()))
            if named_by:
                number = n % len(named_by) + 1
                answer = ANSWERS[n % len(ANSWERS)](number)
                before = await active()
                again = await dispatcher.dispatch(user_id="unknown", text=answer)
                chosen = {index[i] for i in before - await active()}
                tally.counts["numbered_answers"] += 1
                if len(chosen) != 1 or not chosen <= named_by[number]:
                    tally.counterexample(("numbered answer", kind, answer),
                                         {"said": f"{said} / {answer}", "reminders": describe(specs),
                                          "got": f"{again.response!r} cancelled {sorted(chosen)}",
                                          "candidates": sorted(oracle(specs, phrase))})
            for reminder_id in await active():
                assert await leaves.reminder_store.cancel(reminder_id, shared_only=True)
    finally:
        await leaves.close()
    tally.data = tally.report(report)
    return tally


async def test_dispatcher_sample_cancels_only_the_single_candidate_or_the_numbered_one(tmp_path):
    tally = await _dispatch_round(tmp_path, _dispatch_sample(), _describe, _judge, "dispatcher")
    assert tally.counts["cases"] > 300 and tally.counts["correct_cancel"] > 30, tally.message(tally.data)
    assert tally.counts["numbered_answers"] > 200, tally.message(tally.data)
    assert tally.counts["counterexamples"] == 0, tally.message(tally.data)


# ---------------------------------------------------------------------------
# Extended grid (T032-U1 recheck of 94bfab16, fail-safe round failsafe_1).
#
# The recheck's grid of formats the grid above does not generate: clocks the
# earlier cancel grammar did not read ("las 2 y 10", "las 2 10", "las 3 menos
# cuarto", "las 2 menos 10", "las 2 en punto", "las 14 hs", "mediodía",
# "medianoche"), word hours with a period, periods said without a clock ("a
# la tarde", "por la noche", "de la noche", "esta noche"), weekday variants
# ("el próximo lunes", "el lunes que viene", "este lunes"), dates ("del 20",
# "el 20 de noviembre"), the day said before the clock and a leading relative
# clause. Judged by the same oracle (user decision t032_design_20261118): none
# of these time phrases narrows the candidates.
# ---------------------------------------------------------------------------

EXT_PERIOD_HOURS = {"mañana": frozenset(range(0, 12)), "madrugada": frozenset(range(0, 6)),
                    "tarde": frozenset(range(12, 21)), "noche": frozenset(range(20, 24)) | frozenset(range(0, 6))}
EXT_PERIOD_CENTER = {"mañana": 8, "madrugada": 3, "tarde": 15, "noche": 21}
EXT_SUFFIX = {"mañana": "de la mañana", "madrugada": "de la madrugada", "tarde": "de la tarde", "noche": "de la noche"}


@dataclass(frozen=True)
class ExtT:
    """A time phrase: a clock (walls), a day (dates), a period (hours) or a day with a period ("dp")."""

    text: str
    kind: str
    walls: frozenset = frozenset()
    dates: frozenset = frozenset()
    hours: frozenset = frozenset()
    family: str = ""


@dataclass(frozen=True)
class ExtSpec:
    base: str
    items: tuple  # ((prep, ExtT), ...) said in the text
    due: datetime

    @property
    def text(self) -> str:
        return " ".join([self.base] + [f"{p} {t.text}".strip() for p, t in self.items])

    @property
    def feature(self) -> str:
        return "+".join(f"text-{t.kind}:{t.family}" for _, t in self.items) or (
            "plain" if self.base == BASE else f"base:{self.base}")


@dataclass(frozen=True)
class ExtPhrase:
    base: str
    items: tuple = ()
    prefix: str = ""

    @property
    def core(self) -> str:
        return " ".join([f"de {self.base}"] + [f"{p} {t.text}".strip() for p, t in self.items])

    @property
    def spoken(self) -> str:
        return f"{self.prefix} {self.core}".strip()

    @property
    def shape(self) -> str:
        return " ".join(["prefix" if self.prefix else ""] + [f"{p}|{t.kind}:{t.family}" for p, t in self.items]).strip()

    def plus(self, prep: str, t: ExtT) -> "ExtPhrase":
        return ExtPhrase(self.base, self.items + ((prep, t),), self.prefix)


def ext_new_clock_forms(h24: int, minute: int) -> list[ExtT]:
    """Spoken clocks the first grid does not generate, for one wall time."""
    h12 = h24 % 12 or 12
    art = "la" if h12 == 1 else "las"
    out = []
    if minute == 0:
        out += [ExtT(f"{art} {h12} en punto", "clock", walls(h12, 0), family="en punto"),
                ExtT(f"{art} {h12} hs", "clock", walls(h12, 0), family="H hs"),
                ExtT(f"{art} {h12}:00", "clock", walls(h12, 0), family="H:00")]
        if h24 > 12:
            out += [ExtT(f"las {h24} hs", "clock", walls(h24, 0), family="24h hs"),
                    ExtT(f"las {h24} horas", "clock", walls(h24, 0), family="24h horas"),
                    ExtT(f"las {h24}:00", "clock", walls(h24, 0), family="24h:00")]
        for period, suffix in EXT_SUFFIX.items():
            meant = walls(h12, 0, period)
            if (h24, 0) in meant and h24 % 24 in EXT_PERIOD_HOURS[period] | {12}:
                out.append(ExtT(f"{art} {HOUR_WORDS[h12]} {suffix}", "clock", meant, family=f"word+{period}"))
        if (h24, 0) == (12, 0):
            out.append(ExtT("mediodía", "clock", frozenset({(12, 0)}), family="mediodía"))
        if (h24, 0) == (0, 0):
            out += [ExtT("medianoche", "clock", frozenset({(0, 0)}), family="medianoche"),
                    ExtT("la medianoche", "clock", frozenset({(0, 0)}), family="la medianoche")]
    else:
        out.append(ExtT(f"{art} {h12} y {minute}", "clock", walls(h12, minute), family="y N"))
        if minute >= 10:
            out.append(ExtT(f"{art} {h12} {minute}", "clock", walls(h12, minute), family="H MM"))
        if minute == 20:
            out.append(ExtT(f"{art} {h12} y veinte", "clock", walls(h12, minute), family="y veinte"))
        if minute > 30:
            nxt = h12 % 12 + 1
            nart = "la" if nxt == 1 else "las"
            out.append(ExtT(f"{nart} {nxt} menos {60 - minute}", "clock", walls(h12, minute), family="menos N"))
            if minute == 45:
                out.append(ExtT(f"{nart} {nxt} menos cuarto", "clock", walls(h12, minute), family="menos cuarto"))
        if minute < 30:
            for period, suffix in (("tarde", "de la tarde"), ("noche", "de la noche")):
                meant = walls(h12, minute, period)
                if (h24, minute) in meant:
                    out.append(ExtT(f"{art} {h12} y {minute} {suffix}", "clock", meant, family=f"y N+{period}"))
    return out


def ext_old_clock_forms(h24: int, minute: int) -> list[ExtT]:
    return [ExtT(c.text, "clock", c.walls, family=c.family) for c in clock_forms(h24, minute)]


def ext_preps(t: ExtT) -> list[str]:
    if t.family == "mediodía":
        return ["al", "del"]
    if t.family == "medianoche":
        return ["a"]
    return ["de", "a"]


def ext_weekday_forms(weekday: int) -> list[ExtT]:
    name = WEEKDAYS[weekday]
    dates = weekday_dates(weekday)
    return [ExtT(text, "day", dates=dates, family=f"wd:{kind}") for text, kind in (
        (f"el próximo {name}", "el próximo X"), (f"el {name} que viene", "el X que viene"),
        (f"este {name}", "este X"), (f"el {name} próximo", "el X próximo"), (f"del {name}", "del X"),
        (f"el {name}", "el X"))]


def ext_date_forms(day: date) -> list[ExtT]:
    one = frozenset({day})
    name = WEEKDAYS[day.weekday()]
    return [ExtT(f"del {day.day}", "day", dates=one, family="del N"),
            ExtT(f"el {day.day}", "day", dates=one, family="el N"),
            ExtT(f"el {day.day} de noviembre", "day", dates=one, family="el N de mes"),
            ExtT(f"del {name} {day.day}", "day", dates=one, family="del X N")]


def ext_rel_day_forms(day: date) -> list[ExtT]:
    off = (day - TODAY).days
    one = frozenset({day})
    out = []
    if off == 0:
        out += [ExtT("hoy", "day", dates=one, family="hoy"), ExtT("de hoy", "day", dates=one, family="de hoy")]
    if off == 1:
        out += [ExtT("mañana", "day", dates=one, family="mañana"),
                ExtT("para mañana", "day", dates=one, family="para mañana")]
    if off == 2:
        out += [ExtT("pasado mañana", "day", dates=one, family="pasado mañana")]
    return out


def ext_period_forms(period: str) -> list[ExtT]:
    hours = EXT_PERIOD_HOURS[period]
    return [ExtT(f"{lead} {period}", "period", hours=hours, family=f"{lead} X")
            for lead in ("a la", "por la", "de la", "en la")]


def ext_day_period_forms(period: str) -> list[ExtT]:
    return [ExtT(f"esta {period}", "dp", dates=frozenset({TODAY}), hours=EXT_PERIOD_HOURS[period],
                 family=f"esta {period}")]


EXT_TEXT_NEW_CLOCKS = [form for wall in ((14, 10), (14, 45), (14, 20), (14, 0), (12, 0), (0, 0), (20, 0), (2, 50))
                       for form in ext_new_clock_forms(*wall)]
EXT_NUMBER_BASES = {"y N": ("tomar 10 gotas", (14, 0), (14, 10)), "H MM": ("tomar 10 gotas", (14, 0), (14, 10)),
                    "y veinte": ("tomar veinte gotas", (14, 0), (14, 20)),
                    "menos N": ("tomar 10 gotas", (15, 0), (14, 50)),
                    "en punto": ("dar punto final", (14, 0), (2, 0)), "H hs": ("tomar pastillas", (14, 0), (2, 0))}


def ext_reminder_sets() -> list[tuple[str, tuple[ExtSpec, ...]]]:
    out = []
    # N1: a text holds a new-format clock; the other is due at one of its wall times or elsewhere.
    for tc in EXT_TEXT_NEW_CLOCKS:
        prep = ext_preps(tc)[-1]
        for due1 in (at(1, 9, 45), at(1, *sorted(tc.walls)[0])):
            for wall in sorted(tc.walls) + [(17, 0)]:
                for offset in (1, 4):
                    out.append(("N1", (ExtSpec(BASE, ((prep, tc),), due1), ExtSpec(BASE, (), at(offset, *wall)))))
    # N2: the same text twice, one due at the hour and one at the minutes a new format names; and a
    # text that holds the leftover number.
    for _family, (base, w0, w1) in EXT_NUMBER_BASES.items():
        for d0, d1 in ((1, 1), (1, 4)):
            out.append(("N2", (ExtSpec(base, (), at(d0, *w0)), ExtSpec(base, (), at(d1, *w1)))))
    for minute in (10, 20, 45):
        num = {10: "10", 20: "veinte", 45: "45"}[minute]
        clock = ExtT(f"las {num}", "clock", walls(int(num) if num.isdigit() else 20, 0), family="H")
        out.append(("N2b", (ExtSpec(BASE, (("a", clock),), at(1, 14)), ExtSpec(BASE, (), at(1, 14, minute)))))
        out.append(("N2b", (ExtSpec(f"{BASE} {num}", (), at(1, 14)), ExtSpec(BASE, (), at(1, 14, minute)))))
    # N3: a text holds a period without a clock; the other is due inside that period or outside.
    for period in EXT_PERIOD_HOURS:
        for tp in ext_period_forms(period) + ext_day_period_forms(period):
            outside = 9 if period not in ("mañana", "madrugada") else 18
            for due0 in (at(1, outside, 45), at(0, outside, 45)):
                for d, h in ((1, EXT_PERIOD_CENTER[period]), (0, EXT_PERIOD_CENTER[period]),
                             (1, 17 if outside == 9 else 10)):
                    out.append(("N3", (ExtSpec(BASE, (("", tp),), due0), ExtSpec(BASE, (), at(d, h)))))
    # N4: weekday variants and dates in a text; the other due on that day or another.
    for td in ext_weekday_forms(0):
        for due0 in (at(1, 9), at(4, 9)):
            for off in (4, 1, 11):
                out.append(("N4", (ExtSpec(BASE, (("", td),), due0), ExtSpec(BASE, (), at(off, 9)))))
    for td in ext_date_forms(TODAY + timedelta(days=1)):
        for off in (1, 2):
            out.append(("N4", (ExtSpec(BASE, (("", td),), at(2, 9)), ExtSpec(BASE, (), at(off, 9)))))
    twenty = ExtT("las 20", "clock", walls(20, 0), family="24h")
    for off in (1, 2):
        out.append(("N4", (ExtSpec(BASE, (("a", twenty),), at(2, 9)), ExtSpec(BASE, (), at(off, 9)))))
    # N5: day before clock: two plain reminders due at two times and days a clock+day can name.
    for w0, w1 in (((14, 0), (2, 0)), ((14, 0), (14, 30)), ((9, 0), (21, 0))):
        for d0, d1 in ((1, 4), (1, 0), (0, 1), (1, 1)):
            out.append(("N5", (ExtSpec(BASE, (), at(d0, *w0)), ExtSpec(BASE, (), at(d1, *w1)))))
    # N6: crossed text clock and day with a new format.
    for tc in (ExtT("las 2 y 10", "clock", walls(2, 10), family="y N"),
               ExtT("las 3 menos cuarto", "clock", walls(2, 45), family="menos cuarto"),
               ExtT("mediodía", "clock", frozenset({(12, 0)}), family="mediodía")):
        prep = ext_preps(tc)[-1]
        wall = sorted(tc.walls)[-1]
        nine = ExtT("las 9", "clock", walls(9, 0), family="H")
        out.append(("N6", (ExtSpec(BASE, ((prep, tc),), at(1, 9)), ExtSpec(BASE, (("a", nine),), at(1, *wall)))))
        out.append(("N6", (ExtSpec(BASE, ((prep, tc),), at(1, 17)), ExtSpec(BASE, (), at(1, *wall)),
                           ExtSpec(BASE, (("", ext_rel_day_forms(TODAY + timedelta(1))[0]),), at(4, *wall)))))
    return out


def ext_phrases_for(specs: tuple[ExtSpec, ...]) -> list[ExtPhrase]:
    out: dict[str, ExtPhrase] = {}

    def add(phrase: ExtPhrase) -> None:
        out.setdefault(phrase.spoken, phrase)

    wall_times = sorted({(s.due.hour, s.due.minute) for s in specs}
                        | {w for s in specs for _, t in s.items if t.kind == "clock" for w in t.walls})
    clocks = []
    for wall in wall_times:
        clocks += [(p, t) for t in ext_new_clock_forms(*wall) for p in ext_preps(t)]
        clocks += [("de", t) for t in ext_old_clock_forms(*wall)[:3]]
    days = []
    for d in sorted({s.due.date() for s in specs}):
        days += [("", t) for t in ext_rel_day_forms(d)] + [("", t) for t in ext_weekday_forms(d.weekday())[:3]]
        days += [("", t) for t in ext_date_forms(d)[:2]]
    periods = []
    hours = {s.due.hour for s in specs}
    for period, span in EXT_PERIOD_HOURS.items():
        if hours & span or any(t.kind in ("period", "dp") and period in t.text for s in specs for _, t in s.items):
            periods += [("", t) for t in ext_period_forms(period)[:3] + ext_day_period_forms(period)]
    for base in sorted({s.base for s in specs} | {BASE}):
        bare = ExtPhrase(base)
        add(bare)
        add(ExtPhrase(base, (), "que te pedí"))
        for c in clocks:
            timed = bare.plus(*c)
            add(timed)
            add(ExtPhrase(base, timed.items, "que puse"))
            for d in days[:6]:
                add(timed.plus(*d))
                add(bare.plus(*d).plus(*c))  # day before clock
        for d in days:
            add(bare.plus(*d))
            for p in periods:
                add(bare.plus(*d).plus(*p))
        for p in periods:
            add(bare.plus(*p))
            for c in clocks[:8]:
                add(bare.plus(*c).plus(*p))
    for spec in specs:
        if not spec.items:
            continue
        body = ExtPhrase(spec.base, spec.items)
        add(body)
        for c in clocks:
            add(body.plus(*c))
        for d in days:
            add(body.plus(*d))
        for p in periods:
            add(body.plus(*p))
        # The text's own time phrase said in another form.
        for prep, own in spec.items:
            alts = []
            if own.kind == "clock":
                for wall in sorted(own.walls):
                    alts += [(pp, t) for t in ext_new_clock_forms(*wall) + ext_old_clock_forms(*wall)
                             for pp in ext_preps(t)]
            elif own.kind == "day" and own.family.startswith("wd"):
                alts = [("", t) for t in ext_weekday_forms(sorted(own.dates)[0].weekday())]
            elif own.kind == "period":
                alts = [("", t) for t in ext_period_forms(next(k for k, v in EXT_PERIOD_HOURS.items()
                                                                if v == own.hours))]
            for alt in alts:
                restated = ExtPhrase(spec.base, tuple(alt if it == (prep, own) else it for it in spec.items))
                add(restated)
                for c in clocks[:6]:
                    add(restated.plus(*c))
                for d in days[:4]:
                    add(restated.plus(*d))
    return list(out.values())


def _ext_reminders(specs: tuple[ExtSpec, ...]) -> list[Reminder]:
    return [Reminder(id=f"r{i}", user_id="household", text=s.text, trigger_at=s.due.timestamp(),
                     created_at=NOW.timestamp(), scope=ReminderScope.SHARED) for i, s in enumerate(specs)]


def _ext_describe(specs: tuple[ExtSpec, ...]) -> list[str]:
    return [f"r{i} '{s.text}' due {s.due:%a %d %H:%M}" for i, s in enumerate(specs)]


def _ext_class_of(phrase: ExtPhrase, specs: tuple[ExtSpec, ...]) -> str:
    kinds = {t.kind for _, t in phrase.items} | {t.kind for s in specs for _, t in s.items}
    fams = {t.family for _, t in phrase.items} | {t.family for s in specs for _, t in s.items}
    if kinds & {"period", "dp"}:
        return "period without a clock"
    if fams & {"y N", "H MM", "y veinte", "menos N", "menos cuarto", "y N+tarde", "y N+noche"}:
        return "minutes the cancel grammar does not read (y N / menos / H MM)"
    if fams & {"mediodía", "medianoche", "la medianoche"}:
        return "mediodía/medianoche without a clock number"
    if fams & {"en punto", "H hs", "24h hs", "24h horas", "H:00", "24h:00"}:
        return "en punto / hs / horas / :00"
    if any(f.startswith("word+") for f in fams):
        return "word hour with a period"
    if any(f.startswith("wd:") for f in fams):
        return "weekday variant"
    if fams & {"del N", "el N", "el N de mes", "del X N"}:
        return "date (del 20)"
    return "other"


def _ext_judge(tally: Tally, kind: str, specs, phrase: ExtPhrase, cancelled: set[int], asked: bool, said: str,
               listed: set[int] | None = None) -> None:
    _judge(tally, f"{_ext_class_of(phrase, specs)} | {kind}", specs, phrase, cancelled, asked, said, listed=listed,
           shape=phrase.shape or "base", describe=_ext_describe)


EXT_GRID = ext_reminder_sets()


def test_the_extended_oracle_reads_the_user_decisions():
    """The extended oracle is not vacuous: unread minutes, periods and weekdays never narrow, a number base does."""
    two = (ExtSpec("tomar 10 gotas", (), at(1, 14)), ExtSpec("tomar 10 gotas", (), at(1, 14, 10)))
    said = ExtPhrase("tomar 10 gotas").plus("de", ExtT("las 2 y 10", "clock", walls(2, 10), family="y N"))
    assert oracle(two, said) == {0, 1}
    noon = (ExtSpec(BASE, (("al", ExtT("mediodía", "clock", frozenset({(12, 0)}), family="mediodía")),), at(1, 9)),
            ExtSpec(BASE, (), at(1, 12)))
    assert oracle(noon, ExtPhrase(BASE).plus("al", noon[0].items[0][1])) == {0, 1}
    evening = (ExtSpec(BASE, (), at(1, 15)), ExtSpec(BASE, (), at(1, 19)), ExtSpec(BASE, (), at(1, 21)))
    assert oracle(evening, ExtPhrase(BASE).plus("", ext_period_forms("tarde")[0])) == {0, 1, 2}
    monday = (ExtSpec(BASE, (("", ext_weekday_forms(0)[3]),), at(1, 9)), ExtSpec(BASE, (), at(4, 9)))
    assert oracle(monday, ExtPhrase(BASE).plus("", ext_weekday_forms(0)[0])) == {0, 1}
    ten = (ExtSpec(f"{BASE} 10", (), at(1, 14)), ExtSpec(BASE, (), at(1, 14, 10)))
    assert oracle(ten, ExtPhrase(f"{BASE} 10")) == {0} and has_number(f"{BASE} 10")
    assert oracle(ten, ExtPhrase(BASE, (), "que puse").plus("", ext_period_forms("tarde")[0])) == {0, 1}
    assert len(EXT_GRID) > 300


async def test_extended_grid_cancels_only_the_single_candidate():
    """Every set of the extended grid shares its base words, so under the user decision none is cancelled.

    The earlier floor of correct cancels (the time read as a due time) has
    no case left under the oracle change; it is replaced by: no case cancels,
    and the only single candidates are bases holding a number (asked).
    """
    tally = Tally()
    for kind, specs in EXT_GRID:
        manager = _manager(_Store(_ext_reminders(specs)))
        for phrase in ext_phrases_for(specs):
            cancelled, asked, listed = _cancelled(await manager.resolve(None, phrase.spoken, shared_only=True))
            _ext_judge(tally, kind, specs, phrase, cancelled, asked, phrase.spoken, listed)
    data = tally.report("ext-manager")
    assert tally.counts["correct_cancel"] == 0 and tally.counts["correct_no_cancel"] > 10000, tally.message(data)
    assert tally.counts["over_ask_number_base"] == tally.counts["oracle_1"] > 0, tally.message(data)
    assert tally.counts["counterexamples"] == 0, tally.message(data)


# The recheck's end-to-end repros at 94bfab16 (EXT-1..5): each cancelled a reminder another reading could mean.
EXT_RECHECK_REPROS = (
    ("y-N same text", [("tomar 10 gotas", at(1, 14)), ("tomar 10 gotas", at(1, 14, 10))],
     "de tomar 10 gotas de las 2 y 10"),
    ("y-N text number", [("tomar pastillas a las 10", at(1, 14)), ("tomar pastillas", at(1, 14, 10))],
     "de tomar pastillas de las 2 y 10"),
    ("y-N text clock", [("tomar pastillas a las 2 y 10", at(1, 9)), ("tomar pastillas", at(1, 14, 10))],
     "de tomar pastillas de las 2 y 10"),
    ("menos cuarto text clock", [("tomar pastillas a las 3 menos cuarto", at(1, 9)),
                                 ("tomar pastillas", at(1, 14, 45))],
     "de tomar pastillas de las 3 menos cuarto"),
    ("menos cuarto vs 2:45", [("tomar pastillas a las 3 menos cuarto", at(1, 9)), ("tomar pastillas", at(1, 14, 45))],
     "de tomar pastillas de las 2:45"),
    ("en punto text clock", [("tomar pastillas a las 2 en punto", at(1, 9)), ("tomar pastillas", at(1, 14))],
     "de tomar pastillas de las 2 en punto"),
    ("al mediodía text", [("tomar pastillas al mediodía", at(1, 9)), ("tomar pastillas", at(1, 12))],
     "de tomar pastillas del mediodía"),
    ("las 12 vs text mediodía", [("tomar pastillas al mediodía", at(1, 9)), ("tomar pastillas", at(1, 12))],
     "de tomar pastillas de las 12"),
    ("medianoche text", [("tomar pastillas a medianoche", at(1, 9)), ("tomar pastillas", at(1, 0))],
     "de tomar pastillas de medianoche"),
    ("a la noche text", [("tomar pastillas a la noche", at(1, 9)), ("tomar pastillas", at(1, 21))],
     "de tomar pastillas de la noche"),
    ("a la tarde spoken", [("tomar pastillas a la tarde", at(1, 9)), ("tomar pastillas", at(1, 15))],
     "de tomar pastillas a la tarde"),
    ("weekday variant", [("tomar pastillas el próximo lunes", at(1, 9)), ("tomar pastillas", at(4, 9))],
     "de tomar pastillas el lunes que viene"),
    ("date del 20", [("tomar pastillas a las 20", at(2, 9)), ("tomar pastillas", at(1, 9))],
     "de tomar pastillas del 20"),
)


async def test_recheck_repros_ask_through_the_dispatcher(tmp_path):
    """Each EXT-1..5 repro asks which reminder end to end and cancels none."""
    leaves = CountedLeaves(tmp_path)
    await leaves.start()
    leaves.reminder_manager = _manager(leaves.reminder_store)
    dispatcher = leaves.dispatcher()
    manager = leaves.reminder_manager
    rows = []
    try:
        for name, reminders, reference in EXT_RECHECK_REPROS:
            created = [await manager.create("household", t, d.timestamp(), scope=ReminderScope.SHARED)
                       for t, d in reminders]
            ids = {r.id for r in created}
            result = await dispatcher.dispatch(user_id="unknown", text=f"cancelá el recordatorio {reference}")
            active = {r.id for r in await manager.get_active(None, shared_only=True)} & ids
            rows.append((name, active == ids, result.path.value, result.response))
            for rid in active:
                assert await leaves.reminder_store.cancel(rid, shared_only=True)
    finally:
        await leaves.close()
    assert all(kept and path == "fast_reminder" and response.startswith("Encontré 2")
               for _, kept, path, response in rows), rows


EXT_SAMPLE_STRIDE = 797


def _ext_dispatch_sample() -> list[tuple[str, tuple[ExtSpec, ...], ExtPhrase]]:
    cases, k = [], 0
    for kind, specs in EXT_GRID:
        for phrase in ext_phrases_for(specs):
            if k % EXT_SAMPLE_STRIDE == 0:
                cases.append((kind, specs, phrase))
            k += 1
    return cases


async def test_extended_dispatcher_sample_cancels_only_the_single_candidate_or_the_numbered_one(tmp_path):
    tally = await _dispatch_round(tmp_path, _ext_dispatch_sample(), _ext_describe, _ext_judge, "ext-dispatcher")
    assert tally.counts["cases"] > 300 and tally.counts["correct_cancel"] == 0, tally.message(tally.data)
    assert tally.counts["numbered_answers"] > 200, tally.message(tally.data)
    assert tally.counts["counterexamples"] == 0, tally.message(tally.data)
