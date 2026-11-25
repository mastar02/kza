//! Ordered household claims. A claim owns the whole turn before device parsing.

use crate::grammar::{self, GrammarBlock, GrammarMatch, HomeCommand, Operation as DeviceOperation};
use crate::normalize::{WAKE_WORDS, fold, has_phrase, normalize_utterance, words};

mod lexicon;
use lexicon::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HouseholdFamily {
    Timer,
    Reminder,
    List,
    Routine,
    Intercom,
    Notification,
    DeferredAction,
    Compound,
    Energy,
    Calendar,
    Media,
    CameraHealth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HouseholdOperation {
    Create,
    Cancel,
    Stop,
    Pause,
    Resume,
    Disable,
    Query,
    AddItem,
    RemoveItem,
    Execute,
    Propose,
    Clarify,
}

use HouseholdFamily::*;
use HouseholdOperation::*;

#[derive(Debug, Clone, PartialEq)]
pub struct HouseholdPart {
    pub segment: String,
    pub intent: Option<HouseholdIntent>,
    pub command: Option<HomeCommand>,
    pub security_refusal: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HouseholdIntent {
    pub family: HouseholdFamily,
    pub operation: HouseholdOperation,
    pub rule: &'static str,
    pub complement: String,
    pub verb: String,
    pub reason: Option<&'static str>,
    pub parts: Vec<HouseholdPart>,
    /// Byte offsets into the input, for shielding literal intercom content.
    pub literal_span: Option<(usize, usize)>,
}

impl HouseholdIntent {
    fn new(
        family: HouseholdFamily,
        operation: HouseholdOperation,
        rule: &'static str,
        complement: String,
        verb: &str,
    ) -> Self {
        Self {
            family,
            operation,
            rule,
            complement,
            verb: verb.into(),
            reason: None,
            parts: Vec::new(),
            literal_span: None,
        }
    }

    fn because(mut self, reason: &'static str) -> Self {
        self.reason = Some(reason);
        self
    }

    fn controls_routine(&self) -> bool {
        self.family == Routine && matches!(self.operation, Execute | Stop | Disable)
    }

    pub(crate) fn integration_query(&self) -> bool {
        self.operation == Query && matches!(self.family, Energy | Calendar | Media | CameraHealth)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HouseholdIntentSettings {
    pub unvetoed_max_words: usize,
    pub compound_max_parts: usize,
    pub compound_execute: bool,
    pub wake_words: Vec<String>,
}

impl Default for HouseholdIntentSettings {
    fn default() -> Self {
        Self {
            unvetoed_max_words: 6,
            compound_max_parts: 3,
            compound_execute: true,
            wake_words: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HouseholdIntentClassifier {
    settings: HouseholdIntentSettings,
}

impl HouseholdIntentClassifier {
    pub(crate) fn prepare() {
        prepare_patterns();
        energy::prepare_patterns();
    }

    pub fn new(mut settings: HouseholdIntentSettings) -> Self {
        settings.wake_words = settings
            .wake_words
            .iter()
            .map(|word| words(word))
            .filter(|word| !word.is_empty())
            .collect();
        Self { settings }
    }

    pub fn settings(&self) -> &HouseholdIntentSettings {
        &self.settings
    }

    /// Veto changes admission only. It never frees a claimed turn for device execution.
    pub fn requires_veto(&self, text: &str, acoustic_wake: bool, tv_signal: bool) -> bool {
        tv_signal
            || !acoustic_wake
            || text.split_whitespace().count() > self.settings.unvetoed_max_words
    }

    pub fn classify(&self, text: &str) -> Option<HouseholdIntent> {
        self.classify_in_room(text, None)
    }

    pub(crate) fn classify_in_room(
        &self,
        text: &str,
        source_room: Option<&str>,
    ) -> Option<HouseholdIntent> {
        let all = Utterance::tokenize(text);
        let utt = all.strip_fillers(&self.settings.wake_words);
        if utt.is_quoted(0) {
            return None;
        }
        reminder(&utt)
            .or_else(|| intercom(&utt))
            .or_else(|| compound(&utt, &self.settings, source_room))
            .or_else(|| single(&utt))
    }

    /// Recognize a read-only routine status/schedule question without a routine store.
    pub(crate) fn routine_read(&self, core: &str) -> Option<HouseholdIntent> {
        let plain = words(core);
        let m = ROUTINE_READ.captures(&plain)?;
        let name = ["before", "after"].iter().find_map(|group| m.name(group))?;
        let name = name.as_str();
        // Predicates and further clauses cannot stand for a saved routine name.
        if name.split_whitespace().any(|word| {
            matches!(word, "si" | "cuando" | "porque" | "que") || grammar::action(word).is_some()
        }) {
            return None;
        }
        Some(HouseholdIntent::new(
            Routine,
            Query,
            "R1",
            name.to_owned(),
            "",
        ))
    }

    pub(crate) fn ring_stop(&self, text: &str) -> Option<HouseholdIntent> {
        let all = Utterance::tokenize(text);
        let utt = all.strip_fillers(&self.settings.wake_words);
        (!utt.is_quoted(0) && utt.folded == ["listo"]).then(|| {
            HouseholdIntent::new(Timer, Stop, "R1", String::new(), "listo").because("ring_stop")
        })
    }
}

#[derive(Debug, Clone)]
struct Token {
    spoken: String,
    start: usize,
    end: usize,
    quoted: bool,
}

#[derive(Debug, Clone)]
struct Utterance<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    folded: Vec<String>,
}

impl<'a> Utterance<'a> {
    fn tokenize(source: &'a str) -> Self {
        let mut quotes = quote_boundaries(source).into_iter().peekable();
        let mut quoted = false;
        let tokens: Vec<_> = TOKEN
            .find_iter(source)
            .map(|m| {
                while quotes.peek().is_some_and(|&i| i < m.start()) {
                    quotes.next();
                    quoted = !quoted;
                }
                Token {
                    spoken: normalize_utterance(m.as_str()),
                    start: m.start(),
                    end: m.end(),
                    quoted,
                }
            })
            .collect();
        let folded = tokens.iter().map(|token| fold(&token.spoken)).collect();
        Self {
            source,
            tokens,
            folded,
        }
    }

    fn strip_fillers(&self, wakes: &[String]) -> Self {
        let mut start = 0;
        let mut end = self.len();
        for _ in 0..6 {
            if start >= end || self.is_quoted(start) {
                break;
            }
            let mut next = None;
            for wake in wakes
                .iter()
                .map(String::as_str)
                .chain(WAKE_WORDS.iter().copied())
            {
                let mut tokens = self.folded[start..end].iter();
                if wake
                    .split_whitespace()
                    .all(|word| tokens.next().is_some_and(|t| t == word))
                {
                    next = Some(start + wake.split_whitespace().count());
                    break;
                }
            }
            let word = self.at(start);
            let paused_no = word == "no"
                && self.tokens.get(start + 1).is_some_and(|next| {
                    self.source[self.tokens[start].end..next.start]
                        .contains([',', ';', ':', '.', '!'])
                });
            start = match next {
                Some(next) => next,
                None if word == "por" && self.at(start + 1) == "favor" => start + 2,
                None if paused_no
                    || matches!(
                        word,
                        "che"
                            | "ey"
                            | "oye"
                            | "hola"
                            | "porfa"
                            | "podes"
                            | "podrias"
                            | "puedes"
                            | "podras"
                            | "bueno"
                            | "bue"
                            | "dale"
                    ) =>
                {
                    start + 1
                }
                None => break,
            };
        }
        while start < end && !self.is_quoted(end - 1) {
            if self.at(end - 1) == "porfa" {
                end -= 1;
            } else if end - start >= 2 && self.folded[end - 2..end] == ["por", "favor"] {
                end -= 2;
            } else {
                break;
            }
        }
        let core = self.slice(start, end);
        if core.at(1) == "ya" {
            let without = core.slice(0, 1).join(&core.slice(2, core.len()));
            if verb_at(&without, 0, false).is_some_and(|verb| verb.has(CANCEL)) {
                return without;
            }
        }
        core
    }

    fn is_quoted(&self, index: usize) -> bool {
        self.tokens.get(index).is_some_and(|token| token.quoted)
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }
    fn at(&self, index: usize) -> &str {
        self.folded.get(index).map_or("", String::as_str)
    }
    fn text(&self) -> String {
        self.folded.join(" ")
    }
    fn has(&self, vocab: &[&str]) -> bool {
        self.folded
            .iter()
            .any(|token| vocab.contains(&token.as_str()))
    }
    fn rest(&self, start: usize) -> String {
        self.tokens
            .iter()
            .skip(start)
            .map(|token| token.spoken.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn heard(&self) -> String {
        self.rest(0)
    }

    fn literal_after(&self, token: usize) -> Option<(usize, usize)> {
        let first = self.tokens.get(token)?;
        let last = self.tokens.last()?;
        let next = NEXT_WORD.find_at(self.source, last.end);
        let end = next.map_or(self.source.len(), |m| {
            last.end
                + self.source[last.end..m.start()]
                    .trim_end_matches([' ', '\t', ',', ';', ':'])
                    .len()
        });
        (first.end <= end).then_some((first.end, end))
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            source: self.source,
            tokens: self.tokens[start..end].to_vec(),
            folded: self.folded[start..end].to_vec(),
        }
    }
    fn join(mut self, other: &Self) -> Self {
        self.tokens.extend(other.tokens.iter().cloned());
        self.folded.extend(other.folded.iter().cloned());
        self
    }

    fn grammar_text(&self) -> String {
        let mut text = String::new();
        if self.is_quoted(0) {
            text.push('«');
        }
        for (i, token) in self.tokens.iter().enumerate() {
            if i > 0 {
                let previous = &self.tokens[i - 1];
                if let Some(between) = self.source.get(previous.end..token.start)
                    && !between.chars().any(char::is_alphanumeric)
                {
                    text.push_str(between);
                } else {
                    text.push(' ');
                }
            }
            text.push_str(&token.spoken);
        }
        if let Some(last) = self.tokens.last() {
            // A percent immediately after the segment's last token belongs to the segment.
            if self.source[last.end..].trim_start().starts_with('%') {
                text.push('%');
            }
        }
        text
    }
}

// Ignore apostrophes within words and keep a bounded stack for nested spoken quotes.
fn quote_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let mut stack = Vec::new();
    let mut chars = text.char_indices().peekable();
    let mut previous = None;
    while let Some((index, ch)) = chars.next() {
        let next = chars.peek().map(|&(_, ch)| ch);
        let apostrophe = ch == '\''
            && previous.is_some_and(char::is_alphanumeric)
            && next.is_some_and(char::is_alphanumeric);
        if !apostrophe {
            if stack.last() == Some(&ch) {
                stack.pop();
                if stack.is_empty() {
                    boundaries.push(index);
                }
            } else if stack.len() < 6 {
                let closer = match ch {
                    '«' => Some('»'),
                    '“' => Some('”'),
                    '"' | '\'' => Some(ch),
                    _ => None,
                };
                if let Some(closer) = closer {
                    if stack.is_empty() {
                        boundaries.push(index);
                    }
                    stack.push(closer);
                }
            }
        }
        previous = Some(ch);
    }
    boundaries
}

#[derive(Debug, Clone, Copy)]
struct Verb<'a> {
    form: &'a str,
    classes: u16,
    end: usize,
    infinitive: bool,
}

impl Verb<'_> {
    fn has(self, mask: u16) -> bool {
        self.classes & mask != 0
    }
}

fn verb_at<'a>(utt: &'a Utterance<'_>, index: usize, strict: bool) -> Option<Verb<'a>> {
    if utt.is_quoted(index) {
        return None;
    }
    let word = utt.at(index);
    if word == "olvidate" && matches!(utt.at(index + 1), "de" | "del") {
        return Some(Verb {
            form: word,
            classes: CANCEL,
            end: index + if utt.at(index + 1) == "de" { 2 } else { 1 },
            infinitive: false,
        });
    }
    if word == "para" {
        let accented = utt.tokens[index].spoken == "pará";
        if !accented
            && (strict || ((index + 1)..utt.len()).any(|i| verb_at(utt, i, true).is_some()))
        {
            return None;
        }
        return Some(Verb {
            form: word,
            classes: STOP,
            end: index + 1,
            infinitive: false,
        });
    }
    let forms = std::iter::once(word).chain(
        ENCLITICS
            .iter()
            .filter_map(|suffix| word.strip_suffix(suffix).filter(|form| form.len() >= 2)),
    );
    for form in forms {
        let classes = verb_classes(form);
        if classes == 0 {
            continue;
        }
        let infinitive = ["ar", "er", "ir"]
            .iter()
            .any(|ending| form.ends_with(ending));
        if strict && infinitive {
            return None;
        }
        return Some(Verb {
            form,
            classes,
            end: index + 1,
            infinitive,
        });
    }
    None
}

fn object(utt: &Utterance<'_>, i: usize, alarms: bool) -> Option<HouseholdFamily> {
    match utt.at(i) {
        "timer" | "timers" | "temporizador" | "temporizadores" => Some(Timer),
        "recordatorio" | "recordatorios" => Some(Reminder),
        "rutina" | "rutinas" => Some(Routine),
        "lista" | "listas" if !(utt.at(i + 1) == "de" && MUSIC_WORDS.contains(&utt.at(i + 2))) => {
            Some(List)
        }
        word if alarms && ALARM_NOUNS.contains(&word) => Some(Timer),
        _ => None,
    }
}

fn governed(
    utt: &Utterance<'_>,
    mut i: usize,
    adjective: bool,
    alarms: bool,
    automation: bool,
) -> Option<(HouseholdFamily, usize)> {
    if utt.at(i) == "me" {
        i += 1;
    }
    for _ in 0..2 {
        if DETERMINERS.contains(&utt.at(i)) {
            i += 1;
        }
    }
    if adjective && ADJECTIVES.contains(&utt.at(i)) {
        i += 1;
    }
    if automation && matches!(utt.at(i), "automatizacion" | "automatizaciones") {
        return Some((Routine, i));
    }
    object(utt, i, alarms).map(|family| (family, i))
}

fn reminder_pattern() -> &'static regex::Regex {
    &REMINDER_HEAD
}

fn reminder(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    let m = reminder_pattern().captures(&text)?;
    let used = m[0].split_whitespace().count();
    if matches!(utt.at(0), "acordate" | "acuerdate") && utt.at(used) == "que" {
        return None;
    }
    if m.name("other").is_some() {
        return Some(
            HouseholdIntent::new(Reminder, Create, "R3", utt.rest(used), utt.at(0))
                .because("other_person"),
        );
    }
    let mut rest = used;
    while matches!(utt.at(rest), "que" | "de") {
        rest += 1;
    }
    let intent = HouseholdIntent::new(Reminder, Create, "R3", utt.rest(rest), utt.at(0));
    if (used..utt.len())
        .any(|i| CONJUNCTIONS.contains(&utt.at(i)) && clause_head(utt, connector_end(utt, i)))
    {
        return Some(HouseholdIntent {
            operation: Clarify,
            ..intent.because("second_clause")
        });
    }
    Some(intent)
}

fn clock_alarm(tail: &str) -> bool {
    let text = fold(tail);
    if let Some(m) = CLOCK_ALARM_HOUR.find(&text) {
        let next = text[m.end()..]
            .split_whitespace()
            .next()
            .unwrap_or_default();
        if !matches!(
            next,
            "segundo" | "segundos" | "minuto" | "minutos" | "min" | "mins"
        ) {
            return true;
        }
    }
    CLOCK_ALARM_DAY.is_match(&text)
}

fn r1_object(
    utt: &Utterance<'_>,
    verb: Verb<'_>,
) -> Option<(HouseholdFamily, HouseholdOperation, usize)> {
    let (family, i) = governed(utt, verb.end, false, true, verb.has(CANCEL))?;
    if matches!(utt.at(i), "alarma" | "alarmas") && SECURITY_ALARM_VERBS.contains(&verb.form) {
        return None;
    }
    let operation = [
        (CANCEL, Cancel),
        (CLEAR, Cancel),
        (STOP | STOP_RINGING, Stop),
        (PAUSE, Pause),
        (RESUME, Resume),
        (DISABLE, Disable),
    ]
    .into_iter()
    .find_map(|(mask, op)| (verb.has(mask) && (mask != CLEAR || family == List)).then_some(op))?;
    Some((family, operation, i))
}

fn r1_intent(
    utt: &Utterance<'_>,
    family: HouseholdFamily,
    op: HouseholdOperation,
    i: usize,
    verb: &str,
) -> HouseholdIntent {
    let rest = utt.rest(i + 1);
    if ALARM_NOUNS.contains(&utt.at(i))
        && clock_alarm(&rest)
        && matches!(op, Cancel | Stop | Disable | Query)
    {
        return HouseholdIntent::new(
            Reminder,
            if op == Query { Query } else { Cancel },
            "R1",
            rest,
            verb,
        )
        .because("clock_alarm");
    }
    let complement = if family == Routine && matches!(op, Stop | Disable) {
        utterance_literal(utt, i)
    } else {
        rest
    };
    HouseholdIntent::new(family, op, "R1", complement, verb)
}

fn r1(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    if let Some(verb) = verb
        && let Some((family, op, i)) = r1_object(utt, verb)
    {
        return Some(r1_intent(utt, family, op, i, verb.form));
    }
    let head = utt.at(0);
    let (family, i) = if HAVE_HEADS.contains(&head) || READ_HEADS.contains(&head) {
        let skip = usize::from(HAVE_HEADS.contains(&head) && SOME.contains(&utt.at(1)));
        governed(utt, 1 + skip, false, true, false)?
    } else {
        let (i, family) =
            (0..utt.len()).find_map(|i| object(utt, i, true).map(|family| (i, family)))?;
        let remaining = head == "cuanto"
            && family == Timer
            && utt
                .folded
                .iter()
                .take(5)
                .any(|w| matches!(w.as_str(), "falta" | "queda"));
        let query = QUERY_HEADS.contains(&head) && utt.has(QUERY_VERBS);
        let alarm = head == "a"
            && utt.at(1) == "que"
            && utt.at(2) == "hora"
            && ALARM_NOUNS.contains(&utt.at(i));
        if !(remaining || query || alarm) {
            return None;
        }
        (family, i)
    };
    Some(r1_intent(utt, family, Query, i, head))
}

fn duration(text: &str) -> Option<regex::Match<'_>> {
    DURATION.find(text)
}

fn r2(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let Some(verb) = verb else {
        let tail = fold(&utt.rest(1));
        return (matches!(utt.at(0), "timer" | "temporizador")
            && (matches!(utt.at(1), "de" | "para" | "por")
                || duration(&tail).is_some_and(|m| m.start() == 0)))
        .then(|| HouseholdIntent::new(Timer, Create, "R2", utt.rest(1), ""));
    };
    let set = matches!(verb.form, "pone" | "pon" | "ponga" | "poner");
    if verb.has(CREATE | TIMER_CREATE)
        && let Some((family, i)) = governed(utt, verb.end, true, false, false)
        && (family == Timer || (family == List && verb.has(CREATE) && !set))
    {
        return Some(HouseholdIntent::new(
            family,
            Create,
            "R2",
            utt.rest(i + 1),
            verb.form,
        ));
    }
    if !(0..utt.len()).any(|i| object(utt, i, false) == Some(List)) {
        return None;
    }
    let list_target = set
        && !utt.has(MUSIC_WORDS)
        && (0..utt.len()).any(|i| {
            matches!(utt.at(i), "a" | "en")
                && matches!(utt.at(i + 1), "la" | "mi" | "nuestra")
                && object(utt, i + 2, false) == Some(List)
        });
    let op = if verb.has(ADD) || list_target {
        AddItem
    } else if verb.has(REMOVE) {
        RemoveItem
    } else {
        return None;
    };
    Some(HouseholdIntent::new(
        List,
        op,
        "R2",
        utt.rest(verb.end),
        verb.form,
    ))
}

fn alarm(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let (i, form) = if matches!(utt.at(0), "alarma" | "despertador") {
        (0, "")
    } else {
        let verb = verb.filter(|v| ALARM_VERBS.contains(&v.form))?;
        let i = verb.end + usize::from(utt.at(verb.end) == "me");
        let i = i + usize::from(DETERMINERS.contains(&utt.at(i)));
        if !matches!(utt.at(i), "alarma" | "despertador") {
            return None;
        }
        (i, verb.form)
    };
    let tail = utt.rest(i + 1);
    let text = fold(&tail);
    let article_unit = ARTICLE_CLOCK_UNIT.is_match(&text);
    let clock = CLOCK.is_match(&text);
    let (family, op, reason) = if article_unit {
        (Reminder, Create, "clock_time")
    } else if duration(&text).is_some() {
        (Timer, Create, "duration")
    } else if clock {
        (Reminder, Create, "clock_time")
    } else {
        (Timer, Clarify, "no_time")
    };
    Some(HouseholdIntent::new(family, op, "R3a", tail, form).because(reason))
}

fn utterance_literal(utt: &Utterance<'_>, noun: usize) -> String {
    utt.literal_after(noun).map_or_else(
        || utt.rest(noun + 1),
        |(start, end)| utt.source[start..end].trim().to_owned(),
    )
}

fn intercom(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false)?;
    if !verb.has(INTERCOM) || verb.infinitive {
        return None;
    }
    let head = utt.at(0);
    let next = utt.at(verb.end);
    let directed = matches!(next, "en" | "a" | "al" | "por" | "para");
    let announce = head == verb.form
        && (head.starts_with("anunci")
            || head.starts_with("comunic")
            || (matches!(head, "avisa" | "avise") && next == "que"));
    if !announce && (!directed || head.ends_with("me") || head.ends_with("nos")) {
        return None;
    }
    let mut intent = HouseholdIntent::new(
        Intercom,
        Execute,
        "R6",
        utterance_literal(utt, 0),
        verb.form,
    );
    if let Some((start, end)) = utt.literal_after(0) {
        let tail = &utt.source[start..end];
        let clean = tail
            .trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '¿' | '¡'));
        let lead = tail.len() - clean.len();
        let message = if clean.starts_with(['"', '\'', '«']) {
            lead
        } else if directed {
            INTERCOM_MARKER
                .find(clean)
                .map_or(tail.len(), |m| lead + m.end())
        } else if let Some(m) = LEADING_INTERCOM_MARKER.find(clean) {
            lead + m.end()
        } else {
            lead
        };
        intent.literal_span = Some((start + message, end));
    }
    Some(intent)
}

fn routine_creation(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    let m = ROUTINE_REQUEST.captures(&text)?;
    let mut i = m[0].split_whitespace().count();
    let request = m.name("request").map(|m| m.as_str());
    if m.name("que").is_some() && !request.is_some_and(|head| DESIRE_HEADS.contains(&head)) {
        return None;
    }
    let word = utt.at(i);
    let form = std::iter::once(word)
        .chain(
            ENCLITICS
                .iter()
                .filter_map(|suffix| word.strip_suffix(suffix)),
        )
        .find(|form| ROUTINE_CREATE_FORMS.contains(form) || ROUTINE_NEW_ONLY_FORMS.contains(form));
    let slot = form.is_none()
        && (request.is_some() || m.name("clitic").is_some())
        && !known_command_word(word)
        && !DETERMINERS.contains(&word)
        && !ADJECTIVES.contains(&word)
        && !ROUTINE_NOUNS.contains(&word);
    i += usize::from(form.is_some() || slot);
    let mut indefinite = false;
    for _ in 0..2 {
        if DETERMINERS.contains(&utt.at(i)) {
            indefinite |= INDEFINITE.contains(&utt.at(i));
            i += 1;
        }
    }
    let mut new = ADJECTIVES.contains(&utt.at(i));
    i += usize::from(new);
    if !ROUTINE_NOUNS.contains(&utt.at(i)) {
        return None;
    }
    if ADJECTIVES.contains(&utt.at(i + 1)) {
        new |= form.is_none() && !slot;
        i += 1;
    }
    if !(form.is_some_and(|f| ROUTINE_CREATE_FORMS.contains(&f)) || indefinite || new) {
        return None;
    }
    Some(
        HouseholdIntent::new(
            Routine,
            Propose,
            "R6",
            utterance_literal(utt, i).to_lowercase(),
            form.unwrap_or_default(),
        )
        .because("new_routine"),
    )
}

fn r6(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    if let Some(intent) = routine_creation(utt) {
        return Some(intent);
    }
    if let Some(verb) = verb {
        if verb.has(AUTOMATE) {
            return Some(
                HouseholdIntent::new(
                    Routine,
                    Propose,
                    "R6",
                    utterance_literal(utt, verb.end - 1).to_lowercase(),
                    verb.form,
                )
                .because("new_routine"),
            );
        }
        if let Some((Routine, noun)) = governed(utt, verb.end, true, false, false) {
            if verb.has(ROUTINE_EXEC) {
                return Some(HouseholdIntent::new(
                    Routine,
                    Execute,
                    "R6",
                    utterance_literal(utt, noun),
                    verb.form,
                ));
            }
            if verb.has(ROUTINE_PROPOSE) {
                return Some(HouseholdIntent::new(
                    Routine,
                    Propose,
                    "R6",
                    utt.rest(noun + 1),
                    verb.form,
                ));
            }
        }
    }
    let text = utt.text();
    let m = NOTIFICATION.captures(&text)?;
    Some(HouseholdIntent::new(
        Notification,
        if m.name("disable").is_some() {
            Disable
        } else {
            Execute
        },
        "R6",
        utt.heard(),
        verb.map_or("", |v| v.form),
    ))
}

fn deferred(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let verb = verb.filter(|v| v.has(DOMOTICS))?;
    (grammar::non_immediate(&utt.grammar_text()) == Some(GrammarBlock::Deferred))
        .then(|| HouseholdIntent::new(DeferredAction, Create, "R4", utt.heard(), verb.form))
}

fn single(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false);
    reminder(utt)
        .or_else(|| intercom(utt))
        .or_else(|| r1(utt, verb))
        .or_else(|| r2(utt, verb))
        .or_else(|| alarm(utt, verb))
        .or_else(|| deferred(utt, verb))
        .or_else(|| r6(utt, verb))
        .or_else(|| integration(utt))
}

fn clause_head(utt: &Utterance<'_>, i: usize) -> bool {
    !utt.is_quoted(i)
        && (verb_at(utt, i, true).is_some()
            || (i < utt.len() && reminder_pattern().is_match(&utt.folded[i..].join(" "))))
}

fn connector_end(utt: &Utterance<'_>, index: usize) -> usize {
    let mut next = index + 1;
    while CONNECTORS.contains(&utt.at(next)) {
        next += 1;
    }
    next
}

fn split_clauses<'a>(utt: &Utterance<'a>) -> Vec<Utterance<'a>> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut content = false;
    for i in 1..utt.len() {
        if i <= start {
            continue;
        }
        content |= CONTENT_MARKERS.contains(&utt.at(i - 1));
        if utt.is_quoted(i) || !CONJUNCTIONS.contains(&utt.at(i)) {
            continue;
        }
        if verb_classes(utt.at(start)) & INTERCOM != 0
            && intercom(&utt.slice(start, utt.len())).is_some()
        {
            break;
        }
        if content {
            return Vec::new();
        }
        let next = connector_end(utt, i);
        if clause_head(utt, next) {
            segments.push(utt.slice(start, i));
            start = next;
            content = false;
        }
    }
    if !segments.is_empty() {
        segments.push(utt.slice(start, utt.len()));
    }
    segments
}

fn split_objects<'a>(utt: &Utterance<'a>, verb: Verb<'_>) -> Vec<Utterance<'a>> {
    let Some((family, _, noun)) = r1_object(utt, verb) else {
        return Vec::new();
    };
    if family == Reminder && matches!(utt.at(noun + 1), "de" | "para" | "que") {
        return Vec::new();
    }
    let mut cuts = (noun + 1..utt.len())
        .filter_map(|i| {
            if utt.is_quoted(i) || utt.at(i) != "y" {
                return None;
            }
            let next = i + 1 + usize::from(utt.at(i + 1) == "tambien");
            governed(utt, next, false, false, false).map(|_| (i, next))
        })
        .peekable();
    let Some((first, _)) = cuts.peek().copied() else {
        return Vec::new();
    };
    let mut segments = vec![utt.slice(0, first)];
    let head = utt.slice(0, verb.end);
    while let Some((_, start)) = cuts.next() {
        let end = cuts.peek().map_or(utt.len(), |&(end, _)| end);
        segments.push(head.clone().join(&utt.slice(start, end)));
    }
    segments
}

fn room_at(utt: &Utterance<'_>, index: usize) -> bool {
    grammar::ROOM_PREPS.iter().any(|prep| {
        let mut i = index;
        prep.split_whitespace().all(|word| {
            let ok = utt.at(i) == word;
            i += 1;
            ok
        }) && grammar::canonical_room(utt.at(i)).is_some()
    })
}

fn split_targets<'a>(utt: &Utterance<'a>, verb: Verb<'_>) -> Vec<Utterance<'a>> {
    let domotics = verb.has(DOMOTICS);
    if !domotics && !verb.has(CANCEL | CLEAR | STOP | STOP_RINGING | PAUSE | RESUME | DISABLE) {
        return Vec::new();
    }
    let own_object = governed(utt, verb.end, false, false, false).is_some();
    let lists = (0..utt.len()).any(|i| object(utt, i, false) == Some(List));
    let mut segments = Vec::new();
    let mut carried = utt.slice(0, 0);
    let mut start = 0;
    let mut content = false;
    for i in verb.end..utt.len() {
        content |= CONTENT_MARKERS.contains(&utt.at(i - 1));
        if utt.is_quoted(i) || utt.at(i) != "y" || i <= start {
            continue;
        }
        if content {
            return Vec::new();
        }
        let next = i + 1 + usize::from(utt.at(i + 1) == "tambien");
        let target = next + usize::from(DETERMINERS.contains(&utt.at(next)));
        let repeat_room = domotics && !lists && room_at(utt, next);
        let repeat_device =
            domotics && !lists && target > next && grammar::entity_at(utt.at(target)).is_some();
        let repeat_object = !own_object && governed(utt, next, false, false, false).is_some();
        if !(repeat_room || repeat_device || repeat_object) {
            continue;
        }
        let previous = carried.join(&utt.slice(start, i));
        carried = if repeat_room {
            let end = (verb.end..previous.len())
                .find(|&i| room_at(&previous, i))
                .unwrap_or(previous.len());
            previous.slice(0, end)
        } else {
            utt.slice(0, verb.end)
        };
        segments.push(previous);
        start = next;
    }
    if !segments.is_empty() {
        segments.push(carried.join(&utt.slice(start, utt.len())));
    }
    segments
}

fn conflicts(parts: &[HouseholdPart]) -> bool {
    parts
        .iter()
        .filter_map(|part| part.command.as_ref())
        .enumerate()
        .any(|(i, left)| {
            parts
                .iter()
                .filter_map(|part| part.command.as_ref())
                .skip(i + 1)
                .any(|right| {
                    left.domain == right.domain
                        && left.room == right.room
                        && matches!(
                            (left.operation, right.operation),
                            (DeviceOperation::On, DeviceOperation::Off)
                                | (DeviceOperation::Off, DeviceOperation::On)
                                | (DeviceOperation::OpenCover, DeviceOperation::CloseCover)
                                | (DeviceOperation::CloseCover, DeviceOperation::OpenCover)
                                | (DeviceOperation::MediaPlay, DeviceOperation::MediaPause)
                                | (DeviceOperation::MediaPause, DeviceOperation::MediaPlay)
                        )
                })
        })
}

fn compound(
    utt: &Utterance<'_>,
    settings: &HouseholdIntentSettings,
    source_room: Option<&str>,
) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false)?;
    let mut segments = split_clauses(utt);
    if segments.is_empty() {
        segments = split_objects(utt, verb);
    }
    if segments.is_empty() {
        segments = split_targets(utt, verb);
    }
    if segments.len() < 2 {
        return None;
    }
    let clarify = |reason| {
        HouseholdIntent::new(Compound, Clarify, "R5", utt.heard(), verb.form).because(reason)
    };
    let intents: Vec<_> = segments.iter().map(single).collect();
    if intents
        .iter()
        .flatten()
        .any(HouseholdIntent::controls_routine)
    {
        return Some(clarify("routine_control_part"));
    }
    if segments.len() > settings.compound_max_parts {
        return Some(clarify("too_many_parts"));
    }
    if !settings.compound_execute {
        return Some(clarify("compound_disabled"));
    }
    let mut parts = Vec::with_capacity(segments.len());
    for (segment, intent) in segments.iter().zip(intents) {
        let text = segment.grammar_text();
        let security_refusal = intent.is_none() && grammar::security_order(&text);
        if intent.as_ref().is_some_and(|i| {
            i.family == DeferredAction || (i.family == Reminder && i.operation == Create)
        }) || (intent.is_none()
            && ((!security_refusal
                && grammar::non_immediate(&text) == Some(GrammarBlock::Deferred))
                || (security_refusal && grammar::has_deferral_time(&fold(&text)))))
        {
            return Some(clarify("deferred_part"));
        }
        let command = if intent.is_none() && !security_refusal {
            match grammar::parse_core(&text, source_room) {
                GrammarMatch::Command(command) => Some(command),
                _ => return Some(clarify("unparsed_part")),
            }
        } else {
            None
        };
        if intent.as_ref().is_some_and(|i| i.operation == Clarify) {
            return Some(clarify("unparsed_part"));
        }
        parts.push(HouseholdPart {
            segment: segment.heard(),
            intent,
            command,
            security_refusal,
        });
    }
    if conflicts(&parts) {
        return Some(clarify("conflicting_parts"));
    }
    Some(HouseholdIntent {
        parts,
        ..HouseholdIntent::new(Compound, Execute, "R5", utt.heard(), verb.form)
    })
}

fn integration(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    if utt.has(&["camara", "camaras"]) && !VIDEO_DOORBELL.is_match(&text) {
        let health = CAMERA_HEALTH_QUERY.is_match(&text);
        let media = CAMERA_MEDIA_QUERY.is_match(&text);
        if health || media {
            let kind = if media
                || utt.has(&[
                    "video",
                    "foto",
                    "imagen",
                    "grabando",
                    "grabacion",
                    "movimiento",
                    "stream",
                ]) {
                "media"
            } else {
                "health"
            };
            return Some(HouseholdIntent::new(
                CameraHealth,
                Query,
                "P5",
                kind.into(),
                "",
            ));
        }
    }
    if utt.has(&["calendario", "agenda"]) && CALENDAR_QUERY.is_match(&text) {
        let day = if has_phrase(&text, "manana") {
            "tomorrow"
        } else {
            "today"
        };
        return Some(HouseholdIntent::new(Calendar, Query, "P5", day.into(), ""));
    }
    if let Some(kind) = energy::measurement(utt, &text) {
        return Some(HouseholdIntent::new(Energy, Query, "P5", kind.into(), ""));
    }
    let tv = &*TV_QUERY_OR_CONTROL;
    let m = tv.captures(&text)?;
    let room = m.name("room").map_or("", |m| m.as_str());
    if m.name("query").is_some() {
        return Some(HouseholdIntent::new(Media, Query, "P5", room.into(), ""));
    }
    if grammar::has_deferral_time(&text) {
        return None;
    }
    let verb = if m.name("pause").is_some() {
        "pause"
    } else if m.name("up").is_some() {
        "volume_up"
    } else {
        "volume_down"
    };
    Some(HouseholdIntent::new(
        Media,
        Execute,
        "P5",
        room.into(),
        verb,
    ))
}

mod energy;

regex_patterns! {
    TOKEN => r"[0-9]+(?:[:.][0-9]{2}|[.,][0-9]+)?|[\p{L}\p{M}]+";
    NEXT_WORD => r"[0-9]|[\p{L}\p{M}]";
    REMINDER_HEAD => r"^(?:no (?:me|nos) dejes olvidar|hace(?:me|nos) acordar|(?:me|nos) (?:podes|podrias|puedes) (?:recordar|avisar)|(?:me|nos) (?:recordas|avisas)|recorda(?:me|nos|melo)|recuerda(?:me|nos|melo)|acorda(?:me|nos)|acuerda(?:me|nos)|avisa(?:me|nos)|acordate de|acuerdate de|recordar(?:me|nos)|avisar(?:me|nos)|(?P<other>recorda(?:le|les)|recuerda(?:le|les)|avisa(?:le|les)|recordar(?:le|les)|avisar(?:le|les)|(?:le|les) (?:podes|podrias|puedes) (?:recordar|avisar)))\b";
    CLOCK_ALARM_HOUR => r"\b(?:de|a|para) las? (?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veintiun[ao]|veintidos|veintitres)\b";
    CLOCK_ALARM_DAY => r"\b(?:del|al|de la|a la|para el|para la) (?:mediodia|medianoche)\b|\b(?:de|para) (?:hoy|pasado manana|manana)\b|\b(?:del|de este|para el) (?:proximo )?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b";
    DURATION => r"\b(?:[0-9]+(?:[.,][0-9]+)?|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci(?:seis|siete|ocho|nueve)|veinte|veinti(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|media|medio|cuartos?)(?: y \w+)? (?:de )?(?:segundos?|minutos?|mins?|horas?)\b|\b(?:hora y (?:media|cuarto)|minuto y medio)\b";
    ARTICLE_CLOCK_UNIT => r"\b(?:a|para|de) las? (?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veintiun[ao]|veintidos|veintitres) (?:h|hs|hrs|horas?|y \w+ (?:minutos?|mins?))\b";
    CLOCK => r"\b(?:a|para|cuando sean?) (?:las? )?(?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|mediodia|medianoche)\b|\bal mediodia\b";
    INTERCOM_MARKER => r"(?i)\bque\b|:";
    LEADING_INTERCOM_MARKER => r"(?i)^(?:que\b|:)";
    ROUTINE_REQUEST => r"^(?:(?:dale|bueno|bue|ahora|entonces|yo|y|e|eh|ehh|em|emm|mm|mmm|este) )*(?:(?P<request>te pido que|me ayudas a|me ayudarias a|quiero|quisiera|queria|querria|necesito|necesitaria|preciso|me gustaria|nos gustaria|me encantaria|vamos a|voy a|podemos|podriamos|hay que|habria que|se puede|me podes|me podrias|me puedes|nos podes|nos podrias|nos puedes|ayudame a|a ver) )?(?:(?P<que>que) )?(?:(?P<clitic>me|nos) )?";
    NOTIFICATION => r"^(?:(?:(?P<disable>desactiva|desactive|quita|quite|saca|saque)|activa|active|pone|pon) )?(?:el )?(?:modo )?no molestar$";
    VIDEO_DOORBELL => r"\bvideo ?porteros?\b";
    CAMERA_HEALTH_QUERY => r"^(?:como (?:esta|estan|andan|funcionan)|estado (?:tecnico )?de|salud (?:tecnica )?de) ";
    CAMERA_MEDIA_QUERY => r"^(?:mostra(?:me)?|muestra(?:me)?|ver|quiero ver|dame|mandame|(?:esta|estan) grabando) ";
    CALENDAR_QUERY => r"^(?:que (?:hay|tenemos|tengo)(?: (?:hoy|manana))?(?: en (?:el|la|mi|nuestro) (?:calendario|agenda)(?: (?:del hogar|compartid[oa]|personal))?)?|(?:mostrame|decime|lee) (?:el|la|mi) (?:calendario|agenda)|(?:calendario|agenda)(?: (?:del hogar|compartid[oa]))?)(?: (?:hoy|manana))?$";
    ROUTINE_READ => concat!(
        r"^(?:(?:decime|dime|contame) )?(?:(?:como esta|estado de|",
        r"(?:a que hora|cuando) (?:corre|se ejecuta|arranca)|",
        r"(?:esta|sigue|quedo) (?:activa(?:da)?|desactivada|programada|pausada)) ",
        r"(?:la |mi |nuestra )?rutina (?P<before>.+)|",
        r"(?:la |mi |nuestra )?rutina (?P<after>.+?) (?:(?:esta|sigue|quedo) ",
        r"(?:activa(?:da)?|desactivada|programada|pausada)|(?:se ejecuto|funciono|fallo|corrio)(?: hoy| ayer)?))$"
    );
    TV_QUERY_OR_CONTROL => r"^(?:(?P<query>que (?:esta pasando|hay|estan dando) en|como esta|estado de)|(?P<pause>(?:pone|pon|poneme) (?:en )?pausa(?: en| a)?|pausa|pausame)|(?P<up>(?:subi|sube|subime) (?:el )?volumen(?: de| en)?)|(?P<down>(?:baja|bajame) (?:el )?volumen(?: de| en)?)) (?:la |el )?(?:televisor|television|tele|tv)(?:(?: del| de la| de| en el| en la| en) (?P<room>[a-z ]+))?$";
}
