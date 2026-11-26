//! Scope of nominal energy measurements and expenses.

use super::Utterance;

fn expense_intro() -> &'static regex::Regex {
    re!(concat!(
        r"^(?:",
        r"(?:",
        determiner!(),
        r"|del|de|en|por|para|con|sobre|al)",
        r")$",
    ))
}

fn expense_question() -> &'static regex::Regex {
    re!(concat!(r"^(?:", r"cuant[oa]?|que", r")$",))
}

fn finite_utterance() -> &'static regex::Regex {
    re!(concat!(r"\b", finite_verb!(), r"\b",))
}

fn subordinate() -> &'static regex::Regex {
    re!(r"\bque\b")
}

pub(super) fn indirect_expense_question() -> &'static fancy_regex::Regex {
    scoped_re!(r"^(?:decime|mostrame|dime|contame|saber) que (?=gastos?\b)")
}

fn expense_word() -> &'static regex::Regex {
    re!(r"\bgastos?\b")
}

fn participle_word() -> &'static regex::Regex {
    re!(concat!(r"^(?:", participle!(), r")$",))
}

fn nominal_event() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r"\b(?:(?P<head>",
        nominal_head!(),
        r")|(?P<connector>",
        connector!(),
        r")|(?P<existential>",
        existential!(),
        r")|(?P<clause>",
        clause!(),
        r"))\b",
    ))
}

fn of_tail() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(r" ", of_scope!(), r"(?: |$)",))
}

fn excess_tail() -> &'static regex::Regex {
    re!(r" mas(?: |$)")
}

fn in_tail() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(r" ", in_scope!(), r"(?: |$)",))
}

fn in_determiner() -> &'static regex::Regex {
    re!(concat!(
        r" (?:tod[oa]s? )?(?:",
        determiner!(),
        r"|cada|cualquier(?:a)?)\b",
    ))
}

fn in_physical_target() -> &'static regex::Regex {
    re!(
        r" (?:termotanques?|calderas?|motor(?:es)?|banos?|garajes?|oficinas?|taller(?:es)?|piscinas?|piletas?|edificios?|empresas?|negocios?|local(?:es)?|comercios?|fabricas?|hospital(?:es)?|escuelas?|barrios?|ciudad(?:es)?|pais(?:es)?|montevideo|uruguay|espana|lo de)\b"
    )
}

fn expense_tail() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r" (?:",
        in_scope!(),
        r"|",
        tariff!(),
        r"|",
        fiscal!(),
        r"|",
        dated_qualifier!(),
        r")(?: |$)",
    ))
}

pub(super) fn nominal_expenses(text: &str, utterance: &Utterance<'_>) -> Option<Vec<usize>> {
    let question = indirect_expense_question().find(text).ok()?;
    let start = question.map_or(1, |m| m.end());
    let main_end = subordinate()
        .find_at(text, start)
        .map_or(text.len(), |m| m.start());
    let finite = finite_utterance().is_match(&text[..main_end]);
    let mut positions = Vec::new();
    for m in expense_word().find_iter(text) {
        let index = text[..m.start()].bytes().filter(|&b| b == b' ').count();
        if utterance.spoken_at(index) == "gastó" {
            continue;
        }
        let previous = index.checked_sub(1).map_or("", |i| utterance.at(i));
        if m.as_str() == "gastos"
            || index == 0
            || expense_intro().is_match(previous)
            || (expense_question().is_match(previous) && finite)
        {
            positions.push(m.start());
        }
    }
    Some(positions)
}

pub(super) fn nominals_in_scope(text: &str, expenses: &[usize]) -> Option<bool> {
    let mut active = false;
    let mut expense = false;
    let mut position = 0;
    while let Some(event) = nominal_event().captures_from_pos(text, position).ok()? {
        let m = event.get(0)?;
        position = m.end();
        if event.name("head").is_some() {
            if matches!(m.as_str(), "gasto" | "gastos") && !expenses.contains(&m.start()) {
                active = false;
                expense = false;
            } else {
                active = true;
                expense |= expenses.contains(&m.start());
            }
        } else if event.name("existential").is_some() {
            continue;
        } else if event.name("clause").is_some() && !participle_word().is_match(m.as_str()) {
            active = false;
            expense = false;
        } else if active && let Some(connector) = event.name("connector") {
            let tail = &text[position..];
            let matcher = match connector.as_str() {
                "de" | "del" => {
                    if expense
                        && connector.as_str() == "de"
                        && let Some(adverb) = excess_tail().find(tail).filter(|m| m.start() == 0)
                    {
                        position += adverb.end();
                        continue;
                    }
                    of_tail()
                }
                "en" => in_tail(),
                "si" if expense => return Some(false),
                "con" | "sin" | "por" | "para" | "a" | "al" if expense => expense_tail(),
                _ => continue,
            };
            if let Some(complement) = matcher.find(tail).ok()?.filter(|m| m.start() == 0) {
                let end = position + complement.end();
                expense |= expenses
                    .iter()
                    .any(|&start| (position..end).contains(&start));
                position = end;
            } else if connector.as_str() != "en"
                || tail.len() <= 1
                || in_determiner().find(tail).is_some_and(|m| m.start() == 0)
                || in_physical_target()
                    .find(tail)
                    .is_some_and(|m| m.start() == 0)
            {
                return Some(false);
            }
        }
    }
    Some(true)
}

pub(super) fn prepare_patterns() {
    expense_intro();
    expense_question();
    finite_utterance();
    subordinate();
    indirect_expense_question();
    expense_word();
    participle_word();
    nominal_event();
    of_tail();
    excess_tail();
    in_tail();
    in_determiner();
    in_physical_target();
    expense_tail();
}
