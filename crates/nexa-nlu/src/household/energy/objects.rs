//! Scope of objects governed by energy measurement verbs.

fn object_context() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r"(?:",
        adverb!(),
        r"|",
        period!(),
        r"|en ",
        in_scope!(),
        r"|(?:desde|durante|por|a|al|hasta|para|hace) (?:",
        time_endpoint!(),
        r"|(?:",
        determiner!(),
        r" )?",
        document!(),
        r")|entre (?:semana|",
        time_endpoint!(),
        r" y ",
        time_endpoint!(),
        r")|fuera de ",
        tariff_band!(),
        r"|con ",
        tariff!(),
        r"|(?:con|sin) ",
        fiscal!(),
        r"|(?:de|del) ",
        of_scope!(),
        r")(?: |$)",
    ))
}

pub(super) fn object_link() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r"\b(?:(?P<accumulation>",
        accumulation_verb!(),
        r")|",
        verb!(),
        r")(?: |$)",
    ))
}

fn object_participle() -> &'static regex::Regex {
    re!(concat!(participle!(), r"(?: |$)",))
}

fn object_tail() -> &'static regex::Regex {
    re!(concat!(
        r"$|",
        np_determiner!(),
        r"(?:",
        period_modifier!(),
        r" )*",
        measurement_head!(),
        r"\b",
    ))
}

pub(super) fn object_in_scope(mut tail: &str, accumulation: bool) -> Option<bool> {
    loop {
        let modifier = object_context()
            .find(tail)
            .ok()?
            .filter(|m| m.start() == 0)
            .map(|m| m.end())
            .or_else(|| {
                accumulation
                    .then(|| {
                        object_participle()
                            .find(tail)
                            .filter(|m| m.start() == 0)
                            .map(|m| m.end())
                    })
                    .flatten()
            });
        let Some(end) = modifier else {
            return Some(object_tail().find(tail).is_some_and(|m| m.start() == 0));
        };
        tail = &tail[end..];
    }
}

pub(super) fn prepare_patterns() {
    object_context();
    object_link();
    object_participle();
    object_tail();
}
