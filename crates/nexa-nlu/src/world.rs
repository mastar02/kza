//! Household time and weather questions and pure spoken answers.

use chrono::{DateTime, Datelike, Days, NaiveDate, Offset, TimeZone, Timelike, Weekday};

use crate::grammar::{action, canonical_room, entity_at};
use crate::normalize::{WAKE_WORDS, has_phrase, prefix, words};

pub const CLOCK_UNAVAILABLE: &str = "No pude leer el reloj de la casa.";
pub const NO_WEATHER_DATA: &str = "No tengo el dato del clima ahora mismo.";
pub const NO_FORECAST: &str = "No tengo el pronóstico ahora mismo.";
pub const NO_FORECAST_HORIZON: &str = "No tengo ese pronóstico. Puedo decirte el de hoy, mañana, pasado mañana o un día de la semana.";
pub const ONE_DAY_AT_A_TIME: &str = "Preguntame por un día a la vez.";
pub const INDOOR_TEMPERATURE_UNSUPPORTED: &str =
    "Todavía no puedo decirte la temperatura de adentro de la casa, solo la de afuera.";

const WEEKDAYS: [&str; 7] = [
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
];
const MONTHS: [&str; 12] = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeDay {
    Yesterday,
    Today,
    Tomorrow,
    DayAfterTomorrow,
}

impl RelativeDay {
    fn date(self, today: NaiveDate) -> Option<NaiveDate> {
        match self {
            Self::Yesterday => today.pred_opt(),
            Self::Today => Some(today),
            Self::Tomorrow => today.succ_opt(),
            Self::DayAfterTomorrow => today.checked_add_days(Days::new(2)),
        }
    }

    fn spoken(self) -> &'static str {
        match self {
            Self::Yesterday => "ayer",
            Self::Today => "hoy",
            Self::Tomorrow => "mañana",
            Self::DayAfterTomorrow => "pasado mañana",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockQuery {
    Time,
    Date { day: RelativeDay, with_year: bool },
}

fn clock_core<'a>(mut text: &'a str, wakes: &[String]) -> &'a str {
    let leading = [
        "che", "hola", "oye", "ey", "bueno", "a ver", "disculpa", "perdon",
    ];
    let trailing = ["che", "por favor", "porfa", "porfavor"];
    loop {
        let previous = text;
        for wake in wakes
            .iter()
            .map(String::as_str)
            .chain(WAKE_WORDS.iter().copied())
            .chain(leading)
        {
            if !wake.is_empty() && prefix(text, wake) {
                text = text[wake.len()..].trim_start();
            }
        }
        for wake in wakes
            .iter()
            .map(String::as_str)
            .chain(WAKE_WORDS.iter().copied())
            .chain(trailing)
        {
            if !wake.is_empty()
                && let Some(head) = text.strip_suffix(wake)
                && (head.is_empty() || head.ends_with(' '))
            {
                text = head.trim_end();
            }
        }
        if text == previous {
            return text;
        }
    }
}

macro_rules! clock_lead {
    () => { r"(?:decime|dime|deci|me decis|me dices|me dirias|me podes decir|me podrias decir|me puedes decir|podes decirme|podrias decirme|puedes decirme|sabes|sabe|sabes decirme|queria saber|quiero saber)" };
}

/// Match a whole question about the local clock or calendar date.
pub fn parse_clock_query(text: &str, wakes: &[String]) -> Option<ClockQuery> {
    let plain = words(text);
    let core = clock_core(&plain, wakes);
    let time = &*CLOCK_TIME;
    if time.is_match(core) {
        return Some(ClockQuery::Time);
    }
    let date = &*CLOCK_DATE;
    let m = date.captures(core)?;
    let first = m.name("first").map(|m| m.as_str());
    let last = m.name("last").map(|m| m.as_str());
    if first.is_some() && last.is_some() && first != last {
        return None;
    }
    let day = match first
        .or(last)
        .or_else(|| m.name("after").map(|m| m.as_str()))
    {
        Some("ayer") => RelativeDay::Yesterday,
        Some("manana") => RelativeDay::Tomorrow,
        Some("pasado manana") => RelativeDay::DayAfterTomorrow,
        _ => RelativeDay::Today,
    };
    Some(ClockQuery::Date {
        day,
        with_year: has_phrase(core, "fecha"),
    })
}

/// Format the supplied local time. Time-zone and clock acquisition belong to the caller.
pub fn describe_clock<Tz: TimeZone>(query: ClockQuery, now_local: &DateTime<Tz>) -> String {
    // Conversion with checked arithmetic also accepts chrono's representable UTC extremes.
    let Some(now) = now_local
        .naive_utc()
        .checked_add_offset(now_local.offset().fix())
    else {
        return CLOCK_UNAVAILABLE.into();
    };
    match query {
        ClockQuery::Time => {
            let lead = if now.hour() == 1 { "Es la" } else { "Son las" };
            format!("{lead} {}:{:02}.", now.hour(), now.minute())
        }
        ClockQuery::Date { day, with_year } => {
            let Some(date) = day.date(now.date()) else {
                return CLOCK_UNAVAILABLE.into();
            };
            let lead = match day {
                RelativeDay::Yesterday => "Ayer fue",
                RelativeDay::Today => "Hoy es",
                RelativeDay::Tomorrow => "Mañana es",
                RelativeDay::DayAfterTomorrow => "Pasado mañana es",
            };
            let weekday = WEEKDAYS[date.weekday().num_days_from_monday() as usize];
            let month = MONTHS[date.month0() as usize];
            let year = if with_year {
                format!(" de {}", date.year())
            } else {
                String::new()
            };
            format!("{lead} {weekday} {} de {month}{year}.", date.day())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastDay {
    Relative(RelativeDay),
    Weekday { day: Weekday, next_week: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherHorizon {
    Current,
    Forecast(ForecastDay),
    Unsupported,
    SeveralDays,
    Indoor,
}

impl ForecastDay {
    /// Resolve by the household's local calendar date, including UTC rollover.
    pub fn date(self, today: NaiveDate) -> Option<NaiveDate> {
        match self {
            Self::Relative(day) => day.date(today),
            Self::Weekday { day, next_week } => {
                let days = day.days_since(today.weekday());
                today.checked_add_days(Days::new(if days == 0 && next_week {
                    7
                } else {
                    u64::from(days)
                }))
            }
        }
    }

    pub fn spoken(self, today: NaiveDate) -> String {
        match self {
            Self::Relative(day) => day.spoken().into(),
            Self::Weekday { day, next_week } => {
                let weekday = WEEKDAYS[day.num_days_from_monday() as usize];
                if day != today.weekday() {
                    format!("el {weekday}")
                } else if next_week {
                    format!("el {weekday} que viene")
                } else {
                    "hoy".into()
                }
            }
        }
    }
}

macro_rules! weekdays {
    () => {
        r"(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)"
    };
}
macro_rules! months {
    () => { r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)" };
}

pub fn parse_weather_horizon(text: &str) -> WeatherHorizon {
    let plain = words(text);
    if UNSUPPORTED_HORIZON.is_match(&plain) {
        return WeatherHorizon::Unsupported;
    }
    let words: Vec<_> = plain.split_whitespace().collect();
    let mut candidate = None;
    let mut multiple = false;
    for (i, &word) in words.iter().enumerate() {
        let previous = i.checked_sub(1).map_or("", |i| words[i]);
        let following = words.get(i + 1).copied().unwrap_or_default();
        let day = match (word, following) {
            ("pasado", "manana") => ForecastDay::Relative(RelativeDay::DayAfterTomorrow),
            ("manana", _) if !matches!(previous, "esta" | "la" | "pasado") => {
                ForecastDay::Relative(RelativeDay::Tomorrow)
            }
            ("hoy", _) | ("esta", "manana" | "tarde" | "noche") => {
                ForecastDay::Relative(RelativeDay::Today)
            }
            _ => {
                let day = match word {
                    "lunes" => Weekday::Mon,
                    "martes" => Weekday::Tue,
                    "miercoles" => Weekday::Wed,
                    "jueves" => Weekday::Thu,
                    "viernes" => Weekday::Fri,
                    "sabado" => Weekday::Sat,
                    "domingo" => Weekday::Sun,
                    _ => continue,
                };
                if following == "pasado" && words.get(i + 2) != Some(&"manana") {
                    return WeatherHorizon::Unsupported;
                }
                let next_week = previous == "proximo"
                    || following == "proximo"
                    || words.get(i + 1..i + 3) == Some(&["que", "viene"]);
                ForecastDay::Weekday { day, next_week }
            }
        };
        multiple |= candidate.is_some_and(|other| day != other);
        candidate.get_or_insert(day);
    }
    if multiple {
        return WeatherHorizon::SeveralDays;
    }
    let day = candidate.unwrap_or(ForecastDay::Relative(RelativeDay::Today));
    let past = WEATHER_PAST.is_match(&plain);
    let future = WEATHER_FUTURE.is_match(&plain);
    if matches!(
        day,
        ForecastDay::Weekday {
            next_week: false,
            ..
        }
    ) && past
    {
        return WeatherHorizon::Unsupported;
    }
    if day == ForecastDay::Relative(RelativeDay::Today) && !future {
        return WeatherHorizon::Current;
    }
    WeatherHorizon::Forecast(day)
}

fn indoor_location(text: &str, extra: &[String]) -> bool {
    let locations = &*INDOOR_PLACE;
    if locations.captures_iter(text).any(|m| {
        !text[..m.get(0).map_or(0, |m| m.start())].ends_with("fuera ")
            && (canonical_room(&m[1]).is_some()
                || matches!(&m[1], "pieza" | "comedor" | "lavadero")
                || extra.iter().any(|room| room == &m[1]))
    }) {
        return true;
    }
    HOUSE.is_match(text) && !OUTSIDE.is_match(text)
}

/// Identify weather questions, distinguishing room temperature from outdoor data.
pub fn parse_weather_query(text: &str, room_words: &[String]) -> Option<WeatherHorizon> {
    let plain = words(text);
    let temperature = TEMPERATURE_QUERY.is_match(&plain);
    let explicit = EXPLICIT_WEATHER_QUERY.is_match(&plain);
    let fragment = WEATHER_FRAGMENT.is_match(&plain);
    let question = text.contains(['¿', '?']) || QUESTION_HEAD.is_match(&plain);
    let imperative = plain.split_whitespace().next().and_then(action).is_some();
    let has_device = plain
        .split_whitespace()
        .any(|word| entity_at(word).is_some());
    if imperative && has_device && !question {
        return None;
    }
    if !temperature && !explicit && !(fragment && (!imperative || question)) {
        return None;
    }
    if (has_phrase(&plain, "el tiempo del temporizador")
        || has_phrase(&plain, "el clima prendido")
        || has_phrase(&plain, "el clima apagado"))
        && !temperature
    {
        return None;
    }
    if temperature && indoor_location(&plain, room_words) {
        return Some(WeatherHorizon::Indoor);
    }
    Some(parse_weather_horizon(&plain))
}

fn condition(value: &str) -> Option<&'static str> {
    Some(match value.to_ascii_lowercase().as_str() {
        "clear-night" => "despejado",
        "cloudy" => "nublado",
        "exceptional" => "condiciones excepcionales",
        "fog" => "con niebla",
        "hail" => "con granizo",
        "lightning" => "con tormenta eléctrica",
        "lightning-rainy" => "con tormenta y lluvia",
        "partlycloudy" => "parcialmente nublado",
        "pouring" => "lloviendo fuerte",
        "rainy" => "lluvioso",
        "snowy" => "nevando",
        "snowy-rainy" => "con aguanieve",
        "sunny" => "soleado",
        "windy" | "windy-variant" => "ventoso",
        _ => return None,
    })
}

fn rounded(value: Option<f64>) -> Option<String> {
    let value = value?.round_ties_even();
    if !value.is_finite() {
        return None;
    }
    Some(if value == 0.0 {
        "0".into()
    } else {
        format!("{value:.0}")
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CurrentWeather<'a> {
    pub condition: Option<&'a str>,
    pub temperature: Option<f64>,
    pub humidity: Option<f64>,
}

pub fn describe_current(state: Option<CurrentWeather<'_>>) -> String {
    let Some(state) = state else {
        return NO_WEATHER_DATA.into();
    };
    if state.condition.is_some_and(|word| {
        matches!(
            word.to_ascii_lowercase().as_str(),
            "unavailable" | "unknown" | "none"
        )
    }) {
        return NO_WEATHER_DATA.into();
    }
    let mut parts = Vec::new();
    if let Some(value) = rounded(state.temperature) {
        parts.push(format!("{value} grados"));
    }
    if let Some(value) = state.condition.and_then(condition) {
        parts.push(value.into());
    }
    if let Some(value) = rounded(state.humidity) {
        parts.push(format!("{value} por ciento de humedad"));
    }
    if parts.is_empty() {
        NO_WEATHER_DATA.into()
    } else {
        format!("Hay {}.", parts.join(", "))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ForecastEntry<'a> {
    pub datetime: Option<&'a str>,
    pub condition: Option<&'a str>,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastAlignment {
    Aligned,
    NoEntries,
    InvalidEntries,
    MissingDay,
    ConflictingDay,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForecastAnswer {
    pub text: String,
    pub alignment: ForecastAlignment,
}

pub fn weather_refusal(horizon: WeatherHorizon) -> Option<&'static str> {
    match horizon {
        WeatherHorizon::Indoor => Some(INDOOR_TEMPERATURE_UNSUPPORTED),
        WeatherHorizon::SeveralDays => Some(ONE_DAY_AT_A_TIME),
        WeatherHorizon::Unsupported => Some(NO_FORECAST_HORIZON),
        _ => None,
    }
}

fn forecast_parts(entry: &ForecastEntry<'_>) -> Vec<String> {
    let mut parts = Vec::new();
    if let Some(s) = entry.condition.and_then(condition) {
        parts.push(s.into());
    }
    if let Some(high) = rounded(entry.high) {
        if let Some(low) = rounded(entry.low) {
            parts.push(format!("entre {low} y {high} grados"));
        } else {
            parts.push(format!("{high} grados"));
        }
    }
    parts
}

/// Select entries by their local date after applying the caller's time zone.
/// Missing/invalid timestamps take precedence over conflicts and missing days.
pub fn describe_forecast<Tz: TimeZone>(
    entries: &[ForecastEntry<'_>],
    spoken_day: &str,
    target: Option<NaiveDate>,
    zone: &Tz,
) -> ForecastAnswer {
    use ForecastAlignment::*;
    let generic = |alignment| ForecastAnswer {
        text: NO_FORECAST.into(),
        alignment,
    };
    if entries.is_empty() {
        return generic(NoEntries);
    }
    let no_day = |alignment| {
        let day = spoken_day.trim();
        let label = if day.is_empty() { "ese día" } else { day };
        ForecastAnswer {
            text: format!("No tengo el pronóstico para {label}."),
            alignment,
        }
    };
    let mut first = None;
    let mut conflict = false;
    for entry in entries {
        let Some(time) = entry
            .datetime
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        else {
            return generic(InvalidEntries);
        };
        let local = time.with_timezone(zone);
        let Some(local) = local.naive_utc().checked_add_offset(local.offset().fix()) else {
            return generic(InvalidEntries);
        };
        if Some(local.date()) == target {
            let parts = forecast_parts(entry);
            conflict |= first.as_ref().is_some_and(|previous| previous != &parts);
            first.get_or_insert(parts);
        }
    }
    if conflict {
        return no_day(ConflictingDay);
    }
    let Some(parts) = first else {
        return no_day(MissingDay);
    };
    if parts.is_empty() {
        return generic(Aligned);
    }
    let mut chars = spoken_day.chars();
    let day = chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    });
    ForecastAnswer {
        text: format!("{day}: {}.", parts.join(", ")),
        alignment: Aligned,
    }
}

regex_patterns! {
    CLOCK_TIME => concat!(
        r"^(?:(?:",
        clock_lead!(),
        r" )?(?:ahora )?que horas? (?:es|son|tenes|tenemos)|",
        clock_lead!(),
        r" la hora(?: que es)?)(?: (?:ahora(?: mismo)?|en este momento|hoy))?$"
    );
    CLOCK_DATE => concat!(
        r"^(?:(?:",
        clock_lead!(),
        r" )?(?:(?P<first>hoy|pasado manana|manana|ayer) )?que (?:dia(?: de la semana)?|fecha) ",
        r"(?:es|seria|sera|fue|era|tenemos|estamos)(?: (?P<last>hoy|pasado manana|manana|ayer))?|(?:",
        clock_lead!(),
        r" )?(?:(?:a|en) que (?:dia(?: de la semana)?|fecha) estamos|a cuanto estamos)(?: hoy)?|",
        clock_lead!(),
        r" (?:la fecha|el dia)(?: de (?P<after>hoy|pasado manana|manana|ayer))?)$"
    );
    UNSUPPORTED_HORIZON => concat!(r"\b(?:la semana|esta semana|proxima semana|semana pasada|fin(?:es)? de semana|finde|",
        r"mes que viene|proximo mes|este mes|mes pasado|ano que viene|proximo ano|ano pasado|",
        r"proximos dias|estos dias|dias que vienen|ayer|anoche|anteayer|",
        r"pasad[oa] (?:", weekdays!(), r"|semana|mes|ano)|",
        r"(?:en|dentro de) (?:[0-9]+|un|una|dos|tres|cuatro|cinco|seis|siete|diez|quince) (?:dias|semanas?)|",
        r"el [0-9]+|[0-9]+ de ", months!(), r"|", weekdays!(), r" [0-9]+|[0-9]+(?:ro|ero|er)|",
        r"dia [0-9]+|(?:en|de) ", months!(), r"|primero de ", months!(), r")\b");
    WEATHER_PAST => r"\b(?:estuvo|estuvieron|fue|fueron|hizo|hicieron|llovio|hubo|nevo)\b";
    WEATHER_FUTURE => r"\b(?:(?:va|van) a|pronostico|llovera|hara|habra|estara)\b";
    INDOOR_PLACE => r"\b(?:en|del|de|tiene)(?: (?:el|la|los|las))? (\w+)";
    HOUSE => r"\b(?:adentro|dentro|en casa|en la casa|de la casa)\b";
    OUTSIDE => r"\b(?:afuera|fuera|en la calle)\b";
    TEMPERATURE_QUERY => r"\b(?:(?:que|cual es la|cuanta) temperatura|cuantos grados|(?:temperatura|grados) (?:hay|hace|tiene|marca|hara|habra|(?:va|van) a (?:hacer|haber))|esta la temperatura|va a ser la temperatura|temperatura (?:de|para) manana)\b";
    EXPLICIT_WEATHER_QUERY => concat!(r"\b(?:que tiempo (?:hace|va a hacer)|(?:como )?esta el clima|",
        r"como (?:esta|va a estar) el tiempo|como va a estar el clima|temperatura afuera|grados hay afuera|pronostico|",
        r"el tiempo (?:de|del) (?:hoy|manana|pasado manana|", weekdays!(), r"|fin de semana))\b");
    WEATHER_FRAGMENT => r"\b(?:hace (?:calor|frio)|va a hacer (?:calor|frio)|llueve|va a llover|lloviendo|llovera)\b";
    QUESTION_HEAD => r"^(?:que|como|cual|cuanto|cuanta|decime)\b";
}
