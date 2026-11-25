//! Deterministic device commands and slots.

use crate::normalize::{fold, has_phrase, prefix, strip_head, words};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Light,
    Climate,
    Cover,
    Scene,
    Fan,
    MediaPlayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvacMode {
    Off,
    Cool,
    Heat,
    Dry,
    FanOnly,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    On,
    Off,
    SetLight,
    SetTemperature,
    SetHvacMode,
    OpenCover,
    CloseCover,
    StopCover,
    SetCoverPosition,
    ActivateScene,
    MediaPlay,
    MediaPause,
    MediaNext,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slots {
    pub brightness_pct: Option<u8>,
    pub rgb_color: Option<[u8; 3]>,
    pub color_temp_kelvin: Option<u16>,
    pub temperature: Option<f64>,
    pub hvac_mode: Option<HvacMode>,
    pub position: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HomeCommand {
    pub domain: Domain,
    pub operation: Operation,
    pub target: String,
    pub room: Option<String>,
    pub slots: Slots,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarBlock {
    InvalidBrightness,
    InvalidTemperature,
    Negation,
    Conditional,
    ReportedSpeech,
    Past,
    Future,
    Information,
    Deferred,
    Reminder,
    Security,
    Incomplete,
    UnknownRoom,
}

impl GrammarBlock {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidBrightness => "invalid_brightness_percentage",
            Self::InvalidTemperature => "invalid_temperature",
            Self::Negation => "non_immediate:negation",
            Self::Conditional => "non_immediate:subjunctive_cond",
            Self::ReportedSpeech => "non_immediate:report_of_speech",
            Self::Past => "non_immediate:past_action",
            Self::Future => "non_immediate:future_action",
            Self::Information => "non_immediate:information_request",
            Self::Deferred => "non_immediate:deferred",
            Self::Reminder => "non_immediate:reminder_request",
            Self::Security => "security_order",
            Self::Incomplete => "incomplete_home_command",
            Self::UnknownRoom => "unresolved_room",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GrammarMatch {
    Unclaimed,
    Blocked(GrammarBlock),
    Command(HomeCommand),
}

const ROOMS: &[(&str, &str)] = &[
    ("escritorio", "escritorio"),
    ("oficina", "escritorio"),
    ("estudio", "escritorio"),
    ("living", "living"),
    ("sala", "living"),
    ("salon", "living"),
    ("libby", "living"),
    ("livin", "living"),
    ("libin", "living"),
    ("hall", "hall"),
    ("pasillo", "hall"),
    ("entrada", "hall"),
    ("cocina", "cocina"),
    ("kitchen", "cocina"),
    ("bano", "bano"),
    ("banio", "bano"),
    ("bathroom", "bano"),
    ("cuarto", "cuarto"),
    ("dormitorio", "cuarto"),
    ("habitacion", "cuarto"),
];
pub(crate) const ROOM_PREPS: &[&str] = &[
    "de la", "de el", "en el", "en la", "a la", "del", "al", "de", "en",
];

/// An alias must match the entire candidate; there is no fuzzy room substitution.
pub fn canonical_room(alias: &str) -> Option<&'static str> {
    ROOMS
        .iter()
        .find_map(|&(key, room)| (key == alias).then_some(room))
}

fn unqualified_target(target: &str) -> &str {
    ROOM_PREPS
        .iter()
        .find_map(|prep| target.strip_prefix(prep)?.strip_prefix(' '))
        .unwrap_or(target)
}

/// An explicit unresolved location is a clarification, never a microphone fallback.
fn resolve_room(
    target: &str,
    source_room: Option<&str>,
    domain: Domain,
    noun: &str,
) -> Result<Option<String>, GrammarBlock> {
    let explicit = unqualified_target(target);
    if let Some(room) = canonical_room(explicit) {
        return Ok(Some(room.into()));
    }
    if matches!(
        explicit,
        "hogar" | "toda la casa" | "todo el hogar" | "la casa entera"
    ) {
        return Ok(None);
    }
    if explicit != target && (domain != Domain::Light || matches!(noun, "luz" | "luces" | "brillo"))
    {
        return Err(GrammarBlock::UnknownRoom);
    }
    if target.is_empty() {
        return Ok(source_room.map(|room| canonical_room(&fold(room)).unwrap_or(room).to_owned()));
    }
    // Named fixtures can be resolved by the caller, with explicit scope when present.
    for (index, _) in target.match_indices(' ') {
        let tail = &target[index + 1..];
        if let Some(room) = canonical_room(unqualified_target(tail))
            && unqualified_target(tail) != tail
        {
            return Ok(Some(room.into()));
        }
    }
    if ["del", "de la", "de el", "en", "a la"]
        .iter()
        .any(|p| has_phrase(target, p))
    {
        return Err(GrammarBlock::UnknownRoom);
    }
    let known_fixture = matches!(
        (noun, target),
        ("luz" | "lampara", "principal")
            | ("luces" | "lamparas", "principales")
            | ("lampara" | "lamparas", "de pie")
    );
    if known_fixture || domain == Domain::MediaPlayer {
        return Ok(source_room.map(str::to_owned));
    }
    Err(GrammarBlock::UnknownRoom)
}

pub(crate) fn entity_at(word: &str) -> Option<Domain> {
    Some(match word {
        "luz" | "luces" | "lampara" | "lamparas" | "foco" | "focos" | "bombilla" | "bombillas"
        | "led" | "leds" | "velador" | "veladores" | "iluminacion" | "brillo" => Domain::Light,
        "aire" | "ac" | "clima" | "climatizador" | "calefaccion" | "calefactor" | "estufa"
        | "termotanque" | "termostato" | "temperatura" => Domain::Climate,
        "persiana" | "persianas" | "cortina" | "cortinas" | "blackout" | "toldo" | "toldos"
        | "estor" | "estores" => Domain::Cover,
        "escena" | "escenas" => Domain::Scene,
        "ventilador" | "ventiladores" | "extractor" | "extractores" => Domain::Fan,
        "tele" | "tv" | "television" | "televisor" | "musica" | "parlante" | "parlantes"
        | "bocina" | "bocinas" | "altavoz" | "altavoces" | "radio" | "volumen" | "cancion"
        | "tema" | "temas" | "playlist" => Domain::MediaPlayer,
        _ => return None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verb {
    On,
    Off,
    Set,
    Raise,
    Lower,
    Open,
    Close,
    Stop,
    Play,
    Pause,
    Next,
}

pub(crate) fn action(word: &str) -> Option<Verb> {
    Some(match word {
        "prende" | "prenda" | "prendan" | "prender" | "encende" | "encender" | "enciende"
        | "encienda" | "activa" | "activar" | "active" | "ilumina" => Verb::On,
        "apaga" | "apagar" | "apague" | "apaguen" | "corta" | "corte" | "desactiva"
        | "desactivar" | "desactive" => Verb::Off,
        "pone" | "poner" | "pon" | "ponga" | "poneme" | "ponela" | "ponelo" | "ajusta"
        | "ajustar" | "ajuste" | "deja" | "dejar" | "deje" | "configura" | "configurar"
        | "configure" | "cambia" | "cambiar" | "cambie" => Verb::Set,
        "subi" | "sube" | "subir" | "suba" | "levanta" | "levantar" => Verb::Raise,
        "baja" | "bajar" | "baje" | "bajen" => Verb::Lower,
        "abri" | "abre" | "abrir" | "abra" => Verb::Open,
        "cerra" | "cierra" | "cerrar" | "cierre" => Verb::Close,
        "para" | "parar" | "pare" | "detene" | "detener" | "deten" | "detenga" | "frena"
        | "frenar" | "frene" => Verb::Stop,
        "reproduci" | "reproduce" | "reproducir" | "segui" | "sigue" => Verb::Play,
        "pausa" | "pause" | "pausar" | "silencia" => Verb::Pause,
        "siguiente" | "salta" => Verb::Next,
        _ => return None,
    })
}

pub(crate) fn has_deferral_time(text: &str) -> bool {
    DEFERRAL.is_match(text)
}

pub(crate) fn security_order(text: &str) -> bool {
    let plain = words(text);
    LOCK_ORDER.is_match(&plain) || ALARM_ORDER.is_match(&plain)
}

/// Determine why speech with device evidence cannot authorize an immediate action.
pub(crate) fn non_immediate(core: &str) -> Option<GrammarBlock> {
    let folded = words(core);
    let has_device = folded.split_whitespace().any(|w| entity_at(w).is_some());
    let has_action = folded.split_whitespace().any(|w| action(w).is_some());
    if folded
        .split_whitespace()
        .any(|w| entity_at(w) == Some(Domain::Light))
        && brightness(&fold(core)).is_err()
    {
        return Some(GrammarBlock::InvalidBrightness);
    }
    if !has_device && !has_action {
        return None;
    }
    let checks = [
        (GrammarBlock::Negation, &*NEGATION),
        (GrammarBlock::Conditional, &*CONDITIONAL),
        (GrammarBlock::ReportedSpeech, &*REPORTED_SPEECH),
    ];
    for (reason, pattern) in checks {
        if pattern.is_match(if reason == GrammarBlock::ReportedSpeech {
            core
        } else {
            &folded
        }) {
            return Some(reason);
        }
    }
    if PAST_ACTION.is_match(core)
        || PERFECT_ACTION.is_match(&folded)
        || PAST_CONTEXT.is_match(&folded)
    {
        return Some(GrammarBlock::Past);
    }
    if FUTURE_ACTION.is_match(&folded) {
        return Some(GrammarBlock::Future);
    }
    if INFORMATION.is_match(&folded) {
        return Some(GrammarBlock::Information);
    }
    (has_device && has_deferral_time(&folded)).then_some(GrammarBlock::Deferred)
}

const SMALL_NUMBERS: &[&str] = &[
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciseis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidos",
    "veintitres",
    "veinticuatro",
    "veinticinco",
    "veintiseis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
];

fn cardinal(text: &str) -> Option<u8> {
    let (tens, unit) = text.split_once(' ').unwrap_or((text, ""));
    let base = match tens {
        "treinta" => 30,
        "cuarenta" => 40,
        "cincuenta" => 50,
        "sesenta" => 60,
        "setenta" => 70,
        "ochenta" => 80,
        "noventa" => 90,
        _ if unit.is_empty() => {
            return match tens {
                "un" | "una" => Some(1),
                "veintiun" | "veintiuna" => Some(21),
                "cien" | "ciento" => Some(100),
                _ => SMALL_NUMBERS
                    .iter()
                    .position(|&w| w == tens)
                    .and_then(|n| u8::try_from(n).ok()),
            };
        }
        _ => return None,
    };
    if unit.is_empty() {
        return Some(base);
    }
    let unit = unit.strip_prefix("y ").unwrap_or(unit);
    SMALL_NUMBERS[1..10]
        .iter()
        .position(|&w| w == unit)
        .map(|n| base + n as u8 + 1)
}

fn number(text: &str) -> Option<f64> {
    if NUMERIC.is_match(text) {
        return text
            .replace(',', ".")
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite());
    }
    if let Some((whole, fraction)) = [" coma ", " punto ", " con "]
        .iter()
        .find_map(|sep| text.split_once(sep))
    {
        let fraction = cardinal(fraction)?;
        let scale = if fraction < 10 {
            10.0
        } else if fraction < 100 {
            100.0
        } else {
            1000.0
        };
        return Some(f64::from(cardinal(whole)?) + f64::from(fraction) / scale);
    }
    cardinal(text).map(f64::from)
}

fn bounded_percent(value: f64) -> Option<u8> {
    (value.fract() == 0.0 && (0.0..=100.0).contains(&value)).then_some(value as u8)
}

fn brightness(text: &str) -> Result<Option<u8>, GrammarBlock> {
    let patterns = [&*EXPLICIT_PERCENT, &*BARE_PERCENT];
    let mut result = None;
    for pattern in patterns {
        for capture in pattern.captures_iter(text) {
            let value = capture[1]
                .parse::<u8>()
                .ok()
                .filter(|n| *n <= 100)
                .filter(|_| capture[1].bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or(GrammarBlock::InvalidBrightness)?;
            result.get_or_insert(value);
        }
    }
    if result.is_some() {
        return Ok(result);
    }
    let spoken = SPOKEN_PERCENT
        .captures_iter(text)
        .find_map(|m| cardinal(&m[1]));
    if spoken.is_some() {
        return Ok(spoken);
    }
    let text = words(text);
    Ok([
        ("maximo", 100),
        ("maxima", 100),
        ("full", 100),
        ("todo", 100),
        ("fuerte", 90),
        ("potente", 90),
        ("medio", 50),
        ("media", 50),
        ("mitad", 50),
        ("suave", 30),
        ("bajo", 30),
        ("baja", 30),
        ("tenue", 15),
        ("minimo", 10),
        ("minima", 10),
    ]
    .iter()
    .find_map(|&(w, n)| has_phrase(&text, w).then_some(n)))
}

const COLORS: &[(&str, [u8; 3])] = &[
    ("rojo", [255, 0, 0]),
    ("roja", [255, 0, 0]),
    ("verde", [0, 255, 0]),
    ("azul", [0, 0, 255]),
    ("amarillo", [255, 255, 0]),
    ("amarilla", [255, 255, 0]),
    ("naranja", [255, 128, 0]),
    ("naranjo", [255, 128, 0]),
    ("rosa", [255, 105, 180]),
    ("rosado", [255, 105, 180]),
    ("rosada", [255, 105, 180]),
    ("violeta", [148, 0, 211]),
    ("morado", [148, 0, 211]),
    ("morada", [148, 0, 211]),
    ("purpura", [148, 0, 211]),
    ("celeste", [135, 206, 235]),
    ("turquesa", [64, 224, 208]),
    ("cian", [0, 255, 255]),
    ("blanco", [255, 255, 255]),
    ("blanca", [255, 255, 255]),
    ("dorado", [255, 215, 0]),
    ("dorada", [255, 215, 0]),
    ("plateado", [192, 192, 192]),
    ("plateada", [192, 192, 192]),
    ("negro", [0, 0, 0]),
    ("negra", [0, 0, 0]),
    ("magenta", [255, 0, 255]),
    ("fucsia", [255, 0, 255]),
];

const TEMPS: &[(&str, u16)] = &[
    ("calida", 2700),
    ("calido", 2700),
    ("tibia", 2700),
    ("tibio", 2700),
    ("amarilla", 2700),
    ("amarillenta", 2700),
    ("neutra", 4000),
    ("neutro", 4000),
    ("natural", 4000),
    ("fria", 6500),
    ("frio", 6500),
    ("azulada", 6500),
    ("fluorescente", 5500),
];

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotExtractor;

impl SlotExtractor {
    /// Extract the supported light slots; malformed numerical values block the request.
    pub fn extract(&self, text: &str) -> Result<Slots, GrammarBlock> {
        let text = fold(text);
        let numeric = NUMERIC_KELVIN.captures(&text);
        let plain = words(&text);
        let color_temp_kelvin = if let Some(m) = numeric {
            Some(
                m[1].parse::<u16>()
                    .ok()
                    .filter(|n| (1500..=9000).contains(n))
                    .ok_or(GrammarBlock::Incomplete)?,
            )
        } else {
            TEMPS
                .iter()
                .find_map(|&(w, n)| has_phrase(&plain, w).then_some(n))
        };
        Ok(Slots {
            brightness_pct: brightness(&text)?,
            color_temp_kelvin,
            rgb_color: COLORS
                .iter()
                .find_map(|&(w, rgb)| has_phrase(&plain, w).then_some(rgb)),
            ..Slots::default()
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct CommandGrammar {
    wake_words: Vec<String>,
}

impl CommandGrammar {
    pub fn new(wake_words: &[String]) -> Self {
        Self {
            wake_words: wake_words.iter().map(|w| words(w)).collect(),
        }
    }

    /// Parse a device command with explicit targets taking precedence over capturing room.
    pub fn parse(&self, text: &str, source_room: Option<&str>) -> GrammarMatch {
        parse_core(&strip_head(text, &self.wake_words), source_room)
    }
}

pub(crate) fn parse_core(core: &str, source_room: Option<&str>) -> GrammarMatch {
    if security_order(core) {
        return GrammarMatch::Blocked(GrammarBlock::Security);
    }
    if let Some(reason) = non_immediate(core) {
        return GrammarMatch::Blocked(reason);
    }
    let folded = fold(
        core.trim_start_matches(['¡', '¿', '?'])
            .trim_end_matches(['!', '.', ',', '?', '¿']),
    );
    let polite = ["podes ", "puedes ", "podrias ", "podria "]
        .iter()
        .find_map(|lead| folded.strip_prefix(lead));
    let direct_question = core.starts_with(['¿', '?']) || core.ends_with(['?', '¿']);
    if (direct_question && polite.is_none()) || folded.contains([';', '?', '¿']) {
        return GrammarMatch::Unclaimed;
    }
    let core = polite.unwrap_or(&folded);
    let core = core.strip_suffix(" ahora").unwrap_or(core);
    let (head, rest) = core.split_once(' ').unwrap_or((core, ""));
    let verb = action(head);
    let rest = trim_article(if verb.is_some() { rest } else { core });
    let (noun, tail) = rest.split_once(' ').unwrap_or((rest, ""));
    let (domain, tail) = match entity_at(noun) {
        Some(Domain::Climate) if noun == "aire" && prefix(tail, "acondicionado") => (
            Domain::Climate,
            tail.strip_prefix("acondicionado")
                .unwrap_or(tail)
                .trim_start(),
        ),
        Some(Domain::Light) if noun == "brillo" => (
            Domain::Light,
            tail.strip_prefix("de la luz ")
                .or_else(|| tail.strip_prefix("de las luces "))
                .unwrap_or(tail),
        ),
        Some(domain) => (domain, tail),
        None if matches!(head, "ponela" | "ponelo") && core.split_whitespace().count() <= 6 => {
            (Domain::Light, rest)
        }
        None => return GrammarMatch::Unclaimed,
    };
    let parsed = if domain == Domain::Light {
        light(verb, noun, tail)
    } else {
        extended(domain, verb, tail)
    };
    let (operation, target, slots) = match parsed {
        Ok(Some(value)) => value,
        Ok(None) => return GrammarMatch::Unclaimed,
        Err(reason) => return GrammarMatch::Blocked(reason),
    };
    let fixture = domain == Domain::Light
        && !matches!(noun, "luz" | "luces" | "brillo")
        && entity_at(noun).is_some();
    let room = if domain == Domain::Scene {
        None
    } else {
        match resolve_room(target, source_room, domain, noun) {
            Ok(room) => room,
            Err(reason) => return GrammarMatch::Blocked(reason),
        }
    };
    let target = if fixture {
        [noun, target].join(" ").trim_end().to_owned()
    } else {
        target.to_owned()
    };
    GrammarMatch::Command(HomeCommand {
        domain,
        operation,
        target,
        room,
        slots,
    })
}

fn trim_article(text: &str) -> &str {
    ["el ", "la ", "los ", "las "]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text)
}

type CommandParts<'a> = Result<Option<(Operation, &'a str, Slots)>, GrammarBlock>;

fn light<'a>(verb: Option<Verb>, noun: &str, tail: &'a str) -> CommandParts<'a> {
    let tail = strip_light_reason(tail);
    let (target, settings) = split_light_settings(tail);
    let target = target.trim_end_matches([',', ' ']);
    let target = if matches!(target, "encendida" | "encendidas" | "apagada" | "apagadas") {
        ""
    } else {
        target
    };
    let mut slots = Slots::default();
    if let Some(settings) = settings {
        slots = parse_light_settings(settings)?;
    }
    let set = slots != Slots::default();
    let operation = match verb {
        Some(Verb::On) => Operation::On,
        Some(Verb::Off) if !set => Operation::Off,
        Some(Verb::Raise) if !set => Operation::On,
        Some(Verb::Set | Verb::Raise | Verb::Lower) | None if set => Operation::SetLight,
        Some(Verb::Set | Verb::Lower) => return Err(GrammarBlock::Incomplete),
        _ => return Ok(None),
    };
    if verb.is_none()
        && (!matches!(noun, "luz" | "luces" | "brillo")
            || slots.brightness_pct.is_none()
            || slots.color_temp_kelvin.is_some()
            || slots.rgb_color.is_some())
    {
        return Ok(None);
    }
    if target
        .split_whitespace()
        .any(|w| matches!(w, "y" | "tambien" | "o"))
    {
        return Err(GrammarBlock::Incomplete);
    }
    Ok(Some((operation, target, slots)))
}

fn strip_light_reason(tail: &str) -> &str {
    LIGHT_REASON.find(tail).map_or(tail, |m| &tail[..m.start()])
}

fn split_light_settings(text: &str) -> (&str, Option<&str>) {
    for token in TOKEN.find_iter(text) {
        let start = token.start();
        let word = token.as_str();
        let suffix = &text[start..];
        let values = suffix.trim_start_matches(' ');
        let word = word.trim_end_matches('%');
        if LIGHT_ADJUSTMENT.is_match(values)
            || ["a ", "al ", "el ", "del "]
                .iter()
                .any(|p| values.strip_prefix(p).and_then(number).is_some())
            || word.chars().next().is_some_and(|c| c.is_ascii_digit())
            || TEMPS.iter().any(|&(name, _)| name == word)
            || COLORS.iter().any(|&(name, _)| name == word)
            || matches!(
                word,
                "maximo"
                    | "maxima"
                    | "full"
                    | "todo"
                    | "fuerte"
                    | "potente"
                    | "mitad"
                    | "medio"
                    | "media"
                    | "suave"
                    | "bajo"
                    | "baja"
                    | "tenue"
                    | "minimo"
                    | "minima"
            )
        {
            let target = text[..start].trim_end();
            let target = [
                "y al", "y a", "y en", "y con", "hasta el", "al", "a", "en", "con", "y",
            ]
            .iter()
            .find_map(|p| {
                target
                    .strip_suffix(p)
                    .filter(|left| left.is_empty() || left.ends_with(' '))
            })
            .unwrap_or(target)
            .trim_end();
            return (target, Some(values));
        }
    }
    (text, None)
}

fn parse_light_settings(text: &str) -> Result<Slots, GrammarBlock> {
    let text = ["hasta el ", "al ", "a ", "en ", "con ", "el ", "del "]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text);
    let (setting, brightness_tail) = LIGHT_BRIGHTNESS_SETTING
        .captures(text)
        .and_then(|m| Some((m.get(1)?.as_str(), Some(m.get(2)?.as_str()))))
        .unwrap_or((text, None));
    let slots = SlotExtractor.extract(setting)?;
    let percent = |value: &str| {
        let value = PERCENT_SUFFIX.replace(value, "");
        number(&value).and_then(bounded_percent)
    };
    if let Some(value) = percent(setting)
        && brightness_tail.is_none()
    {
        return Ok(Slots {
            brightness_pct: Some(value),
            ..Slots::default()
        });
    }
    let word = setting.split_whitespace().next().unwrap_or_default();
    let single = word == setting
        && (slots.brightness_pct.is_some()
            || slots.color_temp_kelvin.is_some()
            || slots.rgb_color.is_some());
    let kelvin = KELVIN_SETTING.is_match(setting);
    if !single && !kelvin {
        return Err(GrammarBlock::Incomplete);
    }
    if let Some(raw) = brightness_tail {
        let value = percent(raw)
            .filter(|level| *level > 0)
            .ok_or(GrammarBlock::Incomplete)?;
        return Ok(Slots {
            brightness_pct: Some(value),
            ..slots
        });
    }
    Ok(slots)
}

fn extended(domain: Domain, verb: Option<Verb>, tail: &str) -> CommandParts<'_> {
    let Some(verb) = verb else {
        return Ok(None);
    };
    let mut slots = Slots::default();
    let (operation, target) = match (domain, verb) {
        (Domain::Scene, Verb::On | Verb::Set) if !tail.is_empty() => {
            (Operation::ActivateScene, tail)
        }
        (Domain::Climate, Verb::Set) => {
            if let Some(m) = TEMPERATURE_SETTING.captures(tail)
                && let Some(value) = number(m[2].trim())
            {
                if !(10.0..=35.0).contains(&value) {
                    return Err(GrammarBlock::InvalidTemperature);
                }
                slots.temperature = Some(value);
                (
                    Operation::SetTemperature,
                    m.get(1).map_or("", |m| m.as_str()),
                )
            } else if let Some(m) = HVAC_MODE.captures(tail) {
                slots.hvac_mode = Some(match &m[2] {
                    "apagado" | "apagada" | "off" => HvacMode::Off,
                    "frio" | "fria" | "refrigeracion" => HvacMode::Cool,
                    "calor" | "calefaccion" => HvacMode::Heat,
                    "seco" | "deshumidificacion" | "deshumidificar" => HvacMode::Dry,
                    "ventilacion" | "solo ventilador" => HvacMode::FanOnly,
                    _ => HvacMode::Auto,
                });
                (Operation::SetHvacMode, m.get(1).map_or("", |m| m.as_str()))
            } else {
                return Err(GrammarBlock::Incomplete);
            }
        }
        (Domain::Climate, Verb::Off) => {
            slots.hvac_mode = Some(HvacMode::Off);
            (Operation::SetHvacMode, tail)
        }
        (Domain::Climate | Domain::Fan, Verb::On) => (Operation::On, tail),
        (Domain::Fan, Verb::Off) => (Operation::Off, tail),
        (Domain::Cover, Verb::Set | Verb::Raise | Verb::Lower | Verb::Open | Verb::Close) => {
            if let Some(m) = COVER_POSITION.captures(tail) {
                slots.position = Some(
                    number(m[2].trim())
                        .and_then(bounded_percent)
                        .ok_or(GrammarBlock::Incomplete)?,
                );
                (
                    Operation::SetCoverPosition,
                    m.get(1).map_or("", |m| m.as_str()),
                )
            } else if MALFORMED_COVER_POSITION.is_match(tail) {
                return Err(GrammarBlock::Incomplete);
            } else {
                let operation = match verb {
                    Verb::Raise | Verb::Open => Operation::OpenCover,
                    Verb::Lower | Verb::Close => Operation::CloseCover,
                    _ => return Err(GrammarBlock::Incomplete),
                };
                (operation, tail)
            }
        }
        (Domain::Cover, Verb::Stop) => (Operation::StopCover, tail),
        (Domain::MediaPlayer, Verb::Set | Verb::Play) => (Operation::MediaPlay, tail),
        (Domain::MediaPlayer, Verb::Pause | Verb::Stop) => (Operation::MediaPause, tail),
        (Domain::MediaPlayer, Verb::Next) => (Operation::MediaNext, tail),
        (Domain::Scene | Domain::Climate | Domain::Cover, _) => {
            return Err(GrammarBlock::Incomplete);
        }
        _ => return Ok(None),
    };
    if domain != Domain::Scene
        && target
            .split_whitespace()
            .any(|w| matches!(w, "y" | "tambien" | "o"))
    {
        return Err(GrammarBlock::Incomplete);
    }
    Ok(Some((operation, target, slots)))
}

regex_patterns! {
    LOCK_ORDER => concat!(
        r"\b(?:cerr(?:a|ar|e|en|ala|alo|arla|arlo)|cierr(?:a|e|en|ala|alo)|abr(?:i|ir|a|an|ila|ilo|irla|irlo)",
        r"|(?:des)?trab(?:a|ar|e|en|ala|alo|arla|arlo)|(?:des)?bloque(?:a|ar|e|en))\b(?: \S+){0,4}?",
        r" +(?:puertas?|porton(?:es)?|cerraduras?|cerrojos?|candados?|rejas?|garag(?:e|es)|garaj(?:e|es)|llave)\b",
        r"|\b(?:ech|pas)(?:a|ar|e|ale|arle) (?:la )?llave\b"
    );
    ALARM_ORDER => concat!(
        r"\b(?:(?:des)?activ|(?:des)?arm|(?:des)?conect)(?:a|ar|e|en|ala|arla)\b(?: \S+){0,3}? alarmas?\b",
        r"|\b(?:encend|prend)(?:e|er|ela|erla)\b(?: \S+){0,3}? alarmas?\b"
    );
    DEFERRAL => r"\b(?:manana|mas tarde|despues|luego|en (?:\w+ ){1,5}(?:segundos?|minutos?|horas?)|a las? (?:[0-9]+|una?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce))\b";
    NEGATION => r"\b(?:no|nunca|jamas|tampoco) \w+";
    CONDITIONAL => r"\b(?:ojala|si|cuando|aunque|en caso de que) \w+";
    REPORTED_SPEECH => r"\b(?:dije|dijo|dijiste|dijeron|coment[oé]|comentaste|comentaron|pregunt[oé]|preguntaste|preguntaron|avis[oé]|explic[oé])\b";
    PAST_ACTION => r"\b(?:\w+(?:aste|iste|aron|ieron|asteis|isteis)|apagué|prendí|encendí|bajé|cerré|cambié|pausé|frené|puse|puso|pusimos|apag[oó]|prendi[oó]|encendi[oó]|subi[oó]|bajó|cerró|abri[oó]|(?:ajust|configur|dej|activ|desactiv|par|cambi|levant)[éó]|reprodujo|apagaba[ns]?|prend[ií]a[ns]?|encend[ií]a[ns]?|sub[ií]a[ns]?|bajaba[ns]?|cerraba[ns]?|abr[ií]a[ns]?|pon[ií]a[ns]?)\b";
    PERFECT_ACTION => r"\b(?:he|has|ha|hemos|han|habia[ns]?|habiamos|hubo|habras?) (?:\w+ ){0,2}(?:apagado|prendido|encendido|subido|bajado|cerrado|abierto|puesto|activado|desactivado|cambiado)\b";
    PAST_CONTEXT => r"\b(?:ayer|anteayer|anoche|recien|antes|(?:ya|yo) (?:subi|abri))\b";
    FUTURE_ACTION => r"\b(?:(?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner)(?:e|as?|emos|an)|(?:voy|vas|va|vamos|van) a (?:apagar|prender|encender|subir|bajar|cerrar|abrir|activar|desactivar|poner))\b";
    INFORMATION => r"^(?:(?:decime|dime|contame|cuentame|explicame) )?(?:que|como|cuando|donde|quien|por que|para que)\b|^(?:(?:la luz|las luces|el aire|las? persianas?)(?: \w+){0,4} )?(?:esta|estas|estan|sigue|siguen|quedo|quedaron|tengo|tenes)\b";
    NUMERIC => r"^[0-9]+(?:[.,][0-9]+)?$";
    EXPLICIT_PERCENT => r"(?:^|[^\w.,+\-−])([+\-−]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)) *(?:%|por *ciento\b)(?:$|[^\w]|\b)";
    BARE_PERCENT => r"\bal ([+\-−]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+))($|[^\w.,])";
    SPOKEN_PERCENT => r"\b(\w+(?: y \w+)?) (?:por ciento|porciento)\b";
    NUMERIC_KELVIN => r"\b([0-9]+) *(?:k|kelvin)\b";
    LIGHT_REASON => r", (?:hace (?:frio|calor)|llueve|esta lloviendo)\b";
    TOKEN => r"\S+";
    LIGHT_ADJUSTMENT => r"^(?:al?|con|hasta el) (?:[+\-−]?[0-9]|[.,][0-9]|[a-z]+ (?:y [a-z]+ )?(?:por *ciento|porciento)\b)";
    LIGHT_BRIGHTNESS_SETTING => r"^(.*?)(?: y)? (?:al|a|con) (.+)$";
    PERCENT_SUFFIX => r" *(?:%|por *ciento|porciento)$";
    KELVIN_SETTING => r"^[0-9]+ *(?:k|kelvin)$";
    TEMPERATURE_SETTING => r"^(?:(.*?) )?(?:a|en) (.+?)(?: *(?:grados?(?: celsius| centigrados)?|° *c|celsius))?$";
    HVAC_MODE => r"^(?:(.*?) )?(?:en )?(?:modo )?(apagado|apagada|off|frio|fria|refrigeracion|calor|calefaccion|seco|deshumidificacion|deshumidificar|ventilacion|solo ventilador|automatico|automatica|auto)$";
    COVER_POSITION => r"^(?:(.*?) )?(?:al?|hasta el) (.+?) *(?:%|por ciento|porciento)$";
    MALFORMED_COVER_POSITION => r"\b(?:al?|hasta el) [0-9]|%|\bporciento\b|\bpor ciento\b";
}
