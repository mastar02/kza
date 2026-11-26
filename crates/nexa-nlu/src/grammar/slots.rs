//! Spoken and numeric values shared by device settings.

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

pub(super) fn cardinal(text: &str) -> Option<u8> {
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

pub(super) fn number(text: &str) -> Option<f64> {
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

pub(super) fn bounded_percent(value: f64) -> Option<u8> {
    (value.fract() == 0.0 && (0.0..=100.0).contains(&value)).then_some(value as u8)
}

regex_patterns! {
    NUMERIC => r"^[0-9]+(?:[.,][0-9]+)?$";
}
