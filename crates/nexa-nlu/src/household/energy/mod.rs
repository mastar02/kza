//! Sentence scope of household energy questions.
//! Grammar follows the P5 rule in the Python household classifier.

use super::utterance::Utterance;

use nominals::{indirect_expense_question, nominal_expenses, nominals_in_scope};
use objects::{object_in_scope, object_link};

macro_rules! scoped_re {
    ($pattern:expr) => {{
        static PATTERN: std::sync::LazyLock<fancy_regex::Regex> =
            std::sync::LazyLock::new(|| match fancy_regex::Regex::new($pattern) {
                Ok(regex) => regex,
                Err(error) => panic!("invalid energy scope pattern: {error}"),
            });
        &*PATTERN
    }};
}

macro_rules! number { () => { r"(?:\d+|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci(?:seis|siete|ocho|nueve)|veinte|veinti(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|media|medio|cuartos?)" }; }
macro_rules! electricity {
    () => {
        r"(?:energia|electricidad|luz|luces|corriente)"
    };
}
macro_rules! residence { () => { r"(?:casas?|hogar(?:es)?|instalacion(?:es)?|apartamentos?|departamentos?|viviendas?|domicilios?)" }; }
macro_rules! meter_scope {
    () => {
        concat!(r"(?:", residence!(), r"|medidor(?:es)?|red(?:es)?)",)
    };
}
macro_rules! units {
    () => {
        r"(?:kwh|kilowatts? hora|kilovatios? hora)"
    };
}

macro_rules! participle { () => { r"(?:\w+(?:ad|id)|abiert|cubiert|dich|escrit|hech|impres|muert|puest|resuelt|rot|(?:pre)?vist|vuelt)[oa]s?" }; }
macro_rules! accumulation_verb {
    () => {
        r"(?:llev\w*|tien\w*|ten(?:go|es|emos|eis|ia\w*)|tuv\w*)\b(?!\s+que\b)"
    };
}
macro_rules! determiner { () => { r"(?:el|la|los|las|un|una|mis?|tus?|sus?|nuestr[oa]s?|est[ea]|est[oa]s|es[ea]|es[oa]s|aquel|aquell[oa]s?)" }; }
macro_rules! np_determiner {
    () => {
        concat!(r"(?:(?:tod[oa]s? )?", determiner!(), r" )?",)
    };
}

macro_rules! document { () => { r"(?:boletas?|recibos?|facturas?|tarifas?|precios?|costos?|valores?|lecturas?|importes?|montos?)" }; }
macro_rules! adverb {
    () => {
        r"(?:aca|aqui|recien|mas o menos|de mas|\w+mente)"
    };
}

#[macro_use]
mod periods;
#[macro_use]
mod scope_patterns;

mod nominals;
mod objects;

fn other_residence() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r"\b(?:",
        residence_pre_alterity!(),
        r"(?>(?: ",
        residence_modifier!(),
        r")*) ",
        residence!(),
        r"|",
        residence!(),
        r"(?>(?: ",
        residence_modifier!(),
        r")*) ",
        residence_adjective!(),
        r")\b",
    ))
}

fn electricity_subject() -> &'static regex::Regex {
    re!(concat!(
        r"\b(?:",
        electricity!(),
        r"|(?:consumos?|tarifas?|facturas?) electric[oa]s?)\b",
    ))
}

fn household_meter() -> &'static regex::Regex {
    re!(concat!(r"\b", meter_scope!(), r"\b",))
}

fn unit() -> &'static regex::Regex {
    re!(concat!(r"\b", units!(), r"\b",))
}

fn power_unit() -> &'static regex::Regex {
    re!(r"\b(?:potencia|watts?|vatios?|kilowatts?|kilovatios?)\b")
}

fn non_household_target() -> &'static regex::Regex {
    re!(
        r"\b(?:heladeras?|refrigerador(?:es)?|neveras?|freezers?|teles?|televisor(?:es)?|tvs?|computadoras?|pcs?|notebooks?|laptops?|amplificador(?:es)?|electrodomesticos?|lavarropas|lavavajillas|secarropas|secadoras?|microondas|hornos?|estufas?|calefon(?:es)?|calefactor(?:es)?|calefaccion(?:es)?|aires?|ventilador(?:es)?|bombas?|autos?|coches?|carros?|vehiculos?|motos?|bicis?|bicicletas?|monopatin(?:es)?|consolas?|routers?|impresoras?|cafeteras?|licuadoras?|tostadoras?|planchas?|panel(?:es)? solar(?:es)?|aspiradoras?|cargador(?:es)?|secados?|lavados?|teclados?|livings?|cocinas?|dormitorios?)\b"
    )
}

fn meter_reading() -> &'static regex::Regex {
    re!(r"\b(?:consum\w*|gast\w*|usam\w*|medidor(?:es)?)\b")
}

fn accumulated_reading() -> &'static fancy_regex::Regex {
    scoped_re!(concat!(
        r"\b",
        accumulation_verb!(),
        r" (?:(?:",
        temporal!(),
        r"|est[ea] ",
        reading_period!(),
        r") )*",
        participle!(),
        r"\b",
    ))
}

fn cost() -> &'static regex::Regex {
    re!(
        r"\b(?:cost\w*|cuest\w*|facturas?|tarifas?|pag\w*|pesos?|dinero|precios?|sale|val(?:e|en|er))\b"
    )
}

fn spend() -> &'static regex::Regex {
    re!(concat!(
        r"\bgast\w* (?:",
        temporal!(),
        r" )*(?:de|en) (?:la )?(?:luz|electricidad)\b",
    ))
}

fn quantity() -> &'static regex::Regex {
    re!(r"^(?:cuanta|que) energia\b")
}

fn query() -> &'static regex::Regex {
    re!(
        r"^(?:cuant[oa]s?|que|cual(?:es)?|decime|mostrame|dame|consumo|potencia|energia|factura|tarifa|precio|costo|gastos?|pago)\b"
    )
}

fn light_or_cover() -> &'static regex::Regex {
    re!(
        r"\b(?:prend\w*|encend\w*|apag\w*|dame (?:mas )?luz|haya luz|cortinas?|persianas?|toldos?)\b"
    )
}

pub(super) fn measurement(utt: &Utterance<'_>, text: &str) -> Option<&'static str> {
    if (!query().is_match(text) && !indirect_expense_question().is_match(text).ok()?)
        || light_or_cover().is_match(text)
        || non_household_target().is_match(text)
        || other_residence().is_match(text).ok()?
    {
        return None;
    }
    let energy = unit().is_match(text);
    let power = power_unit().is_match(text);
    let electricity = electricity_subject().is_match(text);
    let household = household_meter().is_match(text);
    if !(energy || power || electricity || household) {
        return None;
    }
    let expenses = nominal_expenses(text, utt)?;
    if !nominals_in_scope(text, &expenses)? {
        return None;
    }
    for link in object_link().captures_iter(text) {
        let link = link.ok()?;
        let m = link.get(0)?;
        if !expenses.contains(&m.start())
            && !object_in_scope(&text[m.end()..], link.name("accumulation").is_some())?
        {
            return None;
        }
    }
    if (electricity || energy || (household && power)) && cost().is_match(text) {
        return Some("cost");
    }
    if !(energy || power || quantity().is_match(text)) && spend().is_match(text) {
        return Some("cost");
    }
    let reading = meter_reading().is_match(text) || accumulated_reading().is_match(text).ok()?;
    if energy || power || ((electricity || household) && reading) {
        return Some(if power && !energy { "power" } else { "energy" });
    }
    None
}

pub(super) fn prepare_patterns() {
    nominals::prepare_patterns();
    other_residence();
    objects::prepare_patterns();
    electricity_subject();
    household_meter();
    unit();
    power_unit();
    non_household_target();
    meter_reading();
    accumulated_reading();
    cost();
    spend();
    quantity();
    query();
    light_or_cover();
}
