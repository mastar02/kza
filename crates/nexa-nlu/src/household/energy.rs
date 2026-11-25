//! Sentence scope of household energy questions.
//! Grammar follows the P5 rule in the Python household classifier.

use super::Utterance;

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
macro_rules! time_unit {
    () => {
        r"(?:mes(?:es)?|dias?|semanas?|anos?|bimestres?|periodos?|horas?)"
    };
}
macro_rules! reading_period { () => { r"(?:tardes?|noches?|mananas?|madrugadas?|findes?|fin de semana|quincenas?|trimestres?|semestres?|veranos?|inviernos?|otonos?|primaveras?)" }; }
macro_rules! temporal {
    () => {
        concat!(
            r"(?:hoy|ayer|ahora|actualmente|est[ea] ",
            time_unit!(),
            r"|(?:por|al) ",
            time_unit!(),
            r"|en (?:total|standby))",
        )
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
macro_rules! period_modifier { () => { r"(?:ultim[oa]s?|primer[oa]?s?|segund[oa]s?|tercer[oa]?s?|cuart[oa]s?|proxim[oa]s?|pasad[oa]s?|anterior(?:es)?)" }; }
macro_rules! period_ending {
    () => {
        r"(?: (?:pasad[oa]s?|anterior(?:es)?))?"
    };
}
macro_rules! month { () => { r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|setiembre|septiembre|octubre|noviembre|diciembre)" }; }
macro_rules! time_point {
    () => {
        r"(?:hoy|ayer|anteayer|antes de ayer|anoche|ahora(?: mismo)?|mediodia|medianoche|navidad)"
    };
}
macro_rules! tariff_band {
    () => {
        r"(?:horas?|horarios?) (?:pico|punta|valle)"
    };
}
macro_rules! day_number {
    () => {
        r"(?:[0-2]?\d|3[01])"
    };
}
macro_rules! month_number {
    () => {
        r"(?:0?[1-9]|1[0-2])"
    };
}
macro_rules! year {
    () => {
        r"20\d{2}"
    };
}
macro_rules! date {
    () => {
        concat!(
            r"(?:",
            day_number!(),
            r" de ",
            month!(),
            r"(?: de ",
            year!(),
            r")?|",
            day_number!(),
            r" ",
            month_number!(),
            r" ",
            year!(),
            r"|",
            year!(),
            r" ",
            month_number!(),
            r" ",
            day_number!(),
            r")",
        )
    };
}
macro_rules! period {
    () => {
        concat!(
            r"(?:",
            time_point!(),
            r"|lo que va (?:del|de la) (?:",
            time_unit!(),
            r"|",
            reading_period!(),
            r")|(?:cada |",
            np_determiner!(),
            r")(?:",
            period_modifier!(),
            r" )*(?:",
            date!(),
            r"|(?:",
            number!(),
            r" )?(?:",
            tariff_band!(),
            r"|dia de hoy|(?:fin|principio|comienzo|mitad) (?:de|del) ",
            time_unit!(),
            r"|",
            time_unit!(),
            r"|",
            reading_period!(),
            r"|",
            month!(),
            r"|lunes|martes|miercoles|jueves|viernes|sabado|domingo|",
            time_point!(),
            r"|minutos?|segundos?|momento|fecha|total|promedio|general|standby|pico|punta|valle)",
            period_ending!(),
            r"))",
        )
    };
}
macro_rules! document { () => { r"(?:boletas?|recibos?|facturas?|tarifas?|precios?|costos?|valores?|lecturas?|importes?|montos?)" }; }
macro_rules! adverb {
    () => {
        r"(?:aca|aqui|recien|mas o menos|de mas|\w+mente)"
    };
}
macro_rules! time_endpoint {
    () => {
        concat!(
            r"(?:",
            period!(),
            r"|las? ",
            number!(),
            r"|(?:[01]?\d|2[0-3])[:.][0-5]\d|\d+)",
        )
    };
}
macro_rules! dated_qualifier {
    () => {
        concat!(
            r"(?:",
            time_endpoint!(),
            r" (?:a|al) ",
            time_endpoint!(),
            r"|hace ",
            period!(),
            r"|",
            period!(),
            r"|",
            year!(),
            r"|(?:[01]?\d|2[0-3])[:.][0-5]\d|",
            day_number!(),
            r"|las? ",
            number!(),
            r")",
        )
    };
}
macro_rules! measurement_head {
    () => {
        concat!(
            r"(?:",
            meter_scope!(),
            r"|",
            electricity!(),
            r"|",
            units!(),
            r"|potencias?|(?:kilo)?(?:watts?|vatios?)|consumos?)",
        )
    };
}
macro_rules! verb {
    () => {
        concat!(
            r"(?:",
            accumulation_verb!(),
            r"|(?:consum[ei]\w*|usa\w*|gast\w*|tira\w*|necesit\w*|requier\w*|demand\w*)\b)",
        )
    };
}
macro_rules! existential {
    () => {
        r"(?:hay|hubo|es|son|fue|era\w*)"
    };
}
macro_rules! finite_verb {
    () => {
        concat!(r"(?:", existential!(), r"|tuv\w*|ten\w*|llev\w*|vin\w*)",)
    };
}
macro_rules! connector {
    () => {
        r"(?:de|del|en|con|sin|por|para|a|al|desde|hasta|durante|entre|sobre|si)"
    };
}
macro_rules! clause {
    () => {
        concat!(
            r"(?:",
            verb!(),
            r"|",
            finite_verb!(),
            r"|que|pero|se|estoy|estamos|esta|estan)",
        )
    };
}
macro_rules! prenominal {
    () => {
        concat!(
            r"(?!(?:",
            determiner!(),
            r"|",
            connector!(),
            r"|",
            clause!(),
            r"|y|e|o|u)\b)[^\W\d_]+",
        )
    };
}
macro_rules! np_prefix {
    () => {
        concat!(np_determiner!(), r"(?:", prenominal!(), r" )*?",)
    };
}
macro_rules! nominal_head {
    () => {
        concat!(r"(?:", measurement_head!(), r"|", document!(), r"|gastos?)",)
    };
}
macro_rules! of_scope {
    () => {
        concat!(
            r"(?:",
            dated_qualifier!(),
            r"|",
            np_prefix!(),
            r"(?:",
            meter_scope!(),
            r"(?: enter[oa]s?)?|",
            document!(),
            period_ending!(),
            r"|",
            measurement_head!(),
            r"|gastos?|ute))",
        )
    };
}
macro_rules! means { () => { r"(?:contador(?:es)?|tableros?|apps?|aplicacion(?:es)?|panel(?:es)?|pantallas?|sistemas?|plataformas?)" }; }
macro_rules! state_or_unit { () => { r"(?:standby|funcionamiento|reposo|espera|uso|carga|ciclo|pesos?|dinero|dolares?|euros?|plata|uyu|guita|kw)" }; }
macro_rules! residence_adjective {
    () => {
        r"(?:ajen[oa]s?|vecin[oa]s?|antigu[oa]s?|viej[oa]s?|anterior(?:es)?)"
    };
}
macro_rules! residence_pre_alterity {
    () => {
        concat!(
            r"(?:otr[oa]s?|cualquier(?:a)?|cada|",
            residence_adjective!(),
            r")",
        )
    };
}
macro_rules! residence_modifier {
    () => {
        concat!(
            r"(?!(?:",
            residence_pre_alterity!(),
            r"|",
            period!(),
            r"|",
            nominal_head!(),
            r"|",
            state_or_unit!(),
            r")\b)",
            prenominal!(),
        )
    };
}
macro_rules! in_scope {
    () => {
        concat!(
            r"(?:",
            time_endpoint!(),
            r"|",
            np_prefix!(),
            r"(?:",
            nominal_head!(),
            r"|",
            state_or_unit!(),
            r"|",
            means!(),
            r"))",
        )
    };
}
macro_rules! tariff_type { () => { r"(?:simple|(?:doble|triple)(?: horario)?|(?:bi|tri)?horaria|residencial|social|plana|nocturna|diurna|inteligente|estacional|variable|fija|reducida|(?:de )?(?:pico|punta|valle))" }; }
macro_rules! tariff {
    () => {
        concat!(r"(?:la )?tarifa(?: ", tariff_type!(), r")?",)
    };
}
macro_rules! fiscal { () => { concat!(
        np_determiner!(),
        r"(?:iva|impuestos?|descuentos?|recargos?|aumentos?|bonificacion\w*|subsidios?|cargos? fijos?)",
    ) }; }

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

fn indirect_expense_question() -> &'static fancy_regex::Regex {
    scoped_re!(r"^(?:decime|mostrame|dime|contame|saber) que (?=gastos?\b)")
}

fn expense_word() -> &'static regex::Regex {
    re!(r"\bgastos?\b")
}

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

fn object_link() -> &'static fancy_regex::Regex {
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

fn participle_word() -> &'static regex::Regex {
    re!(concat!(r"^(?:", participle!(), r")$",))
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

fn nominal_expenses(text: &str, utterance: &Utterance<'_>) -> Option<Vec<usize>> {
    let question = indirect_expense_question().find(text).ok()?;
    let start = question.map_or(1, |m| m.end());
    let main_end = subordinate()
        .find_at(text, start)
        .map_or(text.len(), |m| m.start());
    let finite = finite_utterance().is_match(&text[..main_end]);
    let mut positions = Vec::new();
    for m in expense_word().find_iter(text) {
        let index = text[..m.start()].bytes().filter(|&b| b == b' ').count();
        if utterance
            .tokens
            .get(index)
            .is_some_and(|token| token.spoken == "gastó")
        {
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

fn nominals_in_scope(text: &str, expenses: &[usize]) -> Option<bool> {
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

fn object_in_scope(mut tail: &str, accumulation: bool) -> Option<bool> {
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
    expense_intro();
    expense_question();
    finite_utterance();
    subordinate();
    indirect_expense_question();
    expense_word();
    other_residence();
    object_context();
    object_link();
    object_participle();
    participle_word();
    object_tail();
    nominal_event();
    of_tail();
    excess_tail();
    in_tail();
    in_determiner();
    in_physical_target();
    expense_tail();
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
