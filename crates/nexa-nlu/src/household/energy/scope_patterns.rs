//! Shared regex fragments for energy subjects and their complements.

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
