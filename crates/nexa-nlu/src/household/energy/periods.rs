//! Time expressions in energy readings and bills.

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
