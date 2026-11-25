#![cfg(test)]

use chrono::{DateTime, FixedOffset, NaiveDate, Weekday};
use nexa_nlu::world::{
    ClockQuery, CurrentWeather, ForecastAlignment, ForecastDay, ForecastEntry, NO_WEATHER_DATA,
    RelativeDay, WeatherHorizon, describe_clock, describe_current, describe_forecast,
    parse_clock_query, parse_weather_horizon, parse_weather_query,
};

#[test]
fn clock_matches_whole_queries_and_formats_local_time_and_date() {
    let now = DateTime::parse_from_rfc3339("2026-12-31T01:05:00-03:00").unwrap();
    for (text, sentence) in [
        ("Nexa, ¿qué hora es en este momento?", "Es la 1:05."),
        (
            "¿Me decís la fecha de mañana, por favor?",
            "Mañana es viernes 1 de enero de 2027.",
        ),
        ("¿Qué día fue ayer?", "Ayer fue miércoles 30 de diciembre."),
        (
            "Decime el día de pasado mañana",
            "Pasado mañana es sábado 2 de enero.",
        ),
    ] {
        let query = parse_clock_query(text, &[]).expect(text);
        assert_eq!(describe_clock(query, &now), sentence);
    }
    for text in [
        "¿Qué hora es en Madrid?",
        "¿A qué hora empieza el partido?",
        "¿Qué día es Navidad?",
        "¿Hoy qué día es mañana?",
        "¿Qué hora es y prendé la luz?",
    ] {
        assert_eq!(parse_clock_query(text, &[]), None, "{text}");
    }
    let custom = ["compa de casa".into()];
    assert_eq!(
        parse_clock_query("che, compa de casa, qué hora es, compa de casa", &custom),
        Some(ClockQuery::Time)
    );
}

#[test]
fn python_weather_horizon_table() {
    use ForecastDay::{Relative, Weekday as NamedWeekday};
    use RelativeDay::*;
    use WeatherHorizon::*;
    let mut mismatches = Vec::new();
    let weekdays = [
        Weekday::Mon,
        Weekday::Tue,
        Weekday::Wed,
        Weekday::Thu,
        Weekday::Fri,
        Weekday::Sat,
        Weekday::Sun,
    ];
    for line in include_str!("fixtures/weather_horizon.tsv").lines().skip(1) {
        let row: Vec<_> = line.split('\t').collect();
        let expected = match (row[0], row[1], row[2]) {
            ("current", _, _) => Current,
            ("unsupported", _, _) => Unsupported,
            ("several_days", _, _) => SeveralDays,
            ("forecast", "0", _) => Forecast(Relative(Today)),
            ("forecast", "1", _) => Forecast(Relative(Tomorrow)),
            ("forecast", "2", _) => Forecast(Relative(DayAfterTomorrow)),
            ("forecast", "None", index) => Forecast(NamedWeekday {
                day: weekdays[index.parse::<usize>().unwrap()],
                next_week: row[3] == "true",
            }),
            _ => panic!("malformed fixture: {line:?}"),
        };
        let actual = parse_weather_horizon(row[4]);
        if actual != expected {
            mismatches.push(format!(
                "{:?} expected {expected:?}, got {actual:?}",
                row[4]
            ));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn weather_distinguishes_outdoor_and_indoor_locations() {
    use WeatherHorizon::*;
    let extra = ["atelier".into()];
    for (text, expected) in [
        (
            "¿Qué temperatura marca el termómetro en el atelier?",
            Some(Indoor),
        ),
        ("¿Qué temperatura hay afuera de la cocina?", Some(Current)),
        ("Qué temperatura tiene el living", Some(Indoor)),
        ("Qué temperatura hay adentro", Some(Indoor)),
        ("Poné el clima a 22 grados", None),
        ("Prendé la estufa, hace frío", None),
        ("¿A qué hora va a estar el clima prendido?", None),
    ] {
        assert_eq!(parse_weather_query(text, &extra), expected, "{text}");
    }
}

#[test]
fn describes_current_weather_without_guessing_missing_values() {
    assert_eq!(
        describe_current(Some(CurrentWeather {
            condition: Some("partlycloudy"),
            temperature: Some(21.5),
            humidity: Some(70.5),
        })),
        "Hay 22 grados, parcialmente nublado, 70 por ciento de humedad."
    );
    assert_eq!(
        describe_current(Some(CurrentWeather {
            condition: Some("unknown"),
            temperature: Some(25.0),
            humidity: Some(90.0),
        })),
        NO_WEATHER_DATA
    );
    assert_eq!(
        describe_current(Some(CurrentWeather {
            condition: Some("sunny"),
            temperature: Some(f64::NAN),
            humidity: Some(f64::INFINITY),
        })),
        "Hay soleado."
    );
    assert_eq!(describe_current(None), NO_WEATHER_DATA);
}

#[test]
fn weather_forecast_aligns_by_local_date_with_explicit_missing_and_conflict() {
    let zone = FixedOffset::west_opt(3 * 3600).unwrap();
    let today = NaiveDate::from_ymd_opt(2026, 8, 4).unwrap();
    let day = ForecastDay::Weekday {
        day: Weekday::Tue,
        next_week: true,
    };
    assert_eq!(day.date(today), NaiveDate::from_ymd_opt(2026, 8, 11));
    assert_eq!(day.spoken(today), "el martes que viene");
    let cloudy = ForecastEntry {
        datetime: Some("2026-08-05T00:00:00+00:00"),
        condition: Some("cloudy"),
        low: Some(8.0),
        high: Some(17.0),
    };
    let later = ForecastEntry {
        datetime: Some("2026-08-05T03:00:00Z"),
        condition: Some("fog"),
        ..cloudy
    };
    let invalid = ForecastEntry {
        datetime: Some("2026-08-05T01:00:00"),
        ..cloudy
    };
    let conflict = ForecastEntry {
        high: Some(18.0),
        ..cloudy
    };
    let answer = describe_forecast(&[later, cloudy, cloudy], "hoy", Some(today), &zone);
    assert_eq!(answer.alignment, ForecastAlignment::Aligned);
    assert_eq!(answer.text, "Hoy: nublado, entre 8 y 17 grados.");
    let answer = describe_forecast(&[cloudy, conflict], "hoy", Some(today), &zone);
    assert_eq!(answer.alignment, ForecastAlignment::ConflictingDay);
    assert_eq!(answer.text, "No tengo el pronóstico para hoy.");
    assert_eq!(
        describe_forecast(&[cloudy, conflict, invalid], "hoy", Some(today), &zone).alignment,
        ForecastAlignment::InvalidEntries
    );
    assert_eq!(
        describe_forecast(&[later], "hoy", Some(today), &zone).alignment,
        ForecastAlignment::MissingDay
    );
    assert_eq!(
        describe_forecast(&[], "hoy", Some(today), &zone).alignment,
        ForecastAlignment::NoEntries
    );
}
