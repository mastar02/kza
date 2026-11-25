use std::hint::black_box;
use std::time::{Duration, Instant};

use nexa_nlu::RouterEngine;
use nexa_nlu::gate::GateSettings;
use nexa_nlu::router_engine::{RouteContext, RouterSettings};

/// Run explicitly in release mode so debug builds and contention in workspace tests
/// do not determine whether the latency budget passes.
#[test]
#[ignore = "run with cargo test --release -p nexa-nlu --test latency -- --ignored --nocapture"]
fn router_release_latency_budget() {
    let router = RouterEngine::new(RouterSettings {
        gate: GateSettings {
            initial_prompt: Some("Nexa es un asistente de voz para controlar las luces, la temperatura y los dispositivos de la casa.".into()),
            ..GateSettings::default()
        },
        ..RouterSettings::default()
    });
    let context = RouteContext {
        source_room: Some("escritorio"),
        acoustic_wake: true,
        ..RouteContext::default()
    };
    let oversized = "prendé la luz y ".repeat(4096);
    for text in [
        "Prendé la luz del living",
        "Poné el aire del baño a veintidós coma cinco grados",
        "Poné la luz cálida al 50 por ciento",
        "Poné un temporizador de 5 minutos y apagá la luz del living y bajá la persiana del baño al 30%",
        "Recordame que apague la luz de la cocina a las 10",
        "¿Qué hora es?",
        "¿Va a llover pasado mañana?",
        "Cuántos kwh consumimos la última semana en toda la casa",
        "Nexa es un asistente de voz para controlar las luces",
        "Esto es un asistente, apagá, subí.",
        oversized.as_str(),
    ] {
        let mut samples: Vec<_> = (0..1000)
            .map(|_| {
                let start = Instant::now();
                black_box(router.route(black_box(text), black_box(context)));
                start.elapsed()
            })
            .collect();
        samples.sort_unstable();
        let p95 = samples[949];
        let p99 = samples[989];
        let label = if text == oversized {
            "oversized compound"
        } else {
            text
        };
        eprintln!("{label:?}: p95 {p95:?}, p99 {p99:?}");
        assert!(p99 < Duration::from_micros(500), "{label:?} p99 {p99:?}");
    }
}
