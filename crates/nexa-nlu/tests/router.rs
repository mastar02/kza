use nexa_nlu::gate::{
    CommandAcceptanceGate, EarconSettings, GateContext, GateReason, GateSettings, SttConfidence,
    should_play_earcon,
};
use nexa_nlu::grammar::{CommandGrammar, Domain, GrammarMatch, HvacMode, Operation, Slots};
use nexa_nlu::household::{HouseholdFamily, HouseholdIntentSettings, HouseholdOperation};
use nexa_nlu::normalize::{fold, normalize_utterance, words};
use nexa_nlu::router_engine::{Route, RouteContext, RouterSettings};
use nexa_nlu::world::{ClockQuery, ForecastDay, RelativeDay, WeatherHorizon};
use nexa_nlu::{RouterEngine, gate_utterance};
use nexa_proto::v1::PathType;
use proptest::prelude::*;

fn clear_wake() -> RouteContext<'static> {
    RouteContext {
        source_room: Some("escritorio"),
        acoustic_wake: true,
        wake_score: 0.9,
        rms: 0.04,
        ..RouteContext::default()
    }
}

#[test]
fn normalization_preserves_spoken_accents_and_folds_canonical_equivalents() {
    assert_eq!(
        normalize_utterance("  PRENDE\u{301}\t la  luz\n "),
        "prendé la luz"
    );
    assert_eq!(fold("SUBÍ Y CERRÁ el BAÑO"), "subi y cerra el bano");
    assert_eq!(
        words("¡Prendé_la luz, por favor!"),
        "prende_la luz por favor"
    );
    assert_eq!(
        gate_utterance("prendé la luz", 0.7, 2.2).path,
        PathType::SlowLlm
    );
    assert_eq!(
        gate_utterance("algo", 4.0, 2.2).reason.as_deref(),
        Some("high_compression")
    );
}

#[test]
fn admits_imperatives_and_routes_the_rioplatense_corpus() {
    let router = RouterEngine::default();
    let cases = [
        ("Nexa, prendé la luz", PathType::FastDomotics),
        ("Apagá la luz de la cocina", PathType::FastDomotics),
        (
            "Subí el brillo al cincuenta por ciento",
            PathType::FastDomotics,
        ),
        ("Bajá las persianas del salón", PathType::FastDomotics),
        (
            "Poné el aire a veintidós coma cinco grados",
            PathType::FastDomotics,
        ),
        (
            "¿Podés poner el termostato a veintidós coma cinco grados?",
            PathType::FastDomotics,
        ),
        ("Activá la escena cine", PathType::FastDomotics),
        (
            "Poné un temporizador de cinco minutos",
            PathType::FastHousehold,
        ),
        (
            "Recordame apagar la luz a las once",
            PathType::FastHousehold,
        ),
        ("Anotá yerba en la lista", PathType::FastHousehold),
        (
            "Armame una rutina que prenda la luz de noche",
            PathType::FastHousehold,
        ),
        ("Nexa, ¿qué hora es ahora?", PathType::FastTime),
        ("¿Qué temperatura hay en la sala?", PathType::FastWeather),
        ("¿Va a llover mañana?", PathType::FastWeather),
        (
            "Poné un timer de diez minutos y apagá la luz",
            PathType::Compound,
        ),
        ("Apagá la luz del living y de la cocina", PathType::Compound),
        ("¿Por qué el cielo se ve azul?", PathType::SlowLlm),
    ];
    for (text, path) in cases {
        let turn = router.route(text, clear_wake());
        assert_eq!(turn.decision.path, path, "{text}: {turn:?}");
    }
}

#[test]
fn outputs_parameters_and_exact_room_or_microphone_evidence() {
    use Domain::*;
    use Operation::*;
    let grammar = CommandGrammar::default();
    let cases = [
        (
            "Prendé la luz",
            Light,
            On,
            "",
            Some("living"),
            Slots::default(),
        ),
        (
            "Apagá la lámpara de pie del libby",
            Light,
            Off,
            "lampara de pie del libby",
            Some("living"),
            Slots::default(),
        ),
        (
            "Prendé las luces de la oficina",
            Light,
            On,
            "de la oficina",
            Some("escritorio"),
            Slots::default(),
        ),
        (
            "Subí el brillo del baño a treinta y cinco",
            Light,
            SetLight,
            "del bano",
            Some("bano"),
            Slots {
                brightness_pct: Some(35),
                ..Slots::default()
            },
        ),
        (
            "Bajá la luz al 60%",
            Light,
            SetLight,
            "",
            Some("living"),
            Slots {
                brightness_pct: Some(60),
                ..Slots::default()
            },
        ),
        (
            "Poné la luz del salón cálida al treinta por ciento",
            Light,
            SetLight,
            "del salon",
            Some("living"),
            Slots {
                color_temp_kelvin: Some(2700),
                brightness_pct: Some(30),
                ..Slots::default()
            },
        ),
        (
            "Encendé la luz en 4000 kelvin",
            Light,
            On,
            "",
            Some("living"),
            Slots {
                color_temp_kelvin: Some(4000),
                ..Slots::default()
            },
        ),
        (
            "Poné la luz en rojo",
            Light,
            SetLight,
            "",
            Some("living"),
            Slots {
                rgb_color: Some([255, 0, 0]),
                ..Slots::default()
            },
        ),
        (
            "Nexa, poné el aire del salón a 22,5 grados Celsius",
            Climate,
            SetTemperature,
            "del salon",
            Some("living"),
            Slots {
                temperature: Some(22.5),
                ..Slots::default()
            },
        ),
        (
            "Por favor ajustá la temperatura a treinta y cinco °C",
            Climate,
            SetTemperature,
            "",
            Some("living"),
            Slots {
                temperature: Some(35.0),
                ..Slots::default()
            },
        ),
        (
            "Dejá el aire del salón en calor",
            Climate,
            SetHvacMode,
            "del salon",
            Some("living"),
            Slots {
                hvac_mode: Some(HvacMode::Heat),
                ..Slots::default()
            },
        ),
        (
            "Apagá el aire",
            Climate,
            SetHvacMode,
            "",
            Some("living"),
            Slots {
                hvac_mode: Some(HvacMode::Off),
                ..Slots::default()
            },
        ),
        (
            "Cerrá las persianas del salón",
            Cover,
            CloseCover,
            "del salon",
            Some("living"),
            Slots::default(),
        ),
        (
            "Detené la persiana",
            Cover,
            StopCover,
            "",
            Some("living"),
            Slots::default(),
        ),
        (
            "Bajá la persiana del salón al treinta por ciento",
            Cover,
            SetCoverPosition,
            "del salon",
            Some("living"),
            Slots {
                position: Some(30),
                ..Slots::default()
            },
        ),
        (
            "Poné la persiana al 0%",
            Cover,
            SetCoverPosition,
            "",
            Some("living"),
            Slots {
                position: Some(0),
                ..Slots::default()
            },
        ),
        (
            "Activá la escena modo cine",
            Scene,
            ActivateScene,
            "modo cine",
            None,
            Slots::default(),
        ),
    ];
    for (text, domain, operation, target, room, slots) in cases {
        let GrammarMatch::Command(command) = grammar.parse(text, Some("living")) else {
            panic!(
                "could not parse {text:?}: {:?}",
                grammar.parse(text, Some("living"))
            );
        };
        assert_eq!(
            (
                command.domain,
                command.operation,
                command.target.as_str(),
                command.room.as_deref(),
                command.slots
            ),
            (domain, operation, target, room, slots),
            "{text}"
        );
    }
}

#[test]
fn ambiguous_or_non_immediate_device_requests_have_no_effect() {
    let router = RouterEngine::default();
    let cases = [
        "No prendas la luz",
        "Ayer apagué la luz",
        "Si hace calor prendé el aire",
        "Me dijo que apague el aire",
        "El sábado abriremos la persiana",
        "Dejé el aire a 22 grados",
        "¿Está encendida la luz?",
        "¿Prendiste la luz?",
        "Prendé la luz del quincho",
        "Prendé la luz de la cocina del quincho",
        "Poné la luz al -30%",
        "Poné la luz al +20%",
        "Poné la luz al 1000%",
        "Poné la luz al 30,5%",
        "Poné la luz cálida al dos tres por ciento",
        "Poné la luz cálida al 0%",
        "Poné la luz a 9999 kelvin",
        "Poné la luz cálida y la persiana",
        "Poné la persiana al -10%",
        "Bajá la persiana al 101%",
        "Poné la persiana al 30,5%",
        "Poné el aire a 2e1 grados",
        "Poné el aire a 70°F",
        "Poné el aire a -20 grados",
        "Poné el aire a dos tres grados",
        "Poné el aire a cien cinco grados",
        "Poné el aire a veintidós coma cinco seis grados",
        "Poné la luz fría y no la persiana",
        "Cerrá la puerta con llave",
        "Activá la alarma",
        "Activá las alarmas",
        "«apagá la luz y prendé el aire»",
    ];
    for text in cases {
        let turn = router.route(text, clear_wake());
        assert!(
            !matches!(turn.route, Route::Domotics(_) | Route::Compound(_)),
            "{text}: {turn:?}"
        );
    }
}

#[test]
fn gate_orders_hard_rules_and_confidence_shadow_and_detects_prompt_windows() {
    let settings = GateSettings {
        initial_prompt: Some("Interfaz local de voz con muchos componentes. Esta oración menciona el living y luego el escritorio.".into()),
        wake_words: vec!["Nexa".into()],
        enforce_confidence: true,
        enforce_compression_ratio: true,
        ..GateSettings::default()
    };
    let gate = CommandAcceptanceGate::new(settings);
    let both = GateContext {
        confidence: SttConfidence {
            compression_ratio: Some(3.4),
            no_speech_prob: Some(0.8),
            avg_logprob: Some(-2.1),
        },
        ..GateContext::default()
    };
    let cases = [
        ("", GateReason::Empty),
        ("...", GateReason::EmptyAfterNormalization),
        ("Aplausos", GateReason::NoisePhrase("aplausos")),
        ("Gracias", GateReason::FillerWord),
        ("hola hola hola hola", GateReason::WordRepetition),
        (
            "Esto es un asistente, apagá, subí.",
            GateReason::NoisePhrase("esto es un asistente"),
        ),
        (
            "Nexa gracias por ver",
            GateReason::NoisePhrase("gracias por ver"),
        ),
        (
            "menciona el living y luego el escritorio z",
            GateReason::PromptEcho,
        ),
        ("apagá la luz", GateReason::MissingWake),
        ("nexa apagá la luz", GateReason::HighCompression),
    ];
    for (text, expected) in cases {
        let decision = gate.evaluate(text, both);
        assert_eq!(decision.reason, expected, "{text}");
        assert!(!decision.accept);
    }
    let shadow = CommandAcceptanceGate::default().evaluate("apagá la luz encendida", both);
    assert!(shadow.accept);
    assert_eq!(
        shadow.would_reject,
        [GateReason::HighCompression, GateReason::LowConfidence]
    );
    assert_eq!(
        gate.evaluate(
            "nexa prendé la luz",
            GateContext {
                confidence: SttConfidence {
                    compression_ratio: Some(2.2),
                    ..both.confidence
                },
                ..both
            }
        )
        .reason,
        GateReason::LowConfidence
    );
}

#[test]
fn earcon_uses_speech_evidence_acoustic_floors_and_disabled_reasons() {
    let cfg = EarconSettings {
        enabled: true,
        reasons: [
            "empty",
            "high_compression",
            "unverified_intent",
            "noise_phrase",
        ]
        .map(str::to_owned)
        .to_vec(),
        ..EarconSettings::default()
    };
    assert!(should_play_earcon("empty", 0.9, 0.04, &cfg));
    assert!(should_play_earcon("stt_veto", 0.9, 0.001, &cfg));
    assert!(!should_play_earcon("empty", 0.9, 0.001, &cfg));
    assert!(!should_play_earcon("stt_veto", 0.4, 0.04, &cfg));
    for reason in [
        "noise_phrase:listo",
        "filler_word",
        "word_repetition",
        "missing_wake",
        "prompt_echo",
    ] {
        assert!(!should_play_earcon(reason, 1.0, 1.0, &cfg));
    }
    let router = RouterEngine::new(RouterSettings {
        earcon: cfg,
        ..RouterSettings::default()
    });
    let turn = router.route(
        "",
        RouteContext {
            rms: 0.001,
            gate: GateContext {
                stt_vetoed: true,
                ..GateContext::default()
            },
            ..clear_wake()
        },
    );
    assert!(turn.earcon_eligible);
}

#[test]
fn household_veto_ring_and_literal_rules_preserve_ownership() {
    let router = RouterEngine::new(RouterSettings {
        household: HouseholdIntentSettings {
            unvetoed_max_words: 4,
            ..HouseholdIntentSettings::default()
        },
        ..RouterSettings::default()
    });
    let short = "poné un timer de";
    assert!(!router.route(short, clear_wake()).requires_veto);
    assert!(
        router
            .route(
                short,
                RouteContext {
                    tv_signal: true,
                    ..clear_wake()
                }
            )
            .requires_veto
    );
    assert!(
        router
            .route(
                short,
                RouteContext {
                    acoustic_wake: false,
                    ..clear_wake()
                }
            )
            .requires_veto
    );
    let long = router.route("recordame que apague la luz mañana", clear_wake());
    assert!(long.requires_veto);
    assert!(
        matches!(long.route, Route::Household(intent) if intent.family == HouseholdFamily::Reminder)
    );
    let raw = "anunciá en la sala: «No apagues la luz. Gracias por ver, perfecto»";
    let announcement = router.route(raw, clear_wake());
    assert!(matches!(announcement.route, Route::Household(intent)
        if intent.family == HouseholdFamily::Intercom && intent.complement.ends_with("perfecto»")));
    assert!(matches!(
        router.route("listo", clear_wake()).route,
        Route::Rejected
    ));
    let ring = RouteContext {
        gate: GateContext {
            ring_stop_open: true,
            ..GateContext::default()
        },
        ..clear_wake()
    };
    assert!(
        matches!(router.route("bueno, listo, por favor", ring).route, Route::Household(intent)
        if intent.family == HouseholdFamily::Timer && intent.operation == HouseholdOperation::Stop)
    );
    assert!(matches!(
        router.route("listo, apagá la luz", ring).route,
        Route::Rejected
    ));
}

#[test]
fn compound_preserves_order_and_clarifies_unsafe_or_excess_segments() {
    let router = RouterEngine::default();
    let turn = router.route(
        "poné un temporizador de 5 minutos y apagá la luz del living",
        clear_wake(),
    );
    let Route::Compound(intent) = turn.route else {
        panic!("{turn:?}");
    };
    assert_eq!(intent.parts.len(), 2);
    assert!(matches!(&intent.parts[0].intent, Some(i) if i.family == HouseholdFamily::Timer));
    assert!(matches!(&intent.parts[1].command, Some(cmd) if cmd.operation == Operation::Off));
    let cases = [
        ("prendé la luz y apagá la luz", "conflicting_parts"),
        ("prendé la luz y abrí la persiana mañana", "deferred_part"),
        (
            "prendé la luz y ejecutá la rutina cine",
            "routine_control_part",
        ),
        (
            "prendé la luz y apagá la luz de la cocina y apagá la luz del baño y apagá la luz del hall",
            "too_many_parts",
        ),
        ("prendé la luz y apagá la luz del quincho", "unparsed_part"),
    ];
    for (text, reason) in cases {
        let turn = router.route(text, clear_wake());
        assert!(
            matches!(&turn.route, Route::Household(i)
            if i.family == HouseholdFamily::Compound && i.parts.is_empty() && i.reason == Some(reason)),
            "{text}: {turn:?}"
        );
    }
    let disabled = RouterEngine::new(RouterSettings {
        household: HouseholdIntentSettings {
            compound_execute: false,
            ..HouseholdIntentSettings::default()
        },
        ..RouterSettings::default()
    });
    assert!(
        matches!(disabled.route("prendé la luz y abrí la persiana", clear_wake()).route, Route::Household(intent)
        if intent.reason == Some("compound_disabled") && intent.parts.is_empty())
    );
}

#[test]
fn world_routes_use_typed_queries_and_weak_wake() {
    let router = RouterEngine::default();
    let clock = router.route("¿Qué hora es?", RouteContext::default());
    assert!(clock.requires_veto);
    assert_eq!(clock.route, Route::Time(ClockQuery::Time));
    assert_eq!(
        router
            .route("¿Qué temperatura hay en el living?", clear_wake())
            .route,
        Route::Weather(WeatherHorizon::Indoor)
    );
    assert_eq!(
        router
            .route("¿Va a llover pasado mañana?", clear_wake())
            .route,
        Route::Weather(WeatherHorizon::Forecast(ForecastDay::Relative(
            RelativeDay::DayAfterTomorrow
        )))
    );
    assert_eq!(
        router
            .route("¿Qué tiempo hace en marzo?", clear_wake())
            .route,
        Route::Weather(WeatherHorizon::Unsupported)
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn arbitrary_unicode_routes_are_deterministic_and_never_panic(text in any::<String>(), room in any::<String>()) {
        let router = RouterEngine::default();
        let context = RouteContext { source_room: Some(&room), ..clear_wake() };
        let left = router.route(&text, context);
        let right = router.route(&text, context);
        prop_assert_eq!(&left, &right);
        if let Route::Compound(intent) | Route::Household(intent) = left.route {
            prop_assert!(intent.parts.len() <= HouseholdIntentSettings::default().compound_max_parts);
            prop_assert!(intent.parts.iter().all(|part| part.intent.as_ref().is_none_or(|i| i.parts.is_empty())));
        }
    }

    #[test]
    fn repeated_imperatives_and_ellipsis_obey_the_configured_limit(n in 2usize..18, limit in 0usize..8, ellipsis in any::<bool>()) {
        let router = RouterEngine::new(RouterSettings {
            household: HouseholdIntentSettings { compound_max_parts: limit, ..HouseholdIntentSettings::default() },
            ..RouterSettings::default()
        });
        let text = if ellipsis {
            format!("apagá la luz del living{}", " y de la cocina".repeat(n - 1))
        } else {
            format!("prendé la luz{}", " y abrí la persiana".repeat(n - 1))
        };
        let turn = router.route(&text, clear_wake());
        if n <= limit {
            prop_assert!(matches!(&turn.route, Route::Compound(i) if i.parts.len() == n), "{turn:?}");
        } else {
            prop_assert!(matches!(&turn.route, Route::Household(i) if i.parts.is_empty() && i.reason == Some("too_many_parts")), "{turn:?}");
        }
    }
}

#[test]
fn read_only_routine_and_integration_queries_require_a_clear_wake_or_veto() {
    let router = RouterEngine::new(RouterSettings {
        household: HouseholdIntentSettings {
            unvetoed_max_words: 2,
            ..HouseholdIntentSettings::default()
        },
        ..RouterSettings::default()
    });
    for (text, family) in [
        (
            "¿A qué hora corre la rutina riego del fondo?",
            HouseholdFamily::Routine,
        ),
        (
            "¿La rutina riego del fondo está activa?",
            HouseholdFamily::Routine,
        ),
        (
            "Cuántos kwh consumimos en toda la casa",
            HouseholdFamily::Energy,
        ),
    ] {
        let clear = router.route(text, clear_wake());
        let weak = router.route(
            text,
            RouteContext {
                tv_signal: true,
                ..clear_wake()
            },
        );
        assert!(
            matches!(&clear.route, Route::Household(i)
            if i.family == family && i.operation == HouseholdOperation::Query),
            "{text}: {clear:?}"
        );
        assert!(!clear.requires_veto, "{text}");
        assert!(weak.requires_veto, "{text}");
    }
    assert!(matches!(
        router
            .route("¿Qué rutina de skincare me recomendás?", clear_wake())
            .route,
        Route::SlowLlm
    ));
}

#[test]
fn a_deferred_routine_reply_never_dispatches_the_original_command() {
    let router = RouterEngine::default();
    let text = "necesito que prendas las luces a las ocho";
    let rejected = router.route(text, clear_wake());
    assert_eq!(
        rejected.decision.reason.as_deref(),
        Some("non_immediate:deferred")
    );
    let reply = router.route(
        text,
        RouteContext {
            gate: GateContext {
                reply_to_deferred_routine: true,
                ..GateContext::default()
            },
            ..clear_wake()
        },
    );
    assert_eq!(reply.route, Route::RefusalOnly);
}

#[test]
fn temperature_commands_enforce_inclusive_celsius_bounds() {
    let router = RouterEngine::default();
    for value in [10, 22, 35] {
        let text = format!("Poné el aire a {value} grados");
        let turn = router.route(&text, clear_wake());
        assert_eq!(turn.decision.path, PathType::FastDomotics);
        let Route::Domotics(command) = turn.route else {
            panic!("{text}")
        };
        assert_eq!(command.operation, Operation::SetTemperature);
        assert_eq!(command.slots.temperature, Some(f64::from(value)));
    }
    for value in ["0", "5", "9", "9,9", "35,1", "36", "40", "100", "999"] {
        let text = format!("Poné el aire a {value} grados");
        let turn = router.route(&text, clear_wake());
        assert_eq!(turn.route, Route::Rejected, "{text}");
        assert_eq!(turn.decision.reason.as_deref(), Some("invalid_temperature"));
    }
}

#[test]
fn bare_light_targets_require_a_known_room_or_fixture() {
    let router = RouterEngine::default();
    for target in [
        "la luz quincho",
        "la luz garage",
        "la lámpara garage",
        "la luz del quincho",
    ] {
        let turn = router.route(&format!("Prendé {target}"), clear_wake());
        assert_eq!(turn.route, Route::Rejected, "{target}");
        assert_eq!(turn.decision.reason.as_deref(), Some("unresolved_room"));
    }
    for (target, expected_target, room) in [
        ("el velador", "velador", "escritorio"),
        ("la lámpara", "lampara", "escritorio"),
        ("la lámpara de pie", "lampara de pie", "escritorio"),
        ("la luz principal", "principal", "escritorio"),
        ("la luz salón", "salon", "living"),
    ] {
        let turn = router.route(&format!("Prendé {target}"), clear_wake());
        let Route::Domotics(command) = turn.route else {
            panic!("{target}: {turn:?}")
        };
        assert_eq!(command.room.as_deref(), Some(room));
        assert_eq!(command.target, expected_target);
    }
}

#[test]
fn rejects_oversized_compound_input_before_parsing() {
    use std::time::{Duration, Instant};
    let router = RouterEngine::default();
    let text = "prendé la luz y ".repeat(4096);
    assert!(text.len() >= 65_536);
    let start = Instant::now();
    let turn = router.route(&text, clear_wake());
    let elapsed = start.elapsed();
    assert_eq!(turn.route, Route::Rejected);
    assert_eq!(turn.acceptance.reason, GateReason::UtteranceTooLong);
    assert!(!turn.acceptance.accept);
    assert!(turn.decision.normalized_text.is_empty());
    assert!(elapsed < Duration::from_millis(5), "{elapsed:?}");
}

#[test]
fn parses_room_as_light_shorthand_and_trailing_wake_word() {
    let router = RouterEngine::default();
    for (phrase, expected_op, expected_room) in [
        ("prendé el escritorio", Operation::On, "escritorio"),
        ("apagá el escritorio", Operation::Off, "escritorio"),
        ("aprendé el escritorio", Operation::On, "escritorio"),
        ("aprende la luz", Operation::On, "escritorio"),
        ("Nexa, prendé la cocina, Nexa", Operation::On, "cocina"),
        ("apagá el living", Operation::Off, "living"),
        ("prendé el escritorio al 50%", Operation::On, "escritorio"),
        ("apagalo", Operation::Off, "escritorio"),
        ("apagala", Operation::Off, "escritorio"),
        ("prendelo", Operation::On, "escritorio"),
        ("prendela", Operation::On, "escritorio"),
        ("prendeme el escritorio", Operation::On, "escritorio"),
        ("apagame el escritorio", Operation::Off, "escritorio"),
    ] {
        let turn = router.route(phrase, clear_wake());
        let Route::Domotics(command) = turn.route else {
            panic!("{phrase}: {turn:?}");
        };
        assert_eq!(command.domain, Domain::Light, "{phrase}");
        assert_eq!(command.operation, expected_op, "{phrase}");
        assert_eq!(command.room.as_deref(), Some(expected_room), "{phrase}");
    }
    let dim = router.route("subila al 80%", clear_wake());
    let Route::Domotics(dim_cmd) = dim.route else {
        panic!("subila al 80%: {dim:?}");
    };
    assert_eq!(dim_cmd.domain, Domain::Light);
    assert_eq!(dim_cmd.operation, Operation::SetLight);
    assert_eq!(dim_cmd.slots.brightness_pct, Some(80));
    // Non-room / non-light phrases must remain unclaimed (SlowLlm), never misclassified as lights.
    // Unaccented STT slips like "apagara"/"prendera" must not be silently dropped as future tense.
    for phrase in [
        "prendé la tele",
        "aprende a tocar guitarra",
        "bajalo",
        "apagara la luz",
    ] {
        let turn = router.route(phrase, clear_wake());
        assert_eq!(turn.route, Route::SlowLlm, "{phrase}: {turn:?}");
    }

    // State questions and natural observations about devices route to SlowLlm, never FastDomotics.
    for text in [
        "¿Está encendida la luz?",
        "Tengo poca luz acá",
        "¿Cómo está el aire del living?",
    ] {
        let turn = router.route(text, clear_wake());
        assert_eq!(turn.route, Route::SlowLlm, "{text}: {turn:?}");
    }
}
