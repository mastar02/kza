//! Pure NLU routing. The caller owns admission, execution and data acquisition.

use nexa_proto::v1::PathType;

use crate::gate::{
    AcceptanceDecision, CommandAcceptanceGate, EarconSettings, Evidence, GateContext, GateReason,
    GateSettings, should_play_earcon,
};
use crate::grammar::{self, Domain, GrammarMatch, HomeCommand};
use crate::household::{
    HouseholdFamily, HouseholdIntent, HouseholdIntentClassifier, HouseholdIntentSettings,
    HouseholdOperation,
};
use crate::normalize::{normalize_utterance, strip_head, words};
use crate::world::{self, ClockQuery, WeatherHorizon};

const MAX_UTTERANCE_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub path: PathType,
    pub normalized_text: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RouterSettings {
    pub gate: GateSettings,
    pub household: HouseholdIntentSettings,
    pub earcon: EarconSettings,
    /// Wake aliases stripped for commands without imposing a transcript wake.
    pub wake_words: Vec<String>,
    pub indoor_rooms: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RouteContext<'a> {
    pub gate: GateContext,
    pub source_room: Option<&'a str>,
    pub acoustic_wake: bool,
    /// Includes ambient strictness, textual wakes and captures without their own wake.
    pub tv_signal: bool,
    pub wake_score: f32,
    pub rms: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Route {
    Rejected,
    RefusalOnly,
    Household(HouseholdIntent),
    Compound(HouseholdIntent),
    Domotics(HomeCommand),
    Media(HomeCommand),
    Time(ClockQuery),
    Weather(WeatherHorizon),
    SlowLlm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoutedTurn {
    pub decision: RouteDecision,
    pub route: Route,
    pub acceptance: AcceptanceDecision,
    /// Positive local classification still requires a veto before dispatch when set.
    pub requires_veto: bool,
    pub earcon_eligible: bool,
}

#[derive(Debug, Clone)]
pub struct RouterEngine {
    gate: CommandAcceptanceGate,
    household: HouseholdIntentClassifier,
    earcon: EarconSettings,
    wakes: Vec<String>,
    indoor_rooms: Vec<String>,
}

impl Default for RouterEngine {
    fn default() -> Self {
        Self::new(RouterSettings::default())
    }
}

impl RouterEngine {
    /// Prepare the rule tables before the first transcript arrives.
    pub fn new(mut settings: RouterSettings) -> Self {
        grammar::prepare_patterns();
        HouseholdIntentClassifier::prepare();
        world::prepare_patterns();
        let mut wakes: Vec<_> = settings
            .wake_words
            .iter()
            .chain(&settings.household.wake_words)
            .chain(&settings.gate.wake_words)
            .map(|word| words(word))
            .filter(|word| !word.is_empty())
            .collect();
        wakes.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        wakes.dedup();
        settings.household.wake_words.clone_from(&wakes);
        Self {
            gate: CommandAcceptanceGate::new(settings.gate),
            household: HouseholdIntentClassifier::new(settings.household),
            earcon: settings.earcon,
            wakes,
            indoor_rooms: settings
                .indoor_rooms
                .iter()
                .map(|room| words(room))
                .collect(),
        }
    }

    /// Classify a full turn. Only the minimum evidence for the selected path is returned.
    pub fn route(&self, text: &str, context: RouteContext<'_>) -> RoutedTurn {
        if text.len() > MAX_UTTERANCE_BYTES {
            let reason = GateReason::UtteranceTooLong;
            return RoutedTurn {
                decision: RouteDecision {
                    path: PathType::Rejected,
                    normalized_text: String::new(),
                    reason: Some(reason.code().into()),
                },
                route: Route::Rejected,
                acceptance: AcceptanceDecision {
                    accept: false,
                    reason,
                    confidence: context.gate.confidence,
                    would_reject: Vec::new(),
                    ring_stop: false,
                    refusal_only: false,
                },
                requires_veto: false,
                earcon_eligible: false,
            };
        }
        let normalized_text = normalize_utterance(text);
        let core = strip_head(&normalized_text, &self.wakes);
        let claim = self.household.classify_in_room(text, context.source_room);
        let routine_read = (claim.is_none()
            || claim.as_ref().is_some_and(|i| {
                i.family == HouseholdFamily::Routine && i.operation == HouseholdOperation::Query
            }))
        .then(|| self.household.routine_read(&core))
        .flatten();
        let known_routine_read = routine_read.is_some();
        let claim = routine_read.or(claim);
        let ring_stop = if context.gate.ring_stop_open && claim.is_none() {
            self.household.ring_stop(text)
        } else {
            None
        };
        let clock = claim
            .is_none()
            .then(|| world::parse_clock_query(text, &self.wakes))
            .flatten();
        let weather = (claim.is_none() && clock.is_none())
            .then(|| world::parse_weather_query(&core, &self.indoor_rooms))
            .flatten();
        let acceptance = self.gate.evaluate_with_evidence(
            text,
            &core,
            context.gate,
            Evidence {
                claim: claim.as_ref(),
                read_only: clock.is_some() || weather.is_some() || known_routine_read,
                ring_stop: ring_stop.is_some(),
            },
        );
        let weak_wake = !context.acoustic_wake || context.tv_signal;
        let (path, route, reason, requires_veto) = if !acceptance.accept {
            (
                PathType::Rejected,
                Route::Rejected,
                Some(acceptance.reason.code()),
                false,
            )
        } else if acceptance.refusal_only {
            (
                PathType::FastHousehold,
                Route::RefusalOnly,
                Some("non_immediate:deferred"),
                false,
            )
        } else if let Some(intent) = claim.or(ring_stop) {
            let veto = if known_routine_read || intent.integration_query() {
                weak_wake
            } else {
                self.household
                    .requires_veto(text, context.acoustic_wake, context.tv_signal)
            };
            let reason = intent.reason;
            if intent.family == HouseholdFamily::Compound
                && intent.operation == HouseholdOperation::Execute
            {
                (PathType::Compound, Route::Compound(intent), reason, veto)
            } else {
                (
                    PathType::FastHousehold,
                    Route::Household(intent),
                    reason,
                    veto,
                )
            }
        } else if let Some(query) = clock {
            (PathType::FastTime, Route::Time(query), None, weak_wake)
        } else if let Some(query) = weather {
            (
                PathType::FastWeather,
                Route::Weather(query),
                None,
                weak_wake,
            )
        } else {
            match grammar::parse_core(&core, context.source_room) {
                GrammarMatch::Command(cmd) if cmd.domain == Domain::MediaPlayer => {
                    (PathType::FastHousehold, Route::Media(cmd), None, weak_wake)
                }
                GrammarMatch::Command(cmd) => (
                    PathType::FastDomotics,
                    Route::Domotics(cmd),
                    None,
                    weak_wake,
                ),
                GrammarMatch::Blocked(grammar::GrammarBlock::Information) => {
                    (PathType::SlowLlm, Route::SlowLlm, None, false)
                }
                GrammarMatch::Blocked(block) => (
                    PathType::Rejected,
                    Route::Rejected,
                    Some(block.code()),
                    false,
                ),
                GrammarMatch::Unclaimed => (PathType::SlowLlm, Route::SlowLlm, None, false),
            }
        };
        let earcon_eligible = path == PathType::Rejected
            && reason.is_some_and(|reason| {
                should_play_earcon(reason, context.wake_score, context.rms, &self.earcon)
            });
        RoutedTurn {
            decision: RouteDecision {
                path,
                normalized_text,
                reason: reason.map(str::to_owned),
            },
            route,
            acceptance,
            requires_veto,
            earcon_eligible,
        }
    }
}
