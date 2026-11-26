//! Ordered household claims. A claim owns the whole turn before device parsing.

use crate::grammar::HomeCommand;
use crate::normalize::words;

use HouseholdFamily::{Calendar, CameraHealth, Energy, Media, Routine, Timer};
use HouseholdOperation::{Disable, Execute, Query, Stop};
use utterance::Utterance;

mod compound;
mod energy;
mod lexicon;
mod queries;
mod routines;
mod rules;
mod tasks;
mod utterance;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HouseholdFamily {
    Timer,
    Reminder,
    List,
    Routine,
    Intercom,
    Notification,
    DeferredAction,
    Compound,
    Energy,
    Calendar,
    Media,
    CameraHealth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HouseholdOperation {
    Create,
    Cancel,
    Stop,
    Pause,
    Resume,
    Disable,
    Query,
    AddItem,
    RemoveItem,
    Execute,
    Propose,
    Clarify,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HouseholdPart {
    pub segment: String,
    pub intent: Option<HouseholdIntent>,
    pub command: Option<HomeCommand>,
    pub security_refusal: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HouseholdIntent {
    pub family: HouseholdFamily,
    pub operation: HouseholdOperation,
    pub rule: &'static str,
    pub complement: String,
    pub verb: String,
    pub reason: Option<&'static str>,
    pub parts: Vec<HouseholdPart>,
    /// Byte offsets into the input, for shielding literal intercom content.
    pub literal_span: Option<(usize, usize)>,
}

impl HouseholdIntent {
    fn new(
        family: HouseholdFamily,
        operation: HouseholdOperation,
        rule: &'static str,
        complement: String,
        verb: &str,
    ) -> Self {
        Self {
            family,
            operation,
            rule,
            complement,
            verb: verb.into(),
            reason: None,
            parts: Vec::new(),
            literal_span: None,
        }
    }

    fn because(mut self, reason: &'static str) -> Self {
        self.reason = Some(reason);
        self
    }

    fn controls_routine(&self) -> bool {
        self.family == Routine && matches!(self.operation, Execute | Stop | Disable)
    }

    pub(crate) fn integration_query(&self) -> bool {
        self.operation == Query && matches!(self.family, Energy | Calendar | Media | CameraHealth)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HouseholdIntentSettings {
    pub unvetoed_max_words: usize,
    pub compound_max_parts: usize,
    pub compound_execute: bool,
    pub wake_words: Vec<String>,
}

impl Default for HouseholdIntentSettings {
    fn default() -> Self {
        Self {
            unvetoed_max_words: 6,
            compound_max_parts: 3,
            compound_execute: true,
            wake_words: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HouseholdIntentClassifier {
    settings: HouseholdIntentSettings,
}

impl HouseholdIntentClassifier {
    pub(crate) fn prepare() {
        utterance::prepare_patterns();
        rules::prepare_patterns();
        tasks::prepare_patterns();
        routines::prepare_patterns();
        queries::prepare_patterns();
        energy::prepare_patterns();
    }

    pub fn new(mut settings: HouseholdIntentSettings) -> Self {
        settings.wake_words = settings
            .wake_words
            .iter()
            .map(|word| words(word))
            .filter(|word| !word.is_empty())
            .collect();
        Self { settings }
    }

    pub fn settings(&self) -> &HouseholdIntentSettings {
        &self.settings
    }

    /// Veto changes admission only. It never frees a claimed turn for device execution.
    pub fn requires_veto(&self, text: &str, acoustic_wake: bool, tv_signal: bool) -> bool {
        tv_signal
            || !acoustic_wake
            || text.split_whitespace().count() > self.settings.unvetoed_max_words
    }

    pub fn classify(&self, text: &str) -> Option<HouseholdIntent> {
        self.classify_in_room(text, None)
    }

    pub(crate) fn classify_in_room(
        &self,
        text: &str,
        source_room: Option<&str>,
    ) -> Option<HouseholdIntent> {
        let all = Utterance::tokenize(text);
        let utt = all.strip_fillers(&self.settings.wake_words);
        if utt.is_quoted(0) {
            return None;
        }
        rules::message(&utt)
            .or_else(|| compound::compound(&utt, &self.settings, source_room))
            .or_else(|| rules::single_after_messages(&utt))
    }

    pub(crate) fn ring_stop(&self, text: &str) -> Option<HouseholdIntent> {
        let all = Utterance::tokenize(text);
        let utt = all.strip_fillers(&self.settings.wake_words);
        (!utt.is_quoted(0) && utt.folded == ["listo"]).then(|| {
            HouseholdIntent::new(Timer, Stop, "R1", String::new(), "listo").because("ring_stop")
        })
    }
}
