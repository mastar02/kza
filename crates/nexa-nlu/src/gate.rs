//! Admission decisions using transcript evidence and caller-provided STT metadata.

use crate::grammar::{self, GrammarBlock, GrammarMatch, Operation};
use crate::household::{HouseholdFamily, HouseholdIntent, HouseholdIntentClassifier};
use crate::normalize::{strip_head, words};

const NOISE: &[&str] = &[
    "suscribe",
    "suscrib",
    "campanita",
    "gracias por ver",
    "dale like",
    "dale lie",
    "dale mega like",
    "canal de youtube",
    "luz encendida",
    "luz apagada",
    "luces encendidas",
    "luces apagadas",
    "hecho",
    "perfecto",
    "listo",
    "aplausos",
    "esto es un asistente",
];
const STATE_ECHO: &[&str] = &[
    "luz encendida",
    "luz apagada",
    "luces encendidas",
    "luces apagadas",
];
const FILLERS: &[&str] = &["gracias", "si", "no", "ok", "bueno", "dale"];
const SILENT_EARCON: &[&str] = &[
    "noise_phrase",
    "filler_word",
    "word_repetition",
    "missing_wake",
    "prompt_echo",
];

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SttConfidence {
    pub no_speech_prob: Option<f32>,
    pub avg_logprob: Option<f32>,
    pub compression_ratio: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GateContext {
    pub confidence: SttConfidence,
    pub stt_vetoed: bool,
    pub ring_stop_open: bool,
    pub reply_to_deferred_routine: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateSettings {
    /// If supplied, require one of these wake terms in the transcript.
    pub wake_words: Vec<String>,
    pub initial_prompt: Option<String>,
    pub enforce_confidence: bool,
    pub max_no_speech_prob: f32,
    pub min_avg_logprob: f32,
    pub enforce_compression_ratio: bool,
    pub max_compression_ratio: f32,
}

impl Default for GateSettings {
    fn default() -> Self {
        Self {
            wake_words: Vec::new(),
            initial_prompt: None,
            enforce_confidence: false,
            max_no_speech_prob: 0.60,
            min_avg_logprob: -1.20,
            enforce_compression_ratio: false,
            max_compression_ratio: 2.2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReason {
    Ok,
    Empty,
    EmptyAfterNormalization,
    UtteranceTooLong,
    SttVeto,
    NonImmediate(GrammarBlock),
    NoisePhrase(&'static str),
    FillerWord,
    WordRepetition,
    PromptEcho,
    MissingWake,
    HighCompression,
    LowConfidence,
}

impl GateReason {
    pub fn code(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::UtteranceTooLong => "utterance_too_long",
            Self::Empty => "empty",
            Self::EmptyAfterNormalization => "empty_after_norm",
            Self::SttVeto => "stt_veto",
            Self::NonImmediate(reason) => reason.code(),
            Self::NoisePhrase(_) => "noise_phrase",
            Self::FillerWord => "filler_word",
            Self::WordRepetition => "word_repetition",
            Self::PromptEcho => "prompt_echo",
            Self::MissingWake => "missing_wake",
            Self::HighCompression => "high_compression",
            Self::LowConfidence => "low_confidence",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceDecision {
    pub accept: bool,
    pub reason: GateReason,
    pub confidence: SttConfidence,
    pub would_reject: Vec<GateReason>,
    pub ring_stop: bool,
    /// A deferred routine reply may be answered only as a refusal or factual reply.
    pub refusal_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EarconSettings {
    pub enabled: bool,
    pub min_wake_score: f32,
    pub min_rms: f32,
    pub reasons: Vec<String>,
    pub exclude_reasons: Vec<String>,
}

impl Default for EarconSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            min_wake_score: 0.55,
            min_rms: 0.02,
            reasons: Vec::new(),
            exclude_reasons: Vec::new(),
        }
    }
}

/// Eligibility for the caller to play the "no entendí" tone.
pub fn should_play_earcon(
    reason: &str,
    wake_score: f32,
    rms: f32,
    config: &EarconSettings,
) -> bool {
    if !config.enabled
        || !wake_score.is_finite()
        || wake_score < config.min_wake_score
        || SILENT_EARCON
            .iter()
            .any(|prefix| reason.starts_with(prefix))
        || config
            .exclude_reasons
            .iter()
            .any(|prefix| reason.starts_with(prefix))
    {
        return false;
    }
    reason.starts_with("stt_veto")
        || (rms.is_finite()
            && rms >= config.min_rms
            && config
                .reasons
                .iter()
                .any(|prefix| reason.starts_with(prefix)))
}

#[derive(Debug, Clone)]
pub struct CommandAcceptanceGate {
    settings: GateSettings,
    wakes: Vec<String>,
    prompt_sentences: Vec<(String, usize)>,
}

impl Default for CommandAcceptanceGate {
    fn default() -> Self {
        Self::new(GateSettings::default())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Evidence<'a> {
    pub claim: Option<&'a HouseholdIntent>,
    pub read_only: bool,
    pub ring_stop: bool,
}

impl CommandAcceptanceGate {
    pub fn new(settings: GateSettings) -> Self {
        let prompt_sentences = settings
            .initial_prompt
            .as_deref()
            .into_iter()
            .flat_map(|prompt| prompt.split('.'))
            .map(words)
            .filter(|sentence| sentence.split_whitespace().count() >= 4)
            .map(|sentence| {
                let len = sentence.chars().count();
                (sentence, len)
            })
            .collect();
        let wakes = settings
            .wake_words
            .iter()
            .map(|word| words(word))
            .filter(|word| !word.is_empty())
            .collect();
        Self {
            settings,
            wakes,
            prompt_sentences,
        }
    }

    /// Evaluate independently with the default household vocabulary and supplied wake terms.
    pub fn evaluate(&self, text: &str, context: GateContext) -> AcceptanceDecision {
        let household = HouseholdIntentClassifier::new(crate::household::HouseholdIntentSettings {
            wake_words: self.wakes.clone(),
            ..crate::household::HouseholdIntentSettings::default()
        });
        let claim = household.classify(text);
        let core = strip_head(text, &self.wakes);
        let read_only = crate::world::parse_clock_query(text, &self.wakes).is_some()
            || crate::world::parse_weather_query(&core, &[]).is_some();
        self.evaluate_with_evidence(
            text,
            &core,
            context,
            Evidence {
                claim: claim.as_ref(),
                read_only,
                ring_stop: context.ring_stop_open && household.ring_stop(text).is_some(),
            },
        )
    }

    pub(crate) fn evaluate_with_evidence(
        &self,
        text: &str,
        core: &str,
        context: GateContext,
        evidence: Evidence<'_>,
    ) -> AcceptanceDecision {
        let confidence = context.confidence;
        let compression = confidence
            .compression_ratio
            .is_some_and(|n| n > self.settings.max_compression_ratio);
        let low_confidence = confidence
            .no_speech_prob
            .is_some_and(|n| n > self.settings.max_no_speech_prob)
            || confidence
                .avg_logprob
                .is_some_and(|n| n < self.settings.min_avg_logprob);
        let (hard, refusal_only) = self.hard_reason(text, core, context, evidence);
        let reason = if let Some(reason) = hard {
            reason
        } else if compression && self.settings.enforce_compression_ratio {
            GateReason::HighCompression
        } else if low_confidence && self.settings.enforce_confidence {
            GateReason::LowConfidence
        } else {
            GateReason::Ok
        };
        let accept = reason == GateReason::Ok;
        let mut would_reject = Vec::new();
        if accept {
            if compression {
                would_reject.push(GateReason::HighCompression);
            }
            if low_confidence {
                would_reject.push(GateReason::LowConfidence);
            }
        }
        AcceptanceDecision {
            accept,
            reason,
            confidence,
            would_reject,
            ring_stop: accept && evidence.ring_stop,
            refusal_only: accept && refusal_only,
        }
    }

    fn hard_reason(
        &self,
        text: &str,
        core: &str,
        context: GateContext,
        evidence: Evidence<'_>,
    ) -> (Option<GateReason>, bool) {
        use GateReason::*;
        if text.trim().is_empty() {
            return (
                Some(if context.stt_vetoed { SttVeto } else { Empty }),
                false,
            );
        }
        let normalized = words(text);
        if normalized.is_empty() {
            return (Some(EmptyAfterNormalization), false);
        }
        let masked = mask_intercom(text, evidence.claim);
        let safety_text = masked.as_ref().map(|text| strip_head(text, &self.wakes));
        let block = grammar::non_immediate(safety_text.as_deref().unwrap_or(core));
        let refusal_only =
            context.reply_to_deferred_routine && block == Some(GrammarBlock::Deferred);
        if let Some(reason) = block
            && reason != GrammarBlock::Information
            && !refusal_only
            && !evidence
                .claim
                .is_some_and(|intent| exempt_claim(intent, reason))
            && !(evidence.read_only
                && matches!(
                    reason,
                    GrammarBlock::Information | GrammarBlock::Deferred | GrammarBlock::Past
                ))
        {
            return (Some(NonImmediate(reason)), false);
        }
        let masked_noise = masked.as_deref().map(words);
        let noise_text = masked_noise.as_deref().unwrap_or(&normalized);
        for &phrase in NOISE {
            if !noise_text.contains(phrase) || (evidence.ring_stop && phrase == "listo") {
                continue;
            }
            if STATE_ECHO.contains(&phrase)
                && matches!(grammar::parse_core(core, None), GrammarMatch::Command(cmd)
                    if matches!(cmd.operation, Operation::On | Operation::Off))
            {
                continue;
            }
            return (Some(NoisePhrase(phrase)), false);
        }
        if FILLERS.contains(&normalized.as_str()) {
            return (Some(FillerWord), false);
        }
        let mut tokens = normalized.split_whitespace();
        if let Some(first) = tokens.next()
            && tokens.clone().count() >= 3
            && tokens.all(|word| word == first)
        {
            return (Some(WordRepetition), false);
        }
        if self.is_prompt_echo(&normalized) {
            return (Some(PromptEcho), false);
        }
        if !self.wakes.is_empty() && !self.wakes.iter().any(|wake| normalized.contains(wake)) {
            return (Some(MissingWake), false);
        }
        (None, refusal_only)
    }

    fn is_prompt_echo(&self, normalized: &str) -> bool {
        if self.prompt_sentences.is_empty() || normalized.split_whitespace().take(4).count() < 4 {
            return false;
        }
        let size = normalized.chars().count();
        // A contiguous match of at least 90% exists iff a window of that size occurs in the prompt.
        let window = size - size / 10;
        if !self.prompt_sentences.iter().any(|(_, len)| *len >= window) {
            return false;
        }
        let ends = normalized
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(normalized.len()))
            .skip(window);
        normalized
            .char_indices()
            .map(|(i, _)| i)
            .zip(ends)
            .any(|(start, end)| {
                self.prompt_sentences.iter().any(|(sentence, len)| {
                    *len >= window && sentence.contains(&normalized[start..end])
                })
            })
    }
}

fn exempt_claim(intent: &HouseholdIntent, reason: GrammarBlock) -> bool {
    intent.integration_query()
        || matches!(
            intent.family,
            HouseholdFamily::Reminder | HouseholdFamily::Notification
        )
        || intent.reason == Some("new_routine")
        || (intent.family == HouseholdFamily::Compound && reason == GrammarBlock::Deferred)
}

fn mask_intercom(text: &str, claim: Option<&HouseholdIntent>) -> Option<String> {
    let claim = claim?;
    let mut spans: Vec<_> = claim
        .literal_span
        .into_iter()
        .chain(
            claim
                .parts
                .iter()
                .filter_map(|part| part.intent.as_ref()?.literal_span),
        )
        .collect();
    if spans.is_empty() {
        return None;
    }
    spans.sort_unstable();
    let mut masked = String::with_capacity(text.len());
    let mut from = 0;
    for (start, end) in spans {
        let (Some(head), Some(_)) = (text.get(from..start), text.get(start..end)) else {
            continue;
        };
        masked.push_str(head);
        masked.push_str(" mensaje ");
        from = end;
    }
    masked.push_str(&text[from..]);
    Some(masked)
}
