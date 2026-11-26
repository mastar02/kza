//! Deterministic device commands and slots.

use crate::normalize::{strip_head, words};

pub(crate) use guards::{has_deferral_time, non_immediate, security_order};
pub(crate) use lexicon::{action, entity_at};
pub(crate) use parse::parse_core;
pub(crate) use rooms::ROOM_PREPS;

use rooms::ROOMS;

mod guards;
mod lexicon;
mod light;
mod parse;
mod rooms;
mod slots;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Light,
    Climate,
    Cover,
    Scene,
    Fan,
    MediaPlayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvacMode {
    Off,
    Cool,
    Heat,
    Dry,
    FanOnly,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    On,
    Off,
    SetLight,
    SetTemperature,
    SetHvacMode,
    OpenCover,
    CloseCover,
    StopCover,
    SetCoverPosition,
    ActivateScene,
    MediaPlay,
    MediaPause,
    MediaNext,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Slots {
    pub brightness_pct: Option<u8>,
    pub rgb_color: Option<[u8; 3]>,
    pub color_temp_kelvin: Option<u16>,
    pub temperature: Option<f64>,
    pub hvac_mode: Option<HvacMode>,
    pub position: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HomeCommand {
    pub domain: Domain,
    pub operation: Operation,
    pub target: String,
    pub room: Option<String>,
    pub slots: Slots,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarBlock {
    InvalidBrightness,
    InvalidTemperature,
    Negation,
    Conditional,
    ReportedSpeech,
    Past,
    Future,
    Information,
    Deferred,
    Reminder,
    Security,
    Incomplete,
    UnknownRoom,
}

impl GrammarBlock {
    pub fn code(self) -> &'static str {
        match self {
            Self::InvalidBrightness => "invalid_brightness_percentage",
            Self::InvalidTemperature => "invalid_temperature",
            Self::Negation => "non_immediate:negation",
            Self::Conditional => "non_immediate:subjunctive_cond",
            Self::ReportedSpeech => "non_immediate:report_of_speech",
            Self::Past => "non_immediate:past_action",
            Self::Future => "non_immediate:future_action",
            Self::Information => "non_immediate:information_request",
            Self::Deferred => "non_immediate:deferred",
            Self::Reminder => "non_immediate:reminder_request",
            Self::Security => "security_order",
            Self::Incomplete => "incomplete_home_command",
            Self::UnknownRoom => "unresolved_room",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GrammarMatch {
    Unclaimed,
    Blocked(GrammarBlock),
    Command(HomeCommand),
}

/// An alias must match the entire candidate; there is no fuzzy room substitution.
pub fn canonical_room(alias: &str) -> Option<&'static str> {
    ROOMS
        .iter()
        .find_map(|&(key, room)| (key == alias).then_some(room))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SlotExtractor;

#[derive(Debug, Clone, Default)]
pub struct CommandGrammar {
    wake_words: Vec<String>,
}

impl CommandGrammar {
    pub fn new(wake_words: &[String]) -> Self {
        Self {
            wake_words: wake_words.iter().map(|w| words(w)).collect(),
        }
    }

    /// Parse a device command with explicit targets taking precedence over capturing room.
    pub fn parse(&self, text: &str, source_room: Option<&str>) -> GrammarMatch {
        parse_core(&strip_head(text, &self.wake_words), source_room)
    }
}

type CommandParts<'a> = Result<Option<(Operation, &'a str, Slots)>, GrammarBlock>;

pub(crate) fn prepare_patterns() {
    guards::prepare_patterns();
    slots::prepare_patterns();
    light::prepare_patterns();
    parse::prepare_patterns();
}
