//! Codec for KZA's legacy MA1260 simulation and compatible transport adapters.
//! The installed Dayton amplifier exposes a power trigger and analog controls.

use thiserror::Error;

const STX: u8 = 0x02;
const ETX: u8 = 0x03;
pub const ACK: u8 = 0x06;
pub const MAX_VOLUME_STEP: u8 = 38;
pub const LAST_ZONE: u8 = 6;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FrameError {
    #[error("invalid MA1260 frame length or delimiters")]
    Framing,
    #[error("invalid MA1260 zone {0}")]
    Zone(u8),
    #[error("unknown MA1260 command {0:#04x}")]
    Command(u8),
    #[error("invalid MA1260 frame data")]
    Data,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Command {
    Power = 0x01,
    Volume = 0x02,
    Mute = 0x03,
    Source = 0x04,
    Bass = 0x05,
    Treble = 0x06,
    Balance = 0x07,
    QueryStatus = 0x10,
}

impl TryFrom<u8> for Command {
    type Error = FrameError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(Self::Power),
            0x02 => Ok(Self::Volume),
            0x03 => Ok(Self::Mute),
            0x04 => Ok(Self::Source),
            0x05 => Ok(Self::Bass),
            0x06 => Ok(Self::Treble),
            0x07 => Ok(Self::Balance),
            0x10 => Ok(Self::QueryStatus),
            _ => Err(FrameError::Command(value)),
        }
    }
}

/// A command on the legacy wire format. Zone zero addresses all zones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandFrame {
    zone: u8,
    command: Command,
    data: u8,
}

impl CommandFrame {
    pub fn new(zone: u8, command: Command, data: u8) -> Result<Self, FrameError> {
        if zone > LAST_ZONE {
            return Err(FrameError::Zone(zone));
        }
        let valid_data = match command {
            Command::Power | Command::Mute => data <= 1,
            Command::Volume => data <= MAX_VOLUME_STEP,
            Command::Source => (1..=8).contains(&data),
            Command::Bass | Command::Treble | Command::Balance => data <= 20,
            Command::QueryStatus => data == 0,
        };
        if !valid_data {
            return Err(FrameError::Data);
        }
        Ok(Self {
            zone,
            command,
            data,
        })
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FrameError> {
        let &[STX, zone, command, data, ETX] = bytes else {
            return Err(FrameError::Framing);
        };
        Self::new(zone, Command::try_from(command)?, data)
    }

    #[must_use]
    pub fn encode(self) -> [u8; 5] {
        [STX, self.zone, self.command as u8, self.data, ETX]
    }

    #[must_use]
    pub fn zone(self) -> u8 {
        self.zone
    }

    #[must_use]
    pub fn command(self) -> Command {
        self.command
    }

    #[must_use]
    pub fn data(self) -> u8 {
        self.data
    }
}

/// Verify the single-byte command acknowledgment.
#[must_use]
pub fn is_acknowledgment(response: &[u8]) -> bool {
    response == [ACK]
}

/// Nearest wire step. Out-of-range inputs are clamped to the scale.
#[must_use]
pub fn percent_to_step(percent: u32) -> u8 {
    (f64::from(percent.min(100)) * f64::from(MAX_VOLUME_STEP) / 100.0).round() as u8
}

/// Nearest whole percentage represented by a wire step.
#[must_use]
pub fn step_to_percent(step: u8) -> u32 {
    (f64::from(step.min(MAX_VOLUME_STEP)) * 100.0 / f64::from(MAX_VOLUME_STEP)).round() as u32
}
