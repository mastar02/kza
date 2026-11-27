use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use chrono::{DateTime, FixedOffset};
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::time::Instant;

#[derive(Clone, Deserialize)]
pub struct EntityState {
    entity_id: String,
    state: String,
    attributes: Map<String, Value>,
    last_updated: Option<DateTime<FixedOffset>>,
    last_reported: Option<DateTime<FixedOffset>>,
}

impl fmt::Debug for EntityState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EntityState")
            .finish_non_exhaustive()
    }
}

impl EntityState {
    pub fn entity_id(&self) -> &str {
        &self.entity_id
    }

    pub fn state(&self) -> &str {
        &self.state
    }

    pub fn attributes(&self) -> &Map<String, Value> {
        &self.attributes
    }

    pub fn reported_at(&self) -> Option<DateTime<FixedOffset>> {
        self.last_reported.or(self.last_updated)
    }

    pub(crate) fn valid(&self) -> bool {
        // Home Assistant text/sensor entities (e.g. sensor.kza_focus_name) legitimately report "".
        domain(&self.entity_id).is_some()
    }
}

pub(crate) fn domain(entity_id: &str) -> Option<&str> {
    let (domain, name) = entity_id.split_once('.')?;
    let valid_part = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    };
    (valid_part(domain) && valid_part(name)).then_some(domain)
}

#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum StateReadError {
    #[error("no observation for this entity")]
    Missing,
    #[error("entity observation has expired")]
    Stale,
}

#[derive(Clone)]
pub(crate) struct Observation {
    pub state: Arc<EntityState>,
    pub as_of: Instant,
}

pub(crate) fn is_fresh(as_of: Instant, now: Instant, max_staleness: Duration) -> bool {
    now.checked_duration_since(as_of)
        .is_some_and(|age| age < max_staleness)
}

/// A point-in-time view. Publishing a new frame never changes an earlier frame.
#[derive(Clone)]
pub struct StateFrame {
    pub(crate) entries: HashMap<String, Observation>,
    pub(crate) deletions: HashMap<String, Instant>,
    pub(crate) snapshot_at: Option<Instant>,
    pub(crate) max_staleness: Duration,
}

impl fmt::Debug for StateFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StateFrame")
            .field("entity_count", &self.entries.len())
            .field("max_staleness", &self.max_staleness)
            .finish()
    }
}

impl StateFrame {
    pub(crate) fn empty(max_staleness: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            deletions: HashMap::new(),
            snapshot_at: None,
            max_staleness,
        }
    }

    pub fn max_staleness(&self) -> Duration {
        self.max_staleness
    }

    /// Freshness is per entity. An event for one entity cannot renew another.
    pub fn entity(&self, entity_id: &str, now: Instant) -> Result<&EntityState, StateReadError> {
        self.current(entity_id, now)
            .map(|observation| observation.state.as_ref())
    }

    pub(crate) fn current(
        &self,
        entity_id: &str,
        now: Instant,
    ) -> Result<&Observation, StateReadError> {
        let observation = self.entries.get(entity_id).ok_or(StateReadError::Missing)?;
        if !is_fresh(observation.as_of, now, self.max_staleness) {
            return Err(StateReadError::Stale);
        }
        Ok(observation)
    }
}

#[derive(Clone)]
pub(crate) struct Delta {
    pub entity_id: String,
    pub state: Option<Observation>,
    pub fired_at: Option<DateTime<FixedOffset>>,
    pub as_of: Instant,
}

impl Delta {
    pub fn apply(&self, frame: &mut StateFrame) -> bool {
        let version = match &self.state {
            Some(observation) => observation.state.last_updated,
            None => self.fired_at,
        };
        let current_version = frame
            .entries
            .get(&self.entity_id)
            .and_then(|current| current.state.last_updated);
        if matches!((version, current_version), (Some(incoming), Some(current)) if incoming < current)
        {
            return false;
        }
        if let Some(observation) = &self.state {
            frame.deletions.remove(&self.entity_id);
            frame
                .entries
                .insert(self.entity_id.clone(), observation.clone());
        } else {
            frame.entries.remove(&self.entity_id);
            frame.deletions.insert(self.entity_id.clone(), self.as_of);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(entity_id: &str, as_of: Instant) -> Delta {
        Delta {
            entity_id: entity_id.into(),
            state: Some(Observation {
                state: Arc::new(EntityState {
                    entity_id: entity_id.into(),
                    state: "off".into(),
                    attributes: Map::new(),
                    last_updated: None,
                    last_reported: None,
                }),
                as_of,
            }),
            fired_at: None,
            as_of,
        }
    }

    #[test]
    fn updating_one_entity_does_not_renew_another() {
        let mut frame = StateFrame::empty(Duration::from_secs(5));
        let start = Instant::now();
        observation("light.one", start).apply(&mut frame);
        let later = start + Duration::from_secs(5);
        observation("light.two", later).apply(&mut frame);
        assert!(matches!(
            frame.entity("light.one", later),
            Err(StateReadError::Stale)
        ));
        assert!(frame.entity("light.two", later).is_ok());
    }
}
