//! Route announcements to configured zones and report playback per zone.

use std::collections::{HashMap, HashSet};
use std::future::Future;

use crate::{Error, Result, required};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zone {
    pub id: String,
    /// Physical playback channel. A broadcast invokes each channel once.
    pub output: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Targets<'a> {
    All,
    Zones(&'a [&'a str]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Playback {
    Confirmed,
    NotStarted,
    Uncertain,
}

pub trait AnnouncementOutput {
    fn play(&self, output: &str, message: &str) -> impl Future<Output = Playback> + Send;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneReceipt {
    pub zone: String,
    pub playback: Playback,
}

#[derive(Debug, Clone)]
pub struct Intercom {
    zones: Vec<Zone>,
}

impl Intercom {
    pub fn new(zones: Vec<Zone>) -> Result<Self> {
        let mut ids = HashSet::new();
        for zone in &zones {
            required(&zone.id, "intercom zone ID is required")?;
            required(&zone.output, "intercom output ID is required")?;
            if !ids.insert(&zone.id) {
                return Err(Error::InvalidInput("duplicate intercom zone ID"));
            }
        }
        Ok(Self { zones })
    }

    pub fn zones(&self) -> &[Zone] {
        &self.zones
    }

    pub fn zone(&self, id: &str) -> Option<&Zone> {
        self.zones.iter().find(|zone| zone.id == id)
    }

    pub fn route(&self, targets: Targets<'_>) -> Result<Vec<&Zone>> {
        match targets {
            Targets::All => Ok(self.zones.iter().collect()),
            Targets::Zones(ids) => {
                let mut seen = HashSet::new();
                ids.iter()
                    .copied()
                    .filter(|id| seen.insert(*id))
                    .map(|id| self.zone(id).ok_or(Error::NotFound("intercom zone")))
                    .collect()
            }
        }
    }

    pub async fn announce<O: AnnouncementOutput>(
        &self,
        targets: Targets<'_>,
        message: &str,
        output: &O,
    ) -> Result<Vec<ZoneReceipt>> {
        required(message, "intercom message is required")?;
        let zones = self.route(targets)?;
        let mut results = HashMap::new();
        let mut receipts = Vec::with_capacity(zones.len());
        for zone in zones {
            let playback = match results.get(zone.output.as_str()) {
                Some(playback) => *playback,
                None => {
                    let playback = output.play(&zone.output, message).await;
                    results.insert(zone.output.as_str(), playback);
                    playback
                }
            };
            receipts.push(ZoneReceipt {
                zone: zone.id.clone(),
                playback,
            });
        }
        Ok(receipts)
    }
}
