use super::*;

impl Brain {
    pub(crate) fn intercom(&self) -> Result<Intercom, Status> {
        Intercom::new(
            self.config
                .rooms
                .rooms
                .iter()
                .map(|room| Zone {
                    id: self.output_zone(&room.room_id),
                    output: room.ma1260_zone.to_string(),
                })
                .collect(),
        )
        .map_err(household_status)
    }

    pub(super) async fn voice_intercom(
        &self,
        intent: &HouseholdIntent,
        input: &str,
        turn: &Turn,
        conversation: &str,
    ) -> Result<String, Status> {
        let Some((start, end)) = intent.literal_span else {
            return Ok(CLARIFY.into());
        };
        let message = input
            .get(start..end)
            .unwrap_or_default()
            .trim_matches(|c: char| c.is_whitespace() || ['"', '\'', '«', '»'].contains(&c));
        if message.is_empty() || message.chars().count() > 320 {
            return Ok(CLARIFY.into());
        }
        let prefix = words(input.get(..start).unwrap_or_default());
        let directed = prefix
            .split_whitespace()
            .nth(1)
            .is_some_and(|word| matches!(word, "en" | "a" | "al" | "por" | "para"));
        let all = [
            "toda la casa",
            "todo el hogar",
            "todos lados",
            "todas las habitaciones",
        ]
        .iter()
        .any(|phrase| prefix.contains(phrase));
        let room = self.config.rooms.rooms.iter().find(|room| {
            prefix.split_whitespace().any(|word| {
                word == room.room_id || canonical_room(word) == Some(room.room_id.as_str())
            }) || prefix.contains(&words(&room.name))
        });
        let intercom = self.intercom()?;
        let target =
            self.output_zone(room.map_or(turn.zone.as_str(), |room| room.room_id.as_str()));
        let ids = [target.as_str()];
        let targets = if all {
            nexa_household::intercom::Targets::All
        } else {
            if directed && room.is_none() {
                return Ok("¿En qué habitación?".into());
            }
            nexa_household::intercom::Targets::Zones(&ids)
        };
        let mut heard = false;
        let mut uncertain = false;
        let mut channels = std::collections::HashSet::new();
        for zone in intercom.route(targets).map_err(household_status)? {
            if !channels.insert(&zone.output) {
                continue;
            }
            let Some(room) = self.room_for_zone(&zone.id) else {
                continue;
            };
            let playback = if room == turn.zone {
                match self.say(turn, conversation, message).await {
                    Ok(receipt) if receipt.completed => Playback::Confirmed,
                    _ => Playback::Uncertain,
                }
            } else {
                self.announce(room, message).await
            };
            heard |= playback == Playback::Confirmed;
            uncertain |= playback != Playback::Confirmed;
        }
        Ok(match (heard, uncertain) {
            (true, false) => "Aviso enviado.",
            (true, true) => "El aviso se reprodujo en parte de las zonas.",
            _ => "No pude confirmar el aviso.",
        }
        .into())
    }

    /// Feed an alert observation from a local detector and announce new high-priority episodes.
    pub async fn observe_alert(
        &self,
        observation: AlertObservation<'_>,
    ) -> Result<AlertUpdate, Status> {
        let key = observation.key.to_owned();
        let message = observation.message.to_owned();
        let lifetime = observation.lifetime;
        let priority = observation.priority;
        let (update, preferences) = self
            .household
            .call(move |db| {
                let update = db.observe_alert(AlertObservation {
                    key: &key,
                    priority,
                    message: &message,
                    lifetime,
                })?;
                Ok((update, db.alert_preferences()?))
            })
            .await
            .map_err(household_status)?;
        let alert = match &update {
            AlertUpdate::Opened(alert) | AlertUpdate::Escalated(alert) => alert,
            AlertUpdate::Deduplicated(_) => return Ok(update),
        };
        self.event(
            LiveEventKind::Alert,
            "",
            "",
            json!({ "alert_id": alert.id.0, "revision": alert.revision }),
        );
        if update.should_announce() {
            let intercom = self.intercom()?;
            let zones = preferences.selected_zones(
                &intercom,
                Some(&self.config.core.household.default_output_zone),
            );
            let mut channels = std::collections::HashSet::new();
            for zone in zones {
                let Some(output) = intercom.zone(zone).map(|zone| &zone.output) else {
                    continue;
                };
                let Some(room) = self.room_for_zone(zone) else {
                    continue;
                };
                let muted = self
                    .quiet
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .contains(room);
                if !muted && channels.insert(output) {
                    self.announce(room, &alert.message).await;
                }
            }
        }
        Ok(update)
    }
}
