use super::*;

impl Control {
    fn alert_zones(&self, config: AlertPreferences) -> v1::AlertZonesConfig {
        v1::AlertZonesConfig {
            revision: unsigned(config.revision),
            zone_ids: config.zones.unwrap_or_else(|| {
                let default = &self.0.config.core.household.default_output_zone;
                if self.0.room_for_zone(default).is_some() {
                    vec![default.clone()]
                } else {
                    Vec::new()
                }
            }),
        }
    }
}
pub(super) async fn list_alerts(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::AlertsSnapshot> {
    authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
    let alerts = control
        .0
        .household
        .call(|db| db.active_alerts())
        .await
        .map_err(household_status)?;
    ok(v1::AlertsSnapshot {
        alerts: alerts.into_iter().map(alert_record).collect(),
    })
}

pub(super) async fn ack_alert(
    control: &Control,
    request: Request<v1::AlertMutationRequest>,
) -> Reply<v1::AlertRecord> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::AdminManage])?;
    let id = AlertId(id(&request.alert_id)?);
    let expected = revision(request.expected_revision)?;
    let value = control
        .0
        .household
        .call(move |db| {
            if !db.acknowledge_alert(id, expected)? {
                return Err(nexa_household::Error::Conflict(
                    "alert changed or is inactive",
                ));
            }
            db.alert(id)?
                .ok_or(nexa_household::Error::NotFound("alert"))
        })
        .await
        .map_err(household_status)?;
    control.0.event(
        v1::LiveEventKind::Alert,
        "",
        "",
        json!({"event": "acknowledged", "alert_id": id.0}),
    );
    ok(alert_record(value))
}

pub(super) async fn delete_alert(
    control: &Control,
    request: Request<v1::AlertMutationRequest>,
) -> Reply<v1::Empty> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::AdminManage])?;
    let id = AlertId(id(&request.alert_id)?);
    let expected = revision(request.expected_revision)?;
    control
        .0
        .household
        .call(move |db| {
            let value = db
                .alert(id)?
                .ok_or(nexa_household::Error::NotFound("alert"))?;
            if value.revision != expected || !db.resolve_alert(id)? {
                return Err(nexa_household::Error::Conflict(
                    "alert changed or is inactive",
                ));
            }
            Ok(())
        })
        .await
        .map_err(household_status)?;
    control.0.event(
        v1::LiveEventKind::Alert,
        "",
        "",
        json!({"event": "resolved", "alert_id": id.0}),
    );
    ok(v1::Empty {})
}

pub(super) async fn get_alert_zones(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::AlertZonesConfig> {
    authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
    let preferences = control
        .0
        .household
        .call(|db| db.alert_preferences())
        .await
        .map_err(household_status)?;
    ok(control.alert_zones(preferences))
}

pub(super) async fn set_alert_zones(
    control: &Control,
    request: Request<v1::SetAlertZonesRequest>,
) -> Reply<v1::AlertZonesConfig> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::AdminManage])?;
    let revision = revision(request.expected_revision)?;
    let intercom = control.0.intercom()?;
    let value = control
        .0
        .household
        .call(move |db| {
            let zones = request
                .zone_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            db.set_alert_zones(revision, Some(&zones), &intercom)?
                .ok_or(nexa_household::Error::Conflict(
                    "alert zone revision changed",
                ))
        })
        .await
        .map_err(household_status)?;
    ok(control.alert_zones(value))
}
