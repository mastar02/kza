use super::*;

impl Brain {
    pub(crate) async fn run_household(self: Arc<Self>) {
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => return,
                _ = tick.tick() => {
                    if self.poll_household().await.is_err() {
                        tracing::warn!("household polling failed");
                    }
                }
            }
        }
    }

    /// Advance due timers, shared notices, and routine runs using the injected clock.
    pub async fn poll_household(&self) -> Result<(), Status> {
        let clock = Arc::clone(&self.clock);
        let (timers, runs, delivered) = self
            .household
            .call(move |db| {
                // The running snapshot already includes newly claimed active occurrences.
                db.poll_due_scheduled_routines(clock.now())?;
                Ok((
                    db.complete_due_timers()?,
                    db.running_routines()?,
                    db.reminders_with_delivered_actions()?,
                ))
            })
            .await
            .map_err(household_status)?;
        let timers = stream::iter(timers).for_each_concurrent(8, |timer| async move {
            self.event(
                LiveEventKind::Alert,
                &timer.room,
                "",
                json!({ "event": "timer_due", "timer_id": timer.id.0 }),
            );
            self.announce(
                &timer.room,
                &format!("Terminó el temporizador {}.", timer.name),
            )
            .await;
        });
        let rooms = stream::iter(self.config.rooms.rooms.iter())
            .map(|room| self.deliver_notice(&room.room_id))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let runs = stream::iter(runs)
            .map(|run| self.advance_routine(run))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let actions = stream::iter(delivered)
            .map(|id| self.advance_reminder_action(id))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let (_, rooms, runs, actions) = tokio::join!(timers, rooms, runs, actions);
        rooms.into_iter().chain(runs).chain(actions).collect()
    }

    async fn deliver_notice(&self, room: &str) -> Result<(), Status> {
        if self.shutdown.is_cancelled() || self.output_busy(room) {
            return Ok(());
        }
        let zone = room.to_owned();
        let claim = self
            .household
            .call(move |db| db.claim_reminder_notice(&zone))
            .await
            .map_err(household_status)?;
        let Some(claim) = claim else { return Ok(()) };
        let result = self.announce(room, &claim.reminder.message).await;
        let outcome = match result {
            Playback::Confirmed => NoticeOutcome::Delivered,
            Playback::NotStarted => NoticeOutcome::NotStarted,
            Playback::Uncertain => NoticeOutcome::Uncertain,
        };
        let id = claim.reminder.id;
        self.household
            .call(move |db| db.finish_reminder_notice(&claim, outcome))
            .await
            .map_err(household_status)?;
        self.event(
            LiveEventKind::Alert,
            room,
            "",
            json!({ "event": "reminder_notice", "reminder_id": id.0 }),
        );
        Ok(())
    }

    async fn advance_reminder_action(
        &self,
        id: nexa_household::reminders::ReminderId,
    ) -> Result<(), Status> {
        if self.shutdown.is_cancelled() {
            return Ok(());
        }
        let action = self
            .household
            .call(move |db| db.claim_reminder_action(id))
            .await
            .map_err(household_status)?;
        let Some(action) = action else { return Ok(()) };
        let outcome = match self
            .home_effect(
                format!("reminder-{}-{}", id.0, action.position),
                &action.command,
            )
            .await
        {
            StepOutcome::Succeeded => ActionOutcome::Succeeded,
            StepOutcome::Failed => ActionOutcome::Failed,
            StepOutcome::Uncertain => ActionOutcome::Uncertain,
        };
        self.household
            .call(move |db| db.finish_reminder_action(&action, outcome))
            .await
            .map_err(household_status)?;
        Ok(())
    }

    async fn advance_routine(&self, run: RoutineRun) -> Result<(), Status> {
        if self.shutdown.is_cancelled() {
            return Ok(());
        }
        if let Some(nexa_household::routines::RoutineStep::Announce { room, .. }) =
            run.steps.get(run.next_step)
            && self.output_busy(room)
        {
            return Ok(());
        }
        let Some(step) = self
            .household
            .call(move |db| db.claim_routine_step(run.id))
            .await
            .map_err(household_status)?
        else {
            return Ok(());
        };
        let outcome = match &step.effect {
            RoutineEffect::Announce { room, message } => match self.announce(room, message).await {
                Playback::Confirmed => StepOutcome::Succeeded,
                Playback::NotStarted => StepOutcome::Failed,
                Playback::Uncertain => StepOutcome::Uncertain,
            },
            RoutineEffect::Home(action) => {
                self.home_effect(format!("routine-{}-{}", step.run.0, step.position), action)
                    .await
            }
        };
        self.household
            .call(move |db| db.finish_routine_step(&step, outcome))
            .await
            .map_err(household_status)?;
        Ok(())
    }

    async fn home_effect(
        &self,
        request_id: String,
        action: &nexa_household::HomeAction,
    ) -> StepOutcome {
        let Ok(action) = scheduled_action(action) else {
            return StepOutcome::Failed;
        };
        match self
            .execute_home(request_id, action)
            .await
            .as_ref()
            .map(|receipt| receipt.status.as_str())
        {
            Ok("accepted") => StepOutcome::Succeeded,
            Ok("uncertain" | "partial") => StepOutcome::Uncertain,
            _ => StepOutcome::Failed,
        }
    }
}
