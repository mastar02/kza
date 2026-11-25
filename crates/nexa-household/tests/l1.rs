#![allow(clippy::expect_used)]

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

use nexa_household::alerts::{
    ALERT_COOLDOWN, AlertObservation, AlertPriority, AlertUpdate, EpisodeLifetime,
};
use nexa_household::intercom::{AnnouncementOutput, Intercom, Playback, Targets, Zone};
use nexa_household::lists::{ListSnapshot, TodoChange, TodoSync};
use nexa_household::reminders::{
    ActionOutcome, ActionState, ChoiceAnswer, ChoiceId, NoticeOutcome, NoticeState, Reminder,
    ReminderPolicy, ReminderState, VoiceCancel, VoiceContext,
};
use nexa_household::routines::{
    Occurrence, RoutineSpec, RoutineStep, RunClaim, RunState, StepOutcome, VoiceBrake,
};
use nexa_household::timers::{RING_STOP_WINDOW, TimerState};
use nexa_household::{
    Clock, Error, HomeAction, Household, HouseholdStore, HouseholdWorker, Timestamp,
};
use proptest::prelude::*;

#[derive(Debug, Default)]
struct TestClock {
    wall: AtomicI64,
    elapsed: AtomicU64,
}

impl Clock for TestClock {
    fn now(&self) -> Timestamp {
        self.wall.load(Ordering::SeqCst)
    }

    fn monotonic(&self) -> Duration {
        Duration::from_millis(self.elapsed.load(Ordering::SeqCst))
    }
}

impl TestClock {
    fn advance(&self, ms: u64) {
        self.wall
            .fetch_add(i64::try_from(ms).expect("test duration"), Ordering::SeqCst);
        self.elapsed.fetch_add(ms, Ordering::SeqCst);
    }

    fn set_wall(&self, ms: i64) {
        self.wall.store(ms, Ordering::SeqCst);
    }
}

fn household() -> (Household<Arc<TestClock>>, Arc<TestClock>) {
    let clock = Arc::new(TestClock::default());
    let store = HouseholdStore::in_memory().expect("memory SQLite");
    (Household::new(store, Arc::clone(&clock)), clock)
}

fn action() -> HomeAction {
    HomeAction {
        service: "switch.turn_off".into(),
        entity_id: "switch.lamp".into(),
        data: Default::default(),
    }
}

fn voice() -> VoiceContext {
    VoiceContext {
        speaker: "voice-1".into(),
        room: "living".into(),
    }
}

fn choices(result: VoiceCancel) -> (ChoiceId, Vec<Reminder>) {
    let VoiceCancel::Choose { id, candidates } = result else {
        panic!("expected numbered choices, got {result:?}")
    };
    (id, candidates)
}

fn created(claim: RunClaim) -> nexa_household::routines::RoutineRun {
    let RunClaim::Created(run) = claim else {
        panic!("expected a new run")
    };
    run
}

fn routine_spec() -> RoutineSpec {
    RoutineSpec {
        name: "Buenas noches".into(),
        enabled: true,
        steps: vec![
            RoutineStep::Home { action: action() },
            RoutineStep::Delay {
                duration: Duration::from_secs(2),
            },
            RoutineStep::Announce {
                room: "living".into(),
                message: "Buenas noches".into(),
            },
        ],
    }
}

fn intercom() -> Intercom {
    Intercom::new(vec![
        Zone {
            id: "living".into(),
            output: "channel-1".into(),
        },
        Zone {
            id: "comedor".into(),
            output: "channel-1".into(),
        },
        Zone {
            id: "cocina".into(),
            output: "channel-2".into(),
        },
    ])
    .expect("configured zones")
}

#[test]
fn named_timer_pause_resume_cancel_and_ring_stop() -> nexa_household::Result<()> {
    let (mut household, clock) = household();
    let timer = household.create_timer("  Mate  ", "cocina", Duration::from_secs(5))?;
    assert_eq!(
        household.find_active_timer("mATE", "cocina")?,
        Some(timer.clone())
    );
    assert!(
        household
            .create_timer("mate", "cocina", Duration::from_secs(1))
            .is_err()
    );
    let other = household.create_timer("mate", "living", Duration::from_secs(5))?;
    clock.advance(2_000);
    assert!(household.pause_timer(timer.id)?);
    assert!(!household.pause_timer(timer.id)?);
    assert!(household.cancel_timer(other.id)?);
    assert!(!household.resume_timer(other.id)?);
    assert_eq!(
        household.timer(other.id)?.expect("cancelled record").state,
        TimerState::Cancelled
    );
    clock.advance(60_000);
    assert!(household.complete_due_timers()?.is_empty());
    assert_eq!(
        household
            .timer(timer.id)?
            .expect("paused timer")
            .remaining(clock.now()),
        Duration::from_secs(3)
    );
    assert!(household.resume_timer(timer.id)?);
    clock.advance(2_999);
    assert!(household.complete_due_timers()?.is_empty());
    clock.advance(1);
    assert!(!household.cancel_timer(timer.id)?);
    let completed = household.complete_due_timers()?;
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].state, TimerState::Completed);
    assert!(household.complete_due_timers()?.is_empty());
    assert_eq!(household.ringing_timers("cocina")?, completed);
    assert!(!household.stop_timer_ring(timer.id, "living")?);
    assert!(household.stop_timer_ring(timer.id, "cocina")?);
    assert!(!household.stop_timer_ring(timer.id, "cocina")?);
    let reused = household.create_timer("mate", "cocina", Duration::from_millis(1))?;
    assert_ne!(reused.id, timer.id);
    clock.advance(1);
    assert_eq!(household.complete_due_timers()?[0].id, reused.id);
    clock.advance(RING_STOP_WINDOW.as_millis() as u64 - 1);
    assert_eq!(household.ringing_timers("cocina")?.len(), 1);
    clock.advance(1);
    assert!(household.ringing_timers("cocina")?.is_empty());
    assert!(!household.stop_timer_ring(reused.id, "cocina")?);
    Ok(())
}

#[test]
fn reminder_voice_selection_uses_identity_room_deadline_and_snapshot() -> nexa_household::Result<()>
{
    let (mut household, clock) = household();
    let voice = voice();
    let first = household.create_reminder("comprar café", None, "cocina", 100_000, &[])?;
    let second = household.create_reminder("comprar café", None, "living", 200_000, &[])?;
    let personal = household.create_reminder("comprar café", Some("person-1"), "living", 0, &[])?;
    let (choice, items) = choices(household.begin_voice_cancel(&voice, "cafe")?);
    assert_eq!(items, [first.clone(), second.clone()]);
    assert_eq!(
        household.answer_reminder_choice(
            &VoiceContext {
                room: "otra".into(),
                ..voice.clone()
            },
            choice,
            ChoiceAnswer::Ordinal(1)
        )?,
        VoiceCancel::NoMatch
    );
    assert_eq!(
        household.answer_reminder_choice(
            &VoiceContext {
                speaker: "voice-2".into(),
                ..voice.clone()
            },
            choice,
            ChoiceAnswer::Ordinal(1)
        )?,
        VoiceCancel::NoMatch
    );
    clock.advance(30_000);
    assert_eq!(
        household.answer_reminder_choice(&voice, choice, ChoiceAnswer::Ordinal(1))?,
        VoiceCancel::NoMatch
    );
    let (stale, _) = choices(household.begin_voice_cancel(&voice, "cafe")?);
    let (choice, _) = choices(household.begin_voice_cancel(&voice, "cafe")?);
    assert_eq!(
        household.answer_reminder_choice(&voice, stale, ChoiceAnswer::Ordinal(2))?,
        VoiceCancel::NoMatch
    );
    assert!(household.cancel_reminder(first.id, None)?);
    let replacement = household.create_reminder("comprar café", None, "cocina", 50_000, &[])?;
    assert_eq!(
        household.answer_reminder_choice(&voice, choice, ChoiceAnswer::Ordinal(1))?,
        VoiceCancel::NoMatch
    );
    let (choice, _) = choices(household.begin_voice_cancel(&voice, "cafe")?);
    clock.advance(29_999);
    clock.set_wall(-200_000);
    assert_eq!(
        household.answer_reminder_choice(&voice, choice, ChoiceAnswer::Ordinal(2))?,
        VoiceCancel::Cancelled(second.id)
    );
    assert_eq!(household.pending_reminders(None)?, [replacement]);
    assert_eq!(household.pending_reminders(Some("person-1"))?, [personal]);
    Ok(())
}

#[test]
fn voice_cancel_requires_all_content_words_for_direct_match() -> nexa_household::Result<()> {
    let (mut household, _) = household();
    let target = household.create_reminder("llamar papá", None, "living", 5, &[])?;
    assert_eq!(
        household.begin_voice_cancel(&voice(), "mamá")?,
        VoiceCancel::NoMatch
    );
    let (choice, candidates) = choices(household.begin_voice_cancel(&voice(), "llamar mamá")?);
    assert_eq!(candidates.as_slice(), std::slice::from_ref(&target));
    assert_eq!(
        household.answer_reminder_choice(&voice(), choice, ChoiceAnswer::Other)?,
        VoiceCancel::NoMatch
    );
    assert_eq!(
        household.answer_reminder_choice(&voice(), choice, ChoiceAnswer::Ordinal(1))?,
        VoiceCancel::NoMatch
    );
    assert_eq!(
        household.begin_voice_cancel(&voice(), "llamar PAPA")?,
        VoiceCancel::Cancelled(target.id)
    );
    Ok(())
}

proptest! {
    #[test]
    fn only_an_explicitly_listed_and_current_choice_can_cancel(
        deadlines in proptest::collection::vec(-10_000_i64..10_000, 2..8),
        ordinal in 0_usize..8,
        elapsed in 0_u64..60_000,
        other_room in any::<bool>(),
        other_speaker in any::<bool>(),
        remove_selected in any::<bool>(),
    ) {
        let (mut household, clock) = household();
        let context = voice();
        let mut shared = Vec::new();
        for due_at in deadlines {
            shared.push(household.create_reminder("tomar pastillas", None, "living", due_at, &[])?);
        }
        let personal = household.create_reminder("tomar pastillas", Some("p1"), "living", -20_000, &[])?;
        shared.sort_by_key(|reminder| (reminder.due_at, reminder.id.0));
        let (choice, listed) = choices(household.begin_voice_cancel(&context, "pastillas")?);
        prop_assert_eq!(listed.as_slice(), &shared[..shared.len().min(3)]);
        let selected = ordinal.checked_sub(1).and_then(|index| listed.get(index)).map(|reminder| reminder.id);
        if let Some(id) = selected.filter(|_| remove_selected) {
            prop_assert!(household.cancel_reminder(id, None)?);
        }
        clock.advance(elapsed);
        let mut response = context;
        if other_room { response.room = "cocina".into(); }
        if other_speaker { response.speaker = "voice-2".into(); }
        let outcome = household.answer_reminder_choice(&response, choice, ChoiceAnswer::Ordinal(ordinal))?;
        let expected = selected.filter(|_| elapsed < 30_000 && !other_room && !other_speaker && !remove_selected);
        prop_assert_eq!(outcome, expected.map_or(VoiceCancel::NoMatch, VoiceCancel::Cancelled));
        for original in shared {
            let state = household.reminder(original.id, None)?.expect("retained reminder").state;
            let cancelled = expected == Some(original.id) || (remove_selected && selected == Some(original.id));
            prop_assert_eq!(state, if cancelled { ReminderState::Cancelled } else { ReminderState::Pending });
        }
        prop_assert_eq!(household.reminder(personal.id, Some("p1"))?, Some(personal));
    }

    #[test]
    fn late_actions_are_barred_at_millisecond_precision(
        lateness in prop_oneof![Just(119_999_u64), Just(120_000), Just(120_001), 0_u64..240_000],
    ) {
        let (mut household, clock) = household();
        let reminder = household.create_reminder("apagar luz", None, "living", 5, &[action()])?;
        clock.advance(5 + lateness);
        let notice = household.claim_reminder_notice("living")?.expect("due notice");
        prop_assert_eq!(notice.late, lateness > 120_000);
        prop_assert_eq!(household.claim_reminder_action(reminder.id)?, None);
        prop_assert!(household.finish_reminder_notice(&notice, NoticeOutcome::Delivered)?);
        let claimed = household.claim_reminder_action(reminder.id)?;
        if lateness <= 120_000 {
            let effect = claimed.expect("action inside the grace");
            prop_assert!(household.finish_reminder_action(&effect, ActionOutcome::Succeeded)?);
        } else {
            prop_assert_eq!(claimed, None);
            prop_assert_eq!(household.reminder_actions(reminder.id)?[0].state, ActionState::LateNotExecuted);
        }
        prop_assert_eq!(household.reminder(reminder.id, None)?.expect("retained reminder").state, ReminderState::Completed);
    }
}

#[test]
fn grace_is_rechecked_after_audio_and_between_actions() -> nexa_household::Result<()> {
    let (mut household, clock) = household();
    let reminder = household.create_reminder("luces", None, "living", 0, &[action(), action()])?;
    let notice = household.claim_reminder_notice("living")?.expect("due");
    assert!(!notice.late);
    clock.advance(120_000);
    assert!(household.finish_reminder_notice(&notice, NoticeOutcome::Delivered)?);
    let first = household
        .claim_reminder_action(reminder.id)?
        .expect("within grace");
    assert_eq!(first.position, 0);
    assert!(household.claim_reminder_action(reminder.id)?.is_none());
    clock.advance(1);
    assert!(household.finish_reminder_action(&first, ActionOutcome::Succeeded)?);
    assert!(household.claim_reminder_action(reminder.id)?.is_none());
    let states: Vec<_> = household
        .reminder_actions(reminder.id)?
        .into_iter()
        .map(|action| action.state)
        .collect();
    assert_eq!(
        states,
        [ActionState::Succeeded, ActionState::LateNotExecuted]
    );
    Ok(())
}

#[test]
fn retry_backoff_reopens_after_fallback_exhaustion() -> nexa_household::Result<()> {
    let (mut household, clock) = household();
    household.set_reminder_policy(ReminderPolicy {
        notice_fallback_rounds: 1,
        ..Default::default()
    })?;
    let reminder = household.create_reminder("avisar", None, "living", 1, &[])?;
    let personal = household.create_reminder("personal", Some("p1"), "living", 0, &[])?;
    assert!(household.claim_reminder_notice("living")?.is_none());
    clock.advance(1);
    for (index, delay) in [30_000, 60_000, 900_000, 30_000, 60_000]
        .into_iter()
        .enumerate()
    {
        let claim = household
            .claim_reminder_notice("living")?
            .expect("ready for retry");
        let claimed_at = clock.now();
        let time_reporting_failure = if index == 0 { 5_000 } else { 0 };
        clock.advance(time_reporting_failure);
        assert_eq!(claim.attempt, (index + 1) as i64);
        assert!(household.claim_reminder_notice("living")?.is_none());
        assert!(household.finish_reminder_notice(&claim, NoticeOutcome::NotStarted)?);
        assert!(!household.finish_reminder_notice(&claim, NoticeOutcome::Delivered)?);
        let receipt = household
            .reminder_notice(reminder.id)?
            .expect("durable retry");
        assert_eq!(receipt.next_attempt_at, Some(claimed_at + delay as i64));
        clock.advance(delay - time_reporting_failure - 1);
        assert!(household.claim_reminder_notice("living")?.is_none());
        clock.advance(1);
    }
    let last = household
        .claim_reminder_notice("living")?
        .expect("last retry");
    assert!(household.finish_reminder_notice(&last, NoticeOutcome::NotStarted)?);
    assert_eq!(
        household
            .reminder_notice(reminder.id)?
            .expect("parked receipt")
            .state,
        NoticeState::Parked
    );
    assert_eq!(household.reopen_reminder_notices("cocina")?, 0);
    assert_eq!(household.reopen_reminder_notices("living")?, 1);
    let reopened = household
        .claim_reminder_notice("living")?
        .expect("reopened");
    assert_eq!(reopened.attempt, 7);
    assert!(household.finish_reminder_notice(&reopened, NoticeOutcome::Delivered)?);
    assert!(household.claim_reminder_notice("living")?.is_none());
    assert_eq!(household.pending_reminders(Some("p1"))?, [personal]);
    Ok(())
}

#[test]
fn retries_survive_reopen_and_claims_with_unknown_delivery_are_never_replayed()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let clock = Arc::new(TestClock::default());
    let open = || {
        Ok::<_, Error>(Household::new(
            HouseholdStore::open_in(directory.path())?,
            Arc::clone(&clock),
        ))
    };
    let mut first = open()?;
    let reminder = first.create_reminder("avisar", None, "living", 0, &[action()])?;
    let attempt = first.claim_reminder_notice("living")?.expect("ready");
    assert!(first.finish_reminder_notice(&attempt, NoticeOutcome::NotStarted)?);
    drop(first);
    let mut second = open()?;
    second.recover_reminders()?;
    clock.advance(29_999);
    assert!(second.claim_reminder_notice("living")?.is_none());
    clock.advance(1);
    let other_attempt = second.claim_reminder_notice("living")?.expect("retry");
    assert_eq!(other_attempt.attempt, 2);
    drop(second);
    let mut third = open()?;
    third.recover_reminders()?;
    assert_eq!(
        third.reminder_notice(reminder.id)?.expect("receipt").state,
        NoticeState::Uncertain
    );
    assert_eq!(
        third.reminder_actions(reminder.id)?[0].state,
        ActionState::Skipped
    );
    assert!(third.claim_reminder_notice("living")?.is_none());
    assert!(!third.finish_reminder_notice(&other_attempt, NoticeOutcome::Delivered)?);
    assert_eq!(
        third
            .reminder(reminder.id, None)?
            .expect("retained reminder")
            .state,
        ReminderState::Completed
    );
    Ok(())
}

#[test]
fn cancelling_or_dismissing_prevents_new_reminder_effects() -> nexa_household::Result<()> {
    let (mut household, _) = household();
    let cancelled = household.create_reminder("cancelar", None, "living", 0, &[action()])?;
    let dismissed = household.create_reminder("descartar", None, "living", 0, &[action()])?;
    let admitted = household
        .claim_reminder_notice("living")?
        .expect("first claim");
    assert!(!household.cancel_reminder(cancelled.id, Some("p1"))?);
    assert!(household.cancel_reminder(cancelled.id, None)?);
    assert!(household.finish_reminder_notice(&admitted, NoticeOutcome::Delivered)?);
    assert!(household.claim_reminder_action(cancelled.id)?.is_none());
    assert_eq!(
        household.reminder_actions(cancelled.id)?[0].state,
        ActionState::Cancelled
    );
    assert!(household.dismiss_reminder_notice(dismissed.id)?);
    assert_eq!(
        household
            .reminder_notice(dismissed.id)?
            .expect("dismissed")
            .state,
        NoticeState::Dismissed
    );
    assert!(household.claim_reminder_notice("living")?.is_none());
    Ok(())
}

#[test]
fn routine_occurrence_has_one_claim_across_wal_connections()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    const WORKERS: usize = 8;
    let directory = tempfile::tempdir()?;
    let clock = Arc::new(TestClock::default());
    let mut household = Household::new(
        HouseholdStore::open_in(directory.path())?,
        Arc::clone(&clock),
    );
    let routine = household.create_routine(&routine_spec())?;
    let barrier = Barrier::new(WORKERS);
    let claims = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS)
            .map(|_| {
                let clock = Arc::clone(&clock);
                let directory = directory.path();
                let barrier = &barrier;
                scope.spawn(move || {
                    let store = HouseholdStore::open_in(directory);
                    barrier.wait();
                    Household::new(store?, clock).claim_routine(
                        routine.id,
                        Occurrence::Scheduled(0),
                        Some(routine.revision),
                    )
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().expect("claim worker joined"))
            .collect::<nexa_household::Result<Vec<_>>>()
    })?;
    let run = household
        .running_routines()?
        .pop()
        .expect("one durable run");
    let created = claims
        .iter()
        .filter(|claim| matches!(claim, RunClaim::Created(_)))
        .count();
    assert_eq!(created, 1);
    for claim in claims {
        let (RunClaim::Created(claimed) | RunClaim::Existing(claimed)) = claim;
        assert_eq!(claimed, run);
    }
    Ok(())
}

#[test]
fn routine_keeps_revisioned_snapshot_and_voice_brake_stops_the_next_step()
-> nexa_household::Result<()> {
    let (mut household, clock) = household();
    let spec = routine_spec();
    let routine = household.create_routine(&spec)?;
    let run = created(household.claim_routine(routine.id, Occurrence::Voice("turn-1"), Some(1))?);
    let different = household.create_routine(&RoutineSpec {
        name: "otra".into(),
        ..spec.clone()
    })?;
    assert!(matches!(
        household.claim_routine(different.id, Occurrence::Voice("turn-1"), None),
        Err(Error::Conflict(_))
    ));
    let skipped = created(household.claim_routine(routine.id, Occurrence::Scheduled(0), None)?);
    assert_eq!(skipped.state, RunState::Skipped);
    assert_eq!(
        household.claim_routine(routine.id, Occurrence::Scheduled(0), None)?,
        RunClaim::Existing(skipped)
    );
    let updated = household
        .update_routine(
            routine.id,
            1,
            &RoutineSpec {
                steps: vec![RoutineStep::Home { action: action() }],
                ..spec.clone()
            },
        )?
        .expect("current version");
    assert_eq!(updated.revision, 2);
    assert!(household.update_routine(routine.id, 1, &spec)?.is_none());
    let step = household.claim_routine_step(run.id)?.expect("initial step");
    assert!(household.claim_routine_step(run.id)?.is_none());
    assert!(household.finish_routine_step(&step, StepOutcome::Succeeded)?);
    assert!(!household.finish_routine_step(&step, StepOutcome::Succeeded)?);
    assert!(household.claim_routine_step(run.id)?.is_none());
    clock.advance(1_999);
    assert!(household.claim_routine_step(run.id)?.is_none());
    clock.advance(1);
    let step = household
        .claim_routine_step(run.id)?
        .expect("snapshot retained third step");
    assert_eq!(step.position, 2);
    let brake = household
        .brake_routine_by_voice("BUENAS  NOCHES", VoiceBrake::Disable)?
        .expect("exact scoped match");
    assert_eq!(brake.stopped_run, Some(run.id));
    assert!(brake.disabled);
    assert!(!household.finish_routine_step(&step, StepOutcome::Succeeded)?);
    assert!(household.claim_routine_step(run.id)?.is_none());
    let stopped = household.routine_run(run.id)?.expect("stopped run");
    assert_eq!(stopped.state, RunState::Stopped);
    assert_eq!(
        household.claim_routine(routine.id, Occurrence::Voice("turn-1"), None)?,
        RunClaim::Existing(stopped)
    );
    assert!(matches!(
        household.claim_routine(routine.id, Occurrence::Voice("turn-2"), None),
        Err(Error::Conflict(_))
    ));
    assert!(!household.delete_routine(routine.id, 2)?);
    assert!(household.delete_routine(routine.id, 3)?);
    assert!(household.routine(routine.id)?.is_none());
    assert!(household.routine_run(run.id)?.is_some());
    assert_ne!(household.create_routine(&spec)?.id, routine.id);
    Ok(())
}

#[test]
fn routine_recovery_interrupts_claimed_steps_and_leaves_other_steps_resumable()
-> nexa_household::Result<()> {
    let (mut household, clock) = household();
    let routine = household.create_routine(&RoutineSpec {
        name: "aviso".into(),
        enabled: true,
        steps: vec![RoutineStep::Delay {
            duration: Duration::from_millis(30),
        }],
    })?;
    let run =
        created(household.claim_routine(routine.id, Occurrence::Manual("request-1"), None)?);
    assert!(household.claim_routine_step(run.id)?.is_none());
    assert_eq!(household.recover_routines()?, 0);
    clock.advance(30);
    assert!(household.claim_routine_step(run.id)?.is_none());
    assert_eq!(
        household.routine_run(run.id)?.expect("finished run").state,
        RunState::Succeeded
    );

    let routine = household.create_routine(&routine_spec())?;
    let run =
        created(household.claim_routine(routine.id, Occurrence::Manual("request-2"), None)?);
    let admitted = household
        .claim_routine_step(run.id)?
        .expect("in-flight action");
    assert_eq!(household.recover_routines()?, 1);
    assert_eq!(
        household
            .routine_run(run.id)?
            .expect("interrupted run")
            .state,
        RunState::Interrupted
    );
    assert!(household.claim_routine_step(run.id)?.is_none());
    assert!(!household.finish_routine_step(&admitted, StepOutcome::Succeeded)?);
    Ok(())
}

#[test]
fn list_sync_checks_revision_and_rolls_back_mixed_scope_batches() -> nexa_household::Result<()> {
    let (mut household, _) = household();
    let list = household.create_list("Compras", None, Some("todo.shopping"))?;
    let personal = household.create_list("Compras", Some("p1"), None)?;
    assert!(
        household
            .create_list("Compras", Some("p2"), Some("todo.other"))
            .is_err()
    );
    let private_item = household.add_list_item(personal.id, Some("p1"), "Privado")?;
    assert!(household.list_snapshot(personal.id, None)?.is_none());
    assert!(household.list_snapshot(personal.id, Some("p2"))?.is_none());
    assert!(!household.delete_list_item(list.id, None, private_item.id)?);
    let local = household.add_list_item(list.id, None, "leche")?;
    let remote = TodoChange::Add {
        uid: "uid1".into(),
        text: "café".into(),
        completed: false,
    };
    assert!(matches!(
        household.apply_todo_changes(list.id, list.revision, std::slice::from_ref(&remote)),
        Err(Error::Conflict(_))
    ));
    let snapshot = household
        .list_snapshot(list.id, None)?
        .expect("shared list");
    assert!(matches!(
        household.apply_todo_changes(
            list.id,
            snapshot.list.revision,
            &[remote.clone(), TodoChange::Remove(private_item.id)]
        ),
        Err(Error::NotFound(_))
    ));
    assert_eq!(
        household.list_snapshot(list.id, None)?,
        Some(snapshot.clone())
    );
    let merged = household.apply_todo_changes(list.id, snapshot.list.revision, &[remote])?;
    let update = TodoChange::Update {
        item: local.id,
        uid: "uid2".into(),
        text: "leche entera".into(),
        completed: true,
    };
    let merged = household.apply_todo_changes(list.id, merged.list.revision, &[update])?;
    assert_eq!(merged.items.len(), 2);
    assert_eq!(merged.items[0].ha_uid.as_deref(), Some("uid2"));
    assert!(household.update_list_item(list.id, None, local.id, "leche entera", false)?);
    assert!(household.rename_list(list.id, None, "Almacén")?.is_some());
    assert_eq!(household.clear_list(list.id, None)?, 2);
    assert!(household.delete_list(list.id, None)?);
    assert!(!household.delete_list(personal.id, None)?);
    assert_eq!(
        household
            .list_snapshot(personal.id, Some("p1"))?
            .expect("personal")
            .items,
        [private_item]
    );
    Ok(())
}

struct TodoAdapter;

impl TodoSync for TodoAdapter {
    type Error = std::convert::Infallible;

    async fn reconcile(
        &self,
        entity: &str,
        snapshot: &ListSnapshot,
    ) -> std::result::Result<Vec<TodoChange>, Self::Error> {
        assert_eq!(entity, "todo.shopping");
        assert!(snapshot.items.is_empty());
        Ok(vec![TodoChange::Add {
            uid: "uid1".into(),
            text: "pan".into(),
            completed: true,
        }])
    }
}

#[tokio::test]
async fn worker_syncs_a_shared_list_through_the_todo_adapter()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (mut household, _) = household();
    let list_id = household
        .create_list("Compras", None, Some("todo.shopping"))?
        .id;
    let worker = HouseholdWorker::new(household);
    let synced = worker.sync_todo(list_id, &TodoAdapter).await?;
    assert_eq!(synced.items[0].text, "pan");
    assert!(synced.items[0].completed);
    assert_eq!(
        worker
            .call(move |household| household.list_snapshot(list_id, None))
            .await?,
        Some(synced)
    );
    Ok(())
}

#[test]
fn alert_cooldown_ack_escalation_and_zone_preferences() -> nexa_household::Result<()> {
    let (mut household, clock) = household();
    let intercom = intercom();
    let high = AlertObservation {
        key: "sensor:entrada",
        priority: AlertPriority::High,
        message: "La puerta está abierta",
        lifetime: EpisodeLifetime::Cooldown(ALERT_COOLDOWN),
    };
    let update = household.observe_alert(high)?;
    assert!(update.should_announce());
    let AlertUpdate::Opened(initial) = update else {
        panic!("expected new episode")
    };
    assert!(household.acknowledge_alert(initial.id, initial.revision)?);
    assert!(household.active_alerts()?.is_empty());
    clock.advance(10);
    let update = household.observe_alert(high)?;
    assert!(!update.should_announce());
    assert!(matches!(update, AlertUpdate::Deduplicated(_)));
    let update = household.observe_alert(AlertObservation {
        priority: AlertPriority::Critical,
        ..high
    })?;
    assert!(update.should_announce());
    let AlertUpdate::Escalated(escalated) = update else {
        panic!("expected escalation")
    };
    assert_eq!(escalated.id, initial.id);
    assert_eq!(escalated.acknowledged_at, None);
    assert!(!household.acknowledge_alert(initial.id, initial.revision)?);
    assert_eq!(household.active_alerts()?, [escalated]);
    clock.advance(ALERT_COOLDOWN.as_millis() as u64 - 10);
    let AlertUpdate::Opened(next) = household.observe_alert(high)? else {
        panic!("expected new episode")
    };
    assert_ne!(initial.id, next.id);
    assert_eq!(
        household
            .alert(initial.id)?
            .expect("history retained")
            .resolved_at,
        initial.expires_at
    );
    assert!(household.resolve_alert(next.id)?);
    let latched = AlertObservation {
        lifetime: EpisodeLifetime::UntilResolved,
        ..high
    };
    let AlertUpdate::Opened(latched_start) = household.observe_alert(latched)? else {
        panic!("expected latched episode")
    };
    clock.advance(600_000);
    assert_eq!(household.active_alerts()?[0].id, latched_start.id);
    assert!(matches!(
        household.observe_alert(latched)?,
        AlertUpdate::Deduplicated(_)
    ));
    assert_eq!(
        household
            .alert_preferences()?
            .selected_zones(&intercom, Some("living")),
        ["living"]
    );
    assert!(
        household
            .set_alert_zones(0, Some(&["missing"]), &intercom)
            .is_err()
    );
    let preference = household
        .set_alert_zones(0, Some(&["cocina", "cocina"]), &intercom)?
        .expect("new preference");
    assert_eq!(
        preference.selected_zones(&intercom, Some("living")),
        ["cocina"]
    );
    assert!(household.set_alert_zones(0, None, &intercom)?.is_none());
    assert!(
        household
            .set_alert_zones(preference.revision, Some(&[]), &intercom)?
            .expect("mute")
            .selected_zones(&intercom, Some("living"))
            .is_empty()
    );
    assert_eq!(
        household
            .set_alert_zones(preference.revision + 1, None, &intercom)?
            .expect("reset")
            .selected_zones(&intercom, Some("living")),
        ["living"]
    );
    Ok(())
}

#[derive(Default)]
struct RecordedOutput(Mutex<Vec<String>>);

impl AnnouncementOutput for RecordedOutput {
    async fn play(&self, output: &str, message: &str) -> Playback {
        assert_eq!(message, "A comer");
        self.0
            .lock()
            .expect("recording sink")
            .push(output.to_owned());
        if output == "channel-1" {
            Playback::Confirmed
        } else {
            Playback::NotStarted
        }
    }
}

#[tokio::test]
async fn intercom_routes_by_zone_and_calls_each_physical_output_once() -> nexa_household::Result<()>
{
    let intercom = intercom();
    let output = RecordedOutput::default();
    assert!(
        intercom
            .announce(Targets::Zones(&["living", "missing"]), "A comer", &output)
            .await
            .is_err()
    );
    assert!(output.0.lock().expect("recording sink").is_empty());
    let receipts = intercom.announce(Targets::All, "A comer", &output).await?;
    assert_eq!(
        *output.0.lock().expect("recording sink"),
        ["channel-1", "channel-2"]
    );
    assert_eq!(
        receipts
            .into_iter()
            .map(|receipt| (receipt.zone, receipt.playback))
            .collect::<Vec<_>>(),
        [
            ("living".into(), Playback::Confirmed),
            ("comedor".into(), Playback::Confirmed),
            ("cocina".into(), Playback::NotStarted),
        ]
    );
    let zone = intercom.route(Targets::Zones(&["cocina", "cocina"]))?;
    assert_eq!(zone.len(), 1);
    assert_eq!(zone[0].id, "cocina");
    Ok(())
}

#[tokio::test]
async fn exclusive_worker_recovers_interrupted_household_effects_at_startup()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let store = HouseholdStore::open_in(directory.path())?;
    let mut household = Household::new(store, TestClock::default());
    let reminder_id = household
        .create_reminder("luz", None, "living", 0, &[action()])?
        .id;
    let notice = household.claim_reminder_notice("living")?.expect("ready");
    assert!(household.finish_reminder_notice(&notice, NoticeOutcome::Delivered)?);
    assert!(household.claim_reminder_action(reminder_id)?.is_some());
    let routine = household.create_routine(&routine_spec())?;
    let run = created(household.claim_routine(routine.id, Occurrence::Manual("request"), None)?);
    assert!(household.claim_routine_step(run.id)?.is_some());
    drop(household);

    let worker =
        HouseholdWorker::open_in(directory.path().to_owned(), TestClock::default()).await?;
    let run_id = run.id;
    worker
        .clone()
        .call(move |household| {
            assert_eq!(
                household.reminder_actions(reminder_id)?[0].state,
                ActionState::Uncertain
            );
            assert_eq!(
                household
                    .reminder(reminder_id, None)?
                    .expect("closed")
                    .state,
                ReminderState::Completed
            );
            assert!(household.claim_reminder_action(reminder_id)?.is_none());
            assert_eq!(
                household.routine_run(run_id)?.expect("retained").state,
                RunState::Interrupted
            );
            assert!(household.claim_routine_step(run_id)?.is_none());
            Ok(())
        })
        .await?;
    Ok(())
}

#[test]
fn seconds_grace_remains_inclusive_for_existing_callers() {
    use nexa_household::home_actions_within_grace;

    assert!(home_actions_within_grace(0, 120));
    assert!(home_actions_within_grace(120, 120));
    assert!(!home_actions_within_grace(121, 120));
}
