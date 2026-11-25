//! Durable reminder notice and home-action admission, with room-bound voice choices.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, Row, params};

use crate::{
    Clock, Error, HomeAction, Household, Result, Timestamp, json_column, millis, query_all,
    required,
};

pub const LATE_ACTION_GRACE: Duration = Duration::from_secs(120);
pub const VOICE_CHOICE_WINDOW: Duration = Duration::from_secs(30);
const MAX_VOICE_CHOICES: usize = 3;
static NEXT_CHOICE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReminderPolicy {
    pub late_action_grace: Duration,
    pub voice_choice_window: Duration,
    pub notice_max_attempts: u32,
    pub notice_retry_gap: Duration,
    pub notice_fallback_gap: Duration,
    pub notice_fallback_rounds: u32,
}

impl Default for ReminderPolicy {
    fn default() -> Self {
        Self {
            late_action_grace: LATE_ACTION_GRACE,
            voice_choice_window: VOICE_CHOICE_WINDOW,
            notice_max_attempts: 3,
            notice_retry_gap: Duration::from_secs(30),
            notice_fallback_gap: Duration::from_secs(900),
            notice_fallback_rounds: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReminderId(pub i64);

sql_enum! {
    pub enum ReminderState {
        Pending => "pending",
        Completed => "completed",
        Cancelled => "cancelled",
    }
}

sql_enum! {
    pub enum NoticeState {
        Pending => "pending",
        Claimed => "claimed",
        Delivered => "delivered",
        Dismissed => "dismissed",
        Uncertain => "uncertain",
        Cancelled => "cancelled",
        Parked => "parked",
    }
}

sql_enum! {
    pub enum ActionState {
        Pending => "pending",
        Claimed => "claimed",
        Succeeded => "succeeded",
        Failed => "failed",
        Uncertain => "uncertain",
        LateNotExecuted => "late_not_executed",
        Cancelled => "cancelled",
        Skipped => "skipped",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reminder {
    pub id: ReminderId,
    pub message: String,
    /// A person's ID for a personal reminder. Shared reminders have no owner.
    pub owner: Option<String>,
    pub room: String,
    pub due_at: Timestamp,
    pub state: ReminderState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeClaim {
    pub reminder: Reminder,
    pub attempt: i64,
    pub late: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoticeReceipt {
    pub state: NoticeState,
    pub attempts: i64,
    pub next_attempt_at: Option<Timestamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeOutcome {
    Delivered,
    /// The audio adapter confirmed playback never started.
    NotStarted,
    /// Playback was interrupted or its outcome could not be verified.
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReminderAction {
    pub reminder: ReminderId,
    pub position: i64,
    pub command: HomeAction,
    pub state: ActionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionOutcome {
    Succeeded,
    Failed,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VoiceContext {
    pub speaker: String,
    pub room: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChoiceId(u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceCancel {
    NoMatch,
    Cancelled(ReminderId),
    Choose {
        id: ChoiceId,
        candidates: Vec<Reminder>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoiceAnswer {
    /// A disambiguated ordinal from NLU, using one-based numbering.
    Ordinal(usize),
    Other,
}

#[derive(Debug)]
struct Choice {
    id: ChoiceId,
    expires_at: Duration,
    candidates: Vec<ReminderId>,
}

#[derive(Debug, Default)]
pub(crate) struct VoiceChoices {
    pending: HashMap<VoiceContext, Choice>,
}

const COLUMNS: &str = "id, message, owner, room, due_at, state";

fn read(row: &Row<'_>) -> rusqlite::Result<Reminder> {
    Ok(Reminder {
        id: ReminderId(row.get(0)?),
        message: row.get(1)?,
        owner: row.get(2)?,
        room: row.get(3)?,
        due_at: row.get(4)?,
        state: row.get(5)?,
    })
}

fn read_action(row: &Row<'_>) -> rusqlite::Result<ReminderAction> {
    Ok(ReminderAction {
        reminder: ReminderId(row.get(0)?),
        position: row.get(1)?,
        command: json_column(row, 2)?,
        state: row.get(3)?,
    })
}

fn cancel(db: &Connection, id: ReminderId, owner: Option<&str>) -> Result<bool> {
    let changed = db.execute(
        "UPDATE reminders SET state = 'cancelled' WHERE id = ?1 AND owner IS ?2 AND state = 'pending'",
        params![id.0, owner],
    )? == 1;
    if changed {
        db.execute(
            "UPDATE reminder_notices SET status = 'cancelled', next_attempt_at = NULL
             WHERE reminder_id = ?1 AND status IN ('pending', 'parked')",
            [id.0],
        )?;
        db.execute(
            "UPDATE reminder_home_actions SET status = 'cancelled'
             WHERE reminder_id = ?1 AND status = 'pending'",
            [id.0],
        )?;
    }
    Ok(changed)
}

fn bar_late_actions(db: &Connection, id: ReminderId) -> Result<()> {
    db.execute(
        "UPDATE reminder_home_actions SET status = 'late_not_executed'
         WHERE reminder_id = ?1 AND status = 'pending'",
        [id.0],
    )?;
    Ok(())
}

fn skip_actions(db: &Connection, id: ReminderId) -> Result<()> {
    db.execute(
        "UPDATE reminder_home_actions SET status = 'skipped'
         WHERE reminder_id = ?1 AND status = 'pending'",
        [id.0],
    )?;
    Ok(())
}

fn close_if_done(db: &Connection, id: ReminderId) -> Result<()> {
    db.execute(
        "UPDATE reminders SET state = 'completed'
         WHERE id = ?1 AND state = 'pending'
         AND EXISTS (SELECT 1 FROM reminder_notices WHERE reminder_id = ?1
                     AND status IN ('delivered', 'dismissed', 'uncertain'))
         AND NOT EXISTS (SELECT 1 FROM reminder_home_actions WHERE reminder_id = ?1
                         AND status IN ('pending', 'claimed'))",
        [id.0],
    )?;
    Ok(())
}

fn is_late(now: Timestamp, due_at: Timestamp, policy: ReminderPolicy) -> bool {
    now.saturating_sub(due_at) > millis(policy.late_action_grace)
}

fn words(text: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            word.to_lowercase()
                .replace('á', "a")
                .replace('é', "e")
                .replace('í', "i")
                .replace('ó', "o")
                .replace(['ú', 'ü'], "u")
        })
        .filter(|word| {
            !matches!(
                word.as_str(),
                "de" | "del"
                    | "el"
                    | "la"
                    | "las"
                    | "los"
                    | "lo"
                    | "a"
                    | "al"
                    | "para"
                    | "y"
                    | "por"
                    | "en"
                    | "con"
                    | "que"
                    | "un"
            )
        })
        .collect()
}

fn voice_candidates(shared: Vec<Reminder>, spoken: &BTreeSet<String>) -> Vec<Reminder> {
    let indexed: Vec<_> = shared
        .into_iter()
        .map(|reminder| (words(&reminder.message), reminder))
        .collect();
    let known: Vec<_> = spoken
        .iter()
        .filter(|word| indexed.iter().any(|(text, _)| text.contains(*word)))
        .collect();
    if !spoken.is_empty() && known.is_empty() {
        return Vec::new();
    }
    indexed
        .into_iter()
        .filter(|(text, _)| known.iter().all(|word| text.contains(*word)))
        .map(|(_, reminder)| reminder)
        .take(MAX_VOICE_CHOICES)
        .collect()
}

impl<C: Clock> Household<C> {
    pub fn set_reminder_policy(&mut self, policy: ReminderPolicy) -> Result<()> {
        if policy.notice_max_attempts == 0
            || millis(policy.notice_retry_gap) == 0
            || millis(policy.notice_fallback_gap) == 0
            || policy.voice_choice_window.is_zero()
        {
            return Err(Error::InvalidInput(
                "invalid reminder retry or choice policy",
            ));
        }
        self.reminder_policy = policy;
        Ok(())
    }

    pub fn create_reminder(
        &mut self,
        message: &str,
        owner: Option<&str>,
        room: &str,
        due_at: Timestamp,
        actions: &[HomeAction],
    ) -> Result<Reminder> {
        required(message, "reminder message is required")?;
        required(room, "reminder room is required")?;
        if let Some(owner) = owner {
            required(owner, "personal reminder owner is required")?;
            if !actions.is_empty() {
                return Err(Error::InvalidInput(
                    "home actions require a shared reminder",
                ));
            }
        }
        for action in actions {
            action.validate()?;
        }
        self.transaction(|db| {
            let reminder = db.query_row(
                &format!(
                    "INSERT INTO reminders(message, owner, room, due_at, state)
                          VALUES (?1, ?2, ?3, ?4, 'pending') RETURNING {COLUMNS}"
                ),
                params![message.trim(), owner, room, due_at],
                read,
            )?;
            if owner.is_none() {
                db.execute(
                    "INSERT INTO reminder_notices(reminder_id, status, next_attempt_at)
                     VALUES (?1, 'pending', ?2)",
                    params![reminder.id.0, due_at],
                )?;
            }
            for (position, action) in actions.iter().enumerate() {
                db.execute(
                    "INSERT INTO reminder_home_actions(reminder_id, position, command, status)
                     VALUES (?1, ?2, ?3, 'pending')",
                    params![reminder.id.0, position, serde_json::to_string(action)?],
                )?;
            }
            Ok(reminder)
        })
    }

    pub fn reminder(&self, id: ReminderId, owner: Option<&str>) -> Result<Option<Reminder>> {
        Ok(self
            .db()
            .query_row(
                &format!("SELECT {COLUMNS} FROM reminders WHERE id = ?1 AND owner IS ?2"),
                params![id.0, owner],
                read,
            )
            .optional()?)
    }

    pub fn pending_reminders(&self, owner: Option<&str>) -> Result<Vec<Reminder>> {
        query_all(
            self.db(),
            &format!(
                "SELECT {COLUMNS} FROM reminders WHERE owner IS ?1 AND state = 'pending'
                      ORDER BY due_at, id"
            ),
            [owner],
            read,
        )
    }

    pub fn cancel_reminder(&mut self, id: ReminderId, owner: Option<&str>) -> Result<bool> {
        self.transaction(|db| cancel(db, id, owner))
    }

    /// Match parsed content words against shared reminders. Time words parsed
    /// as a separate reference must be excluded by the voice adapter.
    pub fn begin_voice_cancel(
        &mut self,
        context: &VoiceContext,
        content: &str,
    ) -> Result<VoiceCancel> {
        required(&context.speaker, "voice choice requires a speaker")?;
        required(&context.room, "voice choice requires a room")?;
        let now = self.clock.monotonic();
        self.voice_choices
            .pending
            .retain(|key, choice| key != context && choice.expires_at > now);

        let spoken = words(content);
        let (result, candidates) = self.transaction(|db| {
            let shared = query_all(
                db,
                &format!(
                    "SELECT {COLUMNS} FROM reminders WHERE owner IS NULL AND state = 'pending'
                          ORDER BY due_at, id"
                ),
                [],
                read,
            )?;
            let candidates = voice_candidates(shared, &spoken);
            match candidates.as_slice() {
                [] => Ok((Some(VoiceCancel::NoMatch), candidates)),
                [only] if !spoken.is_empty() && spoken.is_subset(&words(&only.message)) => {
                    let result = if cancel(db, only.id, None)? {
                        VoiceCancel::Cancelled(only.id)
                    } else {
                        VoiceCancel::NoMatch
                    };
                    Ok((Some(result), candidates))
                }
                _ => Ok((None, candidates)),
            }
        })?;
        if let Some(result) = result {
            return Ok(result);
        }
        let sequence = NEXT_CHOICE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::Conflict("voice choice IDs exhausted"))?;
        let id = ChoiceId(sequence);
        self.voice_choices.pending.insert(
            context.clone(),
            Choice {
                id,
                expires_at: self
                    .clock
                    .monotonic()
                    .saturating_add(self.reminder_policy.voice_choice_window),
                candidates: candidates.iter().map(|reminder| reminder.id).collect(),
            },
        );
        Ok(VoiceCancel::Choose { id, candidates })
    }

    /// Consume the current choice for this speaker and room.
    pub fn answer_reminder_choice(
        &mut self,
        context: &VoiceContext,
        id: ChoiceId,
        answer: ChoiceAnswer,
    ) -> Result<VoiceCancel> {
        let Some(choice) = self.voice_choices.pending.get(context) else {
            return Ok(VoiceCancel::NoMatch);
        };
        if choice.id != id {
            return Ok(VoiceCancel::NoMatch);
        }
        let Some(choice) = self.voice_choices.pending.remove(context) else {
            return Ok(VoiceCancel::NoMatch);
        };
        if choice.expires_at <= self.clock.monotonic() {
            return Ok(VoiceCancel::NoMatch);
        }
        if let ChoiceAnswer::Ordinal(position) = answer
            && let Some(selected) = position
                .checked_sub(1)
                .and_then(|index| choice.candidates.get(index))
            && self.cancel_reminder(*selected, None)?
        {
            return Ok(VoiceCancel::Cancelled(*selected));
        }
        Ok(VoiceCancel::NoMatch)
    }

    pub fn reminder_notice(&self, id: ReminderId) -> Result<Option<NoticeReceipt>> {
        Ok(self.db().query_row(
            "SELECT status, attempts, next_attempt_at FROM reminder_notices WHERE reminder_id = ?1",
            [id.0], |row| Ok(NoticeReceipt { state: row.get(0)?, attempts: row.get(1)?, next_attempt_at: row.get(2)? }),
        ).optional()?)
    }

    /// Claim the next deliverable shared reminder for a ready output room.
    pub fn claim_reminder_notice(&mut self, room: &str) -> Result<Option<NoticeClaim>> {
        let policy = self.reminder_policy;
        let clock = &self.clock;
        let tx = self.store.immediate()?;
        let now = clock.now();
        let reminder = tx
            .query_row(
                &format!(
                    "SELECT {COLUMNS} FROM reminders WHERE room = ?1 AND state = 'pending'
                      AND id IN (SELECT reminder_id FROM reminder_notices
                                 WHERE status = 'pending' AND next_attempt_at <= ?2)
                      ORDER BY due_at, id LIMIT 1"
                ),
                params![room, now],
                read,
            )
            .optional()?;
        let Some(reminder) = reminder else {
            return Ok(None);
        };
        let late = is_late(now, reminder.due_at, policy);
        if late {
            bar_late_actions(&tx, reminder.id)?;
        }
        let attempt = tx.query_row(
            "UPDATE reminder_notices SET status = 'claimed', next_attempt_at = NULL, claimed_at = ?3,
                 attempts = attempts + 1,
                 fallback_rounds = fallback_rounds + (round_attempts >= ?2),
                 round_attempts = CASE WHEN round_attempts >= ?2 THEN 1 ELSE round_attempts + 1 END
             WHERE reminder_id = ?1 RETURNING attempts",
            params![reminder.id.0, policy.notice_max_attempts, now],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(Some(NoticeClaim {
            reminder,
            attempt,
            late,
        }))
    }

    pub fn finish_reminder_notice(
        &mut self,
        claim: &NoticeClaim,
        outcome: NoticeOutcome,
    ) -> Result<bool> {
        let now = self.clock.now();
        let policy = self.reminder_policy;
        let id = claim.reminder.id;
        self.transaction(|db| {
            let receipt: Option<(u32, u32, Timestamp, ReminderState, Timestamp)> = db
                .query_row(
                    "SELECT n.round_attempts, n.fallback_rounds, n.claimed_at, r.state, r.due_at
                 FROM reminder_notices n JOIN reminders r ON r.id = n.reminder_id
                 WHERE n.reminder_id = ?1 AND n.attempts = ?2 AND n.status = 'claimed'",
                    params![id.0, claim.attempt],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .optional()?;
            let Some((round_attempts, fallback_rounds, claimed_at, reminder_state, due_at)) =
                receipt
            else {
                return Ok(false);
            };
            if reminder_state == ReminderState::Pending && is_late(now, due_at, policy) {
                bar_late_actions(db, id)?;
            }
            let (status, next) = match outcome {
                NoticeOutcome::Delivered => (NoticeState::Delivered, None),
                NoticeOutcome::Uncertain => (NoticeState::Uncertain, None),
                NoticeOutcome::NotStarted if reminder_state == ReminderState::Cancelled => {
                    (NoticeState::Cancelled, None)
                }
                NoticeOutcome::NotStarted => {
                    notice_retry(claimed_at, policy, round_attempts, fallback_rounds)
                }
            };
            db.execute(
                "UPDATE reminder_notices SET status = ?2, next_attempt_at = ?3, claimed_at = NULL
                 WHERE reminder_id = ?1",
                params![id.0, status, next],
            )?;
            if status == NoticeState::Uncertain {
                skip_actions(db, id)?;
            }
            close_if_done(db, id)?;
            Ok(true)
        })
    }

    /// Signal that the output became free. Exhausted rounds can start again.
    pub fn reopen_reminder_notices(&mut self, room: &str) -> Result<usize> {
        let now = self.clock.now();
        Ok(self.db().execute(
            "UPDATE reminder_notices SET status = 'pending', round_attempts = 0, next_attempt_at = ?2
             WHERE (status = 'parked' OR (status = 'pending' AND round_attempts >= ?3))
             AND reminder_id IN (SELECT id FROM reminders
                                 WHERE room = ?1 AND state = 'pending' AND due_at <= ?2)",
            params![room, now, self.reminder_policy.notice_max_attempts],
        )?)
    }

    pub fn dismiss_reminder_notice(&mut self, id: ReminderId) -> Result<bool> {
        self.transaction(|db| {
            let changed = db.execute(
                "UPDATE reminder_notices SET status = 'dismissed', next_attempt_at = NULL
                 WHERE reminder_id = ?1 AND status IN ('pending', 'parked')",
                [id.0],
            )? == 1;
            if changed {
                skip_actions(db, id)?;
                close_if_done(db, id)?;
            }
            Ok(changed)
        })
    }

    pub fn reminder_actions(&self, id: ReminderId) -> Result<Vec<ReminderAction>> {
        query_all(
            self.db(),
            "SELECT reminder_id, position, command, status FROM reminder_home_actions
             WHERE reminder_id = ?1 ORDER BY position",
            [id.0],
            read_action,
        )
    }

    /// Return the next action after confirmed audio. Recheck lateness for each action.
    pub fn claim_reminder_action(&mut self, id: ReminderId) -> Result<Option<ReminderAction>> {
        let policy = self.reminder_policy;
        let tx = self.store.immediate()?;
        let now = self.clock.now();
        let due_at: Option<Timestamp> = tx
            .query_row(
                "SELECT r.due_at FROM reminders r JOIN reminder_notices n ON n.reminder_id = r.id
             WHERE r.id = ?1 AND r.state = 'pending' AND n.status = 'delivered'",
                [id.0],
                |row| row.get(0),
            )
            .optional()?;
        let Some(due_at) = due_at else {
            return Ok(None);
        };
        if is_late(now, due_at, policy) {
            bar_late_actions(&tx, id)?;
            close_if_done(&tx, id)?;
            tx.commit()?;
            return Ok(None);
        }
        let action = tx.query_row(
            "UPDATE reminder_home_actions SET status = 'claimed' WHERE reminder_id = ?1 AND status = 'pending'
             AND position = (SELECT min(position) FROM reminder_home_actions WHERE reminder_id = ?1
                             AND status <> 'succeeded')
             RETURNING reminder_id, position, command, status",
            [id.0], read_action,
        ).optional()?;
        tx.commit()?;
        Ok(action)
    }

    pub fn finish_reminder_action(
        &mut self,
        action: &ReminderAction,
        outcome: ActionOutcome,
    ) -> Result<bool> {
        self.transaction(|db| {
            let status = match outcome {
                ActionOutcome::Succeeded => ActionState::Succeeded,
                ActionOutcome::Failed => ActionState::Failed,
                ActionOutcome::Uncertain => ActionState::Uncertain,
            };
            let changed = db.execute(
                "UPDATE reminder_home_actions SET status = ?3
                 WHERE reminder_id = ?1 AND position = ?2 AND status = 'claimed'",
                params![action.reminder.0, action.position, status],
            )? == 1;
            if changed {
                if outcome != ActionOutcome::Succeeded {
                    skip_actions(db, action.reminder)?;
                }
                close_if_done(db, action.reminder)?;
            }
            Ok(changed)
        })
    }

    /// Startup recovery for the exclusive owner, after all previous workers
    /// have exited. Unverified effects are finalized without replay.
    pub fn recover_reminders(&mut self) -> Result<()> {
        let now = self.clock.now();
        self.transaction(|db| {
            let mut affected = query_all(
                db,
                "UPDATE reminder_notices SET status = 'uncertain', claimed_at = NULL WHERE status = 'claimed'
                 RETURNING reminder_id",
                [],
                |row| row.get::<_, i64>(0).map(ReminderId),
            )?;
            affected.extend(query_all(
                db,
                "UPDATE reminder_home_actions SET status = 'uncertain' WHERE status = 'claimed'
                 RETURNING reminder_id",
                [],
                |row| row.get::<_, i64>(0).map(ReminderId),
            )?);
            for id in affected {
                skip_actions(db, id)?;
                close_if_done(db, id)?;
            }
            db.execute(
                "UPDATE reminder_notices SET status = 'pending', round_attempts = 0, next_attempt_at = ?1
                 WHERE status = 'parked' AND reminder_id IN
                 (SELECT id FROM reminders WHERE state = 'pending' AND due_at <= ?1)",
                [now],
            )?;
            Ok(())
        })
    }

    /// Pending actions whose announcement was already confirmed, including
    /// work to resume after restarting between two completed steps.
    pub fn reminders_with_delivered_actions(&self) -> Result<Vec<ReminderId>> {
        query_all(
            self.db(),
            "SELECT r.id FROM reminders r JOIN reminder_notices n ON n.reminder_id = r.id
             WHERE r.state = 'pending' AND n.status = 'delivered' AND EXISTS
             (SELECT 1 FROM reminder_home_actions a WHERE a.reminder_id = r.id AND a.status = 'pending')
             ORDER BY r.due_at, r.id",
            [],
            |row| row.get::<_, i64>(0).map(ReminderId),
        )
    }
}

fn notice_retry(
    now: Timestamp,
    policy: ReminderPolicy,
    round_attempts: u32,
    fallback_rounds: u32,
) -> (NoticeState, Option<Timestamp>) {
    let delay = if round_attempts < policy.notice_max_attempts {
        millis(policy.notice_retry_gap)
            .saturating_mul(1_i64 << round_attempts.saturating_sub(1).min(62))
    } else if fallback_rounds < policy.notice_fallback_rounds {
        millis(policy.notice_fallback_gap)
    } else {
        return (NoticeState::Parked, None);
    };
    (NoticeState::Pending, Some(now.saturating_add(delay)))
}
