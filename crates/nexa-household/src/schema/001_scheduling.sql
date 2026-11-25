CREATE TABLE timers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    room TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('running', 'paused', 'completed', 'cancelled')),
    deadline INTEGER,
    remaining_ms INTEGER CHECK (remaining_ms >= 0),
    ring_until INTEGER,
    CHECK ((state = 'running') = (deadline IS NOT NULL)),
    CHECK ((state = 'paused') = (remaining_ms IS NOT NULL)),
    CHECK (ring_until IS NULL OR state = 'completed')
) STRICT;
CREATE INDEX timers_due ON timers(deadline) WHERE state = 'running';
CREATE UNIQUE INDEX timers_active_name ON timers(room, name_key)
    WHERE state IN ('running', 'paused');

CREATE TABLE reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    owner TEXT,
    room TEXT NOT NULL,
    due_at INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'completed', 'cancelled'))
) STRICT;
CREATE INDEX reminders_pending ON reminders(due_at, id) WHERE state = 'pending';

CREATE TABLE reminder_home_actions (
    reminder_id INTEGER NOT NULL REFERENCES reminders(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    command TEXT NOT NULL CHECK (json_valid(command)),
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'claimed', 'succeeded', 'failed', 'uncertain', 'late_not_executed', 'cancelled', 'skipped'
    )),
    PRIMARY KEY (reminder_id, position)
) STRICT;

CREATE TABLE reminder_notices (
    reminder_id INTEGER PRIMARY KEY REFERENCES reminders(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN (
        'pending', 'claimed', 'delivered', 'dismissed', 'uncertain', 'cancelled', 'parked'
    )),
    next_attempt_at INTEGER,
    claimed_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    round_attempts INTEGER NOT NULL DEFAULT 0 CHECK (round_attempts >= 0),
    fallback_rounds INTEGER NOT NULL DEFAULT 0 CHECK (fallback_rounds >= 0),
    CHECK ((status = 'pending') = (next_attempt_at IS NOT NULL)),
    CHECK ((status = 'claimed') = (claimed_at IS NOT NULL))
) STRICT;
CREATE INDEX reminder_notices_due ON reminder_notices(next_attempt_at) WHERE status = 'pending';

CREATE TABLE routines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    revision INTEGER NOT NULL CHECK (revision > 0),
    steps TEXT NOT NULL CHECK (json_valid(steps)),
    deleted_at INTEGER
) STRICT;

CREATE UNIQUE INDEX routines_live_name ON routines(name_key) WHERE deleted_at IS NULL;

CREATE TABLE routine_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    routine_id INTEGER NOT NULL REFERENCES routines(id),
    occurrence_key TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('scheduled', 'voice', 'manual')),
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    status TEXT NOT NULL CHECK (status IN (
        'running', 'succeeded', 'failed', 'stopped', 'skipped', 'interrupted'
    )),
    steps TEXT NOT NULL CHECK (json_valid(steps)),
    next_step INTEGER NOT NULL DEFAULT 0 CHECK (next_step >= 0),
    step_claimed INTEGER NOT NULL DEFAULT 0 CHECK (step_claimed IN (0, 1)),
    step_due_at INTEGER,
    UNIQUE (routine_id, occurrence_key),
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (step_claimed = 0 OR (status = 'running' AND step_due_at IS NULL)),
    CHECK (step_due_at IS NULL OR status = 'running')
) STRICT;
CREATE UNIQUE INDEX routine_one_active_run ON routine_runs(routine_id) WHERE status = 'running';
CREATE UNIQUE INDEX routines_voice_occurrence ON routine_runs(occurrence_key) WHERE source = 'voice';
