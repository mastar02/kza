ALTER TABLE routines ADD COLUMN updated_at_ms INTEGER NOT NULL DEFAULT 0;
ALTER TABLE routines ADD COLUMN schedule_json TEXT CHECK (json_valid(schedule_json));
ALTER TABLE routines ADD COLUMN next_due_at_ms INTEGER
    CHECK (next_due_at_ms IS NULL OR (schedule_json IS NOT NULL AND enabled = 1 AND deleted_at IS NULL));
CREATE INDEX routines_due ON routines(next_due_at_ms, id)
    WHERE enabled = 1 AND deleted_at IS NULL AND next_due_at_ms IS NOT NULL;

ALTER TABLE reminders ADD COLUMN revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0);
UPDATE reminders SET revision = 2 WHERE state IN ('completed', 'cancelled');
