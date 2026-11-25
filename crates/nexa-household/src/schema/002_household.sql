CREATE TABLE lists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    owner TEXT,
    ha_entity TEXT UNIQUE,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    CHECK (owner IS NULL OR ha_entity IS NULL)
) STRICT;
CREATE UNIQUE INDEX lists_shared_name ON lists(name_key) WHERE owner IS NULL;
CREATE UNIQUE INDEX lists_personal_name ON lists(owner, name_key) WHERE owner IS NOT NULL;

CREATE TABLE list_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    list_id INTEGER NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    ha_uid TEXT,

    completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    UNIQUE (list_id, ha_uid)
) STRICT;
CREATE INDEX list_items_by_list ON list_items(list_id, id);

CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_key TEXT NOT NULL,
    priority INTEGER NOT NULL CHECK (priority BETWEEN 0 AND 3),
    message TEXT NOT NULL,
    opened_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    expires_at INTEGER,
    acknowledged_at INTEGER,
    resolved_at INTEGER,
    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0)
) STRICT;
CREATE UNIQUE INDEX alerts_open_condition ON alerts(condition_key) WHERE resolved_at IS NULL;

CREATE TABLE alert_preferences (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    zones TEXT NOT NULL CHECK (json_valid(zones)),
    revision INTEGER NOT NULL CHECK (revision > 0)
) STRICT;
