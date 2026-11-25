-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/reminders/reminder_store.py
-- DDL block

                    CREATE TABLE IF NOT EXISTS reminders (
                        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                        trigger_at REAL NOT NULL, created_at REAL NOT NULL,
                        recurrence TEXT, recurrence_end REAL, ha_actions TEXT,
                        state TEXT NOT NULL DEFAULT 'active', last_fired_at REAL,
                        fire_count INTEGER NOT NULL DEFAULT 0,
                        scope TEXT NOT NULL DEFAULT 'private' CHECK (scope IN ('private', 'shared')),
                        delivery_room TEXT
                    )
                
-- DDL block

        CREATE TABLE IF NOT EXISTS timers (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            actor TEXT NOT NULL,
            delivery_room TEXT,
            state TEXT NOT NULL CHECK (state IN ('running', 'paused', 'completed', 'cancelled')),
            duration_s REAL NOT NULL,
            ends_at REAL,
            remaining_s REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1
        )
    
-- DDL block

                    CREATE TABLE IF NOT EXISTS reminder_home_actions (
                        request_id TEXT PRIMARY KEY, body TEXT NOT NULL,
                        status TEXT NOT NULL, reason TEXT
                    )
                
-- DDL block

        CREATE TABLE IF NOT EXISTS alert_preferences (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1
        )
    
