-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/lists/list_store.py
-- DDL block

                    CREATE TABLE IF NOT EXISTS lists (
                        id TEXT PRIMARY KEY, name TEXT NOT NULL, owner_type TEXT NOT NULL,
                        owner_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                    )
                
-- DDL block

                    CREATE TABLE IF NOT EXISTS list_items (
                        id TEXT PRIMARY KEY, list_id TEXT NOT NULL, text TEXT NOT NULL,
                        completed INTEGER NOT NULL DEFAULT 0, added_by TEXT,
                        created_at REAL NOT NULL, completed_at REAL,
                        FOREIGN KEY (list_id) REFERENCES lists(id)
                    )
                
