CREATE TABLE ambient.capture_conversations (
    conversation_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    owner_user_id TEXT,
    started_at_unix_ms INTEGER NOT NULL CHECK (started_at_unix_ms >= 0)
) STRICT;
CREATE INDEX ambient.capture_conversations_owner ON capture_conversations(owner_user_id);
CREATE TABLE ambient.utterances (
    utterance_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES capture_conversations(conversation_id) ON DELETE CASCADE,
    room_id TEXT NOT NULL,
    transcript_preview TEXT NOT NULL,
    captured_at_unix_ms INTEGER NOT NULL CHECK (captured_at_unix_ms >= 0),
    audio_path TEXT
) STRICT;
CREATE INDEX ambient.utterances_conversation ON utterances(conversation_id);
CREATE TABLE ambient.pending_audio_deletions (
    path TEXT PRIMARY KEY,
    requested_at_unix_ms INTEGER NOT NULL CHECK (requested_at_unix_ms >= 0)
) STRICT;
CREATE TABLE ambient.audio_write_intents (
    path TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES capture_conversations(conversation_id) ON DELETE CASCADE,
    utterance_id TEXT NOT NULL,
    expires_at_unix_ms INTEGER NOT NULL CHECK (expires_at_unix_ms >= 0)
) STRICT;
CREATE INDEX ambient.audio_write_intents_conversation ON audio_write_intents(conversation_id);
CREATE TABLE ambient.ambient_runtime_control (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    paused INTEGER NOT NULL CHECK (paused IN (0, 1)),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= 0)
) STRICT;
