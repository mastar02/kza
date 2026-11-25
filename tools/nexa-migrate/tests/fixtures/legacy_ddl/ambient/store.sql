-- archive/pre-purge-v2-2026-11 (18a9e8e4): src/ambient/store.py
-- DDL block

CREATE TABLE IF NOT EXISTS utterances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  room_id TEXT NOT NULL,
  t0 REAL NOT NULL,
  t1 REAL NOT NULL,
  text TEXT NOT NULL,
  speaker TEXT NOT NULL DEFAULT 'unknown',
  speaker_confidence REAL NOT NULL DEFAULT 0,
  azimuth REAL,
  azimuth_stability REAL NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'unknown',
  confidence REAL,
  no_speech_prob REAL,
  vad_prob REAL,
  lang TEXT,
  lang_prob REAL,
  lang_ok INTEGER,
  during_tts INTEGER NOT NULL DEFAULT 0,
  distilled INTEGER NOT NULL DEFAULT 0,
  text_empty INTEGER NOT NULL DEFAULT 0,
  audio_path TEXT,
  capture_id TEXT,
  expires_at REAL,
  conversation_id TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_room_time ON utterances(room_id, t0);
CREATE INDEX IF NOT EXISTS idx_utt_distill ON utterances(distilled, source);
CREATE TABLE IF NOT EXISTS pending_audio_deletions (
  path TEXT PRIMARY KEY,
  requested_at REAL NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT
);
CREATE TABLE IF NOT EXISTS capture_conversations (
  id TEXT PRIMARY KEY, room_id TEXT NOT NULL, started_at REAL NOT NULL, last_seen_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audio_write_intents (
  path TEXT PRIMARY KEY, utterance_id INTEGER NOT NULL,
  conversation_id TEXT NOT NULL, expires_at REAL NOT NULL
);

