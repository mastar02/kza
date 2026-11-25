CREATE TABLE schema_meta (
    model_key TEXT PRIMARY KEY,
    vector_kind TEXT NOT NULL UNIQUE CHECK (vector_kind IN ('fact', 'voice')),
    dimensions INTEGER NOT NULL CHECK (dimensions > 0)
) STRICT;

CREATE TABLE facts (
    fact_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1))
) STRICT;
CREATE INDEX facts_conversation ON facts(conversation_id);
CREATE INDEX facts_owner ON facts(owner_user_id, confirmed);
CREATE TABLE fact_sources (
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    PRIMARY KEY (fact_id, conversation_id)
) STRICT;
CREATE INDEX fact_sources_conversation ON fact_sources(conversation_id);
CREATE TABLE fact_dependencies (
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    depends_on_fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, depends_on_fact_id),
    CHECK (fact_id <> depends_on_fact_id)
) STRICT;
CREATE INDEX fact_dependencies_source ON fact_dependencies(depends_on_fact_id);
CREATE TABLE fact_owners (
    fact_id TEXT PRIMARY KEY REFERENCES facts(fact_id) ON DELETE CASCADE,
    person_id TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('authenticated', 'manual', 'voice')),
    revision INTEGER NOT NULL CHECK (revision > 0)
) STRICT;
CREATE INDEX fact_owners_person ON fact_owners(person_id);
CREATE TABLE fact_owner_revisions (
    fact_id TEXT NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK (revision > 0),
    change_kind TEXT NOT NULL CHECK (change_kind IN ('assign', 'unassign')),
    changed_at_unix_ms INTEGER NOT NULL CHECK (changed_at_unix_ms >= 0),
    PRIMARY KEY (fact_id, revision)
) STRICT;

CREATE TABLE conversation_tombstones (
    conversation_id TEXT PRIMARY KEY,
    erased_at_unix_ms INTEGER NOT NULL CHECK (erased_at_unix_ms >= 0)
) STRICT;
CREATE TABLE deletion_ledger_salt_v1 (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    salt BLOB NOT NULL CHECK (length(salt) = 32)
) STRICT;
CREATE TABLE deletion_ledger_v1 (
    seq INTEGER PRIMARY KEY CHECK (seq > 0),
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'conversation', 'retired_user', 'utterance')),
    subject_sha256 BLOB NOT NULL CHECK (length(subject_sha256) = 32),
    at INTEGER NOT NULL CHECK (at >= 0),
    prev_sha256 BLOB NOT NULL CHECK (length(prev_sha256) = 32),
    row_sha256 BLOB NOT NULL CHECK (length(row_sha256) = 32),
    UNIQUE (kind, subject_sha256)
) STRICT;
CREATE TABLE erasure_ops (
    ledger_seq INTEGER PRIMARY KEY REFERENCES deletion_ledger_v1(seq),
    erasure_id TEXT NOT NULL UNIQUE,
    facts_deleted INTEGER NOT NULL CHECK (facts_deleted BETWEEN 0 AND 4294967295),
    vectors_deleted INTEGER NOT NULL CHECK (vectors_deleted BETWEEN 0 AND 4294967295),
    captures_deleted INTEGER NOT NULL CHECK (captures_deleted BETWEEN 0 AND 4294967295)
) STRICT;

CREATE TABLE voice_profiles (
    profile_id TEXT PRIMARY KEY,
    encoder_model_key TEXT NOT NULL REFERENCES schema_meta(model_key),
    revision INTEGER NOT NULL CHECK (revision > 0)
) STRICT;
CREATE TABLE voice_fingerprints (
    capture_id TEXT PRIMARY KEY,
    utterance_id TEXT NOT NULL UNIQUE,
    encoder_model_key TEXT NOT NULL REFERENCES schema_meta(model_key),
    profile_id TEXT REFERENCES voice_profiles(profile_id) ON DELETE SET NULL,
    embedding BLOB NOT NULL,
    captured_at_unix_ms INTEGER NOT NULL CHECK (captured_at_unix_ms >= 0),
    CHECK (length(embedding) = 768)
) STRICT;
CREATE INDEX voice_fingerprints_profile ON voice_fingerprints(profile_id);
CREATE TABLE voice_fingerprint_sources (
    capture_id TEXT NOT NULL REFERENCES voice_fingerprints(capture_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    PRIMARY KEY (capture_id, conversation_id)
) STRICT;
CREATE INDEX voice_fingerprint_sources_conversation ON voice_fingerprint_sources(conversation_id);
CREATE TABLE voice_capture_reviews (
    capture_id TEXT PRIMARY KEY REFERENCES voice_fingerprints(capture_id) ON DELETE CASCADE,
    decision TEXT NOT NULL CHECK (decision IN ('accept', 'reject')),
    reviewed_at_unix_ms INTEGER NOT NULL CHECK (reviewed_at_unix_ms >= 0)
) STRICT;
CREATE TABLE voice_profile_sources (
    profile_id TEXT NOT NULL REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    capture_id TEXT NOT NULL REFERENCES voice_fingerprints(capture_id) ON DELETE CASCADE,
    PRIMARY KEY (profile_id, capture_id)
) STRICT;
CREATE INDEX voice_profile_sources_capture ON voice_profile_sources(capture_id);
CREATE TABLE voice_groups (
    group_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision > 0)
) STRICT;
CREATE TABLE voice_group_members (
    group_id TEXT NOT NULL REFERENCES voice_groups(group_id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, profile_id)
) STRICT;
CREATE TABLE voice_group_requests (
    group_id TEXT NOT NULL REFERENCES voice_groups(group_id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'accepted', 'rejected')),
    PRIMARY KEY (group_id, profile_id)
) STRICT;
CREATE TABLE voice_profile_people (
    profile_id TEXT PRIMARY KEY REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    person_id TEXT,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= 0)
) STRICT;
CREATE INDEX voice_profile_people_person ON voice_profile_people(person_id);
CREATE TABLE voice_attribution_consents (
    person_id TEXT PRIMARY KEY,
    profile_id TEXT REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    profile_revision INTEGER,
    link_revision INTEGER,
    granted INTEGER NOT NULL CHECK (granted IN (0, 1)),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= 0),
    CHECK (
        (granted = 0 AND profile_id IS NULL AND profile_revision IS NULL AND link_revision IS NULL)
        OR (granted = 1 AND profile_id IS NOT NULL AND profile_revision IS NOT NULL
            AND link_revision IS NOT NULL AND profile_revision > 0 AND link_revision > 0)
    )
) STRICT;
CREATE INDEX voice_consent_profile ON voice_attribution_consents(profile_id);
CREATE TABLE voice_attribution_events (
    event_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('match', 'suspend', 'resume')),
    at INTEGER NOT NULL CHECK (at >= 0)
) STRICT;
CREATE TABLE voice_attribution_suspensions (
    profile_id TEXT PRIMARY KEY REFERENCES voice_profiles(profile_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK (revision > 0),
    suspended_at_unix_ms INTEGER NOT NULL CHECK (suspended_at_unix_ms >= 0)
) STRICT;
CREATE TABLE voice_attribution_control (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= 0)
) STRICT;
CREATE TABLE voice_verification_attempts (
    person_id TEXT NOT NULL,
    attempted_at_unix_ms INTEGER NOT NULL CHECK (attempted_at_unix_ms >= 0),
    outcome TEXT NOT NULL CHECK (outcome IN ('passed', 'failed')),
    PRIMARY KEY (person_id, attempted_at_unix_ms)
) STRICT;
