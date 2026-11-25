"""Import legacy fact, capture and voice provenance into the Nexa memory pair."""

import hashlib
import math
import sqlite3
import struct
from collections import defaultdict
from dataclasses import dataclass

from privacy import Privacy, RETIRED, TOMBSTONED
from storage import Catalog, FACT_MODEL, VOICE_MODEL, decode, identifier, millis, require

FACT_TABLES = {"facts", "fact_sources", "fact_dependencies", "fact_owners", "fact_owner_revisions"}
VOICE_TABLES = {
    "voice_profiles", "voice_profile_people", "voice_fingerprints", "voice_fingerprint_sources",
    "voice_profile_sources", "voice_capture_reviews", "voice_groups", "voice_group_members",
    "voice_group_requests", "voice_attribution_consents", "voice_attribution_control",
    "voice_attribution_events", "voice_attribution_suspensions", "voice_verification_attempts",
    "voice_review_requests",
}
TABLES = FACT_TABLES | VOICE_TABLES | {
    "capture_conversations", "utterances", "pending_audio_deletions", "audio_write_intents",
    "ambient_runtime_control", "conversation_tombstones", "deletion_ledger_v1",
    "deletion_ledger_salt_v1", "erasure_ops",
}
LEGACY_ENCODER = "enc_" + hashlib.sha256(
    b'["speechbrain/spkrec-ecapa-voxceleb", "speechbrain", 16000]'
).hexdigest()
OWNER_METHODS = {"person_session": "authenticated", "operator_assignment": "manual",
                 "voice_proposal": "voice", "authenticated": "authenticated", "manual": "manual",
                 "voice": "voice"}
OWNER_CHANGES = {
    "assigned": "assign", "reassigned": "assign", "owner_returned": "unassign",
    "operator_unassigned": "unassign", "voice_proposed": "assign", "owner_confirmed": "assign",
    "operator_confirmed": "assign", "assign": "assign", "unassign": "unassign",
}


def when(row: dict, native: str, legacy: str) -> int:
    if native not in row:
        return millis(row[legacy])
    value = row[native]
    require(type(value) is int and value >= 0, "invalid source time in milliseconds")
    return value


def people(row: dict) -> tuple:
    return row.get("owner_user_id"), row.get("owner_id"), row.get("owner_person_id"), row.get("user_id")


def tombstones(source: Catalog) -> dict[str, int]:
    rows = source.rows("conversation_tombstones")
    result = {identifier(row["conversation_id"]): when(row, "erased_at_unix_ms", "deleted_at")
              for row in rows}
    require(len(result) == len(rows), "duplicate conversation tombstone")
    return result


def filtered(source: Catalog, table: str, reason: str | None) -> bool:
    if reason:
        source.count(table, reason)
    return bool(reason)


@dataclass
class AmbientInputs:
    conversations: dict[str, dict]
    utterances: list[dict]

    @classmethod
    def read(cls, source: Catalog, privacy: Privacy) -> "AmbientInputs":
        conversations = {}
        for row in source.rows("capture_conversations"):
            cid = identifier(row.get("conversation_id", row.get("id")))
            require(cid not in conversations, "duplicate capture conversation")
            conversations[cid] = row
        utterances = source.rows("utterances")
        for row in utterances:
            key = row.get("utterance_id", row.get("capture_id"))
            if key is None:
                key = row.get("capture_id") or f"legacy-utterance:{row['id']}"
            row["_key"] = identifier(key)
            cid = row.get("conversation_id")
            row["_conversation"] = identifier(cid) if cid is not None else row["_key"]
        for cid, row in list(conversations.items()) + [
            (row["_conversation"], row) for row in utterances
        ]:
            privacy.conversation_owners.setdefault(cid, set()).update(value for value in people(row)
                                                                      if value not in (None, ""))
        return cls(conversations, utterances)

    def migrate(self, source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                now: float, retired_captures: set[str]) -> tuple[dict[str, str], dict[str, str]]:
        blocked = {}
        aliases = {}
        live_conversations = set()
        legacy_ids = {}
        for cid, row in self.conversations.items():
            if filtered(source, "capture_conversations", privacy.reason((cid,))):
                continue
            owner = unique_owner(privacy, cid)
            db.execute("INSERT INTO capture_conversations VALUES (?,?,?,?)",
                       (cid, row["room_id"], owner, when(row, "started_at_unix_ms", "started_at")))
            live_conversations.add(cid)
            source.count("capture_conversations", "migrated")
        for row in self.utterances:
            key, cid = row["_key"], row["_conversation"]
            reason = privacy.reason((cid,), people(row))
            if key in retired_captures or row.get("capture_id") in retired_captures:
                reason = reason or RETIRED
            if privacy.ledger.erased("utterance", key) is not None:
                reason = reason or TOMBSTONED
            if filtered(source, "utterances", reason):
                blocked[key] = reason
                if row.get("capture_id"):
                    blocked[row["capture_id"]] = reason
                continue
            expires = row.get("expires_at")
            if expires is not None and expires <= now:
                source.count("utterances", "discarded")
                blocked[key] = "discarded"
                if row.get("capture_id"):
                    blocked[row["capture_id"]] = "discarded"
                continue
            room = row["room_id"]
            at = when(row, "captured_at_unix_ms", "t0")
            if cid not in live_conversations:
                db.execute("INSERT INTO capture_conversations VALUES (?,?,?,?)",
                           (cid, room, unique_owner(privacy, cid), at))
                live_conversations.add(cid)
            db.execute("INSERT INTO utterances VALUES (?,?,?,?,?,?)", (
                key, cid, room, row.get("transcript_preview", row.get("text")), at, row.get("audio_path"),
            ))
            for alias in {key, row.get("capture_id")} - {None}:
                require(alias not in aliases, "duplicate ambient capture identity")
                aliases[alias] = key
            legacy_ids[str(row.get("id", key))] = key
            source.count("utterances", "migrated")
        for row in source.rows("pending_audio_deletions"):
            db.execute("INSERT INTO pending_audio_deletions VALUES (?,?)",
                       (row["path"], when(row, "requested_at_unix_ms", "requested_at")))
            source.count("pending_audio_deletions", "migrated")
        for row in source.rows("audio_write_intents"):
            cid = row["conversation_id"]
            if filtered(source, "audio_write_intents", privacy.reason((cid,), people(row))):
                continue
            key = legacy_ids.get(str(row["utterance_id"]))
            if key is None or cid not in live_conversations:
                source.count("audio_write_intents", "discarded")
                continue
            db.execute("INSERT INTO audio_write_intents VALUES (?,?,?,?)", (
                row["path"], cid, key, when(row, "expires_at_unix_ms", "expires_at"),
            ))
            source.count("audio_write_intents", "migrated")
        control(source, db, "ambient_runtime_control", "paused")
        return aliases, blocked


def unique_owner(privacy: Privacy, cid: str) -> str | None:
    owners = privacy.conversation_owners.get(cid, set())
    require(len(owners) <= 1, "ambiguous capture conversation owner")
    return next(iter(owners), None)


def control(source: Catalog, db: sqlite3.Connection, table: str, flag: str) -> None:
    rows = source.rows(table)
    require(len(rows) <= 1 and (not rows or rows[0]["id"] == 1),
            "invalid runtime control singleton")
    if rows:
        row = rows[0]
        at = row.get("updated_at_unix_ms", millis(row.get("changed_at", 0)))
        values = (1, row[flag], row["revision"], at)
        source.count(table, "migrated")
    else:
        values = (1, 0, 0, 0)
    db.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", values)


def vector(value: bytes, dimensions: int) -> bytes:
    require(isinstance(value, bytes) and len(value) == 4 * dimensions,
            "invalid embedding dimensions")
    floats = struct.unpack(f"<{dimensions}f", value)
    require(all(map(math.isfinite, floats)) and any(floats), "invalid embedding")
    return value


def migrate_facts(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                  now: float, retired_captures: set[str]) -> dict[str, str]:
    facts = source.rows("facts")
    owners = {row["fact_id"]: row for row in source.rows("fact_owners")}
    sources = source.rows("fact_sources")
    dependencies = source.rows("fact_dependencies")
    conversations = defaultdict(set)
    source_people = defaultdict(set)
    retired_facts = set()
    for row in sources:
        conversations[row["fact_id"]].add(identifier(row["conversation_id"]))
        source_people[row["fact_id"]].update(people(row))
        if row.get("capture_id") in retired_captures:
            retired_facts.add(row["fact_id"])
    reasons = {}
    identities = set()
    for row in facts:
        identity = identifier(row.get("fact_id", row.get("id")))
        row["_key"] = identity
        identities.add(identity)
        if row.get("conversation_id") is not None:
            conversations[identity].add(identifier(row["conversation_id"]))
        owner = owners.get(identity, {})
        reason = privacy.reason(conversations[identity], (
            *people(row), owner.get("person_id"), *source_people[identity],
        ))
        if identity in retired_facts:
            reason = reason or RETIRED
        if privacy.ledger.erased("fact", identity) is not None:
            reason = reason or TOMBSTONED
        state = row.get("state")
        require(state in {None, "proposed", "automatic", "confirmed", "deleted", "quarantined"},
                "invalid legacy fact state")
        expires = row.get("expires_at")
        if not reason and (state in {"deleted", "quarantined"} or
                           expires is not None and expires <= now):
            reason = "discarded"
        if reason:
            reasons[identity] = reason
    dependents = defaultdict(set)
    for row in dependencies:
        parent, child = row.get("depends_on_fact_id", row.get("source_fact_id")), row["fact_id"]
        require(parent in identities and child in identities, "missing fact dependency")
        dependents[parent].add(child)
    # Propagate erasure through all derivatives, with privacy reasons taking
    # precedence over an expiry anywhere upstream.
    for reason in (TOMBSTONED, RETIRED, "discarded"):
        pending = [key for key, value in reasons.items() if value == reason]
        while pending:
            for child in dependents[pending.pop()]:
                old = reasons.get(child)
                if old is None or old == "discarded" and reason != "discarded":
                    reasons[child] = reason
                    pending.append(child)
    live = set()
    inserted_sources = set()
    for row in facts:
        identity = row["_key"]
        if filtered(source, "facts", reasons.get(identity)):
            continue
        cids = conversations[identity]
        require(bool(cids), "live fact has no conversation provenance")
        owner = owners.get(identity, {}).get("person_id")
        require(not owner or not row.get("owner_user_id") or owner == row["owner_user_id"],
                "conflicting fact owners")
        owner = owner or row.get("owner_user_id") or ""
        at = when(row, "created_at_unix_ms", "extracted_at")
        confirmed = row.get("confirmed", int(row.get("state") == "confirmed"))
        db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?)", (
            identity, row.get("conversation_id") or min(cids), owner, row["content"],
            row["category"], at, confirmed,
        ))
        if row.get("embedding") is not None and confirmed:
            require(row.get("model_key", FACT_MODEL) == FACT_MODEL, "incompatible fact encoder")
            db.execute("INSERT INTO fact_vec VALUES (?,?,?)", (identity, vector(row["embedding"], 1024), owner))
        for cid in sorted(cids):
            db.execute("INSERT INTO fact_sources VALUES (?,?)", (identity, cid))
            inserted_sources.add((identity, cid))
        live.add(identity)
        source.count("facts", "migrated")
    seen = set()
    for row in sources:
        identity = row["fact_id"]
        if filtered(source, "fact_sources", reasons.get(identity)):
            continue
        edge = (identity, row["conversation_id"])
        require(edge in inserted_sources, "orphan fact source")
        source.count("fact_sources", "discarded" if edge in seen else "migrated")
        seen.add(edge)
    for row in dependencies:
        parent, child = row.get("depends_on_fact_id", row.get("source_fact_id")), row["fact_id"]
        if filtered(source, "fact_dependencies", reasons.get(child) or reasons.get(parent)):
            continue
        db.execute("INSERT INTO fact_dependencies VALUES (?,?)", (child, parent))
        source.count("fact_dependencies", "migrated")
    for identity, row in owners.items():
        if filtered(source, "fact_owners", reasons.get(identity)):
            continue
        require(identity in live, "orphan fact owner")
        db.execute("INSERT INTO fact_owners VALUES (?,?,?,?)", (
            identity, identifier(row["person_id"]), OWNER_METHODS[row["method"]], row["revision"],
        ))
        source.count("fact_owners", "migrated")
    for row in source.rows("fact_owner_revisions"):
        identity = row["fact_id"]
        if filtered(source, "fact_owner_revisions", reasons.get(identity)):
            continue
        require(identity in live, "orphan fact owner revision")
        db.execute("INSERT INTO fact_owner_revisions VALUES (?,?,?,?)", (
            identity, row["revision"], OWNER_CHANGES[row.get("change_kind", row.get("last_change"))],
            when(row, "changed_at_unix_ms", "changed_at"),
        ))
        source.count("fact_owner_revisions", "migrated")
    return reasons


def provenance(row: dict) -> list[dict]:
    if "sources_json" not in row:
        return []
    items = decode(row["sources_json"])
    require(type(items) is list and all(isinstance(item, dict) and item.get("conversation_id")
                                        for item in items), "invalid voice provenance")
    return items


def retired_capture_ids(source: Catalog, privacy: Privacy) -> set[str]:
    """Identify exact capture IDs associated with a retired person's linked profile."""
    profiles = set()
    for row in source.rows("voice_profile_people"):
        person = row.get("person_id", row.get("user_id"))
        if person is not None and privacy.is_retired(person):
            profiles.add(row["profile_id"])
    if not profiles:
        return set()
    captures = set()
    for row in source.rows("voice_profiles"):
        if row.get("profile_id", row.get("id")) in profiles:
            captures.update(item["capture_id"] for item in provenance(row) if item.get("capture_id"))
    fingerprints = {row["capture_id"] for row in source.rows("voice_profile_sources")
                    if row["profile_id"] in profiles and "capture_id" in row}
    for row in source.rows("voice_fingerprints"):
        if row.get("profile_id") in profiles or row.get("capture_id", row.get("id")) in fingerprints:
            if row.get("utterance_id") is not None:
                captures.add(row["utterance_id"])
            captures.update(item["capture_id"] for item in provenance(row) if item.get("capture_id"))
    return captures


def conversation_ids(row: dict, items: list[dict]) -> set[str]:
    return {row.get("conversation_id"), *(item["conversation_id"] for item in items)} - {None}


def encoder(row: dict) -> str:
    key = row.get("encoder_model_key", row.get("model_key"))
    require(key in {LEGACY_ENCODER, VOICE_MODEL} and row.get("dimensions", 192) == 192,
            "incompatible legacy voice encoder")
    return VOICE_MODEL


def migrate_voice(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                  aliases: dict[str, str], blocked: dict[str, str], fact_reasons: dict[str, str],
                  now: float) -> set[str]:
    def related_reason(items: list[dict]) -> str | None:
        reasons = {blocked.get(item.get("capture_id")) for item in items}
        for item in items:
            for dependency in item.get("fact_dependencies", []):
                fact = identifier(dependency["fact_id"])
                reasons.add(fact_reasons.get(fact))
                if privacy.ledger.erased("fact", fact) is not None:
                    reasons.add(TOMBSTONED)
        return next((reason for reason in (TOMBSTONED, RETIRED) if reason in reasons), None)

    profiles = source.rows("voice_profiles")
    captures = source.rows("voice_fingerprints")
    links = source.rows("voice_profile_people")
    review_rows = source.rows("voice_capture_reviews")
    profile_sources = source.rows("voice_profile_sources")
    capture_sources = source.rows("voice_fingerprint_sources")
    profile_cids, capture_cids = defaultdict(set), defaultdict(set)
    capture_rows = {row.get("capture_id", row.get("id")): row for row in captures}
    for row in capture_sources:
        key = row.get("capture_id", row.get("fingerprint_id"))
        require(key in capture_rows, "orphan fingerprint source")
        capture_cids[key].add(identifier(row["conversation_id"]))
    for key, row in capture_rows.items():
        row["_provenance"] = provenance(row)
        capture_cids[key].update(conversation_ids(row, row["_provenance"]))
    for row in profile_sources:
        cid = row.get("conversation_id")
        if cid is not None:
            profile_cids[row["profile_id"]].add(identifier(cid))
        else:
            require(row["capture_id"] in capture_rows, "orphan voice profile capture")
            profile_cids[row["profile_id"]].update(capture_cids[row["capture_id"]])
    for row in links:
        if row.get("conversation_id") is not None:
            profile_cids[row["profile_id"]].add(identifier(row["conversation_id"]))
    profile_reasons, capture_reasons = {}, {}
    live_profiles, live_links = {}, {}
    captures_by_source = defaultdict(set)
    for key, row in capture_rows.items():
        for item in row["_provenance"]:
            if item.get("capture_id") is not None:
                captures_by_source[item["conversation_id"], item["capture_id"]].add(key)
    retired_profile_captures = set()
    profile_rows = {}
    profile_items = {}
    linked_people = {row["profile_id"]: row.get("person_id", row.get("user_id")) for row in links}
    for row in profiles:
        key = identifier(row.get("profile_id", row.get("id")))
        items = provenance(row)
        profile_items[key] = items
        profile_cids[key].update(conversation_ids(row, items))
        reason = privacy.reason(profile_cids[key], (
            *people(row), linked_people.get(key), *(p for item in items for p in people(item)),
        ))
        reason = reason or related_reason(items)
        if privacy.ledger.erased("voice_profile_deleted", key) is not None:
            reason = reason or TOMBSTONED
        if not reason and (row.get("deleted_at") is not None or "embedding" in row and row["embedding"] is None):
            reason = "discarded"
        if filtered(source, "voice_profiles", reason):
            profile_reasons[key] = reason
            if reason == RETIRED:
                for item in items:
                    retired_profile_captures.update(captures_by_source[
                        item["conversation_id"], item.get("capture_id")
                    ])
            continue
        profile_rows[key] = row
        db.execute("INSERT INTO voice_profiles VALUES (?,?,?)", (key, encoder(row), row["revision"]))
        if row.get("embedding") is not None:
            db.execute("INSERT INTO voice_vec VALUES (?,?)", (key, vector(row["embedding"], 192)))
        live_profiles[key] = row["revision"]
        source.count("voice_profiles", "migrated")
    used_utterances = set()
    live_captures = set()
    for key, row in sorted(capture_rows.items()):
        identifier(key)
        items = row["_provenance"]
        item_keys = {item.get("capture_id") for item in items} | {row.get("utterance_id")}
        item_keys.discard(None)
        profile = row.get("profile_id")
        reason = privacy.reason(capture_cids[key], (
            *people(row), linked_people.get(profile), *(p for item in items for p in people(item)),
        ))
        reason = reason or related_reason(items) or next(
            (r for r in (TOMBSTONED, RETIRED) if r in {blocked.get(item) for item in item_keys}), None
        )
        if key in retired_profile_captures:
            reason = reason or RETIRED
        if not reason and row.get("expires_at") is not None and row["expires_at"] <= now:
            reason = "discarded"
        if filtered(source, "voice_fingerprints", reason):
            capture_reasons[key] = reason
            continue
        utterance = next((aliases.get(item, item) for item in sorted(item_keys)
                          if aliases.get(item, item) not in used_utterances), None)
        utterance = utterance or f"legacy-fingerprint:{key}"
        require(utterance not in used_utterances and bool(capture_cids[key]),
                "ambiguous or missing fingerprint provenance")
        used_utterances.add(utterance)
        db.execute("INSERT INTO voice_fingerprints VALUES (?,?,?,?,?,?)", (
            key, utterance, encoder(row), profile if profile in live_profiles else None,
            vector(row["embedding"], 192), when(row, "captured_at_unix_ms", "created_at"),
        ))
        for cid in sorted(capture_cids[key]):
            db.execute("INSERT INTO voice_fingerprint_sources VALUES (?,?)", (key, cid))
        live_captures.add(key)
        source.count("voice_fingerprints", "migrated")
    for row in capture_sources:
        key = row.get("capture_id", row.get("fingerprint_id"))
        if not filtered(source, "voice_fingerprint_sources", capture_reasons.get(key)):
            source.count("voice_fingerprint_sources", "migrated")
    linked_sources = defaultdict(set)
    for row in profile_sources:
        key = row["profile_id"]
        if filtered(source, "voice_profile_sources", profile_reasons.get(key)):
            continue
        require(key in live_profiles, "orphan voice profile source")
        if "capture_id" in row:
            if filtered(source, "voice_profile_sources", capture_reasons.get(row["capture_id"])):
                continue
            sources = {row["capture_id"]}
        else:
            cid = row["conversation_id"]
            matches = set().union(*(captures_by_source[cid, item.get("capture_id")]
                                    for item in profile_items[key] if item["conversation_id"] == cid))
            sources = matches & live_captures
            if not sources:
                sources = {carry_profile_source(db, profile_rows[key], key, cid)}
            linked_sources[key].add(cid)
        for capture in sorted(sources):
            db.execute("INSERT OR IGNORE INTO voice_profile_sources VALUES (?,?)", (key, capture))
        source.count("voice_profile_sources", "migrated" if sources else "discarded")
        if "capture_id" in row:
            linked_sources[key].update(capture_cids[row["capture_id"]])
    for key, row in profile_rows.items():
        # Old profiles can outlive the capture that produced their vector. Keep
        # those conversations erasable even when there is no retained observation.
        for cid in sorted(profile_cids[key] - linked_sources[key]):
            capture = carry_profile_source(db, row, key, cid)
            db.execute("INSERT INTO voice_profile_sources VALUES (?,?)", (key, capture))
    for row in review_rows:
        key = row.get("capture_id", row.get("fingerprint_id"))
        if filtered(source, "voice_capture_reviews", capture_reasons.get(key)):
            continue
        require(key in live_captures, "orphan capture review")
        if "decision" not in row:
            source.count("voice_capture_reviews", "discarded")
            continue
        db.execute("INSERT INTO voice_capture_reviews VALUES (?,?,?)",
                   (key, row["decision"], row["reviewed_at_unix_ms"]))
        source.count("voice_capture_reviews", "migrated")
    for row in links:
        key = row["profile_id"]
        person = row.get("person_id", row.get("user_id"))
        reason = privacy.reason(profile_cids[key], (person, *people(row))) or profile_reasons.get(key)
        if filtered(source, "voice_profile_people", reason):
            continue
        require(key in live_profiles, "orphan voice profile link")
        if person in privacy.inactive:
            person = None
        db.execute("INSERT INTO voice_profile_people VALUES (?,?,?,?)", (
            key, person, row["revision"], when(row, "updated_at_unix_ms", "updated_at"),
        ))
        live_links[key] = (person, row["revision"])
        source.count("voice_profile_people", "migrated")
    migrate_consents(source, db, privacy, live_links, live_profiles)
    migrate_voice_history(source, db, privacy, live_profiles, profile_reasons, capture_reasons)
    return live_captures


def carry_profile_source(db: sqlite3.Connection, row: dict, profile: str, conversation: str) -> str:
    """Keep an erased or expired legacy observation out while retaining its lineage."""
    embedding = row.get("embedding")
    require(embedding is not None, "voice profile lineage has no linked capture")
    digest = hashlib.sha256((profile + "\0" + conversation).encode()).hexdigest()
    capture = "legacy-profile-source:" + digest
    # This row has no ambient utterance, audio, transcript or evaluation grant.
    # Its embedding is the already retained profile vector, with no new evidence
    # of an observation. Nexa uses the source edge when erasing the profile.
    db.execute("INSERT INTO voice_fingerprints VALUES (?,?,?,?,?,?)", (
        capture, capture, VOICE_MODEL, profile, vector(embedding, 192),
        when(row, "created_at_unix_ms", "confirmed_at"),
    ))
    db.execute("INSERT INTO voice_fingerprint_sources VALUES (?,?)", (capture, conversation))
    return capture


def migrate_consents(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                     links: dict, profiles: dict) -> None:
    withdrawals = sum(kind == "voice_consent_withdrawn" for kind, _ in privacy.ledger.hits)
    for row in source.rows("voice_attribution_consents"):
        person = identifier(row["person_id"])
        profile = row.get("profile_id", row.get("verified_profile_id"))
        prev = row.get("profile_revision", row.get("verified_profile_revision"))
        link = row.get("link_revision", row.get("verified_link_revision"))
        state = row.get("state")
        require(state in {None, "pending_verification", "active", "revoked", "invalidated",
                          "verification_locked"}, "invalid voice consent state")
        at = when(row, "updated_at_unix_ms", "changed_at")
        retired = privacy.is_retired(person)
        source_granted = state == "active" if state is not None else row.get("granted") == 1
        withdrawn = False
        if source_granted and withdrawals:
            if state is None:
                # Native rows have no legacy revision. A hash of person#revision
                # cannot rule out a withdrawal for this person, and timestamps
                # on restored rows cannot establish a new opt-in.
                withdrawn = True
            else:
                revision = row["revision"]
                require(type(revision) is int and revision > 0, "invalid consent revision")
                # Match the legacy restore guard's revision window.
                withdrawn = any(privacy.ledger.erased("voice_consent_withdrawn", f"{person}#{later}")
                                is not None for later in range(revision, revision + withdrawals + 1))
        granted = (
            source_granted
            and profile is not None and prev is not None and link is not None
            and profiles.get(profile) == prev and links.get(profile) == (person, link)
            and not retired and person not in privacy.inactive
            and not withdrawn
        )
        db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,?,?,?,?)", (
            person, profile if granted else None, prev if granted else None,
            link if granted else None, int(granted), at,
        ))
        source.count("voice_attribution_consents", "migrated")
    control(source, db, "voice_attribution_control", "enabled")


def migrate_voice_history(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                          profiles: dict, profile_reasons: dict, capture_reasons: dict) -> None:
    for row in source.rows("voice_attribution_events"):
        key = row["profile_id"]
        if filtered(source, "voice_attribution_events", profile_reasons.get(key)):
            continue
        if row["kind"] not in {"match", "suspend", "resume"}:
            source.count("voice_attribution_events", "discarded")
            continue
        db.execute("INSERT INTO voice_attribution_events VALUES (?,?,?,?)", (
            row.get("event_id", row.get("id")), key, row["kind"],
            when(row, "at_unix_ms", "at") if "event_id" not in row else row["at"],
        ))
        source.count("voice_attribution_events", "migrated")
    for row in source.rows("voice_attribution_suspensions"):
        key = row["profile_id"]
        if filtered(source, "voice_attribution_suspensions", profile_reasons.get(key)):
            continue
        db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,?,?)", (
            key, row["revision"], when(row, "suspended_at_unix_ms", "suspended_at"),
        ))
        source.count("voice_attribution_suspensions", "migrated")
    for row in source.rows("voice_verification_attempts"):
        person = row["person_id"]
        if filtered(source, "voice_verification_attempts", privacy.reason(people=(person,))):
            continue
        outcome = {"failed_foreign": "failed"}.get(row["outcome"], row["outcome"])
        if outcome not in {"passed", "failed"}:
            source.count("voice_verification_attempts", "discarded")
            continue
        db.execute("INSERT INTO voice_verification_attempts VALUES (?,?,?)", (
            person, when(row, "attempted_at_unix_ms", "attempted_at"), outcome,
        ))
        source.count("voice_verification_attempts", "migrated")
    groups = {}
    for row in source.rows("voice_groups"):
        key = row.get("group_id", row.get("id"))
        items = provenance(row)
        reason = privacy.reason(conversation_ids(row, items), (
            *people(row), *(person for item in items for person in people(item)),
        ))
        if reason is None and row.get("expires_at") is not None:
            # Legacy groups cluster fingerprints; Nexa groups contain profiles.
            reason = "discarded"
        groups[key] = reason
        if not filtered(source, "voice_groups", reason):
            db.execute("INSERT INTO voice_groups VALUES (?,?)", (key, row["revision"]))
            source.count("voice_groups", "migrated")
    for table in ("voice_group_members", "voice_group_requests"):
        for row in source.rows(table):
            profile = row.get("profile_id")
            reason = (groups.get(row.get("group_id")) or profile_reasons.get(profile)
                      or capture_reasons.get(row.get("fingerprint_id")))
            if not filtered(source, table, reason):
                if profile is None:
                    source.count(table, "discarded")
                else:
                    require(profile in profiles, "orphan voice group profile")
                    values = [row["group_id"], profile]
                    if table == "voice_group_requests":
                        values.append(row["state"])
                    args = ",".join("?" for _ in values)
                    db.execute(f"INSERT INTO {table} VALUES ({args})", values)
                    source.count(table, "migrated")
    source.omit("voice_review_requests", source.rows("voice_review_requests"))
