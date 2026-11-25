"""Translate the two ledger formats and apply shared erasure fences."""

import hashlib
import hmac
import json
import secrets
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from storage import Catalog, HEX_SHA256, finite, identifier, millis, require, write_json

LEDGER = "deletion_ledger_v1"
SALT = "deletion_ledger_salt_v1"
LEGACY_KINDS = {
    "conversation_erased": "conversation",
    "fact_deleted": "fact",
    "fact_pruned_retired_owner": "fact",
    "user_retired": "retired_user",
    "voice_profile_deleted": None,
    "credential_revoked": None,
    "voice_consent_withdrawn": None,
}
NATIVE_KINDS = {"conversation", "fact", "retired_user", "utterance"}
TOMBSTONED = "filtered_tombstoned"
RETIRED = "filtered_retired"


def native_digest(salt: bytes, kind: str, identity: str) -> bytes:
    return hashlib.sha256(salt + kind.encode() + b"\0" + identity.encode()).digest()


def native_chain(seq: int, kind: str, subject: bytes, at: int, prev: bytes) -> bytes:
    return hashlib.sha256(
        prev + seq.to_bytes(8, "big") + at.to_bytes(8, "big") + subject + kind.encode()
    ).digest()


@dataclass
class Ledger:
    """Validate source rows before using any erasure or consent evidence."""

    salt: bytes
    legacy: bool
    rows: list[dict]
    hits: dict[tuple[str, str | bytes], float] = field(default_factory=dict)
    resolved: dict[tuple[str, str | bytes], str] = field(default_factory=dict)

    @classmethod
    def read(cls, source: Catalog) -> "Ledger":
        rows = sorted(source.rows(LEDGER), key=lambda row: row["seq"])
        salts = source.rows(SALT)
        require((LEDGER in source.tables) == (SALT in source.tables)
                and source.tables.get(LEDGER) is source.tables.get(SALT)
                and (len(salts) == 1 or not salts and not rows),
                "deletion ledger or salt is incomplete")
        salt = salts[0]["salt"] if salts else secrets.token_bytes(32)
        legacy = isinstance(salt, str)
        if legacy:
            require(bool(HEX_SHA256.fullmatch(salt)), "invalid deletion ledger salt")
            salt = bytes.fromhex(salt)
        require(isinstance(salt, bytes) and len(salt) == 32
                and (not salts or salts[0]["id"] == 1), "invalid deletion ledger salt")
        prev = "0" * 64 if legacy else hashlib.sha256(salt).digest()
        hits = {}
        for seq, row in enumerate(rows, 1):
            kind, subject, at = row["kind"], row["subject_sha256"], row["at"]
            require(type(row["seq"]) is int and seq == row["seq"] and row["prev_sha256"] == prev,
                    "broken deletion ledger chain")
            if legacy:
                require(kind in LEGACY_KINDS and isinstance(subject, str)
                        and bool(HEX_SHA256.fullmatch(subject)) and type(at) is float and finite(at) >= 0,
                        "invalid legacy deletion ledger row")
                payload = json.dumps([seq, kind, subject, at], separators=(",", ":"), allow_nan=False)
                prev = hashlib.sha256(prev.encode() + payload.encode()).hexdigest()
            else:
                require(kind in NATIVE_KINDS and isinstance(subject, bytes) and len(subject) == 32
                        and type(at) is int and at >= 0, "invalid Nexa deletion ledger row")
                prev = native_chain(seq, kind, subject, at, prev)
            require(prev == row["row_sha256"] and (kind, subject) not in hits,
                    "broken deletion ledger chain")
            hits[kind, subject] = at
        return cls(salt, legacy, rows, hits)

    def erased(self, kind: str, identity: str) -> int | None:
        if not self.legacy:
            at = self.hits.get((kind, native_digest(self.salt, kind, identity)))
            return at
        subject = hmac.digest(self.salt, identity.encode(), "sha256").hex()
        found = []
        for legacy, native in LEGACY_KINDS.items():
            if kind == (native or legacy) and (legacy, subject) in self.hits:
                self.resolved[legacy, subject] = identity
                found.append(millis(self.hits[legacy, subject]))
        return min(found) if found else None

    def write(self, source: Catalog, db: sqlite3.Connection, root: Path,
              tombstones: dict[str, int], retired: dict[str, float]) -> None:
        db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (1, ?)", (self.salt,))
        source.count(SALT, "migrated", source.counts[SALT].source)
        if self.legacy:
            write_json(root / "memory/legacy-deletion-ledger.json", {
                "format": "kza-python-deletion-ledger-v1", "salt": self.salt.hex(), "rows": self.rows,
            })
        prev = hashlib.sha256(self.salt).digest()
        present = set()

        def append(kind: str, subject: bytes, at: int) -> bool:
            nonlocal prev
            if (kind, subject) in present:
                return False
            seq = len(present) + 1
            digest = native_chain(seq, kind, subject, at, prev)
            db.execute("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)",
                       (seq, kind, subject, at, prev, digest))
            present.add((kind, subject))
            prev = digest
            return True

        for row in self.rows:
            kind, subject, at = row["kind"], row["subject_sha256"], row["at"]
            if self.legacy:
                identity = self.resolved.get((kind, subject))
                target = LEGACY_KINDS[kind]
                if target is None:
                    source.count(LEDGER, "archived")
                    continue
                # Unknown preimages cannot be re-keyed. Retain opaque evidence in
                # the native chain and keep the verifiable original in the archive.
                subject = (native_digest(self.salt, target, identity) if identity is not None else
                           hashlib.sha256(b"kza-legacy-deletion-v1\0" + kind.encode()
                                          + bytes.fromhex(subject)).digest())
                kind, at = target, millis(at)
            if append(kind, subject, at):
                source.count(LEDGER, "migrated")
            else:
                source.count(LEDGER, "archived")

        for conversation, at in sorted(tombstones.items()):
            append("conversation", native_digest(self.salt, "conversation", conversation), at)
        for person, at in sorted(retired.items()):
            append("retired_user", native_digest(self.salt, "retired_user", person), millis(at))

        previous = hashlib.sha256(self.salt).digest()
        for seq, row in enumerate(db.execute(
            "SELECT seq,kind,subject_sha256,at,prev_sha256,row_sha256 "
            "FROM deletion_ledger_v1 ORDER BY seq"
        ), 1):
            number, kind, subject, at, prev, digest = row
            require(number == seq and prev == previous
                    and digest == native_chain(seq, kind, subject, at, previous),
                    "destination deletion ledger chain invalid")
            previous = digest


@dataclass
class Privacy:
    ledger: Ledger
    tombstones: dict[str, int]
    retired: dict[str, float]
    inactive: set[str] = field(default_factory=set)
    conversation_owners: dict[str, set[str]] = field(default_factory=dict)

    def is_tombstoned(self, conversation: str) -> bool:
        identifier(conversation)
        at = self.ledger.erased("conversation", conversation)
        if at is not None:
            self.tombstones.setdefault(conversation, at)
        return conversation in self.tombstones

    def is_retired(self, person: str) -> bool:
        identifier(person)
        at = self.ledger.erased("retired_user", person)
        if at is not None:
            self.retired.setdefault(person, at / 1000)
        return person in self.retired

    def reason(self, conversations=(), people=()) -> str | None:
        conversations = tuple(identifier(item) for item in conversations if item is not None)
        owners = set(people) | set().union(*(self.conversation_owners.get(cid, set())
                                            for cid in conversations))
        tombstoned = [self.is_tombstoned(cid) for cid in conversations]
        retired = [self.is_retired(person) for person in owners if person not in (None, "")]
        if any(tombstoned):
            return TOMBSTONED
        if any(retired):
            return RETIRED
        return None
