"""Import the account catalog, credentials and retirement authority."""

import sqlite3

from privacy import Privacy, RETIRED
from storage import Catalog, HEX_SHA256, decode, encode, finite, identifier, require

LEVELS = ("guest", "child", "teen", "adult", "admin")
TABLES = {"auth_meta", "credentials", "sessions", "retired_users", "nexa_accounts",
          "nexa_purge_pending"}


def read_retired(source: Catalog) -> dict[str, float]:
    rows = source.rows("retired_users")
    result = {identifier(row["user_id"]): finite(row["retired_at"]) for row in rows}
    require(len(result) == len(rows) and all(value >= 0 for value in result.values()),
            "invalid retirement authority")
    return result


def migrate_accounts(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                     catalog: list[dict] | None) -> None:
    rows = source.rows("nexa_accounts")
    require(not rows or catalog is None, "ambiguous account authorities")
    if catalog is not None:
        rows = catalog
        source.count("nexa_accounts", "source", len(rows))
    seen = set()
    for row in rows:
        person = identifier(row.get("user_id", row.get("id")))
        require(person not in seen, "duplicate account identity")
        seen.add(person)
        active = row.get("is_active", True)
        require(type(active) is bool, "invalid account status")
        if not active:
            privacy.inactive.add(person)
        if privacy.is_retired(person):
            source.count("nexa_accounts", RETIRED)
        elif not active:
            source.count("nexa_accounts", "discarded")
        else:
            level = row["permission_level"]
            if type(level) is int:
                require(0 <= level < len(LEVELS), "invalid account permission level")
                level = LEVELS[level]
            require(isinstance(level, str) and level.lower() in LEVELS,
                    "invalid account permission level")
            name = identifier(row["name"])
            db.execute("INSERT INTO nexa_accounts VALUES (?,?,?)", (person, name, level.lower()))
            source.count("nexa_accounts", "migrated")


def migrate_auth(source: Catalog, db: sqlite3.Connection, privacy: Privacy,
                 voice_captures: set[str]) -> None:
    for row in source.rows("credentials"):
        user, identity = row["user_id"], identifier(row["id"])
        if user is not None and privacy.is_retired(user):
            source.count("credentials", RETIRED)
            continue
        if user in privacy.inactive or privacy.ledger.erased("credential_revoked", identity) is not None:
            source.count("credentials", "discarded")
            continue
        digest, role = row["digest"], row["role"]
        require(isinstance(digest, str) and bool(HEX_SHA256.fullmatch(digest))
                and role in {"operator", "home", "evaluator"} and (user is None or role == "home"),
                "invalid credential hash or role")
        issued, expires = finite(row["issued_at"]), finite(row["expires_at"])
        revision = row["revision"]
        require(0 <= issued < expires and type(revision) is int and revision >= 1,
                "invalid credential lifetime")
        samples = decode(row["samples"])
        require(type(samples) is list and len(samples) <= 64 and bool(samples) == (role == "evaluator"),
                "invalid credential samples")
        valid = []
        for sample in samples:
            fingerprint = identifier(sample["fingerprint_id"])
            digests = sample["source_digests"]
            require(type(digests) is list and 1 <= len(digests) <= 256
                    and all(isinstance(d, str) and HEX_SHA256.fullmatch(d) for d in digests)
                    and type(sample["modalities"]) is list
                    and all(m in {"audio", "transcript"} for m in sample["modalities"]),
                    "invalid credential sample scope")
            if fingerprint in voice_captures:
                valid.append({key: sample[key] for key in
                              ("fingerprint_id", "source_digests", "modalities", "expires_at")})
                finite(valid[-1]["expires_at"])
        if role == "evaluator" and not valid:
            source.count("credentials", "discarded")
            continue
        db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)",
                   (identity, digest, identifier(row["subject"]), role, user, issued, expires,
                    revision, encode(valid)))
        source.count("credentials", "migrated")
    for row in source.rows("auth_meta"):
        if row["key"] == "clock_floor":
            floor = float(row["value"])
            require(finite(floor) >= 0, "invalid authority clock floor")
            db.execute("INSERT INTO auth_meta VALUES ('clock_floor',?)", (str(floor),))
            source.count("auth_meta", "migrated")
        else:
            source.count("auth_meta", "discarded")
    source.omit("sessions", source.rows("sessions"))
    db.executemany("INSERT INTO retired_users VALUES (?,?)", sorted(privacy.retired.items()))
    source.count("retired_users", "migrated", source.counts["retired_users"].source)
    for row in source.rows("nexa_purge_pending"):
        db.execute("INSERT INTO nexa_purge_pending VALUES (?)", (row["user_id"],))
        source.count("nexa_purge_pending", "migrated")
