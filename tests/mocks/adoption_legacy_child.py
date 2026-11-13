"""Write a b0866e8-shaped legacy state with b0866e8's own code (U3b-0e2 fixture).

Run as ``python -I -B adoption_legacy_child.py <b0866e8 tree> <container> <logs> <legacy events DDL>``.
The working directory becomes ``<container>``: the production unit's shape,
where WorkingDirectory is the 63ea8d3 container whose config/data/.env links
resolve into the b0866e8 slot state and whose models link resolves into the
legacy checkout. Only the b0866e8 tree is importable: every store, schema,
target registration, fact deletion and conversation erasure below is b0866e8
code (its builders, ledger and journal APIs). Rows that b0866e8 offers no
offline writer for (utterances, policy events, clips, voice profiles, live
receipts, reminders) are inserted with SQL into the tables b0866e8 created.

The legacy pre-policy ``events`` table predates b0866e8 (8558ff1 stopped
creating it and kept it in place); the parent passes its DDL from
``git show 8558ff1^:src/analytics/event_logger.py`` and the rows are written
before b0866e8's EventLogger opens the file, as in production. audit.db is
written by b0866e8's own audit_sqlite. Every text value is a synthetic marker.

The fixture is settled: every erased conversation's steps on targets with a
cleaner are complete and the deleted fact's index deletion is complete, so a
copy has no residue until a test injects one.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
import sqlite3
import sys

TREE, CONTAINER, LOGS, LEGACY_EVENTS_DDL = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, TREE)

import yaml  # noqa: E402

EVENTS_MARKER = "KZA-LEGACY-EVENTS-MARKER"
AUDIT_MARKER = "KZA-LEGACY-AUDIT-MARKER"
NOW = 1_800_000_000.0
CONVERSATION_ERASED = "11111111-1111-4111-8111-111111111111"
CONVERSATION_LIVE = "22222222-2222-4222-8222-222222222222"
CAPTURE_ERASED = "33333333-3333-4333-8333-333333333333"
CAPTURE_LIVE = "44444444-4444-4444-8444-444444444444"
CAPTURE_DELETED_FACT = "55555555-5555-4555-8555-555555555555"
CONVERSATION_DELETED_FACT = "66666666-6666-4666-8666-666666666666"
VOICE_DELETED = "voice_deleted_profile"
VOICE_LIVE = "voice_live_profile"
# Cleaner-less literal targets: their steps never complete (U3b-0d4a).
_HONEST_BLOCKERS = ({"kind": "unresolved_diagnostic_inventory"}, {"kind": "unresolved_stream_inventory"},
                    {"kind": "unresolved_voice_inventory"})


def _source(capture: str, conversation: str):
    from src.memory.fact_ledger import MemoryOrigin, MemorySource
    return MemorySource(capture, conversation, NOW - 60, NOW + 3600, MemoryOrigin.VOICE)


async def build() -> dict:
    from src.ambient.erasure import build_capture_erasure
    from src.ambient.store import AmbientStore
    from src.analytics.erasure import build_event_erasure
    from src.analytics.event_logger import EventLogger
    from src.analytics.suggestion_erasure import build_suggestion_erasure
    from src.analytics.suggestion_store import SuggestionStore
    from src.dashboard.live_erasure import build_live_services
    from src.lists.list_store import ListStore
    from src.memory.fact_ledger import MemoryReviewer
    from src.memory.index_erasure import build_erasure_services
    from src.memory.privacy_services import build_privacy_services
    from src.monitoring.latency_monitor import LatencyMonitor
    from src.obs.diagnostic_erasure import build_diagnostic_services
    from src.orchestrator.context_erasure import build_context_erasure
    from src.policies import audit_sqlite
    from src.reminders.reminder_store import ReminderStore
    from src.routines.routine_store import RoutineStore
    from src.users.voice_registry import build_voice_registry
    from src.wakeword.clip_store import build_wake_clip_store

    config = yaml.safe_load(Path("config/settings.yaml").read_text())
    config.setdefault("observability", {})["metrics_path"] = str(LOGS / "kza-metrics.jsonl")
    actor = MemoryReviewer("fixture:operator")
    data = Path("data")
    # The pre-policy events table, before b0866e8 opens events.db (production order).
    with sqlite3.connect(data / "events.db") as legacy:
        legacy.execute(LEGACY_EVENTS_DDL)
        legacy.executemany(
            "INSERT INTO events(timestamp,event_type,entity_id,action,user_id,user_name,trigger_phrase,"
            "context_json,hour,minute,weekday,is_weekend,time_of_day) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(NOW - index, "ha_action", "light.hall", "turn_on", "user_a", EVENTS_MARKER + "-user",
              EVENTS_MARKER + f"-phrase-{index}", json.dumps({"note": EVENTS_MARKER}), 1, 2, 3, 0, "night")
             for index in range(200)])
    report: dict = {}
    async with AsyncExitStack() as shutdown:
        privacy = await build_privacy_services(config, shutdown=shutdown, clock=lambda: NOW)
        owner = await build_erasure_services(config, privacy=privacy, shutdown=shutdown)
        await build_voice_registry(privacy=privacy, owner=owner, shutdown=shutdown)
        await build_diagnostic_services(config, privacy=privacy, owner=owner)
        await build_capture_erasure(config, privacy=privacy, owner=owner)
        await build_event_erasure(config, privacy=privacy, owner=owner)
        await build_suggestion_erasure(config, privacy=privacy, owner=owner)
        await build_context_erasure(config, privacy=privacy, owner=owner)
        await build_wake_clip_store(config, privacy=privacy, shutdown=shutdown, owner=owner)
        await build_live_services(config, privacy=privacy, owner=owner, shutdown=shutdown)
        ledger = privacy.ledger
        events = EventLogger(config["analytics"]["events_db"], source_policy=ledger, clock=lambda: NOW)
        shutdown.push_async_callback(events.aclose)
        await asyncio.to_thread(events.initialize)
        suggestions = SuggestionStore(events, config["analytics"]["suggestions_path"])
        ambient = AmbientStore(config["ambient"]["db_path"], source_policy=ledger, clock=lambda: NOW)
        await ambient.init()
        shutdown.push_async_callback(ambient.close)
        stores = [ListStore(config["lists"]["db_path"]), ReminderStore(config["reminders"]["db_path"]),
                  RoutineStore(config["routines"]["db_path"])]
        for store in stores:
            await store.initialize()
            shutdown.push_async_callback(store.close)
        LatencyMonitor(config["monitoring"]["db_path"])
        # Facts: one live, one deleted through the review boundary.
        live = ledger.store("KZA fixture live fact", "fact", .9, (_source(CAPTURE_LIVE, CONVERSATION_LIVE),))
        deleted = ledger.store("KZA fixture deleted fact", "fact", .9,
                               (_source(CAPTURE_DELETED_FACT, CONVERSATION_DELETED_FACT),))
        ledger.delete_fact(deleted.id, actor)
        # Payload referencing the conversation that is then erased.
        ledger.store("KZA fixture erased-conversation fact", "fact", .9, (_source(CAPTURE_ERASED, CONVERSATION_ERASED),))
        await ambient._db.execute(
            "INSERT INTO utterances(room_id,t0,t1,text,source,conversation_id,capture_id,expires_at,created_at,audio_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("hall", NOW - 30, NOW - 29, "KZA fixture live utterance", "ambient", CONVERSATION_LIVE, CAPTURE_LIVE,
             NOW + 3600, NOW - 30, str((data / "ambient_audio" / "policy_v1" / "live.flac").resolve())))
        await ambient._db.commit()
        audio = data / "ambient_audio" / "policy_v1"
        audio.mkdir(parents=True, exist_ok=True)
        (audio / "live.flac").write_bytes(b"fLaC fixture")
        with ledger._transaction() as db:
            targets = {(row["participant"], row["version"]): row["id"] for row in db.execute(
                "SELECT id, participant, version FROM erasure_targets")}
            clip_target = db.execute("SELECT id FROM erasure_targets WHERE participant='collectors'"
                                     " AND descriptor LIKE '%\"wake_clips\"%'").fetchone()[0]
            sources = json.dumps([_source(CAPTURE_LIVE, CONVERSATION_LIVE).to_dict()])
            db.execute("INSERT INTO wake_clips_v1(id,target_id,bucket,room_id,score,sources_json,captured_at,"
                       "expires_at,published) VALUES (?,?,?,?,?,?,?,?,?)",
                       ("clip_live", clip_target, "accepted", "hall", .9, sources, NOW - 30, NOW + 3600, 1))
            db.execute("INSERT INTO voice_profiles(id,name,model_key,dimensions,embedding,sources_json,confirmed_at,"
                       "confirmed_by,revision,deleted_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (VOICE_LIVE, "Fixture", "ecapa", 2, b"\x00" * 8, "[]", NOW - 100, "fixture", 1, None))
            db.execute("INSERT INTO voice_profiles(id,name,model_key,dimensions,embedding,sources_json,confirmed_at,"
                       "confirmed_by,revision,deleted_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (VOICE_DELETED, None, "ecapa", 2, None, "[]", NOW - 100, "fixture", 2, NOW - 50))
            db.execute("INSERT INTO live_deliveries(ticket,client_id,connection_id,event_id,nonce,issued_at,expires_at)"
                       " VALUES (?,?,?,?,?,?,?)", ("ticket-1", "client-1", "conn-1", "event-1", "nonce", NOW, NOW + 60))
            db.execute("INSERT INTO live_connections VALUES (?,?,?)", ("conn-1", "client-1", "runtime-1"))
            db.execute("INSERT INTO live_resets VALUES (?,?,?,?,?)", ("reset-1", "client-1", "conn-2", "nonce", 3))
        with sqlite3.connect(config["analytics"]["events_db"]) as db:
            db.execute("INSERT INTO events_policy_v1(timestamp,event_type,entity_id,action,user_id,user_name,"
                       "trigger_phrase,context_json,hour,minute,weekday,is_weekend,time_of_day,sources_json,"
                       "metrics_json,expires_at,text_expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (NOW - 20, "ha_action", "light.hall", "turn_on", "user_a", "Fixture", "enciende",
                        "{}", 1, 2, 3, 0, "night", json.dumps([_source(CAPTURE_ERASED, CONVERSATION_ERASED).to_dict()]),
                        "{}", NOW + 86400, NOW + 3600))
        with sqlite3.connect(config["reminders"]["db_path"]) as db:
            columns = [row[1] for row in db.execute("PRAGMA table_info(reminders)")]
            values = {"id": "rem-1", "user_id": "user_a", "text": "regar plantas", "trigger_at": NOW + 600,
                      "created_at": NOW, "recurrence": None, "recurrence_end": None, "ha_actions": "[]",
                      "state": "pending", "last_fired_at": None, "fire_count": 0}
            db.execute("INSERT INTO reminders(" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")",
                       [values.get(name) for name in columns])
        contexts = data / "contexts" / "policy_v3"
        contexts.mkdir(parents=True, exist_ok=True)
        (contexts / "user_a.json").write_text(json.dumps({"user_id": "user_a", "conversation_id": CONVERSATION_LIVE}))
        captured = data / "wakeword_training" / "captured" / "policy_v1"
        captured.mkdir(parents=True, exist_ok=True)
        (captured / "clip_live.wav").write_bytes(b"RIFF fixture")
        # The erasure of CONVERSATION_ERASED, settled: every step with a cleaner completes.
        ledger.delete_conversation(CONVERSATION_ERASED, actor)
        with ledger._transaction() as db:
            blockers = {targets[key] for key in targets
                        if json.loads(db.execute("SELECT descriptor FROM erasure_targets WHERE id=?",
                                                 (targets[key],)).fetchone()[0]) in _HONEST_BLOCKERS}
            db.execute("UPDATE erasure_steps SET state='complete', last_error=NULL WHERE target_id NOT IN ("
                       + ",".join("?" for _ in blockers) + ")", tuple(blockers))
            db.execute("UPDATE policy_index_deletions SET state='complete'")
        with sqlite3.connect(config["analytics"]["events_db"]) as db:
            db.execute("DELETE FROM events_policy_v1 WHERE sources_json LIKE ?", ("%" + CONVERSATION_ERASED + "%",))
        await ambient._db.commit()
        del suggestions
        import chromadb
        for directory, collection, ids in (
                (config["memory"]["chroma_path"], config["memory"].get("long_term_collection", "user_memories")
                 + "_policy_v1", [live.id, deleted.id]),
                (config["vectordb"]["path"], "home_entities", ["light.hall"])):
            client = chromadb.PersistentClient(path=str(Path(directory)))
            target = client.get_or_create_collection(collection)
            target.add(ids=ids, embeddings=[[0.1 * (index + 1), 0.2, 0.3] for index in range(len(ids))],
                       metadatas=[{"revision": 1} for _ in ids])
        report["facts"] = {"live": live.id, "deleted": deleted.id}
        with ledger._transaction() as db:
            report["targets"] = [dict(row) for row in db.execute("SELECT * FROM erasure_targets ORDER BY id")]
    # The b0866e8 index worker may or may not have drained its jobs before the
    # close; pin the settled outcome: the deleted fact's index deletion is still
    # queued (a crash before reconciliation), so its embedding is not residue.
    with sqlite3.connect(config["memory"]["privacy_db_path"]) as db:
        revision = db.execute("SELECT revision FROM facts WHERE id=?", (report["facts"]["deleted"],)).fetchone()[0]
        db.execute("DELETE FROM index_jobs")
        db.execute("INSERT INTO index_jobs(fact_id, revision, operation, queued_at) VALUES (?,?,?,?)",
                   (report["facts"]["deleted"], revision, "delete", NOW))
    # b0866e8's own audit_sqlite, at the path its unit Environment would give it.
    connection = audit_sqlite._open_db(Path("data/audit.db"))
    connection.executemany("INSERT INTO events(timestamp,kind,payload_json) VALUES (?,?,?)",
                           [(NOW + index, "stt", json.dumps({"text": AUDIT_MARKER})) for index in range(20)])
    connection.commit()
    connection.close()
    return report


def main() -> int:
    os.chdir(CONTAINER)
    report = asyncio.run(build())
    sys.stdout.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
