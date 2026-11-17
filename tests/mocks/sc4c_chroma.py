"""SC-4c: H states with real Chroma stores, written by a world's closing child (tests.mocks.compaction_world).

``chroma_state()`` is child code (``state`` bound) that writes, with synthetic
content only:

* data/memory_db, H's registered semantic_index store, with the policy
  collection ``user_memories_policy_v1`` (long_term.py's name and metadata)
  indexing four facts stored through the production FactLedger: one live
  (kept), one past its expiry (overdue), one owned by a retired user whose
  prune has not run, and one deleted with its conversation (tombstone,
  erasure operation and ledger entry; its record is the residue a pending
  index deletion leaves); plus an orphan record (no fact);
* data/chroma_db, the command index: the legacy base collection, its
  pointer naming a versioned staging collection, that staging collection,
  and home_assistant_routines with the text of two dictated routines, one
  of which no longer exists anywhere else;
* data/auth.db with the retired user in retired_users (the admission
  column contract of that table).

``legacy=True`` adds the legacy 'user_memories' collection to data/memory_db
(LEGACY in a document and LEGACY_META in a metadata value, as the pre-policy
LongTermMemory wrote them); ``extra`` adds one record (document EXTRA) to each
named (store, collection), e.g. a look-alike of a dropped name.
Every distinctive string starts with CANARY so raw-byte scans can find it.
"""
from __future__ import annotations

CANARY = "sc4c-canary"
PERSON = "user_" + "a" * 32
LIVE = CANARY + "-live-fact"
OVERDUE = CANARY + "-overdue-fact"
RETIRED = CANARY + "-retired-owner-fact"
TOMBSTONED = CANARY + "-tombstoned-fact"
ORPHAN = CANARY + "-orphan-record"
ROUTINE = CANARY + "-routine-text"
ROUTINE_GONE = CANARY + "-routine-gone"
COMMAND = CANARY + "-command-entity"
LEGACY = CANARY + "-legacy-memory"
LEGACY_META = CANARY + "-legacy-metadata"
EXTRA = CANARY + "-extra-collection"
STAGING = "home_assistant_commands__20261116000000abcdef"
POLICY = "user_memories_policy_v1"
POLICY_METADATA = {"description": "Policy-managed memory index", "fact_policy": 1}
EMBEDDING = [0.25, 0.5, 0.75]


def _extra(store: str, extra: tuple[tuple[str, str], ...]) -> list[str]:
    return [f"client.get_or_create_collection(name={name!r}).add(ids=['extra-1'], documents=[{EXTRA!r}],"
            f" embeddings=[{EMBEDDING!r}])" for where, name in extra if where == store]


def chroma_state(*, legacy: bool = False, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """Child code for the world's first run (see the module doc)."""
    return "\n".join([
        "import sqlite3, time, json",
        "from contextlib import closing",
        "from uuid import uuid4",
        "import chromadb",
        "from chromadb.config import Settings",
        "from src.memory import fact_ledger as fl",
        "from src.memory.erasure_journal import ErasureParticipant",
        "now = time.time()",
        "def source(at, conversation=None):",
        "    return fl.MemorySource(str(uuid4()), conversation or str(uuid4()), at - 1, at + 86400.0,",
        "                           fl.MemoryOrigin.VOICE, ())",
        "ledger = fl.FactLedger(str(state / 'data' / 'privacy.db'))",
        "ledger.initialize()",
        f"live = ledger.store({LIVE!r}, 'fact', .9, (source(now),))",
        f"overdue = ledger.store({OVERDUE!r}, 'fact', .9, (source(now),))",
        f"owned = ledger.store({RETIRED!r}, 'fact', .9, (source(now),))",
        f"ledger.assign_owner(owned.id, {PERSON!r}, owned.revision, 0, fl.MemoryAssigner('synthetic-operator'),",
        "                     retired=lambda ids: False, active=lambda person: True)",
        "owned = ledger.get(owned.id)",
        "conversation = str(uuid4())",
        f"gone = ledger.store({TOMBSTONED!r}, 'fact', .9, (source(now, conversation),))",
        "ledger.delete_conversation(conversation, fl.MemoryReviewer('synthetic-reviewer'))",
        "ledger.erasures.register_target(ErasureParticipant.SEMANTIC_INDEX, 1,",
        f"    {{'path': str(state / 'data' / 'memory_db'), 'collection': {POLICY!r}}})",
        "ledger.close()",
        "# The expiry passed after H's last sweep (as tools/sc0/residue.owned_fact_expired plants it).",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db')) as db, db:",
        "    db.execute('UPDATE facts SET expires_at=? WHERE id=?', (now - 60, overdue.id))",
        "with closing(sqlite3.connect(state / 'data' / 'auth.db')) as db, db:",
        "    db.execute('CREATE TABLE retired_users (user_id TEXT, retired_at REAL)')",
        f"    db.execute('INSERT INTO retired_users VALUES (?,?)', ({PERSON!r}, now - 60))",
        "settings = Settings(anonymized_telemetry=False)",
        "client = chromadb.PersistentClient(path=str(state / 'data' / 'memory_db'), settings=settings)",
        f"index = client.get_or_create_collection(name={POLICY!r}, metadata={POLICY_METADATA!r})",
        "facts = [live, overdue, owned, gone]",
        "index.add(ids=[f.id for f in facts] + ['orphan-1'],",
        f"    documents=[f.content for f in facts] + [{ORPHAN!r}],",
        "    metadatas=[{'fact_policy': 1, 'revision': f.revision} for f in facts] + [{'fact_policy': 1, 'revision': 1}],",
        f"    embeddings=[{EMBEDDING!r}] * 5)",
        *([f"client.get_or_create_collection(name='user_memories').add(ids=['legacy-1', 'legacy-2'],"
           f" documents=[{LEGACY + '-1'!r}, {LEGACY + '-2'!r}],"
           f" metadatas=[{{'user_id': 'legacy-user', 'note': {LEGACY_META!r}}}, {{'user_id': 'legacy-user'}}],"
           f" embeddings=[{EMBEDDING!r}] * 2)"] if legacy else []),
        *_extra("data/memory_db", extra),
        "client.close()",
        "client = chromadb.PersistentClient(path=str(state / 'data' / 'chroma_db'), settings=settings)",
        "base = client.get_or_create_collection(name='home_assistant_commands', metadata={'hnsw:space': 'cosine'})",
        f"base.add(ids=['light.sala_0'], documents=[{COMMAND + '-base'!r}], metadatas=[{{'entity_id': 'light.sala'}}],",
        f"    embeddings=[{EMBEDDING!r}])",
        f"staging = client.create_collection(name={STAGING!r}, metadata={{'hnsw:space': 'cosine'}})",
        "staging.add(ids=['light.sala_0', 'light.cocina_0'],",
        f"    documents=[{COMMAND + '-sala'!r}, {COMMAND + '-cocina'!r}],",
        "    metadatas=[{'entity_id': 'light.sala', 'area': 'sala'}, {'entity_id': 'light.cocina', 'n': 2}],",
        f"    embeddings=[{EMBEDDING!r}, [0.5, 0.25, 0.125]])",
        "pointer = client.get_or_create_collection(name='home_assistant_commands__pointer')",
        f"pointer.modify(metadata={{'active': {STAGING!r}}})",
        "routines = client.get_or_create_collection(name='home_assistant_routines', metadata={'hnsw:space': 'cosine'})",
        f"routines.add(ids=['r-live_0', 'r-gone_0'], documents=[{ROUTINE!r}, {ROUTINE_GONE!r}],",
        "    metadatas=[{'routine_id': 'r-live', 'full_config': json.dumps({'name': " + repr(ROUTINE) + "})},",
        "               {'routine_id': 'r-gone', 'full_config': json.dumps({'name': " + repr(ROUTINE_GONE) + "})}],",
        f"    embeddings=[{EMBEDDING!r}, {EMBEDDING!r}])",
        *_extra("data/chroma_db", extra),
        "client.close()",
    ])
