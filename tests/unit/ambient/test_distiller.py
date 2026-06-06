"""Tests: Distiller — extracción de hechos con LLM local y marcado."""
import asyncio
import json

from src.ambient.distiller import Distiller, _parse_facts


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []

    async def undistilled_live(self, limit=200):
        return self.rows[:limit]

    async def mark_distilled(self, ids):
        self.marked.extend(ids)


def _row(i, text):
    return {"id": i, "room_id": "escritorio", "t0": 1000.0 + i, "t1": 1002.0 + i,
            "text": text, "speaker": "gabriel", "source": "live"}


def test_distill_once_extracts_and_marks():
    rows = [_row(1, "che acordate que el viernes viene el plomero"),
            _row(2, "me encanta la pizza de acá")]
    store = FakeStore(rows)
    facts_out = []

    async def fake_chat(prompt: str) -> str:
        assert "plomero" in prompt
        return json.dumps([
            {"fact": "El viernes viene el plomero", "category": "fact", "confidence": 0.9},
            {"fact": "Le gusta la pizza del lugar habitual", "category": "preference", "confidence": 0.7},
        ])

    def fake_store_fact(fact, category, confidence=0.8, metadata=None):
        facts_out.append((fact, category, confidence, metadata))
        return f"fact_{len(facts_out)}"

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=fake_store_fact,
                  interval_hours=6, min_batch=1)
    n = asyncio.run(d.distill_once())
    assert n == 2
    assert store.marked == [1, 2]
    assert facts_out[0][1] == "fact"
    assert facts_out[1][1] == "preference"
    # metadata referencia el origen ambiental
    assert facts_out[0][3]["origin"] == "ambient"


def test_distill_below_min_batch_is_noop():
    store = FakeStore([_row(1, "hola")])

    async def fake_chat(prompt):
        raise AssertionError("no debe llamarse con batch < min_batch")

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=5)
    assert asyncio.run(d.distill_once()) == 0
    assert store.marked == []


def test_llm_error_does_not_mark():
    store = FakeStore([_row(1, "a"), _row(2, "b")])

    async def broken_chat(prompt):
        raise RuntimeError("LLM caído")

    d = Distiller(store=store, chat_fn=broken_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1)
    assert asyncio.run(d.distill_once()) == 0
    assert store.marked == []  # sin marcar: se reintenta el próximo ciclo


def test_parse_facts_tolerates_fences_and_garbage():
    raw = "```json\n[{\"fact\": \"X\", \"category\": \"fact\", \"confidence\": 0.8}]\n```"
    assert _parse_facts(raw) == [{"fact": "X", "category": "fact", "confidence": 0.8}]
    assert _parse_facts("no es json") == []
    assert _parse_facts("[]") == []
    # categorías inválidas se descartan, válidas quedan
    mixed = json.dumps([
        {"fact": "ok", "category": "preference", "confidence": 0.9},
        {"fact": "bad", "category": "alien", "confidence": 0.9},
        {"category": "fact", "confidence": 0.9},  # sin fact
    ])
    assert len(_parse_facts(mixed)) == 1
