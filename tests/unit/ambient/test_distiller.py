"""Tests: Distiller — extracción de hechos con LLM local y marcado."""
import asyncio
import json
import logging

from src.ambient.distiller import Distiller, _parse_facts, make_langid_fn


def test_langid_distingue_es_de_en():
    detect = make_langid_fn()
    assert detect("hola, ¿cómo andás? acordate de comprar el pan") == "es"
    assert detect("what do you think about it, I really don't know") == "en"


def test_langid_texto_vacio_devuelve_unknown():
    detect = make_langid_fn()
    assert detect("") == "unknown"
    assert detect("   ") == "unknown"


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []
        self.last_min_vad = None

    async def undistilled_live(self, limit=200, min_vad_prob=0.0):
        self.last_min_vad = min_vad_prob
        return [r for r in self.rows if (r.get("vad_prob") or 0) >= min_vad_prob][:limit]

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


def test_distill_once_aplica_min_vad_prob_al_store():
    # El umbral de calidad configurado se propaga al query del store.
    rows = [_row(1, "voz cerca"), _row(2, "otra cosa")]
    store = FakeStore(rows)

    async def fake_chat(prompt):
        return "[]"

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, min_vad_prob=0.45)
    asyncio.run(d.distill_once())
    assert store.last_min_vad == 0.45


def test_distill_once_loguea_idioma_shadow(caplog):
    # Shadow: detecta idioma del batch y lo loguea, SIN filtrar (vad-only gate).
    rows = [_row(1, "hola que tal todo bien"), _row(2, "what do you think man")]
    store = FakeStore(rows)

    async def fake_chat(prompt):
        return "[]"

    def fake_lang(text):
        return "es" if "hola" in text else "en"

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, lang_detect_fn=fake_lang)
    with caplog.at_level(logging.INFO):
        asyncio.run(d.distill_once())
    text = caplog.text.lower()
    assert "idioma" in text
    assert "'es': 1" in caplog.text and "'en': 1" in caplog.text
    # No filtra: ambas filas se procesan/marcan igual
    assert sorted(store.marked) == [1, 2]


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


def test_make_local_chat_fn_envia_bearer_token(monkeypatch):
    # El distiller habla con el LLM local :8101, que exige auth (bearer
    # compartido LLAMA_API_KEY desde 2026-04-30). Sin el header → 401.
    import http.server
    import threading

    from src.ambient.distiller import make_local_chat_fn

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = b'{"choices":[{"message":{"content":"[]"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("LLAMA_API_KEY", "secret-token-xyz")
        chat = make_local_chat_fn(llm_url=f"http://127.0.0.1:{port}/v1")
        out = asyncio.run(chat("hola"))
    finally:
        srv.shutdown()
    assert out == "[]"
    assert seen["auth"] == "Bearer secret-token-xyz"


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
