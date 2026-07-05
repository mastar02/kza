"""Tests: Distiller — extracción de hechos con LLM local y marcado."""
import asyncio
import json
import logging

from src.ambient.distiller import Distiller, _parse_facts, make_langid_fn


def test_langid_distingue_es_de_en():
    detect = make_langid_fn()
    lang_es, prob_es = detect("hola, ¿cómo andás? acordate de comprar el pan")
    lang_en, prob_en = detect("what do you think about it, I really don't know")
    assert lang_es == "es"
    assert lang_en == "en"
    assert 0.0 <= prob_es <= 1.0 and 0.0 <= prob_en <= 1.0


def test_langid_texto_vacio_devuelve_unknown():
    detect = make_langid_fn()
    assert detect("") == ("unknown", 0.0)
    assert detect("   ") == ("unknown", 0.0)


class FakeStore:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []
        self.last_min_vad = None
        self.last_spanish_only = None

    async def undistilled_live(self, limit=200, min_vad_prob=0.0, spanish_only=False):
        self.last_min_vad = min_vad_prob
        self.last_spanish_only = spanish_only
        rows = [r for r in self.rows if (r.get("vad_prob") or 0) >= min_vad_prob]
        if spanish_only:
            rows = [r for r in rows if r.get("lang_ok", 1) in (1, None)]
        return rows[:limit]

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


def test_distill_once_propaga_spanish_only_al_store():
    # A (flag-no-drop): el distiller consume SOLO lo conservable (lang_ok) vía el
    # filtro spanish_only del store. El no-español no se dropea — queda en la DB.
    rows = [_row(1, "hola che todo bien")]
    store = FakeStore(rows)

    async def fake_chat(prompt):
        return "[]"

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, spanish_only=True)
    asyncio.run(d.distill_once())
    assert store.last_spanish_only is True


def test_distill_once_loguea_idioma_shadow(caplog):
    # Shadow: detecta idioma del batch y lo loguea, SIN filtrar (vad-only gate).
    rows = [_row(1, "hola que tal todo bien"), _row(2, "what do you think man")]
    store = FakeStore(rows)

    async def fake_chat(prompt):
        return "[]"

    def fake_lang(text):
        return ("es", 0.99) if "hola" in text else ("en", 0.99)

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, lang_detect_fn=fake_lang)
    with caplog.at_level(logging.INFO):
        asyncio.run(d.distill_once())
    text = caplog.text.lower()
    assert "idioma" in text
    assert "'es': 1" in caplog.text and "'en': 1" in caplog.text
    # No filtra: ambas filas se procesan/marcan igual
    assert sorted(store.marked) == [1, 2]


def test_distill_once_dropea_ingles_con_confianza_alta():
    # El inglés es bleed de TV (nunca comando/hogar). Gate enforced: dropea
    # 'en' confiado del prompt, conserva español y 'en' dudoso (okay/yeah).
    rows = [_row(1, "che, comprá pan mañana"),        # es
            _row(2, "what are we watching tonight"),   # en alta conf → drop
            _row(3, "okay")]                           # en baja conf → keep
    store = FakeStore(rows)
    captured = {}

    async def fake_chat(prompt):
        captured["prompt"] = prompt
        return "[]"

    def fake_lang(text):
        if text.startswith("che"):
            return ("es", 0.99)
        if text == "okay":
            return ("en", 0.40)        # baja confianza → NO dropear
        return ("en", 0.97)            # alta confianza → dropear

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, lang_detect_fn=fake_lang,
                  drop_language="en", drop_language_min_prob=0.9)
    asyncio.run(d.distill_once())
    assert "comprá pan" in captured["prompt"]            # español → al LLM
    assert "okay" in captured["prompt"]                  # 'en' dudoso → al LLM
    assert "what are we watching" not in captured["prompt"]  # 'en' confiado → dropeado
    # todas marcadas (kept + dropped) — los dropeados no se reprocesan
    assert sorted(store.marked) == [1, 2, 3]


def test_distill_once_batch_todo_ingles_no_llama_llm():
    rows = [_row(1, "what is this"), _row(2, "look at that")]
    store = FakeStore(rows)

    async def fake_chat(prompt):
        raise AssertionError("no debe llamarse: todo el batch es inglés dropeado")

    d = Distiller(store=store, chat_fn=fake_chat, store_fact_fn=lambda *a, **k: "x",
                  interval_hours=6, min_batch=1, lang_detect_fn=lambda t: ("en", 0.98),
                  drop_language="en", drop_language_min_prob=0.9)
    assert asyncio.run(d.distill_once()) == 0
    assert sorted(store.marked) == [1, 2]  # marcadas igual (no reprocesar)


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
