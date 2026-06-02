#!/usr/bin/env python3
"""
Benchmark comparativo: español natural + tool calling + razonamiento + velocidad.
Uso: python3 llm-bench-es.py <label> [host:port]
  label: tag para guardar resultados (ej: "glm-air-baseline" o "gpt-oss-medium")
  host:port: endpoint llama-server (default 127.0.0.1:8200)
Salida: /home/kza/kza/logs/bench-<label>.json + tabla legible en stdout.
"""
import json, sys, time, urllib.request, urllib.error, statistics, os

LABEL = sys.argv[1] if len(sys.argv) > 1 else "test"
HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:8200"
URL = f"http://{HOST}/v1/chat/completions"
KEY = open("/home/kza/secrets/llama-api-key").read().strip()
OUT = f"/home/kza/kza/logs/bench-{LABEL}.json"


def call(messages, max_t=400, tools=None, extra=None):
    body = {
        "model": "x",
        "messages": messages,
        "max_tokens": max_t,
        "temperature": 0.2,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        wall = (time.time() - t0) * 1000
        msg = d["choices"][0]["message"]
        t = d.get("timings", {})
        return {
            "ok": True,
            "wall_ms": round(wall),
            "tg_tps": round(t.get("predicted_per_second", 0), 1),
            "pp_tps": round(t.get("prompt_per_second", 0), 1),
            "n_pred": t.get("predicted_n", 0),
            "n_prompt": t.get("prompt_n", 0),
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish": d["choices"][0].get("finish_reason"),
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# === EJE 1: español natural rioplatense ===
ES_PROMPTS = [
    "Hola, ¿cómo andás? Contame algo interesante que sepas sobre el barrio de Palermo en Buenos Aires.",
    "Si tuvieras que recomendarme un libro corto en español que se lea en una tarde, ¿cuál sería y por qué?",
    "Explicame en dos oraciones qué es el voseo, usando vos vos mismo en la respuesta.",
    "Escribime un mensaje de WhatsApp informal para avisarle a un amigo que llego tarde por el tráfico.",
    "Resumime en tres líneas la diferencia entre 'tomar' y 'agarrar' en español rioplatense.",
]

# === EJE 2: tool calling con web_search ===
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Busca información actual en internet. Útil para clima, noticias, eventos recientes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Términos de búsqueda en lenguaje natural"},
                "lang": {"type": "string", "description": "Código ISO de idioma (es, en, pt). Default es."},
            },
            "required": ["query"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtiene el clima actual o pronóstico para una ciudad.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Nombre de la ciudad"},
                "when": {"type": "string", "description": "'now', 'tomorrow', o fecha ISO YYYY-MM-DD"},
            },
            "required": ["city"],
        },
    },
}]

TOOL_PROMPTS = [
    "¿Qué clima va a hacer mañana en Buenos Aires?",
    "Buscame las últimas noticias sobre fútbol argentino de hoy.",
    "¿Cuál es la temperatura actual en Córdoba?",
]

# === EJE 3: razonamiento en español ===
REASONING_PROMPTS = [
    "Juan tiene el doble de la edad de Pedro. La suma de sus edades es 36. ¿Cuántos años tiene cada uno? Razoná paso a paso en español.",
    "Si un tren sale de Rosario a las 8:00 AM a 80 km/h hacia Buenos Aires (300 km), ¿a qué hora llega? Mostrá el cálculo en español.",
    "¿Por qué el cielo es azul? Explicá la razón física en español, en máximo 5 oraciones.",
]


results = {"label": LABEL, "host": HOST, "started": time.time(), "axes": {}}

# ---------- EJE 1 ----------
section(f"[{LABEL}] EJE 1: español natural rioplatense")
es_results = []
for i, p in enumerate(ES_PROMPTS, 1):
    r = call([
        {"role": "system", "content": "Sos un asistente argentino. Respondé siempre en español rioplatense (vos, no tú). Sé breve y natural."},
        {"role": "user", "content": p},
    ], max_t=200)
    es_results.append({"prompt": p, **r})
    if r["ok"]:
        print(f"\n[{i}] {p}")
        print(f"    wall={r['wall_ms']}ms  tg={r['tg_tps']}t/s  ntok={r['n_pred']}")
        print(f"    > {r['content'][:240].strip()}")
    else:
        print(f"[{i}] ERROR: {r['error']}")
results["axes"]["es_natural"] = es_results

# ---------- EJE 2 ----------
section(f"[{LABEL}] EJE 2: tool calling")
tool_results = []
for i, p in enumerate(TOOL_PROMPTS, 1):
    r = call([
        {"role": "system", "content": "Sos un asistente argentino. Tenés acceso a tools. Si necesitás info actual (clima, noticias), usá la tool apropiada. Respondé siempre en español rioplatense."},
        {"role": "user", "content": p},
    ], max_t=200, tools=TOOLS)
    tool_results.append({"prompt": p, **r})
    if r["ok"]:
        tc = r["tool_calls"]
        emitted = "YES" if tc else "no"
        tc_summary = ""
        if tc:
            f = tc[0].get("function", {})
            tc_summary = f"{f.get('name')}({f.get('arguments', '')[:120]})"
        print(f"\n[{i}] {p}")
        print(f"    wall={r['wall_ms']}ms  tool_call={emitted}  finish={r['finish']}")
        if tc_summary:
            print(f"    > tool: {tc_summary}")
        if r["content"]:
            print(f"    > text: {r['content'][:160].strip()}")
    else:
        print(f"[{i}] ERROR: {r['error']}")
results["axes"]["tool_calling"] = tool_results

# ---------- EJE 3 ----------
section(f"[{LABEL}] EJE 3: razonamiento en español")
reason_results = []
for i, p in enumerate(REASONING_PROMPTS, 1):
    r = call([
        {"role": "system", "content": "Sos un asistente que razona en español. Respondé siempre en español, nunca cambies de idioma."},
        {"role": "user", "content": p},
    ], max_t=400)
    reason_results.append({"prompt": p, **r})
    if r["ok"]:
        print(f"\n[{i}] {p[:80]}...")
        print(f"    wall={r['wall_ms']}ms  tg={r['tg_tps']}t/s  ntok={r['n_pred']}")
        print(f"    > {r['content'][:300].strip()}")
        if r["reasoning"]:
            print(f"    [reasoning {len(r['reasoning'])} chars hidden]")
    else:
        print(f"[{i}] ERROR: {r['error']}")
results["axes"]["reasoning"] = reason_results

# ---------- AGREGADOS DE VELOCIDAD ----------
section(f"[{LABEL}] AGREGADOS")
all_ok = [r for axis in results["axes"].values() for r in axis if r.get("ok")]
if all_ok:
    tgs = [r["tg_tps"] for r in all_ok if r["tg_tps"] > 0]
    walls = [r["wall_ms"] for r in all_ok]
    print(f"runs ok          : {len(all_ok)}/{sum(len(a) for a in results['axes'].values())}")
    print(f"TG tok/s  median : {statistics.median(tgs):.1f}  mean: {statistics.mean(tgs):.1f}")
    print(f"wall_ms   median : {statistics.median(walls):.0f}  max: {max(walls)}")
    tool_calls_emitted = sum(1 for r in results["axes"]["tool_calling"] if r.get("ok") and r.get("tool_calls"))
    print(f"tool calls emit  : {tool_calls_emitted}/{len(results['axes']['tool_calling'])}")
results["finished"] = time.time()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nGuardado: {OUT}")
