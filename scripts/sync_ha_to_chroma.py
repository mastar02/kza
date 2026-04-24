"""
Sync HA → ChromaDB con auto-discovery de capabilities.

Descubre por cada entity light: onoff, brightness, color_temp, color, effect.
Genera frases por capability × preset (ej: brightness × 6 niveles, color × 6 tonos).
Usa vLLM compartido (7B) por HTTP — R1. Flag --use-72b si se quiere consistencia vía Q8.

Metadata en Chroma:
  service         : turn_on|turn_off
  capability      : onoff|brightness|color_temp|color|effect
  service_data    : JSON string con el dict a pasar al service HA (ej: {"brightness_pct": 50})
  entity_id, domain, area, cache_key, is_group

Runtime (src/pipeline/request_router.py) puede sobreescribir service_data con valores
extraídos por slot extractor de la query original (ver Sprint 2.2).

Uso:
  python scripts/sync_ha_to_chroma.py --entity light.escritorio --dry-run
  python scripts/sync_ha_to_chroma.py                                       # full sync
  python scripts/sync_ha_to_chroma.py --include-individual                  # + bombitas l1..
  python scripts/sync_ha_to_chroma.py --use-72b                             # LLM local Q8_0
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_ha")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

HA_URL = os.environ["HOME_ASSISTANT_URL"].rstrip("/")
HA_TOKEN = os.environ["HOME_ASSISTANT_TOKEN"]


# ============================================================
# HA client helpers
# ============================================================
def ha_get(path: str):
    req = urllib.request.Request(HA_URL + path, headers={"Authorization": f"Bearer {HA_TOKEN}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


def ha_template(tpl: str) -> str:
    data = json.dumps({"template": tpl}).encode()
    req = urllib.request.Request(
        HA_URL + "/api/template",
        data=data,
        headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=15).read().decode()


def resolve_areas(entity_ids: list[str]) -> dict[str, str | None]:
    lines = "\n".join(f'  "{eid}",' for eid in entity_ids)
    tpl = (
        "{% set eids = [\n" + lines + "\n] %}"
        "{% for eid in eids %}{{ eid }}|{{ area_name(eid) or '' }}\n{% endfor %}"
    )
    raw = ha_template(tpl)
    out: dict[str, str | None] = {}
    for line in raw.splitlines():
        if "|" in line:
            eid, area = line.strip().split("|", 1)
            out[eid] = area or None
    return out


# ============================================================
# Convención de naming (ver memoria del proyecto)
# ============================================================
GROUP_PREFIX_MAP = {
    "l": "living", "c": "cocina", "e": "escritorio",
    "b": "bano", "p": "pasillo", "cu": "cuarto",
}
KNOWN_GROUPS = {
    "light.living", "light.cocina", "light.escritorio", "light.bano",
    "light.pasillo", "light.cuarto", "light.balcon", "light.hogar",
    "light.escalera", "light.escaleras",
}
INDIVIDUAL_RE = re.compile(r"^([a-z]{1,2})(\d+)(?:_\d+)?$", re.IGNORECASE)


def decode_individual(entity_id: str) -> tuple[str, str] | None:
    suffix = entity_id.split(".", 1)[1]
    m = INDIVIDUAL_RE.fullmatch(suffix)
    if not m:
        return None
    prefix, num = m.group(1).lower(), m.group(2)
    room = GROUP_PREFIX_MAP.get(prefix)
    return (room, num) if room else None


def is_group_entity(entity_id: str, friendly_name: str) -> bool:
    return entity_id in KNOWN_GROUPS


def cache_key(entity_id: str, friendly_name: str, area: str | None, capability: str, value: str) -> str:
    """Cache key incluye capability+value para soportar indexación incremental granular."""
    raw = f"{entity_id}|{friendly_name}|{area or ''}|{capability}|{value}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ============================================================
# Capability discovery (per-entity)
# ============================================================
@dataclass
class CommandSpec:
    """Una variante concreta de comando para una entidad."""
    service: str                                   # turn_on|turn_off
    capability: str                                # onoff|brightness|color_temp|color|effect
    value_label: str                               # "50%", "cálida", "rojo", ...
    service_data: dict                             # {"brightness_pct": 50}, etc.
    prompt_hint: str = ""                          # extra hint para el LLM


BRIGHTNESS_PRESETS = [
    ("tenue", 15),
    ("al 25%", 25),
    ("al 50%", 50),
    ("al 75%", 75),
    ("al 100%", 100),
    ("máxima", 100),
]

# RGB triples son orientativos; HA acepta rgb_color=[R,G,B]
COLOR_PRESETS = [
    ("rojo", [255, 0, 0]),
    ("verde", [0, 255, 0]),
    ("azul", [0, 0, 255]),
    ("amarillo", [255, 255, 0]),
    ("rosa", [255, 105, 180]),
    ("violeta", [148, 0, 211]),
    ("naranja", [255, 128, 0]),
    ("celeste", [135, 206, 235]),
]

# Temperatura de color en Kelvin; HA acepta color_temp_kelvin
TEMP_PRESETS_K = [
    ("cálida", 2700),
    ("neutra", 4000),
    ("fría", 6500),
]


def discover_capabilities(entity: dict) -> list[str]:
    """Devuelve capabilities soportadas: onoff, brightness, color_temp, color, effect."""
    attrs = entity.get("attributes", {}) or {}
    caps = ["onoff"]
    color_modes = set(attrs.get("supported_color_modes") or [])
    NON_ONOFF = {"brightness", "color_temp", "hs", "xy", "rgb", "rgbw", "rgbww", "white"}
    if color_modes & NON_ONOFF:
        caps.append("brightness")
    if "color_temp" in color_modes:
        caps.append("color_temp")
    if color_modes & {"hs", "xy", "rgb", "rgbw", "rgbww"}:
        caps.append("color")
    if attrs.get("effect_list"):
        caps.append("effect")
    return caps


def build_command_specs(entity: dict, capabilities: list[str], max_effects: int = 4) -> list[CommandSpec]:
    """Genera lista de CommandSpecs concretos para esta entidad."""
    specs: list[CommandSpec] = []
    # onoff siempre
    specs.append(CommandSpec("turn_on", "onoff", "prender", {}))
    specs.append(CommandSpec("turn_off", "onoff", "apagar", {}))
    if "brightness" in capabilities:
        for label, pct in BRIGHTNESS_PRESETS:
            specs.append(CommandSpec("turn_on", "brightness", label, {"brightness_pct": pct}))
    if "color_temp" in capabilities:
        for label, kelvin in TEMP_PRESETS_K:
            specs.append(CommandSpec("turn_on", "color_temp", label, {"color_temp_kelvin": kelvin},
                                      prompt_hint=f"{kelvin}K"))
    if "color" in capabilities:
        for label, rgb in COLOR_PRESETS:
            specs.append(CommandSpec("turn_on", "color", label, {"rgb_color": rgb}))
    if "effect" in capabilities:
        effects = (entity.get("attributes", {}).get("effect_list") or [])[:max_effects]
        for effect_name in effects:
            specs.append(CommandSpec("turn_on", "effect", effect_name, {"effect": effect_name}))
    return specs


# ============================================================
# Prompts
# ============================================================
PROMPT_BASE_GROUP = """Generá frases de voz naturales en español argentino para un comando de domótica.

Entidad: {friendly_name} (grupo de luces)
Habitación: {area}
Servicio: {service}
Detalle: {detail}

Reglas:
- Variá entre "vos" (prendé, apagá, poné) y "tú" (prende, apaga, pon) ~50/50.
- Vocabulario estándar; sin modismos inventados.
- 4 frases por respuesta.
- NO incluyas "por favor" ni saludos.
{extra_rules}
Respondé SOLO un array JSON: ["frase 1", "frase 2", "frase 3", "frase 4"]"""

PROMPT_BASE_INDIVIDUAL = """Generá frases de voz naturales en español argentino para controlar UNA bombita específica de una lámpara multi-bombita.

Entidad: {friendly_name}
Código: {code} = bombita número {number} de la habitación "{room}".
Servicio: {service}
Detalle: {detail}

Reglas:
- Variá vos/tú ~50/50. Sin modismos. Sin "por favor".
- 4 frases.
- Formas útiles: "bombita {number} del {room}", "la {code} del {room}", "luz {number} {room}".
{extra_rules}
Respondé SOLO array JSON: ["...", "...", "...", "..."]"""


def capability_prompt_detail(spec: CommandSpec) -> tuple[str, str]:
    """Devuelve (detail_text, extra_rules) para el prompt según la capability."""
    if spec.capability == "onoff" and spec.service == "turn_on":
        return ("prender (encender) la luz",
                "- 2 frases mencionando 'luz/luces' explícito, 2 frases elípticas sin decir 'luz'.\n"
                "- Verbos: prender/encender/iluminar.")
    if spec.capability == "onoff" and spec.service == "turn_off":
        return ("apagar la luz",
                "- 2 frases mencionando 'luz/luces' explícito, 2 elípticas sin 'luz'.\n"
                "- Verbos: apagar/cortar.")
    if spec.capability == "brightness":
        return (f"prender la luz con intensidad {spec.value_label}",
                f"- Usar la frase '{spec.value_label}' tal cual o sinónimo cercano.\n"
                "- Incluir variantes con porcentaje o con adjetivo según corresponda (tenue/suave/medio/fuerte/máximo).")
    if spec.capability == "color_temp":
        return (f"prender la luz con tono {spec.value_label} ({spec.prompt_hint})",
                f"- Usar '{spec.value_label}' como adjetivo natural para temperatura de luz.\n"
                "- Alternar con 'luz blanca', 'luz amarilla', 'luz fría' según corresponda.")
    if spec.capability == "color":
        return (f"poner la luz color {spec.value_label}",
                f"- Usar '{spec.value_label}' como color. Verbos: poner, cambiar a, pintar, color.\n"
                "- No inventar colores que no sean el pedido.")
    if spec.capability == "effect":
        return (f"activar el efecto '{spec.value_label}'",
                f"- Usar 'efecto {spec.value_label}' o el nombre del efecto directo.")
    return (spec.capability, "")


# ============================================================
# LLM client (vLLM HTTP por default, llama-cpp local con --use-72b)
# ============================================================
class VLLMClient:
    """Cliente OpenAI contra vLLM compartido :8100 (infra)."""
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-used", timeout=timeout)
        models = [m.id for m in self.client.models.list().data]
        if model not in models:
            raise RuntimeError(f"Modelo {model} no disponible. Catálogo: {models}")
        logger.info(f"VLLMClient OK → {base_url} (modelo: {model})")

    def complete(self, prompt: str, max_tokens: int = 400, temperature: float = 0.5) -> str:
        # Qwen Instruct → chat completions con system prompt conciso.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "Sos un generador de frases de voz en español argentino para un asistente domótico. "
                            "Respondés SIEMPRE con UN array JSON de strings, sin markdown, sin explicaciones."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class LocalLLMClient:
    """Wrapper del LLMReasoner local (Q8_0/Q6_K) — para --use-72b."""
    def __init__(self, model_path: str):
        from src.llm.reasoner import LLMReasoner
        self.reasoner = LLMReasoner(
            model_path=model_path, n_ctx=8192, n_batch=512, n_threads=24,
            n_gpu_layers=0, chat_format="chatml", rope_freq_base=1000000.0,
        )
        self.reasoner.load()
        logger.info("LocalLLMClient (72B) listo")

    def complete(self, prompt: str, max_tokens: int = 400, temperature: float = 0.5) -> str:
        result = self.reasoner(prompt, max_tokens=max_tokens, temperature=temperature,
                                stop=["\n\n[", "\n```", "\nNota:", "\n---"])
        return result["choices"][0]["text"]


def parse_json_array(text: str) -> list[str] | None:
    """Parser robusto para el primer array JSON en el texto."""
    j0 = text.find("[")
    if j0 < 0:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(text[j0:])
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except json.JSONDecodeError:
        pass
    return None


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="light")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wipe", action="store_true",
                    help="Borra la colección home_assistant_commands antes de indexar")
    ap.add_argument("--include-individual", action="store_true")
    ap.add_argument("--only-individual", action="store_true")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--embedder-device", default="cuda:0")
    ap.add_argument("--use-72b", action="store_true",
                    help="Usar 72B local en lugar de vLLM 7B por HTTP (más lento ~100s/prompt)")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--vllm-model", default="qwen2.5-7b-awq")
    ap.add_argument("--model-path",
                    default=str(ROOT / "models/Qwen2.5-72B-Instruct-Q8_0/Qwen2.5-72B-Instruct-Q8_0-00001-of-00002.gguf"))
    args = ap.parse_args()

    # Fetch + classify
    logger.info("Fetching HA states...")
    states = ha_get("/api/states")
    dom_entities = [e for e in states if e["entity_id"].startswith(f"{args.domain}.")]
    logger.info(f"Dominio {args.domain}: {len(dom_entities)} entidades")

    entity_ids = [e["entity_id"] for e in dom_entities]
    logger.info("Resolviendo areas...")
    areas = resolve_areas(entity_ids)

    selected = []
    for e in dom_entities:
        eid = e["entity_id"]
        if args.entity and eid != args.entity:
            continue
        fname = e["attributes"].get("friendly_name", eid)
        area = areas.get(eid)
        is_group = is_group_entity(eid, fname)
        indiv = decode_individual(eid)
        if args.only_individual and is_group:
            continue
        if not args.include_individual and not args.only_individual and not is_group:
            continue
        caps = discover_capabilities(e)
        selected.append({
            "entity_id": eid, "friendly_name": fname, "area": area,
            "is_group": is_group, "individual": indiv, "capabilities": caps,
            "entity_state": e,
        })

    if args.limit:
        selected = selected[: args.limit]

    logger.info(f"Seleccionadas {len(selected)} entidades:")
    for s in selected:
        tag = "GROUP" if s["is_group"] else f"INDIV n={s['individual'][1]} room={s['individual'][0]}"
        logger.info(f"  {s['entity_id']:40} | caps={s['capabilities']} | {tag}")

    if not selected:
        logger.warning("Nada que indexar. Saliendo.")
        return

    # Embedder (antes que llama-cpp-python por bug libcudart)
    embedder = None
    if not args.dry_run:
        logger.info(f"Cargando BGE-M3 en {args.embedder_device}...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-m3", device=args.embedder_device)

    # LLM
    if args.use_72b:
        llm = LocalLLMClient(args.model_path)
    else:
        llm = VLLMClient(args.vllm_url, args.vllm_model)

    # Chroma
    import chromadb
    chroma_path = str(ROOT / "data/chroma_db")
    logger.info(f"Abriendo Chroma en {chroma_path}")
    client = chromadb.PersistentClient(path=chroma_path)
    if args.wipe:
        try:
            client.delete_collection("home_assistant_commands")
            logger.info("Colección home_assistant_commands borrada (--wipe)")
        except Exception as e:
            logger.debug(f"delete_collection: {e}")
    collection = client.get_or_create_collection(
        name="home_assistant_commands",
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.get(include=["metadatas"])
    existing_keys = {m.get("cache_key") for m in (existing.get("metadatas") or []) if m.get("cache_key")}
    logger.info(f"Cache keys ya indexados: {len(existing_keys)}")

    # Build command specs + filter incremental
    total_specs, to_process = 0, []
    for s in selected:
        specs = build_command_specs(s["entity_state"], s["capabilities"])
        for spec in specs:
            total_specs += 1
            key = cache_key(s["entity_id"], s["friendly_name"], s["area"], spec.capability, spec.value_label)
            if key in existing_keys and not args.force:
                continue
            to_process.append({"entity": s, "spec": spec, "key": key})
    logger.info(f"Total CommandSpecs: {total_specs}; a procesar (nuevos): {len(to_process)}")

    if not to_process:
        logger.info("Todo ya está indexado. --force para re-generar.")
        return

    # Generar + persistir
    stats = {"generated": 0, "phrases": 0, "failed": 0}
    for i, item in enumerate(to_process, 1):
        s = item["entity"]
        spec = item["spec"]
        area_display = (s["area"] or "sin habitación").replace("_", " ")
        detail, extra_rules = capability_prompt_detail(spec)

        if s["is_group"]:
            prompt = PROMPT_BASE_GROUP.format(
                friendly_name=s["friendly_name"], area=area_display,
                service=spec.service, detail=detail, extra_rules=extra_rules,
            )
        else:
            room, num = s["individual"]
            code = s["entity_id"].split(".")[1]
            prompt = PROMPT_BASE_INDIVIDUAL.format(
                friendly_name=s["friendly_name"], code=code, number=num, room=room,
                service=spec.service, detail=detail, extra_rules=extra_rules,
            )

        logger.info(f"[{i}/{len(to_process)}] {s['entity_id']} cap={spec.capability} val='{spec.value_label}' svc={spec.service}")
        t0 = time.time()
        try:
            text = llm.complete(prompt, max_tokens=300, temperature=0.5)
        except Exception as e:
            logger.error(f"  LLM error: {e}")
            stats["failed"] += 1
            continue
        elapsed = time.time() - t0

        phrases = parse_json_array(text)
        if not phrases:
            logger.warning(f"  [{elapsed:.1f}s] JSON parse falló. Fragmento: {text[:150]!r}")
            stats["failed"] += 1
            continue

        logger.info(f"  [{elapsed:.1f}s] {len(phrases)} frases:")
        for p in phrases:
            logger.info(f"    {p}")

        stats["generated"] += 1
        stats["phrases"] += len(phrases)

        if args.dry_run:
            continue

        # Embebear y persistir
        ids, embs, metas, docs = [], [], [], []
        for j, phrase in enumerate(phrases):
            doc_id = f"{item['key']}_{j}"
            ids.append(doc_id)
            embs.append(embedder.encode(phrase).tolist())
            docs.append(phrase)
            metas.append({
                "entity_id": s["entity_id"],
                "friendly_name": s["friendly_name"],
                "area": s["area"] or "",
                "domain": args.domain,
                "service": spec.service,
                "capability": spec.capability,
                "value_label": spec.value_label,
                "service_data": json.dumps(spec.service_data),
                "is_group": s["is_group"],
                "cache_key": item["key"],
            })
        if ids:
            collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

    logger.info(f"Done. Stats: {stats}")


if __name__ == "__main__":
    main()
