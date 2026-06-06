# ¿Hermes 4 + RAG total = potencial enorme para KZA? — Análisis exhaustivo

**Fecha:** 2026-06-06
**Método:** exploración del repo (3 agentes) + workflow multi-agente (5 verificadores web adversariales → 2 analistas → 1 crítico de completitud; ~556K tokens de investigación). Todos los claims clave verificados contra fuentes primarias (HuggingFace org NousResearch, arxiv 2508.18255, BFCL leaderboard, benchmarks EPYC/Threadripper reales).

---

## Veredicto ejecutivo

> **El "potencial enorme" es real, pero NO viene de Hermes (el modelo) — viene del RAG total (los datos) + tool-calling real. Son completamente separables, y separarlos es lo que minimiza el riesgo.**

| Pregunta | Respuesta |
|---|---|
| ¿Adoptar Hermes 4 como modelo (local)? | **NO.** Toda la familia es DENSA; en nuestro CPU sería un downgrade de ~10-20× en velocidad vs el MoE Qwen3-30B-A3B (63 tok/s). |
| ¿Hermes 4 cloud reemplazando MiniMax? | **Marginal.** Hermes-4-70B vía OpenRouter es 2-3× más barato ($0.13/$0.40 vs $0.279/$1.20 por M tok), pero no mejora privacidad ni capacidad. Cambio lateral, no estratégico. |
| ¿El RAG total de KZA? | **SÍ — acá está el valor.** El modelo actual ya es capaz; está **ciego a sus propios datos**. Eventos (90 días), hábitos, preferencias, docs: todo existe y nada es recuperable. |
| ¿Algo de Hermes sirve? | **Sí: patrones de arquitectura** del framework `hermes-agent` (4 tools HA, memoria en capas, skills markdown) — como referencia, no como runtime. Y monitorear un futuro **Hermes MoE** (Teknium: "Yes very probable!" a un 30B-A3B). |

La trampa a evitar: confundir "necesito un modelo más capaz" con "necesito que el modelo **vea** los datos correctos". Arreglar la visión (RAG), no cambiar los ojos (modelo).

---

## 1. Hechos verificados sobre Hermes 4 (junio 2026)

### Catálogo real (org NousResearch en HF)

| Modelo | Base | Arquitectura | Contexto | Licencia | GGUF |
|---|---|---|---|---|---|
| Hermes-4-14B | Qwen3-14B | **densa** | 40K | Apache 2.0 | comunidad (bartowski 9.0GB Q4_K_M) |
| Hermes-4-70B | Llama-3.1-70B | **densa** | 131K | llama3 | comunidad (LM Studio, ~43GB Q4) |
| Hermes-4-405B | Llama-3.1-405B | **densa** | 131K | llama3 | — (inviable local) |
| Hermes-4.3-36B | Seed-OSS-36B (ByteDance) | **densa** | **512K nativo** | Apache 2.0 | **oficial** (21.8GB Q4_K_M) |

- **NO existe Hermes MoE de producción.** El "Hermes 4 35B A3B MoE" que circula en blogs SEO es **contenido alucinado** (verificado contra la colección oficial HF). Nous solo tiene un MoE de research (`moe-10b-a1b`, sin instruct, sin model card). Teknium respondió "Yes very probable!" a un pedido público de Hermes 30B-A3B → monitorear.
- **No hubo checkpoint nuevo entre dic-2025 (4.3) y jun-2026**: el foco de Nous viró a producto (Hermes Agent feb-2026, Hermes Desktop jun-2026).
- **Benchmarks reales** (Technical Report, notación Reasoning/(Non-reasoning)): Hermes-4-70B MMLU 88.4/(76.7), GPQA 66.1/(33.3), AIME'25 67.5/**(7.3)**. Sin thinking colapsa en razonamiento duro → el thinking mode no es opcional, y en CPU multiplica la latencia (el reporte trata el *overthinking* como problema central).
- **BFCL (function calling): NO publicado para ningún Hermes.** Qwen3-30B-A3B sí: 65.1 (Instruct-2507) / 70.8 (original) / 72.4 (Thinking-2507). Líderes BFCL-v3: GLM-4.5 (77.8), Qwen3-32B (75.7). **Migrar a Hermes sería apostar a ciegas en la dimensión que más nos importa.**
- **Español: cero datos publicados** para Hermes 4 (todos los benchmarks del reporte son en inglés). Qwen3 publica suite multilingüe (MultiIF 67.9, MMLU-ProX 72.0, INCLUDE 71.9) — *ver gap §5.2: tampoco mide es-AR específicamente, pero la familia Qwen ya corre en producción rioplatense en KZA (FastRouter + NLU)*.
- El formato `<tool_call>` de Hermes y el parser `hermes` de vLLM son reales, pero **json_schema/GBNF son features del runtime** (llama.cpp/ik_llama/vLLM), agnósticas al modelo — ya las usamos hoy con el LLMCommandRouter.

### Velocidades proyectadas en NUESTRO hardware (calibradas + trianguladas)

Calibración con dato real propio: Qwen3-30B-A3B MoE @ 63 tok/s ⇒ **~126-150 GB/s de ancho de banda efectivo** (~35-42% del teórico 358 GB/s). Triangulado con benchmarks medidos (EPYC 9554: 70B-Q4 = 7.12 t/s con 460 GB/s; Threadripper 7995WX: Mixtral 8×22B = 8 t/s).

| Modelo | TG (gen) | Prefill (PP) | E2E slow path con RAG 4K + 150 tok resp. |
|---|---|---|---|
| Qwen3-30B-A3B MoE (actual) | **63 t/s** | a medir (proyección ~150-300 t/s) | **~15-19s ✅** (proyectado, **medir PP en server**) |
| Hermes-4-14B denso | 12-15 t/s | ~150-300 t/s (ik_llama) | ~37-39s ⚠️ (solo cabe con RAG ≤2K) |
| Hermes-4.3-36B denso | 5-8 t/s | ~60-120 t/s | **~63-74s ❌** |
| Hermes-4-70B denso | 2-4 t/s | ~25-50 t/s | **~150-190s ❌** |

**El cuello de botella del slow path local con RAG no es la generación: es el prefill compute-bound.** ik_llama mejora prefill 1.8-4.1× sobre llama.cpp vanilla (TG solo 1.05-2.1×). El 14B es "marginalmente viable pero dominado" por el MoE actual (mejor descripción que "inviable" — corrección del crítico).

### Framework `hermes-agent` (confirmado, con matices)

- Existe (MIT, lanzado 25-feb-2026, cadencia frenética de releases, Hermes Desktop jun-2026). **Provider-agnostic**: corre con cualquier endpoint OpenAI-compat, incluido nuestro stack.
- Integración HA **oficial**: 4 tools (`ha_list_entities`, `ha_get_state`, `ha_list_services`, `ha_call_service`) + gateway de notificaciones. **Sin voz propia** (la voz la aporta una integración comunitaria vía Assist de HA).
- **NO adoptable como runtime para KZA**: overhead fijo de ~13.9K tokens/call (issue #4379, ~73% de cada llamada) — inviable para nuestro fast path; bug confirmado de prompts oversized contra modelos de contexto chico (issue #23767) — exactamente nuestro régimen.
- **SÍ robar 3 patrones**: (1) las 4 tools HA como esquema mínimo si agregamos function calling; (2) memoria en capas SOUL.md/AGENTS.md/MEMORY.md + FTS5 sobre conversaciones pasadas → blueprint para indexar nuestro historial conversacional (hoy solo summary); (3) skills como markdown procedural con índice liviano → blueprint para `kza_knowledge`.

---

## 2. El RAG total: dónde está el potencial de verdad

### Estado actual: dos RAG aislados + conocimiento invisible

| Conocimiento | Dónde vive | ¿Recuperable hoy? |
|---|---|---|
| Comandos HA (~749 docs) | Chroma `home_assistant_commands` | ✅ (fast path) |
| Escenas (5) | Chroma `home_assistant_routines` | ✅ |
| Memoria de usuario | Chroma `user_memories` | ⚠️ existe pero **FactExtractor apagado** (`_fact_extractor=None`) |
| Eventos 90 días | SQLite `events.db` (con índices hour/weekday/entity) | ❌ **nunca consultado** |
| Hábitos detectados | JSON `data/patterns/` | ❌ solo sugiere rutinas |
| Preferencias | JSON key-value | ❌ sin búsqueda semántica |
| Historial conversacional | solo summary compactado | ❌ turnos literales se pierden |
| Docs/decisiones del sistema | repo | ❌ cero indexado |
| Presencia en vivo | runtime | ❌ no se inyecta al LLM |

**Hallazgo clave verificado en código:** el seam de inyección ya existe y está desconectado. `dispatcher._process_llm_request` (línea ~1713) llama `build_prompt(user_id, text)` **sin pasar `include_home_state`**, parámetro que `context_manager.build_prompt` ya soporta (línea ~543). La infraestructura está a medio cablear y el dispatcher la ignora.

### Arquitectura propuesta: `RetrievalBroker` (capa de orquestación, no un RAG nuevo)

- **Interfaz única** `retrieve(query, user_id, zone_id, intent_hint, budget_ms=400)` → fan-out async paralelo a fuentes plausibles con timeout por fuente → fusión **RRF** (Reciprocal Rank Fusion, k=60, sin LLM, robusto a scores heterogéneos) → bloque de texto capped por chars → entra por `include_home_state` en `build_prompt`. **Solo en slow path** — el fast path <300ms queda intacto.
- **Queries temporales** ("anoche", "el finde", "¿quedó algo prendido?"): NO embedding, NO text-to-SQL libre. `TemporalParser` determinístico (regex es-AR → TimeWindow) + plantillas sobre métodos **ya existentes** de `event_logger` (`get_events`, `get_hourly_distribution`, `get_sequences`). El `EventRetriever` **sintetiza una frase factual** ("quedó prendida la luz del living desde las 23:40, sin turn_off") — el LLM lee conclusiones, no filas.
- **Hábitos recurrentes** ("¿qué hago a las 7am?"): batch nocturno (encadenado al job LoRA existente) genera resúmenes NL desde `pattern_learner` + agregaciones SQL → colección nueva `event_summaries` (BGE-M3 **en CPU**, `embedder_device` ya es parámetro).
- **Auto-explicación** ("¿por qué no me escuchaste?", "¿qué es SPENERGY?"): colección `kza_knowledge` con docs/decisiones/settings chunked. Read-only, riesgo nulo, alto valor diferenciador.
- **Todo degrada a camino determinístico**: function calling es un upgrade opcional (promover TemporalParser a tool con json_schema — patrón que ya usamos en `llm_router.py`), nunca dependencia dura. Independiente del modelo por diseño.

### Fases (1 dev)

| Fase | Qué | Esfuerzo | Valor |
|---|---|---|---|
| **F1** | Cablear `include_home_state` desde dispatcher + activar `FactExtractor` + unificar embedder de `user_memories` a BGE-M3 (*verificar en server primero*) | **S (1-2d)** | **Alto — enciende lo ya construido** |
| **F2** | `RetrievalBroker` v1 (fan-out, RRF, cap, presupuesto) | M (3-5d) | Alto |
| **F3** | `EventRetriever` + `TemporalParser` es-AR | M (3-5d) | Alto (capability más visible) |
| F4 | `event_summaries` batch nocturno | M | Medio-alto |
| F5 | `kza_knowledge` (paralelizable desde día 1) | S-M | Medio |
| F6 | Tool-calling real (opcional, cuando se decida dónde — ver §5.4) | M | Medio |

Camino crítico F1→F2→F3 ≈ 2 semanas para un RAG total funcional sobre estado + memoria + eventos temporales.

---

## 3. Opciones evaluadas (resumen de decisión)

| Opción | Veredicto |
|---|---|
| **A. Hermes denso local (slow path)** | ❌ Rechazada. 36B/70B rompen el presupuesto solo en prefill (44-171s). 14B viable solo con RAG ≤2K y dominado por el MoE actual en todo eje medido. |
| **B. Hermes-4-70B cloud (reemplaza MiniMax)** | ⚠️ Solo como micro-optimización de costo (2-3× más barato por token, PERO sin normalizar por tokens de thinking — el costo por TAREA no está calculado). No mejora privacidad. Español no medido. |
| **C. Qwen3 MoE actual + RAG total + tool-calling nativo** | ✅ **Recomendada. Captura el potencial completo.** |
| **D1. Híbrido: local para >90% + cloud long-tail** | ✅ Es el destino natural; C es cómo se llega. ⚠️ con tensión de privacidad (§5.5). |
| **D2. Esperar Hermes MoE** | Monitorear org NousResearch. No accionable hoy. |
| **E. Upgrade a Qwen3.5-35B-A3B** (feb-2026, mismos ~3B activos, mejor calidad/tools/multilingüe) | Opcional, NO urgente. ⚠️ Verificar GGUF + compat ik_llama en server; rompe cadena LoRA (§5.1). |

---

## 4. Lo único de Hermes que entra al roadmap

1. **Esquema de 4 tools HA** como blueprint de F6 (si/cuando haya function calling real).
2. **Memoria en capas + FTS5 sobre conversaciones** como blueprint para indexar historial conversacional (F4+).
3. **Skills markdown con índice liviano** como blueprint de `kza_knowledge` (F5).
4. **Monitorear** un eventual Hermes 30B-A3B MoE con BFCL+español medidos → reevaluar como drop-in del slow path local.
5. (Opcional, lateral) Hermes-4-70B vía OpenRouter como reductor de costo si nos quedamos en cloud — solo tras medir costo por tarea real.

---

## 5. Gaps abiertos (del crítico de completitud — cerrar ANTES de comprometer esfuerzo grande)

### 5.1 Incoherencia LoRA nocturno (preexistente, verificada en código)
El slow path activo (`HttpReasoner`→cloud) **no carga LoRA** (`reasoner.py:760`: "HttpReasoner no carga LoRA"), y el config tiene **tres base_model contradictorios** (Llama-3.2-3B en `nightly_trainer.py:114`, Qwen2.5-7B en `settings.yaml:670`, Llama-3.1-70B en `settings.yaml:611`). **El sistema entrena adapters que nadie consume.** Cualquier cambio de modelo (Hermes O Qwen3.5) rompe además la compatibilidad de adapters. → Decidir: ¿el LoRA nocturno apunta al FastRouter 7B (único consumidor posible hoy), se rediseña, o se pausa?

### 5.2 Español rioplatense: no zanjado, solo desplazado
Ni Hermes ni Qwen tienen es-AR medido head-to-head; los benchmarks multilingües de Qwen son agregados. La presunción pro-Qwen es más fuerte (familia ya en producción en KZA con el FastRouter + NLU en es-AR diario), pero si se evalúa CUALQUIER modelo nuevo: armar un smoke test es-AR propio (30-50 prompts reales del hogar con voseo) antes de decidir.

### 5.3 Prefill MoE-en-CPU: nunca medido en nuestro hardware
El presupuesto "~15-19s E2E con RAG 4K" descansa en PP proyectado (la referencia 250-330 t/s es de una Jetson GPU). **Medir en el server**: `llama-bench` (ik_llama) con el 30B-A3B, pp2048/pp4096/pp6144 — si el PP real es mucho menor, el RAG inyectado debe ser más corto o el slow path local pierde viabilidad con contexto largo.

### 5.4 ¿Dónde va el function calling? (decisión previa a F6)
El análisis se centró en el slow path, pero el lugar de mayor impacto podría ser el **FastRouter/LLMCommandRouter** (fast path, binding constraint 245-272ms). El costo de grammar/json_schema de tools en el 7B no está cuantificado. Decidir el target antes de darle peso al BFCL de cualquier modelo (hoy, sin tool-calling, el BFCL es irrelevante para el sistema actual).

### 5.5 RAG total + cloud = tensión de privacidad
Inyectar memoria de usuario + eventos + presencia al prompt y mandarlo a MiniMax **expone exactamente lo más sensible**. El long-tail que va a cloud es el que más contexto privado lleva. Opciones: (a) filtro/anonimización del bloque RAG antes de cloud (no diseñado aún), (b) gatear con `cloud_consent`, (c) slow path local para queries que tocan memoria personal (requiere §5.3 resuelto). Diseñar en F2.

---

## 6. Próximos pasos concretos

1. **[server, 30 min]** `llama-bench` de PP del Qwen3-30B-A3B en ik_llama (pp2048/4096/6144) → cierra §5.3.
2. **[server, 15 min]** Verificar embedder real de `user_memories` (¿default Chroma o BGE-M3 por DI?) → define si F1 incluye re-indexado.
3. **[decisión]** Resolver la incoherencia LoRA (§5.1) — independiente de todo lo demás y ya está costando cómputo nocturno inútil.
4. **[laptop, F1]** Cablear `include_home_state` + activar FactExtractor (TDD; el seam es `dispatcher._process_llm_request`).
5. F2 → F3 según §2.
6. **[backlog]** Watch en org NousResearch por Hermes MoE; smoke test es-AR si se evalúa modelo nuevo.

---

## Addendum 2026-06-06: ¿Y MiniMax M2.7-highspeed como cerebro del slow path?

*(Verificación posterior: 2 agentes web adversariales sobre specs/precio/privacidad + calidad/alternativas.)*

### Qué es `-highspeed` (verificado)
**Mismo modelo, mismos pesos** — solo serving más rápido (routing MoE, batching, memoria; NO speculative decoding: los safetensors públicos ni traen pesos MTP). ~100 t/s vs ~60-78 t/s del base servido por MiniMax. **Precio exactamente 2×**: $0.60/$2.40 vs $0.30/$1.20 por M tok (PAYG oficial). El 429 de mayo confirmado: es un tier separado — suscripción Highspeed (~$40/$80/$150/mes, cifras de agregador, no verificadas en la página oficial JS-rendered) o PAYG al endpoint `-highspeed` a 2×.

### El hallazgo que cambia la decisión
**Providers third-party sirven M2.7 BASE a 400+ t/s al MISMO precio base**: Together.ai 416 t/s (TTFT 0.5-7s), SambaNova 436 t/s, vs MiniMax-direct 77.7 t/s (TTFT warm ~2-3s, cold hasta 34s — coincide con nuestra nota "cold 10-20s"). Es decir: **pagar el premium highspeed de MiniMax está dominado por cambiar de provider** (config-only en el gateway LiteLLM / OpenRouter) — 4× más velocidad que highspeed, sin 2× de precio, y mueve los datos fuera de la infra China de MiniMax.

| Config slow path (RAG 4K + ~150 tok hablados + thinking típico) | E2E estimado | Costo | Privacidad |
|---|---|---|---|
| M2.7 base, MiniMax direct (actual) | ~10-28s warm (cold +10-30s) | $0.30/$1.20 | China |
| M2.7-highspeed, MiniMax | ~7-17s warm | **2×** + suscripción | China |
| **M2.7 base vía Together/OpenRouter** | **~3-12s** | $0.30/$1.20 | jurisdicción del provider (verificar flag ZDR por endpoint) |
| Qwen3-30B-A3B local (slot :8200) | ~15-19s (PP a medir) | $0 | **total** |

### Minas descubiertas para el roadmap (importan más que el highspeed)
1. **Strip de `<think>` = mina para F6 (tool-calling multi-turno).** MiniMax exige *preservar* el thinking entre turnos de tool-calling: sin él Tau2 cae -35.9%, BrowseComp -40.1%, GAIA -11.5% (datos propios de MiniMax). Hoy (single-turn → TTS) nuestro strip es correcto; si F6 va al slow path con M2.x, hay que **re-inyectar el thinking entre turnos**, no descartarlo.
2. **El thinking es incontrolable por nuestra ruta actual**: `reasoning_effort` se ignora silenciosamente en el endpoint OpenAI-compat; solo se controla vía el endpoint Anthropic-compat (`api.minimax.io/anthropic`, `thinking={type:'disabled'}`) que MiniMax recomienda oficialmente para M2.x. **Pagamos $1.20/M por cada token de thinking que después strippeamos** (M2.7 es +55% más verboso que M2.5: ~87M tokens para correr el AA Index).
3. **tau2-bench REGRESÓ -11pp en M2.7 vs M2.5** — justo el eje tool-agent-USER, el análogo más cercano al tool-calling conversacional de domótica. Las ganancias de M2.7 son SWE/coding-céntricas.
4. **Español: cero benchmarks** (mismo vacío que Hermes/Qwen) — pero M2.7 tiene lo que ningún otro: **evidencia de campo en KZA** (ya responde en es-AR en producción).
5. **Competencia jun-2026**: M2.7 (AA Index 50) ya no es #1: Kimi K2.6=54, **DeepSeek V4 Pro=52** ($0.44/$0.87 — output MÁS barato, GDPval ~1554>M2.7, 1M ctx, MIT), GLM-5.1=51. **DeepSeek V4 Pro es el candidato A/B natural.** M2.7 pesa 230B/10B MoE con licencia non-commercial "other" — no self-hosteable en nuestro HW de todos modos.
6. **Privacidad**: política MiniMax declara no-train sobre datos de API + modo zero-retention, pero jurisdicción China sin adecuación EU. Con el RAG total inyectando memoria/eventos/presencia, el riesgo §5.5 queda igual con o sin highspeed.

### Veredicto addendum
- **El RAG total funciona con M2.7 YA** — F1→F3 no esperan ninguna decisión de modelo.
- **NO contratar highspeed**: dominado por servir M2.7-base vía Together/OpenRouter (4× más rápido que highspeed, mismo precio, config-only en el gateway). Si se hace, verificar el flag ZDR del endpoint elegido.
- Considerar migrar la ruta a **endpoint Anthropic-compat** (o vía LiteLLM con ese backend) para poder controlar el thinking → menos costo y menos latencia sin cambiar de modelo.
- **A/B sugerido cuando el RAG esté en F3**: M2.7 vs DeepSeek V4 Pro sobre prompts reales es-AR con bloque RAG.
- La conclusión estratégica no cambia: highspeed acelera el interino cloud; el endgame de privacidad para queries con memoria personal sigue siendo el slot local `:8200`.

---

## Fuentes primarias clave

- Catálogo y model cards: `huggingface.co/NousResearch/{Hermes-4-14B,Hermes-4-70B,Hermes-4-405B,Hermes-4.3-36B}` · `nousresearch.com/releases` · `nousresearch.com/introducing-hermes-4-3`
- Technical report: `arxiv.org/abs/2508.18255`
- BFCL: `llm-stats.com/benchmarks/bfcl-v3` (ningún Hermes listado) · Qwen3 model cards
- hermes-agent: `github.com/NousResearch/hermes-agent` (issues #4379 overhead, #23767 contexto chico, #16831 memory cap, #32384 update corrupto) · doc integración HA
- Benchmarks CPU: EPYC 9554 (ahelpme.com) · ik_llama discussions #164 (prefill 1.78-4.13×) · llamafile #450 (TR 7995WX)
- Precios: `openrouter.ai/nousresearch/hermes-4-70b` ($0.13/$0.40) vs `minimax-m2.7` ($0.279/$1.20)
- Alucinación detectada y descartada: "Hermes 4 35B A3B MoE" (blogs SEO: popularaitools, lushbinary — NO existe en la colección oficial)
