"""Distiller — destilar-y-descartar (spec §1 privacidad, §4 componentes).

Cada interval_hours: toma utterances 'live' sin destilar, pide al LLM LOCAL
(:8101 — jamás cloud: conversaciones del hogar) hechos útiles en JSON, los
guarda en LongTermMemory (ChromaDB) y marca las filas. La purga del store
borra el crudo después. Si el LLM falla, NO marca — reintenta el próximo ciclo.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Categorías válidas = las de MemoryFact (src/memory/memory_manager.py)
_VALID_CATEGORIES = {"personal", "preference", "pattern", "fact"}

_SYSTEM_PROMPT = (
    "Sos el extractor de memoria de un asistente del hogar. Recibís "
    "transcripciones ambientales (pueden tener errores de STT). Extraé SOLO "
    "hechos útiles y duraderos: preferencias, planes con fecha, datos de "
    "personas, patrones. NADA de charla trivial ni nada dudoso. Si no hay "
    "nada útil, devolvé []. Respondé SOLO un array JSON: "
    '[{"fact": str, "category": "personal|preference|pattern|fact", '
    '"confidence": 0.0-1.0}]'
)


def _parse_facts(raw: str) -> list[dict]:
    """Parse robusto del JSON del LLM (tolera fences y basura alrededor)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    facts = []
    for item in data:
        if (
            isinstance(item, dict)
            and item.get("fact")
            and item.get("category") in _VALID_CATEGORIES
        ):
            facts.append({
                "fact": str(item["fact"]),
                "category": item["category"],
                "confidence": float(item.get("confidence", 0.7)),
            })
    return facts


def make_langid_fn() -> Callable[[str], str]:
    """Detector de idioma para la señal SHADOW del distiller (py3langid).

    El hogar habla español; el inglés en las transcripciones ambientales es
    bleed de TV/película far-field. py3langid lleva el modelo embebido (sin
    descarga, determinista, solo numpy ya presente). El distiller lo usa para
    LOGUEAR la distribución de idioma de lo que pasa la compuerta de vad —
    insumo para decidir si en el futuro se enforce una compuerta de idioma.

    Returns:
        callable ``text -> código ISO-639-1`` (p.ej. "es", "en"); "unknown"
        si el texto es vacío/whitespace.
    """
    import py3langid

    identifier = py3langid.langid.LanguageIdentifier.from_pickled_model(
        py3langid.langid.MODEL_FILE, norm_probs=True
    )

    def detect(text: str) -> str:
        if not text or not text.strip():
            return "unknown"
        lang, _prob = identifier.classify(text)
        return lang

    return detect


def make_local_chat_fn(
    llm_url: str = "http://127.0.0.1:8101/v1",
    model: str = "local",
    timeout_s: float = 120.0,
) -> Callable[[str], Awaitable[str]]:
    """chat_fn real contra el llama-server local (OpenAI-compat).

    Las transcripciones del hogar SOLO deben ir a un LLM local — si la URL
    configurada no es loopback se avisa fuerte (no se bloquea: una IP LAN del
    propio server puede ser legítima, pero tiene que ser una decisión visible).
    """
    from urllib.parse import urlparse

    import aiohttp

    from src.llm.reasoner import _resolve_api_key

    host = urlparse(llm_url).hostname or ""
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            f"⚠️ Distiller: llm_url apunta a {host!r} (no-loopback). Las "
            f"transcripciones ambientales del hogar JAMÁS deben salir a un "
            f"servicio cloud — verificá que sea un LLM local."
        )

    # El LLM local exige bearer (LLAMA_API_KEY :8101, enforce desde 2026-04-30).
    # Reusa la resolución por puerto del reasoner — sin el header → 401.
    headers = {"Authorization": f"Bearer {_resolve_api_key(llm_url)}"}

    async def chat(prompt: str) -> str:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            async with session.post(
                f"{llm_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    return chat


class Distiller:
    """Job periódico de extracción de hechos."""

    def __init__(
        self,
        store,
        chat_fn: Callable[[str], Awaitable[str]],
        store_fact_fn: Callable[..., str],
        interval_hours: float = 6.0,
        min_batch: int = 5,
        max_batch_chars: int = 12000,
        min_vad_prob: float = 0.0,
        lang_detect_fn: Callable[[str], str] | None = None,
    ):
        """
        Args:
            store: AmbientStore (undistilled_live / mark_distilled).
            chat_fn: prompt → respuesta del LLM LOCAL. DI para tests.
            store_fact_fn: firma de LongTermMemory.store_fact(fact, category,
                confidence=, metadata=).
            min_batch: no destilar con menos utterances (ahorra ciclos LLM).
            min_vad_prob: compuerta de calidad (Silero) — solo se destilan
                utterances con vad_prob ≥ umbral. 0.0 = sin filtrar.
            lang_detect_fn: detector de idioma SHADOW (opcional). Si está, se
                loguea la distribución de idioma del batch SIN filtrar — insumo
                para decidir una futura compuerta de idioma (el ruido dominante
                es bleed de TV en inglés). Ver make_langid_fn.
        """
        self._store = store
        self._chat = chat_fn
        self._store_fact = store_fact_fn
        self.interval_hours = interval_hours
        self.min_batch = min_batch
        self.max_batch_chars = max_batch_chars
        self.min_vad_prob = min_vad_prob
        self._lang_detect = lang_detect_fn
        self._running = False

    async def distill_once(self) -> int:
        """Un ciclo de destilación. Devuelve cantidad de hechos guardados."""
        rows = await self._store.undistilled_live(
            limit=200, min_vad_prob=self.min_vad_prob
        )
        if len(rows) < self.min_batch:
            return 0
        # Truncar el lote por presupuesto de chars del prompt (7B local)
        batch, total = [], 0
        for r in rows:
            total += len(r["text"]) + 40
            if total > self.max_batch_chars:
                break
            batch.append(r)
        if not batch:
            # max_batch_chars patológicamente chico: ni una utterance entra.
            # Sin guard se llamaría al LLM con prompt vacío (riesgo de hechos
            # alucinados) y mark_distilled([]) — ciclo en vano cada interval.
            logger.warning(
                "Distiller: batch vacío tras truncado por chars — revisar "
                f"max_batch_chars={self.max_batch_chars}"
            )
            return 0
        self._log_language_shadow(batch)
        lines = [
            f"[{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['t0']))}] "
            f"({r['room_id']}, {r['speaker']}): {r['text']}"
            for r in batch
        ]
        prompt = "Transcripciones ambientales:\n" + "\n".join(lines)
        try:
            raw = await self._chat(prompt)
        except Exception as e:
            logger.warning(f"Distiller: LLM local falló ({e}) — reintento próximo ciclo")
            return 0
        facts = _parse_facts(raw)
        stored = 0
        for f in facts:
            try:
                # to_thread: LongTermMemory.store_fact es sync (ChromaDB add
                # con cómputo de embeddings) — no bloquear el event loop.
                await asyncio.to_thread(
                    self._store_fact,
                    f["fact"], f["category"],
                    confidence=f["confidence"],
                    metadata={"origin": "ambient"},
                )
                stored += 1
            except Exception as e:
                logger.warning(f"Distiller: store_fact falló: {e}")
        # Marcar TODO el batch procesado (aunque no haya hechos: ya se evaluó)
        await self._store.mark_distilled([r["id"] for r in batch])
        if stored:
            logger.info(f"Distiller: {stored} hechos de {len(batch)} utterances")
        return stored

    def _log_language_shadow(self, batch: list[dict]) -> None:
        """SHADOW: loguear la distribución de idioma del batch (no filtra).

        El ruido dominante del ambient path es bleed de TV en inglés (medido:
        47% del crudo). Este log mide cuánto de lo que pasa la compuerta de vad
        sigue siendo no-español — insumo para una futura compuerta de idioma.
        """
        if self._lang_detect is None:
            return
        from collections import Counter

        dist = Counter(self._lang_detect(r["text"]) for r in batch)
        logger.info("Distiller shadow idioma: %s", dict(dist))

    async def run_forever(self) -> None:
        """Loop del job (lo lanza AmbientTranscriber.start)."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_hours * 3600)
            try:
                await self.distill_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Distiller: ciclo falló (best-effort, sigo)")

    def stop(self) -> None:
        self._running = False
