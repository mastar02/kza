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


def make_local_chat_fn(
    llm_url: str = "http://127.0.0.1:8101/v1",
    model: str = "local",
    timeout_s: float = 120.0,
) -> Callable[[str], Awaitable[str]]:
    """chat_fn real contra el llama-server local (OpenAI-compat)."""
    import aiohttp

    async def chat(prompt: str) -> str:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as session:
            async with session.post(
                f"{llm_url.rstrip('/')}/chat/completions",
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
    ):
        """
        Args:
            store: AmbientStore (undistilled_live / mark_distilled).
            chat_fn: prompt → respuesta del LLM LOCAL. DI para tests.
            store_fact_fn: firma de LongTermMemory.store_fact(fact, category,
                confidence=, metadata=).
            min_batch: no destilar con menos utterances (ahorra ciclos LLM).
        """
        self._store = store
        self._chat = chat_fn
        self._store_fact = store_fact_fn
        self.interval_hours = interval_hours
        self.min_batch = min_batch
        self.max_batch_chars = max_batch_chars
        self._running = False

    async def distill_once(self) -> int:
        """Un ciclo de destilación. Devuelve cantidad de hechos guardados."""
        rows = await self._store.undistilled_live(limit=200)
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
                self._store_fact(
                    f["fact"], f["category"], confidence=f["confidence"],
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
