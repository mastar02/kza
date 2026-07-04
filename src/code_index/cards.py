"""Generador de "cards" por archivo vía gateway :8200 (MiniMax, OpenAI-compat)."""

import logging
import os
import re

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

CARD_PROMPT = """Sos un ingeniero senior documentando un codebase Python de un asistente de voz local (KZA).
Generá una "card" en markdown para el archivo `{path}`. Secciones exactas:

## Propósito
(2-3 frases: qué resuelve este archivo dentro del sistema)

## API pública
(clases/funciones públicas con firma y una línea de descripción cada una)

## Dependencias
(qué usa: otros módulos src.*, servicios externos, hardware)

## Invariantes y gotchas
(supuestos, órdenes de llamada requeridos, edge cases no obvios; si no hay, "—")

Máximo ~300 palabras. Sin preámbulo ni cierre: solo la card.

```python
{source}
```"""


def _strip_think(text: str) -> str:
    """MiniMax emite bloques <think> — sacarlos siempre."""
    return _THINK_RE.sub("", text)


class CardGenerator:
    """Genera resúmenes ("cards") de archivo con MiniMax vía el gateway LiteLLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "MINIMAX_API_KEY",
        timeout: float = 120.0,
        max_source_chars: int = 48_000,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_source_chars = max_source_chars
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=os.environ.get(self.api_key_env, "dummy"),
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, path: str, source: str) -> str:
        """Generar la card markdown de un archivo.

        Raises:
            Exception: si el gateway falla (el caller decide el retry —
            el manifest marca card_done=False y se reintenta en el próximo
            reindex).
        """
        prompt = CARD_PROMPT.format(path=path, source=source[: self.max_source_chars])
        client = self._get_client()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return _strip_think(text).strip()
