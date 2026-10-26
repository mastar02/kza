"""Observable in-memory HTTP/SSE endpoint for async generation tests."""

import asyncio
import json

import httpx
from openai import AsyncOpenAI


def event(text="", finish=None, *, style="chat", choices=True):
    choice = {"index": 0, "finish_reason": finish}
    choice.update({"delta": {"content": text}} if style == "chat" else {"text": text})
    payload = {
        "id": "synthetic", "created": 0, "model": "configured-model",
        "object": "chat.completion.chunk" if style == "chat" else "text_completion",
        "choices": [choice] if choices else [],
    }
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class SSEBody(httpx.AsyncByteStream):
    """Observable stream, with cooperative stalls, errors and explicit close."""

    def __init__(self, parts):
        self.parts = parts
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for part in self.parts:
            if isinstance(part, float):
                self.waiting.set()
                await asyncio.sleep(part)
            elif isinstance(part, asyncio.Event):
                self.waiting.set()
                await part.wait()
            elif isinstance(part, Exception):
                raise part
            else:
                yield part

    async def aclose(self):
        self.closed = True


class Endpoint:
    def __init__(self, body, *, models=("configured-model",), status=200, on_models=None):
        self.body = body
        self.models = models
        self.status = status
        self.on_models = on_models
        self.requests = []
        self.client = None

    async def handle(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.on_models:
                self.on_models()
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "test"}
                for model in self.models
            ]})
        if self.status != 200:
            return httpx.Response(self.status, headers={"location": "https://elsewhere.invalid/v1"},
                                  json={"error": {"message": "synthetic failure"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=self.body)

    def build(self, base_url):
        self.client = AsyncOpenAI(
            base_url=base_url, api_key="synthetic-key", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle), follow_redirects=False),
        )
        return self.client

