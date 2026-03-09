"""
LLM Reasoner Service - FastAPI wrapper for deep reasoning (EXPERIMENTAL)
Runs on CPU with llama-cpp-python (24 cores, ~70GB RAM)

WARNING: This is an EXPERIMENTAL Docker service. It does NOT have full parity
with the canonical runtime (src/main.py). Use src/main.py for production.

PARITY_GAPS vs canonical runtime:
  - No memory-augmented prompting (canonical: MemoryManager injects user context)
  - No personality system (canonical: configurable personality via training)
  - No conversation history tracking (canonical: ContextManager per user)
  - No streaming token output (canonical: ResponseHandler streams TTS as tokens arrive)
  - LoRA hot-load via API but no nightly auto-training (canonical: NightlyTrainer + QLoRA)
  - No routine creation/management integration (canonical: RoutineManager)
  - No user-specific response adaptation (canonical: UserManager preferences)
  - Missing: latency tracking, event logging for LLM calls
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KZA Reasoner Service", version="1.0.0")

# Global model instance
llm_model = None


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: list[str] | None = None


class ChatResponse(BaseModel):
    content: str
    tokens_used: int
    finish_reason: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: list[str] | None = None


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int


def get_model():
    global llm_model
    if llm_model is None:
        from llama_cpp import Llama
        
        model_path = os.getenv("MODEL_PATH", "/app/models/Llama-3.3-70B-Instruct-Q8_0.gguf")
        lora_path = os.getenv("LORA_PATH")
        n_ctx = int(os.getenv("N_CTX", "32768"))
        n_threads = int(os.getenv("N_THREADS", "24"))
        n_batch = int(os.getenv("N_BATCH", "512"))
        
        logger.info(f"Loading LLM: {model_path}")
        logger.info(f"Context: {n_ctx}, Threads: {n_threads}, Batch: {n_batch}")
        
        llm_model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=0,  # CPU only
            chat_format="llama-3",
            verbose=False
        )
        
        # Load LoRA if specified
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA adapter: {lora_path}")
            try:
                llm_model.set_lora(lora_path, scale=1.0)
            except Exception as e:
                logger.warning(f"Could not load LoRA: {e}")
        
        logger.info("LLM loaded successfully")
    return llm_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup"""
    get_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "reasoner"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat completion with the LLM.
    """
    try:
        model = get_model()
        
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop
        )
        
        choice = response["choices"][0]
        
        return ChatResponse(
            content=choice["message"]["content"],
            tokens_used=response["usage"]["total_tokens"],
            finish_reason=choice["finish_reason"]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Raw text generation with the LLM.
    """
    try:
        model = get_model()
        
        response = model(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop
        )
        
        return GenerateResponse(
            text=response["choices"][0]["text"],
            tokens_used=response["usage"]["total_tokens"]
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lora/load")
async def load_lora(lora_path: str, scale: float = 1.0):
    """
    Hot-load a LoRA adapter.
    """
    try:
        model = get_model()
        model.set_lora(lora_path, scale=scale)
        return {"status": "success", "lora_path": lora_path, "scale": scale}
    except Exception as e:
        logger.error(f"LoRA load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lora/unload")
async def unload_lora():
    """
    Unload current LoRA adapter.
    """
    try:
        model = get_model()
        model.set_lora(None)
        return {"status": "success", "message": "LoRA unloaded"}
    except Exception as e:
        logger.error(f"LoRA unload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
