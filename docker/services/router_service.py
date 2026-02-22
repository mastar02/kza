"""
Router Service - FastAPI wrapper for Fast Classification
Runs on GPU 2 with vLLM
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KZA Router Service", version="1.0.0")

# Global model instance
router_model = None


class ClassifyRequest(BaseModel):
    text: str
    categories: list[str] = ["domotica", "rutina", "consulta", "conversacion"]


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    reasoning: str | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.1


class GenerateResponse(BaseModel):
    text: str
    tokens_used: int


def get_model():
    global router_model
    if router_model is None:
        from vllm import LLM, SamplingParams
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        gpu_util = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
        logger.info(f"Loading router model: {model_name}")
        router_model = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True
        )
        logger.info("Router model loaded")
    return router_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup"""
    get_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "router"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """
    Classify user input into categories.
    """
    from vllm import SamplingParams
    
    try:
        model = get_model()
        
        prompt = f"""Clasifica el siguiente texto en UNA de estas categorías: {', '.join(request.categories)}

Texto: "{request.text}"

Responde SOLO con el nombre de la categoría, sin explicación.
Categoría:"""
        
        sampling_params = SamplingParams(
            max_tokens=20,
            temperature=0.1
        )
        
        outputs = model.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text.strip().lower()
        
        # Match to valid category
        category = "conversacion"  # default
        for cat in request.categories:
            if cat.lower() in result:
                category = cat
                break
        
        return ClassifyResponse(
            category=category,
            confidence=0.9 if category in result else 0.7,
            reasoning=result
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text with the router model.
    """
    from vllm import SamplingParams
    
    try:
        model = get_model()
        
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        outputs = model.generate([request.prompt], sampling_params)
        result = outputs[0].outputs[0].text.strip()
        
        return GenerateResponse(
            text=result,
            tokens_used=len(outputs[0].outputs[0].token_ids)
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
