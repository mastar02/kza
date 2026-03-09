"""
Voice Pipeline Service - Main Orchestrator (EXPERIMENTAL)

WARNING: This is an EXPERIMENTAL Docker service. It does NOT have full parity
with the canonical runtime (src/main.py). Use src/main.py for production.

PARITY_GAPS vs canonical runtime:
  - No speaker identification (canonical: per-user context via SpeakerIdentifier)
  - No emotion detection (canonical: EmotionDetector adjusts response tone)
  - No multi-user orchestration (canonical: MultiUserOrchestrator with priority queue)
  - No memory system (canonical: MemoryManager short/long-term per user)
  - No lists or reminders (canonical: ListManager + ReminderManager + ReminderScheduler)
  - No multi-room audio (canonical: MultiRoomAudioLoop with per-room wake word)
  - No Spotify/music routing (canonical: MusicDispatcher + MoodMapper + zone control)
  - No alerts system (canonical: AlertManager + AlertScheduler for security/patterns)
  - No nightly training (canonical: NightlyTrainer + QLoRA + HabitDatasetGenerator)
  - No analytics (canonical: EventLogger + PatternAnalyzer + SuggestionEngine)
  - No latency monitoring (canonical: LatencyMonitor with per-stage tracking)
  - No timers, intercom, or notifications (canonical: NamedTimerManager, IntercomSystem)
  - No presence detection (canonical: PresenceDetector BLE per room)
  - No routine management (canonical: RoutineManager creates HA automations)
  - No wake word detection (canonical: WakeWordDetector per room)
  - HA command execution is a stub — finds nearest vector but does NOT call HA API
  - No streaming TTS (canonical: ResponseHandler streams audio as tokens arrive)
"""

import os
import io
import logging
import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs from environment
STT_URL = os.getenv("STT_SERVICE_URL", "http://stt:8001")
TTS_URL = os.getenv("TTS_SERVICE_URL", "http://tts:8004")
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_SERVICE_URL", "http://embeddings:8002")
ROUTER_URL = os.getenv("ROUTER_SERVICE_URL", "http://router:8003")
REASONER_URL = os.getenv("REASONER_SERVICE_URL", "http://reasoner:8005")
CHROMADB_URL = os.getenv("CHROMADB_URL", "http://chromadb:8000")
HA_URL = os.getenv("HOME_ASSISTANT_URL")
HA_TOKEN = os.getenv("HOME_ASSISTANT_TOKEN")

# HTTP client with connection pooling
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage HTTP client lifecycle"""
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("Pipeline service started")
    yield
    await http_client.aclose()
    logger.info("Pipeline service stopped")


app = FastAPI(
    title="KZA Voice Pipeline",
    version="1.0.0",
    lifespan=lifespan
)


class ProcessRequest(BaseModel):
    text: str
    user_id: str | None = None


class ProcessResponse(BaseModel):
    input_text: str
    category: str
    response_text: str
    action_taken: str | None = None
    latency_ms: float


# =============================================================================
# Health & Status
# =============================================================================

@app.get("/health")
async def health():
    """Health check - verify all services are reachable"""
    services = {
        "stt": STT_URL,
        "tts": TTS_URL,
        "embeddings": EMBEDDINGS_URL,
        "router": ROUTER_URL,
        "reasoner": REASONER_URL,
        "chromadb": CHROMADB_URL
    }
    
    status = {}
    for name, url in services.items():
        try:
            resp = await http_client.get(f"{url}/health", timeout=5.0)
            status[name] = "healthy" if resp.status_code == 200 else "unhealthy"
        except Exception:
            status[name] = "unreachable"
    
    all_healthy = all(s == "healthy" for s in status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": status
    }


# =============================================================================
# Core Processing
# =============================================================================

@app.post("/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    """
    Process user text through the full pipeline.
    
    1. Classify intent (router)
    2. Route to appropriate handler
    3. Generate response (reasoner or direct action)
    4. Return result
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Classify intent
        classify_resp = await http_client.post(
            f"{ROUTER_URL}/classify",
            json={"text": request.text}
        )
        classify_resp.raise_for_status()
        classification = classify_resp.json()
        category = classification["category"]
        
        response_text = ""
        action_taken = None
        
        # Step 2: Route based on category
        if category == "domotica":
            # Search for matching command in ChromaDB
            embed_resp = await http_client.post(
                f"{EMBEDDINGS_URL}/embed",
                json={"texts": [request.text]}
            )
            embed_resp.raise_for_status()
            query_embedding = embed_resp.json()["embeddings"][0]

            # Query ChromaDB for nearest HA command
            chroma_resp = await http_client.post(
                f"{CHROMADB_URL}/api/v1/collections/home_assistant_commands/query",
                json={
                    "query_embeddings": [query_embedding],
                    "n_results": 1
                }
            )

            if chroma_resp.status_code == 200:
                results = chroma_resp.json()
                if results.get("documents") and results["documents"][0]:
                    # NOTE: Vector match found but HA execution is NOT implemented.
                    # The canonical runtime (src/main.py) uses HAClient to call
                    # the HA REST API. This Docker service only does vector search.
                    matched_doc = results["documents"][0][0]
                    action_taken = None
                    response_text = (
                        f"[EXPERIMENTAL] Matched command: {matched_doc}. "
                        "HA execution not implemented in Docker mode."
                    )
                    logger.warning(
                        "Domotica vector match found but HA execution not implemented "
                        "in Docker pipeline service"
                    )
                else:
                    response_text = "No encontre ese comando"
            else:
                response_text = "Error buscando comando"
                
        elif category == "rutina":
            # Create/manage routine
            chat_resp = await http_client.post(
                f"{REASONER_URL}/chat",
                json={
                    "messages": [
                        {"role": "system", "content": "Eres un asistente que ayuda a crear rutinas de domótica."},
                        {"role": "user", "content": request.text}
                    ],
                    "max_tokens": 512
                }
            )
            chat_resp.raise_for_status()
            response_text = chat_resp.json()["content"]
            action_taken = "routine_management"
            
        else:
            # General conversation - use reasoner
            chat_resp = await http_client.post(
                f"{REASONER_URL}/chat",
                json={
                    "messages": [
                        {"role": "system", "content": "Eres KZA, un asistente de voz inteligente para el hogar. Responde de forma concisa y útil."},
                        {"role": "user", "content": request.text}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
            )
            chat_resp.raise_for_status()
            response_text = chat_resp.json()["content"]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ProcessResponse(
            input_text=request.text,
            category=category,
            response_text=response_text,
            action_taken=action_taken,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/audio")
async def process_audio(audio: UploadFile = File(...)):
    """
    Process audio through full pipeline: STT -> Process -> TTS
    Returns audio response.
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Transcribe audio
        audio_data = await audio.read()
        files = {"audio": ("audio.wav", io.BytesIO(audio_data), "audio/wav")}
        
        stt_resp = await http_client.post(f"{STT_URL}/transcribe", files=files)
        stt_resp.raise_for_status()
        transcription = stt_resp.json()
        user_text = transcription["text"]
        
        logger.info(f"Transcribed: {user_text}")
        
        # Step 2: Process text
        process_resp = await http_client.post(
            "http://localhost:8080/process",
            json={"text": user_text}
        )
        process_resp.raise_for_status()
        result = process_resp.json()
        
        # Step 3: Synthesize response
        tts_resp = await http_client.post(
            f"{TTS_URL}/synthesize",
            json={"text": result["response_text"]}
        )
        tts_resp.raise_for_status()
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Full pipeline latency: {latency_ms:.0f}ms")
        
        return StreamingResponse(
            io.BytesIO(tts_resp.content),
            media_type="audio/wav",
            headers={
                "X-Transcription": user_text,
                "X-Category": result["category"],
                "X-Latency-Ms": str(latency_ms)
            }
        )
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket for Real-time Audio
# =============================================================================

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    
    Client sends audio chunks, server responds with audio.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            # Process through pipeline
            files = {"audio": ("chunk.wav", io.BytesIO(data), "audio/wav")}
            
            try:
                # STT
                stt_resp = await http_client.post(f"{STT_URL}/transcribe", files=files)
                if stt_resp.status_code != 200:
                    continue
                    
                text = stt_resp.json()["text"]
                if not text.strip():
                    continue
                
                # Process
                process_resp = await http_client.post(
                    "http://localhost:8080/process",
                    json={"text": text}
                )
                result = process_resp.json()
                
                # TTS
                tts_resp = await http_client.post(
                    f"{TTS_URL}/synthesize",
                    json={"text": result["response_text"]}
                )
                
                # Send response audio
                await websocket.send_bytes(tts_resp.content)
                
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# =============================================================================
# Direct Service Proxies
# =============================================================================

@app.post("/stt/transcribe")
async def proxy_stt(audio: UploadFile = File(...)):
    """Direct proxy to STT service"""
    files = {"audio": (audio.filename, await audio.read(), audio.content_type)}
    resp = await http_client.post(f"{STT_URL}/transcribe", files=files)
    return resp.json()


@app.post("/tts/synthesize")
async def proxy_tts(text: str):
    """Direct proxy to TTS service"""
    resp = await http_client.post(f"{TTS_URL}/synthesize", json={"text": text})
    return StreamingResponse(io.BytesIO(resp.content), media_type="audio/wav")


@app.post("/llm/chat")
async def proxy_chat(messages: list, max_tokens: int = 1024):
    """Direct proxy to reasoner service"""
    resp = await http_client.post(
        f"{REASONER_URL}/chat",
        json={"messages": messages, "max_tokens": max_tokens}
    )
    return resp.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
