"""
STT Service - FastAPI wrapper for Speech-to-Text
Runs on GPU 0
"""

import io
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KZA STT Service", version="1.0.0")

# Global model instance
stt_model = None


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration_ms: float


def get_model():
    global stt_model
    if stt_model is None:
        from faster_whisper import WhisperModel
        model_name = os.getenv("MODEL_NAME", "distil-whisper/distil-small.en")
        device = os.getenv("DEVICE", "cuda")
        logger.info(f"Loading STT model: {model_name} on {device}")
        stt_model = WhisperModel(
            model_name,
            device=device,
            compute_type="float16"
        )
        logger.info("STT model loaded successfully")
    return stt_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup"""
    get_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "stt"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "es"
):
    """
    Transcribe audio to text.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Language code (default: es)
    
    Returns:
        Transcription result with text and metadata
    """
    import time
    start_time = time.time()
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Convert to numpy array
        import soundfile as sf
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torch
            import torchaudio
            audio_tensor = torch.from_numpy(audio_array).float()
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_array = resampler(audio_tensor).numpy()
        
        # Transcribe
        model = get_model()
        segments, info = model.transcribe(
            audio_array,
            language=language,
            beam_size=1,
            vad_filter=True
        )
        
        # Combine segments
        text = " ".join(segment.text for segment in segments).strip()
        
        duration_ms = (time.time() - start_time) * 1000
        
        return TranscriptionResponse(
            text=text,
            language=info.language,
            confidence=info.language_probability,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/bytes")
async def transcribe_bytes(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    language: str = "es"
):
    """
    Transcribe raw audio bytes (for internal service calls).
    """
    import time
    start_time = time.time()
    
    try:
        # Convert bytes to numpy
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        model = get_model()
        segments, info = model.transcribe(
            audio_array,
            language=language,
            beam_size=1
        )
        
        text = " ".join(segment.text for segment in segments).strip()
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "language": info.language,
            "duration_ms": duration_ms
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
