"""
TTS Service - FastAPI wrapper for Text-to-Speech
Runs on GPU 3 with Piper
"""

import os
import io
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KZA TTS Service", version="1.0.0")

# Global model instance
tts_model = None


class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = "default"
    speed: float = 1.0


def get_model():
    global tts_model
    if tts_model is None:
        engine = os.getenv("TTS_ENGINE", "piper")
        
        if engine == "piper":
            from piper import PiperVoice
            model_name = os.getenv("PIPER_MODEL", "es_ES-davefx-medium")
            model_path = f"/app/models/piper/{model_name}.onnx"
            logger.info(f"Loading Piper model: {model_path}")
            tts_model = PiperVoice.load(model_path)
        else:
            # XTTS fallback
            from TTS.api import TTS
            tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            tts_model.to("cuda")
        
        logger.info("TTS model loaded")
    return tts_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup"""
    get_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tts"}


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.
    
    Returns: WAV audio stream
    """
    import wave
    import numpy as np
    
    try:
        model = get_model()
        engine = os.getenv("TTS_ENGINE", "piper")
        
        # Generate audio
        if engine == "piper":
            # Piper returns raw PCM data
            audio_buffer = io.BytesIO()
            
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(22050)
                
                for audio_chunk in model.synthesize_stream_raw(request.text):
                    wav_file.writeframes(audio_chunk)
            
            audio_buffer.seek(0)
        else:
            # XTTS
            wav = model.tts(
                text=request.text,
                language="es"
            )
            audio_buffer = io.BytesIO()
            import soundfile as sf
            sf.write(audio_buffer, wav, 22050, format='WAV')
            audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/bytes")
async def synthesize_bytes(request: SynthesizeRequest):
    """
    Synthesize speech and return raw bytes (for internal use).
    """
    import numpy as np
    
    try:
        model = get_model()
        engine = os.getenv("TTS_ENGINE", "piper")
        
        if engine == "piper":
            audio_chunks = []
            for chunk in model.synthesize_stream_raw(request.text):
                audio_chunks.append(chunk)
            audio_bytes = b''.join(audio_chunks)
        else:
            wav = model.tts(text=request.text, language="es")
            audio_bytes = np.array(wav, dtype=np.float32).tobytes()
        
        return {
            "audio_bytes": audio_bytes.hex(),
            "sample_rate": 22050,
            "channels": 1
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
