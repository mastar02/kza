"""
Embeddings Service - FastAPI wrapper for BGE + Speaker ID (EXPERIMENTAL)
Runs on GPU 1

WARNING: This is an EXPERIMENTAL Docker service. It does NOT have full parity
with the canonical runtime (src/main.py). Use src/main.py for production.

PARITY_GAPS vs canonical runtime:
  - Speaker embedding only — no user identification (canonical: SpeakerIdentifier
    matches against enrolled user profiles with similarity_threshold)
  - No voice enrollment flow (canonical: VoiceEnrollment with multi-sample enrollment)
  - No emotion detection (canonical: EmotionDetector wav2vec2 on same GPU)
  - No user profile management (canonical: UserManager with permissions, preferences)
  - Embeddings model defaults to bge-small-en (canonical: BGE-M3 multilingual)
  - No ChromaDB sync integration (canonical: ChromaSync keeps HA entities indexed)
  - Missing: latency tracking for embedding generation
"""

import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KZA Embeddings Service", version="1.0.0")

# Global model instances
embeddings_model = None
speaker_model = None


class EmbeddingRequest(BaseModel):
    texts: list[str]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


class SpeakerResponse(BaseModel):
    embedding: list[float]
    dimension: int


class SimilarityRequest(BaseModel):
    embedding1: list[float]
    embedding2: list[float]


def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5")
        device = os.getenv("DEVICE", "cuda")
        logger.info(f"Loading embeddings model: {model_name} on {device}")
        embeddings_model = SentenceTransformer(model_name, device=device)
        logger.info("Embeddings model loaded")
    return embeddings_model


def get_speaker_model():
    global speaker_model
    if speaker_model is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        model_name = os.getenv("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        device = os.getenv("DEVICE", "cuda")
        logger.info(f"Loading speaker model: {model_name}")
        speaker_model = SpeakerRecognition.from_hparams(
            source=model_name,
            run_opts={"device": device}
        )
        logger.info("Speaker model loaded")
    return speaker_model


@app.on_event("startup")
async def startup():
    """Pre-load models on startup"""
    get_embeddings_model()
    get_speaker_model()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "embeddings"}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for a list of texts.
    """
    try:
        model = get_embeddings_model()
        embeddings = model.encode(request.texts, convert_to_numpy=True)
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=embeddings.shape[1]
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speaker/embed", response_model=SpeakerResponse)
async def embed_speaker(audio: UploadFile = File(...)):
    """
    Generate speaker embedding from audio file.
    """
    import io
    import soundfile as sf
    import torch
    
    try:
        audio_data = await audio.read()
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        
        model = get_speaker_model()
        embedding = model.encode_batch(audio_tensor)
        
        return SpeakerResponse(
            embedding=embedding.squeeze().cpu().numpy().tolist(),
            dimension=embedding.shape[-1]
        )
    except Exception as e:
        logger.error(f"Speaker embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speaker/similarity")
async def speaker_similarity(request: SimilarityRequest):
    """
    Calculate cosine similarity between two speaker embeddings.
    """
    try:
        emb1 = np.array(request.embedding1)
        emb2 = np.array(request.embedding2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return {"similarity": float(similarity)}
    except Exception as e:
        logger.error(f"Similarity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
