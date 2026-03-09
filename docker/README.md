# KZA Docker Services — EXPERIMENTAL

Docker mode is **experimental** and does **not** have full parity with the
canonical monolith runtime (`python -m src.main`).

## What works

| Service     | Port | GPU/CPU | Status |
|-------------|------|---------|--------|
| STT         | 8001 | GPU 0   | Transcription only (no wake word, no VAD) |
| Embeddings  | 8002 | GPU 1   | Text + speaker embeddings (no user identification) |
| Router      | 8003 | GPU 2   | Intent classification (uncalibrated confidence) |
| TTS         | 8004 | GPU 3   | Piper synthesis only (no streaming, no dual-engine) |
| Reasoner    | 8005 | CPU     | Chat + generation (no memory, no personality) |
| Pipeline    | 8080 | CPU     | Basic text pipeline (no HA execution, no speaker ID) |
| ChromaDB    | 8000 | CPU     | Vector storage |

## What is missing vs production

- Speaker identification and voice enrollment
- Emotion detection
- Multi-user orchestration (priority queue, cancellation)
- Memory system (short/long-term, user preferences)
- Lists and reminders
- Multi-room audio (per-room wake word, zone routing, MA1260)
- Spotify / music path
- Alerts (security, pattern, device monitoring)
- Nightly training (QLoRA, habit learning)
- Analytics (event logging, suggestions)
- HA command execution (vector search works, but API call is not wired)
- Streaming TTS, latency monitoring, timers, intercom, presence detection

Each service file in `docker/services/` has a `PARITY_GAPS` block listing
exactly what it lacks compared to the monolith.

## Running

```bash
# Required environment variables (create .env or export):
#   HOME_ASSISTANT_URL=http://192.168.1.100:8123
#   HOME_ASSISTANT_TOKEN=<your-token>

docker compose up --build
```

## Future

See **BL-013** for the planned approach to extract real workers from the
monolith so Docker services can reach full parity.

For production use today: `python -m src.main`
