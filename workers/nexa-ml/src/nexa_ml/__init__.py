"""Nexa v2 stateless ML inference worker (``nexa.v1.MlService`` over a mode-0600 UDS).

Hard boundaries: zero network, zero SQLite/state, zero audio hardware and zero
business logic. Heavy ML libraries (torch, faster-whisper, kokoro, speechbrain,
sentence-transformers) are imported lazily inside the real model loaders only.
"""

__version__ = "0.1.0"
