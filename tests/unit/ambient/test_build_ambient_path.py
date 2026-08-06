"""Test: I6 (review PR #15) — build_ambient_path()'s audio_dir wiring.

Finding: AmbientStore(audio_dir=...) es lo que activa `_sweep_orphans` en
producción (sin `audio_dir` el sweep se auto-desactiva silenciosamente,
ver `AmbientStore.purge_expired`). Antes de este test, borrar el kwarg
`audio_dir=ka_cfg["dir"]` en `build_ambient_path` dejaba la suite entera
verde — cero cobertura del wiring real que main.py depende.

No hace falta mockear los objetos "pesados" (SileroVAD, SpeakerIdentifier
ECAPA...) que advierte el finding: se inspeccionaron sus constructores y
todos cargan el modelo de forma perezosa vía un método `.load()` explícito
— `__init__` solo guarda atributos. El único objeto que se reemplaza acá es
`AmbientStore`, porque es el sujeto de la aserción (capturar los kwargs con
los que se lo construye) y porque abrir sqlite de verdad no aporta nada al
test.
"""
from __future__ import annotations

from pathlib import Path

import src.ambient.store as store_module
from src.ambient.transcriber import build_ambient_path


class _CapturingAmbientStore:
    """Reemplaza AmbientStore para capturar los kwargs del constructor."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _CapturingAmbientStore.last_kwargs = kwargs


def test_audio_dir_reaches_ambient_store(monkeypatch):
    monkeypatch.setattr(store_module, "AmbientStore", _CapturingAmbientStore)

    ambient_cfg = {
        "keep_audio": {"dir": "./data/ambient_audio_test", "enabled": True},
    }
    build_ambient_path(
        ambient_cfg=ambient_cfg,
        stt_base_cfg={},
        room_ids=["escritorio"],
        store_fact_fn=None,
    )

    assert _CapturingAmbientStore.last_kwargs is not None
    assert _CapturingAmbientStore.last_kwargs["audio_dir"] == "./data/ambient_audio_test"


def test_audio_dir_defaults_match_the_archiver_base_dir(monkeypatch):
    """Sin `keep_audio.dir` explícito, AmbientStore y AudioArchiver deben
    coincidir en el mismo default — si divergen, el sweep de huérfanos barre
    un directorio distinto del que el archiver escribe y nunca encuentra
    nada que barrer (falla silenciosa, el mismo patrón de "proxy mentiroso"
    ya documentado para este módulo)."""
    monkeypatch.setattr(store_module, "AmbientStore", _CapturingAmbientStore)

    path = build_ambient_path(
        ambient_cfg={},
        stt_base_cfg={},
        room_ids=["escritorio"],
        store_fact_fn=None,
    )

    assert Path(_CapturingAmbientStore.last_kwargs["audio_dir"]) == (
        path.transcriber._archiver.base_dir
    )
