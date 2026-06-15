# tests/unit/stt/test_initial_prompt_voseo.py
import yaml
from pathlib import Path


def _prompt() -> str:
    cfg = yaml.safe_load(Path("config/settings.yaml").read_text())
    return cfg["stt"]["initial_prompt"].lower()


def test_prompt_has_voseo_vocabulary():
    p = _prompt()
    for tok in ("prendé", "apagá", "subí", "bajá", "poné"):
        assert tok in p, f"falta voseo: {tok}"


def test_prompt_has_no_verbatim_command_phrases():
    # Guard anti-fantasma (incidente 2026-05-29): nunca frases-comando verbatim.
    p = _prompt()
    for banned in ("nexa prendé", "nexa subí", "nexa apagá"):
        assert banned not in p, f"frase-comando verbatim prohibida: {banned}"
