"""Tests del script de cosecha de alucinaciones (carga por importlib: tools/ no es package)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "harvest_hallucinations",
    Path(__file__).resolve().parents[3] / "tools" / "harvest_hallucinations.py",
)
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)

ANSI = "\x1b[32m"
RESET = "\x1b[0m"

JOURNAL = f"""\
09:44:08.341 INFO src.nlu.command_gate [CommandGate] accept=True reason=ok no_speech=0.0 avg_logprob=-0.50 compression=0.5 would_reject=None text='Aplausos'
09:44:09.059 INFO {ANSI}...ipeline.request_router{RESET} [LLMRouter 718ms] is_command=False intent=None reason=None text='Aplausos'
09:52:28.541 INFO src.nlu.command_gate [CommandGate] accept=True reason=ok text='Nexa, prendé la luz.'
09:52:29.100 INFO ...ipeline.request_router [LLMRouter 500ms] is_command=True intent=turn_on reason=None text='Nexa, prendé la luz.'
10:01:00.000 INFO ...ipeline.request_router [LLMRouter 600ms] is_command=False intent=None reason=None text='Aplausos'
10:02:00.000 INFO ...ipeline.request_router [LLMRouter 600ms] is_command=False intent=None reason=None text='¡Aplausos!'
10:03:00.000 INFO ...ipeline.request_router [LLMRouter 610ms] is_command=False intent=None reason=None text='Gracias por ver el video.'
10:04:00.000 INFO ...ipeline.request_router [LLMRouter 620ms] is_command=False intent=None reason=None text='Una sola vez'
"""


def test_strip_ansi():
    assert harvest.strip_ansi(f"{ANSI}hola{RESET}") == "hola"


def test_parse_counts_router_rejects_normalized():
    cands = harvest.parse_candidates(JOURNAL.splitlines())
    # 'Aplausos' x2 + '¡Aplausos!' x1 agrupan por normalización
    assert cands["aplausos"].count == 3
    assert cands["aplausos"].first_seen == "09:44:09.059"
    assert cands["aplausos"].last_seen == "10:02:00.000"
    # comandos reales (is_command=True) NO son candidatas
    assert "nexa prende la luz" not in cands


def test_filter_excludes_blocked_and_low_count():
    cands = harvest.parse_candidates(JOURNAL.splitlines())
    out = harvest.filter_candidates(
        cands,
        min_count=2,
        blocked_phrases=("gracias por ver",),
        filler_words=frozenset(),
    )
    texts = [c.text for c in out]
    assert texts == ["aplausos"]          # única con count>=2 no bloqueada
    # 'gracias por ver el video' excluida por substring de blocked_phrases
    # 'una sola vez' excluida por min_count


def test_filter_excludes_fillers():
    cands = harvest.parse_candidates(
        ["10:00:00.000 X [LLMRouter 1ms] is_command=False intent=None reason=None text='Dale'"]
    )
    out = harvest.filter_candidates(
        cands, min_count=1, blocked_phrases=(), filler_words=frozenset({"dale"})
    )
    assert out == []


def test_parse_handles_apostrophe_text_double_quoted_repr():
    # repr() usa comillas dobles cuando el texto tiene apóstrofe
    line = (
        "10:00:00.000 INFO x [LLMRouter 1ms] is_command=False intent=None "
        "reason=None text=\"don't forget to subscribe\" (request_router.py:524)"
    )
    cands = harvest.parse_candidates([line])
    assert "don t forget to subscribe" in cands


def test_parse_handles_escaped_quote_and_trailing_suffix():
    # repr() escapa la comilla interna; el journal agrega sufijo (file:line)
    line = (
        "10:01:00.000 INFO x [LLMRouter 2ms] is_command=False intent=None "
        "reason=None text='she said \"it\\'s fine\" ok' (request_router.py:524)"
    )
    cands = harvest.parse_candidates([line])
    assert any("she said" in k and "fine" in k for k in cands)


def test_parse_ignores_malformed_repr():
    line = (
        "10:02:00.000 INFO x [LLMRouter 3ms] is_command=False intent=None "
        "reason=None text='truncad"
    )
    assert harvest.parse_candidates([line]) == {}
