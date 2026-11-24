#!/usr/bin/env python3
"""Offline side-by-side A/B audio & STT comparator (ch0/ch1 x Parakeet/Whisper/Hybrid).

Loads real 1ch or 2ch clips (FLAC/WAV) from a directory or an ``ambient.db``
export, splits ``ch0`` (Conference/NS) and ``ch1`` (ASR without NS) into
browser-playable WAV files, runs:

  1. ``parakeet`` — Parakeet-TDT-0.6B-v3 (CPU, anti-hallucination transducer)
  2. ``whisper``  — Whisper large-v3-turbo with forced ``language="es"``
  3. ``hybrid``   — Parakeet veto + Whisper ``es`` via ``apply_hybrid_veto``

and emits a self-contained offline ``index.html`` + ``comparison.json`` so the
operator can listen to both channels and quick-vote the winning channel and
transcription engine side by side.

Privacy: the output directory holds copies of household audio outside the
database TTL. Created with mode ``0700`` (files ``0600``); delete after review.

Usage:
    # From a folder of 2ch FLAC/WAV clips
    PYTHONPATH=. python3 tools/stt_ab_compare.py \\
        --input-dir data/ambient_audio --out data/ab_review --limit 40

    # Stratified from ambient.db (6 vad_prob buckets)
    PYTHONPATH=. python3 tools/stt_ab_compare.py \\
        --db data/ambient.db --out data/ab_review --per-bucket 7

    # Summarize an exported verdict.json
    PYTHONPATH=. python3 tools/stt_ab_compare.py --summarize data/ab_review/verdict.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.clip_io import (  # noqa: E402
    TARGET_SAMPLE_RATE as TARGET_SR,
    resample_linear_16k as _resample_16k,
    write_private_audio,
    write_private_bytes as _write_private_bytes,
)
from src.core.resource_retirement import ResourceOwner, ResourceSession  # noqa: E402
from src.stt.hybrid import STTEngine, apply_hybrid_veto  # noqa: E402
from src.stt.whisper_fast import STTResult  # noqa: E402
from tools.ambient_groundtruth import SAMPLEABLE_WHERE, sample_stratified  # noqa: E402

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".flac", ".wav"})
CHANNEL_LABELS: dict[str, str] = {
    "ch0": "ch0 · Conference (NS)",
    "ch1": "ch1 · ASR (sin NS)",
}


@dataclass(frozen=True)
class ClipSpec:
    """Metadata for one input clip to evaluate."""

    clip_id: str
    source_path: Path
    room_id: str | None = None
    vad_prob: float | None = None
    db_text: str | None = None


def _write_private_wav(path: Path, mono: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    write_private_audio(
        path, np.clip(mono, -1.0, 1.0), sample_rate, fmt="WAV", subtype="PCM_16"
    )


def read_clip_channels(path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Read a WAV/FLAC clip and return ``({'ch0': ..., 'ch1': ...}, duration_s)``."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0 or data.shape[0] == 0:
        raise ValueError(f"empty_audio:{path.name}")
    resampled = _resample_16k(data, sr)
    duration_s = round(resampled.shape[0] / TARGET_SR, 3)
    channels: dict[str, np.ndarray] = {"ch0": np.ascontiguousarray(resampled[:, 0])}
    if resampled.shape[1] >= 2:
        channels["ch1"] = np.ascontiguousarray(resampled[:, 1])
    return channels, duration_s


def collect_clips_from_dir(input_dir: Path, limit: int = 0) -> list[ClipSpec]:
    """Discover WAV/FLAC clips under ``input_dir``."""
    if not input_dir.is_dir():
        return []
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if limit > 0:
        files = files[:limit]
    clips: list[ClipSpec] = []
    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(input_dir)
        room = rel.parts[0] if len(rel.parts) >= 2 else None
        clips.append(ClipSpec(clip_id=f"clip_{idx:04d}_{path.stem}", source_path=path, room_id=room))
    return clips


def collect_clips_from_db(
    db_path: Path,
    *,
    per_bucket: int = 7,
    seed: int = 42,
    audio_base: Path | None = None,
) -> list[ClipSpec]:
    """Load stratified clips from ``ambient.db`` using the WER study buckets."""
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(r)
            for r in conn.execute(
                f"SELECT id, room_id, vad_prob, text, audio_path "
                f"FROM utterances WHERE {SAMPLEABLE_WHERE} ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()

    sampled = sample_stratified(rows, per_bucket=per_bucket, seed=seed)
    base = audio_base if audio_base is not None else Path.cwd()
    clips: list[ClipSpec] = []
    for row in sampled:
        raw_path = Path(row["audio_path"])
        resolved = raw_path if raw_path.is_absolute() else (base / raw_path)
        if not resolved.is_file():
            logger.warning("Missing audio file for utterance %s: %s", row["id"], resolved)
            continue
        clips.append(
            ClipSpec(
                clip_id=f"utt_{row['id']}",
                source_path=resolved,
                room_id=row.get("room_id"),
                vad_prob=row.get("vad_prob"),
                db_text=row.get("text"),
            )
        )
    return clips


def _make_optional_langid() -> Callable[[str], tuple[str, float]] | None:
    try:
        from src.ambient.distiller import make_langid_fn

        return make_langid_fn()
    except Exception:
        return None


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def evaluate_clips(
    clips: list[ClipSpec],
    out_dir: Path,
    *,
    parakeet: STTEngine,
    whisper: STTEngine,
    langid_fn: Callable[[str], tuple[str, float]] | None = None,
) -> dict:
    """Split channels, run Parakeet/Whisper/Hybrid on each channel, and build report."""
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    items: list[dict] = []
    metrics_acc: dict[tuple[str, str], dict] = {}

    def _record(ch_name: str, engine_name: str, text: str, ms: float, duration_s: float,
                lang: str | None, vetoed: bool = False) -> None:
        bucket = metrics_acc.setdefault(
            (ch_name, engine_name),
            {"latencies": [], "durations": [], "empties": 0, "vetoes": 0, "es_count": 0, "n": 0},
        )
        bucket["n"] += 1
        bucket["latencies"].append(float(ms))
        bucket["durations"].append(float(duration_s))
        if not text.strip():
            bucket["empties"] += 1
        if vetoed:
            bucket["vetoes"] += 1
        if lang == "es":
            bucket["es_count"] += 1

    def _detect_lang(text: str) -> tuple[str | None, float | None]:
        if langid_fn is None or not text.strip():
            return None, None
        try:
            lang, prob = langid_fn(text)
            return lang, round(float(prob), 3)
        except Exception:
            return None, None

    for clip in clips:
        try:
            channels, duration_s = read_clip_channels(clip.source_path)
        except Exception as error:
            logger.warning("Skipping unreadable clip %s: %s", clip.clip_id, type(error).__name__)
            continue

        channel_entries: dict[str, dict] = {}
        for ch_name, mono in channels.items():
            wav_rel = f"audio/{clip.clip_id}_{ch_name}.wav"
            _write_private_wav(out_dir / wav_rel, mono, TARGET_SR)

            pk_res: STTResult = parakeet.transcribe_with_confidence(mono, TARGET_SR)
            wh_res: STTResult = whisper.transcribe_with_confidence(mono, TARGET_SR)
            hy_res = apply_hybrid_veto(wh_res, pk_res)

            pk_text = (pk_res.text or "").strip()
            wh_text = (wh_res.text or "").strip()
            hy_text = (hy_res.text or "").strip()

            pk_lang, pk_lang_prob = _detect_lang(pk_text)
            wh_lang, wh_lang_prob = _detect_lang(wh_text)
            hy_lang, hy_lang_prob = _detect_lang(hy_text)

            _record(ch_name, "parakeet", pk_text, pk_res.elapsed_ms, duration_s, pk_lang)
            _record(ch_name, "whisper", wh_text, wh_res.elapsed_ms, duration_s, wh_lang)
            _record(ch_name, "hybrid", hy_text, hy_res.elapsed_ms, duration_s, hy_lang, vetoed=hy_res.vetoed)

            channel_entries[ch_name] = {
                "label": CHANNEL_LABELS.get(ch_name, ch_name),
                "audio_rel": wav_rel,
                "engines": {
                    "parakeet": {
                        "text": pk_text,
                        "elapsed_ms": round(float(pk_res.elapsed_ms), 1),
                        "lang": pk_lang,
                        "lang_prob": pk_lang_prob,
                    },
                    "whisper": {
                        "text": wh_text,
                        "elapsed_ms": round(float(wh_res.elapsed_ms), 1),
                        "lang": wh_lang,
                        "lang_prob": wh_lang_prob,
                        "compression_ratio": (
                            None if wh_res.compression_ratio is None else round(float(wh_res.compression_ratio), 2)
                        ),
                    },
                    "hybrid": {
                        "text": hy_text,
                        "elapsed_ms": round(float(hy_res.elapsed_ms), 1),
                        "vetoed": hy_res.vetoed,
                        "lang": hy_lang,
                        "lang_prob": hy_lang_prob,
                    },
                },
            }

        items.append(
            {
                "id": clip.clip_id,
                "room_id": clip.room_id,
                "vad_prob": None if clip.vad_prob is None else round(float(clip.vad_prob), 3),
                "duration_s": duration_s,
                "channel_count": len(channels),
                "db_text": clip.db_text,
                "channels": channel_entries,
            }
        )

    summary: dict[str, dict[str, dict]] = {}
    for (ch_name, engine_name), acc in sorted(metrics_acc.items()):
        total_audio = sum(acc["durations"]) or 1e-9
        total_proc_s = sum(acc["latencies"]) / 1000.0
        n = acc["n"] or 1
        summary.setdefault(ch_name, {})[engine_name] = {
            "n": acc["n"],
            "p50_ms": round(_pct(acc["latencies"], 50), 1),
            "p95_ms": round(_pct(acc["latencies"], 95), 1),
            "rtf": round(total_proc_s / total_audio, 4),
            "empty_pct": round(100.0 * acc["empties"] / n, 1),
            "veto_pct": round(100.0 * acc["vetoes"] / n, 1),
            "es_pct": round(100.0 * acc["es_count"] / n, 1),
        }

    run_id = datetime.now(timezone.utc).strftime("ab_%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(items),
        "summary": summary,
        "items": items,
    }
    _write_private_bytes(out_dir / "comparison.json", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    _write_private_bytes(out_dir / "index.html", render_comparison_html(payload).encode("utf-8"))
    return payload


def summarize_verdict(verdict: dict) -> dict:
    """Aggregate channel and engine votes from an exported ``verdict.json``."""
    votes = verdict.get("votes", {})
    if not isinstance(votes, dict):
        raise ValueError("invalid_verdict_votes")
    channel_counts = Counter()
    engine_counts = Counter()
    notes_count = 0
    for entry in votes.values():
        if not isinstance(entry, dict):
            continue
        ch = entry.get("channel_winner")
        eng = entry.get("engine_winner")
        if ch:
            channel_counts[ch] += 1
        if eng:
            engine_counts[eng] += 1
        if (entry.get("note") or "").strip():
            notes_count += 1
    return {
        "run_id": verdict.get("run_id"),
        "total_items": verdict.get("clip_count"),
        "voted_items": len(votes),
        "channel_votes": dict(channel_counts),
        "engine_votes": dict(engine_counts),
        "channel_leader": channel_counts.most_common(1)[0][0] if channel_counts else None,
        "engine_leader": engine_counts.most_common(1)[0][0] if engine_counts else None,
        "notes_count": notes_count,
    }


def render_comparison_html(payload: dict) -> str:
    """Build a self-contained offline HTML viewer for side-by-side listening and voting."""
    embedded_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>KZA — Comparador A/B Audio &amp; STT ({payload.get('run_id', '')})</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 16px; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 16px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
  h1 {{ margin: 0 0 6px; font-size: 1.25rem; }}
  .muted {{ color: var(--muted); font-size: 0.88rem; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
  .grid3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
  audio {{ width: 100%; height: 36px; margin-top: 6px; }}
  button {{
    background: #0b1220; color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 7px 12px; cursor: pointer; font-size: 0.9rem;
  }}
  button:hover {{ border-color: var(--accent); }}
  button.active {{ background: var(--accent); color: #0b1220; border-color: var(--accent); font-weight: 600; }}
  button.primary {{ background: var(--ok); color: #052e16; border-color: var(--ok); font-weight: 600; }}
  .badge {{
    display: inline-block; font-size: 0.75rem; padding: 2px 7px; border-radius: 999px;
    background: #0b1220; border: 1px solid var(--border); margin-right: 4px;
  }}
  .badge.veto {{ border-color: var(--warn); color: var(--warn); }}
  .badge.empty {{ border-color: var(--muted); color: var(--muted); }}
  .badge.es {{ border-color: var(--ok); color: var(--ok); }}
  .transcript {{
    margin-top: 8px; padding: 10px; border-radius: 7px; background: #0b1220;
    min-height: 54px; white-space: pre-wrap; line-height: 1.4;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  input[type="text"] {{
    width: 100%; padding: 8px 10px; border-radius: 7px; border: 1px solid var(--border);
    background: #0b1220; color: var(--text); font-size: 0.9rem;
  }}
  .vote-group {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
  .scoreboard {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.88rem; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="row">
      <div>
        <h1>KZA — Comparador A/B Audio (ch0 / ch1) &amp; STT</h1>
        <div class="muted" id="metaLine"></div>
      </div>
      <div class="row">
        <button id="filterBtn" onclick="toggleFilter()">Mostrar solo sin votar</button>
        <button class="primary" onclick="exportVerdict()">Exportar verdict.json</button>
      </div>
    </div>
    <div id="summaryTable" style="margin-top:10px;"></div>
    <div class="scoreboard" id="scoreboard" style="margin-top:10px;"></div>
  </div>

  <div class="card" id="clipCard">
    <div class="row">
      <div>
        <strong id="clipTitle"></strong>
        <span class="muted" id="clipMeta" style="margin-left:8px;"></span>
      </div>
      <div class="row">
        <button onclick="prevClip()">← Anterior</button>
        <span class="muted" id="counter"></span>
        <button onclick="nextClip()">Siguiente →</button>
      </div>
    </div>

    <div class="grid2" id="audioGrid" style="margin-top:12px;"></div>
    <div id="resultsGrid" style="margin-top:14px;"></div>

    <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px;">
      <div><strong>1. Mejor audio / canal:</strong> <span class="muted">(atajos: tecla 1 = ch0, tecla 2 = ch1)</span></div>
      <div class="vote-group" id="channelVotes"></div>

      <div style="margin-top:10px;"><strong>2. Mejor transcripción:</strong></div>
      <div class="vote-group" id="engineVotes"></div>

      <div style="margin-top:10px;">
        <input type="text" id="noteInput" placeholder="Nota o corrección breve (opcional)..." oninput="saveNote(this.value)" />
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {embedded_json};
const STORAGE_KEY = "kza_stt_ab_votes_" + (DATA.run_id || "default");
const CHANNEL_OPTIONS = [
  ["ch0", "ch0 (Conference)"],
  ["ch1", "ch1 (ASR)"],
  ["tie", "Empate"],
  ["noise", "Ruido / Ininteligible"]
];
const ENGINE_OPTIONS = [
  ["parakeet", "Parakeet-TDT"],
  ["whisper", "Whisper (es)"],
  ["hybrid", "Híbrido"],
  ["tie", "Empate"],
  ["none", "Todos mal / Ruido-TV"]
];

let votes = {{}};
try {{ votes = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }} catch {{ votes = {{}}; }}
let onlyUnvoted = false;
let currentIndex = 0;

function filteredItems() {{
  if (!onlyUnvoted) return DATA.items;
  return DATA.items.filter(it => !(votes[it.id] && votes[it.id].channel_winner && votes[it.id].engine_winner));
}}

function persistVotes() {{
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(votes)); }} catch {{}}
  renderScoreboard();
}}

function escapeHtml(s) {{
  return String(s ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}}

function renderSummaryTable() {{
  document.getElementById("metaLine").textContent =
    `Run ${{DATA.run_id}} · ${{DATA.clip_count}} clips · Parakeet vs Whisper (es) vs Híbrido`;
  const rows = [];
  for (const [ch, engines] of Object.entries(DATA.summary || {{}})) {{
    for (const [eng, st] of Object.entries(engines)) {{
      rows.push(`<tr>
        <td>${{escapeHtml(ch)}}</td><td>${{escapeHtml(eng)}}</td><td>${{st.n}}</td>
        <td>${{st.p50_ms}} ms</td><td>${{st.p95_ms}} ms</td><td>${{st.rtf}}</td>
        <td>${{st.empty_pct}}%</td><td>${{st.veto_pct}}%</td><td>${{st.es_pct}}%</td>
      </tr>`);
    }}
  }}
  document.getElementById("summaryTable").innerHTML = rows.length ? `
    <table>
      <thead><tr><th>Canal</th><th>Motor</th><th>N</th><th>p50</th><th>p95</th><th>RTF</th><th>Vacío</th><th>Veto</th><th>ES</th></tr></thead>
      <tbody>${{rows.join("")}}</tbody>
    </table>` : `<div class="muted">Sin clips procesados.</div>`;
}}

function renderScoreboard() {{
  const chCount = {{}}, engCount = {{}};
  let complete = 0;
  for (const v of Object.values(votes)) {{
    if (v.channel_winner) chCount[v.channel_winner] = (chCount[v.channel_winner] || 0) + 1;
    if (v.engine_winner) engCount[v.engine_winner] = (engCount[v.engine_winner] || 0) + 1;
    if (v.channel_winner && v.engine_winner) complete += 1;
  }}
  const fmt = (obj) => Object.entries(obj).map(([k, v]) => `${{k}}: ${{v}}`).join(" · ") || "sin votos";
  document.getElementById("scoreboard").innerHTML = `
    <span class="badge">Votados: ${{complete}} / ${{DATA.clip_count}}</span>
    <span class="badge">Canal → ${{escapeHtml(fmt(chCount))}}</span>
    <span class="badge">Motor → ${{escapeHtml(fmt(engCount))}}</span>`;
}}

function renderClip() {{
  const list = filteredItems();
  const card = document.getElementById("clipCard");
  if (!list.length) {{
    document.getElementById("clipTitle").textContent = "No hay clips en esta vista.";
    document.getElementById("clipMeta").textContent = "";
    document.getElementById("counter").textContent = "0 / 0";
    document.getElementById("audioGrid").innerHTML = "";
    document.getElementById("resultsGrid").innerHTML = "";
    document.getElementById("channelVotes").innerHTML = "";
    document.getElementById("engineVotes").innerHTML = "";
    return;
  }}
  if (currentIndex >= list.length) currentIndex = list.length - 1;
  if (currentIndex < 0) currentIndex = 0;
  const item = list[currentIndex];
  const vote = votes[item.id] || {{}};

  document.getElementById("clipTitle").textContent = item.id;
  document.getElementById("clipMeta").textContent =
    `room=${{item.room_id || "—"}} · dur=${{item.duration_s}}s · vad=${{item.vad_prob ?? "—"}} · canales=${{item.channel_count}}`;
  document.getElementById("counter").textContent = `${{currentIndex + 1}} / ${{list.length}}`;

  const chEntries = Object.entries(item.channels || {{}});
  document.getElementById("audioGrid").innerHTML = chEntries.map(([chKey, chData], idx) => `
    <div style="border:1px solid var(--border); border-radius:8px; padding:10px;">
      <div><strong>${{escapeHtml(chData.label)}}</strong> <span class="muted">[tecla ${{idx + 1}}]</span></div>
      <audio id="audio_${{chKey}}" controls preload="metadata" src="${{escapeHtml(chData.audio_rel)}}"></audio>
    </div>`).join("");

  document.getElementById("resultsGrid").innerHTML = chEntries.map(([chKey, chData]) => {{
    const cards = ["parakeet", "whisper", "hybrid"].map(engKey => {{
      const e = (chData.engines || {{}})[engKey] || {{}};
      const badges = [];
      badges.push(`<span class="badge">${{e.elapsed_ms ?? 0}} ms</span>`);
      if (e.lang) badges.push(`<span class="badge ${{e.lang === 'es' ? 'es' : ''}}">${{escapeHtml(e.lang)}} ${{e.lang_prob ?? ''}}</span>`);
      if (e.vetoed) badges.push(`<span class="badge veto">VETADO</span>`);
      if (!e.text) badges.push(`<span class="badge empty">VACÍO</span>`);
      return `<div style="border:1px solid var(--border); border-radius:8px; padding:10px;">
        <div class="row"><strong>${{engKey.toUpperCase()}}</strong><div>${{badges.join("")}}</div></div>
        <div class="transcript">${{e.text ? escapeHtml(e.text) : '<span class="muted">(vacío)</span>'}}</div>
      </div>`;
    }}).join("");
    return `<div style="margin-bottom:12px;">
      <div class="muted" style="margin-bottom:6px;">Transcripciones sobre <strong>${{escapeHtml(chData.label)}}</strong></div>
      <div class="grid3">${{cards}}</div>
    </div>`;
  }}).join("");

  document.getElementById("channelVotes").innerHTML = CHANNEL_OPTIONS.map(([val, label]) =>
    `<button class="${{vote.channel_winner === val ? 'active' : ''}}" onclick="setVote('${{item.id}}', 'channel_winner', '${{val}}')">${{label}}</button>`
  ).join("");
  document.getElementById("engineVotes").innerHTML = ENGINE_OPTIONS.map(([val, label]) =>
    `<button class="${{vote.engine_winner === val ? 'active' : ''}}" onclick="setVote('${{item.id}}', 'engine_winner', '${{val}}')">${{label}}</button>`
  ).join("");
  document.getElementById("noteInput").value = vote.note || "";
}}

function setVote(id, field, val) {{
  votes[id] = Object.assign({{}}, votes[id] || {{}}, {{ [field]: val }});
  persistVotes();
  renderClip();
}}

function saveNote(val) {{
  const list = filteredItems();
  if (!list.length) return;
  const id = list[currentIndex].id;
  votes[id] = Object.assign({{}}, votes[id] || {{}}, {{ note: val }});
  persistVotes();
}}

function prevClip() {{ currentIndex = Math.max(0, currentIndex - 1); renderClip(); }}
function nextClip() {{ currentIndex = Math.min(Math.max(0, filteredItems().length - 1), currentIndex + 1); renderClip(); }}
function toggleFilter() {{
  onlyUnvoted = !onlyUnvoted;
  currentIndex = 0;
  document.getElementById("filterBtn").textContent = onlyUnvoted ? "Mostrar todos" : "Mostrar solo sin votar";
  renderClip();
}}

function exportVerdict() {{
  const chCount = {{}}, engCount = {{}};
  for (const v of Object.values(votes)) {{
    if (v.channel_winner) chCount[v.channel_winner] = (chCount[v.channel_winner] || 0) + 1;
    if (v.engine_winner) engCount[v.engine_winner] = (engCount[v.engine_winner] || 0) + 1;
  }}
  const out = {{
    run_id: DATA.run_id,
    exported_at: new Date().toISOString(),
    clip_count: DATA.clip_count,
    channel_votes: chCount,
    engine_votes: engCount,
    votes: votes
  }};
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdict.json";
  a.click();
}}

document.addEventListener("keydown", (ev) => {{
  if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
  if (ev.key === "ArrowLeft") prevClip();
  else if (ev.key === "ArrowRight") nextClip();
  else if (ev.key === "1" || ev.key === "2") {{
    const el = document.getElementById(ev.key === "1" ? "audio_ch0" : "audio_ch1");
    if (el) {{ el.paused ? el.play() : el.pause(); }}
  }}
}});

renderSummaryTable();
renderScoreboard();
renderClip();
</script>
</body>
</html>
"""


def _build_engines(args, session: ResourceSession) -> tuple[STTEngine, STTEngine]:
    from src.stt.parakeet_stt import ParakeetSTT
    from src.stt.whisper_fast import FastWhisperSTT

    pk = ParakeetSTT(model_name=args.parakeet_model, language=args.language)
    session.register(ResourceOwner(pk, owns_resource=True, error_prefix="stt_ab_parakeet"))
    pk.load()

    wh = FastWhisperSTT(
        model=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute,
        language=args.language,
        beam_size=args.whisper_beam_size,
        initial_prompt=args.whisper_prompt,
        vad_filter=False,
    )
    session.register(ResourceOwner(wh, owns_resource=True, error_prefix="stt_ab_whisper"))
    wh.load()
    return pk, wh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-dir", type=Path, help="Folder of WAV/FLAC clips (1ch or 2ch)")
    mode.add_argument("--db", type=Path, help="Path to ambient.db with archived audio_path rows")
    mode.add_argument("--summarize", type=Path, help="Summarize an exported verdict.json")

    ap.add_argument("--out", type=Path, default=Path("data/ab_review"), help="Output directory (0700)")
    ap.add_argument("--limit", type=int, default=40, help="Max clips when scanning --input-dir (0 = all)")
    ap.add_argument("--per-bucket", type=int, default=7, help="Clips per vad_prob bucket with --db")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed for --db")
    ap.add_argument("--audio-base", type=Path, default=None, help="Base dir for relative audio_path in --db")

    ap.add_argument("--language", default="es")
    ap.add_argument("--parakeet-model", default="nemo-parakeet-tdt-0.6b-v3")
    ap.add_argument("--whisper-model", default="./models/whisper-v3-turbo")
    ap.add_argument("--whisper-device", default="cuda:0", help="cuda:0 (ambient GPU) | cpu")
    ap.add_argument("--whisper-compute", default="int8_float16")
    ap.add_argument("--whisper-beam-size", type=int, default=5)
    ap.add_argument("--whisper-prompt", default=None)
    args = ap.parse_args(argv)

    if args.summarize is not None:
        verdict = json.loads(args.summarize.read_text(encoding="utf-8"))
        summary = summarize_verdict(verdict)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.input_dir is not None:
        clips = collect_clips_from_dir(args.input_dir, limit=args.limit)
    else:
        clips = collect_clips_from_db(
            args.db, per_bucket=args.per_bucket, seed=args.seed, audio_base=args.audio_base
        )

    if not clips:
        print("ERROR: no readable clips found.", file=sys.stderr)
        return 1

    langid_fn = _make_optional_langid()
    with ResourceSession(error_prefix="stt_ab") as session:
        pk, wh = _build_engines(args, session)
        payload = evaluate_clips(clips, args.out, parakeet=pk, whisper=wh, langid_fn=langid_fn)

    print(f"Generated A/B comparison for {payload['clip_count']} clips in {args.out / 'index.html'}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
