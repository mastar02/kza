#!/usr/bin/env python3
"""Offline HTML review tool for curated STT clips flagged ``needs_review=True``.

Generates a self-contained private (``0700``/``0600``) HTML reviewer from a
curated dataset directory produced by ``TrainingDatasetHub`` or
``tools/curate_ambient_dataset.py``:
  - Filters to clips where Whisper (``text``) and Parakeet (``alt_text_parakeet``)
    disagreed (``needs_review=True``), or includes all clips with ``--all``.
  - Highlights word-level differences side-by-side and embeds inline ``<audio>``
    playback for the selected ``ch0`` WAV.
  - Lets the operator pick Whisper, pick Parakeet, or edit the final ground
    truth (e.g. combining Whisper's commands with Parakeet's colloquialisms) and
    export ``corrections.json`` in one click for ``curate_ambient_dataset.py --corrections``.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import html
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.clip_io import write_private_bytes  # noqa: E402


def render_word_diff(primary_text: str, alt_text: str) -> tuple[str, str]:
    """Return ``(primary_html, alt_html)`` with differing spans highlighted."""
    a = (primary_text or "").split()
    b = (alt_text or "").split()
    sm = SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b])
    out_a: list[str] = []
    out_b: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a = html.escape(" ".join(a[i1:i2]))
        seg_b = html.escape(" ".join(b[j1:j2]))
        if tag == "equal":
            if seg_a:
                out_a.append(seg_a)
            if seg_b:
                out_b.append(seg_b)
        else:
            if seg_a:
                out_a.append(f'<mark class="diff-w">{seg_a}</mark>')
            if seg_b:
                out_b.append(f'<mark class="diff-p">{seg_b}</mark>')
    return " ".join(out_a), " ".join(out_b)


def build_review_html(
    curated_dir: Path,
    *,
    include_all: bool = False,
    out_html: Path | None = None,
) -> tuple[Path, int]:
    """Generate the offline review HTML page and return ``(html_path, item_count)``."""
    metadata_path = curated_dir / "stt" / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing_stt_metadata:{metadata_path}")

    items: list[dict] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if include_all or rec.get("needs_review") is True:
            primary = str(rec.get("text") or "")
            alt = str(rec.get("alt_text_parakeet") or "")
            diff_w, diff_p = render_word_diff(primary, alt) if alt else (html.escape(primary), "")
            items.append(
                {
                    "utterance_id": rec.get("utterance_id"),
                    "audio_rel": str(rec.get("audio_path") or ""),
                    "room_id": rec.get("room_id") or "unknown",
                    "speaker": rec.get("speaker") or "unknown",
                    "duration_s": rec.get("duration_s"),
                    "agreement_ratio": rec.get("agreement_ratio"),
                    "needs_review": bool(rec.get("needs_review")),
                    "whisper_text": primary,
                    "parakeet_text": alt,
                    "diff_whisper_html": diff_w,
                    "diff_parakeet_html": diff_p,
                }
            )

    target = out_html if out_html is not None else (curated_dir / "review.html")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)

    payload_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>KZA — Revisión STT (Whisper vs Parakeet)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ margin: 0 0 8px; font-size: 20px; }}
  .bar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 18px; position: sticky; top: 0; background: #0f172a; padding: 10px 0; border-bottom: 1px solid #334155; }}
  button {{ cursor: pointer; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; }}
  .btn-export {{ background: #10b981; color: #052e16; }}
  .btn-w {{ background: #3b82f6; color: #fff; }}
  .btn-p {{ background: #a855f7; color: #fff; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
  .meta {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; display: flex; gap: 12px; flex-wrap: wrap; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0; }}
  .box {{ background: #0f172a; border-radius: 6px; padding: 10px; font-size: 14px; line-height: 1.5; }}
  .box h3 {{ margin: 0 0 6px; font-size: 12px; text-transform: uppercase; color: #94a3b8; }}
  mark.diff-w {{ background: #1d4ed8; color: #fff; padding: 1px 3px; border-radius: 3px; }}
  mark.diff-p {{ background: #7e22ce; color: #fff; padding: 1px 3px; border-radius: 3px; }}
  textarea {{ width: 100%; min-height: 62px; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 8px; font-size: 14px; box-sizing: border-box; }}
  audio {{ width: 100%; height: 34px; margin: 4px 0; }}
  .actions {{ display: flex; gap: 8px; margin-top: 8px; align-items: center; }}
  .badge {{ padding: 2px 7px; border-radius: 999px; background: #f59e0b; color: #451a03; font-weight: 700; font-size: 11px; }}
</style>
</head>
<body>
  <h1>KZA — Revisión de Discrepancias STT (Whisper large-v3 vs Parakeet-TDT)</h1>
  <div class="bar">
    <span id="counter"></span>
    <button class="btn-export" onclick="exportCorrections()">Exportar corrections.json</button>
  </div>
  <div id="list"></div>
<script>
const ITEMS = {payload_json};
const state = {{}};

function render() {{
  const list = document.getElementById('list');
  document.getElementById('counter').textContent = `Clips en revisión: ${{ITEMS.length}}`;
  if (!ITEMS.length) {{
    list.innerHTML = '<p>No hay clips pendientes de revisión.</p>';
    return;
  }}
  list.innerHTML = ITEMS.map((it, idx) => {{
    const id = String(it.utterance_id);
    const val = state[id] ?? it.whisper_text;
    const agree = it.agreement_ratio != null ? `${{Math.round(it.agreement_ratio * 100)}}%` : 'n/a';
    return `
      <div class="card">
        <div class="meta">
          <span><strong>ID:</strong> ${{id}}</span>
          <span><strong>Room:</strong> ${{it.room_id}}</span>
          <span><strong>Speaker:</strong> ${{it.speaker}}</span>
          <span><strong>Duración:</strong> ${{it.duration_s ?? '?'}}s</span>
          <span><strong>Acuerdo:</strong> ${{agree}}</span>
          ${{it.needs_review ? '<span class="badge">needs_review</span>' : ''}}
        </div>
        <audio controls preload="none" src="${{it.audio_rel}}"></audio>
        <div class="cols">
          <div class="box">
            <h3>Whisper large-v3 (primario)</h3>
            <div>${{it.diff_whisper_html}}</div>
          </div>
          <div class="box">
            <h3>Parakeet-TDT (gate acústico)</h3>
            <div>${{it.diff_parakeet_html || '<em>(sin alt_text)</em>'}}</div>
          </div>
        </div>
        <textarea id="ta-${{idx}}" oninput="state['${{id}}']=this.value">${{val.replace(/</g, '&lt;')}}</textarea>
        <div class="actions">
          <button class="btn-w" onclick="pick(${{idx}}, '${{id}}', 'whisper')">Usar Whisper</button>
          <button class="btn-p" onclick="pick(${{idx}}, '${{id}}', 'parakeet')">Usar Parakeet</button>
        </div>
      </div>`;
  }}).join('');
}}

function pick(idx, id, which) {{
  const txt = which === 'parakeet' ? ITEMS[idx].parakeet_text : ITEMS[idx].whisper_text;
  state[id] = txt;
  document.getElementById('ta-' + idx).value = txt;
}}

function exportCorrections() {{
  const out = {{}};
  for (const it of ITEMS) {{
    const id = String(it.utterance_id);
    const txt = (state[id] ?? it.whisper_text).trim();
    if (txt) out[id] = txt;
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'corrections.json';
  a.click();
}}

render();
</script>
</body>
</html>
"""
    write_private_bytes(target, page.encode("utf-8"))
    return target, len(items)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated-dir", type=Path, required=True, help="Curated dataset hub directory")
    ap.add_argument("--out", type=Path, default=None, help="Output HTML path (default: <curated-dir>/review.html)")
    ap.add_argument("--all", action="store_true", help="Include all STT items, not only needs_review=true")
    args = ap.parse_args(argv)

    html_path, count = build_review_html(args.curated_dir, include_all=args.all, out_html=args.out)
    print(json.dumps({"html": str(html_path), "items": count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
