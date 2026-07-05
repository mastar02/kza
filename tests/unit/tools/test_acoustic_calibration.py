"""Tests de las funciones de análisis del harness de calibración acústica.

Solo cubren la parte PURA (percentiles, gap voz-vs-ambiente, recomendación
de umbral, carga de JSONL). El loop de captura (sounddevice/openwakeword/
XvfController) es glue fino validado a mano en el server.
"""
import json

import pytest

from tools.acoustic_calibration import (
    summarize,
    signal_gap,
    load_condition,
)


class TestSummarize:
    def test_empty_samples(self):
        s = summarize([])
        assert s == {"count": 0, "p5": None, "p50": None, "p95": None, "max": None}

    def test_known_distribution(self):
        # 0..99 → p50 ≈ 49.5, max = 99
        s = summarize(list(range(100)))
        assert s["count"] == 100
        assert s["max"] == 99.0
        assert 48.0 <= s["p50"] <= 51.0
        assert 3.0 <= s["p5"] <= 6.0
        assert 93.0 <= s["p95"] <= 96.0


class TestSignalGap:
    def test_separable_signal(self):
        # Voz claramente arriba del ambiente: voz p5 > ambiente p95
        voice = [100.0 + i for i in range(50)]      # 100..149
        ambient = [1.0 + i * 0.1 for i in range(50)]  # 1..5.9
        g = signal_gap(voice, ambient)
        assert g["separable"] is True
        assert g["gap"] > 0
        # Umbral recomendado en el medio del gap
        assert g["ambient"]["p95"] < g["recommended_threshold"] < g["voice"]["p5"]

    def test_overlapping_signal_not_separable(self):
        voice = [10.0 + i * 0.1 for i in range(50)]
        ambient = [9.0 + i * 0.1 for i in range(50)]  # solapa con voz
        g = signal_gap(voice, ambient)
        assert g["separable"] is False
        assert g["recommended_threshold"] is None

    def test_empty_side_not_separable(self):
        g = signal_gap([], [1.0, 2.0])
        assert g["separable"] is False
        assert g["gap"] is None


class TestLoadCondition:
    def test_groups_rows_by_kind(self, tmp_path):
        rows = [
            {"t": 1.0, "kind": "rms", "value": 0.01},
            {"t": 1.0, "kind": "wake", "value": 0.3},
            {"t": 1.1, "kind": "spenergy", "value": 0.0},
            {"t": 1.2, "kind": "rms", "value": 0.02},
        ]
        f = tmp_path / "20260605_tv.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows))
        data = load_condition(tmp_path, "tv")
        assert data["rms"] == [0.01, 0.02]
        assert data["wake"] == [0.3]
        assert data["spenergy"] == [0.0]

    def test_missing_condition_returns_empty(self, tmp_path):
        data = load_condition(tmp_path, "voz")
        assert data == {"rms": [], "wake": [], "spenergy": []}

    def test_ignores_meta_rows(self, tmp_path):
        rows = [
            {"meta": True, "condition": "tv", "device": 4},
            {"t": 1.0, "kind": "rms", "value": 0.5},
        ]
        f = tmp_path / "20260605_tv.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows))
        data = load_condition(tmp_path, "tv")
        assert data["rms"] == [0.5]

    def test_glob_no_crossmatch_tv_vs_voz_tv(self, tmp_path):
        """Regresión: *_tv.jsonl NO debe cargar archivos *_voz_tv.jsonl y viceversa."""
        tv_rows = [{"t": 1.0, "kind": "rms", "value": 1.0}]
        voz_tv_rows = [{"t": 1.0, "kind": "rms", "value": 99.0}]
        (tmp_path / "20260605_tv.jsonl").write_text(
            "\n".join(json.dumps(r) for r in tv_rows)
        )
        (tmp_path / "20260605_voz_tv.jsonl").write_text(
            "\n".join(json.dumps(r) for r in voz_tv_rows)
        )
        tv_data = load_condition(tmp_path, "tv")
        assert tv_data["rms"] == [1.0], (
            "load_condition('tv') cargó datos de voz_tv — bug de glob"
        )
        voz_tv_data = load_condition(tmp_path, "voz_tv")
        assert voz_tv_data["rms"] == [99.0], (
            "load_condition('voz_tv') no cargó sus propios datos"
        )
