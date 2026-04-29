#!/usr/bin/env python3
"""
Benchmark de Latencia del Pipeline KZA

Mide la latencia real de cada componente del pipeline:
- STT (Speech-to-Text)
- Speaker ID
- Emotion Detection
- Router (clasificación)
- LLM (razonamiento)
- TTS (Text-to-Speech)
- Home Assistant

Uso:
    python tools/benchmark_latency.py [--iterations N] [--audio FILE]
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual"""
    component: str
    latency_ms: float
    success: bool
    details: str = ""


@dataclass
class BenchmarkSummary:
    """Resumen de benchmarks"""
    component: str
    iterations: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    success_rate: float


class LatencyBenchmark:
    """
    Benchmark de latencia del pipeline KZA.

    Mide cada componente individualmente y el pipeline completo.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.results: dict[str, list[BenchmarkResult]] = {}

        # Componentes (se cargan lazy)
        self._stt = None
        self._tts = None
        self._router = None
        self._llm = None
        self._speaker_id = None
        self._emotion = None
        self._ha = None

    def _generate_test_audio(self, duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
        """Generar audio de prueba (ruido simulando habla)"""
        samples = int(duration_s * sample_rate)
        # Generar "habla" simulada (ruido con envolvente)
        noise = np.random.randn(samples).astype(np.float32) * 0.3
        # Añadir envolvente para simular habla
        envelope = np.concatenate([
            np.linspace(0, 1, samples // 4),
            np.ones(samples // 2),
            np.linspace(1, 0, samples // 4)
        ])
        return noise * envelope[:len(noise)]

    async def benchmark_stt(self, audio: np.ndarray = None, iterations: int = 10) -> BenchmarkSummary:
        """Benchmark de Speech-to-Text"""
        print("\n📝 Benchmarking STT...")

        if self._stt is None:
            try:
                from src.stt.whisper_fast import FastWhisperSTT
                self._stt = FastWhisperSTT(
                    model="distil-whisper/distil-small.en",
                    device="cuda:0",
                    language="es"
                )
                self._stt.load()
            except Exception as e:
                print(f"  ⚠️ STT no disponible: {e}")
                return None

        if audio is None:
            audio = self._generate_test_audio(1.0)

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                text, _ = self._stt.transcribe(audio, 16000)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("stt", latency, True, text[:30]))
            except Exception as e:
                results.append(BenchmarkResult("stt", 0, False, str(e)))

        return self._summarize("STT", results)

    async def benchmark_stt_with_vad(self, audio: np.ndarray = None, iterations: int = 10) -> BenchmarkSummary:
        """Benchmark de STT con VAD temprano"""
        print("\n📝 Benchmarking STT + VAD...")

        if self._stt is None:
            return None

        if audio is None:
            # Audio con habla seguida de silencio
            speech = self._generate_test_audio(0.8)
            silence = np.zeros(int(0.4 * 16000), dtype=np.float32)
            audio = np.concatenate([speech, silence])

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                text, _, early = self._stt.transcribe_with_early_vad(audio, 16000)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("stt_vad", latency, True, f"early={early}"))
            except Exception as e:
                results.append(BenchmarkResult("stt_vad", 0, False, str(e)))

        return self._summarize("STT+VAD", results)

    async def benchmark_tts(self, text: str = "Hola, esta es una prueba de síntesis", iterations: int = 10) -> BenchmarkSummary:
        """Benchmark de Text-to-Speech"""
        print("\n🔊 Benchmarking TTS...")

        if self._tts is None:
            try:
                from src.tts.piper_tts import PiperTTS
                self._tts = PiperTTS()
                self._tts.load(warmup=True)
            except Exception as e:
                print(f"  ⚠️ TTS no disponible: {e}")
                return None

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                audio, _ = self._tts.synthesize(text)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("tts", latency, True, f"{len(audio)} samples"))
            except Exception as e:
                results.append(BenchmarkResult("tts", 0, False, str(e)))

        return self._summarize("TTS", results)

    async def benchmark_router(self, text: str = "prende la luz del living", iterations: int = 10) -> BenchmarkSummary:
        """Benchmark del Router/Clasificador"""
        print("\n🧠 Benchmarking Router...")

        if self._router is None:
            try:
                from src.llm.reasoner import FastRouter
                self._router = FastRouter(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    device="cuda:2",
                    enable_prefix_caching=True
                )
                self._router.load()
            except Exception as e:
                print(f"  ⚠️ Router no disponible: {e}")
                return None

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                needs_deep, response = self._router.classify_and_respond(text)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("router", latency, True, f"deep={needs_deep}"))
            except Exception as e:
                results.append(BenchmarkResult("router", 0, False, str(e)))

        return self._summarize("Router", results)

    async def benchmark_emotion(self, audio: np.ndarray = None, iterations: int = 10) -> BenchmarkSummary:
        """Benchmark de detección de emociones"""
        print("\n😊 Benchmarking Emotion Detection...")

        if self._emotion is None:
            try:
                from src.users.emotion_detector import EmotionDetector
                self._emotion = EmotionDetector(
                    model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                    device="cuda:0",
                )
                self._emotion.load()
            except Exception as e:
                print(f"  ⚠️ Emotion no disponible: {e}")
                return None

        if audio is None:
            audio = self._generate_test_audio(1.0)

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                result = self._emotion.detect(audio)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("emotion", latency, True, result.emotion))
            except Exception as e:
                results.append(BenchmarkResult("emotion", 0, False, str(e)))

        return self._summarize("Emotion", results)

    async def benchmark_emotion_batch(self, iterations: int = 5) -> BenchmarkSummary:
        """Benchmark de detección de emociones en batch"""
        print("\n😊 Benchmarking Emotion Batch...")

        if self._emotion is None:
            return None

        # Generar 3 audios de prueba
        audios = [self._generate_test_audio(1.0) for _ in range(3)]

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                batch_results = self._emotion.batch_detect(audios)
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("emotion_batch", latency, True, f"{len(batch_results)} results"))
            except Exception as e:
                results.append(BenchmarkResult("emotion_batch", 0, False, str(e)))

        return self._summarize("Emotion Batch (3)", results)

    async def benchmark_ha_call(self, iterations: int = 10) -> BenchmarkSummary:
        """Benchmark de llamada a Home Assistant"""
        print("\n🏠 Benchmarking Home Assistant...")

        if self._ha is None:
            try:
                from src.home_assistant.ha_client import HomeAssistantClient
                import yaml
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)
                ha_config = config.get("home_assistant", {})
                self._ha = HomeAssistantClient(
                    url=ha_config.get("url"),
                    token=ha_config.get("token")
                )
            except Exception as e:
                print(f"  ⚠️ HA no disponible: {e}")
                return None

        results = []
        for i in range(iterations):
            start = time.perf_counter()
            try:
                # Llamada simple para medir latencia
                states = await self._ha.get_states()
                latency = (time.perf_counter() - start) * 1000
                results.append(BenchmarkResult("ha", latency, True, f"{len(states)} states"))
            except Exception as e:
                results.append(BenchmarkResult("ha", 0, False, str(e)))

        return self._summarize("Home Assistant", results)

    def _summarize(self, name: str, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Generar resumen de resultados"""
        successful = [r for r in results if r.success]
        latencies = [r.latency_ms for r in successful]

        if not latencies:
            return BenchmarkSummary(
                component=name,
                iterations=len(results),
                min_ms=0, max_ms=0, avg_ms=0, median_ms=0, p95_ms=0, std_ms=0,
                success_rate=0
            )

        return BenchmarkSummary(
            component=name,
            iterations=len(results),
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            success_rate=len(successful) / len(results) * 100
        )

    async def run_full_benchmark(self, iterations: int = 10) -> dict:
        """Ejecutar benchmark completo"""
        print("=" * 60)
        print("🚀 KZA Latency Benchmark")
        print("=" * 60)
        print(f"Iterations per component: {iterations}")

        summaries = {}

        # STT
        summary = await self.benchmark_stt(iterations=iterations)
        if summary:
            summaries["stt"] = summary

        # STT + VAD
        summary = await self.benchmark_stt_with_vad(iterations=iterations)
        if summary:
            summaries["stt_vad"] = summary

        # TTS
        summary = await self.benchmark_tts(iterations=iterations)
        if summary:
            summaries["tts"] = summary

        # Router
        summary = await self.benchmark_router(iterations=iterations)
        if summary:
            summaries["router"] = summary

        # Emotion
        summary = await self.benchmark_emotion(iterations=iterations)
        if summary:
            summaries["emotion"] = summary

        # Emotion Batch
        summary = await self.benchmark_emotion_batch(iterations=iterations // 2)
        if summary:
            summaries["emotion_batch"] = summary

        # Home Assistant
        summary = await self.benchmark_ha_call(iterations=iterations)
        if summary:
            summaries["ha"] = summary

        # Mostrar resumen
        self._print_summary(summaries)

        return summaries

    def _print_summary(self, summaries: dict):
        """Imprimir resumen formateado"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE LATENCIAS")
        print("=" * 60)
        print(f"{'Componente':<20} {'Min':>8} {'Avg':>8} {'Med':>8} {'P95':>8} {'Max':>8} {'Success':>8}")
        print("-" * 60)

        total_avg = 0
        for name, s in summaries.items():
            print(f"{s.component:<20} {s.min_ms:>7.1f}ms {s.avg_ms:>7.1f}ms {s.median_ms:>7.1f}ms {s.p95_ms:>7.1f}ms {s.max_ms:>7.1f}ms {s.success_rate:>7.0f}%")
            total_avg += s.avg_ms

        print("-" * 60)

        # Estimar latencia total del pipeline
        pipeline_components = ["stt", "router", "tts"]
        pipeline_total = sum(summaries[c].avg_ms for c in pipeline_components if c in summaries)

        print(f"\n{'Pipeline estimado:':<20} {pipeline_total:>7.1f}ms (STT + Router + TTS)")

        # Comparar con target
        target = 300
        if pipeline_total <= target:
            print(f"✅ Dentro del objetivo de {target}ms")
        else:
            print(f"⚠️ Excede objetivo de {target}ms por {pipeline_total - target:.1f}ms")

        # Ahorro por optimizaciones
        if "stt" in summaries and "stt_vad" in summaries:
            vad_saving = summaries["stt"].avg_ms - summaries["stt_vad"].avg_ms
            if vad_saving > 0:
                print(f"\n💡 VAD early detection ahorra ~{vad_saving:.1f}ms en promedio")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark de latencia KZA")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Iteraciones por componente")
    parser.add_argument("--config", "-c", default="config/settings.yaml", help="Path a configuración")
    args = parser.parse_args()

    benchmark = LatencyBenchmark(config_path=args.config)
    await benchmark.run_full_benchmark(iterations=args.iterations)


if __name__ == "__main__":
    asyncio.run(main())
