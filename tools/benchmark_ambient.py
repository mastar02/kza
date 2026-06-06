"""Benchmark Fase 0 — transcripción continua multi-pista (spec 2026-06-06).

Mide en el server (cuda:0) los números que el diseño exige antes de comprometer:
  --stt    RTF y VRAM de faster-whisper turbo sobre segmentos de voz
  --vad    throughput de Silero VAD (CPU) en ventanas de 512 samples
  --doa    throughput de GCC-PHAT (CPU) sobre pares de canales
  --smoke  todo lo anterior con audio sintético corto y device=cpu (laptop)

Uso (server):
  .venv/bin/python tools/benchmark_ambient.py --stt --device cuda:0 \
      --model /home/kza/kza/models/whisper-v3-turbo --wav sample.wav
  .venv/bin/python tools/benchmark_ambient.py --vad --doa
Uso (laptop): .venv/bin/python tools/benchmark_ambient.py --smoke
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("benchmark_ambient")

SAMPLE_RATE = 16000


def synth_speech(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Pseudo-voz sintética (tonos modulados) — suficiente para medir throughput.

    NO sirve para WER (eso se mide con --wav y audio real grabado del mic).
    """
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t)
    sig = 0.3 * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t))
    return sig.astype(np.float32)


def bench_stt(model: str, device: str, wav: str | None, iterations: int) -> None:
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)
    from src.stt.whisper_fast import FastWhisperSTT

    if wav:
        import soundfile as sf
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != SAMPLE_RATE:
            raise SystemExit(f"wav debe ser 16kHz (es {sr})")
    else:
        audio = synth_speech(8.0)
        logger.info("⚠️  audio sintético: RTF válido, WER no medible")

    stt = FastWhisperSTT(
        model=model, device=device, compute_type="int8_float16",
        language="es", beam_size=1, initial_prompt=None, vad_filter=False,
    )
    stt.load()
    is_cuda = "cuda" in device
    if is_cuda:
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.synchronize(dev_idx)
        vram_before = torch.cuda.memory_allocated(dev_idx)

    # warmup
    stt.transcribe(audio[: SAMPLE_RATE * 2])

    dur_s = len(audio) / SAMPLE_RATE
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        text, _ = stt.transcribe(audio)
        times.append(time.perf_counter() - t0)
    rtf = float(np.median(times)) / dur_s
    logger.info(f"STT {model} @ {device}")
    logger.info(f"  audio={dur_s:.1f}s iter={iterations}")
    logger.info(f"  latencia mediana={np.median(times)*1000:.0f}ms p95={np.percentile(times,95)*1000:.0f}ms")
    logger.info(f"  RTF={rtf:.3f}  (continuo viable si <0.5; cómodo si <0.2)")
    logger.info(f"  texto: {text[:80]!r}")
    if is_cuda:
        torch.cuda.synchronize(dev_idx)
        logger.info(f"  VRAM torch allocated: {torch.cuda.memory_allocated(dev_idx)/2**20:.0f} MiB "
                    f"(delta carga: ver nvidia-smi — ctranslate2 no reporta a torch)")
        logger.info("  ⚠️ medir VRAM real con: nvidia-smi --query-compute-apps=pid,used_memory --format=csv")


def bench_vad(iterations: int) -> None:
    from src.ambient.segmenter import make_silero_predictor

    predict = make_silero_predictor()
    chunk = synth_speech(0.08)  # 1280 samples = chunk del pipeline
    predict(chunk)  # warmup
    t0 = time.perf_counter()
    for _ in range(iterations):
        predict(chunk)
    per_chunk_ms = (time.perf_counter() - t0) / iterations * 1000
    logger.info(f"Silero VAD: {per_chunk_ms:.2f} ms/chunk de 80ms "
                f"({per_chunk_ms/80*100:.1f}% de un core — continuo viable si <50%)")


def bench_doa(iterations: int) -> None:
    from src.ambient.doa import DoAEstimator

    est = DoAEstimator()
    seg = np.random.default_rng(7).normal(0, 0.1, size=(SAMPLE_RATE * 3, 6)).astype(np.float32)
    est.estimate(seg)  # warmup
    t0 = time.perf_counter()
    for _ in range(iterations):
        est.estimate(seg)
    per_seg_ms = (time.perf_counter() - t0) / iterations * 1000
    logger.info(f"DoA GCC-PHAT: {per_seg_ms:.1f} ms por segmento de 3s (CPU)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stt", action="store_true")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--doa", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="todo en CPU con audio corto (laptop)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="./models/whisper-v3-turbo")
    ap.add_argument("--wav", default=None, help="audio real 16kHz mono para RTF/WER")
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args()

    if args.smoke:
        bench_doa(3)
        bench_vad(10)
        bench_stt(model=args.model, device="cpu", wav=args.wav, iterations=2)
        return
    if not (args.stt or args.vad or args.doa):
        ap.error("elegí al menos uno: --stt/--vad/--doa/--smoke")
    if args.doa:
        bench_doa(args.iterations)
    if args.vad:
        bench_vad(args.iterations)
    if args.stt:
        bench_stt(model=args.model, device=args.device, wav=args.wav, iterations=args.iterations)


if __name__ == "__main__":
    main()
