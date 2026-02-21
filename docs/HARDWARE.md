# KZA — Especificaciones de Hardware

## CPU
- **AMD Ryzen Threadripper PRO 9965WX**
  - 24 cores / 48 threads
  - Arquitectura: Zen 5 (4nm)
  - Frecuencia Base: 4.2 GHz / Boost: 5.4 GHz
  - Cache L3: 128MB
  - TDP: 350W
  - Socket: sWRX8e
  - PCIe: 128 lanes PCIe 5.0
  - Memoria: 8 canales DDR5-6400

## Motherboard
- **ASUS Pro WS WRX90E-SAGE SE**
  - Chipset: AMD WRX90, Socket: sWRX8e (sTR5)
  - Form Factor: EEB
  - RAM: 8x DIMM DDR5, hasta 2TB
  - PCIe: 7x PCIe 5.0 x16 slots
  - Storage: 4x M.2 PCIe 5.0, 2x SlimSAS, 4x SATA
  - Red: 2x Intel 10GbE + 1x Realtek 1GbE (BMC)
  - USB: 2x USB4 40Gbps, 6x USB 10Gbps
  - IPMI/BMC (AST2600), PCIe Q-Release

## RAM
- **A-Tech 128GB (2x64GB) DDR5-5600 RDIMM ECC**
  - Tipo: RDIMM (Registered) - OBLIGATORIO para WRX90
  - ECC: Sí
  - Velocidad: 5600 MT/s (PC5-44800)
  - Configuración: 2Rx4 (Dual Rank), 1.1V
  - Canales activos: 2 de 8 (con 2 módulos)
  - Ancho de banda actual: ~102 GB/s
  - **Expansión planificada**: 4x64GB = 256GB (~205 GB/s)
  - **Máximo soportado**: 8x64GB = 512GB (~358 GB/s)

## GPUs (4x NVIDIA RTX 3070)

| GPU | Uso | VRAM | Modelo Asignado |
|-----|-----|------|-----------------|
| GPU 0 (cuda:0) | Speech-to-Text | 8GB | Faster-Whisper / Moonshine |
| GPU 1 (cuda:1) | Embeddings + Speaker ID + Emotion | 8GB | BGE-small + ECAPA-TDNN + wav2vec2 |
| GPU 2 (cuda:2) | Router/Clasificador | 8GB | Qwen2.5-7B-Instruct (vLLM) |
| GPU 3 (cuda:3) | Text-to-Speech | 8GB | Piper / XTTS-v2 |

Especificaciones RTX 3070: 5888 CUDA Cores, 8GB GDDR6, 448 GB/s, TDP 220W, PCIe 4.0 x16

## Audio
- **Micrófono**: ReSpeaker XVF3800 (XMOS, 4 mics, beamforming) — uno por habitación
- **Extensión USB**: Cat5e/6 hasta 60m para distribución
- **Amplificador**: Monoprice MA1260 (12 canales, 6 zonas, RS-232)
- **Bluetooth**: UGREEN BT 5.3 — uno por habitación para detección de presencia

## Resumen de Recursos

| Recurso | Disponible | Uso Esperado |
|---------|------------|--------------|
| CPU Cores | 24 (48 threads) | LLM 70B inference (~24 threads) |
| RAM | 128GB | LLM 70B Q4 (~45GB) + OS + buffers |
| VRAM Total | 32GB (4x8GB) | STT + Embeddings + Router + TTS |
| PCIe Lanes | 128 | 4 GPUs x 16 = 64 lanes usadas |

## PSU
- Potencia mínima: 1600W (CPU 350W + 4xGPU 880W + otros ~100W + 20% margen)
- Eficiencia: 80+ Platinum o Titanium

## Consumo Eléctrico
- Idle: ~200-300W
- Carga típica (voice + domótica): ~400-600W
- Carga máxima (LLM + todas GPUs): ~1300-1400W
