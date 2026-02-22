# KZA — Especificaciones de Hardware

## CPU
- **AMD Ryzen Threadripper PRO 7965WX**
  - 24 cores / 48 threads
  - Arquitectura: Zen 4 (5nm)
  - Frecuencia Base: 4.2 GHz / Boost: 5.3 GHz
  - Cache L3: 128MB
  - TDP: 350W
  - Socket: sTR5
  - PCIe: 128 lanes PCIe 5.0
  - Memoria: 8 canales DDR5-5200

## Motherboard
- **ASUS Pro WS WRX90E-SAGE SE**
  - Chipset: AMD WRX90, Socket: sTR5
  - Form Factor: EEB
  - RAM: 8x DIMM DDR5, hasta 2TB
  - PCIe: 7x PCIe 5.0 x16 slots
  - Storage: 4x M.2 PCIe 5.0, 2x SlimSAS, 4x SATA
  - Red: 2x Intel 10GbE + 1x Realtek 1GbE (BMC)
  - USB: 2x USB4 40Gbps, 6x USB 10Gbps
  - IPMI/BMC (AST2600), PCIe Q-Release

## RAM
- **NEMIX RAM 128GB (8x16GB) DDR5-5600 RDIMM**
  - Tipo: RDIMM (Registered) - OBLIGATORIO para WRX90
  - Velocidad: 5600 MT/s (PC5-44800)
  - Configuración: 1Rx8 (Single Rank), 1.1V
  - Canales activos: 8 de 8 (todos los slots ocupados)
  - Ancho de banda: ~358 GB/s (máximo para DDR5-5600 en 8 canales)
  - **Expansión**: Requiere reemplazar módulos (8x32GB = 256GB o 8x64GB = 512GB)

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
| CPU Cores | 24 (48 threads) | LLM 72B inference (~24 threads) |
| RAM | 128GB (8x16GB, 8 canales, ~358 GB/s) | LLM 72B Q6_K (~71GB total) + OS + servicios |
| VRAM Total | 32GB (4x8GB) | STT + Embeddings + Router + TTS |
| PCIe Lanes | 128 | 4 GPUs x 16 = 64 lanes usadas |

## PSU
- Potencia mínima: 1600W (CPU 350W + 4xGPU 880W + otros ~100W + 20% margen)
- Eficiencia: 80+ Platinum o Titanium

## Consumo Eléctrico
- Idle: ~200-300W
- Carga típica (voice + domótica): ~400-600W
- Carga máxima (LLM + todas GPUs): ~1300-1400W
