# Hardware y distribución de modelos de Nexa v2

Inventario verificado del servidor `192.168.1.2` al 2026-11-25. CPU AMD Threadripper PRO 7965WX (24 núcleos, 48 hilos), 125 GiB DDR5 y dos NVIDIA GeForce RTX 3070 de 8 GiB cada una. Driver NVIDIA `580.178.04`, CUDA `13.0`. Las cifras de VRAM son aproximadas y corresponden a cada tarjeta física.

| Dispositivo físico | Carga | VRAM aproximada |
| --- | --- | --- |
| `cuda:1` · PCI `C2:00.0` | llama-server Qwen2.5-7B-Instruct-Q4_K_M (`-ngl 99 -c 2048`, ~4,43 GiB), Whisper Command v3-turbo en `int8_float16` + Kokoro-82M en streaming (~1,51 GiB) | 5,94 / 8 GiB medidos |
| `cuda:0` · PCI `C1:00.0` | Dos instancias Whisper Ambient large-v3 en `int8_float16` (~3,8 GiB entre ambas) y una ECAPA-TDNN (~0,3 GiB) | 4,1 / 8 GiB |
| CPU | Embeddings BAAI/bge-m3. ONNX para Parakeet TDT 0.6B v3, Silero VAD y OpenWakeWord. | RAM del host |

El perfil de `llama-server` (`ik_llama.cpp`) usa `-t 8 -tb 8 -c 2048 -ctk q8_0 -ctv q8_0 -ngl 99` con todas las capas del 7B en GPU. La medición con Whisper Command y Kokoro deja unos 2,1 GiB libres. Al validar la asignación, contrastar los procesos con la dirección PCI física, porque CUDA puede renumerar los dispositivos visibles dentro de un servicio. Acordar cualquier reasignación antes de aplicarla.

## Audio por habitación

Los cuatro Seeed reSpeaker XVF3800 USB comparten VID:PID `2886:001a`. El puerto físico permite asociar cada entrada ALSA a su habitación.

| Habitación | Puerto USB |
| --- | --- |
| living | `1-2.3` |
| hall | `7-7.1` |
| cocina | `7-5.3` |
| escritorio | `3-1.4` |

El amplificador MA1260 está en modo `simulation`. La prueba de captura por habitación y la de salida física tienen criterios distintos. Ver el [despliegue nativo](DEPLOYMENT.md) y el [runbook de diagnóstico USB](../runbooks/2026-07-25-diagnostico-mic-mudo-isocronos.md).
