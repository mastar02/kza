#!/usr/bin/env python3
"""
Hardware Validation Script
Verifica que el hardware del sistema cumple con los requisitos.

Uso:
    python scripts/validate_hardware.py [--full]
"""

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    OK = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    INFO = "ℹ️"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    details: str = ""


def print_result(result: CheckResult):
    """Imprimir resultado de un check"""
    print(f"{result.status.value} {result.name}: {result.message}")
    if result.details:
        for line in result.details.split("\n"):
            print(f"   {line}")


def check_python_version() -> CheckResult:
    """Verificar versión de Python"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 13:
        return CheckResult(
            name="Python Version",
            status=Status.OK,
            message=f"{version_str} (>=3.13 requerido)"
        )
    else:
        return CheckResult(
            name="Python Version",
            status=Status.ERROR,
            message=f"{version_str} (>=3.13 requerido)"
        )


def check_cuda() -> CheckResult:
    """Verificar CUDA disponible"""
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return CheckResult(
                name="CUDA",
                status=Status.OK,
                message=f"CUDA {cuda_version} disponible"
            )
        else:
            return CheckResult(
                name="CUDA",
                status=Status.ERROR,
                message="CUDA no disponible"
            )
    except ImportError:
        return CheckResult(
            name="CUDA",
            status=Status.ERROR,
            message="PyTorch no instalado"
        )


def check_gpus() -> CheckResult:
    """Verificar GPUs disponibles"""
    try:
        import torch

        if not torch.cuda.is_available():
            return CheckResult(
                name="GPUs",
                status=Status.ERROR,
                message="CUDA no disponible"
            )

        gpu_count = torch.cuda.device_count()
        expected = 4

        details_lines = []
        total_vram = 0

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            total_vram += vram_gb
            details_lines.append(
                f"GPU {i}: {props.name} ({vram_gb:.1f} GB)"
            )

        details = "\n".join(details_lines)

        if gpu_count >= expected:
            return CheckResult(
                name="GPUs",
                status=Status.OK,
                message=f"{gpu_count} GPUs ({total_vram:.1f} GB total)",
                details=details
            )
        elif gpu_count > 0:
            return CheckResult(
                name="GPUs",
                status=Status.WARNING,
                message=f"{gpu_count}/{expected} GPUs ({total_vram:.1f} GB)",
                details=details
            )
        else:
            return CheckResult(
                name="GPUs",
                status=Status.ERROR,
                message="No hay GPUs disponibles"
            )

    except ImportError:
        return CheckResult(
            name="GPUs",
            status=Status.ERROR,
            message="PyTorch no instalado"
        )


def check_cpu() -> CheckResult:
    """Verificar CPU"""
    try:
        import psutil
    except ImportError:
        psutil = None

    cpu_count = os.cpu_count() or 0
    expected_threads = 48  # Threadripper PRO 7965WX

    # Intentar obtener info del CPU
    cpu_info = "Unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            cpu_info = result.stdout.strip()
    except Exception:
        pass

    details = f"Modelo: {cpu_info}"

    if cpu_count >= expected_threads:
        return CheckResult(
            name="CPU",
            status=Status.OK,
            message=f"{cpu_count} threads ({expected_threads} esperados)",
            details=details
        )
    elif cpu_count >= 16:
        return CheckResult(
            name="CPU",
            status=Status.WARNING,
            message=f"{cpu_count} threads ({expected_threads} esperados)",
            details=details
        )
    else:
        return CheckResult(
            name="CPU",
            status=Status.ERROR,
            message=f"Solo {cpu_count} threads",
            details=details
        )


def check_ram() -> CheckResult:
    """Verificar RAM"""
    try:
        import psutil

        ram = psutil.virtual_memory()
        total_gb = ram.total / (1024**3)
        available_gb = ram.available / (1024**3)
        expected_gb = 128

        details = f"Disponible: {available_gb:.1f} GB"

        if total_gb >= expected_gb:
            return CheckResult(
                name="RAM",
                status=Status.OK,
                message=f"{total_gb:.0f} GB ({expected_gb} GB esperados)",
                details=details
            )
        elif total_gb >= 64:
            return CheckResult(
                name="RAM",
                status=Status.WARNING,
                message=f"{total_gb:.0f} GB ({expected_gb} GB esperados)",
                details=details
            )
        else:
            return CheckResult(
                name="RAM",
                status=Status.ERROR,
                message=f"Solo {total_gb:.0f} GB",
                details=details
            )

    except ImportError:
        return CheckResult(
            name="RAM",
            status=Status.INFO,
            message="psutil no instalado - no se puede verificar"
        )


def check_disk_space() -> CheckResult:
    """Verificar espacio en disco"""
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        # Necesitamos ~100GB para modelos
        min_free = 100

        if free_gb >= min_free:
            return CheckResult(
                name="Disk Space",
                status=Status.OK,
                message=f"{free_gb:.0f} GB libres de {total_gb:.0f} GB"
            )
        elif free_gb >= 50:
            return CheckResult(
                name="Disk Space",
                status=Status.WARNING,
                message=f"{free_gb:.0f} GB libres (recomendado: {min_free}+ GB)"
            )
        else:
            return CheckResult(
                name="Disk Space",
                status=Status.ERROR,
                message=f"Solo {free_gb:.0f} GB libres"
            )

    except Exception as e:
        return CheckResult(
            name="Disk Space",
            status=Status.ERROR,
            message=str(e)
        )


def check_audio_devices() -> CheckResult:
    """Verificar dispositivos de audio"""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_count = sum(1 for d in devices if d["max_input_channels"] > 0)
        output_count = sum(1 for d in devices if d["max_output_channels"] > 0)

        if input_count > 0 and output_count > 0:
            try:
                default_input = sd.query_devices(kind='input')
                default_output = sd.query_devices(kind='output')
                details = f"Input: {default_input['name']}\nOutput: {default_output['name']}"
            except Exception:
                details = ""

            return CheckResult(
                name="Audio",
                status=Status.OK,
                message=f"{input_count} entradas, {output_count} salidas",
                details=details
            )
        elif input_count > 0:
            return CheckResult(
                name="Audio",
                status=Status.WARNING,
                message="Sin salida de audio"
            )
        else:
            return CheckResult(
                name="Audio",
                status=Status.ERROR,
                message="Sin micrófono disponible"
            )

    except ImportError:
        return CheckResult(
            name="Audio",
            status=Status.INFO,
            message="sounddevice no instalado"
        )
    except Exception as e:
        return CheckResult(
            name="Audio",
            status=Status.ERROR,
            message=str(e)
        )


def check_models(models_path: str = "./models") -> CheckResult:
    """Verificar modelos descargados"""
    from pathlib import Path

    path = Path(models_path)

    if not path.exists():
        return CheckResult(
            name="Models",
            status=Status.WARNING,
            message=f"Directorio {models_path} no existe"
        )

    # Buscar modelos
    gguf_files = list(path.glob("**/*.gguf"))
    onnx_files = list(path.glob("**/*.onnx"))

    model_count = len(gguf_files) + len(onnx_files)

    if model_count == 0:
        return CheckResult(
            name="Models",
            status=Status.WARNING,
            message="No se encontraron modelos",
            details=f"Busqué en: {path.absolute()}"
        )

    details_lines = []
    for f in gguf_files[:3]:
        size_gb = f.stat().st_size / (1024**3)
        details_lines.append(f"GGUF: {f.name} ({size_gb:.1f} GB)")
    for f in onnx_files[:3]:
        size_mb = f.stat().st_size / (1024**2)
        details_lines.append(f"ONNX: {f.name} ({size_mb:.1f} MB)")

    return CheckResult(
        name="Models",
        status=Status.OK,
        message=f"{model_count} modelos encontrados",
        details="\n".join(details_lines)
    )


def check_dependencies() -> CheckResult:
    """Verificar dependencias Python"""
    required = [
        "torch",
        "faster_whisper",
        "chromadb",
        "sentence_transformers",
        "sounddevice",
        "aiohttp",
        "pyyaml"
    ]

    missing = []
    installed = []

    for pkg in required:
        try:
            __import__(pkg)
            installed.append(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return CheckResult(
            name="Dependencies",
            status=Status.OK,
            message=f"{len(installed)}/{len(required)} paquetes instalados"
        )
    else:
        return CheckResult(
            name="Dependencies",
            status=Status.WARNING,
            message=f"Faltan: {', '.join(missing)}",
            details="Ejecuta: pip install -r requirements.txt"
        )


def run_full_validation():
    """Ejecutar validación completa"""
    print("=" * 60)
    print("🔍 VALIDACIÓN DE HARDWARE - Home Assistant Voice")
    print("=" * 60)
    print()

    checks = [
        check_python_version(),
        check_cuda(),
        check_gpus(),
        check_cpu(),
        check_ram(),
        check_disk_space(),
        check_audio_devices(),
        check_models(),
        check_dependencies()
    ]

    for result in checks:
        print_result(result)
        print()

    # Resumen
    ok_count = sum(1 for c in checks if c.status == Status.OK)
    warning_count = sum(1 for c in checks if c.status == Status.WARNING)
    error_count = sum(1 for c in checks if c.status == Status.ERROR)

    print("=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"  ✅ OK: {ok_count}")
    print(f"  ⚠️  Warnings: {warning_count}")
    print(f"  ❌ Errors: {error_count}")
    print()

    if error_count > 0:
        print("❌ El sistema NO está listo para producción")
        return 1
    elif warning_count > 0:
        print("⚠️  El sistema puede funcionar con limitaciones")
        return 0
    else:
        print("✅ El sistema está listo para producción")
        return 0


def run_quick_validation():
    """Validación rápida (solo críticos)"""
    checks = [
        check_cuda(),
        check_gpus(),
        check_dependencies()
    ]

    all_ok = all(c.status in [Status.OK, Status.WARNING] for c in checks)

    for result in checks:
        print_result(result)

    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validar hardware del sistema")
    parser.add_argument("--full", action="store_true", help="Validación completa")
    parser.add_argument("--models-path", default="./models", help="Ruta a modelos")
    args = parser.parse_args()

    if args.full:
        sys.exit(run_full_validation())
    else:
        sys.exit(run_quick_validation())
