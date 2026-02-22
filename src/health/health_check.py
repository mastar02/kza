"""
Health Check Module
Verifica el estado de todos los componentes del sistema.
"""

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    """Estado de salud de un componente"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Resultado de health check de un componente"""
    name: str
    status: HealthStatus
    message: str
    details: dict | None = None


@dataclass
class SystemHealth:
    """Resultado completo de health check del sistema"""
    status: HealthStatus
    components: list[ComponentHealth]

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details
                }
                for c in self.components
            ]
        }


class HealthChecker:
    """Verificador de salud del sistema"""

    def __init__(self, config: dict):
        self.config = config

    def check_all(self) -> SystemHealth:
        """Ejecutar todos los health checks"""
        components = []

        # Check each component
        components.append(self.check_home_assistant())
        components.append(self.check_gpus())
        components.append(self.check_models())
        components.append(self.check_chromadb())
        components.append(self.check_audio())

        # Determine overall status
        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return SystemHealth(status=overall, components=components)

    def check_home_assistant(self) -> ComponentHealth:
        """Verificar conexión con Home Assistant"""
        try:
            import requests

            url = self.config.get("home_assistant", {}).get("url")
            token = self.config.get("home_assistant", {}).get("token")

            if not url or not token:
                return ComponentHealth(
                    name="Home Assistant",
                    status=HealthStatus.UNHEALTHY,
                    message="URL o token no configurados"
                )

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{url}/api/", headers=headers, timeout=5)

            if response.status_code == 200:
                return ComponentHealth(
                    name="Home Assistant",
                    status=HealthStatus.HEALTHY,
                    message="Conectado correctamente"
                )
            else:
                return ComponentHealth(
                    name="Home Assistant",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Error HTTP {response.status_code}"
                )

        except requests.exceptions.Timeout:
            return ComponentHealth(
                name="Home Assistant",
                status=HealthStatus.UNHEALTHY,
                message="Timeout de conexión"
            )
        except requests.exceptions.ConnectionError:
            return ComponentHealth(
                name="Home Assistant",
                status=HealthStatus.UNHEALTHY,
                message="No se puede conectar"
            )
        except Exception as e:
            return ComponentHealth(
                name="Home Assistant",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )

    def check_gpus(self) -> ComponentHealth:
        """Verificar disponibilidad de GPUs"""
        try:
            import torch

            if not torch.cuda.is_available():
                return ComponentHealth(
                    name="GPUs",
                    status=HealthStatus.UNHEALTHY,
                    message="CUDA no disponible"
                )

            gpu_count = torch.cuda.device_count()
            expected_gpus = 4

            gpu_details = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                gpu_details.append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": round(memory_gb, 1)
                })

            if gpu_count >= expected_gpus:
                return ComponentHealth(
                    name="GPUs",
                    status=HealthStatus.HEALTHY,
                    message=f"{gpu_count} GPUs disponibles",
                    details={"gpus": gpu_details}
                )
            elif gpu_count > 0:
                return ComponentHealth(
                    name="GPUs",
                    status=HealthStatus.DEGRADED,
                    message=f"Solo {gpu_count}/{expected_gpus} GPUs disponibles",
                    details={"gpus": gpu_details}
                )
            else:
                return ComponentHealth(
                    name="GPUs",
                    status=HealthStatus.UNHEALTHY,
                    message="No hay GPUs disponibles"
                )

        except ImportError:
            return ComponentHealth(
                name="GPUs",
                status=HealthStatus.UNKNOWN,
                message="PyTorch no instalado"
            )
        except Exception as e:
            return ComponentHealth(
                name="GPUs",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )

    def check_models(self) -> ComponentHealth:
        """Verificar que los modelos están descargados"""
        models_path = Path(self.config.get("models_path", "./models"))

        if not models_path.exists():
            return ComponentHealth(
                name="Models",
                status=HealthStatus.UNHEALTHY,
                message=f"Directorio {models_path} no existe"
            )

        # Check for essential model files
        expected_patterns = [
            "*.gguf",       # LLM model
            "*.onnx",       # Piper TTS
        ]

        found_models = []
        for pattern in expected_patterns:
            matches = list(models_path.glob(f"**/{pattern}"))
            found_models.extend([m.name for m in matches])

        if not found_models:
            return ComponentHealth(
                name="Models",
                status=HealthStatus.UNHEALTHY,
                message="No se encontraron modelos",
                details={"path": str(models_path)}
            )

        return ComponentHealth(
            name="Models",
            status=HealthStatus.HEALTHY,
            message=f"{len(found_models)} modelos encontrados",
            details={"models": found_models[:10]}  # Limit list
        )

    def check_chromadb(self) -> ComponentHealth:
        """Verificar ChromaDB"""
        try:
            import chromadb

            chroma_path = self.config.get("vector_db", {}).get("path", "./data/chroma_db")

            # Try to create/connect to client
            client = chromadb.PersistentClient(path=chroma_path)

            # Check collections
            collections = client.list_collections()
            collection_names = [c.name for c in collections]

            return ComponentHealth(
                name="ChromaDB",
                status=HealthStatus.HEALTHY,
                message="Conectado correctamente",
                details={
                    "path": chroma_path,
                    "collections": collection_names
                }
            )

        except ImportError:
            return ComponentHealth(
                name="ChromaDB",
                status=HealthStatus.UNHEALTHY,
                message="chromadb no instalado"
            )
        except Exception as e:
            return ComponentHealth(
                name="ChromaDB",
                status=HealthStatus.UNHEALTHY,
                message=str(e)
            )

    def check_audio(self) -> ComponentHealth:
        """Verificar dispositivos de audio"""
        try:
            import sounddevice as sd

            devices = sd.query_devices()

            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            output_devices = [d for d in devices if d["max_output_channels"] > 0]

            if not input_devices:
                return ComponentHealth(
                    name="Audio",
                    status=HealthStatus.UNHEALTHY,
                    message="No hay dispositivos de entrada (micrófono)"
                )

            if not output_devices:
                return ComponentHealth(
                    name="Audio",
                    status=HealthStatus.DEGRADED,
                    message="No hay dispositivos de salida (parlantes)"
                )

            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')

            return ComponentHealth(
                name="Audio",
                status=HealthStatus.HEALTHY,
                message="Dispositivos de audio disponibles",
                details={
                    "input_count": len(input_devices),
                    "output_count": len(output_devices),
                    "default_input": default_input["name"],
                    "default_output": default_output["name"]
                }
            )

        except ImportError:
            return ComponentHealth(
                name="Audio",
                status=HealthStatus.UNKNOWN,
                message="sounddevice no instalado"
            )
        except Exception as e:
            return ComponentHealth(
                name="Audio",
                status=HealthStatus.UNKNOWN,
                message=str(e)
            )


def quick_check() -> bool:
    """
    Quick health check for Docker healthcheck.
    Returns True if system is operational, False otherwise.
    """
    try:
        # Basic checks without full config
        import chromadb
        import torch

        # Check CUDA
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return False

        # Check ChromaDB
        chroma_path = os.environ.get("CHROMA_PATH", "./data/chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)

        return True

    except Exception as e:
        logger.error(f"Quick check failed: {e}")
        return False


def run_health_check(config: dict) -> dict:
    """Run full health check and return results as dict"""
    checker = HealthChecker(config)
    result = checker.check_all()
    return result.to_dict()


if __name__ == "__main__":
    # Run standalone health check
    import json

    # Minimal config for standalone testing
    config = {
        "home_assistant": {
            "url": os.environ.get("HOME_ASSISTANT_URL"),
            "token": os.environ.get("HOME_ASSISTANT_TOKEN")
        },
        "models_path": os.environ.get("MODELS_PATH", "./models"),
        "vector_db": {
            "path": os.environ.get("CHROMA_PATH", "./data/chroma_db")
        }
    }

    checker = HealthChecker(config)
    result = checker.check_all()

    print(json.dumps(result.to_dict(), indent=2))

    # Exit with appropriate code
    exit(0 if result.is_healthy else 1)
