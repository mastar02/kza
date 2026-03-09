"""
Structured Logging - Sistema de logging con contexto estructurado.

Proporciona:
- Logs en formato JSON para producción
- Logs coloridos para desarrollo
- Contexto automático (request_id, user_id, zone_id)
- Métricas de timing integradas

Uso:
    from src.core.logging import get_logger, LogContext

    logger = get_logger(__name__)

    # Log simple
    logger.info("Mensaje simple")

    # Log con contexto
    with LogContext(user_id="user_123", request_id="req_abc"):
        logger.info("Procesando petición")  # Incluye user_id y request_id

    # Log con timing
    with logger.timed("operacion_lenta"):
        # ... código ...
        pass  # Log automático con duración
"""

import json
import logging
import re
import sys
import time
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

# Context variables para datos de request
_request_context: ContextVar[dict[str, Any]] = ContextVar('request_context', default={})


class LogFormat(StrEnum):
    """Formatos de salida de logs"""
    JSON = "json"       # Para producción, parseable
    COLORED = "colored" # Para desarrollo, legible
    PLAIN = "plain"     # Sin colores


@dataclass
class LogConfig:
    """Configuración del sistema de logging"""
    level: str = "INFO"
    format: LogFormat = LogFormat.COLORED
    include_timestamp: bool = True
    include_location: bool = True  # file:line
    include_context: bool = True   # user_id, request_id, etc.
    json_indent: int | None = None  # None = una línea


# Colores ANSI
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Niveles
    DEBUG = "\033[36m"     # Cyan
    INFO = "\033[32m"      # Green
    WARNING = "\033[33m"   # Yellow
    ERROR = "\033[31m"     # Red
    CRITICAL = "\033[35m"  # Magenta

    # Componentes
    TIMESTAMP = "\033[90m" # Gray
    NAME = "\033[34m"      # Blue
    CONTEXT = "\033[33m"   # Yellow


class StructuredFormatter(logging.Formatter):
    """
    Formatter que produce logs estructurados en JSON o texto colorido.

    Automatically sanitizes sensitive patterns (Bearer tokens, known secret
    key names) from log messages to prevent accidental secret leakage.
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }

    # Pattern to detect and mask Bearer tokens in log messages
    _BEARER_RE = re.compile(r"Bearer\s+(\S{8,})")

    def __init__(self, config: LogConfig = None):
        super().__init__()
        self.config = config or LogConfig()

    @classmethod
    def _sanitize_message(cls, message: str) -> str:
        """Remove secrets that may have leaked into a log message."""
        return cls._BEARER_RE.sub(
            lambda m: f"Bearer {m.group(1)[:4]}***", message
        )

    def format(self, record: logging.LogRecord) -> str:
        # Obtener contexto actual
        ctx = _request_context.get()

        # Construir datos del log — sanitize the message
        log_data = {
            "message": self._sanitize_message(record.getMessage()),
            "level": record.levelname,
            "logger": record.name,
        }

        if self.config.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.config.include_location:
            log_data["location"] = f"{record.filename}:{record.lineno}"

        if self.config.include_context and ctx:
            log_data["context"] = ctx

        # Agregar exception info si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Agregar datos extra si los hay
        if hasattr(record, 'extra_data'):
            log_data["data"] = record.extra_data

        # Formatear según configuración
        if self.config.format == LogFormat.JSON:
            return json.dumps(log_data, default=str, indent=self.config.json_indent)
        elif self.config.format == LogFormat.COLORED:
            return self._format_colored(record, log_data)
        else:
            return self._format_plain(record, log_data)

    def _format_colored(self, record: logging.LogRecord, data: dict) -> str:
        """Formato colorido para desarrollo"""
        parts = []

        # Timestamp
        if self.config.include_timestamp:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            parts.append(f"{Colors.TIMESTAMP}{ts}{Colors.RESET}")

        # Level
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        level = f"{level_color}{record.levelname:8}{Colors.RESET}"
        parts.append(level)

        # Logger name (shortened)
        name = record.name
        if len(name) > 25:
            name = "..." + name[-22:]
        parts.append(f"{Colors.NAME}{name:25}{Colors.RESET}")

        # Message
        parts.append(data["message"])

        # Context
        if data.get("context"):
            ctx_str = " ".join(f"{k}={v}" for k, v in data["context"].items())
            parts.append(f"{Colors.CONTEXT}[{ctx_str}]{Colors.RESET}")

        # Location
        if self.config.include_location:
            parts.append(f"{Colors.DIM}({data['location']}){Colors.RESET}")

        result = " ".join(parts)

        # Exception
        if data.get("exception"):
            result += f"\n{Colors.ERROR}{data['exception']}{Colors.RESET}"

        return result

    def _format_plain(self, record: logging.LogRecord, data: dict) -> str:
        """Formato plano sin colores"""
        parts = []

        if self.config.include_timestamp:
            parts.append(data["timestamp"])

        parts.append(f"[{record.levelname:8}]")
        parts.append(f"[{record.name}]")
        parts.append(data["message"])

        if data.get("context"):
            ctx_str = " ".join(f"{k}={v}" for k, v in data["context"].items())
            parts.append(f"[{ctx_str}]")

        if self.config.include_location:
            parts.append(f"({data['location']})")

        result = " ".join(parts)

        if data.get("exception"):
            result += f"\n{data['exception']}"

        return result


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter que agrega contexto automáticamente.

    Uso:
        logger = get_logger(__name__)
        logger.info("mensaje", user_id="user_123", action="login")
    """

    def process(self, msg, kwargs):
        # Agregar datos extra al record
        extra = kwargs.get('extra', {})

        # Extraer datos especiales del kwargs
        extra_data = {}
        special_keys = ['user_id', 'request_id', 'zone_id', 'action', 'duration_ms', 'data']

        for key in list(kwargs.keys()):
            if key in special_keys:
                extra_data[key] = kwargs.pop(key)

        if extra_data:
            extra['extra_data'] = extra_data

        kwargs['extra'] = extra
        return msg, kwargs

    @contextmanager
    def timed(self, operation: str, level: int = logging.INFO, **extra_context):
        """
        Context manager para logging con timing automático.

        Uso:
            with logger.timed("vector_search"):
                results = chroma.search(query)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.log(
                level,
                f"{operation} completed",
                duration_ms=round(duration_ms, 2),
                **extra_context
            )


class LogContext:
    """
    Context manager para establecer contexto de logging.

    Uso:
        with LogContext(user_id="user_123", request_id="req_abc"):
            logger.info("Procesando")  # Incluye user_id y request_id
            do_something()
            logger.info("Completado")  # También incluye el contexto
    """

    def __init__(self, **context):
        self.context = context
        self.token = None

    def __enter__(self):
        # Merge con contexto existente
        current = _request_context.get()
        new_context = {**current, **self.context}
        self.token = _request_context.set(new_context)
        return self

    def __exit__(self, *args):
        _request_context.reset(self.token)


def set_context(**context):
    """Establecer contexto de forma permanente (hasta reset)"""
    current = _request_context.get()
    _request_context.set({**current, **context})


def clear_context():
    """Limpiar todo el contexto"""
    _request_context.set({})


def get_context() -> dict[str, Any]:
    """Obtener contexto actual"""
    return _request_context.get()


# Configuración global
_config: LogConfig = LogConfig()
_initialized: bool = False


def configure_logging(
    level: str = "INFO",
    format: LogFormat = LogFormat.COLORED,
    include_timestamp: bool = True,
    include_location: bool = True,
    include_context: bool = True,
    json_indent: int | None = None
):
    """
    Configurar el sistema de logging.

    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Formato de salida (JSON, COLORED, PLAIN)
        include_timestamp: Incluir timestamp en logs
        include_location: Incluir file:line
        include_context: Incluir contexto (user_id, etc.)
        json_indent: Indentación para JSON (None = una línea)
    """
    global _config, _initialized

    _config = LogConfig(
        level=level,
        format=format,
        include_timestamp=include_timestamp,
        include_location=include_location,
        include_context=include_context,
        json_indent=json_indent
    )

    # Configurar root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    # Remover handlers existentes
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Agregar nuevo handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(_config))
    root.addHandler(handler)

    _initialized = True


def get_logger(name: str) -> ContextLogger:
    """
    Obtener logger con soporte para contexto estructurado.

    Args:
        name: Nombre del logger (típicamente __name__)

    Returns:
        ContextLogger configurado
    """
    global _initialized

    if not _initialized:
        # Configuración por defecto
        configure_logging()

    logger = logging.getLogger(name)
    return ContextLogger(logger, {})


def generate_request_id() -> str:
    """Generar ID único para una petición"""
    return str(uuid.uuid4())[:8]


# Atajos convenientes
def debug(msg: str, **kwargs):
    get_logger("app").debug(msg, **kwargs)


def info(msg: str, **kwargs):
    get_logger("app").info(msg, **kwargs)


def warning(msg: str, **kwargs):
    get_logger("app").warning(msg, **kwargs)


def error(msg: str, **kwargs):
    get_logger("app").error(msg, **kwargs)


def critical(msg: str, **kwargs):
    get_logger("app").critical(msg, **kwargs)
