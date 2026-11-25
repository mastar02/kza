"""Entry point: ``python -m nexa_ml --socket %t/nexa/ml.sock --config /etc/nexa/nexa.toml``."""

import argparse
import asyncio
import logging
import os
import pathlib
import sys

from nexa_ml.config import DEFAULT_CONFIG_PATH, ConfigError, MlConfig, load_config
from nexa_ml.offline import enforce_offline_env

logger = logging.getLogger("nexa_ml")


def default_socket_path() -> pathlib.Path:
    """``$XDG_RUNTIME_DIR/nexa/ml.sock`` (falls back to ``/run/user/<uid>``)."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return pathlib.Path(runtime_dir) / "nexa" / "ml.sock"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(prog="nexa_ml", description="Nexa v2 stateless ML worker")
    parser.add_argument("--socket", type=pathlib.Path, default=None, help="UDS path for MlService")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG_PATH, help="nexa.toml path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def apply_offline_env(config: MlConfig) -> None:
    """Force (overwrite) Hugging Face offline mode before any model library is imported."""
    if config.offline:
        enforce_offline_env()


def main(argv: list[str] | None = None) -> int:
    """Run the worker until SIGTERM/SIGINT.

    Returns:
        Process exit code (0 clean stop, 2 configuration error).
    """
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    apply_offline_env(config)

    from nexa_ml.notify import SdNotifier  # noqa: PLC0415 - after env setup
    from nexa_ml.runtime import MlRuntime, real_loaders  # noqa: PLC0415
    from nexa_ml.server import serve  # noqa: PLC0415

    runtime = MlRuntime(config, real_loaders(config))
    socket_path = args.socket or default_socket_path()
    try:
        asyncio.run(serve(runtime, socket_path, SdNotifier()))
    except ValueError as exc:
        logger.error("Cannot serve on socket: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
