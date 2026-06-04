"""CLI de tuning del XVF3800 (vendor interface, SOLO RAM).

Lee/escribe parámetros del DSP por nombre usando el mismo XvfController del
pipeline (src/audio/xvf_controller.py). Pensada para el leer-modificar-leer
del plan L-2/L-4 en el server, con el mic conectado.

⚠️ Los writes van a RAM del chip: re-enchufar el mic restaura el preset
persistido. SAVE_CONFIGURATION no está expuesto a propósito (issue #8 del
repo Seeed: puede dejar el device sin enumerar).

Uso (server, venv de kza):
    python -m tools.xvf_tune --list
    python -m tools.xvf_tune --read PP_AGCMAXGAIN
    python -m tools.xvf_tune --read-all
    python -m tools.xvf_tune --write PP_AGCMAXGAIN 16.0
    python -m tools.xvf_tune --write AUDIO_MGR_OP_L 0 1
"""
from __future__ import annotations

import argparse
import sys

from src.audio.xvf_controller import PARAMETERS, XvfController

_FLOAT_TYPES = ("float", "radians")


def parse_values(name: str, raw_values: list[str]) -> list:
    """Convierte los valores crudos de la CLI al tipo del parámetro.

    Args:
        name: Clave de ``PARAMETERS``.
        raw_values: Strings tal como llegan de argv.

    Returns:
        Lista de float o int según el tipo declarado del parámetro.

    Raises:
        ValueError: Parámetro desconocido o cantidad de valores incorrecta.
    """
    spec = PARAMETERS.get(name)
    if spec is None:
        raise ValueError(f"parámetro desconocido: {name!r} (ver --list)")
    _resid, _cmdid, count, _rw, dtype = spec
    if len(raw_values) != count:
        raise ValueError(f"{name} espera {count} valores, recibió {len(raw_values)}")
    conv = float if dtype in _FLOAT_TYPES else int
    return [conv(v) for v in raw_values]


def _print_param(name: str, vals: tuple | None) -> None:
    shown = "<sin respuesta — ¿device conectado? ¿udev rule?>" if vals is None else vals
    print(f"  {name:34s} = {shown}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tuning del XVF3800 por USB vendor interface (solo RAM).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="listar parámetros disponibles")
    group.add_argument("--read", metavar="NAME", help="leer un parámetro")
    group.add_argument("--read-all", action="store_true", help="leer todos los parámetros")
    group.add_argument(
        "--write", nargs="+", metavar=("NAME", "VALUE"),
        help="escribir un parámetro rw (RAM): NAME V1 [V2...]",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(f"{'NOMBRE':34s} {'RW':3s} {'TIPO':8s} CANT")
        for name, (_r, _c, count, rw, dtype) in PARAMETERS.items():
            print(f"{name:34s} {rw:3s} {dtype:8s} {count}")
        return 0

    ctrl = XvfController()
    if not ctrl.open():
        print("ERROR: XVF3800 no accesible (¿conectado? ¿pyusb? ¿udev rule MODE=0666?)")
        return 1

    if args.read:
        _print_param(args.read, ctrl.read_param(args.read))
        return 0

    if args.read_all:
        print("Snapshot del DSP (RAM actual):")
        for name in PARAMETERS:
            _print_param(name, ctrl.read_param(name))
        return 0

    # --write NAME V1 [V2...]
    name, *raw = args.write
    if not raw:
        print(f"ERROR: --write {name} necesita al menos un valor")
        return 2
    values = parse_values(name, raw)
    before = ctrl.read_param(name)
    ok = ctrl.write_param(name, values)
    after = ctrl.read_param(name)
    print(f"{name}: {before} → {after} (write {'OK' if ok else 'FALLÓ'}; RAM, "
          f"re-enchufar restaura el preset)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
