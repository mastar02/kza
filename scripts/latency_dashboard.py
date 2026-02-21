#!/usr/bin/env python3
"""
KZA Voice - Latency Dashboard CLI
Muestra estadísticas de latencia en tiempo real.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.latency_monitor import LatencyMonitor


def clear_screen():
    """Clear terminal screen safely"""
    subprocess.run(["clear"], check=False)


def main():
    parser = argparse.ArgumentParser(description="KZA Latency Dashboard")
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Modo watch (actualización continua)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=2.0,
        help="Intervalo de actualización en segundos (default: 2)"
    )
    parser.add_argument(
        "-d", "--db",
        type=str,
        default="./data/latency.db",
        help="Path a la base de datos de latencia"
    )
    parser.add_argument(
        "--history",
        type=int,
        metavar="HOURS",
        help="Mostrar estadísticas históricas de las últimas N horas"
    )
    parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        default=0,
        help="Mostrar últimos N comandos"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output en formato JSON"
    )

    args = parser.parse_args()

    # Verificar que existe la DB
    if not Path(args.db).exists():
        print(f"❌ Base de datos no encontrada: {args.db}")
        print("   El monitor de latencia se crea cuando se ejecuta el pipeline.")
        sys.exit(1)

    monitor = LatencyMonitor(db_path=args.db)

    # Modo histórico
    if args.history:
        stats = monitor.get_historical_stats(hours=args.history)

        if args.json:
            import json
            print(json.dumps(stats, indent=2))
        else:
            print(f"\n📊 Estadísticas últimas {args.history} horas")
            print("=" * 50)
            print(f"  Comandos totales: {stats['total_commands']}")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
            print(f"  Latencia promedio: {stats['avg_ms']:.1f}ms")
            print(f"  Min/Max: {stats['min_ms']:.1f}ms / {stats['max_ms']:.1f}ms")

            if stats['by_hour']:
                print("\n  Por hora:")
                for h in stats['by_hour']:
                    bar = "█" * int(h['avg_ms'] / 20)
                    print(f"    {h['hour']}:00  {h['count']:3d} cmds  {h['avg_ms']:6.1f}ms {bar}")

            if stats['by_intent']:
                print("\n  Por intent:")
                for i in stats['by_intent']:
                    print(f"    {i['intent']:15s}  {i['count']:4d} cmds  {i['avg_ms']:6.1f}ms avg")

        return

    # Mostrar últimos N comandos
    if args.recent > 0:
        records = monitor.get_recent_records(limit=args.recent)

        if args.json:
            import json
            print(json.dumps(records, indent=2))
        else:
            print(f"\n📝 Últimos {len(records)} comandos")
            print("=" * 70)
            for r in records:
                ts = time.strftime("%H:%M:%S", time.localtime(r['timestamp']))
                status = "✅" if r['met_target'] else "❌"
                user = r['user'] or "?"
                intent = r['intent'] or "?"
                print(f"  {ts} {status} {r['total_ms']:6.1f}ms | {user:10s} | {intent}")

                # Mostrar componentes
                comps = " + ".join([f"{k}={v:.0f}" for k, v in r['components'].items()])
                print(f"           └─ {comps}")

        return

    # Modo watch
    if args.watch:
        try:
            while True:
                clear_screen()
                print(monitor.format_dashboard())

                # Mostrar últimos 5 comandos
                recent = monitor.get_recent_records(5)
                if recent:
                    print("  Recent commands:")
                    for r in recent[-5:]:
                        ts = time.strftime("%H:%M:%S", time.localtime(r['timestamp']))
                        status = "✅" if r['met_target'] else "❌"
                        print(f"    {ts} {status} {r['total_ms']:6.1f}ms")

                print(f"\n  Actualización cada {args.interval}s | Ctrl+C para salir")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n👋 Bye!")

    else:
        # Single shot
        if args.json:
            import json
            print(json.dumps(monitor.get_stats(), indent=2))
        else:
            print(monitor.format_dashboard())


if __name__ == "__main__":
    main()
