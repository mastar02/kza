#!/usr/bin/env python3
"""
KZA Voice - Smart Automations CLI
Herramientas para analizar patrones y gestionar sugerencias.
"""

import argparse
import asyncio
from contextlib import AsyncExitStack
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_stats(args):
    """Mostrar estadisticas de eventos"""
    
    logger = args.event_logger
    stats = logger.get_stats()
    
    print("\n📊 Estadisticas de Eventos")
    print("=" * 50)
    print(f"  Total eventos: {stats['total_events']}")
    print(f"  Ultimos 7 dias: {stats['events_last_week']}")
    print(f"  Entidades unicas: {stats['unique_entities']}")
    
    if stats['first_event']:
        print(f"\n  Primer evento: {stats['first_event']}")
        print(f"  Ultimo evento: {stats['last_event']}")
    
    if stats['top_entities']:
        print("\n  Top 5 entidades:")
        for e in stats['top_entities']:
            print(f"    - {e['entity']}: {e['count']} eventos")
    
    print(f"\n  Base de datos: {stats['db_path']}")
    return 0


def cmd_patterns(args):
    """Analizar y mostrar patrones detectados"""
    from src.analytics.pattern_analyzer import PatternAnalyzer
    
    logger = args.event_logger
    analyzer = PatternAnalyzer(logger)
    
    print(f"\n🔍 Analizando patrones (ultimos {args.days} dias)...")
    patterns = analyzer.analyze_all(days=args.days)
    
    if not patterns:
        print("\n❌ No se detectaron patrones.")
        print("   Necesitas mas eventos para detectar patrones.")
        print("   Usa el sistema por unos dias y vuelve a intentar.")
        return 0
    
    print(f"\n✅ {len(patterns)} patrones detectados:")
    print("=" * 60)
    
    for i, p in enumerate(patterns[:args.limit], 1):
        confidence_bar = "█" * int(p.confidence * 10) + "░" * (10 - int(p.confidence * 10))
        print(f"\n{i}. [{confidence_bar}] {int(p.confidence * 100)}% confianza")
        print(f"   Tipo: {p.pattern_type.value}")
        print(f"   {p.description}")
        print(f"   Ocurrencias: {p.occurrences}")
        
        if p.hour is not None:
            print(f"   Hora: {p.hour:02d}:{p.minute or 0:02d} (±{p.hour_std or 0:.1f}h)")
        
        if p.trigger_entity:
            print(f"   Trigger: {p.trigger_entity}")
    
    return 0


def cmd_suggestions(args):
    """Gestionar sugerencias de automatizacion"""
    from src.analytics.pattern_analyzer import PatternAnalyzer
    from src.analytics.suggestion_engine import SuggestionEngine
    
    logger = args.event_logger
    analyzer = PatternAnalyzer(logger)
    engine = SuggestionEngine(
        event_logger=logger,
        pattern_analyzer=analyzer,
        suggestions_path=args.suggestions_path
    )
    
    if args.action == "list":
        pending = engine.get_pending_suggestions()
        stats = engine.get_stats()
        
        print("\n📋 Sugerencias de Automatizacion")
        print("=" * 50)
        print(f"  Total: {stats['total_suggestions']}")
        print(f"  Por estado: {stats['by_status']}")
        print(f"  Tasa de aceptacion: {stats['acceptance_rate']:.1%}")
        
        if pending:
            print(f"\n  Pendientes ({len(pending)}):")
            for s in pending:
                print(f"\n    ID: {s.id}")
                print(f"    {s.message}")
                print(f"    Confianza: {int(s.pattern.confidence * 100)}%")
        else:
            print("\n  No hay sugerencias pendientes.")
    
    elif args.action == "generate":
        print(f"\n🔄 Generando sugerencias (min confianza: {args.min_confidence})...")
        new_suggestions = engine.generate_suggestions(
            min_confidence=args.min_confidence,
            max_suggestions=args.max
        )
        
        if new_suggestions:
            print(f"\n✅ {len(new_suggestions)} nuevas sugerencias:")
            for s in new_suggestions:
                print(f"\n  📌 {s.message}")
                print(f"     Confianza: {int(s.pattern.confidence * 100)}%")
        else:
            print("\n❌ No se generaron nuevas sugerencias.")
            print("   Puede que no haya patrones suficientes o ya existan sugerencias.")
    
    elif args.action == "accept":
        if not args.id:
            print("Error: --id requerido para aceptar")
            return 1
        result = engine.respond_to_suggestion(args.id, accept=True)
        if result["success"]:
            print(f"\n✅ {result['message']}")
            if "yaml" in result:
                print("\n--- YAML para Home Assistant ---")
                print(result["yaml"])
                print("--------------------------------")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown')}")
    
    elif args.action == "reject":
        if not args.id:
            print("Error: --id requerido para rechazar")
            return 1
        result = engine.respond_to_suggestion(args.id, accept=False)
        print(f"\n{'✅' if result['success'] else '❌'} {result.get('message') or result.get('error')}")
    
    elif args.action == "show-yaml":
        if not args.id:
            print("Error: --id requerido")
            return 1
        
        for s in engine._suggestions.values():
            if s.id == args.id:
                print(f"\n📄 YAML para sugerencia {s.id}:")
                print("-" * 50)
                print(s.automation_yaml)
                return 0
        
        print(f"❌ Sugerencia {args.id} no encontrada")
        return 1
    
    return 0


def cmd_distribution(args):
    """Mostrar distribucion de eventos"""
    
    logger = args.event_logger
    
    if not args.entity:
        # Mostrar distribucion general
        stats = logger.get_stats()
        if not stats['top_entities']:
            print("No hay eventos registrados.")
            return 0
        
        args.entity = stats['top_entities'][0]['entity']
        print(f"Mostrando distribucion para: {args.entity}")
    
    if args.by == "hour":
        dist = logger.get_hourly_distribution(args.entity, args.action, args.days)
        
        print(f"\n📊 Distribucion por hora - {args.entity}")
        print("=" * 50)
        
        max_count = max(dist.values()) if dist.values() else 1
        for hour in range(24):
            count = dist[hour]
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  {hour:02d}:00 | {bar} {count}")
    
    elif args.by == "weekday":
        dist = logger.get_weekday_distribution(args.entity, args.action, args.days)
        days_names = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
        
        print(f"\n📊 Distribucion por dia - {args.entity}")
        print("=" * 50)
        
        max_count = max(dist.values()) if dist.values() else 1
        for day in range(7):
            count = dist[day]
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            print(f"  {days_names[day]} | {bar} {count}")
    
    return 0


def cmd_sequences(args):
    """Mostrar secuencias detectadas"""
    
    logger = args.event_logger
    sequences = logger.get_sequences(
        window_minutes=args.window,
        min_occurrences=args.min_occurrences,
        days=args.days
    )
    
    if not sequences:
        print("\n❌ No se detectaron secuencias frecuentes.")
        return 0
    
    print(f"\n🔗 Secuencias frecuentes (ventana: {args.window} min)")
    print("=" * 60)
    
    for first, second, count in sequences[:args.limit]:
        print(f"\n  {first}")
        print(f"    ↓ ({count} veces)")
        print(f"  {second}")
    
    return 0


async def run_command(args, *, config=None, clock=None):
    """Use the configured leased authority and retain all native CLI work."""
    from src.analytics.services import build_event_logger
    from src.core.settings_schema import resolve_env_vars
    from src.memory.privacy_services import build_privacy_services

    if config is None:
        import yaml
        def read_config():
            with open(args.config) as file:
                return yaml.safe_load(file)
        config = await asyncio.to_thread(read_config)
    config = resolve_env_vars({key: dict(config.get(key, {})) for key in ("memory", "analytics")})
    analytics = config["analytics"]
    analytics["enabled"] = True
    if args.db_path is not None:
        analytics["events_db"] = args.db_path
    if args.suggestions_path is not None:
        analytics["suggestions_path"] = args.suggestions_path
    paths = (config["memory"].get("privacy_db_path", "./data/memory_privacy.db"),
             analytics.get("events_db", "./data/events.db"), analytics.get("suggestions_path", "./data/suggestions.json"))
    if any(type(path) is not str or not path or "${" in path for path in paths):
        raise ValueError("analytics_cli_paths_unresolved")
    args.suggestions_path = paths[2]
    commands = {"stats": cmd_stats, "patterns": cmd_patterns, "suggestions": cmd_suggestions,
                "distribution": cmd_distribution, "sequences": cmd_sequences}
    async with AsyncExitStack() as shutdown:
        privacy = await build_privacy_services(config, shutdown=shutdown, clock=clock)
        events = await build_event_logger(config, shutdown=shutdown, privacy=privacy)
        args.event_logger = events
        try:
            return await events.acall(commands[args.command], args)
        finally:
            del args.event_logger


def main():
    parser = argparse.ArgumentParser(
        description="KZA Smart Automations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Ver estadisticas de eventos
  python smart_automations.py stats

  # Analizar patrones
  python smart_automations.py patterns --days 30

  # Generar sugerencias
  python smart_automations.py suggestions --action generate

  # Ver sugerencias pendientes
  python smart_automations.py suggestions --action list

  # Aceptar sugerencia
  python smart_automations.py suggestions --action accept --id sug_123456

  # Ver distribucion por hora
  python smart_automations.py distribution --entity light.living_room --by hour

  # Ver secuencias frecuentes
  python smart_automations.py sequences
        """
    )
    
    # Argumentos globales
    parser.add_argument("--config", default="config/settings.yaml", help="Configuración existente de KZA")
    parser.add_argument("--db-path", default=None,
                       help="Ruta a la base de datos de eventos")
    parser.add_argument("--suggestions-path", default=None,
                       help="Ruta al archivo de sugerencias")
    
    subparsers = parser.add_subparsers(dest="command", help="Comando")
    
    # Stats
    subparsers.add_parser("stats", help="Estadisticas de eventos")
    
    # Patterns
    patterns_parser = subparsers.add_parser("patterns", help="Analizar patrones")
    patterns_parser.add_argument("--days", type=int, default=30, help="Dias a analizar")
    patterns_parser.add_argument("--limit", type=int, default=10, help="Max patrones a mostrar")
    
    # Suggestions
    sug_parser = subparsers.add_parser("suggestions", help="Gestionar sugerencias")
    sug_parser.add_argument("--action", required=True,
                          choices=["list", "generate", "accept", "reject", "show-yaml"])
    sug_parser.add_argument("--id", help="ID de sugerencia")
    sug_parser.add_argument("--min-confidence", type=float, default=0.7,
                          help="Confianza minima para generar")
    sug_parser.add_argument("--max", type=int, default=5, help="Max sugerencias a generar")
    
    # Distribution
    dist_parser = subparsers.add_parser("distribution", help="Distribucion de eventos")
    dist_parser.add_argument("--entity", help="Entity ID")
    dist_parser.add_argument("--action", help="Filtrar por accion")
    dist_parser.add_argument("--by", choices=["hour", "weekday"], default="hour")
    dist_parser.add_argument("--days", type=int, default=30)
    
    # Sequences
    seq_parser = subparsers.add_parser("sequences", help="Secuencias frecuentes")
    seq_parser.add_argument("--window", type=int, default=10, help="Ventana en minutos")
    seq_parser.add_argument("--min-occurrences", type=int, default=3, help="Min ocurrencias")
    seq_parser.add_argument("--days", type=int, default=30)
    seq_parser.add_argument("--limit", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return asyncio.run(run_command(args))
    except Exception as error:
        # Paths, credentials and private SQL/model errors do not enter output.
        print(f"No pude abrir la analítica privada: {type(error).__name__}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
