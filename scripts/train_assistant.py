#!/usr/bin/env python3
"""
KZA Voice - Assistant Training CLI
Herramientas para personalizar y entrenar la IA.
"""

import argparse
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_personality(args):
    """Gestionar personalidad de la IA"""
    from src.training.personality import PersonalityManager, TONE_TEMPLATES

    manager = PersonalityManager()

    if args.action == "show":
        config = manager.get_config()
        print("\n🤖 Personalidad Actual")
        print("=" * 40)
        print(f"  Nombre: {config['name']}")
        print(f"  Descripción: {config['description']}")
        print(f"  Tono: {config['tone']}")
        print(f"  Idioma: {config['language']}")
        print(f"  Emojis: {'Sí' if config['use_emojis'] else 'No'}")
        print(f"  Verbose: {'Sí' if config['verbose'] else 'No'}")
        print(f"  Reglas personalizadas: {config['custom_rules_count']}")
        print(f"\n  Tonos disponibles: {', '.join(config['available_tones'])}")

    elif args.action == "set-name":
        if not args.value:
            print("Uso: --action set-name --value NuevoNombre")
            return 1
        result = manager.apply_setting("name", args.value)
        print(f"✅ {result}")

    elif args.action == "set-tone":
        if not args.value:
            print(f"Tonos disponibles: {', '.join(TONE_TEMPLATES.keys())}")
            print("Uso: --action set-tone --value friendly")
            return 1
        result = manager.apply_setting("tone", args.value)
        print(f"✅ {result}")

    elif args.action == "add-rule":
        if not args.value:
            print("Uso: --action add-rule --value 'Nunca menciones a Alexa'")
            return 1
        result = manager.apply_setting("rule", args.value)
        print(f"✅ {result}")

    elif args.action == "test-prompt":
        prompt = manager.build_system_prompt({"time": "14:30", "user": "Juan"})
        print("\n📝 System Prompt Generado:")
        print("-" * 40)
        print(prompt)

    return 0


def cmd_commands(args):
    """Gestionar comandos personalizados"""
    from src.training.command_learner import CommandLearner

    # Para esta CLI, usamos mocks simplificados
    class MockHA:
        def get_domotics_entities(self):
            return []
        def call_service(self, *args):
            return True

    class MockChroma:
        def add_custom_command(self, *args, **kwargs):
            pass

    learner = CommandLearner(
        chroma_sync=MockChroma(),
        ha_client=MockHA(),
        llm_reasoner=None
    )

    if args.action == "list":
        commands = learner.get_custom_commands()
        if not commands:
            print("\n📋 No hay comandos personalizados")
            print("   Usa el asistente de voz para enseñar comandos")
            print("   Di: 'Enseñar comando' o 'Nuevo comando'")
        else:
            print(f"\n📋 Comandos Personalizados ({len(commands)})")
            print("=" * 50)
            for cmd in commands:
                print(f"\n  🔹 '{cmd['trigger']}'")
                print(f"     Descripción: {cmd['description']}")
                print(f"     Acciones: {cmd['actions_count']}")
                print(f"     Usos: {cmd['times_used']}")

    elif args.action == "delete":
        if not args.id:
            print("Uso: --action delete --id custom_123456")
            return 1
        if learner.delete_command(args.id):
            print(f"✅ Comando eliminado: {args.id}")
        else:
            print(f"❌ Comando no encontrado: {args.id}")

    return 0


async def _training_data(args):
    from contextlib import AsyncExitStack
    import asyncio
    import yaml
    from src.memory.privacy_services import build_privacy_services
    from src.wakeword.clip_store import build_wake_clip_store
    from src.training.conversation_collector import build_conversation_collector

    def load():
        with open(getattr(args, 'config', 'config/settings.yaml')) as stream:
            return yaml.safe_load(stream)
    config = await asyncio.to_thread(load)
    if type(config) is not dict:
        raise ValueError('invalid_training_config')
    async with AsyncExitStack() as shutdown:
        privacy = await build_privacy_services(config, shutdown=shutdown)
        store = await build_wake_clip_store(config, privacy=privacy, shutdown=shutdown)
        collector = await build_conversation_collector(config, store=store)
        if args.action == "stats":
            stats = await collector.get_stats()
            print(f"Conversaciones vigentes: {stats['total_conversations']}")
            print(f"Turnos vigentes: {stats['total_turns']}")
            print(f"Marcados: {stats['marked_for_training']}")
            print("Los archivos legacy permanecen en cuarentena.")
        else:
            output_dir = getattr(args, 'output_dir', None)
            if not output_dir:
                output_dir = str(Path(args.output).parent) if args.output else config.get('training', {}).get('exports_dir', './data/conversation_exports')
            exported = await collector.export_for_training(output_dir, args.format or 'alpaca',
                only_marked=not args.include_all, limit=getattr(args,'limit',32),
                after_sequence=getattr(args,'after_sequence',0))
            if exported is None:
                print("No hay turnos vigentes para exportar.")
            else:
                print(f"Archivo temporal: {exported.output_path}")
                print(f"Entradas: {exported.exported}; vencimiento: {exported.expires_at}")
                if exported.has_more:
                    print(f"Hay más turnos; siguiente cursor: {exported.next_cursor}")
    return 0


def cmd_training_data(args):
    """Inspect or export only the authoritative, expiring conversation policy."""
    import asyncio
    from src.training.conversation_collector import ConversationCollector
    if args.action == 'feedback-help':
        print(ConversationCollector.get_training_commands())
        return 0
    try:
        return asyncio.run(_training_data(args))
    except (RuntimeError, OSError, ValueError):
        print('No se pudo acceder al catálogo privado o preparar la exportación.')
        return 1


def cmd_lora(args):
    """Refuse untracked datasets and adapters before constructing a trainer."""
    print("LoRA no está disponible: falta una sesión que respete la vigencia y "
          "el borrado de las fuentes (private_training_session_unavailable).", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="KZA Assistant Training Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Ver personalidad actual
  python train_assistant.py personality --action show

  # Cambiar nombre de la IA
  python train_assistant.py personality --action set-name --value "Casa"

  # Cambiar tono a formal
  python train_assistant.py personality --action set-tone --value formal

  # Ver comandos personalizados
  python train_assistant.py commands --action list

  # Ver estadísticas de entrenamiento
  python train_assistant.py training --action stats

  # Exportar datos para entrenamiento
  python train_assistant.py training --action export --format alpaca

  # Consultar disponibilidad del entrenamiento nocturno
  python -m src.training.nightly_trainer --status
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Categoría")

    # Personality
    pers_parser = subparsers.add_parser("personality", help="Gestionar personalidad")
    pers_parser.add_argument("--action", required=True,
                            choices=["show", "set-name", "set-tone", "add-rule", "test-prompt"])
    pers_parser.add_argument("--value", help="Valor para set-*")

    # Commands
    cmd_parser = subparsers.add_parser("commands", help="Comandos personalizados")
    cmd_parser.add_argument("--action", required=True, choices=["list", "delete"])
    cmd_parser.add_argument("--id", help="ID del comando para delete")

    # Training data
    train_parser = subparsers.add_parser("training", help="Datos de entrenamiento")
    train_parser.add_argument("--action", required=True,
                             choices=["stats", "export", "feedback-help"])
    train_parser.add_argument("--config", default="config/settings.yaml", help="Configuración del catálogo privado")
    destination = train_parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", help="Directorio para una exportación temporal")
    destination.add_argument("--output", help="Usar el directorio padre de esta ruta; se devuelve un nombre temporal")
    train_parser.add_argument("--limit", type=int, default=32, help="Máximo de turnos por lote")
    train_parser.add_argument("--after-sequence", type=int, default=0, help="Continuar desde el cursor del lote anterior")
    train_parser.add_argument("--format", choices=["alpaca", "sharegpt", "conversations"],
                             default="alpaca")
    train_parser.add_argument("--include-all", action="store_true",
                             help="Incluir respuestas no marcadas")

    # LoRA
    lora_parser = subparsers.add_parser("lora", help="LoRA (no disponible hasta validar sesiones privadas)")
    lora_parser.add_argument("--action", required=True, choices=["train", "list"])
    lora_parser.add_argument("--data", help="Ruta a datos de entrenamiento")
    lora_parser.add_argument("--name", help="Nombre del adapter")
    lora_parser.add_argument("--base-model", help="Modelo base")
    lora_parser.add_argument("--revision",
                            help="Revision/commit del modelo base en el Hub "
                                 "(requerida si --base-model no es el default)")
    lora_parser.add_argument("--output-dir", help="Directorio de salida")
    lora_parser.add_argument("--epochs", type=int, default=3)
    lora_parser.add_argument("--batch-size", type=int, default=4)
    lora_parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    if args.command == "personality":
        return cmd_personality(args)
    elif args.command == "commands":
        return cmd_commands(args)
    elif args.command == "training":
        return cmd_training_data(args)
    elif args.command == "lora":
        return cmd_lora(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
