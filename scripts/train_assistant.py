#!/usr/bin/env python3
"""
KZA Voice - Assistant Training CLI
Herramientas para personalizar y entrenar la IA.
"""

import argparse
import json
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


def cmd_training_data(args):
    """Gestionar datos de entrenamiento"""
    from src.training.conversation_collector import ConversationCollector

    collector = ConversationCollector()

    if args.action == "stats":
        stats = collector.get_stats()
        print("\n📊 Estadísticas de Conversaciones")
        print("=" * 40)
        print(f"  Total conversaciones: {stats['total_conversations']}")
        print(f"  Total turnos: {stats['total_turns']}")
        print(f"\n  Distribución de calidad:")
        for quality, count in stats['quality_distribution'].items():
            print(f"    - {quality}: {count}")
        print(f"\n  Listos para entrenamiento: {stats['marked_for_training']}")
        print(f"  Directorio: {stats['data_dir']}")

    elif args.action == "export":
        output = args.output or "./data/training_data.jsonl"
        format_type = args.format or "alpaca"

        stats = collector.export_for_training(
            output_path=output,
            format=format_type,
            only_marked=not args.include_all
        )

        print(f"\n✅ Datos exportados:")
        print(f"   Archivo: {stats['output_path']}")
        print(f"   Formato: {stats['format']}")
        print(f"   Entradas: {stats['exported']}")
        print(f"   - Buenos: {stats['good_responses']}")
        print(f"   - Corregidos: {stats['corrected_responses']}")

    elif args.action == "feedback-help":
        print(collector.get_training_commands())

    return 0


def cmd_lora(args):
    """Entrenar modelo con LoRA"""
    from src.training.conversation_collector import LoRATrainer

    trainer = LoRATrainer(
        base_model=args.base_model or "meta-llama/Llama-3.1-8B-Instruct",
        output_dir=args.output_dir or "./models/lora_adapters"
    )

    if args.action == "train":
        if not args.data:
            print("Uso: --action train --data ./data/training_data.jsonl")
            return 1

        print(f"\n🧠 Iniciando entrenamiento LoRA")
        print(f"   Modelo base: {trainer.base_model}")
        print(f"   Datos: {args.data}")
        print(f"   Épocas: {args.epochs}")
        print()

        result = trainer.train(
            training_data_path=args.data,
            adapter_name=args.name or "custom_adapter",
            epochs=args.epochs or 3,
            batch_size=args.batch_size or 4,
            learning_rate=args.lr or 2e-4
        )

        if result["success"]:
            print(f"\n✅ Entrenamiento completado")
            print(f"   Adapter: {result['adapter_path']}")
            print(f"   Parámetros entrenables: {result['trainable_params']:,}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown')}")
            return 1

    elif args.action == "list":
        adapters = trainer.list_adapters()
        if not adapters:
            print("\n📋 No hay adapters entrenados")
        else:
            print(f"\n📋 Adapters LoRA ({len(adapters)})")
            print("=" * 50)
            for adapter in adapters:
                print(f"\n  🔹 {adapter['name']}")
                print(f"     Path: {adapter['path']}")
                print(f"     LoRA r: {adapter['lora_r']}")
                print(f"     LoRA alpha: {adapter['lora_alpha']}")

    return 0


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

  # Entrenar con LoRA
  python train_assistant.py lora --action train --data ./data/training_data.jsonl
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
    train_parser.add_argument("--output", help="Ruta de salida para export")
    train_parser.add_argument("--format", choices=["alpaca", "sharegpt", "conversations"],
                             default="alpaca")
    train_parser.add_argument("--include-all", action="store_true",
                             help="Incluir respuestas no marcadas")

    # LoRA
    lora_parser = subparsers.add_parser("lora", help="Entrenamiento LoRA")
    lora_parser.add_argument("--action", required=True, choices=["train", "list"])
    lora_parser.add_argument("--data", help="Ruta a datos de entrenamiento")
    lora_parser.add_argument("--name", help="Nombre del adapter")
    lora_parser.add_argument("--base-model", help="Modelo base")
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
