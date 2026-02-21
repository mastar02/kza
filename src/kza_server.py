#!/usr/bin/env python3
"""
KZA Server - Sistema de Voz Local con IA
Entry point con integración completa de NightlyTrainer y ModelManager.

Este archivo demuestra cómo integrar:
- ModelManager: Gestión centralizada de modelos GPU
- NightlyTrainer: Entrenamiento QLoRA automático a las 3AM
- Pipeline de voz
- Sistema de alertas

Flujo nocturno:
    3:00 AM → Descargar modelos → Liberar VRAM → Entrenar QLoRA → Recargar → Listo

Uso:
    python -m src.kza_server
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KZAServer:
    """
    Servidor principal de KZA.

    Integra todos los componentes del sistema.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.model_manager = None
        self.nightly_trainer = None
        self.pipeline = None
        self.alert_scheduler = None
        self._shutdown = False

    async def start(self):
        """Iniciar el servidor"""
        logger.info("=" * 60)
        logger.info("🏠 KZA - Sistema de Voz Local con IA")
        logger.info("=" * 60)

        # 1. ModelManager
        await self._init_model_manager()

        # 2. NightlyTrainer
        self._init_nightly_trainer()

        # 3. Pipeline (si está disponible)
        # self._init_pipeline()

        # 4. Alertas
        self._init_alerts()

        logger.info("")
        logger.info("✅ Sistema listo")
        logger.info("")

    async def _init_model_manager(self):
        """Inicializar gestión de modelos GPU"""
        from src.pipeline import ModelManager, ModelManagerConfig, init_model_manager

        config = ModelManagerConfig(
            tts_gpu=0,
            embeddings_gpu=1,
            speaker_id_gpu=1,
            router_gpu=2,
            router_enabled=True,
            stt_gpu=3,
            emotion_gpu=3,
            emotion_enabled=True
        )

        self.model_manager = init_model_manager(config)

        # Cargar modelos
        logger.info("📦 Cargando modelos en GPUs...")
        results = await self.model_manager.load_all()

        # Mostrar estado
        for name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {name}")

        vram = self.model_manager.get_vram_usage()
        for gpu, mb in vram.items():
            logger.info(f"  GPU {gpu}: {mb}MB VRAM")

    def _init_nightly_trainer(self):
        """Inicializar entrenamiento nocturno"""
        from src.training import NightlyTrainer, NightlyConfig

        config = NightlyConfig(
            training_hour=3,
            training_minute=0,
            gpus=[0, 1, 2, 3],
            use_qlora=True,
            base_model="meta-llama/Llama-3.2-3B-Instruct",
            unload_daytime_models=True,
            reload_after_training=True
        )

        # Crear wrappers para los callbacks
        def unload_callback():
            """Descargar modelos de día"""
            if self.model_manager:
                # Ejecutar en el event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.model_manager.unload_all())
                else:
                    loop.run_until_complete(self.model_manager.unload_all())

        def reload_callback():
            """Recargar modelos de día"""
            if self.model_manager:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.model_manager.reload_all())
                else:
                    loop.run_until_complete(self.model_manager.reload_all())

        def alert_callback(level: str, message: str):
            """Callback para alertas del trainer"""
            logger.info(f"[NIGHTLY {level}] {message}")
            # Aquí podrías enviar notificación por TTS, push, etc.

        self.nightly_trainer = NightlyTrainer(
            config=config,
            unload_callback=unload_callback,
            reload_callback=reload_callback,
            alert_callback=alert_callback
        )

        # Iniciar scheduler
        self.nightly_trainer.start_scheduler()
        logger.info(f"🌙 Entrenamiento nocturno: {config.training_hour:02d}:{config.training_minute:02d}")

    def _init_alerts(self):
        """Inicializar sistema de alertas"""
        try:
            from src.alerts import AlertScheduler, AlertManager

            alert_manager = AlertManager()
            self.alert_scheduler = AlertScheduler(alert_manager)
            self.alert_scheduler.start()
            logger.info("🔔 Sistema de alertas activo")
        except ImportError:
            logger.warning("Sistema de alertas no disponible")

    async def run(self):
        """Loop principal del servidor"""
        logger.info("🎤 Escuchando... (Ctrl+C para salir)")

        while not self._shutdown:
            await asyncio.sleep(1)

            # Aquí iría el loop del pipeline de voz
            # if self.pipeline:
            #     await self.pipeline.process_audio()

    async def stop(self):
        """Detener el servidor"""
        logger.info("Cerrando sistema...")
        self._shutdown = True

        # Detener trainer
        if self.nightly_trainer:
            self.nightly_trainer.stop_scheduler()

        # Detener alertas
        if self.alert_scheduler:
            self.alert_scheduler.stop()

        # Descargar modelos
        if self.model_manager:
            await self.model_manager.unload_all()

        logger.info("Sistema cerrado")


async def main():
    """Entry point"""
    server = KZAServer()

    # Manejar Ctrl+C
    loop = asyncio.get_event_loop()

    def handle_signal():
        asyncio.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    try:
        await server.start()
        await server.run()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
