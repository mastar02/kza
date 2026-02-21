"""
Routine Executor
Motor de ejecución de acciones para rutinas
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import time as time_module

from src.home_assistant.circuit_breaker import HACircuitBreaker, get_ha_circuit_breaker

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Tipos de acciones soportadas"""
    # Home Assistant
    HA_SERVICE = "ha_service"        # Llamar servicio de HA
    HA_SCENE = "ha_scene"            # Activar escena
    HA_SCRIPT = "ha_script"          # Ejecutar script

    # Multimedia
    SPOTIFY_PLAY = "spotify_play"
    SPOTIFY_PAUSE = "spotify_pause"
    TTS_SPEAK = "tts_speak"          # Hablar texto

    # Control
    DELAY = "delay"                  # Esperar N segundos
    CONDITION = "condition"          # Verificar condición
    PARALLEL = "parallel"            # Ejecutar en paralelo
    SEQUENCE = "sequence"            # Ejecutar en secuencia

    # Notificaciones
    NOTIFY = "notify"
    LOG = "log"


@dataclass
class ActionResult:
    """Resultado de ejecución de acción"""
    success: bool
    action_type: str
    entity_id: Optional[str] = None
    message: str = ""
    elapsed_ms: float = 0
    data: dict = None


class RoutineExecutor:
    """
    Ejecutor de acciones de rutinas.
    Soporta acciones de Home Assistant, Spotify, TTS y más.
    """

    # PRIORIDAD: Velocidad de respuesta > recursos del servidor
    # Límites altos = máximo paralelismo = mínima latencia
    MAX_PARALLEL_ACTIONS = 100  # Sin límite efectivo
    MAX_PARALLEL_SPOTIFY = 20

    def __init__(
        self,
        ha_client=None,
        spotify_client=None,
        tts_engine=None,
        zone_controller=None,
        max_parallel: int = 100  # Default alto para velocidad
    ):
        self.ha = ha_client
        self.spotify = spotify_client
        self.tts = tts_engine
        self.zones = zone_controller

        # Semáforos con límites altos (prioriza velocidad)
        self._ha_semaphore = asyncio.Semaphore(max_parallel)
        self._spotify_semaphore = asyncio.Semaphore(self.MAX_PARALLEL_SPOTIFY)

        # FIX: Circuit breaker para proteger contra fallos de HA
        self._circuit_breaker = get_ha_circuit_breaker()

        # Handlers por tipo de acción
        self._handlers = {
            ActionType.HA_SERVICE.value: self._execute_ha_service,
            ActionType.HA_SCENE.value: self._execute_ha_scene,
            ActionType.HA_SCRIPT.value: self._execute_ha_script,
            ActionType.SPOTIFY_PLAY.value: self._execute_spotify_play,
            ActionType.SPOTIFY_PAUSE.value: self._execute_spotify_pause,
            ActionType.TTS_SPEAK.value: self._execute_tts_speak,
            ActionType.DELAY.value: self._execute_delay,
            ActionType.NOTIFY.value: self._execute_notify,
            ActionType.PARALLEL.value: self._execute_parallel,
            ActionType.SEQUENCE.value: self._execute_sequence,
        }

    async def execute_actions(
        self,
        actions: list[dict],
        context: dict = None
    ) -> list[ActionResult]:
        """
        Ejecutar lista de acciones en secuencia.

        Args:
            actions: Lista de acciones a ejecutar
            context: Contexto de ejecución (user_id, trigger, etc.)

        Returns:
            Lista de resultados
        """
        results = []
        context = context or {}

        for action in actions:
            try:
                result = await self._execute_single_action(action, context)
                results.append(result)

                # Si una acción falla y es crítica, detener
                if not result.success and action.get("stop_on_error", False):
                    logger.warning(f"Acción crítica falló, deteniendo: {action}")
                    break

            except Exception as e:
                logger.error(f"Error ejecutando acción: {e}")
                results.append(ActionResult(
                    success=False,
                    action_type=action.get("type", "unknown"),
                    message=str(e)
                ))

        return results

    async def _execute_single_action(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Ejecutar una acción individual"""
        action_type = action.get("type", "")

        # Sustituir variables en la acción
        action = self._substitute_variables(action, context)

        handler = self._handlers.get(action_type)

        if handler:
            return await handler(action, context)
        else:
            # Por defecto, intentar como servicio de HA
            return await self._execute_ha_service(action, context)

    def _substitute_variables(self, action: dict, context: dict) -> dict:
        """Sustituir variables en la acción"""
        import copy
        action = copy.deepcopy(action)

        def substitute(obj):
            if isinstance(obj, str):
                # Sustituir {{variable}} por valor del contexto
                for key, value in context.items():
                    obj = obj.replace(f"{{{{{key}}}}}", str(value))
                return obj
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            return obj

        return substitute(action)

    # ==================== Handlers de Home Assistant ====================

    async def _execute_ha_service(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Ejecutar servicio de Home Assistant con circuit breaker"""
        t_start = time_module.perf_counter()

        entity_id = action.get("entity_id", "")
        domain = action.get("domain", entity_id.split(".")[0] if entity_id else "")
        service = action.get("service", "turn_on")
        data = action.get("data", {})

        if not self.ha:
            return ActionResult(
                success=False,
                action_type="ha_service",
                entity_id=entity_id,
                message="Home Assistant no configurado"
            )

        # FIX: Usar circuit breaker para proteger contra fallos de HA
        async def ha_call():
            if hasattr(self.ha, 'call_service_ws'):
                return await self.ha.call_service_ws(domain, service, entity_id, data)
            else:
                return self.ha.call_service(domain, service, entity_id, data)

        success, result = await self._circuit_breaker.call(
            ha_call,
            fallback=False
        )

        elapsed = (time_module.perf_counter() - t_start) * 1000

        if not success:
            # Circuit breaker bloqueó o HA falló
            cb_status = self._circuit_breaker.get_status()
            if cb_status["state"] == "open":
                message = f"HA no disponible (circuit open, recuperación en {cb_status['config']['recovery_timeout']}s)"
            else:
                message = "Error llamando a HA"

            return ActionResult(
                success=False,
                action_type="ha_service",
                entity_id=entity_id,
                message=message,
                elapsed_ms=elapsed
            )

        return ActionResult(
            success=result if isinstance(result, bool) else True,
            action_type="ha_service",
            entity_id=entity_id,
            message=f"{domain}.{service}",
            elapsed_ms=elapsed
        )

    async def _execute_ha_scene(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Activar escena de Home Assistant"""
        scene_id = action.get("scene_id", action.get("entity_id", ""))

        if not scene_id.startswith("scene."):
            scene_id = f"scene.{scene_id}"

        return await self._execute_ha_service({
            "entity_id": scene_id,
            "domain": "scene",
            "service": "turn_on"
        }, context)

    async def _execute_ha_script(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Ejecutar script de Home Assistant"""
        script_id = action.get("script_id", action.get("entity_id", ""))

        if not script_id.startswith("script."):
            script_id = f"script.{script_id}"

        return await self._execute_ha_service({
            "entity_id": script_id,
            "domain": "script",
            "service": "turn_on",
            "data": action.get("variables", {})
        }, context)

    # ==================== Handlers de Spotify ====================

    async def _execute_spotify_play(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Reproducir música en Spotify"""
        t_start = time_module.perf_counter()

        if not self.spotify:
            return ActionResult(
                success=False,
                action_type="spotify_play",
                message="Spotify no configurado"
            )

        try:
            # Determinar qué reproducir
            playlist = action.get("playlist")
            artist = action.get("artist")
            album = action.get("album")
            track = action.get("track")
            mood = action.get("mood")
            device = action.get("device")
            zone = action.get("zone")

            # Si hay zona, usar zone_controller
            if zone and self.zones:
                if mood:
                    await self.zones.play_mood_in_zone(zone, mood)
                elif playlist:
                    await self.zones.play_in_zone(zone, playlist_uri=playlist)
                success = True
            else:
                # Reproducir directamente
                if playlist:
                    success = await self.spotify.play_playlist(playlist, device_id=device)
                elif artist:
                    success = await self.spotify.play_artist(artist, device_id=device)
                elif mood:
                    success = await self.spotify.play_mood(mood, device_id=device)
                else:
                    success = await self.spotify.resume(device_id=device)

            elapsed = (time_module.perf_counter() - t_start) * 1000

            return ActionResult(
                success=success,
                action_type="spotify_play",
                message=f"Playing: {playlist or artist or mood or 'resumed'}",
                elapsed_ms=elapsed
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action_type="spotify_play",
                message=str(e)
            )

    async def _execute_spotify_pause(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Pausar Spotify"""
        if not self.spotify:
            return ActionResult(
                success=False,
                action_type="spotify_pause",
                message="Spotify no configurado"
            )

        try:
            zone = action.get("zone")

            if zone and self.zones:
                await self.zones.stop_zone(zone)
            else:
                await self.spotify.pause()

            return ActionResult(
                success=True,
                action_type="spotify_pause",
                message="Paused"
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type="spotify_pause",
                message=str(e)
            )

    # ==================== Handler de TTS ====================

    async def _execute_tts_speak(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Hablar texto con TTS"""
        t_start = time_module.perf_counter()

        text = action.get("text", action.get("message", ""))

        if not text:
            return ActionResult(
                success=False,
                action_type="tts_speak",
                message="No hay texto para hablar"
            )

        if not self.tts:
            return ActionResult(
                success=False,
                action_type="tts_speak",
                message="TTS no configurado"
            )

        try:
            # Generar y reproducir audio
            audio_data = self.tts.speak(text)

            # Si hay zona específica, reproducir ahí
            zone = action.get("zone")
            if zone and self.zones:
                await self.zones.play_audio_in_zone(zone, audio_data)

            elapsed = (time_module.perf_counter() - t_start) * 1000

            return ActionResult(
                success=True,
                action_type="tts_speak",
                message=text[:50],
                elapsed_ms=elapsed
            )

        except Exception as e:
            return ActionResult(
                success=False,
                action_type="tts_speak",
                message=str(e)
            )

    # ==================== Handlers de Control ====================

    async def _execute_delay(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Esperar N segundos"""
        seconds = action.get("seconds", action.get("delay", 1))

        await asyncio.sleep(seconds)

        return ActionResult(
            success=True,
            action_type="delay",
            message=f"Waited {seconds}s",
            elapsed_ms=seconds * 1000
        )

    async def _execute_parallel(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Ejecutar acciones en paralelo con límite de concurrencia"""
        sub_actions = action.get("actions", [])

        if not sub_actions:
            return ActionResult(
                success=True,
                action_type="parallel",
                message="No actions"
            )

        t_start = time_module.perf_counter()

        # FIX: Usar semáforo para limitar concurrencia y no saturar HA
        async def bounded_action(a):
            async with self._ha_semaphore:
                return await self._execute_single_action(a, context)

        # Ejecutar con límite de concurrencia
        tasks = [bounded_action(a) for a in sub_actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verificar si todas fueron exitosas
        all_success = all(
            isinstance(r, ActionResult) and r.success
            for r in results
        )

        elapsed = (time_module.perf_counter() - t_start) * 1000

        return ActionResult(
            success=all_success,
            action_type="parallel",
            message=f"Executed {len(sub_actions)} actions (max {self.MAX_PARALLEL_ACTIONS} concurrent)",
            elapsed_ms=elapsed,
            data={"results": [r for r in results if isinstance(r, ActionResult)]}
        )

    async def _execute_sequence(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Ejecutar acciones en secuencia"""
        sub_actions = action.get("actions", [])

        if not sub_actions:
            return ActionResult(
                success=True,
                action_type="sequence",
                message="No actions"
            )

        t_start = time_module.perf_counter()
        results = await self.execute_actions(sub_actions, context)

        all_success = all(r.success for r in results)
        elapsed = (time_module.perf_counter() - t_start) * 1000

        return ActionResult(
            success=all_success,
            action_type="sequence",
            message=f"Executed {len(results)} actions",
            elapsed_ms=elapsed,
            data={"results": results}
        )

    async def _execute_notify(
        self,
        action: dict,
        context: dict
    ) -> ActionResult:
        """Enviar notificación"""
        message = action.get("message", "")
        title = action.get("title", "KZA")
        target = action.get("target", "notify.notify")

        if self.ha:
            # Usar servicio de notificación de HA
            return await self._execute_ha_service({
                "entity_id": target,
                "domain": "notify",
                "service": target.split(".")[-1] if "." in target else "notify",
                "data": {
                    "message": message,
                    "title": title
                }
            }, context)
        else:
            logger.info(f"Notificación: {title} - {message}")
            return ActionResult(
                success=True,
                action_type="notify",
                message=message[:50]
            )

    # ==================== Utilidades ====================

    def get_supported_actions(self) -> list[str]:
        """Obtener lista de tipos de acciones soportadas"""
        return [a.value for a in ActionType]

    async def test_action(self, action: dict) -> ActionResult:
        """Probar una acción sin ejecutarla realmente (dry run)"""
        action_type = action.get("type", "unknown")

        return ActionResult(
            success=True,
            action_type=action_type,
            message=f"Dry run: {action_type}",
            data=action
        )
