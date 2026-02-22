"""
Spotify Zone Controller - Puente entre Spotify y las zonas de la casa

Funcionalidades principales:
- Reproducir en zonas específicas: "pon música en la cocina"
- Reproducir en grupos: "pon música en toda la casa"
- Transferir reproducción: "mueve la música al dormitorio"
- Seguir al usuario: la música sigue donde está la persona
- Sincronización multi-room (usando Spotify Connect)
"""

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from enum import StrEnum

from .client import SpotifyClient, SpotifyDevice, PlaybackState
from .speaker_groups import SpeakerGroupManager, Speaker, SpeakerGroup

logger = logging.getLogger(__name__)


class PlaybackMode(StrEnum):
    """Modos de reproducción multi-room"""
    SINGLE = "single"           # Un solo speaker
    GROUP = "group"             # Grupo de speakers (sync)
    FOLLOW = "follow"           # Seguir al usuario
    EVERYWHERE = "everywhere"   # Toda la casa


@dataclass
class ZonePlaybackState:
    """Estado de reproducción por zona"""
    speaker_id: str
    speaker_name: str
    is_playing: bool
    track_name: str | None = None
    artist: str | None = None
    volume: int = 50
    spotify_device_id: str | None = None


class SpotifyZoneController:
    """
    Controlador de zonas para Spotify.

    Integra:
    - SpotifyClient para control de Spotify
    - SpeakerGroupManager para gestión de grupos
    - ZoneManager para zonas de audio físicas (MA1260)

    Uso:
        controller = SpotifyZoneController(
            spotify_client=spotify,
            speaker_manager=speaker_manager,
            zone_manager=zone_manager  # Opcional
        )

        # Reproducir en zona específica
        await controller.play_in_zone("pon jazz en la cocina")

        # Reproducir en grupo
        await controller.play_in_group("downstairs", playlist_uri)

        # Transferir a otra zona
        await controller.transfer_to_zone("dormitorio")

        # Reproducir en toda la casa
        await controller.play_everywhere(track_uris)
    """

    def __init__(
        self,
        spotify_client: SpotifyClient,
        speaker_manager: SpeakerGroupManager,
        zone_manager=None,  # ZoneManager opcional
        auto_sync_devices: bool = True,
        follow_mode_enabled: bool = False
    ):
        self.spotify = spotify_client
        self.speakers = speaker_manager
        self.zones = zone_manager

        self.auto_sync_devices = auto_sync_devices
        self.follow_mode_enabled = follow_mode_enabled

        # Estado actual
        self._current_mode = PlaybackMode.SINGLE
        self._active_targets: list[str] = []  # IDs de speakers activos
        self._last_user_location: str | None = None

        # Cache de dispositivos Spotify
        self._device_cache: dict[str, SpotifyDevice] = {}
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 30.0  # 30 segundos

        # Callbacks
        self._on_zone_change: Callable | None = None
        self._on_playback_start: Callable | None = None

    # =========================================================================
    # Sincronización de Dispositivos
    # =========================================================================

    async def sync_devices(self) -> dict[str, SpotifyDevice]:
        """
        Sincronizar dispositivos Spotify con speakers configurados.

        Intenta hacer match automático por nombre.
        """
        import time

        # Usar cache si es reciente
        if time.time() - self._cache_timestamp < self._cache_ttl:
            return self._device_cache

        devices = await self.spotify.get_devices()
        self._device_cache = {d.id: d for d in devices}
        self._cache_timestamp = time.time()

        logger.info(f"Dispositivos Spotify encontrados: {len(devices)}")

        if self.auto_sync_devices:
            await self._auto_match_devices(devices)

        return self._device_cache

    async def _auto_match_devices(self, devices: list[SpotifyDevice]):
        """Intentar vincular automáticamente dispositivos con speakers"""
        for device in devices:
            device_name_lower = device.name.lower()

            for speaker in self.speakers.speakers.values():
                # Si ya está vinculado, skip
                if speaker.spotify_device_id:
                    continue

                # Buscar coincidencia por nombre
                speaker_name_lower = speaker.name.lower()
                if (speaker_name_lower in device_name_lower or
                    device_name_lower in speaker_name_lower):
                    speaker.spotify_device_id = device.id
                    logger.info(f"Auto-vinculado: {speaker.name} -> {device.name} ({device.id})")
                    break

                # Buscar por alias
                for alias in speaker.aliases:
                    if alias.lower() in device_name_lower:
                        speaker.spotify_device_id = device.id
                        logger.info(f"Auto-vinculado por alias: {speaker.name} -> {device.name}")
                        break

    async def get_available_devices(self) -> list[dict[str, Any]]:
        """Obtener lista de dispositivos disponibles con estado"""
        await self.sync_devices()

        result = []
        for speaker in self.speakers.speakers.values():
            device = self._device_cache.get(speaker.spotify_device_id) if speaker.spotify_device_id else None

            result.append({
                "speaker_id": speaker.id,
                "speaker_name": speaker.name,
                "room": speaker.room,
                "spotify_device_id": speaker.spotify_device_id,
                "spotify_device_name": device.name if device else None,
                "is_active": device.is_active if device else False,
                "is_available": device is not None,
                "volume": device.volume_percent if device else speaker.volume_offset
            })

        return result

    # =========================================================================
    # Reproducción en Zonas
    # =========================================================================

    async def play_in_zone(
        self,
        zone_query: str,
        context_uri: str | None = None,
        uris: list[str] | None = None,
        shuffle: bool = False
    ) -> dict[str, Any]:
        """
        Reproducir en una zona específica.

        Args:
            zone_query: Nombre de zona/room/speaker ("cocina", "planta baja")
            context_uri: URI de playlist/album/artist
            uris: Lista de track URIs
            shuffle: Activar shuffle

        Returns:
            Dict con resultado de la operación
        """
        # Resolver destino
        target = self.speakers.resolve_target(zone_query)

        if not target:
            return {
                "success": False,
                "error": f"No encontré la zona '{zone_query}'",
                "available_zones": self.speakers.list_available_targets()
            }

        # Sincronizar dispositivos
        await self.sync_devices()

        # Obtener device IDs
        device_ids = target.get("spotify_device_ids", [])
        if not device_ids:
            return {
                "success": False,
                "error": f"No hay dispositivos Spotify en {zone_query}",
                "speakers": [s.name for s in target.get("speakers", [])]
            }

        # Reproducir en el primer dispositivo (o master si es grupo)
        primary_device = device_ids[0]

        if target["type"] == "group":
            group = target.get("group")
            if group and group.master_speaker_id:
                master_speaker = self.speakers.get_speaker(group.master_speaker_id)
                if master_speaker and master_speaker.spotify_device_id:
                    primary_device = master_speaker.spotify_device_id

        # Transferir y reproducir
        success = await self.spotify.transfer_playback(primary_device, play=False)
        if not success:
            logger.warning(f"No pude transferir a {primary_device}")

        # Iniciar reproducción
        success = await self.spotify.play(
            device_id=primary_device,
            context_uri=context_uri,
            uris=uris
        )

        if shuffle:
            await self.spotify.set_shuffle(True, device_id=primary_device)

        # Actualizar estado
        self._current_mode = PlaybackMode.GROUP if len(device_ids) > 1 else PlaybackMode.SINGLE
        self._active_targets = [s.id for s in target.get("speakers", [])]

        # Callback
        if self._on_playback_start:
            self._on_playback_start(target)

        zone_name = (
            target.get("speaker", target.get("group", {})).name
            if target.get("speaker") or target.get("group")
            else zone_query
        )

        return {
            "success": success,
            "zone": zone_name,
            "type": target["type"],
            "device_id": primary_device,
            "speakers_count": len(target.get("speakers", []))
        }

    async def play_everywhere(
        self,
        context_uri: str | None = None,
        uris: list[str] | None = None,
        shuffle: bool = True
    ) -> dict[str, Any]:
        """
        Reproducir en toda la casa.

        Nota: Spotify Connect no soporta verdadero multi-room sync.
        Esta función reproduce en el speaker principal y ajusta volúmenes
        de los demás para crear efecto de audio distribuido.
        """
        return await self.play_in_zone(
            "toda la casa",
            context_uri=context_uri,
            uris=uris,
            shuffle=shuffle
        )

    async def play_in_group(
        self,
        group_id: str,
        context_uri: str | None = None,
        uris: list[str] | None = None
    ) -> dict[str, Any]:
        """Reproducir en un grupo específico por ID"""
        group = self.speakers.get_group(group_id)
        if not group:
            return {"success": False, "error": f"Grupo no encontrado: {group_id}"}

        return await self.play_in_zone(
            group.name,
            context_uri=context_uri,
            uris=uris
        )

    # =========================================================================
    # Transferencia entre Zonas
    # =========================================================================

    async def transfer_to_zone(
        self,
        zone_query: str,
        keep_playing: bool = True
    ) -> dict[str, Any]:
        """
        Transferir reproducción actual a otra zona.

        Args:
            zone_query: Destino ("dormitorio", "cocina")
            keep_playing: Continuar reproducción después de transferir

        Returns:
            Dict con resultado
        """
        target = self.speakers.resolve_target(zone_query)

        if not target:
            return {
                "success": False,
                "error": f"No encontré la zona '{zone_query}'"
            }

        device_ids = target.get("spotify_device_ids", [])
        if not device_ids:
            return {
                "success": False,
                "error": f"No hay dispositivos Spotify en {zone_query}"
            }

        # Transferir
        success = await self.spotify.transfer_playback(device_ids[0], play=keep_playing)

        if success:
            self._active_targets = [s.id for s in target.get("speakers", [])]

            if self._on_zone_change:
                self._on_zone_change(target)

        zone_name = target.get("speaker", target.get("group", {}))
        if hasattr(zone_name, "name"):
            zone_name = zone_name.name
        else:
            zone_name = zone_query

        return {
            "success": success,
            "zone": zone_name,
            "action": "transferred"
        }

    async def stop_in_zone(self, zone_query: str) -> dict[str, Any]:
        """Detener reproducción en una zona específica"""
        target = self.speakers.resolve_target(zone_query)

        if not target:
            return {"success": False, "error": f"Zona no encontrada: {zone_query}"}

        device_ids = target.get("spotify_device_ids", [])

        # Pausar en cada dispositivo del target
        success = True
        for device_id in device_ids:
            result = await self.spotify.pause(device_id=device_id)
            success = success and result

        return {"success": success, "zone": zone_query}

    # =========================================================================
    # Modo Seguimiento (Follow Mode)
    # =========================================================================

    def enable_follow_mode(self):
        """Activar modo seguimiento"""
        self.follow_mode_enabled = True
        self._current_mode = PlaybackMode.FOLLOW
        logger.info("Modo seguimiento activado")

    def disable_follow_mode(self):
        """Desactivar modo seguimiento"""
        self.follow_mode_enabled = False
        self._current_mode = PlaybackMode.SINGLE
        logger.info("Modo seguimiento desactivado")

    async def update_user_location(self, zone_id: str):
        """
        Actualizar ubicación del usuario.
        Si está en modo seguimiento, mover la música.

        Args:
            zone_id: ID de la zona donde se detectó al usuario
        """
        if zone_id == self._last_user_location:
            return

        self._last_user_location = zone_id

        if self.follow_mode_enabled:
            # Verificar si hay algo reproduciéndose
            state = await self.spotify.get_playback_state()
            if state and state.is_playing:
                # Transferir a la nueva ubicación
                speaker = self.speakers.get_speaker(zone_id)
                if speaker and speaker.spotify_device_id:
                    await self.transfer_to_zone(speaker.name)
                    logger.info(f"Música transferida a {speaker.name} (follow mode)")

    # =========================================================================
    # Control de Volumen por Zona
    # =========================================================================

    async def set_zone_volume(self, zone_query: str, volume: int) -> dict[str, Any]:
        """
        Ajustar volumen de una zona.

        Args:
            zone_query: Zona objetivo
            volume: Volumen 0-100

        Returns:
            Dict con resultado
        """
        target = self.speakers.resolve_target(zone_query)

        if not target:
            return {"success": False, "error": f"Zona no encontrada: {zone_query}"}

        device_ids = target.get("spotify_device_ids", [])

        success = True
        for device_id in device_ids:
            # Aplicar offset del speaker si existe
            speaker = self.speakers.get_speaker_by_spotify_id(device_id)
            adjusted_volume = volume
            if speaker:
                adjusted_volume = max(0, min(100, volume + speaker.volume_offset))

            result = await self.spotify.set_volume(adjusted_volume, device_id=device_id)
            success = success and result

        return {
            "success": success,
            "zone": zone_query,
            "volume": volume
        }

    async def set_everywhere_volume(self, volume: int) -> dict[str, Any]:
        """Ajustar volumen en toda la casa"""
        return await self.set_zone_volume("toda la casa", volume)

    # =========================================================================
    # Estado
    # =========================================================================

    async def get_zone_playback_states(self) -> list[ZonePlaybackState]:
        """Obtener estado de reproducción de todas las zonas"""
        await self.sync_devices()

        states = []
        for speaker in self.speakers.speakers.values():
            device = self._device_cache.get(speaker.spotify_device_id) if speaker.spotify_device_id else None

            state = ZonePlaybackState(
                speaker_id=speaker.id,
                speaker_name=speaker.name,
                is_playing=device.is_active if device else False,
                volume=device.volume_percent if device else 0,
                spotify_device_id=speaker.spotify_device_id
            )

            # Si es el dispositivo activo, obtener info del track
            if device and device.is_active:
                playback = await self.spotify.get_playback_state()
                if playback and playback.track:
                    state.track_name = playback.track.name
                    state.artist = playback.track.artist_string

            states.append(state)

        return states

    async def get_current_zone(self) -> str | None:
        """Obtener la zona donde se está reproduciendo actualmente"""
        state = await self.spotify.get_playback_state()

        if not state or not state.device:
            return None

        # Buscar speaker correspondiente
        speaker = self.speakers.get_speaker_by_spotify_id(state.device.id)
        if speaker:
            return speaker.name

        # Fallback al nombre del dispositivo Spotify
        return state.device.name

    def get_status(self) -> dict[str, Any]:
        """Obtener estado del controlador"""
        return {
            "mode": self._current_mode.value,
            "follow_enabled": self.follow_mode_enabled,
            "active_targets": self._active_targets,
            "last_user_location": self._last_user_location,
            "cached_devices": len(self._device_cache),
            "speakers_count": len(self.speakers.speakers),
            "groups_count": len(self.speakers.groups)
        }

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_zone_change(self, callback: Callable):
        """Registrar callback cuando cambia la zona de reproducción"""
        self._on_zone_change = callback

    def on_playback_start(self, callback: Callable):
        """Registrar callback cuando inicia reproducción en una zona"""
        self._on_playback_start = callback

    # =========================================================================
    # Utilidades para Comandos de Voz
    # =========================================================================

    def parse_and_resolve_zone(self, text: str) -> tuple[dict | None, str]:
        """
        Parsear zona de un comando de voz y devolver el resto.

        Args:
            text: "pon jazz en la cocina"

        Returns:
            (target, cleaned_text):
            - target: Resultado de resolve_target o None
            - cleaned_text: "pon jazz"
        """
        return self.speakers.parse_zone_from_command(text)

    def get_zone_suggestions(self, partial: str) -> list[str]:
        """
        Sugerir zonas basado en texto parcial (para autocompletado).

        Args:
            partial: Texto parcial ("coc" -> ["cocina"])
        """
        partial_lower = partial.lower()
        suggestions = []

        for target in self.speakers.list_available_targets():
            if partial_lower in target.lower():
                suggestions.append(target)

        return sorted(suggestions)[:5]
