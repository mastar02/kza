"""
Music Dispatcher - Enrutador de comandos de música

Procesa peticiones de música y las enruta al handler apropiado:
- Búsqueda directa: "Pon música de Bad Bunny"
- Mood/contexto: "Pon algo para entrenar"
- Control: "Pausa", "Siguiente canción"
- Recomendaciones: "Pon algo parecido"
- Multi-room: "Pon música en la cocina", "Mueve la música al dormitorio"
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from .client import SpotifyClient, SpotifyTrack
from .mood_mapper import MoodMapper, MoodProfile, AudioFeatures

if TYPE_CHECKING:
    from .zone_controller import SpotifyZoneController

logger = logging.getLogger(__name__)


class MusicIntent(Enum):
    """Tipos de intent para comandos de música"""
    # Reproducción
    PLAY_ARTIST = "play_artist"          # "Pon música de Taylor Swift"
    PLAY_TRACK = "play_track"            # "Pon la canción Blinding Lights"
    PLAY_PLAYLIST = "play_playlist"      # "Pon mi playlist de ejercicio"
    PLAY_MOOD = "play_mood"              # "Pon algo tranquilo"
    PLAY_CONTEXT = "play_context"        # "Pon música para cocinar" (requiere LLM)
    PLAY_SIMILAR = "play_similar"        # "Pon algo parecido"
    PLAY_RECOMMENDATIONS = "play_recommendations"  # "Recomiéndame música"

    # Control
    PAUSE = "pause"
    RESUME = "resume"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME = "volume"
    SHUFFLE = "shuffle"
    REPEAT = "repeat"

    # Multi-room / Zonas
    TRANSFER = "transfer"                # "Mueve la música al dormitorio"
    PLAY_EVERYWHERE = "play_everywhere"  # "Pon música en toda la casa"
    STOP_ZONE = "stop_zone"              # "Para la música en la cocina"
    VOLUME_ZONE = "volume_zone"          # "Sube el volumen en el living"

    # Información
    WHATS_PLAYING = "whats_playing"      # "¿Qué está sonando?"
    WHERE_PLAYING = "where_playing"      # "¿Dónde está sonando música?"
    UNKNOWN = "unknown"


@dataclass
class MusicCommand:
    """Comando de música parseado"""
    intent: MusicIntent
    raw_text: str
    artist: Optional[str] = None
    track: Optional[str] = None
    playlist: Optional[str] = None
    mood_profile: Optional[MoodProfile] = None
    volume_level: Optional[int] = None
    shuffle_state: Optional[bool] = None
    # Zona/ubicación objetivo
    target_zone: Optional[str] = None
    target_resolved: Optional[Dict] = None  # Resultado de resolve_target()


@dataclass
class MusicResult:
    """Resultado de ejecutar un comando de música"""
    success: bool
    response: str
    intent: MusicIntent
    tracks_played: List[SpotifyTrack] = field(default_factory=list)
    latency_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)


class MusicDispatcher:
    """
    Dispatcher de comandos de música con soporte multi-room.

    Uso:
        dispatcher = MusicDispatcher(spotify_client, mood_mapper, llm, zone_controller)

        # Procesar comando simple
        result = await dispatcher.process("Pon música para una cena romántica")

        # Procesar con zona
        result = await dispatcher.process("Pon jazz en la cocina")
        # result.response = "Reproduciendo jazz en la cocina"

        # Transferir
        result = await dispatcher.process("Mueve la música al dormitorio")
    """

    def __init__(
        self,
        spotify_client: SpotifyClient,
        mood_mapper: MoodMapper,
        llm=None,
        zone_controller: "SpotifyZoneController" = None,
        default_limit: int = 20,
    ):
        self.spotify = spotify_client
        self.mood_mapper = mood_mapper
        self.llm = llm
        self.zone_controller = zone_controller
        self.default_limit = default_limit

        # Keywords para detectar intents
        self._control_keywords = {
            MusicIntent.PAUSE: ["pausa", "para la música", "stop", "detén", "deten"],
            MusicIntent.RESUME: ["continúa", "continua", "resume", "sigue", "play"],
            MusicIntent.NEXT: ["siguiente", "next", "skip", "salta", "otra"],
            MusicIntent.PREVIOUS: ["anterior", "previous", "atrás", "atras"],
            MusicIntent.SHUFFLE: ["aleatorio", "shuffle", "mezcla", "random"],
            MusicIntent.WHATS_PLAYING: ["qué suena", "que suena", "qué canción", "que cancion", "qué está sonando"],
            MusicIntent.WHERE_PLAYING: ["dónde suena", "donde suena", "dónde está sonando", "en qué cuarto"],
        }

        # Keywords para transferencia/zonas
        self._zone_keywords = {
            MusicIntent.TRANSFER: ["mueve", "pasa", "transfiere", "cambia"],
            MusicIntent.PLAY_EVERYWHERE: ["toda la casa", "todas partes", "everywhere", "todos los cuartos"],
        }

    def detect_intent(self, text: str) -> MusicCommand:
        """
        Detectar intent y extraer información del texto.

        Args:
            text: Texto del usuario

        Returns:
            MusicCommand con intent y datos extraídos
        """
        text_lower = text.lower().strip()
        target_zone = None
        target_resolved = None
        working_text = text

        # Extraer zona del comando si hay zone_controller
        if self.zone_controller:
            target_resolved, working_text = self.zone_controller.parse_and_resolve_zone(text)
            if target_resolved:
                target_zone = working_text  # El texto sin la zona
                # Verificar si es "toda la casa"
                if target_resolved.get("type") == "group":
                    group = target_resolved.get("group")
                    if group and group.id == "everywhere":
                        return MusicCommand(
                            intent=MusicIntent.PLAY_EVERYWHERE,
                            raw_text=text,
                            target_zone="toda la casa",
                            target_resolved=target_resolved
                        )

        text_lower_working = working_text.lower().strip()

        # Detectar transferencia: "mueve la música al dormitorio"
        for kw in self._zone_keywords.get(MusicIntent.TRANSFER, []):
            if kw in text_lower:
                # Extraer destino
                zone_dest = self._extract_transfer_destination(text_lower)
                if zone_dest and self.zone_controller:
                    resolved = self.zone_controller.speakers.resolve_target(zone_dest)
                    return MusicCommand(
                        intent=MusicIntent.TRANSFER,
                        raw_text=text,
                        target_zone=zone_dest,
                        target_resolved=resolved
                    )

        # Detectar "toda la casa"
        for kw in self._zone_keywords.get(MusicIntent.PLAY_EVERYWHERE, []):
            if kw in text_lower:
                return MusicCommand(
                    intent=MusicIntent.PLAY_EVERYWHERE,
                    raw_text=text,
                    target_zone="toda la casa",
                    target_resolved=target_resolved
                )

        # Primero verificar controles de playback
        for intent, keywords in self._control_keywords.items():
            if any(kw in text_lower_working for kw in keywords):
                return MusicCommand(
                    intent=intent,
                    raw_text=text,
                    target_zone=target_zone,
                    target_resolved=target_resolved
                )

        # Detectar volumen (puede tener zona)
        if "volumen" in text_lower_working or "volume" in text_lower_working:
            volume = self._extract_volume(text_lower_working)
            intent = MusicIntent.VOLUME_ZONE if target_resolved else MusicIntent.VOLUME
            return MusicCommand(
                intent=intent,
                raw_text=text,
                volume_level=volume,
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Detectar artista/track mencionado (usar working_text para mejor extracción)
        extracted = self.mood_mapper.extract_artist_or_track(working_text)

        if extracted["track"]:
            return MusicCommand(
                intent=MusicIntent.PLAY_TRACK,
                raw_text=text,
                track=extracted["track"],
                artist=extracted["artist"],
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        if extracted["artist"]:
            return MusicCommand(
                intent=MusicIntent.PLAY_ARTIST,
                raw_text=text,
                artist=extracted["artist"],
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Detectar playlist
        if "playlist" in text_lower_working or "lista" in text_lower_working:
            playlist_name = self._extract_playlist_name(text_lower_working)
            return MusicCommand(
                intent=MusicIntent.PLAY_PLAYLIST,
                raw_text=text,
                playlist=playlist_name,
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Detectar "algo parecido"
        if any(kw in text_lower_working for kw in ["parecido", "similar", "como esto", "más de esto"]):
            return MusicCommand(
                intent=MusicIntent.PLAY_SIMILAR,
                raw_text=text,
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Detectar mood por keywords
        mood_profile = self.mood_mapper.get_mood_profile(working_text)
        if mood_profile:
            return MusicCommand(
                intent=MusicIntent.PLAY_MOOD,
                raw_text=text,
                mood_profile=mood_profile,
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Si menciona "para" + actividad, es contexto complejo
        if " para " in text_lower_working or "mientras" in text_lower_working:
            return MusicCommand(
                intent=MusicIntent.PLAY_CONTEXT,
                raw_text=text,
                target_zone=target_zone,
                target_resolved=target_resolved
            )

        # Fallback: recomendaciones generales
        return MusicCommand(
            intent=MusicIntent.PLAY_RECOMMENDATIONS,
            raw_text=text,
            target_zone=target_zone,
            target_resolved=target_resolved
        )

    async def process(self, text: str, user_preferences: Optional[Dict] = None) -> MusicResult:
        """
        Procesar comando de música.

        Args:
            text: Texto del usuario
            user_preferences: Preferencias del usuario (géneros favoritos, etc.)

        Returns:
            MusicResult con el resultado de la operación
        """
        start_time = time.time()

        command = self.detect_intent(text)
        logger.info(f"Music intent: {command.intent.value}")

        try:
            result = await self._execute_command(command, user_preferences)
            result.latency_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            logger.error(f"Music command failed: {e}")
            return MusicResult(
                success=False,
                response=f"Error al procesar el comando: {str(e)}",
                intent=command.intent,
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _execute_command(
        self,
        command: MusicCommand,
        user_preferences: Optional[Dict] = None
    ) -> MusicResult:
        """Ejecutar comando de música"""

        # Control de playback
        if command.intent == MusicIntent.PAUSE:
            success = await self.spotify.pause()
            return MusicResult(
                success=success,
                response="Música pausada" if success else "No pude pausar",
                intent=command.intent
            )

        if command.intent == MusicIntent.RESUME:
            success = await self.spotify.play()
            return MusicResult(
                success=success,
                response="Continuando" if success else "No pude reanudar",
                intent=command.intent
            )

        if command.intent == MusicIntent.NEXT:
            success = await self.spotify.next_track()
            return MusicResult(
                success=success,
                response="Siguiente canción" if success else "No pude cambiar",
                intent=command.intent
            )

        if command.intent == MusicIntent.PREVIOUS:
            success = await self.spotify.previous_track()
            return MusicResult(
                success=success,
                response="Canción anterior" if success else "No pude cambiar",
                intent=command.intent
            )

        if command.intent == MusicIntent.VOLUME:
            if command.volume_level is not None:
                success = await self.spotify.set_volume(command.volume_level)
                return MusicResult(
                    success=success,
                    response=f"Volumen al {command.volume_level}%" if success else "No pude cambiar el volumen",
                    intent=command.intent
                )

        if command.intent == MusicIntent.SHUFFLE:
            success = await self.spotify.set_shuffle(True)
            return MusicResult(
                success=success,
                response="Modo aleatorio activado" if success else "No pude activar shuffle",
                intent=command.intent
            )

        if command.intent == MusicIntent.WHATS_PLAYING:
            info = await self.spotify.get_current_track_info()
            if info:
                return MusicResult(success=True, response=info, intent=command.intent)
            return MusicResult(
                success=False,
                response="No hay nada reproduciéndose",
                intent=command.intent
            )

        # Zone-specific intents
        if command.intent == MusicIntent.TRANSFER:
            return await self._handle_transfer(command)

        if command.intent == MusicIntent.PLAY_EVERYWHERE:
            return await self._handle_play_everywhere(command)

        if command.intent == MusicIntent.VOLUME_ZONE:
            return await self._handle_volume_zone(command)

        if command.intent == MusicIntent.WHERE_PLAYING:
            return await self._handle_where_playing()

        # Reproducción (con soporte de zona si está configurada)
        if command.intent == MusicIntent.PLAY_ARTIST:
            return await self._play_artist_with_zone(command)

        if command.intent == MusicIntent.PLAY_TRACK:
            return await self._play_track_with_zone(command)

        if command.intent == MusicIntent.PLAY_PLAYLIST:
            return await self._play_playlist_with_zone(command)

        if command.intent == MusicIntent.PLAY_MOOD:
            return await self._play_mood_with_zone(command, user_preferences)

        if command.intent == MusicIntent.PLAY_CONTEXT:
            return await self._play_context(command.raw_text, user_preferences)

        if command.intent == MusicIntent.PLAY_SIMILAR:
            return await self._play_similar()

        if command.intent == MusicIntent.PLAY_RECOMMENDATIONS:
            return await self._play_recommendations(user_preferences)

        return MusicResult(
            success=False,
            response="No entendí qué música quieres",
            intent=MusicIntent.UNKNOWN
        )

    async def _play_artist(self, artist_name: str) -> MusicResult:
        """Reproducir música de un artista (sin zona)"""
        artists = await self.spotify.search_artists(artist_name, limit=1)
        if not artists:
            return MusicResult(
                success=False,
                response=f"No encontré al artista {artist_name}",
                intent=MusicIntent.PLAY_ARTIST
            )

        artist = artists[0]
        success = await self.spotify.play(context_uri=artist["uri"])

        return MusicResult(
            success=success,
            response=f"Reproduciendo {artist['name']}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_ARTIST,
            details={"artist": artist["name"], "uri": artist["uri"]}
        )

    async def _play_artist_with_zone(self, command: MusicCommand) -> MusicResult:
        """Reproducir artista con soporte de zona"""
        artists = await self.spotify.search_artists(command.artist, limit=1)
        if not artists:
            return MusicResult(
                success=False,
                response=f"No encontré al artista {command.artist}",
                intent=MusicIntent.PLAY_ARTIST
            )

        artist = artists[0]

        # Si hay zona configurada, usar zone_controller
        if command.target_resolved and self.zone_controller:
            result = await self._play_with_zone(
                command,
                context_uri=artist["uri"]
            )
            if result.success:
                zone = command.target_zone or "la zona"
                result.response = f"Reproduciendo {artist['name']} en {zone}"
            return result

        # Sin zona, reproducción normal
        success = await self.spotify.play(context_uri=artist["uri"])
        return MusicResult(
            success=success,
            response=f"Reproduciendo {artist['name']}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_ARTIST,
            details={"artist": artist["name"]}
        )

    async def _play_track(self, track_name: str, artist: Optional[str] = None) -> MusicResult:
        """Reproducir una canción específica (sin zona)"""
        query = f"{track_name} {artist}" if artist else track_name
        tracks = await self.spotify.search_tracks(query, limit=1)
        if not tracks:
            return MusicResult(
                success=False,
                response=f"No encontré la canción {track_name}",
                intent=MusicIntent.PLAY_TRACK
            )

        track = tracks[0]
        success = await self.spotify.play(uris=[track.uri])

        return MusicResult(
            success=success,
            response=f"Reproduciendo {track.name} de {track.artist_string}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_TRACK,
            tracks_played=[track]
        )

    async def _play_track_with_zone(self, command: MusicCommand) -> MusicResult:
        """Reproducir track con soporte de zona"""
        query = f"{command.track} {command.artist}" if command.artist else command.track
        tracks = await self.spotify.search_tracks(query, limit=1)
        if not tracks:
            return MusicResult(
                success=False,
                response=f"No encontré la canción {command.track}",
                intent=MusicIntent.PLAY_TRACK
            )

        track = tracks[0]

        if command.target_resolved and self.zone_controller:
            result = await self._play_with_zone(command, uris=[track.uri])
            if result.success:
                zone = command.target_zone or "la zona"
                result.response = f"Reproduciendo {track.name} en {zone}"
                result.tracks_played = [track]
            return result

        success = await self.spotify.play(uris=[track.uri])
        return MusicResult(
            success=success,
            response=f"Reproduciendo {track.name} de {track.artist_string}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_TRACK,
            tracks_played=[track]
        )

    async def _play_playlist(self, playlist_name: str) -> MusicResult:
        """Reproducir una playlist (sin zona)"""
        playlists = await self.spotify.search_playlists(playlist_name, limit=1)
        if not playlists:
            return MusicResult(
                success=False,
                response=f"No encontré la playlist {playlist_name}",
                intent=MusicIntent.PLAY_PLAYLIST
            )

        playlist = playlists[0]
        success = await self.spotify.play(context_uri=playlist["uri"])
        await self.spotify.set_shuffle(True)

        return MusicResult(
            success=success,
            response=f"Reproduciendo {playlist['name']}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_PLAYLIST,
            details={"playlist": playlist["name"]}
        )

    async def _play_playlist_with_zone(self, command: MusicCommand) -> MusicResult:
        """Reproducir playlist con soporte de zona"""
        playlists = await self.spotify.search_playlists(command.playlist, limit=1)
        if not playlists:
            return MusicResult(
                success=False,
                response=f"No encontré la playlist {command.playlist}",
                intent=MusicIntent.PLAY_PLAYLIST
            )

        playlist = playlists[0]

        if command.target_resolved and self.zone_controller:
            result = await self._play_with_zone(
                command,
                context_uri=playlist["uri"],
                shuffle=True
            )
            if result.success:
                zone = command.target_zone or "la zona"
                result.response = f"Reproduciendo {playlist['name']} en {zone}"
            return result

        success = await self.spotify.play(context_uri=playlist["uri"])
        await self.spotify.set_shuffle(True)
        return MusicResult(
            success=success,
            response=f"Reproduciendo {playlist['name']}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_PLAYLIST,
            details={"playlist": playlist["name"]}
        )

    async def _play_mood(
        self,
        mood_profile: MoodProfile,
        user_preferences: Optional[Dict] = None
    ) -> MusicResult:
        """Reproducir música basada en mood"""
        # Obtener parámetros de audio features
        rec_params = mood_profile.features.to_recommendation_params()

        # Agregar géneros del mood
        genres = mood_profile.genres[:5]

        # Mezclar con preferencias del usuario si existen
        if user_preferences and "favorite_genres" in user_preferences:
            user_genres = user_preferences["favorite_genres"]
            # Combinar: mitad del mood, mitad del usuario
            genres = genres[:3] + user_genres[:2]

        tracks = await self.spotify.get_recommendations(
            seed_genres=genres,
            limit=self.default_limit,
            **rec_params
        )

        if not tracks:
            return MusicResult(
                success=False,
                response="No encontré música para ese mood",
                intent=MusicIntent.PLAY_MOOD
            )

        # Reproducir tracks
        uris = [t.uri for t in tracks]
        success = await self.spotify.play(uris=uris)
        await self.spotify.set_shuffle(True)

        return MusicResult(
            success=success,
            response=f"Reproduciendo música {mood_profile.name.lower()}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_MOOD,
            tracks_played=tracks[:5],
            details={"mood": mood_profile.name, "genres": genres}
        )

    async def _play_mood_with_zone(
        self,
        command: MusicCommand,
        user_preferences: Optional[Dict] = None
    ) -> MusicResult:
        """Reproducir mood con soporte de zona"""
        if not command.mood_profile:
            return MusicResult(
                success=False,
                response="No entendí qué tipo de música quieres",
                intent=MusicIntent.PLAY_MOOD
            )

        mood_profile = command.mood_profile
        rec_params = mood_profile.features.to_recommendation_params()
        genres = mood_profile.genres[:5]

        if user_preferences and "favorite_genres" in user_preferences:
            user_genres = user_preferences["favorite_genres"]
            genres = genres[:3] + user_genres[:2]

        tracks = await self.spotify.get_recommendations(
            seed_genres=genres,
            limit=self.default_limit,
            **rec_params
        )

        if not tracks:
            return MusicResult(
                success=False,
                response="No encontré música para ese mood",
                intent=MusicIntent.PLAY_MOOD
            )

        uris = [t.uri for t in tracks]

        if command.target_resolved and self.zone_controller:
            result = await self._play_with_zone(command, uris=uris, shuffle=True)
            if result.success:
                zone = command.target_zone or "la zona"
                result.response = f"Reproduciendo música {mood_profile.name.lower()} en {zone}"
                result.tracks_played = tracks[:5]
            return result

        success = await self.spotify.play(uris=uris)
        await self.spotify.set_shuffle(True)
        return MusicResult(
            success=success,
            response=f"Reproduciendo música {mood_profile.name.lower()}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_MOOD,
            tracks_played=tracks[:5],
            details={"mood": mood_profile.name}
        )

    async def _play_context(
        self,
        text: str,
        user_preferences: Optional[Dict] = None
    ) -> MusicResult:
        """Reproducir música interpretando contexto con LLM"""
        # Primero intentar con mood mapper keywords
        mood_profile = self.mood_mapper.get_mood_profile(text)

        # Si no hay match, usar LLM
        if not mood_profile and self.llm:
            mood_profile = await self.mood_mapper.interpret_with_llm(text)

        if not mood_profile:
            # Fallback a recomendaciones generales
            return await self._play_recommendations(user_preferences)

        result = await self._play_mood(mood_profile, user_preferences)
        result.intent = MusicIntent.PLAY_CONTEXT
        result.details["interpreted_as"] = mood_profile.name

        return result

    async def _play_similar(self) -> MusicResult:
        """Reproducir música similar a la actual"""
        state = await self.spotify.get_playback_state()

        if not state or not state.track:
            return MusicResult(
                success=False,
                response="No hay nada sonando para encontrar similar",
                intent=MusicIntent.PLAY_SIMILAR
            )

        # Usar track actual como seed
        tracks = await self.spotify.get_recommendations(
            seed_tracks=[state.track.id],
            limit=self.default_limit
        )

        if not tracks:
            return MusicResult(
                success=False,
                response="No encontré música similar",
                intent=MusicIntent.PLAY_SIMILAR
            )

        uris = [t.uri for t in tracks]
        success = await self.spotify.play(uris=uris)

        return MusicResult(
            success=success,
            response=f"Reproduciendo música similar a {state.track.name}" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_SIMILAR,
            tracks_played=tracks[:5]
        )

    async def _play_recommendations(self, user_preferences: Optional[Dict] = None) -> MusicResult:
        """Reproducir recomendaciones personalizadas"""
        # Obtener top artists/tracks del usuario para seeds
        top_tracks = await self.spotify.get_user_top_tracks(time_range="short_term", limit=5)

        seed_tracks = [t.id for t in top_tracks[:3]] if top_tracks else None
        seed_genres = None

        if not seed_tracks:
            # Fallback a géneros populares
            seed_genres = ["pop", "rock", "latin"]
            if user_preferences and "favorite_genres" in user_preferences:
                seed_genres = user_preferences["favorite_genres"][:5]

        tracks = await self.spotify.get_recommendations(
            seed_tracks=seed_tracks,
            seed_genres=seed_genres,
            limit=self.default_limit
        )

        if not tracks:
            return MusicResult(
                success=False,
                response="No pude obtener recomendaciones",
                intent=MusicIntent.PLAY_RECOMMENDATIONS
            )

        uris = [t.uri for t in tracks]
        success = await self.spotify.play(uris=uris)
        await self.spotify.set_shuffle(True)

        return MusicResult(
            success=success,
            response="Reproduciendo recomendaciones para ti" if success else "No pude reproducir",
            intent=MusicIntent.PLAY_RECOMMENDATIONS,
            tracks_played=tracks[:5]
        )

    def _extract_volume(self, text: str) -> Optional[int]:
        """Extraer nivel de volumen del texto"""
        import re

        # Buscar número
        match = re.search(r'(\d+)', text)
        if match:
            vol = int(match.group(1))
            return max(0, min(100, vol))

        # Palabras clave
        if any(kw in text for kw in ["máximo", "maximo", "max", "full"]):
            return 100
        if any(kw in text for kw in ["mínimo", "minimo", "min", "bajo"]):
            return 20
        if any(kw in text for kw in ["medio", "mitad"]):
            return 50
        if "sube" in text or "más" in text or "mas" in text:
            return None  # Incremento relativo (TODO: implementar)
        if "baja" in text or "menos" in text:
            return None

        return 50  # Default

    def _extract_playlist_name(self, text: str) -> str:
        """Extraer nombre de playlist del texto"""
        import re

        patterns = [
            r"playlist\s+(?:de\s+)?['\"]?(.+?)['\"]?$",
            r"lista\s+(?:de\s+)?['\"]?(.+?)['\"]?$",
            r"mi\s+playlist\s+(?:de\s+)?(.+?)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return text  # Fallback al texto completo

    def _extract_transfer_destination(self, text: str) -> Optional[str]:
        """Extraer destino de transferencia del texto"""
        import re

        # Patrones: "mueve la música al dormitorio", "pasa al living"
        patterns = [
            r"(?:mueve|pasa|transfiere|cambia).*(?:al?|hacia)\s+(?:la\s+|el\s+)?(.+?)(?:\s+por favor)?$",
            r"(?:al?|hacia)\s+(?:la\s+|el\s+)?(.+?)(?:\s+por favor)?$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    # =========================================================================
    # Métodos de Reproducción con Zona
    # =========================================================================

    async def _play_with_zone(
        self,
        command: MusicCommand,
        context_uri: Optional[str] = None,
        uris: Optional[List[str]] = None,
        shuffle: bool = False
    ) -> MusicResult:
        """
        Reproducir en una zona específica si está configurada.

        Si hay zona, usa zone_controller. Si no, usa reproducción normal.
        """
        if not command.target_resolved or not self.zone_controller:
            # Reproducción normal sin zona
            success = await self.spotify.play(context_uri=context_uri, uris=uris)
            if shuffle:
                await self.spotify.set_shuffle(True)
            return MusicResult(
                success=success,
                response="Reproduciendo" if success else "No pude reproducir",
                intent=command.intent
            )

        # Reproducción con zona
        zone_name = command.target_zone or "zona"

        result = await self.zone_controller.play_in_zone(
            zone_query=zone_name,
            context_uri=context_uri,
            uris=uris,
            shuffle=shuffle
        )

        if result.get("success"):
            zone_display = result.get("zone", zone_name)
            return MusicResult(
                success=True,
                response=f"Reproduciendo en {zone_display}",
                intent=command.intent,
                details={"zone": zone_display}
            )
        else:
            return MusicResult(
                success=False,
                response=result.get("error", "No pude reproducir en esa zona"),
                intent=command.intent
            )

    async def _handle_transfer(self, command: MusicCommand) -> MusicResult:
        """Manejar transferencia a otra zona"""
        if not self.zone_controller:
            return MusicResult(
                success=False,
                response="No tengo configuración de zonas",
                intent=MusicIntent.TRANSFER
            )

        if not command.target_zone:
            return MusicResult(
                success=False,
                response="No entendí a dónde mover la música",
                intent=MusicIntent.TRANSFER
            )

        result = await self.zone_controller.transfer_to_zone(command.target_zone)

        if result.get("success"):
            return MusicResult(
                success=True,
                response=f"Música movida a {result.get('zone', command.target_zone)}",
                intent=MusicIntent.TRANSFER
            )
        else:
            return MusicResult(
                success=False,
                response=result.get("error", "No pude mover la música"),
                intent=MusicIntent.TRANSFER
            )

    async def _handle_play_everywhere(self, command: MusicCommand) -> MusicResult:
        """Manejar reproducción en toda la casa"""
        if not self.zone_controller:
            return MusicResult(
                success=False,
                response="No tengo configuración de zonas",
                intent=MusicIntent.PLAY_EVERYWHERE
            )

        result = await self.zone_controller.play_everywhere(shuffle=True)

        if result.get("success"):
            count = result.get("speakers_count", 0)
            return MusicResult(
                success=True,
                response=f"Reproduciendo en toda la casa ({count} altavoces)",
                intent=MusicIntent.PLAY_EVERYWHERE
            )
        else:
            return MusicResult(
                success=False,
                response=result.get("error", "No pude reproducir en toda la casa"),
                intent=MusicIntent.PLAY_EVERYWHERE
            )

    async def _handle_volume_zone(self, command: MusicCommand) -> MusicResult:
        """Manejar volumen en zona específica"""
        if not self.zone_controller or not command.target_zone:
            # Volumen normal
            if command.volume_level is not None:
                success = await self.spotify.set_volume(command.volume_level)
                return MusicResult(
                    success=success,
                    response=f"Volumen al {command.volume_level}%",
                    intent=MusicIntent.VOLUME
                )
            return MusicResult(success=False, response="No entendí el volumen", intent=MusicIntent.VOLUME)

        result = await self.zone_controller.set_zone_volume(
            command.target_zone,
            command.volume_level or 50
        )

        if result.get("success"):
            return MusicResult(
                success=True,
                response=f"Volumen en {command.target_zone} al {command.volume_level}%",
                intent=MusicIntent.VOLUME_ZONE
            )
        else:
            return MusicResult(
                success=False,
                response="No pude ajustar el volumen",
                intent=MusicIntent.VOLUME_ZONE
            )

    async def _handle_where_playing(self) -> MusicResult:
        """Informar dónde está sonando música"""
        if not self.zone_controller:
            state = await self.spotify.get_playback_state()
            if state and state.device:
                return MusicResult(
                    success=True,
                    response=f"Sonando en {state.device.name}",
                    intent=MusicIntent.WHERE_PLAYING
                )
            return MusicResult(
                success=False,
                response="No hay música reproduciéndose",
                intent=MusicIntent.WHERE_PLAYING
            )

        zone = await self.zone_controller.get_current_zone()
        if zone:
            return MusicResult(
                success=True,
                response=f"La música está sonando en {zone}",
                intent=MusicIntent.WHERE_PLAYING
            )
        return MusicResult(
            success=False,
            response="No hay música reproduciéndose",
            intent=MusicIntent.WHERE_PLAYING
        )
