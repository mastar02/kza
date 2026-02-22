"""
Spotify API Client

Cliente asíncrono para la API de Spotify con soporte para:
- Búsqueda de tracks, artistas, playlists
- Control de reproducción
- Recomendaciones basadas en audio features
- Gestión de cola de reproducción
"""

import logging
from typing import Any
from dataclasses import dataclass
from enum import StrEnum

import aiohttp

from .auth import SpotifyAuth

logger = logging.getLogger(__name__)

SPOTIFY_API_BASE = "https://api.spotify.com/v1"


class RepeatMode(StrEnum):
    OFF = "off"
    TRACK = "track"
    CONTEXT = "context"


@dataclass
class SpotifyDevice:
    """Dispositivo de reproducción de Spotify"""
    id: str
    name: str
    type: str  # Computer, Smartphone, Speaker, etc.
    is_active: bool
    volume_percent: int

    @classmethod
    def from_api(cls, data: dict) -> "SpotifyDevice":
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            is_active=data["is_active"],
            volume_percent=data.get("volume_percent", 0),
        )


@dataclass
class SpotifyTrack:
    """Track de Spotify"""
    id: str
    uri: str
    name: str
    artists: list[str]
    album: str
    duration_ms: int
    popularity: int

    @classmethod
    def from_api(cls, data: dict) -> "SpotifyTrack":
        return cls(
            id=data["id"],
            uri=data["uri"],
            name=data["name"],
            artists=[a["name"] for a in data.get("artists", [])],
            album=data.get("album", {}).get("name", ""),
            duration_ms=data.get("duration_ms", 0),
            popularity=data.get("popularity", 0),
        )

    @property
    def artist_string(self) -> str:
        return ", ".join(self.artists)


@dataclass
class PlaybackState:
    """Estado actual de reproducción"""
    is_playing: bool
    track: SpotifyTrack | None
    device: SpotifyDevice | None
    progress_ms: int
    shuffle: bool
    repeat: RepeatMode

    @classmethod
    def from_api(cls, data: dict) -> "PlaybackState":
        track = None
        if data.get("item"):
            track = SpotifyTrack.from_api(data["item"])

        device = None
        if data.get("device"):
            device = SpotifyDevice.from_api(data["device"])

        return cls(
            is_playing=data.get("is_playing", False),
            track=track,
            device=device,
            progress_ms=data.get("progress_ms", 0),
            shuffle=data.get("shuffle_state", False),
            repeat=RepeatMode(data.get("repeat_state", "off")),
        )


class SpotifyClient:
    """
    Cliente para la API de Spotify.

    Uso:
        auth = SpotifyAuth(client_id="...")
        client = SpotifyClient(auth)

        # Buscar
        tracks = await client.search_tracks("Bad Bunny")

        # Reproducir
        await client.play_tracks([tracks[0].uri])

        # Recomendaciones
        recs = await client.get_recommendations(
            seed_artists=["bad_bunny_id"],
            target_energy=0.8,
            target_danceability=0.7
        )
    """

    def __init__(self, auth: SpotifyAuth):
        self.auth = auth
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtener o crear sesión HTTP"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Cerrar sesión HTTP"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict | None:
        """Realizar request a la API de Spotify"""
        token = await self.auth.get_access_token()
        if not token:
            logger.error("No access token available")
            return None

        headers = {"Authorization": f"Bearer {token}"}
        url = f"{SPOTIFY_API_BASE}{endpoint}"

        session = await self._get_session()

        try:
            async with session.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
            ) as resp:
                if resp.status == 204:  # No content (éxito sin respuesta)
                    return {}

                if resp.status == 401:  # Token expirado
                    logger.warning("Token expired, refreshing...")
                    if await self.auth.refresh_tokens():
                        return await self._request(method, endpoint, params, json_data)
                    return None

                if resp.status >= 400:
                    error = await resp.text()
                    logger.error(f"Spotify API error {resp.status}: {error}")
                    return None

                if resp.content_type == "application/json":
                    return await resp.json()
                return {}

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    # ==================== Search ====================

    async def search_tracks(
        self,
        query: str,
        limit: int = 10,
        market: str = "ES"
    ) -> list[SpotifyTrack]:
        """Buscar tracks por query"""
        result = await self._request(
            "GET",
            "/search",
            params={"q": query, "type": "track", "limit": limit, "market": market}
        )
        if not result:
            return []

        tracks = result.get("tracks", {}).get("items", [])
        return [SpotifyTrack.from_api(t) for t in tracks]

    async def search_artists(
        self,
        query: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Buscar artistas por query"""
        result = await self._request(
            "GET",
            "/search",
            params={"q": query, "type": "artist", "limit": limit}
        )
        if not result:
            return []

        return result.get("artists", {}).get("items", [])

    async def search_playlists(
        self,
        query: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Buscar playlists por query"""
        result = await self._request(
            "GET",
            "/search",
            params={"q": query, "type": "playlist", "limit": limit}
        )
        if not result:
            return []

        return result.get("playlists", {}).get("items", [])

    # ==================== Playback Control ====================

    async def get_devices(self) -> list[SpotifyDevice]:
        """Obtener dispositivos disponibles"""
        result = await self._request("GET", "/me/player/devices")
        if not result:
            return []

        devices = result.get("devices", [])
        return [SpotifyDevice.from_api(d) for d in devices]

    async def get_playback_state(self) -> PlaybackState | None:
        """Obtener estado actual de reproducción"""
        result = await self._request("GET", "/me/player")
        if not result:
            return None

        return PlaybackState.from_api(result)

    async def play(
        self,
        device_id: str | None = None,
        context_uri: str | None = None,  # album, playlist, artist URI
        uris: list[str] | None = None,   # lista de track URIs
        offset: int | None = None,       # posición en context
        position_ms: int = 0
    ) -> bool:
        """
        Iniciar o reanudar reproducción.

        Args:
            device_id: ID del dispositivo (None = activo)
            context_uri: URI de album/playlist/artist para reproducir
            uris: Lista de track URIs para reproducir
            offset: Posición inicial en el context
            position_ms: Posición inicial en ms

        Returns:
            True si exitoso
        """
        params = {}
        if device_id:
            params["device_id"] = device_id

        body = {}
        if context_uri:
            body["context_uri"] = context_uri
        if uris:
            body["uris"] = uris
        if offset is not None:
            body["offset"] = {"position": offset}
        if position_ms:
            body["position_ms"] = position_ms

        result = await self._request(
            "PUT",
            "/me/player/play",
            params=params if params else None,
            json_data=body if body else None
        )
        return result is not None

    async def pause(self, device_id: str | None = None) -> bool:
        """Pausar reproducción"""
        params = {"device_id": device_id} if device_id else None
        result = await self._request("PUT", "/me/player/pause", params=params)
        return result is not None

    async def next_track(self, device_id: str | None = None) -> bool:
        """Siguiente track"""
        params = {"device_id": device_id} if device_id else None
        result = await self._request("POST", "/me/player/next", params=params)
        return result is not None

    async def previous_track(self, device_id: str | None = None) -> bool:
        """Track anterior"""
        params = {"device_id": device_id} if device_id else None
        result = await self._request("POST", "/me/player/previous", params=params)
        return result is not None

    async def set_volume(self, volume_percent: int, device_id: str | None = None) -> bool:
        """Ajustar volumen (0-100)"""
        volume_percent = max(0, min(100, volume_percent))
        params = {"volume_percent": volume_percent}
        if device_id:
            params["device_id"] = device_id

        result = await self._request("PUT", "/me/player/volume", params=params)
        return result is not None

    async def set_shuffle(self, state: bool, device_id: str | None = None) -> bool:
        """Activar/desactivar shuffle"""
        params = {"state": str(state).lower()}
        if device_id:
            params["device_id"] = device_id

        result = await self._request("PUT", "/me/player/shuffle", params=params)
        return result is not None

    async def set_repeat(self, mode: RepeatMode, device_id: str | None = None) -> bool:
        """Configurar modo de repetición"""
        params = {"state": mode.value}
        if device_id:
            params["device_id"] = device_id

        result = await self._request("PUT", "/me/player/repeat", params=params)
        return result is not None

    async def seek(self, position_ms: int, device_id: str | None = None) -> bool:
        """Saltar a posición en track"""
        params = {"position_ms": position_ms}
        if device_id:
            params["device_id"] = device_id

        result = await self._request("PUT", "/me/player/seek", params=params)
        return result is not None

    async def add_to_queue(self, uri: str, device_id: str | None = None) -> bool:
        """Agregar track a la cola"""
        params = {"uri": uri}
        if device_id:
            params["device_id"] = device_id

        result = await self._request("POST", "/me/player/queue", params=params)
        return result is not None

    async def transfer_playback(self, device_id: str, play: bool = True) -> bool:
        """Transferir reproducción a otro dispositivo"""
        result = await self._request(
            "PUT",
            "/me/player",
            json_data={"device_ids": [device_id], "play": play}
        )
        return result is not None

    # ==================== Recommendations ====================

    async def get_recommendations(
        self,
        seed_artists: list[str] | None = None,
        seed_tracks: list[str] | None = None,
        seed_genres: list[str] | None = None,
        limit: int = 20,
        market: str = "ES",
        # Audio features targets (0.0 - 1.0)
        target_acousticness: float | None = None,
        target_danceability: float | None = None,
        target_energy: float | None = None,
        target_instrumentalness: float | None = None,
        target_liveness: float | None = None,
        target_speechiness: float | None = None,
        target_valence: float | None = None,  # Positividad/alegría
        # Tempo target (BPM)
        target_tempo: float | None = None,
        min_tempo: float | None = None,
        max_tempo: float | None = None,
        # Popularity (0-100)
        min_popularity: int | None = None,
        max_popularity: int | None = None,
    ) -> list[SpotifyTrack]:
        """
        Obtener recomendaciones basadas en seeds y audio features.

        Necesita al menos 1 seed (máximo 5 en total entre artistas, tracks y géneros).

        Args:
            seed_artists: IDs de artistas seed
            seed_tracks: IDs de tracks seed
            seed_genres: Géneros seed (ver available_genre_seeds)
            limit: Número de recomendaciones (max 100)
            target_*: Valores objetivo para audio features (0.0-1.0)
            target_tempo: Tempo objetivo en BPM

        Returns:
            Lista de tracks recomendados
        """
        params = {"limit": limit, "market": market}

        # Seeds
        if seed_artists:
            params["seed_artists"] = ",".join(seed_artists[:5])
        if seed_tracks:
            params["seed_tracks"] = ",".join(seed_tracks[:5])
        if seed_genres:
            params["seed_genres"] = ",".join(seed_genres[:5])

        # Audio features
        if target_acousticness is not None:
            params["target_acousticness"] = target_acousticness
        if target_danceability is not None:
            params["target_danceability"] = target_danceability
        if target_energy is not None:
            params["target_energy"] = target_energy
        if target_instrumentalness is not None:
            params["target_instrumentalness"] = target_instrumentalness
        if target_liveness is not None:
            params["target_liveness"] = target_liveness
        if target_speechiness is not None:
            params["target_speechiness"] = target_speechiness
        if target_valence is not None:
            params["target_valence"] = target_valence

        # Tempo
        if target_tempo is not None:
            params["target_tempo"] = target_tempo
        if min_tempo is not None:
            params["min_tempo"] = min_tempo
        if max_tempo is not None:
            params["max_tempo"] = max_tempo

        # Popularity
        if min_popularity is not None:
            params["min_popularity"] = min_popularity
        if max_popularity is not None:
            params["max_popularity"] = max_popularity

        result = await self._request("GET", "/recommendations", params=params)
        if not result:
            return []

        tracks = result.get("tracks", [])
        return [SpotifyTrack.from_api(t) for t in tracks]

    async def get_available_genres(self) -> list[str]:
        """Obtener géneros disponibles para recomendaciones"""
        result = await self._request("GET", "/recommendations/available-genre-seeds")
        if not result:
            return []

        return result.get("genres", [])

    # ==================== User Library ====================

    async def get_user_top_tracks(
        self,
        time_range: str = "medium_term",  # short_term, medium_term, long_term
        limit: int = 20
    ) -> list[SpotifyTrack]:
        """Obtener top tracks del usuario"""
        result = await self._request(
            "GET",
            "/me/top/tracks",
            params={"time_range": time_range, "limit": limit}
        )
        if not result:
            return []

        return [SpotifyTrack.from_api(t) for t in result.get("items", [])]

    async def get_user_top_artists(
        self,
        time_range: str = "medium_term",
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """Obtener top artistas del usuario"""
        result = await self._request(
            "GET",
            "/me/top/artists",
            params={"time_range": time_range, "limit": limit}
        )
        if not result:
            return []

        return result.get("items", [])

    async def get_recently_played(self, limit: int = 20) -> list[SpotifyTrack]:
        """Obtener tracks recientemente reproducidos"""
        result = await self._request(
            "GET",
            "/me/player/recently-played",
            params={"limit": limit}
        )
        if not result:
            return []

        return [SpotifyTrack.from_api(item["track"]) for item in result.get("items", [])]

    async def get_user_playlists(self, limit: int = 50) -> list[dict[str, Any]]:
        """Obtener playlists del usuario"""
        result = await self._request(
            "GET",
            "/me/playlists",
            params={"limit": limit}
        )
        if not result:
            return []

        return result.get("items", [])

    # ==================== Convenience Methods ====================

    async def play_artist(self, artist_name: str) -> bool:
        """Buscar y reproducir un artista"""
        artists = await self.search_artists(artist_name, limit=1)
        if not artists:
            logger.warning(f"Artist not found: {artist_name}")
            return False

        artist_uri = artists[0]["uri"]
        return await self.play(context_uri=artist_uri)

    async def play_track(self, track_name: str, artist: str | None = None) -> bool:
        """Buscar y reproducir un track"""
        query = f"{track_name} {artist}" if artist else track_name
        tracks = await self.search_tracks(query, limit=1)
        if not tracks:
            logger.warning(f"Track not found: {query}")
            return False

        return await self.play(uris=[tracks[0].uri])

    async def play_playlist(self, playlist_name: str) -> bool:
        """Buscar y reproducir una playlist"""
        playlists = await self.search_playlists(playlist_name, limit=1)
        if not playlists:
            logger.warning(f"Playlist not found: {playlist_name}")
            return False

        playlist_uri = playlists[0]["uri"]
        return await self.play(context_uri=playlist_uri)

    async def get_current_track_info(self) -> str | None:
        """Obtener info del track actual como string"""
        state = await self.get_playback_state()
        if not state or not state.track:
            return None

        track = state.track
        status = "reproduciendo" if state.is_playing else "pausado"
        return f"{track.name} de {track.artist_string} ({status})"
