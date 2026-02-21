"""
Mood Mapper - Mapeo de lenguaje natural a Spotify Audio Features

Convierte expresiones como "música para una cena romántica" en parámetros
de audio features que Spotify usa para recomendaciones.

Audio Features de Spotify:
- energy (0.0-1.0): Intensidad y actividad
- valence (0.0-1.0): Positividad musical (alegre vs triste)
- danceability (0.0-1.0): Qué tan bailable es
- acousticness (0.0-1.0): Probabilidad de ser acústico
- instrumentalness (0.0-1.0): Predice si no tiene vocals
- tempo (BPM): Velocidad de la música
- speechiness (0.0-1.0): Presencia de palabras habladas
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """
    Audio features para recomendaciones de Spotify.
    Valores None se ignoran en la petición.
    """
    energy: Optional[float] = None
    valence: Optional[float] = None
    danceability: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    speechiness: Optional[float] = None
    tempo: Optional[float] = None
    min_tempo: Optional[float] = None
    max_tempo: Optional[float] = None

    def to_recommendation_params(self) -> Dict[str, float]:
        """Convertir a parámetros para la API de recomendaciones"""
        params = {}
        if self.energy is not None:
            params["target_energy"] = self.energy
        if self.valence is not None:
            params["target_valence"] = self.valence
        if self.danceability is not None:
            params["target_danceability"] = self.danceability
        if self.acousticness is not None:
            params["target_acousticness"] = self.acousticness
        if self.instrumentalness is not None:
            params["target_instrumentalness"] = self.instrumentalness
        if self.speechiness is not None:
            params["target_speechiness"] = self.speechiness
        if self.tempo is not None:
            params["target_tempo"] = self.tempo
        if self.min_tempo is not None:
            params["min_tempo"] = self.min_tempo
        if self.max_tempo is not None:
            params["max_tempo"] = self.max_tempo
        return params

    def merge_with(self, other: "AudioFeatures") -> "AudioFeatures":
        """Combinar con otros features (el otro tiene prioridad)"""
        return AudioFeatures(
            energy=other.energy if other.energy is not None else self.energy,
            valence=other.valence if other.valence is not None else self.valence,
            danceability=other.danceability if other.danceability is not None else self.danceability,
            acousticness=other.acousticness if other.acousticness is not None else self.acousticness,
            instrumentalness=other.instrumentalness if other.instrumentalness is not None else self.instrumentalness,
            speechiness=other.speechiness if other.speechiness is not None else self.speechiness,
            tempo=other.tempo if other.tempo is not None else self.tempo,
            min_tempo=other.min_tempo if other.min_tempo is not None else self.min_tempo,
            max_tempo=other.max_tempo if other.max_tempo is not None else self.max_tempo,
        )


@dataclass
class MoodProfile:
    """
    Perfil completo de mood para música.
    Incluye audio features, géneros sugeridos y configuración de playback.
    """
    name: str
    description: str
    features: AudioFeatures
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # Palabras que activan este mood
    shuffle: bool = True
    min_popularity: Optional[int] = None

    def matches_query(self, query: str) -> bool:
        """Verificar si el query matchea con este mood"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)


# =============================================================================
# PERFILES DE MOOD PREDEFINIDOS
# =============================================================================
# Estos mapean contextos comunes a audio features de Spotify

MOOD_PROFILES: Dict[str, MoodProfile] = {
    # -------------------------------------------------------------------------
    # Estados de ánimo
    # -------------------------------------------------------------------------
    "happy": MoodProfile(
        name="Feliz",
        description="Música alegre y positiva",
        features=AudioFeatures(energy=0.7, valence=0.8, danceability=0.7),
        genres=["pop", "dance", "happy"],
        keywords=["feliz", "alegre", "contento", "animado", "positivo", "good vibes"],
    ),
    "sad": MoodProfile(
        name="Triste",
        description="Música melancólica para momentos tristes",
        features=AudioFeatures(energy=0.3, valence=0.2, acousticness=0.6),
        genres=["sad", "acoustic", "singer-songwriter"],
        keywords=["triste", "melancólico", "down", "depre", "llorar"],
    ),
    "calm": MoodProfile(
        name="Tranquilo",
        description="Música calmada y relajante",
        features=AudioFeatures(energy=0.3, valence=0.5, acousticness=0.7, tempo=80),
        genres=["ambient", "chill", "acoustic"],
        keywords=["tranquilo", "calma", "paz", "sereno", "suave"],
    ),
    "energetic": MoodProfile(
        name="Energético",
        description="Música con mucha energía",
        features=AudioFeatures(energy=0.9, valence=0.7, danceability=0.8, min_tempo=120),
        genres=["edm", "dance", "electronic", "pop"],
        keywords=["energía", "energético", "power", "intenso", "potente"],
    ),
    "romantic": MoodProfile(
        name="Romántico",
        description="Música para momentos románticos",
        features=AudioFeatures(energy=0.4, valence=0.6, acousticness=0.5, tempo=90),
        genres=["r-n-b", "soul", "jazz"],
        keywords=["romántico", "amor", "romance", "love", "pareja"],
    ),

    # -------------------------------------------------------------------------
    # Actividades
    # -------------------------------------------------------------------------
    "workout": MoodProfile(
        name="Ejercicio",
        description="Música para entrenar con energía",
        features=AudioFeatures(energy=0.9, danceability=0.7, valence=0.7, min_tempo=130),
        genres=["edm", "hip-hop", "electronic", "work-out"],
        keywords=["ejercicio", "gym", "entrenar", "workout", "correr", "cardio", "pesas"],
    ),
    "focus": MoodProfile(
        name="Concentración",
        description="Música para trabajar y concentrarse",
        features=AudioFeatures(energy=0.4, instrumentalness=0.8, speechiness=0.1, tempo=110),
        genres=["ambient", "electronic", "classical", "study"],
        keywords=["concentrar", "trabajar", "estudiar", "focus", "productivo", "programar"],
    ),
    "sleep": MoodProfile(
        name="Dormir",
        description="Música para dormir",
        features=AudioFeatures(energy=0.1, acousticness=0.8, instrumentalness=0.7, max_tempo=70),
        genres=["ambient", "sleep", "classical"],
        keywords=["dormir", "sleep", "descansar", "noche", "cama", "sueño"],
    ),
    "cooking": MoodProfile(
        name="Cocinar",
        description="Música amena para cocinar",
        features=AudioFeatures(energy=0.6, valence=0.7, danceability=0.6),
        genres=["jazz", "soul", "funk", "bossa-nova"],
        keywords=["cocinar", "cooking", "cocina", "preparar comida"],
    ),
    "party": MoodProfile(
        name="Fiesta",
        description="Música para fiestas",
        features=AudioFeatures(energy=0.85, danceability=0.85, valence=0.8, min_tempo=115),
        genres=["dance", "edm", "pop", "reggaeton", "hip-hop"],
        keywords=["fiesta", "party", "bailar", "celebrar", "antro", "disco"],
    ),
    "driving": MoodProfile(
        name="Manejar",
        description="Música para el auto",
        features=AudioFeatures(energy=0.65, valence=0.6, danceability=0.5),
        genres=["rock", "pop", "indie"],
        keywords=["manejar", "conducir", "auto", "carretera", "driving", "viaje"],
    ),
    "morning": MoodProfile(
        name="Mañana",
        description="Música para empezar el día",
        features=AudioFeatures(energy=0.5, valence=0.7, acousticness=0.4),
        genres=["pop", "indie", "acoustic"],
        keywords=["mañana", "despertar", "morning", "amanecer"],
    ),
    "dinner": MoodProfile(
        name="Cena",
        description="Música de fondo para cenas",
        features=AudioFeatures(energy=0.35, valence=0.55, acousticness=0.6, instrumentalness=0.3),
        genres=["jazz", "bossa-nova", "soul", "lounge"],
        keywords=["cena", "cenar", "dinner", "comida"],
    ),

    # -------------------------------------------------------------------------
    # Géneros específicos
    # -------------------------------------------------------------------------
    "jazz": MoodProfile(
        name="Jazz",
        description="Jazz clásico y moderno",
        features=AudioFeatures(acousticness=0.5, instrumentalness=0.4),
        genres=["jazz"],
        keywords=["jazz"],
    ),
    "classical": MoodProfile(
        name="Clásica",
        description="Música clásica",
        features=AudioFeatures(acousticness=0.8, instrumentalness=0.9),
        genres=["classical"],
        keywords=["clásica", "classical", "orquesta", "sinfónica"],
    ),
    "electronic": MoodProfile(
        name="Electrónica",
        description="Música electrónica",
        features=AudioFeatures(energy=0.7, danceability=0.7, acousticness=0.1),
        genres=["electronic", "edm", "house", "techno"],
        keywords=["electrónica", "electronic", "edm", "house", "techno"],
    ),
    "rock": MoodProfile(
        name="Rock",
        description="Rock en sus variantes",
        features=AudioFeatures(energy=0.75, acousticness=0.2),
        genres=["rock", "hard-rock", "alt-rock", "indie-rock"],
        keywords=["rock"],
    ),
    "latin": MoodProfile(
        name="Latino",
        description="Música latina",
        features=AudioFeatures(energy=0.7, danceability=0.75, valence=0.7),
        genres=["latin", "reggaeton", "salsa", "latin-pop"],
        keywords=["latino", "latina", "reggaeton", "salsa", "bachata", "cumbia"],
    ),
    "hiphop": MoodProfile(
        name="Hip Hop",
        description="Hip hop y rap",
        features=AudioFeatures(energy=0.7, danceability=0.7, speechiness=0.2),
        genres=["hip-hop", "rap"],
        keywords=["hip hop", "hiphop", "rap", "trap"],
    ),
}


class MoodMapper:
    """
    Mapea lenguaje natural a audio features de Spotify.

    Uso:
        mapper = MoodMapper(llm=reasoner)

        # Mapeo directo por keywords
        profile = mapper.get_mood_profile("algo para entrenar")

        # Mapeo con LLM para contextos complejos
        profile = await mapper.interpret_with_llm("música para una cena romántica a la luz de las velas")
    """

    def __init__(self, llm=None, custom_profiles: Optional[Dict[str, MoodProfile]] = None):
        """
        Args:
            llm: LLM para interpretar contextos complejos (opcional)
            custom_profiles: Perfiles personalizados adicionales
        """
        self.llm = llm
        self.profiles = {**MOOD_PROFILES}

        if custom_profiles:
            self.profiles.update(custom_profiles)

    def get_mood_profile(self, query: str) -> Optional[MoodProfile]:
        """
        Obtener perfil de mood basado en keywords.

        Args:
            query: Texto del usuario

        Returns:
            MoodProfile si hay match, None si no
        """
        query_lower = query.lower()

        # Buscar match por keywords
        for profile in self.profiles.values():
            if profile.matches_query(query_lower):
                logger.debug(f"Matched mood profile: {profile.name}")
                return profile

        return None

    def get_profile_by_name(self, name: str) -> Optional[MoodProfile]:
        """Obtener perfil por nombre"""
        return self.profiles.get(name.lower())

    def list_available_moods(self) -> List[str]:
        """Listar moods disponibles"""
        return list(self.profiles.keys())

    async def interpret_with_llm(self, query: str) -> Optional[MoodProfile]:
        """
        Usar LLM para interpretar contextos complejos.

        Args:
            query: Texto del usuario (ej: "música para una cena romántica")

        Returns:
            MoodProfile interpretado por el LLM
        """
        if not self.llm:
            logger.warning("No LLM available for mood interpretation")
            return None

        # Prompt para el LLM
        prompt = f"""Analiza la siguiente petición de música y extrae los parámetros.

Petición: "{query}"

Responde SOLO con un JSON válido con estos campos (usa valores entre 0.0 y 1.0, o null si no aplica):
{{
    "mood_name": "nombre corto del mood",
    "energy": 0.0-1.0 o null,
    "valence": 0.0-1.0 o null (positividad/alegría),
    "danceability": 0.0-1.0 o null,
    "acousticness": 0.0-1.0 o null,
    "instrumentalness": 0.0-1.0 o null,
    "tempo_bpm": número o null,
    "genres": ["lista", "de", "géneros"],
    "reasoning": "breve explicación"
}}

Géneros válidos de Spotify: acoustic, afrobeat, alt-rock, ambient, blues, bossa-nova,
chill, classical, club, country, dance, disco, electronic, folk, funk, groove,
happy, hard-rock, hip-hop, house, indie, indie-pop, jazz, k-pop, latin,
lounge, metal, party, piano, pop, punk, r-n-b, reggae, reggaeton, rock,
romance, sad, salsa, samba, singer-songwriter, sleep, soul, spanish, study,
summer, techno, trance, trip-hop, work-out, world-music"""

        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.3)

            # Extraer JSON de la respuesta
            import json
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                logger.warning("Could not extract JSON from LLM response")
                return None

            data = json.loads(json_match.group())

            # Construir AudioFeatures
            features = AudioFeatures(
                energy=data.get("energy"),
                valence=data.get("valence"),
                danceability=data.get("danceability"),
                acousticness=data.get("acousticness"),
                instrumentalness=data.get("instrumentalness"),
                tempo=data.get("tempo_bpm"),
            )

            profile = MoodProfile(
                name=data.get("mood_name", "custom"),
                description=data.get("reasoning", query),
                features=features,
                genres=data.get("genres", [])[:5],  # Max 5 genres
                keywords=[],
            )

            logger.info(f"LLM interpreted mood: {profile.name} - {profile.description}")
            return profile

        except Exception as e:
            logger.error(f"Failed to interpret mood with LLM: {e}")
            return None

    def add_custom_profile(self, key: str, profile: MoodProfile):
        """Agregar perfil personalizado"""
        self.profiles[key] = profile
        logger.info(f"Added custom mood profile: {key}")

    def extract_artist_or_track(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extraer mención de artista o canción del query.

        Args:
            query: "Pon música de Bad Bunny" o "Pon la canción Blinding Lights"

        Returns:
            {"artist": str|None, "track": str|None}
        """
        result = {"artist": None, "track": None}
        query_lower = query.lower()

        # Patrones para artistas
        artist_patterns = [
            r"(?:música|algo|canciones?)\s+de\s+(.+?)(?:\s+(?:para|mientras|que)|$)",
            r"(?:pon(?:me)?|reproduce)\s+(?:a\s+)?(.+?)(?:\s+(?:para|mientras)|$)",
        ]

        # Patrones para tracks
        track_patterns = [
            r"(?:la\s+)?canci[oó]n\s+['\"]?(.+?)['\"]?(?:\s+de\s+(.+?))?$",
            r"(?:pon(?:me)?|reproduce)\s+['\"](.+?)['\"]",
        ]

        for pattern in artist_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result["artist"] = match.group(1).strip()
                break

        for pattern in track_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result["track"] = match.group(1).strip()
                if match.lastindex and match.lastindex >= 2:
                    result["artist"] = match.group(2).strip() if match.group(2) else None
                break

        return result
