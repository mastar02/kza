# Integración de Spotify

Sistema de control de música por voz con interpretación de contexto usando la API de Spotify.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRADA DE VOZ                                  │
│                        "Pon música para cocinar"                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST DISPATCHER                                 │
│                                                                             │
│   Detecta keywords de música:                                               │
│   • MUSIC_DIRECT: "pon música de", "pausa", "siguiente"  → FAST_MUSIC      │
│   • MUSIC_CONTEXT: "música para", "algo tranquilo"       → SLOW_MUSIC      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │    FAST_MUSIC     │           │    SLOW_MUSIC     │
        │    (~500ms)       │           │    (~3-5s)        │
        ├───────────────────┤           ├───────────────────┤
        │                   │           │                   │
        │ • Búsqueda directa│           │ • MoodMapper      │
        │ • Control playback│           │ • LLM interpreta  │
        │ • Mood por keyword│           │ • Audio features  │
        │                   │           │                   │
        └─────────┬─────────┘           └─────────┬─────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MUSIC DISPATCHER                                   │
│                                                                             │
│   Detecta intent:                                                           │
│   • PLAY_ARTIST    → search_artists + play                                  │
│   • PLAY_TRACK     → search_tracks + play                                   │
│   • PLAY_MOOD      → get_recommendations(audio_features) + play             │
│   • PLAY_CONTEXT   → LLM → MoodProfile → recommendations + play             │
│   • PAUSE/NEXT/... → control directo                                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SPOTIFY API                                        │
│                                                                             │
│   • Search: tracks, artists, playlists                                      │
│   • Recommendations: seed_genres + target_energy + target_valence + ...     │
│   • Playback: play, pause, next, previous, volume, shuffle                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Componentes

### 1. SpotifyAuth (`auth.py`)

Maneja autenticación OAuth 2.0 con PKCE.

```python
from src.spotify import SpotifyAuth

auth = SpotifyAuth(
    client_id="tu_client_id",
    client_secret="tu_client_secret",  # opcional con PKCE
    redirect_port=8888,
    tokens_path="./data/spotify_tokens.json"
)

# Primera vez: autorizar (abre navegador)
if not auth.is_authenticated:
    await auth.authorize()

# Obtener token (refresca automáticamente si expiró)
token = await auth.get_access_token()
```

**Características:**
- OAuth 2.0 con PKCE (no requiere servidor)
- Auto-refresh de tokens expirados
- Persistencia en archivo JSON
- Scopes completos para playback y biblioteca

### 2. SpotifyClient (`client.py`)

Cliente asíncrono para la API de Spotify.

```python
from src.spotify import SpotifyClient

client = SpotifyClient(auth)

# Búsqueda
tracks = await client.search_tracks("Bad Bunny", limit=10)
artists = await client.search_artists("Coldplay")
playlists = await client.search_playlists("workout")

# Reproducción
await client.play(uris=[tracks[0].uri])
await client.play(context_uri="spotify:artist:xxx")
await client.pause()
await client.next_track()
await client.set_volume(50)

# Recomendaciones con audio features
tracks = await client.get_recommendations(
    seed_genres=["jazz", "soul"],
    target_energy=0.4,
    target_valence=0.6,
    target_tempo=90,
    limit=20
)

# Métodos de conveniencia
await client.play_artist("Taylor Swift")
await client.play_track("Blinding Lights")
await client.play_playlist("Chill Vibes")

# Info actual
info = await client.get_current_track_info()
# "Shape of You de Ed Sheeran (reproduciendo)"
```

### 3. MoodMapper (`mood_mapper.py`)

Mapea lenguaje natural a Spotify Audio Features.

```python
from src.spotify import MoodMapper

mapper = MoodMapper(llm=reasoner)

# Mapeo por keywords (rápido)
profile = mapper.get_mood_profile("algo para entrenar")
# MoodProfile(name="Ejercicio", energy=0.9, danceability=0.7, ...)

# Mapeo con LLM (contexto complejo)
profile = await mapper.interpret_with_llm(
    "música para una cena romántica a la luz de las velas"
)
# MoodProfile(name="romantic_dinner", energy=0.3, valence=0.6,
#             genres=["jazz", "soul", "bossa-nova"])
```

**Perfiles predefinidos:**

| Perfil | Energy | Valence | Géneros | Keywords |
|--------|--------|---------|---------|----------|
| happy | 0.7 | 0.8 | pop, dance | feliz, alegre |
| sad | 0.3 | 0.2 | acoustic | triste, melancólico |
| calm | 0.3 | 0.5 | ambient, chill | tranquilo, calma |
| energetic | 0.9 | 0.7 | edm, electronic | energía, intenso |
| romantic | 0.4 | 0.6 | r-n-b, soul, jazz | romántico, amor |
| workout | 0.9 | 0.7 | edm, hip-hop | gym, entrenar |
| focus | 0.4 | - | ambient, classical | concentrar, estudiar |
| sleep | 0.1 | - | ambient, sleep | dormir, descansar |
| cooking | 0.6 | 0.7 | jazz, funk | cocinar, cocina |
| party | 0.85 | 0.8 | dance, reggaeton | fiesta, bailar |
| dinner | 0.35 | 0.55 | jazz, bossa-nova | cena, cenar |

### 4. MusicDispatcher (`music_dispatcher.py`)

Enruta comandos de música al handler correcto.

```python
from src.spotify import MusicDispatcher

dispatcher = MusicDispatcher(
    spotify_client=client,
    mood_mapper=mapper,
    llm=reasoner
)

# Procesar comando
result = await dispatcher.process("Pon música de Bad Bunny")
# MusicResult(
#     success=True,
#     response="Reproduciendo Bad Bunny",
#     intent=MusicIntent.PLAY_ARTIST,
#     latency_ms=450
# )

result = await dispatcher.process("Algo para una cena romántica")
# MusicResult(
#     success=True,
#     response="Reproduciendo jazz romántico",
#     intent=MusicIntent.PLAY_CONTEXT,
#     details={"interpreted_as": "romantic_dinner"}
# )
```

**Intents soportados:**

| Intent | Ejemplo | Latencia |
|--------|---------|----------|
| PLAY_ARTIST | "Pon música de Taylor Swift" | ~500ms |
| PLAY_TRACK | "Pon la canción Blinding Lights" | ~500ms |
| PLAY_PLAYLIST | "Pon mi playlist de ejercicio" | ~500ms |
| PLAY_MOOD | "Pon algo tranquilo" | ~600ms |
| PLAY_CONTEXT | "Música para cocinar" | ~3-5s |
| PLAY_SIMILAR | "Pon algo parecido" | ~800ms |
| PAUSE | "Pausa" | ~200ms |
| RESUME | "Continúa" | ~200ms |
| NEXT | "Siguiente canción" | ~200ms |
| PREVIOUS | "Anterior" | ~200ms |
| VOLUME | "Volumen al 50" | ~200ms |
| WHATS_PLAYING | "¿Qué suena?" | ~300ms |

## Audio Features de Spotify

La API de Spotify proporciona métricas para cada track:

| Feature | Rango | Descripción |
|---------|-------|-------------|
| energy | 0.0-1.0 | Intensidad y actividad |
| valence | 0.0-1.0 | Positividad musical (alegre vs triste) |
| danceability | 0.0-1.0 | Qué tan bailable es |
| acousticness | 0.0-1.0 | Probabilidad de ser acústico |
| instrumentalness | 0.0-1.0 | Predice si no tiene vocals |
| tempo | BPM | Velocidad (60-200 típico) |
| speechiness | 0.0-1.0 | Presencia de palabras habladas |

**Ejemplos de mapeo:**

```
"Música para entrenar"
  → energy=0.9, danceability=0.7, min_tempo=130

"Música para dormir"
  → energy=0.1, acousticness=0.8, max_tempo=70

"Música para una cena romántica"
  → energy=0.35, valence=0.55, genres=[jazz, bossa-nova]
```

## Configuración

### 1. Crear App en Spotify Developer

1. Ir a https://developer.spotify.com/dashboard
2. Crear nueva aplicación
3. Agregar Redirect URI: `http://localhost:8888/callback`
4. Copiar Client ID y Client Secret

### 2. Variables de Entorno

```bash
# .env
SPOTIFY_CLIENT_ID=tu_client_id
SPOTIFY_CLIENT_SECRET=tu_client_secret
```

### 3. Configuración en settings.yaml

```yaml
spotify:
  enabled: true
  client_id: "${SPOTIFY_CLIENT_ID}"
  client_secret: "${SPOTIFY_CLIENT_SECRET}"

  auth:
    redirect_port: 8888
    tokens_path: "./data/spotify_tokens.json"

  playback:
    default_limit: 20
    shuffle_by_default: true
    market: "ES"

  mood_interpretation:
    use_llm_for_context: true
    llm_timeout: 5.0
```

### 4. Ejecutar Setup

```bash
python scripts/setup_spotify.py
```

## Integración con el Orquestador

El dispatcher detecta comandos de música y los enruta apropiadamente:

```python
# En dispatcher.py

# Keywords que activan fast path
MUSIC_DIRECT_KEYWORDS = [
    "pon música de", "música de", "reproduce",
    "pausa", "siguiente", "anterior"
]

# Keywords que activan slow path (LLM)
MUSIC_CONTEXT_KEYWORDS = [
    "música para", "algo para", "algo tranquilo"
]
```

**Flujo en el pipeline:**

```
VoicePipeline.process_command()
    │
    ▼
Dispatcher.dispatch()
    │
    ├─ _classify_request() detecta música
    │
    ├─ FAST_MUSIC → _fast_music_path()
    │                   └─ MusicDispatcher.process()
    │
    └─ SLOW_MUSIC → _slow_music_path()
                        └─ MusicDispatcher.process() con LLM
```

## Preferencias por Usuario

Cada usuario puede tener preferencias de música:

```python
from src.orchestrator import MusicPreferences

prefs = MusicPreferences(
    favorite_genres=["rock", "indie"],
    favorite_artists=["Coldplay", "Arctic Monkeys"],
    disliked_genres=["reggaeton"],
    default_energy=0.6,
    default_valence=0.7
)

# Se usan automáticamente en recomendaciones
result = await music_dispatcher.process(
    "Pon algo alegre",
    user_preferences=prefs.to_dict()
)
# Mezcla géneros del mood con favoritos del usuario
```

## Ejemplos de Uso

### Búsqueda Directa
```
Usuario: "Pon música de Bad Bunny"
Sistema: "Reproduciendo Bad Bunny" (500ms)
```

### Mood por Keyword
```
Usuario: "Pon algo tranquilo"
Sistema: "Reproduciendo música tranquila" (600ms)
  → MoodMapper detecta keyword "tranquilo"
  → Usa perfil predefinido "calm"
```

### Contexto Complejo
```
Usuario: "Pon música para una cena romántica a la luz de las velas"
Sistema: "Reproduciendo jazz romántico para tu cena" (3-5s)
  → LLM interpreta contexto
  → Extrae: mood=romantic, setting=dinner, ambiance=candlelight
  → Genera: energy=0.3, valence=0.6, genres=[jazz, soul, bossa-nova]
```

### Control de Playback
```
Usuario: "Pausa"
Sistema: "Música pausada" (200ms)

Usuario: "Siguiente canción"
Sistema: "Siguiente canción" (200ms)

Usuario: "¿Qué está sonando?"
Sistema: "Shape of You de Ed Sheeran" (300ms)
```

## Limitaciones

1. **Requiere Spotify Premium** - La API de control de playback solo funciona con Premium
2. **Dispositivo activo necesario** - Debe haber un dispositivo de Spotify abierto
3. **Latencia de LLM** - Contextos complejos toman 3-5s por la interpretación
4. **Rate limits** - Spotify tiene límites de requests por minuto

## Troubleshooting

### "No hay dispositivos activos"
- Abre Spotify en tu computadora, teléfono o smart speaker
- Reproduce cualquier canción brevemente para activar el dispositivo

### "Token expirado"
- Los tokens se refrescan automáticamente
- Si falla, ejecuta `python scripts/setup_spotify.py` de nuevo

### "No encontré al artista/canción"
- Verifica la ortografía
- Intenta con el nombre en inglés si es artista internacional
