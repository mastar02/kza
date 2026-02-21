"""
Spotify OAuth Authentication

Maneja el flujo de autenticación OAuth 2.0 con PKCE para Spotify.
Guarda y refresca tokens automáticamente.
"""

import json
import time
import base64
import hashlib
import secrets
import logging
import webbrowser
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs, urlparse
import threading

import aiohttp
import asyncio

logger = logging.getLogger(__name__)

# Spotify OAuth endpoints
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# Scopes necesarios para control completo
SPOTIFY_SCOPES = [
    # Playback
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    # Library
    "user-library-read",
    "user-library-modify",
    # Playlists
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
    # Listening history
    "user-top-read",
    "user-read-recently-played",
    # User profile
    "user-read-private",
    "user-read-email",
]


@dataclass
class SpotifyTokens:
    """Tokens de acceso de Spotify"""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    token_type: str = "Bearer"
    scope: str = ""

    @property
    def is_expired(self) -> bool:
        """Verificar si el token expiró (con 60s de margen)"""
        return time.time() >= (self.expires_at - 60)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpotifyTokens":
        return cls(**data)


class TokenManager:
    """
    Gestiona el almacenamiento y refresco de tokens.
    """

    def __init__(self, tokens_path: str = "./data/spotify_tokens.json"):
        self.tokens_path = Path(tokens_path)
        self.tokens_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens: Optional[SpotifyTokens] = None
        self._load_tokens()

    def _load_tokens(self):
        """Cargar tokens desde archivo"""
        if self.tokens_path.exists():
            try:
                with open(self.tokens_path, "r") as f:
                    data = json.load(f)
                    self._tokens = SpotifyTokens.from_dict(data)
                    logger.info("Spotify tokens loaded from file")
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")
                self._tokens = None

    def save_tokens(self, tokens: SpotifyTokens):
        """Guardar tokens a archivo"""
        self._tokens = tokens
        with open(self.tokens_path, "w") as f:
            json.dump(tokens.to_dict(), f, indent=2)
        logger.info("Spotify tokens saved")

    def get_tokens(self) -> Optional[SpotifyTokens]:
        """Obtener tokens actuales"""
        return self._tokens

    def clear_tokens(self):
        """Eliminar tokens"""
        self._tokens = None
        if self.tokens_path.exists():
            self.tokens_path.unlink()
        logger.info("Spotify tokens cleared")

    @property
    def has_valid_tokens(self) -> bool:
        """Verificar si hay tokens válidos"""
        return self._tokens is not None and not self._tokens.is_expired


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handler para recibir el callback de OAuth"""

    def do_GET(self):
        """Procesar callback con código de autorización"""
        query = parse_qs(urlparse(self.path).query)

        if "code" in query:
            self.server.auth_code = query["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h1>Autorizacion Exitosa!</h1>
                    <p>Puedes cerrar esta ventana.</p>
                    <script>setTimeout(() => window.close(), 2000);</script>
                </body>
                </html>
            """)
        elif "error" in query:
            self.server.auth_error = query.get("error_description", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<h1>Error: {self.server.auth_error}</h1>".encode())

    def log_message(self, format, *args):
        """Silenciar logs del servidor HTTP"""
        pass


class SpotifyAuth:
    """
    Maneja la autenticación OAuth 2.0 con Spotify usando PKCE.

    Uso:
        auth = SpotifyAuth(client_id="tu_client_id")

        # Primera vez: autorizar
        if not auth.is_authenticated:
            await auth.authorize()

        # Obtener token (refresca automáticamente si expiró)
        token = await auth.get_access_token()
    """

    def __init__(
        self,
        client_id: str,
        client_secret: Optional[str] = None,
        redirect_port: int = 8888,
        tokens_path: str = "./data/spotify_tokens.json",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = f"http://localhost:{redirect_port}/callback"
        self.redirect_port = redirect_port
        self.token_manager = TokenManager(tokens_path)

        # PKCE
        self._code_verifier: Optional[str] = None

    @property
    def is_authenticated(self) -> bool:
        """Verificar si hay tokens (pueden necesitar refresh)"""
        return self.token_manager.get_tokens() is not None

    def _generate_pkce(self) -> tuple[str, str]:
        """Generar code_verifier y code_challenge para PKCE"""
        # Code verifier: 43-128 caracteres aleatorios
        code_verifier = secrets.token_urlsafe(64)

        # Code challenge: SHA256 hash del verifier, base64url encoded
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")

        return code_verifier, code_challenge

    def get_auth_url(self) -> str:
        """Obtener URL de autorización"""
        self._code_verifier, code_challenge = self._generate_pkce()

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(SPOTIFY_SCOPES),
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
        }

        return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

    async def authorize(self, open_browser: bool = True) -> bool:
        """
        Iniciar flujo de autorización OAuth.

        Args:
            open_browser: Abrir navegador automáticamente

        Returns:
            True si la autorización fue exitosa
        """
        auth_url = self.get_auth_url()

        # Iniciar servidor HTTP para callback
        server = HTTPServer(("localhost", self.redirect_port), OAuthCallbackHandler)
        server.auth_code = None
        server.auth_error = None
        server.timeout = 120  # 2 minutos para completar

        logger.info(f"Starting OAuth callback server on port {self.redirect_port}")

        if open_browser:
            webbrowser.open(auth_url)
            logger.info("Browser opened for Spotify authorization")
        else:
            print(f"\nAbre esta URL para autorizar:\n{auth_url}\n")

        # Esperar callback en thread separado
        def wait_for_callback():
            while server.auth_code is None and server.auth_error is None:
                server.handle_request()

        thread = threading.Thread(target=wait_for_callback)
        thread.start()
        thread.join(timeout=120)

        server.server_close()

        if server.auth_error:
            logger.error(f"Authorization failed: {server.auth_error}")
            return False

        if not server.auth_code:
            logger.error("Authorization timed out")
            return False

        # Intercambiar código por tokens
        return await self._exchange_code(server.auth_code)

    async def _exchange_code(self, code: str) -> bool:
        """Intercambiar código de autorización por tokens"""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": self._code_verifier,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # Si tenemos client_secret, usarlo
        if self.client_secret:
            auth_str = f"{self.client_id}:{self.client_secret}"
            auth_b64 = base64.b64encode(auth_str.encode()).decode()
            headers["Authorization"] = f"Basic {auth_b64}"

        async with aiohttp.ClientSession() as session:
            async with session.post(SPOTIFY_TOKEN_URL, data=data, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token exchange failed: {error}")
                    return False

                token_data = await resp.json()
                tokens = SpotifyTokens(
                    access_token=token_data["access_token"],
                    refresh_token=token_data["refresh_token"],
                    expires_at=time.time() + token_data["expires_in"],
                    token_type=token_data.get("token_type", "Bearer"),
                    scope=token_data.get("scope", ""),
                )
                self.token_manager.save_tokens(tokens)
                logger.info("Spotify authorization successful")
                return True

    async def refresh_tokens(self) -> bool:
        """Refrescar tokens expirados"""
        tokens = self.token_manager.get_tokens()
        if not tokens:
            logger.error("No tokens to refresh")
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": self.client_id,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if self.client_secret:
            auth_str = f"{self.client_id}:{self.client_secret}"
            auth_b64 = base64.b64encode(auth_str.encode()).decode()
            headers["Authorization"] = f"Basic {auth_b64}"

        async with aiohttp.ClientSession() as session:
            async with session.post(SPOTIFY_TOKEN_URL, data=data, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token refresh failed: {error}")
                    return False

                token_data = await resp.json()
                new_tokens = SpotifyTokens(
                    access_token=token_data["access_token"],
                    refresh_token=token_data.get("refresh_token", tokens.refresh_token),
                    expires_at=time.time() + token_data["expires_in"],
                    token_type=token_data.get("token_type", "Bearer"),
                    scope=token_data.get("scope", tokens.scope),
                )
                self.token_manager.save_tokens(new_tokens)
                logger.info("Spotify tokens refreshed")
                return True

    async def get_access_token(self) -> Optional[str]:
        """
        Obtener token de acceso válido.
        Refresca automáticamente si está expirado.
        """
        tokens = self.token_manager.get_tokens()
        if not tokens:
            return None

        if tokens.is_expired:
            if not await self.refresh_tokens():
                return None
            tokens = self.token_manager.get_tokens()

        return tokens.access_token if tokens else None

    def logout(self):
        """Cerrar sesión (eliminar tokens)"""
        self.token_manager.clear_tokens()
        logger.info("Logged out from Spotify")
