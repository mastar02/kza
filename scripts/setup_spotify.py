#!/usr/bin/env python3
"""
Spotify Setup Script
Configura y autoriza Spotify para el asistente de voz.

Uso:
    python scripts/setup_spotify.py

Requisitos previos:
    1. Crear app en https://developer.spotify.com/dashboard
    2. Agregar redirect URI: http://localhost:8888/callback
    3. Tener SPOTIFY_CLIENT_ID y SPOTIFY_CLIENT_SECRET en .env
"""

import os
import sys
import asyncio

# Agregar src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv


def print_header():
    print("\n" + "=" * 60)
    print("   SPOTIFY SETUP - Home Assistant Voice")
    print("=" * 60 + "\n")


def print_step(step: int, text: str):
    print(f"\n[Paso {step}] {text}")
    print("-" * 50)


async def main():
    print_header()

    # Cargar variables de entorno
    load_dotenv()

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    # Verificar credenciales
    print_step(1, "Verificando credenciales")

    if not client_id or client_id == "your_spotify_client_id_here":
        print("\n❌ SPOTIFY_CLIENT_ID no configurado")
        print("\nPasos para obtenerlo:")
        print("  1. Ve a https://developer.spotify.com/dashboard")
        print("  2. Inicia sesión con tu cuenta de Spotify")
        print("  3. Click en 'Create App'")
        print("  4. Nombre: 'Home Assistant Voice'")
        print("  5. Redirect URI: http://localhost:8888/callback")
        print("  6. Copia el Client ID")
        print("\n  Luego agrega a tu archivo .env:")
        print("  SPOTIFY_CLIENT_ID=tu_client_id_aqui")
        return

    print(f"✓ Client ID: {client_id[:8]}...")

    if client_secret and client_secret != "your_spotify_client_secret_here":
        print(f"✓ Client Secret: {client_secret[:8]}...")
    else:
        print("⚠ Client Secret no configurado (OK si usas PKCE)")

    # Importar después de verificar variables
    print_step(2, "Inicializando cliente de Spotify")

    try:
        from src.spotify import SpotifyAuth, SpotifyClient
    except ImportError as e:
        print(f"\n❌ Error importando módulo: {e}")
        print("  Asegúrate de instalar dependencias: pip install aiohttp")
        return

    auth = SpotifyAuth(
        client_id=client_id,
        client_secret=client_secret if client_secret != "your_spotify_client_secret_here" else None,
        tokens_path="./data/spotify_tokens.json"
    )

    # Verificar si ya está autorizado
    if auth.is_authenticated:
        print("\n✓ Ya tienes tokens guardados")

        # Verificar si funcionan
        print_step(3, "Verificando conexión")

        client = SpotifyClient(auth)
        try:
            devices = await client.get_devices()
            if devices:
                print("\n✓ Conexión exitosa!")
                print("\nDispositivos disponibles:")
                for d in devices:
                    active = " (activo)" if d.is_active else ""
                    print(f"  • {d.name} ({d.type}){active}")
            else:
                print("\n⚠ Conexión OK pero no hay dispositivos activos")
                print("  Abre Spotify en algún dispositivo para reproducir música")

            # Probar búsqueda
            print_step(4, "Probando búsqueda")
            tracks = await client.search_tracks("test", limit=1)
            if tracks:
                print(f"✓ Búsqueda funcionando: encontré '{tracks[0].name}'")

            await client.close()
            print("\n" + "=" * 60)
            print("   ✅ SPOTIFY CONFIGURADO CORRECTAMENTE")
            print("=" * 60 + "\n")
            return

        except Exception as e:
            print(f"\n⚠ Tokens expirados o inválidos: {e}")
            print("  Reautorizando...")
            auth.logout()

    # Autorizar
    print_step(3, "Autorizando con Spotify")

    print("\nSe abrirá tu navegador para autorizar la aplicación.")
    print("Después de autorizar, serás redirigido a localhost.")
    input("\nPresiona ENTER para continuar...")

    success = await auth.authorize(open_browser=True)

    if not success:
        print("\n❌ Error en la autorización")
        print("  Verifica que:")
        print("  1. El Client ID sea correcto")
        print("  2. El redirect URI sea http://localhost:8888/callback")
        print("  3. Tu cuenta tenga Spotify Premium")
        return

    print("\n✓ Autorización exitosa!")

    # Verificar
    print_step(4, "Verificando conexión")

    client = SpotifyClient(auth)
    try:
        devices = await client.get_devices()
        print("\n✓ Conexión verificada!")

        if devices:
            print("\nDispositivos disponibles:")
            for d in devices:
                active = " (activo)" if d.is_active else ""
                print(f"  • {d.name} ({d.type}){active}")
        else:
            print("\n⚠ No hay dispositivos activos")
            print("  Abre Spotify en tu computadora o teléfono")

        await client.close()

    except Exception as e:
        print(f"\n⚠ Error verificando: {e}")

    print("\n" + "=" * 60)
    print("   ✅ SPOTIFY CONFIGURADO CORRECTAMENTE")
    print("=" * 60)
    print("\nAhora puedes usar comandos como:")
    print('  • "Pon música de Bad Bunny"')
    print('  • "Pon algo para entrenar"')
    print('  • "Música para una cena romántica"')
    print('  • "Pausa" / "Siguiente canción"')
    print()


if __name__ == "__main__":
    asyncio.run(main())
