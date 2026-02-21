#!/usr/bin/env python3
"""
KZA Voice - Multi-Zone Setup Helper
Herramienta para configurar y probar el sistema multi-zona.
"""

import argparse
import sys
import time
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_list_devices(args):
    """Listar dispositivos de audio disponibles"""
    import sounddevice as sd
    
    print("\n🎤 Dispositivos de ENTRADA (Micrófonos):")
    print("=" * 60)
    
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            marker = "→" if i == sd.default.device[0] else " "
            print(f"  {marker} [{i}] {dev['name']}")
            print(f"        Canales: {dev['max_input_channels']}, Sample Rate: {int(dev['default_samplerate'])} Hz")
    
    print("\n🔊 Dispositivos de SALIDA (Parlantes):")
    print("=" * 60)
    
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_output_channels'] > 0:
            marker = "→" if i == sd.default.device[1] else " "
            print(f"  {marker} [{i}] {dev['name']}")
            print(f"        Canales: {dev['max_output_channels']}, Sample Rate: {int(dev['default_samplerate'])} Hz")
    
    print("\n  → = dispositivo por defecto")
    return 0


def cmd_test_mic(args):
    """Probar un micrófono"""
    from src.audio.multi_mic_capture import MultiMicCapture
    
    print(f"\n🎤 Probando micrófono {args.device}...")
    print(f"   Duración: {args.duration} segundos")
    print("   Habla algo...")
    
    result = MultiMicCapture.test_microphone(args.device, args.duration)
    
    if result["success"]:
        print(f"\n✅ Prueba exitosa:")
        print(f"   RMS Level: {result['rms_level']:.4f}")
        print(f"   Peak Level: {result['peak_level']:.4f}")
        print(f"   Señal detectada: {'Sí' if result['has_signal'] else 'No'}")
        
        if not result['has_signal']:
            print("\n⚠️  No se detectó señal. Verifica:")
            print("   - El micrófono está conectado")
            print("   - El micrófono no está muteado")
            print("   - El dispositivo correcto está seleccionado")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return 0 if result["success"] else 1


def cmd_test_zone(args):
    """Probar una zona completa (mic -> TTS -> speaker)"""
    from src.audio.zone_manager import ZoneManager, Zone
    from src.audio.ma1260_controller import MA1260Controller, MA1260Source
    import numpy as np
    
    print(f"\n🔊 Probando zona...")
    print(f"   Micrófono: dispositivo {args.mic}")
    print(f"   MA1260 zona: {args.ma1260_zone}")
    print(f"   Modo: {'simulación' if args.simulate else 'real'}")
    
    # Crear controlador MA1260
    if args.simulate:
        ma1260 = MA1260Controller(connection_type="simulation")
    else:
        ma1260 = MA1260Controller(
            connection_type="serial",
            serial_port=args.serial_port
        )
    
    # Crear zona
    zone = Zone(
        id="test_zone",
        name="Test Zone",
        mic_device_index=args.mic,
        ma1260_zone=args.ma1260_zone,
        volume=args.volume
    )
    
    # Crear zone manager
    manager = ZoneManager(
        zones=[zone],
        ma1260_controller=ma1260
    )
    
    # Generar tono de prueba
    print("\n   Generando tono de prueba (440 Hz, 2 segundos)...")
    sample_rate = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Reproducir en zona
    print(f"   Reproduciendo en zona {args.ma1260_zone}...")
    manager.play_to_zone("test_zone", tone, sample_rate)
    
    print("\n✅ Prueba completada")
    print(f"   Estado MA1260: {ma1260.get_all_status()}")
    
    ma1260.close()
    return 0


def cmd_test_detection(args):
    """Probar detección de zona con múltiples micrófonos"""
    from src.audio.multi_mic_capture import MultiMicCapture, MicrophoneConfig
    from src.audio.zone_manager import ZoneManager, Zone
    import numpy as np
    
    print(f"\n🎤 Probando detección de zona...")
    print(f"   Micrófonos: {args.mics}")
    print(f"   Duración: {args.duration} segundos")
    
    # Crear configuración de micrófonos
    mic_configs = []
    zones = []
    for i, mic_idx in enumerate(args.mics):
        zone_id = f"zone_{i+1}"
        mic_configs.append(MicrophoneConfig(
            device_index=mic_idx,
            zone_id=zone_id
        ))
        zones.append(Zone(
            id=zone_id,
            name=f"Zone {i+1}",
            mic_device_index=mic_idx,
            ma1260_zone=i+1
        ))
    
    # Crear zone manager
    manager = ZoneManager(zones=zones)
    
    # Crear multi-mic capture
    capture = MultiMicCapture(
        microphones=mic_configs,
        vad_threshold=args.threshold
    )
    
    print("\n   Iniciando captura. Habla en diferentes habitaciones...")
    print("   Presiona Ctrl+C para terminar.\n")
    
    capture.start()
    
    try:
        start_time = time.time()
        while time.time() - start_time < args.duration:
            # Actualizar niveles en zone manager
            levels = capture.get_audio_levels()
            for zone_id, level in levels.items():
                # Simular update (normalmente vendría del callback)
                if zone_id in manager.zones:
                    manager._audio_levels[zone_id] = level
                    if level > args.threshold:
                        manager._detection_timestamps[zone_id] = time.time()
            
            # Mostrar niveles
            level_str = " | ".join([
                f"{zid}: {'█' * int(lvl * 100):<10} {lvl:.3f}"
                for zid, lvl in sorted(levels.items())
            ])
            print(f"\r   {level_str}", end="", flush=True)
            
            # Detectar zona más activa
            loudest = capture.get_loudest_zone()
            if loudest:
                zone_id, level = loudest
                print(f"\n   → Detectado en {zone_id} (nivel: {level:.3f})")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n   Detenido por usuario")
    finally:
        capture.stop()
    
    print("\n✅ Prueba completada")
    return 0


def cmd_ma1260_status(args):
    """Ver estado del MA1260"""
    from src.audio.ma1260_controller import MA1260Controller
    
    print(f"\n📊 Estado del Dayton MA1260")
    print("=" * 60)
    
    if args.simulate:
        ma1260 = MA1260Controller(connection_type="simulation")
    else:
        ma1260 = MA1260Controller(
            connection_type="serial",
            serial_port=args.serial_port
        )
    
    status = ma1260.get_all_status()
    
    print(f"\n   Conexión: {status['connection']}")
    print(f"   Zonas seleccionadas: {status['selected_zones']}")
    
    print("\n   Zonas:")
    for zone_num, zone_status in status['zones'].items():
        power = "🟢" if zone_status['power'] else "🔴"
        mute = "🔇" if zone_status['muted'] else "🔊"
        print(f"   {power} Zona {zone_num}: Vol={zone_status['volume']} {mute} Src={zone_status['source']}")
    
    ma1260.close()
    return 0


def cmd_ma1260_control(args):
    """Controlar el MA1260"""
    from src.audio.ma1260_controller import MA1260Controller, MA1260Source
    
    if args.simulate:
        ma1260 = MA1260Controller(connection_type="simulation")
    else:
        ma1260 = MA1260Controller(
            connection_type="serial",
            serial_port=args.serial_port
        )
    
    zone = args.zone
    
    if args.action == "power_on":
        ma1260.power_on(zone)
        print(f"✅ Zona {zone}: encendida")
    elif args.action == "power_off":
        ma1260.power_off(zone)
        print(f"✅ Zona {zone}: apagada")
    elif args.action == "mute":
        ma1260.mute_zone(zone)
        print(f"✅ Zona {zone}: silenciada")
    elif args.action == "unmute":
        ma1260.unmute_zone(zone)
        print(f"✅ Zona {zone}: sin silencio")
    elif args.action == "volume":
        ma1260.set_volume(zone, args.value)
        print(f"✅ Zona {zone}: volumen = {args.value}")
    elif args.action == "source":
        ma1260.set_source(zone, MA1260Source(args.value))
        print(f"✅ Zona {zone}: fuente = {args.value}")
    
    ma1260.close()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="KZA Multi-Zone Setup Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Listar dispositivos de audio
  python zone_setup.py devices

  # Probar un micrófono
  python zone_setup.py test-mic --device 2

  # Probar zona completa
  python zone_setup.py test-zone --mic 2 --ma1260-zone 1 --simulate

  # Probar detección multi-zona
  python zone_setup.py test-detection --mics 1 2 3 --duration 30

  # Ver estado del MA1260
  python zone_setup.py ma1260-status --simulate

  # Controlar MA1260
  python zone_setup.py ma1260-control --zone 1 --action volume --value 50
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comando")
    
    # devices
    subparsers.add_parser("devices", help="Listar dispositivos de audio")
    
    # test-mic
    mic_parser = subparsers.add_parser("test-mic", help="Probar micrófono")
    mic_parser.add_argument("--device", type=int, required=True, help="Índice del dispositivo")
    mic_parser.add_argument("--duration", type=float, default=3.0, help="Duración en segundos")
    
    # test-zone
    zone_parser = subparsers.add_parser("test-zone", help="Probar zona completa")
    zone_parser.add_argument("--mic", type=int, required=True, help="Índice del micrófono")
    zone_parser.add_argument("--ma1260-zone", type=int, required=True, help="Zona del MA1260 (1-6)")
    zone_parser.add_argument("--volume", type=int, default=50, help="Volumen (0-100)")
    zone_parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="Puerto serial")
    zone_parser.add_argument("--simulate", action="store_true", help="Modo simulación")
    
    # test-detection
    det_parser = subparsers.add_parser("test-detection", help="Probar detección multi-zona")
    det_parser.add_argument("--mics", type=int, nargs="+", required=True, help="Índices de micrófonos")
    det_parser.add_argument("--duration", type=float, default=60.0, help="Duración en segundos")
    det_parser.add_argument("--threshold", type=float, default=0.02, help="Umbral de detección")
    
    # ma1260-status
    status_parser = subparsers.add_parser("ma1260-status", help="Estado del MA1260")
    status_parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="Puerto serial")
    status_parser.add_argument("--simulate", action="store_true", help="Modo simulación")
    
    # ma1260-control
    ctrl_parser = subparsers.add_parser("ma1260-control", help="Controlar MA1260")
    ctrl_parser.add_argument("--zone", type=int, required=True, help="Zona (1-6, 0=todas)")
    ctrl_parser.add_argument("--action", required=True,
                            choices=["power_on", "power_off", "mute", "unmute", "volume", "source"])
    ctrl_parser.add_argument("--value", type=int, help="Valor para volume/source")
    ctrl_parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="Puerto serial")
    ctrl_parser.add_argument("--simulate", action="store_true", help="Modo simulación")
    
    args = parser.parse_args()
    
    if args.command == "devices":
        return cmd_list_devices(args)
    elif args.command == "test-mic":
        return cmd_test_mic(args)
    elif args.command == "test-zone":
        return cmd_test_zone(args)
    elif args.command == "test-detection":
        return cmd_test_detection(args)
    elif args.command == "ma1260-status":
        return cmd_ma1260_status(args)
    elif args.command == "ma1260-control":
        return cmd_ma1260_control(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
