# Multi-Interface Integration Design (Mic + BT per Room)

**Date:** 2026-02-21
**Status:** Approved
**Approach:** A — MultiRoomAudioLoop

## Problem

The system has well-implemented individual components (ZoneManager, BLEScanner, PresenceDetector, RoomContextManager) but they are not wired together. Audio capture uses a single `sd.InputStream`, room context is not passed through the pipeline, and TTS responses are not routed to the correct zone.

## Physical Topology

Each room has exactly:
- 1x ReSpeaker XMOS XVF3800 (mic, USB over Cat5e extender)
- 1x BLE dongle 5.3 (presence, USB over Cat5e extender)
- 1x MA1260 zone output (stereo or mono)

| Room | Mic | BT | MA1260 Zone | Output |
|------|-----|----|-------------|--------|
| Living | XVF3800 (dev:1) | hci0 | Zone 1 | Stereo |
| Pasillo/Hall | XVF3800 (dev:2) | hci1 | Zone 2 | Mono |
| Cocina | XVF3800 (dev:3) | hci2 | Zone 3 | TBD |
| Escritorio | XVF3800 (dev:4) | hci3 | Zone 4 | Mono |
| Baño | XVF3800 (dev:5) | hci4 | Zone 5 | Stereo |

## Architecture

```
┌─────────────────── MultiRoomAudioLoop ──────────────────┐
│                                                          │
│  RoomStream[living]    ──┐                               │
│  RoomStream[cocina]    ──┤   ┌────────────────┐          │
│  RoomStream[escritorio]──┼──→│ CommandEvent    │          │
│  RoomStream[hall]      ──┤   │  .audio         │         │
│  RoomStream[bano]      ──┘   │  .room_id       │         │
│                              └───────┬────────┘          │
│  Dedup: <200ms between rooms = echo  │                   │
│         >200ms = concurrent commands │                   │
└──────────────────────────────────────┼───────────────────┘
                                       │
              asyncio.create_task() ←──┘  (parallel per room)
                                       │
                                       ▼
  CommandProcessor.process(audio) → (text, user_id, emotion)
                                       │
                                       ▼
  RoomContextManager.resolve_room(room_id, user_id)
    → RoomContext(room, confidence, entities)
    → Cross-validates with PresenceDetector (BLE)
                                       │
                                       ▼
  RequestRouter.route(text, room_context)
    → entity = room_context.entities["light"]  → "light.cocina"
    → ha_client.call_service(...)
                                       │
                                       ▼
  ResponseHandler.respond(text, room_context)
    → TTS(GPU3) → ZoneManager.play_to_zone(room_context.ma1260_zone)
```

## Key Design Decisions

### 1. Five parallel audio streams
Each XVF3800 gets its own `sd.InputStream` with independent wake word detection and echo suppression. When a stream captures a command, it fires a `CommandEvent` with `room_id` attached.

### 2. Concurrent multi-room processing
Two people speaking in different rooms at the same time both get processed. Each command is dispatched as a separate `asyncio.Task`. The `MultiUserOrchestrator` already supports parallel requests via `PriorityRequestQueue`.

### 3. Echo deduplication
If two rooms detect wake word within 200ms, it's likely acoustic echo between rooms. The system keeps the detection with higher RMS and discards the other. If >200ms apart, both are independent commands.

### 4. RoomContext flows through the pipeline
`RoomContext` is passed as a parameter (not global state) through the processing chain. Each concurrent task has its own `RoomContext`.

### 5. Unified config
The `zones` section is absorbed into `rooms`. Each room definition includes mic_device_index, bt_adapter, ma1260_zone, and output_mode (stereo/mono) alongside HA entities.

### 6. TTS zone routing
`ResponseHandler` uses `RoomContext.ma1260_zone` to route TTS output to the correct speaker. Broadcast commands (e.g., "announce dinner in the living room") target specific zones via the existing `IntercomSystem`.

## Data Structures

```python
@dataclass
class RoomStream:
    room_id: str
    device_index: int
    wake_detector: WakeWordDetector
    echo_suppressor: EchoSuppressor
    listening: bool = False
    audio_buffer: list = field(default_factory=list)
    command_start_time: float = 0.0

@dataclass
class CommandEvent:
    audio: np.ndarray
    room_id: str
    mic_device_index: int
    timestamp: float
```

## Room Context Resolution (existing, now integrated)

Priority chain:
1. Spoken room ("turn off the kitchen light") → confidence 1.0
2. Mic + BT match (mic=cocina, BLE=user in cocina) → confidence 1.0
3. Mic only (wake word in cocina) → confidence 0.7
4. BT only (user's phone in cocina) → confidence 0.6
5. Last known room (<5 min) → confidence 0.4
6. Fallback room (configured) → confidence 0.2

## Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `src/pipeline/multi_room_audio_loop.py` | NEW | Replaces AudioLoop with N parallel streams |
| `src/pipeline/voice_pipeline.py` | MODIFY | Accept MultiRoomAudioLoop, process CommandEvent |
| `src/pipeline/audio_manager.py` | MODIFY | Simplify (zone detection moves to MultiRoomAudioLoop) |
| `src/pipeline/request_router.py` | MODIFY | process_command accepts RoomContext parameter |
| `src/pipeline/response_handler.py` | MODIFY | speak/respond use RoomContext for zone routing |
| `src/main.py` | MODIFY | Instantiate RoomContextManager, PresenceDetector, build room_streams |
| `config/settings.yaml` | MODIFY | Unify zones into rooms, add ma1260_zone and output_mode |
| `tests/unit/pipeline/test_multi_room_audio_loop.py` | NEW | Unit tests for MultiRoomAudioLoop |
| `tests/unit/pipeline/test_concurrent_commands.py` | NEW | Integration tests for multi-room concurrency |

## Signature Changes

```python
# Before
VoicePipeline.process_command(audio: np.ndarray) -> dict
RequestRouter.process_command(audio: np.ndarray) -> dict
ResponseHandler.speak(text: str) -> None

# After
VoicePipeline.process_command(event: CommandEvent) -> dict
RequestRouter.process_command(audio: np.ndarray, room_context: RoomContext) -> dict
ResponseHandler.speak(text: str, room_context: RoomContext) -> None
```

## Resource Estimates

- 5x WakeWordDetector (openwakeword, CPU): ~250MB total RAM
- 5x sd.InputStream: negligible (kernel-level audio capture)
- 5x EchoSuppressor: ~5MB total RAM
- Concurrent processing: GPU resources are already partitioned (STT on GPU0, SpeakerID on GPU1, etc.) — concurrent commands queue at the GPU level
