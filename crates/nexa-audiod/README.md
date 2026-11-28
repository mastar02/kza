# nexa-audiod

Run with `--simulation --config-dir DIR` for silent capture and a simulated amplifier and
playback sink. The daemon loads `nexa.toml` and `rooms.toml` from the config directory,
which defaults to `NEXA_CONFIG_DIR` or `/etc/nexa`. `--socket PATH` overrides
`paths.runtime_dir/audio.sock`. The socket directory must be owned by the effective user
with mode `0700`. The daemon creates it if needed.

The capture policy is stored in `paths.data_dir/audiod/capture_seal.json`. Owners left by
an unclean exit carry forward. The operator must confirm the exact carried count through
`ForgetCarriedOwners` before resuming.

## Utterance capture

With an `include_ambient` subscriber, each room segments speech using the existing VAD
and its room noise floor. An ambient clip needs at least 800 ms of voiced audio. It ends
after 640 ms of quiet or 12 s of capture, with a 500 ms cooldown after silence-ended clips
and 0 ms cooldown when chaining capped 12 s clips. Ambient audio goes directly to
subscribers and carries no Parakeet transcript.

A completed wake command enables one follow-up candidate for up to 12 s in the same
room and capture session. By default Parakeet shadow STT is disabled (`parakeet_model = ""`),
and Brain routes commands and follow-ups via the shared `whisper-v3-turbo` ML worker. A new
wake interrupts ambient or follow-up segmentation. Sealing or recovering capture discards
any unfinished clips.

## Capture pipeline & Linux hardware build

Each room runs an isolated pipeline with a bounded SPSC `rtrb` ring (`24 × 80 ms` = `1.92 s`).
The native `alsa-capture` thread writes directly into the ring without per-frame heap
allocations, recovers transient `EPIPE` DMA overruns in-place (escalating to room recovery
on bursts), and tracks `capture_xruns`, `ring_overruns`, and `watchdog_reboots` in
`GetHealth.detail`. XVF3800 DoA telemetry (`AEC_AZIMUTH_VALUES`) is polled out-of-band only
when ambient demand is active and the room is idle. The dead-isochronous watchdog counts
`75` complete `80 ms` frames (`6.0 s`) of exact zeros or `3` consecutive no-audio session
failures, always stopping ALSA before issuing a rate-limited XVF3800 USB reboot (`4/h`).
Playback serializes worker thread exit per zone before reopening ALSA.

Build with `cargo build -p nexa-audiod --features hardware`. The native build uses the ALSA
and libusb development headers. At runtime the daemon needs access to the configured
USB/ALSA devices and the MA1260 TCP endpoint or serial device. The MA1260 transport assumes
a controller that implements KZA's command/ACK frames. Each playback zone `zone_<room_id>`
uses the configured room's XVF3800 playback device.

Set `ORT_DYLIB_PATH` to the CPU ONNX Runtime 1.24 shared library if it is outside the
dynamic linker's search path. Supply these model exports under `paths.models_dir`:

- `audio.silero_vad_model` points to the Silero streaming export with `input`, `state`,
  and `sr` inputs.
- `audio.wake_model` points to the openWakeWord binary classifier. Its directory also
  contains `melspectrogram.onnx` and `embedding_model.onnx`.
- `audio.parakeet_model` points to the onnx-asr Parakeet TDT v3 export directory with
  `nemo128.onnx`, `vocab.txt`, `encoder-model.onnx` and `decoder_joint-model.onnx`.
  Include the encoder's external data if using the full-precision export. The daemon
  also supports a matching pair of `*.int8.onnx` graphs.

The optional CPU smoke test expects `NEXA_TEST_ONNX_DIR` containing `silero_vad.onnx`,
the `parakeet/` directory, the wake frontend files next to the classifier selected by
`NEXA_TEST_WAKE_MODEL` (relative path), and a mono 16 kHz signed-16-bit `hello.wav`
saying “The quick brown fox jumps over the lazy dog”. Run
`cargo test -p nexa-audiod --features hardware --test onnx_assets -- --ignored`.
