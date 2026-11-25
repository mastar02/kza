#![cfg(all(target_os = "linux", feature = "hardware"))]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use nexa_audiod::dsp::FRAME_SAMPLES;
use nexa_audiod::hw::onnx::{OpenWakeWord, Parakeet, Silero, init_cpu};
use nexa_audiod::traits::{ShadowStt, VadEngine, WakeEngine};

#[test]
#[ignore = "requires the external CPU runtime, ONNX assets and a 16 kHz mono speech fixture"]
fn cpu_models_execute_with_external_onnx_assets() {
    let assets = PathBuf::from(std::env::var_os("NEXA_TEST_ONNX_DIR").unwrap());
    let wake_model = assets.join(std::env::var_os("NEXA_TEST_WAKE_MODEL").unwrap());
    init_cpu().unwrap();
    let mut wav = hound::WavReader::open(assets.join("hello.wav")).unwrap();
    assert_eq!(wav.spec().sample_rate, 16_000);
    assert_eq!(wav.spec().channels, 1);
    let samples: Vec<_> = wav
        .samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32768.0)
        .collect();

    let mut vad = Silero::open(&assets.join("silero_vad.onnx")).unwrap();
    let peak = samples
        .chunks(FRAME_SAMPLES)
        .map(|frame| vad.speech_probability(frame).unwrap())
        .fold(0.0, f32::max);
    assert!(peak > 0.5, "Silero peak on speech: {peak}");
    vad.reset();
    assert!(vad.speech_probability(&[0.0; FRAME_SAMPLES]).unwrap() < 0.5);

    let mut wake = OpenWakeWord::open(&wake_model).unwrap();
    for _ in 0..40 {
        assert!(wake.wake_score(&[0.0; FRAME_SAMPLES]).unwrap() < 0.4);
    }
    let mut recognizer = Parakeet::open(&assets.join("parakeet")).unwrap();
    let transcript = recognizer.transcribe(&samples).unwrap().to_lowercase();
    assert!(
        transcript.contains("fox") && transcript.contains("dog"),
        "{transcript}"
    );
}
