use std::path::Path;

use hound::{SampleFormat, WavReader};
use nexa_audiod::dsp::SAMPLE_RATE;

pub fn wav(name: &str) -> Vec<f32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests-v2/fixtures/audio")
        .join(name);
    let mut wav = WavReader::open(path).unwrap();
    let spec = wav.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, SAMPLE_RATE);
    assert_eq!(spec.bits_per_sample, 16);
    assert_eq!(spec.sample_format, SampleFormat::Int);
    wav.samples::<i16>()
        .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
        .collect()
}
