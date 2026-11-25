//! CPU-only ONNX Runtime adapters. Model artifacts are supplied by the installation.

mod parakeet;
mod wake;

use std::path::Path;

use ort::ep::CPU;
use ort::session::{Session, SessionOutputs};
use ort::value::{DynValue, Shape, Tensor, TensorRef, ValueType};

use crate::dsp::SAMPLE_RATE;
use crate::traits::{AudioError, VadEngine};

pub use parakeet::Parakeet;
pub use wake::OpenWakeWord;

pub fn init_cpu() -> Result<(), AudioError> {
    let library = std::env::var_os("ORT_DYLIB_PATH").unwrap_or_else(|| "libonnxruntime.so".into());
    ort::init_from(library)
        .map_err(adapter)?
        .with_execution_providers([CPU::default().build().error_on_failure()])
        .commit();
    Ok(())
}

fn session(path: &Path) -> Result<Session, AudioError> {
    let open = || {
        Session::builder()
            .map_err(adapter)?
            .with_execution_providers([CPU::default().build().error_on_failure()])
            .map_err(adapter)?
            .with_intra_threads(1)
            .map_err(adapter)?
            .with_inter_threads(1)
            .map_err(adapter)?
            .commit_from_file(path)
            .map_err(adapter)
    };
    open().map_err(|error| adapter(format_args!("model {}: {error}", path.display())))
}

fn adapter(error: impl std::fmt::Display) -> AudioError {
    AudioError::Adapter(error.to_string())
}

fn contract(name: &str) -> AudioError {
    adapter(format_args!("unexpected ONNX tensor for {name}"))
}

fn input<'a>(session: &'a Session, name: &str) -> Result<&'a ValueType, AudioError> {
    session
        .inputs()
        .iter()
        .find(|value| value.name() == name)
        .map(|value| value.dtype())
        .ok_or_else(|| contract(name))
}

fn output<'a>(values: &'a SessionOutputs<'_>, name: &str) -> Result<&'a DynValue, AudioError> {
    values.get(name).ok_or_else(|| contract(name))
}

fn single_output<'a>(values: &'a SessionOutputs<'_>) -> Result<&'a DynValue, AudioError> {
    if values.len() != 1 {
        return Err(contract("single output"));
    }
    values
        .keys()
        .next()
        .and_then(|name| values.get(name))
        .ok_or_else(|| contract("single output"))
}

fn float(value: &DynValue) -> Result<(&Shape, &[f32]), AudioError> {
    value.try_extract_tensor::<f32>().map_err(adapter)
}

const SILERO_CHUNK: usize = 512;
const SILERO_CONTEXT: usize = 64;

#[derive(Debug)]
pub struct Silero {
    model: Session,
    state: [f32; 256],
    context: [f32; SILERO_CONTEXT],
    pending: Vec<f32>,
    scalar_rate: bool,
}

impl Silero {
    /// Open the Silero v5+ streaming export with input/state/sr and output/stateN.
    pub fn open(path: &Path) -> Result<Self, AudioError> {
        let model = session(path)?;
        input(&model, "input")?;
        input(&model, "state")?;
        let rate = input(&model, "sr")?
            .tensor_shape()
            .ok_or_else(|| contract("sr"))?;
        let scalar_rate = match rate.as_ref() {
            [] => true,
            [1 | -1] => false,
            _ => return Err(contract("sr")),
        };
        Ok(Self {
            model,
            state: [0.0; 256],
            context: [0.0; SILERO_CONTEXT],
            pending: Vec::new(),
            scalar_rate,
        })
    }
}

impl VadEngine for Silero {
    fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
        self.pending.clear();
    }

    fn speech_probability(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        self.pending.extend_from_slice(samples);
        let mut peak: f32 = 0.0;
        let (chunks, remaining) = self.pending.as_chunks::<SILERO_CHUNK>();
        for chunk in chunks {
            let mut audio = [0.0; SILERO_CHUNK + SILERO_CONTEXT];
            audio[..SILERO_CONTEXT].copy_from_slice(&self.context);
            audio[SILERO_CONTEXT..].copy_from_slice(chunk);
            let rate_shape: &[usize] = if self.scalar_rate { &[] } else { &[1] };
            let output_values = self.model.run(ort::inputs! {
                "input" => TensorRef::from_array_view(([1, audio.len()], audio.as_slice())).map_err(adapter)?,
                "state" => TensorRef::from_array_view(([2, 1, 128], self.state.as_slice())).map_err(adapter)?,
                "sr" => Tensor::from_array((rate_shape, vec![i64::from(SAMPLE_RATE)])).map_err(adapter)?,
            }).map_err(adapter)?;
            let (_, state) = float(output(&output_values, "stateN")?)?;
            let state: [f32; 256] = state.try_into().map_err(|_| contract("stateN"))?;
            let (_, score) = float(output(&output_values, "output")?)?;
            let &[score] = score else {
                return Err(contract("output"));
            };
            if !(0.0..=1.0).contains(&score) {
                return Err(contract("output"));
            }
            peak = peak.max(score);
            self.state = state;
            self.context
                .copy_from_slice(&chunk[SILERO_CHUNK - SILERO_CONTEXT..]);
        }
        let consumed = self.pending.len() - remaining.len();
        self.pending.drain(..consumed);
        Ok(peak)
    }
}
