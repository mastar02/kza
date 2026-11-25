//! Parakeet TDT v3 compatible with the onnx-asr NeMo export.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorElementType, TensorRef, ValueType};

use super::{adapter, contract, float, input, output, session};
use crate::traits::{AudioError, ShadowStt};

const MAX_TOKENS_PER_STEP: usize = 10;
const TDT_DURATIONS: usize = 5; // The v3 export uses durations 0 through 4.

#[derive(Debug)]
pub struct Parakeet {
    features: Session,
    encoder: Session,
    decoder: Session,
    vocabulary: Vec<String>,
    blank: usize,
    state_shapes: [[usize; 3]; 2],
    targets: TensorElementType,
    target_length: TensorElementType,
}

impl Parakeet {
    /// Supply the NeMo 128-bin preprocessor, encoder, decoder-joint and vocabulary
    /// in the configured model directory.
    pub fn open(dir: &Path) -> Result<Self, AudioError> {
        let features = session(&dir.join("nemo128.onnx"))?;
        let full_precision = dir.join("encoder-model.onnx").is_file()
            && dir.join("decoder_joint-model.onnx").is_file();
        let suffix = if full_precision {
            ".onnx"
        } else {
            ".int8.onnx"
        };
        let encoder = session(&dir.join(format!("encoder-model{suffix}")))?;
        let decoder = session(&dir.join(format!("decoder_joint-model{suffix}")))?;
        let vocabulary = read_vocab(&dir.join("vocab.txt"))?;
        let blank = vocabulary
            .iter()
            .position(|word| word == "<blk>")
            .ok_or_else(|| contract("Parakeet blank token"))?;
        let state_shapes = [
            state_shape(&decoder, "input_states_1")?,
            state_shape(&decoder, "input_states_2")?,
        ];
        let targets = int_type(&decoder, "targets")?;
        let target_length = int_type(&decoder, "target_length")?;
        Ok(Self {
            features,
            encoder,
            decoder,
            vocabulary,
            blank,
            state_shapes,
            targets,
            target_length,
        })
    }
}

impl ShadowStt for Parakeet {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, AudioError> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let features = self.features.run(ort::inputs! {
            "waveforms" => TensorRef::from_array_view(([1, samples.len()], samples)).map_err(adapter)?,
            "waveforms_lens" => Tensor::from_array(([1], vec![samples.len() as i64])).map_err(adapter)?,
        }).map_err(adapter)?;
        let encoded = self
            .encoder
            .run(ort::inputs! {
                "audio_signal" => output(&features, "features")?,
                "length" => output(&features, "features_lens")?,
            })
            .map_err(adapter)?;
        let (shape, values) = float(output(&encoded, "outputs")?)?;
        let &[1, width, padded_frames] = shape.as_ref() else {
            return Err(contract("Parakeet encoder outputs"));
        };
        let (width, padded_frames) = (
            usize::try_from(width).map_err(adapter)?,
            usize::try_from(padded_frames).map_err(adapter)?,
        );
        if padded_frames == 0 || width == 0 {
            return Err(contract("Parakeet encoder outputs"));
        }
        let (_, lengths) = output(&encoded, "encoded_lengths")?
            .try_extract_tensor::<i64>()
            .map_err(adapter)?;
        let &[length] = lengths else {
            return Err(contract("encoded_lengths"));
        };
        let length = usize::try_from(length)
            .ok()
            .filter(|&length| length <= padded_frames)
            .ok_or_else(|| contract("encoded_lengths"))?;

        let mut states = self
            .state_shapes
            .map(|shape| vec![0.0f32; shape.iter().product()]);
        let mut text = String::new();
        let mut last_token = self.blank;
        let mut at = 0;
        let mut symbols = 0;
        let mut encoding = vec![0.0; width];
        while at < length {
            for (sample, channel) in encoding.iter_mut().zip(values.chunks_exact(padded_frames)) {
                *sample = channel[at];
            }
            let decoded = self.decoder.run(ort::inputs! {
                "encoder_outputs" => TensorRef::from_array_view(([1, width, 1], encoding.as_slice())).map_err(adapter)?,
                "targets" => integer(self.targets, [1, 1], last_token)?,
                "target_length" => integer(self.target_length, [1], 1)?,
                "input_states_1" => TensorRef::from_array_view((self.state_shapes[0], states[0].as_slice())).map_err(adapter)?,
                "input_states_2" => TensorRef::from_array_view((self.state_shapes[1], states[1].as_slice())).map_err(adapter)?,
            }).map_err(adapter)?;
            let (_, logits) = float(output(&decoded, "outputs")?)?;
            if logits.len() != self.vocabulary.len() + TDT_DURATIONS {
                return Err(contract("Parakeet decoder logits"));
            }
            let (tokens, durations) = logits.split_at(self.vocabulary.len());
            let token = argmax(tokens)?;
            let duration = argmax(durations)?;
            if token != self.blank {
                last_token = token;
                symbols += 1;
                text.push_str(&self.vocabulary[token]);
                for (state, name) in states
                    .iter_mut()
                    .zip(["output_states_1", "output_states_2"])
                {
                    let (_, data) = float(output(&decoded, name)?)?;
                    if data.len() != state.len() {
                        return Err(contract(name));
                    }
                    state.copy_from_slice(data);
                }
            }
            if duration > 0 || token == self.blank || symbols == MAX_TOKENS_PER_STEP {
                at += duration.max(1);
                symbols = 0;
            }
        }
        Ok(text.trim().to_owned())
    }
}

fn read_vocab(path: &Path) -> Result<Vec<String>, AudioError> {
    let mut tokens = BTreeMap::new();
    for line in fs::read_to_string(path)?.lines() {
        let (token, id) = line.rsplit_once(' ').ok_or_else(|| contract("vocab.txt"))?;
        let id = id.parse::<usize>().map_err(adapter)?;
        if tokens.insert(id, token.replace('▁', " ")).is_some() {
            return Err(contract("duplicate vocabulary ID"));
        }
    }
    if tokens.keys().copied().ne(0..tokens.len()) {
        return Err(contract("noncontiguous vocabulary IDs"));
    }
    Ok(tokens.into_values().collect())
}

fn state_shape(session: &Session, name: &str) -> Result<[usize; 3], AudioError> {
    let shape = input(session, name)?
        .tensor_shape()
        .ok_or_else(|| contract(name))?;
    match *shape.as_ref() {
        [layers, 1 | -1, width] if (1..=16).contains(&layers) && (1..=4096).contains(&width) => {
            Ok([layers as usize, 1, width as usize])
        }
        _ => Err(contract(name)),
    }
}

fn int_type(session: &Session, name: &str) -> Result<TensorElementType, AudioError> {
    match input(session, name)? {
        ValueType::Tensor {
            ty: ty @ (TensorElementType::Int32 | TensorElementType::Int64),
            ..
        } => Ok(*ty),
        _ => Err(contract(name)),
    }
}

fn integer<const N: usize>(
    kind: TensorElementType,
    shape: [usize; N],
    value: usize,
) -> Result<DynValue, AudioError> {
    match kind {
        TensorElementType::Int32 => {
            Tensor::from_array((shape, vec![i32::try_from(value).map_err(adapter)?]))
                .map(Tensor::into_dyn)
                .map_err(adapter)
        }
        TensorElementType::Int64 => {
            Tensor::from_array((shape, vec![i64::try_from(value).map_err(adapter)?]))
                .map(Tensor::into_dyn)
                .map_err(adapter)
        }
        _ => Err(contract("integer input")),
    }
}

fn argmax(values: &[f32]) -> Result<usize, AudioError> {
    let mut maximum = f32::NEG_INFINITY;
    let mut chosen = None;
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return Err(contract("NaN in Parakeet logits"));
        }
        if value > maximum {
            maximum = value;
            chosen = Some(index);
        }
    }
    chosen.ok_or_else(|| contract("empty Parakeet logits"))
}
