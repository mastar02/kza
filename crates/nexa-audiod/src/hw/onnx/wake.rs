//! Streaming openWakeWord frontend with one embedding history per room.

use std::collections::VecDeque;
use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use super::{adapter, contract, float, session, single_output};
use crate::dsp::FRAME_SAMPLES;
use crate::traits::{AudioError, WakeEngine};

const PCM_OVERLAP: usize = 480;
const MEL_FRAMES: usize = 76;
const MEL_BINS: usize = 32;
const EMBEDDING_SIZE: usize = 96;

#[derive(Debug)]
pub struct OpenWakeWord {
    mel: Session,
    embedding: Session,
    classifier: Session,
    classifier_input: String,
    window: usize,
    overlap: [f32; PCM_OVERLAP],
    mel_history: VecDeque<f32>,
    embedding_history: VecDeque<f32>,
}

impl OpenWakeWord {
    /// Open a binary wake classifier and its standard sibling ONNX frontends.
    pub fn open(path: &Path) -> Result<Self, AudioError> {
        let mel = session(&path.with_file_name("melspectrogram.onnx"))?;
        let embedding = session(&path.with_file_name("embedding_model.onnx"))?;
        let classifier = session(path)?;
        let [input] = classifier.inputs() else {
            return Err(contract("wake classifier input"));
        };
        let dimensions = input
            .dtype()
            .tensor_shape()
            .ok_or_else(|| contract("wake classifier input"))?;
        let &[1 | -1, frames, 96] = dimensions.as_ref() else {
            return Err(contract("wake classifier input"));
        };
        let window = usize::try_from(frames)
            .ok()
            .filter(|frames| (1..=120).contains(frames))
            .ok_or_else(|| contract("wake classifier window"))?;
        let classifier_input = input.name().to_owned();
        let mut model = Self {
            mel,
            embedding,
            classifier,
            classifier_input,
            window,
            overlap: [0.0; PCM_OVERLAP],
            mel_history: VecDeque::new(),
            embedding_history: VecDeque::new(),
        };
        model.reset();
        Ok(model)
    }
}

impl WakeEngine for OpenWakeWord {
    fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.mel_history.clear();
        self.mel_history.resize(MEL_FRAMES * MEL_BINS, 1.0);
        self.embedding_history.clear();
    }

    fn wake_score(&mut self, samples: &[f32]) -> Result<f32, AudioError> {
        if samples.len() != FRAME_SAMPLES {
            return Err(AudioError::Rejected);
        }
        let mut pcm = [0.0; PCM_OVERLAP + FRAME_SAMPLES];
        pcm[..PCM_OVERLAP].copy_from_slice(&self.overlap);
        for (converted, &sample) in pcm[PCM_OVERLAP..].iter_mut().zip(samples) {
            *converted = (sample.clamp(-1.0, 1.0) * 32_767.0).trunc();
        }
        let mel_outputs = self.mel.run(ort::inputs! {
            "input" => TensorRef::from_array_view(([1, pcm.len()], pcm.as_slice())).map_err(adapter)?,
        }).map_err(adapter)?;
        let (shape, mel) = float(single_output(&mel_outputs)?)?;
        if shape.last() != Some(&(MEL_BINS as i64))
            || mel.is_empty()
            || mel.len() > MEL_FRAMES * MEL_BINS
        {
            return Err(contract("wake mel features"));
        }
        self.overlap
            .copy_from_slice(&pcm[pcm.len() - PCM_OVERLAP..]);
        self.mel_history.drain(..mel.len());
        self.mel_history
            .extend(mel.iter().map(|sample| sample / 10.0 + 2.0));
        let features = self.mel_history.make_contiguous();
        let embedding_outputs = self
            .embedding
            .run(ort::inputs! {
                "input_1" => TensorRef::from_array_view(([1, MEL_FRAMES, MEL_BINS, 1], &*features))
                    .map_err(adapter)?,
            })
            .map_err(adapter)?;
        let (_, embedding) = float(single_output(&embedding_outputs)?)?;
        if embedding.len() != EMBEDDING_SIZE {
            return Err(contract("wake embedding"));
        }
        let required = self.window * EMBEDDING_SIZE;
        if self.embedding_history.len() == required {
            self.embedding_history.drain(..EMBEDDING_SIZE);
        }
        self.embedding_history.extend(embedding);
        if self.embedding_history.len() < required {
            return Ok(0.0);
        }
        let history = self.embedding_history.make_contiguous();
        let prediction = self.classifier.run(ort::inputs! {
            self.classifier_input.as_str() => TensorRef::from_array_view(([1, self.window, EMBEDDING_SIZE], &*history))
                .map_err(adapter)?,
        }).map_err(adapter)?;
        let (_, scores) = float(single_output(&prediction)?)?;
        match scores {
            &[score] if (0.0..=1.0).contains(&score) => Ok(score),
            _ => Err(contract("wake classifier score")),
        }
    }
}
