//! Silent, paced capture and a playback sink for portable daemon integration.

use std::time::Duration;

use crate::traits::{
    AudioError, CaptureSource, PlaybackOutput, PlaybackSession, PlaybackSpec, ShadowStt, VadEngine,
    WakeEngine,
};

#[derive(Debug, Default)]
pub struct Simulation;

impl CaptureSource for Simulation {
    fn device(&self) -> (&str, u32) {
        ("simulation", 1)
    }

    async fn start(&mut self) -> Result<(), AudioError> {
        Ok(())
    }

    async fn read(&mut self, samples: &mut [f32]) -> Result<usize, AudioError> {
        tokio::time::sleep(crate::dsp::frame_duration(samples.len())).await;
        samples.fill(0.0);
        Ok(samples.len())
    }

    async fn stop(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}

impl VadEngine for Simulation {
    fn speech_probability(&mut self, _: &[f32]) -> Result<f32, AudioError> {
        Ok(0.0)
    }
}

impl WakeEngine for Simulation {
    fn wake_score(&mut self, _: &[f32]) -> Result<f32, AudioError> {
        Ok(0.0)
    }
}

impl ShadowStt for Simulation {
    fn transcribe(&mut self, _: &[f32]) -> Result<String, AudioError> {
        Ok(String::new())
    }
}

#[derive(Debug)]
pub struct SimulatedPlayback(u32);

impl PlaybackOutput for Simulation {
    type Session = SimulatedPlayback;

    async fn open(&self, spec: &PlaybackSpec) -> Result<Self::Session, AudioError> {
        Ok(SimulatedPlayback(spec.sample_rate))
    }
}

impl PlaybackSession for SimulatedPlayback {
    fn card(&self) -> &str {
        "simulation"
    }

    async fn write(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        tokio::time::sleep(Duration::from_secs_f64(
            samples.len() as f64 / f64::from(self.0),
        ))
        .await;
        Ok(())
    }

    async fn finish(&mut self) -> Result<(), AudioError> {
        Ok(())
    }
}
