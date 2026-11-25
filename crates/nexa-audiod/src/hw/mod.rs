//! Native audio adapters and their portable wire protocols.

pub mod amp;
pub mod simulation;
pub mod sysfs;
pub mod xvf;

#[cfg(all(target_os = "linux", feature = "hardware"))]
pub mod alsa;
#[cfg(all(target_os = "linux", feature = "hardware"))]
pub mod onnx;
