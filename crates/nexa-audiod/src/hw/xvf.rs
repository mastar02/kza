//! ReSpeaker XVF3800 host-control frames and Linux USB transport.

use nexa_config::XvfTuning;

pub const INTERFACE: u8 = 3;
const REQUEST: u8 = 0;
const WRITE_VENDOR: u8 = 0x40;
const READ_VENDOR: u8 = 0xc0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlSetup {
    pub request_type: u8,
    pub request: u8,
    pub value: u16,
    pub index: u16,
}

impl ControlSetup {
    #[must_use]
    pub fn write(resource: u8, command: u8) -> Self {
        Self {
            request_type: WRITE_VENDOR,
            request: REQUEST,
            value: u16::from(command),
            index: u16::from(resource),
        }
    }

    #[must_use]
    pub fn read(resource: u8, command: u8) -> Self {
        Self {
            request_type: READ_VENDOR,
            request: REQUEST,
            value: 0x80 | u16::from(command),
            index: u16::from(resource),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlWrite<const N: usize> {
    pub setup: ControlSetup,
    pub payload: [u8; N],
}

pub fn tuning_frames(tuning: &XvfTuning) -> [ControlWrite<4>; 4] {
    [
        (11, tuning.pp_agc_max_gain.to_le_bytes()),
        (32, i32::from(tuning.pp_attns_mode).to_le_bytes()),
        (34, tuning.pp_attns_slope.to_le_bytes()),
        (22, tuning.pp_min_nn.to_le_bytes()),
    ]
    .map(|(command, payload)| ControlWrite {
        setup: ControlSetup::write(17, command),
        payload,
    })
}

pub fn reboot_frame() -> ControlWrite<1> {
    ControlWrite {
        setup: ControlSetup::write(48, 7),
        payload: [1],
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DspTelemetry {
    pub spenergy: [f32; 4],
    pub azimuth_rad: [f32; 4],
    pub auto_beam_rad: Option<f32>,
}

#[must_use]
pub fn spenergy_read_setup() -> ControlSetup {
    ControlSetup::read(33, 80)
}

#[must_use]
pub fn azimuth_read_setup() -> ControlSetup {
    ControlSetup::read(33, 75)
}

/// Parse 4 little-endian IEEE-754 floats from a 16-byte XVF3800 control response.
#[must_use]
pub fn parse_f32x4(raw: &[u8]) -> Option<[f32; 4]> {
    if raw.len() < 16 {
        return None;
    }
    let mut out = [0.0_f32; 4];
    for (idx, slot) in out.iter_mut().enumerate() {
        let bytes: [u8; 4] = raw[idx * 4..(idx + 1) * 4].try_into().ok()?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            return None;
        }
        *slot = value;
    }
    Some(out)
}

/// Extract auto-select beam azimuth (index 3, radians) from `AEC_AZIMUTH_VALUES`.
#[must_use]
pub fn parse_auto_beam_f32(raw: &[u8]) -> Option<f32> {
    parse_f32x4(raw).map(|beams| beams[3])
}

#[cfg(all(target_os = "linux", feature = "hardware"))]
mod native {
    use std::time::Duration;

    use rusb::DeviceHandle;

    use super::{
        ControlSetup, DspTelemetry, INTERFACE, azimuth_read_setup, parse_f32x4, reboot_frame,
        spenergy_read_setup, tuning_frames,
    };
    use crate::hw::sysfs::parse_usb_port;
    use crate::traits::{AudioError, XvfTransport};

    const XVF_VID: u16 = 0x2886;
    const XVF_PID: u16 = 0x001a;
    const TIMEOUT: Duration = Duration::from_millis(250);
    const SERVICER_COMMAND_RETRY: u8 = 64;
    const READ_RETRIES: usize = 5;

    #[derive(Debug, Clone)]
    pub struct XvfUsb {
        port: String,
    }

    impl XvfUsb {
        pub fn new(port: String) -> Self {
            Self { port }
        }

        fn open(&self) -> Result<DeviceHandle<rusb::GlobalContext>, AudioError> {
            let (bus, ports) = parse_usb_port(&self.port).ok_or(AudioError::Rejected)?;
            for device in rusb::devices().map_err(usb_error)?.iter() {
                if device.bus_number() != bus
                    || !device.port_numbers().is_ok_and(|chain| chain == ports)
                {
                    continue;
                }
                let descriptor = device.device_descriptor().map_err(usb_error)?;
                if descriptor.vendor_id() == XVF_VID && descriptor.product_id() == XVF_PID {
                    let handle = device.open().map_err(usb_error)?;
                    handle.claim_interface(INTERFACE).map_err(usb_error)?;
                    return Ok(handle);
                }
            }
            Err(AudioError::Unavailable)
        }

        pub(crate) fn apply(&self, tuning: &nexa_config::XvfTuning) -> Result<(), AudioError> {
            if !(1.0..=1000.0).contains(&tuning.pp_agc_max_gain)
                || tuning.pp_attns_mode > 1
                || !(0.0..=5.0).contains(&tuning.pp_attns_slope)
                || !(0.0..=1.0).contains(&tuning.pp_min_nn)
            {
                return Err(AudioError::Rejected);
            }
            let handle = self.open()?;
            for frame in tuning_frames(tuning) {
                write(&handle, frame.setup, &frame.payload)?;
            }
            let readback = read_with_retry(&handle, ControlSetup::read(17, 11), 4)?;
            let actual = f32::from_le_bytes(
                readback
                    .as_slice()
                    .try_into()
                    .map_err(|_| AudioError::Rejected)?,
            );
            if !actual.is_finite() || (actual - tuning.pp_agc_max_gain).abs() > 0.05 {
                return Err(AudioError::Adapter(format!(
                    "XVF3800 PP_AGCMAXGAIN readback mismatch: expected {}, got {actual}",
                    tuning.pp_agc_max_gain
                )));
            }
            Ok(())
        }

        pub(crate) fn read_dsp_telemetry(&self) -> Option<DspTelemetry> {
            let handle = self.open().ok()?;
            let sp_raw = read_with_retry(&handle, spenergy_read_setup(), 16).ok()?;
            let az_raw = read_with_retry(&handle, azimuth_read_setup(), 16).ok()?;
            let spenergy = parse_f32x4(&sp_raw)?;
            let azimuth_rad = parse_f32x4(&az_raw)?;
            Some(DspTelemetry {
                spenergy,
                azimuth_rad,
                auto_beam_rad: Some(azimuth_rad[3]),
            })
        }

        /// Deliberately reboot only the device at this port. The next capture start
        /// resolves the re-enumerated device and reapplies room tuning.
        pub async fn reboot(&self) -> Result<(), AudioError> {
            let device = self.clone();
            tokio::task::spawn_blocking(move || {
                let frame = reboot_frame();
                write(&device.open()?, frame.setup, &frame.payload)
            })
            .await
            .map_err(|_| AudioError::Unavailable)?
        }
    }

    impl XvfTransport for XvfUsb {
        async fn exchange(
            &mut self,
            resource: u8,
            command: u8,
            request: &[u8],
            response: &mut [u8],
        ) -> Result<usize, AudioError> {
            // The generic exchange must never send a reboot or a flash command.
            if !request.is_empty() && (resource != 17 || ![11, 22, 32, 34].contains(&command)) {
                return Err(AudioError::Rejected);
            }
            if command >= 0x80 {
                return Err(AudioError::Rejected);
            }
            let device = self.clone();
            let request = request.to_vec();
            let size = response.len();
            let data = tokio::task::spawn_blocking(move || {
                let handle = device.open()?;
                if !request.is_empty() {
                    write(&handle, ControlSetup::write(resource, command), &request)?;
                }
                if size == 0 {
                    return Ok(Vec::new());
                }
                read_with_retry(&handle, ControlSetup::read(resource, command), size)
            })
            .await
            .map_err(|_| AudioError::Unavailable)??;
            response.copy_from_slice(&data);
            Ok(data.len())
        }
    }

    fn read_with_retry(
        handle: &DeviceHandle<rusb::GlobalContext>,
        setup: ControlSetup,
        size: usize,
    ) -> Result<Vec<u8>, AudioError> {
        let mut data = vec![0; size + 1];
        for attempt in 0..READ_RETRIES {
            let length = handle
                .read_control(
                    setup.request_type,
                    setup.request,
                    setup.value,
                    setup.index,
                    &mut data,
                    TIMEOUT,
                )
                .map_err(usb_error)?;
            if length == data.len() && data[0] == 0 {
                return Ok(data.split_off(1));
            }
            if length >= 1 && data[0] == SERVICER_COMMAND_RETRY && attempt + 1 < READ_RETRIES {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            return Err(AudioError::Rejected);
        }
        Err(AudioError::Unavailable)
    }

    fn write(
        handle: &DeviceHandle<rusb::GlobalContext>,
        setup: ControlSetup,
        data: &[u8],
    ) -> Result<(), AudioError> {
        let count = handle
            .write_control(
                setup.request_type,
                setup.request,
                setup.value,
                setup.index,
                data,
                TIMEOUT,
            )
            .map_err(usb_error)?;
        if count != data.len() {
            return Err(AudioError::Rejected);
        }
        Ok(())
    }

    fn usb_error(error: rusb::Error) -> AudioError {
        AudioError::Adapter(error.to_string())
    }
}

#[cfg(all(target_os = "linux", feature = "hardware"))]
pub use native::XvfUsb;
