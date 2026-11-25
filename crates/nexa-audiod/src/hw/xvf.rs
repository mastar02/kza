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

#[cfg(all(target_os = "linux", feature = "hardware"))]
mod native {
    use std::time::Duration;

    use rusb::DeviceHandle;

    use super::{ControlSetup, INTERFACE, reboot_frame, tuning_frames};
    use crate::hw::sysfs::parse_usb_port;
    use crate::traits::{AudioError, XvfTransport};

    const XVF_VID: u16 = 0x2886;
    const XVF_PID: u16 = 0x001a;
    const TIMEOUT: Duration = Duration::from_millis(250);

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
            Ok(())
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
                let setup = ControlSetup::read(resource, command);
                let mut data = vec![0; size + 1];
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
                if length != data.len() || data[0] != 0 {
                    return Err(AudioError::Rejected);
                }
                Ok(data.split_off(1))
            })
            .await
            .map_err(|_| AudioError::Unavailable)??;
            response.copy_from_slice(&data);
            Ok(data.len())
        }
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
