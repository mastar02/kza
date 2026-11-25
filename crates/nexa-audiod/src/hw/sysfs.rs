//! Resolve ALSA cards from Linux sysfs without consulting ALSA or global state.

use std::fs;
use std::io;
use std::path::Path;

/// Parse the USB bus and physical port chain, e.g. "5-5.4".
pub fn parse_usb_port(value: &str) -> Option<(u8, Vec<u8>)> {
    let (bus, chain) = value.split_once('-')?;
    let bus: u8 = bus.parse().ok()?;
    let chain = chain
        .split('.')
        .map(|part| part.parse::<u8>().ok().filter(|port| *port > 0))
        .collect::<Option<Vec<_>>>()?;
    (bus > 0).then_some((bus, chain))
}

/// Identify the unique card for a physical USB port or USB serial.
pub fn alsa_card(root: &Path, port: Option<&str>, serial: Option<&str>) -> io::Result<String> {
    let (_, card) = resolve(root, port, serial)?;
    Ok(card)
}

pub(crate) fn resolve(
    root: &Path,
    port: Option<&str>,
    serial: Option<&str>,
) -> io::Result<(String, String)> {
    if port.is_some_and(|value| parse_usb_port(value).is_none())
        || (port.is_none() && serial.is_none())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid USB identity",
        ));
    }
    let mut found = None;
    for entry in fs::read_dir(root.join("class/sound"))? {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(index) = file_name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
        else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|digit| digit.is_ascii_digit()) {
            continue;
        }
        let path = match entry.path().join("device").canonicalize() {
            Ok(path) => path,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        // The nearest USB device is the card, even when a parent hub has a serial.
        let Some((device, name)) = path.ancestors().find_map(|ancestor| {
            let name = ancestor.file_name()?.to_str()?;
            parse_usb_port(name).map(|_| (ancestor, name))
        }) else {
            continue;
        };
        if port.is_some_and(|port| name != port)
            || serial.is_some_and(|serial| {
                !fs::read_to_string(device.join("serial")).is_ok_and(|value| value.trim() == serial)
            })
        {
            continue;
        }
        let id = fs::read_to_string(entry.path().join("id"))?;
        let id = id.trim();
        if id.is_empty()
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid ALSA card ID",
            ));
        }
        if found.replace((name.to_owned(), id.to_owned())).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ambiguous ALSA card",
            ));
        }
    }
    found.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "room ALSA card missing"))
}
