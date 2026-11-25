#![allow(clippy::unwrap_used)]

use std::fs;
use std::io;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::time::Duration;

use nexa_audiod::capture_seal::{CaptureControl, CaptureState};
use nexa_audiod::grpc::AudioService;
use nexa_audiod::hw::amp::Ma1260;
use nexa_audiod::hw::simulation::Simulation;
use nexa_audiod::hw::sysfs::{alsa_card, parse_usb_port};
use nexa_audiod::hw::xvf::{ControlSetup, INTERFACE, reboot_frame, tuning_frames};
use nexa_audiod::ma1260::{ACK, Command, CommandFrame};
use nexa_audiod::traits::{AmpTransport, AudioError};
use nexa_config::{Ma1260Connection, XvfTuning};
use nexa_proto::v1::ZoneState;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn card(root: &Path, number: u8, port: &str, id: &str) {
    let device = root.join(format!("devices/pci0000:00/usb5/5-5/{port}"));
    let interface = device.join(format!("{port}:1.0"));
    fs::create_dir_all(&interface).unwrap();
    fs::write(device.join("serial"), format!("SERIAL-{number}\n")).unwrap();
    let card = root.join("class/sound").join(format!("card{number}"));
    fs::create_dir_all(&card).unwrap();
    symlink(interface, card.join("device")).unwrap();
    fs::write(card.join("id"), format!("{id}\n")).unwrap();
}

#[test]
fn usb_topology_resolves_exact_alsa_card_by_port_or_serial() {
    let root = tempfile::tempdir().unwrap();
    card(root.path(), 7, "5-5.4", "Array_1");
    card(root.path(), 13, "5-5.14", "Array_2");
    assert_eq!(parse_usb_port("5-5.4"), Some((5, vec![5, 4])));
    assert_eq!(
        alsa_card(root.path(), Some("5-5.4"), None).unwrap(),
        "Array_1"
    );
    assert_eq!(
        alsa_card(root.path(), None, Some("SERIAL-13")).unwrap(),
        "Array_2"
    );
    assert!(alsa_card(root.path(), Some("5-5.4"), Some("SERIAL-13")).is_err());
    assert!(alsa_card(root.path(), Some("5-5"), None).is_err());
    assert_eq!(
        alsa_card(root.path(), Some("5-5.0"), None)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    card(root.path(), 9, "5-5.4", "Another");
    assert_eq!(
        alsa_card(root.path(), Some("5-5.4"), None)
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn xvf_tuning_and_reboot_use_the_firmware_wire_format() {
    assert_eq!(INTERFACE, 3);
    let writes = tuning_frames(&XvfTuning {
        pp_agc_max_gain: 16.0,
        pp_attns_mode: 1,
        pp_attns_slope: 1.0,
        pp_min_nn: 0.4,
    });
    assert_eq!(
        writes.map(|write| (write.setup, write.payload)),
        [
            (11, [0, 0, 128, 65]),
            (32, [1, 0, 0, 0]),
            (34, [0, 0, 128, 63]),
            (22, [205, 204, 204, 62]),
        ]
        .map(|(value, payload)| (
            ControlSetup {
                request_type: 64,
                request: 0,
                value,
                index: 17
            },
            payload
        ))
    );
    let reboot = reboot_frame();
    assert_eq!(
        reboot.setup,
        ControlSetup {
            request_type: 64,
            request: 0,
            value: 7,
            index: 48
        }
    );
    assert_eq!(reboot.payload, [1]);
    assert_eq!(
        ControlSetup::read(17, 11),
        ControlSetup {
            request_type: 192,
            request: 0,
            value: 139,
            index: 17
        }
    );
}

#[tokio::test]
async fn tcp_round_trips_frames_reconnects_and_requires_explicit_ack() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let amplifier = tokio::spawn(async move {
            for (step, ack) in [(20, ACK), (30, ACK), (38, 0x15), (14, ACK)] {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut frame = [0; 5];
                stream.read_exact(&mut frame).await.unwrap();
                assert_eq!(frame, [2, 3, 2, step, 3]);
                stream.write_all(&[ack]).await.unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        let transport = Ma1260::new(Ma1260Connection::Tcp {
            host: "127.0.0.1".into(),
            port,
        });
        for step in [20, 30, 38, 14] {
            let result = transport
                .send(CommandFrame::new(3, Command::Volume, step).unwrap())
                .await;
            if step == 38 {
                assert!(matches!(result, Err(AudioError::Rejected)));
            } else {
                result.unwrap();
            }
        }
        amplifier.await.unwrap();
    })
    .await
    .unwrap();
}

#[test]
fn duplicate_physical_amplifier_zones_are_rejected() {
    let capture = CaptureControl::restore(CaptureState::default(), true).unwrap();
    let zones = ["hall", "living"].map(|room| ZoneState {
        zone_id: format!("zone_{room}"),
        room_id: room.into(),
        ma1260_zone: 4,
        ..Default::default()
    });
    let result = AudioService::new(
        std::sync::Arc::new(std::sync::Mutex::new(capture)),
        Ma1260::new(Ma1260Connection::Simulation),
        Simulation,
        zones,
    );
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}
