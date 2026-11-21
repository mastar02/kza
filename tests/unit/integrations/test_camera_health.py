"""Camera health scope, authorization and HA evidence projection."""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from weakref import ref

import pytest

from src.home_assistant.control_policy import ActorKind, HomeActionDenied, HomeActor
from src.integrations.camera_health import (
    CameraAvailability,
    CameraDiagnostic,
    CameraHealthMonitor,
    CameraHealthScopeError,
    DiagnosticKind,
    DiagnosticStatus,
    HAHealthSnapshot,
    HAStateReading,
    SourceStatus,
)

NOW = datetime(2026, 11, 21, 12, tzinfo=UTC)
OLDER = NOW - timedelta(seconds=30)
CAMERA = "camera.selected"
READER = HomeActor(ActorKind.DASHBOARD, "reader", frozenset({"home:read"}))


class Provider:
    def __init__(self, read: Callable[[], HAHealthSnapshot]) -> None:
        self.read = read
        self.calls: list[tuple[str, ...]] = []

    async def read_technical_states(
        self, entity_ids: tuple[str, ...]
    ) -> HAHealthSnapshot:
        self.calls.append(entity_ids)
        return self.read()


def monitor(
    provider: Provider | None, diagnostics: tuple[CameraDiagnostic, ...] = ()
) -> CameraHealthMonitor:
    return CameraHealthMonitor(
        provider, {CAMERA: diagnostics}, clock=lambda: NOW, max_staleness_s=30
    )


def reading(
    state: object, observed_at: datetime | None = NOW, **attributes: object
) -> HAStateReading:
    return HAStateReading(observed_at, {"state": state, "attributes": attributes})


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("idle", CameraAvailability.ONLINE),
        ("recording", CameraAvailability.ONLINE),
        ("streaming", CameraAvailability.ONLINE),
        ("unavailable", CameraAvailability.OFFLINE),
        ("unknown", CameraAvailability.UNKNOWN),
        ("on", CameraAvailability.UNKNOWN),
        (["idle"], CameraAvailability.UNKNOWN),
    ],
)
async def test_only_allowed_camera_states_are_projected(state, expected):
    provider = Provider(lambda: HAHealthSnapshot(True, NOW, {CAMERA: reading(state)}))

    report = await monitor(provider).read(READER)

    assert report.source_status is SourceStatus.AVAILABLE
    assert report.observed_at == NOW
    assert report.cameras[0].availability is expected
    assert report.cameras[0].observed_at == NOW
    assert not report.cameras[0].stale
    assert provider.calls == [(CAMERA,)]


async def test_camera_offline_ha_disconnected_and_old_source_are_distinct():
    snapshot = HAHealthSnapshot(True, NOW, {CAMERA: reading("unavailable")})
    provider = Provider(lambda: snapshot)
    service = monitor(provider)
    offline = await service.read(READER)

    snapshot = HAHealthSnapshot(False, NOW, {CAMERA: reading("unavailable")})
    disconnected = await service.read(READER)
    snapshot = HAHealthSnapshot(True, OLDER, {CAMERA: reading("unavailable", OLDER)})
    stale = await service.read(READER)
    snapshot = HAHealthSnapshot(True, NOW, {CAMERA: reading("idle")})
    reconnected = await service.read(READER)

    assert offline.source_status is SourceStatus.AVAILABLE
    assert offline.cameras[0].availability is CameraAvailability.OFFLINE
    assert disconnected.source_status is SourceStatus.DISCONNECTED
    assert disconnected.cameras[0].availability is CameraAvailability.UNKNOWN
    assert disconnected.cameras[0].observed_at is None
    assert stale.source_status is SourceStatus.STALE
    assert stale.observed_at == OLDER
    assert stale.cameras[0].availability is CameraAvailability.UNKNOWN
    assert stale.cameras[0].observed_at == OLDER
    assert stale.cameras[0].stale
    assert reconnected.cameras[0].availability is CameraAvailability.ONLINE


async def test_fresh_source_can_return_expired_missing_or_mismatched_entity_evidence():
    states = {
        CAMERA: reading("streaming", OLDER),
        "camera.other": reading("idle"),
        "camera.mismatched": HAStateReading(
            NOW, {"entity_id": "camera.unselected", "state": "idle"}
        ),
    }
    provider = Provider(lambda: HAHealthSnapshot(True, NOW, states))
    configured = dict.fromkeys((CAMERA, "camera.missing", "camera.mismatched"), ())
    service = CameraHealthMonitor(
        provider, configured, clock=lambda: NOW, max_staleness_s=30
    )

    report = await service.read(READER)

    assert report.source_status is SourceStatus.AVAILABLE
    stale, missing, mismatched = report.cameras
    assert (stale.availability, stale.observed_at, stale.stale) == (
        CameraAvailability.UNKNOWN,
        OLDER,
        True,
    )
    for camera in (missing, mismatched):
        assert (camera.availability, camera.observed_at, camera.stale) == (
            CameraAvailability.UNKNOWN,
            None,
            False,
        )
    assert provider.calls == [tuple(configured)]


@pytest.mark.parametrize(
    "observed_at", [None, datetime(2026, 11, 21), NOW + timedelta(seconds=1)]
)
async def test_unusable_source_timestamps_do_not_yield_current_health(observed_at):
    provider = Provider(
        lambda: HAHealthSnapshot(True, observed_at, {CAMERA: reading("idle")})
    )

    report = await monitor(provider).read(READER)

    assert report.source_status is SourceStatus.UNKNOWN
    assert report.observed_at is None
    assert report.cameras[0].availability is CameraAvailability.UNKNOWN


async def test_missing_configuration_is_reported_without_contacting_ha():
    def unexpected_read() -> HAHealthSnapshot:
        pytest.fail("Empty camera selection must not call the provider")

    provider = Provider(unexpected_read)
    service = CameraHealthMonitor(provider, {}, clock=lambda: NOW, max_staleness_s=30)

    no_cameras = await service.read(READER)
    no_provider = await monitor(None).read(READER)

    assert no_cameras.source_status is SourceStatus.NOT_CONFIGURED
    assert no_cameras.cameras == ()
    assert no_provider.source_status is SourceStatus.NOT_CONFIGURED
    assert no_provider.cameras[0].availability is CameraAvailability.UNKNOWN
    assert provider.calls == []


async def test_diagnostics_report_only_configured_technical_conditions_with_their_own_age():
    diagnostics = (
        CameraDiagnostic("binary_sensor.link", DiagnosticKind.CONNECTIVITY),
        CameraDiagnostic("binary_sensor.supply", DiagnosticKind.POWER),
        CameraDiagnostic("binary_sensor.charge", DiagnosticKind.BATTERY),
        CameraDiagnostic("sensor.card_status", DiagnosticKind.STORAGE),
        CameraDiagnostic("binary_sensor.fault", DiagnosticKind.DEVICE_ERROR),
        CameraDiagnostic("sensor.battery_percent", DiagnosticKind.BATTERY),
    )
    states = {
        CAMERA: reading("idle"),
        "binary_sensor.link": reading("off", device_class="connectivity"),
        "binary_sensor.supply": reading("on", device_class="power"),
        "binary_sensor.charge": reading("on", device_class="battery"),
        "sensor.card_status": reading("storage_full", device_class="enum"),
        "binary_sensor.fault": reading("on", OLDER, device_class="problem"),
        "sensor.battery_percent": reading("2", device_class="battery"),
        "binary_sensor.unselected": reading("on", device_class="problem"),
    }
    provider = Provider(lambda: HAHealthSnapshot(True, NOW, states))

    report = await monitor(provider, diagnostics).read(READER)

    assert provider.calls == [(CAMERA, *(item.entity_id for item in diagnostics))]
    link, power, battery, storage, fault, battery_level = report.cameras[0].diagnostics
    assert report.cameras[0].availability is CameraAvailability.ONLINE
    assert [link.status, power.status, battery.status, storage.status] == [
        DiagnosticStatus.ISSUE,
        DiagnosticStatus.NORMAL,
        DiagnosticStatus.ISSUE,
        DiagnosticStatus.ISSUE,
    ]
    assert (fault.status, fault.observed_at, fault.stale) == (
        DiagnosticStatus.UNKNOWN,
        OLDER,
        True,
    )
    assert (battery_level.status, battery_level.observed_at, battery_level.stale) == (
        DiagnosticStatus.UNKNOWN,
        NOW,
        False,
    )


@pytest.mark.parametrize(
    "entity_id",
    [
        "binary_sensor.backyard_motion",
        "sensor.front_personcount",
        "sensor.occupancy",
        "binary_sensor.front_detectedpeople",
        "binary_sensor.p_r_e_s_e_n_c_e",
        "sensor.deteccion_personas",
    ],
)
def test_privacy_sensitive_diagnostics_are_rejected_at_registration(entity_id):
    with pytest.raises(ValueError, match="private_camera_diagnostic_entity"):
        CameraDiagnostic(entity_id, DiagnosticKind.DEVICE_ERROR)


@pytest.mark.parametrize(
    "device_class", ["motion", "moving", "presence", "occupancy", None]
)
async def test_non_technical_binary_device_class_cannot_leak_event_state_or_time(
    device_class,
):
    diagnostic = CameraDiagnostic("binary_sensor.alias", DiagnosticKind.DEVICE_ERROR)
    provider = Provider(
        lambda: HAHealthSnapshot(
            True,
            NOW,
            {
                diagnostic.entity_id: reading("on", device_class=device_class),
            },
        )
    )

    result = (
        (await monitor(provider, (diagnostic,)).read(READER)).cameras[0].diagnostics[0]
    )

    assert (result.status, result.observed_at, result.stale) == (
        DiagnosticStatus.UNKNOWN,
        None,
        False,
    )


@pytest.mark.parametrize("device_class", [None, "enum", "presence"])
async def test_non_technical_scalar_state_cannot_leak_event_time(device_class):
    diagnostic = CameraDiagnostic("sensor.alias", DiagnosticKind.DEVICE_ERROR)
    provider = Provider(
        lambda: HAHealthSnapshot(
            True,
            NOW,
            {
                diagnostic.entity_id: reading("home", device_class=device_class),
            },
        )
    )

    result = (
        (await monitor(provider, (diagnostic,)).read(READER)).cameras[0].diagnostics[0]
    )

    assert (result.status, result.observed_at, result.stale) == (
        DiagnosticStatus.UNKNOWN,
        None,
        False,
    )


async def test_raw_attributes_and_urls_are_neither_exposed_nor_retained(caplog):
    class AttributePayload(dict):
        def get(self, key, default=None):
            assert key == "device_class", "Only technical device class may be inspected"
            return super().get(key, default)

    captured_attributes = []
    secrets = {
        "entity_picture": "/api/camera_proxy/camera.selected?token=private-picture",
        "access_token": "private-token",
        "frontend_stream_type": "web_rtc",
        "stream_source": "https://account:password@example.invalid/stream.m3u8",
        "rtsp_url": "rtsp://account:password@example.invalid/video",
        "nested": {"arbitrary_key": "https://another:credential@example.invalid/media"},
    }
    diagnostic = CameraDiagnostic("binary_sensor.alert", DiagnosticKind.DEVICE_ERROR)

    def read() -> HAHealthSnapshot:
        attributes = AttributePayload(device_class="problem", **secrets)
        captured_attributes.append(ref(attributes))
        snapshot = HAHealthSnapshot(
            True,
            NOW,
            {
                CAMERA: HAStateReading(
                    NOW, {"state": "streaming", "attributes": attributes}
                ),
                diagnostic.entity_id: HAStateReading(
                    NOW, {"state": "on", "attributes": attributes}
                ),
                "camera.unselected": reading("recording"),
            },
        )
        assert repr(snapshot.states[CAMERA]) == f"HAStateReading(observed_at={NOW!r})"
        assert "private" not in repr(snapshot)
        return snapshot

    provider = Provider(read)
    service = monitor(provider, (diagnostic,))

    with caplog.at_level("DEBUG", logger="src.integrations.camera_health"):
        report = await service.read(READER)
    serialized = json.dumps(asdict(report), default=str)
    exposed = serialized + repr(report) + caplog.text

    for sensitive in (
        *secrets,
        "private-picture",
        "private-token",
        "account:password",
        "another:credential",
        "camera_proxy",
        "m3u8",
        "rtsp://",
        "web_rtc",
        "streaming",
        "camera.unselected",
    ):
        assert sensitive not in exposed
    assert captured_attributes[0]() is None
    assert report.cameras[0].diagnostics[0].status is DiagnosticStatus.ISSUE


async def test_provider_failure_redacts_exception_details_and_reports_source_disconnected(
    caplog,
):
    def disconnected() -> HAHealthSnapshot:
        raise TimeoutError(
            "https://account:password@example.invalid/?access_token=secret"
        )

    report = await monitor(Provider(disconnected)).read(READER)

    assert report.source_status is SourceStatus.DISCONNECTED
    assert report.cameras[0].availability is CameraAvailability.UNKNOWN
    assert caplog.messages == ["Camera health source read failed"]
    assert caplog.records[0].exc_info is None


@pytest.mark.parametrize(
    "operation",
    [
        "ver cámara",
        "mostrar cámara en vivo",
        "pasame el video",
        "video",
        "snapshot",
        "camera.snapshot",
        "camera_proxy",
        "RTSP",
        "HLS",
        "WebRTC",
        "grabación",
        "reconocimiento facial",
        "detección de movimiento",
        "health; snapshot",
        "",
    ],
)
async def test_requests_outside_technical_health_are_denied_even_for_admin(operation):
    provider = Provider(lambda: HAHealthSnapshot(True, NOW))
    admin = HomeActor(ActorKind.DASHBOARD, "admin", frozenset({"admin:manage"}))

    with pytest.raises(
        CameraHealthScopeError, match="Solo puedo consultar el estado técnico"
    ):
        await monitor(provider).read(admin, operation=operation)

    assert provider.calls == []


@pytest.mark.parametrize(
    "permissions",
    [frozenset(), frozenset({"home:control"}), frozenset({"diagnostics:read"})],
)
async def test_unauthorized_actor_cannot_trigger_ha_read(permissions):
    provider = Provider(lambda: HAHealthSnapshot(True, NOW))
    actor = HomeActor(ActorKind.DASHBOARD, "other", permissions)

    with pytest.raises(HomeActionDenied, match="home_read_denied"):
        await monitor(provider).read(actor)

    assert provider.calls == []


async def test_admin_scope_is_accepted_and_revocation_during_read_blocks_the_result():
    current = True

    def revoke_during_read() -> HAHealthSnapshot:
        nonlocal current
        current = False
        return HAHealthSnapshot(True, NOW, {CAMERA: reading("idle")})

    provider = Provider(revoke_during_read)
    admin = HomeActor(ActorKind.DASHBOARD, "admin", frozenset({"admin:manage"}))
    expiring = HomeActor(
        ActorKind.DASHBOARD, "expiring", frozenset({"home:read"}), lambda: current
    )
    service = monitor(provider)

    with pytest.raises(HomeActionDenied, match="home_control_denied"):
        await service.read(expiring)
    with pytest.raises(HomeActionDenied, match="home_control_denied"):
        await service.read(expiring)
    assert provider.calls == [(CAMERA,)]
    assert (await service.read(admin)).cameras[
        0
    ].availability is CameraAvailability.ONLINE
