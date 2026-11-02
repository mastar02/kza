"""Empty metadata requires a real revision and live authority; raw stays strict."""

from dataclasses import replace
from weakref import ref

import pytest

from src.dashboard.live_retention_types import RevisionStamp, SnapshotMetadata
from src.memory.fact_ledger import FactLedger
from src.users.voice_ledger import VoiceLedger
from tests.mocks.memory_sources import memory_source


@pytest.mark.parametrize('channel',['voices:profiles','voices:captures'])
def test_validated_empty_metadata_needs_revision_deadline_and_live_policy(channel):
    ledger = FactLedger(clock=lambda:2000.)
    ledger.initialize()
    voices = VoiceLedger(ledger)
    metadata = SnapshotMetadata('synthetic-event',(),2030.,channel,
        revision=RevisionStamp(ref(voices),voices.projection_revision),empty_metadata=True)
    try:
        assert metadata.available(ledger,2000.)
        assert not ledger.sources_available(())
        assert not replace(metadata,empty_metadata=False).available(ledger,2000.)
        assert not replace(metadata,revision=None).available(ledger,2000.)
        assert not replace(metadata,channel='voices:review').available(ledger,2000.)
        assert not replace(metadata,channel='conversations').available(ledger,2000.)
        assert not metadata.available(ledger,2030.)
        ledger._voice_projection_revision += 1
        assert not metadata.available(ledger,2000.)
    finally:
        ledger.close()
    assert not replace(metadata,revision=RevisionStamp(ref(voices),voices.projection_revision)).available(ledger,2000.)


def test_empty_metadata_marker_does_not_bypass_actual_raw_source_expiry():
    now = [2000.]
    ledger = FactLedger(clock=lambda:now[0])
    ledger.initialize()
    voices = VoiceLedger(ledger)
    metadata = SnapshotMetadata('synthetic-event',(memory_source(now=2000.,ttl=1),),2030.,'voices:captures',
        revision=RevisionStamp(ref(voices),voices.projection_revision),empty_metadata=True)
    try:
        assert metadata.available(ledger,2000.)
        now[0] = 2002.
        assert not metadata.available(ledger,2002.)
    finally:
        ledger.close()
