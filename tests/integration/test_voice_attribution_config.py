"""U3b-4a-1 G2 and hook wiring through the real application graph and the real link writer.

G2: a missing ``memory.voice_proposals`` section means off, the value is read
without writing back, and the effective config digest the runtime admission
compares (src/core/runtime_admission.py:137-163) is unchanged by the build.
"""

import copy
from contextlib import AsyncExitStack
from pathlib import Path
import sqlite3

import pytest
import yaml

from src.core.runtime_admission import _config_digest
from src.main import build_application
from src.memory.voice_attribution import VoiceConsentState, VoiceProposalSettings
from tests.integration.test_application_graph import leaves  # noqa: F401  (fixture)
from tests.integration.test_voice_person_links import people_graph  # noqa: F401  (fixture)
from tests.mocks.application_graph import offline_config, offline_home

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = {
    "absent": None,
    "disabled": {"enabled": False},
    "enabled": {"enabled": True},
    "invalid": {"enabled": True, "min_score": "high"},
}


@pytest.mark.parametrize("section", list(SECTIONS))
async def test_graph_build_reads_the_voice_section_and_never_writes_it(leaves, tmp_path, section):  # noqa: F811
    async with offline_home() as (url, _observations):
        config = offline_config(tmp_path, url)
        assert "voice_proposals" not in config["memory"]
        if SECTIONS[section] is not None:
            config["memory"]["voice_proposals"] = SECTIONS[section]
        before, digest = copy.deepcopy(config), _config_digest(config)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            ledger = graph.privacy.ledger
            assert ledger.voice_settings == VoiceProposalSettings(enabled=section == "enabled")
            state = await graph.privacy.acall(ledger.voice_switch)
            # (a)/(c): absent or false -> off; even 'enabled' is not effective without the switch row.
            assert (state.configured, state.enabled, state.revision, state.effective) == (
                section == "enabled", False, 0, False)
            assert _config_digest(graph.config) == digest
        assert config == before and _config_digest(config) == digest  # (b)


def test_repository_settings_have_no_voice_section_and_read_off():
    """config/settings.yaml stays without the section in 4a and 4b (G2); reading it changes nothing."""
    raw = (ROOT / "config/settings.yaml").read_bytes()
    config = yaml.safe_load(raw)
    assert "voice_proposals" not in config["memory"]
    before = copy.deepcopy(config)
    assert VoiceProposalSettings.from_config(config) == VoiceProposalSettings()
    assert config == before and (ROOT / "config/settings.yaml").read_bytes() == raw


async def test_relinking_a_verified_profile_invalidates_the_consent(people_graph):  # noqa: F811
    """VoicePersonLinks.set_person (operator subject) is an invalidation call site."""
    g = people_graph
    ledger = g.privacy.ledger
    await g.voices.acall(g.links.set_person, g.profile["id"], g.a, 1, 0, g.actor)

    def seed():
        with ledger._transaction() as db:
            db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,2,?,1,1,?,?)",
                       (g.a, VoiceConsentState.ACTIVE.value, g.profile["id"], ledger.clock(), g.a))
    await g.voices.acall(seed)
    await g.voices.acall(g.links.set_person, g.profile["id"], g.b, 1, 1, g.actor)

    def read():
        with ledger._transaction() as db:
            return [tuple(row) for row in db.execute(
                "SELECT state, revision, verified_profile_id, changed_by FROM voice_attribution_consents")]
    [(state, revision, verified, changed_by)] = await g.voices.acall(read)
    assert (state, revision, verified) == ("invalidated", 3, None)
    assert changed_by in (g.actor.subject, "system:operator")


async def test_a_failing_voice_prune_never_skips_expiry_or_the_delay(tmp_path, monkeypatch, caplog):
    """N4: the voice prune runs after expire in its own try; its failure is logged by type only."""
    import asyncio
    import logging

    from src.memory.privacy_services import PrivacyServices
    from tests.mocks.memory_sources import memory_source

    now = [2_000_000_000.0]
    services = PrivacyServices(str(tmp_path / "privacy.db"), clock=lambda: now[0],
                               voice_settings=VoiceProposalSettings(enabled=True))
    await services.acall(services.initialize)
    try:
        fact = await services.acall(services.ledger.store, "Synthetic expiring fact", "fact", .9,
                                    (memory_source(now=now[0] - 1),))
        now[0] = fact.expires_at + 1
        calls, delayed, loop = [], asyncio.Event(), asyncio.get_running_loop()

        def broken():
            calls.append("prune")
            raise sqlite3.OperationalError("synthetic private detail")

        real_delay = services.ledger.next_maintenance_delay

        def delay():
            calls.append("delay")
            loop.call_soon_threadsafe(delayed.set)
            return real_delay()

        monkeypatch.setattr(services.ledger, "prune_voice_attribution", broken)
        monkeypatch.setattr(services.ledger, "next_maintenance_delay", delay)
        with caplog.at_level(logging.WARNING):
            services.start()
            await asyncio.wait_for(delayed.wait(), 10)
        assert calls[:2] == ["prune", "delay"]
        assert await services.acall(services.ledger.get, fact.id) is None
        assert await services.acall(lambda: services.ledger.status()["states"]) == {"deleted": 1}
        messages = [record.getMessage() for record in caplog.records]
        assert "Voice attribution prune deferred: OperationalError" in messages
        assert not any("synthetic private detail" in message for message in messages)
        assert not any("Privacy maintenance deferred" in message for message in messages)
    finally:
        await services.aclose()
