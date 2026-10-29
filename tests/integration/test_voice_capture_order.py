"""Chronological voice pages keep ties, source retention and stable boundaries."""

import struct

import httpx
import pytest

from src.dashboard.api import DashboardAPI
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import VECTOR, graph, web_graph

pytestmark = pytest.mark.asyncio


async def seed(voices, clock, count):
    start = clock.now
    expected = []
    for index in range(count):
        clock.now = start + index // 3
        source = memory_source(now=clock.now, ttl=3600)
        identity = await voices.acall(voices.ledger.put, 'synthetic-order', VECTOR.copy(), 1., (source,))
        expected.append((clock.now, identity))
    return sorted(expected, reverse=True)


async def test_capture_api_is_newest_first_across_pages_and_does_not_repeat_after_new_arrival(tmp_path, dashboard_auth):
    async with web_graph(tmp_path) as (_, _, voices, clock, bus):
        expected = await seed(voices, clock, 115)
        async with ready_browser(bus, version=3) as peer:
            app = DashboardAPI(voice_registry=voices, event_bus=bus).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver',
                    headers={**dashboard_auth, 'X-KZA-Live-Connection': peer.connection}) as client:
                pages, received, cursor = [], [], None
                while True:
                    response = await client.get('/api/voices', params={'cursor': cursor} if cursor else {})
                    assert response.status_code == 200
                    batch = response.json()['snapshot']
                    rows = batch['rows']
                    received.extend((row['created_at'], row['id']) for row in rows)
                    pages.append(len(rows))
                    cursor = batch['next_cursor']
                    await peer.message({'type': 'live_ack', 'receipts': [batch['receipt']]})
                    if len(pages) == 1:
                        assert received == expected[:50]
                        assert cursor.startswith('vc1_') and len(cursor) == 85
                        clock.now += 1
                        newer = memory_source(now=clock.now, ttl=3600)
                        await voices.acall(voices.ledger.put, 'synthetic-order', VECTOR.copy(), 1., (newer,))
                    if not cursor:
                        break
                assert pages == [50, 50, 15]
                assert received == expected


async def test_snapshot_cursor_keeps_position_after_anchor_expires(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        expected = await seed(voices, clock, 101)
        first = await voices.acall(voices.ledger.snapshot)
        assert [(row['created_at'], row['id']) for row in first['fingerprints']] == expected[:100]
        anchor = expected[99][1]
        # Remove just the anchor as retention does between independent requests.
        def expire_anchor():
            with voices.privacy.ledger._transaction() as db:
                db.execute('UPDATE voice_fingerprints SET expires_at=? WHERE id=?', (clock.now, anchor))
            voices.ledger.prune()
        await voices.acall(expire_anchor)
        following = await voices.acall(voices.ledger.snapshot, first['next_cursor'])
        assert [(row['created_at'], row['id']) for row in following['fingerprints']] == expected[100:]
        assert following['next_cursor'] is None


@pytest.mark.parametrize('cursor', ['', 'vc1_bad', 'vc1_' + '0' * 16 + '_' + 'z' * 64,
    'vc1_' + struct.pack('>d', float('nan')).hex() + '_' + 'a' * 64,
    'vc1_' + struct.pack('>d', float('inf')).hex() + '_' + 'a' * 64,
    'vc1_' + struct.pack('>d', -1.).hex() + '_' + 'a' * 64])
async def test_malformed_capture_cursor_is_rejected(tmp_path, cursor):
    async with graph(tmp_path) as (_, _, voices, _):
        with pytest.raises(ValueError, match='invalid_voice_cursor'):
            await voices.acall(voices.ledger.snapshot, cursor)
