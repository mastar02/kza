"""Chronological voice pages keep ties, source retention and stable boundaries."""

import struct
import time
from uuid import uuid4

import httpx
import numpy as np
import pytest

from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryReviewer
from src.users import voice_ledger as voice_ledger_module
from src.users.voice_capture import build_review_capture
from src.users.voice_ledger import VoiceCaptureFilter, VoiceReviewer
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import ACTOR, AUDIO, VECTOR, graph, web_graph

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
            app = DashboardAPI(voice_registry=voices, event_bus=bus, authority=bus.test_authority).app
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


# --- T044-U1: filtered captures query (ledger) ------------------------------------------------

SALA, COCINA = 'Sala sintetica', 'Cocina sintetica'
LONG = np.linspace(-.1, .1, 32000, dtype=np.float32)
CAPTURE_ROW = {'id', 'group_id', 'created_at', 'expires_at', 'duration_s', 'model_key', 'capture_ids', 'review_available',
               'room_id', 'suggested_profile', 'similarity', 'association', 'group_revision', 'group_state',
               'group_reason', 'group_size', 'group_similarity', 'group_expires_at'}


async def capture(voices, clock, *, text='Texto sintetico', room=SALA, review=True, ttl=3600, audio=AUDIO,
                  sources=1, step=1):
    """One synthetic capture (optionally with a review row); the clock advances first."""
    clock.now += step
    origin = tuple(memory_source(now=clock.now, ttl=ttl) for _ in range(sources))
    data = build_review_capture(audio, text, room) if review else None
    identity = await voices.acall(voices.ledger.put, 'synthetic-order', VECTOR.copy(), audio.size / 16000, origin, data)
    assert identity
    return origin[0], identity


def flt(**values):
    base = dict(room_kind=None, room_id=None, since=None, until=None, group_state=None, association=None,
                listenable_only=False, text=None)
    return VoiceCaptureFilter(**{**base, **values})


async def query(voices, filters, cursor=None, actor=ACTOR):
    return await voices.acall(voices.ledger.query_captures, filters, cursor, actor)


def ids(projection):
    return {row['id'] for row in projection.rows}


async def household(voices, clock):
    """Synthetic rooms only: two rooms, an unrecorded room, no review row, a group and a suggestion."""
    made = {}
    made['sala'] = (await capture(voices, clock, room=SALA, text='Prendé la luz de la cámara'))[1]
    made['cocina'] = (await capture(voices, clock, room=COCINA, text='Apagá la cocina'))[1]
    made['none'] = (await capture(voices, clock, room=None, text='Sin habitación'))[1]
    made['bare'] = (await capture(voices, clock, review=False))[1]
    grouped = [(await capture(voices, clock, room=SALA, audio=LONG, text='Persiana del living'))[1] for _ in range(3)]
    made['grouped'] = grouped
    await voices.acall(voices.ledger.confirm, tuple(grouped), 'Synthetic profile', str(uuid4()), ACTOR)
    made['suggested'] = (await capture(voices, clock, room=COCINA, audio=LONG, text='Cámara de la cocina'))[1]
    return made


async def test_each_filter_alone_and_combined_counts_live_rows_with_room_ids(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        made = await household(voices, clock)
        g = set(made['grouped'])
        everything = {made['sala'], made['cocina'], made['none'], made['bare'], made['suggested'], *g}
        cases = [
            (flt(), everything),
            (flt(room_kind='room', room_id=SALA), {made['sala'], *g}),
            (flt(room_kind='room', room_id=COCINA), {made['cocina'], made['suggested']}),
            (flt(room_kind='unrecorded'), {made['none']}),
            (flt(group_state='suggested'), {*g, made['suggested']}),
            (flt(group_state='unknown'), {made['sala'], made['cocina'], made['none'], made['bare']}),
            (flt(group_state='manual'), set()),
            (flt(association='suggested'), {made['suggested']}),
            (flt(association='unknown'), everything - {made['suggested']}),
            (flt(listenable_only=True), everything - {made['bare']}),
            (flt(room_kind='room', room_id=SALA, group_state='suggested'), g),
            (flt(room_kind='room', room_id=COCINA, association='suggested', listenable_only=True), {made['suggested']}),
        ]
        for filters, expected in cases:
            projection = await query(voices, filters)
            assert ids(projection) == expected
            assert projection.summary.matching_count == len(expected) and projection.summary.scan_complete
            assert all(set(row) == CAPTURE_ROW for row in projection.rows)
            assert projection.channel == 'voices:captures' and projection.next_cursor is None
        rooms = {row['id']: row['room_id'] for row in (await query(voices, flt())).rows}
        assert rooms[made['sala']] == SALA and rooms[made['cocina']] == COCINA
        assert rooms[made['none']] is None and rooms[made['bare']] is None
        # Rows without a review row never match any room filter.
        assert made['bare'] not in ids(await query(voices, flt(room_kind='unrecorded')))
        # GET /api/voices rows carry the room too (QA), with the same field set.
        listed = {row['id']: row for row in (await voices.acall(voices.ledger.project, 'captures', None, ACTOR)).rows}
        assert all(set(row) == CAPTURE_ROW for row in listed.values())
        assert listed[made['sala']]['room_id'] == SALA and listed[made['bare']]['room_id'] is None


async def test_time_window_selects_creation_times_and_leaves_rows_metadata_only(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        early = [(await capture(voices, clock))[1] for _ in range(3)]
        middle = clock.now
        late = [(await capture(voices, clock))[1] for _ in range(2)]
        projection = await query(voices, flt(since=middle + .5, until=clock.now + 1))
        assert ids(projection) == set(late) and projection.summary.matching_count == 2
        projection = await query(voices, flt(since=middle - 10, until=middle + .5))
        assert ids(projection) == set(early)
        for row in projection.rows:
            assert not {'transcript', 'audio_base64', 'audio_wav', 'embedding', 'vector'} & set(row)


async def test_count_is_the_window_total_on_every_page_and_cursors_continue_without_gaps(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        made = [(await capture(voices, clock, room=SALA if index % 2 else COCINA, review=index % 2 == 1))[1]
                for index in range(240)]
        expected = [identity for index, identity in enumerate(made) if index % 2][::-1]
        filters = flt(room_kind='room', room_id=SALA)
        received, cursor, pages, total = [], None, 0, 120
        while True:
            projection = await query(voices, filters, cursor)
            # The count is the whole window's live total, never 'remaining'.
            assert projection.summary.matching_count == total and projection.summary.scan_complete
            received += [row['id'] for row in projection.rows]
            pages += 1
            cursor = projection.next_cursor
            if pages == 1:
                assert len(projection.rows) == 50 and cursor.startswith('vc2_') and len(cursor) == 118
                await capture(voices, clock, room=SALA)  # A newer row never appears on a later page.
                total = 121
            if cursor is None:
                break
        assert pages == 3 and received == expected


async def test_cursor_is_bound_to_its_filters_query_and_process(tmp_path, monkeypatch):
    async with graph(tmp_path) as (_, _, voices, clock):
        for _ in range(60):
            await capture(voices, clock)
        sala = flt(room_kind='room', room_id=SALA)
        cursor = (await query(voices, sala)).next_cursor
        assert len((await query(voices, sala, cursor)).rows) == 10
        for other in (flt(), flt(room_kind='room', room_id=COCINA), flt(room_kind='room', room_id=SALA, text='texto')):
            with pytest.raises(ValueError, match='voice_cursor_filter_mismatch'):
                await query(voices, other, cursor)
        for malformed in (cursor[:-1], 'vc1_' + cursor[4:85], cursor.replace('vc2_', 'vc3_')):
            with pytest.raises(ValueError):
                await query(voices, sala, malformed)
        monkeypatch.setattr(voice_ledger_module, '_QUERY_CURSOR_KEY', b'\x01' * 32)  # A restarted process.
        with pytest.raises(ValueError, match='voice_cursor_filter_mismatch'):
            await query(voices, sala, cursor)


async def test_expired_and_erased_rows_leave_rows_count_and_room_options(tmp_path):
    async with graph(tmp_path) as (_, owner, voices, clock):
        short = (await capture(voices, clock, room=COCINA, ttl=30))[1]
        source, erased = await capture(voices, clock, room='Patio sintetico')
        kept = (await capture(voices, clock, room=SALA))[1]
        projection = await query(voices, flt())
        assert ids(projection) == {short, erased, kept}
        assert projection.summary.room_options == (COCINA, 'Patio sintetico', SALA)
        await owner.request(source.conversation_id, MemoryReviewer('synthetic-admin'))
        clock.now += 31
        projection = await query(voices, flt())
        assert ids(projection) == {kept} and projection.summary.matching_count == 1
        assert projection.summary.room_options == (SALA,)
        assert ids(await query(voices, flt(room_kind='room', room_id=COCINA))) == set()


async def test_room_options_ignore_the_room_filter_and_expire_at_each_rooms_last_live_row(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        await capture(voices, clock, room=SALA, ttl=100)
        start = clock.now
        await capture(voices, clock, room=COCINA, ttl=50)
        await capture(voices, clock, room=COCINA, ttl=400)
        await capture(voices, clock, room=None)
        projection = await query(voices, flt(room_kind='room', room_id=SALA))
        assert len(projection.rows) == 1
        assert projection.summary.room_options == (COCINA, SALA)
        # The first moment some listed room has no live row left: Sala's only row.
        assert projection.summary.room_options_expires_at == start + 100
        assert projection.summary.count_expires_at == start + 100
        clock.now = start + 100
        projection = await query(voices, flt(room_kind='room', room_id=SALA))
        assert projection.rows == [] and projection.summary.room_options == (COCINA,)
        assert projection.summary.matching_count == 0 and projection.summary.count_expires_at is None
        # A search never carries room options.
        searched = await query(voices, flt(text='texto'))
        assert searched.summary.room_options is None and searched.summary.room_options_expires_at is None


async def test_count_expires_at_the_earliest_deadline_among_the_counted_rows(tmp_path):
    """contract.count.expiry: no tolerance window, even when the earliest deadline is an off-page row's."""
    async with graph(tmp_path) as (_, _, voices, clock):
        off_page = await capture(voices, clock, ttl=200)
        deadlines = [off_page[0].expires_at]
        for index in range(52):
            # Distinct deadlines, none earlier than the oldest row's, newest-first not in deadline order.
            deadlines.append((await capture(voices, clock, ttl=900 + (index * 37) % 500))[0].expires_at)
        middle = await capture(voices, clock, ttl=400)
        deadlines.append(middle[0].expires_at)
        projection = await query(voices, flt())
        assert len(projection.rows) == 50 and off_page[1] not in ids(projection)
        assert projection.summary.matching_count == 54
        assert projection.summary.count_expires_at == min(deadlines) == off_page[0].expires_at
        clock.now = off_page[0].expires_at
        projection = await query(voices, flt())
        assert projection.summary.matching_count == 53
        assert projection.summary.count_expires_at == middle[0].expires_at == min(deadlines[1:])
        following = await query(voices, flt(), projection.next_cursor)
        assert following.summary.count_expires_at == middle[0].expires_at


async def test_page_capacity_cuts_at_256_sources_without_changing_the_count(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        for _ in range(3):
            await capture(voices, clock, sources=100)
        projection = await query(voices, flt())
        assert len(projection.rows) == 2 and len(projection.sources) == 200
        assert projection.summary.matching_count == 3
        following = await query(voices, flt(), projection.next_cursor)
        assert len(following.rows) == 1 and following.next_cursor is None
        assert following.summary.matching_count == 3


async def test_search_matches_accents_case_and_every_term_over_the_filtered_live_rows(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        made = await household(voices, clock)
        assert ids(await query(voices, flt(text='camara'))) == {made['sala'], made['suggested']}
        assert ids(await query(voices, flt(text='camara cocina'))) == {made['suggested']}
        assert ids(await query(voices, flt(text='camara', room_kind='room', room_id=SALA))) == {made['sala']}
        assert ids(await query(voices, flt(text='persiana'))) == set(made['grouped'])
        absent = await query(voices, flt(text='zzz inexistente'))
        assert absent.rows == [] and absent.summary.matching_count == 0 and absent.summary.scan_complete
        for row in (await query(voices, flt(text='camara'))).rows:
            assert set(row) == CAPTURE_ROW  # Metadata only: no transcript, excerpt or highlight.


async def test_search_never_matches_expired_or_unreviewed_rows(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        await capture(voices, clock, text='Alarma sintetica', ttl=20)
        await capture(voices, clock, review=False)
        clock.now += 25
        projection = await query(voices, flt(text='alarma'))
        assert projection.rows == [] and projection.summary.matching_count == 0


@pytest.mark.parametrize('injection', ['correction', 'profile', 'erasure'])
async def test_fence_change_between_scan_and_final_page_restarts_once_then_withholds_the_count(tmp_path, injection):
    async with graph(tmp_path) as (_, _, voices, clock):
        grouped = [(await capture(voices, clock, audio=LONG))[1] for _ in range(3)]
        lone = [(await capture(voices, clock, audio=LONG, room=COCINA))[1] for _ in range(1)]
        spare = [await capture(voices, clock) for _ in range(4)]
        profile = await voices.acall(voices.ledger.confirm, tuple(grouped), 'Synthetic profile', str(uuid4()), ACTOR)
        page = voices.ledger._query_page
        calls, injected = [], []

        def inject():
            index = len(injected)
            injected.append(True)
            if injection == 'correction':
                current = {row['id']: row for row in voices.ledger.snapshot()['fingerprints']}
                target = grouped[index]
                voices.ledger.groups.correct((target,), {current[target]['group_id']: current[target]['group_revision']},
                                             'unknown', str(uuid4()), ACTOR)
            elif injection == 'profile' and index == 0:
                assert voices.ledger.delete_profile(profile['id'], profile['revision'], ACTOR)
            elif injection == 'profile':
                voices.ledger.ledger._voice_projection_revision += 1  # Another profile change.
            else:
                voices.privacy.ledger.delete_conversation(spare[index][0].conversation_id,
                                                          MemoryReviewer('synthetic-admin'))

        def racing(*args):
            calls.append(True)
            if len(calls) <= limit:
                inject()
            return page(*args)

        voices.ledger._query_page = racing
        try:
            for limit, complete in ((1, True), (2, False)):
                calls.clear()
                projection = await query(voices, flt(group_state='suggested'))
                assert len(calls) == 2
                assert projection.summary.scan_complete is complete
                assert (projection.summary.matching_count is not None) is complete
                assert projection.summary.room_options is None or complete
                # Page rows always pass project()'s revalidation: no stale group state.
                assert all(row['group_state'] == 'suggested' for row in projection.rows)
                if complete:
                    assert projection.summary.matching_count == len(projection.rows)
        finally:
            del voices.ledger._query_page
        assert lone


async def test_actor_revoked_during_the_scan_stops_it(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        for _ in range(300):
            await capture(voices, clock, review=False, step=0)
        checks = []

        def current():
            checks.append(True)
            return len(checks) < 6

        with pytest.raises(PermissionError):
            await query(voices, flt(), actor=VoiceReviewer('synthetic-admin', current))


async def test_budget_full_window_filters_cursor_mode_search_and_worker_time(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        for index in range(4096):
            await capture(voices, clock, review=index < 600, text='Registro sintetico %d' % index,
                          room=SALA, step=.001, ttl=7200)
        timings = []

        async def timed(*args):
            started = time.perf_counter()
            result = await query(voices, *args)
            timings.append(time.perf_counter() - started)
            return result

        everything = await timed(flt())
        assert len(everything.rows) == 50 and everything.summary.matching_count == 4096
        assert everything.summary.scan_complete and everything.summary.room_options == (SALA,)
        # 600 review rows exceed the 512-row search budget: cursor mode, no count.
        first = await timed(flt(text='registro'))
        assert len(first.rows) == 50 and first.summary.matching_count is None and not first.summary.scan_complete
        seen, cursor = {row['id'] for row in first.rows}, first.next_cursor
        while cursor:
            following = await timed(flt(text='registro'), cursor)
            assert following.summary.matching_count is None
            assert not seen & ids(following)
            seen |= ids(following)
            cursor = following.next_cursor
        assert len(seen) == 600
        # A window that fits the budget returns the exact count on every page.
        newest = sorted(row['created_at'] for row in first.rows)
        window = flt(text='registro', since=newest[0] - .0005 - .001 * 70, until=clock.now + 1)
        fitted = await timed(window)
        assert fitted.summary.scan_complete and fitted.summary.matching_count > 50
        following = await timed(window, fitted.next_cursor)
        assert following.summary.matching_count == fitted.summary.matching_count
        timings.sort()
        print('t044 query worker seconds p50=%.3f p95=%.3f max=%.3f n=%d' % (
            timings[len(timings) // 2], timings[int(len(timings) * .95)], timings[-1], len(timings)))


async def test_cursor_key_is_random_per_process():
    """A constant key would let a cursor outlive its process or act as a reversible fingerprint."""
    import subprocess
    import sys
    script = 'from src.users.voice_ledger import _QUERY_CURSOR_KEY as k; print(k.hex())'
    keys = {subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True,
                           timeout=60).stdout.strip() for _ in range(2)}
    assert len(keys) == 2 and all(len(key) == 64 for key in keys)


async def test_row_expiring_between_scan_and_final_page_leaves_rows_count_and_options(tmp_path):
    async with graph(tmp_path) as (_, _, voices, clock):
        await capture(voices, clock, room=COCINA, ttl=40)
        kept = (await capture(voices, clock, room=SALA))[1]
        page = voices.ledger._query_page

        def later(*args):
            clock.now += 45  # Clock-based expiry between chunks and the final transaction.
            return page(*args)

        voices.ledger._query_page = later
        try:
            projection = await query(voices, flt())
        finally:
            del voices.ledger._query_page
        assert ids(projection) == {kept} and projection.summary.matching_count == 1
        assert projection.summary.room_options == (SALA,)
