"""Conversation DTOs, scoped feedback and actual managed export formats."""

import json
from pathlib import Path
from uuid import UUID

import pytest

from src.memory.producer_context import producer_context
from src.training.conversation_collector import ResponseQuality
from tests.mocks.conversations import graph

pytestmark = pytest.mark.asyncio


async def test_valid_turn_has_opaque_file_and_source_deadline(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(input_ttl=20,response_ttl=2)
        identifier = await g.add(sources=sources,user_name='synthetic-user')
        assert str(UUID(identifier)) == identifier
        view, = await g.collector.get_turns()
        assert view.user_input == 'synthetic input' and view.assistant_response == 'synthetic response'
        assert view.sources == sources and view.expires_at == sources[1].expires_at
        path = g.collector.data_dir/'policy_v1'/(identifier+'.json')
        assert path.stat().st_mode & 0o777 == 0o600 and 'synthetic-user' not in path.name
        row, = await g.rows()
        assert row['quality'] == 'unmarked' and row['group_id'] == sources[0].conversation_id


async def test_no_unsourced_or_invalid_record_is_collected(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.collector.add_turn('synthetic','synthetic') is None
        assert await g.add(user_input='x'*4097) is None
        assert await g.add(entities_used=[object()]) is None
        assert not await g.rows() and not g.collector.data_dir.exists()


async def test_mark_last_only_matches_exact_conversation_and_room(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources()
        first = await g.add(sources=sources)
        with producer_context('another-room', (g.feedback(sources[0]),)):
            assert not await g.collector.mark_last_response('good')
        with producer_context('synthetic-room', (g.base.source(),)):
            assert not await g.collector.mark_last_response('good')
        with producer_context('synthetic-room', (g.feedback(sources[0]),)):
            assert await g.collector.mark_last_response('good')
        view, = await g.collector.get_turns()
        assert view.id != first and view.quality is ResponseQuality.GOOD
        assert view.expires_at == min(s.expires_at for s in sources)


async def test_feedback_replaces_file_and_retires_existing_view(tmp_path):
    async with graph(tmp_path) as g:
        await g.add()
        old, = await g.collector.get_turns()
        assert await g.collector.mark_response(old,'corrected','synthetic correction',sources=(g.base.source(),))
        assert not old.available and old.user_input == old.assistant_response == ''
        updated, = await g.collector.get_turns()
        assert updated.correction == 'synthetic correction' and updated.quality is ResponseQuality.CORRECTED
        assert updated.id != old.id


@pytest.mark.parametrize('format', ['alpaca','sharegpt','conversations'])
async def test_export_format_and_lineage_are_preserved(tmp_path,format):
    async with graph(tmp_path) as g:
        await g.add()
        view, = await g.collector.get_turns()
        assert await g.collector.mark_response(view,'good',sources=(g.base.source(),))
        export = await g.collector.export_for_training(str(tmp_path/'exports'),format)
        assert export is not None and export.available and export.exported == export.good_responses == 1
        path = Path(export.output_path)
        payload = json.loads(path.read_text())
        assert payload['_kza']['record_id'] != view.id
        assert payload['_kza']['sources'] and payload['_kza']['expires_at'] == export.expires_at
        if format == 'alpaca':
            assert payload['instruction'] == 'synthetic input' and payload['output'] == 'synthetic response'
        elif format == 'sharegpt':
            assert payload['conversations'][1]['value'] == 'synthetic response'
        else:
            assert payload['user'] == 'synthetic input' and payload['assistant'] == 'synthetic response'
        assert path.stat().st_mode & 0o777 == 0o600


async def test_stats_and_grouping_use_available_records(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources()
        assert g.collector.start_conversation(sources=(sources[0],)) == sources[0].conversation_id
        for _ in range(2):
            assert await g.add(sources=sources)
        stats = await g.collector.get_stats()
        assert stats['total_conversations'] == 1 and stats['total_turns'] == 2
        groups = await g.collector.get_conversations()
        assert len(groups)==1 and len(groups[0].turns)==2
        assert groups[0].user_name is None
        g.collector.end_conversation()
        assert (await g.collector.get_stats())['current_conversation_turns'] == 0
