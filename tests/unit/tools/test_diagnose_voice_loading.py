"""Latency summaries keep tail delays but export no capture IDs or credentials."""

from tools.diagnose_voice_loading import summarize_timings


def test_latency_summary_keeps_slow_tail_separate_from_normal_preview_reads():
    rows = [{'kind':'review','headers_ms':delay,'body_ms':1.,'json_ms':.1,'total_ms':delay+1.1,
             'bytes':1000,'capture_id':'private-capture','credential':'private-token'}
            for delay in (10.,60.,400.)]
    rows.append({'kind':'captures','headers_ms':20.,'body_ms':2.,'json_ms':.2,'total_ms':22.2,'bytes':2000})
    result = summarize_timings(rows)
    assert result['review']['requests'] == 3
    assert result['review']['headers_ms'] == {'min':10.,'p50':60.,'p95':400.,'max':400.}
    assert result['captures']['requests'] == 1
    assert 'private-capture' not in str(result) and 'private-token' not in str(result)
