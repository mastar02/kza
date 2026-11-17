"""Unit checks of the T046-U1 web timing probe (t046/preregistration.json unit_checks_required).

Each test names the preregistered check it covers. Browser-level harness checks on server headless
Chromium live in tests/frontend/browser_web_timing_probe.py (never a measurement).
"""

import asyncio
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from tools import web_timing_probe as probe
from tools.web_timing_probe import CENSORED, INF, Outcome, RunFacts, Timeout


REPOSITORY = Path(__file__).resolve().parents[3]
SHA = 'a' * 64
VOICES_STEPS = probe.CORE_SCRIPT['#voces']


# ------------------------------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------------------------------

class FakeClient:
    """CDP client double: records commands with their class and phase, answers from a script."""

    def __init__(self, answers=None):
        self.log = []
        self.handlers = defaultdict(list)
        self.phase = 'timed'
        self.disconnected = False
        self.events = []
        self.answers = answers or {}

    def on(self, method, handler):
        self.handlers[method].append(handler)

    async def send(self, method, params=None, *, klass, phase=None):
        if not probe.method_class_allowed(method, klass):
            raise ValueError(f'no_deadline_class:{method}:{klass}')
        self.log.append({'method': method, 'params': params, 'class': klass, 'phase': phase or self.phase,
                         'sent': time.monotonic()})
        answer = self.answers.get(method)
        return answer(params) if callable(answer) else (answer or {})

    async def send_pipelined(self, commands, *, klass):
        return [await self.send(method, params, klass=klass) for method, params in commands]

    async def close(self):
        pass


def make_run(view='#voces', viewport='desktop', capture='vf_' + 'c' * 64):
    spec = probe.RunSpec(7, 'cold', viewport, view, 'http://127.0.0.1:1/' + view, 'http://127.0.0.1:1', None,
                         capture_id=capture)
    browser = SimpleNamespace(client=FakeClient({'Browser.getWindowForTarget': {'bounds': {'windowState': 'normal'}}}),
                              alive=lambda: True, argv=[])
    run = probe.Run(spec, browser, FakeClient(), 'target', Path('/nonexistent'), None, None)
    run.config = probe.injection_config(token=run.token, origin=spec.origin, view=view, capture_id=capture)
    run.filter.committed = True
    run.filter.time_origin = 1000.0
    return run


def record(run, kind, **data):
    item = {'token': run.token, 'timeOrigin': 1000.0, 'type': kind, 't': data.get('time', 0), '_received': 0, **data}
    run.filter.accepted.append(item)
    return item


def inputs(index, first, last, types=('pointerdown', 'pointerup', 'click'), path=('button#b',)):
    times = [first] + [last] * (len(types) - 1)
    return [{'event': name, 'timeStamp': stamp, 'interaction': index, 'path': list(path)}
            for name, stamp in zip(types, times)]


def event(interaction, name, start, duration, target='button#b'):
    return {'entryType': 'event', 'interactionId': interaction, 'name': name, 'startTime': start,
            'duration': duration, 'target': target}


# ------------------------------------------------------------------------------------------
# Percentiles, censoring, failures (statistics.percentiles)
# ------------------------------------------------------------------------------------------

def test_percentiles_nearest_rank_failures_infinite_and_censored_below_measured():
    values = [float(value) for value in range(1, 38)] + [INF, INF, CENSORED]
    ordered = probe.sorted_values(values)
    assert ordered[0] == CENSORED and ordered[-1] == INF
    assert probe.nearest_rank(values, 50) == 19.0   # 20th of 40: censored first, then 1..19
    assert probe.nearest_rank(values, 75) == 29.0   # 30th
    assert probe.nearest_rank(values, 95) == 37.0   # 38th
    assert probe.nearest_rank([float(v) for v in range(1, 40)] + [INF], 95) == 38.0
    assert probe.nearest_rank([float(v) for v in range(1, 38)] + [INF] * 3, 95) == INF
    many = [float(value) for value in range(1, 41)]
    assert probe.precision(many)['p75_interval'] == [25.0, 36.0] and probe.precision(many)['p95_lower'] == 36.0
    assert probe.nearest_rank([10.0, 20.0, 30.0, 40.0], 75) == 30.0  # no interpolation (would be 32.5)


def test_failures_count_as_infinity_in_metric_values():
    runs = [{'outcome': Outcome.VALID, 'role': 'cold', 'metrics': {'lcp': 100.0}},
            {'outcome': Outcome.FAILURE, 'role': 'cold', 'reason': 'load_not_settled', 'metrics': {'lcp': 50.0}},
            {'outcome': Outcome.FAILURE, 'role': 'cold', 'reason': 'cdp_timeout:Input.dispatchMouseEvent',
             'metrics': {'lcp': 60.0}},
            {'outcome': Outcome.POST_WINDOW_HANG, 'role': 'cold', 'metrics': {'lcp': 70.0}},
            {'outcome': Outcome.INVALID, 'role': 'cold', 'metrics': {'lcp': 1.0}}]
    assert probe.metric_values(runs, 'lcp') == [100.0, INF, INF, INF]


def test_failure_scope_per_metric_preview_failure_keeps_lcp_finite():
    """V1: a preview-only failure fails C11 preview, never LCP, CLS or (all steps dispatched) INP."""
    runs = []
    for index in range(40):
        failed = index < 12
        runs.append({'viewport': 'desktop', 'view': '#voces', 'role': 'cold',
                     'outcome': Outcome.FAILURE if failed else Outcome.VALID,
                     'reason': 'preview_failed_open:no_playing' if failed else None,
                     'facts': {'product_failures': ['preview_failed_open:no_playing'] if failed else []},
                     'metrics': {'lcp': 800.0 + index, 'cls': 0.0, 'inp': 40.0,
                                 'preview': {'value': INF if failed else 300.0}}})
    row = probe.summary_rows(runs, None)[0]
    assert row['lcp']['p75'] == 829.0 and row['lcp']['failures'] == 0
    assert row['inp']['failures'] == 0 and row['cls']['failures'] == 0
    assert row['c11_preview']['failures'] == 12 and row['c11_preview']['p75'] == INF
    wide = dict(runs[0], reason='load_not_settled', facts={'product_failures': ['load_not_settled']})
    assert probe.run_wide_failure(wide) and not probe.run_wide_failure(runs[0])
    assert probe.metric_values([wide], 'lcp') == [INF]


def test_warm_ups_primers_and_trace_contrast_enter_no_percentile():
    runs = [{'outcome': Outcome.VALID, 'role': 'cold', 'metrics': {'lcp': 1.0}, 'warm_up': True},
            {'outcome': Outcome.VALID, 'role': 'cold', 'metrics': {'lcp': 2.0}, 'primer': True},
            {'outcome': Outcome.VALID, 'role': 'cold', 'metrics': {'lcp': 3.0}, 'trace_contrast': True},
            {'outcome': Outcome.VALID, 'role': 'warm', 'metrics': {'lcp': 4.0}}]
    assert probe.metric_values(runs, 'lcp') == [4.0]


# ------------------------------------------------------------------------------------------
# INP (metrics.C10_INP, R2-5, R3-1, R4-1, R5-4)
# ------------------------------------------------------------------------------------------

def test_inp_counts_only_interaction_ids_takes_max_per_group_and_maps_by_window():
    interactions = probe.interactions_from_inputs(inputs(0, 100.0, 110.0) + inputs(1, 1200.0, 1210.0))
    entries = [event(0, 'pointermove', 1199.0, 400.0), event(0, 'pointerover', 105.0, 300.0),
               event(5, 'pointerdown', 100.0, 24.0), event(5, 'click', 108.0, 40.0),
               event(9, 'click', 1201.0, 32.0)]
    mapping = probe.map_inp(entries, interactions)
    assert mapping['latencies'] == [40.0, 32.0]
    assert mapping['groups_per_interaction'] == [1, 1]
    assert probe.run_inp(mapping, True) == 40.0


def test_sub_threshold_interaction_is_censored_not_missing():
    interactions = probe.interactions_from_inputs(inputs(0, 100.0, 104.0) + inputs(1, 1200.0, 1204.0))
    mapping = probe.map_inp([event(3, 'click', 1201.0, 24.0)], interactions)
    assert mapping['latencies'] == [CENSORED, 24.0]
    assert probe.run_inp(mapping, True) == 24.0
    only = probe.map_inp([], interactions[:1])
    assert probe.run_inp(only, True) == CENSORED
    assert probe.nearest_rank([CENSORED] * 30 + [20.0] * 10, 75) == CENSORED


def test_partial_group_pointerdown_without_entry_maps_by_capture_window():
    interactions = probe.interactions_from_inputs(inputs(0, 1000.0, 1009.0))
    mapping = probe.map_inp([event(4, 'click', 1008.5, 40.0)], interactions)
    assert mapping['latencies'] == [40.0]
    outside = probe.map_inp([event(4, 'click', 1500.0, 40.0)], interactions)
    assert outside['latencies'] == [CENSORED]
    assert outside['unmapped'][0]['types'] == ['click']


def test_detached_target_null_maps_by_time_and_is_counted():
    interactions = probe.interactions_from_inputs(inputs(0, 1000.0, 1009.0))
    mapping = probe.map_inp([event(4, 'pointerup', 1001.0, 40.0, None), event(4, 'click', 1002.0, 40.0, None)],
                            interactions)
    assert mapping['latencies'] == [40.0]
    assert mapping['diagnostics'][0][0]['target'] == ['null']


def test_play_gesture_has_own_window_with_midpoint_boundary_and_enters_inp():
    # 'Escuchar' last event at 2000.0, play gesture first event 2001.2 (< 2 ms apart: margins overlap).
    interactions = probe.interactions_from_inputs(inputs(0, 1990.0, 2000.0) + inputs(1, 2001.2, 2003.0))
    windows = probe.capture_windows(interactions)
    assert windows[0][1] == windows[1][0] == pytest.approx(2000.6)
    mapping = probe.map_inp([event(1, 'click', 1999.0, 16.0), event(2, 'pointerdown', 2001.0, 72.0)], interactions)
    assert mapping['latencies'] == [16.0, 72.0]
    assert probe.run_inp(mapping, True) == 72.0


def test_missing_or_skipped_step_makes_inp_and_cls_infinite():
    interactions = probe.interactions_from_inputs(inputs(0, 100.0, 110.0))
    mapping = probe.map_inp([event(1, 'click', 101.0, 40.0)], interactions)
    assert probe.run_inp(mapping, False) == INF
    shifts = [{'entryType': 'layout-shift', 'startTime': 10.0, 'value': 0.01, 'hadRecentInput': False}]
    assert probe.cls(shifts, False) == INF
    assert probe.cls(shifts, True) == pytest.approx(0.01)


# ------------------------------------------------------------------------------------------
# LCP, CLS, C11 clocks
# ------------------------------------------------------------------------------------------

def test_lcp_stops_at_first_input_and_cap():
    entries = [{'entryType': 'largest-contentful-paint', 'startTime': 300.0, 'renderTime': 300.0, 'loadTime': 0},
               {'entryType': 'largest-contentful-paint', 'startTime': 900.0, 'renderTime': 0, 'loadTime': 900.0},
               {'entryType': 'largest-contentful-paint', 'startTime': 2500.0, 'renderTime': 2500.0}]
    assert probe.lcp(entries, 2000.0) == 900.0
    assert probe.lcp(entries, None) == 2500.0
    assert probe.lcp(entries, 200.0) == INF
    capped = [{'entryType': 'largest-contentful-paint', 'startTime': 10500.0, 'renderTime': 10500.0}]
    assert probe.lcp(entries[:1] + capped, None) == 300.0


def test_cls_session_windows_ignore_recent_input():
    shifts = [{'entryType': 'layout-shift', 'startTime': t, 'value': v, 'hadRecentInput': h}
              for t, v, h in [(0, .05, False), (500, .05, False), (2000, .2, True), (3000, .03, False), (3500, .01, False)]]
    assert probe.cls(shifts, True) == pytest.approx(0.10)


def test_c11_start_is_capture_listener_timestamp_and_sub_threshold_gives_feedback():
    steps = [{'step': 'accion', 'c11': 'action_feedback', 'end': {'kind': 'nonempty_attribute'}}]
    interactions = probe.interactions_from_inputs(inputs(0, 1000.0, 1004.0))
    typed = {'step_end': [{'index': 0, 'time': 1006.0}], 'step_frame': [{'index': 0, 'raf': 1012.0}],
             'perf': [{'entry': event(3, 'pointerdown', 990.0, 8.0)}]}
    trace = {'offset': 50.0, 'candidates': {'DrawFrame': [{'begin': 1070.0, 'time': 1070.0}]}}
    metrics = probe.c11_metrics(typed, interactions, steps, trace)
    assert metrics['action_feedback']['start'] == 1000.0
    assert metrics['action_feedback_raf'] == 12.0
    assert metrics['action_feedback']['presented']['DrawFrame'] == 1020.0
    run = {'outcome': Outcome.VALID, 'metrics': metrics}
    assert probe.c11_feedback_value(run, 'DrawFrame') == 20.0
    assert probe.run_inp(probe.map_inp([], interactions), True) == CENSORED


def test_preview_primary_end_is_playing_on_every_run():
    value = probe.preview(1000.0, 1300.0, 1305.0, 1400.0)
    assert value['value'] == 395.0 and value['unsubtracted'] == 400.0 and value['gap'] == 5.0
    steps = VOICES_STEPS
    interactions = probe.interactions_from_inputs(inputs(0, 10, 12) + inputs(1, 20, 22) + inputs(2, 1000, 1002)
                                                  + inputs(3, 1305, 1307))
    media = [{'event': 'loadedmetadata', 'timeStamp': 1310.0, 'readyState': 1},
             {'event': 'canplay', 'timeStamp': 1320.0, 'readyState': 4},
             {'event': 'playing', 'timeStamp': 1400.0, 'readyState': 4}]
    typed = {'control_attached': [{'type': 'control_attached', 'time': 1300.0, 'readyState': 0,
                                   'target': {'kind': 'button'}}], 'media': media}
    metrics = probe.c11_metrics(typed, interactions, steps, None)
    assert metrics['preview']['value'] == 395.0
    assert metrics['preview']['secondary']['canplay'] == 1320.0
    without_playing = dict(typed, media=media[:2])
    assert probe.c11_metrics(without_playing, interactions, steps, None)['preview']['value'] == INF


def test_preview_gap_over_16_ms_is_invalid_and_secondary_values_recorded():
    value = probe.preview(1000.0, 1300.0, 1317.0, 1400.0)
    assert value['gap_invalid'] and value['gap'] == 17.0 and value['unsubtracted'] == 400.0
    assert value['value'] == 383.0
    assert not probe.preview(1000.0, 1300.0, 1316.0, 1400.0)['gap_invalid']
    facts = RunFacts(navigate_sent=True, environment_faults=['preview_gap_over_16ms'])
    assert probe.classify(facts)[0] == Outcome.INVALID
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#voces', 1, pairs=2, contrast_pairs=0, warm_ups=0)
    added = probe.replacement_slots(plan, plan['slots'][0])
    assert added[0]['ordinal'] == 4 and added[0]['role'] == 'cold'


# ------------------------------------------------------------------------------------------
# Run classification (cdp_deadlines.rule, exclusion_rules)
# ------------------------------------------------------------------------------------------

def test_timed_window_timeout_failure_live_browser_invalid_when_crashed_or_disconnected():
    timed = [Timeout('timed', 'evaluate', 'Runtime.evaluate')]
    assert probe.classify(RunFacts(timeouts=timed, navigate_sent=True))[0] == Outcome.FAILURE
    assert probe.classify(RunFacts(timeouts=timed, crashed=True))[0] == Outcome.INVALID
    assert probe.classify(RunFacts(timeouts=timed, disconnected=True))[0] == Outcome.INVALID


def test_timeout_by_phase_setup_and_collection_invalid_timed_failure():
    for phase in ('setup', 'collect'):
        facts = RunFacts(timeouts=[Timeout(phase, 'collect', 'Tracing.end')])
        assert probe.classify(facts)[0] == Outcome.INVALID
    for method, klass in (('Runtime.evaluate', 'evaluate'), ('Input.dispatchMouseEvent', 'input'),
                          ('Tracing.start', 'interaction_trace')):
        assert probe.classify(RunFacts(timeouts=[Timeout('timed', klass, method)]))[0] == Outcome.FAILURE


def test_hang_spanning_phases_first_timeout_wins():
    facts = RunFacts(timeouts=[Timeout('timed', 'evaluate', 'Runtime.evaluate'), Timeout('collect', 'collect', 'Tracing.end'),
                               Timeout('leave', 'leave', 'Page.navigate')], leave_ok=False)
    assert probe.classify(facts) == (Outcome.FAILURE, 'cdp_timeout:Runtime.evaluate')


def test_post_window_hang_from_leave_step_and_phase3_with_busy_trace():
    assert probe.classify(RunFacts(timeouts=[Timeout('leave', 'leave', 'Page.navigate')], leave_ok=False))[0] \
        == Outcome.POST_WINDOW_HANG
    busy = RunFacts(timeouts=[Timeout('collect', 'collect', 'Tracing.end')], leave_ok=True,
                    renderer_busy_across_deadline=True)
    assert probe.classify(busy)[0] == Outcome.POST_WINDOW_HANG
    for evidence in (False, None):
        lost = RunFacts(timeouts=[Timeout('collect', 'collect', 'Tracing.end')], leave_ok=True,
                        renderer_busy_across_deadline=evidence)
        assert probe.classify(lost)[0] == Outcome.INVALID


def test_renderer_busy_across_deadline_from_trace():
    events = [{'ph': 'M', 'name': 'thread_name', 'pid': 1, 'tid': 2, 'args': {'name': 'CrRendererMain'}},
              {'ph': 'X', 'name': 'FunctionCall', 'pid': 1, 'tid': 2, 'ts': 1_000_000, 'dur': 9_000_000}]
    assert probe.renderer_busy_across(events, 2000.0, 9000.0) is True
    assert probe.renderer_busy_across(events, 500.0, 9000.0) is False
    assert probe.renderer_busy_across([], 2000.0, 9000.0) is None


def test_visibility_or_focus_loss_is_invalid_not_failure():
    start = {'focus': {'visibility': 'visible', 'hasFocus': True, 'time': 1.0}}
    later = [{'point': 'domcontentloaded', 'visibility': 'visible', 'hasFocus': True, 'time': 50.0}]
    normal = [{'point': 'navigation', 'state': 'normal'}]
    assert probe.visibility_faults(start, later, [], normal) == []
    assert probe.visibility_faults(start, later, [{'event': 'visibilitychange'}], normal) == ['visibility_event']
    assert probe.visibility_faults(start, [dict(later[0], hasFocus=False)], [], normal) == ['page_unfocused']
    assert probe.visibility_faults(start, later, [], [{'point': 'flush', 'state': 'minimized'}]) == ['window_state']
    assert probe.visibility_faults(start, later, [], [{'point': 'flush', 'state': None, 'error': 'CdpTimeout'}]) \
        == ['window_state']
    assert probe.visibility_faults(dict(start, focus=dict(start['focus'], visibility='hidden')), later, [], normal) \
        == ['page_hidden']
    facts = RunFacts(navigate_sent=True, environment_faults=['window_state'], product_failures=['load_not_settled'])
    assert probe.classify(facts)[0] == Outcome.INVALID


def test_viewport_mismatch_invalid_mobile_growth_flagged():
    start = {'screen': {'width': 1440, 'height': 900}, 'dpr': 2}
    good = [{'point': 'settled', 'innerWidth': 1440, 'innerHeight': 900, 'dpr': 2}]
    assert probe.viewport_check('desktop', start, good) == ([], [])
    assert 'screen_size' in probe.viewport_check('desktop', dict(start, screen={'width': 1280, 'height': 800}), good)[0]
    assert 'device_pixel_ratio' in probe.viewport_check('desktop', start, [dict(good[0], dpr=1)])[0]
    assert 'inner_size' in probe.viewport_check('desktop', start, [dict(good[0], innerWidth=1425)])[0]
    mobile_start = {'screen': {'width': 390, 'height': 844}, 'dpr': 3}
    faults, flags = probe.viewport_check('mobile', mobile_start, [{'point': 'flush', 'innerWidth': 480,
                                                                     'innerHeight': 1038, 'dpr': 3}])
    assert faults == [] and flags[0]['flag'] == 'viewport_ensanchado'


def test_warm_after_failed_cold_runs_and_slot_invalid_only_without_load():
    assert not probe.warm_slot_invalid({'navigate_sent': True, 'reason': 'load_not_settled'})
    assert probe.warm_slot_invalid({'navigate_sent': False, 'reason': 'activation_failed'})
    assert probe.warm_slot_invalid({'navigate_sent': True, 'reason': 'browser_crashed'})


# ------------------------------------------------------------------------------------------
# Record binding (R5-1)
# ------------------------------------------------------------------------------------------

def test_record_binding_rejects_stale_token_time_origin_and_pre_commit():
    current = probe.RecordFilter('new')
    assert not current.offer({'token': 'old', 'type': 'settled', 'timeOrigin': 5.0})
    current.navigation_sent('loader-1')
    current.frame_navigated({'loaderId': 'other', 'url': 'about:blank'})
    assert not current.offer({'token': 'new', 'type': 'document_start', 'timeOrigin': 9.0})
    current.frame_navigated({'loaderId': 'loader-1', 'url': 'http://127.0.0.1/'})
    assert current.offer({'token': 'new', 'type': 'document_start', 'timeOrigin': 10.0})
    assert current.offer({'token': 'new', 'type': 'settled', 'timeOrigin': 10.0})
    assert not current.offer({'token': 'new', 'type': 'settled', 'timeOrigin': 11.0})
    assert not current.offer({'token': 'old', 'type': 'settled', 'timeOrigin': 10.0})
    assert [item['reason'] for item in current.rejected] == ['token', 'before_commit', 'time_origin', 'token']
    assert [item['record']['token'] for item in current.rejected][0] == 'old'


def test_reused_target_removes_previous_injection_before_registering():
    commands = probe.setup_commands(role='warm', viewport='desktop', script_source='S', previous_script='7')
    methods = [method for method, _ in commands]
    assert methods.index('Page.removeScriptToEvaluateOnNewDocument') < methods.index('Page.addScriptToEvaluateOnNewDocument')
    assert dict(commands)['Page.removeScriptToEvaluateOnNewDocument'] == {'identifier': '7'}
    assert methods.count('Page.addScriptToEvaluateOnNewDocument') == 1
    fresh = probe.setup_commands(role='cold', viewport='desktop', script_source='S', previous_script=None)
    assert 'Page.removeScriptToEvaluateOnNewDocument' not in [method for method, _ in fresh]


def test_injected_source_embeds_token_and_origin_per_run():
    first = probe.injection_config(token='t1', origin='http://127.0.0.1:5', view='#voces', capture_id='vf_x')
    second = probe.injection_config(token='t2', origin='http://127.0.0.1:5', view='#voces', capture_id='vf_y')
    assert first['flush'] != second['flush'] and '"t1"' in probe.injected_source(first)
    assert probe.injected_source(first).index('location.origin !== CONFIG.origin') < probe.injected_source(first).index('deliver')
    assert first['steps'][1]['selector'] == '[data-capture-id="vf_x"] input[type="checkbox"]'


# ------------------------------------------------------------------------------------------
# Load settle under node (R4-4, R5-3, R5-11)
# ------------------------------------------------------------------------------------------

NODE_SIMULATION = r"""
const SOURCE = %s;
const QUIET = %d;
const factory = new Function(SOURCE + '; return {createSettle, instrumentRequests};');
const {createSettle, instrumentRequests} = factory();
async function simulate(timeline) {
  let now = 0, serial = 0, settled = null, loadAt = null;
  const timers = [];
  const settle = createSettle({quietMs: QUIET, lcpCapMs: 10000, now: () => now,
    setTimer: (fn, delay) => { timers.push({at: now + delay, fn, id: ++serial}); return serial; },
    clearTimer: id => { const k = timers.findIndex(t => t.id === id); if (k >= 0) timers.splice(k, 1); },
    takeRecords: () => {}, loadGateHolds: T => loadAt !== null && loadAt <= T, onSettled: data => { settled = data; }});
  class Target { constructor() { this.listeners = {}; } addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
    fire(type, stamp) { for (const fn of this.listeners[type] || []) fn({type, timeStamp: stamp}); } }
  const sockets = [], fetches = [];
  const win = {location: {href: 'http://127.0.0.1:9/'}, fetch: () => new Promise(resolve => fetches.push(resolve)),
    XMLHttpRequest: class extends Target { open() {} send() {} }, WebSocket: class extends Target { constructor(url) { super(); this.url = url; sockets.push(this); } }};
  instrumentRequests(win, settle, () => now, () => {});
  const events = timeline.slice().sort((a, b) => a.t - b.t);
  while (settled === null) {
    const nextTimer = timers.length ? Math.min(...timers.map(t => t.at)) : Infinity;
    const nextEvent = events.length ? events[0].t : Infinity;
    if (nextTimer === Infinity && nextEvent === Infinity) break;
    if (nextEvent <= nextTimer) {
      const item = events.shift(); now = item.t;
      if (item.op === 'load') { loadAt = now; settle.load(now); }
      else if (item.op === 'lcp') settle.reset(item.start === undefined ? now : item.start, 'lcp');
      else if (item.op === 'shift') settle.reset(item.start, 'layout_shift');
      else if (item.op === 'resource') settle.reset(item.end, 'resource');
      else if (item.op === 'fetch') win.fetch('/api/x');
      else if (item.op === 'fetch_end') fetches.shift()({});
      else if (item.op === 'socket') new win.WebSocket('ws://127.0.0.1:9/ws/voices');
      else if (item.op === 'message') sockets[0].fire('message', item.stamp === undefined ? now : item.stamp);
      for (let i = 0; i < 5; i++) await Promise.resolve();
    } else {
      const timer = timers.find(t => t.at === nextTimer); timers.splice(timers.indexOf(timer), 1); now = timer.at; timer.fn();
    }
  }
  return settled;
}
(async () => {
  const cases = JSON.parse(process.argv[process.argv.length - 1]);
  const results = [];
  for (const timeline of cases) { const settled = await simulate(timeline); results.push(settled ? settled.T : null); }
  console.log(JSON.stringify(results));
})();
"""


def node_settle(cases: list) -> list:
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    script = NODE_SIMULATION % (json.dumps(probe.settle_source()), probe.QUIET_PERIOD_MS)
    result = subprocess.run([node, '-e', script, json.dumps(cases)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_load_settle_synthetic_timelines():
    cases = [
        # load at 300, LCP at 400, fetch ending at 700 -> 1200.
        [{'t': 100, 'op': 'fetch'}, {'t': 300, 'op': 'load'}, {'t': 400, 'op': 'lcp'}, {'t': 700, 'op': 'fetch_end'}],
        # fetch ending at 200 and load at 900 -> 900 (load only gates; 1400 would mean load resets).
        [{'t': 50, 'op': 'fetch'}, {'t': 200, 'op': 'fetch_end'}, {'t': 900, 'op': 'load'}],
        # WebSocket at 100, load at 300, first message at 900, data fetch 950..1100 -> 1600.
        [{'t': 100, 'op': 'socket'}, {'t': 300, 'op': 'load'}, {'t': 900, 'op': 'message'},
         {'t': 950, 'op': 'fetch'}, {'t': 1100, 'op': 'fetch_end'}],
        # Named timestamps: LCP startTime 400 delivered at 450; layout shift startTime 430 delivered at 490.
        [{'t': 300, 'op': 'load'}, {'t': 450, 'op': 'lcp', 'start': 400}, {'t': 490, 'op': 'shift', 'start': 430}],
        # Resource at responseEnd 500 delivered at 700; socket message timeStamp 520 delivered at 720.
        [{'t': 50, 'op': 'socket'}, {'t': 300, 'op': 'load'}, {'t': 700, 'op': 'resource', 'end': 500},
         {'t': 720, 'op': 'message', 'stamp': 520}],
        # LCP after the 10 s cap neither counts nor resets.
        [{'t': 100, 'op': 'fetch'}, {'t': 300, 'op': 'load'}, {'t': 550, 'op': 'lcp', 'start': 10500},
         {'t': 600, 'op': 'fetch_end'}],
    ]
    assert node_settle(cases) == [1200, 900, 1600, 930, 1020, 1100]
    assert probe.QUIET_PERIOD_MS == 500


def test_load_not_settled_is_failure_for_every_metric():
    facts = RunFacts(navigate_sent=True, product_failures=['load_not_settled'])
    assert probe.classify(facts) == (Outcome.FAILURE, 'load_not_settled')
    run = {'outcome': Outcome.FAILURE, 'reason': 'load_not_settled', 'role': 'cold',
           'metrics': {'lcp': 10.0, 'cls': 0.0, 'inp': 8.0}}
    assert [probe.metric_values([run], name) for name in ('lcp', 'cls', 'inp')] == [[INF], [INF], [INF]]


def test_injected_source_does_nothing_on_about_blank_or_data_urls():
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    config = probe.injection_config(token='t', origin='http://127.0.0.1:9', view='#hogar', capture_id=None)
    source = probe.injected_source(config)
    script = ("const calls=[];globalThis.window={kzaProbeRecord:v=>calls.push(v)};"
              "for (const origin of ['null','data:']){globalThis.location={origin,href:'about:blank'};" + source + "}"
              "console.log(JSON.stringify({calls:calls.length,flush:Object.keys(globalThis.window).length}))")
    result = subprocess.run([node, '-e', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'calls': 0, 'flush': 1}


# ------------------------------------------------------------------------------------------
# CDP client: absolute deadlines and deadline classes (R2-8, R5-5, R5-13)
# ------------------------------------------------------------------------------------------

class ChattyConnection:
    """Websocket double that streams events every 100 ms and never answers commands."""

    def __init__(self, answer=False):
        self.sent = []
        self.answer = answer
        self.queue = asyncio.Queue()

    async def send(self, text):
        self.sent.append(json.loads(text))
        if self.answer:
            await self.queue.put(json.dumps({'id': self.sent[-1]['id'], 'result': {'ok': True}}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self.queue.get(), 0.1)
        except asyncio.TimeoutError:
            return json.dumps({'method': 'Network.dataReceived', 'params': {}})

    async def close(self):
        pass


async def test_cdp_deadline_is_absolute_from_send_and_events_never_extend_it():
    client = probe.CdpClient(ChattyConnection(), name='browser')
    started = time.monotonic()
    with pytest.raises(probe.CdpTimeout) as raised:
        async with asyncio.timeout(5):
            await client.send('Browser.getVersion', klass='version')
    elapsed = time.monotonic() - started
    assert 2.0 <= elapsed < 2.6 and raised.value.klass == 'version'
    assert len(client.events) >= 10
    assert client.log[-1]['timeout'] and client.log[-1]['deadline'] - client.log[-1]['sent'] == pytest.approx(2.0)
    await client.close()
    answered = probe.CdpClient(ChattyConnection(answer=True), name='page')
    assert await answered.send('Page.enable', klass='setup') == {'ok': True}
    await answered.close()


def test_frozen_deadline_values_and_classes():
    assert probe.DEADLINE_MS == {'setup': 10000, 'interaction_trace': 10000, 'navigate': 15000, 'load_wait': 15000,
                                 'evaluate': 10000, 'input': 5000, 'wait_feedback': 5000, 'wait_preview': 10000,
                                 'wait_other': 5000, 'collect': 60000, 'leave': 10000, 'version': 2000,
                                 'window_state': 10000}
    assert probe.HARNESS_ONLY_CLASSES == {'setup', 'collect', 'version', 'window_state'}
    assert probe.IMMUTABLE_CLASSES.isdisjoint(probe.HARNESS_ONLY_CLASSES)


async def test_client_refuses_methods_without_deadline_class_and_dom_or_accessibility():
    client = probe.CdpClient(ChattyConnection(answer=True), name='page')
    for method, klass in (('DOM.getDocument', 'setup'), ('Accessibility.getFullAXTree', 'setup'),
                          ('Page.captureScreenshot', 'collect'), ('Runtime.evaluate', 'input')):
        with pytest.raises(ValueError, match='no_deadline_class'):
            await client.send(method, klass=klass)
    assert client.log == []
    await client.close()


def test_harness_only_formula_and_immutable_deadlines():
    usage = {'setup': [4200.0, 6100.0], 'collect': [1000.0], 'version': [100.0],
             'navigate': [14000.0], 'wait_preview': [9000.0]}
    updated = probe.harness_only_deadlines(probe.DEADLINE_MS, usage)
    assert updated['setup'] == 13000 and updated['collect'] == 60000 and updated['version'] == 2000
    for klass in probe.IMMUTABLE_CLASSES:
        assert updated[klass] == probe.DEADLINE_MS[klass]


def test_background_branch_and_warm_cache_fallback_follow_preregistered_rules():
    loopback = [{'requests': ['http://127.0.0.1:5000/', 'ws://localhost:1/x']}] * 5
    assert probe.background_branch(loopback) == 'none'
    assert probe.background_branch(loopback[:4] + [{'requests': ['https://clients2.google.com/x']}]) == 'netlog'
    assert probe.warm_cache_outcome(0.95) is None
    assert probe.warm_cache_outcome(0.89) == 'no evaluable por este método: caché caliente no alcanzada'


# ------------------------------------------------------------------------------------------
# Warm runs, cache share, batches (R2-9, R4-7, R4-11, R5-7)
# ------------------------------------------------------------------------------------------

def test_warm_runs_enable_cache_cold_runs_disable_and_clear():
    warm = dict(probe.setup_commands(role='warm', viewport='desktop', script_source='S', previous_script=None))
    assert warm['Network.setCacheDisabled'] == {'cacheDisabled': False} and 'Network.clearBrowserCache' not in warm
    cold = dict(probe.setup_commands(role='cold', viewport='mobile', script_source='S', previous_script=None))
    assert cold['Network.setCacheDisabled'] == {'cacheDisabled': True} and 'Network.clearBrowserCache' in cold
    assert cold['Emulation.setDeviceMetricsOverride'] == {'width': 390, 'height': 844, 'deviceScaleFactor': 3,
                                                          'mobile': True, 'screenWidth': 390, 'screenHeight': 844}
    assert cold['Emulation.setTouchEmulationEnabled']['enabled'] is True


def batch(runs, planned=100, **extra):
    return {'runs': runs, 'planned': planned, 'browser_version': 'Chrome/157', **extra}


def test_warm_share_per_batch_below_90_is_invalid_batch_and_mix_reported():
    warm = [{'role': 'warm', 'outcome': Outcome.VALID, 'cache_sources': {'disk': 8, 'network': 2}}]
    assert probe.warm_share(warm)['mix_percent'] == {'memory': 0.0, 'disk': 80.0, '304': 0.0, 'network': 20.0}
    assert 'warm_share_below_90' in probe.invalid_batch_reasons(batch(warm))
    good = [{'role': 'warm', 'outcome': Outcome.VALID, 'cache_sources': {'memory': 5, 'disk': 4, '304': 1}}]
    assert probe.invalid_batch_reasons(batch(good)) == []


def test_invalid_batch_counts_invalid_runs_only_never_failures():
    failures = [{'role': 'cold', 'outcome': Outcome.FAILURE} for _ in range(30)]
    invalid = [{'role': 'cold', 'outcome': Outcome.INVALID} for _ in range(10)]
    assert probe.invalid_batch_reasons(batch(failures + invalid)) == []
    assert probe.invalid_batch_reasons(batch(failures + invalid + invalid[:1])) == ['invalid_runs_over_10_percent']
    warm_up_invalid = [{'role': 'cold', 'outcome': Outcome.INVALID, 'warm_up': True} for _ in range(20)]
    assert probe.invalid_batch_reasons(batch(warm_up_invalid)) == []


def test_chrome_version_per_run_differing_from_header_is_invalid_batch():
    runs = [{'role': 'cold', 'outcome': Outcome.VALID, 'browser_version': 'Chrome/157'},
            {'role': 'cold', 'outcome': Outcome.VALID, 'browser_version': 'Chrome/158'}]
    assert 'browser_version_changed' in probe.invalid_batch_reasons(batch(runs))


def test_dirty_tree_lifetime_and_quiet_failures_invalidate_batch(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('ignored/\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.gitignore'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'x'],
                   check=True)
    (tmp_path / 'ignored').mkdir()
    (tmp_path / 'ignored' / 'out.json').write_text('{}')
    assert not probe.tree_dirty(tmp_path)
    (tmp_path / 'untracked.js').write_text('x')
    assert probe.tree_dirty(tmp_path)
    assert probe.invalid_batch_reasons(batch([], tree_dirty=True)) == ['served_tree']
    assert probe.invalid_batch_reasons(batch([], elapsed_since_fixture_s=10 * 3600 + 1)) == ['fixture_lifetime']
    assert probe.invalid_batch_reasons(batch([], quiet_end_ok=False)) == ['quiet_machine']


def test_seeds_derive_from_frozen_sha_cell_and_attempt():
    expected = int.from_bytes(__import__('hashlib').sha256(f'{SHA}:batch:mac-chrome:desktop:#hogar:1'.encode())
                              .digest()[:8], 'big')
    first = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 1)
    assert first['seed'] == expected
    assert probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 1)['slots'] == first['slots']
    assert probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 2)['slots'] != first['slots']
    order_seed = int.from_bytes(__import__('hashlib').sha256(f'{SHA}:order:mac-chrome:1'.encode()).digest()[:8], 'big')
    assert probe.derive_seed(SHA, 'order', 'mac-chrome', 1) == order_seed
    slots = first['slots']
    assert [slot.get('warm_up') for slot in slots[:3]] == [True] * 3 and len(slots) == 103 and first['planned'] == 100
    for index in range(3, len(slots), 2):
        assert slots[index]['role'] == 'cold' and slots[index + 1]['role'] == 'warm'
        assert slots[index]['pair'] == slots[index + 1]['pair']
    assert sum(1 for slot in slots if slot.get('trace_contrast')) == 20


def test_replacements_appended_in_invalidation_order_with_continued_ordinals():
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 1, pairs=3, contrast_pairs=0)
    original = [dict(slot) for slot in plan['slots']]
    warm = next(slot for slot in plan['slots'] if slot['role'] == 'warm')
    cold = next(slot for slot in plan['slots'] if slot['role'] == 'cold' and not slot.get('warm_up'))
    first = probe.replacement_slots(plan, warm)
    second = probe.replacement_slots(plan, cold)
    assert plan['slots'][:len(original)] == original
    assert [slot['ordinal'] for slot in first + second] == [9, 10, 11]
    assert first[0]['primer'] and first[0]['role'] == 'cold' and first[1]['role'] == 'warm'
    assert second[0]['role'] == 'cold' and not second[0].get('primer')
    runs = [{'outcome': Outcome.VALID, 'role': 'cold', 'primer': True, 'metrics': {'lcp': 1.0}}]
    assert probe.metric_values(runs, 'lcp') == []


class BatchFakes:
    def __init__(self, outcomes=None, start_ok=True, end_load=0.5):
        self.outcomes = outcomes or {}
        self.events = []
        self.start_ok = start_ok
        self.end_load = end_load
        self.readings = 0

    async def quiet(self):
        return {'quiet': self.start_ok, 'covariates': {'machine_state': 'shared', 'gating': False,
                                                       'counts_by_class': {'chrome': 0 if self.start_ok else 1}}}

    def loadavg(self):
        self.readings += 1
        return 0.5 if self.readings == 1 else self.end_load

    async def sleep(self, seconds):
        self.events.append(f'sleep:{seconds}')

    async def start_fixture(self):
        return {'digest': 'd', 'started_at': time.time(), 'seed': 46001, 'origin': 'http://x', 'tokens': {}}

    async def stop_fixture(self):
        pass

    async def run_slot(self, slot):
        outcome, reason, navigated = self.outcomes.get(slot['ordinal'], (Outcome.VALID, None, True))
        return {'ordinal': slot['ordinal'], 'outcome': outcome, 'reason': reason, 'navigate_sent': navigated,
                'browser_version': 'Chrome/157', 'cache_sources': {'disk': 10}}

    async def close(self):
        self.events.append('close_browser')

    def environment(self):
        return probe.BatchEnvironment(quiet_check=self.quiet, loadavg=self.loadavg, sleep=self.sleep,
                                      start_fixture=self.start_fixture, stop_fixture=self.stop_fixture,
                                      run_slot=self.run_slot, close_browser=self.close,
                                      caffeinate=lambda: None, served_state=lambda: {'commit': 'c', 'dirty': False},
                                      events=self.events)


async def test_batch_load_readings_order_thresholds_and_warm_after_failed_cold():
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 1, pairs=3, contrast_pairs=0, warm_ups=1)
    cold_failed = next(slot['ordinal'] for slot in plan['slots'] if slot['role'] == 'cold' and not slot.get('warm_up'))
    fakes = BatchFakes({cold_failed: (Outcome.FAILURE, 'load_not_settled', True)})
    result = await probe.run_batch(plan, fakes.environment(), idle=0.2, post_batch=0.3)
    events = fakes.events
    assert events.index('load_start') < events.index('fixture_start') < events.index(f'run:{plan["slots"][0]["ordinal"]}')
    assert events.index('fixture_stop') < events.index('wait_60s') < events.index('sleep:60') < events.index('load_end')
    assert f'run:{cold_failed + 1}' in events
    warm = next(run for run in result['runs'] if run['ordinal'] == cold_failed + 1)
    assert warm['outcome'] == Outcome.VALID
    assert not result['discarded']
    # quiet_machine.shared: a load above the old thresholds is recorded at both points and never discards.
    high = BatchFakes(end_load=1.31)
    recorded = await probe.run_batch(probe.plan_batch(SHA, 'm', 'desktop', '#hogar', 1, pairs=1, contrast_pairs=0,
                                                      warm_ups=0), high.environment(), idle=0.2, post_batch=0.3)
    assert recorded['reasons'] == [] and recorded['end']['covariates']['load_1m'] == 1.31
    assert recorded['end']['covariates']['reference_load'] == 0.3
    early = BatchFakes()
    early.loadavg = lambda: 1.21
    ran = await probe.run_batch(plan, early.environment(), idle=0.2, post_batch=0.3)
    assert not ran['discarded'] and 'fixture_start' in early.events
    assert ran['start']['covariates']['load_1m'] == 1.21 and ran['start']['covariates']['reference_load'] == 0.2


async def test_warm_slot_without_load_is_invalid_and_replaced_by_primer_pair():
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#hogar', 1, pairs=2, contrast_pairs=0, warm_ups=0)
    fakes = BatchFakes({0: (Outcome.INVALID, 'activation_failed', False)})
    result = await probe.run_batch(plan, fakes.environment(), idle=1.0, post_batch=1.0)
    by_ordinal = {run['ordinal']: run for run in result['runs']}
    assert by_ordinal[1]['reason'] == 'warm_slot_without_preceding_load' and 'run:1' not in fakes.events
    appended = [run for run in result['runs'] if run.get('replaces') is not None]
    assert [(run['ordinal'], run['role'], bool(run.get('primer'))) for run in appended] == \
        [(4, 'cold', False), (5, 'cold', True), (6, 'warm', False)]


def test_two_consecutive_discarded_batches_stop_the_probe():
    ledger = probe.BatchLedger()
    ledger.record({'discarded': True})
    ledger.record({'discarded': False})
    ledger.record({'discarded': True})
    assert ledger.may_start()
    ledger.record({'discarded': True})
    assert not ledger.may_start()


def test_load_readings_are_parsed_and_have_no_threshold():
    """quiet_machine.shared: the load averages are covariates; no load threshold exists any more."""
    assert not hasattr(probe, 'load_reading_ok')
    assert probe.parse_loadavg('{ 1.52 1.40 1.33 }') == 1.52
    assert probe.parse_loadavgs('{ 1.52 1.40 1.33 }') == {'1m': 1.52, '5m': 1.40, '15m': 1.33}


def test_process_check_exempts_own_ancestry_but_not_other_agents():
    rows = [{'pid': 1, 'ppid': 0, 'command': '/sbin/launchd'},
            {'pid': 10, 'ppid': 1, 'command': '/usr/local/bin/claude --resume'},
            {'pid': 11, 'ppid': 10, 'command': '/bin/zsh -c python -m tools.web_timing_probe measure-cell'},
            {'pid': 12, 'ppid': 11, 'command': 'python -m tools.web_timing_probe measure-cell'},
            {'pid': 20, 'ppid': 1, 'command': '/usr/local/bin/claude'},
            {'pid': 21, 'ppid': 20, 'command': '/bin/zsh -c ls'}]
    assert probe.process_check(rows, 12) == [('agent_worker', 20)]
    assert probe.process_check(rows[:4], 12) == []
    extra = rows[:4] + [{'pid': 30, 'ppid': 1, 'command': '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'},
                        {'pid': 31, 'ppid': 1, 'command': 'python3 -m pytest tests'}]
    assert {kind for kind, _ in probe.process_check(extra, 12)} == {'chrome', 'pytest'}
    assert probe.process_check(rows[:4] + [{'pid': 40, 'ppid': 1, 'command': 'mds_stores'}], 12, cpu={40: 25.0}) \
        == [('cpu_over_10_percent', 40)]


# ------------------------------------------------------------------------------------------
# Plans, scope, samples (Q29, QF, R2-6, R4-5)
# ------------------------------------------------------------------------------------------

def test_no_obs_url_and_no_c11_clock_on_obs():
    cells = probe.batch_plan_views()
    assert all('/obs' not in cell['url_path'] for cell in cells)
    assert len(cells) == 12
    assert {cell['view'] for cell in cells if cell['c11']} == {'#acciones', '#voces'}
    assert '/obs' not in json.dumps(probe.CORE_SCRIPT)


def test_preview_samples_follow_permutation_and_consecutive_runs_differ():
    permutation = [3, 1, 4, 0, 2]
    assert [probe.preview_position(ordinal, permutation) for ordinal in range(6)] == [3, 1, 4, 0, 2, 3]
    full = list(range(50))
    __import__('random').Random(1).shuffle(full)
    opened = [probe.preview_position(ordinal, full) for ordinal in range(120)]
    assert all(left != right for left, right in zip(opened, opened[1:]))
    assert opened[:50] == full


def test_role_to_view_mapping_matches_state_js_and_role_permissions():
    from src.dashboard.authority import DashboardRole, _permissions
    state = (REPOSITORY / 'src/dashboard/frontend/home/nexo/state.js').read_text()
    app = (REPOSITORY / 'src/dashboard/frontend/home/nexo/app.js').read_text()
    views = {name: re.findall(r"'([a-z:]+)'", permissions) for name, permissions in
             re.findall(r"(\w+): \[([^\]]*)\]", re.search(r'const VIEWS = \{([^}]*)\}', state).group(1))}
    hashes = dict(re.findall(r"(\w+): '(#[a-z-]+)'", re.search(r'const viewHashes = \{([^}]*)\}', app).group(1)))
    roles = {'operator': _permissions(DashboardRole.OPERATOR, None), 'home': _permissions(DashboardRole.HOME, 'person'),
             'evaluator': _permissions(DashboardRole.EVALUATOR, None)}
    special = {'evaluation': 'evaluator', 'access': 'operator', 'mymemory': 'home'}
    for view, permissions in views.items():
        role = probe.VIEW_ROLES[hashes[view]]
        assert set(permissions) <= roles[role], view
        assert special.get(view, role) == role
    assert set(probe.VIEW_HASHES) == set(hashes.values())


def test_no_performance_observer_in_product_frontend_outside_vendor():
    root = REPOSITORY / 'src/dashboard/frontend'
    offenders = [str(path) for path in root.rglob('*') if path.is_file() and 'vendor' not in path.parts
                 and path.suffix in ('.js', '.html', '.mjs')
                 and re.search(r'PerformanceObserver|largest-contentful-paint|web-vitals|layout-shift', path.read_text())]
    assert offenders == []


# ------------------------------------------------------------------------------------------
# Before/after (R2-4, R4-12, R5-12)
# ------------------------------------------------------------------------------------------

IDENTITY = {'preregistration_sha256': SHA, 'fixture_digest': 'd', 'script_id': 's', 'browser_version': 'Chrome/157',
            'launch_flags': list(probe.LAUNCH_FLAGS), 'launch_mechanism': 'open -na', 'os_version': '26.6.1',
            'hardware': 'Mac16,8', 'ui_locale': 'es-UY', 'pip_freeze_sha256': 'p', 'machine_state': 'shared'}


def test_before_after_closure_identity_and_extra_rows():
    baseline = {'identity': IDENTITY, 'closure': ['src/memory/fact_ledger.py', 'tests/frontend/browser_support.py',
                                                   'requirements.txt']}
    assert probe.before_after(baseline, dict(IDENTITY), ['docs/readme.md'])['decision'] == 'use_baseline'
    for path in ('src/dashboard/frontend/home/nexo/app.js', 'src/memory/fact_ledger.py', 'requirements.txt',
                 'tests/frontend/browser_support.py', 'tests/mocks/memory_sources.py', 'src/users/voice_ledger.py',
                 'tools/web_timing_probe.py'):
        decision = probe.before_after(baseline, dict(IDENTITY), [path])
        assert decision['decision'] == 'rerun_before', path
    for key in ('browser_version', 'os_version', 'launch_flags', 'hardware', 'ui_locale', 'pip_freeze_sha256',
                'machine_state'):
        decision = probe.before_after(baseline, dict(IDENTITY, **{key: 'other'}), [])
        assert decision['decision'] == 'rerun_before' and decision['identity'][key] == 'other'
    assert probe.before_after(baseline, dict(IDENTITY), [], script_runs=False) == {'decision': 'new_script_id',
                                                                                  'compare_baseline': False}
    assert probe.comparable_rows([{'row': 'core'}, {'row': 'extra', 'extra': True}]) == [{'row': 'core'}]
    assert probe.warm_rows_comparable({'disk': 90, 'network': 10}, {'disk': 81, 'network': 19})
    assert not probe.warm_rows_comparable({'disk': 90, 'network': 10}, {'disk': 79, 'network': 21})


# ------------------------------------------------------------------------------------------
# Overhead and frame event (R4-9, R5-6, R5-14e)
# ------------------------------------------------------------------------------------------

def test_overhead_median_a_minus_c_with_components_and_consequences():
    arms = {'a': {'inp': [40, 48, 56], 'raf_feedback': [20, 22, 24]},
            'b': {'inp': [40, 40, 48], 'raf_feedback': [12, 14, 16]},
            'c': {'inp': [24, 32, 40], 'raf_feedback': [10, 12, 14]}}
    result = probe.overhead(arms)
    point = lambda value: {'low': value, 'high': value, 'evaluable': True}
    assert result['inp']['overhead'] == point(16) and result['inp']['screencast'] == point(8) \
        and result['inp']['tracing'] == point(8)
    assert result['raf_feedback']['screencast'] == point(8) and result['raf_feedback']['overhead'] == point(10)
    assert result['consequences'] == {'screencast_limited_to_contrast': False, 'untraced_inp': False,
                                      'unevaluable': [], 'method_change': False}
    arms['c']['inp'] = [16, 24, 32]
    assert probe.overhead(arms)['consequences']['untraced_inp'] is True
    assert probe.reconfirmation_arm('c11_action_feedback') == 'b'
    assert probe.reconfirmation_arm('inp') == 'c' and probe.reconfirmation_arm('c11_preview') == 'c'


def test_frame_event_acceptance_test_picks_first_passing_candidate():
    def phase(share_one, share_two):
        interactions = [{'PipelineReporter': index < share_one, 'Display::FrameDisplayed': index < share_two,
                         'DrawFrame': True} for index in range(100)]
        return {'present': {'PipelineReporter', 'Display::FrameDisplayed', 'DrawFrame'}, 'interactions': interactions}
    chosen = probe.frame_event_choice([phase(90, 97)])
    assert chosen['candidate'] == 'Display::FrameDisplayed' and not chosen['method_change']
    assert chosen['shares']['PipelineReporter']['share'] == 0.9
    none = probe.frame_event_choice([{'present': set(), 'interactions': [{}]}])
    assert none == {'candidate': None, 'shares': none['shares'], 'method_change': True}


def test_frame_candidates_presented_after_mutation():
    events = [{'name': 'PipelineReporter', 'ph': 'b', 'id': '1', 'pid': 1, 'ts': 1_000_000,
               'args': {'chrome_frame_reporter': {'state': 'STATE_PRESENTED_ALL'}}},
              {'name': 'PipelineReporter', 'ph': 'e', 'id': '1', 'pid': 1, 'ts': 1_030_000, 'args': {}},
              {'name': 'DrawFrame', 'ph': 'I', 'pid': 1, 'ts': 1_010_000}]
    found = probe.frame_candidates(events)
    assert probe.presented_after(found['PipelineReporter'], 999.0) == 1030.0
    assert probe.presented_after(found['PipelineReporter'], 1001.0) is None
    assert probe.presented_after(found['DrawFrame'], 1005.0) == 1010.0


# ------------------------------------------------------------------------------------------
# Launch and activation (R4-6, R5-14a)
# ------------------------------------------------------------------------------------------

def test_launch_argv_is_frozen_open_na_form():
    binary = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    argv = probe.launch_argv(binary, Path('/tmp/p'), Path('/tmp/log'))
    assert argv == ['open', '-na', '/Applications/Google Chrome.app', '--stdout', '/tmp/log', '--stderr', '/tmp/log',
                    '--args', '--no-first-run', '--no-default-browser-check',
                    '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
                    '--disable-background-timer-throttling', '--disable-background-networking',
                    '--disable-component-update', '--user-data-dir=/tmp/p', '--remote-debugging-port=0', 'about:blank']
    with_log = probe.launch_argv(binary, Path('/tmp/p'), Path('/tmp/log'), netlog=Path('/tmp/n.json'))
    assert '--log-net-log=/tmp/n.json' in with_log


async def test_mac_launch_uses_open_na_never_popen(monkeypatch, tmp_path):
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError('stop')
    monkeypatch.setattr(probe.subprocess, 'run', fake_run)
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a, **k: calls.append(('popen', a)))
    browser = probe.Browser('mac', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', tmp_path / 'p',
                            tmp_path / 'log')
    with pytest.raises(RuntimeError, match='stop'):
        await browser.launch()
    assert calls[0][:2] == ['open', '-na'] and all(call[0] != 'popen' for call in calls)


async def test_activation_poll_needs_both_in_one_poll_and_failure_is_invalid():
    answers = iter([False, True, True])
    page = FakeClient({'Runtime.evaluate': lambda params: {'result': {'value': next(answers)}}})
    states = iter(['normal', 'minimized', 'normal'])
    browser = FakeClient({'Browser.getWindowForTarget': lambda params: {'bounds': {'windowState': next(states)}}})
    page.phase = browser.phase = 'setup'
    assert await probe.activation_poll(page, browser, 't', 10.0) == 3
    assert [entry['method'] for entry in page.log][0] == 'Page.bringToFront'
    never = FakeClient({'Runtime.evaluate': {'result': {'value': False}}})
    with pytest.raises(probe.ActivationFailed):
        await probe.activation_poll(never, FakeClient({'Browser.getWindowForTarget': {'bounds': {'windowState': 'normal'}}}),
                                    't', 0.35)
    assert probe.classify(RunFacts(setup_faults=['activation_failed']))[0] == Outcome.INVALID


# ------------------------------------------------------------------------------------------
# Engine: dispatch, missing targets, post-window command log (R4-2, R5-4, R5-8)
# ------------------------------------------------------------------------------------------

async def test_play_gesture_dispatched_from_control_attached_rect_without_renderer_command():
    run = make_run()
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    for index in range(3):
        record(run, 'target_ready', index=index, rect=rect, step=VOICES_STEPS[index]['step'])
    for index in range(2):
        record(run, 'step_next_frame', index=index)
    record(run, 'control_attached', time=500.0, readyState=0,
           target={'kind': 'audio', 'rect': rect, 'point': {'x': 30.0, 'y': 35.0}})
    # V2: a native control attached at readyState 0 is clicked from its control_ready record (loadedmetadata).
    record(run, 'control_ready', time=520.0, readyState=1,
           target={'kind': 'audio', 'rect': rect, 'point': {'x': 31.0, 'y': 36.0}, 'in_viewport': True})
    record(run, 'media', event='playing', timeStamp=600.0, readyState=4)
    started = time.monotonic()
    await run._interactions(run.config['steps'])
    log = run.page.log
    inputs_sent = [index for index, entry in enumerate(log) if entry['method'].startswith('Input.')]
    assert len(inputs_sent) == 12
    escuchar_last, play_first = inputs_sent[8], inputs_sent[9]
    assert all(not log[i]['method'].startswith(('Runtime.', 'DOM.', 'Accessibility.'))
               for i in range(escuchar_last + 1, play_first))
    assert log[play_first]['params'] == {'type': 'mouseMoved', 'x': 31.0, 'y': 36.0}
    assert [step['status'] for step in run.steps] == ['dispatched'] * 4 and not run.facts.product_failures
    assert run.steps[3]['ready_record'] == 'control_ready'
    # R01 (V12): consecutive core steps are at least 1 s apart; the play gesture is not spaced.
    firsts = [log[index]['sent'] for index in inputs_sent[::3]]
    assert firsts[1] - firsts[0] >= 0.99 and firsts[2] - firsts[1] >= 0.99 and firsts[3] - firsts[2] < 0.5
    assert time.monotonic() - started >= 1.98


async def test_missing_target_is_product_failure_with_infinite_inp_and_cls():
    run = make_run(view='#hogar')
    started = time.monotonic()
    await run._interactions(run.config['steps'])
    assert 4.9 < time.monotonic() - started < 6.0
    assert run.facts.product_failures == ['target_absent:hogar_plano']
    assert not [entry for entry in run.page.log if entry['method'].startswith('Input.')]
    run.facts.navigate_sent = True
    record(run, 'document_start', screen={'width': 1440, 'height': 900}, dpr=2,
           focus={'visibility': 'visible', 'hasFocus': True})
    result = run.result()
    assert result['metrics']['inp'] == INF and result['metrics']['cls'] == INF
    assert result['outcome'] == Outcome.FAILURE


async def test_failed_review_open_skips_play_gesture_and_inp_infinite():
    run = make_run()
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    for index in range(3):
        record(run, 'target_ready', index=index, rect=rect, step=VOICES_STEPS[index]['step'])
    for index in range(2):
        record(run, 'step_next_frame', index=index)
    await run._interactions(run.config['steps'])
    assert run.facts.product_failures == ['preview_failed_open:escuchar']
    assert len([entry for entry in run.page.log if entry['method'].startswith('Input.')]) == 9
    run.facts.navigate_sent = True
    record(run, 'document_start', screen={'width': 1440, 'height': 900}, dpr=2,
           focus={'visibility': 'visible', 'hasFocus': True})
    result = run.result()
    assert result['metrics']['inp'] == INF and result['metrics']['preview']['value'] == INF


async def test_missing_preview_sample_row_is_failed_open():
    run = make_run()
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    record(run, 'target_ready', index=0, rect=rect, step='revisar_muestras')
    record(run, 'step_next_frame', index=0)
    await run._interactions(run.config['steps'])
    assert run.facts.product_failures == ['target_absent:muestra']
    run.facts.navigate_sent = True
    record(run, 'document_start', screen={'width': 1440, 'height': 900}, dpr=2,
           focus={'visibility': 'visible', 'hasFocus': True})
    assert run.result()['metrics']['preview']['failed_open'] is True


def test_post_window_command_log_has_no_renderer_command_after_flush():
    log = [{'method': 'Page.navigate'}, {'method': 'Runtime.evaluate', 'flush': True}, {'method': 'Tracing.end'},
           {'method': 'Page.stopScreencast'}, {'method': 'Browser.getVersion'}]
    assert probe.post_window_violations(log) == []
    assert probe.post_window_violations(log + [{'method': 'Runtime.evaluate'}]) == ['Runtime.evaluate']
    methods = [method for method, _, _ in probe.post_window_commands(True, True)]
    assert methods == ['Tracing.end', 'Page.stopScreencast', 'Browser.getWindowForTarget', 'Browser.getVersion']
    assert not any(method.startswith(('Runtime.', 'Input.', 'DOM.')) for method in methods)


async def test_collect_phase_sends_only_browser_side_commands():
    run = make_run(view='#acciones')
    run.tracing = run.screencast = True
    run.trace_complete.set()
    run.browser.client.answers = {'Browser.getVersion': {'product': 'Chrome/157'},
                                  'Browser.getWindowForTarget': {'bounds': {'windowState': 'normal'}}}
    await run._collect()
    sent = [entry['method'] for entry in run.page.log + run.browser.client.log]
    assert sent == ['Page.stopScreencast', 'Tracing.end', 'Browser.getWindowForTarget', 'Browser.getVersion']
    assert run.browser_version == 'Chrome/157' and run.window_states == ['normal']


# ------------------------------------------------------------------------------------------
# Modes: hash check, smoke mode, smoke repeats (R4-3, R5-9)
# ------------------------------------------------------------------------------------------

def test_hash_check_refuses_other_versions(tmp_path):
    path = tmp_path / 'preregistration.json'
    path.write_text('{}')
    digest = probe.file_sha256(path)
    assert probe.verify_preregistration(path, digest) == digest
    for wrong in ('0' * 64, None, ''):
        with pytest.raises(SystemExit, match='preregistration_hash_mismatch'):
            probe.verify_preregistration(path, wrong)


def test_smoke_mode_seeds_outputs_and_baseline_refusal(tmp_path):
    smoke = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#voces', 1, smoke=True)
    expected = int.from_bytes(__import__('hashlib').sha256(f'{SHA}:smoke:batch:mac-chrome:desktop:#voces:1'.encode())
                              .digest()[:8], 'big')
    assert smoke['seed'] == expected != probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#voces', 1)['seed']
    assert probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#voces', 1, smoke=True) == smoke
    assert probe.checked_output('smoke', probe.SMOKE_ROOT / 'x') == (probe.SMOKE_ROOT / 'x').resolve()
    for outside in (probe.OUTPUT_ROOT / 'u1' / 'baseline' / 'x', tmp_path / 'x'):
        with pytest.raises(SystemExit, match='output_outside_mode_root'):
            probe.checked_output('smoke', outside)
    with pytest.raises(SystemExit, match='output_outside_mode_root'):
        probe.checked_output('baseline', probe.SMOKE_ROOT / 'x')
    with pytest.raises(SystemExit, match='smoke_result_refused'):
        probe.refuse_smoke_input({'mode': 'smoke'})
    probe.refuse_smoke_input({'mode': 'baseline'})


def test_smoke_repeat_needs_recorded_cause(tmp_path):
    record_path = tmp_path / 'smoke-record.json'
    smoke = probe.SmokeRecord(record_path)
    entry = smoke.start('cold_focus', SHA)
    smoke.finish(entry, 'complete', {'polls': [1]})
    with pytest.raises(SystemExit, match='smoke_repeat_refused'):
        probe.SmokeRecord(record_path).start('cold_focus', SHA)
    # V9: an enumerated cause the previous run did not record is refused too.
    with pytest.raises(SystemExit, match='smoke_repeat_refused'):
        probe.SmokeRecord(record_path).start('cold_focus', SHA, cause='crash')
    faulted = probe.SmokeRecord(record_path)
    faulted.finish(faulted.data['runs'][-1], 'aborted', {'x': 1}, faults=['quiet_machine'])
    with pytest.raises(SystemExit, match='smoke_repeat_refused'):
        probe.SmokeRecord(record_path).start('cold_focus', SHA, cause='lock')
    again = probe.SmokeRecord(record_path)
    again.start('cold_focus', SHA, cause='quiet_machine')
    # An entry left 'started' (the probe itself stopped) records a crash.
    probe.SmokeRecord(record_path).start('cold_focus', SHA, cause='crash')
    saved = json.loads(record_path.read_text())
    assert saved['mode'] == 'smoke' and [run['cause'] for run in saved['runs']] == [None, 'quiet_machine', 'crash']
    assert oct(record_path.stat().st_mode & 0o777) == '0o600'


# ------------------------------------------------------------------------------------------
# Fixture determinism and position check (R4-5, R5-2)
# ------------------------------------------------------------------------------------------

async def start_fixture(tmp_path, monkeypatch, shift_s: float):
    from contextlib import AsyncExitStack
    from tools import web_timing_fixture as fixture_module
    monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'synthetic-t046-root')
    real = time.time
    monkeypatch.setattr(fixture_module, 'time', SimpleNamespace(time=lambda: real() + shift_s))
    fixture = fixture_module.Fixture(tmp_path / f'run-{shift_s}')
    stack = AsyncExitStack()
    await stack.__aenter__()
    directory = tmp_path / f'graph-{shift_s}'
    directory.mkdir()
    dashboard = await fixture.build(stack, directory)
    return fixture, dashboard, stack


async def test_fixture_determinism_and_in_process_position_check(tmp_path, monkeypatch):
    first, dashboard, stack = await start_fixture(tmp_path, monkeypatch, -3600.0)
    try:
        first_map = first.position_map()
        assert (await first.position_check(1))['ok'] is True
        written = json.loads((first.run_dir / 'position-check-00001.json').read_text())
        assert written['ok'] and len(written['ids']) == 50
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=dashboard.app), base_url='http://127.0.0.1') as client:
            login = await client.post('/api/admin/auth/login', json={'token': first.tokens['operator']})
            assert login.status_code == 200
            # G16: a plain HTTP read without a live, ready /ws/voices subscriber cannot serve the position check.
            assert (await client.get('/api/voices')).status_code == 428
        first.position_ids[0], first.position_ids[1] = first.position_ids[1], first.position_ids[0]
        assert (await first.position_check(2))['ok'] is False
    finally:
        await stack.aclose()
    second, _, stack = await start_fixture(tmp_path, monkeypatch, 0.0)
    try:
        second_map = second.position_map()
        assert second.digest() == first.digest() == first_map['digest']
        assert [(item['position'], item['index']) for item in second_map['positions']] == \
            [(item['position'], item['index']) for item in first_map['positions']]
        assert second_map['preview_permutation'] == first_map['preview_permutation']
        assert second_map['positions'][0]['capture_id'] != first_map['positions'][0]['capture_id']
        assert (await second.position_check(1))['ok'] is True
    finally:
        await stack.aclose()


def test_fixture_plan_is_seed_determined():
    from tools import web_timing_fixture as fixture_module
    plan = fixture_module.build_plan()
    assert plan.digest() == fixture_module.build_plan().digest()
    assert fixture_module.build_plan(46002).digest() != plan.digest()
    assert len(plan.captures) == 120 and len({capture.offset_s for capture in plan.captures}) == 120
    assert all(60 <= capture.offset_s < 6 * 3600 for capture in plan.captures)
    assert all(abs(min(capture.duration_s, 6.0) - 6.0) < 1e-9 for capture in plan.captures)
    assert len({capture.group for capture in plan.captures}) >= 3 and len(plan.profiles) == 3
    assert sorted(plan.permutation) == list(range(50))
    assert fixture_module.derived_uuid(46001, 'capture', 3) == fixture_module.derived_uuid(46001, 'capture', 3)


async def test_fixture_fault_on_position_mismatch_or_no_answer():
    class Pipe:
        def __init__(self, reply):
            self.reply = reply

        async def request(self, command, deadline):
            assert deadline == probe.DEADLINE_MS['setup'] / 1000
            if self.reply is None:
                raise asyncio.TimeoutError
            return self.reply
    assert await probe.position_check(Pipe({'reply': 'position_check', 'ordinal': 3, 'ok': True}), 3)
    assert not await probe.position_check(Pipe({'reply': 'position_check', 'ordinal': 3, 'ok': False}), 3)
    assert not await probe.position_check(Pipe(None), 3)
    assert probe.classify(RunFacts(setup_faults=['position_check_failed']))[0] == Outcome.INVALID


def test_play_target_rule_in_injected_source_is_light_dom_only():
    source = probe.INJECTED_SOURCE
    lookup = source[source.index('function lookup('):source.index('function inViewport(')]
    assert 'shadowRoot' not in lookup and 'querySelectorAll(rule.selector)' in lookup
    assert "el.hasAttribute('aria-label') ? el.getAttribute('aria-label').trim() : (el.textContent || '').trim()" in source
    assert probe.CORE_SCRIPT['play_target'] == {'selector': 'button,[role="button"]', 'name_prefix': 'Reproducir'}


async def test_run_result_play_gesture_enters_voces_inp():
    run = make_run()
    run.steps = [{'step': step['step'], 'status': 'dispatched'} for step in VOICES_STEPS]
    run.facts.navigate_sent = True
    record(run, 'document_start', screen={'width': 1440, 'height': 900}, dpr=2,
           focus={'visibility': 'visible', 'hasFocus': True})
    for index, (first, last) in enumerate([(10, 12), (1010, 1012), (2010, 2012), (2020, 2022)]):
        for item in inputs(index, first, last):
            record(run, 'input', **item)
    for interaction, start, duration in ((1, 11, 24.0), (2, 1011, 32.0), (3, 2011, 40.0), (4, 2020.5, 72.0)):
        record(run, 'perf', entry=event(interaction, 'click', start, duration))
    result = run.result()
    assert result['metrics']['inp'] == 72.0 and result['outcome'] == Outcome.VALID


async def test_warm_target_after_failed_leave_is_new_about_blank_target():
    session = probe.BrowserSession('harness', 'chromium', Path('/nonexistent'))
    old_page = FakeClient()
    session.browser = SimpleNamespace(client=FakeClient({'Target.createTarget': {'targetId': 'fresh'}}),
                                      connect_page=lambda target: asyncio.sleep(0, result=FakeClient()))
    session.page, session.target_id, session.previous_script, session.needs_new_target = old_page, 'hung', '3', True
    warm = probe.RunSpec(1, 'warm', 'desktop', '#hogar', 'u', 'o', None)
    await session.prepare_target(warm)
    assert [entry['method'] for entry in session.browser.client.log] == ['Target.closeTarget', 'Target.createTarget']
    assert session.browser.client.log[1]['params'] == {'url': 'about:blank'}
    assert session.target_id == 'fresh' and session.previous_script is None and session.page is not old_page
    session.previous_script, session.needs_new_target = '4', False
    await session.prepare_target(warm)
    assert len(session.browser.client.log) == 2 and session.previous_script == '4'


async def test_leave_step_timeout_is_post_window_hang():
    run = make_run(view='#rutinas')
    started = time.monotonic()
    await run._leave()
    assert 9.9 < time.monotonic() - started < 11.0
    assert [timeout.phase for timeout in run.facts.timeouts] == ['leave'] and not run.facts.leave_ok
    assert probe.classify(run.facts)[0] == Outcome.POST_WINDOW_HANG


async def test_run_lifecycle_setup_activation_timed_flush_collect_and_leave():
    run = make_run(view='#rutinas', capture=None)
    run.filter.committed = False
    run.filter.time_origin = None
    page, browser = run.page, run.browser.client

    def navigate(params):
        if params['url'] == 'about:blank':
            for handler in list(page.handlers['Page.frameNavigated']):
                handler({'frame': {'url': 'about:blank', 'loaderId': 'blank'}})
            return {'loaderId': 'blank'}
        page.events.append({'method': 'Page.frameNavigated', 'params': {'frame': {'loaderId': 'L', 'url': params['url']}}})
        base = {'token': run.token, 'timeOrigin': 5.0}
        documents = [dict(base, type='document_start', t=1, screen={'width': 1440, 'height': 900}, dpr=2,
                          focus={'visibility': 'visible', 'hasFocus': True}),
                     dict(base, type='settled', t=900, T=900.0)]
        asyncio.get_running_loop().call_soon(lambda: [run.records.offer(json.dumps(item)) for item in documents])
        return {'loaderId': 'L'}

    page.answers = {'Page.navigate': navigate, 'Page.addScriptToEvaluateOnNewDocument': {'identifier': '9'},
                    'Runtime.evaluate': lambda params: {'result': {'value': 'hasFocus' in params['expression'] or 1}}}
    browser.answers = {'Browser.getWindowForTarget': {'bounds': {'windowState': 'normal'}},
                       'Browser.getVersion': {'product': 'Chrome/157'}}
    await run._run()
    methods = [entry['method'] for entry in page.log]
    assert methods.index('Page.addScriptToEvaluateOnNewDocument') < methods.index('Page.bringToFront') \
        < methods.index('Page.navigate')
    assert run.activation_polls == 1 and run.script_identifier == '9'
    assert methods[-2:] == ['Runtime.evaluate', 'Page.navigate'] and page.log[-1]['phase'] == 'leave'
    assert page.log[-2]['phase'] == 'timed' and page.log[-2].get('flush') is True
    assert [entry['method'] for entry in browser.log if entry['phase'] == 'collect'] == \
        ['Browser.getWindowForTarget', 'Browser.getVersion']
    result = run.result()
    assert result['outcome'] == Outcome.VALID and result['post_window_violations'] == []
    assert result['browser_version'] == 'Chrome/157' and result['metrics']['inp'] == probe.LOAD_ONLY_INP


def test_fill_ins_change_only_harness_only_deadlines_never_lower(monkeypatch):
    monkeypatch.setattr(probe, 'DEADLINE_MS', dict(probe.DEADLINE_MS))
    probe.apply_fill_ins({'harness_only_deadlines_ms': {'setup': probe.DEADLINE_MS['setup'] + 5000}})
    assert probe.DEADLINE_MS['setup'] == 15000
    for klass in sorted(probe.IMMUTABLE_CLASSES):
        with pytest.raises(SystemExit):
            probe.apply_fill_ins({'harness_only_deadlines_ms': {klass: probe.DEADLINE_MS[klass] + 1000}})
    with pytest.raises(SystemExit):
        probe.apply_fill_ins({'harness_only_deadlines_ms': {'collect': probe.DEADLINE_MS['collect'] - 1}})


def test_overhead_arm_median_and_frame_event_phases():
    assert probe.numeric_median([30.0, 10.0, 20.0, 40.0]) == 25.0
    assert probe.numeric_median([CENSORED, CENSORED, 20.0]) is None
    assert probe.numeric_median([]) is None
    run = {'metrics': {'action_feedback': {'raf': 100.0, 'screencast': 120.0,
                                           'presented': {'PipelineReporter': 118.0, 'DrawFrame': 90.0}}},
           'trace_candidates_present': ['DrawFrame', 'PipelineReporter'], 'trace_events': 500,
           'steps': [{'status': 'dispatched'}]}
    untraced = {'metrics': {}, 'trace_candidates_present': [], 'trace_events': 0, 'steps': [{'status': 'dispatched'}]}
    phases = probe.frame_event_phases([run, untraced])
    assert phases == [{'present': {'DrawFrame', 'PipelineReporter'},
                       'interactions': [{'PipelineReporter': True, 'DrawFrame': False}]}]
    assert probe.frame_event_choice(phases)['candidate'] == 'PipelineReporter'
    # V14f: a traced interaction phase with no candidate at all still counts (candidate absent there).
    empty = {'metrics': {}, 'trace_candidates_present': [], 'trace_events': 80, 'steps': [{'status': 'dispatched'}]}
    phases = probe.frame_event_phases([run, empty])
    assert len(phases) == 2 and probe.frame_event_choice(phases)['method_change'] is True


# ------------------------------------------------------------------------------------------
# Probe review round 1 (V1..V14, F1..F5): preregistered rules applied by the probe
# ------------------------------------------------------------------------------------------

def focus_doc(has_focus, time_=1.0, visibility='visible'):
    return {'screen': {'width': 1440, 'height': 900}, 'dpr': 2,
            'focus': {'point': 'document_start', 'visibility': visibility, 'hasFocus': has_focus, 'time': time_}}


def test_focus_rule_document_start_false_needs_focus_event_before_next_point():
    """F1: hasFocus false at document_start is valid only when a window 'focus' event of the run document arrives
    before the next sample point and hasFocus is true at every later point."""
    later = [{'point': 'domcontentloaded', 'visibility': 'visible', 'hasFocus': True, 'time': 40.0},
             {'point': 'settled', 'visibility': 'visible', 'hasFocus': True, 'time': 900.0}]
    focus_event = [{'event': 'focus', 'timeStamp': 2.0}]
    normal = [{'point': 'navigation', 'state': 'normal'}]
    assert probe.visibility_faults(focus_doc(False), later, focus_event, normal) == []
    assert probe.visibility_faults(focus_doc(False), later, [], normal) == ['page_unfocused']
    assert probe.visibility_faults(focus_doc(False), later, [{'event': 'focus', 'timeStamp': 41.0}], normal) \
        == ['page_unfocused']
    assert probe.visibility_faults(focus_doc(False), later, focus_event + [{'event': 'blur', 'timeStamp': 3.0}],
                                   normal) == ['visibility_event']
    assert probe.visibility_faults(focus_doc(True), later, [], normal) == []


async def test_records_after_flush_never_invalidate_and_window_reads_at_timed_points():
    """V4: a pagehide or blur after the flush (leave step) is outside the timed phase; window-state reads are taken
    at navigation, load end, every interaction start and end and the flush, and a non-normal one invalidates."""
    run = make_run(view='#rutinas', capture=None)
    run.facts.navigate_sent = True
    record(run, 'document_start', **focus_doc(True))
    record(run, 'focus', point='flush', visibility='visible', hasFocus=True, time=900.0, t=900.0)
    record(run, 'viewport', point='flush', innerWidth=1440, innerHeight=900, dpr=2)
    record(run, 'visibility_event', event='pagehide', timeStamp=5000.0, state='visible', hasFocus=True, time=5000.0,
           t=5000.0)
    for point in ('navigation', 'load_end', 'flush'):
        run._sample_window(point)
    await run._await_window_reads()
    assert sorted(item['point'] for item in run.window_reads) == ['flush', 'load_end', 'navigation']
    assert all(entry['class'] == 'window_state' for entry in run.browser.client.log)
    result = run.result()
    assert result['outcome'] == Outcome.VALID, result['facts']
    late_blur = make_run(view='#rutinas', capture=None)
    late_blur.facts.navigate_sent = True
    record(late_blur, 'document_start', **focus_doc(True))
    record(late_blur, 'focus', point='flush', visibility='visible', hasFocus=True, time=900.0, t=900.0)
    record(late_blur, 'visibility_event', event='blur', timeStamp=300.0, state='visible', hasFocus=False, t=300.0)
    assert late_blur.result()['outcome'] == Outcome.INVALID
    minimised = make_run(view='#rutinas', capture=None)
    minimised.browser.client.answers = {'Browser.getWindowForTarget': {'bounds': {'windowState': 'minimized'}}}
    minimised.facts.navigate_sent = True
    record(minimised, 'document_start', **focus_doc(True))
    minimised._sample_window('load_end')
    await minimised._await_window_reads()
    assert minimised.result()['reason'] == 'window_state'


async def test_interactions_sample_window_at_every_start_and_end():
    run = make_run(view='#acciones', capture=None)
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    record(run, 'target_ready', index=0, rect=rect, step='accion', in_viewport=True)
    record(run, 'step_next_frame', index=0)
    record(run, 'feedback_window', index=0, frames=6)
    await run._interactions(run.config['steps'])
    await run._await_window_reads()
    assert [item['point'] for item in run.window_reads] == ['interaction_start:0', 'interaction_end:0']


async def test_play_gesture_waits_for_control_ready_and_fails_open_without_it():
    """V2: native control at readyState 0 and no loadedmetadata -> failed open (control_not_ready), no click."""
    run = make_run()
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    for index in range(3):
        record(run, 'target_ready', index=index, rect=rect, step=VOICES_STEPS[index]['step'])
    for index in range(2):
        record(run, 'step_next_frame', index=index)
    record(run, 'control_attached', time=500.0, readyState=0,
           target={'kind': 'audio', 'rect': rect, 'point': {'x': 30.0, 'y': 35.0}})
    saved = dict(probe.DEADLINE_MS)
    try:
        probe.DEADLINE_MS['wait_preview'] = 1500
        await run._interactions(run.config['steps'])
    finally:
        probe.DEADLINE_MS.update(saved)
    assert run.facts.product_failures == ['preview_failed_open:control_not_ready']
    assert len([entry for entry in run.page.log if entry['method'].startswith('Input.')]) == 9
    button = {'kind': 'button', 'rect': rect, 'point': {'x': 1.0, 'y': 2.0}}
    assert probe.play_ready({'type': 'control_attached', 'target': button, 'readyState': 0}, None)['target'] == button
    loaded = {'type': 'control_attached', 'target': {'kind': 'audio'}, 'readyState': 1}
    assert probe.play_ready(loaded, None) is loaded
    assert probe.play_ready({'type': 'control_attached', 'target': {'kind': 'audio'}, 'readyState': 0}, None) is None


def test_preview_value_uses_control_ready_and_gap_from_it():
    """V2: primary = (ready - start) + (playing - gesture); the gap is gesture - ready, never gesture - attach."""
    typed = defaultdict(list)
    typed['control_attached'] = [{'type': 'control_attached', 'time': 500.0, 'readyState': 0,
                                  'target': {'kind': 'audio'}}]
    typed['control_ready'] = [{'type': 'control_ready', 'time': 540.0, 'readyState': 1, 'target': {'kind': 'audio'}}]
    typed['media'] = [{'event': 'playing', 'timeStamp': 700.0, 'readyState': 4}]
    interactions = [{'first': 10.0}, {'first': 1000.0}, {'first': 2000.0}, {'first': 548.0}]
    result = probe.c11_metrics(typed, interactions, VOICES_STEPS, None)['preview']
    assert result['gap'] == 8.0 and result['value'] == (540.0 - 2000.0) + (700.0 - 548.0)
    assert result['secondary']['metadata_wait'] == 40.0 and result['secondary']['ready_record'] == 'control_ready'


async def test_target_outside_viewport_is_never_clicked_and_is_a_harness_fault():
    """V3: the dispatch point must lie in the viewport; the probe never scrolls and never clicks outside it."""
    run = make_run(view='#acciones', capture=None)
    record(run, 'target_ready', index=0, rect={'x': 10, 'y': 2000, 'width': 40, 'height': 30}, step='accion',
           in_viewport=False)
    await run._interactions(run.config['steps'])
    assert not [entry for entry in run.page.log if entry['method'].startswith('Input.')]
    assert run.facts.environment_faults == ['target_outside_viewport:accion']
    run.facts.navigate_sent = True
    record(run, 'document_start', **focus_doc(True))
    assert run.result()['outcome'] == Outcome.INVALID


def test_injected_script_scrolls_only_after_previous_step_frame():
    """V3: the scroll lives in the step_next_frame callback path and is gated by mayScroll (index > 0, next pending
    step, no open capture window, previous step's frame done); the MutationObserver path never scrolls step 0."""
    source = probe.INJECTED_SOURCE
    may = source[source.index('function mayScroll('):source.index('function bringIntoView(')]
    assert 'index > 0' in may and 'index === interactions' in may and '!open' in may and 'settledFrame' in may
    assert 'index < STEPS.length' in may
    frame = source[source.index("emit('step_next_frame'"):source.index("emit('step_next_frame'") + 200]
    assert 'settledFrame = true' in frame and 'bringIntoView(index + 1)' in frame
    assert source.count('scrollIntoView(') == 1


def test_raf_lower_bound_never_precedes_the_mutation():
    """V5: the first-rAF lower bound is performance.now() in the callback and never before the end mutation."""
    typed = defaultdict(list)
    typed['step_end'] = [{'index': 0, 'time': 100.0}]
    typed['step_frame'] = [{'index': 0, 'raf': 95.0, 'frame_time': 87.3}]
    steps = probe.CORE_SCRIPT['#acciones']
    result = probe.c11_metrics(typed, [{'first': 90.0}], steps, None)
    assert result['action_feedback']['raf'] == 100.0 and result['action_feedback_raf'] == 10.0
    typed['step_frame'] = [{'index': 0, 'raf': 104.0, 'frame_time': 87.3}]
    assert probe.c11_metrics(typed, [{'first': 90.0}], steps, None)['action_feedback']['raf'] == 104.0
    assert "raf: now(), frame_time: frame" in probe.INJECTED_SOURCE


async def test_tracing_complete_timeout_goes_through_result_without_error():
    """V6: Tracing.end answered but tracingComplete never arrives -> classified through Run.result()."""
    run = make_run(view='#rutinas', capture=None)
    run.facts.navigate_sent = True
    run.tracing = True
    record(run, 'document_start', **focus_doc(True))
    mark = record(run, 'mark', name='kza-probe:start:x:0', time=100.0)
    mark['t'], mark['_received'] = 100.0, 50.0
    run.trace_events = [{'name': 'kza-probe:start:x:0', 'ph': 'R', 'ts': 5_000_000, 'pid': 1, 'tid': 2},
                        {'ph': 'M', 'name': 'thread_name', 'pid': 1, 'tid': 2, 'args': {'name': 'CrRendererMain'}},
                        {'ph': 'X', 'name': 'RunTask', 'pid': 1, 'tid': 2, 'ts': 4_000_000, 'dur': 90_000_000}]
    saved = dict(probe.DEADLINE_MS)
    try:
        probe.DEADLINE_MS['collect'] = 300
        run._phase('collect')
        with pytest.raises(probe.CdpTimeout):
            await run._collect()
    finally:
        probe.DEADLINE_MS.update(saved)
    run.facts.timeouts.append(Timeout('collect', 'collect', 'Tracing.end'))
    entry = run.browser.client.log[-1]
    assert entry['method'] == 'Tracing.tracingComplete' and entry['deadline'] - entry['sent'] == pytest.approx(0.3)
    # Page time 200 ms at the Tracing.end send: the 90 s renderer task (trace 4000..94000 ms) spans the deadline.
    for item in run.filter.accepted:
        item['_received'] = entry['sent'] - 0.2 + item['t'] / 1000
    run.leave_ok = run.facts.leave_ok = True
    result = run.result()
    assert result['facts']['renderer_busy_across_deadline'] is True
    assert result['outcome'] == Outcome.POST_WINDOW_HANG


class Caffeinate:
    def __init__(self):
        self.terminated = False
        self.pid = 4242

    def terminate(self):
        self.terminated = True


async def test_batch_cleans_up_on_fixture_start_error_and_sweeps_browsers():
    """V7: caffeinate and the fixture start inside try/finally; leftover probe browsers are swept at start and exit."""
    fakes = BatchFakes()
    caffeinate = Caffeinate()
    swept = []

    async def broken():
        raise RuntimeError('fixture_not_ready')
    environment = fakes.environment()
    environment.start_fixture = broken
    environment.caffeinate = lambda: caffeinate
    environment.sweep = lambda: swept.append(len(swept)) or []
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#rutinas', 1, pairs=1, contrast_pairs=0)
    result = await probe.run_batch(plan, environment, idle=1.0, post_batch=1.0)
    assert caffeinate.terminated and result['discarded'] and result['reasons'] == ['fixture_fault']
    assert swept == [0, 1] and 'close_browser' in fakes.events
    rows = [{'pid': 5, 'ppid': 1, 'command': '/A/Google Chrome --user-data-dir=/tmp/kza-t046-profile-ab12 --remote-debugging-port=0'},
            {'pid': 6, 'ppid': 5, 'command': '/A/Google Chrome Helper --type=renderer --user-data-dir=/tmp/kza-t046-profile-ab12'},
            {'pid': 7, 'ppid': 1, 'command': '/A/Google Chrome --user-data-dir=/Users/u/Library/Chrome'},
            {'pid': 8, 'ppid': 1, 'command': '/A/Google Chrome --user-data-dir=/var/x/kza-t046-focus-1 x'}]
    killed = []
    assert probe.sweep_probe_browsers(rows, kill=lambda pid, sig: killed.append(pid)) == [5, 8] and killed == [5, 8]
    assert probe.start_caffeinate.__doc__ and '-w' in __import__('inspect').getsource(probe.start_caffeinate)


async def test_browser_close_keeps_profile_while_browser_lives(tmp_path, monkeypatch):
    """V7: the profile is never deleted while a browser still holds it."""
    profile = tmp_path / 'kza-t046-profile-x'
    profile.mkdir()
    browser = probe.Browser('mac', '/A.app/Contents/MacOS/A', profile, tmp_path / 'log')
    monkeypatch.setattr(browser, '_find_pid', lambda: 99)
    monkeypatch.setattr(browser, 'alive', lambda: True)
    monkeypatch.setattr(probe.os, 'kill', lambda pid, sig: None)
    browser.close_wait_s = browser.kill_wait_s = 0.05
    await browser.close()
    assert profile.exists() and browser.leftover


async def test_batch_machine_state_and_caffeinate_assertion_gate_quiet_machine():
    """V8: AC power, display and lock (when chosen) at start and end, and the caffeinate assertion, gate the batch."""
    plan = lambda: probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#rutinas', 1, pairs=1, contrast_pairs=0, warm_ups=0)
    battery = BatchFakes()
    environment = battery.environment()
    environment.machine_state = lambda: {'ac_power': False, 'display_asleep': False, 'locked': False}
    result = await probe.run_batch(plan(), environment, idle=1.0, post_batch=1.0)
    assert result['discarded'] and result['reasons'] == ['quiet_machine'] and 'fixture_start' not in battery.events
    no_assertion = BatchFakes()
    environment = no_assertion.environment()
    environment.assertion_held = lambda process: False
    result = await probe.run_batch(plan(), environment, idle=1.0, post_batch=1.0)
    assert result['discarded'] and 'quiet_machine' in result['reasons']
    assert probe.machine_faults({'ac_power': True, 'display_asleep': None, 'locked': None}, smoke=True) == []
    assert probe.machine_faults({'ac_power': True, 'display_asleep': None, 'locked': None}, smoke=False) \
        == ['display_asleep', 'screen_locked']
    assert probe.chosen_lock_state({'read1': {'present': True, 'locked': True}}, 'read1') is True
    assert probe.chosen_lock_state({'read1': {'present': False, 'locked': None}}, 'read1') is False
    assert probe.chosen_lock_state({'read1': {'error': 'x'}}, 'read1') is None
    assert probe.chosen_lock_state({'read1': {'present': True, 'locked': True}}, None) is None
    held = SimpleNamespace(pid=77)
    assert probe.caffeinate_assertion_held(held, '   pid 77(caffeinate): [0x1] 00:00:01 PreventUserIdleDisplaySleep named')
    assert not probe.caffeinate_assertion_held(held, '   pid 78(caffeinate): [0x1] PreventUserIdleDisplaySleep')


async def test_quiet_check_fails_when_pytest_runs(monkeypatch):
    """R05 (V12): quiet_check records a pytest process (python -m pytest or the console script) as a covariate;
    quiet_machine.shared: 'quiet' is False then, and no caller gates on it."""
    monkeypatch.setattr(probe, 'cpu_seconds', lambda: {})
    rows = iter([[{'pid': 50, 'ppid': 2, 'command': '/usr/bin/python3 -m pytest tests/'}],
                 [{'pid': 51, 'ppid': 2, 'command': '/Users/x/kza/.venv/bin/pytest tests/unit -q'}], []])
    monkeypatch.setattr(probe, 'process_rows', lambda: next(rows))
    saved = probe.asyncio.sleep

    async def fast(_):
        await saved(0)
    monkeypatch.setattr(probe.asyncio, 'sleep', fast)

    def no_pgrep(command, **kwargs):
        # R6-11(a): macOS pgrep has no -c; the count comes from the ps rows only.
        raise AssertionError(f'unexpected command {command}')
    monkeypatch.setattr(probe, 'run_text', no_pgrep)
    busy = await probe.quiet_check(1)
    assert busy['pytest_count'] == 1 and busy['quiet'] is False
    assert busy['covariates']['counts_by_class']['pytest'] == 1 and busy['covariates']['gating'] is False
    console = await probe.quiet_check(1)
    assert console['pytest_count'] == 1 and console['quiet'] is False
    assert console['covariates']['counts_by_class']['pytest'] == 1
    assert (await probe.quiet_check(1))['quiet'] is True and 'ok' not in busy


async def test_smoke_client_never_repeats_a_discarded_batch_and_records_each_cell(monkeypatch, tmp_path):
    """V9: in smoke mode each cell is one smoke-record entry with its faults; a discarded batch is not re-queued."""
    monkeypatch.setattr(probe, 'SMOKE_ROOT', tmp_path / 'smoke')
    monkeypatch.setattr(probe, 'checked_output', lambda mode, path: Path(path))
    calls = []

    async def fake_cell(args, mode, sha):
        calls.append((args.viewport, args.view, args.attempt))
        discarded = len(calls) == 1
        return {'discarded': discarded, 'reasons': ['warm_share_below_90'] if discarded else [],
                'faults': [], 'directory': 'd'}
    monkeypatch.setattr(probe, 'measure_cell', fake_cell)
    write_smoke_prerequisites(tmp_path / 'smoke' / 'smoke-record.json')
    args = SimpleNamespace()
    await probe.measure_client(args, 'smoke', SHA)
    assert len(calls) == 12 and all(attempt == 1 for _, _, attempt in calls)
    saved = json.loads((tmp_path / 'smoke' / 'smoke-record.json').read_text())
    batches = [run for run in saved['runs'] if run['item'].startswith('batch:')]
    assert len(batches) == 12 and batches[0]['status'] == 'discarded'
    assert len({run['item'] for run in batches}) == 12
    assert probe.batch_faults({'reasons': ['quiet_machine'], 'run_outcomes': [{'reason': 'browser_crashed'}],
                               'start': {'machine_faults': ['screen_locked']}}) == {'quiet_machine', 'crash', 'lock'}


def test_warm_rule_follows_smoke_warm_cache_outcome():
    """V9: below 90% in the smoke, warm rows are 'no evaluable' and no batch is discarded for the warm share."""
    runs = [{'role': 'warm', 'outcome': Outcome.VALID, 'cache_sources': {'network': 10}}]
    assert probe.invalid_batch_reasons(batch(runs, planned=80)) == ['warm_share_below_90']
    assert probe.invalid_batch_reasons(batch(runs, planned=80, warm_rule_active=False)) == []
    assert probe.warm_rule_active({}) and not probe.warm_rule_active({'warm_cache': {'reached': False}})
    rows = probe.summary_rows([{'viewport': 'desktop', 'view': '#hogar', 'role': 'warm', 'outcome': Outcome.VALID,
                                'metrics': {'lcp': 1.0, 'cls': 0.0, 'inp': 20.0}, 'cache_sources': {'network': 3}}],
                              None, warm_evaluable=False)
    assert rows[0]['lcp'] == probe.WARM_NOT_EVALUABLE and rows[0]['inp'] == probe.WARM_NOT_EVALUABLE


def test_overhead_is_censor_aware_and_consequences_cover_both_views():
    """V10: '<16 ms' medians are [0, 16) intervals; +inf medians are 'no evaluable' (method change); INP rows come
    from untraced runs when they exist."""
    censored = {arm: {'inp': [CENSORED, CENSORED, 24.0], 'raf_feedback': [5.0, 6.0, 7.0]} for arm in 'abc'}
    result = probe.overhead(censored)
    assert result['inp']['medians']['a'] == CENSORED
    assert result['inp']['tracing'] == {'low': -16.0, 'high': 16.0, 'evaluable': True}
    assert result['consequences']['untraced_inp'] is False and not result['consequences']['method_change']
    high = {'a': {'inp': [64.0] * 3}, 'b': {'inp': [64.0] * 3}, 'c': {'inp': [CENSORED] * 3}}
    assert probe.overhead(high, ('inp',))['inp']['tracing']['low'] == 48.0
    assert probe.overhead(high, ('inp',))['consequences']['untraced_inp'] is True
    failing = {'a': {'inp': [INF] * 3}, 'b': {'inp': [40.0] * 3}, 'c': {'inp': [40.0] * 3}}
    assert probe.overhead(failing, ('inp',))['consequences']['method_change'] is True
    both = probe.overhead_consequences({'#acciones': {arm: {'inp': [40.0], 'raf_feedback': [10.0]} for arm in 'abc'},
                                        '#voces': high})
    assert both['consequences']['untraced_inp'] is True and 'raf_feedback' not in both['views']['#voces']
    runs = [{'viewport': 'desktop', 'view': '#acciones', 'role': 'cold', 'outcome': Outcome.VALID,
             'metrics': {'lcp': 1.0, 'cls': 0.0, 'inp': 80.0}},
            {'viewport': 'desktop', 'view': '#acciones', 'role': 'cold', 'outcome': Outcome.VALID,
             'untraced_inp': True, 'metrics': {'lcp': 500.0, 'cls': 0.0, 'inp': 24.0}}]
    row = probe.summary_rows(runs, None)[0]
    assert row['inp']['p50'] == 24.0 and row['inp_source'] == 'untraced_interaction_phase'
    assert row['lcp']['p50'] == 1.0 and row['n_valid'] == 1


async def test_cdp_error_login_failure_and_unclassified_errors_never_abort_the_batch():
    """V11: a CDP error reply or another exception is a harness fault of its phase (invalid run)."""
    run = make_run(view='#rutinas', capture=None)

    async def failing_setup():
        raise probe.CdpError('Network.enable: {"code": -32000}')
    run._setup = failing_setup
    await run._run()
    assert run.facts.setup_faults[0].startswith('setup_harness_error:CdpError')
    assert probe.classify(run.facts)[0] == Outcome.INVALID
    timed = make_run(view='#rutinas', capture=None)

    async def nothing():
        return None

    async def failing_timed():
        raise probe.CdpError('Input.dispatchMouseEvent: refused')
    timed._setup, timed._timed = nothing, failing_timed

    def leave(params):
        for handler in list(timed.page.handlers['Page.frameNavigated']):
            handler({'frame': {'url': 'about:blank', 'loaderId': 'blank'}})
        return {'loaderId': 'blank'}
    timed.page.answers = {'Page.navigate': leave}
    await timed._run()
    assert timed.facts.environment_faults[0].startswith('timed_harness_error:CdpError')
    assert probe.classify(timed.facts) == (Outcome.INVALID, timed.facts.environment_faults[0])
    login = make_run(view='#rutinas', capture=None)
    login.spec.token_for_login = 'token'

    async def refuse(*args, **kwargs):
        raise RuntimeError('login_failed')
    import tests.frontend.browser_support as support
    saved = support.install_login_session
    support.install_login_session = refuse
    try:
        await login._run()
    finally:
        support.install_login_session = saved
    assert login.facts.setup_faults == ['login_failed:login_failed']


def test_exceptions_and_http_errors_are_recorded_document_error_fails_every_metric():
    """V11: Runtime.exceptionThrown and same-origin 4xx/5xx are recorded per run; a 4xx/5xx main document is a
    run-wide failure (navigation_http_error); other errors fail only the metric that names them."""
    run = make_run(view='#rutinas', capture=None)
    run.facts.navigate_sent = True
    record(run, 'document_start', **focus_doc(True))
    run.page.phase = 'timed'
    run._exception({'exceptionDetails': {'text': 'Uncaught', 'url': 'http://127.0.0.1:1/app.js', 'lineNumber': 3,
                                         'exception': {'description': 'TypeError: x'}}})
    run.network.response({'requestId': '1', 'type': 'XHR', 'response': {'url': 'http://127.0.0.1:1/api/x', 'status': 503}})
    result = run.result()
    assert result['timed_exceptions'] == 1 and result['http_errors'] == [{'path': '/api/x', 'status': 503, 'type': 'XHR'}]
    assert result['outcome'] == Outcome.VALID
    document = make_run(view='#rutinas', capture=None)
    document.facts.navigate_sent = True
    record(document, 'document_start', **focus_doc(True))
    document.network.response({'requestId': '2', 'type': 'Document',
                               'response': {'url': 'http://127.0.0.1:1/', 'status': 500}})
    failed = document.result()
    assert failed['outcome'] == Outcome.FAILURE and failed['reason'] == 'navigation_http_error:500'
    assert failed['run_wide_failure'] is True
    detached = make_run(view='#rutinas', capture=None)
    detached._detached({'reason': 'Render process gone.'})
    assert detached.facts.disconnected and probe.classify(detached.facts)[0] == Outcome.INVALID


class SetupFixture:
    def __init__(self, health=True):
        self.requests = []
        self.health = health

    async def request(self, command, deadline_s):
        self.requests.append(command)
        if command == 'health':
            return {'ok': self.health}
        ordinal = int(command.split()[1])
        return {'reply': 'position_check', 'ordinal': ordinal, 'ok': True}


async def test_setup_runs_health_and_position_check_and_traces_contrast_load():
    """R02, R03, R06 (V12): setup calls the fixture health check and, on #voces, the position check; a contrast
    slot traces the load phase from setup; an unhealthy fixture is a setup fault."""
    run = make_run()
    run.fixture = SetupFixture()
    run.spec.trace_load = True
    run.page.answers = {'Runtime.evaluate': {'result': {'value': True}}}
    await run._setup()
    assert run.fixture.requests == ['health', 'position_check 7']
    assert [entry['method'] for entry in run.browser.client.log if entry['class'] == 'setup'][-1] == 'Tracing.start'
    unhealthy = make_run(view='#rutinas', capture=None)
    unhealthy.fixture = SetupFixture(health=False)
    unhealthy.page.answers = {'Runtime.evaluate': {'result': {'value': True}}}
    with pytest.raises(probe.SetupFault, match='fixture_not_ready'):
        await unhealthy._setup()
    plan = probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#rutinas', 1, pairs=2, contrast_pairs=1, warm_ups=0)
    ready = {'origin': 'http://o', 'tokens': {'home': 't'}}
    for slot in plan['slots']:
        spec = probe.slot_spec(slot, plan, ready, {})
        assert spec.trace_load is bool(slot['trace_contrast'])


async def test_interaction_phase_is_traced_and_screencast_after_settle():
    """R04 (V12): the interaction-phase Tracing.start (class interaction_trace) and Page.startScreencast are sent
    after the settle record and before the first input."""
    run = make_run(view='#acciones', capture=None)
    run.config = probe.injection_config(token=run.token, origin=run.spec.origin, view='#acciones', capture_id=None)
    run.filter.committed = False
    run.filter.time_origin = None
    page = run.page

    def navigate(params):
        page.events.append({'method': 'Page.frameNavigated', 'params': {'frame': {'loaderId': 'L'}}})
        base = {'token': run.token, 'timeOrigin': 5.0}
        items = [dict(base, type='document_start', t=1, **focus_doc(True)), dict(base, type='settled', t=900, T=900.0),
                 dict(base, type='target_ready', t=901, index=0, rect={'x': 1, 'y': 1, 'width': 9, 'height': 9},
                      in_viewport=True, step='accion'),
                 dict(base, type='step_next_frame', t=950, index=0)]
        asyncio.get_running_loop().call_soon(lambda: [run.records.offer(json.dumps(item)) for item in items])
        return {'loaderId': 'L'}
    page.answers = {'Page.navigate': navigate, 'Runtime.evaluate': {'result': {'value': 1}}}
    run.facts.navigate_sent = True
    await run._timed()
    trace = [entry for entry in run.browser.client.log if entry['method'] == 'Tracing.start']
    assert trace and trace[0]['class'] == 'interaction_trace'
    methods = [entry['method'] for entry in page.log]
    assert methods.index('Page.startScreencast') < methods.index('Input.dispatchMouseEvent')
    assert trace[0]['sent'] <= page.log[methods.index('Input.dispatchMouseEvent')]['sent']


def test_records_before_the_navigate_reply_are_buffered_not_lost():
    """V14b: document_start and later records that arrive before the Page.navigate reply are kept."""
    gate = probe.RecordFilter('tok')
    gate.frame_navigated({'loaderId': 'L1'})
    start = {'token': 'tok', 'timeOrigin': 7.0, 'type': 'document_start'}
    settled = {'token': 'tok', 'timeOrigin': 7.0, 'type': 'settled'}
    assert gate.offer(start) is False and gate.offer(settled) is False and gate.rejected == []
    gate.navigation_sent('L1')
    assert [record['type'] for record in gate.accepted] == ['document_start', 'settled']
    other = probe.RecordFilter('tok')
    other.frame_navigated({'loaderId': 'OLD'})
    other.offer(dict(start))
    other.navigation_sent('L2')
    assert other.accepted == [] and [item['reason'] for item in other.rejected] == ['before_commit']
    assert other.offer({'token': 'old', 'timeOrigin': 1.0, 'type': 'settled'}) is False
    assert other.rejected[-1]['reason'] == 'token'


def test_native_play_gesture_window_and_unsubtracted_value():
    """F6: the native control's input events never reach page listeners: the play gesture's INP window is
    [ready, flush], a group there maps to it (none -> '<16 ms'), and the C11 preview value is (playing - start)."""
    run = make_run()
    run.facts.navigate_sent = True
    record(run, 'document_start', **focus_doc(True))
    run.steps = [{'step': step['step'], 'status': 'dispatched'} for step in VOICES_STEPS]
    run.steps[3].update(ready_record='control_ready', target={'kind': 'audio'}, probe_dispatch_delay_ms=3.0)
    for index, (first, last) in enumerate([(10, 12), (1010, 1012), (2010, 2012)]):
        for item in inputs(index, first, last):
            record(run, 'input', **item)
    record(run, 'control_attached', time=2040.0, readyState=0, target={'kind': 'audio', 'point': {'x': 1, 'y': 1}})
    record(run, 'control_ready', time=2060.0, readyState=1, target={'kind': 'audio', 'point': {'x': 1, 'y': 1}})
    record(run, 'media', event='playing', timeStamp=2100.0, readyState=4)
    record(run, 'focus', point='flush', visibility='visible', hasFocus=True, time=2500.0, t=2500.0)
    for interaction, start, duration in ((1, 11, 24.0), (2, 1011, 32.0), (3, 2011, 40.0), (4, 2063.0, 56.0)):
        record(run, 'perf', entry=event(interaction, 'click', start, duration, target=None))
    result = run.result()
    assert result['outcome'] == Outcome.VALID, result['facts']
    assert result['metrics']['inp'] == 56.0 and result['metrics']['inp_mapping']['groups_per_interaction'] == [1, 1, 1, 1]
    preview_value = result['metrics']['preview']
    assert preview_value['definition'] == 'native_unsubtracted' and preview_value['value'] == 2100.0 - 2010.0
    assert preview_value['gap'] is None and not preview_value['gap_invalid']
    slow = make_run()
    slow.facts.navigate_sent = True
    record(slow, 'document_start', **focus_doc(True))
    slow.steps = [{'step': 'reproducir', 'status': 'dispatched', 'ready_record': 'control_ready',
                   'target': {'kind': 'audio'}, 'probe_dispatch_delay_ms': 17.0}]
    assert 'preview_dispatch_delay_over_16ms' in slow.result()['facts']['environment_faults']


# ------------------------------------------------------------------------------------------
# Review round 6 (R6-1..R6-11): smoke inputs, composition, feedback window, Run-level checks
# ------------------------------------------------------------------------------------------

CHOSEN_RULE = {'dx_px': 24.0, 'y_fraction': 0.5, 'status': 'smoke_candidate'}


def write_smoke_prerequisites(path, *, rule=CHOSEN_RULE, items=('quiet_baseline', 'cold_focus', 'background_activity',
                                                                 'fixture_pages')):
    results = {'quiet_baseline': {'idle': 1.5}, 'cold_focus': {'passed': True},
               'background_activity': {'branch': 'none'},
               'fixture_pages': {'observations': {'play_control_rule': {'rule': rule}}}}
    runs = [{'machine_state': 'shared', 'item': item, 'status': 'complete', 'started_at': float(number), 'result': results[item], 'faults': []}
            for number, item in enumerate(items)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'mode': 'smoke', 'runs': runs}))
    return path


class FakeFixture:
    def __init__(self, out_dir, **kwargs):
        positions = out_dir / 'positions.json'
        positions.write_text(json.dumps({'preview_permutation': list(range(50)),
                                         'positions': [{'capture_id': f'vf_{index}'} for index in range(50)]}))
        self.ready = {'origin': 'http://127.0.0.1:1', 'tokens': {'home': 'h', 'operator': 'o', 'evaluator': 'e'},
                      'position_map': str(positions), 'digest': 'dig', 'started_at': time.time(), 'seed': 1}

    async def start(self):
        return self.ready

    async def stop(self):
        pass


class FakeSession:
    specs = []

    def __init__(self, mode, binary, out_dir, profile_root=None, netlog=False):
        self.netlog = netlog

    async def execute(self, spec, fixture):
        FakeSession.specs.append(spec)
        return {'ordinal': spec.ordinal, 'role': spec.role, 'outcome': Outcome.VALID, 'reason': None,
                'metrics': {'inp': 40.0, 'action_feedback_raf': 10.0}, 'facts': {'product_failures': []},
                'browser_version': 'Chrome/157', 'document_start': {'language': 'es', 'languages': ['es']}}

    async def close_browser(self):
        pass


def drive_measure_cell(monkeypatch, tmp_path, mode, *, fill_ins=None, view='#voces'):
    """measure_cell with every Mac side effect faked; run_batch runs the first non-warm-up slot only."""
    import tests.frontend.browser_support as support
    prereg = tmp_path / 'prereg.json'
    prereg.write_text(json.dumps({'fill_ins': fill_ins or {}}))
    monkeypatch.setattr(probe, 'PREREGISTRATION', prereg)
    monkeypatch.setattr(probe, 'SMOKE_ROOT', tmp_path / 'smoke')
    monkeypatch.setattr(probe, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(probe, 'HARNESS_ROOT', tmp_path / 'u1' / 'harness')
    monkeypatch.setattr(probe.sys, 'platform', 'darwin')
    monkeypatch.setattr(support, 'browser_binary', lambda: 'chrome')
    monkeypatch.setattr(probe, 'mac_environment', lambda: {'os_version': '26.0', 'hardware': 'Mac16,8',
                                                           'apple_languages_first': 'es'})
    monkeypatch.setattr(probe, 'pip_freeze_sha256', lambda: 'f' * 64)
    monkeypatch.setattr(probe, 'git', lambda *args: 'c' * 40)
    monkeypatch.setattr(probe, 'audio_output', lambda: {'name': 'Bocinas', 'transport': 'builtin', 'builtin': True})
    monkeypatch.setattr(probe, 'FixtureProcess', FakeFixture)
    monkeypatch.setattr(probe, 'BrowserSession', FakeSession)
    FakeSession.specs = []
    seen = {}

    async def fake_batch(plan, environment, **kwargs):
        seen.update(plan=plan, kwargs=kwargs)
        await environment.start_fixture()
        slot = next(slot for slot in plan['slots'] if not slot.get('warm_up'))
        run = await environment.run_slot(slot)
        return {'cell': plan['cell'], 'attempt': plan['attempt'], 'seed': plan['seed'], 'planned': plan['planned'],
                'runs': [run], 'discarded': False, 'reasons': []}
    monkeypatch.setattr(probe, 'run_batch', fake_batch)
    args = SimpleNamespace(viewport='desktop', view=view, attempt=1)
    return args, seen


async def test_smoke_batches_use_the_play_rule_chosen_by_fixture_pages(monkeypatch, tmp_path):
    """R6-1: a smoke batch reads play_control_rule from the smoke's own complete fixture_pages entry, refuses to
    start without it, sends it to the injected script and writes it into the batch header; the overhead runs use
    the same rule; items follow the preregistered smoke order."""
    args, seen = drive_measure_cell(monkeypatch, tmp_path, 'smoke')
    record_path = tmp_path / 'smoke' / 'smoke-record.json'
    write_smoke_prerequisites(record_path, items=('quiet_baseline', 'cold_focus', 'background_activity'))
    with pytest.raises(SystemExit, match='smoke_order_refused:batches: first complete fixture_pages'):
        await probe.measure_cell(args, 'smoke', SHA)
    write_smoke_prerequisites(record_path, rule=None)
    with pytest.raises(SystemExit, match='smoke_play_control_rule_missing'):
        await probe.measure_cell(args, 'smoke', SHA)
    write_smoke_prerequisites(record_path)
    report = await probe.measure_cell(args, 'smoke', SHA)
    (spec,) = FakeSession.specs
    assert spec.play_rule == CHOSEN_RULE != probe.PLAY_CONTROL_RULE and spec.frame_event is None
    config = probe.injection_config(token='t', origin='o', view='#voces', capture_id='c', play_rule=spec.play_rule)
    assert config['play_rule'] == CHOSEN_RULE
    header = json.loads((Path(report['directory']) / 'batch.json').read_text())
    assert header['play_control_rule'] == CHOSEN_RULE
    assert header['cell_settings']['play_control_rule_source'] == 'smoke_record:fixture_pages'
    # R7-1: the header names the deciding entries the inputs came from.
    assert header['cell_settings']['smoke_inputs']['fixture_pages'] == {'started_at': 3.0, 'reviewed_sha256': None}
    assert header['cell_settings']['smoke_inputs']['quiet_baseline']['started_at'] == 0.0
    # V14c, R6-8: every run result names its method_identity, the audio output included.
    run_file = json.loads(next(Path(report['directory']).glob('run-*.json')).read_text())
    assert run_file['method_identity']['audio_output']['builtin'] is True
    assert run_file['method_identity']['preregistration_sha256'] == SHA
    record = probe.SmokeRecord(record_path)
    assert probe.smoke_order_missing(record, 'instrumentation_overhead')[:1] == ['batch:desktop:#hogar']
    assert 'instrumentation_overhead' in probe.smoke_order_missing(record, 'frame_event')
    assert probe.smoke_order_missing(record, 'fixture_pages') == []
    assert probe.smoke_order_missing(probe.SmokeRecord(tmp_path / 'none.json'), 'cold_focus') == ['quiet_baseline']


def test_smoke_inputs_come_from_the_deciding_entry_and_no_item_repeats_after_a_later_stage(tmp_path):
    """R7-1 (run_rules.which_run_counts, smoke_order): the first complete entry without faults that started before
    any later-stage entry decides an item; a faulted complete entry never decides; no item starts once a later
    stage has an entry, so fixture_pages repeated after a batch is refused, and an entry written after a batch is
    never used."""
    other = {'x': 1, 'y': 2}
    path = write_smoke_prerequisites(tmp_path / 'smoke-record.json')
    data = json.loads(path.read_text())
    fixture = data['runs'][3]
    fixture['faults'] = ['quiet_machine']
    path.write_text(json.dumps(data))
    record = probe.SmokeRecord(path)
    # A complete entry with a recorded fault does not decide: the batches wait for a repeat.
    assert probe.deciding_entry(record, 'fixture_pages') is None
    assert 'fixture_pages' in probe.smoke_order_missing(record, 'batches')
    with pytest.raises(SystemExit, match='smoke_play_control_rule_missing'):
        probe.smoke_play_rule(record)
    repeat = record.start('fixture_pages', SHA, cause='quiet_machine')
    record.finish(repeat, 'complete', {'observations': {'play_control_rule': {'rule': other}}})
    record = probe.SmokeRecord(path)
    assert probe.smoke_play_rule(record) == other
    assert probe.smoke_inputs(record)['fixture_pages'] == {'started_at': repeat['started_at'], 'reviewed_sha256': SHA}
    # The deciding entry is the first fault-free one, not the last complete one.
    data = json.loads(path.read_text())
    data['runs'].append(dict(data['runs'][-1], started_at=data['runs'][-1]['started_at'] + 1,
                             result={'observations': {'play_control_rule': {'rule': CHOSEN_RULE}}}))
    path.write_text(json.dumps(data))
    assert probe.smoke_play_rule(probe.SmokeRecord(path)) == other
    # The idle load comes from the deciding quiet_baseline entry too, never a later or faulted one.
    data['runs'].append({'machine_state': 'shared', 'item': 'quiet_baseline', 'status': 'complete', 'faults': ['quiet_machine'],
                         'started_at': 99.0, 'result': {'idle': 9.9}})
    path.write_text(json.dumps(data))
    assert probe.smoke_idle(probe.SmokeRecord(path)) == 1.5
    # Once a batch has an entry, fixture_pages (and any earlier item) never starts again, even with a cause its last
    # entry recorded (the R7-1 example: caffeinate lost at the end of a complete fixture_pages entry).
    path = write_smoke_prerequisites(tmp_path / 'later.json')
    data = json.loads(path.read_text())
    data['runs'][3]['faults'] = ['quiet_machine']
    batch_time = 9.0
    data['runs'].append({'machine_state': 'shared', 'item': 'batch:desktop:#voces', 'status': 'complete', 'faults': [], 'started_at': batch_time})
    path.write_text(json.dumps(data))
    for item, cause in (('fixture_pages', 'quiet_machine'), ('quiet_baseline', None), ('cold_focus', None)):
        with pytest.raises(SystemExit, match=f'smoke_repeat_refused_after_later_stage:{item}:.*batch:desktop:#voces'):
            probe.SmokeRecord(path).start(item, SHA, cause=cause)
    probe.SmokeRecord(path).start('lock_read_unlocked', SHA)
    # An entry written after a batch entry (outside the probe) is never used.
    data = json.loads(path.read_text())
    data['runs'].append({'machine_state': 'shared', 'item': 'fixture_pages', 'status': 'complete', 'faults': [], 'started_at': batch_time + 1,
                         'result': {'observations': {'play_control_rule': {'rule': CHOSEN_RULE}}}})
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match='smoke_play_control_rule_missing'):
        probe.smoke_play_rule(probe.SmokeRecord(path))
    # A batch cell does not start once the overhead has an entry.
    data['runs'].append({'machine_state': 'shared', 'item': 'instrumentation_overhead', 'status': 'complete', 'faults': [],
                         'started_at': batch_time + 2})
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match='smoke_repeat_refused_after_later_stage:batch:mobile:#hogar'):
        probe.SmokeRecord(path).start('batch:mobile:#hogar', SHA)



def test_a_discarded_voces_cell_with_a_recorded_fault_is_repeated_before_the_overhead(tmp_path):
    """R8-1 (smoke_order, preview_ready_state row (d)): a #voces cell (desktop or mobile) with no complete entry
    whose last entry is discarded with a recorded fault holds the overhead and the frame event until it is repeated
    with that cause; a discarded #voces entry without a fault (no repeat possible) does not hold them, and stays
    unrepeatable."""
    path = write_smoke_prerequisites(tmp_path / 'smoke-record.json')
    data = json.loads(path.read_text())
    for number, cell in enumerate(probe.batch_plan_views()):
        item = f'batch:{cell["viewport"]}:{cell["view"]}'
        entry = {'machine_state': 'shared', 'item': item, 'status': 'complete', 'faults': [], 'started_at': 10.0 + number, 'cause': None}
        if item == 'batch:desktop:#voces':
            entry.update(status='discarded', faults=['quiet_machine'],
                         result={'reasons': ['quiet_machine', 'invalid_runs_over_10_percent']})
        data['runs'].append(entry)
    path.write_text(json.dumps(data))
    record = probe.SmokeRecord(path)
    pending = ['batch:desktop:#voces(repeat --cause quiet_machine)']
    assert probe.f7_repeat_pending(record) == pending
    assert probe.smoke_order_missing(record, 'instrumentation_overhead') == pending
    assert pending[0] in probe.smoke_order_missing(record, 'frame_event')
    with pytest.raises(SystemExit, match=r'smoke_order_refused:instrumentation_overhead: first complete '
                                         r'batch:desktop:#voces\(repeat --cause quiet_machine\)'):
        probe.check_smoke_order(record, 'instrumentation_overhead')
    # The batches stage itself is not held: the client repeats the cell with its recorded cause.
    assert probe.smoke_order_missing(record, 'batches') == []
    assert probe.smoke_cell_action(record, 'batch:desktop:#voces', None) == ('skip', None)
    assert probe.smoke_cell_action(record, 'batch:desktop:#voces', 'quiet_machine') == ('run', 'quiet_machine')
    repeat = record.start('batch:desktop:#voces', SHA, cause='quiet_machine')
    record.finish(repeat, 'complete', {'reasons': []})
    record = probe.SmokeRecord(path)
    assert probe.f7_repeat_pending(record) == []
    assert probe.smoke_order_missing(record, 'instrumentation_overhead') == []
    probe.check_smoke_order(record, 'instrumentation_overhead')
    # The mobile #voces cell holds the overhead the same way (a lock recorded at the batch start).
    data = json.loads(path.read_text())
    mobile = next(run for run in data['runs'] if run['item'] == 'batch:mobile:#voces')
    mobile.update(status='discarded', faults=['lock'])
    path.write_text(json.dumps(data))
    assert probe.smoke_order_missing(probe.SmokeRecord(path), 'instrumentation_overhead') == [
        'batch:mobile:#voces(repeat --cause lock)']
    # A discarded #voces entry without a recorded fault (invalid share from unmapped environment faults) cannot be
    # repeated, so it does not hold the overhead: row (d) with no repeat possible is read as preregistered.
    mobile.update(faults=[], result={'reasons': ['invalid_runs_over_10_percent']})
    path.write_text(json.dumps(data))
    record = probe.SmokeRecord(path)
    assert probe.f7_repeat_pending(record) == []
    assert probe.smoke_order_missing(record, 'instrumentation_overhead') == []
    assert probe.smoke_cell_action(record, 'batch:mobile:#voces', 'quiet_machine') == ('skip', None)
    with pytest.raises(SystemExit, match='smoke_repeat_refused_without_recorded_cause:batch:mobile:#voces'):
        record.start('batch:mobile:#voces', SHA, cause='quiet_machine')
    # A discarded non-#voces cell with a recorded fault does not hold the overhead (F7 does not read it).
    other = next(run for run in data['runs'] if run['item'] == 'batch:desktop:#hogar')
    other.update(status='discarded', faults=['quiet_machine'])
    mobile.update(status='complete')
    path.write_text(json.dumps(data))
    assert probe.smoke_order_missing(probe.SmokeRecord(path), 'instrumentation_overhead') == []

async def test_baseline_needs_every_batch_fill_in_and_uses_the_frozen_composition(monkeypatch, tmp_path):
    """R6-1, R6-2: baseline refuses without play_control_rule (and the other batch fill-ins); the composition is
    3 warm-ups, 40 pairs and 10 contrast pairs, the smoke's 1/5/2, never a CLI value."""
    fill_ins = {'idle_load': 1.0, 'post_batch_load': 2.0, 'background_branch': 'none',
                'frame_event': {'candidate': 'PipelineReporter'}}
    args, seen = drive_measure_cell(monkeypatch, tmp_path, 'baseline', fill_ins=fill_ins)
    with pytest.raises(SystemExit, match='fill_ins_missing:play_control_rule'):
        await probe.measure_cell(args, 'baseline', SHA)
    fill_ins['play_control_rule'] = CHOSEN_RULE
    (tmp_path / 'prereg.json').write_text(json.dumps({'fill_ins': fill_ins}))
    await probe.measure_cell(args, 'baseline', SHA)
    plan = seen['plan']
    assert plan['planned'] == 100 and sum(1 for slot in plan['slots'] if slot.get('warm_up')) == 3
    assert sum(1 for slot in plan['slots'] if slot.get('trace_contrast')) == 20
    assert FakeSession.specs[-1].play_rule == CHOSEN_RULE and FakeSession.specs[-1].frame_event == 'PipelineReporter'
    assert seen['kwargs']['warm_rule'] is True
    write_smoke_prerequisites(tmp_path / 'smoke' / 'smoke-record.json')
    await probe.measure_cell(args, 'smoke', SHA)
    plan = seen['plan']
    assert plan['planned'] == 14 and sum(1 for slot in plan['slots'] if slot.get('warm_up')) == 1
    assert sum(1 for slot in plan['slots'] if slot.get('trace_contrast')) == 4
    # R6-9: no warm-share discard in the smoke.
    # quiet_machine.shared: no end-reading threshold in either mode, so no end_record_only switch exists.
    assert seen['kwargs']['warm_rule'] is False and 'end_record_only' not in seen['kwargs']
    with pytest.raises(SystemExit):
        probe.parse_args(['measure-cell', '--smoke', '--reviewed-sha256', SHA, '--pairs', '5'])
    with pytest.raises(SystemExit):
        probe.parse_args(['measure-client', '--frozen-sha256', SHA, '--contrast-pairs', '3'])


async def test_smoke_client_resumes_after_a_crash_with_the_recorded_cause(monkeypatch, tmp_path):
    """R6-10: complete cells are skipped, the crashed cell is repeated only with --cause, the cause is written."""
    monkeypatch.setattr(probe, 'SMOKE_ROOT', tmp_path / 'smoke')
    monkeypatch.setattr(probe, 'checked_output', lambda mode, path: Path(path))
    path = write_smoke_prerequisites(tmp_path / 'smoke' / 'smoke-record.json')
    order = probe.cell_order(SHA, 'mac-chrome', 1, [(cell['viewport'], cell['view'])
                                                    for cell in probe.batch_plan_views()], smoke=True)
    data = json.loads(path.read_text())
    for viewport, view in order[:3]:
        data['runs'].append({'machine_state': 'shared', 'item': f'batch:{viewport}:{view}', 'status': 'complete', 'faults': [], 'started_at': 9.0})
    data['runs'].append({'machine_state': 'shared', 'item': f'batch:{order[3][0]}:{order[3][1]}', 'status': 'started', 'started_at': 10.0})
    path.write_text(json.dumps(data))
    calls = []

    async def fake_cell(args, mode, sha):
        calls.append((args.viewport, args.view, args.attempt))
        return {'discarded': False, 'reasons': [], 'faults': [], 'directory': 'd'}
    monkeypatch.setattr(probe, 'measure_cell', fake_cell)
    with pytest.raises(SystemExit, match='smoke_repeat_refused'):
        await probe.measure_client(SimpleNamespace(cause=None), 'smoke', SHA)
    assert calls == []
    await probe.measure_client(SimpleNamespace(cause='crash'), 'smoke', SHA)
    assert [(viewport, view) for viewport, view, _ in calls] == order[3:]
    assert calls[0][2] == 2 and all(attempt == 1 for _, _, attempt in calls[1:])
    saved = json.loads(path.read_text())['runs']
    repeated = [run for run in saved if run['item'] == f'batch:{order[3][0]}:{order[3][1]}']
    assert repeated[-1]['cause'] == 'crash' and repeated[-1]['status'] == 'complete'
    assert all(run.get('cause') is None for run in saved if run['item'] == f'batch:{order[4][0]}:{order[4][1]}')


async def test_action_feedback_keeps_the_timed_window_open_for_the_feedback_frame():
    """R6-3: after the feedback step's next frame the probe waits for the injected feedback_window record (the later
    of 100 ms and 6 frames after the mutation) before the flush."""
    run = make_run(view='#acciones', capture=None)
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    record(run, 'target_ready', index=0, rect=rect, step='accion', in_viewport=True)
    record(run, 'step_next_frame', index=0)

    def close_window():
        run.records.offer(json.dumps({'token': run.token, 'timeOrigin': 1000.0, 'type': 'feedback_window', 'index': 0,
                                      'frames': 6, 't': 1.0, 'seq': 99}))
    asyncio.get_running_loop().call_later(0.4, close_window)
    started = time.monotonic()
    await run._interactions(run.config['steps'])
    assert time.monotonic() - started >= 0.39 and run.steps[0]['feedback_window'] == 'closed'
    assert run.config['feedback_window'] == {'ms': 100, 'frames': 6}
    source = probe.INJECTED_SOURCE
    chain = source[source.index("if (step.c11 === 'action_feedback')"):source.index("emit('feedback_window'") + 40]
    assert 'CONFIG.feedback_window.frames' in chain and 'now() - time >= CONFIG.feedback_window.ms' in chain


async def test_feedback_without_a_chosen_frame_after_the_mutation_is_a_harness_fault():
    """R6-3: mutation and rAF present, window closed, trace intact, but no chosen-candidate frame after the
    mutation -> invalid run (harness fault), never a product failure; never applied in the smoke."""
    assert probe.feedback_frame_missing({'mutation': 10.0, 'raf': 12.0, 'presented': {'PipelineReporter': None}},
                                        'PipelineReporter', True, True)
    assert not probe.feedback_frame_missing({'mutation': 10.0, 'raf': 12.0, 'presented': {'PipelineReporter': 30.0}},
                                            'PipelineReporter', True, True)
    assert not probe.feedback_frame_missing({'mutation': 10.0, 'raf': 12.0, 'presented': {}}, None, True, True)
    assert not probe.feedback_frame_missing({'mutation': 10.0, 'raf': 12.0, 'presented': {}}, 'DrawFrame', True, False)
    for frame_event, expected in (('PipelineReporter', Outcome.INVALID), (None, Outcome.VALID)):
        run = make_run(view='#acciones', capture=None)
        run.spec.frame_event = frame_event
        run.steps = [{'step': 'accion', 'status': 'dispatched'}]
        run.facts.navigate_sent = True
        record(run, 'document_start', **focus_doc(True))
        record(run, 'feedback_window', index=0, frames=6)
        for item in inputs(0, 10, 12):
            record(run, 'input', **item)
        record(run, 'step_end', index=0, time=40.0)
        record(run, 'step_frame', index=0, raf=50.0, frame_time=45.0)
        run._trace_intact = lambda: True
        run._c11 = lambda typed, interactions, steps: {'action_feedback': {
            'start': 10.0, 'mutation': 40.0, 'raf': 50.0, 'presented': {'PipelineReporter': None}}}
        result = run.result()
        assert result['outcome'] == expected, result['reason']
        if expected == Outcome.INVALID:
            assert result['reason'] == 'feedback_frame_absent_from_trace'


def test_pipeline_reporter_counts_only_presented_all():
    """R6-3: an 'X' PipelineReporter slice without a state (or another state) is not a presented frame."""
    events = [{'name': 'PipelineReporter', 'ph': 'X', 'pid': 1, 'ts': 1_000_000, 'dur': 5_000, 'args': {}},
              {'name': 'PipelineReporter', 'ph': 'X', 'pid': 1, 'ts': 1_010_000, 'dur': 5_000,
               'args': {'chrome_frame_reporter': {'state': 'STATE_DROPPED'}}},
              {'name': 'PipelineReporter', 'ph': 'X', 'pid': 1, 'ts': 1_020_000, 'dur': 5_000,
               'args': {'chrome_frame_reporter': {'state': 'STATE_PRESENTED_ALL'}}}]
    found = probe.frame_candidates(events)['PipelineReporter']
    assert [item['time'] for item in found] == [1025.0]


async def test_completed_script_with_failed_preview_keeps_lcp_inp_and_cls_finite():
    """R6-5 (R6M04, V1 in Run.result): every core step dispatched and the preview never plays -> the run is a
    failure of C11 preview only: LCP, INP and CLS stay finite and enter their percentiles."""
    run = make_run()
    run.steps = [{'step': step['step'], 'status': 'dispatched'} for step in VOICES_STEPS]
    run.facts.product_failures.append('preview_failed_open:no_playing')
    run.facts.navigate_sent = True
    record(run, 'document_start', **focus_doc(True))
    record(run, 'perf', entry={'entryType': 'largest-contentful-paint', 'startTime': 300.0, 'renderTime': 300.0})
    record(run, 'perf', entry={'entryType': 'layout-shift', 'startTime': 200.0, 'value': 0.01,
                               'hadRecentInput': False})
    for index, (first, last) in enumerate([(1000, 1002), (2000, 2002), (3000, 3002), (3010, 3012)]):
        for item in inputs(index, first, last):
            record(run, 'input', **item)
    record(run, 'perf', entry=event(1, 'click', 1001, 48.0))
    result = run.result()
    assert result['outcome'] == Outcome.FAILURE and not result['run_wide_failure']
    assert result['metrics']['inp'] == 48.0 and result['metrics']['cls'] == pytest.approx(0.01)
    assert result['metrics']['lcp'] == 300.0 and result['metrics']['preview']['value'] == INF
    for metric, expected in (('lcp', 300.0), ('inp', 48.0)):
        assert probe.metric_values([result], metric) == [expected]


class PipelineClient(FakeClient):
    """Input events are accepted only inside one send_pipelined call (one gesture = one batch)."""

    def __init__(self, answers=None):
        super().__init__(answers)
        self.batches = []
        self._inside = False

    async def send(self, method, params=None, *, klass, phase=None):
        if method.startswith('Input.') and not self._inside:
            raise AssertionError('input event sent outside a pipelined gesture')
        return await super().send(method, params, klass=klass, phase=phase)

    async def send_pipelined(self, commands, *, klass):
        self.batches.append([method for method, _ in commands])
        self._inside = True
        try:
            return [await self.send(method, params, klass=klass) for method, params in commands]
        finally:
            self._inside = False


async def test_every_gesture_is_one_pipelined_batch_and_play_target_outside_viewport_is_never_clicked():
    """R6-5 (R6M07, F3; R6M09, V3): each click's events go out in one pipelined write (mouse: 3, touch: 2); a play
    target whose point lies outside the viewport is a harness fault and is never clicked."""
    run = make_run()
    run.page = PipelineClient()
    run.page_log_start = 0
    rect = {'x': 10, 'y': 20, 'width': 40, 'height': 30}
    for index in range(3):
        record(run, 'target_ready', index=index, rect=rect, step=VOICES_STEPS[index]['step'])
    for index in range(2):
        record(run, 'step_next_frame', index=index)
    record(run, 'control_attached', time=500.0, readyState=1,
           target={'kind': 'audio', 'rect': rect, 'point': {'x': 31.0, 'y': 36.0}, 'in_viewport': True})
    record(run, 'media', event='playing', timeStamp=600.0, readyState=4)
    await run._interactions(run.config['steps'])
    assert run.page.batches == [['Input.dispatchMouseEvent'] * 3] * 4 and not run.facts.product_failures
    mobile = make_run(viewport='mobile')
    mobile.page = PipelineClient()
    await probe.dispatch_click(mobile.page, {'x': 1, 'y': 2}, True)
    assert mobile.page.batches == [['Input.dispatchTouchEvent'] * 2]
    outside = make_run()
    for index in range(3):
        record(outside, 'target_ready', index=index, rect=rect, step=VOICES_STEPS[index]['step'])
    for index in range(2):
        record(outside, 'step_next_frame', index=index)
    record(outside, 'control_attached', time=500.0, readyState=1,
           target={'kind': 'audio', 'rect': rect, 'point': {'x': 31.0, 'y': 2000.0}, 'in_viewport': False})
    await outside._interactions(outside.config['steps'])
    assert len([entry for entry in outside.page.log if entry['method'].startswith('Input.')]) == 9
    assert outside.facts.environment_faults == ['target_outside_viewport:reproducir']


def test_frame_event_acceptance_needs_the_first_raf_bound_and_counts_empty_traces():
    """R6-5 (R6M11, R5-6): a candidate frame before the first-rAF lower bound fails even within 16.7 ms of the
    screencast frame. R6-11(d): a traced interaction phase whose trace has no event at all counts as absence."""
    early = {'metrics': {'action_feedback': {'raf': 100.0, 'screencast': 105.0,
                                             'presented': {'PipelineReporter': 95.0}}},
             'trace_candidates_present': ['PipelineReporter'], 'trace_events': 300, 'interaction_traced': True,
             'steps': [{'status': 'dispatched'}]}
    phases = probe.frame_event_phases([early])
    assert phases[0]['interactions'] == [{'PipelineReporter': False}]
    assert probe.frame_event_choice(phases)['candidate'] is None
    empty = {'metrics': {}, 'trace_candidates_present': [], 'trace_events': 0, 'interaction_traced': True,
             'steps': [{'status': 'dispatched'}]}
    untraced = dict(empty, interaction_traced=False)
    assert probe.frame_event_phases([empty, untraced]) == [{'present': set(), 'interactions': []}]


async def test_overhead_arms_trace_a_and_b_and_screencast_only_a(monkeypatch, tmp_path):
    """R6-5 (R6M12, smoke_gate.instrumentation_overhead): arm (a) traced with screencast, (b) traced without, (c)
    untraced; primers untraced; every run uses the smoke's chosen play rule (R6-1)."""
    import tests.frontend.browser_support as support
    monkeypatch.setattr(probe, 'SMOKE_ROOT', tmp_path / 'smoke')
    monkeypatch.setattr(probe, 'checked_output', lambda mode, path: Path(path))
    monkeypatch.setattr(support, 'browser_binary', lambda: 'chrome')
    monkeypatch.setattr(probe, 'FixtureProcess', FakeFixture)
    monkeypatch.setattr(probe, 'BrowserSession', FakeSession)
    FakeSession.specs = []
    result = await probe.smoke_overhead(SHA, CHOSEN_RULE)
    warm = [spec for spec in FakeSession.specs if spec.role == 'warm']
    primers = [spec for spec in FakeSession.specs if spec.role == 'cold']
    assert len(warm) == 60 and len(primers) == 60
    assert all(not spec.trace_interactions and not spec.screencast for spec in primers)
    by_arm = defaultdict(set)
    for path in Path(result['directory']).glob('*-run-*.json'):
        document = json.loads(path.read_text())
        if document['role'] == 'warm':
            spec = next(spec for spec in warm if spec.ordinal == document['ordinal']
                        and spec.view.strip('#') == path.name.split('-run-')[0])
            by_arm[document['arm']].add((spec.trace_interactions, spec.screencast))
    assert by_arm == {'a': {(True, True)}, 'b': {(True, False)}, 'c': {(False, False)}}
    assert all(spec.play_rule == CHOSEN_RULE for spec in FakeSession.specs)
    assert result['play_control_rule'] == CHOSEN_RULE


async def test_window_reads_are_awaited_before_collection():
    """R6-5 (R6M17, V4): pending window-state reads finish before the first collection command."""
    run = make_run(view='#rutinas', capture=None)
    order = []

    async def slow_read():
        await asyncio.sleep(0.2)
        order.append('read')
        run.window_reads.append({'point': 'flush', 'state': 'normal'})
    run.window_tasks.append(asyncio.get_running_loop().create_task(slow_read()))
    original = run.browser.client.send

    async def send(method, params=None, *, klass, phase=None):
        order.append(method)
        return await original(method, params, klass=klass, phase=phase)
    run.browser.client.send = send
    await run._collect()
    assert order[0] == 'read' and run.window_reads


def test_audio_output_is_recorded_required_builtin_and_part_of_method_identity():
    """R6-8: the default CoreAudio output is read per batch, must be the built-in one and enters method_identity."""
    text = json.dumps({'SPAudioDataType': [{'_items': [
        {'_name': 'AirPods', 'coreaudio_device_transport': 'coreaudio_device_type_bluetooth'},
        {'_name': 'Bocinas de MacBook Pro', 'coreaudio_default_audio_output_device': 'spaudio_yes',
         'coreaudio_device_transport': 'coreaudio_device_type_builtin'}]}]})
    assert probe.parse_audio_output(text) == {'name': 'Bocinas de MacBook Pro',
                                              'transport': 'coreaudio_device_type_builtin', 'builtin': True}
    assert probe.parse_audio_output('not json') is None
    healthy = {'ac_power': True, 'display_asleep': False, 'locked': False}
    builtin = {'name': 'x', 'transport': 'coreaudio_device_type_builtin', 'builtin': True}
    bluetooth = {'name': 'y', 'transport': 'coreaudio_device_type_bluetooth', 'builtin': False}
    assert probe.machine_faults({**healthy, 'audio_output': builtin}, smoke=False) == []
    assert probe.machine_faults({**healthy, 'audio_output': bluetooth}, smoke=True) == ['audio_output_not_builtin']
    assert probe.machine_faults({**healthy, 'audio_output': None}, smoke=False) == ['audio_output_not_builtin']
    assert probe.machine_faults({**healthy, 'audio_output': None}, smoke=True) == []
    identity = {key: 'same' for key in probe.IDENTITY_COMPONENTS}
    changed = dict(identity, audio_output='other')
    decision = probe.before_after({'closure': [], 'identity': identity}, changed, [])
    assert decision['decision'] == 'rerun_before' and decision['reasons'] == ['identity:audio_output']


def test_round6_mechanical_gaps():
    """R6-11 (b) the Escuchar selector scopes both alternatives by the capture id; (c) the flush focus sample stays
    in the timed records; (e) every fixture page the checks load exists; R6-7 the fixture-503 views are labelled and
    never compared."""
    (escuchar,) = [step for step in VOICES_STEPS if step['step'] == 'escuchar']
    resolved = probe.injection_config(token='t', origin='o', view='#voces', capture_id='vf_1')['steps'][2]
    assert resolved['selector'] == '[data-capture-id="vf_1"] button,[data-capture-id="vf_1"] [role="button"]'
    flush = {'point': 'flush', 'time': 900.0, 't': 900.4, 'hasFocus': True, 'visibility': 'visible'}
    late = {'event': 'blur', 'timeStamp': 950.0, 't': 950.1}
    assert probe.timed_records([flush, late], 900.0) == [flush]
    source = Path(probe.__file__).read_text()
    pages = set(re.findall(r"fixture_spec\(\d+[^,]*, '(?:cold|warm)', ready, '([a-z-]+)'", source))
    assert pages and all((REPOSITORY / 'tools' / 'web_timing_pages' / f'{page}.html').exists() for page in pages)
    rows = probe.summary_rows([{'viewport': 'desktop', 'view': '#rutinas', 'role': 'cold', 'outcome': Outcome.VALID,
                                'metrics': {'lcp': 1.0, 'cls': 0.0}}], None)
    assert rows[0]['fixture_note'] == probe.FIXTURE_UNAVAILABLE_NOTE and rows[0]['c10_acceptance'] == 'excluido'
    assert probe.comparable_rows([{'view': '#rutinas'}, {'view': '#hogar'}]) == [{'view': '#hogar'}]


async def test_dialog_variant_and_play_click_records(tmp_path):
    """R6-6: the dialog fixture variant reproduces the NEXO desktop context; the play target carries the element hit
    at the dispatch point and the first screencast frame at the click is kept."""
    page = (REPOSITORY / 'tools' / 'web_timing_pages' / 'preview.html').read_text()
    assert "query.get('dialog') === '1'" in page and 'showModal()' in page and 'scrollTop = 240' in page
    assert 'minmax(0,1fr) minmax(0,1fr)' in page and 'overflow:auto' in page
    assert 'hit: hitAt(point)' in probe.INJECTED_SOURCE
    run = make_run()
    run.out_dir = tmp_path
    run.play_click_wall = 100.0
    await run._frame({'sessionId': 1, 'metadata': {'timestamp': 99.0}, 'data': 'AAAA'})
    assert run.play_click_frame is None
    await run._frame({'sessionId': 2, 'metadata': {'timestamp': 100.02}, 'data': 'AAEC'})
    assert run.play_click_frame['timestamp'] == 100.02
    assert (tmp_path / 'play-click-00007.jpg').read_bytes() == b'\x00\x01\x02'
    runs = [{'ordinal': 1, 'outcome': Outcome.VALID, 'reason': None, 'play_click_frame': None,
             'facts': {'product_failures': [], 'environment_faults': []},
             'records': {'accepted': [{'type': 'control_ready', 'target': {'kind': 'audio', 'hit': 'audio'}},
                                      {'type': 'media', 'event': 'playing'}]}}]
    summary = probe.dialog_variant_summary(runs, CHOSEN_RULE)
    assert summary['plays'] is True and summary['runs'][0]['target']['hit'] == 'audio'
    runs[0]['records']['accepted'].pop()
    assert probe.dialog_variant_summary(runs, CHOSEN_RULE)['plays'] is False


async def test_click_uses_the_latest_reported_target_rectangle():
    """R6-5 (R6M03): after a scroll or layout change the target is re-reported; the click uses the latest rectangle,
    never the first (stale, pre-scroll) one."""
    run = make_run(view='#acciones', capture=None)
    record(run, 'target_ready', index=0, rect={'x': 10, 'y': 2000, 'width': 40, 'height': 30}, step='accion',
           in_viewport=False)
    record(run, 'target_ready', index=0, rect={'x': 100, 'y': 300, 'width': 40, 'height': 30}, step='accion',
           in_viewport=True)
    record(run, 'step_next_frame', index=0)
    record(run, 'feedback_window', index=0, frames=6)
    await run._interactions(run.config['steps'])
    first = next(entry for entry in run.page.log if entry['method'].startswith('Input.'))
    assert first['params'] == {'type': 'mouseMoved', 'x': 120.0, 'y': 315.0}
    assert run.steps[0]['rect'] == {'x': 100, 'y': 300, 'width': 40, 'height': 30} and not run.facts.environment_faults


# ------------------------------------------------------------------------------------------
# Shared-machine amendment (quiet_machine.shared, user decision 2026-11-17)
# ------------------------------------------------------------------------------------------

async def test_shared_machine_records_covariates_and_never_gates(tmp_path, monkeypatch):
    """quiet_machine.shared: foreign processes (the user's Chrome, other projects' pytest and builds, agent workers)
    and any load are covariates written into every header with own-versus-foreign counts; they never abort, discard
    or invalidate. The validity gates (AC power, display, lock, caffeinate) still discard. machine_state 'shared'
    is part of method_identity; shared results are never summarised with or compared to another machine state;
    the shared smoke starts with the quiet_baseline repeat --cause quiet_machine and only shared entries decide."""
    rows = [{'pid': 1, 'ppid': 0, 'command': '/sbin/launchd'},
            {'pid': 11, 'ppid': 1, 'command': '/bin/zsh'},
            {'pid': 12, 'ppid': 11, 'command': 'python -m tools.web_timing_probe smoke-item quiet-baseline'},
            {'pid': 30, 'ppid': 1, 'command': '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'},
            {'pid': 31, 'ppid': 30, 'command': '/A/Google Chrome Helper (Renderer) --type=renderer'},
            {'pid': 40, 'ppid': 1, 'command': '/opt/x/.venv/bin/python -m pytest tests'},
            {'pid': 41, 'ppid': 1, 'command': '/usr/local/bin/node /x/npm test'},
            {'pid': 50, 'ppid': 1, 'command': '/usr/local/bin/codex'},
            {'pid': 51, 'ppid': 50, 'command': '/bin/zsh -c make'}]
    sweep = probe.sweep_probe_browsers
    covariates = probe.machine_covariates(rows, 12, cpu={12: 3.0, 40: 80.0, 30: 15.0})
    assert covariates['machine_state'] == 'shared' and covariates['gating'] is False
    counts = covariates['counts_by_class']
    assert counts['chrome'] == 1 and counts['pytest'] == 1 and counts['build_or_test'] == 1
    assert counts['agent_worker'] == 1 and counts['cpu_over_10_percent'] == 2 and counts['uvicorn'] == 0
    assert covariates['own'] == {'processes': 3, 'cpu_percent': 3.0}
    assert covariates['foreign'] == {'processes': 6, 'cpu_percent': 95.0}
    # Batches: a busy machine and loads far above the old thresholds are recorded, never discard.
    busy = BatchFakes(start_ok=False, end_load=9.0)
    environment = busy.environment()
    environment.loadavgs = lambda: {'1m': 9.0, '5m': 8.0, '15m': 7.0}
    plan = lambda: probe.plan_batch(SHA, 'mac-chrome', 'desktop', '#rutinas', 1, pairs=1, contrast_pairs=0,
                                    warm_ups=0)
    result = await probe.run_batch(plan(), environment, idle=0.2, post_batch=0.3)
    assert not result['discarded'] and result['reasons'] == [] and 'fixture_start' in busy.events
    assert result['machine_state'] == 'shared'
    for point, reference in (('start', 0.2), ('end', 0.3)):
        recorded = result[point]['covariates']
        assert recorded['quiet_rule_would_hold'] is False and recorded['reference_load'] == reference
        assert recorded['load_averages'] == {'1m': 9.0, '5m': 8.0, '15m': 7.0} and recorded['gating'] is False
        assert recorded['counts_by_class'] == {'chrome': 1}
    assert result['end']['covariates']['load_1m'] == 9.0
    unplugged = BatchFakes(start_ok=False)
    environment = unplugged.environment()
    environment.machine_state = lambda: {'ac_power': False, 'display_asleep': False, 'locked': False}
    gated = await probe.run_batch(plan(), environment, idle=0.2, post_batch=0.3)
    assert gated['discarded'] and gated['reasons'] == ['quiet_machine'] and 'fixture_start' not in unplugged.events
    # Smoke item start: foreign processes and load are recorded only; AC power still gates.
    async def foreign(own_pid, allowed=None):
        return {'violations': [('chrome', 30)], 'pytest_count': 0, 'quiet': False,
                'covariates': probe.machine_covariates(rows, 12)}
    monkeypatch.setattr(probe, 'quiet_check', foreign)
    monkeypatch.setattr(probe, 'sweep_probe_browsers', lambda rows=None, kill=None: [])
    monkeypatch.setattr(probe, 'loadavg', lambda: 9.0)
    monkeypatch.setattr(probe, 'loadavgs', lambda: {'1m': 9.0, '5m': 8.0, '15m': 7.0})
    state = {'ac_power': True, 'display_asleep': False, 'locked': None}
    monkeypatch.setattr(probe, 'machine_state', lambda lock_choice=None: dict(state))
    start = await probe.smoke_item_quiet_start(0.2)
    assert start['ok'] is True and start['covariates']['counts_by_class']['chrome'] == 1
    assert start['covariates']['load_1m'] == 9.0 and start['machine_state'] == 'shared'
    state['ac_power'] = False
    assert (await probe.smoke_item_quiet_start(0.2))['ok'] is False
    # The shared smoke starts with the quiet_baseline repeat --cause quiet_machine after the aborted quiet entry.
    path = tmp_path / 'smoke-record.json'
    path.write_text(json.dumps({'mode': 'smoke', 'runs': [
        {'item': 'quiet_baseline', 'status': 'aborted', 'faults': ['quiet_machine'], 'started_at': 1.0,
         'cause': None, 'repeat_of': [], 'reviewed_sha256': 'e' * 64,
         'result': {'check': {'ok': False}, 'readings': []}}]}))
    record = probe.SmokeRecord(path)
    with pytest.raises(SystemExit, match='smoke_repeat_refused_without_recorded_cause:quiet_baseline'):
        await probe.smoke_quiet_baseline(record, SHA, None, check=lambda: foreign(12), read_load=lambda: 1.0,
                                         read_loads=dict, sleep=lambda seconds: asyncio.sleep(0))
    loads = iter([3.0, 9.0, 1.0, 4.0, 2.0])
    baseline = await probe.smoke_quiet_baseline(record, SHA, 'quiet_machine', check=lambda: foreign(12),
                                                read_load=lambda: next(loads), read_loads=dict,
                                                sleep=lambda seconds: asyncio.sleep(0))
    assert baseline['idle'] == 3.0 and len(baseline['covariates']) == 5
    assert all(item['quiet_rule_would_hold'] is False for item in baseline['covariates'])
    saved = json.loads(path.read_text())['runs']
    assert [run['status'] for run in saved] == ['aborted', 'complete']
    assert saved[1]['cause'] == 'quiet_machine' and saved[1]['machine_state'] == 'shared'
    assert 'machine_state' not in saved[0] and probe.smoke_idle(probe.SmokeRecord(path)) == 3.0
    # Only an entry recorded under the shared state decides.
    quiet = write_smoke_prerequisites(tmp_path / 'quiet.json')
    data = json.loads(quiet.read_text())
    for run in data['runs']:
        run.pop('machine_state')
    quiet.write_text(json.dumps(data))
    assert probe.deciding_entry(probe.SmokeRecord(quiet), 'quiet_baseline') is None
    assert probe.deciding_entry(probe.SmokeRecord(write_smoke_prerequisites(tmp_path / 'shared.json')),
                                'quiet_baseline')['result'] == {'idle': 1.5}
    # Summaries refuse files of another machine state and label shared rows; before/after never mixes states.
    directory = tmp_path / 'batch'
    directory.mkdir()
    run = {'mode': 'smoke', 'viewport': 'desktop', 'view': '#hogar', 'role': 'cold', 'outcome': 'valid',
           'metrics': {'lcp': 1.0, 'cls': 0.0}, 'method_identity': {'machine_state': 'shared'}}
    (directory / 'run-00000.json').write_text(json.dumps(run))
    monkeypatch.setattr(probe, 'PREREGISTRATION', tmp_path / 'absent.json')
    summary = probe.summarize_directory(directory, 'smoke')
    assert summary['machine_state'] == 'shared' and summary['rows'][0]['machine_state'] == 'shared'
    for other in ({'machine_state': 'quiet'}, {}):
        (directory / 'run-00001.json').write_text(json.dumps(dict(run, method_identity=other)))
        with pytest.raises(SystemExit, match='machine_state_refused'):
            probe.summarize_directory(directory, 'smoke')
    assert 'machine_state' in probe.IDENTITY_COMPONENTS
    identity = {key: 'same' for key in probe.IDENTITY_COMPONENTS}
    decision = probe.before_after({'closure': [], 'identity': dict(identity, machine_state=None)},
                                  dict(identity, machine_state='shared'), [])
    assert decision['decision'] == 'rerun_before' and decision['reasons'] == ['identity:machine_state']
    # The frame-event acceptance reads shared smoke runs only.
    monkeypatch.setattr(probe, 'frame_event_phases', lambda results: [{'count': len(results)}])
    monkeypatch.setattr(probe, 'frame_event_choice', lambda phases: phases[0])
    root = tmp_path / 'smoke'
    root.mkdir()
    (root / 'a-run-00001.json').write_text(json.dumps({'mode': 'smoke', 'outcome': 'valid', 'machine_state': 'shared'}))
    (root / 'b-run-00002.json').write_text(json.dumps({'mode': 'smoke', 'outcome': 'valid'}))
    assert probe.smoke_frame_event(root)['choice'] == {'count': 1}
    # The probe-own sweep never touches the user's Chrome.
    killed = []
    assert sweep(rows, kill=lambda pid, sig: killed.append(pid)) == [] and killed == []
