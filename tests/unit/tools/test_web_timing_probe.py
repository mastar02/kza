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

    async def send(self, method, params=None, *, klass):
        if not probe.method_class_allowed(method, klass):
            raise ValueError(f'no_deadline_class:{method}:{klass}')
        self.log.append({'method': method, 'params': params, 'class': klass, 'phase': self.phase,
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
    browser = SimpleNamespace(client=FakeClient(), alive=lambda: True, argv=[])
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
            {'outcome': Outcome.FAILURE, 'role': 'cold', 'metrics': {'lcp': 50.0}},
            {'outcome': Outcome.POST_WINDOW_HANG, 'role': 'cold', 'metrics': {}},
            {'outcome': Outcome.INVALID, 'role': 'cold', 'metrics': {'lcp': 1.0}}]
    assert probe.metric_values(runs, 'lcp') == [100.0, INF, INF]


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
    run = {'metrics': metrics}
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
    typed = {'control_attached': [{'time': 1300.0, 'readyState': 0, 'target': {}}], 'media': media}
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
    focus = [{'visibility': 'visible', 'hasFocus': True}]
    assert probe.visibility_faults(focus, [], ['normal']) == []
    assert probe.visibility_faults(focus, [{'event': 'visibilitychange'}], ['normal']) == ['visibility_event']
    assert probe.visibility_faults([{'visibility': 'visible', 'hasFocus': False}], [], ['normal']) == ['page_unfocused']
    assert probe.visibility_faults(focus, [], ['minimized']) == ['window_state']
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
    run = {'outcome': Outcome.FAILURE, 'role': 'cold', 'metrics': {'lcp': 10.0, 'cls': 0.0, 'inp': 8.0}}
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
                                 'wait_other': 5000, 'collect': 60000, 'leave': 10000, 'version': 2000}
    assert probe.HARNESS_ONLY_CLASSES == {'setup', 'collect', 'version'}
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
        return {'ok': self.start_ok}

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
    high = BatchFakes(end_load=1.31)
    assert (await probe.run_batch(probe.plan_batch(SHA, 'm', 'desktop', '#hogar', 1, pairs=1, contrast_pairs=0,
                                                   warm_ups=0), high.environment(), idle=0.2, post_batch=0.3))['reasons'] \
        == ['quiet_machine']
    early = BatchFakes()
    early.loadavg = lambda: 1.21
    stopped = await probe.run_batch(plan, early.environment(), idle=0.2, post_batch=0.3)
    assert stopped['discarded'] and 'fixture_start' not in early.events


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


def test_load_thresholds():
    assert probe.load_reading_ok(1.2, 0.2) and not probe.load_reading_ok(1.21, 0.2)
    assert not probe.load_reading_ok(0.1, None)
    assert probe.parse_loadavg('{ 1.52 1.40 1.33 }') == 1.52


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
            'hardware': 'Mac16,8', 'ui_locale': 'es-UY', 'pip_freeze_sha256': 'p'}


def test_before_after_closure_identity_and_extra_rows():
    baseline = {'identity': IDENTITY, 'closure': ['src/memory/fact_ledger.py', 'tests/frontend/browser_support.py',
                                                   'requirements.txt']}
    assert probe.before_after(baseline, dict(IDENTITY), ['docs/readme.md'])['decision'] == 'use_baseline'
    for path in ('src/dashboard/frontend/home/nexo/app.js', 'src/memory/fact_ledger.py', 'requirements.txt',
                 'tests/frontend/browser_support.py', 'tests/mocks/memory_sources.py', 'src/users/voice_ledger.py',
                 'tools/web_timing_probe.py'):
        decision = probe.before_after(baseline, dict(IDENTITY), [path])
        assert decision['decision'] == 'rerun_before', path
    for key in ('browser_version', 'os_version', 'launch_flags', 'hardware', 'ui_locale', 'pip_freeze_sha256'):
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
    assert result['inp']['overhead'] == 16 and result['inp']['screencast'] == 8 and result['inp']['tracing'] == 8
    assert result['raf_feedback']['screencast'] == 8 and result['raf_feedback']['overhead'] == 10
    assert result['consequences'] == {'screencast_limited_to_contrast': False, 'untraced_inp': False}
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
    record(run, 'media', event='playing', timeStamp=600.0, readyState=4)
    await run._interactions(run.config['steps'])
    log = run.page.log
    inputs_sent = [index for index, entry in enumerate(log) if entry['method'].startswith('Input.')]
    assert len(inputs_sent) == 12
    escuchar_last, play_first = inputs_sent[8], inputs_sent[9]
    assert all(not log[i]['method'].startswith(('Runtime.', 'DOM.', 'Accessibility.'))
               for i in range(escuchar_last + 1, play_first))
    assert log[play_first]['params'] == {'type': 'mouseMoved', 'x': 30.0, 'y': 35.0}
    assert [step['status'] for step in run.steps] == ['dispatched'] * 4 and not run.facts.product_failures


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
    again = probe.SmokeRecord(record_path)
    again.start('cold_focus', SHA, cause='crash')
    saved = json.loads(record_path.read_text())
    assert saved['mode'] == 'smoke' and [run['cause'] for run in saved['runs']] == [None, 'crash']
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
           'trace_candidates_present': ['DrawFrame', 'PipelineReporter']}
    phases = probe.frame_event_phases([run, {'metrics': {}, 'trace_candidates_present': []}])
    assert phases == [{'present': {'DrawFrame', 'PipelineReporter'},
                       'interactions': [{'PipelineReporter': True, 'DrawFrame': False}]}]
    assert probe.frame_event_choice(phases)['candidate'] == 'PipelineReporter'
