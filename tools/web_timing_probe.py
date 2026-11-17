#!/usr/bin/env python3
"""T046-U1 web timing probe: NEXO load and interaction timing through CDP.

Implements the method of
``data/autonomous-roadmap-20261102/t046/preregistration.json`` (the
"preregistration"); section names in comments (``runs.order``,
``instrumentation.cdp_deadlines`` ...) point into that file. Local results are
labelled ``Chrome <version>, Mac, con ventana, backend de fixture local, sin
iPhone`` and never accept C10/C11 (``scope.local_label``).

Modes:

* ``baseline``: needs the frozen preregistration sha256 (``--frozen-sha256``);
  refuses any input written in smoke mode (``smoke_gate.run_rules.no_baseline_use``).
* ``smoke`` (``--smoke``): needs the sha256 of the reviewed version
  (``--reviewed-sha256``), derives seeds from ``'<reviewed sha256>:smoke'``
  and writes only under ``t046/smoke/`` (``smoke_gate.smoke_mode``, R5-9).
* ``harness``: server headless Chromium for the probe's own checks only
  (``clients.local.fallback``). Never a measurement: results carry
  ``mode: harness``, metric values are withheld from summaries, and nothing
  is written under ``t046/`` outside ``t046/u1/harness/``.

Only the CDP helpers named in ``instrumentation.cdp_client`` are reused from
``tests/frontend/browser_support.py`` (``browser_binary``,
``browser_profile_root``, ``evidence_directory``, ``install_login_session``);
the probe has its own concurrent CDP reader with absolute per-command deadlines.
"""

import argparse
import asyncio
import base64
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import plistlib
import random
import re
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


logger = logging.getLogger(__name__)

REPOSITORY = Path(__file__).resolve().parents[1]
ROADMAP = REPOSITORY / 'data' / 'autonomous-roadmap-20261102'
PREREGISTRATION = ROADMAP / 't046' / 'preregistration.json'
OUTPUT_ROOT = ROADMAP / 't046'
SMOKE_ROOT = OUTPUT_ROOT / 'smoke'
HARNESS_ROOT = OUTPUT_ROOT / 'u1' / 'harness'
LOCAL_LABEL = 'Chrome {version}, Mac, con ventana, backend de fixture local, sin iPhone'
MODES = ('baseline', 'smoke', 'harness')

# --------------------------------------------------------------------------------------------
# Frozen method constants (preregistration).
# --------------------------------------------------------------------------------------------

# clients.local.launch: exactly these extra flags, plus --remote-debugging-port and --user-data-dir.
LAUNCH_FLAGS = (
    '--no-first-run', '--no-default-browser-check', '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
    '--disable-background-networking', '--disable-component-update',
)
LAUNCH_MECHANISM = 'open -na'
# Harness mode only (server headless Chromium): never a measurement client.
HARNESS_EXTRA_FLAGS = ('--headless', '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader')

VIEWPORTS = {
    'desktop': {'width': 1440, 'height': 900, 'deviceScaleFactor': 2, 'mobile': False},
    'mobile': {'width': 390, 'height': 844, 'deviceScaleFactor': 3, 'mobile': True},
}
VIEW_HASHES = ('#hogar', '#acciones', '#rutinas', '#memoria', '#mis-recuerdos', '#personas', '#voces',
               '#accesos', '#evaluacion')
# runs.views_load: each view loaded by a role that can see it (confirmed against state.js by a unit check).
VIEW_ROLES = {'#hogar': 'home', '#acciones': 'home', '#rutinas': 'home', '#mis-recuerdos': 'home',
              '#memoria': 'operator', '#personas': 'operator', '#voces': 'operator', '#accesos': 'operator',
              '#evaluacion': 'evaluator'}
MOBILE_VIEWS = ('#hogar', '#acciones', '#voces')
INTERACTION_VIEWS = ('#hogar', '#acciones', '#voces')
# R6-7 (fixture_backend.not_represented): views whose services the fixture answers with 503; their C10 rows are
# descriptive only (excluded from C10 acceptance and from before/after claims).
FIXTURE_UNAVAILABLE_VIEWS = ('#rutinas', '#memoria', '#mis-recuerdos')
FIXTURE_UNAVAILABLE_NOTE = 'no representado en la fixture: el servicio responde 503'
# R6-2 (runs.count, smoke_gate.smoke_batches): the batch composition is fixed per mode, never a CLI value.
BATCH_COMPOSITION = {
    'baseline': {'warm_ups': 3, 'pairs': 40, 'contrast_pairs': 10},
    'smoke': {'warm_ups': 1, 'pairs': 5, 'contrast_pairs': 2},
}
# R6-3 (metrics.C11_action_feedback): bounded post-feedback wait inside the timed window, before the flush: the
# later of 100 ms and 6 animation frames after the feedback mutation.
FEEDBACK_WINDOW = {'ms': 100, 'frames': 6}
LOAD_ONLY_INP = 'no medido: vista solo de carga'

# instrumentation.cdp_deadlines.values_ms, by deadline class.
DEADLINE_MS = {
    'setup': 10000,
    'interaction_trace': 10000,
    'navigate': 15000,
    'load_wait': 15000,
    'evaluate': 10000,
    'input': 5000,
    'wait_feedback': 5000,
    'wait_preview': 10000,
    'wait_other': 5000,
    'collect': 60000,
    'leave': 10000,
    'version': 2000,
    # V4: window-state reads at the timed points (browser endpoint, not awaited by the timed sequence).
    'window_state': 10000,
}
# instrumentation.cdp_deadlines.immutable (R5-5): product-failure thresholds the smoke never changes.
IMMUTABLE_CLASSES = frozenset({'interaction_trace', 'navigate', 'load_wait', 'evaluate', 'input', 'wait_feedback',
                               'wait_preview', 'wait_other', 'leave'})
# instrumentation.cdp_deadlines.harness_only_formula.
HARNESS_ONLY_CLASSES = frozenset({'setup', 'collect', 'version', 'window_state'})
QUIET_PERIOD_MS = 500
LCP_CAP_MS = 10000
EVENT_DURATION_THRESHOLD_MS = 16
PREVIEW_GAP_MAX_MS = 16
INPUT_SPACING_S = 1.0
ACTIVATION_POLL_S = 0.1
TRACE_CATEGORIES = ('devtools.timeline', 'loading', 'blink.user_timing', 'disabled-by-default-devtools.timeline.frame')

# Methods each deadline class may send (R5-13: a method with no class is refused; no DOM.* or Accessibility.*).
CLASS_METHODS = {
    'setup': ('Target.', 'Network.', 'Emulation.', 'Page.enable', 'Runtime.enable',
              'Page.removeScriptToEvaluateOnNewDocument', 'Page.addScriptToEvaluateOnNewDocument',
              'Runtime.addBinding', 'Page.bringToFront', 'Runtime.evaluate', 'Browser.getWindowForTarget',
              'Tracing.start', 'Browser.close'),
    'interaction_trace': ('Tracing.start', 'Page.startScreencast'),
    'navigate': ('Page.navigate',),
    'evaluate': ('Runtime.evaluate',),
    'input': ('Input.dispatchMouseEvent', 'Input.dispatchTouchEvent', 'Input.dispatchKeyEvent'),
    'collect': ('Tracing.end', 'Page.stopScreencast', 'Page.screencastFrameAck'),
    'leave': ('Page.navigate',),
    'version': ('Browser.getVersion',),
    'window_state': ('Browser.getWindowForTarget',),
}
FORBIDDEN_PREFIXES = ('DOM.', 'Accessibility.')
PHASES = ('setup', 'timed', 'collect', 'leave')

CENSORED = '<16 ms'
INF = math.inf


class Outcome(str, Enum):
    """Run outcome under exclusion_rules."""

    VALID = 'valid'
    FAILURE = 'failure'
    POST_WINDOW_HANG = 'post_window_hang'
    INVALID = 'invalid'


# --------------------------------------------------------------------------------------------
# Core interaction script (runs.views_interaction.core_script), frozen; its id is part of method_identity.
# --------------------------------------------------------------------------------------------

BUTTON = 'button,[role="button"]'
CORE_SCRIPT = {
    'version': 1,
    'coordinates': 'centre of the reported rectangle; native <audio controls>: play_control rule',
    '#hogar': [
        {'step': 'hogar_plano', 'container': '#map-stage', 'selector': '#view-top',
         'end': {'kind': 'attribute', 'selector': '#view-top', 'attribute': 'aria-pressed', 'value': 'true'},
         'wait': 'wait_other'},
        {'step': 'hogar_ambiente', 'container': '#room-list', 'selector': '[data-room-id="cocina"]',
         'end': {'kind': 'exists', 'selector': '#room-list [data-room-id="cocina"][aria-pressed="true"]'},
         'wait': 'wait_other'},
        {'step': 'hogar_capa', 'container': '#layer-tabs', 'selector': '[data-layer="climate"]',
         'end': {'kind': 'attribute', 'selector': '#layer-tabs [data-layer="climate"]', 'attribute': 'aria-pressed',
                 'value': 'true'},
         'wait': 'wait_other'},
    ],
    # metrics.C11_action_feedback: the receipt marker of the activated card leaves '' (G5, home.js:205-213,
    # app.js updateReceipts): data-status becomes the pending record's status.
    '#acciones': [
        {'step': 'accion', 'container': '#action-list',
         'selector': '.device-card[data-entity-id="light.sala"] button[data-operation="turn_on"]',
         'end': {'kind': 'nonempty_attribute',
                 'selector': '#action-list [data-receipt][data-entity-id="light.sala"][data-operation="turn_on"]',
                 'attribute': 'data-status'},
         'wait': 'wait_feedback', 'c11': 'action_feedback'},
    ],
    '#voces': [
        {'step': 'revisar_muestras', 'container': '#voices-content', 'selector': BUTTON,
         'name_prefix': 'Revisar muestras',
         'end': {'kind': 'attribute_present', 'selector': '#voice-dialog', 'attribute': 'open'},
         'wait': 'wait_other'},
        {'step': 'muestra', 'container': '#voice-dialog',
         'selector': '[data-capture-id="{capture}"] input[type="checkbox"]',
         'end': {'kind': 'checked', 'selector': '#voice-dialog [data-capture-id="{capture}"] input[type="checkbox"]'},
         'wait': 'wait_other'},
        {'step': 'escuchar', 'container': '#voice-dialog',
         'selector': '[data-capture-id="{capture}"] button,[data-capture-id="{capture}"] [role="button"]',
         'name_prefix': 'Escuchar', 'end': {'kind': 'control_attached'}, 'wait': 'wait_preview', 'c11': 'preview'},
        {'step': 'reproducir', 'play_gesture': True, 'end': {'kind': 'playing'}, 'wait': 'wait_preview'},
    ],
    # C11 preview: the review panel holding the attached <audio> (voices.js:209).
    'audio': {'selector': '#voice-dialog .nexo-voice-review audio[src]', 'panel': '.nexo-voice-review'},
    # R4-12, R5-13: play-gesture target rule.
    'play_target': {'selector': BUTTON, 'name_prefix': 'Reproducir'},
}
# runs.views_interaction.extra_rows: controls of later units; no selector exists at this commit, so they are
# recorded as absent (never a failure) until the unit that adds the control defines it under review.
EXTRA_ROWS = {
    '#hogar': [{'label': 'extra: rotar (T042-U5)', 'selector': None}],
    '#voces': [{'label': 'extra: filtro de habitación y búsqueda (T044-U2)', 'selector': None}],
}
# smoke_gate.preview_ready_state fill-in: native play control as a function of the <audio> rectangle.
# Provisional until the smoke records it (run_rules.fill_in_values).
PLAY_CONTROL_RULE = {'dx_px': 20.0, 'y_fraction': 0.5, 'status': 'provisional_until_smoke'}


def script_id(script: dict | None = None) -> str:
    """sha256 of the frozen core script source (method_identity)."""
    source = json.dumps(CORE_SCRIPT if script is None else script, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((source + INJECTED_SOURCE).encode()).hexdigest()


def batch_plan_views() -> list:
    """Every (viewport, view) cell of a client and its URL hash; never an /obs/ URL (Q29, QF)."""
    cells = [('desktop', view) for view in VIEW_HASHES] + [('mobile', view) for view in MOBILE_VIEWS]
    return [{'viewport': viewport, 'view': view, 'url_path': '/' + view, 'role': VIEW_ROLES[view],
             'c11': [step['c11'] for step in CORE_SCRIPT.get(view, []) if 'c11' in step]} for viewport, view in cells]


# --------------------------------------------------------------------------------------------
# Injected script (instrumentation.*). One source per run: CONFIG carries the run token (R5-1).
# --------------------------------------------------------------------------------------------

INJECTED_SOURCE = r"""(() => {
  const CONFIG = __CONFIG__;
  if (location.origin !== CONFIG.origin) return;
  const deliver = window[CONFIG.binding];
  if (typeof deliver !== 'function') return;
  if (window[CONFIG.flush]) return;
  const timeOrigin = performance.timeOrigin;
  let seq = 0;
  const emit = (type, data) => { try { deliver(JSON.stringify(Object.assign({token: CONFIG.token, timeOrigin, seq: ++seq, type, t: performance.now()}, data || {}))); } catch (_) {} };
  const now = () => performance.now();
  const describe = node => {
    if (!node || node.nodeType !== 1) return node === window ? 'window' : node === document ? 'document' : null;
    let text = node.tagName.toLowerCase();
    if (node.id) text += '#' + node.id;
    for (const name of ['data-view', 'data-layer', 'data-room-id', 'data-entity-id', 'data-operation', 'data-capture-id', 'type']) {
      if (node.hasAttribute(name)) text += '[' + name + '="' + node.getAttribute(name).slice(0, 80) + '"]';
    }
    return text;
  };
  const focusState = point => ({point, visibility: document.visibilityState, hasFocus: document.hasFocus(), time: now()});
  const viewport = point => ({point, innerWidth: window.innerWidth, innerHeight: window.innerHeight, dpr: window.devicePixelRatio, time: now()});
  emit('document_start', {href: location.href, language: navigator.language, languages: Array.from(navigator.languages || []),
    screen: {width: screen.width, height: screen.height}, dpr: window.devicePixelRatio,
    supported: Array.from(PerformanceObserver.supportedEntryTypes || []), focus: focusState('document_start'), harness: !!CONFIG.harness});

  // ---- load settle (instrumentation.load_settle); pure core, also run by the unit checks under node ----
  // SETTLE-BEGIN
  function createSettle(options) {
    const quiet = options.quietMs, cap = options.lcpCapMs;
    const state = {gate: false, gateAt: null, lastReset: 0, lastResetKind: 'navigation_start', pending: new Map(), pendingClearedAt: 0, settled: false, timer: null};
    function blocked() { return !state.gate || state.pending.size > 0; }
    function arm() {
      if (state.settled) return;
      if (state.timer !== null) { options.clearTimer(state.timer); state.timer = null; }
      if (blocked()) return;
      const delay = Math.max(0, state.lastReset + quiet - options.now());
      state.timer = options.setTimer(fire, delay);
    }
    function fire() {
      state.timer = null;
      if (state.settled) return;
      const T = options.now();
      options.takeRecords();
      if (blocked() || state.lastReset > T - quiet) { arm(); return; }
      if (!options.loadGateHolds(T)) { arm(); return; }
      state.settled = true;
      const quietAt = state.lastReset + quiet;
      const cleared = [['load', state.gateAt], ['pending', state.pendingClearedAt], ['quiet:' + state.lastResetKind, quietAt]]
        .sort((a, b) => b[1] - a[1])[0][0];
      options.onSettled({T, last_condition: cleared, last_reset: state.lastReset, last_reset_kind: state.lastResetKind, gate_at: state.gateAt, pending_cleared_at: state.pendingClearedAt});
    }
    return {
      state,
      reset(time, kind) {
        if (state.settled) return;
        if (kind === 'lcp' && time > cap) return;
        if (time > state.lastReset) { state.lastReset = time; state.lastResetKind = kind; }
        arm();
      },
      load(time) { if (state.settled || state.gate) return; state.gate = true; state.gateAt = time; arm(); },
      pendingStart(key, kind) { if (state.settled) return; state.pending.set(key, kind); arm(); },
      pendingEnd(key, time) {
        if (state.settled || !state.pending.has(key)) return;
        state.pending.delete(key);
        if (state.pending.size === 0) state.pendingClearedAt = time;
        arm();
      },
    };
  }
  function instrumentRequests(win, settle, now, emit) {
    const sameOrigin = url => { try { return new URL(url, win.location.href).origin === new URL(win.location.href).origin; } catch (_) { return false; } };
    let pendingSerial = 0;
    const nativeFetch = win.fetch;
    win.fetch = function (input, init) {
      const url = input && typeof input === 'object' && 'url' in input ? input.url : String(input);
      const promise = nativeFetch.apply(this, arguments);
      if (!sameOrigin(url)) return promise;
      const key = 'fetch:' + (++pendingSerial);
      settle.pendingStart(key, 'fetch');
      const done = () => { const time = now(); settle.reset(time, 'fetch_end'); settle.pendingEnd(key, time); };
      promise.then(done, done);
      return promise;
    };
    const proto = win.XMLHttpRequest.prototype, nativeOpen = proto.open, nativeSend = proto.send;
    proto.open = function (method, url) { this.__kzaProbeUrl = url; return nativeOpen.apply(this, arguments); };
    proto.send = function () {
      if (sameOrigin(this.__kzaProbeUrl)) {
        const key = 'xhr:' + (++pendingSerial);
        settle.pendingStart(key, 'xhr');
        this.addEventListener('loadend', () => { const time = now(); settle.reset(time, 'xhr_end'); settle.pendingEnd(key, time); }, {once: true});
      }
      return nativeSend.apply(this, arguments);
    };
    const NativeSocket = win.WebSocket;
    win.WebSocket = class extends NativeSocket {
      constructor(url, protocols) {
        super(url, protocols);
        let target;
        try { target = new URL(url, win.location.href); } catch (_) { target = null; }
        if (target && target.host === new URL(win.location.href).host) {
          const key = 'ws:' + (++pendingSerial);
          settle.pendingStart(key, 'websocket');
          emit('websocket', {event: 'construct', key, time: now()});
          for (const type of ['message', 'close', 'error']) {
            this.addEventListener(type, event => { emit('websocket', {event: type, key, time: event.timeStamp}); settle.reset(event.timeStamp, 'ws_' + type); settle.pendingEnd(key, event.timeStamp); });
          }
        }
      }
    };
  }
  // SETTLE-END
  const observers = {};
  const sameOrigin = url => { try { return new URL(url, location.href).origin === location.origin; } catch (_) { return false; } };
  const settle = createSettle({quietMs: CONFIG.quiet_ms, lcpCapMs: CONFIG.lcp_cap_ms, now,
    setTimer: (fn, delay) => setTimeout(fn, delay), clearTimer: id => clearTimeout(id),
    takeRecords: () => { for (const type of ['largest-contentful-paint', 'layout-shift', 'resource']) { const observer = observers[type]; if (observer) handle(observer.takeRecords()); } },
    loadGateHolds: T => { const entry = performance.getEntriesByType('navigation')[0]; return !!entry && entry.loadEventEnd > 0 && entry.loadEventEnd <= T; },
    onSettled: data => { emit('settled', data); emit('viewport', viewport('settled')); emit('focus', focusState('settled')); }});
  window.addEventListener('load', () => setTimeout(() => settle.load(now()), 0));

  // ---- performance entries ----
  const entryData = entry => {
    const data = {entryType: entry.entryType, name: entry.name, startTime: entry.startTime, duration: entry.duration};
    if (entry.entryType === 'largest-contentful-paint') Object.assign(data, {renderTime: entry.renderTime, loadTime: entry.loadTime, size: entry.size, url: entry.url, element: describe(entry.element)});
    else if (entry.entryType === 'layout-shift') Object.assign(data, {value: entry.value, hadRecentInput: entry.hadRecentInput, lastInputTime: entry.lastInputTime});
    else if (entry.entryType === 'event' || entry.entryType === 'first-input') Object.assign(data, {interactionId: entry.interactionId || 0, processingStart: entry.processingStart, processingEnd: entry.processingEnd, target: entry.target ? describe(entry.target) : null});
    else if (entry.entryType === 'navigation') Object.assign(data, {loadEventEnd: entry.loadEventEnd, domContentLoadedEventEnd: entry.domContentLoadedEventEnd, responseEnd: entry.responseEnd});
    else if (entry.entryType === 'resource') Object.assign(data, {initiatorType: entry.initiatorType, requestStart: entry.requestStart, responseStart: entry.responseStart, responseEnd: entry.responseEnd, transferSize: entry.transferSize, encodedBodySize: entry.encodedBodySize, responseStatus: entry.responseStatus});
    return data;
  };
  function handle(entries) {
    for (const entry of entries) {
      emit('perf', {entry: entryData(entry)});
      if (entry.entryType === 'largest-contentful-paint') settle.reset(entry.startTime, 'lcp');
      else if (entry.entryType === 'layout-shift') settle.reset(entry.startTime, 'layout_shift');
      else if (entry.entryType === 'resource' && !['fetch', 'xmlhttprequest'].includes(entry.initiatorType) && sameOrigin(entry.name)) settle.reset(entry.responseEnd, 'resource');
    }
  }
  for (const type of ['largest-contentful-paint', 'layout-shift', 'event', 'first-input', 'paint', 'navigation', 'resource']) {
    try {
      const observer = new PerformanceObserver(list => handle(list.getEntries()));
      observer.observe(type === 'event' ? {type, buffered: true, durationThreshold: CONFIG.duration_threshold} : {type, buffered: true});
      observers[type] = observer;
    } catch (error) { emit('observer_error', {entryType: type, error: String(error)}); }
  }

  // ---- same-origin request and socket observation for load_settle (behaviour unchanged) ----
  instrumentRequests(window, settle, now, emit);

  // ---- capture listener (instrumentation.capture_listener) and visibility ----
  const STEPS = CONFIG.steps;
  const steps = STEPS.map(() => ({started: false, ended: false, lastRect: null, settledFrame: false}));
  let interactions = 0, open = false;
  const FIRST = new Set(['pointerdown', 'touchstart', 'keydown']), LAST = new Set(['click', 'keyup']);
  for (const type of ['pointerdown', 'pointerup', 'click', 'keydown', 'keyup', 'touchstart', 'touchend']) {
    window.addEventListener(type, event => {
      if (FIRST.has(type) && !open) {
        open = true; interactions += 1;
        const index = interactions - 1;
        if (index < steps.length) steps[index].started = true;
        try { performance.mark('kza-probe:start:' + CONFIG.token + ':' + index); } catch (_) {}
        emit('mark', {name: 'kza-probe:start:' + CONFIG.token + ':' + index, time: now()});
        emit('focus', focusState('interaction_start:' + index));
      }
      emit('input', {event: type, timeStamp: event.timeStamp, isTrusted: event.isTrusted, interaction: interactions - 1,
        path: (event.composedPath ? event.composedPath() : []).slice(0, 10).map(describe)});
      if (LAST.has(type)) open = false;
    }, {capture: true, passive: true});
  }
  for (const [target, type] of [[document, 'visibilitychange'], [window, 'pagehide'], [document, 'freeze'], [window, 'blur'], [window, 'focus']]) {
    target.addEventListener(type, event => {
      // blur/focus of an element (for example a control disabled by its own handler) is not a window state.
      if ((type === 'blur' || type === 'focus') && event.target !== window) return;
      emit('visibility_event', {event: type, timeStamp: event.timeStamp, state: document.visibilityState, hasFocus: document.hasFocus()});
    }, {capture: true});
  }
  document.addEventListener('DOMContentLoaded', () => { emit('viewport', viewport('domcontentloaded')); emit('focus', focusState('domcontentloaded')); });

  // ---- core targets (instrumentation.core_targets) and the mutation probe ----
  const nameOf = el => el.hasAttribute('aria-label') ? el.getAttribute('aria-label').trim() : (el.textContent || '').trim();
  const usable = el => el.isConnected && !el.disabled && !el.closest('[inert]') && el.getClientRects().length > 0 && (!el.checkVisibility || el.checkVisibility());
  const rectOf = el => { const r = el.getBoundingClientRect(); return {x: r.left, y: r.top, width: r.width, height: r.height}; };
  function lookup(rule, root) {
    if (!root) return null;
    let found = Array.from(root.querySelectorAll(rule.selector));
    if (rule.name_prefix) { const prefix = rule.name_prefix.toLowerCase(); found = found.filter(el => nameOf(el).toLowerCase().startsWith(prefix)); }
    return found.find(usable) || null;
  }
  // The dispatch point (centre) lies inside the layout viewport (instrumentation.core_targets, V3).
  function inViewport(r) { const x = r.x + r.width / 2, y = r.y + r.height / 2; return x >= 0 && y >= 0 && x < window.innerWidth && y < window.innerHeight; }
  function report(index, el, force) {
    const step = STEPS[index];
    const rect = rectOf(el);
    const last = steps[index].lastRect;
    if (force || !last || last.x !== rect.x || last.y !== rect.y || last.width !== rect.width || last.height !== rect.height) {
      steps[index].lastRect = rect;
      emit('target_ready', {step: step.step, index, rect, in_viewport: inViewport(rect), time: now(), name: nameOf(el).slice(0, 80), describe: describe(el)});
    }
  }
  // V3 (preregistered): the script brings step k+1's target into view only after step k's step_next_frame record
  // (the frame after the frame that painted step k's end) and outside every capture window; never step 0, never
  // during the load phase, never the play gesture.
  function mayScroll(index) {
    return index > 0 && index < STEPS.length && index === interactions && !open && !STEPS[index].play_gesture
      && !steps[index].started
      && steps[index - 1].settledFrame;
  }
  function bringIntoView(index) {
    if (!mayScroll(index)) return;
    const step = STEPS[index];
    const el = lookup(step, document.querySelector(step.container));
    if (!el) return;
    const before = rectOf(el);
    if (inViewport(before)) return;
    el.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'instant'});
    emit('scrolled', {step: step.step, index, before, after: rectOf(el), time: now()});
    report(index, el, true);
  }
  function reportTargets() {
    STEPS.forEach((step, index) => {
      if (step.play_gesture || steps[index].started) return;
      const el = lookup(step, document.querySelector(step.container));
      if (!el) return;
      if (mayScroll(index) && !inViewport(rectOf(el))) { bringIntoView(index); return; }
      report(index, el, false);
    });
  }
  function endHolds(end) {
    if (end.kind === 'attribute') { const el = document.querySelector(end.selector); return !!el && el.getAttribute(end.attribute) === end.value; }
    if (end.kind === 'attribute_present') { const el = document.querySelector(end.selector); return !!el && el.hasAttribute(end.attribute); }
    if (end.kind === 'nonempty_attribute') { const el = document.querySelector(end.selector); return !!el && !!el.getAttribute(end.attribute); }
    if (end.kind === 'exists') return !!document.querySelector(end.selector);
    if (end.kind === 'checked') { const el = document.querySelector(end.selector); return !!el && el.checked === true; }
    return false;
  }
  function stepEnded(index, time) {
    const step = STEPS[index];
    steps[index].ended = true;
    const name = 'kza-probe:end:' + CONFIG.token + ':' + index;
    try { performance.mark(name); } catch (_) {}
    emit('mark', {name, time});
    emit('step_end', {step: step.step, index, time});
    emit('focus', focusState('interaction_end:' + index));
    // V5: the first-rAF lower bound is performance.now() inside the callback (never before the mutation); the
    // callback's frame timestamp is kept as frame_time.
    requestAnimationFrame(frame => {
      emit('step_frame', {step: step.step, index, raf: now(), frame_time: frame});
      requestAnimationFrame(next => {
        emit('step_next_frame', {step: step.step, index, raf: now(), frame_time: next});
        steps[index].settledFrame = true;
        bringIntoView(index + 1);
      });
    });
    // R6-3: C11 action feedback keeps the timed window open until the later of feedback_window.ms and
    // feedback_window.frames animation frames after the mutation, so the trace holds the feedback frame.
    if (step.c11 === 'action_feedback') {
      let frames = 0;
      const tick = () => {
        frames += 1;
        if (frames >= CONFIG.feedback_window.frames && now() - time >= CONFIG.feedback_window.ms) emit('feedback_window', {step: step.step, index, frames, mutation: time, time: now()});
        else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }
  }
  let audioTracked = null;
  const hitAt = point => { try { return describe(document.elementFromPoint(point.x, point.y)); } catch (_) { return null; } };
  function playTarget(audio) {
    const panel = audio.closest(CONFIG.audio.panel) || audio.parentElement;
    const button = lookup(CONFIG.play_target, panel);
    if (button) { const rect = rectOf(button); const point = {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2}; return {kind: 'button', rect, name: nameOf(button).slice(0, 80), point, hit: hitAt(point)}; }
    const rect = rectOf(audio);
    const point = {x: rect.x + CONFIG.play_rule.dx_px, y: rect.y + rect.height * CONFIG.play_rule.y_fraction};
    return {kind: 'audio', rect, point, hit: hitAt(point)};
  }
  function trackAudio() {
    const audio = document.querySelector(CONFIG.audio.selector);
    if (!audio || audio === audioTracked || !audio.isConnected) return;
    const time = now();
    const first = audioTracked === null;
    audioTracked = audio;
    for (const type of ['loadedmetadata', 'canplay', 'playing', 'error', 'abort', 'emptied']) {
      audio.addEventListener(type, event => emit('media', {event: type, timeStamp: event.timeStamp, readyState: audio.readyState}));
    }
    const target = playTarget(audio);
    if (first) {
      const name = 'kza-probe:end:' + CONFIG.token + ':control_attached';
      try { performance.mark(name); } catch (_) {}
      emit('mark', {name, time});
      target.in_viewport = inViewport(target.rect);
      emit('control_attached', {time, readyState: audio.readyState, target});
      // V2 (preregistered): a native <audio controls> target is operable once metadata is loaded; control_ready
      // carries the time (performance.now() in the listener) and the play target rectangle at that moment.
      audio.addEventListener('loadedmetadata', () => {
        const ready = playTarget(audio);
        ready.in_viewport = inViewport(ready.rect);
        emit('control_ready', {time: now(), readyState: audio.readyState, target: ready});
      }, {once: true});
      const index = STEPS.findIndex(step => step.end.kind === 'control_attached');
      if (index >= 0) { steps[index].ended = true; requestAnimationFrame(frame => emit('step_frame', {step: STEPS[index].step, index, raf: now(), frame_time: frame})); }
    } else emit('audio_replaced', {time, readyState: audio.readyState, target});
  }
  function onMutations() {
    const time = now();
    STEPS.forEach((step, index) => { if (steps[index].started && !steps[index].ended && step.end && endHolds(step.end)) stepEnded(index, time); });
    trackAudio();
    reportTargets();
  }
  const mutations = new MutationObserver(onMutations);
  const startObserving = () => { mutations.observe(document, {subtree: true, childList: true, attributes: true, characterData: true}); onMutations(); };
  startObserving();
  window.addEventListener('playing', event => {
    const index = STEPS.findIndex(step => step.end.kind === 'playing');
    if (index >= 0 && steps[index].started && !steps[index].ended) {
      steps[index].ended = true;
      emit('step_end', {step: STEPS[index].step, index, time: event.timeStamp});
      requestAnimationFrame(frame => emit('step_frame', {step: STEPS[index].step, index, raf: now(), frame_time: frame}));
    }
  }, {capture: true});

  // ---- final flush (instrumentation.run_lifecycle phase 2 end) ----
  window[CONFIG.flush] = () => new Promise(resolve => {
    for (const observer of Object.values(observers)) handle(observer.takeRecords());
    emit('viewport', viewport('flush'));
    emit('focus', focusState('flush'));
    requestAnimationFrame(() => setTimeout(() => {
      emit('flushed', {count: seq + 1});
      resolve(seq);
      if (CONFIG.harness) setTimeout(() => window.dispatchEvent(new Event('kzaprobeflush')), 0);
    }, 0));
  });
})();"""


def settle_source() -> str:
    """The pure load-settle core of the injected script (for the node unit checks)."""
    return INJECTED_SOURCE.split('// SETTLE-BEGIN', 1)[1].split('// SETTLE-END', 1)[0]


def injection_config(*, token: str, origin: str, view: str, capture_id: str | None, harness_steps: list | None = None,
                     harness: bool = False, play_rule: dict | None = None) -> dict:
    """Per-run configuration embedded in the injected source (R5-1, R5-4, R5-8)."""
    steps = harness_steps if harness_steps is not None else CORE_SCRIPT.get(view, [])
    resolved = json.loads(json.dumps(steps).replace('{capture}', capture_id or ''))
    return {'token': token, 'origin': origin, 'binding': 'kzaProbeRecord', 'flush': '__kzaProbeFlush_' + token,
            'quiet_ms': QUIET_PERIOD_MS, 'lcp_cap_ms': LCP_CAP_MS, 'duration_threshold': EVENT_DURATION_THRESHOLD_MS,
            'steps': resolved, 'audio': CORE_SCRIPT['audio'], 'play_target': CORE_SCRIPT['play_target'],
            'play_rule': play_rule or PLAY_CONTROL_RULE, 'harness': harness, 'feedback_window': FEEDBACK_WINDOW}


def injected_source(config: dict) -> str:
    return INJECTED_SOURCE.replace('__CONFIG__', json.dumps(config, sort_keys=True))


# --------------------------------------------------------------------------------------------
# Record binding filter (instrumentation.record_binding, R5-1).
# --------------------------------------------------------------------------------------------

class RecordFilter:
    """Accept only the current run's token and the run document's timeOrigin."""

    def __init__(self, token: str):
        self.token = token
        self.loader_id = None
        self.committed = False
        self.time_origin = None
        self.accepted = []
        self.rejected = []
        # V14b: current-token records that arrive before the Page.navigate reply names the loaderId are buffered,
        # and main-frame commits seen before it are remembered; both are resolved when the reply arrives.
        self.buffered = []
        self.main_frame_loaders = []
        self.navigation_replied = False

    def navigation_sent(self, loader_id: str) -> None:
        self.loader_id = loader_id
        self.navigation_replied = True
        if loader_id is not None and loader_id in self.main_frame_loaders:
            self.committed = True
        pending, self.buffered = self.buffered, []
        for record in pending:
            self.offer(record)

    def frame_navigated(self, frame: dict) -> None:
        if frame.get('parentId') is not None:
            return
        self.main_frame_loaders.append(frame.get('loaderId'))
        if self.loader_id is not None and frame.get('loaderId') == self.loader_id:
            self.committed = True

    def offer(self, record: dict) -> bool:
        """Return True and keep the record when it belongs to the run document."""
        reason = None
        if record.get('token') != self.token:
            reason = 'token'
        elif not self.committed and not self.navigation_replied:
            self.buffered.append(record)
            return False
        elif not self.committed:
            reason = 'before_commit'
        elif self.time_origin is None:
            if record.get('type') == 'document_start':
                self.time_origin = record.get('timeOrigin')
            else:
                reason = 'before_document_start'
        if reason is None and record.get('timeOrigin') != self.time_origin:
            reason = 'time_origin'
        if reason is not None:
            self.rejected.append({'reason': reason, 'record': record})
            return False
        self.accepted.append(record)
        return True


# --------------------------------------------------------------------------------------------
# Statistics (statistics.percentiles, statistics.precision).
# --------------------------------------------------------------------------------------------

def _rank_key(value) -> tuple:
    if value == CENSORED:
        return (0, 0.0)
    if value == INF:
        return (2, 0.0)
    return (1, float(value))


def sorted_values(values: list) -> list:
    """Censored '<16 ms' below every measured value, failures (+inf) above."""
    return sorted(values, key=_rank_key)


def nearest_rank(values: list, percentile: float):
    """Nearest-rank percentile over all n valid runs (failures as +inf)."""
    if not values:
        return None
    ordered = sorted_values(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def precision(values: list) -> dict:
    """Distribution-free intervals for n = 40 (p75 25th..36th; p95 lower bound 36th)."""
    ordered = sorted_values(values)
    measured = [value for value in ordered if value not in (CENSORED, INF)]
    result = {'n': len(ordered), 'failures': sum(1 for value in ordered if value == INF),
              'min': ordered[0] if ordered else None, 'max': ordered[-1] if ordered else None,
              'mean': statistics.fmean(measured) if measured and len(measured) == len(ordered) else None}
    if len(ordered) == 40:
        result['p75_interval'] = [ordered[24], ordered[35]]
        result['p95_lower'] = ordered[35]
        result['p95_upper'] = 'no acotado con 40 corridas'
    return result


def summarize(values: list) -> dict:
    return {'p50': nearest_rank(values, 50), 'p75': nearest_rank(values, 75), 'p95': nearest_rank(values, 95),
            **precision(values)}


def percentile_runs(runs: list) -> list:
    """Runs that enter C10/C11 percentiles: valid or failed, never warm-ups, primers or trace-contrast runs."""
    return [run for run in runs if run['outcome'] in (Outcome.VALID, Outcome.FAILURE, Outcome.POST_WINDOW_HANG)
            and run.get('role') in ('cold', 'warm') and not run.get('trace_contrast') and not run.get('warm_up')
            and not run.get('primer')]


# V1 (exclusion_rules.failure_scope): failures that make every metric of the run +inf. Any other product failure
# makes only the metrics it names +inf (they are already +inf in the run's metrics).
RUN_WIDE_PREFIXES = ('load_not_settled', 'navigation_error', 'navigation_http_error', 'cdp_timeout:',
                     'leave_step_timeout', 'collect_timeout_renderer_busy')


def run_wide_failure(run: dict) -> bool:
    """True when the run's failure counts for every metric (load_not_settled, timed-window CDP timeout,
    post_window_hang, failed navigation); False for a valid run or a metric-scoped product failure."""
    if run['outcome'] == Outcome.POST_WINDOW_HANG:
        return True
    if run['outcome'] != Outcome.FAILURE:
        return False
    reasons = [run.get('reason') or ''] + list((run.get('facts') or {}).get('product_failures') or [])
    return any(reason.startswith(RUN_WIDE_PREFIXES) for reason in reasons)


def metric_values(runs: list, metric: str) -> list:
    """Collect a metric with failures as +inf (C11 forbids excluding them); per-metric scope (V1)."""
    values = []
    for run in percentile_runs(runs):
        if run_wide_failure(run):
            values.append(INF)
        else:
            values.append(run['metrics'].get(metric, INF))
    return values


# --------------------------------------------------------------------------------------------
# Metrics (metrics.*).
# --------------------------------------------------------------------------------------------

def interactions_from_inputs(inputs: list) -> list:
    """Group capture-listener events into interactions (first event .. click/keyup)."""
    grouped = defaultdict(list)
    for record in inputs:
        if record.get('interaction', -1) >= 0:
            grouped[record['interaction']].append(record)
    result = []
    for index in sorted(grouped):
        events = sorted(grouped[index], key=lambda item: item['timeStamp'])
        result.append({'index': index, 'first': events[0]['timeStamp'], 'last': events[-1]['timeStamp'],
                       'types': [event['event'] for event in events],
                       'paths': {node for event in events for node in event.get('path', []) if node}})
    return result


def capture_windows(interactions: list) -> list:
    """[first - 1 ms, last + 1 ms]; overlapping margins split at the midpoint (R3-1, R4-1)."""
    windows = [[item['first'] - 1, item['last'] + 1] for item in interactions]
    for index in range(1, len(windows)):
        if windows[index - 1][1] > windows[index][0]:
            middle = (interactions[index - 1]['last'] + interactions[index]['first']) / 2
            windows[index - 1][1] = middle
            windows[index][0] = middle
    return windows


def event_groups(entries: list) -> list:
    """Event Timing groups by interactionId > 0: startTime = min, latency = max duration (R2-5)."""
    grouped = defaultdict(list)
    for entry in entries:
        if entry.get('entryType') == 'event' and (entry.get('interactionId') or 0) > 0:
            grouped[entry['interactionId']].append(entry)
    return [{'interactionId': key, 'startTime': min(item['startTime'] for item in values),
             'latency': max(item['duration'] for item in values), 'types': sorted({item['name'] for item in values}),
             'targets': [item.get('target') for item in values]} for key, values in sorted(grouped.items())]


def map_inp(entries: list, interactions: list) -> dict:
    """Map groups to scripted interactions by capture window time only (R3-1, R4-1)."""
    windows = capture_windows(interactions)
    groups = event_groups(entries)
    latencies = [None] * len(interactions)
    counts = [0] * len(interactions)
    diagnostics = [[] for _ in interactions]
    unmapped = []
    for group in groups:
        match = None
        for index, (low, high) in enumerate(windows):
            if low <= group['startTime'] <= high:
                match = index
                break
        if match is None:
            unmapped.append({'types': group['types'], 'startTime': group['startTime'],
                             'target': [target_diagnostic(target, set()) for target in group['targets']]})
            continue
        counts[match] += 1
        latencies[match] = group['latency'] if latencies[match] is None else max(latencies[match], group['latency'])
        diagnostics[match].append({'types': group['types'], 'offset': group['startTime'] - windows[match][0],
                                   'target': sorted({target_diagnostic(target, interactions[match]['paths'])
                                                     for target in group['targets']})})
    values = [CENSORED if value is None else value for value in latencies]
    distances = [interactions[index]['first'] - interactions[index - 1]['last'] for index in range(1, len(interactions))]
    return {'latencies': values, 'groups_per_interaction': counts, 'diagnostics': diagnostics, 'unmapped': unmapped,
            'window_distances': distances}


def target_diagnostic(target, path: set) -> str:
    if target is None:
        return 'null'
    return 'on_path' if target in path else 'off_path'


def run_inp(mapping: dict, completed: bool):
    """Per-run INP: max over core interactions; +inf when any step failed or was skipped (R5-4)."""
    if not completed:
        return INF
    values = mapping['latencies']
    if not values:
        return INF
    measured = [value for value in values if value != CENSORED]
    return max(measured) if measured else CENSORED


def lcp(entries: list, first_input: float | None):
    """Last LCP entry before the first input or the 10 s cap (renderTime, else loadTime)."""
    value = None
    for entry in sorted((item for item in entries if item.get('entryType') == 'largest-contentful-paint'),
                        key=lambda item: item['startTime']):
        if entry['startTime'] > LCP_CAP_MS or (first_input is not None and entry['startTime'] >= first_input):
            break
        value = entry.get('renderTime') or entry.get('loadTime')
    return INF if value is None else round(value, 1)


def cls(entries: list, completed: bool) -> float:
    """Maximum session window (gap 1 s, window 5 s) without hadRecentInput; +inf when incomplete (R5-4)."""
    if not completed:
        return INF
    shifts = sorted((item for item in entries if item.get('entryType') == 'layout-shift'
                     and not item.get('hadRecentInput')), key=lambda item: item['startTime'])
    best = current = 0.0
    start = previous = None
    for shift in shifts:
        if start is None or shift['startTime'] - previous > 1000 or shift['startTime'] - start > 5000:
            start, current = shift['startTime'], 0.0
        current += shift['value']
        previous = shift['startTime']
        best = max(best, current)
    return best


def action_feedback(start: float | None, end: float | None):
    """C11 action feedback: capture-listener start to the presented frame (never Event Timing startTime)."""
    if start is None or end is None:
        return INF
    return end - start


def preview(start, control_ready, gesture_ts, playing) -> dict:
    """C11 preview: primary = (control_ready - start) + (playing - gesture_ts); gap > 16 ms invalid (R2-6).

    control_ready is control_attached, or for a native <audio controls> target attached before its metadata the
    control_ready record (loadedmetadata, V2), so the metadata wait is product time and only the gap is harness
    wait."""
    if None in (start, control_ready, gesture_ts, playing):
        return {'value': INF, 'failed_open': True, 'gap': None, 'unsubtracted': None, 'gap_invalid': False}
    gap = gesture_ts - control_ready
    return {'value': (control_ready - start) + (playing - gesture_ts), 'failed_open': False, 'gap': gap,
            'unsubtracted': playing - start, 'gap_invalid': gap > PREVIEW_GAP_MAX_MS}


def native_play_interactions(typed: dict, interactions: list, steps: list, flush_time: float | None) -> list:
    """F6: for a native <audio controls> play target the page sees none of the gesture's input events, so the
    play gesture's window is [control_ready (or attach) time, flush sample time]: no other scripted input exists
    there. A light-DOM button target keeps its capture-listener window."""
    play_index = next((index for index, step in enumerate(steps) if step.get('play_gesture')), None)
    attached = typed['control_attached'][0] if typed.get('control_attached') else None
    ready = play_ready(attached, typed['control_ready'][0] if typed.get('control_ready') else None)
    if play_index is None or ready is None or ready['target'].get('kind') != 'audio' \
            or len(interactions) < play_index:
        return interactions
    synthetic = {'index': play_index, 'first': ready['time'] + 1,
                 'last': (flush_time if flush_time is not None else INF) - 1, 'types': ['native_control'],
                 'paths': set(), 'synthetic': True}
    return interactions[:play_index] + [synthetic]


def play_ready(attached: dict | None, ready: dict | None) -> dict | None:
    """V2: the record whose time and target the play gesture uses: control_attached for a light-DOM button or a
    native control that already has metadata (readyState >= 1), else the control_ready record (or None)."""
    if attached is None:
        return None
    if attached['target'].get('kind') == 'button' or (attached.get('readyState') or 0) >= 1:
        return attached
    return ready


# --------------------------------------------------------------------------------------------
# Run classification (instrumentation.cdp_deadlines.rule, exclusion_rules).
# --------------------------------------------------------------------------------------------

@dataclass
class Timeout:
    """One CDP command or wait that reached its absolute deadline."""

    phase: str
    klass: str
    method: str
    sent: float = 0.0
    deadline: float = 0.0


@dataclass
class RunFacts:
    """Facts gathered by a run, independent of the timed-out command."""

    timeouts: list = field(default_factory=list)
    crashed: bool = False
    disconnected: bool = False
    navigate_sent: bool = False
    leave_ok: bool = True
    renderer_busy_across_deadline: bool | None = None
    setup_faults: list = field(default_factory=list)
    environment_faults: list = field(default_factory=list)
    product_failures: list = field(default_factory=list)


def classify(facts: RunFacts) -> tuple:
    """Return (Outcome, reason). First timeout wins; failures are never excluded.

    Order: a crash or disconnect is invalid in any phase; a setup fault (before Page.navigate, so no product
    outcome exists) is invalid; then cdp_deadlines.rule (a) a timed-window timeout is a failure and (b) a
    leave-step timeout is a post_window_hang, neither turned invalid by anything observed later
    (exclusion_rules.invalid_run: "A CDP timeout inside the timed window is a failure, not an invalid run,
    unless one of the crash/disconnect checks shows the crash or disconnect"); then environment faults of the
    timed phase (visibility, emulation, preview gap) are invalid; then rule (c); then product failures.
    """
    if facts.crashed:
        return Outcome.INVALID, 'browser_crashed'
    if facts.disconnected:
        return Outcome.INVALID, 'websocket_disconnected'
    if facts.setup_faults:
        return Outcome.INVALID, facts.setup_faults[0]
    timed = [item for item in facts.timeouts if item.phase == 'timed']
    if timed:
        return Outcome.FAILURE, 'cdp_timeout:' + timed[0].method
    if any(item.phase == 'leave' for item in facts.timeouts):
        return Outcome.POST_WINDOW_HANG, 'leave_step_timeout'
    if facts.environment_faults:
        return Outcome.INVALID, facts.environment_faults[0]
    early = [item for item in facts.timeouts if item.phase in ('setup', 'collect')]
    if early:
        first = early[0]
        if first.phase == 'collect' and facts.leave_ok and facts.renderer_busy_across_deadline is True:
            return Outcome.POST_WINDOW_HANG, 'collect_timeout_renderer_busy'
        return Outcome.INVALID, f'{first.phase}_timeout:{first.method}'
    if facts.product_failures:
        return Outcome.FAILURE, facts.product_failures[0]
    return Outcome.VALID, None


def warm_slot_invalid(cold: dict) -> bool:
    """R5-14b: a warm slot is invalid only when its cold run never navigated or crashed/disconnected."""
    return not cold.get('navigate_sent') or cold.get('reason') in ('browser_crashed', 'websocket_disconnected')


def timed_records(records: list, flush_time: float | None) -> list:
    """V4: only records of the timed phase (page time <= the flush sample) enter the visibility rule."""
    if flush_time is None:
        return list(records)
    # R6-11(c): by the record's own time (the flush sample's emit t is later than its time).
    return [record for record in records
            if record.get('time', record.get('timeStamp', record.get('t', 0.0))) <= flush_time]


def visibility_faults(document_start: dict, focus_records: list, events: list, window_reads: list) -> list:
    """R2-3, F1 (instrumentation.visibility): hidden, unfocused or minimised during the timed phase -> invalid run.

    document_start's hasFocus is a diagnostic only (Chromium dispatches the window focus to a new document after
    its first script, F1); when it is false, a window 'focus' event of the run document must arrive before the
    next sample point. visibilityState is 'visible' at every point, document_start included; hasFocus is true at
    every later sample point; no visibilitychange, pagehide, freeze or window blur event; every window-state read
    (Browser.getWindowForTarget at the timed points) answered 'normal'. Records are those of the timed phase."""
    faults = []
    start_focus = document_start.get('focus') or {}
    samples = sorted(focus_records, key=lambda record: record.get('time', 0.0))
    if start_focus.get('visibility', 'visible') != 'visible' or any(
            record.get('visibility') != 'visible' for record in samples):
        faults.append('page_hidden')
    later_unfocused = any(record.get('hasFocus') is not True for record in samples)
    if start_focus.get('hasFocus') is False:
        next_point = samples[0].get('time') if samples else None
        gained = [event for event in events if event.get('event') == 'focus'
                  and (next_point is None or event.get('timeStamp', INF) <= next_point)]
        if not gained:
            later_unfocused = True
    if later_unfocused:
        faults.append('page_unfocused')
    if any(event.get('event') in ('visibilitychange', 'pagehide', 'freeze', 'blur') for event in events):
        faults.append('visibility_event')
    if any(read.get('state') != 'normal' for read in window_reads):
        faults.append('window_state')
    return faults


def viewport_check(viewport_name: str, document_start: dict, viewports: list) -> tuple:
    """R3-6, R4-10: (faults, flags). Mobile layout-viewport growth is a flagged product outcome."""
    expected = VIEWPORTS[viewport_name]
    faults, flags = [], []
    screen = document_start.get('screen') or {}
    if (screen.get('width'), screen.get('height')) != (expected['width'], expected['height']):
        faults.append('screen_size')
    if document_start.get('dpr') != expected['deviceScaleFactor'] or any(
            record.get('dpr') != expected['deviceScaleFactor'] for record in viewports):
        faults.append('device_pixel_ratio')
    for record in viewports:
        size = (record.get('innerWidth'), record.get('innerHeight'))
        if size != (expected['width'], expected['height']):
            if expected['mobile']:
                flags.append({'flag': 'viewport_ensanchado', 'point': record.get('point'), 'innerWidth': size[0],
                              'innerHeight': size[1]})
            elif 'inner_size' not in faults:
                faults.append('inner_size')
    return faults, flags


# --------------------------------------------------------------------------------------------
# Batches (runs.order, exclusion_rules.invalid_batch, batch_reporting).
# --------------------------------------------------------------------------------------------

def derive_seed(identity_sha: str, *parts, smoke: bool = False) -> int:
    """First 8 bytes of sha256('<sha>[:smoke]:<parts>') (runs.order, smoke_gate.smoke_mode)."""
    prefix = f'{identity_sha}:smoke' if smoke else identity_sha
    text = ':'.join([prefix, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def cell_order(identity_sha: str, client: str, attempt: int, cells: list, *, smoke: bool = False) -> list:
    ordered = list(cells)
    random.Random(derive_seed(identity_sha, 'order', client, attempt, smoke=smoke)).shuffle(ordered)
    return ordered


def plan_batch(identity_sha: str, client: str, viewport: str, view: str, attempt: int, *, smoke: bool = False,
               pairs: int = 40, contrast_pairs: int = 10, warm_ups: int = 3, untraced_inp: bool = False) -> dict:
    """Warm-ups first, then cold/warm and trace-contrast pairs in the batch seed's shuffle."""
    seed = derive_seed(identity_sha, 'batch', client, viewport, view, attempt, smoke=smoke)
    units = [{'kind': 'pair'} for _ in range(pairs)] + [{'kind': 'contrast'} for _ in range(contrast_pairs)]
    if untraced_inp:
        units += [{'kind': 'untraced_inp'} for _ in range(pairs)]
    random.Random(seed).shuffle(units)
    slots = [{'role': 'cold', 'warm_up': True, 'pair': None} for _ in range(warm_ups)]
    for number, unit in enumerate(units):
        contrast, untraced = unit['kind'] == 'contrast', unit['kind'] == 'untraced_inp'
        slots.append({'role': 'cold', 'pair': number, 'trace_contrast': contrast, 'untraced_inp': untraced})
        slots.append({'role': 'warm', 'pair': number, 'trace_contrast': contrast, 'untraced_inp': untraced})
    for ordinal, slot in enumerate(slots):
        slot['ordinal'] = ordinal
    preview_start = derive_seed(identity_sha, 'preview', client, viewport, view, attempt, smoke=True) % 50 if smoke else 0
    return {'seed': seed, 'slots': slots, 'planned': len(slots) - warm_ups, 'preview_start': preview_start,
            'attempt': attempt, 'cell': {'client': client, 'viewport': viewport, 'view': view}}


def replacement_slots(plan: dict, invalid_slot: dict) -> list:
    """Append the replacement of an invalid run at the end, continuing ordinals (runs.order)."""
    next_ordinal = len(plan['slots'])
    base = {key: invalid_slot.get(key) for key in ('pair', 'trace_contrast', 'untraced_inp')}
    if invalid_slot['role'] == 'cold':
        added = [{'role': 'cold', 'replaces': invalid_slot['ordinal'], 'warm_up': invalid_slot.get('warm_up', False),
                  **base}]
    else:
        added = [{'role': 'cold', 'primer': True, 'replaces': invalid_slot['ordinal'], **base},
                 {'role': 'warm', 'replaces': invalid_slot['ordinal'], **base}]
    for offset, slot in enumerate(added):
        slot['ordinal'] = next_ordinal + offset
    plan['slots'].extend(added)
    return added


def preview_position(ordinal: int, permutation: list, start: int = 0) -> int:
    """runs.preview_samples: run i opens permutation[(start + i) mod 50]."""
    return permutation[(start + ordinal) % len(permutation)]


def warm_share(warm_runs: list) -> dict:
    """R2-9: share of same-origin static subresources from memory, disk or 304 over the batch's warm runs."""
    mix = Counter()
    for run in warm_runs:
        mix.update(run.get('cache_sources', {}))
    total = sum(mix.values())
    cached = mix['memory'] + mix['disk'] + mix['304']
    share = cached / total if total else 0.0
    return {'share': share, 'mix_percent': {key: (100 * mix[key] / total if total else 0.0)
                                            for key in ('memory', 'disk', '304', 'network')}, 'requests': total}


def warm_cache_outcome(share: float) -> str | None:
    """smoke_gate.warm_cache preregistered response below 90%."""
    return None if share >= 0.9 else 'no evaluable por este método: caché caliente no alcanzada'


def invalid_batch_reasons(batch: dict) -> list:
    """exclusion_rules.invalid_batch. Failures never count toward the 10% (no selection bias)."""
    reasons = []
    if batch.get('quiet_start_ok') is False or batch.get('quiet_end_ok') is False \
            or batch.get('caffeinate_assertion_start') is False or batch.get('caffeinate_assertion_end') is False:
        reasons.append('quiet_machine')
    warm = [run for run in batch['runs'] if run.get('role') == 'warm' and run['outcome'] != Outcome.INVALID]
    if batch.get('warm_rule_active', True) and warm and warm_share(warm)['share'] < 0.9:
        reasons.append('warm_share_below_90')
    if batch.get('tree_dirty') or batch.get('commit_changed'):
        reasons.append('served_tree')
    header_version = batch.get('browser_version')
    if any(run.get('browser_version') not in (None, header_version) for run in batch['runs']):
        reasons.append('browser_version_changed')
    if batch.get('elapsed_since_fixture_s', 0) > 10 * 3600:
        reasons.append('fixture_lifetime')
    invalid = sum(1 for run in batch['runs'] if run['outcome'] == Outcome.INVALID and not run.get('warm_up'))
    if invalid > 0.1 * batch['planned']:
        reasons.append('invalid_runs_over_10_percent')
    return reasons


class BatchLedger:
    """batch_reporting: two consecutive discarded batches stop the probe with a blocker."""

    def __init__(self):
        self.reports = []
        self.consecutive_discards = 0

    def record(self, report: dict) -> None:
        self.reports.append(report)
        self.consecutive_discards = self.consecutive_discards + 1 if report.get('discarded') else 0

    def may_start(self) -> bool:
        return self.consecutive_discards < 2


# --------------------------------------------------------------------------------------------
# Quiet machine (quiet_machine.mac, R5-7, R5-14d).
# --------------------------------------------------------------------------------------------

# R6-11(a): `python -m pytest` and the bin/pytest (or py.test) console script, read from the ps rows (macOS pgrep
# has no -c).
PYTEST = re.compile(r'python[0-9.]*\b.*-m pytest|(^|/)(pytest|py\.test)(\s|$)')
BUILD_RUNNERS = re.compile(r'(^|/)(xcodebuild|swift-build|clang|cargo|make)(\s|$)')
NODE_BUILD = re.compile(r'(^|/)(node|npm)\b.*\b(build|test)\b')


def ancestry(pid: int, parents: dict) -> set:
    chain = set()
    while pid and pid not in chain:
        chain.add(pid)
        pid = parents.get(pid, 0)
    return chain


def process_check(rows: list, own_pid: int, *, cpu: dict | None = None, allowed: set | None = None) -> list:
    """Return violations; rows are dicts with pid, ppid, command. The probe's ancestry is exempt."""
    parents = {row['pid']: row['ppid'] for row in rows}
    exempt = ancestry(own_pid, parents) | (allowed or set())
    children = defaultdict(list)
    for row in rows:
        children[row['ppid']].append(row)
    violations = []
    for row in rows:
        if row['pid'] in exempt:
            continue
        command = row['command']
        if PYTEST.search(command):
            violations.append(('pytest', row['pid']))
        elif 'tests/frontend/browser_' in command or 'web_timing_probe' in command:
            violations.append(('browser_harness', row['pid']))
        elif 'Google Chrome' in command and '--type=' not in command:
            violations.append(('chrome', row['pid']))
        elif BUILD_RUNNERS.search(command) or NODE_BUILD.search(command):
            violations.append(('build_or_test', row['pid']))
        elif 'uvicorn' in command:
            violations.append(('uvicorn', row['pid']))
        elif re.search(r'(^|/)(claude|codex)(\s|$)', command) and any(
                re.search(r'(^|/)(sh|bash|zsh|python[0-9.]*|node|make|cargo|xcodebuild)(\s|$)', child['command'])
                for child in children[row['pid']]):
            violations.append(('agent_worker', row['pid']))
        if cpu and cpu.get(row['pid'], 0.0) > 10.0 and row['pid'] not in exempt:
            violations.append(('cpu_over_10_percent', row['pid']))
    return violations


def load_reading_ok(reading: float, reference: float | None) -> bool:
    """Batch start <= idle + 1.0; batch end <= post_batch + 1.0 (R5-7)."""
    return reference is not None and reading <= reference + 1.0


def parse_loadavg(text: str) -> float:
    return float(text.strip().strip('{}').split()[0])


# --------------------------------------------------------------------------------------------
# Before/after, overhead, frame event, deadlines and smoke responses.
# --------------------------------------------------------------------------------------------

TRIGGER_PATTERNS = (re.compile(r'^src/dashboard/'), re.compile(r'^src/users/voice'),
                    re.compile(r'^tests/mocks/memory_sources\.py$'), re.compile(r'^tools/web_timing_probe\.py$'))
IDENTITY_COMPONENTS = ('preregistration_sha256', 'fixture_digest', 'script_id', 'browser_version', 'launch_flags',
                       'launch_mechanism', 'os_version', 'hardware', 'ui_locale', 'pip_freeze_sha256',
                       'audio_output')


def before_after(baseline: dict, current_identity: dict, changed_paths: list, *, script_runs: bool = True) -> dict:
    """statistics.before_after: reuse the baseline only when nothing in its closure or identity changed."""
    if not script_runs:
        return {'decision': 'new_script_id', 'compare_baseline': False}
    closure = set(baseline.get('closure', []))
    reasons = [f'path:{path}' for path in changed_paths
               if path in closure or any(pattern.search(path) for pattern in TRIGGER_PATTERNS)]
    reasons += [f'identity:{key}' for key in IDENTITY_COMPONENTS
                if baseline['identity'].get(key) != current_identity.get(key)]
    if reasons:
        return {'decision': 'rerun_before', 'reasons': reasons, 'identity': current_identity}
    return {'decision': 'use_baseline', 'reasons': []}


def comparable_rows(rows: list) -> list:
    """Only core-script rows are compared; extra_rows never are (R2-4), nor fixture-503 views (R6-7)."""
    return [row for row in rows if not row.get('extra') and row.get('view') not in FIXTURE_UNAVAILABLE_VIEWS]


def warm_rows_comparable(before_mix: dict, after_mix: dict) -> bool:
    return all(abs(before_mix.get(key, 0) - after_mix.get(key, 0)) <= 10 for key in ('memory', 'disk', '304', 'network'))


def arm_median(values: list):
    """V10: median of an arm with '<16 ms' ranked below measured values and failures (+inf) above; the result is a
    number, CENSORED or INF (the lower middle value when the two middle values differ in kind)."""
    ordered = sorted_values(values)
    if not ordered:
        return None
    low, high = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
    if CENSORED in (low, high) or INF in (low, high):
        return low
    return (low + high) / 2


def _bounds(median) -> tuple:
    """A median as an interval: '<16 ms' is [0, 16), a number is [value, value]."""
    return (0.0, float(EVENT_DURATION_THRESHOLD_MS)) if median == CENSORED else (float(median), float(median))


def component(minuend, subtrahend) -> dict:
    """Difference of two arm medians as an interval; 'no evaluable' if either is +inf or missing."""
    if minuend is None or subtrahend is None or INF in (minuend, subtrahend):
        return {'low': None, 'high': None, 'evaluable': False}
    (a_low, a_high), (b_low, b_high) = _bounds(minuend), _bounds(subtrahend)
    return {'low': a_low - b_high, 'high': a_high - b_low, 'evaluable': True}


def overhead(arms: dict, metrics: tuple = ('inp', 'raf_feedback')) -> dict:
    """smoke_gate.instrumentation_overhead: overhead = median(a) - median(c); components a-b and b-c.

    V10 (preregistered): censored medians enter as [0, 16) intervals; a consequence triggers when the component's
    lower bound exceeds 8 ms; an arm median of +inf leaves the component 'no evaluable', which is a method change
    (run_rules.method_changes), never a silent default."""
    result = {}
    unevaluable = []
    for metric in metrics:
        median = {arm: arm_median(arms[arm][metric]) for arm in ('a', 'b', 'c')}
        result[metric] = {'overhead': component(median['a'], median['c']),
                          'screencast': component(median['a'], median['b']),
                          'tracing': component(median['b'], median['c']), 'medians': median}
        unevaluable += [f'{metric}:{name}' for name in ('screencast', 'tracing')
                        if not result[metric][name]['evaluable']]
    exceeds = lambda item: item['evaluable'] and item['low'] > 8
    result['consequences'] = {
        'screencast_limited_to_contrast': any(exceeds(result[metric]['screencast']) for metric in metrics),
        'untraced_inp': 'inp' in metrics and exceeds(result['inp']['tracing']),
        'unevaluable': unevaluable,
        'method_change': bool(unevaluable),
    }
    return result


OVERHEAD_METRICS = {'#acciones': ('inp', 'raf_feedback'), '#voces': ('inp',)}


def overhead_consequences(per_view: dict) -> dict:
    """V10: the preregistered consequences over both views (#acciones INP and rAF bound, #voces INP; the C11
    preview uses the #acciones rAF-bound overhead, R5-14e): a consequence applies if it applies on either view."""
    results = {view: overhead(arms, OVERHEAD_METRICS[view]) for view, arms in per_view.items()}
    consequences = {
        'screencast_limited_to_contrast': any(item['consequences']['screencast_limited_to_contrast']
                                              for item in results.values()),
        'untraced_inp': any(item['consequences']['untraced_inp'] for item in results.values()),
        'unevaluable': [f'{view}:{name}' for view, item in results.items()
                        for name in item['consequences']['unevaluable']],
    }
    consequences['method_change'] = bool(consequences['unevaluable'])
    return {'views': results, 'consequences': consequences}


def reconfirmation_arm(metric: str) -> str:
    """statistics.local_trigger_for_T046_U3: arm (b) for C11 action feedback, arm (c) for INP and preview."""
    return 'b' if metric == 'c11_action_feedback' else 'c'


def frame_event_choice(phases: list) -> dict:
    """smoke_gate.frame_event (R5-6): first candidate present in every phase and passing on >= 95%."""
    shares = {}
    for candidate in ('PipelineReporter', 'Display::FrameDisplayed', 'DrawFrame'):
        present = all(candidate in phase['present'] for phase in phases)
        checks = [check[candidate] for phase in phases for check in phase['interactions'] if candidate in check]
        share = sum(1 for value in checks if value) / len(checks) if checks else 0.0
        shares[candidate] = {'present_everywhere': present, 'share': share}
        if present and share >= 0.95:
            return {'candidate': candidate, 'shares': shares, 'method_change': False}
    return {'candidate': None, 'shares': shares, 'method_change': True}


def harness_only_deadlines(current: dict, healthy_use_ms: dict) -> dict:
    """R5-5: max(current, 2 x largest healthy use rounded up to 1 s) for harness-only classes; others unchanged."""
    result = dict(current)
    for klass in HARNESS_ONLY_CLASSES:
        largest = max(healthy_use_ms.get(klass, [0]) or [0])
        result[klass] = max(current[klass], math.ceil(2 * largest / 1000) * 1000)
    return result


def warm_rule_active(fill_ins: dict) -> bool:
    """V9 (runs.warm, smoke_gate.warm_cache): the per-batch 90% warm-source rule discards batches only when the
    smoke reached >= 90%; when it did not, warm rows are 'no evaluable' and no batch is discarded for it."""
    return (fill_ins.get('warm_cache') or {}).get('reached', True) is not False


def background_branch(profiles: list) -> str:
    """smoke_gate.background_activity: any non-loopback request in 10 s on any of 5 profiles -> net log."""
    loopback = re.compile(r'^(?:https?|wss?)://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:/|$)')
    for profile in profiles:
        if any(not loopback.match(url) for url in profile.get('requests', [])):
            return 'netlog'
    return 'none'


def fill_in_allowed(item: str) -> bool:
    return item in ('frame_event', 'preview_ready_state', 'quiet_baseline', 'lock_read', 'mac_environment',
                    'preview_gap', 'inp_grouping', 'load_settle', 'cdp_deadlines', 'background_activity',
                    'warm_cache', 'cold_focus', 'ui_locale', 'instrumentation_overhead')


# --------------------------------------------------------------------------------------------
# Modes, hashes and output locations (freeze_rule, smoke_gate.smoke_mode).
# --------------------------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preregistration(path: Path, expected: str) -> str:
    """The probe verifies the hash at start and refuses any other version."""
    actual = file_sha256(path)
    if not re.fullmatch(r'[0-9a-f]{64}', expected or '') or actual != expected:
        raise SystemExit(f'preregistration_hash_mismatch: expected {expected}, found {actual}')
    return actual


def output_root(mode: str) -> Path:
    if mode == 'smoke':
        return SMOKE_ROOT
    if mode == 'harness':
        return HARNESS_ROOT
    return OUTPUT_ROOT / 'u1' / 'baseline'


def checked_output(mode: str, path: Path) -> Path:
    """Smoke results only under t046/smoke/; harness only under t046/u1/harness/; baseline never in either."""
    resolved = Path(path).resolve()
    root = output_root(mode).resolve()
    if root not in (resolved, *resolved.parents):
        raise SystemExit(f'output_outside_mode_root:{mode}:{resolved}')
    if mode == 'baseline' and any(other.resolve() in resolved.parents for other in (SMOKE_ROOT, HARNESS_ROOT)):
        raise SystemExit('baseline_output_in_smoke_or_harness_root')
    return resolved


def refuse_smoke_input(document: dict) -> None:
    """Baseline mode refuses any input produced in smoke mode (run_rules.no_baseline_use)."""
    if document.get('mode') == 'smoke':
        raise SystemExit('smoke_result_refused_in_baseline_mode')


SMOKE_REPEAT_CAUSES = ('crash', 'disconnect', 'lock', 'quiet_machine', 'fixture_fault')


class SmokeRecord:
    """Every smoke run is recorded in full; a repeat needs a harness or environment cause that the previous run of
    the same item recorded (R4-3, V9). An entry left 'started' (the probe itself stopped) records 'crash'."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {'mode': 'smoke', 'runs': []}

    def previous(self, item: str) -> list:
        return [run for run in self.data['runs'] if run['item'] in (item, 'all') or item == 'all']

    def start(self, item: str, reviewed_sha256: str, cause: str | None = None) -> dict:
        previous = self.previous(item)
        if previous:
            last = previous[-1]
            recorded = set(last.get('faults') or [])
            if last.get('status') == 'started':
                recorded.add('crash')
            if cause not in SMOKE_REPEAT_CAUSES or cause not in recorded:
                raise SystemExit(f'smoke_repeat_refused_without_recorded_cause:{item}:{cause}:recorded={sorted(recorded)}')
        entry = {'item': item, 'reviewed_sha256': reviewed_sha256, 'started_at': time.time(),
                 'repeat_of': [run['started_at'] for run in previous], 'cause': cause, 'status': 'started'}
        self.data['runs'].append(entry)
        write_json_atomic(self.path, self.data)
        return entry

    def finish(self, entry: dict, status: str, result: dict | None = None, faults=None) -> None:
        entry.update(status=status, finished_at=time.time(), result=result, faults=sorted(set(faults or [])))
        write_json_atomic(self.path, self.data)


def smoke_idle(record: SmokeRecord) -> float:
    """The smoke's idle load from its own complete quiet_baseline item (smoke_gate.quiet_baseline)."""
    done = [run for run in record.data['runs'] if run['item'] == 'quiet_baseline' and run['status'] == 'complete']
    if not done:
        raise SystemExit('smoke_quiet_baseline_missing: run `smoke-item quiet-baseline` first')
    return done[-1]['result']['idle']


def _last_complete(record: SmokeRecord, item: str) -> dict | None:
    done = [run for run in record.data['runs'] if run['item'] == item and run['status'] == 'complete']
    return done[-1] if done else None


def smoke_play_rule(record: SmokeRecord) -> dict:
    """R6-1 (smoke_gate.smoke_order, preview_ready_state): the play-control rule the smoke's own complete
    fixture_pages item chose; smoke batches and overhead runs refuse to start without it."""
    entry = _last_complete(record, 'fixture_pages')
    rule = ((((entry or {}).get('result') or {}).get('observations') or {}).get('play_control_rule') or {}).get('rule')
    if not rule:
        raise SystemExit('smoke_play_control_rule_missing: run `smoke-item fixture-pages` first; '
                         'a fixture_pages item without a chosen rule is a method change')
    return rule


def smoke_background_branch(record: SmokeRecord) -> str:
    """R6-1 (smoke_gate.smoke_order): the branch of the smoke's own complete background_activity item."""
    entry = _last_complete(record, 'background_activity')
    branch = ((entry or {}).get('result') or {}).get('branch')
    if branch not in ('none', 'netlog'):
        raise SystemExit('smoke_background_branch_missing: run `smoke-item background` first')
    return branch


# R6-1 (smoke_gate.smoke_order): quiet_baseline; cold_focus and background_activity; fixture_pages; the 12 batches;
# instrumentation_overhead; frame_event. lock_read items may run at any point. An item refuses to start until every
# item before it has a complete entry (a batch cell also counts when its entry was discarded: it is recorded and never
# repeated automatically).
SMOKE_ORDER = ('quiet_baseline', ('cold_focus', 'background_activity'), 'fixture_pages', 'batches',
               'instrumentation_overhead', 'frame_event')


def smoke_order_missing(record: SmokeRecord, item: str) -> list:
    """The earlier smoke items that have no complete entry yet (R6-1)."""
    stage = next(index for index, entry in enumerate(SMOKE_ORDER)
                 if item == entry or (isinstance(entry, tuple) and item in entry))
    missing = []
    for entry in SMOKE_ORDER[:stage]:
        for name in (entry if isinstance(entry, tuple) else (entry,)):
            if name == 'batches':
                for cell in batch_plan_views():
                    runs = record.previous(f'batch:{cell["viewport"]}:{cell["view"]}')
                    if not runs or runs[-1].get('status') not in ('complete', 'discarded'):
                        missing.append(f'batch:{cell["viewport"]}:{cell["view"]}')
            elif _last_complete(record, name) is None:
                missing.append(name)
    return missing


def check_smoke_order(record: SmokeRecord, item: str) -> None:
    missing = smoke_order_missing(record, item)
    if missing:
        raise SystemExit(f'smoke_order_refused:{item}: first complete {",".join(missing)}')


def cell_settings(mode: str, fill_ins: dict, record: SmokeRecord | None = None) -> dict:
    """Per-mode inputs of a batch (R6-1, R6-2, R6-9). Smoke: the smoke's own quiet_baseline idle, fixture_pages
    play rule and background branch, the smoke composition, no warm-share discard. Baseline: the fill-ins (refused
    when missing) and the frozen composition 3/40/10."""
    if mode == 'smoke':
        check_smoke_order(record, 'batches')
        return {'idle': smoke_idle(record), 'post_batch': None, 'play_control_rule': smoke_play_rule(record),
                'play_control_rule_source': 'smoke_record:fixture_pages',
                'background_branch': smoke_background_branch(record), 'composition': dict(BATCH_COMPOSITION['smoke']),
                'warm_rule': False, 'frame_event': None, 'untraced_inp': False, 'overhead': None}
    missing = [key for key in ('idle_load', 'post_batch_load', 'play_control_rule', 'background_branch', 'frame_event')
               if fill_ins.get(key) is None]
    if missing:
        raise SystemExit(f'fill_ins_missing:{",".join(missing)}: the smoke fill-ins are not in the frozen file')
    frame_event = fill_ins.get('frame_event')
    frame_event = frame_event.get('candidate') if isinstance(frame_event, dict) else frame_event
    return {'idle': fill_ins['idle_load'], 'post_batch': fill_ins['post_batch_load'],
            'play_control_rule': fill_ins['play_control_rule'], 'play_control_rule_source': 'fill_ins',
            'background_branch': fill_ins['background_branch'], 'composition': dict(BATCH_COMPOSITION['baseline']),
            'warm_rule': warm_rule_active(fill_ins), 'frame_event': frame_event,
            'untraced_inp': bool((fill_ins.get('overhead') or {}).get('untraced_inp')),
            'overhead': fill_ins.get('overhead')}


def write_json_atomic(path: Path, value) -> None:
    """Atomic JSON write (temp file, fsync, replace), mode 0600."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as output:
            json.dump(value, output, indent=1, sort_keys=True, ensure_ascii=False, default=_json_default)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_gzip_json_atomic(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as output:
            output.write(gzip.compress(json.dumps(value, default=_json_default).encode()))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, float) and math.isinf(value):
        return 'Infinity'
    if hasattr(value, '__dataclass_fields__'):
        return asdict(value)
    raise TypeError(type(value).__name__)


def restore_infinity(value):
    """Inverse of json_safe for metric values: 'Infinity' back to +inf at any depth."""
    if value == 'Infinity':
        return INF
    if isinstance(value, dict):
        return {key: restore_infinity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_infinity(item) for item in value]
    return value


def json_safe(value):
    """Replace +inf with the string 'Infinity' (JSON has no infinity)."""
    if isinstance(value, float) and math.isinf(value):
        return 'Infinity'
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


# --------------------------------------------------------------------------------------------
# CDP client (instrumentation.cdp_client, cdp_deadlines).
# --------------------------------------------------------------------------------------------

class CdpTimeout(Exception):
    """A command reached its absolute deadline."""

    def __init__(self, method: str, klass: str, phase: str):
        super().__init__(f'{method} ({klass}, {phase})')
        self.method, self.klass, self.phase = method, klass, phase


class CdpDisconnected(Exception):
    """The CDP websocket closed."""


class CdpError(Exception):
    """CDP answered with an error."""


def method_class_allowed(method: str, klass: str) -> bool:
    if klass not in CLASS_METHODS or method.startswith(FORBIDDEN_PREFIXES):
        return False
    return any(method == allowed or (allowed.endswith('.') and method.startswith(allowed))
               for allowed in CLASS_METHODS[klass])


class CdpClient:
    """One reader task per websocket; per-command futures with absolute deadlines (G9, G13)."""

    def __init__(self, connection, *, name: str, deadlines: dict | None = None, clock=time.monotonic):
        self.connection = connection
        self.name = name
        self.deadlines = dict(DEADLINE_MS if deadlines is None else deadlines)
        self.clock = clock
        self.sequence = 0
        self.pending = {}
        self.handlers = defaultdict(list)
        self.log = []
        self.events = []
        self.keep_events = True
        self.disconnected = False
        self.phase = 'setup'
        self.reader = asyncio.get_running_loop().create_task(self._read())

    def on(self, method: str, handler: Callable[[dict], Any]) -> None:
        self.handlers[method].append(handler)

    async def _read(self) -> None:
        try:
            async for message in self.connection:
                data = json.loads(message)
                if 'id' in data:
                    future = self.pending.pop(data['id'], None)
                    if future is not None and not future.done():
                        future.set_result(data)
                    continue
                method = data.get('method')
                if self.keep_events and method not in ('Tracing.dataCollected', 'Page.screencastFrame'):
                    self.events.append({'method': method, 'received': self.clock(), 'params': data.get('params')})
                for handler in self.handlers.get(method, ()):
                    try:
                        result = handler(data.get('params', {}))
                        if asyncio.iscoroutine(result):
                            asyncio.get_running_loop().create_task(result)
                    except Exception:  # noqa: BLE001 - a handler error is logged, never a reader stop.
                        logger.exception('[web-timing] CDP handler failed for %s', method)
        except Exception:  # noqa: BLE001 - any close is a disconnect.
            pass
        finally:
            self.disconnected = True
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(CdpDisconnected(self.name))
            self.pending.clear()

    async def send(self, method: str, params: dict | None = None, *, klass: str, phase: str | None = None) -> dict:
        """Send one command and wait at most its class deadline from the send (events never extend it).

        ``phase`` labels a command scheduled in one phase and sent later (window-state reads, V4)."""
        if not method_class_allowed(method, klass):
            raise ValueError(f'no_deadline_class:{method}:{klass}')
        if self.disconnected:
            raise CdpDisconnected(self.name)
        self.sequence += 1
        identity = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[identity] = future
        entry = {'id': identity, 'method': method, 'class': klass, 'phase': phase or self.phase, 'sent': self.clock(),
                 'connection': self.name}
        self.log.append(entry)
        await self.connection.send(json.dumps({'id': identity, 'method': method, 'params': params or {}}))
        try:
            reply = await asyncio.wait_for(future, self.deadlines[klass] / 1000)
        except asyncio.TimeoutError:
            self.pending.pop(identity, None)
            entry.update(timeout=True, deadline=entry['sent'] + self.deadlines[klass] / 1000)
            raise CdpTimeout(method, klass, entry['phase']) from None
        entry['answered'] = self.clock()
        if 'error' in reply:
            entry['error'] = reply['error']
            raise CdpError(f'{method}: {reply["error"]}')
        return reply.get('result', {})

    async def send_pipelined(self, commands: list, *, klass: str) -> list:
        """Write several commands back to back, then wait for each reply under its own absolute deadline.

        Used for the input events of one gesture, so a busy renderer cannot delay the dispatch (and the browser's
        event timestamp) of the later events behind the reply to the first (smoke_gate.preview_gap, R5-8).
        """
        for method, _ in commands:
            if not method_class_allowed(method, klass):
                raise ValueError(f'no_deadline_class:{method}:{klass}')
        if self.disconnected:
            raise CdpDisconnected(self.name)
        waits = []
        for method, params in commands:
            self.sequence += 1
            identity = self.sequence
            future = asyncio.get_running_loop().create_future()
            self.pending[identity] = future
            entry = {'id': identity, 'method': method, 'class': klass, 'phase': self.phase, 'sent': self.clock(),
                     'connection': self.name, 'params': params}
            self.log.append(entry)
            await self.connection.send(json.dumps({'id': identity, 'method': method, 'params': params or {}}))
            waits.append((identity, entry, future))
        results = []
        for identity, entry, future in waits:
            remaining = entry['sent'] + self.deadlines[klass] / 1000 - self.clock()
            try:
                reply = await asyncio.wait_for(future, max(0.0, remaining))
            except asyncio.TimeoutError:
                for other, _, _ in waits:
                    self.pending.pop(other, None)
                entry.update(timeout=True, deadline=entry['sent'] + self.deadlines[klass] / 1000)
                raise CdpTimeout(entry['method'], klass, entry['phase']) from None
            entry['answered'] = self.clock()
            if 'error' in reply:
                entry['error'] = reply['error']
                raise CdpError(f'{entry["method"]}: {reply["error"]}')
            results.append(reply.get('result', {}))
        return results

    async def close(self) -> None:
        self.reader.cancel()
        try:
            await self.connection.close()
        except Exception:  # noqa: BLE001
            pass


class SetupAdapter:
    """install_login_session needs a client with a ``command`` coroutine; setup class deadlines."""

    def __init__(self, client: CdpClient):
        self.client = client

    async def command(self, method, params=None):
        return await self.client.send(method, params, klass='setup')


def post_window_violations(log: list) -> list:
    """R4-2: no Runtime.*, Input.* or DOM.* command after the final flush; records only through the binding."""
    flush = None
    for index, entry in enumerate(log):
        if entry.get('flush'):
            flush = index
    if flush is None:
        return []
    return [entry['method'] for entry in log[flush + 1:]
            if entry['method'].startswith(('Runtime.', 'Input.', 'DOM.', 'Accessibility.'))]


def setup_commands(*, role: str, viewport: str, script_source: str, previous_script: str | None) -> list:
    """Ordered setup commands for a run's target (phase 1)."""
    metrics = VIEWPORTS[viewport]
    commands = [('Page.enable', {}), ('Runtime.enable', {}), ('Network.enable', {})]
    if role == 'cold':
        commands += [('Network.setCacheDisabled', {'cacheDisabled': True}), ('Network.clearBrowserCache', {})]
    else:
        commands.append(('Network.setCacheDisabled', {'cacheDisabled': False}))
    commands += [
        ('Emulation.setDeviceMetricsOverride', {'width': metrics['width'], 'height': metrics['height'],
                                                'deviceScaleFactor': metrics['deviceScaleFactor'],
                                                'mobile': metrics['mobile'], 'screenWidth': metrics['width'],
                                                'screenHeight': metrics['height']}),
        ('Emulation.setTouchEmulationEnabled', {'enabled': metrics['mobile'], **({'maxTouchPoints': 5} if metrics['mobile'] else {})}),
    ]
    if previous_script is not None:
        commands.append(('Page.removeScriptToEvaluateOnNewDocument', {'identifier': previous_script}))
    commands += [('Runtime.addBinding', {'name': 'kzaProbeRecord'}),
                 ('Page.addScriptToEvaluateOnNewDocument', {'source': script_source})]
    return commands


def post_window_commands(trace: bool, screencast: bool) -> list:
    """Phase 3 commands: browser-side only."""
    commands = []
    if trace:
        commands.append(('Tracing.end', 'collect', 'browser'))
    if screencast:
        commands.append(('Page.stopScreencast', 'collect', 'page'))
    commands += [('Browser.getWindowForTarget', 'setup', 'browser'), ('Browser.getVersion', 'version', 'browser')]
    return commands


# --------------------------------------------------------------------------------------------
# Launch and activation (clients.local.launch, R4-6, R5-14a).
# --------------------------------------------------------------------------------------------

def app_bundle(binary: str) -> str:
    marker = binary.find('.app/')
    if marker < 0:
        raise ValueError('browser_binary_not_in_app_bundle')
    return binary[:marker + 4]


def launch_argv(binary: str, profile: Path, run_log: Path, *, netlog: Path | None = None) -> list:
    """The frozen `open -na` launch (LaunchServices activates the new instance)."""
    flags = list(LAUNCH_FLAGS)
    if netlog is not None:
        flags.append(f'--log-net-log={netlog}')
    return ['open', '-na', app_bundle(binary), '--stdout', str(run_log), '--stderr', str(run_log), '--args', *flags,
            f'--user-data-dir={profile}', '--remote-debugging-port=0', 'about:blank']


def harness_argv(binary: str, profile: Path) -> list:
    """Harness mode only: server headless Chromium, never used for a measurement."""
    return [binary, *HARNESS_EXTRA_FLAGS, *LAUNCH_FLAGS, f'--user-data-dir={profile}', '--remote-debugging-port=0',
            'about:blank']


async def activation_poll(page: CdpClient, browser: CdpClient, target_id: str, deadline_s: float) -> int:
    """Page.bringToFront, then poll hasFocus and windowState every 100 ms until both hold in one poll."""
    started = time.monotonic()
    await page.send('Page.bringToFront', klass='setup')
    polls = 0
    while True:
        polls += 1
        focus = await page.send('Runtime.evaluate', {'expression': 'document.hasFocus()', 'returnByValue': True},
                                klass='setup')
        window = await browser.send('Browser.getWindowForTarget', {'targetId': target_id}, klass='setup')
        if focus.get('result', {}).get('value') is True and window.get('bounds', {}).get('windowState') == 'normal':
            return polls
        if time.monotonic() - started + ACTIVATION_POLL_S > deadline_s:
            raise ActivationFailed(polls)
        await asyncio.sleep(ACTIVATION_POLL_S)


class ActivationFailed(Exception):
    """The activation poll never held within the setup deadline (invalid run)."""

    def __init__(self, polls: int):
        super().__init__(f'activation_failed_after_{polls}_polls')
        self.polls = polls


# --------------------------------------------------------------------------------------------
# Trace analysis (instrumentation.frame_clock, smoke_gate.frame_event).
# --------------------------------------------------------------------------------------------

def trace_marks(events: list) -> dict:
    """blink.user_timing marks named kza-probe:* -> trace ts (ms)."""
    marks = {}
    for event in events:
        name = event.get('name', '')
        if name.startswith('kza-probe:') and event.get('ph') in ('R', 'I', 'n', 'b', 'i', 'X', 'e'):
            marks.setdefault(name, event['ts'] / 1000)
    return marks


def trace_offset(trace_marks_ms: dict, page_marks: list) -> float | None:
    """trace_time - page_time, from marks present in both (median for robustness)."""
    offsets = [trace_marks_ms[mark['name']] - mark['time'] for mark in page_marks if mark['name'] in trace_marks_ms]
    return statistics.median(offsets) if offsets else None


def frame_candidates(events: list) -> dict:
    """Candidate presented-frame times in trace ms: PipelineReporter end, Display::FrameDisplayed, DrawFrame."""
    found = {'PipelineReporter': [], 'Display::FrameDisplayed': [], 'DrawFrame': []}
    begins = {}
    for event in events:
        name = event.get('name')
        if name == 'PipelineReporter':
            args = event.get('args', {})
            state = (args.get('chrome_frame_reporter') or {}).get('state') or args.get('state')
            key = (event.get('id') or event.get('id2', {}).get('local'), event.get('pid'))
            if event.get('ph') == 'X':
                if state == 'STATE_PRESENTED_ALL':
                    found[name].append({'begin': event['ts'] / 1000, 'time': (event['ts'] + event.get('dur', 0)) / 1000,
                                        'state': state})
            elif event.get('ph') == 'b':
                begins[key] = (event['ts'] / 1000, state)
            elif event.get('ph') == 'e' and key in begins:
                begin, begin_state = begins.pop(key)
                final = state or begin_state
                if final == 'STATE_PRESENTED_ALL':
                    found[name].append({'begin': begin, 'time': event['ts'] / 1000, 'state': final})
        elif name in ('Display::FrameDisplayed', 'DrawFrame'):
            found[name].append({'begin': event['ts'] / 1000, 'time': event['ts'] / 1000})
    return found


def presented_after(candidates: list, mutation_trace_ms: float) -> float | None:
    """First frame whose rendering began after the end mutation."""
    times = sorted(item['time'] for item in candidates if item['begin'] > mutation_trace_ms)
    return times[0] if times else None


def renderer_busy_across(events: list, sent_trace_ms: float, deadline_trace_ms: float) -> bool | None:
    """R5-14c: a renderer main-thread task that began before the command and was running at its deadline."""
    if not events:
        return None
    threads = {(event['pid'], event['tid']) for event in events
               if event.get('ph') == 'M' and event.get('name') == 'thread_name'
               and (event.get('args') or {}).get('name') == 'CrRendererMain'}
    open_begins = {}
    for event in events:
        key = (event.get('pid'), event.get('tid'))
        if key not in threads:
            continue
        start = event.get('ts', 0) / 1000
        if event.get('ph') == 'X' and start < sent_trace_ms and start + event.get('dur', 0) / 1000 >= deadline_trace_ms:
            return True
        if event.get('ph') == 'B':
            open_begins.setdefault(key, []).append(start)
        elif event.get('ph') == 'E' and open_begins.get(key):
            open_begins[key].pop()
    return any(start < sent_trace_ms for starts in open_begins.values() for start in starts)


# --------------------------------------------------------------------------------------------
# Network cache sources (runs.warm).
# --------------------------------------------------------------------------------------------

STATIC_TYPES = frozenset({'Script', 'Stylesheet', 'Image', 'Font', 'Other'})


class NetworkLedger:
    """Per-request cache source from Network.* events (memory, disk, 304, network)."""

    def __init__(self, origin: str):
        self.origin = origin
        self.requests = {}

    def will_be_sent(self, params: dict) -> None:
        self.requests.setdefault(params['requestId'], {})['url'] = params.get('request', {}).get('url')

    def served_from_cache(self, params: dict) -> None:
        self.requests.setdefault(params['requestId'], {})['memory'] = True

    def response(self, params: dict) -> None:
        item = self.requests.setdefault(params['requestId'], {})
        response = params.get('response', {})
        item.update(url=response.get('url', item.get('url')), type=params.get('type'), status=response.get('status'),
                    disk=bool(response.get('fromDiskCache')), prefetch=bool(response.get('fromPrefetchCache')))

    def errors(self) -> list:
        """V11: same-origin responses with status >= 400 (path, status, resource type), recorded per run."""
        found = []
        for item in self.requests.values():
            url = item.get('url') or ''
            status = item.get('status')
            if url.startswith(self.origin) and isinstance(status, int) and status >= 400:
                found.append({'path': url[len(self.origin):][:200], 'status': status, 'type': item.get('type')})
        return found

    def sources(self) -> dict:
        mix = Counter()
        for item in self.requests.values():
            url = item.get('url') or ''
            if not url.startswith(self.origin) or item.get('type') not in STATIC_TYPES:
                continue
            if item.get('memory'):
                mix['memory'] += 1
            elif item.get('disk'):
                mix['disk'] += 1
            elif item.get('status') == 304:
                mix['304'] += 1
            else:
                mix['network'] += 1
        return dict(mix)


# --------------------------------------------------------------------------------------------
# Method identity, closure, environment (method_identity, statistics.before_after, quiet_machine).
# --------------------------------------------------------------------------------------------

def run_text(command: list, *, timeout: float = 30) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


def git(*args) -> str:
    return run_text(['git', '-C', str(REPOSITORY), *args]).strip()


def tree_dirty(repository: Path = REPOSITORY) -> bool:
    """R5-14f: untracked files count, ignored files do not."""
    return bool(run_text(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=all']).strip())


def pip_freeze_sha256() -> str:
    return hashlib.sha256(run_text([sys.executable, '-m', 'pip', 'freeze'], timeout=120).encode()).hexdigest()


def repository_modules() -> list:
    paths = set()
    for module in list(sys.modules.values()):
        location = getattr(module, '__file__', None)
        if not location:
            continue
        try:
            relative = Path(location).resolve().relative_to(REPOSITORY)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] in ('.venv',):
            continue
        paths.add(str(relative))
    return sorted(paths)


def recorded_closure(fixture_modules: list) -> list:
    """R5-12: fixture and probe repository modules, the served static tree and the dependency files."""
    served = [str(path.relative_to(REPOSITORY)) for path in (REPOSITORY / 'src/dashboard/frontend').rglob('*')
              if path.is_file()]
    return sorted(set(repository_modules()) | set(fixture_modules) | set(served)
                  | {'requirements.txt', 'requirements-test.txt', 'pyproject.toml'})


def mac_environment() -> dict:
    languages = run_text(['defaults', 'read', '-g', 'AppleLanguages'])
    first = re.findall(r'"?([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*)"?', languages)
    return {'os_version': run_text(['sw_vers', '-productVersion']).strip(),
            'hardware': run_text(['sysctl', '-n', 'hw.model']).strip(),
            'apple_languages_first': first[0] if first else None,
            'power': run_text(['pmset', '-g', 'batt']).strip().splitlines()[:1],
            'assertions': run_text(['pmset', '-g', 'assertions']),
            'displays': run_text(['system_profiler', 'SPDisplaysDataType', '-json'], timeout=60)}


def lock_reads() -> dict:
    """quiet_machine.lock_read: read 1 via ioreg, read 2 via CGSessionCopyCurrentDictionary (JXA)."""
    result = {'read1': None, 'read2': None}
    raw = subprocess.run(['ioreg', '-n', 'Root', '-d1', '-a'], capture_output=True, check=False).stdout
    try:
        root = plistlib.loads(raw) if raw else {}
        sessions = root.get('IOConsoleUsers', []) if isinstance(root, dict) else []
        console = [session for session in sessions if session.get('kCGSSessionOnConsoleKey')]
        values = [session['CGSSessionScreenIsLocked'] for session in console if 'CGSSessionScreenIsLocked' in session]
        result['read1'] = {'present': bool(values), 'locked': bool(values[0]) if values else None}
    except Exception as error:  # noqa: BLE001
        result['read1'] = {'error': str(error)[:200]}
    script = ("ObjC.import('CoreGraphics');var d=$.CGSessionCopyCurrentDictionary();"
              "var v=d.js['CGSSessionScreenIsLocked'];JSON.stringify({present:v!==undefined,"
              "locked:v===undefined?null:!!ObjC.unwrap(v)})")
    text = run_text(['osascript', '-l', 'JavaScript', '-e', script]).strip()
    try:
        result['read2'] = json.loads(text)
    except ValueError:
        result['read2'] = {'error': text[:200]}
    return result


def display_asleep() -> bool | None:
    """V8: CGDisplayIsAsleep(CGMainDisplayID()) through JXA (no new dependency); None when unreadable."""
    text = run_text(['osascript', '-l', 'JavaScript', '-e',
                     "ObjC.import('CoreGraphics');JSON.stringify({asleep: !!$.CGDisplayIsAsleep($.CGMainDisplayID())})"])
    try:
        return bool(json.loads(text)['asleep'])
    except (ValueError, KeyError, TypeError):
        return None


def chosen_lock_state(reads: dict, choice: str | None) -> bool | None:
    """quiet_machine.lock_read with the read the smoke chose (fill-in): key true -> locked; absent or false ->
    unlocked; unreadable or no read chosen yet -> None."""
    if choice not in ('read1', 'read2'):
        return None
    read = reads.get(choice) or {}
    if 'error' in read or not read:
        return None
    return bool(read.get('present') and read.get('locked'))


def parse_audio_output(text: str) -> dict | None:
    """R6-8: the default CoreAudio output device from `system_profiler SPAudioDataType -json`; None if unreadable."""
    try:
        groups = json.loads(text).get('SPAudioDataType') or []
    except (ValueError, AttributeError):
        return None
    for group in groups:
        for device in group.get('_items') or []:
            if device.get('coreaudio_default_audio_output_device') == 'spaudio_yes':
                transport = device.get('coreaudio_device_transport')
                return {'name': device.get('_name'), 'transport': transport,
                        'builtin': transport == 'coreaudio_device_type_builtin'}
    return None


def audio_output() -> dict | None:
    return parse_audio_output(run_text(['system_profiler', 'SPAudioDataType', '-json'], timeout=60))


def machine_state(lock_choice: str | None = None) -> dict:
    """V8: AC power, display awake and lock state, read at batch start and end; R6-8: default audio output."""
    power = run_text(['pmset', '-g', 'batt']).strip().splitlines()[:1]
    reads = lock_reads()
    return {'power': power, 'ac_power': bool(power) and "'AC Power'" in power[0],
            'display_asleep': display_asleep(), 'lock_reads': reads, 'lock_choice': lock_choice,
            'locked': chosen_lock_state(reads, lock_choice), 'audio_output': audio_output()}


def start_caffeinate() -> subprocess.Popen:
    """R2-3, V7: caffeinate -dimsu bound to the probe's own pid (-w), so it ends with the probe."""
    return subprocess.Popen(['caffeinate', '-dimsu', '-w', str(os.getpid())])


def caffeinate_assertion_held(process, assertions: str | None = None) -> bool:
    """V8: `pmset -g assertions` lists PreventUserIdleDisplaySleep owned by this caffeinate process."""
    if process is None:
        return False
    text = run_text(['pmset', '-g', 'assertions']) if assertions is None else assertions
    marker = f'pid {process.pid}(caffeinate)'
    return any(marker in line and 'PreventUserIdleDisplaySleep' in line for line in text.splitlines())


PROFILE_PREFIXES = ('kza-t046-profile-', 'kza-t046-focus-', 'kza-t046-background-')


def probe_browser_pids(rows: list) -> list:
    """V7: main browser processes launched by a probe (profile directory under a probe prefix, no --type=)."""
    found = []
    for row in rows:
        command = row['command']
        if '--type=' in command:
            continue
        match = re.search(r'--user-data-dir=(\S+)', command)
        if match and Path(match.group(1)).name.startswith(PROFILE_PREFIXES):
            found.append(row['pid'])
    return found


def sweep_probe_browsers(rows: list | None = None, kill: Callable[[int, int], None] = os.kill) -> list:
    """V7: kill any leftover probe-launched Chrome (with an open DevTools port) at start and exit."""
    pids = probe_browser_pids(process_rows() if rows is None else rows)
    for pid in pids:
        try:
            kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return pids


def process_rows() -> list:
    rows = []
    for line in run_text(['ps', '-axo', 'pid=,ppid=,command=']).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append({'pid': int(parts[0]), 'ppid': int(parts[1]), 'command': parts[2]})
    return rows


def cpu_seconds() -> dict:
    result = {}
    for line in run_text(['ps', '-axo', 'pid=,time=']).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            fields = [float(value) for value in re.split('[:-]', parts[1])]
            seconds = 0.0
            for value in fields:
                seconds = seconds * 60 + value
            result[int(parts[0])] = seconds
    return result


async def quiet_check(own_pid: int, allowed: set | None = None) -> dict:
    """Process check with a 10 s CPU sample (quiet_machine.mac)."""
    before = cpu_seconds()
    await asyncio.sleep(10)
    after = cpu_seconds()
    cpu = {pid: 100 * (after[pid] - before.get(pid, after[pid])) / 10 for pid in after}
    rows = process_rows()
    violations = process_check(rows, own_pid, cpu=cpu, allowed=allowed)
    pytest_count = sum(1 for kind, _ in violations if kind == 'pytest')
    return {'violations': violations, 'pytest_count': pytest_count, 'ok': not violations and pytest_count == 0}


def loadavg() -> float:
    return parse_loadavg(run_text(['sysctl', '-n', 'vm.loadavg']))


# --------------------------------------------------------------------------------------------
# Browser session (one Chrome instance; cold run launches, warm reuses).
# --------------------------------------------------------------------------------------------

class Browser:
    """A launched Chrome/Chromium with its browser endpoint client."""

    def __init__(self, mode: str, binary: str, profile: Path, run_log: Path):
        self.mode = mode
        self.binary = binary
        self.profile = profile
        self.run_log = run_log
        self.process = None
        self.pid = None
        self.port = None
        self.browser_path = None
        self.client = None
        self.argv = None
        self.leftover = False
        self.close_wait_s = 10.0
        self.kill_wait_s = 5.0

    async def launch(self, *, netlog: Path | None = None) -> None:
        if self.mode == 'harness':
            if sys.platform == 'darwin':
                raise RuntimeError('harness_launch_refused_on_the_mac')
            self.argv = harness_argv(self.binary, self.profile)
            self.run_log.parent.mkdir(parents=True, exist_ok=True)
            output = self.run_log.open('ab')
            self.process = subprocess.Popen(self.argv, stdout=output, stderr=output, start_new_session=True)
            output.close()
            self.pid = self.process.pid
        else:
            self.argv = launch_argv(self.binary, self.profile, self.run_log, netlog=netlog)
            subprocess.run(self.argv, check=True, timeout=30)
        active = self.profile / 'DevToolsActivePort'
        deadline = time.monotonic() + DEADLINE_MS['setup'] / 1000
        while not active.exists() or len(active.read_text().splitlines()) < 2:
            if time.monotonic() > deadline or (self.process is not None and self.process.poll() is not None):
                raise RuntimeError('browser_launch_failed')
            await asyncio.sleep(0.05)
        lines = active.read_text().splitlines()
        self.port, self.browser_path = int(lines[0]), lines[1]
        if self.mode != 'harness':
            self.pid = self._find_pid()
        from websockets.asyncio.client import connect
        connection = await connect(f'ws://127.0.0.1:{self.port}{self.browser_path}', max_size=None)
        self.client = CdpClient(connection, name='browser')

    def _find_pid(self) -> int | None:
        for row in process_rows():
            if f'--user-data-dir={self.profile}' in row['command'] and '--type=' not in row['command']:
                return row['pid']
        return None

    def alive(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    async def page_target(self) -> str:
        targets = await self.client.send('Target.getTargets', klass='setup')
        pages = [target for target in targets['targetInfos'] if target['type'] == 'page']
        return pages[0]['targetId']

    async def connect_page(self, target_id: str) -> CdpClient:
        from websockets.asyncio.client import connect
        connection = await connect(f'ws://127.0.0.1:{self.port}/devtools/page/{target_id}', max_size=None)
        return CdpClient(connection, name='page')

    async def close(self) -> None:
        if self.client is not None and not self.client.disconnected:
            try:
                await self.client.send('Browser.close', klass='setup')
            except Exception:  # noqa: BLE001 - the process check below decides.
                pass
            await self.client.close()
        if self.pid is None and self.mode != 'harness':
            for _ in range(5):
                self.pid = self._find_pid()
                if self.pid is not None:
                    break
                await asyncio.sleep(0.2)
        deadline = time.monotonic() + self.close_wait_s
        while self.alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self.alive() and self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except OSError:
                pass
            deadline = time.monotonic() + self.kill_wait_s
            while self.alive() and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
        # V7: never delete the profile while a browser still holds it (the exit sweep handles it).
        if self.alive() or (self.mode != 'harness' and self._find_pid() is not None):
            self.leftover = True
            logger.warning('[web-timing] browser still holds %s; profile kept for the exit sweep', self.profile)
            return
        shutil.rmtree(self.profile, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Fixture process (fixture_backend, control pipe).
# --------------------------------------------------------------------------------------------

class FixtureProcess:
    """The fixture subprocess and its stdin/stdout control pipe."""

    def __init__(self, run_dir: Path, *, harness_pages: bool = False, seed: int | None = None):
        self.run_dir = run_dir
        self.harness_pages = harness_pages
        self.seed = seed
        self.process = None
        self.ready = None
        self.lock = asyncio.Lock()

    async def start(self) -> dict:
        command = [sys.executable, '-m', 'tools.web_timing_fixture', '--run-dir', str(self.run_dir)]
        if self.harness_pages:
            command.append('--harness-pages')
        if self.seed is not None:
            command += ['--seed', str(self.seed)]
        self.process = await asyncio.create_subprocess_exec(
            *command, cwd=str(REPOSITORY), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=(self.run_dir / 'fixture.log').open('ab'))
        line = await asyncio.wait_for(self.process.stdout.readline(), 600)
        self.ready = json.loads(line)
        if self.ready.get('event') != 'ready':
            raise RuntimeError('fixture_not_ready')
        return self.ready

    async def request(self, command: str, deadline_s: float) -> dict:
        async with self.lock:
            self.process.stdin.write((command + '\n').encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), deadline_s)
            return json.loads(line)

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        try:
            await self.request('quit', 10)
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(self.process.wait(), 20)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


async def position_check(fixture: FixtureProcess, ordinal: int) -> bool:
    """Setup command under the setup deadline; a mismatch, error or timeout is a fixture fault (R5-2)."""
    try:
        reply = await fixture.request(f'position_check {ordinal}', DEADLINE_MS['setup'] / 1000)
    except (asyncio.TimeoutError, ValueError, OSError):
        return False
    return reply.get('reply') == 'position_check' and reply.get('ordinal') == ordinal and reply.get('ok') is True


# --------------------------------------------------------------------------------------------
# One run (instrumentation.run_lifecycle).
# --------------------------------------------------------------------------------------------

class RecordStream:
    """Accepted binding records with waiters (no polling)."""

    def __init__(self, record_filter: RecordFilter):
        self.filter = record_filter
        self.changed = asyncio.Event()

    def offer(self, payload: str) -> None:
        try:
            record = json.loads(payload)
        except ValueError:
            return
        record['_received'] = time.monotonic()
        if self.filter.offer(record):
            self.changed.set()

    def find(self, predicate) -> dict | None:
        for record in self.filter.accepted:
            if predicate(record):
                return record
        return None

    def last(self, predicate) -> dict | None:
        for record in reversed(self.filter.accepted):
            if predicate(record):
                return record
        return None

    async def wait(self, predicate, until: float) -> dict:
        """Wait until a matching record arrives or the absolute monotonic deadline passes."""
        while True:
            found = self.find(predicate)
            if found is not None:
                return found
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass


async def dispatch_click(page: CdpClient, point: dict, mobile: bool) -> None:
    """One trusted click (or tap) at a point: the gesture's events are written back to back (pipelined)."""
    x, y = point['x'], point['y']
    if mobile:
        await page.send_pipelined([('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x, 'y': y}]}),
                                   ('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})], klass='input')
        return
    await page.send_pipelined([
        ('Input.dispatchMouseEvent', {'type': 'mouseMoved', 'x': x, 'y': y}),
        ('Input.dispatchMouseEvent', {'type': 'mousePressed', 'x': x, 'y': y, 'button': 'left', 'clickCount': 1}),
        ('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'x': x, 'y': y, 'button': 'left', 'clickCount': 1}),
    ], klass='input')


def centre(rect: dict) -> dict:
    return {'x': rect['x'] + rect['width'] / 2, 'y': rect['y'] + rect['height'] / 2}


@dataclass
class RunSpec:
    """Everything one run needs."""

    ordinal: int
    role: str
    viewport: str
    view: str
    url: str
    origin: str
    token_for_login: str | None
    trace_load: bool = False
    trace_interactions: bool = True
    screencast: bool = True
    capture_id: str | None = None
    harness_steps: list | None = None
    harness: bool = False
    warm_up: bool = False
    primer: bool = False
    trace_contrast: bool = False
    play_rule: dict | None = None
    # R6-3: the chosen frame event (fill-in) when known; None in the smoke, where it is being chosen.
    frame_event: str | None = None
    # Smoke and harness only (smoke_gate.mac_environment): an action after the settle point.
    after_settle: Callable | None = None


class Run:
    """Execute the four phases of one run and gather its facts."""

    def __init__(self, spec: RunSpec, browser: Browser, page: CdpClient, target_id: str, out_dir: Path,
                 fixture: FixtureProcess | None, previous_script: str | None):
        self.spec = spec
        self.browser = browser
        self.page = page
        self.target_id = target_id
        self.out_dir = out_dir
        self.fixture = fixture
        self.previous_script = previous_script
        self.script_identifier = None
        self.token = secrets.token_hex(16)
        self.filter = RecordFilter(self.token)
        self.records = RecordStream(self.filter)
        self.facts = RunFacts()
        self.network = NetworkLedger(spec.origin)
        self.trace_events = []
        self.trace_complete = asyncio.Event()
        self.frames = []
        self.window_states = []
        self.window_reads = []
        self.window_tasks = []
        self.exceptions = []
        self.detached = []
        self.timings = {}
        self.steps = []
        self.activation_polls = None
        self.leave_ok = False
        self.browser_version = None
        self.flush_sent = None
        self.play_click_wall = None
        self.play_click_frame = None
        # One page client serves a cold run and its warm run: each run reports only its own commands.
        self.page_log_start = len(page.log)
        self.browser_log_start = len(browser.client.log)

    def _handlers(self) -> list:
        page, browser = self.page, self.browser.client
        handlers = [
            (page, 'Runtime.bindingCalled', lambda params: self.records.offer(params.get('payload', ''))
             if params.get('name') == 'kzaProbeRecord' else None),
            (page, 'Page.frameNavigated', lambda params: self.filter.frame_navigated(params.get('frame', {}))),
            (page, 'Network.requestWillBeSent', self.network.will_be_sent),
            (page, 'Network.requestServedFromCache', self.network.served_from_cache),
            (page, 'Network.responseReceived', self.network.response),
            (page, 'Page.screencastFrame', self._frame),
            (page, 'Runtime.exceptionThrown', self._exception),
            (page, 'Inspector.detached', self._detached),
            (browser, 'Tracing.dataCollected', lambda params: self.trace_events.extend(params.get('value', []))),
            (browser, 'Tracing.tracingComplete', lambda params: self.trace_complete.set()),
        ]
        for client, method, handler in handlers:
            client.on(method, handler)
        return handlers

    def _unhook(self, handlers: list) -> None:
        for client, method, handler in handlers:
            if handler in client.handlers.get(method, []):
                client.handlers[method].remove(handler)

    async def _frame(self, params: dict) -> None:
        stamp = (params.get('metadata') or {}).get('timestamp')
        self.frames.append({'timestamp': stamp, 'received': time.monotonic()})
        # R6-6 (smoke_gate.preview_ready_state): the first screencast frame at or after the play click is kept.
        if self.play_click_wall is not None and self.play_click_frame is None and stamp and stamp >= self.play_click_wall \
                and params.get('data'):
            self.play_click_frame = {'timestamp': stamp, 'jpeg_base64_sha256':
                                     hashlib.sha256(params['data'].encode()).hexdigest()}
            try:
                path = self.out_dir / f'play-click-{self.spec.ordinal:05d}.jpg'
                path.write_bytes(base64.b64decode(params['data']))
                self.play_click_frame['path'] = str(path)
            except (OSError, ValueError):
                pass
        try:
            await self.page.send('Page.screencastFrameAck', {'sessionId': params['sessionId']}, klass='collect')
        except Exception:  # noqa: BLE001 - an ack failure stops the stream; the cross-check shows it.
            pass

    def _exception(self, params: dict) -> None:
        """V11: JavaScript exceptions are recorded per run with the phase they arrived in (reported, see
        exclusion_rules.failure_scope)."""
        details = params.get('exceptionDetails') or {}
        exception = details.get('exception') or {}
        self.exceptions.append({'phase': self.page.phase, 'received': time.monotonic(),
                                'text': (details.get('text') or '')[:200],
                                'description': (exception.get('description') or '')[:300],
                                'url': details.get('url'), 'line': details.get('lineNumber')})

    def _detached(self, params: dict) -> None:
        """V11: Inspector.detached is a lost CDP session (invalid run, like a websocket disconnect)."""
        self.detached.append({'phase': self.page.phase, 'reason': params.get('reason')})
        self.facts.disconnected = True

    def _phase(self, phase: str) -> None:
        self.page.phase = phase
        self.browser.client.phase = phase

    def _timeout(self, error: CdpTimeout | None, phase: str, klass: str, method: str) -> None:
        self.facts.timeouts.append(Timeout(phase, klass, method if error is None else error.method))

    def _harness_error(self, phase: str, error: BaseException) -> str:
        """V11: an unclassified error becomes a harness fault of its phase (invalid run), never a batch abort."""
        method = getattr(error, 'method', None) or ''
        return f'{phase}_harness_error:{type(error).__name__}:{method or str(error)[:120]}'

    def _sample_window(self, point: str) -> None:
        """V4: Browser.getWindowForTarget at a timed point, on the browser endpoint and not awaited by the timed
        sequence (no renderer work can delay it); an unanswered read is a harness fault, never a failure."""
        phase = self.browser.client.phase

        async def read():
            entry = {'point': point, 'state': None, 'sent': time.monotonic()}
            try:
                result = await self.browser.client.send('Browser.getWindowForTarget', {'targetId': self.target_id},
                                                        klass='window_state', phase=phase)
                entry['state'] = (result.get('bounds') or {}).get('windowState')
            except (CdpTimeout, CdpError, CdpDisconnected) as error:
                entry['error'] = type(error).__name__
            self.window_reads.append(entry)
        try:
            self.window_tasks.append(asyncio.get_running_loop().create_task(read()))
        except RuntimeError:
            pass

    async def _await_window_reads(self) -> None:
        tasks, self.window_tasks = self.window_tasks, []
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def execute(self) -> dict:
        handlers = self._handlers()
        try:
            await self._run()
        finally:
            self._unhook(handlers)
        try:
            return self.result()
        except Exception as error:  # noqa: BLE001 - V11: never abort the batch on an analysis error.
            logger.exception('[web-timing] run %s result failed', self.spec.ordinal)
            return {'ordinal': self.spec.ordinal, 'role': self.spec.role, 'viewport': self.spec.viewport,
                    'view': self.spec.view, 'outcome': Outcome.INVALID,
                    'reason': self._harness_error('result', error), 'navigate_sent': self.facts.navigate_sent,
                    'metrics': {}, 'facts': asdict(self.facts), 'steps': self.steps,
                    'browser_version': self.browser_version, 'script_identifier': self.script_identifier,
                    'records': {'accepted': self.filter.accepted, 'rejected': self.filter.rejected},
                    'post_window_violations': post_window_violations(self.page.log[self.page_log_start:]),
                    'commands': list(self.page.log[self.page_log_start:]) + list(self.browser.client.log[self.browser_log_start:])}

    async def _run(self) -> None:
        # Phase 1: setup on about:blank.
        self._phase('setup')
        try:
            await self._setup()
        except CdpTimeout as error:
            self._timeout(error, 'setup', error.klass, error.method)
            await self._post_failure_probe()
            return
        except ActivationFailed as error:
            self.activation_polls = error.polls
            self.facts.setup_faults.append('activation_failed')
            return
        except SetupFault as error:
            self.facts.setup_faults.append(str(error))
            return
        except CdpDisconnected:
            self.facts.disconnected = True
            return
        except Exception as error:  # noqa: BLE001 - V11: CdpError, login failure and the rest are setup faults.
            self.facts.setup_faults.append(self._harness_error('setup', error))
            return
        # Phase 2: timed window.
        self._phase('timed')
        try:
            await self._timed()
        except CdpTimeout as error:
            self._timeout(error, 'timed', error.klass, error.method)
            await self._post_failure_probe()
        except CdpDisconnected:
            self.facts.disconnected = True
            self.facts.crashed = not self.browser.alive()
            return
        except Exception as error:  # noqa: BLE001 - V11: a CDP error reply in the window is a harness fault.
            self.facts.environment_faults.append(self._harness_error('timed', error))
        # Phase 3: collection, browser-side only.
        self._phase('collect')
        try:
            await self._collect()
        except CdpTimeout as error:
            self._timeout(error, 'collect', error.klass, error.method)
            self._collect_timeout = error
            await self._post_failure_probe()
        except CdpDisconnected:
            self.facts.disconnected = True
            self.facts.crashed = not self.browser.alive()
            return
        except Exception as error:  # noqa: BLE001
            self.facts.environment_faults.append(self._harness_error('collect', error))
        # Phase 4: leave step.
        self._phase('leave')
        await self._leave()

    async def _post_failure_probe(self) -> None:
        """Log Browser.getVersion on the browser endpoint and the process state with every timeout."""
        self.timings.setdefault('timeout_probes', []).append({'alive': self.browser.alive()})
        try:
            version = await self.browser.client.send('Browser.getVersion', klass='version')
            self.timings['timeout_probes'][-1]['version'] = version.get('product')
        except (CdpTimeout, CdpDisconnected, CdpError):
            self.timings['timeout_probes'][-1]['version'] = None
        if not self.browser.alive():
            self.facts.crashed = True
        if self.browser.client.disconnected or self.page.disconnected:
            self.facts.disconnected = True

    async def _setup(self) -> None:
        spec = self.spec
        config = injection_config(token=self.token, origin=spec.origin, view=spec.view, capture_id=spec.capture_id,
                                  harness_steps=spec.harness_steps, harness=spec.harness, play_rule=spec.play_rule)
        self.config = config
        source = injected_source(config)
        for method, params in setup_commands(role=spec.role, viewport=spec.viewport, script_source=source,
                                             previous_script=self.previous_script):
            result = await self.page.send(method, params, klass='setup')
            if method == 'Page.addScriptToEvaluateOnNewDocument':
                self.script_identifier = result.get('identifier')
        if spec.role == 'cold' and spec.token_for_login:
            from tests.frontend.browser_support import install_login_session
            try:
                await install_login_session(SetupAdapter(self.page), spec.origin, spec.token_for_login)
            except RuntimeError as error:
                raise SetupFault('login_failed:' + str(error)[:80]) from None
        polls_started = time.monotonic()
        self.activation_polls = await activation_poll(self.page, self.browser.client, self.target_id,
                                                      DEADLINE_MS['setup'] / 1000)
        self.timings['activation_s'] = time.monotonic() - polls_started
        if spec.trace_load:
            await self._start_tracing('setup')
        if self.fixture is not None:
            if not (await self.fixture.request('health', DEADLINE_MS['setup'] / 1000)).get('ok'):
                raise SetupFault('fixture_not_ready')
            if spec.capture_id is not None and not await position_check(self.fixture, spec.ordinal):
                raise SetupFault('position_check_failed')

    async def _start_tracing(self, klass: str) -> None:
        await self.browser.client.send('Tracing.start', {'traceConfig': {'includedCategories': list(TRACE_CATEGORIES)},
                                                          'transferMode': 'ReportEvents'}, klass=klass)
        self.tracing = True

    async def _timed(self) -> None:
        spec = self.spec
        self.facts.navigate_sent = True
        self._sample_window('navigation')
        navigate_sent = time.monotonic()
        result = await self.page.send('Page.navigate', {'url': spec.url}, klass='navigate')
        if result.get('errorText'):
            self.facts.product_failures.append('navigation_error:' + result['errorText'])
            return
        # The commit event can precede the reply; re-scan the received events once (V14b buffers records).
        for event in self.page.events:
            if event['method'] == 'Page.frameNavigated':
                self.filter.frame_navigated(event['params'].get('frame', {}))
        self.filter.navigation_sent(result.get('loaderId'))
        load_deadline = navigate_sent + DEADLINE_MS['load_wait'] / 1000
        try:
            settled = await self.records.wait(lambda record: record['type'] == 'settled', load_deadline)
        except asyncio.TimeoutError:
            self.facts.product_failures.append('load_not_settled')
            await self._flush()
            return
        self._sample_window('load_end')
        self.timings['settled'] = settled['T']
        if spec.after_settle is not None:
            self.timings['after_settle'] = await spec.after_settle()
        steps = self.config['steps']
        if steps:
            if spec.trace_interactions and not getattr(self, 'tracing', False):
                await self._start_tracing('interaction_trace')
            if spec.screencast:
                await self.page.send('Page.startScreencast', {'format': 'jpeg', 'quality': 40, 'everyNthFrame': 1},
                                     klass='interaction_trace')
                self.screencast = True
            await self._interactions(steps)
        await self._flush()

    def _outside_viewport(self, step: dict, target: dict) -> bool:
        """V3: a dispatch point outside the layout viewport is never clicked (harness fault, invalid run)."""
        if target.get('in_viewport') is False:
            self.facts.environment_faults.append(f'target_outside_viewport:{step["step"]}')
            self.steps.append({'step': step['step'], 'status': 'target_outside_viewport', 'target': target})
            return True
        return False

    async def _interactions(self, steps: list) -> None:
        spec = self.spec
        mobile = VIEWPORTS[spec.viewport]['mobile']
        previous_end = time.monotonic()
        for index, step in enumerate(steps):
            if step.get('play_gesture'):
                attached = self.records.find(lambda record: record['type'] == 'control_attached')
                if attached is None:
                    self.facts.product_failures.append('preview_failed_open:no_control_attached')
                    self.steps.append({'step': step['step'], 'status': 'not_dispatched'})
                    return
                ready = play_ready(attached, None)
                if ready is None:
                    # V2: a native control attached before its metadata is operable at loadedmetadata.
                    try:
                        ready = await self.records.wait(lambda record: record['type'] == 'control_ready',
                                                        self.preview_until)
                    except asyncio.TimeoutError:
                        self.facts.product_failures.append('preview_failed_open:control_not_ready')
                        self.steps.append({'step': step['step'], 'status': 'not_dispatched'})
                        return
                if self._outside_viewport(step, ready['target']):
                    return
                log_start = len(self.page.log)
                self.play_click_wall = time.time()
                await dispatch_click(self.page, ready['target']['point'], mobile)
                self._sample_window(f'interaction_start:{index}')
                sent = next((entry['sent'] for entry in self.page.log[log_start:]
                             if entry['method'].startswith('Input.')), None)
                # F6: probe-side wait from receiving the ready record to sending the first input event.
                delay = (sent - ready['_received']) * 1000 if sent is not None and '_received' in ready else None
                self.steps.append({'step': step['step'], 'status': 'dispatched', 'target': ready['target'],
                                   'ready_record': ready['type'], 'probe_dispatch_delay_ms': delay})
                try:
                    await self.records.wait(lambda record: record['type'] == 'media' and record['event'] == 'playing',
                                            self.preview_until)
                except asyncio.TimeoutError:
                    self.facts.product_failures.append('preview_failed_open:no_playing')
                    return
                self._sample_window(f'interaction_end:{index}')
                previous_end = time.monotonic()
                continue
            target_until = previous_end + DEADLINE_MS['wait_other'] / 1000
            try:
                await self.records.wait(lambda record, i=index: record['type'] == 'target_ready'
                                        and record['index'] == i, target_until)
            except asyncio.TimeoutError:
                self.facts.product_failures.append(f'target_absent:{step["step"]}')
                self.steps.append({'step': step['step'], 'status': 'target_absent'})
                return
            spacing = previous_end + INPUT_SPACING_S - time.monotonic()
            if spacing > 0:
                await asyncio.sleep(spacing)
            ready = self.records.last(lambda record, i=index: record['type'] == 'target_ready' and record['index'] == i)
            if self._outside_viewport(step, ready):
                return
            dispatched = time.monotonic()
            await dispatch_click(self.page, centre(ready['rect']), mobile)
            self._sample_window(f'interaction_start:{index}')
            self.steps.append({'step': step['step'], 'status': 'dispatched', 'rect': ready['rect']})
            wait_until = dispatched + DEADLINE_MS[step['wait']] / 1000
            if step['end']['kind'] == 'control_attached':
                self.preview_until = wait_until
                predicate = lambda record: record['type'] == 'control_attached'
            else:
                predicate = lambda record, i=index: record['type'] == 'step_next_frame' and record['index'] == i
            try:
                await self.records.wait(predicate, wait_until)
            except asyncio.TimeoutError:
                reason = {'wait_feedback': 'action_feedback_missing', 'wait_preview': 'preview_failed_open'}.get(
                    step['wait'], 'interaction_end_missing')
                self.facts.product_failures.append(f'{reason}:{step["step"]}')
                return
            self._sample_window(f'interaction_end:{index}')
            if step.get('c11') == 'action_feedback':
                # R6-3: the timed window stays open until the injected script's feedback_window record (the later
                # of 100 ms and 6 frames after the mutation), bounded by the step's own deadline. A missing record
                # is noted only: the flush that follows has its own deadline (cdp_deadlines.rule (a)).
                try:
                    await self.records.wait(lambda record, i=index: record['type'] == 'feedback_window'
                                            and record['index'] == i, wait_until)
                    self.steps[-1]['feedback_window'] = 'closed'
                except asyncio.TimeoutError:
                    self.steps[-1]['feedback_window'] = 'missing'
            previous_end = time.monotonic()

    async def _flush(self) -> None:
        expression = f'window[{json.dumps(self.config["flush"])}]()'
        self.flush_sent = len(self.page.log)
        await self.page.send('Runtime.evaluate', {'expression': expression, 'awaitPromise': True, 'returnByValue': True},
                             klass='evaluate')
        self.page.log[self.flush_sent]['flush'] = True
        self._sample_window('flush')

    async def _collect(self) -> None:
        await self._await_window_reads()
        for method, klass, side in post_window_commands(getattr(self, 'tracing', False),
                                                        getattr(self, 'screencast', False)):
            client = self.browser.client if side == 'browser' else self.page
            params = {'targetId': self.target_id} if method == 'Browser.getWindowForTarget' else {}
            started = time.monotonic()
            result = await client.send(method, params, klass=klass)
            if method == 'Tracing.end':
                deadline = started + DEADLINE_MS['collect'] / 1000
                try:
                    await asyncio.wait_for(self.trace_complete.wait(), max(0.0, deadline - time.monotonic()))
                except asyncio.TimeoutError:
                    # V6: the completion wait carries the Tracing.end send time and its absolute deadline.
                    self.browser.client.log.append({'method': 'Tracing.tracingComplete', 'class': 'collect',
                                                    'phase': 'collect', 'timeout': True, 'sent': started,
                                                    'deadline': deadline, 'connection': 'browser'})
                    raise CdpTimeout('Tracing.end', 'collect', 'collect') from None
            elif method == 'Browser.getWindowForTarget':
                self.window_states.append(result.get('bounds', {}).get('windowState'))
            elif method == 'Browser.getVersion':
                self.browser_version = result.get('product')

    async def _leave(self) -> None:
        sent = time.monotonic()
        self.timings['leave_sent'] = sent
        try:
            arrived = asyncio.Event()
            def navigated(params):
                frame = params.get('frame', {})
                if frame.get('parentId') is None and frame.get('url', '').startswith('about:blank'):
                    arrived.set()
            self.page.on('Page.frameNavigated', navigated)
            try:
                await self.page.send('Page.navigate', {'url': 'about:blank'}, klass='leave')
                await asyncio.wait_for(arrived.wait(), max(0.0, sent + DEADLINE_MS['leave'] / 1000 - time.monotonic()))
            finally:
                self.page.handlers['Page.frameNavigated'].remove(navigated)
            self.leave_ok = True
        except (CdpTimeout, asyncio.TimeoutError):
            self._timeout(None, 'leave', 'leave', 'Page.navigate(about:blank)')
            await self._post_failure_probe()
        except CdpDisconnected:
            self.facts.disconnected = True
            self.facts.crashed = not self.browser.alive()
        except Exception as error:  # noqa: BLE001 - V11: a CDP error reply of the leave step is a harness fault.
            self.facts.environment_faults.append(self._harness_error('leave', error))
        finally:
            self.facts.leave_ok = self.leave_ok

    def result(self) -> dict:
        """Raw run data, metrics and classification."""
        spec = self.spec
        accepted = self.filter.accepted
        typed = defaultdict(list)
        for record in accepted:
            typed[record['type']].append(record)
        entries = [record['entry'] for record in typed['perf']]
        interactions = interactions_from_inputs(typed['input'])
        steps = self.config['steps'] if hasattr(self, 'config') else []
        # R5-4, V1: the core script completed when every step was dispatched; a later product failure (missing
        # feedback, no 'playing') fails only the metric it names.
        completed = bool(steps) and [s.get('status') for s in self.steps] == ['dispatched'] * len(steps)
        document_start = typed['document_start'][0] if typed['document_start'] else {}
        http_errors = self.network.errors()
        document_errors = [item for item in http_errors if item.get('type') == 'Document']
        if document_errors and not any(f.startswith('navigation_http_error') for f in self.facts.product_failures):
            self.facts.product_failures.insert(0, f'navigation_http_error:{document_errors[0]["status"]}')
        # Environment faults of the timed phase (V4: records up to the flush sample, F1 focus rule).
        flush = next((record for record in reversed(typed['focus']) if record.get('point') == 'flush'), None)
        flush_time = flush.get('time') if flush else None
        if self.facts.navigate_sent and document_start:
            faults = visibility_faults(document_start, timed_records(typed['focus'], flush_time),
                                       timed_records(typed['visibility_event'], flush_time), self.window_reads)
            viewport_faults, flags = viewport_check(spec.viewport, document_start, typed['viewport'])
            self.facts.environment_faults += [fault for fault in faults + viewport_faults
                                              if fault not in self.facts.environment_faults]
        else:
            flags = []
        if self.facts.timeouts and any(t.phase == 'collect' for t in self.facts.timeouts):
            self.facts.renderer_busy_across_deadline = self._renderer_busy()
        metrics = {}
        inp_interactions = native_play_interactions(typed, interactions, steps, flush_time)
        native_step = next((step for step in self.steps if step.get('ready_record') and
                            (step.get('target') or {}).get('kind') == 'audio'), None)
        if native_step is not None and (native_step.get('probe_dispatch_delay_ms') or 0) > PREVIEW_GAP_MAX_MS:
            self.facts.environment_faults.append('preview_dispatch_delay_over_16ms')
        if self.facts.navigate_sent and document_start:
            first_input = interactions[0]['first'] if interactions else None
            metrics['lcp'] = lcp(entries, first_input)
            metrics['cls'] = cls(entries, completed or not steps)
            if steps:
                mapping = map_inp(entries, inp_interactions[:len(steps)])
                metrics['inp'] = run_inp(mapping, completed)
                metrics['inp_mapping'] = mapping
            else:
                metrics['inp'] = LOAD_ONLY_INP
            metrics.update(self._c11(typed, inp_interactions, steps))
            if metrics.get('preview', {}).get('gap_invalid'):
                self.facts.environment_faults.append('preview_gap_over_16ms')
            if feedback_frame_missing(metrics.get('action_feedback'), spec.frame_event,
                                      bool(typed['feedback_window']), self._trace_intact()):
                self.facts.environment_faults.append('feedback_frame_absent_from_trace')
        outcome, reason = classify(self.facts)
        page_log = self.page.log[self.page_log_start:]
        log = page_log + self.browser.client.log[self.browser_log_start:]
        result = {
            'ordinal': spec.ordinal, 'role': spec.role, 'viewport': spec.viewport, 'view': spec.view,
            'warm_up': spec.warm_up, 'primer': spec.primer, 'trace_contrast': spec.trace_contrast,
            'outcome': outcome, 'reason': reason, 'navigate_sent': self.facts.navigate_sent,
            'facts': asdict(self.facts), 'metrics': metrics, 'flags': flags, 'steps': self.steps,
            'browser_version': self.browser_version, 'window_states': self.window_states,
            'window_reads': self.window_reads,
            'activation_polls': self.activation_polls, 'timings': self.timings,
            'cache_sources': self.network.sources(), 'post_window_violations': post_window_violations(page_log),
            'http_errors': http_errors, 'exceptions': self.exceptions, 'inspector_detached': self.detached,
            'timed_exceptions': sum(1 for item in self.exceptions if item['phase'] == 'timed'),
            'commands': [{key: value for key, value in entry.items()} for entry in log],
            'records': {'accepted': accepted, 'rejected': self.filter.rejected},
            'document_start': document_start, 'screencast_frames': self.frames, 'trace_events': len(self.trace_events),
            'trace_candidates_present': sorted(name for name, values in frame_candidates(self.trace_events).items()
                                               if values),
            'script_identifier': self.script_identifier,
            'interaction_traced': bool(steps) and spec.trace_interactions,
            'play_rule': self.config.get('play_rule') if hasattr(self, 'config') else None,
            'frame_event': spec.frame_event, 'play_click_frame': self.play_click_frame,
        }
        result['run_wide_failure'] = run_wide_failure(result)
        return result

    def _trace_intact(self) -> bool:
        """R6-3: the trace completed (tracingComplete, no collect timeout) and holds the probe's marks."""
        if not self.trace_events or not self.trace_complete.is_set():
            return False
        if any(timeout.phase == 'collect' for timeout in self.facts.timeouts):
            return False
        page_marks = [record for record in self.filter.accepted if record['type'] == 'mark']
        return trace_offset(trace_marks(self.trace_events), page_marks) is not None

    def _renderer_busy(self) -> bool | None:
        timeout = next((t for t in self.facts.timeouts if t.phase == 'collect'), None)
        if timeout is None or not self.trace_events:
            return None
        command = next((entry for entry in self.page.log[self.page_log_start:]
                        + self.browser.client.log[self.browser_log_start:]
                        if entry.get('timeout') and entry.get('phase') == 'collect'
                        and 'sent' in entry and 'deadline' in entry), None)
        if command is None:
            return None
        marks = trace_marks(self.trace_events)
        page_marks = [record for record in self.filter.accepted if record['type'] == 'mark']
        offset = trace_offset(marks, page_marks)
        anchor = monotonic_anchor(self.filter.accepted)
        if offset is None or anchor is None:
            return None
        to_trace = lambda mono: (mono - anchor) * 1000 + offset
        return renderer_busy_across(self.trace_events, to_trace(command['sent']), to_trace(command['deadline']))

    def _c11(self, typed: dict, interactions: list, steps: list) -> dict:
        trace = None
        if self.trace_events:
            offset = trace_offset(trace_marks(self.trace_events), typed['mark'])
            trace = {'offset': offset, 'candidates': frame_candidates(self.trace_events)} if offset is not None else None
        return c11_metrics(typed, interactions, steps, trace, self._screencast_after)

    def _screencast_after(self, page_time: float | None) -> float | None:
        if page_time is None or not self.frames or self.filter.time_origin is None:
            return None
        wall = (self.filter.time_origin + page_time) / 1000
        later = sorted(frame['timestamp'] for frame in self.frames if frame['timestamp'] and frame['timestamp'] >= wall)
        return (later[0] * 1000 - self.filter.time_origin) if later else None


def monotonic_anchor(records: list) -> float | None:
    """Monotonic seconds at page time 0, from the record with the smallest delivery delay."""
    pairs = [record['_received'] - record['t'] / 1000 for record in records if '_received' in record and 't' in record]
    return min(pairs) if pairs else None


def c11_metrics(typed: dict, interactions: list, steps: list, trace: dict | None,
                screencast_after: Callable[[float | None], float | None] = lambda _: None) -> dict:
    """C11 clocks from accepted records: starts are capture-listener timeStamps, never Event Timing startTime."""
    result = {}
    names = [step['step'] for step in steps]
    ends = {record['index']: record for record in typed.get('step_end', [])}
    frames = {record['index']: record for record in typed.get('step_frame', [])}
    for index, step in enumerate(steps):
        if step.get('c11') == 'action_feedback':
            start = interactions[index]['first'] if index < len(interactions) else None
            end = ends.get(index)
            mutation = end['time'] if end else None
            raf = frames[index]['raf'] if index in frames else None
            # V5: the first-rAF lower bound is never earlier than the end mutation.
            if raf is not None and mutation is not None:
                raf = max(raf, mutation)
            record = {'start': start, 'mutation': mutation, 'raf': raf,
                      'frame_time': frames[index].get('frame_time') if index in frames else None, 'presented': {}}
            if trace and end:
                for name, values in trace['candidates'].items():
                    presented = presented_after(values, end['time'] + trace['offset'])
                    record['presented'][name] = None if presented is None else presented - trace['offset']
            record['screencast'] = screencast_after(end['time'] if end else None)
            result['action_feedback'] = record
            result['action_feedback_raf'] = action_feedback(start, record['raf'])
        if step.get('c11') == 'preview':
            start = interactions[index]['first'] if index < len(interactions) else None
            attached = typed['control_attached'][0] if typed.get('control_attached') else None
            ready = play_ready(attached, typed['control_ready'][0] if typed.get('control_ready') else None)
            play_index = names.index('reproducir') if 'reproducir' in names else None
            gesture = (interactions[play_index]['first']
                       if play_index is not None and play_index < len(interactions) else None)
            media = typed.get('media', [])
            playing = next((record['timeStamp'] for record in media if record['event'] == 'playing'), None)
            native = play_index is not None and play_index < len(interactions) \
                and interactions[play_index].get('synthetic')
            if native:
                # F6: the native control's input events never reach page listeners, so the gesture time is not
                # observable: the primary value is (playing - start), with no subtraction (one definition for
                # the native target kind; the probe-side dispatch delay is checked instead of the gap).
                value = preview(start, ready['time'] if ready else None, ready['time'] if ready else None, playing)
                value.update(gap=None, gap_invalid=False, definition='native_unsubtracted')
            else:
                value = preview(start, ready['time'] if ready else None, gesture, playing)
                value['definition'] = 'subtracted'
            review = next((entry for entry in (record['entry'] for record in typed.get('perf', []))
                           if entry.get('entryType') == 'resource' and '/review' in entry.get('name', '')), None)
            value['secondary'] = {
                'loadedmetadata': next((r['timeStamp'] for r in media if r['event'] == 'loadedmetadata'), None),
                'canplay': next((r['timeStamp'] for r in media if r['event'] == 'canplay'), None),
                'ready_states': [{'event': r['event'], 'readyState': r['readyState']} for r in media],
                'attach_ready_state': attached['readyState'] if attached else None,
                'control_attached': attached['time'] if attached else None,
                'control_ready': ready['time'] if ready else None,
                'ready_record': ready['type'] if ready else None,
                'metadata_wait': (ready['time'] - attached['time']) if ready and attached else None,
                'play_target': ready['target'] if ready else (attached['target'] if attached else None),
                'network': (review['responseEnd'] - review['requestStart']) if review else None,
            }
            result['preview'] = value
    return result


class SetupFault(Exception):
    """A setup-phase harness or fixture fault (invalid run)."""


def feedback_frame_missing(record: dict | None, frame_event: str | None, window_closed: bool,
                           trace_intact: bool) -> bool:
    """R6-3 (exclusion_rules.invalid_run): the feedback mutation and its rAF are present, the post-feedback window
    closed and the trace is intact, yet the chosen frame event has no frame after the mutation: a harness fault
    (invalid run), never a product failure. Not applied while the frame event is unknown (smoke)."""
    if frame_event is None or not record or not window_closed or not trace_intact:
        return False
    if record.get('mutation') is None or record.get('raf') is None:
        return False
    return (record.get('presented') or {}).get(frame_event) is None


def c11_feedback_value(run: dict, frame_event: str | None) -> float:
    """Final C11 action feedback with the chosen frame event (fill-in from smoke_gate.frame_event)."""
    if run_wide_failure(run):
        return INF
    record = run['metrics'].get('action_feedback')
    if not record or frame_event is None:
        return INF
    presented = (record.get('presented') or {}).get(frame_event)
    return action_feedback(record.get('start'), presented)


# --------------------------------------------------------------------------------------------
# Batch runner (runs.order, quiet_machine, exclusion_rules.invalid_batch). Mac baseline and smoke batches.
# --------------------------------------------------------------------------------------------

@dataclass
class BatchEnvironment:
    """Side effects of a batch, injectable for the unit checks."""

    quiet_check: Callable
    loadavg: Callable
    sleep: Callable
    start_fixture: Callable
    stop_fixture: Callable
    run_slot: Callable
    close_browser: Callable
    caffeinate: Callable
    served_state: Callable
    machine_state: Callable = lambda: {'ac_power': True, 'display_asleep': False, 'locked': False}
    assertion_held: Callable = lambda caffeinate: True
    sweep: Callable = lambda: []
    events: list = field(default_factory=list)


async def maybe_await(value):
    return await value if asyncio.iscoroutine(value) else value


def machine_faults(state: dict, *, smoke: bool) -> list:
    """V8 (quiet_machine.mac): AC power, display awake and screen unlocked at batch start and end. In the smoke the
    lock read is not yet chosen and the display read is being confirmed, so an unreadable value is recorded only."""
    faults = []
    if state.get('ac_power') is not True:
        faults.append('not_on_ac_power')
    if state.get('display_asleep') is True or (state.get('display_asleep') is None and not smoke):
        faults.append('display_asleep')
    if state.get('locked') is True or (state.get('locked') is None and not smoke):
        faults.append('screen_locked')
    # R6-8: the default output must be the built-in one; the key is absent only in callers that predate it.
    if 'audio_output' in state:
        output = state.get('audio_output')
        if (output is None and not smoke) or (output is not None and output.get('builtin') is not True):
            faults.append('audio_output_not_builtin')
    return faults


async def run_batch(plan: dict, environment: BatchEnvironment, *, idle: float | None, post_batch: float | None,
                    browser_version: Callable[[], str | None] = lambda: None, end_record_only: bool = False,
                    warm_rule: bool = True, smoke: bool = False) -> dict:
    """One batch: start checks before the fixture and Chrome, runs with appended replacements, end checks 60 s
    after Chrome and the fixture exit (R4-11, R5-7). Never reads a load average while the probe's Chrome runs.
    V7: caffeinate and the fixture start inside try/finally, leftover probe browsers are swept at start and exit."""
    events = environment.events
    events.append('sweep_start')
    swept_start = environment.sweep()
    events.append('quiet_start')
    start_quiet = await environment.quiet_check()
    events.append('load_start')
    start_load = environment.loadavg()
    start_machine = environment.machine_state()
    start_faults = machine_faults(start_machine, smoke=smoke)
    batch = {'cell': plan['cell'], 'attempt': plan['attempt'], 'seed': plan['seed'], 'planned': plan['planned'],
             'runs': [], 'start': {'quiet': start_quiet, 'load': start_load, 'idle': idle, 'machine': start_machine,
                                   'machine_faults': start_faults, 'swept': swept_start},
             'quiet_start_ok': bool(start_quiet.get('ok')) and load_reading_ok(start_load, idle) and not start_faults,
             'started_at': time.time(), 'warm_rule_active': warm_rule}
    if not batch['quiet_start_ok']:
        batch.update(discarded=True, reasons=['quiet_machine'], ended_at=time.time())
        return batch
    caffeinate = None
    fixture = None
    fault = None
    cold_by_pair = {}
    try:
        caffeinate = environment.caffeinate()
        batch['caffeinate_assertion_start'] = bool(await maybe_await(environment.assertion_held(caffeinate)))
        events.append('fixture_start')
        try:
            fixture = await environment.start_fixture()
        except Exception as error:  # noqa: BLE001 - a fixture that does not start is a fixture fault.
            fault = f'fixture_fault:{type(error).__name__}:{str(error)[:120]}'
        if fixture is not None:
            batch['fixture'] = {key: fixture.get(key) for key in ('digest', 'started_at', 'seed')}
            batch['served'] = environment.served_state()
            index = 0
            while index < len(plan['slots']):
                slot = plan['slots'][index]
                index += 1
                if slot['role'] == 'warm':
                    cold = cold_by_pair.get((slot.get('pair'), slot.get('replaces'), 'cold'))
                    if cold is None or warm_slot_invalid(cold):
                        result = {'ordinal': slot['ordinal'], 'role': 'warm', 'outcome': Outcome.INVALID,
                                  'reason': 'warm_slot_without_preceding_load', 'pair': slot.get('pair')}
                    else:
                        events.append(f'run:{slot["ordinal"]}')
                        result = await environment.run_slot(slot)
                    await environment.close_browser()
                else:
                    events.append(f'run:{slot["ordinal"]}')
                    result = await environment.run_slot(slot)
                    cold_by_pair[(slot.get('pair'), slot.get('replaces'), 'cold')] = result
                    if slot.get('warm_up'):
                        await environment.close_browser()
                result.update({key: slot.get(key) for key in ('pair', 'trace_contrast', 'untraced_inp', 'warm_up',
                                                              'primer', 'replaces')})
                result['role'] = slot['role']
                batch['runs'].append(result)
                if result['outcome'] == Outcome.INVALID:
                    replacement_slots(plan, slot)
            batch['caffeinate_assertion_end'] = bool(await maybe_await(environment.assertion_held(caffeinate)))
    finally:
        await environment.close_browser()
        events.append('fixture_stop')
        await environment.stop_fixture()
        if caffeinate is not None:
            caffeinate.terminate()
        events.append('sweep_end')
        batch['swept_end'] = environment.sweep()
    if fault is not None:
        batch.update(discarded=True, reasons=['fixture_fault'], fault=fault, ended_at=time.time())
        return batch
    batch['elapsed_since_fixture_s'] = time.time() - (fixture.get('started_at') or time.time())
    events.append('wait_60s')
    await environment.sleep(60)
    events.append('load_end')
    end_load = environment.loadavg()
    end_quiet = await environment.quiet_check()
    events.append('quiet_end')
    end_machine = environment.machine_state()
    end_faults = machine_faults(end_machine, smoke=smoke)
    batch['end'] = {'quiet': end_quiet, 'load': end_load, 'post_batch': post_batch, 'machine': end_machine,
                    'machine_faults': end_faults}
    # In the smoke, post_batch is being measured (smoke_gate.quiet_baseline): the end reading is recorded only.
    batch['quiet_end_ok'] = bool(end_quiet.get('ok')) and not end_faults and (
        end_record_only or load_reading_ok(end_load, post_batch))
    after = environment.served_state()
    batch['tree_dirty'] = batch['served'].get('dirty') or after.get('dirty')
    batch['commit_changed'] = after.get('commit') != batch['served'].get('commit')
    batch['browser_version'] = browser_version() or next((run.get('browser_version') for run in batch['runs']
                                                          if run.get('browser_version')), None)
    batch['reasons'] = invalid_batch_reasons(batch)
    batch['discarded'] = bool(batch['reasons'])
    batch['ended_at'] = time.time()
    return batch


def slot_spec(slot: dict, plan: dict, ready: dict, positions: dict, overhead_consequences: dict | None = None, *,
              play_rule: dict | None = None, frame_event: str | None = None) -> RunSpec:
    """Build the run spec of one slot (preview sample by ordinal, tracing per slot kind)."""
    view, viewport = plan['cell']['view'], plan['cell']['viewport']
    capture = None
    if view == '#voces':
        position = preview_position(slot['ordinal'], positions['preview_permutation'], plan.get('preview_start', 0))
        capture = positions['positions'][position]['capture_id']
    consequences = overhead_consequences or {}
    untraced = bool(slot.get('untraced_inp'))
    screencast = not untraced and (not consequences.get('screencast_limited_to_contrast') or bool(slot.get('trace_contrast')))
    return RunSpec(slot['ordinal'], slot['role'], viewport, view, ready['origin'] + '/' + view, ready['origin'],
                   ready['tokens'][VIEW_ROLES[view]], trace_load=bool(slot.get('trace_contrast')),
                   trace_interactions=not untraced, screencast=screencast, capture_id=capture,
                   warm_up=bool(slot.get('warm_up')), primer=bool(slot.get('primer')),
                   trace_contrast=bool(slot.get('trace_contrast')), play_rule=play_rule, frame_event=frame_event)


class BrowserSession:
    """Cold slots launch a fresh browser; warm slots reuse its target, or a new target after a failed leave."""

    def __init__(self, mode: str, binary: str, out_dir: Path, profile_root: Path | None = None,
                 netlog: bool = False):
        self.mode = mode
        self.binary = binary
        self.out_dir = out_dir
        self.profile_root = profile_root
        self.netlog = netlog
        self.browser = self.page = self.target_id = self.previous_script = None
        self.needs_new_target = False

    async def close_browser(self) -> None:
        if self.page is not None:
            await self.page.close()
        if self.browser is not None:
            await self.browser.close()
        self.browser = self.page = self.target_id = self.previous_script = None

    async def prepare_target(self, spec: RunSpec) -> None:
        """Cold: a fresh browser. Warm: the same target, or a new about:blank target when the leave step failed."""
        if spec.role == 'cold':
            await self.close_browser()
            profile = Path(tempfile.mkdtemp(prefix='kza-t046-profile-', dir=self.profile_root))
            self.browser = Browser('harness' if self.mode == 'harness' else 'mac', self.binary, profile,
                                   self.out_dir / f'chrome-{spec.ordinal:05d}.log')
            netlog = self.out_dir / f'netlog-{spec.ordinal:05d}.json' if self.netlog else None
            await self.browser.launch(netlog=netlog)
            if netlog is not None:
                await wait_netlog_quiet(netlog)
            self.target_id = await self.browser.page_target()
            self.page = await self.browser.connect_page(self.target_id)
            self.previous_script, self.needs_new_target = None, False
        elif self.needs_new_target:
            try:
                await self.browser.client.send('Target.closeTarget', {'targetId': self.target_id}, klass='setup')
            except (CdpTimeout, CdpError):
                pass
            created = await self.browser.client.send('Target.createTarget', {'url': 'about:blank'}, klass='setup')
            await self.page.close()
            self.target_id = created['targetId']
            self.page = await self.browser.connect_page(self.target_id)
            self.previous_script, self.needs_new_target = None, False

    async def execute(self, spec: RunSpec, fixture: FixtureProcess | None) -> dict:
        try:
            await self.prepare_target(spec)
        except (RuntimeError, OSError, CdpTimeout, CdpError, CdpDisconnected, SetupFault) as error:
            return {'ordinal': spec.ordinal, 'role': spec.role, 'outcome': Outcome.INVALID,
                    'reason': 'launch_or_target:' + type(error).__name__ + ':' + str(error)[:120],
                    'navigate_sent': False, 'metrics': {}}
        run = Run(spec, self.browser, self.page, self.target_id, self.out_dir, fixture, self.previous_script)
        result = await run.execute()
        self.needs_new_target = not run.leave_ok
        self.previous_script = run.script_identifier if run.leave_ok else None
        result['target_id'] = self.target_id
        result['needs_new_target'] = self.needs_new_target
        result['browser_argv'] = self.browser.argv
        if run.trace_events:
            write_gzip_json_atomic(self.out_dir / f'trace-{spec.ordinal:05d}.json.gz', run.trace_events)
        return result


def netlog_hosts(path: Path) -> list:
    """Non-loopback request URLs in a (possibly unterminated) Chrome net log."""
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return []
    urls = re.findall(r'"url":"((?:https?|wss?)://[^"]+)"', text)
    return [url for url in urls if background_branch([{'requests': [url]}]) == 'netlog']


async def wait_netlog_quiet(path: Path, *, quiet_s: float = 2.0, cap_s: float = 30.0) -> None:
    """smoke_gate.background_activity 'netlog' branch: wait until no new non-loopback request for 2 s, cap 30 s."""
    started = time.monotonic()
    count, changed = len(netlog_hosts(path)), time.monotonic()
    while time.monotonic() - changed < quiet_s:
        if time.monotonic() - started > cap_s:
            raise SetupFault('netlog_quiet_cap_exceeded')
        await asyncio.sleep(0.2)
        current = len(netlog_hosts(path))
        if current != count:
            count, changed = current, time.monotonic()


def served_state() -> dict:
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': tree_dirty()}


def load_fill_ins(preregistration: dict) -> dict:
    """Values the smoke fills in (run_rules.fill_in_values); absent until the smoke ran.

    Read from a top-level ``fill_ins`` object of the preregistration: idle_load, post_batch_load,
    background_branch, overhead (consequences), play_control_rule, frame_event, lock_read and
    harness_only_deadlines_ms.
    """
    return preregistration.get('fill_ins', {})


def apply_fill_ins(fill_ins: dict) -> dict:
    """Apply harness-only deadline fill-ins; an immutable class can never change (R5-5)."""
    deadlines = fill_ins.get('harness_only_deadlines_ms') or {}
    for klass, value in deadlines.items():
        if klass not in HARNESS_ONLY_CLASSES or value < DEADLINE_MS[klass]:
            raise SystemExit(f'fill_in_refused_for_deadline_class:{klass}')
        DEADLINE_MS[klass] = int(value)
    return fill_ins


async def measure_cell(args, mode: str, sha: str) -> dict:
    """Run one (client, viewport, view) batch on the Mac (baseline or smoke); returns the batch record."""
    from tests.frontend.browser_support import browser_binary
    if sys.platform != 'darwin':
        raise SystemExit('measurement_runs_only_on_the_mac')
    preregistration = json.loads(PREREGISTRATION.read_text())
    fill_ins = apply_fill_ins(load_fill_ins(preregistration))
    smoke = mode == 'smoke'
    # R6-1, R6-2, R6-9: every per-mode input comes from cell_settings (smoke record or fill-ins), never the CLI.
    settings = cell_settings(mode, fill_ins, SmokeRecord(SMOKE_ROOT / 'smoke-record.json') if smoke else None)
    idle, post_batch = settings['idle'], settings['post_batch']
    plan = plan_batch(sha, 'mac-chrome', args.viewport, args.view, args.attempt, smoke=smoke,
                      **settings['composition'],
                      untraced_inp=settings['untraced_inp'] and args.view in INTERACTION_VIEWS)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    out_dir = checked_output(mode, output_root(mode) / f'{stamp}-{args.viewport}-{args.view.strip("#")}-a{args.attempt}')
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    session = BrowserSession(mode, browser_binary(), out_dir, netlog=settings['background_branch'] == 'netlog')
    fixture = FixtureProcess(out_dir, harness_pages=False)
    state = {}
    environment_facts = mac_environment()
    identity = {'mode': mode, 'preregistration_sha256': sha, 'script_id': script_id(),
                'launch_flags': list(LAUNCH_FLAGS), 'launch_mechanism': LAUNCH_MECHANISM,
                'os_version': environment_facts['os_version'], 'hardware': environment_facts['hardware'],
                'ui_locale_apple_languages_first': environment_facts['apple_languages_first'],
                'pip_freeze_sha256': pip_freeze_sha256(), 'commit': git('rev-parse', 'HEAD'),
                'audio_output': audio_output()}

    async def start_fixture():
        state['ready'] = await fixture.start()
        state['positions'] = json.loads(Path(state['ready']['position_map']).read_text())
        return state['ready']

    async def run_slot(slot):
        spec = slot_spec(slot, plan, state['ready'], state['positions'], settings['overhead'],
                         play_rule=settings['play_control_rule'], frame_event=settings['frame_event'])
        result = await session.execute(spec, fixture)
        result['mode'] = mode
        # V14c: every result file names the full method_identity it was produced under.
        language = (result.get('document_start') or {})
        result['method_identity'] = {**identity, 'fixture_digest': state['ready'].get('digest'),
                                     'browser_version': result.get('browser_version'),
                                     'ui_locale_navigator': [language.get('language'), language.get('languages')]}
        write_json_atomic(out_dir / f'run-{slot["ordinal"]:05d}.json', json_safe(result))
        return result

    lock_choice = (fill_ins.get('lock_read') or {}).get('read') if isinstance(fill_ins.get('lock_read'), dict) \
        else fill_ins.get('lock_read')
    environment = BatchEnvironment(quiet_check=lambda: quiet_check(os.getpid()), loadavg=loadavg,
                                   sleep=asyncio.sleep, start_fixture=start_fixture, stop_fixture=fixture.stop,
                                   run_slot=run_slot, close_browser=session.close_browser,
                                   caffeinate=start_caffeinate, served_state=served_state,
                                   machine_state=lambda: machine_state(lock_choice),
                                   assertion_held=caffeinate_assertion_wait, sweep=sweep_probe_browsers)
    header = {'mode': mode, 'preregistration_sha256': sha, 'script_id': script_id(), 'plan_seed': plan['seed'],
              'launch_flags': list(LAUNCH_FLAGS), 'launch_mechanism': LAUNCH_MECHANISM,
              'environment': environment_facts, 'method_identity': identity, 'label': LOCAL_LABEL,
              'pip_freeze_sha256': identity['pip_freeze_sha256'], 'cell_settings': settings,
              'play_control_rule': settings['play_control_rule'], 'composition': settings['composition']}
    batch = await run_batch(plan, environment, idle=idle, post_batch=post_batch, end_record_only=smoke,
                            warm_rule=settings['warm_rule'], smoke=smoke)
    header['closure'] = recorded_closure(json.loads((out_dir / 'fixture-modules.json').read_text())
                                         if (out_dir / 'fixture-modules.json').exists() else [])
    batch_record = {**header, **{key: value for key, value in batch.items() if key != 'runs'},
                    'run_outcomes': [{'ordinal': run['ordinal'], 'role': run['role'], 'outcome': run['outcome'],
                                      'reason': run.get('reason')} for run in batch['runs']],
                    'events': environment.events, 'directory': str(out_dir)}
    batch_record['faults'] = sorted(batch_faults(batch_record))
    write_json_atomic(out_dir / 'batch.json', json_safe(batch_record))
    print(json.dumps({'batch': str(out_dir), 'discarded': batch.get('discarded'), 'reasons': batch.get('reasons')}))
    return batch_record


def batch_faults(batch: dict) -> set:
    """V9: the harness or environment causes a batch recorded, which alone may justify a smoke repeat."""
    faults = set()
    reasons = batch.get('reasons') or []
    if 'quiet_machine' in reasons:
        faults.add('quiet_machine')
    if 'fixture_fault' in reasons:
        faults.add('fixture_fault')
    machine = [*(batch.get('start') or {}).get('machine_faults', []), *(batch.get('end') or {}).get('machine_faults', [])]
    if 'screen_locked' in machine:
        faults.add('lock')
    for run in batch.get('run_outcomes') or batch.get('runs') or []:
        reason = run.get('reason') or ''
        if reason == 'browser_crashed':
            faults.add('crash')
        elif reason == 'websocket_disconnected':
            faults.add('disconnect')
        elif reason.startswith(('fixture_not_ready', 'position_check_failed', 'login_failed')):
            faults.add('fixture_fault')
    return faults


async def caffeinate_assertion_wait(process, attempts: int = 10) -> bool:
    """The assertion appears shortly after caffeinate starts; poll it for up to ~2 s."""
    for _ in range(attempts):
        if await asyncio.to_thread(caffeinate_assertion_held, process):
            return True
        await asyncio.sleep(0.2)
    return False


# --------------------------------------------------------------------------------------------
# Harness command (server headless Chromium): functional dry runs, never measurements.
# --------------------------------------------------------------------------------------------

async def run_sequence(mode: str, specs: list, out_dir: Path, fixture: FixtureProcess | None, *,
                       binary: str, profile_root: Path | None) -> list:
    """Run specs in order: a cold spec launches a browser; following warm specs reuse its target."""
    session = BrowserSession(mode, binary, out_dir, profile_root)
    results = []
    try:
        for spec in specs:
            results.append(await session.execute(spec, fixture))
    finally:
        await session.close_browser()
    return results


def withhold_values(result: dict) -> dict:
    """Harness results never report timing values (never a measurement)."""
    copy = dict(result)
    copy['metrics'] = {key: ('withheld' if not isinstance(value, str) else value)
                       for key, value in result.get('metrics', {}).items()}
    return copy


async def harness_dry_run(args) -> int:
    """Functional dry run of the fixture and probe on server headless Chromium (clients.local.fallback)."""
    from tests.frontend.browser_support import browser_binary, browser_profile_root, evidence_directory
    if sys.platform == 'darwin':
        raise SystemExit('harness_mode_is_for_server_headless_chromium_only')
    out_dir = checked_output('harness', Path(args.out) if args.out else evidence_directory(
        'KZA_T046_HARNESS_EVIDENCE', str(HARNESS_ROOT.relative_to(REPOSITORY))))
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fixture = FixtureProcess(out_dir, harness_pages=True)
    try:
        ready = await fixture.start()
        positions = json.loads(Path(ready['position_map']).read_text())
        specs = []
        ordinal = 0
        for viewport, view in [(cell['viewport'], cell['view']) for cell in batch_plan_views()
                               if args.view in (None, cell['view'])]:
            role = VIEW_ROLES[view]
            for slot_role in ('cold', 'warm'):
                capture = None
                if view == '#voces':
                    capture = positions['positions'][preview_position(ordinal, positions['preview_permutation'])]['capture_id']
                specs.append(RunSpec(ordinal, slot_role, viewport, view, ready['origin'] + '/' + view, ready['origin'],
                                     ready['tokens'][role], capture_id=capture))
                ordinal += 1
        results = await run_sequence('harness', specs, out_dir, fixture, binary=browser_binary(),
                                     profile_root=browser_profile_root(mode=0o700))
    finally:
        await fixture.stop()
    summary = [{'ordinal': r['ordinal'], 'view': r['view'], 'viewport': r['viewport'], 'role': r['role'],
                'outcome': r['outcome'], 'reason': r['reason'], 'steps': [s['status'] for s in r['steps']],
                'accepted_records': len(r['records']['accepted']), 'rejected_records': len(r['records']['rejected']),
                'post_window_violations': r['post_window_violations'], 'flags': r['flags'],
                'environment_faults': r.get('facts', {}).get('environment_faults'),
                'product_failures': r.get('facts', {}).get('product_failures'),
                'window_reads': [(item['point'], item['state']) for item in r.get('window_reads', [])],
                'scrolled': [record['step'] for record in r['records']['accepted'] if record['type'] == 'scrolled'],
                'play_ready_record': next((step.get('ready_record') for step in r['steps'] if step.get('ready_record')),
                                          None),
                'http_errors': r.get('http_errors'), 'timed_exceptions': r.get('timed_exceptions')} for r in results]
    for result in results:
        write_json_atomic(out_dir / f'run-{result["ordinal"]:05d}.json', json_safe(withhold_values(result)))
    write_json_atomic(out_dir / 'dry-run-summary.json', {'mode': 'harness', 'not_a_measurement': True,
                                                         'fixture_digest': ready['digest'], 'runs': summary})
    print(json.dumps(summary, indent=1))
    return 0


# --------------------------------------------------------------------------------------------
# Smoke items that are not batches (smoke_gate.quiet_baseline, cold_focus, background_activity, lock read).
# --------------------------------------------------------------------------------------------

async def smoke_quiet_baseline(record: SmokeRecord, sha: str, cause: str | None) -> dict:
    """R5-7: idle = median of 5 readings 60 s apart with the process check passing and no probe Chrome."""
    entry = record.start('quiet_baseline', sha, cause)
    readings = []
    sweep_probe_browsers()
    for number in range(5):
        check = await quiet_check(os.getpid())
        if not check['ok']:
            record.finish(entry, 'aborted', {'check': check, 'readings': readings}, faults=['quiet_machine'])
            raise SystemExit('quiet_baseline_process_check_failed')
        readings.append(loadavg())
        if number < 4:
            await asyncio.sleep(60)
    result = {'readings': readings, 'idle': statistics.median(readings)}
    record.finish(entry, 'complete', result)
    return result


async def smoke_item_quiet_start(idle: float | None) -> dict:
    """V8: the batch-start quiet_machine checks for a smoke item that is not a batch (sweep, process check, load
    reading against the smoke idle, AC power, display; lock recorded)."""
    swept = sweep_probe_browsers()
    check = await quiet_check(os.getpid())
    load = loadavg()
    machine = machine_state()
    faults = machine_faults(machine, smoke=True)
    return {'ok': bool(check.get('ok')) and load_reading_ok(load, idle) and not faults, 'quiet': check,
            'load': load, 'idle': idle, 'machine': machine, 'machine_faults': faults, 'swept': swept}


async def run_smoke_item(record: SmokeRecord, item: str, sha: str, cause: str | None, body: Callable) -> dict:
    """V8, V9: a smoke item under the quiet_machine start check and caffeinate, recorded with its faults."""
    entry = record.start(item, sha, cause)
    start = await smoke_item_quiet_start(smoke_idle(record))
    if not start['ok']:
        faults = ['quiet_machine'] + (['lock'] if 'screen_locked' in start['machine_faults'] else [])
        record.finish(entry, 'aborted', {'quiet_start': start}, faults=faults)
        raise SystemExit(f'smoke_item_quiet_machine_failed:{item}')
    caffeinate = start_caffeinate()
    try:
        held = await caffeinate_assertion_wait(caffeinate)
        if not held:
            record.finish(entry, 'aborted', {'quiet_start': start, 'caffeinate_assertion': False},
                          faults=['quiet_machine'])
            raise SystemExit(f'smoke_item_caffeinate_assertion_missing:{item}')
        result = await body()
        held_end = await caffeinate_assertion_wait(caffeinate)
    except SystemExit:
        raise
    except BaseException as error:
        record.finish(entry, 'aborted', {'quiet_start': start, 'error': f'{type(error).__name__}: {str(error)[:200]}'},
                      faults=['crash'])
        raise
    finally:
        caffeinate.terminate()
        sweep_probe_browsers()
    faults = set(result.pop('faults', []) if isinstance(result, dict) else [])
    if not held_end:
        faults.add('quiet_machine')
    record.finish(entry, 'complete', {**result, 'quiet_start': start, 'caffeinate_assertion_end': held_end},
                  faults=faults)
    return result


async def smoke_cold_focus(out_dir: Path) -> dict:
    """R4-6, R5-14a: 20 consecutive `open -na` launches pass the activation poll on about:blank."""
    from tests.frontend.browser_support import browser_binary
    polls = []
    for number in range(20):
        browser = Browser('mac', browser_binary(), Path(tempfile.mkdtemp(prefix='kza-t046-focus-')),
                          out_dir / f'focus-{number:02d}.log')
        try:
            await browser.launch()
            target = await browser.page_target()
            page = await browser.connect_page(target)
            try:
                polls.append(await activation_poll(page, browser.client, target, DEADLINE_MS['setup'] / 1000))
            except ActivationFailed as error:
                polls.append({'failed': error.polls})
            await page.close()
        finally:
            await browser.close()
    return {'polls': polls, 'passed': all(isinstance(value, int) for value in polls)}


async def smoke_background(out_dir: Path) -> dict:
    """R4-6, R5-5: 5 fresh profiles, first 10 s: non-loopback requests (net log), new targets, dialogs, focus."""
    from tests.frontend.browser_support import browser_binary
    profiles = []
    for number in range(5):
        netlog = out_dir / f'background-netlog-{number}.json'
        browser = Browser('mac', browser_binary(), Path(tempfile.mkdtemp(prefix='kza-t046-background-')),
                          out_dir / f'background-{number}.log')
        observed = {'targets': [], 'dialogs': [], 'focus': []}
        try:
            launched = time.monotonic()
            await browser.launch(netlog=netlog)
            browser.client.on('Target.targetCreated', lambda params: observed['targets'].append(params))
            await browser.client.send('Target.setDiscoverTargets', {'discover': True}, klass='setup')
            target = await browser.page_target()
            page = await browser.connect_page(target)
            page.on('Page.javascriptDialogOpening', lambda params: observed['dialogs'].append(params))
            await page.send('Page.enable', klass='setup')
            while time.monotonic() - launched < 10:
                focus = await page.send('Runtime.evaluate', {'expression': 'document.hasFocus()', 'returnByValue': True},
                                        klass='setup')
                observed['focus'].append(focus.get('result', {}).get('value'))
                await asyncio.sleep(0.5)
            await page.close()
        finally:
            await browser.close()
        profiles.append({'requests': netlog_hosts(netlog), 'new_targets': len(observed['targets']),
                         'dialogs': len(observed['dialogs']), 'focus_lost': False in observed['focus']})
    return {'profiles': profiles, 'branch': background_branch(profiles)}


def smoke_lock_read(record: SmokeRecord, sha: str, cause: str | None, state: str) -> dict:
    """quiet_machine.lock_read: read while the user reports the screen 'locked' or 'unlocked'."""
    entry = record.start(f'lock_read_{state}', sha, cause)
    result = {'state_reported': state, 'reads': lock_reads(), 'display_asleep': display_asleep(), 'time': time.time()}
    record.finish(entry, 'complete', result)
    return result


def smoke_cell_action(record: SmokeRecord, item: str, cause: str | None) -> tuple:
    """R6-10 (run_rules.which_run_counts): resume a smoke client. Returns ('skip', None) for a cell whose last entry
    is complete, or discarded without the given cause among its faults (never repeated automatically); otherwise
    ('run', cause_to_record): None for a first run, the given cause for a repeat (SmokeRecord.start refuses a cause
    the previous run did not record)."""
    previous = record.previous(item)
    if not previous:
        return 'run', None
    last = previous[-1]
    if last.get('status') == 'complete':
        return 'skip', None
    if last.get('status') == 'discarded' and cause not in (last.get('faults') or []):
        return 'skip', None
    return 'run', cause


async def measure_client(args, mode: str, sha: str) -> int:
    """All cells of the Mac client in the seeded cell order; two consecutive discards stop with a blocker.

    V9: in smoke mode every cell is its own smoke record entry and a discarded batch is never repeated
    automatically (a repeat needs a recorded harness or environment cause, run_rules.which_run_counts). R6-10: a
    smoke client resumes after a crash: complete cells are skipped and --cause is passed for the repeated cell."""
    cells = [(cell['viewport'], cell['view']) for cell in batch_plan_views()]
    ledger = BatchLedger()
    attempts = Counter()
    smoke = mode == 'smoke'
    record = SmokeRecord(checked_output('smoke', SMOKE_ROOT / 'smoke-record.json')) if smoke else None
    if smoke:
        # R6-1: refuse before any entry is written (order, play rule, background branch, idle).
        cell_settings('smoke', {}, record)
    order = cell_order(sha, 'mac-chrome', 1, cells, smoke=smoke)
    queue = list(order)
    while queue:
        if not ledger.may_start():
            raise SystemExit('blocker: two consecutive discarded batches; record the cause before continuing')
        viewport, view = queue.pop(0)
        item = f'batch:{viewport}:{view}'
        entry = None
        if smoke:
            action, cause = smoke_cell_action(record, item, getattr(args, 'cause', None))
            if action == 'skip':
                continue
            attempts[(viewport, view)] = len(record.previous(item))
            entry = record.start(item, sha, cause)
        attempts[(viewport, view)] += 1
        args.viewport, args.view, args.attempt = viewport, view, attempts[(viewport, view)]
        try:
            report = await measure_cell(args, mode, sha)
        except BaseException as error:
            if smoke:
                record.finish(entry, 'aborted', {'error': f'{type(error).__name__}: {str(error)[:200]}'},
                              faults=['crash'])
            raise
        ledger.record(report)
        if smoke:
            record.finish(entry, 'discarded' if report.get('discarded') else 'complete',
                          {'directory': report.get('directory'), 'reasons': report.get('reasons'),
                           'composition': report.get('composition'),
                           'play_control_rule': report.get('play_control_rule')},
                          faults=report.get('faults'))
        elif report.get('discarded'):
            queue.insert(0, (viewport, view))
    return 0


# --------------------------------------------------------------------------------------------
# Fixture pages (smoke_gate.cdp_timeout, sub_threshold, preview_gap, mac_environment; harness checks).
# --------------------------------------------------------------------------------------------

FIXTURE_STEP = {'container': '#harness', 'wait': 'wait_other'}
FIXTURE_STEPS = {
    'busy-window': [dict(FIXTURE_STEP, step='bloquear', selector='#busy', end={'kind': 'exists', 'selector': '#never'})],
    'sub-threshold': [dict(FIXTURE_STEP, step='accion', selector='#noop', wait='wait_feedback', c11='action_feedback',
                           end={'kind': 'nonempty_attribute', 'selector': '#harness [data-receipt]',
                                'attribute': 'data-status'})],
    'detach': [dict(FIXTURE_STEP, step='escuchar', selector='#first', end={'kind': 'exists', 'selector': '#replaced'})],
    'missing': [dict(FIXTURE_STEP, step='ausente', selector='#absent', end={'kind': 'exists', 'selector': '#never'})],
    'scroll': [dict(FIXTURE_STEP, step='primero', selector='#first',
                    end={'kind': 'attribute_present', 'selector': '#first', 'attribute': 'data-done'}),
               dict(FIXTURE_STEP, step='segundo', selector='#second',
                    end={'kind': 'attribute_present', 'selector': '#second', 'attribute': 'data-done'})],
    'present': [dict(FIXTURE_STEP, step='presente', selector='#present',
                     end={'kind': 'attribute_present', 'selector': '#present', 'attribute': 'data-never'})],
    'preview': [{'step': 'escuchar', 'container': '#harness', 'selector': 'button', 'name_prefix': 'Escuchar',
                 'end': {'kind': 'control_attached'}, 'wait': 'wait_preview', 'c11': 'preview'},
                {'step': 'reproducir', 'play_gesture': True, 'end': {'kind': 'playing'}, 'wait': 'wait_preview'}],
}


def fixture_spec(ordinal: int, role: str, ready: dict, page: str, steps: list, query: str = '', *,
                 after_settle: Callable | None = None, play_rule: dict | None = None) -> RunSpec:
    return RunSpec(ordinal, role, 'desktop', '#harness', f'{ready["origin"]}/__harness/{page}.html{query}',
                   ready['origin'], None, harness_steps=steps, harness=True, after_settle=after_settle,
                   play_rule=play_rule)


def accepted(result: dict, kind: str) -> list:
    return [record for record in result['records']['accepted'] if record['type'] == kind]


def timeout_phases(result: dict) -> list:
    return [timeout['phase'] for timeout in result['facts']['timeouts']]


def no_environment_fault(result: dict) -> bool:
    """F1 resolved by the preregistered focus rule: a healthy fixture run has no environment fault at all."""
    return result['facts']['environment_faults'] == []


# smoke_gate.preview_ready_state (V2, F4): the provisional native play-control rule and the ordered candidates
# tried on the native fixture preview page (product-independent) when it does not play on 5 of 5 runs.
PLAY_RULE_CANDIDATES = (PLAY_CONTROL_RULE,) + tuple(
    {'dx_px': float(dx), 'y_fraction': 0.5, 'status': 'smoke_candidate'} for dx in (12, 16, 24, 28))


def dialog_variant_summary(runs: list, rule: dict) -> dict:
    """R6-6: per run of the dialog variant: playing, attach/ready targets (rect, point, element hit) and the kept
    screencast frame; 'plays' is True only when every run reached 'playing' after control_ready."""
    rows = []
    for run in runs:
        ready = (accepted(run, 'control_ready') or accepted(run, 'control_attached') or [{}])[0]
        rows.append({'ordinal': run['ordinal'], 'outcome': run['outcome'], 'reason': run['reason'],
                     'playing': bool([r for r in accepted(run, 'media') if r['event'] == 'playing']),
                     'target': ready.get('target'), 'play_click_frame': run.get('play_click_frame'),
                     'failures': run['facts']['product_failures'],
                     'environment_faults': run['facts']['environment_faults']})
    return {'rule': rule, 'runs': rows, 'plays': bool(rows) and all(row['playing'] for row in rows)}


def play_rule_choice(trials: list) -> dict:
    """First rule (in the preregistered order) whose 5 native-control runs all reached 'playing'."""
    for trial in trials:
        if trial['runs'] and all(trial['playing']):
            return {'rule': trial['rule'], 'trials': trials, 'method_change': False}
    return {'rule': None, 'trials': trials, 'method_change': True}


async def fixture_page_checks(mode: str, out_dir: Path, *, binary: str, profile_root: Path | None) -> dict:
    """Run the fixture-page scenarios and evaluate their preregistered expectations (fixture pages only)."""
    fixture = FixtureProcess(out_dir, harness_pages=True)
    checks, observations, raw = {}, {}, {}
    try:
        ready = await fixture.start()
        async def sequence(name, specs):
            results = await run_sequence(mode, specs, out_dir, None, binary=binary, profile_root=profile_root)
            raw[name] = results
            return results

        cold, warm = await sequence('binding', [fixture_spec(0, 'cold', ready, 'plain', []),
                                                fixture_spec(1, 'warm', ready, 'plain', [])])
        page_methods = [entry['method'] for entry in warm['commands'] if entry.get('connection') == 'page']
        cold_token = accepted(cold, 'document_start')[0]['token'] if accepted(cold, 'document_start') else None
        checks['record_binding_one_injection'] = {
            'ok': len(accepted(cold, 'document_start')) == 1 and len(accepted(warm, 'document_start')) == 1
            and 'Page.removeScriptToEvaluateOnNewDocument' in page_methods
            and page_methods.index('Page.removeScriptToEvaluateOnNewDocument')
            < page_methods.index('Page.addScriptToEvaluateOnNewDocument')
            and cold['script_identifier'] is not None and cold['target_id'] == warm['target_id']
            and warm['records']['rejected'] == []
            and all(record['token'] != cold_token for record in warm['records']['accepted']),
            'document_starts': [len(accepted(cold, 'document_start')), len(accepted(warm, 'document_start'))],
            'rejected': [len(cold['records']['rejected']), len(warm['records']['rejected'])]}

        busy, after_busy = await sequence('busy_window', [
            fixture_spec(2, 'cold', ready, 'busy-window', FIXTURE_STEPS['busy-window']),
            fixture_spec(3, 'warm', ready, 'plain', [])])
        checks['timed_window_hang_is_failure'] = {
            'ok': busy['outcome'] == Outcome.FAILURE and timeout_phases(busy)[:1] == ['timed']
            and (after_busy['target_id'] != busy['target_id']) == busy['needs_new_target']
            and not after_busy['facts']['setup_faults'] and not after_busy['facts']['timeouts']
            and no_environment_fault(after_busy) and after_busy['outcome'] == Outcome.VALID,
            'busy': [busy['outcome'], busy['reason'], timeout_phases(busy)],
            'leave_step_timed_out': busy['needs_new_target'],
            'next': [after_busy['outcome'], after_busy['reason'], after_busy['facts']['environment_faults']]}

        hung, after_hung = await sequence('busy_after_flush', [
            fixture_spec(4, 'cold', ready, 'busy-after-flush', []), fixture_spec(5, 'warm', ready, 'plain', [])])
        observations['busy_after_flush'] = {
            'outcome': hung['outcome'], 'reason': hung['reason'], 'timeouts': timeout_phases(hung),
            'leave_step_timed_out': hung['needs_new_target'],
            'new_target_for_next': after_hung['target_id'] != hung['target_id'],
            'next_setup_faults': after_hung['facts']['setup_faults'], 'next_timeouts': timeout_phases(after_hung)}
        checks['hung_page_next_run_has_no_harness_fault'] = {
            'ok': (hung['outcome'] == Outcome.POST_WINDOW_HANG) == hung['needs_new_target']
            and (after_hung['target_id'] != hung['target_id']) == hung['needs_new_target']
            and not after_hung['facts']['setup_faults'] and not after_hung['facts']['timeouts']
            and no_environment_fault(after_hung) and after_hung['outcome'] == Outcome.VALID,
            'next': [after_hung['outcome'], after_hung['reason'], after_hung['facts']['environment_faults']]}

        (sub,) = await sequence('sub_threshold', [fixture_spec(6, 'cold', ready, 'sub-threshold',
                                                               FIXTURE_STEPS['sub-threshold'])])
        feedback = sub['metrics'].get('action_feedback', {})
        first_event = accepted(sub, 'input')[0]['timeStamp'] if accepted(sub, 'input') else None
        mapping = sub['metrics'].get('inp_mapping') or {}
        groups = (mapping.get('groups_per_interaction') or [None])[0]
        inp = sub['metrics'].get('inp')
        # F5 (smoke_gate.sub_threshold): the C11 feedback value exists, its rAF lower bound is not before the start
        # (V5), and INP is '<16 ms' exactly when no Event Timing group maps to the window, else the group's value.
        checks['sub_threshold_feedback'] = {
            'ok': feedback.get('start') == first_event and sub['metrics'].get('action_feedback_raf') not in (None, INF)
            and sub['metrics']['action_feedback_raf'] >= 0 and not sub['facts']['product_failures']
            and no_environment_fault(sub)
            and ((groups == 0 and inp == CENSORED) or (groups and inp == max(
                value for value in mapping['latencies'] if value != CENSORED))),
            'feedback_raf_ms': sub['metrics'].get('action_feedback_raf'), 'presented': feedback.get('presented'),
            'inp': inp, 'groups': groups}
        observations['sub_threshold_inp'] = inp

        (scrolled,) = await sequence('scroll', [fixture_spec(13, 'cold', ready, 'scroll', FIXTURE_STEPS['scroll'])])
        moves = accepted(scrolled, 'scrolled')
        next_frames = [record for record in accepted(scrolled, 'step_next_frame') if record['index'] == 0]
        second_input = [record for record in accepted(scrolled, 'input') if record.get('interaction') == 1]
        # V3: only step 1 is scrolled, after step 0's step_next_frame and before step 1's first input.
        checks['scroll_after_previous_step_frame'] = {
            'ok': [step['status'] for step in scrolled['steps']] == ['dispatched', 'dispatched']
            and [record['index'] for record in moves] == [1] and bool(next_frames) and bool(second_input)
            and next_frames[0]['t'] <= moves[0]['t'] < second_input[0]['timeStamp']
            and not scrolled['facts']['product_failures'] and no_environment_fault(scrolled),
            'scrolled': [(record['index'], record['t']) for record in moves],
            'steps': [step['status'] for step in scrolled['steps']]}

        (detach,) = await sequence('detach', [fixture_spec(7, 'cold', ready, 'detach', FIXTURE_STEPS['detach'])])
        diagnostics = [item for group in detach['metrics'].get('inp_mapping', {}).get('diagnostics', []) for item in group]
        checks['detached_target'] = {
            'ok': detach['metrics'].get('inp') not in (CENSORED, INF, None) and detach['metrics']['inp'] >= 40
            and any('null' in item['target'] for item in diagnostics),
            'inp': detach['metrics'].get('inp'), 'diagnostics': diagnostics}

        (missing,) = await sequence('missing', [fixture_spec(8, 'cold', ready, 'plain', FIXTURE_STEPS['missing'])])
        checks['missing_target'] = {
            'ok': missing['facts']['product_failures'] == ['target_absent:ausente'] and missing['metrics']['inp'] == INF
            and missing['metrics']['cls'] == INF
            and not any(entry['method'].startswith('Input.') for entry in missing['commands']),
            'failures': missing['facts']['product_failures']}

        runs = await sequence('preview', [
            fixture_spec(9, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '?button=1'),
            fixture_spec(10, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '?button=1&longtask=1'),
            fixture_spec(11, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '?aria=1'),
            fixture_spec(12, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '')])
        targets = [(accepted(run, 'control_attached') or [{}])[0].get('target', {}) for run in runs]
        gaps = [run['metrics'].get('preview', {}).get('gap') for run in runs]
        between = []
        for run in runs[:3]:
            page_log = [entry for entry in run['commands'] if entry.get('connection') == 'page']
            inputs = [index for index, entry in enumerate(page_log) if entry['method'].startswith('Input.')]
            between.append([entry['method'] for entry in page_log[inputs[2] + 1:inputs[3]]
                            if entry['method'].startswith(('Runtime.', 'DOM.', 'Accessibility.'))]
                           if len(inputs) >= 4 else None)
        checks['play_gesture_lookup'] = {
            'ok': targets[0].get('kind') == 'button' and targets[0].get('name') == 'Reproducir muestra'
            and targets[2].get('kind') == 'button' and targets[2].get('name') == 'reproducir'
            and targets[3].get('kind') == 'audio',
            'targets': [{key: target.get(key) for key in ('kind', 'name')} for target in targets]}
        checks['play_gesture_dispatch_long_task'] = {
            'ok': all(value == [] for value in between) and None not in gaps[:2] and gaps[1] <= PREVIEW_GAP_MAX_MS
            and all(run['metrics'].get('preview', {}).get('failed_open') is False for run in runs[:3]),
            'gaps_ms': gaps, 'renderer_commands_between': between}
        observations['preview_playing_by_variant'] = {
            name: bool([r for r in accepted(run, 'media') if r['event'] == 'playing'])
            for name, run in zip(('button', 'button_longtask', 'aria', 'native_audio_rule'), runs)}
        observations['native_audio_ready_state'] = runs[3]['metrics'].get('preview', {}).get('secondary', {})
        native = runs[3]['metrics'].get('preview', {})
        native_attach = (accepted(runs[3], 'control_attached') or [{}])[0]
        # V2: a native control attached before its metadata is clicked from control_ready and plays.
        play_step = next((step for step in runs[3]['steps'] if step.get('ready_record')), {})
        native_mapping = runs[3]['metrics'].get('inp_mapping') or {}
        # V2, F6: clicked from control_ready, plays, primary value unsubtracted, probe-side delay <= 16 ms, and the
        # play gesture keeps its own INP slot (window [ready, flush]).
        checks['native_control_waits_for_metadata'] = {
            'ok': native.get('failed_open') is False and native.get('definition') == 'native_unsubtracted'
            and play_step.get('probe_dispatch_delay_ms') is not None
            and play_step['probe_dispatch_delay_ms'] <= PREVIEW_GAP_MAX_MS
            and len(native_mapping.get('latencies') or []) == 2 and no_environment_fault(runs[3])
            and (native_attach.get('readyState', 0) >= 1
                 or native.get('secondary', {}).get('ready_record') == 'control_ready'),
            'attach_ready_state': native_attach.get('readyState'),
            'ready_record': native.get('secondary', {}).get('ready_record'),
            'probe_dispatch_delay_ms': play_step.get('probe_dispatch_delay_ms'),
            'play_inp_groups': (native_mapping.get('groups_per_interaction') or [None, None])[1:],
            'failures': runs[3]['facts']['product_failures']}
        observations['native_control_input_events_seen_by_page'] = [
            len([record for record in accepted(run, 'input') if record.get('interaction') == 1]) for run in runs]

        # smoke_gate.preview_ready_state (V2, F4): the first preregistered rule with 5 of 5 native runs playing.
        trials = []
        for number, rule in enumerate(PLAY_RULE_CANDIDATES):
            specs = [fixture_spec(100 + 5 * number + index, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '',
                                  play_rule=rule) for index in range(5)]
            results = await sequence(f'play_rule_{number}', specs)
            trials.append({'rule': rule, 'runs': len(results),
                           'playing': [bool([r for r in accepted(run, 'media') if r['event'] == 'playing'])
                                       for run in results]})
            if all(trials[-1]['playing']):
                break
        observations['play_control_rule'] = play_rule_choice(trials)

        # R6-6 (smoke_gate.preview_ready_state): the NEXO desktop context of the native control (430 px wide, in the
        # right column of a showModal() dialog whose scroll container was scrolled), 5 cold runs with the chosen
        # rule; per run: playing, the element hit at the dispatch point and the screencast frame at the click.
        chosen = observations['play_control_rule']['rule']
        if chosen is not None:
            dialog_runs = await sequence('preview_dialog', [
                fixture_spec(130 + index, 'cold', ready, 'preview', FIXTURE_STEPS['preview'], '?dialog=1',
                             play_rule=chosen) for index in range(5)])
            observations['preview_dialog'] = dialog_variant_summary(dialog_runs, chosen)

        if mode == 'smoke':
            # smoke_gate.mac_environment: minimising the window and switching focus each invalidate the run.
            async def minimise():
                return await asyncio.to_thread(run_text, [
                    'osascript', '-e', 'tell application "Google Chrome" to set minimized of front window to true'])

            async def switch():
                return await asyncio.to_thread(run_text, ['open', '-a', 'Finder'])
            for ordinal, (name, action) in enumerate((('minimise', minimise), ('switch_focus', switch)), start=14):
                (result,) = await sequence(f'visibility_{name}', [
                    fixture_spec(ordinal, 'cold', ready, 'plain', FIXTURE_STEPS['present'], after_settle=action)])
                # F1 resolved: a healthy fixture run has no environment fault, so any fault here is the action's.
                checks[f'visibility_{name}_invalidates'] = {
                    'ok': result['outcome'] == Outcome.INVALID and bool(result['facts']['environment_faults']),
                    'outcome': result['outcome'], 'faults': result['facts']['environment_faults']}

        every = [run for results in raw.values() for run in results]
        checks['post_window_command_log'] = {'ok': all(run['post_window_violations'] == [] for run in every),
                                             'violations': [run['post_window_violations'] for run in every]}
        checks['no_dom_or_accessibility_command'] = {
            'ok': not any(entry['method'].startswith(('DOM.', 'Accessibility.')) for run in every
                          for entry in run['commands'])}
    finally:
        await fixture.stop()
    for name, results in raw.items():
        for result in results:
            write_json_atomic(out_dir / f'{name}-{result["ordinal"]:05d}.json', json_safe(result))
    summary = {'mode': mode, 'not_a_measurement': True, 'time': time.time(),
               'browser': next((run.get('browser_version') for results in raw.values() for run in results
                                if run.get('browser_version')), None),
               'checks': json_safe(checks), 'observations': json_safe(observations),
               'ok': all(check['ok'] for check in checks.values())}
    write_json_atomic(out_dir / 'fixture-page-checks.json', summary)
    return summary


# --------------------------------------------------------------------------------------------
# Smoke analysis helpers (smoke_gate.frame_event, instrumentation_overhead) and summaries.
# --------------------------------------------------------------------------------------------

FRAME_MATCH_MS = 16.7


def frame_event_phases(results: list) -> list:
    """Acceptance-test input from traced interaction phases: per C11-end interaction and candidate, the candidate
    time is >= the first-rAF lower bound and within 16.7 ms of the matching screencast frame time."""
    phases = []
    for result in results:
        record = result.get('metrics', {}).get('action_feedback')
        present = set(result.get('trace_candidates_present', []))
        # V14f, R6-11(d): every run whose interaction phase was traced counts, including one whose trace has no
        # event or no candidate at all (absence); load-only and untraced runs have no traced interaction phase.
        traced = result.get('interaction_traced')
        if traced is None:
            traced = bool(result.get('trace_events'))
        if not traced or not result.get('steps'):
            continue
        interactions = []
        if record and record.get('raf') is not None and record.get('screencast') is not None:
            check = {}
            for candidate, presented in (record.get('presented') or {}).items():
                check[candidate] = (presented is not None and presented >= record['raf']
                                    and abs(presented - record['screencast']) <= FRAME_MATCH_MS)
            interactions.append(check)
        phases.append({'present': present, 'interactions': interactions})
    return phases


def numeric_median(values: list):
    """Median for overhead arms; a censored ('<16 ms') median cannot enter a difference."""
    ordered = sorted_values(values)
    if not ordered:
        return None
    middle = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
    if any(value in (CENSORED, INF) for value in middle):
        return None
    return (middle[0] + middle[1]) / 2


WARM_NOT_EVALUABLE = 'no evaluable por este método: caché caliente no alcanzada'


def summary_rows(runs: list, frame_event: str | None, *, warm_evaluable: bool = True) -> list:
    """Per (viewport, view, cold/warm): n valid, failures and percentiles; labelled local, never C10/C11 acceptance.

    V1: failures count per metric (run_wide_failure, else the metric's own value). V10: when untraced-INP runs
    exist (instrumentation_overhead consequence), INP rows come from them only and LCP, CLS and C11 rows from the
    original runs. V9: warm rows are 'no evaluable' when the smoke's warm_cache outcome says so."""
    rows = []
    groups = defaultdict(list)
    for run in runs:
        groups[(run['viewport'], run['view'], run['role'])].append(run)
    for (viewport, view, role), members in sorted(groups.items()):
        original = [run for run in members if not run.get('untraced_inp')]
        untraced = [run for run in members if run.get('untraced_inp')]
        row = {'viewport': viewport, 'view': view, 'cache': role, 'label': LOCAL_LABEL,
               'n_valid': len(percentile_runs(original))}
        if view in FIXTURE_UNAVAILABLE_VIEWS:
            # R6-7: descriptive only; excluded from C10 acceptance and from before/after claims.
            row.update(fixture_note=FIXTURE_UNAVAILABLE_NOTE, c10_acceptance='excluido', before_after='excluido')
        if role == 'warm' and not warm_evaluable:
            row.update(status=WARM_NOT_EVALUABLE, lcp=WARM_NOT_EVALUABLE, cls=WARM_NOT_EVALUABLE,
                       inp=WARM_NOT_EVALUABLE if view in INTERACTION_VIEWS else LOAD_ONLY_INP)
            row['cache_mix_percent'] = warm_share(percentile_runs(original))['mix_percent']
            rows.append(row)
            continue
        for metric in ('lcp', 'cls'):
            row[metric] = summarize(metric_values(original, metric))
        if view in INTERACTION_VIEWS:
            source = untraced if untraced else original
            row['inp'] = summarize(metric_values(source, 'inp'))
            row['inp_source'] = 'untraced_interaction_phase' if untraced else 'traced_interaction_phase'
            row['n_valid_inp'] = len(percentile_runs(source))
        else:
            row['inp'] = LOAD_ONLY_INP
        if view == '#acciones':
            values = [c11_feedback_value(run, frame_event) for run in percentile_runs(original)]
            row['c11_action_feedback'] = summarize(values)
        if view == '#voces':
            values = [INF if run_wide_failure(run) else run['metrics'].get('preview', {}).get('value', INF)
                      for run in percentile_runs(original)]
            row['c11_preview'] = summarize(values)
        row['timed_exceptions'] = sum(run.get('timed_exceptions', 0) for run in percentile_runs(members))
        row['http_errors'] = sum(len(run.get('http_errors') or []) for run in percentile_runs(members))
        if role == 'warm':
            row['cache_mix_percent'] = warm_share(percentile_runs(original))['mix_percent']
        rows.append(row)
    return rows


def summarize_directory(directory: Path, mode: str) -> dict:
    """Summary of one batch directory; baseline mode refuses smoke results (run_rules.no_baseline_use)."""
    runs = []
    for path in sorted(directory.glob('run-*.json')):
        document = json.loads(path.read_text())
        if mode == 'baseline':
            refuse_smoke_input(document)
        document = restore_infinity(document)
        document['outcome'] = Outcome(document['outcome'])
        runs.append(document)
    fill_ins = load_fill_ins(json.loads(PREREGISTRATION.read_text())) if PREREGISTRATION.exists() else {}
    frame_event = (fill_ins.get('frame_event') or {}).get('candidate') if isinstance(fill_ins.get('frame_event'), dict) \
        else fill_ins.get('frame_event')
    return {'mode': mode, 'rows': json_safe(summary_rows(runs, frame_event, warm_evaluable=warm_rule_active(fill_ins))),
            'scope': 'NEXO only; no /obs/ row (Q29, QF); local numbers never accept C10/C11'}


async def smoke_overhead(sha: str, play_rule: dict) -> dict:
    """smoke_gate.instrumentation_overhead: #acciones and #voces desktop, warm runs only, 10 per arm, interleaved.

    V10: arm values follow the per-metric failure scope (invalid runs excluded, failures +inf, '<16 ms' kept),
    the consequences come from overhead_consequences() and the preview positions use the smoke preview start."""
    from tests.frontend.browser_support import browser_binary
    out_dir = checked_output('smoke', SMOKE_ROOT / f'overhead-{time.strftime("%Y%m%dT%H%M%S")}')
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fixture = FixtureProcess(out_dir)
    session = BrowserSession('smoke', browser_binary(), out_dir)
    values = {}
    try:
        ready = await fixture.start()
        positions = json.loads(Path(ready['position_map']).read_text())
        for view in ('#acciones', '#voces'):
            arms = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
            random.Random(derive_seed(sha, 'overhead', view, smoke=True)).shuffle(arms)
            start = derive_seed(sha, 'preview', 'overhead', view, smoke=True) % 50
            per_arm = {arm: {'inp': [], 'raf_feedback': []} for arm in 'abc'}
            for number, arm in enumerate(arms):
                capture = None
                if view == '#voces':
                    position = preview_position(2 * number + 1, positions['preview_permutation'], start)
                    capture = positions['positions'][position]['capture_id']
                base = dict(viewport='desktop', view=view, url=ready['origin'] + '/' + view, origin=ready['origin'],
                            token_for_login=ready['tokens'][VIEW_ROLES[view]], capture_id=capture)
                primer = RunSpec(2 * number, 'cold', primer=True, trace_interactions=False, screencast=False,
                                 play_rule=play_rule, **base)
                warm = RunSpec(2 * number + 1, 'warm', trace_interactions=arm in 'ab', screencast=arm == 'a',
                               play_rule=play_rule, **base)
                for spec in (primer, warm):
                    result = await session.execute(spec, fixture)
                    result.update(mode='smoke', arm=arm)
                    write_json_atomic(out_dir / f'{view.strip("#")}-run-{spec.ordinal:05d}.json', json_safe(result))
                if result['outcome'] == Outcome.INVALID:
                    continue
                wide = run_wide_failure(result)
                per_arm[arm]['inp'].append(INF if wide else result['metrics'].get('inp', INF))
                if view == '#acciones':
                    per_arm[arm]['raf_feedback'].append(INF if wide else result['metrics'].get('action_feedback_raf', INF))
            values[view] = per_arm
    finally:
        await session.close_browser()
        await fixture.stop()
    analysis = overhead_consequences(values)
    return {'directory': str(out_dir), 'values': json_safe(values), 'overhead': json_safe(analysis),
            'consequences': analysis['consequences'], 'play_control_rule': play_rule,
            'note': 'overhead per metric = median(a) - median(c) as an interval; censored medians are [0, 16)'}


def smoke_frame_event(root: Path = SMOKE_ROOT) -> dict:
    """smoke_gate.frame_event (V14f): the acceptance test over every smoke run whose interaction phase was traced
    (batches and overhead arm (a) and (b) runs alike); analysis only, no browser."""
    results = []
    for path in sorted(root.rglob('*run-*.json')):
        try:
            document = restore_infinity(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
        if document.get('mode') != 'smoke' or document.get('outcome') == Outcome.INVALID.value:
            continue
        results.append(document)
    phases = frame_event_phases(results)
    return {'phases': len(phases), 'choice': frame_event_choice(phases)}


# --------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    dry = sub.add_parser('harness-dry-run', help='functional dry run on server headless Chromium (not a measurement)')
    dry.add_argument('--out')
    dry.add_argument('--view', choices=VIEW_HASHES)
    for name in ('measure-cell', 'measure-client'):
        measure = sub.add_parser(name, help='baseline (frozen hash) or smoke (--smoke, reviewed hash) on the Mac')
        measure.add_argument('--smoke', action='store_true')
        measure.add_argument('--reviewed-sha256')
        measure.add_argument('--frozen-sha256')
        measure.add_argument('--cause', choices=SMOKE_REPEAT_CAUSES)
        measure.add_argument('--viewport', choices=tuple(VIEWPORTS), default='desktop')
        measure.add_argument('--view', choices=VIEW_HASHES, default='#hogar')
        measure.add_argument('--attempt', type=int, default=1)
        # R6-2: no --pairs/--contrast-pairs: the composition is fixed per mode (BATCH_COMPOSITION).
    smoke = sub.add_parser('smoke-item', help='smoke items that are not batches (Mac, smoke mode)')
    smoke.add_argument('item', choices=('quiet-baseline', 'cold-focus', 'background', 'lock-read', 'fixture-pages',
                                        'overhead', 'frame-event'))
    smoke.add_argument('--reviewed-sha256', required=True)
    smoke.add_argument('--cause', choices=SMOKE_REPEAT_CAUSES)
    smoke.add_argument('--lock-state', choices=('locked', 'unlocked'), default='unlocked')
    summary = sub.add_parser('summarize', help='summary rows of one batch directory')
    summary.add_argument('directory')
    summary.add_argument('--mode', choices=('baseline', 'smoke'), required=True)
    sub.add_parser('identity', help='print the script id and the batch plan cells')
    return parser.parse_args(argv)


def check_measure_mode(args) -> tuple:
    """Hash verification at start: smoke against the reviewed hash, baseline against the frozen hash."""
    if args.smoke:
        return 'smoke', verify_preregistration(PREREGISTRATION, args.reviewed_sha256)
    return 'baseline', verify_preregistration(PREREGISTRATION, args.frozen_sha256)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'identity':
        print(json.dumps({'script_id': script_id(), 'cells': batch_plan_views()}, indent=1, ensure_ascii=False))
        return 0
    if args.command == 'harness-dry-run':
        return asyncio.run(harness_dry_run(args))
    if args.command == 'summarize':
        print(json.dumps(summarize_directory(Path(args.directory), args.mode), indent=1, ensure_ascii=False))
        return 0
    if args.command == 'smoke-item':
        if sys.platform != 'darwin':
            raise SystemExit('smoke_runs_only_on_the_mac')
        sha = verify_preregistration(PREREGISTRATION, args.reviewed_sha256)
        out_dir = checked_output('smoke', SMOKE_ROOT / 'items')
        out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = SmokeRecord(checked_output('smoke', SMOKE_ROOT / 'smoke-record.json'))
        if args.item == 'quiet-baseline':
            result = asyncio.run(smoke_quiet_baseline(record, sha, args.cause))
        elif args.item == 'lock-read':
            result = smoke_lock_read(record, sha, args.cause, args.lock_state)
        elif args.item == 'frame-event':
            check_smoke_order(record, 'frame_event')
            entry = record.start('frame_event', sha, args.cause)
            result = smoke_frame_event()
            record.finish(entry, 'complete', json_safe(result))
        else:
            async def fixture_pages():
                from tests.frontend.browser_support import browser_binary
                pages_dir = checked_output('smoke', SMOKE_ROOT / f'fixture-pages-{time.strftime("%Y%m%dT%H%M%S")}')
                pages_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                summary = await fixture_page_checks('smoke', pages_dir, binary=browser_binary(), profile_root=None)
                return {'ok': summary['ok'], 'directory': str(pages_dir), 'checks': summary['checks'],
                        'observations': summary['observations']}
            bodies = {'cold-focus': ('cold_focus', lambda: smoke_cold_focus(out_dir)),
                      'background': ('background_activity', lambda: smoke_background(out_dir)),
                      'fixture-pages': ('fixture_pages', fixture_pages),
                      'overhead': ('instrumentation_overhead', lambda: smoke_overhead(sha, smoke_play_rule(record)))}
            item, body = bodies[args.item]
            check_smoke_order(record, item)
            if item == 'instrumentation_overhead':
                smoke_play_rule(record)
            result = asyncio.run(run_smoke_item(record, item, sha, args.cause, body))
        print(json.dumps(json_safe(result), indent=1))
        return 0
    mode, sha = check_measure_mode(args)
    if args.command == 'measure-client':
        return asyncio.run(measure_client(args, mode, sha))
    if mode == 'smoke':
        record = SmokeRecord(checked_output('smoke', SMOKE_ROOT / 'smoke-record.json'))
        cell_settings('smoke', {}, record)
        entry = record.start(f'batch:{args.viewport}:{args.view}', sha, args.cause)
        try:
            report = asyncio.run(measure_cell(args, mode, sha))
        except BaseException as error:
            record.finish(entry, 'aborted', {'error': f'{type(error).__name__}: {str(error)[:200]}'}, faults=['crash'])
            raise
        record.finish(entry, 'discarded' if report.get('discarded') else 'complete',
                      {'directory': report.get('directory'), 'reasons': report.get('reasons'),
                       'composition': report.get('composition'), 'play_control_rule': report.get('play_control_rule')},
                      faults=report.get('faults'))
        return 0
    asyncio.run(measure_cell(args, mode, sha))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
