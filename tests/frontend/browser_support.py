"""Shared browser and evidence location helpers for the manual browser harnesses.

Linux keeps the historical ``chromium`` binary and its snap-confined profile
root. macOS uses Google Chrome.app, whose profile may live in the default
temporary directory. Evidence defaults to a fresh run-scoped directory so a new
run never writes into retained historical evidence.

``audit_view`` adds the C08 reflow widths and the C07 keyboard, focus, name,
touch-target and language checks for one NEXO view, using only the Chrome
DevTools Protocol and page JavaScript.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
MAC_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def browser_binary(override_variable: str = 'KZA_BROWSER_BINARY',
                   missing: str = 'browser_binary_missing') -> str:
    """Select an owned headless browser without changing the Linux default.

    Args:
        override_variable: Environment variable holding an absolute browser path.
        missing: RuntimeError message raised when no usable browser is found.

    Returns:
        ``chromium`` on Linux without override, otherwise an absolute executable path.
    """
    override = os.environ.get(override_variable)
    try:
        if override is None and sys.platform != 'darwin':
            if shutil.which('chromium') is not None:
                return 'chromium'
        else:
            selected = Path(override if override is not None else MAC_CHROME)
            if selected.is_absolute() and selected.is_file() and os.access(selected, os.X_OK):
                return str(selected)
    except (OSError, ValueError):
        pass
    raise RuntimeError(missing)


def browser_profile_root(mode: int = 0o777) -> Path | None:
    """Return the temporary-profile parent the selected browser can access.

    Snap Chromium can only read profiles under ``~/snap/chromium/common``; on
    macOS ``None`` keeps ``tempfile``'s default location.
    """
    if sys.platform == 'darwin':
        return None
    root = Path.home() / 'snap/chromium/common'
    root.mkdir(mode=mode, parents=True, exist_ok=True)
    return root


def evidence_directory(variable: str, default_parent: str) -> Path:
    """Resolve the evidence directory for one browser run.

    Args:
        variable: Environment variable that names an explicit evidence directory.
        default_parent: Repository-relative parent for the default run directory.

    Returns:
        The explicit directory, or a new ``run-<ns>`` directory under ``default_parent``.
    """
    explicit = os.environ.get(variable)
    if explicit:
        return Path(explicit)
    return ROOT / default_parent / f'run-{time.time_ns()}'


async def install_login_session(page, base_url: str, token: str) -> None:
    """Sign in through the real login route and hand its opaque session to the browser.

    The dashboard must run with a ``DashboardAuthority``. The server issues the
    session cookie; the raw administrative token is never stored in the browser.

    Args:
        page: CDP page client with a ``command`` coroutine.
        base_url: Origin of the dashboard under test, without a trailing slash.
        token: Synthetic administrative token configured for the fixture.
    """
    import aiohttp
    from src.dashboard.auth import COOKIE_NAME

    async with aiohttp.ClientSession() as client:
        async with client.post(f'{base_url}/api/admin/auth/login', json={'token': token}) as response:
            if response.status != 200:
                raise RuntimeError(f'login_failed:{response.status}')
            morsel = response.cookies.get(COOKIE_NAME)
    if morsel is None or not morsel.value or morsel.value == token:
        raise RuntimeError('login_session_cookie_missing')
    await page.command('Network.enable')
    await page.command('Network.setCookie', {'name': COOKIE_NAME, 'value': morsel.value, 'url': base_url + '/',
                                             'httpOnly': True, 'sameSite': 'Strict'})


# C08 reflow widths plus the historical 1425 px desktop width kept by the harnesses.
C08_WIDTHS = (320, 390, 768, 1280, 1425, 1440)
TOUCH_TARGET_WIDTHS = frozenset({320, 390})
TOUCH_TARGET_MIN_PX = 44
VIEWPORT_HEIGHTS = {320: 760, 390: 844, 768: 1024, 1280: 800, 1425: 1180, 1440: 900}
INTERACTIVE_ROLES = frozenset({
    'button', 'link', 'textbox', 'searchbox', 'combobox', 'listbox', 'checkbox', 'radio',
    'slider', 'spinbutton', 'switch', 'tab', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
})

AUDIT_SCRIPT = r"""(() => {
  if (window.__kzaAudit) return true;
  const FOCUSABLE = 'a[href],area[href],button,input:not([type=hidden]),select,textarea,summary,iframe,audio[controls],video[controls],[contenteditable]:not([contenteditable=false]),[tabindex]';
  const STYLE_KEYS = ['outlineStyle','outlineWidth','outlineColor','boxShadow','borderTopColor','borderBottomColor','backgroundColor','color','textDecorationLine'];
  const shown = el => el.getClientRects().length > 0 && el.checkVisibility({visibilityProperty: true, opacityProperty: true});
  const describe = el => {
    if (!el || el === document.body || el === document.documentElement) return el ? el.tagName.toLowerCase() : 'none';
    let text = el.tagName.toLowerCase();
    if (el.id) text += '#' + el.id;
    const classes = [...el.classList].slice(0, 2);
    if (classes.length) text += '.' + classes.join('.');
    for (const name of ['data-view','data-layer','data-operation','data-room-id','data-person','name','type','aria-label']) {
      if (el.hasAttribute(name)) text += '[' + name + '="' + el.getAttribute(name).slice(0, 32) + '"]';
    }
    const label = (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40);
    return label ? text + ' "' + label + '"' : text;
  };
  const scope = () => document.querySelector('dialog[open]') || document;
  const owns = (el, hit) => !!hit && (hit === el || el.contains(hit) || [...(el.labels || [])].some(label => label.contains(hit)));
  const controls = () => [...scope().querySelectorAll(FOCUSABLE)].filter(el => el.tabIndex >= 0 && !el.closest('[inert]') && shown(el));
  const tabbables = () => {
    const groups = new Set();
    return controls().filter(el => !el.disabled).filter(el => {
      if (el.type !== 'radio' || !el.name) return true;
      const group = [...document.querySelectorAll('input[type=radio]')].filter(item => item.name === el.name && item.form === el.form && !item.disabled && shown(item));
      const chosen = group.find(item => item.checked) || group[0];
      if (chosen !== el || groups.has(el.name)) return false;
      groups.add(el.name);
      return true;
    });
  };
  const snapshot = el => { if (!el) return ''; const style = getComputedStyle(el); return STYLE_KEYS.map(key => style[key]).join('|'); };
  const transparent = color => /^(transparent|rgba\([^)]*,\s*0\))$/.test(color);
  const clipBox = el => {
    let box = {left: 0, top: 0, right: document.documentElement.clientWidth, bottom: innerHeight};
    for (let node = el.parentElement; node && node !== document.documentElement; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (style.overflowX === 'visible' && style.overflowY === 'visible') continue;
      const rect = node.getBoundingClientRect();
      box = {left: Math.max(box.left, rect.left), top: Math.max(box.top, rect.top), right: Math.min(box.right, rect.right), bottom: Math.min(box.bottom, rect.bottom)};
    }
    return box;
  };
  const outlineVisible = el => {
    const style = getComputedStyle(el), width = parseFloat(style.outlineWidth) || 0, offset = parseFloat(style.outlineOffset) || 0;
    if (style.outlineStyle === 'none' || width < 1 || transparent(style.outlineColor)) return false;
    const rect = el.getBoundingClientRect(), clip = clipBox(el);
    const inner = {left: rect.left - offset, top: rect.top - offset, right: rect.right + offset, bottom: rect.bottom + offset};
    const outer = {left: inner.left - width, top: inner.top - width, right: inner.right + width, bottom: inner.bottom + width};
    const intersects = clip.left < outer.right && clip.right > outer.left && clip.top < outer.bottom && clip.bottom > outer.top;
    const contained = clip.left >= inner.left && clip.right <= inner.right && clip.top >= inner.top && clip.bottom <= inner.bottom;
    return intersects && !contained;
  };
  const unobscured = el => {
    const rect = el.getBoundingClientRect();
    const points = [[rect.left + rect.width / 2, rect.top + rect.height / 2], [rect.left + 2, rect.top + 2], [rect.right - 2, rect.top + 2], [rect.left + 2, rect.bottom - 2], [rect.right - 2, rect.bottom - 2]];
    return points.some(([x, y]) => x >= 0 && y >= 0 && x < document.documentElement.clientWidth && y < innerHeight && owns(el, document.elementFromPoint(x, y)));
  };
  window.__kzaAudit = {
    describe,
    language() {
      const foreign = [...document.querySelectorAll('[lang]')].filter(el => !/^es(-|$)/i.test(el.getAttribute('lang'))).map(describe);
      return {lang: document.documentElement.getAttribute('lang'), foreign};
    },
    overflow() {
      const width = document.documentElement.clientWidth, offenders = [], flagged = new Set();
      for (const el of document.body.querySelectorAll('*')) {
        if (!shown(el)) continue;
        const rect = el.getBoundingClientRect();
        if (!rect.width || (rect.right <= width + 1 && rect.left >= -1)) continue;
        if ([...flagged].some(parent => parent.contains(el))) continue;
        if (getComputedStyle(el).position !== 'fixed') {
          // Content inside a clipping or scrolling container that itself fits does not reflow the page.
          let clipped = false;
          for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
            const style = getComputedStyle(node);
            if (style.overflowX !== 'visible') { const box = node.getBoundingClientRect(); clipped = box.right <= width + 1 && box.left >= -1; break; }
          }
          if (clipped) continue;
        }
        flagged.add(el);
        offenders.push({element: describe(el), left: Math.round(rect.left), right: Math.round(rect.right)});
      }
      return {scrollWidth: document.documentElement.scrollWidth, innerWidth, offenders: offenders.slice(0, 20)};
    },
    keyControls(specs) {
      const width = document.documentElement.clientWidth;
      const results = specs.map(([selector, text]) => {
        const el = [...document.querySelectorAll(selector)].find(node => !text || (node.textContent || '').includes(text));
        const label = selector + (text ? ' "' + text + '"' : '');
        if (!el) return {control: label, problem: 'missing'};
        if (!shown(el)) return {control: label, problem: 'not visible'};
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') return {control: label, problem: 'disabled'};
        el.scrollIntoView({block: 'center', inline: 'nearest'});
        const rect = el.getBoundingClientRect();
        if (rect.left < -1 || rect.right > width + 1) return {control: label, problem: 'outside the viewport horizontally', left: rect.left, right: rect.right};
        const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
        if (!owns(el, hit)) return {control: label, problem: 'covered by ' + describe(hit)};
        return {control: label, problem: null};
      });
      window.scrollTo(0, 0);
      return results;
    },
    targets(minimum) {
      return controls().filter(el => el.getBoundingClientRect().bottom + scrollY > 0).map(el => {
        const rects = [el.getBoundingClientRect(), ...[...(el.labels || [])].filter(label => label.contains(el)).map(label => label.getBoundingClientRect())];
        const width = Math.max(...rects.map(rect => rect.width)), height = Math.max(...rects.map(rect => rect.height));
        return {element: describe(el), width: Math.round(width * 10) / 10, height: Math.round(height * 10) / 10};
      }).filter(item => item.width < minimum - 0.05 || item.height < minimum - 0.05);
    },
    prepareKeyboard() {
      const expected = tabbables();
      const before = new Map();
      for (const el of expected) for (const node of [el, el.parentElement, el.parentElement?.parentElement]) if (node && !before.has(node)) before.set(node, snapshot(node));
      window.__kzaKeyboard = {expected, before};
      if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
      window.scrollTo(0, 0);
      return {expected: expected.map(describe), positive: [...document.querySelectorAll('[tabindex]')].filter(el => el.tabIndex > 0).map(describe)};
    },
    step() {
      const el = document.activeElement, state = window.__kzaKeyboard;
      if (!el || el === document.body || el === document.documentElement) return {index: -1, element: null};
      let index = state.expected.indexOf(el);
      if (index < 0) {
        // A periodic catalog refresh may replace a control with an identical one; match it by its signature.
        index = state.expected.findIndex(old => !old.isConnected && describe(old) === describe(el));
        if (index >= 0) {
          const old = state.expected[index];
          for (const [current, previous] of [[el, old], [el.parentElement, old.parentElement], [el.parentElement?.parentElement, old.parentElement?.parentElement]]) {
            if (current && previous && !state.before.has(current) && state.before.has(previous)) state.before.set(current, state.before.get(previous));
          }
          state.expected[index] = el;
        }
      }
      const changed = [el, el.parentElement, el.parentElement?.parentElement].some(node => node && state.before.has(node) && snapshot(node) !== state.before.get(node));
      const rect = el.getBoundingClientRect();
      const inMain = !!el.closest('main') && getComputedStyle(el).position !== 'fixed';
      return {index, element: describe(el), focusVisible: el.matches(':focus-visible'), indicator: outlineVisible(el) || changed,
              unobscured: unobscured(el), shown: shown(el), inMain, top: rect.top + scrollY, bottom: rect.bottom + scrollY, left: rect.left};
    },
  };
  return true;
})()"""


async def _press_tab(page) -> None:
    for kind in ('rawKeyDown', 'keyUp'):
        await page.command('Input.dispatchKeyEvent', {'type': kind, 'key': 'Tab', 'code': 'Tab',
                                                      'windowsVirtualKeyCode': 9, 'nativeVirtualKeyCode': 9})


async def _nameless_controls(page) -> list[str]:
    """Return interactive accessibility-tree nodes whose computed name is empty."""
    tree = await page.command('Accessibility.getFullAXTree')
    nameless = []
    for item in tree.get('nodes', []):
        role = (item.get('role') or {}).get('value')
        if item.get('ignored') or role not in INTERACTIVE_ROLES:
            continue
        if str((item.get('name') or {}).get('value') or '').strip():
            continue
        described = 'unknown node'
        if 'backendDOMNodeId' in item:
            node = (await page.command('DOM.describeNode', {'backendNodeId': item['backendDOMNodeId']}))['node']
            attributes = node.get('attributes', [])
            described = node['nodeName'].lower() + ' ' + json.dumps(dict(zip(attributes[::2], attributes[1::2])))
        nameless.append(f'{role}: {described}')
    return nameless


async def _keyboard(page, width: int) -> tuple[dict, list[str]]:
    """Traverse the view with real Tab key events and assert reachability, order and focus visibility."""
    prepared = await page.evaluate('window.__kzaAudit.prepareKeyboard()')
    expected = prepared['expected']
    findings = [f'positive tabindex disturbs the order: {item}' for item in prepared['positive']]
    steps = []
    seen = set()
    for _ in range(2 * len(expected) + 4):
        await _press_tab(page)
        step = await page.evaluate('window.__kzaAudit.step()')
        if step['index'] >= 0 and step['index'] in seen:
            break
        if step['index'] >= 0:
            seen.add(step['index'])
        steps.append(step)
    await page.evaluate("document.activeElement&&document.activeElement.blur();window.scrollTo(0,0)")
    focused = [step for step in steps if step['element'] is not None]
    for step in focused:
        if step['index'] < 0:
            findings.append(f'focus reached a control outside the visible tab sequence: {step["element"]}')
            continue
        if not step['focusVisible']:
            findings.append(f'keyboard focus does not match :focus-visible: {step["element"]}')
        if not step['indicator']:
            findings.append(f'no visible focus indicator: {step["element"]}')
        if not step['unobscured']:
            findings.append(f'focused control is entirely hidden or obscured: {step["element"]}')
    indices = [step['index'] for step in focused if step['index'] >= 0]
    if 0 in indices:
        start = indices.index(0)
        indices = indices[start:] + indices[:start]
    missing = sorted(set(range(len(expected))) - set(indices))
    findings += [f'not reachable with Tab: {expected[index]}' for index in missing]
    if not missing and indices != list(range(len(expected))):
        findings.append('Tab order differs from the document order: ' + json.dumps(indices))
    if width in TOUCH_TARGET_WIDTHS:
        ordered = [step for step in focused if step['index'] >= 0 and step['inMain']]
        ordered.sort(key=lambda step: (step['index'] - indices[0]) % max(len(expected), 1))
        for previous, current in zip(ordered, ordered[1:]):
            if current['bottom'] <= previous['top']:
                findings.append(f'single-column focus order moves upward: {previous["element"]} -> {current["element"]}')
    return {'expected': len(expected), 'focused': len(focused), 'sequence': [step['element'] for step in focused]}, findings


async def audit_view(page, evidence: Path, view: str, key_controls: Sequence[tuple[str, str | None]],
                     screenshot: Callable[..., Awaitable[None]], *, widths: Sequence[int] = C08_WIDTHS,
                     restore: tuple[int, int, bool] = (1425, 1180, False)) -> dict:
    """Check one NEXO view at every C08 width and record C07 accessibility findings.

    Per width: no horizontal overflow (document and visible elements), key
    controls visible, enabled and hit-testable, a screenshot, accessible names
    from the browser accessibility tree, a real Tab traversal (reachability,
    document order, :focus-visible, a visible indicator, not obscured) and, at
    320/390 px, 44x44 CSS px touch targets. The page language must be Spanish.

    Args:
        page: Connected CDP page exposing ``command`` and ``evaluate``.
        evidence: Run evidence directory receiving screenshots and ``a11y-<view>.json``.
        view: Short view name used in evidence file names.
        key_controls: ``(selector, text)`` pairs; ``text`` narrows by visible text when given.
        screenshot: Harness screenshot coroutine ``(page, path, full=..., scroll_top=...)``.
        widths: CSS pixel widths to audit.
        restore: Viewport ``(width, height, mobile)`` applied after the audit.

    Returns:
        The per-width measurements written to the evidence file.

    Raises:
        AssertionError: When any check fails; every finding is recorded first.
    """
    await page.command('Accessibility.enable')
    await page.evaluate(AUDIT_SCRIPT)
    language = await page.evaluate('window.__kzaAudit.language()')
    findings = []
    if language['lang'] != 'es':
        findings.append(f'html lang is {language["lang"]!r}, expected "es"')
    findings += [f'non-Spanish lang attribute: {item}' for item in language['foreign']]
    report = {'view': view, 'language': language, 'widths': {}}
    for width in widths:
        await page.command('Emulation.setDeviceMetricsOverride', {'width': width, 'height': VIEWPORT_HEIGHTS.get(width, 900),
                                                                  'deviceScaleFactor': 1, 'mobile': width <= 390})
        await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
        await asyncio.sleep(0.1)
        await page.evaluate(AUDIT_SCRIPT)
        local = []
        overflow = await page.evaluate('window.__kzaAudit.overflow()')
        if overflow['scrollWidth'] > overflow['innerWidth']:
            local.append(f'document scrolls horizontally: scrollWidth {overflow["scrollWidth"]} > {overflow["innerWidth"]}')
        local += [f'element overflows the viewport: {item}' for item in overflow['offenders']]
        controls = await page.evaluate('window.__kzaAudit.keyControls(' + json.dumps([list(item) for item in key_controls]) + ')')
        local += [f'key control {item["control"]}: {item["problem"]}' for item in controls if item['problem']]
        shot = evidence / f'{view}-{width}.png'
        await screenshot(page, shot, full=True)
        nameless = await _nameless_controls(page)
        local += [f'control without accessible name: {item}' for item in nameless]
        small = []
        if width in TOUCH_TARGET_WIDTHS:
            small = await page.evaluate(f'window.__kzaAudit.targets({TOUCH_TARGET_MIN_PX})')
            local += [f'touch target below {TOUCH_TARGET_MIN_PX}x{TOUCH_TARGET_MIN_PX}: {item}' for item in small]
        keyboard, keyboard_findings = await _keyboard(page, width)
        local += keyboard_findings
        report['widths'][str(width)] = {'screenshot': shot.name, 'overflow': overflow, 'key_controls': controls,
                                        'small_targets': small, 'keyboard': keyboard, 'findings': local}
        findings += [f'{width}px: {item}' for item in local]
    await page.command('Emulation.setDeviceMetricsOverride', {'width': restore[0], 'height': restore[1],
                                                              'deviceScaleFactor': 1, 'mobile': restore[2]})
    report['findings'] = findings
    report['passed'] = not findings
    path = evidence / f'a11y-{view}.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    assert not findings, f'NEXO {view} reflow/accessibility findings ({len(findings)}), see {path.name}: ' + '; '.join(findings[:12])
    return report


async def reduced_motion_house(page, settle: Callable[[], Awaitable[None]]) -> dict:
    """Assert the illustrative 3D house honours prefers-reduced-motion.

    With ``reduce`` emulated, the house is remounted by ``settle`` and a camera
    change must reach its final frame at once while no CSS animation or
    transition runs. The same gesture without ``reduce`` must still be eased,
    proving the comparison can detect motion.

    Args:
        page: Connected CDP page.
        settle: Coroutine that remounts the home view and waits for the model.

    Returns:
        Observed values for the evidence file.
    """
    observed = {}
    for preference in ('reduce', 'no-preference'):
        await page.command('Emulation.setEmulatedMedia', {'features': [{'name': 'prefers-reduced-motion', 'value': preference}]})
        await settle()
        assert await page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches") is (preference == 'reduce')
        await page.evaluate("document.querySelector('#view-3d').click()")
        await asyncio.sleep(1.2)
        await page.evaluate("window.__kzaFrame0=document.querySelector('#house-canvas canvas').toDataURL()")
        await page.evaluate("document.querySelector('#view-top').click()")
        await asyncio.sleep(0.15)
        await page.evaluate("window.__kzaFrame1=document.querySelector('#house-canvas canvas').toDataURL()")
        await asyncio.sleep(1.2)
        result = await page.evaluate("""(()=>{const last=document.querySelector('#house-canvas canvas').toDataURL();
          const longest=value=>Math.max(...value.split(',').map(parseFloat));
          const durations=[...document.querySelectorAll('*')].filter(el=>el.getClientRects().length).flatMap(el=>{const style=getComputedStyle(el);
            const animated=(style.animationName!=='none'&&longest(style.animationDuration)>0)||(style.transitionProperty!=='none'&&longest(style.transitionDuration)>0);
            return animated?[window.__kzaAudit.describe(el)]:[]});
          return {moved:window.__kzaFrame1!==window.__kzaFrame0,settledEarly:window.__kzaFrame1===last,
                  running:document.getAnimations().filter(item=>item.playState==='running').length,animated:durations.slice(0,10)}})()""")
        observed[preference] = result
        if preference == 'reduce':
            assert result['moved'], 'Reduced motion: the camera change did not render'
            assert result['settledEarly'], 'Reduced motion: the 3D camera still eases after the first frame'
            assert result['running'] == 0 and not result['animated'], f'Reduced motion: CSS motion remains {result}'
        else:
            assert not result['settledEarly'], 'Default motion: the camera is expected to ease, so the reduced check is meaningful'
        await page.evaluate("document.querySelector('#reset-view').click()")
    await page.command('Emulation.setEmulatedMedia', {'features': []})
    return observed
