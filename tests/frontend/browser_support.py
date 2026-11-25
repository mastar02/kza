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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
import errno
import json
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
MAC_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


class Page:
    """Small sequential CDP client using an existing development dependency.

    Moved here from the retired legacy ``browser_typed_controls`` harness so
    every browser script shares one implementation.
    """

    def __init__(self, connection):
        self.connection = connection
        self.sequence = 0
        self.errors = []

    async def command(self, method, params=None):
        """Send one CDP command and return its result, recording page exceptions."""
        self.sequence += 1
        identity = self.sequence
        await self.connection.send(json.dumps({"id": identity, "method": method, "params": params or {}}))
        while True:
            result = json.loads(await asyncio.wait_for(self.connection.recv(), 15))
            if result.get("method") == "Runtime.exceptionThrown":
                self.errors.append(result["params"])
            elif result.get("method") == "Network.requestWillBeSent":
                self.on_request(result["params"]["request"])
            if result.get("id") == identity:
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result.get("result", {})

    def on_request(self, request):
        """Observe a CDP request when a harness enables the Network domain."""

    async def evaluate(self, expression):
        """Evaluate an expression in the page and return its value."""
        result = await self.command("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    async def wait_for(self, expression):
        """Poll a page condition until it is truthy or fail after about five seconds."""
        for _ in range(100):
            if await self.evaluate(expression):
                return
            await asyncio.sleep(0.05)
        raise AssertionError("Browser condition timed out: " + expression)


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


@asynccontextmanager
async def temporary_browser_profile(prefix: str) -> AsyncIterator[Path]:
    """Create a profile and retry cleanup if Chromium finishes writing during removal."""
    directory = tempfile.TemporaryDirectory(prefix=prefix, dir=browser_profile_root())
    try:
        yield Path(directory.name)
    finally:
        for delay in (0.05, 0.1, 0.2):
            try:
                await asyncio.to_thread(directory.cleanup)
                break
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
                await asyncio.sleep(delay)
        else:
            await asyncio.to_thread(directory.cleanup)


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


class FixtureRows:
    """Find synthetic records by visible copy and give the harness stable CSS locators.

    The attribute contains only a test-local counter. Each lookup must match a
    single row. Reinstall the matcher on reload for the refresh/restart tests.
    """

    def __init__(self, root: str, label: str):
        self.root = root
        self.label = label
        self.keys: dict[str, str] = {}

    def selector(self, identity: str, inner: str = '') -> str:
        local = self.keys.setdefault(identity, f'fixture-{len(self.keys)}')
        return f'[data-fixture-row="{local}"] {inner}'.strip()

    async def bind(self, page: Page, identity: str, text: str, *, prefix: bool = False) -> None:
        await self.bind_many(page, [(identity, text)], prefix=prefix)

    async def bind_many(self, page: Page, labels: Sequence[tuple[str, str]], *, prefix: bool = False) -> None:
        rows = [(self.keys.setdefault(identity, f'fixture-{len(self.keys)}'), label) for identity, label in labels]
        source = r"""((selector, label, rows, prefix) => {
          const refresh = () => {
            const entries = [...document.querySelectorAll(selector)].map(row => [row, row.querySelector(label)?.textContent]);
            for (const [local, text] of rows) {
              const matches = entries.filter(([, title]) => prefix ? title?.startsWith(text) : title === text);
              if (matches.length > 1) throw new Error('Ambiguous visible fixture label: ' + text);
              const next = matches[0]?.[0];
              const previous = document.querySelector('[data-fixture-row="' + local + '"]');
              if (previous && previous !== next) previous.removeAttribute('data-fixture-row');
              if (next && next !== previous) next.setAttribute('data-fixture-row', local);
            }
          };
          new MutationObserver(refresh).observe(document, {subtree: true, childList: true, characterData: true});
          refresh();
        })(%s, %s, %s, %s)""" % tuple(map(json.dumps, (self.root, self.label, rows, prefix)))
        await page.command('Page.addScriptToEvaluateOnNewDocument', {'source': source})
        await page.evaluate(source)

async def choose_option(page: Page, selector: str, label: str) -> None:
    """Choose one rendered option by its label and dispatch the native change."""
    await page.evaluate(r"""((selector, label) => {
      const select = document.querySelector(selector);
      const choices = [...select.options].filter(option => option.textContent === label);
      if (choices.length !== 1) throw new Error('Expected one option for ' + label);
      select.value = choices[0].value;
      select.dispatchEvent(new Event('change', {bubbles: true}));
    })(%s, %s)""" % (json.dumps(selector), json.dumps(label)))


_UUID = r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}'
_CANONICAL_ID = (rf'(?:{_UUID}|(?<![0-9a-f])(?:[0-9a-f]{{32}}|[0-9a-f]{{12}})(?![0-9a-f])|'
                 r'\b(?:fact|vf|user|routine|run|proposal|profile|capture|group)_[a-z0-9-]+\b)')
_HA_ENTITY_ID = (
    r'\b(?:light|switch|climate|cover|scene|sensor|binary_sensor|input_boolean|'
    r'automation|script|button|fan|media_player|lock|device_tracker|weather|camera|number|select|'
    r'input_number|input_select|vacuum|water_heater|humidifier|alarm_control_panel|person)\.[a-z0-9_]+\b'
)
_CANONICAL_ATTRIBUTES = re.compile(rf'{_CANONICAL_ID}|{_HA_ENTITY_ID}', re.IGNORECASE)
# Escalera is also a display name. Match the lower-case ID as supplied by the fixture.
_FIXTURE_ROOM_IDS = re.compile(r'\b(?:living|kitchen|escalera|annex)\b')
_INTERNAL_COPY = re.compile(
    rf'{_CANONICAL_ID}|{_HA_ENTITY_ID}|'
    r'\b[a-z][a-z0-9_]+\.[a-z][a-z0-9_]*\b|'
    r'\b(?:backends?|estado\s+observado|entidad(?:es)?|cat[aá]logos?|owners?|endpoints?|tokens?|(?-i:HA)|'
    r'loading|save|cancel|submit|delete|edit|refresh|retry|dashboard|login|logout|'
    r'success|failed|timeout|fallback|unknown|pending|score|settings|close|confirm|next|back|closed|open|off|on)\b|'
    r'\bReferencia:\s*[0-9a-f]{8}\b|\bReferencia\s+\d+\b',
    re.IGNORECASE,
)

# Exact text and DOM predicates for product exceptions. Only the matching
# fragments are removed from the scan. An identical string elsewhere is still tested.
_COPY_EXCEPTIONS = r"""root => {
  const allowed = new Set(), problems = [];
  const accept = (node, pattern) => {
    const text = node?.textContent.trim();
    if (text && pattern.test(text) && root.contains(node)) allowed.add(node);
  };
  const normalize = name => name.normalize('NFD').replace(/\p{M}/gu, '').trim().replace(/\s+/g, ' ').toLocaleLowerCase();
  const choices = (select, options, parse, reference) => {
    if (!select?.checkVisibility() || !root.contains(select)) return;
    const entries = options.map(option => [option, parse(option.textContent)]);
    for (const [option, item] of entries) {
      const siblings = entries.filter(([, other]) => normalize(other.name) === normalize(item.name)).length;
      if (/ · Referencia\b/.test(option.textContent) && !item.ref) problems.push('Formato inválido de la referencia de opción');
      if (Boolean(item.ref) !== (siblings > 1)) problems.push('Referencia de opción fuera de su condición de homónimos');
      if (item.ref && siblings > 1) allowed.add(option);
    }
    const selected = entries.find(([option]) => option.selected)?.[1];
    const count = selected ? entries.filter(([, item]) => normalize(item.name) === normalize(selected.name)).length : 0;
    const expected = reference.format(selected?.ref, count);
    if ((reference.node?.textContent.trim() ?? '') !== expected) problems.push('La referencia de la persona seleccionada no coincide');
    if (expected && reference.node?.textContent.trim() === expected) allowed.add(reference.node);
  };
  // Canonical reference required to distinguish homonyms while linking a voice.
  const voice = document.querySelector('#voice-person-choice');
  const accountId = '(?:user_[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})';
  const voicePattern = new RegExp('^(.+) · Referencia: (' + accountId + ')$');
  choices(voice, [...voice?.options ?? []].filter(option => /^\d+$/.test(option.value)), text => {
    const match = text.match(voicePattern);
    return {name: match?.[1] ?? text, ref: match?.[2]};
  }, {node: document.querySelector('#voice-person-reference'),
      format: (id, count) => id ? `Hay ${count} personas con este nombre. Confirmá por la referencia: ${id}` : ''});
  // Local numeric references in the access person picker only distinguish homonyms.
  const access = document.querySelector('#sistema-person-reference')?.previousElementSibling?.querySelector('select');
  choices(access, [...access?.options ?? []].filter(option => /^\d+$/.test(option.value)), text => {
    const match = text.match(/^(.+) · (Referencia \d+)$/);
    return {name: match?.[1] ?? text, ref: match?.[2]};
  }, {node: document.querySelector('#sistema-person-reference'), format: value => value ?? ''});
  // Local credential references are shown only when all four displayed fields collide.
  const accessRows = [...document.querySelectorAll('#access-view [data-credential-id] .row-main')];
  for (const row of accessRows) {
    const fields = copy => [...[...copy.children].slice(0, 3).map(child => normalize(child.textContent)),
      copy.parentElement.querySelector('.tag')?.textContent];
    const duplicates = accessRows.filter(other => fields(other).every((field, i) => field === fields(row)[i]));
    const note = row.querySelector('[id^="sistema-access-reference-"]');
    const valid = note && /^sistema-access-reference-\d+$/.test(note.id) &&
      note.textContent === 'Referencia ' + note.id.split('-').at(-1);
    if (root.contains(row) && (Boolean(note) !== (duplicates.length > 1) || note && !valid)) {
      problems.push('Referencia local de acceso fuera de su condición o formato');
    }
    if (duplicates.length > 1 && valid) accept(note, /^Referencia \d+$/);
  }
  const accessDialog = document.querySelector('#access-dialog[open]');
  const accessSubject = accessDialog?.querySelector('input[readonly]')?.value ?? accessDialog?.querySelector(':scope > strong')?.textContent;
  for (const note of accessDialog?.querySelectorAll(':scope > p.row-meta') ?? []) {
    if (accessRows.some(row => row.querySelector('.row-title')?.textContent === accessSubject &&
        row.querySelector('[id^="sistema-access-reference-"]')?.textContent === note.textContent)) accept(note, /^Referencia \d+$/);
  }
  // A paused profile without a verified name uses its first eight hex digits.
  const pausedRefs = [];
  for (const row of document.querySelectorAll('#voice-attribution [data-voice-suspended]')) {
    const reference = row.querySelector('span[id^="paused-ref-"]');
    const unnamed = row.querySelector('.row-title')?.textContent === 'Perfil de voz sin nombre disponible';
    const valid = /^Referencia: [a-f0-9]{8}$/.test(reference?.textContent ?? '');
    if (root.contains(row) && (unnamed ? !valid : Boolean(reference))) {
      problems.push('Referencia de un perfil en pausa fuera de su condición o formato');
    }
    if (unnamed && valid) {
      if (root.contains(row)) allowed.add(reference);
      pausedRefs.push(reference.textContent);
    }
  }
  // The confirmation shows the same reference only for an unnamed paused profile.
  const pauseDialog = document.querySelector('#voice-attribution-confirm-dialog[open]');
  const pauseNote = pauseDialog?.querySelector(':scope > p.row-meta');
  if (pauseDialog && root.contains(pauseDialog) && pauseDialog.querySelector('h2')?.textContent === 'Reanudar propuestas por voz') {
    const unnamed = pauseDialog.querySelector(':scope > p.modal-description')?.textContent.startsWith('Perfil de voz sin nombre disponible. ');
    if (unnamed && !pausedRefs.includes(pauseNote?.textContent)) problems.push('La confirmación no conserva la referencia del perfil sin nombre');
    if (unnamed && pausedRefs.includes(pauseNote?.textContent)) accept(pauseNote, /^Referencia: [a-f0-9]{8}$/);
  }
  // The free-text condition accepts Home Assistant's exact state values.
  for (const help of root.querySelectorAll('#routine-dialog p.routine-field-label[id^="routine-condition-state-help-"]')) {
    accept(help, /^Cerrado = closed · Abierto = open · Encendido = on · Apagado = off$/);
  }
  return {allowed, problems};
}"""


async def assert_no_canonical_ids(page: Page, root_selector: str) -> None:
    """Check attributes and live form values, including hidden descendants."""
    fields = await page.evaluate(r"""(selector => {
      // A browser-generated audio blob handle is an ephemeral URL, not a register ID.
      const audio = '#voice-dialog .nexo-voice-review audio, #evaluation-content [data-evaluation-sample] audio';
      const blobPrefix = 'blob:' + location.origin + '/';
      const browserBlob = (el, attr) => attr.name === 'src' && el.matches(audio) &&
        attr.value.startsWith(blobPrefix) && /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(attr.value.slice(blobPrefix.length));
      return [...document.querySelectorAll(selector)].flatMap(root =>
        [root, ...root.querySelectorAll('*')].flatMap(el => {
          const values = [...el.attributes].filter(attr => !browserBlob(el, attr))
            .map(attr => [attr.name, attr.value]);
          if (el.matches('input, select, textarea')) {
            values.push(['.value', el.value]);
            if (typeof el.defaultValue === 'string') values.push(['.defaultValue', el.defaultValue]);
          }
          return values.map(([name, value]) => [root.id || root.localName, el.localName, name, value]);
        }));
    })(%s)""" % json.dumps(root_selector))
    for scope, tag, name, value in fields:
        assert not any(pattern.search(name) or pattern.search(value)
                       for pattern in (_CANONICAL_ATTRIBUTES, _FIXTURE_ROOM_IDS)), (
            f'Canonical record ID in #{scope} <{tag}> attribute/property {name}: {value!r}')


async def assert_nexo_copy(page: Page, root_selector: str) -> None:
    """Scan visible copy and native picker options, honoring checked UI exceptions."""
    samples = await page.evaluate(r"""(selector => [...document.querySelectorAll(selector)]
      .filter(root => root.checkVisibility()).map(root => {
        const {allowed, problems} = (%s)(root);
        // Check each rendered text node and option by location before checking full phrases in innerText.
        const texts = [], walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        for (let text = walker.nextNode(); text; text = walker.nextNode()) {
          const parent = text.parentElement;
          if (!allowed.has(parent) && (parent.checkVisibility() ||
              parent.matches('option') && parent.closest('select')?.checkVisibility())) texts.push(text.data);
        }
        return {scope: root.id || root.localName, text: root.innerText, allowed: [...allowed].map(node => node.textContent.trim()), problems, texts};
      }))(%s)""" % (_COPY_EXCEPTIONS, json.dumps(root_selector)))
    for sample in samples:
        assert not sample['problems'], (sample['scope'], sample['problems'])
        # Native options collapse whitespace in innerText, even when textContent preserves it.
        text = ' '.join(sample['text'].split())
        for allowed in sample['allowed']:
            text = text.replace(' '.join(allowed.split()), '', 1)
        for candidate in [text, *sample['texts']]:
            match = _INTERNAL_COPY.search(candidate) or _FIXTURE_ROOM_IDS.search(candidate)
            assert not match, f"Internal language in #{sample['scope']}: {match[0]!r}" if match else ''


async def assert_nexo_presentation(page: Page) -> None:
    """Audit mounted record views, and the visible copy of the current screen and dialogs."""
    await assert_no_canonical_ids(
        page, '#home-view, #actions-view, #activity-view, .memory-voice-slot .memory-voice, #routines-view, #tasks-view, #memory-view, #my-memory-view, #people-view, #voices-view, '
        '#access-view, #evaluation-view, body > dialog')
    await assert_nexo_copy(page, '#login-panel, #nexo-workspace > section, #home-view, #workspace-toolbar, dialog[open]')


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
WCAG_TARGET_MIN_PX = 24
WCAG_SCRIPT = Path(__file__).with_name('browser_wcag.js').read_text(encoding='utf-8')
VIEWPORT_HEIGHTS = {320: 760, 390: 844, 768: 1024, 1280: 800, 1425: 1180, 1440: 900}
INTERACTIVE_ROLES = frozenset({
    'button', 'link', 'textbox', 'searchbox', 'combobox', 'listbox', 'checkbox', 'radio',
    'slider', 'spinbutton', 'switch', 'tab', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
})

AUDIT_SCRIPT = r"""(() => {
  if (window.__kzaAudit) return true;
  const FOCUSABLE = 'a[href],area[href],button,input:not([type=hidden]),select,textarea,summary,iframe,audio[controls],video[controls],[contenteditable]:not([contenteditable=false]),[tabindex]';
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
  const scope = () => [...document.querySelectorAll('dialog:modal')].at(-1) || document.body;
  const owns = (el, hit) => !!hit && (hit === el || el.contains(hit) || [...(el.labels || [])].some(label => label.contains(hit)));
  const controls = (root = scope()) => [...root.querySelectorAll(FOCUSABLE)].filter(el => el.tabIndex >= 0 && !el.closest('[inert]') && shown(el));
  const tabbables = root => {
    const candidates = controls(root).filter(el => !el.matches(':disabled'));
    return candidates.filter(el => {
      if (el.type !== 'radio' || !el.name) return true;
      const group = candidates.filter(item => item.type === 'radio' && item.name === el.name && item.form === el.form);
      return (group.find(item => item.checked) || group[0]) === el;
    });
  };
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
          // An inline <svg> is overflow:hidden by default but is not a container: look past it.
          for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
            if (node instanceof SVGElement) continue;
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
    async keyControls(specs, fresh, timeoutMs) {
      // A view whose controls depend on an expiring observation names a freshness
      // predicate. The check waits for it, then runs in the same task, so no
      // refresh or expiry timer can interleave between the two.
      let freshness = null;
      if (fresh) {
        const started = performance.now();
        while (!fresh() && performance.now() - started < timeoutMs) await new Promise(resolve => setTimeout(resolve, 25));
        freshness = {waited_ms: Math.round(performance.now() - started), current: !!fresh()};
        if (!freshness.current) return {freshness, results: specs.map(([selector, text]) => ({control: selector + (text ? ' "' + text + '"' : ''), problem: 'observation not current after ' + timeoutMs + ' ms'}))};
      }
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
      return {freshness, results};
    },
    targets(minimum) {
      return controls().filter(el => el.getBoundingClientRect().bottom + scrollY > 0).map(el => {
        const rects = [el.getBoundingClientRect(), ...[...(el.labels || [])].filter(label => label.contains(el)).map(label => label.getBoundingClientRect())];
        const width = Math.max(...rects.map(rect => rect.width)), height = Math.max(...rects.map(rect => rect.height));
        return {element: describe(el), width: Math.round(width * 10) / 10, height: Math.round(height * 10) / 10};
      }).filter(item => item.width < minimum - 0.05 || item.height < minimum - 0.05);
    },
    prepareKeyboard(selector) {
      const root = selector ? document.querySelector(selector) : scope();
      const modal = root.matches('dialog[open]');
      const expected = tabbables(root);
      const initial = document.activeElement;
      const scroll = [scrollX, scrollY];
      initial?.blur();
      const before = new Map(expected.map(el => [el, window.__kzaWcag.focusSnapshot(el)]));
      const names = expected.map(window.__kzaWcag.controlName);
      const visualFindings = [];
      const rects = expected.map(el => el.getBoundingClientRect());
      for (let i = 1; i < expected.length; i++) {
        const previous = expected[i - 1], current = expected[i];
        if (previous.matches('.skip-link') || current.matches('.skip-link')
          || previous.contains(current) || current.contains(previous)) continue;
        const a = rects[i - 1], b = rects[i];
        const rowTolerance = Math.max(8, Math.min(a.height, b.height) / 2);
        const row = Math.abs((a.top + a.bottom - b.top - b.bottom) / 2) <= rowTolerance;
        const overlapsColumn = Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1;
        if ((row && b.right <= a.left + 1) || (overlapsColumn && b.bottom <= a.top + 1)) {
          visualFindings.push('DOM order differs from visual order: ' + names[i - 1] + ' -> ' + names[i]);
        }
      }
      // Bracket the real tab order so an outside-modal trap on the last item
      // cannot pass just because every expected control has already been visited.
      const anchors = modal ? [] : [document.createElement('span'), document.createElement('span')];
      for (const anchor of anchors) {
        anchor.tabIndex = 0;
        anchor.style.cssText = 'position:fixed;inline-size:0;block-size:0';
        anchor.setAttribute('aria-hidden', 'true');
      }
      if (anchors.length) {
        root.prepend(anchors[0]);
        root.append(anchors[1]);
      }
      window.__kzaKeyboard = {expected, before, modal, anchors, initial, scroll};
      const positive = [...root.querySelectorAll('[tabindex]')].filter(el => el.tabIndex > 0)
        .map(window.__kzaWcag.controlName);
      return {expected: names, positive, modal, visual_findings: visualFindings};
    },
    startKeyboard(backward) {
      const {expected, modal, anchors} = window.__kzaKeyboard;
      if (modal) (backward ? expected[0] : expected.at(-1))?.focus();
      else (backward ? anchors[1] : anchors[0]).focus({preventScroll: true});
    },
    endKeyboard() {
      const {anchors, initial, scroll} = window.__kzaKeyboard;
      for (const anchor of anchors) anchor.remove();
      if (initial === document.body || !initial?.isConnected) document.activeElement?.blur();
      else initial.focus({preventScroll: true});
      window.scrollTo(...scroll);
      delete window.__kzaKeyboard;
    },
    step() {
      const el = document.activeElement, state = window.__kzaKeyboard;
      if (state.anchors.includes(el)) return {index: -2, element: 'document boundary'};
      if (!el || el === document.body || el === document.documentElement) return {index: -1, element: 'document'};
      let index = state.expected.indexOf(el);
      if (index < 0) {
        // A periodic catalog refresh may replace a control with an identical one.
        index = state.expected.findIndex(old => !old.isConnected && describe(old) === describe(el));
        if (index >= 0) {
          state.before.set(el, state.before.get(state.expected[index]));
          state.expected[index] = el;
        }
      }
      const rect = el.getBoundingClientRect();
      const width = document.documentElement.clientWidth;
      const within = rect.left >= -1 && rect.right <= width + 1
        && (rect.height > innerHeight ? rect.bottom > 0 && rect.top < innerHeight
          : rect.top >= -1 && rect.bottom <= innerHeight + 1);
      const ring = window.__kzaWcag.focusContrast(el, state.before.get(el));
      const nativeSegments = el.matches('input[type=datetime-local],input[type=time],input[type=date],audio[controls]');
      // Chromium retargets the focus event to the host when Tab enters a native editor segment.
      const focusVisible = el.matches(':focus-visible') || (nativeSegments && el.matches(':focus-within'));
      return {index, element: window.__kzaWcag.controlName(el), focusVisible,
              ring, indicator: ring.part === 'outline' ? outlineVisible(el) : ring.width > 0,
              native_segments: nativeSegments,
              unobscured: unobscured(el), within_viewport: within,
              rect: [rect.left, rect.top, rect.right, rect.bottom].map(Math.round), viewport: [width, innerHeight]};
    },
  };
  return true;
})()"""


async def assert_frame_consistent(page: Page, view: str, theme: Literal['claro', 'oscuro']) -> None:
    """Check the rendered frame, navigation labels and interactive states against NEXO tokens."""
    await page.evaluate(WCAG_SCRIPT)
    problems = await page.evaluate(r"""(([view, theme]) => {
      const issues = [];
      const root = document.documentElement;
      const style = getComputedStyle;
      const probe = document.createElement('span');
      document.body.append(probe);
      const token = name => {
        probe.style.color = 'var(--nexo-' + name + ')';
        return style(probe).color;
      };
      const expect = (label, actual, wanted) => {
        if (actual !== wanted) issues.push({label, actual, expected: wanted});
      };
      const rgb = color => color.match(/[\d.]+/g).slice(0, 3).map(Number);
      const contrast = (label, foreground, background, minimum) => {
        const ratio = window.__kzaWcag.contrast(rgb(foreground), rgb(background));
        if (ratio < minimum) issues.push({label, ratio, minimum});
      };
      const check = (selector, fg, bg, exactBackground = true) => {
        const element = document.querySelector(selector);
        if (!element) {
          issues.push({missing: selector});
          return;
        }
        const actual = style(element);
        expect(selector + ' color', actual.color, token(fg));
        if (exactBackground) expect(selector + ' background', actual.backgroundColor, token(bg));
        contrast(selector, actual.color, token(bg), 4.5);
      };

      try {
        expect('active theme', root.dataset.theme, theme);
        expect('visible view', document.querySelector(view)?.checkVisibility(), true);
        expect('color-scheme', style(document.body).colorScheme, theme === 'claro' ? 'light' : 'dark');
        for (const selector of ['html', 'body', view]) {
          const element = document.querySelector(selector);
          if (!element) continue;
          for (const alias of ['--bg', '--text', '--muted', '--border', '--mint']) {
            expect(selector + ' ' + alias, style(element).getPropertyValue(alias).trim(), '');
          }
        }
        for (const surface of ['ground', 'card', 'segment']) {
          contrast('focus on ' + surface, token('focus'), token(surface), 3);
        }
        for (const selector of [
          'body', '#main', '.topbar', '#app-nav', '#rail-nav', '#section-views',
          '#workspace-toolbar', '#theme-slot',
        ]) check(selector, 'ink', 'ground');
        check('.wordmark', document.querySelector('.wordmark').matches(':hover') ? 'focus' : 'ink', 'ground', false);
        check('.home-picker', 'muted', 'card');
        check('#session-label', 'muted', 'ground', false);
        check('#demo-banner', 'ink', 'card');
        check('.skip-link', 'ground', 'ink');
        check('#theme-slot select', 'ink', document.querySelector('#theme-slot select').matches(':hover') ? 'segment' : 'card');
        check('#refresh', 'ink', document.querySelector('#refresh').matches(':hover:enabled') ? 'segment' : 'card');

        const status = document.querySelector('#connection-status');
        const stale = status.classList.contains('is-stale');
        try {
          check('#connection-status', stale ? 'danger' : 'muted', 'ground', false);
          status.classList.toggle('is-stale', !stale);
          check('#connection-status', stale ? 'muted' : 'danger', 'ground', false);
        } finally {
          status.classList.toggle('is-stale', stale);
        }

        for (const button of document.querySelectorAll('#app-nav button')) {
          const label = button.textContent.trim() || button.getAttribute('aria-label');
          const actual = style(button);
          const active = button.matches('[aria-current], :hover');
          expect(label + ' color', actual.color, token('ink'));
          expect(label + ' background', actual.backgroundColor, token(active ? 'segment' : 'ground'));
          contrast(label, actual.color, actual.backgroundColor, 4.5);

          if (innerWidth <= 390 && button.checkVisibility() && button.classList.contains('section-link')) {
            const range = document.createRange();
            range.selectNodeContents(button);
            expect(label + ' text lines', range.getClientRects().length, 1);
          }
        }

        const link = document.querySelector('#people-view:not([hidden]) [data-person-access] a');
        if (link?.checkVisibility()) {
          const actual = style(link);
          expect('access link destination', link.getAttribute('href'), '#accesos');
          expect('access link decoration', actual.textDecorationLine, 'underline');
          const highlighted = link.matches(':hover, :focus-visible');
          expect('access link color', actual.color, token(highlighted ? 'ink' : 'focus'));
          contrast('access link', actual.color, token(highlighted ? 'segment' : 'card'), 4.5);
        }
      } finally {
        probe.remove();
      }
      return issues;
    })(""" + json.dumps([view, theme]) + ")")
    context = f'NEXO frame {view}, {theme}'
    assert not problems, f'{context}: {problems}'

    # A modal makes the frame inert. Its palette still applies behind the backdrop.
    if await page.evaluate("!!document.querySelector('dialog[open]')"):
        return

    await page.command('DOM.enable')
    await page.command('CSS.enable')
    root = (await page.command('DOM.getDocument'))['root']['nodeId']
    selector = (
        '.wordmark, .skip-link, #main, #app-nav button, #theme-slot select, #refresh, '
        '#people-view:not([hidden]) [data-person-access] a'
    )
    node_ids = (await page.command('DOM.querySelectorAll', {'nodeId': root, 'selector': selector}))['nodeIds']
    try:
        for node_id in node_ids:
            await page.command('CSS.forcePseudoState', {'nodeId': node_id, 'forcedPseudoClasses': ['focus-visible']})
        problems = await page.evaluate(r"""(selector => {
          const issues = [];
          const probe = document.createElement('span');
          probe.style.color = 'var(--nexo-focus)';
          document.body.append(probe);
          const expected = getComputedStyle(probe).color;
          probe.remove();
          for (const element of document.querySelectorAll(selector)) {
            if (!element.checkVisibility() || element.matches(':disabled')) continue;
            const label = element.id || element.textContent.trim();
            const style = getComputedStyle(element);
            const width = parseFloat(style.outlineWidth);
            if (style.outlineColor !== expected || style.outlineStyle !== 'solid' || width < 3) {
              issues.push({label, color: style.outlineColor, style: style.outlineStyle, width});
            }
            const rect = element.getBoundingClientRect();
            if (element.id !== 'main' && (rect.width < 44 || rect.height < 44)) {
              issues.push({label, width: rect.width, height: rect.height});
            }
          }
          return issues;
        })(""" + json.dumps(selector) + ")")
        assert not problems, f'{context}: focus or target size: {problems}'
    finally:
        await page.command('CSS.disable')
        await page.command('DOM.disable')


async def _press_tab(page, *, backward: bool = False) -> None:
    for kind in ('rawKeyDown', 'keyUp'):
        await dispatch_key(page, kind, 'Tab', 'Tab', 9, modifiers=8 if backward else 0)


async def dispatch_key(page: Page, kind: str, key: str, code: str, key_code: int, **params: str | int) -> None:
    """Send a trusted key event to the page without routing it to Chrome's window."""
    # A native virtual key code can route unhandled events to the browser and hide its Mac headless target.
    await page.command('Input.dispatchKeyEvent', {'type': kind, 'key': key, 'code': code,
                                                  'windowsVirtualKeyCode': key_code, **params})


async def press_enter(page) -> None:
    """Activate the focused control with a real Enter key event, as a keyboard user does."""
    await dispatch_key(page, 'keyDown', 'Enter', 'Enter', 13, text='\r')
    await dispatch_key(page, 'keyUp', 'Enter', 'Enter', 13)


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


async def audit_keyboard(page: Page, root_selector: str | None, width: int) -> tuple[dict, list[str]]:
    """Check exact forward and reverse Tab sequences, including their boundaries."""
    await page.evaluate(AUDIT_SCRIPT)
    await page.evaluate(WCAG_SCRIPT)
    prepared = await page.evaluate(f'window.__kzaAudit.prepareKeyboard({json.dumps(root_selector)})')
    expected = prepared['expected']
    findings = [f'positive tabindex disturbs the order: {item}' for item in prepared['positive']]
    findings += prepared['visual_findings']
    report = {'expected': len(expected), 'focused': 0, 'sequence': [], 'reverse_sequence': []}
    try:
        for backward, name in ((False, 'sequence'), (True, 'reverse_sequence')):
            direction = 'Shift+Tab' if backward else 'Tab'
            await page.evaluate(f'window.__kzaAudit.startKeyboard({json.dumps(backward)})')
            indices = list(range(len(expected)))
            if backward:
                indices.reverse()
            # Modal boundaries wrap, while a page must let focus pass beyond its last control.
            boundary = indices[0] if prepared['modal'] and indices else -2
            pairs = [*indices, boundary]
            previous = boundary if prepared['modal'] else -2
            for index in pairs:
                await _press_tab(page, backward=backward)
                step = await page.evaluate('window.__kzaAudit.step()')
                for _ in range(12):
                    if step['index'] == index or step['index'] != previous or not step.get('native_segments'):
                        break
                    await _press_tab(page, backward=backward)
                    step = await page.evaluate('window.__kzaAudit.step()')
                if step['index'] != index:
                    want = expected[index] if index >= 0 else 'the end of the document'
                    findings.append(f'{direction} expected {want}, reached {step["element"]}; '
                                    'check tab order or keyboard trap')
                    break
                previous = index
                if index < 0:
                    continue
                report[name].append(step['element'])
                if not step['focusVisible']:
                    findings.append(f'{direction} did not expose :focus-visible on {step["element"]}')
                if not step['unobscured'] or not step['within_viewport']:
                    findings.append(f'{direction} focused an obscured or offscreen control: {step["element"]} '
                                    f'{step["rect"]} in {step["viewport"]}, hit={step["unobscured"]}')
                ring = step['ring']
                if not step['indicator']:
                    findings.append(f'{direction} focused a control without a perceptible focus change: '
                                    f'{step["element"]} ({ring["candidates"]})')
                elif width in {320, 390, 1440} and (ring['ratio'] or 0) < 3:
                    findings.append(f'{direction} focus indicator on {step["element"]}: '
                                    f'{ring["changedWidth"]} changed px, {ring["ratio"]} contrast')
    finally:
        await page.evaluate('window.__kzaAudit.endKeyboard()')
    for name in ('sequence', 'reverse_sequence'):
        if prepared['modal'] and len(report[name]) == len(expected) + 1:
            report[name].pop()
        if len(report[name]) != len(expected):
            findings.append(f'{name} visited {len(report[name])} of {len(expected)} expected controls')
    report['focused'] = len(report['sequence'])
    report['reverse_focused'] = len(report['reverse_sequence'])
    return report, findings


async def _range_paint(page: Page, node_id: int) -> dict:
    """Read the actual Chromium track and thumb styles in the user-agent shadow tree."""
    root = (await page.command('DOM.describeNode', {'nodeId': node_id, 'depth': -1, 'pierce': True}))['node']

    def descendants(node: dict):
        yield node
        for child in node.get('children', []) + node.get('shadowRoots', []):
            yield from descendants(child)

    parts = {}
    for node in descendants(root):
        attributes = node.get('attributes', [])
        attribute = dict(zip(attributes[::2], attributes[1::2]))
        for name, pseudo in (('track', '-webkit-slider-runnable-track'), ('thumb', '-webkit-slider-thumb')):
            if attribute.get('pseudo') == pseudo or attribute.get('id') == name:
                parts[name] = node['backendNodeId']
    if parts.keys() != {'track', 'thumb'}:
        return {}
    ids = (await page.command('DOM.pushNodesByBackendIdsToFrontend', {
        'backendNodeIds': [parts['track'], parts['thumb']],
    }))['nodeIds']
    track, thumb = [
        {item['name']: item['value'] for item in
         (await page.command('CSS.getComputedStyleForNode', {'nodeId': identity}))['computedStyle']}
        for identity in ids
    ]
    return {'track': track['background-color'], 'thumb': thumb['background-color'],
            'thumbBorder': thumb['border-top-color']}


class _ContrastControlReplaced(Exception):
    """The control can no longer be measured at its original position."""


def _is_missing_cdp_node(error: RuntimeError) -> bool:
    return error.args == ({'code': -32000, 'message': 'Could not find node with given id'},)


async def _measure_control_contrast(page: Page, document: int, index: int, control: dict | None) -> list[dict]:
    if control is None:
        raise _ContrastControlReplaced

    node = (await page.command('DOM.querySelector', {
        'nodeId': document, 'selector': control['selector'],
    }))['nodeId']
    if not node:
        raise _ContrastControlReplaced

    async def measure(state: str | dict, *, range_paint: dict | None = None) -> dict | None:
        result = await page.evaluate(
            f'window.__kzaWcag.measureContrast({index}, {json.dumps(state)}, '
            f'{json.dumps(range_paint)}, {json.dumps(control["selector"])})')
        if result is not None and any(issue['part'] == 'control-replaced' for issue in result.get('issues', [])):
            raise _ContrastControlReplaced
        return result

    range_paint = await _range_paint(page, node) if control['range'] else None
    baseline = await measure('normal', range_paint=range_paint)
    measurements = [baseline['measurement']]
    if control['disabled']:
        return measurements

    disabled = await measure('disabled')
    if disabled is not None:
        measurements.append(disabled)

    states = [('hover', ['hover'])]
    if control['tabbable']:
        states.append(('focus-visible', ['focus', 'focus-visible']))
    forced = complete = False
    try:
        for name, pseudo in states:
            await page.command('CSS.forcePseudoState', {'nodeId': node, 'forcedPseudoClasses': pseudo})
            forced = True
            range_paint = await _range_paint(page, node) if control['range'] else None
            measurements.append(await measure({'name': name, 'before': baseline['before']}, range_paint=range_paint))
        complete = True
    finally:
        if forced:
            try:
                await page.command('CSS.forcePseudoState', {'nodeId': node, 'forcedPseudoClasses': []})
            except RuntimeError as error:
                # A detached node has no state to clear. Preserve any earlier measurement error.
                if complete or not _is_missing_cdp_node(error):
                    raise
    return measurements


async def audit_component_contrast(page: Page, root_selector: str | None, theme: str) -> dict:
    """Measure authored visual cues in Chromium without activating the controls."""
    await page.evaluate(WCAG_SCRIPT)
    await page.command('DOM.enable')
    await page.command('CSS.enable')
    document = (await page.command('DOM.getDocument'))['root']['nodeId']
    measurements = []
    retries = 0
    try:
        controls = await page.evaluate(f'window.__kzaWcag.prepareContrast({json.dumps(root_selector)})')
        assert controls is not None, f'Contrast audit root is not rendered: {root_selector}'
        for index, control in enumerate(controls):
            candidate = control
            for attempt in range(2):
                if attempt:
                    retries += 1
                    candidate = await page.evaluate(
                        f'window.__kzaWcag.rebindContrast({index}, {json.dumps(control["selector"])})')
                try:
                    measured = await _measure_control_contrast(page, document, index, candidate)
                except (RuntimeError, _ContrastControlReplaced) as error:
                    if isinstance(error, RuntimeError) and not _is_missing_cdp_node(error):
                        raise
                else:
                    measurements.extend(measured)
                    break
            else:
                measurements.append({
                    'control': control['control'], 'selector': control['selector'], 'state': 'normal',
                    'issues': [{'part': 'control-replaced'}],
                })
    finally:
        await page.evaluate('window.__kzaWcag.endContrast()')
        await page.command('CSS.disable')
        await page.command('DOM.disable')
    findings = [
        f'{item["control"]} ({item["state"]}) {issue["part"]}: '
        + (f'{issue["ratio"]:.2f}:1 < {issue["minimum"]}:1' if issue.get('ratio') is not None
           else f'{issue["width"]} px < {issue["minimum"]} px' if 'width' in issue
           else 'missing or unmeasurable') + f' [{item["selector"]}]'
        for item in measurements for issue in item['issues']
    ]
    return {'theme': theme, 'controls': len(controls), 'retries': retries,
            'measurements': measurements, 'findings': findings}


async def audit_wcag(page, key_controls: Sequence[tuple[str, str | None]] = (),
                     status_messages: Sequence[tuple[str, str]] = ()) -> list[dict]:
    """Collect deterministic, non-blocking WCAG and product findings for the current viewport."""
    await page.evaluate(WCAG_SCRIPT)
    return await page.evaluate(
        f'window.__kzaWcag.scan({json.dumps(key_controls)}, {WCAG_TARGET_MIN_PX}, {TOUCH_TARGET_MIN_PX}, '
        f'{json.dumps(status_messages)})'
    )


async def audit_modal_focus(page: Page, selector: str, *, initial: str | None = None) -> dict:
    """Check the dialog's opening focus and reuse the real keyboard audit."""
    await page.evaluate(WCAG_SCRIPT)
    report = await page.evaluate(f'window.__kzaWcag.modalOpening({json.dumps(selector)}, {json.dumps(initial)})')
    findings = report['findings'] = []
    if not report['found']:
        findings.append('dialog_missing')
        return report
    if not report['modal_semantics']:
        findings.append('modal_semantics_missing')
    if not report['initial_ok']:
        findings.append('initial_focus_incorrect')
    report['keyboard'], keyboard_findings = await audit_keyboard(page, selector, report['width'])
    findings.extend(keyboard_findings)
    if not report['keyboard']['expected']:
        findings.append('no_tabbable_control')
    return report


async def press_key(page: Page, key: str, key_code: int) -> None:
    """Send a trusted press for a key that does not insert text."""
    for kind in ('rawKeyDown', 'keyUp'):
        await dispatch_key(page, kind, key, key, key_code)


async def press_escape(page: Page) -> None:
    """Dismiss the current cancellable surface with a trusted Escape key event."""
    await press_key(page, 'Escape', 27)


async def audit_current_accessibility(page: Page, evidence: Path, view: str,
                                      root_selector: str | None = None, *, initial: str | None = None) -> dict:
    """Record both audits at the current viewport, then enforce their findings."""
    theme, width = await page.evaluate("[document.documentElement.dataset.theme, innerWidth]")
    contrast = await audit_component_contrast(page, root_selector, theme)
    is_modal = root_selector and await page.evaluate(f'document.querySelector({json.dumps(root_selector)}).matches("dialog:modal")')
    if is_modal:
        report = await audit_modal_focus(page, root_selector, initial=initial)
    else:
        keyboard, findings = await audit_keyboard(page, root_selector, width)
        report = {'keyboard': keyboard, 'findings': findings}
    report['contrast'] = contrast
    report['findings'].extend(contrast['findings'])
    name = f'{view}-{theme}-{width}'
    _write_a11y(evidence, name, report)
    assert not report['findings'], f'NEXO {name}: ' + '; '.join(report['findings'][:12])
    return report


def _write_a11y(evidence: Path, view: str, report: dict) -> None:
    (evidence / f'a11y-{view}.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )


async def audit_modal_return(page, evidence: Path, view: str, report: dict, *,
                             dialog: str, opener: str) -> None:
    """Record focus return after the harness closes the modal in its normal workflow."""
    await page.evaluate(WCAG_SCRIPT)
    issue = await page.evaluate(f'window.__kzaWcag.returnFocus({json.dumps(dialog)}, {json.dumps(opener)})')
    if issue:
        report['findings'].append(f'{dialog}: {issue}; expected focus at {opener}')
    report['focus_return_ok'] = issue is None
    focus = {key: value for key, value in report.items() if key != 'contrast'}
    _write_a11y(evidence, view, {'view': view, 'dialog': dialog, **focus})
    assert not report['findings'], f'NEXO {view}: {report["findings"]}'


async def audit_status(page, evidence: Path, view: str, state: str, selector: str) -> None:
    """Record live semantics for an already-reached asynchronous UI state."""
    await page.evaluate(WCAG_SCRIPT)
    findings = await page.evaluate(f'window.__kzaWcag.statuses({json.dumps([(state, selector)])})')
    _write_a11y(evidence, f'{view}-{state}', {
        'view': view, 'state': state, 'selector': selector, 'findings': findings,
    })


async def audit_view(page, evidence: Path, view: str, key_controls: Sequence[tuple[str, str | None]],
                     screenshot: Callable[..., Awaitable[None]], *, widths: Sequence[int] = C08_WIDTHS,
                     restore: tuple[int, int, bool] = (1425, 1180, False), fresh: str | None = None,
                     fresh_timeout_ms: int = 12000,
                     status_messages: Sequence[tuple[str, str]] = (),
                     text_zoom: bool = False,
                     theme: Literal['claro', 'oscuro'] | None = None) -> dict:
    """Check one NEXO view at every C08 width and record C07 accessibility findings.

    Per width: no horizontal overflow (document and visible elements), key
    controls visible, enabled and hit-testable, a screenshot, accessible names
    from the browser accessibility tree, a real Tab traversal (reachability,
    document order, :focus-visible, a visible indicator, not obscured) and, at
    320/390 px, 44x44 CSS px touch targets. The page language must be Spanish.
    Additional WCAG and product probes record structured findings at every
    width without changing the legacy assertion or the meaning of ``passed``.

    Args:
        page: Connected CDP page exposing ``command`` and ``evaluate``.
        evidence: Run evidence directory receiving screenshots and ``a11y-<view>.json``.
        view: Short view name used in evidence file names.
        key_controls: ``(selector, text)`` pairs; ``text`` narrows by visible text when given.
        screenshot: Harness screenshot coroutine ``(page, path, full=..., scroll_top=...)``.
        widths: CSS pixel widths to audit.
        fresh: JS expression that is true while the view's observed data is current.
            Key controls are checked only once it holds, in the same task; if it
            never holds within ``fresh_timeout_ms`` every key control fails.
        fresh_timeout_ms: Bound for that wait (the catalog request timeout).
        status_messages: Expected ``(state, message selector)`` pairs for this view.
        text_zoom: Also measure controls and keyboard at 320px with text doubled.
        restore: Viewport ``(width, height, mobile)`` applied after the audit.
        theme: Optional rendered NEXO theme. Suffixes the evidence names and
            temporarily synchronizes the theme selector without saving a preference.

    Returns:
        The per-width measurements written to the evidence file.

    Raises:
        AssertionError: When any check fails; every finding is recorded first.
    """
    if theme is None:
        active_theme = await page.evaluate("document.documentElement.dataset.theme || 'oscuro'")
        results = {}
        failures = []
        for current_theme in ('claro', 'oscuro'):
            try:
                results[current_theme] = await audit_view(
                    page, evidence, view, key_controls, screenshot, widths=widths, restore=restore, fresh=fresh,
                    fresh_timeout_ms=fresh_timeout_ms, status_messages=status_messages,
                    text_zoom=text_zoom, theme=current_theme,
                )
            except AssertionError as error:
                failures.append(str(error))
        assert not failures, '\n'.join(failures)
        _write_a11y(evidence, view, results[active_theme])
        return results[active_theme]
    evidence_view = f'{view}-{theme}'
    await page.command('Accessibility.enable')
    await page.evaluate(AUDIT_SCRIPT)
    language = await page.evaluate('window.__kzaAudit.language()')
    findings = []
    if language['lang'] != 'es':
        findings.append(f'html lang is {language["lang"]!r}, expected "es"')
    findings += [f'non-Spanish lang attribute: {item}' for item in language['foreign']]
    report = {'view': view, 'language': language, 'widths': {}}
    previous_theme = None
    if theme is not None:
        report['theme'] = theme
        previous_theme = await page.evaluate("""(() => {
            const root = document.documentElement;
            const body = document.body;
            const select = document.querySelector('#theme-slot select');
            const previous = [root.getAttribute('data-theme'), body.classList.contains('night'), select?.value ?? null];
            root.dataset.theme = """ + json.dumps(theme) + """;
            body.classList.toggle('night', root.dataset.theme === 'oscuro');
            if (select) select.value = root.dataset.theme;
            return previous;
        })()""")
    try:
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
            predicate = 'null' if fresh is None else f'() => ({fresh})'
            checked = await page.evaluate('window.__kzaAudit.keyControls(' + json.dumps([list(item) for item in key_controls])
                                          + f', {predicate}, {int(fresh_timeout_ms)})')
            controls, freshness = checked['results'], checked['freshness']
            local += [f'key control {item["control"]}: {item["problem"]}' for item in controls if item['problem']]
            shot = evidence / f'{evidence_view}-{width}.png'
            await screenshot(page, shot, full=True)
            nameless = await _nameless_controls(page)
            local += [f'control without accessible name: {item}' for item in nameless]
            small = []
            if width in TOUCH_TARGET_WIDTHS:
                small = await page.evaluate(f'window.__kzaAudit.targets({TOUCH_TARGET_MIN_PX})')
                local += [f'touch target below {TOUCH_TARGET_MIN_PX}x{TOUCH_TARGET_MIN_PX}: {item}' for item in small]
            dialog = await page.evaluate("document.querySelector('dialog:modal')?.id || null")
            if dialog:
                modal = await audit_modal_focus(page, '#' + dialog)
                keyboard, keyboard_findings = modal['keyboard'], modal['findings']
            else:
                keyboard, keyboard_findings = await audit_keyboard(page, None, width)
            contrast = None
            if width in {390, 1440}:
                contrast = await audit_component_contrast(page, None, theme)
                _write_a11y(evidence, f'{evidence_view}-{width}-contrast', contrast)
                local += contrast['findings']
            if text_zoom and width == 320:
                await page.evaluate('window.__kzaWcag.beginTextZoom()')
                try:
                    enlarged = await audit_component_contrast(page, None, theme)
                    zoom_keyboard, zoom_findings = await audit_keyboard(page, None, width)
                    _write_a11y(evidence, f'{evidence_view}-{width}-text200', {
                        'contrast': enlarged, 'keyboard': zoom_keyboard, 'findings': enlarged['findings'] + zoom_findings,
                    })
                    local += [f'200% text: {item}' for item in enlarged['findings'] + zoom_findings]
                finally:
                    await page.evaluate('window.__kzaWcag.endTextZoom()')
            wcag = await audit_wcag(page, key_controls, status_messages)
            local += keyboard_findings
            report['widths'][str(width)] = {'screenshot': shot.name, 'overflow': overflow, 'key_controls': controls,
                                            'freshness': freshness,
                                            'small_targets': small, 'keyboard': keyboard, 'findings': local,
                                            'wcag_findings': wcag}
            findings += [f'{width}px: {item}' for item in local]
    finally:
        if previous_theme is not None:
            await page.evaluate("""(([theme, night, selection]) => {
                const root = document.documentElement;
                const select = document.querySelector('#theme-slot select');
                if (theme === null) root.removeAttribute('data-theme');
                else root.dataset.theme = theme;
                document.body.classList.toggle('night', night);
                if (select && selection !== null) select.value = selection;
            })(""" + json.dumps(previous_theme) + ")")
        await page.command('Emulation.setDeviceMetricsOverride', {'width': restore[0], 'height': restore[1],
                                                                  'deviceScaleFactor': 1, 'mobile': restore[2]})
    report['findings'] = findings
    report['passed'] = not findings
    path = evidence / f'a11y-{evidence_view}.json'
    _write_a11y(evidence, evidence_view, report)
    assert not findings, f'NEXO {evidence_view} reflow/accessibility findings ({len(findings)}), see {path.name}: ' + '; '.join(findings[:12])
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
