// T042-U5: source-level guards for Q24 (text and field floors) and Q26 (WCAG 2.5.7 rotate buttons).
// Computed styles, focus and the WebGL yaw are checked in the browser harnesses (browser_nexo_home.py, browser_obs_panel.py).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { pathToFileURL } = require('node:url');

const frontend = path.join(__dirname, '..', '..', 'src', 'dashboard', 'frontend');
const read = relative => fs.readFileSync(path.join(frontend, relative), 'utf8');
const nexo = path.join(frontend, 'home', 'nexo');
const stylesheets = fs.readdirSync(nexo).filter(name => name.endsWith('.css')).sort()
  .map(name => `home/nexo/${name}`);
const tokens = read('home/nexo/tokens.css');
const textSizes = new Map([...tokens.matchAll(/(--nexo-text-[\w-]+):\s*([\d.]+)(px|rem);/g)]
  .map(([, token, size, unit]) => [token, Number(size) * (unit === 'rem' ? 16 : 1)]));

test('NEXO loads every stylesheet once and keeps CSS out of the page markup', () => {
  const html = read('home/index.html');
  const linked = [...html.matchAll(/<link rel="stylesheet" href="\/([^"]+)">/g)].map(([, name]) => name);
  assert.deepEqual(linked.sort(), stylesheets);
  assert.doesNotMatch(html, /<style(?:\s|>)/i);
});

test('NEXO view styles use defined tokens, with no literal colors or text below 14 px', () => {
  const defined = new Set([...tokens.matchAll(/(--nexo-[\w-]+):/g)].map(([, name]) => name));
  for (const file of stylesheets.filter(name => name !== 'home/nexo/tokens.css')) {
    const css = read(file).replace(/\/\*[\s\S]*?\*\//g, '');
    for (const [, property, value] of css.matchAll(/(?:^|[;{])\s*([\w-]+)\s*:\s*([^;{}]+)/g)) {
      assert.doesNotMatch(value, /#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])|\b(?:rgb|hsl)a?\s*\(/i,
        `${file}: literal color in ${property}: ${value}`);
      for (const [, token] of value.matchAll(/\bvar\(\s*(--[\w-]+)/g)) {
        assert.ok(defined.has(token), `${file}: undefined token ${token}`);
      }
      if (property === 'font-size' || (property === 'font' && value.trim() !== 'inherit')) {
        const size = /^\s*(?:[1-9]00\s+)?var\((--nexo-text-[\w-]+)\)/.exec(value)?.[1];
        assert.ok(textSizes.get(size) >= 14, `${file}: ${property}: ${value} must use a text token of at least 14 px`);
      }
    }
  }
});

test('no declared font-size in the NEXO stylesheets, home page or /obs/ is below 11 px', () => {
  for (const file of [...stylesheets, 'home/index.html', 'obs/index.html']) {
    for (const match of read(file).matchAll(/(?<!")font-size:\s*([0-9.]+)px/g)) {
      assert.ok(Number(match[1]) >= 11, `${file}: font-size ${match[0]} is below 11px`);
    }
  }
});

test('inline sizes below 11 px in the /obs/ views are raised by the page stylesheet', () => {
  const obs = read('obs/index.html');
  assert.match(obs, /\[style\*="font-size: 10px"\][^{]*\{ font-size: 12px !important; \}/);
  assert.match(obs, /\[style\*="font-size: 9px"\][^{]*\{ font-size: 11px !important; \}/);
  const sources = path.join(frontend, 'obs', 'src');
  const smallest = [];
  const walk = directory => { for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(full);
    else for (const match of fs.readFileSync(full, 'utf8').matchAll(/fontSize: *([0-9.]+)/g)) smallest.push(Number(match[1]));
  } };
  walk(sources);
  assert.ok(smallest.every(size => size >= 9), `unexpected inline font size: ${Math.min(...smallest)}`);
});

test('phone widths keep a 16 px floor for inputs, selects and textareas', () => {
  const css = read('home/nexo/base.css');
  assert.equal(textSizes.get('--nexo-text-body'), 16);
  assert.match(css, /@media \(max-width: 38\.75rem\)\s*\{[^]*?textarea\s*\{\s*font-size: var\(--nexo-text-body\) !important;/);
  assert.match(read('obs/index.html'), /input, select, textarea \{ font-size: 16px; \}/);
});

test('voice dialog buttons and filters reach 44 px and dialogs use dynamic viewport height', () => {
  const css = stylesheets.map(read).join('\n');
  assert.doesNotMatch(css, /\.nexo-voice-dialog[^{}]*\{[^}]*min-height:\s*(3\d|40)px/);
  assert.doesNotMatch(css, /#voices-view \.identity-secondary[^{}]*\{[^}]*min-height:\s*(3\d|40)px/);
  assert.match(read('home/nexo/base.css'), /(?:^|\n)dialog\s*\{[^}]*max-block-size: calc\(100dvh - var\(--nexo-space-6\)\)/);
  assert.match(tokens, /--nexo-space-6: 24px;/);
  assert.match(read('obs/index.html'), /\.login-dialog \{ max-height: calc\(100dvh - 24px\); \}/);
});

test('both pages use viewport-fit=cover with safe-area insets', () => {
  for (const file of ['home/index.html', 'obs/index.html']) assert.match(read(file), /viewport-fit=cover/, file);
  assert.match(read('home/nexo/marco.css'), /env\(safe-area-inset-bottom/);
  assert.match(read('obs/index.html'), /env\(safe-area-inset-left\)/);
});

test('the credential field carries entry hints', () => {
  const token = /<input id="token"[^>]*>/.exec(read('home/index.html'))[0];
  for (const attribute of ['autocomplete="current-password"', 'autocapitalize="off"', 'spellcheck="false"', 'enterkeyhint="go"']) {
    assert.ok(token.includes(attribute), `${attribute} missing`);
  }
});

test('Q26: rotate buttons sit in the map tools, are named buttons and stay wired to the house API', () => {
  const html = read('home/index.html');
  const tools = /<div class="map-tools">([^]*?)<\/div>/.exec(html)[1];
  const order = [...tools.matchAll(/<button\b[^>]*\bid="([a-z-]+)"/g)].map(item => item[1]);
  assert.deepEqual(order, ['zoom-in', 'zoom-out', 'rotate-left', 'rotate-right', 'reset-view', 'expand-map']);
  for (const [id, label] of [['rotate-left', 'Rotar a la izquierda'], ['rotate-right', 'Rotar a la derecha']]) {
    const control = [...tools.matchAll(/<button\b[^>]*>/g)].find(([tag]) => tag.includes(`id="${id}"`))[0];
    for (const attribute of ['type="button"', 'class="btn"', `aria-label="${label}"`]) assert.ok(control.includes(attribute));
  }
  const app = read('home/nexo/app.js');
  assert.match(app, /\['rotate-left', \(\) => house\?\.rotate\(-1\)\]/);
  assert.match(app, /\['rotate-right', \(\) => house\?\.rotate\(1\)\]/);
  assert.match(app, /'zoom-out', 'rotate-left', 'rotate-right', 'reset-view'\]\) byId\(id\)\.disabled = !ready/);
  assert.match(read('home/nexo/components.css'), /\.btn,\s*\.sw\s*\{[^}]*min-inline-size: var\(--nexo-touch-target\);[^}]*min-block-size: var\(--nexo-touch-target\)/);
  assert.match(read('home/nexo/tokens.css'), /--nexo-touch-target: 44px/);
});

test('R2: the 16 px phone floor includes range and checkbox inputs', () => {
  const css = read('home/nexo/base.css');
  assert.match(css, /input:not\(\[type="hidden"\], \[type="button"\], \[type="submit"\]\),\s*select,\s*textarea\s*\{\s*font-size: var\(--nexo-text-body\) !important;/);
});

test('R2: voice sample checkboxes are 44 px targets and the expanded map tools lie in a row', () => {
  assert.match(read('home/nexo/tokens.css'), /--nexo-touch-target:\s*44px;/);
  const checkbox = read('home/nexo/personas.css').match(/#voice-dialog input\[type="checkbox"\]\s*\{([^}]+)\}/)?.[1] ?? '';
  assert.match(checkbox, /inline-size:\s*var\(--nexo-touch-target\);/);
  assert.match(checkbox, /block-size:\s*var\(--nexo-touch-target\);/);
  assert.match(read('home/nexo/casa.css'), /\.map-toolbar,\s*\.view-switch,\s*\.map-tools,\s*#layer-tabs\s*\{[^}]*display: flex;[^}]*flex-wrap: wrap/);
});

test('R2: /obs/ reserves the top safe area, mutes text at AA and hints the credential field', () => {
  const obs = read('obs/index.html');
  assert.match(obs, /\.sidebar \{[^}]*padding: calc\(8px \+ env\(safe-area-inset-top, 0px\)\)/);
  assert.match(obs, /--ink-mute: #9a9aa6;/);
  const luminance = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)).reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
  const ratio = (a, b) => { const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
  assert.ok(ratio('#9a9aa6', '#18181c') >= 4.5);
  assert.ok(ratio('#6e6e7a', '#18181c') >= 3);
  const field = /<input id=\{`\$\{id\}-token`\}[^>]*>/.exec(read('obs/src/views/admin.jsx'))[0];
  for (const attribute of ['autoComplete="current-password"', 'autoCapitalize="off"', 'spellCheck={false}', 'enterKeyHint="go"']) assert.ok(field.includes(attribute), attribute);
});

test('Q26: without WebGL the house falls back to inert rotate/yaw and mounts no canvas', async () => {
  const saved = Object.fromEntries(['document', 'cancelAnimationFrame', 'matchMedia', 'window'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const fallback = { className: '', removed: false, setAttribute() {}, remove() { this.removed = true; } };
  const container = { dataset: {}, after(node) { this.fallback = node; }, clientWidth: 0, clientHeight: 0 };
  Object.defineProperties(globalThis, {
    document: { value: { createElement: () => fallback, hidden: false }, configurable: true, writable: true },
    window: { value: { devicePixelRatio: 1 }, configurable: true, writable: true },
    cancelAnimationFrame: { value() {}, configurable: true, writable: true },
    matchMedia: { value: () => ({ matches: false }), configurable: true, writable: true },
  });
  try {
    const { createHouse } = await import(pathToFileURL(path.join(frontend, 'home/nexo/house.js')).href);
    const ready = [];
    const house = createHouse({ container, modelUrl: '/example.glb', onReady: value => ready.push(value) });
    assert.deepEqual(ready, [false]);
    assert.equal(container.fallback, fallback);
    assert.equal(fallback.className, 'house-fallback');
    house.rotate(1); house.rotate(-1);
    assert.equal(house.yaw(), null);
    assert.equal(container.dataset.yaw, undefined);
    house.destroy();
    assert.equal(fallback.removed, true);
  } finally {
    for (const [key, descriptor] of Object.entries(saved)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  }
});
