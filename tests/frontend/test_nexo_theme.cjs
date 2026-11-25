const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const home = path.join(__dirname, '..', '..', 'src', 'dashboard', 'frontend', 'home');
const read = file => readFileSync(path.join(home, file), 'utf8');
const tokens = read('nexo/tokens.css');
const components = read('nexo/components.css');
const themeScript = read('nexo/theme.js');

function properties(selector) {
  const start = tokens.indexOf(`${selector} {`);
  assert.ok(start >= 0, `missing ${selector}`);
  const block = tokens.slice(start, tokens.indexOf('}', start));
  return Object.fromEntries([...block.matchAll(/--nexo-([\w-]+):\s*([^;]+);/g)]
    .map(([, name, value]) => [name, value]));
}

const dark = properties(':root');
const light = { ...dark, ...properties(':root[data-theme="claro"]') };

test('NEXO defines the design tokens in both themes and uses the local display font', () => {
  const common = {
    accent: '#F2B33D', 'on-accent': '#17202A', 'text-small': '0.875rem',
    'text-body': '1rem', 'text-heading': '1.5rem', 'text-title': '2rem',
    'radius-small': '4px', radius: '6px', 'touch-target': '44px', 'focus-width': '3px',
  };
  for (const [palette, expected] of [
    [light, {
      scheme: 'light', ground: '#EEF0EC', ink: '#17202A', muted: '#3F4A55',
      line: '#C9CEC6', card: '#FFFFFF', segment: '#DDE1DA', focus: '#1F5FD1',
    }],
    [dark, {
      scheme: 'dark', ground: '#14171A', ink: '#EEF1F3', muted: '#B4BCC4',
      line: '#3A424A', card: '#1C2025', segment: '#23282D', focus: '#8FB4FF',
    }],
  ]) {
    for (const [name, value] of Object.entries({ ...common, ...expected })) {
      assert.equal(palette[name], value, `${palette.scheme} ${name}`);
    }
    for (const step of [1, 2, 3, 4, 5, 6, 8, 10]) {
      assert.equal(palette[`space-${step}`], `${step * 4}px`);
    }
    assert.match(palette.danger, /^#[\dA-F]{6}$/);
    assert.match(palette['font-body'], /^Inter,/);
    assert.match(palette['font-heading'], /^"Space Grotesk", var\(--nexo-font-body\)$/);
  }
  assert.match(tokens, /@font-face\s*\{[^}]*font-family: Inter;[^}]*url\("\.\/vendor\/inter-latin\.woff2"\)/);
  assert.equal(readFileSync(path.join(home, 'nexo/vendor/inter-latin.woff2')).subarray(0, 4).toString(), 'wOF2');
  assert.match(tokens, /@font-face\s*\{[^}]*font-family: "Space Grotesk";[^}]*url\("\.\/vendor\/space-grotesk-latin\.woff2"\)/);
  assert.equal(readFileSync(path.join(home, 'nexo/vendor/space-grotesk-latin.woff2')).subarray(0, 4).toString(), 'wOF2');
});

function luminance(hex) {
  const [r, g, b] = hex.slice(1).match(/../g).map(channel => {
    const value = parseInt(channel, 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

test('text meets AA and the focus indicator contrasts with both themes’ surfaces', () => {
  for (const palette of [light, dark]) {
    const verify = (foreground, background, minimum) => {
      const a = luminance(palette[foreground]);
      const b = luminance(palette[background]);
      const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      assert.ok(ratio >= minimum, `${palette.scheme} ${foreground}/${background}: ${ratio.toFixed(2)} < ${minimum}`);
    };
    for (const surface of ['ground', 'card', 'segment']) {
      for (const text of ['ink', 'muted', 'danger']) verify(text, surface, 4.5);
      verify('focus', surface, 3);
    }
    verify('on-accent', 'accent', 4.5);
  }
});

test('shared components use tokenized colors, text of at least 14 px and 44 px targets', () => {
  assert.doesNotMatch(components, /#[\da-f]{3,8}\b|(?:rgb|hsl)a?\(/i);
  const sizes = [...components.matchAll(/\bfont-size:\s*([^;]+);/g)];
  assert.ok(sizes.length > 0);
  for (const [, value] of sizes) {
    const name = /^var\(--nexo-(text-[\w-]+)\)$/.exec(value)?.[1];
    for (const palette of [light, dark]) {
      const [, n, unit] = /^([\d.]+)(px|rem)$/.exec(palette[name] ?? value) ?? [];
      assert.ok(Number(n) * (unit === 'rem' ? 16 : 1) >= 14, value);
    }
  }
  assert.match(components, /\.btn,\s*\.sw\s*\{[^}]*min-inline-size: var\(--nexo-touch-target\);[^}]*min-block-size: var\(--nexo-touch-target\);/);
  assert.match(components, /\.btn:focus-visible,\s*\.sw:focus-visible\s*\{[^}]*outline: var\(--nexo-focus-width\) solid var\(--nexo-focus\);/);
});

test('the head loads the theme before the style sheets and the app module', () => {
  const head = /<head>([^]*?)<\/head>/.exec(read('index.html'))[1];
  const assets = [
    '<script src="/home/nexo/theme.js"></script>',
    'href="/home/nexo/tokens.css"', 'href="/home/nexo/base.css"', 'href="/home/nexo/components.css"',
    'href="/home/nexo/marco.css"', '<script type="module" src="/home/nexo/app.js">',
  ];
  let previous = -1;
  for (const asset of assets) {
    const next = head.indexOf(asset);
    assert.ok(next > previous, `missing or out of order: ${asset}`);
    previous = next;
  }
});

class Element extends EventTarget {
  children = [];
  attributes = new Map();

  constructor(tagName) {
    super();
    this.tagName = tagName;
  }

  append(child) {
    this.children.push(child);
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }
}

function runTheme({ saved, systemDark = false, slot = null, storageBlocked = false, writeBlocked = false } = {}) {
  const key = 'nexo-theme';
  const storage = new Map(saved === undefined ? [] : [[key, saved]]);
  const bodyClasses = new Set();
  const document = Object.assign(new EventTarget(), {
    documentElement: { dataset: {} },
    body: null,
    readyState: 'loading',
    getElementById: id => id === 'theme-slot' ? slot : null,
    createElement: name => new Element(name),
  });
  const media = Object.assign(new EventTarget(), { matches: systemDark });
  const window = {
    matchMedia(query) {
      assert.equal(query, '(prefers-color-scheme: dark)');
      return media;
    },
    get localStorage() {
      if (storageBlocked) throw new Error('blocked storage access');
      return {
        getItem: name => storage.get(name) ?? null,
        setItem(name, value) {
          if (writeBlocked) throw new Error('storage is full');
          storage.set(name, value);
        },
      };
    },
  };
  runInNewContext(themeScript, { document, window }, { filename: 'theme.js' });
  return {
    get theme() { return document.documentElement.dataset.theme; },
    get night() { return bodyClasses.has('night'); },
    get saved() { return storage.get(key); },
    ready() {
      document.body = {
        classList: {
          toggle(name, enabled) {
            if (enabled) bodyClasses.add(name);
            else bodyClasses.delete(name);
          },
        },
      };
      document.readyState = 'interactive';
      document.dispatchEvent(new Event('DOMContentLoaded'));
    },
    system(darkMode) {
      media.matches = darkMode;
      media.dispatchEvent(new Event('change'));
    },
  };
}

function choose(select, value) {
  select.value = value;
  select.dispatchEvent(new Event('change'));
}

test('missing or unrecognized preferences default to auto before the body exists', () => {
  for (const saved of [undefined, 'unsupported']) {
    for (const systemDark of [false, true]) {
      const slot = new Element('div');
      const page = runTheme({ saved, systemDark, slot });
      assert.equal(page.theme, systemDark ? 'oscuro' : 'claro');
      assert.equal(page.night, false);
      page.ready();
      assert.equal(slot.children[0].value, 'auto');
      assert.equal(page.night, systemDark);
      page.system(!systemDark);
      assert.equal(page.theme, systemDark ? 'claro' : 'oscuro');
      assert.equal(page.night, !systemDark);
    }
  }
});

test('the theme slot receives one named native selector with Spanish options and persisted changes', () => {
  const slot = new Element('div');
  const page = runTheme({ saved: 'claro', systemDark: true, slot });
  assert.equal(page.theme, 'claro');
  assert.equal(slot.children.length, 0);
  page.ready();
  assert.equal(slot.children.length, 1);
  const [select] = slot.children;
  assert.equal(select.tagName, 'select');
  assert.equal(select.className, 'btn');
  assert.equal(select.attributes.get('aria-label'), 'Tema');
  assert.deepEqual(select.children.map(option => [option.value, option.textContent]), [
    ['auto', 'Automático'], ['claro', 'Claro'], ['oscuro', 'Oscuro'],
  ]);
  assert.equal(select.value, 'claro');
  assert.equal(page.night, false);
  choose(select, 'oscuro');
  assert.equal(page.theme, 'oscuro');
  assert.equal(page.night, true);
  assert.equal(page.saved, 'oscuro');
});

test('auto follows system changes while an explicit preference takes precedence', () => {
  const slot = new Element('div');
  const page = runTheme({ saved: 'auto', systemDark: true, slot });
  assert.equal(page.theme, 'oscuro');
  page.ready();
  const [select] = slot.children;
  assert.equal(select.value, 'auto');
  page.system(false);
  assert.equal(page.theme, 'claro');
  assert.equal(page.night, false);
  assert.equal(page.saved, 'auto');
  choose(select, 'claro');
  page.system(true);
  assert.equal(page.theme, 'claro');
  assert.equal(page.night, false);
});

test('the selector still changes themes when storage access or writing fails', () => {
  for (const failure of [{ storageBlocked: true }, { writeBlocked: true }]) {
    const slot = new Element('div');
    const page = runTheme({ slot, ...failure });
    assert.equal(page.theme, 'claro');
    page.ready();
    const [select] = slot.children;
    assert.equal(select.value, 'auto');
    choose(select, 'oscuro');
    assert.equal(page.theme, 'oscuro');
    assert.equal(page.night, true);
    assert.equal(page.saved, undefined);
  }
});
