"""Run with python -m tests.frontend.browser_nexo_css_parse under the browser lock."""

import argparse
from collections import Counter
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import tempfile

from tests.frontend.browser_support import ROOT, WCAG_SCRIPT, browser_binary, browser_profile_root


STYLES_DIR = ROOT / 'src/dashboard/frontend/home/nexo'
SHEETS = ('tokens', 'base', 'components', 'marco', 'casa', 'rutinas', 'tareas',
          'personas', 'memoria', 'sistema', 'actividad')
LEGACY_ALIASES = ('--bg', '--text', '--muted', '--border', '--mint')
COMPATIBILITY_SELECTORS = {
    # Keep the exact Firefox range rules in the count when the browser supports them.
    'casa': {
        ':is(#room-controls, #actions-view) input[type="range"]::-moz-range-track': 1,
        ':is(#room-controls, #actions-view) input[type="range"]::-moz-range-thumb': 1,
    },
}
CSS_TOKENS = re.compile(r"""/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'|\\[\s\S]|[{}()\[\]]""")
CLOSING = {'}': '{', ')': '(', ']': '['}

PROBE_HTML = """<!doctype html>
<a class="skip-link" href="#main">Ir al contenido</a>
<div class="app-shell">
<header class="topbar"><a class="wordmark">KZA</a></header>
<nav id="app-nav"><div id="rail-nav"></div><div id="section-views"></div><div id="theme-slot"></div></nav>
<main id="main">
<section id="login-panel" class="panel"></section>
<section id="nexo-workspace" hidden>
  <div id="workspace-toolbar"></div>
  <section id="actions-view" hidden></section>
  <section id="home-view" hidden></section>
  <section id="routines-view" hidden></section>
  <section id="tasks-view" hidden></section>
  <section id="memory-view" hidden></section>
  <section id="my-memory-view" hidden></section>
  <section id="people-view" hidden></section>
  <section id="voices-view" hidden></section>
  <section id="access-view" hidden></section>
  <section id="evaluation-view" hidden></section>
  <section id="activity-view" hidden></section>
</section></main></div>
<dialog id="info-dialog"></dialog>
<pre id="result"></pre>
<script id="config" type="application/json">__CONFIG__</script>
<script>
const { sources, aliases, compatibility } = JSON.parse(document.querySelector('#config').textContent);
const sheets = {};
const parsed = {};
function countRules(rules) {
  return [...rules].reduce((count, rule) => count + 1 + (rule.cssRules ? countRules(rule.cssRules) : 0), 0);
}
for (const [name, source] of Object.entries(sources)) {
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(source);
  sheets[name] = sheet;
  parsed[name] = countRules(sheet.cssRules);
}

const failures = [];
const check = (scope, property, actual, expected) => {
  if (actual !== expected || expected === '') failures.push({scope, property, actual, expected});
};
const look = (element, property) => getComputedStyle(element).getPropertyValue(property).trim();
const body = document.body;
const sentinel = new CSSStyleSheet();
sentinel.replaceSync('body {' + aliases.map(name => name + ':outside-view').join(';') + '}');
const personas = document.querySelectorAll('#people-view, #voices-view');
const workspace = document.querySelector('#nexo-workspace');
const login = document.querySelector('#login-panel');
const help = document.querySelector('#info-dialog');
const views = [...workspace.querySelectorAll('section')];
const frame = ['#main', '.topbar', '#app-nav', '#rail-nav', '#section-views', '#theme-slot', '#workspace-toolbar'];
const shared = sheets.marco.cssRules;
check('marco stylesheet', 'rule count', shared.length, 1);
check('marco stylesheet', 'selector', shared[0]?.selectorText, 'body');
document.adoptedStyleSheets = [sheets.tokens, sheets.base, sentinel, sheets.marco, sheets.personas];
function tokenColor(token) {
  const probe = document.createElement('span');
  probe.style.color = 'var(--nexo-' + token + ')';
  body.append(probe);
  const color = look(probe, 'color');
  probe.remove();
  return color;
}
for (const theme of ['claro', 'oscuro']) {
  const root = document.documentElement;
  root.dataset.theme = theme;
  const scheme = look(root, '--nexo-scheme');
  check({theme}, 'viewport color-scheme', look(root, 'color-scheme'), scheme);
  const ground = tokenColor('ground');
  const card = tokenColor('card');
  const ink = tokenColor('ink');
  for (const active of [null, help, ...views]) {
    const scope = {theme, active: active?.id ?? login.id};
    workspace.hidden = !views.includes(active);
    login.hidden = !workspace.hidden;
    for (const view of views) view.hidden = view !== active;
    if (active === help) help.showModal();
    check(scope, 'color-scheme', look(body, 'color-scheme'), scheme);
    check(scope, 'body background', look(body, 'background-color'), ground);
    check(scope, 'body color', look(body, 'color'), ink);
    for (const selector of frame) {
      const element = document.querySelector(selector);
      check({...scope, selector}, 'background-color', look(element, 'background-color'), ground);
      check({...scope, selector}, 'color', look(element, 'color'), ink);
    }
    if (!login.hidden) check({...scope, element: login.id}, 'background-color', look(login, 'background-color'), card);
    if (help.open) {
      check(scope, 'help background', look(help, 'background-color'), card);
      check(scope, 'help color', look(help, 'color'), ink);
      help.close();
    }
    // Personas still uses the chosen palette and never shadows retired properties.
    for (const element of [body, ...personas]) {
      check({...scope, element: element.id || 'body'}, 'color-scheme', look(element, 'color-scheme'), scheme);
      check({...scope, element: element.id || 'body'}, 'color', look(element, 'color'), ink);
      for (const alias of aliases) {
        check({...scope, element: element.id || 'body'}, alias, look(element, alias), 'outside-view');
      }
    }
  }
}
const unsupported = {};
for (const [name, selectors] of Object.entries(compatibility)) {
  unsupported[name] = Object.keys(selectors).filter(selector => !CSS.supports('selector(' + selector + ')'));
}
document.querySelector('#result').textContent = JSON.stringify({parsed, failures, unsupported});
</script>
"""

FOCUS_PROBE_HTML = """
<style>
#focus-probe { background: white; }
#focus-probe button { background: white; color: black; border: 1px solid black; outline: none; }
#focus-probe .permanent { outline: 3px solid black; outline-offset: 2px; }
#focus-probe #new-ring:focus-visible { outline: 3px solid black; outline-offset: 2px; }
#focus-probe #same-ring-wider:focus-visible { outline-width: 4px; }
#focus-probe #color-ring { outline: 3px solid #888; outline-offset: 2px; }
#focus-probe #color-ring:focus-visible { outline-color: black; }
#focus-probe #same-shadow { box-shadow: 0 0 0 3px black; }
#focus-probe #same-shadow:focus-visible { box-shadow: 0 0 0 3px black, 0 1px 3px black; }
</style>
<div id="focus-probe">
  <button class="permanent" id="first">Primero</button>
  <button class="permanent" id="second">Segundo</button>
  <button id="new-ring">Borde con anillo nuevo</button>
  <button class="permanent" id="same-ring-wider">Contorno un píxel más ancho</button>
  <button id="color-ring">Contorno con cambio de color</button>
  <button id="same-shadow">Sombra permanente</button>
</div>
<script>__WCAG__</script>
<script>
(() => {
  document.adoptedStyleSheets = [];
  const audit = window.__kzaWcag;
  audit.prepareContrast('#focus-probe');
  const focus = [...document.querySelectorAll('#focus-probe button')].map(button => {
    const previous = audit.focusSnapshot(button);
    button.focus();
    const ring = audit.focusContrast(button, previous);
    return {id: button.id, focusVisible: button.matches(':focus-visible'),
      width: ring.width ?? 0, ratio: ring.ratio};
  });
  const output = document.querySelector('#result');
  output.textContent = JSON.stringify({...JSON.parse(output.textContent), focus});
})();
</script>
"""


def delimiter_error(name: str, source: str, tokens: list[re.Match[str]]) -> str | None:
    """Reject unbalanced delimiters, including in unsupported rules and their bodies."""
    opened = []
    for token in tokens:
        value = token.group()
        if value in CLOSING.values():
            opened.append(token)
        elif value in CLOSING:
            if not opened or opened.pop().group() != CLOSING[value]:
                line = source.count('\n', 0, token.start()) + 1
                return f'{name}.css:{line}: unbalanced delimiter {value!r}'
    if opened:
        line = source.count('\n', 0, opened[-1].start()) + 1
        return f'{name}.css:{line}: unclosed delimiter {opened[-1].group()!r}'
    return None


def rule_preludes(source: str, tokens: list[re.Match[str]]) -> Counter[str]:
    """Find full rule preludes at brace boundaries, including in minified sheets."""
    counts: Counter[str] = Counter()
    start = 0
    for token in tokens:
        value = token.group()
        if value == '{':
            counts[source[start:token.start()].strip()] += 1
        if value in ('{', '}') or (value.startswith('/*') and not source[start:token.start()].strip()):
            start = token.end()
    return counts


def check_stylesheets(styles_dir: Path = STYLES_DIR) -> dict:
    """Compare every braced rule in the NEXO sheets with Chromium's CSSOM."""
    sources = {name: (styles_dir / f'{name}.css').read_text(encoding='utf-8') for name in SHEETS}
    # These sheets use braced rules, including @font-face and nested @media.
    # Ignore braces in comments and strings when counting the source rules.
    tokens = {name: list(CSS_TOKENS.finditer(source)) for name, source in sources.items()}
    expected = {name: sum(token.group() == '{' for token in entries) for name, entries in tokens.items()}
    errors = [error for name, source in sources.items()
              if (error := delimiter_error(name, source, tokens[name]))]
    config = json.dumps({'sources': sources, 'aliases': LEGACY_ALIASES,
                         'compatibility': COMPATIBILITY_SELECTORS}).replace('<', '\\u003c')
    with tempfile.TemporaryDirectory(prefix='kza-nexo-css-', dir=browser_profile_root()) as temporary:
        entry = Path(temporary, 'probe.html')
        html = PROBE_HTML.replace('__CONFIG__', config) + FOCUS_PROBE_HTML.replace('__WCAG__', WCAG_SCRIPT)
        entry.write_text(html, encoding='utf-8')
        result = subprocess.run(
            [browser_binary(), '--headless', '--no-first-run', '--disable-background-networking',
             f'--user-data-dir={entry.parent / "profile"}', '--dump-dom', entry.as_uri()],
            capture_output=True, text=True, timeout=30, check=True,
        )
    match = re.search(r'<pre id="result">(.+?)</pre>', result.stdout)
    assert match, 'Chromium did not render the NEXO stylesheet probe'
    probe = json.loads(unescape(match.group(1)))
    parsed = probe['parsed']
    exclusions = {}
    for name, selectors in probe['unsupported'].items():
        counts = rule_preludes(sources[name], tokens[name])
        for selector in selectors:
            occurrences = COMPATIBILITY_SELECTORS[name][selector]
            if counts[selector] == occurrences:
                expected[name] -= occurrences
                exclusions.setdefault(name, []).extend([selector] * occurrences)
    errors.extend(
        f'{name}.css: Chromium parsed {parsed[name]} of {expected[name]} CSS rules'
        for name in SHEETS if expected[name] != parsed[name]
    )
    assert not errors, '\n'.join(errors)
    assert not probe['failures'], f'Incorrect NEXO theme scope: {probe["failures"]}'
    accepted = {'new-ring', 'color-ring'}
    for sample in probe['focus']:
        assert sample['focusVisible'], f'Focus probe did not reach {sample["id"]}'
        passes = sample['width'] >= 2 and (sample['ratio'] or 0) >= 3
        assert passes == (sample['id'] in accepted), f'Incorrect focus-change measurement: {sample}'
    return {'passed': True, 'rules': parsed, 'excluded_unsupported': exclusions,
            'personas_themes': ['claro', 'oscuro'], 'focus_probe': probe['focus']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--styles-dir', type=Path, default=STYLES_DIR)
    args = parser.parse_args()
    print(json.dumps(check_stylesheets(args.styles_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
