'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { setImmediate: nextTurn } = require('node:timers/promises');
const { test } = require('node:test');

const nexo = path.join(__dirname, '../../src/dashboard/frontend/home/nexo');
const { createPeople } = require(path.join(nexo, 'people.js'));
const { createSession } = require(path.join(nexo, 'state.js'));

// Minimal DOM for the list and its detail hosts. The detail callbacks are probes
// so the matrix measures what Personas mounts and disposes under the real session.
class Element extends EventTarget {
  constructor(tag, document) {
    super();
    this.tag = tag;
    this.ownerDocument = document;
    this.children = [];
    this.parent = null;
    this.attributes = {};
    this.dataset = {};
    this.className = '';
    this.text = '';
    this.classList = {
      toggle: (name, enabled) => {
        const classes = new Set(this.className.split(/\s+/).filter(Boolean));
        if (enabled) classes.add(name);
        else classes.delete(name);
        this.className = [...classes].join(' ');
      },
    };
  }

  get textContent() {
    return this.text + this.children.map(child => child.textContent).join('');
  }

  set textContent(value) {
    this.replaceChildren();
    this.text = String(value);
  }

  append(...children) {
    for (const child of children) {
      if (child.parent) child.parent.children = child.parent.children.filter(node => node !== child);
      child.parent = this;
      this.children.push(child);
    }
  }

  replaceChildren(...children) {
    for (const child of this.children) child.parent = null;
    this.children = [];
    this.text = '';
    this.append(...children);
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  descendants() {
    return this.children.flatMap(child => [child, ...child.descendants()]);
  }

  contains(node) {
    return this === node || this.descendants().includes(node);
  }

  get isConnected() {
    return this.ownerDocument.root.contains(this);
  }

  querySelectorAll(selector) {
    const matches = {
      '[data-person-admin]': node => Object.hasOwn(node.dataset, 'personAdmin'),
      'input,select,textarea': node => ['input', 'select', 'textarea'].includes(node.tag),
      'button[type=submit],#confirm-person-removal': node =>
        (node.tag === 'button' && node.type === 'submit') || node.id === 'confirm-person-removal',
      '.person-dialog-status': node => node.className.split(/\s+/).includes('person-dialog-status'),
    }[selector];
    assert.ok(matches, 'Unexpected selector in Personas test DOM: ' + selector);
    return this.descendants().filter(matches);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  click() {
    this.dispatchEvent(new Event('click'));
  }
}

const PERMISSIONS = ['admin:manage', 'auth:manage', 'voices:read', 'voices:review'];
const ACCOUNTS = [
  { id: 'u-1', name: 'Ana María', permission_level: 'adult', is_active: true },
  { id: 'u-2', name: 'ANA  MARIA ', permission_level: 'admin', is_active: true },
  { id: 'u-3', name: 'Bruno', permission_level: 'adult', is_active: true },
];

async function verifyScope(role, permissions, selection, closure) {
  const doc = { createElement: tag => new Element(tag, doc) };
  doc.root = doc.createElement('body');
  const container = doc.createElement('section');
  const dialog = doc.createElement('dialog');
  doc.root.append(container, dialog);

  const session = createSession({ wall: () => 1000, monotonic: () => 0 });
  assert.equal(session.accept({ ok: true, role, permissions, expires_at: 10 }), true);
  const requests = [];
  const mounts = [];
  const track = kind => context => {
    const { personId, current, ambiguous } = context;
    assert.ok(mounts.every(mount =>
      mount.disposals === 1 || (mount.personId === personId && mount.kind !== kind)),
    'Dispose previous detail owners before mounting new ones');
    const mount = { kind, personId, current, ambiguous, currentAtMount: current(), disposals: 0 };
    mounts.push(mount);
    return () => { mount.disposals += 1; };
  };
  const people = createPeople({
    container, dialog, session,
    api: async (endpoint, { method = 'GET' } = {}) => {
      requests.push(method + ' ' + endpoint);
      return ACCOUNTS.map(account => ({ ...account }));
    },
    voiceDetails: track('voice'),
    accessDetails: track('access'),
  });
  people.open();
  await nextTurn();

  const canList = permissions.includes('admin:manage');
  const canSeeVoice = canList && permissions.includes('voices:read') && permissions.includes('voices:review');
  const canSeeAccess = canList && role === 'operator' && permissions.includes('auth:manage');
  assert.deepEqual(requests, canList ? ['GET /api/admin/users'] : [], 'Personas catalog requests');

  const grid = container.descendants().find(node => node.attributes['aria-label'] === 'Personas registradas');
  assert.equal(Boolean(grid), canList, 'Personas catalog visibility');
  if (grid && selection) {
    const card = grid.children.find(node => node.children[0]?.textContent === selection.name);
    assert.ok(card, 'Selectable card for ' + selection.name);
    card.click();
  }

  const dom = container.descendants();
  const detail = dom.find(node => node.tag === 'aside');
  const selected = selection ?? ACCOUNTS[0];
  assert.equal(detail?.children[0]?.children[0]?.textContent, canList ? selected.name : undefined);
  assert.equal(Boolean(detail?.descendants().some(node => node.attributes['aria-label'] === 'Perfiles de la persona')),
    canSeeVoice, 'Voice detail DOM requires voices:read and voices:review');
  assert.equal(Boolean(detail?.descendants().some(node => Object.hasOwn(node.dataset, 'personAccess'))),
    canSeeAccess, 'Access detail DOM requires operator, admin:manage and auth:manage');

  const visited = selection ? [ACCOUNTS[0].id, selected.id] : [selected.id];
  for (const [kind, allowed] of [['voice', canSeeVoice], ['access', canSeeAccess]]) {
    const scoped = mounts.filter(mount => mount.kind === kind);
    assert.deepEqual(scoped.map(mount => mount.personId), allowed ? visited : [], kind + ' mounts');
    for (const [index, mount] of scoped.entries()) {
      if (kind === 'access') {
        assert.equal(mount.ambiguous, mount.personId !== ACCOUNTS[2].id, 'Normalized homonym guard');
      }
      const active = index === scoped.length - 1;
      assert.equal(mount.currentAtMount, true, kind + ' current on mount');
      assert.equal(mount.current(), active, kind + ' current after selection');
      assert.equal(mount.disposals, active ? 0 : 1, kind + ' disposals after selection');
    }
  }

  for (const node of doc.root.descendants()) {
    const values = [node.text, node.id ?? '', node.className,
      ...Object.entries(node.attributes).flat(), ...Object.entries(node.dataset).flat()];
    for (const value of values) {
      assert.ok(!ACCOUNTS.some(({ id }) => String(value).includes(id)), 'Canonical person ID in DOM: ' + value);
    }
  }

  if (closure === 'logout') {
    session.clear();
    assert.ok(mounts.every(mount => !mount.current()), 'Logout invalidates active detail owners');
  }
  if (closure === 'leave') people.leave();
  else people.clear();
  assert.ok(mounts.every(mount => !mount.current() && mount.disposals === 1), 'All detail owners disposed once');
  assert.equal(container.children.length, 0, 'Personas DOM cleared on close');
  assert.deepEqual(requests, canList ? ['GET /api/admin/users'] : [], 'Selection and close make no catalog or detail requests');
}

for (const role of ['operator', 'home', 'evaluator']) {
  for (let mask = 0; mask < 2 ** PERMISSIONS.length; mask++) {
    const permissions = PERMISSIONS.filter((_, index) => mask & (1 << index));
    for (const selection of [null, ACCOUNTS[1], ACCOUNTS[2]]) {
      for (const closure of ['clear', 'leave', 'logout']) {
        const label = [role, permissions.join('+') || 'no permissions', selection?.name.trim() ?? 'default person', closure];
        test('Personas scope: ' + label.join(' / '), () => verifyScope(role, permissions, selection, closure));
      }
    }
  }
}
