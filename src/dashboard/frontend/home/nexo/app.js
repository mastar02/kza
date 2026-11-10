import '../request-id.js';
import '../home.js';
import './state.js';
import './people.js';
import './person-voices.js';
import '../../obs/src/live-privacy.js';
import '../../obs/src/conversation-channel.js';
import '../../obs/src/voice-privacy.js';
import '../../obs/src/voice-channel.js';
import './voice-owner.js';
import './voices.js';
import './access-owner.js';
import './access.js';
import './evaluation-owner.js';
import './evaluation.js';
import './routine-owner.js';
import './routines.js';
import './memory-owner.js';
import './memory.js';
import { createHouse } from './house.js';

const core = window.HomeUI, state = window.NexoState;
const byId = id => document.getElementById(id);
const session = state.createSession();
const roleLabels = { operator: 'Administración', home: 'Acceso de hogar', evaluator: 'Evaluación temporal' };
let epoch = 0, poller, sessionTimer, ticker, catalog = null, clock = null, lastError = null;
let roomId, layer = 'lights', house, logoutPending = false, authBusy = false, activeView = null;
let navigationVersion = 0, desiredView = null, accessView = null, viewsReady = true, viewDrain = Promise.resolve(true), privateSuspended = false;
const viewIds = { home: 'home-view', actions: 'actions-view', routines: 'routines-view', memory: 'memory-view', people: 'people-view', voices: 'voices-view', access: 'access-view', evaluation: 'evaluation-view' };
const viewHashes = { home: '#hogar', actions: '#acciones', routines: '#rutinas', memory: '#memoria', people: '#personas', voices: '#voces', access: '#accesos', evaluation: '#evaluacion' };
const auxiliaryView = view => ['people', 'voices', 'access', 'evaluation', 'routines', 'memory'].includes(view);
try { logoutPending = sessionStorage.getItem('kza.logoutPending') === 'true'; } catch (_) { /* Storage is optional. */ }

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  use.setAttribute('href', `/home/nexo/icons.svg#i-${name}`); svg.append(use); return svg;
}
function button(text, callback, className = 'text-button') {
  const control = node('button', className, text); control.type = 'button'; control.addEventListener('click', callback); return control;
}
const transport = core.createApi({ onDemo() { byId('demo-banner').hidden = false; } });
const api = state.createScopedApi({ session, transport, onRetired() { retire('Iniciá sesión para consultar tu hogar.'); } });
const actions = core.createActionController({ api, onChange: updateReceipts });
let voiceDetailSerial=0;
const people = window.NexoPeople.createPeople({ container: byId('people-view'), dialog: byId('person-dialog'), api, session, icon,
  voiceDetails({personId,container,current}) {
    let active=true;
    const valid=()=>active&&current()===true;
    if(!['voices:read','voices:review'].every(permission=>session.can(permission))) {
      container.replaceChildren(node('p','person-voice-status','Tu acceso permite gestionar la cuenta. La revisión de voces requiere permisos adicionales.'));
      return ()=>{active=false;container.replaceChildren();};
    }
    const connection=node('p','person-voice-status','Preparando consulta privada…');connection.dataset.voiceConnection='';container.replaceChildren(connection);
    const config=Object.freeze({scopeKey:'person-detail:'+ ++voiceDetailSerial,personId,profilesOnly:true,current:valid,
      mount:controller=>window.NexoPersonVoices.mountPersonVoices({container,controller,personId,current:valid,icon,onManage:()=>activate('voices')}),
      onStatus:message=>{if(valid()){const label=container.querySelector('[data-voice-connection]');if(label)label.textContent=message;}}
    });
    voices.open(config).then(opened=>{if(!opened&&valid())container.replaceChildren(node('p','person-voice-status','No se pudo abrir la consulta de perfiles. Actualizá la persona.'));});
    return ()=>{active=false;voices.leave();container.replaceChildren();};
  }
});
const voices = window.NexoVoiceOwner.createVoiceOwner({ session,
  fetch: window.fetch.bind(window), connect: url => new WebSocket(url),
  socketUrl: () => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/voices`,
  mount: controller => window.NexoVoices.mountVoices({ container: byId('voices-content'), dialog: byId('voice-dialog'), linkDialog: byId('voice-link-dialog'), controller, icon,
    onEvaluate: session.role() === 'operator' && session.can('auth:manage') ? prepareEvaluation : undefined }),
  onStatus: message => { byId('voice-connection').textContent = message; },
  verifyIdentity: async () => { const identity = await api('/api/admin/auth/whoami'); return identity?.ok === true && ['voices:read', 'voices:review'].every(permission => identity.permissions?.includes(permission)); },
  onUnauthorized: () => retire('Tu acceso a Voces dejó de estar vigente. Iniciá sesión nuevamente.')
});

async function verifyIdentity(permissions, role, options = {}) {
  const generation = options.generation ?? session.generation();
  if (generation !== session.generation() || !session.active()) return null;
  try {
    const identity = await transport('/api/admin/auth/whoami', { signal: options.signal });
    if (generation !== session.generation() || !session.active()) return null;
    return identity?.ok === true && identity.role === role && permissions.every(permission => identity.permissions?.includes(permission));
  } catch (error) {
    if (generation !== session.generation() || !session.active()) return null;
    return [401, 403].includes(error.status) ? false : null;
  }
}
const access = window.NexoAccessOwner.createAccessOwner({ session, fetch: window.fetch.bind(window),
  verifyIdentity: options => verifyIdentity(['auth:manage'], 'operator', options),
  onUnauthorized: () => retire('Tu acceso de administración dejó de estar vigente.') });
const evaluator = window.NexoEvaluationOwner.createEvaluationOwner({ session,
  fetch: window.fetch.bind(window), connect: url => new WebSocket(url),
  socketUrl: () => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/evaluation`,
  mount: controller => window.NexoEvaluation.mountEvaluation({ container: byId('evaluation-content'), controller }),
  onStatus: message => { byId('evaluation-connection').textContent = message; },
  verifyIdentity: options => verifyIdentity(['evaluation:read'], 'evaluator', options),
  onUnauthorized: () => retire('Tu acceso de evaluación dejó de estar vigente.') });
// One owner per page: it survives navigation so writes are never cut, and is cleared only at session boundaries.
// Another tab switched the shared cookie to another account: end this tab's cached session and
// read the current one again; nothing pending of the previous account is kept.
const routines = window.NexoRoutineOwner.createRoutineOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let routinesView = null;
function closeRoutines() {
  const dispose = routinesView; routinesView = null; dispose?.();
  return routines.leave();
}
// Memoria follows the same owner rules; every open asks whoami before reading (F5).
const memory = window.NexoMemoryOwner.createMemoryOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let memoryView = null;
function closeMemory() {
  const dispose = memoryView; memoryView = null; dispose?.();
  return memory.leave();
}
function closeAccess() {
  const dispose = accessView; accessView = null; dispose?.();
  return access.leave();
}
async function prepareEvaluation(values) {
  const generation = session.generation();
  if (activeView !== 'voices' || session.role() !== 'operator' || !session.can('auth:manage')) return false;
  let selections;
  try { selections = window.NexoAccessOwner.parseSelections(values); } catch { return false; }
  if (!selections.length || !await activate('access')) return false;
  if (generation !== session.generation() || activeView !== 'access' || !session.can('auth:manage')) return false;
  return accessView?.openEvaluation(selections) === true;
}
function trackViewDrain(promises) {
  viewsReady = false;
  const pending = Promise.all(promises).then(values => values.every(value => value !== false), () => false);
  viewDrain = pending;
  pending.then(ok => { if (viewDrain === pending) { viewsReady = ok; lockAuthentication(authBusy); } });
  return pending;
}
function drainViews() {
  people.leave();
  return trackViewDrain([voices.leave(), evaluator.leave(), closeAccess(), closeRoutines(), closeMemory()]);
}

function pendingLogout(value) {
  logoutPending = value;
  try { if (value) sessionStorage.setItem('kza.logoutPending', 'true'); else sessionStorage.removeItem('kza.logoutPending'); } catch (_) { /* No credentials or identity use storage. */ }
  byId('retry-logout').hidden = !value;
}
function lockAuthentication(value) {
  authBusy = value;
  for (const control of byId('login-form').elements) control.disabled = value || !viewsReady;
  for (const control of [byId('retry-logout'), byId('logout')]) control.disabled = value;
}
function destroyHouse() {
  house?.destroy(); house = null;
  document.body.classList.remove('map-expanded');
  byId('expand-map').setAttribute('aria-expanded', 'false');
}
// endSession=false only when the page is hidden or the server could not be asked: pending
// routine identities then survive in this tab's sessionStorage until the owning session ends.
function retire(message = '', { endSession = true } = {}) {
  epoch++; navigationVersion++; desiredView = null; privateSuspended = false; session.clear(); transport.invalidateSession();
  clearTimeout(sessionTimer); poller?.stop(); poller = null; ticker?.stop(); ticker = null;
  people.clear(); const accessClosing = closeAccess(), routinesClosing = closeRoutines(), memoryClosing = closeMemory();
  trackViewDrain([voices.clear(), evaluator.clear(), access.clear(), accessClosing, routinesClosing, routines.clear({ wipe: endSession }),
    memoryClosing, memory.clear({ wipe: endSession })]);
  actions.clear(); destroyHouse(); catalog = clock = lastError = null; roomId = undefined; activeView = null;
  byId('voice-connection').textContent = ''; byId('evaluation-connection').textContent = '';
  byId('view-transition-status').hidden = true;
  for (const id of ['room-list', 'room-controls', 'action-list', 'app-nav', 'rail-nav', 'additional-tools']) byId(id).replaceChildren();
  for (const id of ['lights-total', 'entities-total', 'rooms-total', 'room-count']) byId(id).textContent = '—';
  byId('nexo-workspace').hidden = true; byId('login-panel').hidden = false;
  byId('logout').hidden = true;
  byId('demo-banner').hidden = true; byId('token').value = '';
  byId('session-label').textContent = 'Acceso local'; byId('login-error').textContent = message;
}
function armSessionTimer() {
  clearTimeout(sessionTimer);
  if (!session.active()) { retire('Tu sesión venció. Iniciá sesión nuevamente.'); return; }
  sessionTimer = setTimeout(armSessionTimer, Math.min(session.remaining(), 30000));
}
function freshness() {
  return { ...(clock?.read() || { stale: true, ageMs: null }), stale: !!lastError || !session.can('home:read') || !clock || clock.read().stale };
}
function entityById(id) { return catalog?.entities?.find(entity => entity.entity_id === id) || catalog?.unassigned_entities?.find(entity => entity.entity_id === id); }
function observedValue(entity) {
  if (entity.domain === 'climate') return Number.isFinite(entity.temperature) ? `${entity.temperature} °C · objetivo` : 'Temperatura objetivo desconocida';
  if (entity.domain === 'light') return Number.isFinite(entity.brightness_pct) ? `${entity.brightness_pct} % · brillo observado` : 'Brillo desconocido';
  if (entity.domain === 'cover') return Number.isFinite(entity.position) ? `${entity.position} % · abierto` : 'Posición desconocida';
  return null;
}
function renderObservedValue(output, entity) {
  if (entity.domain === 'climate' && Number.isFinite(entity.temperature)) {
    output.replaceChildren(node('strong', '', `${entity.temperature} °C`), node('span', '', ' · objetivo'));
  } else output.textContent = observedValue(entity);
}
function updateObservedCards() {
  for (const card of document.querySelectorAll('.device-card[data-entity-id]')) {
    const entity = entityById(card.dataset.entityId);
    card.querySelector('.device-state').textContent = entity ? state.entityState(entity) : 'Ya no figura en el catálogo.';
    if (entity) card.querySelector('h3').textContent = entity.name;
    const value = card.querySelector('.device-value');
    if (value) { if (entity) renderObservedValue(value, entity); else value.textContent = 'Estado desconocido'; }
  }
  const heading = byId('room-controls').querySelector('.selected-room-heading');
  if (heading) heading.textContent = catalog.rooms.find(room => room.room_id === roomId)?.name || 'Sin ambiente asignado';
}
function canExecute(entity, operation) {
  return session.can('home:control') && !!entity && core.entityCanControl(entity, operation, freshness().stale, clock?.now());
}
function updateTemporal() {
  if (!session.active()) { retire('Tu sesión venció. Iniciá sesión nuevamente.'); return; }
  const observed = freshness();
  byId('connection-status').textContent = lastError ? `No se pudo actualizar. ${lastError.message}` : !catalog ? 'Consultando el catálogo…' : observed.stale ? 'El estado está vencido o no está disponible. Actualizá antes de operar.' : `Estado observado · ${Math.max(0, Math.floor(observed.ageMs / 1000))} s`;
  byId('connection-status').classList.toggle('is-stale', observed.stale);
  for (const control of document.querySelectorAll('[data-control]')) {
    const entity = entityById(control.dataset.entityId);
    const record = actions.get(control.dataset.entityId, control.dataset.operation);
    control.disabled = !canExecute(entity, control.dataset.operation) || record?.pending === true;
  }
}
function updateReceipts() {
  if (!session.active()) return;
  for (const output of document.querySelectorAll('[data-receipt]')) {
    const record = actions.get(output.dataset.entityId, output.dataset.operation);
    output.textContent = record ? [core.actionLabel(record.status), core.effectSummary(record.effects)].filter(Boolean).join(' · ') : '';
    output.dataset.status = record?.status || '';
  }
  updateTemporal();
}
async function execute(id, operation, values, errorNode) {
  const entity = entityById(id);
  const fields = core.controlFields(entity, operation);
  if (!canExecute(entity, operation) || fields.some(field => !core.validControlValue(field, values[field.name]))) {
    errorNode.textContent = 'El estado, los permisos o los valores cambiaron. Actualizá el control.';
    updateTemporal(); return;
  }
  errorNode.textContent = '';
  await actions.execute(id, operation, values);
  if (session.can('home:read')) poller?.refresh();
}
function deviceCard(entity, expanded = new Set()) {
  const card = node('article', 'device-card panel'); card.dataset.entityId = entity.entity_id; card.dataset.domain = entity.domain;
  const heading = node('div', 'device-heading'); heading.append(icon(entity.domain === 'climate' ? 'climate' : entity.domain === 'cover' ? 'blind' : 'bulb'));
  const title = node('div'); title.append(node('h3', '', entity.name), node('p', 'device-state', state.entityState(entity))); heading.append(title); card.append(heading);
  if (observedValue(entity) !== null) { const value = node('p', 'device-value'); renderObservedValue(value, entity); card.append(value); }
  const operations = (entity.capabilities || []).filter(operation => Object.hasOwn(core.OPERATION_LABELS, operation));
  if (!operations.length) card.append(node('p', 'fine-print', 'No hay controles autorizados para este dispositivo.'));
  const binary = node('div', 'power-actions'); card.append(binary);
  for (const operation of operations) {
    const form = node('form', 'operation-form'); form.dataset.entityId = entity.entity_id;
    const fields = core.controlFields(entity, operation), inputs = new Map();
    for (const field of fields) {
      const label = node('label', 'control-field', field.label), input = node(field.options ? 'select' : 'input');
      input.name = field.name; input.setAttribute('aria-label', `${entity.name} · ${field.label}`);
      if (field.options) {
        for (const [value, text] of field.options) { const option = node('option', '', text); option.value = value; input.append(option); }
        input.value = field.name === 'hvac_mode' ? entity.state : '';
      } else {
        input.type = 'number'; input.placeholder = 'Elegir valor';
        if (Number.isFinite(field.min)) input.min = field.min;
        if (Number.isFinite(field.max)) input.max = field.max;
        if (Number.isFinite(field.step)) input.step = field.step;
        const value = ({ brightness_pct: entity.brightness_pct, color_temp_kelvin: entity.color_temp_kelvin, temperature: entity.temperature, position: entity.position })[field.name];
        if (Number.isFinite(value) && !field.optional) input.value = value;
        input.required = !field.optional;
        if (field.name === 'brightness_pct' && Number.isFinite(value) && !field.optional) {
          input.type = 'range';
          const output = node('output', 'range-output', `${value} %`);
          input.addEventListener('input', () => { output.textContent = `${input.value} %`; }); label.append(output);
        }
      }
      input.dataset.control = ''; input.dataset.entityId = entity.entity_id; input.dataset.operation = operation;
      label.append(input); form.append(label); inputs.set(field.name, input);
    }
    const submit = node('button', fields.length ? 'apply-button' : 'operation-button', core.OPERATION_LABELS[operation]); submit.type = 'submit';
    submit.dataset.control = ''; submit.dataset.entityId = entity.entity_id; submit.dataset.operation = operation; form.append(submit);
    const error = node('p', 'control-error'); error.setAttribute('role', 'alert'); form.append(error);
    const receipt = node('p', 'action-receipt'); receipt.dataset.receipt = ''; receipt.dataset.entityId = entity.entity_id; receipt.dataset.operation = operation; receipt.setAttribute('role', 'status'); form.append(receipt);
    form.addEventListener('submit', event => {
      event.preventDefault();
      const values = Object.fromEntries(fields.map(field => { const input = inputs.get(field.name); return [field.name, field.options ? input.value : input.value === '' ? null : input.valueAsNumber]; }));
      execute(entity.entity_id, operation, values, error).catch(() => { if (session.active()) error.textContent = 'No se pudo consultar el resultado. Actualizá para verificarlo.'; });
    });
    if (!fields.length) { form.classList.add('binary-operation'); binary.append(form); }
    else if (operation === 'color_temp_kelvin') {
      const detail = node('details', 'advanced-control'); detail.dataset.entityId = entity.entity_id; detail.open = expanded.has(entity.entity_id);
      detail.append(node('summary', '', 'Temperatura de luz'), form); card.append(detail);
    } else card.append(form);
  }
  return card;
}
function selectedEntities() {
  const values = roomId === null ? catalog?.unassigned_entities || [] : state.roomEntities(catalog, roomId, 'all');
  return values.filter(entity => layer === 'all' || entity.domain === (layer === 'climate' ? 'climate' : 'light'));
}
function renderControls() {
  if (!catalog || !session.can('home:read')) return;
  const expanded = new Set([...document.querySelectorAll('.advanced-control[open]')].map(detail => detail.dataset.entityId));
  const controls = byId('room-controls'); controls.replaceChildren();
  const selected = catalog.rooms.find(room => room.room_id === roomId);
  controls.append(node('h2', 'selected-room-heading', selected?.name || 'Sin ambiente asignado'));
  const entities = selectedEntities();
  for (const entity of entities) controls.append(deviceCard(entity, expanded));
  if (!entities.length) controls.append(node('p', 'panel empty-panel', 'Este ambiente no tiene dispositivos de esta capa en el catálogo.'));
  const all = byId('action-list'); all.replaceChildren();
  for (const entity of catalog.entities) all.append(deviceCard(entity, expanded));
  if (!catalog.entities.length) all.append(node('p', 'panel empty-panel', 'No hay dispositivos en el catálogo actual.'));
  updateReceipts();
}
function renderCatalog() {
  if (!catalog || !session.can('home:read')) return;
  const rooms = catalog.rooms;
  const previousRoom = roomId;
  if (roomId === undefined || roomId !== null && !rooms.some(room => room.room_id === roomId)) roomId = rooms[0]?.room_id ?? null;
  byId('rooms-total').textContent = String(rooms.length); byId('room-count').textContent = String(rooms.length);
  byId('entities-total').textContent = String(catalog.entities.length);
  byId('lights-total').textContent = String(catalog.entities.filter(entity => entity.domain === 'light' && entity.available && entity.state === 'on').length);
  const list = byId('room-list'); list.replaceChildren();
  function addRoom(id, name, count) {
    const control = button('', () => { roomId = id; renderCatalog(); }, 'room-choice'); control.classList.toggle('active', id === roomId); control.setAttribute('aria-pressed', String(id === roomId));
    if (id !== null) control.dataset.roomId = id;
    control.append(icon('house')); const text = node('span'); text.append(node('strong', '', name), node('small', '', `${count} dispositivo${count === 1 ? '' : 's'}`)); control.append(text); list.append(control);
  }
  for (const room of rooms) addRoom(room.room_id, room.name, room.entity_ids.length);
  if (catalog.unassigned_entities.length) addRoom(null, 'Sin ambiente', catalog.unassigned_entities.length);
  if (!rooms.length && !catalog.unassigned_entities.length) list.append(node('p', 'empty-panel', 'No hay ambientes en el catálogo.'));
  // Refreshing an observation must not discard a keyboard user's draft or focus.
  const focused = document.activeElement?.closest('.operation-form');
  const stillSelected = !focused?.closest('#room-controls') || selectedEntities().some(entity => entity.entity_id === focused.dataset.entityId);
  if (!focused || previousRoom !== roomId || !entityById(focused.dataset.entityId) || !stillSelected) renderControls();
  else updateObservedCards();
  updateTemporal();
}
function renderNavigation() {
  const navigation = [
    { id: 'home', label: 'Hogar 3D', icon: 'house' },
    { id: 'voices', label: 'Voces', icon: 'mic' },
    { id: 'people', label: 'Personas', icon: 'user' },
    { id: 'actions', label: 'Acciones', icon: 'list' },
    { id: 'routines', label: 'Rutinas', icon: 'clock' },
    { id: 'memory', label: 'Memoria', icon: 'layers' },
    { id: 'access', label: 'Accesos', icon: 'shield' },
    { id: 'evaluation', label: 'Evaluación', icon: 'mic' }
  ].filter(view => session.views().includes(view.id));
  for (const target of ['app-nav', 'rail-nav']) {
    const container = byId(target); container.replaceChildren();
    for (const view of navigation) {
      const control = view.href ? node('a', target === 'rail-nav' ? 'rail-button' : '', '') : button('', () => activate(view.id), target === 'rail-nav' ? 'rail-button' : '');
      if (view.href) control.href = view.href;
      control.dataset.view = view.id; control.setAttribute('aria-label', view.label); control.append(icon(view.icon));
      control.append(node('span', target === 'rail-nav' ? 'rail-tooltip' : '', view.label)); container.append(control);
    }
  }
  railOverflow();
  const tools = byId('additional-tools'); tools.replaceChildren();
  if (session.can('diagnostics:read')) { const link = node('a', 'text-button', 'Micrófonos y diagnóstico ↗'); link.href = '/home/legacy.html'; tools.append(link); }
}
async function activate(view, push = true, force = false) {
  if (!session.views().includes(view) || !Object.hasOwn(viewIds, view)) return false;
  if (!force && activeView === view && desiredView === view) return true;
  const version = ++navigationVersion, generation = session.generation();
  desiredView = view;
  if (push && activeView !== view) history.pushState({ view }, '', viewHashes[view]);
  activeView = null;
  for (const id of Object.values(viewIds)) byId(id).hidden = true;
  byId('workspace-toolbar').hidden = true;
  const status = byId('view-transition-status'); status.hidden = false; status.textContent = 'Preparando vista…';
  poller?.setHidden(true); ticker?.stop();
  if (view !== 'home') destroyHouse();
  const stillCurrent = () => version === navigationVersion && desiredView === view && generation === session.generation() && session.views().includes(view);
  try {
    const drained = await drainViews();
    if (!stillCurrent()) return false;
    if (!drained) throw Error('La vista anterior no terminó de liberar sus recursos. Recargá la página para continuar.');
    activeView = view; byId(viewIds[view]).hidden = false; status.hidden = true;
    byId('restricted-view').hidden = true;
    byId('workspace-toolbar').hidden = auxiliaryView(view);
    if (document.hidden && auxiliaryView(view)) {
      privateSuspended = true;
      status.hidden = false; status.textContent = 'La consulta se reanudará cuando vuelvas a esta pestaña.';
      return true;
    }
    let opened = true;
    if (view === 'people') people.open();
    if (view === 'voices') opened = await voices.open();
    if (view === 'access') {
      accessView = window.NexoAccess.mountAccess({ container: byId('access-view'), dialog: byId('access-dialog'),
        secretDialog: byId('access-secret-dialog'), controller: access, icon });
      opened = await access.open();
    }
    if (view === 'evaluation') opened = await evaluator.open();
    if (view === 'routines') {
      routinesView = window.NexoRoutines.mountRoutines({ container: byId('routines-view'), dialog: byId('routine-dialog'),
        confirmDialog: byId('routine-confirm-dialog'), owner: routines, icon });
      opened = await routines.open();
    }
    if (view === 'memory') {
      memoryView = window.NexoMemory.mountMemory({ container: byId('memory-view'), confirmDialog: byId('memory-confirm-dialog'), owner: memory, icon });
      opened = await memory.open();
    }
    if (!stillCurrent()) return false;
    if (!opened) throw Error('No pudimos abrir esta vista. Podés reintentar la consulta.');
    privateSuspended = false;
    poller?.setHidden(document.hidden || auxiliaryView(view));
    if (document.hidden || auxiliaryView(view)) ticker?.stop(); else ticker?.start();
    for (const control of document.querySelectorAll('[data-view]')) {
      const selected = control.dataset.view === view; control.classList.toggle('active', selected);
      if (selected) control.setAttribute('aria-current', 'page'); else control.removeAttribute('aria-current');
    }
    revealActiveRail();
    if (view === 'home' && !house) {
      byId('map-loading').hidden = false;
      house = createHouse({ container: byId('house-canvas'), onReady(ready) {
        byId('map-loading').hidden = true;
        for (const id of ['view-3d', 'view-top', 'zoom-in', 'zoom-out', 'reset-view']) byId(id).disabled = !ready;
      } });
      cameraView('3d', false);
    }
    if (!auxiliaryView(view)) renderControls();
    return true;
  } catch (error) {
    if (stillCurrent()) { status.hidden = false; status.replaceChildren(node('span', '', error.message), button('Reintentar', () => activate(view, false, true))); }
    return false;
  }
}
function requestedView() {
  const requested = Object.keys(viewHashes).find(view => viewHashes[view] === location.hash) || 'home';
  const available = session.views().filter(view => Object.hasOwn(viewIds, view));
  return available.includes(requested) ? requested : available[0];
}
function cameraView(view, apply = true) {
  if (apply) house?.setView(view);
  for (const [id, selected] of [['view-3d', view === '3d'], ['view-top', view === 'top']]) {
    byId(id).classList.toggle('active', selected); byId(id).setAttribute('aria-pressed', String(selected));
  }
}
async function bootstrap() {
  if (logoutPending) { retire('No se confirmó el cierre anterior. Reintentá cerrar la sesión.'); return; }
  const started = ++epoch;
  try {
    const identity = await api('/api/admin/auth/whoami');
    if (started !== epoch) return;
    if (!session.accept(identity)) { retire('El servicio no devolvió una sesión vigente.'); return; }
    byId('login-panel').hidden = true; byId('nexo-workspace').hidden = false; byId('logout').hidden = false;
    byId('session-label').textContent = roleLabels[session.role()]; byId('login-error').textContent = '';
    renderNavigation(); armSessionTimer();
    const home = session.can('home:read'); byId('restricted-view').hidden = Boolean(requestedView());
    for (const id of Object.values(viewIds)) byId(id).hidden = true;
    byId('refresh').hidden = !home;
    if (requestedView()) await activate(requestedView(), false, true);
    if (started !== epoch || !session.active()) return;
    if (!home) { byId('connection-status').textContent = 'Sesión vigente.'; return; }
    const generation = session.generation();
    poller = core.createPoller({
      load: async signal => { await actions.reconcile(); return api('/api/home/catalog', { signal }); },
      onData(data) {
        if (session.generation() !== generation || !session.can('home:read')) return;
        if (![data?.rooms, data?.entities, data?.unassigned_entities].every(Array.isArray)) { lastError = new Error('El catálogo no está disponible.'); updateTemporal(); return; }
        catalog = data; clock = core.createSnapshotClock(data); lastError = null; renderCatalog();
      },
      onError(error) { if (session.generation() === generation && session.active()) { lastError = error; updateTemporal(); } }
    });
    ticker = core.createFreshnessTicker({ read: () => freshness(), onTick: updateTemporal });
    poller.setHidden(document.hidden || auxiliaryView(activeView)); if (!document.hidden && !auxiliaryView(activeView)) ticker.start(); poller.start();
  } catch (error) {
    if (started === epoch) retire(error.status === 401 ? 'Ingresá para consultar tu hogar.' : error.message, { endSession: error.status === 401 });
  }
}

byId('login-form').addEventListener('submit', async event => {
  event.preventDefault(); if (authBusy) { byId('token').value = ''; return; } lockAuthentication(true);
  let token = byId('token').value; retire(); pendingLogout(false); const started = epoch;
  try {
    if (!await viewDrain || started !== epoch) throw Error('La vista anterior todavía no terminó de cerrar.');
    await api('/api/admin/auth/login', { method: 'POST', body: { token } }); if (started === epoch) await bootstrap();
  }
  catch (error) { if (started === epoch || error.status === 401 && !session.active()) byId('login-error').textContent = error.status === 401 ? 'La credencial no es válida.' : error.message; }
  finally { token = null; lockAuthentication(false); }
});
async function logout() {
  if (authBusy) return;
  lockAuthentication(true);
  pendingLogout(true); retire('Cerrando sesión…');
  try { await api('/api/admin/auth/logout', { method: 'POST' }); pendingLogout(false); byId('login-error').textContent = 'Sesión cerrada.'; }
  catch (error) {
    if (error.status === 401) { pendingLogout(false); byId('login-error').textContent = 'La sesión ya no está vigente.'; }
    else byId('login-error').textContent = 'Se ocultaron los datos. No pudimos confirmar el cierre de la sesión.';
  }
  finally { lockAuthentication(false); }
}
byId('logout').addEventListener('click', logout); byId('retry-logout').addEventListener('click', logout);
byId('refresh').addEventListener('click', () => { if (session.can('home:read')) { actions.reconcile(); poller?.refresh(); } });
for (const control of byId('layer-tabs').querySelectorAll('[data-layer]')) control.addEventListener('click', () => {
  layer = control.dataset.layer;
  for (const item of byId('layer-tabs').querySelectorAll('[data-layer]')) { const active = item === control; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); }
  renderControls();
});
for (const [id, callback] of [
  ['view-3d', () => cameraView('3d')], ['view-top', () => cameraView('top')],
  ['zoom-in', () => house?.zoom(.12)], ['zoom-out', () => house?.zoom(-.12)], ['reset-view', () => { house?.reset(); cameraView('3d', false); }]
]) byId(id).addEventListener('click', callback);
byId('expand-map').addEventListener('click', () => { const expanded = document.body.classList.toggle('map-expanded'); byId('expand-map').setAttribute('aria-expanded', String(expanded)); });
// On phones the view group scrolls inside the bottom rail: edge fades show there is more, the
// focused or active item is kept fully visible, and every target stays 44 px.
const RAIL_ITEM = 44, RAIL_PEEK = 22;
function railOverflow() {
  const rail = byId('rail-nav'), items = rail.querySelectorAll('[data-view]').length;
  // When the views do not fit, space them so the item at the edge is cut in half: the fade then lies over
  // a visibly partial item instead of a whole one.
  const width = rail.clientWidth - parseFloat(getComputedStyle(rail).paddingLeft) - parseFloat(getComputedStyle(rail).paddingRight);
  const whole = Math.floor((width - RAIL_PEEK) / (RAIL_ITEM + 4));
  const gap = window.matchMedia('(max-width:620px)').matches && items * RAIL_ITEM > width && whole > 0 ? (width - RAIL_PEEK) / whole - RAIL_ITEM : 0;
  const value = gap.toFixed(2) + 'px';
  if (rail.style.getPropertyValue('--rail-gap') !== value) rail.style.setProperty('--rail-gap', value);
  const max = rail.scrollWidth - rail.clientWidth;
  rail.classList.toggle('more-start', max > 1 && rail.scrollLeft > 1);
  rail.classList.toggle('more-end', max > 1 && rail.scrollLeft < max - 1);
}
function revealActiveRail() { railOverflow(); byId('rail-nav').querySelector('.active')?.scrollIntoView({ block: 'nearest', inline: 'nearest' }); railOverflow(); }
byId('rail-nav').addEventListener('focusin', event => event.target.closest?.('[data-view]')?.scrollIntoView({ block: 'nearest', inline: 'nearest' }));
byId('rail-nav').addEventListener('scroll', railOverflow, { passive: true });
// A layout change (rotation, phone width) keeps the active view inside the rail.
window.addEventListener('resize', revealActiveRail);
byId('help-button').addEventListener('click', () => byId('info-dialog').showModal());
for (const id of ['dialog-close', 'dialog-done']) byId(id).addEventListener('click', () => byId('info-dialog').close());
document.addEventListener('keydown', event => { if (event.key === 'Escape') { document.body.classList.remove('map-expanded'); byId('expand-map').setAttribute('aria-expanded', 'false'); } });
document.addEventListener('visibilitychange', () => {
  if (!session.active()) { if (!byId('nexo-workspace').hidden) retire('Tu sesión venció.'); return; }
  armSessionTimer();
  if (document.hidden || auxiliaryView(activeView)) ticker?.stop(); else { updateTemporal(); ticker?.start(); }
  poller?.setHidden(document.hidden || auxiliaryView(activeView));
  const privateView = desiredView || activeView;
  if (auxiliaryView(privateView)) {
    if (document.hidden) { privateSuspended = true; ++navigationVersion; activeView = privateView; void drainViews(); }
    else if (privateSuspended) void activate(privateView, false, true);
    else if (privateView === 'people') people.refresh();
  }
});
window.addEventListener('popstate', () => activate(requestedView(), false));
window.addEventListener('pagehide', () => retire('', { endSession: false }));
window.addEventListener('pageshow', event => { if (event.persisted) bootstrap(); });
bootstrap();
