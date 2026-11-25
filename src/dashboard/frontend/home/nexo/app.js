import '../request-id.js';
import '../home.js';
import './sections.js';
import './state.js';
import './people.js';
import './person-voices.js';
import '../../obs/src/live-privacy.js';
import '../../obs/src/conversation-channel.js';
import '../../obs/src/voice-privacy.js';
import '../../obs/src/voice-channel.js';
import '../../obs/src/voice-filters.js';
import './voice-owner.js';
import './voices.js';
import './access-owner.js';
import './access.js';
import './evaluation-owner.js';
import './evaluation.js';
import './notices-owner.js';
import './notices.js';
import './routine-owner.js';
import './routines.js';
import './tasks-owner.js';
import './tasks.js';
import './activity-owner.js';
import './activity.js';
import './memory-owner.js';
import './memory.js';
import './my-memory-owner.js';
import './my-memory.js';
import './my-voice-owner.js';
import './my-voice.js';
import './voice-consents-owner.js';
import './voice-consents.js';
import { createHouse, thermalReading } from './house.js';

const core = window.HomeUI, state = window.NexoState, { sectionDestination } = window.NexoSections;
const byId = id => document.getElementById(id);
const session = state.createSession();
const roleLabels = { operator: 'Administración', home: 'Acceso de hogar', evaluator: 'Evaluación temporal' };
let epoch = 0, poller, sessionTimer, ticker, catalog = null, clock = null, lastError = null;
let displayNames = new Map();
const entityIds = new WeakMap(), roomIds = new WeakMap();
let roomId, layer = 'lights', house, logoutPending = false, authBusy = false, activeView = null;
let staleFocus = null;
let navigationVersion = 0, desiredView = null, accessView = null, viewsReady = true, viewDrain = Promise.resolve(true), privateSuspended = false;
const viewIds = { home: 'home-view', actions: 'actions-view', routines: 'routines-view', tasks: 'tasks-view', activity: 'activity-view', memory: 'memory-view', mymemory: 'my-memory-view', people: 'people-view', voices: 'voices-view', access: 'access-view', evaluation: 'evaluation-view' };
const lastSectionViews = new Map();
const auxiliaryView = view => ['people', 'voices', 'access', 'evaluation', 'routines', 'tasks', 'activity', 'memory', 'mymemory'].includes(view);
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
const actions = core.createActionController({ api, onChange(_key, record) {
  activity.recordAction(record);
  updateReceipts();
} });
function activityAccountChanged(generation) {
  if (generation !== session.generation()) return;
  activity.clear();
  setTimeout(() => {
    if (generation !== session.generation()) return;
    retire('La cuenta cambió en otra pestaña.'); void bootstrap();
  }, 0);
}
const notices = window.NexoNoticesOwner.createNoticesOwner({ api, session, onAccountChanged: activityAccountChanged });
let noticesPanel = null;
function closeNotices({ endSession = false } = {}) {
  if (noticesPanel) { noticesPanel.dispose(); noticesPanel.element.remove(); noticesPanel = null; }
  if (endSession) notices.clear();
  else notices.leave();
}
function openNotices() {
  if (document.hidden || activeView !== 'home' || noticesPanel || !state.canReadNotices(session)) return;
  const element = node('section', 'panel notice-panel'); element.id = 'home-notices'; element.hidden = true;
  byId('home-view').querySelector('.main-column').append(element);
  noticesPanel = { element, dispose: window.NexoNotices.mountNotices({ container: element, owner: notices }) };
  notices.open();
}
let voiceDetailSerial=0;
const people = window.NexoPeople.createPeople({ container: byId('people-view'), dialog: byId('person-dialog'), api, session,
  accessDetails({ personId, personName, ambiguous, container, current }) {
    const dispose = window.NexoAccess.mountAccess({ container, dialog: byId('access-dialog'),
      secretDialog: byId('access-secret-dialog'), controller: access, personId, personName, ambiguous, current,
      onManage: () => activate('access'), icon });
    if (current()) void access.open();
    return dispose;
  },
  voiceDetails({personId,container,current}) {
    let active=true;
    const valid=()=>active&&current()===true;
    const connection=node('p','person-voice-status','Consultando perfiles de voz…');connection.dataset.voiceConnection='';container.replaceChildren(connection);
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
    roomName: id => session.can('home:read') ? catalog?.rooms.find(room => room.room_id === id)?.name : null,
    onEvaluate: session.role() === 'operator' && session.can('auth:manage') ? prepareEvaluation : undefined,
    // U3b-4a-3: 'Verificada por …' and the pause next to each profile, from the voice administration owner.
    profileNote: id => voiceAdmin.profileNote(id), onProfileNotes: listener => voiceAdmin.subscribe(listener) }),
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
  mount: controller => window.NexoEvaluation.mountEvaluation({ container: byId('evaluation-content'), controller,
    getSession: () => session.active() ? { role: session.role(), expiresAt: session.expiresAt() } : null }),
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
const tasks = window.NexoTasksOwner.createTasksOwner({ api, session,
  onAccountChanged(generation) {
    setTimeout(() => {
      if (generation !== session.generation()) return;
      retire('La cuenta cambió en otra pestaña.'); void bootstrap();
    }, 0);
  } });
let tasksView = null;
function closeTasks() {
  const dispose = tasksView; tasksView = null; dispose?.();
  return tasks.leave();
}
// Memoria follows the same owner rules; every open asks whoami before reading (F5).
const memory = window.NexoMemoryOwner.createMemoryOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let memoryView = null;
function closeMemory() {
  const dispose = memoryView; memoryView = null; dispose?.();
  return memory.leave();
}
// 'Mis recuerdos' has its own owner (no shared state with Memoria); the same session rules apply.
const myMemory = window.NexoMyMemoryOwner.createMyMemoryOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let myMemoryView = null;
// U3b-4a-3: the person's voice opt-in lives inside 'Mis recuerdos' with its own owner and the same session rules.
const myVoice = window.NexoMyVoiceOwner.createMyVoiceOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let myVoiceView = null;
function closeMyMemory() {
  const dispose = myMemoryView, disposeVoice = myVoiceView; myMemoryView = myVoiceView = null; dispose?.(); disposeVoice?.();
  return Promise.all([myMemory.leave(), myVoice.leave()]).then(values => values.every(Boolean));
}
// U3b-4a-3: the voice stage administration (switch, consents, resume) is a panel inside Voces.
const voiceAdmin = window.NexoVoiceConsentsOwner.createVoiceConsentsOwner({ api, session,
  onAccountChanged: () => { setTimeout(() => { retire('La cuenta cambió en otra pestaña.'); void bootstrap(); }, 0); } });
let voiceAdminView = null;
function closeVoiceAdmin() {
  const dispose = voiceAdminView; voiceAdminView = null; dispose?.();
  return voiceAdmin.leave();
}
const activity = window.NexoActivityOwner.createActivityOwner({ api, session, actions, notices,
  onAccountChanged: activityAccountChanged,
  deviceLabel(id) {
    const name = displayNames.get(id);
    const entityRoom = entityById(id)?.room_id;
    const room = catalog?.rooms.find(item => item.room_id === entityRoom)?.name;
    return name ? [name, room].filter(Boolean).join(' · ') : null;
  }
});
let activityView = null;
function closeActivity({ endSession = false } = {}) {
  const dispose = activityView; activityView = null; dispose?.();
  if (endSession) activity.clear();
  else activity.leave();
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
  closeActivity();
  return trackViewDrain([voices.leave(), evaluator.leave(), closeAccess(), closeRoutines(), closeTasks(), closeMemory(), closeMyMemory(), closeVoiceAdmin()]);
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
  updateHouse();
  document.body.classList.remove('map-expanded');
  byId('expand-map').setAttribute('aria-expanded', 'false');
}
// endSession=false only when the page is hidden or the server could not be asked: pending
// routine identities then survive in this tab's sessionStorage until the owning session ends.
function retire(message = '', { endSession = true } = {}) {
  epoch++; navigationVersion++; desiredView = null; privateSuspended = false; session.clear(); transport.invalidateSession();
  clearTimeout(sessionTimer); poller?.stop(); poller = null; ticker?.stop(); ticker = null;
  closeNotices({ endSession: true });
  people.clear(); const accessClosing = closeAccess(), routinesClosing = closeRoutines(), memoryClosing = closeMemory(), myMemoryClosing = closeMyMemory(), voiceAdminClosing = closeVoiceAdmin();
  trackViewDrain([voices.clear(), evaluator.clear(), access.clear(), accessClosing, routinesClosing, routines.clear({ wipe: endSession }),
    closeTasks(), tasks.clear({ wipe: endSession }), memoryClosing, memory.clear({ wipe: endSession }),
    myMemoryClosing, myMemory.clear({ wipe: endSession }), myVoice.clear({ wipe: endSession }), voiceAdminClosing, voiceAdmin.clear({ wipe: endSession })]);
  closeActivity({ endSession: true });
  actions.clear(); destroyHouse(); catalog = clock = lastError = staleFocus = null; roomId = undefined; activeView = null;
  displayNames.clear();
  byId('voice-connection').textContent = ''; byId('evaluation-connection').textContent = '';
  byId('view-transition-status').hidden = true;
  for (const id of ['room-list', 'room-controls', 'global-scenes', 'action-list']) byId(id).replaceChildren();
  byId('room-controls').hidden = true;
  lastSectionViews.clear(); renderNavigation();
  byId('global-scenes').hidden = true;
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
const entityName = entity => displayNames.get(entity.entity_id);
function observedValue(entity) {
  if (entity.domain === 'climate') return Number.isFinite(entity.temperature) ? `${entity.temperature} °C · objetivo` : entity.state === 'off' ? 'Apagado' : 'Temperatura objetivo desconocida';
  if (entity.domain === 'light') return Number.isFinite(entity.brightness_pct) ? `${entity.brightness_pct} % · brillo` : ({ off: 'Apagada', on: 'Brillo desconocido' })[entity.state] || 'Estado desconocido';
  if (entity.domain === 'cover') return Number.isFinite(entity.position) ? `${entity.position} % · abierto` : 'Posición desconocida';
  return null;
}
function renderObservedValue(output, entity) {
  if (entity.domain === 'climate' && Number.isFinite(entity.temperature)) {
    output.replaceChildren(node('strong', '', `${entity.temperature} °C`), node('span', '', ' · objetivo'));
  } else output.textContent = observedValue(entity);
}
function renderCurrentTemperature(output, entity) {
  const room = catalog?.rooms.find(item => item.room_id === entity?.room_id);
  const temperature = entity?.current_temperature ?? room?.current_temperature;
  output.hidden = !Number.isFinite(temperature);
  output.textContent = output.hidden ? '' : `${temperature} °C · actual`;
}
function updateObservedCards() {
  for (const card of document.querySelectorAll('.device-card')) {
    const entity = entityById(entityIds.get(card));
    card.querySelector('.device-state').textContent = entity ? state.entityState(entity) : 'Ya no está disponible.';
    if (entity) card.querySelector('.device-name').textContent = entityName(entity);
    const value = card.querySelector('.device-value');
    if (value) { if (entity) renderObservedValue(value, entity); else value.textContent = 'Estado desconocido'; }
    const current = card.querySelector('.device-current');
    if (current) renderCurrentTemperature(current, entity);
    const power = card.querySelector('.device-power');
    if (power && entity) updateDevicePower(power, entity);
  }
  updateRoomHeading();
}
function canExecute(entity, operation) {
  return session.can('home:control') && !!entity && core.entityCanControl(entity, operation, freshness().stale, clock?.now());
}
function updateTemporal() {
  if (!session.active()) { retire('Tu sesión venció. Iniciá sesión nuevamente.'); return; }
  const observed = freshness();
  byId('connection-status').textContent = lastError ? 'No se pudo actualizar el estado de la casa.' : !catalog ? 'Consultando los dispositivos…' : observed.stale ? 'Datos desactualizados. Actualizá para usar los controles.' : `Actualizado hace ${Math.max(0, Math.floor(observed.ageMs / 1000))} s`;
  byId('connection-status').classList.toggle('is-stale', observed.stale);
  const live = byId('live-badge');
  live.hidden = observed.stale || live.dataset.linked !== 'true';
  const now = clock?.now();
  for (const card of document.querySelectorAll('.device-card')) {
    const entity = entityById(entityIds.get(card));
    card.dataset.observation = !observed.stale && entity && core.entityObservationCurrent(entity, now) ? 'current' : 'stale';
  }
  for (const control of document.querySelectorAll('[data-control]')) {
    const id = entityIds.get(control);
    const entity = entityById(id);
    const record = actions.get(id, control.dataset.operation);
    const otherPower = control.matches('.device-power') && actions.get(id,
      control.dataset.operation === 'turn_on' ? 'turn_off' : 'turn_on')?.pending;
    const disabled = !canExecute(entity, control.dataset.operation) || record?.pending === true || otherPower === true;
    // Disabling (stale data or a pending action) drops focus; remember it so re-enabling returns it.
    if (disabled && !control.disabled && control === document.activeElement) staleFocus = control;
    control.disabled = disabled;
  }
  restoreStaleFocus();
}
function restoreStaleFocus() {
  if (!staleFocus) return;
  if (!staleFocus.isConnected) { staleFocus = null; return; }
  const active = document.activeElement;
  if (staleFocus.disabled || active && active !== document.body && active !== staleFocus) return;
  const control = staleFocus; staleFocus = null;
  control.focus({ preventScroll: true });
}
function updateReceipts() {
  if (!session.active()) return;
  for (const output of document.querySelectorAll('[data-receipt]')) {
    const record = actions.get(entityIds.get(output), output.dataset.operation);
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
function bindControl(control, entity, operation) {
  control.dataset.control = ''; entityIds.set(control, entity.entity_id); control.dataset.operation = operation;
}
function actionFeedback(target, entity, operation) {
  const error = node('p', 'control-error'); error.setAttribute('role', 'alert');
  target.append(error);
  appendReceipt(target, entity, operation);
  return error;
}
function appendReceipt(target, entity, operation) {
  const receipt = node('p', 'action-receipt'); receipt.dataset.receipt = '';
  entityIds.set(receipt, entity.entity_id); receipt.dataset.operation = operation; receipt.setAttribute('role', 'status');
  target.append(receipt);
}
function submitAction(entity, operation, values, error) {
  void execute(entity.entity_id, operation, values, error).catch(() => {
    if (session.active()) error.textContent = 'No se pudo consultar el resultado. Actualizá para verificarlo.';
  });
}
function touchChoices(entity, operation, fields, inputs, error, inRoom) {
  const choices = node('div', 'touch-choices');
  const field = fields[0];
  if (operation === 'set_temperature') {
    for (const [direction, label, text] of [[-1, 'Bajar', '−'], [1, 'Subir', '+']]) {
      const step = button(text, () => {
        const current = entityById(entity.entity_id);
        if (!current) return;
        const { min, max, step } = core.controlFields(current, operation)[0];
        const target = current.temperature ?? min;
        const top = Math.floor((max - min) / step + 1e-7);
        const index = Math.max(0, Math.min(top, Math.round((target - min) / step) + direction));
        const temperature = Number((min + index * step).toFixed(8));
        if (temperature !== current.temperature) submitAction(entity, operation, { temperature }, error);
      }, inRoom ? 'btn thermostat-step' : 'thermostat-step');
      step.setAttribute('aria-label', `${label} temperatura objetivo de ${entityName(entity)}`);
      bindControl(step, entity, operation); choices.append(step);
    }
  } else {
    const presets = operation === 'set_hvac_mode' ? field.options : [
      [2700, 'Cálida 2700 K'], [4000, 'Natural 4000 K'], [5000, 'Fría 5000 K']
    ].filter(([value]) => core.validControlValue(field, value));
    for (const [value, label] of presets) {
      const preset = button(label, () => {
        inputs.get(field.name).value = value;
        submitAction(entity, operation, { [field.name]: value }, error);
      }, inRoom ? 'btn touch-preset' : 'touch-preset');
      bindControl(preset, entity, operation); choices.append(preset);
    }
  }
  return choices;
}
function updateDevicePower(control, entity) {
  const on = entity.state === 'on';
  bindControl(control, entity, on ? 'turn_off' : 'turn_on');
  control.setAttribute('aria-checked', String(on));
  control.setAttribute('aria-label', entityName(entity));
}
function deviceCard(entity, expanded, inRoom = false) {
  const card = node('article', `device-card ${inRoom ? 'row' : 'panel'}`); entityIds.set(card, entity.entity_id); card.dataset.domain = entity.domain;
  const heading = node('div', 'device-heading');
  if (!inRoom) heading.append(icon(entity.domain === 'climate' ? 'climate' : entity.domain === 'cover' ? 'blind' : 'bulb'));
  const title = node('div'); title.append(node(inRoom ? 'h4' : 'h3', 'device-name', entityName(entity)), node('p', 'device-state', state.entityState(entity)));
  heading.append(title); card.append(heading);
  let operations = (entity.capabilities || []).filter(operation => Object.hasOwn(core.OPERATION_LABELS, operation));
  if (inRoom && operations.includes('turn_on') && operations.includes('turn_off')) {
    const power = node('form', 'operation-form device-switch-form'); entityIds.set(power, entity.entity_id);
    const control = button('', () => submitAction(entity, control.dataset.operation, {}, error), 'sw device-power');
    control.setAttribute('role', 'switch');
    updateDevicePower(control, entity);
    power.append(control);
    const error = actionFeedback(power, entity, 'turn_on');
    appendReceipt(power, entity, 'turn_off');
    card.append(power);
    operations = operations.filter(operation => operation !== 'turn_on' && operation !== 'turn_off');
  }
  if (observedValue(entity) !== null) { const value = node('p', 'device-value'); renderObservedValue(value, entity); card.append(value); }
  if (entity.domain === 'climate') {
    const current = node('p', 'device-current'); renderCurrentTemperature(current, entity); card.append(current);
  }
  if (!entity.capabilities?.some(operation => Object.hasOwn(core.OPERATION_LABELS, operation))) {
    card.append(node('p', 'device-permissions', 'Sin controles disponibles.'));
  }
  const binary = node('div', 'power-actions'); card.append(binary);
  for (const operation of operations) {
    const form = node('form', 'operation-form'); entityIds.set(form, entity.entity_id);
    const fields = core.controlFields(entity, operation), inputs = new Map();
    for (const field of fields) {
      const label = node('label', 'control-field', field.label), input = node(field.options ? 'select' : 'input');
      input.name = field.name; input.setAttribute('aria-label', `${entityName(entity)} · ${field.label}`);
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
        if (operation === 'brightness') {
          input.type = 'range';
          input.value = Number.isFinite(value) ? value : field.min;
          const output = node('output', 'range-output', `${input.value} %`);
          input.addEventListener('input', () => { output.textContent = `${input.value} %`; });
          input.addEventListener('change', () => form.requestSubmit());
          label.append(output);
        }
      }
      bindControl(input, entity, operation);
      label.append(input); form.append(label); inputs.set(field.name, input);
    }
    const submit = node('button', inRoom ? 'btn' : fields.length ? 'apply-button' : 'operation-button', core.OPERATION_LABELS[operation]); submit.type = 'submit';
    bindControl(submit, entity, operation); form.append(submit);
    const error = actionFeedback(form, entity, operation);
    if (['set_temperature', 'set_hvac_mode', 'color_temp_kelvin'].includes(operation)) {
      form.insertBefore(touchChoices(entity, operation, fields, inputs, error, inRoom), submit);
    }
    form.addEventListener('submit', event => {
      event.preventDefault();
      const values = Object.fromEntries(fields.map(field => { const input = inputs.get(field.name); return [field.name, field.options ? input.value : input.value === '' ? null : input.valueAsNumber]; }));
      submitAction(entity, operation, values, error);
    });
    if (!fields.length) { form.classList.add('binary-operation'); binary.append(form); }
    else if (operation === 'color_temp_kelvin') {
      const detail = node('details', 'advanced-control'); entityIds.set(detail, entity.entity_id); detail.open = expanded.has(entity.entity_id);
      detail.append(node('summary', '', 'Temperatura de luz'), form); card.append(detail);
    } else card.append(form);
  }
  return card;
}
function selectedEntities() {
  const values = roomId === null ? catalog?.unassigned_entities || [] : state.roomEntities(catalog, roomId, 'all');
  return values.filter(entity => entity.domain !== 'scene');
}
function quickToggle(room) {
  const climate = layer === 'climate', domain = climate ? 'climate' : 'light';
  const entities = state.roomEntities(catalog, room.room_id, 'all').filter(entity => entity.domain === domain);
  function toggle(entity) {
    if (climate) {
      const modes = entity.control_limits?.hvac_modes ?? [];
      const onMode = modes.find(mode => mode !== 'off');
      if (!entity.capabilities?.includes('set_hvac_mode') || !modes.includes('off') || !onMode) return null;
      const on = entity.state !== 'off' && !['unknown', 'unavailable'].includes(entity.state);
      return { entity, operation: 'set_hvac_mode', on, values: { hvac_mode: on ? 'off' : onMode } };
    }
    if (!entity.capabilities?.some(operation => operation === 'turn_on' || operation === 'turn_off')) return null;
    const on = entity.state === 'on', operation = on ? 'turn_off' : 'turn_on';
    return { entity, operation, on, values: {} };
  }
  const candidates = entities.map(toggle).filter(Boolean);
  const preferred = room[climate ? 'default_climate' : 'default_light'];
  const choice = candidates.find(entry => entry.entity.entity_id === preferred) ?? (candidates.length === 1 ? candidates[0] : null);
  return choice?.entity.capabilities.includes(choice.operation) ? choice : null;
}
function renderGlobalScenes() {
  const section = byId('global-scenes');
  const scenes = session.can('home:control') ? core.catalogScenes(catalog) : [];
  const focused = entityIds.get(document.activeElement.closest('#global-scenes [data-control]'));
  section.hidden = !scenes.length; section.replaceChildren();
  if (!scenes.length) return;
  const names = core.catalogSceneNames(scenes, catalog.rooms, displayNames);
  section.append(node('h2', '', 'Escenas del hogar'));
  const strip = node('div', 'scene-strip'); section.append(strip);
  for (const entity of scenes) {
    const item = node('div', 'scene-tile'), operation = 'activate_scene';
    const control = button(names.get(entity.entity_id), () => submitAction(entity, operation, {}, error), 'btn');
    bindControl(control, entity, operation); item.append(control);
    const error = actionFeedback(item, entity, operation);
    strip.append(item);
  }
  return [...strip.querySelectorAll('button')].find(control => entityIds.get(control) === focused);
}
const deviceGroups = { light: 'Luces', climate: 'Clima', cover: 'Persianas', switch: 'Interruptores' };
function updateRoomHeading() {
  const heading = byId('room-controls').querySelector('.selected-room-heading');
  if (!heading) return;
  const selected = catalog.rooms.find(room => room.room_id === roomId);
  heading.textContent = selected?.name || (catalog.unassigned_entities.some(entity => entity.domain !== 'scene') ? 'Sin ambiente' : 'Ambientes');
  const temperature = heading.nextElementSibling;
  const thermal = selected && layer === 'climate' ? thermalReading(selected, selectedEntities()) : null;
  const reading = thermal?.actual ?? (Number.isFinite(selected?.current_temperature) ? `${selected.current_temperature} °C` : null);
  temperature.textContent = [reading && `Temperatura actual · ${reading}`, thermal?.status].filter(Boolean).join(' · ');
  temperature.hidden = !temperature.textContent;
}
function renderControls() {
  if (!catalog || !session.can('home:read')) return;
  const expanded = new Set([...document.querySelectorAll('.advanced-control[open]')].map(detail => entityIds.get(detail)));
  const controls = byId('room-controls');
  const headingFocused = document.activeElement?.id === 'selected-room-heading';
  controls.replaceChildren();
  controls.hidden = false;
  const heading = node('h2', 'selected-room-heading'); heading.id = 'selected-room-heading'; heading.tabIndex = -1;
  const header = node('header', 'room-heading'); header.append(heading, node('p', 'room-temperature')); controls.append(header);
  updateRoomHeading();
  const entities = selectedEntities();
  const groups = new Map([...Object.keys(deviceGroups), 'other'].map(domain => [domain, []]));
  for (const entity of entities) (groups.get(entity.domain) ?? groups.get('other')).push(entity);
  for (const [domain, devices] of groups) {
    if (!devices.length) continue;
    const group = node('section', 'device-group');
    const rows = node('div', 'rows'); rows.append(...devices.map(entity => deviceCard(entity, expanded, true)));
    group.append(node('h3', '', deviceGroups[domain] ?? 'Otros dispositivos'), rows);
    controls.append(group);
  }
  if (!entities.length) controls.append(node('p', 'room-empty', 'No hay dispositivos para mostrar.'));
  const all = byId('action-list'); all.replaceChildren();
  for (const entity of catalog.entities) all.append(deviceCard(entity, expanded));
  if (!catalog.entities.length) all.append(node('p', 'panel empty-panel', 'Todavía no hay dispositivos para mostrar.'));
  updateReceipts();
  if (headingFocused) heading.focus({ preventScroll: true });
}
function selectRoom(id, { fromMap = false } = {}) {
  if (id !== null && !catalog?.rooms.some(room => room.room_id === id)) return;
  if (roomId !== id) { roomId = id; renderCatalog(true); }
  if (fromMap) return;
  if (document.body.classList.contains('map-expanded')) byId('expand-map').click();
  const heading = byId('selected-room-heading');
  const rect = heading?.getBoundingClientRect();
  if (rect && (rect.top < 0 || rect.bottom + 120 > innerHeight)) {
    heading.scrollIntoView({ block: 'start' });
    heading.focus({ preventScroll: true });
  }
}
function updateHouse() {
  const linked = house?.update({ catalog, layer, selected: roomId });
  const live = byId('live-badge');
  live.dataset.linked = String(!!linked);
  live.hidden = !linked || freshness().stale;
}
function renderCatalog(forceControls = false) {
  if (!catalog || !session.can('home:read')) return;
  const rooms = catalog.rooms;
  const previousRoom = roomId;
  if (roomId === undefined || roomId !== null && !rooms.some(room => room.room_id === roomId)) roomId = rooms[0]?.room_id ?? null;
  byId('rooms-total').textContent = String(rooms.length); byId('room-count').textContent = String(rooms.length);
  byId('entities-total').textContent = String(catalog.entities.filter(entity => entity.domain !== 'scene').length);
  byId('lights-total').textContent = String(catalog.entities.filter(entity => entity.domain === 'light' && entity.available && entity.state === 'on').length);
  const list = byId('room-list');
  const focus = document.activeElement.closest('#room-list button');
  let replacement;
  list.replaceChildren();
  function addRoom(id, name, count, quick) {
    const row = node('div', 'room-row');
    const control = button('', () => selectRoom(id), 'btn room-choice'); control.classList.toggle('active', id === roomId); control.setAttribute('aria-pressed', String(id === roomId));
    control.setAttribute('aria-controls', 'room-controls');
    roomIds.set(control, id);
    control.append(icon('house')); const title = node('span'); title.append(node('strong', '', name), node('small', '', `${count} dispositivo${count === 1 ? '' : 's'}`)); control.append(title); row.append(control);
    if (focus?.classList.contains('room-choice') && roomIds.get(focus) === id) replacement = control;
    if (quick) {
      const { entity, operation, on, values } = quick;
      const power = button('', () => submitAction(entity, operation, values, error), 'sw room-power');
      bindControl(power, entity, operation);
      power.setAttribute('role', 'switch'); power.setAttribute('aria-checked', String(on));
      power.setAttribute('aria-label', `${name} · ${entityName(entity)}`);
      row.append(power);
      const error = actionFeedback(row, entity, operation);
      if (entity.domain === 'light') appendReceipt(row, entity, on ? 'turn_on' : 'turn_off');
      if (focus?.classList.contains('room-power') && entityIds.get(focus) === entity.entity_id) replacement = power;
    }
    list.append(row);
  }
  const deviceIds = new Set(catalog.entities.filter(entity => entity.domain !== 'scene').map(entity => entity.entity_id));
  for (const room of rooms) addRoom(room.room_id, room.name,
    room.entity_ids.filter(id => deviceIds.has(id)).length, session.can('home:control') && quickToggle(room));
  const unassigned = catalog.unassigned_entities.filter(entity => entity.domain !== 'scene').length;
  if (unassigned) addRoom(null, 'Sin ambiente', unassigned);
  if (!rooms.length && !unassigned) list.append(node('p', 'room-empty', 'No hay ambientes para mostrar.'));
  // Refreshing an observation must not discard a keyboard user's draft or focus.
  // A control disabled by stale data keeps its draft until fresh data re-enables it.
  const focused = document.activeElement?.closest('.operation-form') || (staleFocus?.isConnected ? staleFocus.closest('.operation-form') : null);
  const focusedEntity = entityIds.get(focused);
  const stillSelected = !focused?.closest('#room-controls') || selectedEntities().some(entity => entity.entity_id === focusedEntity);
  if (forceControls || !focused || previousRoom !== roomId || !entityById(focusedEntity) || !stillSelected) renderControls();
  else updateObservedCards();
  const sceneFocus = renderGlobalScenes();
  updateHouse();
  updateReceipts();
  (replacement ?? sceneFocus)?.focus({ preventScroll: true });
}
function renderNavigation() {
  const primary = byId('rail-nav'), secondary = byId('section-views');
  const sections = session.sections();
  const selection = sections.find(section => section.views.some(view => view.id === desiredView));
  if (selection) lastSectionViews.set(selection.id, desiredView);

  // Keep the primary buttons in place so moving between sections preserves keyboard focus.
  const sectionIds = sections.map(section => section.id).join(' ');
  if (primary.dataset.sections !== sectionIds) {
    primary.replaceChildren(...sections.map(section => {
      const control = button(section.label, () => activate(control.dataset.view), 'section-link');
      control.dataset.section = section.id;
      return control;
    }));
    primary.dataset.sections = sectionIds;
  }
  sections.forEach((section, index) => {
    const control = primary.children[index];
    control.dataset.view = sectionDestination(section, lastSectionViews.get(section.id));
    if (section === selection) control.setAttribute('aria-current', 'true');
    else control.removeAttribute('aria-current');
  });

  const views = selection?.views.length > 1 ? selection.views : [];
  const secondaryViewIds = views.map(view => view.id).join(' ');
  if (secondary.dataset.views !== secondaryViewIds) {
    const hadFocus = secondary.contains(document.activeElement);
    secondary.replaceChildren(...views.map(view => {
      const control = button(view.label, () => activate(view.id), 'section-link');
      control.dataset.view = view.id;
      return control;
    }));
    secondary.dataset.views = secondaryViewIds;
    if (hadFocus) primary.querySelector('[aria-current]')?.focus({ preventScroll: true });
  }
  secondary.hidden = !views.length;
  if (views.length) secondary.setAttribute('aria-label', `Vistas de ${selection.label}`);
  else secondary.removeAttribute('aria-label');
  for (const control of secondary.children) {
    if (control.dataset.view === desiredView) control.setAttribute('aria-current', 'page');
    else control.removeAttribute('aria-current');
  }
}
async function activate(view, push = true, force = false) {
  if (!session.views().includes(view) || !Object.hasOwn(viewIds, view)) return false;
  if (!force && activeView === view && desiredView === view) return true;
  const version = ++navigationVersion, generation = session.generation();
  desiredView = view;
  if (push && activeView !== view) history.pushState({ view }, '', state.VIEW_HASHES[view]);
  renderNavigation();
  activeView = null;
  for (const id of Object.values(viewIds)) byId(id).hidden = true;
  byId('workspace-toolbar').hidden = true;
  const status = byId('view-transition-status'); status.hidden = false; status.textContent = 'Preparando vista…';
  poller?.setHidden(true); ticker?.stop();
  closeNotices();
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
    if (view === 'voices') {
      voiceAdminView = window.NexoVoiceConsents.mountVoiceConsents({ container: byId('voice-attribution'), confirmDialog: byId('voice-attribution-confirm-dialog'), owner: voiceAdmin, icon });
      opened = await voices.open();
      void voiceAdmin.open();
    }
    if (view === 'access') {
      accessView = window.NexoAccess.mountAccess({ container: byId('access-view'), dialog: byId('access-dialog'),
        secretDialog: byId('access-secret-dialog'), controller: access,
        getSession: () => session.active() ? { role: session.role(), expiresAt: session.expiresAt() } : null });
      opened = await access.open();
    }
    if (view === 'evaluation') opened = await evaluator.open();
    if (view === 'routines') {
      routinesView = window.NexoRoutines.mountRoutines({ container: byId('routines-view'), dialog: byId('routine-dialog'),
        confirmDialog: byId('routine-confirm-dialog'), owner: routines, icon });
      opened = await routines.open();
    }
    if (view === 'tasks') {
      tasksView = window.NexoTasks.mountTasks({ container: byId('tasks-view'), confirmDialog: byId('tasks-confirm-dialog'), owner: tasks, deviceName: id => catalog && session.can('home:read') ? displayNames.get(id) ?? null : null });
      opened = await tasks.open();
    }
    if (view === 'activity' && activity.permissions().visible) {
      activityView = window.NexoActivity.mountActivity({ container: byId('activity-view'), owner: activity });
      opened = await activity.open();
    }
    if (view === 'memory') {
      memoryView = window.NexoMemory.mountMemory({ container: byId('memory-view'), confirmDialog: byId('memory-confirm-dialog'), owner: memory, icon });
      opened = await memory.open();
    }
    if (view === 'mymemory') {
      const voiceSlot = node('div', 'memory-voice-slot');
      myVoiceView = window.NexoMyVoice.mountMyVoice({ container: voiceSlot, confirmDialog: byId('my-voice-confirm-dialog'), owner: myVoice, icon });
      myMemoryView = window.NexoMyMemory.mountMyMemory({ container: byId('my-memory-view'), confirmDialog: byId('my-memory-confirm-dialog'), owner: myMemory, icon,
        returnNotice: window.NexoMyMemoryOwner.RETURN_NOTICE, voiceReturnNotice: window.NexoMyMemoryOwner.VOICE_RETURN_NOTICE, voiceSlot,
        voiceEnabled: () => myVoice.getState().status?.enabled === true, onVoiceChange: listener => myVoice.subscribe(listener) });
      opened = await myMemory.open();
      void myVoice.open();
    }
    if (!stillCurrent()) return false;
    if (!opened) throw Error('No pudimos abrir esta vista. Podés reintentar la consulta.');
    privateSuspended = false;
    poller?.setHidden(document.hidden || auxiliaryView(view));
    if (document.hidden || auxiliaryView(view)) ticker?.stop(); else ticker?.start();
    if (view === 'home') openNotices();
    if (view === 'home' && !house) {
      byId('map-loading').hidden = false;
      house = createHouse({ container: byId('house-canvas'), pins: byId('room-pins'), onSelect: id => selectRoom(id, { fromMap: true }), onReady(ready) {
        byId('map-loading').hidden = true;
        for (const id of ['view-3d', 'view-top', 'zoom-in', 'zoom-out', 'rotate-left', 'rotate-right', 'reset-view']) byId(id).disabled = !ready;
        // The model loads asynchronously: show the observed state as soon as it is on screen.
        updateHouse();
      } });
      cameraView('3d', false);
      updateHouse();
    }
    if (!auxiliaryView(view)) renderControls();
    return true;
  } catch (error) {
    if (stillCurrent()) { status.hidden = false; status.replaceChildren(node('span', '', error.message), button('Reintentar', () => activate(view, false, true))); }
    return false;
  }
}
function requestedView() {
  const available = session.views().filter(view => Object.hasOwn(viewIds, view));
  return state.resolveViewHash(location.hash, available);
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
    armSessionTimer();
    const view = requestedView(), home = session.can('home:read'); byId('restricted-view').hidden = Boolean(view);
    for (const id of Object.values(viewIds)) byId(id).hidden = true;
    byId('refresh').hidden = !home;
    if (view) await activate(view, false, true);
    else renderNavigation();
    if (started !== epoch || !session.active()) return;
    if (!home) { byId('connection-status').textContent = 'Sesión vigente.'; return; }
    const generation = session.generation();
    poller = core.createPoller({
      load: async signal => { await actions.reconcile(); return api('/api/home/catalog', { signal }); },
      // Refresh against the earliest held expiry, not only a fixed cadence.
      deadline: () => clock?.read().remainingMs ?? Infinity,
      onData(data) {
        if (session.generation() !== generation || !session.can('home:read')) return;
        if (![data?.rooms, data?.entities, data?.unassigned_entities].every(Array.isArray)) { lastError = new Error('La lista de dispositivos no está disponible.'); updateTemporal(); return; }
        catalog = data; displayNames = core.catalogNames(data); clock = core.createSnapshotClock(data); lastError = null; renderCatalog();
      },
      onError(error) { if (session.generation() === generation && session.active()) { lastError = error; updateTemporal(); } }
    });
    ticker = core.createFreshnessTicker({ read: () => freshness(), onTick: updateTemporal });
    poller.setHidden(document.hidden || auxiliaryView(activeView)); if (!document.hidden && !auxiliaryView(activeView)) ticker.start(); poller.start();
  } catch (error) {
    if (started === epoch) retire(error.status === 401 ? 'Ingresá para consultar tu hogar.' : error.message, { endSession: error.status === 401 });
  }
}

function loginErrorMessage(error) {
  if (error?.localMessage && typeof error.message === 'string') return error.message;
  if (error?.status === 401 || error?.kind === 'unauthorized') return 'La credencial no es válida o venció.';
  if (error?.status === 403 || error?.kind === 'forbidden') {
    return error?.detail === 'origin denied'
      ? 'Este origen no está autorizado para iniciar sesión.'
      : 'El acceso fue rechazado por origen o permisos.';
  }
  if (error?.status === 503 || error?.kind === 'unavailable') return 'El servicio no está disponible temporalmente. Probá de nuevo en unos instantes.';
  if (error?.kind === 'timeout' || error?.name === 'TimeoutError') return 'El servidor no respondió a tiempo. Probá de nuevo en unos instantes.';
  if (error?.kind === 'network' || error instanceof TypeError) return 'No se pudo conectar con el servidor. Revisá la red y probá de nuevo.';
  if (error?.kind === 'aborted' || error?.name === 'AbortError') return 'Se canceló el inicio de sesión.';
  if (Number.isInteger(error?.status)) return `No se pudo iniciar sesión (${error.status}).`;
  return 'No se pudo iniciar sesión. Probá de nuevo en unos instantes.';
}

byId('login-form').addEventListener('submit', async event => {
  event.preventDefault(); if (authBusy) { byId('token').value = ''; return; } lockAuthentication(true);
  let token = byId('token').value; retire(); pendingLogout(false); const started = epoch;
  try {
    if (!await viewDrain || started !== epoch) throw Object.assign(new Error('La vista anterior todavía no terminó de cerrar.'), { localMessage: true });
    await api('/api/admin/auth/login', { method: 'POST', body: { token } }); if (started === epoch) await bootstrap();
  }
  catch (error) {
    if (started === epoch || !session.active() && (error?.status === 401 || state.sessionRetired(error))) {
      byId('login-error').textContent = loginErrorMessage(error);
    }
  }
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
byId('refresh').addEventListener('click', () => {
  if (session.can('home:read')) { actions.reconcile(); poller?.refresh(); void notices.refresh(); }
});
for (const control of byId('layer-tabs').querySelectorAll('[data-layer]')) control.addEventListener('click', () => {
  layer = control.dataset.layer;
  for (const item of byId('layer-tabs').querySelectorAll('[data-layer]')) { const active = item === control; item.classList.toggle('active', active); item.setAttribute('aria-pressed', String(active)); }
  renderCatalog();
});
for (const [id, callback] of [
  ['view-3d', () => cameraView('3d')], ['view-top', () => cameraView('top')],
  ['zoom-in', () => house?.zoom(.12)], ['zoom-out', () => house?.zoom(-.12)],
  ['rotate-left', () => house?.rotate(-1)], ['rotate-right', () => house?.rotate(1)], ['reset-view', () => { house?.reset(); cameraView('3d', false); }]
]) byId(id).addEventListener('click', callback);
byId('expand-map').addEventListener('click', () => { const expanded = document.body.classList.toggle('map-expanded'); byId('expand-map').setAttribute('aria-expanded', String(expanded)); });
byId('help-button').addEventListener('click', () => byId('info-dialog').showModal());
for (const id of ['dialog-close', 'dialog-done']) byId(id).addEventListener('click', () => byId('info-dialog').close());
// Any focus or pointer choice by the user supersedes a focus held across stale data.
document.addEventListener('pointerdown', () => { staleFocus = null; }, true);
document.addEventListener('focusin', event => { if (event.target !== staleFocus) staleFocus = null; });
document.addEventListener('focusin', ({target}) => {
  if (target.matches(':is(#people-grid, #voice-profile-list) > .pick:focus-visible')) {
    target.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }
});
// APG modal dialog: Tab and Shift+Tab wrap inside the topmost open modal instead of reaching the page behind or the browser chrome.
document.addEventListener('keydown', event => {
  if (event.key !== 'Tab' || event.altKey || event.ctrlKey || event.metaKey) return;
  const dialog = [...document.querySelectorAll('dialog[open]')].filter(item => item.matches(':modal')).at(-1);
  if (!dialog) return;
  const items = [...dialog.querySelectorAll('a[href],button,input:not([type=hidden]),select,textarea,summary,audio[controls],[tabindex]')]
    .filter(item => !item.disabled && item.tabIndex >= 0 && item.getClientRects().length && !item.closest('[inert]'));
  if (!items.length) { event.preventDefault(); dialog.focus(); return; }
  const active = document.activeElement, first = items[0], last = items.at(-1);
  if (!dialog.contains(active)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); }
  else if (event.shiftKey && active === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && active === last) { event.preventDefault(); first.focus(); }
});
document.addEventListener('keydown', event => { if (event.key === 'Escape') { document.body.classList.remove('map-expanded'); byId('expand-map').setAttribute('aria-expanded', 'false'); } });
document.addEventListener('visibilitychange', () => {
  if (!session.active()) { if (!byId('nexo-workspace').hidden) retire('Tu sesión venció.'); return; }
  armSessionTimer();
  if (document.hidden || auxiliaryView(activeView)) ticker?.stop(); else { updateTemporal(); ticker?.start(); }
  poller?.setHidden(document.hidden || auxiliaryView(activeView));
  if (document.hidden) closeNotices(); else openNotices();
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
