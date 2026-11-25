/* Activity projects authorized receipts into read-only, display-safe rows. */
(function (root) {
  'use strict';
  const isNode = typeof module !== 'undefined' && module.exports;
  const home = isNode ? require('../home.js') : root.HomeUI;
  const { canReadNotices, canReadRoutineRuns } = isNode ? require('./state.js') : root.NexoState;
  const { parseRoutine } = isNode ? require('./routine-owner.js') : root.NexoRoutineOwner;
  const WHOAMI = '/api/admin/auth/whoami', ACTION_LIMIT = 50;
  const RUN_LABELS = { running: 'En curso', succeeded: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta: Home Assistant no confirmó todas las acciones', skipped: 'Omitida por condición', interrupted: 'Interrumpida' };
  const SOURCE_LABELS = { manual: 'Manual', schedule: 'Programada', scheduled: 'Programada', household_voice: 'Por voz' };
  const RUN_STATES = { running: 'running', succeeded: 'confirmed', failed: 'failed', uncertain: 'uncertain', skipped: 'skipped', interrupted: 'uncertain' };
  const ACTION_STATES = { in_progress: 'running', confirmed: 'confirmed', failed: 'failed', uncertain: 'uncertain' };
  const NOTICE_STATES = { pending: 'pending', sending: 'running', uncertain: 'uncertain' };
  const DEVICE_TYPES = { light: 'Una luz', climate: 'Un aire acondicionado', cover: 'Una persiana', switch: 'Un interruptor', scene: 'Una escena' };
  const label = (labels, key, fallback) => Object.hasOwn(labels, key) ? labels[key] : fallback;
  const block = (status, rows = [], updatedAt = null) => ({ status, rows, updatedAt });
  const result = (rows, updatedAt) => block(rows.length ? 'ready' : 'empty', rows, updatedAt);
  const ordered = rows => rows.sort((a, b) => (b.at ?? -Infinity) - (a.at ?? -Infinity) || a.what.localeCompare(b.what, 'es'));

  function activityPermissions(session) {
    const visible = ['home', 'operator'].includes(session.role()) && session.can('home:read');
    return {
      visible,
      action: visible && session.can('home:control'),
      notice: visible && canReadNotices(session) && !!session.subject(),
      routine: visible && canReadRoutineRuns(session.can)
    };
  }

  function noticeRow(notice) {
    return { block: 'notice', at: notice.trigger_at * 1000, atKind: 'scheduled',
      what: notice.text?.trim() || 'Aviso sin texto', where: null, origin: 'Aviso compartido',
      status: NOTICE_STATES[notice.status], label: notice.label, detail: '' };
  }

  function routineRows(data) {
    if (!Array.isArray(data)) throw Error('invalid_activity_routines');
    const rows = new Map();
    for (const value of data) {
      const { name, latest_run: run } = parseRoutine(value);
      if (!run) continue;
      const scheduled = run.source === 'schedule' || run.source === 'scheduled';
      const at = scheduled ? home.observationMillis(run.scheduled_for) : null;
      const count = run.actions.filter(action => action.status === 'confirmed').length;
      const summary = run.actions.length ? `${count} de ${run.actions.length} acciones confirmadas` : '';
      rows.set(run.run_id, { block: 'routine', at, atKind: at === null ? null : 'scheduled',
        what: name.trim() || 'Rutina sin nombre', where: null, origin: label(SOURCE_LABELS, run.source, 'Origen desconocido'),
        status: label(RUN_STATES, run.status, 'unknown'), label: label(RUN_LABELS, run.status, 'Estado desconocido'),
        detail: [summary, run.status === 'running' ? 'Actualizá para ver el resultado' : ''].filter(Boolean).join(' · ') });
    }
    return ordered([...rows.values()]);
  }

  function createActivityOwner({ api, session, actions, notices, clock = Date.now,
    deviceLabel = () => null, onAccountChanged = () => {} }) {
    const receipts = new Map(), listeners = new Set();
    let generation = session.generation(), opened = false, reading = null, stopNotices = null;
    let actionStatus = 'loading', routine = block('loading');
    const permissions = () => activityPermissions(session);
    const current = () => opened && generation === session.generation() && permissions().visible;
    const emit = () => { for (const listener of listeners) listener(); };

    function actionRow({ entityId, operation, at, status, detail }) {
      const where = deviceLabel(entityId) || label(DEVICE_TYPES, entityId.split('.')[0], 'Un dispositivo');
      return { block: 'action', at, atKind: 'sent_here', what: home.OPERATION_LABELS[operation],
        where, origin: 'Desde esta pestaña', status: label(ACTION_STATES, status, 'unknown'),
        label: home.actionLabel(status), detail };
    }

    function getState() {
      if (!current()) return { visible: false };
      const allowed = permissions(), blocks = {};
      if (allowed.action) blocks.action = actionStatus === 'ready'
        ? result(ordered([...receipts.values()].map(actionRow))) : block(actionStatus);
      if (allowed.notice) {
        const shared = notices.getState();
        blocks.notice = block(shared.status ?? 'loading', ordered((shared.notices ?? []).map(noticeRow)));
      }
      if (allowed.routine) blocks.routine = routine;
      const statuses = Object.values(blocks).map(value => value.status);
      const failed = statuses.filter(status => status === 'error' || status === 'stale').length;
      const status = statuses.includes('loading') ? 'loading'
        : failed === 0 ? 'ready' : failed === statuses.length ? 'error' : 'partial';
      return { visible: true, blocks, status, refreshing: reading !== null };
    }

    function leave() {
      opened = false;
      reading?.abort();
      reading = null;
      if (stopNotices) {
        stopNotices(); stopNotices = null;
        notices.leave();
      }
      actionStatus = 'loading';
      routine = block('loading');
      emit();
    }

    function clear() {
      receipts.clear();
      generation = session.generation();
      leave();
    }

    function recordAction(record) {
      if (generation !== session.generation()) clear();
      if (!permissions().action || !record?.request_id || typeof record.entity_id !== 'string' ||
          !Object.hasOwn(home.OPERATION_LABELS, record.operation)) return;
      const previous = receipts.get(record.request_id);
      if (!previous && !(record.sending && record.status === 'in_progress')) return;
      receipts.set(record.request_id, { entityId: record.entity_id, operation: record.operation,
        at: previous?.at ?? clock(), status: record.status, detail: home.effectSummary(record.effects) });
      if (receipts.size > ACTION_LIMIT) receipts.delete(receipts.keys().next().value);
      if (current()) emit();
    }

    function canRead(controller, blockName) {
      return reading === controller && current() && permissions()[blockName];
    }

    async function verify(controller, blockName) {
      if (!canRead(controller, blockName)) return false;
      const identity = await api(WHOAMI, { signal: controller.signal });
      if (!canRead(controller, blockName)) return false;
      if (identity?.ok !== true || (identity.subject ?? null) !== session.subject() ||
          identity.role !== session.role() || (identity.account_binding ?? null) !== session.accountBinding() ||
          (identity.routine_actor ?? null) !== session.routineActor()) {
        const previous = generation;
        clear(); onAccountChanged(previous);
        return false;
      }
      const can = permission => Array.isArray(identity.permissions) && identity.permissions.includes(permission);
      if (!can('home:read') || (blockName === 'action' ? !can('home:control')
        : !canReadRoutineRuns(can) || can('routines:manage') !== session.can('routines:manage'))) {
        throw Error('activity_read_denied');
      }
      return true;
    }

    async function readActions(controller) {
      try {
        if (!await verify(controller, 'action') || !canRead(controller, 'action')) return;
        await actions.reconcile();
        if (await verify(controller, 'action') && canRead(controller, 'action')) actionStatus = 'ready';
      } catch (_) {
        if (reading === controller && current()) actionStatus = 'error';
      }
      if (reading === controller && current()) emit();
    }

    async function readRoutines(controller) {
      try {
        if (!await verify(controller, 'routine') || !canRead(controller, 'routine')) return;
        const data = await api('/api/routines', { signal: controller.signal });
        if (await verify(controller, 'routine') && canRead(controller, 'routine')) routine = result(routineRows(data), clock());
      } catch (_) {
        if (reading === controller && current()) routine = block('error');
      }
      if (reading === controller && current()) emit();
    }

    async function refresh({ opening = false } = {}) {
      if (!current() || reading) return;
      const allowed = permissions();
      const controller = new AbortController();
      reading = controller;
      if (allowed.action) actionStatus = 'loading';
      if (allowed.routine) routine = block('loading');
      emit();
      try {
        if (!allowed.notice && stopNotices) {
          stopNotices(); stopNotices = null;
          notices.leave();
        }
        await Promise.all([
          allowed.action && readActions(controller),
          allowed.routine && readRoutines(controller),
          allowed.notice && !opening && notices.refresh()
        ]);
      } finally {
        if (reading === controller) { reading = null; emit(); }
      }
    }

    async function open() {
      if (generation !== session.generation()) clear();
      const allowed = permissions();
      if (!allowed.visible) return false;
      opened = true;
      if (allowed.notice && !stopNotices) {
        stopNotices = notices.subscribe(emit);
        notices.open();
      }
      await refresh({ opening: true });
      return current();
    }

    return { permissions, recordAction, getState, open, refresh, leave, clear,
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); } };
  }

  const exported = { activityPermissions, createActivityOwner };
  root.NexoActivityOwner = exported;
  if (isNode) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
