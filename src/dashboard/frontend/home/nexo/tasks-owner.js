/* One owner per tab. Scope comes from the dashboard session and the selected section. */
(function (root) {
  'use strict';
  const WHOAMI = '/api/admin/auth/whoami';
  const LISTS = '/api/lists', REMINDERS = '/api/reminders';
  const GRANTS = ['tasks:read', 'tasks:write', 'tasks:personal'];
  const itemPath = id => LISTS + '/' + encodeURIComponent(id) + '/items';
  const entryPath = (path, id) => path + '/' + encodeURIComponent(id);

  function failureMessage(error, writing = false) {
    switch (error?.status) {
      case 401: return 'Tu sesión venció. Iniciá sesión para consultar tus tareas.';
      case 403: return writing ? 'Esta sesión no tiene permiso para cambiar estas tareas.' : 'Esta sesión no tiene permiso para consultar estas tareas.';
      case 404: return 'La tarea ya no está disponible.';
      case 409: return 'La tarea cambió mientras la consultabas. Revisá el listado actualizado.';
      case 422: return 'El servicio rechazó los datos. Revisalos antes de volver a enviarlos.';
      case 503: return 'El servicio de tareas no está disponible. Volvé a consultar en unos minutos.';
      default: return writing
        ? 'No pudimos confirmar el cambio. Revisá el listado antes de volver a intentarlo.'
        : 'No pudimos consultar las tareas. Revisá la conexión y volvé a actualizar.';
    }
  }

  function rows(value) {
    if (!Array.isArray(value)) throw Error('invalid_tasks_response');
    return value;
  }

  function createTasksOwner({ api, session, onAccountChanged = () => {} }) {
    const listeners = new Set();
    let generation = -1, visit = 0, opened = false;
    let read = null, write = null, data = null, error = '', feedback = '';
    const emit = () => { for (const listener of listeners) listener(); };
    const current = () => opened && generation === session.generation() && session.can('tasks:read');

    function getState() {
      if (!current()) return { visible: false };
      return { visible: true, data, error, feedback, busy: Boolean(read || write),
        canWrite: session.can('tasks:write'), writing: Boolean(write) };
    }

    function leave() {
      opened = false; visit++;
      read?.abort(); read = null;
      write?.preflight?.abort();
      data = null; error = feedback = '';
      emit();
      return Promise.resolve(true);
    }

    function clear({ wipe = true } = {}) {
      generation = -1;
      // There are no persisted drafts or receipts. A sent write may finish after leaving.
      const drained = leave();
      if (wipe) write = null;
      return drained;
    }

    async function verify(signal, stillCurrent) {
      const identity = await api(WHOAMI, { signal });
      if (!stillCurrent()) return false;
      if (identity?.ok !== true || identity.role !== session.role() || identity.subject !== session.subject() ||
          (identity.account_binding ?? null) !== session.accountBinding() ||
          (identity.routine_actor ?? null) !== session.routineActor() ||
          !Array.isArray(identity.permissions) || GRANTS.some(grant => identity.permissions.includes(grant) !== session.can(grant))) {
        const previous = generation;
        void clear(); onAccountChanged(previous);
        return false;
      }
      return true;
    }

    async function loadScope(scope, signal) {
      const [lists, reminders] = await Promise.all([
        api(LISTS + '?scope=' + scope, { signal }),
        api(REMINDERS + '?scope=' + scope, { signal })
      ]);
      const scopedLists = rows(lists).filter(list => list?.owner_type === (scope === 'shared' ? 'shared' : 'user'));
      return {
        lists: await Promise.all(scopedLists.map(async ({ id, name }) => ({
          id, name, items: rows(await api(itemPath(id), { signal }))
        }))),
        reminders: rows(reminders).filter(reminder => reminder?.scope === (scope === 'shared' ? 'shared' : 'private'))
      };
    }

    async function refresh({ keepFeedback = false } = {}) {
      if (!current() || read || write) return false;
      const controller = new AbortController(), { signal } = controller;
      const stillCurrent = () => read === controller && current();
      read = controller; error = '';
      if (!keepFeedback) feedback = '';
      emit();
      try {
        if (!await verify(signal, stillCurrent) || !stillCurrent()) return false;
        const [shared, personal] = await Promise.all([
          loadScope('shared', signal),
          session.can('tasks:personal') ? loadScope('personal', signal) : null
        ]);
        if (!stillCurrent() || !await verify(signal, stillCurrent) || !stillCurrent()) return false;
        data = { shared, personal };
        return true;
      } catch (failure) {
        if (stillCurrent()) { data = null; error = failureMessage(failure); }
        return false;
      } finally {
        controller.abort();
        if (stillCurrent()) { read = null; emit(); }
      }
    }

    function open() {
      if (!session.can('tasks:read') || !session.subject()) return false;
      if (generation !== session.generation()) {
        void clear(); generation = session.generation();
      }
      opened = true;
      void refresh();
      return true;
    }

    async function mutate(scope, path, options, success) {
      if (!['shared', 'personal'].includes(scope) || !current() || read || write || !data?.[scope] || !session.can('tasks:write') ||
          (scope === 'personal' && !session.can('tasks:personal'))) return false;
      const ticket = visit, started = generation;
      const operation = { preflight: new AbortController() };
      const stillCurrent = () => current() && visit === ticket && write === operation;
      write = operation; feedback = ''; emit();
      let committed = false;
      try {
        if (!await verify(operation.preflight.signal, stillCurrent) || !stillCurrent()) return false;
        // Once dispatched, let the write finish even if its view is closed.
        operation.preflight = null;
        const result = await api(path, options);
        committed = result?.outcome?.status === 'completed' && result.outcome.persistence === 'committed';
        if (stillCurrent()) feedback = committed ? success : failureMessage(null, true);
      } catch (failure) {
        if (stillCurrent()) feedback = failureMessage(failure, true);
      } finally {
        if (write === operation) {
          write = null;
          if (current() && generation === started) await refresh({ keepFeedback: true });
        }
      }
      return committed;
    }

    const list = (scope, id) => data?.[scope]?.lists.find(entry => entry.id === id);
    return {
      open, leave, clear, refresh, getState,
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      createList: (scope, name) => mutate(scope, LISTS, { method: 'POST', body: { name, shared: scope === 'shared' } }, 'Lista creada.'),
      addItem: (scope, id, text) => list(scope, id) &&
        mutate(scope, itemPath(id), { method: 'POST', body: { text } }, 'Elemento añadido.'),
      removeItem: (scope, id, itemId) => list(scope, id)?.items.some(item => item.id === itemId) &&
        mutate(scope, entryPath(itemPath(id), itemId), { method: 'DELETE' }, 'Elemento eliminado.'),
      createReminder: (scope, text, trigger_at) => mutate(scope, REMINDERS,
        { method: 'POST', body: { text, trigger_at, scope: scope === 'personal' ? 'private' : 'shared' } }, 'Recordatorio creado.'),
      removeReminder: (scope, id) => data?.[scope]?.reminders.some(reminder => reminder.id === id) &&
        mutate(scope, entryPath(REMINDERS, id), { method: 'DELETE' }, 'Recordatorio eliminado.')
    };
  }

  const exported = { createTasksOwner };
  root.NexoTasksOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
