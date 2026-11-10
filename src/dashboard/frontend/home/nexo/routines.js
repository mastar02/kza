/* NEXO Rutinas view: DOM only. Cards are keyed by routine and rebuilt only when
 * their rendered data changes; focus returns to the control with the same key. */
(function (root) {
  'use strict';
  const RUN_LABELS = { running: 'En curso', succeeded: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta: Home Assistant no confirmó todas las acciones', skipped: 'Omitida por condición', interrupted: 'Interrumpida' };
  const ACTION_LABELS = { pending: 'Pendiente', sending: 'Enviando', confirmed: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta', not_executed: 'No ejecutada' };
  const SOURCE_LABELS = { manual: 'Manual', schedule: 'Programada' };
  const DAYS = [['Lu', 'Lunes'], ['Ma', 'Martes'], ['Mi', 'Miércoles'], ['Ju', 'Jueves'], ['Vi', 'Viernes'], ['Sá', 'Sábado'], ['Do', 'Domingo']];

  function mountRoutines({ container, dialog, confirmDialog, owner, icon }) {
    const doc = container.ownerDocument, home = root.HomeUI, routines = root.NexoRoutineOwner;
    const OPERATIONS = home.OPERATION_LABELS;
    const cards = new Map();
    let active = true, lastAnnouncement = 0, editorSignature = null, editorInvoker = null, confirmResolve = null;

    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }
    function button(text, key, callback, className = 'identity-secondary') {
      const element = node('button', className, text); element.type = 'button'; element.dataset.key = key;
      element.addEventListener('click', callback); return element;
    }
    function entityName(state, id) { return (state.catalog?.entities || []).find(item => item.entity_id === id)?.name || id || 'Dispositivo sin elegir'; }
    function scheduleText(routine) {
      if (!routine.schedule) return 'Sólo manual';
      return routine.schedule.weekdays.map(day => DAYS[day]?.[0] || '?').join(', ') + ' · ' + routine.schedule.at + ' · Montevideo';
    }
    function focusKey() { const element = doc.activeElement; return element && container.contains(element) || dialog.contains(element) ? element.dataset?.key || null : null; }
    function restoreFocus(scope, key, selection) {
      if (!key) return;
      const target = [...scope.querySelectorAll('[data-key]')].find(element => element.dataset.key === key);
      if (target && !target.disabled) {
        target.focus({ preventScroll: true });
        if (selection && typeof target.setSelectionRange === 'function') { try { target.setSelectionRange(...selection); } catch (_) { /* Not a text input. */ } }
      }
    }

    // Static skeleton: heading, freshness, permission note, live region and grid.
    const heading = node('div', 'identity-heading routines-heading');
    const intro = node('div');
    const title = node('h2', '', 'Rutinas de tu hogar.'); title.id = 'routines-title';
    intro.append(node('p', 'eyebrow', 'TU CASA, A TU RITMO'), title, node('p', '', 'Horarios y acciones guardados en KZA. Las automatizaciones de Home Assistant no aparecen aquí.'));
    const side = node('div', 'identity-heading-side routines-heading-side');
    const create = button('Crear rutina', 'create', () => { editorInvoker = 'create'; void startEditor('create', replace => owner.newDraft({ replace })); }, 'identity-primary');
    create.id = 'routines-create';
    // aria-disabled instead of disabled: a disabled focused button drops keyboard focus to <body>.
    const refresh = button('Actualizar', 'refresh', () => { if (refresh.getAttribute('aria-disabled') !== 'true') void owner.refresh(); });
    refresh.id = 'routines-refresh';
    side.append(create, refresh); heading.append(intro, side);
    const permission = node('p', 'routines-permission');
    const freshness = node('p', 'routines-freshness'); freshness.id = 'routines-freshness';
    const createNotice = node('p', 'routine-notice'); createNotice.dataset.routineNotice = 'new';
    const sessionNotice = node('p', 'routine-notice routine-session-notice'); sessionNotice.dataset.routineNotice = 'session';
    // A draft kept while its save was uncertain, offered back when the save turned out not applied.
    const recovery = node('div', 'routine-recovery');
    const recoveryText = node('p', 'routine-notice');
    const recover = button('Recuperar borrador', 'recover', () => { editorInvoker = 'recover'; owner.reopenDraft(); }, 'identity-primary');
    const discard = button('Descartar borrador', 'discard', () => owner.discardDraft());
    recovery.append(recoveryText, recover, discard);
    const live = node('p', 'routines-live'); live.id = 'routines-live'; live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
    const grid = node('div', 'routine-grid'); grid.id = 'routines-list';
    const empty = node('p', 'identity-empty routines-empty');
    container.replaceChildren(heading, permission, sessionNotice, freshness, createNotice, recovery, live, grid, empty);

    function runBlock(state, run, caption) {
      const block = node('div', 'routine-run'); block.dataset.runId = run.run_id;
      block.append(node('p', 'routine-run-caption', caption));
      const status = node('p', 'routine-run-status', (SOURCE_LABELS[run.source] || 'Origen desconocido') + ' · ' + (RUN_LABELS[run.status] || 'Estado desconocido'));
      status.dataset.runStatus = run.status; block.append(status);
      if (run.reason === 'routine_busy') block.append(node('p', 'routine-detail', 'La rutina ya estaba en curso.'));
      const list = node('ol', 'routine-run-actions');
      for (const item of run.actions) {
        const line = node('li', '', entityName(state, item.entity_id) + ' · ' + (ACTION_LABELS[item.status] || 'Estado desconocido'));
        line.dataset.actionStatus = item.status; list.append(line);
      }
      block.append(list); return block;
    }
    function cardSignature(state, routine) {
      const id = routine.routine_id;
      return JSON.stringify([routine, state.attempts[id] || null, state.writes[id] || null, state.notices[id] || null, state.busy.includes(id), state.locked[id],
        state.capabilities, routine.actions.map(item => entityName(state, item.entity_id)), (routine.latest_run?.actions || []).map(item => entityName(state, item.entity_id))]);
    }
    function buildCard(state, routine, card) {
      const id = routine.routine_id, attempt = state.attempts[id], write = state.writes[id], caps = state.capabilities, locked = state.locked[id];
      const top = node('div', 'routine-card-top');
      const badge = node('span', 'scene-icon ' + (routine.schedule ? 'lavender' : 'mint')); badge.append(icon(routine.schedule ? 'clock' : 'bolt'));
      const name = node('div', 'routine-card-name'); name.append(node('h3', '', routine.name), node('span', 'identity-badge', routine.enabled ? 'Activa' : 'Desactivada'));
      top.append(badge, name); card.append(top);
      if (routine.description) card.append(node('p', 'routine-detail', routine.description));
      card.append(node('p', 'routine-detail', scheduleText(routine)));
      card.append(node('p', 'routine-detail', `${routine.actions.length} ${routine.actions.length === 1 ? 'acción' : 'acciones'} · Revisión ${routine.revision}`));
      if (write) card.append(node('p', 'routine-state', write.state === 'pending' ? 'Enviando el cambio…' : 'No sabemos si se aplicó el último cambio. Consultando…'));
      if (attempt?.state === 'unresolved') card.append(node('p', 'routine-state', state.busy.includes(id) ? 'Enviando la ejecución…' : 'Resultado pendiente de consulta.'));
      if (attempt?.state === 'active') card.append(node('p', 'routine-state', 'Ejecución en curso.'));
      if (attempt?.run) card.append(runBlock(state, attempt.run, 'Resultado de tu intento'));
      if (routine.latest_run && routine.latest_run.run_id !== attempt?.run?.run_id) card.append(runBlock(state, routine.latest_run, 'Última ejecución registrada'));
      if (attempt?.run || routine.latest_run) card.append(node('p', 'routine-footnote', '«Confirmada» es lo que informó Home Assistant; no comprueba el resultado físico.'));
      const message = node('p', 'routine-notice', state.notices[id] || ''); message.dataset.routineNotice = id; card.append(message);
      const controls = node('div', 'routine-controls');
      if (caps.execute) {
        const running = routine.latest_run?.status === 'running' || attempt?.state === 'active';
        if (!attempt || attempt.state === 'terminal') {
          const run = button(attempt ? 'Ejecutar otra vez' : 'Ejecutar', id + ':execute', () => runRoutine(id), 'identity-primary');
          run.disabled = locked || running || !routine.enabled; run.dataset.routineWrite = 'execute'; controls.append(run);
        } else if (attempt.state === 'unresolved') {
          const retry = button('Reintentar este intento', id + ':retry', () => owner.retryAttempt(id));
          retry.disabled = state.busy.includes(id); retry.dataset.routineWrite = 'retry'; controls.append(retry);
        }
      }
      if (caps.manage && (attempt || routine.latest_run)) {
        const consult = button('Consultar resultado', id + ':consult', () => owner.consult(id)); consult.disabled = state.busy.includes(id); controls.append(consult);
      }
      if (caps.manage) {
        const editControl = button('Editar', id + ':edit', () => { editorInvoker = id + ':edit'; void startEditor(id + ':edit', replace => owner.edit(id, { replace })); }); editControl.disabled = locked; editControl.dataset.routineWrite = 'edit';
        const toggle = button(routine.enabled ? 'Desactivar' : 'Activar', id + ':toggle', () => owner.setEnabled(id, !routine.enabled)); toggle.disabled = locked; toggle.dataset.routineWrite = 'toggle';
        const removal = button('Borrar', id + ':delete', () => confirmDelete(id), 'identity-danger'); removal.disabled = locked; removal.dataset.routineWrite = 'delete';
        controls.append(editControl, toggle, removal);
      }
      if (controls.childElementCount) card.append(controls);
      if (!routine.enabled && caps.execute) card.append(node('p', 'routine-footnote', 'Activá la rutina para poder ejecutarla.'));
    }
    function renderCards(state) {
      const key = focusKey(), seen = new Set();
      let previous = null;
      for (const routine of state.routines) {
        const id = routine.routine_id, signature = cardSignature(state, routine);
        seen.add(id);
        let entry = cards.get(id);
        if (!entry) { const card = node('article', 'routine-card panel'); card.dataset.routineId = id; card.setAttribute('aria-label', 'Rutina ' + routine.name); entry = { card, signature: null }; cards.set(id, entry); }
        if (entry.signature !== signature) {
          const hadFocus = entry.card.contains(doc.activeElement);
          entry.card.replaceChildren(); entry.card.setAttribute('aria-label', 'Rutina ' + routine.name);
          buildCard(state, routine, entry.card); entry.signature = signature;
          if (hadFocus) restoreFocus(entry.card, key);
        }
        const expected = previous ? previous.nextSibling : grid.firstChild;
        if (expected !== entry.card) grid.insertBefore(entry.card, expected);
        previous = entry.card;
      }
      for (const [id, entry] of cards) if (!seen.has(id)) { entry.card.remove(); cards.delete(id); }
    }

    async function runRoutine(id, confirmed = false) {
      const result = await owner.execute(id, { confirmed });
      if (!active || !result?.needsConfirmation) return;
      const state = owner.getState(), routine = state.routines.find(item => item.routine_id === id);
      const items = result.needsConfirmation.map(item => entityName(state, item.entity_id) + ' · ' + (ACTION_LABELS[item.status] || 'Estado desconocido'));
      const accepted = await ask({ title: 'Repetir una ejecución incierta', body: `La última ejecución de «${routine?.name || 'la rutina'}» no se confirmó por completo. Puede que algunas acciones ya se hayan aplicado. Revisá la casa antes de repetir.`,
        items, confirm: 'Ejecutar igualmente', danger: false, invoker: id + ':execute' });
      if (accepted && active) await owner.execute(id, { confirmed: true });
    }
    /** A kept draft is replaced only after the person chooses to discard it. */
    async function startEditor(invoker, start) {
      const kept = owner.getState().draft;
      if (!kept || kept.open) { start(false); return; }
      if (kept.unknown) return;
      const accepted = await ask({ title: 'Hay un borrador conservado', body: `«${kept.value.name || 'Sin nombre'}» no se guardó. Si seguís, ese borrador se descarta. Para conservarlo, cancelá y elegí «Recuperar borrador».`,
        items: [], confirm: 'Descartar borrador y seguir', danger: true, invoker });
      if (accepted && active) { editorInvoker = invoker; start(true); }
    }
    async function confirmDelete(id) {
      const routine = owner.getState().routines.find(item => item.routine_id === id);
      if (!routine) return;
      const accepted = await ask({ title: 'Borrar rutina', body: `Vas a borrar «${routine.name}» (revisión ${routine.revision}). Sus resultados anteriores se conservan en el servidor.`, items: [], confirm: 'Borrar rutina', danger: true, invoker: id + ':delete' });
      if (accepted && active) await owner.remove(id);
    }
    function ask({ title: heading, body, items, confirm, danger, invoker }) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        const titleNode = node('h2', '', heading); titleNode.id = 'routine-confirm-title';
        const text = node('p', 'modal-description', body);
        const list = node('ul', 'routine-confirm-items'); items.forEach(item => list.append(node('li', '', item)));
        const actions = node('div', 'modal-actions routine-confirm-actions');
        const cancelButton = button('Cancelar', 'confirm-cancel', () => finish(false));
        const accept = button(confirm, 'confirm-accept', () => finish(true), danger ? 'identity-danger' : 'identity-primary');
        actions.append(cancelButton, accept);
        confirmDialog.replaceChildren(titleNode, text, ...(items.length ? [list] : []), actions);
        confirmDialog.setAttribute('aria-labelledby', 'routine-confirm-title');
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          restoreFocus(container, invoker);
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish;
        if (!confirmDialog.open) confirmDialog.showModal();
        cancelButton.focus();
      });
    }

    // Editor: rebuilt only on structural changes so typing never loses focus.
    function editorStructure(state) {
      const draft = state.draft;
      if (!draft) return null;
      const value = draft.value;
      return JSON.stringify([draft.editingId, draft.editingRevision, Boolean(value.schedule), value.schedule?.weekdays || null, value.enabled,
        value.conditions.map(item => item.entity_id), value.actions.map(item => [item.entity_id, item.operation, home.controlFields(entityByIdFrom(state, item.entity_id), item.operation, item).map(field => field.name)]),
        draft.validation, draft.error, draft.message, draft.preview, draft.saving, draft.previewing, draft.unknown, state.catalogUsable, (state.catalog?.entities || []).length]);
    }
    function entityByIdFrom(state, id) { return (state.catalog?.entities || []).find(item => item.entity_id === id); }
    function field(label, control, key, className = 'routine-field') {
      const wrap = node('label', className); control.dataset.key = key;
      wrap.append(node('span', 'routine-field-label', label), control); return wrap;
    }
    function input(type, value, onInput) {
      const element = node('input'); element.type = type;
      if (type === 'checkbox') element.checked = Boolean(value); else element.value = value ?? '';
      element.addEventListener(type === 'checkbox' ? 'change' : 'input', () => onInput(type === 'checkbox' ? element.checked : element.value));
      return element;
    }
    function select(options, value, onChange) {
      const element = node('select');
      for (const [id, label] of options) { const option = node('option', '', label); option.value = id; element.append(option); }
      element.value = value ?? ''; element.addEventListener('change', () => onChange(element.value)); return element;
    }
    function update(mutator) {
      const draft = owner.getState().draft;
      if (!draft) return;
      const value = draft.value; mutator(value); owner.setDraft(value);
    }
    function entitySelect(state, value, onChange, actionable) {
      const options = [['', 'Elegí un dispositivo'], ...(state.catalog?.entities || []).filter(item => !actionable || item.capabilities?.some(cap => Object.hasOwn(OPERATIONS, cap)))
        .map(item => [item.entity_id, item.name || item.entity_id])];
      if (value && !options.some(([id]) => id === value)) options.push([value, value + ' (fuera del catálogo)']);
      return select(options, value || '', onChange);
    }
    function renderEditor(state) {
      const draft = state.draft, value = draft.value;
      const form = node('form', 'identity-form routine-form'); form.noValidate = true;
      const titleNode = node('h2', '', draft.editingId ? 'Editar rutina' : 'Nueva rutina'); titleNode.id = 'routine-editor-title';
      const top = node('div', 'dialog-top routine-dialog-top'); top.append(node('p', 'eyebrow', draft.editingId ? `REVISIÓN ${draft.editingRevision}` : 'RUTINA NUEVA'));
      form.append(top, titleNode);
      if (!state.catalogUsable) form.append(node('p', 'routine-warning', state.catalogStatus.loading ? 'Consultando el catálogo de dispositivos…' : 'Actualizá el catálogo de dispositivos antes de guardar.'));
      const fields = node('fieldset', 'routine-fieldset'); fields.disabled = draft.saving || draft.unknown;
      fields.append(node('legend', 'routine-section', 'Datos'));
      fields.append(field('Nombre', input('text', value.name, text => update(d => { d.name = text; })), 'name'));
      fields.append(field('Descripción (opcional)', input('text', value.description, text => update(d => { d.description = text; })), 'description'));
      const enabled = input('checkbox', value.enabled, checked => update(d => { d.enabled = checked; }));
      fields.append(field('Activa', enabled, 'enabled', 'routine-field routine-check'));
      const when = node('fieldset', 'routine-fieldset'); when.append(node('legend', 'routine-section', 'Cuándo'));
      when.append(field('Modo', select([['manual', 'Sólo manual'], ['scheduled', 'Horario y días']], value.schedule ? 'scheduled' : 'manual',
        mode => update(d => { d.schedule = mode === 'manual' ? null : { at: '', weekdays: [] }; })), 'mode'));
      if (value.schedule) {
        // Two selects instead of input[type=time]: its inner segments trap Tab and vary by browser locale.
        const [hour = '', minute = ''] = (value.schedule.at || '').split(':');
        const pad = number => String(number).padStart(2, '0');
        // Read the current draft: the other select may have changed without a rebuild.
        const setTime = (part, next) => update(d => {
          const [currentHour = '', currentMinute = ''] = (d.schedule.at || '').split(':');
          const nextHour = part === 'hour' ? next : currentHour, nextMinute = part === 'minute' ? next : currentMinute;
          d.schedule.at = nextHour || nextMinute ? `${nextHour}:${nextMinute}` : '';
        });
        const time = node('div', 'routine-time');
        time.append(field('Hora (Montevideo)', select([['', 'Elegí la hora'], ...Array.from({ length: 24 }, (_, n) => [pad(n), pad(n)])], hour, next => setTime('hour', next)), 'hour'),
          field('Minutos', select([['', 'Elegí los minutos'], ...Array.from({ length: 60 }, (_, n) => [pad(n), pad(n)])], minute, next => setTime('minute', next)), 'minute'));
        when.append(time);
        const days = node('div', 'routine-days'); days.setAttribute('role', 'group'); days.setAttribute('aria-label', 'Días de la semana');
        DAYS.forEach(([short, full], index) => {
          const pressed = value.schedule.weekdays.includes(index);
          const toggle = button(short, 'day:' + index, () => update(d => { d.schedule.weekdays = pressed ? d.schedule.weekdays.filter(day => day !== index) : [...d.schedule.weekdays, index].sort((a, b) => a - b); }), 'routine-day');
          toggle.setAttribute('aria-pressed', String(pressed)); toggle.setAttribute('aria-label', full); days.append(toggle);
        });
        when.append(node('p', 'routine-field-label', 'Días'), days);
      }
      const conditions = node('fieldset', 'routine-fieldset'); conditions.append(node('legend', 'routine-section', 'Condiciones de estado (opcionales)'));
      value.conditions.forEach((item, index) => {
        const row = node('div', 'routine-row');
        row.append(field(`Dispositivo de la condición ${index + 1}`, entitySelect(state, item.entity_id, id => update(d => { d.conditions[index].entity_id = id; }), false), `condition:${index}:entity`));
        row.append(field(`Estado requerido de la condición ${index + 1} (por ejemplo, on)`, input('text', item.state, text => update(d => { d.conditions[index].state = text; })), `condition:${index}:state`));
        row.append(button(`Quitar condición ${index + 1}`, `condition:${index}:remove`, () => update(d => { d.conditions.splice(index, 1); })));
        conditions.append(row);
      });
      conditions.append(button('Agregar condición', 'condition:add', () => update(d => { d.conditions.push({ entity_id: '', state: '' }); })));
      const actions = node('fieldset', 'routine-fieldset'); actions.append(node('legend', 'routine-section', 'Acciones en orden'));
      value.actions.forEach((item, index) => {
        const row = node('div', 'routine-row'), entity = entityByIdFrom(state, item.entity_id);
        const options = Object.entries(OPERATIONS).filter(([key]) => entity?.capabilities?.includes(key));
        if (item.operation && !options.some(([key]) => key === item.operation)) options.unshift([item.operation, (OPERATIONS[item.operation] || item.operation) + ' (no disponible)']);
        row.append(field(`Dispositivo de la acción ${index + 1}`, entitySelect(state, item.entity_id, id => update(d => { d.actions[index] = { entity_id: id, operation: '' }; }), true), `action:${index}:entity`));
        row.append(field(`Operación de la acción ${index + 1}`, select([['', 'Elegí una operación'], ...options], item.operation, operation => update(d => { d.actions[index] = { entity_id: item.entity_id, operation }; })), `action:${index}:operation`));
        for (const descriptor of home.controlFields(entity, item.operation, item)) {
          const change = text => update(d => { d.actions[index][descriptor.name] = text === '' ? null : descriptor.options ? text : Number(text); });
          const control = descriptor.options ? select([['', 'Elegí un modo'], ...descriptor.options], item[descriptor.name] || '', change) : input('number', item[descriptor.name] ?? '', change);
          if (!descriptor.options) { if (Number.isFinite(descriptor.min)) control.min = descriptor.min; if (Number.isFinite(descriptor.max)) control.max = descriptor.max; control.step = descriptor.step || 1; }
          row.append(field(`${descriptor.label} de la acción ${index + 1}`, control, `action:${index}:${descriptor.name}`));
        }
        const controls = node('div', 'routine-row-controls');
        const move = offset => update(d => { const entry = d.actions.splice(index, 1)[0]; d.actions.splice(index + offset, 0, entry); });
        const up = button(`Subir acción ${index + 1}`, `action:${index}:up`, () => move(-1)); up.disabled = index === 0;
        const down = button(`Bajar acción ${index + 1}`, `action:${index}:down`, () => move(1)); down.disabled = index === value.actions.length - 1;
        controls.append(up, down, button(`Quitar acción ${index + 1}`, `action:${index}:remove`, () => update(d => { d.actions.splice(index, 1); })));
        row.append(controls); actions.append(row);
      });
      actions.append(button('Agregar acción', 'action:add', () => update(d => { d.actions.push({ entity_id: '', operation: '' }); })));
      form.append(fields, when, conditions, actions);
      const problems = [...draft.validation, ...(draft.error ? [draft.error.message] : [])];
      const summary = node('div', 'routine-errors'); summary.id = 'routine-errors'; summary.tabIndex = -1; summary.setAttribute('role', 'alert'); summary.dataset.key = 'errors';
      if (problems.length) { const list = node('ul'); problems.forEach(text => list.append(node('li', '', text))); summary.append(list); }
      form.append(summary);
      if (draft.message) form.append(node('p', 'routine-state', draft.message));
      if (draft.error?.conflict) form.append(button('Descartar borrador y recargar', 'reload', () => owner.discardAndReload()));
      if (draft.preview) {
        const panel = node('section', 'routine-preview'); panel.setAttribute('aria-label', 'Vista previa');
        panel.append(node('p', 'identity-badge', 'Vista previa · sin guardar ni ejecutar'), node('p', 'routine-detail', scheduleText(draft.preview)));
        const list = node('ol', 'routine-run-actions');
        draft.preview.actions.forEach(item => list.append(node('li', '', entityName(state, item.entity_id) + ' · ' + (OPERATIONS[item.operation] || 'Operación desconocida'))));
        panel.append(list, node('p', 'routine-footnote', 'La vista previa no consulta el estado de los dispositivos ni evalúa condiciones.')); form.append(panel);
      }
      const footer = node('div', 'modal-actions routine-editor-actions');
      const close = button('Cerrar', 'close', () => owner.closeEditor()); close.disabled = draft.saving;
      const previewButton = button(draft.previewing ? 'Consultando vista previa…' : 'Previsualizar', 'preview', () => owner.preview()); previewButton.disabled = draft.previewing || draft.unknown || !state.catalogUsable;
      const saveButton = node('button', 'identity-primary', draft.saving ? 'Guardando…' : 'Guardar'); saveButton.type = 'submit'; saveButton.dataset.key = 'save';
      saveButton.disabled = draft.saving || draft.unknown || !state.catalogUsable;
      footer.append(close, previewButton, saveButton); form.append(footer);
      form.addEventListener('submit', async event => {
        event.preventDefault();
        const result = await owner.save();
        // Failed saves move focus to the summary that explains why.
        if (active && result && !result.ok && !result.unknown && dialog.open) dialog.querySelector('#routine-errors')?.focus();
      });
      return form;
    }
    function renderDialog(state) {
      if (!state.draft?.open) {
        editorSignature = null;
        if (dialog.open) { dialog.close(); dialog.replaceChildren(); restoreFocus(container, editorInvoker); }
        return;
      }
      const signature = editorStructure(state);
      if (signature !== editorSignature) {
        const element = doc.activeElement, key = dialog.contains(element) ? element.dataset?.key : null;
        const selection = key && typeof element.selectionStart === 'number' ? [element.selectionStart, element.selectionEnd] : null;
        dialog.replaceChildren(renderEditor(state)); dialog.setAttribute('aria-labelledby', 'routine-editor-title');
        editorSignature = signature;
        if (!dialog.open) { dialog.showModal(); dialog.querySelector('[data-key=name]')?.focus(); }
        else restoreFocus(dialog, key, selection);
      }
    }
    function onCancel(event) { event.preventDefault(); owner.closeEditor(); }
    dialog.addEventListener('cancel', onCancel);

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      create.hidden = !caps.manage; create.disabled = Boolean(state.draft?.saving) || Boolean(state.writes.new);
      refresh.setAttribute('aria-disabled', String(state.list.loading));
      permission.textContent = caps.manage ? (caps.execute ? '' : 'Tu acceso permite editar rutinas pero no ejecutarlas.') :
        'Tu acceso permite ver las rutinas y su último resultado. Crearlas, cambiarlas o ejecutarlas requiere un acceso de administración.';
      permission.hidden = !permission.textContent;
      const observed = state.list.observedAt ? 'Consultadas a las ' + new Date(state.list.observedAt).toLocaleTimeString('es-UY', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }) : 'Sin consulta confirmada';
      freshness.textContent = state.list.error ? `${state.list.error.message} ${state.list.observedAt ? 'Se muestran los últimos datos. ' + observed + '.' : ''}`.trim()
        : state.list.loading && !state.list.loaded ? 'Consultando rutinas…' : observed + '.';
      freshness.classList.toggle('is-error', Boolean(state.list.error));
      createNotice.textContent = state.writes.new ? (state.writes.new.state === 'pending' ? 'Enviando la nueva rutina…' : 'No sabemos si se guardó la nueva rutina. Consultando…') : state.notices.new || '';
      createNotice.hidden = !createNotice.textContent;
      sessionNotice.textContent = state.notices.session || ''; sessionNotice.hidden = !sessionNotice.textContent;
      const hidden = state.draft && !state.draft.open ? state.draft : null;
      recovery.hidden = !hidden;
      if (hidden) {
        recoveryText.textContent = hidden.unknown ? 'Tu borrador se conserva mientras confirmamos si se guardó.' : `Borrador conservado: ${hidden.value.name || 'sin nombre'}. ${hidden.message || ''}`.trim();
        recover.hidden = discard.hidden = Boolean(hidden.unknown);
      }
      renderCards(state);
      empty.hidden = !(state.list.loaded && !state.routines.length);
      empty.textContent = caps.manage ? 'Todavía no hay rutinas. Creá la primera con dispositivos de tu casa.' : 'Todavía no hay rutinas.';
      if (state.announcement && state.announcement.serial !== lastAnnouncement) { lastAnnouncement = state.announcement.serial; live.textContent = state.announcement.text; }
      renderDialog(state);
    }
    const stop = owner.subscribe(render);
    if (owner.getState().announcement) lastAnnouncement = owner.getState().announcement.serial;
    render();
    return () => {
      active = false; stop(); dialog.removeEventListener('cancel', onCancel);
      if (confirmResolve) confirmResolve(false);
      if (dialog.open) dialog.close();
      dialog.replaceChildren(); container.replaceChildren(); cards.clear();
    };
  }
  const exported = { mountRoutines };
  root.NexoRoutines = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
