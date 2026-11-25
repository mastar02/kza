/* NEXO Rutinas view: DOM only. Rows are keyed by routine and rebuilt only when
 * their rendered data changes; focus returns to the control with the same key. */
(function (root) {
  'use strict';
  const RUN_LABELS = { running: 'En curso', succeeded: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta: Home Assistant no confirmó todas las acciones', skipped: 'Omitida por condición', interrupted: 'Interrumpida' };
  const ACTION_LABELS = { pending: 'Pendiente', sending: 'Enviando', confirmed: 'Confirmada', failed: 'Fallida', uncertain: 'Incierta', not_executed: 'No ejecutada' };
  const SOURCE_LABELS = { manual: 'Manual', schedule: 'Programada', scheduled: 'Programada', household_voice: 'Por voz' };
  const BRAKE_LABELS = { stop_requested: 'Último freno por voz: se detuvieron las acciones pendientes.',
    nothing_to_stop: 'Último freno por voz: no quedaban acciones pendientes.',
    disabled: 'Último freno por voz: se desactivó la rutina para revisión.',
    already_disabled: 'Último freno por voz: la rutina ya estaba desactivada.' };
  const DAYS = [['Lu', 'Lunes'], ['Ma', 'Martes'], ['Mi', 'Miércoles'], ['Ju', 'Jueves'], ['Vi', 'Viernes'], ['Sá', 'Sábado'], ['Do', 'Domingo']];
  const PROPOSAL_LABELS = { create: 'Crear rutina', update: 'Editar rutina', delete: 'Borrar rutina', enable: 'Activar rutina', disable: 'Desactivar rutina' };
  const PARAMETER_LABELS = { brightness: 'Brillo', brightness_pct: 'Brillo (%)', color_temp_kelvin: 'Temperatura de color', temperature: 'Temperatura', hvac_mode: 'Modo del aire', position: 'Posición' };
  const STATE_LABELS = { on: 'Encendido', off: 'Apagado', open: 'Abierto', closed: 'Cerrado', opening: 'Abriendo', closing: 'Cerrando', unavailable: 'No disponible', unknown: 'Desconocido',
    cool: 'Frío', heat: 'Calor', dry: 'Deshumidificar', fan_only: 'Ventilación', auto: 'Automático', heat_cool: 'Frío y calor' };
  const CONDITION_EXAMPLES = ['closed', 'open', 'on', 'off'].map(value => `${STATE_LABELS[value]} = ${value}`).join(' · ');
  const DATE_TIME = { dateStyle: 'short', timeStyle: 'short', timeZone: 'America/Montevideo' };

  function mountRoutines({ container, dialog, confirmDialog, owner }) {
    const doc = container.ownerDocument, home = root.HomeUI;
    const OPERATIONS = home.OPERATION_LABELS;
    const cards = new Map();
    const controlKeys = new WeakMap();
    let active = true, lastAnnouncement = 0, editorSignature = null, editorInvoker = null, confirmResolve = null;
    let deviceNames = new Map();

    function recordControl(control, key) {
      controlKeys.set(control, key);
      control.dataset.key = key.slice(key.lastIndexOf(':') + 1);
      return control;
    }
    function recordButton(text, key, callback, className = 'btn') {
      return recordControl(button(text, '', callback, className), key);
    }
    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }
    function button(text, key, callback, className = 'btn') {
      const element = node('button', className, text); element.type = 'button'; element.dataset.key = key;
      element.addEventListener('click', callback); return element;
    }
    function deviceOrdinals(...groups) {
      const ordinals = new Map();
      for (const items of groups) {
        for (const { entity_id: id } of items ?? []) {
          if (id && !ordinals.has(id)) ordinals.set(id, ordinals.size + 1);
        }
      }
      return ordinals;
    }
    function routineOrdinals(state, routine, extraActions) {
      return deviceOrdinals(routine?.conditions, routine?.actions, routine?.latest_run?.actions,
        state.attempts[routine?.routine_id]?.run?.actions, extraActions);
    }
    function entityName(state, id, ordinals) {
      if (!id) return 'Dispositivo sin elegir';
      return deviceNames.get(id) || (state.catalog ? 'Dispositivo no disponible' : 'Dispositivo sin nombre ' + ordinals.get(id));
    }
    function routineMessage(text) {
      if (text === 'Actualizá el catálogo de dispositivos antes de guardar.') return 'Actualizá la lista de dispositivos antes de guardar.';
      return text.replace(/(R|r)evisión (?=\d)/g, (_, initial) => initial === 'R' ? 'Versión ' : 'versión ');
    }
    function scheduleText(routine) {
      if (!routine.schedule) return 'Sin horario programado';
      return routine.schedule.weekdays.map(day => DAYS[day]?.[0] || '?').join(', ') + ' · ' + routine.schedule.at + ' · Montevideo';
    }
    function focusKey() { const element = doc.activeElement; return element && container.contains(element) || dialog.contains(element) ? (controlKeys.get(element) || element.dataset?.key) || null : null; }
    function restoreFocus(scope, key, selection) {
      if (!key) return;
      const target = [...scope.querySelectorAll('[data-key]')].find(element => (controlKeys.get(element) || element.dataset.key) === key);
      if (target && !target.disabled) {
        target.focus({ preventScroll: true });
        if (selection && typeof target.setSelectionRange === 'function') { try { target.setSelectionRange(...selection); } catch (_) { /* Not a text input. */ } }
      }
    }

    const heading = node('header', 'page-head');
    const intro = node('div');
    const title = node('h1', '', 'Rutinas'); title.id = 'routines-title';
    intro.append(title, node('p', '', 'Acciones guardadas en KZA para ejecutar ahora o a una hora programada.'));
    const side = node('div', 'row-actions');
    const create = button('Crear rutina', 'create', () => { editorInvoker = 'create'; void startEditor('create', replace => owner.newDraft({ replace })); }, 'btn primary');
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
    const recovery = node('div', 'detail');
    const recoveryText = node('p', 'routine-notice');
    const recover = button('Recuperar borrador', 'recover', () => { editorInvoker = 'recover'; owner.reopenDraft(); }, 'btn primary');
    const discard = button('Descartar borrador', 'discard', () => owner.discardDraft(), 'btn quiet');
    const recoveryActions = node('div', 'row-actions');
    recoveryActions.append(recover, discard);
    recovery.append(recoveryText, recoveryActions);
    const live = node('p', 'routines-live'); live.id = 'routines-live'; live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
    const routineSection = node('section', 'routine-section');
    const routineTitle = node('h2', '', 'Tus rutinas'); routineTitle.id = 'routines-list-title';
    routineSection.setAttribute('aria-labelledby', routineTitle.id);
    const list = node('div', 'rows'); list.id = 'routines-list';
    const empty = node('p', 'empty routines-empty');
    list.append(empty);
    routineSection.append(routineTitle, list);
    container.replaceChildren(heading, permission, sessionNotice, freshness, createNotice, recovery, live, routineSection);

    const proposalSection = node('section', 'routine-section'); proposalSection.id = 'routine-proposals';
    const proposalTitle = node('h2', '', 'Propuestas por voz'); proposalTitle.id = 'routine-proposals-title';
    proposalSection.setAttribute('aria-labelledby', proposalTitle.id);
    const proposalStatus = node('p', 'routine-notice'); proposalStatus.setAttribute('role', 'status');
    const proposalRefresh = button('Actualizar propuestas', 'proposals-refresh', () => owner.refreshProposals());
    const proposalHeading = node('header', 'routine-section-head');
    proposalHeading.append(proposalTitle, proposalRefresh);
    const proposalList = node('div', 'rows');
    proposalSection.append(proposalHeading, proposalStatus, proposalList);
    let proposalSignature;

    function proposalDefinition(state, label, value, ordinals) {
      const panel = node('div', 'detail routine-definition'); panel.dataset.proposalDiff = label === 'Antes' ? 'before' : 'after';
      panel.append(node('h4', '', label));
      if (value === null) { panel.append(node('p', 'row-meta', 'Sin rutina.')); return panel; }
      const fields = node('dl', 'routine-definition-fields');
      for (const [term, detail] of [
        ['Nombre', value.name], ['Descripción', value.description || 'Sin descripción'],
        ['Estado', value.enabled ? 'Activa' : 'Desactivada'], ['Horario', scheduleText(value)]
      ]) fields.append(node('dt', '', term), node('dd', '', detail));
      panel.append(fields);
      if (value.conditions.length) {
        const conditions = node('ul', 'routine-run-actions');
        for (const item of value.conditions) {
          const status = Object.hasOwn(STATE_LABELS, item.state) ? STATE_LABELS[item.state] : item.state;
          conditions.append(node('li', '', `${entityName(state, item.entity_id, ordinals)} · ${status}`));
        }
        panel.append(node('p', 'routine-run-caption', 'Condiciones'), conditions);
      }
      const actions = node('ol', 'routine-run-actions');
      for (const item of value.actions) {
        // A device may have left the catalog since the voice request. Keep every parameter visible.
        const entity = entityByIdFrom(state, item.entity_id);
        const fields = home.controlFields(entity, item.operation, item);
        const values = (home.OPERATION_PARAMETERS[item.operation] || []).filter(key => Object.hasOwn(item, key))
          .map(key => {
            const field = fields.find(entry => entry.name === key);
            const label = field?.label || PARAMETER_LABELS[key];
            const value = field?.options?.find(([id]) => id === item[key])?.[1]
              ?? (key === 'hvac_mode' && Object.hasOwn(STATE_LABELS, item[key]) ? STATE_LABELS[item[key]] : item[key]);
            return `${label} ${value}`;
          });
        actions.append(node('li', '', [entityName(state, item.entity_id, ordinals), OPERATIONS[item.operation], ...values].filter(Boolean).join(' · ')));
      }
      panel.append(node('p', 'routine-run-caption', 'Acciones en orden'), actions);
      return panel;
    }

    function proposalCard(state, id, item, selected, decision) {
      const card = node('article', 'row routine-proposal');
      const summary = node('div', 'row-main');
      summary.append(node('h3', 'row-title', item ? PROPOSAL_LABELS[item.operation] : 'Decisión sobre una propuesta por voz'));
      if (item && (!selected || ['pending', 'conflict', 'error'].includes(selected.status))) {
        summary.append(node('p', 'row-meta', 'Pendiente · vence ' + new Date(item.expiresAt).toLocaleString('es-UY', DATE_TIME)));
      }
      if (decision?.message || selected?.message) summary.append(node('p', 'routine-state', decision?.message || selected.message));
      card.append(summary);
      if (item && selected?.preview) {
        const { before, after } = selected.preview;
        const ordinals = deviceOrdinals(before?.conditions, before?.actions, after?.conditions, after?.actions);
        const diff = node('section', 'routine-proposal-diff'); diff.setAttribute('aria-label', 'Vista previa de la propuesta');
        diff.append(proposalDefinition(state, 'Antes', before, ordinals), proposalDefinition(state, 'Después', after, ordinals));
        card.append(diff);
      }
      const controls = node('div', 'row-actions');
      if (item || decision) {
        const review = recordButton(selected?.loading ? 'Consultando vista previa…' : selected ? 'Volver a consultar' : 'Revisar propuesta',
          id + ':review', () => owner.reviewProposal(id));
        review.setAttribute('aria-disabled', String(Boolean(selected?.loading || selected && decision)));
        controls.append(review);
      }
      if (selected && decision) {
        const confirm = decision.action === 'confirm';
        const retry = recordButton(decision.sending ? (confirm ? 'Confirmando…' : 'Descartando…') :
          (confirm ? 'Reintentar confirmación' : 'Reintentar descarte'),
          id + ':' + decision.action, () => owner.decideProposal(id, decision.action));
        retry.setAttribute('aria-disabled', String(decision.sending)); controls.append(retry);
      } else if (item && selected?.preview) {
        if (selected.status === 'pending' && !selected.preview.stale) {
          controls.append(recordButton('Confirmar propuesta', id + ':confirm', () => owner.decideProposal(id, 'confirm'), 'btn primary'));
        }
        controls.append(recordButton('Descartar', id + ':reject', () => owner.decideProposal(id, 'reject'), 'btn quiet'));
      }
      card.append(controls);
      return card;
    }

    function renderProposals(state) {
      const tray = state.proposals;
      if (!tray) { proposalSection.remove(); proposalList.replaceChildren(); proposalSignature = undefined; return; }
      if (!container.contains(proposalSection)) container.insertBefore(proposalSection, routineSection);
      proposalRefresh.setAttribute('aria-disabled', String(tray.loading));
      const { items, decisions, selected } = tray;
      const signature = JSON.stringify([items, decisions, selected, selected?.preview ? [state.catalog?.entities, [...deviceNames]] : null]);
      if (signature !== proposalSignature) {
        const focused = proposalList.contains(doc.activeElement), key = focused ? focusKey() : null;
        const rows = new Map(items.map(item => [item.id, item]));
        for (const id of Object.keys(decisions)) if (!rows.has(id)) rows.set(id, null);
        if (selected && !rows.has(selected.id)) rows.set(selected.id, null);
        proposalList.replaceChildren(...[...rows].map(([id, item]) =>
          proposalCard(state, id, item, selected?.id === id ? selected : null, decisions[id])));
        proposalSignature = signature;
        if (focused) {
          restoreFocus(proposalList, key);
          if (selected && !proposalList.contains(doc.activeElement)) restoreFocus(proposalList, selected.id + ':review');
          if (!proposalList.contains(doc.activeElement)) proposalRefresh.focus({ preventScroll: true });
        }
      }
      const text = tray.error || (tray.loading ? 'Consultando propuestas…' :
        tray.loaded && !items.length ? 'No hay propuestas pendientes.' : '');
      if (proposalStatus.textContent !== text) proposalStatus.textContent = text;
    }

    function runBlock(state, run, caption, ordinals) {
      const block = node('div', 'routine-run');
      const scheduled = run.scheduled_for ? new Date(run.scheduled_for) : null;
      const title = node('p', 'routine-run-caption', caption);
      if (scheduled && Number.isFinite(scheduled.getTime())) {
        const time = node('time', 'time', scheduled.toLocaleString('es-UY', DATE_TIME));
        time.dateTime = scheduled.toISOString();
        title.append(' · ', time);
      }
      block.append(title);
      const status = node('p', 'routine-run-status', (SOURCE_LABELS[run.source] || 'Origen desconocido') + ' · ' + (RUN_LABELS[run.status] || 'Estado desconocido'));
      status.dataset.runStatus = run.status; block.append(status);
      if (run.reason === 'routine_busy') block.append(node('p', 'routine-detail', 'La rutina ya estaba en curso.'));
      const list = node('ol', 'routine-run-actions');
      for (const item of run.actions) {
        const line = node('li', '', entityName(state, item.entity_id, ordinals) + ' · ' + (ACTION_LABELS[item.status] || 'Estado desconocido'));
        line.dataset.actionStatus = item.status; list.append(line);
      }
      block.append(list); return block;
    }
    function cardSignature(state, routine, ordinals) {
      const id = routine.routine_id;
      const labels = run => (run?.actions || []).map(item => entityName(state, item.entity_id, ordinals));
      return JSON.stringify([routine, state.attempts[id] || null, state.writes[id] || null, state.notices[id] || null, state.busy.includes(id), state.locked[id],
        state.capabilities, labels(state.attempts[id]?.run), labels(routine.latest_run)]);
    }
    function buildCard(state, routine, card, ordinals) {
      const id = routine.routine_id, attempt = state.attempts[id], write = state.writes[id], caps = state.capabilities, locked = state.locked[id];
      const summary = node('div', 'row-main');
      const title = node('h3', 'row-title', routine.name);
      const status = routine.enabled ? 'Activa' : 'Desactivada';
      if (!caps.manage) title.append(' ', node('span', routine.enabled ? 'tag on' : 'tag', status));
      summary.append(title);
      if (routine.description) summary.append(node('p', '', routine.description));
      summary.append(node('p', 'row-meta', scheduleText(routine)),
        node('p', 'row-meta', `${routine.actions.length} ${routine.actions.length === 1 ? 'acción' : 'acciones'} · Versión ${routine.revision}`));
      card.append(summary);
      const controls = node('div', 'row-actions');
      if (caps.execute) {
        const running = routine.latest_run?.status === 'running' || attempt?.state === 'active';
        if (!attempt || attempt.state === 'terminal') {
          const run = recordButton(attempt ? 'Ejecutar otra vez' : 'Ejecutar ahora', id + ':execute', () => runRoutine(id));
          run.disabled = locked || running || !routine.enabled; run.dataset.routineWrite = 'execute'; controls.append(run);
        } else if (attempt.state === 'unresolved') {
          const retry = recordButton('Reintentar este intento', id + ':retry', () => owner.retryAttempt(id));
          retry.disabled = state.busy.includes(id); retry.dataset.routineWrite = 'retry'; controls.append(retry);
        }
      }
      if (caps.consult && (attempt || routine.latest_run)) {
        const consult = recordButton('Consultar resultado', id + ':consult', () => owner.consult(id)); consult.disabled = state.busy.includes(id); controls.append(consult);
      }
      if (caps.manage) {
        const editControl = recordButton('Editar', id + ':edit', () => { editorInvoker = id + ':edit'; void startEditor(id + ':edit', replace => owner.edit(id, { replace })); }); editControl.disabled = locked; editControl.dataset.routineWrite = 'edit';
        const toggle = recordButton(status, id + ':toggle', () => owner.setEnabled(id, !routine.enabled), 'sw');
        toggle.setAttribute('role', 'switch');
        toggle.setAttribute('aria-checked', String(routine.enabled));
        toggle.setAttribute('aria-label', `Activa o desactivada: ${routine.name}`);
        toggle.disabled = locked; toggle.dataset.routineWrite = 'toggle';
        const removal = recordButton('Borrar', id + ':delete', () => confirmDelete(id), 'btn danger'); removal.disabled = locked; removal.dataset.routineWrite = 'delete';
        controls.append(editControl, toggle, removal);
      }
      if (controls.childElementCount) card.append(controls);
      const receipts = node('div', 'routine-receipts');
      if (write) receipts.append(node('p', 'routine-state', write.state === 'pending' ? 'Enviando el cambio…' : 'No sabemos si se aplicó el último cambio. Consultando…'));
      if (attempt?.state === 'unresolved') receipts.append(node('p', 'routine-state', state.busy.includes(id) ? 'Enviando la ejecución…' : 'Resultado pendiente de consulta.'));
      if (attempt?.state === 'active') receipts.append(node('p', 'routine-state', 'Ejecución en curso.'));
      if (attempt?.run) receipts.append(runBlock(state, attempt.run, 'Resultado de tu intento', ordinals));
      if (routine.latest_run && routine.latest_run.run_id !== attempt?.run?.run_id) receipts.append(runBlock(state, routine.latest_run, 'Última ejecución registrada', ordinals));
      if (routine.latest_voice_brake) receipts.append(node('p', 'row-meta', BRAKE_LABELS[routine.latest_voice_brake.status]));
      if (attempt?.run || routine.latest_run) receipts.append(node('p', 'routine-footnote', '«Confirmada» es lo que informó Home Assistant; no comprueba el resultado físico.'));
      if (state.notices[id]) {
        const message = node('p', 'routine-notice', state.notices[id]); receipts.append(message);
      }
      if (!routine.enabled && caps.execute) receipts.append(node('p', 'routine-footnote', 'La rutina debe estar activa para iniciar una ejecución.'));
      if (receipts.childElementCount) card.append(receipts);
    }
    function renderCards(state) {
      const key = focusKey(), seen = new Set();
      let previous = null;
      for (const routine of state.routines) {
        const id = routine.routine_id, ordinals = routineOrdinals(state, routine);
        const signature = cardSignature(state, routine, ordinals);
        seen.add(id);
        let entry = cards.get(id);
        if (!entry) { const card = node('article', 'row routine-card'); entry = { card, signature: null }; cards.set(id, entry); }
        if (entry.signature !== signature) {
          const hadFocus = entry.card.contains(doc.activeElement);
          entry.card.replaceChildren(); entry.card.setAttribute('aria-label', 'Rutina ' + routine.name);
          buildCard(state, routine, entry.card, ordinals); entry.signature = signature;
          if (hadFocus) restoreFocus(entry.card, key);
        }
        const expected = previous ? previous.nextSibling : list.firstChild;
        if (expected !== entry.card) list.insertBefore(entry.card, expected);
        previous = entry.card;
      }
      for (const [id, entry] of cards) if (!seen.has(id)) { entry.card.remove(); cards.delete(id); }
    }

    async function runRoutine(id, confirmed = false) {
      const result = await owner.execute(id, { confirmed });
      if (!active || !result?.needsConfirmation) return;
      const state = owner.getState(), routine = state.routines.find(item => item.routine_id === id);
      const ordinals = routineOrdinals(state, routine, result.needsConfirmation);
      const items = result.needsConfirmation.map(item => entityName(state, item.entity_id, ordinals) + ' · ' + (ACTION_LABELS[item.status] || 'Estado desconocido'));
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
      const accepted = await ask({ title: 'Borrar rutina', body: `Vas a borrar «${routine.name}» (versión ${routine.revision}). Sus resultados anteriores quedan guardados.`, items: [], confirm: 'Borrar rutina', danger: true, invoker: id + ':delete' });
      if (accepted && active) await owner.remove(id);
    }
    function ask({ title: heading, body, items, confirm, danger, invoker }) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        const titleNode = node('h2', '', heading); titleNode.id = 'routine-confirm-title';
        const text = node('p', '', body);
        const list = node('ul', 'routine-confirm-items'); items.forEach(item => list.append(node('li', '', item)));
        const actions = node('div', 'row-actions');
        const cancelButton = button('Cancelar', 'confirm-cancel', () => finish(false));
        const accept = button(confirm, 'confirm-accept', () => finish(true), danger ? 'btn danger' : 'btn primary');
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
        draft.validation, draft.error, draft.message, draft.preview, draft.saving, draft.previewing, draft.unknown,
        state.catalogUsable, state.catalog?.entities.length, [...deviceNames]]);
    }
    function entityByIdFrom(state, id) { return (state.catalog?.entities || []).find(item => item.entity_id === id); }
    function field(label, control, key, className = 'field') {
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
    function entitySelect(state, value, onChange, actionable, ordinals) {
      const options = [['', 'Elegí un dispositivo'], ...(state.catalog?.entities || []).filter(item => !actionable || item.capabilities?.some(cap => Object.hasOwn(OPERATIONS, cap)))
        .map(item => [item.entity_id, entityName(state, item.entity_id, ordinals)])];
      if (value && !options.some(([id]) => id === value)) {
        options.push([value, state.catalog ? 'Dispositivo no disponible' : entityName(state, value, ordinals)]);
      }
      const ids = new Map(options.map(([id], index) => [index ? String(index) : '', id]));
      const choice = [...ids].find(([, id]) => id === (value || ''))?.[0];
      return select(options.map(([, label], index) => [index ? String(index) : '', label]), choice,
        key => onChange(ids.get(key) ?? ''));
    }
    function renderEditor(state) {
      const draft = state.draft, value = draft.value;
      const ordinals = deviceOrdinals(value.conditions, value.actions, draft.preview?.conditions, draft.preview?.actions);
      const form = node('form', 'routine-form'); form.noValidate = true;
      const titleNode = node('h2', '', draft.editingId ? 'Editar rutina' : 'Nueva rutina'); titleNode.id = 'routine-editor-title';
      form.append(titleNode);
      if (draft.editingId) form.append(node('p', 'row-meta', `Versión ${draft.editingRevision}`));
      if (!state.catalogUsable) form.append(node('p', 'routine-warning', state.catalogStatus.loading ? 'Consultando los dispositivos…' : 'Actualizá la lista de dispositivos antes de guardar.'));
      const fields = node('fieldset', 'routine-fieldset'); fields.disabled = draft.saving || draft.unknown;
      fields.append(node('legend', 'routine-section', 'Datos'));
      fields.append(field('Nombre', input('text', value.name, text => update(d => { d.name = text; })), 'name'));
      fields.append(field('Descripción (opcional)', input('text', value.description, text => update(d => { d.description = text; })), 'description'));
      const enabled = input('checkbox', value.enabled, checked => update(d => { d.enabled = checked; }));
      fields.append(field('Activa', enabled, 'enabled', 'field routine-check'));
      const when = node('fieldset', 'routine-fieldset'); when.append(node('legend', 'routine-section', 'Cuándo'));
      when.append(field('Modo', select([['manual', 'Sin horario programado'], ['scheduled', 'Horario y días']], value.schedule ? 'scheduled' : 'manual',
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
          const toggle = button(short, 'day:' + index, () => update(d => { d.schedule.weekdays = pressed ? d.schedule.weekdays.filter(day => day !== index) : [...d.schedule.weekdays, index].sort((a, b) => a - b); }), 'btn routine-day');
          toggle.setAttribute('aria-pressed', String(pressed)); toggle.setAttribute('aria-label', full); days.append(toggle);
        });
        when.append(node('p', 'routine-field-label', 'Días'), days);
      }
      const conditions = node('fieldset', 'routine-fieldset'); conditions.append(node('legend', 'routine-section', 'Condiciones de estado (opcionales)'));
      value.conditions.forEach((item, index) => {
        const row = node('div', 'routine-row');
        row.append(field(`Dispositivo de la condición ${index + 1}`, entitySelect(state, item.entity_id, id => update(d => { d.conditions[index].entity_id = id; }), false, ordinals), `condition:${index}:entity`));
        const condition = input('text', item.state, text => update(d => { d.conditions[index].state = text; }));
        const help = node('p', 'routine-field-label', CONDITION_EXAMPLES);
        help.id = `routine-condition-state-help-${index}`;
        condition.setAttribute('aria-describedby', help.id);
        row.append(field(`Estado requerido de la condición ${index + 1} (tal como lo informa Home Assistant)`, condition, `condition:${index}:state`), help);
        row.append(button(`Quitar condición ${index + 1}`, `condition:${index}:remove`, () => update(d => { d.conditions.splice(index, 1); })));
        conditions.append(row);
      });
      conditions.append(button('Agregar condición', 'condition:add', () => update(d => { d.conditions.push({ entity_id: '', state: '' }); })));
      const actions = node('fieldset', 'routine-fieldset'); actions.append(node('legend', 'routine-section', 'Acciones en orden'));
      value.actions.forEach((item, index) => {
        const row = node('div', 'routine-row'), entity = entityByIdFrom(state, item.entity_id);
        const options = Object.entries(OPERATIONS).filter(([key]) => entity?.capabilities?.includes(key));
        if (item.operation && !options.some(([key]) => key === item.operation)) options.unshift([item.operation, (OPERATIONS[item.operation] || 'Acción') + ' (no disponible)']);
        row.append(field(`Dispositivo de la acción ${index + 1}`, entitySelect(state, item.entity_id, id => update(d => { d.actions[index] = { entity_id: id, operation: '' }; }), true, ordinals), `action:${index}:entity`));
        row.append(field(`Operación de la acción ${index + 1}`, select([['', 'Elegí una operación'], ...options], item.operation, operation => update(d => { d.actions[index] = { entity_id: item.entity_id, operation }; })), `action:${index}:operation`));
        for (const descriptor of home.controlFields(entity, item.operation, item)) {
          const change = text => update(d => { d.actions[index][descriptor.name] = text === '' ? null : descriptor.options ? text : Number(text); });
          const control = descriptor.options ? select([['', 'Elegí un modo'], ...descriptor.options], item[descriptor.name] || '', change) : input('number', item[descriptor.name] ?? '', change);
          if (!descriptor.options) { if (Number.isFinite(descriptor.min)) control.min = descriptor.min; if (Number.isFinite(descriptor.max)) control.max = descriptor.max; control.step = descriptor.step || 1; }
          row.append(field(`${descriptor.label} de la acción ${index + 1}`, control, `action:${index}:${descriptor.name}`));
        }
        const controls = node('div', 'row-actions');
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
      if (problems.length) { const list = node('ul'); problems.forEach(text => list.append(node('li', '', routineMessage(text)))); summary.append(list); }
      form.append(summary);
      if (draft.message) form.append(node('p', 'routine-state', draft.message));
      if (draft.error?.conflict) form.append(button('Descartar borrador y recargar', 'reload', () => owner.discardAndReload()));
      if (draft.preview) {
        const panel = node('section', 'detail routine-preview'); panel.setAttribute('aria-label', 'Vista previa');
        panel.append(node('h3', 'row-title', 'Vista previa · sin guardar ni ejecutar'), node('p', 'row-meta', scheduleText(draft.preview)));
        const list = node('ol', 'routine-run-actions');
        for (const item of draft.preview.actions) {
          list.append(node('li', '', entityName(state, item.entity_id, ordinals) + ' · ' + (OPERATIONS[item.operation] || 'Acción desconocida')));
        }
        panel.append(list, node('p', 'routine-footnote', 'La vista previa no consulta el estado de los dispositivos ni evalúa condiciones.')); form.append(panel);
      }
      const footer = node('div', 'row-actions');
      const close = button('Cerrar', 'close', () => owner.closeEditor()); close.disabled = draft.saving;
      const previewButton = button(draft.previewing ? 'Consultando vista previa…' : 'Previsualizar', 'preview', () => owner.preview()); previewButton.disabled = draft.previewing || draft.unknown || !state.catalogUsable;
      const saveButton = node('button', 'btn primary', draft.saving ? 'Guardando…' : 'Guardar'); saveButton.type = 'submit'; saveButton.dataset.key = 'save';
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
        else {
          restoreFocus(dialog, key, selection);
          if (!dialog.contains(doc.activeElement)) dialog.querySelector('[data-key=name]')?.focus({ preventScroll: true });
        }
      }
    }
    function onCancel(event) { event.preventDefault(); owner.closeEditor(); }
    dialog.addEventListener('cancel', onCancel);

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      deviceNames = state.catalog ? home.catalogNames({
        rooms: state.catalog.rooms || [], entities: state.catalog.entities, unassigned_entities: state.catalog.unassigned_entities || [],
      }) : new Map();
      create.hidden = !caps.manage; create.disabled = Boolean(state.draft?.saving) || Boolean(state.writes.new);
      refresh.setAttribute('aria-disabled', String(state.list.loading));
      permission.textContent = caps.manage ? (caps.execute ? '' : 'Tu acceso permite editar rutinas pero no ejecutarlas.') :
        caps.execute ? 'Podés ejecutar rutinas y consultar tus resultados. Para crear o editar rutinas se requiere acceso de administración.' :
        caps.consult ? 'Podés ver las rutinas y consultar tus resultados.' : 'Podés ver las rutinas del hogar.';
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
        recoveryActions.hidden = Boolean(hidden.unknown);
      }
      renderCards(state);
      empty.hidden = !(state.list.loaded && !state.routines.length);
      list.hidden = !state.routines.length && empty.hidden;
      empty.textContent = caps.manage ? 'Todavía no hay rutinas. Creá la primera con dispositivos de tu casa.' : 'Todavía no hay rutinas.';
      const announcement = state.announcement?.serial || 0;
      if (announcement !== lastAnnouncement) { lastAnnouncement = announcement; live.textContent = routineMessage(state.announcement?.text || ''); }
      renderProposals(state);
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
