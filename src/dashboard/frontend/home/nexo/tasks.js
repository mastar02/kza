/* NEXO lists and reminders, rendered only from the current owner's snapshot. */
(function (root) {
  'use strict';
  const DEVICE_TYPES = {
    light: 'Luz', switch: 'Interruptor', climate: 'Aire acondicionado', cover: 'Persiana', scene: 'Escena'
  };

  function mountTasks({ container, confirmDialog, owner, deviceName = () => null }) {
    const doc = container.ownerDocument;
    const controlKeys = new WeakMap();
    const operationLabels = root.HomeUI.OPERATION_LABELS;
    let state = owner.getState(), shownData, confirmation = null;

    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    }
    function button(text, callback, className = 'btn') {
      const control = node('button', className, text);
      control.type = 'button';
      control.addEventListener('click', () => {
        if (control.getAttribute('aria-disabled') !== 'true') callback(control);
      });
      return control;
    }
    function labelledInput(label, type, maxLength) {
      const field = node('label', 'field'), input = node('input');
      input.type = type;
      input.required = true;
      if (maxLength) input.maxLength = maxLength;
      field.append(node('span', '', label), input);
      return { field, input };
    }
    function taskForm(fieldLabel, maxLength, save, { reminder = false, submitText = 'Guardar' } = {}) {
      const form = node('form', 'addrow tasks-form');
      const { field, input } = labelledInput(fieldLabel, 'text', maxLength);
      form.append(field);
      input.addEventListener('input', () => input.setCustomValidity(''));
      const time = reminder ? labelledInput('Fecha y hora', 'datetime-local') : null;
      if (time) form.append(time.field);
      const submit = node('button', 'btn primary', submitText);
      submit.type = 'submit';
      form.append(submit);
      form.addEventListener('submit', event => {
        event.preventDefault();
        if (state.busy || !state.canWrite) return;
        const value = input.value.trim();
        if (!value) {
          input.setCustomValidity('Escribí un texto antes de guardar.');
          input.reportValidity();
          return;
        }
        void save(value, time ? new Date(time.input.value).getTime() / 1000 : undefined);
      });
      return form;
    }
    function editor(label, fieldLabel, maxLength, save, options) {
      const wrapper = node('details', 'tasks-editor');
      wrapper.append(node('summary', 'btn', label), taskForm(fieldLabel, maxLength, save, options));
      return wrapper;
    }

    const heading = node('header', 'page-head');
    const title = node('h1', '', 'Tareas'); title.id = 'tasks-title';
    const refresh = button('Actualizar tareas', () => owner.refresh()); refresh.id = 'tasks-refresh';
    heading.append(title, refresh);
    const status = node('p', 'tasks-status'); status.id = 'tasks-status'; status.setAttribute('role', 'status');
    const feedback = node('p', 'tasks-status'); feedback.id = 'tasks-feedback'; feedback.setAttribute('role', 'status');
    const sections = node('div', 'tasks-sections');
    container.replaceChildren(heading, status, feedback, sections);

    function closeConfirmation(restoreFocus = true) {
      const opener = confirmation?.opener;
      confirmation = null;
      if (confirmDialog.open) confirmDialog.close();
      confirmDialog.replaceChildren();
      if (restoreFocus && opener) (opener.isConnected ? opener : refresh).focus({ preventScroll: true });
    }
    function ask(opener, titleText, targetText, explanation, remove) {
      if (state.busy || !state.canWrite) return;
      const heading = node('h2', '', titleText); heading.id = 'tasks-confirm-title';
      const target = node('p', 'row-title', targetText); target.dir = 'auto'; target.id = 'tasks-confirm-target';
      const description = node('p', '', explanation); description.id = 'tasks-confirm-description';
      const cancel = button('Cancelar', () => closeConfirmation()); cancel.id = 'tasks-confirm-cancel';
      const accept = button('Eliminar', () => {
        closeConfirmation();
        void remove();
      }, 'btn danger');
      const actions = node('div', 'row-actions'); actions.append(cancel, accept);
      confirmDialog.setAttribute('aria-labelledby', heading.id);
      confirmDialog.setAttribute('aria-describedby', target.id + ' ' + description.id);
      confirmDialog.replaceChildren(heading, target, description, actions);
      confirmation = { opener, cancel, accept };
      confirmDialog.showModal(); cancel.focus();
    }
    function cancelDialog(event) { event.preventDefault(); closeConfirmation(); }
    function keepModalFocus(event) {
      if (event.key !== 'Tab' || !confirmation) return;
      const { cancel, accept } = confirmation;
      const last = event.shiftKey ? cancel : accept;
      if (doc.activeElement !== last) return;
      event.preventDefault();
      (event.shiftKey ? accept : cancel).focus();
    }
    confirmDialog.addEventListener('cancel', cancelDialog);
    confirmDialog.addEventListener('keydown', keepModalFocus);

    function listCard(scope, list) {
      const card = node('article', 'rows tasks-list');
      const name = node('h4', 'row tasks-list-title', list.name); name.dir = 'auto';
      card.append(name);
      if (!list.items.length) card.append(node('p', 'empty', 'Lista vacía.'));
      else {
        const items = node('ul', 'tasks-items');
        items.setAttribute('role', 'list');
        for (const item of list.items) {
          const row = node('li', 'row');
          const text = node('span', 'row-main row-title', item.text); text.dir = 'auto';
          const actions = node('div', 'row-actions');
          actions.append(node('span', 'tag', item.completed ? 'Hecha' : 'Pendiente'));
          row.append(text, actions);
          if (state.canWrite) {
            const remove = button('Eliminar', opener => {
              ask(opener, 'Eliminar elemento', item.text, 'Se quitará de la lista ' + list.name + '.',
                () => owner.removeItem(scope, list.id, item.id));
            }, 'btn quiet');
            remove.dataset.taskKey = 'item';
            controlKeys.set(remove, JSON.stringify([scope, 'item', list.id, item.id]));
            remove.setAttribute('aria-label', 'Eliminar elemento ' + item.text);
            actions.append(remove);
          }
          items.append(row);
        }
        card.append(items);
      }
      if (state.canWrite) card.append(taskForm('Añadir elemento', 500, text => owner.addItem(scope, list.id, text),
        { submitText: 'Añadir' }));
      return card;
    }

    function reminderRow(scope, reminder) {
      const row = node('li', 'row');
      const main = node('div', 'row-main');
      const title = node('span', 'row-title', reminder.text); title.dir = 'auto';
      main.append(title);
      const date = new Date(reminder.trigger_at * 1000);
      const line = node('p', 'row-meta', 'Programado para ');
      if (Number.isFinite(date.getTime())) {
        const time = node('time', 'time', date.toLocaleString('es-UY', { dateStyle: 'medium', timeStyle: 'short' }));
        time.dateTime = date.toISOString(); line.append(time);
      } else line.textContent = 'Fecha y hora sin confirmar';
      main.append(line);
      if (Array.isArray(reminder.ha_actions) && reminder.ha_actions.length) {
        const block = node('div', 'tasks-actions row-meta');
        const actions = node('ul');
        for (const action of reminder.ha_actions) {
          const { domain, service, entity_id: id, room } = action ?? {};
          const operation = domain === 'scene' && service === 'turn_on' ? 'Activar escena' :
            Object.hasOwn(operationLabels, service) ? operationLabels[service] : 'Acción programada';
          const device = (id && deviceName(id)) ||
            (Object.hasOwn(DEVICE_TYPES, domain) ? DEVICE_TYPES[domain] : 'Dispositivo');
          const place = ' · ' + (room || 'Ambiente no informado');
          const target = device.endsWith(place) ? device : device + place;
          actions.append(node('li', '', operation + ' · ' + target));
        }
        block.append(node('p', '', 'Acciones del hogar'), actions); main.append(block);
      }
      row.append(main);
      if (state.canWrite) {
        const remove = button('Eliminar', opener => {
          ask(opener, 'Eliminar recordatorio', reminder.text, 'Se cancelará el recordatorio y sus acciones programadas.',
            () => owner.removeReminder(scope, reminder.id));
        }, 'btn quiet');
        remove.dataset.taskKey = 'reminder';
        controlKeys.set(remove, JSON.stringify([scope, 'reminder', reminder.id]));
        remove.setAttribute('aria-label', 'Eliminar recordatorio ' + reminder.text);
        const actions = node('div', 'row-actions'); actions.append(remove); row.append(actions);
      }
      return row;
    }

    function scopePanel(scope, { lists, reminders }) {
      const shared = scope === 'shared';
      const panel = node('section', 'tasks-scope'); panel.dataset.scope = scope;
      const heading = node('h2', '', shared ? 'Tareas del hogar' : 'Mis tareas');
      heading.id = 'tasks-' + scope + '-title'; panel.setAttribute('aria-labelledby', heading.id);
      const header = node('header', 'tasks-scope-heading');
      header.append(heading, node('span', 'tag', shared ? 'Compartidas' : 'Personal'));
      const grid = node('div', 'tasks-grid');
      const listsPanel = node('div', 'tasks-group'), remindersPanel = node('div', 'tasks-group');
      listsPanel.append(node('h3', '', 'Listas'));
      remindersPanel.append(node('h3', '', 'Recordatorios'));
      if (state.canWrite) {
        listsPanel.append(editor(shared ? 'Nueva lista compartida' : 'Nueva lista personal', 'Nombre de la lista', 200,
          name => owner.createList(scope, name)));
        remindersPanel.append(editor(shared ? 'Nuevo recordatorio compartido' : 'Nuevo recordatorio personal', 'Texto del recordatorio', 500,
          (text, time) => owner.createReminder(scope, text, time), { reminder: true }));
      }
      if (!lists.length) {
        const empty = node('div', 'rows');
        empty.append(node('p', 'empty', 'No hay listas.'));
        listsPanel.append(empty);
      }
      listsPanel.append(...lists.map(list => listCard(scope, list)));
      if (reminders.length) {
        const rows = node('ul', 'rows tasks-items');
        rows.setAttribute('role', 'list');
        rows.append(...reminders.map(reminder => reminderRow(scope, reminder)));
        remindersPanel.append(rows);
      } else {
        const empty = node('div', 'rows');
        empty.append(node('p', 'empty', 'No hay recordatorios activos.'));
        remindersPanel.append(empty);
      }
      grid.append(listsPanel, remindersPanel); panel.append(header, grid);
      return panel;
    }

    function render() {
      state = owner.getState();
      const { visible, data, busy, writing, error, feedback: message } = state;
      heading.hidden = !visible;
      status.textContent = visible ? error || (writing ? 'Guardando cambio…' : busy ? 'Consultando tareas…' : '') : '';
      status.dataset.kind = error ? 'error' : '';
      feedback.textContent = visible ? message : '';
      refresh.setAttribute('aria-disabled', String(Boolean(busy)));
      if (!visible || data !== shownData) {
        if (confirmation) closeConfirmation(Boolean(visible));
        const focused = doc.activeElement, restore = visible && sections.contains(focused);
        const key = restore && controlKeys.get(focused);
        shownData = data;
        sections.replaceChildren(...(data ? [scopePanel('shared', data.shared)] : []));
        if (data?.personal) sections.append(scopePanel('personal', data.personal));
        if (restore) {
          const next = key && [...sections.querySelectorAll('[data-task-key]')].find(control => controlKeys.get(control) === key);
          (next || refresh).focus({ preventScroll: Boolean(next) });
        }
      }
      for (const control of sections.querySelectorAll('button')) control.setAttribute('aria-disabled', String(Boolean(busy)));
    }
    const unsubscribe = owner.subscribe(render);
    render();
    return () => {
      unsubscribe(); closeConfirmation(false); container.replaceChildren();
      confirmDialog.removeEventListener('cancel', cancelDialog);
      confirmDialog.removeEventListener('keydown', keepModalFocus);
    };
  }

  root.NexoTasks = { mountTasks };
})(typeof window !== 'undefined' ? window : globalThis);
