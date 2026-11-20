/* NEXO lists and reminders, rendered only from the current owner's snapshot. */
(function (root) {
  'use strict';
  const SCOPE_LABELS = { shared: 'Compartida del hogar', personal: 'Personal' };

  function mountTasks({ container, confirmDialog, owner }) {
    const doc = container.ownerDocument;
    let state = owner.getState(), shownData, confirmation = null;

    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    }
    function button(text, callback, className = 'identity-secondary') {
      const control = node('button', className, text);
      control.type = 'button';
      control.addEventListener('click', () => {
        if (control.getAttribute('aria-disabled') !== 'true') callback(control);
      });
      return control;
    }
    function labelledInput(label, type, maxLength) {
      const field = node('label', '', label), input = node('input');
      input.type = type;
      input.required = true;
      if (maxLength) input.maxLength = maxLength;
      field.append(input);
      return { field, input };
    }
    function editor(label, fieldLabel, maxLength, save, { scope, reminder = false } = {}) {
      const wrapper = node('details'), summary = node('summary', '', label), form = node('form', 'identity-form tasks-form');
      const { field, input } = labelledInput(fieldLabel, 'text', maxLength);
      if (scope) form.append(node('p', 'tasks-detail', 'Ámbito · ' + SCOPE_LABELS[scope]));
      form.append(field);
      input.addEventListener('input', () => input.setCustomValidity(''));
      const time = reminder ? labelledInput('Fecha y hora', 'datetime-local') : null;
      if (time) form.append(time.field);
      const submit = node('button', 'identity-primary', 'Guardar');
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
      wrapper.append(summary, form);
      return wrapper;
    }

    const heading = node('div', 'identity-heading');
    const title = node('h2', '', 'Tareas'); title.id = 'tasks-title';
    const refresh = button('Actualizar tareas', () => owner.refresh()); refresh.id = 'tasks-refresh';
    heading.append(title, refresh);
    const status = node('p', 'tasks-status'); status.id = 'tasks-status'; status.setAttribute('role', 'status');
    const feedback = node('p', 'tasks-status'); feedback.id = 'tasks-feedback'; feedback.setAttribute('role', 'status');
    const sections = node('div');
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
      const target = node('p', 'memory-content', targetText); target.dir = 'auto'; target.id = 'tasks-confirm-target';
      const description = node('p', 'modal-description', explanation); description.id = 'tasks-confirm-description';
      const cancel = button('Cancelar', () => closeConfirmation()); cancel.id = 'tasks-confirm-cancel';
      const accept = button('Eliminar', () => {
        closeConfirmation();
        void remove();
      }, 'identity-danger');
      const actions = node('div', 'modal-actions'); actions.append(cancel, accept);
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
      const card = node('article', 'tasks-card');
      card.append(node('h5', 'memory-content', list.name));
      if (!list.items.length) card.append(node('p', 'tasks-detail', 'Lista vacía.'));
      else {
        const items = node('ul');
        for (const item of list.items) {
          const row = node('li');
          row.append(node('p', 'memory-content', item.text));
          if (item.completed) row.append(node('p', 'tasks-detail', 'Completado'));
          if (state.canWrite) {
            const remove = button('Eliminar elemento', opener => {
              ask(opener, 'Eliminar elemento', item.text, 'Se quitará de la lista ' + list.name + '.',
                () => owner.removeItem(scope, list.id, item.id));
            }, 'identity-danger');
            remove.dataset.taskKey = scope + ':item:' + list.id + ':' + item.id;
            remove.setAttribute('aria-label', 'Eliminar elemento ' + item.text);
            row.append(remove);
          }
          items.append(row);
        }
        card.append(items);
      }
      if (state.canWrite) card.append(editor('Añadir elemento', 'Elemento', 500, text => owner.addItem(scope, list.id, text)));
      return card;
    }

    function reminderCard(scope, reminder) {
      const card = node('article', 'tasks-card');
      card.append(node('h5', 'memory-content', reminder.text));
      const date = new Date(reminder.trigger_at * 1000);
      const line = node('p', 'tasks-detail', 'Programado para ');
      if (Number.isFinite(date.getTime())) {
        const time = node('time', '', date.toLocaleString('es-UY', { dateStyle: 'medium', timeStyle: 'short' }));
        time.dateTime = date.toISOString(); line.append(time);
      } else line.textContent = 'Fecha y hora sin confirmar';
      card.append(line);
      if (Array.isArray(reminder.ha_actions) && reminder.ha_actions.length) {
        const block = node('div', 'tasks-actions');
        const actions = node('ul');
        for (const action of reminder.ha_actions) {
          const row = node('li', 'tasks-detail');
          row.append(
            node('div', '', 'Acción · ' + ([action?.domain, action?.service].filter(Boolean).join('/') || 'Sin identificar')),
            node('div', '', 'Dispositivo · ' + (action?.entity_id || 'Sin identificar')),
            node('div', '', 'Habitación · ' + (action?.room || 'Sin identificar'))
          );
          actions.append(row);
        }
        block.append(node('p', 'tasks-detail', 'Acciones del hogar'), actions); card.append(block);
      }
      if (state.canWrite) {
        const remove = button('Eliminar recordatorio', opener => {
          ask(opener, 'Eliminar recordatorio', reminder.text, 'Se cancelará el recordatorio y sus acciones programadas.',
            () => owner.removeReminder(scope, reminder.id));
        }, 'identity-danger');
        remove.dataset.taskKey = scope + ':reminder:' + reminder.id;
        remove.setAttribute('aria-label', 'Eliminar recordatorio ' + reminder.text);
        card.append(remove);
      }
      return card;
    }

    function scopePanel(scope, { lists, reminders }) {
      const panel = node('section', 'panel tasks-scope'); panel.dataset.scope = scope;
      const heading = node('h3', '', scope === 'shared' ? 'Tareas compartidas' : 'Mis tareas');
      heading.id = 'tasks-' + scope + '-title'; panel.setAttribute('aria-labelledby', heading.id);
      const grid = node('div', 'tasks-grid'), listsPanel = node('div'), remindersPanel = node('div');
      listsPanel.append(node('h4', '', 'Listas'));
      remindersPanel.append(node('h4', '', 'Recordatorios'));
      if (state.canWrite) {
        const label = SCOPE_LABELS[scope].toLowerCase();
        listsPanel.append(editor('Nueva lista · ' + label, 'Nombre de la lista', 200,
          name => owner.createList(scope, name), { scope }));
        remindersPanel.append(editor('Nuevo recordatorio · ' + label, 'Texto del recordatorio', 500,
          (text, time) => owner.createReminder(scope, text, time), { scope, reminder: true }));
      }
      if (!lists.length) listsPanel.append(node('p', 'tasks-detail', 'No hay listas.'));
      if (!reminders.length) remindersPanel.append(node('p', 'tasks-detail', 'No hay recordatorios activos.'));
      listsPanel.append(...lists.map(list => listCard(scope, list)));
      remindersPanel.append(...reminders.map(reminder => reminderCard(scope, reminder)));
      grid.append(listsPanel, remindersPanel); panel.append(heading, grid);
      return panel;
    }

    function render() {
      state = owner.getState();
      const { visible, data, busy, writing, error, feedback: message } = state;
      heading.hidden = !visible;
      status.textContent = visible ? error || (writing ? 'Guardando cambio…' : busy ? 'Consultando tareas…' : '') : '';
      feedback.textContent = visible ? message : '';
      refresh.setAttribute('aria-disabled', String(Boolean(busy)));
      if (!visible || data !== shownData) {
        if (confirmation) closeConfirmation(Boolean(visible));
        const focused = doc.activeElement, restore = visible && sections.contains(focused);
        const key = restore && focused.dataset.taskKey;
        shownData = data;
        sections.replaceChildren(...(data ? [scopePanel('shared', data.shared)] : []));
        if (data?.personal) sections.append(scopePanel('personal', data.personal));
        if (restore) {
          const next = key && [...sections.querySelectorAll('[data-task-key]')].find(control => control.dataset.taskKey === key);
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
