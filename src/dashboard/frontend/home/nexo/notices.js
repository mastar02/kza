/* NEXO Home presentation for shared reminder notices. */
(function (root) {
  'use strict';
  const triggerTime = new Intl.DateTimeFormat('es-UY', {
    timeZone: 'America/Montevideo', day: 'numeric', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
  });

  function mountNotices({ container, owner }) {
    const document = container.ownerDocument;
    function node(tag, className, text) {
      const element = document.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined) element.textContent = text;
      return element;
    }
    function setText(element, text) { if (element.textContent !== text) element.textContent = text; }
    function button(text, callback, className = 'btn') {
      const control = node('button', className, text);
      control.type = 'button'; control.addEventListener('click', callback); return control;
    }

    const heading = node('h2', '', 'Avisos compartidos'); heading.id = 'notices-heading';
    const refresh = button('Actualizar avisos', () => { void owner.refresh(); });
    const header = node('header', 'page-head'); header.append(heading, refresh);
    const status = node('p', 'notice-status'); status.setAttribute('role', 'status');
    const feedback = node('p', 'notice-feedback'); feedback.setAttribute('role', 'status');
    const list = node('ul', 'rows notices-list'); list.setAttribute('role', 'list');
    const rows = new Map();
    container.classList.add('detail');
    container.setAttribute('aria-labelledby', heading.id);

    function updateRow(entry, state) {
      let row = rows.get(entry.key);
      if (!row) {
        const item = node('li', 'row notice-row'), main = node('div', 'row-main');
        const title = node('span', 'row-title'); title.dir = 'auto';
        const meta = node('div', 'row-meta notices-meta'), time = node('time', 'time');
        const label = node('span', 'tag notice-label'), actions = node('div', 'row-actions');
        meta.append(time, node('span', '', 'Ambiente no informado'));
        main.append(title, meta, label); item.append(main);
        const dismiss = button('Descartar', () => { void owner.dismiss(entry.key); }, 'btn quiet');
        dismiss.dataset.noticeDismiss = '';
        actions.append(dismiss);
        row = { item, title, time, label, actions, dismiss };
        rows.set(entry.key, row);
      }
      const time = new Date(entry.trigger_at * 1000), when = triggerTime.format(time);
      const title = entry.text || 'Aviso sin texto disponible';
      row.item.dataset.noticeStatus = entry.status;
      setText(row.title, title); setText(row.time, when); setText(row.label, entry.label);
      row.time.dateTime = time.toISOString();
      if (state.canWrite && entry.status !== 'sending') {
        row.dismiss.setAttribute('aria-label', `Descartar aviso: ${title} (${when})`);
        row.dismiss.setAttribute('aria-disabled', String(state.locked));
        if (!row.actions.parentNode) row.item.append(row.actions);
      } else row.actions.remove();
      return row.item;
    }

    function render() {
      const state = owner.getState();
      container.hidden = !state.visible;
      if (!state.visible) { container.replaceChildren(); list.replaceChildren(); rows.clear(); return; }
      if (!header.parentNode) container.append(header, status, list, feedback);
      container.dataset.noticeState = state.status;
      const messages = {
        loading: 'Consultando avisos…',
        empty: 'No hay avisos compartidos pendientes ni con entrega incierta.',
        stale: ['La lista de avisos está desactualizada.', state.error].filter(Boolean).join(' '),
        error: state.error,
        ready: ''
      };
      setText(status, messages[state.status]);
      status.classList.toggle('empty', state.status === 'empty');
      list.hidden = !state.notices.length;
      setText(feedback, state.feedback?.text || '');
      feedback.dataset.kind = state.feedback?.kind || '';
      const focus = document.activeElement, hadFocus = list.contains(focus);
      const keys = new Set(state.notices.map(entry => entry.key));
      for (const [key, row] of rows) {
        if (!keys.has(key)) { row.item.remove(); rows.delete(key); }
      }
      let position = list.firstElementChild;
      for (const entry of state.notices) {
        const element = updateRow(entry, state);
        if (element !== position) list.insertBefore(element, position);
        position = element.nextElementSibling;
      }
      if (hadFocus && document.activeElement !== focus) {
        // Do not send focus to the page body when an observed row disappears.
        if (list.contains(focus)) focus.focus({ preventScroll: true });
        else refresh.focus();
      }
    }

    const unsubscribe = owner.subscribe(render);
    render();
    return () => { unsubscribe(); container.replaceChildren(); };
  }

  root.NexoNotices = { mountNotices };
})(window);
