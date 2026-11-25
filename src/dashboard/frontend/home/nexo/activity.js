/* Read-only presentation for the three independently authorized activity sources. */
(function (root) {
  'use strict';
  const BLOCKS = {
    action: {
      id: 'activity-actions', title: 'Acciones',
      note: 'Acá aparecen solo las acciones que enviaste desde esta pestaña durante esta sesión. No hay un historial en el servidor.',
      empty: 'Todavía no enviaste acciones desde esta pestaña.',
      loading: 'Consultando acciones…', error: 'No se pudieron consultar las acciones.'
    },
    notice: {
      id: 'activity-notices', title: 'Avisos',
      empty: 'No hay avisos compartidos pendientes de confirmación.',
      loading: 'Consultando avisos…', error: 'No se pudieron consultar los avisos.',
      stale: 'Los avisos pueden estar desactualizados.'
    },
    routine: {
      id: 'activity-routines', title: 'Rutinas',
      note: 'Se muestra la última ejecución disponible de cada rutina.',
      empty: 'No hay ejecuciones de rutinas para mostrar.',
      loading: 'Consultando rutinas…', error: 'No se pudieron consultar las rutinas.'
    }
  };
  const TIMES = { sent_here: 'Envío', scheduled: 'Programación' };
  const dateTime = new Intl.DateTimeFormat('es-UY', { dateStyle: 'short', timeStyle: 'short' });

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function time(at, prefix) {
    if (at === null) return node('span', 'row-meta', 'Sin hora registrada');
    const value = node('time', 'row-meta', prefix + ' · ' + dateTime.format(at));
    value.dateTime = new Date(at).toISOString();
    return value;
  }

  function renderRow(row) {
    const item = node('li', 'row');
    const main = node('div', 'row-main');
    main.append(node('span', 'row-title', [row.what, row.where].filter(Boolean).join(' · ')),
      time(row.at, TIMES[row.atKind]), node('span', 'row-meta', [row.origin, row.detail].filter(Boolean).join(' · ')));
    const tag = node('span', row.status === 'confirmed' ? 'tag on' : 'tag', row.label);
    item.append(main, tag);
    return item;
  }

  function renderBlock({ id, title, note, ...messages }, { rows, status, updatedAt }) {
    const block = node('section', 'detail activity-block');
    const heading = node('h2', '', title);
    heading.id = id;
    block.setAttribute('aria-labelledby', heading.id);
    block.append(heading);
    if (note) block.append(node('p', 'row-meta', note));
    if (updatedAt !== null) block.append(time(updatedAt, 'Última consulta'));
    if (messages[status]) block.append(node('p', 'empty', messages[status]));
    if (rows.length) {
      const list = node('ul', 'rows');
      list.setAttribute('role', 'list');
      list.append(...rows.map(renderRow));
      block.append(list);
    }
    return block;
  }

  function mountActivity({ container, owner }) {
    const content = node('div', 'activity-content');
    const summary = node('div', 'detail activity-summary');
    const head = node('header', 'page-head');
    const title = node('h1', '', 'Actividad'); title.id = 'activity-title';
    const refresh = node('button', 'btn', 'Actualizar'); refresh.type = 'button'; refresh.id = 'activity-refresh';
    refresh.addEventListener('click', () => {
      if (refresh.getAttribute('aria-disabled') !== 'true') void owner.refresh();
    });
    head.append(title, refresh);
    const status = node('p', 'activity-status'); status.setAttribute('role', 'status');
    summary.append(head, status);
    const blocks = node('div', 'activity-blocks');
    content.append(summary, blocks);
    let signature;

    function render() {
      const view = owner.getState();
      if (!view.visible) { container.replaceChildren(); return; }
      if (!content.isConnected) container.replaceChildren(content);
      const allowed = owner.permissions();
      const sources = Object.keys(BLOCKS).filter(key => allowed[key]);
      const busy = view.refreshing || view.status === 'loading';
      refresh.hidden = sources.length === 0;
      refresh.setAttribute('aria-disabled', String(busy));
      const message = sources.length === 0 ? 'No hay actividad disponible para mostrar.'
        : busy ? 'Actualizando actividad…'
          : { ready: 'Consulta actualizada.', partial: 'Se pudo actualizar solo parte de la actividad.',
            error: 'No se pudo actualizar la actividad.' }[view.status];
      if (status.textContent !== message) status.textContent = message;
      const states = sources.map(key => [key, view.blocks[key]]);
      const nextSignature = JSON.stringify(states);
      if (nextSignature !== signature) {
        blocks.replaceChildren(...states.map(([key, state]) => renderBlock(BLOCKS[key],
          state ?? { rows: [], status: 'loading', updatedAt: null })));
        signature = nextSignature;
      }
    }

    const unsubscribe = owner.subscribe(render);
    render();
    return () => { unsubscribe(); container.replaceChildren(); };
  }

  root.NexoActivity = { mountActivity };
})(window);
