/* Canonical account metadata. Voices and dashboard grants have separate owners. */
(function (root) {
  'use strict';
  const MEMORY_REMOVAL = 'También se eliminan, sin posibilidad de recuperarlos, los recuerdos asignados a esta persona en Memoria, junto con los recuerdos que dependen de ellos.';
  const LEVELS = { guest: 'Invitado', child: 'Niño/a', teen: 'Adolescente', adult: 'Adulto/a', admin: 'Administrador/a' };
  function normalizeName(name) {
    return name.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase().replace(/\s+/g, ' ').trim();
  }
  function parseAccounts(value) {
    const ids = new Set();
    if (!Array.isArray(value)) throw new Error('La lista de personas no está disponible.');
    return value.map(row => {
      if (!row || Object.keys(row).sort().join(',') !== 'id,is_active,name,permission_level' ||
          typeof row.id !== 'string' || !row.id || typeof row.name !== 'string' ||
          typeof row.permission_level !== 'string' || !Object.hasOwn(LEVELS, row.permission_level) ||
          typeof row.is_active !== 'boolean' || ids.has(row.id)) {
        throw new Error('La lista de personas no está disponible.');
      }
      ids.add(row.id);
      return { id: row.id, name: row.name, permission_level: row.permission_level, is_active: row.is_active };
    });
  }
  function mutationError(error) {
    if (error?.detail?.credentials_retired === true) return 'Se confirmó el retiro de credenciales, pero no la baja de la cuenta. Actualizá la lista antes de reintentar.';
    if (error?.detail?.credentials_retired === null) return 'No se pudo confirmar el retiro de credenciales. La operación puede seguir en curso. Actualizá la lista antes de volver a enviarla.';
    if (error?.status === 409) return error.detail === 'No puedes eliminar al único administrador' ?
      'No se puede retirar al último administrador activo.' : 'No se aceptó el cambio. Revisá los datos y actualizá la lista.';
    if (error?.status === 404) return 'La persona ya no está disponible. Actualizá la lista.';
    if (error?.status === 403) return 'Tu sesión no permite este cambio.';
    if (error?.status === 422) return 'Revisá el nombre y el nivel de cuenta.';
    return 'No se confirmó el resultado. Puede haberse aplicado. Actualizá la lista antes de repetirlo.';
  }
  function createPeople({ container, dialog, api, session, voiceDetails, accessDetails }) {
    const doc = container.ownerDocument;
    let lifecycle = 0, readVersion = 0, generation = -1, visible = false, busy = false, known = false;
    let rows = [], selected = null, readController, mutationController, grid, detail, status, refresh, add;
    let notice = '', noticeError = false, detailDisposers = [], detailSerial = 0;
    const cards = new Map();
    function retireDetails() {
      for (const dispose of detailDisposers.splice(0)) dispose();
    }
    function mountDetail(personId, host, mount, allowed) {
      const started = lifecycle, login = generation, version = readVersion, serial = detailSerial;
      const stillCurrent = () => current(started, login) && allowed() && visible && known && !busy
        && version === readVersion && serial === detailSerial && selected === personId
        && host.isConnected && detail?.contains(host);
      if (stillCurrent()) detailDisposers.push(mount({ personId, container: host, current: stillCurrent }));
    }
    function node(tag, classes, text) {
      const element = doc.createElement(tag);
      if (classes) element.className = classes;
      if (text !== undefined) element.textContent = text;
      return element;
    }
    function control(text, handler, classes = 'btn') {
      const element = node('button', classes, text); element.type = 'button'; element.addEventListener('click', handler); return element;
    }
    function current(start = lifecycle, login = generation) {
      return start === lifecycle && login === generation && login === session.generation() && session.can('admin:manage');
    }
    function closeDialog() {
      for (const input of dialog.querySelectorAll('input,select,textarea')) input.value = '';
      if (dialog.open) dialog.close();
      dialog.replaceChildren();
    }
    dialog.addEventListener('cancel', event => { event.preventDefault(); closeDialog(); });
    function report(text, error = false) {
      notice = text; noticeError = error;
      if (visible && status) { status.textContent = text; status.classList.toggle('error-message', error); }
      const modalStatus = dialog.querySelector('.person-dialog-status');
      if (modalStatus) { modalStatus.textContent = text; modalStatus.classList.toggle('error-message', error); }
    }
    function updateBusy() {
      if (!visible) return;
      add.disabled = busy || !known; refresh.disabled = busy;
      for (const button of detail.querySelectorAll('[data-person-admin]')) button.disabled = busy || !known;
      for (const button of dialog.querySelectorAll('button[type=submit],#confirm-person-removal')) button.disabled = busy || !known;
    }
    function renderDetail() {
      retireDetails(); detailSerial++; detail.replaceChildren();
      const person = rows.find(row => row.id === selected);
      detail.hidden = !person;
      if (!person) return;
      const header = node('div', 'personas-heading');
      header.append(node('h2', '', person.name), node('span', 'tag', person.is_active ? 'Cuenta activa' : 'Cuenta inactiva'));
      detail.append(header);
      const level = node('div', 'person-role-info row');
      const account = node('div', 'row-main');
      account.append(node('h3', 'row-title', 'Nivel de cuenta'), node('span', 'row-meta', LEVELS[person.permission_level]));
      const change = control('Cambiar nivel', () => openForm(person), 'btn identity-text-link');
      change.dataset.personAdmin = ''; level.append(account, change); detail.append(level);
      const canSeeVoices = () => session.can('voices:read') && session.can('voices:review');
      if (voiceDetails && canSeeVoices()) {
        const host = node('section', 'person-voice-details personas-section');
        host.setAttribute('aria-label', 'Perfiles de la persona');
        detail.append(host);
        if (person.is_active) mountDetail(person.id, host, voiceDetails, canSeeVoices);
        else host.append(node('p', 'empty', 'La consulta de voz requiere una cuenta activa.'));
      }
      const canSeeAccess = () => session.role() === 'operator' && session.can('admin:manage') && session.can('auth:manage');
      if (accessDetails && canSeeAccess()) {
        const host = node('section', 'personas-section');
        host.setAttribute('aria-label', 'Acceso al panel de KZA');
        host.dataset.personAccess = '';
        detail.append(host);
        const name = normalizeName(person.name);
        const ambiguous = rows.some(other => other.id !== person.id && normalizeName(other.name) === name);
        mountDetail(person.id, host,
          context => accessDetails({ ...context, personName: person.name, ambiguous }), canSeeAccess);
      }
      if (session.can('auth:manage')) {
        const actions = node('div', 'row-actions');
        const remove = control('Retirar persona', () => openRemoval(person), 'btn danger identity-danger');
        remove.dataset.personAdmin = ''; actions.append(remove); detail.append(actions);
      }
      updateBusy();
    }
    function select(id) {
      retireDetails(); selected = id;
      for (const [personId, card] of cards) {
        card.setAttribute('aria-pressed', String(personId === id));
      }
      renderDetail();
    }
    function renderRows() {
      if (!visible) return;
      retireDetails(); cards.clear(); grid.replaceChildren(); grid.hidden = !rows.length && !known;
      for (const person of rows) {
        const card = control('', () => select(person.id), 'pick'); card.dataset.person = '';
        cards.set(person.id, card);
        card.setAttribute('aria-pressed', String(person.id === selected));
        card.append(node('span', 'row-title', person.name),
          node('span', 'row-meta', `${LEVELS[person.permission_level]} · ${person.is_active ? 'Cuenta activa' : 'Cuenta inactiva'}`));
        grid.append(card);
      }
      if (!rows.length && known) grid.append(node('p', 'empty', 'Todavía no hay personas registradas.'));
      renderDetail(); updateBusy();
    }
    async function load({ confirmed = '', focusSelected = false } = {}) {
      if (!visible || !current() || busy) return;
      retireDetails();
      const started = lifecycle, login = generation, version = ++readVersion;
      readController?.abort(); readController = new AbortController(); known = false;
      const priorSelection = selected; rows = []; renderRows(); report(confirmed ? `${confirmed} Actualizando la lista…` : 'Consultando personas…');
      try {
        const value = await api('/api/admin/users', { signal: readController.signal });
        if (!current(started, login) || !visible || version !== readVersion) return;
        rows = parseAccounts(value); known = true;
        selected = rows.some(row => row.id === priorSelection) ? priorSelection : rows[0]?.id || null;
        report(confirmed || `${rows.length} ${rows.length === 1 ? 'persona registrada' : 'personas registradas'}.`);
        renderRows();
        if (focusSelected) cards.get(selected)?.focus();
      } catch (error) {
        if (!current(started, login) || !visible || version !== readVersion) return;
        rows = []; selected = null; renderRows();
        report(confirmed ? `${confirmed} No pudimos actualizar la lista. Usá Actualizar personas.` : 'La lista de personas no está disponible. Usá Actualizar personas.', true);
      } finally {
        if (current(started, login) && version === readVersion) { readController = null; updateBusy(); }
      }
    }
    function shell() {
      retireDetails(); container.replaceChildren();
      const heading = node('header', 'page-head'), copy = node('div');
      const title = node('h1', '', 'Personas'); title.id = 'people-title';
      copy.append(title);
      add = control('Nueva persona', () => openForm(), 'btn primary'); add.id = 'add-person';
      heading.append(copy, add); container.append(heading);
      const toolbar = node('div', 'personas-toolbar'); status = node('p', ''); status.id = 'people-status'; status.setAttribute('role', 'status');
      refresh = control('Actualizar personas', () => load()); refresh.id = 'refresh-people'; toolbar.append(status, refresh); container.append(toolbar);
      const workspace = node('div', 'personas-columns'); grid = node('section', 'rows'); grid.id = 'people-grid'; grid.setAttribute('aria-label', 'Personas registradas');
      detail = node('aside', 'detail'); detail.id = 'person-detail'; detail.setAttribute('aria-label', 'Persona seleccionada');
      workspace.append(grid, detail); container.append(workspace);
      renderRows();
      report(busy ? 'Operación enviada. Esperando el resultado…' : notice, noticeError); updateBusy();
    }
    function modal(title, description) {
      closeDialog();
      const top = node('div', 'personas-toolbar'); top.append(control('Cerrar', closeDialog, 'btn quiet icon-button'));
      top.lastChild.setAttribute('aria-label', 'Cerrar diálogo');
      const heading = node('h2', '', title); heading.id = 'person-dialog-title';
      const descriptionNode = node('p', 'modal-description', description); descriptionNode.id = 'person-dialog-description';
      dialog.append(top, heading, descriptionNode);
      const modalStatus = node('p', 'person-dialog-status'); modalStatus.setAttribute('role', 'status'); dialog.append(modalStatus);
      dialog.setAttribute('aria-labelledby', heading.id); dialog.setAttribute('aria-describedby', descriptionNode.id);
    }
    async function mutate(path, method, body, confirmation) {
      if (!current() || busy || !known) return;
      retireDetails();
      const started = lifecycle, login = generation;
      busy = true; mutationController = new AbortController(); readController?.abort(); readVersion++;
      closeDialog(); updateBusy(); report('Operación enviada. Esperando el resultado…');
      try {
        const receipt = await api(path, { method, ...(body === undefined ? {} : { body }), signal: mutationController.signal });
        body = null;
        if (!current(started, login)) return;
        if (receipt?.ok !== true) throw new Error('unverified_account_receipt');
        if (method === 'POST' && typeof receipt.user_id === 'string') selected = receipt.user_id;
        busy = false; report(confirmation);
        if (visible) await load({ confirmed: confirmation, focusSelected: true });
      } catch (error) {
        if (!current(started, login)) return;
        busy = false; known = false; rows = []; selected = null;
        if (visible) renderRows();
        report(mutationError(error), true);
      } finally {
        body = null;
        if (current(started, login)) { mutationController = null; busy = false; updateBusy(); }
      }
    }
    function openForm(person = null) {
      if (!current() || busy || !known || !visible) return;
      modal(person ? `El nivel de ${person.name}` : 'Crear persona', person ?
        'Cambiá el nivel de esta cuenta. El acceso al panel de KZA se administra por separado.' : 'Creá su cuenta en KZA. Las voces y las credenciales se administran por separado.');
      const form = node('form', 'personas-form'); form.id = 'person-form';
      if (!person) {
        const label = node('label', 'field'); label.append(node('span', '', 'Nombre')); const input = node('input'); input.name = 'personName'; input.required = true; input.maxLength = 64; input.autocomplete = 'off'; label.append(input); form.append(label);
      }
      const label = node('label', 'field'); label.append(node('span', '', 'Nivel de cuenta')); const level = node('select'); level.name = 'permission_level';
      for (const [value, text] of Object.entries(LEVELS)) { const option = node('option', '', text); option.value = value; level.append(option); }
      level.value = person?.permission_level || 'adult'; label.append(level); form.append(label);
      const submit = node('button', 'btn primary', person ? 'Guardar nivel' : 'Crear persona'); submit.type = 'submit'; form.append(submit);
      form.addEventListener('submit', event => {
        event.preventDefault(); if (!current() || busy) return;
        const permission_level = level.value;
        if (!Object.hasOwn(LEVELS, permission_level)) return;
        if (person) void mutate(`/api/admin/users/${encodeURIComponent(person.id)}/permissions`, 'PUT', { permission_level }, 'Nivel de cuenta actualizado.');
        else {
          const name = form.elements.personName.value.trim();
          if (!name) { form.elements.personName.setCustomValidity('Ingresá un nombre.'); form.elements.personName.reportValidity(); return; }
          void mutate('/api/admin/users', 'POST', { name, permission_level }, 'Persona creada.');
        }
      });
      if (!person) form.elements.personName.addEventListener('input', () => form.elements.personName.setCustomValidity(''));
      dialog.append(form); dialog.showModal();
    }
    function openRemoval(person) {
      if (!current() || busy || !known || !visible || !session.can('auth:manage')) return;
      // U3b-1b/D7: retiring an account deletes the memories assigned to that person (u3b-1b follow-up).
      modal(`Retirar a ${person.name}`, `Se retirarán sus credenciales y vínculos vocales antes de borrar la cuenta. Los perfiles de voz se conservan sin persona asignada. ${MEMORY_REMOVAL}`);
      const actions = node('div', 'row-actions modal-actions');
      actions.append(control('Conservar persona', closeDialog, 'btn identity-secondary'), control('Retirar persona', () => {
        void mutate(`/api/admin/users/${encodeURIComponent(person.id)}`, 'DELETE', undefined, 'Persona retirada.');
      }, 'btn danger'));
      actions.lastChild.id = 'confirm-person-removal'; dialog.append(actions); dialog.showModal();
      actions.firstChild.focus(); // The safe choice ("Conservar persona") takes the initial focus, never the destructive one.
    }
    function leave() {
      retireDetails(); visible = false; readVersion++; readController?.abort(); readController = null;
      rows = []; cards.clear(); selected = null; known = false; closeDialog(); container.replaceChildren();
      grid = detail = status = refresh = add = null;
    }
    function clear() {
      lifecycle++; mutationController?.abort(); mutationController = null; busy = false; notice = ''; noticeError = false; generation = -1; leave();
    }
    function open() {
      if (!session.can('admin:manage')) { clear(); return; }
      if (generation !== session.generation()) { clear(); generation = session.generation(); }
      if (visible) return;
      visible = true; shell();
      if (!busy) void load();
    }
    return { open, leave, clear, refresh: () => load() };
  }
  const api = { createPeople, parseAccounts, mutationError, MEMORY_REMOVAL };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.NexoPeople = api;
})(typeof window !== 'undefined' ? window : globalThis);
