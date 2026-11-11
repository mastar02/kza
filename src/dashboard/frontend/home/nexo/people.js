/* Canonical account metadata. Voices and dashboard grants have separate owners. */
(function (root) {
  'use strict';
  const MEMORY_REMOVAL = 'También se eliminan, sin posibilidad de recuperarlos, los recuerdos asignados a esta persona en Memoria, junto con los recuerdos que dependen de ellos.';
  const LEVELS = { guest: 'Invitado', child: 'Niño/a', teen: 'Adolescente', adult: 'Adulto/a', admin: 'Administrador/a' };
  function parseAccounts(value) {
    const ids = new Set();
    if (!Array.isArray(value)) throw new Error('El catálogo de personas no está disponible.');
    return value.map(row => {
      if (!row || Object.keys(row).sort().join(',') !== 'id,is_active,name,permission_level' ||
          typeof row.id !== 'string' || !row.id || typeof row.name !== 'string' ||
          typeof row.permission_level !== 'string' || !Object.hasOwn(LEVELS, row.permission_level) ||
          typeof row.is_active !== 'boolean' || ids.has(row.id)) {
        throw new Error('El catálogo de personas no está disponible.');
      }
      ids.add(row.id);
      return { id: row.id, name: row.name, permission_level: row.permission_level, is_active: row.is_active };
    });
  }
  function mutationError(error) {
    if (error?.detail?.credentials_retired === true) return 'Se confirmó el retiro de credenciales, pero no la baja del catálogo. Actualizá la lista antes de reintentar.';
    if (error?.detail?.credentials_retired === null) return 'No se pudo confirmar el retiro de credenciales. La operación puede seguir en curso. Actualizá la lista antes de volver a enviarla.';
    if (error?.status === 409) return error.detail === 'No puedes eliminar al único administrador' ?
      'No se puede retirar al último administrador activo.' : 'El backend rechazó el cambio. Revisá los datos y actualizá el catálogo.';
    if (error?.status === 404) return 'La persona ya no está disponible. Actualizá el catálogo.';
    if (error?.status === 403) return 'Tu sesión no permite este cambio.';
    if (error?.status === 422) return 'Revisá el nombre y el nivel de cuenta.';
    return 'No se confirmó el resultado. Puede haberse aplicado. Actualizá el catálogo antes de repetirlo.';
  }
  function createPeople({ container, dialog, api, session, icon, voiceDetails = () => () => {} }) {
    const doc = container.ownerDocument;
    let lifecycle = 0, readVersion = 0, generation = -1, visible = false, busy = false, known = false;
    let rows = [], selected = null, readController, mutationController, grid, detail, status, refresh, add;
    let notice = '', noticeError = false, voiceRetire = null, detailSerial = 0;
    function retireVoices() { const stop = voiceRetire; voiceRetire = null; if (stop) stop(); }
    function mountVoices(personId,host) {
      const started=lifecycle,login=generation,version=readVersion,serial=detailSerial;
      voiceRetire=voiceDetails({personId,container:host,current:()=>current(started,login) && visible && known && !busy
        && version===readVersion && serial===detailSerial && selected===personId && host.isConnected && detail?.contains(host)});
    }
    function node(tag, classes, text) {
      const element = doc.createElement(tag);
      if (classes) element.className = classes;
      if (text !== undefined) element.textContent = text;
      return element;
    }
    function control(text, handler, classes = 'identity-secondary') {
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
    function avatar(person) {
      const result = node('span', 'person-avatar', Array.from(person.name.trim())[0]?.toLocaleUpperCase() || '·');
      result.setAttribute('aria-hidden', 'true'); return result;
    }
    function updateBusy() {
      if (!visible) return;
      add.disabled = busy || !known; refresh.disabled = busy;
      for (const button of detail.querySelectorAll('[data-person-admin]')) button.disabled = busy || !known;
      for (const button of dialog.querySelectorAll('button[type=submit],#confirm-person-removal')) button.disabled = busy || !known;
    }
    function renderDetail() {
      retireVoices(); detailSerial++; detail.replaceChildren();
      const person = rows.find(row => row.id === selected);
      if (!person) { detail.append(node('p', 'identity-empty', 'Elegí una persona para ver su cuenta.')); return; }
      const header = node('div', 'person-detail-header');
      const title = node('div'); title.append(node('h3', '', person.name), node('span', '', person.is_active ? 'Cuenta activa' : 'Cuenta inactiva'));
      header.append(avatar(person), title); detail.append(header);
      const level = node('div', 'person-role-info');
      level.append(node('p', 'eyebrow', 'NIVEL DE CUENTA'), node('p', '', LEVELS[person.permission_level]));
      const change = control('Cambiar nivel', () => openForm(person), 'identity-text-link');
      change.dataset.personAdmin='';level.append(change); detail.append(level);
      detail.append(node('p', 'person-authority-note', 'El nivel de cuenta se administra aparte del acceso al dashboard y de los perfiles de voz.'));
      if (session.can('auth:manage')) { const remove=control('Retirar persona', () => openRemoval(person), 'identity-danger full');remove.dataset.personAdmin='';detail.append(remove); }
      const voiceHost=node('section','person-voice-details');voiceHost.setAttribute('aria-label','Perfiles de la persona');detail.append(voiceHost);
      if(person.is_active)mountVoices(person.id,voiceHost);else voiceHost.append(node('p','person-voice-status','La cuenta inactiva no tiene una consulta vocal disponible.'));
      updateBusy();
    }
    function select(id) {
      retireVoices(); selected = id;
      for (const card of grid.querySelectorAll('[data-person]')) {
        const active = card.dataset.person === id;
        card.classList.toggle('selected', active); card.setAttribute('aria-pressed', String(active));
      }
      renderDetail();
    }
    function renderRows() {
      if (!visible) return;
      retireVoices(); grid.replaceChildren();
      for (const person of rows) {
        const card = control('', () => select(person.id), 'person-card panel'); card.dataset.person = person.id;
        card.setAttribute('aria-pressed', String(person.id === selected)); card.classList.toggle('selected', person.id === selected);
        const top = node('span', 'person-card-top'); top.append(avatar(person), node('span', 'identity-badge neutral', person.is_active ? 'Activa' : 'Inactiva'));
        card.append(top, node('h3', '', person.name), node('span', 'identity-badge neutral', LEVELS[person.permission_level]),
                    node('span', 'person-card-footer', 'Ver persona →')); grid.append(card);
      }
      if (!rows.length) grid.append(node('p', 'identity-empty panel', known ? 'Todavía no hay personas registradas.' : 'El catálogo de personas no está disponible.'));
      renderDetail(); updateBusy();
    }
    async function load({ confirmed = '', focusSelected = false } = {}) {
      if (!visible || !current() || busy) return;
      retireVoices();
      const started = lifecycle, login = generation, version = ++readVersion;
      readController?.abort(); readController = new AbortController(); known = false;
      const priorSelection = selected; rows = []; renderRows(); report(confirmed ? `${confirmed} Actualizando el catálogo…` : 'Consultando personas…');
      try {
        const value = await api('/api/admin/users', { signal: readController.signal });
        if (!current(started, login) || !visible || version !== readVersion) return;
        rows = parseAccounts(value); known = true;
        selected = rows.some(row => row.id === priorSelection) ? priorSelection : rows[0]?.id || null;
        report(confirmed || `${rows.length} ${rows.length === 1 ? 'persona registrada' : 'personas registradas'}.`);
        renderRows();
        if (focusSelected) [...grid.querySelectorAll('[data-person]')].find(card => card.dataset.person === selected)?.focus();
      } catch (error) {
        if (!current(started, login) || !visible || version !== readVersion) return;
        rows = []; selected = null; renderRows();
        report(confirmed ? `${confirmed} No pudimos actualizar el catálogo. Usá Actualizar personas.` : 'El catálogo de personas no está disponible. Usá Actualizar personas.', true);
      } finally {
        if (current(started, login) && version === readVersion) { readController = null; updateBusy(); }
      }
    }
    function shell() {
      retireVoices(); container.replaceChildren();
      const heading = node('div', 'identity-heading'), copy = node('div');
      const title = node('h2', '', 'Tu hogar tiene su gente.'); title.id = 'people-title';
      copy.append(node('p', 'eyebrow', 'CADA PERSONA, SU ESPACIO'), title, node('p', '', 'Personas y niveles de cuenta, desde el catálogo de KZA.'));
      add = control('Nueva persona', () => openForm(), 'identity-primary'); add.id = 'add-person'; add.prepend(icon('plus'));
      heading.append(copy, add); container.append(heading);
      const toolbar = node('div', 'people-toolbar'); status = node('p', ''); status.id = 'people-status'; status.setAttribute('role', 'status');
      refresh = control('Actualizar personas', () => load()); refresh.id = 'refresh-people'; toolbar.append(status, refresh); container.append(toolbar);
      const workspace = node('div', 'people-workspace'); grid = node('section', 'people-grid'); grid.id = 'people-grid'; grid.setAttribute('aria-label', 'Personas registradas');
      detail = node('aside', 'person-detail panel'); detail.id = 'person-detail'; detail.setAttribute('aria-label', 'Persona seleccionada');
      workspace.append(grid, detail); container.append(workspace);
      const note = node('p', 'identity-contract-note', 'Crear una persona o cambiar su nivel de cuenta no crea una credencial de acceso ni vincula un perfil de voz.');
      note.prepend(icon('shield')); container.append(note); renderRows();
      report(busy ? 'Operación enviada. Esperando el resultado…' : notice, noticeError); updateBusy();
    }
    function modal(title, description) {
      closeDialog();
      const top = node('div', 'dialog-top'); top.append(node('p', 'eyebrow', 'KZA / PERSONAS'), control('×', closeDialog, 'icon-button'));
      top.lastChild.setAttribute('aria-label', 'Cerrar diálogo');
      const heading = node('h2', '', title); heading.id = 'person-dialog-title';
      const descriptionNode = node('p', 'modal-description', description); descriptionNode.id = 'person-dialog-description';
      dialog.append(top, heading, descriptionNode);
      const modalStatus = node('p', 'person-dialog-status'); modalStatus.setAttribute('role', 'status'); dialog.append(modalStatus);
      dialog.setAttribute('aria-labelledby', heading.id); dialog.setAttribute('aria-describedby', descriptionNode.id);
    }
    async function mutate(path, method, body, confirmation) {
      if (!current() || busy || !known) return;
      retireVoices();
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
      modal(person ? `El nivel de ${person.name}` : 'Una persona más en casa.', person ?
        'Cambiá el nivel de esta cuenta. El acceso al dashboard se administra por separado.' : 'Creá su cuenta en KZA. Las voces y las credenciales se administran por separado.');
      const form = node('form', 'identity-form'); form.id = 'person-form';
      if (!person) {
        const label = node('label', '', 'Nombre'); const input = node('input'); input.name = 'personName'; input.required = true; input.maxLength = 64; input.autocomplete = 'off'; label.append(input); form.append(label);
      }
      const label = node('label', '', 'Nivel de cuenta'); const level = node('select'); level.name = 'permission_level';
      for (const [value, text] of Object.entries(LEVELS)) { const option = node('option', '', text); option.value = value; level.append(option); }
      level.value = person?.permission_level || 'adult'; label.append(level); form.append(label);
      const submit = node('button', 'identity-primary full', person ? 'Guardar nivel' : 'Crear persona'); submit.type = 'submit'; form.append(submit);
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
      const actions = node('div', 'modal-actions');
      actions.append(control('Conservar persona', closeDialog), control('Retirar persona', () => {
        void mutate(`/api/admin/users/${encodeURIComponent(person.id)}`, 'DELETE', undefined, 'Persona retirada.');
      }, 'identity-danger'));
      actions.lastChild.id = 'confirm-person-removal'; dialog.append(actions); dialog.showModal();
    }
    function leave() {
      retireVoices(); visible = false; readVersion++; readController?.abort(); readController = null;
      rows = []; selected = null; known = false; closeDialog(); container.replaceChildren();
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
