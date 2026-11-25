(function (root) {
  'use strict';
  const LABELS = { operator: 'Administración', home: 'Acceso de hogar', evaluator: 'Evaluación temporal' };
  const ACTIONS = { issue: 'Crear', revise: 'Modificar', revoke: 'Retirar' };
  const REASONS = { actor_retired: 'La autorización dejó de estar vigente.', scope_unavailable: 'Las muestras elegidas no están disponibles.',
    grant_invalid: 'La configuración no fue aceptada.', target_missing: 'El acceso ya no está disponible.', writer_busy: 'Había otra operación pendiente.',
    revision_conflict: 'El acceso cambió. Actualizá la lista.', account_unavailable: 'La persona ya no está activa.', capacity: 'Se alcanzó la capacidad disponible.' };
  // Match normalizeName in people.js when identifying homonyms.
  function normalizeName(value) {
    return value.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase().replace(/\s+/g, ' ').trim();
  }
  function counts(keys) {
    const result = new Map();
    for (const key of keys) result.set(key, (result.get(key) || 0) + 1);
    return result;
  }
  function createConnection(doc, getSession = () => null) {
    const panel = doc.createElement('section'), title = doc.createElement('h2');
    const rows = doc.createElement('div');
    const values = ['Sesión', 'Última lectura'].map(label => {
      const row = doc.createElement('div'), copy = doc.createElement('div');
      const name = doc.createElement('strong'), text = doc.createElement('span');
      row.className = 'row'; copy.className = 'row-main';
      name.className = 'row-title'; name.textContent = label; text.className = 'row-meta';
      copy.append(name, text); row.append(copy); rows.append(row);
      return text;
    });
    const freshness = doc.createElement('span');
    freshness.className = 'tag'; rows.lastChild.append(freshness);
    panel.className = 'sistema-section'; panel.setAttribute('aria-label', 'Conexión'); panel.hidden = true;
    title.textContent = 'Conexión'; rows.className = 'rows'; panel.append(title, rows);
    return { panel, update(readAt, status) {
      const session = getSession();
      const confirmed = session && Object.hasOwn(LABELS, session.role) && Number.isFinite(session.expiresAt);
      panel.hidden = !confirmed;
      if (!confirmed) { values.forEach(value => { value.textContent = ''; }); freshness.textContent = ''; return; }
      values[0].textContent = `${LABELS[session.role]} · Vence el ${new Date(session.expiresAt).toLocaleString()}`;
      values[1].textContent = readAt === null ? 'Todavía sin lectura confirmada' : new Date(readAt).toLocaleString();
      freshness.textContent = { real: 'Datos disponibles', loading: 'Actualizando', expired: 'Datos vencidos',
        unavailable: 'Consulta no disponible', waiting: 'Esperando confirmación' }[status] || 'Esperando confirmación';
    } };
  }
  function mountAccess({ container, dialog, secretDialog = dialog, controller, getSession, personId = null, personName, ambiguous = false, current = () => true, onManage }) {
    const doc = container.ownerDocument, scoped = personId !== null;
    const mustUseSystem = scoped && ambiguous;
    let active = true, state = controller.snapshot(), formState = null, secretSerial = 0, secretOpen = false;
    function node(tag, classes, text) { const value = doc.createElement(tag); if (classes) value.className = classes; if (text !== undefined) value.textContent = text; return value; }
    function button(text, handler, classes = scoped ? 'identity-secondary' : 'btn') { const value = node('button', classes, text); value.type = 'button'; value.addEventListener('click', handler); return value; }
    function format(value) { const date = new Date(value * 1000); return Number.isFinite(value) && Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Sin fecha confirmada'; }
    function localDate(value) { const date = new Date(value * 1000); return Number.isFinite(date.getTime()) ? new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 23) : ''; }
    function clearInputs(target) { for (const input of target.querySelectorAll('input,select,textarea')) input.value = ''; }
    let opener = null, restorePending = false;
    const rowControls = new WeakMap();
    function restoreFocus() {
      if (!restorePending || !active || !current() || !state.visible || dialog.open || secretDialog.open) return;
      const focused = doc.activeElement;
      if (focused && focused !== doc.body && focused !== opener && focused !== title) {
        opener = null; restorePending = false; return;
      }
      if (opener?.isConnected && !opener.disabled) {
        opener.focus({ preventScroll: true }); opener = null; restorePending = false;
      } else if (add.isConnected && !add.disabled) {
        add.focus({ preventScroll: true }); opener = null; restorePending = false;
      } else if (title.isConnected) {
        title.focus({ preventScroll: true });
      }
    }
    function closeSecret() {
      const wasOpen = secretDialog.open;
      secretSerial++; secretOpen = false; clearInputs(secretDialog);
      if (wasOpen) secretDialog.close();
      secretDialog.replaceChildren();
      if (wasOpen) { restorePending = true; restoreFocus(); }
    }
    function closeForm(restore = true) {
      const wasOpen = dialog.open;
      formState = null; clearInputs(dialog);
      if (wasOpen) dialog.close();
      dialog.replaceChildren();
      if (wasOpen && restore) { restorePending = true; restoreFocus(); }
    }
    function modal(target, title, description, close) {
      const isForm = target === dialog && target !== secretDialog;
      if (isForm) target.classList.toggle('personas-access-dialog', scoped);
      const top = node('div', isForm && scoped ? 'personas-toolbar' : 'sistema-dialog-top');
      top.append(button('Cerrar', () => close(), 'btn quiet'));
      top.lastChild.setAttribute('aria-label', 'Cerrar diálogo');
      const heading = node('h2', '', title), copy = node('p', 'modal-description', description);
      heading.id = target === secretDialog ? 'access-secret-title' : 'access-dialog-title';
      target.replaceChildren(top, heading, copy); target.setAttribute('aria-labelledby', heading.id);
    }
    function showSecret(secret, metadata) {
      if (!active || !state.visible) return;
      closeForm(); closeSecret(); const serial = secretSerial;
      modal(secretDialog, 'Guardá esta credencial.', 'Se muestra una sola vez. Al cerrar este panel no se puede recuperar; el recibo conserva únicamente el resultado de la operación.', closeSecret);
      const label = node('label', '', 'Credencial recién emitida'), input = node('input', 'access-secret');
      input.type = 'password'; input.readOnly = true; input.autocomplete = 'off'; input.spellcheck = false; input.value = secret;
      label.append(input); const status = node('p', 'access-secret-status'); status.setAttribute('role', 'status');
      const reveal = button('Mostrar', () => { input.type = input.type === 'password' ? 'text' : 'password'; reveal.textContent = input.type === 'password' ? 'Mostrar' : 'Ocultar'; }, 'btn');
      const copy = button('Copiar credencial', async () => {
        if (!active || !state.visible || serial !== secretSerial) return;
        if (!root.navigator?.clipboard?.writeText) { input.type = 'text'; input.focus(); input.select(); status.textContent = 'Seleccioná el texto y copialo con el navegador.'; return; }
        try {
          // Do not keep another JS string while the browser owns its copy operation.
          const pending = root.navigator.clipboard.writeText(input.value);
          await pending;
          if (active && state.visible && serial === secretSerial) status.textContent = 'Credencial copiada por tu solicitud.';
        } catch (_) { if (active && state.visible && serial === secretSerial) status.textContent = 'No pudimos copiarla. Mostrala y seleccioná el texto.'; }
      }, 'btn primary');
      const actions = node('div', 'row-actions'); actions.append(reveal, copy, button('Ya la guardé', closeSecret, 'btn'));
      secretDialog.append(label);
      if (!scoped) secretDialog.append(node('p', 'row-meta', operationLabel(metadata.request_id)));
      secretDialog.append(status, actions);
      secret = metadata = null; secretOpen = true; if (!secretDialog.open) secretDialog.showModal();
    }
    const operationNumbers = new Map();
    let lastOperation = 0;
    function operationLabel(id) {
      if (!operationNumbers.has(id)) operationNumbers.set(id, ++lastOperation);
      return `Operación ${operationNumbers.get(id)}`;
    }
    const heading = node('header', 'page-head'), intro = node('div');
    const title = node(scoped ? 'h2' : 'h1', '', 'Accesos'); title.id = 'access-title';
    title.tabIndex = -1;
    intro.append(title, node('p', '', 'Creá y administrá credenciales para entrar al panel de KZA.'));
    const add = button('Nuevo acceso', event => openForm(null, null, event.currentTarget), 'btn primary'); add.id = 'add-access';
    heading.append(intro, add);
    const link = !scoped ? createConnection(doc, getSession) : null;
    const toolbar = node('div', 'sistema-toolbar'), status = node('p', 'row-meta'), refresh = button('Actualizar accesos', () => controller.refresh());
    status.id = 'access-status'; status.setAttribute('role', 'status'); toolbar.append(status, refresh);
    const panel = node('section', 'sistema-section'), rows = node('div', 'rows');
    const listTitle = node('h2', '', 'Credenciales');
    panel.setAttribute('aria-label', 'Credenciales'); panel.append(listTitle, toolbar, rows);
    const page = node('div', 'row-actions'), prior = button('Anterior', () => controller.previous()), next = button('Siguiente', () => controller.next()), pageInfo = node('span', 'row-meta');
    page.append(prior, pageInfo, next); panel.append(page);
    const pendingPanel = node('section', 'sistema-section access-receipts');
    pendingPanel.setAttribute('aria-label', 'Recibos de esta sesión');
    const note = node('p', 'row-meta', 'El acceso al panel, el nivel de cuenta y los perfiles de voz se gestionan por separado.');
    if (scoped) status.className = '';
    const personRows = scoped ? node('div', 'rows') : null;
    const management = scoped ? node('a', 'btn quiet', 'Ver todos los accesos y recibos') : null;
    if (scoped) {
      management.href = '#accesos';
      management.addEventListener('click', event => {
        if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        if (current()) void onManage();
      });
      title.textContent = 'Acceso al panel de KZA';
      title.id = 'person-access-title';
      add.className = 'btn primary';
      add.id = 'add-person-access';
      add.textContent = 'Crear credencial';
      const actions = node('div', 'personas-toolbar');
      if (!mustUseSystem) actions.append(add);
      actions.append(refresh, management);
      container.replaceChildren(title, status, personRows, page);
      if (mustUseSystem) {
        container.append(node('p', 'row-meta', 'Hay otra persona con el mismo nombre. Creá o cambiá el acceso desde Sistema → Accesos.'));
      }
      container.append(actions);
      page.className = 'row-actions';
      for (const control of [prior, next, refresh]) control.className = 'btn';
    } else {
      panel.append(note);
      container.replaceChildren(heading, link.panel, panel, pendingPanel);
    }

    function updateForm() {
      const value = formState; if (!value) return;
      let valid = state.canMutate;
      if (value.row) valid = valid && state.rows.some(row => row.id === value.row.id && row.revision === value.row.revision);
      try { if (value.read) value.read(); } catch (_) { valid = false; }
      value.submit.disabled = !valid;
      value.notice.textContent = value.row && !state.rows.some(row => row.id === value.row.id && row.revision === value.row.revision)
        ? 'El acceso cambió. Cerrá el diálogo y volvé a elegirlo.'
        : !state.known ? (scoped ? 'Consultando accesos…' : state.status === 'unavailable'
          ? 'No pudimos verificar los accesos. Cerrá el diálogo y actualizá la lista.'
          : 'Consultando los accesos. Esperá el resultado para guardar.')
        : scoped && value.read && !state.people.some(account => account.id === personId && account.is_active)
          ? 'Para guardar, la cuenta debe estar activa y disponible.'
          : scoped ? personNotice(state) : '';
    }
    function personNotice({ notice, known }) {
      const text = notice.replace('El catálogo no está disponible.', 'No pudimos consultar los accesos.');
      if (known && text.startsWith('No pudimos consultar los accesos.')) return '';
      return text.replace('El backend confirmó', 'El servidor confirmó');
    }
    function accessNotice(value) {
      const notice = value.notice.replaceAll('backend', 'servidor').replace('El catálogo no está disponible.', 'No pudimos consultar los accesos.');
      return value.known && notice.startsWith('No pudimos consultar los accesos.') ? '' : notice;
    }
    function field(label, control) {
      const wrapper = node('label', 'field');
      wrapper.append(node('span', '', label), control);
      return wrapper;
    }
    function describePerson() {
      const { person, personReference, choices } = formState;
      const reference = choices.get(person.value)?.reference || '';
      personReference.textContent = reference;
      if (reference) person.setAttribute('aria-describedby', personReference.id);
      else person.removeAttribute('aria-describedby');
    }
    function refillPeople(initialId = null) {
      if (!formState?.person || scoped || formState.people === state.people) return;
      const selectedId = initialId ?? formState.choices.get(formState.person.value)?.id ?? formState.unavailablePerson;
      formState.people = state.people;
      const select = formState.person, choices = new Map();
      select.replaceChildren(node('option', '', 'Sin persona vinculada')); select.firstChild.value = '';
      const people = state.people.filter(person => person.is_active);
      const names = counts(people.map(person => normalizeName(person.name)));
      for (const [index, person] of people.entries()) {
        const key = String(index + 1);
        const reference = names.get(normalizeName(person.name)) > 1 ? `Referencia ${key}` : '';
        const option = node('option', '', reference ? `${person.name} · ${reference}` : person.name);
        choices.set(key, { id: person.id, reference }); option.value = key; select.append(option);
        if (person.id === selectedId) select.value = key;
      }
      if (selectedId && !choices.has(select.value)) {
        const unavailable = node('option', '', 'Persona pendiente de verificar o no disponible');
        unavailable.value = 'unavailable'; unavailable.disabled = true; select.append(unavailable); select.value = unavailable.value;
      }
      formState.choices = choices;
      formState.unavailablePerson = select.value === 'unavailable' ? selectedId : null;
      describePerson();
    }
    function openForm(row = null, imported = null, trigger = null, reference = '') {
      if (!active || !current() || !state.canMutate || mustUseSystem) return;
      closeSecret(); closeForm();
      opener = trigger || add; restorePending = false;
      const description = scoped
        ? 'Acceso de hogar para ' + personName + '. La operación se envía una sola vez y se verifica con su recibo.'
        : 'Elegí el alcance de esta credencial. La operación se envía una sola vez y se verifica con su recibo.';
      modal(dialog, row ? 'Modificar acceso' : 'Crear un acceso', description, closeForm);
      if (reference) dialog.append(node('p', 'row-meta', reference));
      const form = node('form', scoped ? 'personas-form' : 'sistema-form'), subject = node('input');
      subject.type = 'text'; subject.required = true; subject.maxLength = 256; subject.autocomplete = 'off'; subject.value = row?.subject || ''; subject.readOnly = Boolean(row);
      const role = node('select');
      for (const [id, label] of Object.entries({ operator: 'Administración', home: 'Hogar', evaluator: 'Evaluación' })) {
        const option = node('option', '', label); option.value = id; role.append(option);
      }
      role.value = imported ? 'evaluator' : row?.role || 'home';
      const expires = node('input'); expires.type = 'datetime-local'; expires.step = '0.001'; expires.required = true;
      expires.value = localDate(row?.expires_at || (state.observed_at || Date.now() / 1000) + 3600);
      const home = node('div', 'sistema-section'), person = node('select');
      const personReference = node('p', 'row-meta'); personReference.id = 'sistema-person-reference';
      home.append(field('Persona del hogar (opcional)', person), personReference);
      const samples = node('section', 'sistema-section access-selections'), sampleRows = node('div', 'sistema-section');
      const sampleTitle = node('h3', '', 'Muestras autorizadas'); samples.append(sampleTitle, node('p', 'row-meta', 'Ingresá la referencia de la muestra o importá una selección desde Voces. Elegí si se puede escuchar el audio, leer la transcripción o ambas cosas.'), sampleRows);
      const sampleIds = new WeakMap();
      let sampleNumber = 0;
      function addSample(seed = null) {
        if (sampleRows.children.length >= 64) return;
        const line = node('fieldset', 'access-sample'), identity = node('input'), deadline = node('input');
        identity.type = 'text';
        if (seed) {
          sampleIds.set(identity, seed.fingerprint_id);
          identity.value = `Muestra ${++sampleNumber}`;
          identity.readOnly = true;
        } else {
          identity.required = true;
          identity.pattern = 'vf_[0-9a-f]{64}';
        }
        deadline.type = 'datetime-local'; deadline.step = '0.001'; deadline.required = true; deadline.value = localDate(seed?.expires_at || new Date(expires.value).getTime() / 1000);
        const modes = node('div', 'sistema-sample-modes');
        for (const [id, label] of [['audio_preview', 'Audio'], ['transcript_full', 'Transcripción']]) {
          const choice = node('label', '', label), input = node('input'); input.type = 'checkbox'; input.value = id; input.checked = Boolean(seed?.modalities?.includes(id)); choice.prepend(input); modes.append(choice);
        }
        line.append(field('Referencia de la muestra', identity), field('Vencimiento de esta muestra', deadline), modes, button('Quitar muestra', () => { line.remove(); updateForm(); })); sampleRows.append(line);
      }
      samples.append(button('Añadir muestra', () => { addSample(); updateForm(); }));
      if (!scoped) for (const item of imported || row?.samples || [null]) addSample(item);
      const notice = node('p', scoped ? 'access-dialog-status' : 'access-dialog-status row-meta'); notice.setAttribute('role', 'status');
      const submit = node('button', 'btn primary', row ? 'Enviar modificación' : 'Crear credencial'); submit.type = 'submit';
      const actions = node('div', 'row-actions'); actions.append(button('Cancelar', () => closeForm()), submit);
      let visitExpiry = null;
      if (!row && !scoped) {
        const visit = button('Visita (7 días)', () => {
          role.value = 'home'; person.value = '';
          visitExpiry = Date.now() / 1000 + 7 * 86400;
          expires.value = localDate(visitExpiry);
          if (!subject.value.trim()) subject.value = 'Visita';
          roleChanged();
        });
        visit.classList.add('access-visit-preset');
        form.append(visit);
        expires.addEventListener('input', () => { visitExpiry = null; });
      }
      form.append(field('Nombre del acceso', subject), scoped ? node('p', 'tag', LABELS.home) : field('Tipo de acceso', role),
        field('Vence el', expires));
      if (scoped) actions.firstChild.className = 'btn';
      else form.append(home, samples);
      form.append(notice, actions);
      dialog.append(form);
      function read() {
        const selected = [];
        if (role.value === 'evaluator') for (const line of sampleRows.children) {
          const inputs = line.querySelectorAll('input');
          selected.push({ fingerprint_id: sampleIds.get(inputs[0]) ?? inputs[0].value.trim(), expires_at: new Date(inputs[1].value).getTime() / 1000,
            modalities: [...line.querySelectorAll('input[type=checkbox]:checked')].map(input => input.value) });
        }
        const chosen = scoped ? personId : role.value === 'home' ? formState.choices.get(person.value)?.id : null;
        if (scoped ? !state.people.some(account => account.id === chosen && account.is_active)
          : role.value === 'home' && person.value && !chosen) throw Error('unavailable_person');
        const deadline = visitExpiry !== null && expires.value === localDate(visitExpiry)
          ? visitExpiry : new Date(expires.value).getTime() / 1000;
        const grant = { role: scoped ? 'home' : role.value, expires_at: deadline,
          user_id: chosen || null, selections: selected, ...(row ? {} : { subject: subject.value.trim() }) };
        return root.NexoAccessOwner.normalizeGrant(grant, !row);
      }
      formState = { row, submit, notice, read, person, personReference, choices: new Map() };
      if (!scoped) {
        refillPeople(row?.user_id);
        person.addEventListener('change', () => { formState.unavailablePerson = null; describePerson(); });
      }
      function roleChanged() {
        home.hidden = role.value !== 'home'; samples.hidden = role.value !== 'evaluator';
        for (const control of samples.querySelectorAll('input,button')) control.disabled = role.value !== 'evaluator';
        if (role.value !== 'home') { person.value = ''; formState.unavailablePerson = null; }
        if (!scoped) describePerson();
        if (role.value === 'home') void controller.loadPeople();
        updateForm();
      }
      role.addEventListener('change', roleChanged); form.addEventListener('input', updateForm);
      form.addEventListener('submit', event => {
        event.preventDefault(); if (!current() || !formState || submit.disabled) return;
        let spec; try { spec = read(); } catch (_) { updateForm(); return; }
        const target = row && { id: row.id, revision: row.revision }; closeForm(Boolean(target));
        if (target) void controller.revise(target.id, target.revision, spec);
        else {
          const pending = controller.issue(spec, { acceptSecret: showSecret });
          void pending.finally(() => {
            if (!formState && !secretOpen && !restorePending && opener) { restorePending = true; restoreFocus(); }
          });
        }
        spec = null;
      });
      roleChanged(); if (!dialog.open) dialog.showModal(); updateForm();
    }
    function openRemoval(row, trigger = null, reference = '') {
      if (!active || !current() || !state.canMutate) return;
      closeSecret(); closeForm();
      opener = trigger || add; restorePending = false;
      modal(dialog, 'Retirar este acceso', 'Se revocará esta credencial y sus sesiones derivadas. La persona y sus perfiles vocales se conservan.', closeForm);
      dialog.append(node('strong', '', row.subject), node('p', 'tag', LABELS[row.role]));
      if (reference) dialog.append(node('p', 'row-meta', reference));
      const notice = node('p', scoped ? 'access-dialog-status' : 'access-dialog-status row-meta'), submit = button('Retirar acceso', () => { if (!current()) return; const id = row.id, version = row.revision; closeForm(); void controller.revoke(id, version); }, 'btn danger');
      const actions = node('div', 'row-actions'); actions.append(button('Cancelar', () => closeForm(), 'btn'), submit); dialog.append(notice, actions);
      formState = { row, submit, notice }; if (!dialog.open) dialog.showModal(); updateForm();
      actions.firstChild.focus();
    }
    function renderPersonAccess(value) {
      status.textContent = personNotice(value) || (value.known ? '' : 'Consultando accesos…');
      status.classList.toggle('error-message', value.error);
      add.disabled = !value.canMutate;
      refresh.disabled = !value.visible || value.busy || value.loading;
      prior.disabled = !value.canPrevious;
      next.disabled = !value.canNext;
      page.hidden = !value.canPrevious && !value.canNext && value.page <= 1;
      pageInfo.textContent = `Página ${value.page}`;
      const entries = value.rows.filter(entry => entry.user_id === personId);
      personRows.hidden = !value.known;
      const openerKey = rowControls.get(opener);
      personRows.replaceChildren();
      if (value.known && !entries.length) {
        personRows.append(node('p', 'empty', 'Esta persona no tiene credenciales en esta página.'));
      }
      for (const entry of entries) {
        const line = node('div', 'row'); line.dataset.credentialId = '';
        const copy = node('div', 'row-main');
        copy.append(node('strong', 'row-title', entry.subject),
          node('span', 'row-meta', `${LABELS[entry.role]} · Vence el ${format(entry.expires_at)}`));
        const badge = node('span', 'tag', entry.pending ? 'Cambio pendiente' : entry.active ? 'Vigente' : 'Vencido o retirado');
        const actions = node('div', 'row-actions');
        const edit = mustUseSystem ? null : button('Modificar', event => openForm(entry, null, event.currentTarget), 'btn');
        const remove = button('Retirar', event => openRemoval(entry, event.currentTarget), 'btn danger');
        for (const [action, control] of [['edit', edit], ['remove', remove]]) {
          if (!control) continue;
          control.disabled = !value.canMutate;
          rowControls.set(control, { id: entry.id, action });
          if (openerKey?.id === entry.id && openerKey.action === action) opener = control;
          actions.append(control);
        }
        line.append(copy, badge, actions); personRows.append(line);
      }
      updateForm();
    }
    function render(value) {
      if (!active || !current()) return;
      state = value;
      if (scoped) { renderPersonAccess(value); restoreFocus(); return; }
      if (!value.visible) {
        closeSecret(); closeForm(); rows.replaceChildren(); pendingPanel.replaceChildren(); link.update(null, 'waiting');
        return;
      }
      link.update(Number.isFinite(value.observed_at) ? value.observed_at * 1000 : null, value.status);
      status.textContent = accessNotice(value) || (value.known ? '' : value.loading ? 'Consultando accesos…' : 'Esperando la consulta de accesos.');
      status.classList.toggle('sistema-error', value.error);
      add.disabled = !value.canMutate; refresh.disabled = value.busy || value.loading; prior.disabled = !value.canPrevious; next.disabled = !value.canNext;
      page.hidden = !value.canPrevious && !value.canNext && value.page <= 1;
      pageInfo.textContent = `Página ${value.page}`;
      renderCredentials(value);
      const receipts = node('div', 'rows');
      pendingPanel.replaceChildren(node('h2', '', 'Recibos de esta sesión'), receipts);
      if (!value.pending.length) receipts.append(node('p', 'empty', 'Todavía no hay operaciones en esta sesión.'));
      // Number attempts in creation order and release the references retired by the owner.
      const visibleOperations = new Set(value.pending.map(item => item.request_id));
      for (const id of operationNumbers.keys()) if (!visibleOperations.has(id)) operationNumbers.delete(id);
      for (const item of value.pending) operationLabel(item.request_id);
      for (const item of value.pending.slice().reverse()) {
        const line = node('div', 'row'), copy = node('div', 'row-main');
        const description = item.state === 'terminal'
          ? item.receipt.outcome === 'committed' ? 'Cambio confirmado' : REASONS[item.receipt.reason] || 'No se aplicó el cambio'
          : 'Resultado sin confirmar';
        copy.append(node('strong', 'row-title', `${ACTIONS[item.action]} · ${description}`),
          node('span', 'row-meta', operationLabel(item.request_id)));
        line.append(copy);
        if (item.state !== 'terminal') {
          const check = button('Consultar recibo', () => controller.reconcile(item.request_id));
          check.disabled = value.busy; line.append(check);
        }
        receipts.append(line);
      }
      refillPeople(); updateForm(); restoreFocus();
    }
    function renderCredentials(value) {
      // Use only names already returned to this owner under its existing people permission.
      const people = new Map(value.people.map(person => [person.id, person.name]));
      const entries = value.rows.map(entry => {
        const linked = entry.user_id ? people.get(entry.user_id) || 'Persona vinculada' : 'Sin persona vinculada';
        const expires = `Vence el ${format(entry.expires_at)}`;
        const status = entry.pending ? 'Cambio pendiente' : entry.active ? 'Vigente' : 'No vigente';
        return { entry, linked, expires, status, key: JSON.stringify([normalizeName(entry.subject), entry.role, expires, normalizeName(linked), status]) };
      });
      const occurrences = counts(entries.map(row => row.key));
      const focusKey = rowControls.get(doc.activeElement), openerKey = rowControls.get(opener);
      rows.replaceChildren();
      for (const [index, { entry, linked, expires, status, key }] of entries.entries()) {
        const line = node('div', 'row'); line.dataset.credentialId = '';
        const copy = node('div', 'row-main'), actions = node('div', 'row-actions');
        const referenceText = occurrences.get(key) > 1 ? `Referencia ${index + 1}` : '';
        const edit = button('Modificar', event => openForm(entry, null, event.currentTarget, referenceText));
        const remove = button('Retirar', event => openRemoval(entry, event.currentTarget, referenceText), 'btn danger');
        edit.disabled = remove.disabled = !value.canMutate;
        copy.append(node('strong', 'row-title', entry.subject), node('span', 'row-meta', `${LABELS[entry.role]} · ${expires}`), node('span', 'row-meta', linked));
        if (referenceText) {
          const reference = node('span', 'row-meta', referenceText);
          reference.id = `sistema-access-reference-${index + 1}`;
          copy.append(reference);
          for (const control of [edit, remove]) control.setAttribute('aria-describedby', reference.id);
        }
        actions.append(edit, remove); line.append(copy, node('span', 'tag', status), actions); rows.append(line);
        for (const [action, control] of [edit, remove].entries()) {
          rowControls.set(control, { id: entry.id, action });
          if (openerKey?.id === entry.id && openerKey.action === action) opener = control;
          if (focusKey?.id === entry.id && focusKey.action === action && !control.disabled) control.focus({ preventScroll: true });
        }
      }
      if (!entries.length) rows.append(node('p', 'empty', value.known ? 'No hay credenciales en esta página.' : value.loading ? 'Consultando accesos…' : 'No hay accesos disponibles para mostrar.'));
    }
    const unsubscribe = controller.subscribe(render);
    const cancelForm = event => { event.preventDefault(); if (secretDialog === dialog && secretOpen) closeSecret(); else closeForm(); }, cancelSecret = event => { event.preventDefault(); closeSecret(); };
    dialog.addEventListener('cancel', cancelForm); if (secretDialog !== dialog) secretDialog.addEventListener('cancel', cancelSecret);
    function dispose() {
      if (!active) return; active = false; unsubscribe(); closeSecret(); closeForm(); container.replaceChildren();
      dialog.removeEventListener('cancel', cancelForm); if (secretDialog !== dialog) secretDialog.removeEventListener('cancel', cancelSecret);
      void controller.leave();
    }
    // App integration passes only detached real selection metadata from the current Voices owner.
    dispose.openEvaluation = values => {
      if (!active || !state.canMutate) return false;
      try { const normalized = root.NexoAccessOwner.parseSelections(values); if (!normalized.length) return false; openForm(null, normalized); return true; }
      catch (_) { return false; }
    };
    return dispose;
  }
  root.NexoAccess = { mountAccess, createConnection };
  if (typeof module !== 'undefined' && module.exports) module.exports = { mountAccess };
})(typeof window !== 'undefined' ? window : globalThis);
