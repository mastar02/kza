(function (root) {
  'use strict';
  const LABELS = { operator: 'Administración', home: 'Acceso de hogar', evaluator: 'Evaluación temporal' };
  const ACTIONS = { issue: 'Crear', revise: 'Modificar', revoke: 'Retirar' };
  const REASONS = { actor_retired: 'La autorización dejó de estar vigente.', scope_unavailable: 'Las muestras elegidas no están disponibles.',
    grant_invalid: 'La configuración no fue aceptada.', target_missing: 'El acceso ya no está disponible.', writer_busy: 'Había otra operación pendiente.',
    revision_conflict: 'La revisión cambió; consultá el catálogo.', account_unavailable: 'La persona ya no está activa.', capacity: 'Se alcanzó la capacidad disponible.' };
  function mountAccess({ container, dialog, secretDialog = dialog, controller, icon = () => container.ownerDocument.createTextNode('') }) {
    const doc = container.ownerDocument;
    let active = true, state = controller.snapshot(), formState = null, secretSerial = 0, secretOpen = false;
    function node(tag, classes, text) { const value = doc.createElement(tag); if (classes) value.className = classes; if (text !== undefined) value.textContent = text; return value; }
    function button(text, handler, classes = 'identity-secondary') { const value = node('button', classes, text); value.type = 'button'; value.addEventListener('click', handler); return value; }
    function format(value) { const date = new Date(value * 1000); return Number.isFinite(value) && Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Sin fecha confirmada'; }
    function localDate(value) { const date = new Date(value * 1000); return Number.isFinite(date.getTime()) ? new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 23) : ''; }
    function clearInputs(target) { for (const input of target.querySelectorAll('input,select,textarea')) input.value = ''; }
    function closeSecret() { secretSerial++; secretOpen = false; clearInputs(secretDialog); if (secretDialog.open) secretDialog.close(); secretDialog.replaceChildren(); }
    function closeForm() { formState = null; clearInputs(dialog); if (dialog.open) dialog.close(); dialog.replaceChildren(); }
    function modal(target, title, description, close) {
      const top = node('div', 'dialog-top'); top.append(node('p', 'eyebrow', 'KZA / ACCESOS'), button('×', close, 'icon-button'));
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
      const reveal = button('Mostrar', () => { input.type = input.type === 'password' ? 'text' : 'password'; reveal.textContent = input.type === 'password' ? 'Mostrar' : 'Ocultar'; });
      const copy = button('Copiar credencial', async () => {
        if (!active || !state.visible || serial !== secretSerial) return;
        if (!root.navigator?.clipboard?.writeText) { input.type = 'text'; input.focus(); input.select(); status.textContent = 'Seleccioná el texto y copialo con el navegador.'; return; }
        try {
          // Do not keep another JS string while the browser owns its copy operation.
          const pending = root.navigator.clipboard.writeText(input.value);
          await pending;
          if (active && state.visible && serial === secretSerial) status.textContent = 'Credencial copiada por tu solicitud.';
        } catch (_) { if (active && state.visible && serial === secretSerial) status.textContent = 'No pudimos copiarla. Mostrala y seleccioná el texto.'; }
      }, 'identity-primary');
      const actions = node('div', 'modal-actions'); actions.append(reveal, copy, button('Ya la guardé', closeSecret));
      secretDialog.append(label, node('p', 'identity-muted', `Recibo ${metadata.request_id}`), status, actions);
      secret = metadata = null; secretOpen = true; if (!secretDialog.open) secretDialog.showModal();
    }
    const heading = node('div', 'identity-heading'), intro = node('div');
    const title = node('h2', '', 'Cada acceso, un alcance.'); title.id = 'access-title';
    intro.append(node('p', 'eyebrow', 'PERMISOS EXPLÍCITOS'), title, node('p', '', 'Administrá credenciales y consultá los recibos de sus cambios. Los niveles de cuenta y los perfiles de voz se gestionan por separado.'));
    const add = button('Nuevo acceso', () => openForm(), 'identity-primary'); add.id = 'add-access'; add.prepend(icon('plus')); heading.append(intro, add);
    const toolbar = node('div', 'people-toolbar'), status = node('p'), refresh = button('Actualizar accesos', () => controller.refresh());
    status.id = 'access-status'; status.setAttribute('role', 'status'); toolbar.append(status, refresh);
    const panel = node('section', 'panel access-catalog'), table = node('table', 'access-table'), head = node('thead'), tr = node('tr'), rows = node('tbody');
    for (const label of ['Acceso', 'Rol', 'Vigencia', 'Persona', 'Estado', 'Acciones']) { const cell = node('th', '', label); cell.scope = 'col'; tr.append(cell); }
    head.append(tr); table.append(head, rows); panel.append(table);
    const page = node('div', 'voice-list-foot'), prior = button('Anterior', () => controller.previous()), next = button('Siguiente', () => controller.next()), pageInfo = node('span');
    page.append(prior, pageInfo, next); panel.append(page);
    const pendingPanel = node('section', 'panel access-receipts'); pendingPanel.setAttribute('aria-label', 'Recibos de esta sesión');
    const note = node('p', 'identity-contract-note', 'Una credencial no identifica una voz. El rol del dashboard no cambia el nivel de cuenta de una persona. La vigencia efectiva de cada muestra la determina el backend.'); note.prepend(icon('shield'));
    container.replaceChildren(heading, toolbar, panel, pendingPanel, note);

    function updateForm() {
      const value = formState; if (!value) return;
      let valid = state.canMutate;
      if (value.row) valid = valid && state.rows.some(row => row.id === value.row.id && row.revision === value.row.revision);
      try { if (value.read) value.read(); } catch (_) { valid = false; }
      value.submit.disabled = !valid;
      value.notice.textContent = value.row && !state.rows.some(row => row.id === value.row.id && row.revision === value.row.revision)
        ? 'Esta revisión ya no está en el catálogo actual. Cerrá el diálogo y volvé a elegir el acceso.'
        : !state.known ? 'Consultando el estado actual. El envío permanece deshabilitado.' : state.notice;
    }
    function refillPeople() {
      if (!formState?.person) return;
      const current = formState.person.value, select = formState.person, choices = new Map();
      select.replaceChildren(node('option', '', 'Compartido · sin persona vinculada')); select.firstChild.value = '';
      const names = new Map(); for (const person of state.people) if (person.is_active) names.set(person.name.toLocaleLowerCase(), (names.get(person.name.toLocaleLowerCase()) || 0) + 1);
      for (const person of state.people) {
        if (!person.is_active) continue;
        choices.set(person.id, person); const option = node('option', '', names.get(person.name.toLocaleLowerCase()) > 1 ? `${person.name} · ${person.id}` : person.name);
        option.value = person.id; select.append(option);
      }
      if (current && !choices.has(current)) { const unavailable = node('option', '', 'Persona pendiente de verificar o no disponible'); unavailable.value = current; unavailable.disabled = true; select.append(unavailable); }
      select.value = current; formState.choices = choices;
    }
    function openForm(row = null, imported = null) {
      if (!active || !state.canMutate) return;
      closeSecret(); closeForm();
      modal(dialog, row ? 'Modificar acceso' : 'Crear un acceso', 'Elegí el alcance de esta credencial. La operación se envía una sola vez y se verifica con su recibo.', closeForm);
      const form = node('form', 'identity-form'), subjectLabel = node('label', '', 'Nombre del acceso'), subject = node('input');
      subject.type = 'text'; subject.required = true; subject.maxLength = 256; subject.autocomplete = 'off'; subject.value = row?.subject || ''; subject.readOnly = Boolean(row); subjectLabel.append(subject);
      const roleLabel = node('label', '', 'Rol del dashboard'), role = node('select');
      for (const [id, label] of Object.entries(LABELS)) { const option = node('option', '', label); option.value = id; role.append(option); }
      role.value = imported ? 'evaluator' : row?.role || 'home'; roleLabel.append(role);
      const expiresLabel = node('label', '', 'Vence el'), expires = node('input'); expires.type = 'datetime-local'; expires.step = '0.001'; expires.required = true;
      expires.value = localDate(row?.expires_at || (state.observed_at || Date.now() / 1000) + 3600); expiresLabel.append(expires);
      const home = node('div'), personLabel = node('label', '', 'Persona del hogar (opcional)'), person = node('select');
      personLabel.append(person); home.append(personLabel, node('p', 'identity-muted', 'Sólo se usan personas activas y sus IDs canónicos. Elegir una persona no cambia su nivel de cuenta.'));
      const samples = node('section', 'access-selections'), sampleRows = node('div');
      const sampleTitle = node('h3', '', 'Muestras autorizadas'); samples.append(sampleTitle, node('p', 'identity-muted', 'Usá IDs de muestras reales, o importá la selección de Voces. Audio y transcripción se autorizan por separado.'), sampleRows);
      function addSample(seed = null) {
        if (sampleRows.children.length >= 64) return;
        const line = node('fieldset', 'access-sample'), identity = node('input'), deadline = node('input');
        identity.type = 'text'; identity.required = true; identity.pattern = 'vf_[0-9a-f]{64}'; identity.placeholder = 'ID de la muestra real'; identity.value = seed?.fingerprint_id || '';
        identity.setAttribute('aria-label', 'ID de muestra'); deadline.type = 'datetime-local'; deadline.step = '0.001'; deadline.required = true; deadline.value = localDate(seed?.expires_at || new Date(expires.value).getTime() / 1000); deadline.setAttribute('aria-label', 'Vencimiento de esta muestra');
        const modes = node('div');
        for (const [id, label] of [['audio_preview', 'Audio'], ['transcript_full', 'Transcripción']]) {
          const choice = node('label', '', label), input = node('input'); input.type = 'checkbox'; input.value = id; input.checked = Boolean(seed?.modalities?.includes(id)); choice.prepend(input); modes.append(choice);
        }
        line.append(identity, deadline, modes, button('Quitar muestra', () => { line.remove(); updateForm(); })); sampleRows.append(line);
      }
      samples.append(button('Añadir muestra', () => { addSample(); updateForm(); }));
      for (const item of imported || row?.samples || [null]) addSample(item);
      const notice = node('p', 'access-dialog-status'); notice.setAttribute('role', 'status');
      const submit = node('button', 'identity-primary', row ? 'Enviar modificación' : 'Crear credencial'); submit.type = 'submit';
      const actions = node('div', 'modal-actions'); actions.append(button('Cancelar', closeForm), submit);
      form.append(subjectLabel, roleLabel, expiresLabel, home, samples, notice, actions); dialog.append(form);
      function read() {
        const selected = [];
        if (role.value === 'evaluator') for (const line of sampleRows.children) {
          const inputs = line.querySelectorAll('input');
          selected.push({ fingerprint_id: inputs[0].value.trim(), expires_at: new Date(inputs[1].value).getTime() / 1000,
            modalities: [...line.querySelectorAll('input[type=checkbox]:checked')].map(input => input.value) });
        }
        const chosen = role.value === 'home' ? person.value : '';
        if (chosen && !formState.choices.has(chosen)) throw Error('unavailable_person');
        const grant = { role: role.value, expires_at: new Date(expires.value).getTime() / 1000,
          user_id: chosen || null, selections: selected, ...(row ? {} : { subject: subject.value.trim() }) };
        return root.NexoAccessOwner.normalizeGrant(grant, !row);
      }
      formState = { row, submit, notice, read, person, choices: new Map() };
      refillPeople(); person.value = row?.user_id || '';
      if (person.value !== (row?.user_id || '')) { const unknown = node('option', '', 'Persona pendiente de verificar'); unknown.value = row.user_id; unknown.disabled = true; person.append(unknown); person.value = row.user_id; }
      function roleChanged() {
        home.hidden = role.value !== 'home'; samples.hidden = role.value !== 'evaluator';
        for (const control of samples.querySelectorAll('input,button')) control.disabled = role.value !== 'evaluator';
        if (role.value !== 'home') person.value = '';
        if (role.value === 'home') void controller.loadPeople();
        updateForm();
      }
      role.addEventListener('change', roleChanged); form.addEventListener('input', updateForm);
      form.addEventListener('submit', event => {
        event.preventDefault(); if (!formState || submit.disabled) return;
        let spec; try { spec = read(); } catch (_) { updateForm(); return; }
        const target = row && { id: row.id, revision: row.revision }; closeForm();
        if (target) void controller.revise(target.id, target.revision, spec);
        else void controller.issue(spec, { acceptSecret: showSecret });
        spec = null;
      });
      roleChanged(); if (!dialog.open) dialog.showModal(); updateForm();
    }
    function openRemoval(row) {
      if (!active || !state.canMutate) return;
      closeSecret(); closeForm();
      modal(dialog, 'Retirar este acceso', 'Se revocará esta credencial y sus sesiones derivadas. La persona y sus perfiles vocales se conservan.', closeForm);
      dialog.append(node('strong', '', row.subject), node('p', 'identity-muted', `${LABELS[row.role]} · revisión ${row.revision}`));
      const notice = node('p', 'access-dialog-status'), submit = button('Retirar acceso', () => { const id = row.id, version = row.revision; closeForm(); void controller.revoke(id, version); }, 'identity-danger');
      const actions = node('div', 'modal-actions'); actions.append(button('Cancelar', closeForm), submit); dialog.append(notice, actions);
      formState = { row, submit, notice }; if (!dialog.open) dialog.showModal(); updateForm();
    }
    function render(value) {
      if (!active) return; state = value;
      if (!value.visible) { closeSecret(); closeForm(); rows.replaceChildren(); pendingPanel.replaceChildren(); return; }
      status.textContent = value.notice || (value.known ? `${value.rows.length} accesos en esta página.` : 'Consultando accesos…'); status.classList.toggle('error-message', value.error);
      add.disabled = !value.canMutate; refresh.disabled = value.busy || value.loading; prior.disabled = !value.canPrevious; next.disabled = !value.canNext;
      pageInfo.textContent = value.known ? `Página ${value.page} · ${value.rows.length} accesos visibles` : 'Estado pendiente de consulta'; rows.replaceChildren();
      for (const entry of value.rows) {
        const line = node('tr'); line.dataset.credentialId = entry.id;
        const titleCell = node('td'); titleCell.append(node('strong', '', entry.subject), node('div', 'identity-muted', entry.id));
        const actions = node('td'), edit = button('Modificar', () => openForm(entry)), remove = button('Retirar', () => openRemoval(entry), 'identity-danger'); edit.disabled = remove.disabled = !value.canMutate; actions.append(edit, remove);
        line.append(titleCell, node('td', '', LABELS[entry.role]), node('td', '', format(entry.expires_at)), node('td', '', entry.user_id || 'Sin persona vinculada'), node('td', '', entry.pending ? 'Cambio pendiente' : entry.active ? 'Vigente' : 'No vigente'), actions); rows.append(line);
      }
      if (!value.rows.length) { const line = node('tr'), cell = node('td', 'identity-empty', value.known ? 'No hay accesos en esta página.' : 'No hay un catálogo vigente para mostrar.'); cell.colSpan = 6; line.append(cell); rows.append(line); }
      pendingPanel.replaceChildren(node('h3', '', 'Recibos de esta sesión'));
      for (const item of value.pending.slice().reverse()) {
        const line = node('div', 'access-receipt'), description = item.state === 'terminal'
          ? item.receipt.outcome === 'committed' ? 'Aplicado por el backend' : REASONS[item.receipt.reason] || 'No se aplicó el cambio'
          : 'Resultado sin confirmar';
        line.append(node('strong', '', `${ACTIONS[item.action]} · ${description}`), node('p', 'identity-muted', item.request_id));
        if (item.state !== 'terminal') { const check = button('Consultar recibo', () => controller.reconcile(item.request_id)); check.disabled = value.busy; line.append(check); }
        pendingPanel.append(line);
      }
      refillPeople(); updateForm();
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
  root.NexoAccess = { mountAccess };
  if (typeof module !== 'undefined' && module.exports) module.exports = { mountAccess };
})(typeof window !== 'undefined' ? window : globalThis);
