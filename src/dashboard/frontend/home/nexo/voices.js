/* NEXO presentation over the existing, expiring voice stores. */
(function (root) {
  'use strict';
  // Match normalizeName in people.js when identifying homonyms.
  function normalizeName(name) {
    return name.normalize('NFD').replace(/\p{M}/gu, '').toLocaleLowerCase().replace(/\s+/g, ' ').trim();
  }
  function mountVoices({ container, dialog, linkDialog, controller, onEvaluate, roomName = () => null, profileNote = () => null, onProfileNotes = null }) {
    const doc = container.ownerDocument, state = controller.state, selected = new Set();
    const listeningButtons = new Map();
    let active = true, profileId = null, renderedReview = null, busy = false, needsRefresh = false, dialogActive = false;
    let requestedReview = null, startingReview = false;
    let linkToken = null, catalogue = null, catalogueStatus = 'waiting', linkDraft = null;
    function node(tag, text = '', className = '') { const value = doc.createElement(tag); value.textContent = text; value.className = className; return value; }
    function button(text, callback, className = 'btn') { const value = node('button', text, className); value.type = 'button'; value.addEventListener('click', callback); return value; }
    function message(store, empty) { return ({ waiting: 'Esperando conexión…', loading: 'Consultando…', unavailable: 'No se pudo consultar. Volvé a actualizar.', expired: 'La consulta venció. Actualizá para ver los datos vigentes.', closed: 'La sesión se cerró.', forbidden: 'Sin permiso para consultar estas muestras.' })[store.status()] || empty; }
    const header = node('header', '', 'page-head'), title = node('div'), headerActions = node('div', '', 'row-actions');
    title.append(node('h1', 'Voces'));
    const headerRefresh = button('Actualizar', refresh), dialogRefresh = button('Actualizar', refresh);
    headerActions.append(headerRefresh, button('Revisar muestras', openDialog, 'btn primary')); header.append(title, headerActions);
    const workspace = node('div', '', 'personas-columns');
    const profiles = node('section', '', 'personas-stack'), list = node('div', '', 'rows'), profilePage = node('div', '', 'row-actions');
    list.id = 'voice-profile-list';
    profiles.append(node('h2', 'Perfiles de voz'), list, profilePage);
    profiles.setAttribute('aria-label', 'Perfiles confirmados');
    const detail = node('section', '', 'detail'); detail.setAttribute('aria-label', 'Detalle del perfil vocal');
    workspace.append(profiles, detail);
    const queue = node('section', '', 'personas-stack'), groups = node('div', '', 'rows'), capturePage = node('div', '', 'row-actions');
    queue.append(node('h2', 'Voces por identificar'),
      node('p', 'Las muestras se muestran de más reciente a más antigua y se conservan hasta 48 horas.', 'row-meta'), groups, capturePage);
    const notice = node('p', '', 'identity-form-error'); notice.setAttribute('role', 'alert');
    container.replaceChildren(header, workspace, queue, notice);
    const progress = node('p', '', 'row-meta'), dialogProgress = node('p', '', 'row-meta');
    for (const status of [progress, dialogProgress]) status.setAttribute('role', 'status');
    headerActions.append(progress);
    const sampleList = node('div', '', 'rows nexo-voice-samples'), review = node('section', '', 'detail nexo-voice-review'), form = node('form', '', 'personas-form'); review.setAttribute('aria-label', 'Escuchar muestra');
    const roomNumbers = new Map();
    function sampleRoom(id) {
      if (!id) return 'Sin habitación registrada';
      const name = roomName(id);
      if (name) return name;
      if (!roomNumbers.has(id)) roomNumbers.set(id, roomNumbers.size + 1);
      return `Habitación ${roomNumbers.get(id)}`;
    }
    const filterView = root.KzaVoiceFilters.mountVoiceFilters({doc,controller,button,roomLabel:sampleRoom});
    const dialogPage = node('div','','row-actions');
    const label = node('label', '', 'field'), name = node('input'), selection = node('p', '', 'review-selection'), error = node('p', '', 'identity-form-error'); error.setAttribute('role', 'alert');
    name.type = 'text'; name.maxLength = 64; name.autocomplete = 'off'; name.required = true; label.append(node('span', 'Nombre del perfil'), name);
    const confirm = button('Confirmar perfil', () => {}, 'btn primary'); confirm.type = 'submit';
    const merge = button('Unir grupos', () => correct('merge')), split = button('Separar muestras', () => correct('split')), unknown = button('Dejar como desconocido', () => correct('unknown'));
    const corrections = node('div', '', 'row-actions'); corrections.append(merge, split, unknown);
    const evaluation = node('section', '', 'voice-evaluation-selection'), evaluationModes = [];
    const prepareEvaluation = button('Preparar acceso de evaluación', () => {
      if (prepareEvaluation.disabled || !active) return;
      const modalities = evaluationModes.filter(input => input.checked).map(input => input.value);
      const rows = state.captures.snapshot().filter(row => selected.has(row.id) && row.review_available);
      if (rows.length !== selected.size || !modalities.length) return;
      const grants = rows.map(row => Object.freeze({ fingerprint_id: row.id, expires_at: row.expires_at, modalities: [...modalities] }));
      void onEvaluate?.(grants);
    });
    if (onEvaluate) {
      evaluation.append(node('h3', 'Evaluación temporal'), node('p', 'Elegí una o más muestras y autorizá cada modalidad. En Sistema → Accesos podés revisar el alcance antes de emitir una credencial.', 'row-meta'));
      for (const [value, text] of [['audio_preview', 'Audio breve'], ['transcript_full', 'Transcripción completa']]) {
        const label = node('label', text), input = node('input'); input.type = 'checkbox'; input.value = value;
        input.addEventListener('change', updateSelection); evaluationModes.push(input); label.prepend(input); evaluation.append(label);
      }
      evaluation.append(prepareEvaluation);
    }
    const modal = node('div', '', 'nexo-voice-modal'), results = node('div','','nexo-voice-results');
    results.append(sampleList,dialogPage); modal.append(results,review); form.append(selection, label, confirm);
    dialog.replaceChildren(button('Cerrar', closeDialog), dialogRefresh, node('h2', 'Revisar muestras de voz'), node('p', 'El perfil confirmado se conserva hasta que lo borres. El audio y la transcripción siguen venciendo con su captura original.', 'modal-description'), filterView.element, modal, corrections,
      node('p', 'Unir incluye todas las muestras de los grupos elegidos, también las de otras páginas. Separar crea otro grupo sólo con las muestras marcadas.', 'relationship-note'), form, ...(onEvaluate ? [evaluation] : []), dialogProgress, error);
    function closeDialog() {
      if (dialog.open) dialog.close();
      if (!dialogActive) return;
      dialogActive = false; selected.clear(); name.value = '';
      for (const input of evaluationModes) input.checked = false;
      renderedReview = null; state.review.detachMedia(); review.replaceChildren();
      controller.closeReview(); state.review.committed(null);
      if (active) controller.resetFilters();
      updateSelection();
    }
    function openDialog() {
      if (!active) return;
      error.textContent = ''; if (!dialog.open) dialog.showModal();
      dialogActive = true; drawSamples();
    }
    function updateSelection() {
      const chosen = state.captures.snapshot().filter(row => selected.has(row.id)), chosenGroups = new Map(chosen.map(row => [row.group_id, row]));
      selection.textContent = `${selected.size} muestras seleccionadas · elegí de 3 a 32 para confirmar.`;
      headerRefresh.disabled = dialogRefresh.disabled = busy || reviewPending();
      const locked = busy || needsRefresh || reviewPending();
      confirm.disabled = locked || selected.size < 3 || selected.size > 32 || !name.value.trim();
      merge.disabled = locked || chosenGroups.size < 2 || [...chosenGroups.values()].reduce((total, row) => total + row.group_size, 0) > 32;
      split.disabled = locked || !selected.size || chosenGroups.size !== 1 || selected.size >= [...chosenGroups.values()][0]?.group_size;
      unknown.disabled = locked || !selected.size;
      prepareEvaluation.disabled = locked || !selected.size || chosen.length !== selected.size || chosen.some(row => !row.review_available) || !evaluationModes.some(input => input.checked);
      for (const input of evaluationModes) input.disabled = locked;
      for (const input of sampleList.querySelectorAll('input')) input.disabled = locked || input.dataset.available !== 'true';
      for (const control of detail.querySelectorAll('button')) control.disabled = locked || control.id === 'voice-person-load' && catalogueStatus === 'loading';
      updateListeningButtons();
    }
    async function mutate(operation) {
      if (busy || needsRefresh || reviewPending() || !active) return; busy = true; error.textContent = notice.textContent = '';
      progress.textContent = dialogProgress.textContent = 'Aplicando cambios…'; updateSelection();
      try { needsRefresh = !await operation(); }
      catch { needsRefresh = true; }
      finally {
        busy = false;
        if (active) { progress.textContent = dialogProgress.textContent = ''; drawProfiles(); if (needsRefresh) error.textContent = notice.textContent = 'No se pudo confirmar el resultado. Actualizá antes de reintentar.'; updateSelection(); }
      }
    }
    async function refresh() {
      if (busy || reviewPending() || !active) return; busy = true;
      progress.textContent = dialogProgress.textContent = 'Actualizando…'; updateSelection();
      try { await controller.refresh(); await controller.settled(); if (active && ['profiles', 'captures'].every(kind => state[kind].status() === 'real')) { needsRefresh = false; error.textContent = notice.textContent = ''; } }
      finally { busy = false; if (active) { progress.textContent = dialogProgress.textContent = ''; drawProfiles(); updateSelection(); } }
    }
    name.addEventListener('input', updateSelection);
    form.addEventListener('submit', event => { event.preventDefault(); if (confirm.disabled) return; const ids = [...selected]; let draft = name.value.trim(); name.value = ''; mutate(() => { const pending = controller.confirm(ids, draft); draft = null; return pending; }); });
    function correct(action) { const ids = [...selected]; mutate(() => controller.correctGroups(ids, action, root.HomeRequestIdentity.newRequestId())); }
    function closeLink() {
      if (linkDialog?.open) linkDialog.close();
      linkDraft?.choices.clear(); linkDraft = null; linkDialog?.replaceChildren();
    }
    function clearPeople() {
      closeLink(); catalogue?.clear(); catalogue = null; catalogueStatus = 'waiting';
    }
    function bindingFor(row) {
      return {token:state.profiles.deliveryToken(),id:row.id,profileRevision:row.revision,linkRevision:row.person_link.revision};
    }
    function matchesBinding(binding) {
      return active && controller.includePerson && state.profiles.deliveryToken() === binding.token && binding.token !== null
        && profileId === binding.id && state.profiles.snapshot().some(row => row.id === binding.id
          && row.revision === binding.profileRevision && row.person_link.revision === binding.linkRevision);
    }
    function loadPeople(row) {
      if (!active || busy || catalogueStatus === 'loading') return;
      const binding = bindingFor(row); catalogueStatus = 'loading';
      const loading = container.querySelector('#voice-person-load');
      if (loading) { loading.textContent = 'Consultando personas…'; loading.disabled = true; }
      controller.loadPeople(accounts => {
        if (!matchesBinding(binding)) return;
        catalogue = new Map(accounts.map(account => [account.id,account])); catalogueStatus = 'real'; drawProfiles();
      }).then(() => {
        if (active && state.profiles.deliveryToken() === binding.token && catalogueStatus === 'loading') {
          catalogueStatus = 'unavailable'; drawProfiles();
        }
      });
    }
    function openLink(row) {
      const binding = bindingFor(row);
      if (!matchesBinding(binding) || busy || needsRefresh || catalogueStatus !== 'real' || !linkDialog) return;
      closeLink();
      const form = node('form', '', 'personas-form'), label = node('label', '', 'field'), select = node('select');
      const reference = node('p', '', 'row-meta'); reference.id = 'voice-person-reference';
      reference.setAttribute('role', 'status');
      select.id = 'voice-person-choice'; select.required = true;
      select.setAttribute('aria-describedby', reference.id); label.append(node('span', 'Persona del hogar'), select);
      const placeholder = node('option','Elegí una persona…'); placeholder.value = ''; placeholder.disabled = true; select.append(placeholder);
      const unlink = node('option','Sin persona asignada · desvincular'); unlink.value = 'unlink'; select.append(unlink);
      const accounts = [...catalogue.values()].filter(account => account.is_active), names = new Map(), choices = new Map();
      for (const account of accounts) { const name = normalizeName(account.name); names.set(name,(names.get(name)||0)+1); }
      const references = new Map();
      let chosen = row.person_link.user_id === null ? 'unlink' : '';
      accounts.forEach((account,index) => {
        const value = String(index + 1), count = names.get(normalizeName(account.name));
        const option = node('option', account.name + (count > 1 ? ' · Referencia: ' + account.id : ''));
        if (count > 1) references.set(value, { count, id: account.id });
        option.value = value; choices.set(value,account.id); select.append(option);
        if (account.id === row.person_link.user_id) chosen = value;
      });
      function showReference() {
        const selected = references.get(select.value);
        reference.textContent = selected
          ? `Hay ${selected.count} personas con este nombre. Confirmá por la referencia: ${selected.id}`
          : '';
      }
      select.value = chosen;
      select.addEventListener('change', showReference); showReference();
      linkDraft = {binding,choices};
      const save = button('Guardar vínculo',() => {},'btn primary'); save.type = 'submit'; form.append(label, reference, save);
      form.addEventListener('submit',event => {
        event.preventDefault();
        const draft = linkDraft;
        if (!draft || !matchesBinding(draft.binding) || busy || needsRefresh) { closeLink(); return; }
        let userId = select.value === 'unlink' ? null : draft.choices.get(select.value);
        if (userId === undefined) return;
        const original = draft.binding; closeLink();
        mutate(() => { const pending = controller.setPerson(original.id,userId,original.profileRevision,original.linkRevision); userId = null; return pending; });
      });
      linkDialog.replaceChildren(button('Cerrar', closeLink, 'btn quiet'),
        node('h2', 'Vincular voz y persona'), node('p', 'El vínculo organiza el perfil. No verifica la identidad ni concede permisos o consentimiento.', 'modal-description'), form);
      linkDialog.showModal();
    }
    function drawPerson(row) {
      if (!controller.includePerson) return;
      const link = row.person_link, account = catalogue?.get(link.user_id), known = catalogueStatus === 'real' && account?.is_active;
      const connection = node('div', '', 'personas-stack');
      const personName = link.user_id === null ? 'Sin persona asignada' : known ? account.name : 'Persona no disponible';
      const personLabel = node('span', personName, 'row-meta'); personLabel.id = 'voice-linked-person';
      connection.append(node('h3', 'Persona vinculada'), personLabel); detail.append(connection);
      if (catalogueStatus === 'real') {
        const edit = button(link.user_id === null ? 'Vincular persona' : 'Cambiar persona',() => openLink(row),'btn primary'); edit.id = 'voice-person-edit'; edit.disabled = busy || needsRefresh; detail.append(edit);
      } else {
        const load = button(catalogueStatus === 'loading' ? 'Consultando personas…' : 'Consultar personas',() => loadPeople(row)); load.id = 'voice-person-load'; load.disabled = busy || needsRefresh || catalogueStatus === 'loading'; detail.append(load);
      }
      if (catalogueStatus === 'waiting' && !busy && !needsRefresh) loadPeople(row);
    }
    function drawProfiles() {
      const token = state.profiles.deliveryToken();
      if (token !== linkToken) { clearPeople(); linkToken = token; }
      const rows = state.profiles.snapshot();
      if (!rows.some(row => row.id === profileId)) profileId = rows[0]?.id ?? null; list.replaceChildren(); profilePage.replaceChildren();
      if (!rows.length) list.append(node('p', message(state.profiles, 'Aún no hay perfiles confirmados.'), 'empty'));
      for (const row of rows) {
        const item = button('', () => { closeLink(); profileId = row.id; drawProfiles(); }, 'pick');
        item.dataset.voiceProfile = '';
        item.setAttribute('aria-pressed', String(profileId === row.id));
        const note = profileNote(row.id);
        item.append(node('span', row.name, 'row-title'),
          node('span', typeof note === 'string' && note ? 'Perfil confirmado · ' + note : 'Perfil confirmado', 'row-meta'));
        list.append(item);
      }
      if (controller.hasNext('profiles')) profilePage.append(button('Más perfiles', () => controller.next('profiles')));
      const row = rows.find(profile => profile.id === profileId); detail.replaceChildren();
      detail.hidden = !row;
      if (row) {
        const remove = button('Borrar perfil', () => mutate(() => controller.deleteProfile(row.id, row.revision)), 'btn danger');
        remove.disabled = busy || needsRefresh || reviewPending();
        detail.append(node('h2', row.name), node('p', `Confirmado el ${new Date(row.confirmed_at * 1000).toLocaleString()}`, 'row-meta'),
          node('p', 'El perfil se conserva hasta que lo borres. Su nombre no concede permisos.'));
        drawPerson(row); detail.append(remove);
      }
      state.profiles.committed(state.profiles.deliveryToken());
    }
    function reviewPending() { return requestedReview !== null || state.review.status() === 'loading'; }
    function updateListeningButtons() {
      const loading = reviewPending();
      for (const [id, { listen, read, available }] of listeningButtons) {
        listen.disabled = read.disabled = !available || busy || loading;
        listen.textContent = requestedReview?.id === id && requestedReview.variant === 'audio' ? 'Cargando audio…' : 'Escuchar';
        read.textContent = requestedReview?.id === id && requestedReview.variant === 'text' ? 'Cargando transcripción…' : 'Leer transcripción';
      }
    }
    function openReview(id,variant) {
      if (!active || !dialog.open || doc.hidden || busy || reviewPending()) return;
      const [row] = state.review.snapshot();
      if (row?.id !== id || variant === 'audio' && row.variant !== 'audio') {
        requestedReview = {id,variant}; startingReview = true; updateSelection();
        try { if (!(variant === 'text' ? controller.transcript(id) : controller.review(id))) requestedReview = null; }
        finally { startingReview = false; updateSelection(); }
      }
      review.scrollIntoView({ block: 'nearest', behavior: 'auto' });
    }
    function drawSamples() {
      const rows = state.captures.snapshot().slice().sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id));
      for (const id of selected) if (!rows.some(row => row.id === id)) selected.delete(id); sampleList.replaceChildren(); listeningButtons.clear();
      if (!rows.length) sampleList.append(node('p', message(state.captures, 'Ninguna muestra vigente coincide con estos filtros.'), 'empty'));
      const groups = new Map([...new Set(rows.map(row => row.group_id))].map((id, index) => [id, index + 1]));
      for (const row of rows) {
        const item = node('div', '', 'row review-sample'), choice = node('input'), copy = node('div', '', 'row-main'); choice.type = 'checkbox'; choice.checked = selected.has(row.id); choice.dataset.available = String(row.review_available);
        choice.setAttribute('aria-label', `Seleccionar muestra de ${new Date(row.created_at * 1000).toLocaleString()}`);
        choice.addEventListener('change', () => { if (choice.checked && selected.size < 32) selected.add(row.id); else { selected.delete(row.id); choice.checked = false; } updateSelection(); });
        copy.append(node('strong', new Date(row.created_at * 1000).toLocaleString(), 'row-title'),
          node('span', `${row.duration_s.toFixed(1)} s · Grupo ${groups.get(row.group_id)} · ${sampleRoom(row.room_id)}`, 'row-meta'));
        const listen = button('Escuchar', () => openReview(row.id,'audio'));
        const read = button('Leer transcripción', () => openReview(row.id,'text'));
        const sampleActions = node('div','','row-actions voice-sample-actions'); sampleActions.append(listen,read);
        listeningButtons.set(row.id, { listen, read, available: row.review_available });
        item.append(choice, copy, sampleActions); sampleList.append(item);
      } updateSelection();
    }
    function drawCaptures() {
      const rows = state.captures.snapshot(), grouped = new Map();
      for (const row of rows.slice().sort((a, b) => b.created_at - a.created_at)) { if (!grouped.has(row.group_id)) grouped.set(row.group_id, []); grouped.get(row.group_id).push(row); }
      groups.replaceChildren(); capturePage.replaceChildren();
      if (!rows.length) groups.append(node('p', message(state.captures, 'No hay muestras vigentes para revisar.'), 'empty'));
      for (const members of grouped.values()) {
        const row = members[0], card = node('article', '', 'row'), copy = node('div', '', 'row-main');
        card.dataset.voiceGroup = '';
        const labels = { suggested: 'Grupo sugerido', manual: 'Grupo revisado', unknown: 'Voz sin identificar' };
        copy.append(node('h3', labels[row.group_state], 'row-title'),
          node('p', `${members.length} de ${row.group_size} muestras · Última: ${new Date(row.created_at * 1000).toLocaleString()}`, 'row-meta'));
        card.append(copy, button('Revisar muestras', openDialog)); groups.append(card);
      }
      dialogPage.replaceChildren();
      if (controller.hasNext('captures')) {
        for (const page of [capturePage,dialogPage]) page.append(button('Más muestras', () => controller.next('captures')));
      }
      drawSamples(); state.captures.committed(state.captures.deliveryToken());
    }
    function drawReview() {
      const row = state.review.snapshot()[0], loading = state.review.status() === 'loading';
      if (!startingReview && (row || !loading)) requestedReview = null;
      review.setAttribute('aria-busy', String(loading));
      updateSelection();
      if (row && row === renderedReview) { state.review.committed(state.review.deliveryToken()); return; }
      state.review.detachMedia(); renderedReview = row ?? null; review.replaceChildren(node('h3', 'Escuchar muestra'));
      if (!row) {
        const status = node('p', loading ? requestedReview?.variant === 'text' ? 'Cargando transcripción…' : 'Cargando audio…'
          : message(state.review, 'Elegí una muestra para escucharla o leer su transcripción.'));
        status.setAttribute('role', 'status'); review.append(status);
      }
      else {
        const transcriptDetails = root.KzaVoicePrivacy.metadataDetails('transcript', row.transcript_metadata);
        review.append(node('p', sampleRoom(row.room_id), 'row-meta'));
        if (row.variant === 'audio') {
          const player = root.KzaVoicePrivacy.createSamplePlayer({ doc, store: state.review, row, preload: 'metadata', buttonClass: 'btn primary' });
          review.append(player,node('p', `Fragmento de ${row.clip_duration_s.toFixed(1)} s${row.clip_truncated ? ` de un segmento de ${row.duration_s.toFixed(1)} s` : ''}.`, 'row-meta'),
            ...root.KzaVoicePrivacy.metadataDetails('audio',row.audio_metadata).map(text => node('p',text,'row-meta')));
        }
        review.append(node('h4', 'Transcripción del segmento'),
          ...transcriptDetails.map(text => node('p', text, 'row-meta')),
          node('p', row.transcript || 'Sin transcripción disponible.'));
        if (row.variant === 'audio') review.append(node('p', 'El fragmento de audio y la transcripción pueden corresponder a ventanas y canales distintos.', 'row-meta'));
      }
      state.review.committed(state.review.deliveryToken());
    }
    const stops = [state.profiles.subscribe(drawProfiles), state.captures.subscribe(drawCaptures), state.review.subscribe(drawReview), filterView.stop, ...(typeof onProfileNotes === 'function' ? [onProfileNotes(() => { if (active) drawProfiles(); })] : [])], check = () => { controller.checkFilters(); for (const key of ['profiles', 'captures', 'review']) state[key].check(); };
    const onVisibility = () => { if (doc.hidden) controller.closeReview(); else check(); };
    const cancelDialog = event => { event.preventDefault(); closeDialog(); }, cancelLink = event => { event.preventDefault(); closeLink(); };
    const onDialogClosed = () => { if (!dialog.open) closeDialog(); };
    dialog.addEventListener('cancel', cancelDialog); dialog.addEventListener('close',onDialogClosed); linkDialog?.addEventListener('cancel',cancelLink); doc.addEventListener('visibilitychange', onVisibility); root.addEventListener('focus', check);
    drawProfiles(); drawCaptures(); drawReview();
    return () => { active = false; clearPeople(); linkToken = null; linkDialog?.removeEventListener('cancel',cancelLink); for (const stop of stops) stop(); dialog.removeEventListener('cancel', cancelDialog); dialog.removeEventListener('close',onDialogClosed); doc.removeEventListener('visibilitychange', onVisibility); root.removeEventListener('focus', check); closeDialog(); selected.clear(); name.value = ''; container.replaceChildren(); dialog.replaceChildren(); listeningButtons.clear(); roomNumbers.clear(); profileId = renderedReview = requestedReview = null; controller.leave(); for (const key of ['profiles', 'captures', 'review']) state[key].committed(null); };
  }
  root.NexoVoices = { mountVoices };
})(typeof window !== 'undefined' ? window : globalThis);
