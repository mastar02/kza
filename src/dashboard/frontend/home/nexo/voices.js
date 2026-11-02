/* NEXO presentation over the existing, expiring voice stores. */
(function (root) {
  'use strict';
  function mountVoices({ container, dialog, controller, icon }) {
    const doc = container.ownerDocument, state = controller.state, selected = new Set();
    let active = true, profileId = null, renderedAudio = null, busy = false, needsRefresh = false;
    function node(tag, text = '', className = '') { const value = doc.createElement(tag); value.textContent = text; value.className = className; return value; }
    function button(text, callback, className = 'identity-secondary') { const value = node('button', text, className); value.type = 'button'; value.addEventListener('click', callback); return value; }
    function message(store, empty) { return ({ waiting: 'Esperando conexión…', loading: 'Consultando…', unavailable: 'No se pudo consultar. Volvé a actualizar.', expired: 'La consulta venció. Actualizá para ver los datos vigentes.', closed: 'La sesión se cerró.' })[store.status()] || empty; }
    const header = node('header', '', 'identity-heading'), title = node('div'), headerActions = node('div', '', 'identity-heading-side');
    title.append(node('p', 'IDENTIDAD QUE SE CONSTRUYE', 'eyebrow'), node('h2', 'Voces que tienen nombre.'), node('p', 'Revisá las muestras locales y confirmá un perfil vocal.'));
    headerActions.append(button('Actualizar', refresh), button('Revisar muestras', openDialog, 'identity-primary')); header.append(title, headerActions);
    const metrics = node('div', '', 'identity-metrics'), counts = [node('strong', '—'), node('strong', '—'), node('strong', '—')];
    ['perfiles en esta página', 'muestras en esta página', 'grupos visibles'].forEach((label, index) => { const item = node('div'); item.append(icon(index ? 'clock' : 'mic'), counts[index], node('span', label)); metrics.append(item); });
    const workspace = node('div', '', 'voice-workspace'), profiles = node('section', '', 'voice-list-panel panel'), list = node('div'), profilePage = node('div', '', 'voice-list-foot');
    list.id = 'voice-profile-list'; profiles.append(node('h3', 'Perfiles vocales'), list, profilePage); profiles.setAttribute('aria-label', 'Perfiles confirmados');
    const detail = node('section', '', 'voice-detail-panel panel'), guide = node('aside', '', 'voice-guide panel'); detail.setAttribute('aria-label', 'Detalle del perfil vocal');
    guide.append(icon('mic'), node('p', 'PASO A PASO', 'eyebrow'), node('h3', 'Una voz, un perfil.'));
    const steps = node('ol');
    for (const [label, text] of [['Escuchá', 'Revisá el audio y la transcripción antes de elegir las muestras.'], ['Confirmá', 'Elegí entre 3 y 32 muestras de la misma voz y asigná un nombre.'], ['Revisá', 'Los grupos y las similitudes orientan la revisión; no verifican identidad.']]) { const item = node('li'), copy = node('div'); copy.append(node('strong', label), node('p', text)); item.append(copy); steps.append(item); }
    guide.append(steps, node('p', 'Un nombre vocal no concede permisos ni acceso a información personal.', 'voice-guide-note')); workspace.append(profiles, detail, guide);
    const queue = node('section', '', 'capture-queue'), groups = node('div', '', 'capture-group-grid'), capturePage = node('div', '', 'voice-list-foot');
    queue.append(node('h3', 'Muestras por revisar'), node('p', 'Más recientes primero · vencen con su captura original, como máximo a las 48 horas.', 'identity-muted'), groups, capturePage);
    const notice = node('p', '', 'identity-form-error'); notice.setAttribute('role', 'alert'); container.replaceChildren(header, metrics, workspace, queue, notice);
    const sampleList = node('div', '', 'nexo-voice-samples'), review = node('section', '', 'nexo-voice-review'), form = node('form', '', 'identity-form'); review.setAttribute('aria-label', 'Escuchar muestra');
    const label = node('label', 'Nombre del perfil'), name = node('input'), selection = node('p', '', 'review-selection'), error = node('p', '', 'identity-form-error'); error.setAttribute('role', 'alert');
    name.type = 'text'; name.maxLength = 64; name.autocomplete = 'off'; name.required = true; label.append(name);
    const confirm = button('Confirmar perfil', () => {}, 'identity-primary'); confirm.type = 'submit';
    const merge = button('Unir grupos', () => correct('merge')), split = button('Separar muestras', () => correct('split')), unknown = button('Dejar como desconocido', () => correct('unknown'));
    const corrections = node('div', '', 'review-corrections'); corrections.append(merge, split, unknown);
    const modal = node('div', '', 'nexo-voice-modal'); modal.append(sampleList, review); form.append(selection, label, confirm);
    dialog.replaceChildren(button('Cerrar', closeDialog), button('Actualizar', refresh), node('h2', 'Revisá antes de confirmar.'), node('p', 'El perfil confirmado se conserva hasta que lo borres. El audio y la transcripción siguen venciendo con su captura original.', 'modal-description'), modal, corrections,
      node('p', 'Unir incluye todas las muestras de los grupos elegidos, también las de otras páginas. Separar crea otro grupo sólo con las muestras marcadas.', 'relationship-note'), form, error);
    function closeDialog() { if (dialog.open) dialog.close(); selected.clear(); name.value = ''; renderedAudio = null; state.review.detachMedia(); review.replaceChildren(); controller.closeReview(); state.review.committed(null); updateSelection(); }
    function openDialog() { if (!active) return; error.textContent = ''; if (!dialog.open) dialog.showModal(); drawSamples(); }
    function updateSelection() {
      const chosen = state.captures.snapshot().filter(row => selected.has(row.id)), chosenGroups = new Map(chosen.map(row => [row.group_id, row]));
      selection.textContent = `${selected.size} muestras seleccionadas · elegí de 3 a 32 para confirmar.`;
      const locked = busy || needsRefresh;
      confirm.disabled = locked || selected.size < 3 || selected.size > 32 || !name.value.trim();
      merge.disabled = locked || chosenGroups.size < 2 || [...chosenGroups.values()].reduce((total, row) => total + row.group_size, 0) > 32;
      split.disabled = locked || !selected.size || chosenGroups.size !== 1 || selected.size >= [...chosenGroups.values()][0]?.group_size;
      unknown.disabled = locked || !selected.size;
      for (const input of sampleList.querySelectorAll('input')) input.disabled = locked || input.dataset.available !== 'true';
      for (const control of detail.querySelectorAll('button')) control.disabled = locked;
    }
    async function mutate(operation) {
      if (busy || needsRefresh || !active) return; busy = true; error.textContent = notice.textContent = ''; updateSelection();
      try { needsRefresh = !await operation(); }
      catch { needsRefresh = true; }
      finally {
        busy = false;
        if (active) { if (needsRefresh) error.textContent = notice.textContent = 'No se pudo confirmar el resultado. Actualizá antes de reintentar.'; updateSelection(); }
      }
    }
    async function refresh() {
      if (busy || !active) return; busy = true; updateSelection();
      try { await controller.refresh(); await controller.settled(); if (active && ['profiles', 'captures'].every(kind => state[kind].status() === 'real')) { needsRefresh = false; error.textContent = notice.textContent = ''; } }
      finally { busy = false; if (active) updateSelection(); }
    }
    name.addEventListener('input', updateSelection);
    form.addEventListener('submit', event => { event.preventDefault(); if (confirm.disabled) return; const ids = [...selected]; let draft = name.value.trim(); name.value = ''; mutate(() => { const pending = controller.confirm(ids, draft); draft = null; return pending; }); });
    function correct(action) { const ids = [...selected]; mutate(() => controller.correctGroups(ids, action, root.HomeRequestIdentity.newRequestId())); }
    function drawProfiles() {
      const rows = state.profiles.snapshot(); counts[0].textContent = state.profiles.status() === 'real' ? String(rows.length) : '—';
      if (!rows.some(row => row.id === profileId)) profileId = rows[0]?.id ?? null; list.replaceChildren(); profilePage.replaceChildren();
      if (!rows.length) list.append(node('p', message(state.profiles, 'Aún no hay perfiles confirmados.'), 'identity-empty'));
      for (const row of rows) { const item = button('', () => { profileId = row.id; drawProfiles(); }, 'voice-profile-row'), copy = node('span', '', 'voice-profile-info'); item.dataset.voiceProfile = row.id; item.classList.toggle('selected', profileId === row.id); item.setAttribute('aria-pressed', String(profileId === row.id)); copy.append(node('strong', row.name), node('span', 'Perfil confirmado')); item.append(icon('mic'), copy); list.append(item); }
      if (controller.hasNext('profiles')) profilePage.append(button('Más perfiles', () => controller.next('profiles')));
      const row = rows.find(profile => profile.id === profileId); detail.replaceChildren();
      if (!row) detail.append(node('p', 'Elegí un perfil confirmado para revisarlo.', 'identity-empty'));
      else {
        const remove = button('Borrar perfil', () => mutate(() => controller.deleteProfile(row.id, row.revision)), 'identity-danger'); remove.disabled = busy || needsRefresh;
        const signature = node('div', '', 'voice-signature'), wave = node('div', '', 'voice-wave large'); wave.setAttribute('aria-hidden', 'true');
        for (let index = 0; index < 41; index++) { const bar = node('i'); bar.style.setProperty('--bar', `${15 + Math.abs(Math.sin(index * 1.73) * Math.cos(index * .41)) * 75}%`); wave.append(bar); }
        signature.append(wave, node('span', 'PERFIL VOCAL · SÍMBOLO VISUAL'));
        detail.append(node('span', 'CONFIRMADO', 'identity-badge mint-badge'), node('h3', row.name), node('p', `Confirmado el ${new Date(row.confirmed_at * 1000).toLocaleString()}`, 'voice-detail-description'), signature,
          node('p', 'Se conserva hasta que lo borres. El nombre vocal no concede permisos. La vinculación con una persona se revisa por separado.', 'voice-guide-note'), remove);
      }
      state.profiles.committed(state.profiles.deliveryToken());
    }
    function drawSamples() {
      const rows = state.captures.snapshot().slice().sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id));
      for (const id of selected) if (!rows.some(row => row.id === id)) selected.delete(id); sampleList.replaceChildren();
      if (!rows.length) sampleList.append(node('p', message(state.captures, 'No hay muestras vigentes.'), 'identity-empty'));
      for (const row of rows) {
        const item = node('div', '', 'review-sample'), choice = node('input'), copy = node('div', '', 'sample-content'); choice.type = 'checkbox'; choice.checked = selected.has(row.id); choice.dataset.available = String(row.review_available);
        choice.setAttribute('aria-label', `Seleccionar muestra de ${new Date(row.created_at * 1000).toLocaleString()}`);
        choice.addEventListener('change', () => { if (choice.checked && selected.size < 32) selected.add(row.id); else { selected.delete(row.id); choice.checked = false; } updateSelection(); });
        copy.append(node('strong', new Date(row.created_at * 1000).toLocaleString()), node('span', `${row.duration_s.toFixed(1)} s · grupo ${row.group_id.slice(0, 8)}`));
        const listen = button('Escuchar', () => { controller.review(row.id); drawSamples(); }); listen.disabled = !row.review_available || busy || state.review.status() === 'loading'; item.dataset.captureId = row.id; item.append(choice, copy, listen); sampleList.append(item);
      } updateSelection();
    }
    function drawCaptures() {
      const rows = state.captures.snapshot(), grouped = new Map();
      for (const row of rows.slice().sort((a, b) => b.created_at - a.created_at)) { if (!grouped.has(row.group_id)) grouped.set(row.group_id, []); grouped.get(row.group_id).push(row); }
      const real = state.captures.status() === 'real'; counts[1].textContent = real ? String(rows.length) : '—'; counts[2].textContent = real ? String(grouped.size) : '—'; groups.replaceChildren(); capturePage.replaceChildren();
      if (!rows.length) groups.append(node('p', message(state.captures, 'No hay muestras vigentes para revisar.'), 'identity-empty'));
      for (const members of grouped.values()) { const row = members[0], card = node('article', '', 'capture-card panel'); card.dataset.voiceGroup = row.group_id; card.append(node('span', { suggested: 'GRUPO SUGERIDO', manual: 'AGRUPACIÓN MANUAL', unknown: 'DESCONOCIDO' }[row.group_state], 'identity-badge purple'), node('h3', `${members.length} de ${row.group_size} muestras visibles`), node('p', `Última captura: ${new Date(row.created_at * 1000).toLocaleString()}`), button('Revisar muestras →', openDialog)); groups.append(card); }
      if (controller.hasNext('captures')) capturePage.append(button('Más muestras', () => controller.next('captures'))); drawSamples(); state.captures.committed(state.captures.deliveryToken());
    }
    function drawReview() {
      const row = state.review.snapshot()[0]; drawSamples();
      if (row && row.audio_url === renderedAudio) { state.review.committed(state.review.deliveryToken()); return; }
      state.review.detachMedia(); renderedAudio = row?.audio_url ?? null; review.replaceChildren(node('h3', 'Escuchar muestra'));
      if (!row) review.append(node('p', message(state.review, 'Elegí una muestra para escucharla.')));
      else { const audio = doc.createElement('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = row.audio_url; audio.setAttribute('aria-label', 'Audio de la muestra seleccionada'); review.append(audio, node('p', row.room_id ? `Habitación: ${row.room_id}` : 'Habitación sin registrar', 'identity-muted'), node('p', `Fragmento de ${row.clip_duration_s.toFixed(1)} s${row.clip_truncated ? ` de un segmento de ${row.duration_s.toFixed(1)} s` : ''}.`, 'identity-muted'), node('h4', 'Transcripción del segmento'), node('p', row.transcript || 'Sin transcripción disponible.')); state.review.attachMedia(audio); }
      state.review.committed(state.review.deliveryToken());
    }
    const stops = [state.profiles.subscribe(drawProfiles), state.captures.subscribe(drawCaptures), state.review.subscribe(drawReview)], check = () => { for (const key of ['profiles', 'captures', 'review']) state[key].check(); };
    const cancelDialog = event => { event.preventDefault(); closeDialog(); }; dialog.addEventListener('cancel', cancelDialog); doc.addEventListener('visibilitychange', check); root.addEventListener('focus', check);
    drawProfiles(); drawCaptures(); drawReview();
    return () => { active = false; for (const stop of stops) stop(); dialog.removeEventListener('cancel', cancelDialog); doc.removeEventListener('visibilitychange', check); root.removeEventListener('focus', check); closeDialog(); selected.clear(); name.value = ''; container.replaceChildren(); dialog.replaceChildren(); profileId = renderedAudio = null; controller.leave(); for (const key of ['profiles', 'captures', 'review']) state[key].committed(null); };
  }
  root.NexoVoices = { mountVoices };
})(typeof window !== 'undefined' ? window : globalThis);
