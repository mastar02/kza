/* The evaluator owner supplies only granted metadata and one modality. */
(function (root) {
  'use strict';
  const LABELS = { audio_preview: 'Audio breve', transcript_full: 'Transcripción del segmento' };
  function mountEvaluation({ container, controller, getSession }) {
    const doc = container.ownerDocument, store = controller.store;
    let active = true, choosing = false, rendered = null, renderedToken = null, renderedStatus = null;
    let lastReading = null, lastGrants = null;
    const connection = root.NexoAccess.createConnection(doc, getSession);
    let unsubscribe = () => {}, unsubscribeStore = () => {};
    function node(tag, text = '', classes = '') {
      const value = doc.createElement(tag); value.textContent = text; value.className = classes; return value;
    }
    function button(text, handler, classes = 'btn') {
      const value = node('button', text, classes); value.type = 'button'; value.addEventListener('click', handler); return value;
    }
    const refresh = button('Actualizar permisos de muestra', () => { if (active) void controller.refresh(); });
    const status = node('p', '', 'row-meta'); status.setAttribute('role', 'status'); status.dataset.evaluationStatus = '';
    const form = node('form', '', 'detail'), choices = node('select'), modes = node('select');
    choices.id = 'evaluation-fingerprint'; choices.name = 'fingerprint_id';
    modes.id = 'evaluation-modality'; modes.name = 'modality';
    const choiceLabel = node('label', '', 'field'), modeLabel = node('label', '', 'field');
    choiceLabel.append(node('span', 'Muestra autorizada'), choices);
    modeLabel.append(node('span', 'Qué querés consultar'), modes);
    const deadline = node('p', '', 'row-meta'), load = node('button', 'Consultar muestra', 'btn primary'); load.type = 'submit';
    const actions = node('div', '', 'row-actions'); actions.append(load);
    const formHeading = node('div', '', 'sistema-toolbar');
    formHeading.append(node('h2', 'Elegir muestra'), refresh);
    deadline.id = 'sistema-evaluation-deadline';
    choices.setAttribute('aria-describedby', deadline.id);
    form.append(formHeading, status, choiceLabel, modeLabel, deadline, actions);
    const content = node('section', '', 'detail'); content.setAttribute('aria-label', 'Muestra de evaluación');
    content.dataset.evaluationSample = '';
    function metadataText(text) {
      return text.replaceAll('STT', 'reconocimiento de voz').replace(' Hz', ' hercios')
        .replace('Estado DSP desconocido.', 'No se conoce el procesamiento del audio.')
        .replace('DSP declarado; aplicación sin verificar.', 'Se informó el procesamiento del audio, pero no se confirmó su aplicación.')
        .replace('Estado DSP observado.', 'Se comprobó el procesamiento del audio.');
    }
    function clearContent() {
      store.detachMedia(); content.replaceChildren(); rendered = renderedToken = renderedStatus = null; store.committed(null);
    }
    function renderSample() {
      if (!active) return;
      const row = store.snapshot()[0] || null, token = store.deliveryToken();
      content.setAttribute('aria-busy', String(store.status() === 'loading'));
      if (row === rendered && token === renderedToken && (row || renderedStatus === store.status())) return;
      clearContent();
      if (!row) {
        const text = { loading: 'Consultando la modalidad elegida…', expired: 'La muestra venció. Actualizá los permisos de muestra.',
          unavailable: 'La muestra ya no está disponible o no se pudo consultar.', closed: 'La consulta se cerró.' }[store.status()]
          || 'Elegí una muestra y tocá Consultar muestra.';
        const message = node('p', text, 'empty'); message.setAttribute('role', 'status');
        renderedStatus = store.status(); content.append(message); return;
      }
      rendered = row; renderedToken = token;
      content.append(node('h2', LABELS[row.modality]),
        node('p', `Consulta vigente hasta ${new Date(row.expires_at * 1000).toLocaleString()}`, 'row-meta'));
      if (row.modality === 'audio_preview') {
        const details = root.KzaVoicePrivacy.metadataDetails('audio', row.audio_metadata);
        const player = root.KzaVoicePrivacy.createSamplePlayer({ doc, store, row, preload: 'none', buttonClass: 'btn primary' });
        content.append(player, node('p', `Audio breve · ${row.clip_duration_s.toFixed(1)} s`, 'row-meta'));
        content.append(...details.map(text => node('p', metadataText(text), 'row-meta')));
      } else {
        const details = root.KzaVoicePrivacy.metadataDetails('transcript', row.transcript_metadata);
        const transcript = node('pre', row.transcript, 'evaluation-transcript');
        content.append(transcript);
        content.append(...details.map(text => node('p', metadataText(text), 'row-meta')));
      }
      // The real receipt remains outstanding while these nodes or a decoder own the sample.
      store.committed(token);
    }
    function render() {
      if (!active) return;
      const state = controller.snapshot(), focused = doc.activeElement;
      const choiceFocus = focused === choices, modeFocus = focused === modes;
      if (state.status === 'real' && state.grants !== lastGrants) {
        lastReading = Date.now(); lastGrants = state.grants;
      }
      connection.update(lastReading, state.status);
      choices.replaceChildren(); modes.replaceChildren();
      if (!state.grants.length) {
        const option = node('option', state.status === 'real' ? 'Sin muestras autorizadas' : 'Lista no disponible'); option.value = ''; choices.append(option);
      }
      for (let index = 0; index < state.grants.length; index++) {
        const option = node('option', `Muestra ${index + 1}`);
        option.value = String(index + 1); choices.append(option);
      }
      const index = state.grants.findIndex(value => value.fingerprint_id === state.selected?.fingerprint_id);
      choices.value = index >= 0 ? String(index + 1) : '';
      const grant = state.grants[index];
      for (const mode of grant?.modalities || []) { const option = node('option', mode === 'transcript_full' ? 'Transcripción' : LABELS[mode]); option.value = mode; modes.append(option); }
      modes.value = state.selected?.modality || '';
      const locked = !state.ready || state.busy || choosing;
      choices.disabled = modes.disabled = locked || !grant; load.disabled = locked || !state.selected;
      load.textContent = store.status() === 'loading' ? 'Consultando muestra…' : 'Consultar muestra';
      refresh.disabled = !state.ready || state.busy || choosing;
      status.textContent = state.status === 'real' ? (state.grants.length ? '' : 'No hay muestras autorizadas para esta sesión.')
        : { waiting: 'Esperando la conexión…', loading: 'Consultando los permisos de muestra…',
          expired: 'Los permisos de muestra cambiaron. Actualizando…',
          unavailable: 'No pudimos consultar los permisos de muestra. Usá Actualizar permisos de muestra.' }[state.status] || 'Consulta no disponible.';
      deadline.textContent = state.selected ? `Permiso vigente hasta ${new Date(state.selected.expires_at * 1000).toLocaleString()}` : '';
      if (choiceFocus && !choices.disabled) choices.focus(); else if (modeFocus && !modes.disabled) modes.focus();
      renderSample();
    }
    async function choose() {
      if (!active || choosing) return;
      const state = controller.snapshot();
      const grant = state.grants[choices.selectedIndex];
      if (!grant) return;
      const modality = grant.modalities.includes(modes.value) ? modes.value : grant.modalities[0];
      choosing = true; render();
      try { await controller.choose(grant.fingerprint_id, modality); }
      finally { choosing = false; if (active) render(); }
    }
    choices.addEventListener('change', choose); modes.addEventListener('change', choose);
    const onVisibility = () => { if (doc.hidden) controller.clearSample(); else store.check(); };
    form.addEventListener('submit', event => {
      event.preventDefault(); const state = controller.snapshot();
      if (active && !doc.hidden && state.ready && !state.busy && !choosing && state.selected) controller.load();
    });
    try {
      container.replaceChildren(connection.panel, form, content);
      unsubscribe = controller.subscribe(render); unsubscribeStore = store.subscribe(render);
      render();
      doc.addEventListener('visibilitychange', onVisibility);
    } catch (error) {
      active = false; unsubscribe(); unsubscribeStore(); clearContent(); container.replaceChildren(); throw error;
    }
    return () => {
      if (!active) return;
      active = false; doc.removeEventListener('visibilitychange', onVisibility);
      unsubscribe(); unsubscribeStore(); clearContent(); container.replaceChildren();
    };
  }
  const api = { mountEvaluation };
  root.NexoEvaluation = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
