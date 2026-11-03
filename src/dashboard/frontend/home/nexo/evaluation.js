/* The evaluator owner supplies only granted metadata and one modality. */
(function (root) {
  'use strict';
  const LABELS = { audio_preview: 'Audio breve', transcript_full: 'Transcripción del segmento' };
  function mountEvaluation({ container, controller }) {
    const doc = container.ownerDocument, store = controller.store;
    let active = true, choosing = false, rendered = null, renderedToken = null, renderedStatus = null;
    let unsubscribe = () => {}, unsubscribeStore = () => {};
    function node(tag, text = '', classes = '') {
      const value = doc.createElement(tag); value.textContent = text; value.className = classes; return value;
    }
    function button(text, handler, classes = 'identity-secondary') {
      const value = node('button', text, classes); value.type = 'button'; value.addEventListener('click', handler); return value;
    }
    const heading = node('header', '', 'identity-heading'), copy = node('div');
    copy.append(node('p', 'ACCESO TEMPORAL', 'eyebrow'), node('h2', 'Sólo las muestras concedidas.'),
      node('p', 'Elegí una concesión vigente y la modalidad autorizada. Cada consulta vence con su muestra.'));
    const refresh = button('Actualizar concesiones', () => { if (active) void controller.refresh(); });
    heading.append(copy, refresh);
    const status = node('p', '', 'identity-muted'); status.setAttribute('role', 'status'); status.dataset.evaluationStatus = '';
    const form = node('form', '', 'identity-form panel'), choices = node('select'), modes = node('select');
    choices.id = 'evaluation-fingerprint'; choices.name = 'fingerprint_id';
    modes.id = 'evaluation-modality'; modes.name = 'modality';
    for (const select of [choices, modes]) { select.style.width = '100%'; select.style.maxWidth = '100%'; select.style.minWidth = '0'; }
    const choiceLabel = node('label', 'Muestra concedida'), modeLabel = node('label', 'Modalidad');
    choiceLabel.htmlFor = choices.id; modeLabel.htmlFor = modes.id;
    choiceLabel.append(choices); modeLabel.append(modes);
    const deadline = node('p', '', 'identity-muted'), load = node('button', 'Consultar muestra', 'identity-primary'); load.type = 'submit';
    form.append(choiceLabel, modeLabel, deadline, load);
    const content = node('section', '', 'voice-detail-panel panel'); content.setAttribute('aria-label', 'Muestra de evaluación');
    content.dataset.evaluationSample = '';
    const note = node('p', 'Una concesión no garantiza que el audio o el texto sigan disponibles. Este acceso no permite gestionar voces, personas ni el hogar.', 'identity-contract-note');
    function clearContent() {
      store.detachMedia(); content.replaceChildren(); rendered = renderedToken = renderedStatus = null; store.committed(null);
    }
    function renderSample() {
      if (!active) return;
      const row = store.snapshot()[0] || null, token = store.deliveryToken();
      if (row === rendered && token === renderedToken && (row || renderedStatus === store.status())) return;
      clearContent();
      if (!row) {
        const text = { loading: 'Consultando la modalidad elegida…', expired: 'La muestra venció. Actualizá las concesiones.',
          unavailable: 'La muestra ya no está disponible o no se pudo consultar.', closed: 'La consulta se cerró.' }[store.status()]
          || 'Elegí una muestra y tocá Consultar muestra.';
        renderedStatus = store.status(); content.append(node('p', text, 'identity-empty')); return;
      }
      rendered = row; renderedToken = token;
      content.append(node('p', LABELS[row.modality], 'eyebrow'), node('p', 'Muestra autorizada para esta sesión.', 'identity-muted'),
        node('p', `Consulta vigente hasta ${new Date(row.expires_at * 1000).toLocaleString()}`, 'identity-muted'));
      if (row.modality === 'audio_preview') {
        const details = root.KzaVoicePrivacy.metadataDetails('audio', row.audio_metadata);
        const audio = doc.createElement('audio'); audio.controls = true; audio.preload = 'none';
        audio.style.width = '100%'; audio.style.maxWidth = '100%'; audio.style.minWidth = '0';
        audio.setAttribute('aria-label', 'Escuchar audio breve de evaluación'); audio.src = row.audio_url;
        store.attachMedia(audio); content.append(audio, node('p', `Audio breve · ${row.clip_duration_s.toFixed(1)} s`, 'identity-muted'));
        content.append(...details.map(text => node('p', text, 'identity-muted')));
      } else {
        const details = root.KzaVoicePrivacy.metadataDetails('transcript', row.transcript_metadata);
        const transcript = node('pre', row.transcript, 'evaluation-transcript');
        transcript.style.whiteSpace = 'pre-wrap'; transcript.style.overflowWrap = 'anywhere'; content.append(transcript);
        content.append(...details.map(text => node('p', text, 'identity-muted')));
      }
      // The real receipt remains outstanding while these nodes or a decoder own the sample.
      store.committed(token);
    }
    function render() {
      if (!active) return;
      const state = controller.snapshot(), focused = doc.activeElement;
      const choiceFocus = focused === choices, modeFocus = focused === modes;
      choices.replaceChildren(); modes.replaceChildren();
      if (!state.grants.length) {
        const option = node('option', state.status === 'real' ? 'Sin concesiones vigentes' : 'Catálogo no disponible'); option.value = ''; choices.append(option);
      }
      for (const [index, grant] of state.grants.entries()) {
        const option = node('option', `Muestra ${index + 1} · vence ${new Date(grant.expires_at * 1000).toLocaleString()}`);
        option.value = grant.fingerprint_id; choices.append(option);
      }
      choices.value = state.selected?.fingerprint_id || '';
      const grant = state.grants.find(value => value.fingerprint_id === state.selected?.fingerprint_id);
      for (const mode of grant?.modalities || []) { const option = node('option', LABELS[mode]); option.value = mode; modes.append(option); }
      modes.value = state.selected?.modality || '';
      const locked = !state.ready || state.busy || choosing;
      choices.disabled = modes.disabled = locked || !grant; load.disabled = locked || !state.selected;
      refresh.disabled = !state.ready || state.busy || choosing;
      status.textContent = state.status === 'real' ? (state.grants.length ? `${state.grants.length} ${state.grants.length === 1 ? 'concesión vigente' : 'concesiones vigentes'} en esta sesión.` : 'No hay concesiones vigentes para esta sesión.')
        : { waiting: 'Esperando el canal privado…', loading: 'Consultando concesiones…', expired: 'Las concesiones cambiaron. Actualizando…',
          unavailable: 'No pudimos consultar las concesiones. Usá Actualizar concesiones.' }[state.status] || 'Consulta no disponible.';
      deadline.textContent = state.selected ? `Concesión vigente hasta ${new Date(state.selected.expires_at * 1000).toLocaleString()}` : '';
      if (choiceFocus && !choices.disabled) choices.focus(); else if (modeFocus && !modes.disabled) modes.focus();
      renderSample();
    }
    async function choose() {
      if (!active || choosing) return;
      const state = controller.snapshot();
      const grant = state.grants.find(value => value.fingerprint_id === choices.value);
      if (!grant) return;
      const modality = grant.modalities.includes(modes.value) ? modes.value : grant.modalities[0];
      choosing = true; render();
      try { await controller.choose(grant.fingerprint_id, modality); }
      finally { choosing = false; if (active) render(); }
    }
    choices.addEventListener('change', choose); modes.addEventListener('change', choose);
    form.addEventListener('submit', event => {
      event.preventDefault(); const state = controller.snapshot();
      if (active && state.ready && !state.busy && !choosing && state.selected) controller.load();
    });
    try {
      container.replaceChildren(heading, status, form, content, note);
      unsubscribe = controller.subscribe(render); unsubscribeStore = store.subscribe(render);
      render();
    } catch (error) {
      active = false; unsubscribe(); unsubscribeStore(); clearContent(); container.replaceChildren(); throw error;
    }
    return () => {
      if (!active) return;
      active = false; unsubscribe(); unsubscribeStore(); clearContent(); container.replaceChildren();
    };
  }
  const api = { mountEvaluation };
  root.NexoEvaluation = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
