/* NEXO administration panel for the voice stage (U3b-4a-3), inside Voces: the G1 switch, the
 * consents list with the verified profile, the multi-profile notice, reopen and resume. DOM only;
 * it never shows scores, attempts, windows, rooms or phrases (none exist in what it reads). */
(function (root) {
  'use strict';
  const COPY = {
    off: 'Se dejan de atribuir propuestas por voz y se cierran las ventanas de verificación abiertas. No cambia la configuración ni borra inscripciones.',
    on: 'Las personas inscriptas y verificadas vuelven a recibir propuestas por voz.',
    reopen: 'va a poder volver a verificar su voz. Se borran los intentos fallidos contados. Si se había dado de baja mientras estaba bloqueada, sigue de baja.',
    resume: 'Ese perfil vuelve a recibir propuestas por voz. La pausa se activó porque varias propuestas fueron devueltas como «No es mío».',
    listOff: 'La lista de inscripciones no está disponible: la configuración del servidor no habilita la memoria por voz.',
    truncated: 'Se muestran las primeras 256 inscripciones.',
    empty: 'Todavía nadie pidió inscribirse.',
  };

  function mountVoiceConsents({ container, confirmDialog, owner, icon }) {
    const doc = container.ownerDocument, Owner = root.NexoVoiceConsentsOwner;
    let active = true, lastAnnouncement = 0, confirmResolve = null;
    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }
    function button(text, key, callback, className = 'identity-secondary') {
      const element = node('button', className, text); element.type = 'button'; element.dataset.key = key;
      element.addEventListener('click', () => { if (element.getAttribute('aria-disabled') !== 'true') callback(); });
      return element;
    }
    const section = node('section', 'voice-attribution panel'); section.setAttribute('aria-labelledby', 'voice-attribution-title');
    const title = node('h3', '', 'Memoria por voz'); title.id = 'voice-attribution-title'; title.tabIndex = -1;
    const disclosure = node('p', 'voice-attribution-note', Owner.VOICES_DISCLOSURE);
    const freshness = node('p', 'voice-attribution-note'); freshness.id = 'voice-attribution-freshness';
    const sessionNotice = node('p', 'voice-attribution-note'); sessionNotice.setAttribute('role', 'status');
    const live = node('p', 'voice-attribution-note'); live.id = 'voice-attribution-live'; live.setAttribute('aria-live', 'polite');
    const switchBox = node('div', 'voice-attribution-block'); switchBox.id = 'voice-attribution-switch';
    const listBox = node('div', 'voice-attribution-block'); listBox.id = 'voice-attribution-consents';
    const multipleBox = node('div', 'voice-attribution-block'); multipleBox.id = 'voice-attribution-multiple';
    const suspendedBox = node('div', 'voice-attribution-block'); suspendedBox.id = 'voice-attribution-suspended';
    const refresh = button('Actualizar', 'voice-attribution-refresh', () => { void owner.reload(); });
    const head = node('div', 'voice-attribution-head'); head.append(title, refresh);
    section.append(head, disclosure, freshness, sessionNotice, live, switchBox, listBox, multipleBox, suspendedBox);
    container.replaceChildren(section);

    function dialog(heading, texts, acceptText, acceptClass, invoker) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        const h = node('h2', '', heading); h.id = 'voice-attribution-confirm-title';
        const row = node('div', 'modal-actions memory-confirm-actions');
        const cancelButton = button('Cancelar', 'voice-admin-cancel', () => finish(false)), accept = button(acceptText, 'voice-admin-accept', () => finish(true), acceptClass);
        row.append(cancelButton, accept);
        confirmDialog.replaceChildren(h, ...texts.map(text => node('p', 'modal-description', text)), row);
        confirmDialog.setAttribute('aria-labelledby', 'voice-attribution-confirm-title');
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          const back = [...container.querySelectorAll('[data-key]')].find(element => element.dataset.key === invoker);
          if (active) (back || title).focus({ preventScroll: true });
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish;
        if (!confirmDialog.open) confirmDialog.showModal();
        cancelButton.focus();
      });
    }
    async function toggle(enabled) {
      const accepted = await dialog(enabled ? 'Encender la memoria por voz' : 'Apagar la memoria por voz', [enabled ? COPY.on : COPY.off],
        enabled ? 'Encender' : 'Apagar', enabled ? 'identity-primary' : 'identity-danger', 'voice-switch');
      if (accepted && active) await owner.setSwitch(enabled, { confirmed: true });
    }
    async function reopen(row) {
      const accepted = await dialog('Habilitar un nuevo intento', [row.name + ' ' + COPY.reopen], 'Habilitar', 'identity-primary', 'reopen:' + row.person_id);
      if (accepted && active) await owner.reopen(row.person_id, { confirmed: true });
    }
    async function resume(profileId, label) {
      const accepted = await dialog('Reanudar propuestas por voz', [label + '. ' + COPY.resume], 'Reanudar', 'identity-primary', 'resume:' + profileId);
      if (accepted && active) await owner.resume(profileId, { confirmed: true });
    }
    function rebuild(box, signature, build) {
      if (box.dataset.signature === signature) return;
      const key = box.contains(doc.activeElement) ? doc.activeElement.dataset.key : null;
      box.replaceChildren(); build();
      box.dataset.signature = signature;
      if (key) { const target = [...box.querySelectorAll('[data-key]')].find(element => element.dataset.key === key); (target || title).focus({ preventScroll: true }); }
    }
    function noticeLine(state, key) { const line = node('p', 'voice-attribution-note', state.notices[key] || ''); line.dataset.voiceAdminNotice = key; line.hidden = !line.textContent; return line; }
    function profileLabel(state, profileId) {
      const named = state.list?.consents.find(row => row.verified?.id === profileId && row.verified.name)?.verified.name;
      return named ? `Perfil «${named}»` : 'Perfil de voz ' + profileId.slice(0, 8);
    }

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      section.hidden = !caps.read;
      if (!caps.read) return;
      refresh.setAttribute('aria-disabled', String(state.read.loading));
      freshness.textContent = state.read.error || (!state.read.loaded ? (state.read.loading ? 'Consultando la memoria por voz…' : '') : '');
      freshness.hidden = !freshness.textContent;
      sessionNotice.textContent = state.notices.session || ''; sessionNotice.hidden = !sessionNotice.textContent;
      rebuild(switchBox, JSON.stringify([state.switch, state.markers['switch:switch'] || null, state.notices['switch:switch'] || null, caps.toggle]), () => {
        const value = state.switch;
        if (!value) return;
        switchBox.append(node('h4', '', 'Interruptor'),
          node('p', 'voice-attribution-note', value.configured ? 'La configuración del servidor habilita la memoria por voz.' : 'La configuración del servidor no habilita la memoria por voz: queda apagada aunque el interruptor esté encendido.'),
          node('p', 'voice-attribution-state', 'Interruptor ' + (value.enabled ? 'encendido' : 'apagado') + ' · ' + (value.effective ? 'funcionando' : 'sin efecto')));
        const sending = state.markers['switch:switch']?.state === 'sending';
        if (caps.toggle && (value.enabled || value.configured)) {
          const control = button(value.enabled ? 'Apagar…' : 'Encender…', 'voice-switch', () => toggle(!value.enabled), value.enabled ? 'identity-danger' : 'identity-primary');
          control.setAttribute('aria-disabled', String(sending)); switchBox.append(control);
        }
        switchBox.append(noticeLine(state, 'switch:switch'));
      });
      const list = state.list;
      rebuild(listBox, JSON.stringify([list?.consents || null, list?.truncated ?? null, state.listReason, state.markers, state.notices, caps.reopen]), () => {
        if (!state.read.loaded) return;
        listBox.append(node('h4', '', 'Inscripciones'));
        if (!list) { listBox.append(node('p', 'voice-attribution-note', state.listReason === 'disabled' ? COPY.listOff : '')); return; }
        if (list.truncated) listBox.append(node('p', 'voice-attribution-note', COPY.truncated));
        if (!list.consents.length) listBox.append(node('p', 'voice-attribution-note', COPY.empty));
        const rows = node('ul', 'voice-attribution-list');
        for (const row of list.consents) {
          const item = node('li', 'voice-attribution-row'); item.dataset.voiceConsent = row.person_id;
          const name = node('strong', '', row.name); name.dir = 'auto';
          const profile = row.verified ? node('span', '', 'Verificada con ' + (row.verified.name ? `«${row.verified.name}»` : 'un perfil que ya no existe')) : node('span', '', 'Sin voz verificada');
          item.append(name, node('span', 'voice-attribution-state', Owner.STATE_LABELS[row.state]), profile);
          if (row.linked > 1) item.append(node('span', 'voice-attribution-warning', Owner.MULTI_PROFILE_NOTICE));
          const key = 'reopen:' + row.person_id;
          if (row.state === 'verification_locked' && caps.reopen) {
            const control = button('Habilitar un nuevo intento…', key, () => reopen(row));
            control.setAttribute('aria-disabled', String(state.markers[key]?.state === 'sending')); item.append(control);
          }
          item.append(noticeLine(state, key));
          rows.append(item);
        }
        listBox.append(rows);
      });
      rebuild(multipleBox, JSON.stringify(list?.multiple || null), () => {
        if (!list?.multiple.length) return;
        multipleBox.append(node('h4', '', 'Personas con más de una voz'));
        const rows = node('ul', 'voice-attribution-list');
        for (const row of list.multiple) { const item = node('li', 'voice-attribution-row'); const name = node('strong', '', row.name); name.dir = 'auto'; item.append(name, node('span', 'voice-attribution-warning', Owner.MULTI_PROFILE_NOTICE)); rows.append(item); }
        multipleBox.append(rows);
      });
      rebuild(suspendedBox, JSON.stringify([list?.suspended || null, list?.consents.map(row => row.verified) || null, state.markers, state.notices, caps.resume]), () => {
        if (!list?.suspended.length) return;
        suspendedBox.append(node('h4', '', 'Perfiles en pausa'));
        const rows = node('ul', 'voice-attribution-list');
        for (const row of list.suspended) {
          const label = profileLabel(state, row.profile_id), key = 'resume:' + row.profile_id;
          const item = node('li', 'voice-attribution-row'); item.dataset.voiceSuspended = row.profile_id;
          const text = node('strong', '', label); text.dir = 'auto';
          item.append(text, node('span', '', 'Propuestas por voz en pausa: varias fueron devueltas como «No es mío».'));
          if (caps.resume) { const control = button('Reanudar…', key, () => resume(row.profile_id, label), 'identity-primary'); control.setAttribute('aria-disabled', String(state.markers[key]?.state === 'sending')); item.append(control); }
          item.append(noticeLine(state, key));
          rows.append(item);
        }
        suspendedBox.append(rows);
      });
      if (state.announcement && state.announcement.serial !== lastAnnouncement) { lastAnnouncement = state.announcement.serial; live.textContent = state.announcement.text; }
    }
    const stop = owner.subscribe(render);
    if (owner.getState().announcement) lastAnnouncement = owner.getState().announcement.serial;
    render();
    return () => {
      active = false; stop();
      if (confirmResolve) confirmResolve(false);
      if (confirmDialog.open) confirmDialog.close();
      confirmDialog.replaceChildren(); container.replaceChildren();
    };
  }
  const exported = { mountVoiceConsents, COPY };
  root.NexoVoiceConsents = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
