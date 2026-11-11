/* NEXO 'Memoria por voz' view (U3b-4a-3): DOM only, inside 'Mis recuerdos', for a household account
 * bound to a person. It shows only the caller's own consent, never another person's state, window,
 * room or profile. With the voice stage off it shows the status and the revocation only. */
(function (root) {
  'use strict';
  const COPY = {
    title: 'Memoria por voz',
    window: 'Cada ventana dura 60 segundos. Podés abrir hasta 6 por día; cada intento cuenta, también si la habitación no está escuchando.',
    noPhrase: 'La frase para decir en voz alta todavía no está disponible en esta versión. La ventana se cierra sola.',
    noRooms: 'No hay habitaciones disponibles para verificar tu voz.',
    revoke: 'Lo que pidas recordar con tu voz deja de quedar como propuesta tuya. Las propuestas pendientes siguen en Mis recuerdos para que las descartes o digas «No es mío».',
    locked: 'Si te das de baja, la verificación sigue bloqueada hasta que la administración la habilite.',
    disclosureSummary: 'Cómo funciona y qué se guarda',
  };

  function mountMyVoice({ container, confirmDialog, owner, icon }) {
    const doc = container.ownerDocument, Owner = root.NexoMyVoiceOwner;
    let active = true, lastAnnouncement = 0, confirmResolve = null, selectedRoom = '';

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
    function time(seconds) {
      return new Date(seconds * 1000).toLocaleTimeString('es-UY', { timeZone: 'America/Montevideo', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
    }

    const section = node('section', 'memory-voice panel'); section.setAttribute('aria-labelledby', 'my-voice-title');
    const heading = node('h3', 'memory-section-title', COPY.title); heading.id = 'my-voice-title'; heading.tabIndex = -1;
    const top = node('div', 'memory-card-top'); const badge = node('span', 'scene-icon lavender'); badge.append(icon('mic')); top.append(badge, heading);
    const status = node('p', 'memory-note memory-voice-status'); status.id = 'my-voice-status';
    const message = node('p', 'memory-note memory-voice-message'); message.id = 'my-voice-message';
    const freshness = node('p', 'memory-note'); freshness.id = 'my-voice-freshness';
    const notice = node('p', 'memory-note memory-voice-notice'); notice.id = 'my-voice-notice'; notice.setAttribute('role', 'status');
    const live = node('p', 'memory-note memory-live'); live.id = 'my-voice-live'; live.setAttribute('aria-live', 'polite');
    const windowBox = node('div', 'memory-voice-window'); windowBox.id = 'my-voice-window';
    const verify = node('div', 'memory-voice-verify'); verify.id = 'my-voice-verify';
    const controls = node('div', 'memory-controls'); controls.id = 'my-voice-controls';
    const disclosure = node('details', 'memory-sources memory-voice-disclosure'); disclosure.id = 'my-voice-disclosure';
    disclosure.append(node('summary', '', COPY.disclosureSummary), node('p', 'memory-detail', Owner.OPT_IN_DISCLOSURE));
    section.append(top, status, message, freshness, notice, live, windowBox, verify, controls, disclosure);
    container.replaceChildren(section);

    function dialog(nodes, focusTarget, invoker, bind) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        confirmDialog.replaceChildren(...nodes);
        confirmDialog.setAttribute('aria-labelledby', 'my-voice-confirm-title');
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          const back = [...container.querySelectorAll('[data-key]')].find(element => element.dataset.key === invoker);
          if (active) (back || heading).focus({ preventScroll: true });
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish; bind(finish);
        if (!confirmDialog.open) confirmDialog.showModal();
        focusTarget.focus();
      });
    }
    function confirmParts(title, texts, acceptText, acceptClass) {
      const h = node('h2', '', title); h.id = 'my-voice-confirm-title';
      const row = node('div', 'modal-actions memory-confirm-actions');
      const cancelButton = button('Cancelar', 'voice-confirm-cancel', () => {});
      const accept = button(acceptText, 'voice-confirm-accept', () => {}, acceptClass);
      row.append(cancelButton, accept);
      return { nodes: [h, ...texts.map(text => node('p', 'modal-description', text)), row], cancelButton, accept };
    }
    async function confirmRequest() {
      const { nodes, cancelButton, accept } = confirmParts('Inscribirte en la memoria por voz', [Owner.OPT_IN_DISCLOSURE], 'Inscribirme', 'identity-primary');
      const accepted = await dialog(nodes, cancelButton, 'voice-request', finish => { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); });
      if (accepted && active) await owner.requestConsent({ confirmed: true });
    }
    async function confirmRevoke() {
      const locked = owner.getState().status?.state === 'verification_locked';
      const { nodes, cancelButton, accept } = confirmParts('Darte de baja de la memoria por voz', locked ? [COPY.revoke, COPY.locked] : [COPY.revoke], 'Darme de baja', 'identity-danger');
      const accepted = await dialog(nodes, cancelButton, 'voice-revoke', finish => { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); });
      if (accepted && active) await owner.revoke({ confirmed: true });
    }

    function render() {
      if (!active) return;
      const state = owner.getState(), current = state.status, allowed = state.allowed, sending = state.marker?.state === 'sending';
      section.hidden = !state.capabilities.voice;
      if (!state.capabilities.voice) return;
      status.textContent = !current ? '' : (current.state ? Owner.STATE_LABELS[current.state] : Owner.NOT_ENROLLED) + (current.enabled ? '' : ' ' + Owner.OFF + (allowed.revoke ? ' ' + Owner.OFF_REVOCABLE : ''));
      status.hidden = !status.textContent;
      const failures = current?.enabled && current.failuresLeft !== null ? ` Te quedan ${current.failuresLeft} ${current.failuresLeft === 1 ? 'intento' : 'intentos'} antes de que se bloquee la verificación.` : '';
      message.textContent = (state.message || '') + failures; message.hidden = !message.textContent;
      freshness.textContent = state.read.error || (!state.read.loaded ? (state.read.loading ? 'Consultando tu inscripción por voz…' : '') : '');
      freshness.hidden = !freshness.textContent; freshness.classList.toggle('is-error', Boolean(state.read.error));
      notice.textContent = state.notice || ''; notice.hidden = !notice.textContent;
      disclosure.hidden = !current?.enabled;
      // The open window: room, deadline and the phrase, kept only in memory while the view is open.
      const window = current?.enabled ? current.window : null;
      const windowSignature = JSON.stringify(window);
      if (windowBox.dataset.signature !== windowSignature) {
        windowBox.replaceChildren();
        if (window) {
          const room = (state.rooms || []).find(item => item.room_id === window.room_id)?.name || 'la habitación elegida';
          windowBox.append(node('p', 'memory-note', `Ventana abierta en ${room} hasta las ${time(window.expires_at)}.`));
          if (window.phrase) { const phrase = node('p', 'memory-voice-phrase', '«' + window.phrase + '»'); phrase.dir = 'auto'; windowBox.append(node('p', 'memory-detail', 'Decí en voz alta, en esa habitación:'), phrase); }
          else windowBox.append(node('p', 'memory-detail', COPY.noPhrase));
        }
        windowBox.dataset.signature = windowSignature;
      }
      windowBox.hidden = !window;
      // Room choice and the window button only while verification is possible.
      const rooms = state.rooms || [];
      const verifySignature = JSON.stringify([allowed.verify, rooms, sending]);
      if (verify.dataset.signature !== verifySignature) {
        const hadFocus = verify.contains(doc.activeElement) ? doc.activeElement.dataset.key : null;
        verify.replaceChildren();
        if (allowed.verify) {
          if (!rooms.some(room => room.room_id === selectedRoom)) selectedRoom = rooms[0]?.room_id || '';
          const label = node('label', 'memory-voice-room', 'Habitación donde vas a hablar');
          const select = node('select'); select.id = 'my-voice-room'; select.dataset.key = 'voice-room';
          for (const room of rooms) { const option = node('option', '', room.name); option.value = room.room_id; option.selected = room.room_id === selectedRoom; select.append(option); }
          select.addEventListener('change', () => { selectedRoom = select.value; });
          label.append(select);
          const openWindow = button('Abrir ventana de verificación', 'voice-window', () => { void owner.openWindow(selectedRoom, { confirmed: true }); }, 'identity-primary');
          openWindow.setAttribute('aria-disabled', String(sending || !rooms.length));
          verify.append(node('p', 'memory-detail', rooms.length ? COPY.window : COPY.noRooms), ...(rooms.length ? [label] : []), openWindow);
          if (hadFocus) { const target = [...verify.querySelectorAll('[data-key]')].find(element => element.dataset.key === hadFocus); target?.focus({ preventScroll: true }); }
        }
        verify.dataset.signature = verifySignature;
      }
      verify.hidden = !allowed.verify;
      const controlSignature = JSON.stringify([allowed.request, allowed.revoke, sending]);
      if (controls.dataset.signature !== controlSignature) {
        const hadFocus = controls.contains(doc.activeElement) ? doc.activeElement.dataset.key : null;
        controls.replaceChildren();
        if (allowed.request) { const control = button('Inscribirme…', 'voice-request', confirmRequest, 'identity-primary'); control.setAttribute('aria-disabled', String(sending)); controls.append(control); }
        if (allowed.revoke) { const control = button('Darme de baja…', 'voice-revoke', confirmRevoke, 'identity-danger'); control.setAttribute('aria-disabled', String(sending)); controls.append(control); }
        if (hadFocus) { const target = [...controls.querySelectorAll('[data-key]')].find(element => element.dataset.key === hadFocus); (target || heading).focus({ preventScroll: true }); }
        controls.dataset.signature = controlSignature;
      }
      controls.hidden = !controls.childElementCount;
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
  const exported = { mountMyVoice, COPY };
  root.NexoMyVoice = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
