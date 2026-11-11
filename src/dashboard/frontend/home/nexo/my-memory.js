/* NEXO 'Mis recuerdos' view: DOM only, for a household account bound to a person.
 * It shows only what GET /api/memory/mine returned for that person; nothing here
 * reads voice links, names or another person's memories. */
(function (root) {
  'use strict';
  const CATEGORY = { personal: 'Personal', preference: 'Preferencia', pattern: 'Hábito', fact: 'Dato' };
  const METHOD = { operator_assignment: 'Asignado por la administración', person_session: 'Agregado desde tu cuenta', voice_proposal: 'Atribuido por tu voz' };
  const OP_LABEL = { delete: 'eliminar', return: 'devolver' };
  const CLAMP = 280;
  const SHOW_STEP = 100;
  // Privacy copy (u3b design v1 nexo_ui.household_view, v4 person_notice, P2_A, answers out of scope).
  const COPY = {
    what: 'Recuerdos que la administración del hogar te asignó. Ves solo los tuyos.',
    admin: 'Las cuentas de administración del hogar ven todos los recuerdos, también estos, y pueden eliminarlos o reasignarlos.',
    answers: 'En esta versión KZA no usa estos recuerdos para responder.',
    misattribution: 'Si un recuerdo no es tuyo, usá «No es mío»: vuelve a la administración y no se borra.',
    empty: 'Todavía no tenés recuerdos asignados. KZA no reconoce voces para asignarte recuerdos automáticamente.',
    erasure: 'Las copias en índices de búsqueda se limpian en segundo plano. No borra la conversación de origen (vence sola en 48 horas como máximo) ni el texto que ya se haya enviado a la nube para deducirlo.',
    hidden: 'También se eliminan datos derivados de este recuerdo que no se muestran acá.',
    // Always shown: the deletion removes what depends on it when the server applies it, and no count comes back (R6).
    cascade: 'Se elimina además todo lo que dependa de este recuerdo al momento de borrarlo, aunque haya aparecido después de abrir este aviso. No se informa cuántos datos derivados se eliminaron.',
  };

  function mountMyMemory({ container, confirmDialog, owner, icon, returnNotice }) {
    const doc = container.ownerDocument;
    const cards = new Map(), expanded = new Set();
    let active = true, lastAnnouncement = 0, confirmResolve = null, shown = SHOW_STEP, moreFocus = null;

    function node(tag, className, text) {
      const element = doc.createElement(tag);
      if (className) element.className = className;
      if (text !== undefined && text !== null) element.textContent = String(text);
      return element;
    }
    function button(text, key, callback, className = 'identity-secondary') {
      const element = node('button', className, text); element.type = 'button'; element.dataset.key = key;
      element.addEventListener('click', callback); return element;
    }
    function date(seconds) {
      return new Date(seconds * 1000).toLocaleString('es-UY', { timeZone: 'America/Montevideo', dateStyle: 'medium', timeStyle: 'short' });
    }
    function content(item, clamp = true) {
      const long = item.content.length > CLAMP, open = !clamp || expanded.has(item.id);
      const text = node('p', 'memory-content', long && !open ? item.content.slice(0, CLAMP).trimEnd() + '…' : item.content);
      text.dir = 'auto'; return { text, long };
    }
    function stateLine(item) {
      const kind = item.state === 'confirmed' ? (item.method === 'operator_assignment' ? 'Confirmado por la administración · no vence' : 'Confirmado · no vence')
        : item.state === 'proposed' ? 'Propuesta · vence ' + date(item.expires_at) : 'Automático · vence ' + date(item.expires_at);
      return (CATEGORY[item.category] || 'Otro') + ' · ' + kind;
    }
    function focusKey() { const element = doc.activeElement; return element && container.contains(element) ? element.dataset?.key || null : null; }
    function focusByKey(key) {
      if (!key) return false;
      const target = [...container.querySelectorAll('[data-key]')].find(element => element.dataset.key === key);
      if (target && !target.disabled) { target.focus({ preventScroll: true }); return true; }
      return false;
    }

    const heading = node('div', 'identity-heading memory-heading');
    const intro = node('div');
    const title = node('h2', '', 'Mis recuerdos.'); title.id = 'my-memory-title';
    intro.append(node('p', 'eyebrow', 'TUS RECUERDOS Y TU PRIVACIDAD'), title, node('p', '', COPY.what + ' ' + COPY.answers),
      node('p', '', COPY.admin), node('p', '', COPY.misattribution));
    const side = node('div', 'identity-heading-side memory-heading-side');
    const refresh = button('Actualizar', 'refresh', () => { if (refresh.getAttribute('aria-disabled') !== 'true') void owner.reload(); }); refresh.id = 'my-memory-refresh';
    side.append(refresh); heading.append(intro, side);
    const permission = node('p', 'memory-note memory-permission');
    const sessionNotice = node('p', 'memory-note memory-session-notice'); sessionNotice.setAttribute('role', 'status');
    const freshness = node('p', 'memory-note memory-freshness'); freshness.id = 'my-memory-freshness';
    // The 'No es mío' outcome stays visible after the card leaves the list (v4 person_notice).
    const returned = node('p', 'memory-note memory-return-notice'); returned.id = 'my-memory-return-notice';
    const live = node('p', 'memory-note memory-live'); live.id = 'my-memory-live'; live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
    const incomplete = node('p', 'memory-note memory-cap');
    const pendingTitle = node('h3', 'memory-section-title', 'Operaciones sin confirmar'); pendingTitle.tabIndex = -1;
    const pendingList = node('div', 'memory-grid'); pendingList.id = 'my-memory-pending';
    const listTitle = node('h3', 'memory-section-title', 'Tus recuerdos'); listTitle.id = 'my-memory-list-title'; listTitle.tabIndex = -1;
    const grid = node('div', 'memory-grid'); grid.id = 'my-memory-facts';
    const more = node('div', 'memory-more');
    const empty = node('p', 'memory-note memory-empty', COPY.empty); empty.id = 'my-memory-empty';
    const sections = node('div', 'memory-sections');
    sections.append(pendingTitle, pendingList, listTitle, incomplete, grid, more, empty);
    container.replaceChildren(heading, permission, sessionNotice, freshness, returned, live, sections);

    function markerText(marker) { return marker.state === 'sending' ? 'Enviando…' : 'Sin confirmar: ' + OP_LABEL[marker.op]; }
    function buildCard(state, item, card) {
      const marker = state.markers[item.id], caps = state.capabilities;
      const top = node('div', 'memory-card-top');
      const badge = node('span', 'scene-icon mint'); badge.append(icon('layers'));
      top.append(badge, node('p', 'memory-meta', stateLine(item))); card.append(top);
      const { text, long } = content(item); card.append(text);
      if (long) {
        const open = expanded.has(item.id);
        const toggle = button(open ? 'Ver menos' : 'Ver completo', item.id + ':expand', () => { if (expanded.has(item.id)) expanded.delete(item.id); else expanded.add(item.id); render(); });
        toggle.setAttribute('aria-expanded', String(open)); card.append(toggle);
      }
      card.append(node('p', 'memory-detail', (METHOD[item.method] || 'Asignado') + ' · ' + date(item.assigned_at)));
      if (marker) { const line = node('p', 'memory-state', markerText(marker)); line.dataset.memoryMarker = marker.state; card.append(line); }
      const message = node('p', 'memory-notice', state.notices[item.id] || ''); message.dataset.memoryNotice = item.id; card.append(message);
      const controls = node('div', 'memory-controls');
      if (caps.write) {
        if (marker) {
          if (!marker.restored) { const again = button('Reintentar ' + OP_LABEL[marker.op], item.id + ':retry', () => owner.retry(item.id)); again.disabled = marker.state === 'sending'; controls.append(again); }
        } else {
          const blocked = owner.blocked(item.id);
          const giveBack = button('No es mío…', item.id + ':return', () => confirmReturn(item.id)); giveBack.disabled = blocked;
          giveBack.setAttribute('aria-label', 'No es mío: devolver este recuerdo a la administración');
          const removal = button('Eliminar…', item.id + ':delete', () => confirmDelete(item.id), 'identity-danger'); removal.disabled = blocked;
          removal.setAttribute('aria-label', 'Eliminar este recuerdo');
          controls.append(giveBack, removal);
        }
      }
      if (controls.childElementCount) card.append(controls);
    }
    function place(state, items) {
      const key = focusKey(), seen = new Set();
      let previous = null, lost = null;
      for (const item of items) {
        seen.add(item.id);
        let entry = cards.get(item.id);
        if (!entry) { const card = node('article', 'memory-card panel'); card.dataset.myMemoryId = item.id; card.tabIndex = -1; entry = { card, signature: null }; cards.set(item.id, entry); }
        const signature = JSON.stringify([item, state.markers[item.id] || null, state.notices[item.id] || null, state.capabilities, expanded.has(item.id), owner.blocked(item.id)]);
        if (entry.signature !== signature) {
          const hadFocus = entry.card.contains(doc.activeElement);
          entry.card.replaceChildren(); entry.card.setAttribute('aria-label', 'Recuerdo ' + (CATEGORY[item.category] || ''));
          buildCard(state, item, entry.card); entry.signature = signature;
          if (hadFocus && !focusByKey(key)) (entry.card.querySelector('button:not([disabled])') || entry.card).focus({ preventScroll: true });
        }
        const expected = previous ? previous.nextSibling : grid.firstChild;
        if (expected !== entry.card) grid.insertBefore(entry.card, expected);
        previous = entry.card;
      }
      [...grid.children].forEach((card, index) => {
        const id = card.dataset.myMemoryId;
        if (seen.has(id)) return;
        if (card.contains(doc.activeElement)) lost = index;
        card.remove(); cards.delete(id);
      });
      if (lost !== null) {
        const target = grid.children[Math.min(lost, grid.children.length - 1)];
        (target ? target.querySelector('button:not([disabled])') || target : listTitle).focus({ preventScroll: true });
      }
    }

    function dialog(nodes, focusTarget, invoker, bind) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        confirmDialog.replaceChildren(...nodes);
        confirmDialog.setAttribute('aria-labelledby', 'my-memory-confirm-title');
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          if (active && !focusByKey(invoker)) listTitle.focus({ preventScroll: true });
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish; bind(finish);
        if (!confirmDialog.open) confirmDialog.showModal();
        focusTarget.focus();
      });
    }
    function target(item) {
      const box = node('div', 'memory-confirm-target');
      box.append(content(item, false).text, node('p', 'memory-detail', stateLine(item)));
      return box;
    }
    function actions(acceptText, acceptClass) {
      const row = node('div', 'modal-actions memory-confirm-actions');
      const cancelButton = button('Cancelar', 'confirm-cancel', () => {});
      const accept = button(acceptText, 'confirm-accept', () => {}, acceptClass);
      row.append(cancelButton, accept);
      return { row, cancelButton, accept };
    }
    function deleteParts(item, owned, changed) {
      const heading = node('h2', '', 'Eliminar este recuerdo'); heading.id = 'my-memory-confirm-title';
      const nodes = [heading];
      if (changed) { const note = node('p', 'memory-detail memory-changed', 'Los recuerdos que se eliminan junto con este cambiaron mientras decidías. Revisalos antes de confirmar.'); note.setAttribute('role', 'alert'); nodes.push(note); }
      nodes.push(target(item), node('p', 'modal-description', 'Es definitivo: no se puede deshacer.' + (owned.length ? ' También se eliminan estos recuerdos tuyos que dependen de él:' : '')));
      if (owned.length) {
        const list = node('ul', 'memory-confirm-items');
        for (const other of owned) {
          const line = node('li');
          if (other.content === null) line.append(node('span', 'memory-detail', 'Un recuerdo tuyo que no figura en esta lista.'));
          else { const text = node('span', 'memory-content', other.content.length > 120 ? other.content.slice(0, 120).trimEnd() + '…' : other.content); text.dir = 'auto'; line.append(text); }
          list.append(line);
        }
        nodes.push(list);
      }
      if (item.hidden_dependents) nodes.push(node('p', 'memory-detail', COPY.hidden));
      const cascade = node('p', 'memory-detail', COPY.cascade); cascade.id = 'my-memory-delete-cascade';
      nodes.push(cascade, node('p', 'memory-detail', COPY.erasure));
      return nodes;
    }
    async function confirmDelete(id) {
      let item = owner.getState().facts.find(entry => entry.id === id);
      if (!item) return;
      let owned = owner.ownedDependents(id), changed = false;
      // The cascade accepted must be the one sent: re-ask when a read changed it meanwhile (N5).
      for (let round = 0; round < 5; round++) {
        const { row, cancelButton, accept } = actions('Eliminar definitivamente', 'identity-danger');
        const accepted = await dialog([...deleteParts(item, owned, changed), row], cancelButton, id + ':delete',
          finish => { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); });
        if (!accepted || !active) return;
        const now = owner.getState().facts.find(entry => entry.id === id);
        if (!now) return;
        const current = owner.ownedDependents(id);
        if (current.length === owned.length && current.every(entry => owned.some(other => other.id === entry.id)) && now.hidden_dependents === item.hidden_dependents) break;
        owned = current; item = now; changed = true;
        if (round === 4) return;
      }
      await owner.remove(id, { confirmed: true });
    }
    async function confirmReturn(id) {
      const item = owner.getState().facts.find(entry => entry.id === id);
      if (!item) return;
      const heading = node('h2', '', 'No es mío'); heading.id = 'my-memory-confirm-title';
      // The same sentence before acting as after (v4 person_notice).
      const { row, cancelButton, accept } = actions('Devolver a la administración', 'identity-primary');
      const accepted = await dialog([heading, target(item), node('p', 'modal-description', 'Se devuelve a la administración del hogar y deja de figurar en Mis recuerdos.'),
        node('p', 'modal-description memory-return-before', returnNotice), row], cancelButton, id + ':return',
        finish => { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); });
      if (accepted && active) await owner.giveBack(id, { confirmed: true });
    }

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      refresh.setAttribute('aria-disabled', String(state.read.loading));
      permission.textContent = caps.write ? '' : 'Este acceso permite ver tus recuerdos pero no cambiarlos.'; permission.hidden = !permission.textContent;
      sessionNotice.textContent = state.notices.session || ''; sessionNotice.hidden = !sessionNotice.textContent;
      const time = state.read.observedWall ? new Date(state.read.observedWall).toLocaleTimeString('es-UY', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }) : null;
      freshness.textContent = state.read.error ? `${state.read.error}${time ? ' Última consulta confirmada a las ' + time + '.' : ''}${state.loaded ? ' Se muestran esos datos.' : ''}`
        : !state.loaded ? (state.read.loading ? 'Consultando tus recuerdos…' : 'Sin consulta confirmada.') : 'Consultados a las ' + time + '.';
      freshness.classList.toggle('is-error', Boolean(state.read.error));
      returned.textContent = state.returnNotice || ''; returned.hidden = !returned.textContent;
      const listed = new Set(state.facts.map(item => item.id));
      const orphan = Object.entries(state.markers).filter(([id]) => !listed.has(id));
      // Rebuilt only when it changes, keeping focus on a retry control.
      const orphanSignature = JSON.stringify([orphan, orphan.map(([id]) => state.notices[id] || null), caps.write]);
      if (pendingList.dataset.signature !== orphanSignature) {
        const key = focusKey();
        pendingList.replaceChildren(...orphan.map(([id, marker]) => {
          const card = node('article', 'memory-card panel memory-orphan'); card.setAttribute('aria-label', 'Operación sin confirmar');
          card.append(node('p', 'memory-state', markerText(marker) + ' un recuerdo que no figura en la lista.'));
          const message = node('p', 'memory-notice', state.notices[id] || ''); message.dataset.memoryNotice = id; card.append(message);
          if (caps.write && !marker.restored) { const again = button('Reintentar ' + OP_LABEL[marker.op], id + ':retry', () => owner.retry(id)); again.disabled = marker.state === 'sending'; card.append(again); }
          return card;
        }));
        pendingList.dataset.signature = orphanSignature;
        if (key && key.endsWith(':retry') && !focusByKey(key) && !container.contains(doc.activeElement)) listTitle.focus({ preventScroll: true });
      }
      pendingTitle.hidden = pendingList.hidden = !orphan.length || !state.loaded;
      const show = state.loaded;
      listTitle.hidden = grid.hidden = !show;
      incomplete.textContent = show && !state.complete ? 'Tu lista es más larga de lo que esta vista puede recorrer; se muestran los primeros recuerdos.' : '';
      incomplete.hidden = !incomplete.textContent;
      place(state, show ? state.facts.slice(0, shown) : []);
      const remaining = show ? state.facts.length - Math.min(shown, state.facts.length) : 0;
      const existing = more.querySelector('button');
      if (remaining > 0) {
        const text = `Mostrar más recuerdos (quedan ${remaining})`;
        if (!existing || existing.textContent !== text) {
          const hadFocus = existing === doc.activeElement;
          const control = button(text, 'more', () => { moreFocus = shown; shown += SHOW_STEP; render(); });
          more.replaceChildren(control); if (hadFocus) control.focus({ preventScroll: true });
        }
      } else if (existing) more.replaceChildren();
      if (moreFocus !== null) { const card = grid.children[moreFocus]; if (card) (card.querySelector('button:not([disabled])') || card).focus({ preventScroll: true }); moreFocus = null; }
      empty.hidden = !show || state.facts.length > 0;
      if (state.announcement && state.announcement.serial !== lastAnnouncement) { lastAnnouncement = state.announcement.serial; live.textContent = state.announcement.text; }
    }
    const stop = owner.subscribe(render);
    if (owner.getState().announcement) lastAnnouncement = owner.getState().announcement.serial;
    render();
    return () => {
      active = false; stop();
      if (confirmResolve) confirmResolve(false);
      if (confirmDialog.open) confirmDialog.close();
      confirmDialog.replaceChildren(); container.replaceChildren(); cards.clear(); expanded.clear();
    };
  }
  const exported = { mountMyMemory, COPY };
  root.NexoMyMemory = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
