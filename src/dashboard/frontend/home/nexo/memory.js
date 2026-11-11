/* NEXO Memoria view: DOM only. Cards are keyed by fact id and rebuilt only when
 * their rendered data changes; expiry ticks never rebuild a focused card. */
(function (root) {
  'use strict';
  const CATEGORY = { personal: 'Personal', preference: 'Preferencia', pattern: 'Hábito', fact: 'Dato' };
  const ORIGIN = { ambient: 'Captura ambiental', voice: 'Comando de voz', interaction: 'Conversación' };
  const OP_LABEL = { confirm: 'conservar', reject: 'descartar', delete: 'eliminar', assign: 'asignar', unassign: 'quitar la asignación' };
  const STATE_LABEL = { confirmed: 'confirmado', proposed: 'propuesta', automatic: 'automático' };
  const CLAMP = 280;
  // Cards rendered per section before 'Mostrar más'; the owner already holds the whole walked list.
  const SHOW_STEP = 100;
  // v3 assignment_confirms.retention_disclosure and the D5 sentence (u3b design v2/v3).
  const ASSIGN_PERMANENCE = 'Al asignarlo, el recuerdo queda confirmado y no vence: se guarda hasta que la persona o la administración lo eliminen. La persona va a poder verlo, eliminarlo o devolverlo con «No es mío».';
  const ASSIGN_OTHERS = 'Si el recuerdo habla de otra persona, asignarlo se lo muestra a quien elijas.';
  // The server deletes the derived memories that exist when it applies the assignment and names their ids;
  // the card names them by the label shown here, and only counts the ones this view never showed.
  const DEPENDENTS_SCOPE = 'Se eliminan los recuerdos derivados que existan al momento de asignar, que pueden no ser exactamente los de esta lista si cambian mientras decidís. Si la lista cambió, el aviso final dice cuáles difieren, con el texto que ves acá; los que no figuran en esta vista solo se cuentan.';
  /** A dependent's label in the dialog (and on the card afterwards): its content, clamped to 120 characters. */
  const dependentLabel = content => content.length > 120 ? content.slice(0, 120).trimEnd() + '…' : content;

  function mountMemory({ container, confirmDialog, owner, icon }) {
    const doc = container.ownerDocument;
    const cards = new Map(), expanded = new Set();
    const shown = { proposals: SHOW_STEP, facts: SHOW_STEP, returned: SHOW_STEP };
    let active = true, lastAnnouncement = 0, confirmResolve = null;

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
    // Untrusted ASR/LLM text: textContent only, isolated direction (F15).
    function content(item, clamp = true) {
      const long = item.content.length > CLAMP, open = !clamp || expanded.has(item.id);
      const text = node('p', 'memory-content', long && !open ? item.content.slice(0, CLAMP).trimEnd() + '…' : item.content);
      text.dir = 'auto'; return { text, long };
    }
    function stateLine(item, kept) {
      const kind = item.state === 'proposed' ? 'Propuesta por revisar' : item.state === 'confirmed' ? 'Confirmado · no vence' : 'Automático';
      const expiry = item.expires_at === null ? '' : ' · vence ' + date(item.expires_at);
      return (CATEGORY[item.category] || 'Otro') + ' · ' + kind + expiry + (kept ? ' · Fuera de la parte recorrida de la lista' : '');
    }
    function confirmedBy(state, item) {
      if (item.state !== 'confirmed' || !item.confirmed_at) return null;
      const who = item.confirmed_by && item.confirmed_by === state.subject ? 'desde esta cuenta' : 'desde otra cuenta de administración';
      return 'Confirmado ' + who + ' · ' + date(item.confirmed_at);
    }
    function personName(state, id) {
      const entry = (state.people || []).find(item => item.id === id);
      return entry ? entry.name : null;
    }
    // Owner label (P2_A: administration sees every owner). A name is account metadata, never a voice label.
    function ownerLine(state, item) {
      if (item.owner) {
        const name = personName(state, item.owner.person_id);
        const label = name === null ? (state.people ? 'una persona que no figura en el catálogo' : 'una persona del hogar') : name;
        return { text: 'Asignado a ' + label + ' · desde ' + date(item.owner.assigned_at), name };
      }
      if (item.owner_change === 'owner_returned') return { text: 'Devuelto por la persona («No es mío») · sin asignar' };
      if (item.owner_change === 'operator_unassigned') return { text: 'Sin asignar · la administración quitó la asignación' };
      return { text: 'Sin asignar' };
    }
    function focusKey() { const element = doc.activeElement; return element && container.contains(element) ? element.dataset?.key || null : null; }
    function focusByKey(key) {
      if (!key) return false;
      const target = [...container.querySelectorAll('[data-key]')].find(element => element.dataset.key === key);
      if (target && !target.disabled) { target.focus({ preventScroll: true }); return true; }
      return false;
    }

    // Static skeleton.
    const heading = node('div', 'identity-heading memory-heading');
    const intro = node('div');
    const title = node('h2', '', 'Recuerdos del hogar.'); title.id = 'memory-title';
    intro.append(node('p', 'eyebrow', 'RECUERDOS Y PRIVACIDAD'), title,
      // Pinned by tests/unit/memory/test_fact_reader_callers.py: no module outside src/memory reads facts for answers.
      node('p', '', 'KZA guarda automáticamente datos que deduce de lo que escucha en el ambiente; ese texto puede procesarse en la nube. Los conserva hasta que vencen (como máximo 30 días) o los eliminás. En esta versión KZA no usa estos recuerdos para responder. Las propuestas quedan acá hasta que las confirmes o descartes.'),
      node('p', '', 'La memoria es compartida por todo el hogar: las cuentas de administración ven y revisan los recuerdos de todos. Una cuenta del hogar vinculada a una persona ve solo los recuerdos que le asignes, en Mis recuerdos; las demás cuentas del hogar no tienen acceso.'));
    const side = node('div', 'identity-heading-side memory-heading-side');
    // aria-disabled instead of disabled: a disabled focused button drops keyboard focus to <body>.
    const refresh = button('Actualizar', 'refresh', () => { if (refresh.getAttribute('aria-disabled') !== 'true') void owner.reload(); }); refresh.id = 'memory-refresh';
    side.append(refresh); heading.append(intro, side);
    const permission = node('p', 'memory-note memory-permission');
    const sessionNotice = node('p', 'memory-note memory-session-notice'); sessionNotice.setAttribute('role', 'status');
    const freshness = node('p', 'memory-note memory-freshness'); freshness.id = 'memory-freshness';
    const counts = node('p', 'memory-note memory-counts'); counts.id = 'memory-owner-counts';
    const prune = node('p', 'memory-note memory-prune'); prune.id = 'memory-prune-pending'; prune.setAttribute('role', 'status');
    const peopleNote = node('p', 'memory-note memory-people');
    const cleanup = node('p', 'memory-note memory-cleanup');
    const indexing = node('p', 'memory-note memory-indexing');
    const quarantine = node('p', 'memory-note');
    // Visible as well as announced: a resolved card may have left the list with its outcome.
    const live = node('p', 'memory-note memory-live'); live.id = 'memory-live'; live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
    const pendingTitle = node('h3', 'memory-section-title', 'Operaciones sin confirmar'); pendingTitle.id = 'memory-pending-title'; pendingTitle.tabIndex = -1;
    const pendingList = node('div', 'memory-grid'); pendingList.id = 'memory-pending';
    const returnedTitle = node('h3', 'memory-section-title', 'Devueltos por las personas'); returnedTitle.id = 'memory-returned-title'; returnedTitle.tabIndex = -1;
    const returnedNote = node('p', 'memory-note', 'Una persona marcó estos recuerdos como «No es mío». Siguen guardados, confirmados y sin vencimiento hasta que los reasignes o los elimines.');
    const returnedStatus = node('p', 'memory-note memory-returned-status');
    const returnedGrid = node('div', 'memory-grid'); returnedGrid.id = 'memory-returned';
    const returnedMore = node('div', 'memory-more');
    const returnedEmpty = node('p', 'memory-note memory-empty', 'No hay recuerdos devueltos.');
    const proposalsTitle = node('h3', 'memory-section-title', 'Propuestas pendientes'); proposalsTitle.id = 'memory-proposals-title'; proposalsTitle.tabIndex = -1;
    const proposalsCap = node('p', 'memory-note memory-cap');
    const proposalsGrid = node('div', 'memory-grid'); proposalsGrid.id = 'memory-proposals';
    const proposalsMore = node('div', 'memory-more');
    const proposalsEmpty = node('p', 'memory-note memory-empty', 'No hay propuestas pendientes.');
    const factsTitle = node('h3', 'memory-section-title', 'Recuerdos guardados'); factsTitle.id = 'memory-facts-title'; factsTitle.tabIndex = -1;
    const factsCap = node('p', 'memory-note memory-cap');
    const factsGrid = node('div', 'memory-grid'); factsGrid.id = 'memory-facts';
    const factsMore = node('div', 'memory-more');
    const factsEmpty = node('p', 'memory-note memory-empty', 'No hay recuerdos guardados.');
    const sections = node('div', 'memory-sections');
    sections.append(pendingTitle, pendingList, returnedTitle, returnedNote, returnedStatus, returnedGrid, returnedMore, returnedEmpty,
      proposalsTitle, proposalsCap, proposalsGrid, proposalsMore, proposalsEmpty, factsTitle, factsCap, factsGrid, factsMore, factsEmpty);
    container.replaceChildren(heading, permission, sessionNotice, freshness, counts, prune, peopleNote, cleanup, indexing, quarantine, live, sections);

    function markerText(marker) {
      if (!marker) return null;
      if (marker.state === 'sending') return 'Enviando…';
      return 'Sin confirmar: ' + OP_LABEL[marker.op];
    }
    function cardSignature(state, item, section) {
      return JSON.stringify([item, section, state.markers[item.id] || null, state.notices[item.id] || null, state.capabilities, state.subject,
        expanded.has(item.id), state.markers[item.id] ? null : owner.deleteBlocked(item.id), ownerLine(state, item).text, owner.assignable(item.id)]);
    }
    function buildCard(state, item, section, card) {
      const marker = state.markers[item.id], caps = state.capabilities;
      const top = node('div', 'memory-card-top');
      const badge = node('span', 'scene-icon ' + (item.state === 'proposed' ? 'lavender' : 'mint')); badge.append(icon(item.state === 'proposed' ? 'spark' : 'layers'));
      const meta = node('p', 'memory-meta', stateLine(item, section === 'kept'));
      top.append(badge, meta); card.append(top);
      const { text, long } = content(item); card.append(text);
      if (long) {
        const open = expanded.has(item.id);
        const toggle = button(open ? 'Ver menos' : 'Ver completo', item.id + ':expand', () => { if (expanded.has(item.id)) expanded.delete(item.id); else expanded.add(item.id); render(); });
        toggle.setAttribute('aria-expanded', String(open)); card.append(toggle);
      }
      const ownerInfo = node('p', 'memory-detail memory-owner', ownerLine(state, item).text); ownerInfo.dataset.memoryOwner = item.id; card.append(ownerInfo);
      const by = confirmedBy(state, item);
      if (by) card.append(node('p', 'memory-detail', by));
      const isDerived = item.sources.some(source => source.depends_on.length);
      if (isDerived) card.append(node('p', 'memory-detail', caps.assign ? 'Derivado de otro recuerdo: no se puede asignar a una persona.' : 'Derivado de otro recuerdo.'));
      if (item.sources.length) {
        const details = node('details', 'memory-sources'); const summary = node('summary', '', 'Ver procedencia'); summary.dataset.key = item.id + ':sources';
        details.append(summary);
        for (const source of item.sources) details.append(node('p', 'memory-detail', (ORIGIN[source.origin] || 'Origen desconocido') + ' · ' + date(source.captured_at)));
        card.append(details);
      }
      const status = markerText(marker);
      if (status) { const line = node('p', 'memory-state', status); line.dataset.memoryMarker = marker.state; card.append(line); }
      const message = node('p', 'memory-notice', state.notices[item.id] || ''); message.dataset.memoryNotice = item.id; card.append(message);
      const controls = node('div', 'memory-controls');
      const writable = marker ? (['confirm', 'reject', 'delete'].includes(marker.op) ? caps.review : caps.assign) : false;
      if (marker && writable) {
        const again = button('Reintentar ' + OP_LABEL[marker.op], item.id + ':retry', () => owner.retry(item.id));
        again.disabled = marker.state === 'sending'; again.dataset.memoryWrite = 'retry'; controls.append(again);
      } else if (!marker) {
        if (caps.review && item.state === 'proposed') {
          const keep = button('Conservar sin vencimiento', item.id + ':confirm', () => owner.confirm(item.id), 'identity-primary'); keep.dataset.memoryWrite = 'confirm';
          const drop = button('Descartar propuesta', item.id + ':reject', () => owner.reject(item.id)); drop.dataset.memoryWrite = 'reject';
          controls.append(keep, drop);
        }
        if (owner.assignable(item.id)) {
          const assign = button(item.owner || section === 'returned' ? 'Reasignar…' : 'Asignar a una persona…', item.id + ':assign', () => openAssign(item.id));
          assign.dataset.memoryWrite = 'assign'; assign.disabled = owner.deleteBlocked(item.id); controls.append(assign);
        }
        if (caps.assign && item.owner) {
          const release = button('Quitar asignación…', item.id + ':unassign', () => openUnassign(item.id)); release.dataset.memoryWrite = 'unassign';
          controls.append(release);
        }
        if (caps.review && item.state !== 'proposed') {
          const removal = button('Eliminar…', item.id + ':delete', () => confirmDelete(item.id), 'identity-danger'); removal.dataset.memoryWrite = 'delete';
          removal.setAttribute('aria-label', 'Eliminar este recuerdo');
          removal.disabled = owner.deleteBlocked(item.id);
          controls.append(removal);
        }
      }
      if (controls.childElementCount) card.append(controls);
    }
    // Keyed placement: a card that disappears hands focus to the card now at its position, else the heading.
    function place(grid, entries, state, heading) {
      const key = focusKey(), seen = new Set();
      let previous = null, lostFocus = null;
      for (const [item, section] of entries) {
        seen.add(item.id);
        let entry = cards.get(item.id);
        if (entry && entry.grid !== grid) { if (entry.card.contains(doc.activeElement)) lostFocus = lostFocus || { index: 0, key }; entry.card.remove(); cards.delete(item.id); entry = null; }
        if (!entry) {
          const card = node('article', 'memory-card panel'); card.dataset.memoryId = item.id; card.tabIndex = -1;
          entry = { card, signature: null, grid }; cards.set(item.id, entry);
        }
        const signature = cardSignature(state, item, section);
        if (entry.signature !== signature) {
          const hadFocus = entry.card.contains(doc.activeElement);
          entry.card.replaceChildren(); entry.card.setAttribute('aria-label', (item.state === 'proposed' ? 'Propuesta' : 'Recuerdo') + ' ' + (CATEGORY[item.category] || ''));
          buildCard(state, item, section, entry.card); entry.signature = signature;
          // A control disabled while its write is sent keeps focus on its card, never on the page body.
          if (hadFocus && !focusByKey(key)) (entry.card.querySelector('button:not([disabled]),summary') || entry.card).focus({ preventScroll: true });
        }
        const expected = previous ? previous.nextSibling : grid.firstChild;
        if (expected !== entry.card) grid.insertBefore(entry.card, expected);
        previous = entry.card;
      }
      [...grid.children].forEach((card, index) => {
        const id = card.dataset.memoryId;
        if (seen.has(id)) return;
        if (card.contains(doc.activeElement)) lostFocus = { index, key };
        card.remove(); if (cards.get(id)?.grid === grid) cards.delete(id);
      });
      if (lostFocus) return () => {
        // The moved card (a confirmation) keeps focus on its own heading when it is listed elsewhere.
        const target = grid.children[Math.min(lostFocus.index, grid.children.length - 1)];
        if (target) (target.querySelector('button:not([disabled]),summary') || target).focus({ preventScroll: true });
        else heading.focus({ preventScroll: true });
      };
      return null;
    }
    // 'Mostrar más' keeps focus on itself while more remain, else moves to the first newly shown card.
    function moreControl(host, section, total, label) {
      const remaining = total - Math.min(shown[section], total);
      const key = 'more:' + section, existing = host.querySelector('button');
      if (remaining <= 0) { if (existing) { const hadFocus = existing === doc.activeElement; host.replaceChildren(); if (hadFocus) pendingFocus = { section }; } return; }
      const text = `Mostrar más ${label} (quedan ${remaining})`;
      if (existing && existing.textContent === text) return;
      const control = button(text, key, () => { shown[section] += SHOW_STEP; firstNew = { section, index: shown[section] - SHOW_STEP }; render(); });
      const hadFocus = existing === doc.activeElement;
      host.replaceChildren(control);
      if (hadFocus) control.focus({ preventScroll: true });
    }
    let pendingFocus = null, firstNew = null;

    function itemFor(id) {
      const state = owner.getState();
      return [...state.facts, ...state.kept, ...state.proposals, ...state.returned].find(entry => entry.id === id) || null;
    }
    async function confirmDelete(id) {
      const state = owner.getState();
      const item = [...state.facts, ...state.kept, ...state.returned].find(entry => entry.id === id);
      if (!item) return;
      let others = owner.dependents(id), changed = false;
      // The cascade accepted must be the cascade sent: re-ask when a read changed it meanwhile (N5).
      for (let round = 0; round < 5; round++) {
        const accepted = await ask(item, others, state.factsFull || state.proposalsFull, id + ':delete', changed);
        if (!accepted || !active) return;
        const now = owner.getState();
        if (![...now.facts, ...now.kept, ...now.returned].some(entry => entry.id === id)) return;
        const current = owner.dependents(id);
        const same = current.length === others.length && current.every(entry => others.some(other => other.id === entry.id));
        if (same) break;
        others = current; changed = true;
        if (round === 4) return;
      }
      {
        const result = await owner.remove(id, { confirmed: true });
        if (active && result?.applied) { const target = factsGrid.querySelector('button:not([disabled]),summary'); (target || factsTitle).focus({ preventScroll: true }); }
      }
    }
    // One dialog at a time; a new one cancels the previous. Closing the view or hiding the tab empties it (F8).
    function dialog(parts, invoker, labelledBy) {
      if (confirmResolve) confirmResolve(false);
      return new Promise(resolve => {
        confirmDialog.replaceChildren(...parts.nodes);
        confirmDialog.setAttribute('aria-labelledby', labelledBy);
        let done = false;
        function finish(value) {
          if (done) return; done = true; confirmResolve = null;
          confirmDialog.removeEventListener('cancel', cancelled);
          if (confirmDialog.open) confirmDialog.close();
          confirmDialog.replaceChildren();
          if (active && !focusByKey(invoker)) factsTitle.focus({ preventScroll: true });
          resolve(value);
        }
        function cancelled(event) { event.preventDefault(); finish(false); }
        confirmDialog.addEventListener('cancel', cancelled);
        confirmResolve = finish;
        parts.bind(finish);
        if (!confirmDialog.open) confirmDialog.showModal();
        parts.focus();
      });
    }
    function target(item) {
      const box = node('div', 'memory-confirm-target');
      box.append(content(item, true).text, node('p', 'memory-detail', stateLine(item, false)));
      return box;
    }
    function actions(cancelText, acceptText, acceptClass) {
      const row = node('div', 'modal-actions memory-confirm-actions');
      const cancelButton = button(cancelText, 'confirm-cancel', () => {});
      const accept = button(acceptText, 'confirm-accept', () => {}, acceptClass);
      row.append(cancelButton, accept);
      return { row, cancelButton, accept };
    }
    function ask(item, others, full, invoker, changed = false) {
      const titleNode = node('h2', '', 'Eliminar este recuerdo'); titleNode.id = 'memory-confirm-title';
      const body = node('p', 'modal-description', 'Es definitivo: no se puede deshacer.' + (others.length ? ' También se eliminan los recuerdos derivados de este:' : ' No hay recuerdos derivados visibles.'));
      const nodes = [titleNode, target(item), body];
      if (changed) { const note = node('p', 'memory-detail memory-changed', 'La lista de recuerdos derivados cambió mientras decidías. Revisala antes de confirmar.'); note.setAttribute('role', 'alert'); nodes.splice(1, 0, note); }
      if (others.length) {
        const list = node('ul', 'memory-confirm-items');
        for (const other of others) {
          const line = node('li');
          const text = node('span', 'memory-content', other.content.length > 120 ? other.content.slice(0, 120).trimEnd() + '…' : other.content); text.dir = 'auto';
          line.append(text, node('span', 'memory-detail', ' · ' + (CATEGORY[other.category] || 'Otro') + ' · ' + STATE_LABEL[other.state]));
          list.append(line);
        }
        nodes.push(list);
      }
      if (full) nodes.push(node('p', 'memory-detail', 'Puede haber otros que no se muestran.'));
      if (item.owner) nodes.push(node('p', 'memory-detail', 'Está asignado a una persona: deja de verlo en Mis recuerdos.'));
      nodes.push(node('p', 'memory-detail', 'Las copias en índices de búsqueda se limpian en segundo plano; si la limpieza queda pendiente se indica en esta vista. No borra la conversación de origen (vence sola en 48 horas como máximo) ni el texto que ya se haya enviado a la nube para deducirlo.'));
      const { row, cancelButton, accept } = actions('Cancelar', 'Eliminar definitivamente', 'identity-danger');
      nodes.push(row);
      return dialog({ nodes, bind(finish) { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); }, focus: () => cancelButton.focus() },
        invoker, 'memory-confirm-title');
    }

    // Assignment: pick an active person; the copy states permanence, D5 and the server's A16 notice.
    async function openAssign(id) {
      const state = owner.getState(), item = itemFor(id);
      if (!item || !owner.assignable(id)) return;
      const choices = (state.people || []).filter(entry => entry.active && entry.id !== item.owner?.person_id);
      const titleNode = node('h2', '', item.owner ? 'Reasignar este recuerdo' : 'Asignar este recuerdo a una persona'); titleNode.id = 'memory-confirm-title';
      const nodes = [titleNode, target(item)];
      let select = null;
      if (choices.length) {
        const label = node('label', 'memory-person-field', 'Persona'); label.htmlFor = 'memory-assign-person';
        select = node('select'); select.id = 'memory-assign-person'; select.dataset.key = 'assign-person';
        for (const entry of choices) { const option = node('option', '', entry.name); option.value = entry.id; option.dir = 'auto'; select.append(option); }
        nodes.push(label, select);
      } else nodes.push(node('p', 'modal-description', 'No hay otras personas activas en el catálogo para asignar este recuerdo.'));
      const permanence = node('p', 'modal-description', ASSIGN_PERMANENCE); permanence.id = 'memory-assign-permanence';
      nodes.push(permanence, node('p', 'memory-detail', ASSIGN_OTHERS));
      if (state.assignNotice) { const a16 = node('p', 'memory-detail memory-assign-notice', state.assignNotice); nodes.push(a16); }
      if (item.owner) nodes.push(node('p', 'memory-detail', 'Quien lo tiene asignado ahora deja de verlo en Mis recuerdos.'));
      const { row, cancelButton, accept } = actions('Cancelar', item.owner ? 'Reasignar' : 'Asignar', 'identity-primary');
      if (!choices.length) accept.remove();
      nodes.push(row);
      const chosen = await dialog({ nodes, bind(finish) { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(select ? select.value : false); },
        focus: () => (select || cancelButton).focus() }, id + ':assign', 'memory-confirm-title');
      if (!chosen || !active) return;
      let result = await owner.assign(id, chosen, { confirmed: true });
      // 409 has_dependents: nothing changed; show ids and states and ask before deleting them (C8).
      if (result?.dependents && active) {
        const accepted = await askDependents(id, result, chosen);
        if (!accepted || !active) return;
        // The owner compares the receipt's ids with this list and says so on the card (R1).
        await owner.assign(id, chosen, { confirmed: true, deleteDependents: true, expected: result.expected, expectedDependents: accepted.count, expectedDependentIds: accepted.ids, expectedDependentLabels: accepted.labels });
      }
    }
    function askDependents(id, result, personId) {
      const item = itemFor(id);
      if (!item) return Promise.resolve(false);
      const state = owner.getState();
      const titleNode = node('h2', '', 'Este recuerdo tiene recuerdos derivados'); titleNode.id = 'memory-confirm-title';
      const nodes = [titleNode, target(item), node('p', 'modal-description', result.notice)];
      const list = node('ul', 'memory-confirm-items memory-dependents'); list.id = 'memory-assign-dependents';
      const labels = {};
      for (const dependent of result.dependents) {
        const line = node('li'); line.dataset.dependentId = dependent.id;
        const known = itemFor(dependent.id);
        if (known) { const text = node('span', 'memory-content', dependentLabel(known.content)); labels[dependent.id] = dependentLabel(known.content); text.dir = 'auto'; line.append(text, node('span', 'memory-detail', ' · ')); }
        else line.append(node('span', 'memory-detail', 'Recuerdo que no figura en esta vista · '));
        line.append(node('span', 'memory-detail', STATE_LABEL[dependent.state] + ' · '), node('code', 'memory-id', dependent.id));
        list.append(line);
      }
      const scope = node('p', 'memory-detail', DEPENDENTS_SCOPE); scope.id = 'memory-assign-dependents-scope';
      nodes.push(list, scope, node('p', 'memory-detail', 'Eliminarlos es definitivo. ' + ASSIGN_PERMANENCE));
      if (state.assignNotice) nodes.push(node('p', 'memory-detail memory-assign-notice', state.assignNotice));
      const name = personName(state, personId);
      const { row, cancelButton, accept } = actions('Cancelar', 'Eliminar derivados y asignar' + (name ? ' a ' + name : ''), 'identity-danger');
      nodes.push(row);
      return dialog({ nodes, bind(finish) { cancelButton.onclick = () => finish(false); accept.onclick = () => finish({ count: result.dependents.length, ids: result.dependents.map(dependent => dependent.id), labels }); },
        focus: () => cancelButton.focus() }, id + ':assign', 'memory-confirm-title');
    }
    async function openUnassign(id) {
      const state = owner.getState(), item = itemFor(id);
      if (!item?.owner || !state.capabilities.assign) return;
      const titleNode = node('h2', '', 'Quitar la asignación'); titleNode.id = 'memory-confirm-title';
      const name = personName(state, item.owner.person_id);
      const nodes = [titleNode, target(item), node('p', 'modal-description', (name ? name : 'La persona') + ' deja de verlo en Mis recuerdos. El recuerdo sigue guardado, confirmado y sin vencimiento hasta que lo elimines o lo vuelvas a asignar.')];
      const { row, cancelButton, accept } = actions('Cancelar', 'Quitar asignación', 'identity-primary');
      nodes.push(row);
      const accepted = await dialog({ nodes, bind(finish) { cancelButton.onclick = () => finish(false); accept.onclick = () => finish(true); }, focus: () => cancelButton.focus() },
        id + ':unassign', 'memory-confirm-title');
      if (accepted && active) await owner.unassign(id, { confirmed: true });
    }

    function render() {
      if (!active) return;
      const state = owner.getState(), caps = state.capabilities;
      refresh.setAttribute('aria-disabled', String(state.read.loading));
      permission.textContent = caps.review ? '' : 'Este acceso permite ver la memoria pero no revisarla.';
      permission.hidden = !permission.textContent;
      sessionNotice.textContent = state.notices.session || ''; sessionNotice.hidden = !sessionNotice.textContent;
      const time = state.read.observedWall ? new Date(state.read.observedWall).toLocaleTimeString('es-UY', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }) : null;
      freshness.textContent = state.read.error ? `${state.read.error}${time ? ' Última consulta confirmada a las ' + time + '.' : ''}${state.loaded ? ' Se muestran esos datos.' : ''}`
        : !state.loaded ? (state.read.loading ? 'Consultando la memoria…' : 'Sin consulta confirmada.') : 'Consultada a las ' + time + '.';
      freshness.classList.toggle('is-error', Boolean(state.read.error));
      const show = state.loaded;
      counts.textContent = show && caps.assign ? (state.counts
        ? `Asignados a personas: ${state.counts.currently_owned} · Devueltos: ${state.counts.returned} · Propuestas asignadas: ${state.counts.owned_proposals}`
        : state.countsError || '') : '';
      counts.hidden = !counts.textContent;
      prune.textContent = show && state.counts?.prune_pending ? 'Hay recuerdos asignados a cuentas ya retiradas que todavía no se terminaron de eliminar. KZA lo reintenta automáticamente; si persiste, revisá el estado del servidor.' : '';
      prune.hidden = !prune.textContent;
      peopleNote.textContent = show && caps.assign ? (!caps.people ? 'Este acceso no puede consultar las personas del hogar: no puede asignar recuerdos.'
        : state.peopleError || (!state.assignNotice ? 'El servidor no informó las condiciones de asignación; no se puede asignar por ahora.' : '')) : '';
      peopleNote.hidden = !peopleNote.textContent;
      cleanup.textContent = state.pendingCleanup > 0 ? 'Hay copias en índices de búsqueda pendientes de limpieza.' : '';
      cleanup.hidden = !cleanup.textContent;
      indexing.textContent = state.pendingIndex > 0 ? 'Actualizando el índice de búsqueda.' : ''; indexing.hidden = !indexing.textContent;
      quarantine.textContent = state.legacyPolicy === 'quarantined' ? 'Los registros anteriores a esta política están en cuarentena y no se muestran acá.' : '';
      quarantine.hidden = !quarantine.textContent;
      // Pending operations whose fact is not listed (after a reload, beyond the walk or once gone).
      const listed = new Set([...state.facts, ...state.proposals, ...state.kept, ...state.returned].map(item => item.id));
      const orphan = Object.entries(state.markers).filter(([id]) => !listed.has(id));
      const orphanKeys = new Set(orphan.map(([id]) => 'orphan:' + id));
      for (const element of [...pendingList.children]) if (!orphanKeys.has(element.dataset.orphan)) element.remove();
      for (const [id, marker] of orphan) {
        const keyName = 'orphan:' + id, writable = ['confirm', 'reject', 'delete'].includes(marker.op) ? caps.review : caps.assign;
        const signature = JSON.stringify([marker, state.notices[id] || null, writable]);
        let card = [...pendingList.children].find(element => element.dataset.orphan === keyName);
        if (!card) { card = node('article', 'memory-card panel memory-orphan'); card.dataset.orphan = keyName; card.setAttribute('aria-label', 'Operación sin confirmar'); pendingList.append(card); }
        if (card.dataset.signature !== signature) {
          const hadFocus = card.contains(doc.activeElement);
          card.replaceChildren(node('p', 'memory-state', markerText(marker) + ' un recuerdo que no figura en la lista.'));
          const message = node('p', 'memory-notice', state.notices[id] || ''); message.dataset.memoryNotice = id; card.append(message);
          if (writable) { const again = button('Reintentar ' + OP_LABEL[marker.op], id + ':retry', () => owner.retry(id)); again.disabled = marker.state === 'sending'; card.append(again); }
          card.dataset.signature = signature;
          if (hadFocus) focusByKey(id + ':retry');
        }
      }
      pendingTitle.hidden = pendingList.hidden = !orphan.length || !state.loaded;
      // 'Devueltos' (operator with memory:assign): each fact is shown once, here rather than in the lists below.
      const showReturned = show && caps.assign;
      for (const element of [returnedTitle, returnedNote, returnedGrid]) element.hidden = !showReturned;
      returnedStatus.textContent = showReturned ? (state.returnedError || (state.returnedLoaded && !state.returnedComplete ? 'La lista de devueltos es más larga de lo que esta vista puede recorrer; se muestran los primeros.' : '')) : '';
      returnedStatus.hidden = !returnedStatus.textContent;
      const returnedIds = new Set(showReturned ? state.returned.map(item => item.id) : []);
      const returned = showReturned ? state.returned : [];
      const proposals = show ? state.proposals.filter(item => !returnedIds.has(item.id)) : [];
      const facts = show ? [...state.kept.map(item => [item, 'kept']), ...state.facts.filter(item => !returnedIds.has(item.id)).map(item => [item, 'facts'])] : [];
      for (const element of [proposalsTitle, proposalsGrid, factsTitle, factsGrid]) element.hidden = !show;
      proposalsCap.textContent = show && state.proposalsFull ? 'La lista de propuestas es más larga de lo que esta vista puede recorrer. Las que no aparecen no se pueden revisar desde acá; vencen solas.' : '';
      factsCap.textContent = show && state.factsFull ? 'La lista de recuerdos es más larga de lo que esta vista puede recorrer. Los que no aparecen, incluidos recuerdos confirmados que no vencen, no se pueden revisar ni eliminar desde acá.' : '';
      proposalsCap.hidden = !proposalsCap.textContent; factsCap.hidden = !factsCap.textContent;
      const restoreReturned = place(returnedGrid, returned.slice(0, shown.returned).map(item => [item, 'returned']), state, returnedTitle);
      const restoreProposals = place(proposalsGrid, proposals.slice(0, shown.proposals).map(item => [item, 'proposals']), state, proposalsTitle);
      const restoreFacts = place(factsGrid, facts.slice(0, shown.facts), state, factsTitle);
      moreControl(returnedMore, 'returned', returned.length, 'devueltos');
      moreControl(proposalsMore, 'proposals', proposals.length, 'propuestas');
      moreControl(factsMore, 'facts', facts.length, 'recuerdos');
      // A confirmation moves its card to Recuerdos guardados: focus follows it there (F1c).
      const movedCard = restoreProposals && lastConfirmed ? cards.get(lastConfirmed) : null;
      if (movedCard && movedCard.grid === factsGrid) { (movedCard.card.querySelector('button:not([disabled]),summary') || factsTitle).focus({ preventScroll: true }); }
      else restoreProposals?.();
      if (restoreProposals) lastConfirmed = null;
      restoreFacts?.(); restoreReturned?.();
      for (const request of [firstNew, pendingFocus]) {
        if (!request) continue;
        const grid = { returned: returnedGrid, proposals: proposalsGrid, facts: factsGrid }[request.section];
        const card = grid.children[request.index ?? grid.children.length - 1];
        if (card) (card.querySelector('button:not([disabled]),summary') || card).focus({ preventScroll: true });
      }
      firstNew = pendingFocus = null;
      returnedEmpty.hidden = !showReturned || returned.length > 0 || Boolean(state.returnedError);
      proposalsEmpty.hidden = !show || proposals.length > 0; factsEmpty.hidden = !show || facts.length > 0;
      if (state.announcement && state.announcement.serial !== lastAnnouncement) { lastAnnouncement = state.announcement.serial; live.textContent = state.announcement.text; }
    }
    let lastConfirmed = null;
    container.addEventListener('click', event => {
      const key = event.target.closest?.('[data-key]')?.dataset.key || '';
      if (key.endsWith(':confirm')) lastConfirmed = key.slice(0, -':confirm'.length);
    }, true);
    const stop = owner.subscribe(render);
    if (owner.getState().announcement) lastAnnouncement = owner.getState().announcement.serial;
    render();
    // Private text lives in the dialog too: closing the view empties both (F8).
    return () => {
      active = false; stop();
      if (confirmResolve) confirmResolve(false);
      if (confirmDialog.open) confirmDialog.close();
      confirmDialog.replaceChildren(); container.replaceChildren(); cards.clear(); expanded.clear();
    };
  }
  const exported = { mountMemory, ASSIGN_PERMANENCE, ASSIGN_OTHERS, DEPENDENTS_SCOPE, dependentLabel };
  root.NexoMemory = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
