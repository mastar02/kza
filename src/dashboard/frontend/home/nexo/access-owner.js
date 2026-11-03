(function (root) {
  'use strict';
  const BASE = '/api/admin/auth';
  const ROLES = new Set(['operator', 'home', 'evaluator']);
  const MODES = new Set(['audio_preview', 'transcript_full']);
  const ACTIONS = new Set(['issue', 'revise', 'revoke']);
  const REASONS = new Set(['applied', 'actor_retired', 'scope_unavailable', 'grant_invalid', 'target_missing', 'writer_busy', 'revision_conflict', 'account_unavailable', 'capacity']);
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
  const FP = /^vf_[0-9a-f]{64}$/;
  function fields(value, required, optional = []) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || required.some(key => !Object.hasOwn(value, key))
        || Object.keys(value).some(key => !required.includes(key) && !optional.includes(key))) throw Error('invalid_access_metadata');
  }
  function identifier(value, maximum = 128) {
    if (typeof value !== 'string' || !value.length || value.length > maximum * 2 || [...value].length > maximum || /[\x00-\x1f\x7f]/.test(value)
        || [...value].some(char => char.length === 1 && char.charCodeAt(0) >= 0xd800 && char.charCodeAt(0) <= 0xdfff)) throw Error('invalid_access_identifier');
    return value;
  }
  function time(value) { if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw Error('invalid_access_time'); return value; }
  function revision(value) { if (!Number.isSafeInteger(value) || value < 1) throw Error('invalid_access_revision'); return value; }
  function selections(values) {
    if (!Array.isArray(values) || values.length > 64) throw Error('invalid_access_selections');
    const seen = new Set();
    return Object.freeze(values.map(value => {
      fields(value, ['fingerprint_id', 'modalities', 'expires_at']);
      if (typeof value.fingerprint_id !== 'string' || !FP.test(value.fingerprint_id) || seen.has(value.fingerprint_id)
          || !Array.isArray(value.modalities) || !value.modalities.length || value.modalities.length > 2
          || new Set(value.modalities).size !== value.modalities.length || value.modalities.some(mode => !MODES.has(mode))) throw Error('invalid_access_selection');
      seen.add(value.fingerprint_id);
      return Object.freeze({ fingerprint_id: value.fingerprint_id, modalities: Object.freeze(value.modalities.slice().sort()), expires_at: time(value.expires_at) });
    }).sort((a, b) => a.fingerprint_id.localeCompare(b.fingerprint_id)));
  }
  function normalizeGrant(value, issue = true) {
    fields(value, issue ? ['subject', 'role', 'expires_at'] : ['role', 'expires_at'], ['user_id', 'selections']);
    if (!ROLES.has(value.role)) throw Error('invalid_access_role');
    const user = value.user_id == null ? null : identifier(value.user_id), chosen = selections(value.selections === undefined ? [] : value.selections);
    if (user !== null && value.role !== 'home' || (value.role === 'evaluator') !== Boolean(chosen.length)) throw Error('invalid_access_scope');
    return Object.freeze({ ...(issue ? { subject: identifier(value.subject) } : {}), role: value.role,
      expires_at: time(value.expires_at), user_id: user, selections: chosen });
  }
  function parseCatalog(document) {
    fields(document, ['rows', 'observed_at', 'next_cursor']);
    if (!Array.isArray(document.rows) || document.rows.length > 50) throw Error('invalid_access_catalog');
    const seen = new Set();
    const rows = document.rows.map(row => {
      fields(row, ['id', 'subject', 'role', 'user_id', 'issued_at', 'expires_at', 'revision', 'active', 'pending', 'samples']);
      const id = identifier(row.id), user = row.user_id === null ? null : identifier(row.user_id);
      if (seen.has(id) || !ROLES.has(row.role) || typeof row.active !== 'boolean' || typeof row.pending !== 'boolean'
          || user !== null && row.role !== 'home' || time(row.issued_at) >= time(row.expires_at)) throw Error('invalid_access_catalog');
      seen.add(id);
      const samples = selections(row.samples);
      if ((row.role === 'evaluator') !== Boolean(samples.length)) throw Error('invalid_access_catalog');
      return Object.freeze({ id, subject: identifier(row.subject), role: row.role, user_id: user,
        issued_at: row.issued_at, expires_at: row.expires_at, revision: revision(row.revision), active: row.active, pending: row.pending, samples });
    });
    const next = document.next_cursor === null ? null : identifier(document.next_cursor, 256);
    if (next !== null && (!rows.length || next !== rows.at(-1).id)) throw Error('invalid_access_cursor');
    return Object.freeze({ rows: Object.freeze(rows), observed_at: time(document.observed_at), next_cursor: next });
  }
  function parsePeople(document) {
    if (!Array.isArray(document) || document.length > 256) throw Error('invalid_access_people');
    const seen = new Set();
    return Object.freeze(document.map(row => {
      fields(row, ['id', 'name', 'permission_level', 'is_active']);
      const id = identifier(row.id);
      if (seen.has(id) || typeof row.is_active !== 'boolean' || !['guest', 'child', 'teen', 'adult', 'admin'].includes(row.permission_level)) throw Error('invalid_access_people');
      seen.add(id);
      // The account's PermissionLevel is deliberately not projected as a DashboardRole.
      return Object.freeze({ id, name: identifier(row.name), is_active: row.is_active });
    }));
  }
  function parseReceipt(value) {
    fields(value, ['request_id', 'action', 'credential_id', 'expected_revision', 'outcome', 'result_revision', 'reason', 'completed_at']);
    if (typeof value.request_id !== 'string' || !UUID.test(value.request_id) || !ACTIONS.has(value.action) || !REASONS.has(value.reason)
        || !['committed', 'no_effect'].includes(value.outcome) || (value.outcome === 'committed') !== (value.reason === 'applied')) throw Error('invalid_access_receipt');
    const expected = value.expected_revision === null ? null : revision(value.expected_revision);
    const result = value.result_revision === null ? null : revision(value.result_revision), id = identifier(value.credential_id);
    if ((value.action === 'issue') !== (expected === null) || value.action === 'issue' && id !== value.request_id
        || value.outcome === 'no_effect' && result !== null || value.outcome === 'committed' && (result === null
          || value.action === 'issue' && result !== 1 || value.action === 'revise' && result !== expected + 1 || value.action === 'revoke' && result !== expected)) throw Error('invalid_access_receipt');
    return Object.freeze({ request_id: value.request_id, action: value.action, credential_id: id, expected_revision: expected,
      outcome: value.outcome, result_revision: result, reason: value.reason, completed_at: time(value.completed_at) });
  }
  function buildCommand(action, id, expected, spec, requestId) {
    if (!ACTIONS.has(action) || typeof requestId !== 'string' || !UUID.test(requestId)) throw Error('invalid_access_command');
    if (action === 'issue') return Object.freeze({ request_id: requestId, ...normalizeGrant(spec, true) });
    identifier(id); const body = { request_id: requestId, expected_revision: revision(expected) };
    return Object.freeze(action === 'revise' ? { ...body, ...normalizeGrant(spec, false) } : body);
  }
  function defaultId() {
    if (typeof root.HomeRequestIdentity?.newRequestId !== 'function') throw Error('access_request_identity_unavailable');
    return root.HomeRequestIdentity.newRequestId();
  }
  function createAccessOwner({ session, fetch: rawFetch = root.fetch?.bind(root), verifyIdentity,
      onUnauthorized = () => {}, newRequestId = defaultId, timeoutMs = 12000 } = {}) {
    if (!session || typeof rawFetch !== 'function') throw Error('access_owner_dependencies');
    const listeners = new Set(), tasks = new Set(), attempts = new Map();
    let context = null, intentGeneration = null, openingVersion = 0, closing = Promise.resolve(true);
    const allowed = () => session.can('auth:manage') && session.role() === 'operator';
    const owns = ctx => context === ctx && !ctx.sealed && ctx.generation === session.generation() && allowed();
    const uncertain = () => [...attempts.values()].some(value => value.state !== 'terminal');
    function snapshot() {
      const ctx = context, live = ctx && owns(ctx);
      return Object.freeze({ generation: session.generation(), visible: Boolean(live), status: live ? ctx.status : 'closed',
        known: Boolean(live && ctx.known), busy: Boolean(live && ctx.busy), loading: Boolean(live && ctx.loading),
        canMutate: Boolean(live && ctx.known && !ctx.busy && !ctx.loading && !uncertain()),
        rows: live ? ctx.rows : Object.freeze([]), people: live ? ctx.people : Object.freeze([]), peopleKnown: Boolean(live && ctx.peopleKnown),
        observed_at: live ? ctx.observed : null, canNext: Boolean(live && ctx.known && !ctx.busy && !ctx.loading && ctx.next),
        canPrevious: Boolean(live && ctx.index > 0 && !ctx.busy && !ctx.loading), page: live ? ctx.index + 1 : 0,
        notice: live ? ctx.notice : '', error: Boolean(live && ctx.error),
        pending: Object.freeze((intentGeneration === session.generation() ? [...attempts.values()] : []).map(value => Object.freeze({ ...value }))) });
    }
    function emit() { const value = snapshot(); for (const listener of listeners) { try { listener(value); } catch (_) { /* A view cannot break ownership. */ } } }
    function report(ctx, text, error = false) { if (owns(ctx)) { ctx.notice = text; ctx.error = error; emit(); } }
    function track(operation) {
      let task; task = Promise.resolve().then(operation).catch(() => false).finally(() => { tasks.delete(task); });
      tasks.add(task); return task;
    }
    async function verify(ctx) {
      const abort = new AbortController(); ctx.aborts.add(abort);
      const timer = setTimeout(() => abort.abort(), timeoutMs);
      let response = null, document = null, ok = null;
      try {
        if (!owns(ctx)) return false;
        if (verifyIdentity) ok = await verifyIdentity({ signal: abort.signal });
        else {
          response = await rawFetch(`${BASE}/whoami`, { credentials: 'same-origin', cache: 'no-store', signal: abort.signal });
          if (!owns(ctx) || abort.signal.aborted) { await response.body?.cancel(); return false; }
          if (response.ok) { document = await response.json(); ok = document?.ok === true && document.role === 'operator' && document.permissions?.includes('auth:manage'); }
          else { if ([401, 403].includes(response.status)) ok = false; await response.body?.cancel(); }
        }
        if (owns(ctx) && !abort.signal.aborted && ok === false) onUnauthorized();
        return owns(ctx) && !abort.signal.aborted && ok === true;
      } catch (_) { return false; }
      finally { clearTimeout(timer); ctx.aborts.delete(abort); response = document = null; }
    }
    async function receive(ctx, path, method, encoded, accept) {
      const abort = new AbortController(); ctx.aborts.add(abort);
      const timer = setTimeout(() => abort.abort(), timeoutMs);
      let response = null, document = null;
      try {
        if (!owns(ctx)) return false;
        response = await rawFetch(path, { method, credentials: 'same-origin', cache: 'no-store', signal: abort.signal,
          ...(encoded === null ? {} : { body: encoded, headers: { 'Content-Type': 'application/json' } }) });
        if (!owns(ctx) || abort.signal.aborted) { await response.body?.cancel(); return false; }
        if (!response.ok) {
          const status = response.status; await response.body?.cancel(); response = null;
          // A target can retire a token response while its operator remains valid.
          if (status === 401 || status === 403) await verify(ctx);
          return false;
        }
        document = await response.json();
        if (!owns(ctx) || abort.signal.aborted) return false;
        accept(document); return true;
      } finally { clearTimeout(timer); ctx.aborts.delete(abort); response = document = accept = encoded = null; }
    }
    async function readCatalog(ctx) {
      if (!owns(ctx)) return false;
      const version = ++ctx.readVersion, cursor = ctx.cursors[ctx.index];
      ctx.loading = true; ctx.known = false; ctx.rows = Object.freeze([]); ctx.status = 'loading'; emit();
      try {
        const ok = await receive(ctx, `${BASE}/credentials${cursor === null ? '' : `?after=${encodeURIComponent(cursor)}`}`, 'GET', null, document => {
          const value = parseCatalog(document);
          if (owns(ctx) && version === ctx.readVersion) { ctx.rows = value.rows; ctx.observed = value.observed_at; ctx.next = value.next_cursor; ctx.known = true; ctx.status = 'real'; }
        });
        if (!owns(ctx) || version !== ctx.readVersion) return false;
        if (!ok || !ctx.known) { ctx.status = 'unavailable'; report(ctx, 'No pudimos consultar los accesos. Actualizá para verificar el estado.', true); return false; }
        return true;
      } catch (_) { if (owns(ctx) && version === ctx.readVersion) { ctx.status = 'unavailable'; report(ctx, 'El catálogo no está disponible.', true); } return false; }
      finally { if (owns(ctx) && version === ctx.readVersion) { ctx.loading = false; emit(); } }
    }
    function saveAttempt(ctx, id, changes) {
      if (intentGeneration !== ctx.generation || !attempts.has(id)) return;
      attempts.set(id, Object.freeze({ ...attempts.get(id), ...changes })); if (owns(ctx)) emit();
    }
    async function reconcileOne(ctx, id) {
      const original = attempts.get(id);
      if (!owns(ctx) || !UUID.test(id) || !original) return false;
      let terminal = false;
      try {
        const ok = await receive(ctx, `${BASE}/credential-attempts/${id}`, 'GET', null, document => {
          fields(document, ['state', 'receipt']);
          if (!['terminal', 'pending', 'unconfirmed'].includes(document.state)) throw Error('invalid_access_attempt');
          if (document.state !== 'terminal') { if (document.receipt !== null) throw Error('invalid_access_attempt'); saveAttempt(ctx, id, { state: document.state }); return; }
          const receipt = parseReceipt(document.receipt);
          if (receipt.request_id !== id || receipt.action !== original.action || receipt.credential_id !== original.credential_id
              || receipt.expected_revision !== original.expected_revision) throw Error('invalid_access_attempt');
          terminal = true; saveAttempt(ctx, id, { state: 'terminal', receipt });
          report(ctx, receipt.outcome === 'committed' ? 'El recibo confirma que la operación se aplicó. La credencial secreta no se puede recuperar.' : 'El recibo confirma que no se aplicó el cambio.');
        });
        if (ok && terminal) await readCatalog(ctx);
        else if (owns(ctx)) report(ctx, 'El resultado sigue sin confirmarse. Consultá nuevamente el recibo; no se reenvió la operación.', true);
        return ok && terminal;
      } catch (_) { report(ctx, 'No pudimos consultar el recibo. La operación no se reenvió.', true); return false; }
    }
    function mutation(action, id, expected, spec, acceptSecret) {
      const ctx = context;
      if (!ctx || !owns(ctx) || !ctx.known || ctx.busy || ctx.loading || uncertain() || tasks.size >= 8) return Promise.resolve(false);
      let command, encoded, requestId;
      try {
        if (attempts.size >= 64) { const old = [...attempts.values()].find(value => value.state === 'terminal'); if (old) attempts.delete(old.request_id); else throw Error('access_attempt_capacity'); }
        requestId = newRequestId(); if (attempts.has(requestId)) throw Error('duplicate_access_intention');
        command = buildCommand(action, id, expected, spec, requestId); encoded = JSON.stringify(command);
      } catch (_) { report(ctx, 'Revisá los datos del acceso antes de enviarlo.', true); return Promise.resolve(false); }
      const target = action === 'issue' ? requestId : id;
      attempts.set(requestId, Object.freeze({ request_id: requestId, action, credential_id: target,
        expected_revision: action === 'issue' ? null : expected, state: 'pending', receipt: null }));
      ctx.busy = true; report(ctx, 'Operación enviada. Esperando el recibo…');
      const path = action === 'issue' ? `${BASE}/credentials` : `${BASE}/credentials/${encodeURIComponent(id)}`;
      const method = { issue: 'POST', revise: 'PUT', revoke: 'DELETE' }[action];
      return track(async () => {
        let ok = false;
        try {
          ok = await receive(ctx, path, method, encoded, document => {
            fields(document, ['receipt', 'applied'], ['token']);
            const receipt = parseReceipt(document.receipt);
            if (receipt.request_id !== requestId || receipt.action !== action || receipt.credential_id !== target
                || receipt.expected_revision !== (action === 'issue' ? null : expected) || typeof document.applied !== 'boolean'
                || document.applied && receipt.outcome !== 'committed') throw Error('invalid_access_result');
            let secret = Object.hasOwn(document, 'token') ? document.token : null;
            try {
              if (secret !== null && (typeof secret !== 'string' || action !== 'issue' || !document.applied || !/^kza_[A-Za-z0-9_-]{43}$/.test(secret))) throw Error('invalid_access_secret');
              if (action === 'issue' && document.applied && secret === null) throw Error('missing_access_secret');
              saveAttempt(ctx, requestId, { state: 'terminal', receipt });
              if (secret !== null && owns(ctx) && typeof acceptSecret === 'function') {
                try { acceptSecret(secret, Object.freeze({ request_id: requestId, credential_id: target })); }
                catch (_) { /* The durable receipt stays terminal even if the view cannot show its one-time value. */ }
              }
              report(ctx, receipt.outcome === 'committed' ? 'El backend confirmó la operación.' : 'El backend confirmó que el cambio no se aplicó.');
            } finally { secret = null; }
          });
          if (!ok) { saveAttempt(ctx, requestId, { state: 'unconfirmed' }); if (owns(ctx)) await reconcileOne(ctx, requestId); }
          else if (owns(ctx)) await readCatalog(ctx);
          return ok;
        } catch (_) {
          saveAttempt(ctx, requestId, { state: 'unconfirmed' }); report(ctx, 'No se confirmó el resultado. Consultá el recibo; no vuelvas a crear el mismo acceso.', true);
          if (owns(ctx)) await reconcileOne(ctx, requestId); return false;
        } finally { encoded = command = spec = acceptSecret = null; ctx.busy = false; if (owns(ctx)) emit(); }
      });
    }
    function leave() {
      openingVersion++; const previous = context; context = null;
      if (previous) { previous.sealed = true; previous.readVersion++; for (const abort of previous.aborts) abort.abort(); previous.rows = previous.people = Object.freeze([]); previous.known = previous.peopleKnown = false; }
      emit(); closing = Promise.allSettled([...tasks]).then(() => true); return closing;
    }
    async function open() {
      if (context && owns(context)) return true;
      if (context) leave();
      const generation = session.generation(), version = ++openingVersion;
      if (!allowed()) { await leave(); return false; }
      if (intentGeneration !== generation) { attempts.clear(); intentGeneration = generation; }
      await closing;
      if (version !== openingVersion || !allowed() || session.generation() !== generation) return false;
      const ctx = { generation, sealed: false, aborts: new Set(), readVersion: 0, cursors: [null], index: 0,
        next: null, rows: Object.freeze([]), people: Object.freeze([]), peopleKnown: false, peopleLoading: false, known: false,
        busy: false, loading: false, reconciling: new Set(), observed: null, status: 'loading', notice: '', error: false };
      context = ctx; emit();
      await track(async () => { await readCatalog(ctx); for (const item of [...attempts.values()]) if (owns(ctx) && item.state !== 'terminal') await reconcileOne(ctx, item.request_id); return owns(ctx); });
      return owns(ctx);
    }
    return {
      open, leave, clear() { attempts.clear(); intentGeneration = null; return leave(); }, settled: () => Promise.all([closing, ...tasks]).then(() => true), snapshot,
      subscribe(listener) { listeners.add(listener); listener(snapshot()); return () => listeners.delete(listener); },
      refresh() { const ctx = context; if (!ctx || !owns(ctx) || ctx.busy || ctx.loading || tasks.size >= 8) return Promise.resolve(false); ctx.loading = true; return track(() => readCatalog(ctx)); },
      next() { const ctx = context; if (!ctx || !owns(ctx) || ctx.busy || ctx.loading || !ctx.next || tasks.size >= 8) return Promise.resolve(false); ctx.cursors = ctx.cursors.slice(0, ctx.index + 1); ctx.cursors.push(ctx.next); ctx.index++; ctx.loading = true; return track(() => readCatalog(ctx)); },
      previous() { const ctx = context; if (!ctx || !owns(ctx) || ctx.busy || ctx.loading || !ctx.index || tasks.size >= 8) return Promise.resolve(false); ctx.index--; ctx.loading = true; return track(() => readCatalog(ctx)); },
      loadPeople() {
        const ctx = context; if (!ctx || !owns(ctx) || ctx.peopleLoading || tasks.size >= 8 || !session.can('admin:manage')) return Promise.resolve(false);
        ctx.peopleLoading = true;
        return track(async () => { try { const ok = await receive(ctx, '/api/admin/users', 'GET', null, document => { ctx.people = parsePeople(document); ctx.peopleKnown = true; });
          if (!ok && owns(ctx)) { ctx.people = Object.freeze([]); ctx.peopleKnown = false; } if (owns(ctx)) emit(); return ok;
        } catch (_) { if (owns(ctx)) { ctx.people = Object.freeze([]); ctx.peopleKnown = false; emit(); } return false; }
        finally { ctx.peopleLoading = false; } });
      },
      issue: (spec, options = {}) => mutation('issue', null, null, spec, options.acceptSecret),
      revise: (id, expected, spec) => mutation('revise', id, expected, spec, null), revoke: (id, expected) => mutation('revoke', id, expected, null, null),
      reconcile(id) {
        const ctx = context;
        if (!ctx || !owns(ctx) || ctx.busy || ctx.reconciling.has(id) || tasks.size >= 8) return Promise.resolve(false);
        ctx.reconciling.add(id);
        return track(async () => { try { return await reconcileOne(ctx, id); } finally { ctx.reconciling.delete(id); } });
      }
    };
  }
  const exported = { createAccessOwner, normalizeGrant, parseSelections: selections, parseCatalog, parsePeople, parseReceipt, buildCommand };
  root.NexoAccessOwner = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
