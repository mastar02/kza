/* Session and catalog presentation; voice labels never select authority. */
(function (root) {
  'use strict';
  const VIEWS = { home: ['home:read'], actions: ['home:read'], routines: ['home:read'], tasks: ['tasks:read'], memory: ['memory:read'], mymemory: ['memory:own:read'], voices: ['voices:read', 'voices:review'], people: ['admin:manage'], access: ['auth:manage'], evaluation: ['evaluation:read'] };
  function createSession({ wall = Date.now, monotonic = () => root.performance.now() } = {}) {
    let identity = null, deadline = 0, generation = 0;
    function clear() { identity = null; deadline = 0; generation++; }
    function accept(value) {
      clear();
      const expires = value?.expires_at * 1000;
      if (value?.ok !== true || !['operator', 'home', 'evaluator'].includes(value.role)
          || !Array.isArray(value.permissions) || value.permissions.some(permission => typeof permission !== 'string')
          || !Number.isFinite(expires) || expires <= wall()) return false;
      // Account identifiers scope this tab's data; permissions come from the server grant.
      identity = { role: value.role, permissions: new Set(value.permissions), expires,
        subject: typeof value.subject === 'string' && value.subject ? value.subject : null,
        routineActor: typeof value.routine_actor === 'string' && value.routine_actor ? value.routine_actor : null,
        accountBinding: typeof value.account_binding === 'string' ? value.account_binding : null };
      deadline = monotonic() + Math.max(0, expires - wall());
      return true;
    }
    function remaining() { return identity ? Math.max(0, Math.min(deadline - monotonic(), identity.expires - wall())) : 0; }
    function active() { return identity !== null && remaining() > 0; }
    function can(permission) { return active() && identity.permissions.has(permission); }
    return { accept, clear, active, can, remaining, generation: () => generation,
      role: () => active() ? identity.role : null,
      subject: () => active() ? identity.subject : null,
      routineActor: () => active() ? identity.routineActor : null,
      accountBinding: () => active() ? identity.accountBinding : null,
      views: () => Object.keys(VIEWS).filter(view => VIEWS[view].every(can)
        && (view !== 'evaluation' || identity?.role === 'evaluator') && (view !== 'access' || identity?.role === 'operator')
        // 'Mis recuerdos' is a person's own view: only household credentials bound to a person carry memory:own:read.
        && (view !== 'mymemory' || identity?.role === 'home')) };
  }
  function sessionRetired(error) {
    const code = typeof error?.detail === 'string' ? error.detail : error?.detail?.code;
    return error?.status === 401 || error?.status === 403 &&
      ['dashboard_session_retired', 'dashboard_authority_retired'].includes(code);
  }
  const canReadRoutineRuns = can => can('routines:execute') || can('routines:manage');
  const canExecuteRoutine = can => can('home:control') && canReadRoutineRuns(can);
  const canReviewRoutineProposals = session => ['operator', 'home'].includes(session.role?.()) && session.can('routines:manage');
  const ROUTINE_PROPOSALS = '/api/routines/proposals';
  const ROUTINE_PROPOSAL_ID = /^[0-9a-f]{32}$/;

  // Mirrors the effective server requirement: auth.py route permission plus the handler's own set.
  function routinePermission(pathname, method) {
    const item = /^\/api\/routines\/([^/]+)(?:\/(.+))?$/.exec(pathname);
    if (pathname === '/api/routines') return method === 'GET' ? 'home:read' : method === 'POST' ? 'routines:manage' : null;
    if (!item) return null;
    const [, id, rest] = item;
    if (id === 'preview') return rest === undefined && method === 'POST' ? 'routines:manage' : null;
    if (id === 'proposals') {
      if (rest === undefined) return method === 'GET' ? 'routines:manage' : null;
      const [proposalId, decision, extra] = rest.split('/');
      if (!ROUTINE_PROPOSAL_ID.test(proposalId) || extra !== undefined) return null;
      return (decision === undefined && method === 'GET' ||
        ['confirm', 'reject'].includes(decision) && method === 'POST') ? 'routines:manage' : null;
    }
    if (rest === undefined) return method === 'GET' ? 'home:read' : ['PUT', 'DELETE'].includes(method) ? 'routines:manage' : null;
    if (['enable', 'disable'].includes(rest) && method === 'POST') return 'routines:manage';
    if (rest === 'execute' && method === 'POST') return canExecuteRoutine;
    if ((rest === 'attempts' || /^runs\/[^/]+$/.test(rest)) && method === 'GET') return canReadRoutineRuns;
    return null;
  }
  // Only the routes the Memoria and Mis recuerdos views use, each with the effective server requirement:
  // auth.py _route_permissions plus the handler's own permission (memory_api.py). The read-only
  // /facts and /proposals lists and unknown /mine subpaths stay unmapped (refused locally).
  const FACT = 'fact_[0-9a-f]{64}';
  function memoryPermission(pathname, method) {
    if (pathname === '/api/memory') return method === 'GET' ? 'memory:read' : null;
    if (method === 'POST' && new RegExp(`^/api/memory/proposals/${FACT}/(?:confirm|reject)$`).test(pathname)) return 'memory:review';
    if (method === 'DELETE' && new RegExp(`^/api/memory/facts/${FACT}$`).test(pathname)) return 'memory:review';
    // Operator owner routes: route memory:assign (+ memory:read for reads), handler memory:assign.
    if (method === 'POST' && new RegExp(`^/api/memory/facts/${FACT}/owner$`).test(pathname)) return 'memory:assign';
    if (method === 'GET' && ['/api/memory/returned', '/api/memory/owners/counts'].includes(pathname)) return ['memory:assign', 'memory:read'];
    // The person's own memories: memory:own:read for reads, memory:own:write otherwise.
    if (pathname === '/api/memory/mine') return method === 'GET' ? 'memory:own:read' : null;
    if (method === 'DELETE' && new RegExp(`^/api/memory/mine/facts/${FACT}$`).test(pathname)) return 'memory:own:write';
    if (method === 'POST' && new RegExp(`^/api/memory/mine/facts/${FACT}/return$`).test(pathname)) return 'memory:own:write';
    return voicePermission(pathname, method);
  }
  // U3b-4a-3: the voice opt-in routes (memory_voice_api.py), each with auth.py's route permission
  // plus the handler's own guard. Unknown /mine/voice subpaths stay unmapped (refused locally).
  const PERSON_ID = '[A-Za-z0-9_.:@-]{1,128}';
  const PROFILE_ID = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}';
  function voicePermission(pathname, method) {
    if (pathname === '/api/memory/mine/voice') return method === 'GET' ? 'memory:own:voice' : null;
    if (pathname === '/api/memory/mine/voice/consent') return ['POST', 'DELETE'].includes(method) ? 'memory:own:voice' : null;
    if (pathname === '/api/memory/mine/voice/verification') return method === 'POST' ? 'memory:own:voice' : null;
    if (pathname === '/api/memory/voice/consents') return method === 'GET' ? ['memory:assign', 'memory:read'] : null;
    if (method === 'POST' && new RegExp(`^/api/memory/voice/consents/${PERSON_ID}/reopen$`).test(pathname)) return 'memory:assign';
    if (pathname === '/api/memory/voice/switch') return method === 'GET' ? ['memory:assign', 'memory:read'] : method === 'POST' ? ['admin:manage', 'memory:assign'] : null;
    if (method === 'POST' && new RegExp(`^/api/voices/profiles/${PROFILE_ID}/attribution/resume$`).test(pathname)) return ['admin:manage', 'memory:assign', 'voices:review'];
    return null;
  }
  const NOTICE_LIST = '/api/reminders/notices', NOTICE_DISMISS = NOTICE_LIST + '/dismiss';
  function canReadNotices(session) {
    return ['home', 'operator'].includes(session.role()) && session.can('home:read') && session.can('tasks:read');
  }
  function taskPermission(pathname, method, query, body) {
    const collection = pathname === '/api/lists' || pathname === '/api/reminders';
    const items = /^\/api\/lists\/[^/]+\/items$/.test(pathname);
    const item = /^\/api\/lists\/[^/]+\/items\/[^/]+$/.test(pathname);
    const reminder = /^\/api\/reminders\/[^/]+$/.test(pathname) &&
      ![NOTICE_LIST, '/api/reminders/status'].includes(pathname);
    if (!(collection || items || item || reminder)) return null;
    if (query !== undefined && !(collection && method === 'GET')) return null;
    const params = new URLSearchParams(query);
    const scopes = params.getAll('scope');
    if ([...params.keys()].some(key => key !== 'scope') || scopes.length > 1 ||
        scopes.some(scope => !['shared', 'personal'].includes(scope))) return null;
    if (method === 'GET' && (collection || items)) {
      return scopes[0] === 'personal' ? ['tasks:read', 'tasks:personal'] : 'tasks:read';
    }
    if (method === 'POST' && (collection || items)) {
      if (body && Object.hasOwn(body, 'user_id')) return null;
      const personal = body && collection && (pathname === '/api/lists' ? body.shared !== true : body.scope !== 'shared');
      return personal ? ['tasks:write', 'tasks:personal'] : 'tasks:write';
    }
    return method === 'DELETE' && (item || reminder) ? 'tasks:write' : null;
  }
  function createScopedApi({ session, transport, onRetired = () => {} }) {
    return async function api(path, options = {}) {
      const method = (options.method || 'GET').toUpperCase();
      const [pathname, ...queryParts] = typeof path === 'string' ? path.split('?') : [''];
      const query = queryParts.length ? queryParts.join('?') : undefined;
      const publicAuth = pathname === '/api/admin/auth/whoami' && method === 'GET' ||
        ['/api/admin/auth/login', '/api/admin/auth/logout'].includes(pathname) && method === 'POST';
      const noticeRoute = pathname === NOTICE_LIST || pathname === NOTICE_DISMISS;
      const proposalRoute = pathname === ROUTINE_PROPOSALS || pathname.startsWith(ROUTINE_PROPOSALS + '/');
      const permission = pathname === '/api/home/catalog' && method === 'GET' ? 'home:read' :
        pathname === '/api/home/actions' && method === 'GET' ? 'home:read' :
        pathname === '/api/home/actions' && method === 'POST' ? 'home:control' :
        pathname === '/api/home/microphones' && method === 'GET' ? 'diagnostics:read' :
        pathname === NOTICE_LIST && method === 'GET' ? 'tasks:read' :
        pathname === NOTICE_DISMISS && method === 'POST' ? 'tasks:write' :
        pathname === '/api/admin/users' && ['GET', 'POST'].includes(method) ? 'admin:manage' :
        /^\/api\/admin\/users\/[^/]+\/permissions$/.test(pathname) && method === 'PUT' ? 'admin:manage' :
        /^\/api\/admin\/users\/[^/]+$/.test(pathname) && method === 'DELETE' ? ['admin:manage', 'auth:manage'] :
        routinePermission(pathname, method) || memoryPermission(pathname, method) ||
        taskPermission(pathname, method, query, options.body);
      const permitted = () => (!noticeRoute || canReadNotices(session)) &&
        (!proposalRoute || canReviewRoutineProposals(session)) && (
        typeof permission === 'function' ? permission(session.can) :
        Boolean(permission) && (Array.isArray(permission) ? permission : [permission]).every(session.can));
      if (!publicAuth && !permitted()) {
        throw Object.assign(new Error('Este acceso no permite la operación.'), { status: 403, sent: false });
      }
      const generation = session.generation();
      try {
        const result = await transport(path, options);
        if (generation !== session.generation() || !publicAuth && !permitted()) {
          // The server already answered: callers that track side effects must not read this as a denial.
          throw Object.assign(new Error('La sesión dejó de estar vigente.'), { status: 403, detail: 'dashboard_session_retired', sent: true, answered: true });
        }
        return result;
      } catch (error) {
        if (generation === session.generation() && sessionRetired(error)) onRetired(error);
        throw error;
      }
    };
  }
  function roomEntities(catalog, roomId, layer) {
    const room = catalog?.rooms?.find(room => room.room_id === roomId);
    const ids = new Set(room?.entity_ids || []);
    return (catalog?.entities || []).filter(entity => ids.has(entity.entity_id) &&
      (layer === 'climate' ? entity.domain === 'climate' : layer === 'lights' ? entity.domain === 'light' : true));
  }
  const MODE_LABELS = { off: 'Apagado', cool: 'Frío', heat: 'Calor', dry: 'Deshumidificar', fan_only: 'Ventilación', auto: 'Automático', heat_cool: 'Frío y calor' };
  function entityState(entity) {
    if (entity.available !== true || ['unknown', 'unavailable'].includes(entity.state)) return 'Estado desconocido';
    if (entity.domain === 'climate') return `Modo ${MODE_LABELS[entity.state] || entity.state}`;
    return ({ on: 'Encendido', off: 'Apagado', open: 'Abierto', closed: 'Cerrado', opening: 'Abriendo', closing: 'Cerrando' })[entity.state] || entity.state || 'Estado desconocido';
  }
  const exported = { createSession, createScopedApi, sessionRetired, canReadRoutineRuns, canExecuteRoutine,
    ROUTINE_PROPOSALS, ROUTINE_PROPOSAL_ID, canReviewRoutineProposals,
    memoryPermission, voicePermission, roomEntities, entityState, NOTICE_LIST, NOTICE_DISMISS, canReadNotices };
  root.NexoState = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
