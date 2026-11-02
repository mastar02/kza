/* Session and catalog presentation; voice labels never select authority. */
(function (root) {
  'use strict';
  const VIEWS = { home: ['home:read'], actions: ['home:read'], voices: ['voices:read'], people: ['admin:manage'] };
  function createSession({ wall = Date.now, monotonic = () => root.performance.now() } = {}) {
    let identity = null, deadline = 0, generation = 0;
    function clear() { identity = null; deadline = 0; generation++; }
    function accept(value) {
      clear();
      const expires = value?.expires_at * 1000;
      if (value?.ok !== true || !['operator', 'home', 'evaluator'].includes(value.role)
          || !Array.isArray(value.permissions) || value.permissions.some(permission => typeof permission !== 'string')
          || !Number.isFinite(expires) || expires <= wall()) return false;
      identity = { role: value.role, permissions: new Set(value.permissions), expires };
      deadline = monotonic() + Math.max(0, expires - wall());
      return true;
    }
    function remaining() { return identity ? Math.max(0, Math.min(deadline - monotonic(), identity.expires - wall())) : 0; }
    function active() { return identity !== null && remaining() > 0; }
    function can(permission) { return active() && identity.permissions.has(permission); }
    return { accept, clear, active, can, remaining, generation: () => generation,
      role: () => active() ? identity.role : null,
      views: () => Object.keys(VIEWS).filter(view => VIEWS[view].every(can)) };
  }
  function sessionRetired(error) {
    return error?.status === 401 || error?.status === 403 &&
      ['dashboard_session_retired', 'dashboard_authority_retired'].includes(error.detail);
  }
  function createScopedApi({ session, transport, onRetired = () => {} }) {
    return async function api(path, options = {}) {
      const method = (options.method || 'GET').toUpperCase();
      const pathname = typeof path === 'string' ? path.split('?')[0] : '';
      const publicAuth = pathname === '/api/admin/auth/whoami' && method === 'GET' ||
        ['/api/admin/auth/login', '/api/admin/auth/logout'].includes(pathname) && method === 'POST';
      const permission = pathname === '/api/home/catalog' && method === 'GET' ? 'home:read' :
        pathname === '/api/home/actions' && method === 'GET' ? 'home:read' :
        pathname === '/api/home/actions' && method === 'POST' ? 'home:control' :
        pathname === '/api/home/microphones' && method === 'GET' ? 'diagnostics:read' : null;
      if (!publicAuth && (!permission || !session.can(permission))) {
        throw Object.assign(new Error('Este acceso no permite la operación.'), { status: 403, sent: false });
      }
      const generation = session.generation();
      try {
        const result = await transport(path, options);
        if (generation !== session.generation() || !publicAuth && !session.can(permission)) {
          throw Object.assign(new Error('La sesión dejó de estar vigente.'), { status: 403, detail: 'dashboard_session_retired' });
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
  const exported = { createSession, createScopedApi, sessionRetired, roomEntities, entityState };
  root.NexoState = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
