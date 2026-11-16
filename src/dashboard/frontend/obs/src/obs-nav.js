/* Hash <-> view logic for the technical panel navigation (WEB-03, WEB-05 /obs/).
 *
 * The URL hash names the open view so that Back and Forward restore it and a
 * reload opens the same view. An unknown or empty hash opens the fallback view
 * without rewriting the URL. Pure functions only: app.jsx owns the DOM, the
 * history writes and the focus move.
 */
(function (root) {
  'use strict';

  function viewFromHash(hash, ids, fallback) {
    const requested = String(hash || '').replace(/^#/, '');
    return ids.includes(requested) ? requested : fallback;
  }

  function hashForView(id) {
    return '#' + id;
  }

  // A click or key on a navigation entry. `show` opens the view now; `write` is
  // the hash to push so that Back returns to the previous view (null: none).
  function planNavigation(current, target, hash, ids) {
    if (!ids.includes(target) || target === current) return { show: false, write: null };
    const wanted = hashForView(target);
    return { show: true, write: hash === wanted ? null : wanted };
  }

  // A hashchange (Back, Forward, a typed or followed #view): the view to open.
  function planHashChange(current, hash, ids, fallback) {
    const view = viewFromHash(hash, ids, fallback);
    return { view, changed: view !== current };
  }

  function announcement(label) {
    return `Vista activa: ${label}`;
  }

  const api = { viewFromHash, hashForView, planNavigation, planHashChange, announcement };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.KzaObsNav = api;
})(typeof window === 'undefined' ? globalThis : window);
