/* Navigation groups. View permissions and URL resolution live in state.js. */
(function (root) {
  'use strict';
  const SECTIONS = [
    { id: 'home', label: 'Casa', defaultView: 'home',
      views: { home: 'Vista de la casa', actions: 'Acciones' } },
    { id: 'routines', label: 'Rutinas', defaultView: 'routines',
      views: { routines: 'Rutinas' } },
    { id: 'tasks', label: 'Tareas', defaultView: 'tasks',
      views: { tasks: 'Tareas' } },
    { id: 'activity', label: 'Actividad', defaultView: 'activity', views: { activity: 'Actividad' } },
    { id: 'people', label: 'Personas', defaultView: 'people',
      views: { people: 'Lista de personas', voices: 'Voces' } },
    { id: 'memory', label: 'Memoria', defaultView: 'memory',
      views: { memory: 'Recuerdos del hogar', mymemory: 'Mis recuerdos' } },
    { id: 'system', label: 'Sistema', defaultView: 'access',
      views: { access: 'Accesos', evaluation: 'Evaluación' } }
  ];

  function visibleSections(allowedViews) {
    return SECTIONS.map(({ views, ...section }) => ({
      ...section,
      views: Object.entries(views).filter(([id]) => allowedViews.includes(id))
        .map(([id, label]) => ({ id, label }))
    })).filter(section => section.views.length > 0);
  }

  function sectionDestination(section, lastView) {
    return [lastView, section.defaultView].find(id => section.views.some(view => view.id === id))
      ?? section.views[0]?.id;
  }

  const exported = { SECTIONS, visibleSections, sectionDestination };
  root.NexoSections = exported;
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof window !== 'undefined' ? window : globalThis);
