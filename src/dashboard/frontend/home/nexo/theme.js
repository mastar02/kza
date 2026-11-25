(() => {
  'use strict';

  const key = 'nexo-theme';
  const options = new Map([
    ['auto', 'Automático'],
    ['claro', 'Claro'],
    ['oscuro', 'Oscuro'],
  ]);
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  let preference = 'auto';
  try {
    const saved = window.localStorage.getItem(key);
    if (options.has(saved)) preference = saved;
  } catch {
    // Storage may be blocked. The preference still works for this page.
  }

  function applyTheme() {
    const night = preference === 'oscuro' || (preference === 'auto' && systemTheme.matches);
    document.documentElement.dataset.theme = night ? 'oscuro' : 'claro';
    document.body?.classList.toggle('night', night);
  }

  function mountSelector() {
    applyTheme();
    const slot = document.getElementById('theme-slot');
    if (!slot) return;

    const select = document.createElement('select');
    select.className = 'btn';
    select.setAttribute('aria-label', 'Tema');
    for (const [value, label] of options) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.append(option);
    }
    select.value = preference;
    select.addEventListener('change', () => {
      if (!options.has(select.value)) return;
      preference = select.value;
      applyTheme();
      try {
        window.localStorage.setItem(key, preference);
      } catch {
        // Keep the in-page selection even if it cannot be saved.
      }
    });
    slot.append(select);
  }

  applyTheme();
  systemTheme.addEventListener('change', () => {
    if (preference === 'auto') applyTheme();
  });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountSelector, { once: true });
  } else {
    mountSelector();
  }
})();
