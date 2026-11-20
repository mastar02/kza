// Deterministic DOM probes used by browser_support.py. Never serialize text or form values.
(() => {
  if (window.__kzaWcag) return true;

  const interactive = 'a[href],button,input:not([type=hidden]),select,textarea,summary,audio[controls],video[controls],'
    + '[role=button],[role=link],[role=checkbox],[role=radio],[role=switch],[role=tab],[role=menuitem],'
    + '[role=menuitemcheckbox],[role=menuitemradio],[role=slider],[role=spinbutton],[role=combobox]';
  const focusable = interactive + ',[tabindex],[contenteditable=true],iframe';
  const describe = el => {
    if (!el || el === document.body) return null;
    const path = [];
    for (let node = el; node && node !== document.body; node = node.parentElement) {
      const siblings = [...(node.parentElement?.children || [])].filter(sibling => sibling.localName === node.localName);
      path.unshift(`${node.localName}:nth-of-type(${siblings.indexOf(node) + 1})`);
    }
    return 'body > ' + path.join(' > ');
  };
  const shown = el => el.getClientRects().length > 0
    && el.checkVisibility({visibilityProperty: true, opacityProperty: true});
  const disabled = el => el.matches(':disabled') || !!el.closest('[inert],[aria-disabled=true]');
  const modalSelector = 'dialog:modal,[role=dialog][aria-modal=true],[role=alertdialog][aria-modal=true]';
  const modal = selector => [...document.querySelectorAll(selector || modalSelector)].filter(shown).at(-1);
  const scope = () => modal() || document.body;
  const finding = (criterion, issue, el, detail = {}) => ({criterion, issue, element: describe(el), ...detail});

  // WCAG relative luminance in sRGB, including CSS source-over for translucent paint.
  const color = value => {
    const parts = /^rgba?\(([^)]+)\)$/.exec(value)?.[1].split(/[, /]+/).map(Number);
    return parts?.length >= 3 && parts.every(Number.isFinite)
      ? [parts[0], parts[1], parts[2], parts[3] ?? 1] : null;
  };
  const composite = (front, back) => {
    const alpha = front[3] + back[3] * (1 - front[3]);
    return alpha ? [...front.slice(0, 3).map((channel, i) =>
      (channel * front[3] + back[i] * back[3] * (1 - front[3])) / alpha), alpha] : [0, 0, 0, 0];
  };
  const luminance = rgb => rgb.slice(0, 3).map(channel => {
    const srgb = channel / 255;
    return srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4;
  }).reduce((sum, channel, index) => sum + channel * [0.2126, 0.7152, 0.0722][index], 0);
  const contrast = (a, b) => {
    const levels = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (levels[0] + 0.05) / (levels[1] + 0.05);
  };
  const canvas = () => {
    const probe = document.createElement('span');
    probe.style.cssText = 'display:none;color:Canvas';
    document.documentElement.append(probe);
    const result = color(getComputedStyle(probe).color);
    probe.remove();
    return result;
  };
  let canvasColor;
  const background = el => {
    let result = [0, 0, 0, 0];
    for (let node = el; node && result[3] < 1; node = node.parentElement) {
      const style = getComputedStyle(node), layer = color(style.backgroundColor);
      if (!layer || style.backgroundImage !== 'none' || style.backdropFilter !== 'none') return null;
      result = composite(result, layer);
    }
    return composite(result, canvasColor);
  };
  const nonuniformPaint = el => {
    for (let node = el; node; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (+style.opacity !== 1 || style.mixBlendMode !== 'normal' || style.filter !== 'none') return true;
    }
    return false;
  };
  const checkContrast = (findings, criterion, el, fg, bg, minimum, part) => {
    if (!fg || !bg) {
      findings.push(finding(criterion, 'paint_needs_review', el, {part, level: 'review'}));
    } else {
      const ratio = contrast(composite(fg, bg), bg);
      if (ratio < minimum) findings.push(finding(criterion, 'low_contrast', el, {part, ratio, minimum}));
    }
  };
  const textNodes = root => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const result = [];
    while (walker.nextNode()) {
      const node = walker.currentNode, el = node.parentElement;
      if (node.data.trim() && !el.closest('script,style,option,svg title,svg desc') && shown(el)) result.push(node);
    }
    return result;
  };
  const textRects = node => {
    const range = document.createRange();
    range.selectNodeContents(node);
    return [...range.getClientRects()].filter(rect => rect.width && rect.height);
  };
  const visuallyHidden = el => !!el.closest('.sr-only,.visually-hidden,[hidden]')
    || (el.clientWidth <= 1 && el.clientHeight <= 1 && getComputedStyle(el).position === 'absolute');
  const textContrast = root => {
    const findings = [];
    const elements = new Set(textNodes(root).filter(node => textRects(node).length).map(node => node.parentElement));
    for (const el of root.querySelectorAll('input:not([type]),input[type=text],input[type=search],input[type=password],input[type=number],textarea,select')) {
      if (shown(el)) elements.add(el);
    }
    for (const el of elements) {
      if (disabled(el) || visuallyHidden(el)) continue;
      const style = getComputedStyle(el), size = parseFloat(style.fontSize);
      const minimum = size >= 24 || (size >= 56 / 3 && +style.fontWeight >= 700) ? 3 : 4.5;
      const bg = nonuniformPaint(el) ? null : background(el);
      checkContrast(findings, '1.4.3', el, color(style.color), bg, minimum, 'text');
      if (!el.value && el.hasAttribute('placeholder')) {
        const placeholder = getComputedStyle(el, '::placeholder');
        const fg = color(placeholder.color);
        if (fg) fg[3] *= +placeholder.opacity;
        checkContrast(findings, '1.4.3', el, fg, bg, minimum, 'placeholder');
      }
    }
    return findings;
  };

  const targets = root => [...root.querySelectorAll(interactive)].filter(el =>
    shown(el) && !disabled(el) && !visuallyHidden(el) && getComputedStyle(el).pointerEvents !== 'none');
  const hasVisibleLabel = control => textNodes(control).some(node => {
    const el = node.parentElement;
    return !el.closest('svg') && !visuallyHidden(el) && textRects(node).length > 0;
  });
  const componentContrast = root => {
    const findings = [];
    for (const el of targets(root)) {
      // Text-labeled action buttons may be identifiable without their boundaries.
      if (!el.matches('input,select,textarea,[aria-pressed],[aria-checked],[role=switch],[role=slider]')) continue;
      const style = getComputedStyle(el);
      if (style.appearance !== 'none' && el.matches('input[type=checkbox],input[type=radio],input[type=range],select')) {
        findings.push(finding('1.4.11', 'native_control_needs_review', el, {level: 'review'}));
        continue;
      }
      const outer = nonuniformPaint(el) ? null : background(el.parentElement), inner = background(el);
      const border = color(style.borderTopColor);
      const paint = !['none', 'hidden'].includes(style.borderTopStyle) && parseFloat(style.borderTopWidth) && border && inner
        ? composite(border, inner) : inner;
      const strongest = outer && inner && paint && contrast(paint, outer) > contrast(inner, outer) ? paint : inner;
      checkContrast(findings, '1.4.11', el, strongest, outer, 3, 'control_boundary');
      if (el.hasAttribute('aria-pressed') || el.hasAttribute('aria-checked')) {
        findings.push(finding('1.4.11', 'state_indicator_needs_review', el, {level: 'review'}));
      }
    }
    for (const svg of root.querySelectorAll('svg')) {
      const control = svg.closest(interactive);
      if (!shown(svg) || disabled(svg) || (control ? hasVisibleLabel(control) : !svg.matches('[role=img]'))) continue;
      if (svg.querySelector('use')) {
        findings.push(finding('1.4.11', 'svg_reference_needs_review', svg, {level: 'review'}));
      }
      for (const shape of svg.querySelectorAll('path,line,rect,circle,ellipse,polygon,polyline')) {
        const style = getComputedStyle(shape);
        const bg = nonuniformPaint(shape) ? null : background(svg);
        for (const part of ['fill', 'stroke']) {
          if (style[part] === 'none' || (part === 'stroke' && !parseFloat(style.strokeWidth))) continue;
          const fg = color(style[part]);
          if (fg) fg[3] *= +style[part + 'Opacity'];
          if (fg?.[3] === 0) continue;
          checkContrast(findings, '1.4.11', svg, fg, bg, 3, part);
        }
      }
    }
    return findings;
  };

  const targetSizes = (root, minimum, productGoal) => {
    const boxes = targets(root).flatMap(el => {
      const labels = [...(el.labels || [])].filter(shown);
      const rects = [...el.getClientRects(), ...labels.flatMap(label => [...label.getClientRects()])]
        .filter(rect => rect.width > 0 && rect.height > 0);
      if (!rects.length) return [];
      const size = rects.reduce((best, rect) =>
        Math.min(rect.width, rect.height) > Math.min(best.width, best.height) ? rect : best);
      // A fragment determines the available size. The spacing circle is centered
      // on the smallest box enclosing the entire pointer target.
      const x = (Math.min(...rects.map(rect => rect.left)) + Math.max(...rects.map(rect => rect.right))) / 2;
      const y = (Math.min(...rects.map(rect => rect.top)) + Math.max(...rects.map(rect => rect.bottom))) / 2;
      return [{el, rects, size, x, y, small: size.width < minimum || size.height < minimum}];
    });
    const findings = [];
    for (const {el, size, small, x, y} of boxes) {
      const {width, height} = size;
      if (width < productGoal || height < productGoal) {
        findings.push(finding('product', 'target_below_goal', el, {width, height, minimum: productGoal}));
      }
      if (!small) continue;
      if (el.matches('a[href]') && getComputedStyle(el).display === 'inline'
          && [...el.parentElement.childNodes].some(node => node.nodeType === Node.TEXT_NODE && node.data.trim())) continue;
      const neighbor = boxes.find(other => {
        if (other.el === el) return false;
        if (other.small && Math.hypot(x - other.x, y - other.y) < minimum) return true;
        return other.rects.some(rect =>
          Math.hypot(Math.max(rect.left - x, 0, x - rect.right), Math.max(rect.top - y, 0, y - rect.bottom)) < minimum / 2);
      });
      if (neighbor) findings.push(finding('2.5.8', 'target_below_minimum_without_spacing', el,
        {width, height, minimum, neighbor: describe(neighbor.el), level: 'review'}));
    }
    return findings;
  };

  const clipping = node => {
    const el = node.parentElement;
    if (!el?.isConnected || !shown(el)) return true;
    const rects = textRects(node);
    if (!rects.length) return true;
    for (let parent = el; parent && parent !== document.body; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      const x = ['hidden', 'clip'].includes(style.overflowX);
      const y = ['hidden', 'clip'].includes(style.overflowY);
      if (!x && !y) continue;
      const rect = parent.getBoundingClientRect(), left = rect.left + parent.clientLeft, top = rect.top + parent.clientTop;
      if (rects.some(text => (x && (text.left < left - 1 || text.right > left + parent.clientWidth + 1))
                         || (y && (text.top < top - 1 || text.bottom > top + parent.clientHeight + 1)))) return true;
    }
    return false;
  };
  const usable = el => {
    if (!shown(el)) return false;
    const root = document.documentElement, body = document.body;
    const rootStyle = getComputedStyle(root), bodyStyle = body && getComputedStyle(body);
    // CSS Overflow §3.3 applies the body overflow to the viewport when both
    // root axes are visible. The element supplying it does not also clip locally.
    const bodyPropagates = root === body?.parentElement && body.localName === 'body'
      && rootStyle.overflowX === 'visible' && rootStyle.overflowY === 'visible'
      && rootStyle.contain === 'none' && bodyStyle.contain === 'none' && bodyStyle.display !== 'none';
    const scrollPosition = node => node ? [node.scrollLeft, node.scrollTop] : [window.scrollX, window.scrollY];
    const snapshotScroll = (node, {overflowX, overflowY}) => {
      const [x, y] = scrollPosition(node);
      return {node, overflowX, overflowY, x, y};
    };
    const ancestors = [];
    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
      if (parent === root || (parent === body && bodyPropagates)) continue;
      const style = getComputedStyle(parent);
      if (style.overflowX !== 'visible' || style.overflowY !== 'visible') {
        ancestors.push(snapshotScroll(parent, style));
      }
    }
    const viewport = snapshotScroll(null, bodyPropagates ? bodyStyle : rootStyle);
    const restoreScroll = blockedOnly => {
      for (const {node, overflowX, overflowY, x, y} of [...ancestors, viewport]) {
        const [currentX, currentY] = scrollPosition(node);
        const left = !blockedOnly || ['hidden', 'clip'].includes(overflowX) ? x : currentX;
        const top = !blockedOnly || ['hidden', 'clip'].includes(overflowY) ? y : currentY;
        if (currentX !== left || currentY !== top) (node || window).scrollTo({left, top, behavior: 'instant'});
      }
    };
    const hitVisiblePart = () => {
      const rect = el.getBoundingClientRect();
      let left = Math.max(0, rect.left), right = Math.min(root.clientWidth, rect.right);
      let top = Math.max(0, rect.top), bottom = Math.min(innerHeight, rect.bottom);
      for (const {node: parent, overflowX, overflowY} of ancestors) {
        const box = parent.getBoundingClientRect();
        const x = box.left + parent.clientLeft, y = box.top + parent.clientTop;
        if (overflowX !== 'visible') {
          left = Math.max(left, x);
          right = Math.min(right, x + parent.clientWidth);
        }
        if (overflowY !== 'visible') {
          top = Math.max(top, y);
          bottom = Math.min(bottom, y + parent.clientHeight);
        }
      }
      if (left >= right || top >= bottom) return false;
      const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2);
      return el.contains(hit) || [...(el.labels || [])].some(label => label.contains(hit));
    };
    if (hitVisiblePart()) return true;
    try {
      el.scrollIntoView({block: 'center', inline: 'nearest', behavior: 'instant'});
      // The viewport treats clip like hidden: the browser can scroll it by
      // script. Undo movement on blocked axes before testing the visible area.
      restoreScroll(true);
      return hitVisiblePart();
    } finally {
      restoreScroll(false);
    }
  };
  const measureText = (root, criterion, keyControls) => {
    const nodes = textNodes(root).filter(node => !visuallyHidden(node.parentElement) && !clipping(node));
    const controls = keyControls.map(([selector, label]) =>
      [...root.querySelectorAll(selector)].find(el => !label || el.textContent.includes(label))).filter(el => el && shown(el));
    const usableControls = controls.filter(usable);
    const elements = [root, ...root.querySelectorAll('*')].filter(el => el instanceof HTMLElement && shown(el));
    const sizes = elements.map(el => {
      const style = getComputedStyle(el);
      return [el, el.getAttribute('style'), parseFloat(style.fontSize), style.lineHeight];
    });
    const zoom = criterion === '1.4.4', findings = [];
    const overflow = () => document.documentElement.scrollWidth - document.documentElement.clientWidth;
    const beforeOverflow = overflow();
    try {
      for (const [el, , fontSize, lineHeight] of sizes) {
        const set = (property, value) => el.style.setProperty(property, value, 'important');
        if (zoom) {
          set('font-size', (2 * fontSize) + 'px');
          if (lineHeight.endsWith('px')) set('line-height', (2 * parseFloat(lineHeight)) + 'px');
        } else {
          set('line-height', '1.5');
          set('letter-spacing', '.12em');
          set('word-spacing', '.16em');
          if (el.matches('p,[role=paragraph]')) set('margin-block-end', '2em');
        }
      }
      if (overflow() > Math.max(1, beforeOverflow)) {
        findings.push(finding(criterion, 'increased_horizontal_overflow', root, {overflow_px: overflow()}));
      }
      const clipped = new Set(nodes.filter(clipping).map(node => node.parentElement));
      for (const el of clipped) findings.push(finding(criterion, 'text_newly_clipped', el));
      for (const el of usableControls) {
        if (!usable(el)) findings.push(finding(criterion, 'key_control_no_longer_usable', el));
      }
    } finally {
      for (const [el, original] of sizes) {
        if (original === null) el.removeAttribute('style');
        else el.setAttribute('style', original);
      }
    }
    return findings;
  };

  const statuses = specs => specs.flatMap(([state, selector]) => {
    const detail = {state, selector};
    const message = [...document.querySelectorAll(selector)].find(el => shown(el) && el.textContent.trim());
    if (!message) return [finding('4.1.3', 'status_message_missing', null, detail)];
    if (message.closest('[aria-hidden=true],[inert]')) return [finding('4.1.3', 'status_hidden_from_at', message, detail)];
    for (let el = message; el; el = el.parentElement) {
      const live = el.getAttribute('aria-live');
      const role = el.getAttribute('role')?.split(/\s+/)[0];
      if (live === 'off') break;
      if (['polite', 'assertive'].includes(live) || ['status', 'alert', 'log'].includes(role)) return [];
    }
    return [finding('4.1.3', 'live_semantics_missing', message, detail)];
  });

  let focusProbe;
  const beginModal = (selector, initialSelector) => {
    const dialog = modal(selector);
    const width = document.documentElement.clientWidth;
    if (!dialog) return {found: false, width};
    const initial = document.activeElement;
    const candidates = [...dialog.querySelectorAll(focusable)]
      .filter(el => shown(el) && !el.matches(':disabled') && !el.closest('[inert]') && el.tabIndex >= 0);
    const ordered = candidates.filter(el => el.type !== 'radio' || !el.name
      || (candidates.find(other => other.type === 'radio' && other.name === el.name && other.form === el.form && other.checked)
        || candidates.find(other => other.type === 'radio' && other.name === el.name && other.form === el.form)) === el)
      .sort((a, b) => (a.tabIndex || Infinity) - (b.tabIndex || Infinity));
    focusProbe = {dialog, initial, ordered};
    return {
      found: true,
      width,
      modal_semantics: dialog.matches(modalSelector),
      initial_ok: !!initial && (dialog.contains(initial) && (!initialSelector || initial.matches(initialSelector))),
      tabbables: ordered.length,
    };
  };
  const focusModalStart = backward => {
    const {ordered} = focusProbe;
    const start = backward ? ordered.at(-1) : ordered[0];
    start.focus({preventScroll: true});
    return document.activeElement === start;
  };
  const checkModalStep = (backward, step) => {
    const {dialog, ordered} = focusProbe;
    if (!shown(dialog)) return 'dialog_closed';
    if (!dialog.contains(document.activeElement)) return 'focus_escaped';
    const offset = step % ordered.length;
    const expected = backward ? ordered.at(-1 - offset) : ordered[offset];
    if (document.activeElement === expected) return null;
    return offset === 0 ? 'focus_did_not_wrap' : 'focus_out_of_order';
  };
  const endModal = () => {
    const {initial} = focusProbe;
    focusProbe = undefined;
    if (initial === document.body) document.activeElement?.blur();
    else if (initial?.isConnected) initial.focus({preventScroll: true});
  };
  const returnFocus = (dialogSelector, openerSelector) => {
    if (modal(dialogSelector)) return 'dialog_still_open';
    const opener = document.querySelector(openerSelector);
    if (!opener || disabled(opener) || !shown(opener)) return 'return_target_unavailable';
    return document.activeElement === opener ? null : 'focus_not_returned';
  };

  window.__kzaWcag = {
    contrast,
    statuses,
    beginModal, focusModalStart, checkModalStep, endModal, returnFocus,
    scan(keyControls, minimum, productGoal, statusMessages) {
      canvasColor = canvas() || [255, 255, 255, 1];
      const root = scope();
      return [
        ...textContrast(root), ...componentContrast(root), ...targetSizes(root, minimum, productGoal),
        ...measureText(root, '1.4.12', keyControls), ...measureText(root, '1.4.4', keyControls),
        ...statuses(statusMessages),
      ];
    },
  };
  return true;
})()
