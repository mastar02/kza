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
    const rgb = /^rgba?\(([^)]+)\)$/.exec(value);
    const srgb = /^color\(srgb ([^)]+)\)$/.exec(value);
    const parts = (rgb || srgb)?.[1].split(/[, /]+/).map(Number);
    if (!parts || parts.length < 3 || !parts.every(Number.isFinite)) return null;
    const scale = rgb ? 1 : 255;
    return [...parts.slice(0, 3).map(channel => Math.min(255, Math.max(0, channel * scale))), parts[3] ?? 1];
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
  const textNodes = (root, isShown = shown) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const result = [];
    while (walker.nextNode()) {
      const node = walker.currentNode, el = node.parentElement;
      if (node.data.trim() && !el.closest('script,style,option,svg title,svg desc') && isShown(el)) result.push(node);
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
      // The component audit measures pressed-state signals such as an inset stripe.
      if (!el.hasAttribute('aria-pressed')) {
        const outer = nonuniformPaint(el) ? null : background(el.parentElement), inner = background(el);
        const border = color(style.borderTopColor);
        const paint = !['none', 'hidden'].includes(style.borderTopStyle) && parseFloat(style.borderTopWidth) && border && inner
          ? composite(border, inner) : inner;
        const strongest = outer && inner && paint && contrast(paint, outer) > contrast(inner, outer) ? paint : inner;
        checkContrast(findings, '1.4.11', el, strongest, outer, 3, 'control_boundary');
      }
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
  const resizeText = (root, zoom) => {
    const elements = [root, ...root.querySelectorAll('*')].filter(el => el instanceof HTMLElement && shown(el));
    const sizes = elements.map(el => {
      const style = getComputedStyle(el);
      return [el, el.getAttribute('style'), parseFloat(style.fontSize), style.lineHeight];
    });
    const restore = () => {
      for (const [el, original] of sizes) {
        if (original !== null) {
          el.setAttribute('style', original);
        } else if (el.getAttribute('style') !== null) {
          el.removeAttribute('style');
        }
      }
    };
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
    } catch (error) {
      restore();
      throw error;
    }
    return restore;
  };

  const overflow = el => el.scrollWidth - el.clientWidth;
  const horizontalOffenders = (root, left, right) => {
    const offenders = new Map();
    const record = (el, rect, clipFrom) => {
      let visibleLeft = rect.left, visibleRight = rect.right;
      if (visibleLeft >= left - 0.5 && visibleRight <= right + 0.5) return;
      for (let parent = clipFrom; parent && parent !== root; parent = parent.parentElement) {
        if (getComputedStyle(parent).overflowX === 'visible') continue;
        const clipLeft = parent.getBoundingClientRect().left + parent.clientLeft;
        visibleLeft = Math.max(visibleLeft, clipLeft);
        visibleRight = Math.min(visibleRight, clipLeft + parent.clientWidth);
      }
      if (visibleRight <= visibleLeft) return;
      const excess = Math.max(left - visibleLeft, visibleRight - right);
      if (excess <= 0.5 || (offenders.get(el)?.excess_px ?? 0) >= excess) return;
      offenders.set(el, {
        element: describe(el), left: rect.left, right: rect.right, excess_px: excess,
        client_width: el.clientWidth, scroll_width: el.scrollWidth,
      });
    };
    for (const el of root.querySelectorAll('*')) {
      if (shown(el) && !visuallyHidden(el)) record(el, el.getBoundingClientRect(), el.parentElement);
    }
    for (const node of textNodes(root)) {
      if (!visuallyHidden(node.parentElement)) {
        for (const rect of textRects(node)) record(node.parentElement, rect, node.parentElement);
      }
    }
    return [...offenders.values()].sort((a, b) => b.excess_px - a.excess_px).slice(0, 8);
  };
  const viewportOffenders = () => horizontalOffenders(
    document.body, -window.scrollX, document.documentElement.clientWidth - window.scrollX);

  const measureText = (root, criterion, keyControls) => {
    const nodes = textNodes(root).filter(node => !visuallyHidden(node.parentElement) && !clipping(node));
    const controls = keyControls.map(([selector, label]) =>
      [...root.querySelectorAll(selector)].find(el => !label || el.textContent.includes(label))).filter(el => el && shown(el));
    const usableControls = controls.filter(usable);
    const findings = [], documentRoot = document.documentElement;
    const beforeOverflow = overflow(documentRoot);
    const restore = resizeText(root, criterion === '1.4.4');
    try {
      const afterOverflow = overflow(documentRoot);
      if (afterOverflow > Math.max(1, beforeOverflow)) {
        const offenders = viewportOffenders();
        findings.push(finding(criterion, 'increased_horizontal_overflow', root, {
          element: offenders[0]?.element ?? 'html',
          overflow_px: afterOverflow, before_overflow_px: beforeOverflow, offenders,
        }));
      }
      const clipped = new Set(nodes.filter(clipping).map(node => node.parentElement));
      for (const el of clipped) findings.push(finding(criterion, 'text_newly_clipped', el));
      for (const el of usableControls) {
        if (!usable(el)) findings.push(finding(criterion, 'key_control_no_longer_usable', el));
      }
    } finally {
      restore();
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

  const modalOpening = (selector, initialSelector) => {
    const dialog = modal(selector), initial = document.activeElement;
    if (!dialog) return {found: false};
    return {
      found: true,
      width: document.documentElement.clientWidth,
      modal_semantics: dialog.matches(modalSelector),
      initial_ok: dialog.contains(initial) && (!initialSelector || initial.matches(initialSelector)),
    };
  };
  const returnFocus = (dialogSelector, openerSelector) => {
    if (modal(dialogSelector)) return 'dialog_still_open';
    const opener = document.querySelector(openerSelector);
    if (!opener || disabled(opener) || !shown(opener)) return 'return_target_unavailable';
    return document.activeElement === opener ? null : 'focus_not_returned';
  };

  const auditedControls = interactive + ',.pick,.btn,.sw';
  const boundaryControls = 'input:not([type=button],[type=submit],[type=reset]),select,textarea,'
    + '[role=checkbox],[role=radio],[role=switch],[role=combobox]';
  const stateAttributes = ['aria-pressed', 'aria-checked', 'aria-selected', 'aria-current'];
  const transparentPaint = [0, 0, 0, 0];
  const colorPattern = /rgba?\([^)]*\)/g;
  const alpha = (paint, opacity) => [...paint.slice(0, 3), paint[3] * opacity];
  const px = value => parseFloat(value) || 0;
  const auditable = el => el.getClientRects().length > 0
    && el.checkVisibility({visibilityProperty: true}) && !visuallyHidden(el);

  // CSS opacity is applied to the entire painted group, including descendant text.
  // Measure both ends and intermediate colors for the sRGB gradient on the map.
  const paints = style => {
    const fill = color(style.backgroundColor);
    if (!fill || style.mixBlendMode !== 'normal' || style.filter !== 'none') return [];
    if (style.backgroundImage === 'none') return [fill];
    if (!/^(?:linear|radial)-gradient\(/.test(style.backgroundImage) || /\), \w+-gradient/.test(style.backgroundImage)) return [];
    const stops = [...style.backgroundImage.matchAll(colorPattern)].map(([value]) => color(value));
    if (stops.length < 2 || stops.some(stop => !stop || stop[3] !== 1)) return [];
    return stops.slice(1).flatMap((end, index) => [0, .25, .5, .75, 1].map(t =>
      end.map((channel, i) => channel * t + stops[index][i] * (1 - t))));
  };
  const paintPairs = (el, foreground, backgrounds) => {
    let pairs = [[foreground, transparentPaint]];
    for (let node = el; node; node = node.parentElement) {
      // A fully opaque pair blocks the backdrop, unless this ancestor fades the group.
      const style = getComputedStyle(node);
      const layers = node === el && backgrounds ? backgrounds : paints(style);
      if (!layers.length && pairs.some(pair => pair.some(paint => paint[3] < 1))) return [];
      pairs = pairs.flatMap(pair => (layers.length ? layers : [transparentPaint]).map(layer =>
        pair.map(paint => alpha(composite(paint, layer), +style.opacity))));
    }
    const base = canvasColor || (canvasColor = canvas());
    return pairs.map(pair => pair.map(paint => composite(paint, base)));
  };
  const ratioFor = (el, foreground, backgrounds) => {
    if (!foreground) return {ratio: null};
    const pairs = paintPairs(el, foreground, backgrounds);
    if (!pairs.length) return {ratio: null};
    const [fg, bg] = pairs.reduce((lowest, pair) =>
      contrast(...pair) < contrast(...lowest) ? pair : lowest);
    const differences = pairs.map(([front, back]) => luminance(front) - luminance(back));
    // A continuous gradient contains an equal-luminance point if the sign changes.
    const crosses = differences.some(value => value < 0) && differences.some(value => value > 0);
    return {ratio: crosses ? 1 : contrast(fg, bg), foreground: fg, background: bg};
  };
  const borders = style => ['Top', 'Right', 'Bottom', 'Left'].flatMap(side => {
    const width = px(style['border' + side + 'Width']);
    const paint = color(style['border' + side + 'Color']);
    return width && paint?.[3] && !['none', 'hidden'].includes(style['border' + side + 'Style'])
      ? [{part: 'border-' + side.toLowerCase(), paint, width}] : [];
  });
  const shadows = style => [...style.boxShadow.matchAll(/(rgba?\([^)]*\))([^,]*)/g)].flatMap(([, value, tail]) => {
    const [x, y, blur, spread = 0] = tail.match(/-?[\d.]+px/g)?.map(px) || [];
    if (blur || (!x && !y && !spread)) return [];
    return [{part: tail.includes('inset') ? 'inset-shadow' : 'shadow', paint: color(value),
      width: Math.max(Math.abs(x), Math.abs(y), spread), ring: !x && !y && spread > 0}];
  });
  const surfaces = (el, pseudo) => {
    const style = getComputedStyle(el, pseudo);
    return [
      {part: 'fill', paint: color(style.backgroundColor), width: 0},
      ...borders(style), ...shadows(style),
    ];
  };
  const strongest = samples => samples.reduce((best, sample) =>
    (sample.ratio ?? -1) > (best.ratio ?? -1) ? sample : best, {ratio: null});
  const textSamples = el => {
    const samples = [];
    const add = (node, style, part) => {
      const size = px(style.fontSize), weight = Number(style.fontWeight);
      const minimum = size >= 24 || (size >= 56 / 3 && weight >= 700) ? 3 : 4.5;
      const fg = color(style.color);
      samples.push({part, minimum, ...ratioFor(node, fg && alpha(fg, part === 'placeholder' ? +style.opacity : 1))});
    };
    for (const node of new Set(textNodes(el, auditable).filter(node => !node.parentElement.closest('svg')
      && textRects(node).length).map(node => node.parentElement))) {
      add(node, getComputedStyle(node), 'text');
    }
    if (el.matches('select,textarea,input:not([type=radio],[type=checkbox],[type=range],[type=color],[type=file])')) {
      if (el.value || !el.hasAttribute('placeholder')) add(el, getComputedStyle(el), 'field-text');
      else add(el, getComputedStyle(el, '::placeholder'), 'placeholder');
    }
    return samples;
  };

  const focusTargets = el => [el, ...(el.matches('.room-pin') ? el.querySelectorAll('.tag') : [])];
  const focusRings = node => {
    const style = getComputedStyle(node), offset = px(style.outlineOffset);
    const rings = shadows(style).filter(item => item.ring).map(item =>
      ({...item, offset: item.part === 'inset-shadow' ? -item.width : 0}));
    if (!['none', 'hidden', 'auto'].includes(style.outlineStyle)) {
      rings.push({part: 'outline', paint: color(style.outlineColor), width: px(style.outlineWidth), offset});
    }
    return rings.filter(ring => ring.paint?.[3] && ring.width > 0).map(ring => {
      const {offset, width, paint} = ring;
      const outer = ratioFor(offset < -width ? node : node.parentElement, paint);
      const inner = ratioFor(offset < 0 ? node : node.parentElement, paint);
      const ratio = inner.ratio === null || outer.ratio === null ? null : Math.min(inner.ratio, outer.ratio);
      return {part: ring.part, offset, width, ratio, inner, outer};
    });
  };
  const focusSnapshot = el => focusTargets(el).map(focusRings);
  const changedRingWidth = (ring, previous) => {
    // Subtract any span that already had a ring of similar contrast. Treat
    // outlines and shadows alike when they paint the same part of the control.
    const end = ring.offset + ring.width;
    let cursor = ring.offset, changed = 0;
    for (const prior of previous.slice().sort((a, b) => a.offset - b.offset)) {
      const priorEnd = prior.offset + prior.width;
      if (priorEnd <= cursor || prior.offset >= end) continue;
      const colorChanged = ['inner', 'outer'].every(side => ring[side].foreground && prior[side].foreground
        && contrast(ring[side].foreground, prior[side].foreground) >= 3);
      if (colorChanged) continue;
      changed += Math.max(0, prior.offset - cursor);
      cursor = Math.min(end, priorEnd);
    }
    return changed + end - cursor;
  };
  const focusContrast = (el, before = []) => {
    const candidates = focusSnapshot(el).flatMap((rings, index) => rings.map(ring =>
      ({...ring, changedWidth: before[index] ? changedRingWidth(ring, before[index]) : 0})));
    return {...strongest(candidates.filter(ring => ring.changedWidth >= 2)), candidates};
  };
  const nativeBoundary = el => {
    // Chromium paints the indicator of these platform-native widgets. The field
    // surface, text and authored focus ring are still checked wherever present.
    if (getComputedStyle(el).appearance === 'none') return null;
    if (el.matches('input[type=checkbox],input[type=radio],input[type=range],input[type=color],input[type=file]')) {
      return 'Chromium paints this native widget and its state';
    }
    return null;
  };
  const stateMarker = el => {
    if (el.matches('.sw')) return [el, '::before'];
    if (el.matches('.room-pin')) return [el.querySelector('.tag'), null];
    return [el, null];
  };
  const statePaint = el => {
    const [marker, pseudo] = stateMarker(el);
    const style = getComputedStyle(marker, pseudo);
    const samples = surfaces(marker, pseudo).flatMap(({part, paint}) => {
      if (!paint?.[3]) return [];
      const backdrop = pseudo ? marker : part === 'fill' || part === 'shadow' ? marker.parentElement : marker;
      const measurement = ratioFor(backdrop, paint);
      return [{part, paint: measurement.foreground, adjacent: measurement.background}];
    });
    if (pseudo === '::before') {
      const thumb = paints(style)[0], track = color(style.backgroundColor);
      if (thumb && track && px(style.backgroundSize) > 0) {
        const measured = ratioFor(marker, thumb, [track]);
        const trackPaint = ratioFor(marker, track).foreground;
        const side = style.backgroundPositionX;
        samples.push({part: 'thumb-' + side, paint: measured.foreground, adjacent: measured.background});
        samples.push({part: 'track', paint: trackPaint, adjacent: measured.background});
      }
    }
    return samples;
  };
  const compareSelection = (el, attribute) => {
    const original = el.getAttribute(attribute), activeClass = el.classList.contains('active');
    const active = original !== 'false';
    const before = statePaint(el);
    let after;
    try {
      if (attribute === 'aria-current') el.removeAttribute(attribute);
      else el.setAttribute(attribute, String(!active));
      if (activeClass) el.classList.remove('active');
      after = statePaint(el);
    } finally {
      el.setAttribute(attribute, original);
      if (activeClass) el.classList.add('active');
    }
    const [selected, unselected] = active ? [before, after] : [after, before];
    return strongest(selected.flatMap(marker => {
      if (!marker.paint || !marker.adjacent) return [];
      const other = unselected.find(sample => sample.part === marker.part);
      const inactive = other?.paint
        || (marker.part.startsWith('thumb-') && unselected.find(sample => sample.part === 'track')?.paint)
        || marker.adjacent;
      return [{part: marker.part, ratio: Math.min(contrast(marker.paint, inactive), contrast(marker.paint, marker.adjacent)),
        selected: marker.paint, unselected: inactive, adjacent: marker.adjacent}];
    }));
  };
  const component = (el, state, before, range) => {
    const isDisabled = disabled(el);
    const selection = Object.fromEntries(stateAttributes.filter(attribute => el.hasAttribute(attribute))
      .map(attribute => [attribute, el.getAttribute(attribute)]));
    const measurements = [];
    const exemptions = {};
    const issues = [];
    const check = (sample, minimum, part) => {
      measurements.push({...sample, minimum, criterion_part: part, part: sample.part || part});
      if (sample.ratio === null || sample.ratio < minimum) {
        issues.push({part, ratio: sample.ratio, minimum, measured_part: sample.part});
      }
    };
    const text = textSamples(el);
    for (const sample of text) check(sample, isDisabled ? 1.01 : sample.minimum, sample.part);
    if (state === 'normal' && el.matches('.pick')) {
      const {top, bottom} = el.getBoundingClientRect();
      if (textNodes(el).flatMap(textRects).some(rect => rect.top < top - 1 || rect.bottom > bottom + 1)) {
        issues.push({part: 'selectable-row-text-overflows'});
      }
    }
    const native = nativeBoundary(el);
    const icon = el.querySelector('svg');
    if (icon && text.length) exemptions.svg = 'Decorative SVG alongside measured visible text';
    else if (icon) {
      // The shared sprite uses currentColor for its strokes and no fill.
      check(ratioFor(icon, color(getComputedStyle(icon).stroke)), isDisabled ? 1.01 : 3, 'icon');
    }
    if (isDisabled) {
      exemptions.disabled = 'Inactive control; text must remain visible and its styling must change';
      const appearance = styleSignature(el);
      if (!shown(el)) issues.push({part: 'disabled-invisible'});
      if (before?.signature === appearance && !native) issues.push({part: 'disabled-indistinguishable'});
    } else {
      if (native) exemptions.native = native;
      if (!native && el.matches(boundaryControls)) {
        const [marker, pseudo] = stateMarker(el);
        const outside = pseudo ? el : el.parentElement;
        const markerStyle = getComputedStyle(marker, pseudo);
        const fill = color(markerStyle.backgroundColor);
        const paintsToCheck = el.matches('input[type=range]')
          ? [{part: 'range-track', paint: color(range?.track), width: 0}]
          : surfaces(marker, pseudo);
        const samples = paintsToCheck.map(sample => {
          const paint = sample.paint && sample.part.startsWith('border') && fill && markerStyle.backgroundClip === 'border-box'
            ? composite(sample.paint, fill) : sample.paint;
          return {part: sample.part, ...ratioFor(outside, paint)};
        });
        const borderSamples = samples.filter(sample => sample.part.startsWith('border'));
        const candidates = samples.filter(sample => !sample.part.startsWith('border'));
        if (borderSamples.length) {
          const least = borderSamples.reduce((worst, sample) =>
            (sample.ratio ?? -1) < (worst.ratio ?? -1) ? sample : worst);
          candidates.push(least);
        }
        check({...strongest(candidates), candidates: samples}, 3, 'control-boundary');
        if (range) {
          const border = color(range.thumbBorder), thumb = color(range.thumb);
          // Chromium paints the thumb over the track and the surrounding field.
          const overTrack = color(range.track);
          const trackRatio = ratioFor(outside, border, overTrack && [overTrack]);
          const outsideRatio = ratioFor(outside, thumb);
          check(trackRatio, 3, 'range-thumb-against-track');
          check(outsideRatio, 3, 'range-thumb-against-surroundings');
        }
      }
      if (state === 'focus-visible') {
        const focus = focusContrast(el, before?.focus);
        check(focus, 3, 'focus-indicator');
        if (!(focus.width >= 2)) issues.push({part: 'focus-width', width: focus.width ?? 0, minimum: 2});
      }
      if (state === 'normal' && !native) {
        for (const [attr, value] of Object.entries(selection)) {
          if (value && (attr !== 'aria-current' || value !== 'false')) {
            check(compareSelection(el, attr), 3, attr + '-indicator');
          }
        }
      }
    }
    return {selector: describe(el), control: controlName(el), state, selection, disabled: isDisabled,
      measurements, exemptions, issues};
  };
  const styleSignature = el => {
    const style = getComputedStyle(el);
    return [style.color, style.backgroundColor, style.opacity, style.borderTopColor, style.backgroundImage].join('|');
  };
  const enabledSignature = el => {
    const attributes = ['disabled', 'aria-disabled'].filter(name => el.hasAttribute(name))
      .map(name => [name, el.getAttribute(name)]);
    try {
      for (const [name] of attributes) el.removeAttribute(name);
      return disabled(el) ? undefined : styleSignature(el);
    } finally {
      for (const [name, value] of attributes) el.setAttribute(name, value);
    }
  };
  const controlName = el => el.localName + (el.id ? '#' + el.id : '') + [...el.classList].map(cls => '.' + cls).join('');
  const contrastControl = el => ({
    selector: describe(el), control: controlName(el), disabled: disabled(el), tabbable: el.tabIndex >= 0,
    range: el.matches('input[type=range]') && !nativeBoundary(el),
  });
  // A sibling can move into the old selector. Compare identifying content in the page.
  const contrastIdentity = el => [controlName(el), el.textContent.trim(),
    ...['role', 'type', 'name', 'href', 'aria-label', 'aria-labelledby', 'data-key'].map(name => el.getAttribute(name))];
  let contrastProbe, contrastInitial;
  const prepareContrast = selector => {
    canvasColor = canvas();
    const root = selector ? document.querySelector(selector) : scope();
    if (!root || !auditable(root)) return null;
    contrastInitial = document.activeElement;
    contrastInitial?.blur();
    contrastProbe = [...root.querySelectorAll(auditedControls)].filter(auditable);
    return contrastProbe.map(contrastControl);
  };
  const rebindContrast = (index, selector) => {
    const previous = contrastProbe[index];
    if (previous.isConnected) return auditable(previous) ? contrastControl(previous) : null;
    const current = document.querySelector(selector);
    if (!current?.matches(auditedControls) || !auditable(current)) return null;
    const identity = contrastIdentity(previous);
    if (!contrastIdentity(current).every((value, i) => value === identity[i])) return null;
    contrastProbe[index] = current;
    return contrastControl(current);
  };
  const endContrast = () => {
    if (contrastInitial?.isConnected && contrastInitial !== document.body) contrastInitial.focus({preventScroll: true});
    contrastProbe = contrastInitial = undefined;
  };
  const measureContrast = (index, state, range, selector) => {
    const el = contrastProbe[index];
    if (!el.isConnected || !el.matches(selector)) return {issues: [{part: 'control-replaced'}]};
    if (state === 'normal') {
      const inactive = disabled(el);
      const baseline = inactive ? {signature: enabledSignature(el)} : undefined;
      return {measurement: component(el, inactive ? 'disabled' : state, baseline, range),
        before: {focus: focusSnapshot(el), signature: styleSignature(el)}};
    }
    if (state === 'disabled') {
      if (!('disabled' in el)) return null;
      const before = {signature: styleSignature(el)};
      const attribute = el.getAttribute('disabled');
      try {
        el.disabled = true;
        return component(el, state, before);
      } finally {
        if (attribute === null) el.removeAttribute('disabled');
        else el.setAttribute('disabled', attribute);
      }
    }
    return component(el, state.name, state.before, range);
  };

  let restoreTextZoom;
  window.__kzaWcag = {
    contrast, prepareContrast, rebindContrast, measureContrast, endContrast, focusSnapshot, focusContrast, controlName,
    statuses,
    modalOpening, returnFocus,
    beginTextZoom() {
      const root = scope(), documentRoot = document.documentElement;
      const dialog = modal();
      const before = {document: overflow(documentRoot), dialog: dialog ? overflow(dialog) : null};
      restoreTextZoom = resizeText(root, true);
      const documentOverflow = overflow(documentRoot), dialogOverflow = dialog ? overflow(dialog) : null;
      const dialogLeft = dialog ? dialog.getBoundingClientRect().left + dialog.clientLeft : 0;
      return {
        before, document: documentOverflow, dialog: dialogOverflow,
        offenders: documentOverflow > 0 ? viewportOffenders() : [],
        dialog_offenders: dialogOverflow > 0
          ? horizontalOffenders(dialog, dialogLeft - dialog.scrollLeft, dialogLeft + dialog.clientWidth - dialog.scrollLeft)
          : [],
      };
    },
    endTextZoom() {
      restoreTextZoom?.();
      restoreTextZoom = undefined;
    },
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
