/* A small DOM for Node view tests (U3b-4a-3): enough of Element and Document for the NEXO views. */
class ClassList {
  constructor(element) { this.element = element; }
  get values() { return this.element.className.split(/\s+/).filter(Boolean); }
  contains(name) { return this.values.includes(name); }
  add(name) { if (!this.contains(name)) this.element.className = [...this.values, name].join(' '); }
  remove(name) { this.element.className = this.values.filter(value => value !== name).join(' '); }
  toggle(name, force) { const on = force === undefined ? !this.contains(name) : Boolean(force); if (on) this.add(name); else this.remove(name); return on; }
}
class Element {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase(); this.ownerDocument = doc; this.children = []; this.parentNode = null; this.dataset = {}; this.attributes = {};
    this.listeners = {}; this.className = ''; this._text = ''; this.hidden = false; this.id = ''; this.value = ''; this.open = false; this.disabled = false;
    this.classList = new ClassList(this); this.selected = false;
  }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.children.forEach(child => { child.parentNode = null; }); this.children = []; this._text = String(value); }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() { const siblings = this.parentNode?.children || []; return siblings[siblings.indexOf(this) + 1] || null; }
  get childElementCount() { return this.children.length; }
  append(...nodes) { for (const node of nodes) { if (typeof node === 'string') { this._text += node; continue; } node.parentNode?.removeChild(node); node.parentNode = this; this.children.push(node); } }
  replaceChildren(...nodes) { this.children.forEach(child => { child.parentNode = null; }); this.children = []; this._text = ''; this.append(...nodes); }
  insertBefore(node, reference) { node.parentNode?.removeChild(node); node.parentNode = this; const index = reference ? this.children.indexOf(reference) : -1; if (index < 0) this.children.push(node); else this.children.splice(index, 0, node); }
  removeChild(node) { this.children = this.children.filter(child => child !== node); node.parentNode = null; }
  remove() { this.parentNode?.removeChild(this); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return Object.hasOwn(this.attributes, name) ? this.attributes[name] : null; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  removeEventListener(type, callback) { this.listeners[type] = (this.listeners[type] || []).filter(item => item !== callback); }
  dispatch(type, event = {}) { for (const callback of [...(this.listeners[type] || [])]) callback({ preventDefault() {}, ...event }); }
  click() { if (this.onclick) this.onclick(); this.dispatch('click'); }
  focus() { this.ownerDocument.activeElement = this; }
  contains(node) { for (let current = node; current; current = current.parentNode) if (current === this) return true; return false; }
  descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
  matches(selector) {
    if (selector === 'button:not([disabled])') return this.tagName === 'BUTTON' && !this.disabled;
    const attribute = /^\[data-([a-z-]+)(?:="([^"]*)")?\]$/.exec(selector);
    if (attribute) { const key = attribute[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase()); return Object.hasOwn(this.dataset, key) && (attribute[2] === undefined || this.dataset[key] === attribute[2]); }
    if (selector.startsWith('#')) return this.id === selector.slice(1);
    if (selector.startsWith('.')) return this.classList.contains(selector.slice(1));
    return this.tagName === selector.toUpperCase();
  }
  querySelectorAll(selector) { return this.descendants().filter(element => element.matches(selector)); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  showModal() { this.open = true; }
  close() { this.open = false; }
}
function createDocument() {
  const doc = { activeElement: null, createElement(tag) { return new Element(tag, doc); }, addEventListener() {}, removeEventListener() {} };
  return doc;
}
const icon = doc => () => doc.createElement('svg');
/** Visible text of an element and its non-hidden descendants. */
function visibleText(element) { return element.hidden ? '' : element._text + element.children.map(visibleText).join(''); }
function buttons(element) { return element.querySelectorAll('button').filter(button => !hiddenWithin(button, element)); }
function hiddenWithin(node, root) { for (let current = node; current && current !== root.parentNode; current = current.parentNode) if (current.hidden) return true; return false; }
module.exports = { Element, createDocument, icon, visibleText, buttons };
