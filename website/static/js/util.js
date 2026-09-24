// Small shared helpers: DOM building, formatting, data loading, class palette.

/** h('div.card#x', {attrs}, children...) - tiny hyperscript. Text children are escaped. */
export function h(tag, attrs = {}, ...children) {
  const [, name = 'div', rest = ''] = tag.match(/^([a-z0-9-]*)(.*)$/i) || [];
  const node = document.createElement(name || 'div');
  for (const part of rest.match(/[.#][^.#]+/g) || []) {
    if (part[0] === '.') node.classList.add(part.slice(1));
    else node.id = part.slice(1);
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') {
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith('--')) node.style.setProperty(prop, val); else node.style[prop] = val;
      }
    } else if (k === 'href' || k === 'src') node.setAttribute(k, safeUrl(v));
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** Only allow http(s), mailto, blob and relative URLs from data files. */
export function safeUrl(u) {
  const str = String(u ?? '').trim();
  if (/^(https?:|mailto:|blob:|#|\.{0,2}\/|[\w-]+(\/|\.|$))/i.test(str) && !/^\s*javascript:/i.test(str)) return str;
  return '#';
}

export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

/** Replace a node's children, skipping null/false entries. */
export function fill(node, ...kids) {
  clear(node).append(...kids.flat().filter((k) => k !== null && k !== undefined && k !== false));
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);

/** Fetch JSON; resolves to `fallback` (and logs) instead of throwing, so one missing file never blanks the page. */
export async function loadJSON(url, fallback = null) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn(`[sentinel] could not load ${url}:`, err.message);
    return fallback;
  }
}

export const arr = (x) => (Array.isArray(x) ? x : []);
export const obj = (x) => (x && typeof x === 'object' && !Array.isArray(x) ? x : {});
export const num = (x, d = 0) => (Number.isFinite(+x) ? +x : d);

/** 83.4 -> "1:23.4" */
export function fmtTime(sec, decimals = 1) {
  const s = Math.max(0, num(sec));
  const m = Math.floor(s / 60);
  const rest = (s - m * 60).toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0');
  return `${m}:${rest}`;
}

export const fmtNum = (x, d = 2) => (Number.isFinite(+x) ? (+x).toFixed(d) : '—');
export const pretty = (label) => String(label || '').replace(/_/g, ' ');

// ---- class palette ---------------------------------------------------------
const FALLBACK_COLORS = ['#ff3864', '#ff8fb1', '#b46bff', '#e04cd9', '#ffab7a', '#3b9dff', '#2dd4bf',
  '#a3e635', '#e6c86e', '#f2f27a', '#8e9bff', '#9aa7b8', '#d6c3a5', '#ff5a1f'];
const palette = new Map();
const objectPalette = new Map();
let classInfo = [];

export function setPalette(classesJson) {
  classInfo = arr(obj(classesJson).classes);
  classInfo.forEach((c, i) => palette.set(c.id, c.color || FALLBACK_COLORS[i % FALLBACK_COLORS.length]));
  for (const [k, v] of Object.entries(obj(obj(classesJson).objects))) objectPalette.set(k, v);
}

export function classColor(label) {
  if (!palette.has(label)) {
    palette.set(label, FALLBACK_COLORS[palette.size % FALLBACK_COLORS.length]);
  }
  return palette.get(label);
}

export const objectColor = (name) => objectPalette.get(name) || '#c9d1d9';
export const classList = () => classInfo.map((c) => c.id);
export const classRule = (id) => (classInfo.find((c) => c.id === id) || {}).rule || '';

/** Order labels the canonical way (14-class order), unknown labels last. */
export function sortLabels(labels) {
  const order = classList();
  const idx = (l) => { const i = order.indexOf(l); return i < 0 ? 999 : i; };
  return [...labels].sort((a, b) => idx(a) - idx(b) || String(a).localeCompare(String(b)));
}

export function riskColor(v) {
  if (v >= 0.5) return 'var(--risk-high)';
  if (v >= 0.25) return 'var(--risk-mid)';
  return 'var(--risk-low)';
}

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** A little "badge" chip for an event class. */
export function classChip(label, extra = {}) {
  return h('span.chip', { style: { '--c': classColor(label) }, ...extra }, h('span.chip-dot', { 'aria-hidden': 'true' }), pretty(label));
}

/** Normalise an events array to [[s,e,label]] with s<=e, dropping junk. */
export function cleanEvents(events) {
  return arr(events)
    .filter((e) => Array.isArray(e) && e.length >= 3)
    .map(([s, e, l]) => { const a = num(s), b = num(e); return [Math.min(a, b), Math.max(a, b), String(l)]; })
    .sort((a, b) => a[0] - b[0]);
}

/** Normalise [[t,v]] pairs. */
export function cleanPairs(pairs) {
  return arr(pairs).filter((p) => Array.isArray(p) && p.length >= 2).map(([t, v]) => [num(t), num(v)]);
}

export function download(filename, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Accessible tab strip. onSelect(id) fires on click/arrow keys. */
export function tabStrip(items, activeId, onSelect, label = 'Choose') {
  const strip = h('div.tabs', { role: 'tablist', 'aria-label': label });
  const buttons = items.map(({ id, text }) => h('button.tab', {
    role: 'tab', type: 'button', 'aria-selected': String(id === activeId), tabindex: id === activeId ? '0' : '-1',
    dataset: { id }, onclick: () => select(id, true),
  }, text));
  function select(id, focus) {
    buttons.forEach((b) => {
      const on = b.dataset.id === id;
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus();
    });
    onSelect(id);
  }
  strip.addEventListener('keydown', (e) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const i = buttons.findIndex((b) => b.getAttribute('aria-selected') === 'true');
    let j = i;
    if (e.key === 'ArrowRight') j = (i + 1) % buttons.length;
    if (e.key === 'ArrowLeft') j = (i - 1 + buttons.length) % buttons.length;
    if (e.key === 'Home') j = 0;
    if (e.key === 'End') j = buttons.length - 1;
    select(buttons[j].dataset.id, true);
  });
  strip.append(...buttons);
  strip.select = (id) => select(id, false);
  return strip;
}

export function placeholderBanner(isPlaceholder, what) {
  return isPlaceholder
    ? h('p.placeholder-note', { role: 'note' }, h('strong', {}, 'Placeholder data. '), `${what} will be replaced with real numbers before judging.`)
    : document.createComment('real data');  // safe for Node.append (null would print "null")
}
