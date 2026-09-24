// Dependency-free canvas line chart: multi-series, hover read-out, optional
// playhead + click/drag/keyboard seeking. Used for risk curves, object counts,
// brightness and density. Redraws only on data/size change; the playhead is a
// CSS-transformed overlay so video sync costs nothing.
import { h, fmtNum } from './util.js';

const PAD = { l: 38, r: 10, t: 10, b: 22 };

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function lowerBound(xs, x) {
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function niceStep(range, target) {
  const raw = range / Math.max(1, target);
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const n = raw / p;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
}

export class LineChart {
  constructor(host, opts) {
    this.opts = { height: 170, yMin: 0, formatX: (x) => `${fmtNum(x, 1)}s`, formatY: (y) => fmtNum(y, 2), ...opts };
    this.hidden = new Set();
    this.playhead = null;
    this.root = h('div.chart', { style: { height: `${this.opts.height}px` } });
    this.canvas = h('canvas', {
      role: 'img',
      'aria-label': this.opts.ariaLabel || 'Line chart',
      tabindex: this.opts.onSeek ? '0' : null,
    });
    this.head = h('div.chart-playhead', { 'aria-hidden': 'true', hidden: true });
    this.cursor = h('div.chart-cursor', { 'aria-hidden': 'true', hidden: true });
    this.tip = h('div.chart-tip', { 'aria-hidden': 'true', hidden: true });
    this.root.append(this.canvas, this.cursor, this.head, this.tip);
    host.append(this.root);
    this.computeBounds();
    this.bind();
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.root);
  }

  computeBounds() {
    const { series = [], xMin, xMax, yMin, yMax } = this.opts;
    let lo = Infinity, hi = -Infinity, top = -Infinity;
    for (const s of series) {
      if (this.hidden.has(s.name) || !s.xs?.length) continue;
      lo = Math.min(lo, s.xs[0]);
      hi = Math.max(hi, s.xs[s.xs.length - 1]);
      for (const y of s.ys) if (y > top) top = y;
    }
    this.x0 = xMin ?? (Number.isFinite(lo) ? lo : 0);
    this.x1 = xMax ?? (Number.isFinite(hi) && hi > this.x0 ? hi : this.x0 + 1);
    this.y0 = yMin;
    this.y1 = yMax ?? (Number.isFinite(top) && top > yMin ? top * 1.1 : yMin + 1);
  }

  setSeriesVisible(name, visible) {
    if (visible) this.hidden.delete(name); else this.hidden.add(name);
    if (!this.opts.yMax) this.computeBounds();
    this.draw();
  }

  px(x) { return PAD.l + ((x - this.x0) / (this.x1 - this.x0)) * this.w; }
  py(y) { return PAD.t + (1 - (y - this.y0) / (this.y1 - this.y0)) * this.hgt; }
  xAt(clientX) {
    const r = this.canvas.getBoundingClientRect();
    const f = (clientX - r.left - PAD.l) / this.w;
    return this.x0 + Math.min(1, Math.max(0, f)) * (this.x1 - this.x0);
  }

  draw() {
    const W = this.root.clientWidth, H = this.root.clientHeight;
    if (!W || !H) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(H * dpr);
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${H}px`;
    this.w = W - PAD.l - PAD.r;
    this.hgt = H - PAD.t - PAD.b;
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    this.drawGrid(ctx);
    for (const band of this.opts.bands || []) this.drawBand(ctx, band);
    for (const s of this.opts.series || []) if (!this.hidden.has(s.name)) this.drawSeries(ctx, s);
    for (const line of this.opts.hlines || []) this.drawHLine(ctx, line);
    this.placePlayhead();
  }

  drawGrid(ctx) {
    const grid = cssVar('--chart-grid', '#2a3038');
    const muted = cssVar('--muted', '#9aa3ae');
    ctx.font = `11px ${cssVar('--font-mono', 'monospace')}`;
    ctx.lineWidth = 1;
    ctx.fillStyle = muted;
    const yStep = niceStep(this.y1 - this.y0, 4);
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let y = Math.ceil(this.y0 / yStep) * yStep; y <= this.y1 + 1e-9; y += yStep) {
      const yy = Math.round(this.py(y)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(PAD.l, yy); ctx.lineTo(PAD.l + this.w, yy); ctx.stroke();
      ctx.fillText(this.opts.formatY(y), PAD.l - 6, yy);
    }
    const xStep = niceStep(this.x1 - this.x0, Math.max(2, Math.floor(this.w / 70)));
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let x = Math.ceil(this.x0 / xStep) * xStep; x <= this.x1 + 1e-9; x += xStep) {
      ctx.fillText(this.opts.formatX(x), this.px(x), PAD.t + this.hgt + 6);
    }
  }

  drawBand(ctx, { from, to, color }) {
    ctx.fillStyle = color;
    const a = this.px(Math.max(this.x0, from)), b = this.px(Math.min(this.x1, to));
    ctx.fillRect(a, PAD.t, Math.max(1, b - a), this.hgt);
  }

  drawHLine(ctx, { y, label, color, dash = [4, 4] }) {
    const yy = Math.round(this.py(y)) + 0.5;
    ctx.save();
    ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, yy); ctx.lineTo(PAD.l + this.w, yy); ctx.stroke();
    if (label) {
      ctx.setLineDash([]);
      ctx.font = `600 10px ${cssVar('--font-mono', 'monospace')}`;
      ctx.fillStyle = color; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(label, PAD.l + this.w - 2, yy - 3);
    }
    ctx.restore();
  }

  strokeFor(ctx, s) {
    if (!s.gradient) return s.color;
    // vertical gradient keyed to y values, e.g. risk: green -> amber -> red
    const g = ctx.createLinearGradient(0, this.py(this.y0), 0, this.py(this.y1));
    for (const [y, c] of s.gradient) g.addColorStop(Math.min(1, Math.max(0, (y - this.y0) / (this.y1 - this.y0))), c);
    return g;
  }

  drawSeries(ctx, s) {
    const { xs, ys } = s;
    if (!xs?.length) return;
    const stroke = this.strokeFor(ctx, s);
    ctx.beginPath();
    xs.forEach((x, i) => (i ? ctx.lineTo(this.px(x), this.py(ys[i])) : ctx.moveTo(this.px(x), this.py(ys[i]))));
    ctx.lineWidth = s.width || 1.6;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = stroke;
    ctx.stroke();
    if (s.fill) {
      ctx.lineTo(this.px(xs[xs.length - 1]), this.py(this.y0));
      ctx.lineTo(this.px(xs[0]), this.py(this.y0));
      ctx.closePath();
      ctx.globalAlpha = s.fillAlpha ?? 0.16;
      ctx.fillStyle = stroke;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  setPlayhead(x) {
    this.playhead = x;
    this.placePlayhead();
  }

  placePlayhead() {
    if (this.playhead === null || !this.w) { this.head.hidden = true; return; }
    this.head.hidden = false;
    this.head.style.transform = `translateX(${this.px(Math.min(this.x1, Math.max(this.x0, this.playhead)))}px)`;
    this.head.style.top = `${PAD.t}px`;
    this.head.style.height = `${this.hgt}px`;
  }

  showTip(clientX) {
    const x = this.xAt(clientX);
    const rows = [];
    for (const s of this.opts.series || []) {
      if (this.hidden.has(s.name) || !s.xs?.length) continue;
      const i = Math.min(s.xs.length - 1, lowerBound(s.xs, x));
      const j = i > 0 && Math.abs(s.xs[i - 1] - x) < Math.abs(s.xs[i] - x) ? i - 1 : i;
      rows.push(h('div.chart-tip-row', {}, h('i', { style: { background: s.swatch || s.color } }), `${s.name} `, h('b', {}, this.opts.formatY(s.ys[j]))));
    }
    this.tip.replaceChildren(h('div.chart-tip-x', {}, this.opts.formatX(x)), ...rows);
    this.tip.hidden = false;
    this.cursor.hidden = false;
    const px = this.px(x);
    this.cursor.style.transform = `translateX(${px}px)`;
    this.cursor.style.top = `${PAD.t}px`;
    this.cursor.style.height = `${this.hgt}px`;
    const tw = this.tip.offsetWidth;
    const left = px + 12 + tw > this.root.clientWidth ? px - 12 - tw : px + 12;
    this.tip.style.transform = `translate(${Math.max(0, left)}px, ${PAD.t}px)`;
  }

  hideTip() { this.tip.hidden = true; this.cursor.hidden = true; }

  bind() {
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => {
      this.showTip(e.clientX);
      if (this.dragging) this.opts.onSeek?.(this.xAt(e.clientX));
    });
    c.addEventListener('pointerleave', () => this.hideTip());
    if (!this.opts.onSeek) return;
    c.style.cursor = 'pointer';
    c.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      c.setPointerCapture(e.pointerId);
      this.opts.onSeek(this.xAt(e.clientX));
    });
    const stop = () => { this.dragging = false; };
    c.addEventListener('pointerup', stop);
    c.addEventListener('pointercancel', stop);
    c.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 5 : 1;
      const cur = this.playhead ?? this.x0;
      if (e.key === 'ArrowRight') this.opts.onSeek(Math.min(this.x1, cur + step));
      else if (e.key === 'ArrowLeft') this.opts.onSeek(Math.max(this.x0, cur - step));
      else if (e.key === 'Home') this.opts.onSeek(this.x0);
      else if (e.key === 'End') this.opts.onSeek(this.x1);
      else return;
      e.preventDefault();
    });
  }

  destroy() {
    this.ro.disconnect();
    this.root.remove();
  }
}

/** Legend with toggle buttons bound to a chart. */
export function legend(chart, items) {
  return h('div.legend', { role: 'group', 'aria-label': 'Toggle series' }, items.map(({ name, color }) => {
    const btn = h('button.legend-item', {
      type: 'button', 'aria-pressed': 'true', style: { '--c': color },
      onclick: () => {
        const on = btn.getAttribute('aria-pressed') !== 'true';
        btn.setAttribute('aria-pressed', String(on));
        chart.setSeriesVisible(name, on);
      },
    }, h('span.legend-swatch', { 'aria-hidden': 'true' }), name);
    return btn;
  }));
}
