// EDA: per-video footage profile, interactive counts / density / brightness
// charts, motion heatmaps + trajectories + learned directions, and findings.
import { h, clear, arr, obj, num, fmtNum, fmtTime, objectColor, tabStrip, placeholderBanner } from '../util.js';
import { LineChart, legend } from '../chart.js';

const OBJECTS = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle'];

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

function lighting(b) {
  if (!Number.isFinite(b)) return '—';
  if (b >= 110) return 'day';
  if (b >= 60) return 'dusk / overcast';
  return 'night';
}

function objectKeys(counts) {
  const keys = Object.keys(counts).filter((k) => k !== 't' && Array.isArray(counts[k]));
  return [...OBJECTS.filter((k) => keys.includes(k)), ...keys.filter((k) => !OBJECTS.includes(k))];
}

/** total road users per sample = density proxy */
function density(counts) {
  const keys = objectKeys(counts);
  const t = arr(counts.t);
  return t.map((_, i) => keys.reduce((a, k) => a + num(counts[k][i]), 0));
}

function overviewTable(videos) {
  const rows = videos.map((v) => {
    const c = obj(v.counts);
    const b = mean(arr(obj(v.brightness).v).map(Number));
    const d = density(c);
    const [w, hgt] = arr(v.resolution);
    return h('tr', {},
      h('th', { scope: 'row' }, v.id),
      h('td.num', {}, w && hgt ? `${w}×${hgt}` : '—'),
      h('td.num', {}, fmtNum(v.fps, 0)),
      h('td.num', {}, fmtTime(v.duration, 0)),
      h('td.num', {}, v.size_mb !== undefined ? `${fmtNum(v.size_mb, 1)} MB` : '—'),
      h('td.num', {}, fmtNum(b, 0)),
      h('td', {}, lighting(b)),
      h('td.num', {}, fmtNum(mean(d), 1)),
      h('td.num', {}, d.length ? String(Math.max(...d)) : '—'));
  });
  return h('div.table-wrap', { tabindex: '0', role: 'region', 'aria-label': 'Footage overview table' },
    h('table.data', {},
      h('caption', {}, 'Footage profile per sample video'),
      h('thead', {}, h('tr', {}, ['Video', 'Resolution', 'FPS', 'Length', 'Size', 'Mean brightness', 'Lighting', 'Mean road users', 'Peak'].map((x) => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, rows)));
}

function totalsBars(totals) {
  const entries = Object.entries(obj(totals)).filter(([, v]) => Number.isFinite(+v)).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  const max = Math.max(...entries.map(([, v]) => +v), 1);
  return h('div.card', {}, h('h3', {}, 'Road users across all samples'), h('p.muted.small', {}, 'Unique tracks per class.'),
    h('ul.hbars', {}, entries.map(([k, v]) => h('li', {},
      h('span.hbar-label.cap', {}, k),
      h('span.hbar-track', {}, h('span.hbar-fill', { style: { width: `${(v / max) * 100}%`, '--c': objectColor(k) } })),
      h('span.hbar-val', {}, String(v))))));
}

function figure(src, caption, alt) {
  if (!src) return null;
  const img = h('img', { src, alt, loading: 'lazy', decoding: 'async', width: 640, height: 360 });
  const fig = h('figure.eda-fig', {}, img, h('figcaption', {}, caption));
  img.addEventListener('error', () => { img.replaceWith(h('div.img-missing', {}, 'image not generated yet')); });
  return fig;
}

function videoPanel(v, charts) {
  const c = obj(v.counts);
  const t = arr(c.t).map(Number);
  const keys = objectKeys(c);
  const panel = h('div.eda-panel');
  const grid = h('div.eda-charts');
  panel.append(grid);

  if (t.length && keys.length) {
    const box = h('div.card.chart-card', {}, h('h3', {}, 'Objects in frame over time'), h('p.muted.small', {}, 'Detections per analysed frame, by class. Toggle classes below.'));
    grid.append(box);
    const series = keys.map((k) => ({ name: k, color: objectColor(k), xs: t, ys: arr(c[k]).map(Number) }));
    const ch = new LineChart(box, { height: 200, series, formatX: (x) => fmtTime(x, 0), formatY: (y) => fmtNum(y, 0), ariaLabel: `Object counts over time for ${v.id}` });
    box.append(legend(ch, series));
    charts.push(ch);

    const d = density(c);
    const dbox = h('div.card.chart-card', {}, h('h3', {}, 'Density'), h('p.muted.small', {}, 'All road users in frame - the signal the congestion rule builds on.'));
    grid.append(dbox);
    charts.push(new LineChart(dbox, { height: 160, series: [{ name: 'road users', color: '#4fd8eb', xs: t, ys: d, fill: true }], formatX: (x) => fmtTime(x, 0), formatY: (y) => fmtNum(y, 0), ariaLabel: `Density over time for ${v.id}` }));
  } else {
    grid.append(h('p.muted', {}, 'No per-frame counts for this video.'));
  }

  const b = obj(v.brightness);
  if (arr(b.t).length) {
    const bbox = h('div.card.chart-card', {}, h('h3', {}, 'Brightness'), h('p.muted.small', {}, 'Mean frame luma (0-255). Drives the day / night split.'));
    grid.append(bbox);
    charts.push(new LineChart(bbox, { height: 160, yMin: 0, yMax: 255, series: [{ name: 'luma', color: '#e6c86e', xs: arr(b.t).map(Number), ys: arr(b.v).map(Number) }], formatX: (x) => fmtTime(x, 0), formatY: (y) => fmtNum(y, 0), ariaLabel: `Brightness over time for ${v.id}` }));
  }

  panel.append(h('div.eda-figs', {},
    figure(v.heatmap, 'Motion heatmap - where road users spend time', `Motion heatmap for ${v.id}`),
    figure(v.trajectories, 'Trajectories - one line per track', `Vehicle trajectories for ${v.id}`),
    figure(v.directions, 'Learned lane directions - arrows = dominant heading per cell', `Learned direction field for ${v.id}`)));
  return panel;
}

export function renderEDA(host, ctx) {
  const eda = ctx.eda;
  const videos = arr(eda.videos);
  host.append(placeholderBanner(eda.placeholder, 'EDA numbers and images'));
  if (!videos.length) { host.append(h('p.muted', {}, 'EDA data not published yet.')); return; }

  host.append(h('div.eda-top', {}, overviewTable(videos), totalsBars(eda.totals)));

  let charts = [];
  const panelHost = h('div', { role: 'tabpanel' });
  const show = (id) => {
    charts.forEach((c) => c.destroy());
    charts = [];
    const v = videos.find((x) => x.id === id) || videos[0];
    clear(panelHost).append(videoPanel(v, charts));
    panelHost.setAttribute('aria-label', `EDA for ${v.id}`);
  };
  host.append(h('h3.sub', {}, 'Per video'), tabStrip(videos.map((v) => ({ id: v.id, text: v.id })), videos[0].id, show, 'EDA video'), panelHost);
  show(videos[0].id);

  const findings = arr(eda.findings);
  if (findings.length) {
    host.append(h('h3.sub', {}, 'Findings that shaped the solution'),
      h('ol.findings', {}, findings.map((f, i) => h('li.finding', {},
        h('span.finding-n', { 'aria-hidden': 'true' }, String(i + 1).padStart(2, '0')),
        h('div', {}, h('h4', {}, obj(f).title || ''), h('p', {}, obj(f).text || ''))))));
  }
}
