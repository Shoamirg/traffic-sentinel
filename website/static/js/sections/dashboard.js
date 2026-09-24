// Operator dashboard built purely from videos.json: KPIs, events per class,
// per video (stacked), rate per hour of footage, alarm exposure and a
// when-in-the-clip heat strip. A class filter drives every panel.
import { h, clear, arr, num, fmtNum, fmtTime, classColor, classChip, cleanEvents, cleanPairs, pretty, sortLabels } from '../util.js';

const BINS = 24;

function kpis(videos, evs) {
  const seconds = videos.reduce((a, v) => a + num(v.duration), 0);
  const perHour = seconds ? (evs.length / seconds) * 3600 : 0;
  let alarm = 0, riskSpan = 0;
  for (const v of videos) {
    const r = cleanPairs(v.risk);
    for (let i = 1; i < r.length; i++) {
      const dt = r[i][0] - r[i - 1][0];
      riskSpan += dt;
      if (r[i][1] >= 0.5) alarm += dt;
    }
  }
  const byClass = count(evs, (e) => e.label);
  const top = Object.entries(byClass).sort((a, b) => b[1] - a[1])[0];
  return h('div.tiles.tiles-dash', {},
    tile('Footage', `${fmtNum(seconds / 60, 1)} min`, `${videos.length} cameras / clips`),
    tile('Events', String(evs.length), 'after filter'),
    tile('Rate', `${fmtNum(perHour, 0)}/h`, 'events per hour of footage'),
    tile('Under alarm', riskSpan ? `${fmtNum((alarm / riskSpan) * 100, 1)}%` : '—', 'of time, risk ≥ 0.5'),
    tile('Most frequent', top ? pretty(top[0]) : '—', top ? `${top[1]} events` : '', true));
}

function tile(k, v, note, cap = false) {
  return h('div.tile', {}, h('div.tile-k', {}, k), h(cap ? 'div.tile-v.cap' : 'div.tile-v', {}, v), h('div.tile-note', {}, note));
}

function count(items, key) {
  const out = {};
  for (const it of items) out[key(it)] = (out[key(it)] || 0) + 1;
  return out;
}

function byClassPanel(evs) {
  const c = count(evs, (e) => e.label);
  const labels = sortLabels(Object.keys(c)).sort((a, b) => c[b] - c[a]);
  const max = Math.max(1, ...Object.values(c));
  return h('div.card', {}, h('h3', {}, 'Events per class'),
    labels.length ? h('ul.hbars', {}, labels.map((l) => h('li', {},
      h('span.hbar-label.cap', {}, pretty(l)),
      h('span.hbar-track', {}, h('span.hbar-fill', { style: { width: `${(c[l] / max) * 100}%`, '--c': classColor(l) } })),
      h('span.hbar-val', {}, String(c[l]))))) : h('p.muted', {}, 'No events match the filter.'));
}

function perVideoPanel(videos, evs) {
  const max = Math.max(1, ...videos.map((v) => evs.filter((e) => e.v === v.id).length));
  return h('div.card', {}, h('h3', {}, 'Events per video, by class'),
    h('ul.hbars.stacked', {}, videos.map((v) => {
      const mine = evs.filter((e) => e.v === v.id);
      const c = count(mine, (e) => e.label);
      return h('li', {},
        h('span.hbar-label', {}, v.id),
        h('span.hbar-track', {}, h('span.hbar-stack', { style: { width: `${(mine.length / max) * 100}%` } },
          sortLabels(Object.keys(c)).map((l) => h('i', { style: { flexGrow: c[l], '--c': classColor(l) }, title: `${pretty(l)}: ${c[l]}` })))),
        h('span.hbar-val', {}, String(mine.length)));
    })));
}

function ratePanel(videos, evs) {
  const rates = videos.map((v) => ({ id: v.id, r: num(v.duration) ? (evs.filter((e) => e.v === v.id).length / num(v.duration)) * 3600 : 0 }));
  const max = Math.max(1, ...rates.map((x) => x.r));
  return h('div.card', {}, h('h3', {}, 'Events per hour'), h('p.muted.small', {}, 'Clips carry no wall-clock time, so this is the rate per hour of footage.'),
    h('div.vbars', { role: 'img', 'aria-label': rates.map((x) => `${x.id}: ${fmtNum(x.r, 0)} per hour`).join('; ') },
      rates.map((x) => h('div.vbar', {},
        h('span.vbar-v', {}, fmtNum(x.r, 0)),
        h('span.vbar-col', {}, h('span.vbar-fill', { style: { height: `${(x.r / max) * 100}%` } })),
        h('span.vbar-k', {}, x.id.replace(/^sample_?/, '#'))))));
}

function heatStrip(videos, evs, jumpTo) {
  const rows = videos.map((v) => {
    const d = num(v.duration) || 1;
    const bins = new Array(BINS).fill(0);
    const labels = Array.from({ length: BINS }, () => new Set());
    for (const e of evs.filter((x) => x.v === v.id)) {
      for (let b = Math.floor((e.s / d) * BINS); b <= Math.min(BINS - 1, Math.floor((e.e / d) * BINS)); b++) {
        bins[b] += 1; labels[b].add(e.label);
      }
    }
    return { v, d, bins, labels };
  });
  const max = Math.max(1, ...rows.flatMap((r) => r.bins));
  return h('div.card.heat-card', {}, h('h3', {}, 'When things happen'),
    h('p.muted.small', {}, `Each clip split into ${BINS} slices; brighter = more concurrent events. Click a slice to watch it.`),
    h('div.heat', {}, rows.map(({ v, d, bins, labels }) => h('div.heat-row', {},
      h('span.heat-label', {}, v.id.replace(/^sample_?/, '#')),
      h('div.heat-cells', {}, bins.map((n, b) => {
        const t = (b / BINS) * d;
        const names = [...labels[b]].map(pretty).join(', ');
        return h('button.heat-cell', {
          type: 'button', style: { '--a': n ? 0.18 + 0.82 * (n / max) : 0 },
          title: `${v.id} ${fmtTime(t, 0)}–${fmtTime(((b + 1) / BINS) * d, 0)}${names ? `: ${names}` : ''}`,
          'aria-label': `${v.id} at ${fmtTime(t, 0)}, ${n} active events${names ? `: ${names}` : ''}. Watch.`,
          onclick: () => jumpTo(v.id, t),
        });
      }))))),
    h('p.muted.small', {}, 'Per-lane breakdown needs lane ids from the direction field - listed under next steps.'));
}

export function renderDashboard(host, ctx) {
  const videos = ctx.videos.filter((v) => v && v.id);
  if (!videos.length) { host.append(h('p.muted', {}, 'No predictions to summarise yet.')); return; }
  const all = videos.flatMap((v) => cleanEvents(v.events).map(([s, e, label]) => ({ v: v.id, s, e, label })));
  const labels = sortLabels([...new Set(all.map((e) => e.label))]);
  const active = new Set(labels);

  const filter = h('div.filter', { role: 'group', 'aria-label': 'Filter by event class' });
  const body = h('div.dash');
  const allBtn = h('button.filter-all', { type: 'button', onclick: () => { labels.forEach((l) => active.add(l)); sync(); } }, 'All classes');
  const chips = labels.map((l) => {
    const b = h('button.filter-chip', { type: 'button', 'aria-pressed': 'true', onclick: () => { if (active.has(l)) active.delete(l); else active.add(l); sync(); } }, classChip(l));
    b.dataset.label = l;
    return b;
  });
  filter.append(allBtn, ...chips);

  function sync() {
    chips.forEach((b) => b.setAttribute('aria-pressed', String(active.has(b.dataset.label))));
    const evs = all.filter((e) => active.has(e.label));
    clear(body).append(
      kpis(videos, evs),
      h('div.dash-grid', {}, byClassPanel(evs), perVideoPanel(videos, evs), ratePanel(videos, evs)),
      heatStrip(videos, evs, ctx.jumpTo || (() => {})));
  }
  host.append(filter, body);
  sync();
}
