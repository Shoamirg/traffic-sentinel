// Results: one EventPlayer per sample video (tabbed), examples per class across
// all samples, dev-set evaluation (per-class F1, error breakdown, ablations,
// runtime) and honest failure cases - each linked back into the player.
import { h, clear, arr, obj, num, fmtNum, fmtTime, classChip, classList, cleanEvents, pretty, tabStrip, placeholderBanner } from '../util.js';
import { createEventPlayer } from '../event-player.js';

function examplesPerClass(videos, jumpTo) {
  const ids = classList();
  const rows = ids.map((label) => {
    const hits = [];
    for (const v of videos) for (const e of cleanEvents(v.events)) if (e[2] === label) hits.push({ v, e });
    return h(hits.length ? 'li.ex-row' : 'li.ex-row.is-empty', {},
      h('div.ex-label', {}, classChip(label), h('span.ex-count', {}, String(hits.length))),
      h('div.ex-jumps', {}, hits.length
        ? hits.slice(0, 6).map(({ v, e }) => h('button.jump', {
          type: 'button', onclick: () => jumpTo(v.id, e[0]),
          'aria-label': `${pretty(label)} in ${v.id} at ${fmtTime(e[0])}`,
        }, h('span.jump-v', {}, v.id.replace(/^sample_?/, '#')), ` ${fmtTime(e[0])}`))
        : h('span.muted.small', {}, 'not present in the samples')));
  });
  return h('div.card', {}, h('h3', {}, 'Examples per class'),
    h('p.muted.small', {}, 'Every predicted instance across the sample videos. Click to watch it.'),
    h('ul.ex-list', {}, rows));
}

function scoreTiles(dev) {
  const tiles = [['Score A', dev.score_a, 'event detection (temporal IoU F1)'], ['Score B', dev.score_b, 'risk anticipation (not scored: no accidents in the samples)'], ['Model score', dev.model_score, 'combined']];
  return h('div.tiles', {}, tiles.filter(([, v]) => v !== undefined).map(([k, v, note]) => h('div.tile', {},
    h('div.tile-k', {}, k), h('div.tile-v', {}, v === null ? '—' : fmtNum(v, 3)), h('div.tile-note', {}, note))));
}

function bar(v) {
  const x = Math.max(0, Math.min(1, num(v)));
  return h('span.mini', {}, h('span.mini-bar', { style: { width: `${x * 100}%` } }), h('span.mini-v', {}, fmtNum(v, 2)));
}

function perClassTable(perClass) {
  const ids = [...new Set([...classList(), ...Object.keys(perClass)])].filter((k) => perClass[k]);
  if (!ids.length) return null;
  return h('div.table-wrap', { tabindex: '0', role: 'region', 'aria-label': 'Per-class metrics' },
    h('table.data.metrics', {},
      h('caption', {}, 'Per class on the dev set - F1 at three temporal-IoU thresholds, and where the errors are'),
      h('thead', {}, h('tr', {}, ['Class', 'F1@0.3', 'F1@0.5', 'F1@0.7', 'TP', 'FP', 'FN', 'Errors'].map((x) => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, ids.map((id) => {
        const m = obj(perClass[id]);
        const tp = num(m.tp), fp = num(m.fp), fn = num(m.fn), tot = Math.max(1, tp + fp + fn);
        return h('tr', {},
          h('th', { scope: 'row' }, classChip(id)),
          h('td', {}, bar(m['f1@0.3'])), h('td', {}, bar(m['f1@0.5'])), h('td', {}, bar(m['f1@0.7'])),
          h('td.num', {}, String(tp)), h('td.num', {}, String(fp)), h('td.num', {}, String(fn)),
          h('td', {}, h('span.stack', { title: `TP ${tp} · FP ${fp} · FN ${fn}`, 'aria-label': `TP ${tp}, FP ${fp}, FN ${fn}` },
            h('i.st-tp', { style: { width: `${(tp / tot) * 100}%` } }),
            h('i.st-fp', { style: { width: `${(fp / tot) * 100}%` } }),
            h('i.st-fn', { style: { width: `${(fn / tot) * 100}%` } }))));
      }))),
    h('p.muted.small.stack-key', {}, h('i.st-tp'), 'true positive ', h('i.st-fp'), 'false positive ', h('i.st-fn'), 'missed'));
}

function ablationTable(rows) {
  if (!rows.length) return null;
  const base = num(rows[0].score_a);
  return h('div.table-wrap', { tabindex: '0', role: 'region', 'aria-label': 'Ablations' },
    h('table.data', {}, h('caption', {}, 'Score A by development stage, on our (draft) dev labels'),
      h('thead', {}, h('tr', {}, ['Stage', 'Score A', 'Δ vs first', 'What changed'].map((x) => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, rows.map((r, i) => {
        const d = num(r.score_a) - base;
        return h('tr', {}, h('th', { scope: 'row' }, r.name || ''), h('td.num', {}, fmtNum(r.score_a, 3)),
          h(i === 0 ? 'td.num' : d < 0 ? 'td.num.neg' : 'td.num.pos', {}, i === 0 ? '—' : `${d >= 0 ? '+' : ''}${fmtNum(d, 3)}`),
          h('td', {}, r.notes || ''));
      }))));
}

function runtimeTable(rows) {
  if (!rows.length) return null;
  return h('div.table-wrap', { tabindex: '0', role: 'region', 'aria-label': 'Runtime' },
    h('table.data', {}, h('caption', {}, 'Runtime - official harness, Part A + Part B (budget: 3× video length)'),
      h('thead', {}, h('tr', {}, ['Video', 'Length', 'Processing', 'Of video length'].map((x) => h('th', { scope: 'col' }, x)))),
      h('tbody', {}, rows.map((r) => h('tr', {}, h('th', { scope: 'row' }, r.video || ''),
        h('td.num', {}, fmtTime(r.duration, 0)), h('td.num', {}, `${fmtNum(r.seconds, 1)} s`),
        h('td.num', {}, `${fmtNum(r.x_realtime, 2)}×`))))));
}

function failures(list, jumpTo, videos) {
  if (!list.length) return null;
  return h('div.failures', {}, list.map((f) => {
    const hasVideo = videos.some((v) => v.id === f.video);
    return h('article.failure', {},
      h('h4', {}, f.title || 'Failure'),
      h('p', {}, f.text || ''),
      hasVideo ? h('button.btn.btn-small', { type: 'button', onclick: () => jumpTo(f.video, num(f.t)) }, `Watch ${f.video} @ ${fmtTime(f.t)}`) : null);
  }));
}

export function renderResults(host, ctx) {
  const videos = ctx.videos.filter((v) => v && v.id);
  const metrics = ctx.metrics;
  let player = null;
  let current = null;
  let tabs = null;

  function jumpTo(id, t) {
    if (!tabs || !videos.some((v) => v.id === id)) return;
    tabs.select(id);
    player?.seek(t);
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    document.getElementById('results').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth' });
  }
  const meta = h('p.video-meta');
  const playerHost = h('div', { role: 'tabpanel' });

  function show(id) {
    const v = videos.find((x) => x.id === id);
    if (!v || current === id) return;
    current = id;
    player?.destroy();
    clear(playerHost);
    const evs = cleanEvents(v.events);
    meta.replaceChildren(h('strong', {}, v.title || v.id), ` · ${v.width && v.height ? `${v.width}×${v.height} · ` : ''}${v.fps ? `${fmtNum(v.fps, 0)} fps · ` : ''}${fmtTime(v.duration, 0)} · ${evs.length} events`);
    player = createEventPlayer(playerHost, { title: v.title || v.id, video: v.video, poster: v.poster, events: v.events, risk: v.risk, duration: v.duration });
    playerHost.setAttribute('aria-label', `Results for ${v.id}`);
  }

  const placeholder = videos.some((v) => v.placeholder);
  host.append(placeholderBanner(placeholder, 'Sample predictions and videos'));
  if (!videos.length) {
    host.append(h('p.muted', {}, 'Sample results not published yet.'));
  } else {
    tabs = tabStrip(videos.map((v) => ({ id: v.id, text: v.id })), videos[0].id, show, 'Sample video');
    host.append(tabs, meta, playerHost);
    show(videos[0].id);
    host.append(examplesPerClass(videos, jumpTo));
  }

  // ---- evaluation --------------------------------------------------------------
  const dev = obj(metrics.dev_set);
  const evalBox = h('div.eval');
  evalBox.append(h('h3.sub', {}, 'Evaluation on our dev split'), placeholderBanner(metrics.placeholder, 'Scores, ablations and runtimes'), scoreTiles(dev));
  evalBox.append(perClassTable(obj(dev.per_class)) || h('p.muted', {}, 'Per-class metrics not published yet.'));
  evalBox.append(h('div.stacked-tables', {}, ablationTable(arr(metrics.ablations)), runtimeTable(arr(metrics.runtime))));
  const fails = failures(arr(metrics.failures), jumpTo, videos);
  if (fails) evalBox.append(h('h3.sub', {}, 'Where it fails'), h('p.muted', {}, 'Stated plainly, with a link to the moment it happens.'), fails);
  host.append(evalBox);

  return { jumpTo };
}
