// EventPlayer - the one reusable "video + event timeline + risk curve" widget.
// Used by Results (sample videos) and the Live Demo (uploaded video).
//
//   const p = createEventPlayer(host, {title, video, fallbackVideo, poster, events, risk, duration});
//   p.seek(12.5); p.destroy();
//
// Clicking an event bar (or any point on a track / the risk chart) seeks the
// video; a shared playhead runs across timeline and risk chart while playing.
// If the video cannot load, the widget keeps working on a virtual clock.
import { h, clear, classColor, classChip, cleanEvents, cleanPairs, fmtTime, fmtNum, pretty, sortLabels, num } from './util.js';
import { LineChart } from './chart.js';

const ALARM = 0.5;
const RISK_GRADIENT = [[0, '#5fbf7f'], [0.25, '#5fbf7f'], [0.38, '#f0a13a'], [0.5, '#e5564b'], [1, '#e5564b']];
let uid = 0;

function riskLevel(v) {
  if (v >= ALARM) return { key: 'high', text: 'ALARM' };
  if (v >= 0.25) return { key: 'mid', text: 'ELEVATED' };
  return { key: 'low', text: 'LOW' };
}

/** Contiguous spans where risk >= threshold -> [[s, e]]. */
function alarmSpans(risk, thr = ALARM) {
  const out = [];
  let start = null;
  risk.forEach(([t, v], i) => {
    if (v >= thr && start === null) start = t;
    if ((v < thr || i === risk.length - 1) && start !== null) { out.push([start, t]); start = null; }
  });
  return out;
}

function riskAt(risk, t) {
  if (!risk.length) return null;
  let lo = 0, hi = risk.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (risk[m][0] <= t) lo = m; else hi = m - 1; }
  return risk[lo][1];
}

export function createEventPlayer(host, data, opts = {}) {
  const id = `ep${++uid}`;
  const events = cleanEvents(data.events);
  const risk = cleanPairs(data.risk).sort((a, b) => a[0] - b[0]);
  const spans = alarmSpans(risk);
  let duration = num(data.duration) || Math.max(1, ...events.map((e) => e[1]), ...risk.map((r) => r[0]));
  let virtualTime = 0;
  let videoOk = Boolean(data.video || data.fallbackVideo);
  let raf = 0;

  // ---- video ---------------------------------------------------------------
  const video = h('video', {
    controls: true, playsinline: true, preload: 'metadata', poster: data.poster || null,
    'aria-label': `${data.title || 'Video'} - annotated`,
  });
  const stageMsg = h('div.ep-stage-msg', { hidden: true });
  const sources = [data.video, data.fallbackVideo].filter(Boolean);
  let srcIdx = 0;
  if (sources.length) video.src = sources[0];
  video.addEventListener('error', () => {
    srcIdx += 1;
    if (srcIdx < sources.length) { video.src = sources[srcIdx]; return; }
    videoOk = false;
    video.hidden = true;
    stageMsg.hidden = false;
    stageMsg.replaceChildren(h('strong', {}, 'Annotated video not available.'),
      h('span', {}, ' Timeline and risk curve still work - click them to scrub.'));
  });
  if (!sources.length) {
    video.hidden = true; stageMsg.hidden = false;
    stageMsg.textContent = 'No video for this result. Timeline and risk curve still work.';
  }
  video.addEventListener('loadedmetadata', () => {
    if (!num(data.duration) && Number.isFinite(video.duration)) { duration = video.duration; rebuild(); }
  });

  const now = () => (videoOk && !video.hidden ? video.currentTime : virtualTime);

  // ---- "now" bar -----------------------------------------------------------
  const clock = h('span.ep-clock', {}, '0:00.0');
  const riskVal = h('span.ep-risk-val', {}, '—');
  const riskTag = h('span.ep-risk-tag', {}, '');
  const activeBox = h('div.ep-active', { 'aria-live': 'off' });
  const nowBar = h('div.ep-now', {},
    h('div.ep-now-cell', {}, h('span.ep-k', {}, 'TIME'), clock, h('span.ep-dur', {}, ` / ${fmtTime(duration)}`)),
    h('div.ep-now-cell.ep-now-risk', {}, h('span.ep-k', {}, 'RISK'), riskVal, riskTag),
    h('div.ep-now-cell.ep-now-events', {}, h('span.ep-k', {}, 'ACTIVE'), activeBox));

  // ---- timeline --------------------------------------------------------------
  const tip = h('div.ep-tip', { role: 'tooltip', id: `${id}-tip`, hidden: true });
  const timeline = h('div.ep-tl', { role: 'group', 'aria-label': 'Event timeline. Activate an event to jump the video to it.' });
  let tlHead, eventButtons = [];

  function showTip(btn, text) {
    tip.textContent = text;
    tip.hidden = false;
    const r = btn.getBoundingClientRect(), p = timeline.getBoundingClientRect();
    const x = Math.min(p.width - tip.offsetWidth - 4, Math.max(4, r.left - p.left + r.width / 2 - tip.offsetWidth / 2));
    tip.style.transform = `translate(${x}px, ${r.top - p.top - tip.offsetHeight - 6}px)`;
  }
  const hideTip = () => { tip.hidden = true; };

  function trackSeek(e, track) {
    if (e.target !== track) return;
    const r = track.getBoundingClientRect();
    seek(((e.clientX - r.left) / r.width) * duration);
  }

  function lane(labelNode, items, kind) {
    const track = h('div.ep-track', { onclick: (e) => trackSeek(e, track) });
    for (const it of items) {
      const [s, e, label] = it;
      const left = (s / duration) * 100;
      const width = Math.max(0, ((e - s) / duration) * 100);
      const text = kind === 'alarm'
        ? `Risk ≥ ${ALARM} · ${fmtTime(s)} – ${fmtTime(e)}`
        : `${pretty(label)} · ${fmtTime(s)} – ${fmtTime(e)} (${fmtNum(e - s, 1)} s)`;
      const btn = h(`button.ep-ev${kind === 'alarm' ? '.ep-ev-alarm' : ''}`, {
        type: 'button', style: { left: `${left}%`, width: `${width}%`, '--c': kind === 'alarm' ? 'var(--risk-high)' : classColor(label) },
        'aria-label': `${text}. Jump to start.`, 'aria-describedby': `${id}-tip`,
        onclick: () => seek(s),
        onpointerenter: () => showTip(btn, text), onpointerleave: hideTip,
        onfocus: () => showTip(btn, text), onblur: hideTip,
      });
      btn._span = [s, e];
      track.append(btn);
      eventButtons.push(btn);
    }
    return [h('div.ep-lane-label', {}, labelNode), track];
  }

  function axis() {
    const ticks = h('div.ep-axis');
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => duration / s <= 8) || 600;
    for (let t = 0; t <= duration + 1e-6; t += step) {
      ticks.append(h('span.ep-tick', { style: { left: `${(t / duration) * 100}%` } }, fmtTime(t, 0)));
    }
    return [h('div.ep-lane-label', { 'aria-hidden': 'true' }), ticks];
  }

  function buildTimeline() {
    eventButtons = [];
    const labels = sortLabels([...new Set(events.map((e) => e[2]))]);
    const grid = h('div.ep-tl-grid');
    for (const label of labels) grid.append(...lane(classChip(label), events.filter((e) => e[2] === label), 'event'));
    if (risk.length) grid.append(...lane(h('span.chip.chip-alarm', {}, h('span.chip-dot'), `risk ≥ ${ALARM}`), spans.map(([s, e]) => [s, e, 'alarm']), 'alarm'));
    grid.append(...axis());
    tlHead = h('div.ep-tl-head', { 'aria-hidden': 'true' });
    grid.append(h('div.ep-tl-overlay', { 'aria-hidden': 'true' }, tlHead));
    const empty = labels.length ? null : h('p.ep-empty', {}, 'No events detected in this clip.');
    clear(timeline).append(...[empty, grid, tip].filter(Boolean));
  }

  // ---- risk chart --------------------------------------------------------------
  const riskHost = h('div.ep-risk');
  let chart = null;
  function buildChart() {
    chart?.destroy();
    chart = null;
    clear(riskHost);
    if (!risk.length) { riskHost.append(h('p.ep-empty', {}, 'No risk curve for this clip.')); return; }
    riskHost.append(h('div.ep-risk-head', {},
      h('h4', {}, 'Accident risk (causal, 5 s horizon)'),
      h('span.ep-risk-legend', {}, h('i.lv-low'), 'low ', h('i.lv-mid'), 'elevated ', h('i.lv-high'), `alarm ≥ ${ALARM}`)));
    const peak = Math.max(...risk.map((r) => r[1]));
    chart = new LineChart(riskHost, {
      height: 150, yMin: 0, yMax: 1, xMin: 0, xMax: duration,
      ariaLabel: `Risk curve. Peak ${fmtNum(peak)}; ${spans.length} alarm span(s). Use arrow keys to scrub.`,
      series: [{ name: 'risk', xs: risk.map((r) => r[0]), ys: risk.map((r) => r[1]), color: '#e5564b', swatch: '#e5564b', gradient: RISK_GRADIENT, fill: true, fillAlpha: 0.22, width: 2 }],
      bands: spans.map(([s, e]) => ({ from: s, to: e, color: 'rgba(255,77,77,0.08)' })),
      hlines: [{ y: ALARM, label: `ALARM ${ALARM}`, color: '#e5564b' }],
      formatX: (x) => fmtTime(x, 0),
      formatY: (y) => fmtNum(y, 2),
      onSeek: (x) => seek(x),
    });
  }

  // ---- event list ("examples per class") ----------------------------------------
  const list = h('div.ep-list');
  function buildList() {
    clear(list);
    if (!events.length) return;
    const labels = sortLabels([...new Set(events.map((e) => e[2]))]);
    list.append(h('h4', {}, `Events (${events.length})`));
    const grid = h('div.ep-list-grid');
    for (const label of labels) {
      const items = events.filter((e) => e[2] === label);
      grid.append(h('div.ep-list-row', {}, classChip(label), h('div.ep-list-jumps', {}, items.map(([s, e]) =>
        h('button.jump', { type: 'button', onclick: () => seek(s), 'aria-label': `${pretty(label)} at ${fmtTime(s)}, jump` },
          `${fmtTime(s)}`, h('span', {}, ` · ${fmtNum(e - s, 1)}s`))))));
    }
    list.append(grid);
  }

  // ---- sync ------------------------------------------------------------------------
  function update() {
    const t = now();
    const pct = Math.min(100, Math.max(0, (t / duration) * 100));
    if (tlHead) tlHead.style.left = `${pct}%`;
    chart?.setPlayhead(t);
    clock.textContent = fmtTime(t);
    const r = riskAt(risk, t);
    if (r === null) { riskVal.textContent = '—'; riskTag.textContent = ''; } else {
      const lv = riskLevel(r);
      riskVal.textContent = fmtNum(r, 2);
      riskTag.textContent = lv.text;
      riskTag.dataset.level = lv.key;
    }
    const active = events.filter(([s, e]) => t >= s && t <= e);
    const key = active.map((a) => a.join()).join('|');
    if (key !== activeBox.dataset.key) {
      activeBox.dataset.key = key;
      activeBox.replaceChildren(...(active.length ? active.map((a) => classChip(a[2])) : [h('span.ep-none', {}, 'none')]));
    }
    for (const b of eventButtons) b.classList.toggle('is-active', t >= b._span[0] && t <= b._span[1]);
  }

  function loop() {
    update();
    raf = video.paused || video.ended ? 0 : requestAnimationFrame(loop);
  }
  video.addEventListener('play', () => { if (!raf) raf = requestAnimationFrame(loop); });
  video.addEventListener('seeked', update);
  video.addEventListener('timeupdate', () => { if (!raf) update(); });

  function seek(t) {
    const target = Math.min(duration, Math.max(0, num(t)));
    virtualTime = target;
    if (videoOk && !video.hidden) {
      try { video.currentTime = target; } catch { /* not seekable yet */ }
    }
    update();
  }

  function rebuild() {
    nowBar.querySelector('.ep-dur').textContent = ` / ${fmtTime(duration)}`;
    buildTimeline(); buildChart(); buildList(); update();
  }

  const root = h('div.ep', { id },
    h('div.ep-stage', {}, video, stageMsg),
    nowBar, timeline, riskHost, list);
  host.append(root);
  rebuild();

  return {
    el: root,
    seek,
    video,
    destroy() {
      cancelAnimationFrame(raf);
      video.pause();
      video.removeAttribute('src');
      video.load();
      chart?.destroy();
      root.remove();
    },
  };
}
