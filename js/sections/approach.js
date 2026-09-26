// Approach: inline-SVG pipeline diagram (horizontal on desktop, vertical on
// phones - same graph data, re-laid out) and the per-class rulebook.
import { h, clear, classList, classColor, classRule, classStatus, pretty } from '../util.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const TAG = { learned: 'LEARNED', rule: 'RULES', manual: 'HAND-DRAWN', io: 'DATA' };

// col = pipeline stage (left->right), row = parallel track. Main flow on row 1.
const PART_A = {
  title: 'PART A · offline event detection',
  nodes: [
    { id: 'clip', col: 0, row: 1, kind: 'io', lines: ['CCTV clip', 'whole video'] },
    { id: 'sample', col: 1, row: 1, kind: 'rule', lines: ['Frame sampler', '~10 fps · 1280 px'] },
    { id: 'yolo', col: 2, row: 1, kind: 'learned', lines: ['YOLO11m', 'COCO-pretrained'] },
    { id: 'fire', col: 2, row: 3, kind: 'learned', lines: ['Fire / smoke', 'fine-tuned det.'] },
    { id: 'track', col: 3, row: 1, kind: 'rule', lines: ['Tracker', 'ByteTrack-style'] },
    { id: 'traj', col: 4, row: 1, kind: 'rule', lines: ['Trajectories', 'smoothed 0.6 s'] },
    { id: 'scene', col: 5, row: 0, kind: 'learned', lines: ['Scene model', 'directions · road'] },
    { id: 'layout', col: 5, row: 2, kind: 'manual', lines: ['Layout', '+ signal (Otsu)'] },
    { id: 'rules', col: 6, row: 1, kind: 'rule', lines: ['14 class rules', 'thresholds'] },
    { id: 'seg', col: 7, row: 1, kind: 'rule', lines: ['Segments', 'merge · min length'] },
    { id: 'out', col: 8, row: 1, kind: 'io', lines: ['[s, e, label]', 'events.json'] },
  ],
  edges: [['clip', 'sample'], ['sample', 'yolo'], ['sample', 'fire'], ['yolo', 'track'], ['track', 'traj'],
    ['traj', 'scene'], ['traj', 'rules'], ['scene', 'rules'], ['layout', 'rules'],
    ['fire', 'rules'], ['rules', 'seg'], ['seg', 'out']],
};

const PART_B = {
  title: 'PART B · causal risk (never sees the future)',
  nodes: [
    { id: 'frames', col: 0, row: 0, kind: 'io', lines: ['Frames ≤ now', '~5 fps'] },
    { id: 'online', col: 1, row: 0, kind: 'learned', lines: ['Detector +', 'online tracker'] },
    { id: 'hazard', col: 2, row: 0, kind: 'rule', lines: ['TTC + braking', 'hazard'] },
    { id: 'calib', col: 3, row: 0, kind: 'learned', lines: ['Logistic', 'calibration'] },
    { id: 'smooth', col: 4, row: 0, kind: 'rule', lines: ['Fast attack /', 'slow decay'] },
    { id: 'risk', col: 5, row: 0, kind: 'io', lines: ['risk ∈ [0, 1]', 'alarm ≥ 0.5'] },
  ],
  edges: [['frames', 'online'], ['online', 'hazard'], ['hazard', 'calib'], ['calib', 'smooth'], ['smooth', 'risk']],
};

function el(name, attrs = {}, ...kids) {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  kids.forEach((k) => n.append(k));
  return n;
}

function drawGraph(graph, vertical) {
  const G = vertical
    ? { w: 88, h: 60, gx: 8, gy: 24, font: 12 }
    : { w: 114, h: 62, gx: 22, gy: 14, font: 12 };
  const cols = Math.max(...graph.nodes.map((n) => n.col)) + 1;
  const rows = Math.max(...graph.nodes.map((n) => n.row)) + 1;
  // In vertical mode stages go top->bottom and parallel tracks left->right.
  const pos = (n) => (vertical
    ? { x: n.row * (G.w + G.gx), y: n.col * (G.h + G.gy) }
    : { x: n.col * (G.w + G.gx), y: n.row * (G.h + G.gy) });
  const W = vertical ? rows * (G.w + G.gx) - G.gx : cols * (G.w + G.gx) - G.gx;
  const H = vertical ? cols * (G.h + G.gy) - G.gy : rows * (G.h + G.gy) - G.gy;
  const pad = 4;
  const mid = `arrow-${graph.nodes[0].id}-${vertical ? 'v' : 'h'}`;
  const svg = el('svg', { viewBox: `${-pad} ${-pad} ${W + 2 * pad} ${H + 2 * pad}`, class: 'pipe', role: 'img', 'aria-label': graph.title, style: `max-width:${W + 2 * pad}px` });
  svg.append(el('defs', {}, el('marker', { id: mid, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    el('path', { d: 'M0,0 L8,4 L0,8 z', class: 'pipe-arrowhead' }))));

  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const edges = el('g', { class: 'pipe-edges' });
  for (const [a, b] of graph.edges) {
    const pa = pos(byId.get(a)), pb = pos(byId.get(b));
    let d;
    if (vertical) {
      const x1 = pa.x + G.w / 2, y1 = pa.y + G.h, x2 = pb.x + G.w / 2, y2 = pb.y;
      const my = y2 - G.gy / 2;
      d = `M${x1},${y1} L${x1},${my} L${x2},${my} L${x2},${y2 - 1}`;
    } else {
      const x1 = pa.x + G.w, y1 = pa.y + G.h / 2, x2 = pb.x, y2 = pb.y + G.h / 2;
      const mx = x2 - G.gx / 2;
      d = `M${x1},${y1} L${mx},${y1} L${mx},${y2} L${x2 - 1},${y2}`;
    }
    edges.append(el('path', { d, class: 'pipe-edge', 'marker-end': `url(#${mid})` }));
  }
  svg.append(edges);

  for (const n of graph.nodes) {
    const p = pos(n);
    const g = el('g', { class: `pipe-node pipe-${n.kind}`, transform: `translate(${p.x},${p.y})` });
    g.append(el('rect', { width: G.w, height: G.h, rx: 6, class: 'pipe-box' }));
    const tag = el('text', { x: 8, y: 14, class: 'pipe-tag' });
    tag.textContent = TAG[n.kind];
    g.append(tag);
    n.lines.forEach((line, i) => {
      const t = el('text', { x: 8, y: 32 + i * (G.font + 4), class: i === 0 ? 'pipe-title' : 'pipe-sub', 'font-size': i === 0 ? G.font + 1 : G.font - 1 });
      t.textContent = line;
      g.append(t);
    });
    svg.append(g);
  }
  return svg;
}

function renderPipeline(host) {
  const mq = window.matchMedia('(max-width: 760px)');
  const draw = () => {
    clear(host);
    for (const graph of [PART_A, PART_B]) {
      host.append(h('div.pipe-block', {}, h('div.pipe-label', {}, graph.title), drawGraph(graph, mq.matches)));
    }
  };
  draw();
  mq.addEventListener('change', draw);
}

function renderRulebook(host) {
  const ids = classList();
  if (!ids.length) { host.append(h('p.muted', {}, 'Rulebook unavailable.')); return; }
  for (const id of ids) {
    host.append(h('article.rule', { style: { '--c': classColor(id) } },
      h('h4', {}, h('span.chip-dot', { 'aria-hidden': 'true' }), pretty(id)),
      h('p', {}, classRule(id)),
      classStatus(id) ? h('p.rule-status', {}, classStatus(id)) : null));
  }
}

export function renderApproach() {
  renderPipeline(document.getElementById('pipeline-svg'));
  renderRulebook(document.getElementById('rulebook'));
}
