// Hero (tagline, system board, event feed) and Team cards.
import { h, clear, arr, obj, num, fmtNum, fmtTime, classColor, pretty } from '../util.js';

const SEVERITY = ['accident', 'fire_smoke', 'near_miss', 'wrong_way', 'red_light', 'failure_to_yield', 'jaywalking'];

export function renderHero(ctx) {
  const { site, metrics, videos } = ctx;
  if (site.tagline) document.getElementById('hero-tagline').textContent = site.tagline;
  if (site.event) document.getElementById('hero-event').textContent = site.event;
  if (site.team_name) document.getElementById('footer-team').textContent = site.team_name;

  const dev = obj(metrics.dev_set);
  const runtime = arr(metrics.runtime);
  const xrt = runtime.length ? runtime.reduce((a, r) => a + num(r.x_realtime), 0) / runtime.length : null;
  const totalEvents = videos.reduce((a, v) => a + arr(v.events).length, 0);
  const stats = [
    ['Score A · events', dev.score_a !== undefined ? fmtNum(dev.score_a, 3) : '—'],
    ['Score B · risk', dev.score_b != null ? fmtNum(dev.score_b, 3) : 'not scored'],
    ['Event classes', '9 of 14 output'],
    ['Time per video', xrt ? `${fmtNum(xrt, 2)}× its length` : '—'],
    ['Sample videos', String(videos.length)],
    ['Events found', String(totalEvents)],
  ];
  const dl = clear(document.getElementById('hero-stats'));
  for (const [k, v] of stats) dl.append(h('div.stat', {}, h('dt', {}, k), h('dd', {}, v)));
  if (metrics.placeholder) dl.append(h('p.board-note', {}, 'placeholder numbers'));

  // A feed of the most severe events across all samples - each is a jump link.
  const all = [];
  for (const v of videos) for (const e of arr(v.events)) if (Array.isArray(e) && e.length >= 3) all.push({ v, s: num(e[0]), label: String(e[2]) });
  const rank = (l) => { const i = SEVERITY.indexOf(l); return i < 0 ? 99 : i; };
  all.sort((a, b) => rank(a.label) - rank(b.label) || a.s - b.s);
  const feed = clear(document.getElementById('hero-feed'));
  for (const it of all.slice(0, 6)) {
    feed.append(h('li', {}, h('a.feed-item', {
      href: '#results', style: { '--c': classColor(it.label) },
      onclick: (e) => { e.preventDefault(); ctx.jumpTo(it.v.id, it.s); },
    }, h('span.feed-t', {}, fmtTime(it.s)), h('span.feed-l', {}, pretty(it.label)), h('span.feed-v', {}, it.v.id))));
  }
  if (!all.length) feed.append(h('li.feed-empty', {}, 'No events yet.'));
}

function link(href, text, label) {
  if (!href) return null;
  return h('a.pill-link', { href, target: '_blank', rel: 'noopener noreferrer', 'aria-label': label }, text);
}

function initials(name) {
  return String(name || '?').split(/\s+/).filter((w) => w && !/^TODO$/i.test(w)).slice(0, 2).map((w) => w[0]).join('').toUpperCase() || '?';
}

export function renderTeam(host, ctx) {
  const members = arr(ctx.site.members);
  if (!members.length) { host.append(h('p.muted', {}, 'Team details coming soon.')); return; }
  members.forEach((m, i) => {
    const mm = obj(m);
    host.append(h('article.member', { style: { '--i': i } },
      h('div.member-top', {},
        h('div.avatar', { 'aria-hidden': 'true' }, mm.photo ? h('img', { src: mm.photo, alt: '', width: 64, height: 64, loading: 'lazy' }) : initials(mm.name)),
        h('div', {}, h('h3.member-name', {}, mm.name || 'Team member'), h('p.member-role', {}, mm.role || ''))),
      arr(mm.did).length ? h('div.member-block', {}, h('h4', {}, 'Built'), h('ul.did', {}, arr(mm.did).map((d) => h('li', {}, d)))) : null,
      arr(mm.projects).length ? h('div.member-block', {}, h('h4', {}, 'Previous projects'),
        h('ul.projects', {}, arr(mm.projects).map((p) => h('li', {},
          p.url ? h('a', { href: p.url, target: '_blank', rel: 'noopener noreferrer' }, p.title || p.url) : h('strong', {}, p.title || ''),
          p.blurb ? h('span.muted', {}, ` - ${p.blurb}`) : null)))) : null,
      h('div.member-links', {},
        link(mm.github, 'GitHub', `${mm.name} on GitHub`),
        link(mm.linkedin, 'LinkedIn', `${mm.name} on LinkedIn`),
        link(mm.portfolio, 'Portfolio', `${mm.name}'s portfolio`))));
  });
}
