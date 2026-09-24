// One-page report (worked / did not / next) and the Links grid.
import { h, arr, obj, safeUrl, placeholderBanner } from '../util.js';

function column(title, items, kind) {
  return h(`article.report-col.report-${kind}`, {},
    h('h3', {}, title),
    items.length ? h('ul', {}, items.map((t) => h('li', {}, marked(t)))) : h('p.muted', {}, 'To be written.'));
}

/** Highlight "TODO" so unfinished text is impossible to miss. */
function marked(text) {
  const parts = String(text).split(/(TODO:?)/);
  return parts.map((p) => (/^TODO:?$/.test(p) ? h('span.todo', {}, 'TODO') : p));
}

export function renderReport(host, ctx) {
  const r = ctx.report;
  host.append(placeholderBanner(r.placeholder, 'The report text'));
  if (r.summary) host.append(h('p.report-summary', {}, marked(r.summary)));
  host.append(h('div.report-cols', {},
    column('What worked', arr(r.worked), 'good'),
    column('What did not', arr(r.didnt), 'bad'),
    column('Next steps', arr(r.next), 'next')));
}

export function renderLinks(host, ctx) {
  const links = obj(ctx.site.links);
  const items = [
    ['Source code', links.repo, 'GitHub repository - pipeline, evaluation, this website'],
    ['Model weights', links.weights, 'YOLO11m + fine-tuned fire/smoke detector'],
    ['predictions_samples.json', links.predictions, 'Our Part A + Part B output on every sample video'],
    ['Live demo', ctx.site.demo_page, 'Hosted analysis server (Hugging Face Space)'],
  ];
  for (const [title, url, note] of items) {
    const ok = url && safeUrl(url) !== '#' && !/TODO/.test(url);
    host.append(ok
      ? h('a.link-card', { href: url, target: '_blank', rel: 'noopener noreferrer' }, h('span.link-title', {}, title), h('span.link-note', {}, note), h('span.link-url', {}, url))
      : h('div.link-card.is-pending', {}, h('span.link-title', {}, title), h('span.link-note', {}, note), h('span.link-url', {}, h('span.todo', {}, 'TODO'), ' link not published yet')));
  }
}
