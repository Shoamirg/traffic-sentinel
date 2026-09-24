// Entry point: load every data file in parallel, then render each section.
// Each section renders independently and failures are contained, so one bad
// JSON file degrades one section instead of the whole page.
import { loadJSON, setPalette, arr, obj } from './util.js';
import { renderHero, renderTeam } from './sections/team.js';
import { renderApproach } from './sections/approach.js';
import { renderEDA } from './sections/eda.js';
import { renderResults } from './sections/results.js';
import { renderDashboard } from './sections/dashboard.js';
import { renderDemo } from './sections/demo.js';
import { renderReport, renderLinks } from './sections/report.js';

function safely(name, fn) {
  try {
    return fn();
  } catch (err) {
    console.error(`[sentinel] section "${name}" failed to render`, err);
    const host = document.getElementById(`${name}-body`);
    if (host) host.textContent = 'This section could not be rendered. See the browser console for details.';
    return null;
  }
}

async function main() {
  const [site, classes, videos, eda, metrics, report] = await Promise.all([
    loadJSON('data/site.json', {}),
    loadJSON('data/classes.json', {}),
    loadJSON('data/videos.json', []),
    loadJSON('data/eda.json', {}),
    loadJSON('data/metrics.json', {}),
    loadJSON('data/report.json', {}),
  ]);
  setPalette(classes);
  const ctx = { site: obj(site), classes: obj(classes), videos: arr(videos), eda: obj(eda), metrics: obj(metrics), report: obj(report) };

  const results = safely('results', () => renderResults(document.getElementById('results-body'), ctx));
  ctx.jumpTo = (videoId, t) => results?.jumpTo(videoId, t);

  safely('hero', () => renderHero(ctx));
  safely('team', () => renderTeam(document.getElementById('team-body'), ctx));
  safely('approach', () => renderApproach(ctx));
  safely('eda', () => renderEDA(document.getElementById('eda-body'), ctx));
  safely('dash', () => renderDashboard(document.getElementById('dash-body'), ctx));
  safely('demo', () => renderDemo(document.getElementById('demo-body'), ctx));
  safely('report', () => renderReport(document.getElementById('report-body'), ctx));
  safely('links', () => renderLinks(document.getElementById('links-body'), ctx));
  observeNav();
}

/** Highlight the nav link of the section in view. */
function observeNav() {
  const links = new Map([...document.querySelectorAll('.nav a')].map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.removeAttribute('aria-current'));
      const a = links.get(e.target.id);
      if (a) {
        a.setAttribute('aria-current', 'true');
        const nav = a.parentElement;
        if (nav.scrollWidth > nav.clientWidth) nav.scrollTo({ left: a.offsetLeft - 24, behavior: 'auto' });
      }
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  document.querySelectorAll('main section[id]').forEach((s) => io.observe(s));
}

main();
