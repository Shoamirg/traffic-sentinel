// Live demo: drag & drop upload -> job queue -> progress polling -> the same
// EventPlayer as the Results section, plus JSON downloads. Falls back to an
// "offline" card when no backend answers (e.g. static hosting on GitHub Pages).
import { h, clear, fill, fmtNum, fmtTime, safeUrl, download, cleanEvents, cleanPairs, num } from '../util.js';
import { createEventPlayer } from '../event-player.js';

const ALLOWED = ['.mp4', '.mov', '.avi'];
const POLL_MS = 1000;
const HEALTH_TIMEOUT_MS = 5000;
const MAX_POLL_FAILURES = 8;

const extOf = (name) => (String(name).match(/\.[^.]+$/) || [''])[0].toLowerCase();
const mb = (bytes) => bytes / (1024 * 1024);

async function health(api) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(`${api}/api/health`, { signal: ctl.signal, cache: 'no-store' });
    if (!res.ok) return null;
    const body = await res.json();
    return body && body.ok ? body : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function probeDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement('video');
    v.preload = 'metadata';
    const done = (d) => { URL.revokeObjectURL(url); resolve(d); };
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    setTimeout(() => done(null), 4000);
    v.src = url;
  });
}

function offlineCard(site) {
  const where = site.demo_page || site.demo_api;
  const href = where && safeUrl(where) !== '#' ? where : null;
  return h('div.card.offline', { role: 'status' },
    h('div.offline-lamp', { 'aria-hidden': 'true' }),
    h('div', {},
      h('h3', {}, 'Demo backend is offline here'),
      h('p', {}, 'This copy of the site is served as static files, so there is no analysis server behind it.'),
      href ? h('p', {}, 'The live demo runs at ', h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, where), '.')
        : h('p.muted', {}, 'Run it locally: see website/README.md (one uvicorn command).'),
      h('p.muted.small', {}, 'Everything else on this page - results, timelines, risk curves - works without it.')));
}

export async function renderDemo(host, ctx) {
  const site = ctx.site;
  const api = String(site.demo_api || '').replace(/\/+$/, '');
  const status = h('p.muted', { role: 'status' }, 'Checking the analysis server…');
  host.append(status);
  const info = await health(api);
  status.remove();
  if (!info) { host.append(offlineCard(site)); return; }

  const limits = { mb: num(info.max_mb, 200), seconds: num(info.max_seconds, 120) };
  document.getElementById('demo-limit-mb').textContent = `${fmtNum(limits.mb, 0)} MB`;
  document.getElementById('demo-limit-s').textContent = `${fmtNum(limits.seconds, 0)} s`;
  new DemoFlow(host, api, limits, info.mode);
}

class DemoFlow {
  constructor(host, api, limits, mode) {
    Object.assign(this, { host, api, limits, mode, file: null, localUrl: null, player: null, pollTimer: 0, started: 0 });
    this.root = h('div.demo');
    host.append(this.root);
    this.renderPicker();
  }

  // ---- step 1: choose a file -------------------------------------------------------
  renderPicker(message) {
    this.reset();
    const input = h('input#demo-file.visually-hidden', { type: 'file', accept: 'video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi' });
    const zone = h('label.drop', { for: 'demo-file' },
      h('span.drop-icon', { 'aria-hidden': 'true' }),
      h('span.drop-title', {}, 'Drop a road-camera video here'),
      h('span.drop-sub', {}, `or click to choose · .mp4 .mov .avi · up to ${fmtNum(this.limits.mb, 0)} MB · first ${fmtNum(this.limits.seconds, 0)} s analysed`));
    const err = h('p.form-error', { role: 'alert' }, message || '');
    input.addEventListener('change', () => input.files[0] && this.pick(input.files[0], err));
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('is-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('is-over'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('is-over');
      const f = e.dataTransfer?.files?.[0];
      if (f) this.pick(f, err);
    });
    fill(this.root, input, zone, err,
      this.mode === 'fake' ? h('p.muted.small', {}, 'Server is in development mode (SENTINEL_FAKE=1): results are simulated.') : null);
  }

  validate(file) {
    if (!ALLOWED.includes(extOf(file.name))) return `“${file.name}” is not a supported video. Use .mp4, .mov or .avi.`;
    if (file.size === 0) return 'That file is empty.';
    if (mb(file.size) > this.limits.mb) return `That file is ${fmtNum(mb(file.size), 0)} MB; the limit is ${fmtNum(this.limits.mb, 0)} MB. Trim or re-encode it first.`;
    return null;
  }

  async pick(file, errNode) {
    const problem = this.validate(file);
    if (problem) { errNode.textContent = problem; return; }
    this.file = file;
    const duration = await probeDuration(file);
    const long = duration && duration > this.limits.seconds;
    clear(this.root).append(h('div.card.file-card', {},
      h('div.file-meta', {},
        h('strong.file-name', {}, file.name),
        h('span.muted', {}, `${fmtNum(mb(file.size), 1)} MB${duration ? ` · ${fmtTime(duration, 0)}` : ''}`),
        long ? h('span.warn', {}, `Only the first ${fmtNum(this.limits.seconds, 0)} s will be analysed.`) : null),
      h('div.file-actions', {},
        h('button.btn.btn-primary', { type: 'button', onclick: () => this.upload() }, 'Analyse video'),
        h('button.btn', { type: 'button', onclick: () => this.renderPicker() }, 'Choose another'))));
  }

  // ---- step 2: upload + poll ----------------------------------------------------------
  renderProgress() {
    this.steps = ['Upload', 'Queue', 'Analyse', 'Done'].map((s) => h('li', {}, s));
    this.bar = h('div.progress-fill');
    this.stage = h('p.progress-stage', { role: 'status', 'aria-live': 'polite' }, 'Uploading…');
    this.pct = h('span.progress-pct', {}, '0%');
    this.elapsed = h('span.muted.small', {}, '');
    this.progressEl = h('div.progress', { role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0', 'aria-label': 'Analysis progress' }, this.bar);
    clear(this.root).append(h('div.card.progress-card', {},
      h('ol.stepper', {}, this.steps),
      h('div.progress-row', {}, this.progressEl, this.pct),
      this.stage, this.elapsed,
      h('button.btn.btn-small', { type: 'button', onclick: () => this.renderPicker() }, 'Cancel and choose another')));
  }

  setProgress(stepIdx, frac, text) {
    this.steps.forEach((s, i) => { s.className = i < stepIdx ? 'is-done' : i === stepIdx ? 'is-now' : ''; });
    const p = Math.round(Math.max(0, Math.min(1, frac)) * 100);
    this.bar.style.transform = `scaleX(${p / 100})`;
    this.pct.textContent = `${p}%`;
    this.progressEl.setAttribute('aria-valuenow', String(p));
    this.stage.textContent = text;
    if (this.started) this.elapsed.textContent = `elapsed ${fmtTime((Date.now() - this.started) / 1000, 0)}`;
  }

  upload() {
    this.renderProgress();
    this.started = Date.now();
    const form = new FormData();
    form.append('file', this.file, this.file.name);
    const xhr = new XMLHttpRequest();
    this.xhr = xhr;
    xhr.open('POST', `${this.api}/api/jobs`);
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) this.setProgress(0, e.loaded / e.total, `Uploading ${fmtNum(mb(e.loaded), 1)} / ${fmtNum(mb(e.total), 1)} MB`);
    };
    xhr.onerror = () => this.fail('Upload failed - the connection dropped. Please try again.');
    xhr.onload = () => {
      const body = xhr.response || {};
      if (xhr.status !== 202 && xhr.status !== 200) { this.fail(body.error || `Upload rejected (HTTP ${xhr.status}).`); return; }
      if (!/^[0-9a-f]{32}$/.test(body.job_id || '')) { this.fail('Unexpected server response.'); return; }
      this.jobId = body.job_id;
      this.pollFailures = 0;
      this.setProgress(1, 0, 'Waiting for the analysis worker…');
      this.poll();
    };
    xhr.send(form);
  }

  async poll() {
    const jobId = this.jobId;
    if (!jobId) return;
    try {
      const res = await fetch(`${this.api}/api/jobs/${jobId}`, { cache: 'no-store' });
      const body = await res.json().catch(() => ({}));
      if (jobId !== this.jobId) return; // cancelled while the request was in flight
      if (res.status === 404) { this.fail(body.error || 'This job has expired.'); return; }
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      this.pollFailures = 0;
      if (body.status === 'queued') {
        this.setProgress(1, 0, body.queue_position ? `Queued - position ${body.queue_position}. One video is analysed at a time.` : 'Queued…');
      } else if (body.status === 'running') {
        this.setProgress(2, num(body.progress), body.stage || 'Analysing…');
      } else if (body.status === 'done') {
        this.setProgress(3, 1, 'Done');
        this.showResult(body.result || {});
        return;
      } else if (body.status === 'error') {
        this.fail(body.error || 'Analysis failed.');
        return;
      }
    } catch {
      if (jobId !== this.jobId) return;
      this.pollFailures += 1;
      if (this.pollFailures >= MAX_POLL_FAILURES) { this.fail('Lost contact with the analysis server. Please try again later.'); return; }
    }
    this.pollTimer = setTimeout(() => this.poll(), POLL_MS);
  }

  fail(message) {
    this.jobId = null;
    this.renderPicker(message);
  }

  // ---- step 3: result --------------------------------------------------------------------
  showResult(result) {
    const events = cleanEvents(result.events);
    const risk = cleanPairs(result.risk);
    const peak = risk.length ? Math.max(...risk.map((r) => r[1])) : null;
    this.localUrl = URL.createObjectURL(this.file);
    const base = this.file.name.replace(/\.[^.]+$/, '');
    const payload = { video: this.file.name, duration: result.duration, fps: result.fps, events };
    const playerHost = h('div');
    fill(this.root,
      h('div.result-head', {},
        h('div.tiles', {},
          h('div.tile', {}, h('div.tile-k', {}, 'Events'), h('div.tile-v', {}, String(events.length))),
          h('div.tile', {}, h('div.tile-k', {}, 'Peak risk'), h('div.tile-v', {}, peak === null ? '—' : fmtNum(peak, 2))),
          h('div.tile', {}, h('div.tile-k', {}, 'Analysed'), h('div.tile-v', {}, fmtTime(result.duration, 0))),
          h('div.tile', {}, h('div.tile-k', {}, 'Processing'), h('div.tile-v', {}, result.elapsed_sec ? `${fmtNum(result.elapsed_sec, 0)} s` : '—'))),
        h('div.file-actions', {},
          h('button.btn.btn-primary', { type: 'button', onclick: () => download(`${base}_events.json`, JSON.stringify(payload, null, 1)) }, 'Download events JSON'),
          h('button.btn', { type: 'button', onclick: () => download(`${base}_result.json`, JSON.stringify({ ...payload, risk }, null, 1)) }, 'Full result (with risk)'),
          h('button.btn', { type: 'button', onclick: () => this.renderPicker() }, 'Analyse another'))),
      result.video_url ? null : h('p.muted.small', {}, 'The server returned no annotated video; playing your original file instead.'),
      playerHost);
    this.player = createEventPlayer(playerHost, {
      title: this.file.name,
      video: result.video_url ? `${this.api}${result.video_url}` : null,
      fallbackVideo: this.localUrl,
      events, risk, duration: result.duration,
    });
  }

  reset() {
    clearTimeout(this.pollTimer);
    if (this.xhr && this.xhr.readyState !== 4) this.xhr.abort();
    this.xhr = null;
    this.jobId = null;
    this.started = 0;
    this.player?.destroy();
    this.player = null;
    if (this.localUrl) { URL.revokeObjectURL(this.localUrl); this.localUrl = null; }
  }
}
