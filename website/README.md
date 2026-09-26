# Traffic Sentinel - website and live demo

A single-page site (plain HTML/CSS/JS, no build step, no JS libraries) plus a small
FastAPI server that runs the live demo. All page content comes from JSON files in
`static/data/`, so updating the site means replacing those files.

```
website/
  Dockerfile               Hugging Face Docker Space (port 7860). Build context = repo root
  server/
    app.py                 FastAPI app: static site + /api/jobs
    jobs.py                job queue: one worker thread, 1 h expiry, result sanitising
    fake_analyze.py        stand-in for sentinel.demo.analyze (SENTINEL_FAKE=1)
    requirements.txt
    tests/test_api.py      pytest: lifecycle, queueing, validation, range requests, expiry
  static/
    index.html             every section: Team, Approach, EDA, Results, Dashboard, Demo, Report, Links
    css/site.css
    js/main.js             loads data/*.json, renders each section on its own
    js/event-player.js     the one video + timeline + risk-curve widget (Results and Demo both use it)
    js/chart.js            canvas line chart with hover, playhead and click/drag/keyboard seeking
    js/util.js             DOM helper, formatting, class colours
    js/sections/*.js       one module per section
    data/*.json            content (placeholders for now, see below)
    media/                 annotated sample videos, posters, EDA images
```

## Run locally

From the repo root (`traffic-sentinel/`):

```bash
pip install -r website/server/requirements.txt

# development: simulated analysis, no torch needed
SENTINEL_FAKE=1 uvicorn website.server.app:app --port 8000
# PowerShell: $env:SENTINEL_FAKE="1"; uvicorn website.server.app:app --port 8000

# real pipeline (needs the root requirements.txt and src/sentinel/demo.py)
uvicorn website.server.app:app --port 8000
```

Open http://localhost:8000. Tests: `pytest website/server/tests -q`.

Static only (no demo): `python -m http.server -d website/static 8080`. The Demo section
detects that `/api/health` is missing and shows an "offline" card that links to `demo_page`.

### Server settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `SENTINEL_FAKE` | unset | `1` = use `fake_analyze.py` |
| `SENTINEL_FAKE_DELAY` | `1.0` | how slow the fake analysis is (0 = instant) |
| `SENTINEL_MAX_MB` | `200` (Dockerfile: `500`) | upload limit |
| `SENTINEL_MAX_SECONDS` | `120` (Dockerfile: `30`) | seconds of video passed to `analyze` as `max_seconds` |
| `SENTINEL_JOB_TTL` | `3600` | jobs and their files are deleted after this many seconds |
| `SENTINEL_WORKDIR` | temp dir | where uploads and outputs are written |
| `SENTINEL_CORS_ORIGINS` | `*` | origins allowed to call the API (for a static site hosted elsewhere) |

### API

| Method | Path | Result |
|---|---|---|
| GET | `/api/health` | `{"ok", "mode", "max_mb", "max_seconds", "allowed", "pending"}` |
| POST | `/api/jobs` | multipart field `file` (.mp4/.mov/.avi, max 200 MB) → `202 {"job_id"}`; errors are `{"error": "..."}` with 400/413/415/429/503 |
| GET | `/api/jobs/{job_id}` | `{"status": queued/running/done/error, "progress", "stage", "queue_position", "error", "result"}` |
| GET | `/api/jobs/{job_id}/video` | annotated mp4, supports `Range` |

`result` = `{"events": [[s, e, label]], "risk": [[t, score]], "duration", "fps", "counts", "video_url", "elapsed_sec"}`.
The server re-validates everything `analyze` returns (numbers finite, risk clamped to [0,1],
video file must sit inside the job's output folder) and never returns file paths or exception text.

Safeguards: 32-hex job ids, extension and magic-byte check, size limit enforced while the body
streams in, max 3 uploads at once, max 20 pending jobs, one analysis at a time,
hourly expiry. Run it with **one** uvicorn worker - the queue lives in memory.

## Contract with `sentinel.demo.analyze` (to be provided by the pipeline)

`src/sentinel/demo.py` must define:

```python
def analyze(video_path: str, out_dir: str,
            progress: Callable[[float, str], None],
            max_seconds: float = 120.0) -> dict:
    ...
    return {
        "events": [[start_sec, end_sec, label], ...],
        "risk": [[t_sec, score], ...],
        "duration": float,          # seconds actually analysed
        "fps": float,
        "video": "annotated.mp4",   # file name inside out_dir
        "counts": {"t": [...], "person": [...], "car": [...], ...},
    }
```

- Call `progress(fraction_0_to_1, "stage text")` often (every few %). The stage text is shown to the user.
- **The annotated video must be H.264 + yuv420p mp4**, or browsers will not play it. OpenCV's
  `mp4v` output does not play in Chrome/Firefox. Write frames with OpenCV, then re-encode:
  `ffmpeg -y -i raw.mp4 -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 28 -movflags +faststart annotated.mp4`
  (ffmpeg is installed in the Docker image). If no video comes back, the demo plays the user's original file.
- Only analyse the first `max_seconds` seconds.
- Raise an exception on unreadable input; the user sees a generic message and the traceback is logged.

## Data files (all in `static/data/`)

Every file currently has `"placeholder": true` (for `videos.json`, on each item) and the page
shows a yellow "Placeholder data" note wherever that flag is set. Remove the flag when the
real file is in place. The code tolerates missing keys and empty arrays.

| File | Schema | Used by |
|---|---|---|
| `site.json` | `team_name, tagline, event, demo_api, demo_page, members[{name, role, did[], github, linkedin, portfolio, photo?, projects[{title, url, blurb}]}], links{repo, weights, predictions}` | Hero, Team, Demo, Links |
| `videos.json` | `[{id, title, duration, fps, width, height, video, poster, events[[s,e,label]], risk[[t,v]], counts{t[], person[], car[], bus[], truck[], motorcycle[], bicycle[]}}]` | Results, Dashboard, Hero feed |
| `eda.json` | `{videos[{id, resolution[w,h], fps, duration, size_mb, brightness{t[],v[]}, counts{...}, heatmap, trajectories, directions}], findings[{title,text}], totals{class: int}}` | EDA |
| `metrics.json` | `{dev_set{score_a, score_b, model_score, per_class{cls{f1@0.3, f1@0.5, f1@0.7, tp, fp, fn}}}, ablations[{name, score_a, notes}], runtime[{video, duration, seconds, x_realtime}], failures[{title, text, video, t}]}` | Results/Evaluation, Hero |
| `report.json` | `{summary, worked[], didnt[], next[]}` - any "TODO" is highlighted | Report |
| `classes.json` | `{classes[{id, color, rule}], objects{name: color}}` - class colours and the rulebook text | everywhere |

`demo_api` = base URL of the API ("" = same origin). Set it to the Space URL when the static
site is hosted elsewhere (e.g. GitHub Pages); `demo_page` is the link shown when the API is offline.
Ablation deltas are computed against the **first** ablation row, so keep "Full system" first.
`failures[].video` must match a `videos.json` id to get a "Watch" button.

Media: `media/<id>_annotated.mp4` (H.264, `+faststart`), `media/<id>.jpg` poster,
`media/<id>_heatmap.jpg`, `media/<id>_traj.jpg`, `media/<id>_dirs.jpg` (16:9 works best).
The current media files are generated test patterns stamped "PLACEHOLDER".

## Deploy

**Hugging Face Docker Space.** A Space needs `Dockerfile` at its repo root, so either push the
whole `traffic-sentinel/` repo to the Space and copy `website/Dockerfile` to `./Dockerfile`, or
build it yourself with `docker build -f website/Dockerfile -t traffic-sentinel .` from the repo root.
Add `app_port: 7860` and `sdk: docker` to the Space README front-matter. Weights must be in
`weights/` (Git LFS on the Space) - the container does not download them. Consider a root
`.dockerignore` excluding `.git`, raw sample videos and caches to keep the image small.

**GitHub Pages (static only).** Publish `website/static/` as-is; set `demo_api` in `site.json` to the
Space URL so the demo still works (CORS is open by default), or leave it empty to show the offline card.
