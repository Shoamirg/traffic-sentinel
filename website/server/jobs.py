"""In-memory job queue for the live demo.

One background worker runs analyses strictly one at a time (CPU inference);
everything else waits in a FIFO queue. A janitor thread deletes jobs and their
files once they are older than the TTL. Nothing here knows about HTTP.
"""
from __future__ import annotations

import logging
import math
import os
import queue
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("sentinel.jobs")

AnalyzeFn = Callable[..., dict]
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_RISK_POINTS = 4000
MAX_COUNT_POINTS = 2000
MAX_EVENTS = 5000
GENERIC_FAILURE = "Analysis failed: the video could not be processed. Try a different file (H.264 .mp4 works best)."


def is_valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(job_id or ""))


@dataclass
class Job:
    job_id: str
    job_dir: Path
    input_path: Path
    created: float = field(default_factory=time.time)
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    stage: str = "Waiting in queue"
    error: Optional[str] = None
    result: Optional[dict] = None
    video_path: Optional[Path] = None


# ---------------------------------------------------------------- sanitising
def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _downsample(seq: list, limit: int) -> list:
    if len(seq) <= limit:
        return seq
    step = len(seq) / limit
    return [seq[int(i * step)] for i in range(limit)]


def sanitize_result(raw: dict, video_url: Optional[str]) -> dict:
    """Coerce the analyzer output into the exact public schema (no paths leak)."""
    raw = raw if isinstance(raw, dict) else {}
    events = []
    for ev in list(raw.get("events") or [])[:MAX_EVENTS]:
        if not isinstance(ev, (list, tuple)) or len(ev) < 3:
            continue
        s, e, label = ev[0], ev[1], ev[2]
        s, e = _num(s), _num(e)
        if e < s:
            s, e = e, s
        events.append([round(s, 2), round(e, 2), str(label)[:40]])
    events.sort(key=lambda x: (x[0], x[1]))

    risk = []
    for pt in list(raw.get("risk") or []):
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        t, v = pt[0], pt[1]
        risk.append([round(_num(t), 3), round(min(1.0, max(0.0, _num(v))), 4)])
    risk = _downsample(risk, MAX_RISK_POINTS)

    counts: dict[str, list] = {}
    raw_counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}
    for key, values in raw_counts.items():
        if isinstance(values, (list, tuple)):
            counts[str(key)[:30]] = _downsample([round(_num(v), 3) for v in values], MAX_COUNT_POINTS)

    return {
        "events": events,
        "risk": risk,
        "duration": round(_num(raw.get("duration")), 3),
        "fps": round(_num(raw.get("fps")), 3),
        "counts": counts,
        "video_url": video_url,
    }


# ------------------------------------------------------------------ manager
class JobManager:
    def __init__(self, analyze: AnalyzeFn, workdir: Path, ttl_sec: float = 3600.0,
                 max_seconds: float = 120.0, max_pending: int = 20,
                 cleanup_interval: float = 60.0) -> None:
        self._analyze = analyze
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = ttl_sec
        self.max_seconds = max_seconds
        self.max_pending = max_pending
        self._cleanup_interval = cleanup_interval
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []          # queued job ids, FIFO
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        for target, name in ((self._worker, "sentinel-worker"), (self._janitor, "sentinel-janitor")):
            th = threading.Thread(target=target, name=name, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")  # wake the worker

    # public API --------------------------------------------------------
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def new_job_dir(self) -> tuple[str, Path]:
        job_id = secrets.token_hex(16)
        job_dir = self.workdir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        return job_id, job_dir

    def submit(self, job_id: str, job_dir: Path, input_path: Path) -> Job:
        job = Job(job_id=job_id, job_dir=job_dir, input_path=input_path)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            position = None
            if job.status == "queued" and job_id in self._order:
                position = self._order.index(job_id) + 1
            return {
                "status": job.status,
                "progress": round(job.progress, 4),
                "stage": job.stage,
                "queue_position": position,
                "error": job.error,
                "result": job.result,
            }

    # internals ---------------------------------------------------------
    def _set(self, job: Job, **changes: Any) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(job, k, v)

    def _worker(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if not job_id:
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job_id in self._order:
                    self._order.remove(job_id)
            if job is None:          # expired while queued
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        self._set(job, status="running", stage="Starting analysis", progress=0.0)
        out_dir = job.job_dir / "out"
        out_dir.mkdir(exist_ok=True)

        def progress(frac: float, stage: str = "") -> None:
            frac = min(1.0, max(0.0, _num(frac)))
            self._set(job, progress=max(job.progress, frac), stage=str(stage)[:120] or job.stage)

        started = time.time()
        try:
            raw = self._analyze(str(job.input_path), str(out_dir), progress, max_seconds=self.max_seconds)
            video_path = self._resolve_video(out_dir, raw)
            url = f"/api/jobs/{job.job_id}/video" if video_path else None
            result = sanitize_result(raw, url)
            result["elapsed_sec"] = round(time.time() - started, 2)
            self._set(job, status="done", progress=1.0, stage="Done", result=result, video_path=video_path)
            log.info("job %s done in %.1fs (%d events)", job.job_id, time.time() - started, len(result["events"]))
        except Exception:  # noqa: BLE001 - any analyzer failure becomes a job error
            log.exception("job %s failed", job.job_id)
            self._set(job, status="error", stage="Failed", error=GENERIC_FAILURE)

    @staticmethod
    def _resolve_video(out_dir: Path, raw: Any) -> Optional[Path]:
        name = raw.get("video") if isinstance(raw, dict) else None
        if not name:
            return None
        candidate = (out_dir / os.path.basename(str(name))).resolve()
        try:
            candidate.relative_to(out_dir.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _janitor(self) -> None:
        while not self._stop.wait(self._cleanup_interval):
            self.cleanup()

    def cleanup(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        expired: list[Job] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status != "running" and now - job.created > self.ttl_sec:
                    expired.append(self._jobs.pop(job_id))
                    if job_id in self._order:
                        self._order.remove(job_id)
        for job in expired:
            shutil.rmtree(job.job_dir, ignore_errors=True)
        if expired:
            log.info("expired %d job(s)", len(expired))
        return len(expired)
