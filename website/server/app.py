"""Traffic Sentinel - public website + live demo API.

Run (from the repo root, i.e. traffic-sentinel/):
    SENTINEL_FAKE=1 uvicorn website.server.app:app --port 8000

Endpoints
    GET  /api/health                  -> limits + mode
    POST /api/jobs                    multipart field "file" -> {"job_id"}
    GET  /api/jobs/{job_id}           -> status / progress / result
    GET  /api/jobs/{job_id}/video     -> annotated mp4 (HTTP Range supported)
    GET  /                            -> static site (website/static)

Environment
    SENTINEL_FAKE=1           use fake_analyze instead of sentinel.demo.analyze
    SENTINEL_WORKDIR          where uploads/outputs live (default: a temp dir)
    SENTINEL_MAX_MB           upload limit in MB (default 200)
    SENTINEL_MAX_SECONDS      seconds of video analysed (default 120)
    SENTINEL_JOB_TTL          seconds before a job and its files are deleted (default 3600)
    SENTINEL_CORS_ORIGINS     comma-separated origins allowed to call the API (default "*")
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

try:  # works both as "website.server.app" and as a plain script directory
    from .jobs import JobManager, is_valid_job_id
except ImportError:  # pragma: no cover
    from jobs import JobManager, is_valid_job_id  # type: ignore

log = logging.getLogger("sentinel.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SERVER_DIR = Path(__file__).resolve().parent
WEBSITE_DIR = SERVER_DIR.parent
REPO_ROOT = WEBSITE_DIR.parent
STATIC_DIR = WEBSITE_DIR / "static"

ALLOWED_EXT = {".mp4", ".mov", ".avi"}
MAX_CONCURRENT_UPLOADS = 3
CHUNK = 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ------------------------------------------------------------- analyzer pick
def load_analyzer() -> tuple[Callable[..., dict], str]:
    """Return (analyze_fn, mode). The real module is imported lazily per call so
    the site still starts (and says so) when model deps are missing."""
    if os.environ.get("SENTINEL_FAKE", "").strip() in ("1", "true", "yes"):
        try:
            from .fake_analyze import analyze as fake
        except ImportError:  # pragma: no cover
            from fake_analyze import analyze as fake  # type: ignore
        return fake, "fake"

    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    def real(video_path: str, out_dir: str, progress, max_seconds: float = 120.0) -> dict:
        from sentinel.demo import analyze  # imported on first job (heavy: torch, ultralytics)
        return analyze(video_path, out_dir, progress, max_seconds=max_seconds)

    return real, "real"


# ------------------------------------------------------------ upload helpers
class BodyTooLarge(Exception):
    pass


def _sniff_container(head: bytes) -> bool:
    """Cheap magic-byte check: ISO-BMFF (mp4/mov) or RIFF AVI."""
    if len(head) >= 12 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip"):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"AVI "


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


# ------------------------------------------------------------- range serving
def _range_response(path: Path, range_header: Optional[str]) -> Response:
    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    start, end = 0, size - 1
    status = 200
    if range_header and range_header.startswith("bytes="):
        spec = range_header[6:].split(",")[0].strip()
        try:
            a, b = spec.split("-", 1)
            if a == "":                      # suffix range: last N bytes
                n = int(b)
                start, end = max(0, size - n), size - 1
            else:
                start = int(a)
                end = int(b) if b else size - 1
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if start >= size or start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        end = min(end, size - 1)
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = end - start + 1
    headers["Content-Length"] = str(length)

    def body():
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=status, media_type="video/mp4", headers=headers)


class RangeStaticFiles(StaticFiles):
    """StaticFiles that honours HTTP Range for media, so <video> can seek the
    sample videos (Starlette < 0.39 FileResponse ignores Range)."""

    def file_response(self, full_path, stat_result, scope, status_code=200):  # type: ignore[override]
        headers = dict((k.decode("latin-1").lower(), v.decode("latin-1")) for k, v in scope.get("headers", []))
        path = Path(full_path)
        if path.suffix.lower() in {".mp4", ".webm", ".mov"} and status_code == 200:
            response = _range_response(path, headers.get("range"))
            response.headers["Cache-Control"] = "public, max-age=3600"
            return response
        return super().file_response(full_path, stat_result, scope, status_code)


# ---------------------------------------------------------------- app factory
def create_app(analyze: Optional[Callable[..., dict]] = None, mode: str = "custom",
               workdir: Optional[Path] = None, serve_static: bool = True,
               start_worker: bool = True) -> FastAPI:
    if analyze is None:
        analyze, mode = load_analyzer()
    max_mb = _env_float("SENTINEL_MAX_MB", 200)
    max_bytes = int(max_mb * 1024 * 1024)
    max_seconds = _env_float("SENTINEL_MAX_SECONDS", 120)
    workdir = Path(workdir or os.environ.get("SENTINEL_WORKDIR") or tempfile.mkdtemp(prefix="sentinel_jobs_"))

    manager = JobManager(analyze, workdir, ttl_sec=_env_float("SENTINEL_JOB_TTL", 3600),
                         max_seconds=max_seconds)
    uploads = threading.BoundedSemaphore(MAX_CONCURRENT_UPLOADS)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            manager.start()
            log.info("analysis mode=%s", mode)
        yield
        manager.stop()

    app = FastAPI(title="Traffic Sentinel", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.jobs = manager
    origins = [o.strip() for o in os.environ.get("SENTINEL_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], allow_credentials=False,
                       expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"])

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "mode": mode, "max_mb": max_mb, "max_seconds": max_seconds,
                "allowed": sorted(ALLOWED_EXT), "pending": manager.pending_count()}

    @app.post("/api/jobs")
    async def create_job(request: Request):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes + MULTIPART_OVERHEAD:
            return _error(413, f"File too large: the limit is {max_mb:.0f} MB.")
        if manager.pending_count() >= manager.max_pending:
            return _error(503, "The demo is busy right now. Please try again in a few minutes.")
        if not uploads.acquire(blocking=False):
            return _error(429, "Too many uploads in progress. Please retry shortly.")
        try:
            return await _receive_upload(request)
        finally:
            uploads.release()

    async def _receive_upload(request: Request):
        received = 0
        original_receive = request._receive  # count bytes as they stream in

        async def counting_receive():
            nonlocal received
            message = await original_receive()
            received += len(message.get("body", b""))
            if received > max_bytes + MULTIPART_OVERHEAD:
                raise BodyTooLarge()
            return message

        request._receive = counting_receive
        try:
            form = await request.form(max_files=1, max_fields=5)
        except BodyTooLarge:
            return _error(413, f"File too large: the limit is {max_mb:.0f} MB.")
        except Exception:  # noqa: BLE001 - malformed multipart
            return _error(400, "Malformed upload. Send multipart/form-data with a 'file' field.")

        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return _error(400, "Missing 'file' field.")
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            await upload.close()
            return _error(415, "Unsupported file type. Upload an .mp4, .mov or .avi video.")

        job_id, job_dir = manager.new_job_dir()
        target = job_dir / f"input{ext}"
        size = 0
        head = b""
        try:
            with open(target, "wb") as fh:
                while True:
                    chunk = await upload.read(CHUNK)
                    if not chunk:
                        break
                    if len(head) < 16:
                        head += chunk[:16]
                    size += len(chunk)
                    if size > max_bytes:
                        raise BodyTooLarge()
                    await run_in_threadpool(fh.write, chunk)
        except BodyTooLarge:
            _discard(job_dir)
            return _error(413, f"File too large: the limit is {max_mb:.0f} MB.")
        finally:
            await upload.close()

        if size == 0:
            _discard(job_dir)
            return _error(400, "The uploaded file is empty.")
        if not _sniff_container(head):
            _discard(job_dir)
            return _error(415, "That file does not look like a video container (mp4/mov/avi).")

        manager.submit(job_id, job_dir, target)
        log.info("job %s queued (%.1f MB)", job_id, size / 1e6)
        return JSONResponse(status_code=202, content={"job_id": job_id})

    def _discard(job_dir: Path) -> None:
        shutil.rmtree(job_dir, ignore_errors=True)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        if not is_valid_job_id(job_id):
            return _error(400, "Invalid job id.")
        snap = manager.snapshot(job_id)
        if snap is None:
            return _error(404, "Job not found (jobs expire after one hour).")
        return snap

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str, request: Request):
        if not is_valid_job_id(job_id):
            return _error(400, "Invalid job id.")
        job = manager.get(job_id)
        if job is None:
            return _error(404, "Job not found (jobs expire after one hour).")
        if job.status != "done" or job.video_path is None or not job.video_path.is_file():
            return _error(404, "No annotated video for this job.")
        return _range_response(job.video_path, request.headers.get("range"))

    if serve_static and STATIC_DIR.is_dir():
        app.mount("/", RangeStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
