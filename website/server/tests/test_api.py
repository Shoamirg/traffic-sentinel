"""API tests for the live-demo server (run from the repo root: pytest website/server/tests)."""
from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SENTINEL_FAKE", "1")
os.environ["SENTINEL_FAKE_DELAY"] = "0"

from website.server.app import create_app  # noqa: E402
from website.server.fake_analyze import analyze as fake_analyze  # noqa: E402
from website.server.jobs import sanitize_result  # noqa: E402

# Smallest thing that passes the container sniff: an ISO-BMFF "ftyp" box + padding.
FAKE_MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 2048


def _wait(client: TestClient, job_id: str, want: str = "done", timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job never reached {want}: {body}")


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(analyze=fake_analyze, mode="fake", workdir=tmp_path, serve_static=False)
    with TestClient(app) as c:
        yield c


def test_health_reports_limits(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["max_mb"] == 200
    assert body["max_seconds"] == 120


def test_full_job_lifecycle_and_ranged_video(client):
    res = client.post("/api/jobs", files={"file": ("clip.mp4", FAKE_MP4, "video/mp4")})
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    body = _wait(client, job_id)
    assert body["progress"] == 1.0
    result = body["result"]
    assert set(result) >= {"events", "risk", "duration", "fps", "counts", "video_url"}
    assert all(len(ev) == 3 and ev[0] <= ev[1] for ev in result["events"])
    assert all(0.0 <= v <= 1.0 for _, v in result["risk"])
    assert result["video_url"] == f"/api/jobs/{job_id}/video"
    assert "\\" not in str(result) and str(Path.home()) not in str(result)  # no paths leak

    full = client.get(result["video_url"])
    assert full.status_code == 200 and full.content == FAKE_MP4
    part = client.get(result["video_url"], headers={"Range": "bytes=4-11"})
    assert part.status_code == 206
    assert part.content == b"ftypisom"
    assert part.headers["content-range"] == f"bytes 4-11/{len(FAKE_MP4)}"


@pytest.mark.parametrize("name,data,status", [
    ("notes.txt", FAKE_MP4, 415),            # wrong extension
    ("clip.mp4", b"hello world, not a video" * 10, 415),  # wrong magic bytes
    ("clip.mp4", b"", 400),                  # empty
])
def test_rejects_bad_uploads(client, name, data, status):
    res = client.post("/api/jobs", files={"file": (name, data, "application/octet-stream")})
    assert res.status_code == status
    assert "error" in res.json()


def test_rejects_oversize_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_MAX_MB", "0.001")  # ~1 KB
    app = create_app(analyze=fake_analyze, mode="fake", workdir=tmp_path, serve_static=False)
    with TestClient(app) as c:
        res = c.post("/api/jobs", files={"file": ("clip.mp4", FAKE_MP4 * 50, "video/mp4")})
    assert res.status_code == 413
    assert "too large" in res.json()["error"]


def test_invalid_and_unknown_job_ids(client):
    assert client.get("/api/jobs/../../etc").status_code in (400, 404)
    assert client.get("/api/jobs/not-a-job").status_code == 400
    assert client.get("/api/jobs/" + "a" * 32).status_code == 404
    assert client.get("/api/jobs/" + "a" * 32 + "/video").status_code == 404


def test_single_worker_queues_second_job(tmp_path):
    gate = threading.Event()

    def slow(video_path, out_dir, progress, max_seconds=120.0):
        progress(0.5, "Thinking")
        gate.wait(10)
        shutil.copyfile(video_path, os.path.join(out_dir, "a.mp4"))
        return {"events": [[1, 2, "congestion"]], "risk": [[0, 0.1]], "duration": 3, "fps": 25,
                "video": "a.mp4", "counts": {"t": [0], "car": [3]}}

    app = create_app(analyze=slow, mode="test", workdir=tmp_path, serve_static=False)
    with TestClient(app) as c:
        first = c.post("/api/jobs", files={"file": ("a.mp4", FAKE_MP4, "video/mp4")}).json()["job_id"]
        _wait(c, first, "running")
        second = c.post("/api/jobs", files={"file": ("b.mp4", FAKE_MP4, "video/mp4")}).json()["job_id"]
        snap = c.get(f"/api/jobs/{second}").json()
        assert snap["status"] == "queued" and snap["queue_position"] == 1
        gate.set()
        _wait(c, first)
        _wait(c, second)


def test_analyzer_failure_is_reported_without_details(tmp_path):
    def boom(video_path, out_dir, progress, max_seconds=120.0):
        raise RuntimeError(f"secret path {video_path}")

    app = create_app(analyze=boom, mode="test", workdir=tmp_path, serve_static=False)
    with TestClient(app) as c:
        job_id = c.post("/api/jobs", files={"file": ("a.mp4", FAKE_MP4, "video/mp4")}).json()["job_id"]
        body = _wait(c, job_id, "error")
    assert "secret" not in body["error"] and str(tmp_path) not in body["error"]


def test_expired_jobs_are_deleted(tmp_path):
    app = create_app(analyze=fake_analyze, mode="fake", workdir=tmp_path, serve_static=False)
    with TestClient(app) as c:
        job_id = c.post("/api/jobs", files={"file": ("a.mp4", FAKE_MP4, "video/mp4")}).json()["job_id"]
        _wait(c, job_id)
        manager = app.state.jobs
        assert manager.cleanup(now=time.time() + 7200) == 1
        assert not (tmp_path / job_id).exists()
        assert c.get(f"/api/jobs/{job_id}").status_code == 404


def test_sanitize_result_coerces_bad_values():
    out = sanitize_result({"events": [[5, 2, "x"], "junk", [1]], "risk": [[0, 7], [1, float("nan")]],
                           "counts": {"car": [1, "2", None]}, "duration": "12"}, None)
    assert out["events"] == [[2.0, 5.0, "x"]]
    assert out["risk"] == [[0.0, 1.0], [1.0, 0.0]]
    assert out["counts"]["car"] == [1.0, 2.0, 0.0]
    assert out["duration"] == 12.0 and out["video_url"] is None
