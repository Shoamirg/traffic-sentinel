"""Stand-in for sentinel.demo.analyze used for local web development.

Same signature and return schema as the real function. It reads fps/duration
with OpenCV when available (falls back to ffprobe, then to defaults), walks
through the real pipeline's stage names with a fake delay, invents plausible
events / risk / counts, and copies the input as the "annotated" video.

SENTINEL_FAKE_DELAY scales the fake processing time (default 1.0; 0 = instant).
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from typing import Callable

STAGES = [
    (0.05, "Reading video"),
    (0.15, "Detecting road users (YOLO11)"),
    (0.45, "Tracking (ByteTrack-style)"),
    (0.60, "Learning scene: direction field + road mask"),
    (0.72, "Applying event rules"),
    (0.82, "Scoring accident risk (TTC + braking)"),
    (0.92, "Rendering annotated video"),
]
FAKE_LABELS = ["stopped_vehicle", "jaywalking", "congestion", "near_miss", "red_light",
               "wrong_way", "failure_to_yield", "solid_line_crossing"]
CLASSES = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]


def _probe(path: str) -> tuple[float, float]:
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps > 0 and frames > 0:
            return float(fps), float(frames / fps)
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate:format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=20, check=True).stdout
        info = json.loads(out)
        num, den = info["streams"][0]["r_frame_rate"].split("/")
        return float(num) / float(den or 1), float(info["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 30.0, 60.0


def analyze(video_path: str, out_dir: str, progress: Callable[[float, str], None],
            max_seconds: float = 120.0) -> dict:
    delay = float(os.environ.get("SENTINEL_FAKE_DELAY", "1.0"))
    rng = random.Random(os.path.getsize(video_path))
    fps, duration = _probe(video_path)
    duration = max(1.0, min(duration, max_seconds))

    for frac, stage in STAGES:
        progress(frac, stage)
        time.sleep(0.8 * delay)

    events = []
    n_events = max(2, int(duration / 12))
    for _ in range(n_events):
        s = rng.uniform(0, max(0.1, duration - 3))
        e = min(duration, s + rng.uniform(1.5, 10.0))
        events.append([round(s, 2), round(e, 2), rng.choice(FAKE_LABELS)])
    events.sort()

    risk, level, step = [], 0.05, 0.2
    peaks = [ev[0] for ev in events if ev[2] == "near_miss"] or [duration * 0.6]
    t = 0.0
    while t <= duration:
        target = 0.05 + 0.1 * rng.random()
        for p in peaks:
            if p - 3.0 <= t <= p + 0.5:
                target = max(target, 0.35 + 0.6 * (1 - abs(p - t) / 3.0))
        level = target if target > level else level * 0.85 + target * 0.15  # fast attack / slow decay
        risk.append([round(t, 2), round(min(1.0, level), 3)])
        t += step

    ts = [round(i * 0.5, 2) for i in range(int(duration / 0.5) + 1)]
    counts: dict[str, list] = {"t": ts}
    base = {"person": 4, "car": 14, "bus": 1, "truck": 2, "motorcycle": 3, "bicycle": 1}
    for cls in CLASSES:
        phase = rng.uniform(0, 6.28)
        counts[cls] = [max(0, round(base[cls] * (1 + 0.4 * math.sin(x / 9 + phase)) + rng.gauss(0, 0.6)))
                       for x in ts]

    progress(0.97, "Packaging results")
    name = "annotated.mp4"
    shutil.copyfile(video_path, os.path.join(out_dir, name))
    return {"events": events, "risk": risk, "duration": round(duration, 3), "fps": round(fps, 3),
            "video": name, "counts": counts}
