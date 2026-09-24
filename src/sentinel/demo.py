"""Website live demo: one call that analyses an uploaded clip end to end.

Contract used by website/server/app.py:
    analyze(video_path, out_dir, progress, max_seconds=120) -> dict

To stay fast on a CPU host the risk curve is computed from the Part A
detections, fed frame by frame in time order through the same causal
estimator (``CausalRisk.step_detections``), instead of running the detector a
second time. The official harness path (solution.RiskEstimator) still runs its
own detector on the frames it is handed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from .extract import Extraction, extract
from .pipeline import build_context, run_rules
from .render import render
from .risk import CausalRisk
from .tracks import TrackSet

COUNT_KINDS = ("person", "car", "bus", "truck", "motorcycle", "bicycle")


def risk_curve(ex: Extraction, fps: float) -> list[list[float]]:
    est = CausalRisk()
    est.reset({"fps": fps, "width": ex.meta.width})
    return [[round(t, 3), round(est.step_detections(d, t), 4)] for t, d in ex.frame_dets]


def counts_per_second(tracks: TrackSet, duration: float) -> dict[str, list]:
    """Number of visible road users of each kind, sampled once per second."""
    times = np.arange(0.0, max(duration, 1.0), 1.0)
    out: dict[str, list] = {"t": times.round(1).tolist()}
    per_t = [Counter(tr.kind for tr in tracks.active_at(t, tol=0.1)) for t in times]
    for kind in COUNT_KINDS:
        out[kind] = [c.get(kind, 0) for c in per_t]
    return out


def analyze(video_path: str, out_dir: str, progress: Callable[[float, str], None],
            max_seconds: float = 120.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress(0.02, "detecting and tracking road users")
    ex = extract(video_path, progress=lambda f: progress(0.02 + 0.63 * f, "detecting and tracking road users"),
                 max_seconds=max_seconds)
    progress(0.66, "applying event rules")
    events = run_rules(build_context(ex))
    progress(0.70, "computing accident risk")
    risk = risk_curve(ex, ex.meta.fps)
    progress(0.72, "rendering annotated video")
    video_name = "annotated.mp4"
    render(video_path, out / video_name, ex.tracks, events, risk, max_seconds=max_seconds,
           progress=lambda f: progress(0.72 + 0.27 * f, "rendering annotated video"))
    progress(1.0, "done")
    return {
        "events": [ev.as_list() for ev in events],
        "risk": risk,
        "duration": round(ex.tracks.duration, 3),
        "fps": ex.meta.fps,
        "video": video_name,
        "counts": counts_per_second(ex.tracks, ex.tracks.duration),
    }
