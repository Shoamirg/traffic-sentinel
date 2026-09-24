"""Synthetic trajectories for rule tests: no video or GPU needed."""
from __future__ import annotations

import numpy as np

from sentinel.rules import Context
from sentinel.scene import Layout, SceneModel
from sentinel.tracks import Track, TrackSet

W, H = 1280, 720
FPS = 10.0


def make_track(tid: int, kind: str, t0: float, pts: list[tuple[float, float, float]], box_w: float = 80,
               box_h: float = 50) -> Track:
    """pts = [(t, x, y), ...] ground points; linearly interpolated at FPS."""
    ts = np.arange(t0, pts[-1][0] + 1e-9, 1 / FPS)
    kt = [p[0] for p in pts]
    xs = np.interp(ts, kt, [p[1] for p in pts])
    ys = np.interp(ts, kt, [p[2] for p in pts])
    boxes = [np.array([x - box_w / 2, y - box_h, x + box_w / 2, y], np.float32) for x, y in zip(xs, ys)]
    return Track(tid=tid, kind=kind, t=list(ts), box=boxes).finalize()


def eastbound_traffic(n: int = 30, y: float = 400, t_gap: float = 2.0, start_id: int = 1000) -> list[Track]:
    """Normal traffic crossing the frame left -> right at 200 px/s."""
    return [make_track(start_id + k, "car", k * t_gap, [(k * t_gap, 0, y), (k * t_gap + 6.4, W, y)])
            for k in range(n)]


def context(tracks: list[Track], duration: float = 120.0, layout: Layout | None = None,
            prior: list[Track] | None = None) -> Context:
    ts = TrackSet({tr.tid: tr for tr in tracks}, np.arange(0, duration, 1 / FPS), (W, H), duration)
    scene = SceneModel.empty(W, H).add_tracks(prior if prior is not None else tracks, 1 / FPS)
    thumbs = np.zeros((0, 1, 1, 3), np.uint8)
    return Context(ts, scene, layout or Layout(), thumbs, np.zeros(0), duration, {}, [])
