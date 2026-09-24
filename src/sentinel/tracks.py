"""Trajectory container and kinematics derived from tracker output."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Track:
    """One tracked road user. Points are the bottom-centre of the box (ground contact)."""
    tid: int
    kind: str                                  # person / car / truck / ...
    t: list[float] = field(default_factory=list)
    box: list[np.ndarray] = field(default_factory=list)   # xyxy per observation

    # derived by finalize()
    ts: np.ndarray | None = None               # (N,)
    xy: np.ndarray | None = None               # (N, 2) smoothed ground point
    vel: np.ndarray | None = None              # (N, 2) px/s
    speed: np.ndarray | None = None            # (N,)
    size: np.ndarray | None = None             # (N,) box diagonal, px

    @property
    def start(self) -> float:
        return self.t[0]

    @property
    def end(self) -> float:
        return self.t[-1]

    @property
    def is_vehicle(self) -> bool:
        return self.kind in config.VEHICLE_TYPES

    def finalize(self) -> "Track":
        boxes = np.asarray(self.box, np.float32)
        ts = np.asarray(self.t, np.float64)
        raw = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1).astype(np.float64)
        self.ts = ts
        self.xy, self.vel = local_linear_fit(ts, raw, config.SMOOTH_WINDOW_SEC)
        self.speed = np.linalg.norm(self.vel, axis=1)
        self.size = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        return self

    def at(self, t: float) -> int:
        """Index of the observation nearest to time t."""
        return int(np.argmin(np.abs(self.ts - t)))


def local_linear_fit(ts: np.ndarray, xy: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity from a least-squares line fitted in a centred time window.

    Unlike a moving average, the fit is exact for constant velocity whatever the
    number of samples in the window, so floating-point frame times do not make
    the speed jitter. Computed in O(N) with cumulative sums.
    """
    n = len(ts)
    if n < 3:
        vel = np.gradient(xy, ts, axis=0) if n == 2 else np.zeros_like(xy)
        return xy.copy(), vel
    tc = ts - ts.mean()
    half = window / 2 + 1e-6
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")

    def wsum(v: np.ndarray) -> np.ndarray:
        c = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)])
        return c[hi] - c[lo]

    cnt = (hi - lo).astype(np.float64)
    s_t, s_tt = wsum(tc), wsum(tc * tc)
    s_x, s_tx = wsum(xy), wsum(xy * tc[:, None])
    den = cnt * s_tt - s_t ** 2
    ok = den > 1e-9
    slope = np.zeros_like(xy)
    slope[ok] = (cnt[ok, None] * s_tx[ok] - s_t[ok, None] * s_x[ok]) / den[ok, None]
    mean_x, mean_t = s_x / cnt[:, None], s_t / cnt
    pos = mean_x + slope * (tc - mean_t)[:, None]
    return pos, slope


class TrackSet:
    """All tracks of one video plus the frame times that were analysed."""

    def __init__(self, tracks: dict[int, Track], frame_times: np.ndarray, frame_size: tuple[int, int],
                 duration: float) -> None:
        self.tracks = tracks
        self.frame_times = frame_times
        self.frame_size = frame_size            # (width, height) of the working frame
        self.duration = duration

    def usable(self, min_obs: int = config.TRACK_MIN_OBS) -> list[Track]:
        return [tr for tr in self.tracks.values() if len(tr.t) >= min_obs]

    def vehicles(self) -> list[Track]:
        return [tr for tr in self.usable() if tr.is_vehicle]

    def people(self) -> list[Track]:
        return [tr for tr in self.usable() if tr.kind == "person"]

    def active_at(self, t: float, tol: float = 0.25) -> list[Track]:
        return [tr for tr in self.usable() if tr.start - tol <= t <= tr.end + tol]
