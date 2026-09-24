"""Part B: causal accident-anticipation score.

Runs its own detector+tracker online on every k-th frame (only frames already
handed to ``step``) and never touches Part A output. Per update, all lengths in
*sizes* (box diagonals) so near and far road users are judged alike:

* every nearby pair that involves a moving vehicle gets a closest-point-of-
  approach estimate from ~1 s of smoothed ground-point velocity. Only pairs
  whose paths actually meet (predicted miss under MISS_REL sizes, within
  HORIZON s) count, and they are scored by DRAC - the deceleration needed to
  avoid the collision, closing_speed^2 / (2 * gap). Cars rolling up to a queue
  or following at similar speed need gentle braking and score low; fast
  closing on a short gap scores high. Pedestrian-pedestrian pairs and pairs of
  standing vehicles are ignored: with ~50 tracked road users per frame, box
  jitter otherwise keeps some pair "on a collision course" at all times;
* a moving vehicle's observed hard deceleration (same unit) also counts.

The per-update hazard must persist for PERSIST_SEC (one-frame spikes are
tracking glitches), is mapped through a logistic calibrated on the sample
videos (treated as normal traffic: median ~0.02, 99.9th percentile ~0.5) and
smoothed with a fast-attack / slow-decay filter so alarms do not flicker.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import config
from .config import COCO_TO_TYPE, VEHICLE_TYPES
from .detector import Detections
from .models import shared_detector
from .tracker import ByteTracker
from .video import resize_to_work, stride_for

HISTORY_SEC = 1.0            # velocity is fitted over this much track history
MIN_HISTORY = 4
HORIZON = 3.0                # s: only conflicts predicted this soon count
MISS_REL = 0.6               # predicted miss distance (sizes) below which paths meet
MIN_GAP_REL = 0.25           # sizes: floor on the bumper gap so DRAC stays finite
MIN_CLOSING_REL = 0.5        # closing speed in sizes/second needed to matter
MOVING_REL = 0.8             # a vehicle slower than this (sizes/second) is standing or creeping
PERSIST_SEC = 0.5            # hazard must hold this long before it raises the score
# logistic on the persistent hazard (sizes/s^2): score = 1 / (1 + exp(-(h - CAL_MID) / CAL_WIDTH));
# CAL_MID = 99.9th percentile of the sample videos, CAL_WIDTH puts h = 0 at 0.02 (see tools/calibrate_risk.py)
CAL_MID = 6.1                # 99.9th pct of persistent hazard on the 4 samples (2026-09-24)
CAL_WIDTH = CAL_MID / np.log(49.0)
DECAY_PER_SEC = 0.5


def _fit_velocity(hist: deque) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares position/velocity at the latest time from (t, x, y, size) history."""
    arr = np.asarray(hist)
    t = arr[:, 0] - arr[-1, 0]
    A = np.stack([np.ones_like(t), t], axis=1)
    coef, *_ = np.linalg.lstsq(A, arr[:, 1:3], rcond=None)
    return coef[0], coef[1], float(np.median(arr[:, 3]))


def pair_hazard(p1, v1, s1, p2, v2, s2) -> float:
    """DRAC in sizes/s^2 for two road users whose paths meet within HORIZON s, else 0."""
    dp, dv = p2 - p1, v2 - v1
    vv = float(dv @ dv)
    scale = 0.5 * (s1 + s2)
    if vv < 1e-6:
        return 0.0
    dist = float(np.linalg.norm(dp))
    closing = -float(dp @ dv) / (dist + 1e-6) / scale
    if closing < MIN_CLOSING_REL:
        return 0.0
    t_star = -float(dp @ dv) / vv
    if not 0.0 < t_star < HORIZON:
        return 0.0
    miss = float(np.linalg.norm(dp + dv * t_star)) / scale
    if miss >= MISS_REL:
        return 0.0
    gap = max(dist / scale - 1.0, MIN_GAP_REL)
    return float(closing ** 2 / (2.0 * gap) * (1.0 - (miss / MISS_REL) ** 2))


class CausalRisk:
    def __init__(self) -> None:
        self.detector = None
        self.reset({"fps": 25.0, "width": config.WORK_WIDTH})

    def reset(self, meta: dict) -> None:
        self.fps = float(meta.get("fps") or 25.0)
        width = int(meta.get("width") or config.WORK_WIDTH)
        self.scale = min(1.0, config.WORK_WIDTH / width)
        self.stride = stride_for(self.fps, config.PART_B_TARGET_FPS)
        self.tracker = ByteTracker()
        self.hist: dict[int, deque] = {}
        self.prev_speed: dict[int, tuple[float, float]] = {}
        self.kind: dict[int, str] = {}
        self.recent: deque = deque()
        self.n = 0
        self.score = 0.0
        self.last_t = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        n, self.n = self.n, self.n + 1
        if n % self.stride:
            return self.score
        if self.detector is None:
            self.detector = shared_detector()
        return self.step_detections(self.detector([resize_to_work(frame, self.scale)])[0], t_sec)

    def step_detections(self, dets: Detections, t_sec: float) -> float:
        """Advance with detections of the frame at t_sec (frames must arrive in time order)."""
        tracked = self.tracker.update(dets, t_sec)
        raw = self._persistent(self._hazard(tracked, t_sec), t_sec)
        target = 1.0 / (1.0 + np.exp(-(raw - CAL_MID) / CAL_WIDTH))
        decay = DECAY_PER_SEC ** max(t_sec - self.last_t, 0.0)
        self.score = float(max(target, self.score * decay))
        self.last_t = t_sec
        return self.score

    def _persistent(self, hazard: float, t: float) -> float:
        """Lowest hazard over the last PERSIST_SEC: a level counts only once it has held that long."""
        self.recent.append((t, hazard))
        while t - self.recent[0][0] > PERSIST_SEC:
            self.recent.popleft()
        if t - self.recent[0][0] < PERSIST_SEC * 0.5:     # not enough history yet
            return 0.0
        return min(h for _, h in self.recent)

    def _hazard(self, tracked, t: float) -> float:
        live = set()
        for tid, box, cls, _ in tracked:
            live.add(tid)
            self.kind[tid] = COCO_TO_TYPE.get(int(cls), "other")
            h = self.hist.setdefault(tid, deque())
            h.append((t, (box[0] + box[2]) / 2, box[3], float(np.hypot(box[2] - box[0], box[3] - box[1]))))
        for tid in list(self.hist):
            h = self.hist[tid]
            while h and t - h[0][0] > HISTORY_SEC:
                h.popleft()
            if not h:
                del self.hist[tid]
                self.prev_speed.pop(tid, None)
                self.kind.pop(tid, None)

        states = {tid: _fit_velocity(h) for tid, h in self.hist.items()
                  if tid in live and len(h) >= MIN_HISTORY}
        rel_speed = {tid: float(np.linalg.norm(v)) / max(s, 1.0) for tid, (_, v, s) in states.items()}
        moving_vehicle = {tid for tid in states
                          if self.kind.get(tid) in VEHICLE_TYPES and rel_speed[tid] >= MOVING_REL}
        hazard = 0.0
        ids = sorted(states)
        for a in range(len(ids)):
            p1, v1, s1 = states[ids[a]]
            for b in range(a + 1, len(ids)):
                if ids[a] not in moving_vehicle and ids[b] not in moving_vehicle:
                    continue
                p2, v2, s2 = states[ids[b]]
                if np.linalg.norm(p2 - p1) > 6 * max(s1, s2):
                    continue
                hazard = max(hazard, pair_hazard(p1, v1, s1, p2, v2, s2))
        for tid, speed in rel_speed.items():
            prev = self.prev_speed.get(tid)
            if prev is not None and t > prev[1] and self.kind.get(tid) in VEHICLE_TYPES and prev[0] >= MOVING_REL:
                decel = (prev[0] - speed) / (t - prev[1])
                hazard = max(hazard, decel)
            self.prev_speed[tid] = (speed, t)
        return hazard
