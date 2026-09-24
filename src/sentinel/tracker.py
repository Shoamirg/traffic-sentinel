"""Compact ByteTrack-style multi-object tracker.

Two-stage association (high-confidence detections first, then low-confidence
ones against the leftovers) with a constant-velocity box prediction. Written
in-house instead of importing Ultralytics' tracker internals, whose signatures
change between releases; the algorithm follows ByteTrack (Zhang et al., 2022).
Deterministic: no randomness anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detections

HIGH_CONF = 0.45
NEW_TRACK_CONF = 0.5
MATCH_IOU = 0.2
MAX_LOST_SEC = 1.5
VEL_ALPHA = 0.6          # EMA factor for box velocity


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


@dataclass
class _Live:
    tid: int
    box: np.ndarray
    cls: int
    t_last: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    hits: int = 1

    def predict(self, t: float) -> np.ndarray:
        return self.box + self.vel * (t - self.t_last)

    def update(self, box: np.ndarray, cls: int, t: float) -> None:
        dt = max(t - self.t_last, 1e-3)
        self.vel = VEL_ALPHA * (box - self.box) / dt + (1 - VEL_ALPHA) * self.vel
        self.box, self.cls, self.t_last = box, cls, t
        self.hits += 1


class ByteTracker:
    """Call ``update(dets, t)`` once per analysed frame; returns [(tid, box, cls, conf), ...]."""

    def __init__(self) -> None:
        self.live: list[_Live] = []
        self.next_id = 1

    def _associate(self, tracks: list[_Live], boxes: np.ndarray, t: float) -> tuple[list, list, list]:
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        pred = np.stack([tr.predict(t) for tr in tracks])
        iou = iou_matrix(pred, boxes)
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= MATCH_IOU]
        mt, md = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def update(self, dets: Detections, t: float) -> list[tuple[int, np.ndarray, int, float]]:
        high = dets.conf >= HIGH_CONF
        out = []

        hi_idx = np.flatnonzero(high)
        pairs, un_t, un_d = self._associate(self.live, dets.xyxy[hi_idx], t)
        for r, c in pairs:
            j = hi_idx[c]
            self.live[r].update(dets.xyxy[j], int(dets.cls[j]), t)
            out.append((self.live[r].tid, dets.xyxy[j], int(dets.cls[j]), float(dets.conf[j])))

        lo_idx = np.flatnonzero(~high)
        leftovers = [self.live[i] for i in un_t]
        pairs2, un_t2, _ = self._associate(leftovers, dets.xyxy[lo_idx], t)
        for r, c in pairs2:
            j = lo_idx[c]
            leftovers[r].update(dets.xyxy[j], int(dets.cls[j]), t)
            out.append((leftovers[r].tid, dets.xyxy[j], int(dets.cls[j]), float(dets.conf[j])))

        for c in un_d:
            j = hi_idx[c]
            if dets.conf[j] >= NEW_TRACK_CONF:
                tr = _Live(self.next_id, dets.xyxy[j].copy(), int(dets.cls[j]), t)
                self.next_id += 1
                self.live.append(tr)
                out.append((tr.tid, dets.xyxy[j], tr.cls, float(dets.conf[j])))

        self.live = [tr for tr in self.live if t - tr.t_last <= MAX_LOST_SEC]
        return out
