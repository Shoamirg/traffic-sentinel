"""fire_smoke: persistent fire or smoke from the fine-tuned detector.

The D-Fire model (weights/FIRE_SMOKE.md) fires on headlight glare at night and
boxes whole grey road surfaces as "smoke", so a hit must pass three filters:

* confidence >= MIN_CONF and box area between MIN_AREA_FRAC and MAX_AREA_FRAC
  of the frame (single headlights / whole-frame boxes are rejected);
* a "fire" box mostly covering a tracked vehicle is ignored (lamps, glare);
* persistence: the same class in >= PERSIST_HITS of the last PERSIST_WINDOW
  samples (1 per second) with overlapping boxes.

Start = first sample of a persistent run, end = last hit (or the video end).
"""
from __future__ import annotations

import numpy as np

from ..segments import Event, runs
from ..tracker import iou_matrix
from . import Context

FIRE, SMOKE = 0, 1
MIN_CONF = 0.6
MIN_AREA_FRAC = 0.001
MAX_AREA_FRAC = 0.5
VEHICLE_OVERLAP = 0.5        # fraction of the fire box inside a vehicle box
PERSIST_WINDOW = 4
PERSIST_HITS = 3
MIN_DURATION = 3.0


def _boxes_on_vehicles(ctx: Context, t: float) -> np.ndarray:
    boxes = [tr.box[tr.at(t)] for tr in ctx.tracks.active_at(t, tol=0.3) if tr.is_vehicle]
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def _inside_fraction(box: np.ndarray, others: np.ndarray) -> float:
    if len(others) == 0:
        return 0.0
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    return float(inter.max() / area)


def _valid_boxes(ctx: Context, t: float, det) -> dict[int, np.ndarray]:
    w, h = ctx.tracks.frame_size
    frame_area = max(w * h, 1)
    vehicles = None
    out: dict[int, list] = {FIRE: [], SMOKE: []}
    for box, conf, cls in zip(det.xyxy, det.conf, det.cls):
        frac = (box[2] - box[0]) * (box[3] - box[1]) / frame_area
        if conf < MIN_CONF or not MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC or int(cls) not in out:
            continue
        if int(cls) == FIRE:
            vehicles = _boxes_on_vehicles(ctx, t) if vehicles is None else vehicles
            if _inside_fraction(box, vehicles) >= VEHICLE_OVERLAP:
                continue
        out[int(cls)].append(box)
    return {c: np.asarray(b, np.float32).reshape(-1, 4) for c, b in out.items()}


def _persistent(samples: list[dict[int, np.ndarray]], cls: int) -> np.ndarray:
    """flags[k]: class seen in >= PERSIST_HITS of the window ending at k, boxes overlapping."""
    flags = np.zeros(len(samples), bool)
    for k in range(len(samples)):
        cur = samples[k][cls]
        if len(cur) == 0:
            continue
        window = samples[max(0, k - PERSIST_WINDOW + 1):k + 1]
        hits = sum(1 for s in window if len(s[cls]) and iou_matrix(cur, s[cls]).max() > 0.05)
        flags[k] = hits >= PERSIST_HITS
    return flags


def detect(ctx: Context) -> list[Event]:
    if not ctx.fire:
        return []
    times = np.asarray([t for t, _ in ctx.fire])
    samples = [_valid_boxes(ctx, t, det) for t, det in ctx.fire]
    flags = _persistent(samples, FIRE) | _persistent(samples, SMOKE)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    events = []
    for s, e in runs(times, flags, max_gap=3 * step):
        start = max(0.0, s - (PERSIST_HITS - 1) * step)          # the run became persistent after a few hits
        end = ctx.duration if ctx.duration - e < 2 * step else e
        if end - start >= MIN_DURATION:
            events.append(Event(start, end, "fire_smoke"))
    return events
