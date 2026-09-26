"""failure_to_yield: a vehicle drives through a crossing while a pedestrian is on it.

For every pass of a moving vehicle through a crossing polygon, look for a
pedestrian who is on the same crossing (not on a refuge island) and close to
the vehicle's path at that moment - within NEAR_REL vehicle sizes. A walker on
the far end of a long crossing, someone waiting on the kerb end of it, or a
rider does not count. Start = the vehicle enters the crossing, end = it leaves.

Calibrated on our dev labels of the samples (labels/dev_labels.json):
- the vehicle is on the crossing while any point of its box's lower edge is
  inside the polygon (a single ground point is inside for only ~1 s of a
  ~2-3 s pass), and only while it moves: a car standing on the crossing for
  its queue is not driving through it, and a pass longer than MAX_PASS_SEC is
  a queue, not a pass;
- the pedestrian test uses the fixed road-edge margin, not the per-track
  hysteresis of jaywalking: the crossings span narrow slip lanes where nobody
  gets 24 px deep into the road;
- annotators mark entry/exit later than the box edge does (median 0.4 s and
  0.9 s over 14 matched passes), so segments are shifted by START_LAG/END_LAG.
Together: F1 (mean over tIoU 0.3/0.5/0.7) 0.06 -> 0.49 on the dev labels.
"""
from __future__ import annotations

import numpy as np

from ..geometry import enter_leave, inside
from ..segments import Event
from . import Context
from .interaction import in_vehicle_mask, on_road, rider_mask

MIN_PASS_SPEED = 20.0       # px/s: a vehicle crawling/stopped for the pedestrian is yielding
NEAR_REL = 1.5              # pedestrian within this many vehicle sizes of the vehicle
MIN_PASS_SEC = 0.2
MAX_PASS_SEC = 8.0
START_LAG = 0.4             # s, annotated entry after the box edge enters
END_LAG = 0.9               # s, annotated exit after the box edge leaves


def _pedestrians_on(ctx: Context, poly: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(times, ground points) of each pedestrian while on the crossing and off every refuge."""
    out = []
    for tr in ctx.tracks.people():
        on = inside(poly, tr.xy)
        if not on.any():
            continue
        on &= on_road(ctx, tr.xy) & ~rider_mask(ctx, tr) & ~in_vehicle_mask(ctx, tr)   # road part; no riders/drivers
        for refuge in ctx.layout.refuges:
            on &= ~inside(refuge, tr.xy)
        if on.any():
            out.append((tr.ts[on], tr.xy[on]))
    return out


def _pedestrian_near(peds, t0: float, t1: float, tr) -> bool:
    for ts, xy in peds:
        sel = (ts >= t0) & (ts <= t1)
        for t, p in zip(ts[sel], xy[sel]):
            i = tr.at(t)
            if np.linalg.norm(tr.xy[i] - p) < NEAR_REL * tr.size[i]:
                return True
    return False


def _box_on(poly: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """True where any point of the box's lower edge (corners, centre, a quarter up) is inside poly."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    probes = [(x1, y2), (cx, y2), (x2, y2), (cx, (y1 + 3 * y2) / 4)]
    return np.any([inside(poly, np.stack([px, py], axis=1)) for px, py in probes], axis=0)


def detect(ctx: Context) -> list[Event]:
    events = []
    for poly in ctx.layout.crossings:
        peds = _pedestrians_on(ctx, poly)
        if not peds:
            continue
        for tr in ctx.tracks.vehicles():
            passing = _box_on(poly, np.asarray(tr.box)) & (tr.speed >= MIN_PASS_SPEED)
            if not passing.any():
                continue
            for s, e in enter_leave(tr.ts, passing):
                if not MIN_PASS_SEC <= e - s <= MAX_PASS_SEC:
                    continue
                if _pedestrian_near(peds, s, e, tr):
                    events.append(Event(s + START_LAG, min(e + END_LAG, ctx.duration), "failure_to_yield",
                                        tids=(tr.tid,)))
    return events
