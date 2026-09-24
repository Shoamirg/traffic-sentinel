"""failure_to_yield: a vehicle drives through a crossing while a pedestrian is on it.

For every pass of a moving vehicle through a crossing polygon, look for a
pedestrian who is on the same crossing (not on a refuge island) and close to
the vehicle's path at that moment - within NEAR_REL vehicle sizes. A walker on
the far end of a long crossing, someone waiting on the kerb end of it, or a
rider does not count. Start = the vehicle enters the
crossing, end = it leaves.
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


def detect(ctx: Context) -> list[Event]:
    events = []
    for poly in ctx.layout.crossings:
        peds = _pedestrians_on(ctx, poly)
        if not peds:
            continue
        for tr in ctx.tracks.vehicles():
            in_crossing = inside(poly, tr.xy)
            if not in_crossing.any():
                continue
            for s, e in enter_leave(tr.ts, in_crossing):
                sel = (tr.ts >= s) & (tr.ts <= e)
                if e - s < MIN_PASS_SEC or tr.speed[sel].max(initial=0.0) < MIN_PASS_SPEED:
                    continue
                if _pedestrian_near(peds, s, e, tr):
                    events.append(Event(s, e, "failure_to_yield", tids=(tr.tid,)))
    return events
