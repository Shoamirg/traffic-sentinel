"""wrong_way: a vehicle moving against the learned traffic direction of its carriageway.

Per observation we ask the direction field what share of past traffic at that
spot moved the same way. Only lane-disciplined road counts: points inside the
junction box or on a crossing are ignored (turning traffic there goes every
way), and when the layout defines carriageways the vehicle must be on one. A
sustained run of "almost nobody drives like this here" over a real distance is
wrong-way driving. Start = the run begins, end = run ends or the vehicle leaves.
"""
from __future__ import annotations

import numpy as np

from ..scene import inside
from ..segments import Event, runs
from . import Context

MIN_VOTES = 15           # cell must have seen this much traffic to judge direction
MAX_SHARE = 0.04         # "almost nobody drives this way here"
MIN_SPEED_REL = 0.4      # sizes/s, only judge clearly moving vehicles
MIN_DURATION = 2.5       # s
MIN_TRAVEL_REL = 2.0     # box sizes travelled while wrong-way (rejects jitter)


def _lane_road(ctx: Context, xy: np.ndarray) -> np.ndarray:
    ok = np.ones(len(xy), bool)
    if ctx.layout.carriageways:
        ok = np.zeros(len(xy), bool)
        for c in ctx.layout.carriageways:
            ok |= inside(c["poly"], xy)
    if ctx.layout.intersection is not None:
        ok &= ~inside(ctx.layout.intersection, xy)
    for poly in ctx.layout.crossings:
        ok &= ~inside(poly, xy)
    return ok


def detect(ctx: Context) -> list[Event]:
    events = []
    for tr in ctx.tracks.vehicles():
        share, votes = ctx.scene.direction_share(tr.xy, tr.vel)
        moving = tr.speed / np.maximum(tr.size, 1.0) >= MIN_SPEED_REL
        flags = moving & (votes >= MIN_VOTES) & (share <= MAX_SHARE)
        if not flags.any():
            continue
        flags &= _lane_road(ctx, tr.xy)
        for s, e in runs(tr.ts, flags, max_gap=1.0):
            if e - s < MIN_DURATION:
                continue
            i, j = tr.at(s), tr.at(e)
            if np.linalg.norm(tr.xy[j] - tr.xy[i]) < MIN_TRAVEL_REL * float(np.median(tr.size[i:j + 1])):
                continue
            end = tr.end if tr.end - e < 1.0 else e      # left the frame while wrong-way
            events.append(Event(s, end, "wrong_way", tids=(tr.tid,)))
    return events
