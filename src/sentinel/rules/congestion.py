"""congestion: traffic at a standstill or crawling across a whole direction.

Vehicles are grouped by the dominant lane direction at their position (from
the direction field, so a stopped car still belongs to its lane). Each second,
a direction is congested when enough of its vehicles are present and their
median size-normalised speed is a crawl. Sustained runs become events.
"""
from __future__ import annotations

import numpy as np

from ..segments import Event, runs
from . import Context

MIN_VEHICLES = 6
CRAWL_REL = 0.25        # sizes per second
MIN_SLOW_FRACTION = 0.7
MIN_DURATION = 75.0      # longer than a red phase + queue discharge (cycle here is ~80 s, red ~40 s)
STEP = 1.0


def _lane_direction(ctx: Context, xy: np.ndarray) -> np.ndarray:
    """Dominant heading bin of the direction field at each point, grouped into 2 halves."""
    cy, cx = ctx.scene.cells(xy)
    dom = ctx.scene.headings[cy, cx].argmax(axis=1)
    return (dom * 2 // ctx.scene.headings.shape[2]).astype(int)   # 0 or 1: coarse direction


def detect(ctx: Context) -> list[Event]:
    vehicles = ctx.tracks.vehicles()
    if not vehicles:
        return []
    times = np.arange(0.0, ctx.duration, STEP)
    flags = {0: np.zeros(len(times), bool), 1: np.zeros(len(times), bool)}
    for k, t in enumerate(times):
        pts, rel = [], []
        for tr in vehicles:
            if tr.start <= t <= tr.end:
                i = tr.at(t)
                pts.append(tr.xy[i])
                rel.append(tr.speed[i] / max(tr.size[i], 1.0))
        if len(pts) < MIN_VEHICLES:
            continue
        group = _lane_direction(ctx, np.asarray(pts))
        rel = np.asarray(rel)
        for g in (0, 1):
            sel = rel[group == g]
            if len(sel) >= MIN_VEHICLES and np.mean(sel < CRAWL_REL) >= MIN_SLOW_FRACTION:
                flags[g][k] = True
    events = []
    for g in (0, 1):
        for s, e in runs(times, flags[g], max_gap=3.0):
            if e - s >= MIN_DURATION:
                events.append(Event(s, e + STEP, "congestion"))
    return events
