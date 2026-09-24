"""jaywalking: a pedestrian on the carriageway outside a marked crossing.

The pedestrian's ground point must be well inside the drawn road area (see
``interaction.on_road``: people waiting on the kerb do not count), outside
every crossing and refuge island, and the person must actually move across the
road while there - standing at a kerb for a minute is not jaywalking. Riders
are rejected: a "person" that overlaps a two-wheeler or moves at vehicle speed
is not a pedestrian.
"""
from __future__ import annotations

import numpy as np

from ..segments import Event, runs
from . import Context
from .interaction import in_vehicle_mask, near_polygon, on_road, rider_mask

MIN_DURATION = 1.0
MIN_TRAVEL_REL = 1.5        # body heights walked while on the road
MAX_WALK_REL = 2.5          # body heights per second; faster = rider
CROSSING_MARGIN_PX = 15.0   # walking just beside the painted stripes still counts as using the crossing
MIN_HEIGHT_PX = 18.0        # smaller (far-away) person detections are too unreliable to judge


def detect(ctx: Context) -> list[Event]:
    events = []
    for tr in ctx.tracks.people():
        road = on_road(ctx, tr.xy)
        if not road.any():
            continue
        for poly in ctx.layout.crossings + ctx.layout.refuges:
            road &= ~near_polygon(poly, tr.xy, CROSSING_MARGIN_PX)
        heights = np.asarray([b[3] - b[1] for b in tr.box])
        walking = tr.speed / np.maximum(heights, 1.0) <= MAX_WALK_REL
        big_enough = heights >= MIN_HEIGHT_PX
        flags = road & walking & big_enough & ~rider_mask(ctx, tr) & ~in_vehicle_mask(ctx, tr)
        for s, e in runs(tr.ts, flags, max_gap=1.0):
            i, j = tr.at(s), tr.at(e)
            travel = np.linalg.norm(tr.xy[j] - tr.xy[i]) / max(float(np.median(heights[i:j + 1])), 1.0)
            if e - s >= MIN_DURATION and travel >= MIN_TRAVEL_REL:
                events.append(Event(s, e, "jaywalking", tids=(tr.tid,)))
    return events
