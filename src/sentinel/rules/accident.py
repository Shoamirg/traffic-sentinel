"""accident: contact between road users that ends with them standing together.

In dense junction traffic boxes overlap all the time (occlusion, queues), so a
contact only counts when the whole signature of a crash is there:

1. approach: the pair was closing fast in the 1.5 s before contact;
2. contact: boxes overlap and ground points are within half a body length;
3. impact: at least one party loses a lot of speed within REACTION_SEC;
4. aftermath: both parties then stand still, next to each other, for a while.

Contacts where either box changes size abruptly (partial occlusion by poles,
buses, merged detections) are rejected: that is how tracking artefacts mimic
a sudden stop. Start = first contact; end = both parties stopped moving.
"""
from __future__ import annotations

import numpy as np

from ..segments import Event
from . import Context
from .interaction import shared_pairs, rel_speed, stable_size

CONTACT_IOU = 0.15
CONTACT_DIST_REL = 0.5
APPROACH_REL = 1.2          # closing speed (sizes/s) before contact
REACTION_SEC = 1.0
SPEED_DROP_REL = 1.2        # sizes/s lost within REACTION_SEC
STILL_REL = 0.15
AFTER_STILL_SEC = 3.0       # both stand still at least this long ...
AFTER_WINDOW_SEC = 4.0      # ... starting within this long after contact
TOGETHER_REL = 1.2          # and stay this close
SIZE_JUMP = 1.4             # max/min box size within +-1 s of contact
MAX_EVENT_SEC = 30.0


def _speed_drop(tr, t0: float) -> float:
    rs = rel_speed(tr)
    before = rs[(tr.ts >= t0 - 0.5) & (tr.ts <= t0)]
    after = rs[(tr.ts >= t0) & (tr.ts <= t0 + REACTION_SEC)]
    return float(before.max() - after.min()) if len(before) and len(after) else 0.0


def _stop_time(tr, t0: float) -> float | None:
    """When the track starts standing still for AFTER_STILL_SEC (None if it never does)."""
    rs = rel_speed(tr)
    for k in np.flatnonzero((tr.ts >= t0) & (tr.ts <= t0 + AFTER_WINDOW_SEC) & (rs < STILL_REL)):
        window = (tr.ts >= tr.ts[k]) & (tr.ts <= tr.ts[k] + AFTER_STILL_SEC)
        if tr.ts[k] + AFTER_STILL_SEC <= tr.end and np.all(rs[window] < STILL_REL):
            return float(tr.ts[k])
    return None


def _crash(ps, k: int) -> float | None:
    """End time if the contact at index k has the full crash signature, else None."""
    t0 = ps.t[k]
    if ps.iou[k] < CONTACT_IOU or ps.dist_rel[k] > CONTACT_DIST_REL:
        return None
    before = (ps.t >= t0 - 1.5) & (ps.t < t0)
    if not before.any() or ps.closing_rel[before].max() < APPROACH_REL:
        return None
    if not (stable_size(ps.a, t0, SIZE_JUMP) and stable_size(ps.b, t0, SIZE_JUMP)):
        return None
    if max(_speed_drop(ps.a, t0), _speed_drop(ps.b, t0)) < SPEED_DROP_REL:
        return None
    stops = [_stop_time(tr, t0) for tr in (ps.a, ps.b)]
    if any(s is None for s in stops):
        return None
    after = (ps.t >= max(stops)) & (ps.t <= max(stops) + AFTER_STILL_SEC)
    if not after.any() or ps.dist_rel[after].max() > TOGETHER_REL:
        return None
    return min(max(max(stops), t0 + 1.0), t0 + MAX_EVENT_SEC)


def detect(ctx: Context) -> list[Event]:
    events = []
    for ps in shared_pairs(ctx):
        if not (ps.a.is_vehicle or ps.b.is_vehicle):
            continue
        for k in np.flatnonzero(ps.iou >= CONTACT_IOU):
            end = _crash(ps, k)
            if end is not None:
                events.append(Event(float(ps.t[k]), end, "accident", tids=(ps.a.tid, ps.b.tid)))
                break
    return events
