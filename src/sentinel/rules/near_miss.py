"""near_miss: hard braking or swerving to avoid another road user, no contact.

For each pair on a collision course (short time-to-contact while closing fast,
and paths that actually meet: predicted closest approach under MISS_REL sizes),
look for an evasive action by a *vehicle* of the pair: strong deceleration or
a sharp heading change. Pedestrian boxes are too small and jittery for their
size-normalised acceleration to mean anything, and a car stopping for people
at a crossing is ordinary yielding. If the boxes never overlap it is a near miss. Start = onset
of the evasive action, end = the pair is clear (opening again and apart).
"""
from __future__ import annotations

import numpy as np

from ..segments import Event
from . import Context
from .interaction import candidate_pairs, heading_change, rel_speed, stable_size

TTC_MAX = 1.2               # s
MIN_CLOSING_REL = 1.5       # sizes/s
BRAKE_REL = 2.5             # sizes/s^2 sustained deceleration
BRAKE_WINDOW_SEC = 0.4      # ...averaged over this window: a pointwise derivative spikes on box jitter
DANGER_MIN_SAMPLES = 2      # the collision course must hold on consecutive samples, not one frame
SWERVE_RAD = np.deg2rad(35)
CONTACT_IOU = 0.12
CLEAR_DIST_REL = 1.5
MAX_EVENT_SEC = 8.0
MISS_REL = 0.6              # predicted closest approach (sizes) below which paths meet


def _evasive_onset(tr, t_lo: float, t_hi: float) -> float | None:
    sel = np.flatnonzero((tr.ts >= t_lo) & (tr.ts <= t_hi))
    if len(sel) < 3:
        return None
    rs = rel_speed(tr)
    for k in sel:
        j = tr.at(tr.ts[k] + BRAKE_WINDOW_SEC)
        if j > k and (rs[k] - rs[j]) / (tr.ts[j] - tr.ts[k]) >= BRAKE_REL:
            return float(tr.ts[k])
    for k in range(len(sel) - 1):
        j = tr.at(tr.ts[sel[k]] + 1.0)
        if rs[sel[k]] > 0.5 and heading_change(tr, sel[k], j) >= SWERVE_RAD:
            return float(tr.ts[sel[k]])
    return None


def _paths_meet(ps, k: int) -> bool:
    """Closest point of approach, from sample k of the pair, within TTC_MAX s and MISS_REL sizes."""
    a, b = ps.a, ps.b
    dp = b.xy[ps.ib[k]] - a.xy[ps.ia[k]]
    dv = b.vel[ps.ib[k]] - a.vel[ps.ia[k]]
    vv = float(dv @ dv)
    if vv < 1e-6:
        return False
    t_star = min(max(-float(dp @ dv) / vv, 0.0), TTC_MAX)
    size = 0.5 * (a.size[ps.ia[k]] + b.size[ps.ib[k]])
    return float(np.linalg.norm(dp + dv * t_star)) / max(size, 1.0) < MISS_REL


def detect(ctx: Context) -> list[Event]:
    events = []
    for ps in candidate_pairs(ctx.tracks.usable()):
        if not (ps.a.is_vehicle or ps.b.is_vehicle) or ps.iou.max() >= CONTACT_IOU:
            continue
        ttc = np.where(ps.closing_rel > MIN_CLOSING_REL, ps.dist_rel / np.maximum(ps.closing_rel, 1e-6), np.inf)
        meet = np.zeros(len(ps.t), bool)
        for k in np.flatnonzero(ttc < TTC_MAX):
            meet[k] = _paths_meet(ps, k)
        held = np.convolve(meet.astype(int), np.ones(DANGER_MIN_SAMPLES, int), mode="valid") >= DANGER_MIN_SAMPLES
        if not held.any():
            continue
        t_d = ps.t[int(np.argmax(held))]
        onsets = [o for o in (_evasive_onset(tr, t_d - 1.0, t_d + 1.5) for tr in (ps.a, ps.b) if tr.is_vehicle)
                  if o is not None]
        if not onsets:
            continue
        start = min(onsets)
        if not (stable_size(ps.a, start) and stable_size(ps.b, start)):
            continue
        clear = np.flatnonzero((ps.t > t_d) & (ps.closing_rel < 0) & (ps.dist_rel > CLEAR_DIST_REL))
        end = ps.t[clear[0]] if clear.size else min(ps.t[-1], start + MAX_EVENT_SEC)
        if end - start > 0.3:
            events.append(Event(start, min(end, start + MAX_EVENT_SEC), "near_miss", tids=(ps.a.tid, ps.b.tid)))
    return events
