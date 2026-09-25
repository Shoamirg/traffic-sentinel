"""Pairwise kinematics shared by the accident and near-miss rules."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..tracker import iou_matrix
from ..tracks import Track

MAX_PAIR_DIST_REL = 4.0     # only pairs that come within 4 box sizes are considered


@dataclass(frozen=True)
class PairSeries:
    a: Track
    b: Track
    t: np.ndarray           # common timestamps
    ia: np.ndarray          # indices into a
    ib: np.ndarray          # indices into b
    dist_rel: np.ndarray    # ground distance / mean size
    closing_rel: np.ndarray # closing speed / mean size (positive = approaching)
    iou: np.ndarray


def _nearest(ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the nearest entry of sorted `ts` for every value in `t`."""
    j = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    return np.where(np.abs(ts[j - 1] - t) <= np.abs(ts[j] - t), j - 1, j)


def pair_series(a: Track, b: Track) -> PairSeries | None:
    lo, hi = max(a.start, b.start), min(a.end, b.end)
    if hi - lo < 0.5 or len(b.ts) < 2:
        return None
    t = a.ts[(a.ts >= lo) & (a.ts <= hi)]
    if len(t) < 3:
        return None
    ib = _nearest(b.ts, t)
    ok = np.abs(b.ts[ib] - t) < 0.15
    if ok.sum() < 3:
        return None
    t, ib = t[ok], ib[ok]
    ia = np.searchsorted(a.ts, t)
    dp = b.xy[ib] - a.xy[ia]
    dv = b.vel[ib] - a.vel[ia]
    size = 0.5 * (a.size[ia] + b.size[ib])
    dist = np.linalg.norm(dp, axis=1)
    closing = -(dp * dv).sum(axis=1) / np.maximum(dist, 1e-6)
    boxes_a = np.asarray(a.box)[ia]
    boxes_b = np.asarray(b.box)[ib]
    iou = np.array([iou_matrix(boxes_a[k:k + 1], boxes_b[k:k + 1])[0, 0] for k in range(len(t))])
    return PairSeries(a, b, t, ia, ib, dist / size, closing / size, iou)


def candidate_pairs(tracks: list[Track]) -> list[PairSeries]:
    """Pairs that coexist in time and come within MAX_PAIR_DIST_REL sizes of each other.

    Cheap prefilters (time overlap, trajectory bounding boxes) keep this near-linear
    on busy junctions with hundreds of tracks.
    """
    order = sorted(tracks, key=lambda tr: tr.start)
    boxes = {id(tr): (tr.xy.min(axis=0), tr.xy.max(axis=0), float(tr.size.max())) for tr in order}
    out = []
    for i, a in enumerate(order):
        amin, amax, asz = boxes[id(a)]
        for b in order[i + 1:]:
            if b.start > a.end - 0.5:
                break
            bmin, bmax, bsz = boxes[id(b)]
            gap = np.maximum(0.0, np.maximum(amin - bmax, bmin - amax))
            if np.hypot(*gap) > MAX_PAIR_DIST_REL * max(asz, bsz):
                continue
            ps = pair_series(a, b)
            if ps is not None and ps.dist_rel.min() < MAX_PAIR_DIST_REL:
                out.append(ps)
    return out


def rel_speed(tr: Track) -> np.ndarray:
    return tr.speed / np.maximum(tr.size, 1.0)


def heading_change(tr: Track, i: int, j: int) -> float:
    """Absolute heading change (radians) between observations i and j."""
    h = np.arctan2(tr.vel[[i, j], 1], tr.vel[[i, j], 0])
    return float(abs(np.angle(np.exp(1j * (h[1] - h[0])))))


def stable_size(tr: Track, t0: float, max_ratio: float = 1.4, window: float = 1.0) -> bool:
    """False when the box size jumps around t0 - partial occlusion or merged detections,
    the usual way tracking artefacts imitate sudden braking."""
    sizes = tr.size[(tr.ts >= t0 - window) & (tr.ts <= t0 + window)]
    return len(sizes) >= 3 and sizes.max() / max(sizes.min(), 1.0) <= max_ratio


RIDER_IOU = 0.3
ROAD_EDGE_MARGIN_PX = 12.0


def rider_mask(ctx, person: Track) -> np.ndarray:
    """True at observations where this "person" overlaps a bicycle/motorcycle (a rider)."""
    riders = [v for v in ctx.tracks.vehicles() if v.kind in ("bicycle", "motorcycle")
              and v.start <= person.end and v.end >= person.start]
    mask = np.zeros(len(person.ts), bool)
    for k, t in enumerate(person.ts):
        box = person.box[k][None]
        mask[k] = any(v.start <= t <= v.end and iou_matrix(box, v.box[v.at(t)][None])[0, 0] > RIDER_IOU
                      for v in riders)
    return mask


ROAD_ENTER_PX = 24.0        # hysteresis for a track's samples: enter the road this far past the edge...
ROAD_EXIT_PX = 4.0          # ...and leave only when back within this distance of it


def hysteresis(values: np.ndarray, enter: float, leave: float) -> np.ndarray:
    """True from the first sample >= enter until the next sample < leave (samples in time order)."""
    out = np.zeros(len(values), bool)
    inside = False
    for k, v in enumerate(values):
        inside = v >= (leave if inside else enter)
        out[k] = inside
    return out


def on_road(ctx, xy: np.ndarray, track_order: bool = False) -> np.ndarray:
    """True where a ground point is well inside the drawn road area (carriageways + junction box),
    at least ROAD_EDGE_MARGIN_PX from its edge so kerbside people do not count, and on cells that
    vehicles actually drive through. Falls back to the learned carriageway mask when no layout is drawn.

    With track_order=True, `xy` is one track's samples in time order and the edge test uses
    hysteresis (ROAD_ENTER_PX in, ROAD_EXIT_PX out) instead of the single margin: someone walking
    along the kerb, whose distance to the edge hovers around the margin, would otherwise flicker
    on and off the road with every pixel of detector noise - i.e. differently on every machine.
    """
    areas = [c["poly"] for c in ctx.layout.carriageways] + list(ctx.layout.road_areas)
    if ctx.layout.intersection is not None:
        areas.append(ctx.layout.intersection)
    if not areas:
        return ctx.scene.road(xy)
    depth = np.array([max(cv2.pointPolygonTest(poly, (float(x), float(y)), True) for poly in areas)
                      for x, y in xy], float)
    drawn = hysteresis(depth, ROAD_ENTER_PX, ROAD_EXIT_PX) if track_order else depth >= ROAD_EDGE_MARGIN_PX
    return drawn & ctx.scene.driven(xy)


def near_polygon(poly: np.ndarray, xy: np.ndarray, margin: float) -> np.ndarray:
    """True for points inside the polygon or within `margin` px of it."""
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -margin for x, y in xy], bool)


IN_VEHICLE_FRAC = 0.7


def in_vehicle_mask(ctx, person: Track) -> np.ndarray:
    """True where the "person" box lies mostly inside a vehicle box (a driver seen through the window)."""
    vehicles = [v for v in ctx.tracks.vehicles() if v.kind in ("car", "bus", "truck")
                and v.start <= person.end and v.end >= person.start]
    mask = np.zeros(len(person.ts), bool)
    for k, t in enumerate(person.ts):
        px1, py1, px2, py2 = person.box[k]
        area = max((px2 - px1) * (py2 - py1), 1.0)
        for v in vehicles:
            if not v.start <= t <= v.end:
                continue
            vx1, vy1, vx2, vy2 = v.box[v.at(t)]
            inter = max(0.0, min(px2, vx2) - max(px1, vx1)) * max(0.0, min(py2, vy2) - max(py1, vy1))
            if inter / area >= IN_VEHICLE_FRAC:
                mask[k] = True
                break
    return mask
