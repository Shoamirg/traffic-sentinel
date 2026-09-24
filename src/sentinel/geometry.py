"""2-D geometry on ground points: line sides, crossings, polygon membership."""
from __future__ import annotations

import numpy as np

from .scene import inside


def signed_side(line: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Signed distance of points to the infinite line through line[0] -> line[1]."""
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    return (xy - a) @ n


def within_extent(line: np.ndarray, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """True where the point projects onto the segment (with a margin in pixels)."""
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    d = b - a
    L = np.linalg.norm(d) + 1e-9
    proj = (xy - a) @ (d / L)
    return (proj >= -margin) & (proj <= L + margin)


def crossings(ts: np.ndarray, side: np.ndarray, valid: np.ndarray) -> list[tuple[float, int]]:
    """Times where `side` changes sign between consecutive valid samples: (t, +1|-1)."""
    out = []
    idx = np.flatnonzero(valid)
    for i, j in zip(idx, idx[1:]):
        if side[i] == 0 or np.sign(side[i]) == np.sign(side[j]):
            continue
        frac = abs(side[i]) / (abs(side[i]) + abs(side[j]))
        out.append((float(ts[i] + frac * (ts[j] - ts[i])), int(np.sign(side[j]))))
    return out


def polyline_side(polyline: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(signed distance to the nearest segment, point projects onto the polyline) per point."""
    best = np.full(len(xy), np.inf)
    side = np.zeros(len(xy))
    on = np.zeros(len(xy), bool)
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = np.array([a, b])
        s = signed_side(seg, xy)
        ok = within_extent(seg, xy)
        closer = ok & (np.abs(s) < best)
        best[closer] = np.abs(s[closer])
        side[closer] = s[closer]
        on |= ok
    return side, on


def enter_leave(ts: np.ndarray, flags: np.ndarray) -> list[tuple[float, float]]:
    """Intervals where flags is True (contiguous samples)."""
    out, start = [], None
    for t, f in zip(ts, flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            out.append((start, prev))
            start = None
        prev = t
    if start is not None:
        out.append((start, ts[-1]))
    return out


__all__ = ["signed_side", "within_extent", "crossings", "polyline_side", "enter_leave", "inside"]
