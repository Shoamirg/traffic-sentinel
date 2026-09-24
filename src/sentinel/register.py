"""Camera registration: map scene knowledge drawn on the reference view into a video.

The camera is fixed but was re-mounted between recording days (samples differ
by up to ~40 px). Everything we hand-draw (scene/layout.json) or learn offline
(scene/prior.npz, scene/background.jpg) lives in the coordinates of
scene/reference.jpg. For each video we estimate a similarity transform
reference -> video from ORB features (RANSAC), then move the layout, the prior
grids and the background into the video's own pixel frame. Tracks stay native.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from . import config
from .scene import Layout, SceneModel

REFERENCE = "reference.jpg"
MIN_INLIERS = 40
MAX_SHIFT_PX = 150.0          # larger estimated motion is treated as a failed match
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def load_reference() -> np.ndarray | None:
    path = config.SCENE_DIR / REFERENCE
    return cv2.imread(str(path)) if path.exists() else None


def estimate(frame: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """2x3 similarity transform mapping reference pixels to `frame` pixels (identity on failure)."""
    reference = load_reference() if reference is None else reference
    if reference is None or frame is None:
        return IDENTITY.copy()
    if reference.shape[:2] != frame.shape[:2]:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(_gray(reference), None)
    k2, d2 = orb.detectAndCompute(_gray(frame), None)
    if d1 is None or d2 is None:
        return IDENTITY.copy()
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < MIN_INLIERS:
        return IDENTITY.copy()
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=5000, confidence=0.999, refineIters=20)
    if M is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return IDENTITY.copy()
    if np.hypot(M[0, 2], M[1, 2]) > MAX_SHIFT_PX or abs(np.hypot(M[0, 0], M[1, 0]) - 1) > 0.15:
        return IDENTITY.copy()
    return M


def apply_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


def _box(M: np.ndarray, box: list[float]) -> list[int]:
    x1, y1, x2, y2 = box
    c = apply_points(M, [[(x1 + x2) / 2, (y1 + y2) / 2]])[0]
    s = float(np.hypot(M[0, 0], M[1, 0]))
    hw, hh = (x2 - x1) * s / 2, (y2 - y1) * s / 2
    return [int(round(c[0] - hw)), int(round(c[1] - hh)), int(round(c[0] + hw)), int(round(c[1] + hh))]


def transform_layout(layout: Layout, M: np.ndarray) -> Layout:
    polys = lambda ps: [apply_points(M, p) for p in ps]  # noqa: E731
    return replace(
        layout,
        crossings=polys(layout.crossings),
        refuges=polys(layout.refuges),
        solid_lines=polys(layout.solid_lines),
        no_u_turn=polys(layout.no_u_turn),
        u_turn_allowed=polys(layout.u_turn_allowed),
        intersection=apply_points(M, layout.intersection) if layout.intersection is not None else None,
        stop_lines=[{**sl, "line": apply_points(M, sl["line"]).tolist()} for sl in layout.stop_lines],
        signals={n: {lamp: _box(M, roi) for lamp, roi in rois.items()} for n, rois in layout.signals.items()},
        prohibited_turns=[{"from": apply_points(M, m["from"]), "to": apply_points(M, m["to"])}
                          for m in layout.prohibited_turns],
        carriageways=[{**c, "poly": apply_points(M, c["poly"])} for c in layout.carriageways],
        road_areas=polys(layout.road_areas),
    )


def transform_scene(scene: SceneModel, M: np.ndarray) -> SceneModel:
    """Warp the prior's grids (reference cells -> video cells). Headings are unchanged by a small similarity."""
    if np.allclose(M, IDENTITY):
        return scene
    sx, sy = config.GRID_COLS / scene.width, config.GRID_ROWS / scene.height
    Mg = M.copy()
    Mg[0, 2] *= sx
    Mg[1, 2] *= sy
    size = (config.GRID_COLS, config.GRID_ROWS)
    warp = lambda a: cv2.warpAffine(a.astype(np.float32), Mg, size, flags=cv2.INTER_NEAREST)  # noqa: E731
    headings = np.stack([warp(scene.headings[..., b]) for b in range(scene.headings.shape[2])], axis=2)
    return SceneModel(scene.width, scene.height, warp(scene.visits), headings, warp(scene.dwell))


def invert(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M)
