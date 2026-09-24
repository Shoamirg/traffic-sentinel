"""Scene knowledge: a learned direction field plus hand-drawn layout geometry.

``SceneModel`` is learned from vehicle trajectories: a coarse grid where each cell
stores how often vehicles pass (the carriageway mask), a histogram of their
headings (the lane directions), and how long vehicles stand still there (queue
zones at signals). A prior is built offline from the sample videos
(``scene/prior.npz``) and is blended with the test video's own tracks.

``Layout`` holds geometry drawn by hand from the camera view (crossings, stop
lines, solid markings, signal-head ROI) in ``scene/layout.json``. Every field is
optional; rules that need a missing field are skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config
from .tracks import Track

BIN_WIDTH = 2 * np.pi / config.HEADING_BINS


def heading_bin(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    ang = np.mod(np.arctan2(vy, vx), 2 * np.pi)
    return (np.floor(ang / BIN_WIDTH + 0.5) % config.HEADING_BINS).astype(int)


@dataclass
class SceneModel:
    width: int
    height: int
    visits: np.ndarray        # (R, C) distinct moving-vehicle visits
    headings: np.ndarray      # (R, C, B) distinct visits per heading bin
    dwell: np.ndarray         # (R, C) vehicle-seconds spent stationary

    @staticmethod
    def empty(width: int, height: int) -> "SceneModel":
        r, c, b = config.GRID_ROWS, config.GRID_COLS, config.HEADING_BINS
        return SceneModel(width, height, np.zeros((r, c)), np.zeros((r, c, b)), np.zeros((r, c)))

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx = np.clip((xy[:, 0] / self.width * config.GRID_COLS).astype(int), 0, config.GRID_COLS - 1)
        cy = np.clip((xy[:, 1] / self.height * config.GRID_ROWS).astype(int), 0, config.GRID_ROWS - 1)
        return cy, cx

    # ---- learning -------------------------------------------------------------
    def add_tracks(self, tracks: list[Track], dt: float) -> "SceneModel":
        visits, headings, dwell = self.visits.copy(), self.headings.copy(), self.dwell.copy()
        for tr in tracks:
            if not tr.is_vehicle or tr.xy is None:
                continue
            cy, cx = self.cells(tr.xy)
            moving = tr.speed >= config.MIN_MOVING_SPEED
            bins = heading_bin(tr.vel[:, 0], tr.vel[:, 1])
            seen: set[tuple[int, int, int]] = set()
            for r, c, b, m in zip(cy, cx, bins, moving):
                if m and (r, c, b) not in seen:
                    seen.add((r, c, b))
                    headings[r, c, b] += 1
            for r, c in {(r, c) for r, c, _ in seen}:
                visits[r, c] += 1
            np.add.at(dwell, (cy[~moving], cx[~moving]), dt)
        return SceneModel(self.width, self.height, visits, headings, dwell)

    def blend(self, other: "SceneModel", weight: float = 1.0) -> "SceneModel":
        return SceneModel(self.width, self.height, self.visits + weight * other.visits,
                          self.headings + weight * other.headings, self.dwell + weight * other.dwell)

    # ---- queries ------------------------------------------------------------------
    def road(self, xy: np.ndarray) -> np.ndarray:
        """True where the point lies on (or next to) observed carriageway."""
        mask = cv2.dilate((self.visits >= config.ROAD_MIN_VISITS).astype(np.uint8), np.ones((3, 3), np.uint8))
        cy, cx = self.cells(xy)
        return mask[cy, cx].astype(bool)

    def driven(self, xy: np.ndarray, min_visits: int = 8) -> np.ndarray:
        """Stricter than road(): cells vehicles actually drive through (no dilation onto kerbs)."""
        cy, cx = self.cells(xy)
        return self.visits[cy, cx] >= min_visits

    def direction_share(self, xy: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(share of traffic moving like `vel`, total heading votes) at each point.

        Neighbouring cells and adjacent heading bins are pooled so that lane
        edges and slight curvature do not look like violations.
        """
        pooled = cv2.blur(self.headings.astype(np.float32), (3, 3), borderType=cv2.BORDER_REPLICATE)
        pooled = pooled + np.roll(pooled, 1, axis=2) * 0.5 + np.roll(pooled, -1, axis=2) * 0.5
        cy, cx = self.cells(xy)
        b = heading_bin(vel[:, 0], vel[:, 1])
        cell = pooled[cy, cx]                          # (N, B)
        total = cell.sum(axis=1) / 2.0                 # undo the double counting of the bin pooling
        share = cell[np.arange(len(b)), b] / np.maximum(cell.sum(axis=1), 1e-6)
        return share, total

    def queue_zone(self, xy: np.ndarray, min_dwell: float = 60.0) -> np.ndarray:
        """True where vehicles habitually stand (signal queues, parking bays)."""
        cy, cx = self.cells(xy)
        return self.dwell[cy, cx] >= min_dwell

    # ---- persistence ----------------------------------------------------------------
    def save(self, path: Path) -> None:
        np.savez_compressed(path, size=np.array([self.width, self.height]), visits=self.visits,
                            headings=self.headings, dwell=self.dwell)

    @staticmethod
    def load(path: Path) -> "SceneModel":
        z = np.load(path)
        w, h = (int(v) for v in z["size"])
        return SceneModel(w, h, z["visits"], z["headings"], z["dwell"])


def load_prior(width: int, height: int) -> SceneModel:
    path = config.SCENE_DIR / "prior.npz"
    if not path.exists():
        return SceneModel.empty(width, height)
    prior = SceneModel.load(path)
    if (prior.width, prior.height) != (width, height):
        return SceneModel(width, height, prior.visits, prior.headings, prior.dwell)  # grid is resolution-free
    return prior


# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """Hand-drawn geometry in working-frame pixels (see scene/layout.json)."""
    crossings: list[np.ndarray] = field(default_factory=list)       # polygons
    refuges: list[np.ndarray] = field(default_factory=list)         # pedestrian islands (not carriageway)
    stop_lines: list[dict] = field(default_factory=list)            # {"line": [[x,y],[x,y]], "signal": name}
    solid_lines: list[np.ndarray] = field(default_factory=list)     # polylines
    signals: dict[str, dict[str, list[int]]] = field(default_factory=dict)  # name -> {"red"|"green": [x1,y1,x2,y2]}
    no_u_turn: list[np.ndarray] = field(default_factory=list)       # polygons where U-turns are prohibited
    u_turn_allowed: list[np.ndarray] = field(default_factory=list)  # polygons where U-turns are legal
    intersection: np.ndarray | None = None                          # polygon of the junction box
    prohibited_turns: list[dict] = field(default_factory=list)      # {"from": polygon, "to": polygon}
    carriageways: list[dict] = field(default_factory=list)          # {"name", "poly"}: one-way road sections
    road_areas: list[np.ndarray] = field(default_factory=list)      # other drivable surface (pedestrian rules)

    @staticmethod
    def load(path: Path | None = None) -> "Layout":
        path = path or config.SCENE_DIR / "layout.json"
        if not path.exists():
            return Layout()
        d = json.loads(path.read_text())
        arr = lambda ps: [np.asarray(p, np.float32) for p in ps]  # noqa: E731
        return Layout(
            crossings=arr(d.get("crossings", [])),
            refuges=arr(d.get("refuges", [])),
            stop_lines=d.get("stop_lines", []),
            solid_lines=arr(d.get("solid_lines", [])),
            signals=d.get("signals", {}),
            no_u_turn=arr(d.get("no_u_turn", [])),
            u_turn_allowed=arr(d.get("u_turn_allowed", [])),
            intersection=np.asarray(d["intersection"], np.float32) if d.get("intersection") else None,
            prohibited_turns=[{"from": np.asarray(m["from"], np.float32), "to": np.asarray(m["to"], np.float32)}
                              for m in d.get("prohibited_turns", [])],
            carriageways=[{"name": c["name"], "poly": np.asarray(c["poly"], np.float32)}
                          for c in d.get("carriageways", [])],
            road_areas=arr(d.get("road_areas", [])),
        )


def inside(poly: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon test."""
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in xy], bool)
