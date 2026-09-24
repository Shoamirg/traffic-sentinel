"""Central constants: paths, detector settings, and rule thresholds.

Every tunable number lives here so ablations touch one file. Distances are in
*working-frame pixels* (frames are resized to WORK_WIDTH before anything else)
and times are in seconds.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = ROOT / "weights"
SCENE_DIR = ROOT / "scene"

SEED = 0

# ---- frame sampling ---------------------------------------------------------
WORK_WIDTH = 1280            # frames are downscaled to this width before detection
PART_A_TARGET_FPS = float(os.environ.get("SENTINEL_PART_A_FPS", 10.0))   # Part A analyses ~10 frames/s
PART_B_TARGET_FPS = 5.0      # Part B runs its causal tracker at ~5 fps

# ---- detector ---------------------------------------------------------------
DETECTOR_WEIGHTS = os.environ.get("SENTINEL_WEIGHTS", str(WEIGHTS_DIR / "yolo11m.pt"))
DETECTOR_IMGSZ = int(os.environ.get("SENTINEL_IMGSZ", 1280))
DETECTOR_CONF = 0.25
DETECTOR_BATCH = 8

# COCO ids we keep, mapped to our coarse road-user types.
COCO_TO_TYPE = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",  # cat dog horse sheep cow
}
VEHICLE_TYPES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle"})

# ---- fire / smoke detector (fine-tuned YOLO, optional) ------------------------
FIRE_WEIGHTS = WEIGHTS_DIR / "fire_smoke.pt"
FIRE_EVERY_SEC = 1.0
FIRE_CONF = 0.25             # raw detections kept; the rule applies its own stricter threshold
FIRE_IMGSZ = 640            # training resolution of weights/fire_smoke.pt

# ---- tracking ---------------------------------------------------------------
TRACK_MIN_OBS = 5            # tracks shorter than this are ignored by the rules
SMOOTH_WINDOW_SEC = 0.6      # centred smoothing window for positions

# ---- scene model (learned direction field) ----------------------------------
GRID_COLS, GRID_ROWS = 48, 27
HEADING_BINS = 8
MIN_MOVING_SPEED = 25.0      # px/s: below this a vehicle counts as stationary
ROAD_MIN_VISITS = 3          # cell visits needed to count as carriageway

# ---- segment post-processing -------------------------------------------------
MERGE_GAP_SEC = 1.0
MIN_EVENT_SEC = 0.5
