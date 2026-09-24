"""Rule-based event detectors. Each module exposes ``detect(ctx) -> list[Event]``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..detector import Detections
from ..scene import Layout, SceneModel
from ..signal import SignalSeries
from ..tracks import TrackSet


@dataclass(frozen=True)
class Context:
    """Everything a rule may look at for one video (Part A: whole video is allowed)."""
    tracks: TrackSet
    scene: SceneModel
    layout: Layout
    thumbs: np.ndarray
    thumb_times: np.ndarray
    duration: float
    signals: dict[str, SignalSeries]
    fire: list[tuple[float, Detections]]
    transform: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0, 0], [0, 1.0, 0]]))  # reference -> video
