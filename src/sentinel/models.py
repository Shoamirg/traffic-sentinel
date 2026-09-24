"""Process-wide model cache: weights are loaded once and shared by Part A and
Part B (the weights are shared, never their outputs)."""
from __future__ import annotations

from functools import lru_cache

from . import config
from .detector import Detector


@lru_cache(maxsize=1)
def shared_detector() -> Detector:
    return Detector()


@lru_cache(maxsize=1)
def fire_detector() -> Detector | None:
    """Fine-tuned fire/smoke YOLO (classes 0=fire, 1=smoke), or None if not shipped."""
    if not config.FIRE_WEIGHTS.exists():
        return None
    return Detector(str(config.FIRE_WEIGHTS), imgsz=config.FIRE_IMGSZ, conf=config.FIRE_CONF, classes=None)
