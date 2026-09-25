"""YOLO road-user detector (Ultralytics, open weights)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass(frozen=True)
class Detections:
    """Detections for one frame, in working-frame pixels."""
    xyxy: np.ndarray   # (N, 4) float32
    conf: np.ndarray   # (N,) float32
    cls: np.ndarray    # (N,) int, COCO ids

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))


def _device() -> str:
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


ROAD_USER_CLASSES = tuple(sorted(config.COCO_TO_TYPE))


class Detector:
    def __init__(self, weights: str = config.DETECTOR_WEIGHTS, imgsz: int = config.DETECTOR_IMGSZ,
                 conf: float = config.DETECTOR_CONF, classes: tuple[int, ...] | None = ROAD_USER_CLASSES) -> None:
        # the evaluation runs offline: skip Ultralytics' online check and usage analytics,
        # which would otherwise open network connections (and wait on timeouts) at import
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        import torch

        torch.manual_seed(config.SEED)
        torch.backends.cudnn.benchmark = False       # fixed conv algorithms -> repeatable outputs
        torch.backends.cudnn.deterministic = True
        self.model = YOLO(weights)
        self.device = _device()
        self.half = self.device.startswith("cuda")
        self.imgsz = imgsz
        self.conf = conf
        self.classes = list(classes) if classes is not None else None

    def __call__(self, frames: list[np.ndarray]) -> list[Detections]:
        if not frames:
            return []
        results = self.model.predict(
            frames, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
            device=self.device, quantize=16 if self.half else 32, verbose=False,
        )
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(Detections.empty())
                continue
            out.append(Detections(
                xyxy=b.xyxy.cpu().numpy().astype(np.float32),
                conf=b.conf.cpu().numpy().astype(np.float32),
                cls=b.cls.cpu().numpy().astype(int),
            ))
        return out
