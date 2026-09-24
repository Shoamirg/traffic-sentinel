"""Video metadata and strided frame reading.

Skipped frames use ``grab()`` (decode without the BGR conversion/copy), which is
the cheapest way to stride through H.264 with OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def scale(self) -> float:
        """Factor from source pixels to working-frame pixels."""
        return min(1.0, config.WORK_WIDTH / self.width) if self.width else 1.0


def read_meta(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        return VideoMeta(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 25.0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def stride_for(fps: float, target_fps: float) -> int:
    return max(1, round(fps / target_fps))


def resize_to_work(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def iter_frames(meta: VideoMeta, stride: int, max_seconds: float | None = None,
                keep_full: bool = False) -> Iterator[tuple]:
    """Yield (frame_index, t_sec, working-size BGR frame) for every stride-th frame.

    With keep_full=True the full-resolution frame is appended to each tuple.
    """
    last = int(max_seconds * meta.fps) if max_seconds else None
    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {meta.path}")
    idx = 0
    try:
        while last is None or idx < last:
            if idx % stride:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                work = resize_to_work(frame, meta.scale)
                yield (idx, idx / meta.fps, work, frame) if keep_full else (idx, idx / meta.fps, work)
            idx += 1
    finally:
        cap.release()
