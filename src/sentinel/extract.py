"""Part A perception pass: video -> tracks, signal lamps, appearance samples.

One pass over the video at ~PART_A_TARGET_FPS: batched YOLO, then the tracker.
On the first frame the camera is registered against scene/reference.jpg so the
hand-drawn layout can be mapped into this video; signal lamps are then read
from full-resolution crops on every analysed frame. Alongside, 1 thumbnail per
second (obstacles, EDA) and fire/smoke detections are kept. Results are
pickled so rule tuning does not re-run the GPU.
"""
from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config, register
from .detector import Detections, Detector
from .models import fire_detector, shared_detector
from .scene import Layout
from .signal import lamp_score, locate_lamps
from .tracker import ByteTracker
from .tracks import Track, TrackSet
from .video import VideoMeta, iter_frames, read_meta, stride_for

THUMB_WIDTH = 480
THUMB_EVERY_SEC = 1.0


@dataclass
class Extraction:
    meta: VideoMeta
    tracks: TrackSet
    thumbs: np.ndarray        # (K, h, w, 3) uint8 BGR, one per THUMB_EVERY_SEC
    thumb_times: np.ndarray   # (K,)
    lamp_scores: dict[str, dict[str, np.ndarray]]   # signal -> lamp -> score per analysed frame
    fire: list[tuple[float, Detections]]            # (t, fire/smoke detections), every FIRE_EVERY_SEC
    frame_dets: list[tuple[float, Detections]]      # (t, road-user detections) per analysed frame
    transform: np.ndarray = field(default_factory=lambda: register.IDENTITY.copy())  # reference -> video


class _Batcher:
    """Collects frames and runs a detector on full batches."""

    def __init__(self, model: Detector, sink: Callable[[float, Detections], None]) -> None:
        self.model, self.sink, self.items = model, sink, []

    def add(self, t: float, frame: np.ndarray) -> None:
        self.items.append((t, frame))
        if len(self.items) >= config.DETECTOR_BATCH:
            self.flush()

    def flush(self) -> None:
        if self.items:
            for (t, _), d in zip(self.items, self.model([f for _, f in self.items])):
                self.sink(t, d)
            self.items = []


class _LampProbe:
    """Registers the camera on the first frame, then scores every signal lamp per frame."""

    def __init__(self, layout: Layout, scale: float) -> None:
        self.layout, self.scale = layout, scale
        self.transform = register.IDENTITY.copy()
        self.rois: dict[str, dict[str, list[int]]] = {}
        self.scores: dict[str, dict[str, list[float]]] = {}

    def setup(self, work: np.ndarray, full: np.ndarray) -> None:
        self.transform = register.estimate(work)
        mapped = register.transform_layout(self.layout, self.transform)
        self.rois = {n: locate_lamps(full, r, self.scale, n) for n, r in mapped.signals.items()}
        self.scores = {n: {lamp: [] for lamp in r} for n, r in self.rois.items()}

    def __call__(self, full: np.ndarray) -> None:
        for name, rois in self.rois.items():
            for lamp, roi in rois.items():
                self.scores[name][lamp].append(lamp_score(full, roi, lamp))


def _build_tracks(raw: dict[int, dict], frame_times: list[float], size: tuple[int, int],
                  duration: float) -> TrackSet:
    tracks = {}
    for tid, rec in raw.items():
        kind = config.COCO_TO_TYPE[Counter(rec["cls"]).most_common(1)[0][0]]
        tracks[tid] = Track(tid=tid, kind=kind, t=rec["t"], box=rec["box"]).finalize()
    return TrackSet(tracks, np.asarray(frame_times), size, duration)


def extract(video_path: str, detector: Detector | None = None, layout: Layout | None = None,
            target_fps: float = config.PART_A_TARGET_FPS,
            progress: Callable[[float], None] | None = None,
            max_seconds: float | None = None) -> Extraction:
    meta = read_meta(video_path)
    stride = stride_for(meta.fps, target_fps)
    thumb_stride = max(1, round(THUMB_EVERY_SEC * meta.fps / stride))
    fire_stride = max(1, round(config.FIRE_EVERY_SEC * meta.fps / stride))
    total = min(meta.n_frames, max_seconds * meta.fps) if max_seconds else meta.n_frames

    tracker = ByteTracker()
    raw: dict[int, dict] = {}
    frame_dets: list[tuple[float, Detections]] = []
    fire: list[tuple[float, Detections]] = []

    def on_dets(t: float, d: Detections) -> None:
        frame_dets.append((t, d))
        for tid, box, cls, _conf in tracker.update(d, t):
            rec = raw.setdefault(tid, {"t": [], "box": [], "cls": []})
            rec["t"].append(t)
            rec["box"].append(box.copy())
            rec["cls"].append(cls)

    roads = _Batcher(detector or shared_detector(), on_dets)
    fire_model = fire_detector()
    fires = _Batcher(fire_model, lambda t, d: fire.append((t, d))) if fire_model is not None else None
    lamps = _LampProbe(layout or Layout.load(), meta.scale)

    frame_times: list[float] = []
    thumbs, thumb_times = [], []
    size = (0, 0)
    for n, (idx, t, frame, full) in enumerate(iter_frames(meta, stride, max_seconds, keep_full=True)):
        if n == 0:
            lamps.setup(frame, full)
        size = (frame.shape[1], frame.shape[0])
        frame_times.append(t)
        lamps(full)
        if n % thumb_stride == 0:
            h = round(frame.shape[0] * THUMB_WIDTH / frame.shape[1])
            thumbs.append(cv2.resize(frame, (THUMB_WIDTH, h), interpolation=cv2.INTER_AREA))
            thumb_times.append(t)
        roads.add(t, frame)
        if fires is not None and n % fire_stride == 0:
            fires.add(t, frame)
        if progress is not None and n % 50 == 0 and total:
            progress(min(1.0, idx / total))
    roads.flush()
    if fires is not None:
        fires.flush()

    duration = meta.duration or (frame_times[-1] if frame_times else 0.0)
    if max_seconds:
        duration = min(duration, max_seconds)
    return Extraction(
        meta=meta,
        tracks=_build_tracks(raw, frame_times, size, duration),
        thumbs=np.stack(thumbs) if thumbs else np.zeros((0, 1, 1, 3), np.uint8),
        thumb_times=np.asarray(thumb_times),
        lamp_scores={n: {k: np.asarray(v) for k, v in d.items()} for n, d in lamps.scores.items()},
        fire=fire,
        frame_dets=frame_dets,
        transform=lamps.transform,
    )


def save(ex: Extraction, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ex, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> Extraction:
    with open(path, "rb") as f:
        return pickle.load(f)
