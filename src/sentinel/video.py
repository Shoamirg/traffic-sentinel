"""Video metadata and strided frame reading.

Two readers yield the same (index, t, working frame[, full frame]) tuples:

* ``iter_frames`` - OpenCV, every frame decoded; skipped ones use ``grab()``.
* ``iter_ref_frames`` - PyAV with ``skip_frame=NONREF``: frames nothing else
  references (the B-frames of the camera's I-B-B-P GOP) are never decoded, the
  scaler writes straight to working size, and the full-resolution frame is a
  lazy view that converts only the slices it is asked for (signal lamps).
  At 140 Mbit/s 10-bit 4:2:2 decoding is CPU-bound, and the organisers'
  harness already decodes every frame once more for Part B, so Part A's decode
  is the part of the time budget we control. Falls back to OpenCV when PyAV is
  missing or cannot open the file.
"""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Iterator

import cv2
import numpy as np

try:  # optional fast path; OpenCV is the fallback
    import av
except ImportError:  # pragma: no cover
    av = None

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


# BT.601 limited-range YUV -> BGR, the matrix OpenCV's FFmpeg backend applies,
# so lamp crops read the same colours as the OpenCV path.
_KR, _KB = 0.299, 0.114


class LazyFullFrame:
    """Full-resolution BGR view of a planar YUV frame; slicing converts only that region."""

    def __init__(self, frame) -> None:
        self._frame = frame
        self.shape = (frame.height, frame.width, 3)
        self._bits = frame.format.components[0].bits
        chroma = frame.planes[1]
        self._sub_x = max(1, round(frame.width / chroma.width))
        self._sub_y = max(1, round(frame.height / chroma.height))
        dtype = np.uint16 if self._bits > 8 else np.uint8
        self._planes = []
        for plane in frame.planes[:3]:
            row = plane.line_size // np.dtype(dtype).itemsize
            self._planes.append(np.frombuffer(plane, dtype=dtype).reshape(-1, row)[:plane.height])

    def __getitem__(self, key) -> np.ndarray:
        ys, xs = key[0], key[1]
        y0, y1, _ = ys.indices(self.shape[0])
        x0, x1, _ = xs.indices(self.shape[1])
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0, 3), np.uint8)
        peak = float((1 << self._bits) - 1)
        scale = peak / 255.0
        yy = self._planes[0][y0:y1, x0:x1].astype(np.float32) / scale
        cy = np.arange(y0, y1) // self._sub_y
        cx = np.arange(x0, x1) // self._sub_x
        u = self._planes[1][np.ix_(cy, cx)].astype(np.float32) / scale - 128.0
        v = self._planes[2][np.ix_(cy, cx)].astype(np.float32) / scale - 128.0
        y = (yy - 16.0) * (255.0 / 219.0)
        u *= 255.0 / 224.0
        v *= 255.0 / 224.0
        r = y + 2 * (1 - _KR) * v
        b = y + 2 * (1 - _KB) * u
        g = (y - _KR * r - _KB * b) / (1 - _KR - _KB)
        return np.clip(np.stack([b, g, r], axis=-1) + 0.5, 0, 255).astype(np.uint8)


def iter_ref_frames(meta: VideoMeta, stride: int, max_seconds: float | None = None,
                    keep_full: bool = False) -> Iterator[tuple]:
    """Like iter_frames, but decodes reference frames only and keeps one per `stride` frames of time.

    Kept frames sit on the stream's reference-frame grid (e.g. indices 2, 5, 8, ...
    for I-B-B-P at stride 3); when references are sparser than `stride` every one is kept.
    """
    if av is None:
        yield from iter_frames(meta, stride, max_seconds, keep_full)
        return
    try:
        container = av.open(meta.path)
    except Exception:  # unreadable by PyAV: fall back rather than fail the video
        yield from iter_frames(meta, stride, max_seconds, keep_full)
        return
    last = int(max_seconds * meta.fps) if max_seconds else None
    width = round(meta.width * meta.scale)
    height = round(meta.height * meta.scale)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "NONREF"
        tb = float(stream.time_base)
        origin = stream.start_time or 0
        next_idx = 0
        for frame in container.decode(stream):
            idx = round((frame.pts - origin) * tb * meta.fps) if frame.pts is not None else next_idx
            if last is not None and idx >= last:
                break
            if idx < next_idx:
                continue
            next_idx = idx + stride
            work = frame.reformat(width=width, height=height, format="bgr24",
                                  interpolation="AREA").to_ndarray()
            t = idx / meta.fps
            yield (idx, t, work, LazyFullFrame(frame)) if keep_full else (idx, t, work)
    finally:
        container.close()


_DONE = object()


def prefetch(frames: Iterator[tuple], depth: int = 8) -> Iterator[tuple]:
    """Run a frame iterator in a background thread, `depth` items ahead of the consumer.

    Decoding (FFmpeg, GIL released) then overlaps with detection on the GPU
    instead of alternating with it. Order is preserved, so results are unchanged;
    an exception in the reader is re-raised in the consumer.
    """
    buf: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def work() -> None:
        try:
            for item in frames:
                while not stop.is_set():
                    try:
                        buf.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            buf.put(_DONE)
        except BaseException as exc:  # handed to the consumer
            buf.put(exc)

    thread = threading.Thread(target=work, name="frame-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buf.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)
