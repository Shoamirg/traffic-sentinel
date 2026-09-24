"""Annotated video rendering: tracks, active events, risk gauge and a timeline strip.

Frames are re-read from the source, the nearest analysed observation of every
track is drawn, and the result is piped to ffmpeg (H.264, yuv420p, faststart)
so browsers can play and seek it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .segments import Event
from .tracks import TrackSet
from .video import iter_frames, read_meta, stride_for

OUT_WIDTH = 1280
STRIP_H = 46
TRAIL_SEC = 2.0

# BGR colours per event class (kept in sync with the website palette by name only)
CLASS_COLOURS = {
    "accident": (60, 60, 235), "near_miss": (40, 150, 255), "red_light": (80, 80, 200),
    "wrong_way": (200, 60, 200), "illegal_u_turn": (230, 120, 170), "stopped_vehicle": (0, 200, 255),
    "jaywalking": (80, 220, 120), "failure_to_yield": (60, 190, 190), "illegal_turn": (220, 160, 60),
    "solid_line_crossing": (255, 200, 80), "stop_line": (150, 110, 240), "congestion": (120, 120, 120),
    "road_obstacle": (40, 110, 180), "fire_smoke": (30, 60, 255),
}
KIND_COLOURS = {"person": (90, 230, 90), "bicycle": (230, 200, 60), "motorcycle": (230, 140, 60),
                "car": (240, 200, 120), "bus": (200, 120, 240), "truck": (160, 160, 250), "animal": (60, 160, 250)}


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class H264Writer:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the annotated video")


def _draw_tracks(img: np.ndarray, tracks: TrackSet, t: float, sx: float) -> None:
    for tr in tracks.active_at(t, tol=0.15):
        i = tr.at(t)
        if abs(tr.ts[i] - t) > 0.2:
            continue
        colour = KIND_COLOURS.get(tr.kind, (200, 200, 200))
        x1, y1, x2, y2 = (tr.box[i] * sx).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{tr.kind[:3]} {tr.tid}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
        past = (tr.ts >= t - TRAIL_SEC) & (tr.ts <= t)
        pts = (tr.xy[past] * sx).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, colour, 2, cv2.LINE_AA)


def _draw_events(img: np.ndarray, events: list[Event], t: float) -> None:
    active = [e for e in events if e.start <= t <= e.end]
    for k, ev in enumerate(active[:6]):
        colour = CLASS_COLOURS.get(ev.label, (255, 255, 255))
        y = 14 + k * 30
        label = f"{ev.label.replace('_', ' ').upper()}  {ev.start:.1f}-{ev.end:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (10, y), (26 + tw, y + th + 12), colour, -1)
        cv2.putText(img, label, (18, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_risk(img: np.ndarray, risk: float) -> None:
    h, w = img.shape[:2]
    x0, y0, bw, bh = w - 230, 14, 210, 22
    colour = (60, 60, 235) if risk >= 0.5 else (40, 180, 255) if risk >= 0.25 else (90, 200, 90)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y0), (x0 + int(bw * risk), y0 + bh), colour, -1)
    cv2.putText(img, f"accident risk {risk:.2f}", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def _strip(width: int, events: list[Event], duration: float) -> np.ndarray:
    strip = np.full((STRIP_H, width, 3), 24, np.uint8)
    labels = sorted({e.label for e in events})
    lane_h = max(4, (STRIP_H - 6) // max(1, len(labels)))
    for ev in events:
        k = labels.index(ev.label)
        x1 = int(ev.start / duration * width)
        x2 = max(int(ev.end / duration * width), x1 + 2)
        colour = CLASS_COLOURS.get(ev.label, (255, 255, 255))
        cv2.rectangle(strip, (x1, 3 + k * lane_h), (x2, 3 + (k + 1) * lane_h - 1), colour, -1)
    return strip


def render(video_path: str, out_path: Path, tracks: TrackSet, events: list[Event],
           risk: list[list[float]] | None = None, out_fps: float = 12.5, max_seconds: float | None = None,
           progress: Callable[[float], None] | None = None) -> Path:
    meta = read_meta(video_path)
    duration = min(meta.duration, max_seconds) if max_seconds else meta.duration
    stride = stride_for(meta.fps, out_fps)
    fps = meta.fps / stride
    risk_t = np.asarray([r[0] for r in risk]) if risk else np.zeros(0)
    risk_v = np.asarray([r[1] for r in risk]) if risk else np.zeros(0)
    writer = None
    base_strip = None
    try:
        for idx, t, frame in iter_frames(meta, stride):
            if t > duration:
                break
            sx = OUT_WIDTH / frame.shape[1]
            img = cv2.resize(frame, (OUT_WIDTH, round(frame.shape[0] * sx) // 2 * 2))
            if writer is None:
                writer = H264Writer(out_path, OUT_WIDTH, img.shape[0] + STRIP_H, fps)
                base_strip = _strip(OUT_WIDTH, events, max(duration, 1e-6))
            _draw_tracks(img, tracks, t, sx)
            _draw_events(img, events, t)
            if len(risk_t):
                _draw_risk(img, float(risk_v[min(np.searchsorted(risk_t, t), len(risk_v) - 1)]))
            strip = base_strip.copy()
            x = int(t / max(duration, 1e-6) * OUT_WIDTH)
            cv2.line(strip, (x, 0), (x, STRIP_H), (255, 255, 255), 2)
            writer.write(np.vstack([img, strip]))
            if progress is not None and idx % 100 == 0:
                progress(min(1.0, t / max(duration, 1e-6)))
    finally:
        if writer is not None:
            writer.close()
    return out_path
