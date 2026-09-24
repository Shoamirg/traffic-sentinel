"""Export EDA, results and annotated renders of the sample videos for the website.

    python tools/publish_site_data.py --videos samples --cache cache \
        --pred predictions_samples.json [--report cache/dev_predictions.report.json] [--no-render]

Writes website/static/data/{eda,videos}.json (+ metrics.json when a dev report
is given) and website/static/media/* (annotated mp4, poster, heatmap,
trajectories, learned lane directions).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel import config, render  # noqa: E402
from sentinel.demo import counts_per_second  # noqa: E402
from sentinel.extract import Extraction, load  # noqa: E402
from sentinel.scene import SceneModel  # noqa: E402
from sentinel.segments import Event  # noqa: E402

SITE = ROOT / "website" / "static"
RISK_HZ = 5.0


def background(ex: Extraction) -> np.ndarray:
    bg = np.median(ex.thumbs[:: max(1, len(ex.thumbs) // 120)], axis=0).astype(np.uint8)
    w, h = ex.tracks.frame_size
    return cv2.resize(bg, (w, h))


def heatmap(ex: Extraction, bg: np.ndarray) -> np.ndarray:
    h, w = bg.shape[:2]
    acc = np.zeros((h // 4, w // 4), np.float32)
    for tr in ex.tracks.usable():
        moving = tr.speed > config.MIN_MOVING_SPEED
        pts = (tr.xy[moving] / 4).astype(int)
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < w // 4) & (pts[:, 1] >= 0) & (pts[:, 1] < h // 4)
        np.add.at(acc, (pts[ok, 1], pts[ok, 0]), 1)
    acc = cv2.GaussianBlur(np.log1p(acc), (0, 0), 3)
    acc = (acc / max(acc.max(), 1e-6) * 255).astype(np.uint8)
    colour = cv2.applyColorMap(cv2.resize(acc, (w, h)), cv2.COLORMAP_INFERNO)
    alpha = (cv2.resize(acc, (w, h)) / 255.0)[..., None] * 0.85
    return (bg * (1 - alpha) * 0.7 + colour * alpha).astype(np.uint8)


def trajectories(ex: Extraction, bg: np.ndarray) -> np.ndarray:
    out = (bg * 0.45).astype(np.uint8)
    for tr in ex.tracks.usable():
        if len(tr.xy) < 10:
            continue
        colour = render.KIND_COLOURS.get(tr.kind, (200, 200, 200))
        cv2.polylines(out, [tr.xy.astype(np.int32)], False, colour, 1, cv2.LINE_AA)
    return out


def directions(ex: Extraction, bg: np.ndarray) -> np.ndarray:
    w, h = ex.tracks.frame_size
    scene = SceneModel.empty(w, h).add_tracks(ex.tracks.usable(), 1.0 / config.PART_A_TARGET_FPS)
    out = (bg * 0.5).astype(np.uint8)
    cw, ch = w / config.GRID_COLS, h / config.GRID_ROWS
    for r in range(config.GRID_ROWS):
        for c in range(config.GRID_COLS):
            hist = scene.headings[r, c]
            if hist.sum() < config.ROAD_MIN_VISITS:
                continue
            ang = hist.argmax() * 2 * np.pi / config.HEADING_BINS
            cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
            tip = (int(cx + 0.45 * cw * np.cos(ang)), int(cy + 0.45 * ch * np.sin(ang)))
            hue = int(ang / (2 * np.pi) * 180)
            colour = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(out, (int(cx), int(cy)), tip, colour, 2, cv2.LINE_AA, tipLength=0.4)
    return out


def brightness(ex: Extraction) -> dict:
    v = [float(cv2.cvtColor(t, cv2.COLOR_BGR2HSV)[..., 2].mean()) for t in ex.thumbs]
    return {"t": [round(float(t), 1) for t in ex.thumb_times], "v": [round(x, 1) for x in v]}


def downsample_risk(risk: list, hz: float = RISK_HZ) -> list:
    out, next_t = [], 0.0
    for t, s in risk:
        if t >= next_t:
            out.append([round(t, 2), round(s, 3)])
            next_t = t + 1.0 / hz
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--findings", default=None, help="JSON list of {title, text} EDA findings")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())["videos"]
    media = SITE / "media"
    media.mkdir(parents=True, exist_ok=True)
    eda_rows, video_rows, totals = [], [], Counter()
    for vid in sorted(pred):
        ex = load(Path(args.cache) / f"{vid}.pkl")
        stem = Path(vid).stem
        bg = background(ex)
        for name, img in (("heatmap", heatmap(ex, bg)), ("traj", trajectories(ex, bg)), ("dirs", directions(ex, bg))):
            cv2.imwrite(str(media / f"{stem}_{name}.jpg"), cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(media / f"{stem}.jpg"), cv2.resize(bg, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        counts = counts_per_second(ex.tracks, ex.tracks.duration)
        totals.update(Counter(tr.kind for tr in ex.tracks.usable()))
        video_path = Path(args.videos) / vid
        eda_rows.append({
            "id": stem, "resolution": [ex.meta.width, ex.meta.height], "fps": ex.meta.fps,
            "duration": round(ex.tracks.duration, 1),
            "size_mb": round(video_path.stat().st_size / 1e6, 1) if video_path.exists() else None,
            "brightness": brightness(ex), "counts": counts,
            "heatmap": f"media/{stem}_heatmap.jpg", "trajectories": f"media/{stem}_traj.jpg",
            "directions": f"media/{stem}_dirs.jpg",
        })
        events = [Event(s, e, lab) for s, e, lab in pred[vid]["events"]]
        risk = pred[vid].get("risk", [])
        if not args.no_render and video_path.exists():
            render.render(str(video_path), media / f"{stem}_annotated.mp4", ex.tracks, events, risk, out_fps=12.5)
        video_rows.append({
            "id": stem, "title": vid, "duration": round(ex.tracks.duration, 2), "fps": ex.meta.fps,
            "width": ex.meta.width, "height": ex.meta.height, "video": f"media/{stem}_annotated.mp4",
            "poster": f"media/{stem}.jpg", "events": pred[vid]["events"], "risk": downsample_risk(risk),
            "counts": counts,
        })
        print(f"[{vid}] exported")

    findings = json.loads(Path(args.findings).read_text()) if args.findings else []
    data = SITE / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "eda.json").write_text(json.dumps({"videos": eda_rows, "findings": findings, "totals": dict(totals)}))
    (data / "videos.json").write_text(json.dumps(video_rows))
    if args.report:
        rep = json.loads(Path(args.report).read_text())
        per_class = {c: {"f1@0.3": v["0.3"]["f1"], "f1@0.5": v["0.5"]["f1"], "f1@0.7": v["0.7"]["f1"],
                         "tp": v["0.5"]["tp"], "fp": v["0.5"]["fp"], "fn": v["0.5"]["fn"]}
                     for c, v in rep["part_a"]["per_class"].items()}
        metrics_path = data / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics.pop("placeholder", None)
        metrics["dev_set"] = {"score_a": rep["part_a"]["score_a"],
                              "score_b": (rep.get("part_b") or {}).get("score_b"),
                              "model_score": rep["model_score"], "per_class": per_class}
        metrics_path.write_text(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
