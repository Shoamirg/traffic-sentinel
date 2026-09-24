#!/usr/bin/env bash
# Reproduce weights/fire_smoke.pt from scratch (YOLO11s fine-tuned on D-Fire, CC0).
# Tested on NVIDIA DGX Spark (GB10, aarch64, Ubuntu 24.04), python 3.12,
# torch 2.14+cu130, ultralytics 8.4.160, opencv-python-headless, pyarrow.
#
#   WORK=~/wiut/firesmoke VENV=~/wiut/venv COCO_WEIGHTS=~/wiut/repo/weights/yolo11s.pt \
#       bash train_fire_smoke.sh
#
# Steps: 1) download the pinned HF parquet mirror of D-Fire  2) convert to YOLO
# format (class remap to 0=fire,1=smoke; seeded 10% val split from official train;
# official test kept for reporting)  3) train 75 epochs, seed=0 deterministic
# 4) evaluate on the test split  5) false-positive analysis on fire-free traffic video.
set -euo pipefail
WORK=${WORK:-$HOME/wiut/firesmoke}
VENV=${VENV:-$HOME/wiut/venv}
COCO_WEIGHTS=${COCO_WEIGHTS:-$HOME/wiut/repo/weights/yolo11s.pt}   # ultralytics yolo11s.pt (COCO)
EPOCHS=${EPOCHS:-75}
PY="$VENV/bin/python"
mkdir -p "$WORK" && cd "$WORK"

# pyarrow is only needed for the parquet conversion; keep it out of the shared venv.
[ -d pylib/pyarrow ] || "$VENV/bin/pip" install -q --target "$WORK/pylib" pyarrow

# ---------------------------------------------------------------- 1. download
cat > download.sh <<'SH'
#!/usr/bin/env bash
# Download D-Fire (CC0) parquet mirror from Hugging Face, pinned revision.
set -euo pipefail
REPO=badsaarow/d-fire
REV=27f81e6f7d32fe3b29d366da9573b6150a3d148f
DST=${DST:-$HOME/wiut/firesmoke/raw}
mkdir -p "$DST"
for f in test-00000-of-00003 test-00001-of-00003 test-00002-of-00003 \
         train-00000-of-00009 train-00001-of-00009 train-00002-of-00009 \
         train-00003-of-00009 train-00004-of-00009 train-00005-of-00009 \
         train-00006-of-00009 train-00007-of-00009 train-00008-of-00009; do
  out="$DST/$f.parquet"
  [ -s "$out" ] && { echo "have $f"; continue; }
  echo "get $f $(date +%T)"
  curl -fsSL --retry 5 -C - -o "$out.part" "https://huggingface.co/datasets/$REPO/resolve/$REV/data/$f.parquet"
  mv "$out.part" "$out"
done
echo DONE $(date +%T)
SH
DST="$WORK/raw" bash download.sh

# ---------------------------------------------------------------- 2. convert
cat > convert.py <<'PYEOF'
"""Convert the D-Fire HF parquet mirror into an Ultralytics YOLO dataset.

D-Fire class ids: 0=smoke, 1=fire. We remap to 0=fire, 1=smoke.
Official split: train (17,221) / test (4,306). We hold out a seeded 10% of
train as `val` (used for model selection) and keep `test` untouched for the
final report.
"""
import glob
import os
import random
import sys
from collections import Counter

import pyarrow.parquet as pq

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/wiut/firesmoke/raw")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/wiut/firesmoke/dfire")
VAL_FRACTION = 0.10
SEED = 0
REMAP = {"0": 1, "1": 0}  # D-Fire smoke->1, fire->0


def clean_label(text):
    lines = []
    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5 or parts[0] not in REMAP:
            raise ValueError(f"bad label line: {raw!r}")
        x, y, w, h = (min(max(float(v), 0.0), 1.0) for v in parts[1:])
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{REMAP[parts[0]]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def iter_rows(split):
    for path in sorted(glob.glob(os.path.join(RAW, f"{split}-*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()


def write(split, row, stats):
    name = os.path.splitext(os.path.basename(row["filename"]))[0]
    img_dir = os.path.join(OUT, "images", split)
    lbl_dir = os.path.join(OUT, "labels", split)
    with open(os.path.join(img_dir, name + ".jpg"), "wb") as f:
        f.write(row["image"]["bytes"])
    lines = clean_label(row["label"] or "")
    with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    stats[split, "images"] += 1
    stats[split, "negatives"] += int(not lines)
    for ln in lines:
        stats[split, "fire" if ln[0] == "0" else "smoke"] += 1


def main():
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    stats = Counter()
    train_names = sorted(r["filename"] for r in iter_rows_labels("train"))
    rng = random.Random(SEED)
    val_set = set(rng.sample(train_names, round(len(train_names) * VAL_FRACTION)))
    for row in iter_rows("train"):
        write("val" if row["filename"] in val_set else "train", row, stats)
    for row in iter_rows("test"):
        write("test", row, stats)
    for key in sorted(stats):
        print(key, stats[key])


def iter_rows_labels(split):
    for path in sorted(glob.glob(os.path.join(RAW, f"{split}-*.parquet"))):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["filename"]):
            yield from batch.to_pylist()


if __name__ == "__main__":
    main()
PYEOF
PYTHONPATH="$WORK/pylib" "$PY" convert.py "$WORK/raw" "$WORK/dfire"

cat > dfire.yaml <<YAML
# D-Fire (CC0 1.0) - https://github.com/gaiasd/DFireDataset
# HF mirror: https://huggingface.co/datasets/badsaarow/d-fire @ 27f81e6f7d32fe3b29d366da9573b6150a3d148f
path: $WORK/dfire
train: images/train
val: images/val
test: images/test
names:
  0: fire
  1: smoke
YAML

# ---------------------------------------------------------------- 3. train
"$VENV/bin/yolo" detect train \
  model="$COCO_WEIGHTS" data="$WORK/dfire.yaml" \
  epochs="$EPOCHS" imgsz=640 batch=32 workers=12 \
  optimizer=SGD lr0=0.01 cos_lr=True close_mosaic=10 patience=0 \
  seed=0 deterministic=True amp=True \
  project="$WORK/runs" name=y11s_dfire exist_ok=True plots=True
BEST="$WORK/runs/y11s_dfire/weights/best.pt"

# ---------------------------------------------------------------- 4. evaluate (held-out official test split)
"$VENV/bin/yolo" detect val model="$BEST" data="$WORK/dfire.yaml" split=test imgsz=640 batch=32 \
  project="$WORK/runs" name=test_eval exist_ok=True

# ---------------------------------------------------------------- 5. false positives on fire-free traffic
cat > fp_eval.py <<'PYEOF'
"""False-positive analysis on fire-free traffic video.

Samples one frame every STEP_S seconds (OpenCV seeking, no copies), runs the
detector at a low confidence floor, and reports how many frames contain any
fire/smoke box at several thresholds, with and without a min-area filter.
Usage: fp_eval.py WEIGHTS OUT_JSON VIDEO [VIDEO ...]
"""
import json
import os
import sys

import cv2
from ultralytics import YOLO

STEP_S = 2.0
CONF_FLOOR = 0.10
THRESHOLDS = (0.25, 0.4, 0.5, 0.6)
MIN_AREA_FRACS = (0.0, 0.001, 0.005)  # fraction of frame area


SEEK_MIN_BYTES = 500 * 1024 * 1024  # large CCTV files: seek; small clips: sequential decode


def sample_frames(path):
    if os.path.getsize(path) < SEEK_MIN_BYTES:
        yield from sample_frames_sequential(path)
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * STEP_S))
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        yield idx / fps, frame
    cap.release()


def sample_frames_sequential(path):
    """Decode every frame (robust for webm/ogv where frame counts are unreliable)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    next_t = 0.0
    while cap.grab():
        # container timestamps: webm headers often report a bogus FPS (1000, 30000)
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if ok:
            yield t, frame
        next_t += STEP_S
    cap.release()


def main():
    weights, out_json, videos = sys.argv[1], sys.argv[2], sys.argv[3:]
    model = YOLO(weights)
    names = model.names
    report = {"weights": weights, "step_s": STEP_S, "videos": {}}
    crops_dir = os.path.splitext(out_json)[0] + "_hits"
    os.makedirs(crops_dir, exist_ok=True)
    for vid in videos:
        frames = []
        for t, frame in sample_frames(vid):
            h, w = frame.shape[:2]
            res = model.predict(frame, imgsz=640, conf=CONF_FLOOR, verbose=False)[0]
            dets = []
            for (x1, y1, x2, y2), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                dets.append({"cls": names[int(k)], "conf": round(c, 4),
                             "area_frac": round((x2 - x1) * (y2 - y1) / (w * h), 6),
                             "xyxy": [round(v) for v in (x1, y1, x2, y2)]})
            frames.append({"t": round(t, 2), "dets": dets})
            if any(d["conf"] >= THRESHOLDS[0] for d in dets):
                cv2.imwrite(os.path.join(crops_dir, f"{os.path.basename(vid)}_{t:08.1f}.jpg"), res.plot())
        summary = {}
        for thr in THRESHOLDS:
            for amin in MIN_AREA_FRACS:
                hit = [f for f in frames if any(d["conf"] >= thr and d["area_frac"] >= amin for d in f["dets"])]
                summary[f"conf{thr}_area{amin}"] = {
                    "frames_with_det": len(hit),
                    "fire": sum(any(d["cls"] == "fire" and d["conf"] >= thr and d["area_frac"] >= amin for d in f["dets"]) for f in frames),
                    "smoke": sum(any(d["cls"] == "smoke" and d["conf"] >= thr and d["area_frac"] >= amin for d in f["dets"]) for f in frames),
                }
        report["videos"][vid] = {"n_frames": len(frames), "summary": summary, "frames": frames}
        print(vid, len(frames), json.dumps({k: v for k, v in summary.items() if k.endswith("area0.0")}))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
PYEOF
cat > neg_images_eval.py <<'PYEOF'
"""Image-level false-positive rate on the D-Fire *test* images that have no fire/smoke labels."""
import glob
import json
import os
import sys

from ultralytics import YOLO

THRESHOLDS = (0.25, 0.4, 0.5, 0.6)


def main():
    weights, root, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    negs = [p for p in sorted(glob.glob(os.path.join(root, "labels/test/*.txt"))) if os.path.getsize(p) == 0]
    imgs = [p.replace("/labels/", "/images/").replace(".txt", ".jpg") for p in negs]
    model = YOLO(weights)
    maxconf = []
    for i in range(0, len(imgs), 64):
        for r in model.predict(imgs[i:i + 64], imgsz=640, conf=0.1, verbose=False):
            maxconf.append(max(r.boxes.conf.tolist(), default=0.0))
    res = {"n_negative_images": len(imgs),
           "images_with_det": {str(t): sum(c >= t for c in maxconf) for t in THRESHOLDS}}
    print(json.dumps(res))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
PYEOF
cat > fetch_negvideos.py <<'PYEOF'
"""Download fire-free public road-traffic clips from Wikimedia Commons for FP testing."""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

TITLES = [
    "File:Cars driving at night.webm",
    "File:Jane M. Byrne Interchange Traffic.webm",
    "File:Road traffic – cobblestone road.webm",
    "File:West Linn Roundabout Open.webm",
    "File:Motorway A40 - on bridge above the traffic.webm",
    "File:M62 motorway from the Greystone Road footbridge.ogv",
    "File:E18 nord for Ramsum, Vestfold.webm",
    "File:Intersection Pie-IX-Sherbrooke.webm",
    "File:A intersection with traffic in Chiang Mai, Thailand.ogv",
    "File:Ayalon trafic congestion time lapse.webm",
    "File:Avtocesta.webm",
    "File:Dogbone roundabout gnangarra.ogv",
]
UA = {"User-Agent": "wiut-hackathon-firesmoke-eval/1.0 (research; contact via github)"}
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/wiut/firesmoke/negvideos")


def download(url, fname, attempts=8):
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r, open(fname, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == attempts - 1:
                raise
            time.sleep(30 * (i + 1))  # Wikimedia rate limit: back off


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for title in TITLES:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                                    "iiprop": "url|size|sha1|extmetadata",
                                    "iiextmetadatafilter": "LicenseShortName|Artist"})
        req = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + q, headers=UA)
        page = next(iter(json.load(urllib.request.urlopen(req))["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        fname = os.path.join(OUT, title[5:].replace(" ", "_").replace("/", "_"))
        if not os.path.exists(fname) or os.path.getsize(fname) != ii["size"]:
            download(ii["url"], fname)
        meta = ii["extmetadata"]
        manifest.append({"title": title, "file": fname, "url": ii["descriptionurl"], "sha1": ii["sha1"],
                         "license": meta.get("LicenseShortName", {}).get("value")})
        print(manifest[-1]["license"], title, flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
PYEOF
cat > pos_images_eval.py <<'PYEOF'
"""Image-level recall on D-Fire test images that contain fire and/or smoke, per threshold."""
import glob, json, os, sys
from ultralytics import YOLO
TH = (0.25, 0.4, 0.5, 0.6)
weights, root, out_json = sys.argv[1:4]
lbls = [p for p in sorted(glob.glob(os.path.join(root, "labels/test/*.txt"))) if os.path.getsize(p) > 0]
model = YOLO(weights)
res = {"fire": {"n": 0, **{str(t): 0 for t in TH}}, "smoke": {"n": 0, **{str(t): 0 for t in TH}}}
for i in range(0, len(lbls), 64):
    chunk = lbls[i:i + 64]
    preds = model.predict([p.replace("/labels/", "/images/").replace(".txt", ".jpg") for p in chunk], imgsz=640, conf=0.1, verbose=False)
    for lp, r in zip(chunk, preds):
        gt = {int(l.split()[0]) for l in open(lp) if l.strip()}
        for k, name in ((0, "fire"), (1, "smoke")):
            if k not in gt:
                continue
            res[name]["n"] += 1
            mx = max((c for c, cl in zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()) if int(cl) == k), default=0)
            for t in TH:
                res[name][str(t)] += mx >= t
print(json.dumps(res))
json.dump(res, open(out_json, "w"), indent=1)
PYEOF
cat > fp_summary.py <<'PYEOF'
import json, os, sys
r = json.load(open(sys.argv[1]))
TH = (0.25, 0.4, 0.5, 0.6); AR = (0.0, 0.001, 0.005)
tot = sum(v["n_frames"] for v in r["videos"].values())
print("total frames", tot)
for t in TH:
    row = []
    for a in AR:
        row.append(sum(v["summary"][f"conf{t}_area{a}"]["frames_with_det"] for v in r["videos"].values()))
    print(f"conf>={t}: frames with det (area>=0/0.1%/0.5%): {row}")
# persistence: >= K consecutive sampled frames (2 s apart) with a det
for t in TH:
    for k in (2, 3):
        n = 0
        for v in r["videos"].values():
            run = 0
            for f in v["frames"]:
                hit = any(d["conf"] >= t and d["area_frac"] >= 0.001 for d in f["dets"])
                run = run + 1 if hit else 0
                n += run == k
        print(f"conf>={t} area>=0.1% persist>={k} samples ({2*(k-1)}s+): alarm events={n}")
print("per video conf0.25/0.4/0.5/0.6 (area0):")
for vid, v in r["videos"].items():
    s = v["summary"]
    mx = max((d["conf"] for f in v["frames"] for d in f["dets"]), default=0)
    print(f"  {os.path.basename(vid)[:45]:45s} n={v['n_frames']:4d}", [s[f"conf{t}_area0.0"]["frames_with_det"] for t in TH], "max", round(mx, 3),
          "cls", sorted({d["cls"] for f in v["frames"] for d in f["dets"] if d["conf"] >= 0.25}))

# Recommended post-filter: 0.1% <= box area <= 50% of frame (whole-frame "smoke" on
# uniform grey asphalt/haze is the dominant FP), plus temporal persistence.
def ok(d, t):
    return d["conf"] >= t and 0.001 <= d["area_frac"] <= 0.5
print("\nwith area band [0.1%, 50%]:")
for t in TH:
    frames = sum(any(ok(d, t) for d in f["dets"]) for v in r["videos"].values() for f in v["frames"])
    ev = {}
    for k in (2, 3):
        n = 0
        for v in r["videos"].values():
            run = 0
            for f in v["frames"]:
                run = run + 1 if any(ok(d, t) for d in f["dets"]) else 0
                n += run == k
        ev[k] = n
    per = {os.path.basename(p)[:20]: sum(any(ok(d, t) for d in f["dets"]) for f in v["frames"]) for p, v in r["videos"].items()}
    print(f"conf>={t}: frames={frames}/{tot}  alarms(persist 2 samples/>=2s)={ev[2]}  alarms(persist 3 samples/>=4s)={ev[3]}  per-video={ {k: n for k, n in per.items() if n} }")
PYEOF
"$PY" fetch_negvideos.py "$WORK/negvideos"
"$PY" neg_images_eval.py "$BEST" "$WORK/dfire" "$WORK/neg_images_fp.json"
VIDEOS=("$WORK"/negvideos/*.webm "$WORK"/negvideos/*.ogv)
[ -f "$HOME/wiut/smoke/traffic.mp4" ] && VIDEOS+=("$HOME/wiut/smoke/traffic.mp4")
"$PY" pos_images_eval.py "$BEST" "$WORK/dfire" "$WORK/pos_images_recall.json"
"$PY" fp_eval.py "$BEST" "$WORK/fp_final.json" "${VIDEOS[@]}"
"$PY" fp_summary.py "$WORK/fp_final.json" | tee "$WORK/fp_summary.txt"
sha256sum "$BEST"
