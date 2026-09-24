# fire_smoke.pt: open-weights fire and smoke detector

| | |
|---|---|
| File | `weights/fire_smoke.pt` (Ultralytics YOLO11s, 2 classes) |
| Size | 19,178,650 bytes (18.3 MiB) |
| SHA-256 | `df59c165af2b74f02fc04e94ee36a9770ed2336133a9f6e26a7d72f331b28376` |
| Classes | `0: fire`, `1: smoke` |
| Input | 640 px (letterboxed), BGR frames, via `ultralytics.YOLO` |
| Base weights | Ultralytics `yolo11s.pt` (COCO), fine-tuned |
| Reproduce | `tools/train_fire_smoke.sh` (download, conversion, training, evaluation, FP analysis) |

## Dataset

- **D-Fire**: an image dataset for fire and smoke detection, by Gaia, solutions on demand.
  - Source: https://github.com/gaiasd/DFireDataset
  - **License: CC0 1.0 Universal** (see the repository's `LICENSE`: "made available to the public domain under the Creative Commons Zero v1.0 Universal license").
  - Mirror used, pinned to a revision so the download is reproducible: https://huggingface.co/datasets/badsaarow/d-fire @ `27f81e6f7d32fe3b29d366da9573b6150a3d148f`. It holds 12 parquet shards (3.1 GB), keeps the official train/test split, and carries YOLO-format labels.
- **Citation:** P. V. A. B. de Venâncio, A. C. Lisboa, A. V. Barbosa, "An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices", *Neural Computing and Applications*, 2022. https://doi.org/10.1007/s00521-022-07467-z
- **Class remap:** D-Fire uses `0=smoke, 1=fire`. We remap to `0=fire, 1=smoke`.
- **Splits:** the official train split has 17,221 images. We hold out a seeded 10% of it (`random.Random(0)`) as val for model selection. The official test split (4,306 images) is used only for the final report.

| split | images | no fire/smoke (negatives) | fire boxes | smoke boxes |
|---|---|---|---|---|
| train | 15,499 | 7,065 | 10,569 | 8,590 |
| val | 1,722 | 768 | 1,238 | 953 |
| test | 4,306 | 2,005 | 2,878 | 2,311 |
| **total** | **21,527** | **9,838** | 14,685 | 11,854 |

## Training

The model was trained on an NVIDIA DGX Spark (GB10, aarch64) with torch 2.14+cu130 and ultralytics 8.4.160.

```bash
yolo detect train model=yolo11s.pt data=dfire.yaml \
  epochs=75 imgsz=640 batch=32 workers=12 \
  optimizer=SGD lr0=0.01 cos_lr=True close_mosaic=10 patience=0 \
  seed=0 deterministic=True amp=True \
  project=runs name=y11s_dfire exist_ok=True plots=True
```

- Seed 0 with `deterministic=True`. We trained 75 epochs; the best checkpoint by fitness is **epoch 48**, and that is the one delivered (`best.pt`).
- Training took **2.84 h** (75 epochs × about 136 s per epoch, 485 iterations at 3.8 it/s).
- Inference is about 2–3.5 ms per 640 px image on the GB10.

## Accuracy

**Held-out official test split (4,306 images):**

| class | images | boxes | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 4306 | 5189 | 0.793 | 0.725 | **0.791** | **0.460** |
| fire | 1115 | 2878 | 0.746 | 0.660 | 0.731 | 0.385 |
| smoke | 2081 | 2311 | 0.840 | 0.791 | 0.851 | 0.534 |

**Val split (1,722 images, used for model selection):** all mAP50 0.794 / mAP50-95 0.473. Fire scores 0.743 / 0.400 and smoke 0.846 / 0.546.

**Image-level recall on the test split** is the share of positive images where the model fires on the right class at or above the threshold:

| conf ≥ | 0.25 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|
| fire images (n=1115) | 95.2% | 89.4% | 82.2% | 68.7% |
| smoke images (n=2081) | 94.7% | 87.7% | 79.3% | 69.7% |

## False-positive analysis on normal traffic, with no fire present

**Test set 1: D-Fire test negatives.** These are 2,005 images with no fire or smoke. The table counts how many of them produced any detection:

| conf ≥ | 0.25 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|
| images with a detection | 38 (1.9%) | 23 (1.1%) | 11 (0.55%) | 2 (0.10%) |

**Test set 2: road-traffic video with no fire.** We sampled one frame every 2 s, 418 frames in total, from 13 clips:
- `~/wiut/smoke/traffic.mp4`: a top-down view of a parking lot or road.
- 12 CC-licensed Wikimedia Commons traffic clips: highway at night, motorway, roundabouts, intersections, a time-lapse and a cobblestone street. The list with licenses is in `fetch_negvideos.py` and `negvideos/manifest.json`.

The fixed CCTV samples `~/wiut/samples/*.MP4` had not finished downloading (only partial chunks were present), so they were **not evaluated**.

Frames with any detection, with no filtering:

| conf ≥ | 0.25 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|
| all 418 frames | 88 | 57 | 30 | 8 |
| traffic.mp4 (16) | 13 | 10 | 7 | 3 |
| Cars driving at night (80) | 48 | 30 | 14 | 0 |
| Avtocesta, night motorway (12) | 6 | 5 | 5 | 3 |
| West Linn roundabout (100) | 21 | 12 | 4 | 2 |
| other 9 clips (210) | 0 | 0 | 0 | 0 |

The false positives fall into three groups:

1. **Whole-frame "smoke" on uniform grey asphalt with faint haze** (`traffic.mp4`). The box covers about 96% of the frame.
2. **Clusters of headlight glare at night**, flagged as "fire" (0.58–0.80) or large dark-sky "smoke" boxes.
3. **A real white steam plume from an industrial chimney** in the West Linn clip (0.63). It is visually real smoke or steam, but it is not a road incident.

**With the recommended post-filter** (box area between 0.1% and 50% of the frame, plus temporal persistence), across all 418 frames:

| conf ≥ | frames with a detection | alarm events (≥ 2 consecutive samples, 2 s or more) | alarm events (≥ 3 consecutive samples, 4 s or more) |
|---|---|---|---|
| 0.25 | 74 | 8 | 7 |
| 0.4 | 47 | 8 | 7 |
| 0.5 | 23 | 4 | 3 |
| **0.6** | **5** | **1** | **0** |

## Recommended thresholds for the traffic system

- **Confidence ≥ 0.6** for alarms. Use 0.5 only if missing an incident costs more than a false alarm. Detections between 0.4 and 0.6 can be logged as "watch" events for review, but should not raise alerts.
- **Box area between 0.1% and 50% of the frame.** The lower bound drops single headlights and tail or brake lights. The upper bound drops whole-frame "smoke" boxes on fog, haze or grey asphalt, which is the largest false-positive source measured here. Real incident smoke seen from a fixed road camera starts as a local plume.
- **Persistence:** raise an alarm only when the same class is detected in **at least 3 of the last 4 samples, with samples about 1–2 s apart (4 s or more in total)**, and the boxes overlap (IoU > 0.1) or their centres stay within about 10% of the frame. Headlight glare and brake lights move with the vehicle or flicker. A fire stays in one place and grows.
- **Optional extras for night scenes:**
  - Require "fire" boxes not to be tracked as moving vehicles: suppress a fire box when its IoU with a COCO car, truck or bus box is above 0.3.
  - Mask the sky and horizon region per camera, which removes chimney steam and city-glow "smoke".
- At conf 0.6 with the filters above, the measured false-alarm count on 418 fire-free traffic frames (about 14 minutes of video) was **0 alarms** at 4 s persistence. Image-level recall on D-Fire at 0.6 is about 69–70%, and persistence over several frames recovers much of the per-frame miss rate for sustained fires.

**Caveats:**
- D-Fire is mostly wildfire, urban and indoor imagery. It contains no night highway negatives, which is why night glare is the weak spot.
- For deployment on a specific camera, the best next step is to collect 1–2 hours of that camera's normal footage (day, night and rain), add the frames that trigger detections as background images (empty label files), and fine-tune for 10–20 epochs.

## Files on spark-c111 (`~/wiut/firesmoke/`)

- `train.sh`: the exact training command.
- `dfire.yaml`: the dataset config.
- `convert.py` and `download.sh`: dataset preparation. `raw_sha256.txt` holds the parquet shard hashes.
- `runs/y11s_dfire/`: training curves, `results.csv` and weights.
- `runs/test_eval/`: test-set plots.
- `fp_final.json`, `fp_summary.txt` and `fp_final_hits/`: per-frame false-positive detections, including annotated frames.
- `neg_images_fp.json` and `pos_images_recall.json`.
