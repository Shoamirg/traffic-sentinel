# Traffic Sentinel — AMITY-Tigers

WIUT Hackathon 2026, Computer Vision track. A fixed CCTV road camera is turned into
(A) a list of traffic events `[start_sec, end_sec, label]` over 14 classes and
(B) a causal, per-frame probability that an accident starts within 5 s.

- Website + live demo: _link added at submission_
- Predictions on the sample videos: [`predictions_samples.json`](predictions_samples.json)

## Run it

```bash
pip install -r requirements.txt           # Python >= 3.10; CUDA wheels of torch on Linux x86 by default
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

No manual steps: all weights are committed in `weights/` (yolo11m.pt 40 MB, yolo11s.pt 19 MB,
fire_smoke.pt). `weights/download.sh` only verifies checksums and re-fetches the public
YOLO11 files if they are missing. Nothing is downloaded at run time.

`run_submission.py` and `evaluate.py` are the organisers' files, unchanged.

## Approach

```
video ──► I/P-frame decoder (10 fps, 1280 px) ──► YOLO11m (COCO road users) ──► ByteTrack-style tracker
                   │                                                            │
                   ├─► lamp probes (signal state) ─┐                           ▼
                   ├─► fire/smoke YOLO (2 fps) ────┤              trajectories (ground points,
                   └─► 1-fps thumbnails ───────────┤              local-linear smoothed velocity)
                                                   ▼                            │
                    scene prior (learned from samples) + layout ──► 14 per-class rules ──► segment cleanup ──► events
```

| Stage | Learned or rule-based | Details |
|---|---|---|
| Road-user detection | learned (COCO-pretrained YOLO11m, not fine-tuned) | `src/sentinel/detector.py`; fp16, batched, 1280 px |
| Tracking | algorithmic | `src/sentinel/tracker.py`: ByteTrack two-stage IoU association, constant-velocity prediction, deterministic |
| Kinematics | algorithmic | `src/sentinel/tracks.py`: bottom-centre ground point, local linear regression in a 0.6 s window |
| Scene model | learned statistically | `src/sentinel/scene.py`: per-cell heading histograms (lane directions), carriageway mask, queue zones, built from sample-video trajectories (`scene/prior.npz`) and blended with the test video's own tracks |
| Scene layout | hand-drawn | `scene/layout.json`: stop lines, crossings, solid lines, signal lamp ROIs, junction box |
| Signal state | adaptive rule | `src/sentinel/signal.py`: lamp brightness, per-video Otsu threshold |
| Fire / smoke | learned (fine-tuned YOLO11) | `weights/FIRE_SMOKE.md` |
| Event classes | rule-based | `src/sentinel/rules/*.py`, one module per family, documented at the top of each file |
| Part B risk | rule-based, calibrated | `src/sentinel/risk.py`: for pairs involving a moving vehicle whose paths meet within 3 s (closest point of approach), the deceleration needed to avoid contact (DRAC, in box sizes/s²), plus observed hard braking; must persist 0.5 s; logistic calibrated on the samples as normal traffic (median ≈ 0.03, 99.9th pct = 0.5, `tools/calibrate_risk.py`); fast-attack/slow-decay smoothing |

Part B runs its **own** detector + tracker inside `RiskEstimator.step` on every 5th frame it is
handed; it never opens the video and never reads Part A output.

## Repository layout

```
solution.py             interface for the harness (thin adapter)
src/sentinel/           pipeline: video, detector, tracker, tracks, scene, signal, geometry, rules/, risk, render, demo
scene/                  reference.jpg, prior.npz, background.jpg, signal_main.png, layout.json (scene facts)
weights/                model weights + download.sh + FIRE_SMOKE.md
tools/                  extract_all, dev_eval, build_prior, snapshots, publish_site_data, train_fire_smoke.sh
labels/dev_labels.json  our own annotations of the sample videos (dev set)
tests/                  unit + behavioural tests (synthetic trajectories)
website/                FastAPI demo server + static site (Dockerfile for a Hugging Face Space)
```

## Reproducing

```bash
python tools/make_reference.py samples/C3897.MP4 --at 75            # scene/reference.jpg (registration anchor)
python tools/extract_all.py --videos samples --cache cache          # GPU pass, cached
python tools/build_prior.py --cache cache                           # scene/prior.npz + background.jpg
python tools/calibrate_risk.py --cache cache                        # Part B calibration check (CAL_MID)
python tools/dev_eval.py --cache cache --gt labels/dev_labels.json  # rules + official metric
python run_submission.py --videos samples --out predictions_samples.json --team AMITY-Tigers
bash tools/train_fire_smoke.sh                                      # fire/smoke model from scratch
pytest -q tests website/server/tests
```

Every file in `scene/` except the hand-drawn `layout.json` is **derived from the organisers'
sample videos** by the tools above: `reference.jpg` is one working-resolution frame,
`background.jpg` the median of sampled frames, `prior.npz` statistics of tracked vehicles.
They ship in the repository because the pipeline needs them at run time (no network).
Rebuilding `prior.npz` on another OS/CPU can differ by a few cell counts (floating-point
differences in image registration); this does not change the detected events on the samples.

## Runtime

The sample videos are H.264 High 4:2:2, 10-bit, 140 Mbit/s 4K. That profile has no hardware
decoder before NVIDIA Blackwell (a T4's NVDEC handles 8-bit 4:2:0 only), so decoding is CPU work,
and the organisers' harness itself decodes every frame once for Part B. Part A therefore decodes
with PyAV using `skip_frame=NONREF`: the camera's GOP is I-B-B-P, the B-frames are never decoded,
and the remaining I/P frames are exactly the 10 fps Part A analyses; the scaler writes straight
to 1280 px and only the signal-lamp crops are converted at full resolution
(`src/sentinel/video.py`, `SENTINEL_DECODER=cv2` restores the plain OpenCV path).

| C3905 (127.6 s), Part A only | 20 cores | 2 cores |
|---|---|---|
| OpenCV, every frame | 80.6 s | 318.7 s |
| PyAV, reference frames only | 76.6 s | 270.6 s |

Decoding runs in a background thread a few frames ahead of detection (`video.prefetch`), so the
CPU decode and the GPU detector overlap instead of alternating; output is unchanged.

Scaling on C3905 (Spark, cores pinned with `taskset`; budget 383 s). "Harness floor" is the
organisers' harness with `tools/null_solution.py`, i.e. its own decode of every frame for Part B:

| cores | harness floor | our Part A | total ≈ floor + Part A + Part B detector |
|---|---|---|---|
| 2 | 248 s | 272 s | over budget for any solution on this CPU class |
| 4 | 134 s | 166 s | ≈ 315 s |
| 8 | 57 s | 79 s | ≈ 150 s |
| 20 | — | 61 s | ≈ 110 s |

On a Colab T4 (2 x86 vCPUs) the harness floor alone is 387 s, over the 383 s budget.

`tools/null_solution.py` times the harness alone (`--solution tools/null_solution.py`): that is the
decode floor on a given machine, and our share of the 3 x duration budget is what remains above it.
Ultralytics runs with `YOLO_OFFLINE=1`, so it never opens a network connection.

## Determinism

Seed 0 everywhere (`config.SEED`); cuDNN benchmark off and deterministic kernels on; the tracker
and all rules are deterministic. Two runs on the same machine give identical `predictions.json`
up to fp16 noise in detector scores.

## Data and models used

| Asset | Use | Licence |
|---|---|---|
| YOLO11m / YOLO11s COCO weights (Ultralytics) | road-user detection | AGPL-3.0 |
| COCO 2017 (via the pretrained weights) | detector pre-training | CC BY 4.0 |
| Fire/smoke dataset — see `weights/FIRE_SMOKE.md` | fire/smoke fine-tuning | see file |
| Organisers' sample videos | dev labels, scene prior | hackathon use only |

Open-source code reused: Ultralytics (AGPL-3.0). The tracker follows the ByteTrack paper
(Zhang et al., ECCV 2022) and is implemented from scratch.

## Team — AMITY-Tigers

- Shoamir Shorustamov — captain
- Afzal Qodirov
- Zohirjon Shokiriy

## Results

_Filled in after the dev-set evaluation._
