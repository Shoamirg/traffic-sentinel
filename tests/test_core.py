"""Unit tests for segments, tracker, signal classification, geometry and risk maths."""
from __future__ import annotations

import numpy as np

from sentinel.detector import Detections
from sentinel.geometry import crossings, enter_leave, polyline_side, signed_side
from sentinel.risk import pair_hazard
from sentinel.segments import Event, finalize, merge_segments, runs
from sentinel.signal import classify
from sentinel.tracker import ByteTracker, iou_matrix
from sentinel.tracks import local_linear_fit


# ---- segments -----------------------------------------------------------------
def test_runs_bridges_small_gaps_only():
    t = np.arange(0, 10, 0.5)
    flags = (t < 2) | ((t >= 2.5) & (t < 4)) | (t >= 8)
    assert runs(t, flags, max_gap=1.0) == [(0.0, 3.5), (8.0, 9.5)]


def test_finalize_merges_same_class_and_drops_blips():
    evs = [Event(1, 5, "jaywalking"), Event(4, 8, "jaywalking"), Event(20, 20.2, "jaywalking"),
           Event(3, 6, "accident")]
    out = finalize(evs, duration=10.0)
    assert [(e.start, e.end, e.label) for e in out] == [(1, 8, "jaywalking"), (3, 6, "accident")]


def test_finalize_clips_to_duration():
    out = finalize([Event(8, 15, "congestion")], duration=10.0)
    assert out[0].end == 10.0


def test_merge_segments_sorted_and_overlapping():
    assert merge_segments([(5, 6), (1, 2), (1.5, 3)], max_gap=0.0) == [(1, 3), (5, 6)]


# ---- tracker -------------------------------------------------------------------
def _det(boxes, conf=0.9, cls=2):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return Detections(boxes, np.full(len(boxes), conf, np.float32), np.full(len(boxes), cls, int))


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b), [[1 / 3, 0.0]], atol=1e-4)


def test_tracker_keeps_identity_for_moving_objects():
    tr = ByteTracker()
    ids = []
    for k in range(20):
        out = tr.update(_det([[10 + 5 * k, 10, 60 + 5 * k, 40], [300 - 5 * k, 100, 350 - 5 * k, 130]]), k * 0.1)
        ids.append(sorted(o[0] for o in out))
    assert all(i == ids[0] for i in ids) and len(ids[0]) == 2


def test_tracker_survives_short_occlusion():
    tr = ByteTracker()
    first = tr.update(_det([[0, 0, 50, 30]]), 0.0)[0][0]
    for k in range(1, 5):
        tr.update(_det([[5 * k, 0, 50 + 5 * k, 30]]), k * 0.1)
    tr.update(Detections.empty(), 0.5)
    tr.update(Detections.empty(), 0.6)
    again = tr.update(_det([[35, 0, 85, 30]]), 0.7)
    assert again[0][0] == first


def test_tracker_is_deterministic():
    def run():
        tr = ByteTracker()
        rng = np.random.default_rng(0)
        out = []
        for k in range(30):
            boxes = rng.uniform(0, 500, (6, 2))
            dets = _det(np.hstack([boxes, boxes + 40]), conf=0.8)
            out.append([(o[0], tuple(o[1])) for o in tr.update(dets, k * 0.1)])
        return out
    assert run() == run()


# ---- smoothing -----------------------------------------------------------------
def test_local_linear_fit_exact_on_constant_velocity():
    ts = np.arange(0, 5, 0.1)
    xy = np.stack([3 + 20 * ts, 7 - 4 * ts], axis=1)
    pos, vel = local_linear_fit(ts, xy, 0.6)
    np.testing.assert_allclose(pos, xy, atol=1e-6)
    np.testing.assert_allclose(vel, np.tile([20, -4], (len(ts), 1)), atol=1e-6)


# ---- signal ----------------------------------------------------------------------
def test_signal_classify_finds_red_phases():
    t = np.arange(0, 60, 0.1)
    red = ((t // 15) % 2 == 1)                      # 15 s green / 15 s red cycles
    scores = np.where(red, 120.0, 20.0) + np.random.default_rng(0).normal(0, 3, len(t))
    sig = classify(t, scores)
    assert sig.known
    assert sig.red_at(20.0) and not sig.red_at(5.0)
    assert abs(sig.red_since(20.0) - 5.0) < 0.3
    assert abs(sig.next_green(20.0) - 30.0) < 0.3


def test_signal_short_glare_flips_are_absorbed():
    t = np.arange(0, 60, 0.1)
    scores = np.where((t // 20) % 2 == 1, 120.0, 20.0)       # red 20-40 s
    scores[(t > 5) & (t < 6.5)] = 120.0                      # 1.5 s glare during green
    scores[(t > 30) & (t < 31)] = 20.0                       # 1 s occlusion during red
    sig = classify(t, scores)
    assert not sig.red_at(5.5) and sig.red_at(30.5)


def test_signal_unknown_when_lamp_never_changes():
    t = np.arange(0, 60, 0.1)
    assert not classify(t, np.full(len(t), 50.0)).known


# ---- geometry ----------------------------------------------------------------------
def test_line_crossing_time_and_direction():
    line = np.array([[0, 100], [200, 100]])
    ts = np.arange(0, 3, 0.5)
    xy = np.stack([np.full(len(ts), 50.0), 40 + 40 * ts], axis=1)   # y: 40 -> 140
    side = signed_side(line, xy)
    cr = crossings(ts, side, np.ones(len(ts), bool))
    assert len(cr) == 1 and abs(cr[0][0] - 1.5) < 1e-6


def test_polyline_side_and_enter_leave():
    poly = np.array([[0, 0], [100, 0], [100, 100]], float)
    side, on = polyline_side(poly, np.array([[50, 10], [50, -10], [300, 300]], float))
    assert np.sign(side[0]) != np.sign(side[1]) and on[0] and on[1]
    assert enter_leave(np.arange(6.0), np.array([0, 1, 1, 0, 1, 1], bool)) == [(1.0, 2.0), (4.0, 5.0)]


# ---- risk --------------------------------------------------------------------------
def test_pair_hazard_head_on_is_high_and_parallel_is_zero():
    head_on = pair_hazard(np.array([0., 0]), np.array([100., 0]), 80, np.array([300., 0]), np.array([-100., 0]), 80)
    parallel = pair_hazard(np.array([0., 0]), np.array([100., 0]), 80, np.array([0., 200]), np.array([100., 0]), 80)
    diverging = pair_hazard(np.array([0., 0]), np.array([-100., 0]), 80, np.array([300., 0]), np.array([100., 0]), 80)
    assert head_on > 0.5 and parallel == 0.0 and diverging == 0.0


def test_pair_hazard_ignores_paths_that_do_not_meet_and_gentle_approach():
    # adjacent lanes, opposite directions: they pass 1.5 sizes apart and never meet
    passing = pair_hazard(np.array([0., 0]), np.array([100., 0]), 80, np.array([300., 120]), np.array([-100., 0]), 80)
    # rolling up to a standing queue at walking pace: needs only gentle braking
    gentle = pair_hazard(np.array([0., 0]), np.array([60., 0]), 80, np.array([400., 0]), np.array([0., 0]), 80)
    fast = pair_hazard(np.array([0., 0]), np.array([400., 0]), 80, np.array([200., 0]), np.array([0., 0]), 80)
    assert passing == 0.0
    assert gentle < 0.5 < fast


def test_risk_needs_persistence_before_it_rises():
    from sentinel.risk import CausalRisk
    est = CausalRisk()
    est.reset({"fps": 25.0, "width": 1280})
    assert est._persistent(10.0, 0.0) == 0.0                     # a single spike does not count
    for k, h in enumerate([10.0, 10.0, 10.0, 0.0]):
        value = est._persistent(h, 0.2 * (k + 1))
    assert value == 0.0                                          # the dip resets the level
    for k in range(5):
        value = est._persistent(10.0, 1.0 + 0.2 * k)
    assert value == 10.0


def test_shipped_scene_assets_exist():
    # the pipeline registers each video against these and loads the learned prior; a clone without them
    # silently falls back to a much weaker scene model
    from sentinel import config
    for name in ("layout.json", "prior.npz", "reference.jpg", "background.jpg"):
        assert (config.SCENE_DIR / name).is_file(), name
