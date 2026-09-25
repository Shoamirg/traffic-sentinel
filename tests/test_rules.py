"""Behavioural tests for the trajectory rules on synthetic scenes."""
from __future__ import annotations

import numpy as np

from sentinel.rules import accident, jaywalking, near_miss, stopped_vehicle, u_turn, wrong_way
from sentinel.scene import Layout
from synth import W, context, eastbound_traffic, make_track


def test_wrong_way_flags_vehicle_against_learned_flow():
    traffic = eastbound_traffic(n=60, t_gap=1.0)       # a well-observed lane
    rogue = make_track(1, "car", 50.0, [(50.0, W - 10, 400), (56.0, 10, 400)])
    events = wrong_way.detect(context(traffic + [rogue], prior=traffic))
    assert len(events) == 1
    assert events[0].start < 52.0 and events[0].end > 54.0


def test_wrong_way_silent_for_normal_traffic():
    traffic = eastbound_traffic()
    assert wrong_way.detect(context(traffic, prior=traffic)) == []


CARRIAGEWAYS = [
    {"name": "eastbound", "poly": np.array([[0, 380], [650, 380], [650, 420], [0, 420]], np.float32)},
    {"name": "westbound", "poly": np.array([[0, 480], [650, 480], [650, 520], [0, 520]], np.float32)},
]
U_TURN = [(0, 100, 400), (5, 600, 400), (7, 700, 450), (9, 600, 500), (14, 100, 500)]


def test_u_turn_detected_with_boundaries():
    ctx = context([make_track(1, "car", 0, U_TURN)], layout=Layout(carriageways=CARRIAGEWAYS))
    events = u_turn.detect(ctx)
    assert len(events) == 1
    assert 4.0 < events[0].start < 6.5 and 8.5 < events[0].end < 11.0


def test_straight_track_is_not_u_turn():
    ctx = context([make_track(1, "car", 0, [(0, 0, 400), (6, W, 400)])], layout=Layout(carriageways=CARRIAGEWAYS))
    assert u_turn.detect(ctx) == []


def test_left_turn_off_the_road_is_not_u_turn():
    # leaves the eastbound carriageway and heads down/left towards the camera, never joins westbound
    track = make_track(1, "car", 0, [(0, 100, 400), (5, 600, 400), (7, 700, 550), (10, 500, 700)])
    assert u_turn.detect(context([track], layout=Layout(carriageways=CARRIAGEWAYS))) == []


def test_u_turn_allowed_zone_suppresses():
    zone = np.array([[500, 300], [900, 300], [900, 600], [500, 600]], np.float32)
    ctx = context([make_track(1, "car", 0, U_TURN)], layout=Layout(carriageways=CARRIAGEWAYS, u_turn_allowed=[zone]))
    assert u_turn.detect(ctx) == []


def test_stopped_vehicle_after_moving():
    traffic = eastbound_traffic(y=380)          # the adjacent lane keeps flowing past the stopped car
    car = make_track(1, "car", 10, [(10, 0, 400), (13, 600, 400), (40, 600, 400), (43, W, 400)])
    events = stopped_vehicle.detect(context(traffic + [car], prior=traffic + [car]))
    assert len(events) == 1
    assert abs(events[0].start - 13) < 1.5 and abs(events[0].end - 40) < 1.5


def test_short_stop_is_ignored():
    car = make_track(1, "car", 10, [(10, 0, 400), (13, 600, 400), (18, 600, 400), (21, W, 400)])
    assert stopped_vehicle.detect(context([car] + eastbound_traffic())) == []


def test_jaywalker_on_road_detected_but_sidewalk_ignored():
    # a three-lane carriageway (y 360-440): a single synthetic lane is narrower than MIN_TRAVEL_REL
    traffic = eastbound_traffic(y=360) + eastbound_traffic(y=400, start_id=2000) + eastbound_traffic(y=440, start_id=3000)
    walker = make_track(1, "person", 20, [(20, 640, 300), (24, 640, 500)], box_w=30, box_h=80)   # ~0.6 heights/s
    sidewalk = make_track(2, "person", 20, [(20, 100, 700), (30, 600, 700)], box_w=30, box_h=80)
    events = jaywalking.detect(context(traffic + [walker, sidewalk], prior=traffic))
    assert len(events) == 1
    assert 20 <= events[0].start <= 23


def test_rear_end_collision_is_accident_not_near_miss():
    lead = make_track(1, "car", 0, [(0, 300, 400), (4, 500, 400), (5, 530, 400), (15, 532, 400)])
    chaser = make_track(2, "car", 0, [(0, -300, 400), (4, 460, 400), (5, 470, 400), (15, 471, 400)])
    ctx = context([lead, chaser])
    acc = accident.detect(ctx)
    assert len(acc) == 1 and 3.0 < acc[0].start < 5.0
    assert near_miss.detect(ctx) == []


def test_hard_brake_before_other_car_is_near_miss():
    stopped = make_track(1, "car", 0, [(0, 800, 400), (10, 800, 400)])
    # emergency stop: ~300 px/s (3 car lengths/s) to standstill in 0.6 s, right behind a stopped car
    braker = make_track(2, "car", 0, [(0, -150, 400), (2.5, 600, 400), (3.1, 690, 400), (10, 690, 400)])
    events = near_miss.detect(context([stopped, braker]))
    assert len(events) == 1
    assert 1.5 < events[0].start < 3.5


def test_jaywalker_split_by_id_switch_is_one_crossing():
    # the same crossing as above, but the tracker swaps IDs halfway: neither piece passes the gates alone
    traffic = eastbound_traffic(y=360) + eastbound_traffic(y=400, start_id=2000) + eastbound_traffic(y=440, start_id=3000)
    first = make_track(1, "person", 20, [(20, 640, 300), (22.1, 640, 405)], box_w=30, box_h=80)
    second = make_track(2, "person", 22.3, [(22.3, 642, 412), (24, 642, 500)], box_w=30, box_h=80)
    events = jaywalking.detect(context(traffic + [first, second], prior=traffic))
    assert len(events) == 1
    assert set(events[0].tids) == {1, 2}
