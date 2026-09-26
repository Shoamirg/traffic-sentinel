# Dev-label checklist (DRAFT)

`labels/dev_labels.json` was drafted from rendered frames of the sample videos, not yet checked by a person.
For each row: open the image (and the video at that time), then tick **OK** or write the fix
(new start/end, other class, or delete). Conventions: the task PDF's *Event classes* table; overlapping
same-class events are one segment. After confirming, edit `dev_labels.json` and re-run
`python tools/dev_eval.py --cache cache --gt labels/dev_labels.json`.

## C3896.MP4

| # | start-end (s) | class | conf. | model | evidence | image | OK / fix |
|---|---|---|---|---|---|---|---|
| 1 | 53.5-55.5 | failure_to_yield | high | missed | Black SUV crosses lower part of foreground yellow zebra (slip lane, right->left) ~1 m in front of a pedestrian walking on the zebra (53.6-55.5) | [img](evidence/C3896_01_failure_to_yield.jpg) | [ ] |
| 2 | 77.0-81.0 | failure_to_yield | low | missed | Black sedans then white SUV from the median-side lane of the top-left approach drive over the top crosswalk while 3 pedestrians are on its left part (several lanes away, walking toward the cars' path); may be a turn-arrow phase | [img](evidence/C3896_02_failure_to_yield.jpg) | [ ] |
| 3 | 104.5-132.0 | jaywalking | medium | predicted | Pedestrian steps off the bus stop onto the far carriageway (~105), walks diagonally across/along the lanes far from the crosswalk, reaches the kerb near the crossing waiting group ~132 (end is approximate) | [img](evidence/C3896_03_jaywalking.jpg) | [ ] |
| 4 | 109.5-113.0 | failure_to_yield | medium | predicted | Two white cars turning into the bottom-left leg drive over the upper end of the foreground zebra while a courier pushing a bike and a pedestrian are on the zebra (a few metres away) | [img](evidence/C3896_04_failure_to_yield.jpg) | [ ] |
| 5 | 179.5-181.0 | failure_to_yield | high | missed | White car crosses the lower part of the foreground zebra right in front of two pedestrians on it (180.0) | [img](evidence/C3896_05_failure_to_yield.jpg) | [ ] |
| 6 | 245.5-247.5 | failure_to_yield | medium | missed | Grey sedan drives across lower part of foreground zebra while two pedestrians are on the zebra | [img](evidence/C3896_06_failure_to_yield.jpg) | [ ] |
| 7 | 286.5-290.5 | failure_to_yield | medium | missed | White hatchback (286.5-288.5) then grey SUV (288.5-290.5) drive through the foreground zebra while a woman in red stands on it | [img](evidence/C3896_07_failure_to_yield.jpg) | [ ] |
| 8 | 297.5-328.0 | stop_line | medium | predicted | White car in median-side lane of top-left approach stops past the stop line, front at the crosswalk, ahead of the queue; signal head shows red, pedestrians cross in front (309); signal green ~328, moves 329 | [img](evidence/C3896_08_stop_line.jpg) | [ ] |
| 9 | 320.0-321.5 | failure_to_yield | medium | missed | Grey SUV drives across lower part of foreground zebra while the woman in red stands on it | [img](evidence/C3896_09_failure_to_yield.jpg) | [ ] |

Model events judged false positives on C3896.MP4 (check a few):

- near_miss 126.6-128.1: normal traffic on far carriageway; no braking/swerve visible
- near_miss 146.8-147.3: cars far away behind bus; ordinary flow
- near_miss 203.9-208.7: mixer truck and cars moving normally through junction; no evasive action
- stopped_vehicle 5.2-22.9: black SUV waiting mid-junction in the regular waiting position (signal-controlled), under 20 s, moves with the flow
- stopped_vehicle 38.7-108.1: merged multi-track; vehicles queued or waiting in junction for signal phase (80-107 group waits then leaves)
- stopped_vehicle 125.9-340.3: merged queued vehicles; incl. 2 cars waiting mid-junction 297-329 that leave with the signal cycle - queued, not breakdown
- illegal_u_turn 41.8-61.3: Yandex taxi does appear to U-turn around the median (near carriageway to far one), but no prohibiting sign/marking visible - legality unjudgeable
- illegal_u_turn 134.0-158.9: black SUV makes a normal through/turn movement, yields at the crosswalk 147-155; not a U-turn
- red_light 78.8-99.0: white SUV from median-side lane enters while other lanes wait; could be a turn-arrow phase, signal for that lane not visible (logged as low-confidence failure_to_yield 77-81 instead)
- failure_to_yield 46.1-49.4: delivery moped rides along the zebra with pedestrians; no vehicle drives through the crossing

## C3897.MP4

| # | start-end (s) | class | conf. | model | evidence | image | OK / fix |
|---|---|---|---|---|---|---|---|
| 10 | 0.0-26.5 | stopped_vehicle | low | missed (model's 4.5-26.3 box is a queued car) | Black SUV stationary mid-junction (below upper crossing, ~x840,y495 @1280) from video start while its approach is red and traffic later passes it; moves ~26.5s. Could be judged 'waiting for signal' -> reject if so | [img](evidence/C3897_10_stopped_vehicle.jpg) | [ ] |
| 11 | 4.0-21.5 | jaywalking | high | predicted (3.2-6.2 + 10.6-18.2) | Man leaves upper zebra at ~4s, walks on asphalt to pink island and on; then crowd (~12 people) leaves zebra ~11s and walks diagonally across road to lower crossing, reached ~21.5s | [img](evidence/C3897_11_jaywalking.jpg) | [ ] |
| 12 | 79.0-89.5 | jaywalking | high | predicted | Group of 5 leaves upper zebra ~79s and cuts diagonally across carriageway to the pink island, arriving ~89s | [img](evidence/C3897_12_jaywalking.jpg) | [ ] |
| 13 | 143.0-157.5 | jaywalking | medium | predicted (150.5-154.7) | Upper-right leg: woman steps off far kerb ~143s and crosses lanes to the zebra (~151s); elderly person then crosses same lanes 152-157s outside the crossing | [img](evidence/C3897_13_jaywalking.jpg) | [ ] |
| 14 | 177.5-179.5 | failure_to_yield | high | predicted | Black SUV turning right drives over the top of the lower diagonal zebra passing right beside a pedestrian who is on the crossing | [img](evidence/C3897_14_failure_to_yield.jpg) | [ ] |
| 15 | 180.8-182.2 | failure_to_yield | medium | missed | White sedan crosses top end of lower zebra while the same pedestrian is mid-crossing (a few metres away) | [img](evidence/C3897_15_failure_to_yield.jpg) | [ ] |
| 16 | 182.5-184.2 | failure_to_yield | medium | predicted (182.3-183.1) | Grey sedan crosses top end of lower zebra while pedestrian still on it | [img](evidence/C3897_16_failure_to_yield.jpg) | [ ] |
| 17 | 191.5-194.8 | failure_to_yield | medium | missed | Motorcyclist rides along/through the lower zebra among 3 pedestrians on it, exits frame bottom ~194.8 | [img](evidence/C3897_17_failure_to_yield.jpg) | [ ] |
| 18 | 209.5-248.5 | stopped_vehicle | low | missed | Dark sedan stopped just past the upper zebra inside the junction (~x430,y345 @960 frame) for ~39s during red, queue behind at stop line; leaves when queue releases. Alternative reading: stop_line/queued at signal | [img](evidence/C3897_18_stopped_vehicle.jpg) | [ ] |
| 19 | 214.0-217.2 | failure_to_yield | medium | predicted (214.2-215.0) | Black SUV then white taxi drive over the lower end of the lower zebra while two women are on the crossing 1-3 m away (merged, overlapping) | [img](evidence/C3897_19_failure_to_yield.jpg) | [ ] |
| 20 | 295.5-304.5 | jaywalking | high | predicted (294.4-298.6) | Couple (orange/brown) leaves upper zebra ~295.5s, walks across road via island, reaches lower zebra ~304.5s | [img](evidence/C3897_20_jaywalking.jpg) | [ ] |
| 21 | 307.0-317.8 | jaywalking | high | missed | Another group leaves upper zebra from ~307s and walks diagonally on the road toward the island; two still on the road at video end | [img](evidence/C3897_21_jaywalking.jpg) | [ ] |

Model events judged false positives on C3897.MP4 (check a few):

- near_miss 15.8-17.5: normal far-lane traffic; overlapping boxes/ID swaps, no braking or swerve
- near_miss 111.2-115.2: distant far-road cars in normal flow
- near_miss 123.5-124.0: tracker ID swap between two cars in same lane, no evasive action
- near_miss 233.7-235.0: delivery scooter crosses road ahead of slow white car; no visible braking/swerve (checked nm_a.jpg)
- near_miss 243.9-245.0: far-lane traffic, normal
- stopped_vehicle 4.5-26.3: tracked car is in the signal queue at the stop line (queued at red)
- stopped_vehicle 35.4-317.8: merged fragments of moving/queued vehicles across whole video
- illegal_u_turn 105.6-127.1: real U-turn (from top-left approach back onto far road, waits near right island) but no visible prohibition; U-turns routine here - human to decide legality
- illegal_u_turn 129.0-141.4: same U-turn pattern (white car), legality not determinable
- illegal_u_turn 196.9-217.5: same U-turn pattern (dark car), legality not determinable
- stop_line 0.1-19.9: front-row car in queue sits at/just behind the stop line before the zebra; not clearly past it
- stop_line 61.7-94.7: moving red car/fragmented track, no stop past line
- jaywalking 177.2-180.4: tracked 'person' is a delivery motorcyclist riding in the lane
- jaywalking 273.8-276.5: tracked object is a scooter rider near far kerb/sidewalk, not a pedestrian on road
- road_obstacle 177.2-187.3: no object on carriageway; only moving traffic and a courier motorcycle

## C3902.MP4

| # | start-end (s) | class | conf. | model | evidence | image | OK / fix |
|---|---|---|---|---|---|---|---|
| 22 | 52.5-55.5 | failure_to_yield | medium | predicted | Black sedan turning right from left queue waits 49-52s then drives over the lower-left zebra while a pedestrian stream is still on it (pedestrian ~1m from its front at 53s) | [img](evidence/C3902_22_failure_to_yield.jpg) | [ ] |
| 23 | 82.0-95.0 | jaywalking | high | predicted | Group of 3 pedestrians leave the lower zebra/island and walk diagonally across the open junction to the right end of the horizontal zebra | [img](evidence/C3902_23_jaywalking.jpg) | [ ] |
| 24 | 97.5-117.5 | stop_line | medium | predicted | Orange delivery moped rides in from the left and stops between the stop line and the zebra ahead of the red-light queue; left signal head turns green ~117.5s, moped leaves at 118s | [img](evidence/C3902_24_stop_line.jpg) | [ ] |
| 25 | 181.0-192.0 | jaywalking | medium | predicted | Several pedestrians step off the horizontal zebra and cut diagonally across the carriageway down to the pink islands/lower zebra | [img](evidence/C3902_25_jaywalking.jpg) | [ ] |
| 26 | 225.5-232.5 | jaywalking | medium | missed | Man in black walks from the right pink island across open carriageway up to the horizontal zebra (buses turning nearby) | [img](evidence/C3902_26_jaywalking.jpg) | [ ] |
| 27 | 259.5-269.0 | jaywalking | high | predicted | Three pedestrians leave the horizontal zebra and walk diagonally across the carriageway to the pink islands/lower zebra | [img](evidence/C3902_27_jaywalking.jpg) | [ ] |
| 28 | 274.5-279.5 | jaywalking | low | missed | Man in white shirt walks from horizontal zebra diagonally down across the road to the top pink island; short | [img](evidence/C3902_28_jaywalking.jpg) | [ ] |
| 29 | 284.5-290.5 | failure_to_yield | high | predicted | Two dark SUVs turning right cross the lower-left zebra back-to-back with pedestrians directly beside/in front (man in suit + girl at 285.5-286.5s; group at 288-289.5s); merged since contiguous | [img](evidence/C3902_29_failure_to_yield.jpg) | [ ] |

Model events judged false positives on C3902.MP4 (check a few):

- failure_to_yield 25.9-40.6: Orange moped rides onto the lower zebra and moves along it together with pedestrians (acting as a pedestrian), no vehicle forcing through a crossing
- failure_to_yield 292.6-306.1: Gray SUV waited at the zebra 291-304s and only crossed after pedestrians had cleared - it yielded
- near_miss 17.9-18.9: Cyclist and pedestrian on the far-right sidewalk, no evasive action
- near_miss 80.8-81.8: Distant cars in normal flow on the far road
- near_miss 151.0-152.1: Distant cars in normal flow; box/track overlap only
- near_miss 279.0-279.9: Distant cars in dense but normal traffic
- near_miss 286.6-287.6: Distant cars in dense but normal traffic
- stopped_vehicle 2.9-40.9: White van queued in the left approach at a red signal
- stopped_vehicle 52.5-123.0: Articulated bus crawling in slow exit traffic at bottom-right (moves steadily 60-84s) plus queued cars; not stationary >=10s
- stopped_vehicle 135.8-162.1: Cars queued on the right approach waiting for their signal
- stopped_vehicle 208.1-242.8: Buses briefly held (~8s) inside the junction behind other turning buses; queue, not a stopped vehicle
- stopped_vehicle 258.7-278.4: Vehicles queued at signals (left and right approaches)
- stopped_vehicle 296.2-311.1: Cars in dense slow-moving traffic
- illegal_u_turn 288.6-307.8: Bus making a normal right turn; track ID switch, no U-turn
- stop_line 7.3-37.1: Front-row car of the left queue is behind the stop line
- stop_line 253.7-277.1: Front-row car of the left queue is behind the stop line
- jaywalking 95.0-146.7: Tail of model segment 85.4-146.7: after 95s pedestrians are on zebras or sidewalks (and 136-142s only short hops between island and zebra)
- jaywalking 150.6-181.0: Head of model segment 150.6-186.2: pedestrians on zebras/far sidewalk before 181s
- congestion 60.0-84.0: Recall-pass idea: bunching at the bottom-right exit around the bus, but traffic keeps moving; not a standstill across all lanes
- congestion 84.0-119.0: Recall-pass idea: long left-approach queue is a normal red-light queue that clears at green (~118s)

## C3905.MP4

| # | start-end (s) | class | conf. | model | evidence | image | OK / fix |
|---|---|---|---|---|---|---|---|
| 30 | 14.0-15.0 | failure_to_yield | low | predicted | white hatchback drives across the lower (bottom-left diagonal) crosswalk while a cyclist and 3 pedestrians are on its upper end ~near curb | [img](evidence/C3905_30_failure_to_yield.jpg) | [ ] |
| 31 | 18.3-19.3 | failure_to_yield | medium | predicted | dark sedan crosses lower crosswalk while group of 3 pedestrians + man with bike stand on the zebra between the islands | [img](evidence/C3905_31_failure_to_yield.jpg) | [ ] |
| 32 | 28.8-29.8 | failure_to_yield | medium | predicted | white sedan passes lower crosswalk right next to the group of 4 pedestrians standing on the zebra | [img](evidence/C3905_32_failure_to_yield.jpg) | [ ] |
| 33 | 39.0-41.5 | failure_to_yield | high | predicted | white Cobalt drives through lower crosswalk between a man walking a bicycle and 3 pedestrians, all on the zebra | [img](evidence/C3905_33_failure_to_yield.jpg) | [ ] |
| 34 | 42.5-47.5 | failure_to_yield | medium | predicted | white sedan then black SUV cross upper part of lower crosswalk while a pedestrian and cyclist are mid-crossing on it | [img](evidence/C3905_34_failure_to_yield.jpg) | [ ] |
| 35 | 77.5-114.5 | stop_line | medium | predicted | signal (head at ~770,255) amber 72.5, red 75.5, green 114.5; white van + black SUV creep forward and stop on the upper crosswalk at ~77.5 while red, pedestrians walk around them until green | [img](evidence/C3905_35_stop_line.jpg) | [ ] |
| 36 | 104.5-106.5 | failure_to_yield | medium | missed | black sedan crosses lower crosswalk while pedestrians step onto its upper end | [img](evidence/C3905_36_failure_to_yield.jpg) | [ ] |
| 37 | 108.5-112.0 | failure_to_yield | high | missed | black sedan then white Nexia cross lower crosswalk while several pedestrians are walking on it (white car passes ~1 m from a pedestrian at 111 s) | [img](evidence/C3905_37_failure_to_yield.jpg) | [ ] |

Model events judged false positives on C3905.MP4 (check a few):

- near_miss 18.1-19.0: two cars on the far carriageway moving normally; no braking/swerve visible
- stopped_vehicle 5.1-28.8: far top-left cars are a signal queue / parked cars off the carriageway, tracks jump between vehicles
- stopped_vehicle 56.3-116.8: semi-truck and cars stationary ~85-107 s are queued behind traffic in a gridlocked junction, not an isolated stopped vehicle; track IDs switch
- illegal_u_turn 62.1-87.8: track ID switches between different cars (and finally a pedestrian); no vehicle performs a U-turn
- jaywalking 32.6-35.2: yellow-vest courier is riding a bicycle between queued cars through the junction, not a pedestrian
- failure_to_yield 9.6-31.8: merged span too long; replaced by the individual crosswalk passes at 14,18.3,28.8 s
- failure_to_yield 36.5-45.8: span refined into 39.0-41.5 and 42.5-47.5
- failure_to_yield 60.0-62.0: white SUV crosses lower crosswalk but pedestrians are still on the sidewalk
- failure_to_yield 100.0-103.5: black SUV crosses lower crosswalk with no pedestrian on it (only on the sidewalk)
