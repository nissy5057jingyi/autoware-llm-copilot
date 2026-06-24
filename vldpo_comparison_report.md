# VL-DPO vs Numeric Preference Selection: Comparison Report

## Experimental Setup

- **Numeric method**: top-1 reward = chosen, bottom-1 = rejected (1 pair/sample)
- **VL-DPO method**: Frozen VLM (Gemini-2.5-Flash) selects best trajectory from BEV images (N-1 pairs/sample)
- **Scenarios**: 4 driving scenes, 4 candidates each

## Main Comparison Table

| Scene | Goal | #Cands | Numeric Chosen | Numeric Rejected | VL-DPO Chosen | #Pairs (VL-DPO) | Agree |
|-------|------|--------|---------------|-----------------|---------------|-----------------|-------|
| Clear road, straight | FORWARD | 3 | rank=1 (r=0.92) | rank=3 (r=0.55) | rank=1 (r=0.92) | 2 | ✓ |
| Obstacle ahead, decelerate | FORWARD | 4 | rank=1 (r=0.85) | rank=4 (r=0.25) | rank=1 (r=0.85) | 3 | ✓ |
| Slow vehicle ahead, left lane change | FORWARD | 4 | rank=1 (r=0.90) | rank=4 (r=0.35) | rank=1 (r=0.90) | 3 | ✓ |
| Intersection, turn right | TURN_RIGHT | 4 | rank=1 (r=0.91) | rank=4 (r=0.20) | rank=1 (r=0.91) | 3 | ✓ |

## Training Signal Comparison

| Scene | Numeric #Pairs | VL-DPO #Pairs | Signal Multiplier |
|-------|---------------|---------------|-------------------|
| Clear road, straight | 1 | 2 | 2.0x |
| Obstacle ahead, decelerate | 1 | 3 | 3.0x |
| Slow vehicle ahead, left lane change | 1 | 3 | 3.0x |
| Intersection, turn right | 1 | 3 | 3.0x |
| **Total** | **4** | **11** | **2.8x** |

## VLM Semantic Reasoning

### S1_clear_road: Clear road, straight
- **Goal**: `FORWARD`
- **Numeric chosen**: straight+accelerate(y=34.5m)
- **VL-DPO chosen**: straight+accelerate(y=34.5m)
- **VLM reasoning**: The mission is FORWARD with a clear road ahead and a heading speed of 8.5. Candidate 1 shows a straight trajectory with increasing distance between waypoints, indicating acceleration, which is appropriate for the given speed and clear road. Candidate 2 maintains speed, and Candidate 3 turns left, neither of which aligns with the optimal action for the given scenario.
- **Consistency**: ✓ Consistent with numeric

### S2_obstacle_ahead: Obstacle ahead, decelerate
- **Goal**: `FORWARD`
- **Numeric chosen**: straight+decelerate/stop(y=11.5m)
- **VL-DPO chosen**: straight+decelerate/stop(y=11.5m)
- **VLM reasoning**: The mission is to go FORWARD and the scene requires deceleration due to a static obstacle 15m ahead. Candidate 1 shows a clear deceleration, with the waypoints becoming closer together over time, indicating a reduction in speed. The final waypoint at t6 is around 12m, which is a safe distance from the 15m obstacle given the initial speed of 10m/s. Candidate 2 shows very aggressive deceleration, which might be too sudden. Candidates 3 and 4 show acceleration or maintaining speed, which would lead to a collision with the obstacle.
- **Consistency**: ✓ Consistent with numeric

### S3_lane_change: Slow vehicle ahead, left lane change
- **Goal**: `FORWARD`
- **Numeric chosen**: left+accelerate(y=31.5m)
- **VL-DPO chosen**: left+accelerate(y=31.5m)
- **VLM reasoning**: The mission is FORWARD, and there is a slow vehicle 8m ahead. The left lane is clear. Candidate 1 shows a smooth lane change to the left, which is the safest and most efficient way to proceed given the clear left lane and slow vehicle ahead. Candidate 2 and 4 involve staying behind the slow vehicle or braking, which is not optimal when a clear lane is available. Candidate 3 attempts a lane change to the right, which is not indicated as clear.
- **Consistency**: ✓ Consistent with numeric

### S4_turn_right: Intersection, turn right
- **Goal**: `TURN_RIGHT`
- **Numeric chosen**: right+decelerate/stop(y=9.0m)
- **VL-DPO chosen**: right+decelerate/stop(y=9.0m)
- **VLM reasoning**: The mission goal is to TURN_RIGHT. Candidate 1 clearly shows a trajectory that turns to the right, aligning with the driving intent. The other candidates either go straight, turn left, or decelerate significantly without turning.
- **Consistency**: ✓ Consistent with numeric

## Key Findings

1. **Agreement rate**: 4/4 (100%) — VL-DPO and numeric method agree in majority of cases.
2. **Training signal**: VL-DPO generates **2.8x** more preference pairs (11 vs 4).
3. **Semantic understanding**: In this run, VLM fully agreed with numeric reward (0 divergent scenes). VLM reasoning consistently references scene context (obstacle distance, lane availability) to justify selection.
4. **Goal alignment**: VLM explicitly incorporates mission goal (FORWARD, TURN_RIGHT) in selection. In 4/4 scenes, VLM selected the trajectory most consistent with the stated goal.