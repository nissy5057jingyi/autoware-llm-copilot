"""
多场景评测脚本：VL-DPO vs 纯数值方法
使用 DriveTDPA 论文的完整 metric 体系：
ADE↓  FDE↓  Goal↑  Consistency↑  Unsup↓  Lat↓
对齐 VL-DPO 论文的评估思路
"""

import json
import sys
import os
import math
sys.path.insert(0, os.path.expanduser('~/autonomous_driving/DAPO'))
from build_preference_vldpo import build_preference_pairs_vldpo
from rescore_rollout_r2 import (
    compute_goal_reward,
    compute_traj_reward,
    compute_act_traj_reward,
    compute_unsupported_penalty,
)
from pathlib import Path


# ── Metric 计算入口 ──────────────────────────────────────────────────
def compute_all_metrics(sample, candidate):
    """
    对一个候选输出计算全套 metric
    candidate 需要有 trajectory / actions / predict 字段
    """
    # 构造 response_parse 格式（rescore_rollout_r2 需要的格式）
    response_parse = {
        "parse_ok":   candidate["parse"]["parse_ok"],
        "trajectory": candidate["trajectory"],
        "actions":    candidate.get("actions", []),
        "think":      candidate.get("think", ""),
    }

    traj_info    = compute_traj_reward(sample, response_parse)
    goal_score   = compute_goal_reward(sample, response_parse)
    act_traj     = compute_act_traj_reward(sample, response_parse)
    unsup_info   = compute_unsupported_penalty(sample, response_parse)

    consistency  = (act_traj["lat_consistency"] + act_traj["lon_consistency"]) / 2.0

    return {
        "ade":         traj_info["ade"],
        "fde":         traj_info["fde"],
        "goal":        goal_score,
        "consistency": consistency,
        "lat_consistency": act_traj["lat_consistency"],
        "lon_consistency": act_traj["lon_consistency"],
        "unsup":       unsup_info["unsupported_penalty"],
        "lat_mag":     unsup_info["lateral_magnitude"],
    }


# ── 纯数值方法选 chosen ───────────────────────────────────────────────
def select_by_numeric(candidates):
    valid = [c for c in candidates if c["parse"]["parse_ok"]]
    if not valid:
        return None
    return sorted(valid, key=lambda x: x["reward"]["total"], reverse=True)[0]


# ── 10个测试场景 ──────────────────────────────────────────────────────
def make_scenes():
    return [
        # S1: 畅通直行，中速
        {
            "id": "S1_forward_clear",
            "desc": "Clear road, straight ahead",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (8.5) Clear road, no obstacles",
            "gt_trajectory": [[0.0,4.25],[0.0,8.5],[0.0,12.75],[0.0,17.0],[0.0,21.25],[0.0,25.5]],
            "candidates": [
                {"trajectory":[[0.0,4.5],[0.0,9.5],[0.0,15.0],[0.0,21.0],[0.0,27.5],[0.0,34.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate forward</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,4.5),(0.0,9.5),(0.0,15.0),(0.0,21.0),(0.0,27.5),(0.0,34.5)]",
                 "think":"accelerate forward","reward":{"total":0.92},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,4.25],[0.0,8.5],[0.0,12.75],[0.0,17.0],[0.0,21.25],[0.0,25.5]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>maintain speed</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,4.25),(0.0,8.5),(0.0,12.75),(0.0,17.0),(0.0,21.25),(0.0,25.5)]",
                 "think":"maintain speed","reward":{"total":0.88},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[-0.8,3.5],[-2.0,7.0],[-3.2,10.0],[-4.0,13.0],[-4.5,16.0],[-4.8,19.0]],
                 "actions":[["TURN_LEFT","MAINTAIN"],["TURN_LEFT","MAINTAIN"],["TURN_LEFT","MAINTAIN"]],
                 "predict":"<think>turn left</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN']]\n3-second trajectory: [(-0.8,3.5),(-2.0,7.0),(-3.2,10.0),(-4.0,13.0),(-4.5,16.0),(-4.8,19.0)]",
                 "think":"turn left","reward":{"total":0.55},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,2.0],[0.0,3.5],[0.0,4.5],[0.0,5.0],[0.0,5.2],[0.0,5.3]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)]",
                 "think":"brake","reward":{"total":0.40},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S2: 前方障碍物，需减速
        {
            "id": "S2_obstacle_decel",
            "desc": "Static obstacle 15m ahead",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (10.0) Static obstacle 15m ahead",
            "gt_trajectory": [[0.0,3.0],[0.0,5.5],[0.0,7.5],[0.0,9.0],[0.0,10.5],[0.0,11.5]],
            "candidates": [
                {"trajectory":[[0.0,3.0],[0.0,5.5],[0.0,7.5],[0.0,9.0],[0.0,10.5],[0.0,11.5]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>gradual decel</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,5.5),(0.0,7.5),(0.0,9.0),(0.0,10.5),(0.0,11.5)]",
                 "think":"gradual decel","reward":{"total":0.85},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,2.0],[0.0,3.5],[0.0,4.5],[0.0,5.0],[0.0,5.2],[0.0,5.3]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)]",
                 "think":"emergency brake","reward":{"total":0.80},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.0,5.5],[0.0,11.5],[0.0,18.0],[0.0,25.0],[0.0,32.5],[0.0,40.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate dangerous</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,5.5),(0.0,11.5),(0.0,18.0),(0.0,25.0),(0.0,32.5),(0.0,40.5)]",
                 "think":"accelerate dangerous","reward":{"total":0.30},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,5.0],[0.0,10.0],[0.0,15.0],[0.0,20.0],[0.0,25.0],[0.0,30.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>maintain dangerous</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,5.0),(0.0,10.0),(0.0,15.0),(0.0,20.0),(0.0,25.0),(0.0,30.0)]",
                 "think":"maintain dangerous","reward":{"total":0.25},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S3: 左变道超车
        {
            "id": "S3_left_lane_change",
            "desc": "Slow vehicle ahead, left lane clear",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (12.0) Slow vehicle 8m ahead, left lane clear",
            "gt_trajectory": [[-0.5,4.5],[-1.2,9.5],[-1.8,15.0],[-1.8,20.5],[-1.5,26.0],[-1.0,31.5]],
            "candidates": [
                {"trajectory":[[-0.5,4.5],[-1.2,9.5],[-1.8,15.0],[-1.8,20.5],[-1.5,26.0],[-1.0,31.5]],
                 "actions":[["TURN_LEFT","MAINTAIN"],["TURN_LEFT","ACCELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>left lane change</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','ACCELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,4.5),(-1.2,9.5),(-1.8,15.0),(-1.8,20.5),(-1.5,26.0),(-1.0,31.5)]",
                 "think":"left lane change","reward":{"total":0.90},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,3.5],[0.0,6.5],[0.0,9.0],[0.0,11.0],[0.0,13.0],[0.0,15.0]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>follow slow</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.5),(0.0,6.5),(0.0,9.0),(0.0,11.0),(0.0,13.0),(0.0,15.0)]",
                 "think":"follow slow","reward":{"total":0.65},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.8,4.5],[1.5,9.0],[1.8,14.0],[1.8,19.0],[1.5,24.0],[1.0,29.0]],
                 "actions":[["TURN_RIGHT","MAINTAIN"],["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>wrong right change</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.8,4.5),(1.5,9.0),(1.8,14.0),(1.8,19.0),(1.5,24.0),(1.0,29.0)]",
                 "think":"wrong right change","reward":{"total":0.50},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,2.0],[0.0,3.0],[0.0,3.8],[0.0,4.2],[0.0,4.4],[0.0,4.5]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.0),(0.0,3.8),(0.0,4.2),(0.0,4.4),(0.0,4.5)]",
                 "think":"emergency brake","reward":{"total":0.35},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S4: 路口右转
        {
            "id": "S4_turn_right",
            "desc": "Intersection, turn right",
            "mission_goal": "RIGHT",
            "question_raw": "Heading Speed: (6.0) Intersection, turn right, clear",
            "gt_trajectory": [[0.5,3.0],[1.5,5.5],[2.5,7.5],[3.5,8.5],[4.5,9.0],[5.5,9.0]],
            "candidates": [
                {"trajectory":[[0.5,3.0],[1.5,5.5],[2.5,7.5],[3.5,8.5],[4.5,9.0],[5.5,9.0]],
                 "actions":[["TURN_RIGHT","DECELERATE"],["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>turn right</think>\nCorrect action: [['TURN_RIGHT','DECELERATE'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.5,3.0),(1.5,5.5),(2.5,7.5),(3.5,8.5),(4.5,9.0),(5.5,9.0)]",
                 "think":"turn right","reward":{"total":0.91},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,3.0],[0.0,6.0],[0.0,9.0],[0.0,12.0],[0.0,15.0],[0.0,18.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>go straight wrong</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,6.0),(0.0,9.0),(0.0,12.0),(0.0,15.0),(0.0,18.0)]",
                 "think":"go straight wrong","reward":{"total":0.40},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[-0.5,3.0],[-1.5,5.5],[-2.5,7.5],[-3.5,8.5],[-4.5,9.0],[-5.5,9.0]],
                 "actions":[["TURN_LEFT","MAINTAIN"],["TURN_LEFT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>turn left wrong</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,3.0),(-1.5,5.5),(-2.5,7.5),(-3.5,8.5),(-4.5,9.0),(-5.5,9.0)]",
                 "think":"turn left wrong","reward":{"total":0.30},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,1.5],[0.0,2.5],[0.0,3.0],[0.0,3.2],[0.0,3.3],[0.0,3.3]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>brake stop</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,1.5),(0.0,2.5),(0.0,3.0),(0.0,3.2),(0.0,3.3),(0.0,3.3)]",
                 "think":"brake stop","reward":{"total":0.20},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S5: 雨天直行，低速
        {
            "id": "S5_rain_forward",
            "desc": "Rainy road, cautious forward",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (5.0) Rainy conditions, reduced visibility",
            "gt_trajectory": [[0.0,2.5],[0.0,5.0],[0.0,7.5],[0.0,10.0],[0.0,12.5],[0.0,15.0]],
            "candidates": [
                {"trajectory":[[0.0,2.5],[0.0,5.0],[0.0,7.5],[0.0,10.0],[0.0,12.5],[0.0,15.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>maintain rainy</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,2.5),(0.0,5.0),(0.0,7.5),(0.0,10.0),(0.0,12.5),(0.0,15.0)]",
                 "think":"maintain rainy","reward":{"total":0.88},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,3.5],[0.0,7.5],[0.0,12.0],[0.0,17.0],[0.0,22.5],[0.0,28.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate rainy unsafe</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,3.5),(0.0,7.5),(0.0,12.0),(0.0,17.0),(0.0,22.5),(0.0,28.5)]",
                 "think":"accelerate rainy unsafe","reward":{"total":0.60},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.3,2.0],[0.5,4.0],[0.4,6.5],[0.2,9.0],[0.0,12.0],[0.0,15.0]],
                 "actions":[["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>slight drift right</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.3,2.0),(0.5,4.0),(0.4,6.5),(0.2,9.0),(0.0,12.0),(0.0,15.0)]",
                 "think":"slight drift right","reward":{"total":0.50},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,1.5],[0.0,2.5],[0.0,3.2],[0.0,3.8],[0.0,4.2],[0.0,4.5]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>over brake rainy</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,1.5),(0.0,2.5),(0.0,3.2),(0.0,3.8),(0.0,4.2),(0.0,4.5)]",
                 "think":"over brake rainy","reward":{"total":0.35},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S6: 夜间直行
        {
            "id": "S6_night_forward",
            "desc": "Night driving, maintain speed",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (7.0) Night conditions, limited visibility",
            "gt_trajectory": [[0.0,3.5],[0.0,7.0],[0.0,10.5],[0.0,14.0],[0.0,17.5],[0.0,21.0]],
            "candidates": [
                {"trajectory":[[0.0,3.5],[0.0,7.0],[0.0,10.5],[0.0,14.0],[0.0,17.5],[0.0,21.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>maintain night</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.5),(0.0,7.0),(0.0,10.5),(0.0,14.0),(0.0,17.5),(0.0,21.0)]",
                 "think":"maintain night","reward":{"total":0.86},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,2.5],[0.0,4.8],[0.0,6.8],[0.0,8.5],[0.0,10.0],[0.0,11.5]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>cautious night decel</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,2.5),(0.0,4.8),(0.0,6.8),(0.0,8.5),(0.0,10.0),(0.0,11.5)]",
                 "think":"cautious night decel","reward":{"total":0.75},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[-0.3,3.0],[-0.5,6.0],[-0.4,9.5],[-0.2,13.0],[0.0,17.0],[0.0,21.0]],
                 "actions":[["TURN_LEFT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>drift left night</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.3,3.0),(-0.5,6.0),(-0.4,9.5),(-0.2,13.0),(0.0,17.0),(0.0,21.0)]",
                 "think":"drift left night","reward":{"total":0.55},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,4.5],[0.0,9.5],[0.0,15.5],[0.0,22.0],[0.0,29.5],[0.0,38.0]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate night unsafe</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,4.5),(0.0,9.5),(0.0,15.5),(0.0,22.0),(0.0,29.5),(0.0,38.0)]",
                 "think":"accelerate night unsafe","reward":{"total":0.30},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S7: 行人横穿，紧急制动
        {
            "id": "S7_pedestrian_brake",
            "desc": "Pedestrian crossing, emergency brake",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (9.0) Pedestrian crossing 10m ahead",
            "gt_trajectory": [[0.0,2.0],[0.0,3.5],[0.0,4.5],[0.0,5.2],[0.0,5.8],[0.0,6.0]],
            "candidates": [
                {"trajectory":[[0.0,2.0],[0.0,3.5],[0.0,4.5],[0.0,5.2],[0.0,5.8],[0.0,6.0]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>emergency brake pedestrian</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.2),(0.0,5.8),(0.0,6.0)]",
                 "think":"emergency brake pedestrian","reward":{"total":0.90},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,3.0],[0.0,5.5],[0.0,7.5],[0.0,9.0],[0.0,10.5],[0.0,11.5]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>gradual decel pedestrian</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,5.5),(0.0,7.5),(0.0,9.0),(0.0,10.5),(0.0,11.5)]",
                 "think":"gradual decel pedestrian","reward":{"total":0.75},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.0,4.5],[0.0,9.5],[0.0,15.0],[0.0,21.0],[0.0,27.5],[0.0,34.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate ignore pedestrian</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,4.5),(0.0,9.5),(0.0,15.0),(0.0,21.0),(0.0,27.5),(0.0,34.5)]",
                 "think":"accelerate ignore pedestrian","reward":{"total":0.20},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.5,3.0],[1.0,6.0],[1.2,9.5],[1.0,13.0],[0.5,17.0],[0.0,21.0]],
                 "actions":[["TURN_RIGHT","MAINTAIN"],["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>evade right pedestrian</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.5,3.0),(1.0,6.0),(1.2,9.5),(1.0,13.0),(0.5,17.0),(0.0,21.0)]",
                 "think":"evade right pedestrian","reward":{"total":0.35},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S8: 高速直行
        {
            "id": "S8_high_speed",
            "desc": "Highway, high speed forward",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (20.0) Highway, clear road",
            "gt_trajectory": [[0.0,10.0],[0.0,20.0],[0.0,30.0],[0.0,40.0],[0.0,50.0],[0.0,60.0]],
            "candidates": [
                {"trajectory":[[0.0,10.0],[0.0,20.0],[0.0,30.0],[0.0,40.0],[0.0,50.0],[0.0,60.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>highway maintain</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,10.0),(0.0,20.0),(0.0,30.0),(0.0,40.0),(0.0,50.0),(0.0,60.0)]",
                 "think":"highway maintain","reward":{"total":0.93},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,11.0],[0.0,22.5],[0.0,34.5],[0.0,47.0],[0.0,60.0],[0.0,73.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>highway accelerate</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,11.0),(0.0,22.5),(0.0,34.5),(0.0,47.0),(0.0,60.0),(0.0,73.5)]",
                 "think":"highway accelerate","reward":{"total":0.82},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.5,9.0],[1.0,18.5],[1.2,28.5],[1.0,38.5],[0.5,49.0],[0.0,59.5]],
                 "actions":[["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>slight drift highway</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.5,9.0),(1.0,18.5),(1.2,28.5),(1.0,38.5),(0.5,49.0),(0.0,59.5)]",
                 "think":"slight drift highway","reward":{"total":0.55},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,5.0],[0.0,9.0],[0.0,12.0],[0.0,14.5],[0.0,16.5],[0.0,18.0]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>over brake highway</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,5.0),(0.0,9.0),(0.0,12.0),(0.0,14.5),(0.0,16.5),(0.0,18.0)]",
                 "think":"over brake highway","reward":{"total":0.30},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S9: 左转路口
        {
            "id": "S9_turn_left",
            "desc": "Intersection, turn left",
            "mission_goal": "LEFT",
            "question_raw": "Heading Speed: (6.0) Intersection, turn left, clear",
            "gt_trajectory": [[-0.5,3.0],[-1.5,5.5],[-2.5,7.5],[-3.5,8.5],[-4.5,9.0],[-5.5,9.0]],
            "candidates": [
                {"trajectory":[[-0.5,3.0],[-1.5,5.5],[-2.5,7.5],[-3.5,8.5],[-4.5,9.0],[-5.5,9.0]],
                 "actions":[["TURN_LEFT","DECELERATE"],["TURN_LEFT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>turn left</think>\nCorrect action: [['TURN_LEFT','DECELERATE'],['TURN_LEFT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,3.0),(-1.5,5.5),(-2.5,7.5),(-3.5,8.5),(-4.5,9.0),(-5.5,9.0)]",
                 "think":"turn left","reward":{"total":0.91},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[0.0,3.0],[0.0,6.0],[0.0,9.0],[0.0,12.0],[0.0,15.0],[0.0,18.0]],
                 "actions":[["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>go straight wrong</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,6.0),(0.0,9.0),(0.0,12.0),(0.0,15.0),(0.0,18.0)]",
                 "think":"go straight wrong","reward":{"total":0.40},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.5,3.0],[1.5,5.5],[2.5,7.5],[3.5,8.5],[4.5,9.0],[5.5,9.0]],
                 "actions":[["TURN_RIGHT","MAINTAIN"],["TURN_RIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>turn right wrong</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.5,3.0),(1.5,5.5),(2.5,7.5),(3.5,8.5),(4.5,9.0),(5.5,9.0)]",
                 "think":"turn right wrong","reward":{"total":0.25},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,1.5],[0.0,2.5],[0.0,3.0],[0.0,3.2],[0.0,3.3],[0.0,3.3]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>brake stop</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,1.5),(0.0,2.5),(0.0,3.0),(0.0,3.2),(0.0,3.3),(0.0,3.3)]",
                 "think":"brake stop","reward":{"total":0.20},"parse":{"parse_ok":True},"rank":4},
            ]
        },
        # S10: 切入车辆，需让行
        {
            "id": "S10_cut_in",
            "desc": "Vehicle cutting in from right, yield",
            "mission_goal": "FORWARD",
            "question_raw": "Heading Speed: (11.0) Vehicle cutting in from right lane 5m ahead",
            "gt_trajectory": [[0.0,3.5],[0.0,6.5],[0.0,9.0],[0.0,11.0],[0.0,13.0],[0.0,15.0]],
            "candidates": [
                {"trajectory":[[0.0,3.5],[0.0,6.5],[0.0,9.0],[0.0,11.0],[0.0,13.0],[0.0,15.0]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>decel yield cut-in</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.5),(0.0,6.5),(0.0,9.0),(0.0,11.0),(0.0,13.0),(0.0,15.0)]",
                 "think":"decel yield cut-in","reward":{"total":0.89},"parse":{"parse_ok":True},"rank":1},
                {"trajectory":[[-0.5,4.0],[-0.8,8.5],[-0.8,13.5],[-0.5,18.5],[0.0,23.5],[0.0,28.5]],
                 "actions":[["TURN_LEFT","MAINTAIN"],["STRAIGHT","MAINTAIN"],["STRAIGHT","MAINTAIN"]],
                 "predict":"<think>evade left cut-in</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,4.0),(-0.8,8.5),(-0.8,13.5),(-0.5,18.5),(0.0,23.5),(0.0,28.5)]",
                 "think":"evade left cut-in","reward":{"total":0.72},"parse":{"parse_ok":True},"rank":2},
                {"trajectory":[[0.0,5.5],[0.0,11.5],[0.0,18.0],[0.0,25.0],[0.0,32.5],[0.0,40.5]],
                 "actions":[["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"],["STRAIGHT","ACCELERATE"]],
                 "predict":"<think>accelerate dangerous cut-in</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,5.5),(0.0,11.5),(0.0,18.0),(0.0,25.0),(0.0,32.5),(0.0,40.5)]",
                 "think":"accelerate dangerous cut-in","reward":{"total":0.25},"parse":{"parse_ok":True},"rank":3},
                {"trajectory":[[0.0,2.0],[0.0,3.5],[0.0,4.5],[0.0,5.0],[0.0,5.2],[0.0,5.3]],
                 "actions":[["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"],["STRAIGHT","DECELERATE"]],
                 "predict":"<think>over brake cut-in</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)]",
                 "think":"over brake cut-in","reward":{"total":0.40},"parse":{"parse_ok":True},"rank":4},
            ]
        },
    ]


# ── 主评测流程 ─────────────────────────────────────────────────────────
def main():
    scenes = make_scenes()
    all_results = []

    for scene in scenes:
        print(f"\n{'='*55}")
        print(f"Scene: {scene['id']} | {scene['desc']}")
        print(f"{'='*55}")

        valid = [c for c in scene["candidates"] if c["parse"]["parse_ok"]]

        # 方法1: 纯数值
        num_chosen = select_by_numeric(valid)

        # 方法2: VL-DPO
        vlm_chosen, vlm_rejected, vlm_reason = build_preference_pairs_vldpo(
            valid,
            mission_goal=scene["mission_goal"],
            question=scene["question_raw"],
        )

        # 计算两种方法选出的 chosen 的 metric
        num_metrics = compute_all_metrics(scene, num_chosen)
        vlm_metrics = compute_all_metrics(scene, vlm_chosen)

        agree = (num_chosen["rank"] == vlm_chosen["rank"])

        result = {
            "scene_id":    scene["id"],
            "scene_desc":  scene["desc"],
            "mission_goal": scene["mission_goal"],
            "condition":   "day" if "night" not in scene["id"] and "rain" not in scene["id"] else
                           ("night" if "night" in scene["id"] else "rain"),

            # 数值方法
            "numeric_rank":        num_chosen["rank"],
            "numeric_reward":      num_chosen["reward"]["total"],
            "numeric_ade":         num_metrics["ade"],
            "numeric_fde":         num_metrics["fde"],
            "numeric_goal":        num_metrics["goal"],
            "numeric_consistency": num_metrics["consistency"],
            "numeric_unsup":       num_metrics["unsup"],
            "numeric_lat":         num_metrics["lat_mag"],

            # VL-DPO方法
            "vldpo_rank":          vlm_chosen["rank"],
            "vldpo_reward":        vlm_chosen["reward"]["total"],
            "vldpo_ade":           vlm_metrics["ade"],
            "vldpo_fde":           vlm_metrics["fde"],
            "vldpo_goal":          vlm_metrics["goal"],
            "vldpo_consistency":   vlm_metrics["consistency"],
            "vldpo_unsup":         vlm_metrics["unsup"],
            "vldpo_lat":           vlm_metrics["lat_mag"],

            "methods_agree":  agree,
            "vlm_reason":     vlm_reason,
            "num_pairs_vldpo": len(vlm_rejected),
        }
        all_results.append(result)

        print(f"  Numeric  → rank={num_chosen['rank']} | "
              f"ADE={num_metrics['ade']:.3f} FDE={num_metrics['fde']:.3f} "
              f"Goal={num_metrics['goal']:.3f} Cons={num_metrics['consistency']:.3f} "
              f"Unsup={num_metrics['unsup']:.3f} Lat={num_metrics['lat_mag']:.3f}")
        print(f"  VL-DPO   → rank={vlm_chosen['rank']} | "
              f"ADE={vlm_metrics['ade']:.3f} FDE={vlm_metrics['fde']:.3f} "
              f"Goal={vlm_metrics['goal']:.3f} Cons={vlm_metrics['consistency']:.3f} "
              f"Unsup={vlm_metrics['unsup']:.3f} Lat={vlm_metrics['lat_mag']:.3f}")
        print(f"  Agree: {'✓' if agree else '✗'} | VLM reason: {vlm_reason[:80]}")

    # ── 汇总报告 ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("SUMMARY: DriveTDPA Metrics — Numeric vs VL-DPO")
    print(f"{'='*55}")

    def avg(key): return sum(r[key] for r in all_results) / len(all_results)

    print(f"\n{'Metric':<18} {'Numeric':>10} {'VL-DPO':>10} {'Delta':>10}")
    print("-" * 50)
    metrics_pairs = [
        ("ADE↓",         "numeric_ade",         "vldpo_ade",         -1),
        ("FDE↓",         "numeric_fde",         "vldpo_fde",         -1),
        ("Goal↑",        "numeric_goal",        "vldpo_goal",        +1),
        ("Consistency↑", "numeric_consistency", "vldpo_consistency", +1),
        ("Unsup↓",       "numeric_unsup",       "vldpo_unsup",       -1),
        ("Lat↓",         "numeric_lat",         "vldpo_lat",         -1),
    ]
    for name, nk, vk, direction in metrics_pairs:
        n_val = avg(nk)
        v_val = avg(vk)
        delta = (v_val - n_val) * direction  # positive = VL-DPO better
        icon = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{name:<18} {n_val:>10.4f} {v_val:>10.4f} {delta:>+9.4f}{icon}")

    agree_rate = sum(1 for r in all_results if r["methods_agree"]) / len(all_results)
    total_pairs_num = len(all_results)
    total_pairs_vlm = sum(r["num_pairs_vldpo"] for r in all_results)
    print(f"\n{'Agreement rate':<18} {agree_rate*100:.0f}%")
    print(f"{'Numeric pairs':<18} {total_pairs_num}")
    print(f"{'VL-DPO pairs':<18} {total_pairs_vlm} ({total_pairs_vlm/total_pairs_num:.1f}x)")

    # 保存结果
    base = os.path.expanduser("~/autonomous_driving/DAPO")
    json_path = Path(f"{base}/eval_results.json")
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    # Markdown 报告
    md = ["# DriveTDPA: VL-DPO vs Numeric — Full Metric Evaluation\n"]
    md.append("## Per-Scene Results\n")
    md.append("| Scene | Goal | Num ADE↓ | VL ADE↓ | Num Goal↑ | VL Goal↑ | Num Cons↑ | VL Cons↑ | Agree |")
    md.append("|-------|------|---------|---------|----------|----------|----------|----------|-------|")
    for r in all_results:
        icon = "✓" if r["methods_agree"] else "✗"
        md.append(f"| {r['scene_desc'][:25]} | {r['mission_goal']} "
                  f"| {r['numeric_ade']:.3f} | {r['vldpo_ade']:.3f} "
                  f"| {r['numeric_goal']:.3f} | {r['vldpo_goal']:.3f} "
                  f"| {r['numeric_consistency']:.3f} | {r['vldpo_consistency']:.3f} "
                  f"| {icon} |")

    md.append("\n## Average Metrics\n")
    md.append("| Metric | Numeric | VL-DPO | Delta | Better |")
    md.append("|--------|---------|--------|-------|--------|")
    for name, nk, vk, direction in metrics_pairs:
        n_val = avg(nk)
        v_val = avg(vk)
        delta = (v_val - n_val) * direction
        better = "VL-DPO" if delta > 0.001 else ("Numeric" if delta < -0.001 else "Tie")
        md.append(f"| {name} | {n_val:.4f} | {v_val:.4f} | {delta:+.4f} | {better} |")

    md.append(f"\n## Key Statistics\n")
    md.append(f"- Agreement rate: {agree_rate*100:.0f}% ({sum(1 for r in all_results if r['methods_agree'])}/{len(all_results)} scenes)")
    md.append(f"- Training signal: VL-DPO generates {total_pairs_vlm/total_pairs_num:.1f}x more preference pairs")
    md.append(f"- Total scenes: {len(all_results)} (Day/Rain/Night conditions)")

    md_path = Path(f"{base}/eval_report.md")
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"\n✓ JSON: {json_path}")
    print(f"✓ Markdown: {md_path}")


if __name__ == "__main__":
    main()
