import json
import sys
import os
sys.path.insert(0, os.path.expanduser('~/autonomous_driving/DAPO'))
from build_preference_vldpo import build_preference_pairs_vldpo
from pathlib import Path

SCENES = [
    {
        "id": "scene_obstacle_001",
        "desc": "Obstacle ahead, decelerate",
        "mission_goal": "FORWARD",
        "question_raw": "Heading Speed: (10.0) Static obstacle detected 15m ahead, need to decelerate",
        "candidates": [
            {"predict": "<think>gradual deceleration</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,5.5),(0.0,7.5),(0.0,9.0),(0.0,10.5),(0.0,11.5)]",
             "trajectory": [(0.0,3.0),(0.0,5.5),(0.0,7.5),(0.0,9.0),(0.0,10.5),(0.0,11.5)],
             "reward": {"total": 0.85}, "parse": {"parse_ok": True}, "rank": 1},
            {"predict": "<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)]",
             "trajectory": [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)],
             "reward": {"total": 0.80}, "parse": {"parse_ok": True}, "rank": 2},
            {"predict": "<think>accelerate (dangerous)</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,5.5),(0.0,11.5),(0.0,18.0),(0.0,25.0),(0.0,32.5),(0.0,40.5)]",
             "trajectory": [(0.0,5.5),(0.0,11.5),(0.0,18.0),(0.0,25.0),(0.0,32.5),(0.0,40.5)],
             "reward": {"total": 0.30}, "parse": {"parse_ok": True}, "rank": 3},
            {"predict": "<think>maintain speed (dangerous)</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,5.0),(0.0,10.0),(0.0,15.0),(0.0,20.0),(0.0,25.0),(0.0,30.0)]",
             "trajectory": [(0.0,5.0),(0.0,10.0),(0.0,15.0),(0.0,20.0),(0.0,25.0),(0.0,30.0)],
             "reward": {"total": 0.25}, "parse": {"parse_ok": True}, "rank": 4},
        ],
    },
    {
        "id": "scene_lanechange_001",
        "desc": "Slow vehicle ahead, left lane change",
        "mission_goal": "FORWARD",
        "question_raw": "Heading Speed: (12.0) Slow vehicle 8m ahead (4m/s), left lane is clear",
        "candidates": [
            {"predict": "<think>left lane change to overtake</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','ACCELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,4.5),(-1.2,9.5),(-1.8,15.0),(-1.8,20.5),(-1.5,26.0),(-1.0,31.5)]",
             "trajectory": [(-0.5,4.5),(-1.2,9.5),(-1.8,15.0),(-1.8,20.5),(-1.5,26.0),(-1.0,31.5)],
             "reward": {"total": 0.90}, "parse": {"parse_ok": True}, "rank": 1},
            {"predict": "<think>follow and decelerate</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.5),(0.0,6.5),(0.0,9.0),(0.0,11.0),(0.0,13.0),(0.0,15.0)]",
             "trajectory": [(0.0,3.5),(0.0,6.5),(0.0,9.0),(0.0,11.0),(0.0,13.0),(0.0,15.0)],
             "reward": {"total": 0.65}, "parse": {"parse_ok": True}, "rank": 2},
            {"predict": "<think>right lane change (wrong)</think>\nCorrect action: [['TURN_RIGHT','MAINTAIN'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.8,4.5),(1.5,9.0),(1.8,14.0),(1.8,19.0),(1.5,24.0),(1.0,29.0)]",
             "trajectory": [(0.8,4.5),(1.5,9.0),(1.8,14.0),(1.8,19.0),(1.5,24.0),(1.0,29.0)],
             "reward": {"total": 0.50}, "parse": {"parse_ok": True}, "rank": 3},
            {"predict": "<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.0),(0.0,3.8),(0.0,4.2),(0.0,4.4),(0.0,4.5)]",
             "trajectory": [(0.0,2.0),(0.0,3.0),(0.0,3.8),(0.0,4.2),(0.0,4.4),(0.0,4.5)],
             "reward": {"total": 0.35}, "parse": {"parse_ok": True}, "rank": 4},
        ],
    },
    {
        "id": "scene_turn_right_001",
        "desc": "Intersection, turn right",
        "mission_goal": "TURN_RIGHT",
        "question_raw": "Heading Speed: (6.0) Intersection ahead, need to turn right, no incoming traffic",
        "candidates": [
            {"predict": "<think>turn right</think>\nCorrect action: [['TURN_RIGHT','DECELERATE'],['TURN_RIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.5,3.0),(1.5,5.5),(2.5,7.5),(3.5,8.5),(4.5,9.0),(5.5,9.0)]",
             "trajectory": [(0.5,3.0),(1.5,5.5),(2.5,7.5),(3.5,8.5),(4.5,9.0),(5.5,9.0)],
             "reward": {"total": 0.91}, "parse": {"parse_ok": True}, "rank": 1},
            {"predict": "<think>go straight (wrong)</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,3.0),(0.0,6.0),(0.0,9.0),(0.0,12.0),(0.0,15.0),(0.0,18.0)]",
             "trajectory": [(0.0,3.0),(0.0,6.0),(0.0,9.0),(0.0,12.0),(0.0,15.0),(0.0,18.0)],
             "reward": {"total": 0.40}, "parse": {"parse_ok": True}, "rank": 2},
            {"predict": "<think>turn left (wrong)</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(-0.5,3.0),(-1.5,5.5),(-2.5,7.5),(-3.5,8.5),(-4.5,9.0),(-5.5,9.0)]",
             "trajectory": [(-0.5,3.0),(-1.5,5.5),(-2.5,7.5),(-3.5,8.5),(-4.5,9.0),(-5.5,9.0)],
             "reward": {"total": 0.30}, "parse": {"parse_ok": True}, "rank": 3},
            {"predict": "<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,1.5),(0.0,2.5),(0.0,3.0),(0.0,3.2),(0.0,3.3),(0.0,3.3)]",
             "trajectory": [(0.0,1.5),(0.0,2.5),(0.0,3.0),(0.0,3.2),(0.0,3.3),(0.0,3.3)],
             "reward": {"total": 0.20}, "parse": {"parse_ok": True}, "rank": 4},
        ],
    },
]


def run_scene(scene):
    print(f"\n{'='*60}")
    print(f"Scene: {scene['id']}")
    print(f"Desc:  {scene['desc']}")
    print(f"Goal:  {scene['mission_goal']}")
    print(f"{'='*60}")
    valid = [c for c in scene["candidates"] if c["parse"]["parse_ok"]]
    chosen, rejected_list, reason = build_preference_pairs_vldpo(
        valid,
        mission_goal=scene["mission_goal"],
        question=scene["question_raw"],
    )
    is_correct = (chosen["rank"] == 1)
    print(f"  Numeric top reward: rank=1, reward={valid[0]['reward']['total']}")
    print(f"  VLM selected:       rank={chosen['rank']}, reward={chosen['reward']['total']}")
    print(f"  Consistent with numeric reward: {'YES' if is_correct else 'NO (VLM diverges)'}")
    print(f"  Preference pairs generated: {len(rejected_list)}")
    return {
        "scene_id": scene["id"],
        "scene_desc": scene["desc"],
        "mission_goal": scene["mission_goal"],
        "vlm_chosen_rank": chosen["rank"],
        "vlm_chosen_reward": chosen["reward"]["total"],
        "vlm_reason": reason,
        "num_pairs": len(rejected_list),
        "consistent_with_reward": is_correct,
    }


def main():
    results = []
    for scene in SCENES:
        r = run_scene(scene)
        results.append(r)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r["consistent_with_reward"] else "✗"
        print(f"{icon} {r['scene_id']:30s} | goal={r['mission_goal']:12s} | vlm_rank={r['vlm_chosen_rank']} | reward={r['vlm_chosen_reward']:.2f}")
        print(f"   Reason: {r['vlm_reason'][:100]}")

    out_path = Path(os.path.expanduser("~/autonomous_driving/DAPO/vldpo_multi_scene_results.json"))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
