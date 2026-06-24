"""
VL-DPO 偏好对构建脚本
替换原来纯数值的 build_preference_from_rollout.py
用 VLM (OpenRouter API) 做语义评判，选出最优轨迹作为 chosen
"""

import json
import base64
import io
import os
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
VLM_MODEL = "google/gemini-2.5-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def render_bev(trajectory, candidate_idx=0, title=None):
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.axvline(x=0, color="#444466", linewidth=1, linestyle="--")
    ego = mpatches.FancyArrow(0, 0, 0, 0.5,
                               width=0.3, head_width=0.5, head_length=0.3,
                               color="#00ff88", zorder=5)
    ax.add_patch(ego)
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    colors = plt.cm.plasma(np.linspace(0.3, 1.0, len(xs)))
    ax.plot(xs, ys, color="#ffffff", linewidth=1.5, alpha=0.6, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.scatter(x, y, color=colors[i], s=40, zorder=4)
        ax.text(x + 0.15, y, f"t{i+1}", color="#cccccc", fontsize=6)
    all_y = ys + [0]
    y_min = min(all_y) - 2
    y_max = max(all_y) + 2
    ax.set_xlim(-4, 4)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x (right, m)", color="#aaaaaa", fontsize=7)
    ax.set_ylabel("y (forward, m)", color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#aaaaaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    label = title if title else f"Candidate {candidate_idx + 1}"
    ax.set_title(label, color="#ffffff", fontsize=8, pad=4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def call_vlm_selector(bev_images_b64, candidate_texts, mission_goal, question):
    content = []
    content.append({
        "type": "text",
        "text": (
            f"You are an autonomous driving expert judge.\n"
            f"Mission goal: {mission_goal}\n"
            f"Scene description: {question}\n\n"
            f"Below are BEV (bird's eye view) images of {len(bev_images_b64)} candidate trajectories "
            f"showing waypoints for the next 3 seconds (t1~t6, 0.5s interval).\n"
            f"y-axis=forward, x-axis=right, green arrow=ego vehicle current position.\n\n"
            f"Select the safest trajectory that best matches the driving intent.\n"
            f"Return ONLY JSON: {{\"selected_index\": <number starting from 1>, \"reason\": \"<brief reason>\"}}"
        )
    })
    for i, b64 in enumerate(bev_images_b64):
        content.append({"type": "text", "text": f"Candidate {i+1}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    actions_text = "\n".join([
        f"Candidate {i+1} output: {c.get('predict','')[:200]}"
        for i, c in enumerate(candidate_texts)
    ])
    content.append({"type": "text", "text": f"\nCandidate model outputs (for reference):\n{actions_text}"})

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    result_text = resp.json()["choices"][0]["message"]["content"].strip()
    print(f"  [VLM reply]: {result_text}")
    try:
        clean = result_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        idx = int(parsed["selected_index"]) - 1
        idx = max(0, min(idx, len(bev_images_b64) - 1))
        return {"selected_index": idx, "reason": parsed.get("reason", "")}
    except Exception as e:
        print(f"  [Warning] VLM parse failed: {e}, defaulting to 0")
        return {"selected_index": 0, "reason": "parse_failed"}


def build_preference_pairs_vldpo(candidates, mission_goal, question):
    print(f"  Rendering {len(candidates)} BEV images...")
    bev_images = []
    for i, c in enumerate(candidates):
        traj = c.get("trajectory", [])
        if not traj:
            traj = [(0.0, (i+1)*2.0 + j*4.0) for j in range(6)]
        b64 = render_bev(traj, candidate_idx=i)
        bev_images.append(b64)
    print(f"  Calling VLM judge...")
    vlm_result = call_vlm_selector(bev_images, candidates, mission_goal, question)
    chosen_idx = vlm_result["selected_index"]
    reason = vlm_result["reason"]
    chosen = candidates[chosen_idx]
    rejected_list = [c for i, c in enumerate(candidates) if i != chosen_idx]
    print(f"  VLM selected: candidate {chosen_idx+1} | reason: {reason}")
    return chosen, rejected_list, reason


def make_test_sample():
    candidates_raw = [
        {"predict": "<think>straight accelerate</think>\nCorrect action: [['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE'],['STRAIGHT','ACCELERATE']]\n3-second trajectory: [(0.0,4.5),(0.0,9.5),(0.0,15.0),(0.0,21.0),(0.0,27.5),(0.0,34.5)]",
         "trajectory": [(0.0,4.5),(0.0,9.5),(0.0,15.0),(0.0,21.0),(0.0,27.5),(0.0,34.5)],
         "reward": {"total": 0.92}, "parse": {"parse_ok": True}, "rank": 1},
        {"predict": "<think>straight maintain</think>\nCorrect action: [['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN'],['STRAIGHT','MAINTAIN']]\n3-second trajectory: [(0.0,4.25),(0.0,8.5),(0.0,12.75),(0.0,17.0),(0.0,21.25),(0.0,25.5)]",
         "trajectory": [(0.0,4.25),(0.0,8.5),(0.0,12.75),(0.0,17.0),(0.0,21.25),(0.0,25.5)],
         "reward": {"total": 0.88}, "parse": {"parse_ok": True}, "rank": 2},
        {"predict": "<think>turn left</think>\nCorrect action: [['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN'],['TURN_LEFT','MAINTAIN']]\n3-second trajectory: [(-0.8,3.5),(-2.0,7.0),(-3.2,10.0),(-4.0,13.0),(-4.5,16.0),(-4.8,19.0)]",
         "trajectory": [(-0.8,3.5),(-2.0,7.0),(-3.2,10.0),(-4.0,13.0),(-4.5,16.0),(-4.8,19.0)],
         "reward": {"total": 0.55}, "parse": {"parse_ok": True}, "rank": 3},
        {"predict": "<think>emergency brake</think>\nCorrect action: [['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE'],['STRAIGHT','DECELERATE']]\n3-second trajectory: [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)]",
         "trajectory": [(0.0,2.0),(0.0,3.5),(0.0,4.5),(0.0,5.0),(0.0,5.2),(0.0,5.3)],
         "reward": {"total": 0.60}, "parse": {"parse_ok": True}, "rank": 4},
        {"predict": "unknown",
         "trajectory": [(0.0,1.0)]*6,
         "reward": {"total": 0.30}, "parse": {"parse_ok": False}, "rank": 5},
    ]
    return {
        "id": "test_scene_001",
        "image": None,
        "question_raw": "Heading Speed: (8.5) Clear road ahead, no obstacles",
        "mission_goal": "FORWARD",
        "gt_trajectory": [(0.0,4.25),(0.0,8.5),(0.0,12.75),(0.0,17.0),(0.0,21.25),(0.0,25.5)],
        "candidates": candidates_raw,
    }


def main():
    print("=" * 60)
    print("VL-DPO Preference Pair Construction Test")
    print("=" * 60)
    sample = make_test_sample()
    print(f"\n[Sample] id={sample['id']}, candidates={len(sample['candidates'])}")
    valid_candidates = [c for c in sample["candidates"] if c["parse"]["parse_ok"]]
    print(f"[Filter] parse_ok candidates={len(valid_candidates)}")
    chosen, rejected_list, reason = build_preference_pairs_vldpo(
        valid_candidates,
        mission_goal=sample["mission_goal"],
        question=sample["question_raw"],
    )
    output_pairs = []
    for rej in rejected_list:
        pair = {
            "id": sample["id"],
            "question": sample["question_raw"],
            "chosen": chosen["predict"],
            "rejected": rej["predict"],
            "chosen_reward": chosen["reward"]["total"],
            "rejected_reward": rej["reward"]["total"],
            "vlm_reason": reason,
            "vlm_model": VLM_MODEL,
            "gt_waypoints": sample["gt_trajectory"],
        }
        output_pairs.append(pair)
    out_path = Path(f"{os.path.expanduser('~')}/autonomous_driving/DAPO/vldpo_test_output.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for p in output_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("\n" + "=" * 60)
    print(f"[OK] Preference pairs generated: {len(output_pairs)}")
    print(f"[OK] chosen: rank={chosen['rank']}, reward={chosen['reward']['total']}")
    print(f"[OK] Output saved to: {out_path}")
    print("=" * 60)
    print("\n[First pair preview]")
    print(json.dumps(output_pairs[0], ensure_ascii=False, indent=2)[:600])

    # 保存 BEV 示例图
    traj = [(0.0,4.5),(0.0,9.5),(0.0,15.0),(0.0,21.0),(0.0,27.5),(0.0,34.5)]
    b64 = render_bev(traj, candidate_idx=0, title="Straight Accelerate")
    img_path = Path(f"{os.path.expanduser('~')}/autonomous_driving/DAPO/bev_sample.png")
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"[OK] BEV sample image saved to: {img_path}")


if __name__ == "__main__":
    main()
